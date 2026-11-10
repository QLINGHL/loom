//! 常驻感知的功耗（设计文档第 21 章开放问题 6）：「随时待命」和
//! 「一天一充」直接冲突。可行方向是事件驱动而非轮询——绝大多数时间
//! NPU 休眠，由廉价传感器或规则引擎触发唤醒。但「什么事件值得唤醒」
//! 又是一个准确率问题：唤醒太积极，误唤醒烧掉电量；太保守，真实
//! 意图被漏掉，用户只能手动唤醒。
//!
//! 本模块是这个权衡的原型，全部用确定性规则表达：
//!
//! * **三档功耗状态**：`DeepSleep`（深睡，传感器 fabric 最低功耗）、
//!   `Listening`（监听，廉价传感器 + 规则引擎常开）、`Awake`（NPU 上
//!   电处理意图）。状态切换即功耗切换，能量按「功率 × 停留时间」
//!   精确记账，单位统一为微焦（µJ）：1 mW × 1 ms = 1 µJ。
//! * **唤醒规则**：每条规则绑定一种廉价传感器事件、最低信号强度、
//!   先验概率、生效时段、冷却时间。事件得分 = 先验 × 信号强度，
//!   达到阈值才唤醒 NPU；阈值不是写死的——每次唤醒后由结果反馈
//!   自校准（误唤醒多了就抬高，漏掉意图就降低）。
//! * **能量预算**：每日感知预算（mWh），预算耗尽后只放行标记为
//!   `critical` 的规则；深睡时段同样只放行关键规则。
//! * **诚实性规则**：误唤醒率 / 唤醒精确率在没有唤醒数据时是
//!   `None`，不报 0% 也不报 100%；能量只在事件与状态切换边界
//!   记账，读数前调用 [`SensingDaemon::sync`] 把停留时间结算到
//!   当前时刻——不结算就读数会低估，方法文档里写明这一点。
//!
//! 轮询与事件驱动的成本对比见 [`polling_daily_mwh`] 与
//! [`event_driven_daily_mwh`]：轮询的成本由频率决定，事件驱动的
//! 成本由真实意图数决定——意图越稀疏，事件驱动赢得越多。

use crate::Clock;
use std::collections::{BTreeMap, VecDeque};

/// 深睡功耗：只有传感器 fabric 的最低限度供电（mW）。
pub const POWER_DEEP_SLEEP_MW: f64 = 0.5;
/// 监听功耗：廉价传感器 + 规则引擎常开（mW）。
pub const POWER_LISTENING_MW: f64 = 3.0;
/// 唤醒功耗：NPU 上电处理意图（mW）。
pub const POWER_AWAKE_MW: f64 = 3000.0;

/// NPU 一次上电的固定开销（µJ）：时钟、供电、权重载入启动。
pub const WAKE_SPINUP_UJ: u64 = 50_000;

/// 默认唤醒阈值：事件得分达到它才唤醒 NPU。
pub const DEFAULT_WAKE_THRESHOLD: f64 = 0.5;

/// 阈值自校准的目标误唤醒率：超过它就抬高阈值。
pub const TARGET_FALSE_WAKE_RATE: f64 = 0.10;

/// 误唤醒率超过目标多少个百分点才触发抬高（死区，避免抖动）。
pub const FALSE_WAKE_DEADBAND: f64 = 0.05;

/// 阈值每次调整的步长。
pub const THRESHOLD_STEP: f64 = 0.05;

/// 阈值上下限。
pub const THRESHOLD_MIN: f64 = 0.05;
pub const THRESHOLD_MAX: f64 = 0.95;

/// 阈值校准至少积累这么多次唤醒结果才开始抬高（样本太少不调）。
pub const CALIBRATION_MIN_SAMPLES: usize = 5;

/// 1 mWh = 3.6 J = 3_600_000 µJ。
pub const UJ_PER_MWH: f64 = 3_600_000.0;

/// 一天的毫秒数。
pub const DAY_MS: u64 = 86_400_000;

/// NPU 的功耗状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerState {
    /// 深睡：传感器 fabric 最低功耗，只放行关键规则的唤醒。
    DeepSleep,
    /// 监听：廉价传感器 + 规则引擎常开，默认待机状态。
    Listening,
    /// 唤醒：NPU 上电，正在处理意图。
    Awake,
}

impl PowerState {
    /// 该状态的功耗（mW）。
    pub fn power_mw(self) -> f64 {
        match self {
            PowerState::DeepSleep => POWER_DEEP_SLEEP_MW,
            PowerState::Listening => POWER_LISTENING_MW,
            PowerState::Awake => POWER_AWAKE_MW,
        }
    }
}

/// 廉价传感器事件的种类。注意：这些传感器本身不跑模型，只产生
/// 「可能有意图」的廉价信号；判断是否值得唤醒 NPU 是规则引擎的事。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SensorKind {
    /// 唤醒词检测（如 "Hey Loom"）：信号强，先验最高。
    VoiceKeyword,
    /// 动作/拿起设备。
    Motion,
    /// 屏幕点亮。
    ScreenOn,
    /// 进入/离开地理围栏。
    Geofence,
    /// 定时 tick（如整点、日程前 10 分钟）。
    ScheduledTick,
    /// 某个应用回到前台。
    AppForeground,
}

impl SensorKind {
    /// 产生一次该类事件，廉价传感器本身消耗的能量（µJ）。
    pub fn sensor_cost_uj(self) -> u64 {
        match self {
            SensorKind::VoiceKeyword => 200,
            SensorKind::Motion => 50,
            SensorKind::ScreenOn => 20,
            SensorKind::Geofence => 100,
            SensorKind::ScheduledTick => 10,
            SensorKind::AppForeground => 30,
        }
    }

    /// 该类事件「背后真有用户意图」的先验概率（经验初值，
    /// 规则可以覆盖它）。
    pub fn default_prior(self) -> f64 {
        match self {
            SensorKind::VoiceKeyword => 0.70,
            SensorKind::Motion => 0.15,
            SensorKind::ScreenOn => 0.40,
            SensorKind::Geofence => 0.30,
            SensorKind::ScheduledTick => 0.50,
            SensorKind::AppForeground => 0.35,
        }
    }
}

/// 一次廉价传感器事件。
#[derive(Debug, Clone, Copy)]
pub struct SensorEvent {
    pub kind: SensorKind,
    /// 信号强度 [0, 1]：唤醒词置信度、动作幅度等归一化后的值。
    pub strength: f64,
    /// 事件发生的时刻（ms，调用方时钟）。
    pub at_ms: u64,
    /// 事件发生的小时（0–23），用于时段规则。
    pub hour: u8,
}

impl SensorEvent {
    pub fn new(kind: SensorKind, strength: f64, at_ms: u64, hour: u8) -> Self {
        SensorEvent { kind, strength: strength_clamped(strength), at_ms, hour }
    }
}

fn strength_clamped(s: f64) -> f64 {
    s.clamp(0.0, 1.0)
}

/// 一条唤醒规则：什么事件、在什么条件下值得唤醒 NPU。
#[derive(Debug, Clone)]
pub struct WakeRule {
    pub name: String,
    pub kind: SensorKind,
    /// 信号强度低于它直接不唤醒（再高的先验也不行）。
    pub min_strength: f64,
    /// 该规则的先验概率：这类事件背后真有意图的概率。
    pub prior: f64,
    /// 生效时段（小时，左闭右开），如 (7, 23)；`None` 表示全天。
    /// 支持跨午夜，如 (22, 7) 表示 22:00–次日 07:00。
    pub active_hours: Option<(u8, u8)>,
    /// 同一规则两次唤醒之间的最短间隔（ms）：防抖。
    pub cooldown_ms: u64,
    /// 关键规则：预算耗尽或深睡时段仍可唤醒。
    pub critical: bool,
}

impl WakeRule {
    /// 用该传感器种类的默认先验建一条规则。
    pub fn with_defaults(
        name: impl Into<String>,
        kind: SensorKind,
        min_strength: f64,
        cooldown_ms: u64,
        critical: bool,
    ) -> Self {
        WakeRule {
            name: name.into(),
            kind,
            min_strength,
            prior: kind.default_prior(),
            active_hours: None,
            cooldown_ms,
            critical,
        }
    }

    /// 该规则在给定小时是否生效。
    pub fn active_at(&self, hour: u8) -> bool {
        match self.active_hours {
            None => true,
            Some((start, end)) => {
                if start == end {
                    return true; // 起止相同视为全天
                }
                if start < end {
                    hour >= start && hour < end
                } else {
                    hour >= start || hour < end
                }
            }
        }
    }

    /// 事件得分 = 先验 × 信号强度。
    pub fn score(&self, event: &SensorEvent) -> f64 {
        self.prior * event.strength
    }
}

/// 事件没有唤醒 NPU 的原因。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StayAsleepReason {
    /// 没有规则匹配该类事件。
    NoMatchingRule,
    /// 信号强度低于规则的最低要求。
    BelowMinStrength,
    /// 不在规则的生效时段内。
    OutsideActiveHours,
    /// 得分低于当前唤醒阈值。
    BelowThreshold { score: f64 },
    /// 在冷却期内，防抖压住。
    Cooldown { remaining_ms: u64 },
    /// 每日能量预算已耗尽，且该规则不是关键规则。
    BudgetExhausted,
    /// 深睡时段，且该规则不是关键规则。
    DeepSleep,
}

/// 一次事件的唤醒判决。
#[derive(Debug, Clone, PartialEq)]
pub enum WakeDecision {
    /// 唤醒 NPU：附带命中的规则名与得分。
    Wake { rule: String, score: f64 },
    /// 继续睡：附带原因。
    StayAsleep { reason: StayAsleepReason },
}

/// 常驻感知守护进程：持有唤醒规则、阈值、能量账本与唤醒统计，
/// 是「事件驱动唤醒」机制的完整实现。时间一律走注入的 [`Clock`]，
/// 所以能量记账与阈值校准都可确定性测试。
pub struct SensingDaemon<'a> {
    clock: &'a dyn Clock,
    rules: Vec<WakeRule>,
    threshold: f64,
    state: PowerState,
    /// 当前状态的起始时刻（ms）。
    state_since_ms: u64,
    /// 守护进程启动时刻（ms），用于占空比的分母。
    started_ms: u64,
    /// 累计消耗的能量（µJ），已结算到上一次事件/切换边界。
    energy_uj: u64,
    /// 累计处于 Awake 状态的时间（ms）。
    awake_ms: u64,
    /// 每日能量预算（mWh），`None` 表示不限。
    daily_budget_mwh: Option<f64>,
    /// 唤醒总次数。
    wakes: u32,
    /// 有用的唤醒（唤醒后确实处理了用户意图）。
    useful_wakes: u32,
    /// 误唤醒（唤醒后发现没有意图，白白上电）。
    false_wakes: u32,
    /// 漏掉的意图数（用户不得不手动唤醒——阈值太高的直接证据）。
    missed_intents: u32,
    /// 最近若干次唤醒的结果（true = 有用），用于阈值校准。
    recent_outcomes: VecDeque<bool>,
    /// 每条规则上次唤醒的时刻（ms），用于冷却。
    last_wake_ms: BTreeMap<String, u64>,
    /// 每条规则的唤醒次数（用于统计与报告）。
    wakes_per_rule: BTreeMap<String, u32>,
}

impl<'a> SensingDaemon<'a> {
    pub fn new(
        clock: &'a dyn Clock,
        rules: Vec<WakeRule>,
        threshold: f64,
        daily_budget_mwh: Option<f64>,
    ) -> Self {
        let now = clock.now_ms();
        SensingDaemon {
            clock,
            rules,
            threshold: threshold.clamp(THRESHOLD_MIN, THRESHOLD_MAX),
            state: PowerState::Listening,
            state_since_ms: now,
            started_ms: now,
            energy_uj: 0,
            awake_ms: 0,
            daily_budget_mwh,
            wakes: 0,
            useful_wakes: 0,
            false_wakes: 0,
            missed_intents: 0,
            recent_outcomes: VecDeque::new(),
            last_wake_ms: BTreeMap::new(),
            wakes_per_rule: BTreeMap::new(),
        }
    }

    pub fn state(&self) -> PowerState {
        self.state
    }

    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// 把当前状态的停留时间结算到当前时刻。能量读数（
    /// [`energy_mwh`]、[`duty_cycle`]、[`budget_remaining_mwh`]）
    /// 在调用本方法后才是准确的；不调用则读数停在上一次事件或
    /// 状态切换的边界——这是记账方式，不是 bug。
    pub fn sync(&mut self) {
        self.accrue();
    }

    /// 结算从 `state_since_ms` 到现在的停留能量。
    fn accrue(&mut self) {
        let now = self.clock.now_ms();
        let elapsed = now.saturating_sub(self.state_since_ms);
        if elapsed > 0 {
            let uj = (self.state.power_mw() * elapsed as f64).round() as u64;
            self.energy_uj = self.energy_uj.saturating_add(uj);
            if self.state == PowerState::Awake {
                self.awake_ms = self.awake_ms.saturating_add(elapsed);
            }
            self.state_since_ms = now;
        }
    }

    fn set_state(&mut self, next: PowerState) {
        if next != self.state {
            self.accrue();
            self.state = next;
        }
    }

    /// 进入深睡（例如夜间）：功耗降到最低，只有关键规则能唤醒。
    pub fn enter_deep_sleep(&mut self) {
        self.set_state(PowerState::DeepSleep);
    }

    /// 回到监听状态（默认待机）。
    pub fn enter_listening(&mut self) {
        self.set_state(PowerState::Listening);
    }

    /// NPU 处理完意图后回到监听状态。
    pub fn go_back_to_sleep(&mut self) {
        self.set_state(PowerState::Listening);
    }

    /// 处理一次廉价传感器事件，返回唤醒判决。无论唤醒与否，
    /// 产生事件的廉价传感器本身的能量都会记账。
    pub fn on_event(&mut self, event: &SensorEvent) -> WakeDecision {
        self.accrue();
        self.energy_uj = self.energy_uj.saturating_add(event.kind.sensor_cost_uj());

        let rule = match self.rules.iter().find(|r| r.kind == event.kind) {
            Some(r) => r.clone(),
            None => return WakeDecision::StayAsleep { reason: StayAsleepReason::NoMatchingRule },
        };

        if event.strength < rule.min_strength {
            return WakeDecision::StayAsleep {
                reason: StayAsleepReason::BelowMinStrength,
            };
        }
        if !rule.active_at(event.hour) {
            return WakeDecision::StayAsleep {
                reason: StayAsleepReason::OutsideActiveHours,
            };
        }
        if self.state == PowerState::DeepSleep && !rule.critical {
            return WakeDecision::StayAsleep { reason: StayAsleepReason::DeepSleep };
        }
        if let Some(last) = self.last_wake_ms.get(&rule.name) {
            let elapsed = event.at_ms.saturating_sub(*last);
            if elapsed < rule.cooldown_ms {
                return WakeDecision::StayAsleep {
                    reason: StayAsleepReason::Cooldown {
                        remaining_ms: rule.cooldown_ms - elapsed,
                    },
                };
            }
        }
        let score = rule.score(event);
        if score < self.threshold {
            return WakeDecision::StayAsleep {
                reason: StayAsleepReason::BelowThreshold { score },
            };
        }
        if !rule.critical && self.over_budget() {
            return WakeDecision::StayAsleep {
                reason: StayAsleepReason::BudgetExhausted,
            };
        }

        // 唤醒：记上电开销，切到 Awake 状态。
        self.energy_uj = self.energy_uj.saturating_add(WAKE_SPINUP_UJ);
        self.set_state(PowerState::Awake);
        self.wakes += 1;
        *self.wakes_per_rule.entry(rule.name.clone()).or_insert(0) += 1;
        self.last_wake_ms.insert(rule.name.clone(), event.at_ms);
        WakeDecision::Wake { rule: rule.name, score }
    }

    /// 上报一次唤醒的结果：唤醒后是否真的处理了用户意图。
    /// 误唤醒积累到超过目标就抬高阈值（样本不足时不调，避免
    /// 开局几次偶然误唤醒就把阈值抬死）。
    pub fn report_wake_outcome(&mut self, useful: bool) {
        if useful {
            self.useful_wakes += 1;
        } else {
            self.false_wakes += 1;
        }
        self.recent_outcomes.push_back(useful);
        if self.recent_outcomes.len() > 20 {
            self.recent_outcomes.pop_front();
        }
        if self.recent_outcomes.len() >= CALIBRATION_MIN_SAMPLES {
            let n = self.recent_outcomes.len() as f64;
            let false_rate =
                self.recent_outcomes.iter().filter(|o| !**o).count() as f64 / n;
            if false_rate > TARGET_FALSE_WAKE_RATE + FALSE_WAKE_DEADBAND {
                self.threshold =
                    (self.threshold + THRESHOLD_STEP).min(THRESHOLD_MAX);
            }
        }
    }

    /// 上报一次漏掉的意图（用户不得不手动唤醒）：阈值太高的
    /// 直接证据，立即降低阈值一档。
    pub fn report_missed_intent(&mut self) {
        self.missed_intents += 1;
        self.threshold = (self.threshold - THRESHOLD_STEP).max(THRESHOLD_MIN);
    }

    /// 累计消耗的能量（mWh）。调用前先 [`sync`](Self::sync) 结算
    /// 到当前时刻，否则读数停在上一次事件/切换边界。
    pub fn energy_mwh(&self) -> f64 {
        self.energy_uj as f64 / UJ_PER_MWH
    }

    /// 每日能量预算是否已耗尽（无预算时永不耗尽）。
    pub fn over_budget(&self) -> bool {
        match self.daily_budget_mwh {
            None => false,
            Some(b) => self.energy_mwh() >= b,
        }
    }

    /// 预算剩余（mWh），无预算时为 `None`；超预算时为负数，
    /// 如实报超了多少，不截断到 0。
    pub fn budget_remaining_mwh(&self) -> Option<f64> {
        self.daily_budget_mwh.map(|b| b - self.energy_mwh())
    }

    /// 占空比：处于 Awake 状态的时间占自启动以来的比例。
    /// 启动后尚未流逝时间时为 `None`。
    pub fn duty_cycle(&self) -> Option<f64> {
        let total = self.clock.now_ms().saturating_sub(self.started_ms);
        if total == 0 {
            None
        } else {
            Some(self.awake_ms as f64 / total as f64)
        }
    }

    /// 误唤醒率 = 误唤醒 / 已判定结果的唤醒。尚无已判定结果时
    /// 为 `None`（唤醒了但结果还没上报，不算）。
    pub fn false_wake_rate(&self) -> Option<f64> {
        let judged = self.useful_wakes + self.false_wakes;
        if judged == 0 {
            None
        } else {
            Some(self.false_wakes as f64 / judged as f64)
        }
    }

    /// 唤醒精确率 = 有用唤醒 / 已判定结果的唤醒。尚无已判定
    /// 结果时为 `None`。
    pub fn wake_precision(&self) -> Option<f64> {
        let judged = self.useful_wakes + self.false_wakes;
        if judged == 0 {
            None
        } else {
            Some(self.useful_wakes as f64 / judged as f64)
        }
    }

    pub fn wakes(&self) -> u32 {
        self.wakes
    }

    pub fn useful_wakes(&self) -> u32 {
        self.useful_wakes
    }

    pub fn false_wakes(&self) -> u32 {
        self.false_wakes
    }

    pub fn missed_intents(&self) -> u32 {
        self.missed_intents
    }

    pub fn wakes_per_rule(&self) -> &BTreeMap<String, u32> {
        &self.wakes_per_rule
    }

    /// 一行汇总：状态、阈值、能量、占空比、误唤醒率、漏意图数。
    pub fn render(&self) -> String {
        let mut out = String::from("常驻感知（第 21 章 Q6）——事件驱动唤醒\n");
        out.push_str(&format!(
            "状态：{:?}，唤醒阈值：{:.2}\n",
            self.state, self.threshold
        ));
        out.push_str(&format!(
            "能量：{:.2} mWh（预算剩余：{}）\n",
            self.energy_mwh(),
            match self.budget_remaining_mwh() {
                None => "不限".to_string(),
                Some(r) => format!("{r:.2} mWh"),
            }
        ));
        out.push_str(&format!(
            "占空比：{}\n",
            match self.duty_cycle() {
                None => "—（无数据）".to_string(),
                Some(d) => format!("{:.2}%", d * 100.0),
            }
        ));
        out.push_str(&format!(
            "唤醒：{} 次（有用 {}，误唤醒 {}，误唤醒率 {}）\n",
            self.wakes,
            self.useful_wakes,
            self.false_wakes,
            match self.false_wake_rate() {
                None => "—（无数据）".to_string(),
                Some(r) => format!("{:.1}%", r * 100.0),
            }
        ));
        out.push_str(&format!("漏掉的意图：{} 次\n", self.missed_intents));
        out
    }
}

/// 轮询方案一天的能量（mWh）：每隔 `poll_interval_ms` 把 NPU 上电
/// `per_poll_active_ms` 检查一次，其余时间以 `idle_power_mw` 待机。
/// 轮询的成本由频率决定，与有没有意图无关。
pub fn polling_daily_mwh(
    poll_interval_ms: u64,
    per_poll_active_ms: u64,
    active_power_mw: f64,
    idle_power_mw: f64,
) -> f64 {
    if poll_interval_ms == 0 {
        return f64::INFINITY;
    }
    let polls = DAY_MS / poll_interval_ms;
    let active_ms = polls.saturating_mul(per_poll_active_ms).min(DAY_MS);
    let idle_ms = DAY_MS - active_ms;
    let total_uj =
        active_ms as f64 * active_power_mw + idle_ms as f64 * idle_power_mw;
    total_uj / UJ_PER_MWH
}

/// 事件驱动方案一天的能量（mWh）：平时以 `idle_power_mw` 监听，
/// 每天 `wakes_per_day` 次唤醒，每次消耗 `per_wake_uj`（上电开销 +
/// 处理意图的能量）。成本由真实意图数决定，与轮询频率无关。
pub fn event_driven_daily_mwh(
    idle_power_mw: f64,
    wakes_per_day: u64,
    per_wake_uj: f64,
) -> f64 {
    let idle_uj = DAY_MS as f64 * idle_power_mw;
    let wake_uj = wakes_per_day as f64 * per_wake_uj;
    (idle_uj + wake_uj) / UJ_PER_MWH
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ManualClock;

    fn voice_rule() -> WakeRule {
        WakeRule::with_defaults("唤醒词", SensorKind::VoiceKeyword, 0.3, 5_000, true)
    }

    fn motion_rule() -> WakeRule {
        WakeRule::with_defaults("动作", SensorKind::Motion, 0.5, 10_000, false)
    }

    /// 测试脚手架：调用方持有 clock，daemon 借用它。
    struct Fixture {
        clock: ManualClock,
    }

    impl Fixture {
        fn new() -> Self {
            Fixture { clock: ManualClock::new(0) }
        }

        fn daemon(&self, rules: Vec<WakeRule>, budget: Option<f64>) -> SensingDaemon<'_> {
            SensingDaemon::new(&self.clock, rules, DEFAULT_WAKE_THRESHOLD, budget)
        }
    }

    fn event(kind: SensorKind, strength: f64, hour: u8) -> SensorEvent {
        SensorEvent::new(kind, strength, 0, hour)
    }

    #[test]
    fn listening_draws_far_less_than_awake() {
        let f = Fixture::new();
        let mut d = f.daemon(vec![voice_rule()], None);
        // 监听 1 小时。
        f.clock.advance(3_600_000);
        d.sync();
        let listening_mwh = d.energy_mwh();
        assert!((listening_mwh - 3.0).abs() < 0.01, "got {listening_mwh}");
        // 唤醒词唤醒，处理 10 秒。
        let ev = SensorEvent::new(SensorKind::VoiceKeyword, 0.9, f.clock.now_ms(), 10);
        assert!(matches!(d.on_event(&ev), WakeDecision::Wake { .. }));
        f.clock.advance(10_000);
        d.go_back_to_sleep();
        d.sync();
        // 10 秒 Awake ≈ 3000 mW × 10 s = 8.33 mWh，远大于 1 小时监听的 3 mWh。
        let awake_part = d.energy_mwh() - listening_mwh;
        assert!(awake_part > 8.0 && awake_part < 9.0, "got {awake_part}");
    }

    #[test]
    fn strong_voice_keyword_wakes() {
        let f = Fixture::new();
        let mut d = f.daemon(vec![voice_rule()], None);
        let ev = event(SensorKind::VoiceKeyword, 0.9, 10);
        let dec = d.on_event(&ev);
        // 得分 0.7 × 0.9 = 0.63 ≥ 0.5 阈值。
        assert_eq!(
            dec,
            WakeDecision::Wake { rule: "唤醒词".to_string(), score: 0.63 }
        );
        assert_eq!(d.state(), PowerState::Awake);
        assert_eq!(d.wakes(), 1);
    }

    #[test]
    fn weak_event_below_threshold_stays_asleep() {
        let f = Fixture::new();
        let mut d = f.daemon(vec![motion_rule()], None);
        // 动作先验 0.15 × 强度 0.8 = 0.12 < 0.5。
        let ev = event(SensorKind::Motion, 0.8, 10);
        assert_eq!(
            d.on_event(&ev),
            WakeDecision::StayAsleep { reason: StayAsleepReason::BelowThreshold { score: 0.12 } }
        );
        assert_eq!(d.state(), PowerState::Listening);
        assert_eq!(d.wakes(), 0);
    }

    #[test]
    fn unmatched_kind_has_no_rule() {
        let f = Fixture::new();
        let mut d = f.daemon(vec![voice_rule()], None);
        let ev = event(SensorKind::Motion, 1.0, 10);
        assert_eq!(
            d.on_event(&ev),
            WakeDecision::StayAsleep { reason: StayAsleepReason::NoMatchingRule }
        );
    }

    #[test]
    fn below_min_strength_is_rejected_before_scoring() {
        let f = Fixture::new();
        let mut d = f.daemon(vec![voice_rule()], None);
        let ev = event(SensorKind::VoiceKeyword, 0.2, 10);
        assert_eq!(
            d.on_event(&ev),
            WakeDecision::StayAsleep { reason: StayAsleepReason::BelowMinStrength }
        );
    }

    #[test]
    fn cooldown_suppresses_repeat_wake() {
        let f = Fixture::new();
        let mut d = f.daemon(vec![voice_rule()], None);
        let ev = SensorEvent::new(SensorKind::VoiceKeyword, 0.9, 0, 10);
        assert!(matches!(d.on_event(&ev), WakeDecision::Wake { .. }));
        d.go_back_to_sleep();
        // 2 秒后同一规则再触发：冷却 5 秒未过。
        f.clock.advance(2_000);
        let ev2 = SensorEvent::new(SensorKind::VoiceKeyword, 0.9, f.clock.now_ms(), 10);
        assert_eq!(
            d.on_event(&ev2),
            WakeDecision::StayAsleep {
                reason: StayAsleepReason::Cooldown { remaining_ms: 3_000 }
            }
        );
        // 冷却过后可以再次唤醒。
        f.clock.advance(3_000);
        let ev3 = SensorEvent::new(SensorKind::VoiceKeyword, 0.9, f.clock.now_ms(), 10);
        assert!(matches!(d.on_event(&ev3), WakeDecision::Wake { .. }));
    }

    #[test]
    fn active_hours_window_is_enforced_and_wraps_midnight() {
        let mut rule = voice_rule();
        rule.active_hours = Some((22, 7)); // 22:00–次日 07:00
        assert!(rule.active_at(23));
        assert!(rule.active_at(3));
        assert!(!rule.active_at(12));
        assert!(!rule.active_at(7)); // 结束时刻不含

        let f = Fixture::new();
        let mut d = f.daemon(vec![rule], None);
        let ev = event(SensorKind::VoiceKeyword, 0.9, 12);
        assert_eq!(
            d.on_event(&ev),
            WakeDecision::StayAsleep { reason: StayAsleepReason::OutsideActiveHours }
        );
    }

    #[test]
    fn false_wakes_raise_threshold_after_enough_samples() {
        let f = Fixture::new();
        let mut d = f.daemon(vec![voice_rule()], None);
        assert_eq!(d.threshold(), 0.5);
        for _ in 0..CALIBRATION_MIN_SAMPLES {
            d.report_wake_outcome(false);
        }
        // 5 次全是误唤醒，误唤醒率 100% 远超目标：阈值被抬高。
        assert!(d.threshold() > 0.5, "threshold {}", d.threshold());
        assert_eq!(d.false_wake_rate(), Some(1.0));
        assert_eq!(d.wake_precision(), Some(0.0));
    }

    #[test]
    fn useful_wakes_do_not_raise_threshold() {
        let f = Fixture::new();
        let mut d = f.daemon(vec![voice_rule()], None);
        for _ in 0..CALIBRATION_MIN_SAMPLES {
            d.report_wake_outcome(true);
        }
        assert_eq!(d.threshold(), 0.5);
        assert_eq!(d.wake_precision(), Some(1.0));
        assert_eq!(d.false_wake_rate(), Some(0.0));
    }

    #[test]
    fn missed_intent_lowers_threshold_immediately() {
        let f = Fixture::new();
        let mut d = f.daemon(vec![voice_rule()], None);
        d.report_missed_intent();
        assert!((d.threshold() - 0.45).abs() < 1e-9);
        assert_eq!(d.missed_intents(), 1);
        // 降到下限后不再降。
        for _ in 0..20 {
            d.report_missed_intent();
        }
        assert_eq!(d.threshold(), THRESHOLD_MIN);
    }

    #[test]
    fn rates_are_none_before_any_wake() {
        let f = Fixture::new();
        let d = f.daemon(vec![voice_rule()], None);
        assert_eq!(d.false_wake_rate(), None);
        assert_eq!(d.wake_precision(), None);
        assert_eq!(d.duty_cycle(), None);
    }

    #[test]
    fn budget_exhausted_blocks_noncritical_but_not_critical() {
        let f = Fixture::new();
        // 预算 0.01 mWh：一次监听几分钟就耗尽。动作规则先验调高，
        // 让它能过阈值、走到预算检查这一步（否则先被阈值拦住）。
        let mut motion = motion_rule();
        motion.prior = 0.9;
        let mut d = f.daemon(vec![voice_rule(), motion], Some(0.01));
        f.clock.advance(60_000);
        d.sync();
        assert!(d.over_budget());
        // 非关键的动作规则被拦。
        let ev = event(SensorKind::Motion, 1.0, 10);
        assert_eq!(
            d.on_event(&ev),
            WakeDecision::StayAsleep { reason: StayAsleepReason::BudgetExhausted }
        );
        // 关键的唤醒词规则仍可唤醒。
        let ev2 = event(SensorKind::VoiceKeyword, 0.9, 10);
        assert!(matches!(d.on_event(&ev2), WakeDecision::Wake { .. }));
        assert!(d.budget_remaining_mwh().unwrap() < 0.0);
    }

    #[test]
    fn deep_sleep_only_allows_critical_rules() {
        let f = Fixture::new();
        let mut d = f.daemon(vec![voice_rule(), motion_rule()], None);
        d.enter_deep_sleep();
        assert_eq!(d.state(), PowerState::DeepSleep);
        let ev = event(SensorKind::Motion, 1.0, 2);
        assert_eq!(
            d.on_event(&ev),
            WakeDecision::StayAsleep { reason: StayAsleepReason::DeepSleep }
        );
        let ev2 = event(SensorKind::VoiceKeyword, 0.9, 2);
        assert!(matches!(d.on_event(&ev2), WakeDecision::Wake { .. }));
    }

    #[test]
    fn deep_sleep_draws_less_than_listening() {
        let f = Fixture::new();
        let mut d = f.daemon(vec![voice_rule()], None);
        d.enter_deep_sleep();
        f.clock.advance(3_600_000);
        d.sync();
        // 深睡 1 小时 = 0.5 mWh，监听 1 小时 = 3 mWh。
        assert!((d.energy_mwh() - 0.5).abs() < 0.01, "got {}", d.energy_mwh());
    }

    #[test]
    fn duty_cycle_counts_awake_time_only() {
        let f = Fixture::new();
        let mut d = f.daemon(vec![voice_rule()], None);
        f.clock.advance(90_000); // 监听 90 秒
        let ev = SensorEvent::new(SensorKind::VoiceKeyword, 0.9, f.clock.now_ms(), 10);
        assert!(matches!(d.on_event(&ev), WakeDecision::Wake { .. }));
        f.clock.advance(10_000); // 处理 10 秒
        d.go_back_to_sleep();
        f.clock.advance(0);
        d.sync();
        // 总 100 秒，Awake 10 秒：占空比 10%。
        let dc = d.duty_cycle().unwrap();
        assert!((dc - 0.10).abs() < 1e-9, "got {dc}");
    }

    #[test]
    fn wake_charges_spinup_energy() {
        let f = Fixture::new();
        let mut d = f.daemon(vec![voice_rule()], None);
        let before = d.energy_mwh();
        let ev = event(SensorKind::VoiceKeyword, 0.9, 10);
        assert!(matches!(d.on_event(&ev), WakeDecision::Wake { .. }));
        // 至少收了上电开销 + 传感器成本。
        let spinup_mwh = WAKE_SPINUP_UJ as f64 / UJ_PER_MWH;
        assert!(d.energy_mwh() - before >= spinup_mwh);
        assert_eq!(d.wakes_per_rule().get("唤醒词"), Some(&1));
    }

    #[test]
    fn polling_costs_more_than_event_driven_for_sparse_intents() {
        // 每 10 秒轮询一次，每次 NPU 上电 100 ms 检查：
        // 8640 次 × 0.3 J ≈ 720 mWh + 待机 ≈ 791 mWh/天。
        let polling = polling_daily_mwh(10_000, 100, POWER_AWAKE_MW, POWER_LISTENING_MW);
        // 事件驱动：监听 3 mW 常开 + 每天 20 次唤醒，
        // 每次上电 50 mJ + 处理 2 秒 × 3000 mW = 6.05 J：
        // 72 + 33.6 ≈ 105.6 mWh/天。
        let per_wake_uj = WAKE_SPINUP_UJ as f64 + 2_000.0 * POWER_AWAKE_MW;
        let event_driven =
            event_driven_daily_mwh(POWER_LISTENING_MW, 20, per_wake_uj);
        assert!(polling > event_driven, "polling {polling} vs event-driven {event_driven}");
        assert!((event_driven - 105.6).abs() < 1.0, "got {event_driven}");
    }

    #[test]
    fn polling_cost_explodes_as_interval_shrinks() {
        let slow = polling_daily_mwh(60_000, 100, POWER_AWAKE_MW, POWER_LISTENING_MW);
        let fast = polling_daily_mwh(1_000, 100, POWER_AWAKE_MW, POWER_LISTENING_MW);
        assert!(fast > slow);
        // 每秒轮询：NPU 几乎常开，接近 3000 mW × 24 h = 72000 mWh。
        assert!(fast > 7_000.0, "got {fast}");
    }

    #[test]
    fn render_lists_state_energy_and_wake_stats() {
        let f = Fixture::new();
        let mut d = f.daemon(vec![voice_rule()], Some(500.0));
        let ev = event(SensorKind::VoiceKeyword, 0.9, 10);
        assert!(matches!(d.on_event(&ev), WakeDecision::Wake { .. }));
        d.report_wake_outcome(true);
        d.go_back_to_sleep();
        let text = d.render();
        for key in ["常驻感知", "唤醒阈值", "能量", "占空比", "误唤醒率", "漏掉的意图"] {
            assert!(text.contains(key), "missing {key} in:\n{text}");
        }
    }
}
