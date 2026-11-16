//! 四种同时变稀缺的资源（设计文档第 3 章）：
//!
//! 操作系统每次演进都源于一种新资源变稀缺；这一次是四种同时变稀缺，
//! 而且互相冲突：
//!
//! * **R1 内存带宽**：生成 token 的每一步都要搬权重，搬不动就是慢，
//!   堆核无效。单位是字节。
//! * **R2 上下文窗口**：新的「内存」，但是会遗忘的内存——容量有限、
//!   访问有损、写错会一直错。单位是 token。
//! * **R3 可信上下文**：稀缺的不是信息，是能证明来源的信息。混进
//!   一段网页文本，整条决策链就被污染。单位是可信字节。
//! * **R4 人的注意力**：Agent 越多，请求确认越多。人是系统里吞吐
//!   最低、最容易饱和的组件。单位是打断次数。
//!
//! 本模块是四种资源的统一管家 [`ResourceGovernor`]：
//!
//! * 每种资源有独立的预算（容量），请求按种类入账；
//! * 超预算时的处理按种类区分，不是统一拒绝：
//!   * R1 超预算 → [`Decision::Defer`]，附带缓解建议「降精度」
//!     （第 2 章：先降精度再降模型/排队）；
//!   * R2 超预算 → [`Decision::Reject`]，硬上限不可超，附带缓解
//!     建议「逐出/摘要低价值上下文」；
//!   * R3 无可信出处 → [`Decision::Reject`]，来源不可证明的信息
//!     一律不进决策链（第 13 章的污点规则在资源层的落点）；
//!     可信字节超预算 → [`Decision::Defer`]，等下一窗口；
//!   * R4 超预算 → [`Decision::Defer`]，附带缓解建议「折进下一批
//!     摘要」（第 14 章的注意力拥塞控制在资源层的落点）；
//! * R1 / R2 是「占用型」资源（用完归还），R3 / R4 是「窗口内消耗
//!   型」资源（[`ResourceGovernor::reset_window`] 清零）；
//! * [`ResourceGovernor::binding_constraint`] 指出当前利用率最高、
//!   也就是真正卡住系统吞吐的那一种资源——四种资源互相冲突时，
//!   决定系统能走多快的是最紧的那一个；
//! * [`ResourceGovernor::report`] 输出一份稀缺报告。

/// 四种稀缺资源（设计文档第 3 章 R1–R4）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ResourceKind {
    /// R1 内存带宽，单位字节。
    MemoryBandwidth,
    /// R2 上下文窗口，单位 token。
    ContextWindow,
    /// R3 可信上下文，单位可信字节。
    TrustedContext,
    /// R4 人的注意力，单位打断次数。
    Attention,
}

impl ResourceKind {
    /// 章节里的编号。
    pub fn code(self) -> &'static str {
        match self {
            ResourceKind::MemoryBandwidth => "R1",
            ResourceKind::ContextWindow => "R2",
            ResourceKind::TrustedContext => "R3",
            ResourceKind::Attention => "R4",
        }
    }

    /// 资源名。
    pub fn name(self) -> &'static str {
        match self {
            ResourceKind::MemoryBandwidth => "内存带宽",
            ResourceKind::ContextWindow => "上下文窗口",
            ResourceKind::TrustedContext => "可信上下文",
            ResourceKind::Attention => "人的注意力",
        }
    }

    /// 计量单位。
    pub fn unit(self) -> &'static str {
        match self {
            ResourceKind::MemoryBandwidth => "字节",
            ResourceKind::ContextWindow => "token",
            ResourceKind::TrustedContext => "可信字节",
            ResourceKind::Attention => "次打断",
        }
    }

    /// 是否「占用型」资源：用完归还。R1 / R2 是占用型，R3 / R4 是
    /// 窗口内消耗型，只能等窗口重置。
    pub fn is_held(self) -> bool {
        matches!(self, ResourceKind::MemoryBandwidth | ResourceKind::ContextWindow)
    }

    /// 四种资源，按章节顺序。
    pub fn all() -> [ResourceKind; 4] {
        [
            ResourceKind::MemoryBandwidth,
            ResourceKind::ContextWindow,
            ResourceKind::TrustedContext,
            ResourceKind::Attention,
        ]
    }
}

/// 超预算时可采取的缓解动作。`Decision` 里附带它，是为了让调用方
/// 不用再查一遍文档就知道该往哪个方向退。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mitigation {
    /// R1：先降精度（8bit → 4bit），再考虑降模型/排队（第 2 章）。
    DropPrecision,
    /// R2：逐出或摘要低价值上下文，腾出窗口。
    EvictContext,
    /// R4：把这次打断折进下一批摘要，不单独打扰人。
    BatchInterruptions,
}

/// 资源请求的裁决。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    /// 准入，已入账。
    Admit,
    /// 暂缓：预算在当前窗口内不够，等释放或等下一窗口。附带缓解建议。
    Defer { reason: String, hint: Option<Mitigation> },
    /// 拒绝：硬性规则不允许（硬上限、或无可信出处），重试也不会通过。
    Reject { reason: String, hint: Option<Mitigation> },
}

/// 四种资源的预算（容量）。`0` 表示该资源未设限（无限）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Capacities {
    /// R1：每窗口可搬运的字节数。
    pub memory_bandwidth_bytes: u64,
    /// R2：上下文窗口总 token 数。
    pub context_window_tokens: u64,
    /// R3：每窗口可接纳的可信字节数。
    pub trusted_context_bytes: u64,
    /// R4：每窗口允许的打断次数。
    pub attention_interruptions: u64,
}

impl Capacities {
    pub fn capacity(&self, kind: ResourceKind) -> u64 {
        match kind {
            ResourceKind::MemoryBandwidth => self.memory_bandwidth_bytes,
            ResourceKind::ContextWindow => self.context_window_tokens,
            ResourceKind::TrustedContext => self.trusted_context_bytes,
            ResourceKind::Attention => self.attention_interruptions,
        }
    }
}

/// 四种稀缺资源的统一管家。
#[derive(Debug)]
pub struct ResourceGovernor {
    capacities: Capacities,
    used: [u64; 4],
}

impl ResourceGovernor {
    pub fn new(capacities: Capacities) -> Self {
        ResourceGovernor { capacities, used: [0; 4] }
    }

    fn idx(kind: ResourceKind) -> usize {
        kind as usize
    }

    /// 该资源当前已用多少。
    pub fn used(&self, kind: ResourceKind) -> u64 {
        self.used[Self::idx(kind)]
    }

    /// 该资源当前利用率（0.0–1.0 之间；容量为 0 即未设限时返回 0.0）。
    pub fn utilization(&self, kind: ResourceKind) -> f64 {
        let cap = self.capacities.capacity(kind);
        if cap == 0 {
            0.0
        } else {
            self.used(kind) as f64 / cap as f64
        }
    }

    /// 当前剩余量；容量为 0（未设限）时返回 `u64::MAX`。
    pub fn remaining(&self, kind: ResourceKind) -> u64 {
        let cap = self.capacities.capacity(kind);
        if cap == 0 {
            u64::MAX
        } else {
            cap.saturating_sub(self.used(kind))
        }
    }

    /// 请求 `amount` 单位的 `kind` 资源。
    ///
    /// `provenance_trusted` 只对 R3 有意义：可信上下文必须能证明来源，
    /// 无可信出处一律拒绝（第 13 章污点规则的资源层落点）。
    /// 其它资源忽略该参数。
    pub fn request(
        &mut self,
        kind: ResourceKind,
        amount: u64,
        provenance_trusted: bool,
    ) -> Decision {
        if amount == 0 {
            return Decision::Admit;
        }
        // R3 的出处门：来源不可证明，预算再多也不接纳。
        if kind == ResourceKind::TrustedContext && !provenance_trusted {
            return Decision::Reject {
                reason: "无可信出处：来源不可证明的信息不进决策链".to_string(),
                hint: None,
            };
        }
        let cap = self.capacities.capacity(kind);
        if cap != 0 && self.used(kind).saturating_add(amount) > cap {
            return match kind {
                ResourceKind::MemoryBandwidth => Decision::Defer {
                    reason: format!(
                        "内存带宽不足：需要 {amount} 字节，剩余 {} 字节",
                        self.remaining(kind)
                    ),
                    hint: Some(Mitigation::DropPrecision),
                },
                ResourceKind::ContextWindow => Decision::Reject {
                    reason: format!(
                        "上下文窗口已满：需要 {amount} token，剩余 {} token（硬上限）",
                        self.remaining(kind)
                    ),
                    hint: Some(Mitigation::EvictContext),
                },
                ResourceKind::TrustedContext => Decision::Defer {
                    reason: format!(
                        "可信上下文预算不足：需要 {amount} 可信字节，剩余 {} 可信字节",
                        self.remaining(kind)
                    ),
                    hint: None,
                },
                ResourceKind::Attention => Decision::Defer {
                    reason: format!(
                        "注意力预算用完：本窗口只剩 {} 次打断",
                        self.remaining(kind)
                    ),
                    hint: Some(Mitigation::BatchInterruptions),
                },
            };
        }
        self.used[Self::idx(kind)] = self.used(kind).saturating_add(amount);
        Decision::Admit
    }

    /// 归还占用型资源（R1 / R2）。归还量超过已用量时按已用量归零。
    /// 对窗口消耗型资源（R3 / R4）调用是无操作，返回 `false`。
    pub fn release(&mut self, kind: ResourceKind, amount: u64) -> bool {
        if !kind.is_held() {
            return false;
        }
        let i = Self::idx(kind);
        self.used[i] = self.used[i].saturating_sub(amount);
        true
    }

    /// 进入下一窗口：窗口内消耗型资源（R3 / R4）清零，占用型资源
    /// （R1 / R2）不受影响——占用要靠归还，不靠窗口刷新。
    pub fn reset_window(&mut self) {
        for kind in ResourceKind::all() {
            if !kind.is_held() {
                self.used[Self::idx(kind)] = 0;
            }
        }
    }

    /// 当前最紧的约束资源：利用率最高的那一种。四种资源互相冲突时，
    /// 系统吞吐由它决定。没有任何资源被使用时返回 `None`。
    pub fn binding_constraint(&self) -> Option<ResourceKind> {
        ResourceKind::all()
            .into_iter()
            .filter(|k| self.capacities.capacity(*k) != 0 && self.used(*k) > 0)
            .max_by(|a, b| {
                self.utilization(*a)
                    .partial_cmp(&self.utilization(*b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// 稀缺报告：四种资源的用量/容量/利用率，以及当前最紧的约束。
    pub fn report(&self) -> String {
        let mut out = String::from("稀缺资源报告（第 3 章 R1–R4）：\n");
        for kind in ResourceKind::all() {
            let cap = self.capacities.capacity(kind);
            let cap_text =
                if cap == 0 { "未设限".to_string() } else { format!("{cap}") };
            out.push_str(&format!(
                "  {} {}：已用 {} / 容量 {} {}，利用率 {:.1}%\n",
                kind.code(),
                kind.name(),
                self.used(kind),
                cap_text,
                kind.unit(),
                self.utilization(kind) * 100.0
            ));
        }
        match self.binding_constraint() {
            Some(k) => out.push_str(&format!(
                "当前最紧约束：{} {}（利用率 {:.1}%）\n",
                k.code(),
                k.name(),
                self.utilization(k) * 100.0
            )),
            None => out.push_str("当前无约束：四种资源均未使用\n"),
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps() -> Capacities {
        Capacities {
            memory_bandwidth_bytes: 1_000,
            context_window_tokens: 100,
            trusted_context_bytes: 500,
            attention_interruptions: 3,
        }
    }

    #[test]
    fn admit_within_budget_and_report_tracks_usage() {
        let mut g = ResourceGovernor::new(caps());
        assert_eq!(g.request(ResourceKind::MemoryBandwidth, 400, true), Decision::Admit);
        assert_eq!(g.request(ResourceKind::ContextWindow, 50, true), Decision::Admit);
        assert_eq!(g.request(ResourceKind::TrustedContext, 200, true), Decision::Admit);
        assert_eq!(g.request(ResourceKind::Attention, 1, true), Decision::Admit);
        assert_eq!(g.used(ResourceKind::MemoryBandwidth), 400);
        assert!((g.utilization(ResourceKind::ContextWindow) - 0.5).abs() < 1e-9);
        let report = g.report();
        assert!(report.contains("R1 内存带宽"));
        assert!(report.contains("R4 人的注意力"));
    }

    #[test]
    fn r1_over_budget_defers_with_precision_hint_and_admits_after_release() {
        let mut g = ResourceGovernor::new(caps());
        assert_eq!(g.request(ResourceKind::MemoryBandwidth, 800, true), Decision::Admit);
        let d = g.request(ResourceKind::MemoryBandwidth, 300, true);
        assert_eq!(
            d,
            Decision::Defer {
                reason: "内存带宽不足：需要 300 字节，剩余 200 字节".to_string(),
                hint: Some(Mitigation::DropPrecision),
            }
        );
        // 暂缓不入账。
        assert_eq!(g.used(ResourceKind::MemoryBandwidth), 800);
        assert!(g.release(ResourceKind::MemoryBandwidth, 400));
        assert_eq!(g.request(ResourceKind::MemoryBandwidth, 300, true), Decision::Admit);
    }

    #[test]
    fn r2_over_cap_rejects_with_evict_hint_and_no_partial_admit() {
        let mut g = ResourceGovernor::new(caps());
        assert_eq!(g.request(ResourceKind::ContextWindow, 90, true), Decision::Admit);
        let d = g.request(ResourceKind::ContextWindow, 20, true);
        assert_eq!(
            d,
            Decision::Reject {
                reason: "上下文窗口已满：需要 20 token，剩余 10 token（硬上限）".to_string(),
                hint: Some(Mitigation::EvictContext),
            }
        );
        assert_eq!(g.used(ResourceKind::ContextWindow), 90);
        // 归还后可以再进。
        assert!(g.release(ResourceKind::ContextWindow, 50));
        assert_eq!(g.request(ResourceKind::ContextWindow, 20, true), Decision::Admit);
    }

    #[test]
    fn r3_rejects_untrusted_provenance_even_with_budget_left() {
        let mut g = ResourceGovernor::new(caps());
        let d = g.request(ResourceKind::TrustedContext, 10, false);
        assert!(matches!(d, Decision::Reject { hint: None, .. }));
        assert_eq!(g.used(ResourceKind::TrustedContext), 0);
        assert_eq!(g.request(ResourceKind::TrustedContext, 10, true), Decision::Admit);
    }

    #[test]
    fn r3_trusted_over_budget_defers_until_next_window() {
        let mut g = ResourceGovernor::new(caps());
        assert_eq!(g.request(ResourceKind::TrustedContext, 500, true), Decision::Admit);
        assert!(matches!(
            g.request(ResourceKind::TrustedContext, 1, true),
            Decision::Defer { hint: None, .. }
        ));
        g.reset_window();
        assert_eq!(g.used(ResourceKind::TrustedContext), 0);
        assert_eq!(g.request(ResourceKind::TrustedContext, 1, true), Decision::Admit);
    }

    #[test]
    fn r4_over_budget_defers_with_batch_hint_and_window_reset_frees_it() {
        let mut g = ResourceGovernor::new(caps());
        for _ in 0..3 {
            assert_eq!(g.request(ResourceKind::Attention, 1, true), Decision::Admit);
        }
        let d = g.request(ResourceKind::Attention, 1, true);
        assert_eq!(
            d,
            Decision::Defer {
                reason: "注意力预算用完：本窗口只剩 0 次打断".to_string(),
                hint: Some(Mitigation::BatchInterruptions),
            }
        );
        // 注意力是窗口消耗型，不能靠 release 归还。
        assert!(!g.release(ResourceKind::Attention, 1));
        assert_eq!(g.used(ResourceKind::Attention), 3);
        g.reset_window();
        assert_eq!(g.request(ResourceKind::Attention, 1, true), Decision::Admit);
    }

    #[test]
    fn reset_window_leaves_held_resources_alone() {
        let mut g = ResourceGovernor::new(caps());
        assert_eq!(g.request(ResourceKind::MemoryBandwidth, 100, true), Decision::Admit);
        assert_eq!(g.request(ResourceKind::Attention, 1, true), Decision::Admit);
        g.reset_window();
        assert_eq!(g.used(ResourceKind::MemoryBandwidth), 100);
        assert_eq!(g.used(ResourceKind::Attention), 0);
    }

    #[test]
    fn binding_constraint_is_the_highest_utilization() {
        let mut g = ResourceGovernor::new(caps());
        assert_eq!(g.binding_constraint(), None);
        assert_eq!(g.request(ResourceKind::MemoryBandwidth, 500, true), Decision::Admit);
        assert_eq!(g.request(ResourceKind::ContextWindow, 90, true), Decision::Admit);
        // R2 90% > R1 50%，最紧的是 R2。
        assert_eq!(g.binding_constraint(), Some(ResourceKind::ContextWindow));
        // R1 追到 95% 后，最紧的变成 R1。
        assert_eq!(g.request(ResourceKind::MemoryBandwidth, 450, true), Decision::Admit);
        assert_eq!(g.binding_constraint(), Some(ResourceKind::MemoryBandwidth));
    }

    #[test]
    fn zero_amount_admits_without_accounting() {
        let mut g = ResourceGovernor::new(caps());
        assert_eq!(g.request(ResourceKind::Attention, 0, true), Decision::Admit);
        assert_eq!(g.used(ResourceKind::Attention), 0);
    }

    #[test]
    fn unlimited_capacity_never_blocks() {
        let mut g = ResourceGovernor::new(Capacities::default());
        assert_eq!(
            g.request(ResourceKind::MemoryBandwidth, 1_000_000, true),
            Decision::Admit
        );
        assert_eq!(g.utilization(ResourceKind::MemoryBandwidth), 0.0);
        assert_eq!(g.remaining(ResourceKind::MemoryBandwidth), u64::MAX);
    }

    #[test]
    fn report_names_binding_constraint() {
        let mut g = ResourceGovernor::new(caps());
        assert_eq!(g.request(ResourceKind::Attention, 3, true), Decision::Admit);
        let report = g.report();
        assert!(report.contains("当前最紧约束：R4 人的注意力"));
        assert!(report.contains("利用率 100.0%"));
    }
}
