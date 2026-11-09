//! OS 级评估基准（设计文档第 21 章开放问题 5）：OSWorld 只测「能不能
//! 完成任务」，不测「成本多少、打扰几次、撤销几次」——而这三项恰恰是
//! 操作系统的核心指标。本模块是缺失的那块基准的原型：一次任务运行
//! 记录完成情况之外的全部 OS 成本，套件汇总成可比较的分数，两个套件
//! 之间做 A/B 对比。
//!
//! 诚实性规则（与第 22 章度量看板一致）：
//!
//! * 没有数据的率一律是 `None`，不报 0% 也不报 100%。
//! * 未完成的任务 OS 分为 0——绕远路做对了和没做对是两回事，但没
//!   做对就是没做对，不靠成本分找补。
//! * 综合分是 v1 口径：只在同一基准版本、同一任务集内可比；跨版本、
//!   跨任务集比较分数没有意义，对比函数只接受同口径的两个套件。
//! * 完成率差距过大（绝对值 > 10 个百分点）时不直接比综合分：
//!   一个完成 50% 但很便宜的 agent 和一个完成 95% 但贵一点的
//!   agent，分高低说明不了谁更好，先报完成率差距。

/// 综合分 v1 的成本半衰点：花掉这么多 token，成本因子恰好为 0.5。
pub const SCORE_COST_HALF_LIFE_TOKENS: f64 = 10_000.0;

/// 完成率差距超过这个绝对值，对比结论记为「不可直接比综合分」。
pub const COMPARABILITY_COMPLETION_GAP: f64 = 0.10;

/// 综合分相差小于这个值，视为打平。
pub const SCORE_TIE_EPSILON: f64 = 0.005;

/// 一次任务运行的完整记录。`minimal_steps` 是任务定义里给出的参考
/// 最优路径步数（人工标注或上一版最优 agent 的步数）；未知时为
/// `None`，此时不扣路径效率分，但也不计入平均效率。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskRun {
    pub task_id: String,
    pub completed: bool,
    pub steps_taken: u32,
    pub minimal_steps: Option<u32>,
    /// 本次运行消耗的 token 总数（全部模型调用合计）。
    pub cost_tokens: u64,
    pub model_calls: u32,
    pub wall_ms: u64,
    /// 运行中向用户发起的打扰次数（澄清问题、确认、纠错提示合计）。
    pub interruptions: u32,
    /// 运行中实际撤销/补偿掉的已执行动作数。
    pub undos: u32,
}

impl TaskRun {
    pub fn new(task_id: impl Into<String>) -> Self {
        TaskRun {
            task_id: task_id.into(),
            completed: false,
            steps_taken: 0,
            minimal_steps: None,
            cost_tokens: 0,
            model_calls: 0,
            wall_ms: 0,
            interruptions: 0,
            undos: 0,
        }
    }

    /// 路径效率 = 参考最优步数 / 实际步数。超过参考路径（比最优还
    /// 少步）时封顶为 1.0——基准不奖励「比参考更优」，只记录不劣于
    /// 参考。未完成、零步、或参考步数未知时为 `None`。
    pub fn efficiency(&self) -> Option<f64> {
        if !self.completed || self.steps_taken == 0 {
            return None;
        }
        let minimal = self.minimal_steps?;
        if minimal == 0 {
            return None;
        }
        Some((minimal as f64 / self.steps_taken as f64).min(1.0))
    }

    /// 单次运行的 OS 综合分 v1，取值 [0, 1]：
    /// 未完成 = 0；完成时 = 路径效率 × 成本因子 × 打扰因子 × 撤销因子。
    /// 每个因子都是 1/(1+x) 形：零成本/零打扰/零撤销时为 1，每多一
    /// 单位成本就递减，永不归零也不为负。
    pub fn os_score(&self) -> f64 {
        if !self.completed {
            return 0.0;
        }
        let efficiency = self.efficiency().unwrap_or(1.0);
        let cost_factor = 1.0 / (1.0 + self.cost_tokens as f64 / SCORE_COST_HALF_LIFE_TOKENS);
        let interrupt_factor = 1.0 / (1.0 + self.interruptions as f64);
        let undo_factor = 1.0 / (1.0 + self.undos as f64);
        efficiency * cost_factor * interrupt_factor * undo_factor
    }
}

/// 向套件里加一条运行记录时的失败原因。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BenchmarkError {
    /// 同一任务已有一条记录：一个任务在一个套件里只跑一次，
    /// 重跑必须新建套件，否则完成率会被重复计数灌水。
    DuplicateTaskId(String),
}

/// 一个 agent（或同一 agent 的一个版本）在一组任务上的完整跑分。
#[derive(Debug, Default, Clone)]
pub struct BenchmarkSuite {
    pub name: String,
    runs: Vec<TaskRun>,
}

impl BenchmarkSuite {
    pub fn new(name: impl Into<String>) -> Self {
        BenchmarkSuite { name: name.into(), runs: Vec::new() }
    }

    pub fn add_run(&mut self, run: TaskRun) -> Result<(), BenchmarkError> {
        if self.runs.iter().any(|r| r.task_id == run.task_id) {
            return Err(BenchmarkError::DuplicateTaskId(run.task_id));
        }
        self.runs.push(run);
        Ok(())
    }

    pub fn runs(&self) -> &[TaskRun] {
        &self.runs
    }

    pub fn len(&self) -> usize {
        self.runs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.runs.is_empty()
    }

    pub fn summary(&self) -> SuiteSummary {
        let total = self.runs.len();
        if total == 0 {
            return SuiteSummary::empty(&self.name);
        }
        let completed: Vec<&TaskRun> = self.runs.iter().filter(|r| r.completed).collect();
        let n_completed = completed.len();
        let completion_rate = n_completed as f64 / total as f64;
        let mean_os_score =
            self.runs.iter().map(TaskRun::os_score).sum::<f64>() / total as f64;
        // 平均效率只统计「完成且参考步数已知」的运行，未完成或参考
        // 未知的运行不参与——不拿缺失数据拉低或抬高均值。
        let efficiencies: Vec<f64> =
            self.runs.iter().filter_map(TaskRun::efficiency).collect();
        let mean_efficiency = if efficiencies.is_empty() {
            None
        } else {
            Some(efficiencies.iter().sum::<f64>() / efficiencies.len() as f64)
        };
        let mean_cost_per_completed = if n_completed == 0 {
            None
        } else {
            Some(
                completed.iter().map(|r| r.cost_tokens).sum::<u64>() as f64
                    / n_completed as f64,
            )
        };
        let mean_interruptions =
            self.runs.iter().map(|r| r.interruptions).sum::<u32>() as f64 / total as f64;
        let mean_undos =
            self.runs.iter().map(|r| r.undos).sum::<u32>() as f64 / total as f64;
        let mean_steps_per_completed = if n_completed == 0 {
            None
        } else {
            Some(
                completed.iter().map(|r| r.steps_taken).sum::<u32>() as f64
                    / n_completed as f64,
            )
        };
        SuiteSummary {
            name: self.name.clone(),
            total_runs: total,
            completed_runs: n_completed,
            completion_rate: Some(completion_rate),
            mean_os_score: Some(mean_os_score),
            mean_efficiency,
            mean_cost_per_completed,
            mean_interruptions_per_task: Some(mean_interruptions),
            mean_undos_per_task: Some(mean_undos),
            mean_steps_per_completed,
        }
    }
}

/// 一个套件的汇总分数。空套件时所有率都是 `None`。
#[derive(Debug, Clone)]
pub struct SuiteSummary {
    pub name: String,
    pub total_runs: usize,
    pub completed_runs: usize,
    pub completion_rate: Option<f64>,
    pub mean_os_score: Option<f64>,
    pub mean_efficiency: Option<f64>,
    pub mean_cost_per_completed: Option<f64>,
    pub mean_interruptions_per_task: Option<f64>,
    pub mean_undos_per_task: Option<f64>,
    pub mean_steps_per_completed: Option<f64>,
}

impl SuiteSummary {
    fn empty(name: &str) -> Self {
        SuiteSummary {
            name: name.to_string(),
            total_runs: 0,
            completed_runs: 0,
            completion_rate: None,
            mean_os_score: None,
            mean_efficiency: None,
            mean_cost_per_completed: None,
            mean_interruptions_per_task: None,
            mean_undos_per_task: None,
            mean_steps_per_completed: None,
        }
    }

    /// 一行一个指标：名、值、单位。无数据的指标写「无数据」。
    pub fn render(&self) -> String {
        let mut out = format!("OS 评估基准（第 21 章 Q5）——套件「{}」\n", self.name);
        out.push_str(&format!(
            "任务数：{}，完成：{}，完成率：{}\n",
            self.total_runs,
            self.completed_runs,
            fmt_rate(self.completion_rate)
        ));
        out.push_str(&format!("综合分 v1（均值）：{}\n", fmt_score(self.mean_os_score)));
        out.push_str(&format!("平均路径效率：{}\n", fmt_rate(self.mean_efficiency)));
        out.push_str(&format!(
            "每完成任务平均成本：{}\n",
            fmt_tokens(self.mean_cost_per_completed)
        ));
        out.push_str(&format!(
            "每任务平均打扰：{}\n",
            fmt_count(self.mean_interruptions_per_task)
        ));
        out.push_str(&format!(
            "每任务平均撤销：{}\n",
            fmt_count(self.mean_undos_per_task)
        ));
        out.push_str(&format!(
            "每完成任务平均步数：{}\n",
            fmt_count(self.mean_steps_per_completed)
        ));
        out
    }
}

fn fmt_rate(rate: Option<f64>) -> String {
    match rate {
        None => "—（无数据）".to_string(),
        Some(r) => format!("{:.1}%", r * 100.0),
    }
}

fn fmt_score(score: Option<f64>) -> String {
    match score {
        None => "—（无数据）".to_string(),
        Some(s) => format!("{s:.3}"),
    }
}

fn fmt_tokens(tokens: Option<f64>) -> String {
    match tokens {
        None => "—（无数据）".to_string(),
        Some(t) => format!("{t:.0} tokens"),
    }
}

fn fmt_count(count: Option<f64>) -> String {
    match count {
        None => "—（无数据）".to_string(),
        Some(c) => format!("{c:.2}"),
    }
}

/// 两个套件对比的结论。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// A 的综合分更高，且完成率差距在可比范围内。
    A,
    /// B 的综合分更高，且完成率差距在可比范围内。
    B,
    /// 综合分在误差内打平。
    Tie,
    /// 完成率差距过大：先比完成率，不直接比综合分。
    /// `completion_leader_is_a` 说明哪边完成率更高。
    Incomparable { completion_leader_is_a: bool },
}

/// 两个同口径套件的 A/B 对比。差值一律按「B − A」计：
/// 完成率/综合分为正表示 B 更好，成本/打扰/撤销/步数为正表示 B 更贵。
#[derive(Debug, Clone)]
pub struct Comparison {
    pub a_name: String,
    pub b_name: String,
    pub completion_rate_delta: Option<f64>,
    pub os_score_delta: Option<f64>,
    pub efficiency_delta: Option<f64>,
    pub cost_per_completed_delta: Option<f64>,
    pub interruptions_delta: Option<f64>,
    pub undos_delta: Option<f64>,
    pub verdict: Verdict,
}

fn delta(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(y - x),
        _ => None,
    }
}

/// 对比两个套件。任一套件为空时无法对比，结论为 `Tie` 以外的
/// `Incomparable` 是不可能的——此时直接返回完成率领跑方未知的
/// `Incomparable` 是错的，所以空套件对比返回 `Tie` 并把全部差值
/// 置 `None`，调用方应先检查 `total_runs`。
pub fn compare(a: &SuiteSummary, b: &SuiteSummary) -> Comparison {
    let completion_rate_delta = delta(a.completion_rate, b.completion_rate);
    let os_score_delta = delta(a.mean_os_score, b.mean_os_score);
    let verdict = match (a.completion_rate, b.completion_rate, a.mean_os_score, b.mean_os_score)
    {
        (Some(ca), Some(cb), Some(sa), Some(sb)) => {
            if (ca - cb).abs() > COMPARABILITY_COMPLETION_GAP {
                Verdict::Incomparable { completion_leader_is_a: ca > cb }
            } else if (sb - sa).abs() < SCORE_TIE_EPSILON {
                Verdict::Tie
            } else if sb > sa {
                Verdict::B
            } else {
                Verdict::A
            }
        }
        // 任一没有数据：无法下结论，记为打平，差值为 None。
        _ => Verdict::Tie,
    };
    Comparison {
        a_name: a.name.clone(),
        b_name: b.name.clone(),
        completion_rate_delta,
        os_score_delta,
        efficiency_delta: delta(a.mean_efficiency, b.mean_efficiency),
        cost_per_completed_delta: delta(a.mean_cost_per_completed, b.mean_cost_per_completed),
        interruptions_delta: delta(a.mean_interruptions_per_task, b.mean_interruptions_per_task),
        undos_delta: delta(a.mean_undos_per_task, b.mean_undos_per_task),
        verdict,
    }
}

fn fmt_signed(delta: Option<f64>, unit: &str) -> String {
    match delta {
        None => "—".to_string(),
        Some(d) => format!("{d:+.3}{unit}"),
    }
}

impl Comparison {
    /// 对比报告：先给结论，再逐项列差值（B − A）。
    pub fn render(&self) -> String {
        let mut out = format!("OS 评估基准对比：{} vs {}\n", self.a_name, self.b_name);
        let verdict_text = match self.verdict {
            Verdict::A => format!("结论：{} 胜（综合分更高）", self.a_name),
            Verdict::B => format!("结论：{} 胜（综合分更高）", self.b_name),
            Verdict::Tie => "结论：打平（综合分在误差内，或缺少可比数据）".to_string(),
            Verdict::Incomparable { completion_leader_is_a } => {
                let leader = if completion_leader_is_a { &self.a_name } else { &self.b_name };
                format!(
                    "结论：完成率差距超过 {:.0} 个百分点，不直接比综合分——{leader} 完成率更高",
                    COMPARABILITY_COMPLETION_GAP * 100.0
                )
            }
        };
        out.push_str(&verdict_text);
        out.push('\n');
        out.push_str(&format!(
            "完成率差值（B−A）：{}\n",
            fmt_signed(self.completion_rate_delta, "")
        ));
        out.push_str(&format!(
            "综合分差值（B−A）：{}\n",
            fmt_signed(self.os_score_delta, "")
        ));
        out.push_str(&format!(
            "路径效率差值（B−A）：{}\n",
            fmt_signed(self.efficiency_delta, "")
        ));
        out.push_str(&format!(
            "每完成任务成本差值（B−A）：{}\n",
            fmt_signed(self.cost_per_completed_delta, " tokens")
        ));
        out.push_str(&format!(
            "每任务打扰差值（B−A）：{}\n",
            fmt_signed(self.interruptions_delta, " 次")
        ));
        out.push_str(&format!(
            "每任务撤销差值（B−A）：{}\n",
            fmt_signed(self.undos_delta, " 次")
        ));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(actual: Option<f64>, expected: f64) {
        let v = actual.expect("expected data");
        assert!((v - expected).abs() < 1e-9, "expected {expected}, got {v}");
    }

    fn run(
        id: &str,
        completed: bool,
        steps: u32,
        minimal: Option<u32>,
        tokens: u64,
        interruptions: u32,
        undos: u32,
    ) -> TaskRun {
        TaskRun {
            task_id: id.to_string(),
            completed,
            steps_taken: steps,
            minimal_steps: minimal,
            cost_tokens: tokens,
            model_calls: 1,
            wall_ms: 1000,
            interruptions,
            undos,
        }
    }

    #[test]
    fn empty_suite_reports_no_data() {
        let s = BenchmarkSuite::new("空").summary();
        assert_eq!(s.total_runs, 0);
        assert_eq!(s.completion_rate, None);
        assert_eq!(s.mean_os_score, None);
        assert_eq!(s.mean_efficiency, None);
        assert_eq!(s.mean_cost_per_completed, None);
        assert!(s.render().contains("无数据"));
    }

    #[test]
    fn efficiency_is_minimal_over_steps_capped_at_one() {
        // 10 步参考，实际 20 步：效率 0.5。
        assert_eq!(run("a", true, 20, Some(10), 0, 0, 0).efficiency(), Some(0.5));
        // 实际比参考还少：封顶 1.0，不奖励超参考。
        assert_eq!(run("b", true, 8, Some(10), 0, 0, 0).efficiency(), Some(1.0));
        // 未完成没有效率。
        assert_eq!(run("c", false, 20, Some(10), 0, 0, 0).efficiency(), None);
        // 参考步数未知时没有效率。
        assert_eq!(run("d", true, 20, None, 0, 0, 0).efficiency(), None);
        // 参考步数为 0 是无效标注，不产生效率。
        assert_eq!(run("e", true, 20, Some(0), 0, 0, 0).efficiency(), None);
    }

    #[test]
    fn clean_completion_scores_one() {
        // 零成本、零打扰、零撤销、步数等于参考：满分。
        let r = run("a", true, 10, Some(10), 0, 0, 0);
        assert!((r.os_score() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn incomplete_run_scores_zero_no_matter_the_cost() {
        let r = run("a", false, 5, Some(10), 0, 0, 0);
        assert_eq!(r.os_score(), 0.0);
    }

    #[test]
    fn score_factors_multiply_as_documented() {
        // 效率 0.5，成本 10_000 token（因子 0.5），打扰 1 次（因子 0.5），
        // 撤销 1 次（因子 0.5）：0.5^4 = 0.0625。
        let r = run("a", true, 20, Some(10), 10_000, 1, 1);
        assert!((r.os_score() - 0.0625).abs() < 1e-9, "got {}", r.os_score());
    }

    #[test]
    fn unknown_reference_path_takes_no_efficiency_penalty() {
        // 参考未知：效率因子按 1.0 计，只扣成本分。
        let r = run("a", true, 20, None, 10_000, 0, 0);
        assert!((r.os_score() - 0.5).abs() < 1e-9, "got {}", r.os_score());
    }

    #[test]
    fn duplicate_task_id_is_rejected() {
        let mut suite = BenchmarkSuite::new("s");
        suite.add_run(run("t1", true, 5, Some(5), 100, 0, 0)).unwrap();
        let err = suite.add_run(run("t1", true, 5, Some(5), 100, 0, 0)).unwrap_err();
        assert_eq!(err, BenchmarkError::DuplicateTaskId("t1".to_string()));
        assert_eq!(suite.len(), 1);
    }

    #[test]
    fn summary_rates_follow_their_definitions() {
        let mut suite = BenchmarkSuite::new("s");
        suite.add_run(run("t1", true, 10, Some(10), 1_000, 0, 0)).unwrap();
        suite.add_run(run("t2", true, 20, Some(10), 3_000, 2, 1)).unwrap();
        suite.add_run(run("t3", false, 30, Some(10), 5_000, 3, 2)).unwrap();
        let s = suite.summary();
        approx(s.completion_rate, 2.0 / 3.0);
        // 平均效率只统计完成且参考已知的两条：(1.0 + 0.5) / 2。
        approx(s.mean_efficiency, 0.75);
        // 每完成任务平均成本只统计完成的两条：(1000 + 3000) / 2。
        approx(s.mean_cost_per_completed, 2000.0);
        // 打扰/撤销按全部三条平均。
        approx(s.mean_interruptions_per_task, 5.0 / 3.0);
        approx(s.mean_undos_per_task, 1.0);
        // 未完成那条的 OS 分为 0，拉低综合分均值。
        let expected = (run("t1", true, 10, Some(10), 1_000, 0, 0).os_score()
            + run("t2", true, 20, Some(10), 3_000, 2, 1).os_score())
            / 3.0;
        approx(s.mean_os_score, expected);
    }

    #[test]
    fn render_lists_every_os_metric() {
        let mut suite = BenchmarkSuite::new("s");
        suite.add_run(run("t1", true, 10, Some(10), 1_000, 0, 0)).unwrap();
        let rendered = suite.summary().render();
        for name in ["完成率", "综合分", "路径效率", "成本", "打扰", "撤销", "步数"] {
            assert!(rendered.contains(name), "missing {name} in:\n{rendered}");
        }
    }

    #[test]
    fn comparison_picks_the_higher_score_when_comparable() {
        let mut a = BenchmarkSuite::new("A");
        a.add_run(run("t1", true, 10, Some(10), 1_000, 0, 0)).unwrap();
        let mut b = BenchmarkSuite::new("B");
        b.add_run(run("t1", true, 10, Some(10), 1_000, 1, 0)).unwrap();
        // 完成率相同（都是 100%），B 多打扰一次，综合分更低：A 胜。
        let c = compare(&a.summary(), &b.summary());
        assert_eq!(c.verdict, Verdict::A);
        assert!(c.os_score_delta.unwrap() < 0.0);
        assert!(c.interruptions_delta.unwrap() > 0.0);
        assert!(c.render().contains("A 胜"));
    }

    #[test]
    fn large_completion_gap_blocks_score_comparison() {
        let mut a = BenchmarkSuite::new("A");
        for i in 0..10 {
            a.add_run(run(&format!("t{i}"), true, 10, Some(10), 1_000, 0, 0)).unwrap();
        }
        let mut b = BenchmarkSuite::new("B");
        for i in 0..10 {
            // B 只完成一半，但完成的那几条极便宜：综合分可能反超 A。
            let done = i < 5;
            b.add_run(run(&format!("t{i}"), done, 10, Some(10), 10, 0, 0)).unwrap();
        }
        let c = compare(&a.summary(), &b.summary());
        assert_eq!(
            c.verdict,
            Verdict::Incomparable { completion_leader_is_a: true }
        );
        assert!(c.render().contains("不直接比综合分"));
    }

    #[test]
    fn near_equal_scores_tie() {
        let mut a = BenchmarkSuite::new("A");
        a.add_run(run("t1", true, 10, Some(10), 1_000, 0, 0)).unwrap();
        let mut b = BenchmarkSuite::new("B");
        b.add_run(run("t1", true, 10, Some(10), 1_001, 0, 0)).unwrap();
        assert_eq!(compare(&a.summary(), &b.summary()).verdict, Verdict::Tie);
    }

    #[test]
    fn empty_suite_comparison_has_no_deltas() {
        let c = compare(
            &BenchmarkSuite::new("A").summary(),
            &BenchmarkSuite::new("B").summary(),
        );
        assert_eq!(c.verdict, Verdict::Tie);
        assert_eq!(c.completion_rate_delta, None);
        assert_eq!(c.os_score_delta, None);
    }
}
