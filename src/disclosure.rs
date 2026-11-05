//! Progressive disclosure rendering (doc ch. 19, §2).
//!
//! The intent bar returns a finished result, not links — but a finished
//! result the user cannot inspect is a black box, and a black box is not
//! trusted. The design rule is **progressive disclosure**: the first
//! screen shows only the conclusion and the single most important action;
//! the user can drill one level deeper into the evidence (which files,
//! which memories the conclusion relied on), and one level further into
//! the full execution trace. Most of the time the user stops at layer
//! one, but the drill-down must always be available — that is where
//! trust comes from.
//!
//! Two structural guarantees make this safe to render, not just a
//! formatting convention:
//!
//! * **No leakage downward.** The conclusion layer is built from the
//!   conclusion and the primary action only. Evidence details and trace
//!   steps can never appear on the first screen, whatever the caller
//!   stuffed into the deeper layers.
//! * **Monotone deepening.** Each deeper layer is rendered as the
//!   previous layer's text plus one more section, so drilling down
//!   never changes or contradicts what the user already read — it only
//!   adds. [`ExecutionResult::render`] is implemented by appending for
//!   exactly this reason.
//!
//! Evidence is also checked, not just displayed: every evidence item
//! must name its source, and any trace step it cites must exist
//! ([`ExecutionResult::validate`]). An evidence item with no source is
//! not evidence, and a conclusion with no evidence at all renders an
//! explicit "no evidence recorded" marker instead of silently looking
//! as solid as a sourced one.

/// The three disclosure layers, shallow to deep.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DisclosureLevel {
    /// Conclusion + the single most important action. First screen only.
    Conclusion,
    /// Conclusion + the evidence the conclusion relied on.
    Evidence,
    /// Evidence + the full execution trace, step by step.
    Trace,
}

impl DisclosureLevel {
    pub fn name(&self) -> &'static str {
        match self {
            DisclosureLevel::Conclusion => "结论",
            DisclosureLevel::Evidence => "证据",
            DisclosureLevel::Trace => "完整轨迹",
        }
    }
}

/// What kind of thing an evidence item cites.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceKind {
    /// A file (or a section of one) the conclusion read.
    File,
    /// A memory fact from the blackboard / memory store.
    Memory,
    /// The result of a tool call.
    ToolResult,
    /// A policy, rule or quota the conclusion obeyed.
    Policy,
}

impl EvidenceKind {
    pub fn label(&self) -> &'static str {
        match self {
            EvidenceKind::File => "文件",
            EvidenceKind::Memory => "记忆",
            EvidenceKind::ToolResult => "工具结果",
            EvidenceKind::Policy => "规则",
        }
    }
}

/// One piece of evidence behind the conclusion: what was relied on, and
/// where it came from. `trace_refs` cites the trace steps (by `seq`)
/// that produced or checked this item, so a user drilling from evidence
/// into the trace lands on the exact steps.
#[derive(Debug, Clone)]
pub struct EvidenceItem {
    pub kind: EvidenceKind,
    /// Where the evidence came from, e.g. `差旅政策.md §3` or
    /// `memory:fact#42`. Must be non-empty: an unnamed source is not
    /// evidence.
    pub source: String,
    /// What was taken from the source, in one line.
    pub detail: String,
    /// Trace step sequence numbers backing this item.
    pub trace_refs: Vec<usize>,
}

impl EvidenceItem {
    pub fn new(
        kind: EvidenceKind,
        source: impl Into<String>,
        detail: impl Into<String>,
    ) -> Self {
        EvidenceItem { kind, source: source.into(), detail: detail.into(), trace_refs: Vec::new() }
    }

    pub fn citing(mut self, trace_seqs: &[usize]) -> Self {
        self.trace_refs = trace_seqs.to_vec();
        self
    }
}

/// What kind of work one trace step did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceStepKind {
    Intent,
    Plan,
    ToolCall,
    ToolResult,
    Check,
    SideEffect,
}

impl TraceStepKind {
    pub fn label(&self) -> &'static str {
        match self {
            TraceStepKind::Intent => "意图",
            TraceStepKind::Plan => "计划",
            TraceStepKind::ToolCall => "工具调用",
            TraceStepKind::ToolResult => "工具结果",
            TraceStepKind::Check => "校验",
            TraceStepKind::SideEffect => "副作用",
        }
    }
}

/// One step of the full execution trace. `seq` numbers the steps in
/// execution order starting at 1; rendering sorts by `seq`, so steps
/// recorded out of order still read in order.
#[derive(Debug, Clone)]
pub struct TraceStep {
    pub seq: usize,
    pub kind: TraceStepKind,
    /// Short label, e.g. `calendar.create`.
    pub label: String,
    /// One line of detail, e.g. the arguments or the outcome.
    pub detail: String,
}

impl TraceStep {
    pub fn new(
        seq: usize,
        kind: TraceStepKind,
        label: impl Into<String>,
        detail: impl Into<String>,
    ) -> Self {
        TraceStep { seq, kind, label: label.into(), detail: detail.into() }
    }
}

/// An execution result in three layers. The layers are stored together
/// because they describe one run, but they are rendered separately:
/// [`ExecutionResult::render`] only ever reads the layers at or above
/// the requested depth, which is what keeps the first screen clean.
#[derive(Debug, Clone, Default)]
pub struct ExecutionResult {
    /// The one-line outcome. Must be non-empty.
    pub conclusion: String,
    /// The single most important next action, shown on the first screen
    /// next to the conclusion. `None` means the task is fully done and
    /// needs nothing from the user.
    pub primary_action: Option<String>,
    pub evidence: Vec<EvidenceItem>,
    pub trace: Vec<TraceStep>,
}

impl ExecutionResult {
    pub fn new(conclusion: impl Into<String>, primary_action: Option<String>) -> Self {
        ExecutionResult {
            conclusion: conclusion.into(),
            primary_action,
            evidence: Vec::new(),
            trace: Vec::new(),
        }
    }

    pub fn with_evidence(mut self, items: Vec<EvidenceItem>) -> Self {
        self.evidence = items;
        self
    }

    pub fn with_trace(mut self, steps: Vec<TraceStep>) -> Self {
        self.trace = steps;
        self
    }

    /// Structural checks on the result. Returns one message per problem;
    /// an empty vec means the result is well-formed. Rendering never
    /// calls this — a malformed result still renders, with its gaps
    /// marked — but a caller that builds results from live runs should
    /// treat any error here as a bug in the run, not in the renderer.
    pub fn validate(&self) -> Vec<String> {
        let mut problems = Vec::new();
        if self.conclusion.trim().is_empty() {
            problems.push("conclusion is empty".to_string());
        }
        for (i, item) in self.evidence.iter().enumerate() {
            if item.source.trim().is_empty() {
                problems.push(format!("evidence[{i}] has no source"));
            }
            if item.detail.trim().is_empty() {
                problems.push(format!("evidence[{i}] has no detail"));
            }
            for seq in &item.trace_refs {
                if !self.trace.iter().any(|s| s.seq == *seq) {
                    problems.push(format!(
                        "evidence[{i}] cites trace step #{seq}, which does not exist"
                    ));
                }
            }
        }
        let mut seqs: Vec<usize> = self.trace.iter().map(|s| s.seq).collect();
        seqs.sort_unstable();
        for w in seqs.windows(2) {
            if w[0] == w[1] {
                problems.push(format!("trace step #{} appears twice", w[0]));
            }
        }
        if let Some(first) = seqs.first() {
            if *first != 1 {
                problems.push(format!("trace starts at step #{first}, not #1"));
            }
        }
        problems
    }

    /// Render the result at the requested depth. Deeper renders are
    /// built by appending to the shallower text, so drilling down never
    /// rewrites what the user already read.
    pub fn render(&self, level: DisclosureLevel) -> String {
        let mut out = String::new();
        out.push_str(&format!("结论：{}\n", self.conclusion));
        match &self.primary_action {
            Some(action) => out.push_str(&format!("下一步：{action}\n")),
            None => out.push_str("下一步：（无后续操作）\n"),
        }
        if level == DisclosureLevel::Conclusion {
            return out;
        }

        out.push_str(&format!("\n证据（{} 条）：\n", self.evidence.len()));
        if self.evidence.is_empty() {
            out.push_str("（未记录证据——此结论暂无可核查依据）\n");
        }
        for (i, item) in self.evidence.iter().enumerate() {
            out.push_str(&format!(
                "{}. [{}] {} —— {}\n",
                i + 1,
                item.kind.label(),
                item.source,
                item.detail
            ));
            if !item.trace_refs.is_empty() {
                let refs: Vec<String> =
                    item.trace_refs.iter().map(|s| format!("#{s}")).collect();
                out.push_str(&format!("    依据轨迹步骤：{}\n", refs.join("、")));
            }
        }
        if level == DisclosureLevel::Evidence {
            return out;
        }

        let mut steps: Vec<&TraceStep> = self.trace.iter().collect();
        steps.sort_by_key(|s| s.seq);
        out.push_str(&format!("\n完整轨迹（{} 步）：\n", steps.len()));
        if steps.is_empty() {
            out.push_str("（未记录轨迹）\n");
        }
        for step in steps {
            out.push_str(&format!(
                "{}. [{}] {} —— {}\n",
                step.seq,
                step.kind.label(),
                step.label,
                step.detail
            ));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> ExecutionResult {
        ExecutionResult::new(
            "已为你整理好下周上海出差的报销单，共 3 笔合计 ¥1,840。",
            Some("确认后提交报销".to_string()),
        )
        .with_evidence(vec![
            EvidenceItem::new(EvidenceKind::File, "发票/2026-10-28.pdf", "机票 ¥620，CA1831")
                .citing(&[3]),
            EvidenceItem::new(EvidenceKind::Memory, "memory:fact#42", "公司差旅标准：机票上限 ¥800"),
            EvidenceItem::new(EvidenceKind::ToolResult, "calendar.list", "下周三无冲突会议"),
        ])
        .with_trace(vec![
            TraceStep::new(1, TraceStepKind::Intent, "整理报销单", "用户：把上个月的发票整理成报销单"),
            TraceStep::new(2, TraceStepKind::Plan, "三步计划", "读发票 → 查标准 → 填表"),
            TraceStep::new(3, TraceStepKind::ToolCall, "ocr.read", "发票/2026-10-28.pdf"),
            TraceStep::new(4, TraceStepKind::Check, "限额校验", "¥620 ≤ ¥800，通过"),
            TraceStep::new(5, TraceStepKind::SideEffect, "form.fill", "报销单已填好，未提交"),
        ])
    }

    #[test]
    fn conclusion_layer_shows_only_conclusion_and_action() {
        let r = sample();
        let text = r.render(DisclosureLevel::Conclusion);
        assert!(text.contains("已为你整理好下周上海出差的报销单"));
        assert!(text.contains("确认后提交报销"));
        // No leakage downward: no evidence source, no evidence detail,
        // no trace label may appear on the first screen.
        for leaked in ["发票/2026-10-28.pdf", "机票 ¥620", "memory:fact#42", "ocr.read", "限额校验"] {
            assert!(!text.contains(leaked), "first screen leaked: {leaked}");
        }
    }

    #[test]
    fn deeper_layers_only_add_never_rewrite() {
        let r = sample();
        let c = r.render(DisclosureLevel::Conclusion);
        let e = r.render(DisclosureLevel::Evidence);
        let t = r.render(DisclosureLevel::Trace);
        assert!(e.starts_with(&c), "evidence layer must keep the conclusion text verbatim");
        assert!(t.starts_with(&e), "trace layer must keep the evidence text verbatim");
    }

    #[test]
    fn evidence_layer_lists_sources_with_kinds_and_trace_refs() {
        let r = sample();
        let text = r.render(DisclosureLevel::Evidence);
        assert!(text.contains("证据（3 条）"));
        assert!(text.contains("[文件] 发票/2026-10-28.pdf"));
        assert!(text.contains("[记忆] memory:fact#42"));
        assert!(text.contains("[工具结果] calendar.list"));
        assert!(text.contains("依据轨迹步骤：#3"));
        // Evidence must not pull the trace section in with it.
        assert!(!text.contains("完整轨迹"));
        assert!(!text.contains("ocr.read"));
    }

    #[test]
    fn trace_layer_renders_steps_sorted_by_seq() {
        let mut r = sample();
        r.trace.reverse(); // recorded out of order; render must still read in order
        let text = r.render(DisclosureLevel::Trace);
        let pos = |needle: &str| text.find(needle).unwrap();
        assert!(pos("1. [意图]") < pos("2. [计划]"));
        assert!(pos("2. [计划]") < pos("3. [工具调用]"));
        assert!(pos("4. [校验]") < pos("5. [副作用]"));
        assert!(text.contains("完整轨迹（5 步）"));
    }

    #[test]
    fn no_evidence_is_marked_not_hidden() {
        let r = ExecutionResult::new("已完成。", None);
        let text = r.render(DisclosureLevel::Evidence);
        assert!(text.contains("未记录证据"));
        assert!(text.contains("下一步：（无后续操作）"));
        assert!(r.render(DisclosureLevel::Trace).contains("未记录轨迹"));
    }

    #[test]
    fn validate_accepts_a_well_formed_result() {
        assert!(sample().validate().is_empty());
    }

    #[test]
    fn validate_flags_empty_conclusion_and_sourceless_evidence() {
        let r = ExecutionResult::new("   ", None).with_evidence(vec![
            EvidenceItem::new(EvidenceKind::File, "", "没有出处"),
            EvidenceItem::new(EvidenceKind::Memory, "memory:fact#1", ""),
        ]);
        let problems = r.validate();
        assert!(problems.iter().any(|p| p.contains("conclusion is empty")));
        assert!(problems.iter().any(|p| p.contains("evidence[0] has no source")));
        assert!(problems.iter().any(|p| p.contains("evidence[1] has no detail")));
    }

    #[test]
    fn validate_flags_dangling_trace_refs_and_duplicate_steps() {
        let r = ExecutionResult::new("已完成。", None)
            .with_evidence(vec![
                EvidenceItem::new(EvidenceKind::ToolResult, "tool.x", "结果").citing(&[9]),
            ])
            .with_trace(vec![
                TraceStep::new(1, TraceStepKind::Intent, "a", "a"),
                TraceStep::new(1, TraceStepKind::Plan, "b", "b"),
            ]);
        let problems = r.validate();
        assert!(problems.iter().any(|p| p.contains("cites trace step #9")));
        assert!(problems.iter().any(|p| p.contains("step #1 appears twice")));
    }

    #[test]
    fn validate_flags_trace_not_starting_at_one() {
        let r = ExecutionResult::new("已完成。", None)
            .with_trace(vec![TraceStep::new(2, TraceStepKind::Intent, "a", "a")]);
        assert!(r.validate().iter().any(|p| p.contains("starts at step #2")));
    }

    #[test]
    fn level_names_are_chinese() {
        assert_eq!(DisclosureLevel::Conclusion.name(), "结论");
        assert_eq!(DisclosureLevel::Evidence.name(), "证据");
        assert_eq!(DisclosureLevel::Trace.name(), "完整轨迹");
    }
}
