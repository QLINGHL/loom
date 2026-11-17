//! Decision trace: a debugger for a probabilistic system (doc ch. 15,
//! "可重放").
//!
//! A traditional system can be debugged by replaying the same input — an AI
//! system cannot, because the same input may yield a different plan from the
//! model. So the system layer records a *decision trajectory*: at every
//! decision point, the input digest, the model version and parameters, the
//! output digest, the tool returns seen, and the capability set held at that
//! moment. With that record you can time-travel: go back to step 7 and see
//! what the model saw then, and why it picked that tool.
//!
//! The trace is also the substrate for compliance audit, responsibility
//! attribution, and model regression testing: re-run the recorded inputs
//! against a new model version and diff the decisions.
//!
//! Privacy by construction: only digests (SHA-256) of inputs, outputs and
//! tool results are stored — never the raw text. A digest proves *what* was
//! decided on and lets two runs be compared for equality, without the trace
//! itself becoming a second copy of the user's data. Callers supply a short
//! human-readable `summary` label (tool name, arg keys, outcome) so a human
//! can still read the trajectory.

use crate::crypto::{hex, sha256, sha256_hex};

/// What kind of decision this point records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionKind {
    /// The planner produced (or reused) a plan.
    Plan,
    /// A tool was invoked.
    ToolCall,
    /// A memory recall was consulted.
    Recall,
    /// The agent yielded to the user for a decision.
    Yield,
    /// A transaction was committed.
    Commit,
    /// A transaction was aborted / rolled back.
    Abort,
    /// A determinized macro ran with zero model calls.
    MacroRun,
}

impl DecisionKind {
    pub fn label(self) -> &'static str {
        match self {
            DecisionKind::Plan => "plan",
            DecisionKind::ToolCall => "tool_call",
            DecisionKind::Recall => "recall",
            DecisionKind::Yield => "yield",
            DecisionKind::Commit => "commit",
            DecisionKind::Abort => "abort",
            DecisionKind::MacroRun => "macro_run",
        }
    }
}

/// The model configuration behind one decision: version plus the sampling
/// parameters that make the same input produce different outputs.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelParams {
    pub version: String,
    pub temperature: f64,
    pub top_p: f64,
}

impl ModelParams {
    pub fn new(version: impl Into<String>, temperature: f64, top_p: f64) -> Self {
        ModelParams {
            version: version.into(),
            temperature,
            top_p,
        }
    }

    /// Canonical rendering used in the entry hash: the same model + params
    /// must always hash the same way.
    fn canonical(&self) -> String {
        format!("{}|{:.3}|{:.3}", self.version, self.temperature, self.top_p)
    }
}

/// One recorded decision point. `input_digest` / `output_digest` /
/// `tool_result_digest` are SHA-256 hex digests of the canonical inputs,
/// outputs and tool returns — digests, not the data itself (see module docs).
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionPoint {
    /// Monotonic sequence number inside this trace (0-based).
    pub seq: u64,
    /// Wall-clock time the decision was recorded, in ms.
    pub at_ms: u64,
    /// The agent that made the decision.
    pub agent: String,
    pub kind: DecisionKind,
    /// Plan step index this decision belongs to (0 for task-level points).
    pub step: u32,
    /// Short human-readable label, e.g. "cal.read 2026-11-24" — never raw
    /// user data, just enough to read the trajectory.
    pub summary: String,
    pub input_digest: String,
    pub model: ModelParams,
    pub output_digest: String,
    /// Tool invoked at this point, if any.
    pub tool: Option<String>,
    /// Digest of the tool's return value, if any.
    pub tool_result_digest: Option<String>,
    /// Capability ids held by the agent at this moment (sorted).
    pub capabilities: Vec<String>,
    /// Hash of the previous entry (all zeroes for the first entry).
    pub prev_hash: [u8; 32],
    /// Hash of this entry: sha256(prev_hash || canonical(point)).
    /// Chaining makes silent edits detectable — audit depends on it.
    pub entry_hash: [u8; 32],
}

impl DecisionPoint {
    /// Canonical bytes of everything that defines this point, for hashing.
    fn canonical(&self) -> String {
        let caps = self.capabilities.join(",");
        format!(
            "{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}",
            self.seq,
            self.at_ms,
            self.agent,
            self.kind.label(),
            self.step,
            self.summary,
            self.input_digest,
            self.model.canonical(),
            self.output_digest,
            self.tool.clone().unwrap_or_default(),
            self.tool_result_digest.clone().unwrap_or_default(),
            caps,
        )
    }

    fn compute_hash(&self) -> [u8; 32] {
        let mut bytes = self.prev_hash.to_vec();
        bytes.extend_from_slice(self.canonical().as_bytes());
        sha256(&bytes)
    }
}

/// What the model saw at one step: the decision point itself, plus the
/// tool returns it had already observed by then and the capabilities it
/// held. This is the "go back to step N" view.
#[derive(Debug, Clone, PartialEq)]
pub struct ReplayView {
    pub point: DecisionPoint,
    /// (tool, result digest) for every tool call recorded up to this point.
    pub tool_results_so_far: Vec<(String, String)>,
    /// Capability ids held at this point.
    pub capabilities: Vec<String>,
}

/// One step where two runs of the same input diverged — the unit of a
/// model regression report.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionDiff {
    pub seq: u64,
    pub input_digest: String,
    pub model_a: String,
    pub model_b: String,
    pub output_digest_a: String,
    pub output_digest_b: String,
}

/// Append-only decision trace with a tamper-evident hash chain.
#[derive(Debug, Default)]
pub struct DecisionTrace {
    points: Vec<DecisionPoint>,
}

impl DecisionTrace {
    pub fn new() -> Self {
        DecisionTrace { points: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Record one decision point and return its sequence number. Digests
    /// are computed here from the raw strings, so callers never handle
    /// hashes directly; the raw strings are dropped after hashing.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        at_ms: u64,
        agent: impl Into<String>,
        kind: DecisionKind,
        step: u32,
        summary: impl Into<String>,
        input: &str,
        model: ModelParams,
        output: &str,
        tool: Option<String>,
        tool_result: Option<&str>,
        capabilities: Vec<String>,
    ) -> u64 {
        let seq = self.points.len() as u64;
        let prev_hash = self
            .points
            .last()
            .map(|p| p.entry_hash)
            .unwrap_or([0u8; 32]);
        let mut caps = capabilities;
        caps.sort();
        caps.dedup();
        let mut point = DecisionPoint {
            seq,
            at_ms,
            agent: agent.into(),
            kind,
            step,
            summary: summary.into(),
            input_digest: sha256_hex(input.as_bytes()),
            model,
            output_digest: sha256_hex(output.as_bytes()),
            tool,
            tool_result_digest: tool_result.map(|r| sha256_hex(r.as_bytes())),
            capabilities: caps,
            prev_hash,
            entry_hash: [0u8; 32],
        };
        point.entry_hash = point.compute_hash();
        self.points.push(point);
        seq
    }

    pub fn point(&self, seq: u64) -> Option<&DecisionPoint> {
        self.points.get(seq as usize)
    }

    pub fn points(&self) -> &[DecisionPoint] {
        &self.points
    }

    pub fn points_for_agent(&self, agent: &str) -> Vec<&DecisionPoint> {
        self.points.iter().filter(|p| p.agent == agent).collect()
    }

    /// Recompute the hash chain from scratch. `false` means some entry was
    /// edited, reordered or dropped after recording.
    pub fn verify_chain(&self) -> bool {
        let mut prev = [0u8; 32];
        for p in &self.points {
            if p.prev_hash != prev {
                return false;
            }
            if p.entry_hash != p.compute_hash() {
                return false;
            }
            prev = p.entry_hash;
        }
        true
    }

    /// How many leading entries two traces share (identical entry hashes).
    /// A fork or a tampered prefix shortens this; it is how a copied trace
    /// is checked against the original without re-reading every decision.
    pub fn common_prefix_len(&self, other: &DecisionTrace) -> usize {
        self.points
            .iter()
            .zip(other.points.iter())
            .take_while(|(a, b)| a.entry_hash == b.entry_hash)
            .count()
    }

    /// Time-travel: reconstruct what the agent knew at step `seq` — the
    /// decision itself, the tool returns observed so far, and the
    /// capabilities held then. Later decisions are invisible by design.
    pub fn replay_at(&self, seq: u64) -> Option<ReplayView> {
        let point = self.point(seq)?.clone();
        let tool_results_so_far = self.points[..=(seq as usize)]
            .iter()
            .filter_map(|p| match (&p.tool, &p.tool_result_digest) {
                (Some(tool), Some(digest)) => Some((tool.clone(), digest.clone())),
                _ => None,
            })
            .collect();
        Some(ReplayView {
            capabilities: point.capabilities.clone(),
            point,
            tool_results_so_far,
        })
    }

    /// (input digest, output digest, model version) per decision — the
    /// regression fixture. Feed the inputs to a new model version and diff
    /// the outputs; the trace never needs to reveal the raw inputs.
    pub fn regression_pairs(&self) -> Vec<(String, String, String)> {
        self.points
            .iter()
            .map(|p| {
                (
                    p.input_digest.clone(),
                    p.output_digest.clone(),
                    p.model.version.clone(),
                )
            })
            .collect()
    }

    /// Compare this trace against another run of the same inputs (e.g. a
    /// newer model version). Returns one diff per step where the same input
    /// digest produced a different output digest, in trace order. Steps
    /// present in only one trace are ignored — this reports decision
    /// divergence, not coverage.
    pub fn diff_decisions(&self, other: &DecisionTrace) -> Vec<DecisionDiff> {
        let mut diffs = Vec::new();
        for a in &self.points {
            let Some(b) = other.points.iter().find(|p| p.input_digest == a.input_digest)
            else {
                continue;
            };
            if a.output_digest != b.output_digest {
                diffs.push(DecisionDiff {
                    seq: a.seq,
                    input_digest: a.input_digest.clone(),
                    model_a: a.model.version.clone(),
                    model_b: b.model.version.clone(),
                    output_digest_a: a.output_digest.clone(),
                    output_digest_b: b.output_digest.clone(),
                });
            }
        }
        diffs
    }

    /// One-line-per-step human-readable rendering of the trajectory.
    pub fn render(&self) -> String {
        let mut out = String::new();
        for p in &self.points {
            out.push_str(&format!(
                "#{} [{}] {} step={} {} in={}.. out={}..",
                p.seq,
                p.kind.label(),
                p.agent,
                p.step,
                p.summary,
                &p.input_digest[..8],
                &p.output_digest[..8],
            ));
            if let Some(tool) = &p.tool {
                out.push_str(&format!(" tool={tool}"));
            }
            out.push_str(&format!(" hash={}\n", &hex(&p.entry_hash)[..12]));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(version: &str) -> ModelParams {
        ModelParams::new(version, 0.7, 0.9)
    }

    fn record_trip(trace: &mut DecisionTrace, version: &str) {
        trace.record(
            1000,
            "planner",
            DecisionKind::Plan,
            0,
            "plan trip to Shanghai",
            "intent: shanghai trip next week",
            model(version),
            "plan: cal.read, flight.search, cal.write",
            None,
            None,
            vec!["cap:cal.read".into()],
        );
        trace.record(
            1010,
            "planner",
            DecisionKind::ToolCall,
            1,
            "cal.read 2026-11-24",
            "cal.read date=2026-11-24",
            model(version),
            "slot free 09:00-18:00",
            Some("cal.read".into()),
            Some("events: []"),
            vec!["cap:cal.read".into(), "cap:flight.search".into()],
        );
        trace.record(
            1020,
            "planner",
            DecisionKind::ToolCall,
            2,
            "flight.search PVG-SHA",
            "flight.search PVG-SHA 2026-11-24",
            model(version),
            "CA1831 09:20",
            Some("flight.search".into()),
            Some("CA1831 09:20 PVG-SHA 620"),
            vec!["cap:cal.read".into(), "cap:flight.search".into()],
        );
        trace.record(
            1030,
            "planner",
            DecisionKind::Commit,
            3,
            "write calendar event",
            "cal.write event 2026-11-24 09:20",
            model(version),
            "txn committed",
            Some("cal.write".into()),
            Some("event id 42"),
            vec!["cap:cal.read".into(), "cap:flight.search".into(), "cap:cal.write".into()],
        );
    }

    #[test]
    fn record_assigns_sequential_seqs_and_stores_digests() {
        let mut trace = DecisionTrace::new();
        let seq = trace.record(
            1000, "a", DecisionKind::Plan, 0, "s", "input", model("m1"), "output", None, None,
            vec![],
        );
        assert_eq!(seq, 0);
        let p = trace.point(0).unwrap();
        assert_eq!(p.input_digest, sha256_hex(b"input"));
        assert_eq!(p.output_digest, sha256_hex(b"output"));
        // Raw text never lands in the trace.
        assert!(!p.summary.contains("input"));
        assert_eq!(trace.len(), 1);
    }

    #[test]
    fn same_input_same_digest_different_input_different_digest() {
        let mut t1 = DecisionTrace::new();
        let mut t2 = DecisionTrace::new();
        t1.record(0, "a", DecisionKind::Plan, 0, "s", "same", model("m1"), "o", None, None, vec![]);
        t2.record(0, "a", DecisionKind::Plan, 0, "s", "same", model("m1"), "o", None, None, vec![]);
        assert_eq!(t1.point(0).unwrap().input_digest, t2.point(0).unwrap().input_digest);
        let mut t3 = DecisionTrace::new();
        t3.record(0, "a", DecisionKind::Plan, 0, "s", "other", model("m1"), "o", None, None, vec![]);
        assert_ne!(t1.point(0).unwrap().input_digest, t3.point(0).unwrap().input_digest);
    }

    #[test]
    fn chain_verifies_when_untampered() {
        let mut trace = DecisionTrace::new();
        record_trip(&mut trace, "m1");
        assert!(trace.verify_chain());
        assert_eq!(trace.len(), 4);
    }

    #[test]
    fn common_prefix_detects_a_forked_trace() {
        let mut a = DecisionTrace::new();
        record_trip(&mut a, "m1");
        let mut b = DecisionTrace::new();
        record_trip(&mut b, "m1");
        assert_eq!(a.common_prefix_len(&b), 4, "identical runs share the whole chain");
        // A run that diverges at step 2 shares only the first two entries.
        let mut c = DecisionTrace::new();
        c.record(1000, "planner", DecisionKind::Plan, 0, "plan trip to Shanghai",
            "intent: shanghai trip next week", model("m1"),
            "plan: cal.read, flight.search, cal.write", None, None,
            vec!["cap:cal.read".into()]);
        c.record(1010, "planner", DecisionKind::ToolCall, 1, "cal.read 2026-11-24",
            "cal.read date=2026-11-24", model("m1"), "slot free 09:00-18:00",
            Some("cal.read".into()), Some("events: []"),
            vec!["cap:cal.read".into(), "cap:flight.search".into()]);
        c.record(1025, "planner", DecisionKind::ToolCall, 2, "flight.search PVG-SHA",
            "flight.search PVG-SHA 2026-11-24", model("m1"), "MU5102 13:45",
            Some("flight.search".into()), Some("MU5102 13:45 PVG-SHA 540"),
            vec!["cap:cal.read".into(), "cap:flight.search".into()]);
        assert_eq!(a.common_prefix_len(&c), 2);
    }

    #[test]
    fn replay_at_shows_only_what_was_known_then() {
        let mut trace = DecisionTrace::new();
        record_trip(&mut trace, "m1");
        let view = trace.replay_at(1).unwrap();
        assert_eq!(view.point.seq, 1);
        assert_eq!(view.point.summary, "cal.read 2026-11-24");
        // Only the cal.read result had been observed by step 1.
        assert_eq!(view.tool_results_so_far.len(), 1);
        assert_eq!(view.tool_results_so_far[0].0, "cal.read");
        assert_eq!(
            view.tool_results_so_far[0].1,
            sha256_hex(b"events: []"),
            "tool result is identified by digest, not raw text"
        );
        // The write capability was not held yet at step 1.
        assert!(!view.capabilities.contains(&"cap:cal.write".to_string()));
        assert!(view.capabilities.contains(&"cap:cal.read".to_string()));
        // At the last step, everything is visible.
        let last = trace.replay_at(3).unwrap();
        assert_eq!(last.tool_results_so_far.len(), 3);
        assert!(last.capabilities.contains(&"cap:cal.write".to_string()));
        assert!(trace.replay_at(4).is_none());
    }

    #[test]
    fn regression_pairs_carry_model_version_for_re_runs() {
        let mut trace = DecisionTrace::new();
        record_trip(&mut trace, "m1");
        let pairs = trace.regression_pairs();
        assert_eq!(pairs.len(), 4);
        assert!(pairs.iter().all(|(_, _, v)| v == "m1"));
        assert_eq!(pairs[0].0, sha256_hex(b"intent: shanghai trip next week"));
    }

    #[test]
    fn diff_decisions_flags_only_changed_outputs_for_same_input() {
        let mut old = DecisionTrace::new();
        record_trip(&mut old, "m1");
        let mut new = DecisionTrace::new();
        // Same inputs, same outputs except the flight choice changes.
        new.record(1000, "planner", DecisionKind::Plan, 0, "plan trip to Shanghai",
            "intent: shanghai trip next week", model("m2"),
            "plan: cal.read, flight.search, cal.write", None, None,
            vec!["cap:cal.read".into()]);
        new.record(1010, "planner", DecisionKind::ToolCall, 1, "cal.read 2026-11-24",
            "cal.read date=2026-11-24", model("m2"), "slot free 09:00-18:00",
            Some("cal.read".into()), Some("events: []"),
            vec!["cap:cal.read".into(), "cap:flight.search".into()]);
        new.record(1020, "planner", DecisionKind::ToolCall, 2, "flight.search PVG-SHA",
            "flight.search PVG-SHA 2026-11-24", model("m2"), "MU5102 13:45",
            Some("flight.search".into()), Some("MU5102 13:45 PVG-SHA 540"),
            vec!["cap:cal.read".into(), "cap:flight.search".into()]);
        new.record(1030, "planner", DecisionKind::Commit, 3, "write calendar event",
            "cal.write event 2026-11-24 09:20", model("m2"), "txn committed",
            Some("cal.write".into()), Some("event id 42"),
            vec!["cap:cal.read".into(), "cap:flight.search".into(), "cap:cal.write".into()]);
        let diffs = old.diff_decisions(&new);
        assert_eq!(diffs.len(), 1, "only the flight choice diverged");
        assert_eq!(diffs[0].seq, 2);
        assert_eq!(diffs[0].model_a, "m1");
        assert_eq!(diffs[0].model_b, "m2");
        assert_eq!(diffs[0].input_digest, sha256_hex(b"flight.search PVG-SHA 2026-11-24"));
        // Identical rerun: no diffs at all.
        let mut same = DecisionTrace::new();
        record_trip(&mut same, "m2");
        assert!(old.diff_decisions(&same).is_empty());
    }

    #[test]
    fn empty_trace_verifies_and_replays_to_nothing() {
        let trace = DecisionTrace::new();
        assert!(trace.is_empty());
        assert!(trace.verify_chain());
        assert!(trace.replay_at(0).is_none());
        assert!(trace.regression_pairs().is_empty());
    }

    #[test]
    fn points_for_agent_filters_by_agent() {
        let mut trace = DecisionTrace::new();
        record_trip(&mut trace, "m1");
        trace.record(1040, "auditor", DecisionKind::Recall, 0, "audit recall",
            "recall trip", model("m1"), "2 items", None, None, vec![]);
        assert_eq!(trace.points_for_agent("planner").len(), 4);
        assert_eq!(trace.points_for_agent("auditor").len(), 1);
        assert!(trace.points_for_agent("nobody").is_empty());
    }

    #[test]
    fn capabilities_are_sorted_and_deduped() {
        let mut trace = DecisionTrace::new();
        trace.record(0, "a", DecisionKind::Plan, 0, "s", "i", model("m1"), "o", None, None,
            vec!["cap:b".into(), "cap:a".into(), "cap:b".into()]);
        assert_eq!(trace.point(0).unwrap().capabilities, vec!["cap:a", "cap:b"]);
    }

    #[test]
    fn render_shows_one_line_per_step_with_digests() {
        let mut trace = DecisionTrace::new();
        record_trip(&mut trace, "m1");
        let text = trace.render();
        assert_eq!(text.lines().count(), 4);
        assert!(text.contains("#2 [tool_call] planner step=2 flight.search PVG-SHA"));
        assert!(text.contains("tool=flight.search"));
        assert!(!text.contains("intent: shanghai trip next week"), "no raw input in render");
    }
}
