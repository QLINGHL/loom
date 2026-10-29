//! Macro ecology management (doc ch. 21, open problem 2).
//!
//! Chapter 4's only anti-ossification rule is coarse — "N consecutive
//! precondition failures demote the macro". That says nothing about the
//! opposite failure mode: a macro that never fails because it never runs.
//! Left alone, a macro library grows monotonically while the user's habits
//! move on, and near-duplicate macros accumulate for intents that drifted
//! apart only in wording. This module answers the chapter's question —
//! *which macros to keep, which to retire, which to merge* — with three
//! deliberately asymmetric rules:
//!
//! * **Retire** is automatic but conservative: only a macro whose last
//!   successful run is provably older than `idle_ttl_ms` is pruned, the
//!   boundary is "exceeds" (exactly `idle_ttl_ms` idle is still alive,
//!   matching the delegation lease boundary), a macro with an *enabled*
//!   trigger is never pruned (its automation authorization is explicit
//!   and outlives idleness), and a macro whose age is unknown — the
//!   engine was built without a clock — is never pruned. A failed macro
//!   attempt does not refresh idleness (see `Macro::last_used_ms`).
//! * **Merge is report-only, never automatic.** Merging changes a
//!   user-confirmed macro; the report names a survivor (higher stable
//!   runs, then more recent use, then the lexicographically smaller
//!   intent key, so the choice is deterministic) and a similarity score,
//!   but the merge itself stays a human decision.
//! * **Keep is the default.** Anything the rules cannot prove idle or
//!   duplicative stays, and the report says why each macro was kept out
//!   of the prune set (unknown age, armed trigger) instead of silently
//!   omitting it.
//!
//! Similarity is structural, not textual: tool-name sets, slot sets and
//! frozen (non-slot) template leaves, combined 0.4 / 0.3 / 0.3. Two
//! macros for different intents with the same tool sequence, slots and
//! frozen arguments score 1.0; changing only a frozen argument still
//! scores high enough to surface, because that is exactly the drift a
//! human should look at before merging.

use crate::determinization::{DetEngine, Macro};
use std::collections::HashSet;

/// Idle time after which a macro becomes a prune candidate.
/// Default: 30 days — long enough that a seasonal habit (e.g. a yearly
/// trip macro) is not pruned for merely being monthly.
pub const DEFAULT_IDLE_TTL_MS: u64 = 30 * 24 * 60 * 60 * 1000;
/// Similarity at or above which two macros are reported as merge candidates.
pub const DEFAULT_MERGE_THRESHOLD: f64 = 0.70;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EcologyPolicy {
    pub idle_ttl_ms: u64,
    pub merge_threshold: f64,
}

impl Default for EcologyPolicy {
    fn default() -> Self {
        EcologyPolicy { idle_ttl_ms: DEFAULT_IDLE_TTL_MS, merge_threshold: DEFAULT_MERGE_THRESHOLD }
    }
}

impl EcologyPolicy {
    pub fn with_idle_ttl(idle_ttl_ms: u64) -> Self {
        EcologyPolicy { idle_ttl_ms, ..EcologyPolicy::default() }
    }
}

/// A macro provably idle past the policy TTL, without an enabled trigger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IdleCandidate {
    pub intent_key: String,
    pub idle_ms: u64,
    pub last_used_ms: u64,
    pub stable_runs: u32,
}

/// Two macros similar enough that a human should consider merging them.
/// This is a report, not an action: nothing here merges anything.
#[derive(Debug, Clone, PartialEq)]
pub struct MergeCandidate {
    pub first: String,
    pub second: String,
    pub similarity: f64,
    /// Deterministic survivor suggestion; the merge itself is human work.
    pub survivor: String,
}

/// A macro actually removed by [`sweep_idle`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetiredMacro {
    pub intent_key: String,
    pub idle_ms: u64,
    pub stable_runs: u32,
    /// Whether an (already exhausted/disabled) trigger entry was removed
    /// with the macro. Enabled triggers block pruning, so a retired macro
    /// never had an enabled trigger.
    pub trigger_removed: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EcologyReport {
    pub total_macros: usize,
    pub idle_candidates: Vec<IdleCandidate>,
    pub merge_candidates: Vec<MergeCandidate>,
    /// Macros skipped for pruning because their age is unknown (no clock).
    pub unknown_age: Vec<String>,
    /// Macros skipped for pruning despite idleness: an enabled trigger.
    pub kept_with_trigger: Vec<String>,
}

fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let inter = a.intersection(b).count() as f64;
    let union = a.union(b).count() as f64;
    inter / union
}

fn tool_names(mac: &Macro) -> HashSet<String> {
    mac.template.steps.iter().map(|s| s.tool.clone()).collect()
}

fn slot_set(mac: &Macro) -> HashSet<String> {
    mac.slots.iter().map(|(i, p)| format!("{i}:{p}")).collect()
}

/// Frozen leaves of the template: slot positions hold `Value::Null`
/// placeholders and are excluded, so only arguments the macro froze at
/// compile time count here.
fn frozen_leaves(mac: &Macro) -> HashSet<String> {
    let mut out = HashSet::new();
    let slots: HashSet<(usize, String)> = mac.slots.iter().cloned().collect();
    for (i, step) in mac.template.steps.iter().enumerate() {
        for path in step.args.leaf_paths() {
            if slots.contains(&(i, path.clone())) {
                continue;
            }
            let split: Vec<&str> = path.split('.').collect();
            if let Some(v) = step.args.get_path(&split) {
                out.insert(format!("{i}:{path}={}", v.canonical()));
            }
        }
    }
    out
}

/// Structural similarity in [0, 1]: 0.4 tools + 0.3 slots + 0.3 frozen
/// arguments. Intent keys are deliberately ignored — two intents that
/// compile to the same script are the merge candidates that matter.
pub fn macro_similarity(a: &Macro, b: &Macro) -> f64 {
    0.4 * jaccard(&tool_names(a), &tool_names(b))
        + 0.3 * jaccard(&slot_set(a), &slot_set(b))
        + 0.3 * jaccard(&frozen_leaves(a), &frozen_leaves(b))
}

fn survivor_of(a: &Macro, b: &Macro) -> String {
    if a.stable_runs != b.stable_runs {
        return if a.stable_runs > b.stable_runs { a.intent_key.clone() } else { b.intent_key.clone() };
    }
    match (a.last_used_ms, b.last_used_ms) {
        (Some(x), Some(y)) if x != y => {
            if x > y { a.intent_key.clone() } else { b.intent_key.clone() }
        }
        _ => a.intent_key.clone().min(b.intent_key.clone()),
    }
}

/// Read-only ecology report. Never removes or merges anything.
pub fn report(engine: &DetEngine, now_ms: u64, policy: EcologyPolicy) -> EcologyReport {
    let macros = engine.macros_snapshot();
    let mut idle_candidates = Vec::new();
    let mut unknown_age = Vec::new();
    let mut kept_with_trigger = Vec::new();
    for mac in &macros {
        match mac.last_used_ms {
            None => unknown_age.push(mac.intent_key.clone()),
            Some(last) => {
                let idle = now_ms.saturating_sub(last);
                if idle > policy.idle_ttl_ms {
                    let trigger_armed = engine
                        .trigger_for(&mac.intent_key)
                        .map(|t| t.enabled)
                        .unwrap_or(false);
                    if trigger_armed {
                        kept_with_trigger.push(mac.intent_key.clone());
                    } else {
                        idle_candidates.push(IdleCandidate {
                            intent_key: mac.intent_key.clone(),
                            idle_ms: idle,
                            last_used_ms: last,
                            stable_runs: mac.stable_runs,
                        });
                    }
                }
            }
        }
    }
    idle_candidates.sort_by(|a, b| b.idle_ms.cmp(&a.idle_ms).then(a.intent_key.cmp(&b.intent_key)));

    let mut merge_candidates = Vec::new();
    for i in 0..macros.len() {
        for j in (i + 1)..macros.len() {
            let similarity = macro_similarity(&macros[i], &macros[j]);
            if similarity >= policy.merge_threshold {
                merge_candidates.push(MergeCandidate {
                    first: macros[i].intent_key.clone(),
                    second: macros[j].intent_key.clone(),
                    similarity,
                    survivor: survivor_of(&macros[i], &macros[j]),
                });
            }
        }
    }
    merge_candidates.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.first.cmp(&b.first))
            .then(a.second.cmp(&b.second))
    });

    EcologyReport {
        total_macros: macros.len(),
        idle_candidates,
        merge_candidates,
        unknown_age,
        kept_with_trigger,
    }
}

/// Remove every idle candidate from the engine and return what was
/// retired. Merge candidates are reported only — this never merges.
pub fn sweep_idle(engine: &mut DetEngine, now_ms: u64, policy: EcologyPolicy) -> Vec<RetiredMacro> {
    let rep = report(engine, now_ms, policy);
    let mut retired = Vec::new();
    for cand in rep.idle_candidates {
        let trigger_removed = engine.trigger_for(&cand.intent_key).is_some();
        if let Some(mac) = engine.remove_macro(&cand.intent_key) {
            retired.push(RetiredMacro {
                intent_key: mac.intent_key,
                idle_ms: cand.idle_ms,
                stable_runs: mac.stable_runs,
                trigger_removed,
            });
        }
    }
    retired
}

/// One-line-per-section text rendering for logs and demos.
pub fn render_report(rep: &EcologyReport) -> String {
    let mut out = vec![format!("宏生态位：共 {} 个宏", rep.total_macros)];
    if rep.idle_candidates.is_empty() {
        out.push("淘汰候选：无".to_string());
    } else {
        for c in &rep.idle_candidates {
            out.push(format!(
                "淘汰候选：{}（闲置 {} ms，稳定运行 {} 次）",
                c.intent_key, c.idle_ms, c.stable_runs
            ));
        }
    }
    if rep.merge_candidates.is_empty() {
        out.push("合并候选：无".to_string());
    } else {
        for m in &rep.merge_candidates {
            out.push(format!(
                "合并候选：{} + {}（相似度 {:.2}，建议保留 {}，需人工确认）",
                m.first, m.second, m.similarity, m.survivor
            ));
        }
    }
    if !rep.unknown_age.is_empty() {
        out.push(format!("年龄未知（不淘汰）：{}", rep.unknown_age.join(", ")));
    }
    if !rep.kept_with_trigger.is_empty() {
        out.push(format!("触发器已启用（不淘汰）：{}", rep.kept_with_trigger.join(", ")));
    }
    out.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::determinization::{slot_key, TriggerBudget};
    use crate::taint::TrustLabel;
    use crate::typesys::{Condition, Plan, PlanStep, Predicate};
    use crate::ManualClock;
    use std::collections::HashMap;
    use std::rc::Rc;

    const DAY: u64 = 24 * 60 * 60 * 1000;

    fn trip_plan(date: &str) -> Plan {
        let mut p = Plan::new(format!("plan-{date}"), "trip.arrange", TrustLabel::User);
        p.steps = vec![
            PlanStep::new(
                "flight.search",
                "flight",
                crate::Value::map(vec![
                    ("to", crate::Value::text("Shanghai")),
                    ("date", crate::Value::text(date)),
                ]),
            ),
            PlanStep::new(
                "cal.write",
                "calendar",
                crate::Value::map(vec![
                    ("title", crate::Value::text("Shanghai trip")),
                    ("date", crate::Value::text(date)),
                ]),
            ),
        ];
        p.pre = vec![Condition::new("slot free", Predicate::SlotFree("cal:slot".into()))];
        p
    }

    fn mail_plan(date: &str) -> Plan {
        let mut p = Plan::new(format!("plan-{date}"), "mail.archive", TrustLabel::User);
        p.steps = vec![PlanStep::new(
            "mail.archive",
            "mail",
            crate::Value::map(vec![("folder", crate::Value::text("invoices")), ("date", crate::Value::text(date))]),
        )];
        p.pre = vec![];
        p
    }

    fn versions() -> HashMap<String, u32> {
        HashMap::from([("flight.search".to_string(), 1), ("cal.write".to_string(), 1), ("mail.archive".to_string(), 1)])
    }

    fn free_facts() -> crate::typesys::FactBase {
        crate::typesys::FactBase::new().with_slot_free("cal:slot")
    }

    fn compile_trip(engine: &mut DetEngine) {
        for i in 1..=5 {
            engine.record_success(&trip_plan(&format!("2026-11-0{i}")), &versions());
        }
        engine.compile_macro("trip.arrange", true).unwrap();
    }

    fn compile_mail(engine: &mut DetEngine) {
        for i in 1..=5 {
            engine.record_success(&mail_plan(&format!("2026-11-0{i}")), &versions());
        }
        engine.compile_macro("mail.archive", true).unwrap();
    }

    fn slot_values(date: &str) -> HashMap<String, crate::Value> {
        HashMap::from([
            (slot_key(0, "date"), crate::Value::text(date)),
            (slot_key(1, "date"), crate::Value::text(date)),
        ])
    }

    fn policy() -> EcologyPolicy {
        EcologyPolicy::with_idle_ttl(30 * DAY)
    }

    #[test]
    fn idle_past_ttl_is_pruned_exact_ttl_is_kept() {
        let clock = Rc::new(ManualClock::new(1_000_000));
        let mut engine = DetEngine::with_clock(clock.clone());
        compile_trip(&mut engine);
        // Exactly at the TTL: still alive (boundary is "exceeds").
        let rep = report(&engine, 1_000_000 + 30 * DAY, policy());
        assert!(rep.idle_candidates.is_empty(), "exactly TTL must not prune");
        // One ms past: candidate, and sweep removes it.
        let retired = sweep_idle(&mut engine, 1_000_000 + 30 * DAY + 1, policy());
        assert_eq!(retired.len(), 1);
        assert_eq!(retired[0].intent_key, "trip.arrange");
        assert!(engine.macro_for("trip.arrange").is_none());
    }

    #[test]
    fn successful_run_refreshes_idleness_failed_run_does_not() {
        let clock = Rc::new(ManualClock::new(1_000_000));
        let mut engine = DetEngine::with_clock(clock.clone());
        compile_trip(&mut engine);
        clock.advance(20 * DAY);
        // A failed attempt (slot busy) must not count as a hit.
        let busy = crate::typesys::FactBase::new();
        assert!(engine.run_macro("trip.arrange", &slot_values("2026-12-24"), &busy, &versions()).is_err());
        let rep = report(&engine, 1_000_000 + 31 * DAY, policy());
        assert_eq!(rep.idle_candidates.len(), 1, "failed attempt must not refresh");
        // A successful run at day 31 resets the clock: idle again from there.
        clock.advance(11 * DAY);
        engine.run_macro("trip.arrange", &slot_values("2026-12-24"), &free_facts(), &versions()).unwrap();
        let rep = report(&engine, 1_000_000 + 31 * DAY + 30 * DAY, policy());
        assert!(rep.idle_candidates.is_empty(), "success at day 31, exactly TTL later");
        assert_eq!(report(&engine, 1_000_000 + 31 * DAY + 30 * DAY + 1, policy()).idle_candidates.len(), 1);
    }

    #[test]
    fn enabled_trigger_blocks_pruning() {
        let clock = Rc::new(ManualClock::new(1_000_000));
        let mut engine = DetEngine::with_clock(clock.clone());
        compile_trip(&mut engine);
        for _ in 0..20 {
            engine.run_macro("trip.arrange", &slot_values("2026-12-24"), &free_facts(), &versions()).unwrap();
        }
        engine
            .arm_trigger("trip.arrange", true, TriggerBudget { max_runs: 100, max_side_effects: 100 })
            .unwrap();
        let rep = report(&engine, 1_000_000 + 365 * DAY, policy());
        assert!(rep.idle_candidates.is_empty());
        assert_eq!(rep.kept_with_trigger, vec!["trip.arrange".to_string()]);
        assert!(sweep_idle(&mut engine, 1_000_000 + 365 * DAY, policy()).is_empty());
        assert!(engine.macro_for("trip.arrange").is_some());
    }

    #[test]
    fn unknown_age_macro_is_never_pruned() {
        let mut engine = DetEngine::new(); // no clock
        compile_trip(&mut engine);
        let rep = report(&engine, u64::MAX, policy());
        assert!(rep.idle_candidates.is_empty());
        assert_eq!(rep.unknown_age, vec!["trip.arrange".to_string()]);
        assert!(sweep_idle(&mut engine, u64::MAX, policy()).is_empty());
        assert!(engine.macro_for("trip.arrange").is_some());
    }

    #[test]
    fn identical_templates_are_merge_candidates_dissimilar_are_not() {
        let mut engine = DetEngine::new();
        compile_trip(&mut engine);
        compile_mail(&mut engine);
        let rep = report(&engine, 0, policy());
        assert!(rep.merge_candidates.is_empty(), "different tools must not merge: {rep:?}");

        // A second trip macro with the same template under another intent key.
        let mut engine2 = DetEngine::new();
        compile_trip(&mut engine2);
        for i in 1..=5 {
            let mut p = trip_plan(&format!("2026-11-0{i}"));
            p.intent_key = "trip.arrange.v2".to_string();
            p.id = format!("plan2-{i}");
            engine2.record_success(&p, &versions());
        }
        engine2.compile_macro("trip.arrange.v2", true).unwrap();
        let rep = report(&engine2, 0, policy());
        assert_eq!(rep.merge_candidates.len(), 1);
        let m = &rep.merge_candidates[0];
        assert!((m.similarity - 1.0).abs() < f64::EPSILON);
        // Equal stable runs, both unknown age -> lexicographically smaller survives.
        assert_eq!(m.survivor, "trip.arrange");
        // Merge is report-only: sweep must not remove either macro.
        sweep_idle(&mut engine2, 0, policy());
        assert!(engine2.macro_for("trip.arrange").is_some());
        assert!(engine2.macro_for("trip.arrange.v2").is_some());
    }

    #[test]
    fn survivor_prefers_more_stable_runs() {
        let mut engine = DetEngine::new();
        compile_trip(&mut engine);
        for i in 1..=5 {
            let mut p = trip_plan(&format!("2026-11-0{i}"));
            p.intent_key = "trip.arrange.v2".to_string();
            p.id = format!("plan2-{i}");
            engine.record_success(&p, &versions());
        }
        engine.compile_macro("trip.arrange.v2", true).unwrap();
        // v2 runs 3 times, original never runs after compile.
        for _ in 0..3 {
            engine.run_macro("trip.arrange.v2", &slot_values("2026-12-24"), &free_facts(), &versions()).unwrap();
        }
        let rep = report(&engine, 0, policy());
        assert_eq!(rep.merge_candidates[0].survivor, "trip.arrange.v2");
    }

    #[test]
    fn render_mentions_prune_merge_and_unknown_sections() {
        let mut engine = DetEngine::new();
        compile_trip(&mut engine);
        compile_mail(&mut engine);
        let text = render_report(&report(&engine, 0, policy()));
        assert!(text.contains("共 2 个宏"));
        assert!(text.contains("年龄未知"));
        assert!(text.contains("合并候选：无"));
    }
}
