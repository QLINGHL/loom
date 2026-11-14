//! P0 acceptance suite (doc ch. 23): the three things the doc says to
//! validate first, before any more capability work —
//!
//! 1. **Does the determinization rate actually climb?** If it does not,
//!    the whole ch. 4 thesis is wrong. A scripted workload of repeated
//!    intents runs through [`DetEngine`]; the rate of intents served by the
//!    L1 plan cache or an L2 macro is measured per window and must climb
//!    toward the ch. 22 target (80%).
//! 2. **Can the false-interrupt rate stay under 5%?** If it cannot, users
//!    switch every confirmation off. A scripted interruption stream runs
//!    through the ch. 14 [`AttentionScheduler`]; the false rate among
//!    delivered interruptions is compared against the same stream with no
//!    scheduler at all (the control).
//! 3. **How many bad plans does the type checker stop?** A scripted mix
//!    of good and bad plans runs through the ch. 7 [`TypeChecker`]; every
//!    bad plan must be rejected at compile time (target: > 99.9%) and no
//!    good plan may be blocked.
//!
//! The suite is scripted and deterministic: no model, no clock, no I/O.
//! [`run_p0_suite`] runs all three trials with the default workloads and
//! returns one [`P0Report`] with a single pass/fail verdict.

use crate::attention::{AttentionDecision, AttentionScheduler, InterruptRequest, Priority};
use crate::determinization::{slot_key, DetEngine};
use crate::taint::TrustLabel;
use crate::typesys::{Effect, FactBase, Plan, PlanStep, TypeChecker};
use crate::Value;
use std::collections::{BTreeSet, HashMap};

/// Ch. 22 target: share of intents served by a deterministic path within a year.
pub const DETERMINIZATION_TARGET: f64 = 0.80;
/// Ch. 22 target: interruptions the user judges not worth it, over all interruptions.
pub const FALSE_INTERRUPT_TARGET: f64 = 0.05;
/// Ch. 22 target: privilege-escalation attempts intercepted by the kernel.
pub const INTERCEPTION_TARGET: f64 = 0.999;

// ---------------------------------------------------------------------------
// 1. Determinization climb
// ---------------------------------------------------------------------------

/// How one intent occurrence was served.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServedBy {
    /// Full probabilistic path: the model planned it.
    Model,
    /// L1 plan cache hit: same structure, parameters re-validated downstream.
    Cache,
    /// L2 macro: zero model calls by construction.
    Macro,
}

impl ServedBy {
    fn is_deterministic(self) -> bool {
        !matches!(self, ServedBy::Model)
    }
}

/// One intent kind's plan for the climb trial: two steps whose structure is
/// fixed across repetitions while one leaf (`date`) varies, so the
/// determinization engine must discover it as a macro slot.
fn climb_plan(kind: usize, rep: u32) -> Plan {
    let mut plan =
        Plan::new(format!("trial-{kind}-{rep}"), format!("trial.intent.{kind}"), TrustLabel::User);
    plan.steps = vec![
        PlanStep::new(
            "mail.search",
            "inbox",
            Value::map(vec![("query", Value::text("inbox"))]),
        ),
        PlanStep::new(
            "file.write",
            format!("note-{kind}.txt"),
            Value::map(vec![
                ("path", Value::text(format!("note-{kind}.txt"))),
                ("date", Value::int(i64::from(rep))),
            ]),
        ),
    ];
    plan
}

#[derive(Debug, Clone)]
pub struct ClimbReport {
    /// Deterministic share per window, in run order.
    pub windows: Vec<f64>,
    pub total: u64,
    pub deterministic: u64,
    pub final_rate: f64,
    /// True when the last window beats the first: the rate climbs.
    pub climbed: bool,
    pub meets_target: bool,
}

impl ClimbReport {
    pub fn render(&self) -> String {
        let verdict = if self.meets_target && self.climbed { "PASS" } else { "FAIL" };
        let windows: Vec<String> =
            self.windows.iter().map(|w| format!("{:.0}%", w * 100.0)).collect();
        format!(
            "[{verdict}] determinization climb: {}/{} deterministic, final rate {:.0}% \
             (target {:.0}%), windows [{}]",
            self.deterministic,
            self.total,
            self.final_rate * 100.0,
            DETERMINIZATION_TARGET * 100.0,
            windows.join(", ")
        )
    }
}

/// Run `kinds` intent kinds `repeats` times each (round-robin), measuring
/// the deterministic share in windows of `window` occurrences.
pub fn determinization_climb(kinds: usize, repeats: u32, window: usize) -> ClimbReport {
    let mut det = DetEngine::new();
    let versions: HashMap<String, u32> =
        [("mail.search".to_string(), 1), ("file.write".to_string(), 1)]
            .into_iter()
            .collect();
    let facts = FactBase::new();
    let mut served: Vec<ServedBy> = Vec::new();

    for rep in 0..repeats {
        for kind in 0..kinds {
            let key = format!("trial.intent.{kind}");
            let plan = climb_plan(kind, rep);
            let by = if det.macro_for(&key).is_some() {
                // L2: fill the macro's slots from this run's own request —
                // slot values come from the user, never from a model.
                let slots = det.macro_for(&key).expect("checked").slots.clone();
                let mut values = HashMap::new();
                for (idx, path) in &slots {
                    let split: Vec<&str> = path.split('.').collect();
                    if let Some(v) = plan.steps[*idx].args.get_path(&split) {
                        values.insert(slot_key(*idx, path), v.clone());
                    }
                }
                match det.run_macro(&key, &values, &facts, &versions) {
                    Ok(_) => ServedBy::Macro,
                    Err(_) => ServedBy::Model, // fall back; must not happen in the trial
                }
            } else if det.lookup_cache(&key).is_some() {
                det.record_success(&plan, &versions);
                if det.eligible_for_macro(&key) {
                    let _ = det.compile_macro(&key, true);
                }
                ServedBy::Cache
            } else {
                det.record_success(&plan, &versions);
                if det.eligible_for_macro(&key) {
                    let _ = det.compile_macro(&key, true);
                }
                ServedBy::Model
            };
            served.push(by);
        }
    }

    let total = served.len() as u64;
    let deterministic = served.iter().filter(|s| s.is_deterministic()).count() as u64;
    let windows: Vec<f64> = served
        .chunks(window.max(1))
        .map(|chunk| {
            let det_count = chunk.iter().filter(|s| s.is_deterministic()).count();
            det_count as f64 / chunk.len() as f64
        })
        .collect();
    let final_rate = windows.last().copied().unwrap_or(0.0);
    let climbed = windows.len() > 1 && final_rate > windows[0];
    ClimbReport {
        windows,
        total,
        deterministic,
        final_rate,
        climbed,
        meets_target: final_rate >= DETERMINIZATION_TARGET,
    }
}

// ---------------------------------------------------------------------------
// 2. False-interrupt rate
// ---------------------------------------------------------------------------

/// One scripted interruption with the user's ground-truth verdict for it.
#[derive(Debug, Clone)]
pub struct InterruptCase {
    pub agent: &'static str,
    pub group: String,
    pub category: &'static str,
    pub priority: Priority,
    /// Whether the user, on seeing it, judges it worth interrupting for.
    pub worth_it: bool,
}

fn interrupt_case(
    agent: &'static str,
    group: impl Into<String>,
    category: &'static str,
    priority: Priority,
    worth_it: bool,
) -> InterruptCase {
    InterruptCase { agent, group: group.into(), category, priority, worth_it }
}

#[derive(Debug, Clone)]
pub struct InterruptionReport {
    pub delivered: u64,
    pub false_interrupts: u64,
    pub false_rate: f64,
    /// Same stream with no scheduler: every request interrupts.
    pub control_false_rate: f64,
    /// A rejected category goes silent and its repeat is default-denied.
    pub silence_demo: bool,
    pub meets_target: bool,
}

impl InterruptionReport {
    pub fn render(&self) -> String {
        let verdict = if self.meets_target { "PASS" } else { "FAIL" };
        format!(
            "[{verdict}] false-interrupt rate: {}/{} delivered ({:.1}%, target < {:.0}%), \
             no-scheduler control {:.0}%, silence-after-rejection {}",
            self.false_interrupts,
            self.delivered,
            self.false_rate * 100.0,
            FALSE_INTERRUPT_TARGET * 100.0,
            self.control_false_rate * 100.0,
            if self.silence_demo { "works" } else { "BROKEN" }
        )
    }
}

/// Phase A measures the false-interrupt rate on an established agent
/// (allowance warmed to the slow-start threshold, as the scheduler's own
/// tests do); phase B verifies the silence-after-rejection property on a
/// second agent. The control replays the same cases with no scheduler.
pub fn interruption_trial() -> InterruptionReport {
    let mut sched = AttentionScheduler::new();
    for _ in 0..3 {
        sched.on_responded("assistant"); // established agent: allowance 8
    }
    let mut now_ms: u64 = 0;
    let mut delivered: u64 = 0;
    let mut false_interrupts: u64 = 0;

    // Phase A: 6 distinct confirmations, 1 coalesced group of 3, 12
    // progress pings, 4 info notes, 1 urgent transfer — in that order,
    // so the P0 overdraw (and its debt) lands last.
    let mut cases: Vec<InterruptCase> = Vec::new();
    for i in 0..6 {
        cases.push(interrupt_case(
            "assistant",
            format!("confirm-{i}"),
            "confirm",
            Priority::P1,
            true,
        ));
    }
    for _ in 0..3 {
        cases.push(interrupt_case("assistant", "trip", "confirm", Priority::P1, true));
    }
    for i in 0..12 {
        cases.push(interrupt_case(
            "assistant",
            format!("progress-{i}"),
            "progress",
            Priority::P2,
            false,
        ));
    }
    for i in 0..4 {
        cases.push(interrupt_case(
            "assistant",
            format!("info-{i}"),
            "info",
            Priority::P3,
            false,
        ));
    }
    cases.push(interrupt_case(
        "assistant",
        "transfer",
        "transfer-confirm",
        Priority::P0,
        true,
    ));

    for case in &cases {
        now_ms += 10_000;
        let req = InterruptRequest {
            agent: case.agent.into(),
            intent_group: case.group.clone(),
            category: case.category.into(),
            priority: case.priority,
        };
        match sched.request(&req, now_ms) {
            AttentionDecision::Immediate | AttentionDecision::Queued => {
                delivered += 1;
                if case.worth_it {
                    sched.on_responded(case.agent);
                } else {
                    false_interrupts += 1;
                    sched.on_rejected(case.agent, case.category, now_ms);
                }
            }
            // Coalesced into an already-pending interruption: one delivery,
            // already counted when it queued. Digest/ledger/silence/denied:
            // never an interruption.
            _ => {}
        }
    }
    // The coalesced group was delivered exactly once.
    assert_eq!(sched.drain_pending("assistant", "trip"), 3);
    let _ = sched.flush_digest();

    // Phase B: silence after rejection, on its own agent.
    for _ in 0..3 {
        sched.on_responded("spammer");
    }
    now_ms += 10_000;
    let promo = InterruptRequest {
        agent: "spammer".into(),
        intent_group: "promo-1".into(),
        category: "promo".into(),
        priority: Priority::P1,
    };
    let first = sched.request(&promo, now_ms);
    let mut silence_demo = false;
    if first == AttentionDecision::Queued {
        let _ = sched.drain_pending("spammer", "promo-1");
        sched.on_rejected("spammer", "promo", now_ms);
        let repeat = sched.request(&promo, now_ms + 1_000);
        silence_demo = repeat == AttentionDecision::Silenced;
    }

    // Control: no scheduler — every request interrupts the user.
    let control_delivered = cases.len() as f64;
    let control_false = cases.iter().filter(|c| !c.worth_it).count() as f64;
    let control_false_rate = control_false / control_delivered;

    let false_rate = if delivered == 0 { 0.0 } else { false_interrupts as f64 / delivered as f64 };
    InterruptionReport {
        delivered,
        false_interrupts,
        false_rate,
        control_false_rate,
        silence_demo,
        meets_target: false_rate < FALSE_INTERRUPT_TARGET && silence_demo,
    }
}

// ---------------------------------------------------------------------------
// 3. Type-check interception rate
// ---------------------------------------------------------------------------

fn trial_checker() -> TypeChecker {
    let mut tc = TypeChecker::new();
    let read: BTreeSet<Effect> = [Effect::Read].into_iter().collect();
    let send: BTreeSet<Effect> = [Effect::Send].into_iter().collect();
    let write: BTreeSet<Effect> = [Effect::Write].into_iter().collect();
    let transfer: BTreeSet<Effect> = [Effect::Transfer].into_iter().collect();
    let delete: BTreeSet<Effect> = [Effect::Delete].into_iter().collect();
    tc.register("mail.search", read.clone());
    tc.register("mail.send", send);
    tc.register("file.read", read);
    tc.register("file.write", write);
    tc.register("bank.transfer", transfer);
    tc.register("file.delete", delete);
    tc
}

fn held(pairs: &[(&str, Effect)]) -> BTreeSet<(String, Effect)> {
    pairs.iter().map(|(t, e)| (t.to_string(), *e)).collect()
}

fn plan_with(steps: Vec<PlanStep>) -> Plan {
    let mut p = Plan::new("trial-plan", "trial.typecheck", TrustLabel::User);
    p.steps = steps;
    p
}

#[derive(Debug, Clone)]
pub struct TypeCheckReport {
    pub good_total: u64,
    pub good_passed: u64,
    pub bad_total: u64,
    pub bad_blocked: u64,
    pub interception_rate: f64,
    pub meets_target: bool,
}

impl TypeCheckReport {
    pub fn render(&self) -> String {
        let verdict = if self.meets_target { "PASS" } else { "FAIL" };
        format!(
            "[{verdict}] type-check interception: {}/{} bad plans blocked ({:.1}%, target > {:.1}%), \
             {}/{} good plans passed",
            self.bad_blocked,
            self.bad_total,
            self.interception_rate * 100.0,
            INTERCEPTION_TARGET * 100.0,
            self.good_passed,
            self.good_total
        )
    }
}

/// A read-only agent (search/read/write, no send/transfer/delete) faces
/// 10 good plans and 10 bad ones: sends, transfers and deletes it holds
/// no capability for, plus one unknown tool.
pub fn typecheck_trial() -> TypeCheckReport {
    let tc = trial_checker();
    let held_caps = held(&[
        ("mail.search", Effect::Read),
        ("file.read", Effect::Read),
        ("file.write", Effect::Write),
    ]);

    let mut good: Vec<Plan> = Vec::new();
    for i in 0..4 {
        good.push(plan_with(vec![
            PlanStep::new("mail.search", "inbox", Value::text(format!("q{i}"))),
            PlanStep::new("file.read", format!("doc-{i}.txt"), Value::Null),
        ]));
    }
    for i in 0..6 {
        good.push(plan_with(vec![PlanStep::new(
            "file.write",
            format!("note-{i}.txt"),
            Value::text(format!("body {i}")),
        )]));
    }

    let mut bad: Vec<Plan> = Vec::new();
    for i in 0..4 {
        bad.push(plan_with(vec![PlanStep::new(
            "mail.send",
            format!("draft-{i}"),
            Value::text("pay the invoice"),
        )]));
    }
    for i in 0..3 {
        bad.push(plan_with(vec![PlanStep::new(
            "bank.transfer",
            format!("acct-{i}"),
            Value::int(100 + i as i64),
        )]));
    }
    for i in 0..2 {
        bad.push(plan_with(vec![PlanStep::new(
            "file.delete",
            format!("old-{i}.log"),
            Value::Null,
        )]));
    }
    bad.push(plan_with(vec![PlanStep::new("root.shell", "host", Value::Null)]));

    let good_total = good.len() as u64;
    let good_passed = good.iter().filter(|p| tc.check(p, &held_caps).is_ok()).count() as u64;
    let bad_total = bad.len() as u64;
    let bad_blocked = bad.iter().filter(|p| tc.check(p, &held_caps).is_err()).count() as u64;
    let interception_rate = if bad_total == 0 { 0.0 } else { bad_blocked as f64 / bad_total as f64 };
    TypeCheckReport {
        good_total,
        good_passed,
        bad_total,
        bad_blocked,
        interception_rate,
        meets_target: interception_rate >= INTERCEPTION_TARGET && good_passed == good_total,
    }
}

// ---------------------------------------------------------------------------
// The suite
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct P0Report {
    pub climb: ClimbReport,
    pub interruption: InterruptionReport,
    pub typecheck: TypeCheckReport,
}

impl P0Report {
    /// All three ch. 23 validations pass.
    pub fn verdict(&self) -> bool {
        self.climb.meets_target
            && self.climb.climbed
            && self.interruption.meets_target
            && self.typecheck.meets_target
    }

    pub fn render(&self) -> String {
        let verdict = if self.verdict() { "PASS" } else { "FAIL" };
        format!(
            "P0 acceptance suite (doc ch. 23): {verdict}\n{}\n{}\n{}",
            self.climb.render(),
            self.interruption.render(),
            self.typecheck.render()
        )
    }
}

/// Run all three validations with the default scripted workloads.
pub fn run_p0_suite() -> P0Report {
    P0Report {
        climb: determinization_climb(4, 8, 8),
        interruption: interruption_trial(),
        typecheck: typecheck_trial(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn climb_rate_reaches_target_and_climbs() {
        let rep = determinization_climb(4, 8, 8);
        assert_eq!(rep.total, 32);
        assert_eq!(rep.deterministic, 28, "4 model + 16 cache + 12 macro");
        assert_eq!(rep.windows.len(), 4);
        assert!(rep.windows[0] < rep.final_rate, "rate must climb, not plateau");
        assert!(rep.climbed);
        assert!(rep.final_rate >= DETERMINIZATION_TARGET);
        assert!(rep.meets_target);
    }

    #[test]
    fn climb_discovers_the_varying_leaf_as_a_macro_slot() {
        let mut det = DetEngine::new();
        let versions: HashMap<String, u32> =
            [("mail.search".to_string(), 1), ("file.write".to_string(), 1)]
                .into_iter()
                .collect();
        for rep in 0..5 {
            det.record_success(&climb_plan(0, rep), &versions);
        }
        let mac = det.compile_macro("trial.intent.0", true).expect("5 successes");
        assert_eq!(mac.slots, vec![(1, "date".to_string())]);
    }

    #[test]
    fn interruption_scheduler_keeps_false_rate_below_target() {
        let rep = interruption_trial();
        assert_eq!(rep.delivered, 8, "6 confirmations + 1 coalesced group + 1 P0");
        assert_eq!(rep.false_interrupts, 0, "pings went to the digest, never the user");
        assert_eq!(rep.false_rate, 0.0);
        assert!(rep.control_false_rate > 0.5, "control: {}", rep.control_false_rate);
        assert!(rep.silence_demo, "a rejected category must go silent");
        assert!(rep.meets_target);
    }

    #[test]
    fn typecheck_blocks_all_bad_and_passes_all_good() {
        let rep = typecheck_trial();
        assert_eq!(rep.good_total, 10);
        assert_eq!(rep.good_passed, 10, "no false blocks on good plans");
        assert_eq!(rep.bad_total, 10);
        assert_eq!(rep.bad_blocked, 10);
        assert_eq!(rep.interception_rate, 1.0);
        assert!(rep.meets_target);
    }

    #[test]
    fn full_suite_verdict_is_pass() {
        let rep = run_p0_suite();
        assert!(rep.verdict());
        let text = rep.render();
        assert!(text.contains("P0 acceptance suite"));
        assert!(text.contains("PASS"));
        assert!(text.contains("determinization climb"));
        assert!(text.contains("false-interrupt rate"));
        assert!(text.contains("type-check interception"));
    }
}
