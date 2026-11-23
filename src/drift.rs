//! Intent drift: versioned intents and incremental replanning (doc ch. 16,
//! "意图漂移：一个被忽略的问题").
//!
//! The doc's example: the user says "book a meeting room", the agent starts
//! executing, and mid-task the user says "never mind, make it online
//! instead". The naive response is to abort and start over — but the work
//! already done (looking up attendees' free time) is still valid and should
//! not be thrown away. So an intent needs to be *versioned*:
//!
//! ```text
//! intent@v1 → intent@v2
//! ```
//!
//! The system computes the diff between versions, judges which completed
//! steps are still valid, and only redoes the affected ones. This is the
//! same idea as incremental compilation in a build system: a change
//! invalidates exactly the outputs that depend on it, nothing more.
//!
//! How validity is decided: every step of a running plan declares which
//! aspects of the intent its correctness depends on — [`Aspect`]. The diff
//! is the set of aspects that changed between versions. A completed step
//! survives iff its dependency set does not intersect the diff; otherwise
//! it must be redone. A completed step that already produced a side effect
//! cannot just be rerun — per axiom III (doc ch. 5, "everything must be
//! compensable") its effect must be compensated first, so redo steps with
//! side effects are flagged in the replan instead of silently re-executed.
//!
//! Steps that have not run yet need no compensation, but a pending step
//! whose dependencies changed was derived from the old intent, so it is
//! marked stale: it must be re-derived from the new intent before it may
//! run. Failed steps are left alone — a drift is not a retry, and the
//! failure taxonomy ([`FailureKind`](crate::watchdog::FailureKind)) owns
//! their handling.
//!
//! The module is deliberately deterministic and model-free: the diff is a
//! structural comparison of intent signatures, and the replan is a set
//! intersection. Deciding *what the new intent is* remains a model
//! judgement that happens outside the kernel; once the user has restated
//! it, everything below is exact.

use crate::intent::IntentSignature;
use std::collections::BTreeSet;

/// One aspect of an intent that a plan step can depend on. The four aspects
/// are the fields of [`IntentSignature`] that a restatement can change:
/// what is to be done, its scope, how success is judged, and what it may
/// cost. `Key` (the intent kind) is not an aspect: two versions of one
/// intent share a key by construction, and a different key is a different
/// intent, not a drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Aspect {
    /// Action and object — "book" / "meeting room".
    Goal,
    /// Scope — "next week", "attendees A and B".
    Scope,
    /// Success criterion — "a filled expense form".
    Success,
    /// Budget — step / model-call / side-effect limits.
    Budget,
}

impl Aspect {
    pub fn name(self) -> &'static str {
        match self {
            Aspect::Goal => "goal",
            Aspect::Scope => "scope",
            Aspect::Success => "success",
            Aspect::Budget => "budget",
        }
    }
}

/// An intent with a version number (doc ch. 16: `intent@v1 → intent@v2`).
/// The first version is v1; every restatement bumps the version and keeps
/// the superseded signatures, so a drift is auditable after the fact.
#[derive(Debug, Clone)]
pub struct VersionedIntent {
    signature: IntentSignature,
    version: u64,
    history: Vec<IntentSignature>,
}

impl VersionedIntent {
    pub fn new(signature: IntentSignature) -> Self {
        VersionedIntent { signature, version: 1, history: Vec::new() }
    }

    pub fn version(&self) -> u64 {
        self.version
    }

    pub fn signature(&self) -> &IntentSignature {
        &self.signature
    }

    /// Past versions, oldest first. The current signature is not in here.
    pub fn history(&self) -> &[IntentSignature] {
        &self.history
    }

    /// Restate the intent. Returns the diff against the current version
    /// and moves this intent to the next version. A restatement identical
    /// to the current signature is still a new version — the user spoke
    /// again — but its diff is empty, so a replan keeps everything.
    pub fn revise(&mut self, new_signature: IntentSignature) -> IntentDiff {
        let diff = IntentDiff::between(self.version, &self.signature, &new_signature);
        self.history.push(std::mem::replace(&mut self.signature, new_signature));
        self.version += 1;
        diff
    }
}

/// What changed between two versions of one intent. The change set is a
/// set of [`Aspect`]s: a step is affected by the drift exactly when one
/// of the aspects it depends on is in this set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntentDiff {
    pub from_version: u64,
    pub to_version: u64,
    /// Aspects whose value differs between the two versions.
    pub changed: BTreeSet<Aspect>,
    /// The old and new scope, for the human-readable report. `None` on a
    /// side means the version had no scope.
    pub scope_before: Option<String>,
    pub scope_after: Option<String>,
}

impl IntentDiff {
    pub fn between(from_version: u64, old: &IntentSignature, new: &IntentSignature) -> Self {
        let mut changed = BTreeSet::new();
        if old.action != new.action || old.object != new.object {
            changed.insert(Aspect::Goal);
        }
        if old.scope != new.scope {
            changed.insert(Aspect::Scope);
        }
        if old.success != new.success {
            changed.insert(Aspect::Success);
        }
        if old.budget != new.budget {
            changed.insert(Aspect::Budget);
        }
        IntentDiff {
            from_version,
            to_version: from_version + 1,
            changed,
            scope_before: old.scope.clone(),
            scope_after: new.scope.clone(),
        }
    }

    /// True when the restatement changed nothing: a replan is a no-op.
    pub fn is_empty(&self) -> bool {
        self.changed.is_empty()
    }

    pub fn touches(&self, aspect: Aspect) -> bool {
        self.changed.contains(&aspect)
    }

    /// A goal change is the most destructive drift: the thing to be done
    /// itself moved. Steps that depend on nothing but the scope (looking
    /// up free time, fetching a document) can still survive it — the
    /// dependency sets decide, not the size of the diff.
    pub fn goal_changed(&self) -> bool {
        self.changed.contains(&Aspect::Goal)
    }
}

/// Whether a tracked step has run yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepStatus {
    /// Ran to completion; its result is in hand.
    Done,
    /// Has not run; derived from the intent version it was planned under.
    Pending,
    /// Ran and failed; owned by the failure taxonomy, not by drift.
    Failed,
}

/// One step of the plan that was running when the user restated the intent.
/// The dependency set is the whole contract: it is what the planner
/// believed the step's correctness rested on, stated in intent aspects.
/// A step that declares no dependencies is unconditional — it survives
/// every drift.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrackedStep {
    /// Stable id within the plan (step index or name).
    pub id: String,
    /// Human-readable description, used in the report.
    pub description: String,
    /// Intent aspects this step's correctness depends on.
    pub depends_on: BTreeSet<Aspect>,
    pub status: StepStatus,
    /// True if the step already changed the outside world (a write, a
    /// send, a booking). Redoing such a step needs compensation first.
    pub has_side_effect: bool,
}

impl TrackedStep {
    pub fn new(
        id: impl Into<String>,
        description: impl Into<String>,
        depends_on: &[Aspect],
        status: StepStatus,
        has_side_effect: bool,
    ) -> Self {
        TrackedStep {
            id: id.into(),
            description: description.into(),
            depends_on: depends_on.iter().copied().collect(),
            status,
            has_side_effect,
        }
    }

    /// True when the drift invalidates this step: at least one aspect it
    /// depends on changed.
    pub fn invalidated_by(&self, diff: &IntentDiff) -> bool {
        self.depends_on.iter().any(|a| diff.touches(*a))
    }
}

/// The outcome of replanning after a drift: which completed steps survive,
/// which must be redone, which pending steps are stale, and what share of
/// the completed work was salvaged.
#[derive(Debug, Clone, PartialEq)]
pub struct Replan {
    /// Completed steps whose dependencies are untouched: keep the result.
    pub kept: Vec<String>,
    /// Completed steps invalidated by the drift: run again.
    pub redo: Vec<String>,
    /// Of `redo`, the steps that already produced a side effect. Their
    /// effect must be compensated (axiom III) before the step is rerun;
    /// the replan flags them, it does not compensate them.
    pub redo_needs_compensation: Vec<String>,
    /// Pending steps invalidated by the drift: derived from the old
    /// intent, so they must be re-derived before they may run. No
    /// compensation is involved — they never ran.
    pub stale_pending: Vec<String>,
    /// Completed steps that were kept, over completed steps in total.
    /// 1.0 when nothing had completed yet (nothing to lose).
    pub salvage_rate: f64,
}

impl Replan {
    /// One-line human-readable summary, e.g. for the action ledger.
    pub fn summary(&self) -> String {
        format!(
            "drift replan: kept {} step(s), redo {} ({} need compensation), {} pending step(s) stale, salvage {:.0}%",
            self.kept.len(),
            self.redo.len(),
            self.redo_needs_compensation.len(),
            self.stale_pending.len(),
            self.salvage_rate * 100.0,
        )
    }
}

/// Replan the running plan after an intent drift (doc ch. 16: "replan,
/// keeping still-valid completed steps"). The rule is a set intersection:
/// a step survives iff none of the aspects it depends on changed.
/// Failed steps are carried through untouched — a drift does not retry
/// them, and their failure handling stays with the watchdog.
pub fn replan(diff: &IntentDiff, steps: &[TrackedStep]) -> Replan {
    let mut kept = Vec::new();
    let mut redo = Vec::new();
    let mut redo_needs_compensation = Vec::new();
    let mut stale_pending = Vec::new();
    let mut completed = 0usize;
    let mut completed_kept = 0usize;

    for step in steps {
        match step.status {
            StepStatus::Done => {
                completed += 1;
                if step.invalidated_by(diff) {
                    redo.push(step.id.clone());
                    if step.has_side_effect {
                        redo_needs_compensation.push(step.id.clone());
                    }
                } else {
                    completed_kept += 1;
                    kept.push(step.id.clone());
                }
            }
            StepStatus::Pending => {
                if step.invalidated_by(diff) {
                    stale_pending.push(step.id.clone());
                }
            }
            StepStatus::Failed => {
                // Not drift's business: leave the failure to its handler.
            }
        }
    }

    let salvage_rate = if completed == 0 {
        1.0
    } else {
        completed_kept as f64 / completed as f64
    };

    Replan { kept, redo, redo_needs_compensation, stale_pending, salvage_rate }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intent::Budget;

    fn meeting_intent() -> IntentSignature {
        IntentSignature::new(
            "meeting.book",
            "book",
            "meeting room",
            "room booked for all attendees",
        )
        .scope("attendees Alice, Bob; next Tuesday 10:00")
    }

    fn steps_for_meeting() -> Vec<TrackedStep> {
        vec![
            // Looking up free time depends on the attendees and the time
            // window — the scope — not on which room gets booked.
            TrackedStep::new(
                "lookup-free-time",
                "look up attendees' free time",
                &[Aspect::Scope],
                StepStatus::Done,
                false,
            ),
            // Booking the room depends on the goal itself.
            TrackedStep::new(
                "book-room",
                "book meeting room R3",
                &[Aspect::Goal, Aspect::Scope],
                StepStatus::Done,
                true,
            ),
            // Sending the invite depends on the goal and the scope.
            TrackedStep::new(
                "send-invite",
                "send calendar invite",
                &[Aspect::Goal, Aspect::Scope],
                StepStatus::Pending,
                true,
            ),
        ]
    }

    #[test]
    fn scope_only_drift_keeps_attendee_lookup_redoes_room_booking() {
        // Doc ch. 16 example: "book a meeting room" → "make it online".
        let mut intent = VersionedIntent::new(meeting_intent());
        let mut restated = IntentSignature::new(
            "meeting.book",
            "book",
            "online meeting",
            "online meeting link for all attendees",
        );
        restated = restated.scope("attendees Alice, Bob; next Tuesday 10:00");
        let diff = intent.revise(restated);

        assert_eq!(intent.version(), 2);
        assert!(diff.goal_changed());
        assert!(!diff.touches(Aspect::Scope));

        let plan = replan(&diff, &steps_for_meeting());
        // Free-time lookup depended only on the unchanged scope: kept.
        assert_eq!(plan.kept, vec!["lookup-free-time".to_string()]);
        // The room booking is now meaningless and already changed the
        // world: redo, and compensate the booking first.
        assert_eq!(plan.redo, vec!["book-room".to_string()]);
        assert_eq!(plan.redo_needs_compensation, vec!["book-room".to_string()]);
        // The invite was never sent, but it was derived for a room
        // meeting: stale, re-derive it for the online meeting.
        assert_eq!(plan.stale_pending, vec!["send-invite".to_string()]);
        assert!((plan.salvage_rate - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn scope_change_invalidates_only_scope_dependent_steps() {
        let mut intent = VersionedIntent::new(meeting_intent());
        let restated = meeting_intent().scope("attendees Alice, Carol; next Tuesday 10:00");
        let diff = intent.revise(restated);

        assert!(!diff.goal_changed());
        assert!(diff.touches(Aspect::Scope));
        assert_eq!(diff.scope_before.as_deref(), Some("attendees Alice, Bob; next Tuesday 10:00"));
        assert_eq!(diff.scope_after.as_deref(), Some("attendees Alice, Carol; next Tuesday 10:00"));

        let plan = replan(&diff, &steps_for_meeting());
        // Bob left, Carol joined: the free-time lookup is stale.
        assert!(plan.kept.is_empty());
        assert_eq!(plan.redo, vec!["lookup-free-time".to_string(), "book-room".to_string()]);
        assert_eq!(plan.redo_needs_compensation, vec!["book-room".to_string()]);
        assert!((plan.salvage_rate - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn budget_tightening_only_hits_budget_dependent_steps() {
        let mut intent = VersionedIntent::new(meeting_intent());
        let mut restated = meeting_intent();
        restated = restated.budget(Budget::new(4, 0, 1));
        let diff = intent.revise(restated);

        assert_eq!(diff.changed, BTreeSet::from([Aspect::Budget]));
        let steps = vec![
            TrackedStep::new("lookup-free-time", "look up free time", &[Aspect::Scope], StepStatus::Done, false),
            TrackedStep::new(
                "estimate-cost",
                "estimate cost of the booking",
                &[Aspect::Budget],
                StepStatus::Done,
                false,
            ),
        ];
        let plan = replan(&diff, &steps);
        assert_eq!(plan.kept, vec!["lookup-free-time".to_string()]);
        assert_eq!(plan.redo, vec!["estimate-cost".to_string()]);
        assert!(plan.redo_needs_compensation.is_empty());
    }

    #[test]
    fn identical_restatement_is_a_noop_drift() {
        let mut intent = VersionedIntent::new(meeting_intent());
        let diff = intent.revise(meeting_intent());
        assert!(diff.is_empty());
        assert_eq!(intent.version(), 2);
        assert_eq!(intent.history().len(), 1);

        let plan = replan(&diff, &steps_for_meeting());
        assert_eq!(plan.kept, vec!["lookup-free-time".to_string(), "book-room".to_string()]);
        assert!(plan.redo.is_empty());
        assert!(plan.stale_pending.is_empty());
        assert!((plan.salvage_rate - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn unconditional_steps_survive_any_drift() {
        let mut intent = VersionedIntent::new(meeting_intent());
        let restated = IntentSignature::new("meeting.book", "cancel", "meeting", "meeting cancelled");
        let diff = intent.revise(restated);
        assert!(diff.goal_changed());

        let steps = vec![TrackedStep::new(
            "log-request",
            "log the request in the ledger",
            &[],
            StepStatus::Done,
            false,
        )];
        let plan = replan(&diff, &steps);
        assert_eq!(plan.kept, vec!["log-request".to_string()]);
    }

    #[test]
    fn failed_steps_are_left_to_the_failure_taxonomy() {
        let mut intent = VersionedIntent::new(meeting_intent());
        let restated = meeting_intent().scope("attendees Alice, Bob; next Tuesday 11:00");
        let diff = intent.revise(restated);

        let steps = vec![
            TrackedStep::new("lookup-free-time", "look up free time", &[Aspect::Scope], StepStatus::Failed, false),
            TrackedStep::new("book-room", "book room", &[Aspect::Goal], StepStatus::Done, false),
        ];
        let plan = replan(&diff, &steps);
        // The failed lookup is neither kept nor redone by the drift.
        assert!(!plan.kept.contains(&"lookup-free-time".to_string()));
        assert!(!plan.redo.contains(&"lookup-free-time".to_string()));
        assert_eq!(plan.kept, vec!["book-room".to_string()]);
    }

    #[test]
    fn salvage_rate_is_one_when_nothing_completed_yet() {
        let mut intent = VersionedIntent::new(meeting_intent());
        let diff = intent.revise(meeting_intent().scope("other scope"));
        let steps = vec![TrackedStep::new(
            "send-invite",
            "send invite",
            &[Aspect::Scope],
            StepStatus::Pending,
            false,
        )];
        let plan = replan(&diff, &steps);
        assert!((plan.salvage_rate - 1.0).abs() < f64::EPSILON);
        assert_eq!(plan.stale_pending, vec!["send-invite".to_string()]);
    }

    #[test]
    fn replan_summary_reports_counts_and_salvage() {
        let mut intent = VersionedIntent::new(meeting_intent());
        let restated = IntentSignature::new(
            "meeting.book",
            "book",
            "online meeting",
            "online meeting link for all attendees",
        )
        .scope("attendees Alice, Bob; next Tuesday 10:00");
        let diff = intent.revise(restated);
        let plan = replan(&diff, &steps_for_meeting());
        assert_eq!(
            plan.summary(),
            "drift replan: kept 1 step(s), redo 1 (1 need compensation), 1 pending step(s) stale, salvage 50%"
        );
    }

    #[test]
    fn drift_maps_to_the_intent_drift_failure_kind() {
        // The taxonomy already classifies this failure; this module is the
        // mechanism its handler promises ("replan, keeping still-valid
        // completed steps").
        use crate::watchdog::FailureKind;
        assert_eq!(
            FailureKind::IntentDrift.handler(),
            "replan, keeping still-valid completed steps"
        );
        assert_eq!(FailureKind::IntentDrift.cpu_analogy(), "signal");
    }
}
