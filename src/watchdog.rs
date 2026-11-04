//! The failure taxonomy and its detectors (doc ch. 16): agents need an
//! exception vector table just like CPUs.
//!
//! This module implements the two detectors with real mechanics:
//!
//! * **No-progress watchdog.** The task state — plan progress, confirmed
//!   facts, completed side effects — is hashed into a fingerprint each
//!   round. If the fingerprint does not change for N consecutive rounds,
//!   the task is declared stalled (the agent is looping) and must abort,
//!   switch strategy, or escalate to the human.
//! * **Intent versioning.** When the user changes their mind mid-task, the
//!   intent is revised (v1 → v2) rather than restarted: the diff keeps
//!   completed steps whose goal-part survives in the new version and
//!   invalidates only the affected ones — incremental compilation for
//!   intentions.
//!
//! * **Semantic no-progress recheck (this item, doc ch. 16 + ch. 21 open
//!   problem 3).** A fingerprint alone misjudges in both directions: a
//!   long task doing internal work trips the watchdog while its
//!   fingerprint is frozen, and a task rereading different fragments of
//!   the same file keeps changing its fingerprint while getting no closer
//!   to its goal. So a watchdog trip never kills directly: it triggers a
//!   recheck against the task's postcondition — "how far from the
//!   postcondition are we?" — and only a distance that fails to shrink
//!   across `grace` consecutive rechecks confirms the stall. The mirror
//!   direction is covered too: while fingerprints keep changing, a
//!   distance that stays flat for `semantic_window` observations is a
//!   semantic stall (busy, not advancing).
//!
//!   The distance itself is a model judgement in the real system (doc
//!   ch. 16: the check runs only when the watchdog trips, so its cost is
//!   acceptable). This crate therefore takes the distance as an input
//!   and owns only the deterministic verdict logic — exactly like the
//!   `PlanModel` slot in [`crate::runtime`]. The [`SemanticJudge`] trait
//!   is the model slot; [`DeterministicJudge`] is the cheap fallback
//!   that scores unmet postcondition requirements from the snapshot.
//!   That fallback has one honest limitation, documented on the type:
//!   it is a pure function of the snapshot, so on a fingerprint-frozen
//!   task it returns the same distance and can only confirm, never
//!   reprieve — which is precisely why the model slot exists. It is
//!   still useful in the mirror direction, where fingerprints churn.

use crate::crypto::sha256;

/// The exception vector table of doc ch. 16. Each kind pairs a symptom with
/// a deterministic OS response; none of the responses involve asking the
/// model what to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureKind {
    /// Said something with no provenance ("unsupported assertion").
    UnsupportedAssertion,
    /// Repeats the same action without progress.
    NoProgressLoop,
    /// Requested a capability it does not hold.
    PrivilegeEscalation,
    /// Plan carries instructions from untrusted content.
    InjectionContamination,
    /// Token / memory / time budgets exhausted.
    ResourceExhaustion,
    /// A precondition no longer holds: the world changed.
    WorldChanged,
    /// The user changed their mind mid-task.
    IntentDrift,
}

impl FailureKind {
    /// The OS handler for this failure class (doc ch. 16, "OS 处理").
    pub fn handler(self) -> &'static str {
        match self {
            FailureKind::UnsupportedAssertion => "reject output; retry or mark uncertain",
            FailureKind::NoProgressLoop => "abort; switch strategy or escalate to human",
            FailureKind::PrivilegeEscalation => "deterministic deny + ledger entry",
            FailureKind::InjectionContamination => "downgrade plan to needs-confirmation",
            FailureKind::ResourceExhaustion => "preempt, checkpoint, degrade",
            FailureKind::WorldChanged => "roll back and replan",
            FailureKind::IntentDrift => "replan, keeping still-valid completed steps",
        }
    }

    pub fn cpu_analogy(self) -> &'static str {
        match self {
            FailureKind::NoProgressLoop => "deadlock detection",
            FailureKind::PrivilegeEscalation => "protection fault",
            FailureKind::ResourceExhaustion => "OOM killer",
            FailureKind::WorldChanged => "optimistic-concurrency failure",
            FailureKind::IntentDrift => "signal",
            _ => "—",
        }
    }
}

/// What the watchdog observes each round: how far the plan has advanced,
/// which facts are confirmed, which side effects are done.
#[derive(Debug, Clone, Default)]
pub struct TaskSnapshot {
    pub progress: u64,
    pub confirmed_facts: Vec<String>,
    pub completed_effects: Vec<String>,
}

impl TaskSnapshot {
    /// State fingerprint: a hash over (progress, facts, effects), order
    /// independent for the set-valued components.
    pub fn fingerprint(&self) -> [u8; 32] {
        let mut facts = self.confirmed_facts.clone();
        facts.sort();
        let mut effects = self.completed_effects.clone();
        effects.sort();
        sha256(format!("{}|{}|{}", self.progress, facts.join(","), effects.join(",")).as_bytes())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WatchState {
    Progressing,
    /// Fingerprint unchanged for the configured number of rounds.
    Stalled,
}

#[derive(Debug)]
pub struct ProgressWatchdog {
    window: usize,
    last: Option<[u8; 32]>,
    unchanged: usize,
}

impl ProgressWatchdog {
    /// `window` = how many consecutive identical fingerprints constitute a
    /// stall (doc ch. 16 suggests a small N; 3 here).
    pub fn new(window: usize) -> Self {
        ProgressWatchdog { window: window.max(1), last: None, unchanged: 0 }
    }

    pub fn observe(&mut self, snapshot: &TaskSnapshot) -> WatchState {
        let fp = snapshot.fingerprint();
        match self.last {
            Some(prev) if prev == fp => {
                self.unchanged += 1;
            }
            _ => {
                self.unchanged = 0;
            }
        }
        self.last = Some(fp);
        if self.unchanged >= self.window {
            WatchState::Stalled
        } else {
            WatchState::Progressing
        }
    }

    pub fn reset(&mut self) {
        self.last = None;
        self.unchanged = 0;
    }
}

/// One step of an intent, tagged with the part of the goal it serves. The
/// tag is what revision diffs match on.
#[derive(Debug, Clone, PartialEq)]
pub struct StepRecord {
    pub id: String,
    pub goal_tag: String,
    pub done: bool,
}

#[derive(Debug, Clone)]
pub struct IntentRecord {
    pub id: String,
    pub version: u32,
    pub steps: Vec<StepRecord>,
}

/// The result of revising an intent: which completed work survived, which
/// was invalidated, what is newly required.
#[derive(Debug, Clone, PartialEq)]
pub struct IntentDiff {
    pub from_version: u32,
    pub to_version: u32,
    pub kept_done: Vec<String>,
    pub invalidated: Vec<String>,
    pub added: Vec<String>,
}

/// Revise an intent to a new version (doc ch. 16, intent drift).
/// `new_steps` is the new version's step list as (id, goal_tag) pairs.
/// Steps whose goal_tag survives keep their completion state; the rest are
/// invalidated; brand-new tags are added as pending work.
pub fn revise_intent(record: &mut IntentRecord, new_steps: Vec<(String, String)>) -> IntentDiff {
    let from = record.version;
    let old_tags: Vec<String> = record.steps.iter().map(|s| s.goal_tag.clone()).collect();
    let new_tags: Vec<String> = new_steps.iter().map(|(_, t)| t.clone()).collect();

    let mut kept_done = Vec::new();
    let mut invalidated = Vec::new();
    for step in &record.steps {
        if new_tags.contains(&step.goal_tag) {
            if step.done {
                kept_done.push(step.id.clone());
            }
        } else {
            invalidated.push(step.id.clone());
        }
    }
    let mut added = Vec::new();
    for (id, tag) in &new_steps {
        if !old_tags.contains(tag) {
            added.push(id.clone());
        }
    }

    record.steps = new_steps
        .into_iter()
        .map(|(id, tag)| StepRecord {
            done: record
                .steps
                .iter()
                .find(|s| s.goal_tag == tag)
                .map(|s| s.done)
                .unwrap_or(false),
            id,
            goal_tag: tag,
        })
        .collect();
    record.version += 1;

    IntentDiff { from_version: from, to_version: record.version, kept_done, invalidated, added }
}

/// What the task must hold when it is done: the facts that must be
/// confirmed, the side effects that must be completed, and the minimum
/// plan progress. Distance to this postcondition is what the semantic
/// recheck measures (doc ch. 16, "距离目标的后置条件还有多远").
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Postcondition {
    pub required_facts: Vec<String>,
    pub required_effects: Vec<String>,
    /// Minimum plan progress (in the snapshot's `progress` units).
    pub min_progress: u64,
}

impl Postcondition {
    pub fn new(
        required_facts: Vec<String>,
        required_effects: Vec<String>,
        min_progress: u64,
    ) -> Self {
        Postcondition { required_facts, required_effects, min_progress }
    }

    /// Total requirement units: one per required fact, one per required
    /// effect, one per unit of minimum progress.
    pub fn total(&self) -> usize {
        self.required_facts.len()
            + self.required_effects.len()
            + usize::try_from(self.min_progress).unwrap_or(usize::MAX)
    }

    /// Distance from a snapshot to this postcondition: how many
    /// requirement units are still unmet. Smaller is closer; zero means
    /// the postcondition holds.
    pub fn distance(&self, snapshot: &TaskSnapshot) -> Distance {
        let missing_facts = self
            .required_facts
            .iter()
            .filter(|f| !snapshot.confirmed_facts.contains(f))
            .count();
        let missing_effects = self
            .required_effects
            .iter()
            .filter(|e| !snapshot.completed_effects.contains(e))
            .count();
        let progress_gap =
            usize::try_from(self.min_progress.saturating_sub(snapshot.progress))
                .unwrap_or(usize::MAX);
        Distance { remaining: missing_facts + missing_effects + progress_gap, total: self.total() }
    }
}

/// How far a task is from its postcondition. Only `remaining` is compared
/// by the recheck logic; `total` exists so callers can report progress
/// ("3 of 8 requirements met") and so a zero-total postcondition is
/// visibly satisfied rather than vacuously small.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Distance {
    pub remaining: usize,
    pub total: usize,
}

impl Distance {
    pub fn is_satisfied(&self) -> bool {
        self.remaining == 0
    }

    pub fn met(&self) -> usize {
        self.total.saturating_sub(self.remaining)
    }
}

/// The model slot for the semantic distance judgement (doc ch. 16). A
/// real implementation looks at the agent's recent trajectory — its
/// reasoning and intermediate work, which the state fingerprint does
/// not capture — and estimates how far the task is from its
/// postcondition. The checker below only consumes the estimate; it
/// never asks the model what to do about a stall.
pub trait SemanticJudge {
    fn estimate(&self, snapshot: &TaskSnapshot, postcondition: &Postcondition) -> Distance;
}

/// Cheap fallback judge: scores unmet postcondition requirements from
/// the snapshot alone. Limitation, stated plainly: because this is a
/// pure function of the snapshot, and the fingerprint is also a function
/// of the snapshot, a fingerprint-frozen task always gets the same
/// distance here — so this judge can confirm a watchdog trip but never
/// reprieve one. Reprieve needs a real model judge whose estimate can
/// move while the fingerprint stands still. Where this judge earns its
/// keep is the mirror direction: fingerprints that churn while the
/// distance stands still are a semantic stall it can catch.
#[derive(Debug, Clone, Copy, Default)]
pub struct DeterministicJudge;

impl SemanticJudge for DeterministicJudge {
    fn estimate(&self, snapshot: &TaskSnapshot, postcondition: &Postcondition) -> Distance {
        postcondition.distance(snapshot)
    }
}

/// Verdict of a watchdog-triggered semantic recheck.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecheckVerdict {
    /// Give the task another window: its distance shrank, it is the
    /// first recheck (no baseline yet), it has not yet exhausted its
    /// grace, or its postcondition is already satisfied (a satisfied
    /// task is never killed for stalling).
    Reprieve,
    /// Distance failed to shrink for `grace` consecutive rechecks: the
    /// stall is confirmed, and the ch. 16 handler applies (abort, switch
    /// strategy, or escalate to the human).
    ConfirmedStall,
}

/// Verdict of a semantic observation made while fingerprints change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticObservation {
    /// Distance shrank, or this is the first observation: the task is
    /// approaching its postcondition (or nothing is known against it).
    Approaching,
    /// Distance stood still for `semantic_window` consecutive
    /// observations while fingerprints kept changing: busy, not
    /// advancing.
    SemanticallyStalled,
}

/// The deterministic verdict half of the semantic no-progress criterion. It
/// tracks the best-known distance to the postcondition and decides, on
/// each recheck or observation, whether the task is approaching it.
/// The two counters are deliberately independent: recheck grace counts
/// watchdog trips, observation windows count fingerprint-changing
/// rounds, and a task can be reprieved on one axis while stalling on
/// the other. `last_distance` is shared because it is one fact — the
/// most recent distance estimate, whatever produced it.
#[derive(Debug, Clone)]
pub struct SemanticNoProgressCheck {
    /// Consecutive non-shrinking rechecks that confirm a stall.
    grace: usize,
    /// Consecutive flat-distance observations that flag a semantic stall.
    semantic_window: usize,
    last_distance: Option<usize>,
    stagnant_rechecks: usize,
    stagnant_observations: usize,
}

impl SemanticNoProgressCheck {
    pub fn new(grace: usize, semantic_window: usize) -> Self {
        SemanticNoProgressCheck {
            grace: grace.max(1),
            semantic_window: semantic_window.max(1),
            last_distance: None,
            stagnant_rechecks: 0,
            stagnant_observations: 0,
        }
    }

    /// Recheck called when the fingerprint watchdog trips. Never kills
    /// on a single trip: the first recheck only establishes the
    /// baseline, and a stall is confirmed only after `grace`
    /// consecutive rechecks with no shrinkage.
    pub fn recheck(&mut self, distance: &Distance) -> RecheckVerdict {
        if distance.is_satisfied() {
            self.last_distance = Some(0);
            self.stagnant_rechecks = 0;
            return RecheckVerdict::Reprieve;
        }
        match self.last_distance {
            None => {
                self.last_distance = Some(distance.remaining);
                RecheckVerdict::Reprieve
            }
            Some(prev) if distance.remaining < prev => {
                self.last_distance = Some(distance.remaining);
                self.stagnant_rechecks = 0;
                RecheckVerdict::Reprieve
            }
            Some(_) => {
                self.last_distance = Some(distance.remaining);
                self.stagnant_rechecks += 1;
                if self.stagnant_rechecks >= self.grace {
                    RecheckVerdict::ConfirmedStall
                } else {
                    RecheckVerdict::Reprieve
                }
            }
        }
    }

    /// Observation made on a round whose fingerprint changed. A flat or
    /// regressing distance (regression is not approach) counts as
    /// stagnant; shrinkage resets the count.
    pub fn observe(&mut self, distance: &Distance) -> SemanticObservation {
        if distance.is_satisfied() {
            self.last_distance = Some(0);
            self.stagnant_observations = 0;
            return SemanticObservation::Approaching;
        }
        match self.last_distance {
            None => {
                self.last_distance = Some(distance.remaining);
                SemanticObservation::Approaching
            }
            Some(prev) if distance.remaining < prev => {
                self.last_distance = Some(distance.remaining);
                self.stagnant_observations = 0;
                SemanticObservation::Approaching
            }
            Some(_) => {
                self.last_distance = Some(distance.remaining);
                self.stagnant_observations += 1;
                if self.stagnant_observations >= self.semantic_window {
                    SemanticObservation::SemanticallyStalled
                } else {
                    SemanticObservation::Approaching
                }
            }
        }
    }

    pub fn reset(&mut self) {
        self.last_distance = None;
        self.stagnant_rechecks = 0;
        self.stagnant_observations = 0;
    }
}

/// Combined verdict of one guarded round.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardVerdict {
    /// Fingerprint and distance both behave: keep running.
    Progressing,
    /// Watchdog tripped but the recheck reprieved the task: it gets a
    /// fresh fingerprint window instead of being killed.
    Reprieved,
    /// Watchdog tripped and the recheck confirmed the stall.
    ConfirmedStall,
    /// Fingerprints churn but distance stands still: busy, not advancing.
    SemanticStall,
}

/// A fingerprint watchdog with its semantic recheck wired in — the
/// usable form of the ch. 16判据. Each round the caller supplies the
/// snapshot and the current distance estimate (from a model judge, or
/// from [`DeterministicJudge`] where its limitation is acceptable):
///
/// * fingerprint frozen past the window → recheck; reprieve resets the
///   watchdog so the task gets a fresh window, confirmation kills it;
/// * fingerprint changed → semantic observation; a flat distance for
///   `semantic_window` such rounds is a semantic stall;
/// * fingerprint frozen but not yet past the window → progressing; the
///   semantic observation is deliberately not consulted on those
///   rounds, or a slow-but-real approach would be flagged before the
///   watchdog even trips.
#[derive(Debug)]
pub struct GuardedWatchdog {
    watchdog: ProgressWatchdog,
    check: SemanticNoProgressCheck,
    last_fingerprint: Option<[u8; 32]>,
}

impl GuardedWatchdog {
    pub fn new(window: usize, grace: usize, semantic_window: usize) -> Self {
        GuardedWatchdog {
            watchdog: ProgressWatchdog::new(window),
            check: SemanticNoProgressCheck::new(grace, semantic_window),
            last_fingerprint: None,
        }
    }

    pub fn observe(&mut self, snapshot: &TaskSnapshot, distance: &Distance) -> GuardVerdict {
        let fp = snapshot.fingerprint();
        let fingerprint_changed = self.last_fingerprint != Some(fp);
        self.last_fingerprint = Some(fp);
        match self.watchdog.observe(snapshot) {
            WatchState::Stalled => match self.check.recheck(distance) {
                RecheckVerdict::Reprieve => {
                    // Fresh fingerprint window for the reprieved task.
                    // The guard's own last fingerprint is kept, so the
                    // next identical snapshot is still "unchanged" and
                    // does not spuriously count as a semantic observation.
                    self.watchdog.reset();
                    GuardVerdict::Reprieved
                }
                RecheckVerdict::ConfirmedStall => GuardVerdict::ConfirmedStall,
            },
            WatchState::Progressing => {
                if fingerprint_changed {
                    match self.check.observe(distance) {
                        SemanticObservation::SemanticallyStalled => GuardVerdict::SemanticStall,
                        SemanticObservation::Approaching => GuardVerdict::Progressing,
                    }
                } else {
                    GuardVerdict::Progressing
                }
            }
        }
    }

    /// Convenience: estimate the distance with a judge, then observe.
    pub fn observe_judged<J: SemanticJudge>(
        &mut self,
        snapshot: &TaskSnapshot,
        judge: &J,
        postcondition: &Postcondition,
    ) -> GuardVerdict {
        let distance = judge.estimate(snapshot, postcondition);
        self.observe(snapshot, &distance)
    }

    pub fn reset(&mut self) {
        self.watchdog.reset();
        self.check.reset();
        self.last_fingerprint = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(progress: u64) -> TaskSnapshot {
        TaskSnapshot { progress, confirmed_facts: vec!["f1".into()], completed_effects: vec![] }
    }

    #[test]
    fn taxonomy_handlers_cover_the_table() {
        // Every failure kind must have a deterministic handler — the table
        // is only useful if it is total.
        for kind in [
            FailureKind::UnsupportedAssertion,
            FailureKind::NoProgressLoop,
            FailureKind::PrivilegeEscalation,
            FailureKind::InjectionContamination,
            FailureKind::ResourceExhaustion,
            FailureKind::WorldChanged,
            FailureKind::IntentDrift,
        ] {
            assert!(!kind.handler().is_empty());
        }
        assert_eq!(FailureKind::NoProgressLoop.cpu_analogy(), "deadlock detection");
        assert_eq!(FailureKind::PrivilegeEscalation.cpu_analogy(), "protection fault");
    }

    #[test]
    fn fingerprint_ignores_set_ordering() {
        let a = TaskSnapshot {
            progress: 1,
            confirmed_facts: vec!["x".into(), "y".into()],
            completed_effects: vec!["e1".into(), "e2".into()],
        };
        let b = TaskSnapshot {
            progress: 1,
            confirmed_facts: vec!["y".into(), "x".into()],
            completed_effects: vec!["e2".into(), "e1".into()],
        };
        assert_eq!(a.fingerprint(), b.fingerprint());
        let c = TaskSnapshot { progress: 2, ..a.clone() };
        assert_ne!(a.fingerprint(), c.fingerprint());
    }

    #[test]
    fn watchdog_trips_after_window_of_identical_fingerprints() {
        let mut wd = ProgressWatchdog::new(3);
        let s = snap(1);
        assert_eq!(wd.observe(&s), WatchState::Progressing);
        assert_eq!(wd.observe(&s), WatchState::Progressing);
        assert_eq!(wd.observe(&s), WatchState::Progressing);
        assert_eq!(wd.observe(&s), WatchState::Stalled, "3 unchanged rounds = stalled");
    }

    #[test]
    fn watchdog_resets_when_progress_resumes() {
        let mut wd = ProgressWatchdog::new(2);
        let s1 = snap(1);
        wd.observe(&s1);
        wd.observe(&s1);
        assert_eq!(wd.observe(&snap(2)), WatchState::Progressing);
        wd.observe(&snap(2));
        assert_eq!(wd.observe(&snap(2)), WatchState::Stalled);
    }

    #[test]
    fn fact_changes_count_as_progress() {
        let mut wd = ProgressWatchdog::new(2);
        let mut s = snap(1);
        wd.observe(&s);
        wd.observe(&s);
        s.confirmed_facts.push("f2".into());
        assert_eq!(wd.observe(&s), WatchState::Progressing);
    }

    fn intent_v1() -> IntentRecord {
        IntentRecord {
            id: "intent-1".into(),
            version: 1,
            steps: vec![
                StepRecord { id: "s1".into(), goal_tag: "check-availability".into(), done: true },
                StepRecord { id: "s2".into(), goal_tag: "book-room".into(), done: false },
                StepRecord { id: "s3".into(), goal_tag: "reserve-projector".into(), done: false },
            ],
        }
    }

    #[test]
    fn intent_revision_keeps_still_valid_completed_work() {
        // "Book a meeting room" becomes "make it online instead": checking
        // availability stays valid, booking the room and the projector go.
        let mut record = intent_v1();
        let diff = revise_intent(
            &mut record,
            vec![
                ("s1".into(), "check-availability".into()),
                ("s4".into(), "create-video-link".into()),
            ],
        );
        assert_eq!(diff.from_version, 1);
        assert_eq!(diff.to_version, 2);
        assert_eq!(diff.kept_done, vec!["s1".to_string()]);
        assert_eq!(diff.invalidated, vec!["s2".to_string(), "s3".to_string()]);
        assert_eq!(diff.added, vec!["s4".to_string()]);
        assert_eq!(record.version, 2);
        assert!(record.steps.iter().find(|s| s.id == "s1").unwrap().done);
        assert!(!record.steps.iter().find(|s| s.id == "s4").unwrap().done);
    }

    // -- Semantic no-progress recheck (doc ch. 16) ---------------------

    fn postcondition() -> Postcondition {
        Postcondition::new(vec!["booked".into()], vec!["paid".into()], 2)
    }

    fn dist(remaining: usize) -> Distance {
        Distance { remaining, total: 4 }
    }

    #[test]
    fn postcondition_distance_counts_unmet_requirements() {
        let post = postcondition();
        let empty = TaskSnapshot::default();
        assert_eq!(post.distance(&empty), Distance { remaining: 4, total: 4 });
        let one_fact = TaskSnapshot {
            progress: 1,
            confirmed_facts: vec!["booked".into()],
            completed_effects: vec![],
        };
        assert_eq!(post.distance(&one_fact).remaining, 2, "one fact + one progress unit met of 4");
        let done = TaskSnapshot {
            progress: 2,
            confirmed_facts: vec!["booked".into()],
            completed_effects: vec!["paid".into()],
        };
        let d = post.distance(&done);
        assert!(d.is_satisfied());
        assert_eq!(d.met(), 4);
    }

    #[test]
    fn recheck_reprieves_when_distance_shrinks() {
        let mut check = SemanticNoProgressCheck::new(1, 1);
        assert_eq!(check.recheck(&dist(5)), RecheckVerdict::Reprieve, "baseline only");
        assert_eq!(check.recheck(&dist(3)), RecheckVerdict::Reprieve, "approaching: reprieve");
        assert_eq!(check.recheck(&dist(3)), RecheckVerdict::ConfirmedStall, "flat at grace 1: confirm");
    }

    #[test]
    fn recheck_confirms_only_after_grace_stagnant_rechecks() {
        let mut check = SemanticNoProgressCheck::new(2, 1);
        assert_eq!(check.recheck(&dist(5)), RecheckVerdict::Reprieve);
        assert_eq!(check.recheck(&dist(5)), RecheckVerdict::Reprieve, "1st stagnant: benefit of doubt");
        assert_eq!(check.recheck(&dist(5)), RecheckVerdict::ConfirmedStall, "2nd stagnant: confirmed");
    }

    #[test]
    fn recheck_never_kills_a_satisfied_task() {
        let mut check = SemanticNoProgressCheck::new(1, 1);
        assert_eq!(check.recheck(&dist(0)), RecheckVerdict::Reprieve);
        assert_eq!(check.recheck(&dist(0)), RecheckVerdict::Reprieve, "satisfied twice: still reprieve");
        assert_eq!(check.observe(&dist(0)), SemanticObservation::Approaching);
    }

    #[test]
    fn observe_flags_semantic_stall_despite_changing_inputs() {
        let mut check = SemanticNoProgressCheck::new(1, 3);
        assert_eq!(check.observe(&dist(4)), SemanticObservation::Approaching, "baseline");
        assert_eq!(check.observe(&dist(4)), SemanticObservation::Approaching);
        assert_eq!(check.observe(&dist(4)), SemanticObservation::Approaching);
        assert_eq!(check.observe(&dist(4)), SemanticObservation::SemanticallyStalled, "3 flat observations");
    }

    #[test]
    fn observe_resets_when_distance_shrinks_and_regression_counts_as_stagnant() {
        let mut check = SemanticNoProgressCheck::new(1, 2);
        assert_eq!(check.observe(&dist(4)), SemanticObservation::Approaching);
        assert_eq!(check.observe(&dist(5)), SemanticObservation::Approaching, "regression is not approach");
        assert_eq!(check.observe(&dist(3)), SemanticObservation::Approaching, "shrinkage resets");
        assert_eq!(check.observe(&dist(3)), SemanticObservation::Approaching);
        assert_eq!(check.observe(&dist(3)), SemanticObservation::SemanticallyStalled);
    }

    #[test]
    fn deterministic_judge_is_a_pure_snapshot_function() {
        // Documents the fallback's limitation: identical snapshots give
        // identical distances, so it can confirm but never reprieve a
        // fingerprint-frozen task — reprieve needs a real model judge.
        let judge = DeterministicJudge;
        let post = postcondition();
        let s = snap(1);
        assert_eq!(judge.estimate(&s, &post), post.distance(&s));
        assert_eq!(judge.estimate(&s, &post), judge.estimate(&s, &post));
    }

    #[test]
    fn guarded_watchdog_reprieves_a_long_task_whose_distance_shrinks() {
        // Fingerprint never changes, but the distance shrinks between
        // watchdog trips: the old watchdog alone would kill this task.
        let mut guard = GuardedWatchdog::new(2, 2, 3);
        let s = snap(1);
        assert_eq!(guard.observe(&s, &dist(5)), GuardVerdict::Progressing);
        assert_eq!(guard.observe(&s, &dist(5)), GuardVerdict::Progressing);
        assert_eq!(guard.observe(&s, &dist(5)), GuardVerdict::Reprieved, "1st trip: stagnant but grace left");
        assert_eq!(guard.observe(&s, &dist(4)), GuardVerdict::Progressing);
        assert_eq!(guard.observe(&s, &dist(4)), GuardVerdict::Progressing);
        assert_eq!(
            guard.observe(&s, &dist(4)),
            GuardVerdict::Reprieved,
            "2nd trip: distance shrank 5 -> 4, so reprieve, not kill"
        );
    }

    #[test]
    fn guarded_watchdog_confirms_a_true_loop() {
        let mut guard = GuardedWatchdog::new(2, 2, 3);
        let s = snap(1);
        assert_eq!(guard.observe(&s, &dist(5)), GuardVerdict::Progressing);
        assert_eq!(guard.observe(&s, &dist(5)), GuardVerdict::Progressing);
        assert_eq!(guard.observe(&s, &dist(5)), GuardVerdict::Reprieved, "1st trip: grace left");
        assert_eq!(guard.observe(&s, &dist(5)), GuardVerdict::Progressing);
        assert_eq!(guard.observe(&s, &dist(5)), GuardVerdict::Progressing);
        assert_eq!(guard.observe(&s, &dist(5)), GuardVerdict::ConfirmedStall, "2nd trip, still flat: kill");
    }

    #[test]
    fn guarded_watchdog_flags_busy_but_not_advancing() {
        // Doc ch. 16's mirror case: the agent rereads different
        // fragments, so the fingerprint changes every round, but none of
        // the fragments is the required fact — distance never moves.
        // The deterministic judge scores exactly this situation.
        let mut guard = GuardedWatchdog::new(10, 2, 3);
        let judge = DeterministicJudge;
        let post = Postcondition::new(vec!["booked".into()], vec![], 0);
        for i in 0..3 {
            let s = TaskSnapshot {
                progress: 1,
                confirmed_facts: vec![format!("fragment-{i}")],
                completed_effects: vec![],
            };
            assert_eq!(guard.observe_judged(&s, &judge, &post), GuardVerdict::Progressing);
        }
        let s = TaskSnapshot {
            progress: 1,
            confirmed_facts: vec!["fragment-3".into()],
            completed_effects: vec![],
        };
        assert_eq!(
            guard.observe_judged(&s, &judge, &post),
            GuardVerdict::SemanticStall,
            "fingerprint churned 4 rounds, distance flat: semantic stall"
        );
    }

    #[test]
    fn unchanged_intent_revision_keeps_everything() {
        let mut record = intent_v1();
        let diff = revise_intent(
            &mut record,
            vec![
                ("s1".into(), "check-availability".into()),
                ("s2".into(), "book-room".into()),
                ("s3".into(), "reserve-projector".into()),
            ],
        );
        assert_eq!(diff.kept_done, vec!["s1".to_string()]);
        assert!(diff.invalidated.is_empty());
        assert!(diff.added.is_empty());
    }
}
