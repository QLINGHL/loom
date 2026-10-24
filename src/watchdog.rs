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
