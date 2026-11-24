//! Graceful degradation as a first-class citizen (doc ch. 5, axiom V, and
//! the testable criterion of doc ch. 6).
//!
//! Axiom V: when the model is unavailable, the network is down, the budget
//! is exhausted, or the battery is critical, the system must fall back to
//! traditional interaction and stay usable. The doc's test for whether the
//! layering is real: *delete all model files — the system must still boot,
//! still open files, still complete everyday operations the manual way.* If
//! it cannot boot without a model, it is not an operating system, it is an
//! application dressed up as one.
//!
//! The degradation ladder has three rungs:
//!
//! ```text
//! Full ── model / network / budget fails ──► MacroOnly ── battery critical ──► Manual
//!   │                                          │                                 │
//!   │ model calls, macros,                     │ no model calls;                 │ no model calls,
//!   │ deterministic steps,                     │ macros + deterministic          │ no macros;
//!   │ manual actions                           │ steps + manual actions        │ manual actions only
//! ```
//!
//! Every rung spells out what is allowed and what is forbidden
//! ([`DegradationMode::allows`]). Degradation and recovery are both
//! recorded in an append-only event log ([`DegradationEvent`]) so the
//! current mode — and how the system got there — is always queryable.
//!
//! Axiom I does **not** degrade (doc ch. 5): a side effect still needs a
//! verified capability at every rung. [`DegradationManager::authorize`]
//! therefore takes the capability-verification result as an argument and
//! refuses an unverified side effect before it even looks at the mode.
//! The manager never verifies capabilities itself — that stays with the
//! kernel ([`crate::capabilities::CapabilityAuthority`]); the manager only
//! decides whether the *mode* permits the operation at all.
//!
//! This is deliberately distinct from the precision downgrade that already
//! lives in [`crate::schedulers`] (8bit → 4bit when a budget is overrun):
//! that is one gear *inside* the Full rung, not a step down the ladder.
//! A precision downgrade keeps model calls; a mode downgrade forbids them.
//!
//! The module is deterministic and model-free: triggers are reported by
//! probes (model loader, network monitor, budget ledger, battery sensor)
//! and the mode is a pure function of the set of active triggers. Deciding
//! *why* a model failed or *whether* the network is really down happens
//! outside; once a probe reports, everything below is exact.

use crate::Clock;
use std::collections::BTreeSet;
use std::fmt;
use std::rc::Rc;

/// One rung of the degradation ladder (doc ch. 5, axiom V).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DegradationMode {
    /// Everything works: model calls, macros, deterministic steps, manual
    /// actions.
    Full,
    /// No model calls. Macros and deterministic paths still run, and the
    /// user can still do everything by hand.
    MacroOnly,
    /// No model calls, no macros. Only direct manual interaction — the
    /// traditional way, which must always keep working (doc ch. 6).
    Manual,
}

impl DegradationMode {
    pub fn name(self) -> &'static str {
        match self {
            DegradationMode::Full => "full",
            DegradationMode::MacroOnly => "macro-only",
            DegradationMode::Manual => "manual",
        }
    }

    /// Ladder height: 0 = full, 2 = manual. Recovery climbs back down.
    pub fn level(self) -> u8 {
        match self {
            DegradationMode::Full => 0,
            DegradationMode::MacroOnly => 1,
            DegradationMode::Manual => 2,
        }
    }

    /// Whether an operation of this kind may run in this mode.
    pub fn allows(self, kind: OperationKind) -> bool {
        match (self, kind) {
            (DegradationMode::Full, _) => true,
            (DegradationMode::MacroOnly, OperationKind::ModelCall) => false,
            (DegradationMode::MacroOnly, _) => true,
            (DegradationMode::Manual, OperationKind::ManualAction) => true,
            (DegradationMode::Manual, _) => false,
        }
    }
}

/// What pushed the system down the ladder. Four triggers, per axiom V.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Trigger {
    /// The model (or its files) is unavailable.
    ModelUnavailable,
    /// The network is down — no cloud model, no remote tools.
    Offline,
    /// The interaction budget is exhausted.
    BudgetExceeded,
    /// The battery is critically low.
    BatteryCritical,
}

impl Trigger {
    pub fn name(self) -> &'static str {
        match self {
            Trigger::ModelUnavailable => "model-unavailable",
            Trigger::Offline => "offline",
            Trigger::BudgetExceeded => "budget-exceeded",
            Trigger::BatteryCritical => "battery-critical",
        }
    }

    /// How far down the ladder this trigger pushes on its own. The first
    /// three triggers cost one rung (model calls go away); a critical
    /// battery costs two (macros go away too — running them burns the
    /// power the user needs to keep the machine on at all).
    fn severity(self) -> u8 {
        match self {
            Trigger::BatteryCritical => 2,
            _ => 1,
        }
    }
}

/// A kind of operation the degradation gate can allow or refuse.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationKind {
    /// A model call (planning, generation, understanding).
    ModelCall,
    /// Running a determinized macro.
    MacroRun,
    /// One step of a deterministic plan (type-checked, no model).
    DeterministicStep,
    /// A direct manual action by the user (open a file, type, click).
    ManualAction,
}

impl OperationKind {
    pub fn name(self) -> &'static str {
        match self {
            OperationKind::ModelCall => "model-call",
            OperationKind::MacroRun => "macro-run",
            OperationKind::DeterministicStep => "deterministic-step",
            OperationKind::ManualAction => "manual-action",
        }
    }
}

/// One operation presented to the degradation gate. `side_effect` marks
/// whether the operation changes the outside world; side-effecting
/// operations need a verified capability at every rung (axiom I).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Operation {
    pub kind: OperationKind,
    pub side_effect: bool,
}

impl Operation {
    pub fn model_call() -> Self {
        Operation { kind: OperationKind::ModelCall, side_effect: false }
    }

    pub fn macro_run(side_effect: bool) -> Self {
        Operation { kind: OperationKind::MacroRun, side_effect }
    }

    pub fn deterministic_step(side_effect: bool) -> Self {
        Operation { kind: OperationKind::DeterministicStep, side_effect }
    }

    pub fn manual_action(side_effect: bool) -> Self {
        Operation { kind: OperationKind::ManualAction, side_effect }
    }
}

/// Why the gate refused an operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DegradationError {
    /// The operation is forbidden in the current mode.
    DeniedByMode { mode: DegradationMode, kind: OperationKind },
    /// A side effect without a verified capability. Refused at every rung:
    /// axiom I does not degrade (doc ch. 5).
    CapabilityRequired { kind: OperationKind },
}

impl fmt::Display for DegradationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DegradationError::DeniedByMode { mode, kind } => write!(
                f,
                "{} is not allowed in {} mode",
                kind.name(),
                mode.name()
            ),
            DegradationError::CapabilityRequired { kind } => write!(
                f,
                "{} changes the outside world and needs a verified capability (axiom I)",
                kind.name()
            ),
        }
    }
}

impl std::error::Error for DegradationError {}

/// One entry in the degradation ledger: every mode change, in both
/// directions, is recorded with the time and the trigger behind it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DegradationEvent {
    /// When the transition happened (ms, from the injected clock).
    pub at_ms: u64,
    pub from: DegradationMode,
    pub to: DegradationMode,
    /// The trigger whose report caused this transition, if any.
    pub trigger: Option<Trigger>,
    /// True when the transition climbs back up the ladder.
    pub recovered: bool,
}

/// Tunables for the degradation manager.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DegradationConfig {
    /// Battery percentage at or below which the battery is critical
    /// (default 10). Matches the "battery critical" wording of axiom V;
    /// the speculation gate's 20% "low" threshold in
    /// [`crate::speculation`] is a warning, not a ladder step.
    pub battery_critical_pct: u8,
}

impl Default for DegradationConfig {
    fn default() -> Self {
        DegradationConfig { battery_critical_pct: 10 }
    }
}

/// Outcome of the doc ch. 6 acceptance check: delete all model files and
/// the system must still boot, open files, and complete everyday
/// operations manually.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcceptanceReport {
    /// The system boots without any model.
    pub booted: bool,
    /// A manual file open succeeds.
    pub manual_open_file_ok: bool,
    /// A manual write (a side effect) succeeds once its capability is
    /// verified — axiom I still holds while degraded.
    pub manual_write_with_capability_ok: bool,
    /// Model calls are refused, not retried forever.
    pub model_call_denied: bool,
    /// The mode the system landed in.
    pub mode: DegradationMode,
}

impl AcceptanceReport {
    /// The ch. 6 criterion, as one boolean: usable without a model.
    pub fn passes(&self) -> bool {
        self.booted
            && self.manual_open_file_ok
            && self.manual_write_with_capability_ok
            && self.model_call_denied
    }
}

/// The degradation manager: tracks active triggers, computes the current
/// rung from them, gates operations, and logs every transition.
pub struct DegradationManager {
    clock: Rc<dyn Clock>,
    config: DegradationConfig,
    active: BTreeSet<Trigger>,
    mode: DegradationMode,
    events: Vec<DegradationEvent>,
}

impl DegradationManager {
    pub fn new(clock: Rc<dyn Clock>) -> Self {
        Self::with_config(clock, DegradationConfig::default())
    }

    pub fn with_config(clock: Rc<dyn Clock>, config: DegradationConfig) -> Self {
        DegradationManager {
            clock,
            config,
            active: BTreeSet::new(),
            mode: DegradationMode::Full,
            events: Vec::new(),
        }
    }

    /// The current rung of the ladder.
    pub fn mode(&self) -> DegradationMode {
        self.mode
    }

    /// True once the system has left Full.
    pub fn is_degraded(&self) -> bool {
        self.mode != DegradationMode::Full
    }

    /// The triggers currently holding the system down, most severe first.
    pub fn active_triggers(&self) -> Vec<Trigger> {
        let mut triggers: Vec<Trigger> = self.active.iter().copied().collect();
        triggers.sort_by_key(|t| std::cmp::Reverse(t.severity()));
        triggers
    }

    /// The append-only ledger of every mode change.
    pub fn events(&self) -> &[DegradationEvent] {
        &self.events
    }

    // -- probes: each report flips one trigger and recomputes the mode. --

    /// The model (or its files) went away.
    pub fn report_model_unavailable(&mut self) {
        self.set_trigger(Trigger::ModelUnavailable, true);
    }

    /// The model is back (files restored, service reachable).
    pub fn report_model_restored(&mut self) {
        self.set_trigger(Trigger::ModelUnavailable, false);
    }

    /// The network went down.
    pub fn report_offline(&mut self) {
        self.set_trigger(Trigger::Offline, true);
    }

    /// The network is back.
    pub fn report_online(&mut self) {
        self.set_trigger(Trigger::Offline, false);
    }

    /// The interaction budget is exhausted.
    pub fn report_budget_exceeded(&mut self) {
        self.set_trigger(Trigger::BudgetExceeded, true);
    }

    /// Budget replenished (new window, user topped up).
    pub fn report_budget_replenished(&mut self) {
        self.set_trigger(Trigger::BudgetExceeded, false);
    }

    /// Battery sensor reading. At or below the critical threshold the
    /// battery trigger engages; above it, it clears.
    pub fn report_battery(&mut self, pct: u8) {
        self.set_trigger(Trigger::BatteryCritical, pct <= self.config.battery_critical_pct);
    }

    // -- the gate --

    /// Authorize one operation in the current mode.
    ///
    /// `capability_verified` is the kernel's verdict on the capability
    /// presented for this operation (true when the operation carries no
    /// capability because it has no side effect, or when the kernel
    /// verified it). A side effect without a verified capability is
    /// refused at every rung — axiom I does not degrade.
    pub fn authorize(
        &self,
        op: &Operation,
        capability_verified: bool,
    ) -> Result<(), DegradationError> {
        if op.side_effect && !capability_verified {
            return Err(DegradationError::CapabilityRequired { kind: op.kind });
        }
        if !self.mode.allows(op.kind) {
            return Err(DegradationError::DeniedByMode { mode: self.mode, kind: op.kind });
        }
        Ok(())
    }

    /// The doc ch. 6 acceptance check, run against the current state:
    /// pretend every model file is gone and ask whether the system is
    /// still a usable operating system. The check itself does not change
    /// the mode — callers report the trigger first.
    pub fn acceptance_boot(&self) -> AcceptanceReport {
        let manual_open = Operation::manual_action(false);
        let manual_write = Operation::manual_action(true);
        AcceptanceReport {
            booted: self.authorize(&manual_open, false).is_ok(),
            manual_open_file_ok: self.authorize(&manual_open, false).is_ok(),
            manual_write_with_capability_ok: self
                .authorize(&manual_write, true)
                .is_ok(),
            model_call_denied: self
                .authorize(&Operation::model_call(), false)
                .is_err(),
            mode: self.mode,
        }
    }

    /// Flip one trigger, recompute the mode from the whole active set,
    /// and log the transition if the rung changed. Re-reporting a trigger
    /// that is already in the requested state is a no-op — no duplicate
    /// ledger entries.
    fn set_trigger(&mut self, trigger: Trigger, active: bool) {
        let changed = if active { self.active.insert(trigger) } else { self.active.remove(&trigger) };
        if !changed {
            return;
        }
        let target = self.mode_for(&self.active);
        if target == self.mode {
            return;
        }
        let from = self.mode;
        self.mode = target;
        self.events.push(DegradationEvent {
            at_ms: self.clock.now_ms(),
            from,
            to: target,
            trigger: Some(trigger),
            recovered: target.level() < from.level(),
        });
    }

    /// The rung implied by a trigger set: the most severe active trigger
    /// decides. No trigger → Full; any one-rung trigger → MacroOnly; a
    /// critical battery → Manual.
    fn mode_for(&self, active: &BTreeSet<Trigger>) -> DegradationMode {
        let worst = active.iter().map(|t| t.severity()).max().unwrap_or(0);
        match worst {
            0 => DegradationMode::Full,
            1 => DegradationMode::MacroOnly,
            _ => DegradationMode::Manual,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ManualClock;

    fn manager() -> DegradationManager {
        DegradationManager::new(Rc::new(ManualClock::new(1_000)))
    }

    fn clock_manager() -> (Rc<ManualClock>, DegradationManager) {
        let clock = Rc::new(ManualClock::new(1_000));
        let mgr = DegradationManager::new(clock.clone());
        (clock, mgr)
    }

    #[test]
    fn starts_full_and_allows_everything() {
        let mgr = manager();
        assert_eq!(mgr.mode(), DegradationMode::Full);
        assert!(!mgr.is_degraded());
        assert!(mgr.active_triggers().is_empty());
        for (op, cap) in [
            (Operation::model_call(), false),
            (Operation::macro_run(true), true),
            (Operation::deterministic_step(true), true),
            (Operation::manual_action(true), true),
        ] {
            assert!(mgr.authorize(&op, cap).is_ok(), "{op:?} should pass in Full");
        }
    }

    #[test]
    fn model_unavailable_drops_one_rung_and_denies_model_calls() {
        let mut mgr = manager();
        mgr.report_model_unavailable();
        assert_eq!(mgr.mode(), DegradationMode::MacroOnly);
        assert!(mgr.is_degraded());
        assert_eq!(mgr.active_triggers(), vec![Trigger::ModelUnavailable]);
        assert_eq!(
            mgr.authorize(&Operation::model_call(), false),
            Err(DegradationError::DeniedByMode {
                mode: DegradationMode::MacroOnly,
                kind: OperationKind::ModelCall
            })
        );
        // Macros, deterministic steps and manual actions keep working.
        assert!(mgr.authorize(&Operation::macro_run(false), false).is_ok());
        assert!(mgr.authorize(&Operation::deterministic_step(false), false).is_ok());
        assert!(mgr.authorize(&Operation::manual_action(false), false).is_ok());
    }

    #[test]
    fn offline_and_budget_triggers_each_drop_to_macro_only() {
        let mut mgr = manager();
        mgr.report_offline();
        assert_eq!(mgr.mode(), DegradationMode::MacroOnly);
        mgr.report_online();
        assert_eq!(mgr.mode(), DegradationMode::Full);

        mgr.report_budget_exceeded();
        assert_eq!(mgr.mode(), DegradationMode::MacroOnly);
        mgr.report_budget_replenished();
        assert_eq!(mgr.mode(), DegradationMode::Full);
    }

    #[test]
    fn battery_critical_drops_two_rungs_and_denies_macros() {
        let mut mgr = manager();
        mgr.report_battery(5);
        assert_eq!(mgr.mode(), DegradationMode::Manual);
        assert_eq!(
            mgr.authorize(&Operation::macro_run(false), false),
            Err(DegradationError::DeniedByMode {
                mode: DegradationMode::Manual,
                kind: OperationKind::MacroRun
            })
        );
        assert!(mgr.authorize(&Operation::deterministic_step(false), false).is_err());
        assert!(mgr.authorize(&Operation::model_call(), false).is_err());
        // Manual interaction is the rung that never goes away.
        assert!(mgr.authorize(&Operation::manual_action(false), false).is_ok());
    }

    #[test]
    fn stepwise_degrade_then_stepwise_recover() {
        let mut mgr = manager();
        mgr.report_model_unavailable();
        assert_eq!(mgr.mode(), DegradationMode::MacroOnly);
        mgr.report_battery(8);
        assert_eq!(mgr.mode(), DegradationMode::Manual);
        // Battery recovers first: back to MacroOnly, not straight to Full,
        // because the model is still gone.
        mgr.report_battery(80);
        assert_eq!(mgr.mode(), DegradationMode::MacroOnly);
        mgr.report_model_restored();
        assert_eq!(mgr.mode(), DegradationMode::Full);
        assert!(!mgr.is_degraded());
    }

    #[test]
    fn clearing_one_trigger_keeps_degraded_while_another_holds() {
        let mut mgr = manager();
        mgr.report_offline();
        mgr.report_budget_exceeded();
        assert_eq!(mgr.mode(), DegradationMode::MacroOnly);
        mgr.report_online();
        assert_eq!(mgr.mode(), DegradationMode::MacroOnly);
        assert_eq!(mgr.active_triggers(), vec![Trigger::BudgetExceeded]);
        mgr.report_budget_replenished();
        assert_eq!(mgr.mode(), DegradationMode::Full);
    }

    #[test]
    fn axiom_i_side_effects_still_need_capability_in_every_mode() {
        let mut mgr = manager();
        // Full, MacroOnly and Manual: a side effect without a verified
        // capability is refused everywhere; with one it passes everywhere.
        let modes: Vec<DegradationMode> = vec![
            DegradationMode::Full,
            DegradationMode::MacroOnly,
            DegradationMode::Manual,
        ];
        for (i, mode) in modes.into_iter().enumerate() {
            match i {
                0 => {}
                1 => mgr.report_model_unavailable(),
                _ => mgr.report_battery(5),
            }
            assert_eq!(mgr.mode(), mode);
            let write = Operation::manual_action(true);
            assert_eq!(
                mgr.authorize(&write, false),
                Err(DegradationError::CapabilityRequired { kind: OperationKind::ManualAction })
            );
            assert!(mgr.authorize(&write, true).is_ok());
            let macro_write = Operation::macro_run(true);
            if mode.allows(OperationKind::MacroRun) {
                assert_eq!(
                    mgr.authorize(&macro_write, false),
                    Err(DegradationError::CapabilityRequired { kind: OperationKind::MacroRun })
                );
                assert!(mgr.authorize(&macro_write, true).is_ok());
            }
        }
    }

    #[test]
    fn ch6_acceptance_boot_without_models() {
        let mut mgr = manager();
        // Delete all model files: the model probe reports unavailable.
        mgr.report_model_unavailable();
        let report = mgr.acceptance_boot();
        assert!(report.passes(), "{report:?}");
        assert_eq!(report.mode, DegradationMode::MacroOnly);
        // And even at the bottom rung the system still boots manually.
        mgr.report_battery(3);
        let bottom = mgr.acceptance_boot();
        assert!(bottom.passes(), "{bottom:?}");
        assert_eq!(bottom.mode, DegradationMode::Manual);
    }

    #[test]
    fn acceptance_fails_in_full_because_model_calls_are_allowed() {
        // In Full mode the ch. 6 check is vacuous: a model call is allowed,
        // so "still usable without a model" is not demonstrated. The
        // report must say so rather than claim a pass.
        let mgr = manager();
        let report = mgr.acceptance_boot();
        assert!(!report.passes());
        assert!(!report.model_call_denied);
    }

    #[test]
    fn ledger_records_degrade_and_recover_with_timestamps() {
        let (clock, mut mgr) = clock_manager();
        mgr.report_model_unavailable();
        clock.advance(500);
        mgr.report_battery(5);
        clock.advance(250);
        mgr.report_battery(90);
        assert_eq!(mgr.events().len(), 3);
        let events = mgr.events();
        assert_eq!(
            events[0],
            DegradationEvent {
                at_ms: 1_000,
                from: DegradationMode::Full,
                to: DegradationMode::MacroOnly,
                trigger: Some(Trigger::ModelUnavailable),
                recovered: false,
            }
        );
        assert_eq!(events[1].at_ms, 1_500);
        assert_eq!(events[1].from, DegradationMode::MacroOnly);
        assert_eq!(events[1].to, DegradationMode::Manual);
        assert!(!events[1].recovered);
        assert_eq!(events[2].at_ms, 1_750);
        assert_eq!(events[2].to, DegradationMode::MacroOnly);
        assert!(events[2].recovered);
        assert_eq!(events[2].trigger, Some(Trigger::BatteryCritical));
    }

    #[test]
    fn re_reporting_a_trigger_adds_no_duplicate_event() {
        let mut mgr = manager();
        mgr.report_model_unavailable();
        mgr.report_model_unavailable();
        mgr.report_offline();
        // Offline is a second trigger but the rung does not change.
        assert_eq!(mgr.events().len(), 1);
        assert_eq!(mgr.mode(), DegradationMode::MacroOnly);
        mgr.report_online();
        // Still degraded on the budget of the remaining trigger: no event.
        assert_eq!(mgr.events().len(), 1);
    }

    #[test]
    fn battery_threshold_is_configurable() {
        let clock = Rc::new(ManualClock::new(0));
        let mut mgr = DegradationManager::with_config(
            clock,
            DegradationConfig { battery_critical_pct: 20 },
        );
        mgr.report_battery(15);
        assert_eq!(mgr.mode(), DegradationMode::Manual);
        mgr.report_battery(21);
        assert_eq!(mgr.mode(), DegradationMode::Full);
    }

    #[test]
    fn battery_above_threshold_never_triggers() {
        let mut mgr = manager();
        mgr.report_battery(11);
        assert_eq!(mgr.mode(), DegradationMode::Full);
        mgr.report_battery(10);
        assert_eq!(mgr.mode(), DegradationMode::Manual);
    }
}
