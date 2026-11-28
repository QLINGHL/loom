//! Robot safety monitor: e-stop and collision interception (doc ch. 24,
//! first embodied-AI landing form, P0 item queued 2026-11-27).
//!
//! The strategy decision in `docs/strategy-first-device.md` made this a
//! first-class citizen: **interruption and e-stop outrank the model** — the
//! e-stop path never goes through the model and never queues behind it.
//! This module is the P0 implementation of that decision:
//!
//! * **Latching e-stop.** [`SafetyMonitor::trigger_estop`] latches the
//!   monitor; while latched, no physical action is dispatched. A latched
//!   monitor only ever unlatches through an explicit human reset — the
//!   model, a timer, or any automatic recovery cannot reset it.
//! * **Held, never auto-run.** Actions submitted while latched are parked
//!   in a held queue. Resetting the e-stop does *not* release them: each
//!   held action needs an explicit re-dispatch
//!   ([`SafetyMonitor::dispatch_held`]), so nothing the robot was about to
//!   do sneaks out after a reset.
//! * **Collision auto-latches.** A collision sensor event whose severity
//!   meets the threshold triggers the e-stop the same way a manual trigger
//!   does — no model call, no queue, no confirmation dialog in between.
//! * **Auditable.** Every trigger and every reset is recorded in the
//!   decision trace ([`crate::trace::DecisionTrace`],
//!   [`crate::trace::DecisionKind::Safety`]) so the safety history is
//!   replayable and tamper-evident like any other decision.
//!
//! What this module deliberately does **not** do:
//!
//! * It does not drive the chassis or any real-time motion-control loop
//!   (ch. 24 P0 excludes that layer); it is a pure gate in front of
//!   dispatch, like the transaction manager is a pure gate in front of
//!   `act()`.
//! * It does not invent its own reversibility rules. Physical actions are
//!   judged with the existing [`Compensability`] three-tier scale
//!   (`typesys.rs`): reversible / windowed actions dispatch directly, and
//!   irreversible ones are routed to the existing cooling-off staging in
//!   `transactions.rs` ([`crate::transactions::TxnManager::stage_irreversible`])
//!   instead of being dispatched — the caller stages them there; this
//!   module only classifies.
//!
//! TCB note (ch. 24 P2): this monitor is the named candidate for inclusion
//! in the trusted computing base at formal-proof strength. This round is
//! the P0 implementation only — **no formal verification is claimed**; the
//! module is written so the latch invariant (latched ⇒ nothing dispatches)
//! is a small, checkable property for that later work.

use crate::trace::{DecisionKind, DecisionTrace, ModelParams};
use crate::typesys::Compensability;

/// A physical action the robot is about to perform: moving an arm, driving
/// a wheel, closing a gripper. The monitor gates dispatch; the kernel (or
/// the chassis driver below it) performs the motion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalAction {
    /// Stable id, so a held action can be re-dispatched by reference.
    pub id: u64,
    /// Short human-readable label, e.g. "drive forward 0.5 m".
    pub description: String,
    /// Reversibility tier, reused from `typesys.rs` — physical actions are
    /// `Irreversible` by default in the physical world, and the tier is
    /// what decides whether cooling-off staging applies.
    pub compensability: Compensability,
}

impl PhysicalAction {
    pub fn new(id: u64, description: impl Into<String>, compensability: Compensability) -> Self {
        PhysicalAction { id, description: description.into(), compensability }
    }
}

/// Who is asking to reset a latched e-stop. Only [`ResetAuthority::Human`]
/// is accepted; everything else is refused by construction — the model
/// cannot clear its own safety interlock, and no timer or auto-recovery
/// may either.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResetAuthority {
    /// An explicit human operator reset, e.g. pressing the reset button
    /// after inspecting the scene.
    Human { operator: String },
    /// The model asking for a reset (refused).
    Model,
    /// A timer / watchdog expiry asking for a reset (refused).
    Timer,
    /// Any automatic recovery path asking for a reset (refused).
    Automatic,
}

impl ResetAuthority {
    fn label(&self) -> &'static str {
        match self {
            ResetAuthority::Human { .. } => "human",
            ResetAuthority::Model => "model",
            ResetAuthority::Timer => "timer",
            ResetAuthority::Automatic => "automatic",
        }
    }
}

/// A collision sensor event: which sensor fired and how hard the impact
/// was. Any event at or above the monitor's threshold auto-triggers the
/// e-stop.
#[derive(Debug, Clone, PartialEq)]
pub struct CollisionEvent {
    pub sensor: String,
    /// Impact severity in the sensor's own units; compared against the
    /// monitor's threshold (default 0.0: any reported contact latches).
    pub severity: f64,
}

impl CollisionEvent {
    pub fn new(sensor: impl Into<String>, severity: f64) -> Self {
        CollisionEvent { sensor: sensor.into(), severity }
    }
}

/// What happened to a submitted physical action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitOutcome {
    /// Latch clear and the action is reversible (or windowed): approved
    /// for dispatch.
    Dispatched { action_id: u64 },
    /// Latch clear but the action is irreversible: do not dispatch
    /// directly — stage it through the transactions cooling-off path
    /// (`TxnManager::stage_irreversible`) first.
    StagedForCoolingOff { action_id: u64 },
    /// E-stop latched: the action was parked in the held queue and was
    /// *not* dispatched. It stays held until explicitly re-dispatched.
    HeldByEStop { action_id: u64 },
}

impl SubmitOutcome {
    pub fn action_id(self) -> u64 {
        match self {
            SubmitOutcome::Dispatched { action_id }
            | SubmitOutcome::StagedForCoolingOff { action_id }
            | SubmitOutcome::HeldByEStop { action_id } => action_id,
        }
    }
}

/// Errors from the safety monitor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SafetyError {
    /// A reset was requested while the e-stop is not latched.
    NotLatched,
    /// A non-human reset was attempted. The monitor stays latched.
    ResetForbidden { attempted_by: &'static str },
}

impl std::fmt::Display for SafetyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SafetyError::NotLatched => write!(f, "e-stop is not latched; nothing to reset"),
            SafetyError::ResetForbidden { attempted_by } => write!(
                f,
                "e-stop reset refused: only an explicit human reset may clear it (attempted by {attempted_by})"
            ),
        }
    }
}
impl std::error::Error for SafetyError {}

/// Latching e-stop + collision interception gate for physical actions.
///
/// Pure state machine: no I/O, no model calls, no threads. Time is passed
/// in as `at_ms` so tests can drive it deterministically.
#[derive(Debug, Default)]
pub struct SafetyMonitor {
    latched: bool,
    latched_reason: Option<String>,
    latched_since_ms: Option<u64>,
    /// Severity at or above which a collision event latches the e-stop.
    /// Default 0.0: any reported contact latches.
    collision_threshold: f64,
    held: Vec<PhysicalAction>,
    trigger_count: u64,
    reset_count: u64,
}

impl SafetyMonitor {
    pub fn new() -> Self {
        SafetyMonitor::default()
    }

    /// Set the collision severity threshold (default 0.0). Events below
    /// the threshold are ignored; events at or above it latch the e-stop.
    pub fn with_collision_threshold(mut self, threshold: f64) -> Self {
        self.collision_threshold = threshold;
        self
    }

    pub fn is_latched(&self) -> bool {
        self.latched
    }

    /// Why the e-stop latched, if it is latched.
    pub fn latched_reason(&self) -> Option<&str> {
        self.latched_reason.as_deref()
    }

    pub fn latched_since_ms(&self) -> Option<u64> {
        self.latched_since_ms
    }

    /// How many times the e-stop has latched (repeat triggers while
    /// already latched do not count — see [`SafetyMonitor::trigger_estop`]).
    pub fn trigger_count(&self) -> u64 {
        self.trigger_count
    }

    pub fn reset_count(&self) -> u64 {
        self.reset_count
    }

    /// Actions parked by the e-stop, in submission order. They are held,
    /// not dispatched, and a reset does not release them.
    pub fn held_actions(&self) -> &[PhysicalAction] {
        &self.held
    }

    pub fn held_count(&self) -> usize {
        self.held.len()
    }

    /// Submit a physical action for dispatch.
    ///
    /// * Latched → the action is parked in the held queue and
    ///   [`SubmitOutcome::HeldByEStop`] is returned. Nothing is dispatched.
    /// * Clear + reversible/windowed → [`SubmitOutcome::Dispatched`].
    /// * Clear + irreversible → [`SubmitOutcome::StagedForCoolingOff`]:
    ///   the caller must stage it via `TxnManager::stage_irreversible`
    ///   (cooling-off, `transactions.rs`); the monitor never dispatches an
    ///   irreversible physical action directly.
    pub fn submit(&mut self, action: PhysicalAction) -> SubmitOutcome {
        if self.latched {
            let id = action.id;
            self.held.push(action);
            return SubmitOutcome::HeldByEStop { action_id: id };
        }
        match action.compensability {
            Compensability::Irreversible => {
                SubmitOutcome::StagedForCoolingOff { action_id: action.id }
            }
            _ => SubmitOutcome::Dispatched { action_id: action.id },
        }
    }

    /// Explicitly re-dispatch one held action after a reset. Returns
    /// `None` if no held action has that id. If the e-stop has latched
    /// again since, the action stays held and
    /// [`SubmitOutcome::HeldByEStop`] is returned — a reset never
    /// auto-releases held actions, and neither does this call while
    /// latched.
    pub fn dispatch_held(&mut self, action_id: u64) -> Option<SubmitOutcome> {
        let pos = self.held.iter().position(|a| a.id == action_id)?;
        if self.latched {
            return Some(SubmitOutcome::HeldByEStop { action_id });
        }
        let action = self.held.remove(pos);
        Some(match action.compensability {
            Compensability::Irreversible => {
                SubmitOutcome::StagedForCoolingOff { action_id: action.id }
            }
            _ => SubmitOutcome::Dispatched { action_id: action.id },
        })
    }

    /// Drop one held action without dispatching it (e.g. the operator
    /// discards a stale motion after inspecting the scene). Returns false
    /// if no held action has that id.
    pub fn drop_held(&mut self, action_id: u64) -> bool {
        match self.held.iter().position(|a| a.id == action_id) {
            Some(pos) => {
                self.held.remove(pos);
                true
            }
            None => false,
        }
    }

    /// Trigger the e-stop. Returns `true` if this call newly latched the
    /// monitor, `false` if it was already latched — repeat triggers are
    /// idempotent: the original reason and latch time stand, and no
    /// duplicate audit entry is written.
    ///
    /// The trigger is recorded in the decision trace
    /// (`DecisionKind::Safety`) so the safety history is auditable. The
    /// e-stop path never touches the model: the trace entry carries a
    /// fixed `safety-monitor` model label only because the trace schema
    /// carries one.
    pub fn trigger_estop(
        &mut self,
        reason: impl Into<String>,
        at_ms: u64,
        trace: &mut DecisionTrace,
    ) -> bool {
        if self.latched {
            return false;
        }
        let reason = reason.into();
        self.latched = true;
        self.latched_reason = Some(reason.clone());
        self.latched_since_ms = Some(at_ms);
        self.trigger_count += 1;
        record_safety_event(
            trace,
            at_ms,
            format!("estop latched: {reason}"),
            format!("estop:{reason}"),
            "latched",
        );
        true
    }

    /// Handle a collision sensor event. Events at or above the threshold
    /// auto-trigger the e-stop (same latching semantics as
    /// [`SafetyMonitor::trigger_estop`]); below-threshold events are
    /// ignored. Returns `true` only if this event newly latched the
    /// monitor.
    pub fn on_collision(
        &mut self,
        event: CollisionEvent,
        at_ms: u64,
        trace: &mut DecisionTrace,
    ) -> bool {
        if event.severity < self.collision_threshold {
            return false;
        }
        self.trigger_estop(
            format!("collision: sensor {} severity {}", event.sensor, event.severity),
            at_ms,
            trace,
        )
    }

    /// Reset a latched e-stop. Only an explicit human reset is accepted;
    /// model, timer and automatic resets are refused and the monitor stays
    /// latched. Resetting does not release held actions — they stay held
    /// until explicitly re-dispatched via [`SafetyMonitor::dispatch_held`].
    /// The reset is recorded in the decision trace.
    pub fn reset_estop(
        &mut self,
        authority: ResetAuthority,
        at_ms: u64,
        trace: &mut DecisionTrace,
    ) -> Result<(), SafetyError> {
        if !self.latched {
            return Err(SafetyError::NotLatched);
        }
        let ResetAuthority::Human { operator } = authority else {
            return Err(SafetyError::ResetForbidden { attempted_by: authority.label() });
        };
        self.latched = false;
        self.latched_reason = None;
        self.latched_since_ms = None;
        self.reset_count += 1;
        record_safety_event(
            trace,
            at_ms,
            format!("estop reset by human operator {operator}"),
            format!("estop-reset:{operator}"),
            "reset",
        );
        Ok(())
    }
}

/// Record one safety event in the decision trace. The model label is a
/// fixed placeholder: the e-stop path is deterministic and never consults
/// a model.
fn record_safety_event(
    trace: &mut DecisionTrace,
    at_ms: u64,
    summary: String,
    input: String,
    output: &str,
) {
    trace.record(
        at_ms,
        "safety-monitor",
        DecisionKind::Safety,
        0,
        summary,
        &input,
        ModelParams::new("safety-monitor", 0.0, 0.0),
        output,
        None,
        None,
        vec![],
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transactions::{compensation_status, AppliedStep, CompStatus};
    use crate::Value;

    fn drive(id: u64) -> PhysicalAction {
        // Driving is irreversible in the physical world: the wheels moved.
        PhysicalAction::new(id, format!("drive forward {id} m"), Compensability::Irreversible)
    }

    fn grip(id: u64) -> PhysicalAction {
        // Gripper close is undoable inside a window (re-open).
        PhysicalAction::new(
            id,
            format!("close gripper {id}"),
            Compensability::Partial { window_ms: 5_000 },
        )
    }

    fn human() -> ResetAuthority {
        ResetAuthority::Human { operator: "op-1".into() }
    }

    #[test]
    fn estop_latch_blocks_dispatch_and_parks_action() {
        let mut trace = DecisionTrace::new();
        let mut m = SafetyMonitor::new();
        assert_eq!(m.submit(grip(1)), SubmitOutcome::Dispatched { action_id: 1 });
        assert!(m.trigger_estop("operator pressed e-stop", 100, &mut trace));
        assert!(m.is_latched());
        assert_eq!(m.latched_reason(), Some("operator pressed e-stop"));
        assert_eq!(m.latched_since_ms(), Some(100));
        // While latched, nothing dispatches — the action is only held.
        assert_eq!(m.submit(grip(2)), SubmitOutcome::HeldByEStop { action_id: 2 });
        assert_eq!(m.submit(drive(3)), SubmitOutcome::HeldByEStop { action_id: 3 });
        assert_eq!(m.held_count(), 2);
        assert_eq!(m.held_actions()[0].id, 2);
    }

    #[test]
    fn held_actions_do_not_auto_run_after_reset() {
        let mut trace = DecisionTrace::new();
        let mut m = SafetyMonitor::new();
        m.trigger_estop("bump", 100, &mut trace);
        assert_eq!(m.submit(grip(1)), SubmitOutcome::HeldByEStop { action_id: 1 });
        m.reset_estop(human(), 200, &mut trace).unwrap();
        assert!(!m.is_latched());
        // Reset alone releases nothing: the action is still held.
        assert_eq!(m.held_count(), 1);
        // New actions dispatch again, but the held one stays parked.
        assert_eq!(m.submit(grip(2)), SubmitOutcome::Dispatched { action_id: 2 });
        assert_eq!(m.held_count(), 1);
        // Only an explicit re-dispatch releases it.
        assert_eq!(m.dispatch_held(1), Some(SubmitOutcome::Dispatched { action_id: 1 }));
        assert_eq!(m.held_count(), 0);
        assert_eq!(m.dispatch_held(1), None, "already released");
    }

    #[test]
    fn dispatch_held_while_relatched_stays_held() {
        let mut trace = DecisionTrace::new();
        let mut m = SafetyMonitor::new();
        m.trigger_estop("first", 100, &mut trace);
        m.submit(grip(1));
        m.reset_estop(human(), 200, &mut trace).unwrap();
        m.trigger_estop("second", 300, &mut trace);
        assert_eq!(m.dispatch_held(1), Some(SubmitOutcome::HeldByEStop { action_id: 1 }));
        assert_eq!(m.held_count(), 1, "still parked, not dispatched");
    }

    #[test]
    fn collision_event_auto_latches_estop() {
        let mut trace = DecisionTrace::new();
        let mut m = SafetyMonitor::new();
        assert!(m.on_collision(CollisionEvent::new("bumper-front", 1.0), 100, &mut trace));
        assert!(m.is_latched());
        assert!(m.latched_reason().unwrap().contains("bumper-front"));
        assert_eq!(m.submit(grip(1)), SubmitOutcome::HeldByEStop { action_id: 1 });
    }

    #[test]
    fn collision_below_threshold_is_ignored() {
        let mut trace = DecisionTrace::new();
        let mut m = SafetyMonitor::new().with_collision_threshold(2.0);
        assert!(!m.on_collision(CollisionEvent::new("bumper-front", 1.5), 100, &mut trace));
        assert!(!m.is_latched());
        assert_eq!(m.submit(grip(1)), SubmitOutcome::Dispatched { action_id: 1 });
        assert!(m.on_collision(CollisionEvent::new("bumper-front", 2.0), 200, &mut trace));
        assert!(m.is_latched(), "at-threshold event latches");
    }

    #[test]
    fn repeat_trigger_is_idempotent() {
        let mut trace = DecisionTrace::new();
        let mut m = SafetyMonitor::new();
        assert!(m.trigger_estop("first", 100, &mut trace));
        assert!(!m.trigger_estop("second", 200, &mut trace), "already latched");
        assert!(!m.on_collision(CollisionEvent::new("bumper", 9.0), 300, &mut trace));
        assert_eq!(m.trigger_count(), 1);
        assert_eq!(m.latched_reason(), Some("first"), "first reason stands");
        assert_eq!(m.latched_since_ms(), Some(100));
        assert_eq!(trace.len(), 1, "no duplicate audit entry");
    }

    #[test]
    fn only_explicit_human_reset_clears_the_latch() {
        let mut trace = DecisionTrace::new();
        let mut m = SafetyMonitor::new();
        m.trigger_estop("bump", 100, &mut trace);
        for authority in
            [ResetAuthority::Model, ResetAuthority::Timer, ResetAuthority::Automatic]
        {
            let label = authority.label();
            let err = m.reset_estop(authority, 200, &mut trace).unwrap_err();
            assert_eq!(err, SafetyError::ResetForbidden { attempted_by: label });
            assert!(m.is_latched(), "still latched after {label} reset attempt");
        }
        assert_eq!(m.reset_count(), 0);
        m.reset_estop(human(), 300, &mut trace).unwrap();
        assert!(!m.is_latched());
        assert_eq!(m.reset_count(), 1);
    }

    #[test]
    fn reset_when_clear_is_an_error() {
        let mut trace = DecisionTrace::new();
        let mut m = SafetyMonitor::new();
        assert_eq!(m.reset_estop(human(), 100, &mut trace), Err(SafetyError::NotLatched));
        assert_eq!(trace.len(), 0, "failed reset leaves no audit entry");
    }

    #[test]
    fn trigger_and_reset_leave_auditable_trace_entries() {
        let mut trace = DecisionTrace::new();
        let mut m = SafetyMonitor::new();
        m.trigger_estop("operator pressed e-stop", 100, &mut trace);
        m.reset_estop(human(), 200, &mut trace).unwrap();
        assert_eq!(trace.len(), 2);
        assert!(trace.verify_chain(), "hash chain intact");
        let trigger = trace.point(0).unwrap();
        assert_eq!(trigger.kind, DecisionKind::Safety);
        assert_eq!(trigger.agent, "safety-monitor");
        assert!(trigger.summary.contains("estop latched"));
        let reset = trace.point(1).unwrap();
        assert_eq!(reset.kind, DecisionKind::Safety);
        assert!(reset.summary.contains("estop reset by human"));
        assert!(reset.summary.contains("op-1"));
        // The safety history is replayable like any other decision.
        let view = trace.replay_at(0).unwrap();
        assert_eq!(view.point.seq, 0);
    }

    #[test]
    fn irreversible_physical_action_routes_to_cooling_off_not_dispatch() {
        let mut m = SafetyMonitor::new();
        // Reversible/windowed actions dispatch directly.
        assert_eq!(m.submit(grip(1)), SubmitOutcome::Dispatched { action_id: 1 });
        // Irreversible physical actions never dispatch directly: they go
        // to the existing cooling-off staging in transactions.rs.
        assert_eq!(
            m.submit(drive(2)),
            SubmitOutcome::StagedForCoolingOff { action_id: 2 }
        );
        // The reuse is real, not just a label: the transactions layer
        // agrees an irreversible step can never be compensated.
        let step = AppliedStep {
            tool: "drive".into(),
            resource: "chassis".into(),
            args: Value::text("forward 0.5 m"),
            result: Value::text("moved"),
            tier: Compensability::Irreversible,
            applied_ms: 100,
            idempotency_key: "drive-2".into(),
        };
        assert_eq!(compensation_status(&step, 10_000), CompStatus::Never);
        let reversible = AppliedStep { tier: Compensability::Full, ..step.clone() };
        assert_eq!(compensation_status(&reversible, 10_000), CompStatus::Yes);
    }

    #[test]
    fn drop_held_discards_without_dispatch() {
        let mut trace = DecisionTrace::new();
        let mut m = SafetyMonitor::new();
        m.trigger_estop("bump", 100, &mut trace);
        m.submit(grip(1));
        m.submit(grip(2));
        assert!(m.drop_held(1));
        assert!(!m.drop_held(1), "already dropped");
        assert_eq!(m.held_count(), 1);
        assert_eq!(m.held_actions()[0].id, 2);
    }

    #[test]
    fn error_messages_are_human_readable() {
        assert_eq!(
            SafetyError::NotLatched.to_string(),
            "e-stop is not latched; nothing to reset"
        );
        assert!(SafetyError::ResetForbidden { attempted_by: "model" }
            .to_string()
            .contains("only an explicit human reset"));
    }
}
