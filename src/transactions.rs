//! Effect transactions (doc ch. 15): every `act()` runs inside a
//! transaction carrying three pieces of metadata — an idempotency key, a
//! compensation, and preconditions.
//!
//! The manager is a pure state machine: it tracks what was applied, what is
//! staged, and what may be compensated; the *kernel* performs the actual
//! dispatch and compensation through the tool registry. Key semantics:
//!
//! * **Idempotency.** A retried step with an already-seen key returns the
//!   recorded result without re-executing (agent retry loops cannot
//!   double-order).
//! * **Saga compensation.** Aborting runs compensations in exact reverse
//!   order of application.
//! * **Cooling-off.** Irreversible effects are never dispatched at `act()`
//!   time; they are *staged*. Commit before the cooling deadline requires
//!   an explicit user confirmation; cancelling during the window drops the
//!   staged effect — nothing ever reached the world.
//! * **`undo` is a primitive.** A committed transaction can be undone
//!   later; the manager yields its compensation plan, marking each step's
//!   compensability status (full / window-expired / never).

use crate::typesys::{Compensability, Condition, Facts};
use crate::Value;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxnState {
    Open,
    /// At least one irreversible effect is staged, awaiting the cooling-off
    /// deadline or an explicit confirmation.
    PendingCoolingOff,
    Committed,
    Aborted,
    Undone,
}

/// A step whose effect has been dispatched to the world.
#[derive(Debug, Clone)]
pub struct AppliedStep {
    pub tool: String,
    pub resource: String,
    pub args: Value,
    pub result: Value,
    pub tier: Compensability,
    pub applied_ms: u64,
    pub idempotency_key: String,
}

/// An irreversible step withheld until commit (the cooling-off staging).
#[derive(Debug, Clone)]
pub struct StagedStep {
    pub tool: String,
    pub resource: String,
    pub args: Value,
    pub idempotency_key: String,
}

#[derive(Debug, Clone)]
pub struct Txn {
    pub id: u64,
    pub agent: String,
    pub state: TxnState,
    pub applied: Vec<AppliedStep>,
    pub staged: Vec<StagedStep>,
    pub preconditions: Vec<Condition>,
    pub created_ms: u64,
    pub cooling_deadline_ms: Option<u64>,
}

/// Whether a step can still be compensated *now*.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompStatus {
    Yes,
    WindowExpired,
    Never,
}

pub fn compensation_status(step: &AppliedStep, now_ms: u64) -> CompStatus {
    match step.tier {
        Compensability::Full => CompStatus::Yes,
        Compensability::Partial { window_ms } => {
            if now_ms.saturating_sub(step.applied_ms) <= window_ms {
                CompStatus::Yes
            } else {
                CompStatus::WindowExpired
            }
        }
        Compensability::Irreversible => CompStatus::Never,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TxnError {
    NotFound { id: u64 },
    WrongState { id: u64, state: TxnState },
    PreconditionFailed { name: String },
    CoolingOffActive { remaining_ms: u64 },
}

impl std::fmt::Display for TxnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TxnError::NotFound { id } => write!(f, "transaction {id} not found"),
            TxnError::WrongState { id, state } => write!(f, "transaction {id} is in state {state:?}"),
            TxnError::PreconditionFailed { name } => {
                write!(f, "precondition failed: {name} (transaction never started)")
            }
            TxnError::CoolingOffActive { remaining_ms } => {
                write!(f, "irreversible effect is in its cooling-off window ({remaining_ms} ms left)")
            }
        }
    }
}
impl std::error::Error for TxnError {}

struct IdemRecord {
    result: Value,
}

pub struct TxnManager {
    txns: HashMap<u64, Txn>,
    next_id: u64,
    idem: HashMap<String, IdemRecord>,
}

impl TxnManager {
    pub fn new() -> Self {
        TxnManager { txns: HashMap::new(), next_id: 1, idem: HashMap::new() }
    }

    pub fn begin(&mut self, agent: &str, preconditions: Vec<Condition>, now_ms: u64) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.txns.insert(
            id,
            Txn {
                id,
                agent: agent.to_string(),
                state: TxnState::Open,
                applied: Vec::new(),
                staged: Vec::new(),
                preconditions,
                created_ms: now_ms,
                cooling_deadline_ms: None,
            },
        );
        id
    }

    pub fn get(&self, id: u64) -> Option<&Txn> {
        self.txns.get(&id)
    }

    pub fn state(&self, id: u64) -> Option<TxnState> {
        self.txns.get(&id).map(|t| t.state)
    }

    /// Preconditions are checked *before* anything executes: if the world
    /// is not in the required state, the transaction never starts (doc
    /// ch. 15) — it does not fail halfway.
    pub fn check_preconditions(&self, id: u64, facts: &dyn Facts) -> Result<(), TxnError> {
        let txn = self.txns.get(&id).ok_or(TxnError::NotFound { id })?;
        for c in &txn.preconditions {
            if !c.eval(facts) {
                return Err(TxnError::PreconditionFailed { name: c.name.clone() });
            }
        }
        Ok(())
    }

    /// Result of an already-executed idempotency key, if any. Callers must
    /// consult this before dispatching a step.
    pub fn lookup_idempotent(&self, key: &str) -> Option<Value> {
        self.idem.get(key).map(|r| r.result.clone())
    }

    /// Record a freshly dispatched step (and its idempotency key).
    pub fn record_step(&mut self, id: u64, step: AppliedStep) -> Result<(), TxnError> {
        self.idem.insert(step.idempotency_key.clone(), IdemRecord { result: step.result.clone() });
        let txn = self.txns.get_mut(&id).ok_or(TxnError::NotFound { id })?;
        txn.applied.push(step);
        Ok(())
    }

    /// Stage an irreversible step instead of dispatching it, and put the
    /// transaction into its cooling-off window.
    pub fn stage_irreversible(
        &mut self,
        id: u64,
        step: StagedStep,
        cooling_ms: u64,
        now_ms: u64,
    ) -> Result<(), TxnError> {
        let txn = self.txns.get_mut(&id).ok_or(TxnError::NotFound { id })?;
        if txn.state != TxnState::Open && txn.state != TxnState::PendingCoolingOff {
            return Err(TxnError::WrongState { id, state: txn.state });
        }
        txn.staged.push(step);
        txn.state = TxnState::PendingCoolingOff;
        let deadline = now_ms + cooling_ms;
        txn.cooling_deadline_ms = Some(match txn.cooling_deadline_ms {
            Some(d) => d.max(deadline),
            None => deadline,
        });
        Ok(())
    }

    pub fn cooling_remaining_ms(&self, id: u64, now_ms: u64) -> Option<u64> {
        self.txns.get(&id).and_then(|t| {
            t.cooling_deadline_ms.map(|d| d.saturating_sub(now_ms))
        })
    }

    /// First half of commit: release staged irreversible steps for dispatch.
    /// Without `user_confirmed`, the cooling-off deadline must have passed.
    pub fn begin_commit(
        &mut self,
        id: u64,
        now_ms: u64,
        user_confirmed: bool,
    ) -> Result<Vec<StagedStep>, TxnError> {
        let txn = self.txns.get_mut(&id).ok_or(TxnError::NotFound { id })?;
        match txn.state {
            TxnState::Open | TxnState::PendingCoolingOff => {}
            s => return Err(TxnError::WrongState { id, state: s }),
        }
        if !txn.staged.is_empty() && !user_confirmed {
            let remaining = txn.cooling_deadline_ms.map(|d| d.saturating_sub(now_ms)).unwrap_or(0);
            if remaining > 0 {
                return Err(TxnError::CoolingOffActive { remaining_ms: remaining });
            }
        }
        Ok(std::mem::take(&mut txn.staged))
    }

    /// Second half of commit: record the dispatched staged steps and seal
    /// the transaction.
    pub fn finish_commit(&mut self, id: u64, dispatched: Vec<AppliedStep>) -> Result<(), TxnError> {
        for step in dispatched {
            self.record_step(id, step)?;
        }
        let txn = self.txns.get_mut(&id).ok_or(TxnError::NotFound { id })?;
        txn.state = TxnState::Committed;
        txn.cooling_deadline_ms = None;
        Ok(())
    }

    /// Abort: returns the applied steps so the caller can compensate them
    /// in reverse order. Staged (never dispatched) steps are dropped —
    /// cancelling inside the cooling-off window is free.
    pub fn abort(&mut self, id: u64) -> Result<Vec<AppliedStep>, TxnError> {
        let txn = self.txns.get_mut(&id).ok_or(TxnError::NotFound { id })?;
        match txn.state {
            TxnState::Open | TxnState::PendingCoolingOff => {}
            s => return Err(TxnError::WrongState { id, state: s }),
        }
        txn.state = TxnState::Aborted;
        txn.staged.clear();
        txn.cooling_deadline_ms = None;
        Ok(txn.applied.clone())
    }

    /// The compensation plan for undoing a committed transaction: applied
    /// steps in exact reverse order of application.
    pub fn undo_plan(&self, id: u64) -> Result<Vec<AppliedStep>, TxnError> {
        let txn = self.txns.get(&id).ok_or(TxnError::NotFound { id })?;
        if txn.state != TxnState::Committed {
            return Err(TxnError::WrongState { id, state: txn.state });
        }
        let mut steps = txn.applied.clone();
        steps.reverse();
        Ok(steps)
    }

    pub fn mark_undone(&mut self, id: u64) -> Result<(), TxnError> {
        let txn = self.txns.get_mut(&id).ok_or(TxnError::NotFound { id })?;
        txn.state = TxnState::Undone;
        Ok(())
    }
}

impl Default for TxnManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typesys::{FactBase, Predicate};

    /// A tiny fake world the tests execute against, standing in for the
    /// kernel+registry dispatch: applying a step appends to `log` and
    /// mutates `balances`; compensating reverses it.
    #[derive(Default)]
    struct FakeWorld {
        balances: HashMap<String, i64>,
        log: Vec<String>,
        apply_count: u32,
    }

    impl FakeWorld {
        fn apply(&mut self, step: &AppliedStep) {
            self.apply_count += 1;
            self.log.push(format!("apply:{}", step.tool));
            if step.tool == "bank.transfer" {
                let amt = step.args.get("amount").and_then(|v| v.as_int()).unwrap_or(0);
                *self.balances.entry(step.resource.clone()).or_insert(1_000) -= amt;
            }
        }
        fn compensate(&mut self, step: &AppliedStep) -> Result<(), String> {
            self.log.push(format!("undo:{}", step.tool));
            if step.tool == "bank.transfer" {
                let amt = step.args.get("amount").and_then(|v| v.as_int()).unwrap_or(0);
                *self.balances.entry(step.resource.clone()).or_insert(1_000) += amt;
            }
            Ok(())
        }
    }

    fn step(tool: &str, tier: Compensability, key: &str, now: u64) -> AppliedStep {
        AppliedStep {
            tool: tool.to_string(),
            resource: "acct".to_string(),
            args: Value::map(vec![("amount", Value::int(50))]),
            result: Value::text("ok"),
            tier,
            applied_ms: now,
            idempotency_key: key.to_string(),
        }
    }

    #[test]
    fn idempotent_retry_returns_recorded_result_without_reapplying() {
        let mut mgr = TxnManager::new();
        let mut world = FakeWorld::default();
        let txn = mgr.begin("agent", vec![], 0);
        let s = step("bank.transfer", Compensability::Irreversible, "order-42", 0);
        // First execution.
        if mgr.lookup_idempotent("order-42").is_none() {
            world.apply(&s);
            mgr.record_step(txn, s.clone()).unwrap();
        }
        // Retry with the same key (network flaked, agent retries): no re-apply.
        let cached = mgr.lookup_idempotent("order-42");
        assert_eq!(cached, Some(Value::text("ok")));
        if cached.is_none() {
            world.apply(&s);
        }
        assert_eq!(world.apply_count, 1, "the transfer must happen exactly once");
        assert_eq!(world.balances["acct"], 950);
    }

    #[test]
    fn abort_compensates_in_exact_reverse_order() {
        let mut mgr = TxnManager::new();
        let mut world = FakeWorld::default();
        let txn = mgr.begin("agent", vec![], 0);
        for (i, tool) in ["file.write", "cal.create", "note.append"].iter().enumerate() {
            let s = step(tool, Compensability::Full, &format!("k{i}"), 0);
            world.apply(&s);
            mgr.record_step(txn, s).unwrap();
        }
        let applied = mgr.abort(txn).unwrap();
        for s in applied.iter().rev() {
            world.compensate(s).unwrap();
        }
        let undos: Vec<&str> = world
            .log
            .iter()
            .filter(|l| l.starts_with("undo:"))
            .map(|l| l.as_str())
            .collect();
        assert_eq!(undos, vec!["undo:note.append", "undo:cal.create", "undo:file.write"]);
        assert_eq!(mgr.state(txn), Some(TxnState::Aborted));
    }

    #[test]
    fn failed_precondition_means_the_transaction_never_starts() {
        let mut mgr = TxnManager::new();
        let pre = vec![Condition::new("slot free", Predicate::SlotFree("cal:mon-9am".into()))];
        let txn = mgr.begin("agent", pre, 0);
        let facts = FactBase::new(); // slot is NOT free
        let err = mgr.check_preconditions(txn, &facts).unwrap_err();
        assert_eq!(err, TxnError::PreconditionFailed { name: "slot free".into() });
        let ok_facts = FactBase::new().with_slot_free("cal:mon-9am");
        assert!(mgr.check_preconditions(txn, &ok_facts).is_ok());
    }

    #[test]
    fn irreversible_step_is_staged_not_dispatched_during_cooling_off() {
        let mut mgr = TxnManager::new();
        let mut world = FakeWorld::default();
        let txn = mgr.begin("agent", vec![], 1_000);
        let staged = StagedStep {
            tool: "bank.transfer".into(),
            resource: "acct".into(),
            args: Value::map(vec![("amount", Value::int(50))]),
            idempotency_key: "pay-1".into(),
        };
        mgr.stage_irreversible(txn, staged, 10_000, 1_000).unwrap();
        assert_eq!(mgr.state(txn), Some(TxnState::PendingCoolingOff));
        assert_eq!(world.apply_count, 0, "nothing reached the world yet");
        // Commit too early without user confirmation: refused.
        let err = mgr.begin_commit(txn, 5_000, false).unwrap_err();
        assert_eq!(err, TxnError::CoolingOffActive { remaining_ms: 6_000 });
        // After the deadline it releases for dispatch.
        let released = mgr.begin_commit(txn, 11_000, false).unwrap();
        assert_eq!(released.len(), 1);
        let applied = step("bank.transfer", Compensability::Irreversible, "pay-1", 11_000);
        world.apply(&applied);
        mgr.finish_commit(txn, vec![applied]).unwrap();
        assert_eq!(mgr.state(txn), Some(TxnState::Committed));
        assert_eq!(world.balances["acct"], 950);
    }

    #[test]
    fn cancelling_during_cooling_off_costs_nothing() {
        let mut mgr = TxnManager::new();
        let world = FakeWorld::default();
        let txn = mgr.begin("agent", vec![], 0);
        mgr.stage_irreversible(
            txn,
            StagedStep {
                tool: "bank.transfer".into(),
                resource: "acct".into(),
                args: Value::Null,
                idempotency_key: "pay-2".into(),
            },
            10_000,
            0,
        )
        .unwrap();
        let applied = mgr.abort(txn).unwrap();
        assert!(applied.is_empty());
        assert_eq!(world.apply_count, 0);
        assert_eq!(mgr.state(txn), Some(TxnState::Aborted));
    }

    #[test]
    fn explicit_user_confirmation_ends_cooling_off_early() {
        let mut mgr = TxnManager::new();
        let txn = mgr.begin("agent", vec![], 0);
        mgr.stage_irreversible(
            txn,
            StagedStep {
                tool: "mail.send".into(),
                resource: "draft-1".into(),
                args: Value::Null,
                idempotency_key: "send-1".into(),
            },
            60_000,
            0,
        )
        .unwrap();
        let released = mgr.begin_commit(txn, 1_000, true).unwrap();
        assert_eq!(released.len(), 1);
    }

    #[test]
    fn undo_plan_reverses_and_marks_undone() {
        let mut mgr = TxnManager::new();
        let mut world = FakeWorld::default();
        let txn = mgr.begin("agent", vec![], 0);
        let s = step("bank.transfer", Compensability::Irreversible, "t-1", 0);
        world.apply(&s);
        mgr.record_step(txn, s).unwrap();
        mgr.begin_commit(txn, 0, false).unwrap();
        mgr.finish_commit(txn, vec![]).unwrap();
        let plan = mgr.undo_plan(txn).unwrap();
        assert_eq!(plan.len(), 1);
        // compensation_status tells the kernel this one can never be undone.
        assert_eq!(compensation_status(&plan[0], 0), CompStatus::Never);
        mgr.mark_undone(txn).unwrap();
        assert_eq!(mgr.state(txn), Some(TxnState::Undone));
    }

    #[test]
    fn partial_compensation_window_expires() {
        let s = step("mail.send", Compensability::Partial { window_ms: 30_000 }, "s-1", 1_000);
        assert_eq!(compensation_status(&s, 20_000), CompStatus::Yes);
        assert_eq!(compensation_status(&s, 31_001), CompStatus::WindowExpired);
        let f = step("file.write", Compensability::Full, "f-1", 0);
        assert_eq!(compensation_status(&f, u64::MAX - 1), CompStatus::Yes);
    }

    #[test]
    fn wrong_state_operations_are_rejected() {
        let mut mgr = TxnManager::new();
        let txn = mgr.begin("agent", vec![], 0);
        assert!(matches!(mgr.undo_plan(txn), Err(TxnError::WrongState { .. })));
        mgr.abort(txn).unwrap();
        assert!(matches!(mgr.abort(txn), Err(TxnError::WrongState { .. })));
        assert!(matches!(mgr.begin_commit(999, 0, false), Err(TxnError::NotFound { .. })));
    }
}
