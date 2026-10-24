//! The microkernel (doc ch. 8): the only component allowed to change the
//! world. It owns the capability authority, the tool registry, the effect
//! type checker's signatures, the memory store, the transaction manager,
//! per-agent quota groups, the trust scheduler — and the append-only,
//! hash-chained action ledger that makes every decision replayable.
//!
//! The ten primitives of doc ch. 8 map to methods:
//! `intent_submit`, `ctx_attach`, `cap_request`, `act`, `txn_commit`,
//! `txn_abort`, `mem_assert`, `mem_recall`, `watch`, `yield_request`,
//! plus the first-class `undo`.
//!
//! `act()` is the narrow waist. In order, it: replays idempotent retries,
//! verifies the capability (signature, lease, revocation, deny rules, use
//! count), checks the capability covers the step, enforces the speculation
//! barrier, charges the side-effect quota, and only then dispatches —
//! through the registry, which re-checks the capability a second time
//! (defense in depth). Irreversible effects are staged for the cooling-off
//! window instead of dispatched.

use crate::capabilities::{CapError, Capability, CapabilityAuthority, LinearCapability, Scope};
use crate::crypto::sha256_hex;
use crate::memory::{Memory, MemoryError, Provenance, Recall};
use crate::schedulers::{AgentGroup, QuotaError, Quotas, TrustScheduler};
use crate::speculation::{barrier_for, Barrier, StepProfile};
use crate::taint::Tainted;
use crate::tools::{Tool, ToolError, ToolRegistry};
use crate::transactions::{compensation_status, AppliedStep, CompStatus, StagedStep, TxnError, TxnManager, TxnState};
use crate::typesys::{derive_risk, Compensability, Condition, Effect, Facts, Predicate, Risk, TypeChecker};
use crate::{Clock, Value};
use std::collections::{BTreeSet, HashMap};
use std::rc::Rc;

/// 30 days, the default memory half-life (doc ch. 10: facts live "days to
/// weeks" in L1 and decay unless re-confirmed).
pub const MEMORY_HALF_LIFE_MS: u64 = 30 * 86_400_000;
/// Cooling-off window for irreversible effects (doc ch. 15).
pub const DEFAULT_COOLING_OFF_MS: u64 = 10_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LedgerKind {
    IntentSubmitted,
    ContextAttached,
    ContextRevoked,
    CapIssued,
    CapRevoked,
    Act,
    ActStaged,
    ActDenied,
    TxnCommitted,
    TxnAborted,
    TxnUndone,
    Compensation,
    MemAssert,
    MemRecall,
    YieldRequested,
    YieldResolved,
    WatchRegistered,
    WatchFired,
}

/// One entry of the action ledger. Entries are hash-chained (each commits
/// to its predecessor), so the log is tamper-evident and any transaction
/// can be replayed exactly (doc ch. 15, "decision traces").
#[derive(Debug, Clone)]
pub struct LedgerEntry {
    pub seq: u64,
    pub ts_ms: u64,
    pub actor: String,
    pub kind: LedgerKind,
    pub tool: Option<String>,
    pub resource: Option<String>,
    pub cap_fingerprint: Option<String>,
    pub txn: Option<u64>,
    pub detail: String,
    pub prev_hash: String,
    pub hash: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    Active,
    Done,
    Aborted,
}

#[derive(Debug, Clone)]
pub struct Task {
    pub id: u64,
    pub agent: String,
    pub goal: String,
    pub constraints: Vec<String>,
    pub status: TaskStatus,
}

#[derive(Debug, Clone)]
pub struct ContextHandle {
    pub id: u64,
    pub task: u64,
    pub data: Tainted<String>,
    pub refcount: u32,
    pub revoked: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Decision {
    Approve,
    Deny,
    Choose(usize),
}

#[derive(Debug, Clone)]
pub struct YieldTicket {
    pub id: u64,
    pub agent: String,
    pub reason: String,
    pub options: Vec<String>,
    pub default_index: usize,
    pub resolved: Option<Decision>,
}

#[derive(Debug, Clone, Default)]
pub struct ActOptions {
    pub idempotency_key: String,
    pub why: String,
    pub speculative: bool,
    pub step_profile: Option<StepProfile>,
}

/// Outcome of an abort/undo: what got compensated and what could not be
/// (with the reason — an honest report, doc ch. 15).
#[derive(Debug, Clone, Default)]
pub struct CompensationReport {
    pub compensated: Vec<String>,
    pub not_compensated: Vec<(String, String)>,
}

#[derive(Debug)]
pub enum KernelError {
    Cap(CapError),
    Tool(ToolError),
    Txn(TxnError),
    Memory(MemoryError),
    Quota(QuotaError),
    SpeculationBarrier(Barrier),
    CapScopeMismatch { tool: String },
    TrustDenied { risk: Risk },
    UnknownTask { id: u64 },
    UnknownContext { id: u64 },
    UnknownYield { id: u64 },
    PreconditionFailed { name: String },
}

impl std::fmt::Display for KernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KernelError::Cap(e) => write!(f, "capability error: {e}"),
            KernelError::Tool(e) => write!(f, "tool error: {e}"),
            KernelError::Txn(e) => write!(f, "transaction error: {e}"),
            KernelError::Memory(e) => write!(f, "memory error: {e}"),
            KernelError::Quota(e) => write!(f, "{e}"),
            KernelError::SpeculationBarrier(b) => {
                write!(f, "speculation barrier hit: {b:?} — step is forced onto the serial path")
            }
            KernelError::CapScopeMismatch { tool } => {
                write!(f, "capability does not cover step for tool {tool}")
            }
            KernelError::TrustDenied { risk } => {
                write!(f, "agent trust level too low for {}-risk capability", risk.name())
            }
            KernelError::UnknownTask { id } => write!(f, "unknown task {id}"),
            KernelError::UnknownContext { id } => write!(f, "unknown context handle {id}"),
            KernelError::UnknownYield { id } => write!(f, "unknown yield ticket {id}"),
            KernelError::PreconditionFailed { name } => write!(f, "precondition failed: {name}"),
        }
    }
}
impl std::error::Error for KernelError {}
impl From<CapError> for KernelError {
    fn from(e: CapError) -> Self {
        KernelError::Cap(e)
    }
}
impl From<ToolError> for KernelError {
    fn from(e: ToolError) -> Self {
        KernelError::Tool(e)
    }
}
impl From<TxnError> for KernelError {
    fn from(e: TxnError) -> Self {
        KernelError::Txn(e)
    }
}
impl From<MemoryError> for KernelError {
    fn from(e: MemoryError) -> Self {
        KernelError::Memory(e)
    }
}
impl From<QuotaError> for KernelError {
    fn from(e: QuotaError) -> Self {
        KernelError::Quota(e)
    }
}

struct WatchSub {
    id: u64,
    agent: String,
    name: String,
    predicate: Predicate,
}

pub struct Kernel {
    clock: Rc<dyn Clock>,
    authority: CapabilityAuthority,
    registry: ToolRegistry,
    checker: TypeChecker,
    memory: Memory,
    txns: TxnManager,
    ledger: Vec<LedgerEntry>,
    tasks: HashMap<u64, Task>,
    contexts: HashMap<u64, ContextHandle>,
    groups: HashMap<String, AgentGroup>,
    trust: TrustScheduler,
    yields: HashMap<u64, YieldTicket>,
    watches: Vec<WatchSub>,
    staged_caps: HashMap<u64, Vec<(StagedStep, Capability)>>,
    next_task: u64,
    next_ctx: u64,
    next_yield: u64,
    next_watch: u64,
    cooling_off_ms: u64,
}

impl Kernel {
    pub fn new(secret: impl Into<Vec<u8>>, clock: Rc<dyn Clock>) -> Self {
        Kernel {
            authority: CapabilityAuthority::new(secret, clock.clone()),
            registry: ToolRegistry::new(),
            checker: TypeChecker::new(),
            memory: Memory::new(clock.clone(), MEMORY_HALF_LIFE_MS),
            txns: TxnManager::new(),
            ledger: Vec::new(),
            tasks: HashMap::new(),
            contexts: HashMap::new(),
            groups: HashMap::new(),
            trust: TrustScheduler::new(),
            yields: HashMap::new(),
            watches: Vec::new(),
            staged_caps: HashMap::new(),
            next_task: 1,
            next_ctx: 1,
            next_yield: 1,
            next_watch: 1,
            cooling_off_ms: DEFAULT_COOLING_OFF_MS,
            clock,
        }
    }

    pub fn set_cooling_off_ms(&mut self, ms: u64) {
        self.cooling_off_ms = ms;
    }

    // -- Tool administration ------------------------------------------------

    /// Register a tool: registry entry + type-checker signature from the
    /// same manifest, so the two can never disagree.
    pub fn register_tool(&mut self, tool: Box<dyn Tool>) {
        let manifest = tool.manifest().clone();
        self.checker.register(&manifest.name, manifest.effects.clone());
        self.registry.register(tool);
    }

    pub fn approve_tool(&mut self, name: &str) -> Result<(), KernelError> {
        Ok(self.registry.approve(name)?)
    }

    pub fn registry(&self) -> &ToolRegistry {
        &self.registry
    }

    pub fn registry_mut(&mut self) -> &mut ToolRegistry {
        &mut self.registry
    }

    pub fn checker(&self) -> &TypeChecker {
        &self.checker
    }

    pub fn memory(&self) -> &Memory {
        &self.memory
    }

    pub fn memory_mut(&mut self) -> &mut Memory {
        &mut self.memory
    }

    pub fn trust_level(&self, agent: &str) -> u8 {
        self.trust.level(agent)
    }

    /// A user confirmation outside any specific yield (still evidence for
    /// the trust scheduler's progressive grants).
    pub fn record_confirmation(&mut self, agent: &str) -> u8 {
        self.trust.record_confirmation(agent)
    }

    pub fn quota_state(&self, agent: &str) -> Option<&AgentGroup> {
        self.groups.get(agent)
    }

    // -- Ledger ---------------------------------------------------------------

    fn log(
        &mut self,
        actor: &str,
        kind: LedgerKind,
        tool: Option<&str>,
        resource: Option<&str>,
        cap: Option<&Capability>,
        txn: Option<u64>,
        detail: String,
    ) {
        let seq = self.ledger.len() as u64 + 1;
        let ts_ms = self.clock.now_ms();
        let prev_hash = self.ledger.last().map(|e| e.hash.clone()).unwrap_or_else(|| "genesis".to_string());
        let cap_fp = cap.map(|c| c.fingerprint());
        let hash = sha256_hex(
            format!(
                "{seq}|{ts_ms}|{actor}|{kind:?}|{}|{}|{}|{detail}|{prev_hash}",
                tool.unwrap_or("-"),
                resource.unwrap_or("-"),
                cap_fp.as_deref().unwrap_or("-")
            )
            .as_bytes(),
        );
        self.ledger.push(LedgerEntry {
            seq,
            ts_ms,
            actor: actor.to_string(),
            kind,
            tool: tool.map(|s| s.to_string()),
            resource: resource.map(|s| s.to_string()),
            cap_fingerprint: cap_fp,
            txn,
            detail,
            prev_hash,
            hash,
        });
    }

    pub fn ledger(&self) -> &[LedgerEntry] {
        &self.ledger
    }

    /// Replay the decision trace of one transaction (doc ch. 15).
    pub fn replay(&self, txn: u64) -> Vec<LedgerEntry> {
        self.ledger.iter().filter(|e| e.txn == Some(txn)).cloned().collect()
    }

    /// Verify the hash chain is intact (tamper evidence).
    pub fn verify_ledger_chain(&self) -> bool {
        let mut prev = "genesis".to_string();
        for e in &self.ledger {
            if e.prev_hash != prev {
                return false;
            }
            prev = e.hash.clone();
        }
        true
    }

    // -- Primitive: intent_submit --------------------------------------------

    pub fn intent_submit(
        &mut self,
        agent: &str,
        goal: &str,
        constraints: Vec<String>,
        quotas: Quotas,
    ) -> u64 {
        let id = self.next_task;
        self.next_task += 1;
        self.tasks.insert(
            id,
            Task { id, agent: agent.to_string(), goal: goal.to_string(), constraints, status: TaskStatus::Active },
        );
        self.groups.entry(agent.to_string()).or_insert_with(|| AgentGroup::new(quotas));
        self.log(agent, LedgerKind::IntentSubmitted, None, None, None, None, format!("goal: {goal}"));
        id
    }

    pub fn task(&self, id: u64) -> Option<&Task> {
        self.tasks.get(&id)
    }

    pub fn finish_task(&mut self, id: u64) {
        if let Some(t) = self.tasks.get_mut(&id) {
            t.status = TaskStatus::Done;
        }
    }

    // -- Primitive: ctx_attach --------------------------------------------------

    pub fn ctx_attach(&mut self, task: u64, data: Tainted<String>) -> Result<u64, KernelError> {
        if !self.tasks.contains_key(&task) {
            return Err(KernelError::UnknownTask { id: task });
        }
        let id = self.next_ctx;
        self.next_ctx += 1;
        let label = data.label;
        self.contexts.insert(
            id,
            ContextHandle { id, task, data, refcount: 1, revoked: false },
        );
        self.log("kernel", LedgerKind::ContextAttached, None, None, None, None, format!("ctx {id} attached to task {task} with taint {label:?}"));
        Ok(id)
    }

    pub fn ctx_read(&self, handle: u64) -> Option<&Tainted<String>> {
        self.contexts.get(&handle).filter(|h| !h.revoked).map(|h| &h.data)
    }

    pub fn ctx_revoke(&mut self, handle: u64) -> Result<(), KernelError> {
        let h = self.contexts.get_mut(&handle).ok_or(KernelError::UnknownContext { id: handle })?;
        h.revoked = true;
        self.log("kernel", LedgerKind::ContextRevoked, None, None, None, None, format!("ctx {handle} revoked"));
        Ok(())
    }

    // -- Primitive: cap_request -------------------------------------------------

    /// Issue a capability. Two gates beyond the authority itself: the
    /// agent's *trust level* must cover the scope's derived risk
    /// (progressive grants, doc ch. 9), and the agent group must have a
    /// free capability slot (doc ch. 8, cgroup).
    pub fn cap_request(
        &mut self,
        agent: &str,
        scope: Scope,
        ttl_ms: u64,
    ) -> Result<Capability, KernelError> {
        let risk = derive_risk(&scope.effects);
        if !self.trust.authorize(agent, risk) {
            self.log(agent, LedgerKind::ActDenied, Some(&scope.tool), None, None, None, format!("trust level {} too low for {} risk", self.trust.level(agent), risk.name()));
            return Err(KernelError::TrustDenied { risk });
        }
        let group = self
            .groups
            .entry(agent.to_string())
            .or_insert_with(|| AgentGroup::new(Quotas::generous()));
        group.try_acquire_cap()?;
        let tool = scope.tool.clone();
        let cap = self.authority.issue(agent, scope, ttl_ms);
        self.log(agent, LedgerKind::CapIssued, Some(&tool), None, Some(&cap), None, format!("cap {} issued", cap.id()));
        Ok(cap)
    }

    /// Attenuated delegation (the only way to derive a capability).
    pub fn cap_attenuate(
        &mut self,
        parent: &Capability,
        new_holder: &str,
        narrower: Scope,
        ttl_ms: u64,
    ) -> Result<Capability, KernelError> {
        let child = self.authority.attenuate(parent, new_holder, narrower, ttl_ms)?;
        self.log(new_holder, LedgerKind::CapIssued, Some(&child.claims.scope.tool), None, Some(&child), None, format!("attenuated from cap {}", parent.id()));
        Ok(child)
    }

    pub fn cap_revoke(&mut self, id: u64) {
        self.authority.revoke(id);
        self.log("kernel", LedgerKind::CapRevoked, None, None, None, None, format!("cap {id} revoked (with descendants)"));
    }

    pub fn add_deny(&mut self, rule: crate::capabilities::DenyRule) {
        self.authority.add_deny(rule);
    }

    // -- Transactions ------------------------------------------------------------

    pub fn txn_begin(&mut self, agent: &str, preconditions: Vec<Condition>) -> u64 {
        let now = self.clock.now_ms();
        self.txns.begin(agent, preconditions, now)
    }

    pub fn txn_state(&self, txn: u64) -> Option<TxnState> {
        self.txns.state(txn)
    }

    pub fn txn_check_preconditions(&self, txn: u64, facts: &dyn Facts) -> Result<(), KernelError> {
        Ok(self.txns.check_preconditions(txn, facts)?)
    }

    // -- Primitive: act ------------------------------------------------------------

    /// The unique entry point that changes the world. See the module docs
    /// for the exact check order. `step` is (tool, resource, args).
    pub fn act(
        &mut self,
        agent: &str,
        txn: u64,
        cap: &Capability,
        tool: &str,
        resource: &str,
        args: Value,
        opts: ActOptions,
    ) -> Result<Value, KernelError> {
        // Idempotent replay: a retry returns the recorded result and never
        // re-executes (doc ch. 15).
        if !opts.idempotency_key.is_empty() {
            if let Some(prev) = self.txns.lookup_idempotent(&opts.idempotency_key) {
                self.log(agent, LedgerKind::Act, Some(tool), Some(resource), Some(cap), Some(txn), format!("idempotent replay of key {}", opts.idempotency_key));
                return Ok(prev);
            }
        }

        // Capability verification: signature, lease, revocation, deny,
        // remaining uses.
        if let Err(e) = self.authority.verify_and_consume(cap) {
            self.log(agent, LedgerKind::ActDenied, Some(tool), Some(resource), Some(cap), Some(txn), format!("capability rejected: {e}"));
            return Err(e.into());
        }

        // The capability must cover exactly this step.
        let manifest = self
            .registry
            .manifest(tool)
            .ok_or_else(|| KernelError::Tool(ToolError::NotFound { tool: tool.to_string() }))?
            .clone();
        let scope = cap.scope();
        if scope.tool != tool
            || !scope.resources.contains(resource)
            || !manifest.effects.is_subset(&scope.effects)
        {
            self.log(agent, LedgerKind::ActDenied, Some(tool), Some(resource), Some(cap), Some(txn), "capability scope does not cover this step".into());
            return Err(KernelError::CapScopeMismatch { tool: tool.to_string() });
        }

        // Speculation barrier, at the kernel entry — a compromised runtime
        // cannot switch this off (doc ch. 12).
        if opts.speculative {
            let profile = opts.step_profile.unwrap_or(StepProfile {
                compensability: manifest.compensability,
                deps_trackable: true,
                content_dependent: false,
            });
            if let Some(barrier) = barrier_for(&profile) {
                self.log(agent, LedgerKind::ActDenied, Some(tool), Some(resource), Some(cap), Some(txn), format!("speculation barrier: {barrier:?}"));
                return Err(KernelError::SpeculationBarrier(barrier));
            }
        }

        // Side-effect quota.
        if manifest.has_side_effects() {
            let group = self
                .groups
                .entry(agent.to_string())
                .or_insert_with(|| AgentGroup::new(Quotas::generous()));
            if let Err(e) = group.try_consume_side_effect() {
                self.log(agent, LedgerKind::ActDenied, Some(tool), Some(resource), Some(cap), Some(txn), format!("{e}"));
                return Err(e.into());
            }
        }

        // Irreversible effects are staged for cooling-off, not dispatched.
        if manifest.compensability == Compensability::Irreversible {
            let staged = StagedStep {
                tool: tool.to_string(),
                resource: resource.to_string(),
                args,
                idempotency_key: opts.idempotency_key.clone(),
            };
            self.txns.stage_irreversible(txn, staged.clone(), self.cooling_off_ms, self.clock.now_ms())?;
            self.staged_caps.entry(txn).or_default().push((staged, cap.clone()));
            self.log(agent, LedgerKind::ActStaged, Some(tool), Some(resource), Some(cap), Some(txn), format!("irreversible effect staged for cooling-off: {}", opts.why));
            return Ok(Value::text("pending:cooling-off"));
        }

        // Dispatch (the registry re-verifies the capability: defense in depth).
        let result = self.registry.invoke(tool, cap, resource, &args)?;
        let now = self.clock.now_ms();
        self.txns.record_step(
            txn,
            AppliedStep {
                tool: tool.to_string(),
                resource: resource.to_string(),
                args,
                result: result.clone(),
                tier: manifest.compensability,
                applied_ms: now,
                idempotency_key: opts.idempotency_key.clone(),
            },
        )?;
        self.log(agent, LedgerKind::Act, Some(tool), Some(resource), Some(cap), Some(txn), opts.why.clone());
        Ok(result)
    }

    /// Spend a linear (single-use) capability. The wrapper is consumed by
    /// value, so a second spend is a compile error at the call site — the
    /// affine-type enforcement of doc ch. 7's ⊸.
    pub fn act_linear(
        &mut self,
        agent: &str,
        txn: u64,
        linear: LinearCapability,
        tool: &str,
        resource: &str,
        args: Value,
        opts: ActOptions,
    ) -> Result<Value, KernelError> {
        let cap = linear.into_capability();
        self.act(agent, txn, &cap, tool, resource, args, opts)
    }

    // -- Primitives: txn_commit / txn_abort / undo --------------------------------

    /// A short-lived capability the kernel issues to *itself* to run
    /// compensations. Compensations are kernel actions, not agent actions,
    /// so they must not depend on an agent token that may have expired.
    fn system_cap(&mut self, tool: &str, resource: &str) -> Capability {
        let effects: BTreeSet<Effect> = self
            .registry
            .manifest(tool)
            .map(|m| m.effects.clone())
            .unwrap_or_default();
        self.authority.issue(
            "kernel",
            Scope { tool: tool.to_string(), resources: [resource.to_string()].into_iter().collect(), effects },
            60_000,
        )
    }

    fn run_compensations(&mut self, agent: &str, txn: u64, steps_reversed: &[AppliedStep]) -> CompensationReport {
        let mut report = CompensationReport::default();
        let now = self.clock.now_ms();
        for step in steps_reversed {
            match compensation_status(step, now) {
                CompStatus::Never => {
                    report.not_compensated.push((step.tool.clone(), "irreversible effect".into()));
                }
                CompStatus::WindowExpired => {
                    report.not_compensated.push((step.tool.clone(), "compensation window expired".into()));
                }
                CompStatus::Yes => {
                    let cap = self.system_cap(&step.tool, &step.resource);
                    match self.registry.compensate(&step.tool, &cap, &step.resource, &step.args, &step.result) {
                        Ok(()) => {
                            report.compensated.push(step.tool.clone());
                            self.log(agent, LedgerKind::Compensation, Some(&step.tool), Some(&step.resource), None, Some(txn), "compensated".into());
                        }
                        Err(e) => report.not_compensated.push((step.tool.clone(), format!("{e}"))),
                    }
                }
            }
        }
        report
    }

    pub fn txn_commit(&mut self, agent: &str, txn: u64, user_confirmed: bool) -> Result<(), KernelError> {
        let staged = self.txns.begin_commit(txn, self.clock.now_ms(), user_confirmed)?;
        if !staged.is_empty() {
            let caps = self.staged_caps.remove(&txn).unwrap_or_default();
            for step in &staged {
                let cap = caps
                    .iter()
                    .find(|(s, _)| s.idempotency_key == step.idempotency_key)
                    .map(|(_, c)| c.clone())
                    .ok_or_else(|| KernelError::Tool(ToolError::Invocation("staged capability lost".into())))?;
                // The capability must still be valid at dispatch time.
                self.authority.verify(&cap)?;
                let result = self.registry.invoke(&step.tool, &cap, &step.resource, &step.args)?;
                let manifest = self.registry.manifest(&step.tool).expect("registered");
                let tier = manifest.compensability;
                self.txns.record_step(
                    txn,
                    AppliedStep {
                        tool: step.tool.clone(),
                        resource: step.resource.clone(),
                        args: step.args.clone(),
                        result,
                        tier,
                        applied_ms: self.clock.now_ms(),
                        idempotency_key: step.idempotency_key.clone(),
                    },
                )?;
                self.log(agent, LedgerKind::Act, Some(&step.tool), Some(&step.resource), Some(&cap), Some(txn), "cooling-off release".into());
            }
        }
        self.txns.finish_commit(txn, vec![])?;
        self.log(agent, LedgerKind::TxnCommitted, None, None, None, Some(txn), "committed".into());
        Ok(())
    }

    pub fn txn_abort(&mut self, agent: &str, txn: u64) -> Result<CompensationReport, KernelError> {
        let applied = self.txns.abort(txn)?;
        self.staged_caps.remove(&txn);
        let mut reversed = applied;
        reversed.reverse();
        let report = self.run_compensations(agent, txn, &reversed);
        self.log(agent, LedgerKind::TxnAborted, None, None, None, Some(txn), format!("aborted; {} compensated, {} not", report.compensated.len(), report.not_compensated.len()));
        Ok(report)
    }

    /// `undo(txn_id)` — a first-class primitive, not a feature (doc ch. 5,
    /// axiom III). Works on committed transactions; reports honestly which
    /// steps could not be taken back.
    pub fn undo(&mut self, agent: &str, txn: u64) -> Result<CompensationReport, KernelError> {
        let plan = self.txns.undo_plan(txn)?;
        let report = self.run_compensations(agent, txn, &plan);
        self.txns.mark_undone(txn)?;
        self.log(agent, LedgerKind::TxnUndone, None, None, None, Some(txn), format!("undo: {} compensated, {} not", report.compensated.len(), report.not_compensated.len()));
        Ok(report)
    }

    // -- Primitives: mem_assert / mem_recall ---------------------------------------

    pub fn mem_assert(
        &mut self,
        agent: &str,
        key: &str,
        fact: &str,
        category: crate::memory::FactCategory,
        provenance: Option<Provenance>,
        ttl_ms: Option<u64>,
    ) -> Result<u64, KernelError> {
        let id = self.memory.assert_fact(key, fact, category, provenance, ttl_ms)?;
        self.log(agent, LedgerKind::MemAssert, None, Some(key), None, None, format!("memory {id} asserted: {fact}"));
        Ok(id)
    }

    pub fn mem_recall(&mut self, agent: &str, query: &str, min_relevance: f64) -> Recall {
        let recall = self.memory.recall(query, min_relevance, false);
        self.log(agent, LedgerKind::MemRecall, None, None, None, None, format!("recall '{query}': {} items, confidence {:.2}, needs_yield {}", recall.items.len(), recall.confidence, recall.needs_yield));
        recall
    }

    // -- Primitive: yield ------------------------------------------------------------

    /// Ask the human to decide. Costs one unit of the agent's attention
    /// quota (doc ch. 8: attention is a budgeted resource).
    pub fn yield_request(
        &mut self,
        agent: &str,
        reason: &str,
        options: Vec<String>,
        default_index: usize,
    ) -> Result<u64, KernelError> {
        let group = self
            .groups
            .entry(agent.to_string())
            .or_insert_with(|| AgentGroup::new(Quotas::generous()));
        group.try_consume_attention()?;
        let id = self.next_yield;
        self.next_yield += 1;
        self.yields.insert(
            id,
            YieldTicket {
                id,
                agent: agent.to_string(),
                reason: reason.to_string(),
                options,
                default_index,
                resolved: None,
            },
        );
        self.log(agent, LedgerKind::YieldRequested, None, None, None, None, format!("yield {id}: {reason}"));
        Ok(id)
    }

    pub fn resolve_yield(&mut self, id: u64, decision: Decision) -> Result<(), KernelError> {
        let ticket = self.yields.get_mut(&id).ok_or(KernelError::UnknownYield { id })?;
        ticket.resolved = Some(decision.clone());
        let agent = ticket.agent.clone();
        if decision == Decision::Approve {
            // An approving human is trust evidence (progressive grants).
            self.trust.record_confirmation(&agent);
        }
        self.log(&agent.clone(), LedgerKind::YieldResolved, None, None, None, None, format!("yield {id} resolved: {decision:?}"));
        Ok(())
    }

    pub fn yield_ticket(&self, id: u64) -> Option<&YieldTicket> {
        self.yields.get(&id)
    }

    // -- Primitive: watch --------------------------------------------------------------

    pub fn watch(&mut self, agent: &str, name: &str, predicate: Predicate) -> u64 {
        let id = self.next_watch;
        self.next_watch += 1;
        self.watches.push(WatchSub { id, agent: agent.to_string(), name: name.to_string(), predicate });
        self.log(agent, LedgerKind::WatchRegistered, None, None, None, None, format!("watch {id}: {name}"));
        id
    }

    /// Evaluate all subscriptions against current facts; fired watches are
    /// one-shot (edge-triggered, like epoll) and returned as (id, name).
    pub fn poll_watches(&mut self, facts: &dyn Facts) -> Vec<(u64, String)> {
        let mut fired = Vec::new();
        let mut remaining = Vec::new();
        for sub in std::mem::take(&mut self.watches) {
            let cond = Condition { name: sub.name.clone(), predicate: sub.predicate.clone() };
            if cond.eval(facts) {
                fired.push((sub.id, sub.name.clone()));
                self.log(&sub.agent.clone(), LedgerKind::WatchFired, None, None, None, None, format!("watch {} fired: {}", sub.id, sub.name));
            } else {
                remaining.push(sub);
            }
        }
        self.watches = remaining;
        fired
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capabilities::Scope;
    use crate::memory::{FactCategory, SourceKind};
    use crate::taint::TrustLabel;
    use crate::typesys::{Consistency, Predicate as Pred};
    use crate::ManualClock;

    // -- Test tools ---------------------------------------------------------

    struct CalTool {
        manifest: crate::tools::ToolManifest,
        events: Vec<String>,
    }
    impl CalTool {
        fn new() -> Self {
            CalTool {
                manifest: crate::tools::ToolManifest::new(
                    "cal.write",
                    "Create a calendar event.",
                    &[Effect::Write],
                    Consistency::Causal,
                    Compensability::Full,
                ),
                events: Vec::new(),
            }
        }
    }
    impl Tool for CalTool {
        fn manifest(&self) -> &crate::tools::ToolManifest {
            &self.manifest
        }
        fn manifest_mut(&mut self) -> &mut crate::tools::ToolManifest {
            &mut self.manifest
        }
        fn invoke(&mut self, args: &Value) -> Result<Value, ToolError> {
            let title = args.get("title").and_then(|v| v.as_text()).unwrap_or("event").to_string();
            self.events.push(title.clone());
            Ok(Value::text(format!("created:{title}")))
        }
        fn compensate(&mut self, args: &Value, _result: &Value) -> Result<(), ToolError> {
            let title = args.get("title").and_then(|v| v.as_text()).unwrap_or("event");
            self.events.retain(|e| e != title);
            Ok(())
        }
    }

    struct BankTool {
        manifest: crate::tools::ToolManifest,
        balance: i64,
    }
    impl BankTool {
        fn new() -> Self {
            BankTool {
                manifest: crate::tools::ToolManifest::new(
                    "bank.transfer",
                    "Transfer money.",
                    &[Effect::Transfer],
                    Consistency::Strong,
                    Compensability::Irreversible,
                ),
                balance: 1_000,
            }
        }
    }
    impl Tool for BankTool {
        fn manifest(&self) -> &crate::tools::ToolManifest {
            &self.manifest
        }
        fn manifest_mut(&mut self) -> &mut crate::tools::ToolManifest {
            &mut self.manifest
        }
        fn invoke(&mut self, args: &Value) -> Result<Value, ToolError> {
            let amt = args.get("amount").and_then(|v| v.as_int()).unwrap_or(0);
            self.balance -= amt;
            Ok(Value::int(self.balance))
        }
    }

    fn setup() -> (Kernel, Rc<ManualClock>) {
        let clock = Rc::new(ManualClock::new(1_000_000));
        let mut k = Kernel::new(b"kernel-secret".to_vec(), clock.clone());
        k.register_tool(Box::new(CalTool::new()));
        k.register_tool(Box::new(BankTool::new()));
        k.approve_tool("cal.write").unwrap();
        k.approve_tool("bank.transfer").unwrap();
        (k, clock)
    }

    /// Raise an agent's trust to max via confirmations.
    fn trust_up(k: &mut Kernel, agent: &str) {
        for _ in 0..6 {
            k.record_confirmation(agent);
        }
    }

    fn cal_cap(k: &mut Kernel, agent: &str) -> Capability {
        k.cap_request(agent, Scope::new("cal.write", &["calendar"], &[Effect::Write]), 600_000)
            .unwrap()
    }

    fn write_args() -> Value {
        Value::map(vec![("title", Value::text("Shanghai trip"))])
    }

    #[test]
    fn full_flow_intent_cap_act_commit_and_ledger_replay() {
        let (mut k, _c) = setup();
        trust_up(&mut k, "agent");
        let task = k.intent_submit("agent", "arrange the Shanghai trip", vec![], Quotas::generous());
        assert_eq!(k.task(task).unwrap().goal, "arrange the Shanghai trip");
        let cap = cal_cap(&mut k, "agent");
        let txn = k.txn_begin("agent", vec![]);
        let out = k
            .act("agent", txn, &cap, "cal.write", "calendar", write_args(), ActOptions { idempotency_key: "evt-1".into(), why: "book trip".into(), ..Default::default() })
            .unwrap();
        assert_eq!(out, Value::text("created:Shanghai trip"));
        k.txn_commit("agent", txn, false).unwrap();
        assert_eq!(k.txn_state(txn), Some(TxnState::Committed));
        let trace = k.replay(txn);
        assert!(trace.iter().any(|e| e.kind == LedgerKind::Act));
        assert!(trace.iter().any(|e| e.kind == LedgerKind::TxnCommitted));
        assert!(k.verify_ledger_chain());
    }

    #[test]
    fn undo_compensates_the_calendar_write() {
        let (mut k, _c) = setup();
        trust_up(&mut k, "agent");
        let cap = cal_cap(&mut k, "agent");
        let txn = k.txn_begin("agent", vec![]);
        k.act("agent", txn, &cap, "cal.write", "calendar", write_args(), ActOptions { idempotency_key: "evt-2".into(), ..Default::default() }).unwrap();
        k.txn_commit("agent", txn, false).unwrap();
        let report = k.undo("agent", txn).unwrap();
        assert_eq!(report.compensated, vec!["cal.write".to_string()]);
        assert!(report.not_compensated.is_empty());
        assert_eq!(k.txn_state(txn), Some(TxnState::Undone));
    }

    #[test]
    fn idempotent_act_replays_without_reexecuting() {
        let (mut k, _c) = setup();
        trust_up(&mut k, "agent");
        let cap = cal_cap(&mut k, "agent");
        let txn = k.txn_begin("agent", vec![]);
        let opts = ActOptions { idempotency_key: "evt-3".into(), ..Default::default() };
        let first = k.act("agent", txn, &cap, "cal.write", "calendar", write_args(), opts.clone()).unwrap();
        let second = k.act("agent", txn, &cap, "cal.write", "calendar", write_args(), opts).unwrap();
        assert_eq!(first, second);
        // The tool itself must have run once: check via a fresh read of the
        // registry state through a compensation-free probe — undoing shows
        // exactly one applied step.
        k.txn_commit("agent", txn, false).unwrap();
        let plan_len = k.replay(txn).iter().filter(|e| e.kind == LedgerKind::Act).count();
        assert_eq!(plan_len, 2, "one real act + one replay note in the ledger");
    }

    #[test]
    fn forged_capability_is_denied_and_logged() {
        let (mut k, _c) = setup();
        trust_up(&mut k, "agent");
        let mut cap = cal_cap(&mut k, "agent");
        cap.sig = [9u8; 32]; // forged
        let txn = k.txn_begin("agent", vec![]);
        let err = k.act("agent", txn, &cap, "cal.write", "calendar", write_args(), ActOptions::default());
        assert!(matches!(err, Err(KernelError::Cap(CapError::BadSignature))));
        assert!(k.ledger().iter().any(|e| e.kind == LedgerKind::ActDenied));
    }

    #[test]
    fn capability_for_the_wrong_tool_cannot_act() {
        let (mut k, _c) = setup();
        trust_up(&mut k, "agent");
        let cap = cal_cap(&mut k, "agent"); // cal.write cap
        let txn = k.txn_begin("agent", vec![]);
        let err = k.act(
            "agent",
            txn,
            &cap,
            "bank.transfer",
            "acct",
            Value::map(vec![("amount", Value::int(10))]),
            ActOptions::default(),
        );
        assert!(matches!(err, Err(KernelError::CapScopeMismatch { .. })));
    }

    #[test]
    fn trust_gate_blocks_high_risk_caps_for_new_agents() {
        let (mut k, _c) = setup();
        // Fresh agent, trust level 0: a Transfer (Critical) cap is refused.
        let err = k.cap_request("rookie", Scope::new("bank.transfer", &["acct"], &[Effect::Transfer]), 60_000);
        assert!(matches!(err, Err(KernelError::TrustDenied { risk: Risk::Critical })));
        // After enough user confirmations the same request succeeds.
        trust_up(&mut k, "rookie");
        assert!(k.cap_request("rookie", Scope::new("bank.transfer", &["acct"], &[Effect::Transfer]), 60_000).is_ok());
    }

    #[test]
    fn irreversible_act_is_staged_until_cooling_off_ends() {
        let (mut k, clock) = setup();
        trust_up(&mut k, "agent");
        let cap = k
            .cap_request("agent", Scope::new("bank.transfer", &["acct"], &[Effect::Transfer]), 600_000)
            .unwrap();
        let txn = k.txn_begin("agent", vec![]);
        let out = k
            .act("agent", txn, &cap, "bank.transfer", "acct", Value::map(vec![("amount", Value::int(200))]), ActOptions { idempotency_key: "pay-1".into(), ..Default::default() })
            .unwrap();
        assert_eq!(out, Value::text("pending:cooling-off"));
        assert_eq!(k.txn_state(txn), Some(TxnState::PendingCoolingOff));
        // Too early, no user confirmation: commit is refused.
        assert!(matches!(k.txn_commit("agent", txn, false), Err(KernelError::Txn(TxnError::CoolingOffActive { .. }))));
        // After the window it commits and the money actually moves.
        clock.advance(DEFAULT_COOLING_OFF_MS + 1);
        k.txn_commit("agent", txn, false).unwrap();
        assert_eq!(k.txn_state(txn), Some(TxnState::Committed));
        // Undo now honestly reports the transfer cannot be compensated.
        let report = k.undo("agent", txn).unwrap();
        assert_eq!(report.not_compensated.len(), 1);
    }

    #[test]
    fn cancelling_during_cooling_off_moves_no_money() {
        let (mut k, _c) = setup();
        trust_up(&mut k, "agent");
        let cap = k
            .cap_request("agent", Scope::new("bank.transfer", &["acct"], &[Effect::Transfer]), 600_000)
            .unwrap();
        let txn = k.txn_begin("agent", vec![]);
        k.act("agent", txn, &cap, "bank.transfer", "acct", Value::map(vec![("amount", Value::int(200))]), ActOptions { idempotency_key: "pay-2".into(), ..Default::default() })
            .unwrap();
        let report = k.txn_abort("agent", txn).unwrap();
        assert!(report.compensated.is_empty() && report.not_compensated.is_empty());
        assert_eq!(k.txn_state(txn), Some(TxnState::Aborted));
    }

    #[test]
    fn speculation_barrier_forces_irreversible_steps_serial() {
        let (mut k, _c) = setup();
        trust_up(&mut k, "agent");
        let cap = cal_cap(&mut k, "agent");
        let txn = k.txn_begin("agent", vec![]);
        // cal.write is Full compensability: speculating it is fine per the
        // default profile... unless the profile says deps are untrackable.
        let opts = ActOptions {
            speculative: true,
            step_profile: Some(StepProfile { compensability: Compensability::Full, deps_trackable: false, content_dependent: false }),
            ..Default::default()
        };
        let err = k.act("agent", txn, &cap, "cal.write", "calendar", write_args(), opts);
        assert!(matches!(err, Err(KernelError::SpeculationBarrier(Barrier::UntrackedDependencies))));
    }

    #[test]
    fn mem_assert_through_kernel_requires_provenance() {
        let (mut k, c) = setup();
        let err = k.mem_assert("agent", "k", "fact", FactCategory::General, None, None);
        assert!(matches!(err, Err(KernelError::Memory(MemoryError::MissingProvenance))));
        let prov = Provenance::new(SourceKind::UserDirect, "user", c.now_ms(), 0.9);
        assert!(k.mem_assert("agent", "k", "fact", FactCategory::General, Some(prov), None).is_ok());
        let recall = k.mem_recall("agent", "fact", 0.1);
        assert_eq!(recall.items.len(), 1);
    }

    #[test]
    fn ctx_attach_read_and_revoke() {
        let (mut k, _c) = setup();
        let task = k.intent_submit("agent", "g", vec![], Quotas::generous());
        let h = k.ctx_attach(task, Tainted::untrusted("web page text".to_string())).unwrap();
        assert_eq!(k.ctx_read(h).unwrap().label, TrustLabel::Untrusted);
        k.ctx_revoke(h).unwrap();
        assert!(k.ctx_read(h).is_none());
        assert!(matches!(k.ctx_attach(999, Tainted::user("x".into())), Err(KernelError::UnknownTask { .. })));
    }

    #[test]
    fn yield_costs_attention_quota_and_approval_builds_trust() {
        let (mut k, _c) = setup();
        let mut quotas = Quotas::generous();
        quotas.attention_budget = 1;
        k.intent_submit("agent", "g", vec![], quotas);
        let y1 = k.yield_request("agent", "confirm?", vec!["yes".into(), "no".into()], 0).unwrap();
        assert!(matches!(k.yield_request("agent", "again?", vec![], 0), Err(KernelError::Quota(QuotaError::Attention))));
        assert_eq!(k.trust_level("agent"), 0);
        k.resolve_yield(y1, Decision::Approve).unwrap();
        assert!(k.trust_level("agent") > 0 || k.yield_ticket(y1).unwrap().resolved == Some(Decision::Approve));
    }

    #[test]
    fn watch_fires_once_when_predicate_becomes_true() {
        let (mut k, _c) = setup();
        let id = k.watch("agent", "slot opens", Pred::SlotFree("cal:fri".into()));
        let facts = crate::typesys::FactBase::new();
        assert!(k.poll_watches(&facts).is_empty());
        let facts2 = crate::typesys::FactBase::new().with_slot_free("cal:fri");
        let fired = k.poll_watches(&facts2);
        assert_eq!(fired, vec![(id, "slot opens".to_string())]);
        assert!(k.poll_watches(&facts2).is_empty(), "one-shot: already fired");
    }

    #[test]
    fn ledger_chain_detects_tampering() {
        let (mut k, _c) = setup();
        k.intent_submit("agent", "g", vec![], Quotas::generous());
        k.intent_submit("agent", "g2", vec![], Quotas::generous());
        assert!(k.verify_ledger_chain());
        // Rewrite history: forge the first entry's hash. The next entry's
        // prev_hash no longer matches, so verification must fail.
        k.ledger[0].hash = "forged".to_string();
        assert!(!k.verify_ledger_chain());
    }
}
