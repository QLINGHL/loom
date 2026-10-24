//! The intent compiler's effect type system (doc ch. 7).
//!
//! Plans are data (a structured IR), and before any step executes, this
//! checker statically rejects bad plans at zero cost:
//!
//! * **Linear effects.** `send`, `transfer` and `delete` are typed with a
//!   linear arrow (⊸): a given resource may be consumed by such an effect at
//!   most once per plan. "Send the same draft twice" does not type-check.
//! * **Capability reachability.** The plan's required effect set must be a
//!   subset of the agent's held capabilities; otherwise compilation fails
//!   *before* step 1 runs, instead of dying halfway through.
//! * **Risk is derived, never asserted.** The plan's risk level is computed
//!   from its effect row by a fixed table, so no prompt injection can talk
//!   the system into believing a transfer is "low risk".

use crate::taint::TrustLabel;
use crate::Value;
use std::collections::{BTreeSet, HashMap, HashSet};

/// A single kind of effect a tool can have on the world.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Effect {
    Read,
    Write,
    Send,
    Publish,
    Transfer,
    Delete,
}

impl Effect {
    pub fn name(self) -> &'static str {
        match self {
            Effect::Read => "read",
            Effect::Write => "write",
            Effect::Send => "send",
            Effect::Publish => "publish",
            Effect::Transfer => "transfer",
            Effect::Delete => "delete",
        }
    }

    /// Linear effects (⊸) consume their resource exactly once: repeating
    /// them on the same resource inside one plan is a type error.
    pub fn is_linear(self) -> bool {
        matches!(self, Effect::Send | Effect::Transfer | Effect::Delete)
    }

    /// Effects that change the world (everything except pure reads). These
    /// are what the side-effect quota and the action ledger account for.
    pub fn is_side_effect(self) -> bool {
        !matches!(self, Effect::Read)
    }
}

/// Risk level of a plan, derived from its effect row (doc ch. 7 table).
/// `Ord` so gates can write `risk >= Risk::High`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Risk {
    Low,
    Medium,
    High,
    Critical,
}

impl Risk {
    pub fn name(self) -> &'static str {
        match self {
            Risk::Low => "low",
            Risk::Medium => "medium",
            Risk::High => "high",
            Risk::Critical => "critical",
        }
    }
}

/// The fixed derivation table of doc ch. 7. No model output can influence
/// this function, which is precisely the point.
pub fn derive_risk(effects: &BTreeSet<Effect>) -> Risk {
    if effects.contains(&Effect::Transfer) {
        Risk::Critical
    } else if effects.contains(&Effect::Send)
        || effects.contains(&Effect::Publish)
        || effects.contains(&Effect::Delete)
    {
        Risk::High
    } else if effects.contains(&Effect::Write) {
        Risk::Medium
    } else {
        Risk::Low
    }
}

/// How reversible an effect is (doc ch. 15). Decided by the tool manifest,
/// never by the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compensability {
    /// Fully undoable at any time (local file writes, drafts).
    Full,
    /// Undoable only inside a time window (e.g. message recall).
    Partial { window_ms: u64 },
    /// Cannot be undone (payments, sent mail, physical actions).
    Irreversible,
}

/// Freshness contract of a tool, declared in its manifest (doc ch. 11).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Consistency {
    /// Always re-read; caching is forbidden (payments, balances).
    Strong,
    /// Read-your-writes must hold (calendar, tasks, files).
    Causal,
    /// Stale reads acceptable within the TTL (weather, news).
    Eventual { ttl_ms: u64 },
    /// Local sensors; served directly, never cached.
    BestEffort,
}

/// An executable world-state assertion attached to a plan or a macro.
/// Predicates are a closed enum — deliberately *not* natural language — so
/// they can be checked deterministically at the kernel boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Predicate {
    /// The named slot/resource is currently free.
    SlotFree(String),
    /// A named counter is strictly below the bound.
    CountLessThan(String, i64),
    /// A named counter is at least the bound.
    CountAtLeast(String, i64),
    /// A named object exists.
    Exists(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Condition {
    pub name: String,
    pub predicate: Predicate,
}

impl Condition {
    pub fn new(name: impl Into<String>, predicate: Predicate) -> Self {
        Condition { name: name.into(), predicate }
    }

    pub fn eval(&self, facts: &dyn Facts) -> bool {
        match &self.predicate {
            Predicate::SlotFree(k) => facts.slot_free(k),
            Predicate::CountLessThan(k, bound) => facts.count(k) < *bound,
            Predicate::CountAtLeast(k, bound) => facts.count(k) >= *bound,
            Predicate::Exists(k) => facts.exists(k),
        }
    }
}

/// Read-only view of world state that conditions evaluate against.
pub trait Facts {
    fn slot_free(&self, key: &str) -> bool;
    fn count(&self, key: &str) -> i64;
    fn exists(&self, key: &str) -> bool;
}

/// A simple in-memory [`Facts`] implementation for tests, demos and macro
/// precondition checks.
#[derive(Debug, Default, Clone)]
pub struct FactBase {
    pub free_slots: HashSet<String>,
    pub counts: HashMap<String, i64>,
    pub present: HashSet<String>,
}

impl FactBase {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_slot_free(mut self, key: &str) -> Self {
        self.free_slots.insert(key.to_string());
        self
    }
    pub fn with_count(mut self, key: &str, n: i64) -> Self {
        self.counts.insert(key.to_string(), n);
        self
    }
    pub fn with_present(mut self, key: &str) -> Self {
        self.present.insert(key.to_string());
        self
    }
}

impl Facts for FactBase {
    fn slot_free(&self, key: &str) -> bool {
        self.free_slots.contains(key)
    }
    fn count(&self, key: &str) -> i64 {
        self.counts.get(key).copied().unwrap_or(0)
    }
    fn exists(&self, key: &str) -> bool {
        self.present.contains(key)
    }
}

/// One step of a plan IR: invoke `tool` on `args`, where `resource` names
/// the identity of the principal value the step acts on (a draft id, an
/// account, a file path). Linear effects consume that resource.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanStep {
    pub tool: String,
    pub resource: String,
    pub args: Value,
}

impl PlanStep {
    pub fn new(tool: impl Into<String>, resource: impl Into<String>, args: Value) -> Self {
        PlanStep { tool: tool.into(), resource: resource.into(), args }
    }
}

/// A structured plan — the only thing allowed to cross the airlock between
/// the planner (which may have seen untrusted content) and the executor
/// (which holds the tools). It is data, never prose (doc ch. 13).
#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    pub id: String,
    /// Stable key identifying the *kind* of intent ("trip.arrange"), used by
    /// the determinization engine to recognize repetitions.
    pub intent_key: String,
    pub steps: Vec<PlanStep>,
    pub pre: Vec<Condition>,
    pub post: Vec<Condition>,
    /// Worst trust label of everything the planner used to produce this plan.
    pub taint: TrustLabel,
}

impl Plan {
    pub fn new(id: impl Into<String>, intent_key: impl Into<String>, taint: TrustLabel) -> Self {
        Plan { id: id.into(), intent_key: intent_key.into(), steps: Vec::new(), pre: Vec::new(), post: Vec::new(), taint }
    }
    pub fn step(mut self, step: PlanStep) -> Self {
        self.steps.push(step);
        self
    }
    pub fn pre_condition(mut self, c: Condition) -> Self {
        self.pre.push(c);
        self
    }
    pub fn post_condition(mut self, c: Condition) -> Self {
        self.post.push(c);
        self
    }
}

/// A plan that survived the checker, bundled with what the checker proved
/// about it. Executors should accept `CheckedPlan`, not raw `Plan`.
#[derive(Debug, Clone)]
pub struct CheckedPlan {
    pub plan: Plan,
    pub risk: Risk,
    /// Every (tool, effect) pair the plan needs — the reachability proof's
    /// witness set.
    pub required: BTreeSet<(String, Effect)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeError {
    UnknownTool { tool: String },
    /// The same linear resource would be consumed twice in one plan.
    LinearReuse { effect: Effect, resource: String },
    /// The plan needs an effect the agent does not hold a capability for.
    MissingCapability { tool: String, effect: Effect },
}

impl std::fmt::Display for TypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TypeError::UnknownTool { tool } => write!(f, "unknown tool: {tool}"),
            TypeError::LinearReuse { effect, resource } => {
                write!(f, "linear effect {} would consume resource {resource} twice", effect.name())
            }
            TypeError::MissingCapability { tool, effect } => {
                write!(f, "plan requires {}:{} but the agent holds no such capability", tool, effect.name())
            }
        }
    }
}
impl std::error::Error for TypeError {}

/// The static checker. Holds the effect signature of every known tool.
pub struct TypeChecker {
    signatures: HashMap<String, BTreeSet<Effect>>,
}

impl TypeChecker {
    pub fn new() -> Self {
        TypeChecker { signatures: HashMap::new() }
    }

    pub fn register(&mut self, tool: &str, effects: BTreeSet<Effect>) {
        self.signatures.insert(tool.to_string(), effects);
    }

    pub fn effects_of(&self, tool: &str) -> Option<&BTreeSet<Effect>> {
        self.signatures.get(tool)
    }

    /// Statically check `plan` against the capabilities in `held`
    /// (a set of (tool, effect) pairs). Pure: no side effects, no I/O —
    /// a rejected plan costs nothing and changes nothing.
    pub fn check(
        &self,
        plan: &Plan,
        held: &BTreeSet<(String, Effect)>,
    ) -> Result<CheckedPlan, TypeError> {
        let mut required: BTreeSet<(String, Effect)> = BTreeSet::new();
        let mut all_effects: BTreeSet<Effect> = BTreeSet::new();
        let mut consumed: HashSet<(Effect, String)> = HashSet::new();

        for step in &plan.steps {
            let effects = self
                .signatures
                .get(&step.tool)
                .ok_or_else(|| TypeError::UnknownTool { tool: step.tool.clone() })?;
            for &effect in effects {
                required.insert((step.tool.clone(), effect));
                all_effects.insert(effect);
                if effect.is_linear() {
                    let key = (effect, step.resource.clone());
                    if !consumed.insert(key) {
                        return Err(TypeError::LinearReuse {
                            effect,
                            resource: step.resource.clone(),
                        });
                    }
                }
            }
        }

        for (tool, effect) in &required {
            if !held.contains(&(tool.clone(), *effect)) {
                return Err(TypeError::MissingCapability {
                    tool: tool.clone(),
                    effect: *effect,
                });
            }
        }

        Ok(CheckedPlan { plan: plan.clone(), risk: derive_risk(&all_effects), required })
    }
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checker() -> TypeChecker {
        let mut tc = TypeChecker::new();
        tc.register("mail.search", [Effect::Read].into_iter().collect());
        tc.register("mail.send", [Effect::Send].into_iter().collect());
        tc.register("file.read", [Effect::Read].into_iter().collect());
        tc.register("file.write", [Effect::Write].into_iter().collect());
        tc.register("bank.transfer", [Effect::Transfer].into_iter().collect());
        tc.register("file.delete", [Effect::Delete].into_iter().collect());
        tc
    }

    fn held(pairs: &[(&str, Effect)]) -> BTreeSet<(String, Effect)> {
        pairs.iter().map(|(t, e)| (t.to_string(), *e)).collect()
    }

    fn plan_with(steps: Vec<PlanStep>) -> Plan {
        let mut p = Plan::new("p1", "test.intent", TrustLabel::User);
        p.steps = steps;
        p
    }

    #[test]
    fn risk_table_read_only_is_low() {
        let tc = checker();
        let p = plan_with(vec![PlanStep::new("mail.search", "inbox", Value::Null)]);
        let checked = tc.check(&p, &held(&[("mail.search", Effect::Read)])).unwrap();
        assert_eq!(checked.risk, Risk::Low);
    }

    #[test]
    fn risk_table_local_write_is_medium() {
        let tc = checker();
        let p = plan_with(vec![
            PlanStep::new("file.read", "a.txt", Value::Null),
            PlanStep::new("file.write", "b.txt", Value::Null),
        ]);
        let checked = tc
            .check(&p, &held(&[("file.read", Effect::Read), ("file.write", Effect::Write)]))
            .unwrap();
        assert_eq!(checked.risk, Risk::Medium);
    }

    #[test]
    fn risk_table_send_and_delete_are_high() {
        let tc = checker();
        let p = plan_with(vec![PlanStep::new("mail.send", "draft-1", Value::Null)]);
        let checked = tc.check(&p, &held(&[("mail.send", Effect::Send)])).unwrap();
        assert_eq!(checked.risk, Risk::High);
        let p2 = plan_with(vec![PlanStep::new("file.delete", "old.log", Value::Null)]);
        let checked2 = tc.check(&p2, &held(&[("file.delete", Effect::Delete)])).unwrap();
        assert_eq!(checked2.risk, Risk::High);
    }

    #[test]
    fn risk_table_transfer_is_critical_and_dominates() {
        let tc = checker();
        let p = plan_with(vec![
            PlanStep::new("file.read", "a.txt", Value::Null),
            PlanStep::new("bank.transfer", "acct-1", Value::Null),
        ]);
        let checked = tc
            .check(&p, &held(&[("file.read", Effect::Read), ("bank.transfer", Effect::Transfer)]))
            .unwrap();
        assert_eq!(checked.risk, Risk::Critical);
    }

    #[test]
    fn sending_the_same_draft_twice_does_not_typecheck() {
        let tc = checker();
        let p = plan_with(vec![
            PlanStep::new("mail.send", "draft-1", Value::Null),
            PlanStep::new("mail.send", "draft-1", Value::Null),
        ]);
        let err = tc.check(&p, &held(&[("mail.send", Effect::Send)])).unwrap_err();
        assert_eq!(
            err,
            TypeError::LinearReuse { effect: Effect::Send, resource: "draft-1".to_string() }
        );
    }

    #[test]
    fn sending_two_different_drafts_is_legal() {
        let tc = checker();
        let p = plan_with(vec![
            PlanStep::new("mail.send", "draft-1", Value::Null),
            PlanStep::new("mail.send", "draft-2", Value::Null),
        ]);
        assert!(tc.check(&p, &held(&[("mail.send", Effect::Send)])).is_ok());
    }

    #[test]
    fn transferring_twice_from_same_account_fails() {
        let tc = checker();
        let p = plan_with(vec![
            PlanStep::new("bank.transfer", "acct-1", Value::Null),
            PlanStep::new("bank.transfer", "acct-1", Value::Null),
        ]);
        let h = held(&[("bank.transfer", Effect::Transfer)]);
        assert!(matches!(tc.check(&p, &h), Err(TypeError::LinearReuse { .. })));
    }

    #[test]
    fn missing_capability_fails_at_compile_time() {
        let tc = checker();
        // Plan needs send:Mail but the agent only holds read:File + write:Ledger
        // (the worked example from doc ch. 7).
        let p = plan_with(vec![
            PlanStep::new("file.read", "invoice.pdf", Value::Null),
            PlanStep::new("mail.send", "draft-9", Value::Null),
        ]);
        let h = held(&[("file.read", Effect::Read)]);
        let err = tc.check(&p, &h).unwrap_err();
        assert_eq!(
            err,
            TypeError::MissingCapability { tool: "mail.send".to_string(), effect: Effect::Send }
        );
    }

    #[test]
    fn unknown_tool_is_rejected() {
        let tc = checker();
        let p = plan_with(vec![PlanStep::new("root.shell", "host", Value::Null)]);
        assert!(matches!(tc.check(&p, &held(&[])), Err(TypeError::UnknownTool { .. })));
    }

    #[test]
    fn conditions_evaluate_against_facts() {
        let facts = FactBase::new()
            .with_slot_free("cal:2026-11-02T09")
            .with_count("mail.unread", 12);
        let free = Condition::new("slot free", Predicate::SlotFree("cal:2026-11-02T09".into()));
        let few_unread = Condition::new("few unread", Predicate::CountLessThan("mail.unread".into(), 50));
        let many_unread = Condition::new("many unread", Predicate::CountAtLeast("mail.unread".into(), 50));
        assert!(free.eval(&facts));
        assert!(few_unread.eval(&facts));
        assert!(!many_unread.eval(&facts));
    }

    #[test]
    fn derive_risk_empty_effects_is_low() {
        assert_eq!(derive_risk(&BTreeSet::new()), Risk::Low);
    }
}
