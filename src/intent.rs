//! The intent compiler (doc ch. 7): turn a natural-language intent into an
//! executable plan *without* producing any side effect along the way.
//!
//! The pipeline the design doc sketches:
//!
//! ```text
//! natural language
//!   ↓ ① parse
//! intent signature { action, object, scope, success criterion, budget }
//!   ↓ ② retrieve
//! candidate plan (plan library / plan cache / skill library) — a hit
//! skips synthesis entirely, so a repeated intent costs zero model calls
//!   ↓ ③ synthesize (only on a miss; the LLM slot — see below)
//! plan IR (structured, never prose)
//!   ↓ ④ type-check  ← the key step: a zero-side-effect stage
//!   ↓ ⑤ cost & risk estimate
//! execute / clarify / reject
//! ```
//!
//! What this module owns is stages ①, ②, ④ and ⑤ plus the pre/post
//! condition checks that bracket execution. Stage ③, synthesis, is
//! deliberately *not* implemented here: synthesizing a plan is a model
//! judgement, and the kernel never calls a model. The caller hands a
//! synthesized [`Plan`](crate::typesys::Plan) in; the compiler treats it as
//! untrusted data and proves it safe before anything runs. A cached plan is
//! preferred over a synthesized one whenever both exist — determinization
//! (ch. 4) is the point of the whole system, and the compiler is where the
//! saving is booked: a cache hit costs zero model calls.
//!
//! The three guarantees the doc asks the compiler to provide:
//!
//! * **Bad plans die at zero cost.** Type errors (linear reuse, missing
//!   capability, unknown tool), failed preconditions and over-budget plans
//!   are all rejected before step 1 runs — never halfway through, never
//!   after a side effect.
//! * **Risk is derived, never asserted.** The
//!   [`Disposition`] (execute directly / execute reversibly / confirm /
//!   confirm + cooling-off + second factor) comes from the plan's effect
//!   row through the fixed table in [`crate::typesys::derive_risk`], so no
//!   prompt injection can talk the system into executing a transfer
//!   without confirmation.
//! * **Pre/post conditions bracket execution.** A plan carries executable
//!   world-state assertions; preconditions are checked before execution
//!   (fail → replan, not half-execute) and postconditions after (fail → the
//!   caller is told exactly which assertion failed, so compensation can run).
//!   This is what makes zero-model macros safe to run at all.

use crate::typesys::{CheckedPlan, Effect, Facts, Plan, Risk, TypeChecker, TypeError};
use std::collections::{BTreeSet, HashMap};

/// What the intent may cost. A plan that exceeds any bound is rejected at
/// compile time, before it can spend anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budget {
    /// Maximum number of plan steps.
    pub max_steps: usize,
    /// Maximum model calls the compilation may spend. A plan-library hit
    /// costs 0; a synthesized plan costs 1 (the synthesis call itself).
    pub max_model_calls: u32,
    /// Maximum number of steps that carry a side effect.
    pub max_side_effects: usize,
}

impl Budget {
    pub fn new(max_steps: usize, max_model_calls: u32, max_side_effects: usize) -> Self {
        Budget { max_steps, max_model_calls, max_side_effects }
    }
}

impl Default for Budget {
    fn default() -> Self {
        Budget { max_steps: 16, max_model_calls: 1, max_side_effects: 4 }
    }
}

/// Stage ① output: the parsed form of a natural-language intent. The
/// parser itself is a model judgement and lives outside the kernel; this is
/// the struct it must produce, so the compiler can stay deterministic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntentSignature {
    /// Stable key for the kind of intent ("invoice.archive"), used to look
    /// a candidate plan up in the plan library.
    pub key: String,
    /// What to do ("archive").
    pub action: String,
    /// What to act on ("invoices").
    pub object: String,
    /// Scope of the action, if any ("last month").
    pub scope: Option<String>,
    /// How the user will know it worked ("expense report generated").
    pub success: String,
    /// What the compilation may spend.
    pub budget: Budget,
}

impl IntentSignature {
    pub fn new(
        key: impl Into<String>,
        action: impl Into<String>,
        object: impl Into<String>,
        success: impl Into<String>,
    ) -> Self {
        IntentSignature {
            key: key.into(),
            action: action.into(),
            object: object.into(),
            scope: None,
            success: success.into(),
            budget: Budget::default(),
        }
    }

    pub fn scope(mut self, scope: impl Into<String>) -> Self {
        self.scope = Some(scope.into());
        self
    }

    pub fn budget(mut self, budget: Budget) -> Self {
        self.budget = budget;
        self
    }

    /// Names of the required fields that are blank. Empty means the
    /// signature is complete enough to compile.
    pub fn missing_fields(&self) -> Vec<&'static str> {
        let mut missing = Vec::new();
        if self.key.trim().is_empty() {
            missing.push("key");
        }
        if self.action.trim().is_empty() {
            missing.push("action");
        }
        if self.object.trim().is_empty() {
            missing.push("object");
        }
        if self.success.trim().is_empty() {
            missing.push("success");
        }
        missing
    }
}

/// Where the compiled plan came from. The distinction is the cost story:
/// a cached plan needed no model at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanSource {
    /// Retrieved from the plan library (macro / plan cache / skill library).
    Cached,
    /// Handed in by the caller after model synthesis.
    Synthesized,
}

/// Stage ⑤ output: what the plan costs to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CostEstimate {
    /// Number of plan steps.
    pub steps: usize,
    /// Number of steps carrying a side effect.
    pub side_effects: usize,
    /// Model calls spent compiling: 0 for a cache hit, 1 for synthesis.
    pub model_calls: u32,
    /// Whether the plan came from the library.
    pub cache_hit: bool,
}

/// The default handling for a checked plan, derived from its risk level by
/// the doc ch. 7 table. No model output participates in this derivation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Disposition {
    /// Read-only plan: run it.
    Execute,
    /// Local writes only: run it; it can be undone afterwards.
    ExecuteReversible,
    /// Outbound effects (send / publish / delete): ask first.
    Confirm,
    /// Money movement: confirm + cooling-off period + second verification.
    ConfirmCoolingOffSecondFactor,
}

impl Disposition {
    /// The fixed derivation table of doc ch. 7 (risk → default handling).
    pub fn for_risk(risk: Risk) -> Self {
        match risk {
            Risk::Low => Disposition::Execute,
            Risk::Medium => Disposition::ExecuteReversible,
            Risk::High => Disposition::Confirm,
            Risk::Critical => Disposition::ConfirmCoolingOffSecondFactor,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Disposition::Execute => "execute",
            Disposition::ExecuteReversible => "execute-reversible",
            Disposition::Confirm => "confirm",
            Disposition::ConfirmCoolingOffSecondFactor => {
                "confirm-cooling-off-second-factor"
            }
        }
    }
}

/// Why compilation failed. Every variant is a *zero-cost* failure: nothing
/// executed, nothing changed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompileError {
    /// The plan does not type-check (linear reuse, missing capability,
    /// unknown tool). The inner error says which.
    Type(TypeError),
    /// A precondition does not hold against current world state. The plan
    /// is not run; the caller should replan, not half-execute.
    PreconditionFailed { condition: String },
    /// The plan exceeds the intent's budget.
    OverBudget { what: &'static str, limit: usize, actual: usize },
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompileError::Type(e) => write!(f, "type error: {e}"),
            CompileError::PreconditionFailed { condition } => {
                write!(f, "precondition failed: {condition}")
            }
            CompileError::OverBudget { what, limit, actual } => {
                write!(f, "over budget: {what} {actual} exceeds limit {limit}")
            }
        }
    }
}
impl std::error::Error for CompileError {}

/// The result of running the compiler on one intent.
#[derive(Debug)]
pub enum CompileOutcome {
    /// The plan type-checked, its preconditions hold, and it fits the
    /// budget. Ready to hand to the executor with the given disposition.
    Ready {
        plan: CheckedPlan,
        disposition: Disposition,
        cost: CostEstimate,
        source: PlanSource,
    },
    /// The intent signature is missing a required field; the user (or the
    /// parser) must supply it before compilation can start.
    NeedsClarification { reason: String },
    /// No candidate plan exists in the library and none was synthesized;
    /// synthesis is required before this intent can compile.
    NeedsSynthesis { key: String },
    /// Compilation failed at zero cost; nothing executed.
    Rejected { error: CompileError },
}

/// The intent compiler: stages ①, ②, ④, ⑤ of the doc ch. 7 pipeline.
///
/// Holds the tool effect signatures (via [`TypeChecker`]), the set of
/// (tool, effect) capabilities the agent holds, and the plan library used
/// for stage ② retrieval. Pure: compiling never performs a side effect.
pub struct IntentCompiler {
    checker: TypeChecker,
    held: BTreeSet<(String, Effect)>,
    library: HashMap<String, Plan>,
}

impl IntentCompiler {
    pub fn new() -> Self {
        IntentCompiler {
            checker: TypeChecker::new(),
            held: BTreeSet::new(),
            library: HashMap::new(),
        }
    }

    /// Register a tool's effect signature (stage ④ needs it).
    pub fn register_tool(&mut self, tool: &str, effects: BTreeSet<Effect>) {
        self.checker.register(tool, effects);
    }

    /// Grant the agent a (tool, effect) capability (stage ④ reachability).
    pub fn grant(&mut self, tool: &str, effect: Effect) {
        self.held.insert((tool.to_string(), effect));
    }

    /// Store a candidate plan in the library, keyed by its intent key
    /// (stage ② retrieval source).
    pub fn register_plan(&mut self, plan: Plan) {
        self.library.insert(plan.intent_key.clone(), plan);
    }

    /// Compile one intent. `synthesized` is the plan stage ③ produced, if
    /// any; a library hit always wins over it (zero model calls beat one).
    /// `facts` is the world state the preconditions are checked against.
    pub fn compile(
        &self,
        sig: &IntentSignature,
        facts: &dyn Facts,
        synthesized: Option<Plan>,
    ) -> CompileOutcome {
        // ① The signature must be complete, or there is nothing to compile.
        let missing = sig.missing_fields();
        if !missing.is_empty() {
            return CompileOutcome::NeedsClarification {
                reason: format!("intent is missing: {}", missing.join(", ")),
            };
        }

        // ② Retrieve a candidate plan; a library hit costs zero model calls.
        let (plan, source) = match self.library.get(&sig.key) {
            Some(cached) => (cached.clone(), PlanSource::Cached),
            None => match synthesized {
                Some(plan) => (plan, PlanSource::Synthesized),
                None => return CompileOutcome::NeedsSynthesis { key: sig.key.clone() },
            },
        };

        // ④ Type-check: the zero-side-effect stage. A bad plan dies here.
        let checked = match self.checker.check(&plan, &self.held) {
            Ok(checked) => checked,
            Err(e) => return CompileOutcome::Rejected { error: CompileError::Type(e) },
        };

        // Preconditions: checked against the world before anything runs.
        if let Some(failed) = plan.pre.iter().find(|c| !c.eval(facts)) {
            return CompileOutcome::Rejected {
                error: CompileError::PreconditionFailed { condition: failed.name.clone() },
            };
        }

        // ⑤ Cost estimate, then the budget check.
        let cost = self.estimate(&plan, source);
        if cost.steps > sig.budget.max_steps {
            return CompileOutcome::Rejected {
                error: CompileError::OverBudget {
                    what: "steps",
                    limit: sig.budget.max_steps,
                    actual: cost.steps,
                },
            };
        }
        if cost.model_calls > sig.budget.max_model_calls {
            return CompileOutcome::Rejected {
                error: CompileError::OverBudget {
                    what: "model calls",
                    limit: sig.budget.max_model_calls as usize,
                    actual: cost.model_calls as usize,
                },
            };
        }
        if cost.side_effects > sig.budget.max_side_effects {
            return CompileOutcome::Rejected {
                error: CompileError::OverBudget {
                    what: "side effects",
                    limit: sig.budget.max_side_effects,
                    actual: cost.side_effects,
                },
            };
        }

        // ⑤ Disposition is derived from the risk, never asserted.
        CompileOutcome::Ready {
            disposition: Disposition::for_risk(checked.risk),
            plan: checked,
            cost,
            source,
        }
    }

    /// Estimate what a plan costs. A library hit costs 0 model calls; a
    /// synthesized plan costs 1 (the synthesis call itself).
    fn estimate(&self, plan: &Plan, source: PlanSource) -> CostEstimate {
        let side_effects = plan
            .steps
            .iter()
            .filter(|step| {
                self.checker
                    .effects_of(&step.tool)
                    .map(|effects| effects.iter().any(|e| e.is_side_effect()))
                    .unwrap_or(false)
            })
            .count();
        CostEstimate {
            steps: plan.steps.len(),
            side_effects,
            model_calls: match source {
                PlanSource::Cached => 0,
                PlanSource::Synthesized => 1,
            },
            cache_hit: source == PlanSource::Cached,
        }
    }

    /// Check a plan's postconditions against the world after execution.
    /// Returns the names of the failed assertions; empty means every
    /// postcondition holds. A non-empty result means compensation should run.
    pub fn verify_post(&self, plan: &Plan, facts: &dyn Facts) -> Vec<String> {
        plan.post
            .iter()
            .filter(|c| !c.eval(facts))
            .map(|c| c.name.clone())
            .collect()
    }

    /// Check a plan's preconditions on their own (useful for a replan loop
    /// that wants the failing names without compiling the whole plan).
    pub fn verify_pre(&self, plan: &Plan, facts: &dyn Facts) -> Vec<String> {
        plan.pre
            .iter()
            .filter(|c| !c.eval(facts))
            .map(|c| c.name.clone())
            .collect()
    }
}

impl Default for IntentCompiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typesys::{Condition, FactBase, PlanStep, Predicate};
    use crate::Value;

    fn compiler() -> IntentCompiler {
        let mut c = IntentCompiler::new();
        c.register_tool("mail.search", [Effect::Read].into_iter().collect());
        c.register_tool("mail.send", [Effect::Send].into_iter().collect());
        c.register_tool("file.read", [Effect::Read].into_iter().collect());
        c.register_tool("file.write", [Effect::Write].into_iter().collect());
        c.register_tool("ledger.write", [Effect::Write].into_iter().collect());
        c.register_tool("bank.transfer", [Effect::Transfer].into_iter().collect());
        for (tool, effect) in [
            ("mail.search", Effect::Read),
            ("mail.send", Effect::Send),
            ("file.read", Effect::Read),
            ("file.write", Effect::Write),
            ("ledger.write", Effect::Write),
            ("bank.transfer", Effect::Transfer),
        ] {
            c.grant(tool, effect);
        }
        c
    }

    fn archive_plan() -> Plan {
        // The doc ch. 7 worked example: archive invoices.
        Plan::new("p-archive", "invoice.archive", crate::taint::TrustLabel::User)
            .step(PlanStep::new("file.read", "invoice.pdf", Value::Null))
            .step(PlanStep::new("ledger.write", "ledger", Value::Null))
            .step(PlanStep::new("mail.send", "draft-1", Value::Null))
            .pre_condition(Condition::new(
                "few unread",
                Predicate::CountLessThan("mail.unread".into(), 50),
            ))
            .post_condition(Condition::new(
                "report exists",
                Predicate::Exists("report".into()),
            ))
    }

    fn sig() -> IntentSignature {
        IntentSignature::new(
            "invoice.archive",
            "archive",
            "invoices",
            "expense report generated",
        )
        .scope("last month")
    }

    fn facts_ok() -> FactBase {
        FactBase::new().with_count("mail.unread", 12)
    }

    #[test]
    fn cached_plan_compiles_with_zero_model_calls_and_high_disposition() {
        let mut c = compiler();
        c.register_plan(archive_plan());
        let out = c.compile(&sig(), &facts_ok(), None);
        match out {
            CompileOutcome::Ready { plan, disposition, cost, source } => {
                assert_eq!(source, PlanSource::Cached);
                assert!(cost.cache_hit);
                assert_eq!(cost.model_calls, 0);
                assert_eq!(cost.steps, 3);
                assert_eq!(cost.side_effects, 2);
                // send:Mail in the effect row -> High -> confirm first.
                assert_eq!(plan.risk, Risk::High);
                assert_eq!(disposition, Disposition::Confirm);
            }
            other => panic!("expected Ready, got {other:?}"),
        }
    }

    #[test]
    fn library_hit_beats_a_synthesized_plan() {
        let mut c = compiler();
        c.register_plan(archive_plan());
        let synth = Plan::new("p-synth", "invoice.archive", crate::taint::TrustLabel::User)
            .step(PlanStep::new("file.read", "invoice.pdf", Value::Null));
        let out = c.compile(&sig(), &facts_ok(), Some(synth));
        match out {
            CompileOutcome::Ready { source, cost, .. } => {
                assert_eq!(source, PlanSource::Cached);
                assert_eq!(cost.model_calls, 0);
            }
            other => panic!("expected Ready from cache, got {other:?}"),
        }
    }

    #[test]
    fn synthesized_plan_costs_one_model_call() {
        let c = compiler();
        let out = c.compile(&sig(), &facts_ok(), Some(archive_plan()));
        match out {
            CompileOutcome::Ready { source, cost, .. } => {
                assert_eq!(source, PlanSource::Synthesized);
                assert_eq!(cost.model_calls, 1);
                assert!(!cost.cache_hit);
            }
            other => panic!("expected Ready, got {other:?}"),
        }
    }

    #[test]
    fn read_only_plan_executes_directly() {
        let c = compiler();
        let plan = Plan::new("p-read", "mail.search", crate::taint::TrustLabel::User)
            .step(PlanStep::new("mail.search", "inbox", Value::Null));
        let s = IntentSignature::new("mail.search", "search", "mail", "results shown");
        let out = c.compile(&s, &facts_ok(), Some(plan));
        match out {
            CompileOutcome::Ready { disposition, cost, .. } => {
                assert_eq!(disposition, Disposition::Execute);
                assert_eq!(cost.side_effects, 0);
            }
            other => panic!("expected Ready, got {other:?}"),
        }
    }

    #[test]
    fn local_write_plan_is_reversible() {
        let c = compiler();
        let plan = Plan::new("p-write", "file.write", crate::taint::TrustLabel::User)
            .step(PlanStep::new("file.write", "notes.txt", Value::Null));
        let s = IntentSignature::new("file.write", "write", "notes", "file updated");
        let out = c.compile(&s, &facts_ok(), Some(plan));
        match out {
            CompileOutcome::Ready { disposition, .. } => {
                assert_eq!(disposition, Disposition::ExecuteReversible)
            }
            other => panic!("expected Ready, got {other:?}"),
        }
    }

    #[test]
    fn transfer_plan_needs_confirm_cooling_off_and_second_factor() {
        let c = compiler();
        let plan = Plan::new("p-pay", "bill.pay", crate::taint::TrustLabel::User)
            .step(PlanStep::new("bank.transfer", "acct-1", Value::Null));
        let s = IntentSignature::new("bill.pay", "pay", "bill", "payment sent");
        let out = c.compile(&s, &facts_ok(), Some(plan));
        match out {
            CompileOutcome::Ready { disposition, .. } => {
                assert_eq!(disposition, Disposition::ConfirmCoolingOffSecondFactor)
            }
            other => panic!("expected Ready, got {other:?}"),
        }
    }

    #[test]
    fn incomplete_signature_needs_clarification() {
        let c = compiler();
        let s = IntentSignature::new("invoice.archive", "", "invoices", "done");
        let out = c.compile(&s, &facts_ok(), Some(archive_plan()));
        match out {
            CompileOutcome::NeedsClarification { reason } => {
                assert!(reason.contains("action"), "reason: {reason}")
            }
            other => panic!("expected NeedsClarification, got {other:?}"),
        }
    }

    #[test]
    fn no_candidate_plan_needs_synthesis() {
        let c = compiler();
        let out = c.compile(&sig(), &facts_ok(), None);
        assert!(matches!(out, CompileOutcome::NeedsSynthesis { key } if key == "invoice.archive"));
    }

    #[test]
    fn double_send_is_rejected_at_zero_cost() {
        let c = compiler();
        let plan = Plan::new("p-dup", "mail.blast", crate::taint::TrustLabel::User)
            .step(PlanStep::new("mail.send", "draft-1", Value::Null))
            .step(PlanStep::new("mail.send", "draft-1", Value::Null));
        let s = IntentSignature::new("mail.blast", "send", "mail", "mail sent");
        let out = c.compile(&s, &facts_ok(), Some(plan));
        match out {
            CompileOutcome::Rejected { error: CompileError::Type(TypeError::LinearReuse { .. }) } => {}
            other => panic!("expected linear-reuse rejection, got {other:?}"),
        }
    }

    #[test]
    fn missing_capability_is_rejected_before_execution() {
        // A fresh compiler that never granted mail.send.
        let mut c = IntentCompiler::new();
        c.register_tool("mail.send", [Effect::Send].into_iter().collect());
        let plan = Plan::new("p-send", "mail.send", crate::taint::TrustLabel::User)
            .step(PlanStep::new("mail.send", "draft-1", Value::Null));
        let s = IntentSignature::new("mail.send", "send", "mail", "mail sent");
        let out = c.compile(&s, &facts_ok(), Some(plan));
        match out {
            CompileOutcome::Rejected {
                error: CompileError::Type(TypeError::MissingCapability { .. }),
            } => {}
            other => panic!("expected missing-capability rejection, got {other:?}"),
        }
    }

    #[test]
    fn failed_precondition_rejects_instead_of_half_executing() {
        let c = compiler();
        let facts = FactBase::new().with_count("mail.unread", 87);
        let out = c.compile(&sig(), &facts, Some(archive_plan()));
        match out {
            CompileOutcome::Rejected {
                error: CompileError::PreconditionFailed { condition },
            } => assert_eq!(condition, "few unread"),
            other => panic!("expected precondition rejection, got {other:?}"),
        }
    }

    #[test]
    fn over_budget_plan_is_rejected() {
        let c = compiler();
        let s = sig().budget(Budget::new(2, 1, 4)); // archive plan has 3 steps
        let out = c.compile(&s, &facts_ok(), Some(archive_plan()));
        match out {
            CompileOutcome::Rejected {
                error: CompileError::OverBudget { what, limit, actual },
            } => {
                assert_eq!(what, "steps");
                assert_eq!((limit, actual), (2, 3));
            }
            other => panic!("expected over-budget rejection, got {other:?}"),
        }
    }

    #[test]
    fn zero_model_call_budget_rejects_synthesis_but_not_cache() {
        let mut c = compiler();
        c.register_plan(archive_plan());
        let s = sig().budget(Budget::new(16, 0, 4));
        // Cache hit: 0 model calls, fits.
        assert!(matches!(c.compile(&s, &facts_ok(), None), CompileOutcome::Ready { .. }));
        // No cache hit for this key: synthesis would cost 1, over budget.
        let s2 = IntentSignature::new("other.task", "do", "thing", "done")
            .budget(Budget::new(16, 0, 4));
        let plan = Plan::new("p-other", "other.task", crate::taint::TrustLabel::User)
            .step(PlanStep::new("file.read", "a.txt", Value::Null));
        let out = c.compile(&s2, &facts_ok(), Some(plan));
        match out {
            CompileOutcome::Rejected {
                error: CompileError::OverBudget { what, .. },
            } => assert_eq!(what, "model calls"),
            other => panic!("expected model-call over-budget, got {other:?}"),
        }
    }

    #[test]
    fn verify_post_reports_exactly_the_failed_assertions() {
        let c = compiler();
        let plan = archive_plan();
        // Report does not exist yet: postcondition fails.
        let failed = c.verify_post(&plan, &facts_ok());
        assert_eq!(failed, vec!["report exists".to_string()]);
        // After execution the report exists: all hold.
        let done = facts_ok().with_present("report");
        assert!(c.verify_post(&plan, &done).is_empty());
    }

    #[test]
    fn verify_pre_names_the_failing_preconditions() {
        let c = compiler();
        let facts = FactBase::new().with_count("mail.unread", 87);
        assert_eq!(c.verify_pre(&archive_plan(), &facts), vec!["few unread".to_string()]);
        assert!(c.verify_pre(&archive_plan(), &facts_ok()).is_empty());
    }

    #[test]
    fn disposition_table_covers_all_risk_levels() {
        assert_eq!(Disposition::for_risk(Risk::Low), Disposition::Execute);
        assert_eq!(Disposition::for_risk(Risk::Medium), Disposition::ExecuteReversible);
        assert_eq!(Disposition::for_risk(Risk::High), Disposition::Confirm);
        assert_eq!(
            Disposition::for_risk(Risk::Critical),
            Disposition::ConfirmCoolingOffSecondFactor
        );
    }
}
