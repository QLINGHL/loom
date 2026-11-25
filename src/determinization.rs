//! The determinization engine (doc ch. 4) — the north-star mechanism.
//! A mature AI OS should need its model *less* over time: every repeated
//! intent gets cheaper, faster and safer, until it needs no model at all.
//!
//! * **L1 plan cache** — a structurally identical intent reuses the plan
//!   IR, skipping the planning stage (parameters are still re-validated
//!   downstream by the type checker on every run).
//! * **L2 intent macros** — after ≥ 5 successful executions *and* explicit
//!   user confirmation, a plan compiles into a deterministic script with
//!   slots. Argument positions that varied across the recorded runs become
//!   slots; everything else is frozen. Macro execution involves no model
//!   anywhere — this module contains no model handle by construction.
//! * **L3 triggers** — a macro that ran stably ≥ 20 times may fire
//!   automatically, but only under an explicit user authorization with a
//!   hard budget (max runs, max side effects), and only while its firing
//!   conditions hold against current world state. Exhaustion disables it.
//!   Firing a trigger *runs the macro* and returns its plan: the trigger
//!   is the automatic path, not a separate permission check.
//!
//! Macros must be able to go stale, or the system ossifies:
//! * every macro carries preconditions — checked before each run; failure
//!   falls back to the probabilistic path;
//! * N consecutive precondition failures demote the macro into the
//!   recompile queue;
//! * every macro is bound to the schema versions of its tools — a tool
//!   interface change invalidates it immediately.
//!
//! One canonical determinization statistic (doc ch. 22/23): every served
//! intent is classified by [`ServedBy`] — L0 model planning, L1 cache,
//! L2 macro, L3 trigger — and the determinization rate is the share of
//! served intents that came from a deterministic rung. The P0 acceptance
//! suite and the metrics dashboard both reuse this definition; the engine
//! is the only place the classification is made, so the three can never
//! drift apart.

use crate::crypto::sha256_hex;
use crate::typesys::{Condition, Facts, Plan};
use crate::Clock;
use crate::Value;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

/// Consecutive precondition failures that demote a macro (doc ch. 4).
pub const DEMOTE_THRESHOLD: u32 = 3;
/// Successful runs required before a macro may be compiled.
pub const MACRO_MIN_SUCCESSES: u32 = 5;
/// Stable macro runs required before a trigger may be armed.
pub const TRIGGER_MIN_STABLE_RUNS: u32 = 20;

/// Structural signature of a plan for one intent kind: the step tools in
/// order plus the *shape* of their arguments (leaf paths), ignoring the
/// literal values that will become slots.
pub fn structural_signature(plan: &Plan) -> String {
    let mut parts = vec![format!("intent:{}", plan.intent_key)];
    for step in &plan.steps {
        let mut paths = step.args.leaf_paths();
        paths.sort();
        parts.push(format!("{}({})", step.tool, paths.join(",")));
    }
    parts.join("|")
}

pub fn fingerprint(plan: &Plan) -> String {
    sha256_hex(structural_signature(plan).as_bytes())
}

/// Key under which a slot value is supplied at macro-run time.
pub fn slot_key(step_index: usize, path: &str) -> String {
    format!("{step_index}:{path}")
}

/// How one intent occurrence was served — the single classification
/// behind the determinization rate (doc ch. 22: "被确定性路径（L1/L2/L3）
/// 服务的意图占比"). The P0 acceptance suite (ch. 23) and the metrics
/// dashboard both consume this type; nothing else classifies a serving.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ServedBy {
    /// Full probabilistic path: the model planned it.
    Model,
    /// L1 plan cache hit: same structure, parameters re-validated downstream.
    Cache,
    /// L2 macro: zero model calls by construction.
    Macro,
    /// L3 trigger: a macro fired automatically when its conditions held.
    Trigger,
}

impl ServedBy {
    pub fn is_deterministic(self) -> bool {
        !matches!(self, ServedBy::Model)
    }
}

/// Served-intent counts per rung, as recorded by [`DetEngine`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ServedCounts {
    pub l0_model: u64,
    pub l1_cache: u64,
    pub l2_macro: u64,
    pub l3_trigger: u64,
}

impl ServedCounts {
    pub fn total(self) -> u64 {
        self.l0_model + self.l1_cache + self.l2_macro + self.l3_trigger
    }

    pub fn deterministic(self) -> u64 {
        self.l1_cache + self.l2_macro + self.l3_trigger
    }

    /// The ch. 22/23 determinization rate: deterministic-served / all
    /// served. `None` while nothing has been served — no data, never a
    /// flattering 0% or 100%.
    pub fn determination_rate(self) -> Option<f64> {
        let total = self.total();
        if total == 0 {
            None
        } else {
            Some(self.deterministic() as f64 / total as f64)
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Macro {
    pub id: String,
    pub intent_key: String,
    pub fingerprint: String,
    /// Plan template; slot positions hold `Value::Null` placeholders.
    pub template: Plan,
    /// (step index, dot path) pairs to fill at run time.
    pub slots: Vec<(usize, String)>,
    pub schema_versions: HashMap<String, u32>,
    pub stable_runs: u32,
    pub consecutive_pre_failures: u32,
    /// Wall-clock ms at compile time, when the engine was built with a
    /// clock. `None` means the age is unknown — the macro ecology (ch. 21)
    /// never prunes a macro whose age it cannot prove.
    pub created_ms: Option<u64>,
    /// Wall-clock ms of the last *successful* macro run. A failed attempt
    /// (precondition failure, missing slot) does not refresh this: a stale
    /// macro must not be kept alive by repeatedly failing.
    pub last_used_ms: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MacroError {
    NotFound { intent_key: String },
    PreconditionFailed { name: String },
    Demoted { intent_key: String },
    SchemaChanged { tool: String },
    MissingSlot { key: String },
}

impl std::fmt::Display for MacroError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MacroError::NotFound { intent_key } => write!(f, "no macro for intent {intent_key}"),
            MacroError::PreconditionFailed { name } => {
                write!(f, "macro precondition failed: {name} (fall back to the probabilistic path)")
            }
            MacroError::Demoted { intent_key } => {
                write!(f, "macro for {intent_key} demoted to the recompile queue")
            }
            MacroError::SchemaChanged { tool } => {
                write!(f, "tool {tool} schema version changed; macro invalidated")
            }
            MacroError::MissingSlot { key } => write!(f, "no value supplied for macro slot {key}"),
        }
    }
}
impl std::error::Error for MacroError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompileError {
    NotEnoughSuccesses { have: u32, need: u32 },
    NotConfirmed,
    AlreadyCompiled,
    UnknownIntent,
}

#[derive(Debug, Clone, Copy)]
pub struct TriggerBudget {
    pub max_runs: u32,
    pub max_side_effects: u32,
}

#[derive(Debug, Clone)]
pub struct Trigger {
    pub intent_key: String,
    /// Firing conditions, evaluated against current world state before
    /// every automatic run (doc ch. 4: "条件满足即自动执行"). Empty
    /// means unconditional — the trigger is due whenever it is armed.
    pub conditions: Vec<Condition>,
    pub budget: TriggerBudget,
    pub used_runs: u32,
    pub used_side_effects: u32,
    pub enabled: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TriggerError {
    NotAuthorized,
    NoMacro,
    InsufficientStability { stable_runs: u32, need: u32 },
    BudgetExhausted,
    NotArmed,
    /// A firing condition does not hold right now. The trigger is not
    /// due — this charges no budget and never runs the macro.
    ConditionNotMet { name: String },
    /// The macro itself failed to run (precondition, schema drift,
    /// missing slot). The trigger charges no budget for a run that
    /// never happened; if the macro demoted or invalidated, the
    /// trigger is removed with it.
    MacroFailed(MacroError),
}

struct Record {
    successes: u32,
    baseline: Plan,
    slot_paths: HashSet<(usize, String)>,
    schema_versions: HashMap<String, u32>,
}

pub struct DetEngine {
    cache: HashMap<String, (String, Plan)>, // intent_key -> (fingerprint, plan)
    records: HashMap<String, Record>,
    macros: HashMap<String, Macro>,
    triggers: HashMap<String, Trigger>,
    recompile_queue: Vec<String>,
    /// Served-intent counts per rung — the single source for the
    /// determinization rate. L1/L2/L3 are recorded by the engine itself
    /// when it serves an intent; L0 is recorded by the caller through
    /// [`DetEngine::record_served`] when the model planned it.
    served: ServedCounts,
    /// Optional time source. `None` (the `new()` case) keeps the engine
    /// fully deterministic with no wall-clock dependence; macro age is
    /// then unknown and ecology pruning is disabled for those macros.
    clock: Option<Rc<dyn Clock>>,
}

impl DetEngine {
    pub fn new() -> Self {
        DetEngine {
            cache: HashMap::new(),
            records: HashMap::new(),
            macros: HashMap::new(),
            triggers: HashMap::new(),
            recompile_queue: Vec::new(),
            served: ServedCounts::default(),
            clock: None,
        }
    }

    /// Build an engine that timestamps macro compilation and successful
    /// macro runs, so the macro ecology can judge idleness.
    pub fn with_clock(clock: Rc<dyn Clock>) -> Self {
        let mut engine = Self::new();
        engine.clock = Some(clock);
        engine
    }

    fn now_ms(&self) -> Option<u64> {
        self.clock.as_ref().map(|c| c.now_ms())
    }

    // -- Recording --------------------------------------------------------

    /// Record one successful probabilistic-path execution. The first run
    /// becomes the baseline; later runs reveal which argument leaves vary
    /// (those become macro slots).
    pub fn record_success(&mut self, plan: &Plan, schema_versions: &HashMap<String, u32>) {
        let fp = fingerprint(plan);
        self.cache.insert(plan.intent_key.clone(), (fp.clone(), plan.clone()));
        // Structure drift under the same intent key: the old baseline is
        // meaningless — restart learning from this run.
        if let Some(rec) = self.records.get(&plan.intent_key) {
            if fingerprint(&rec.baseline) != fp {
                self.records.remove(&plan.intent_key);
            }
        }
        match self.records.get_mut(&plan.intent_key) {
            None => {
                self.records.insert(
                    plan.intent_key.clone(),
                    Record {
                        successes: 1,
                        baseline: plan.clone(),
                        slot_paths: HashSet::new(),
                        schema_versions: schema_versions.clone(),
                    },
                );
            }
            Some(rec) => {
                rec.successes += 1;
                // Compare every leaf of every step against the baseline.
                for (i, step) in plan.steps.iter().enumerate() {
                    let base_step = &rec.baseline.steps[i];
                    let mut paths: HashSet<String> =
                        step.args.leaf_paths().into_iter().collect();
                    paths.extend(base_step.args.leaf_paths());
                    for path in paths {
                        let split: Vec<&str> = path.split('.').collect();
                        let now = step.args.get_path(&split);
                        let base = base_step.args.get_path(&split);
                        if now != base {
                            rec.slot_paths.insert((i, path));
                        }
                    }
                }
                rec.schema_versions = schema_versions.clone();
            }
        }
    }

    pub fn successes(&self, intent_key: &str) -> u32 {
        self.records.get(intent_key).map(|r| r.successes).unwrap_or(0)
    }

    /// Record that one intent occurrence was served by `by`. The engine
    /// calls this itself for the rungs it serves (L1 cache hits in
    /// [`DetEngine::lookup_cache`], L2 runs in [`DetEngine::run_macro`],
    /// L3 firings in [`DetEngine::fire_trigger`]); the L0 probabilistic
    /// path is the caller's responsibility — the model planned it, so
    /// only the caller knows it happened.
    pub fn record_served(&mut self, by: ServedBy) {
        match by {
            ServedBy::Model => self.served.l0_model += 1,
            ServedBy::Cache => self.served.l1_cache += 1,
            ServedBy::Macro => self.served.l2_macro += 1,
            ServedBy::Trigger => self.served.l3_trigger += 1,
        }
    }

    // -- L1: plan cache ----------------------------------------------------

    /// L1 lookup: a cached plan for this intent kind, only if its
    /// fingerprint still matches the recorded structure. A hit counts
    /// as one intent served by the L1 rung.
    pub fn lookup_cache(&mut self, intent_key: &str) -> Option<Plan> {
        let (fp, plan) = self.cache.get(intent_key)?;
        if fingerprint(plan) != *fp {
            return None;
        }
        let plan = plan.clone();
        self.record_served(ServedBy::Cache);
        Some(plan)
    }

    // -- L2: macros --------------------------------------------------------

    pub fn eligible_for_macro(&self, intent_key: &str) -> bool {
        self.successes(intent_key) >= MACRO_MIN_SUCCESSES && !self.macros.contains_key(intent_key)
    }

    /// Compile a macro. Requires ≥ 5 recorded successes *and* the user's
    /// explicit confirmation — determinization is never silent (doc ch. 4).
    pub fn compile_macro(
        &mut self,
        intent_key: &str,
        user_confirmed: bool,
    ) -> Result<Macro, CompileError> {
        if self.macros.contains_key(intent_key) {
            return Err(CompileError::AlreadyCompiled);
        }
        let rec = self.records.get(intent_key).ok_or(CompileError::UnknownIntent)?;
        if rec.successes < MACRO_MIN_SUCCESSES {
            return Err(CompileError::NotEnoughSuccesses {
                have: rec.successes,
                need: MACRO_MIN_SUCCESSES,
            });
        }
        if !user_confirmed {
            return Err(CompileError::NotConfirmed);
        }
        let mut template = rec.baseline.clone();
        let mut slots: Vec<(usize, String)> = rec.slot_paths.iter().cloned().collect();
        slots.sort();
        for (step_idx, path) in &slots {
            let split: Vec<&str> = path.split('.').collect();
            template.steps[*step_idx].args.set_path(&split, Value::Null);
        }
        let mac = Macro {
            id: format!("macro:{}", intent_key),
            intent_key: intent_key.to_string(),
            fingerprint: fingerprint(&rec.baseline),
            template,
            slots,
            schema_versions: rec.schema_versions.clone(),
            stable_runs: 0,
            consecutive_pre_failures: 0,
            created_ms: self.now_ms(),
            last_used_ms: self.now_ms(),
        };
        self.macros.insert(intent_key.to_string(), mac.clone());
        Ok(mac)
    }

    pub fn macro_for(&self, intent_key: &str) -> Option<&Macro> {
        self.macros.get(intent_key)
    }

    fn schema_still_valid(
        mac: &Macro,
        current_versions: &HashMap<String, u32>,
    ) -> Result<(), MacroError> {
        for (tool, bound) in &mac.schema_versions {
            if current_versions.get(tool) != Some(bound) {
                return Err(MacroError::SchemaChanged { tool: tool.clone() });
            }
        }
        Ok(())
    }

    /// Execute a macro: verify schema binding, check preconditions, fill
    /// slots, return a concrete plan. There is deliberately no model
    /// parameter — a macro run is zero-model-call *by construction*.
    /// A successful run counts as one intent served by the L2 rung.
    ///
    /// On precondition failure the caller must fall back to the
    /// probabilistic path. On repeated failure the macro demotes itself
    /// into the recompile queue; on schema drift it invalidates at once.
    pub fn run_macro(
        &mut self,
        intent_key: &str,
        slot_values: &HashMap<String, Value>,
        facts: &dyn Facts,
        current_versions: &HashMap<String, u32>,
    ) -> Result<Plan, MacroError> {
        let plan = self.run_macro_inner(intent_key, slot_values, facts, current_versions)?;
        self.record_served(ServedBy::Macro);
        Ok(plan)
    }

    /// The macro run itself, without the serving classification — the
    /// L3 trigger path uses this directly so a trigger firing is
    /// counted once, as L3, not once as L2 and once as L3.
    fn run_macro_inner(
        &mut self,
        intent_key: &str,
        slot_values: &HashMap<String, Value>,
        facts: &dyn Facts,
        current_versions: &HashMap<String, u32>,
    ) -> Result<Plan, MacroError> {
        let mac = self
            .macros
            .get(intent_key)
            .cloned()
            .ok_or_else(|| MacroError::NotFound { intent_key: intent_key.to_string() })?;

        if let Err(e) = Self::schema_still_valid(&mac, current_versions) {
            self.macros.remove(intent_key);
            self.recompile_queue.push(intent_key.to_string());
            return Err(e);
        }

        for cond in &mac.template.pre {
            if !cond.eval(facts) {
                let entry = self.macros.get_mut(intent_key).expect("present");
                entry.consecutive_pre_failures += 1;
                let failures = entry.consecutive_pre_failures;
                if failures >= DEMOTE_THRESHOLD {
                    self.macros.remove(intent_key);
                    self.recompile_queue.push(intent_key.to_string());
                    return Err(MacroError::Demoted { intent_key: intent_key.to_string() });
                }
                return Err(MacroError::PreconditionFailed { name: cond.name.clone() });
            }
        }

        let mut plan = mac.template.clone();
        for (step_idx, path) in &mac.slots {
            let key = slot_key(*step_idx, path);
            let value = slot_values
                .get(&key)
                .ok_or_else(|| MacroError::MissingSlot { key: key.clone() })?;
            let split: Vec<&str> = path.split('.').collect();
            plan.steps[*step_idx].args.set_path(&split, value.clone());
        }

        let now = self.now_ms();
        let entry = self.macros.get_mut(intent_key).expect("present");
        entry.consecutive_pre_failures = 0;
        entry.stable_runs += 1;
        if let Some(now) = now {
            entry.last_used_ms = Some(now);
        }
        Ok(plan)
    }

    /// All live macros, sorted by intent key (deterministic order for
    /// ecology reports). The macro ecology reads through this snapshot
    /// and prunes through [`DetEngine::remove_macro`].
    pub fn macros_snapshot(&self) -> Vec<Macro> {
        let mut macros: Vec<Macro> = self.macros.values().cloned().collect();
        macros.sort_by(|a, b| a.intent_key.cmp(&b.intent_key));
        macros
    }

    /// Remove a macro and any trigger armed for it, so a pruned macro
    /// leaves no dangling trigger behind. Returns the removed macro.
    /// (Pre-existing demotion/invalidation paths deliberately keep their
    /// own behaviour and do not go through this.)
    pub fn remove_macro(&mut self, intent_key: &str) -> Option<Macro> {
        self.triggers.remove(intent_key);
        self.macros.remove(intent_key)
    }

    pub fn recompile_queue(&self) -> &[String] {
        &self.recompile_queue
    }

    // -- L3: triggers ------------------------------------------------------

    /// Arm an automatic trigger for a macro. Requires the macro to exist,
    /// ≥ 20 stable macro runs, and the user's *explicit* authorization.
    /// `conditions` are evaluated against world state before every
    /// firing — a trigger with an unmet condition is simply not due.
    /// The budget is a hard cap, not a guideline (doc ch. 4).
    pub fn arm_trigger(
        &mut self,
        intent_key: &str,
        explicit_authorization: bool,
        conditions: Vec<Condition>,
        budget: TriggerBudget,
    ) -> Result<(), TriggerError> {
        if !explicit_authorization {
            return Err(TriggerError::NotAuthorized);
        }
        let mac = self.macros.get(intent_key).ok_or(TriggerError::NoMacro)?;
        if mac.stable_runs < TRIGGER_MIN_STABLE_RUNS {
            return Err(TriggerError::InsufficientStability {
                stable_runs: mac.stable_runs,
                need: TRIGGER_MIN_STABLE_RUNS,
            });
        }
        self.triggers.insert(
            intent_key.to_string(),
            Trigger {
                intent_key: intent_key.to_string(),
                conditions,
                budget,
                used_runs: 0,
                used_side_effects: 0,
                enabled: true,
            },
        );
        Ok(())
    }

    /// Is this trigger due right now — armed, enabled, budget headroom,
    /// and every firing condition holds? Pure query: charges nothing
    /// and runs nothing, so a scheduler can poll it freely.
    pub fn trigger_ready(&self, intent_key: &str, facts: &dyn Facts) -> bool {
        let Some(trig) = self.triggers.get(intent_key) else { return false };
        trig.enabled
            && trig.used_runs < trig.budget.max_runs
            && trig.used_side_effects < trig.budget.max_side_effects
            && trig.conditions.iter().all(|c| c.eval(facts))
    }

    /// Fire an armed trigger: check it is due, run its macro, charge the
    /// authorization budget, and return the macro's plan for execution.
    /// A condition failure charges nothing and never runs the macro; a
    /// macro failure charges nothing either — the budget pays only for
    /// runs that actually happened. If the macro demoted or invalidated
    /// while firing, the trigger goes with it (no dangling trigger).
    pub fn fire_trigger(
        &mut self,
        intent_key: &str,
        slot_values: &HashMap<String, Value>,
        facts: &dyn Facts,
        current_versions: &HashMap<String, u32>,
        side_effects: u32,
    ) -> Result<Plan, TriggerError> {
        let trig = self.triggers.get(intent_key).ok_or(TriggerError::NotArmed)?;
        if !trig.enabled
            || trig.used_runs + 1 > trig.budget.max_runs
            || trig.used_side_effects + side_effects > trig.budget.max_side_effects
        {
            self.triggers.get_mut(intent_key).expect("checked").enabled = false;
            return Err(TriggerError::BudgetExhausted);
        }
        for cond in &trig.conditions {
            if !cond.eval(facts) {
                return Err(TriggerError::ConditionNotMet { name: cond.name.clone() });
            }
        }
        let plan = match self.run_macro_inner(intent_key, slot_values, facts, current_versions) {
            Ok(plan) => plan,
            Err(e) => {
                // The macro is gone (demoted / schema-invalidated): the
                // trigger must not outlive it.
                if !self.macros.contains_key(intent_key) {
                    self.triggers.remove(intent_key);
                }
                return Err(TriggerError::MacroFailed(e));
            }
        };
        let trig = self.triggers.get_mut(intent_key).expect("trigger still armed");
        trig.used_runs += 1;
        trig.used_side_effects += side_effects;
        self.record_served(ServedBy::Trigger);
        Ok(plan)
    }

    pub fn trigger_for(&self, intent_key: &str) -> Option<&Trigger> {
        self.triggers.get(intent_key)
    }

    // -- Metrics -----------------------------------------------------------

    /// Served counts per rung — the raw counters behind the
    /// determinization rate (doc ch. 22/23). Same numbers the P0
    /// acceptance suite and the metrics dashboard report.
    pub fn served_counts(&self) -> ServedCounts {
        self.served
    }

    /// The determinization rate (ch. 22 north-star metric): share of
    /// served intents that came from a deterministic rung (L1/L2/L3).
    /// `None` until the first intent is served.
    pub fn determination_rate(&self) -> Option<f64> {
        self.served.determination_rate()
    }

    /// (total served intents, L1 hits, L2 hits) — the served-based
    /// counters behind the determinization rate. L3 trigger firings
    /// count in the total (and in [`DetEngine::determination_rate`])
    /// but are not L1/L2 hits; see [`DetEngine::served_counts`] for
    /// the per-rung breakdown.
    pub fn stats(&self) -> (u64, u64, u64) {
        (self.served.total(), self.served.l1_cache, self.served.l2_macro)
    }
}

impl Default for DetEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::taint::TrustLabel;
    use crate::typesys::{Condition, PlanStep, Predicate};

    const INTENT: &str = "trip.arrange";

    fn trip_plan(date: &str) -> Plan {
        let mut p = Plan::new(format!("plan-{date}"), INTENT, TrustLabel::User);
        p.steps = vec![
            PlanStep::new(
                "flight.search",
                "flight",
                Value::map(vec![("to", Value::text("Shanghai")), ("date", Value::text(date))]),
            ),
            PlanStep::new(
                "cal.write",
                "calendar",
                Value::map(vec![("title", Value::text("Shanghai trip")), ("date", Value::text(date))]),
            ),
        ];
        p.pre = vec![Condition::new("slot free", Predicate::SlotFree("cal:slot".into()))];
        p
    }

    fn versions() -> HashMap<String, u32> {
        HashMap::from([("flight.search".to_string(), 1), ("cal.write".to_string(), 1)])
    }

    fn engine_with_macro() -> DetEngine {
        let mut det = DetEngine::new();
        for i in 1..=5 {
            det.record_success(&trip_plan(&format!("2026-11-0{i}")), &versions());
        }
        det.compile_macro(INTENT, true).unwrap();
        det
    }

    fn slot_values(date: &str) -> HashMap<String, Value> {
        HashMap::from([
            (slot_key(0, "date"), Value::text(date)),
            (slot_key(1, "date"), Value::text(date)),
        ])
    }

    fn free_facts() -> crate::typesys::FactBase {
        crate::typesys::FactBase::new().with_slot_free("cal:slot")
    }

    #[test]
    fn l1_cache_returns_the_recorded_plan() {
        let mut det = DetEngine::new();
        det.record_success(&trip_plan("2026-11-01"), &versions());
        let cached = det.lookup_cache(INTENT).expect("L1 hit");
        assert_eq!(cached.steps.len(), 2);
        assert!(det.lookup_cache("unknown.intent").is_none());
        let (_, l1, _) = det.stats();
        assert_eq!(l1, 1);
    }

    #[test]
    fn macro_requires_five_successes_and_user_confirmation() {
        let mut det = DetEngine::new();
        for i in 1..=4 {
            det.record_success(&trip_plan(&format!("2026-11-0{i}")), &versions());
        }
        assert_eq!(
            det.compile_macro(INTENT, true),
            Err(CompileError::NotEnoughSuccesses { have: 4, need: 5 })
        );
        det.record_success(&trip_plan("2026-11-05"), &versions());
        assert_eq!(det.compile_macro(INTENT, false), Err(CompileError::NotConfirmed));
        assert!(det.compile_macro(INTENT, true).is_ok());
        assert_eq!(det.compile_macro(INTENT, true), Err(CompileError::AlreadyCompiled));
    }

    #[test]
    fn varying_arguments_become_slots_constants_stay_frozen() {
        let det = engine_with_macro();
        let mac = det.macro_for(INTENT).unwrap();
        // "date" varied across the 5 runs -> slot in both steps.
        assert!(mac.slots.contains(&(0, "date".to_string())));
        assert!(mac.slots.contains(&(1, "date".to_string())));
        // "to"/"title" never varied -> frozen into the template.
        assert!(!mac.slots.iter().any(|(_, p)| p == "to"));
        assert_eq!(
            mac.template.steps[0].args.get_path(&["date"]),
            Some(&Value::Null),
            "slot positions are placeholders in the template"
        );
        assert_eq!(
            mac.template.steps[0].args.get_path(&["to"]),
            Some(&Value::text("Shanghai"))
        );
    }

    #[test]
    fn macro_run_fills_slots_and_needs_no_model() {
        let mut det = engine_with_macro();
        let plan = det
            .run_macro(INTENT, &slot_values("2026-12-24"), &free_facts(), &versions())
            .unwrap();
        assert_eq!(plan.steps[0].args.get_path(&["date"]), Some(&Value::text("2026-12-24")));
        assert_eq!(plan.steps[1].args.get_path(&["title"]), Some(&Value::text("Shanghai trip")));
        let mac = det.macro_for(INTENT).unwrap();
        assert_eq!(mac.stable_runs, 1);
        let (_, _, l2) = det.stats();
        assert_eq!(l2, 1);
    }

    #[test]
    fn macro_run_with_missing_slot_value_fails() {
        let mut det = engine_with_macro();
        let err = det.run_macro(INTENT, &HashMap::new(), &free_facts(), &versions()).unwrap_err();
        assert!(matches!(err, MacroError::MissingSlot { .. }));
    }

    #[test]
    fn precondition_failure_falls_back_and_repeated_failure_demotes() {
        let mut det = engine_with_macro();
        let busy = crate::typesys::FactBase::new(); // slot NOT free
        for i in 1..=2 {
            let err = det.run_macro(INTENT, &slot_values("2026-12-24"), &busy, &versions()).unwrap_err();
            assert!(matches!(err, MacroError::PreconditionFailed { .. }), "failure {i}");
            assert!(det.macro_for(INTENT).is_some(), "still alive before threshold");
        }
        let err = det.run_macro(INTENT, &slot_values("2026-12-24"), &busy, &versions()).unwrap_err();
        assert_eq!(err, MacroError::Demoted { intent_key: INTENT.to_string() });
        assert!(det.macro_for(INTENT).is_none());
        assert_eq!(det.recompile_queue(), &[INTENT.to_string()]);
    }

    #[test]
    fn schema_version_change_invalidates_the_macro() {
        let mut det = engine_with_macro();
        let mut v2 = versions();
        v2.insert("cal.write".to_string(), 2); // the tool's interface changed
        let err = det.run_macro(INTENT, &slot_values("2026-12-24"), &free_facts(), &v2).unwrap_err();
        assert_eq!(err, MacroError::SchemaChanged { tool: "cal.write".to_string() });
        assert!(det.macro_for(INTENT).is_none());
    }

    #[test]
    fn trigger_needs_authorization_stability_and_budget() {
        let mut det = engine_with_macro();
        let budget = TriggerBudget { max_runs: 3, max_side_effects: 10 };
        // Not stable enough yet (0 macro runs).
        assert_eq!(
            det.arm_trigger(INTENT, true, vec![], budget),
            Err(TriggerError::InsufficientStability { stable_runs: 0, need: 20 })
        );
        // Run the macro 20 times to stabilize it.
        for _ in 0..20 {
            det.run_macro(INTENT, &slot_values("2026-12-24"), &free_facts(), &versions()).unwrap();
        }
        // Still needs explicit authorization.
        assert_eq!(det.arm_trigger(INTENT, false, vec![], budget), Err(TriggerError::NotAuthorized));
        det.arm_trigger(INTENT, true, vec![], budget).unwrap();
        // Fire within budget, then exhaust it. Each firing returns the
        // macro's plan with slots filled.
        for _ in 0..3 {
            let plan = det
                .fire_trigger(INTENT, &slot_values("2026-12-24"), &free_facts(), &versions(), 3)
                .unwrap();
            assert_eq!(plan.steps[0].args.get_path(&["date"]), Some(&Value::text("2026-12-24")));
        }
        assert_eq!(
            det.fire_trigger(INTENT, &slot_values("2026-12-24"), &free_facts(), &versions(), 1),
            Err(TriggerError::BudgetExhausted)
        );
        assert!(!det.trigger_for(INTENT).unwrap().enabled);
        // 20 manual macro runs + 3 trigger firings, all deterministic.
        assert_eq!(det.served_counts().l2_macro, 20);
        assert_eq!(det.served_counts().l3_trigger, 3);
        assert_eq!(det.determination_rate(), Some(1.0));
    }

    fn stable_engine_with_trigger() -> DetEngine {
        let mut det = engine_with_macro();
        for _ in 0..20 {
            det.run_macro(INTENT, &slot_values("2026-12-24"), &free_facts(), &versions()).unwrap();
        }
        det
    }

    #[test]
    fn trigger_fires_only_when_its_condition_holds() {
        let mut det = stable_engine_with_trigger();
        let condition = Condition::new("inbox has unread", Predicate::Exists("inbox:unread".to_string()));
        let budget = TriggerBudget { max_runs: 5, max_side_effects: 50 };
        det.arm_trigger(INTENT, true, vec![condition], budget).unwrap();

        // No unread mail: not due, nothing charged, macro untouched.
        assert!(!det.trigger_ready(INTENT, &free_facts()));
        assert_eq!(
            det.fire_trigger(INTENT, &slot_values("2026-12-24"), &free_facts(), &versions(), 1),
            Err(TriggerError::ConditionNotMet { name: "inbox has unread".to_string() })
        );
        let trig = det.trigger_for(INTENT).unwrap();
        assert_eq!(trig.used_runs, 0);
        assert!(trig.enabled);
        assert_eq!(det.macro_for(INTENT).unwrap().stable_runs, 20);

        // Unread mail arrives: due, fires, returns the plan, charges one run.
        let facts = free_facts().with_present("inbox:unread");
        assert!(det.trigger_ready(INTENT, &facts));
        let plan = det
            .fire_trigger(INTENT, &slot_values("2026-12-24"), &facts, &versions(), 1)
            .unwrap();
        assert_eq!(plan.steps[1].args.get_path(&["date"]), Some(&Value::text("2026-12-24")));
        assert_eq!(det.trigger_for(INTENT).unwrap().used_runs, 1);
    }

    #[test]
    fn trigger_firing_counts_once_as_l3_not_l2() {
        let mut det = stable_engine_with_trigger();
        let budget = TriggerBudget { max_runs: 5, max_side_effects: 50 };
        det.arm_trigger(INTENT, true, vec![], budget).unwrap();
        det.fire_trigger(INTENT, &slot_values("2026-12-24"), &free_facts(), &versions(), 1)
            .unwrap();
        let counts = det.served_counts();
        assert_eq!(counts.l2_macro, 20, "the 20 stabilizing runs stay L2");
        assert_eq!(counts.l3_trigger, 1, "the firing is L3, not a second L2");
        assert_eq!(counts.total(), 21);
        assert_eq!(det.determination_rate(), Some(1.0));
        assert_eq!(det.stats(), (21, 0, 20));
    }

    #[test]
    fn trigger_dies_with_its_macro_on_schema_drift() {
        let mut det = stable_engine_with_trigger();
        let budget = TriggerBudget { max_runs: 5, max_side_effects: 50 };
        det.arm_trigger(INTENT, true, vec![], budget).unwrap();
        let mut v2 = versions();
        v2.insert("cal.write".to_string(), 2);
        let err = det
            .fire_trigger(INTENT, &slot_values("2026-12-24"), &free_facts(), &v2, 1)
            .unwrap_err();
        assert_eq!(err, TriggerError::MacroFailed(MacroError::SchemaChanged { tool: "cal.write".to_string() }));
        assert!(det.trigger_for(INTENT).is_none(), "no dangling trigger");
        assert!(det.macro_for(INTENT).is_none());
        assert_eq!(det.recompile_queue(), &[INTENT.to_string()]);
    }

    #[test]
    fn determination_rate_counts_served_intents_not_recorded_successes() {
        // No data yet: no rate at all, not 0%.
        let det = DetEngine::new();
        assert_eq!(det.determination_rate(), None);
        assert_eq!(det.stats(), (0, 0, 0));

        // Five recorded successes are *learning*, not serving: they do
        // not move the rate. (The old stats() counted them as the total,
        // which could push the rate past 100% once macros ran.)
        let mut det = engine_with_macro();
        assert_eq!(det.determination_rate(), None);
        det.lookup_cache(INTENT).unwrap(); // L1
        det.run_macro(INTENT, &slot_values("2026-12-24"), &free_facts(), &versions()).unwrap(); // L2
        assert_eq!(det.determination_rate(), Some(1.0));
        det.record_served(ServedBy::Model);
        det.record_served(ServedBy::Model);
        assert_eq!(det.determination_rate(), Some(0.5));
        assert_eq!(det.stats(), (4, 1, 1));
        assert_eq!(
            det.served_counts(),
            ServedCounts { l0_model: 2, l1_cache: 1, l2_macro: 1, l3_trigger: 0 }
        );
    }

    #[test]
    fn fingerprint_is_stable_across_slot_values() {
        // Same structure, different dates -> same fingerprint (that is what
        // makes repetitions recognizable).
        assert_eq!(fingerprint(&trip_plan("2026-11-01")), fingerprint(&trip_plan("2027-01-15")));
        let mut other = trip_plan("2026-11-01");
        other.steps.pop();
        assert_ne!(fingerprint(&trip_plan("2026-11-01")), fingerprint(&other));
    }
}
