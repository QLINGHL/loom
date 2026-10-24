//! Speculative execution and commit validation (doc ch. 12).
//!
//! Most agent latency is waiting on tools, so the runtime may predict the
//! next action and execute it early in a sandbox. Two hard rules:
//!
//! * **Prediction is not evidence.** A speculative observation may feed
//!   downstream reasoning, but it becomes fact only after all four
//!   validation predicates pass at commit time:
//!   `V_act` (the real action matches the predicted one) ∧
//!   `V_dep` (everything it read is still at the same version) ∧
//!   `V_record` (the observation is complete) ∧
//!   `V_effect` (no side effects beyond the declared set).
//! * **Speculation is a schedulable, budgeted policy.** It burns real
//!   compute and battery, so system-level gates can switch it off, and a
//!   class of steps — the *non-speculatable barriers* — must never be
//!   speculated at all. The barrier check lives at the kernel's `act()`
//!   entry, not in userland, so a compromised runtime cannot disable it.

use crate::typesys::Compensability;
use crate::Value;

/// System conditions under which speculation may be considered at all.
#[derive(Debug, Clone, Copy)]
pub struct SpecContext {
    pub battery_pct: u8,
    pub thermal_high: bool,
    /// True when a user is actively waiting (their time is what speculation
    /// buys); background batch work should not speculate.
    pub foreground: bool,
    /// Fraction `f` of wall time spent waiting on tools. The theoretical
    /// speedup cap is 1/(1−f); below f = 0.4 it cannot pay for itself.
    pub tool_time_fraction: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecGate {
    BatteryLow,
    Thermal,
    Background,
    LowToolFraction,
}

/// The budget gate (doc ch. 12): all conditions must hold to speculate.
pub fn speculation_allowed(ctx: &SpecContext) -> Result<(), SpecGate> {
    if ctx.battery_pct < 20 {
        return Err(SpecGate::BatteryLow);
    }
    if ctx.thermal_high {
        return Err(SpecGate::Thermal);
    }
    if !ctx.foreground {
        return Err(SpecGate::Background);
    }
    if ctx.tool_time_fraction < 0.4 {
        return Err(SpecGate::LowToolFraction);
    }
    Ok(())
}

/// Why a step may never be speculatively executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Barrier {
    /// Irreversible side effects: speculating them would bypass
    /// confirmation and the cooling-off window.
    IrreversibleEffect,
    /// The read set cannot be fully captured (remote APIs, shared
    /// processes): the result could not be validated, hence not trusted.
    UntrackedDependencies,
    /// The action's content depends on data it reads (draft editing):
    /// there is nothing well-defined to predict.
    ContentDependent,
}

/// Static profile of a planned step, from its tool manifest and plan shape.
#[derive(Debug, Clone, Copy)]
pub struct StepProfile {
    pub compensability: Compensability,
    pub deps_trackable: bool,
    pub content_dependent: bool,
}

/// The non-speculatable barrier check. The kernel runs this at the `act()`
/// entry for any step submitted as a speculative candidate; a hit forces
/// the serial path.
pub fn barrier_for(step: &StepProfile) -> Option<Barrier> {
    if step.compensability == Compensability::Irreversible {
        return Some(Barrier::IrreversibleEffect);
    }
    if !step.deps_trackable {
        return Some(Barrier::UntrackedDependencies);
    }
    if step.content_dependent {
        return Some(Barrier::ContentDependent);
    }
    None
}

/// A speculative execution's output, held in the sandbox until validated.
#[derive(Debug, Clone)]
pub struct SpeculativeResult {
    pub predicted_action: String,
    /// (object, version) pairs the speculation read.
    pub read_versions: Vec<(String, u64)>,
    pub observation: Value,
    pub observation_complete: bool,
    /// Side effects the speculation actually caused in its sandbox.
    pub extra_effects: Vec<String>,
}

/// Per-predicate verdict. A speculative result may commit iff all four
/// hold; the report says exactly which one failed when it cannot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PredicateReport {
    pub v_act: bool,
    pub v_dep: bool,
    pub v_record: bool,
    pub v_effect: bool,
}

impl PredicateReport {
    pub fn all(&self) -> bool {
        self.v_act && self.v_dep && self.v_record && self.v_effect
    }

    pub fn failed(&self) -> Vec<&'static str> {
        let mut out = Vec::new();
        if !self.v_act {
            out.push("V_act");
        }
        if !self.v_dep {
            out.push("V_dep");
        }
        if !self.v_record {
            out.push("V_record");
        }
        if !self.v_effect {
            out.push("V_effect");
        }
        out
    }
}

/// Validate a speculative result at commit time against the real action,
/// the world's current versions, and the step's declared effects.
pub fn validate(
    spec: &SpeculativeResult,
    actual_action: &str,
    current_versions: &[(String, u64)],
    declared_effects: &[String],
) -> PredicateReport {
    let v_act = spec.predicted_action == actual_action;
    let v_dep = spec.read_versions.iter().all(|(obj, ver)| {
        current_versions
            .iter()
            .find(|(o, _)| o == obj)
            .map(|(_, cur)| cur == ver)
            .unwrap_or(false)
    });
    let v_record = spec.observation_complete;
    let v_effect = spec.extra_effects.iter().all(|e| declared_effects.contains(e));
    PredicateReport { v_act, v_dep, v_record, v_effect }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn good_ctx() -> SpecContext {
        SpecContext { battery_pct: 80, thermal_high: false, foreground: true, tool_time_fraction: 0.6 }
    }

    fn profile() -> StepProfile {
        StepProfile { compensability: Compensability::Full, deps_trackable: true, content_dependent: false }
    }

    fn spec_result() -> SpeculativeResult {
        SpeculativeResult {
            predicted_action: "cal.read".into(),
            read_versions: vec![("cal:event".into(), 3)],
            observation: Value::text("free"),
            observation_complete: true,
            extra_effects: vec![],
        }
    }

    #[test]
    fn budget_gates_block_speculation() {
        assert!(speculation_allowed(&good_ctx()).is_ok());
        assert_eq!(
            speculation_allowed(&SpecContext { battery_pct: 19, ..good_ctx() }),
            Err(SpecGate::BatteryLow)
        );
        assert_eq!(
            speculation_allowed(&SpecContext { thermal_high: true, ..good_ctx() }),
            Err(SpecGate::Thermal)
        );
        assert_eq!(
            speculation_allowed(&SpecContext { foreground: false, ..good_ctx() }),
            Err(SpecGate::Background)
        );
        assert_eq!(
            speculation_allowed(&SpecContext { tool_time_fraction: 0.39, ..good_ctx() }),
            Err(SpecGate::LowToolFraction)
        );
        assert!(speculation_allowed(&SpecContext { tool_time_fraction: 0.40, ..good_ctx() }).is_ok());
    }

    #[test]
    fn irreversible_steps_are_barriers() {
        let p = StepProfile { compensability: Compensability::Irreversible, ..profile() };
        assert_eq!(barrier_for(&p), Some(Barrier::IrreversibleEffect));
    }

    #[test]
    fn untracked_deps_and_content_dependence_are_barriers() {
        let p = StepProfile { deps_trackable: false, ..profile() };
        assert_eq!(barrier_for(&p), Some(Barrier::UntrackedDependencies));
        let p2 = StepProfile { content_dependent: true, ..profile() };
        assert_eq!(barrier_for(&p2), Some(Barrier::ContentDependent));
        assert_eq!(barrier_for(&profile()), None);
    }

    #[test]
    fn valid_speculation_commits() {
        let report = validate(&spec_result(), "cal.read", &[("cal:event".into(), 3)], &[]);
        assert!(report.all());
        assert!(report.failed().is_empty());
    }

    #[test]
    fn wrong_predicted_action_fails_v_act() {
        let report = validate(&spec_result(), "cal.write", &[("cal:event".into(), 3)], &[]);
        assert!(!report.v_act);
        assert_eq!(report.failed(), vec!["V_act"]);
    }

    #[test]
    fn stale_dependency_fails_v_dep() {
        // The world moved from version 3 to 4 after the speculation read it.
        let report = validate(&spec_result(), "cal.read", &[("cal:event".into(), 4)], &[]);
        assert!(!report.v_dep);
        assert!(!report.all());
    }

    #[test]
    fn incomplete_observation_fails_v_record() {
        let mut spec = spec_result();
        spec.observation_complete = false;
        let report = validate(&spec, "cal.read", &[("cal:event".into(), 3)], &[]);
        assert!(!report.v_record);
    }

    #[test]
    fn undeclared_side_effect_fails_v_effect() {
        let mut spec = spec_result();
        spec.extra_effects = vec!["write:cal".into()];
        let report = validate(&spec, "cal.read", &[("cal:event".into(), 3)], &[]);
        assert!(!report.v_effect);
        // Declaring it makes the same speculation committable.
        let report2 = validate(&spec, "cal.read", &[("cal:event".into(), 3)], &["write:cal".into()]);
        assert!(report2.all());
    }
}
