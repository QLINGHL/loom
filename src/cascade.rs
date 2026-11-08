//! Cascade routing (doc ch. 21, open problem 4): a small model handles most
//! requests and the hard ones go to the large model, because that is the
//! only realistic way to control cost. The catch the design doc names is
//! that "how hard is this?" is itself decided by a classifier, and the
//! classifier makes mistakes — asymmetrically:
//!
//! * An **easy task sent to the large model** only costs time and money.
//! * A **hard task sent to the small model** is executed wrongly.
//!
//! So the router is biased toward the safe direction: anything the
//! classifier is unsure about goes to the large model, and any plan with
//! irreversible effects goes there unconditionally. The router then learns
//! from outcomes — a costly misroute (small model failed, had to re-run on
//! large) lowers the bar for "easy" immediately, while the bar only rises
//! after several consecutive cheap misroutes (large was overkill). One
//! costly error outweighs many cheap ones, because that is what the cost
//! asymmetry says.
//!
//! The [`DifficultyClassifier`] trait is the model slot (like the
//! [`SemanticJudge`](crate::watchdog::SemanticJudge) slot in the watchdog):
//! the real classifier is a model judgement. [`HeuristicClassifier`] is the
//! cheap deterministic fallback that scores observable features of the
//! request; it is honest about being a heuristic — its confidence is just
//! the margin from the decision boundary, capped when the request carries
//! unresolved references.

/// Observable, deterministic features of one request. Nothing here needs a
/// model to compute: they come from the plan the intent compiler produced.
#[derive(Debug, Clone, Default)]
pub struct TaskFeatures {
    /// Distinct tools the plan may call.
    pub planned_tool_calls: u32,
    /// The plan contains an irreversible effect (send, transfer, delete).
    pub irreversible_effects: bool,
    /// Ambiguous slots / missing facts the plan still needs.
    pub unresolved_references: u32,
    /// Input size in tokens.
    pub input_tokens: u32,
    /// The plan needs a fact verified against the world before acting.
    pub needs_verified_fact: bool,
    /// How many times this task has already been retried.
    pub retries_so_far: u32,
}

/// A classifier's verdict: how hard the task looks, and how sure it is.
/// `difficulty` runs 0.0 (trivial) to 1.0 (definitely needs the large
/// model); `confidence` runs 0.0 to 1.0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Classification {
    pub difficulty: f64,
    pub confidence: f64,
}

/// The model slot: judges how hard a task is. Implementations may call a
/// model; [`HeuristicClassifier`] is the deterministic fallback.
pub trait DifficultyClassifier {
    fn classify(&self, features: &TaskFeatures) -> Classification;
}

/// Deterministic fallback classifier: a weighted score over the observable
/// features. Weights are deliberately coarse — this is a stand-in for a
/// learned classifier, not a claim that difficulty is linear in tool calls.
/// Confidence is the margin from the 0.5 boundary, capped at 0.75 while the
/// request still has unresolved references (an ambiguous request is never
/// a confident classification, however easy it looks).
#[derive(Debug, Default)]
pub struct HeuristicClassifier;

impl HeuristicClassifier {
    fn score(features: &TaskFeatures) -> f64 {
        let mut d = 0.10f64;
        // Tool calls beyond the first each add difficulty.
        let extra_tools = features.planned_tool_calls.saturating_sub(1);
        d += (extra_tools as f64 * 0.08).min(0.35);
        d += (features.unresolved_references as f64 * 0.12).min(0.30);
        d += (features.retries_so_far as f64 * 0.10).min(0.20);
        d += (features.input_tokens as f64 / 10_000.0).min(0.10);
        if features.needs_verified_fact {
            d += 0.08;
        }
        if features.irreversible_effects {
            d += 0.35;
        }
        d.clamp(0.0, 1.0)
    }
}

impl DifficultyClassifier for HeuristicClassifier {
    fn classify(&self, features: &TaskFeatures) -> Classification {
        let difficulty = Self::score(features);
        let mut confidence = 0.5 + (difficulty - 0.5).abs();
        if features.unresolved_references > 0 {
            confidence = confidence.min(0.75);
        }
        Classification {
            difficulty,
            confidence: confidence.clamp(0.0, 0.95),
        }
    }
}

/// Which model a request was sent to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    Small,
    Large,
}

/// Why the router chose this route. The reason is logged with the decision
/// so a later audit can tell a confident easy call from a cautious one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteReason {
    /// Confidently easy: below the threshold with enough confidence.
    ConfidentEasy,
    /// Hard enough that the small model should not touch it.
    Hard,
    /// Looked easy but the classifier was unsure: safe direction wins.
    Uncertain,
    /// Irreversible effects: always the large model, no classifier vote.
    Irreversible,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RouteDecision {
    pub route: Route,
    pub difficulty: f64,
    pub confidence: f64,
    pub reason: RouteReason,
}

/// What happened after a routed request ran. The router only learns from
/// outcomes that say something about the routing decision itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// Routed small and finished: the call was right.
    SmallSucceeded,
    /// Routed small, failed, had to re-run on large: a costly misroute.
    SmallFailedNeedsLarge,
    /// Routed large, but the task turned out trivial: a cheap misroute.
    LargeSucceededEasily,
    /// Routed large and the large model was actually needed.
    LargeSucceeded,
    /// Routed large and still failed: says nothing about routing.
    LargeFailed,
}

/// Cascade router: difficulty classifier + asymmetric decision rule +
/// outcome-driven threshold calibration.
pub struct CascadeRouter {
    classifier: Box<dyn DifficultyClassifier>,
    /// Difficulty at or below this goes to the small model (when confident).
    small_threshold: f64,
    min_threshold: f64,
    max_threshold: f64,
    threshold_step: f64,
    /// Minimum classifier confidence for a small-model route.
    confidence_floor: f64,
    /// Cheap misroutes needed in a row before the bar rises.
    cheap_streak_to_raise: u32,
    cheap_streak: u32,
    small_cost: f64,
    large_cost: f64,
    pub routed_small: u64,
    pub routed_large: u64,
    pub costly_misroutes: u64,
    pub cheap_misroutes: u64,
    pub total_cost: f64,
}

impl CascadeRouter {
    pub fn new(classifier: Box<dyn DifficultyClassifier>) -> Self {
        CascadeRouter {
            classifier,
            small_threshold: 0.45,
            min_threshold: 0.20,
            max_threshold: 0.70,
            threshold_step: 0.05,
            confidence_floor: 0.60,
            cheap_streak_to_raise: 3,
            cheap_streak: 0,
            small_cost: 1.0,
            large_cost: 10.0,
            routed_small: 0,
            routed_large: 0,
            costly_misroutes: 0,
            cheap_misroutes: 0,
            total_cost: 0.0,
        }
    }

    /// Convenience constructor with the deterministic fallback classifier.
    pub fn heuristic() -> Self {
        Self::new(Box::new(HeuristicClassifier))
    }

    pub fn with_costs(mut self, small_cost: f64, large_cost: f64) -> Self {
        self.small_cost = small_cost;
        self.large_cost = large_cost;
        self
    }

    pub fn small_threshold(&self) -> f64 {
        self.small_threshold
    }

    /// Route one request. Irreversible effects bypass the classifier
    /// entirely; everything else follows the asymmetric rule.
    pub fn route(&mut self, features: &TaskFeatures) -> RouteDecision {
        if features.irreversible_effects {
            self.routed_large += 1;
            self.total_cost += self.large_cost;
            return RouteDecision {
                route: Route::Large,
                difficulty: 1.0,
                confidence: 1.0,
                reason: RouteReason::Irreversible,
            };
        }
        let Classification {
            difficulty,
            confidence,
        } = self.classifier.classify(features);
        if difficulty <= self.small_threshold && confidence >= self.confidence_floor {
            self.routed_small += 1;
            self.total_cost += self.small_cost;
            RouteDecision {
                route: Route::Small,
                difficulty,
                confidence,
                reason: RouteReason::ConfidentEasy,
            }
        } else if difficulty <= self.small_threshold {
            self.routed_large += 1;
            self.total_cost += self.large_cost;
            RouteDecision {
                route: Route::Large,
                difficulty,
                confidence,
                reason: RouteReason::Uncertain,
            }
        } else {
            self.routed_large += 1;
            self.total_cost += self.large_cost;
            RouteDecision {
                route: Route::Large,
                difficulty,
                confidence,
                reason: RouteReason::Hard,
            }
        }
    }

    /// Feed an outcome back. A costly misroute lowers the bar at once;
    /// the bar only rises after `cheap_streak_to_raise` cheap misroutes
    /// in a row. Both moves are bounded so calibration can neither shut
    /// the small model off nor let it take everything.
    pub fn report_outcome(&mut self, outcome: Outcome) {
        match outcome {
            Outcome::SmallFailedNeedsLarge => {
                self.costly_misroutes += 1;
                // The re-run on the large model costs as well.
                self.total_cost += self.large_cost;
                self.cheap_streak = 0;
                self.small_threshold =
                    (self.small_threshold - self.threshold_step).max(self.min_threshold);
            }
            Outcome::LargeSucceededEasily => {
                self.cheap_misroutes += 1;
                self.cheap_streak += 1;
                if self.cheap_streak >= self.cheap_streak_to_raise {
                    self.cheap_streak = 0;
                    self.small_threshold =
                        (self.small_threshold + self.threshold_step).min(self.max_threshold);
                }
            }
            Outcome::SmallSucceeded | Outcome::LargeSucceeded | Outcome::LargeFailed => {
                // No routing signal: a success on the chosen model says the
                // call was at least not wrong, a large-model failure says
                // nothing about whether small could have done it.
            }
        }
    }

    /// Cost saved against the all-large baseline. Negative when costly
    /// misroutes (small + re-run on large) outweighed the savings.
    pub fn cost_saved(&self) -> f64 {
        let routed = self.routed_small + self.routed_large;
        routed as f64 * self.large_cost - self.total_cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    fn easy() -> TaskFeatures {
        TaskFeatures {
            planned_tool_calls: 1,
            input_tokens: 100,
            ..Default::default()
        }
    }

    /// A classifier that always returns the same verdict, for testing the
    /// router's decision rule in isolation from the heuristic.
    struct FixedClassifier {
        difficulty: f64,
        confidence: f64,
    }

    impl DifficultyClassifier for FixedClassifier {
        fn classify(&self, _f: &TaskFeatures) -> Classification {
            Classification {
                difficulty: self.difficulty,
                confidence: self.confidence,
            }
        }
    }

    fn router_with(difficulty: f64, confidence: f64) -> CascadeRouter {
        CascadeRouter::new(Box::new(FixedClassifier {
            difficulty,
            confidence,
        }))
    }

    #[test]
    fn heuristic_scores_an_easy_request_low() {
        let c = HeuristicClassifier.classify(&easy());
        // 0.10 base + 100/10_000 tokens; a single tool call adds nothing.
        assert!(close(c.difficulty, 0.11), "got {}", c.difficulty);
        assert!(close(c.confidence, 0.89), "got {}", c.confidence);
    }

    #[test]
    fn heuristic_difficulty_grows_with_tools_refs_and_retries() {
        let hard = TaskFeatures {
            planned_tool_calls: 6,
            unresolved_references: 2,
            retries_so_far: 1,
            input_tokens: 100,
            ..Default::default()
        };
        let c = HeuristicClassifier.classify(&hard);
        assert!(c.difficulty > 0.7, "got {}", c.difficulty);
        // Unresolved references cap confidence even for a clear verdict.
        assert!(c.confidence <= 0.75);
    }

    #[test]
    fn easy_confident_request_goes_small() {
        let mut r = router_with(0.2, 0.9);
        let d = r.route(&easy());
        assert_eq!(d.route, Route::Small);
        assert_eq!(d.reason, RouteReason::ConfidentEasy);
        assert_eq!(r.routed_small, 1);
    }

    #[test]
    fn hard_request_goes_large() {
        let mut r = router_with(0.8, 0.9);
        let d = r.route(&easy());
        assert_eq!(d.route, Route::Large);
        assert_eq!(d.reason, RouteReason::Hard);
    }

    #[test]
    fn easy_but_uncertain_goes_large_the_safe_direction() {
        // Looks easy (0.2) but the classifier is unsure (0.5 < 0.60 floor):
        // the cheap error is sending it to large, the expensive error is
        // the reverse, so it goes large.
        let mut r = router_with(0.2, 0.5);
        let d = r.route(&easy());
        assert_eq!(d.route, Route::Large);
        assert_eq!(d.reason, RouteReason::Uncertain);
    }

    #[test]
    fn irreversible_effects_bypass_the_classifier() {
        // Even a classifier screaming "trivial" cannot override this.
        let mut r = router_with(0.05, 0.99);
        let mut f = easy();
        f.irreversible_effects = true;
        let d = r.route(&f);
        assert_eq!(d.route, Route::Large);
        assert_eq!(d.reason, RouteReason::Irreversible);
    }

    #[test]
    fn heuristic_end_to_end_easy_goes_small_hard_goes_large() {
        let mut r = CascadeRouter::heuristic();
        assert_eq!(r.route(&easy()).route, Route::Small);
        let hard = TaskFeatures {
            planned_tool_calls: 6,
            unresolved_references: 2,
            retries_so_far: 1,
            input_tokens: 100,
            ..Default::default()
        };
        assert_eq!(r.route(&hard).route, Route::Large);
    }

    #[test]
    fn costly_misroute_lowers_the_threshold_at_once() {
        let mut r = router_with(0.2, 0.9);
        assert!(close(r.small_threshold(), 0.45));
        r.route(&easy());
        r.report_outcome(Outcome::SmallFailedNeedsLarge);
        assert_eq!(r.costly_misroutes, 1);
        assert!(close(r.small_threshold(), 0.40), "got {}", r.small_threshold());
    }

    #[test]
    fn cheap_misroutes_raise_the_threshold_only_after_a_streak() {
        let mut r = router_with(0.2, 0.9);
        r.report_outcome(Outcome::LargeSucceededEasily);
        r.report_outcome(Outcome::LargeSucceededEasily);
        assert!(close(r.small_threshold(), 0.45), "raised too early");
        r.report_outcome(Outcome::LargeSucceededEasily);
        assert!(close(r.small_threshold(), 0.50), "got {}", r.small_threshold());
        assert_eq!(r.cheap_misroutes, 3);
    }

    #[test]
    fn a_costly_misroute_resets_the_cheap_streak() {
        let mut r = router_with(0.2, 0.9);
        r.report_outcome(Outcome::LargeSucceededEasily);
        r.report_outcome(Outcome::LargeSucceededEasily);
        r.report_outcome(Outcome::SmallFailedNeedsLarge);
        r.report_outcome(Outcome::LargeSucceededEasily);
        r.report_outcome(Outcome::LargeSucceededEasily);
        // Only 2 cheap in a row after the reset: threshold fell, not rose.
        assert!(close(r.small_threshold(), 0.40), "got {}", r.small_threshold());
    }

    #[test]
    fn threshold_is_bounded_on_both_sides() {
        let mut r = router_with(0.2, 0.9);
        for _ in 0..20 {
            r.report_outcome(Outcome::SmallFailedNeedsLarge);
        }
        assert!(close(r.small_threshold(), 0.20), "got {}", r.small_threshold());
        let mut r2 = router_with(0.2, 0.9);
        for _ in 0..30 {
            r2.report_outcome(Outcome::LargeSucceededEasily);
        }
        assert!(close(r2.small_threshold(), 0.70), "got {}", r2.small_threshold());
    }

    #[test]
    fn cost_accounting_counts_savings_against_all_large() {
        let mut r = CascadeRouter::heuristic();
        assert_eq!(r.route(&easy()).route, Route::Small); // cost 1
        let hard = TaskFeatures {
            planned_tool_calls: 6,
            ..Default::default()
        };
        assert_eq!(r.route(&hard).route, Route::Large); // cost 10
        assert!(close(r.total_cost, 11.0));
        assert!(close(r.cost_saved(), 9.0), "got {}", r.cost_saved());
    }

    #[test]
    fn costly_misroute_charges_the_re_run() {
        let mut r = router_with(0.2, 0.9);
        r.route(&easy()); // small, cost 1
        r.report_outcome(Outcome::SmallFailedNeedsLarge); // re-run, cost 10
        assert!(close(r.total_cost, 11.0));
        // One small + one re-run cost more than sending it large directly.
        assert!(close(r.cost_saved(), -1.0), "got {}", r.cost_saved());
    }

    #[test]
    fn neutral_outcomes_do_not_move_the_threshold() {
        let mut r = router_with(0.2, 0.9);
        r.report_outcome(Outcome::SmallSucceeded);
        r.report_outcome(Outcome::LargeSucceeded);
        r.report_outcome(Outcome::LargeFailed);
        assert!(close(r.small_threshold(), 0.45));
    }
}
