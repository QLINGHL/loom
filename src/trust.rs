//! User trust calibration (doc ch. 21, open question 7).
//!
//! Users drift to one of two extremes: **over-trust** (approve without
//! reading) or **over-suspicion** (reject everything). Chapter 14's attention
//! congestion control reduces *how often* the system interrupts, but it
//! cannot guarantee that each interruption asks at the critical point — a
//! confirmation the user rubber-stamps is no confirmation at all, and a
//! confirmation the user reflexively rejects teaches the system nothing.
//!
//! This module closes that loop:
//!
//! * **Signals** — every confirmation interaction is recorded as a
//!   [`DecisionEvent`]: risk tier, friction shown, decision, dwell time
//!   (how long the user looked before deciding), whether evidence was
//!   actually viewed, and the later outcome (stood, or undone = regretted).
//! * **Calibration index** — a rolling score in `[-1, +1]`: positive means
//!   the user is drifting toward over-trust, negative toward over-suspicion.
//!   Blind approvals of medium-or-higher risk push it up; blanket rejects
//!   of low-risk items push it down; approvals later undone count double
//!   (they are the only ground truth that trust was misplaced). The score
//!   decays toward neutral over time so a single bad afternoon does not
//!   permanently re-tune the system.
//! * **Adaptive friction** — [`TrustCalibrator::friction_for`] maps
//!   (risk tier, calibration) to a [`FrictionLevel`]. Over-trust escalates
//!   friction on medium+ risk (evidence must be viewed before approve is
//!   enabled); over-suspicion de-escalates low/medium risk (batch into a
//!   digest instead of interrupting) so the user is not trained to reject
//!   by sheer fatigue. Critical risk always demands explicit confirmation
//!   regardless of calibration — calibration tunes the *presentation*, never
//!   the safety floor.
//! * **Self-calibration** — the thresholds that define "blind" (dwell) and
//!   "blanket" (reject rate) adapt: a rising regret rate lowers the blind
//!   threshold (less patience for fast approvals) and raises the evidence
//!   requirement. Hysteresis on the state boundaries keeps the friction
//!   level from flip-flopping between consecutive confirmations.

use crate::Clock;
use std::collections::VecDeque;

/// How bad it is if this action turns out to be wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RiskTier {
    /// Reversible, trivially undoable (mark read, reorder a list).
    Low,
    /// Reversible but costly to undo (bulk edit, send a draft).
    Medium,
    /// Hard to undo (publish, share externally, spend money).
    High,
    /// Irreversible (delete, transfer, grant access).
    Critical,
}

/// What the user must do before the decision counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FrictionLevel {
    /// No interruption: act, log it, report later.
    None,
    /// One-line rationale shown inline; approve is one tap.
    Summary,
    /// Evidence must be opened and viewed before approve is enabled.
    EvidenceRequired,
    /// User must restate the key fact (type it / pick it from a list):
    /// proves they read the confirmation, not just the button.
    Explicit,
}

impl FrictionLevel {
    /// One step up the ladder, saturating at [`FrictionLevel::Explicit`].
    pub fn escalate(self) -> FrictionLevel {
        match self {
            FrictionLevel::None => FrictionLevel::Summary,
            FrictionLevel::Summary => FrictionLevel::EvidenceRequired,
            FrictionLevel::EvidenceRequired => FrictionLevel::Explicit,
            FrictionLevel::Explicit => FrictionLevel::Explicit,
        }
    }

    /// One step down the ladder, saturating at [`FrictionLevel::None`].
    pub fn de_escalate(self) -> FrictionLevel {
        match self {
            FrictionLevel::None => FrictionLevel::None,
            FrictionLevel::Summary => FrictionLevel::None,
            FrictionLevel::EvidenceRequired => FrictionLevel::Summary,
            FrictionLevel::Explicit => FrictionLevel::EvidenceRequired,
        }
    }
}

/// Which way the user's trust is miscalibrated, if either.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrustDrift {
    /// Approving without reading: rubber-stamping.
    OverTrust,
    /// Rejecting (nearly) everything, including the safe stuff.
    OverSuspicion,
    /// Within tolerance: friction stays at the risk-based default.
    Calibrated,
}

/// One confirmation interaction, as observed by the system.
#[derive(Debug, Clone, Copy)]
pub struct DecisionEvent {
    /// Risk of the action being confirmed.
    pub risk: RiskTier,
    /// Friction that was actually presented.
    pub friction: FrictionLevel,
    /// Whether the user approved.
    pub approved: bool,
    /// Milliseconds between showing the confirmation and the decision.
    pub dwell_ms: u64,
    /// Whether the user opened the evidence view before deciding.
    pub evidence_viewed: bool,
    /// Later outcome: true if an approval was later undone / regretted.
    /// Only meaningful for approvals; ignored for rejections.
    pub regretted: bool,
}

impl DecisionEvent {
    /// A quick approve of a non-trivial action without looking at evidence:
    /// the signature of over-trust. `blind_dwell_ms` is the calibrated
    /// threshold — approvals faster than this count as blind.
    pub fn is_blind_approve(&self, blind_dwell_ms: u64) -> bool {
        self.approved
            && self.risk >= RiskTier::Medium
            && self.dwell_ms < blind_dwell_ms
            && !self.evidence_viewed
    }

    /// A reject of a low-risk item: harmless on its own, but a *pattern* of
    /// them is the signature of over-suspicion.
    pub fn is_blanket_reject(&self) -> bool {
        !self.approved && self.risk == RiskTier::Low
    }
}

/// Snapshot of the calibrator's current state, for the metrics ledger.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrustReport {
    /// Calibration index in [-1, +1]; + = over-trust, − = over-suspicion.
    pub index: f64,
    pub drift: TrustDrift,
    /// Blind-approve rate over the window (0..1).
    pub blind_approve_rate: f64,
    /// Low-risk reject rate over the window (0..1).
    pub blanket_reject_rate: f64,
    /// Approvals later undone, over the window (0..1).
    pub regret_rate: f64,
    /// Current blind-dwell threshold (ms); self-calibrates with regret.
    pub blind_dwell_ms: u64,
    /// Events in the rolling window.
    pub window_events: usize,
}

/// The trust calibrator: keeps a rolling window of decision events and
/// derives both a drift classification and the friction to present next.
pub struct TrustCalibrator<'a> {
    clock: &'a dyn Clock,
    /// Rolling window of recent events (bounded).
    window: VecDeque<DecisionEvent>,
    /// Calibration index in [-1, +1].
    index: f64,
    /// ms since the index was last decayed toward neutral.
    last_decay_ms: u64,
    /// Dwell below which an approve counts as blind; self-calibrates.
    blind_dwell_ms: u64,
    /// Regret rate at the last self-calibration check.
    last_regret_rate: f64,
}

const WINDOW_CAP: usize = 50;
const DECAY_INTERVAL_MS: u64 = 60_000;
const DECAY_PER_INTERVAL: f64 = 0.02;
const DEFAULT_BLIND_DWELL_MS: u64 = 1_500;
const MAX_BLIND_DWELL_MS: u64 = 4_000;
/// Drift classification needs a margin: inside ±this the user counts as
/// calibrated, so friction does not flip-flop on noise.
const DRIFT_MARGIN: f64 = 0.25;

impl<'a> TrustCalibrator<'a> {
    pub fn new(clock: &'a dyn Clock) -> Self {
        TrustCalibrator {
            clock,
            window: VecDeque::new(),
            index: 0.0,
            last_decay_ms: clock.now_ms(),
            blind_dwell_ms: DEFAULT_BLIND_DWELL_MS,
            last_regret_rate: 0.0,
        }
    }

    /// Record one confirmation outcome and update the calibration index.
    pub fn record(&mut self, event: DecisionEvent) {
        self.decay();
        if self.window.len() == WINDOW_CAP {
            self.window.pop_front();
        }
        self.window.push_back(event);

        // Weight each signal; regret is ground truth and counts double.
        let mut delta = 0.0;
        if event.is_blind_approve(self.blind_dwell_ms) {
            delta += 0.10;
        }
        if event.approved && event.regretted {
            delta += 0.20;
        }
        if event.is_blanket_reject() {
            delta -= 0.08;
        }
        // A considered approval of a risky item (evidence viewed, slow
        // dwell) is evidence the user *is* engaging: pull back toward
        // neutral from the over-trust side only.
        if event.approved
            && event.risk >= RiskTier::Medium
            && event.evidence_viewed
            && event.dwell_ms >= self.blind_dwell_ms
        {
            delta -= 0.05;
        }
        // A considered reject of a high-risk item is correct caution, not
        // suspicion: pull back toward neutral from the over-suspicion side.
        if !event.approved && event.risk >= RiskTier::High {
            delta += 0.05;
        }
        self.index = (self.index + delta).clamp(-1.0, 1.0);
        self.self_calibrate();
    }

    /// Which way the user is drifting, with a dead band around neutral.
    pub fn drift(&self) -> TrustDrift {
        if self.index > DRIFT_MARGIN {
            TrustDrift::OverTrust
        } else if self.index < -DRIFT_MARGIN {
            TrustDrift::OverSuspicion
        } else {
            TrustDrift::Calibrated
        }
    }

    /// The friction to present for a confirmation of this risk, given the
    /// current calibration. The safety floor is absolute: Critical risk is
    /// always [`FrictionLevel::Explicit`], whatever the calibration says.
    pub fn friction_for(&self, risk: RiskTier) -> FrictionLevel {
        let base = match risk {
            RiskTier::Low => FrictionLevel::None,
            RiskTier::Medium => FrictionLevel::Summary,
            RiskTier::High => FrictionLevel::EvidenceRequired,
            RiskTier::Critical => FrictionLevel::Explicit,
        };
        if risk == RiskTier::Critical {
            return base;
        }
        match self.drift() {
            TrustDrift::OverTrust => {
                // Rubber-stamping: make medium+ risk impossible to approve
                // without looking. Low risk stays cheap — escalating it
                // would only train the user to dismiss friction entirely.
                if risk >= RiskTier::Medium {
                    base.escalate()
                } else {
                    base
                }
            }
            TrustDrift::OverSuspicion => {
                // Reflexive rejecting is usually fatigue: stop interrupting
                // for the safe stuff so the interruptions that remain are
                // worth reading.
                if risk <= RiskTier::Medium {
                    base.de_escalate()
                } else {
                    base
                }
            }
            TrustDrift::Calibrated => base,
        }
    }

    /// Should this confirmation interrupt the user now, or wait for a
    /// natural pause / digest? Critical risk always interrupts; anything
    /// else is deferred while the user is over-suspicious (their "no" is
    /// currently unreliable) and while the recent interruption count is
    /// already high. `recent_interruptions` is the count in the last hour,
    /// supplied by the attention scheduler (ch. 14).
    pub fn should_interrupt_now(
        &self,
        risk: RiskTier,
        recent_interruptions: u32,
        hourly_budget: u32,
    ) -> bool {
        if risk == RiskTier::Critical {
            return true;
        }
        if recent_interruptions >= hourly_budget {
            return false;
        }
        match self.drift() {
            // Over-suspicious users reject reflexively: a non-critical
            // interruption now is wasted, defer it to a digest.
            TrustDrift::OverSuspicion => risk == RiskTier::High,
            _ => true,
        }
    }

    pub fn report(&self) -> TrustReport {
        let n = self.window.len();
        let (mut blind, mut blanket, mut approvals, mut regrets) = (0usize, 0usize, 0usize, 0usize);
        for e in &self.window {
            if e.is_blind_approve(self.blind_dwell_ms) {
                blind += 1;
            }
            if e.is_blanket_reject() {
                blanket += 1;
            }
            if e.approved {
                approvals += 1;
                if e.regretted {
                    regrets += 1;
                }
            }
        }
        let rate = |x: usize| if n == 0 { 0.0 } else { x as f64 / n as f64 };
        TrustReport {
            index: self.index,
            drift: self.drift(),
            blind_approve_rate: rate(blind),
            blanket_reject_rate: rate(blanket),
            regret_rate: if approvals == 0 {
                0.0
            } else {
                regrets as f64 / approvals as f64
            },
            blind_dwell_ms: self.blind_dwell_ms,
            window_events: n,
        }
    }

    /// Decay the index toward neutral with elapsed time, and re-tune the
    /// blind threshold from the observed regret rate.
    fn decay(&mut self) {
        let now = self.clock.now_ms();
        let elapsed = now.saturating_sub(self.last_decay_ms);
        if elapsed >= DECAY_INTERVAL_MS {
            let steps = (elapsed / DECAY_INTERVAL_MS) as f64;
            let pull = DECAY_PER_INTERVAL * steps;
            self.index = if self.index > 0.0 {
                (self.index - pull).max(0.0)
            } else {
                (self.index + pull).min(0.0)
            };
            self.last_decay_ms = now;
        }
    }

    /// Self-calibration: a rising regret rate means the system is being too
    /// patient with fast approvals, so the blind threshold grows (more
    /// approvals count as blind, friction escalates sooner). A zero regret
    /// rate over a full window relaxes the threshold back toward default.
    fn self_calibrate(&mut self) {
        let regret = self.report().regret_rate;
        if regret > self.last_regret_rate + 0.05 {
            // Regret is climbing: tighten.
            self.blind_dwell_ms = (self.blind_dwell_ms + 500).min(MAX_BLIND_DWELL_MS);
        } else if regret == 0.0 && self.window.len() >= WINDOW_CAP / 2 {
            // Clean record over a long window: relax one notch toward the
            // default, never below it.
            self.blind_dwell_ms = self
                .blind_dwell_ms
                .saturating_sub(250)
                .max(DEFAULT_BLIND_DWELL_MS);
        }
        self.last_regret_rate = regret;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ManualClock;

    fn ev(risk: RiskTier, approved: bool, dwell_ms: u64, evidence: bool, regretted: bool) -> DecisionEvent {
        DecisionEvent {
            risk,
            friction: FrictionLevel::Summary,
            approved,
            dwell_ms,
            evidence_viewed: evidence,
            regretted,
        }
    }

    #[test]
    fn fresh_calibrator_is_neutral_with_risk_based_friction() {
        let clock = ManualClock::new(0);
        let cal = TrustCalibrator::new(&clock);
        assert_eq!(cal.drift(), TrustDrift::Calibrated);
        assert_eq!(cal.friction_for(RiskTier::Low), FrictionLevel::None);
        assert_eq!(cal.friction_for(RiskTier::Medium), FrictionLevel::Summary);
        assert_eq!(cal.friction_for(RiskTier::High), FrictionLevel::EvidenceRequired);
        assert_eq!(cal.friction_for(RiskTier::Critical), FrictionLevel::Explicit);
    }

    #[test]
    fn repeated_blind_approvals_push_to_over_trust_and_escalate_friction() {
        let clock = ManualClock::new(0);
        let mut cal = TrustCalibrator::new(&clock);
        for _ in 0..5 {
            cal.record(ev(RiskTier::Medium, true, 200, false, false));
        }
        assert_eq!(cal.drift(), TrustDrift::OverTrust);
        // Medium risk now demands evidence before approve is enabled.
        assert_eq!(cal.friction_for(RiskTier::Medium), FrictionLevel::EvidenceRequired);
        assert_eq!(cal.friction_for(RiskTier::High), FrictionLevel::Explicit);
        // Low risk stays cheap: escalating it would train dismissal.
        assert_eq!(cal.friction_for(RiskTier::Low), FrictionLevel::None);
    }

    #[test]
    fn blanket_rejects_push_to_over_suspicion_and_de_escalate_low_risk() {
        let clock = ManualClock::new(0);
        let mut cal = TrustCalibrator::new(&clock);
        for _ in 0..8 {
            cal.record(ev(RiskTier::Low, false, 300, false, false));
        }
        assert_eq!(cal.drift(), TrustDrift::OverSuspicion);
        // Medium risk drops to no interruption (digest instead).
        assert_eq!(cal.friction_for(RiskTier::Medium), FrictionLevel::None);
        // High risk keeps its evidence requirement: safety floor holds.
        assert_eq!(cal.friction_for(RiskTier::High), FrictionLevel::EvidenceRequired);
    }

    #[test]
    fn critical_risk_is_always_explicit_whatever_the_drift() {
        let clock = ManualClock::new(0);
        let mut cal = TrustCalibrator::new(&clock);
        for _ in 0..8 {
            cal.record(ev(RiskTier::Low, false, 300, false, false));
        }
        assert_eq!(cal.drift(), TrustDrift::OverSuspicion);
        assert_eq!(cal.friction_for(RiskTier::Critical), FrictionLevel::Explicit);
        for _ in 0..5 {
            cal.record(ev(RiskTier::Medium, true, 100, false, false));
        }
        assert_eq!(cal.friction_for(RiskTier::Critical), FrictionLevel::Explicit);
    }

    #[test]
    fn regretted_approvals_count_double_and_tighten_blind_threshold() {
        let clock = ManualClock::new(0);
        let mut cal = TrustCalibrator::new(&clock);
        let before = cal.report().blind_dwell_ms;
        for _ in 0..3 {
            cal.record(ev(RiskTier::High, true, 200, false, true));
        }
        assert_eq!(cal.drift(), TrustDrift::OverTrust);
        assert!(cal.report().blind_dwell_ms > before);
        assert!(cal.report().regret_rate > 0.0);
    }

    #[test]
    fn considered_approvals_pull_back_from_over_trust() {
        let clock = ManualClock::new(0);
        let mut cal = TrustCalibrator::new(&clock);
        for _ in 0..4 {
            cal.record(ev(RiskTier::Medium, true, 200, false, false));
        }
        assert_eq!(cal.drift(), TrustDrift::OverTrust);
        // Slow, evidence-backed approvals show the user is engaging again.
        for _ in 0..6 {
            cal.record(ev(RiskTier::Medium, true, 5_000, true, false));
        }
        assert_eq!(cal.drift(), TrustDrift::Calibrated);
    }

    #[test]
    fn index_decays_toward_neutral_over_time() {
        let clock = ManualClock::new(0);
        let mut cal = TrustCalibrator::new(&clock);
        for _ in 0..5 {
            cal.record(ev(RiskTier::Medium, true, 200, false, false));
        }
        assert_eq!(cal.drift(), TrustDrift::OverTrust);
        clock.advance(60_000 * 60); // an hour of no interaction
        cal.record(ev(RiskTier::Low, true, 5_000, true, false));
        // Decay (60 steps x 0.02) wipes the +0.5 index back to neutral.
        assert_eq!(cal.drift(), TrustDrift::Calibrated);
    }

    #[test]
    fn should_interrupt_now_defers_noncritical_when_over_suspicious() {
        let clock = ManualClock::new(0);
        let mut cal = TrustCalibrator::new(&clock);
        for _ in 0..8 {
            cal.record(ev(RiskTier::Low, false, 300, false, false));
        }
        assert!(!cal.should_interrupt_now(RiskTier::Medium, 0, 10));
        assert!(cal.should_interrupt_now(RiskTier::High, 0, 10));
        assert!(cal.should_interrupt_now(RiskTier::Critical, 0, 10));
        // Hourly budget exhausted: even high risk waits (critical excepted).
        assert!(!cal.should_interrupt_now(RiskTier::High, 10, 10));
        assert!(cal.should_interrupt_now(RiskTier::Critical, 10, 10));
    }

    #[test]
    fn report_rates_reflect_the_window() {
        let clock = ManualClock::new(0);
        let mut cal = TrustCalibrator::new(&clock);
        cal.record(ev(RiskTier::Medium, true, 200, false, false));
        cal.record(ev(RiskTier::Low, false, 300, false, false));
        cal.record(ev(RiskTier::High, true, 5_000, true, false));
        let r = cal.report();
        assert_eq!(r.window_events, 3);
        assert!((r.blind_approve_rate - 1.0 / 3.0).abs() < 1e-9);
        assert!((r.blanket_reject_rate - 1.0 / 3.0).abs() < 1e-9);
        assert_eq!(r.regret_rate, 0.0);
    }

    #[test]
    fn friction_ladder_saturates_at_both_ends() {
        assert_eq!(FrictionLevel::Explicit.escalate(), FrictionLevel::Explicit);
        assert_eq!(FrictionLevel::None.de_escalate(), FrictionLevel::None);
        assert_eq!(FrictionLevel::Summary.escalate(), FrictionLevel::EvidenceRequired);
        assert_eq!(FrictionLevel::EvidenceRequired.de_escalate(), FrictionLevel::Summary);
    }
}
