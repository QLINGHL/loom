//! Metrics dashboard (doc ch. 22): the OS-level numbers that task
//! completion alone never shows — how often intents are served by a
//! deterministic path, how often they land first try, how often the
//! system interrupts for nothing, how often executed actions get taken
//! back, and how often a request finds its model already resident.
//!
//! Two honest input paths, kept separate on purpose:
//!
//! * [`Metrics::record`] takes typed [`MetricEvent`]s at the point where
//!   the fact is known (the determinization engine knows which path
//!   served an intent; the attention scheduler learns whether an
//!   interruption was worthwhile). These are the only exact source for
//!   the five headline rates.
//! * [`Metrics::absorb_ledger`] derives what the kernel ledger can prove
//!   without guessing: every `Act` is one executed action, every
//!   `Compensation` is one action actually taken back, so the ledger
//!   yields the revocation rate directly. It deliberately does *not*
//!   derive the other rates: a ledger entry has no field for "served
//!   deterministically", "clarifications asked", "interruption was
//!   worthwhile", or "model was resident", and parsing the free-text
//!   `detail` field to invent those would be fabrication. Likewise
//!   `TxnUndone` / `TxnAborted` are transaction-level markers, not
//!   extra undone actions — counting them on top of their
//!   `Compensation` steps would double-count. And `ActDenied` mixes
//!   scope, speculation-barrier and quota denials with privilege
//!   denials, so the privilege-interception rate is event-only too.
//!
//! A rate is `None` until its denominator is non-zero: "no data" is
//! reported as no data, never as a flattering 0% or 100%.

use crate::kernel::{LedgerEntry, LedgerKind};

/// Doc ch. 22 targets. Determination / first-try / residency are
/// lower bounds ("at least"); false-interrupt / revocation /
/// privilege-interception-miss are upper bounds ("at most").
pub const TARGET_DETERMINIZATION: f64 = 0.80;
pub const TARGET_FIRST_TRY: f64 = 0.70;
pub const TARGET_FALSE_INTERRUPT_MAX: f64 = 0.05;
pub const TARGET_REVOCATION_MAX: f64 = 0.03;
pub const TARGET_RESIDENCY: f64 = 0.90;
pub const TARGET_PRIVILEGE_INTERCEPTION: f64 = 0.999;

/// One countable fact, recorded where it is known.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricEvent {
    /// An intent was served; `deterministic` means via an L1/L2/L3
    /// deterministic path (doc ch. 22 "确定化率"), not ad-hoc planning.
    IntentServed { deterministic: bool },
    /// An intent completed after `clarifications` follow-up questions;
    /// zero means it landed first try.
    IntentCompleted { clarifications: u32 },
    /// The system interrupted the user; `worthwhile` is the user's own
    /// verdict afterwards. A not-worthwhile interruption is a false one.
    Interruption { worthwhile: bool },
    /// One action executed through the kernel.
    ActionExecuted,
    /// One executed action taken back (undo / compensation).
    ActionUndone,
    /// One inference request; `resident_hit` means its model was
    /// already loaded, so no weights had to move.
    ModelRequest { resident_hit: bool },
    /// One privilege check; `denied` means the kernel blocked it.
    PrivilegeAttempt { denied: bool },
}

/// Raw counters behind a [`Dashboard`]. Saturating: a counter that
/// somehow reached `u64::MAX` stays put rather than wrapping into a
/// rate that looks better than it is.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Metrics {
    pub intents_served: u64,
    pub deterministic_served: u64,
    pub intents_completed: u64,
    pub first_try_completed: u64,
    pub interruptions: u64,
    pub worthless_interruptions: u64,
    pub actions_executed: u64,
    pub actions_undone: u64,
    pub model_requests: u64,
    pub resident_hits: u64,
    pub privilege_attempts: u64,
    pub privilege_denied: u64,
}

fn rate(numerator: u64, denominator: u64) -> Option<f64> {
    if denominator == 0 {
        None
    } else {
        Some(numerator as f64 / denominator as f64)
    }
}

impl Metrics {
    pub fn new() -> Self {
        Metrics::default()
    }

    pub fn record(&mut self, event: MetricEvent) {
        match event {
            MetricEvent::IntentServed { deterministic } => {
                self.intents_served = self.intents_served.saturating_add(1);
                if deterministic {
                    self.deterministic_served = self.deterministic_served.saturating_add(1);
                }
            }
            MetricEvent::IntentCompleted { clarifications } => {
                self.intents_completed = self.intents_completed.saturating_add(1);
                if clarifications == 0 {
                    self.first_try_completed = self.first_try_completed.saturating_add(1);
                }
            }
            MetricEvent::Interruption { worthwhile } => {
                self.interruptions = self.interruptions.saturating_add(1);
                if !worthwhile {
                    self.worthless_interruptions =
                        self.worthless_interruptions.saturating_add(1);
                }
            }
            MetricEvent::ActionExecuted => {
                self.actions_executed = self.actions_executed.saturating_add(1);
            }
            MetricEvent::ActionUndone => {
                self.actions_undone = self.actions_undone.saturating_add(1);
            }
            MetricEvent::ModelRequest { resident_hit } => {
                self.model_requests = self.model_requests.saturating_add(1);
                if resident_hit {
                    self.resident_hits = self.resident_hits.saturating_add(1);
                }
            }
            MetricEvent::PrivilegeAttempt { denied } => {
                self.privilege_attempts = self.privilege_attempts.saturating_add(1);
                if denied {
                    self.privilege_denied = self.privilege_denied.saturating_add(1);
                }
            }
        }
    }

    /// Fold a kernel ledger into the counters it can honestly feed:
    /// `Act` -> executed, `Compensation` -> undone. Everything else is
    /// ignored (see the module docs for why).
    pub fn absorb_ledger(&mut self, entries: &[LedgerEntry]) {
        for entry in entries {
            match entry.kind {
                LedgerKind::Act => self.record(MetricEvent::ActionExecuted),
                LedgerKind::Compensation => self.record(MetricEvent::ActionUndone),
                _ => {}
            }
        }
    }

    /// Add another counter set into this one (e.g. per-device shards).
    pub fn merge(&mut self, other: &Metrics) {
        self.intents_served = self.intents_served.saturating_add(other.intents_served);
        self.deterministic_served =
            self.deterministic_served.saturating_add(other.deterministic_served);
        self.intents_completed = self.intents_completed.saturating_add(other.intents_completed);
        self.first_try_completed =
            self.first_try_completed.saturating_add(other.first_try_completed);
        self.interruptions = self.interruptions.saturating_add(other.interruptions);
        self.worthless_interruptions =
            self.worthless_interruptions.saturating_add(other.worthless_interruptions);
        self.actions_executed = self.actions_executed.saturating_add(other.actions_executed);
        self.actions_undone = self.actions_undone.saturating_add(other.actions_undone);
        self.model_requests = self.model_requests.saturating_add(other.model_requests);
        self.resident_hits = self.resident_hits.saturating_add(other.resident_hits);
        self.privilege_attempts =
            self.privilege_attempts.saturating_add(other.privilege_attempts);
        self.privilege_denied = self.privilege_denied.saturating_add(other.privilege_denied);
    }

    pub fn snapshot(&self) -> Dashboard {
        Dashboard {
            determination_rate: rate(self.deterministic_served, self.intents_served),
            first_try_rate: rate(self.first_try_completed, self.intents_completed),
            false_interrupt_rate: rate(self.worthless_interruptions, self.interruptions),
            revocation_rate: rate(self.actions_undone, self.actions_executed),
            residency_hit_rate: rate(self.resident_hits, self.model_requests),
            privilege_interception_rate: rate(self.privilege_denied, self.privilege_attempts),
            counts: *self,
        }
    }
}

/// Whether a rate meets its ch. 22 target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetStatus {
    /// The target is met.
    Met,
    /// The target is missed.
    Missed,
    /// No data yet — neither met nor missed.
    NoData,
}

fn check_at_least(rate: Option<f64>, target: f64) -> TargetStatus {
    match rate {
        None => TargetStatus::NoData,
        Some(r) => {
            if r >= target {
                TargetStatus::Met
            } else {
                TargetStatus::Missed
            }
        }
    }
}

fn check_at_most(rate: Option<f64>, target: f64) -> TargetStatus {
    match rate {
        None => TargetStatus::NoData,
        Some(r) => {
            if r <= target {
                TargetStatus::Met
            } else {
                TargetStatus::Missed
            }
        }
    }
}

/// The dashboard: one snapshot of the ch. 22 metrics. Every rate is
/// `None` while its denominator is zero.
#[derive(Debug, Clone, Copy)]
pub struct Dashboard {
    /// Intents served by a deterministic (L1/L2/L3) path / all intents.
    /// Target: >= 80% (the north-star metric).
    pub determination_rate: Option<f64>,
    /// Intents completed with zero clarifications / all completions.
    /// Target: > 70%.
    pub first_try_rate: Option<f64>,
    /// Interruptions the user judged not worthwhile / all interruptions.
    /// Target: < 5%.
    pub false_interrupt_rate: Option<f64>,
    /// Actions taken back / actions executed. Target: < 3%.
    pub revocation_rate: Option<f64>,
    /// Requests hitting an already-resident model / all requests.
    /// Target: > 90%.
    pub residency_hit_rate: Option<f64>,
    /// Privilege attempts blocked / all privilege attempts.
    /// Target: > 99.9%.
    pub privilege_interception_rate: Option<f64>,
    /// The raw counters these rates were computed from.
    pub counts: Metrics,
}

fn fmt_rate(rate: Option<f64>) -> String {
    match rate {
        None => "—（无数据）".to_string(),
        Some(r) => format!("{:.1}%", r * 100.0),
    }
}

fn fmt_status(status: TargetStatus) -> &'static str {
    match status {
        TargetStatus::Met => "达标",
        TargetStatus::Missed => "未达标",
        TargetStatus::NoData => "无数据",
    }
}

impl Dashboard {
    pub fn determination_status(&self) -> TargetStatus {
        check_at_least(self.determination_rate, TARGET_DETERMINIZATION)
    }

    pub fn first_try_status(&self) -> TargetStatus {
        check_at_least(self.first_try_rate, TARGET_FIRST_TRY)
    }

    pub fn false_interrupt_status(&self) -> TargetStatus {
        check_at_most(self.false_interrupt_rate, TARGET_FALSE_INTERRUPT_MAX)
    }

    pub fn revocation_status(&self) -> TargetStatus {
        check_at_most(self.revocation_rate, TARGET_REVOCATION_MAX)
    }

    pub fn residency_status(&self) -> TargetStatus {
        check_at_least(self.residency_hit_rate, TARGET_RESIDENCY)
    }

    pub fn privilege_interception_status(&self) -> TargetStatus {
        check_at_least(self.privilege_interception_rate, TARGET_PRIVILEGE_INTERCEPTION)
    }

    /// One line per metric: name, rate, target, verdict. Deterministic
    /// output (fixed order, fixed precision) so dashboards are diffable.
    pub fn render(&self) -> String {
        let rows = [
            ("确定化率", self.determination_rate, "≥80%", self.determination_status()),
            ("一次到位率", self.first_try_rate, ">70%", self.first_try_status()),
            ("误中断率", self.false_interrupt_rate, "<5%", self.false_interrupt_status()),
            ("撤销率", self.revocation_rate, "<3%", self.revocation_status()),
            ("驻留命中率", self.residency_hit_rate, ">90%", self.residency_status()),
            (
                "越权拦截率",
                self.privilege_interception_rate,
                ">99.9%",
                self.privilege_interception_status(),
            ),
        ];
        let mut out = String::from("Loom 度量看板（第 22 章）\n");
        for (name, rate, target, status) in rows {
            out.push_str(&format!(
                "{name}: {}（目标 {target}，{}）\n",
                fmt_rate(rate),
                fmt_status(status)
            ));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(actual: Option<f64>, expected: f64) {
        let v = actual.expect("rate should have data");
        assert!((v - expected).abs() < 1e-9, "expected {expected}, got {v}");
    }

    fn populated() -> Metrics {
        let mut m = Metrics::new();
        for _ in 0..8 {
            m.record(MetricEvent::IntentServed { deterministic: true });
        }
        for _ in 0..2 {
            m.record(MetricEvent::IntentServed { deterministic: false });
        }
        for _ in 0..7 {
            m.record(MetricEvent::IntentCompleted { clarifications: 0 });
        }
        for _ in 0..3 {
            m.record(MetricEvent::IntentCompleted { clarifications: 2 });
        }
        for _ in 0..19 {
            m.record(MetricEvent::Interruption { worthwhile: true });
        }
        m.record(MetricEvent::Interruption { worthwhile: false });
        for _ in 0..100 {
            m.record(MetricEvent::ActionExecuted);
        }
        m.record(MetricEvent::ActionUndone);
        for _ in 0..9 {
            m.record(MetricEvent::ModelRequest { resident_hit: true });
        }
        m.record(MetricEvent::ModelRequest { resident_hit: false });
        for _ in 0..999 {
            m.record(MetricEvent::PrivilegeAttempt { denied: true });
        }
        m.record(MetricEvent::PrivilegeAttempt { denied: false });
        m
    }

    #[test]
    fn empty_dashboard_reports_no_data_not_zero() {
        let d = Metrics::new().snapshot();
        assert_eq!(d.determination_rate, None);
        assert_eq!(d.first_try_rate, None);
        assert_eq!(d.false_interrupt_rate, None);
        assert_eq!(d.revocation_rate, None);
        assert_eq!(d.residency_hit_rate, None);
        assert_eq!(d.privilege_interception_rate, None);
        assert_eq!(d.determination_status(), TargetStatus::NoData);
        assert!(d.render().contains("无数据"));
    }

    #[test]
    fn rates_match_their_definitions() {
        let d = populated().snapshot();
        approx(d.determination_rate, 0.8);
        approx(d.first_try_rate, 0.7);
        approx(d.false_interrupt_rate, 0.05);
        approx(d.revocation_rate, 0.01);
        approx(d.residency_hit_rate, 0.9);
        approx(d.privilege_interception_rate, 0.999);
    }

    #[test]
    fn target_verdicts_follow_the_ch22_directions() {
        let d = populated().snapshot();
        // Lower-bound targets are met exactly at the bound.
        assert_eq!(d.determination_status(), TargetStatus::Met);
        assert_eq!(d.first_try_status(), TargetStatus::Met);
        assert_eq!(d.residency_status(), TargetStatus::Met);
        assert_eq!(d.privilege_interception_status(), TargetStatus::Met);
        // Upper-bound targets are met at-or-below the bound.
        assert_eq!(d.false_interrupt_status(), TargetStatus::Met);
        assert_eq!(d.revocation_status(), TargetStatus::Met);

        let mut m = Metrics::new();
        m.record(MetricEvent::Interruption { worthwhile: false });
        m.record(MetricEvent::Interruption { worthwhile: true });
        // 50% false interruptions: far above the 5% ceiling.
        assert_eq!(m.snapshot().false_interrupt_status(), TargetStatus::Missed);
    }

    #[test]
    fn first_try_counts_only_zero_clarification_completions() {
        let mut m = Metrics::new();
        m.record(MetricEvent::IntentCompleted { clarifications: 1 });
        m.record(MetricEvent::IntentCompleted { clarifications: 0 });
        approx(m.snapshot().first_try_rate, 0.5);
    }

    fn ledger_entry(kind: LedgerKind) -> LedgerEntry {
        LedgerEntry {
            seq: 1,
            ts_ms: 0,
            actor: "test".to_string(),
            kind,
            tool: None,
            resource: None,
            cap_fingerprint: None,
            txn: None,
            detail: String::new(),
            prev_hash: "genesis".to_string(),
            hash: "hash".to_string(),
        }
    }

    #[test]
    fn ledger_feed_yields_only_the_revocation_rate() {
        let mut entries = Vec::new();
        for _ in 0..4 {
            entries.push(ledger_entry(LedgerKind::Act));
        }
        entries.push(ledger_entry(LedgerKind::Compensation));
        // Transaction-level markers must not double-count the same undo.
        entries.push(ledger_entry(LedgerKind::TxnUndone));
        entries.push(ledger_entry(LedgerKind::TxnAborted));
        // ActDenied mixes privilege, scope, barrier and quota denials:
        // it must not be laundered into an interception rate.
        entries.push(ledger_entry(LedgerKind::ActDenied));
        entries.push(ledger_entry(LedgerKind::IntentSubmitted));

        let mut m = Metrics::new();
        m.absorb_ledger(&entries);
        let d = m.snapshot();
        approx(d.revocation_rate, 0.25);
        assert_eq!(d.determination_rate, None);
        assert_eq!(d.first_try_rate, None);
        assert_eq!(d.false_interrupt_rate, None);
        assert_eq!(d.residency_hit_rate, None);
        assert_eq!(d.privilege_interception_rate, None);
    }

    #[test]
    fn merge_adds_counters() {
        let mut a = Metrics::new();
        a.record(MetricEvent::IntentServed { deterministic: true });
        let mut b = Metrics::new();
        b.record(MetricEvent::IntentServed { deterministic: false });
        a.merge(&b);
        approx(a.snapshot().determination_rate, 0.5);
    }

    #[test]
    fn render_lists_every_metric_with_its_target() {
        let rendered = populated().snapshot().render();
        for name in ["确定化率", "一次到位率", "误中断率", "撤销率", "驻留命中率", "越权拦截率"] {
            assert!(rendered.contains(name), "missing {name} in:\n{rendered}");
        }
        assert!(rendered.contains("80.0%"));
        assert!(rendered.contains("达标"));
    }
}
