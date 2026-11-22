//! The attention scheduler (doc ch. 14): human attention is the scarcest
//! resource in the system (R4), so interruptions are congestion-controlled
//! exactly the way TCP controls packets — the scheduler cannot know the
//! user's true capacity, so it infers it from the "loss" signals of
//! rejections and ignores:
//!
//! * **Slow start** — a new agent starts with a tiny allowance that doubles
//!   per acknowledged interruption until it hits a threshold.
//! * **AIMD** — additive increase while the user keeps responding;
//!   multiplicative decrease (halve) on rejection or ignore.
//! * **RTT** — the latency from interruption to response is itself a
//!   congestion signal: a slow response means the user is busy, so the
//!   scheduler slows down proactively instead of waiting for a rejection
//!   or an ignore. Each response updates a per-agent RTT EWMA (exponentially
//!   weighted moving average); a response slower than the slow-response
//!   threshold is a multiplicative decrease, a timely one is the usual
//!   additive increase.
//! * **Pricing** — interruptions are charged against the agent's allowance
//!   by priority, so agents learn to coalesce confirmations and prefer
//!   "act safely, report later" over asking.
//! * **Priorities P0–P3** — only P0 (irreversible + high-risk) may interrupt
//!   immediately and override a silence period; P1 queues for a natural
//!   pause, P2 folds into a digest, P3 only writes the ledger.
//! * **Coalescing** — repeated confirmation requests for the same intent
//!   merge into one pending interruption.
//! * **Silence periods** — after a rejection, requests of the same category
//!   are default-denied for a while instead of nagging.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Priority {
    /// Irreversible + high-risk (transfer, send, delete): interrupt now.
    P0,
    /// Reversible but significant (bulk edits): queue for a natural pause.
    P1,
    /// Progress reports, low-risk confirmations: merge into a digest.
    P2,
    /// Pure information: ledger only, never interrupt.
    P3,
}

impl Priority {
    /// The price of an interruption, charged to the agent's allowance.
    /// P0 is expensive — deliberately: only things that truly matter should
    /// be worth paying it.
    pub fn price(self) -> f64 {
        match self {
            Priority::P0 => 5.0,
            Priority::P1 => 2.0,
            Priority::P2 => 1.0,
            Priority::P3 => 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum AttentionDecision {
    /// Show it right now (P0, or a P1 the user is clearly waiting on).
    Immediate,
    /// Queued for the next natural pause point (P1).
    Queued,
    /// Merged into an already-pending request for the same intent.
    Coalesced { pending_count: u32 },
    /// Folded into the periodic digest (P2).
    Digest,
    /// Recorded only (P3).
    LedgerOnly,
    /// Same category was recently rejected: default-deny during silence.
    Silenced,
    /// The agent cannot afford this interruption right now.
    DeniedBudget,
}

#[derive(Debug, Clone)]
pub struct InterruptRequest {
    pub agent: String,
    pub intent_group: String,
    pub category: String,
    pub priority: Priority,
}

#[derive(Debug, Clone)]
struct AgentState {
    allowance: f64,
    ssthresh: f64,
    slow_start: bool,
    charged_total: f64,
    /// EWMA of the user's response latency (ms) for this agent's
    /// interruptions; `None` until the first timed response arrives.
    rtt_ewma_ms: Option<f64>,
}

pub struct AttentionScheduler {
    agents: std::collections::HashMap<String, AgentState>,
    pending: std::collections::HashMap<(String, String), u32>,
    silenced_until: std::collections::HashMap<(String, String), u64>,
    digest: Vec<String>,
    initial_allowance: f64,
    max_allowance: f64,
    silence_ms: u64,
    /// A response slower than this (ms) means the user is busy: slow down
    /// proactively instead of waiting for a rejection or an ignore.
    slow_rtt_ms: u64,
}

impl AttentionScheduler {
    pub fn new() -> Self {
        AttentionScheduler {
            agents: Default::default(),
            pending: Default::default(),
            silenced_until: Default::default(),
            digest: Vec::new(),
            initial_allowance: 1.0,
            max_allowance: 16.0,
            silence_ms: 600_000,
            slow_rtt_ms: 60_000,
        }
    }

    pub fn with_silence_ms(mut self, ms: u64) -> Self {
        self.silence_ms = ms;
        self
    }

    /// Responses slower than `ms` count as "the user is busy" and trigger a
    /// multiplicative decrease instead of an additive increase.
    pub fn with_slow_rtt_ms(mut self, ms: u64) -> Self {
        self.slow_rtt_ms = ms;
        self
    }

    /// The scheduler's current slow-response threshold (ms).
    pub fn slow_rtt_ms(&self) -> u64 {
        self.slow_rtt_ms
    }

    /// The agent's current RTT EWMA (ms), if any timed response arrived.
    pub fn rtt_ewma_ms(&mut self, agent: &str) -> Option<f64> {
        self.state(agent).rtt_ewma_ms
    }

    fn state(&mut self, agent: &str) -> &mut AgentState {
        let initial = self.initial_allowance;
        self.agents.entry(agent.to_string()).or_insert(AgentState {
            allowance: initial,
            ssthresh: 8.0,
            slow_start: true,
            charged_total: 0.0,
            rtt_ewma_ms: None,
        })
    }

    pub fn allowance_of(&mut self, agent: &str) -> f64 {
        self.state(agent).allowance
    }

    pub fn pending_count(&self, agent: &str, intent_group: &str) -> u32 {
        self.pending
            .get(&(agent.to_string(), intent_group.to_string()))
            .copied()
            .unwrap_or(0)
    }

    /// Ask permission to interrupt the user. The scheduler — not the agent —
    /// decides what the interruption is worth and whether it happens.
    pub fn request(&mut self, req: &InterruptRequest, now_ms: u64) -> AttentionDecision {
        // P3 never interrupts anyone.
        if req.priority == Priority::P3 {
            return AttentionDecision::LedgerOnly;
        }
        // Silence: a recently rejected category stays denied — except P0.
        if req.priority != Priority::P0 {
            let until = self
                .silenced_until
                .get(&(req.agent.clone(), req.category.clone()))
                .copied()
                .unwrap_or(0);
            if now_ms < until {
                return AttentionDecision::Silenced;
            }
        }
        match req.priority {
            Priority::P0 => {
                // Always immediate; charged anyway (may overdraw — the debt
                // slows the agent's future non-urgent interruptions).
                let st = self.state(&req.agent);
                st.allowance -= Priority::P0.price();
                st.charged_total += Priority::P0.price();
                AttentionDecision::Immediate
            }
            Priority::P1 => {
                let key = (req.agent.clone(), req.intent_group.clone());
                if let Some(count) = self.pending.get_mut(&key) {
                    // Coalescing: merging is free — this is the incentive.
                    *count += 1;
                    return AttentionDecision::Coalesced { pending_count: *count };
                }
                let st = self.state(&req.agent);
                if st.allowance >= Priority::P1.price() {
                    st.allowance -= Priority::P1.price();
                    st.charged_total += Priority::P1.price();
                    self.pending.insert(key, 1);
                    AttentionDecision::Queued
                } else {
                    AttentionDecision::DeniedBudget
                }
            }
            Priority::P2 => {
                self.digest.push(format!("{}:{}", req.agent, req.intent_group));
                AttentionDecision::Digest
            }
            Priority::P3 => AttentionDecision::LedgerOnly,
        }
    }

    /// The pending interruption was presented and consumed.
    pub fn drain_pending(&mut self, agent: &str, intent_group: &str) -> u32 {
        self.pending
            .remove(&(agent.to_string(), intent_group.to_string()))
            .unwrap_or(0)
    }

    pub fn flush_digest(&mut self) -> Vec<String> {
        std::mem::take(&mut self.digest)
    }

    /// Multiplicative decrease (TCP congestion avoidance): halve the
    /// allowance, drop the slow-start threshold to the new level, and leave
    /// slow start. Shared by rejection, ignore, and slow-response feedback.
    fn multiplicative_decrease(&mut self, agent: &str) {
        let st = self.state(agent);
        st.allowance /= 2.0;
        st.ssthresh = st.allowance.max(1.0);
        st.slow_start = false;
    }

    /// The additive-increase half of `on_responded`: slow start doubles per
    /// response until `ssthresh`, afterwards growth is linear. A negative
    /// allowance is debt from a P0 overdraw — see `on_responded`.
    fn additive_increase(&mut self, agent: &str) {
        let max = self.max_allowance;
        let initial = self.initial_allowance;
        let st = self.state(agent);
        if st.allowance < 0.0 {
            st.allowance += 1.0;
            if st.allowance >= 0.0 {
                st.allowance = initial;
            }
        } else if st.slow_start {
            st.allowance = (st.allowance * 2.0).min(max);
            if st.allowance >= st.ssthresh {
                st.slow_start = false;
            }
        } else {
            st.allowance = (st.allowance + 1.0).min(max);
        }
    }

    /// Feedback: the user responded to an interruption. Additive increase —
    /// in slow start the allowance doubles per response until `ssthresh`,
    /// afterwards it grows linearly (TCP congestion avoidance).
    ///
    /// A negative allowance is debt from a P0 overdraw. Doubling it would
    /// punish the agent for the one interruption it was allowed to make,
    /// and doubling zero would strand it at zero — so a response while in
    /// debt retires one unit of debt instead, and once the debt clears
    /// slow start restarts from the initial allowance.
    ///
    /// This is the untimed form: the response is assumed timely. Use
    /// `on_responded_with_latency` when the response latency is known.
    pub fn on_responded(&mut self, agent: &str) {
        self.additive_increase(agent);
    }

    /// Feedback: the user responded after `rtt_ms` milliseconds. The
    /// latency itself is a congestion signal (doc ch. 14's RTT analogy):
    /// a slow response means the user is busy, so the scheduler slows down
    /// proactively — multiplicative decrease, no waiting for a rejection or
    /// an ignore. A timely response is the usual additive increase.
    /// Either way the per-agent RTT EWMA is updated (alpha 0.25).
    ///
    /// A slow response while the agent is in P0 debt changes nothing: the
    /// debt is neither deepened nor retired — only timely responses retire
    /// it, so a busy user never pays for the agent's one urgent
    /// interruption twice over.
    pub fn on_responded_with_latency(&mut self, agent: &str, rtt_ms: u64) {
        let in_debt = {
            let st = self.state(agent);
            st.rtt_ewma_ms = Some(match st.rtt_ewma_ms {
                None => rtt_ms as f64,
                Some(ewma) => 0.75 * ewma + 0.25 * rtt_ms as f64,
            });
            st.allowance < 0.0
        };
        if rtt_ms > self.slow_rtt_ms {
            if in_debt {
                return; // in P0 debt: a slow response neither deepens nor retires it
            }
            self.multiplicative_decrease(agent);
        } else {
            self.additive_increase(agent);
        }
    }

    /// Feedback: the user rejected an interruption of `category`.
    /// Multiplicative decrease + the category goes silent for a while
    /// (during silence, same-category requests default to denied).
    pub fn on_rejected(&mut self, agent: &str, category: &str, now_ms: u64) {
        self.multiplicative_decrease(agent);
        self.silenced_until
            .insert((agent.to_string(), category.to_string()), now_ms + self.silence_ms);
    }

    /// Feedback: the user ignored an interruption (an RTT/loss signal: they
    /// are busy). Multiplicative decrease, no silence.
    pub fn on_ignored(&mut self, agent: &str) {
        self.multiplicative_decrease(agent);
    }
}

impl Default for AttentionScheduler {
    fn default() -> Self {
        Self::new()
    }
}

/// Max-min fair allocation of a shared attention budget across agents:
/// everyone is guaranteed an equal share first; agents wanting less free
/// their surplus to the rest (water-filling). No talkative agent can starve
/// the others (doc ch. 14).
pub fn max_min_allocate(total: f64, demands: &[(String, f64)]) -> Vec<(String, f64)> {
    let mut result: Vec<(String, f64)> = demands.iter().map(|(a, _)| (a.clone(), 0.0)).collect();
    let mut remaining = total;
    let mut unsatisfied: Vec<usize> = (0..demands.len()).collect();
    while !unsatisfied.is_empty() && remaining > 0.0 {
        let share = remaining / unsatisfied.len() as f64;
        let mut still = Vec::new();
        let mut progressed = false;
        for &i in &unsatisfied {
            let want = demands[i].1 - result[i].1;
            if want <= share {
                result[i].1 += want;
                remaining -= want;
                progressed = true;
            } else {
                still.push(i);
            }
        }
        if !progressed {
            // Everyone wants more than the equal share: split evenly.
            for &i in &still {
                result[i].1 += share;
            }
            break;
        }
        unsatisfied = still;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(agent: &str, group: &str, category: &str, p: Priority) -> InterruptRequest {
        InterruptRequest {
            agent: agent.into(),
            intent_group: group.into(),
            category: category.into(),
            priority: p,
        }
    }

    #[test]
    fn new_agent_starts_in_slow_start_with_tiny_allowance() {
        let mut sched = AttentionScheduler::new();
        assert_eq!(sched.allowance_of("rookie"), 1.0);
        // One P1 costs 2.0 — a brand-new agent cannot afford to interrupt.
        let d = sched.request(&req("rookie", "g", "confirm", Priority::P1), 0);
        assert_eq!(d, AttentionDecision::DeniedBudget);
    }

    #[test]
    fn slow_start_doubles_allowance_per_response() {
        let mut sched = AttentionScheduler::new();
        sched.on_responded("a");
        assert_eq!(sched.allowance_of("a"), 2.0);
        sched.on_responded("a");
        assert_eq!(sched.allowance_of("a"), 4.0);
        sched.on_responded("a");
        assert_eq!(sched.allowance_of("a"), 8.0);
    }

    #[test]
    fn after_slow_start_growth_is_additive() {
        let mut sched = AttentionScheduler::new();
        for _ in 0..3 {
            sched.on_responded("a"); // reaches ssthresh 8
        }
        sched.on_responded("a");
        assert_eq!(sched.allowance_of("a"), 9.0);
        sched.on_responded("a");
        assert_eq!(sched.allowance_of("a"), 10.0);
    }

    #[test]
    fn rejection_halves_allowance_and_silences_the_category() {
        let mut sched = AttentionScheduler::new().with_silence_ms(60_000);
        for _ in 0..3 {
            sched.on_responded("a");
        }
        assert_eq!(sched.allowance_of("a"), 8.0);
        sched.on_rejected("a", "confirm", 1_000);
        assert_eq!(sched.allowance_of("a"), 4.0);
        // Same category during silence: default-deny.
        let d = sched.request(&req("a", "g", "confirm", Priority::P1), 30_000);
        assert_eq!(d, AttentionDecision::Silenced);
        // A different category is unaffected.
        let d2 = sched.request(&req("a", "g", "progress", Priority::P1), 30_000);
        assert_eq!(d2, AttentionDecision::Queued);
        // After the silence expires, the category may ask again.
        let d3 = sched.request(&req("a", "g2", "confirm", Priority::P1), 61_001);
        assert_eq!(d3, AttentionDecision::Queued);
    }

    #[test]
    fn ignore_also_halves_allowance() {
        let mut sched = AttentionScheduler::new();
        for _ in 0..3 {
            sched.on_responded("a");
        }
        sched.on_ignored("a");
        assert_eq!(sched.allowance_of("a"), 4.0);
    }

    #[test]
    fn p0_interrupts_immediately_and_overrides_silence() {
        let mut sched = AttentionScheduler::new().with_silence_ms(60_000);
        sched.on_rejected("a", "transfer-confirm", 0);
        let d = sched.request(&req("a", "g", "transfer-confirm", Priority::P0), 1_000);
        assert_eq!(d, AttentionDecision::Immediate);
    }

    #[test]
    fn p1_requests_for_the_same_intent_coalesce_for_free() {
        let mut sched = AttentionScheduler::new();
        for _ in 0..3 {
            sched.on_responded("a"); // allowance 8
        }
        let d1 = sched.request(&req("a", "trip", "confirm", Priority::P1), 0);
        assert_eq!(d1, AttentionDecision::Queued);
        assert_eq!(sched.allowance_of("a"), 6.0, "first request pays the price");
        let d2 = sched.request(&req("a", "trip", "confirm", Priority::P1), 0);
        assert_eq!(d2, AttentionDecision::Coalesced { pending_count: 2 });
        assert_eq!(sched.allowance_of("a"), 6.0, "merging is free");
        assert_eq!(sched.drain_pending("a", "trip"), 2);
        assert_eq!(sched.pending_count("a", "trip"), 0);
    }

    #[test]
    fn p2_goes_to_digest_and_p3_only_to_ledger() {
        let mut sched = AttentionScheduler::new();
        assert_eq!(sched.request(&req("a", "g", "progress", Priority::P2), 0), AttentionDecision::Digest);
        assert_eq!(sched.request(&req("a", "g", "info", Priority::P3), 0), AttentionDecision::LedgerOnly);
        assert_eq!(sched.flush_digest(), vec!["a:g".to_string()]);
        assert!(sched.flush_digest().is_empty());
    }

    #[test]
    fn response_while_in_p0_debt_retires_the_debt_instead_of_doubling_it() {
        let mut sched = AttentionScheduler::new();
        // Fresh agent spends its one P0: 1.0 - 5.0 = -4.0 of debt.
        let d = sched.request(&req("a", "g", "transfer", Priority::P0), 0);
        assert_eq!(d, AttentionDecision::Immediate);
        assert_eq!(sched.allowance_of("a"), -4.0);
        // Each acknowledged interruption retires one unit of debt.
        sched.on_responded("a");
        assert_eq!(sched.allowance_of("a"), -3.0);
        sched.on_responded("a");
        sched.on_responded("a");
        sched.on_responded("a");
        // Debt cleared: slow start restarts from the initial allowance
        // instead of doubling zero and stranding the agent there.
        assert_eq!(sched.allowance_of("a"), 1.0);
        sched.on_responded("a");
        assert_eq!(sched.allowance_of("a"), 2.0);
    }

    #[test]
    fn p0_debt_does_not_block_later_p1_forever() {
        let mut sched = AttentionScheduler::new();
        sched.request(&req("a", "g", "transfer", Priority::P0), 0);
        for _ in 0..4 {
            sched.on_responded("a"); // retire the -4.0 debt
        }
        sched.on_responded("a"); // 1.0 -> 2.0, slow start
        let d = sched.request(&req("a", "g2", "confirm", Priority::P1), 0);
        assert_eq!(d, AttentionDecision::Queued);
    }

    #[test]
    fn max_min_gives_everyone_a_floor_first() {
        let alloc = max_min_allocate(
            10.0,
            &[
                ("greedy".into(), 9.0),
                ("modest".into(), 2.0),
                ("quiet".into(), 2.0),
            ],
        );
        let get = |n: &str| alloc.iter().find(|(a, _)| a == n).unwrap().1;
        assert_eq!(get("modest"), 2.0);
        assert_eq!(get("quiet"), 2.0);
        assert_eq!(get("greedy"), 6.0, "surplus flows to the unsatisfied");
    }

    #[test]
    fn max_min_equal_split_when_all_hungry() {
        let alloc = max_min_allocate(9.0, &[
            ("a".into(), 100.0),
            ("b".into(), 100.0),
            ("c".into(), 100.0),
        ]);
        assert!(alloc.iter().all(|(_, v)| *v == 3.0));
    }

    #[test]
    fn slow_response_halves_allowance_instead_of_increasing() {
        let mut sched = AttentionScheduler::new();
        for _ in 0..3 {
            sched.on_responded("a"); // allowance 8, slow start over
        }
        // Default slow threshold is 60_000 ms: a 2-minute response is slow.
        sched.on_responded_with_latency("a", 120_000);
        assert_eq!(sched.allowance_of("a"), 4.0);
        // Slow start is exited: the next timely response adds, not doubles.
        sched.on_responded_with_latency("a", 1_000);
        assert_eq!(sched.allowance_of("a"), 5.0);
    }

    #[test]
    fn timely_response_via_latency_path_still_increases() {
        let mut sched = AttentionScheduler::new();
        sched.on_responded_with_latency("a", 500);
        assert_eq!(sched.allowance_of("a"), 2.0);
        sched.on_responded_with_latency("a", 2_000);
        assert_eq!(sched.allowance_of("a"), 4.0);
    }

    #[test]
    fn response_at_the_slow_threshold_counts_as_timely() {
        let mut sched = AttentionScheduler::new().with_slow_rtt_ms(5_000);
        sched.on_responded_with_latency("a", 5_000);
        assert_eq!(sched.allowance_of("a"), 2.0, "exactly at threshold: timely");
        sched.on_responded_with_latency("a", 5_001);
        assert_eq!(sched.allowance_of("a"), 1.0, "one ms over: slow, halved");
    }

    #[test]
    fn rtt_ewma_tracks_response_latency() {
        let mut sched = AttentionScheduler::new();
        assert_eq!(sched.rtt_ewma_ms("a"), None);
        sched.on_responded_with_latency("a", 1_000);
        assert_eq!(sched.rtt_ewma_ms("a"), Some(1_000.0));
        sched.on_responded_with_latency("a", 3_000);
        // alpha 0.25: 0.75 * 1000 + 0.25 * 3000 = 1500.
        assert_eq!(sched.rtt_ewma_ms("a"), Some(1_500.0));
        sched.on_responded_with_latency("a", 1_500);
        assert_eq!(sched.rtt_ewma_ms("a"), Some(1_500.0));
    }

    #[test]
    fn slow_response_during_p0_debt_changes_nothing() {
        let mut sched = AttentionScheduler::new();
        sched.request(&req("a", "g", "transfer", Priority::P0), 0);
        assert_eq!(sched.allowance_of("a"), -4.0);
        // A slow response neither deepens nor retires the debt.
        sched.on_responded_with_latency("a", 120_000);
        assert_eq!(sched.allowance_of("a"), -4.0);
        // Only timely responses retire it, one unit at a time.
        sched.on_responded_with_latency("a", 500);
        assert_eq!(sched.allowance_of("a"), -3.0);
    }

    #[test]
    fn repeated_slow_responses_keep_halving_toward_the_floor() {
        let mut sched = AttentionScheduler::new().with_slow_rtt_ms(5_000);
        for _ in 0..3 {
            sched.on_responded("a"); // allowance 8
        }
        sched.on_responded_with_latency("a", 60_000);
        assert_eq!(sched.allowance_of("a"), 4.0);
        sched.on_responded_with_latency("a", 60_000);
        assert_eq!(sched.allowance_of("a"), 2.0);
        // A busy user never gets charged for the slowdown itself.
        assert_eq!(sched.rtt_ewma_ms("a"), Some(60_000.0));
    }
}
