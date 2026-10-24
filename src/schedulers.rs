//! Schedulers (doc ch. 9): of the five schedulers, this module implements
//! the three with concrete mechanics here — inference, trust, and the
//! agent-group quotas. (Compute scheduling is inherited from the host OS in
//! the P0 shell; context scheduling lives with memory recall; attention has
//! its own module.)
//!
//! The inference scheduler is built on doc ch. 2's corrected premise:
//! on-device inference is **memory-bandwidth-bound**, so the scheduler
//! minimizes *bytes moved*, not FLOPs:
//!
//! * **Resident-aware routing** — switching models means reloading weights;
//!   at most 2 models may be resident, and a resident model wins over a
//!   marginally stronger non-resident one.
//! * **Prefix reuse** — shared prefixes are prefilled once and their KV is
//!   reused across agents (an LRU prefix cache), like a page cache.
//! * **Admission control** — inference is non-preemptible, so a request
//!   whose estimated latency exceeds the interactive budget is queued or
//!   downgraded to the small model instead of being "preempted later".

use crate::typesys::Risk;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Inference scheduler
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Small,
    Large,
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub weights_bytes: u64,
    pub kv_bytes_per_token: u64,
    pub tier: Tier,
}

impl ModelInfo {
    fn can_serve(&self, needs_large: bool) -> bool {
        !needs_large || self.tier == Tier::Large
    }
}

#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub agent: String,
    pub interactive: bool,
    pub output_tokens: u32,
    pub prefix: String,
    pub prefix_tokens: u32,
    pub needs_large: bool,
    pub latency_budget_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Admission {
    Admitted,
    /// Estimated latency exceeds the interactive budget: wait instead.
    Queued,
    /// Routed to the small resident model to fit the budget.
    Downgraded,
}

#[derive(Debug, Clone)]
pub struct RouteDecision {
    pub model: String,
    pub resident_hit: bool,
    pub prefix_hit_tokens: u32,
    pub bytes_moved: u64,
    pub est_latency_ms: u64,
    pub admission: Admission,
}

struct PrefixEntry {
    tokens: u32,
}

pub struct InferenceScheduler {
    models: Vec<ModelInfo>,
    /// Resident model names, LRU-first order.
    resident: Vec<String>,
    max_resident: usize,
    prefix_cache: HashMap<u64, PrefixEntry>,
    prefix_lru: Vec<u64>,
    prefix_capacity_entries: usize,
    bandwidth_bytes_per_ms: u64,
    pub total_bytes_moved: u64,
    pub resident_hits: u64,
    pub resident_requests: u64,
    pub prefix_hits: u64,
}

impl InferenceScheduler {
    pub fn new(models: Vec<ModelInfo>, bandwidth_bytes_per_ms: u64) -> Self {
        InferenceScheduler {
            models,
            resident: Vec::new(),
            max_resident: 2,
            prefix_cache: HashMap::new(),
            prefix_lru: Vec::new(),
            prefix_capacity_entries: 64,
            bandwidth_bytes_per_ms: bandwidth_bytes_per_ms.max(1),
            total_bytes_moved: 0,
            resident_hits: 0,
            resident_requests: 0,
            prefix_hits: 0,
        }
    }

    pub fn resident_models(&self) -> &[String] {
        &self.resident
    }

    pub fn resident_hit_rate(&self) -> f64 {
        if self.resident_requests == 0 {
            0.0
        } else {
            self.resident_hits as f64 / self.resident_requests as f64
        }
    }

    fn model(&self, name: &str) -> &ModelInfo {
        self.models.iter().find(|m| m.name == name).expect("known model")
    }

    fn touch_resident(&mut self, name: &str) {
        if let Some(pos) = self.resident.iter().position(|n| n == name) {
            let n = self.resident.remove(pos);
            self.resident.push(n);
        }
    }

    /// Load a model into the resident set (evicting the LRU model when the
    /// hard cap of 2 is reached — doc ch. 2, model thrashing defense).
    /// Returns the weight bytes that had to be moved.
    fn load(&mut self, name: &str) -> u64 {
        if self.resident.iter().any(|n| n == name) {
            self.touch_resident(name);
            return 0;
        }
        if self.resident.len() >= self.max_resident {
            self.resident.remove(0);
        }
        self.resident.push(name.to_string());
        self.model(name).weights_bytes
    }

    fn prefix_key(prefix: &str) -> u64 {
        let digest = crate::crypto::sha256(prefix.as_bytes());
        u64::from_be_bytes(digest[..8].try_into().expect("8 bytes"))
    }

    fn lookup_prefix(&mut self, key: u64) -> Option<u32> {
        let tokens = self.prefix_cache.get(&key).map(|e| e.tokens)?;
        if let Some(pos) = self.prefix_lru.iter().position(|k| *k == key) {
            let k = self.prefix_lru.remove(pos);
            self.prefix_lru.push(k);
        }
        Some(tokens)
    }

    fn store_prefix(&mut self, key: u64, tokens: u32) {
        if !self.prefix_cache.contains_key(&key) {
            while self.prefix_cache.len() >= self.prefix_capacity_entries {
                let victim = self.prefix_lru.remove(0);
                self.prefix_cache.remove(&victim);
            }
            self.prefix_lru.push(key);
        }
        self.prefix_cache.insert(key, PrefixEntry { tokens });
    }

    /// Route one inference request. Deterministic accounting model: bytes
    /// moved = weight reload (if any) + KV bytes for uncached prefix tokens
    /// and generated tokens; latency = bytes / bandwidth.
    pub fn route(&mut self, req: &InferenceRequest) -> RouteDecision {
        self.resident_requests += 1;
        // 1. Choose a model: resident-and-able beats stronger-but-cold.
        let resident_choice = self
            .resident
            .iter()
            .rev()
            .find(|name| self.model(name).can_serve(req.needs_large))
            .cloned();
        let preferred_tier = if req.needs_large { Tier::Large } else { Tier::Small };
        let chosen = match resident_choice {
            Some(name) => name,
            None => self
                .models
                .iter()
                .find(|m| m.tier == preferred_tier)
                .or_else(|| self.models.iter().find(|m| m.can_serve(req.needs_large)))
                .map(|m| m.name.clone())
                .expect("at least one model must be able to serve"),
        };
        let was_resident = self.resident.iter().any(|n| *n == chosen);
        if was_resident {
            self.resident_hits += 1;
        }

        let mut decision = self.decide_with(&chosen, was_resident, req);

        // 2. Admission control: interactive requests that cannot fit the
        //    budget are downgraded (if a smaller resident can serve) or
        //    queued. Non-interactive work always runs in spare capacity.
        if req.interactive && decision.est_latency_ms > req.latency_budget_ms {
            let downgrade = self
                .resident
                .iter()
                .rev()
                .find(|name| {
                    let m = self.model(name);
                    m.tier == Tier::Small && m.can_serve(false) && **name != chosen
                })
                .cloned();
            match downgrade {
                Some(small) if !req.needs_large => {
                    // Refund the estimate; only the downgraded run counts.
                    self.total_bytes_moved -= decision.bytes_moved;
                    let mut d = self.decide_with(&small, true, req);
                    d.admission = Admission::Downgraded;
                    decision = d;
                }
                _ => {
                    decision.admission = Admission::Queued;
                    // A queued request has not run: refund its accounting.
                    self.total_bytes_moved -= decision.bytes_moved;
                    return decision;
                }
            }
        }
        decision
    }

    fn decide_with(&mut self, model_name: &str, was_resident: bool, req: &InferenceRequest) -> RouteDecision {
        let load_bytes = self.load(model_name);
        let model = self.model(model_name).clone();
        // Prefix reuse: cached prefix tokens cost no KV movement.
        let key = Self::prefix_key(&req.prefix);
        let hit_tokens = self.lookup_prefix(key).unwrap_or(0).min(req.prefix_tokens);
        if hit_tokens > 0 {
            self.prefix_hits += 1;
        }
        self.store_prefix(key, req.prefix_tokens);
        let uncached_prefix = req.prefix_tokens - hit_tokens;
        let kv_bytes =
            model.kv_bytes_per_token * (uncached_prefix + req.output_tokens) as u64;
        let bytes_moved = load_bytes + kv_bytes;
        let est_latency_ms = bytes_moved / self.bandwidth_bytes_per_ms;
        self.total_bytes_moved += bytes_moved;
        RouteDecision {
            model: model_name.to_string(),
            resident_hit: was_resident,
            prefix_hit_tokens: hit_tokens,
            bytes_moved,
            est_latency_ms,
            admission: Admission::Admitted,
        }
    }
}

// ---------------------------------------------------------------------------
// Trust scheduler: permissions as a gradually granted, revocable resource.
// ---------------------------------------------------------------------------

pub struct TrustScheduler {
    levels: HashMap<String, u8>,
    confirmations: HashMap<String, u32>,
}

impl TrustScheduler {
    pub fn new() -> Self {
        TrustScheduler { levels: HashMap::new(), confirmations: HashMap::new() }
    }

    /// 0 = read-only narrow grants; grows with user confirmations.
    pub fn level(&self, agent: &str) -> u8 {
        self.levels.get(agent).copied().unwrap_or(0)
    }

    /// Each user confirmation is evidence; two confirmations raise the
    /// level by one, capped at 3 (doc ch. 9: progressive grants).
    pub fn record_confirmation(&mut self, agent: &str) -> u8 {
        let n = self.confirmations.entry(agent.to_string()).or_insert(0);
        *n += 1;
        let level = (*n / 2).min(3) as u8;
        self.levels.insert(agent.to_string(), level);
        level
    }

    /// Minimum trust level per plan risk: Low 0, Medium 1, High 2,
    /// Critical 3.
    pub fn authorize(&self, agent: &str, risk: Risk) -> bool {
        let needed = match risk {
            Risk::Low => 0,
            Risk::Medium => 1,
            Risk::High => 2,
            Risk::Critical => 3,
        };
        self.level(agent) >= needed
    }

    /// Task end / incident response: drop the agent back to zero.
    pub fn reset(&mut self, agent: &str) {
        self.levels.insert(agent.to_string(), 0);
        self.confirmations.insert(agent.to_string(), 0);
    }
}

impl Default for TrustScheduler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Agent group quotas: the cgroup analogue with five resource kinds.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Quotas {
    pub token_budget: u64,
    pub bytes_budget: u64,
    pub cap_slots: usize,
    pub side_effect_budget: u32,
    pub attention_budget: u32,
}

impl Quotas {
    pub fn generous() -> Self {
        Quotas {
            token_budget: 1_000_000,
            bytes_budget: 1_000_000_000_000,
            cap_slots: 16,
            side_effect_budget: 100,
            attention_budget: 20,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuotaError {
    Tokens,
    Bytes,
    CapSlots,
    SideEffects,
    Attention,
}

impl std::fmt::Display for QuotaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            QuotaError::Tokens => "token budget",
            QuotaError::Bytes => "bandwidth (bytes-moved) budget",
            QuotaError::CapSlots => "capability slots",
            QuotaError::SideEffects => "side-effect budget",
            QuotaError::Attention => "attention budget",
        };
        write!(f, "quota exceeded: {name}")
    }
}
impl std::error::Error for QuotaError {}

#[derive(Debug, Clone)]
pub struct AgentGroup {
    pub quotas: Quotas,
    pub used_tokens: u64,
    pub used_bytes: u64,
    pub held_caps: usize,
    pub used_side_effects: u32,
    pub used_attention: u32,
}

impl AgentGroup {
    pub fn new(quotas: Quotas) -> Self {
        AgentGroup {
            quotas,
            used_tokens: 0,
            used_bytes: 0,
            held_caps: 0,
            used_side_effects: 0,
            used_attention: 0,
        }
    }

    pub fn try_consume_tokens(&mut self, n: u64) -> Result<(), QuotaError> {
        if self.used_tokens + n > self.quotas.token_budget {
            return Err(QuotaError::Tokens);
        }
        self.used_tokens += n;
        Ok(())
    }

    pub fn try_consume_bytes(&mut self, n: u64) -> Result<(), QuotaError> {
        if self.used_bytes + n > self.quotas.bytes_budget {
            return Err(QuotaError::Bytes);
        }
        self.used_bytes += n;
        Ok(())
    }

    pub fn try_acquire_cap(&mut self) -> Result<(), QuotaError> {
        if self.held_caps >= self.quotas.cap_slots {
            return Err(QuotaError::CapSlots);
        }
        self.held_caps += 1;
        Ok(())
    }

    pub fn release_cap(&mut self) {
        self.held_caps = self.held_caps.saturating_sub(1);
    }

    pub fn try_consume_side_effect(&mut self) -> Result<(), QuotaError> {
        if self.used_side_effects >= self.quotas.side_effect_budget {
            return Err(QuotaError::SideEffects);
        }
        self.used_side_effects += 1;
        Ok(())
    }

    pub fn try_consume_attention(&mut self) -> Result<(), QuotaError> {
        if self.used_attention >= self.quotas.attention_budget {
            return Err(QuotaError::Attention);
        }
        self.used_attention += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn models() -> Vec<ModelInfo> {
        vec![
            ModelInfo { name: "small-2b".into(), weights_bytes: 2_000_000, kv_bytes_per_token: 1_000, tier: Tier::Small },
            ModelInfo { name: "large-30b".into(), weights_bytes: 30_000_000, kv_bytes_per_token: 4_000, tier: Tier::Large },
        ]
    }

    fn request(needs_large: bool, interactive: bool, budget_ms: u64) -> InferenceRequest {
        InferenceRequest {
            agent: "a".into(),
            interactive,
            output_tokens: 100,
            prefix: "system prompt v1".into(),
            prefix_tokens: 500,
            needs_large,
            latency_budget_ms: budget_ms,
        }
    }

    #[test]
    fn first_request_loads_model_and_pays_weight_bytes() {
        // Bandwidth 1,000,000 bytes/ms.
        let mut sched = InferenceScheduler::new(models(), 1_000_000);
        let d = sched.route(&request(false, false, u64::MAX));
        assert_eq!(d.model, "small-2b");
        assert!(!d.resident_hit);
        // 2,000,000 load + (500 prefix + 100 out) * 1,000 KV.
        assert_eq!(d.bytes_moved, 2_600_000);
    }

    #[test]
    fn resident_model_is_preferred_and_avoids_reload() {
        let mut sched = InferenceScheduler::new(models(), 1_000_000);
        sched.route(&request(false, false, u64::MAX));
        let d = sched.route(&InferenceRequest { prefix: "other prefix".into(), ..request(false, false, u64::MAX) });
        assert!(d.resident_hit);
        assert_eq!(d.bytes_moved, 600_000, "no reload: only KV movement remains");
    }

    #[test]
    fn prefix_reuse_skips_kv_for_cached_tokens() {
        let mut sched = InferenceScheduler::new(models(), 1_000_000);
        sched.route(&request(false, false, u64::MAX));
        let d = sched.route(&request(false, false, u64::MAX));
        assert_eq!(d.prefix_hit_tokens, 500);
        assert_eq!(d.bytes_moved, 100_000, "only the 100 output tokens move KV");
        assert_eq!(sched.prefix_hits, 1);
    }

    #[test]
    fn resident_set_is_capped_at_two_with_lru_eviction() {
        let mut many = models();
        many.push(ModelInfo { name: "mid-8b".into(), weights_bytes: 8_000_000, kv_bytes_per_token: 2_000, tier: Tier::Large });
        let mut sched = InferenceScheduler::new(many, 1_000_000);
        sched.route(&request(false, false, u64::MAX)); // small resident
        sched.route(&request(true, false, u64::MAX)); // large resident
        assert_eq!(sched.resident_models().len(), 2);
        // Force-load a third distinct model by requesting large while asking
        // for the mid model through a needs_large route after evicting small:
        // resident-aware routing would reuse large, so evict by direct load.
        sched.load("mid-8b");
        assert_eq!(sched.resident_models().len(), 2);
        assert!(!sched.resident_models().contains(&"small-2b".to_string()), "LRU victim evicted");
    }

    #[test]
    fn interactive_over_budget_is_queued_when_no_downgrade_exists() {
        let mut sched = InferenceScheduler::new(models(), 1_000); // very slow link
        let d = sched.route(&request(true, true, 10));
        assert_eq!(d.admission, Admission::Queued);
        assert_eq!(sched.total_bytes_moved, 0, "queued work has not run");
    }

    #[test]
    fn background_work_is_admitted_regardless_of_budget() {
        let mut sched = InferenceScheduler::new(models(), 1_000);
        let d = sched.route(&request(true, false, 10));
        assert_eq!(d.admission, Admission::Admitted);
    }

    #[test]
    fn trust_levels_grow_with_confirmations_and_gate_risk() {
        let mut trust = TrustScheduler::new();
        assert!(trust.authorize("a", Risk::Low));
        assert!(!trust.authorize("a", Risk::Medium));
        trust.record_confirmation("a");
        trust.record_confirmation("a");
        assert_eq!(trust.level("a"), 1);
        assert!(trust.authorize("a", Risk::Medium));
        assert!(!trust.authorize("a", Risk::High));
        for _ in 0..4 {
            trust.record_confirmation("a");
        }
        assert_eq!(trust.level("a"), 3);
        assert!(trust.authorize("a", Risk::Critical));
        trust.reset("a");
        assert!(!trust.authorize("a", Risk::Medium));
    }

    #[test]
    fn quotas_enforce_all_five_resource_kinds() {
        let mut g = AgentGroup::new(Quotas {
            token_budget: 100,
            bytes_budget: 1_000,
            cap_slots: 1,
            side_effect_budget: 1,
            attention_budget: 1,
        });
        assert!(g.try_consume_tokens(100).is_ok());
        assert_eq!(g.try_consume_tokens(1), Err(QuotaError::Tokens));
        assert!(g.try_consume_bytes(1_000).is_ok());
        assert_eq!(g.try_consume_bytes(1), Err(QuotaError::Bytes));
        assert!(g.try_acquire_cap().is_ok());
        assert_eq!(g.try_acquire_cap(), Err(QuotaError::CapSlots));
        g.release_cap();
        assert!(g.try_acquire_cap().is_ok());
        assert!(g.try_consume_side_effect().is_ok());
        assert_eq!(g.try_consume_side_effect(), Err(QuotaError::SideEffects));
        assert!(g.try_consume_attention().is_ok());
        assert_eq!(g.try_consume_attention(), Err(QuotaError::Attention));
    }
}
