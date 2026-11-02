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
//! * **Block-boundary yield** — a long background decode is split into
//!   blocks of N tokens; at each block boundary the scheduler checks
//!   whether an interactive (higher-priority) request is pending and, if
//!   so, yields without consuming any token of the next block. A block
//!   once started always runs to completion: there is no mid-block
//!   preemption, because the NPU cannot be swapped out mid-inference.

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
    /// Blocks completed by chunked decodes (see `run_block`).
    pub blocks_run: u64,
    /// Times a chunked decode yielded at a block boundary.
    pub block_yields: u64,
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
            blocks_run: 0,
            block_yields: 0,
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
        let was_resident = self.resident.contains(&chosen);
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
// Chunked decode with block-boundary yield (doc ch. 9)
// ---------------------------------------------------------------------------

/// A long decode split into blocks so a background job can yield to an
/// interactive request. Priority here is exactly the request's
/// `interactive` flag: interactive work holds the reserved real-time
/// channel, background work fills the remainder. A pending request only
/// preempts at a boundary and only when it is strictly higher priority
/// than the running job, so an interactive job never yields and a
/// background job never yields to another background job.
#[derive(Debug, Clone)]
pub struct ChunkedJob {
    request: InferenceRequest,
    /// Tokens already generated; also the resume point after a yield.
    pub generated: u32,
    /// Blocks completed so far (a yielded block is not counted).
    pub blocks: u64,
    /// Times this job has yielded at a boundary.
    pub yields: u64,
    pub finished: bool,
}

/// Outcome of one `run_block` call. `Yielded` consumes nothing: the
/// caller retries the same block later and no token is lost or duplicated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockOutcome {
    /// The block ran to completion; `generated` / `remaining` are post-block.
    Advanced { generated: u32, remaining: u32 },
    /// A higher-priority request was pending at the boundary; the block
    /// did not start. `generated` / `remaining` are unchanged.
    Yielded { generated: u32, remaining: u32 },
    /// The job has generated all its output tokens.
    Completed { generated: u32 },
    /// The block was admission-queued (interactive over budget); like a
    /// yield, it consumed nothing and may be retried later.
    Queued { generated: u32, remaining: u32 },
}

impl InferenceScheduler {
    /// Start a chunked decode. Nothing runs yet: the first block is not a
    /// boundary, so it always runs even if a higher-priority request is
    /// already pending (the scheduler would simply not have started the
    /// job in that case; once started, only boundaries yield).
    pub fn start_chunked(&self, req: &InferenceRequest) -> ChunkedJob {
        ChunkedJob {
            request: req.clone(),
            generated: 0,
            blocks: 0,
            yields: 0,
            finished: req.output_tokens == 0,
        }
    }

    /// Run one block of at most `block_tokens` tokens for `job`.
    ///
    /// Boundary rule, in order: a finished job stays finished; a job
    /// that has not completed any block yet runs its first block
    /// unconditionally; afterwards, if `interactive_pending` is set and
    /// the running job is background work, the block does not start and
    /// the job yields. Each block reuses the existing routing, resident
    /// set and prefix cache, so a resumed job pays no weight reload and
    /// no prefix KV twice.
    pub fn run_block(
        &mut self,
        job: &mut ChunkedJob,
        block_tokens: u32,
        interactive_pending: bool,
    ) -> BlockOutcome {
        if job.finished {
            return BlockOutcome::Completed { generated: job.generated };
        }
        let remaining = job.request.output_tokens - job.generated;
        if job.blocks > 0 && interactive_pending && !job.request.interactive {
            job.yields += 1;
            self.block_yields += 1;
            return BlockOutcome::Yielded { generated: job.generated, remaining };
        }
        let block = block_tokens.max(1).min(remaining);
        let sub = InferenceRequest { output_tokens: block, ..job.request.clone() };
        let decision = self.route(&sub);
        if decision.admission == Admission::Queued {
            return BlockOutcome::Queued { generated: job.generated, remaining };
        }
        job.generated += block;
        job.blocks += 1;
        self.blocks_run += 1;
        if job.generated >= job.request.output_tokens {
            job.finished = true;
            BlockOutcome::Completed { generated: job.generated }
        } else {
            BlockOutcome::Advanced {
                generated: job.generated,
                remaining: job.request.output_tokens - job.generated,
            }
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

// ---------------------------------------------------------------------------
// Context scheduler: semantic page faults (doc ch. 9)
// ---------------------------------------------------------------------------

/// A semantic page fault: a recall that did not land cleanly. Unlike a
/// real MMU fault, a semantic fault is probabilistic — the recall may
/// have returned the wrong thing, or nothing at all — so the fault carries
/// an explicit confidence instead of a guaranteed backing page. The
/// scheduler turns "I don't know" into a schedulable event instead of
/// letting the model fabricate an answer.
#[derive(Debug, Clone)]
pub struct SemanticFault {
    pub query: String,
    /// Explicit page-fault confidence in [0, 1]; NaN is treated as 0.
    pub confidence: f64,
    /// Rephrase attempts already spent on this fault (0 on first sight).
    pub attempt: u32,
}

/// Which tier a confidence value falls in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultLevel {
    /// High enough to use as-is.
    High,
    /// Worth one more try with a different phrasing / supplementary
    /// retrieval, within the retry budget.
    Medium,
    /// Too low to trust: stop and ask the human.
    Low,
}

/// Outcome of handling one semantic page fault. There is deliberately no
/// "fabricate" outcome: every path either uses a trusted recall, retries
/// with a genuinely different query, or yields to the human.
#[derive(Debug, Clone, PartialEq)]
pub enum FaultOutcome {
    /// Confidence high: use the recalled context directly.
    UseDirectly { confidence: f64 },
    /// Confidence medium and budget remains: retry with this query.
    RetryRephrased { attempt: u32, rephrased_query: String },
    /// Confidence low, retries exhausted, or rephrasing made no
    /// progress: stop and ask the human instead of guessing.
    YieldToHuman { question: String, confidence: f64 },
}

pub struct ContextScheduler {
    /// `confidence >= high_threshold` uses the recall directly.
    pub high_threshold: f64,
    /// `medium_threshold <= confidence < high_threshold` retries.
    pub medium_threshold: f64,
    /// Max rephrase attempts per fault before escalating to a yield.
    pub max_retries: u32,
    pub faults_seen: u64,
    pub direct_uses: u64,
    pub retries: u64,
    pub yields: u64,
}

impl ContextScheduler {
    /// Defaults: high 0.8, medium 0.5, at most 2 rephrase attempts.
    pub fn new() -> Self {
        ContextScheduler {
            high_threshold: 0.8,
            medium_threshold: 0.5,
            max_retries: 2,
            faults_seen: 0,
            direct_uses: 0,
            retries: 0,
            yields: 0,
        }
    }

    /// Custom thresholds, clamped to [0, 1]; `medium` is capped at `high`
    /// so the three tiers always partition the range without overlap.
    pub fn with_thresholds(high: f64, medium: f64, max_retries: u32) -> Self {
        let high = if high.is_nan() { 0.8 } else { high.clamp(0.0, 1.0) };
        let medium = if medium.is_nan() { 0.5 } else { medium.clamp(0.0, 1.0) };
        ContextScheduler {
            high_threshold: high,
            medium_threshold: medium.min(high),
            max_retries,
            faults_seen: 0,
            direct_uses: 0,
            retries: 0,
            yields: 0,
        }
    }

    fn sanitize(confidence: f64) -> f64 {
        if confidence.is_nan() { 0.0 } else { confidence.clamp(0.0, 1.0) }
    }

    pub fn classify(&self, confidence: f64) -> FaultLevel {
        let c = Self::sanitize(confidence);
        if c >= self.high_threshold {
            FaultLevel::High
        } else if c >= self.medium_threshold {
            FaultLevel::Medium
        } else {
            FaultLevel::Low
        }
    }

    fn yield_to_human(&mut self, fault: &SemanticFault, confidence: f64) -> FaultOutcome {
        self.yields += 1;
        FaultOutcome::YieldToHuman {
            question: format!(
                "上下文召回置信度过低（{confidence:.2}），无法可靠回答，请确认：{}",
                fault.query
            ),
            confidence,
        }
    }

    /// Handle one semantic page fault.
    ///
    /// `rephrase` maps `(query, next_attempt)` to a new query — the
    /// "换表述重试 / 补充检索" step. It is the caller's job to actually
    /// change the phrasing; if it returns the original query (or an
    /// empty one), the fault escalates to a yield rather than looping
    /// forever on the same recall.
    pub fn resolve(
        &mut self,
        fault: &SemanticFault,
        rephrase: &dyn Fn(&str, u32) -> String,
    ) -> FaultOutcome {
        self.faults_seen += 1;
        let confidence = Self::sanitize(fault.confidence);
        match self.classify(confidence) {
            FaultLevel::High => {
                self.direct_uses += 1;
                FaultOutcome::UseDirectly { confidence }
            }
            FaultLevel::Low => self.yield_to_human(fault, confidence),
            FaultLevel::Medium => {
                if fault.attempt >= self.max_retries {
                    // Still not high after the budget: escalate, never
                    // fabricate from a medium-confidence recall.
                    return self.yield_to_human(fault, confidence);
                }
                let next_attempt = fault.attempt + 1;
                let rephrased = rephrase(&fault.query, next_attempt);
                if rephrased.trim().is_empty() || rephrased == fault.query {
                    return self.yield_to_human(fault, confidence);
                }
                self.retries += 1;
                FaultOutcome::RetryRephrased { attempt: next_attempt, rephrased_query: rephrased }
            }
        }
    }
}

impl Default for ContextScheduler {
    fn default() -> Self {
        Self::new()
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

    fn chunked_request(output_tokens: u32, interactive: bool) -> InferenceRequest {
        InferenceRequest { output_tokens, ..request(false, interactive, u64::MAX) }
    }

    #[test]
    fn background_decode_yields_at_block_boundary_and_resumes_losslessly() {
        let mut sched = InferenceScheduler::new(models(), 1_000_000);
        let mut job = sched.start_chunked(&chunked_request(40, false));
        // First block is not a boundary: it runs even with a pending
        // interactive request.
        assert_eq!(
            sched.run_block(&mut job, 16, true),
            BlockOutcome::Advanced { generated: 16, remaining: 24 }
        );
        // At the boundary the pending interactive request wins: the next
        // block does not start and no token is consumed.
        assert_eq!(
            sched.run_block(&mut job, 16, true),
            BlockOutcome::Yielded { generated: 16, remaining: 24 }
        );
        assert_eq!(job.yields, 1);
        assert_eq!(sched.block_yields, 1);
        // Yielding again costs nothing either.
        assert_eq!(
            sched.run_block(&mut job, 16, true),
            BlockOutcome::Yielded { generated: 16, remaining: 24 }
        );
        // Once the pending request clears, the job resumes from exactly
        // where it yielded and finishes with a partial final block.
        assert_eq!(
            sched.run_block(&mut job, 16, false),
            BlockOutcome::Advanced { generated: 32, remaining: 8 }
        );
        assert_eq!(
            sched.run_block(&mut job, 16, false),
            BlockOutcome::Completed { generated: 40 }
        );
        assert!(job.finished);
        assert_eq!(job.blocks, 3, "16 + 16 + 8, no block lost or duplicated");
        // A finished job stays finished and moves no more bytes.
        let bytes = sched.total_bytes_moved;
        assert_eq!(sched.run_block(&mut job, 16, false), BlockOutcome::Completed { generated: 40 });
        assert_eq!(sched.total_bytes_moved, bytes);
    }

    #[test]
    fn yield_needs_strictly_higher_priority() {
        let mut sched = InferenceScheduler::new(models(), 1_000_000);
        // Background job, no interactive pending: no yield.
        let mut bg = sched.start_chunked(&chunked_request(32, false));
        sched.run_block(&mut bg, 16, false);
        assert_eq!(
            sched.run_block(&mut bg, 16, false),
            BlockOutcome::Completed { generated: 32 }
        );
        // Interactive job never yields, even to an interactive pending
        // request: it holds the reserved real-time channel.
        let mut fg = sched.start_chunked(&chunked_request(32, true));
        sched.run_block(&mut fg, 16, false);
        assert_eq!(
            sched.run_block(&mut fg, 16, true),
            BlockOutcome::Completed { generated: 32 }
        );
        assert_eq!(fg.yields, 0);
    }

    #[test]
    fn resumed_job_pays_no_second_weight_reload_or_prefix() {
        let mut sched = InferenceScheduler::new(models(), 1_000_000);
        let mut job = sched.start_chunked(&chunked_request(200, false));
        sched.run_block(&mut job, 100, false);
        let bytes_after_first = sched.total_bytes_moved;
        // Yield at the boundary, then resume: only the next block's
        // output KV moves, no weight reload and no prefix KV again.
        assert!(matches!(sched.run_block(&mut job, 100, true), BlockOutcome::Yielded { .. }));
        sched.run_block(&mut job, 100, false);
        assert_eq!(sched.total_bytes_moved - bytes_after_first, 100_000, "100 output tokens x 1,000 KV bytes");
    }

    #[test]
    fn zero_token_chunked_job_is_immediately_complete() {
        let mut sched = InferenceScheduler::new(models(), 1_000_000);
        let mut job = sched.start_chunked(&chunked_request(0, false));
        assert_eq!(sched.run_block(&mut job, 16, true), BlockOutcome::Completed { generated: 0 });
        assert_eq!(sched.total_bytes_moved, 0);
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

    fn fault(query: &str, confidence: f64, attempt: u32) -> SemanticFault {
        SemanticFault { query: query.into(), confidence, attempt }
    }

    fn rephrase_suffix(query: &str, attempt: u32) -> String {
        format!("{query}（换表述第{attempt}次）")
    }

    #[test]
    fn high_confidence_fault_is_used_directly() {
        let mut sched = ContextScheduler::new();
        assert_eq!(
            sched.resolve(&fault("用户喜欢靠窗座位吗", 0.9, 0), &rephrase_suffix),
            FaultOutcome::UseDirectly { confidence: 0.9 }
        );
        assert_eq!(sched.direct_uses, 1);
        assert_eq!(sched.faults_seen, 1);
    }

    #[test]
    fn thresholds_partition_the_range_at_the_boundaries() {
        let sched = ContextScheduler::new();
        assert_eq!(sched.classify(0.8), FaultLevel::High, "exactly high uses directly");
        assert_eq!(sched.classify(0.799), FaultLevel::Medium);
        assert_eq!(sched.classify(0.5), FaultLevel::Medium, "exactly medium retries");
        assert_eq!(sched.classify(0.499), FaultLevel::Low);
    }

    #[test]
    fn medium_confidence_retries_with_a_rephrased_query() {
        let mut sched = ContextScheduler::new();
        assert_eq!(
            sched.resolve(&fault("用户喜欢靠窗座位吗", 0.6, 0), &rephrase_suffix),
            FaultOutcome::RetryRephrased {
                attempt: 1,
                rephrased_query: "用户喜欢靠窗座位吗（换表述第1次）".into(),
            }
        );
        assert_eq!(sched.retries, 1);
        // A second medium fault for the same query keeps counting attempts.
        assert_eq!(
            sched.resolve(&fault("用户喜欢靠窗座位吗", 0.6, 1), &rephrase_suffix),
            FaultOutcome::RetryRephrased {
                attempt: 2,
                rephrased_query: "用户喜欢靠窗座位吗（换表述第2次）".into(),
            }
        );
        assert_eq!(sched.retries, 2);
    }

    #[test]
    fn exhausted_retry_budget_escalates_to_yield_never_fabricates() {
        let mut sched = ContextScheduler::new();
        let outcome = sched.resolve(&fault("用户喜欢靠窗座位吗", 0.6, 2), &rephrase_suffix);
        assert!(matches!(outcome, FaultOutcome::YieldToHuman { .. }), "budget spent, still medium -> ask human");
        assert_eq!(sched.yields, 1);
        assert_eq!(sched.retries, 0, "no retry was issued past the budget");
    }

    #[test]
    fn low_confidence_yields_immediately_with_the_query() {
        let mut sched = ContextScheduler::new();
        let outcome = sched.resolve(&fault("用户喜欢靠窗座位吗", 0.2, 0), &rephrase_suffix);
        match outcome {
            FaultOutcome::YieldToHuman { question, confidence } => {
                assert!(question.contains("用户喜欢靠窗座位吗"));
                assert_eq!(confidence, 0.2);
            }
            other => panic!("expected yield, got {other:?}"),
        }
        assert_eq!(sched.yields, 1);
    }

    #[test]
    fn rephrase_that_changes_nothing_yields_instead_of_looping() {
        let mut sched = ContextScheduler::new();
        let identity = |q: &str, _attempt: u32| q.to_string();
        let outcome = sched.resolve(&fault("用户喜欢靠窗座位吗", 0.6, 0), &identity);
        assert!(matches!(outcome, FaultOutcome::YieldToHuman { .. }));
        assert_eq!(sched.retries, 0);
        let empty = |_q: &str, _attempt: u32| "   ".to_string();
        let outcome = sched.resolve(&fault("用户喜欢靠窗座位吗", 0.6, 0), &empty);
        assert!(matches!(outcome, FaultOutcome::YieldToHuman { .. }));
    }

    #[test]
    fn nan_confidence_is_treated_as_zero_and_yields() {
        let mut sched = ContextScheduler::new();
        assert_eq!(sched.classify(f64::NAN), FaultLevel::Low);
        let outcome = sched.resolve(&fault("用户喜欢靠窗座位吗", f64::NAN, 0), &rephrase_suffix);
        assert!(matches!(outcome, FaultOutcome::YieldToHuman { confidence: 0.0, .. }));
    }

    #[test]
    fn full_fault_lifecycle_counts_every_outcome() {
        let mut sched = ContextScheduler::new();
        sched.resolve(&fault("q1", 0.9, 0), &rephrase_suffix); // direct
        sched.resolve(&fault("q2", 0.6, 0), &rephrase_suffix); // retry
        sched.resolve(&fault("q3", 0.1, 0), &rephrase_suffix); // yield
        sched.resolve(&fault("q4", 0.6, 2), &rephrase_suffix); // yield, budget spent
        assert_eq!(sched.faults_seen, 4);
        assert_eq!(sched.direct_uses, 1);
        assert_eq!(sched.retries, 1);
        assert_eq!(sched.yields, 2);
    }
}
