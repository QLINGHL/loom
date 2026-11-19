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
//! * **Dynamic precision** — one model keeps two weight copies, 4-bit and
//!   8-bit; difficulty picks the precision (routine work runs 4-bit,
//!   demanding reasoning runs 8-bit), so precision is a scheduling
//!   variable, not a model property. Residency is keyed by
//!   (model, precision): the 8-bit copy of an already-resident 4-bit
//!   model still pays a weight reload. An interactive request over
//!   budget at 8-bit first drops to 4-bit before it drops to the small
//!   model or queues.
//! * **Bytes per useful token** — the ch. 2 headline metric: total bytes
//!   moved divided by tokens actually generated (queued work moves
//!   nothing and generates nothing), split by precision so the cost of
//!   running demanding work at 8-bit is visible.
//! * **Block-boundary yield** — a long background decode is split into
//!   blocks of N tokens; at each block boundary the scheduler checks
//!   whether an interactive (higher-priority) request is pending and, if
//!   so, yields without consuming any token of the next block. A block
//!   once started always runs to completion: there is no mid-block
//!   preemption, because the NPU cannot be swapped out mid-inference.

use crate::device::{DeviceProfile, DeviceTier};
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

/// Weight precision of one resident copy of a model (doc ch. 2,
/// conclusion one: dynamic precision). The same model keeps a 4-bit
/// copy for routine work and an 8-bit copy for demanding reasoning;
/// precision is a scheduling variable, not a model property.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Precision {
    FourBit,
    EightBit,
}

impl Precision {
    /// Weight bytes at 8-bit are twice the 4-bit copy (same parameter
    /// count, twice the bits per weight).
    pub fn weight_multiplier(self) -> u64 {
        match self {
            Precision::FourBit => 1,
            Precision::EightBit => 2,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Precision::FourBit => "4bit",
            Precision::EightBit => "8bit",
        }
    }
}

/// How hard the reasoning is. Difficulty — not the model — picks the
/// precision: routine work runs at 4-bit, demanding reasoning at 8-bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Difficulty {
    Routine,
    Demanding,
}

impl Difficulty {
    pub fn precision(self) -> Precision {
        match self {
            Difficulty::Routine => Precision::FourBit,
            Difficulty::Demanding => Precision::EightBit,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    /// Weight bytes of the 4-bit copy; the 8-bit copy is twice that.
    pub weights_bytes: u64,
    pub kv_bytes_per_token: u64,
    pub tier: Tier,
}

impl ModelInfo {
    fn can_serve(&self, needs_large: bool) -> bool {
        !needs_large || self.tier == Tier::Large
    }

    /// Weight bytes of the copy at `precision`. KV bytes per token are
    /// held constant in this prototype: only the weight copy is
    /// precision-switched (doc ch. 2 keeps the two copies per model).
    pub fn weights_bytes_at(&self, precision: Precision) -> u64 {
        self.weights_bytes * precision.weight_multiplier()
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
    pub difficulty: Difficulty,
    pub latency_budget_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Admission {
    Admitted,
    /// Estimated latency exceeds the interactive budget: wait instead.
    Queued,
    /// Routed to the small resident model to fit the budget.
    Downgraded,
    /// Ran at 4-bit instead of the requested 8-bit to fit the
    /// interactive budget. The model did not change, only the precision.
    PrecisionReduced,
}

#[derive(Debug, Clone)]
pub struct RouteDecision {
    pub model: String,
    pub precision: Precision,
    pub resident_hit: bool,
    pub prefix_hit_tokens: u32,
    pub bytes_moved: u64,
    pub est_latency_ms: u64,
    pub admission: Admission,
    /// The device budget forced a smaller model or a lower precision
    /// than the request asked for (doc ch. 2, conclusion three).
    pub device_limited: bool,
    /// Low-tier device: the caller must run determinized macros and
    /// single-step tool calls instead of multi-step planning.
    pub macro_only: bool,
}

struct PrefixEntry {
    tokens: u32,
}

pub struct InferenceScheduler {
    models: Vec<ModelInfo>,
    /// Resident (model, precision) copies, LRU-first order. One model
    /// may hold two slots at once — its 4-bit and 8-bit copies are
    /// separate residents, each paying its own weight reload.
    resident: Vec<(String, Precision)>,
    max_resident: usize,
    prefix_cache: HashMap<u64, PrefixEntry>,
    prefix_lru: Vec<u64>,
    prefix_capacity_entries: usize,
    bandwidth_bytes_per_ms: u64,
    /// Device budget (doc ch. 2, conclusion three), when set: the tier
    /// gates which models may run and whether multi-step planning is
    /// allowed at all. Unset means "no device constraint known".
    device: Option<DeviceProfile>,
    pub total_bytes_moved: u64,
    /// Bytes moved by runs at 4-bit precision (weight reload + KV).
    /// `bytes_moved_4bit + bytes_moved_8bit == total_bytes_moved`.
    pub bytes_moved_4bit: u64,
    /// Bytes moved by runs at 8-bit precision (weight reload + KV).
    pub bytes_moved_8bit: u64,
    /// Output tokens actually generated by admitted runs. Queued work
    /// generates none, so this is the "useful token" denominator.
    pub useful_tokens: u64,
    /// Times an interactive request was run at 4-bit instead of 8-bit
    /// to fit its latency budget.
    pub precision_reductions: u64,
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
            device: None,
            total_bytes_moved: 0,
            bytes_moved_4bit: 0,
            bytes_moved_8bit: 0,
            useful_tokens: 0,
            precision_reductions: 0,
            resident_hits: 0,
            resident_requests: 0,
            prefix_hits: 0,
            blocks_run: 0,
            block_yields: 0,
        }
    }

    /// Attach the device budget (doc ch. 2, conclusion three). The
    /// resident cap follows the tier: a Low device holds only its one
    /// small model, Mid/High keep the two-copy cap.
    pub fn set_device_profile(&mut self, profile: DeviceProfile) {
        self.max_resident = profile.tier().max_resident_models();
        self.device = Some(profile);
    }

    pub fn device_profile(&self) -> Option<&DeviceProfile> {
        self.device.as_ref()
    }

    /// Distinct resident model names (a model holding both precision
    /// copies still appears once).
    pub fn resident_models(&self) -> Vec<String> {
        let mut names: Vec<String> = Vec::new();
        for (name, _) in &self.resident {
            if !names.contains(name) {
                names.push(name.clone());
            }
        }
        names
    }

    /// Resident (model, precision) copies, LRU-first order.
    pub fn resident_copies(&self) -> &[(String, Precision)] {
        &self.resident
    }

    /// The ch. 2 headline metric: bytes moved per useful token
    /// generated. 0.0 until the first token is generated; queued work
    /// moves no bytes and generates no tokens, so it cannot inflate
    /// or deflate the figure.
    pub fn bytes_per_useful_token(&self) -> f64 {
        if self.useful_tokens == 0 {
            0.0
        } else {
            self.total_bytes_moved as f64 / self.useful_tokens as f64
        }
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

    fn touch_resident(&mut self, name: &str, precision: Precision) {
        if let Some(pos) = self
            .resident
            .iter()
            .position(|(n, p)| n == name && *p == precision)
        {
            let entry = self.resident.remove(pos);
            self.resident.push(entry);
        }
    }

    /// Load one (model, precision) copy into the resident set
    /// (evicting the LRU copy when the hard cap of 2 is reached — doc
    /// ch. 2, model thrashing defense). Returns the weight bytes that
    /// had to be moved: 0 when that exact copy is already resident,
    /// the full copy size otherwise — even when the same model is
    /// resident at the other precision.
    fn load(&mut self, name: &str, precision: Precision) -> u64 {
        if self.resident.iter().any(|(n, p)| n == name && *p == precision) {
            self.touch_resident(name, precision);
            return 0;
        }
        if self.resident.len() >= self.max_resident {
            self.resident.remove(0);
        }
        self.resident.push((name.to_string(), precision));
        self.model(name).weights_bytes_at(precision)
    }

    /// Book one admitted run's movement: total, its precision bucket,
    /// and the useful-token denominator.
    fn account(&mut self, decision: &RouteDecision, output_tokens: u32) {
        self.total_bytes_moved += decision.bytes_moved;
        match decision.precision {
            Precision::FourBit => self.bytes_moved_4bit += decision.bytes_moved,
            Precision::EightBit => self.bytes_moved_8bit += decision.bytes_moved,
        }
        self.useful_tokens += output_tokens as u64;
    }

    /// Undo `account` for a run that did not actually happen (queued,
    /// or replaced by a downgraded run). Resident-set and prefix-cache
    /// side effects are deliberately left in place: a queued request
    /// still warms them, as before.
    fn refund(&mut self, decision: &RouteDecision, output_tokens: u32) {
        self.total_bytes_moved -= decision.bytes_moved;
        match decision.precision {
            Precision::FourBit => self.bytes_moved_4bit -= decision.bytes_moved,
            Precision::EightBit => self.bytes_moved_8bit -= decision.bytes_moved,
        }
        self.useful_tokens -= output_tokens as u64;
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
    /// and generated tokens; latency = bytes / bandwidth. Precision is
    /// picked from the request's difficulty, not from the model.
    pub fn route(&mut self, req: &InferenceRequest) -> RouteDecision {
        self.resident_requests += 1;
        // 0. Device budget (doc ch. 2, conclusion three): a Low-tier
        //    device never runs the large model and never plans
        //    multi-step — the decision tells the caller to run macros
        //    and single-step calls instead.
        let (needs_large, mut device_limited, macro_only) = match &self.device {
            Some(profile) => {
                let low = profile.tier() == DeviceTier::Low;
                (
                    req.needs_large && !low,
                    req.needs_large && low,
                    !profile.tier().planning_allowed(),
                )
            }
            None => (req.needs_large, false, false),
        };
        let mut precision = req.difficulty.precision();
        // 1. Choose a model: a resident (model, precision) copy that can
        // serve beats a stronger-but-cold model. A copy resident at the
        // other precision does not count: it still needs its own reload.
        let resident_choice = self
            .resident
            .iter()
            .rev()
            .find(|(name, p)| *p == precision && self.model(name).can_serve(needs_large))
            .map(|(name, _)| name.clone());
        let preferred_tier = if needs_large { Tier::Large } else { Tier::Small };
        let mut chosen = match resident_choice {
            Some(name) => name,
            None => self
                .models
                .iter()
                .find(|m| m.tier == preferred_tier)
                .or_else(|| self.models.iter().find(|m| m.can_serve(needs_large)))
                .map(|m| m.name.clone())
                .expect("at least one model must be able to serve"),
        };
        // 1b. Memory fit: a model that does not fit the device first
        //     drops precision (8-bit -> 4-bit), then drops to the
        //     smallest model that fits.
        if let Some(profile) = &self.device {
            if !profile.fits(self.model(&chosen), precision) {
                if profile.fits(self.model(&chosen), Precision::FourBit) {
                    precision = Precision::FourBit;
                    device_limited = true;
                } else if let Some(fallback) =
                    profile.smallest_fitting(&self.models, Precision::FourBit)
                {
                    chosen = fallback.name.clone();
                    precision = Precision::FourBit;
                    device_limited = true;
                } else {
                    // Nothing fits: still route to the smallest model so
                    // the caller gets a decision it can report, flagged.
                    device_limited = true;
                    if let Some(smallest) = self
                        .models
                        .iter()
                        .min_by_key(|m| m.weights_bytes_at(Precision::FourBit))
                    {
                        chosen = smallest.name.clone();
                        precision = Precision::FourBit;
                    }
                }
            }
        }
        let was_resident = self
            .resident
            .iter()
            .any(|(n, p)| n == &chosen && *p == precision);
        if was_resident {
            self.resident_hits += 1;
        }

        let mut decision =
            self.decide_with(&chosen, precision, was_resident, req, device_limited, macro_only);

        // 2. Admission control: interactive requests that cannot fit the
        //    budget degrade in order — first precision (8-bit -> 4-bit,
        //    same model), then model (small resident model), then queue.
        //    Non-interactive work always runs in spare capacity.
        if req.interactive && decision.est_latency_ms > req.latency_budget_ms {
            // 2a. Precision is a scheduling variable: drop to 4-bit and
            //     re-estimate before giving up on this model.
            if decision.precision == Precision::EightBit {
                let was_resident_4bit = self
                    .resident
                    .iter()
                    .any(|(n, p)| n == &chosen && *p == Precision::FourBit);
                self.refund(&decision, req.output_tokens);
                let reduced = self.decide_with(
                    &chosen,
                    Precision::FourBit,
                    was_resident_4bit,
                    req,
                    device_limited,
                    macro_only,
                );
                if reduced.est_latency_ms <= req.latency_budget_ms {
                    self.precision_reductions += 1;
                    let mut d = reduced;
                    d.admission = Admission::PrecisionReduced;
                    return d;
                }
                decision = reduced;
            }
            // 2b. Model downgrade (only when the request does not need
            //     the large tier): a resident small model at the current
            //     precision, so no weight reload is paid.
            let downgrade = self
                .resident
                .iter()
                .rev()
                .find(|(name, p)| {
                    let m = self.model(name);
                    *p == decision.precision
                        && m.tier == Tier::Small
                        && m.can_serve(false)
                        && name != &chosen
                })
                .map(|(name, _)| name.clone());
            match downgrade {
                Some(small) if !needs_large => {
                    // Refund the estimate; only the downgraded run counts.
                    self.refund(&decision, req.output_tokens);
                    let mut d = self.decide_with(
                        &small,
                        decision.precision,
                        true,
                        req,
                        device_limited,
                        macro_only,
                    );
                    d.admission = Admission::Downgraded;
                    decision = d;
                }
                _ => {
                    decision.admission = Admission::Queued;
                    // A queued request has not run: refund its accounting.
                    self.refund(&decision, req.output_tokens);
                    return decision;
                }
            }
        }
        decision
    }

    fn decide_with(
        &mut self,
        model_name: &str,
        precision: Precision,
        was_resident: bool,
        req: &InferenceRequest,
        device_limited: bool,
        macro_only: bool,
    ) -> RouteDecision {
        let load_bytes = self.load(model_name, precision);
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
        let decision = RouteDecision {
            model: model_name.to_string(),
            precision,
            resident_hit: was_resident,
            prefix_hit_tokens: hit_tokens,
            bytes_moved,
            est_latency_ms,
            admission: Admission::Admitted,
            device_limited,
            macro_only,
        };
        self.account(&decision, req.output_tokens);
        decision
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
// Reserved real-time channel (doc ch. 2, conclusion two, third leg)
// ---------------------------------------------------------------------------

/// Outcome of asking the real-time channel whether a request may run
/// in the current window. Every outcome that defers or rejects a
/// request consumes nothing, so the request may be retried unchanged
/// (next window) or after chunking / downgrading it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelDecision {
    /// Interactive work, running inside its reserved slice.
    Reserved,
    /// Interactive work, reserved slice exhausted but idle spare
    /// capacity available: runs now, borrowed from the spare slice.
    Borrowed,
    /// Background work, running inside the spare slice.
    Spare,
    /// Interactive work, both slices full: wait for the next window.
    WaitNextWindow,
    /// Interactive work larger than the whole window: it can never
    /// fit one window even by borrowing the entire spare slice. The
    /// caller must chunk it (doc ch. 9 block-boundary decode) or
    /// downgrade it before retrying.
    TooBigForSlice,
    /// Background work, spare slice exhausted: deferred to the next
    /// window.
    Deferred,
    /// Background work larger than the whole spare slice: it can never
    /// fit one window and must be chunked before retrying.
    TooBigForSpare,
}

/// The reserved real-time channel: the third leg of doc ch. 2,
/// conclusion two. Inference is non-preemptible — a background decode
/// already three seconds in cannot be yanked out to serve an
/// interactive request — so the scheduler cannot rely on preemption
/// to protect latency. Admission control (`route`) and block-boundary
/// yield (`run_block`) cover the other two legs; this is the third:
/// every scheduling window is split into a reserved slice for
/// interactive work and a spare slice that background work may fill.
/// Background work never touches the reserved slice, so an
/// interactive request is never starved by background load;
/// interactive work that finds its slice exhausted may borrow idle
/// spare capacity, and the borrow is accounted so the books stay
/// honest.
///
/// The channel is a pure time-budget gate: the caller estimates a
/// request's latency (e.g. from `RouteDecision::est_latency_ms`) and
/// asks the channel before starting the decode. A `WaitNextWindow`
/// for an interactive request is the signal that feeds
/// `run_block`'s `interactive_pending` flag.
#[derive(Debug, Clone)]
pub struct RealtimeChannel {
    window_ms: u64,
    reserved_ms: u64,
    reserved_used_ms: u64,
    spare_used_ms: u64,
    /// Interactive milliseconds served from the spare slice this window.
    borrowed_ms: u64,
    windows: u64,
    interactive_waits: u64,
    background_deferrals: u64,
    oversized_interactive: u64,
    oversized_background: u64,
    total_borrowed_ms: u64,
}

impl RealtimeChannel {
    /// `window_ms` is one scheduling window; `reserved_ms` of it is
    /// reserved for interactive work, the rest is spare capacity for
    /// background work. Degenerate inputs are clamped, not rejected:
    /// a window below 1 ms becomes 1 ms, a reserved slice outside
    /// `1..=window_ms` is clamped into it, so the channel always has
    /// both a reserved slice and (unless the whole window is
    /// reserved) a spare slice.
    pub fn new(window_ms: u64, reserved_ms: u64) -> Self {
        let window_ms = window_ms.max(1);
        let reserved_ms = reserved_ms.clamp(1, window_ms);
        RealtimeChannel {
            window_ms,
            reserved_ms,
            reserved_used_ms: 0,
            spare_used_ms: 0,
            borrowed_ms: 0,
            windows: 0,
            interactive_waits: 0,
            background_deferrals: 0,
            oversized_interactive: 0,
            oversized_background: 0,
            total_borrowed_ms: 0,
        }
    }

    pub fn window_ms(&self) -> u64 {
        self.window_ms
    }

    pub fn reserved_ms(&self) -> u64 {
        self.reserved_ms
    }

    /// Spare capacity per window: what background work may fill.
    pub fn spare_ms(&self) -> u64 {
        self.window_ms - self.reserved_ms
    }

    pub fn reserved_used_ms(&self) -> u64 {
        self.reserved_used_ms
    }

    pub fn spare_used_ms(&self) -> u64 {
        self.spare_used_ms
    }

    /// Interactive milliseconds borrowed from the spare slice in the
    /// current window.
    pub fn borrowed_ms(&self) -> u64 {
        self.borrowed_ms
    }

    /// Windows completed (each `advance_window` counts one).
    pub fn windows(&self) -> u64 {
        self.windows
    }

    pub fn interactive_waits(&self) -> u64 {
        self.interactive_waits
    }

    pub fn background_deferrals(&self) -> u64 {
        self.background_deferrals
    }

    pub fn oversized_interactive(&self) -> u64 {
        self.oversized_interactive
    }

    pub fn oversized_background(&self) -> u64 {
        self.oversized_background
    }

    /// Interactive milliseconds served from spare capacity, all windows.
    pub fn total_borrowed_ms(&self) -> u64 {
        self.total_borrowed_ms
    }

    /// Fraction of the reserved slice already consumed this window.
    pub fn reserved_utilization(&self) -> f64 {
        self.reserved_used_ms as f64 / self.reserved_ms as f64
    }

    /// Fraction of the spare slice already consumed this window
    /// (reserved + borrowed). 0.0 when the whole window is reserved.
    pub fn spare_utilization(&self) -> f64 {
        if self.spare_ms() == 0 {
            0.0
        } else {
            self.spare_used_ms as f64 / self.spare_ms() as f64
        }
    }

    /// Dispatch on the request's interactive flag.
    pub fn admit(&mut self, interactive: bool, est_ms: u64) -> ChannelDecision {
        if interactive {
            self.admit_interactive(est_ms)
        } else {
            self.admit_background(est_ms)
        }
    }

    /// Interactive admission, in order: the reserved slice first,
    /// then the remainder borrowed from idle spare capacity (a request
    /// may span both — it runs in the reserved slice and borrows the
    /// rest), then wait for the next window. A request larger than the
    /// whole window is rejected outright — it must be chunked or
    /// downgraded, not waited on, because no future window will fit
    /// it either.
    pub fn admit_interactive(&mut self, est_ms: u64) -> ChannelDecision {
        if est_ms > self.window_ms {
            self.oversized_interactive += 1;
            return ChannelDecision::TooBigForSlice;
        }
        let reserved_remaining = self.reserved_ms - self.reserved_used_ms;
        if est_ms <= reserved_remaining {
            self.reserved_used_ms += est_ms;
            return ChannelDecision::Reserved;
        }
        let spare_remaining = self.spare_ms() - self.spare_used_ms;
        if est_ms <= reserved_remaining + spare_remaining {
            let borrowed = est_ms - reserved_remaining;
            self.reserved_used_ms = self.reserved_ms;
            self.spare_used_ms += borrowed;
            self.borrowed_ms += borrowed;
            self.total_borrowed_ms += borrowed;
            return ChannelDecision::Borrowed;
        }
        self.interactive_waits += 1;
        ChannelDecision::WaitNextWindow
    }

    /// Background admission: the spare slice only, never the reserved
    /// one. A request larger than the whole spare slice is rejected
    /// outright — it must be chunked, because no future window will
    /// fit it either.
    pub fn admit_background(&mut self, est_ms: u64) -> ChannelDecision {
        if est_ms > self.spare_ms() {
            self.oversized_background += 1;
            return ChannelDecision::TooBigForSpare;
        }
        if self.spare_used_ms + est_ms <= self.spare_ms() {
            self.spare_used_ms += est_ms;
            return ChannelDecision::Spare;
        }
        self.background_deferrals += 1;
        ChannelDecision::Deferred
    }

    /// Close the current window and open the next: per-window budgets
    /// reset, cumulative counters (waits, deferrals, borrows, windows)
    /// are kept.
    pub fn advance_window(&mut self) {
        self.reserved_used_ms = 0;
        self.spare_used_ms = 0;
        self.borrowed_ms = 0;
        self.windows += 1;
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
        request_difficulty(needs_large, interactive, budget_ms, Difficulty::Routine)
    }

    fn request_difficulty(
        needs_large: bool,
        interactive: bool,
        budget_ms: u64,
        difficulty: Difficulty,
    ) -> InferenceRequest {
        InferenceRequest {
            agent: "a".into(),
            interactive,
            output_tokens: 100,
            prefix: "system prompt v1".into(),
            prefix_tokens: 500,
            needs_large,
            difficulty,
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
        sched.load("mid-8b", Precision::EightBit);
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
        chunked_request_difficulty(output_tokens, interactive, Difficulty::Routine)
    }

    fn chunked_request_difficulty(
        output_tokens: u32,
        interactive: bool,
        difficulty: Difficulty,
    ) -> InferenceRequest {
        InferenceRequest {
            output_tokens,
            difficulty,
            ..request(false, interactive, u64::MAX)
        }
    }

    #[test]
    fn demanding_request_runs_at_8bit_and_pays_double_weights() {
        let mut sched = InferenceScheduler::new(models(), 1_000_000);
        let d = sched.route(&request_difficulty(false, false, u64::MAX, Difficulty::Demanding));
        assert_eq!(d.model, "small-2b");
        assert_eq!(d.precision, Precision::EightBit);
        // 2 x 2,000,000 weights (8-bit copy) + (500 prefix + 100 out) * 1,000 KV.
        assert_eq!(d.bytes_moved, 4_600_000);
        assert_eq!(sched.bytes_moved_8bit, 4_600_000);
        assert_eq!(sched.bytes_moved_4bit, 0);
        assert_eq!(sched.useful_tokens, 100);
        // Routine work on the same model pays the 4-bit copy instead.
        // A different prefix keeps the KV cost visible (no prefix hit).
        let d = sched.route(&InferenceRequest {
            prefix: "another prompt".into(),
            ..request(false, false, u64::MAX)
        });
        assert_eq!(d.precision, Precision::FourBit);
        assert_eq!(d.bytes_moved, 2_600_000);
        assert_eq!(sched.bytes_moved_4bit, 2_600_000);
    }

    #[test]
    fn same_model_at_other_precision_still_pays_a_reload() {
        let mut sched = InferenceScheduler::new(models(), 1_000_000);
        sched.route(&request(false, false, u64::MAX)); // (small, 4-bit) resident
        // A different prefix keeps the KV cost visible (no prefix hit).
        let d = sched.route(&InferenceRequest {
            prefix: "another prompt".into(),
            ..request_difficulty(false, false, u64::MAX, Difficulty::Demanding)
        });
        assert!(!d.resident_hit, "the 4-bit copy does not stand in for the 8-bit copy");
        assert_eq!(d.bytes_moved, 4_600_000, "full 8-bit weight reload");
        assert_eq!(sched.resident_models(), vec!["small-2b".to_string()], "one model, two copies");
        assert_eq!(sched.resident_copies().len(), 2);
        assert!(sched.resident_copies().contains(&("small-2b".to_string(), Precision::FourBit)));
        assert!(sched.resident_copies().contains(&("small-2b".to_string(), Precision::EightBit)));
    }

    #[test]
    fn interactive_demanding_over_budget_drops_precision_before_model_or_queue() {
        let mut sched = InferenceScheduler::new(models(), 1_000_000);
        // 8-bit: 4,600,000 bytes = 4.6 ms; 4-bit: the 8-bit attempt already
        // warmed the prefix cache, so 2,000,000 load + 100,000 KV = 2.1 ms.
        let d = sched.route(&request_difficulty(false, true, 3, Difficulty::Demanding));
        assert_eq!(d.admission, Admission::PrecisionReduced);
        assert_eq!(d.model, "small-2b", "same model, only the precision changed");
        assert_eq!(d.precision, Precision::FourBit);
        assert_eq!(d.bytes_moved, 2_100_000);
        assert_eq!(sched.precision_reductions, 1);
        assert_eq!(sched.bytes_moved_4bit, 2_100_000);
        assert_eq!(sched.bytes_moved_8bit, 0, "the 8-bit estimate was refunded");
        assert_eq!(sched.useful_tokens, 100, "only the run that happened counts");
    }

    #[test]
    fn precision_reduction_that_still_misses_budget_queues() {
        let mut sched = InferenceScheduler::new(models(), 1_000_000);
        // Even 4-bit (2.1 ms after the prefix was warmed) misses a 1 ms
        // budget; no other small model exists to downgrade to, so the
        // request queues.
        let d = sched.route(&request_difficulty(false, true, 1, Difficulty::Demanding));
        assert_eq!(d.admission, Admission::Queued);
        assert_eq!(sched.total_bytes_moved, 0);
        assert_eq!(sched.useful_tokens, 0);
        assert_eq!(sched.precision_reductions, 0, "a queued run is not a reduction");
    }

    #[test]
    fn bytes_per_useful_token_counts_only_admitted_runs() {
        let mut sched = InferenceScheduler::new(models(), 1_000_000);
        assert_eq!(sched.bytes_per_useful_token(), 0.0, "no tokens yet");
        sched.route(&request(false, false, u64::MAX)); // 2,600,000 bytes, 100 tokens
        assert_eq!(sched.bytes_per_useful_token(), 26_000.0);
        // A queued interactive request moves nothing and generates nothing.
        let d = sched.route(&request(true, true, 10));
        assert_eq!(d.admission, Admission::Queued);
        assert_eq!(sched.bytes_per_useful_token(), 26_000.0);
        assert_eq!(sched.useful_tokens, 100);
        sched.route(&InferenceRequest {
            prefix: "another prompt".into(),
            ..request_difficulty(false, false, u64::MAX, Difficulty::Demanding)
        });
        // (2,600,000 + 4,600,000) / 200 tokens.
        assert_eq!(sched.bytes_per_useful_token(), 36_000.0);
        assert_eq!(
            sched.bytes_moved_4bit + sched.bytes_moved_8bit,
            sched.total_bytes_moved,
            "precision buckets partition the total"
        );
    }

    #[test]
    fn chunked_demanding_job_runs_at_8bit_and_resumes_without_reload() {
        let mut sched = InferenceScheduler::new(models(), 1_000_000);
        let mut job = sched.start_chunked(&chunked_request_difficulty(200, false, Difficulty::Demanding));
        assert!(matches!(
            sched.run_block(&mut job, 100, false),
            BlockOutcome::Advanced { generated: 100, remaining: 100 }
        ));
        // First block: 2 x 2,000,000 weights + (500 + 100) x 1,000 KV.
        assert_eq!(sched.total_bytes_moved, 4_600_000);
        assert_eq!(sched.bytes_moved_8bit, 4_600_000);
        assert!(sched.resident_copies().contains(&("small-2b".to_string(), Precision::EightBit)));
        // Second block: only the 100 new output tokens' KV moves.
        assert!(matches!(
            sched.run_block(&mut job, 100, false),
            BlockOutcome::Completed { generated: 200 }
        ));
        assert_eq!(sched.total_bytes_moved, 4_700_000);
        assert_eq!(sched.useful_tokens, 200);
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

    // --- Device budget (doc ch. 2, conclusion three) -----------------

    fn realistic_models() -> Vec<ModelInfo> {
        vec![
            ModelInfo { name: "small-2b".into(), weights_bytes: 2_000_000_000, kv_bytes_per_token: 1_000, tier: Tier::Small },
            ModelInfo { name: "large-70b".into(), weights_bytes: 35_000_000_000, kv_bytes_per_token: 4_000, tier: Tier::Large },
        ]
    }

    const GB_S_PER_MS: u64 = 1_000_000; // 1 GB/s in bytes/ms.
    const GIB: u64 = 1 << 30;

    #[test]
    fn low_tier_device_never_runs_large_and_is_macro_only() {
        let mut sched = InferenceScheduler::new(realistic_models(), 60 * GB_S_PER_MS);
        // 4 GiB phone: Low tier.
        sched.set_device_profile(DeviceProfile::new(4 * GIB, 60 * GB_S_PER_MS, 32_768));
        let d = sched.route(&request(true, false, u64::MAX));
        assert_eq!(d.model, "small-2b", "low device must not load the large model");
        assert!(d.macro_only, "low device must run macros, not multi-step planning");
        assert!(d.device_limited);
        // Only the small model ever becomes resident on a Low device.
        sched.route(&request(false, false, u64::MAX));
        assert_eq!(sched.resident_models(), vec!["small-2b".to_string()]);
    }

    #[test]
    fn mid_tier_device_falls_back_when_large_model_does_not_fit() {
        let mut sched = InferenceScheduler::new(realistic_models(), 30 * GB_S_PER_MS);
        // 10 GiB: Mid tier, but the 35 GB large model does not fit.
        sched.set_device_profile(DeviceProfile::new(10 * GIB, 30 * GB_S_PER_MS, 32_768));
        let d = sched.route(&request(true, false, u64::MAX));
        assert_eq!(d.model, "small-2b", "unfitting large model falls back to small");
        assert!(d.device_limited);
        assert!(!d.macro_only, "mid tier still allows multi-step planning");
    }

    #[test]
    fn high_tier_device_runs_large_model_unchanged() {
        let mut sched = InferenceScheduler::new(realistic_models(), 60 * GB_S_PER_MS);
        // 48 GiB / 60 GB/s: High tier, large model fits at 4-bit.
        sched.set_device_profile(DeviceProfile::new(48 * GIB, 60 * GB_S_PER_MS, 32_768));
        let d = sched.route(&request(true, false, u64::MAX));
        assert_eq!(d.model, "large-70b");
        assert!(!d.device_limited);
        assert!(!d.macro_only);
    }

    #[test]
    fn no_device_profile_means_no_device_limit() {
        let mut sched = InferenceScheduler::new(models(), 1_000_000);
        let d = sched.route(&request(true, false, u64::MAX));
        assert_eq!(d.model, "large-30b");
        assert!(!d.device_limited);
        assert!(!d.macro_only);
    }

    // --- Reserved real-time channel (doc ch. 2, conclusion two) ---

    /// Window 1000 ms, reserved slice 400 ms, spare slice 600 ms.
    fn channel() -> RealtimeChannel {
        RealtimeChannel::new(1000, 400)
    }

    #[test]
    fn interactive_runs_inside_reserved_slice() {
        let mut ch = channel();
        assert_eq!(ch.admit_interactive(300), ChannelDecision::Reserved);
        assert_eq!(ch.admit_interactive(100), ChannelDecision::Reserved);
        assert_eq!(ch.reserved_used_ms(), 400);
        assert_eq!(ch.reserved_utilization(), 1.0);
        assert_eq!(ch.spare_used_ms(), 0, "interactive must not touch spare while reserved remains");
    }

    #[test]
    fn interactive_borrows_idle_spare_when_reserved_full() {
        let mut ch = channel();
        assert_eq!(ch.admit_interactive(400), ChannelDecision::Reserved);
        assert_eq!(ch.admit_interactive(200), ChannelDecision::Borrowed);
        assert_eq!(ch.borrowed_ms(), 200);
        assert_eq!(ch.total_borrowed_ms(), 200);
        assert_eq!(ch.spare_used_ms(), 200);
        assert_eq!(ch.reserved_used_ms(), 400);
    }

    #[test]
    fn interactive_waits_when_both_slices_full_then_runs_next_window() {
        let mut ch = channel();
        assert_eq!(ch.admit_interactive(400), ChannelDecision::Reserved);
        assert_eq!(ch.admit_interactive(600), ChannelDecision::Borrowed);
        assert_eq!(ch.admit_interactive(1), ChannelDecision::WaitNextWindow);
        assert_eq!(ch.interactive_waits(), 1);
        assert_eq!(ch.reserved_used_ms(), 400, "a wait consumes nothing");
        ch.advance_window();
        assert_eq!(ch.windows(), 1);
        assert_eq!(ch.reserved_used_ms(), 0);
        assert_eq!(ch.spare_used_ms(), 0);
        assert_eq!(ch.borrowed_ms(), 0);
        assert_eq!(ch.interactive_waits(), 1, "cumulative stats survive the window");
        assert_eq!(ch.total_borrowed_ms(), 600, "borrow total survives the window");
        assert_eq!(ch.admit_interactive(400), ChannelDecision::Reserved);
    }

    #[test]
    fn oversized_interactive_is_rejected_not_queued() {
        let mut ch = channel();
        // 1001 ms can never fit the 1000 ms window, even by borrowing
        // the entire spare slice: waiting would never help.
        assert_eq!(ch.admit_interactive(1001), ChannelDecision::TooBigForSlice);
        assert_eq!(ch.oversized_interactive(), 1);
        assert_eq!(ch.reserved_used_ms(), 0);
        assert_eq!(ch.spare_used_ms(), 0);
        assert_eq!(ch.interactive_waits(), 0);
        // Just under the window still fits, spanning reserved + borrow.
        assert_eq!(ch.admit_interactive(1000), ChannelDecision::Borrowed);
        assert_eq!(ch.reserved_used_ms(), 400);
        assert_eq!(ch.borrowed_ms(), 600);
    }

    #[test]
    fn background_only_fills_spare_and_defers_when_full() {
        let mut ch = channel();
        assert_eq!(ch.admit_background(600), ChannelDecision::Spare);
        assert_eq!(ch.admit_background(1), ChannelDecision::Deferred);
        assert_eq!(ch.background_deferrals(), 1);
        assert_eq!(ch.reserved_used_ms(), 0, "background must never touch the reserved slice");
        // The reserved slice is still fully available afterwards.
        assert_eq!(ch.admit_interactive(400), ChannelDecision::Reserved);
    }

    #[test]
    fn background_never_starves_interactive() {
        let mut ch = channel();
        assert_eq!(ch.admit_background(600), ChannelDecision::Spare);
        // Spare is full, but the reserved slice is untouched.
        assert_eq!(ch.admit_interactive(400), ChannelDecision::Reserved);
        // Both slices are now full, so a further interactive request
        // waits — but it was never displaced by background work.
        assert_eq!(ch.admit_interactive(1), ChannelDecision::WaitNextWindow);
    }

    #[test]
    fn oversized_background_must_be_chunked() {
        let mut ch = channel();
        assert_eq!(ch.admit_background(601), ChannelDecision::TooBigForSpare);
        assert_eq!(ch.oversized_background(), 1);
        assert_eq!(ch.spare_used_ms(), 0);
        assert_eq!(ch.background_deferrals(), 0);
    }

    #[test]
    fn zero_cost_request_admits_without_consuming() {
        let mut ch = channel();
        assert_eq!(ch.admit_interactive(0), ChannelDecision::Reserved);
        assert_eq!(ch.admit_background(0), ChannelDecision::Spare);
        assert_eq!(ch.reserved_used_ms(), 0);
        assert_eq!(ch.spare_used_ms(), 0);
    }

    #[test]
    fn admit_dispatches_on_interactive_flag() {
        let mut ch = channel();
        assert_eq!(ch.admit(true, 100), ChannelDecision::Reserved);
        assert_eq!(ch.admit(false, 100), ChannelDecision::Spare);
    }

    #[test]
    fn borrowed_spare_is_unavailable_to_later_background() {
        let mut ch = channel();
        assert_eq!(ch.admit_interactive(400), ChannelDecision::Reserved);
        assert_eq!(ch.admit_interactive(500), ChannelDecision::Borrowed);
        // Only 100 ms of spare remain; a 200 ms background job defers.
        assert_eq!(ch.admit_background(200), ChannelDecision::Deferred);
        assert_eq!(ch.admit_background(100), ChannelDecision::Spare);
    }

    #[test]
    fn degenerate_window_and_reserved_are_clamped() {
        let ch = RealtimeChannel::new(0, 0);
        assert_eq!(ch.window_ms(), 1);
        assert_eq!(ch.reserved_ms(), 1);
        assert_eq!(ch.spare_ms(), 0);
        let ch = RealtimeChannel::new(100, 500);
        assert_eq!(ch.reserved_ms(), 100, "reserved cannot exceed the window");
        assert_eq!(ch.spare_ms(), 0);
        // With no spare slice, background work can never fit.
        let mut ch = RealtimeChannel::new(100, 500);
        assert_eq!(ch.admit_background(1), ChannelDecision::TooBigForSpare);
    }
}
