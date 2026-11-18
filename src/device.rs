//! Explicit model budget (doc ch. 2, conclusion three): the device, not
//! the request, decides which model tier may run.
//!
//! The doc's premise is that on-device inference is memory-bandwidth
//! bound, so a system needs an explicit notion of *model budget*: from
//! the device's available memory and bandwidth, determine which model
//! tier the device can run, and route accordingly. The sharp edge is
//! for low-end devices: "8 GB 内存的手机撑不起稳定的端侧 AI 体验" —
//! such a device should not run multi-step planning at all; it should
//! go straight to determinized macros and single-step tool calls.
//!
//! The thresholds below are derived from the doc's own numbers, not
//! invented: a 70B model at 4-bit is ~35 GB of weights, and a phone's
//! memory bandwidth is ~60 GB/s. A High-tier device must be able to
//! hold those weights plus KV cache plus an activation reserve; a
//! Low-tier device is anything under the 8 GB line (or under 10 GB/s
//! of bandwidth, where even reloading a 1–3B model is slow).
//!
//! The budget check is mechanical: a (model, precision) copy fits when
//! `weights + KV cache for the context window + activation reserve`
//! fits in device memory. The inference scheduler consults this module
//! before routing: a model that does not fit first drops precision
//! (8-bit → 4-bit) and then drops to the smallest fitting model, and a
//! Low-tier device never routes to the large model at all — the
//! decision carries `macro_only`, telling the caller to run macros and
//! single-step calls instead of planning.

use crate::schedulers::{ModelInfo, Precision};

const GIB: u64 = 1 << 30;

/// Below this much memory the device is Low-tier: the doc's 8 GB line
/// ("8 GB 内存的手机撑不起稳定的端侧 AI 体验").
pub const LOW_MEMORY_BYTES: u64 = 8 * GIB;

/// Below this bandwidth (10 GB/s) the device is Low-tier even with
/// more memory: reloading even a 1–3B model takes too long.
pub const LOW_BANDWIDTH_BYTES_PER_MS: u64 = 10_000_000;

/// A High-tier device must hold a 70B-class model: ~35 GB of 4-bit
/// weights (doc ch. 2) plus KV cache plus the activation reserve, so
/// the bar is set at 40 GiB.
pub const HIGH_MEMORY_BYTES: u64 = 40 * GIB;

/// High-tier bandwidth: near the doc's ~60 GB/s phone figure.
pub const HIGH_BANDWIDTH_BYTES_PER_MS: u64 = 50_000_000;

/// Memory held back for activations (neither weights nor KV cache)
/// when testing whether a model fits.
pub const ACTIVATION_RESERVE_BYTES: u64 = 512 * (1 << 20);

/// Which model tier a device can run (doc ch. 2, conclusion three).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceTier {
    /// Under the 8 GB / 10 GB/s line: no multi-step planning, no large
    /// model — determinized macros and single-step tool calls only.
    Low,
    /// Small model resident, large model on demand when it fits.
    Mid,
    /// Can hold a 70B-class model plus KV cache: full planning.
    High,
}

impl DeviceTier {
    pub fn label(self) -> &'static str {
        match self {
            DeviceTier::Low => "low",
            DeviceTier::Mid => "mid",
            DeviceTier::High => "high",
        }
    }

    /// Multi-step planning is only allowed above Low. On a Low device
    /// the only execution paths are determinized macros and single-step
    /// tool calls (doc ch. 2: "低端设备……根本不该走多步规划").
    pub fn planning_allowed(self) -> bool {
        self != DeviceTier::Low
    }

    /// How many (model, precision) copies may be resident at once.
    /// The doc caps residency at two overall; a Low device holds only
    /// its one small model.
    pub fn max_resident_models(self) -> usize {
        match self {
            DeviceTier::Low => 1,
            DeviceTier::Mid | DeviceTier::High => 2,
        }
    }
}

/// What the device tier permits the caller to do with a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutePolicy {
    /// Low tier: run determinized macros and single-step tool calls;
    /// never plan multi-step, never load the large model.
    MacroOrSingleStepOnly,
    /// Mid/High tier: full planning with the models that fit.
    FullPlanning,
}

/// A device's memory and bandwidth budget for inference.
#[derive(Debug, Clone, Copy)]
pub struct DeviceProfile {
    /// Usable memory for weights + KV cache + activations, in bytes.
    pub memory_bytes: u64,
    /// Memory bandwidth, in bytes per millisecond.
    pub bandwidth_bytes_per_ms: u64,
    /// Context-window size the KV cache is budgeted for, in tokens.
    pub kv_context_tokens: u32,
}

impl DeviceProfile {
    pub fn new(memory_bytes: u64, bandwidth_bytes_per_ms: u64, kv_context_tokens: u32) -> Self {
        DeviceProfile {
            memory_bytes,
            bandwidth_bytes_per_ms,
            kv_context_tokens,
        }
    }

    /// The device tier: Low below the 8 GB / 10 GB/s line, High only
    /// when both memory and bandwidth clear the 70B-class bar, Mid
    /// otherwise. Either dimension can hold the device back.
    pub fn tier(&self) -> DeviceTier {
        if self.memory_bytes < LOW_MEMORY_BYTES
            || self.bandwidth_bytes_per_ms < LOW_BANDWIDTH_BYTES_PER_MS
        {
            DeviceTier::Low
        } else if self.memory_bytes >= HIGH_MEMORY_BYTES
            && self.bandwidth_bytes_per_ms >= HIGH_BANDWIDTH_BYTES_PER_MS
        {
            DeviceTier::High
        } else {
            DeviceTier::Mid
        }
    }

    pub fn route_policy(&self) -> RoutePolicy {
        match self.tier() {
            DeviceTier::Low => RoutePolicy::MacroOrSingleStepOnly,
            DeviceTier::Mid | DeviceTier::High => RoutePolicy::FullPlanning,
        }
    }

    /// Does this (model, precision) copy fit in device memory?
    /// Weights at the precision, plus KV cache for the full budgeted
    /// context window, plus the activation reserve.
    pub fn fits(&self, model: &ModelInfo, precision: Precision) -> bool {
        let weights = model.weights_bytes_at(precision) as u128;
        let kv = model.kv_bytes_per_token as u128 * self.kv_context_tokens as u128;
        let reserve = ACTIVATION_RESERVE_BYTES as u128;
        weights + kv + reserve <= self.memory_bytes as u128
    }

    /// The models that fit at this precision, cheapest first.
    pub fn allowed_models<'a>(
        &self,
        models: &'a [ModelInfo],
        precision: Precision,
    ) -> Vec<&'a ModelInfo> {
        let mut fitting: Vec<&ModelInfo> = models
            .iter()
            .filter(|m| self.fits(m, precision))
            .collect();
        fitting.sort_by_key(|m| m.weights_bytes_at(precision));
        fitting
    }

    /// The smallest model that fits at this precision, if any.
    pub fn smallest_fitting<'a>(
        &self,
        models: &'a [ModelInfo],
        precision: Precision,
    ) -> Option<&'a ModelInfo> {
        self.allowed_models(models, precision).into_iter().next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schedulers::Tier;

    const GB_S_PER_MS: u64 = 1_000_000; // 1 GB/s in bytes/ms.

    fn small_model() -> ModelInfo {
        ModelInfo {
            name: "small-2b".into(),
            weights_bytes: 2_000_000_000, // 2 GB at 4-bit.
            kv_bytes_per_token: 1_000,
            tier: Tier::Small,
        }
    }

    fn large_model() -> ModelInfo {
        ModelInfo {
            name: "large-70b".into(),
            weights_bytes: 35_000_000_000, // 35 GB at 4-bit (doc ch. 2).
            kv_bytes_per_token: 4_000,
            tier: Tier::Large,
        }
    }

    #[test]
    fn tier_boundaries_follow_the_doc_numbers() {
        // Under 8 GiB: Low, even with fast bandwidth.
        let low = DeviceProfile::new(4 * GIB, 60 * GB_S_PER_MS, 32_768);
        assert_eq!(low.tier(), DeviceTier::Low);
        // Exactly 8 GiB with decent bandwidth: Mid.
        let mid = DeviceProfile::new(8 * GIB, 20 * GB_S_PER_MS, 32_768);
        assert_eq!(mid.tier(), DeviceTier::Mid);
        // 40 GiB but 5 GB/s: Low — the bandwidth gate holds it back
        // on its own, even with ample memory.
        let low_bw = DeviceProfile::new(40 * GIB, 5 * GB_S_PER_MS, 32_768);
        assert_eq!(low_bw.tier(), DeviceTier::Low);
        // 40 GiB but only 20 GB/s: Mid — memory clears the High bar,
        // bandwidth does not.
        let mid_slow = DeviceProfile::new(40 * GIB, 20 * GB_S_PER_MS, 32_768);
        assert_eq!(mid_slow.tier(), DeviceTier::Mid);
        // Both bars cleared: High.
        let high = DeviceProfile::new(48 * GIB, 60 * GB_S_PER_MS, 32_768);
        assert_eq!(high.tier(), DeviceTier::High);
        // 8 GiB but under 10 GB/s: Low on bandwidth alone.
        let low_bw = DeviceProfile::new(16 * GIB, 5 * GB_S_PER_MS, 32_768);
        assert_eq!(low_bw.tier(), DeviceTier::Low);
    }

    #[test]
    fn low_tier_forbids_planning_and_holds_one_resident() {
        assert!(!DeviceTier::Low.planning_allowed());
        assert!(DeviceTier::Mid.planning_allowed());
        assert!(DeviceTier::High.planning_allowed());
        assert_eq!(DeviceTier::Low.max_resident_models(), 1);
        assert_eq!(DeviceTier::Mid.max_resident_models(), 2);
        assert_eq!(DeviceTier::High.max_resident_models(), 2);
    }

    #[test]
    fn route_policy_matches_tier() {
        let low = DeviceProfile::new(4 * GIB, 60 * GB_S_PER_MS, 32_768);
        assert_eq!(low.route_policy(), RoutePolicy::MacroOrSingleStepOnly);
        let mid = DeviceProfile::new(16 * GIB, 30 * GB_S_PER_MS, 32_768);
        assert_eq!(mid.route_policy(), RoutePolicy::FullPlanning);
    }

    #[test]
    fn fits_counts_weights_kv_and_activation_reserve() {
        let profile = DeviceProfile::new(8 * GIB, 60 * GB_S_PER_MS, 32_768);
        // Small: 2 GB weights + 32,768 * 1,000 KV + 512 MiB reserve ≈ 2.5 GiB.
        assert!(profile.fits(&small_model(), Precision::FourBit));
        // Small at 8-bit: 4 GB weights, still fits 8 GiB.
        assert!(profile.fits(&small_model(), Precision::EightBit));
        // Large at 4-bit: 35 GB weights alone exceed 8 GiB.
        assert!(!profile.fits(&large_model(), Precision::FourBit));
        // Large fits a 48 GiB device at 4-bit, not at 8-bit (70 GB).
        let big = DeviceProfile::new(48 * GIB, 60 * GB_S_PER_MS, 32_768);
        assert!(big.fits(&large_model(), Precision::FourBit));
        assert!(!big.fits(&large_model(), Precision::EightBit));
    }

    #[test]
    fn allowed_models_filters_and_sorts_by_size() {
        let profile = DeviceProfile::new(8 * GIB, 60 * GB_S_PER_MS, 32_768);
        let models = vec![large_model(), small_model()];
        let allowed = profile.allowed_models(&models, Precision::FourBit);
        assert_eq!(allowed.len(), 1);
        assert_eq!(allowed[0].name, "small-2b");
        assert_eq!(
            profile.smallest_fitting(&models, Precision::FourBit).unwrap().name,
            "small-2b"
        );
        let big = DeviceProfile::new(48 * GIB, 60 * GB_S_PER_MS, 32_768);
        let allowed = big.allowed_models(&models, Precision::FourBit);
        assert_eq!(allowed.len(), 2);
        assert_eq!(allowed[0].name, "small-2b");
        assert_eq!(allowed[1].name, "large-70b");
    }
}
