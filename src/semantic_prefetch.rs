//! Semantic speculative prefetch (doc ch. 12, §2).
//!
//! Action-level speculation predicts the *next tool call* and executes it
//! early in a sandbox (see [`crate::speculation`]). Semantic prefetch is
//! the cheaper sibling the OS can do with its global view: when an intent
//! arrives, it predicts *what the next request will need* and stages it
//! in cache before the agent asks:
//!
//! * **Prefetch context** — e.g. the user says "next week in Shanghai",
//!   so frequent-flyer info, the last trip and the travel policy are
//!   loaded into L1 before the agent requests them.
//! * **Pre-resident model** — a long-context step is coming, so the large
//!   model is loaded into memory now instead of paying a reload on the
//!   first call.
//! * **Warm tool** — the calendar will be needed, so its connection is
//!   established now (the week's events themselves arrive as context
//!   items in the same plan).
//!
//! The hard property that makes this safe where action speculation is
//! risky: **prefetch only fills cache, it never produces a result.**
//! `prefetch()` returns a report of keys filled and keys skipped — no
//! values. The staged value is always fetched from the authoritative
//! source by the caller's read-only fetcher, never predicted, so the
//! prediction is only about *which* keys will be needed. A misprediction
//! therefore wastes bandwidth (counted in the report) but cannot produce
//! a wrong result: a real request for another key never sees a staged
//! value, and a real request for the staged key sees the authoritatively
//! fetched value. Staged entries are also isolated from real caches:
//! eviction and invalidation here touch only this staging store.
//!
//! Prefetch still shares the ch. 12 speculation budget gate
//! ([`crate::speculation::speculation_allowed`]): battery below 20% or a
//! thermal trip closes it entirely, and the gate is checked *before* any
//! fetcher runs, so a closed gate wastes no bandwidth at all. One extra
//! guard of its own: a tool is only warmed when it is read-only —
//! warming a tool that sends, pays or deletes would stage it one step
//! closer to an unconfirmed side effect, which is exactly what the
//! non-speculatable barrier forbids.

use crate::speculation::{speculation_allowed, SpecContext, SpecGate};
use crate::Value;

/// Which kind of staging one prefetch item asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefetchKind {
    Context,
    Model,
    Tool,
}

/// One predicted need in a prefetch plan, supplied when an intent arrives.
/// `key` is a context key, a model name, or a tool name depending on
/// `kind`. Only the fields relevant to the kind are consulted.
#[derive(Debug, Clone)]
pub struct PrefetchItem {
    pub kind: PrefetchKind,
    pub key: String,
    /// Context only: how long the staged value stays fresh. `None`
    /// means fresh until evicted or invalidated.
    pub ttl_ms: Option<u64>,
    /// Tool only: warming is refused unless the tool is read-only.
    pub read_only: bool,
    /// Accounting: bytes this prefetch moves (fetch size, or model
    /// weight bytes for a model residency). A misprediction wastes
    /// exactly this much and nothing else.
    pub cost_bytes: u64,
}

impl PrefetchItem {
    pub fn context(key: impl Into<String>, ttl_ms: Option<u64>, cost_bytes: u64) -> Self {
        PrefetchItem {
            kind: PrefetchKind::Context,
            key: key.into(),
            ttl_ms,
            read_only: true,
            cost_bytes,
        }
    }

    pub fn model(name: impl Into<String>, weights_bytes: u64) -> Self {
        PrefetchItem {
            kind: PrefetchKind::Model,
            key: name.into(),
            ttl_ms: None,
            read_only: true,
            cost_bytes: weights_bytes,
        }
    }

    pub fn tool(name: impl Into<String>, read_only: bool) -> Self {
        PrefetchItem {
            kind: PrefetchKind::Tool,
            key: name.into(),
            ttl_ms: None,
            read_only,
            cost_bytes: 0,
        }
    }
}

/// Why a planned item was not staged. Every skip is cheap by
/// construction: a skipped item costs no fetch and no bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipReason {
    /// The system-level speculation gate is closed (battery, thermal,
    /// background, or low tool-time fraction).
    Gate(SpecGate),
    /// Already staged and still fresh: refetching would be pure waste.
    AlreadyPresent,
    /// A tool that is not read-only is never warmed.
    NotReadOnly,
}

impl SkipReason {
    pub fn name(&self) -> &'static str {
        match self {
            SkipReason::Gate(_) => "gate-closed",
            SkipReason::AlreadyPresent => "already-present",
            SkipReason::NotReadOnly => "not-read-only",
        }
    }
}

/// The whole observable outcome of a prefetch run. Deliberately contains
/// no values: prefetch produces cache fills, never results.
#[derive(Debug, Clone, Default)]
pub struct PrefetchReport {
    /// (kind, key) pairs actually staged by this run.
    pub filled: Vec<(PrefetchKind, String)>,
    /// (key, reason) pairs skipped by this run.
    pub skipped: Vec<(String, SkipReason)>,
    /// Bytes moved by this run — the entire cost of any misprediction.
    pub bytes_moved: u64,
    /// Staged entries evicted to make room (prefetch-store entries only;
    /// real caches are never touched).
    pub evicted: Vec<String>,
}

impl PrefetchReport {
    pub fn filled_count(&self) -> usize {
        self.filled.len()
    }
}

#[derive(Debug, Clone)]
struct StagedContext {
    value: Value,
    fetched_ms: u64,
    ttl_ms: Option<u64>,
}

impl StagedContext {
    fn fresh(&self, now_ms: u64) -> bool {
        match self.ttl_ms {
            None => true,
            Some(ttl) => now_ms.saturating_sub(self.fetched_ms) < ttl,
        }
    }
}

/// The staging store: prefetched context (LRU, capped), resident models
/// (LRU, capped at 2 — the ch. 2 model-thrashing defense), and warmed
/// tools. A real request observes staged context only through
/// [`PrefetchStore::context_hit`], for exactly the key it asked for.
pub struct PrefetchStore {
    context: Vec<(String, StagedContext)>,
    max_context_entries: usize,
    /// Resident model names, LRU-first order.
    models: Vec<String>,
    /// Warmed tool names, LRU-first order.
    tools: Vec<String>,
    max_tools: usize,
    pub prefetched: u64,
    pub hits: u64,
    pub misses: u64,
    pub expired: u64,
    pub evicted: u64,
    pub bytes_moved: u64,
}

/// Hard cap on resident models, mirroring the inference scheduler.
pub const MAX_RESIDENT_MODELS: usize = 2;

impl PrefetchStore {
    pub fn new(max_context_entries: usize) -> Self {
        PrefetchStore {
            context: Vec::new(),
            max_context_entries: max_context_entries.max(1),
            models: Vec::new(),
            tools: Vec::new(),
            max_tools: max_context_entries.max(1),
            prefetched: 0,
            hits: 0,
            misses: 0,
            expired: 0,
            evicted: 0,
            bytes_moved: 0,
        }
    }

    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    pub fn staged_context_keys(&self) -> Vec<String> {
        self.context.iter().map(|(k, _)| k.clone()).collect()
    }

    pub fn resident_models(&self) -> &[String] {
        &self.models
    }

    pub fn is_model_resident(&self, name: &str) -> bool {
        self.models.iter().any(|n| n == name)
    }

    pub fn warmed_tools(&self) -> &[String] {
        &self.tools
    }

    pub fn is_tool_warmed(&self, name: &str) -> bool {
        self.tools.iter().any(|n| n == name)
    }

    fn touch_context(&mut self, key: &str) {
        if let Some(pos) = self.context.iter().position(|(k, _)| k == key) {
            let entry = self.context.remove(pos);
            self.context.push(entry);
        }
    }

    fn evict_lru_context(&mut self, report: &mut PrefetchReport) {
        if !self.context.is_empty() {
            let (victim, _) = self.context.remove(0);
            self.evicted += 1;
            report.evicted.push(victim);
        }
    }

    /// Stage a whole prefetch plan. The budget gate runs first: when it
    /// is closed, every item is skipped and the fetcher is never called.
    /// Context items call the fetcher (which must read the authoritative
    /// source — it is never asked to invent a value); model and tool
    /// items stage residency/warmth markers and need no fetch.
    pub fn prefetch(
        &mut self,
        ctx: &SpecContext,
        items: &[PrefetchItem],
        now_ms: u64,
        fetcher: &mut dyn FnMut(&str) -> Value,
    ) -> PrefetchReport {
        let mut report = PrefetchReport::default();
        if let Err(gate) = speculation_allowed(ctx) {
            for item in items {
                report.skipped.push((item.key.clone(), SkipReason::Gate(gate)));
            }
            return report;
        }
        for item in items {
            match item.kind {
                PrefetchKind::Context => {
                    if let Some(pos) = self.context.iter().position(|(k, _)| k == &item.key) {
                        if self.context[pos].1.fresh(now_ms) {
                            report.skipped.push((item.key.clone(), SkipReason::AlreadyPresent));
                            continue;
                        }
                        // Stale staged value: drop it so the refetch below
                        // replaces, rather than duplicates, the entry.
                        self.context.remove(pos);
                        self.expired += 1;
                    }
                    while self.context.len() >= self.max_context_entries {
                        self.evict_lru_context(&mut report);
                    }
                    let value = fetcher(&item.key);
                    self.context.push((
                        item.key.clone(),
                        StagedContext { value, fetched_ms: now_ms, ttl_ms: item.ttl_ms },
                    ));
                    self.prefetched += 1;
                    self.bytes_moved += item.cost_bytes;
                    report.bytes_moved += item.cost_bytes;
                    report.filled.push((PrefetchKind::Context, item.key.clone()));
                }
                PrefetchKind::Model => {
                    if self.is_model_resident(&item.key) {
                        self.touch_model(&item.key);
                        report.skipped.push((item.key.clone(), SkipReason::AlreadyPresent));
                        continue;
                    }
                    while self.models.len() >= MAX_RESIDENT_MODELS {
                        let victim = self.models.remove(0);
                        self.evicted += 1;
                        report.evicted.push(victim);
                    }
                    self.models.push(item.key.clone());
                    self.prefetched += 1;
                    self.bytes_moved += item.cost_bytes;
                    report.bytes_moved += item.cost_bytes;
                    report.filled.push((PrefetchKind::Model, item.key.clone()));
                }
                PrefetchKind::Tool => {
                    if !item.read_only {
                        report.skipped.push((item.key.clone(), SkipReason::NotReadOnly));
                        continue;
                    }
                    if self.is_tool_warmed(&item.key) {
                        self.touch_tool(&item.key);
                        report.skipped.push((item.key.clone(), SkipReason::AlreadyPresent));
                        continue;
                    }
                    while self.tools.len() >= self.max_tools {
                        let victim = self.tools.remove(0);
                        self.evicted += 1;
                        report.evicted.push(victim);
                    }
                    self.tools.push(item.key.clone());
                    self.prefetched += 1;
                    report.filled.push((PrefetchKind::Tool, item.key.clone()));
                }
            }
        }
        report
    }

    fn touch_model(&mut self, name: &str) {
        if let Some(pos) = self.models.iter().position(|n| n == name) {
            let n = self.models.remove(pos);
            self.models.push(n);
        }
    }

    fn touch_tool(&mut self, name: &str) {
        if let Some(pos) = self.tools.iter().position(|n| n == name) {
            let n = self.tools.remove(pos);
            self.tools.push(n);
        }
    }

    /// The real-request path: a request for `key` consumes a staged hit
    /// (fresh entries only). A request for any other key sees nothing —
    /// a mispredicted prefetch is invisible to it, by construction.
    pub fn context_hit(&mut self, key: &str, now_ms: u64) -> Option<Value> {
        let pos = match self.context.iter().position(|(k, _)| k == key) {
            Some(pos) => pos,
            None => {
                self.misses += 1;
                return None;
            }
        };
        if !self.context[pos].1.fresh(now_ms) {
            self.context.remove(pos);
            self.expired += 1;
            self.misses += 1;
            return None;
        }
        self.hits += 1;
        let value = self.context[pos].1.value.clone();
        self.touch_context(key);
        Some(value)
    }

    /// Drop a staged context entry, e.g. on an external invalidation.
    /// Returns false when nothing was staged for the key.
    pub fn invalidate_context(&mut self, key: &str) -> bool {
        if let Some(pos) = self.context.iter().position(|(k, _)| k == key) {
            self.context.remove(pos);
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn good_ctx() -> SpecContext {
        SpecContext { battery_pct: 80, thermal_high: false, foreground: true, tool_time_fraction: 0.6 }
    }

    fn fetch_text(v: &str) -> impl FnMut(&str) -> Value {
        let v = v.to_string();
        move |_k: &str| Value::text(v.clone())
    }

    #[test]
    fn three_kinds_fill_staging() {
        let mut store = PrefetchStore::new(8);
        let items = vec![
            PrefetchItem::context("travel:policy", None, 128),
            PrefetchItem::model("large", 4096),
            PrefetchItem::tool("calendar", true),
        ];
        let report = store.prefetch(&good_ctx(), &items, 0, &mut fetch_text("policy-v1"));
        assert_eq!(report.filled_count(), 3);
        assert_eq!(store.context_hit("travel:policy", 0), Some(Value::text("policy-v1")));
        assert!(store.is_model_resident("large"));
        assert!(store.is_tool_warmed("calendar"));
    }

    #[test]
    fn mispredicted_prefetch_is_invisible_to_other_keys() {
        // Prefetching the wrong key wastes bytes but produces no result:
        // the report carries keys and bytes only, and a real request for
        // a different key observes nothing staged.
        let mut store = PrefetchStore::new(8);
        let items = vec![PrefetchItem::context("travel:shanghai", None, 512)];
        let report = store.prefetch(&good_ctx(), &items, 0, &mut fetch_text("shanghai-data"));
        assert_eq!(report.bytes_moved, 512);
        assert_eq!(store.context_hit("travel:beijing", 0), None);
        assert_eq!(store.misses, 1);
    }

    #[test]
    fn gate_closed_fetches_nothing() {
        let mut store = PrefetchStore::new(8);
        let mut calls = 0u32;
        let mut fetch = |_k: &str| {
            calls += 1;
            Value::text("x")
        };
        let low_battery =
            SpecContext { battery_pct: 19, thermal_high: false, foreground: true, tool_time_fraction: 0.6 };
        let items = vec![
            PrefetchItem::context("travel:policy", None, 128),
            PrefetchItem::model("large", 4096),
            PrefetchItem::tool("calendar", true),
        ];
        let report = store.prefetch(&low_battery, &items, 0, &mut fetch);
        assert_eq!(calls, 0, "a closed gate must waste no bandwidth");
        assert_eq!(report.filled_count(), 0);
        assert_eq!(report.skipped.len(), 3);
        assert!(report.skipped.iter().all(|(_, r)| *r == SkipReason::Gate(SpecGate::BatteryLow)));
        assert!(store.staged_context_keys().is_empty());
        assert!(!store.is_model_resident("large"));
        assert!(!store.is_tool_warmed("calendar"));
    }

    #[test]
    fn duplicate_prefetch_does_not_refetch() {
        let mut store = PrefetchStore::new(8);
        let mut calls = 0u32;
        let mut fetch = |_k: &str| {
            calls += 1;
            Value::text("v")
        };
        let items = vec![PrefetchItem::context("k", None, 10)];
        let first = store.prefetch(&good_ctx(), &items, 0, &mut fetch);
        assert_eq!(first.filled_count(), 1);
        let second = store.prefetch(&good_ctx(), &items, 1, &mut fetch);
        assert_eq!(calls, 1);
        assert_eq!(second.filled_count(), 0);
        assert_eq!(second.skipped, vec![("k".to_string(), SkipReason::AlreadyPresent)]);
    }

    #[test]
    fn non_readonly_tool_is_never_warmed() {
        let mut store = PrefetchStore::new(8);
        let items = vec![PrefetchItem::tool("payment-send", false)];
        let report = store.prefetch(&good_ctx(), &items, 0, &mut fetch_text("x"));
        assert_eq!(report.filled_count(), 0);
        assert_eq!(report.skipped, vec![("payment-send".to_string(), SkipReason::NotReadOnly)]);
        assert!(!store.is_tool_warmed("payment-send"));
    }

    #[test]
    fn expired_context_is_not_a_hit() {
        let mut store = PrefetchStore::new(8);
        let items = vec![PrefetchItem::context("weather", Some(100), 10)];
        store.prefetch(&good_ctx(), &items, 0, &mut fetch_text("sunny"));
        assert_eq!(store.context_hit("weather", 99), Some(Value::text("sunny")));
        assert_eq!(store.context_hit("weather", 100), None, "TTL boundary is exclusive, like ch. 11 Eventual");
        assert_eq!(store.expired, 1);
    }

    #[test]
    fn capacity_evicts_lru_prefetch_only() {
        let mut store = PrefetchStore::new(2);
        let items = vec![
            PrefetchItem::context("a", None, 1),
            PrefetchItem::context("b", None, 1),
            PrefetchItem::context("c", None, 1),
        ];
        let report = store.prefetch(&good_ctx(), &items, 0, &mut fetch_text("v"));
        assert_eq!(report.evicted, vec!["a".to_string()]);
        assert_eq!(store.evicted, 1);
        assert_eq!(store.context_hit("a", 0), None);
        assert!(store.context_hit("b", 0).is_some());
        assert!(store.context_hit("c", 0).is_some());
    }

    #[test]
    fn model_residency_caps_at_two_lru() {
        let mut store = PrefetchStore::new(8);
        let items = vec![
            PrefetchItem::model("m1", 100),
            PrefetchItem::model("m2", 100),
            PrefetchItem::model("m3", 100),
        ];
        let report = store.prefetch(&good_ctx(), &items, 0, &mut fetch_text("x"));
        assert_eq!(report.evicted, vec!["m1".to_string()]);
        assert_eq!(store.resident_models(), &["m2".to_string(), "m3".to_string()]);
    }

    #[test]
    fn invalidate_drops_staged_context() {
        let mut store = PrefetchStore::new(8);
        store.prefetch(&good_ctx(), &[PrefetchItem::context("k", None, 1)], 0, &mut fetch_text("v"));
        assert!(store.invalidate_context("k"));
        assert_eq!(store.context_hit("k", 0), None);
        assert!(!store.invalidate_context("k"));
    }

    #[test]
    fn stale_staged_value_is_refetched_not_duplicated() {
        let mut store = PrefetchStore::new(8);
        let mut calls = 0u32;
        let mut fetch = |_k: &str| {
            calls += 1;
            Value::text(format!("v{calls}"))
        };
        store.prefetch(&good_ctx(), &[PrefetchItem::context("k", Some(10), 1)], 0, &mut fetch);
        store.prefetch(&good_ctx(), &[PrefetchItem::context("k", Some(10), 1)], 20, &mut fetch);
        assert_eq!(calls, 2);
        assert_eq!(store.staged_context_keys(), vec!["k".to_string()]);
        assert_eq!(store.context_hit("k", 20), Some(Value::text("v2")));
    }
}
