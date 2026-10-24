//! The reality cache (doc ch. 11): cache coherence for the *outside world*.
//! Agents constantly need world state (is the calendar slot free? has this
//! mail been read?), but the world changes underneath them. The cache
//! borrows MESI's structure with adapted semantics:
//!
//! * **E** — exclusive-fresh: just fetched, no concurrent writer known.
//! * **M** — modified: we changed it locally, server ack outstanding
//!   (reads still see our write: read-your-writes).
//! * **S** — shared: was correct, but another holder may have changed it;
//!   readable, but high-risk operations must revalidate first.
//! * **I** — invalid: known stale; must refetch before any read.
//!
//! The consistency level is declared by the *tool's manifest*, never chosen
//! by the caller: payments are Strong (caching forbidden), calendar is
//! Causal, weather is Eventual-with-TTL, sensors are BestEffort. When an
//! external change hits an M entry, that is a *conflict*: the cache never
//! auto-merges reality — it records the conflict and the caller escalates
//! to a human `yield`.

use crate::typesys::Consistency;
use crate::Value;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheState {
    E,
    M,
    S,
    I,
}

#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub value: Value,
    pub state: CacheState,
    pub version: u64,
    pub fetched_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadSource {
    Cache,
    Refetch,
}

#[derive(Debug, Clone)]
pub struct ReadOutcome {
    pub value: Value,
    pub source: ReadSource,
    pub state: CacheState,
    /// True when served from S: fine for display, not for high-risk action
    /// without a revalidation.
    pub stale_risk: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChangeOutcome {
    /// Entry invalidated (was E/S/I); next read refetches.
    Invalidated,
    /// We hold an unacknowledged local write and the world moved too:
    /// a human must decide. Both versions are preserved in the outcome.
    Conflict { local: Value, remote: Value },
    /// Nobody had it cached; nothing to do.
    Uncached,
}

pub struct WorldCache {
    entries: HashMap<String, CacheEntry>,
    conflicts: Vec<(String, Value, Value)>,
}

impl WorldCache {
    pub fn new() -> Self {
        WorldCache { entries: HashMap::new(), conflicts: Vec::new() }
    }

    pub fn state_of(&self, key: &str) -> Option<CacheState> {
        self.entries.get(key).map(|e| e.state)
    }

    pub fn version_of(&self, key: &str) -> Option<u64> {
        self.entries.get(key).map(|e| e.version)
    }

    /// The sync-conflict record awaiting a human decision (doc ch. 11:
    /// conflicts trigger `yield`, never auto-merge).
    pub fn conflicts(&self) -> &[(String, Value, Value)] {
        &self.conflicts
    }

    fn refetch(
        &mut self,
        key: &str,
        now_ms: u64,
        fetcher: &mut dyn FnMut(&str) -> Value,
    ) -> ReadOutcome {
        let value = fetcher(key);
        let version = self.entries.get(key).map(|e| e.version + 1).unwrap_or(1);
        self.entries.insert(
            key.to_string(),
            CacheEntry { value: value.clone(), state: CacheState::E, version, fetched_ms: now_ms },
        );
        ReadOutcome { value, source: ReadSource::Refetch, state: CacheState::E, stale_risk: false }
    }

    fn hit(&self, entry: &CacheEntry) -> ReadOutcome {
        ReadOutcome {
            value: entry.value.clone(),
            source: ReadSource::Cache,
            state: entry.state,
            stale_risk: entry.state == CacheState::S,
        }
    }

    /// Read `key` under the tool-declared consistency level. `fetcher`
    /// performs the authoritative re-read when the level demands one.
    pub fn read(
        &mut self,
        key: &str,
        consistency: Consistency,
        now_ms: u64,
        fetcher: &mut dyn FnMut(&str) -> Value,
    ) -> ReadOutcome {
        match consistency {
            // Payments & balances: caching is forbidden, always re-read.
            Consistency::Strong => self.refetch(key, now_ms, fetcher),
            Consistency::BestEffort => match self.entries.get(key) {
                Some(e) if e.state != CacheState::I => self.hit(e),
                _ => self.refetch(key, now_ms, fetcher),
            },
            Consistency::Eventual { ttl_ms } => match self.entries.get(key) {
                Some(e) if e.state != CacheState::I && now_ms.saturating_sub(e.fetched_ms) < ttl_ms => {
                    self.hit(e)
                }
                _ => self.refetch(key, now_ms, fetcher),
            },
            Consistency::Causal => match self.entries.get(key) {
                // M and E are safe; S is readable but flagged; I refetches.
                Some(e) if e.state != CacheState::I => self.hit(e),
                _ => self.refetch(key, now_ms, fetcher),
            },
        }
    }

    /// Forced revalidation — the fallback for high-risk operations, applied
    /// even when the entry looks fresh (doc ch. 11).
    pub fn revalidate(
        &mut self,
        key: &str,
        now_ms: u64,
        fetcher: &mut dyn FnMut(&str) -> Value,
    ) -> ReadOutcome {
        self.refetch(key, now_ms, fetcher)
    }

    /// Record a local write: state M until the server acks. Reads under
    /// Causal consistency see this value (read-your-writes).
    pub fn write_local(&mut self, key: &str, value: Value, now_ms: u64) {
        let version = self.entries.get(key).map(|e| e.version + 1).unwrap_or(1);
        self.entries.insert(
            key.to_string(),
            CacheEntry { value, state: CacheState::M, version, fetched_ms: now_ms },
        );
    }

    /// Server acknowledged our write: M → E.
    pub fn ack(&mut self, key: &str) {
        if let Some(e) = self.entries.get_mut(key) {
            if e.state == CacheState::M {
                e.state = CacheState::E;
            }
        }
    }

    /// Another holder started reading this key: E → S.
    pub fn mark_shared(&mut self, key: &str) {
        if let Some(e) = self.entries.get_mut(key) {
            if e.state == CacheState::E {
                e.state = CacheState::S;
            }
        }
    }

    /// The world changed underneath us (push invalidation — the best of the
    /// three invalidation sources in doc ch. 11).
    pub fn external_change(&mut self, key: &str, remote: Value) -> ChangeOutcome {
        match self.entries.get(key) {
            None => ChangeOutcome::Uncached,
            Some(e) => match e.state {
                CacheState::M => {
                    let local = e.value.clone();
                    self.entries.get_mut(key).expect("present").state = CacheState::I;
                    self.conflicts.push((key.to_string(), local.clone(), remote.clone()));
                    ChangeOutcome::Conflict { local, remote }
                }
                _ => {
                    self.entries.get_mut(key).expect("present").state = CacheState::I;
                    ChangeOutcome::Invalidated
                }
            },
        }
    }

    /// Resolve a recorded conflict with the human's decision (from a
    /// `yield`): keep the local write (state M, still to be pushed) or take
    /// the remote version (state E).
    pub fn resolve_conflict(&mut self, key: &str, keep_local: bool, now_ms: u64) -> bool {
        if let Some(pos) = self.conflicts.iter().position(|(k, _, _)| k == key) {
            let (_, local, remote) = self.conflicts.remove(pos);
            let (value, state) =
                if keep_local { (local, CacheState::M) } else { (remote, CacheState::E) };
            let version = self.entries.get(key).map(|e| e.version + 1).unwrap_or(1);
            self.entries.insert(
                key.to_string(),
                CacheEntry { value, state, version, fetched_ms: now_ms },
            );
            true
        } else {
            false
        }
    }
}

impl Default for WorldCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fetcher_with(v: &str) -> impl FnMut(&str) -> Value {
        let v = v.to_string();
        move |_k: &str| Value::text(v.clone())
    }

    #[test]
    fn strong_consistency_never_serves_cache() {
        let mut cache = WorldCache::new();
        let mut calls = 0u32;
        let mut fetch = |_k: &str| {
            calls += 1;
            Value::int(calls as i64)
        };
        let r1 = cache.read("balance", Consistency::Strong, 0, &mut fetch);
        let r2 = cache.read("balance", Consistency::Strong, 1, &mut fetch);
        assert_eq!(r1.source, ReadSource::Refetch);
        assert_eq!(r2.source, ReadSource::Refetch);
        assert_eq!(r2.value, Value::int(2), "every strong read re-reads the world");
    }

    #[test]
    fn eventual_consistency_serves_within_ttl_then_refetches() {
        let mut cache = WorldCache::new();
        let level = Consistency::Eventual { ttl_ms: 600_000 }; // weather: 10 min
        let r1 = cache.read("weather", level, 0, &mut fetcher_with("sunny"));
        assert_eq!(r1.source, ReadSource::Refetch);
        let r2 = cache.read("weather", level, 599_999, &mut fetcher_with("rainy"));
        assert_eq!(r2.source, ReadSource::Cache);
        assert_eq!(r2.value, Value::text("sunny"));
        let r3 = cache.read("weather", level, 600_000, &mut fetcher_with("rainy"));
        assert_eq!(r3.source, ReadSource::Refetch);
        assert_eq!(r3.value, Value::text("rainy"));
    }

    #[test]
    fn causal_read_your_writes() {
        let mut cache = WorldCache::new();
        cache.read("cal:event", Consistency::Causal, 0, &mut fetcher_with("free"));
        cache.write_local("cal:event", Value::text("booked-by-me"), 10);
        assert_eq!(cache.state_of("cal:event"), Some(CacheState::M));
        // Even though the "server" would still say "free", our own write wins.
        let r = cache.read("cal:event", Consistency::Causal, 20, &mut fetcher_with("free"));
        assert_eq!(r.source, ReadSource::Cache);
        assert_eq!(r.value, Value::text("booked-by-me"));
    }

    #[test]
    fn ack_moves_modified_to_exclusive() {
        let mut cache = WorldCache::new();
        cache.write_local("k", Value::int(1), 0);
        assert_eq!(cache.state_of("k"), Some(CacheState::M));
        cache.ack("k");
        assert_eq!(cache.state_of("k"), Some(CacheState::E));
    }

    #[test]
    fn external_change_invalidates_fresh_entry() {
        let mut cache = WorldCache::new();
        cache.read("k", Consistency::Causal, 0, &mut fetcher_with("v1"));
        assert_eq!(cache.state_of("k"), Some(CacheState::E));
        let outcome = cache.external_change("k", Value::text("v2"));
        assert_eq!(outcome, ChangeOutcome::Invalidated);
        assert_eq!(cache.state_of("k"), Some(CacheState::I));
        let r = cache.read("k", Consistency::Causal, 5, &mut fetcher_with("v2"));
        assert_eq!(r.source, ReadSource::Refetch);
    }

    #[test]
    fn external_change_during_local_write_is_a_conflict_not_a_merge() {
        let mut cache = WorldCache::new();
        cache.read("meeting", Consistency::Causal, 0, &mut fetcher_with("10:00"));
        cache.write_local("meeting", Value::text("11:00 (mine)"), 10);
        let outcome = cache.external_change("meeting", Value::text("12:00 (theirs)"));
        assert_eq!(
            outcome,
            ChangeOutcome::Conflict {
                local: Value::text("11:00 (mine)"),
                remote: Value::text("12:00 (theirs)"),
            }
        );
        assert_eq!(cache.conflicts().len(), 1);
        // Human picks theirs at the yield.
        assert!(cache.resolve_conflict("meeting", false, 20));
        assert_eq!(cache.state_of("meeting"), Some(CacheState::E));
        let r = cache.read("meeting", Consistency::Causal, 30, &mut fetcher_with("x"));
        assert_eq!(r.value, Value::text("12:00 (theirs)"));
        assert!(cache.conflicts().is_empty());
    }

    #[test]
    fn shared_entries_are_flagged_stale_risk_and_revalidate() {
        let mut cache = WorldCache::new();
        cache.read("doc", Consistency::Causal, 0, &mut fetcher_with("rev1"));
        cache.mark_shared("doc");
        assert_eq!(cache.state_of("doc"), Some(CacheState::S));
        let r = cache.read("doc", Consistency::Causal, 5, &mut fetcher_with("rev2"));
        assert_eq!(r.source, ReadSource::Cache);
        assert!(r.stale_risk);
        let r2 = cache.revalidate("doc", 6, &mut fetcher_with("rev2"));
        assert_eq!(r2.state, CacheState::E);
        assert_eq!(r2.value, Value::text("rev2"));
    }

    #[test]
    fn best_effort_serves_anything_cached() {
        let mut cache = WorldCache::new();
        cache.read("battery", Consistency::BestEffort, 0, &mut fetcher_with("82%"));
        cache.mark_shared("battery");
        let r = cache.read("battery", Consistency::BestEffort, 999_999, &mut fetcher_with("81%"));
        assert_eq!(r.source, ReadSource::Cache);
    }

    #[test]
    fn versions_increase_on_refetch_and_write() {
        let mut cache = WorldCache::new();
        cache.read("k", Consistency::Strong, 0, &mut fetcher_with("a"));
        assert_eq!(cache.version_of("k"), Some(1));
        cache.read("k", Consistency::Strong, 1, &mut fetcher_with("a"));
        assert_eq!(cache.version_of("k"), Some(2));
        cache.write_local("k", Value::text("b"), 2);
        assert_eq!(cache.version_of("k"), Some(3));
    }
}
