//! Cross-device reality-cache sync with version vectors (doc ch. 18).
//!
//! Device clocks cannot be trusted, so sync never compares timestamps.
//! Each replica instead keeps a **version vector**: one counter per
//! device, counting how many versions from that device this replica has
//! seen. The vector can say exactly what a timestamp cannot — "A has
//! seen B's version 3 but not version 4" is `get("B") == 3`.
//!
//! Comparison is component-wise, over the union of devices (a device
//! absent from a vector counts as 0):
//!
//! * **Equal** — same history, same knowledge.
//! * **Dominates** — every counter at least as large, and at least one
//!   strictly larger: this history includes the other's.
//! * **Dominated** — the mirror image: this history is strictly older.
//! * **Concurrent** — neither includes the other: two devices wrote
//!   without seeing each other. That is a real divergence, not an
//!   ordering to be guessed.
//!
//! [`SyncStore`] applies that comparison per cache key, because a
//! version vector is a per-origin counter, not a per-key clock: the
//! entry for key K carries the vector snapshot of the moment it was
//! written, and a sync decision compares the incoming entry's vector
//! against the local entry for the *same* key. Consequences, all
//! deliberate:
//!
//! * An incoming entry that the local entry dominates is **stale** and
//!   is dropped (`AlreadySeen`) — the local value stands.
//! * An incoming entry that dominates the local entry **fast-forwards**
//!   it (`Applied`).
//! * Concurrent entries are a **conflict**: the local value is kept,
//!   the remote value is recorded, and nothing is auto-merged — the
//!   same rule as [`crate::worldcache`] (ch. 11: the cache never
//!   auto-merges reality; a human resolves it at a `yield`).
//! * Resolving a conflict marks that remote version as seen for that
//!   key, so re-delivering the same update afterwards is `AlreadySeen`
//!   instead of a second, duplicate conflict. The marker is per key,
//!   not global: seeing version 4 of a device through one key must
//!   never suppress a first-seen version 3 of the same device for a
//!   different key.
//!
//! Two honest simplifications, in line with the rest of this prototype:
//! the store is single-process (a real deployment would serialize
//! [`VersionedEntry`] over the device link and authenticate it — the
//! [`crate::delegation`] attestation is where that authentication plugs
//! in); and resolved-version markers are kept forever, where a real
//! system would prune them once every peer's vector dominates them.

use crate::Value;
use std::collections::{BTreeMap, HashMap};

// ---------------------------------------------------------------------------
// VersionVector
// ---------------------------------------------------------------------------

/// One counter per device: how many versions from that device have been
/// seen. Absent devices count as 0.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VersionVector {
    counters: BTreeMap<String, u64>,
}

impl VersionVector {
    pub fn new() -> Self {
        VersionVector::default()
    }

    /// Versions seen from `device` (0 if that device was never seen).
    pub fn get(&self, device: &str) -> u64 {
        self.counters.get(device).copied().unwrap_or(0)
    }

    /// Record one more version from `device` (our own write, or a
    /// version just accepted from that device). Returns the new count.
    /// Saturating: a counter that somehow reached `u64::MAX` stays put
    /// rather than wrapping to a version that looks older than all.
    pub fn increment(&mut self, device: &str) -> u64 {
        let next = self.get(device).saturating_add(1);
        self.counters.insert(device.to_string(), next);
        next
    }

    /// Element-wise maximum: the union of what both vectors have seen.
    /// Idempotent and commutative — merging the same update twice, or
    /// in either order, lands in the same state.
    pub fn merge(&mut self, other: &VersionVector) {
        for (device, count) in &other.counters {
            let merged = self.get(device).max(*count);
            self.counters.insert(device.clone(), merged);
        }
    }

    /// A fresh vector holding the element-wise maximum of both.
    pub fn merged(&self, other: &VersionVector) -> VersionVector {
        let mut out = self.clone();
        out.merge(other);
        out
    }

    /// Component-wise comparison over the union of devices.
    pub fn compare(&self, other: &VersionVector) -> VersionOrder {
        let mut self_greater = false;
        let mut other_greater = false;
        // Union of keys: iterate both maps without allocating a set.
        let mut keys: Vec<&str> =
            self.counters.keys().map(|k| k.as_str()).collect();
        for k in other.counters.keys().map(|k| k.as_str()) {
            if !keys.contains(&k) {
                keys.push(k);
            }
        }
        for device in keys {
            let a = self.get(device);
            let b = other.get(device);
            if a > b {
                self_greater = true;
            } else if b > a {
                other_greater = true;
            }
        }
        match (self_greater, other_greater) {
            (false, false) => VersionOrder::Equal,
            (true, false) => VersionOrder::Dominates,
            (false, true) => VersionOrder::Dominated,
            (true, true) => VersionOrder::Concurrent,
        }
    }

    /// True when `self` includes `other`'s history (equal or dominates).
    pub fn dominates_or_equal(&self, other: &VersionVector) -> bool {
        matches!(self.compare(other), VersionOrder::Equal | VersionOrder::Dominates)
    }

    /// True when `self` strictly includes `other`'s history.
    pub fn dominates(&self, other: &VersionVector) -> bool {
        self.compare(other) == VersionOrder::Dominates
    }

    /// The doc's sentence, as a predicate: "have we seen version
    /// `version` of `device`?" Version 0 is trivially seen (nothing).
    pub fn has_seen(&self, device: &str, version: u64) -> bool {
        self.get(device) >= version
    }

    /// Devices this vector is behind on, as seen from `other`: every
    /// device where `other` has seen more than we have. Each gap names
    /// the device and both counts, so a sync peer knows exactly which
    /// versions to send — no clock, no "last modified" guesswork.
    pub fn gaps_against(&self, other: &VersionVector) -> Vec<VersionGap> {
        let mut gaps = Vec::new();
        for (device, peer_seen) in &other.counters {
            let seen = self.get(device);
            if *peer_seen > seen {
                gaps.push(VersionGap { device: device.clone(), seen, peer_seen: *peer_seen });
            }
        }
        gaps.sort_by(|a, b| a.device.cmp(&b.device));
        gaps
    }

    pub fn is_empty(&self) -> bool {
        self.counters.is_empty() || self.counters.values().all(|c| *c == 0)
    }

    pub fn devices(&self) -> Vec<String> {
        self.counters.keys().cloned().collect()
    }
}

/// One device this vector is behind on, from [`VersionVector::gaps_against`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VersionGap {
    pub device: String,
    /// Versions of `device` this vector has seen.
    pub seen: u64,
    /// Versions of `device` the peer has seen.
    pub peer_seen: u64,
}

/// Outcome of [`VersionVector::compare`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VersionOrder {
    Equal,
    /// `self` includes `other`'s history and more.
    Dominates,
    /// `other` includes `self`'s history and more.
    Dominated,
    /// Neither includes the other: divergent histories.
    Concurrent,
}

// ---------------------------------------------------------------------------
// VersionedEntry + SyncStore
// ---------------------------------------------------------------------------

/// One reality-cache entry as it travels between devices: the value
/// plus the version-vector snapshot of the moment it was written.
/// The snapshot is the causal history — two entries for the same key
/// with concurrent snapshots were written without seeing each other.
#[derive(Debug, Clone, PartialEq)]
pub struct VersionedEntry {
    pub value: Value,
    pub vector: VersionVector,
    /// Device that wrote this version.
    pub writer: String,
}

/// What [`SyncStore::receive`] decided for one incoming entry.
#[derive(Debug, Clone, PartialEq)]
pub enum SyncDecision {
    /// The entry was stored (new key, or it dominates the local entry).
    Applied,
    /// The entry adds nothing: stale, a re-delivery, or a version
    /// already resolved for this key. The local entry is untouched.
    AlreadySeen,
    /// Concurrent with the local entry: the local entry is kept and
    /// the remote entry is recorded in [`SyncStore::conflicts`] for a
    /// human decision. Never an auto-merge.
    Conflict { local: VersionedEntry, remote: VersionedEntry },
}

/// A recorded sync conflict awaiting a human decision.
#[derive(Debug, Clone, PartialEq)]
pub struct SyncConflict {
    pub key: String,
    pub local: VersionedEntry,
    pub remote: VersionedEntry,
}

/// One device's side of a reality-cache sync: its entries, what it has
/// seen overall ([`SyncStore::knowledge`]), and its unresolved
/// conflicts. Sync itself is just [`SyncStore::export`] on one side
/// and [`SyncStore::import`] on the other — no timestamps anywhere.
pub struct SyncStore {
    device_id: String,
    knowledge: VersionVector,
    entries: HashMap<String, VersionedEntry>,
    conflicts: Vec<SyncConflict>,
    /// (key, remote vector) pairs whose conflict was already resolved:
    /// re-delivering such an update is `AlreadySeen`, not a new
    /// conflict. Per key, so it can never suppress a different key.
    resolved: Vec<(String, VersionVector)>,
}

impl SyncStore {
    pub fn new(device_id: &str) -> Self {
        SyncStore {
            device_id: device_id.to_string(),
            knowledge: VersionVector::new(),
            entries: HashMap::new(),
            conflicts: Vec::new(),
            resolved: Vec::new(),
        }
    }

    pub fn device_id(&self) -> &str {
        &self.device_id
    }

    /// Everything this replica has observed: its own writes plus every
    /// accepted remote entry, merged. `has_seen` on this vector is the
    /// "seen version 3 but not version 4" statement for the replica.
    pub fn knowledge(&self) -> &VersionVector {
        &self.knowledge
    }

    pub fn has_seen(&self, device: &str, version: u64) -> bool {
        self.knowledge.has_seen(device, version)
    }

    pub fn get(&self, key: &str) -> Option<&VersionedEntry> {
        self.entries.get(key)
    }

    /// Conflicts awaiting a human decision (see [`SyncStore::resolve`]).
    pub fn conflicts(&self) -> &[SyncConflict] {
        &self.conflicts
    }

    /// Local write: bumps our own counter and snapshots the resulting
    /// vector onto the entry. Returns the stored entry.
    pub fn write(&mut self, key: &str, value: Value) -> VersionedEntry {
        self.knowledge.increment(&self.device_id);
        let entry = VersionedEntry {
            value,
            vector: self.knowledge.clone(),
            writer: self.device_id.clone(),
        };
        self.entries.insert(key.to_string(), entry.clone());
        entry
    }

    /// Accept one entry from another device, decided by comparing its
    /// vector against the local entry for the same key.
    pub fn receive(&mut self, key: &str, remote: VersionedEntry) -> SyncDecision {
        // A version already resolved for this key never conflicts twice.
        if self.resolved.iter().any(|(k, v)| {
            k == key && v.dominates_or_equal(&remote.vector)
        }) {
            return SyncDecision::AlreadySeen;
        }
        match self.entries.get(key) {
            None => self.apply(key, remote),
            Some(local) => {
                let local = local.clone();
                match local.vector.compare(&remote.vector) {
                    VersionOrder::Equal => {
                        if local.value == remote.value {
                            SyncDecision::AlreadySeen
                        } else {
                            self.record_conflict(key, local, remote)
                        }
                    }
                    // Local history includes the remote one: stale.
                    VersionOrder::Dominates => SyncDecision::AlreadySeen,
                    // Remote history includes the local one: fast-forward.
                    VersionOrder::Dominated => self.apply(key, remote),
                    // Divergent histories: keep local, record, no merge.
                    VersionOrder::Concurrent => self.record_conflict(key, local, remote),
                }
            }
        }
    }

    fn apply(&mut self, key: &str, remote: VersionedEntry) -> SyncDecision {
        self.knowledge.merge(&remote.vector);
        self.entries.insert(key.to_string(), remote);
        SyncDecision::Applied
    }

    fn record_conflict(
        &mut self,
        key: &str,
        local: VersionedEntry,
        remote: VersionedEntry,
    ) -> SyncDecision {
        let decision =
            SyncDecision::Conflict { local: local.clone(), remote: remote.clone() };
        // Re-delivery before resolution re-reports the same conflict
        // without duplicating the record.
        if !self.conflicts.iter().any(|c| {
            c.key == key && c.local.vector == local.vector && c.remote.vector == remote.vector
        }) {
            self.conflicts.push(SyncConflict { key: key.to_string(), local, remote });
        }
        decision
    }

    /// Resolve a recorded conflict with the human's decision: keep the
    /// local entry (the remote version is still marked seen, so its
    /// re-delivery is `AlreadySeen`) or take the remote entry. Returns
    /// false when no conflict is recorded for `key`.
    pub fn resolve(&mut self, key: &str, keep_local: bool) -> bool {
        let Some(pos) = self.conflicts.iter().position(|c| c.key == key) else {
            return false;
        };
        let conflict = self.conflicts.remove(pos);
        self.resolved.push((key.to_string(), conflict.remote.vector.clone()));
        if !keep_local {
            self.knowledge.merge(&conflict.remote.vector);
            self.entries.insert(key.to_string(), conflict.remote);
        } else {
            self.knowledge.merge(&conflict.remote.vector);
        }
        true
    }

    /// Entries a peer at `peer_vector` has not fully seen: every entry
    /// whose history the peer's vector does not yet include. Sorted by
    /// key so a sync round is deterministic. This is what a device
    /// sends after comparing vectors — it never needs the peer's clock.
    pub fn entries_unseen_by(&self, peer_vector: &VersionVector) -> Vec<(String, VersionedEntry)> {
        let mut out: Vec<(String, VersionedEntry)> = self
            .entries
            .iter()
            .filter(|(_, e)| !peer_vector.dominates_or_equal(&e.vector))
            .map(|(k, e)| (k.clone(), e.clone()))
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// Everything this store holds, sorted by key: the payload of a
    /// sync round (the peer runs [`SyncStore::import`] on it).
    pub fn export(&self) -> Vec<(String, VersionedEntry)> {
        let mut out: Vec<(String, VersionedEntry)> =
            self.entries.iter().map(|(k, e)| (k.clone(), e.clone())).collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// Apply a batch of entries from another device, in order, and
    /// report one decision per entry.
    pub fn import(
        &mut self,
        entries: Vec<(String, VersionedEntry)>,
    ) -> Vec<(String, SyncDecision)> {
        entries.into_iter().map(|(k, e)| {
            let decision = self.receive(&k, e);
            (k, decision)
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec(pairs: &[(&str, u64)]) -> VersionVector {
        let mut v = VersionVector::new();
        for (device, count) in pairs {
            for _ in 0..*count {
                v.increment(device);
            }
        }
        v
    }

    #[test]
    fn unseen_device_counts_as_zero() {
        let v = VersionVector::new();
        assert_eq!(v.get("phone"), 0);
        assert!(v.has_seen("phone", 0));
        assert!(!v.has_seen("phone", 1));
        assert_eq!(v.compare(&vec(&[("phone", 1)])), VersionOrder::Dominated);
    }

    #[test]
    fn seen_version_3_not_version_4() {
        // The doc's sentence, verbatim in code: A saw B's v3, not v4.
        let mut seen = VersionVector::new();
        for _ in 0..3 {
            seen.increment("phone");
        }
        assert!(seen.has_seen("phone", 3));
        assert!(!seen.has_seen("phone", 4));
        // And a peer that has seen v4 is strictly ahead on that device.
        let peer = vec(&[("phone", 4)]);
        assert_eq!(seen.compare(&peer), VersionOrder::Dominated);
        assert_eq!(peer.compare(&seen), VersionOrder::Dominates);
        assert_eq!(
            seen.gaps_against(&peer),
            vec![VersionGap { device: "phone".to_string(), seen: 3, peer_seen: 4 }]
        );
    }

    #[test]
    fn compare_covers_equal_dominates_and_concurrent() {
        let a = vec(&[("phone", 2), ("glasses", 1)]);
        assert_eq!(a.compare(&a.clone()), VersionOrder::Equal);
        assert!(a.dominates_or_equal(&vec(&[("phone", 2)])));
        assert!(a.dominates(&vec(&[("phone", 2)])));
        assert!(!vec(&[("phone", 2)]).dominates(&a));
        // Each side has something the other lacks: divergent.
        let b = vec(&[("phone", 1), ("glasses", 2)]);
        assert_eq!(a.compare(&b), VersionOrder::Concurrent);
        assert_eq!(b.compare(&a), VersionOrder::Concurrent);
    }

    #[test]
    fn merge_is_elementwise_max_idempotent_and_commutative() {
        let a = vec(&[("phone", 2), ("glasses", 1)]);
        let b = vec(&[("phone", 1), ("glasses", 3)]);
        assert_eq!(a.merged(&b), vec(&[("phone", 2), ("glasses", 3)]));
        assert_eq!(b.merged(&a), a.merged(&b));
        assert_eq!(a.merged(&a), a);
    }

    #[test]
    fn gaps_list_every_device_the_peer_is_ahead_on() {
        let mine = vec(&[("phone", 3)]);
        let peer = vec(&[("phone", 4), ("glasses", 2)]);
        assert_eq!(
            mine.gaps_against(&peer),
            vec![
                VersionGap { device: "glasses".to_string(), seen: 0, peer_seen: 2 },
                VersionGap { device: "phone".to_string(), seen: 3, peer_seen: 4 },
            ]
        );
        assert!(peer.gaps_against(&mine).is_empty());
    }

    #[test]
    fn local_write_bumps_only_our_counter_and_snapshots_it() {
        let mut store = SyncStore::new("phone");
        let e1 = store.write("cal:event", Value::text("free"));
        assert_eq!(e1.writer, "phone");
        assert_eq!(e1.vector.get("phone"), 1);
        let e2 = store.write("cal:event", Value::text("booked"));
        assert_eq!(e2.vector.get("phone"), 2);
        assert_eq!(store.get("cal:event").unwrap().value, Value::text("booked"));
        assert!(store.has_seen("phone", 2));
        assert!(!store.has_seen("phone", 3));
    }

    #[test]
    fn receive_new_key_applies_and_advances_knowledge() {
        let mut phone = SyncStore::new("phone");
        let entry = phone.write("doc", Value::text("v1"));
        let mut glasses = SyncStore::new("glasses");
        assert_eq!(glasses.receive("doc", entry), SyncDecision::Applied);
        assert_eq!(glasses.get("doc").unwrap().value, Value::text("v1"));
        assert!(glasses.has_seen("phone", 1));
    }

    #[test]
    fn stale_update_is_dropped_and_local_value_stands() {
        let mut phone = SyncStore::new("phone");
        let v1 = phone.write("doc", Value::text("v1"));
        let _v2 = phone.write("doc", Value::text("v2"));
        let mut glasses = SyncStore::new("glasses");
        // Glasses first sees v2; a delayed v1 copy arriving afterwards
        // is dominated by the local entry and must not roll it back.
        assert_eq!(glasses.receive("doc", _v2.clone()), SyncDecision::Applied);
        assert_eq!(glasses.receive("doc", v1), SyncDecision::AlreadySeen);
        assert_eq!(glasses.get("doc").unwrap().value, Value::text("v2"));
    }

    #[test]
    fn dominating_update_fast_forwards() {
        let mut phone = SyncStore::new("phone");
        let mut glasses = SyncStore::new("glasses");
        let v1 = phone.write("doc", Value::text("v1"));
        assert_eq!(glasses.receive("doc", v1), SyncDecision::Applied);
        let v2 = phone.write("doc", Value::text("v2"));
        assert_eq!(glasses.receive("doc", v2), SyncDecision::Applied);
        assert_eq!(glasses.get("doc").unwrap().value, Value::text("v2"));
        assert!(glasses.has_seen("phone", 2));
    }

    #[test]
    fn concurrent_writes_conflict_and_never_auto_merge() {
        let mut phone = SyncStore::new("phone");
        let mut glasses = SyncStore::new("glasses");
        // Both start from nothing and write the same key blind: neither
        // vector includes the other, so the histories are concurrent.
        let _ = phone.write("doc", Value::text("phone-edit"));
        let glasses_entry = glasses.write("doc", Value::text("glasses-edit"));
        let decision = phone.receive("doc", glasses_entry.clone());
        assert!(matches!(decision, SyncDecision::Conflict { .. }));
        assert_eq!(phone.get("doc").unwrap().value, Value::text("phone-edit"));
        assert_eq!(phone.conflicts().len(), 1);
        assert_eq!(phone.conflicts()[0].remote, glasses_entry);
        // Re-delivery before resolution re-reports but does not duplicate.
        assert!(matches!(phone.receive("doc", glasses_entry), SyncDecision::Conflict { .. }));
        assert_eq!(phone.conflicts().len(), 1);
    }

    #[test]
    fn resolve_keep_remote_replaces_and_keep_local_marks_seen() {
        let mut phone = SyncStore::new("phone");
        let mut glasses = SyncStore::new("glasses");
        let _ = phone.write("doc", Value::text("phone-edit"));
        let glasses_entry = glasses.write("doc", Value::text("glasses-edit"));
        phone.receive("doc", glasses_entry.clone());
        assert!(phone.resolve("doc", false));
        assert_eq!(phone.get("doc").unwrap().value, Value::text("glasses-edit"));
        assert!(phone.conflicts().is_empty());
        assert!(!phone.resolve("doc", false), "nothing left to resolve");

        // Second scenario: keep local, then re-delivery is already-seen.
        let mut phone2 = SyncStore::new("phone");
        let mut glasses2 = SyncStore::new("glasses");
        let _ = phone2.write("doc", Value::text("phone-edit"));
        let remote = glasses2.write("doc", Value::text("glasses-edit"));
        phone2.receive("doc", remote.clone());
        assert!(phone2.resolve("doc", true));
        assert_eq!(phone2.get("doc").unwrap().value, Value::text("phone-edit"));
        assert_eq!(phone2.receive("doc", remote), SyncDecision::AlreadySeen);
        assert!(phone2.conflicts().is_empty());
    }

    #[test]
    fn export_import_round_trip_syncs_two_stores() {
        let mut phone = SyncStore::new("phone");
        phone.write("a", Value::int(1));
        phone.write("b", Value::int(2));
        let mut glasses = SyncStore::new("glasses");
        let decisions = glasses.import(phone.export());
        assert!(decisions.iter().all(|(_, d)| *d == SyncDecision::Applied));
        assert_eq!(glasses.get("a").unwrap().value, Value::int(1));
        assert_eq!(glasses.get("b").unwrap().value, Value::int(2));
        // A second round sends nothing new: all already seen.
        let decisions = glasses.import(phone.export());
        assert!(decisions.iter().all(|(_, d)| *d == SyncDecision::AlreadySeen));
    }

    #[test]
    fn entries_unseen_by_only_lists_what_the_peer_lacks() {
        let mut phone = SyncStore::new("phone");
        phone.write("a", Value::int(1));
        phone.write("b", Value::int(2));
        // Peer has seen phone's v1 (key a) but not v2 (key b): b's
        // vector is phone=2, which the peer's phone=1 does not include.
        let peer_vector = vec(&[("phone", 1)]);
        let unseen = phone.entries_unseen_by(&peer_vector);
        assert_eq!(unseen.len(), 1);
        assert_eq!(unseen[0].0, "b");
        // A peer that has seen everything needs nothing.
        assert!(phone.entries_unseen_by(&vec(&[("phone", 2)])).is_empty());
    }
}
