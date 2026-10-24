//! Multi-agent optimistic concurrency (doc ch. 17): no locks, validate at
//! commit time, roll the whole transaction back on conflict.
//!
//! Why no locks: an agent may hold a resource for minutes while it thinks,
//! so a lock would block every other agent touching the same object. The
//! database answer, ported here, is optimistic concurrency:
//!
//! * **Isolation by snapshot.** Each agent forks a copy-on-write snapshot
//!   of committed state at `begin` (doc ch. 17, "each agent one overlay").
//!   Reads see the snapshot plus the agent's own overlay writes; writes
//!   land only in the overlay, so intermediate states are invisible to
//!   other agents. Nothing reaches committed state until commit.
//! * **Validation at commit.** Commit checks the read set — every object
//!   the agent read must still be at the version it saw — and the write
//!   set — every object it wrote blind (without reading) must still be at
//!   its snapshot version, which catches write-write conflicts too. Any
//!   mismatch means the whole transaction rolls back: the overlay is
//!   discarded, nothing is applied, and the agent re-reads and retries.
//! * **Retry with backoff, then yield.** A retry may conflict again, so
//!   retries back off exponentially (capped) and, past the retry budget,
//!   escalate to a human decision (`yield` in doc terms) instead of
//!   spinning forever. The escalation is a value, not a panic: the caller
//!   owns the yield path.
//!
//! This is a single-threaded prototype, like the rest of the crate: there
//! are no threads and no real locks. Concurrent agents are modelled by
//! interleaving commits — in tests, the `interfere` callback of
//! [`run_with_retry`] stands in for other agents committing between this
//! agent's begin and commit. A real scheduler would interleave real agents
//! at exactly that point.

use crate::Value;
use std::collections::BTreeMap;

/// One committed object: its value and a monotonically increasing version.
/// Absent objects are version 0, so "created since the snapshot" is also a
/// version change (0 -> 1) and is caught by validation.
#[derive(Debug, Clone)]
struct Entry {
    value: Value,
    version: u64,
}

/// Committed state shared by all agents. Only [`AgentTxn::commit`] may
/// change it; agents otherwise interact through snapshots and overlays.
#[derive(Debug, Default)]
pub struct VersionedStore {
    entries: BTreeMap<String, Entry>,
}

impl VersionedStore {
    pub fn new() -> Self {
        VersionedStore { entries: BTreeMap::new() }
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.entries.get(key).map(|e| &e.value)
    }

    /// Version of an object, or 0 when it does not exist yet.
    pub fn version(&self, key: &str) -> u64 {
        self.entries.get(key).map(|e| e.version).unwrap_or(0)
    }

    /// Begin an agent transaction: fork a snapshot of committed state.
    pub fn begin(&self, agent: &str) -> AgentTxn {
        let snapshot = self
            .entries
            .iter()
            .map(|(k, e)| (k.clone(), (e.value.clone(), e.version)))
            .collect();
        AgentTxn {
            agent: agent.to_string(),
            snapshot,
            read_set: BTreeMap::new(),
            overlay: BTreeMap::new(),
        }
    }
}

/// A single agent's in-flight transaction: a snapshot, a read set, and an
/// overlay of uncommitted writes. Dropping it without commit discards
/// everything — that is the rollback.
#[derive(Debug)]
pub struct AgentTxn {
    agent: String,
    /// key -> (value, version) as of `begin`.
    snapshot: BTreeMap<String, (Value, u64)>,
    /// key -> version seen on first read.
    read_set: BTreeMap<String, u64>,
    /// key -> new value, not yet committed.
    overlay: BTreeMap<String, Value>,
}

impl AgentTxn {
    pub fn agent(&self) -> &str {
        &self.agent
    }

    /// Read through the overlay, then the snapshot. The first read of a
    /// key records its snapshot version in the read set; re-reads do not
    /// move it (validation is against what the agent decided on).
    pub fn read(&mut self, key: &str) -> Option<Value> {
        if let Some(v) = self.overlay.get(key) {
            // A read of an own-write still pins the snapshot version: the
            // agent decided on the basis of the snapshot, not its own edit.
            let ver = self.snapshot_version(key);
            self.read_set.entry(key.to_string()).or_insert(ver);
            return Some(v.clone());
        }
        match self.snapshot.get(key) {
            Some((v, ver)) => {
                self.read_set.entry(key.to_string()).or_insert(*ver);
                Some(v.clone())
            }
            None => {
                self.read_set.entry(key.to_string()).or_insert(0);
                None
            }
        }
    }

    /// Write to the overlay only. Nothing is visible to other agents, and
    /// nothing is applied, until a successful commit.
    pub fn write(&mut self, key: &str, value: Value) {
        self.overlay.insert(key.to_string(), value);
    }

    pub fn read_set(&self) -> &BTreeMap<String, u64> {
        &self.read_set
    }

    pub fn write_keys(&self) -> Vec<&str> {
        self.overlay.keys().map(|k| k.as_str()).collect()
    }

    fn snapshot_version(&self, key: &str) -> u64 {
        self.snapshot.get(key).map(|(_, v)| *v).unwrap_or(0)
    }

    /// Validate the read set and the blind-write set against live state,
    /// then either apply the whole overlay (bumping one version per key)
    /// or discard it entirely and report the conflicting keys.
    pub fn commit(self, store: &mut VersionedStore) -> Result<CommitReceipt, CommitConflict> {
        let mut conflicting: Vec<String> = Vec::new();
        for (key, seen) in &self.read_set {
            if store.version(key) != *seen {
                conflicting.push(key.clone());
            }
        }
        for key in self.overlay.keys() {
            if !self.read_set.contains_key(key) && store.version(key) != self.snapshot_version(key) {
                conflicting.push(key.clone());
            }
        }
        conflicting.sort();
        conflicting.dedup();
        if !conflicting.is_empty() {
            return Err(CommitConflict { agent: self.agent, conflicting_keys: conflicting });
        }
        let mut applied = Vec::new();
        for (key, value) in self.overlay {
            let entry = store.entries.entry(key.clone()).or_insert(Entry { value: Value::Null, version: 0 });
            entry.value = value;
            entry.version += 1;
            applied.push((key, entry.version));
        }
        Ok(CommitReceipt { agent: self.agent, applied })
    }
}

/// What a successful commit applied: each written key and its new version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommitReceipt {
    pub agent: String,
    pub applied: Vec<(String, u64)>,
}

/// A failed commit: the overlay was discarded, nothing was applied, and
/// these keys changed underneath the agent since its snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommitConflict {
    pub agent: String,
    pub conflicting_keys: Vec<String>,
}

impl std::fmt::Display for CommitConflict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "agent {} conflicts on {} (transaction rolled back)",
            self.agent,
            self.conflicting_keys.join(", ")
        )
    }
}
impl std::error::Error for CommitConflict {}

/// Exponential backoff with a cap, plus the retry budget. `attempt` counts
/// failed attempts so far: attempt 0 is the first retry decision, taken
/// after the first commit failure.
#[derive(Debug, Clone, Copy)]
pub struct RetryPolicy {
    pub base_delay_ms: u64,
    pub max_delay_ms: u64,
    pub max_retries: u32,
}

impl RetryPolicy {
    /// Delay before retry number `attempt`: base * 2^attempt, capped.
    /// Saturating arithmetic keeps absurd policies finite.
    pub fn delay_for(&self, attempt: u32) -> u64 {
        let mut delay = self.base_delay_ms;
        for _ in 0..attempt {
            delay = delay.saturating_mul(2);
            if delay >= self.max_delay_ms {
                return self.max_delay_ms;
            }
        }
        delay.min(self.max_delay_ms)
    }

    /// Retry while the budget lasts; past it, escalate to a human.
    pub fn decision(&self, failed_attempts: u32) -> RetryDecision {
        if failed_attempts < self.max_retries {
            RetryDecision::Retry { delay_ms: self.delay_for(failed_attempts) }
        } else {
            RetryDecision::Escalate
        }
    }
}

/// What to do after a conflicting commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryDecision {
    Retry { delay_ms: u64 },
    /// Retry budget exhausted: stop and yield to a human (doc ch. 17).
    Escalate,
}

/// Outcome of [`run_with_retry`].
#[derive(Debug)]
pub enum RetryOutcome {
    /// Committed after `attempts` tries; `delays_ms` is the backoff waited
    /// (in this prototype, recorded rather than slept) before each retry.
    Committed { attempts: u32, delays_ms: Vec<u64>, receipt: CommitReceipt },
    /// Budget exhausted: the last conflict, for the yield to a human.
    /// Nothing from any attempt was applied.
    Escalated { attempts: u32, delays_ms: Vec<u64>, conflict: CommitConflict },
}

impl RetryOutcome {
    /// Human-readable reason for an escalation, for the yield prompt.
    pub fn escalation_reason(conflict: &CommitConflict) -> String {
        format!(
            "optimistic commit kept conflicting on {} after the retry budget; needs a human decision",
            conflict.conflicting_keys.join(", ")
        )
    }
}

/// Run `work` inside an optimistic transaction, retrying on conflict with
/// exponential backoff and escalating past the budget.
///
/// Each attempt re-begins from a fresh snapshot, so the agent re-reads
/// current state before deciding again — retrying the same stale overlay
/// would conflict forever. `interfere` runs after `work` and before commit
/// on every attempt: it models other agents committing concurrently (in a
/// real scheduler, real agents interleave here). Pass `|_, _| {}` when
/// nothing contends.
pub fn run_with_retry<F, G>(
    store: &mut VersionedStore,
    agent: &str,
    policy: &RetryPolicy,
    mut work: F,
    mut interfere: G,
) -> RetryOutcome
where
    F: FnMut(&mut AgentTxn),
    G: FnMut(&mut VersionedStore, u32),
{
    let mut delays_ms = Vec::new();
    let mut attempt: u32 = 0;
    loop {
        let mut txn = store.begin(agent);
        work(&mut txn);
        interfere(store, attempt);
        match txn.commit(store) {
            Ok(receipt) => {
                return RetryOutcome::Committed { attempts: attempt + 1, delays_ms, receipt };
            }
            Err(conflict) => match policy.decision(attempt) {
                RetryDecision::Retry { delay_ms } => {
                    delays_ms.push(delay_ms);
                    attempt += 1;
                }
                RetryDecision::Escalate => {
                    return RetryOutcome::Escalated {
                        attempts: attempt + 1,
                        delays_ms,
                        conflict,
                    };
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store_with(key: &str, value: &str) -> VersionedStore {
        let mut store = VersionedStore::new();
        let mut txn = store.begin("seed");
        txn.write(key, Value::text(value));
        txn.commit(&mut store).unwrap();
        store
    }

    #[test]
    fn disjoint_writes_both_commit() {
        let mut store = store_with("a", "1");
        let mut ta = store.begin("a-agent");
        let mut tb = store.begin("b-agent");
        ta.write("x", Value::int(1));
        tb.write("y", Value::int(2));
        // Order does not matter: neither read nor wrote the other's key.
        assert!(ta.commit(&mut store).is_ok());
        assert!(tb.commit(&mut store).is_ok());
        assert_eq!(store.get("x"), Some(&Value::int(1)));
        assert_eq!(store.get("y"), Some(&Value::int(2)));
    }

    #[test]
    fn stale_read_conflicts_and_rolls_back_the_whole_overlay() {
        let mut store = store_with("doc", "v1");
        let mut reader = store.begin("reader");
        assert_eq!(reader.read("doc"), Some(Value::text("v1")));
        reader.write("doc", Value::text("reader-edit"));
        reader.write("other", Value::text("reader-new"));
        // Another agent commits a change to `doc` in between.
        let mut writer = store.begin("writer");
        writer.write("doc", Value::text("v2"));
        writer.commit(&mut store).unwrap();
        // The reader's whole overlay is discarded, not partially applied.
        let err = reader.commit(&mut store).unwrap_err();
        assert_eq!(err.conflicting_keys, vec!["doc".to_string()]);
        assert_eq!(store.get("doc"), Some(&Value::text("v2")));
        assert_eq!(store.get("other"), None, "rollback must not leak the other write");
    }

    #[test]
    fn blind_write_write_conflict_is_caught() {
        let mut store = store_with("slot", "free");
        // Agent A writes `slot` without ever reading it.
        let mut ta = store.begin("a");
        ta.write("slot", Value::text("a-booking"));
        // Agent B commits the same key first.
        let mut tb = store.begin("b");
        tb.write("slot", Value::text("b-booking"));
        tb.commit(&mut store).unwrap();
        let err = ta.commit(&mut store).unwrap_err();
        assert_eq!(err.conflicting_keys, vec!["slot".to_string()]);
        assert_eq!(store.get("slot"), Some(&Value::text("b-booking")));
    }

    #[test]
    fn read_only_transaction_validates_but_applies_nothing() {
        let mut store = store_with("doc", "v1");
        let mut txn = store.begin("reader");
        assert_eq!(txn.read("doc"), Some(Value::text("v1")));
        let receipt = txn.commit(&mut store).unwrap();
        assert!(receipt.applied.is_empty());
        assert_eq!(store.version("doc"), 1);
    }

    #[test]
    fn overlay_is_invisible_until_commit() {
        let mut store = store_with("doc", "v1");
        let mut txn = store.begin("writer");
        txn.write("doc", Value::text("draft"));
        // A second agent beginning now still sees committed state only.
        let mut other = store.begin("other");
        assert_eq!(other.read("doc"), Some(Value::text("v1")));
        txn.commit(&mut store).unwrap();
        assert_eq!(store.get("doc"), Some(&Value::text("draft")));
    }

    #[test]
    fn backoff_doubles_then_caps_and_escalates_past_budget() {
        let policy = RetryPolicy { base_delay_ms: 100, max_delay_ms: 1000, max_retries: 3 };
        assert_eq!(policy.delay_for(0), 100);
        assert_eq!(policy.delay_for(1), 200);
        assert_eq!(policy.delay_for(2), 400);
        assert_eq!(policy.delay_for(3), 800);
        assert_eq!(policy.delay_for(4), 1000, "capped, not 1600");
        assert_eq!(policy.delay_for(30), 1000, "saturating, never overflows");
        assert!(matches!(policy.decision(0), RetryDecision::Retry { delay_ms: 100 }));
        assert!(matches!(policy.decision(2), RetryDecision::Retry { delay_ms: 400 }));
        assert_eq!(policy.decision(3), RetryDecision::Escalate);
    }

    #[test]
    fn retry_after_conflict_rereads_and_commits() {
        let mut store = store_with("counter", "0");
        let policy = RetryPolicy { base_delay_ms: 50, max_delay_ms: 1000, max_retries: 3 };
        let outcome = run_with_retry(
            &mut store,
            "counter-agent",
            &policy,
            |txn| {
                // Decide from a fresh read each attempt: bump the counter.
                let cur = txn.read("counter").and_then(|v| v.as_text().map(|s| s.to_string()));
                let n: i64 = cur.as_deref().unwrap_or("0").parse().unwrap_or(0);
                txn.write("counter", Value::text((n + 1).to_string()));
            },
            // Only the first attempt contends: another agent commits first.
            |store, attempt| {
                if attempt == 0 {
                    let mut rival = store.begin("rival");
                    rival.write("counter", Value::text("99"));
                    rival.commit(store).unwrap();
                }
            },
        );
        match outcome {
            RetryOutcome::Committed { attempts, delays_ms, .. } => {
                assert_eq!(attempts, 2);
                assert_eq!(delays_ms, vec![50]);
                // Second attempt re-read "99" and bumped to 100.
                assert_eq!(store.get("counter"), Some(&Value::text("100")));
            }
            RetryOutcome::Escalated { .. } => panic!("should have committed on retry"),
        }
    }

    #[test]
    fn permanent_contention_escalates_and_applies_nothing() {
        let mut store = store_with("doc", "v1");
        let policy = RetryPolicy { base_delay_ms: 100, max_delay_ms: 1000, max_retries: 2 };
        let outcome = run_with_retry(
            &mut store,
            "agent",
            &policy,
            |txn| {
                txn.read("doc");
                txn.write("doc", Value::text("mine"));
            },
            // Every attempt contends: the rival always wins the race.
            |store, _| {
                let mut rival = store.begin("rival");
                rival.write("doc", Value::text("rival"));
                rival.commit(store).unwrap();
            },
        );
        match outcome {
            RetryOutcome::Escalated { attempts, delays_ms, conflict } => {
                assert_eq!(attempts, 3, "initial try + 2 retries");
                assert_eq!(delays_ms, vec![100, 200]);
                assert_eq!(conflict.conflicting_keys, vec!["doc".to_string()]);
                assert!(
                    RetryOutcome::escalation_reason(&conflict).contains("doc"),
                    "yield prompt names the contested key"
                );
                // The store holds only rival commits, never the agent's edit.
                assert_eq!(store.get("doc"), Some(&Value::text("rival")));
            }
            RetryOutcome::Committed { .. } => panic!("permanent contention must escalate"),
        }
    }

    #[test]
    fn uncontended_run_commits_first_try_with_no_delays() {
        let mut store = store_with("doc", "v1");
        let policy = RetryPolicy { base_delay_ms: 100, max_delay_ms: 1000, max_retries: 3 };
        let outcome = run_with_retry(
            &mut store,
            "agent",
            &policy,
            |txn| {
                txn.write("note", Value::text("hello"));
            },
            |_, _| {},
        );
        match outcome {
            RetryOutcome::Committed { attempts, delays_ms, receipt } => {
                assert_eq!(attempts, 1);
                assert!(delays_ms.is_empty());
                assert_eq!(receipt.applied, vec![("note".to_string(), 1)]);
            }
            RetryOutcome::Escalated { .. } => panic!("nothing contended"),
        }
    }
}
