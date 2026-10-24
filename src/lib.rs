//! Loom — an AI-native operating system architecture prototype.
//!
//! Design rule zero (doc ch. 5, axiom I): *the model proposes, the kernel
//! executes*. Everything in this crate hangs off that split: probabilistic
//! components (planner, memory recall, speculation) only ever produce data;
//! every side effect flows through the deterministic kernel, which checks a
//! signed capability, an effect type, and a transaction before anything in
//! the outside world changes.
//!
//! The crate has no external dependencies and no network access. Time is an
//! injected [`Clock`] so every time-based rule (capability leases, memory
//! half-lives, cooling-off periods) is deterministically testable.

pub mod attention;
pub mod capabilities;
pub mod crypto;
pub mod memory;
pub mod schedulers;
pub mod speculation;
pub mod taint;
pub mod tools;
pub mod transactions;
pub mod typesys;
pub mod watchdog;
pub mod worldcache;

use std::cell::Cell;
use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Clock
// ---------------------------------------------------------------------------

/// Monotonic-ish millisecond clock. All time-based kernel rules read time
/// exclusively through this trait.
pub trait Clock {
    fn now_ms(&self) -> u64;
}

/// Deterministic clock for tests and simulations: time only moves when
/// explicitly advanced.
#[derive(Debug, Default)]
pub struct ManualClock {
    now: Cell<u64>,
}

impl ManualClock {
    pub fn new(start_ms: u64) -> Self {
        ManualClock { now: Cell::new(start_ms) }
    }
    pub fn advance(&self, ms: u64) {
        self.now.set(self.now.get() + ms);
    }
    pub fn set(&self, ms: u64) {
        self.now.set(ms);
    }
}

impl Clock for ManualClock {
    fn now_ms(&self) -> u64 {
        self.now.get()
    }
}

/// Wall-clock time source for the demo binary.
#[derive(Debug, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_ms(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Value: the one data type that flows through plans, tools and memory.
// ---------------------------------------------------------------------------

/// A dynamically-typed value: tool arguments, tool results, memory facts and
/// macro slot fillings are all `Value`s. Maps use `BTreeMap` so the canonical
/// encoding (and therefore every hash/fingerprint derived from it) is stable.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Text(String),
    List(Vec<Value>),
    Map(BTreeMap<String, Value>),
}

impl Value {
    pub fn text(s: impl Into<String>) -> Value {
        Value::Text(s.into())
    }

    pub fn int(i: i64) -> Value {
        Value::Int(i)
    }

    /// Build a map from key/value pairs.
    pub fn map(pairs: Vec<(&str, Value)>) -> Value {
        Value::Map(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            Value::Text(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            _ => None,
        }
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Map(m) => m.get(key),
            _ => None,
        }
    }

    /// Read a nested value by dot-separated path of map keys
    /// (e.g. `passenger.seat`). Lists are not traversed.
    pub fn get_path(&self, path: &[&str]) -> Option<&Value> {
        let mut cur = self;
        for key in path {
            cur = cur.get(key)?;
        }
        Some(cur)
    }

    /// Write a nested value by path, creating intermediate maps as needed.
    /// Returns false if an existing non-map value blocks the path.
    pub fn set_path(&mut self, path: &[&str], val: Value) -> bool {
        if path.is_empty() {
            *self = val;
            return true;
        }
        match self {
            Value::Map(m) => {
                let entry = m.entry(path[0].to_string()).or_insert(Value::Null);
                if path.len() == 1 {
                    *entry = val;
                    true
                } else {
                    if matches!(entry, Value::Null) {
                        *entry = Value::Map(BTreeMap::new());
                    }
                    entry.set_path(&path[1..], val)
                }
            }
            _ => false,
        }
    }

    /// Deterministic canonical encoding. Two values are structurally equal
    /// iff their canonical encodings are equal, which is what makes it safe
    /// to hash for fingerprints, idempotency keys and ledger chaining.
    pub fn canonical(&self) -> String {
        match self {
            Value::Null => "null".to_string(),
            Value::Bool(b) => format!("bool:{b}"),
            Value::Int(i) => format!("int:{i}"),
            Value::Text(s) => format!("text:{}:{}", s.len(), s),
            Value::List(items) => {
                let inner: Vec<String> = items.iter().map(|v| v.canonical()).collect();
                format!("list:[{}]", inner.join(","))
            }
            Value::Map(m) => {
                let inner: Vec<String> =
                    m.iter().map(|(k, v)| format!("{k}={}", v.canonical())).collect();
                format!("map:{{{}}}", inner.join(","))
            }
        }
    }

    /// Collect the dot-paths of all scalar leaves (used by the
    /// determinization engine to discover which argument positions vary
    /// across runs and must become macro slots).
    pub fn leaf_paths(&self) -> Vec<String> {
        let mut out = Vec::new();
        self.collect_leaf_paths("", &mut out);
        out
    }

    fn collect_leaf_paths(&self, prefix: &str, out: &mut Vec<String>) {
        match self {
            Value::Map(m) => {
                for (k, v) in m {
                    let p = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
                    v.collect_leaf_paths(&p, out);
                }
            }
            Value::List(items) => {
                for (i, v) in items.iter().enumerate() {
                    let p = format!("{prefix}[{i}]");
                    v.collect_leaf_paths(&p, out);
                }
            }
            _ => out.push(prefix.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manual_clock_only_moves_when_told() {
        let c = ManualClock::new(1_000);
        assert_eq!(c.now_ms(), 1_000);
        c.advance(250);
        assert_eq!(c.now_ms(), 1_250);
        c.set(42);
        assert_eq!(c.now_ms(), 42);
    }

    #[test]
    fn canonical_encoding_is_order_independent_for_maps() {
        let a = Value::map(vec![("b", Value::int(2)), ("a", Value::int(1))]);
        let b = Value::map(vec![("a", Value::int(1)), ("b", Value::int(2))]);
        assert_eq!(a.canonical(), b.canonical());
    }

    #[test]
    fn canonical_encoding_distinguishes_types() {
        assert_ne!(Value::int(1).canonical(), Value::text("1").canonical());
        assert_ne!(Value::Bool(true).canonical(), Value::int(1).canonical());
    }

    #[test]
    fn path_get_and_set_round_trip() {
        let mut v = Value::map(vec![("passenger", Value::map(vec![("seat", Value::text("window"))]))]);
        assert_eq!(v.get_path(&["passenger", "seat"]), Some(&Value::text("window")));
        assert!(v.set_path(&["passenger", "seat"], Value::text("aisle")));
        assert_eq!(v.get_path(&["passenger", "seat"]), Some(&Value::text("aisle")));
        assert!(v.set_path(&["meta", "source"], Value::text("test")));
        assert_eq!(v.get_path(&["meta", "source"]), Some(&Value::text("test")));
    }

    #[test]
    fn leaf_paths_cover_scalars() {
        let v = Value::map(vec![
            ("a", Value::int(1)),
            ("b", Value::map(vec![("c", Value::text("x"))])),
        ]);
        let mut paths = v.leaf_paths();
        paths.sort();
        assert_eq!(paths, vec!["a".to_string(), "b.c".to_string()]);
    }
}
