//! The memory subsystem (doc ch. 10): L0 working set, L1 episodic, L2
//! semantic, L3 cold archive — guarded by three non-negotiable rules:
//!
//! 1. **No provenance, no persistence.** Every stored fact carries who said
//!    it, where it came from, when, and with what confidence. `mem_assert`
//!    without provenance is rejected outright.
//! 2. **Confidence decays.** Facts have a half-life; stale L2 facts demote
//!    back to L1 instead of quietly staying "true" forever.
//! 3. **Memory is auditable and deletable.** The user can review everything
//!    the system remembers and strike it (L3 is append-only by design; the
//!    rest is user-deletable).
//!
//! Poisoning defenses (doc ch. 10): content from untrusted channels can
//! only ever land in L1 — promotion to L2 requires a trusted source
//! (user-direct or system-signed), a decayed confidence ≥ 0.7, and, for
//! high-risk categories (permission / payment / identity), corroboration
//! by two independent sources.

use crate::taint::TrustLabel;
use crate::Clock;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

/// Where a fact came from. Determines both its taint label and whether it
/// may ever be promoted to L2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceKind {
    UserDirect,
    SystemSigned,
    WebFetch,
    Mail,
    ThirdPartyApi,
    AgentDerived,
}

impl SourceKind {
    pub fn label(self) -> TrustLabel {
        match self {
            SourceKind::UserDirect => TrustLabel::User,
            SourceKind::SystemSigned => TrustLabel::Verified,
            _ => TrustLabel::Untrusted,
        }
    }

    /// Only verified channels may write L2 (doc ch. 10, write whitelist).
    pub fn may_promote_to_l2(self) -> bool {
        matches!(self, SourceKind::UserDirect | SourceKind::SystemSigned)
    }
}

/// Mandatory provenance attached to every stored fact.
#[derive(Debug, Clone, PartialEq)]
pub struct Provenance {
    pub source_kind: SourceKind,
    pub source_id: String,
    pub at_ms: u64,
    pub initial_confidence: f64,
}

impl Provenance {
    pub fn new(source_kind: SourceKind, source_id: impl Into<String>, at_ms: u64, conf: f64) -> Self {
        Provenance {
            source_kind,
            source_id: source_id.into(),
            at_ms,
            initial_confidence: conf.clamp(0.0, 1.0),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FactCategory {
    General,
    Preference,
    Permission,
    Payment,
    Identity,
}

impl FactCategory {
    /// High-risk facts need two independent sources before L2 (doc ch. 10).
    pub fn is_high_risk(self) -> bool {
        matches!(self, FactCategory::Permission | FactCategory::Payment | FactCategory::Identity)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    L0,
    L1,
    L2,
    L3,
}

#[derive(Debug, Clone)]
pub struct MemoryItem {
    pub id: u64,
    pub key: String,
    pub fact: String,
    pub category: FactCategory,
    pub provenance: Provenance,
    pub level: Level,
    pub expires_ms: Option<u64>,
    pub recall_count: u32,
    /// Distinct source ids that have asserted the same key+fact.
    pub corroborations: HashSet<String>,
}

impl MemoryItem {
    /// Confidence after exponential half-life decay (doc ch. 10, rule 2).
    pub fn decayed_confidence(&self, now_ms: u64, half_life_ms: u64) -> f64 {
        if now_ms <= self.provenance.at_ms || half_life_ms == 0 {
            return self.provenance.initial_confidence;
        }
        let elapsed = (now_ms - self.provenance.at_ms) as f64;
        let half_lives = elapsed / half_life_ms as f64;
        self.provenance.initial_confidence * 0.5f64.powf(half_lives)
    }

    pub fn is_expired(&self, now_ms: u64) -> bool {
        self.expires_ms.map(|e| now_ms >= e).unwrap_or(false)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryError {
    MissingProvenance,
    NotFound { id: u64 },
    PromotionDenied { reason: String },
    Expired { id: u64 },
    Archived { id: u64 },
}

impl std::fmt::Display for MemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryError::MissingProvenance => write!(f, "memory write rejected: no provenance"),
            MemoryError::NotFound { id } => write!(f, "memory item {id} not found"),
            MemoryError::PromotionDenied { reason } => write!(f, "promotion to L2 denied: {reason}"),
            MemoryError::Expired { id } => write!(f, "memory item {id} is expired"),
            MemoryError::Archived { id } => write!(f, "memory item {id} is in the append-only L3 archive"),
        }
    }
}
impl std::error::Error for MemoryError {}

#[derive(Debug, Clone)]
pub struct RecalledItem {
    pub item: MemoryItem,
    pub relevance: f64,
    /// relevance × decayed confidence, trust-weighted for ranking.
    pub effective_confidence: f64,
    /// The label callers must display when this memory enters a prompt
    /// (recall-time trust annotation, doc ch. 10).
    pub label: TrustLabel,
}

#[derive(Debug, Clone)]
pub struct Recall {
    pub items: Vec<RecalledItem>,
    /// Best effective confidence found (0 when nothing matched). Recall is
    /// probabilistic: the caller must look at this number (doc ch. 8).
    pub confidence: f64,
    /// True when confidence is below the threshold — the correct behavior
    /// is to escalate to a `yield` and ask the human.
    pub needs_yield: bool,
}

#[derive(Debug, Clone)]
pub struct HealthFlag {
    pub id: u64,
    pub issue: String,
}

/// Confidence below which recall must escalate to a human (doc ch. 8).
pub const RECALL_YIELD_THRESHOLD: f64 = 0.5;
/// Minimum decayed confidence for L2 promotion.
pub const PROMOTION_MIN_CONFIDENCE: f64 = 0.7;
/// L2 facts decayed below this demote back to L1.
pub const DEMOTION_CONFIDENCE: f64 = 0.2;

pub struct Memory {
    clock: Rc<dyn Clock>,
    half_life_ms: u64,
    next_id: u64,
    items: HashMap<u64, MemoryItem>,
    working_set: Vec<String>,
}

fn tokenize(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_string())
        .collect()
}

/// Word-overlap relevance in [0, 1]: fraction of query tokens present in
/// the fact's key or text. Deterministic by construction.
fn relevance(query: &str, item: &MemoryItem) -> f64 {
    let q = tokenize(query);
    if q.is_empty() {
        return 0.0;
    }
    let hay: HashSet<String> = tokenize(&format!("{} {}", item.key, item.fact)).into_iter().collect();
    let hits = q.iter().filter(|t| hay.contains(*t)).count();
    hits as f64 / q.len() as f64
}

fn trust_weight(label: TrustLabel) -> f64 {
    match label {
        TrustLabel::User => 1.0,
        TrustLabel::Verified => 0.9,
        TrustLabel::Untrusted => 0.6,
    }
}

impl Memory {
    pub fn new(clock: Rc<dyn Clock>, half_life_ms: u64) -> Self {
        Memory { clock, half_life_ms, next_id: 1, items: HashMap::new(), working_set: Vec::new() }
    }

    // -- L0 working set: ephemeral, auto-written, freely discardable. -----
    pub fn l0_push(&mut self, note: impl Into<String>) {
        self.working_set.push(note.into());
    }
    pub fn l0_clear(&mut self) {
        self.working_set.clear();
    }
    pub fn l0(&self) -> &[String] {
        &self.working_set
    }

    /// `mem_assert`: write a fact to L1. Provenance is *mandatory* — pass
    /// `None` and the write is rejected (no provenance, no persistence).
    /// Asserting the same key+fact again from another source corroborates
    /// the existing item instead of duplicating it.
    pub fn assert_fact(
        &mut self,
        key: &str,
        fact: &str,
        category: FactCategory,
        provenance: Option<Provenance>,
        ttl_ms: Option<u64>,
    ) -> Result<u64, MemoryError> {
        let prov = provenance.ok_or(MemoryError::MissingProvenance)?;
        if let Some(existing) = self
            .items
            .values_mut()
            .find(|i| i.key == key && i.fact == fact && i.level != Level::L3)
        {
            existing.corroborations.insert(prov.source_id.clone());
            return Ok(existing.id);
        }
        let now = self.clock.now_ms();
        let id = self.next_id;
        self.next_id += 1;
        let mut corroborations = HashSet::new();
        corroborations.insert(prov.source_id.clone());
        self.items.insert(
            id,
            MemoryItem {
                id,
                key: key.to_string(),
                fact: fact.to_string(),
                category,
                provenance: prov,
                level: Level::L1,
                expires_ms: ttl_ms.map(|t| now + t),
                recall_count: 0,
                corroborations,
            },
        );
        Ok(id)
    }

    /// Promote an L1 fact to L2 (semantic memory). Enforces the doc ch. 10
    /// gates: trusted channel only, confidence threshold, dual-source
    /// corroboration for high-risk categories.
    pub fn promote(&mut self, id: u64) -> Result<(), MemoryError> {
        let now = self.clock.now_ms();
        let item = self.items.get(&id).ok_or(MemoryError::NotFound { id })?;
        if item.is_expired(now) {
            return Err(MemoryError::Expired { id });
        }
        if item.level != Level::L1 {
            return Err(MemoryError::PromotionDenied { reason: format!("item is {:?}, not L1", item.level) });
        }
        if !item.provenance.source_kind.may_promote_to_l2() {
            return Err(MemoryError::PromotionDenied {
                reason: "untrusted channels may only write L1 (memory-poisoning defense)".to_string(),
            });
        }
        if item.decayed_confidence(now, self.half_life_ms) < PROMOTION_MIN_CONFIDENCE {
            return Err(MemoryError::PromotionDenied {
                reason: format!("decayed confidence below {PROMOTION_MIN_CONFIDENCE}"),
            });
        }
        if item.category.is_high_risk() && item.corroborations.len() < 2 {
            return Err(MemoryError::PromotionDenied {
                reason: "high-risk fact needs two independent sources".to_string(),
            });
        }
        self.items.get_mut(&id).expect("checked").level = Level::L2;
        Ok(())
    }

    /// Move an L2 fact into the append-only L3 cold archive.
    pub fn archive(&mut self, id: u64) -> Result<(), MemoryError> {
        let item = self.items.get_mut(&id).ok_or(MemoryError::NotFound { id })?;
        item.level = Level::L3;
        Ok(())
    }

    /// `mem_recall`: semantic recall returning items *and* a confidence.
    /// Searching covers L1+L2 (plus L3 when `include_archive` is set).
    /// Results are trust-weighted; expired items are invisible.
    pub fn recall(&mut self, query: &str, min_relevance: f64, include_archive: bool) -> Recall {
        let now = self.clock.now_ms();
        let mut found: Vec<RecalledItem> = Vec::new();
        let mut ids: Vec<u64> = Vec::new();
        for item in self.items.values() {
            let level_ok = matches!(item.level, Level::L1 | Level::L2)
                || (include_archive && item.level == Level::L3);
            if !level_ok || item.is_expired(now) {
                continue;
            }
            let rel = relevance(query, item);
            if rel < min_relevance || rel == 0.0 {
                continue;
            }
            let label = item.provenance.source_kind.label();
            let effective =
                rel * item.decayed_confidence(now, self.half_life_ms) * trust_weight(label);
            ids.push(item.id);
            found.push(RecalledItem {
                item: item.clone(),
                relevance: rel,
                effective_confidence: effective,
                label,
            });
        }
        for id in ids {
            if let Some(i) = self.items.get_mut(&id) {
                i.recall_count += 1;
            }
        }
        found.sort_by(|a, b| {
            b.effective_confidence
                .partial_cmp(&a.effective_confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let confidence = found.first().map(|r| r.effective_confidence).unwrap_or(0.0);
        Recall { items: found, confidence, needs_yield: confidence < RECALL_YIELD_THRESHOLD }
    }

    /// User-initiated deletion ("memory must be deletable"). L3 is an
    /// append-only archive, so deletion there is refused by design.
    pub fn forget(&mut self, id: u64) -> Result<(), MemoryError> {
        match self.items.get(&id) {
            None => Err(MemoryError::NotFound { id }),
            Some(i) if i.level == Level::L3 => Err(MemoryError::Archived { id }),
            Some(_) => {
                self.items.remove(&id);
                Ok(())
            }
        }
    }

    /// Full audit view: everything remembered, with provenance — the
    /// "credit report" of doc ch. 10, rule 3.
    pub fn audit(&self) -> Vec<&MemoryItem> {
        let mut all: Vec<&MemoryItem> = self.items.values().collect();
        all.sort_by_key(|i| i.id);
        all
    }

    /// Demote L2 facts whose decayed confidence fell below the floor.
    /// Returns how many were demoted.
    pub fn demote_stale(&mut self) -> usize {
        let now = self.clock.now_ms();
        let mut n = 0;
        for item in self.items.values_mut() {
            if item.level == Level::L2
                && item.decayed_confidence(now, self.half_life_ms) < DEMOTION_CONFIDENCE
            {
                item.level = Level::L1;
                n += 1;
            }
        }
        n
    }

    /// Periodic memory health check (doc ch. 10): flags L2 facts that are
    /// stale, isolated (never recalled), or contradicted by another L2
    /// fact with the same key.
    pub fn health_scan(&self) -> Vec<HealthFlag> {
        let now = self.clock.now_ms();
        let mut flags = Vec::new();
        let l2: Vec<&MemoryItem> =
            self.items.values().filter(|i| i.level == Level::L2).collect();
        for item in &l2 {
            if item.decayed_confidence(now, self.half_life_ms) < DEMOTION_CONFIDENCE {
                flags.push(HealthFlag { id: item.id, issue: "stale: confidence below demotion floor".into() });
            }
            if item.recall_count == 0 {
                flags.push(HealthFlag { id: item.id, issue: "isolated: never recalled".into() });
            }
            if l2.iter().any(|o| o.id != item.id && o.key == item.key && o.fact != item.fact) {
                flags.push(HealthFlag { id: item.id, issue: "contradicted by another L2 fact".into() });
            }
        }
        flags
    }

    pub fn get(&self, id: u64) -> Option<&MemoryItem> {
        self.items.get(&id)
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ManualClock;

    const DAY: u64 = 86_400_000;

    fn setup() -> (Memory, Rc<ManualClock>) {
        let clock = Rc::new(ManualClock::new(1_000_000_000));
        // Half-life of 30 days.
        (Memory::new(clock.clone(), 30 * DAY), clock)
    }

    fn user_prov(conf: f64, at: u64) -> Provenance {
        Provenance::new(SourceKind::UserDirect, "user", at, conf)
    }

    #[test]
    fn write_without_provenance_is_rejected() {
        let (mut mem, _c) = setup();
        let err = mem.assert_fact("seat", "user likes window seats", FactCategory::Preference, None, None);
        assert_eq!(err, Err(MemoryError::MissingProvenance));
        assert!(mem.is_empty());
    }

    #[test]
    fn fact_lands_in_l1_with_provenance() {
        let (mut mem, c) = setup();
        let id = mem
            .assert_fact("seat", "user likes window seats", FactCategory::Preference, Some(prov(&c, 0.9)), None)
            .unwrap();
        let item = mem.get(id).unwrap();
        assert_eq!(item.level, Level::L1);
        assert_eq!(item.provenance.source_id, "user");
    }

    fn prov(c: &ManualClock, conf: f64) -> Provenance {
        user_prov(conf, c.now_ms())
    }

    #[test]
    fn trusted_fact_promotes_to_l2() {
        let (mut mem, c) = setup();
        let id = mem
            .assert_fact("seat", "user likes window seats", FactCategory::Preference, Some(prov(&c, 0.95)), None)
            .unwrap();
        mem.promote(id).unwrap();
        assert_eq!(mem.get(id).unwrap().level, Level::L2);
    }

    #[test]
    fn web_fact_can_never_promote_to_l2() {
        // The memory-poisoning scenario of doc ch. 10: a web page says
        // "the user authorized transfers" — it can sit in L1, never L2.
        let (mut mem, c) = setup();
        let web = Provenance::new(SourceKind::WebFetch, "evil.example", c.now_ms(), 0.99);
        let id = mem
            .assert_fact("transfer.auth", "user authorized transfers to X", FactCategory::Permission, Some(web), None)
            .unwrap();
        let err = mem.promote(id).unwrap_err();
        assert!(matches!(err, MemoryError::PromotionDenied { .. }));
        assert_eq!(mem.get(id).unwrap().level, Level::L1);
    }

    #[test]
    fn low_confidence_fact_cannot_promote() {
        let (mut mem, c) = setup();
        let id = mem
            .assert_fact("seat", "user maybe likes window seats", FactCategory::Preference, Some(prov(&c, 0.5)), None)
            .unwrap();
        assert!(mem.promote(id).is_err());
    }

    #[test]
    fn confidence_decays_with_half_life() {
        let (mut mem, c) = setup();
        let id = mem
            .assert_fact("seat", "window seat", FactCategory::Preference, Some(prov(&c, 0.8)), None)
            .unwrap();
        c.advance(30 * DAY); // one half-life
        let conf = mem.get(id).unwrap().decayed_confidence(c.now_ms(), 30 * DAY);
        assert!((conf - 0.4).abs() < 1e-9);
    }

    #[test]
    fn stale_l2_fact_demotes_back_to_l1() {
        let (mut mem, c) = setup();
        let id = mem
            .assert_fact("seat", "window seat", FactCategory::Preference, Some(prov(&c, 0.75)), None)
            .unwrap();
        mem.promote(id).unwrap();
        c.advance(90 * DAY); // three half-lives: 0.75 -> ~0.09
        assert_eq!(mem.demote_stale(), 1);
        assert_eq!(mem.get(id).unwrap().level, Level::L1);
    }

    #[test]
    fn high_risk_fact_needs_two_independent_sources() {
        let (mut mem, c) = setup();
        let id = mem
            .assert_fact("pay.limit", "daily limit 1000", FactCategory::Payment, Some(prov(&c, 0.95)), None)
            .unwrap();
        assert!(mem.promote(id).is_err(), "one source is not enough for payment facts");
        // Second, independent source asserts the same fact.
        let signed = Provenance::new(SourceKind::SystemSigned, "bank-api", c.now_ms(), 0.95);
        let id2 = mem
            .assert_fact("pay.limit", "daily limit 1000", FactCategory::Payment, Some(signed), None)
            .unwrap();
        assert_eq!(id, id2, "corroboration should fold into the same item");
        mem.promote(id).unwrap();
        assert_eq!(mem.get(id).unwrap().level, Level::L2);
    }

    #[test]
    fn recall_returns_confidence_and_flags_low_confidence_for_yield() {
        let (mut mem, c) = setup();
        mem.assert_fact("seat pref", "user likes window seats on flights", FactCategory::Preference, Some(prov(&c, 0.95)), None)
            .unwrap();
        let r = mem.recall("window seats", 0.1, false);
        assert_eq!(r.items.len(), 1);
        assert!(r.confidence > 0.5);
        assert!(!r.needs_yield);
        let empty = mem.recall("quantum chromodynamics", 0.1, false);
        assert!(empty.items.is_empty());
        assert_eq!(empty.confidence, 0.0);
        assert!(empty.needs_yield, "knowing that we don't know must escalate");
    }

    #[test]
    fn recall_ranks_trusted_above_untrusted_at_equal_relevance() {
        let (mut mem, c) = setup();
        let web = Provenance::new(SourceKind::WebFetch, "blog", c.now_ms(), 0.95);
        mem.assert_fact("hotel", "user prefers hotel brand A", FactCategory::Preference, Some(web), None)
            .unwrap();
        mem.assert_fact("hotel", "user prefers hotel brand A downtown", FactCategory::Preference, Some(prov(&c, 0.95)), None)
            .unwrap();
        let r = mem.recall("hotel brand", 0.1, false);
        assert_eq!(r.items.len(), 2);
        assert_eq!(r.items[0].label, TrustLabel::User);
        assert_eq!(r.items[1].label, TrustLabel::Untrusted);
    }

    #[test]
    fn expired_facts_are_invisible_to_recall() {
        let (mut mem, c) = setup();
        mem.assert_fact("gate", "gate is B7", FactCategory::General, Some(prov(&c, 0.9)), Some(1_000))
            .unwrap();
        c.advance(2_000);
        let r = mem.recall("gate", 0.1, false);
        assert!(r.items.is_empty());
    }

    #[test]
    fn user_can_forget_but_archive_is_append_only() {
        let (mut mem, c) = setup();
        let id = mem
            .assert_fact("seat", "window seat", FactCategory::Preference, Some(prov(&c, 0.95)), None)
            .unwrap();
        mem.promote(id).unwrap();
        mem.archive(id).unwrap();
        assert_eq!(mem.forget(id), Err(MemoryError::Archived { id }));
        let id2 = mem
            .assert_fact("meal", "no peanuts", FactCategory::Preference, Some(prov(&c, 0.9)), None)
            .unwrap();
        mem.forget(id2).unwrap();
        assert!(mem.get(id2).is_none());
    }

    #[test]
    fn health_scan_flags_contradictions() {
        let (mut mem, c) = setup();
        let a = mem
            .assert_fact("home.airport", "home airport is PVG", FactCategory::Preference, Some(prov(&c, 0.95)), None)
            .unwrap();
        let b = mem
            .assert_fact("home.airport", "home airport is SHA", FactCategory::Preference, Some(prov(&c, 0.95)), None)
            .unwrap();
        mem.promote(a).unwrap();
        mem.promote(b).unwrap();
        let flags = mem.health_scan();
        assert!(flags.iter().any(|f| f.issue.contains("contradicted")));
    }

    #[test]
    fn l0_working_set_is_ephemeral_scratch() {
        let (mut mem, _c) = setup();
        mem.l0_push("current prompt tokens");
        assert_eq!(mem.l0().len(), 1);
        mem.l0_clear();
        assert!(mem.l0().is_empty());
        assert!(mem.is_empty(), "L0 notes are not stored facts");
    }
}
