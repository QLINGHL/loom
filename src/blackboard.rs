//! The blackboard (doc ch. 17): the shared fact base agents use instead of
//! messaging each other directly.
//!
//! Why a blackboard and not messages: N agents messaging each other needs
//! N² connections, and every pair has to agree on a protocol. A blackboard
//! is one shared place: an agent posts a structured fact ("this supplier's
//! payment terms are 60 days"), any other agent reads it, and nobody needs
//! to know who else is listening.
//!
//! The three rules from the design doc, enforced here by construction:
//!
//! * **Append-only, never edited.** [`Blackboard::entries`] is only ever
//!   pushed to, never mutated in place. There is no update or delete API.
//!   A fact, once posted, stays readable forever with its original value,
//!   author, provenance and trust label.
//! * **Supersede by appending, not by overwriting.** To correct a fact, an
//!   agent posts a new fact whose [`FactEntry::supersedes`] points at the
//!   old entry's id. Whether an entry is still live is *derived* — an entry
//!   is superseded iff some later entry points at it — so correcting a
//!   fact cannot rewrite the record of what was believed before. Any
//!   moment's reasoning stays auditable through [`Blackboard::lineage`]:
//!   follow `supersedes` links back to the original fact.
//! * **Every fact carries provenance and a trust label.** Posting reuses
//!   [`Provenance`] from the memory subsystem (doc ch. 10: no provenance,
//!   no persistence), so an empty source id is rejected outright. The
//!   trust label is derived from the provenance's source kind at post
//!   time (user-direct → [`TrustLabel::User`], system-signed →
//!   [`TrustLabel::Verified`], everything fetched →
//!   [`TrustLabel::Untrusted`]) and stored on the entry, so a reader can
//!   filter by trust without re-deriving it — and a later change to the
//!   derivation rules can never rewrite posted history.
//!
//! Supersede validation keeps the history a set of chains, never a fork:
//! only a *live* entry may be superseded, and only by a fact with the
//! *same key*. Superseding an already-superseded entry, an unknown id, or
//! a fact under a different key is rejected and posts nothing, so a
//! failed correction can never leave a half-written chain behind.
//!
//! This is the sharing half of ch. 17 only. The concurrency half —
//! optimistic commit validation when two agents write at once — lives in
//! [`crate::concurrency`]; the blackboard itself is single-threaded like
//! the rest of this prototype, and interleaved agents are modelled by
//! sequential posts in tests.

use crate::memory::Provenance;
use crate::taint::TrustLabel;
use crate::Value;

/// One posted fact. Immutable once posted: the blackboard never mutates
/// an entry, it only appends new ones.
#[derive(Debug, Clone, PartialEq)]
pub struct FactEntry {
    /// Monotonically increasing id, assigned at post time (starts at 1).
    pub id: u64,
    /// Subject of the fact, e.g. `"supplier.payment_terms"`. A superseding
    /// fact must carry the same key as the fact it replaces.
    pub key: String,
    /// The structured fact itself.
    pub value: Value,
    /// Which agent posted it.
    pub author: String,
    /// Who said it, where it came from, when, with what confidence.
    pub provenance: Provenance,
    /// Trust label derived from `provenance.source_kind` at post time.
    pub label: TrustLabel,
    /// Id of the fact this entry replaces, if any. `None` for a fresh fact.
    pub supersedes: Option<u64>,
}

impl FactEntry {
    /// True when no later entry points at this one, i.e. this is still
    /// the board's current word on its key.
    pub fn is_live(&self, board: &Blackboard) -> bool {
        board.is_live(self.id)
    }
}

/// Rejection reasons for a post. A rejected post changes nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlackboardError {
    /// The fact key was empty: a keyless fact cannot be found again.
    EmptyKey,
    /// The provenance named no source: no provenance, no persistence
    /// (doc ch. 10, applied to the board as well as to memory).
    EmptySource,
    /// The fact to supersede does not exist on this board.
    UnknownSuperseded { id: u64 },
    /// The fact to supersede is itself already superseded: superseding
    /// it again would fork the history chain.
    AlreadySuperseded { id: u64 },
    /// A superseding fact must carry the same key as the fact it
    /// replaces, otherwise the chain would jump subjects mid-history.
    KeyMismatch { superseded_key: String, new_key: String },
}

impl std::fmt::Display for BlackboardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BlackboardError::EmptyKey => write!(f, "blackboard post rejected: empty key"),
            BlackboardError::EmptySource => {
                write!(f, "blackboard post rejected: provenance has no source")
            }
            BlackboardError::UnknownSuperseded { id } => {
                write!(f, "blackboard post rejected: no fact with id {id} to supersede")
            }
            BlackboardError::AlreadySuperseded { id } => {
                write!(f, "blackboard post rejected: fact {id} is already superseded")
            }
            BlackboardError::KeyMismatch { superseded_key, new_key } => write!(
                f,
                "blackboard post rejected: superseding fact key {new_key} does not match superseded fact key {superseded_key}"
            ),
        }
    }
}

impl std::error::Error for BlackboardError {}

/// The board itself: an append-only log of [`FactEntry`]s.
#[derive(Debug, Default)]
pub struct Blackboard {
    entries: Vec<FactEntry>,
    next_id: u64,
}

impl Blackboard {
    pub fn new() -> Self {
        Blackboard { entries: Vec::new(), next_id: 1 }
    }

    /// Post a fresh fact (one that supersedes nothing). Returns its id.
    pub fn post(
        &mut self,
        author: &str,
        key: &str,
        value: Value,
        provenance: Provenance,
    ) -> Result<u64, BlackboardError> {
        self.post_inner(author, key, value, provenance, None)
    }

    /// Post a fact that replaces `supersedes`: the new fact becomes the
    /// live fact for the key, the old one stays readable in history.
    pub fn post_superseding(
        &mut self,
        author: &str,
        key: &str,
        value: Value,
        provenance: Provenance,
        supersedes: u64,
    ) -> Result<u64, BlackboardError> {
        let target = self
            .entries
            .iter()
            .find(|e| e.id == supersedes)
            .ok_or(BlackboardError::UnknownSuperseded { id: supersedes })?;
        if !self.is_live(supersedes) {
            return Err(BlackboardError::AlreadySuperseded { id: supersedes });
        }
        if target.key != key {
            return Err(BlackboardError::KeyMismatch {
                superseded_key: target.key.clone(),
                new_key: key.to_string(),
            });
        }
        self.post_inner(author, key, value, provenance, Some(supersedes))
    }

    fn post_inner(
        &mut self,
        author: &str,
        key: &str,
        value: Value,
        provenance: Provenance,
        supersedes: Option<u64>,
    ) -> Result<u64, BlackboardError> {
        if key.is_empty() {
            return Err(BlackboardError::EmptyKey);
        }
        if provenance.source_id.is_empty() {
            return Err(BlackboardError::EmptySource);
        }
        let id = self.next_id;
        self.next_id += 1;
        self.entries.push(FactEntry {
            id,
            key: key.to_string(),
            value,
            author: author.to_string(),
            label: provenance.source_kind.label(),
            provenance,
            supersedes,
        });
        Ok(id)
    }

    /// A single entry by id, live or superseded: history is never erased.
    pub fn get(&self, id: u64) -> Option<&FactEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// True when the entry exists and no later entry supersedes it.
    /// Derived by scan, never stored: storing it would mean mutating an
    /// old entry at supersede time, which append-only forbids.
    pub fn is_live(&self, id: u64) -> bool {
        self.entries.iter().any(|e| e.id == id)
            && !self.entries.iter().any(|e| e.supersedes == Some(id))
    }

    /// The board's current word on `key`: the latest live entry, if any.
    /// Superseded entries are skipped, not deleted.
    pub fn current(&self, key: &str) -> Option<&FactEntry> {
        self.entries
            .iter()
            .rev()
            .find(|e| e.key == key && self.is_live(e.id))
    }

    /// Like [`Blackboard::current`], but only when the live fact is at
    /// least as trusted as `max_label` (rank order: User < Verified <
    /// Untrusted, so "at least as trusted" means rank <= `max_label`'s).
    /// An untrusted live fact is hidden from a reader that only accepts
    /// verified-or-better facts, rather than being silently trusted.
    pub fn current_trusted(&self, key: &str, max_label: TrustLabel) -> Option<&FactEntry> {
        self.current(key).filter(|e| e.label.rank() <= max_label.rank())
    }

    /// Every live fact on the board, in post order.
    pub fn all_current(&self) -> Vec<&FactEntry> {
        self.entries.iter().filter(|e| self.is_live(e.id)).collect()
    }

    /// Every live fact at least as trusted as `max_label`, in post order.
    pub fn all_current_trusted(&self, max_label: TrustLabel) -> Vec<&FactEntry> {
        self.all_current()
            .into_iter()
            .filter(|e| e.label.rank() <= max_label.rank())
            .collect()
    }

    /// Full history for a key in post order: live fact last, every
    /// superseded version before it, none erased.
    pub fn history(&self, key: &str) -> Vec<&FactEntry> {
        self.entries.iter().filter(|e| e.key == key).collect()
    }

    /// The supersede chain ending at `id`, origin first: walk
    /// `supersedes` links back to the original fact, then reverse.
    /// Returns `None` only if the chain is broken, which post-time
    /// validation makes impossible for entries this board created.
    pub fn lineage(&self, id: u64) -> Option<Vec<&FactEntry>> {
        let mut chain = Vec::new();
        let mut cur = self.get(id)?;
        loop {
            chain.push(cur);
            match cur.supersedes {
                Some(prev) => cur = self.get(prev)?,
                None => break,
            }
        }
        chain.reverse();
        Some(chain)
    }

    /// Number of facts ever posted, live plus superseded.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::SourceKind;

    fn prov(kind: SourceKind, source_id: &str) -> Provenance {
        Provenance::new(kind, source_id, 1_000, 0.9)
    }

    #[test]
    fn post_assigns_ids_and_current_returns_the_live_fact_with_derived_label() {
        let mut board = Blackboard::new();
        let id = board
            .post("agent-a", "supplier.payment_terms", Value::text("60 days"), prov(SourceKind::UserDirect, "user"))
            .unwrap();
        assert_eq!(id, 1);
        let cur = board.current("supplier.payment_terms").unwrap();
        assert_eq!(cur.value, Value::text("60 days"));
        assert_eq!(cur.author, "agent-a");
        assert_eq!(cur.label, TrustLabel::User, "user-direct source derives User label");
        assert_eq!(cur.supersedes, None);
        assert!(board.current("supplier.unknown").is_none());
    }

    #[test]
    fn supersede_keeps_history_intact_and_old_fact_stays_readable() {
        let mut board = Blackboard::new();
        let v1 = board
            .post("agent-a", "supplier.payment_terms", Value::text("60 days"), prov(SourceKind::WebFetch, "web"))
            .unwrap();
        let v2 = board
            .post_superseding(
                "agent-b",
                "supplier.payment_terms",
                Value::text("30 days"),
                prov(SourceKind::SystemSigned, "erp"),
                v1,
            )
            .unwrap();
        assert_ne!(v1, v2);
        // The old fact is superseded but NOT erased or rewritten.
        let old = board.get(v1).unwrap();
        assert_eq!(old.value, Value::text("60 days"));
        assert!(!board.is_live(v1));
        assert!(board.is_live(v2));
        // Current word is the new fact; history holds both, in post order.
        assert_eq!(board.current("supplier.payment_terms").unwrap().id, v2);
        let hist = board.history("supplier.payment_terms");
        assert_eq!(hist.iter().map(|e| e.id).collect::<Vec<_>>(), vec![v1, v2]);
        assert_eq!(board.len(), 2, "supersede appends, it never replaces in place");
    }

    #[test]
    fn lineage_walks_back_to_the_original_fact() {
        let mut board = Blackboard::new();
        let v1 = board
            .post("a", "k", Value::int(1), prov(SourceKind::UserDirect, "user"))
            .unwrap();
        let v2 = board
            .post_superseding("a", "k", Value::int(2), prov(SourceKind::UserDirect, "user"), v1)
            .unwrap();
        let v3 = board
            .post_superseding("a", "k", Value::int(3), prov(SourceKind::UserDirect, "user"), v2)
            .unwrap();
        let chain = board.lineage(v3).unwrap();
        assert_eq!(chain.iter().map(|e| e.id).collect::<Vec<_>>(), vec![v1, v2, v3]);
        assert_eq!(chain[0].value, Value::int(1), "origin first");
        // Lineage of the origin is just itself.
        assert_eq!(board.lineage(v1).unwrap().iter().map(|e| e.id).collect::<Vec<_>>(), vec![v1]);
        assert!(board.lineage(999).is_none());
    }

    #[test]
    fn supersede_rejects_unknown_already_superseded_and_key_mismatch_without_posting() {
        let mut board = Blackboard::new();
        let v1 = board
            .post("a", "k", Value::int(1), prov(SourceKind::UserDirect, "user"))
            .unwrap();
        let v2 = board
            .post_superseding("a", "k", Value::int(2), prov(SourceKind::UserDirect, "user"), v1)
            .unwrap();

        assert_eq!(
            board.post_superseding("a", "k", Value::int(9), prov(SourceKind::UserDirect, "user"), 999),
            Err(BlackboardError::UnknownSuperseded { id: 999 })
        );
        assert_eq!(
            board.post_superseding("a", "k", Value::int(9), prov(SourceKind::UserDirect, "user"), v1),
            Err(BlackboardError::AlreadySuperseded { id: v1 }),
            "superseding a superseded fact would fork the chain"
        );
        assert_eq!(
            board.post_superseding("a", "other", Value::int(9), prov(SourceKind::UserDirect, "user"), v2),
            Err(BlackboardError::KeyMismatch {
                superseded_key: "k".to_string(),
                new_key: "other".to_string()
            })
        );
        assert_eq!(board.len(), 2, "rejected supersedes post nothing");
        assert_eq!(board.current("k").unwrap().id, v2, "live fact unchanged by rejections");
    }

    #[test]
    fn empty_key_and_empty_source_are_rejected() {
        let mut board = Blackboard::new();
        assert_eq!(
            board.post("a", "", Value::int(1), prov(SourceKind::UserDirect, "user")),
            Err(BlackboardError::EmptyKey)
        );
        assert_eq!(
            board.post("a", "k", Value::int(1), prov(SourceKind::UserDirect, "")),
            Err(BlackboardError::EmptySource)
        );
        assert!(board.is_empty(), "no provenance, no persistence — even on the board");
    }

    #[test]
    fn trust_filter_hides_untrusted_live_facts_from_strict_readers() {
        let mut board = Blackboard::new();
        board
            .post("a", "rumor", Value::text("x"), prov(SourceKind::WebFetch, "web"))
            .unwrap();
        board
            .post("a", "confirmed", Value::text("y"), prov(SourceKind::SystemSigned, "erp"))
            .unwrap();
        // WebFetch derives Untrusted: invisible to a Verified-or-better reader.
        assert!(board.current_trusted("rumor", TrustLabel::Verified).is_none());
        assert!(board.current_trusted("rumor", TrustLabel::Untrusted).is_some());
        assert!(board.current_trusted("confirmed", TrustLabel::Verified).is_some());
        let trusted = board.all_current_trusted(TrustLabel::Verified);
        assert_eq!(trusted.iter().map(|e| e.key.as_str()).collect::<Vec<_>>(), vec!["confirmed"]);
        assert_eq!(board.all_current().len(), 2, "filtering hides, never deletes");
    }

    #[test]
    fn superseding_one_key_leaves_other_keys_live() {
        let mut board = Blackboard::new();
        let a1 = board
            .post("a", "a-key", Value::int(1), prov(SourceKind::UserDirect, "user"))
            .unwrap();
        let b1 = board
            .post("a", "b-key", Value::int(10), prov(SourceKind::UserDirect, "user"))
            .unwrap();
        board
            .post_superseding("a", "a-key", Value::int(2), prov(SourceKind::UserDirect, "user"), a1)
            .unwrap();
        assert_eq!(board.current("a-key").unwrap().value, Value::int(2));
        assert_eq!(board.current("b-key").unwrap().id, b1);
        assert!(board.is_live(b1));
    }
}
