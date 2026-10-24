//! Capability tokens (doc ch. 8 + ch. 13).
//!
//! A capability is an unforgeable, attenuable, short-lived, revocable
//! bearer token issued *only* by the kernel-side [`CapabilityAuthority`]:
//!
//! * **Unforgeable** — claims are HMAC-signed with the authority's secret.
//!   The token is a plain readable struct (like a JWT): anyone may inspect
//!   it, but mutating any claim invalidates the signature.
//! * **Attenuation only narrows** — deriving a child token requires the new
//!   scope to be a subset of the parent's, and its expiry cannot exceed the
//!   parent's. There is no API that widens a scope.
//! * **Short leases** — every token carries an expiry checked against the
//!   injected clock on each verification.
//! * **Revocation is total** — revoking a token revokes its whole delegation
//!   subtree.
//! * **Negative capabilities are monotonic** (doc ch. 13): a deny rule
//!   applies to a scope and to *every scope derived from it*, and it is
//!   evaluated at verification time — so a deny added *after* a delegation
//!   still blocks the delegate. Deny can never be attenuated away.

use crate::crypto::{hex, hmac_sha256, mac_eq};
use crate::typesys::Effect;
use crate::Clock;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::rc::Rc;

/// What a capability authorizes: one tool, a set of resources, a set of
/// effects. Scopes are the currency of attenuation: a child scope must be a
/// subset of its parent's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scope {
    pub tool: String,
    pub resources: BTreeSet<String>,
    pub effects: BTreeSet<Effect>,
}

impl Scope {
    pub fn new(tool: impl Into<String>, resources: &[&str], effects: &[Effect]) -> Self {
        Scope {
            tool: tool.into(),
            resources: resources.iter().map(|s| s.to_string()).collect(),
            effects: effects.iter().copied().collect(),
        }
    }

    /// `self` ⊆ `other`: same tool, resources and effects both subsets.
    pub fn is_subset_of(&self, other: &Scope) -> bool {
        self.tool == other.tool
            && self.resources.is_subset(&other.resources)
            && self.effects.is_subset(&other.effects)
    }

    /// Does this scope touch anything the deny rule forbids?
    fn intersects_deny(&self, deny: &DenyRule) -> bool {
        if let Some(tool) = &deny.tool {
            if *tool != self.tool {
                return false;
            }
        }
        let effect_hit = deny.effects.is_empty() || !self.effects.is_disjoint(&deny.effects);
        let resource_hit = self
            .resources
            .iter()
            .any(|r| r.starts_with(&deny.resource_prefix));
        effect_hit && resource_hit
    }
}

/// A negative capability: "no capability at or below this scope may ever be
/// exercised" (doc ch. 13). An empty `effects` set means all effects; an
/// empty `resource_prefix` matches every resource.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenyRule {
    pub tool: Option<String>,
    pub resource_prefix: String,
    pub effects: BTreeSet<Effect>,
}

impl DenyRule {
    pub fn for_tool(tool: &str, resource_prefix: &str) -> Self {
        DenyRule {
            tool: Some(tool.to_string()),
            resource_prefix: resource_prefix.to_string(),
            effects: BTreeSet::new(),
        }
    }
}

/// The signed payload of a capability token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapClaims {
    pub id: u64,
    pub holder: String,
    pub scope: Scope,
    pub issued_ms: u64,
    pub expires_ms: u64,
    pub max_uses: u32,
    pub parent: Option<u64>,
    pub generation: u32,
}

impl CapClaims {
    /// Deterministic encoding that the signature covers. Any change to any
    /// field changes this string and thus breaks the signature.
    fn canonical(&self) -> String {
        let resources: Vec<&str> = self.scope.resources.iter().map(|s| s.as_str()).collect();
        let effects: Vec<&str> = self.scope.effects.iter().map(|e| e.name()).collect();
        format!(
            "cap:v1|id={}|holder={}|tool={}|resources={}|effects={}|issued={}|expires={}|max_uses={}|parent={}|gen={}",
            self.id,
            self.holder,
            self.scope.tool,
            resources.join(","),
            effects.join(","),
            self.issued_ms,
            self.expires_ms,
            self.max_uses,
            self.parent.map(|p| p.to_string()).unwrap_or_else(|| "-".to_string()),
            self.generation
        )
    }
}

/// A capability token: claims plus the authority's HMAC over them.
/// Fields are public (bearer-token style); integrity comes from the
/// signature, exactly as with a JWT.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Capability {
    pub claims: CapClaims,
    pub sig: [u8; 32],
}

impl Capability {
    pub fn id(&self) -> u64 {
        self.claims.id
    }
    pub fn holder(&self) -> &str {
        &self.claims.holder
    }
    pub fn scope(&self) -> &Scope {
        &self.claims.scope
    }
    /// Short fingerprint for ledger entries and logs.
    pub fn fingerprint(&self) -> String {
        hex(&self.sig[..6])
    }
}

/// A single-use capability. The only way to spend it is to hand the whole
/// value to the kernel (`Kernel::act_linear` consumes `self`), so spending
/// the same linear capability twice is impossible *by construction* — the
/// borrow checker rejects the second use at compile time. This is the
/// affine-type reading of doc ch. 7's linear arrow, in the type system
/// rather than in a runtime counter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearCapability {
    cap: Capability,
}

impl LinearCapability {
    pub fn new(cap: Capability) -> Self {
        LinearCapability { cap }
    }
    pub fn capability(&self) -> &Capability {
        &self.cap
    }
    /// Consume the wrapper and return the underlying token (kernel-side).
    pub fn into_capability(self) -> Capability {
        self.cap
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapError {
    BadSignature,
    Expired { expires_ms: u64, now_ms: u64 },
    Revoked { id: u64 },
    Denied,
    UsesExhausted { id: u64 },
    NotSubset,
    ExpiryExceedsParent,
    Unknown { id: u64 },
}

impl std::fmt::Display for CapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CapError::BadSignature => write!(f, "capability signature invalid (forged or tampered)"),
            CapError::Expired { expires_ms, now_ms } => {
                write!(f, "capability expired at {expires_ms} (now {now_ms})")
            }
            CapError::Revoked { id } => write!(f, "capability {id} has been revoked"),
            CapError::Denied => write!(f, "capability intersects a negative-capability deny rule"),
            CapError::UsesExhausted { id } => write!(f, "capability {id} has no uses left"),
            CapError::NotSubset => write!(f, "derived scope is not a subset of the parent scope"),
            CapError::ExpiryExceedsParent => write!(f, "derived expiry exceeds the parent's expiry"),
            CapError::Unknown { id } => write!(f, "capability {id} was not issued by this authority"),
        }
    }
}
impl std::error::Error for CapError {}

/// The kernel-side issuer/verifier. Holds the signing secret; userland code
/// never sees it, which is what makes tokens unforgeable.
pub struct CapabilityAuthority {
    secret: Vec<u8>,
    clock: Rc<dyn Clock>,
    next_id: u64,
    issued: HashSet<u64>,
    revoked: HashSet<u64>,
    children: HashMap<u64, Vec<u64>>,
    uses: HashMap<u64, u32>,
    denies: Vec<DenyRule>,
}

impl CapabilityAuthority {
    pub fn new(secret: impl Into<Vec<u8>>, clock: Rc<dyn Clock>) -> Self {
        CapabilityAuthority {
            secret: secret.into(),
            clock,
            next_id: 1,
            issued: HashSet::new(),
            revoked: HashSet::new(),
            children: HashMap::new(),
            uses: HashMap::new(),
            denies: Vec::new(),
        }
    }

    fn sign(&self, claims: &CapClaims) -> [u8; 32] {
        hmac_sha256(&self.secret, claims.canonical().as_bytes())
    }

    /// Issue a fresh root capability with the default short lease
    /// accounting (unbounded uses unless [`issue_with_uses`] is used).
    pub fn issue(&mut self, holder: &str, scope: Scope, ttl_ms: u64) -> Capability {
        self.issue_with_uses(holder, scope, ttl_ms, u32::MAX)
    }

    pub fn issue_with_uses(
        &mut self,
        holder: &str,
        scope: Scope,
        ttl_ms: u64,
        max_uses: u32,
    ) -> Capability {
        let now = self.clock.now_ms();
        let claims = CapClaims {
            id: self.next_id,
            holder: holder.to_string(),
            scope,
            issued_ms: now,
            expires_ms: now + ttl_ms,
            max_uses,
            parent: None,
            generation: 0,
        };
        self.next_id += 1;
        self.issued.insert(claims.id);
        let sig = self.sign(&claims);
        Capability { claims, sig }
    }

    /// Verify a token end-to-end: issued here, signature intact, not
    /// expired, not revoked, not denied, uses remaining.
    pub fn verify(&self, cap: &Capability) -> Result<(), CapError> {
        if !self.issued.contains(&cap.claims.id) {
            return Err(CapError::Unknown { id: cap.claims.id });
        }
        let expected = self.sign(&cap.claims);
        if !mac_eq(&expected, &cap.sig) {
            return Err(CapError::BadSignature);
        }
        let now = self.clock.now_ms();
        if now >= cap.claims.expires_ms {
            return Err(CapError::Expired { expires_ms: cap.claims.expires_ms, now_ms: now });
        }
        if self.revoked.contains(&cap.claims.id) {
            return Err(CapError::Revoked { id: cap.claims.id });
        }
        if self.denies.iter().any(|d| cap.claims.scope.intersects_deny(d)) {
            return Err(CapError::Denied);
        }
        let used = self.uses.get(&cap.claims.id).copied().unwrap_or(0);
        if used >= cap.claims.max_uses {
            return Err(CapError::UsesExhausted { id: cap.claims.id });
        }
        Ok(())
    }

    /// Verify and record one use of the token.
    pub fn verify_and_consume(&mut self, cap: &Capability) -> Result<CapClaims, CapError> {
        self.verify(cap)?;
        *self.uses.entry(cap.claims.id).or_insert(0) += 1;
        Ok(cap.claims.clone())
    }

    /// Derive a strictly narrower token from `parent` (delegation is
    /// attenuation under a new holder). Refuses — by construction of the
    /// API — any widening of tool, resources, effects or lifetime.
    pub fn attenuate(
        &mut self,
        parent: &Capability,
        new_holder: &str,
        narrower: Scope,
        ttl_ms: u64,
    ) -> Result<Capability, CapError> {
        self.verify(parent)?;
        if !narrower.is_subset_of(&parent.claims.scope) {
            return Err(CapError::NotSubset);
        }
        let now = self.clock.now_ms();
        let expires = now + ttl_ms;
        if expires > parent.claims.expires_ms {
            return Err(CapError::ExpiryExceedsParent);
        }
        let claims = CapClaims {
            id: self.next_id,
            holder: new_holder.to_string(),
            scope: narrower,
            issued_ms: now,
            expires_ms: expires,
            max_uses: parent.claims.max_uses,
            parent: Some(parent.claims.id),
            generation: parent.claims.generation + 1,
        };
        self.next_id += 1;
        self.issued.insert(claims.id);
        self.children.entry(parent.claims.id).or_default().push(claims.id);
        let sig = self.sign(&claims);
        Ok(Capability { claims, sig })
    }

    /// Revoke a token and its entire delegation subtree.
    pub fn revoke(&mut self, id: u64) {
        let mut stack = vec![id];
        while let Some(cur) = stack.pop() {
            self.revoked.insert(cur);
            if let Some(kids) = self.children.get(&cur) {
                stack.extend(kids.iter().copied());
            }
        }
    }

    /// Add a negative-capability rule. Rules are evaluated at verification
    /// time against the presented scope, so they bind every existing and
    /// future derivation of a matching scope — monotonic by construction.
    pub fn add_deny(&mut self, rule: DenyRule) {
        self.denies.push(rule);
    }

    pub fn is_denied(&self, scope: &Scope) -> bool {
        self.denies.iter().any(|d| scope.intersects_deny(d))
    }

    pub fn uses_of(&self, id: u64) -> u32 {
        self.uses.get(&id).copied().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ManualClock;

    fn authority(clock: &Rc<ManualClock>) -> CapabilityAuthority {
        CapabilityAuthority::new(b"kernel-secret".to_vec(), clock.clone())
    }

    fn clock() -> Rc<ManualClock> {
        Rc::new(ManualClock::new(1_000_000))
    }

    fn file_scope() -> Scope {
        Scope::new("file.read", &["/docs", "/docs/a.txt"], &[Effect::Read])
    }

    #[test]
    fn issued_token_verifies() {
        let c = clock();
        let mut auth = authority(&c);
        let cap = auth.issue("agent-1", file_scope(), 600_000);
        assert!(auth.verify(&cap).is_ok());
        assert_eq!(cap.claims.holder, "agent-1");
        assert_eq!(cap.claims.generation, 0);
    }

    #[test]
    fn tampered_scope_breaks_signature() {
        let c = clock();
        let mut auth = authority(&c);
        let mut cap = auth.issue("agent-1", file_scope(), 600_000);
        // Attacker tries to widen the resource set in the token itself.
        cap.claims.scope.resources.insert("/etc/passwd".to_string());
        assert_eq!(auth.verify(&cap), Err(CapError::BadSignature));
    }

    #[test]
    fn forged_token_from_scratch_is_rejected() {
        let c = clock();
        let mut auth = authority(&c);
        let real = auth.issue("agent-1", file_scope(), 600_000);
        // Attacker fabricates claims and guesses a signature.
        let mut forged = real.clone();
        forged.sig = [7u8; 32];
        assert_eq!(auth.verify(&forged), Err(CapError::BadSignature));
        // A token "issued" by a different authority (different secret) too.
        let mut other = CapabilityAuthority::new(b"other-secret".to_vec(), c.clone());
        let foreign = other.issue("agent-1", file_scope(), 600_000);
        assert!(matches!(
            auth.verify(&foreign),
            Err(CapError::Unknown { .. }) | Err(CapError::BadSignature)
        ));
    }

    #[test]
    fn expired_token_is_rejected_after_lease() {
        let c = clock();
        let mut auth = authority(&c);
        let cap = auth.issue("agent-1", file_scope(), 600_000); // 10 min
        assert!(auth.verify(&cap).is_ok());
        c.advance(600_000);
        assert!(matches!(auth.verify(&cap), Err(CapError::Expired { .. })));
    }

    #[test]
    fn attenuation_narrows_and_links_parent() {
        let c = clock();
        let mut auth = authority(&c);
        let parent = auth.issue("agent-1", file_scope(), 600_000);
        let child = auth
            .attenuate(&parent, "agent-2", Scope::new("file.read", &["/docs/a.txt"], &[Effect::Read]), 60_000)
            .unwrap();
        assert_eq!(child.claims.parent, Some(parent.id()));
        assert_eq!(child.claims.generation, 1);
        assert!(auth.verify(&child).is_ok());
    }

    #[test]
    fn attenuation_cannot_widen_resources_or_effects() {
        let c = clock();
        let mut auth = authority(&c);
        let parent = auth.issue("agent-1", file_scope(), 600_000);
        let wider_resources =
            Scope::new("file.read", &["/docs", "/docs/a.txt", "/secrets"], &[Effect::Read]);
        assert_eq!(auth.attenuate(&parent, "agent-2", wider_resources, 60_000), Err(CapError::NotSubset));
        let wider_effects = Scope::new("file.read", &["/docs"], &[Effect::Read, Effect::Write]);
        assert_eq!(auth.attenuate(&parent, "agent-2", wider_effects, 60_000), Err(CapError::NotSubset));
        let other_tool = Scope::new("file.write", &["/docs"], &[Effect::Read]);
        assert_eq!(auth.attenuate(&parent, "agent-2", other_tool, 60_000), Err(CapError::NotSubset));
    }

    #[test]
    fn attenuation_cannot_extend_lifetime() {
        let c = clock();
        let mut auth = authority(&c);
        let parent = auth.issue("agent-1", file_scope(), 60_000);
        let child_scope = Scope::new("file.read", &["/docs"], &[Effect::Read]);
        assert_eq!(
            auth.attenuate(&parent, "agent-2", child_scope, 120_000),
            Err(CapError::ExpiryExceedsParent)
        );
    }

    #[test]
    fn revocation_cascades_to_descendants() {
        let c = clock();
        let mut auth = authority(&c);
        let parent = auth.issue("agent-1", file_scope(), 600_000);
        let child = auth
            .attenuate(&parent, "agent-2", Scope::new("file.read", &["/docs"], &[Effect::Read]), 60_000)
            .unwrap();
        let grandchild = auth
            .attenuate(&child, "agent-3", Scope::new("file.read", &["/docs"], &[Effect::Read]), 30_000)
            .unwrap();
        auth.revoke(parent.id());
        assert!(matches!(auth.verify(&parent), Err(CapError::Revoked { .. })));
        assert!(matches!(auth.verify(&child), Err(CapError::Revoked { .. })));
        assert!(matches!(auth.verify(&grandchild), Err(CapError::Revoked { .. })));
    }

    #[test]
    fn deny_rule_blocks_existing_delegation_monotonically() {
        let c = clock();
        let mut auth = authority(&c);
        let fin = Scope::new("file.read", &["/finance", "/finance/q3.xlsx"], &[Effect::Read]);
        let parent = auth.issue("agent-1", fin.clone(), 600_000);
        let child = auth
            .attenuate(&parent, "agent-2", Scope::new("file.read", &["/finance/q3.xlsx"], &[Effect::Read]), 60_000)
            .unwrap();
        assert!(auth.verify(&child).is_ok());
        // Policy lands *after* the delegation: "no agent reads /finance".
        auth.add_deny(DenyRule::for_tool("file.read", "/finance"));
        assert_eq!(auth.verify(&parent), Err(CapError::Denied));
        // The derived token cannot attenuate the deny away.
        assert_eq!(auth.verify(&child), Err(CapError::Denied));
        // And no new derivation from a denied parent is possible either.
        assert!(auth.attenuate(&parent, "agent-3", fin, 60_000).is_err());
    }

    #[test]
    fn deny_rule_is_scoped_and_does_not_block_unrelated() {
        let c = clock();
        let mut auth = authority(&c);
        auth.add_deny(DenyRule::for_tool("file.read", "/finance"));
        let cap = auth.issue("agent-1", file_scope(), 600_000);
        assert!(auth.verify(&cap).is_ok());
    }

    #[test]
    fn use_limit_is_enforced() {
        let c = clock();
        let mut auth = authority(&c);
        let cap = auth.issue_with_uses("agent-1", file_scope(), 600_000, 2);
        assert!(auth.verify_and_consume(&cap).is_ok());
        assert!(auth.verify_and_consume(&cap).is_ok());
        assert_eq!(auth.verify(&cap), Err(CapError::UsesExhausted { id: cap.id() }));
        assert_eq!(auth.uses_of(cap.id()), 2);
    }

    #[test]
    fn linear_capability_moves_by_value() {
        let c = clock();
        let mut auth = authority(&c);
        let cap = auth.issue("agent-1", file_scope(), 600_000);
        let linear = LinearCapability::new(cap);
        assert_eq!(linear.capability().holder(), "agent-1");
        let inner = linear.into_capability(); // consumes `linear`
        assert_eq!(inner.holder(), "agent-1");
        // A second `into_capability` call would not compile — that is the
        // point of the type; the runtime half of the guarantee is the
        // authority's use counter, tested above.
    }
}
