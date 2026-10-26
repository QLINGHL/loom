//! Cross-device capability delegation (doc ch. 18).
//!
//! One device's capability can be delegated to an agent on another device —
//! the doc's example is a phone camera delegated to glasses for 60 seconds,
//! read-only, single use. That is strictly more dangerous than local
//! delegation, so ch. 18 adds two gates on top of the attenuation rules in
//! [`crate::capabilities`] (a delegate may only ever narrow scope and
//! lifetime, never widen them):
//!
//! * **Attestation.** The receiving device must prove it runs an untampered
//!   system before any capability is handed over. [`DeviceAttestation`] is
//!   an HMAC-signed claim binding a device id to a system measurement, with
//!   its own expiry. Delegating with a forged, expired, wrong-device, or
//!   unexpected-measurement attestation is refused and creates no token.
//! * **Lease on connectivity.** A delegation is only usable while the two
//!   devices have been in contact recently. The registry tracks the last
//!   heartbeat per delegation; once silence exceeds the lease, the
//!   delegation is dead — [`DelegationRegistry::check`] refuses it even
//!   though the underlying token may not have expired yet, and
//!   [`DelegationRegistry::sweep_expired`] revokes the token outright so a
//!   stale copy cannot be used through the raw authority either. The local
//!   agent must stop instead of acting on a device it can no longer reach:
//!   "disconnected" is a status the caller can see, and "expired" is a
//!   hard refusal, never a silent pass.
//!
//! Two honest simplifications, in line with the rest of this prototype:
//! attestation here is an HMAC-signed *placeholder* measurement, not
//! hardware remote attestation (no TPM / secure enclave is involved — a
//! real deployment would verify a hardware-signed quote against a known
//! measurement allowlist, and the `expected_measurement` argument is where
//! that allowlist plugs in); and delegation is single-hop only, so a
//! delegated capability cannot itself be re-delegated to a third device
//! (that would need an attestation chain, which is out of scope).
//!
//! Time, as everywhere in this crate, comes from the injected [`Clock`]:
//! the lease counts silence since the last heartbeat (or delegation, if
//! none yet), so every disconnect scenario is deterministically testable.
//! The boundary is "exceeds": silence exactly equal to the lease is still
//! usable; one millisecond more is not.

use crate::capabilities::{CapError, Capability, CapabilityAuthority, Scope};
use crate::crypto::{hmac_sha256, mac_eq};
use crate::Clock;
use std::collections::HashMap;
use std::rc::Rc;

// ---------------------------------------------------------------------------
// Attestation
// ---------------------------------------------------------------------------

/// A device's proof that it runs an untampered system: its id bound to a
/// measurement of that system (e.g. a hash of firmware + OS image), signed
/// by the attestation issuer and carrying its own expiry. Anyone may inspect
/// it; mutating any claim breaks the signature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceAttestation {
    pub device_id: String,
    pub measurement: String,
    pub issued_ms: u64,
    pub expires_ms: u64,
    pub sig: [u8; 32],
}

impl DeviceAttestation {
    fn canonical(device_id: &str, measurement: &str, issued_ms: u64, expires_ms: u64) -> String {
        format!("attest:v1|device={device_id}|measurement={measurement}|issued={issued_ms}|expires={expires_ms}")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttestationError {
    BadSignature,
    DeviceMismatch { expected: String, got: String },
    MeasurementMismatch { expected: String, got: String },
    EmptyMeasurement,
    Expired { expires_ms: u64, now_ms: u64 },
    NotYetValid { issued_ms: u64, now_ms: u64 },
}

impl std::fmt::Display for AttestationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AttestationError::BadSignature => write!(f, "attestation signature invalid"),
            AttestationError::DeviceMismatch { expected, got } => {
                write!(f, "attestation is for device {expected}, not {got}")
            }
            AttestationError::MeasurementMismatch { expected, got } => {
                write!(f, "device measurement {got} does not match expected {expected}")
            }
            AttestationError::EmptyMeasurement => write!(f, "attestation carries no measurement"),
            AttestationError::Expired { expires_ms, now_ms } => {
                write!(f, "attestation expired at {expires_ms} (now {now_ms})")
            }
            AttestationError::NotYetValid { issued_ms, now_ms } => {
                write!(f, "attestation issued at {issued_ms} is not yet valid (now {now_ms})")
            }
        }
    }
}
impl std::error::Error for AttestationError {}

/// Issues attestations. Holds the signing secret; receivers never see it.
pub struct AttestationIssuer {
    secret: Vec<u8>,
    clock: Rc<dyn Clock>,
}

impl AttestationIssuer {
    pub fn new(secret: impl Into<Vec<u8>>, clock: Rc<dyn Clock>) -> Self {
        AttestationIssuer { secret: secret.into(), clock }
    }

    pub fn issue(&self, device_id: &str, measurement: &str, ttl_ms: u64) -> DeviceAttestation {
        let now = self.clock.now_ms();
        let sig = hmac_sha256(
            &self.secret,
            DeviceAttestation::canonical(device_id, measurement, now, now + ttl_ms).as_bytes(),
        );
        DeviceAttestation {
            device_id: device_id.to_string(),
            measurement: measurement.to_string(),
            issued_ms: now,
            expires_ms: now + ttl_ms,
            sig,
        }
    }
}

/// Verifies attestations against the issuer's secret and the caller's
/// expectations. `expected_measurement` is the known-good measurement
/// allowlist entry for this device; passing `None` accepts any non-empty
/// measurement (signature + freshness only) and is the weaker check.
pub struct AttestationVerifier {
    secret: Vec<u8>,
    clock: Rc<dyn Clock>,
}

impl AttestationVerifier {
    pub fn new(secret: impl Into<Vec<u8>>, clock: Rc<dyn Clock>) -> Self {
        AttestationVerifier { secret: secret.into(), clock }
    }

    pub fn verify(
        &self,
        att: &DeviceAttestation,
        expected_device: &str,
        expected_measurement: Option<&str>,
    ) -> Result<(), AttestationError> {
        if att.device_id != expected_device {
            return Err(AttestationError::DeviceMismatch {
                expected: expected_device.to_string(),
                got: att.device_id.clone(),
            });
        }
        if att.measurement.is_empty() {
            return Err(AttestationError::EmptyMeasurement);
        }
        if let Some(expected) = expected_measurement {
            if att.measurement != expected {
                return Err(AttestationError::MeasurementMismatch {
                    expected: expected.to_string(),
                    got: att.measurement.clone(),
                });
            }
        }
        let expected_sig = hmac_sha256(
            &self.secret,
            DeviceAttestation::canonical(&att.device_id, &att.measurement, att.issued_ms, att.expires_ms)
                .as_bytes(),
        );
        if !mac_eq(&expected_sig, &att.sig) {
            return Err(AttestationError::BadSignature);
        }
        let now = self.clock.now_ms();
        if att.issued_ms > now {
            return Err(AttestationError::NotYetValid { issued_ms: att.issued_ms, now_ms: now });
        }
        if now >= att.expires_ms {
            return Err(AttestationError::Expired { expires_ms: att.expires_ms, now_ms: now });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Delegation registry: leases over delegated capabilities
// ---------------------------------------------------------------------------

/// Lease status of one delegation at a moment in time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DelegationStatus {
    /// Connected (or never yet disconnected) and within the lease.
    Active,
    /// The peer reported a disconnect, but silence has not exceeded the
    /// lease yet: still usable, but the caller can see it should honestly
    /// downgrade (e.g. not start new long-running work on that device).
    DisconnectedWithinLease,
    /// Silence exceeded the lease: dead, whether or not the peer reported
    /// a disconnect. The local agent must stop.
    Expired,
    /// Explicitly revoked.
    Revoked,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DelegationError {
    Attestation(AttestationError),
    Capability(CapError),
    LeaseExpired { cap_id: u64, device_id: String, lease_ms: u64, silent_ms: u64 },
    UnknownDelegation { cap_id: u64 },
    Revoked { cap_id: u64 },
}

impl std::fmt::Display for DelegationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DelegationError::Attestation(e) => write!(f, "attestation rejected: {e}"),
            DelegationError::Capability(e) => write!(f, "capability rejected: {e}"),
            DelegationError::LeaseExpired { cap_id, device_id, lease_ms, silent_ms } => write!(
                f,
                "delegation {cap_id} to {device_id} expired: silent {silent_ms}ms, lease {lease_ms}ms"
            ),
            DelegationError::UnknownDelegation { cap_id } => {
                write!(f, "no delegation recorded for capability {cap_id}")
            }
            DelegationError::Revoked { cap_id } => {
                write!(f, "delegation {cap_id} has been revoked")
            }
        }
    }
}
impl std::error::Error for DelegationError {}

impl From<AttestationError> for DelegationError {
    fn from(e: AttestationError) -> Self {
        DelegationError::Attestation(e)
    }
}

impl From<CapError> for DelegationError {
    fn from(e: CapError) -> Self {
        DelegationError::Capability(e)
    }
}

/// One recorded delegation: the lease bookkeeping around a delegated token.
/// The token itself lives in the [`CapabilityAuthority`]; this record only
/// tracks which device holds it and how recently the devices were in
/// contact.
#[derive(Debug, Clone)]
struct DelegationRecord {
    device_id: String,
    lease_ms: u64,
    last_seen_ms: u64,
    disconnected_at_ms: Option<u64>,
    revoked: bool,
}

impl DelegationRecord {
    /// Silence since last contact (saturating: the clock never goes back
    /// in this prototype, but a defensive floor costs nothing).
    fn silent_ms(&self, now_ms: u64) -> u64 {
        now_ms.saturating_sub(self.last_seen_ms)
    }

    fn status(&self, now_ms: u64) -> DelegationStatus {
        if self.revoked {
            DelegationStatus::Revoked
        } else if self.silent_ms(now_ms) > self.lease_ms {
            DelegationStatus::Expired
        } else if self.disconnected_at_ms.is_some() {
            DelegationStatus::DisconnectedWithinLease
        } else {
            DelegationStatus::Active
        }
    }
}

/// A capability successfully delegated to another device. The holder is
/// the device id, so the token itself names its destination.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DelegatedCapability {
    pub capability: Capability,
    pub device_id: String,
    pub lease_ms: u64,
}

impl DelegatedCapability {
    pub fn cap_id(&self) -> u64 {
        self.capability.id()
    }
}

/// Parameters for one delegation, bundled so [`DelegationRegistry::delegate`]
/// keeps a short signature. The scope is moved in; the rest are borrowed.
pub struct DelegationRequest<'a> {
    pub device_id: &'a str,
    pub attestation: &'a DeviceAttestation,
    pub parent: &'a Capability,
    pub narrower: Scope,
    pub ttl_ms: u64,
    pub lease_ms: u64,
    pub expected_measurement: Option<&'a str>,
}

/// Tracks cross-device delegations and their connectivity leases. The
/// authority stays the sole issuer/verifier of tokens; this registry only
/// adds the ch. 18 gates (attestation at delegate time, lease at use time)
/// and is passed the authority on each call so no token state is duplicated.
pub struct DelegationRegistry {
    verifier: AttestationVerifier,
    clock: Rc<dyn Clock>,
    records: HashMap<u64, DelegationRecord>,
}

impl DelegationRegistry {
    pub fn new(verifier: AttestationVerifier, clock: Rc<dyn Clock>) -> Self {
        DelegationRegistry { verifier, clock, records: HashMap::new() }
    }

    /// Delegate `request.parent` to `request.device_id`, narrowed to
    /// `request.narrower` scope and `request.ttl_ms` lifetime, for a
    /// connectivity lease of `request.lease_ms`.
    /// Refuses before creating anything if the attestation fails or the
    /// scope/lifetime would widen the parent.
    pub fn delegate(
        &mut self,
        auth: &mut CapabilityAuthority,
        request: DelegationRequest<'_>,
    ) -> Result<DelegatedCapability, DelegationError> {
        self.verifier.verify(request.attestation, request.device_id, request.expected_measurement)?;
        let cap = auth.attenuate(request.parent, request.device_id, request.narrower, request.ttl_ms)?;
        let now = self.clock.now_ms();
        self.records.insert(
            cap.id(),
            DelegationRecord {
                device_id: request.device_id.to_string(),
                lease_ms: request.lease_ms,
                last_seen_ms: now,
                disconnected_at_ms: None,
                revoked: false,
            },
        );
        Ok(DelegatedCapability {
            capability: cap,
            device_id: request.device_id.to_string(),
            lease_ms: request.lease_ms,
        })
    }

    /// Record contact from the peer device: the lease clock restarts here.
    pub fn heartbeat(&mut self, cap_id: u64) -> Result<(), DelegationError> {
        let now = self.clock.now_ms();
        let rec = self
            .records
            .get_mut(&cap_id)
            .ok_or(DelegationError::UnknownDelegation { cap_id })?;
        if rec.revoked {
            return Err(DelegationError::Revoked { cap_id });
        }
        rec.last_seen_ms = now;
        rec.disconnected_at_ms = None;
        Ok(())
    }

    /// Record that the peer is gone. The delegation stays usable only
    /// until silence exceeds the lease; this flag lets callers see the
    /// disconnect early and downgrade honestly.
    pub fn disconnect(&mut self, cap_id: u64) -> Result<(), DelegationError> {
        let now = self.clock.now_ms();
        let rec = self
            .records
            .get_mut(&cap_id)
            .ok_or(DelegationError::UnknownDelegation { cap_id })?;
        if rec.revoked {
            return Err(DelegationError::Revoked { cap_id });
        }
        rec.disconnected_at_ms = Some(now);
        Ok(())
    }

    pub fn status(&self, cap_id: u64) -> Option<DelegationStatus> {
        self.records.get(&cap_id).map(|r| r.status(self.clock.now_ms()))
    }

    /// The use-time gate: the delegation must exist, not be revoked, still
    /// within its connectivity lease, and its token must still verify with
    /// the authority (signature, expiry, deny rules, uses). Any failure
    /// refuses the act — the local agent stops here.
    pub fn check(&self, auth: &CapabilityAuthority, cap: &Capability) -> Result<(), DelegationError> {
        let rec = self
            .records
            .get(&cap.id())
            .ok_or(DelegationError::UnknownDelegation { cap_id: cap.id() })?;
        if rec.revoked {
            return Err(DelegationError::Revoked { cap_id: cap.id() });
        }
        let now = self.clock.now_ms();
        if rec.silent_ms(now) > rec.lease_ms {
            return Err(DelegationError::LeaseExpired {
                cap_id: cap.id(),
                device_id: rec.device_id.clone(),
                lease_ms: rec.lease_ms,
                silent_ms: rec.silent_ms(now),
            });
        }
        auth.verify(cap).map_err(DelegationError::Capability)
    }

    /// `check` plus recording one use of the token.
    pub fn consume(
        &self,
        auth: &mut CapabilityAuthority,
        cap: &Capability,
    ) -> Result<crate::capabilities::CapClaims, DelegationError> {
        self.check(auth, cap)?;
        auth.verify_and_consume(cap).map_err(DelegationError::Capability)
    }

    /// Explicitly kill a delegation: the record is marked revoked and the
    /// token (and any children) is revoked in the authority.
    pub fn revoke(&mut self, auth: &mut CapabilityAuthority, cap_id: u64) -> Result<(), DelegationError> {
        let rec = self
            .records
            .get_mut(&cap_id)
            .ok_or(DelegationError::UnknownDelegation { cap_id })?;
        rec.revoked = true;
        auth.revoke(cap_id);
        Ok(())
    }

    /// Revoke every delegation whose silence has exceeded its lease and
    /// return their capability ids. After a sweep, even a raw
    /// [`CapabilityAuthority::verify`] of a stale delegated token fails.
    pub fn sweep_expired(&mut self, auth: &mut CapabilityAuthority) -> Vec<u64> {
        let now = self.clock.now_ms();
        let mut swept = Vec::new();
        for (cap_id, rec) in self.records.iter_mut() {
            if !rec.revoked && rec.silent_ms(now) > rec.lease_ms {
                rec.revoked = true;
                auth.revoke(*cap_id);
                swept.push(*cap_id);
            }
        }
        swept.sort_unstable();
        swept
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typesys::Effect;
    use crate::ManualClock;

    const MEASUREMENT: &str = "phone-fw-4.2/os-18.1";

    struct Fixture {
        clock: Rc<ManualClock>,
        auth: CapabilityAuthority,
        issuer: AttestationIssuer,
        registry: DelegationRegistry,
    }

    impl Fixture {
        fn new() -> Self {
            let clock = Rc::new(ManualClock::new(1_000_000));
            let auth = CapabilityAuthority::new(b"kernel-secret".to_vec(), clock.clone());
            let issuer = AttestationIssuer::new(b"attest-secret".to_vec(), clock.clone());
            let verifier = AttestationVerifier::new(b"attest-secret".to_vec(), clock.clone());
            let registry = DelegationRegistry::new(verifier, clock.clone());
            Fixture { clock, auth, issuer, registry }
        }

        fn camera_scope() -> Scope {
            Scope::new("camera.read", &["/camera", "/camera/front"], &[Effect::Read])
        }

        fn phone_cap(&mut self) -> Capability {
            self.auth.issue("phone-agent", Self::camera_scope(), 600_000)
        }

        fn glasses_attestation(&self) -> DeviceAttestation {
            self.issuer.issue("glasses-1", MEASUREMENT, 600_000)
        }

        /// The doc's example: phone camera to glasses, 60 s, read-only, one
        /// resource. Lease 5 s of silence for fast tests.
        fn delegate_example(&mut self) -> DelegatedCapability {
            let parent = self.phone_cap();
            let att = self.glasses_attestation();
            self.registry
                .delegate(
                    &mut self.auth,
                    DelegationRequest {
                        device_id: "glasses-1",
                        attestation: &att,
                        parent: &parent,
                        narrower: Scope::new("camera.read", &["/camera/front"], &[Effect::Read]),
                        ttl_ms: 60_000,
                        lease_ms: 5_000,
                        expected_measurement: Some(MEASUREMENT),
                    },
                )
                .unwrap()
        }
    }

    #[test]
    fn delegate_with_valid_attestation_narrows_and_names_device() {
        let mut fx = Fixture::new();
        let delegated = fx.delegate_example();
        assert_eq!(delegated.device_id, "glasses-1");
        assert_eq!(delegated.capability.holder(), "glasses-1");
        assert_eq!(delegated.capability.claims.scope.resources.len(), 1);
        assert!(fx.registry.check(&fx.auth, &delegated.capability).is_ok());
        assert_eq!(fx.registry.status(delegated.cap_id()), Some(DelegationStatus::Active));
    }

    #[test]
    fn attestation_for_wrong_device_is_refused_and_creates_nothing() {
        let mut fx = Fixture::new();
        let parent = fx.phone_cap();
        // Attestation belongs to glasses-1 but is presented for glasses-2.
        let att = fx.glasses_attestation();
        let err = fx
            .registry
            .delegate(
                    &mut fx.auth,
                    DelegationRequest {
                        device_id: "glasses-2",
                        attestation: &att,
                        parent: &parent,
                        narrower: Scope::new("camera.read", &["/camera/front"], &[Effect::Read]),
                        ttl_ms: 60_000,
                        lease_ms: 5_000,
                        expected_measurement: Some(MEASUREMENT),
                    },
                )
            .unwrap_err();
        assert!(matches!(err, DelegationError::Attestation(AttestationError::DeviceMismatch { .. })));
        assert_eq!(fx.registry.status(2), None, "refused delegation leaves no record");
    }

    #[test]
    fn tampered_attestation_signature_is_refused() {
        let mut fx = Fixture::new();
        let parent = fx.phone_cap();
        let mut att = fx.glasses_attestation();
        att.measurement = "tampered-os".to_string(); // claim changed, sig now stale
        let err = fx
            .registry
            .delegate(
                    &mut fx.auth,
                    DelegationRequest {
                        device_id: "glasses-1",
                        attestation: &att,
                        parent: &parent,
                        narrower: Scope::new("camera.read", &["/camera/front"], &[Effect::Read]),
                        ttl_ms: 60_000,
                        lease_ms: 5_000,
                        expected_measurement: None,
                    },
                )
            .unwrap_err();
        assert!(matches!(err, DelegationError::Attestation(AttestationError::BadSignature)));
    }

    #[test]
    fn unexpected_measurement_is_refused() {
        let mut fx = Fixture::new();
        let parent = fx.phone_cap();
        let att = fx.glasses_attestation();
        let err = fx
            .registry
            .delegate(
                    &mut fx.auth,
                    DelegationRequest {
                        device_id: "glasses-1",
                        attestation: &att,
                        parent: &parent,
                        narrower: Scope::new("camera.read", &["/camera/front"], &[Effect::Read]),
                        ttl_ms: 60_000,
                        lease_ms: 5_000,
                        expected_measurement: Some("phone-fw-9.9/os-99.0") // known-good allowlist says otherwise,
                    },
                )
            .unwrap_err();
        assert!(matches!(
            err,
            DelegationError::Attestation(AttestationError::MeasurementMismatch { .. })
        ));
    }

    #[test]
    fn expired_attestation_is_refused() {
        let mut fx = Fixture::new();
        let parent = fx.phone_cap();
        let att = fx.issuer.issue("glasses-1", MEASUREMENT, 1_000);
        fx.clock.advance(1_000);
        let err = fx
            .registry
            .delegate(
                    &mut fx.auth,
                    DelegationRequest {
                        device_id: "glasses-1",
                        attestation: &att,
                        parent: &parent,
                        narrower: Scope::new("camera.read", &["/camera/front"], &[Effect::Read]),
                        ttl_ms: 60_000,
                        lease_ms: 5_000,
                        expected_measurement: Some(MEASUREMENT),
                    },
                )
            .unwrap_err();
        assert!(matches!(err, DelegationError::Attestation(AttestationError::Expired { .. })));
    }

    #[test]
    fn delegation_cannot_widen_scope() {
        let mut fx = Fixture::new();
        let parent = fx.phone_cap();
        let att = fx.glasses_attestation();
        let wider = Scope::new(
            "camera.read",
            &["/camera", "/camera/front", "/mic"],
            &[Effect::Read],
        );
        let err = fx
            .registry
            .delegate(
                    &mut fx.auth,
                    DelegationRequest {
                        device_id: "glasses-1",
                        attestation: &att,
                        parent: &parent,
                        narrower: wider,
                        ttl_ms: 60_000,
                        lease_ms: 5_000,
                        expected_measurement: Some(MEASUREMENT),
                    },
                )
            .unwrap_err();
        assert!(matches!(err, DelegationError::Capability(CapError::NotSubset)));
    }

    #[test]
    fn heartbeat_keeps_lease_alive_past_original_window() {
        let mut fx = Fixture::new();
        let delegated = fx.delegate_example();
        for _ in 0..4 {
            fx.clock.advance(4_000); // inside the 5 s lease each time
            fx.registry.heartbeat(delegated.cap_id()).unwrap();
        }
        // 16 s after delegation, but never silent more than 4 s: still fine.
        assert!(fx.registry.check(&fx.auth, &delegated.capability).is_ok());
    }

    #[test]
    fn lease_boundary_exactly_at_lease_usable_one_ms_past_dead() {
        let mut fx = Fixture::new();
        let delegated = fx.delegate_example();
        fx.clock.advance(5_000);
        assert!(fx.registry.check(&fx.auth, &delegated.capability).is_ok());
        fx.clock.advance(1);
        assert!(matches!(
            fx.registry.check(&fx.auth, &delegated.capability),
            Err(DelegationError::LeaseExpired { silent_ms: 5001, .. })
        ));
    }

    #[test]
    fn disconnect_beyond_lease_stops_the_local_agent() {
        let mut fx = Fixture::new();
        let delegated = fx.delegate_example();
        fx.registry.disconnect(delegated.cap_id()).unwrap();
        // Inside the lease the disconnect is visible but not yet fatal.
        assert_eq!(
            fx.registry.status(delegated.cap_id()),
            Some(DelegationStatus::DisconnectedWithinLease)
        );
        assert!(fx.registry.check(&fx.auth, &delegated.capability).is_ok());
        // The agent acts only while check passes: simulate its loop.
        let mut acts = 0;
        fx.clock.advance(6_000); // silence now exceeds the 5 s lease
        while fx.registry.check(&fx.auth, &delegated.capability).is_ok() {
            acts += 1;
            if acts > 10 {
                break;
            }
        }
        assert_eq!(acts, 0, "agent must stop, not act on a device it lost");
        assert_eq!(fx.registry.status(delegated.cap_id()), Some(DelegationStatus::Expired));
    }

    #[test]
    fn sweep_revokes_stale_token_even_for_raw_verification() {
        let mut fx = Fixture::new();
        let delegated = fx.delegate_example();
        fx.clock.advance(6_000);
        let swept = fx.registry.sweep_expired(&mut fx.auth);
        assert_eq!(swept, vec![delegated.cap_id()]);
        // The stale copy is now dead even if presented straight to the
        // authority, bypassing the registry.
        assert!(matches!(fx.auth.verify(&delegated.capability), Err(CapError::Revoked { .. })));
        // A second sweep finds nothing left to do.
        assert!(fx.registry.sweep_expired(&mut fx.auth).is_empty());
    }

    #[test]
    fn doc_example_single_use_delegation() {
        let mut fx = Fixture::new();
        let parent = fx.auth.issue_with_uses("phone-agent", Fixture::camera_scope(), 600_000, 1);
        let att = fx.glasses_attestation();
        let delegated = fx
            .registry
            .delegate(
                    &mut fx.auth,
                    DelegationRequest {
                        device_id: "glasses-1",
                        attestation: &att,
                        parent: &parent,
                        narrower: Scope::new("camera.read", &["/camera/front"], &[Effect::Read]),
                        ttl_ms: 60_000,
                        lease_ms: 5_000,
                        expected_measurement: Some(MEASUREMENT),
                    },
                )
            .unwrap();
        assert!(fx.registry.consume(&mut fx.auth, &delegated.capability).is_ok());
        assert!(matches!(
            fx.registry.consume(&mut fx.auth, &delegated.capability),
            Err(DelegationError::Capability(CapError::UsesExhausted { .. }))
        ));
    }

    #[test]
    fn explicit_revoke_blocks_use_and_reports_status() {
        let mut fx = Fixture::new();
        let delegated = fx.delegate_example();
        fx.registry.revoke(&mut fx.auth, delegated.cap_id()).unwrap();
        assert_eq!(fx.registry.status(delegated.cap_id()), Some(DelegationStatus::Revoked));
        assert!(matches!(
            fx.registry.check(&fx.auth, &delegated.capability),
            Err(DelegationError::Revoked { .. })
        ));
    }

    #[test]
    fn unknown_capability_has_no_delegation() {
        let mut fx = Fixture::new();
        let stray = fx.phone_cap(); // issued locally, never delegated
        assert!(matches!(
            fx.registry.check(&fx.auth, &stray),
            Err(DelegationError::UnknownDelegation { .. })
        ));
    }
}
