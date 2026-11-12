//! Indirect prompt injection guard — plan-level semantic audit (doc ch. 21 Q1).
//!
//! Ch. 13 taint tracking plus ch. 7 effect types stop *direct* injection
//! ("read content → execute it") and type-illegal plans. They do not stop
//! the case ch. 21 Q1 names as the field's most important open problem:
//! untrusted content shapes the *plan*, and the plan itself looks
//! completely reasonable — "forward the invoice to finance". Type-checking
//! cannot catch it, because it is type-legal.
//!
//! This module audits a plan *before* execution, per step, on three
//! questions the type checker cannot answer:
//!
//! 1. **Justification provenance.** Is there any trusted evidence behind
//!    this consequential step, or does its entire rationale trace back to
//!    untrusted content? A step whose only justification is an email body,
//!    a web page, or a third-party message is escalated, however benign it
//!    sounds.
//! 2. **Beneficiary.** Does the step benefit a party the user never named?
//!    The classic attack benefits someone outside the user's intent.
//! 3. **Payload honesty.** Does the step's description actually disclose
//!    the consequential payload in its arguments, or is the payload smuggled
//!    under a benign description ("summarize the email" that also sends it
//!    to an outside address)? A mismatch is a block, not a confirmation:
//!    the planner is misrepresenting what the step does.
//!
//! Honest limitation (doc ch. 21 Q1): this is probabilistic detection plus
//! human confirmation, not a proof. A well-formed attack that names a
//! plausible beneficiary and is justified only by untrusted text can only
//! be escalated to [`Verdict::Confirm`], never proven malicious. The
//! guard's job is to make that escalation *automatic* and *auditable*, so
//! no consequential step ever executes on untrusted-only justification
//! without a human having seen the provenance. Read-only steps are never
//! blocked: they cannot change the world, and their taint is already
//! tracked by [`crate::taint`].

use crate::taint::TrustLabel;
use crate::typesys::Effect;
use crate::Value;
use std::collections::{BTreeMap, BTreeSet};

/// What a plan step is justified by: one piece of evidence with its
/// provenance label and its laundering depth.
#[derive(Debug, Clone, PartialEq)]
pub struct Evidence {
    /// Short human-readable summary of the evidence ("email body from
    /// vendor@example.com", "user's message").
    pub summary: String,
    /// Trust label of the evidence's ultimate source (see [`crate::taint`]:
    /// transformation never improves this label).
    pub label: TrustLabel,
    /// How many untrusted rephrasing hops the content passed through before
    /// reaching the planner (summarized, quoted, rephrased). `0` means the
    /// planner saw the source directly. Two or more hops on untrusted
    /// content is *laundering*: a clean-looking summary whose ultimate
    /// source is untrusted.
    pub hops: u32,
}

impl Evidence {
    pub fn new(summary: impl Into<String>, label: TrustLabel, hops: u32) -> Self {
        Evidence { summary: summary.into(), label, hops }
    }

    pub fn user(summary: impl Into<String>) -> Self {
        Evidence::new(summary, TrustLabel::User, 0)
    }

    pub fn verified(summary: impl Into<String>) -> Self {
        Evidence::new(summary, TrustLabel::Verified, 0)
    }

    pub fn untrusted(summary: impl Into<String>) -> Self {
        Evidence::new(summary, TrustLabel::Untrusted, 0)
    }

    /// True when untrusted content was rephrased at least twice before the
    /// planner saw it — the shape laundering takes.
    pub fn is_laundered(&self) -> bool {
        self.label.is_untrusted() && self.hops >= 2
    }
}

/// One step of a plan, as the planner presents it to the kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanStep {
    /// Stable id within the plan, used to point at the step in the audit.
    pub id: String,
    /// What the planner claims the step does. This is the text a human
    /// would read when confirming the step — so it must disclose the
    /// consequential payload (rule 3).
    pub description: String,
    /// Effects the step will have (from the tool manifest, ch. 7).
    pub effects: BTreeSet<Effect>,
    /// The actual tool arguments. The consequential payload lives here.
    pub payload: Value,
    /// Who benefits from the step's side effect, if any
    /// (e.g. `"finance"`, `"alice@example.com"`). `None` for read-only
    /// steps and for steps whose beneficiary the planner did not state.
    pub beneficiary: Option<String>,
    /// What the step's rationale rests on.
    pub evidence: Vec<Evidence>,
}

impl PlanStep {
    pub fn new(
        id: impl Into<String>,
        description: impl Into<String>,
        effects: &[Effect],
        payload: Value,
    ) -> Self {
        PlanStep {
            id: id.into(),
            description: description.into(),
            effects: effects.iter().copied().collect(),
            payload,
            beneficiary: None,
            evidence: Vec::new(),
        }
    }

    pub fn with_beneficiary(mut self, beneficiary: impl Into<String>) -> Self {
        self.beneficiary = Some(beneficiary.into());
        self
    }

    pub fn with_evidence(mut self, evidence: Vec<Evidence>) -> Self {
        self.evidence = evidence;
        self
    }

    /// True when the step changes the world (anything but a pure read).
    pub fn is_consequential(&self) -> bool {
        self.effects.iter().any(|e| e.is_side_effect())
    }
}

/// The user's stated intent, reduced to the one fact this audit needs:
/// which parties the user named. `"self"` is always implicitly included.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct UserIntent {
    /// Lower-cased party names the user mentioned (people, teams, addresses).
    pub beneficiaries: BTreeSet<String>,
}

impl UserIntent {
    pub fn new(beneficiaries: &[&str]) -> Self {
        UserIntent {
            beneficiaries: beneficiaries.iter().map(|b| b.to_lowercase()).collect(),
        }
    }

    /// True when `party` is the user themself or a party the user named.
    /// Matching is case-insensitive, and an address matches when its local
    /// part (before `@`) or its full text was named.
    pub fn names(&self, party: &str) -> bool {
        let lower = party.to_lowercase();
        if lower == "self" || lower == "me" || lower == "user" {
            return true;
        }
        if self.beneficiaries.contains(&lower) {
            return true;
        }
        if let Some((local, _)) = lower.split_once('@') {
            if self.beneficiaries.contains(local) {
                return true;
            }
        }
        // A named party matches an address that mentions it ("finance" was
        // named, the step sends to "finance@example.com").
        self.beneficiaries.iter().any(|b| lower.contains(b.as_str()))
    }
}

/// Audit outcome for one step (and, by worst-step, for a whole plan).
/// Ordered so the worst verdict wins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Verdict {
    /// Execute as planned.
    Allow,
    /// Execute only after a human has seen the reasons and confirmed.
    Confirm,
    /// Do not execute. The step misrepresents what it does.
    Block,
}

impl Verdict {
    pub fn name(self) -> &'static str {
        match self {
            Verdict::Allow => "allow",
            Verdict::Confirm => "confirm",
            Verdict::Block => "block",
        }
    }
}

/// The audit result for a single plan step.
#[derive(Debug, Clone, PartialEq)]
pub struct StepAudit {
    pub step_id: String,
    pub verdict: Verdict,
    /// Human-readable reasons, in rule order. Empty for `Allow`.
    pub reasons: Vec<String>,
    /// Justification strength: 2 = backed by user evidence, 1 = backed by
    /// verified evidence, 0 = untrusted-only or no evidence. Laundered
    /// evidence never raises the score.
    pub justification_score: u8,
}

/// The audit result for a whole plan.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanAudit {
    pub steps: Vec<StepAudit>,
    /// The worst step verdict. A plan is only as safe as its worst step.
    pub verdict: Verdict,
}

impl PlanAudit {
    /// True when any step needs a human decision before execution.
    pub fn needs_confirmation(&self) -> bool {
        self.verdict != Verdict::Allow
    }

    /// Ids of the steps that are not clean, in plan order.
    pub fn flagged_steps(&self) -> Vec<&str> {
        self.steps
            .iter()
            .filter(|s| s.verdict != Verdict::Allow)
            .map(|s| s.step_id.as_str())
            .collect()
    }
}

/// Payload keys whose values are consequential: if the step carries one of
/// these, its description must disclose the value (rule 3).
const CONSEQUENTIAL_KEYS: [&str; 5] = ["to", "recipient", "destination", "amount", "account"];

/// Audit a plan against the user's stated intent.
///
/// Read-only steps are always allowed (they cannot change the world).
/// Consequential steps are checked against the three rules in the module
/// docs; the step verdict is the worst applicable rule outcome.
pub fn audit_plan(intent: &UserIntent, steps: &[PlanStep]) -> PlanAudit {
    let mut audits = Vec::with_capacity(steps.len());
    let mut worst = Verdict::Allow;
    for step in steps {
        let audit = audit_step(intent, step);
        if audit.verdict > worst {
            worst = audit.verdict;
        }
        audits.push(audit);
    }
    PlanAudit { steps: audits, verdict: worst }
}

fn audit_step(intent: &UserIntent, step: &PlanStep) -> StepAudit {
    let mut reasons = Vec::new();
    let mut verdict = Verdict::Allow;
    let score = justification_score(&step.evidence);

    if !step.is_consequential() {
        return StepAudit {
            step_id: step.id.clone(),
            verdict: Verdict::Allow,
            reasons,
            justification_score: score,
        };
    }

    // Rule 1 — justification provenance: a consequential step needs at
    // least one trusted (User / Verified) piece of evidence.
    if score == 0 {
        verdict = Verdict::Confirm;
        if step.evidence.is_empty() {
            reasons.push(
                "no evidence at all justifies this consequential step".to_string(),
            );
        } else {
            reasons.push(
                "justification rests entirely on untrusted content; no trusted \
                 evidence supports this consequential step"
                    .to_string(),
            );
        }
    }

    // Laundering: untrusted content rephrased twice or more before the
    // planner saw it. It does not change the score (already 0), but it is
    // called out explicitly: the planner was shown a clean-looking summary
    // whose ultimate source is untrusted.
    if step.evidence.iter().any(|e| e.is_laundered()) {
        if verdict < Verdict::Confirm {
            verdict = Verdict::Confirm;
        }
        reasons.push(
            "evidence arrived via two or more untrusted rephrasing hops \
             (laundered provenance)"
                .to_string(),
        );
    }

    // Rule 2 — beneficiary: the step must benefit the user or a party the
    // user named, or state no beneficiary for a non-addressed effect.
    let addressed = step.effects.contains(&Effect::Send)
        || step.effects.contains(&Effect::Transfer)
        || step.effects.contains(&Effect::Publish);
    match (&step.beneficiary, addressed) {
        (Some(b), _) if !intent.names(b) => {
            if verdict < Verdict::Confirm {
                verdict = Verdict::Confirm;
            }
            reasons.push(format!(
                "beneficiary '{b}' was never named in the user's intent"
            ));
        }
        (None, true) => {
            if verdict < Verdict::Confirm {
                verdict = Verdict::Confirm;
            }
            reasons.push(
                "consequential step names no beneficiary; the recipient is \
                 unstated"
                    .to_string(),
            );
        }
        _ => {}
    }

    // Rule 3 — payload honesty: the description must disclose every
    // consequential payload field. A step that hides its payload behind a
    // benign description is blocked outright.
    for omitted in undisclosed_payload_fields(step) {
        verdict = Verdict::Block;
        reasons.push(format!(
            "description omits consequential payload field '{omitted}'; \
             the step does more than it says"
        ));
    }

    StepAudit { step_id: step.id.clone(), verdict, reasons, justification_score: score }
}

/// Best trusted evidence: User = 2, Verified = 1, Untrusted / none = 0.
/// Laundered evidence counts as untrusted (it already is, by label).
fn justification_score(evidence: &[Evidence]) -> u8 {
    evidence
        .iter()
        .map(|e| match e.label {
            TrustLabel::User => 2,
            TrustLabel::Verified => 1,
            TrustLabel::Untrusted => 0,
        })
        .max()
        .unwrap_or(0)
}

/// Consequential payload keys present in the step's arguments whose values
/// the step's description does not mention.
fn undisclosed_payload_fields(step: &PlanStep) -> Vec<String> {
    let mut omitted = Vec::new();
    if !step.is_consequential() {
        return omitted;
    }
    let Value::Map(args) = &step.payload else {
        return omitted;
    };
    let desc = step.description.to_lowercase();
    for key in CONSEQUENTIAL_KEYS {
        if let Some(value) = args.get(key) {
            if !description_mentions(&desc, value) {
                omitted.push(key.to_string());
            }
        }
    }
    omitted
}

/// True when the description mentions a payload value: the literal value,
/// the local part of an address, or the digits of a number.
fn description_mentions(desc_lower: &str, value: &Value) -> bool {
    match value {
        Value::Text(s) => {
            let lower = s.to_lowercase();
            if desc_lower.contains(&lower) {
                return true;
            }
            if let Some((local, _)) = lower.split_once('@') {
                if !local.is_empty() && desc_lower.contains(local) {
                    return true;
                }
            }
            false
        }
        Value::Int(n) => desc_lower.contains(&n.to_string()),
        _ => true, // Non-scalar payloads have no single value to disclose.
    }
}

/// Convenience: the consequential payload of a step as a flat map, for
/// logging the audit trail (ch. 22 ledger).
pub fn payload_summary(step: &PlanStep) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    if let Value::Map(args) = &step.payload {
        for key in CONSEQUENTIAL_KEYS {
            if let Some(v) = args.get(key) {
                let s = match v {
                    Value::Text(t) => t.clone(),
                    Value::Int(n) => n.to_string(),
                    Value::Bool(b) => b.to_string(),
                    _ => v.canonical(),
                };
                out.insert(key.to_string(), s);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn intent_with(names: &[&str]) -> UserIntent {
        UserIntent::new(names)
    }

    fn read_step(id: &str, desc: &str) -> PlanStep {
        PlanStep::new(id, desc, &[Effect::Read], Value::Null)
            .with_evidence(vec![Evidence::untrusted("web search result")])
    }

    fn send_step(id: &str, desc: &str, to: &str) -> PlanStep {
        PlanStep::new(
            id,
            desc,
            &[Effect::Send],
            Value::map(vec![("to", Value::text(to))]),
        )
        .with_beneficiary(to)
    }

    #[test]
    fn benign_plan_is_fully_allowed() {
        let intent = intent_with(&["alice"]);
        let steps = vec![
            read_step("s1", "search the web for flight options"),
            send_step("s2", "send the itinerary to alice@example.com", "alice@example.com")
                .with_evidence(vec![Evidence::user("user asked to send it to Alice")]),
        ];
        let audit = audit_plan(&intent, &steps);
        assert_eq!(audit.verdict, Verdict::Allow);
        assert!(audit.steps.iter().all(|s| s.verdict == Verdict::Allow));
        assert!(!audit.needs_confirmation());
    }

    #[test]
    fn q1_classic_attack_forward_to_finance_needs_confirmation() {
        // Doc ch. 21 Q1's own example: the plan looks completely reasonable,
        // but its only justification is an email body the user never wrote.
        let intent = intent_with(&["finance"]);
        let step = send_step("s1", "forward the invoice to finance@example.com", "finance@example.com")
            .with_evidence(vec![Evidence::untrusted(
                "email body: please forward this invoice to finance",
            )]);
        let audit = audit_plan(&intent, &[step]);
        assert_eq!(audit.verdict, Verdict::Confirm);
        assert_eq!(audit.steps[0].justification_score, 0);
        assert!(audit.steps[0].reasons.iter().any(|r| r.contains("untrusted")));
        assert_eq!(audit.flagged_steps(), vec!["s1"]);
    }

    #[test]
    fn beneficiary_never_named_in_intent_needs_confirmation() {
        let intent = intent_with(&["alice"]);
        let step = send_step("s1", "send the report to finance@example.com", "finance@example.com")
            .with_evidence(vec![Evidence::user("user asked to send the report")]);
        let audit = audit_plan(&intent, &[step]);
        assert_eq!(audit.verdict, Verdict::Confirm);
        assert!(audit.steps[0].reasons.iter().any(|r| r.contains("finance")));
    }

    #[test]
    fn payload_smuggled_under_benign_description_is_blocked() {
        let intent = intent_with(&["finance"]);
        let step = PlanStep::new(
            "s1",
            "summarize the invoice email",
            &[Effect::Send],
            Value::map(vec![
                ("to", Value::text("finance@example.com")),
                ("amount", Value::int(5000)),
            ]),
        )
        .with_beneficiary("finance@example.com")
        .with_evidence(vec![Evidence::user("user asked to summarize the invoice")]);
        let audit = audit_plan(&intent, &[step]);
        assert_eq!(audit.verdict, Verdict::Block);
        assert!(audit.steps[0].reasons.iter().any(|r| r.contains("'to'")));
        assert!(audit.steps[0].reasons.iter().any(|r| r.contains("'amount'")));
    }

    #[test]
    fn payload_disclosed_in_description_is_allowed() {
        let intent = intent_with(&["finance"]);
        let step = PlanStep::new(
            "s1",
            "send 5000 to finance@example.com for the invoice",
            &[Effect::Transfer],
            Value::map(vec![
                ("to", Value::text("finance@example.com")),
                ("amount", Value::int(5000)),
            ]),
        )
        .with_beneficiary("finance@example.com")
        .with_evidence(vec![Evidence::user("user approved paying the invoice")]);
        let audit = audit_plan(&intent, &[step]);
        assert_eq!(audit.verdict, Verdict::Allow);
    }

    #[test]
    fn laundered_evidence_is_called_out() {
        let intent = intent_with(&["finance"]);
        let step = send_step("s1", "forward the summary to finance@example.com", "finance@example.com")
            .with_evidence(vec![Evidence::new(
                "meeting notes summarizing the vendor email",
                TrustLabel::Untrusted,
                3,
            )]);
        let audit = audit_plan(&intent, &[step]);
        assert_eq!(audit.verdict, Verdict::Confirm);
        assert!(audit.steps[0].reasons.iter().any(|r| r.contains("laundered")));
    }

    #[test]
    fn mixed_evidence_with_one_trusted_source_is_allowed() {
        let intent = intent_with(&["alice"]);
        let step = send_step("s1", "send the draft to alice@example.com", "alice@example.com")
            .with_evidence(vec![
                Evidence::untrusted("web page suggesting Alice as contact"),
                Evidence::verified("company directory confirms alice@example.com"),
            ]);
        let audit = audit_plan(&intent, &[step]);
        assert_eq!(audit.verdict, Verdict::Allow);
        assert_eq!(audit.steps[0].justification_score, 1);
    }

    #[test]
    fn read_only_steps_are_never_blocked_by_untrusted_evidence() {
        let intent = intent_with(&[]);
        let audit = audit_plan(&intent, &[read_step("s1", "read the vendor page")]);
        assert_eq!(audit.verdict, Verdict::Allow);
    }

    #[test]
    fn addressed_step_without_beneficiary_needs_confirmation() {
        let intent = intent_with(&["alice"]);
        let step = PlanStep::new(
            "s1",
            "send the file to someone@example.com",
            &[Effect::Send],
            Value::map(vec![("to", Value::text("someone@example.com"))]),
        )
        .with_evidence(vec![Evidence::user("user said send it")]);
        let audit = audit_plan(&intent, &[step]);
        assert_eq!(audit.verdict, Verdict::Confirm);
        assert!(audit.steps[0].reasons.iter().any(|r| r.contains("no beneficiary")));
    }

    #[test]
    fn worst_step_verdict_wins_for_the_plan() {
        let intent = intent_with(&["alice"]);
        let ok = send_step("ok", "send the draft to alice@example.com", "alice@example.com")
            .with_evidence(vec![Evidence::user("user asked for this")]);
        let bad = PlanStep::new(
            "bad",
            "summarize the email",
            &[Effect::Send],
            Value::map(vec![("to", Value::text("eve@example.com"))]),
        )
        .with_beneficiary("eve@example.com")
        .with_evidence(vec![Evidence::user("user asked to summarize")]);
        let audit = audit_plan(&intent, &[ok, bad]);
        assert_eq!(audit.verdict, Verdict::Block);
        assert_eq!(audit.flagged_steps(), vec!["bad"]);
    }

    #[test]
    fn empty_plan_is_allowed() {
        let audit = audit_plan(&intent_with(&[]), &[]);
        assert_eq!(audit.verdict, Verdict::Allow);
        assert!(!audit.needs_confirmation());
    }

    #[test]
    fn user_attestation_upgrades_untrusted_evidence() {
        // The taint module's sanctioned upgrade path feeds the audit: once
        // the user vouches for the content, the step is justified.
        let intent = intent_with(&["finance"]);
        let attested =
            crate::taint::Tainted::untrusted("vendor email".to_string()).user_attest();
        let step = send_step("s1", "forward the invoice to finance@example.com", "finance@example.com")
            .with_evidence(vec![Evidence::new("vendor email (user-attested)", attested.label, 0)]);
        let audit = audit_plan(&intent, &[step]);
        assert_eq!(audit.verdict, Verdict::Allow);
        assert_eq!(audit.steps[0].justification_score, 2);
    }

    #[test]
    fn payload_summary_lists_consequential_fields() {
        let step = PlanStep::new(
            "s1",
            "pay the invoice",
            &[Effect::Transfer],
            Value::map(vec![
                ("to", Value::text("finance@example.com")),
                ("amount", Value::int(5000)),
                ("note", Value::text("invoice 42")),
            ]),
        );
        let summary = payload_summary(&step);
        assert_eq!(summary.get("to").map(String::as_str), Some("finance@example.com"));
        assert_eq!(summary.get("amount").map(String::as_str), Some("5000"));
        assert!(!summary.contains_key("note"));
    }
}
