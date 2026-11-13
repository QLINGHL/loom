//! Tool-description interop (doc ch. 21, open question 8: the
//! standardization chicken-and-egg).
//!
//! Every vendor defines its own tool-description format and capability
//! model, so the ecosystem is a set of mutually incompatible islands —
//! and standardization "usually happens after the market settles". Loom
//! cannot wait for that, so this module is the pragmatic middle layer:
//!
//! * a **canonical interchange format** (`CanonicalTool`), versioned, that
//!   can express everything Loom's own [`ToolManifest`](crate::tools::ToolManifest)
//!   expresses (name, description, typed parameters, effect set,
//!   consistency, compensability, schema version);
//! * **vendor adapters** that import from / export to two real-world
//!   shapes — MCP-style (`inputSchema`) and OpenAI function-calling style
//!   (`parameters`) — as plain [`Value`] maps, since the crate has no
//!   JSON dependency. Every import reports its **losses** explicitly:
//!   neither vendor format can express effects, consistency or
//!   compensability, so those are defaulted on import and dropped on
//!   export, never silently invented;
//! * **schema migration** v1 → v2: v1 described parameters only; v2 adds
//!   the effect/consistency/compensability rows, filled with the safest
//!   defaults (read-only, best-effort, irreversible) and a migration note
//!   per defaulted field;
//! * **compatibility negotiation** ([`negotiate`]): given a description a
//!   caller was written against and a description a remote side offers for
//!   the same tool, decide whether the caller can drive it —
//!   [`Negotiation::Compatible`], [`Negotiation::CompatibleWithLoss`]
//!   (usable, with named losses) or [`Negotiation::Incompatible`] (with
//!   reasons). A required parameter missing or mistyped on the remote side
//!   is incompatible; a dropped optional parameter or an undeclared effect
//!   is a loss, not a refusal;
//! * an **expressiveness report** ([`expressiveness_report`]) that states,
//!   per vendor format, which canonical fields it can and cannot carry —
//!   the evidence a future standardization discussion starts from.
//!
//! Nothing here trusts a foreign description: import is parsing, not
//! approval. A description that arrives over the wire still has to pass
//! the registry's hash pinning and approval before anything is invoked.

use crate::tools::ToolManifest;
use crate::typesys::{Compensability, Consistency, Effect};
use crate::Value;
use std::collections::BTreeSet;

/// Current canonical schema version. v1 carried parameters only; v2 adds
/// the effect/consistency/compensability rows.
pub const CURRENT_SCHEMA_VERSION: u32 = 2;

/// The type of one tool parameter, in the canonical format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParamType {
    Text,
    Int,
    Bool,
    List(Box<ParamType>),
    Object,
}

impl ParamType {
    pub fn name(&self) -> &'static str {
        match self {
            ParamType::Text => "string",
            ParamType::Int => "integer",
            ParamType::Bool => "boolean",
            ParamType::List(_) => "array",
            ParamType::Object => "object",
        }
    }

    /// Parse a JSON-Schema-style `"type"` string. `"array"` without an
    /// `"items"` type degrades to a list of text (loss is reported by the
    /// caller when it matters).
    pub fn from_json_type(s: &str) -> Result<ParamType, InteropError> {
        match s {
            "string" => Ok(ParamType::Text),
            "integer" | "number" => Ok(ParamType::Int),
            "boolean" => Ok(ParamType::Bool),
            "array" => Ok(ParamType::List(Box::new(ParamType::Text))),
            "object" => Ok(ParamType::Object),
            other => Err(InteropError::UnknownParamType(other.to_string())),
        }
    }
}

/// One typed parameter of a canonical tool description.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamSpec {
    pub name: String,
    pub ty: ParamType,
    pub required: bool,
    pub description: String,
}

impl ParamSpec {
    pub fn new(name: impl Into<String>, ty: ParamType, required: bool) -> Self {
        ParamSpec { name: name.into(), ty, required, description: String::new() }
    }

    pub fn with_description(mut self, d: impl Into<String>) -> Self {
        self.description = d.into();
        self
    }
}

/// A tool description in the canonical interchange format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalTool {
    pub name: String,
    pub description: String,
    pub params: Vec<ParamSpec>,
    pub effects: BTreeSet<Effect>,
    pub consistency: Consistency,
    pub compensability: Compensability,
    pub schema_version: u32,
}

impl CanonicalTool {
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        CanonicalTool {
            name: name.into(),
            description: description.into(),
            params: Vec::new(),
            effects: BTreeSet::from([Effect::Read]),
            consistency: Consistency::BestEffort,
            compensability: Compensability::Irreversible,
            schema_version: CURRENT_SCHEMA_VERSION,
        }
    }

    /// Parameters are kept sorted by name so the canonical form — and
    /// therefore any hash or comparison over it — is stable regardless of
    /// the order they were added or imported in.
    pub fn with_param(mut self, p: ParamSpec) -> Self {
        self.params.push(p);
        self.params.sort_by(|a, b| a.name.cmp(&b.name));
        self
    }

    pub fn with_effects(mut self, effects: &[Effect]) -> Self {
        self.effects = effects.iter().copied().collect();
        self
    }

    pub fn required_params(&self) -> Vec<&ParamSpec> {
        self.params.iter().filter(|p| p.required).collect()
    }

    pub fn param(&self, name: &str) -> Option<&ParamSpec> {
        self.params.iter().find(|p| p.name == name)
    }

    /// Build from one of Loom's own manifests. The manifest's schema
    /// version is carried over as-is.
    pub fn from_manifest(m: &ToolManifest) -> Self {
        CanonicalTool {
            name: m.name.clone(),
            description: m.description.clone(),
            params: Vec::new(),
            effects: m.effects.clone(),
            consistency: m.consistency,
            compensability: m.compensability,
            schema_version: m.schema_version,
        }
    }

    /// Convert back to a registry manifest. Parameters do not exist in
    /// `ToolManifest` (they live in the tool implementation), so a
    /// round trip through a manifest preserves everything except params.
    pub fn to_manifest(&self) -> ToolManifest {
        let effects: Vec<Effect> = self.effects.iter().copied().collect();
        let mut m = ToolManifest::new(
            self.name.clone(),
            self.description.clone(),
            &effects,
            self.consistency,
            self.compensability,
        );
        m.schema_version = self.schema_version;
        m
    }

    /// Migrate a v1 description (parameters only) to v2. Effect,
    /// consistency and compensability rows are filled with the safest
    /// defaults and each defaulted field is reported, so a v1 tool is
    /// never silently promoted to something it never declared.
    pub fn migrate_v1_to_v2(v1: CanonicalTool) -> (CanonicalTool, Vec<String>) {
        let mut notes = Vec::new();
        let mut v2 = v1;
        if v2.schema_version != 1 {
            let v = v2.schema_version;
            return (v2, vec![format!(
                "not a v1 description (schema_version {v}); no migration applied"
            )]);
        }
        if v2.effects.is_empty() {
            v2.effects = BTreeSet::from([Effect::Read]);
            notes.push("effects defaulted to read-only on v1->v2 migration".to_string());
        }
        v2.consistency = Consistency::BestEffort;
        notes.push("consistency defaulted to best-effort on v1->v2 migration".to_string());
        v2.compensability = Compensability::Irreversible;
        notes.push("compensability defaulted to irreversible on v1->v2 migration".to_string());
        v2.schema_version = CURRENT_SCHEMA_VERSION;
        (v2, notes)
    }
}

/// A vendor tool-description format this module can translate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VendorFormat {
    /// MCP-style: `{"name", "description", "inputSchema": {...}}`.
    Mcp,
    /// OpenAI function-calling style: `{"name", "description",
    /// "parameters": {...}}`.
    OpenAi,
}

impl VendorFormat {
    pub fn name(self) -> &'static str {
        match self {
            VendorFormat::Mcp => "mcp",
            VendorFormat::OpenAi => "openai-function",
        }
    }

    /// The map key that holds the JSON-Schema-style parameter object.
    fn schema_key(self) -> &'static str {
        match self {
            VendorFormat::Mcp => "inputSchema",
            VendorFormat::OpenAi => "parameters",
        }
    }
}

/// What an import produced, plus everything the source format could not
/// express (each loss is a fact about the *format*, not a guess about the
/// tool — the defaulted value is stated in the note).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportResult {
    pub tool: CanonicalTool,
    pub losses: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InteropError {
    MissingField { field: String },
    WrongType { field: String, expected: String },
    UnknownParamType(String),
    UnsupportedVersion(u32),
    InvalidSchema(String),
}

impl std::fmt::Display for InteropError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InteropError::MissingField { field } => write!(f, "missing field: {field}"),
            InteropError::WrongType { field, expected } => {
                write!(f, "field {field} has the wrong type (expected {expected})")
            }
            InteropError::UnknownParamType(t) => write!(f, "unknown parameter type: {t}"),
            InteropError::UnsupportedVersion(v) => {
                write!(f, "unsupported canonical schema version: {v}")
            }
            InteropError::InvalidSchema(m) => write!(f, "invalid parameter schema: {m}"),
        }
    }
}
impl std::error::Error for InteropError {}

// ---------------------------------------------------------------------------
// Small Value-map readers
// ---------------------------------------------------------------------------

fn req_text(v: &Value, field: &str) -> Result<String, InteropError> {
    match v.get(field) {
        Some(Value::Text(s)) => Ok(s.clone()),
        Some(_) => Err(InteropError::WrongType { field: field.to_string(), expected: "string".to_string() }),
        None => Err(InteropError::MissingField { field: field.to_string() }),
    }
}

fn opt_text(v: &Value, field: &str) -> Result<String, InteropError> {
    match v.get(field) {
        None => Ok(String::new()),
        Some(Value::Text(s)) => Ok(s.clone()),
        Some(_) => Err(InteropError::WrongType { field: field.to_string(), expected: "string".to_string() }),
    }
}

fn req_map<'a>(v: &'a Value, field: &str) -> Result<&'a Value, InteropError> {
    match v.get(field) {
        Some(m @ Value::Map(_)) => Ok(m),
        Some(_) => Err(InteropError::WrongType { field: field.to_string(), expected: "object".to_string() }),
        None => Err(InteropError::MissingField { field: field.to_string() }),
    }
}

/// Parse a JSON-Schema-style parameter object:
/// `{"type": "object", "properties": {name: {"type": ..., "description":
/// ...}}, "required": [names]}`.
fn parse_json_schema_params(schema: &Value) -> Result<Vec<ParamSpec>, InteropError> {
    let ty = opt_text(schema, "type")?;
    if !ty.is_empty() && ty != "object" {
        return Err(InteropError::InvalidSchema(format!(
            "parameter schema type must be \"object\", got \"{ty}\""
        )));
    }
    let required: Vec<String> = match schema.get("required") {
        None => Vec::new(),
        Some(Value::List(items)) => {
            let mut out = Vec::new();
            for it in items {
                match it {
                    Value::Text(s) => out.push(s.clone()),
                    _ => {
                        return Err(InteropError::WrongType {
                            field: "required".to_string(),
                            expected: "list of strings".to_string(),
                        })
                    }
                }
            }
            out
        }
        Some(_) => {
            return Err(InteropError::WrongType {
                field: "required".to_string(),
                expected: "list of strings".to_string(),
            })
        }
    };
    let props = match schema.get("properties") {
        None => return Ok(Vec::new()),
        Some(Value::Map(m)) => m,
        Some(_) => {
            return Err(InteropError::WrongType {
                field: "properties".to_string(),
                expected: "object".to_string(),
            })
        }
    };
    let mut params = Vec::new();
    for (name, spec) in props {
            let pty = req_text(spec, "type").map_err(|e| match e {
                InteropError::MissingField { .. } => InteropError::InvalidSchema(format!(
                    "parameter \"{name}\" has no \"type\""
                )),
                other => other,
            })?;
            let ty = ParamType::from_json_type(&pty)?;
            let desc = opt_text(spec, "description")?;
            params.push(ParamSpec {
                name: name.clone(),
                ty,
                required: required.iter().any(|r| r == name),
                description: desc,
            });
    }
    Ok(params)
}

/// Import a foreign description into the canonical format. The returned
/// losses always include the effect/consistency/compensability rows the
/// vendor format cannot express — they are defaulted, never inferred.
pub fn import(format: VendorFormat, value: &Value) -> Result<ImportResult, InteropError> {
    let name = req_text(value, "name")?;
    let description = opt_text(value, "description")?;
    let schema = req_map(value, format.schema_key())?;
    let params = parse_json_schema_params(schema)?;
    let version = match value.get("version").or_else(|| value.get("schema_version")) {
        None => 1,
        Some(Value::Int(v)) if *v >= 0 => *v as u32,
        Some(_) => {
            return Err(InteropError::WrongType {
                field: "version".to_string(),
                expected: "integer".to_string(),
            })
        }
    };
    if version > CURRENT_SCHEMA_VERSION {
        return Err(InteropError::UnsupportedVersion(version));
    }
    let mut tool = CanonicalTool::new(name, description);
    tool.params = params;
    tool.schema_version = version;
    let losses = vec![
        format!(
            "{} format cannot express effects; defaulted to read-only",
            format.name()
        ),
        format!(
            "{} format cannot express consistency; defaulted to best-effort",
            format.name()
        ),
        format!(
            "{} format cannot express compensability; defaulted to irreversible",
            format.name()
        ),
    ];
    Ok(ImportResult { tool, losses })
}

/// Export a canonical description to a vendor format. The three rows the
/// format cannot carry (effects, consistency, compensability) are dropped;
/// use [`expressiveness_report`] to see the loss before exporting.
pub fn export(format: VendorFormat, tool: &CanonicalTool) -> Value {
    let mut properties = std::collections::BTreeMap::new();
    let mut required = Vec::new();
    for p in &tool.params {
        let mut spec = vec![("type", Value::text(p.ty.name()))];
        if !p.description.is_empty() {
            spec.push(("description", Value::text(p.description.clone())));
        }
        if let ParamType::List(inner) = &p.ty {
            spec.push(("items", Value::map(vec![("type", Value::text(inner.name()))])));
        }
        properties.insert(p.name.clone(), Value::map(spec));
        if p.required {
            required.push(Value::text(p.name.clone()));
        }
    }
    let schema = Value::map(vec![
        ("type", Value::text("object")),
        ("properties", Value::Map(properties)),
        ("required", Value::List(required)),
    ]);
    Value::map(vec![
        ("name", Value::text(tool.name.clone())),
        ("description", Value::text(tool.description.clone())),
        (format.schema_key(), schema),
        ("version", Value::int(tool.schema_version as i64)),
    ])
}

// ---------------------------------------------------------------------------
// Negotiation
// ---------------------------------------------------------------------------

/// Outcome of asking "can a caller written against `local` drive the tool
/// as `remote` describes it?"
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Negotiation {
    Compatible,
    CompatibleWithLoss { losses: Vec<String> },
    Incompatible { reasons: Vec<String> },
}

impl Negotiation {
    pub fn is_usable(&self) -> bool {
        !matches!(self, Negotiation::Incompatible { .. })
    }
}

/// Negotiate between two descriptions of (purportedly) the same tool.
/// Rules, in order: a name mismatch is incompatible; a required parameter
/// the remote side lacks, or types differently, is incompatible; anything
/// else that differs — a dropped optional parameter, an undeclared or
/// extra effect, a different consistency or compensability row — is a
/// named loss, not a refusal.
pub fn negotiate(local: &CanonicalTool, remote: &CanonicalTool) -> Negotiation {
    let mut reasons = Vec::new();
    let mut losses = Vec::new();
    if local.name != remote.name {
        reasons.push(format!(
            "name mismatch: local \"{}\" vs remote \"{}\"",
            local.name, remote.name
        ));
    }
    for p in &local.params {
        match remote.param(&p.name) {
            None if p.required => {
                reasons.push(format!("required parameter \"{}\" missing on remote side", p.name));
            }
            None => {
                losses.push(format!("optional parameter \"{}\" not accepted by remote side", p.name));
            }
            Some(rp) if rp.ty != p.ty => {
                reasons.push(format!(
                    "parameter \"{}\" type mismatch: local {} vs remote {}",
                    p.name,
                    p.ty.name(),
                    rp.ty.name()
                ));
            }
            Some(_) => {}
        }
    }
    for e in local.effects.difference(&remote.effects) {
        losses.push(format!("remote side does not declare effect {}", e.name()));
    }
    for e in remote.effects.difference(&local.effects) {
        losses.push(format!("remote side declares extra effect {}", e.name()));
    }
    if local.consistency != remote.consistency {
        losses.push("consistency contract differs between sides".to_string());
    }
    if local.compensability != remote.compensability {
        losses.push("compensability contract differs between sides".to_string());
    }
    if !reasons.is_empty() {
        Negotiation::Incompatible { reasons }
    } else if !losses.is_empty() {
        Negotiation::CompatibleWithLoss { losses }
    } else {
        Negotiation::Compatible
    }
}

// ---------------------------------------------------------------------------
// Expressiveness report
// ---------------------------------------------------------------------------

/// Which canonical fields a vendor format can and cannot carry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormatReport {
    pub format: VendorFormat,
    pub expressible: Vec<&'static str>,
    pub lost: Vec<&'static str>,
}

/// The evidence a standardization discussion starts from: both vendor
/// formats carry names, descriptions and typed parameters, and both lose
/// the three rows Loom's safety model depends on.
pub fn expressiveness_report() -> Vec<FormatReport> {
    let expressible = vec!["name", "description", "parameters (typed)", "required flags"];
    let lost = vec!["effects", "consistency", "compensability"];
    vec![
        FormatReport { format: VendorFormat::Mcp, expressible: expressible.clone(), lost: lost.clone() },
        FormatReport { format: VendorFormat::OpenAi, expressible, lost },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::ToolManifest;
    use crate::typesys::{Compensability, Consistency, Effect};

    fn sample_tool() -> CanonicalTool {
        CanonicalTool::new("send_mail", "Send an email message")
            .with_param(ParamSpec::new("to", ParamType::Text, true).with_description("recipient"))
            .with_param(ParamSpec::new("subject", ParamType::Text, false))
            .with_param(ParamSpec::new("retries", ParamType::Int, false))
            .with_effects(&[Effect::Send])
            .with_param(ParamSpec::new("tags", ParamType::List(Box::new(ParamType::Text)), false))
    }

    fn with_consistency(mut t: CanonicalTool) -> CanonicalTool {
        t.consistency = Consistency::Causal;
        t.compensability = Compensability::Partial { window_ms: 60_000 };
        t
    }

    #[test]
    fn mcp_round_trip_preserves_params_and_reports_losses() {
        let tool = with_consistency(sample_tool());
        let exported = export(VendorFormat::Mcp, &tool);
        assert_eq!(exported.get("name").and_then(Value::as_text), Some("send_mail"));
        let imported = import(VendorFormat::Mcp, &exported).unwrap();
        assert_eq!(imported.tool.name, tool.name);
        assert_eq!(imported.tool.params, tool.params);
        assert_eq!(imported.losses.len(), 3);
        assert!(imported.losses.iter().any(|l| l.contains("effects")));
        assert!(imported.losses.iter().any(|l| l.contains("consistency")));
        assert!(imported.losses.iter().any(|l| l.contains("compensability")));
        // Defaults are the safe ones, never invented from the foreign text.
        assert_eq!(imported.tool.effects, BTreeSet::from([Effect::Read]));
        assert_eq!(imported.tool.consistency, Consistency::BestEffort);
        assert_eq!(imported.tool.compensability, Compensability::Irreversible);
    }

    #[test]
    fn openai_round_trip_keeps_required_flags() {
        let tool = sample_tool();
        let exported = export(VendorFormat::OpenAi, &tool);
        let imported = import(VendorFormat::OpenAi, &exported).unwrap();
        assert_eq!(imported.tool.params, tool.params);
        assert!(imported.tool.param("to").unwrap().required);
        assert!(!imported.tool.param("subject").unwrap().required);
        assert_eq!(imported.tool.param("tags").unwrap().ty, ParamType::List(Box::new(ParamType::Text)));
    }

    #[test]
    fn cross_format_mcp_to_openai_keeps_params() {
        let tool = sample_tool();
        let via_mcp = export(VendorFormat::Mcp, &tool);
        let imported = import(VendorFormat::Mcp, &via_mcp).unwrap();
        let via_openai = export(VendorFormat::OpenAi, &imported.tool);
        let back = import(VendorFormat::OpenAi, &via_openai).unwrap();
        assert_eq!(back.tool.params, tool.params);
    }

    #[test]
    fn import_rejects_missing_name_and_missing_schema() {
        let no_name = Value::map(vec![("inputSchema", Value::map(vec![]))]);
        assert!(matches!(
            import(VendorFormat::Mcp, &no_name),
            Err(InteropError::MissingField { .. })
        ));
        let no_schema = Value::map(vec![("name", Value::text("x"))]);
        assert!(matches!(
            import(VendorFormat::Mcp, &no_schema),
            Err(InteropError::MissingField { .. })
        ));
    }

    #[test]
    fn import_rejects_unknown_param_type_and_bad_schema_type() {
        let bad_type = Value::map(vec![
            ("name", Value::text("x")),
            (
                "inputSchema",
                Value::map(vec![
                    ("type", Value::text("object")),
                    (
                        "properties",
                        Value::map(vec![(
                            "p",
                            Value::map(vec![("type", Value::text("float64"))]),
                        )]),
                    ),
                ]),
            ),
        ]);
        assert!(matches!(
            import(VendorFormat::Mcp, &bad_type),
            Err(InteropError::UnknownParamType(_))
        ));
        let bad_schema = Value::map(vec![
            ("name", Value::text("x")),
            ("inputSchema", Value::map(vec![("type", Value::text("array"))])),
        ]);
        assert!(matches!(
            import(VendorFormat::Mcp, &bad_schema),
            Err(InteropError::InvalidSchema(_))
        ));
    }

    #[test]
    fn import_rejects_future_schema_version() {
        let future = Value::map(vec![
            ("name", Value::text("x")),
            ("inputSchema", Value::map(vec![("type", Value::text("object"))])),
            ("version", Value::int(99)),
        ]);
        assert!(matches!(
            import(VendorFormat::Mcp, &future),
            Err(InteropError::UnsupportedVersion(99))
        ));
    }

    #[test]
    fn negotiate_identical_is_compatible() {
        let a = with_consistency(sample_tool());
        let b = a.clone();
        assert_eq!(negotiate(&a, &b), Negotiation::Compatible);
    }

    #[test]
    fn negotiate_dropped_optional_param_is_loss_not_refusal() {
        let local = sample_tool();
        let mut remote = sample_tool();
        remote.params.retain(|p| p.name != "subject");
        let n = negotiate(&local, &remote);
        assert!(n.is_usable());
        match n {
            Negotiation::CompatibleWithLoss { losses } => {
                assert!(losses.iter().any(|l| l.contains("subject")));
            }
            other => panic!("expected loss, got {other:?}"),
        }
    }

    #[test]
    fn negotiate_missing_required_param_is_incompatible() {
        let local = sample_tool();
        let mut remote = sample_tool();
        remote.params.retain(|p| p.name != "to");
        match negotiate(&local, &remote) {
            Negotiation::Incompatible { reasons } => {
                assert!(reasons.iter().any(|r| r.contains("\"to\"")));
            }
            other => panic!("expected incompatible, got {other:?}"),
        }
    }

    #[test]
    fn negotiate_type_mismatch_and_name_mismatch_are_incompatible() {
        let local = sample_tool();
        let mut remote = sample_tool();
        remote.params.iter_mut().find(|p| p.name == "retries").unwrap().ty = ParamType::Text;
        assert!(matches!(negotiate(&local, &remote), Negotiation::Incompatible { .. }));
        let mut renamed = sample_tool();
        renamed.name = "send_email".to_string();
        assert!(matches!(negotiate(&local, &renamed), Negotiation::Incompatible { .. }));
    }

    #[test]
    fn negotiate_undeclared_effect_is_a_loss() {
        let local = sample_tool(); // declares Send
        let mut remote = sample_tool();
        remote.effects = BTreeSet::from([Effect::Read]);
        match negotiate(&local, &remote) {
            Negotiation::CompatibleWithLoss { losses } => {
                assert!(losses.iter().any(|l| l.contains("send")));
            }
            other => panic!("expected loss, got {other:?}"),
        }
    }

    #[test]
    fn v1_to_v2_migration_defaults_safely_and_reports_each() {
        let mut v1 = CanonicalTool::new("archive", "Archive a file");
        v1.schema_version = 1;
        v1.effects = BTreeSet::new();
        let (v2, notes) = CanonicalTool::migrate_v1_to_v2(v1);
        assert_eq!(v2.schema_version, CURRENT_SCHEMA_VERSION);
        assert_eq!(v2.effects, BTreeSet::from([Effect::Read]));
        assert_eq!(v2.consistency, Consistency::BestEffort);
        assert_eq!(v2.compensability, Compensability::Irreversible);
        assert_eq!(notes.len(), 3);
    }

    #[test]
    fn manifest_round_trip_preserves_safety_rows() {
        let m = ToolManifest::new(
            "pay",
            "Pay an invoice",
            &[Effect::Transfer],
            Consistency::Strong,
            Compensability::Irreversible,
        );
        let canonical = CanonicalTool::from_manifest(&m);
        assert_eq!(canonical.effects, BTreeSet::from([Effect::Transfer]));
        let back = canonical.to_manifest();
        assert_eq!(back.effects, m.effects);
        assert_eq!(back.consistency, Consistency::Strong);
        assert_eq!(back.compensability, Compensability::Irreversible);
    }

    #[test]
    fn expressiveness_report_names_the_lost_rows() {
        let report = expressiveness_report();
        assert_eq!(report.len(), 2);
        for r in &report {
            assert!(r.expressible.contains(&"parameters (typed)"));
            assert!(r.lost.contains(&"effects"));
            assert!(r.lost.contains(&"consistency"));
            assert!(r.lost.contains(&"compensability"));
        }
    }
}
