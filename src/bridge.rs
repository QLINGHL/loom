//! Automation bridge (doc ch. 19, §3).
//!
//! Compatibility, not rewrite: every new operating system dies for lack
//! of applications, and Loom's answer is *reverse adaptation*. The bridge
//! takes a legacy application's accessibility / UI-automation element
//! tree and exposes it as a list of system tools — a declarative
//! manifest, the same "tools are drivers" idea as [`crate::tools`].
//! The legacy app changes not one line of code and gains the ability to
//! be called by an agent.
//!
//! Two reliability tiers, straight from the design doc:
//!
//! * [`Reliability::Guessed`] — the tool was derived from the element's
//!   role and accessible label. Usable, but low reliability: a label can
//!   be missing, ambiguous, or misleading, and a button's real effect is
//!   only ever inferred.
//! * [`Reliability::Semantic`] — the app *chose* to provide a semantic
//!   action ("渐进升级"): a named, stable action bound to an element.
//!   Semantic actions replace the guessed tool for their element, never
//!   duplicate it.
//!
//! The adapter is a *placeholder* in one specific sense: there is no
//! real platform UI-automation backend in this prototype (no OS calls,
//! no network), so [`BridgeManifest::invoke`] executes against a
//! simulated copy of the element tree. The mapping, validation, and
//! reliability rules are the real deliverable; a platform backend would
//! only swap the executor.

use crate::tools::ToolManifest;
use crate::typesys::{Compensability, Consistency, Effect};
use crate::Value;

/// Maximum depth of an element tree (root counts as depth 1).
pub const MAX_DEPTH: usize = 8;
/// Maximum number of elements in one tree.
pub const MAX_ELEMENTS: usize = 256;

/// What kind of element the accessibility tree reports. Only the
/// interactive kinds become tools; the rest are context — they give the
/// tree meaning, but an agent cannot act on a label or an image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementKind {
    /// Pressable. Becomes an `activate` tool.
    Button,
    /// Single-line text input. Becomes a `set_text` tool.
    TextField,
    /// Yes/no toggle. Becomes a `toggle` tool.
    Checkbox,
    /// Choice from a fixed list. Becomes a `choose` tool.
    Select,
    /// Navigational link. Becomes an `open` tool.
    Link,
    /// Static text. No tool.
    Label,
    /// Static image. No tool.
    Image,
    /// Structural grouping. No tool; its children may still be tools.
    Container,
}

impl ElementKind {
    /// Whether an element of this kind can be acted on at all.
    pub fn is_interactive(self) -> bool {
        matches!(
            self,
            ElementKind::Button
                | ElementKind::TextField
                | ElementKind::Checkbox
                | ElementKind::Select
                | ElementKind::Link
        )
    }

    /// The verb of the tool an element of this kind becomes.
    pub fn verb(self) -> &'static str {
        match self {
            ElementKind::Button => "activate",
            ElementKind::TextField => "set_text",
            ElementKind::Checkbox => "toggle",
            ElementKind::Select => "choose",
            ElementKind::Link => "open",
            ElementKind::Label | ElementKind::Image | ElementKind::Container => "read",
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            ElementKind::Button => "Button",
            ElementKind::TextField => "TextField",
            ElementKind::Checkbox => "Checkbox",
            ElementKind::Select => "Select",
            ElementKind::Link => "Link",
            ElementKind::Label => "Label",
            ElementKind::Image => "Image",
            ElementKind::Container => "Container",
        }
    }
}

/// One node of a legacy app's accessibility / UI-automation tree.
///
/// `id` is the app's own stable automation id (e.g. `"save-btn"`); it is
/// what makes a derived tool name stable across runs. `label` is the
/// accessible name the user hears — it may be absent, which is exactly
/// the case the guessed tier has to survive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UiElement {
    pub id: String,
    pub kind: ElementKind,
    pub label: Option<String>,
    /// Current value: text content for a [`ElementKind::TextField`],
    /// `"true"`/`"false"` for a [`ElementKind::Checkbox`], the selected
    /// option for a [`ElementKind::Select`].
    pub value: Option<String>,
    /// The fixed choices of a [`ElementKind::Select`]; empty otherwise.
    pub options: Vec<String>,
    pub enabled: bool,
    pub visible: bool,
    pub children: Vec<UiElement>,
}

impl UiElement {
    pub fn new(id: impl Into<String>, kind: ElementKind) -> Self {
        UiElement {
            id: id.into(),
            kind,
            label: None,
            value: None,
            options: Vec::new(),
            enabled: true,
            visible: true,
            children: Vec::new(),
        }
    }

    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn value(mut self, value: impl Into<String>) -> Self {
        self.value = Some(value.into());
        self
    }

    pub fn options(mut self, options: Vec<String>) -> Self {
        self.options = options;
        self
    }

    pub fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    pub fn visible(mut self, visible: bool) -> Self {
        self.visible = visible;
        self
    }

    pub fn child(mut self, node: UiElement) -> Self {
        self.children.push(node);
        self
    }

    pub fn children(mut self, nodes: Vec<UiElement>) -> Self {
        self.children.extend(nodes);
        self
    }

    /// Find an element by id anywhere in this subtree.
    pub fn find(&self, id: &str) -> Option<&UiElement> {
        if self.id == id {
            return Some(self);
        }
        self.children.iter().find_map(|c| c.find(id))
    }

    fn find_mut(&mut self, id: &str) -> Option<&mut UiElement> {
        if self.id == id {
            return Some(self);
        }
        self.children.iter_mut().find_map(|c| c.find_mut(id))
    }
}

/// The action a bridged tool performs on its element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BridgeAction {
    /// Press a button.
    Activate,
    /// Replace a text field's content. Arg: `text`.
    SetText,
    /// Flip a checkbox. No args.
    Toggle,
    /// Pick one of a select's options. Arg: `option`.
    Choose,
    /// Follow a link.
    Open,
}

impl BridgeAction {
    /// The action a given element kind maps to. `None` for kinds that
    /// are not interactive and therefore never become tools.
    pub fn for_kind(kind: ElementKind) -> Option<BridgeAction> {
        match kind {
            ElementKind::Button => Some(BridgeAction::Activate),
            ElementKind::TextField => Some(BridgeAction::SetText),
            ElementKind::Checkbox => Some(BridgeAction::Toggle),
            ElementKind::Select => Some(BridgeAction::Choose),
            ElementKind::Link => Some(BridgeAction::Open),
            ElementKind::Label | ElementKind::Image | ElementKind::Container => None,
        }
    }

    pub fn verb(self) -> &'static str {
        match self {
            BridgeAction::Activate => "activate",
            BridgeAction::SetText => "set_text",
            BridgeAction::Toggle => "toggle",
            BridgeAction::Choose => "choose",
            BridgeAction::Open => "open",
        }
    }

    /// The argument names this action takes.
    pub fn params(self) -> &'static [&'static str] {
        match self {
            BridgeAction::SetText => &["text"],
            BridgeAction::Choose => &["option"],
            BridgeAction::Activate | BridgeAction::Toggle | BridgeAction::Open => &[],
        }
    }

    /// The effect signature of the tool, for its manifest. Everything
    /// the bridge does to a legacy app is a write to that app's state,
    /// except that nothing here can be observed to be compensable: a
    /// button press cannot generally be undone from outside the app, so
    /// it is [`Compensability::Irreversible`]; text, toggle and choice
    /// can be set back, so they are [`Compensability::Full`].
    pub fn effects(self) -> (Effect, Compensability) {
        match self {
            BridgeAction::Activate | BridgeAction::Open => {
                (Effect::Write, Compensability::Irreversible)
            }
            BridgeAction::SetText | BridgeAction::Toggle | BridgeAction::Choose => {
                (Effect::Write, Compensability::Full)
            }
        }
    }
}

/// How much the system trusts a bridged tool's description of what it
/// does. This is the doc's "不提供也能用，只是可靠性低" made explicit:
/// a guessed tool works, but the agent (and the user approving it)
/// must know it was inferred from a label, not promised by the app.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reliability {
    /// Derived from the element's role and label. Usable, low reliability.
    Guessed,
    /// Provided by the app as a semantic action. The recommended path.
    Semantic,
}

impl Reliability {
    pub fn name(self) -> &'static str {
        match self {
            Reliability::Guessed => "guessed",
            Reliability::Semantic => "semantic",
        }
    }
}

/// A semantic action supplied by the app itself (the "渐进升级" path):
/// a named, stable action bound to one element. When one exists for an
/// element, the guessed tool for that element is replaced by this one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticAction {
    /// The tool name the app wants, e.g. `"notes.save"`. Must start
    /// with `"<app>."` and contain only letters, digits, `.`, `-`, `_`.
    pub name: String,
    pub description: String,
    /// The element id this action drives.
    pub element_id: String,
    pub action: BridgeAction,
    /// Argument names, in order. Must match the action's params unless
    /// the action takes none.
    pub params: Vec<String>,
}

impl SemanticAction {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        element_id: impl Into<String>,
        action: BridgeAction,
    ) -> Self {
        SemanticAction {
            name: name.into(),
            description: description.into(),
            element_id: element_id.into(),
            action,
            params: action.params().iter().map(|p| (*p).to_string()).collect(),
        }
    }
}

/// One tool exposed by the bridge: the derived [`ToolManifest`] the
/// registry enforces, plus the bridge-specific provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct BridgeTool {
    /// Tool name, e.g. `"notes.save-btn.activate"`.
    pub name: String,
    pub description: String,
    pub action: BridgeAction,
    /// Slash-joined path of element ids from the tree root, e.g.
    /// `"window/toolbar/save-btn"`.
    pub element_path: String,
    /// The element id this tool drives.
    pub element_id: String,
    pub reliability: Reliability,
    /// True when the element had no accessible label: the tool exists,
    /// but the agent is guessing at an unnamed control.
    pub unlabeled: bool,
    pub manifest: ToolManifest,
}

/// An interactive element that produced no tool, and why. Recorded,
/// never silently dropped: a caller that expected the tool must notice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkippedElement {
    pub path: String,
    pub reason: String,
}

/// What went wrong invoking a bridged tool against the simulated tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BridgeError {
    NotFound { tool: String },
    Disabled { tool: String },
    BadArgs(String),
    UnknownOption { tool: String, option: String },
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BridgeError::NotFound { tool } => write!(f, "no bridged tool named {tool}"),
            BridgeError::Disabled { tool } => {
                write!(f, "tool {tool}: its element is currently disabled")
            }
            BridgeError::BadArgs(m) => write!(f, "bad arguments: {m}"),
            BridgeError::UnknownOption { tool, option } => {
                write!(f, "tool {tool}: {option:?} is not one of the select's options")
            }
        }
    }
}
impl std::error::Error for BridgeError {}

/// The full exposure of one legacy app: every tool the bridge derived,
/// the elements it skipped, and the simulated element tree the tools
/// drive. Register each [`BridgeTool::manifest`] with a
/// [`crate::tools::ToolRegistry`] to get approval pinning and capability
/// checks; the registry is the enforcement point, this module is the
/// adapter.
#[derive(Debug, Clone)]
pub struct BridgeManifest {
    pub app: String,
    pub tools: Vec<BridgeTool>,
    pub skipped: Vec<SkippedElement>,
    tree: UiElement,
}

impl BridgeManifest {
    /// Expose `root` as tools. Returns every structural problem found
    /// instead of a half-built manifest: duplicate element ids, tool
    /// names that collide after sanitizing, a select with no options,
    /// a tree past the depth/node limits, or a semantic action that
    /// names an unknown element, repeats an element, or mismatches its
    /// element's kind.
    pub fn expose(
        app: &str,
        root: UiElement,
        semantic_actions: &[SemanticAction],
    ) -> Result<BridgeManifest, Vec<String>> {
        let mut problems = Vec::new();
        if !is_valid_app_name(app) {
            problems.push(format!(
                "app: {app:?} is not a valid app name (letters, digits, - _ only, non-empty)"
            ));
        }

        let mut count = 0usize;
        let mut seen_ids: Vec<String> = Vec::new();
        Self::validate_tree(&root, "root", 1, &mut count, &mut seen_ids, &mut problems);
        if count > MAX_ELEMENTS {
            problems.push(format!(
                "root: tree has {count} elements, limit is {MAX_ELEMENTS}"
            ));
        }

        // Semantic actions are checked against the tree before any tool
        // is derived, so a bad override fails the whole exposure.
        let mut semantic_by_element: Vec<&SemanticAction> = Vec::new();
        for action in semantic_actions {
            if !is_valid_tool_name(&action.name) {
                problems.push(format!(
                    "semantic action {:?}: {:?} is not a valid tool name",
                    action.element_id, action.name
                ));
            } else if !action.name.starts_with(&format!("{app}.")) {
                problems.push(format!(
                    "semantic action {:?}: tool name {:?} must start with {app:?}.",
                    action.element_id, action.name
                ));
            }
            match root.find(&action.element_id) {
                None => problems.push(format!(
                    "semantic action {:?}: no element with id {:?} in the tree",
                    action.name, action.element_id
                )),
                Some(el) => {
                    if BridgeAction::for_kind(el.kind) != Some(action.action) {
                        problems.push(format!(
                            "semantic action {:?}: action {:?} does not fit element {:?} ({})",
                            action.name,
                            action.action.verb(),
                            action.element_id,
                            el.kind.name()
                        ));
                    }
                    if !el.enabled || !el.visible {
                        problems.push(format!(
                            "semantic action {:?}: element {:?} is disabled or hidden",
                            action.name, action.element_id
                        ));
                    }
                }
            }
            if semantic_by_element.iter().any(|a: &&SemanticAction| a.element_id == action.element_id) {
                problems.push(format!(
                    "semantic action {:?}: element {:?} already has a semantic action",
                    action.name, action.element_id
                ));
            } else {
                semantic_by_element.push(action);
            }
            let expected = action.action.params();
            if action.params.iter().map(String::as_str).collect::<Vec<&str>>() != expected {
                problems.push(format!(
                    "semantic action {:?}: params must be {:?}, got {:?}",
                    action.name, expected, action.params
                ));
            }
        }

        if !problems.is_empty() {
            return Err(problems);
        }

        let mut manifest = BridgeManifest {
            app: app.to_string(),
            tools: Vec::new(),
            skipped: Vec::new(),
            tree: root,
        };
        manifest.derive_tools(&semantic_by_element, &mut problems);
        if !problems.is_empty() {
            return Err(problems);
        }
        Ok(manifest)
    }

    fn validate_tree(
        node: &UiElement,
        path: &str,
        depth: usize,
        count: &mut usize,
        seen_ids: &mut Vec<String>,
        problems: &mut Vec<String>,
    ) {
        *count += 1;
        if depth > MAX_DEPTH {
            problems.push(format!(
                "{path}: nesting depth {depth} exceeds limit {MAX_DEPTH}"
            ));
            return;
        }
        if node.id.trim().is_empty() {
            problems.push(format!("{path}: element id is empty"));
        } else if seen_ids.contains(&node.id) {
            problems.push(format!(
                "{path}: duplicate element id {:?}",
                node.id
            ));
        } else {
            seen_ids.push(node.id.clone());
        }
        if node.kind == ElementKind::Select && node.options.is_empty() {
            problems.push(format!(
                "{path}: select {:?} declares no options",
                node.id
            ));
        }
        if sanitize_segment(&node.id).is_empty() && !node.id.trim().is_empty() {
            problems.push(format!(
                "{path}: element id {:?} sanitizes to an empty tool-name segment",
                node.id
            ));
        }
        for (i, child) in node.children.iter().enumerate() {
            Self::validate_tree(
                child,
                &format!("{path}/{}/{}", node.kind.name(), child.id),
                depth + 1,
                count,
                seen_ids,
                problems,
            );
            let _ = i;
        }
    }

    fn derive_tools(
        &mut self,
        semantic_by_element: &[&SemanticAction],
        problems: &mut Vec<String>,
    ) {
        let root = self.tree.clone();
        self.walk(&root, "root", semantic_by_element, problems);
    }

    fn walk(
        &mut self,
        node: &UiElement,
        path: &str,
        semantic_by_element: &[&SemanticAction],
        problems: &mut Vec<String>,
    ) {
        let node_path = if path == "root" {
            node.id.clone()
        } else {
            format!("{path}/{}", node.id)
        };
        if node.kind.is_interactive() {
            if !node.enabled || !node.visible {
                self.skipped.push(SkippedElement {
                    path: node_path.clone(),
                    reason: if !node.enabled {
                        "element is disabled".to_string()
                    } else {
                        "element is hidden".to_string()
                    },
                });
            } else {
                let action = BridgeAction::for_kind(node.kind)
                    .expect("interactive kinds all map to an action");
                let unlabeled =
                    node.label.as_ref().is_none_or(|l| l.trim().is_empty());
                if let Some(semantic) =
                    semantic_by_element.iter().find(|a| a.element_id == node.id)
                {
                    let (effect, compensability) = semantic.action.effects();
                    let tool = BridgeTool {
                        name: semantic.name.clone(),
                        description: semantic.description.clone(),
                        action: semantic.action,
                        element_path: node_path.clone(),
                        element_id: node.id.clone(),
                        reliability: Reliability::Semantic,
                        unlabeled,
                        manifest: ToolManifest::new(
                            semantic.name.clone(),
                            semantic.description.clone(),
                            &[effect],
                            Consistency::Causal,
                            compensability,
                        ),
                    };
                    self.push_tool(tool, problems);
                } else {
                    let name = format!(
                        "{}.{}.{}",
                        self.app,
                        sanitize_segment(&node.id),
                        action.verb()
                    );
                    let description = describe_guessed(&self.app, node, action);
                    let (effect, compensability) = action.effects();
                    let tool = BridgeTool {
                        name: name.clone(),
                        description: description.clone(),
                        action,
                        element_path: node_path.clone(),
                        element_id: node.id.clone(),
                        reliability: Reliability::Guessed,
                        unlabeled,
                        manifest: ToolManifest::new(
                            name,
                            description,
                            &[effect],
                            Consistency::Causal,
                            compensability,
                        ),
                    };
                    self.push_tool(tool, problems);
                }
            }
        }
        for child in &node.children {
            self.walk(child, &node_path, semantic_by_element, problems);
        }
    }

    fn push_tool(&mut self, tool: BridgeTool, problems: &mut Vec<String>) {
        if self.tools.iter().any(|t| t.name == tool.name) {
            problems.push(format!(
                "{}: duplicate tool name {:?} (two elements sanitize to the same name)",
                tool.element_path, tool.name
            ));
            return;
        }
        self.tools.push(tool);
    }

    /// The tools, in tree order.
    pub fn tools(&self) -> &[BridgeTool] {
        &self.tools
    }

    /// Find one tool by name.
    pub fn tool(&self, name: &str) -> Option<&BridgeTool> {
        self.tools.iter().find(|t| t.name == name)
    }

    /// How many tools are semantic (app-provided) vs guessed.
    pub fn reliability_counts(&self) -> (usize, usize) {
        let semantic =
            self.tools.iter().filter(|t| t.reliability == Reliability::Semantic).count();
        (semantic, self.tools.len() - semantic)
    }

    /// Human-readable listing of the exposure: what the app can now do,
    /// how much of it the app promised vs the bridge guessed, and what
    /// was left out.
    pub fn report(&self) -> String {
        let (semantic, guessed) = self.reliability_counts();
        let mut out = format!(
            "Automation bridge for app {:?}: {} tools ({} semantic, {} guessed)\n",
            self.app,
            self.tools.len(),
            semantic,
            guessed
        );
        for tool in &self.tools {
            out.push_str(&format!(
                "- {} [{}] ({}) — {}\n",
                tool.name,
                tool.action.verb(),
                tool.reliability.name(),
                tool.description
            ));
        }
        if !self.skipped.is_empty() {
            out.push_str("Skipped (not exposed):\n");
            for s in &self.skipped {
                out.push_str(&format!("- {}: {}\n", s.path, s.reason));
            }
        }
        out
    }

    /// Invoke a bridged tool against the simulated element tree.
    /// Validates arguments against the action's declared params, refuses
    /// a tool whose element has since become disabled, and applies the
    /// state change (set text, flip a checkbox, pick an option) so the
    /// result reflects what the app would now show.
    pub fn invoke(&mut self, name: &str, args: &Value) -> Result<Value, BridgeError> {
        let tool = self
            .tool(name)
            .ok_or_else(|| BridgeError::NotFound { tool: name.to_string() })?
            .clone();
        let element = self
            .tree
            .find(&tool.element_id)
            .ok_or_else(|| BridgeError::NotFound { tool: name.to_string() })?;
        if !element.enabled || !element.visible {
            return Err(BridgeError::Disabled { tool: name.to_string() });
        }
        match tool.action {
            BridgeAction::Activate | BridgeAction::Open => {
                if !matches!(args, Value::Null) && !is_empty_map(args) {
                    return Err(BridgeError::BadArgs(format!(
                        "{} takes no arguments",
                        tool.action.verb()
                    )));
                }
                Ok(Value::map(vec![
                    ("action", Value::text(tool.action.verb())),
                    ("element", Value::text(&tool.element_id)),
                    ("result", Value::text("done")),
                ]))
            }
            BridgeAction::SetText => {
                let Some(text) = args.get("text").and_then(Value::as_text) else {
                    return Err(BridgeError::BadArgs(
                        "set_text requires a text argument".to_string(),
                    ));
                };
                let text = text.to_string();
                let element = self.tree.find_mut(&tool.element_id).expect("checked above");
                element.value = Some(text.clone());
                Ok(Value::map(vec![
                    ("action", Value::text("set_text")),
                    ("element", Value::text(&tool.element_id)),
                    ("value", Value::text(text)),
                ]))
            }
            BridgeAction::Toggle => {
                if !matches!(args, Value::Null) && !is_empty_map(args) {
                    return Err(BridgeError::BadArgs("toggle takes no arguments".to_string()));
                }
                let element = self.tree.find_mut(&tool.element_id).expect("checked above");
                let now = element.value.as_deref() != Some("true");
                element.value = Some(now.to_string());
                Ok(Value::map(vec![
                    ("action", Value::text("toggle")),
                    ("element", Value::text(&tool.element_id)),
                    ("value", Value::Bool(now)),
                ]))
            }
            BridgeAction::Choose => {
                let Some(option) = args.get("option").and_then(Value::as_text) else {
                    return Err(BridgeError::BadArgs(
                        "choose requires an option argument".to_string(),
                    ));
                };
                let element = self.tree.find(&tool.element_id).expect("checked above");
                if !element.options.iter().any(|o| o == option) {
                    return Err(BridgeError::UnknownOption {
                        tool: name.to_string(),
                        option: option.to_string(),
                    });
                }
                let option = option.to_string();
                let element = self.tree.find_mut(&tool.element_id).expect("checked above");
                element.value = Some(option.clone());
                Ok(Value::map(vec![
                    ("action", Value::text("choose")),
                    ("element", Value::text(&tool.element_id)),
                    ("value", Value::text(option)),
                ]))
            }
        }
    }

    /// The simulated tree, for tests and for re-exposure after the app
    /// changes its UI. The tree is the adapter's input, not state the
    /// system owns: a real backend would re-read it from the app.
    pub fn tree(&self) -> &UiElement {
        &self.tree
    }
}

fn is_empty_map(v: &Value) -> bool {
    matches!(v, Value::Map(m) if m.is_empty())
}

/// A guessed tool's description. When the element has a label the
/// description names it; without one it says so plainly, because an
/// agent acting on an unnamed control is guessing twice over.
fn describe_guessed(app: &str, node: &UiElement, action: BridgeAction) -> String {
    let kind = node.kind.name().to_lowercase();
    match node.label.as_ref().filter(|l| !l.trim().is_empty()) {
        Some(label) => format!(
            "{} the {:?} {} in app {:?} (bridged from its accessibility tree)",
            capitalize(action.verb()),
            label,
            kind,
            app
        ),
        None => format!(
            "{} an unlabeled {} at {:?} in app {:?} (bridged from its accessibility tree — low reliability: no accessible name)",
            capitalize(action.verb()),
            kind,
            node.id,
            app
        ),
    }
}

fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(first) => first.to_ascii_uppercase().to_string() + chars.as_str(),
        None => String::new(),
    }
}

/// An app name is one identifier segment: letters, digits, `-`, `_`.
fn is_valid_app_name(app: &str) -> bool {
    !app.is_empty()
        && app.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

/// A tool name is dot-separated identifier segments.
fn is_valid_tool_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .split('.')
            .all(|seg| !seg.is_empty() && seg.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'))
}

/// Sanitize one element id into a tool-name segment: lowercase, runs of
/// anything that is not a letter, digit, `-` or `_` become one `_`,
/// leading and trailing separators trimmed. `.` is sanitized away so an
/// id can never smuggle an extra tool-name segment in.
fn sanitize_segment(id: &str) -> String {
    let mut out = String::new();
    let mut last_sep = true; // trims leading separators
    for c in id.to_lowercase().chars() {
        if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
            out.push(c);
            last_sep = false;
        } else if !last_sep {
            out.push('_');
            last_sep = true;
        }
    }
    while out.ends_with('_') || out.ends_with('-') {
        out.pop();
    }
    out.trim_start_matches(['_', '-']).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small legacy notes app: a window with a toolbar (new, save),
    /// a search field, a sort select, an auto-save checkbox, and a
    /// disabled delete button plus a hidden debug button.
    fn notes_app() -> UiElement {
        UiElement::new("window", ElementKind::Container).children(vec![
            UiElement::new("toolbar", ElementKind::Container).children(vec![
                UiElement::new("new-btn", ElementKind::Button).label("新建笔记"),
                UiElement::new("save-btn", ElementKind::Button).label("保存"),
                UiElement::new("delete-btn", ElementKind::Button)
                    .label("删除")
                    .enabled(false),
                UiElement::new("debug-btn", ElementKind::Button)
                    .label("调试")
                    .visible(false),
            ]),
            UiElement::new("search", ElementKind::TextField).label("搜索笔记"),
            UiElement::new("sort", ElementKind::Select)
                .label("排序")
                .options(vec!["最新".to_string(), "标题".to_string()])
                .value("最新"),
            UiElement::new("autosave", ElementKind::Checkbox)
                .label("自动保存")
                .value("false"),
            UiElement::new("hint", ElementKind::Label).label("共 3 条笔记"),
        ])
    }

    #[test]
    fn legacy_tree_becomes_a_tool_manifest() {
        let m = BridgeManifest::expose("notes", notes_app(), &[]).unwrap();
        let names: Vec<&str> = m.tools().iter().map(|t| t.name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "notes.new-btn.activate",
                "notes.save-btn.activate",
                "notes.search.set_text",
                "notes.sort.choose",
                "notes.autosave.toggle",
            ]
        );
        assert!(m.tools().iter().all(|t| t.reliability == Reliability::Guessed));
        let (semantic, guessed) = m.reliability_counts();
        assert_eq!((semantic, guessed), (0, 5));
        // Labels and containers are context, not tools.
        assert!(m.tool("notes.hint.read").is_none());
        // Disabled and hidden buttons are skipped, and the skip is recorded.
        assert_eq!(m.skipped.len(), 2);
        assert!(m.skipped.iter().any(|s| s.path.contains("delete-btn")));
        assert!(m.skipped.iter().any(|s| s.path.contains("debug-btn")));
    }

    #[test]
    fn manifests_carry_the_right_effects_and_compensability() {
        let m = BridgeManifest::expose("notes", notes_app(), &[]).unwrap();
        let save = m.tool("notes.save-btn.activate").unwrap();
        assert!(save.manifest.effects.contains(&Effect::Write));
        assert_eq!(save.manifest.compensability, Compensability::Irreversible);
        assert_eq!(save.manifest.consistency, Consistency::Causal);
        let set = m.tool("notes.search.set_text").unwrap();
        assert_eq!(set.manifest.compensability, Compensability::Full);
        let toggle = m.tool("notes.autosave.toggle").unwrap();
        assert_eq!(toggle.manifest.compensability, Compensability::Full);
    }

    #[test]
    fn unlabeled_control_is_exposed_but_flagged() {
        let tree = UiElement::new("window", ElementKind::Container).child(
            UiElement::new("btn-7", ElementKind::Button),
        );
        let m = BridgeManifest::expose("notes", tree, &[]).unwrap();
        let tool = &m.tools()[0];
        assert!(tool.unlabeled);
        assert_eq!(tool.reliability, Reliability::Guessed);
        assert!(tool.description.contains("unlabeled"));
        assert!(tool.description.contains("low reliability"));
    }

    #[test]
    fn duplicate_element_ids_are_rejected() {
        let tree = UiElement::new("window", ElementKind::Container).children(vec![
            UiElement::new("save-btn", ElementKind::Button).label("保存"),
            UiElement::new("save-btn", ElementKind::Button).label("另存为"),
        ]);
        let err = BridgeManifest::expose("notes", tree, &[]).unwrap_err();
        assert!(err.iter().any(|p| p.contains("duplicate element id")));
    }

    #[test]
    fn ids_that_sanitize_to_the_same_tool_name_are_rejected() {
        let tree = UiElement::new("window", ElementKind::Container).children(vec![
            UiElement::new("save btn", ElementKind::Button).label("保存"),
            UiElement::new("save_btn", ElementKind::Button).label("另存为"),
        ]);
        let err = BridgeManifest::expose("notes", tree, &[]).unwrap_err();
        assert!(err.iter().any(|p| p.contains("duplicate tool name")));
    }

    #[test]
    fn select_without_options_is_rejected() {
        let tree = UiElement::new("window", ElementKind::Container)
            .child(UiElement::new("sort", ElementKind::Select).label("排序"));
        let err = BridgeManifest::expose("notes", tree, &[]).unwrap_err();
        assert!(err.iter().any(|p| p.contains("declares no options")));
    }

    #[test]
    fn semantic_action_replaces_the_guessed_tool() {
        let semantic = SemanticAction::new(
            "notes.save_note",
            "Save the current note to disk.",
            "save-btn",
            BridgeAction::Activate,
        );
        let m = BridgeManifest::expose("notes", notes_app(), &[semantic]).unwrap();
        assert_eq!(m.tools().len(), 5);
        let save = m.tool("notes.save_note").unwrap();
        assert_eq!(save.reliability, Reliability::Semantic);
        assert_eq!(save.element_id, "save-btn");
        assert_eq!(save.description, "Save the current note to disk.");
        assert!(m.tool("notes.save-btn.activate").is_none());
        let (semantic_count, guessed) = m.reliability_counts();
        assert_eq!((semantic_count, guessed), (1, 4));
    }

    #[test]
    fn bad_semantic_actions_are_rejected() {
        let tree = notes_app();
        // Unknown element.
        let err = BridgeManifest::expose(
            "notes",
            tree.clone(),
            &[SemanticAction::new("notes.x", "x", "no-such", BridgeAction::Activate)],
        )
        .unwrap_err();
        assert!(err.iter().any(|p| p.contains("no element with id")));
        // Action that does not fit the element's kind.
        let err = BridgeManifest::expose(
            "notes",
            tree.clone(),
            &[SemanticAction::new("notes.x", "x", "save-btn", BridgeAction::SetText)],
        )
        .unwrap_err();
        assert!(err.iter().any(|p| p.contains("does not fit element")));
        // Two actions for one element.
        let err = BridgeManifest::expose(
            "notes",
            tree.clone(),
            &[
                SemanticAction::new("notes.a", "a", "save-btn", BridgeAction::Activate),
                SemanticAction::new("notes.b", "b", "save-btn", BridgeAction::Activate),
            ],
        )
        .unwrap_err();
        assert!(err.iter().any(|p| p.contains("already has a semantic action")));
        // Name not under the app.
        let err = BridgeManifest::expose(
            "notes",
            tree,
            &[SemanticAction::new("other.save", "x", "save-btn", BridgeAction::Activate)],
        )
        .unwrap_err();
        assert!(err.iter().any(|p| p.contains("must start with")));
    }

    #[test]
    fn invoke_drives_the_simulated_tree() {
        let mut m = BridgeManifest::expose("notes", notes_app(), &[]).unwrap();
        let out = m
            .invoke("notes.search.set_text", &Value::map(vec![("text", Value::text("报销"))]))
            .unwrap();
        assert_eq!(out.get("value"), Some(&Value::text("报销")));
        assert_eq!(
            m.tree().find("search").unwrap().value.as_deref(),
            Some("报销")
        );
        let out = m.invoke("notes.autosave.toggle", &Value::Null).unwrap();
        assert_eq!(out.get("value"), Some(&Value::Bool(true)));
        let out = m
            .invoke("notes.sort.choose", &Value::map(vec![("option", Value::text("标题"))]))
            .unwrap();
        assert_eq!(out.get("value"), Some(&Value::text("标题")));
        let out = m.invoke("notes.save-btn.activate", &Value::Null).unwrap();
        assert_eq!(out.get("result"), Some(&Value::text("done")));
    }

    #[test]
    fn invoke_rejects_bad_args_unknown_options_and_missing_tools() {
        let mut m = BridgeManifest::expose("notes", notes_app(), &[]).unwrap();
        assert_eq!(
            m.invoke("notes.search.set_text", &Value::Null),
            Err(BridgeError::BadArgs("set_text requires a text argument".to_string()))
        );
        assert_eq!(
            m.invoke(
                "notes.sort.choose",
                &Value::map(vec![("option", Value::text("不存在"))])
            ),
            Err(BridgeError::UnknownOption {
                tool: "notes.sort.choose".to_string(),
                option: "不存在".to_string(),
            })
        );
        assert_eq!(
            m.invoke("notes.nope.activate", &Value::Null),
            Err(BridgeError::NotFound { tool: "notes.nope.activate".to_string() })
        );
        assert_eq!(
            m.invoke("notes.save-btn.activate", &Value::map(vec![("x", Value::int(1))])),
            Err(BridgeError::BadArgs("activate takes no arguments".to_string()))
        );
    }

    #[test]
    fn report_lists_tools_and_skips() {
        let m = BridgeManifest::expose("notes", notes_app(), &[]).unwrap();
        let report = m.report();
        assert!(report.contains("Automation bridge for app \"notes\": 5 tools (0 semantic, 5 guessed)"));
        assert!(report.contains("notes.save-btn.activate [activate] (guessed)"));
        assert!(report.contains("Skipped (not exposed):"));
        assert!(report.contains("delete-btn"));
    }

    #[test]
    fn empty_and_label_only_trees_expose_nothing() {
        let tree = UiElement::new("window", ElementKind::Container)
            .child(UiElement::new("hint", ElementKind::Label).label("hello"));
        let m = BridgeManifest::expose("notes", tree, &[]).unwrap();
        assert!(m.tools().is_empty());
        assert!(m.skipped.is_empty());
        assert!(m.report().contains("0 tools"));
    }

    #[test]
    fn depth_limit_is_enforced() {
        let mut node = UiElement::new("leaf", ElementKind::Button).label("底");
        for i in 0..MAX_DEPTH {
            node = UiElement::new(format!("c{i}"), ElementKind::Container).child(node);
        }
        let err = BridgeManifest::expose("notes", node, &[]).unwrap_err();
        assert!(err.iter().any(|p| p.contains("nesting depth")));
    }

    #[test]
    fn sanitize_segment_collapses_runs_and_trims() {
        assert_eq!(sanitize_segment("Save Btn!"), "save_btn");
        assert_eq!(sanitize_segment("---a--b---"), "a--b");
        assert_eq!(sanitize_segment("a.b"), "a_b");
        assert_eq!(sanitize_segment(""), "");
    }
}
