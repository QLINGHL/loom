//! Constrained generative UI (doc ch. 19, §1).
//!
//! Letting a model generate UI directly is tempting and unsafe: the
//! output is untestable, inaccessible, and unauditable. Loom's rule is
//! **constrained decoding to a component vocabulary**: the system ships a
//! fixed set of designed, tested, accessibility-labelled components, and
//! the model may only output a *declarative description* — which
//! components, what content, what layout. The system renders it.
//!
//! In one sentence: *the model decides what to say, the system decides
//! how to draw it.*
//!
//! This module is the system side of that split:
//!
//! * [`ComponentKind`] is the closed vocabulary. There is no escape
//!   hatch: a description naming a component outside the vocabulary is
//!   rejected, not rendered.
//! * [`UiNode`] is the declarative description: a component kind, a
//!   fixed set of string props, and children. It carries no code —
//!   there is nowhere in it to put code.
//! * [`UiDescription::validate`] enforces the vocabulary, the per
//!   component prop rules, the child rules, the accessibility rules
//!   (every interactive component needs a label, every image needs alt
//!   text), and a no-code rule on every prop value (no markup, no
//!   `javascript:` URIs, no event-handler props).
//! * [`UiDescription::render_text`] is the deterministic renderer. It
//!   refuses to render an invalid description, so a pixel on screen can
//!   only ever come from validated, trusted code — never from model
//!   output directly.

/// The closed component vocabulary. The model may only pick from these.
///
/// Each variant documents its contract: which props it takes, which are
/// required, and what children it accepts. Anything outside this list
/// does not exist as far as the renderer is concerned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComponentKind {
    /// One line of body text. Leaf. Prop: `content` (required).
    Text,
    /// A heading. Leaf. Props: `content` (required), `level` ("1"|"2"|"3").
    Heading,
    /// A pressable action. Leaf. Props: `label` (required, the
    /// accessible name), `action` (required, an intent/tool id).
    Button,
    /// A single-line text input. Leaf. Props: `label` (required),
    /// `placeholder` (optional), `value` (optional).
    TextField,
    /// A choice from a fixed list. Leaf. Props: `label` (required),
    /// `options` (required, `|`-separated, at least two non-empty entries).
    Select,
    /// A yes/no toggle. Leaf. Props: `label` (required),
    /// `checked` (optional, "true"|"false").
    Checkbox,
    /// A short notice. Leaf. Props: `content` (required),
    /// `tone` (optional, "info"|"warn"|"error").
    Alert,
    /// An image. Leaf. Props: `src` (required), `alt` (required — an
    /// image without alt text is not renderable).
    Image,
    /// A visual separator. Leaf. No props.
    Divider,
    /// A 0–100 completion indicator. Leaf. Props: `value` (required,
    /// integer 0–100), `label` (optional).
    Progress,
    /// A titled container. Children: any component.
    Card,
    /// A list. Children: only [`ComponentKind::ListItem`].
    List,
    /// One list row. Leaf. Prop: `content` (required).
    ListItem,
    /// A form. Children: only [`ComponentKind::TextField`],
    /// [`ComponentKind::Select`], [`ComponentKind::Checkbox`],
    /// [`ComponentKind::Button`].
    Form,
    /// A table. Props: `columns` (required, `|`-separated, at least one
    /// non-empty entry). Children: only [`ComponentKind::TableRow`].
    Table,
    /// One table row. Children: only [`ComponentKind::TableCell`].
    TableRow,
    /// One table cell. Leaf. Prop: `content` (required).
    TableCell,
}

/// Maximum nesting depth of a description (root counts as depth 1).
pub const MAX_DEPTH: usize = 5;
/// Maximum number of nodes in one description.
pub const MAX_NODES: usize = 64;

impl ComponentKind {
    /// Every component in the vocabulary, in the order the model is
    /// shown them.
    pub fn all() -> &'static [ComponentKind] {
        &[
            ComponentKind::Text,
            ComponentKind::Heading,
            ComponentKind::Button,
            ComponentKind::TextField,
            ComponentKind::Select,
            ComponentKind::Checkbox,
            ComponentKind::Alert,
            ComponentKind::Image,
            ComponentKind::Divider,
            ComponentKind::Progress,
            ComponentKind::Card,
            ComponentKind::List,
            ComponentKind::ListItem,
            ComponentKind::Form,
            ComponentKind::Table,
            ComponentKind::TableRow,
            ComponentKind::TableCell,
        ]
    }

    /// The name the model writes in a description.
    pub fn name(&self) -> &'static str {
        match self {
            ComponentKind::Text => "Text",
            ComponentKind::Heading => "Heading",
            ComponentKind::Button => "Button",
            ComponentKind::TextField => "TextField",
            ComponentKind::Select => "Select",
            ComponentKind::Checkbox => "Checkbox",
            ComponentKind::Alert => "Alert",
            ComponentKind::Image => "Image",
            ComponentKind::Divider => "Divider",
            ComponentKind::Progress => "Progress",
            ComponentKind::Card => "Card",
            ComponentKind::List => "List",
            ComponentKind::ListItem => "ListItem",
            ComponentKind::Form => "Form",
            ComponentKind::Table => "Table",
            ComponentKind::TableRow => "TableRow",
            ComponentKind::TableCell => "TableCell",
        }
    }

    /// Look a component up by the name the model wrote. `None` means
    /// the model invented a component — reject the description.
    pub fn from_name(name: &str) -> Option<ComponentKind> {
        Self::all().iter().copied().find(|k| k.name() == name)
    }

    /// One line the model is shown when it is told what this component
    /// is for. The vocabulary sheet is part of the constraint: the
    /// model picks from this list, it never defines its own.
    pub fn description(&self) -> &'static str {
        match self {
            ComponentKind::Text => "body text",
            ComponentKind::Heading => "a section heading",
            ComponentKind::Button => "a pressable action with an accessible label",
            ComponentKind::TextField => "a single-line text input",
            ComponentKind::Select => "a choice from a fixed list",
            ComponentKind::Checkbox => "a yes/no toggle",
            ComponentKind::Alert => "a short notice (info, warn or error)",
            ComponentKind::Image => "an image with alt text",
            ComponentKind::Divider => "a visual separator",
            ComponentKind::Progress => "a 0-100 completion indicator",
            ComponentKind::Card => "a titled container for other components",
            ComponentKind::List => "a list of ListItem rows",
            ComponentKind::ListItem => "one row inside a List",
            ComponentKind::Form => "a form holding inputs and a submit button",
            ComponentKind::Table => "a table with named columns",
            ComponentKind::TableRow => "one row inside a Table",
            ComponentKind::TableCell => "one cell inside a TableRow",
        }
    }

    /// Props this component accepts, in declaration order.
    pub fn allowed_props(&self) -> &'static [&'static str] {
        match self {
            ComponentKind::Text => &["content"],
            ComponentKind::Heading => &["content", "level"],
            ComponentKind::Button => &["label", "action"],
            ComponentKind::TextField => &["label", "placeholder", "value"],
            ComponentKind::Select => &["label", "options"],
            ComponentKind::Checkbox => &["label", "checked"],
            ComponentKind::Alert => &["content", "tone"],
            ComponentKind::Image => &["src", "alt"],
            ComponentKind::Divider => &[],
            ComponentKind::Progress => &["value", "label"],
            ComponentKind::Card => &["title"],
            ComponentKind::List => &[],
            ComponentKind::ListItem => &["content"],
            ComponentKind::Form => &["title", "submit_label"],
            ComponentKind::Table => &["columns"],
            ComponentKind::TableRow => &[],
            ComponentKind::TableCell => &["content"],
        }
    }

    /// Props that must be present and non-empty. Accessibility lives
    /// here: every interactive component requires a `label`, and
    /// [`ComponentKind::Image`] requires `alt`.
    pub fn required_props(&self) -> &'static [&'static str] {
        match self {
            ComponentKind::Text => &["content"],
            ComponentKind::Heading => &["content"],
            ComponentKind::Button => &["label", "action"],
            ComponentKind::TextField => &["label"],
            ComponentKind::Select => &["label", "options"],
            ComponentKind::Checkbox => &["label"],
            ComponentKind::Alert => &["content"],
            ComponentKind::Image => &["src", "alt"],
            ComponentKind::Divider => &[],
            ComponentKind::Progress => &["value"],
            ComponentKind::Card => &[],
            ComponentKind::List => &[],
            ComponentKind::ListItem => &["content"],
            ComponentKind::Form => &[],
            ComponentKind::Table => &["columns"],
            ComponentKind::TableRow => &[],
            ComponentKind::TableCell => &["content"],
        }
    }

    /// Which child kinds this component accepts. `None` means "any
    /// component"; `Some(&[])` means "no children at all" (a leaf).
    pub fn allowed_children(&self) -> Option<&'static [ComponentKind]> {
        match self {
            ComponentKind::Card => None,
            ComponentKind::List => Some(&[ComponentKind::ListItem]),
            ComponentKind::Form => Some(&[
                ComponentKind::TextField,
                ComponentKind::Select,
                ComponentKind::Checkbox,
                ComponentKind::Button,
            ]),
            ComponentKind::Table => Some(&[ComponentKind::TableRow]),
            ComponentKind::TableRow => Some(&[ComponentKind::TableCell]),
            _ => Some(&[]),
        }
    }

    /// Whether this component is interactive (needs an accessible
    /// label to be operable at all).
    pub fn is_interactive(&self) -> bool {
        matches!(
            self,
            ComponentKind::Button
                | ComponentKind::TextField
                | ComponentKind::Select
                | ComponentKind::Checkbox
        )
    }
}

/// One node of a declarative UI description: which component, what
/// props, what children. There is deliberately no way to express code
/// here — no event handlers, no style strings, no markup. A model that
/// wants a button writes `Button { label, action }`; the system decides
/// what that button looks like and what pressing it is allowed to do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UiNode {
    pub kind: ComponentKind,
    /// Props in the order the model wrote them; keys are checked
    /// against [`ComponentKind::allowed_props`].
    pub props: Vec<(String, String)>,
    pub children: Vec<UiNode>,
}

impl UiNode {
    pub fn new(kind: ComponentKind) -> Self {
        UiNode { kind, props: Vec::new(), children: Vec::new() }
    }

    pub fn prop(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.props.push((key.into(), value.into()));
        self
    }

    pub fn child(mut self, node: UiNode) -> Self {
        self.children.push(node);
        self
    }

    pub fn children(mut self, nodes: Vec<UiNode>) -> Self {
        self.children.extend(nodes);
        self
    }

    /// The value of one prop, if present.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.props.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }

    /// Build a node from a component name the model wrote. Returns
    /// `None` for an invented component — the caller must surface that
    /// as a validation error, not silently substitute something else.
    pub fn from_name(name: &str) -> Option<Self> {
        ComponentKind::from_name(name).map(UiNode::new)
    }
}

/// A complete declarative UI description: one root node. The root must
/// be a container ([`ComponentKind::Card`], [`ComponentKind::Form`],
/// [`ComponentKind::List`] or [`ComponentKind::Table`]) — a bare leaf
/// is not a screen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UiDescription {
    pub root: UiNode,
}

impl UiDescription {
    pub fn new(root: UiNode) -> Self {
        UiDescription { root }
    }

    /// Validate the whole description. Returns every problem found, as
    /// `path: message` strings; an empty vec means the description is
    /// renderable. Validation is all-or-nothing at render time, but
    /// collecting every problem (instead of failing on the first) lets
    /// the model fix all of them in one retry.
    pub fn validate(&self) -> Vec<String> {
        let mut problems = Vec::new();
        if !matches!(
            self.root.kind,
            ComponentKind::Card
                | ComponentKind::Form
                | ComponentKind::List
                | ComponentKind::Table
        ) {
            problems.push(format!(
                "root: root must be a container (Card, Form, List or Table), not {}",
                self.root.kind.name()
            ));
        }
        let mut count = 0usize;
        self.validate_node(&self.root, "root", 1, &mut count, &mut problems);
        if count > MAX_NODES {
            problems.push(format!(
                "root: description has {count} nodes, limit is {MAX_NODES}"
            ));
        }
        problems
    }

    fn validate_node(
        &self,
        node: &UiNode,
        path: &str,
        depth: usize,
        count: &mut usize,
        problems: &mut Vec<String>,
    ) {
        *count += 1;
        if depth > MAX_DEPTH {
            problems.push(format!(
                "{path}: nesting depth {depth} exceeds limit {MAX_DEPTH}"
            ));
            return;
        }

        // Props: only the vocabulary's props, no duplicates, required
        // ones present and non-empty, values free of code.
        let mut seen: Vec<&str> = Vec::new();
        for (key, value) in &node.props {
            if !node.kind.allowed_props().contains(&key.as_str()) {
                problems.push(format!(
                    "{path}: {key} is not a prop of {} (model tried to smuggle behaviour)",
                    node.kind.name()
                ));
                continue;
            }
            if seen.contains(&key.as_str()) {
                problems.push(format!("{path}: prop {key} given twice"));
            }
            seen.push(key.as_str());
            if let Some(problem) = check_prop_value(node.kind, key, value) {
                problems.push(format!("{path}: {problem}"));
            }
        }
        for required in node.kind.required_props() {
            match node.get(required) {
                None => problems.push(format!(
                    "{path}: missing required prop {required} on {}",
                    node.kind.name()
                )),
                Some(v) if v.trim().is_empty() => problems.push(format!(
                    "{path}: required prop {required} on {} is empty",
                    node.kind.name()
                )),
                _ => {}
            }
        }

        // Children: leaves take none, containers only their kinds.
        match node.kind.allowed_children() {
            Some(&[]) if !node.children.is_empty() => {
                problems.push(format!(
                    "{path}: {} is a leaf and takes no children",
                    node.kind.name()
                ));
            }
            Some(allowed) => {
                for (i, child) in node.children.iter().enumerate() {
                    if !allowed.contains(&child.kind) {
                        problems.push(format!(
                            "{path}.children[{i}]: {} cannot contain {}",
                            node.kind.name(),
                            child.kind.name()
                        ));
                    }
                }
            }
            None => {}
        }
        for (i, child) in node.children.iter().enumerate() {
            self.validate_node(
                child,
                &format!("{path}.children[{i}]"),
                depth + 1,
                count,
                problems,
            );
        }
    }

    /// Render the description to deterministic text. This is the
    /// system drawing the screen: it refuses (`Err`) an invalid
    /// description, so model output can never reach a pixel unchecked.
    pub fn render_text(&self) -> Result<String, Vec<String>> {
        let problems = self.validate();
        if !problems.is_empty() {
            return Err(problems);
        }
        let mut out = String::new();
        render_node(&self.root, 0, &mut out);
        Ok(out)
    }
}

/// Check one prop value for code and for its per-prop constraints.
/// Returns `Some(message)` when the value is not acceptable.
fn check_prop_value(kind: ComponentKind, key: &str, value: &str) -> Option<String> {
    // No-code rule: a declarative description never contains markup,
    // script URIs, or handler-like text. Any of these means the model
    // tried to write code instead of choosing a component.
    let lower = value.to_lowercase();
    if value.contains('<') || value.contains('>') {
        return Some(format!("prop {key} contains markup characters"));
    }
    if lower.contains("javascript:") || lower.contains("data:text/html") {
        return Some(format!("prop {key} contains a script URI"));
    }
    if lower.contains("onclick") || lower.contains("onload") || lower.contains("onerror") {
        return Some(format!("prop {key} contains an event handler"));
    }

    match (kind, key) {
        // An action is an intent/tool id, not code: letters, digits,
        // dots, dashes and underscores only, no spaces.
        (ComponentKind::Button, "action") => {
            if !value
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_'))
            {
                return Some(format!(
                    "prop action {value:?} is not a valid intent id (letters, digits, . - _ only)"
                ));
            }
            None
        }
        (ComponentKind::Heading, "level") => match value {
            "1" | "2" | "3" => None,
            _ => Some(format!("prop level must be 1, 2 or 3, got {value:?}")),
        },
        (ComponentKind::Alert, "tone") => match value {
            "info" | "warn" | "error" => None,
            _ => Some(format!("prop tone must be info, warn or error, got {value:?}")),
        },
        (ComponentKind::Checkbox, "checked") => match value {
            "true" | "false" => None,
            _ => Some(format!("prop checked must be true or false, got {value:?}")),
        },
        (ComponentKind::Progress, "value") => match value.parse::<i64>() {
            Ok(v) if (0..=100).contains(&v) => None,
            Ok(v) => Some(format!("prop value must be 0-100, got {v}")),
            Err(_) => Some(format!("prop value must be an integer 0-100, got {value:?}")),
        },
        (ComponentKind::Select, "options") => {
            let entries: Vec<&str> =
                value.split('|').map(str::trim).filter(|e| !e.is_empty()).collect();
            let raw = value.split('|').count();
            if entries.len() < 2 {
                return Some(format!(
                    "prop options needs at least two choices, got {entries:?}"
                ));
            }
            if entries.len() != raw {
                return Some("prop options contains an empty choice".to_string());
            }
            None
        }
        (ComponentKind::Table, "columns") => {
            let entries: Vec<&str> =
                value.split('|').map(str::trim).filter(|e| !e.is_empty()).collect();
            if entries.is_empty() {
                return Some("prop columns needs at least one column".to_string());
            }
            if entries.len() != value.split('|').count() {
                return Some("prop columns contains an empty column".to_string());
            }
            None
        }
        _ => None,
    }
}

fn render_node(node: &UiNode, indent: usize, out: &mut String) {
    let pad = "  ".repeat(indent);
    match node.kind {
        ComponentKind::Text => {
            out.push_str(&format!("{pad}{}\n", node.get("content").unwrap_or("")));
        }
        ComponentKind::Heading => {
            let level: usize =
                node.get("level").and_then(|v| v.parse().ok()).unwrap_or(1);
            out.push_str(&format!(
                "{pad}{} {}\n",
                "#".repeat(level),
                node.get("content").unwrap_or("")
            ));
        }
        ComponentKind::Button => {
            out.push_str(&format!(
                "{pad}[ {} ] -> {}\n",
                node.get("label").unwrap_or(""),
                node.get("action").unwrap_or("")
            ));
        }
        ComponentKind::TextField => {
            let placeholder = node.get("placeholder").unwrap_or("");
            let value = node.get("value").unwrap_or("");
            if placeholder.is_empty() {
                out.push_str(&format!(
                    "{pad}{}: [{}]\n",
                    node.get("label").unwrap_or(""),
                    value
                ));
            } else {
                out.push_str(&format!(
                    "{pad}{}: [{placeholder}] {value}\n",
                    node.get("label").unwrap_or("")
                ));
            }
        }
        ComponentKind::Select => {
            out.push_str(&format!(
                "{pad}{}: <{}>\n",
                node.get("label").unwrap_or(""),
                node.get("options").unwrap_or("")
            ));
        }
        ComponentKind::Checkbox => {
            let mark = if node.get("checked") == Some("true") { "x" } else { " " };
            out.push_str(&format!(
                "{pad}[{mark}] {}\n",
                node.get("label").unwrap_or("")
            ));
        }
        ComponentKind::Alert => {
            out.push_str(&format!(
                "{pad}! [{}] {}\n",
                node.get("tone").unwrap_or("info"),
                node.get("content").unwrap_or("")
            ));
        }
        ComponentKind::Image => {
            out.push_str(&format!(
                "{pad}[image: {}]\n",
                node.get("alt").unwrap_or("")
            ));
        }
        ComponentKind::Divider => {
            out.push_str(&format!("{pad}---\n"));
        }
        ComponentKind::Progress => {
            let value: i64 =
                node.get("value").and_then(|v| v.parse().ok()).unwrap_or(0);
            let filled = (value / 10).max(0) as usize;
            let bar = format!("{}{}", "#".repeat(filled), "-".repeat(10 - filled));
            match node.get("label") {
                Some(label) => {
                    out.push_str(&format!("{pad}{label}: [{bar}] {value}%\n"));
                }
                None => out.push_str(&format!("{pad}[{bar}] {value}%\n")),
            }
        }
        ComponentKind::Card => {
            match node.get("title") {
                Some(title) => out.push_str(&format!("{pad}## {title}\n")),
                None => out.push_str(&format!("{pad}##\n")),
            }
            for child in &node.children {
                render_node(child, indent + 1, out);
            }
        }
        ComponentKind::List => {
            for child in &node.children {
                render_node(child, indent, out);
            }
        }
        ComponentKind::ListItem => {
            out.push_str(&format!("{pad}- {}\n", node.get("content").unwrap_or("")));
        }
        ComponentKind::Form => {
            match node.get("title") {
                Some(title) => out.push_str(&format!("{pad}## {title}\n")),
                None => out.push_str(&format!("{pad}## form\n")),
            }
            for child in &node.children {
                render_node(child, indent + 1, out);
            }
        }
        ComponentKind::Table => {
            out.push_str(&format!(
                "{pad}| {} |\n",
                node.get("columns").unwrap_or("").replace('|', " | ")
            ));
            for child in &node.children {
                render_node(child, indent, out);
            }
        }
        ComponentKind::TableRow => {
            let cells: Vec<String> = node
                .children
                .iter()
                .map(|c| c.get("content").unwrap_or("").to_string())
                .collect();
            out.push_str(&format!("{pad}| {} |\n", cells.join(" | ")));
        }
        ComponentKind::TableCell => {
            out.push_str(&format!("{pad}{}\n", node.get("content").unwrap_or("")));
        }
    }
}

/// The vocabulary sheet handed to the model: one line per component.
/// The model picks from this list; it never invents a component.
pub fn vocabulary_sheet() -> String {
    let mut out = String::from("Available components (pick only from this list):\n");
    for kind in ComponentKind::all() {
        out.push_str(&format!(
            "- {}: {} (props: {}; required: {})\n",
            kind.name(),
            kind.description(),
            kind.allowed_props().join(", "),
            kind.required_props().join(", ")
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_form() -> UiDescription {
        UiDescription::new(
            UiNode::new(ComponentKind::Form)
                .prop("title", "报销单")
                .child(
                    UiNode::new(ComponentKind::TextField)
                        .prop("label", "发票金额")
                        .prop("placeholder", "0.00"),
                )
                .child(
                    UiNode::new(ComponentKind::Select)
                        .prop("label", "费用类型")
                        .prop("options", "差旅|餐饮|住宿"),
                )
                .child(
                    UiNode::new(ComponentKind::Checkbox)
                        .prop("label", "已核对发票")
                        .prop("checked", "true"),
                )
                .child(
                    UiNode::new(ComponentKind::Button)
                        .prop("label", "提交报销")
                        .prop("action", "expense.submit"),
                ),
        )
    }

    fn sample_card() -> UiDescription {
        UiDescription::new(
            UiNode::new(ComponentKind::Card)
                .prop("title", "行程")
                .child(UiNode::new(ComponentKind::Heading).prop("content", "去上海").prop("level", "2"))
                .child(UiNode::new(ComponentKind::Text).prop("content", "周五 21:30 出发"))
                .child(
                    UiNode::new(ComponentKind::List).children(vec![
                        UiNode::new(ComponentKind::ListItem).prop("content", "机票已出"),
                        UiNode::new(ComponentKind::ListItem).prop("content", "酒店已订"),
                    ]),
                )
                .child(UiNode::new(ComponentKind::Progress).prop("value", "70").prop("label", "准备进度"))
                .child(UiNode::new(ComponentKind::Alert).prop("content", "航班可能延误").prop("tone", "warn")),
        )
    }

    #[test]
    fn valid_form_and_card_pass_validation() {
        assert!(sample_form().validate().is_empty());
        assert!(sample_card().validate().is_empty());
    }

    #[test]
    fn render_text_is_deterministic_and_system_drawn() {
        let a = sample_card().render_text().unwrap();
        let b = sample_card().render_text().unwrap();
        assert_eq!(a, b);
        assert!(a.contains("## 行程"));
        assert!(a.contains("## 去上海"));
        assert!(a.contains("- 机票已出"));
        assert!(a.contains("[#######---] 70%"));
        assert!(a.contains("! [warn] 航班可能延误"));
        let form = sample_form().render_text().unwrap();
        assert!(form.contains("发票金额: [0.00]"));
        assert!(form.contains("费用类型: <差旅|餐饮|住宿>"));
        assert!(form.contains("[x] 已核对发票"));
        assert!(form.contains("[ 提交报销 ] -> expense.submit"));
    }

    #[test]
    fn render_refuses_invalid_description() {
        let bad = UiDescription::new(
            UiNode::new(ComponentKind::Card)
                .child(UiNode::new(ComponentKind::Button).prop("label", "按我")),
        );
        assert!(bad.render_text().is_err());
    }

    #[test]
    fn invented_component_name_is_not_in_vocabulary() {
        assert_eq!(ComponentKind::from_name("Video"), None);
        assert_eq!(ComponentKind::from_name("button"), None);
        assert_eq!(ComponentKind::from_name("Button"), Some(ComponentKind::Button));
        assert_eq!(ComponentKind::all().len(), 17);
    }

    #[test]
    fn root_must_be_a_container() {
        let d = UiDescription::new(UiNode::new(ComponentKind::Text).prop("content", "hi"));
        assert!(d.validate().iter().any(|p| p.contains("root must be a container")));
    }

    #[test]
    fn unknown_prop_is_rejected_as_smuggled_behaviour() {
        let d = UiDescription::new(
            UiNode::new(ComponentKind::Card).child(
                UiNode::new(ComponentKind::Button)
                    .prop("label", "点我")
                    .prop("action", "x.go")
                    .prop("onClick", "steal()"),
            ),
        );
        let problems = d.validate();
        assert!(problems.iter().any(|p| p.contains("onClick is not a prop of Button")));
    }

    #[test]
    fn markup_and_script_uris_are_rejected_everywhere() {
        let d = UiDescription::new(
            UiNode::new(ComponentKind::Card)
                .child(UiNode::new(ComponentKind::Text).prop("content", "<script>alert(1)</script>"))
                .child(
                    UiNode::new(ComponentKind::Button)
                        .prop("label", "x")
                        .prop("action", "javascript:evil()"),
                ),
        );
        let problems = d.validate();
        assert!(problems.iter().any(|p| p.contains("markup characters")));
        assert!(problems.iter().any(|p| p.contains("script URI") || p.contains("not a valid intent id")));
    }

    #[test]
    fn missing_required_props_are_flagged() {
        let d = UiDescription::new(
            UiNode::new(ComponentKind::Card)
                .child(UiNode::new(ComponentKind::Image).prop("src", "cat.png"))
                .child(UiNode::new(ComponentKind::Button).prop("label", "提交")),
        );
        let problems = d.validate();
        assert!(problems.iter().any(|p| p.contains("missing required prop alt")));
        assert!(problems.iter().any(|p| p.contains("missing required prop action")));
    }

    #[test]
    fn wrong_child_kind_and_leaf_children_are_flagged() {
        let d = UiDescription::new(
            UiNode::new(ComponentKind::Card)
                .child(
                    UiNode::new(ComponentKind::List).child(
                        UiNode::new(ComponentKind::Text).prop("content", "不是 ListItem"),
                    ),
                )
                .child(
                    UiNode::new(ComponentKind::Text)
                        .prop("content", "叶子")
                        .child(UiNode::new(ComponentKind::Text).prop("content", "不该有")),
                ),
        );
        let problems = d.validate();
        assert!(problems.iter().any(|p| p.contains("List cannot contain Text")));
        assert!(problems.iter().any(|p| p.contains("is a leaf and takes no children")));
    }

    #[test]
    fn prop_value_constraints_are_checked() {
        let d = UiDescription::new(
            UiNode::new(ComponentKind::Card)
                .child(UiNode::new(ComponentKind::Heading).prop("content", "h").prop("level", "9"))
                .child(UiNode::new(ComponentKind::Progress).prop("value", "140"))
                .child(
                    UiNode::new(ComponentKind::Select)
                        .prop("label", "选")
                        .prop("options", "只有一个"),
                )
                .child(
                    UiNode::new(ComponentKind::Button)
                        .prop("label", "x")
                        .prop("action", "do thing now"),
                ),
        );
        let problems = d.validate();
        assert!(problems.iter().any(|p| p.contains("level must be 1, 2 or 3")));
        assert!(problems.iter().any(|p| p.contains("value must be 0-100")));
        assert!(problems.iter().any(|p| p.contains("at least two choices")));
        assert!(problems.iter().any(|p| p.contains("not a valid intent id")));
    }

    #[test]
    fn deep_nesting_is_rejected() {
        let mut node = UiNode::new(ComponentKind::Text).prop("content", "底");
        for _ in 0..MAX_DEPTH {
            node = UiNode::new(ComponentKind::Card).child(node);
        }
        let d = UiDescription::new(node);
        assert!(d.validate().iter().any(|p| p.contains("nesting depth")));
    }

    #[test]
    fn table_requires_columns_and_only_rows_of_cells() {
        let ok = UiDescription::new(
            UiNode::new(ComponentKind::Table)
                .prop("columns", "航班|时间")
                .child(
                    UiNode::new(ComponentKind::TableRow).children(vec![
                        UiNode::new(ComponentKind::TableCell).prop("content", "MU123"),
                        UiNode::new(ComponentKind::TableCell).prop("content", "21:30"),
                    ]),
                ),
        );
        assert!(ok.validate().is_empty());
        assert!(ok.render_text().unwrap().contains("| MU123 | 21:30 |"));

        let bad = UiDescription::new(
            UiNode::new(ComponentKind::Table).child(
                UiNode::new(ComponentKind::Text).prop("content", "不是行"),
            ),
        );
        let problems = bad.validate();
        assert!(problems.iter().any(|p| p.contains("Table cannot contain Text")));
        assert!(problems.iter().any(|p| p.contains("missing required prop columns")));
    }

    #[test]
    fn vocabulary_sheet_lists_every_component() {
        let sheet = vocabulary_sheet();
        for kind in ComponentKind::all() {
            assert!(sheet.contains(kind.name()), "sheet missing {}", kind.name());
        }
        assert!(sheet.contains("pick only from this list"));
    }
}
