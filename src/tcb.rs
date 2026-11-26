//! Trusted computing base boundary and verification-strength grading
//! (doc ch. 20, "关键路径的验证范围").
//!
//! The whole point of pushing the model out of the kernel is that the
//! trusted computing base stays small enough to verify. This module is
//! the registry that says, for every component on the kernel's critical
//! paths, *which trust domain it lives in and how strongly it is
//! verified* — the table from doc ch. 20, made queryable so the rest of
//! the system (and its tests) can check the boundary instead of trusting
//! a comment:
//!
//! | 组件 | 信任域 | 验证强度 | 理由 |
//! |---|---|---|
//! | 能力撤销 | TCB 内 | 形式化证明 | 撤销不彻底等于没有撤销 |
//! | 负能力传播 | TCB 内 | 形式化证明 | 单调性是安全属性的核心 |
//! | IPC 消息隔离 | TCB 内 | 形式化证明 | seL4 已有，复用即可 |
//! | 调度器 | TCB 内 | 模型检验 | 不涉及安全，但涉及公平性 |
//! | AI 运行时 | TCB 外 | 不做形式化 | 它本来就不被信任 |
//!
//! The last row is the key: *because the AI runtime is not in the TCB,
//! it does not need to be verified*. The runtime is therefore broken
//! out into its five sub-components (intent compiler, model routing,
//! context, speculation, validation) — every one of them is outside the
//! TCB with no formal verification, by design.
//!
//! ## Implementation paths — do not reinvent what already exists
//!
//! Doc ch. 20 forbids re-implementing two layers, and this module
//! records that as data rather than prose:
//!
//! * **CHERI / Arm Morello** — capability hardware. The kernel's
//!   capability model is built on CHERI capability pointers (bounds,
//!   permissions, type, validity bit); the Morello board is the first
//!   industrial implementation and CheriBSD already runs on it. The
//!   capability rules that are new to Loom (revocation, negative
//!   capability propagation) still need formal proofs, but the pointer
//!   machinery underneath them is hardware, not a software simulation
//!   we wrote.
//! * **seL4** — the formally verified microkernel. IPC message
//!   isolation is *reused* from seL4, not re-proven here: seL4 is the
//!   only microkernel with code-level formal verification, it is
//!   capability-based already, and its IPC is among the fastest in its
//!   class. Starting from seL4 means the TCB's correctness already has
//!   a mathematical proof and only the parts Loom adds need proving.
//!
//! Everything else in the table is Loom's own code, written in Rust in
//! user space where it belongs (doc ch. 20, "用户态用 Rust": memory
//! safety is a hard requirement for long-running services that process
//! untrusted input, and Rust's type system is a natural fit for the
//! ch. 7 effect-type checks).

/// Where a component lives relative to the trust boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrustDomain {
    /// Inside the trusted computing base: a bug here can break a
    /// security property, so the component must be verified to the
    /// strength its [`TcbEntry`] demands.
    InTcb,
    /// Outside the trusted computing base: the component may read,
    /// think and speak, but it can never change the outside world on
    /// its own — every side effect goes through the kernel after a
    /// capability check (doc ch. 5, axiom I). Untrusted by design, so
    /// no formal verification is owed.
    OutsideTcb,
}

impl TrustDomain {
    /// Short Chinese label, matching the doc table.
    pub fn label(self) -> &'static str {
        match self {
            TrustDomain::InTcb => "TCB 内",
            TrustDomain::OutsideTcb => "TCB 外",
        }
    }
}

/// How strongly a component is verified (doc ch. 20, "验证强度").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum VerificationStrength {
    /// Mathematical proof of the security property (e.g. revocation is
    /// total, negative-capability propagation is monotone).
    FormalProof,
    /// Exhaustive state-space checking of the relevant property
    /// (fairness for the scheduler); weaker than a proof, but the
    /// property at stake is not a security property.
    ModelChecking,
    /// Deliberately unverified: the component is outside the TCB, so
    /// its correctness is not load-bearing for security.
    NoFormalVerification,
}

impl VerificationStrength {
    /// Short Chinese label, matching the doc table.
    pub fn label(self) -> &'static str {
        match self {
            VerificationStrength::FormalProof => "形式化证明",
            VerificationStrength::ModelChecking => "模型检验",
            VerificationStrength::NoFormalVerification => "不做形式化",
        }
    }
}

/// Where a component's implementation comes from (doc ch. 20:
/// "不自研已经存在的东西").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImplementationPath {
    /// Loom's own code (Rust, user space unless the row says kernel).
    /// This is the only path that can carry a fresh proof obligation.
    SelfDeveloped,
    /// Reused from seL4: the formally verified microkernel already
    /// proves this property, so Loom re-proves nothing.
    ReusedFromSeL4,
    /// Built on CHERI capability hardware (Arm Morello): the capability
    /// pointer machinery is hardware, not a software simulation. The
    /// Loom-specific rules on top of it still need their own proofs.
    OnCheriHardware,
}

impl ImplementationPath {
    /// Short Chinese label.
    pub fn label(self) -> &'static str {
        match self {
            ImplementationPath::SelfDeveloped => "自研（Rust）",
            ImplementationPath::ReusedFromSeL4 => "复用 seL4",
            ImplementationPath::OnCheriHardware => "CHERI 硬件之上",
        }
    }
}

/// A component on one of the kernel's critical paths (doc ch. 20 table,
/// with the AI runtime row expanded into its five sub-components).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Component {
    /// 能力撤销：撤销不彻底等于没有撤销。
    CapabilityRevocation,
    /// 负能力传播：单调性是安全属性的核心。
    NegativeCapabilityPropagation,
    /// IPC 消息隔离：seL4 已有，复用即可。
    IpcMessageIsolation,
    /// 调度器：不涉及安全，但涉及公平性。
    Scheduler,
    /// AI 运行时——意图编译器（doc ch. 7）。
    AiRuntimeIntentCompiler,
    /// AI 运行时——模型路由（doc ch. 21 开放问题 4，级联路由）。
    AiRuntimeModelRouting,
    /// AI 运行时——上下文（doc ch. 9，上下文调度器）。
    AiRuntimeContext,
    /// AI 运行时——投机（doc ch. 12，语义投机预取）。
    AiRuntimeSpeculation,
    /// AI 运行时——验证（计划级语义审计等概率性检查）。
    AiRuntimeValidation,
}

impl Component {
    /// Every component in the table, in doc order.
    pub const fn all() -> &'static [Component] {
        &[
            Component::CapabilityRevocation,
            Component::NegativeCapabilityPropagation,
            Component::IpcMessageIsolation,
            Component::Scheduler,
            Component::AiRuntimeIntentCompiler,
            Component::AiRuntimeModelRouting,
            Component::AiRuntimeContext,
            Component::AiRuntimeSpeculation,
            Component::AiRuntimeValidation,
        ]
    }

    /// Chinese name, matching the doc table.
    pub fn name(self) -> &'static str {
        match self {
            Component::CapabilityRevocation => "能力撤销",
            Component::NegativeCapabilityPropagation => "负能力传播",
            Component::IpcMessageIsolation => "IPC 消息隔离",
            Component::Scheduler => "调度器",
            Component::AiRuntimeIntentCompiler => "AI 运行时：意图编译器",
            Component::AiRuntimeModelRouting => "AI 运行时：模型路由",
            Component::AiRuntimeContext => "AI 运行时：上下文",
            Component::AiRuntimeSpeculation => "AI 运行时：投机",
            Component::AiRuntimeValidation => "AI 运行时：验证",
        }
    }

    /// True for the five AI-runtime sub-components: the model proposes,
    /// the kernel executes, and none of these may ever perform a side
    /// effect on their own.
    pub fn is_ai_runtime(self) -> bool {
        matches!(
            self,
            Component::AiRuntimeIntentCompiler
                | Component::AiRuntimeModelRouting
                | Component::AiRuntimeContext
                | Component::AiRuntimeSpeculation
                | Component::AiRuntimeValidation
        )
    }
}

impl std::fmt::Display for Component {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// One row of the doc ch. 20 verification-scope table: component →
/// trust domain → verification strength, plus why, and where the
/// implementation comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TcbEntry {
    /// The component this row grades.
    pub component: Component,
    /// Which side of the trust boundary it lives on.
    pub domain: TrustDomain,
    /// How strongly it is verified.
    pub strength: VerificationStrength,
    /// The doc's stated reason for this strength.
    pub rationale: &'static str,
    /// Where the implementation comes from; CHERI and seL4 are reused,
    /// never re-implemented.
    pub implementation: ImplementationPath,
}

impl TcbEntry {
    /// True iff the component is inside the TCB.
    pub fn in_tcb(self) -> bool {
        self.domain == TrustDomain::InTcb
    }
}

/// The full doc ch. 20 table, as data. Keep in doc order; [`entry`]
/// looks rows up by component.
pub static TCB_TABLE: &[TcbEntry] = &[
    TcbEntry {
        component: Component::CapabilityRevocation,
        domain: TrustDomain::InTcb,
        strength: VerificationStrength::FormalProof,
        rationale: "撤销不彻底等于没有撤销",
        implementation: ImplementationPath::OnCheriHardware,
    },
    TcbEntry {
        component: Component::NegativeCapabilityPropagation,
        domain: TrustDomain::InTcb,
        strength: VerificationStrength::FormalProof,
        rationale: "单调性是安全属性的核心",
        implementation: ImplementationPath::OnCheriHardware,
    },
    TcbEntry {
        component: Component::IpcMessageIsolation,
        domain: TrustDomain::InTcb,
        strength: VerificationStrength::FormalProof,
        rationale: "seL4 已有，复用即可",
        implementation: ImplementationPath::ReusedFromSeL4,
    },
    TcbEntry {
        component: Component::Scheduler,
        domain: TrustDomain::InTcb,
        strength: VerificationStrength::ModelChecking,
        rationale: "不涉及安全，但涉及公平性",
        implementation: ImplementationPath::SelfDeveloped,
    },
    TcbEntry {
        component: Component::AiRuntimeIntentCompiler,
        domain: TrustDomain::OutsideTcb,
        strength: VerificationStrength::NoFormalVerification,
        rationale: "它本来就不被信任",
        implementation: ImplementationPath::SelfDeveloped,
    },
    TcbEntry {
        component: Component::AiRuntimeModelRouting,
        domain: TrustDomain::OutsideTcb,
        strength: VerificationStrength::NoFormalVerification,
        rationale: "它本来就不被信任",
        implementation: ImplementationPath::SelfDeveloped,
    },
    TcbEntry {
        component: Component::AiRuntimeContext,
        domain: TrustDomain::OutsideTcb,
        strength: VerificationStrength::NoFormalVerification,
        rationale: "它本来就不被信任",
        implementation: ImplementationPath::SelfDeveloped,
    },
    TcbEntry {
        component: Component::AiRuntimeSpeculation,
        domain: TrustDomain::OutsideTcb,
        strength: VerificationStrength::NoFormalVerification,
        rationale: "它本来就不被信任",
        implementation: ImplementationPath::SelfDeveloped,
    },
    TcbEntry {
        component: Component::AiRuntimeValidation,
        domain: TrustDomain::OutsideTcb,
        strength: VerificationStrength::NoFormalVerification,
        rationale: "它本来就不被信任",
        implementation: ImplementationPath::SelfDeveloped,
    },
];

/// Look up a component's table row. Every [`Component`] has exactly
/// one row; a missing row is a bug in this module, so it panics rather
/// than returning a default that could silently mislabel a component.
pub fn entry(component: Component) -> &'static TcbEntry {
    TCB_TABLE
        .iter()
        .find(|row| row.component == component)
        .expect("TCB table must cover every component exactly once")
}

/// Which trust domain a component lives in.
pub fn trust_domain(component: Component) -> TrustDomain {
    entry(component).domain
}

/// True iff the component is inside the trusted computing base.
pub fn in_tcb(component: Component) -> bool {
    entry(component).in_tcb()
}

/// How strongly a component is verified.
pub fn verification_strength(component: Component) -> VerificationStrength {
    entry(component).strength
}

/// All components inside the TCB, in doc order.
pub fn tcb_components() -> Vec<Component> {
    Component::all().iter().copied().filter(|c| in_tcb(*c)).collect()
}

/// The three components that carry a formal-proof obligation
/// (doc ch. 20): capability revocation, negative capability
/// propagation, IPC message isolation.
pub fn formal_proof_components() -> Vec<Component> {
    Component::all()
        .iter()
        .copied()
        .filter(|c| verification_strength(*c) == VerificationStrength::FormalProof)
        .collect()
}

/// Components verified by model checking (the scheduler: fairness, not
/// safety).
pub fn model_checked_components() -> Vec<Component> {
    Component::all()
        .iter()
        .copied()
        .filter(|c| verification_strength(*c) == VerificationStrength::ModelChecking)
        .collect()
}

/// The five AI-runtime sub-components: outside the TCB by design, and
/// therefore owed no formal verification.
pub fn ai_runtime_components() -> Vec<Component> {
    Component::all().iter().copied().filter(|c| c.is_ai_runtime()).collect()
}

/// Check the boundary invariants that make the table meaningful:
///
/// * every component appears exactly once;
/// * a formal proof is only ever claimed for a component inside the
///   TCB — a proof claimed outside it would be a lie about what the
///   proof protects;
/// * every AI-runtime component is outside the TCB *and* carries no
///   formal verification — the model is untrusted, so a proof for it
///   would be both pointless and a sign the boundary had been drawn
///   in the wrong place;
/// * every component inside the TCB has at least model checking —
///   nothing trusted goes unverified.
pub fn boundary_holds() -> bool {
    // Exactly one row per component.
    if TCB_TABLE.len() != Component::all().len() {
        return false;
    }
    for component in Component::all() {
        if TCB_TABLE.iter().filter(|row| row.component == *component).count() != 1 {
            return false;
        }
    }
    for row in TCB_TABLE {
        // No formal proof outside the TCB.
        if row.strength == VerificationStrength::FormalProof && !row.in_tcb() {
            return false;
        }
        // AI runtime: outside the TCB, no formal verification.
        if row.component.is_ai_runtime()
            && (row.in_tcb() || row.strength != VerificationStrength::NoFormalVerification)
        {
            return false;
        }
        // Nothing inside the TCB goes unverified.
        if row.in_tcb() && row.strength == VerificationStrength::NoFormalVerification {
            return false;
        }
    }
    true
}

/// Render the table as a human-readable boundary report: one line per
/// component plus the headline consequence — the AI runtime is not in
/// the TCB, which is exactly why it needs no verification.
pub fn boundary_report() -> String {
    let mut out = String::from("可信计算基边界（设计文档第 20 章）\n");
    for row in TCB_TABLE {
        out.push_str(&format!(
            "- {}：{}，{}（{}；{}）\n",
            row.component.name(),
            row.domain.label(),
            row.strength.label(),
            row.rationale,
            row.implementation.label()
        ));
    }
    out.push_str("结论：AI 运行时不在 TCB 内，因此不需要形式化验证。\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_covers_every_component_exactly_once() {
        assert_eq!(TCB_TABLE.len(), Component::all().len());
        for component in Component::all() {
            let rows = TCB_TABLE.iter().filter(|row| row.component == *component).count();
            assert_eq!(rows, 1, "{component} must have exactly one table row");
            assert_eq!(entry(*component).component, *component);
        }
    }

    #[test]
    fn ai_runtime_components_are_outside_tcb_and_unverified() {
        let runtime = ai_runtime_components();
        assert_eq!(runtime.len(), 5, "intent compiler, model routing, context, speculation, validation");
        for component in runtime {
            assert!(component.is_ai_runtime());
            assert!(!in_tcb(component), "{component} must not be in the TCB");
            assert_eq!(trust_domain(component), TrustDomain::OutsideTcb);
            assert_eq!(
                verification_strength(component),
                VerificationStrength::NoFormalVerification,
                "{component} is untrusted by design, so no formal verification is owed"
            );
        }
    }

    #[test]
    fn formal_proof_set_is_exactly_the_three_doc_items() {
        let formal = formal_proof_components();
        assert_eq!(
            formal,
            vec![
                Component::CapabilityRevocation,
                Component::NegativeCapabilityPropagation,
                Component::IpcMessageIsolation,
            ]
        );
        for component in formal {
            assert!(in_tcb(component), "{component} carries a proof, so it must be in the TCB");
            assert_eq!(verification_strength(component), VerificationStrength::FormalProof);
        }
    }

    #[test]
    fn scheduler_is_model_checked_not_formally_proven() {
        assert!(in_tcb(Component::Scheduler));
        assert_eq!(
            verification_strength(Component::Scheduler),
            VerificationStrength::ModelChecking
        );
        assert_eq!(model_checked_components(), vec![Component::Scheduler]);
    }

    #[test]
    fn ipc_isolation_is_reused_from_sel4_not_reproven() {
        let row = entry(Component::IpcMessageIsolation);
        assert_eq!(row.implementation, ImplementationPath::ReusedFromSeL4);
        assert_eq!(row.rationale, "seL4 已有，复用即可");
    }

    #[test]
    fn capability_rules_ride_on_cheri_hardware() {
        for component in
            [Component::CapabilityRevocation, Component::NegativeCapabilityPropagation]
        {
            let row = entry(component);
            assert_eq!(row.implementation, ImplementationPath::OnCheriHardware);
            // The rules on top of the hardware still need their own proofs.
            assert_eq!(row.strength, VerificationStrength::FormalProof);
        }
    }

    #[test]
    fn boundary_invariants_hold() {
        assert!(boundary_holds());
        assert_eq!(tcb_components().len(), 4);
    }

    #[test]
    fn boundary_report_lists_every_component_and_the_conclusion() {
        let report = boundary_report();
        for component in Component::all() {
            assert!(report.contains(component.name()), "report must list {component}");
        }
        assert!(report.contains("形式化证明"));
        assert!(report.contains("模型检验"));
        assert!(report.contains("不做形式化"));
        assert!(report.contains("AI 运行时不在 TCB 内"));
    }
}
