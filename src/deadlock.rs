//! Deadlock detection and prevention (doc ch. 16 failure taxonomy,
//! "deadlock detection", and ch. 17, "resource global ordering").
//!
//! Two agents can wait on each other forever — A waits for the file lock
//! B holds while B waits for the calendar capability A holds. The design
//! doc gives two answers, and this module implements both:
//!
//! * **Prevention — global resource order.** Every agent must request
//!   resources in one canonical order. The order is lexicographic on the
//!   resource id: deterministic, total, and derivable at plan-compile time
//!   from the declared effect rows, so no agent can invent a private
//!   order. [`check_acquire_order`] validates a batch up front, and
//!   [`OrderedAllocator::request`] refuses a request for a resource ranked
//!   below anything the agent already holds — an agent that obeys the
//!   order can never be part of a cycle, because every wait edge then
//!   points from a lower-ranked holder to a higher-ranked one.
//! * **Detection — wait-for graph.** Prevention only binds agents that
//!   go through the allocator; a legacy tool or a buggy plan can still
//!   wait out of order. [`WaitForGraph`] therefore records who holds
//!   what and who waits for what, and every new wait edge is checked
//!   for a cycle. A cycle is a deadlock, full stop — no timeout
//!   heuristics, no guessing.
//!
//! The OS handling for a deadlock follows doc ch. 16: abort, switch
//! strategy, or escalate to a human. [`WaitForGraph::resolve`] aborts
//! the youngest waiter in the cycle (the one that waited most recently,
//! so the most work survives), releases everything it held, and returns
//! the resolution as a value — the caller owns the escalation path
//! (a `yield` in doc terms). The resolution maps to
//! [`FailureKind::Deadlock`](crate::watchdog::FailureKind::Deadlock) in
//! the failure taxonomy.
//!
//! This is a single-threaded prototype, like the rest of the crate:
//! agents are modelled by interleaving requests, and a real scheduler
//! would interleave real agents at exactly those points.

use std::collections::{BTreeSet, HashMap, HashSet};

/// A batch of resources requested out of the global order (doc ch. 17).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrderViolation {
    /// The resource that appeared first but ranks after `later`.
    pub earlier: String,
    /// The resource that should have come first.
    pub later: String,
    /// Index of `later` in the requested batch.
    pub position: usize,
}

impl std::fmt::Display for OrderViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "acquire order violated: {:?} must precede {:?} (position {})",
            self.later, self.earlier, self.position
        )
    }
}

impl std::error::Error for OrderViolation {}

/// Check a batch of acquire requests against the global resource order
/// (doc ch. 17): the batch must be in strictly ascending lexicographic
/// order of resource id. A duplicate in one batch is also a violation —
/// acquiring the same resource twice in one plan is a plan bug, not a
/// second acquisition.
pub fn check_acquire_order(resources: &[&str]) -> Result<(), OrderViolation> {
    for (i, window) in resources.windows(2).enumerate() {
        if window[0] >= window[1] {
            return Err(OrderViolation {
                earlier: window[0].to_string(),
                later: window[1].to_string(),
                position: i + 1,
            });
        }
    }
    Ok(())
}

/// The outcome of a single resource request against the wait-for graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequestOutcome {
    /// The resource was free (or already held by this agent) and is now held.
    Granted,
    /// The resource is held by another agent; the requester now waits.
    Waiting {
        /// The agent currently holding the resource.
        held_by: String,
    },
    /// This wait closed a cycle in the wait-for graph: a deadlock.
    /// `cycle` lists the agents in the cycle, starting at the requester,
    /// following wait edges back to the requester.
    Deadlock {
        cycle: Vec<String>,
    },
}

/// The outcome of a single ordered resource request: the global order is
/// enforced on top of the wait-for graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrderedOutcome {
    /// The request went through to the graph; see [`RequestOutcome`].
    Request(RequestOutcome),
    /// Refused without touching the graph: the agent already holds a
    /// resource ranked above the requested one, so granting this wait
    /// could close a cycle. The agent must release and re-acquire in
    /// order instead.
    OrderViolation {
        /// The highest-ranked resource the agent already holds.
        held_higher: String,
    },
}

/// What resolving a deadlock did (doc ch. 16: abort, then escalate).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeadlockResolution {
    /// The aborted agent: the youngest waiter in the cycle.
    pub aborted: String,
    /// Resources the aborted agent held, now released (sorted).
    pub released: Vec<String>,
    /// The cycle that was broken.
    pub cycle: Vec<String>,
}

impl DeadlockResolution {
    /// The failure-taxonomy class of this event (doc ch. 16).
    pub fn failure_kind(&self) -> crate::watchdog::FailureKind {
        crate::watchdog::FailureKind::Deadlock
    }
}

/// The wait-for graph: who holds which resource, who waits for which.
/// A wait edge runs from a waiting agent to the holder of the resource
/// it waits for; a cycle in that graph is a deadlock.
#[derive(Debug, Default)]
pub struct WaitForGraph {
    /// Resource id -> holding agent.
    holders: HashMap<String, String>,
    /// Agent -> resources it holds.
    held: HashMap<String, BTreeSet<String>>,
    /// Agent -> resources it is waiting for.
    waiting: HashMap<String, BTreeSet<String>>,
    /// Agent -> sequence number of its most recent wait. Higher means
    /// the agent waited more recently, i.e. it is the younger waiter.
    wait_seq: HashMap<String, u64>,
    next_seq: u64,
}

impl WaitForGraph {
    /// An empty graph.
    pub fn new() -> Self {
        WaitForGraph::default()
    }

    /// The agent currently holding `resource`, if any.
    pub fn holder_of(&self, resource: &str) -> Option<&str> {
        self.holders.get(resource).map(String::as_str)
    }

    /// Resources `agent` currently holds, in global order.
    pub fn held_by(&self, agent: &str) -> Vec<String> {
        self.held
            .get(agent)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Resources `agent` is currently waiting for, in global order.
    pub fn waiting_on(&self, agent: &str) -> Vec<String> {
        self.waiting
            .get(agent)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Request `resource` for `agent`: grant it if free, otherwise record
    /// the wait and check whether the new edge closed a cycle.
    pub fn request(&mut self, agent: &str, resource: &str) -> RequestOutcome {
        // Already held by this agent: idempotent, like an act() retried
        // with the same idempotency key (doc ch. 15).
        if self.holders.get(resource).map(String::as_str) == Some(agent) {
            return RequestOutcome::Granted;
        }
        match self.holders.get(resource) {
            None => {
                self.waiting
                    .entry(agent.to_string())
                    .or_default()
                    .remove(resource);
                self.holders.insert(resource.to_string(), agent.to_string());
                self.held
                    .entry(agent.to_string())
                    .or_default()
                    .insert(resource.to_string());
                RequestOutcome::Granted
            }
            Some(holder) => {
                let holder = holder.clone();
                self.waiting
                    .entry(agent.to_string())
                    .or_default()
                    .insert(resource.to_string());
                self.next_seq += 1;
                self.wait_seq.insert(agent.to_string(), self.next_seq);
                match self.cycle_from(agent) {
                    Some(cycle) => RequestOutcome::Deadlock { cycle },
                    None => RequestOutcome::Waiting { held_by: holder },
                }
            }
        }
    }

    /// Release one resource held by `agent`. Returns `true` if the agent
    /// held it. A released resource does not auto-grant to a waiter —
    /// the waiter re-requests and is granted then, which keeps the
    /// grant point (and its cycle check) in exactly one place.
    pub fn release(&mut self, agent: &str, resource: &str) -> bool {
        if self.holders.get(resource).map(String::as_str) != Some(agent) {
            return false;
        }
        self.holders.remove(resource);
        if let Some(held) = self.held.get_mut(agent) {
            held.remove(resource);
        }
        true
    }

    /// Release everything `agent` holds and drop its outstanding waits.
    /// Returns the released resources, in global order.
    pub fn release_all(&mut self, agent: &str) -> Vec<String> {
        let released = self.held_by(agent);
        for resource in &released {
            self.holders.remove(resource);
        }
        self.held.remove(agent);
        self.waiting.remove(agent);
        self.wait_seq.remove(agent);
        released
    }

    /// Abort the youngest waiter in `cycle` and release everything it
    /// held, breaking the deadlock (doc ch. 16: abort, then escalate).
    /// The youngest waiter is the one that waited most recently; aborting
    /// it preserves the most already-done work in the cycle. Ties on the
    /// wait sequence are broken by agent id so the choice is deterministic.
    pub fn resolve(&mut self, cycle: &[String]) -> DeadlockResolution {
        let aborted = cycle
            .iter()
            .max_by(|a, b| {
                let sa = self.wait_seq.get(*a).copied().unwrap_or(0);
                let sb = self.wait_seq.get(*b).copied().unwrap_or(0);
                sa.cmp(&sb).then_with(|| a.cmp(b))
            })
            .cloned()
            .unwrap_or_default();
        let released = self.release_all(&aborted);
        DeadlockResolution {
            aborted,
            released,
            cycle: cycle.to_vec(),
        }
    }

    /// Follow wait edges from `start`: `agent -> holder of each resource
    /// it waits for`. Returns a cycle as a list of agents starting at the
    /// first repeated agent if one is reachable, `None` otherwise.
    /// Every wait edge out of an agent is explored: a cycle through a
    /// second waited resource must be found just as surely as one
    /// through the first.
    fn cycle_from(&self, start: &str) -> Option<Vec<String>> {
        let mut path: Vec<String> = Vec::new();
        let mut on_path: HashSet<String> = HashSet::new();
        let mut visited: HashSet<String> = HashSet::new();
        self.dfs(start, &mut path, &mut on_path, &mut visited)
    }

    fn dfs(
        &self,
        node: &str,
        path: &mut Vec<String>,
        on_path: &mut HashSet<String>,
        visited: &mut HashSet<String>,
    ) -> Option<Vec<String>> {
        if on_path.contains(node) {
            let at = path.iter().position(|a| a == node).unwrap();
            return Some(path[at..].to_vec());
        }
        if !visited.insert(node.to_string()) {
            // Fully explored from here before with no cycle found.
            return None;
        }
        on_path.insert(node.to_string());
        path.push(node.to_string());
        if let Some(waits) = self.waiting.get(node) {
            for resource in waits {
                if let Some(holder) = self.holders.get(resource) {
                    if let Some(cycle) = self.dfs(holder, path, on_path, visited) {
                        return Some(cycle);
                    }
                }
            }
        }
        path.pop();
        on_path.remove(node);
        None
    }
}

/// An allocator that enforces the global resource order on top of the
/// wait-for graph (doc ch. 17): prevention first, detection as the
/// safety net for requests that bypass the order.
#[derive(Debug, Default)]
pub struct OrderedAllocator {
    graph: WaitForGraph,
}

impl OrderedAllocator {
    /// An empty allocator.
    pub fn new() -> Self {
        OrderedAllocator::default()
    }

    /// The underlying wait-for graph (read-only view for inspection).
    pub fn graph(&self) -> &WaitForGraph {
        &self.graph
    }

    /// Request `resource` for `agent`, enforcing the global order: an
    /// agent holding any resource ranked above `resource` is refused
    /// outright — it must release and re-acquire in order. Requests
    /// that pass the order check go to the graph, where a cycle is
    /// still detected if the order was somehow bypassed elsewhere.
    pub fn request(&mut self, agent: &str, resource: &str) -> OrderedOutcome {
        if let Some(held_higher) = self
            .graph
            .held_by(agent)
            .into_iter()
            .filter(|held| held.as_str() > resource)
            .max()
        {
            return OrderedOutcome::OrderViolation { held_higher };
        }
        OrderedOutcome::Request(self.graph.request(agent, resource))
    }

    /// Release one resource held by `agent`; see [`WaitForGraph::release`].
    pub fn release(&mut self, agent: &str, resource: &str) -> bool {
        self.graph.release(agent, resource)
    }

    /// Release everything `agent` holds; see [`WaitForGraph::release_all`].
    pub fn release_all(&mut self, agent: &str) -> Vec<String> {
        self.graph.release_all(agent)
    }

    /// Break a detected deadlock; see [`WaitForGraph::resolve`].
    pub fn resolve(&mut self, cycle: &[String]) -> DeadlockResolution {
        self.graph.resolve(cycle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::watchdog::FailureKind;

    #[test]
    fn ordered_batch_passes() {
        assert!(check_acquire_order(&["cap:calendar", "cap:mail", "file:doc"]).is_ok());
        assert!(check_acquire_order(&["only-one"]).is_ok());
        assert!(check_acquire_order(&[]).is_ok());
    }

    #[test]
    fn out_of_order_batch_reports_the_pair() {
        let err = check_acquire_order(&["cap:mail", "cap:calendar"]).unwrap_err();
        assert_eq!(err.earlier, "cap:mail");
        assert_eq!(err.later, "cap:calendar");
        assert_eq!(err.position, 1);
        assert!(err.to_string().contains("cap:calendar"));
    }

    #[test]
    fn duplicate_in_one_batch_is_a_violation() {
        assert!(check_acquire_order(&["file:doc", "file:doc"]).is_err());
    }

    #[test]
    fn single_resource_grant_and_release() {
        let mut g = WaitForGraph::new();
        assert_eq!(g.request("a", "file:doc"), RequestOutcome::Granted);
        assert_eq!(g.holder_of("file:doc"), Some("a"));
        assert_eq!(g.held_by("a"), vec!["file:doc".to_string()]);
        assert!(g.release("a", "file:doc"));
        assert_eq!(g.holder_of("file:doc"), None);
        assert!(!g.release("a", "file:doc"), "releasing twice is a no-op");
        assert_eq!(g.request("b", "file:doc"), RequestOutcome::Granted);
    }

    #[test]
    fn requesting_a_resource_you_hold_is_idempotent() {
        let mut g = WaitForGraph::new();
        assert_eq!(g.request("a", "file:doc"), RequestOutcome::Granted);
        assert_eq!(g.request("a", "file:doc"), RequestOutcome::Granted);
        assert_eq!(g.held_by("a"), vec!["file:doc".to_string()]);
    }

    #[test]
    fn two_agent_cycle_is_detected() {
        let mut g = WaitForGraph::new();
        assert_eq!(g.request("a", "file:doc"), RequestOutcome::Granted);
        assert_eq!(g.request("b", "cap:calendar"), RequestOutcome::Granted);
        // A waits for the calendar B holds; B waits for the doc A holds.
        assert!(matches!(
            g.request("a", "cap:calendar"),
            RequestOutcome::Waiting { .. }
        ));
        let outcome = g.request("b", "file:doc");
        match outcome {
            RequestOutcome::Deadlock { cycle } => {
                assert_eq!(cycle.len(), 2);
                assert!(cycle.contains(&"a".to_string()));
                assert!(cycle.contains(&"b".to_string()));
                assert_eq!(cycle[0], "b", "cycle starts at the requester");
            }
            other => panic!("expected a deadlock, got {other:?}"),
        }
    }

    #[test]
    fn three_agent_cycle_is_detected() {
        let mut g = WaitForGraph::new();
        assert_eq!(g.request("a", "r1"), RequestOutcome::Granted);
        assert_eq!(g.request("b", "r2"), RequestOutcome::Granted);
        assert_eq!(g.request("c", "r3"), RequestOutcome::Granted);
        assert!(matches!(g.request("a", "r2"), RequestOutcome::Waiting { .. }));
        assert!(matches!(g.request("b", "r3"), RequestOutcome::Waiting { .. }));
        let outcome = g.request("c", "r1");
        match outcome {
            RequestOutcome::Deadlock { cycle } => assert_eq!(cycle.len(), 3),
            other => panic!("expected a deadlock, got {other:?}"),
        }
    }

    #[test]
    fn cycle_through_a_second_waited_resource_is_found() {
        let mut g = WaitForGraph::new();
        assert_eq!(g.request("a", "r3"), RequestOutcome::Granted);
        assert_eq!(g.request("b", "r2"), RequestOutcome::Granted);
        assert_eq!(g.request("c", "r1"), RequestOutcome::Granted);
        // A waits on r1 (held by C, a dead end) and on r2 (held by B).
        assert!(matches!(g.request("a", "r1"), RequestOutcome::Waiting { .. }));
        // B waits on r3, which A holds: the cycle A -> B -> A runs
        // through A's *second* waited resource, not the first.
        assert!(matches!(g.request("b", "r3"), RequestOutcome::Waiting { .. }));
        match g.request("a", "r2") {
            RequestOutcome::Deadlock { cycle } => {
                assert_eq!(cycle.len(), 2);
                assert!(cycle.contains(&"a".to_string()));
                assert!(cycle.contains(&"b".to_string()));
            }
            other => panic!("expected a deadlock, got {other:?}"),
        }
    }

    #[test]
    fn chain_without_a_cycle_only_waits() {
        let mut g = WaitForGraph::new();
        assert_eq!(g.request("a", "r1"), RequestOutcome::Granted);
        assert_eq!(g.request("b", "r2"), RequestOutcome::Granted);
        assert_eq!(g.request("c", "r3"), RequestOutcome::Granted);
        // A waits on B, B waits on C, C waits on nobody: a chain, not a cycle.
        assert!(matches!(g.request("a", "r2"), RequestOutcome::Waiting { .. }));
        assert!(matches!(
            g.request("b", "r3"),
            RequestOutcome::Waiting { .. }
        ));
        // C asks for a free resource: granted, no deadlock anywhere.
        assert_eq!(g.request("c", "r4"), RequestOutcome::Granted);
    }

    #[test]
    fn resolve_aborts_youngest_and_frees_its_holds() {
        let mut g = WaitForGraph::new();
        assert_eq!(g.request("a", "file:doc"), RequestOutcome::Granted);
        assert_eq!(g.request("b", "cap:calendar"), RequestOutcome::Granted);
        assert!(matches!(
            g.request("a", "cap:calendar"),
            RequestOutcome::Waiting { .. }
        ));
        let cycle = match g.request("b", "file:doc") {
            RequestOutcome::Deadlock { cycle } => cycle,
            other => panic!("expected a deadlock, got {other:?}"),
        };
        // B waited most recently, so B is the youngest waiter and is aborted.
        let resolution = g.resolve(&cycle);
        assert_eq!(resolution.aborted, "b");
        assert_eq!(resolution.released, vec!["cap:calendar".to_string()]);
        assert_eq!(resolution.failure_kind(), FailureKind::Deadlock);
        assert_eq!(g.holder_of("cap:calendar"), None);
        // A can now proceed: the calendar is free and A still holds the doc.
        assert_eq!(g.request("a", "cap:calendar"), RequestOutcome::Granted);
        assert_eq!(g.holder_of("file:doc"), Some("a"));
    }

    #[test]
    fn ordered_allocator_refuses_to_wait_downhill() {
        let mut alloc = OrderedAllocator::new();
        // B takes the higher-ranked resource first, then asks for the
        // lower-ranked one: refused — B must release and re-acquire in order.
        assert!(matches!(
            alloc.request("b", "cap:calendar"),
            OrderedOutcome::Request(RequestOutcome::Granted)
        ));
        match alloc.request("b", "cap:calendar2") {
            // "cap:calendar2" ranks above "cap:calendar": fine, still uphill.
            OrderedOutcome::Request(RequestOutcome::Granted) => {}
            other => panic!("uphill request must pass, got {other:?}"),
        }
        match alloc.request("b", "cap:agenda") {
            OrderedOutcome::OrderViolation { held_higher } => {
                assert_eq!(held_higher, "cap:calendar2");
            }
            other => panic!("downhill request must be refused, got {other:?}"),
        }
        // The refusal touched nothing: B still holds both resources.
        assert_eq!(
            alloc.graph().held_by("b"),
            vec!["cap:calendar".to_string(), "cap:calendar2".to_string()]
        );
    }

    #[test]
    fn global_order_makes_the_classic_cycle_impossible() {
        let mut alloc = OrderedAllocator::new();
        // Both agents acquire in global order (ascending resource id).
        // A takes cap:calendar, B takes file:doc; A then waits for
        // file:doc (uphill, allowed). B asking for cap:calendar while
        // holding file:doc is downhill — refused, so no cycle can form.
        assert!(matches!(
            alloc.request("a", "cap:calendar"),
            OrderedOutcome::Request(RequestOutcome::Granted)
        ));
        assert!(matches!(
            alloc.request("b", "file:doc"),
            OrderedOutcome::Request(RequestOutcome::Granted)
        ));
        assert!(matches!(
            alloc.request("a", "file:doc"),
            OrderedOutcome::Request(RequestOutcome::Waiting { .. })
        ));
        assert!(matches!(
            alloc.request("b", "cap:calendar"),
            OrderedOutcome::OrderViolation { .. }
        ));
        // And the graph confirms: no cycle exists.
        assert_eq!(alloc.graph().holder_of("cap:calendar"), Some("a"));
        assert_eq!(alloc.graph().holder_of("file:doc"), Some("b"));
    }

    #[test]
    fn deadlock_taxonomy_entry() {
        assert_eq!(
            FailureKind::Deadlock.handler(),
            "abort the youngest waiter in the cycle, release its holds, escalate to a human"
        );
        assert_eq!(FailureKind::Deadlock.cpu_analogy(), "deadlock");
    }
}
