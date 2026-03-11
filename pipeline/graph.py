"""
Core pipeline graph engine.

Every model is a node. Every data flow dependency is an edge.
The graph executor traverses edges in topological order, skipping nodes
whose preconditions are unmet (gate nodes, conditional stages, etc.).
"""
from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set


def _sanitize_edge_token(raw: str) -> str:
    text = str(raw or "").replace(" ", "_").replace(":", "_").replace(".", "_")
    text = "".join(ch for ch in text if ch.isalnum() or ch in {"_", "-"})
    return text.strip("_-") or "flow"


def _default_edge_id(
    source_id: str,
    target_id: str,
    *,
    label: str = "",
    condition_id: str = "",
    target_function: str = "",
) -> str:
    tail = _sanitize_edge_token(target_function or condition_id or label or "flow")
    return f"{source_id}__to__{target_id}__{tail}"


@dataclass(frozen=True)
class RuntimeNodeShape:
    """Execution-layer identity for a node."""

    object_type: str = "object"
    faculty: str = "other"
    archetype: str = ""
    layer: str = "execution"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_type": str(self.object_type),
            "faculty": str(self.faculty),
            "archetype": str(self.archetype),
            "layer": str(self.layer),
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class EdgeReactionSpec:
    """Execution-layer annotation for an edge reaction."""

    reaction_name: str = ""
    target_function: str = ""
    defaults: Dict[str, Any] = field(default_factory=dict)
    layer: str = "execution"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def merged(self, incoming: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        merged = dict(self.defaults or {})
        if incoming:
            merged.update(dict(incoming))
        return merged

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reaction_name": str(self.reaction_name),
            "target_function": str(self.target_function),
            "defaults": dict(self.defaults or {}),
            "layer": str(self.layer),
            "metadata": dict(self.metadata or {}),
        }


# ---------------------------------------------------------------------------
# Base node
# ---------------------------------------------------------------------------

class PipelineNode(ABC):
    """Base class for every step in the training pipeline.

    Subclasses implement :meth:`execute` and optionally override
    :meth:`should_run` to express gate preconditions.  Each node reads from
    and writes back to the shared :class:`~pipeline.context.PipelineContext`.
    """

    @property
    @abstractmethod
    def node_id(self) -> str:
        """Unique string identifier (e.g. ``"stage_0_pregestation"``)."""
        ...

    @property
    def description(self) -> str:
        return f"Node[{self.node_id}]"

    @property
    def runtime_object_type(self) -> str:
        return "object"

    @property
    def runtime_faculty(self) -> str:
        return "other"

    @property
    def runtime_archetype(self) -> str:
        return type(self).__name__

    @property
    def runtime_shape(self) -> RuntimeNodeShape:
        return RuntimeNodeShape(
            object_type=self.runtime_object_type,
            faculty=self.runtime_faculty,
            archetype=self.runtime_archetype,
        )

    def should_run(self, ctx: "PipelineContext") -> bool:  # noqa: F821
        """Return True if this node should execute this round.

        The default always returns True; override to add gate logic.
        """
        return True

    @abstractmethod
    def execute(self, ctx: "PipelineContext") -> None:  # noqa: F821
        """Run this node's logic.  Reads from *ctx*, writes results back."""
        ...

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} id={self.node_id!r}>"


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------

@dataclass
class PipelineEdge:
    """Directed connection between two nodes.

    An edge carries an optional *condition* callable that inspects the context
    at execution time.  If the condition returns False the edge is inactive and
    the target node is skipped (unless another active edge also feeds it).

    ``on_traverse`` is an optional callback fired by the executor for every
    *active* incoming edge immediately before the target node executes.
    This is the mechanism by which a source node (e.g. DataNode) prepares
    exactly the data the target needs — driven by the edge list, not by the
    target node or by branching logic inside the source node's execute().

    ``reaction`` records the execution-layer function and mergeable default
    options for the edge.  The executor merges any runtime overrides from the
    context before invoking ``on_traverse``.

    ``label`` is purely informational and shows up in debug output.
    """

    source_id: str
    target_id: str
    condition: Optional[Callable[["PipelineContext"], bool]] = None  # noqa: F821
    label: str = ""
    condition_id: str = ""
    on_traverse: Optional[Callable[..., None]] = None  # noqa: F821
    edge_id: str = ""
    layer: str = "execution"
    reaction: EdgeReactionSpec = field(default_factory=EdgeReactionSpec)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_active(self, ctx: "PipelineContext") -> bool:  # noqa: F821
        if self.condition is None:
            return True
        return bool(self.condition(ctx))

    def merged_reaction_config(self, incoming: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.reaction.merged(incoming)

    def invoke(self, ctx: "PipelineContext", incoming: Optional[Dict[str, Any]] = None) -> None:  # noqa: F821
        if self.on_traverse is None:
            return
        _invoke_edge_callback(
            self.on_traverse,
            ctx,
            self.merged_reaction_config(incoming),
        )


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

class PipelineGraph:
    """Directed (acyclic) graph of :class:`PipelineNode` objects.

    Nodes are added with :meth:`add_node`; edges with :meth:`add_edge` (or the
    fluent ``>>`` helpers provided by :class:`~pipeline.nodes.base.NodeRef`).
    :meth:`build_sequence` returns a topologically sorted execution order;
    :meth:`execute_sequence` runs it against a context.

    The outer training loop (cycles × rounds) lives in the orchestrator and
    calls :meth:`execute_sequence` once per round.
    """

    def __init__(self, name: str = "pipeline"):
        self.name = name
        self._nodes: Dict[str, PipelineNode] = {}
        self._edges: List[PipelineEdge] = []

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def add_node(self, node: PipelineNode) -> "PipelineGraph":
        """Register a node.  Returns *self* for method chaining."""
        if node.node_id in self._nodes:
            raise ValueError(
                f"Duplicate node id {node.node_id!r} in graph {self.name!r}"
            )
        self._nodes[node.node_id] = node
        return self

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        *,
        condition: Optional[Callable[["PipelineContext"], bool]] = None,  # noqa: F821
        label: str = "",
        condition_id: str = "",
        on_traverse: Optional[Callable[..., None]] = None,  # noqa: F821
        edge_id: str = "",
        layer: str = "execution",
        target_function: str = "",
        reaction_name: str = "",
        reaction_defaults: Optional[Dict[str, Any]] = None,
        reaction_metadata: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "PipelineGraph":
        """Add a directed edge from *source_id* → *target_id*.

        ``condition`` is an optional callable ``(ctx) -> bool``; when it
        returns False the edge is inactive for that execution round and the
        target will be skipped (unless another active edge reaches it).

        ``on_traverse`` is an optional callback fired by the executor for each
        active incoming edge just before the target node runs.  Use it to
        deliver exactly the data the target node needs (e.g. DataNode loader
        builders assigned here drive data preparation by the edge list alone).

        ``reaction_defaults`` stores the default config blob for the edge's
        execution-layer reaction.  At runtime it merges with any overrides from
        ``ctx.edge_reaction_overrides`` keyed by ``edge_id``.
        """
        for nid in (source_id, target_id):
            if nid not in self._nodes:
                raise ValueError(
                    f"Unknown node {nid!r} referenced in edge "
                    f"({source_id!r} → {target_id!r}).  Add the node first."
                )

        resolved_target_function = str(
            target_function
            or getattr(getattr(on_traverse, "__func__", on_traverse), "__name__", "")
            or ""
        )
        resolved_reaction_name = str(reaction_name or label or resolved_target_function or "flow")
        resolved_edge_id = str(
            edge_id
            or _default_edge_id(
                source_id,
                target_id,
                label=label,
                condition_id=condition_id,
                target_function=resolved_target_function,
            )
        )
        resolved_layer = str(layer or "execution")

        self._edges.append(
            PipelineEdge(
                source_id=source_id,
                target_id=target_id,
                condition=condition,
                label=label,
                condition_id=str(condition_id or ""),
                on_traverse=on_traverse,
                edge_id=resolved_edge_id,
                layer=resolved_layer,
                reaction=EdgeReactionSpec(
                    reaction_name=resolved_reaction_name,
                    target_function=resolved_target_function,
                    defaults=dict(reaction_defaults or {}),
                    layer=resolved_layer,
                    metadata=dict(reaction_metadata or {}),
                ),
                metadata=dict(metadata or {}),
            )
        )
        return self

    @property
    def nodes(self) -> Dict[str, PipelineNode]:
        return dict(self._nodes)

    @property
    def edges(self) -> List[PipelineEdge]:
        return list(self._edges)

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------

    def build_sequence(self, entry_ids: Optional[List[str]] = None) -> List[str]:
        """Return a stable topological ordering of all reachable nodes.

        If *entry_ids* is given, only nodes reachable from those roots are
        included.  Otherwise every node with no incoming edges is treated as
        a root.
        """
        successors: Dict[str, List[str]] = {nid: [] for nid in self._nodes}
        global_in_degree: Dict[str, int] = {nid: 0 for nid in self._nodes}

        for edge in self._edges:
            successors[edge.source_id].append(edge.target_id)
            global_in_degree[edge.target_id] += 1

        if entry_ids is not None:
            roots = [nid for nid in entry_ids if nid in self._nodes]
        else:
            roots = [nid for nid, deg in global_in_degree.items() if deg == 0]

        reachable: Set[str] = set()
        queue: deque[str] = deque(roots)
        while queue:
            nid = queue.popleft()
            if nid in reachable:
                continue
            reachable.add(nid)
            for succ in successors.get(nid, []):
                if succ not in reachable:
                    queue.append(succ)

        if not reachable:
            return []

        in_degree: Dict[str, int] = {nid: 0 for nid in reachable}
        for edge in self._edges:
            if edge.source_id in reachable and edge.target_id in reachable:
                in_degree[edge.target_id] += 1

        ready = deque([nid for nid in roots if nid in reachable and in_degree[nid] == 0])
        if not ready:
            ready = deque(sorted(nid for nid, deg in in_degree.items() if deg == 0))

        order: List[str] = []
        while ready:
            nid = ready.popleft()
            order.append(nid)
            for succ in successors.get(nid, []):
                if succ not in in_degree:
                    continue
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    ready.append(succ)

        if len(order) != len(reachable):
            remaining = sorted(nid for nid in reachable if nid not in order)
            raise ValueError(
                f"Graph {self.name!r} has a cycle or unreachable dependency among: {remaining}"
            )

        return order

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute_sequence(
        self,
        ctx: "PipelineContext",  # noqa: F821
        sequence: Optional[List[str]] = None,
        *,
        verbose: bool = True,
    ) -> Dict[str, str]:
        """Execute nodes in *sequence* order, honouring edges and skip logic.

        Returns a status dict: ``{node_id: "ran" | "skipped:<reason>" | "failed:<msg>"}``.

        Nodes are skipped when:
        * All incoming edges are inactive (condition returned False), **or**
        * The node's :meth:`~PipelineNode.should_run` returns False.

        Nodes with *no* incoming edges are always candidates to run (their
        ``should_run`` is still consulted).
        """
        if sequence is None:
            sequence = self.build_sequence()

        statuses: Dict[str, str] = {}
        raw_edge_overrides = getattr(ctx, "edge_reaction_overrides", {})
        edge_overrides = raw_edge_overrides if isinstance(raw_edge_overrides, dict) else {}

        for node_id in sequence:
            node = self._nodes.get(node_id)
            if node is None:
                statuses[node_id] = "missing"
                continue

            # -- edge gate -------------------------------------------------
            incoming = [e for e in self._edges if e.target_id == node_id]
            active_incoming = [e for e in incoming if e.is_active(ctx)]
            if incoming and not active_incoming:
                statuses[node_id] = "skipped:edge"
                if verbose:
                    _log(f"[graph] SKIP {node_id!r} (all incoming edges inactive)")
                continue

            # -- node gate -------------------------------------------------
            if not node.should_run(ctx):
                statuses[node_id] = "skipped:node"
                if verbose:
                    _log(f"[graph] SKIP {node_id!r} (should_run=False)")
                continue

            # -- on_traverse callbacks (data provision by edge list) -------
            for edge in active_incoming:
                incoming_cfg = edge_overrides.get(edge.edge_id)
                if incoming_cfg is None:
                    incoming_cfg = edge_overrides.get(f"{edge.source_id}->{edge.target_id}")
                edge.invoke(ctx, incoming_cfg if isinstance(incoming_cfg, dict) else None)

            # -- execute ---------------------------------------------------
            if verbose:
                _log(f"[graph] RUN  {node_id!r}  ({node.description})")
            try:
                node.execute(ctx)
                statuses[node_id] = "ran"
            except Exception as exc:
                statuses[node_id] = f"failed:{exc}"
                if getattr(ctx, "raise_on_node_failure", True):
                    raise
                _log(f"[graph] FAIL {node_id!r}: {exc}")

        return statuses

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = [f"PipelineGraph({self.name!r}) - {len(self._nodes)} nodes, {len(self._edges)} edges"]
        seq = self.build_sequence()
        for nid in seq:
            node = self._nodes[nid]
            incoming = [e.source_id for e in self._edges if e.target_id == nid]
            cond_tags = [
                e.condition_id or e.label or "cond"
                for e in self._edges
                if e.target_id == nid and e.condition
            ]
            prefix = f"  <-[{', '.join(incoming)}]" if incoming else "  (root)"
            cond_str = f"  if({', '.join(cond_tags)})" if cond_tags else ""
            lines.append(f"  {nid:<42}{prefix}{cond_str}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(msg, flush=True)


def _invoke_edge_callback(
    callback: Callable[..., None],
    ctx: "PipelineContext",  # noqa: F821
    reaction_config: Dict[str, Any],
) -> None:
    try:
        sig = inspect.signature(callback)
    except (TypeError, ValueError):
        callback(ctx)
        return

    params = list(sig.parameters.values())
    positional = [
        p for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    keyword_only = [
        p.name for p in params
        if p.kind == inspect.Parameter.KEYWORD_ONLY
    ]
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
    config = dict(reaction_config or {})

    if accepts_kwargs:
        callback(ctx, **config)
        return
    if len(positional) >= 2:
        callback(ctx, config)
        return
    if keyword_only:
        kwargs = {name: config[name] for name in keyword_only if name in config}
        callback(ctx, **kwargs)
        return
    callback(ctx)
