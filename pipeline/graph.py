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

    @property
    def runtime_execution_policy(self) -> tuple:
        """Return ``(policy_name, config_dict)`` describing skip semantics.

        The IR records this so Mermaid diagrams and plan consumers can see
        *why* a node may be skipped without inspecting the Python source.

        Built-in policies:

        * ``"always"``   — runs every round (default).
        * ``"once"``     — runs once per pipeline lifetime (OneTimeNode).
        * ``"gated"``    — runs only after named gates pass (GatedNode).
        * ``"periodic"`` — runs every *N*-th round (e.g. CheckpointSaveNode).
        """
        return ("always", {})

    def should_run(self, ctx: "PipelineContext") -> bool:  # noqa: F821
        """Return True if this node should execute this round.

        The default always returns True; override to add gate logic.
        """
        return True

    @abstractmethod
    def execute(self, ctx: "PipelineContext") -> None:  # noqa: F821
        """Run this node's logic.  Reads from *ctx*, writes results back."""
        ...

    def declare_subnodes(self) -> List[Dict[str, Any]]:
        """Declare internal sub-processes for IR introspection.

        Override to expose the inner faculties of this node — individual
        training cores, evaluation passes, data preparation steps, etc.
        Each entry is a dict with at least ``subnode_id`` and ``kind``.
        The plan exporter calls this when building the action registry.
        """
        return []

    def declare_training_mechanics(self) -> Dict[str, Any]:
        """Declare NN-training mechanics for IR-first visualization.

        Training nodes can override this to describe their conceptual inputs,
        outputs, losses, and optimization/data-flow edges without exposing
        Python control flow. The plan exporter persists this into node metadata
        so alternate graph views can render the system as a neural-design canvas.
        """
        return {}

    @property
    def gpu_models(self) -> List[str]:
        """Context attribute names of ``nn.Module`` objects this node accesses.

        Override in subclasses that use GPU models so the graph executor
        can pin them via the :class:`GPUResidenceManager` before calling
        :meth:`execute` and release them afterwards.

        Example::

            gpu_models = ["classifier"]           # uses ctx.classifier
            gpu_models = ["transformer", "classifier"]  # uses both
        """
        return []

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

    def build_sequence_from_program(self, execution_program: Optional[Dict[str, Any]]) -> List[str]:
        """Return node order from the persisted execution program when present."""
        if not isinstance(execution_program, dict) or not execution_program:
            return self.build_sequence()

        ordered: List[str] = []
        seen: Set[str] = set()
        for node_id in list(execution_program.get("sequence_node_ids", []) or []):
            resolved_id = str(node_id or "").strip()
            if not resolved_id or resolved_id in seen or resolved_id not in self._nodes:
                continue
            ordered.append(resolved_id)
            seen.add(resolved_id)
        if ordered:
            return ordered

        step_rows: List[tuple[float, str]] = []
        for step in list(execution_program.get("steps", []) or []):
            if str(step.get("kind", "node") or "node") != "node":
                continue
            node_id = str(step.get("node_id", "") or "").strip()
            if not node_id or node_id in seen or node_id not in self._nodes:
                continue
            ordinal = float(step.get("display_order", step.get("ordinal", 0)) or 0)
            step_rows.append((ordinal, node_id))
            seen.add(node_id)
        if step_rows:
            step_rows.sort(key=lambda item: (item[0], item[1]))
            return [node_id for _, node_id in step_rows]

        return self.build_sequence()

    def execute_program(
        self,
        ctx: "PipelineContext",  # noqa: F821
        execution_program: Optional[Dict[str, Any]],
        *,
        condition_resolver: Optional[Callable[[str, "PipelineContext"], bool]] = None,  # noqa: F821
        verbose: bool = True,
    ) -> Dict[str, str]:
        """Execute the graph using control-flow steps and transitions from *execution_program*."""
        if _program_has_control_flow(execution_program):
            return self._execute_program_flow(
                ctx,
                execution_program,
                condition_resolver=condition_resolver,
                verbose=verbose,
            )

        program_steps = _program_steps_by_node(execution_program)
        sequence = self.build_sequence_from_program(execution_program)
        return self._execute_node_sequence(
            ctx,
            sequence,
            verbose=verbose,
            program_steps=program_steps,
            condition_resolver=condition_resolver,
        )

    def _execute_program_flow(
        self,
        ctx: "PipelineContext",  # noqa: F821
        execution_program: Optional[Dict[str, Any]],
        *,
        condition_resolver: Optional[Callable[[str, "PipelineContext"], bool]] = None,  # noqa: F821
        verbose: bool,
    ) -> Dict[str, str]:
        program_model = _program_flow_model(execution_program)
        step_map = dict(program_model.get("step_map", {}) or {})
        transitions_by_from = dict(program_model.get("transitions_by_from", {}) or {})
        transitions_to_step = dict(program_model.get("transitions_to_step", {}) or {})
        entry_step_id = str(program_model.get("entry_step_id", "") or "")
        max_visits = int(program_model.get("max_visits", 0) or 0)

        statuses: Dict[str, str] = {}
        node_trace: List[Dict[str, Any]] = []
        program_trace: List[Dict[str, Any]] = []

        if not entry_step_id or entry_step_id not in step_map:
            setattr(ctx, "last_execution_trace", node_trace)
            setattr(ctx, "last_program_trace", program_trace)
            return statuses

        current_step_id = entry_step_id
        visit_count = 0
        while current_step_id:
            visit_count += 1
            if max_visits > 0 and visit_count > max_visits:
                limit_entry = {
                    "tick": int(len(program_trace) + 1),
                    "entry_kind": "halt",
                    "reason": "max_program_visits_exceeded",
                    "step_id": str(current_step_id),
                    "max_visits": int(max_visits),
                    "status": "halted:max_visits",
                }
                program_trace.append(limit_entry)
                setattr(ctx, "last_execution_trace", node_trace)
                setattr(ctx, "last_program_trace", program_trace)
                if getattr(ctx, "raise_on_node_failure", True):
                    raise RuntimeError(f"Program traversal exceeded max_visits={max_visits}")
                _log(f"[graph] HALT program traversal exceeded max_visits={max_visits}")
                break

            step = dict(step_map.get(current_step_id, {}) or {})
            if not step:
                program_trace.append(
                    {
                        "tick": int(len(program_trace) + 1),
                        "entry_kind": "step",
                        "step_id": str(current_step_id),
                        "kind": "missing",
                        "status": "missing",
                    }
                )
                break

            step_kind = str(step.get("kind", "node") or "node")
            step_label = str(step.get("label", current_step_id) or current_step_id)
            call_ref = str(step.get("call_ref", "") or "")
            frame_keys = list(step.get("frame_keys", []) or [])

            if step_kind == "decision":
                target_node_id = str(dict(step.get("config", {}) or {}).get("target_node_id", "") or "")
                guard_condition_ids = [
                    str(condition_id or "").strip()
                    for condition_id in list(step.get("guard_condition_ids", []) or [])
                    if str(condition_id or "").strip()
                ]
                incoming = [edge for edge in self._edges if target_node_id and edge.target_id == target_node_id]
                guard_results = {
                    condition_id: _evaluate_program_guard(
                        condition_id,
                        ctx,
                        incoming,
                        condition_resolver=condition_resolver,
                    )
                    for condition_id in guard_condition_ids
                }
                chosen_branch = "pass" if all(guard_results.values()) else "hold"
                decision_entry = {
                    "tick": int(len(program_trace) + 1),
                    "entry_kind": "step",
                    "step_id": str(current_step_id),
                    "kind": "decision",
                    "label": step_label,
                    "call_ref": call_ref,
                    "target_node_id": target_node_id,
                    "guard_condition_ids": list(guard_condition_ids),
                    "guard_results": dict(guard_results),
                    "frame_keys": list(frame_keys),
                    "status": chosen_branch,
                }
                program_trace.append(decision_entry)
                transition = _select_program_transition(
                    list(transitions_by_from.get(current_step_id, []) or []),
                    preferred_branch=chosen_branch,
                )
                if transition is None:
                    break
                program_trace.append(_program_transition_trace(transition, tick=len(program_trace) + 1))
                current_step_id = str(transition.get("to_step_id", "") or "")
                continue

            if step_kind == "hold":
                # Consult CycleGate objects from ctx — the IR cycle edges
                # caused these objects to exist; they own the iteration
                # decision, not hardcoded counters.
                _cycle_gates = list(getattr(ctx, "cycle_gates", None) or [])
                _repeat_gate = None
                for _gate in _cycle_gates:
                    branch = _gate.evaluate(ctx)
                    if branch == "repeat":
                        _repeat_gate = _gate
                        break
                if _repeat_gate is not None:
                    program_trace.append(
                        {
                            "tick": int(len(program_trace) + 1),
                            "entry_kind": "step",
                            "step_id": str(current_step_id),
                            "kind": "cycle_gate",
                            "label": step_label,
                            "call_ref": "CycleGate.evaluate",
                            "edge_id": str(_repeat_gate.edge_id),
                            "iteration": int(_repeat_gate.iteration),
                            "max_iterations": int(_repeat_gate.max_iterations),
                            "frame_keys": list(frame_keys),
                            "status": "repeat",
                        }
                    )
                    # Jump back to the program entry — the CycleGate decided.
                    current_step_id = entry_step_id
                    continue
                # All gates exhausted (or none exist) — halt.
                program_trace.append(
                    {
                        "tick": int(len(program_trace) + 1),
                        "entry_kind": "step",
                        "step_id": str(current_step_id),
                        "kind": "hold",
                        "label": step_label,
                        "call_ref": call_ref,
                        "frame_keys": list(frame_keys),
                        "status": "hold",
                    }
                )
                break

            if step_kind == "node":
                node_id = str(step.get("node_id", "") or "")
                step_spec = dict(step)
                if _program_step_has_decision_predecessor(current_step_id, transitions_to_step, step_map):
                    step_spec["guard_condition_ids"] = []
                try:
                    single_statuses = self._execute_node_sequence(
                        ctx,
                        [node_id],
                        verbose=verbose,
                        program_steps={node_id: step_spec},
                        condition_resolver=condition_resolver,
                    )
                except Exception:
                    node_entries = list(getattr(ctx, "last_execution_trace", []) or [])
                    node_trace.extend(dict(entry) for entry in node_entries)
                    if node_entries:
                        node_entry = dict(node_entries[-1])
                        program_trace.append(
                            {
                                "tick": int(len(program_trace) + 1),
                                "entry_kind": "step",
                                "step_id": str(current_step_id),
                                "kind": "node",
                                "node_id": str(node_entry.get("node_id", node_id) or node_id),
                                "label": step_label,
                                "call_ref": str(node_entry.get("call_ref", call_ref) or call_ref),
                                "frame_keys": list(node_entry.get("frame_keys", []) or []),
                                "guard_condition_ids": list(node_entry.get("guard_condition_ids", []) or []),
                                "guard_results": dict(node_entry.get("guard_results", {}) or {}),
                                "status": str(node_entry.get("status", "failed") or "failed"),
                            }
                        )
                    setattr(ctx, "last_execution_trace", node_trace)
                    setattr(ctx, "last_program_trace", program_trace)
                    raise

                node_entries = list(getattr(ctx, "last_execution_trace", []) or [])
                node_trace.extend(dict(entry) for entry in node_entries)
                node_entry = dict(node_entries[-1]) if node_entries else {
                    "node_id": node_id,
                    "status": str(single_statuses.get(node_id, "missing") or "missing"),
                    "frame_keys": list(frame_keys),
                    "guard_condition_ids": list(step_spec.get("guard_condition_ids", []) or []),
                    "guard_results": {},
                    "call_ref": call_ref,
                }
                statuses[node_id] = str(single_statuses.get(node_id, node_entry.get("status", "missing")) or "missing")
                program_trace.append(
                    {
                        "tick": int(len(program_trace) + 1),
                        "entry_kind": "step",
                        "step_id": str(current_step_id),
                        "kind": "node",
                        "node_id": str(node_entry.get("node_id", node_id) or node_id),
                        "label": step_label,
                        "call_ref": str(node_entry.get("call_ref", call_ref) or call_ref),
                        "frame_keys": list(node_entry.get("frame_keys", []) or []),
                        "guard_condition_ids": list(node_entry.get("guard_condition_ids", []) or []),
                        "guard_results": dict(node_entry.get("guard_results", {}) or {}),
                        "status": str(node_entry.get("status", statuses[node_id]) or statuses[node_id]),
                    }
                )
                transition = _select_program_transition(list(transitions_by_from.get(current_step_id, []) or []))
                if transition is None:
                    break
                program_trace.append(_program_transition_trace(transition, tick=len(program_trace) + 1))
                current_step_id = str(transition.get("to_step_id", "") or "")
                continue

            program_trace.append(
                {
                    "tick": int(len(program_trace) + 1),
                    "entry_kind": "step",
                    "step_id": str(current_step_id),
                    "kind": step_kind,
                    "label": step_label,
                    "call_ref": call_ref,
                    "frame_keys": list(frame_keys),
                    "status": "visited",
                }
            )
            transition = _select_program_transition(list(transitions_by_from.get(current_step_id, []) or []))
            if transition is None:
                break
            program_trace.append(_program_transition_trace(transition, tick=len(program_trace) + 1))
            current_step_id = str(transition.get("to_step_id", "") or "")

        setattr(ctx, "last_execution_trace", node_trace)
        setattr(ctx, "last_program_trace", program_trace)
        return statuses

    def execute_sequence(
        self,
        ctx: "PipelineContext",  # noqa: F821
        sequence: Optional[List[str]] = None,
        *,
        verbose: bool = True,
    ) -> Dict[str, str]:
        """Execute nodes in *sequence* order, honouring edges and skip logic."""
        if sequence is None:
            sequence = self.build_sequence()
        return self._execute_node_sequence(ctx, list(sequence), verbose=verbose)

    def _execute_node_sequence(
        self,
        ctx: "PipelineContext",  # noqa: F821
        sequence: List[str],
        *,
        verbose: bool,
        program_steps: Optional[Dict[str, Dict[str, Any]]] = None,
        condition_resolver: Optional[Callable[[str, "PipelineContext"], bool]] = None,  # noqa: F821
    ) -> Dict[str, str]:
        statuses: Dict[str, str] = {}
        trace: List[Dict[str, Any]] = []
        raw_edge_overrides = getattr(ctx, "edge_reaction_overrides", {})
        edge_overrides = raw_edge_overrides if isinstance(raw_edge_overrides, dict) else {}
        step_specs = dict(program_steps or {})

        for step_index, node_id in enumerate(sequence, start=1):
            node = self._nodes.get(node_id)
            step_spec = dict(step_specs.get(node_id, {}) or {})
            program_step_id = str(step_spec.get("step_id", "") or "")
            execution_mode = "program" if step_spec else "topology"
            program_frame_keys = list(step_spec.get("frame_keys", []) or [])
            call_ref = str(step_spec.get("call_ref", f"{type(node).__name__}.execute" if node is not None else "node.execute") or "node.execute")
            if node is None:
                statuses[node_id] = "missing"
                trace.append(
                    {
                        "step": int(step_index),
                        "node_id": str(node_id),
                        "status": "missing",
                        "incoming_edge_ids": [],
                        "active_edge_ids": [],
                        "guard_condition_ids": [],
                        "frame_keys": list(program_frame_keys),
                        "program_step_id": program_step_id,
                        "execution_mode": execution_mode,
                    }
                )
                continue

            incoming = [e for e in self._edges if e.target_id == node_id]
            incoming_edge_ids = [str(e.edge_id) for e in incoming]
            inferred_guard_ids = sorted({str(e.condition_id) for e in incoming if str(e.condition_id or '').strip()})
            configured_guard_ids = [
                str(condition_id or '').strip()
                for condition_id in list(step_spec.get("guard_condition_ids", []) or [])
                if str(condition_id or '').strip()
            ]
            guard_condition_ids = configured_guard_ids or inferred_guard_ids
            guard_results: Dict[str, bool] = {}

            if configured_guard_ids:
                for condition_id in guard_condition_ids:
                    guard_results[condition_id] = _evaluate_program_guard(
                        condition_id,
                        ctx,
                        incoming,
                        condition_resolver=condition_resolver,
                    )
                if not all(guard_results.values()):
                    frame_keys = _execution_frame_keys(
                        ctx,
                        node_id,
                        [],
                        {},
                        program_frame_keys=program_frame_keys,
                    )
                    statuses[node_id] = "skipped:guard"
                    trace.append(
                        {
                            "step": int(step_index),
                            "node_id": str(node_id),
                            "status": "skipped:guard",
                            "incoming_edge_ids": incoming_edge_ids,
                            "active_edge_ids": [],
                            "guard_condition_ids": guard_condition_ids,
                            "guard_results": dict(guard_results),
                            "frame_keys": list(frame_keys),
                            "call_ref": call_ref,
                            "program_step_id": program_step_id,
                            "execution_mode": execution_mode,
                            "gates": _gate_status_snapshot(ctx),
                        }
                    )
                    if verbose:
                        guard_summary = ", ".join(
                            f"{condition_id}={'1' if passed else '0'}"
                            for condition_id, passed in guard_results.items()
                        )
                        _log(f"[graph] SKIP {step_index:02d} {node_id!r} (program guard blocked: {guard_summary})")
                    continue

            if configured_guard_ids:
                active_incoming = []
                for edge in incoming:
                    edge_condition_id = str(edge.condition_id or "").strip()
                    if edge_condition_id:
                        if guard_results.get(edge_condition_id, False):
                            active_incoming.append(edge)
                        continue
                    if edge.is_active(ctx):
                        active_incoming.append(edge)
            else:
                active_incoming = [e for e in incoming if e.is_active(ctx)]
                for condition_id in guard_condition_ids:
                    guard_results[condition_id] = any(
                        str(edge.condition_id or "").strip() == condition_id and edge in active_incoming
                        for edge in incoming
                    )

            active_edge_ids = [str(e.edge_id) for e in active_incoming]
            if incoming and not active_incoming:
                statuses[node_id] = "skipped:edge"
                trace.append(
                    {
                        "step": int(step_index),
                        "node_id": str(node_id),
                        "status": "skipped:edge",
                        "incoming_edge_ids": incoming_edge_ids,
                        "active_edge_ids": [],
                        "guard_condition_ids": guard_condition_ids,
                        "guard_results": dict(guard_results),
                        "frame_keys": [],
                        "call_ref": call_ref,
                        "program_step_id": program_step_id,
                        "execution_mode": execution_mode,
                        "gates": _gate_status_snapshot(ctx),
                    }
                )
                if verbose:
                    _log(f"[graph] SKIP {step_index:02d} {node_id!r} (all incoming edges inactive)")
                continue

            active_edge_configs: Dict[str, Dict[str, Any]] = {}
            for edge in active_incoming:
                incoming_cfg = edge_overrides.get(edge.edge_id)
                if incoming_cfg is None:
                    incoming_cfg = edge_overrides.get(f"{edge.source_id}->{edge.target_id}")
                override_cfg = incoming_cfg if isinstance(incoming_cfg, dict) else None
                active_edge_configs[edge.edge_id] = edge.merged_reaction_config(override_cfg)

            for edge in active_incoming:
                edge.invoke(ctx, active_edge_configs.get(edge.edge_id))

            frame_keys = _execution_frame_keys(
                ctx,
                node_id,
                active_incoming,
                active_edge_configs,
                program_frame_keys=program_frame_keys,
            )
            frame_suffix = f" frame={_format_frame_keys(frame_keys)}" if frame_keys else ""

            if not node.should_run(ctx):
                statuses[node_id] = "skipped:node"
                trace.append(
                    {
                        "step": int(step_index),
                        "node_id": str(node_id),
                        "status": "skipped:node",
                        "incoming_edge_ids": incoming_edge_ids,
                        "active_edge_ids": active_edge_ids,
                        "guard_condition_ids": guard_condition_ids,
                        "guard_results": dict(guard_results),
                        "frame_keys": list(frame_keys),
                        "reaction_names": [str(e.reaction.reaction_name or '') for e in active_incoming],
                        "call_ref": call_ref,
                        "program_step_id": program_step_id,
                        "execution_mode": execution_mode,
                        "gates": _gate_status_snapshot(ctx),
                    }
                )
                if verbose:
                    _log(f"[graph] SKIP {step_index:02d} {node_id!r} (should_run=False){frame_suffix}")
                continue

            if verbose:
                _log(f"[graph] RUN  {step_index:02d} {node_id!r}  ({node.description}){frame_suffix}")
            try:
                _execute_with_residence(node, ctx)
                statuses[node_id] = "ran"
                trace.append(
                    {
                        "step": int(step_index),
                        "node_id": str(node_id),
                        "status": "ran",
                        "incoming_edge_ids": incoming_edge_ids,
                        "active_edge_ids": active_edge_ids,
                        "guard_condition_ids": guard_condition_ids,
                        "guard_results": dict(guard_results),
                        "frame_keys": list(frame_keys),
                        "reaction_names": [str(e.reaction.reaction_name or '') for e in active_incoming],
                        "call_ref": call_ref,
                        "program_step_id": program_step_id,
                        "execution_mode": execution_mode,
                        "gates": _gate_status_snapshot(ctx),
                    }
                )
            except Exception as exc:
                statuses[node_id] = f"failed:{exc}"
                trace.append(
                    {
                        "step": int(step_index),
                        "node_id": str(node_id),
                        "status": f"failed:{exc}",
                        "incoming_edge_ids": incoming_edge_ids,
                        "active_edge_ids": active_edge_ids,
                        "guard_condition_ids": guard_condition_ids,
                        "guard_results": dict(guard_results),
                        "frame_keys": list(frame_keys),
                        "reaction_names": [str(e.reaction.reaction_name or '') for e in active_incoming],
                        "call_ref": call_ref,
                        "program_step_id": program_step_id,
                        "execution_mode": execution_mode,
                        "gates": _gate_status_snapshot(ctx),
                    }
                )
                if getattr(ctx, "raise_on_node_failure", True):
                    setattr(ctx, "last_execution_trace", trace)
                    raise
                _log(f"[graph] FAIL {step_index:02d} {node_id!r}: {exc}")

        setattr(ctx, "last_execution_trace", trace)
        setattr(ctx, "last_program_trace", [])
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


def _execute_with_residence(node: PipelineNode, ctx: "PipelineContext") -> None:
    """Execute *node* with GPU residence management when available.

    If ``ctx.gpu_residence`` is active and the node declares :attr:`gpu_models`,
    the listed models are pinned on ``ctx.device`` for the duration of
    ``node.execute(ctx)`` and released (eviction-eligible) afterwards.
    """
    mgr = getattr(ctx, "gpu_residence", None)
    model_names = node.gpu_models
    if mgr is None or not getattr(mgr, "enabled", False) or not model_names:
        node.execute(ctx)
        return

    device = ctx.device or __import__("torch").device("cpu")
    models = []
    for attr_name in model_names:
        module = getattr(ctx, attr_name, None)
        if module is not None and hasattr(module, "parameters"):
            models.append((module, attr_name))

    if not models:
        node.execute(ctx)
        return

    mgr.require_many(models, device)
    try:
        node.execute(ctx)
    finally:
        mgr.release_many([name for _, name in models])


def _gate_status_snapshot(ctx: "PipelineContext") -> Dict[str, bool]:  # noqa: F821
    gate_names = [
        "gate_pregestation",
        "gate_gestation",
        "gate_berkeley",
        "gate_transformer",
        "gate_generator",
        "gate_wave",
    ]
    snapshot: Dict[str, bool] = {}
    for gate_name in gate_names:
        gate = getattr(ctx, gate_name, None)
        snapshot[gate_name] = bool(getattr(gate, "passed", False))
    return snapshot


def _program_has_control_flow(execution_program: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(execution_program, dict):
        return False
    steps = list(execution_program.get("steps", []) or [])
    transitions = list(execution_program.get("transitions", []) or [])
    if not steps:
        return False
    return bool(transitions) or any(str(step.get("kind", "node") or "node") != "node" for step in steps)


# ---------------------------------------------------------------------------
# CycleGate — runtime object instantiated from IR cycle edges
# ---------------------------------------------------------------------------

class CycleGate:
    """Runtime object caused into existence by an IR cycle edge.

    The plan's ``layer="cycle"`` edges carry ``cycle_control`` dicts that
    fully describe the desired repetition.  When the orchestrator (or any
    execution driver) reads the plan, each such edge **causes** a CycleGate
    to be instantiated.  That object — not hardcoded loop counters — owns
    the iteration state and decides when to repeat or exhaust.
    """

    def __init__(
        self,
        edge_id: str,
        source_node_id: str,
        target_node_id: str,
        cycle_control: Dict[str, Any],
    ):
        self.edge_id = str(edge_id)
        self.source_node_id = str(source_node_id)
        self.target_node_id = str(target_node_id)
        self.max_iterations = int(cycle_control.get("max_iterations", 1) or 1)
        self.cycles = int(cycle_control.get("cycles", 1) or 1)
        self.rounds_per_cycle = int(cycle_control.get("rounds_per_cycle", 1) or 1)
        self.stride = int(cycle_control.get("stride", 0) or 0)
        self.iteration = 0

    # -- primary interface ------------------------------------------------

    def evaluate(self, ctx: Optional[Any] = None) -> str:
        """Advance iteration and return ``'repeat'`` or ``'exhaust'``.

        When *ctx* is supplied, ``ctx.cycle``, ``ctx.round_id`` and
        ``ctx.total_rounds_completed`` are updated so that the rest of
        the runtime sees the current position without any external counter.
        """
        self.iteration += 1
        if ctx is not None:
            rpc = self.rounds_per_cycle if self.rounds_per_cycle > 0 else 1
            ctx.cycle = (self.iteration - 1) // rpc + 1
            ctx.round_id = (self.iteration - 1) % rpc + 1
            ctx.total_rounds_completed = getattr(ctx, "total_rounds_completed", 0)
            if ctx.total_rounds_completed < self.iteration:
                ctx.total_rounds_completed = self.iteration
        if self.iteration < self.max_iterations:
            return "repeat"
        return "exhaust"

    @property
    def exhausted(self) -> bool:
        return self.iteration >= self.max_iterations

    # -- factory ----------------------------------------------------------

    @classmethod
    def from_plan(cls, plan: Any) -> List["CycleGate"]:
        """Instantiate CycleGate objects from every cycle edge in *plan*.

        This is the mechanism by which IR cycle edges *cause* runtime
        objects to exist.  No sniffing, no hardcoded wiring — the edge's
        ``cycle_control`` dict is the complete specification.
        """
        gates: List["CycleGate"] = []
        for edge in list(getattr(plan, "edges", []) or []):
            if str(getattr(edge, "layer", "") or "") != "cycle":
                continue
            if not bool(getattr(edge, "enabled", True)):
                continue
            cc = dict(getattr(edge, "cycle_control", {}) or {})
            if int(cc.get("max_iterations", 0) or 0) <= 0:
                continue
            gates.append(
                cls(
                    edge_id=str(getattr(edge, "edge_id", "") or ""),
                    source_node_id=str(getattr(edge, "source_node_id", "") or ""),
                    target_node_id=str(getattr(edge, "target_node_id", "") or ""),
                    cycle_control=cc,
                )
            )
        return gates

    def __repr__(self) -> str:
        return (
            f"CycleGate(edge_id={self.edge_id!r}, "
            f"iteration={self.iteration}/{self.max_iterations}, "
            f"cycles={self.cycles}, rounds_per_cycle={self.rounds_per_cycle})"
        )


def _program_ordered_steps(execution_program: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(execution_program, dict):
        return []
    steps = [dict(step) for step in list(execution_program.get("steps", []) or []) if isinstance(step, dict)]
    return sorted(steps, key=lambda step: (float(step.get("display_order", step.get("ordinal", 0)) or 0), str(step.get("step_id", ""))))


def _program_flow_model(execution_program: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    ordered_steps = _program_ordered_steps(execution_program)
    step_map = {
        str(step.get("step_id", "") or ""): dict(step)
        for step in ordered_steps
        if str(step.get("step_id", "") or "").strip()
    }
    transitions_by_from: Dict[str, List[Dict[str, Any]]] = {}
    transitions_to_step: Dict[str, List[Dict[str, Any]]] = {}
    for transition in list((execution_program or {}).get("transitions", []) or []):
        if not isinstance(transition, dict):
            continue
        from_step_id = str(transition.get("from_step_id", "") or "").strip()
        to_step_id = str(transition.get("to_step_id", "") or "").strip()
        if not from_step_id or not to_step_id:
            continue
        blob = dict(transition)
        transitions_by_from.setdefault(from_step_id, []).append(blob)
        transitions_to_step.setdefault(to_step_id, []).append(blob)
    for transition_list in transitions_by_from.values():
        transition_list.sort(key=lambda item: (float(item.get("ordinal", 0) or 0), str(item.get("transition_id", ""))))
    entry_step_id = str((execution_program or {}).get("entry_step_id", "") or "").strip()
    if not entry_step_id and ordered_steps:
        entry_step_id = str(ordered_steps[0].get("step_id", "") or "")
    max_visits = int((execution_program or {}).get("max_visits_per_round", 0) or 0)
    if max_visits <= 0:
        max_visits = max(32, len(ordered_steps) * 8, sum(len(items) for items in transitions_by_from.values()) * 4)
    return {
        "ordered_steps": ordered_steps,
        "step_map": step_map,
        "transitions_by_from": transitions_by_from,
        "transitions_to_step": transitions_to_step,
        "entry_step_id": entry_step_id,
        "max_visits": int(max_visits),
    }


def _select_program_transition(
    transitions: List[Dict[str, Any]],
    *,
    preferred_branch: str = "",
) -> Optional[Dict[str, Any]]:
    ordered = [dict(item) for item in list(transitions or [])]
    if not ordered:
        return None
    branch = str(preferred_branch or "").strip()
    if branch:
        for transition in ordered:
            if str(transition.get("branch", "") or "").strip() == branch:
                return transition
    if len(ordered) == 1:
        return ordered[0]
    for transition in ordered:
        if not str(transition.get("branch", "") or "").strip():
            return transition
    return ordered[0]


def _program_transition_trace(transition: Dict[str, Any], *, tick: int) -> Dict[str, Any]:
    return {
        "tick": int(tick),
        "entry_kind": "transition",
        "transition_id": str(transition.get("transition_id", "") or ""),
        "from_step_id": str(transition.get("from_step_id", "") or ""),
        "to_step_id": str(transition.get("to_step_id", "") or ""),
        "label": str(transition.get("label", "") or ""),
        "kind": str(transition.get("kind", "sequence") or "sequence"),
        "branch": str(transition.get("branch", "") or ""),
        "call_ref": str(transition.get("call_ref", "") or ""),
        "status": "traversed",
    }


def _program_step_has_decision_predecessor(
    step_id: str,
    transitions_to_step: Dict[str, List[Dict[str, Any]]],
    step_map: Dict[str, Dict[str, Any]],
) -> bool:
    for transition in list(transitions_to_step.get(str(step_id or ""), []) or []):
        source_step = dict(step_map.get(str(transition.get("from_step_id", "") or ""), {}) or {})
        if str(source_step.get("kind", "") or "") == "decision":
            return True
    return False


def _program_steps_by_node(execution_program: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    steps: Dict[str, Dict[str, Any]] = {}
    if not isinstance(execution_program, dict):
        return steps
    for step in list(execution_program.get("steps", []) or []):
        if str(step.get("kind", "node") or "node") != "node":
            continue
        node_id = str(step.get("node_id", "") or "").strip()
        if not node_id or node_id in steps:
            continue
        steps[node_id] = dict(step)
    return steps


def _evaluate_program_guard(
    condition_id: str,
    ctx: "PipelineContext",  # noqa: F821
    incoming: List[PipelineEdge],
    *,
    condition_resolver: Optional[Callable[[str, "PipelineContext"], bool]] = None,  # noqa: F821
) -> bool:
    resolved_id = str(condition_id or "").strip()
    if not resolved_id:
        return True
    if callable(condition_resolver):
        return bool(condition_resolver(resolved_id, ctx))
    matching = [edge for edge in incoming if str(edge.condition_id or "").strip() == resolved_id]
    if not matching:
        return False
    return any(edge.is_active(ctx) for edge in matching)


NODE_FRAME_HINT_TABLE: List[tuple] = [
    # (match_type, match_value, hints)
    # match_type: "exact" = node_id ==, "contains" = substring in node_id, "in_set" = node_id in set
    ("exact", "data_node", ["class_names", "semantic_term_to_idx", "symbol_pool", "label_embedding_bank"]),
    ("contains", "pregestation", ["classifier", "gate_classifier", "pregestation_loader", "pregestation_eval_loader", "gate_pregestation"]),
    ("exact", "stage_0_pregestation", ["classifier", "gate_classifier", "pregestation_loader", "pregestation_eval_loader", "gate_pregestation"]),
    ("contains", "gestation", ["classifier", "gate_classifier", "gestation_loader", "gestation_eval_loader", "gate_pregestation", "gate_gestation"]),
    ("exact", "stage_1_gestation", ["classifier", "gate_classifier", "gestation_loader", "gestation_eval_loader", "gate_pregestation", "gate_gestation"]),
    ("contains", "berkeley", ["classifier", "gate_classifier", "berkeley_refresh_loader", "payload_validation_loader", "payload_bank", "gate_gestation", "gate_berkeley"]),
    ("in_set", {"stage_c_lora", "stage_fake_feedback"}, ["classifier", "gate_classifier", "berkeley_refresh_loader", "payload_validation_loader", "payload_bank", "gate_gestation", "gate_berkeley"]),
    ("contains", "transformer", ["transformer", "classifier", "payload_bank", "gate_berkeley", "gate_transformer"]),
    ("exact", "config_search", ["transformer", "classifier", "payload_bank", "gate_berkeley", "gate_transformer"]),
    ("contains", "generator", ["generator", "discriminator", "payload_bank", "payload_conditions", "gate_transformer", "gate_generator"]),
    ("contains", "gan", ["generator", "discriminator", "payload_bank", "payload_conditions", "gate_transformer", "gate_generator"]),
    ("contains", "wave", ["wave_classifier", "transformer", "gate_transformer", "gate_wave"]),
    ("in_set", {"sync_gate_replica", "checkpoint_save", "viewer_ipc"}, ["classifier", "gate_classifier", "gate_berkeley", "gate_transformer", "gate_generator", "gate_wave"]),
]


def _resolve_frame_hints(node_id: str) -> List[str]:
    """Look up frame-hint resources for *node_id* from :data:`NODE_FRAME_HINT_TABLE`."""
    for match_type, match_value, hints in NODE_FRAME_HINT_TABLE:
        if match_type == "exact" and node_id == match_value:
            return list(hints)
        if match_type == "contains" and match_value in node_id:
            return list(hints)
        if match_type == "in_set" and node_id in match_value:
            return list(hints)
    return []


def _node_frame_hints(node_id: str) -> List[str]:
    return _resolve_frame_hints(node_id)


def _symbolic_attr_token(ctx: "PipelineContext", attr: str) -> str:  # noqa: F821
    value = getattr(ctx, attr, None)
    if attr.startswith("gate_"):
        if value is None:
            return ""
        return f"{attr}={'1' if bool(getattr(value, 'passed', False)) else '0'}"
    if value is None:
        return ""
    if isinstance(value, bool):
        return attr if value else ""
    if isinstance(value, str):
        return attr if value.strip() else ""
    if isinstance(value, (list, tuple, dict, set)):
        return attr if len(value) > 0 else ""
    return attr


def _execution_frame_keys(
    ctx: "PipelineContext",  # noqa: F821
    node_id: str,
    active_incoming: List[PipelineEdge],
    active_edge_configs: Dict[str, Dict[str, Any]],
    *,
    program_frame_keys: Optional[List[str]] = None,
) -> List[str]:
    keys: List[str] = []

    def _push(token: str) -> None:
        token = str(token or "").strip()
        if token and token not in keys:
            keys.append(token)

    for token in list(program_frame_keys or []):
        _push(str(token))

    for edge in active_incoming:
        config = dict(active_edge_configs.get(edge.edge_id, {}) or {})
        for resource in list(config.get("resources", []) or []):
            _push(str(resource))
        if str(edge.condition_id or "").strip():
            _push(str(edge.condition_id))

    for attr in _node_frame_hints(node_id):
        _push(_symbolic_attr_token(ctx, attr))

    return keys[:8]


def _format_frame_keys(frame_keys: List[str], *, limit: int = 6) -> str:
    clipped = list(frame_keys[:limit])
    if len(frame_keys) > limit:
        clipped.append(f"+{len(frame_keys) - limit}")
    return ", ".join(clipped)


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
