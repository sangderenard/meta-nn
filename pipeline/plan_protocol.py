"""
Serializable graph plan and GUI/worker IPC contract for the nodus pipeline.

This module is intentionally logical rather than executable:
  * It records what the training graph is, not how it is currently implemented.
  * It uses stable ids so plans, GUI layouts, and worker state can round-trip.
  * It is JSON-friendly so the batch file can eventually be replaced by plan files.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.graph import PipelineGraph


PLAN_SCHEMA_VERSION = 3
IPC_PROTOCOL_VERSION = 1

DEFAULT_PLAN_FILENAME = "training_graph_plan.json"
DEFAULT_RUNTIME_SNAPSHOT_FILENAME = "graph_runtime_snapshot.json"

MESSAGE_TYPE_WORKER_HELLO = "worker_hello"
MESSAGE_TYPE_PLAN_SNAPSHOT = "plan_snapshot"
MESSAGE_TYPE_PLAN_APPLY = "plan_apply"
MESSAGE_TYPE_PLAN_PATCH = "plan_patch"
MESSAGE_TYPE_RUN_CONTROL = "run_control"
MESSAGE_TYPE_RUNTIME_SNAPSHOT = "runtime_snapshot"
MESSAGE_TYPE_EXECUTION_EVENT = "execution_event"
MESSAGE_TYPE_GUI_SELECTION = "gui_selection"


def _clean_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    return {str(k): v for k, v in data.items() if v is not None}


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(value.__dict__)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _stable_digest(data: Any) -> str:
    blob = json.dumps(_jsonable(data), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def _default_label(raw_id: str) -> str:
    text = str(raw_id or "").replace("-", " ").replace("_", " ").strip()
    return " ".join(part.capitalize() for part in text.split())


def _edge_identity(source_id: str, target_id: str, kind: str, condition_id: str, ordinal: int) -> str:
    tail = str(kind or condition_id or "flow").replace(" ", "_").replace(":", "_")
    tail = "".join(ch for ch in tail if ch.isalnum() or ch == "_").strip("_") or "flow"
    return f"{source_id}__to__{target_id}__{tail}_{ordinal:02d}"


def serialize_config_blobs(configs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for key, value in configs.items():
        blob = _jsonable(value)
        out[str(key)] = blob if isinstance(blob, dict) else {"value": blob}
    return out


@dataclass
class NodePosition:
    x: float = 0.0
    y: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"x": float(self.x), "y": float(self.y)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodePosition":
        return cls(
            x=float(data.get("x", 0.0)),
            y=float(data.get("y", 0.0)),
        )


@dataclass
class ActionRecord:
    """Unified action object — any callable unit in the IR.

    Captures edge traversals, node-internal methods, subnode processes,
    and condition checks.  Every executable behaviour in the graph is
    represented by an ActionRecord so that the plan is both a complete
    dependency description *and* a runnable program.
    """

    action_id: str
    kind: str = "edge_traverse"  # edge_traverse | node_method | subnode_process | condition_check
    callable_ref: str = ""       # e.g. "DataNode.provide_pregestation_loader"
    owner_node_id: str = ""      # node that owns this action (empty for edge-global)
    reaction_name: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    resources_in: list[str] = field(default_factory=list)
    resources_out: list[str] = field(default_factory=list)
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": str(self.action_id),
            "kind": str(self.kind),
            "callable_ref": str(self.callable_ref),
            "owner_node_id": str(self.owner_node_id),
            "reaction_name": str(self.reaction_name),
            "parameters": _jsonable(self.parameters),
            "resources_in": [str(r) for r in self.resources_in],
            "resources_out": [str(r) for r in self.resources_out],
            "enabled": bool(self.enabled),
            "metadata": _jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionRecord":
        return cls(
            action_id=str(data["action_id"]),
            kind=str(data.get("kind", "edge_traverse")),
            callable_ref=str(data.get("callable_ref", "")),
            owner_node_id=str(data.get("owner_node_id", "")),
            reaction_name=str(data.get("reaction_name", "")),
            parameters=dict(data.get("parameters", {})),
            resources_in=[str(r) for r in data.get("resources_in", [])],
            resources_out=[str(r) for r in data.get("resources_out", [])],
            enabled=bool(data.get("enabled", True)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class SubnodeRecord:
    """An inner process owned by one node — analogous to a class method.

    Subnodes model the internal faculties of a node: individual training
    cores, evaluation passes, data preparation steps, etc.  They are
    definitively inside the purview of a single node, not utilities, not
    global-scope faculties, not cross-boundary behaviours.
    """

    subnode_id: str
    parent_node_id: str
    kind: str = "execute"  # execute | training_core | gate_eval | data_prep | io | sync
    label: str = ""
    action_ids: list[str] = field(default_factory=list)
    order: int = 0
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subnode_id": str(self.subnode_id),
            "parent_node_id": str(self.parent_node_id),
            "kind": str(self.kind),
            "label": str(self.label),
            "action_ids": [str(aid) for aid in self.action_ids],
            "order": int(self.order),
            "enabled": bool(self.enabled),
            "metadata": _jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SubnodeRecord":
        return cls(
            subnode_id=str(data["subnode_id"]),
            parent_node_id=str(data["parent_node_id"]),
            kind=str(data.get("kind", "execute")),
            label=str(data.get("label", "")),
            action_ids=[str(aid) for aid in data.get("action_ids", [])],
            order=int(data.get("order", 0)),
            enabled=bool(data.get("enabled", True)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class GraphNodeRecord:
    node_id: str
    kind: str
    label: str = ""
    icon: str = ""
    config_id: str = ""
    group_id: str = ""
    object_type: str = "object"
    faculty: str = "other"
    archetype: str = ""
    layer: str = "execution"
    enabled: bool = True
    execution_policy: str = "always"  # always | once | gated | periodic
    execution_policy_config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    subnodes: list[SubnodeRecord] = field(default_factory=list)
    owned_action_ids: list[str] = field(default_factory=list)
    gpu_models: list[str] = field(default_factory=list)
    run_condition_expr: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": str(self.node_id),
            "kind": str(self.kind),
            "label": str(self.label),
            "icon": str(self.icon),
            "config_id": str(self.config_id),
            "group_id": str(self.group_id),
            "object_type": str(self.object_type),
            "faculty": str(self.faculty),
            "archetype": str(self.archetype),
            "layer": str(self.layer),
            "enabled": bool(self.enabled),
            "execution_policy": str(self.execution_policy),
            "execution_policy_config": _jsonable(self.execution_policy_config),
            "run_condition_expr": str(self.run_condition_expr),
            "metadata": _jsonable(self.metadata),
            "subnodes": [sn.to_dict() for sn in self.subnodes],
            "owned_action_ids": [str(aid) for aid in self.owned_action_ids],
            "gpu_models": [str(m) for m in self.gpu_models],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphNodeRecord":
        return cls(
            node_id=str(data["node_id"]),
            kind=str(data["kind"]),
            label=str(data.get("label", "")),
            icon=str(data.get("icon", "")),
            config_id=str(data.get("config_id", "")),
            group_id=str(data.get("group_id", "")),
            object_type=str(data.get("object_type", "object")),
            faculty=str(data.get("faculty", "other")),
            archetype=str(data.get("archetype", "")),
            layer=str(data.get("layer", "execution")),
            enabled=bool(data.get("enabled", True)),
            execution_policy=str(data.get("execution_policy", "always")),
            execution_policy_config=dict(data.get("execution_policy_config", {})),
            metadata=dict(data.get("metadata", {})),
            subnodes=[
                SubnodeRecord.from_dict(dict(sn))
                for sn in data.get("subnodes", [])
                if isinstance(sn, dict)
            ],
            owned_action_ids=[str(aid) for aid in data.get("owned_action_ids", [])],
            gpu_models=[str(m) for m in data.get("gpu_models", [])],
            run_condition_expr=str(data.get("run_condition_expr", "")),
        )


@dataclass
class GraphEdgeRecord:
    edge_id: str
    kind: str
    source_node_id: str
    target_node_id: str
    condition_id: str = ""
    layer: str = "execution"
    target_function: str = ""
    reaction_name: str = ""
    reaction_defaults: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    action_id: str = ""
    cycle_control: Dict[str, Any] = field(default_factory=dict)
    condition_expr: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": str(self.edge_id),
            "kind": str(self.kind),
            "source_node_id": str(self.source_node_id),
            "target_node_id": str(self.target_node_id),
            "condition_id": str(self.condition_id),
            "condition_expr": str(self.condition_expr),
            "layer": str(self.layer),
            "target_function": str(self.target_function),
            "reaction_name": str(self.reaction_name),
            "reaction_defaults": _jsonable(self.reaction_defaults),
            "enabled": bool(self.enabled),
            "metadata": _jsonable(self.metadata),
            "action_id": str(self.action_id),
            "cycle_control": _jsonable(self.cycle_control),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphEdgeRecord":
        return cls(
            edge_id=str(data["edge_id"]),
            kind=str(data["kind"]),
            source_node_id=str(data["source_node_id"]),
            target_node_id=str(data["target_node_id"]),
            condition_id=str(data.get("condition_id", "")),
            layer=str(data.get("layer", "execution")),
            target_function=str(data.get("target_function", "")),
            reaction_name=str(data.get("reaction_name", "")),
            reaction_defaults=dict(data.get("reaction_defaults", {})),
            enabled=bool(data.get("enabled", True)),
            metadata=dict(data.get("metadata", {})),
            action_id=str(data.get("action_id", "")),
            cycle_control=dict(data.get("cycle_control", {})),
            condition_expr=str(data.get("condition_expr", "")),
        )


@dataclass
class GraphLayoutRecord:
    positions: Dict[str, NodePosition] = field(default_factory=dict)
    groups: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    annotations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "positions": {
                str(node_id): pos.to_dict()
                for node_id, pos in self.positions.items()
            },
            "groups": _jsonable(self.groups),
            "annotations": _jsonable(self.annotations),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphLayoutRecord":
        raw_positions = data.get("positions", {})
        positions = {
            str(node_id): NodePosition.from_dict(dict(pos))
            for node_id, pos in raw_positions.items()
            if isinstance(pos, dict)
        }
        return cls(
            positions=positions,
            groups=dict(data.get("groups", {})),
            annotations=list(data.get("annotations", [])),
        )


@dataclass
class TrainingGraphPlan:
    plan_id: str
    name: str
    revision: int = 0
    schema_version: int = PLAN_SCHEMA_VERSION
    nodes: List[GraphNodeRecord] = field(default_factory=list)
    edges: List[GraphEdgeRecord] = field(default_factory=list)
    actions: List[ActionRecord] = field(default_factory=list)
    entry_node_ids: List[str] = field(default_factory=list)
    config_blobs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    condition_blobs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    worker_hints: Dict[str, Any] = field(default_factory=dict)
    graph_layers: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    execution_program: Dict[str, Any] = field(default_factory=dict)
    layout: GraphLayoutRecord = field(default_factory=GraphLayoutRecord)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        node_ids = [node.node_id for node in self.nodes]
        edge_ids = [edge.edge_id for edge in self.edges]

        if not str(self.plan_id).strip():
            raise ValueError("TrainingGraphPlan.plan_id must not be empty.")
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("TrainingGraphPlan contains duplicate node ids.")
        if len(set(edge_ids)) != len(edge_ids):
            raise ValueError("TrainingGraphPlan contains duplicate edge ids.")

        known_nodes = set(node_ids)
        for edge in self.edges:
            if edge.source_node_id not in known_nodes:
                raise ValueError(
                    f"Edge {edge.edge_id!r} references unknown source node {edge.source_node_id!r}."
                )
            if edge.target_node_id not in known_nodes:
                raise ValueError(
                    f"Edge {edge.edge_id!r} references unknown target node {edge.target_node_id!r}."
                )
        for entry_node_id in self.entry_node_ids:
            if entry_node_id not in known_nodes:
                raise ValueError(
                    f"Entry node {entry_node_id!r} is not present in the node list."
                )

        in_degree = {node_id: 0 for node_id in known_nodes}
        children = {node_id: [] for node_id in known_nodes}
        for edge in self.edges:
            if not edge.enabled:
                continue
            if edge.layer == "cycle" or edge.cycle_control:
                continue  # cycle edges are intentional back-edges; skip DAG check
            in_degree[edge.target_node_id] += 1
            children[edge.source_node_id].append(edge.target_node_id)

        ready = deque(sorted(node_id for node_id, deg in in_degree.items() if deg == 0))
        visited = 0
        while ready:
            node_id = ready.popleft()
            visited += 1
            for child in children.get(node_id, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    ready.append(child)
        if visited != len(known_nodes):
            raise ValueError("TrainingGraphPlan contains a cycle in its non-cycle edges.")

        action_ids = [action.action_id for action in self.actions]
        if len(set(action_ids)) != len(action_ids):
            raise ValueError("TrainingGraphPlan contains duplicate action ids.")
        action_id_set = set(action_ids)
        for edge in self.edges:
            if edge.action_id and edge.action_id not in action_id_set:
                raise ValueError(
                    f"Edge {edge.edge_id!r} references unknown action {edge.action_id!r}."
                )
        for node in self.nodes:
            for aid in node.owned_action_ids:
                if aid not in action_id_set:
                    raise ValueError(
                        f"Node {node.node_id!r} references unknown owned action {aid!r}."
                    )
            for subnode in node.subnodes:
                if subnode.parent_node_id != node.node_id:
                    raise ValueError(
                        f"Subnode {subnode.subnode_id!r} parent mismatch: "
                        f"expected {node.node_id!r}, got {subnode.parent_node_id!r}."
                    )
                for aid in subnode.action_ids:
                    if aid not in action_id_set:
                        raise ValueError(
                            f"Subnode {subnode.subnode_id!r} references unknown action {aid!r}."
                        )

    def node_map(self) -> Dict[str, GraphNodeRecord]:
        return {node.node_id: node for node in self.nodes}

    def edge_map(self) -> Dict[str, GraphEdgeRecord]:
        return {edge.edge_id: edge for edge in self.edges}

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "schema_version": int(self.schema_version),
            "plan_id": str(self.plan_id),
            "name": str(self.name),
            "revision": int(self.revision),
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "actions": [action.to_dict() for action in self.actions],
            "entry_node_ids": [str(node_id) for node_id in self.entry_node_ids],
            "config_blobs": _jsonable(self.config_blobs),
            "condition_blobs": _jsonable(self.condition_blobs),
            "worker_hints": _jsonable(self.worker_hints),
            "graph_layers": _jsonable(self.graph_layers),
            "execution_program": _jsonable(self.execution_program),
            "layout": self.layout.to_dict(),
            "metadata": _jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingGraphPlan":
        plan = cls(
            schema_version=int(data.get("schema_version", PLAN_SCHEMA_VERSION)),
            plan_id=str(data["plan_id"]),
            name=str(data.get("name", "")),
            revision=int(data.get("revision", 0)),
            nodes=[
                GraphNodeRecord.from_dict(dict(node))
                for node in data.get("nodes", [])
                if isinstance(node, dict)
            ],
            edges=[
                GraphEdgeRecord.from_dict(dict(edge))
                for edge in data.get("edges", [])
                if isinstance(edge, dict)
            ],
            actions=[
                ActionRecord.from_dict(dict(a))
                for a in data.get("actions", [])
                if isinstance(a, dict)
            ],
            entry_node_ids=[str(node_id) for node_id in data.get("entry_node_ids", [])],
            config_blobs=dict(data.get("config_blobs", {})),
            condition_blobs=dict(data.get("condition_blobs", {})),
            worker_hints=dict(data.get("worker_hints", {})),
            graph_layers=dict(data.get("graph_layers", {})),
            execution_program=dict(data.get("execution_program", {})),
            layout=GraphLayoutRecord.from_dict(dict(data.get("layout", {}))),
            metadata=dict(data.get("metadata", {})),
        )
        plan.validate()
        return plan

    def save_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "TrainingGraphPlan":
        p = Path(path)
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))


@dataclass
class ProtocolEnvelope:
    message_type: str
    payload: Dict[str, Any]
    protocol_version: int = IPC_PROTOCOL_VERSION
    session_id: str = ""
    worker_id: str = ""
    plan_id: str = ""
    revision: int = 0
    message_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return _clean_dict(
            {
                "protocol_version": int(self.protocol_version),
                "message_type": str(self.message_type),
                "session_id": str(self.session_id),
                "worker_id": str(self.worker_id),
                "plan_id": str(self.plan_id),
                "revision": int(self.revision),
                "message_id": str(self.message_id),
                "payload": _jsonable(self.payload),
            }
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProtocolEnvelope":
        return cls(
            protocol_version=int(data.get("protocol_version", IPC_PROTOCOL_VERSION)),
            message_type=str(data["message_type"]),
            session_id=str(data.get("session_id", "")),
            worker_id=str(data.get("worker_id", "")),
            plan_id=str(data.get("plan_id", "")),
            revision=int(data.get("revision", 0)),
            message_id=str(data.get("message_id", "")),
            payload=dict(data.get("payload", {})),
        )


@dataclass
class WorkerHelloPayload:
    worker_id: str
    worker_label: str = ""
    capabilities: Dict[str, Any] = field(default_factory=dict)
    loaded_plan_id: str = ""
    loaded_revision: int = 0
    execution_state: str = "idle"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker_id": str(self.worker_id),
            "worker_label": str(self.worker_label),
            "capabilities": _jsonable(self.capabilities),
            "loaded_plan_id": str(self.loaded_plan_id),
            "loaded_revision": int(self.loaded_revision),
            "execution_state": str(self.execution_state),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkerHelloPayload":
        return cls(
            worker_id=str(data.get("worker_id", "")),
            worker_label=str(data.get("worker_label", "")),
            capabilities=dict(data.get("capabilities", {})),
            loaded_plan_id=str(data.get("loaded_plan_id", "")),
            loaded_revision=int(data.get("loaded_revision", 0)),
            execution_state=str(data.get("execution_state", "idle")),
        )


@dataclass
class PlanSnapshotPayload:
    plan: TrainingGraphPlan

    def to_dict(self) -> Dict[str, Any]:
        return {"plan": self.plan.to_dict()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanSnapshotPayload":
        return cls(plan=TrainingGraphPlan.from_dict(dict(data["plan"])))


@dataclass
class PlanApplyPayload:
    plan: TrainingGraphPlan
    replace_current: bool = True
    activate_after_apply: bool = True
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "replace_current": bool(self.replace_current),
            "activate_after_apply": bool(self.activate_after_apply),
            "reason": str(self.reason),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanApplyPayload":
        return cls(
            plan=TrainingGraphPlan.from_dict(dict(data["plan"])),
            replace_current=bool(data.get("replace_current", True)),
            activate_after_apply=bool(data.get("activate_after_apply", True)),
            reason=str(data.get("reason", "")),
        )


@dataclass
class PlanPatchPayload:
    patch_ops: List[Dict[str, Any]] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patch_ops": _jsonable(self.patch_ops),
            "reason": str(self.reason),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanPatchPayload":
        return cls(
            patch_ops=list(data.get("patch_ops", [])),
            reason=str(data.get("reason", "")),
        )


@dataclass
class RunControlPayload:
    command: Literal["start", "pause", "resume", "stop", "step"] = "start"
    selected_cycle_ids: List[int] = field(default_factory=list)
    selected_node_ids: List[str] = field(default_factory=list)
    gate_override: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command": str(self.command),
            "selected_cycle_ids": [int(x) for x in self.selected_cycle_ids],
            "selected_node_ids": [str(x) for x in self.selected_node_ids],
            "gate_override": bool(self.gate_override),
            "metadata": _jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunControlPayload":
        return cls(
            command=str(data.get("command", "start")),
            selected_cycle_ids=[int(x) for x in data.get("selected_cycle_ids", [])],
            selected_node_ids=[str(x) for x in data.get("selected_node_ids", [])],
            gate_override=bool(data.get("gate_override", False)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class RuntimeNodeState:
    node_id: str
    status: str
    last_result: str = ""
    last_started_at: float = 0.0
    last_finished_at: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": str(self.node_id),
            "status": str(self.status),
            "last_result": str(self.last_result),
            "last_started_at": float(self.last_started_at),
            "last_finished_at": float(self.last_finished_at),
            "metrics": _jsonable(self.metrics),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuntimeNodeState":
        return cls(
            node_id=str(data.get("node_id", "")),
            status=str(data.get("status", "")),
            last_result=str(data.get("last_result", "")),
            last_started_at=float(data.get("last_started_at", 0.0)),
            last_finished_at=float(data.get("last_finished_at", 0.0)),
            metrics=dict(data.get("metrics", {})),
        )


@dataclass
class RuntimeSnapshotPayload:
    worker_id: str
    execution_state: str
    active_node_ids: List[str] = field(default_factory=list)
    selected_cycle_ids: List[int] = field(default_factory=list)
    gate_override: bool = False
    node_states: List[RuntimeNodeState] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker_id": str(self.worker_id),
            "execution_state": str(self.execution_state),
            "active_node_ids": [str(x) for x in self.active_node_ids],
            "selected_cycle_ids": [int(x) for x in self.selected_cycle_ids],
            "gate_override": bool(self.gate_override),
            "node_states": [node_state.to_dict() for node_state in self.node_states],
            "metrics": _jsonable(self.metrics),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuntimeSnapshotPayload":
        return cls(
            worker_id=str(data.get("worker_id", "")),
            execution_state=str(data.get("execution_state", "")),
            active_node_ids=[str(x) for x in data.get("active_node_ids", [])],
            selected_cycle_ids=[int(x) for x in data.get("selected_cycle_ids", [])],
            gate_override=bool(data.get("gate_override", False)),
            node_states=[
                RuntimeNodeState.from_dict(dict(node_state))
                for node_state in data.get("node_states", [])
                if isinstance(node_state, dict)
            ],
            metrics=dict(data.get("metrics", {})),
        )


@dataclass
class ExecutionEventPayload:
    event_id: str
    node_id: str = ""
    edge_id: str = ""
    kind: str = ""
    phase: str = ""
    status: str = ""
    ts: float = 0.0
    message: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": str(self.event_id),
            "node_id": str(self.node_id),
            "edge_id": str(self.edge_id),
            "kind": str(self.kind),
            "phase": str(self.phase),
            "status": str(self.status),
            "ts": float(self.ts),
            "message": str(self.message),
            "metrics": _jsonable(self.metrics),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionEventPayload":
        return cls(
            event_id=str(data.get("event_id", "")),
            node_id=str(data.get("node_id", "")),
            edge_id=str(data.get("edge_id", "")),
            kind=str(data.get("kind", "")),
            phase=str(data.get("phase", "")),
            status=str(data.get("status", "")),
            ts=float(data.get("ts", 0.0)),
            message=str(data.get("message", "")),
            metrics=dict(data.get("metrics", {})),
        )


@dataclass
class GuiSelectionPayload:
    selected_node_ids: List[str] = field(default_factory=list)
    selected_edge_ids: List[str] = field(default_factory=list)
    selected_cycle_ids: List[int] = field(default_factory=list)
    inspector_target: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_node_ids": [str(x) for x in self.selected_node_ids],
            "selected_edge_ids": [str(x) for x in self.selected_edge_ids],
            "selected_cycle_ids": [int(x) for x in self.selected_cycle_ids],
            "inspector_target": str(self.inspector_target),
            "metadata": _jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GuiSelectionPayload":
        return cls(
            selected_node_ids=[str(x) for x in data.get("selected_node_ids", [])],
            selected_edge_ids=[str(x) for x in data.get("selected_edge_ids", [])],
            selected_cycle_ids=[int(x) for x in data.get("selected_cycle_ids", [])],
            inspector_target=str(data.get("inspector_target", "")),
            metadata=dict(data.get("metadata", {})),
        )


_PAYLOAD_TYPES = {
    MESSAGE_TYPE_WORKER_HELLO: WorkerHelloPayload,
    MESSAGE_TYPE_PLAN_SNAPSHOT: PlanSnapshotPayload,
    MESSAGE_TYPE_PLAN_APPLY: PlanApplyPayload,
    MESSAGE_TYPE_PLAN_PATCH: PlanPatchPayload,
    MESSAGE_TYPE_RUN_CONTROL: RunControlPayload,
    MESSAGE_TYPE_RUNTIME_SNAPSHOT: RuntimeSnapshotPayload,
    MESSAGE_TYPE_EXECUTION_EVENT: ExecutionEventPayload,
    MESSAGE_TYPE_GUI_SELECTION: GuiSelectionPayload,
}


def make_envelope(
    message_type: str,
    payload: Any,
    *,
    session_id: str = "",
    worker_id: str = "",
    plan_id: str = "",
    revision: int = 0,
    message_id: str = "",
) -> ProtocolEnvelope:
    payload_dict = payload.to_dict() if hasattr(payload, "to_dict") else dict(payload)
    return ProtocolEnvelope(
        message_type=message_type,
        payload=payload_dict,
        session_id=session_id,
        worker_id=worker_id,
        plan_id=plan_id,
        revision=int(revision),
        message_id=message_id,
    )


def is_protocol_envelope_message(data: Any) -> bool:
    return isinstance(data, dict) and ("message_type" in data) and ("payload" in data)


def parse_payload(message_type: str, payload: Dict[str, Any]) -> Any:
    payload_type = _PAYLOAD_TYPES.get(str(message_type))
    if payload_type is None:
        return dict(payload)
    return payload_type.from_dict(dict(payload))


def parse_envelope(data: Dict[str, Any]) -> tuple[ProtocolEnvelope, Any]:
    envelope = ProtocolEnvelope.from_dict(dict(data))
    return envelope, parse_payload(envelope.message_type, envelope.payload)


def save_protocol_message(path: str | Path, envelope: ProtocolEnvelope) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(envelope.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _topological_sort_node_ids(
    node_ids: List[str], edges: List[GraphEdgeRecord],
) -> List[str]:
    """Return node IDs in topological order, excluding cycle edges."""
    ids = list(node_ids)
    children: Dict[str, List[str]] = {nid: [] for nid in ids}
    in_deg: Dict[str, int] = {nid: 0 for nid in ids}
    for edge in edges:
        if not edge.enabled or edge.layer == "cycle" or edge.cycle_control:
            continue
        if edge.source_node_id in children and edge.target_node_id in in_deg:
            children[edge.source_node_id].append(edge.target_node_id)
            in_deg[edge.target_node_id] += 1
    ready = deque(sorted(nid for nid, d in in_deg.items() if d == 0))
    order: List[str] = []
    while ready:
        nid = ready.popleft()
        order.append(nid)
        for child in children.get(nid, []):
            in_deg[child] -= 1
            if in_deg[child] == 0:
                ready.append(child)
    # Append any remaining (shouldn't happen in a valid DAG, but be safe).
    remaining = [nid for nid in ids if nid not in set(order)]
    order.extend(sorted(remaining))
    return order


def _root_node_ids(node_ids: Iterable[str], edges: List[GraphEdgeRecord]) -> List[str]:
    in_degree = {node_id: 0 for node_id in node_ids}
    for edge in edges:
        if not edge.enabled or edge.layer == "cycle" or edge.cycle_control:
            continue
        if edge.target_node_id in in_degree:
            in_degree[edge.target_node_id] += 1
    return sorted(node_id for node_id, deg in in_degree.items() if deg == 0)


def build_layout_from_records(
    nodes: List[GraphNodeRecord],
    edges: List[GraphEdgeRecord],
) -> GraphLayoutRecord:
    node_ids = [node.node_id for node in nodes]
    children: Dict[str, List[str]] = {node_id: [] for node_id in node_ids}
    in_degree: Dict[str, int] = {node_id: 0 for node_id in node_ids}

    for edge in edges:
        if edge.layer == "cycle" or edge.cycle_control:
            continue
        if edge.source_node_id not in children or edge.target_node_id not in children:
            continue
        children[edge.source_node_id].append(edge.target_node_id)
        in_degree[edge.target_node_id] += 1

    depth: Dict[str, int] = {node_id: 0 for node_id in node_ids}
    ready = deque(sorted(node_id for node_id, deg in in_degree.items() if deg == 0))
    while ready:
        node_id = ready.popleft()
        base_depth = depth.get(node_id, 0)
        for child in children.get(node_id, []):
            depth[child] = max(depth.get(child, 0), base_depth + 1)
            in_degree[child] -= 1
            if in_degree[child] == 0:
                ready.append(child)

    lane_order = [
        "bootstrap",
        "build",
        "vocab",
        "data",
        "train",
        "gates",
        "housekeeping",
        "other",
    ]
    lane_index = {lane: idx for idx, lane in enumerate(lane_order)}
    grouped_nodes: Dict[str, List[GraphNodeRecord]] = defaultdict(list)
    for node in nodes:
        lane = node.group_id or "other"
        grouped_nodes[lane].append(node)

    positions: Dict[str, NodePosition] = {}
    groups: Dict[str, Dict[str, Any]] = {}
    for lane, lane_nodes in grouped_nodes.items():
        groups[lane] = {
            "label": _default_label(lane),
            "lane_index": int(lane_index.get(lane, len(lane_order))),
        }

        local_rows: Dict[int, int] = defaultdict(int)
        ordered = sorted(
            lane_nodes,
            key=lambda node: (depth.get(node.node_id, 0), node.label or node.node_id),
        )
        base_y = float(lane_index.get(lane, len(lane_order)) * 150.0)
        for node in ordered:
            col = int(depth.get(node.node_id, 0))
            x = float(col * 240.0)
            y = base_y + float(local_rows[col] * 56.0)
            local_rows[col] += 1
            positions[node.node_id] = NodePosition(x=x, y=y)

    return GraphLayoutRecord(positions=positions, groups=groups, annotations=[])


# ---------------------------------------------------------------------------
# Subnode / action introspection
# ---------------------------------------------------------------------------

def _has_overridden_method(cls: type, method_name: str) -> bool:
    """True if *cls* defines *method_name* rather than inheriting it unchanged."""
    method = getattr(cls, method_name, None)
    if method is None:
        return False
    for base in cls.__mro__[1:]:
        base_method = getattr(base, method_name, None)
        if base_method is not None:
            return method is not base_method
    return False


def _parse_declared_subnodes(
    node_id: str,
    raw: list,
) -> tuple[list[SubnodeRecord], list[ActionRecord]]:
    """Convert raw ``declare_subnodes()`` output to typed records."""
    subnodes: list[SubnodeRecord] = []
    actions: list[ActionRecord] = []
    for idx, item in enumerate(raw):
        d = dict(item) if isinstance(item, dict) else {}
        sn_id = str(d.get("subnode_id", f"{node_id}::sub_{idx}"))
        kind = str(d.get("kind", "execute"))
        label = str(d.get("label", sn_id))
        callable_ref = str(d.get("callable_ref", ""))
        action_id = f"action::node::{node_id}::{kind}_{idx}"
        if callable_ref:
            actions.append(ActionRecord(
                action_id=action_id,
                kind="subnode_process",
                callable_ref=callable_ref,
                owner_node_id=node_id,
                metadata=dict(d.get("metadata", {})),
            ))
        subnode_action_ids = [action_id] if callable_ref else []
        subnodes.append(SubnodeRecord(
            subnode_id=sn_id,
            parent_node_id=node_id,
            kind=kind,
            label=label,
            action_ids=subnode_action_ids,
            order=int(d.get("order", idx)),
            metadata=dict(d.get("metadata", {})),
        ))
    return subnodes, actions


def _infer_subnodes_and_actions(
    node_id: str,
    node: Any,
) -> tuple[list[SubnodeRecord], list[ActionRecord]]:
    """Discover inner faculties of *node* for IR representation.

    If the node implements ``declare_subnodes()``, those are used verbatim.
    Otherwise a minimal set of subnodes is inferred from class structure:
    an ``execute`` process (always), and a ``gate_check`` process when the
    node overrides ``should_run``.
    """
    declare_fn = getattr(node, "declare_subnodes", None)
    if callable(declare_fn):
        try:
            raw = declare_fn()
            if isinstance(raw, list) and raw:
                return _parse_declared_subnodes(node_id, raw)
        except Exception:
            pass

    class_name = type(node).__name__
    subnodes: list[SubnodeRecord] = []
    actions: list[ActionRecord] = []

    exec_action_id = f"action::node::{node_id}::execute"
    actions.append(ActionRecord(
        action_id=exec_action_id,
        kind="node_method",
        callable_ref=f"{class_name}.execute",
        owner_node_id=node_id,
    ))

    has_custom_should_run = _has_overridden_method(type(node), "should_run")
    should_run_action_ids: list[str] = []
    if has_custom_should_run:
        gate_action_id = f"action::node::{node_id}::should_run"
        actions.append(ActionRecord(
            action_id=gate_action_id,
            kind="condition_check",
            callable_ref=f"{class_name}.should_run",
            owner_node_id=node_id,
        ))
        should_run_action_ids.append(gate_action_id)
        subnodes.append(SubnodeRecord(
            subnode_id=f"{node_id}::should_run",
            parent_node_id=node_id,
            kind="gate_check",
            label="Gate check",
            action_ids=should_run_action_ids,
            order=-1,
        ))

    subnodes.append(SubnodeRecord(
        subnode_id=f"{node_id}::execute",
        parent_node_id=node_id,
        kind="execute",
        label=str(getattr(node, "description", "") or class_name),
        action_ids=[exec_action_id],
        order=0,
    ))

    return subnodes, actions


def _synthesize_edge_condition_expr(condition_id: str) -> str:
    """Return a portable condition expression for a given condition_id."""
    if not str(condition_id or "").strip():
        return ""
    from pipeline.condition_expr import expr_for_condition_id
    return expr_for_condition_id(str(condition_id).strip())


def _synthesize_node_run_expr(node: Any) -> str:
    """Return a run_condition_expr for a live PipelineNode."""
    try:
        from pipeline.condition_expr import expr_for_node_should_run
        return expr_for_node_should_run(node)
    except Exception:
        return ""


def _extract_training_mechanics(node: Any) -> Dict[str, Any]:
    """Return a JSON-stable training-mechanics declaration for *node*."""
    declare_fn = getattr(node, "declare_training_mechanics", None)
    if not callable(declare_fn):
        return {}
    try:
        raw = declare_fn()
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    if not raw:
        return {}

    def _norm_item(item: Any) -> Dict[str, Any]:
        data = dict(item) if isinstance(item, dict) else {}
        return {
            "id": str(data.get("id", "") or "").strip(),
            "label": str(data.get("label", "") or "").strip(),
            "kind": str(data.get("kind", "state") or "state").strip(),
            "detail": str(data.get("detail", "") or "").strip(),
            "role": str(data.get("role", "") or "").strip(),
        }

    def _norm_flow(item: Any) -> Dict[str, Any]:
        data = dict(item) if isinstance(item, dict) else {}
        return {
            "source": str(data.get("source", "") or "").strip(),
            "target": str(data.get("target", "") or "").strip(),
            "label": str(data.get("label", "") or "").strip(),
            "kind": str(data.get("kind", "flow") or "flow").strip(),
        }

    return {
        "enabled": bool(raw.get("enabled", True)),
        "module_family": str(raw.get("module_family", "trainer") or "trainer").strip(),
        "module_label": str(raw.get("module_label", "") or "").strip(),
        "summary": str(raw.get("summary", "") or "").strip(),
        "inputs": [item for item in (_norm_item(x) for x in list(raw.get("inputs", []) or [])) if item["id"]],
        "losses": [item for item in (_norm_item(x) for x in list(raw.get("losses", []) or [])) if item["id"]],
        "outputs": [item for item in (_norm_item(x) for x in list(raw.get("outputs", []) or [])) if item["id"]],
        "flows": [
            flow
            for flow in (_norm_flow(x) for x in list(raw.get("flows", []) or []))
            if flow["source"] and flow["target"]
        ],
    }


def _extract_ir_node_contract(node: Any) -> Dict[str, Any]:
    """Return a JSON-stable executable-contract declaration for *node*."""
    declare_fn = getattr(node, "declare_ir_node_contract", None)
    if not callable(declare_fn):
        return {}
    try:
        raw = declare_fn()
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    if not raw:
        return {}
    return _jsonable(dict(raw))


def plan_from_pipeline_graph(
    graph: "PipelineGraph",
    *,
    name: str = "",
    revision: int = 1,
    config_blobs: Optional[Dict[str, Any]] = None,
    condition_blobs: Optional[Dict[str, Any]] = None,
    worker_hints: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    node_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    graph_layers: Optional[Dict[str, Any]] = None,
    execution_program: Optional[Dict[str, Any]] = None,
) -> TrainingGraphPlan:
    raw_nodes = getattr(graph, "nodes", None)
    raw_nodes = raw_nodes if isinstance(raw_nodes, dict) else getattr(graph, "_nodes", {})
    raw_edges = getattr(graph, "edges", None)
    raw_edges = raw_edges if isinstance(raw_edges, list) else getattr(graph, "_edges", [])

    if execution_program is None:
        try:
            from pipeline.graph_layers import build_execution_program as _build_execution_program

            execution_program = _build_execution_program(graph)
        except Exception:
            execution_program = {}

    node_records: List[GraphNodeRecord] = []
    node_meta_map = dict(node_metadata or {})
    for node_id, node in raw_nodes.items():
        meta = dict(node_meta_map.get(str(node_id), {}))
        runtime_shape = getattr(node, "runtime_shape", None)
        if callable(runtime_shape):
            try:
                runtime_shape = runtime_shape()
            except TypeError:
                runtime_shape = getattr(node, "runtime_shape", None)
        shape_metadata = dict(getattr(runtime_shape, "metadata", {}) or {})
        record_meta = dict(shape_metadata)
        record_meta.update(dict(meta.get("metadata", {})))
        record_meta.setdefault("description", str(getattr(node, "description", "") or ""))
        record_meta.setdefault("node_class", type(node).__name__)
        training_mechanics = _extract_training_mechanics(node)
        if training_mechanics:
            record_meta["training_mechanics"] = _jsonable(training_mechanics)
        ir_node_contract = _extract_ir_node_contract(node)
        if ir_node_contract:
            record_meta["ir_node_contract"] = _jsonable(ir_node_contract)
        gpu_models_raw = getattr(node, "gpu_models", [])
        if callable(gpu_models_raw):
            try:
                gpu_models_raw = gpu_models_raw.fget(node) if isinstance(gpu_models_raw, property) else gpu_models_raw
            except Exception:
                gpu_models_raw = []
        if not isinstance(gpu_models_raw, (list, tuple)):
            gpu_models_raw = []

        # Execution policy — the IR must record *why* a node may skip itself.
        _exec_policy_raw = getattr(node, "runtime_execution_policy", None)
        if callable(_exec_policy_raw) and not isinstance(_exec_policy_raw, property):
            try:
                _exec_policy_raw = _exec_policy_raw()
            except Exception:
                _exec_policy_raw = None
        if isinstance(_exec_policy_raw, (tuple, list)) and len(_exec_policy_raw) >= 2:
            _exec_policy_name = str(_exec_policy_raw[0])
            _exec_policy_cfg = dict(_exec_policy_raw[1]) if isinstance(_exec_policy_raw[1], dict) else {}
        else:
            _exec_policy_name = "always"
            _exec_policy_cfg = {}

        node_records.append(
            GraphNodeRecord(
                node_id=str(node_id),
                kind=str(meta.get("kind", type(node).__name__)),
                label=str(meta.get("label", _default_label(str(node_id)))),
                icon=str(meta.get("icon", "")),
                config_id=str(meta.get("config_id", "")),
                group_id=str(meta.get("group_id", "other")),
                object_type=str(meta.get("object_type", getattr(runtime_shape, "object_type", "object"))),
                faculty=str(meta.get("faculty", getattr(runtime_shape, "faculty", "other"))),
                archetype=str(meta.get("archetype", getattr(runtime_shape, "archetype", type(node).__name__))),
                layer=str(meta.get("layer", getattr(runtime_shape, "layer", "execution"))),
                enabled=bool(meta.get("enabled", True)),
                execution_policy=_exec_policy_name,
                execution_policy_config=_jsonable(_exec_policy_cfg),
                metadata=_jsonable(record_meta),
                gpu_models=[str(m) for m in gpu_models_raw],
                run_condition_expr=_synthesize_node_run_expr(node),
            )
        )

    pair_counts: Dict[tuple[str, str], int] = defaultdict(int)
    edge_records: List[GraphEdgeRecord] = []
    for edge in raw_edges:
        source_id = str(getattr(edge, "source_id"))
        target_id = str(getattr(edge, "target_id"))
        kind = str(getattr(edge, "label", "") or "flow")
        condition_id = str(getattr(edge, "condition_id", "") or "")
        pair_key = (source_id, target_id)
        pair_counts[pair_key] += 1

        reaction = getattr(edge, "reaction", None)
        reaction_defaults = dict(getattr(reaction, "defaults", {}) or {})
        serialized_defaults = _jsonable(reaction_defaults)
        if not isinstance(serialized_defaults, dict):
            serialized_defaults = {"value": serialized_defaults}

        record_meta = dict(getattr(reaction, "metadata", {}) or {})
        record_meta.update(dict(getattr(edge, "metadata", {}) or {}))
        record_meta.update(
            _clean_dict(
                {
                    "label": str(getattr(edge, "label", "") or ""),
                    "condition_callable": getattr(getattr(edge, "condition", None), "__name__", ""),
                }
            )
        )

        edge_records.append(
            GraphEdgeRecord(
                edge_id=str(
                    getattr(edge, "edge_id", "")
                    or _edge_identity(source_id, target_id, kind, condition_id, pair_counts[pair_key])
                ),
                kind=kind,
                source_node_id=source_id,
                target_node_id=target_id,
                condition_id=condition_id,
                layer=str(getattr(edge, "layer", getattr(reaction, "layer", "execution")) or "execution"),
                target_function=str(
                    getattr(reaction, "target_function", "")
                    or getattr(getattr(edge, "on_traverse", None), "__name__", "")
                ),
                reaction_name=str(getattr(reaction, "reaction_name", "") or kind or "flow"),
                reaction_defaults=serialized_defaults,
                enabled=True,
                metadata=_jsonable(record_meta),
                condition_expr=_synthesize_edge_condition_expr(condition_id),
            )
        )

    serialized_configs = serialize_config_blobs(dict(config_blobs or {}))
    serialized_conditions = serialize_config_blobs(dict(condition_blobs or {}))
    serialized_layers = _jsonable(dict(graph_layers or {}))
    serialized_execution_program = _jsonable(dict(execution_program or {}))

    # ── synthesize cycle edges from orchestrator loop metadata ────────────
    _hints = dict(worker_hints or {})
    _orch_cycles = int(_hints.get("orchestration_cycles", 0) or 0)
    _orch_rounds = int(_hints.get("orchestration_rounds", 0) or 0)
    if _orch_cycles > 0 and _orch_rounds > 0:
        # Determine topological ends: entry (root) and terminal (sink) nodes
        # among the *non-cycle* edges built so far.
        _all_node_ids = {nr.node_id for nr in node_records}
        _sources = {e.source_node_id for e in edge_records if e.enabled}
        _targets = {e.target_node_id for e in edge_records if e.enabled}
        _sink_ids = sorted(_all_node_ids & _sources - _targets | {nid for nid in _all_node_ids if nid not in _sources and nid not in _targets})
        _root_ids = sorted(_all_node_ids - _targets)
        # Prefer well-known node names when available.
        _cycle_source = "checkpoint_save" if "checkpoint_save" in _all_node_ids else (_sink_ids[0] if _sink_ids else None)
        _cycle_target = "wave_pool" if "wave_pool" in _all_node_ids else (_root_ids[0] if _root_ids else None)
        if _cycle_source and _cycle_target and _cycle_source != _cycle_target:
            _total_iters = _orch_cycles * _orch_rounds
            # Compute topological stride (distance) from target back to source.
            _topo_order = _topological_sort_node_ids([nr.node_id for nr in node_records], edge_records)
            _topo_idx = {nid: idx for idx, nid in enumerate(_topo_order)}
            _stride = abs(_topo_idx.get(_cycle_source, 0) - _topo_idx.get(_cycle_target, 0))
            edge_records.append(
                GraphEdgeRecord(
                    edge_id="cycle::round_return",
                    kind="cycle_return",
                    source_node_id=_cycle_source,
                    target_node_id=_cycle_target,
                    condition_id="",
                    layer="cycle",
                    target_function="",
                    reaction_name="round_return",
                    reaction_defaults={},
                    enabled=True,
                    metadata={
                        "style_role": "cycle",
                        "label": f"round return ({_total_iters}x)",
                        "description": (
                            f"Orchestrator repeats the graph for "
                            f"{_orch_cycles} cycle(s) x {_orch_rounds} round(s) = {_total_iters} iterations."
                        ),
                    },
                    cycle_control={
                        "stride": int(_stride),
                        "max_iterations": int(_total_iters),
                        "cycles": int(_orch_cycles),
                        "rounds_per_cycle": int(_orch_rounds),
                    },
                )
            )

    # ── build unified action records ─────────────────────────────────────
    action_records: list[ActionRecord] = []
    for edge_record in edge_records:
        resources_out = [
            str(r)
            for r in list(edge_record.reaction_defaults.get("resources", []) or [])
            if str(r or "").strip()
        ]
        edge_action = ActionRecord(
            action_id=f"action::edge::{edge_record.edge_id}",
            kind="edge_traverse",
            callable_ref=str(edge_record.target_function),
            owner_node_id=str(edge_record.source_node_id),
            reaction_name=str(edge_record.reaction_name),
            parameters=dict(edge_record.reaction_defaults),
            resources_out=resources_out,
            metadata={"edge_id": str(edge_record.edge_id)},
        )
        action_records.append(edge_action)
        edge_record.action_id = edge_action.action_id

    for node_record in node_records:
        node_obj = raw_nodes.get(str(node_record.node_id))
        if node_obj is None:
            continue
        subnodes, node_actions = _infer_subnodes_and_actions(
            str(node_record.node_id), node_obj,
        )
        node_record.subnodes = subnodes
        action_records.extend(node_actions)
        node_record.owned_action_ids = [a.action_id for a in node_actions]

    # Materialize stack_view from enriched node records
    from pipeline.graph_layers import _build_stack_view_layer  # local to avoid circular import
    serialized_layers["stack_view"] = _build_stack_view_layer(node_records, edge_records)

    plan_name = str(name or getattr(graph, "name", "Training Graph"))
    plan_digest = _stable_digest(
        {
            "name": plan_name,
            "nodes": [node.to_dict() for node in node_records],
            "edges": [edge.to_dict() for edge in edge_records],
            "actions": [action.to_dict() for action in action_records],
            "configs": serialized_configs,
            "conditions": serialized_conditions,
            "layers": serialized_layers,
            "execution_program": serialized_execution_program,
        }
    )
    plan_id = f"{str(getattr(graph, 'name', 'graph')).strip() or 'graph'}:{plan_digest}"

    plan = TrainingGraphPlan(
        plan_id=plan_id,
        name=plan_name,
        revision=int(revision),
        nodes=node_records,
        edges=edge_records,
        actions=action_records,
        entry_node_ids=_root_node_ids([node.node_id for node in node_records], edge_records),
        config_blobs=serialized_configs,
        condition_blobs=serialized_conditions,
        worker_hints=_jsonable(dict(worker_hints or {})),
        graph_layers=serialized_layers,
        execution_program=serialized_execution_program,
        layout=build_layout_from_records(node_records, edge_records),
        metadata=_jsonable(dict(metadata or {})),
    )
    plan.validate()
    return plan
