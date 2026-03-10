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


PLAN_SCHEMA_VERSION = 1
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
class GraphNodeRecord:
    node_id: str
    kind: str
    label: str = ""
    icon: str = ""
    config_id: str = ""
    group_id: str = ""
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": str(self.node_id),
            "kind": str(self.kind),
            "label": str(self.label),
            "icon": str(self.icon),
            "config_id": str(self.config_id),
            "group_id": str(self.group_id),
            "enabled": bool(self.enabled),
            "metadata": _jsonable(self.metadata),
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
            enabled=bool(data.get("enabled", True)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class GraphEdgeRecord:
    edge_id: str
    kind: str
    source_node_id: str
    target_node_id: str
    condition_id: str = ""
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": str(self.edge_id),
            "kind": str(self.kind),
            "source_node_id": str(self.source_node_id),
            "target_node_id": str(self.target_node_id),
            "condition_id": str(self.condition_id),
            "enabled": bool(self.enabled),
            "metadata": _jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphEdgeRecord":
        return cls(
            edge_id=str(data["edge_id"]),
            kind=str(data["kind"]),
            source_node_id=str(data["source_node_id"]),
            target_node_id=str(data["target_node_id"]),
            condition_id=str(data.get("condition_id", "")),
            enabled=bool(data.get("enabled", True)),
            metadata=dict(data.get("metadata", {})),
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
    entry_node_ids: List[str] = field(default_factory=list)
    config_blobs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    condition_blobs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    worker_hints: Dict[str, Any] = field(default_factory=dict)
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
            raise ValueError("TrainingGraphPlan contains a cycle.")

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
            "entry_node_ids": [str(node_id) for node_id in self.entry_node_ids],
            "config_blobs": _jsonable(self.config_blobs),
            "condition_blobs": _jsonable(self.condition_blobs),
            "worker_hints": _jsonable(self.worker_hints),
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
            entry_node_ids=[str(node_id) for node_id in data.get("entry_node_ids", [])],
            config_blobs=dict(data.get("config_blobs", {})),
            condition_blobs=dict(data.get("condition_blobs", {})),
            worker_hints=dict(data.get("worker_hints", {})),
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


def _root_node_ids(node_ids: Iterable[str], edges: List[GraphEdgeRecord]) -> List[str]:
    in_degree = {node_id: 0 for node_id in node_ids}
    for edge in edges:
        if edge.enabled and edge.target_node_id in in_degree:
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
) -> TrainingGraphPlan:
    raw_nodes = getattr(graph, "nodes", None)
    raw_nodes = raw_nodes if isinstance(raw_nodes, dict) else getattr(graph, "_nodes", {})
    raw_edges = getattr(graph, "edges", None)
    raw_edges = raw_edges if isinstance(raw_edges, list) else getattr(graph, "_edges", [])

    node_records: List[GraphNodeRecord] = []
    node_meta_map = dict(node_metadata or {})
    for node_id, node in raw_nodes.items():
        meta = dict(node_meta_map.get(str(node_id), {}))
        record_meta = dict(meta.get("metadata", {}))
        record_meta.setdefault("description", str(getattr(node, "description", "") or ""))
        record_meta.setdefault("node_class", type(node).__name__)
        node_records.append(
            GraphNodeRecord(
                node_id=str(node_id),
                kind=str(meta.get("kind", type(node).__name__)),
                label=str(meta.get("label", _default_label(str(node_id)))),
                icon=str(meta.get("icon", "")),
                config_id=str(meta.get("config_id", "")),
                group_id=str(meta.get("group_id", "other")),
                enabled=bool(meta.get("enabled", True)),
                metadata=_jsonable(record_meta),
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
        edge_records.append(
            GraphEdgeRecord(
                edge_id=_edge_identity(source_id, target_id, kind, condition_id, pair_counts[pair_key]),
                kind=kind,
                source_node_id=source_id,
                target_node_id=target_id,
                condition_id=condition_id,
                enabled=True,
                metadata=_clean_dict(
                    {
                        "label": str(getattr(edge, "label", "") or ""),
                        "condition_callable": getattr(getattr(edge, "condition", None), "__name__", ""),
                    }
                ),
            )
        )

    serialized_configs = serialize_config_blobs(dict(config_blobs or {}))
    serialized_conditions = serialize_config_blobs(dict(condition_blobs or {}))
    plan_name = str(name or getattr(graph, "name", "Training Graph"))
    plan_digest = _stable_digest(
        {
            "name": plan_name,
            "nodes": [node.to_dict() for node in node_records],
            "edges": [edge.to_dict() for edge in edge_records],
            "configs": serialized_configs,
            "conditions": serialized_conditions,
        }
    )
    plan_id = f"{str(getattr(graph, 'name', 'graph')).strip() or 'graph'}:{plan_digest}"

    plan = TrainingGraphPlan(
        plan_id=plan_id,
        name=plan_name,
        revision=int(revision),
        nodes=node_records,
        edges=edge_records,
        entry_node_ids=_root_node_ids([node.node_id for node in node_records], edge_records),
        config_blobs=serialized_configs,
        condition_blobs=serialized_conditions,
        worker_hints=_jsonable(dict(worker_hints or {})),
        layout=build_layout_from_records(node_records, edge_records),
        metadata=_jsonable(dict(metadata or {})),
    )
    plan.validate()
    return plan
