"""Layered graph views and README flowchart generation helpers.

The executable training graph is only one layer. This module projects that
runtime graph into other graph views so the plan, docs, and later UIs can use a
shared source of truth.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from pipeline.graph import PipelineGraph
from pipeline.plan_protocol import GraphEdgeRecord, GraphNodeRecord, TrainingGraphPlan

README_EXECUTION_START = "<!-- BEGIN:GENERATED_EXECUTION_LAYER -->"
README_EXECUTION_END = "<!-- END:GENERATED_EXECUTION_LAYER -->"
README_INFERENCE_START = "<!-- BEGIN:GENERATED_INFERENCE_LAYER -->"
README_INFERENCE_END = "<!-- END:GENERATED_INFERENCE_LAYER -->"

NODE_STYLE_MAP: Dict[str, Dict[str, str]] = {
    "bootstrap": {"fill": "#E9F1F7", "stroke": "#4B6B88", "color": "#102A43"},
    "build": {"fill": "#F8EFE5", "stroke": "#B07219", "color": "#40210F"},
    "vocab": {"fill": "#FFF7CC", "stroke": "#9A7D0A", "color": "#3D3100"},
    "data": {"fill": "#DFF6F5", "stroke": "#127475", "color": "#053B3C"},
    "train": {"fill": "#FFE8D6", "stroke": "#C05621", "color": "#4A1D05"},
    "gates": {"fill": "#FDE2E4", "stroke": "#C0392B", "color": "#4A0F13"},
    "housekeeping": {"fill": "#E8F5E9", "stroke": "#2E7D32", "color": "#102A12"},
    "io": {"fill": "#DDEBFF", "stroke": "#2563EB", "color": "#0F172A"},
    "inference": {"fill": "#F4F1DE", "stroke": "#3D405B", "color": "#1B1F2A"},
    "buffer": {"fill": "#E0FBFC", "stroke": "#006D77", "color": "#00313A"},
    "gate": {"fill": "#FDE2E4", "stroke": "#C0392B", "color": "#4A0F13"},
    "other": {"fill": "#F3F4F6", "stroke": "#6B7280", "color": "#111827"},
}

EXECUTION_FACULTY_ORDER = [
    "bootstrap",
    "build",
    "vocab",
    "data",
    "train",
    "gates",
    "housekeeping",
    "other",
]

INFERENCE_FACULTY_ORDER = [
    "io",
    "inference",
    "buffer",
    "gate",
    "other",
]


def _layer_node(
    node_id: str,
    label: str,
    *,
    object_type: str = "object",
    faculty: str = "other",
    archetype: str = "node",
    shape: str = "process",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "node_id": str(node_id),
        "label": str(label),
        "object_type": str(object_type),
        "faculty": str(faculty),
        "archetype": str(archetype),
        "shape": str(shape),
        "metadata": dict(metadata or {}),
    }


def _layer_edge(
    source_node_id: str,
    target_node_id: str,
    *,
    label: str = "",
    readme_label: str = "",
    reaction_name: str = "",
    target_function: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "source_node_id": str(source_node_id),
        "target_node_id": str(target_node_id),
        "label": str(label),
        "readme_label": str(readme_label),
        "reaction_name": str(reaction_name),
        "target_function": str(target_function),
        "metadata": dict(metadata or {}),
    }


def build_graph_layers(graph: PipelineGraph) -> Dict[str, Dict[str, Any]]:
    return {
        "execution": _build_execution_layer(graph),
        "inference": _build_inference_layer(),
        "provenance": {
            "layer_id": "provenance",
            "label": "Provenance layer",
            "description": "Reserved layer for ownership, storage residency, and lineage edges.",
            "status": "planned",
            "mermaid_direction": "TD",
            "nodes": [],
            "edges": [],
        },
        "stack_view": {
            "layer_id": "stack_view",
            "label": "Operations stack view",
            "description": "Reserved layer for Nodus-style stack rendering, sub-node ports, and dependency-timed ticks.",
            "status": "planned",
            "mermaid_direction": "LR",
            "nodes": [],
            "edges": [],
        },
    }


def build_execution_layer_from_records(
    nodes: Iterable[GraphNodeRecord],
    edges: Iterable[GraphEdgeRecord],
) -> Dict[str, Any]:
    layer_nodes: List[Dict[str, Any]] = []
    layer_edges: List[Dict[str, Any]] = []

    for node in nodes:
        metadata = dict(node.metadata or {})
        faculty = str(node.faculty or node.group_id or "other")
        layer_nodes.append(
            _layer_node(
                node.node_id,
                _execution_record_label(node),
                object_type=str(node.object_type or "object"),
                faculty=faculty,
                archetype=str(node.archetype or node.kind or "node"),
                shape=str(metadata.get("shape", "process") or "process"),
                metadata={
                    **metadata,
                    "node_id": str(node.node_id),
                    "class_name": str(node.kind or ""),
                    "description": str(metadata.get("description", "") or ""),
                },
            )
        )

    for edge in edges:
        metadata = dict(edge.metadata or {})
        edge_label = str(metadata.get("label", edge.kind) or edge.kind or "")
        readme_label = str(metadata.get("readme_label", edge_label) or edge_label)
        layer_edges.append(
            _layer_edge(
                str(edge.source_node_id),
                str(edge.target_node_id),
                label=edge_label,
                readme_label=readme_label,
                reaction_name=str(edge.reaction_name or ""),
                target_function=str(edge.target_function or ""),
                metadata={
                    **metadata,
                    "edge_id": str(edge.edge_id),
                    "condition_id": str(edge.condition_id or ""),
                    "layer": str(edge.layer or "execution"),
                },
            )
        )

    return {
        "layer_id": "execution",
        "label": "Execution layer",
        "description": "The executable training graph used by the current worker runtime.",
        "status": "active",
        "mermaid_direction": "TD",
        "nodes": layer_nodes,
        "edges": layer_edges,
    }


def build_graph_layers_from_plan(plan: TrainingGraphPlan) -> Dict[str, Dict[str, Any]]:
    layers = dict(getattr(plan, "graph_layers", {}) or {})
    execution_nodes = [
        node
        for node in list(getattr(plan, "nodes", []) or [])
        if bool(getattr(node, "enabled", True))
        and str(getattr(node, "layer", "execution") or "execution") == "execution"
    ]
    execution_edges = [
        edge
        for edge in list(getattr(plan, "edges", []) or [])
        if bool(getattr(edge, "enabled", True))
        and str(getattr(edge, "layer", "execution") or "execution") == "execution"
    ]
    if execution_nodes:
        layers["execution"] = build_execution_layer_from_records(execution_nodes, execution_edges)
    layers.setdefault("inference", _build_inference_layer())
    layers.setdefault(
        "provenance",
        {
            "layer_id": "provenance",
            "label": "Provenance layer",
            "description": "Reserved layer for ownership, storage residency, and lineage edges.",
            "status": "planned",
            "mermaid_direction": "TD",
            "nodes": [],
            "edges": [],
        },
    )
    layers.setdefault(
        "stack_view",
        {
            "layer_id": "stack_view",
            "label": "Operations stack view",
            "description": "Reserved layer for Nodus-style stack rendering, sub-node ports, and dependency-timed ticks.",
            "status": "planned",
            "mermaid_direction": "LR",
            "nodes": [],
            "edges": [],
        },
    )
    return layers


def render_plan_mermaid_flowchart(
    plan: TrainingGraphPlan,
    layer_id: str = "execution",
    *,
    view: str = "dense",
    editable: bool = False,
) -> str:
    layers = build_graph_layers_from_plan(plan)
    if layer_id not in layers:
        raise KeyError(f"Unknown graph layer {layer_id!r}")
    return render_mermaid_flowchart(layers[layer_id], view=view, emit_identity_comments=editable)


def _build_execution_layer(graph: PipelineGraph) -> Dict[str, Any]:
    from pipeline.orchestrator import _node_archetype, _node_faculty

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    for node_id, node in graph.nodes.items():
        shape = getattr(node, "runtime_shape", None)
        if callable(shape):
            shape = shape()
        object_type = str(getattr(shape, "object_type", "") or "").strip() or "object"
        faculty = str(getattr(shape, "faculty", "") or "").strip()
        if not faculty or faculty == "other":
            faculty = _node_faculty(node_id)
        archetype = str(getattr(shape, "archetype", "") or "").strip()
        if not archetype or archetype == type(node).__name__:
            archetype = _node_archetype(node_id) or type(node).__name__
        nodes.append(
            _layer_node(
                node_id,
                _execution_node_label(node_id, node),
                object_type=object_type,
                faculty=faculty or "other",
                archetype=archetype,
                metadata={
                    "node_id": str(node_id),
                    "class_name": type(node).__name__,
                    "description": str(getattr(node, "description", "") or ""),
                },
            )
        )

    for edge in graph.edges:
        reaction = getattr(edge, "reaction", None)
        label = str(getattr(edge, "label", "") or "")
        edges.append(
            _layer_edge(
                str(edge.source_id),
                str(edge.target_id),
                label=label,
                readme_label=label,
                reaction_name=str(getattr(reaction, "reaction_name", "") or ""),
                target_function=str(getattr(reaction, "target_function", "") or ""),
                metadata={
                    "condition_id": str(getattr(edge, "condition_id", "") or ""),
                    "layer": str(getattr(edge, "layer", "execution") or "execution"),
                },
            )
        )

    return {
        "layer_id": "execution",
        "label": "Execution layer",
        "description": "The executable training graph used by the current worker runtime.",
        "status": "active",
        "mermaid_direction": "TD",
        "nodes": nodes,
        "edges": edges,
    }


def _build_inference_layer() -> Dict[str, Any]:
    nodes = [
        _layer_node("input_bus", "Input Bus", object_type="bus", faculty="io", archetype="input_bus", shape="bus"),
        _layer_node("wave_classifier_runtime", "Wave Classifier", faculty="inference", archetype="wave_classifier"),
        _layer_node("transformer_runtime", "Transformer", faculty="inference", archetype="transformer"),
        _layer_node("wave_sidecar_extract", "Wave Sidecar Extract", faculty="buffer", archetype="wave_sidecar_extractor"),
        _layer_node("classifier_runtime", "Classifier", faculty="inference", archetype="classifier"),
        _layer_node("generator_runtime", "GAN Generator", faculty="inference", archetype="generator"),
        _layer_node("discriminator_runtime", "Discriminator", faculty="gate", archetype="discriminator"),
        _layer_node("wave_repack", "Wave Repack", faculty="buffer", archetype="wave_repacker"),
        _layer_node("output_bus", "Output Bus", object_type="bus", faculty="io", archetype="output_bus", shape="bus"),
    ]
    edges = [
        _layer_edge("input_bus", "wave_classifier_runtime", label="ingress", readme_label="ingress wave", reaction_name="bus.read", target_function="input_bus.read"),
        _layer_edge("wave_classifier_runtime", "transformer_runtime", label="classify", readme_label="bitplane window", reaction_name="infer.classify_wave", target_function="wave_classifier.infer"),
        _layer_edge("transformer_runtime", "wave_sidecar_extract", label="extract", readme_label="carrier sidecar", reaction_name="wave.extract_sidecar", target_function="transformer.extract_wave_sidecar"),
        _layer_edge("transformer_runtime", "classifier_runtime", label="interpret", readme_label="transformed bitplane", reaction_name="infer.transform", target_function="transformer.infer"),
        _layer_edge("classifier_runtime", "generator_runtime", label="condition", readme_label="semantic condition", reaction_name="infer.condition", target_function="classifier.interpret"),
        _layer_edge("generator_runtime", "discriminator_runtime", label="candidate", readme_label="candidate synthesis", reaction_name="infer.generate", target_function="generator.infer"),
        _layer_edge("discriminator_runtime", "generator_runtime", label="retry", readme_label="reject and retry", reaction_name="infer.retry_loop", target_function="discriminator.reject_loop", metadata={"loop_until": "accepted"}),
        _layer_edge("discriminator_runtime", "wave_repack", label="accept", readme_label="accepted candidate", reaction_name="infer.accept", target_function="discriminator.accept"),
        _layer_edge("wave_sidecar_extract", "wave_repack", label="preserve", readme_label="preserved wave surround", reaction_name="wave.sidecar_attach", target_function="wave_repack.attach_sidecar"),
        _layer_edge("transformer_runtime", "wave_repack", label="repack", readme_label="bitplane pre/post transform", reaction_name="wave.repack", target_function="wave_repack.attach_bitplane"),
        _layer_edge("wave_repack", "output_bus", label="egress", readme_label="egress wave", reaction_name="bus.write", target_function="output_bus.write"),
    ]
    return {
        "layer_id": "inference",
        "label": "Inference layer",
        "description": "Runtime inference flow from input bus through model objects and back to an output bus.",
        "status": "planned",
        "activation_rule": "default when all training criteria pass, but representable as an always-open runtime mode",
        "mermaid_direction": "LR",
        "nodes": nodes,
        "edges": edges,
    }


def render_mermaid_flowchart(
    layer: Dict[str, Any],
    *,
    view: str = "dense",
    emit_identity_comments: bool = False,
) -> str:
    layer_id = str(layer.get("layer_id", "") or "execution")
    direction = str(layer.get("mermaid_direction", "TD") or "TD")
    nodes = list(layer.get("nodes", []) or [])
    edges = list(layer.get("edges", []) or [])
    alias_map: Dict[str, str] = {}
    used_aliases: set[str] = set()
    lines: List[str] = [
        "%%{init: {'theme':'base','flowchart':{'curve':'basis','htmlLabels':true}}}%%",
        f"flowchart {direction}",
    ]

    for ordinal, node in enumerate(nodes, start=1):
        node_id = str(node.get("node_id", f"node_{ordinal}"))
        alias_map[node_id] = _mermaid_alias(node_id, ordinal=ordinal, used=used_aliases)

    if view == "dense":
        lines.extend(_render_dense_nodes(layer_id, nodes, alias_map, emit_identity_comments=emit_identity_comments))
    else:
        lines.extend(_render_plain_nodes(nodes, alias_map, emit_identity_comments=emit_identity_comments))

    if nodes and edges:
        lines.append("")

    for edge in edges:
        source_id = str(edge.get("source_node_id", ""))
        target_id = str(edge.get("target_node_id", ""))
        if source_id not in alias_map or target_id not in alias_map:
            continue
        edge_id = str(dict(edge.get("metadata", {}) or {}).get("edge_id", "") or "").strip()
        if emit_identity_comments and edge_id:
            lines.append(f"    %% edge_id:{edge_id}")
        edge_label = _edge_label(edge, view=view)
        if edge_label:
            lines.append(
                f'    {alias_map[source_id]} -- "{_escape_mermaid(edge_label)}" --> {alias_map[target_id]}'
            )
        else:
            lines.append(f"    {alias_map[source_id]} --> {alias_map[target_id]}")

    if view == "dense":
        lines.append("")
        lines.extend(_render_class_definitions())
        lines.extend(_render_node_classes(nodes, alias_map))
        lines.extend(_render_edge_styles(layer_id, edges))
    elif view == "reaction":
        lines.append("")
        lines.extend(_render_edge_styles(layer_id, edges))

    return "\n".join(lines)


def render_readme_layer_section(layer: Dict[str, Any]) -> str:
    title = str(layer.get("label", "Layer"))
    dense_chart = _mermaid_code_block(render_mermaid_flowchart(layer, view="dense"))
    minimal_chart = _mermaid_code_block(render_mermaid_flowchart(layer, view="minimal"))
    reaction_chart = _mermaid_code_block(render_mermaid_flowchart(layer, view="reaction"))
    legend = _render_layer_legend(layer)

    return "\n\n".join(
        [
            f"**{title} Dense Infographic**",
            dense_chart,
            legend,
            "<details>",
            "<summary>Minimal schematic view</summary>",
            "",
            minimal_chart,
            "</details>",
            "<details>",
            "<summary>Reaction-colored view</summary>",
            "",
            reaction_chart,
            "</details>",
        ]
    )


def update_readme_flowcharts(readme_path: Path, *, graph: Optional[PipelineGraph] = None) -> Dict[str, str]:
    graph = graph if graph is not None else build_default_graph()
    layers = build_graph_layers(graph)
    execution_block = render_readme_layer_section(layers["execution"])
    inference_block = render_readme_layer_section(layers["inference"])

    text = readme_path.read_text(encoding="utf-8")
    text = _replace_marked_block(text, README_EXECUTION_START, README_EXECUTION_END, execution_block)
    text = _replace_marked_block(text, README_INFERENCE_START, README_INFERENCE_END, inference_block)
    readme_path.write_text(text, encoding="utf-8")
    return {
        "execution": execution_block,
        "inference": inference_block,
    }


def parse_mermaid_flowchart(body: str) -> Dict[str, Any]:
    direction = "TD"
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    alias_to_node_id: Dict[str, str] = {}
    current_faculty = "other"
    pending_node_id = ""
    pending_edge_id = ""

    for raw_line in str(body or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("%%{"):
            continue
        if line.startswith("%% node_id:"):
            pending_node_id = line.split(":", 1)[1].strip()
            continue
        if line.startswith("%% edge_id:"):
            pending_edge_id = line.split(":", 1)[1].strip()
            continue
        if line.startswith("flowchart "):
            direction = line.split(None, 1)[1].strip() or "TD"
            continue
        if line.startswith("subgraph "):
            match = re.match(r"subgraph\s+([^\[]+)(?:\[(.*)\])?$", line)
            token = str(match.group(1) if match else "other").strip()
            current_faculty = token[len("group_"):] if token.startswith("group_") else _sanitize_mermaid_token(token or "other")
            continue
        if line == "end":
            current_faculty = "other"
            continue
        if line.startswith("classDef ") or line.startswith("class ") or line.startswith("linkStyle "):
            continue

        edge_match = re.match(
            r'(?P<source>[A-Za-z_][A-Za-z0-9_]*)\s+--\s+"(?P<label>.*?)"\s+-->\s+(?P<target>[A-Za-z_][A-Za-z0-9_]*)$',
            line,
        )
        if edge_match is None:
            edge_match = re.match(
                r'(?P<source>[A-Za-z_][A-Za-z0-9_]*)\s+-->\s+(?P<target>[A-Za-z_][A-Za-z0-9_]*)$',
                line,
            )
        if edge_match is not None:
            source_alias = str(edge_match.group("source"))
            target_alias = str(edge_match.group("target"))
            label = _base_edge_label(str(edge_match.groupdict().get("label", "") or ""))
            edges.append(
                {
                    "edge_id": pending_edge_id,
                    "source_node_id": alias_to_node_id.get(source_alias, source_alias),
                    "target_node_id": alias_to_node_id.get(target_alias, target_alias),
                    "label": label,
                    "readme_label": label,
                    "metadata": {"edge_id": pending_edge_id} if pending_edge_id else {},
                }
            )
            pending_edge_id = ""
            continue

        node_match = re.match(r'(?P<alias>[A-Za-z_][A-Za-z0-9_]*)\[\["(?P<label>.*)"\]\]$', line)
        shape = "bus"
        if node_match is None:
            node_match = re.match(r'(?P<alias>[A-Za-z_][A-Za-z0-9_]*)\["(?P<label>.*)"\]$', line)
            shape = "process"
        if node_match is None:
            continue

        alias = str(node_match.group("alias"))
        node_id = str(pending_node_id or alias_to_node_id.get(alias) or alias)
        alias_to_node_id[alias] = node_id
        nodes.append(
            {
                "node_id": node_id,
                "label": _base_node_label(str(node_match.group("label") or node_id)),
                "faculty": current_faculty or "other",
                "shape": shape,
                "metadata": {"node_id": node_id},
            }
        )
        pending_node_id = ""

    return {
        "layer_id": "execution",
        "mermaid_direction": direction,
        "nodes": nodes,
        "edges": edges,
    }


def apply_mermaid_execution_edit(plan: TrainingGraphPlan, mermaid_text: str) -> TrainingGraphPlan:
    parsed = parse_mermaid_flowchart(mermaid_text)
    original = TrainingGraphPlan.from_dict(plan.to_dict())
    exec_nodes = [
        node
        for node in list(original.nodes or [])
        if str(getattr(node, "layer", "execution") or "execution") == "execution"
    ]
    exec_edges = [
        edge
        for edge in list(original.edges or [])
        if str(getattr(edge, "layer", "execution") or "execution") == "execution"
    ]
    other_nodes = [node for node in list(original.nodes or []) if node not in exec_nodes]
    other_edges = [edge for edge in list(original.edges or []) if edge not in exec_edges]
    node_map = {node.node_id: node for node in exec_nodes}
    edge_map = {edge.edge_id: edge for edge in exec_edges}

    updated_nodes: List[GraphNodeRecord] = []
    active_node_ids: List[str] = []
    for parsed_node in list(parsed.get("nodes", []) or []):
        node_id = str(parsed_node.get("node_id", "") or "").strip()
        if node_id not in node_map:
            raise ValueError(f"Editable Mermaid references unknown execution node {node_id!r}")
        base = node_map[node_id]
        metadata = dict(base.metadata or {})
        metadata["shape"] = str(parsed_node.get("shape", metadata.get("shape", "process")) or metadata.get("shape", "process"))
        faculty = str(parsed_node.get("faculty", base.faculty or base.group_id or "other") or base.faculty or base.group_id or "other")
        updated_nodes.append(
            GraphNodeRecord(
                node_id=base.node_id,
                kind=base.kind,
                label=_base_node_label(str(parsed_node.get("label", base.label) or base.label)),
                icon=base.icon,
                config_id=base.config_id,
                group_id=faculty,
                object_type=base.object_type,
                faculty=faculty,
                archetype=base.archetype,
                layer=base.layer,
                enabled=True,
                metadata=metadata,
            )
        )
        active_node_ids.append(base.node_id)

    updated_edges: List[GraphEdgeRecord] = []
    for parsed_edge in list(parsed.get("edges", []) or []):
        edge_id = str(parsed_edge.get("edge_id", "") or dict(parsed_edge.get("metadata", {}) or {}).get("edge_id", "") or "").strip()
        if edge_id not in edge_map:
            raise ValueError(
                "Editable Mermaid must preserve %% edge_id comments for execution edges; "
                f"could not resolve {edge_id or '<missing>'!r}"
            )
        source_id = str(parsed_edge.get("source_node_id", "") or "").strip()
        target_id = str(parsed_edge.get("target_node_id", "") or "").strip()
        if source_id not in active_node_ids or target_id not in active_node_ids:
            raise ValueError(f"Edge {edge_id!r} references a node removed from the execution layer")
        base = edge_map[edge_id]
        metadata = dict(base.metadata or {})
        edge_label = _base_edge_label(str(parsed_edge.get("label", metadata.get("label", base.kind)) or metadata.get("label", base.kind) or base.kind))
        metadata["label"] = edge_label
        metadata["readme_label"] = edge_label
        updated_edges.append(
            GraphEdgeRecord(
                edge_id=base.edge_id,
                kind=edge_label or base.kind,
                source_node_id=source_id,
                target_node_id=target_id,
                condition_id=base.condition_id,
                layer=base.layer,
                target_function=base.target_function,
                reaction_name=base.reaction_name,
                reaction_defaults=dict(base.reaction_defaults or {}),
                enabled=True,
                metadata=metadata,
            )
        )

    original.nodes = other_nodes + updated_nodes
    original.edges = other_edges + updated_edges
    original.entry_node_ids = _root_node_ids_from_records(original.nodes, original.edges)
    layers = dict(original.graph_layers or {})
    layers["execution"] = build_execution_layer_from_records(updated_nodes, updated_edges)
    original.graph_layers = layers
    original.revision = int(getattr(original, "revision", 0) or 0) + 1
    original.validate()
    return original


def build_default_graph() -> PipelineGraph:
    from pipeline.orchestrator import _build_configs_from_args, build_pipeline_graph

    class _FakeArgs:
        def __getattr__(self, _name):
            return None

    args = _FakeArgs()
    cfg = _build_configs_from_args(args)
    return build_pipeline_graph(
        classifier_cfg=cfg["classifier"],
        transformer_cfg=cfg["transformer"],
        generator_cfg=cfg["generator"],
        wave_cfg=cfg["wave"],
        vocab_cfg=cfg["vocab"],
        embedding_cfg=cfg["embedding"],
        wave_pool_cfg=cfg["wave_pool"],
        pregestation_cfg=cfg["pregestation"],
        gestation_cfg=cfg["gestation"],
        berkeley_payload_cfg=cfg["berkeley_payload"],
        berkeley_data_cfg=cfg["berkeley_data"],
        berkeley_gate_cfg=cfg["berkeley_gate"],
        transformer_gate_cfg=cfg["transformer_gate"],
        generator_gate_cfg=cfg["generator_gate"],
        wave_gate_cfg=cfg["wave_gate"],
        save_every_n_rounds=1,
        berkeley_refresh_every_n_rounds=4,
    )


def _render_dense_nodes(
    layer_id: str,
    nodes: List[Dict[str, Any]],
    alias_map: Dict[str, str],
    *,
    emit_identity_comments: bool = False,
) -> List[str]:
    lines: List[str] = []
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for node in nodes:
        faculty = str(node.get("faculty", "other") or "other")
        grouped.setdefault(faculty, []).append(node)

    for faculty in _faculty_order(layer_id, grouped.keys()):
        lane_nodes = grouped.get(faculty, [])
        if not lane_nodes:
            continue
        lines.append(f"    subgraph group_{_sanitize_mermaid_token(faculty)}[{_escape_mermaid(_faculty_label(faculty))}]")
        for node in lane_nodes:
            label = _dense_node_label(node)
            if emit_identity_comments:
                lines.append(f"        %% node_id:{str(node.get('node_id', 'node'))}")
            lines.append(f"        {alias_map[str(node['node_id'])]}{_mermaid_node_shape(label, shape=str(node.get('shape', 'process')))}")
        lines.append("    end")
    return lines


def _render_plain_nodes(
    nodes: List[Dict[str, Any]],
    alias_map: Dict[str, str],
    *,
    emit_identity_comments: bool = False,
) -> List[str]:
    lines: List[str] = []
    for node in nodes:
        node_id = str(node.get("node_id", "node"))
        if emit_identity_comments:
            lines.append(f"    %% node_id:{node_id}")
        lines.append(f"    {alias_map[node_id]}{_mermaid_node_shape(str(node.get('label', node_id)), shape=str(node.get('shape', 'process')))}")
    return lines


def _edge_label(edge: Dict[str, Any], *, view: str) -> str:
    base = str(edge.get("readme_label", "") or edge.get("label", "") or "").strip()
    if view == "minimal":
        return ""
    if view == "reaction":
        reaction_name = str(edge.get("reaction_name", "") or "").strip()
        if base and reaction_name:
            return f"{base} | {reaction_name}"
        return reaction_name or base
    return base


def _render_class_definitions() -> List[str]:
    lines: List[str] = []
    for faculty, style in NODE_STYLE_MAP.items():
        lines.append(
            f"    classDef faculty_{_sanitize_mermaid_token(faculty)} fill:{style['fill']},stroke:{style['stroke']},color:{style['color']},stroke-width:2px;"
        )
    return lines


def _render_node_classes(nodes: List[Dict[str, Any]], alias_map: Dict[str, str]) -> List[str]:
    grouped_aliases: Dict[str, List[str]] = {}
    for node in nodes:
        faculty = str(node.get("faculty", "other") or "other")
        grouped_aliases.setdefault(faculty, []).append(alias_map[str(node["node_id"])])

    lines: List[str] = []
    for faculty, aliases in grouped_aliases.items():
        lines.append(f"    class {','.join(aliases)} faculty_{_sanitize_mermaid_token(faculty)};")
    return lines


def _render_edge_styles(layer_id: str, edges: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for index, edge in enumerate(edges):
        style = _edge_style(layer_id, edge)
        lines.append(
            f"    linkStyle {index} stroke:{style['stroke']},stroke-width:{style['width']},opacity:{style['opacity']},stroke-dasharray:{style['dasharray']};"
        )
    return lines


def _render_layer_legend(layer: Dict[str, Any]) -> str:
    layer_id = str(layer.get("layer_id", "execution") or "execution")
    if layer_id == "inference":
        return "Node colors group buses, inference objects, buffers, and acceptance gates. Edge colors separate ingress, interpretation, synthesis, retry-loop, repack, and egress paths."
    return "Node colors group bootstrap, build, vocab, data, train, gate, and housekeeping faculties. Edge colors separate startup, per-round, data-provision, gated progression, and end-of-round reactions."


def _faculty_order(layer_id: str, faculties: Iterable[str]) -> List[str]:
    known = INFERENCE_FACULTY_ORDER if layer_id == "inference" else EXECUTION_FACULTY_ORDER
    out = [faculty for faculty in known if faculty in faculties]
    for faculty in faculties:
        if faculty not in out:
            out.append(faculty)
    return out


def _faculty_label(faculty: str) -> str:
    return faculty.replace("_", " ").title()


def _execution_record_label(node: GraphNodeRecord) -> str:
    special = {
        "data_node": "Data Authority",
        "stage_c_lora": "LoRA Round",
    }
    if node.node_id in special:
        return special[node.node_id]
    return str(node.label or _faculty_label(str(node.node_id).replace("_", " "))).strip() or str(node.node_id)


def _execution_node_label(node_id: str, node: Any) -> str:
    special = {
        "data_node": "Data Authority",
        "stage_c_lora": "LoRA Round",
    }
    if node_id in special:
        return special[node_id]

    class_name = type(node).__name__
    if class_name.endswith("Node"):
        class_name = class_name[:-4]
    class_name = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", class_name)
    class_name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", class_name)
    return class_name.strip() or node_id.replace("_", " ").title()


def _dense_node_label(node: Dict[str, Any]) -> str:
    base = str(node.get("label", node.get("node_id", "node"))).strip()
    object_type = str(node.get("object_type", "object") or "object").strip()
    faculty = str(node.get("faculty", "other") or "other").strip()
    archetype = str(node.get("archetype", "node") or "node").strip()
    detail = " / ".join(part for part in [object_type, faculty, archetype] if part)
    if not detail:
        return base
    return f"{base}<br/>{detail}"


def _edge_style(layer_id: str, edge: Dict[str, Any]) -> Dict[str, str]:
    label = str(edge.get("label", "") or "").strip().lower()
    readme_label = str(edge.get("readme_label", "") or "").strip().lower()
    if layer_id == "inference":
        if label == "retry":
            return {"stroke": "#D62828", "width": "4px", "opacity": "0.95", "dasharray": "8 4"}
        if label == "ingress":
            return {"stroke": "#0077B6", "width": "3px", "opacity": "0.9", "dasharray": "0"}
        if label == "classify":
            return {"stroke": "#F4A261", "width": "3px", "opacity": "0.9", "dasharray": "0"}
        if label == "extract":
            return {"stroke": "#2A9D8F", "width": "3px", "opacity": "0.9", "dasharray": "2 2"}
        if label == "interpret":
            return {"stroke": "#E9C46A", "width": "3px", "opacity": "0.9", "dasharray": "0"}
        if label == "condition":
            return {"stroke": "#264653", "width": "3px", "opacity": "0.9", "dasharray": "0"}
        if label == "candidate":
            return {"stroke": "#E76F51", "width": "3px", "opacity": "0.9", "dasharray": "0"}
        if label == "accept":
            return {"stroke": "#2B9348", "width": "3px", "opacity": "0.95", "dasharray": "0"}
        if label == "preserve":
            return {"stroke": "#577590", "width": "3px", "opacity": "0.9", "dasharray": "4 2"}
        if label == "repack":
            return {"stroke": "#7F5539", "width": "3px", "opacity": "0.9", "dasharray": "0"}
        if label == "egress":
            return {"stroke": "#3A86FF", "width": "3px", "opacity": "0.95", "dasharray": "0"}
        if "wave" in readme_label:
            return {"stroke": "#0077B6", "width": "3px", "opacity": "0.9", "dasharray": "0"}
        return {"stroke": "#6B7280", "width": "2px", "opacity": "0.75", "dasharray": "0"}

    if label.startswith("provides:"):
        return {"stroke": "#1D70A2", "width": "3px", "opacity": "0.95", "dasharray": "0"}
    if label == "startup":
        return {"stroke": "#2A9D8F", "width": "3px", "opacity": "0.9", "dasharray": "0"}
    if label == "per_round":
        return {"stroke": "#E9C46A", "width": "3px", "opacity": "0.9", "dasharray": "2 2"}
    if label == "after_gate1":
        return {"stroke": "#F4A261", "width": "3px", "opacity": "0.9", "dasharray": "0"}
    if label == "after_all_gates":
        return {"stroke": "#E76F51", "width": "3px", "opacity": "0.95", "dasharray": "0"}
    if label == "after_transformer_gate":
        return {"stroke": "#B56576", "width": "3px", "opacity": "0.9", "dasharray": "0"}
    if label == "end_of_round":
        return {"stroke": "#2B9348", "width": "3px", "opacity": "0.9", "dasharray": "4 2"}
    if label == "if_gan_mode":
        return {"stroke": "#577590", "width": "3px", "opacity": "0.85", "dasharray": "8 3"}
    return {"stroke": "#6B7280", "width": "2px", "opacity": "0.75", "dasharray": "0"}


def _base_node_label(label: str) -> str:
    return str(label or "").split("<br/>", 1)[0].strip()


def _base_edge_label(label: str) -> str:
    return str(label or "").split(" | ", 1)[0].strip()


def _root_node_ids_from_records(
    nodes: Iterable[GraphNodeRecord],
    edges: Iterable[GraphEdgeRecord],
) -> List[str]:
    node_ids = [str(node.node_id) for node in nodes]
    enabled_edges = [edge for edge in edges if bool(getattr(edge, "enabled", True))]
    in_degree = {node_id: 0 for node_id in node_ids}
    for edge in enabled_edges:
        target_id = str(edge.target_node_id)
        if target_id in in_degree:
            in_degree[target_id] += 1
    return sorted(node_id for node_id, degree in in_degree.items() if degree == 0)


def _replace_marked_block(text: str, start_marker: str, end_marker: str, body: str) -> str:
    start = text.find(start_marker)
    end = text.find(end_marker)
    if start < 0 or end < 0 or end < start:
        raise ValueError(f"README markers missing or misordered: {start_marker} / {end_marker}")
    start_body = start + len(start_marker)
    return text[:start_body] + "\n" + body + "\n" + text[end:]


def _mermaid_code_block(body: str) -> str:
    return f"```mermaid\n{body}\n```"


def _mermaid_alias(node_id: str, *, ordinal: int, used: set[str]) -> str:
    base = _sanitize_mermaid_token(str(node_id or f"node_{ordinal}")) or f"node_{ordinal}"
    alias = base
    suffix = 2
    while alias in used:
        alias = f"{base}_{suffix}"
        suffix += 1
    used.add(alias)
    return alias


def _sanitize_mermaid_token(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(text or "")).strip("_")


def _mermaid_node_shape(label: str, *, shape: str = "process") -> str:
    escaped = _escape_mermaid(label)
    if shape == "bus":
        return f'[["{escaped}"]]'
    return f'["{escaped}"]'


def _escape_mermaid(text: str) -> str:
    return str(text or "").replace('"', "'")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate layered Mermaid flowcharts from the graph definitions.")
    parser.add_argument("--layer", choices=["execution", "inference"], default="", help="Print a single layer Mermaid flowchart.")
    parser.add_argument("--view", choices=["dense", "minimal", "reaction"], default="dense", help="Rendering mode for --layer output.")
    parser.add_argument("--write-readme", action="store_true", help="Update README.md generated Mermaid sections in place.")
    parser.add_argument("--readme-path", default="README.md", help="README path to update when --write-readme is set.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    graph = build_default_graph()
    layers = build_graph_layers(graph)

    if args.layer:
        print(render_mermaid_flowchart(layers[args.layer], view=args.view))

    if args.write_readme:
        readme_path = Path(str(args.readme_path))
        update_readme_flowcharts(readme_path, graph=graph)
        print(f"[graph-layers] README flowcharts updated: {readme_path}")

    if not args.layer and not args.write_readme:
        parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
