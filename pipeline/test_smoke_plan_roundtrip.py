"""
Smoke test: graph export → JSON round-trip → rebuild → topology verification.

Run from the toys_to_survive_development directory:
    python pipeline/test_smoke_plan_roundtrip.py

Checks:
  1. build_pipeline_graph() returns a valid stateless PipelineGraph
  2. plan_from_pipeline_graph() serialises it to a TrainingGraphPlan
  3. TrainingGraphPlan.save_json() / .load_json() round-trips without data loss
  4. build_training_graph_from_plan() reconstructs a graph with the same topology
  5. worker_hints capabilities list includes "plan_apply"
  6. _run_from_plan() would construct the expected SimpleNamespace fields
  7. execution-layer node and edge annotations survive export
  8. Mermaid layer rendering supports dense, minimal, and reaction views
  9. edge reaction defaults merge runtime overrides during execution
"""
from __future__ import annotations

import json
import sys
import tempfile
import types
from pathlib import Path

# ── path bootstrap ────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── import check ──────────────────────────────────────────────────────────────
try:
    import torch  # noqa: F401
except ImportError:
    print("SKIP: torch is not available — install it to run the smoke test.")
    sys.exit(0)

from pipeline.orchestrator import (
    build_pipeline_graph,
    build_training_graph_from_plan,
)
from pipeline.nodes.berkeley_classifier_node import BerkeleyClassifierConfig
from pipeline.nodes.transformer_node import TransformerConfig
from pipeline.nodes.generator_node import GeneratorConfig
from pipeline.nodes.wave_classifier_node import WaveClassifierConfig
from pipeline.nodes.vocab_node import VocabConfig
from pipeline.nodes.label_embedding_node import LabelEmbeddingConfig
from pipeline.nodes.data_nodes import (
    WavePoolConfig,
    PregestationDataConfig,
    GestationDataConfig,
    BerkeleyPayloadConfig,
    BerkeleyDataConfig,
)
from pipeline.nodes.gate_nodes import (
    BerkeleyGateConfig,
    TransformerGateConfig,
    GeneratorGateConfig,
    WaveGateConfig,
)
from pipeline.plan_protocol import plan_from_pipeline_graph, TrainingGraphPlan
from pipeline.graph import PipelineGraph, PipelineNode
from pipeline.context import PipelineContext
from pipeline.graph_layers import build_graph_layers, render_mermaid_flowchart, update_readme_flowcharts

# ── helpers ───────────────────────────────────────────────────────────────────

def _ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    sys.exit(1)


def _assert(cond: bool, msg: str) -> None:
    _ok(msg) if cond else _fail(msg)


# ── test: build_pipeline_graph ────────────────────────────────────────────────

def test_build_pipeline_graph():
    print("\n--- test_build_pipeline_graph ---")
    graph = build_pipeline_graph(
        classifier_cfg=BerkeleyClassifierConfig(),
        transformer_cfg=TransformerConfig(),
        generator_cfg=GeneratorConfig(),
        wave_cfg=WaveClassifierConfig(),
        vocab_cfg=VocabConfig(),
        embedding_cfg=LabelEmbeddingConfig(),
        wave_pool_cfg=WavePoolConfig(),
        pregestation_cfg=PregestationDataConfig(),
        gestation_cfg=GestationDataConfig(),
        berkeley_payload_cfg=BerkeleyPayloadConfig(),
        berkeley_data_cfg=BerkeleyDataConfig(),
        berkeley_gate_cfg=BerkeleyGateConfig(),
        transformer_gate_cfg=TransformerGateConfig(),
        generator_gate_cfg=GeneratorGateConfig(),
        wave_gate_cfg=WaveGateConfig(),
    )
    node_count = len(graph.nodes)
    edge_count = len(graph.edges)
    _assert(node_count > 0, f"graph has {node_count} nodes")
    _assert(edge_count > 0, f"graph has {edge_count} edges")
    sequence = graph.build_sequence()
    _assert(len(sequence) == node_count, f"sequence length matches node count ({node_count})")
    return graph, node_count, edge_count


# ── test: plan export ─────────────────────────────────────────────────────────

def test_plan_export(graph, node_count):
    print("\n--- test_plan_export ---")
    plan = plan_from_pipeline_graph(
        graph,
        name="Smoke Test Plan",
        revision=1,
        worker_hints={
            "output_dir": "/tmp/smoke",
            "capabilities": ["plan_apply", "run_control"],
        },
        graph_layers=build_graph_layers(graph),
    )
    _assert(isinstance(plan, TrainingGraphPlan), "plan_from_pipeline_graph returns TrainingGraphPlan")
    _assert(len(plan.nodes) == node_count, f"plan has {len(plan.nodes)} nodes (expected {node_count})")
    _assert(bool(plan.plan_id), "plan has a non-empty plan_id")
    caps = (plan.worker_hints or {}).get("capabilities", [])
    _assert("plan_apply" in caps, "worker_hints.capabilities includes 'plan_apply'")
    _assert("execution" in (plan.graph_layers or {}), "plan graph_layers includes execution layer")
    _assert("inference" in (plan.graph_layers or {}), "plan graph_layers includes inference layer")
    data_node = next((node for node in plan.nodes if node.node_id == "data_node"), None)
    _assert(data_node is not None, "plan includes data_node record")
    _assert(data_node.object_type == "object", "data_node object_type exported")
    _assert(bool(data_node.faculty), "data_node faculty exported")
    _assert(bool(data_node.archetype), "data_node archetype exported")
    data_edge = next(
        (edge for edge in plan.edges if edge.source_node_id == "data_node" and edge.target_node_id == "stage_0_pregestation"),
        None,
    )
    _assert(data_edge is not None, "plan includes data_node -> stage_0_pregestation edge")
    _assert(data_edge.layer == "execution", "execution-layer edge exported")
    _assert(data_edge.reaction_name == "data.provide", "edge reaction name exported")
    _assert(data_edge.target_function == "DataNode.provide_pregestation", "edge target function exported")
    _assert("pregestation_loader" in (data_edge.reaction_defaults or {}).get("resources", []), "edge default resources exported")
    return plan


# ── test: JSON round-trip ─────────────────────────────────────────────────────

def test_json_roundtrip(plan: TrainingGraphPlan, node_count: int, edge_count: int):
    print("\n--- test_json_roundtrip ---")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "plan.json"
        plan.save_json(path)
        _assert(path.exists(), "save_json created file")

        raw = json.loads(path.read_text(encoding="utf-8"))
        _assert("nodes" in raw, "JSON contains 'nodes' key")
        _assert("edges" in raw, "JSON contains 'edges' key")
        _assert(len(raw["nodes"]) == node_count, f"JSON has {len(raw['nodes'])} nodes")
        _assert(len(raw["edges"]) == edge_count, f"JSON has {len(raw['edges'])} edges")

        reloaded = TrainingGraphPlan.load_json(path)
        _assert(reloaded.plan_id == plan.plan_id, "reloaded plan_id matches")
        _assert(len(reloaded.nodes) == node_count, "reloaded node count matches")
        _assert(len(reloaded.edges) == edge_count, "reloaded edge count matches")
    return reloaded


# ── test: rebuild from plan ───────────────────────────────────────────────────

def test_rebuild_from_plan(plan: TrainingGraphPlan, node_count: int, edge_count: int):
    print("\n--- test_rebuild_from_plan ---")
    rebuilt_graph = build_training_graph_from_plan(plan)
    rebuilt_node_count = len(rebuilt_graph.nodes)
    rebuilt_edge_count = len(rebuilt_graph.edges)
    _assert(
        rebuilt_node_count == node_count,
        f"rebuilt graph has {rebuilt_node_count} nodes (expected {node_count})",
    )
    _assert(
        rebuilt_edge_count == edge_count,
        f"rebuilt graph has {rebuilt_edge_count} edges (expected {edge_count})",
    )
    rebuilt_sequence = rebuilt_graph.build_sequence()
    _assert(
        len(rebuilt_sequence) == node_count,
        f"rebuilt sequence length matches ({len(rebuilt_sequence)})",
    )
    return rebuilt_graph


# ── test: _run_from_plan namespace ────────────────────────────────────────────

def test_run_from_plan_namespace(plan: TrainingGraphPlan):
    print("\n--- test_run_from_plan_namespace ---")
    hints = plan.worker_hints or {}
    # Replicate the logic from wav_pipeline_graph._run_from_plan()
    ns = types.SimpleNamespace(
        output_dir=str(hints.get("output_dir", "output")),
        device=str(hints.get("device_preference", "auto")),
        orchestration_cycles=int(hints.get("orchestration_cycles", 1)),
        orchestration_rounds=int(hints.get("orchestration_rounds", 1)),
        cycles=int(hints.get("orchestration_cycles", 1)),
        rounds_per_cycle=int(hints.get("orchestration_rounds", 1)),
    )
    _assert(hasattr(ns, "output_dir"), "namespace has output_dir")
    _assert(hasattr(ns, "device"), "namespace has device")
    _assert(hasattr(ns, "orchestration_cycles"), "namespace has orchestration_cycles")
    _assert(hasattr(ns, "orchestration_rounds"), "namespace has orchestration_rounds")
    _assert(isinstance(ns.orchestration_cycles, int), "orchestration_cycles is int")
    _assert(isinstance(ns.orchestration_rounds, int), "orchestration_rounds is int")


# ── test: plan validation guardrails ──────────────────────────────────────────

def test_plan_validation_guardrails(plan: TrainingGraphPlan):
    print("\n--- test_plan_validation_guardrails ---")
    broken = TrainingGraphPlan.from_dict(plan.to_dict())
    broken.worker_hints = dict(broken.worker_hints or {})
    broken.worker_hints["save_every_n_rounds"] = 0
    try:
        build_training_graph_from_plan(broken)
    except ValueError as exc:
        _assert("save_every_n_rounds" in str(exc), "invalid save_every_n_rounds raises ValueError")
    else:
        _fail("expected ValueError for invalid save_every_n_rounds")




class _DummyNode(PipelineNode):
    def __init__(self, node_id: str):
        self._node_id = node_id

    @property
    def node_id(self) -> str:
        return self._node_id

    def execute(self, ctx) -> None:
        ctx.last_node_statuses[self._node_id] = "ran"


def test_layer_rendering_and_readme_sync(graph):
    print("\n--- test_layer_rendering_and_readme_sync ---")
    layers = build_graph_layers(graph)
    _assert("inference" in layers, "graph layers include inference")
    inference_dense = render_mermaid_flowchart(layers["inference"])
    inference_minimal = render_mermaid_flowchart(layers["inference"], view="minimal")
    inference_reaction = render_mermaid_flowchart(layers["inference"], view="reaction")
    execution_dense = render_mermaid_flowchart(layers["execution"])
    _assert("Input Bus" in inference_dense, "inference dense Mermaid includes Input Bus")
    _assert("Discriminator" in inference_dense, "inference dense Mermaid includes Discriminator")
    _assert("subgraph group_io" in inference_dense, "dense inference view groups nodes by faculty")
    _assert("linkStyle 0" in inference_dense, "dense inference view emits styled edges")
    _assert("ingress wave" not in inference_minimal, "minimal inference view omits edge labels")
    _assert("infer.generate" in inference_reaction, "reaction inference view includes reaction names")
    _assert("classDef faculty_train" in execution_dense, "dense execution view emits faculty class definitions")
    _assert("subgraph group_build" in execution_dense, "dense execution view groups build faculty nodes")
    _assert("Data Authority" in execution_dense, "dense execution view emits readable execution labels")

    with tempfile.TemporaryDirectory() as td:
        readme_path = Path(td) / "README.md"
        readme_path.write_text(
            """# Temp

<!-- BEGIN:GENERATED_EXECUTION_LAYER -->
old
<!-- END:GENERATED_EXECUTION_LAYER -->

<!-- BEGIN:GENERATED_INFERENCE_LAYER -->
old
<!-- END:GENERATED_INFERENCE_LAYER -->
""",
            encoding="utf-8",
        )
        update_readme_flowcharts(readme_path, graph=graph)
        updated = readme_path.read_text(encoding="utf-8")
        _assert("Wave Pool" in updated, "README sync writes execution Mermaid")
        _assert("Input Bus" in updated, "README sync writes inference Mermaid")
        _assert("Dense Infographic" in updated, "README sync writes dense infographic heading")
        _assert("<details>" in updated, "README sync writes alternate collapsible views")
        _assert("Reaction-colored view" in updated, "README sync writes reaction view section")
        _assert("linkStyle 0" in updated, "README sync preserves styled edge output")


def test_edge_reaction_merge():
    print("\n--- test_edge_reaction_merge ---")
    graph = PipelineGraph(name="edge_merge")
    graph.add_node(_DummyNode("source"))
    graph.add_node(_DummyNode("target"))

    def _provide(ctx, *, resource: str = "fallback", mode: str = "default"):
        ctx.label_embedding_info["edge_merge"] = {
            "resource": resource,
            "mode": mode,
        }

    graph.add_edge(
        "source",
        "target",
        on_traverse=_provide,
        target_function="dummy.provide",
        reaction_name="data.provide",
        reaction_defaults={"resource": "alpha", "mode": "default"},
    )

    ctx = PipelineContext()
    edge_id = graph.edges[0].edge_id
    ctx.edge_reaction_overrides[edge_id] = {"mode": "override"}
    statuses = graph.execute_sequence(ctx, verbose=False)
    merged = dict(ctx.label_embedding_info.get("edge_merge", {}))

    _assert(statuses.get("target") == "ran", "target node executed with edge reaction")
    _assert(merged.get("resource") == "alpha", "edge default resource preserved")
    _assert(merged.get("mode") == "override", "edge runtime override merged")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Smoke test: plan round-trip ===")
    graph, node_count, edge_count = test_build_pipeline_graph()
    plan = test_plan_export(graph, node_count)
    reloaded_plan = test_json_roundtrip(plan, node_count, edge_count)
    test_rebuild_from_plan(reloaded_plan, node_count, edge_count)
    test_run_from_plan_namespace(reloaded_plan)
    test_plan_validation_guardrails(reloaded_plan)
    test_layer_rendering_and_readme_sync(graph)
    test_edge_reaction_merge()
    print("\n=== All checks passed ===")


if __name__ == "__main__":
    main()
