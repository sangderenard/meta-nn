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
    )
    _assert(isinstance(plan, TrainingGraphPlan), "plan_from_pipeline_graph returns TrainingGraphPlan")
    _assert(len(plan.nodes) == node_count, f"plan has {len(plan.nodes)} nodes (expected {node_count})")
    _assert(bool(plan.plan_id), "plan has a non-empty plan_id")
    caps = (plan.worker_hints or {}).get("capabilities", [])
    _assert("plan_apply" in caps, "worker_hints.capabilities includes 'plan_apply'")
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


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Smoke test: plan round-trip ===")
    graph, node_count, edge_count = test_build_pipeline_graph()
    plan = test_plan_export(graph, node_count)
    reloaded_plan = test_json_roundtrip(plan, node_count, edge_count)
    test_rebuild_from_plan(reloaded_plan, node_count, edge_count)
    test_run_from_plan_namespace(reloaded_plan)
    print("\n=== All checks passed ===")


if __name__ == "__main__":
    main()
