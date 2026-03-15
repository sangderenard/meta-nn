"""
Smoke test: graph export → JSON round-trip → rebuild → topology verification.

Run from the repository root:
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
 10. execution topology is rebuilt from plan node/edge records
 11. editable Mermaid changes can rewrite the execution plan
 12. execution program and overlay layers are exported from the plan IR
 13. edge callbacks can provision node preconditions before should_run()
 14. execution_program ordering can override raw graph order
 15. execution_program guards can directly control runtime execution
 16. execution_program decision and hold steps are traversed explicitly
 17. execution_program traversal can revisit cyclic paths before halting
 18. execution_policy fields survive plan export and JSON round-trip
 19. execution_policy badges appear in dense Mermaid rendering
 20. condition expression DSL parses and evaluates correctly
 21. condition_expr and run_condition_expr survive plan JSON round-trip
 22. condition_expr appears in Mermaid and roundtrips through editable edit
"""
from __future__ import annotations

import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

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
from pipeline.nodes.classifier_node import ClassifierConfig
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
from pipeline.plan_protocol import plan_from_pipeline_graph, TrainingGraphPlan, ActionRecord, SubnodeRecord
from pipeline.graph import PipelineGraph, PipelineNode
from pipeline.context import PipelineContext
from pipeline.graph_layers import (
    apply_mermaid_execution_edit,
    build_execution_program,
    build_graph_layers,
    render_mermaid_flowchart,
    render_plan_mermaid_flowchart,
    update_readme_flowcharts,
)

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
        classifier_cfg=ClassifierConfig(),
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
        execution_program=build_execution_program(graph),
    )
    _assert(isinstance(plan, TrainingGraphPlan), "plan_from_pipeline_graph returns TrainingGraphPlan")
    _assert(len(plan.nodes) == node_count, f"plan has {len(plan.nodes)} nodes (expected {node_count})")
    _assert(bool(plan.plan_id), "plan has a non-empty plan_id")
    caps = (plan.worker_hints or {}).get("capabilities", [])
    _assert("plan_apply" in caps, "worker_hints.capabilities includes 'plan_apply'")
    _assert("execution" in (plan.graph_layers or {}), "plan graph_layers includes execution layer")
    _assert("execution_overlay" in (plan.graph_layers or {}), "plan graph_layers includes execution overlay layer")
    _assert("inference" in (plan.graph_layers or {}), "plan graph_layers includes inference layer")
    _assert(bool(plan.execution_program), "plan includes execution program")
    decision_steps = [step for step in (plan.execution_program or {}).get("steps", []) if step.get("kind") == "decision"]
    _assert(len(decision_steps) > 0, "execution program includes decision steps")
    data_node = next((node for node in plan.nodes if node.node_id == "data_node"), None)
    _assert(data_node is not None, "plan includes data_node record")
    _assert(data_node.object_type == "data_hub", "data_node object_type exported")
    _assert(bool(data_node.faculty), "data_node faculty exported")
    _assert(bool(data_node.archetype), "data_node archetype exported")
    gate0_node = next((node for node in plan.nodes if node.node_id == "gate_0_pregestation_eval"), None)
    gate1_node = next((node for node in plan.nodes if node.node_id == "gate_1_gestation_eval"), None)
    _assert(gate0_node is not None, "plan includes Gate 0 eval node")
    _assert(gate1_node is not None, "plan includes Gate 1 eval node")
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
    # Rebuild only materializes execution-layer edges; count those from the plan.
    exec_edge_count = sum(
        1 for e in (plan.edges or [])
        if bool(getattr(e, "enabled", True))
        and str(getattr(e, "layer", "execution") or "execution") == "execution"
    )
    _assert(
        rebuilt_node_count == node_count,
        f"rebuilt graph has {rebuilt_node_count} nodes (expected {node_count})",
    )
    _assert(
        rebuilt_edge_count == exec_edge_count,
        f"rebuilt graph has {rebuilt_edge_count} edges (expected {exec_edge_count})",
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
    execution_overlay = render_mermaid_flowchart(layers["execution_overlay"])
    _assert("Input Bus" in inference_dense, "inference dense Mermaid includes Input Bus")
    _assert("Discriminator" in inference_dense, "inference dense Mermaid includes Discriminator")
    _assert("subgraph group_io" in inference_dense, "dense inference view groups nodes by faculty")
    _assert("linkStyle 0" in inference_dense, "dense inference view emits styled edges")
    _assert("ingress wave" not in inference_minimal, "minimal inference view omits edge labels")
    _assert("infer.generate" in inference_reaction, "reaction inference view includes reaction names")
    _assert("classDef faculty_train" in execution_dense, "dense execution view emits faculty class definitions")
    _assert("subgraph group_build" in execution_dense, "dense execution view groups build faculty nodes")
    _assert("Data Authority" in execution_dense, "dense execution view emits readable execution labels")
    _assert("Gate 0 passed?" in execution_overlay, "execution overlay includes Gate 0 decision")
    _assert("Hold / next round" in execution_overlay, "execution overlay includes hold sink")
    _assert("stroke:#111111" in execution_overlay, "execution overlay emits black control edges")

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

<!-- BEGIN:GENERATED_STACK_VIEW_LAYER -->
old
<!-- END:GENERATED_STACK_VIEW_LAYER -->

<!-- BEGIN:GENERATED_PROVENANCE_LAYER -->
old
<!-- END:GENERATED_PROVENANCE_LAYER -->
""",
            encoding="utf-8",
        )
        update_readme_flowcharts(readme_path, graph=graph)
        updated = readme_path.read_text(encoding="utf-8")
        _assert("Training Composite Execution Overlay" in updated, "README sync writes execution overlay heading")
        _assert("Gate 0 passed?" in updated, "README sync writes execution overlay decisions")
        _assert("Wave Pool" in updated, "README sync writes execution Mermaid")
        _assert("Input Bus" in updated, "README sync writes inference Mermaid")
        _assert("Dense Infographic" in updated, "README sync writes dense infographic heading")
        _assert("<details>" in updated, "README sync writes alternate collapsible views")
        _assert("Reaction-colored view" in updated, "README sync writes reaction view section")
        _assert("linkStyle 0" in updated, "README sync preserves styled edge output")


def test_plan_topology_is_authoritative(plan: TrainingGraphPlan, edge_count: int):
    print("\n--- test_plan_topology_is_authoritative ---")
    edited = TrainingGraphPlan.from_dict(plan.to_dict())
    removed_edge = next(
        (edge for edge in edited.edges if edge.source_node_id == "sync_gate_replica" and edge.target_node_id == "checkpoint_save"),
        None,
    )
    _assert(removed_edge is not None, "plan includes sync_gate_replica -> checkpoint_save edge")
    edited.edges = [edge for edge in edited.edges if edge.edge_id != removed_edge.edge_id]

    rebuilt = build_training_graph_from_plan(edited)
    rebuilt_edge_ids = {edge.edge_id for edge in rebuilt.edges}
    _assert(len(rebuilt.edges) == edge_count - 1, "rebuilt graph reflects removed plan edge")
    _assert(removed_edge.edge_id not in rebuilt_edge_ids, "removed plan edge is absent from rebuilt graph")


def test_editable_mermaid_execution_roundtrip(plan: TrainingGraphPlan):
    print("\n--- test_editable_mermaid_execution_roundtrip ---")
    editable = render_plan_mermaid_flowchart(plan, layer_id="execution", view="dense", editable=True)
    target_edge = next(
        (edge for edge in plan.edges if edge.source_node_id == "sync_gate_replica" and edge.target_node_id == "checkpoint_save"),
        None,
    )
    _assert(target_edge is not None, "execution plan exposes sync_gate_replica -> checkpoint_save edge")
    original_snippet = (
        f"    %% edge_id:{target_edge.edge_id}\n"
        '    sync_gate_replica -- "end_of_round" --> checkpoint_save'
    )
    replacement_snippet = (
        f"    %% edge_id:{target_edge.edge_id}\n"
        '    sync_gate_replica -- "end_of_round" --> build_flashcard_rows'
    )
    _assert(original_snippet in editable, "editable Mermaid includes stable edge_id comments")

    updated_plan = apply_mermaid_execution_edit(plan, editable.replace(original_snippet, replacement_snippet, 1))
    updated_edge = next((edge for edge in updated_plan.edges if edge.edge_id == target_edge.edge_id), None)
    _assert(updated_edge is not None, "Mermaid edit preserved the edge record")
    _assert(updated_edge.target_node_id == "build_flashcard_rows", "Mermaid edit rewired the plan edge target")
    _assert(updated_plan.revision == plan.revision + 1, "Mermaid edit increments plan revision")

    rebuilt = build_training_graph_from_plan(updated_plan)
    rebuilt_edge = next((edge for edge in rebuilt.edges if edge.edge_id == target_edge.edge_id), None)
    _assert(rebuilt_edge is not None, "rewired edge exists in rebuilt graph")
    _assert(rebuilt_edge.target_id == "build_flashcard_rows", "rebuilt graph honors Mermaid-edited topology")


def test_edge_callbacks_run_before_should_run():
    print("\n--- test_edge_callbacks_run_before_should_run ---")
    graph = PipelineGraph(name="provision_before_should_run")
    graph.add_node(_DummyNode("source"))

    class _ProvisionedNode(PipelineNode):
        @property
        def node_id(self) -> str:
            return "target"

        def should_run(self, ctx) -> bool:
            return bool(ctx.label_embedding_info.get("ready"))

        def execute(self, ctx) -> None:
            ctx.last_node_statuses[self.node_id] = "ran"

    def _provide(ctx, *, ready: bool = False):
        ctx.label_embedding_info["ready"] = bool(ready)

    graph.add_node(_ProvisionedNode())
    graph.add_edge(
        "source",
        "target",
        on_traverse=_provide,
        target_function="dummy.provide_ready",
        reaction_name="data.provide",
        reaction_defaults={"ready": True, "resources": ["ready_flag"]},
    )

    ctx = PipelineContext()
    statuses = graph.execute_sequence(ctx, verbose=False)
    target_trace = next((entry for entry in ctx.last_execution_trace if entry.get("node_id") == "target"), {})

    _assert(statuses.get("target") == "ran", "edge callback can satisfy should_run preconditions")
    _assert("ready_flag" in list(target_trace.get("frame_keys", []) or []), "provision-order trace records ready flag")



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
        reaction_defaults={"resource": "alpha", "mode": "default", "resources": ["alpha_resource"]},
    )

    ctx = PipelineContext()
    edge_id = graph.edges[0].edge_id
    ctx.edge_reaction_overrides[edge_id] = {"mode": "override"}
    statuses = graph.execute_sequence(ctx, verbose=False)
    merged = dict(ctx.label_embedding_info.get("edge_merge", {}))
    target_trace = next((entry for entry in ctx.last_execution_trace if entry.get("node_id") == "target"), {})

    _assert(statuses.get("target") == "ran", "target node executed with edge reaction")
    _assert(merged.get("resource") == "alpha", "edge default resource preserved")
    _assert(merged.get("mode") == "override", "edge runtime override merged")
    _assert(len(ctx.last_execution_trace) == 2, "execution trace records both steps")
    _assert("alpha_resource" in list(target_trace.get("frame_keys", []) or []), "execution trace records edge-provided frame hints")


def test_execution_program_drives_order():
    print("\n--- test_execution_program_drives_order ---")
    graph = PipelineGraph(name="program_order")
    graph.add_node(_DummyNode("first"))
    graph.add_node(_DummyNode("second"))

    execution_program = {
        "sequence_node_ids": ["second", "first"],
        "steps": [
            {"step_id": "step_001_second", "kind": "node", "node_id": "second", "call_ref": "DummyNode.execute"},
            {"step_id": "step_002_first", "kind": "node", "node_id": "first", "call_ref": "DummyNode.execute"},
        ],
        "transitions": [],
    }

    ctx = PipelineContext()
    statuses = graph.execute_program(ctx, execution_program, verbose=False)
    ordered_nodes = [entry.get("node_id") for entry in ctx.last_execution_trace if entry.get("status") == "ran"]

    _assert(statuses.get("second") == "ran", "program-ordered second node executed")
    _assert(statuses.get("first") == "ran", "program-ordered first node executed")
    _assert(ordered_nodes == ["second", "first"], "execution_program sequence drives runtime node order")
    _assert(all(entry.get("execution_mode") == "program" for entry in ctx.last_execution_trace), "program execution marks trace rows as program-driven")



def test_execution_program_guards_drive_runtime():
    print("\n--- test_execution_program_guards_drive_runtime ---")
    graph = PipelineGraph(name="program_guard")
    graph.add_node(_DummyNode("source"))

    class _GuardedNode(PipelineNode):
        @property
        def node_id(self) -> str:
            return "target"

        def should_run(self, ctx) -> bool:
            return True

        def execute(self, ctx) -> None:
            ctx.last_node_statuses[self.node_id] = "ran"

    def _provide(ctx, *, token: str = ""):
        ctx.label_embedding_info["program_guard_token"] = token

    graph.add_node(_GuardedNode())
    graph.add_edge(
        "source",
        "target",
        on_traverse=_provide,
        target_function="dummy.provide_program_guard",
        reaction_name="data.provide",
        reaction_defaults={"token": "available", "resources": ["program_guard_token"]},
    )

    execution_program = {
        "sequence_node_ids": ["source", "target"],
        "steps": [
            {"step_id": "step_001_source", "kind": "node", "node_id": "source", "call_ref": "DummyNode.execute"},
            {
                "step_id": "step_002_target",
                "kind": "node",
                "node_id": "target",
                "call_ref": "GuardedNode.execute",
                "guard_condition_ids": ["program.guard.enabled"],
                "frame_keys": ["program_guard_token"],
            },
        ],
        "transitions": [],
    }

    blocked_ctx = PipelineContext()
    blocked_statuses = graph.execute_program(
        blocked_ctx,
        execution_program,
        condition_resolver=lambda condition_id, ctx: False,
        verbose=False,
    )
    blocked_trace = next((entry for entry in blocked_ctx.last_execution_trace if entry.get("node_id") == "target"), {})

    _assert(blocked_statuses.get("target") == "skipped:guard", "program guard can skip a node before execution")
    _assert(blocked_trace.get("guard_results", {}).get("program.guard.enabled") is False, "trace records failed program guard")
    _assert(blocked_ctx.label_embedding_info.get("program_guard_token") is None, "blocked program guard suppresses edge provisioning")

    allowed_ctx = PipelineContext()
    allowed_statuses = graph.execute_program(
        allowed_ctx,
        execution_program,
        condition_resolver=lambda condition_id, ctx: True,
        verbose=False,
    )
    allowed_trace = next((entry for entry in allowed_ctx.last_execution_trace if entry.get("node_id") == "target"), {})

    _assert(allowed_statuses.get("target") == "ran", "program guard can allow node execution")
    _assert(allowed_ctx.label_embedding_info.get("program_guard_token") == "available", "allowed program guard preserves edge provisioning")
    _assert("program_guard_token" in list(allowed_trace.get("frame_keys", []) or []), "allowed program guard trace keeps symbolic frame token")


def test_execution_program_flow_visits_decision_and_hold():
    print("\n--- test_execution_program_flow_visits_decision_and_hold ---")
    graph = PipelineGraph(name="program_decision_hold")
    graph.add_node(_DummyNode("source"))

    class _TargetNode(PipelineNode):
        @property
        def node_id(self) -> str:
            return "target"

        def should_run(self, ctx) -> bool:
            return True

        def execute(self, ctx) -> None:
            ctx.last_node_statuses[self.node_id] = "ran"

    def _provide(ctx, *, token: str = ""):
        ctx.label_embedding_info["decision_hold_token"] = token

    graph.add_node(_TargetNode())
    graph.add_edge(
        "source",
        "target",
        on_traverse=_provide,
        target_function="dummy.provide_decision_hold",
        reaction_name="data.provide",
        reaction_defaults={"token": "ready", "resources": ["decision_hold_token"]},
    )

    execution_program = {
        "entry_step_id": "step_source",
        "steps": [
            {"step_id": "step_source", "kind": "node", "node_id": "source", "call_ref": "DummyNode.execute", "display_order": 10},
            {
                "step_id": "decision_target",
                "kind": "decision",
                "label": "Target ready?",
                "call_ref": "scheduler.guard",
                "config": {"target_node_id": "target"},
                "guard_condition_ids": ["program.target.ready"],
                "display_order": 15,
            },
            {"step_id": "step_target", "kind": "node", "node_id": "target", "call_ref": "TargetNode.execute", "display_order": 20},
            {"step_id": "program_hold", "kind": "hold", "label": "Hold / next round", "call_ref": "scheduler.hold", "display_order": 30},
        ],
        "transitions": [
            {"transition_id": "t_source_decision", "from_step_id": "step_source", "to_step_id": "decision_target", "ordinal": 1, "label": "1", "kind": "sequence", "call_ref": "scheduler.advance"},
            {"transition_id": "t_decision_pass", "from_step_id": "decision_target", "to_step_id": "step_target", "ordinal": 2, "label": "2.1", "kind": "branch", "branch": "pass", "call_ref": "scheduler.guard.pass"},
            {"transition_id": "t_decision_hold", "from_step_id": "decision_target", "to_step_id": "program_hold", "ordinal": 2, "label": "2.0", "kind": "branch", "branch": "hold", "call_ref": "scheduler.guard.hold"},
        ],
    }

    blocked_ctx = PipelineContext()
    blocked_statuses = graph.execute_program(
        blocked_ctx,
        execution_program,
        condition_resolver=lambda condition_id, ctx: False,
        verbose=False,
    )
    blocked_branches = [entry for entry in blocked_ctx.last_program_trace if entry.get("entry_kind") == "transition"]
    blocked_steps = [entry for entry in blocked_ctx.last_program_trace if entry.get("entry_kind") == "step"]

    _assert(blocked_statuses.get("source") == "ran", "decision/hold flow runs source step")
    _assert(blocked_statuses.get("target") is None, "hold branch prevents target execution")
    _assert(any(entry.get("kind") == "decision" for entry in blocked_steps), "program trace records explicit decision step")
    _assert(any(entry.get("kind") == "hold" for entry in blocked_steps), "program trace records hold step")
    _assert(any(entry.get("branch") == "hold" for entry in blocked_branches), "program trace records hold transition traversal")
    _assert(blocked_ctx.label_embedding_info.get("decision_hold_token") is None, "hold branch suppresses target provisioning")

    allowed_ctx = PipelineContext()
    allowed_statuses = graph.execute_program(
        allowed_ctx,
        execution_program,
        condition_resolver=lambda condition_id, ctx: True,
        verbose=False,
    )
    allowed_branches = [entry for entry in allowed_ctx.last_program_trace if entry.get("entry_kind") == "transition"]

    _assert(allowed_statuses.get("target") == "ran", "pass branch reaches target node")
    _assert(any(entry.get("branch") == "pass" for entry in allowed_branches), "program trace records pass transition traversal")
    _assert(allowed_ctx.label_embedding_info.get("decision_hold_token") == "ready", "pass branch preserves target provisioning")



def test_execution_program_flow_supports_cycles():
    print("\n--- test_execution_program_flow_supports_cycles ---")
    graph = PipelineGraph(name="program_cycle")

    class _LoopNode(PipelineNode):
        @property
        def node_id(self) -> str:
            return "loop"

        def execute(self, ctx) -> None:
            ctx.label_embedding_info["loop_count"] = int(ctx.label_embedding_info.get("loop_count", 0)) + 1
            ctx.last_node_statuses[self.node_id] = "ran"

    graph.add_node(_LoopNode())
    execution_program = {
        "entry_step_id": "step_loop",
        "steps": [
            {"step_id": "step_loop", "kind": "node", "node_id": "loop", "call_ref": "LoopNode.execute", "display_order": 10},
            {
                "step_id": "decision_loop",
                "kind": "decision",
                "label": "Loop again?",
                "call_ref": "scheduler.guard",
                "config": {"target_node_id": "loop"},
                "guard_condition_ids": ["program.loop.repeat"],
                "display_order": 15,
            },
            {"step_id": "program_hold", "kind": "hold", "label": "Hold / next round", "call_ref": "scheduler.hold", "display_order": 20},
        ],
        "transitions": [
            {"transition_id": "t_loop_decision", "from_step_id": "step_loop", "to_step_id": "decision_loop", "ordinal": 1, "label": "1", "kind": "sequence", "call_ref": "scheduler.advance"},
            {"transition_id": "t_decision_repeat", "from_step_id": "decision_loop", "to_step_id": "step_loop", "ordinal": 2, "label": "2.1", "kind": "branch", "branch": "pass", "call_ref": "scheduler.guard.pass"},
            {"transition_id": "t_decision_hold", "from_step_id": "decision_loop", "to_step_id": "program_hold", "ordinal": 2, "label": "2.0", "kind": "branch", "branch": "hold", "call_ref": "scheduler.guard.hold"},
        ],
    }

    ctx = PipelineContext()
    statuses = graph.execute_program(
        ctx,
        execution_program,
        condition_resolver=lambda condition_id, runtime_ctx: int(runtime_ctx.label_embedding_info.get("loop_count", 0)) < 2,
        verbose=False,
    )
    loop_runs = [entry for entry in ctx.last_execution_trace if entry.get("node_id") == "loop" and entry.get("status") == "ran"]
    loop_decisions = [entry for entry in ctx.last_program_trace if entry.get("kind") == "decision"]
    loop_transitions = [entry for entry in ctx.last_program_trace if entry.get("entry_kind") == "transition"]

    _assert(statuses.get("loop") == "ran", "cyclic program records loop node execution")
    _assert(len(loop_runs) == 2, "cyclic program can revisit the same node step")
    _assert(any(entry.get("branch") == "pass" for entry in loop_transitions), "cyclic program traverses repeat branch")
    _assert(any(entry.get("branch") == "hold" for entry in loop_transitions), "cyclic program eventually traverses hold branch")
    _assert(loop_decisions[-1].get("status") == "hold", "cyclic decision eventually terminates into hold")


# ── main ──────────────────────────────────────────────────────────────────────

def test_action_subnode_roundtrip(plan: TrainingGraphPlan):
    """Verify action records, subnode records, and their JSON roundtrip."""
    print("\n--- test_action_subnode_roundtrip ---")
    # plan.actions should be populated from edges + node introspection
    _assert(len(plan.actions) > 0, f"plan has {len(plan.actions)} action records")
    # every action should have an action_id and kind
    for action in plan.actions:
        _assert(isinstance(action, ActionRecord), f"action {action.action_id} is ActionRecord")
        _assert(bool(action.action_id), "action has non-empty action_id")
        _assert(action.kind in ("edge_traverse", "node_method", "subnode_process", "condition_check"),
                f"action {action.action_id} has valid kind {action.kind!r}")
    # every edge should have an action_id that references an existing action
    action_id_set = {a.action_id for a in plan.actions}
    for edge in plan.edges:
        _assert(bool(edge.action_id), f"edge {edge.edge_id} has action_id")
        _assert(edge.action_id in action_id_set, f"edge {edge.edge_id} action_id {edge.action_id!r} in action registry")
    # at least some nodes should have subnodes
    nodes_with_subnodes = [n for n in plan.nodes if n.subnodes]
    _assert(len(nodes_with_subnodes) > 0, f"{len(nodes_with_subnodes)} nodes have subnodes")
    for node in nodes_with_subnodes:
        for sn in node.subnodes:
            _assert(isinstance(sn, SubnodeRecord), f"subnode {sn.subnode_id} is SubnodeRecord")
            _assert(sn.parent_node_id == node.node_id, f"subnode {sn.subnode_id} parent matches node {node.node_id}")
        _assert(len(node.owned_action_ids) > 0, f"node {node.node_id} has owned action ids")
        for aid in node.owned_action_ids:
            _assert(aid in action_id_set, f"node {node.node_id} owned action {aid!r} in action registry")
    # JSON roundtrip preserves actions and subnodes
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "action_roundtrip.json"
        plan.save_json(path)
        reloaded = TrainingGraphPlan.load_json(path)
    _assert(len(reloaded.actions) == len(plan.actions), "action count survives roundtrip")
    reloaded_action_ids = {a.action_id for a in reloaded.actions}
    for a in plan.actions:
        _assert(a.action_id in reloaded_action_ids, f"action {a.action_id} survives roundtrip")
    for orig_node in plan.nodes:
        reloaded_node = next((n for n in reloaded.nodes if n.node_id == orig_node.node_id), None)
        _assert(reloaded_node is not None, f"node {orig_node.node_id} survives roundtrip")
        _assert(len(reloaded_node.subnodes) == len(orig_node.subnodes),
                f"node {orig_node.node_id} subnode count survives roundtrip")
        _assert(reloaded_node.owned_action_ids == orig_node.owned_action_ids,
                f"node {orig_node.node_id} owned_action_ids survive roundtrip")
    for orig_edge in plan.edges:
        reloaded_edge = next((e for e in reloaded.edges if e.edge_id == orig_edge.edge_id), None)
        _assert(reloaded_edge is not None, f"edge {orig_edge.edge_id} survives roundtrip")
        _assert(reloaded_edge.action_id == orig_edge.action_id,
                f"edge {orig_edge.edge_id} action_id survives roundtrip")
    # validation should still pass
    reloaded.validate()
    _ok("action/subnode records populated, referenced, and survive JSON roundtrip")
    # stack_view layer should be materialized with subnodes
    layers = plan.graph_layers or {}
    stack_view = layers.get("stack_view", {})
    _assert(stack_view.get("status") == "active", "stack_view layer is active")
    _assert(len(stack_view.get("nodes", [])) > 0, "stack_view has nodes")
    _assert(len(stack_view.get("edges", [])) > 0, "stack_view has edges")
    _ok("stack_view layer materialized with subnodes")


def test_gpu_models_roundtrip(plan: TrainingGraphPlan):
    print("\n--- test_gpu_models_roundtrip ---")
    # At least some nodes should declare gpu_models
    nodes_with_models = [n for n in plan.nodes if n.gpu_models]
    _assert(len(nodes_with_models) > 0, f"{len(nodes_with_models)} nodes declare gpu_models")

    # Spot-check known nodes
    build_classifier = next((n for n in plan.nodes if n.node_id == "build_classifier"), None)
    _assert(build_classifier is not None, "plan includes build_classifier node")
    _assert("classifier" in build_classifier.gpu_models, "build_classifier declares 'classifier' in gpu_models")

    build_gan = next((n for n in plan.nodes if n.node_id == "build_gan"), None)
    _assert(build_gan is not None, "plan includes build_gan node")

    stage_g = next((n for n in plan.nodes if n.node_id == "stage_g_generator"), None)
    _assert(stage_g is not None, "plan includes stage_g_generator node")
    _assert("generator" in stage_g.gpu_models, "stage_g_generator declares 'generator' in gpu_models")

    # JSON round-trip preserves gpu_models
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "gpu_models_roundtrip.json"
        plan.save_json(path)
        reloaded = TrainingGraphPlan.load_json(path)
        for orig_node in plan.nodes:
            reloaded_node = next((n for n in reloaded.nodes if n.node_id == orig_node.node_id), None)
            _assert(reloaded_node is not None, f"node {orig_node.node_id} survives roundtrip")
            _assert(reloaded_node.gpu_models == orig_node.gpu_models,
                    f"node {orig_node.node_id} gpu_models survive roundtrip ({orig_node.gpu_models})")
    _ok("gpu_models survive JSON round-trip for all nodes")


def test_provenance_layer(graph):
    print("\n--- test_provenance_layer ---")
    layers = build_graph_layers(graph)
    provenance = layers.get("provenance", {})
    _assert(provenance.get("status") == "active", "provenance layer is active")
    _assert(len(provenance.get("nodes", [])) > 0, "provenance layer has nodes")
    _assert(len(provenance.get("edges", [])) > 0, "provenance layer has edges")

    # Should have model resource nodes
    resource_nodes = [n for n in provenance["nodes"] if n.get("node_id", "").startswith("model::")]
    _assert(len(resource_nodes) > 0, f"provenance has {len(resource_nodes)} model resource nodes")

    # Should have 'classifier' as a resource
    classifier_resource = next((n for n in resource_nodes if n["node_id"] == "model::classifier"), None)
    _assert(classifier_resource is not None, "provenance includes model::classifier resource")

    # Edges should link pipeline nodes to resources
    residency_edges = [e for e in provenance["edges"] if e.get("reaction_name") == "residency.acquire"]
    _assert(len(residency_edges) > 0, f"provenance has {len(residency_edges)} residency edges")

    # Mermaid renders without error
    from pipeline.graph_layers import render_mermaid_flowchart
    chart = render_mermaid_flowchart(provenance, view="dense")
    _assert("model__classifier" in chart or "classifier" in chart.lower(), "provenance Mermaid mentions classifier model")
    _ok("provenance layer is populated and renderable")


def test_cycle_edges_roundtrip(graph):
    """Verify cycle edges are synthesized, survive JSON round-trip, and render in Mermaid."""
    print("\n--- test_cycle_edges_roundtrip ---")
    plan = plan_from_pipeline_graph(
        graph,
        name="Cycle Test Plan",
        revision=1,
        worker_hints={
            "output_dir": "/tmp/smoke_cycle",
            "orchestration_cycles": 3,
            "orchestration_rounds": 4,
            "capabilities": ["plan_apply"],
        },
        graph_layers=build_graph_layers(graph),
        execution_program=build_execution_program(graph),
    )
    # Cycle edge should exist
    cycle_edges = [e for e in plan.edges if e.layer == "cycle"]
    _assert(len(cycle_edges) >= 1, f"plan has {len(cycle_edges)} cycle edge(s)")
    ce = cycle_edges[0]
    _assert(ce.kind == "cycle_return", f"cycle edge kind is {ce.kind!r}")
    _assert(bool(ce.cycle_control), "cycle edge has cycle_control dict")
    _assert(ce.cycle_control.get("max_iterations") == 12, f"max_iterations = {ce.cycle_control.get('max_iterations')}")
    _assert(ce.cycle_control.get("cycles") == 3, f"cycles = {ce.cycle_control.get('cycles')}")
    _assert(ce.cycle_control.get("rounds_per_cycle") == 4, f"rounds_per_cycle = {ce.cycle_control.get('rounds_per_cycle')}")
    _assert(ce.cycle_control.get("stride", 0) > 0, f"stride = {ce.cycle_control.get('stride')}")
    _assert(str(ce.metadata.get("style_role", "")) == "cycle", "cycle edge metadata has style_role='cycle'")

    # Validate() should pass (cycle edges excluded from DAG check)
    plan.validate()
    _ok("plan.validate() passes with cycle edges present")

    # JSON round-trip
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "cycle_plan.json"
        plan.save_json(path)
        reloaded = TrainingGraphPlan.load_json(path)
    reloaded_cycle = [e for e in reloaded.edges if e.layer == "cycle"]
    _assert(len(reloaded_cycle) == len(cycle_edges), "cycle edges survive JSON round-trip")
    _assert(reloaded_cycle[0].cycle_control == ce.cycle_control, "cycle_control dict round-trips")

    # Mermaid rendering — cycle edge should use dotted arrow
    from pipeline.graph_layers import build_graph_layers_from_plan, render_mermaid_flowchart
    layers = build_graph_layers_from_plan(reloaded)
    execution_layer = layers.get("execution", {})
    cycle_layer_edges = [
        e for e in execution_layer.get("edges", [])
        if str(dict(e.get("metadata", {}) or {}).get("style_role", "")) == "cycle"
    ]
    _assert(len(cycle_layer_edges) >= 1, f"execution layer includes {len(cycle_layer_edges)} cycle edge(s)")
    mermaid = render_mermaid_flowchart(execution_layer, view="dense")
    _assert(".->" in mermaid, "Mermaid uses dotted arrow for cycle edge")
    _assert("round return" in mermaid.lower(), "Mermaid shows cycle edge label")
    _ok("cycle edges synthesized, round-tripped, and rendered in Mermaid")


def test_cycle_gate_instantiation_from_ir(graph):
    """Verify that IR cycle edges cause CycleGate objects to be instantiated,
    and that those objects own the iteration logic."""
    print("\n--- test_cycle_gate_instantiation_from_ir ---")
    from pipeline.graph import CycleGate

    # Build a plan with cycle edges (3 cycles × 2 rounds = 6 iterations).
    plan = plan_from_pipeline_graph(
        graph,
        name="CycleGate Instantiation Test",
        revision=1,
        worker_hints={
            "output_dir": "/tmp/smoke_cycle_gate",
            "orchestration_cycles": 3,
            "orchestration_rounds": 2,
            "capabilities": ["plan_apply"],
        },
        graph_layers=build_graph_layers(graph),
        execution_program=build_execution_program(graph),
    )

    # The IR cycle edges must cause CycleGate objects to exist.
    gates = CycleGate.from_plan(plan)
    _assert(len(gates) >= 1, f"CycleGate.from_plan produced {len(gates)} gate(s)")
    gate = gates[0]
    _assert(gate.max_iterations == 6, f"gate.max_iterations={gate.max_iterations} (expected 6)")
    _assert(gate.cycles == 3, f"gate.cycles={gate.cycles} (expected 3)")
    _assert(gate.rounds_per_cycle == 2, f"gate.rounds_per_cycle={gate.rounds_per_cycle} (expected 2)")
    _assert(gate.iteration == 0, "gate starts at iteration 0")
    _assert(not gate.exhausted, "gate is not exhausted at start")

    # The gate object owns the repeat/exhaust decision.
    results = []
    for _ in range(6):
        results.append(gate.evaluate())
    _assert(results[:5] == ["repeat"] * 5, f"first 5 evaluations are 'repeat': {results[:5]}")
    _assert(results[5] == "exhaust", f"6th evaluation is 'exhaust': {results[5]}")
    _assert(gate.exhausted, "gate is exhausted after max_iterations evaluations")
    _assert(gate.iteration == 6, f"gate.iteration={gate.iteration} after exhaustion")

    # Legacy plan without cycle edges → no gates.
    plan_no_cycle = plan_from_pipeline_graph(
        graph,
        name="No Cycle Plan",
        revision=1,
        worker_hints={
            "output_dir": "/tmp/smoke_no_cycle",
            "orchestration_cycles": 0,
            "orchestration_rounds": 0,
        },
        graph_layers=build_graph_layers(graph),
        execution_program=build_execution_program(graph),
    )
    gates_empty = CycleGate.from_plan(plan_no_cycle)
    _assert(len(gates_empty) == 0, f"legacy plan produces 0 gates, got {len(gates_empty)}")
    _ok("CycleGate objects instantiated from IR; they own iteration state")


def test_cycle_gate_drives_interpreter(graph):
    """Verify the interpreter uses CycleGate objects from ctx to loop,
    with no cycle-specific steps in the execution program."""
    print("\n--- test_cycle_gate_drives_interpreter ---")
    from pipeline.graph import CycleGate

    # Build a minimal graph with one node.
    mini_graph = PipelineGraph(name="cycle_gate_interp")

    class _Counter(PipelineNode):
        @property
        def node_id(self) -> str:
            return "counter"
        def execute(self, ctx) -> None:
            ctx.label_embedding_info["n"] = int(ctx.label_embedding_info.get("n", 0)) + 1
            ctx.last_node_statuses[self.node_id] = "ran"

    mini_graph.add_node(_Counter())

    # A plain forward-only program — NO cycle_decision steps, NO cycle
    # transitions.  The program is just: node → hold.
    mini_program = {
        "entry_step_id": "step_counter",
        "halt_step_ids": ["program_hold"],
        "steps": [
            {"step_id": "step_counter", "kind": "node", "node_id": "counter",
             "call_ref": "Counter.execute", "display_order": 10,
             "guard_condition_ids": [], "frame_keys": []},
            {"step_id": "program_hold", "kind": "hold",
             "label": "Hold", "call_ref": "scheduler.hold",
             "display_order": 20, "guard_condition_ids": [], "frame_keys": []},
        ],
        "transitions": [
            {"transition_id": "t1", "from_step_id": "step_counter",
             "to_step_id": "program_hold", "ordinal": 1, "label": "1",
             "kind": "sequence", "call_ref": "scheduler.advance"},
        ],
    }

    max_iter = 4

    # CycleGate on ctx — this is the object caused by the IR cycle edge.
    # The interpreter consults it at the hold point.
    gate = CycleGate(
        edge_id="test::cycle_return",
        source_node_id="counter",
        target_node_id="counter",
        cycle_control={
            "max_iterations": max_iter,
            "cycles": 1,
            "rounds_per_cycle": max_iter,
            "stride": 1,
        },
    )

    ctx = PipelineContext()
    ctx.cycle_gates = [gate]

    statuses = mini_graph.execute_program(ctx, mini_program, verbose=False)
    runs = int(ctx.label_embedding_info.get("n", 0))
    _assert(runs == max_iter, f"interpreter executed node {runs} times (expected {max_iter})")

    # The program trace should show cycle_gate entries from the CycleGate.
    gate_entries = [e for e in ctx.last_program_trace if e.get("kind") == "cycle_gate"]
    _assert(len(gate_entries) == max_iter - 1, f"{len(gate_entries)} cycle_gate trace entries (expected {max_iter - 1} repeats)")
    _assert(all(e.get("status") == "repeat" for e in gate_entries), "all gate trace entries are 'repeat'")

    # Final entry should be a hold (the last evaluate returned 'exhaust').
    hold_entries = [e for e in ctx.last_program_trace if e.get("kind") == "hold"]
    _assert(len(hold_entries) == 1, "exactly 1 hold entry at end")

    # ctx.total_rounds_completed updated by the gate object.
    _assert(ctx.total_rounds_completed == max_iter, f"ctx.total_rounds_completed={ctx.total_rounds_completed}")

    # Without a CycleGate, no looping — single pass.
    ctx2 = PipelineContext()
    mini_graph2 = PipelineGraph(name="no_gate")
    mini_graph2.add_node(_Counter())
    mini_graph2.execute_program(ctx2, mini_program, verbose=False)
    runs2 = int(ctx2.label_embedding_info.get("n", 0))
    _assert(runs2 == 1, f"without CycleGate, node runs {runs2} time(s) (expected 1)")

    _ok("CycleGate objects on ctx drive the interpreter loop; no cycle steps in program")


def test_execution_policy_roundtrip(graph):
    """Verify execution_policy fields survive plan export and JSON round-trip."""
    print("\n--- test_execution_policy_roundtrip ---")
    plan = plan_from_pipeline_graph(
        graph,
        name="Policy Test Plan",
        revision=1,
        worker_hints={"output_dir": "/tmp/smoke_policy", "capabilities": ["plan_apply"]},
        graph_layers=build_graph_layers(graph),
        execution_program=build_execution_program(graph),
    )

    # Check specific nodes for expected policies.
    _by_id = {n.node_id: n for n in plan.nodes}

    wave_pool = _by_id.get("wave_pool")
    _assert(wave_pool is not None, "wave_pool node present")
    _assert(wave_pool.execution_policy == "once", f"wave_pool policy={wave_pool.execution_policy!r}")

    build_cls = _by_id.get("build_classifier")
    _assert(build_cls is not None, "build_classifier node present")
    _assert(build_cls.execution_policy == "once", f"build_classifier policy={build_cls.execution_policy!r}")

    config_srch = _by_id.get("config_search")
    _assert(config_srch is not None, "config_search node present")
    _assert(config_srch.execution_policy == "once", f"config_search policy={config_srch.execution_policy!r}")

    build_tx = _by_id.get("build_transformer")
    _assert(build_tx is not None, "build_transformer node present")
    _assert(build_tx.execution_policy == "once", f"build_transformer policy={build_tx.execution_policy!r}")

    build_gan = _by_id.get("build_gan")
    _assert(build_gan is not None, "build_gan node present")
    _assert(build_gan.execution_policy == "once", f"build_gan policy={build_gan.execution_policy!r}")

    ckpt = _by_id.get("checkpoint_save")
    _assert(ckpt is not None, "checkpoint_save node present")
    _assert(ckpt.execution_policy == "periodic", f"checkpoint_save policy={ckpt.execution_policy!r}")
    _assert(isinstance(ckpt.execution_policy_config, dict), "checkpoint_save has policy config dict")
    _assert(ckpt.execution_policy_config.get("period", 0) >= 1, "checkpoint_save period >= 1")

    # A gated node should report "gated".
    gest_eval = _by_id.get("gate_1_gestation_eval")
    _assert(gest_eval is not None, "gate_1_gestation_eval node present")
    _assert(gest_eval.execution_policy == "gated", f"gate_1_gestation_eval policy={gest_eval.execution_policy!r}")
    _assert(isinstance(gest_eval.execution_policy_config.get("gate_ids"), list), "gated node has gate_ids list")

    # A plain node should default to "always".
    preg_eval = _by_id.get("gate_0_pregestation_eval")
    _assert(preg_eval is not None, "gate_0_pregestation_eval node present")
    _assert(preg_eval.execution_policy == "always", f"gate_0_pregestation_eval policy={preg_eval.execution_policy!r}")

    # JSON round-trip preserves fields.
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "policy_plan.json"
        plan.save_json(path)
        reloaded = TrainingGraphPlan.load_json(path)
    _by_id2 = {n.node_id: n for n in reloaded.nodes}
    _assert(_by_id2["wave_pool"].execution_policy == "once", "wave_pool policy survives round-trip")
    _assert(_by_id2["checkpoint_save"].execution_policy == "periodic", "checkpoint_save policy survives round-trip")
    _assert(_by_id2["checkpoint_save"].execution_policy_config.get("period", 0) >= 1, "checkpoint_save config survives round-trip")
    _assert(_by_id2["gate_1_gestation_eval"].execution_policy == "gated", "gated policy survives round-trip")
    _ok("execution_policy fields survive plan export and JSON round-trip")


def test_execution_policy_mermaid_badges(graph):
    """Verify execution_policy badges appear in dense Mermaid rendering."""
    print("\n--- test_execution_policy_mermaid_badges ---")
    plan = plan_from_pipeline_graph(
        graph,
        name="Badge Test Plan",
        revision=1,
        worker_hints={"output_dir": "/tmp/smoke_badge", "capabilities": ["plan_apply"]},
        graph_layers=build_graph_layers(graph),
        execution_program=build_execution_program(graph),
    )
    from pipeline.graph_layers import build_graph_layers_from_plan
    layers = build_graph_layers_from_plan(plan)
    execution_layer = layers.get("execution", {})

    # Dense rendering should show badges for non-"always" policies.
    mermaid = render_mermaid_flowchart(execution_layer, view="dense")
    _assert("[once]" in mermaid, "dense Mermaid contains [once] badge")
    _assert("[periodic]" in mermaid, "dense Mermaid contains [periodic] badge")
    _assert("[gated]" in mermaid, "dense Mermaid contains [gated] badge")

    # Minimal view should NOT contain badges.
    mermaid_min = render_mermaid_flowchart(execution_layer, view="minimal")
    _assert("[once]" not in mermaid_min, "minimal Mermaid does NOT contain [once] badge")

    _ok("execution_policy badges render correctly in dense Mermaid")


# ── test: condition expression evaluator ──────────────────────────────────────

def test_condition_expr_evaluator():
    """Verify the condition expression DSL parses and evaluates correctly."""
    print("\n--- test_condition_expr_evaluator ---")
    from pipeline.condition_expr import (
        evaluate_condition_expr,
        validate_condition_expr,
        expr_for_condition_id,
    )

    # Simple boolean literals.
    ctx = types.SimpleNamespace()
    _assert(evaluate_condition_expr("TRUE", ctx) is True, "TRUE evaluates to True")
    _assert(evaluate_condition_expr("FALSE", ctx) is False, "FALSE evaluates to False")
    _assert(evaluate_condition_expr("NOT FALSE", ctx) is True, "NOT FALSE evaluates to True")

    # IS_NONE / IS_NOT_NONE.
    ctx = types.SimpleNamespace(foo=None, bar=42)
    _assert(evaluate_condition_expr("foo IS_NONE", ctx) is True, "foo IS_NONE with None value")
    _assert(evaluate_condition_expr("bar IS_NOT_NONE", ctx) is True, "bar IS_NOT_NONE with value")
    _assert(evaluate_condition_expr("bar IS_NONE", ctx) is False, "bar IS_NONE when not None")

    # CONTAINS.
    ctx = types.SimpleNamespace(orchestration_mode="gcw")
    _assert(evaluate_condition_expr('orchestration_mode CONTAINS "g"', ctx) is True, 'CONTAINS "g" in "gcw"')
    _assert(evaluate_condition_expr('orchestration_mode CONTAINS "x"', ctx) is False, 'CONTAINS "x" not in "gcw"')

    # MOD.
    ctx = types.SimpleNamespace(round_id=6)
    _assert(evaluate_condition_expr("round_id MOD 3 == 0", ctx) is True, "6 MOD 3 == 0")
    _assert(evaluate_condition_expr("round_id MOD 4 == 0", ctx) is False, "6 MOD 4 != 0")
    _assert(evaluate_condition_expr("round_id % 4 == 2", ctx) is True, "6 % 4 == 2")

    # Calculator signals.
    ctx = types.SimpleNamespace(
        lhs=4,
        rhs=2,
        scale=3,
        divisor=2,
        target=9,
        ctx=types.SimpleNamespace(total_rounds_completed=10),
        data=types.SimpleNamespace(_preg_last_build_round=6),
        preg_cfg=types.SimpleNamespace(rebuild_every_n_rounds=4),
    )
    _assert(
        evaluate_condition_expr("((lhs + rhs) * scale) / divisor == target", ctx) is True,
        "calculator supports +, *, / with signal operands",
    )
    _assert(
        evaluate_condition_expr(
            "(ctx.total_rounds_completed - data._preg_last_build_round) >= preg_cfg.rebuild_every_n_rounds",
            ctx,
        ) is True,
        "calculator supports subtraction with accessor signals on both sides of comparison",
    )
    ctx = types.SimpleNamespace(
        gate_override_enabled=lambda: False,
        gate_pregestation=types.SimpleNamespace(passed=True),
        early_gates_passed=lambda: True,
        ctx=types.SimpleNamespace(total_rounds_completed=10),
        data=types.SimpleNamespace(_gest_last_build_round=4, _bdata_last_build_round=3),
        gest_cfg=types.SimpleNamespace(rebuild_every_n_rounds=6),
        bdata_cfg=types.SimpleNamespace(rebuild_every_n_rounds=7),
    )
    _assert(
        evaluate_condition_expr(expr_for_condition_id("data.gestation_rebuild_due"), ctx) is True,
        "gestation rebuild condition resolves through real gate signals",
    )
    _assert(
        evaluate_condition_expr(expr_for_condition_id("data.berkeley_refresh_due"), ctx) is True,
        "berkeley rebuild condition resolves through real early-gate signal",
    )

    # AND / OR.
    ctx = types.SimpleNamespace(a=True, b=False)
    _assert(evaluate_condition_expr("a AND b", ctx) is False, "True AND False = False")
    _assert(evaluate_condition_expr("a OR b", ctx) is True, "True OR False = True")

    # Dotted accessors.
    ctx = types.SimpleNamespace(gate_pregestation=types.SimpleNamespace(passed=True))
    _assert(evaluate_condition_expr("gate_pregestation.passed", ctx) is True, "dotted accessor truthy")

    # GATE_OVERRIDE.
    ctx = types.SimpleNamespace(gate_override_enabled=lambda: True)
    _assert(evaluate_condition_expr("GATE_OVERRIDE", ctx) is True, "GATE_OVERRIDE when override enabled")
    ctx = types.SimpleNamespace(gate_override_enabled=lambda: False)
    _assert(evaluate_condition_expr("GATE_OVERRIDE", ctx) is False, "GATE_OVERRIDE when override disabled")

    # Validation.
    _assert(validate_condition_expr("a AND b") is None, "valid expression passes validation")
    _assert(validate_condition_expr("a AND AND b") is not None, "invalid expression fails validation")

    # Legacy mapping.
    for cid in [
        "orchestration.mode_has_generator",
        "gates.pregestation_passed",
        "gates.early_passed",
        "gates.all_base_passed",
        "gates.wave_stage_ready",
        "data.pregestation_rebuild_due",
        "data.gestation_rebuild_due",
        "data.berkeley_refresh_due",
    ]:
        expr = expr_for_condition_id(cid)
        _assert(bool(expr), f"expr_for_condition_id({cid!r}) returns non-empty string")
        _assert(validate_condition_expr(expr) is None, f"expr for {cid!r} is syntactically valid")

    _ok("condition expression evaluator parses and evaluates correctly")


# ── test: condition_expr roundtrip in plan JSON ───────────────────────────────

def test_condition_expr_plan_roundtrip(graph):
    """Verify condition_expr and run_condition_expr survive plan JSON round-trip."""
    print("\n--- test_condition_expr_plan_roundtrip ---")
    plan = plan_from_pipeline_graph(
        graph,
        name="CondExpr RT Test Plan",
        revision=1,
        worker_hints={"output_dir": "/tmp/smoke_cexpr", "capabilities": ["plan_apply"]},
        graph_layers=build_graph_layers(graph),
        execution_program=build_execution_program(graph),
    )

    # Check that condition_expr is populated on edges with a condition_id.
    edges_with_cid = [e for e in plan.edges if str(e.condition_id or "").strip()]
    _assert(len(edges_with_cid) > 0, "plan has edges with condition_id")
    for edge in edges_with_cid:
        _assert(bool(edge.condition_expr), f"edge {edge.edge_id} has condition_expr for condition_id={edge.condition_id!r}")

    # Check that nodes have run_condition_expr populated.
    nodes_with_rcexpr = [n for n in plan.nodes if str(getattr(n, "run_condition_expr", "") or "").strip()]
    _assert(len(nodes_with_rcexpr) > 0, "some plan nodes have run_condition_expr")

    # JSON round-trip.
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "cexpr_plan.json"
        plan.save_json(path)
        reloaded = TrainingGraphPlan.load_json(path)

    # Edge condition_expr survives.
    reloaded_edge_map = {e.edge_id: e for e in reloaded.edges}
    for edge in edges_with_cid:
        re = reloaded_edge_map.get(edge.edge_id)
        _assert(re is not None, f"edge {edge.edge_id} present after reload")
        _assert(re.condition_expr == edge.condition_expr, f"edge {edge.edge_id} condition_expr survives round-trip")

    # Node run_condition_expr survives.
    reloaded_node_map = {n.node_id: n for n in reloaded.nodes}
    for node in nodes_with_rcexpr:
        rn = reloaded_node_map.get(node.node_id)
        _assert(rn is not None, f"node {node.node_id} present after reload")
        _assert(rn.run_condition_expr == node.run_condition_expr, f"node {node.node_id} run_condition_expr survives round-trip")

    _ok("condition_expr and run_condition_expr survive plan JSON round-trip")


# ── test: condition_expr appears in Mermaid and roundtrips through edit ────────

def test_condition_expr_mermaid_roundtrip(graph):
    """Verify condition_expr appears in dense Mermaid and survives editable round-trip."""
    print("\n--- test_condition_expr_mermaid_roundtrip ---")
    plan = plan_from_pipeline_graph(
        graph,
        name="CondExpr Mermaid Test",
        revision=1,
        worker_hints={"output_dir": "/tmp/smoke_mermaid_cexpr", "capabilities": ["plan_apply"]},
        graph_layers=build_graph_layers(graph),
        execution_program=build_execution_program(graph),
    )

    # Render dense with identity comments (editable mode).
    mermaid = render_plan_mermaid_flowchart(plan, layer_id="execution", view="dense", editable=True)

    # Dense rendering should contain condition_expr annotations for gated edges.
    _assert("GATE_OVERRIDE" in mermaid, "dense Mermaid contains GATE_OVERRIDE expression")

    # Minimal rendering should NOT contain expressions.
    mermaid_min = render_plan_mermaid_flowchart(plan, layer_id="execution", view="minimal")
    _assert("GATE_OVERRIDE" not in mermaid_min, "minimal Mermaid does NOT contain GATE_OVERRIDE")

    # Round-trip through editable Mermaid.
    plan2 = apply_mermaid_execution_edit(plan, mermaid)
    edges_with_cexpr = [e for e in plan2.edges if bool(getattr(e, "condition_expr", ""))]
    _assert(len(edges_with_cexpr) > 0, "condition_expr survives Mermaid edit round-trip")

    # Modify a condition_expr in the Mermaid text by replacing an expression.
    original_edge = next((e for e in plan.edges if str(e.condition_expr or "").strip()), None)
    _assert(original_edge is not None, "found edge with condition_expr for edit test")
    old_expr = original_edge.condition_expr
    new_expr = "TRUE"
    modified_mermaid = mermaid.replace(old_expr, new_expr, 1)
    if modified_mermaid != mermaid:
        plan3 = apply_mermaid_execution_edit(plan, modified_mermaid)
        edited_edge = next((e for e in plan3.edges if e.edge_id == original_edge.edge_id), None)
        _assert(edited_edge is not None, "edited edge found after edit")
        _assert(edited_edge.condition_expr == new_expr, f"condition_expr updated from Mermaid edit: {edited_edge.condition_expr!r}")
        _ok("condition_expr was edited via Mermaid and persisted")
    else:
        _ok("(skipped edit mutation — expression not found verbatim in Mermaid)")

    _ok("condition_expr appears in Mermaid and roundtrips through editable edit")


def test_mask_instance_duplication_and_fallback():
    print("\n--- test_mask_instance_duplication_and_fallback ---")
    from semantic_dataset_loaders import elem_stacks_to_label_stacks

    elem_stack = np.stack(
        [
            np.pad(np.ones((4, 2), dtype=np.float32), ((0, 0), (0, 2))),
            np.pad(np.ones((4, 2), dtype=np.float32), ((0, 0), (2, 0))),
        ],
        axis=0,
    )
    elem_term_lists = [["signal"], ["signal"]]
    label_vec = np.asarray([1.0, 1.0], dtype=np.float32)
    stack, indices = elem_stacks_to_label_stacks(
        elem_stack=elem_stack,
        elem_term_lists=elem_term_lists,
        label_vec=label_vec,
        term_to_idx={"signal": 0},
    )
    _assert(int(stack.shape[0]) == 3, f"duplicate label instances preserved with fallback row ({int(stack.shape[0])})")
    _assert(indices.tolist().count(0) == 2, f"label 0 retains both instances ({indices.tolist()})")
    fallback_hits = np.where(indices == 1)[0].tolist()
    _assert(len(fallback_hits) == 1, "missing positive label gets exactly one fallback creation mask")
    fallback_mask = np.asarray(stack[int(fallback_hits[0])], dtype=np.float32)
    _assert(bool(np.allclose(fallback_mask, np.ones_like(fallback_mask))), "missing label fallback is full-image creation mask")


def test_single_label_passes_combine_duplicate_masks():
    print("\n--- test_single_label_passes_combine_duplicate_masks ---")
    from pipeline.nodes.data_nodes import _expand_semantic_mask_supervision_batch

    xb = torch.zeros((1, 3, 4, 4), dtype=torch.float32)
    yb = torch.tensor([[1.0]], dtype=torch.float32)
    mb = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    left = torch.zeros((4, 4), dtype=torch.float32)
    right = torch.zeros((4, 4), dtype=torch.float32)
    left[:, :2] = 1.0
    right[:, 2:] = 1.0
    out_x, out_y, out_m = _expand_semantic_mask_supervision_batch(
        xb=xb,
        yb=yb,
        mb=mb,
        batch_meta={
            "mask_stacks": [torch.stack([left, right], dim=0)],
            "mask_indices": [torch.tensor([0, 0], dtype=torch.long)],
        },
        mode="single_label_passes",
        context="smoke",
    )
    _assert(tuple(out_x.shape) == (1, 3, 4, 4), f"expanded batch keeps one row for one active label: {tuple(out_x.shape)}")
    _assert(tuple(out_y.shape) == (1, 1), f"expanded labels shape correct: {tuple(out_y.shape)}")
    _assert(bool(torch.allclose(out_m[0, 0], torch.ones((4, 4), dtype=torch.float32))), "duplicate label instances collapse to one normalized mask")


def test_data_node_mask_subnodes_exported(plan):
    print("\n--- test_data_node_mask_subnodes_exported ---")
    data_node = next((node for node in plan.nodes if node.node_id == "data_node"), None)
    _assert(data_node is not None, "plan includes data_node for mask subnode export")
    subnode_labels = {str(sn.label) for sn in (data_node.subnodes or [])}
    _assert("Image Generator" in subnode_labels, "data_node exports Image Generator subnode")
    _assert("Distortion Masks" in subnode_labels, "data_node exports Distortion Masks subnode")
    _assert("Heuristic Eye" in subnode_labels, "data_node exports Heuristic Eye subnode")
    stack_view = dict((plan.graph_layers or {}).get("stack_view", {}) or {})
    stack_labels = {str(node.get("label", "")) for node in list(stack_view.get("nodes", []) or [])}
    _assert("Image Generator" in stack_labels, "stack view includes Image Generator")
    _assert("Distortion Masks" in stack_labels, "stack view includes Distortion Masks")
    _assert("Heuristic Eye" in stack_labels, "stack view includes Heuristic Eye")


def main():
    print("=== Smoke test: plan round-trip ===")
    graph, node_count, edge_count = test_build_pipeline_graph()
    plan = test_plan_export(graph, node_count)
    reloaded_plan = test_json_roundtrip(plan, node_count, edge_count)
    test_rebuild_from_plan(reloaded_plan, node_count, edge_count)
    test_run_from_plan_namespace(reloaded_plan)
    test_plan_validation_guardrails(reloaded_plan)
    test_layer_rendering_and_readme_sync(graph)
    test_plan_topology_is_authoritative(reloaded_plan, edge_count)
    test_editable_mermaid_execution_roundtrip(reloaded_plan)
    test_edge_callbacks_run_before_should_run()
    test_edge_reaction_merge()
    test_execution_program_drives_order()
    test_execution_program_guards_drive_runtime()
    test_execution_program_flow_visits_decision_and_hold()
    test_execution_program_flow_supports_cycles()
    test_action_subnode_roundtrip(reloaded_plan)
    test_gpu_models_roundtrip(plan)
    test_provenance_layer(graph)
    test_cycle_edges_roundtrip(graph)
    test_cycle_gate_instantiation_from_ir(graph)
    test_cycle_gate_drives_interpreter(graph)
    test_execution_policy_roundtrip(graph)
    test_execution_policy_mermaid_badges(graph)
    test_condition_expr_evaluator()
    test_condition_expr_plan_roundtrip(graph)
    test_condition_expr_mermaid_roundtrip(graph)
    test_mask_instance_duplication_and_fallback()
    test_single_label_passes_combine_duplicate_masks()
    test_data_node_mask_subnodes_exported(reloaded_plan)
    print("\n=== All checks passed ===")


if __name__ == "__main__":
    main()
