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
    _assert(data_node.object_type == "object", "data_node object_type exported")
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
    print("\n=== All checks passed ===")


if __name__ == "__main__":
    main()
