"""
Pipeline Orchestrator — builds the graph, defines all edge sequences, runs the loop.

This is the only place where the overall execution order and conditional routing
is specified.  If you want to change *when* a stage runs relative to another,
or add a new conditional bypass, this is where you make that change.

Graph topology (sequential stages per round)
--------------------------------------------

    [init_vocab]
         │
    [vocab_churn]  ← condition: on churn schedule
         │
    [build_symbol_pool]
         │
    [build_label_embedding]
         │
    [data_node] ─────────────────────►[stage_0_pregestation]
                ─(after_gate0)──────►[stage_1_gestation]
                ─(after_gate1)──────►[stage_2_berkeley]
                ─(after_gate1)──────►[gate_berkeley]
                ─(after_all_gates)──►[stage_g_generator]
                ──────────────────►[build_flashcard_rows]
                                   │
                              [stage_0_pregestation]
                                   │
                              [stage_1_gestation]  ← cond: gate_pregestation
                                   │
                              [stage_2_berkeley]   ← cond: early_gates
                                   │
                              [gate_berkeley]             ← cond: early_gates
                                   │
    [config_search] ───────────────►[build_transformer]
                                   │
                              [stage_r_transformer]       ← cond: early_gates
                                   │
                              [gate_transformer]          ← cond: early_gates
                                   │
                              [build_gan]                 ← cond: early_gates
                                   │
                              [stage_g_generator]         ← cond: all_gates
                                   │
                              [gate_generator]            ← cond: all_gates
                                   │
                              [stage_c_lora]              ← cond: all_gates
                                   │
                              [stage_fake_feedback]       ← cond: all_gates
                                   │
    [build_wave_classifier] ───────►[stage_w_wave_classifier]  ← cond: wave_ready
                                   │
                              [gate_wave]                 ← cond: wave_ready
                                   │
                              [sync_gate_replica]
                                   │
                              [checkpoint_save]
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from pipeline.context import PipelineContext
from pipeline.nodes.interrupts import StageStopRequested
from pipeline.graph import EdgeReactionSpec, PipelineGraph
from pipeline.graph_layers import (
    build_execution_program as _materialize_execution_program,
    build_graph_layers as _materialize_graph_layers,
)
from pipeline.plan_protocol import (
    DEFAULT_PLAN_FILENAME,
    DEFAULT_RUNTIME_SNAPSHOT_FILENAME,
    MESSAGE_TYPE_RUNTIME_SNAPSHOT,
    ExecutionEventPayload,
    PredicateGraph,
    PredicateNode,
    RuntimeNodeState,
    RuntimeSnapshotPayload,
    SignalRecord,
    WorkerHelloPayload,
    make_envelope,
    plan_from_pipeline_graph,
    save_protocol_message,
)

# --- node imports ----------------------------------------------------------
from pipeline.nodes.classifier_node import (
    ClassifierConfig,
    BuildClassifierNode,
    PregestationTrainNode,
    GestationTrainNode,
    BerkeleyRefreshTrainNode,
    LoRARoundNode,
    FakeClassFeedbackNode,
    SyncGateReplicaNode,
)
from pipeline.nodes.transformer_node import (
    TransformerConfig,
    ConfigSearchNode,
    BuildTransformerNode,
    TransformerTrainNode,
)
from pipeline.nodes.generator_node import (
    GeneratorConfig,
    BuildGANNode,
    GeneratorTrainNode,
)
from pipeline.nodes.wave_classifier_node import (
    WaveClassifierConfig,
    BuildWaveClassifierNode,
    WaveClassifierTrainNode,
)
from pipeline.nodes.vocab_node import (
    VocabConfig,
    InitVocabNode,
    VocabChurnNode,
    BuildSymbolPoolNode,
    BuildFlashcardRowsNode,
)
from pipeline.nodes.label_embedding_node import (
    LabelEmbeddingConfig,
    BuildLabelEmbeddingNode,
)
from pipeline.nodes.data_nodes import (
    WavePoolConfig,
    WavePoolNode,
    PregestationDataConfig,
    GestationDataConfig,
    BerkeleyPayloadConfig,
    BerkeleyDataConfig,
    DataNode,
)
from pipeline.nodes.gate_nodes import (
    BerkeleyGateConfig,
    BerkeleyGateNode,
    PregestationEvalNode,
    GestationEvalNode,
    TransformerGateConfig,
    TransformerGateNode,
    GeneratorGateConfig,
    GeneratorGateNode,
    WaveGateConfig,
    WaveGateNode,
)
from pipeline.nodes.save_restore_node import SaveRestoreNode
from pipeline.nodes.viewer_ipc_node import ViewerIPCNode


# ---------------------------------------------------------------------------
# Arg / runtime helpers
# ---------------------------------------------------------------------------

_ARG_MISSING = object()


def _arg_value(args, *names: str, default=None):
    if args is None:
        return default
    for name in names:
        if not name:
            continue
        value = getattr(args, name, _ARG_MISSING)
        if value is not _ARG_MISSING and value is not None:
            return value
    return default


def _resolve_device(args) -> torch.device:
    requested = str(_arg_value(args, "device", default="cuda:0") or "cuda:0").strip()
    if requested.lower() == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    if "cuda" in requested.lower() and not torch.cuda.is_available():
        _log(f"[orchestrator] requested CUDA device {requested!r} but CUDA is unavailable; falling back to CPU")
        requested = "cpu"
    return torch.device(requested)


def _resolve_non_training_device(args, device: torch.device) -> torch.device:
    requested = str(_arg_value(args, "non_training_device", default="auto") or "auto").strip()
    lower = requested.lower()
    if lower in ("", "auto", "inherit", "same"):
        return device
    if lower == "cpu":
        return torch.device("cpu")
    if lower == "cuda":
        return device if device.type == "cuda" else torch.device("cpu")
    if "cuda" in lower and not torch.cuda.is_available():
        _log(
            f"[orchestrator] requested non-training CUDA device {requested!r} but CUDA is unavailable; "
            "falling back to CPU"
        )
        return torch.device("cpu")
    try:
        return torch.device(requested)
    except Exception:
        _log(f"[orchestrator] invalid non-training device {requested!r}; falling back to primary device {device}")
        return device


def _load_resume_state(args, output_dir: Path) -> dict:
    from pipeline.nodes.base import _load_json, _torch_load_cpu

    resume_enabled = bool(_arg_value(args, "auto_resume", default=False)) or bool(
        str(_arg_value(args, "resume_from", default="") or "").strip()
    )
    resume_dir = (
        Path(str(_arg_value(args, "resume_from", default="") or "").strip())
        if str(_arg_value(args, "resume_from", default="") or "").strip()
        else Path(output_dir)
    )

    resume_summary = None
    resume_pipeline_ckpt = None
    if resume_enabled:
        summary_path = resume_dir / "summary.json"
        ckpt_path = resume_dir / "pipeline_checkpoint.pt"
        resume_summary = _load_json(summary_path) if summary_path.exists() else None
        if ckpt_path.exists():
            try:
                resume_pipeline_ckpt = _torch_load_cpu(str(ckpt_path))
            except Exception as exc:
                _log(f"[orchestrator] WARNING: could not load resume checkpoint {ckpt_path}: {exc}")
        _log(
            "[orchestrator] resume mode: "
            f"dir={resume_dir} summary={1 if summary_path.exists() else 0} "
            f"ckpt={1 if ckpt_path.exists() else 0}"
        )

    return {
        "enabled": resume_enabled,
        "dir": resume_dir,
        "summary": resume_summary,
        "pipeline_ckpt": resume_pipeline_ckpt,
    }


def _prepare_runtime(args, output_dir: Path, device: torch.device) -> dict:
    from pipeline.utils import (
        _hard_wipe_pipeline_caches,
        _soft_reset_label_caches,
    )
    from wav_ml_models import (
        configure_torch_runtime,
        set_seed,
    )

    set_seed(int(_arg_value(args, "seed", default=1337)))
    configure_torch_runtime(
        device=device,
        cudnn_benchmark=bool(_arg_value(args, "cudnn_benchmark", default=True)),
        allow_tf32=bool(_arg_value(args, "allow_tf32", default=True)),
        matmul_precision=str(_arg_value(args, "matmul_precision", default="high")),
    )

    semantic_cache_nonce = ""
    if bool(_arg_value(args, "hard_wipe_caches", default=False)):
        semantic_cache_nonce = str(int(time.time_ns()))
        wipe_info = _hard_wipe_pipeline_caches(
            output_dir=output_dir,
            berkeley_data_root=str(_arg_value(args, "berkeley_data_root", default="") or ""),
            semantic_stage_cache_dir=str(_arg_value(args, "semantic_stage_cache_dir", default="") or ""),
        )
        _log(
            "[orchestrator] hard wipe: "
            f"removed={len(wipe_info.get('removed', []))} "
            f"errors={len(wipe_info.get('errors', []))}"
        )
    elif bool(_arg_value(args, "soft_reset_labels", default=False)):
        semantic_cache_nonce = str(int(time.time_ns()))
        reset_info = _soft_reset_label_caches(
            output_dir=output_dir,
            berkeley_data_root=str(_arg_value(args, "berkeley_data_root", default="") or ""),
            semantic_stage_cache_dir=str(_arg_value(args, "semantic_stage_cache_dir", default="") or ""),
        )
        _log(
            "[orchestrator] soft label reset: "
            f"removed={len(reset_info.get('removed', []))} "
            f"errors={len(reset_info.get('errors', []))}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "semantic_cache_nonce": semantic_cache_nonce,
        "run_tag": time.strftime("%Y%m%d_%H%M%S"),
        "resume": _load_resume_state(args, output_dir),
    }


def _initialize_loss_logger(ctx: PipelineContext) -> None:
    out_dir = getattr(ctx, "output_dir", None)
    if out_dir is None:
        return
    try:
        from wav_ml_viewer import _LossFileLogger
    except Exception as exc:
        _log(f"[loss-logger] WARNING: could not import loss logger: {exc}")
        return

    loss_log_path = Path(out_dir) / "loss_log.bin"
    loss_log_prev_path = Path(out_dir) / "loss_log_prev.bin"
    try:
        if loss_log_path.exists():
            loss_log_path.replace(loss_log_prev_path)
            _log("[loss-logger] rotated loss_log.bin -> loss_log_prev.bin for new session")
    except Exception as exc:
        _log(f"[loss-logger] WARNING: could not rotate loss log: {exc}")

    try:
        ctx.loss_logger = _LossFileLogger(path=loss_log_path)
    except Exception as exc:
        _log(f"[loss-logger] WARNING: could not open loss log: {exc}")


def _signal_runtime_exit(store: Any, reason: str) -> None:
    if store is None:
        return
    fn = getattr(store, "set_exit_requested", None)
    if not callable(fn):
        return
    try:
        fn(True, reason=str(reason or "training_exit"))
    except Exception:
        pass


def _initialize_runtime_control(ctx: PipelineContext) -> None:
    try:
        import atexit
        from pipeline.nodus_loss_store import NodusRuntimeControlStore

        store = NodusRuntimeControlStore.get_global()
        store.clear()
        store.clear_exit_requested()
        ctx.runtime_control_store = store
        atexit.register(_signal_runtime_exit, store, "training_process_exit")
        _log("[orchestrator] runtime control store connected")
    except Exception as exc:
        _log(f"[orchestrator] WARNING: could not initialize runtime control store: {exc}")


def _restore_context_from_resume(ctx: PipelineContext) -> None:
    from wav_ml_core import RenderConfig

    resume_summary = ctx.resume_summary if isinstance(ctx.resume_summary, dict) else {}
    resume_ckpt = ctx.resume_pipeline_ckpt if isinstance(ctx.resume_pipeline_ckpt, dict) else {}

    if resume_ckpt:
        try:
            ctx.startup_restore_round = max(0, int(resume_ckpt.get("round_id", 0) or 0))
        except Exception:
            ctx.startup_restore_round = 0
        try:
            ctx.startup_restore_cycle = max(0, int(resume_ckpt.get("cycle", 0) or 0))
        except Exception:
            ctx.startup_restore_cycle = 0
        ctx.startup_restore_pending = bool(
            ctx.startup_restore_round > 0
            or ctx.startup_restore_cycle > 0
            or bool(resume_ckpt.get("total_rounds_completed", 0))
        )

    total_rounds = resume_summary.get("total_rounds", resume_ckpt.get("total_rounds_completed", 0))
    try:
        ctx.total_rounds_completed = max(ctx.total_rounds_completed, int(total_rounds))
    except Exception:
        pass

    gate_blob = resume_ckpt.get("gate_status", resume_summary.get("gate_status", resume_summary.get("gates", {})))
    if isinstance(gate_blob, dict):
        ctx.gate_pregestation.passed = bool(gate_blob.get("pregestation", ctx.gate_pregestation.passed))
        ctx.gate_gestation.passed = bool(gate_blob.get("gestation", ctx.gate_gestation.passed))
        ctx.gate_berkeley.passed = bool(gate_blob.get("berkeley", ctx.gate_berkeley.passed))
        ctx.gate_transformer.passed = bool(gate_blob.get("transformer", ctx.gate_transformer.passed))
        ctx.gate_generator.passed = bool(gate_blob.get("generator", ctx.gate_generator.passed))
        ctx.gate_wave.passed = bool(gate_blob.get("wave", ctx.gate_wave.passed))

    metrics = resume_ckpt.get(
        "metrics_history",
        resume_ckpt.get("orchestration_history", resume_summary.get("metrics", [])),
    )
    if isinstance(metrics, list):
        ctx.metrics_history = list(metrics)

    resume_dir = ctx.resume_dir if isinstance(ctx.resume_dir, Path) else None
    if resume_dir is not None:
        _restore_rng_from_resume_dir(resume_dir)

    if ctx.render_config is None and not bool(_arg_value(ctx.args, "force_config_search", default=False)):
        best_cfg_blob = None
        if resume_dir is not None:
            cfg_path = resume_dir / "best_render_config.json"
            if cfg_path.exists():
                try:
                    best_cfg_blob = json.loads(cfg_path.read_text(encoding="utf-8"))
                except Exception:
                    best_cfg_blob = None
        if best_cfg_blob is None:
            best_cfg_blob = resume_ckpt.get("best_cfg")
        if isinstance(best_cfg_blob, dict):
            try:
                ctx.render_config = RenderConfig.from_dict(best_cfg_blob)
                ctx.resumed_from_checkpoint = True
            except Exception as exc:
                _log(f"[orchestrator] WARNING: could not restore render config from resume state: {exc}")


def _restore_rng_from_resume_dir(resume_dir: Path) -> None:
    from pipeline.nodes.base import _torch_load_cpu

    bank_path = Path(resume_dir) / "seed_bank.pt"
    if not bank_path.exists():
        return
    try:
        bank_blob = _torch_load_cpu(str(bank_path))
    except Exception as exc:
        _log(f"[orchestrator] WARNING: could not load resume seed bank {bank_path}: {exc}")
        return
    if not isinstance(bank_blob, list) or not bank_blob:
        return
    latest = bank_blob[-1]
    if not isinstance(latest, dict):
        return

    restored = False
    try:
        python_state = latest.get("python_state")
        if python_state is not None:
            random.setstate(python_state)
            restored = True
    except Exception as exc:
        _log(f"[orchestrator] WARNING: could not restore Python RNG state: {exc}")
    try:
        numpy_state = latest.get("numpy_state")
        if numpy_state is not None:
            np.random.set_state(numpy_state)
            restored = True
    except Exception as exc:
        _log(f"[orchestrator] WARNING: could not restore NumPy RNG state: {exc}")
    try:
        torch_cpu_state = latest.get("torch_cpu_state")
        if torch_cpu_state is not None:
            torch.random.set_rng_state(torch_cpu_state)
            restored = True
    except Exception as exc:
        _log(f"[orchestrator] WARNING: could not restore Torch CPU RNG state: {exc}")
    cuda_states = latest.get("torch_cuda_states")
    if isinstance(cuda_states, dict) and torch.cuda.is_available():
        for dev_idx, state in cuda_states.items():
            try:
                torch.cuda.set_rng_state(state, int(dev_idx))
                restored = True
            except Exception as exc:
                _log(f"[orchestrator] WARNING: could not restore Torch CUDA RNG state for device {dev_idx}: {exc}")
    if restored:
        _log(
            "[orchestrator] restored RNG state from seed bank: "
            f"round={int(latest.get('round_id', 0))} cycle={int(latest.get('cycle', 0))}"
        )


# ---------------------------------------------------------------------------
# Graph plan / GUI IPC helpers
# ---------------------------------------------------------------------------

_CONDITION_ID_GENERATOR_MODE = "orchestration.mode_has_generator"
_CONDITION_ID_PREGESTATION_GATE = "gates.pregestation_passed"
_CONDITION_ID_EARLY_GATES = "gates.early_passed"
_CONDITION_ID_ALL_GATES = "gates.all_base_passed"
_CONDITION_ID_WAVE_STAGE_READY = "gates.wave_stage_ready"
_CONDITION_ID_STARTUP_RESTORE_PENDING = "resume.startup_restore_pending"
_CONDITION_ID_SHUTDOWN_SAVE_PENDING = "runtime.shutdown_save_pending"
# Data-rebuild schedule conditions (wired to the data-node "provides" edges)
_CONDITION_ID_PREG_REBUILD = "data.pregestation_rebuild_due"
_CONDITION_ID_GEST_REBUILD = "data.gestation_rebuild_due"
_CONDITION_ID_BERKELEY_REFRESH = "data.berkeley_refresh_due"


def _worker_id_for_output(output_dir: Path) -> str:
    name = "".join(ch if ch.isalnum() else "_" for ch in output_dir.name.strip()) or "wav_ml_pipeline"
    return f"worker:{name.lower()}"


def _build_condition_blobs() -> dict:
    from pipeline.condition_expr import expr_for_condition_id
    return {
        _CONDITION_ID_GENERATOR_MODE: {
            "label": "Generator mode enabled",
            "description": "True when orchestration_mode includes the generator stage.",
            "callable": "_generator_exists",
            "condition_expr": expr_for_condition_id(_CONDITION_ID_GENERATOR_MODE),
        },
        _CONDITION_ID_PREGESTATION_GATE: {
            "label": "Pregestation gate passed",
            "description": "True when Stage 0 has cleared, or the GUI gate override is active.",
            "callable": "_gate_pregestation_passed",
            "condition_expr": expr_for_condition_id(_CONDITION_ID_PREGESTATION_GATE),
        },
        _CONDITION_ID_EARLY_GATES: {
            "label": "Early gates passed",
            "description": "True when pregestation and gestation have both cleared, or the GUI gate override is active.",
            "callable": "_early_gates_passed",
            "condition_expr": expr_for_condition_id(_CONDITION_ID_EARLY_GATES),
        },
        _CONDITION_ID_ALL_GATES: {
            "label": "All base gates passed",
            "description": "True when pregestation, gestation, and Berkeley have cleared, or the GUI gate override is active.",
            "callable": "_all_gates_passed",
            "condition_expr": expr_for_condition_id(_CONDITION_ID_ALL_GATES),
        },
        _CONDITION_ID_WAVE_STAGE_READY: {
            "label": "Wave stage ready",
            "description": "True when the transformer exists and the upstream gates allow the wave classifier stage.",
            "callable": "_wave_stage_ready",
            "condition_expr": expr_for_condition_id(_CONDITION_ID_WAVE_STAGE_READY),
        },
        _CONDITION_ID_STARTUP_RESTORE_PENDING: {
            "label": "Startup restore pending",
            "description": "True when a resume checkpoint exists and the startup restore has not been consumed yet.",
            "callable": "_startup_restore_pending",
            "condition_expr": expr_for_condition_id(_CONDITION_ID_STARTUP_RESTORE_PENDING),
        },
        _CONDITION_ID_SHUTDOWN_SAVE_PENDING: {
            "label": "Shutdown save pending",
            "description": "True when the GUI requested Stop and Save and the runtime has not flushed the checkpoint yet.",
            "callable": "_shutdown_save_pending",
            "condition_expr": expr_for_condition_id(_CONDITION_ID_SHUTDOWN_SAVE_PENDING),
        },
        # Data-rebuild schedule conditions — these gate the "provides" edges from data_node
        # so the callbacks only fire when a cache rebuild is actually due.
        _CONDITION_ID_PREG_REBUILD: {
            "label": "Pregestation rebuild due",
            "description": "True when pregestation data has never been built or has since expired.",
            "callable": "_pregestation_rebuild_due",
            "condition_expr": "data._preg_needs_rebuild",
        },
        _CONDITION_ID_GEST_REBUILD: {
            "label": "Gestation rebuild due (gated)",
            "description": (
                "True when pregestation gate has cleared AND "
                "gestation data has never been built or has since expired."
            ),
            "callable": "_gestation_rebuild_due",
            "condition_expr": "(GATE_OVERRIDE OR gate_pregestation.passed) AND data._gest_needs_rebuild",
        },
        _CONDITION_ID_BERKELEY_REFRESH: {
            "label": "Berkeley data refresh due (gated)",
            "description": (
                "True when early gates have cleared AND "
                "Berkeley data has never been built or has since expired."
            ),
            "callable": "_berkeley_refresh_due",
            "condition_expr": "(GATE_OVERRIDE OR early_gates_passed) AND data._bdata_needs_rebuild",
        },
    }


def _build_graph_layers(graph: PipelineGraph) -> dict:
    return _materialize_graph_layers(graph)


def _build_execution_program(graph: PipelineGraph) -> dict:
    return _materialize_execution_program(graph)


# ---------------------------------------------------------------------------
# Signal & predicate-graph synthesis  (plan-level IR objects)
# ---------------------------------------------------------------------------

def _build_plan_signals() -> list[SignalRecord]:
    """Mint signal declarations for the training plan.

    These are IR-level descriptors — not runtime values.  The runtime
    ``SignalStore`` on ``ctx.signals`` is initialised from these at startup.
    """
    return [
        SignalRecord(signal_id="gate_pregestation.passed", kind="toggle", initial_value=0.0,
                     metadata={"description": "1.0 when the pregestation gate has cleared"}),
        SignalRecord(signal_id="gate_gestation.passed", kind="toggle", initial_value=0.0,
                     metadata={"description": "1.0 when the gestation gate has cleared"}),
        SignalRecord(signal_id="gate_berkeley.passed", kind="toggle", initial_value=0.0,
                     metadata={"description": "1.0 when the Berkeley gate has cleared"}),
        SignalRecord(signal_id="gate_transformer.passed", kind="toggle", initial_value=0.0,
                     metadata={"description": "1.0 when the transformer gate has cleared"}),
        SignalRecord(signal_id="gate_override_active", kind="toggle", initial_value=0.0,
                     metadata={"description": "1.0 when the GUI gate override is active"}),
    ]


def _build_plan_predicate_graphs() -> list[PredicateGraph]:
    """Build predicate-graph DAGs for data-flow edges.

    Each graph is a serialisable decision tree that evaluates to an
    output pin.  Edges carry ``pin_effects`` which map each pin to
    ``{activate, on_traverse}`` booleans.
    """
    # --- pg:gate_berkeley_data_flow ---
    # Controls the data_node → gate_berkeley edge.  Distinguishes three
    # scenarios: gates not yet passed ("hold"), gates passed but data
    # still fresh ("pass_cached"), gates passed and data stale ("rebuild").
    # Entry node first checks that stage_2_berkeley is selected; if not,
    # short-circuits to "hold" so provide_berkeley_data is never invoked.
    gate_berkeley_data_flow = PredicateGraph(
        graph_id="pg:gate_berkeley_data_flow",
        entry_node_id="check_selected",
        nodes=[
            PredicateNode(
                node_id="check_selected",
                kind="condition",
                condition_expr="stage_2_berkeley_selected",
                branches={"true": "check_gates", "false": "hold"},
            ),
            PredicateNode(
                node_id="check_gates",
                kind="condition",
                condition_expr="GATE_OVERRIDE OR early_gates_passed",
                branches={"true": "check_rebuild", "false": "hold"},
            ),
            PredicateNode(
                node_id="check_rebuild",
                kind="condition",
                condition_expr="data._bdata_needs_rebuild",
                branches={"true": "rebuild", "false": "pass_cached"},
            ),
            PredicateNode(node_id="hold", kind="terminal", output_pin="hold"),
            PredicateNode(node_id="pass_cached", kind="terminal", output_pin="pass_cached"),
            PredicateNode(node_id="rebuild", kind="terminal", output_pin="rebuild"),
        ],
        output_pins=["hold", "pass_cached", "rebuild"],
        metadata={"description": "Data-flow gate for data_node → gate_berkeley"},
    )

    # --- pg:berkeley_refresh_flow ---
    # Controls the data_node → stage_2_berkeley edge.  Same three-way
    # split so stage 2 still trains on cached loaders between refreshes.
    # Entry node first checks that stage_2_berkeley is selected; if not,
    # short-circuits to "hold" so provide_berkeley_data is never invoked.
    berkeley_refresh_flow = PredicateGraph(
        graph_id="pg:berkeley_refresh_flow",
        entry_node_id="check_selected",
        nodes=[
            PredicateNode(
                node_id="check_selected",
                kind="condition",
                condition_expr="stage_2_berkeley_selected",
                branches={"true": "check_gates", "false": "hold"},
            ),
            PredicateNode(
                node_id="check_gates",
                kind="condition",
                condition_expr="GATE_OVERRIDE OR early_gates_passed",
                branches={"true": "check_rebuild", "false": "hold"},
            ),
            PredicateNode(
                node_id="check_rebuild",
                kind="condition",
                condition_expr="data._bdata_needs_rebuild",
                branches={"true": "rebuild", "false": "pass_cached"},
            ),
            PredicateNode(node_id="hold", kind="terminal", output_pin="hold"),
            PredicateNode(node_id="pass_cached", kind="terminal", output_pin="pass_cached"),
            PredicateNode(node_id="rebuild", kind="terminal", output_pin="rebuild"),
        ],
        output_pins=["hold", "pass_cached", "rebuild"],
        metadata={"description": "Data-flow gate for data_node → stage_2_berkeley"},
    )

    return [gate_berkeley_data_flow, berkeley_refresh_flow]


# Standard pin_effects map shared by data-flow predicate graphs.
_DATA_FLOW_PIN_EFFECTS: dict[str, dict[str, bool]] = {
    "hold":        {"activate": False, "on_traverse": False},
    "pass_cached": {"activate": True,  "on_traverse": False},
    "rebuild":     {"activate": True,  "on_traverse": True},
}


def _node_group_id(node_id: str) -> str:
    if node_id in {"wave_pool"}:
        return "bootstrap"
    if node_id.startswith("gate_"):
        return "gates"
    if node_id in {"sync_gate_replica", "checkpoint_save", "viewer_ipc"}:
        return "housekeeping"
    if node_id.endswith("_data") or node_id in {"data_node", "berkeley_payload", "payload_validation_data"}:
        return "data"
    if node_id.startswith("stage_"):
        return "train"
    if node_id in {"init_vocab", "vocab_churn", "build_symbol_pool", "build_label_embedding", "build_flashcard_rows"}:
        return "vocab"
    if node_id in {
        "build_classifier",
        "config_search",
        "build_transformer",
        "build_gan",
        "build_wave_classifier",
    }:
        return "build"
    return "other"


def _node_faculty(node_id: str) -> str:
    return _node_group_id(node_id)


def _node_archetype(node_id: str) -> str:
    if node_id == "data_node":
        return "storage_authority"
    if node_id.startswith("gate_"):
        return "gate"
    if node_id.startswith("stage_"):
        return "stage"
    if node_id.startswith("build_"):
        return "builder"
    if node_id in {"wave_pool", "init_vocab", "vocab_churn"}:
        return "seed"
    if node_id in {"build_symbol_pool", "build_label_embedding", "build_flashcard_rows"}:
        return "materializer"
    if node_id in {"sync_gate_replica", "checkpoint_save", "viewer_ipc"}:
        return "housekeeping"
    return "node"


def _edge_schedule_reaction_name(label: str) -> str:
    mapping = {
        "startup": "schedule.bootstrap",
        "per_round": "schedule.round",
        "end_of_round": "schedule.finalize",
    }
    return mapping.get(str(label or "").strip(), "schedule.transition")


def _edge_provided_resources(method_name: str, label: str) -> list[str]:
    mapping = {
        "provide_pregestation": ["pregestation_loader", "pregestation_dataset"],
        "provide_pregestation_eval": ["pregestation_eval_loader", "pregestation_eval_dataset"],
        "provide_gestation": ["gestation_loader", "gestation_dataset"],
        "provide_gestation_eval": ["gestation_eval_loader", "gestation_eval_dataset"],
        "provide_berkeley_data": ["berkeley_refresh_loader", "berkeley_gate_val_loader", "berkeley_cache"],
        "provide_payload": ["payload_bank", "payload_conditions", "payload_masks", "payload_bank_ready"],
        "provide_payload_validation": ["payload_validation_loader", "payload_validation_dataset"],
        "provide_gate_data": [
            "payload_validation_loader",
            "payload_validation_dataset",
        ],
    }
    resources = list(mapping.get(str(method_name or ""), []))
    if resources:
        return resources
    label_text = str(label or "")
    if label_text.startswith("provides:"):
        return [label_text.split(":", 1)[1]]
    return []


def _apply_execution_layer_metadata(graph: PipelineGraph) -> None:
    raw_edges = getattr(graph, "_edges", None)
    if not isinstance(raw_edges, list):
        return

    for edge in raw_edges:
        if str(getattr(edge, "layer", "execution") or "execution") not in ("execution", ""):
            continue
        edge.layer = "execution"
        edge.metadata = dict(getattr(edge, "metadata", {}) or {})
        edge.metadata.setdefault("execution_layer", True)

        if edge.on_traverse is not None:
            method_name = getattr(getattr(edge.on_traverse, "__func__", edge.on_traverse), "__name__", "")
            owner = getattr(edge.on_traverse, "__self__", None)
            owner_name = type(owner).__name__ if owner is not None else ""
            is_data_provider = owner_name == "DataNode"
            defaults = {
                "consumer": str(edge.target_id),
                "resources": _edge_provided_resources(method_name, edge.label),
            }
            if edge.condition_id:
                defaults["condition_id"] = str(edge.condition_id)
            edge.reaction = EdgeReactionSpec(
                reaction_name=("data.provide" if is_data_provider else "runtime.prepare"),
                target_function=(
                    f"{owner_name}.{method_name}" if owner_name and method_name else method_name or "runtime.prepare"
                ),
                defaults=defaults,
                layer="execution",
                metadata={
                    "template": ("data.provide" if is_data_provider else "runtime.prepare"),
                    "source_node_id": str(edge.source_id),
                    "target_node_id": str(edge.target_id),
                },
            )
            continue

        phase = str(edge.label or "flow")
        defaults = {
            "phase": phase,
            "target_node_id": str(edge.target_id),
        }
        if edge.condition_id:
            defaults["condition_id"] = str(edge.condition_id)
        edge.reaction = EdgeReactionSpec(
            reaction_name=_edge_schedule_reaction_name(phase),
            target_function="scheduler.activate",
            defaults=defaults,
            layer="execution",
            metadata={
                "template": _edge_schedule_reaction_name(phase),
                "source_node_id": str(edge.source_id),
                "target_node_id": str(edge.target_id),
            },
        )


def _node_icon(node_id: str) -> str:
    if "wave" in node_id:
        return "wave"
    if "vocab" in node_id or "symbol" in node_id or "flashcard" in node_id:
        return "vocab"
    if "transformer" in node_id or "config_search" in node_id:
        return "transformer"
    if "gan" in node_id or "generator" in node_id:
        return "generator"
    if "classifier" in node_id or "berkeley" in node_id or "lora" in node_id or "fake_feedback" in node_id:
        return "classifier"
    if node_id == "data_node" or node_id.endswith("_data") or "payload" in node_id:
        return "data"
    if node_id.startswith("gate_"):
        return "gate"
    if node_id == "checkpoint_save":
        return "checkpoint"
    if node_id == "sync_gate_replica":
        return "sync"
    if node_id == "viewer_ipc":
        return "ipc"
    return "node"


def _node_config_id(node_id: str) -> str:
    mapping = {
        "wave_pool": "wave_pool",
        "init_vocab": "vocab",
        "vocab_churn": "vocab",
        "build_symbol_pool": "vocab",
        "build_label_embedding": "embedding",
        "build_flashcard_rows": "vocab",
        "data_node": "data",
        "pregestation_data": "pregestation",
        "gestation_data": "gestation",
        "berkeley_payload": "berkeley_payload",
        "berkeley_data": "berkeley_data",
        "build_classifier": "classifier",
        "stage_0_pregestation": "classifier",
        "stage_1_gestation": "classifier",
        "stage_2_berkeley": "classifier",
        "stage_c_lora": "classifier",
        "stage_fake_feedback": "classifier",
        "config_search": "transformer",
        "build_transformer": "transformer",
        "stage_r_transformer": "transformer",
        "build_gan": "generator",
        "stage_g_generator": "generator",
        "build_wave_classifier": "wave",
        "stage_w_wave_classifier": "wave",
        "gate_0_pregestation_eval": "classifier",
        "gate_1_gestation_eval": "classifier",
        "gate_berkeley": "berkeley_gate",
        "gate_transformer": "transformer_gate",
        "gate_generator": "generator_gate",
        "gate_wave": "wave_gate",
        "checkpoint_save": "checkpoint",
        "viewer_ipc": "viewer_ipc",
    }
    return mapping.get(node_id, "")


def _build_plan_node_metadata(graph: PipelineGraph) -> dict:
    node_metadata = {}
    for node_id, node in graph.nodes.items():
        node_metadata[node_id] = {
            "label": str(getattr(node, "description", "") or node_id.replace("_", " ").title()),
            "icon": _node_icon(node_id),
            "config_id": _node_config_id(node_id),
            "group_id": _node_group_id(node_id),
            "object_type": "object",
            "faculty": _node_faculty(node_id),
            "archetype": _node_archetype(node_id),
            "layer": "execution",
            "metadata": {
                "node_id": node_id,
                "description": str(getattr(node, "description", "") or ""),
            },
        }
    return node_metadata


def _make_viewer_proxy(args, cycles: int):
    port_file = str(_arg_value(args, "viewer_port_file", default="") or "").strip()
    if not port_file:
        return None
    required = bool(_arg_value(args, "stage_opengl_preview_required", default=False))
    try:
        from wav_ml_viewer import ViewerIPCProxy

        image_size = int(_arg_value(args, "image_size", default=128))
        proxy = ViewerIPCProxy(
            port_file=port_file,
            enabled=True,
            image_hw=(image_size, image_size),
            scale=max(1, int(_arg_value(args, "stage_opengl_preview_scale", "transformer_viz_scale", default=1))),
            cycle_slots=max(0, int(cycles)),
        )
        if required and (proxy is None or not bool(getattr(proxy, "enabled", False))):
            raise RuntimeError(
                "stage OpenGL preview is required, but the standalone GUI IPC connection "
                f"could not be established via {port_file!r}"
            )
        return proxy
    except Exception as exc:
        if required:
            raise
        _log(f"[orchestrator] WARNING: viewer IPC setup failed: {exc}")
        return None


def build_training_graph_plan(args, output_dir: Path, *, graph: Optional[PipelineGraph] = None, cfg: Optional[dict] = None):
    cfg = cfg if cfg is not None else _build_configs_from_args(args)
    if graph is None:
        graph = build_pipeline_graph(
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
            save_every_n_rounds=int(_arg_value(args, "checkpoint_every_round", "save_every_n_rounds", default=1)),
            berkeley_refresh_every_n_rounds=int(_arg_value(args, "berkeley_refresh_round_every", "berkeley_refresh_every", default=4)),
        )

    worker_hints = {
        "output_dir": str(Path(output_dir)),
        "device_preference": str(_arg_value(args, "device", default="auto") or "auto"),
        "non_training_device_preference": str(_arg_value(args, "non_training_device", default="auto") or "auto"),
        "orchestration_mode": str(_arg_value(args, "orchestration_mode", default="staged_cgrw") or "staged_cgrw"),
        "orchestration_cycles": int(_arg_value(args, "orchestration_cycles", "cycles", default=1)),
        "orchestration_rounds": int(_arg_value(args, "orchestration_rounds", "rounds_per_cycle", default=1)),
        "entrypoint": "wav_pipeline_graph.py",
        # Capability flags — the GUI reads these to know which IPC messages the
        # worker will act on.  Add a new flag here when a new handler is wired.
        "capabilities": [
            "plan_apply",    # worker applies graph swaps between cycles
            "run_control",   # worker honours stop/resume/gate_override
            "pause",         # worker honours pause/play toggle
            "preview_toggle",  # worker honours preview on/off
            "scrub_editor_toggle",  # worker honours scrub editor on/off
        ],
    }
    metadata = {
        "graph_name": graph.name,
        "graph_summary": graph.summary(),
        "generated_by": "pipeline.orchestrator.build_training_graph_plan",
    }
    return plan_from_pipeline_graph(
        graph,
        name="WAV ML Training Plan",
        revision=1,
        config_blobs=cfg,
        condition_blobs=_build_condition_blobs(),
        worker_hints=worker_hints,
        metadata=metadata,
        node_metadata=_build_plan_node_metadata(graph),
        graph_layers=_build_graph_layers(graph),
        execution_program=_build_execution_program(graph),
        signals=_build_plan_signals(),
        predicate_graphs=_build_plan_predicate_graphs(),
    )


# ---------------------------------------------------------------------------
# Phase 3: plan-driven worker bootstrap
# ---------------------------------------------------------------------------

# Maps the config_blobs key used in TrainingGraphPlan → the config dataclass type.
# This is the full registry of all 15 config dataclasses used by build_pipeline_graph().
_CONFIG_CLASS_REGISTRY = {
    "classifier":        ClassifierConfig,
    "transformer":       TransformerConfig,
    "generator":         GeneratorConfig,
    "wave":              WaveClassifierConfig,
    "vocab":             VocabConfig,
    "embedding":         LabelEmbeddingConfig,
    "wave_pool":         WavePoolConfig,
    "pregestation":      PregestationDataConfig,
    "gestation":         GestationDataConfig,
    "berkeley_payload":  BerkeleyPayloadConfig,
    "berkeley_data":     BerkeleyDataConfig,
    "berkeley_gate":     BerkeleyGateConfig,
    "transformer_gate":  TransformerGateConfig,
    "generator_gate":    GeneratorGateConfig,
    "wave_gate":         WaveGateConfig,
}


def _reconstruct_config(cls, blob: dict):
    """Reconstruct a dataclass config instance from a plain dict.

    Unknown keys in *blob* are silently ignored so that a plan created with a
    newer version of the code can still be loaded by an older worker (forward
    compatibility).  Missing keys fall back to the dataclass field defaults.
    """
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in blob.items() if k in valid_fields})


def _parse_positive_int(value, *, field_name: str, default: int) -> int:
    """Parse integer worker hints with clear validation errors."""
    if value is None:
        return int(default)
    try:
        parsed = int(value)
    except Exception as exc:
        raise ValueError(f"Invalid plan worker_hints[{field_name!r}]={value!r}: expected integer") from exc
    if parsed < 1:
        raise ValueError(f"Invalid plan worker_hints[{field_name!r}]={value!r}: expected >= 1")
    return parsed


def _validate_training_graph_plan(plan) -> None:
    """Validate minimally required plan fields before materializing nodes."""
    if plan is None:
        raise ValueError("TrainingGraphPlan is required")

    blobs = getattr(plan, "config_blobs", None)
    if blobs is not None and not isinstance(blobs, dict):
        raise ValueError("Invalid plan: config_blobs must be a mapping")

    hints = dict(getattr(plan, "worker_hints", {}) or {})
    _parse_positive_int(hints.get("save_every_n_rounds", 1), field_name="save_every_n_rounds", default=1)
    _parse_positive_int(
        hints.get("berkeley_refresh_every_n_rounds", 4),
        field_name="berkeley_refresh_every_n_rounds",
        default=4,
    )


def _execution_plan_nodes(plan) -> list[Any]:
    nodes = [
        node
        for node in list(getattr(plan, "nodes", []) or [])
        if bool(getattr(node, "enabled", True))
        and str(getattr(node, "layer", "execution") or "execution") == "execution"
    ]
    return nodes if nodes else [node for node in list(getattr(plan, "nodes", []) or []) if bool(getattr(node, "enabled", True))]


def _execution_plan_edges(plan) -> list[Any]:
    edges = [
        edge
        for edge in list(getattr(plan, "edges", []) or [])
        if bool(getattr(edge, "enabled", True))
        and str(getattr(edge, "layer", "execution") or "execution") == "execution"
    ]
    return edges if edges else [edge for edge in list(getattr(plan, "edges", []) or []) if bool(getattr(edge, "enabled", True))]


def _condition_for_plan_edge(condition_id: str, *, condition_blobs: dict | None = None, condition_expr: str = ""):
    registry = {
        _CONDITION_ID_GENERATOR_MODE: _generator_exists,
        _CONDITION_ID_PREGESTATION_GATE: _gate_pregestation_passed,
        _CONDITION_ID_EARLY_GATES: _early_gates_passed,
        _CONDITION_ID_ALL_GATES: _all_gates_passed,
        _CONDITION_ID_WAVE_STAGE_READY: _wave_stage_ready,
    }
    resolved_id = str(condition_id or "").strip()
    resolved_expr = str(condition_expr or "").strip()

    # Registry functions take precedence for known condition IDs — they implement
    # the full runtime logic including stage-deselection auto-bypass which the
    # condition_expr grammar cannot express.  condition_expr is only used for
    # conditions that have no registered Python implementation.
    if resolved_id in registry:
        return registry[resolved_id]

    # Portable, IR-first path for conditions not in the registry.
    if resolved_expr:
        from pipeline.condition_expr import evaluate_condition_expr
        def _expr_condition(ctx, _e=resolved_expr):
            return evaluate_condition_expr(_e, ctx)
        return _expr_condition

    if not resolved_id:
        return None
    blob = dict((condition_blobs or {}).get(resolved_id, {}) or {})

    # Check if the blob carries a condition_expr.
    blob_expr = str(blob.get("condition_expr", "") or "").strip()
    if blob_expr:
        from pipeline.condition_expr import evaluate_condition_expr
        def _blob_expr_condition(ctx, _e=blob_expr):
            return evaluate_condition_expr(_e, ctx)
        return _blob_expr_condition

    predicate_name = str(blob.get("callable_ref", "") or blob.get("callable", "") or "").rsplit(".", 1)[-1].strip()
    if predicate_name:
        def _plan_condition(ctx, _name=predicate_name):
            fn = getattr(ctx, _name, None)
            return bool(fn()) if callable(fn) else False
        return _plan_condition
    raise ValueError(f"Unsupported plan edge condition_id {resolved_id!r}")


def _resolve_callback_from_ref(
    graph: PipelineGraph,
    callable_ref: str,
    owner_node_id: str,
    *,
    edge_id: str = "",
    strict: bool = False,
):
    """Resolve a callable_ref to an actual method on a graph node."""
    owner_ref = str(callable_ref or "").rsplit(".", 1)[0].strip()
    method_name = str(callable_ref or "").rsplit(".", 1)[-1].strip()
    source_id = str(owner_node_id or "").strip()
    if not method_name or not source_id:
        if strict:
            raise ValueError(f"Plan edge {edge_id!r} is missing callback metadata")
        return None
    source_node = graph.nodes.get(source_id)
    if source_node is None:
        if strict:
            raise ValueError(f"Plan edge references unknown callback source node {source_id!r}")
        return None
    callback = getattr(source_node, method_name, None)
    if callback is None and owner_ref:
        owner_token = owner_ref.rsplit(".", 1)[-1].strip()
        for candidate_id, candidate_node in graph.nodes.items():
            if str(type(candidate_node).__name__) != owner_token:
                continue
            candidate_callback = getattr(candidate_node, method_name, None)
            if candidate_callback is None:
                continue
            return candidate_callback
    if callback is None and strict:
        raise ValueError(
            f"Plan edge {edge_id!r} expects callback {callable_ref!r} "
            f"but node {source_id!r} does not provide it"
        )
    return callback


def _resolve_plan_edge_callback(graph: PipelineGraph, edge_record, *, action_map: dict | None = None):
    """Resolve the on_traverse callback for a plan edge.

    Resolution order:
      1. Action registry (action_id → callable_ref)
      2. Legacy edge metadata (reaction_name + target_function)
    """
    edge_id_str = str(getattr(edge_record, "edge_id", "<unknown>"))
    action_map = dict(action_map or {})
    action_id = str(getattr(edge_record, "action_id", "") or "").strip()

    if action_id and action_id in action_map:
        action = action_map[action_id]
        callable_ref = str(getattr(action, "callable_ref", "") or "").strip()
        owner_id = str(
            getattr(action, "owner_node_id", "") or getattr(edge_record, "source_node_id", "") or ""
        ).strip()
        if callable_ref and owner_id:
            resolved = _resolve_callback_from_ref(graph, callable_ref, owner_id, edge_id=edge_id_str)
            if resolved is not None:
                return resolved

    reaction_name = str(getattr(edge_record, "reaction_name", "") or "").strip()
    target_function = str(getattr(edge_record, "target_function", "") or "").strip()
    if reaction_name not in {"data.provide", "runtime.prepare"}:
        return None
    source_id = str(getattr(edge_record, "source_node_id", "") or "").strip()
    return _resolve_callback_from_ref(
        graph, target_function, source_id, edge_id=edge_id_str, strict=True,
    )


def _plan_edge_label(edge_record) -> str:
    metadata = dict(getattr(edge_record, "metadata", {}) or {})
    return str(metadata.get("label", getattr(edge_record, "kind", "")) or getattr(edge_record, "kind", "") or "")


def build_training_graph_from_plan(plan) -> PipelineGraph:
    """Materialize an executable PipelineGraph from a saved TrainingGraphPlan.

    The plan's execution node/edge records are the authoritative topology.
    Config blobs still reconstruct the node objects, but the saved execution
    layer decides which nodes exist and how they are connected.
    """
    blobs = dict(plan.config_blobs or {})
    cfg: dict = {}

    for key, cls in _CONFIG_CLASS_REGISTRY.items():
        blob = blobs.get(key)
        if isinstance(blob, dict) and blob and not (set(blob.keys()) == {"value"}):
            cfg[key] = _reconstruct_config(cls, blob)
        else:
            cfg[key] = cls()

    _validate_training_graph_plan(plan)
    hints = dict(plan.worker_hints or {})
    save_every = _parse_positive_int(hints.get("save_every_n_rounds", 1), field_name="save_every_n_rounds", default=1)
    berkeley_refresh = _parse_positive_int(
        hints.get("berkeley_refresh_every_n_rounds", 4),
        field_name="berkeley_refresh_every_n_rounds",
        default=4,
    )

    base_graph = build_pipeline_graph(
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
        save_every_n_rounds=save_every,
        berkeley_refresh_every_n_rounds=berkeley_refresh,
    )

    available_nodes = base_graph.nodes
    selected_nodes = _execution_plan_nodes(plan)
    selected_edges = _execution_plan_edges(plan)
    action_map = {
        a.action_id: a
        for a in (getattr(plan, "actions", None) or [])
    }
    condition_blobs = dict(getattr(plan, "condition_blobs", {}) or {})
    graph_name = str(getattr(plan, "name", "") or getattr(base_graph, "name", "wav_ml_pipeline"))
    graph = PipelineGraph(name=graph_name)

    selected_node_ids: set[str] = set()
    for node_record in selected_nodes:
        node_id = str(getattr(node_record, "node_id", "") or "").strip()
        node = available_nodes.get(node_id)
        if node is None:
            raise ValueError(f"Plan references unsupported execution node {node_id!r}")
        graph.add_node(node)
        selected_node_ids.add(node_id)

    for edge_record in selected_edges:
        source_id = str(getattr(edge_record, "source_node_id", "") or "").strip()
        target_id = str(getattr(edge_record, "target_node_id", "") or "").strip()
        if source_id not in selected_node_ids or target_id not in selected_node_ids:
            raise ValueError(
                f"Plan edge {getattr(edge_record, 'edge_id', '<unknown>')!r} references nodes outside the execution node set"
            )
        edge_metadata = dict(getattr(edge_record, "metadata", {}) or {})
        graph.add_edge(
            source_id,
            target_id,
            condition=_condition_for_plan_edge(
                str(getattr(edge_record, "condition_id", "") or ""),
                condition_blobs=condition_blobs,
                condition_expr=str(getattr(edge_record, "condition_expr", "") or ""),
            ),
            label=_plan_edge_label(edge_record),
            condition_id=str(getattr(edge_record, "condition_id", "") or ""),
            on_traverse=_resolve_plan_edge_callback(graph, edge_record, action_map=action_map),
            edge_id=str(getattr(edge_record, "edge_id", "") or ""),
            layer=str(getattr(edge_record, "layer", "execution") or "execution"),
            target_function=str(getattr(edge_record, "target_function", "") or ""),
            reaction_name=str(getattr(edge_record, "reaction_name", "") or ""),
            reaction_defaults=dict(getattr(edge_record, "reaction_defaults", {}) or {}),
            reaction_metadata=edge_metadata,
            metadata=edge_metadata,
        )

    return graph


def _gate_status_blob(ctx: PipelineContext) -> dict:
    return {
        "pregestation": ctx.gate_pregestation.passed,
        "gestation": ctx.gate_gestation.passed,
        "berkeley": ctx.gate_berkeley.passed,
        "transformer": ctx.gate_transformer.passed,
        "generator": ctx.gate_generator.passed,
        "wave": ctx.gate_wave.passed,
    }


def _status_bucket(status: str) -> str:
    if status == "ran":
        return "completed"
    if str(status).startswith("skipped"):
        return "skipped"
    if str(status).startswith("failed"):
        return "failed"
    return str(status)


def _build_runtime_snapshot(
    ctx: PipelineContext,
    statuses: dict,
    execution_state: str,
    *,
    default_cycle_ids: Optional[list[int]] = None,
) -> RuntimeSnapshotPayload:
    selected_cycles = ctx.selected_cycle_ids() or list(default_cycle_ids or [])
    node_states = [
        RuntimeNodeState(
            node_id=str(node_id),
            status=_status_bucket(str(status)),
            last_result=str(status),
        )
        for node_id, status in statuses.items()
    ]
    active_node_ids = [str(node_id) for node_id, status in statuses.items() if status == "ran"]
    return RuntimeSnapshotPayload(
        worker_id=str(ctx.worker_id),
        execution_state=str(execution_state),
        active_node_ids=active_node_ids,
        selected_cycle_ids=selected_cycles,
        gate_override=ctx.gate_override_enabled(),
        node_states=node_states,
        metrics={
            "cycle": int(ctx.cycle),
            "round": int(ctx.round_id),
            "total_rounds_completed": int(ctx.total_rounds_completed),
            "gates": _gate_status_blob(ctx),
            "class_count": int(len(ctx.class_names)),
            "metrics_history_size": int(len(ctx.metrics_history)),
            "schedule_index": int(getattr(ctx, "schedule_index", -1)),
            "schedule_row_label": str(getattr(ctx, "schedule_row_label", "")),
            "schedule_total_rows": int(len(getattr(getattr(ctx, "schedule", None), "rows", []) or [])),
        },
    )


def _save_runtime_snapshot(ctx: PipelineContext, snapshot: RuntimeSnapshotPayload) -> None:
    if ctx.runtime_snapshot_path is None:
        return
    envelope = make_envelope(
        MESSAGE_TYPE_RUNTIME_SNAPSHOT,
        snapshot,
        session_id=ctx.session_id,
        worker_id=ctx.worker_id,
        plan_id=ctx.plan_id,
        revision=int(getattr(ctx.graph_plan, "revision", 0) or 0),
    )
    try:
        save_protocol_message(ctx.runtime_snapshot_path, envelope)
    except Exception as exc:
        _log(f"[orchestrator] WARNING: could not write runtime snapshot: {exc}")
    viewer = ctx.viewer_proxy
    if viewer is None:
        return
    send_runtime = getattr(viewer, "send_runtime_snapshot", None)
    if not callable(send_runtime):
        return
    try:
        send_runtime(
            snapshot,
            session_id=ctx.session_id,
            worker_id=ctx.worker_id,
            plan_id=ctx.plan_id,
            revision=int(getattr(ctx.graph_plan, "revision", 0) or 0),
        )
    except Exception as exc:
        _log(f"[orchestrator] WARNING: could not send runtime snapshot to GUI: {exc}")


def _send_viewer_bootstrap(ctx: PipelineContext) -> None:
    viewer = ctx.viewer_proxy
    if viewer is None or ctx.graph_plan is None:
        return
    try:
        hello = WorkerHelloPayload(
            worker_id=ctx.worker_id,
            worker_label=str(ctx.output_dir.name if ctx.output_dir is not None else "wav_ml_pipeline"),
            capabilities={
                "graph_plan": True,
                "runtime_snapshot": True,
                "execution_event": True,
                "plan_apply": True,
                "plan_patch": False,
                "plan_path": str(ctx.graph_plan_path) if ctx.graph_plan_path is not None else "",
            },
            loaded_plan_id=ctx.plan_id,
            loaded_revision=int(getattr(ctx.graph_plan, "revision", 0) or 0),
            execution_state="initializing",
        )
        send_hello = getattr(viewer, "send_worker_hello", None)
        if callable(send_hello):
            send_hello(
                hello,
                session_id=ctx.session_id,
                plan_id=ctx.plan_id,
                revision=int(getattr(ctx.graph_plan, "revision", 0) or 0),
            )

        send_plan = getattr(viewer, "send_plan_snapshot", None)
        if callable(send_plan):
            send_plan(
                ctx.graph_plan,
                session_id=ctx.session_id,
                worker_id=ctx.worker_id,
                revision=int(getattr(ctx.graph_plan, "revision", 0) or 0),
            )
    except Exception as exc:
        _log(f"[orchestrator] WARNING: could not send graph bootstrap to GUI: {exc}")


def _emit_execution_event(
    ctx: PipelineContext,
    *,
    event_id: str,
    phase: str,
    status: str,
    message: str,
    metrics: Optional[dict] = None,
) -> None:
    viewer = ctx.viewer_proxy
    if viewer is None:
        return
    send_event = getattr(viewer, "send_execution_event", None)
    if not callable(send_event):
        return
    try:
        send_event(
            ExecutionEventPayload(
                event_id=event_id,
                kind="orchestration",
                phase=phase,
                status=status,
                message=message,
                ts=float(time.time()),
                metrics=dict(metrics or {}),
            ),
            session_id=ctx.session_id,
            worker_id=ctx.worker_id,
            plan_id=ctx.plan_id,
            revision=int(getattr(ctx.graph_plan, "revision", 0) or 0),
        )
    except Exception as exc:
        _log(f"[orchestrator] WARNING: could not send execution event to GUI: {exc}")


# ---------------------------------------------------------------------------
# Schedule helpers
# ---------------------------------------------------------------------------


def _apply_pending_schedule(ctx: PipelineContext) -> bool:
    """Consume any queued schedule_apply from the GUI and store it on ctx.

    Returns True if a new schedule was applied.
    """
    proxy = ctx.viewer_proxy
    if proxy is None:
        return False
    consume_fn = getattr(proxy, "consume_pending_schedule", None)
    if not callable(consume_fn):
        return False
    try:
        payload = consume_fn()
    except Exception:
        return False
    if payload is None:
        return False
    from pipeline.plan_protocol import ScheduleApplyPayload
    if not isinstance(payload, ScheduleApplyPayload):
        return False
    new_schedule = payload.schedule
    if not new_schedule or not new_schedule.rows:
        return False
    ctx.schedule = new_schedule
    _log(
        f"[orchestrator] schedule_apply: {len(new_schedule.rows)} row(s) "
        f"id={new_schedule.schedule_id!r} reason={payload.reason!r}"
    )
    return True


def _apply_row_config(ctx: PipelineContext, row: Any, row_idx: int) -> None:
    """Activate a schedule row: update stage filter, config overrides, and
    gate_override on the viewer proxy.
    """
    from pipeline.plan_protocol import ScheduleRow
    ctx.schedule_index = int(row_idx)
    ctx.schedule_row_label = str(getattr(row, "label", "") or f"row_{row_idx}")

    # Stage filter — frozenset or None (None = all)
    stages = list(getattr(row, "active_stages", []) or [])
    ctx.schedule_active_stages = frozenset(str(s) for s in stages) if stages else None

    # Config overrides — replace entirely from this row
    overrides = dict(getattr(row, "config_overrides", {}) or {})
    ctx.node_config_overrides = {str(k): dict(v) for k, v in overrides.items()}

    # Push gate_override into proxy if available
    gate_override = bool(getattr(row, "gate_override", False))
    proxy = ctx.viewer_proxy
    if proxy is not None:
        try:
            proxy._gate_override = gate_override
        except Exception:
            pass

    _log(
        f"[orchestrator] schedule row {row_idx}: label={ctx.schedule_row_label!r} "
        f"cycles={getattr(row, 'cycles', 1)} rounds={getattr(row, 'rounds_per_cycle', 1)} "
        f"stages={stages or 'all'} overrides={list(overrides.keys())}"
    )


# ---------------------------------------------------------------------------
# Edge condition lambdas
# ---------------------------------------------------------------------------


def _gate_override_enabled(ctx: PipelineContext) -> bool:
    return ctx.gate_override_enabled()


def _stage_deselected(ctx: PipelineContext, node_id: str) -> bool:
    """True when the GUI or schedule has explicitly deselected *node_id*."""
    try:
        return not bool(ctx.is_node_selected(node_id))
    except Exception:
        return False


def _gate_pregestation_passed(ctx: PipelineContext) -> bool:
    return ctx.gate_effectively_passed("gate_pregestation")


def _early_gates_passed(ctx: PipelineContext) -> bool:
    return ctx.early_gates_passed()


def _all_gates_passed(ctx: PipelineContext) -> bool:
    return ctx.all_base_gates_passed()

def _wave_stage_ready(ctx: PipelineContext) -> bool:
    return ctx.wave_stage_ready()

def _generator_exists(ctx: PipelineContext) -> bool:
    mode = str(ctx.orchestration_mode or getattr(ctx.args, "orchestration_mode", "")).lower()
    return "g" in mode

def _startup_restore_pending(ctx: PipelineContext) -> bool:
    return bool(getattr(ctx, "startup_restore_pending", False))

def _shutdown_save_pending(ctx: PipelineContext) -> bool:
    return bool(getattr(ctx, "shutdown_save_pending", False))

def _make_preg_rebuild_cond(data_node):
    """Return an edge condition that fires when pregestation data needs a (re)build."""
    def _cond(ctx: PipelineContext) -> bool:
        return data_node._preg_needs_rebuild
    _cond.__name__ = "_pregestation_rebuild_due"
    return _cond


def _make_gest_rebuild_cond(data_node):
    """Return an edge condition that fires when pregestation gate has cleared AND gestation data needs a (re)build."""
    def _cond(ctx: PipelineContext) -> bool:
        if not _gate_pregestation_passed(ctx):
            return False
        return data_node._gest_needs_rebuild
    _cond.__name__ = "_gestation_rebuild_due"
    return _cond


def _make_berk_refresh_cond(data_node):
    """Return an edge condition that fires when early gates have cleared AND berkeley data needs a (re)build."""
    def _cond(ctx: PipelineContext) -> bool:
        if not _early_gates_passed(ctx):
            return False
        return data_node._bdata_needs_rebuild
    _cond.__name__ = "_berkeley_refresh_due"
    return _cond


def _always(ctx: PipelineContext) -> bool:
    return True


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_pipeline_graph(
    classifier_cfg: ClassifierConfig,
    transformer_cfg: TransformerConfig,
    generator_cfg: GeneratorConfig,
    wave_cfg: WaveClassifierConfig,
    vocab_cfg: VocabConfig,
    embedding_cfg: LabelEmbeddingConfig,
    wave_pool_cfg: WavePoolConfig,
    pregestation_cfg: PregestationDataConfig,
    gestation_cfg: GestationDataConfig,
    berkeley_payload_cfg: BerkeleyPayloadConfig,
    berkeley_data_cfg: BerkeleyDataConfig,
    berkeley_gate_cfg: BerkeleyGateConfig,
    transformer_gate_cfg: TransformerGateConfig,
    generator_gate_cfg: GeneratorGateConfig,
    wave_gate_cfg: WaveGateConfig,
    save_every_n_rounds: int = 1,
    berkeley_refresh_every_n_rounds: int = 4,
) -> PipelineGraph:
    """Construct and return the fully-wired pipeline graph.

    All nodes are registered; all edges with their conditions are defined here.
    The returned graph is stateless — the context carries all mutable state.
    """
    g = PipelineGraph(name="wav_ml_pipeline")

    # ---------------------------------------------------------------
    # Register all nodes
    # ---------------------------------------------------------------

    # Initialisation (run-once)
    g.add_node(WavePoolNode(wave_pool_cfg))
    g.add_node(InitVocabNode(vocab_cfg))
    g.add_node(BuildClassifierNode(classifier_cfg))
    g.add_node(ConfigSearchNode(transformer_cfg))
    g.add_node(BuildTransformerNode(transformer_cfg))
    g.add_node(BuildGANNode(generator_cfg))
    g.add_node(BuildWaveClassifierNode(wave_cfg))

    # Per-round vocab / embedding
    g.add_node(VocabChurnNode(vocab_cfg))
    g.add_node(BuildSymbolPoolNode(vocab_cfg))
    g.add_node(BuildLabelEmbeddingNode(embedding_cfg))
    g.add_node(BuildFlashcardRowsNode(vocab_cfg))

    # Data (single node providing all training dataloaders)
    _data_node = DataNode(
        preg_cfg=pregestation_cfg,
        gest_cfg=gestation_cfg,
        payload_cfg=berkeley_payload_cfg,
        bdata_cfg=berkeley_data_cfg,
    )
    g.add_node(_data_node)

    # Edge-condition closures for data-rebuild scheduling.
    # These mirror the DataPossession.expiry_fn logic so the IR topology
    # is the single authoritative description of when caches are rebuilt.
    _preg_rebuild_cond = _make_preg_rebuild_cond(_data_node)
    _gest_rebuild_cond = _make_gest_rebuild_cond(_data_node)
    _berk_refresh_cond = _make_berk_refresh_cond(_data_node)

    # Training stages
    g.add_node(PregestationTrainNode(classifier_cfg))
    g.add_node(GestationTrainNode(classifier_cfg))
    g.add_node(BerkeleyRefreshTrainNode(classifier_cfg))
    g.add_node(TransformerTrainNode(transformer_cfg))
    g.add_node(GeneratorTrainNode(generator_cfg))
    g.add_node(WaveClassifierTrainNode(wave_cfg))
    g.add_node(LoRARoundNode(classifier_cfg))
    g.add_node(FakeClassFeedbackNode(classifier_cfg))

    # Gate checks
    g.add_node(PregestationEvalNode(classifier_cfg))
    g.add_node(GestationEvalNode(classifier_cfg))
    g.add_node(BerkeleyGateNode(berkeley_gate_cfg))
    g.add_node(TransformerGateNode(transformer_gate_cfg))
    g.add_node(GeneratorGateNode(generator_gate_cfg, generator_cfg=generator_cfg))
    g.add_node(WaveGateNode(wave_gate_cfg))

    # Housekeeping
    g.add_node(SyncGateReplicaNode(classifier_cfg))
    _save_restore_node = SaveRestoreNode(save_every_n_rounds)
    g.add_node(_save_restore_node)
    g.add_node(ViewerIPCNode())

    # ---------------------------------------------------------------
    # Edge sequences  (source → target, optional condition)
    # ---------------------------------------------------------------

    # == Initialisation chain (roots → downstream) ====================

    # Wave pool feeds into config search and training stages (indirectly via ctx)
    g.add_edge("wave_pool", "init_vocab", label="startup")
    g.add_edge("init_vocab", "build_classifier", label="startup")
    g.add_edge("build_classifier", "config_search", label="startup")
    g.add_edge("config_search", "build_transformer", label="startup")
    g.add_edge("build_transformer", "build_gan",
               condition=_generator_exists, label="if_gan_mode", condition_id=_CONDITION_ID_GENERATOR_MODE)
    g.add_edge("wave_pool", "build_wave_classifier", label="startup")

    # == Per-round vocab / embedding ================================

    g.add_edge("build_classifier", "vocab_churn", label="per_round")
    g.add_edge("vocab_churn", "build_symbol_pool", label="per_round")
    g.add_edge("build_symbol_pool", "build_label_embedding", label="per_round")

    # == DataNode feeds all training stages =============================
    # DataNode runs after label embedding (needs latest vocab/embeddings)
    g.add_edge("build_label_embedding", "data_node", label="per_round")

    # Pregestation: data_node rebuilds ctx.pregestation_loader when the rebuild period is due.
    # The conditional edge fires the rebuild callback only on schedule; the unconditional
    # per-round edge keeps the path open so stage_0 trains on the cached loader every round.
    g.add_edge("data_node", "stage_0_pregestation",
               condition=_preg_rebuild_cond, label="provides:pregestation_loader",
               condition_id=_CONDITION_ID_PREG_REBUILD,
               on_traverse=_data_node.provide_pregestation)
    g.add_edge("data_node", "stage_0_pregestation", label="per_round")
    g.add_edge("data_node", "gate_0_pregestation_eval",
               condition=_preg_rebuild_cond, label="provides:pregestation_eval_loader",
               condition_id=_CONDITION_ID_PREG_REBUILD,
               on_traverse=_data_node.provide_pregestation_eval)
    g.add_edge("data_node", "gate_0_pregestation_eval", label="per_round")

    # Gestation: conditional edge fires provide_gestation when a rebuild is due;
    # unconditional per_round edge keeps the path open every round (matches stage_0 pattern).
    # Without per_round, data.gestation_rebuild_due enters the execution-program guard and
    # prevents stage_1_gestation from running after the first round.
    g.add_edge("data_node", "stage_1_gestation",
               condition=_gest_rebuild_cond, label="provides:gestation_loader",
               condition_id=_CONDITION_ID_GEST_REBUILD,
               on_traverse=_data_node.provide_gestation)
    g.add_edge("data_node", "stage_1_gestation", label="per_round")
    g.add_edge("data_node", "gate_1_gestation_eval",
               condition=_gest_rebuild_cond, label="provides:gestation_eval_loader",
               condition_id=_CONDITION_ID_GEST_REBUILD,
               on_traverse=_data_node.provide_gestation_eval)
    g.add_edge("data_node", "gate_1_gestation_eval", label="per_round")

    # Berkeley refresh: predicate graph governs activate vs on_traverse so
    # stage 2 still trains on cached loaders between refresh intervals.
    # Without per_round, data.berkeley_refresh_due enters the execution-program guard and
    # prevents stage_2_berkeley from running after the first data build (same issue as stage_1_gestation).
    g.add_edge("data_node", "stage_2_berkeley",
               condition=_berk_refresh_cond, label="provides:berkeley_refresh_loader",
               condition_id=_CONDITION_ID_BERKELEY_REFRESH,
               on_traverse=_data_node.provide_berkeley_data,
               predicate_graph_id="pg:berkeley_refresh_flow",
               pin_effects=_DATA_FLOW_PIN_EFFECTS)
    g.add_edge("data_node", "stage_2_berkeley", label="per_round")

    # Gate 2 eval: data_node provides berkeley loaders + payload validation loader.
    # Predicate graph distinguishes hold / pass_cached / rebuild so we don't
    # reconstruct data every round under gate override.
    g.add_edge("data_node", "gate_berkeley",
               condition=_early_gates_passed, label="provides:gate_val_loader+payload_val_loader",
               condition_id=_CONDITION_ID_EARLY_GATES,
               on_traverse=_data_node.provide_gate_data,
               predicate_graph_id="pg:gate_berkeley_data_flow",
               pin_effects=_DATA_FLOW_PIN_EFFECTS)

    # GAN training: data_node provides ctx.payload_bank + ctx.payload_conditions (after all gates)
    g.add_edge("data_node", "stage_g_generator",
               condition=_all_gates_passed, label="provides:payload_bank",
               condition_id=_CONDITION_ID_ALL_GATES,
               on_traverse=_data_node.provide_payload)

    # Flashcard prep sits immediately before the generator path so payload work
    # does not begin near the top of the round merely because later stages are enabled.
    # The data edge still provisions payload JIT, but only when this support node is next.
    g.add_edge("data_node", "build_flashcard_rows",
               condition=_all_gates_passed,
               condition_id=_CONDITION_ID_ALL_GATES,
               label="provides:payload_images",
               on_traverse=_data_node.provide_payload)

    # == Stage 0 / 1 gate evals =====================================

    g.add_edge("stage_0_pregestation", "gate_0_pregestation_eval", label="after_stage0")
    g.add_edge("gate_0_pregestation_eval", "stage_1_gestation",
               condition=_gate_pregestation_passed, label="after_gate0", condition_id=_CONDITION_ID_PREGESTATION_GATE)
    g.add_edge("stage_1_gestation", "gate_1_gestation_eval",
               condition=_gate_pregestation_passed, label="after_stage1", condition_id=_CONDITION_ID_PREGESTATION_GATE)
    g.add_edge("gate_1_gestation_eval", "stage_2_berkeley",
               condition=_early_gates_passed, label="after_gate1", condition_id=_CONDITION_ID_EARLY_GATES)

    # == Stage 2 → gate_berkeley ====================================

    g.add_edge("stage_2_berkeley", "gate_berkeley",
               condition=_early_gates_passed, label="after_gate1", condition_id=_CONDITION_ID_EARLY_GATES)

    # == Stage R — Transformer =====================================

    g.add_edge("gate_berkeley", "stage_r_transformer",
               condition=_early_gates_passed, label="after_gate1", condition_id=_CONDITION_ID_EARLY_GATES)
    g.add_edge("stage_r_transformer", "gate_transformer",
               condition=_early_gates_passed, label="after_gate1", condition_id=_CONDITION_ID_EARLY_GATES)

    # == Stage G — Generator =======================================

    g.add_edge("gate_transformer", "build_flashcard_rows",
               condition=_all_gates_passed, label="before_generator", condition_id=_CONDITION_ID_ALL_GATES)
    g.add_edge("build_flashcard_rows", "stage_g_generator",
               condition=_all_gates_passed, label="flashcards_ready", condition_id=_CONDITION_ID_ALL_GATES)
    g.add_edge("gate_transformer", "stage_g_generator",
               condition=_all_gates_passed, label="after_all_gates", condition_id=_CONDITION_ID_ALL_GATES)
    g.add_edge("stage_g_generator", "gate_generator",
               condition=_all_gates_passed, label="after_all_gates", condition_id=_CONDITION_ID_ALL_GATES)

    # == Stage C — LoRA + fake-class ================================

    g.add_edge("gate_berkeley", "stage_c_lora",
               condition=_all_gates_passed, label="after_all_gates", condition_id=_CONDITION_ID_ALL_GATES)
    g.add_edge("stage_c_lora", "stage_fake_feedback",
               condition=_all_gates_passed, label="after_all_gates", condition_id=_CONDITION_ID_ALL_GATES)

    # == Stage W — Wave classifier ==================================

    g.add_edge("build_wave_classifier", "stage_w_wave_classifier",
               condition=_wave_stage_ready, label="after_transformer_gate", condition_id=_CONDITION_ID_WAVE_STAGE_READY)
    g.add_edge("stage_w_wave_classifier", "gate_wave",
               condition=_wave_stage_ready, label="after_transformer_gate", condition_id=_CONDITION_ID_WAVE_STAGE_READY)

    # == Housekeeping at end of every round =========================

    g.add_edge("gate_wave", "sync_gate_replica", label="end_of_round")
    g.add_edge("gate_transformer", "sync_gate_replica",
               condition=_always, label="end_of_round",
               condition_id=_CONDITION_ID_EARLY_GATES)
    g.add_edge("gate_berkeley", "sync_gate_replica",
               condition=_always, label="end_of_round",
               condition_id=_CONDITION_ID_EARLY_GATES)
    g.add_edge(
        "sync_gate_replica",
        "checkpoint_save",
        label="end_of_round",
        on_traverse=_save_restore_node.prepare_runtime_checkpoint,
    )
    g.add_edge("checkpoint_save", "viewer_ipc", label="end_of_round")

    # Runtime control edges — viewer_ipc broadcasts pause/preview/scrub
    # state back into nodes that honour those toggles.
    g.add_edge("viewer_ipc", "checkpoint_save",
               label="scrub_editor",
               layer="runtime_control",
               metadata={"style_role": "runtime_control"})

    _apply_execution_layer_metadata(g)
    return g


def _runtime_execution_program(plan) -> dict:
    return dict(getattr(plan, "execution_program", {}) or {}) if plan is not None else {}


def _program_condition_resolver(condition_id: str, ctx: PipelineContext) -> bool:
    """Fallback resolver when no plan is available."""
    condition = _condition_for_plan_edge(condition_id)
    if condition is None:
        return True
    return bool(condition(ctx))


def _make_condition_resolver(plan):
    """Build a plan-aware condition resolver that prefers condition_expr from the IR."""
    expr_map: dict[str, str] = {}
    blobs: dict = {}
    if plan is not None:
        for edge in getattr(plan, "edges", []) or []:
            cid = (getattr(edge, "condition_id", None)
                   or (edge.get("condition_id") if isinstance(edge, dict) else "")
                   or "")
            expr = (getattr(edge, "condition_expr", None)
                    or (edge.get("condition_expr") if isinstance(edge, dict) else "")
                    or "")
            if cid and expr and cid not in expr_map:
                expr_map[cid] = expr
        blobs = dict(getattr(plan, "condition_blobs", {}) or {})

    def _resolver(condition_id: str, ctx: PipelineContext) -> bool:
        expr = expr_map.get(condition_id, "")
        condition = _condition_for_plan_edge(condition_id, condition_blobs=blobs, condition_expr=expr)
        if condition is None:
            return True
        return bool(condition(ctx))

    return _resolver


def _runtime_node_sequence(graph: PipelineGraph, plan) -> list[str]:
    execution_program = _runtime_execution_program(plan)
    if execution_program:
        return graph.build_sequence_from_program(execution_program)
    return graph.build_sequence()


def _execute_runtime_pass(graph: PipelineGraph, ctx: PipelineContext, *, sequence: Optional[list[str]] = None) -> dict[str, str]:
    plan = getattr(ctx, "graph_plan", None)
    execution_program = _runtime_execution_program(plan)
    if execution_program:
        resolver = _make_condition_resolver(plan)
        return graph.execute_program(
            ctx,
            execution_program,
            condition_resolver=resolver,
        )
    return graph.execute_sequence(ctx, sequence=sequence)


def _cleanup_context_resources(ctx: PipelineContext) -> None:
    """Release DataLoader workers and free GPU/CPU memory after a stop (no save)."""
    import gc
    loader_attrs = [
        "pregestation_loader", "pregestation_dataset",
        "pregestation_eval_loader", "pregestation_eval_dataset",
        "pregestation_logic_rows",
        "gestation_loader", "gestation_dataset",
        "gestation_eval_loader", "gestation_eval_dataset",
        "berkeley_refresh_loader", "berkeley_gate_val_loader", "berkeley_cache",
    ]
    for attr in loader_attrs:
        try:
            setattr(ctx, attr, None)
        except Exception:
            pass
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    _log("[orchestrator] context resources released after stop")


def _run_shutdown_save_pass(graph: PipelineGraph, ctx: PipelineContext, *, sequence: Optional[list[str]] = None) -> Optional[dict[str, str]]:
    try:
        ctx.shutdown_save_pending = True
        statuses = _execute_runtime_pass(graph, ctx, sequence=sequence)
        ctx.last_node_statuses = dict(statuses)
        _log_statuses("shutdown-save", statuses)
        return statuses
    except StageStopRequested as exc:
        _log(f"[orchestrator] shutdown-save interrupted before checkpoint completed: {exc}")
    except Exception as exc:
        _log(f"[orchestrator] WARNING: stop-save checkpoint failed: {exc}")
    finally:
        ctx.shutdown_save_pending = False
    return None


def _execute_runtime_pass_handling_stop(
    graph: PipelineGraph,
    ctx: PipelineContext,
    *,
    sequence: Optional[list[str]] = None,
    stop_context: str,
) -> tuple[Optional[dict[str, str]], bool]:
    try:
        statuses = _execute_runtime_pass(graph, ctx, sequence=sequence)
        ctx.last_node_statuses = dict(statuses)
        # Training may have set stop_now and returned normally (no exception).
        # Check the flag so we don't start another round.
        if ctx.stop_requested():
            save_on_stop = False
            try:
                save_on_stop = bool(ctx.shutdown_save())
            except Exception:
                save_on_stop = False
            _log(f"[orchestrator] stop detected after {stop_context} (save={save_on_stop})")
            if save_on_stop:
                _run_shutdown_save_pass(graph, ctx, sequence=sequence)
            else:
                _cleanup_context_resources(ctx)
            return None, True
        return statuses, False
    except StageStopRequested as exc:
        save_on_stop = bool(getattr(exc, "save_requested", False))
        if not save_on_stop:
            try:
                save_on_stop = bool(ctx.shutdown_save())
            except Exception:
                save_on_stop = False
        _log(f"[orchestrator] GUI requested stop during {stop_context} (save={save_on_stop})")
        if save_on_stop:
            _run_shutdown_save_pass(graph, ctx, sequence=sequence)
        else:
            _cleanup_context_resources(ctx)
        return None, True


# ---------------------------------------------------------------------------
# Execution loop
# ---------------------------------------------------------------------------

def run(args, output_dir: Path, initial_plan=None) -> None:
    """Main execution entry point called from the CLI shim.

    Builds the pipeline context, constructs node configs from parsed args,
    assembles the graph, and runs cycles × rounds_per_cycle.
    """

    output_dir = Path(output_dir)

    # -- Device / runtime -------------------------------------------------
    device = _resolve_device(args)
    non_training_device = _resolve_non_training_device(args, device)
    runtime = _prepare_runtime(args, output_dir, device)
    _log(f"[orchestrator] device={device} non_training_device={non_training_device}")

    # -- Context ----------------------------------------------------------
    ctx = PipelineContext(
        args=args,
        device=device,
        non_training_device=non_training_device,
        non_training_device_preference=str(_arg_value(args, "non_training_device", default="auto") or "auto"),
        output_dir=output_dir,
        berkeley_data_root=str(_arg_value(args, "berkeley_data_root", default="") or ""),
        semantic_stage_cache_dir=str(_arg_value(args, "semantic_stage_cache_dir", default="") or ""),
        amp_enabled=bool(_arg_value(args, "amp", default=False)),
        resume_dir=runtime["resume"]["dir"],
        resume_summary=runtime["resume"]["summary"],
        resume_pipeline_ckpt=runtime["resume"]["pipeline_ckpt"],
        run_tag=str(runtime["run_tag"]),
        semantic_cache_nonce=str(runtime["semantic_cache_nonce"]),
    )
    ctx.worker_id = _worker_id_for_output(output_dir)
    ctx.session_id = f"{ctx.worker_id}:{ctx.run_tag or int(time.time())}"
    ctx.graph_plan_path = output_dir / DEFAULT_PLAN_FILENAME
    ctx.runtime_snapshot_path = output_dir / DEFAULT_RUNTIME_SNAPSHOT_FILENAME
    if ctx.amp_enabled:
        from pipeline.nodes.base import resolve_amp_dtype
        ctx.amp_dtype = resolve_amp_dtype(str(_arg_value(args, "amp_dtype", default="float16")))

    _restore_context_from_resume(ctx)
    _initialize_runtime_control(ctx)

    # -- GPU residence manager (VRAM budget enforcement) ------------------
    stage_offload = bool(_arg_value(args, "stage_module_offload", default=False))
    offload_empty_cache = bool(_arg_value(args, "stage_module_offload_empty_cache", default=False))
    vram_limit = int(_arg_value(args, "vram_limit_mb", default=0) or 0)
    max_resident = int(_arg_value(args, "max_resident_models", default=0) or 0)
    if stage_offload or vram_limit > 0 or max_resident > 0:
        from pipeline.nodes.base import GPUResidenceManager
        ctx.gpu_residence = GPUResidenceManager(
            max_models=max_resident if max_resident > 0 else 4,
            max_bytes=vram_limit * 1048576 if vram_limit > 0 else 0,
            empty_cache=offload_empty_cache,
        )
        _log(
            f"[orchestrator] GPU residence manager: "
            f"max_models={ctx.gpu_residence.max_models} "
            f"max_bytes={ctx.gpu_residence.max_bytes} "
            f"empty_cache={ctx.gpu_residence.empty_cache}"
        )

    # -- Node configs / orchestration parameters --------------------------
    plan_hints = dict(getattr(initial_plan, "worker_hints", {}) or {}) if initial_plan is not None else {}
    cfg = _build_configs_from_args(args)
    cycles = _parse_positive_int(
        plan_hints.get("orchestration_cycles", _arg_value(args, "orchestration_cycles", "cycles", default=1)),
        field_name="orchestration_cycles",
        default=1,
    )
    rounds_per_cycle = _parse_positive_int(
        plan_hints.get("orchestration_rounds", _arg_value(args, "orchestration_rounds", "rounds_per_cycle", default=1)),
        field_name="orchestration_rounds",
        default=1,
    )
    default_cycle_ids = [int(i) for i in range(1, cycles + 1)]

    # Store orchestration parameters in context so nodes can read them
    # without falling back to ctx.args (needed for plan-driven mode).
    ctx.orchestration_mode = str(plan_hints.get("orchestration_mode", _arg_value(args, "orchestration_mode", default="staged_cgrw")) or "staged_cgrw")
    ctx.orchestration_cycles = cycles
    ctx.orchestration_rounds = rounds_per_cycle
    ctx.training_preview_topk = int(_arg_value(args, "training_preview_topk", default=6))
    ctx.viewer_proxy = _make_viewer_proxy(args, cycles)
    _initialize_loss_logger(ctx)
    if ctx.viewer_proxy is not None:
        set_cycle_roster = getattr(ctx.viewer_proxy, "set_cycle_roster", None)
        if callable(set_cycle_roster):
            try:
                set_cycle_roster(total_cycles=cycles)
            except Exception as exc:
                _log(f"[orchestrator] WARNING: could not initialize GUI cycle roster: {exc}")

    # -- Graph construction -----------------------------------------------
    if initial_plan is not None:
        graph = build_training_graph_from_plan(initial_plan)
        ctx.graph_plan = initial_plan
    else:
        graph = build_pipeline_graph(
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
            save_every_n_rounds=int(_arg_value(args, "checkpoint_every_round", "save_every_n_rounds", default=1)),
            berkeley_refresh_every_n_rounds=int(_arg_value(args, "berkeley_refresh_round_every", "berkeley_refresh_every", default=4)),
        )
        ctx.graph_plan = build_training_graph_plan(args, output_dir, graph=graph, cfg=cfg)
    ctx.plan_id = str(ctx.graph_plan.plan_id)
    ctx.graph_layers = dict(getattr(ctx.graph_plan, "graph_layers", {}) or {})

    # -- Initialise signals and predicate graphs from the plan IR ----------
    plan_signals = list(getattr(ctx.graph_plan, "signals", []) or [])
    if plan_signals:
        ctx.signals.init_from_records(plan_signals)
    plan_pgs = list(getattr(ctx.graph_plan, "predicate_graphs", []) or [])
    ctx.predicate_graphs = {pg.graph_id: pg for pg in plan_pgs}

    if ctx.graph_plan_path is not None:
        try:
            ctx.graph_plan.save_json(ctx.graph_plan_path)
        except Exception as exc:
            _log(f"[orchestrator] WARNING: could not write training graph plan: {exc}")

    # -- IR-authoritative: instantiate CycleGate from plan's cycle edges --
    # The cycle edges in the IR *cause* CycleGate objects to exist.
    # These objects own iteration state and make the repeat/exhaust
    # decision — no hardcoded loop counters anywhere.
    from pipeline.graph import CycleGate
    _cycle_gates = CycleGate.from_plan(ctx.graph_plan)
    ctx.cycle_gates = _cycle_gates
    if _cycle_gates:
        _gate = _cycle_gates[0]
        cycles = _gate.cycles
        rounds_per_cycle = _gate.rounds_per_cycle
        ctx.orchestration_cycles = cycles
        ctx.orchestration_rounds = rounds_per_cycle
        default_cycle_ids = [int(i) for i in range(1, cycles + 1)]
        _log(f"[orchestrator] CycleGate instantiated from IR: {_gate!r}")

    _send_viewer_bootstrap(ctx)

    # -- Wire SaveRestoreNode into context and viewer proxy ---------------
    _sr_node = graph.nodes.get("checkpoint_save")
    if isinstance(_sr_node, SaveRestoreNode):
        _sr_node.initialise(ctx)
        ctx.save_restore_node = _sr_node
        if ctx.viewer_proxy is not None:
            _set_sr = getattr(ctx.viewer_proxy, "set_save_restore_node", None)
            if callable(_set_sr):
                _set_sr(_sr_node)
            # Wire restore: when the GUI clicks RESTORE STATE, translate scrub
            # offset into the round/cycle that was checkpointed there and
            # schedule a restore on the SaveRestoreNode.
            def _restore_from_scrub(scrub_offset: int) -> None:
                """Convert a visual scrub offset to (round,cycle) and request restore."""
                if _sr_node.seed_bank is not None:
                    snap = _sr_node.seed_bank.latest()
                    if snap is not None:
                        _sr_node.request_restore(snap.round_id, snap.cycle)
                        _log(f"[orchestrator] restore requested: offset={scrub_offset} "
                             f"→ round={snap.round_id} cycle={snap.cycle}")
            _on_restore = getattr(ctx.viewer_proxy, "_on_restore", None)
            if _on_restore is None:
                # IPC proxy uses _on_restore callback
                ctx.viewer_proxy._on_restore = _restore_from_scrub

    # -- Wire DataNode references into context for condition_expr ----------
    _data_node_ref = graph.nodes.get("data_node")
    if _data_node_ref is not None:
        ctx.data = _data_node_ref
        ctx.preg_cfg = getattr(_data_node_ref, "preg_cfg", None)
        ctx.gest_cfg = getattr(_data_node_ref, "gest_cfg", None)
        ctx.bdata_cfg = getattr(_data_node_ref, "bdata_cfg", None)

    # -- Wire ViewerIPCNode -----------------------------------------------
    _ipc_node = graph.nodes.get("viewer_ipc")
    if isinstance(_ipc_node, ViewerIPCNode) and ctx.viewer_proxy is not None:
        _ipc_node.wire(ctx.viewer_proxy, _sr_node if isinstance(_sr_node, SaveRestoreNode) else None)
        _ipc_node.start_background_pump()

    _log(graph.summary())

    # Build a fixed runtime sequence once from the persisted execution program.
    sequence = _runtime_node_sequence(graph, ctx.graph_plan)

    # -- Startup pause: hold here until the GUI releases play -------------
    # The proxy (and GUI) default to paused=True so stage/eval options can
    # be configured before any work begins.  Without a viewer proxy this
    # exits immediately and the run proceeds as normal.
    from pipeline.graph import _wait_while_paused
    _wait_while_paused(ctx)

    # -- One-time initialisation pass (no loop) ---------------------------
    # The first pass runs init_vocab, build_classifier, wave_pool, etc.
    # Subsequent passes will hit the `_done` guards on one-shot nodes.
    statuses, interrupted = _execute_runtime_pass_handling_stop(
        graph,
        ctx,
        sequence=sequence,
        stop_context="initialization",
    )
    if interrupted or statuses is None:
        _signal_runtime_exit(getattr(ctx, "runtime_control_store", None), "training_initialization_interrupted")
        return
    _log_statuses("init", statuses)
    _save_runtime_snapshot(
        ctx,
        _build_runtime_snapshot(ctx, statuses, "initializing", default_cycle_ids=default_cycle_ids),
    )
    _emit_execution_event(
        ctx,
        event_id=f"{ctx.session_id}:init",
        phase="init",
        status="ok",
        message="Initial graph bootstrap executed.",
        metrics={
            "ran": sum(1 for s in statuses.values() if s == "ran"),
            "skipped": sum(1 for s in statuses.values() if str(s).startswith("skipped")),
            "failed": sum(1 for s in statuses.values() if str(s).startswith("failed")),
        },
    )

    # -- Apply any schedule sent during startup pause, or build default ----
    _apply_pending_schedule(ctx)
    if ctx.schedule is None:
        from pipeline.plan_protocol import TrainingSchedule
        ctx.schedule = TrainingSchedule.default_from_plan(ctx.graph_plan)
        _log(f"[orchestrator] no schedule provided — using default "
             f"({len(ctx.schedule.rows)} row, "
             f"{ctx.schedule.rows[0].cycles}c × {ctx.schedule.rows[0].rounds_per_cycle}r)")

    # -- Outer orchestration loop — schedule-driven -----------------------
    # Walk the TrainingSchedule row by row.  Each row instantiates a fresh
    # CycleGate from its own cycles/rounds spec.  The original plan-derived
    # CycleGates (_cycle_gates) are kept as a fallback for plan_apply mid-run.
    stop_requested = False
    _primary_gate = _cycle_gates[0] if _cycle_gates else None

    schedule_rows = list(ctx.schedule.rows)
    row_idx = 0

    while row_idx < len(schedule_rows):
        row = schedule_rows[row_idx]
        _apply_row_config(ctx, row, row_idx)
        _emit_execution_event(
            ctx,
            event_id=f"{ctx.session_id}:schedule:row:{row_idx}:start",
            phase="schedule_row",
            status="start",
            message=f"Schedule row {row_idx}: {ctx.schedule_row_label!r}",
            metrics={"row_index": row_idx, "cycles": row.cycles,
                     "rounds_per_cycle": row.rounds_per_cycle},
        )

        # Build a CycleGate for this row
        from pipeline.graph import CycleGate
        row_max_iters = max(1, int(row.cycles) * max(1, int(row.rounds_per_cycle)))
        _row_gate = CycleGate(
            edge_id=f"schedule_row_{row_idx}",
            source_node_id="schedule",
            target_node_id="schedule",
            cycle_control={
                "max_iterations": row_max_iters,
                "cycles": row.cycles,
                "rounds_per_cycle": row.rounds_per_cycle,
            },
        )

        while True:
            if ctx.stop_requested():
                save_on_stop = ctx.shutdown_save()
                _log(f"[orchestrator] GUI requested stop (row={row_idx}, save={save_on_stop})")
                stop_requested = True
                if save_on_stop:
                    _run_shutdown_save_pass(graph, ctx, sequence=sequence)
                else:
                    _cleanup_context_resources(ctx)
                break

            # Row gate advance — exhaust when row's cycles×rounds are done
            branch = _row_gate.evaluate(ctx)
            if branch == "exhaust":
                break
            cycle_idx = ctx.cycle
            round_idx = ctx.round_id

            # Mid-process pause: between any two cycles
            from pipeline.graph import _wait_while_paused
            _wait_while_paused(ctx)
            if ctx.stop_requested():
                continue  # will be caught at top of inner while

            # READ POINT A — apply a new plan or schedule received from GUI
            if ctx.viewer_proxy is not None:
                apply_payload = ctx.viewer_proxy.consume_pending_plan_apply()
                if apply_payload is not None:
                    _log(
                        f"[orchestrator] plan_apply received before cycle {cycle_idx}: "
                        f"reason={apply_payload.reason!r}"
                    )
                    try:
                        new_graph = build_training_graph_from_plan(apply_payload.plan)
                        graph = new_graph
                        sequence = _runtime_node_sequence(new_graph, apply_payload.plan)
                        ctx.graph_plan = apply_payload.plan
                        ctx.plan_id = str(apply_payload.plan.plan_id)
                        ctx.graph_layers = dict(getattr(apply_payload.plan, "graph_layers", {}) or {})
                        _cycle_gates = CycleGate.from_plan(apply_payload.plan)
                        ctx.cycle_gates = _cycle_gates
                        _primary_gate = _cycle_gates[0] if _cycle_gates else None
                        if ctx.graph_plan_path is not None:
                            try:
                                apply_payload.plan.save_json(ctx.graph_plan_path)
                            except Exception as _save_exc:
                                _log(f"[orchestrator] WARNING: could not persist applied plan: {_save_exc}")
                        _log(f"[orchestrator] plan_apply complete: {len(sequence)} nodes in new sequence")
                        _send_viewer_bootstrap(ctx)
                    except Exception as apply_exc:
                        _log(f"[orchestrator] ERROR: plan_apply failed — keeping existing graph: {apply_exc}")

                # READ POINT B — schedule_apply replaces the schedule list;
                # complete the current row first, then reload from new schedule.
                if _apply_pending_schedule(ctx):
                    schedule_rows = list(ctx.schedule.rows)
                    _log(f"[orchestrator] schedule replaced mid-run: "
                         f"{len(schedule_rows)} row(s); completing current row then restarting")

            if not ctx.is_cycle_selected(cycle_idx):
                deselected = {node_id: "skipped:cycle_deselected" for node_id in sequence}
                ctx.last_node_statuses = dict(deselected)
                _log(f"[orchestrator] cycle={cycle_idx} skipped by GUI selection")
                _save_runtime_snapshot(
                    ctx,
                    _build_runtime_snapshot(ctx, deselected, "idle", default_cycle_ids=default_cycle_ids),
                )
                _emit_execution_event(
                    ctx,
                    event_id=f"{ctx.session_id}:cycle:{cycle_idx}:skipped",
                    phase="cycle",
                    status="skipped",
                    message=f"Cycle {cycle_idx} skipped by GUI selection.",
                    metrics={"cycle": int(cycle_idx), "row_index": row_idx},
                )
                continue

            _log(f"\n{'=' * 60}")
            _log(f"[orchestrator] row={row_idx}/{len(schedule_rows)-1}  "
                 f"cycle={cycle_idx}/{row.cycles}  round={round_idx}/{row.rounds_per_cycle}"
                 f"  total={ctx.total_rounds_completed}")
            _log(f"{'=' * 60}")

            statuses, interrupted = _execute_runtime_pass_handling_stop(
                graph,
                ctx,
                sequence=sequence,
                stop_context=f"row={row_idx} cycle={cycle_idx} round={round_idx}",
            )
            if interrupted or statuses is None:
                stop_requested = True
                break
            _log_statuses(f"r{row_idx}c{cycle_idx}r{round_idx}", statuses)
            _save_runtime_snapshot(
                ctx,
                _build_runtime_snapshot(ctx, statuses, "running", default_cycle_ids=default_cycle_ids),
            )
            _emit_execution_event(
                ctx,
                event_id=f"{ctx.session_id}:row:{row_idx}:cycle:{cycle_idx}:round:{round_idx}",
                phase="round",
                status="ok",
                message=f"Row {row_idx} cycle {cycle_idx} round {round_idx}.",
                metrics={
                    "row_index": row_idx,
                    "cycle": int(cycle_idx),
                    "round": int(round_idx),
                    "ran": sum(1 for s in statuses.values() if s == "ran"),
                    "skipped": sum(1 for s in statuses.values() if str(s).startswith("skipped")),
                    "failed": sum(1 for s in statuses.values() if str(s).startswith("failed")),
                },
            )

            if ctx.vocab_rotation_cycle == 0:
                ctx.vocab_rotation_cycle = 1

        # Inner loop ended — either exhausted, stopped, or stop was set
        if stop_requested:
            break

        # Row complete
        _emit_execution_event(
            ctx,
            event_id=f"{ctx.session_id}:schedule:row:{row_idx}:complete",
            phase="schedule_row",
            status="complete",
            message=f"Schedule row {row_idx} complete: {ctx.schedule_row_label!r}",
            metrics={"row_index": row_idx},
        )

        # Pause between rows — user can adjust options before next row begins
        _wait_while_paused(ctx)
        if ctx.stop_requested():
            stop_requested = True
            break

        row_idx += 1

    # -- Final summary ----------------------------------------------------
    summary_path = output_dir / "pipeline_run_summary.json"
    legacy_summary_path = output_dir / "summary.json"
    final_execution_state = "stopped" if stop_requested else "completed"
    _save_runtime_snapshot(
        ctx,
        _build_runtime_snapshot(
            ctx,
            ctx.last_node_statuses,
            final_execution_state,
            default_cycle_ids=default_cycle_ids,
        ),
    )
    _emit_execution_event(
        ctx,
        event_id=f"{ctx.session_id}:complete",
        phase="complete",
        status=final_execution_state,
        message=f"Run {final_execution_state}.",
        metrics={"total_rounds_completed": int(ctx.total_rounds_completed)},
    )
    _write_summary(ctx, summary_path)
    _write_summary(ctx, legacy_summary_path)
    if ctx.loss_logger is not None:
        try:
            ctx.loss_logger.close()
        except Exception:
            pass
    _signal_runtime_exit(getattr(ctx, "runtime_control_store", None), f"training_{final_execution_state}")
    _log(f"[orchestrator] run complete. summary -> {summary_path}")
    return stop_requested


# ---------------------------------------------------------------------------
# Config factories — translate argparse.Namespace into typed config objects
# ---------------------------------------------------------------------------

def _build_configs_from_args(args) -> dict:
    """Map every CLI argument to the appropriate typed config dataclass."""
    from pipeline.utils import _resolve_semantic_stage_cache_cap_mb

    def _g(*names, default=None):
        return _arg_value(args, *names, default=default)

    rounds_per_cycle = _parse_positive_int(
        _arg_value(args, "orchestration_rounds", "rounds_per_cycle", default=1),
        field_name="orchestration_rounds",
        default=1,
    )

    global_stage_cache_mb = int(_g("semantic_stage_cache_max_mb", default=0))

    classifier = ClassifierConfig(
        base_ch=int(_g("classifier_base_ch", default=64)),
        max_ch=int(_g("classifier_max_ch", default=384)),
        context_blocks=int(_g("classifier_context_blocks", default=8)),
        context_dropout=float(_g("classifier_context_dropout", default=0.05)),
        mask_decoder_channels=int(_g("mask_decoder_channels", default=-1)),
        label_embedding_backend=str(_g("label_embedding_backend", default="sentence_transformers")),
        label_embedding_model=str(_g("label_embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")),
        label_embedding_dim=int(_g("label_embedding_dim", default=384)),
        label_embedding_temperature=float(_g("label_embedding_temperature", default=10.0)),
        semantic_cosine_weight=float(_g("classifier_semantic_cosine_weight", default=0.35)),
        lr=float(_g("classifier_lr", default=2e-3)),
        weight_decay=float(_g("classifier_weight_decay", default=1e-5)),
        grad_clip=float(_g("classifier_grad_clip", default=1.0)),
        grad_accum_steps=int(_g("grad_accum_steps", "classifier_grad_accum", default=1)),
        lr_cycles=float(_g("lr_sine_cycles", "classifier_lr_cycles", default=1.0)),
        lr_tail_fraction=float(_g("lr_sine_tail_fraction", default=0.15)),
        lr_min_scale=float(_g("lr_sine_min_scale", default=0.0)),
        amp=bool(_g("amp", default=False)),
        amp_dtype=str(_g("amp_dtype", default="float16")),
        channels_last=bool(_g("channels_last", default=False)),
        compile_model=bool(_g("compile_models", "compile", default=False)),
        stage0_epochs=int(_g("pregestation_stage_deck_passes", "pregestation_epochs", default=3)),
        stage0_samples_per_combo=int(_g("semantic_vocab_pregestation_samples_per_combo", "pregestation_samples_per_combo", default=64)),
        stage0_batch_size=int(_g("pregestation_stage_batch_size", "pregestation_batch_size", default=32)),
        stage0_loss_target=float(_g("gate_pregestation_loss_target", "pregestation_loss_target", default=0.80)),
        stage0_required_consecutive=int(_g("gate_pregestation_maintain_rounds", "pregestation_required_consecutive", default=2)),
        stage1_epochs=int(_g("gestation_epochs", default=3)),
        stage1_batch_size=int(_g("gate_gestation_batch_size", "gestation_batch_size", default=32)),
        stage1_loss_target=float(_g("gate_gestation_loss_target", "gestation_loss_target", default=0.80)),
        stage1_required_consecutive=int(_g("gate_gestation_maintain_rounds", "gestation_required_consecutive", default=2)),
        stage2_epochs=int(_g("berkeley_refresh_epochs", "berkeley_epochs", default=1)),
        stage2_batch_size=int(_g("berkeley_refresh_batch_size", "berkeley_batch_size", default=16)),
        stage2_num_workers=int(_g("berkeley_refresh_workers", "num_workers", default=0)),
        stage2_loss_target=float(_g("gate_berkeley_loss_target", "berkeley_loss_target", default=0.0)),
        stage2_confidence_target=float(_g("gate_berkeley_min_confidence", "berkeley_confidence_target", default=0.55)),
        stage2_f1_target=float(_g("gate_berkeley_min_macro_f1", "berkeley_f1_target", default=0.40)),
        stage2_required_consecutive=int(_g("gate_berkeley_maintain_rounds", "berkeley_gate_consecutive", default=1)),
        lora_enabled=bool(_g("stage_c_lora_enabled", default=False)),
        lora_rank=int(_g("stage_c_lora_rank", default=8)),
        lora_alpha=float(_g("stage_c_lora_alpha", default=16.0)),
        stageC_lora_rank=int(_g("stage_c_lora_rank", default=8)),
        stageC_lora_alpha=float(_g("stage_c_lora_alpha", default=16.0)),
        stageC_max_terms=int(_g("stage_c_lora_max_terms", default=50)),
        fake_class_enabled=bool(_g("generator_fake_feedback_enabled", "fake_class_feedback", default=True)),
        fake_class_steps=int(_g("generator_fake_feedback_steps_per_round", "fake_class_steps", default=32)),
        fake_class_batch_size=int(_g("generator_fake_feedback_batch_size", default=16)),
        fake_class_disc_weight=float(_g("generator_fake_feedback_condition_weight", default=1.0)),
        classifier_init_ckpt=str(_g("classifier_init_ckpt", "classifier_init", default="") or ""),
        classifier_init_scope=str(_g("classifier_init_scope", default="all") or "all"),
        label_dropout_rate=float(_g("target_label_knockout_prob", default=0.0)),
        label_dropout_max_drop_frac=float(_g("target_label_knockout_max_drop_frac", default=1.0)),
        label_dropout_min_keep=int(_g("target_label_knockout_min_keep", default=1)),
        label_dropout_network_rate=float(_g("target_label_knockout_network_dropout", default=0.0)),
        label_dropout_dataset_threshold=int(_g("target_label_knockout_dataset_threshold", default=-1)),
        label_dropout_min_keep_dataset=int(_g("target_label_knockout_min_keep_dataset", default=1)),
    )

    transformer = TransformerConfig(
        d_model=int(_g("transformer_d_model", "d_model", default=256)),
        nhead=int(_g("transformer_nhead", "nhead", default=8)),
        num_layers=int(_g("transformer_num_layers", "num_layers", default=6)),
        ff_mult=int(_g("transformer_ff_mult", "ff_mult", default=4)),
        dropout=float(_g("transformer_dropout", default=0.10)),
        image_size=int(_g("image_size", default=128)),
        patch_size=int(_g("patch_size", default=16)),
        compile_model=bool(_g("compile_models", default=False)),
        compile_mode=str(_g("compile_mode", default="default")),
        lr=float(_g("transformer_lr", default=3e-4)),
        grad_clip=float(_g("transformer_grad_clip", default=1.0)),
        grad_accum_steps=int(_g("grad_accum_steps", default=1)),
        lr_cycles=float(_g("lr_sine_cycles", default=1.0)),
        lr_tail_fraction=float(_g("lr_sine_tail_fraction", default=0.15)),
        lr_min_scale=float(_g("lr_sine_min_scale", default=0.0)),
        amp=bool(_g("amp", default=False)),
        amp_dtype=str(_g("amp_dtype", default="float16")),
        channels_last=bool(_g("channels_last", default=False)),
        feature_score_weight=float(_g("transformer_loss_score_target_weight", "feature_score_weight", default=1.0)),
        rank_loss_weight=float(_g("transformer_loss_score_rank_weight", "rank_loss_weight", default=0.10)),
        spurious_weight=float(_g("transformer_loss_score_spurious_weight", "spurious_weight", default=0.05)),
        entropy_weight=float(_g("transformer_loss_entropy_weight", "entropy_weight", default=0.01)),
        high_bit_weight=float(_g("transformer_loss_high_bit_weight", default=0.20)),
        low_bit_weight=float(_g("transformer_loss_low_bit_weight", default=0.10)),
        wave_l1_weight=float(_g("transformer_loss_wave_l1_weight", default=0.05)),
        steps_per_round=int(_g("transformer_steps_per_round", "transformer_steps", default=64)),
        log_every=int(_g("transformer_log_every", default=0)),
        degrade_curriculum_enabled=bool(_g("transformer_degrade_inputs", default=True)),
        config_search_trials=int(_g("config_trials", "config_search_trials", default=32)),
        config_search_epochs_per_trial=int(_g("config_epochs", default=1)),
        config_search_scipy_refine=bool(_g("try_scipy_refine", default=False)),
        gate_score_target=float(_g("gate_transformer_min_score_after", "transformer_gate_target", default=0.60)),
        gate_entropy_min=float(_g("transformer_entropy_min", default=0.40)),
        gate_required_consecutive=int(_g("gate_transformer_maintain_rounds", "transformer_gate_consecutive", default=3)),
        seed=int(_g("seed", default=42)),
        config_search_max_points=int(_g("chunk_samples", "max_points", default=0)),
        config_search_batch_size=int(_g("batch_size", default=16)),
        config_search_topk=int(_g("score_topk", default=3)),
        config_search_threshold=float(_g("score_threshold", default=0.35)),
        config_search_w_topk=float(_g("score_w_topk", default=0.60)),
        config_search_w_cov=float(_g("score_w_cov", default=0.30)),
        config_search_w_mean=float(_g("score_w_mean", default=0.10)),
        transformer_init_ckpt=str(_g("resume_transformer", "transformer_init", default="") or ""),
        sample_bits=int(_g("sample_bits", default=16)),
    )

    generator = GeneratorConfig(
        z_dim=int(_g("generator_z_dim", "z_dim", default=128)),
        g_depth=int(_g("generator_depth", "g_depth", default=4)),
        g_base_ch=int(_g("generator_base_ch", "g_base_ch", default=64)),
        d_depth=int(_g("discriminator_depth", "d_depth", default=4)),
        d_base_ch=int(_g("discriminator_base_ch", "d_base_ch", default=64)),
        g_lr=float(_g("generator_lr", "g_lr", default=1e-4)),
        d_lr=float(_g("discriminator_lr", "d_lr", default=4e-4)),
        amp=bool(_g("amp", default=False)),
        amp_dtype=str(_g("amp_dtype", default="float16")),
        steps_per_round=int(_g("generator_steps_per_round", "gan_steps", default=64)),
        d_steps_per_g_step=int(_g("discriminator_steps_per_generator_step", default=1)),
        adv_weight=float(_g("generator_loss_adv_weight", "adv_weight", default=1.0)),
        feature_score_weight=float(_g("generator_loss_cls_weight", "gan_feature_weight", default=0.5)),
        wave_recon_weight=float(_g("generator_loss_wave_weight", default=0.1)),
        r1_weight=float(_g("r1_weight", default=10.0)),
        gate_feature_score_target=float(_g("generator_gate_min_target_prob", "generator_gate_target", default=0.50)),
        gate_required_consecutive=int(_g("generator_gate_maintain_rounds", "generator_gate_consecutive", default=2)),
        vocab_snapshot_enabled=bool(_g("gd_vocab_library_enabled", "vocab_snapshot", default=True)),
        vocab_snapshot_dir=str(_g("gd_vocab_library_dir", default="") or ""),
        compile_model=bool(_g("compile_models", default=False)),
        joint_mode=bool(_g("joint_enabled", default=False)),
        generator_init_ckpt=str(_g("generator_init", default="") or ""),
        discriminator_init_ckpt=str(_g("discriminator_init", default="") or ""),
        g_mask_decoder_channels=int(_g("generator_mask_decoder_channels", "g_mask_decoder_ch", default=64)),
    )

    wave = WaveClassifierConfig(
        base_ch=int(_g("wave_cls_base_ch", "classifier_base_ch", default=48)),
        max_ch=int(_g("wave_cls_max_ch", "classifier_max_ch", default=256)),
        lr=float(_g("wave_cls_lr", default=1e-3)),
        epochs_per_round=int(_g("wave_cls_epochs_per_round", "wave_cls_epochs", default=1)),
        batch_size=int(_g("wave_cls_batch_size", default=32)),
        label_mode=str(_g("label_mode", "wave_label_mode", default="folder")),
        max_accepted_chunks=int(_g("wave_cls_train_samples", default=1024)),
        zero_shot_enabled=bool(_g("label_embeddings_enabled", default=True)),
        zero_shot_query_terms=str(_g("wave_zero_shot_query_texts", default="") or ""),
        zero_shot_top_k=int(_g("wave_zero_shot_topk", default=5)),
        gate_feature_score_min=float(_g("wave_gate_score", default=0.55)),
        gate_entropy_min=float(_g("wave_gate_entropy", default=0.40)),
        gate_required_consecutive=int(_g("wave_gate_consecutive", default=3)),
    )

    vocab = VocabConfig(
        extra_terms_json=str(_g("semantic_vocab_extra_json", "extra_terms_json", default="") or ""),
        total_slots=int(_g("semantic_vocab_extra_slots", "num_classes", default=50)),
        churn_n=int(_g("semantic_vocab_churn_replace_per_cycle", "vocab_churn_n", default=4)),
        churn_every_n_cycles=1,
        symbol_pool_mode="auto" if bool(_g("semantic_vocab_auto_symbol_pool", default=False)) else str(_g("symbol_pool_mode", default="synthetic")),
        symbol_pool_samples_per_term=int(_g("semantic_vocab_symbol_samples_per_term", "symbol_samples_per_term", default=16)),
        flashcard_enabled=bool(_g("semantic_vocab_reference_flashcards", "flashcard_enabled", default=True)),
        flashcard_rows_per_term=int(_g("semantic_vocab_reference_flashcards_per_term", default=4)),
        seed=int(_g("seed", default=0)),
        image_size=int(_g("image_size", default=128)),
        symbol_pool_root=str(_g("semantic_vocab_symbol_pool_root", default="") or ""),
        symbol_include_pictograms=bool(_g("semantic_vocab_symbol_include_pictograms", default=True)),
        symbol_bootstrap_origin_label=str(_g("semantic_vocab_bootstrap_origin_label", default="internal bootstrap root vocab") or ""),
        extra_terms_cli=str(_g("semantic_vocab_extra_texts", "extra_semantic_terms", default="") or ""),
        classifier_init_ckpt=str(_g("classifier_init_ckpt", "classifier_init", default="") or ""),
    )

    embedding = LabelEmbeddingConfig(
        backend=str(_g("label_embedding_backend", default="sentence_transformers")),
        model_name=str(_g("label_embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")),
        label_text_json=str(_g("label_text_json", default="") or ""),
        target_dim=int(_g("label_embedding_dim", default=384)),
        temperature=float(_g("label_embedding_temperature", default=10.0)),
    )

    wave_pool = WavePoolConfig(
        wav_dir=str(_g("wav_root", "wav_dir", default="") or ""),
        latent_pool_dir=str(_g("latent_pool_dir", default="") or ""),
        latent_pool_size=int(_g("latent_fallback_count", "latent_pool_size", default=64)),
        latent_pool_sample_rate=int(_g("latent_fallback_rate", default=22050)),
        latent_pool_duration_s=float(_g("latent_fallback_seconds", default=5.0)),
        max_streams=int(_g("max_files", default=0)),
        seed=int(_g("seed", default=42)),
        latent_pool_noise_std=float(_g("latent_pool_noise_std", default=0.5)),
        latent_reinject_dir=str(_g("latent_reinject_dir", default="") or ""),
        latent_reinject_ratio=float(_g("latent_reinject_ratio", default=0.0)),
        latent_reinject_copy_gain=float(_g("latent_reinject_copy_gain", default=0.9)),
        latent_reinject_noise_gain=float(_g("latent_reinject_noise_gain", default=0.1)),
        latent_structured_ratio=float(_g("latent_structured_ratio", default=0.25)),
        latent_structured_gain=float(_g("latent_structured_gain", default=0.5)),
        latent_structured_noise_gain=float(_g("latent_structured_noise_gain", default=0.1)),
    )

    pregestation = PregestationDataConfig(
        samples_per_combo=int(_g("semantic_vocab_pregestation_samples_per_combo", "pregestation_samples_per_combo", default=64)),
        image_size=int(_g("image_size", default=128)),
        batch_size=int(_g("pregestation_stage_batch_size", "pregestation_batch_size", default=32)),
        cache_mb=int(
            _resolve_semantic_stage_cache_cap_mb(
                global_cap_mb=int(global_stage_cache_mb),
                specific_cap_mb=int(_g("pregestation_stage_cache_max_mb", "pregestation_cache_mb", default=-1)),
            )
        ),
        displacement_temperature=float(_g("pregestation_circle_displacement_temperature", default=0.4)),
        circle_radius_temperature=float(_g("pregestation_circle_radius_temperature", default=1.0)),
        seed=int(_g("seed", default=42)),
        gpu_preprocess=bool(_g("semantic_gpu_preprocess", default=False)),
    )

    gestation = GestationDataConfig(
        image_size=int(_g("image_size", default=128)),
        batch_size=int(_g("gate_gestation_batch_size", "gestation_batch_size", default=32)),
        samples_per_term=int(_g("semantic_vocab_symbol_samples_per_term", "gestation_samples_per_term", default=64)),
        deformations_per_clean=int(_g("gestation_deformations_per_clean", default=4)),
        cache_mb=int(
            _resolve_semantic_stage_cache_cap_mb(
                global_cap_mb=int(global_stage_cache_mb),
                specific_cap_mb=int(_g("gestation_stage_cache_max_mb", default=-1)),
            )
        ),
        gpu_preprocess=bool(_g("semantic_gpu_preprocess", default=False)),
    )

    berkeley_payload = BerkeleyPayloadConfig(
        berkeley_data_root=str(_g("berkeley_data_root", default="") or ""),
        payload_bank_dir=str(_g("berkeley_payload_source_root", default="") or ""),
        image_size=int(_g("berkeley_image_size", "image_size", default=128)),
        build_batch_size=int(_g("berkeley_refresh_loader_batch_size", "berkeley_build_batch", default=32)),
        build_num_workers=int(_g("berkeley_refresh_workers", default=0)),
        max_images=int(_g("berkeley_payload_max_samples", default=0)),
        force_cache_rebuild=bool(_g("berkeley_payload_cache_rebuild", default=False)),
        seed=int(_g("seed", default=42)),
        auto_install_scipy=bool(_g("berkeley_auto_install_scipy", "auto_install_scipy", default=False)),
    )

    berkeley_data = BerkeleyDataConfig(
        berkeley_data_root=str(_g("berkeley_data_root", default="") or ""),
        image_size=int(_g("berkeley_image_size", "image_size", default=128)),
        batch_size=int(_g("berkeley_refresh_batch_size", "berkeley_batch_size", default=16)),
        num_workers=int(_g("berkeley_refresh_workers", "num_workers", default=0)),
        max_train=int(_g("berkeley_refresh_max_train", default=0)),
        prefetch_factor=int(_g("loader_prefetch_factor", default=0)),
        prebuild_batches=int(_g("berkeley_refresh_cache_batches", default=0)),
        cache_device=str(_g("berkeley_refresh_cache_device", default="auto") or "auto"),
        seed=int(_g("seed", default=42)),
        wheel_max_bytes=int(max(0, int(_g("berkeley_wheel_max_mb", default=0)))) * 1024 * 1024,
        wheel_sanity_cap_bytes=int(max(1, int(_g("berkeley_wheel_sanity_cap_mb", default=8192)))) * 1024 * 1024,
        wheel_allow_large_override=bool(_g("berkeley_wheel_allow_large_override", default=False)),
        wheel_lookahead_batches=int(_g("berkeley_wheel_lookahead_batches", default=0)),
        wheel_use_rare_term_deck=bool(_g("berkeley_wheel_use_rare_term_deck", default=True)),
        refresh_deformations_per_clean=int(_g("berkeley_refresh_deformations_per_clean", default=1)),
        refresh_include_clean=bool(_g("berkeley_refresh_include_clean", default=True)),
        gpu_preprocess=bool(_g("semantic_gpu_preprocess", default=False)),
        preload_workers=int(_g("semantic_preload_workers", default=0)),
        gate_val_batch_size=int(_g("gate_berkeley_batch_size", default=32)),
        gate_val_num_workers=int(_g("berkeley_refresh_workers", "num_workers", default=0)),
        gate_val_max_val=int(_g("gate_berkeley_max_val", default=0)),
    )

    berkeley_gate = BerkeleyGateConfig(
        confidence_target=float(_g("gate_berkeley_min_confidence", "berkeley_confidence_target", default=0.55)),
        f1_target=float(_g("gate_berkeley_min_macro_f1", "berkeley_f1_target", default=0.40)),
        loss_target=float(_g("gate_berkeley_loss_target", "berkeley_loss_target", default=0.0)),
        required_consecutive=int(_g("gate_berkeley_maintain_rounds", "berkeley_gate_consecutive", default=1)),
        eval_batch_size=int(_g("gate_berkeley_batch_size", default=32)),
    )

    transformer_gate = TransformerGateConfig(
        score_target=float(_g("gate_transformer_min_score_after", "transformer_gate_target", default=0.60)),
        feature_score_min=float(_g("gate_transformer_min_score_after", "transformer_score_min", default=0.50)),
        entropy_min=float(_g("transformer_entropy_min", default=0.40)),
        required_consecutive=int(_g("gate_transformer_maintain_rounds", "transformer_gate_consecutive", default=3)),
    )

    generator_gate = GeneratorGateConfig(
        feature_score_target=float(_g("generator_gate_min_target_prob", "generator_gate_target", default=0.50)),
        required_consecutive=int(_g("generator_gate_maintain_rounds", "generator_gate_consecutive", default=2)),
    )

    wave_gate = WaveGateConfig(
        entropy_min=float(_g("wave_gate_entropy", default=0.40)),
        feature_score_min=float(_g("wave_gate_score", default=0.55)),
        required_consecutive=int(_g("wave_gate_consecutive", default=3)),
    )

    return {
        "classifier": classifier,
        "transformer": transformer,
        "generator": generator,
        "wave": wave,
        "vocab": vocab,
        "embedding": embedding,
        "wave_pool": wave_pool,
        "pregestation": pregestation,
        "gestation": gestation,
        "berkeley_payload": berkeley_payload,
        "berkeley_data": berkeley_data,
        "berkeley_gate": berkeley_gate,
        "transformer_gate": transformer_gate,
        "generator_gate": generator_gate,
        "wave_gate": wave_gate,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _log_statuses(label: str, statuses: dict) -> None:
    ran = sum(1 for s in statuses.values() if s == "ran")
    skipped = sum(1 for s in statuses.values() if s.startswith("skipped"))
    failed = {k: v for k, v in statuses.items() if v.startswith("failed")}
    _log(f"[{label}] ran={ran} skipped={skipped} failed={len(failed)}")
    for nid, err in failed.items():
        _log(f"  ! {nid}: {err}")


def _write_summary(ctx: PipelineContext, path: Path) -> None:
    gate_status = _gate_status_blob(ctx)
    summary = {
        "run_tag": ctx.run_tag,
        "session_id": ctx.session_id,
        "worker_id": ctx.worker_id,
        "plan_id": ctx.plan_id,
        "graph_plan_path": str(ctx.graph_plan_path) if ctx.graph_plan_path is not None else "",
        "runtime_snapshot_path": str(ctx.runtime_snapshot_path) if ctx.runtime_snapshot_path is not None else "",
        "total_rounds": ctx.total_rounds_completed,
        "gates": gate_status,
        "gate_status": gate_status,
        "final_class_names": ctx.class_names,
        "supervised_class_names": ctx.supervised_class_names,
        "last_node_statuses": ctx.last_node_statuses,
        "metrics": ctx.metrics_history[-50:],  # last 50 entries
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except Exception as exc:
        _log(f"[orchestrator] WARNING: could not write summary: {exc}")


def _log(msg: str) -> None:
    print(msg, flush=True)
