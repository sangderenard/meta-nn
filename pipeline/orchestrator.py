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
    [pregestation_data] ─► [stage_0_pregestation]
                                   │
    [gestation_data] ─────────────►[stage_1_gestation]  ← cond: gate_pregestation
                                   │
    [berkeley_payload] ────────────►[payload_validation_data]
                                   │
    [berkeley_data] ───────────────►[stage_2_berkeley]   ← cond: early_gates
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
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch

from pipeline.context import PipelineContext
from pipeline.graph import PipelineGraph
from pipeline.plan_protocol import (
    DEFAULT_PLAN_FILENAME,
    DEFAULT_RUNTIME_SNAPSHOT_FILENAME,
    MESSAGE_TYPE_RUNTIME_SNAPSHOT,
    ExecutionEventPayload,
    RuntimeNodeState,
    RuntimeSnapshotPayload,
    WorkerHelloPayload,
    make_envelope,
    plan_from_pipeline_graph,
    save_protocol_message,
)

# --- node imports ----------------------------------------------------------
from pipeline.nodes.berkeley_classifier_node import (
    BerkeleyClassifierConfig,
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
    PregestationDataNode,
    GestationDataConfig,
    GestationDataNode,
    BerkeleyPayloadConfig,
    BerkeleyPayloadNode,
    BerkeleyDataConfig,
    BerkeleyDataNode,
    PayloadValidationDataNode,
)
from pipeline.nodes.gate_nodes import (
    BerkeleyGateConfig,
    BerkeleyGateNode,
    TransformerGateConfig,
    TransformerGateNode,
    GeneratorGateConfig,
    GeneratorGateNode,
    WaveGateConfig,
    WaveGateNode,
    CheckpointSaveNode,
)


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


def _restore_context_from_resume(ctx: PipelineContext) -> None:
    from wav_ml_core import RenderConfig

    resume_summary = ctx.resume_summary if isinstance(ctx.resume_summary, dict) else {}
    resume_ckpt = ctx.resume_pipeline_ckpt if isinstance(ctx.resume_pipeline_ckpt, dict) else {}

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

    metrics = resume_ckpt.get("metrics_history", resume_summary.get("metrics", []))
    if isinstance(metrics, list):
        ctx.metrics_history = list(metrics)

    if ctx.render_config is None and not bool(_arg_value(ctx.args, "force_config_search", default=False)):
        best_cfg_blob = None
        resume_dir = ctx.resume_dir if isinstance(ctx.resume_dir, Path) else None
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


# ---------------------------------------------------------------------------
# Graph plan / GUI IPC helpers
# ---------------------------------------------------------------------------

_CONDITION_ID_GENERATOR_MODE = "orchestration.mode_has_generator"
_CONDITION_ID_PREGESTATION_GATE = "gates.pregestation_passed"
_CONDITION_ID_EARLY_GATES = "gates.early_passed"
_CONDITION_ID_ALL_GATES = "gates.all_base_passed"
_CONDITION_ID_WAVE_STAGE_READY = "gates.wave_stage_ready"


def _worker_id_for_output(output_dir: Path) -> str:
    name = "".join(ch if ch.isalnum() else "_" for ch in output_dir.name.strip()) or "wav_ml_pipeline"
    return f"worker:{name.lower()}"


def _build_condition_blobs() -> dict:
    return {
        _CONDITION_ID_GENERATOR_MODE: {
            "label": "Generator mode enabled",
            "description": "True when orchestration_mode includes the generator stage.",
            "callable": "_generator_exists",
        },
        _CONDITION_ID_PREGESTATION_GATE: {
            "label": "Pregestation gate passed",
            "description": "True when Stage 0 has cleared, or the GUI gate override is active.",
            "callable": "_gate_pregestation_passed",
        },
        _CONDITION_ID_EARLY_GATES: {
            "label": "Early gates passed",
            "description": "True when pregestation and gestation have both cleared, or the GUI gate override is active.",
            "callable": "_early_gates_passed",
        },
        _CONDITION_ID_ALL_GATES: {
            "label": "All base gates passed",
            "description": "True when pregestation, gestation, and Berkeley have cleared, or the GUI gate override is active.",
            "callable": "_all_gates_passed",
        },
        _CONDITION_ID_WAVE_STAGE_READY: {
            "label": "Wave stage ready",
            "description": "True when the transformer exists and the upstream gates allow the wave classifier stage.",
            "callable": "_wave_stage_ready",
        },
    }


def _node_group_id(node_id: str) -> str:
    if node_id in {"wave_pool"}:
        return "bootstrap"
    if node_id.startswith("gate_"):
        return "gates"
    if node_id in {"sync_gate_replica", "checkpoint_save"}:
        return "housekeeping"
    if node_id.endswith("_data") or node_id in {"berkeley_payload", "payload_validation_data"}:
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
    if node_id.endswith("_data") or "payload" in node_id:
        return "data"
    if node_id.startswith("gate_"):
        return "gate"
    if node_id == "checkpoint_save":
        return "checkpoint"
    if node_id == "sync_gate_replica":
        return "sync"
    return "node"


def _node_config_id(node_id: str) -> str:
    mapping = {
        "wave_pool": "wave_pool",
        "init_vocab": "vocab",
        "vocab_churn": "vocab",
        "build_symbol_pool": "vocab",
        "build_label_embedding": "embedding",
        "build_flashcard_rows": "vocab",
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
        "gate_berkeley": "berkeley_gate",
        "gate_transformer": "transformer_gate",
        "gate_generator": "generator_gate",
        "gate_wave": "wave_gate",
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
    try:
        from wav_ml_viewer import ViewerIPCProxy

        image_size = int(_arg_value(args, "image_size", default=64))
        return ViewerIPCProxy(
            port_file=port_file,
            enabled=True,
            image_hw=(image_size, image_size),
            scale=max(1, int(_arg_value(args, "stage_opengl_preview_scale", "transformer_viz_scale", default=1))),
            cycle_slots=max(0, int(cycles)),
        )
    except Exception as exc:
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
        "orchestration_mode": str(_arg_value(args, "orchestration_mode", default="staged_cgrw") or "staged_cgrw"),
        "orchestration_cycles": int(_arg_value(args, "orchestration_cycles", "cycles", default=1)),
        "orchestration_rounds": int(_arg_value(args, "orchestration_rounds", "rounds_per_cycle", default=1)),
        "entrypoint": "wav_pipeline_graph.py",
        # Capability flags — the GUI reads these to know which IPC messages the
        # worker will act on.  Add a new flag here when a new handler is wired.
        "capabilities": [
            "plan_apply",    # worker applies graph swaps between cycles
            "run_control",   # worker honours stop/resume/gate_override
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
    )


# ---------------------------------------------------------------------------
# Phase 3: plan-driven worker bootstrap
# ---------------------------------------------------------------------------

# Maps the config_blobs key used in TrainingGraphPlan → the config dataclass type.
# This is the full registry of all 15 config dataclasses used by build_pipeline_graph().
_CONFIG_CLASS_REGISTRY = {
    "classifier":        BerkeleyClassifierConfig,
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


def build_training_graph_from_plan(plan) -> PipelineGraph:
    """Materialize an executable PipelineGraph from a saved TrainingGraphPlan.

    This is the reverse of build_training_graph_plan(): it takes the durable
    JSON-serializable plan (loaded from training_graph_plan.json or received
    over the worker↔GUI IPC channel) and produces an executable PipelineGraph
    that can be passed directly to run().

    The reconstructed graph is functionally identical to one built directly via
    build_pipeline_graph() with the same configs — both have the same 30 nodes,
    33 edges, and runtime condition callables.  The plan is only the source of
    the hyperparameter values; the node objects and edge callables are always
    created fresh.

    Parameters
    ----------
    plan : TrainingGraphPlan
        A plan instance (loaded from disk or received via IPC).

    Returns
    -------
    PipelineGraph
        A fully wired, executable PipelineGraph ready for run().
    """
    blobs = dict(plan.config_blobs or {})
    cfg: dict = {}

    for key, cls in _CONFIG_CLASS_REGISTRY.items():
        blob = blobs.get(key)
        if isinstance(blob, dict) and blob and not (set(blob.keys()) == {"value"}):
            # Non-empty dict that isn't a scalar wrapper → reconstruct the dataclass
            cfg[key] = _reconstruct_config(cls, blob)
        else:
            # Key not present in plan (e.g. older plan file) → use defaults
            cfg[key] = cls()

    _validate_training_graph_plan(plan)
    hints = dict(plan.worker_hints or {})
    save_every = _parse_positive_int(hints.get("save_every_n_rounds", 1), field_name="save_every_n_rounds", default=1)
    berkeley_refresh = _parse_positive_int(
        hints.get("berkeley_refresh_every_n_rounds", 4),
        field_name="berkeley_refresh_every_n_rounds",
        default=4,
    )

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
        save_every_n_rounds=save_every,
        berkeley_refresh_every_n_rounds=berkeley_refresh,
    )


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
# Edge condition lambdas
# ---------------------------------------------------------------------------


def _gate_override_enabled(ctx: PipelineContext) -> bool:
    return ctx.gate_override_enabled()

def _gate_pregestation_passed(ctx: PipelineContext) -> bool:
    return _gate_override_enabled(ctx) or ctx.gate_pregestation.passed

def _early_gates_passed(ctx: PipelineContext) -> bool:
    return _gate_override_enabled(ctx) or ctx.early_gates_passed()

def _all_gates_passed(ctx: PipelineContext) -> bool:
    return _gate_override_enabled(ctx) or ctx.all_base_gates_passed()

def _wave_stage_ready(ctx: PipelineContext) -> bool:
    if _gate_override_enabled(ctx):
        return ctx.transformer is not None
    return ctx.wave_stage_ready()

def _generator_exists(ctx: PipelineContext) -> bool:
    mode = str(ctx.orchestration_mode or getattr(ctx.args, "orchestration_mode", "")).lower()
    return "g" in mode

def _berkeley_every_n_rounds(n: int):
    def _cond(ctx: PipelineContext) -> bool:
        return ctx.total_rounds_completed % max(1, n) == 0
    return _cond

def _always(ctx: PipelineContext) -> bool:
    return True


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_pipeline_graph(
    classifier_cfg: BerkeleyClassifierConfig,
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
    g.add_node(BerkeleyPayloadNode(berkeley_payload_cfg))

    # Per-round vocab / embedding
    g.add_node(VocabChurnNode(vocab_cfg))
    g.add_node(BuildSymbolPoolNode(vocab_cfg))
    g.add_node(BuildLabelEmbeddingNode(embedding_cfg))
    g.add_node(BuildFlashcardRowsNode(vocab_cfg))

    # Data
    g.add_node(PregestationDataNode(pregestation_cfg))
    g.add_node(GestationDataNode(gestation_cfg))
    g.add_node(BerkeleyDataNode(berkeley_data_cfg))
    g.add_node(PayloadValidationDataNode())

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
    g.add_node(BerkeleyGateNode(berkeley_gate_cfg))
    g.add_node(TransformerGateNode(transformer_gate_cfg))
    g.add_node(GeneratorGateNode(generator_gate_cfg))
    g.add_node(WaveGateNode(wave_gate_cfg))

    # Housekeeping
    g.add_node(SyncGateReplicaNode(classifier_cfg))
    g.add_node(CheckpointSaveNode(save_every_n_rounds))

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

    # == Data construction ==========================================

    g.add_edge("build_label_embedding", "pregestation_data", label="per_round")
    g.add_edge("build_label_embedding", "gestation_data", label="per_round")
    g.add_edge("build_label_embedding", "berkeley_payload", label="per_round")
    g.add_edge("berkeley_payload", "payload_validation_data", label="per_round")
    g.add_edge("berkeley_payload", "build_flashcard_rows", label="per_round")

    g.add_edge("build_label_embedding", "berkeley_data",
               condition=_gate_pregestation_passed, label="after_gate0", condition_id=_CONDITION_ID_PREGESTATION_GATE)

    # == Stage 0 — Pre-gestation ===================================

    g.add_edge("pregestation_data", "stage_0_pregestation", label="stage0")

    # == Stage 1 — Gestation =======================================

    g.add_edge("stage_0_pregestation", "gestation_data",
               condition=_gate_pregestation_passed, label="after_gate0", condition_id=_CONDITION_ID_PREGESTATION_GATE)
    g.add_edge("gestation_data", "stage_1_gestation",
               condition=_gate_pregestation_passed, label="after_gate0", condition_id=_CONDITION_ID_PREGESTATION_GATE)

    # == Stage 2 — Berkeley refresh ================================

    g.add_edge("stage_1_gestation", "stage_2_berkeley",
               condition=_early_gates_passed, label="after_gate1", condition_id=_CONDITION_ID_EARLY_GATES)
    g.add_edge("berkeley_data", "stage_2_berkeley",
               condition=_early_gates_passed, label="after_gate1", condition_id=_CONDITION_ID_EARLY_GATES)

    g.add_edge("stage_2_berkeley", "gate_berkeley",
               condition=_early_gates_passed, label="after_gate1", condition_id=_CONDITION_ID_EARLY_GATES)

    # == Stage R — Transformer =====================================

    g.add_edge("gate_berkeley", "stage_r_transformer",
               condition=_early_gates_passed, label="after_gate1", condition_id=_CONDITION_ID_EARLY_GATES)
    g.add_edge("stage_r_transformer", "gate_transformer",
               condition=_early_gates_passed, label="after_gate1", condition_id=_CONDITION_ID_EARLY_GATES)

    # == Stage G — Generator =======================================

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
    g.add_edge("gate_transformer", "sync_gate_replica", label="end_of_round")
    g.add_edge("gate_berkeley", "sync_gate_replica", label="end_of_round")
    g.add_edge("sync_gate_replica", "checkpoint_save", label="end_of_round")

    return g


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
    runtime = _prepare_runtime(args, output_dir, device)
    _log(f"[orchestrator] device={device}")

    # -- Context ----------------------------------------------------------
    ctx = PipelineContext(
        args=args,
        device=device,
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
    ctx.viewer_proxy = _make_viewer_proxy(args, cycles)
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
    if ctx.graph_plan_path is not None:
        try:
            ctx.graph_plan.save_json(ctx.graph_plan_path)
        except Exception as exc:
            _log(f"[orchestrator] WARNING: could not write training graph plan: {exc}")

    _send_viewer_bootstrap(ctx)

    _log(graph.summary())

    # Build a fixed execution sequence once (topological order)
    sequence = graph.build_sequence()

    # -- One-time initialisation pass (no loop) ---------------------------
    # The first pass runs init_vocab, build_classifier, wave_pool, etc.
    # Subsequent passes will hit the `_done` guards on one-shot nodes.
    statuses = graph.execute_sequence(ctx, sequence=sequence)
    ctx.last_node_statuses = dict(statuses)
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

    # -- Outer orchestration loop -----------------------------------------
    stop_requested = False

    for cycle_idx in range(1, cycles + 1):
        if ctx.stop_requested():
            _log("[orchestrator] GUI requested stop before next cycle; stopping run")
            stop_requested = True
            break

        # READ POINT A — apply a new plan received from the GUI between cycles
        if ctx.viewer_proxy is not None:
            apply_payload = ctx.viewer_proxy.consume_pending_plan_apply()
            if apply_payload is not None:
                _log(
                    f"[orchestrator] plan_apply received before cycle {cycle_idx}: "
                    f"reason={apply_payload.reason!r} replace_current={apply_payload.replace_current}"
                )
                try:
                    new_graph = build_training_graph_from_plan(apply_payload.plan)
                    sequence = new_graph.build_sequence()
                    graph = new_graph
                    ctx.graph_plan = apply_payload.plan
                    ctx.plan_id = str(apply_payload.plan.plan_id)
                    if ctx.graph_plan_path is not None:
                        try:
                            apply_payload.plan.save_json(ctx.graph_plan_path)
                        except Exception as _save_exc:
                            _log(f"[orchestrator] WARNING: could not persist applied plan: {_save_exc}")
                    _log(f"[orchestrator] plan_apply complete: {len(sequence)} nodes in new sequence")
                    _send_viewer_bootstrap(ctx)
                except Exception as apply_exc:
                    _log(f"[orchestrator] ERROR: plan_apply failed — keeping existing graph: {apply_exc}")

        if not ctx.is_cycle_selected(cycle_idx):
            ctx.cycle = cycle_idx
            ctx.round_id = 0
            deselected = {node_id: "skipped:cycle_deselected" for node_id in sequence}
            ctx.last_node_statuses = dict(deselected)
            _log(f"[orchestrator] cycle={cycle_idx}/{cycles} skipped by GUI selection")
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
                metrics={"cycle": int(cycle_idx)},
            )
            continue
        ctx.cycle = cycle_idx
        for round_idx in range(1, rounds_per_cycle + 1):
            if ctx.stop_requested():
                _log("[orchestrator] GUI requested stop; stopping run")
                stop_requested = True
                break
            ctx.round_id = round_idx
            ctx.total_rounds_completed += 1

            _log(f"\n{'=' * 60}")
            _log(f"[orchestrator] cycle={cycle_idx}/{cycles}  round={round_idx}/{rounds_per_cycle}"
                 f"  total={ctx.total_rounds_completed}")
            _log(f"{'=' * 60}")

            statuses = graph.execute_sequence(ctx, sequence=sequence)
            ctx.last_node_statuses = dict(statuses)
            _log_statuses(f"c{cycle_idx}r{round_idx}", statuses)
            _save_runtime_snapshot(
                ctx,
                _build_runtime_snapshot(ctx, statuses, "running", default_cycle_ids=default_cycle_ids),
            )
            _emit_execution_event(
                ctx,
                event_id=f"{ctx.session_id}:cycle:{cycle_idx}:round:{round_idx}",
                phase="round",
                status="ok",
                message=f"Executed cycle {cycle_idx} round {round_idx}.",
                metrics={
                    "cycle": int(cycle_idx),
                    "round": int(round_idx),
                    "ran": sum(1 for s in statuses.values() if s == "ran"),
                    "skipped": sum(1 for s in statuses.values() if str(s).startswith("skipped")),
                    "failed": sum(1 for s in statuses.values() if str(s).startswith("failed")),
                },
            )

            if ctx.vocab_rotation_cycle == 0:
                ctx.vocab_rotation_cycle = 1
        if stop_requested:
            break

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
    _log(f"[orchestrator] run complete. summary -> {summary_path}")


# ---------------------------------------------------------------------------
# Config factories — translate argparse.Namespace into typed config objects
# ---------------------------------------------------------------------------

def _build_configs_from_args(args) -> dict:
    """Map every CLI argument to the appropriate typed config dataclass."""

    def _g(*names, default=None):
        return _arg_value(args, *names, default=default)

    classifier = BerkeleyClassifierConfig(
        base_ch=int(_g("classifier_base_ch", default=64)),
        max_ch=int(_g("classifier_max_ch", default=384)),
        context_blocks=int(_g("classifier_context_blocks", default=8)),
        context_dropout=float(_g("classifier_context_dropout", default=0.05)),
        mask_decoder_channels=int(_g("mask_decoder_channels", default=0)),
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
        stage0_loss_target=float(_g("gate_pregestation_loss_target", "pregestation_loss_target", default=0.35)),
        stage0_required_consecutive=int(_g("gate_pregestation_maintain_rounds", "pregestation_required_consecutive", default=2)),
        stage1_epochs=int(_g("gestation_epochs", default=1)),
        stage1_batch_size=int(_g("gate_gestation_batch_size", "gestation_batch_size", default=32)),
        stage1_loss_target=float(_g("gate_gestation_loss_target", "gestation_loss_target", default=0.30)),
        stage1_required_consecutive=int(_g("gate_gestation_maintain_rounds", "gestation_required_consecutive", default=2)),
        stage2_epochs=int(_g("berkeley_refresh_epochs", "berkeley_epochs", default=1)),
        stage2_batch_size=int(_g("berkeley_refresh_batch_size", "berkeley_batch_size", default=16)),
        stage2_num_workers=int(_g("berkeley_refresh_workers", "num_workers", default=0)),
        stage2_loss_target=float(_g("gate_berkeley_loss_target", "berkeley_loss_target", default=0.0)),
        stage2_confidence_target=float(_g("gate_berkeley_min_confidence", "berkeley_confidence_target", default=0.55)),
        stage2_f1_target=float(_g("gate_berkeley_min_macro_f1", "berkeley_f1_target", default=0.40)),
        stage2_required_consecutive=int(_g("gate_berkeley_maintain_rounds", "berkeley_gate_consecutive", default=1)),
        lora_enabled=bool(_g("lora_enabled", default=False)),
        lora_rank=int(_g("lora_rank", default=8)),
        fake_class_enabled=bool(_g("generator_fake_feedback_enabled", "fake_class_feedback", default=True)),
        fake_class_steps=int(_g("generator_fake_feedback_steps_per_round", "fake_class_steps", default=32)),
        fake_class_batch_size=int(_g("generator_fake_feedback_batch_size", default=16)),
        fake_class_disc_weight=float(_g("generator_fake_feedback_condition_weight", default=1.0)),
        classifier_init_ckpt=str(_g("classifier_init_ckpt", "classifier_init", default="") or ""),
        classifier_init_scope=str(_g("classifier_init_scope", default="all") or "all"),
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
        cache_mb=int(_g("pregestation_stage_cache_max_mb", "pregestation_cache_mb", default=256)),
        displacement_temperature=float(_g("pregestation_circle_displacement_temperature", default=0.4)),
        circle_radius_temperature=float(_g("pregestation_circle_radius_temperature", default=1.0)),
        seed=int(_g("seed", default=42)),
    )

    gestation = GestationDataConfig(
        image_size=int(_g("image_size", default=128)),
        batch_size=int(_g("gate_gestation_batch_size", "gestation_batch_size", default=32)),
        samples_per_term=int(_g("semantic_vocab_symbol_samples_per_term", "gestation_samples_per_term", default=32)),
        cache_mb=int(_g("gestation_stage_cache_max_mb", default=128)),
    )

    berkeley_payload = BerkeleyPayloadConfig(
        berkeley_data_root=str(_g("berkeley_data_root", default="") or ""),
        payload_bank_dir=str(_g("berkeley_payload_source_root", default="") or ""),
        image_size=int(_g("berkeley_image_size", "image_size", default=128)),
        build_batch_size=int(_g("berkeley_refresh_loader_batch_size", "berkeley_build_batch", default=32)),
        build_num_workers=int(_g("berkeley_refresh_workers", default=0)),
        max_images=int(_g("berkeley_payload_max_samples", default=0)),
        seed=int(_g("seed", default=42)),
        auto_install_scipy=bool(_g("auto_install_scipy", default=False)),
    )

    berkeley_data = BerkeleyDataConfig(
        batch_size=int(_g("berkeley_refresh_batch_size", "berkeley_batch_size", default=16)),
        num_workers=int(_g("berkeley_refresh_workers", "num_workers", default=0)),
        prefetch_factor=int(_g("loader_prefetch_factor", default=0)),
        prebuild_batches=int(_g("berkeley_refresh_cache_batches", default=0)),
        rebuild_every_n_rounds=int(_g("berkeley_refresh_round_every", "berkeley_refresh_every", default=4)),
        gate_val_batch_size=int(_g("gate_berkeley_batch_size", default=32)),
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
