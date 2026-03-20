"""
Classifier Node — TinyConvClassifier training across all stages.

This file is the authoritative description of everything idiosyncratic to the
semantic classifier in this pipeline:

  Stage 0  Pre-gestation   — synthetic geometric direction+colour logic images
  Stage 1  Gestation       — bootstrap primitive symbol images
  Stage 2  Berkeley        — full Berkeley SBD + external payload images
  Stage C  LoRA rounds     — per-term LoRA adapter slot switching
  Fake-class feedback      — discriminator-confidence-weighted fake image curriculum

Model architecture choices owned here
--------------------------------------
  * TinyConvClassifier with configurable base_ch / max_ch / context_blocks
  * Optional mask decoder head (mask_decoder_channels)
  * Optional LoRA adapters on Linear and Conv1x1 layers
  * Label embedding bank from sentence-transformers (cosine similarity head)
  * Frozen CPU gate replica (gate_classifier) for gate evaluation

Loss composition owned here
----------------------------
  * BCE multi-label loss  (primary)
  * Semantic cosine embedding loss weighted by CLASSIFIER_SEMANTIC_COSINE_WEIGHT
  * Optional spatial mask BCE loss (mask supervision)
  * Fake-class sentinel BCE on GAN outputs

Training schedule owned here
------------------------------
  * SinusoidalLR with configurable cycles / tail_fraction / min_scale
  * Gradient accumulation steps
  * Per-stage epoch/step counts
  * Mixed precision (AMP fp16 / bf16)
  * channels_last memory format support
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pipeline.context import PipelineContext
from pipeline.graph import PipelineNode
from pipeline.nodes.base import (
    GatedNode,
    IRLossTermSpec,
    IRStateSpec,
    IRTrainingNode,
    IRTensorPortSpec,
    StageSkipBack,
    StageSkipForward,
    StageStopRequested,
    autocast_context,
    freeze,
    make_runtime_weight_publish_callback,
    make_grad_scaler,
    module_device,
    resolve_amp_dtype,
    resolve_non_training_device,
    unwrap_compiled,
    _extract_state_dict,
    _strip_module_prefix,
    _torch_load_cpu,
    _try_partial_classifier_head_load,
)
from pipeline.nodes.data_nodes import (
    _auto_berkeley_refresh_batch_size,
    _expand_semantic_mask_supervision_batch,
    _forward_classifier_outputs_require_mask,
    _payload_condition_bank_tensor,
    _semantic_mask_bce_loss,
    _unpack_masked_semantic_batch,
)
from pipeline.nodes.vocab_node import _label_knockout_tensor_batch
from pipeline.preview import make_classifier_step_preview_callback
from pipeline.utils import (
    _classifier_supervision_loss,
    _cuda_mem_diag,
    _refresh_cache_is_staging_safe,
    _unwrap_module_for_replica,
)
from semantic_dataset_loaders import (
    _composite_mask_stack,
    _is_dataset_label_term,
    build_creation_label_mask_stack,
    build_label_mask_stack,
    combine_label_mask_stacks,
)
import gc
import math
import re
import time
import numpy as np

# Module-level constants (originally in wav_config_transformer_pipeline)
CLASSIFIER_LOSS_SCALE = 1.0
CLASSIFIER_SEMANTIC_COSINE_WEIGHT = 0.35


# ---------------------------------------------------------------------------
# Node config — all idiosyncrasies in one place
# ---------------------------------------------------------------------------

@dataclass
class ClassifierConfig:
    """Every hyperparameter specific to the TinyConvClassifier."""

    # ---- architecture ---------------------------------------------------
    base_ch: int = 64
    max_ch: int = 384
    context_blocks: int = 8
    context_dropout: float = 0.05
    mask_decoder_channels: int = -1      # -1 = auto (max(32, base_ch//2)); 0 = disabled

    # ---- label embedding ------------------------------------------------
    label_embedding_backend: str = "sentence_transformers"
    label_embedding_model: str = "all-MiniLM-L6-v2"
    label_embedding_dim: int = 0         # 0 = model native dim
    label_embedding_temperature: float = 10.0
    semantic_cosine_weight: float = 0.35

    # ---- LoRA -----------------------------------------------------------
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_enabled: bool = True
    lora_churn_slots: int = 4            # number of active LoRA slots per vocab cycle

    # ---- optimiser / LR -------------------------------------------------
    lr: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    grad_accum_steps: int = 1

    # Sinusoidal LR schedule
    lr_cycles: float = 1.0
    lr_tail_fraction: float = 0.15
    lr_min_scale: float = 0.0

    # ---- AMP ------------------------------------------------------------
    amp: bool = False
    amp_dtype: str = "fp16"
    channels_last: bool = False
    compile_model: bool = False

    # ---- Stage 0 — pre-gestation ----------------------------------------
    stage0_epochs: int = 3
    stage0_samples_per_combo: int = 64
    stage0_batch_size: int = 32
    stage0_loss_target: float = 0.80
    stage0_required_consecutive: int = 2

    # ---- Stage 1 — gestation (bootstrap primitives) ----------------------
    stage1_epochs: int = 5
    stage1_batch_size: int = 32
    stage1_loss_target: float = 0.80
    stage1_required_consecutive: int = 2

    # ---- Stage 2 — Berkeley SBD refresh ----------------------------------
    stage2_epochs: int = 2
    stage2_batch_size: int = 16
    stage2_num_workers: int = 0
    stage2_prefetch_factor: int = 0
    stage2_cache_mb: int = 512
    stage2_loss_target: float = 0.25
    stage2_confidence_target: float = 0.55
    stage2_f1_target: float = 0.40
    stage2_required_consecutive: int = 1

    # ---- Stage C — LoRA round (per vocab churn group) -------------------
    stageC_steps_per_slot: int = 64
    stageC_lora_rank: int = 8
    stageC_lora_alpha: float = 16.0
    stageC_max_terms: int = 50
    stageC_batch_size: int = 16

    # ---- Fake-class feedback -------------------------------------------
    fake_class_enabled: bool = True
    fake_class_steps: int = 32
    fake_class_batch_size: int = 16
    fake_class_disc_weight: float = 1.0   # discriminator-confidence weighting

    # ---- Progress logging (applies to all _run_classifier_refresh_epochs calls) --
    # 0 = silent; N = print one progress line every N steps
    log_every: int = 50

    # ---- Gate replica sync cadence -------------------------------------
    gate_replica_sync_every_n_rounds: int = 1

    # ---- Checkpoint init -----------------------------------------------
    classifier_init_ckpt: str = ""
    classifier_init_scope: str = "all"


# ---------------------------------------------------------------------------
# Classifier build / init node (runs once at startup)
# ---------------------------------------------------------------------------

class BuildClassifierNode(PipelineNode):
    """Instantiate TinyConvClassifier and its optimizer.

    Runs exactly once.  If a checkpoint is present in ctx.args it loads weights.
    """

    node_id = "build_classifier"
    description = "Instantiate TinyConvClassifier + optimizer"
    runtime_object_type = "builder"
    runtime_faculty = "build"
    gpu_models = ["classifier"]

    def __init__(self, cfg: ClassifierConfig) -> None:
        self.cfg = cfg
        self._built = False

    @property
    def runtime_execution_policy(self) -> tuple:
        return ("once", {})

    def should_run(self, ctx: PipelineContext) -> bool:
        return not self._built

    def execute(self, ctx: PipelineContext) -> None:
        from wav_ml_models import (
            TinyConvClassifier,
            maybe_compile_module,
            prime_tiny_classifier_label_bank_for_state_dict,
            restore_tiny_classifier_lora_snapshot,
        )

        n_classes = len(ctx.class_names) if ctx.class_names else 1
        mask_ch = self.cfg.mask_decoder_channels
        if mask_ch < 0:
            mask_ch = max(32, int(self.cfg.base_ch) // 2)
        model = TinyConvClassifier(
            num_classes=n_classes,
            base_ch=self.cfg.base_ch,
            max_ch=self.cfg.max_ch,
            context_blocks=self.cfg.context_blocks,
            context_dropout=self.cfg.context_dropout,
            mask_decoder_channels=mask_ch,
        ).to(ctx.device)

        if self.cfg.channels_last:
            model = model.to(memory_format=torch.channels_last)

        if self.cfg.compile_model:
            model = maybe_compile_module(model, enabled=True)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        ctx.classifier = model
        ctx.classifier_optimizer = optimizer
        ctx.classifier_lr_controller = None

        if self.cfg.amp:
            ctx.classifier_grad_scaler = make_grad_scaler(enabled=True)

        resume_ckpt = ctx.resume_pipeline_ckpt if isinstance(ctx.resume_pipeline_ckpt, dict) else None
        if resume_ckpt is not None:
            if "classifier_state" in resume_ckpt:
                try:
                    if isinstance(resume_ckpt.get("classifier_lora"), dict):
                        restore_tiny_classifier_lora_snapshot(model, resume_ckpt.get("classifier_lora"))
                    prime_tiny_classifier_label_bank_for_state_dict(
                        model,
                        resume_ckpt["classifier_state"],
                        temperature=float(self.cfg.label_embedding_temperature),
                    )
                    model.load_state_dict(resume_ckpt["classifier_state"], strict=False)
                    _log("[classifier] resumed model state from pipeline checkpoint")
                except Exception as exc:
                    _log(f"[classifier] WARNING: could not resume model state: {exc}")
            if "classifier_optimizer_state" in resume_ckpt:
                try:
                    optimizer.load_state_dict(resume_ckpt["classifier_optimizer_state"])
                    _log("[classifier] resumed optimizer state from pipeline checkpoint")
                except Exception as exc:
                    _log(f"[classifier] WARNING: could not resume optimizer state: {exc}")
            scaler = ctx.classifier_grad_scaler
            if scaler is not None and "classifier_grad_scaler_state" in resume_ckpt:
                try:
                    scaler.load_state_dict(resume_ckpt["classifier_grad_scaler_state"])
                    _log("[classifier] resumed grad-scaler state from pipeline checkpoint")
                except Exception as exc:
                    _log(f"[classifier] WARNING: could not resume grad-scaler state: {exc}")

        # Load checkpoint if available
        ckpt_path = self.cfg.classifier_init_ckpt or getattr(ctx.args, "classifier_init", "") or ""
        ckpt_scope = self.cfg.classifier_init_scope or "all"
        if str(ckpt_path).strip():
            _load_classifier_checkpoint(model, str(ckpt_path), scope=str(ckpt_scope))

        # Build frozen gate replica on CPU
        ctx.gate_classifier = _build_gate_replica(model, ctx)

        _log(f"[classifier] built: n_classes={n_classes} "
             f"base_ch={self.cfg.base_ch} max_ch={self.cfg.max_ch} "
             f"context_blocks={self.cfg.context_blocks} "
             f"mask_decoder_channels={mask_ch}")
        self._built = True


# ---------------------------------------------------------------------------
# Stage 0 — Pre-gestation training node
# ---------------------------------------------------------------------------

class PregestationTrainNode(IRTrainingNode):
    """Stage 0: Train classifier on synthetic geometric direction+colour logic images.

    This is the very first stage; the classifier learns basic colour/direction/shape
    semantics from programmatically generated images before seeing any real data.
    Gate progression is evaluated by a separate passive Gate-0 eval node
    against the validation split.
    """

    node_id = "stage_0_pregestation"
    description = "Stage 0: pre-gestation classifier training (geometric logic)"
    gpu_models = ["classifier"]
    model_attr = "classifier"
    optimizer_attrs = ["classifier_optimizer"]

    def __init__(self, cfg: ClassifierConfig) -> None:
        self.cfg = cfg

    def ir_input_ports(self) -> List[IRTensorPortSpec]:
        return [
            IRTensorPortSpec("pregestation_images", "Pregestation images", io="input", dtype="float32", shape="B x 3 x H x W", semantic="image_batch", detail="synthetic logic render batch"),
            IRTensorPortSpec("pregestation_targets", "Pregestation targets", io="input", dtype="float32", shape="B x C", semantic="multilabel_targets", detail="multi-hot classifier targets"),
            IRTensorPortSpec("pregestation_masks", "Pregestation masks", io="input", dtype="float32", shape="B x 1 x H x W", semantic="segmentation_mask", detail="mask supervision targets"),
        ]

    def ir_output_ports(self) -> List[IRTensorPortSpec]:
        return [
            IRTensorPortSpec("classifier_logits", "Classifier logits", io="output", dtype="float32", shape="B x C", semantic="class_logits", detail="stage-0 semantic predictions"),
            IRTensorPortSpec("classifier_mask_logits", "Mask logits", io="output", dtype="float32", shape="B x 1 x H x W", semantic="mask_logits", detail="stage-0 mask decoder predictions"),
        ]

    def ir_loss_terms(self) -> List[IRLossTermSpec]:
        return [
            IRLossTermSpec("stage0_classifier_loss", "Stage 0 classifier objective", kind="bce+cosine+mask_bce", optimizer_targets=["classifier_optimizer"], source_ports=["pregestation_images", "pregestation_targets", "pregestation_masks", "classifier_logits", "classifier_mask_logits"], detail="BCE supervision plus semantic cosine loss plus mask BCE"),
        ]

    def ir_state_inputs(self) -> List[IRStateSpec]:
        return [
            IRStateSpec("label_embedding_bank", "Label embedding bank", role="semantic_teacher", detail="semantic cosine targets"),
        ]

    def declare_training_mechanics(self) -> Dict[str, Any]:
        return {
            "module_family": "classifier",
            "module_label": "Stage 0 Pregestation",
            "summary": "Synthetic geometry supervision warms the shared classifier before real-image stages.",
            "inputs": [
                {"id": "pregestation_loader", "label": "Pregestation deck", "kind": "dataset", "detail": "logic RGB images, multihot labels, masks"},
                {"id": "classifier_model", "label": "Classifier weights", "kind": "model", "detail": "TinyConv backbone plus semantic head"},
                {"id": "label_embedding_bank", "label": "Label embedding bank", "kind": "state", "detail": "semantic cosine target vectors"},
            ],
            "losses": [
                {"id": "loss_stage0_classifier", "label": "BCE + cosine + mask BCE", "kind": "loss", "detail": "Stage-0 classifier supervision objective"},
            ],
            "outputs": [
                {"id": "classifier_model", "label": "Classifier weights", "kind": "model", "detail": "updated shared classifier state"},
            ],
            "flows": [
                {"source": "pregestation_loader", "target": "self", "label": "logic images + labels + masks", "kind": "consume"},
                {"source": "classifier_model", "target": "self", "label": "trainable weights", "kind": "consume"},
                {"source": "label_embedding_bank", "target": "self", "label": "semantic targets", "kind": "condition"},
                {"source": "self", "target": "loss_stage0_classifier", "label": "logits + mask logits", "kind": "predict"},
                {"source": "pregestation_loader", "target": "loss_stage0_classifier", "label": "targets + masks", "kind": "supervise"},
                {"source": "label_embedding_bank", "target": "loss_stage0_classifier", "label": "cosine anchors", "kind": "supervise"},
                {"source": "loss_stage0_classifier", "target": "classifier_model", "label": "AdamW update", "kind": "optimize"},
            ],
        }

    def should_run(self, ctx: PipelineContext) -> bool:
        # Always attempt until gate passes; harmless to run again after passing.
        return ctx.pregestation_loader is not None and ctx.classifier is not None

    def execute(self, ctx: PipelineContext) -> None:
        ensure_vocab_lora_active(ctx, self.cfg)
        from pipeline.nodes.base import make_training_progress_callback
        preview_callback = make_classifier_step_preview_callback(ctx, self.node_id)
        weight_update_callback = make_runtime_weight_publish_callback(
            ctx,
            model_name="classifier",
            model=ctx.classifier,
            node_id=self.node_id,
        )
        result = _run_classifier_refresh_epochs(
            classifier=ctx.classifier,
            optimizer=ctx.classifier_optimizer,
            loader=ctx.pregestation_loader,
            device=ctx.device,
            epochs=self.cfg.stage0_epochs,
            amp_enabled=ctx.amp_enabled,
            amp_dtype=ctx.amp_dtype,
            grad_clip=self.cfg.grad_clip,
            grad_accum_steps=self.cfg.grad_accum_steps,
            semantic_cosine_weight=self.cfg.semantic_cosine_weight,
            grad_scaler=ctx.classifier_grad_scaler,
            channels_last=self.cfg.channels_last,
            stage_label="stage0_pregestation",
            log_every=self.cfg.log_every,
            step_preview_callback=preview_callback,
            progress_callback=make_training_progress_callback(
                ctx,
                self.node_id,
                "stage0_pregestation",
                publish_loss=(preview_callback is None),
            ),
            stop_requested=ctx.stop_requested,
            pause_requested=ctx.paused,
            ipc_pump=getattr(ctx.viewer_proxy, "pump", None),
            weight_update_callback=weight_update_callback,
            args=ctx.args,
        )

        loss = float(result.get("loss", float("inf")))
        ctx.log_metric("stage0", "loss", loss)
        _log(f"[stage0] loss={loss:.4f}")


# ---------------------------------------------------------------------------
# Stage 1 — Gestation training node
# ---------------------------------------------------------------------------

class GestationTrainNode(IRTrainingNode):
    """Stage 1: Train classifier on bootstrap primitive symbols.

    Requires Gate 0 to have passed.  Teaches the classifier to recognise
    noise profile terms, basic colours, directions, and geometric structures
    before exposure to Berkeley SBD.

    Gate progression is evaluated by a separate passive Gate-1 eval node
    against the gestation validation split.
    """

    node_id = "stage_1_gestation"
    description = "Stage 1: gestation classifier training (bootstrap primitives)"
    required_gates = ["gate_pregestation"]
    gpu_models = ["classifier"]
    model_attr = "classifier"
    optimizer_attrs = ["classifier_optimizer"]

    def __init__(self, cfg: ClassifierConfig) -> None:
        self.cfg = cfg

    def ir_input_ports(self) -> List[IRTensorPortSpec]:
        return [
            IRTensorPortSpec("gestation_images", "Gestation images", io="input", dtype="float32", shape="B x 3 x H x W", semantic="image_batch", detail="bootstrap symbol render batch"),
            IRTensorPortSpec("gestation_targets", "Gestation targets", io="input", dtype="float32", shape="B x C", semantic="multilabel_targets", detail="multi-hot classifier targets"),
            IRTensorPortSpec("gestation_masks", "Gestation masks", io="input", dtype="float32", shape="B x 1 x H x W", semantic="segmentation_mask", detail="mask supervision targets"),
        ]

    def ir_output_ports(self) -> List[IRTensorPortSpec]:
        return [
            IRTensorPortSpec("classifier_logits", "Classifier logits", io="output", dtype="float32", shape="B x C", semantic="class_logits", detail="stage-1 semantic predictions"),
            IRTensorPortSpec("classifier_mask_logits", "Mask logits", io="output", dtype="float32", shape="B x 1 x H x W", semantic="mask_logits", detail="stage-1 mask decoder predictions"),
        ]

    def ir_loss_terms(self) -> List[IRLossTermSpec]:
        return [
            IRLossTermSpec("stage1_classifier_loss", "Stage 1 classifier objective", kind="bce+cosine+mask_bce", optimizer_targets=["classifier_optimizer"], source_ports=["gestation_images", "gestation_targets", "gestation_masks", "classifier_logits", "classifier_mask_logits"], detail="BCE supervision plus semantic cosine loss plus mask BCE"),
        ]

    def ir_state_inputs(self) -> List[IRStateSpec]:
        return [
            IRStateSpec("label_embedding_bank", "Label embedding bank", role="semantic_teacher", detail="semantic cosine targets"),
        ]

    def declare_training_mechanics(self) -> Dict[str, Any]:
        return {
            "module_family": "classifier",
            "module_label": "Stage 1 Gestation",
            "summary": "Bootstrap symbol supervision extends the classifier from synthetic logic to primitive semantic images.",
            "inputs": [
                {"id": "gestation_loader", "label": "Gestation deck", "kind": "dataset", "detail": "bootstrap symbols, labels, masks"},
                {"id": "classifier_model", "label": "Classifier weights", "kind": "model", "detail": "shared TinyConv classifier"},
                {"id": "label_embedding_bank", "label": "Label embedding bank", "kind": "state", "detail": "semantic cosine target vectors"},
            ],
            "losses": [
                {"id": "loss_stage1_classifier", "label": "BCE + cosine + mask BCE", "kind": "loss", "detail": "Gestation classifier supervision objective"},
            ],
            "outputs": [
                {"id": "classifier_model", "label": "Classifier weights", "kind": "model", "detail": "updated shared classifier state"},
            ],
            "flows": [
                {"source": "gestation_loader", "target": "self", "label": "symbol images + labels + masks", "kind": "consume"},
                {"source": "classifier_model", "target": "self", "label": "trainable weights", "kind": "consume"},
                {"source": "label_embedding_bank", "target": "self", "label": "semantic targets", "kind": "condition"},
                {"source": "self", "target": "loss_stage1_classifier", "label": "logits + mask logits", "kind": "predict"},
                {"source": "gestation_loader", "target": "loss_stage1_classifier", "label": "targets + masks", "kind": "supervise"},
                {"source": "label_embedding_bank", "target": "loss_stage1_classifier", "label": "cosine anchors", "kind": "supervise"},
                {"source": "loss_stage1_classifier", "target": "classifier_model", "label": "AdamW update", "kind": "optimize"},
            ],
        }

    def should_run(self, ctx: PipelineContext) -> bool:
        return ctx.gestation_loader is not None and ctx.classifier is not None

    def execute(self, ctx: PipelineContext) -> None:
        ensure_vocab_lora_active(ctx, self.cfg)
        from pipeline.nodes.base import make_training_progress_callback
        preview_callback = make_classifier_step_preview_callback(ctx, self.node_id)
        weight_update_callback = make_runtime_weight_publish_callback(
            ctx,
            model_name="classifier",
            model=ctx.classifier,
            node_id=self.node_id,
        )
        result = _run_classifier_refresh_epochs(
            classifier=ctx.classifier,
            optimizer=ctx.classifier_optimizer,
            loader=ctx.gestation_loader,
            device=ctx.device,
            epochs=self.cfg.stage1_epochs,
            amp_enabled=ctx.amp_enabled,
            amp_dtype=ctx.amp_dtype,
            grad_clip=self.cfg.grad_clip,
            grad_accum_steps=self.cfg.grad_accum_steps,
            semantic_cosine_weight=self.cfg.semantic_cosine_weight,
            grad_scaler=ctx.classifier_grad_scaler,
            channels_last=self.cfg.channels_last,
            stage_label="stage1_gestation",
            log_every=self.cfg.log_every,
            step_preview_callback=preview_callback,
            progress_callback=make_training_progress_callback(
                ctx,
                self.node_id,
                "stage1_gestation",
                publish_loss=(preview_callback is None),
            ),
            stop_requested=ctx.stop_requested,
            pause_requested=ctx.paused,
            ipc_pump=getattr(ctx.viewer_proxy, "pump", None),
            weight_update_callback=weight_update_callback,
            args=ctx.args,
        )

        loss = float(result.get("loss", float("inf")))
        ctx.log_metric("stage1", "loss", loss)
        _log(f"[stage1] loss={loss:.4f}")


# ---------------------------------------------------------------------------
# Stage 2 — Berkeley SBD refresh training node
# ---------------------------------------------------------------------------

class BerkeleyRefreshTrainNode(IRTrainingNode):
    """Stage 2: Full classifier refresh on Berkeley SBD + external payload images.

    Requires Gates 0+1.  Runs every N rounds (controlled by orchestrator edge
    condition).  This is the main real-image supervision stage.

    Gate 2 passes when confidence AND macro-F1 exceed their targets AND
    optionally loss drops below stage2_loss_target.
    """

    node_id = "stage_2_berkeley"
    description = "Stage 2: Berkeley SBD refresh (full multi-label classification)"
    required_gates = ["gate_pregestation", "gate_gestation"]
    gpu_models = ["classifier"]
    model_attr = "classifier"
    optimizer_attrs = ["classifier_optimizer"]

    def __init__(self, cfg: ClassifierConfig) -> None:
        self.cfg = cfg

    def ir_input_ports(self) -> List[IRTensorPortSpec]:
        return [
            IRTensorPortSpec("berkeley_images", "Berkeley images", io="input", dtype="float32", shape="B x 3 x H x W", semantic="image_batch", detail="real-image semantic supervision batch"),
            IRTensorPortSpec("berkeley_targets", "Berkeley targets", io="input", dtype="float32", shape="B x C", semantic="multilabel_targets", detail="multi-hot semantic targets"),
            IRTensorPortSpec("berkeley_masks", "Berkeley masks", io="input", dtype="float32", shape="B x 1 x H x W", semantic="segmentation_mask", detail="mask supervision targets"),
        ]

    def ir_output_ports(self) -> List[IRTensorPortSpec]:
        return [
            IRTensorPortSpec("classifier_logits", "Classifier logits", io="output", dtype="float32", shape="B x C", semantic="class_logits", detail="stage-2 semantic predictions"),
            IRTensorPortSpec("classifier_mask_logits", "Mask logits", io="output", dtype="float32", shape="B x 1 x H x W", semantic="mask_logits", detail="stage-2 mask decoder predictions"),
        ]

    def ir_loss_terms(self) -> List[IRLossTermSpec]:
        return [
            IRLossTermSpec("stage2_classifier_loss", "Stage 2 classifier objective", kind="bce+cosine+mask_bce", optimizer_targets=["classifier_optimizer"], source_ports=["berkeley_images", "berkeley_targets", "berkeley_masks", "classifier_logits", "classifier_mask_logits"], detail="Real-image BCE supervision plus semantic cosine loss plus mask BCE"),
        ]

    def ir_state_inputs(self) -> List[IRStateSpec]:
        return [
            IRStateSpec("label_embedding_bank", "Label embedding bank", role="semantic_teacher", detail="semantic cosine targets"),
        ]

    def declare_training_mechanics(self) -> Dict[str, Any]:
        return {
            "module_family": "classifier",
            "module_label": "Stage 2 Berkeley Refresh",
            "summary": "Full real-image semantic refresh trains the classifier on Berkeley and payload supervision.",
            "inputs": [
                {"id": "berkeley_refresh_loader", "label": "Berkeley refresh deck", "kind": "dataset", "detail": "real images, multihot labels, masks"},
                {"id": "classifier_model", "label": "Classifier weights", "kind": "model", "detail": "shared TinyConv classifier"},
                {"id": "label_embedding_bank", "label": "Label embedding bank", "kind": "state", "detail": "semantic cosine target vectors"},
            ],
            "losses": [
                {"id": "loss_stage2_classifier", "label": "BCE + cosine + mask BCE", "kind": "loss", "detail": "Real-image classifier supervision objective"},
            ],
            "outputs": [
                {"id": "classifier_model", "label": "Classifier weights", "kind": "model", "detail": "updated shared classifier state"},
            ],
            "flows": [
                {"source": "berkeley_refresh_loader", "target": "self", "label": "real images + labels + masks", "kind": "consume"},
                {"source": "classifier_model", "target": "self", "label": "trainable weights", "kind": "consume"},
                {"source": "label_embedding_bank", "target": "self", "label": "semantic targets", "kind": "condition"},
                {"source": "self", "target": "loss_stage2_classifier", "label": "logits + mask logits", "kind": "predict"},
                {"source": "berkeley_refresh_loader", "target": "loss_stage2_classifier", "label": "targets + masks", "kind": "supervise"},
                {"source": "label_embedding_bank", "target": "loss_stage2_classifier", "label": "cosine anchors", "kind": "supervise"},
                {"source": "loss_stage2_classifier", "target": "classifier_model", "label": "AdamW update", "kind": "optimize"},
            ],
        }

    def execute(self, ctx: PipelineContext) -> None:
        ensure_vocab_lora_active(ctx, self.cfg)
        from pipeline.nodes.base import make_training_progress_callback
        preview_callback = make_classifier_step_preview_callback(ctx, self.node_id)
        weight_update_callback = make_runtime_weight_publish_callback(
            ctx,
            model_name="classifier",
            model=ctx.classifier,
            node_id=self.node_id,
        )
        result = _run_classifier_refresh_epochs(
            classifier=ctx.classifier,
            optimizer=ctx.classifier_optimizer,
            loader=ctx.berkeley_refresh_loader,
            device=ctx.device,
            epochs=self.cfg.stage2_epochs,
            amp_enabled=ctx.amp_enabled,
            amp_dtype=ctx.amp_dtype,
            grad_clip=self.cfg.grad_clip,
            grad_accum_steps=self.cfg.grad_accum_steps,
            semantic_cosine_weight=self.cfg.semantic_cosine_weight,
            grad_scaler=ctx.classifier_grad_scaler,
            channels_last=self.cfg.channels_last,
            stage_label="stage2_berkeley",
            log_every=self.cfg.log_every,
            step_preview_callback=preview_callback,
            progress_callback=make_training_progress_callback(
                ctx,
                self.node_id,
                "stage2_berkeley",
                publish_loss=(preview_callback is None),
            ),
            stop_requested=ctx.stop_requested,
            pause_requested=ctx.paused,
            ipc_pump=getattr(ctx.viewer_proxy, "pump", None),
            weight_update_callback=weight_update_callback,
            args=ctx.args,
            remap_targets_from_terms=True,
            active_class_names=list(ctx.class_names),
            source_class_names=list(ctx.class_names),
        )

        loss = float(result.get("loss", float("inf")))
        ctx.log_metric("stage2", "loss", loss)
        _log(f"[stage2] loss={loss:.4f}")


# ---------------------------------------------------------------------------
# Stage C — LoRA slot training node
# ---------------------------------------------------------------------------

class LoRARoundNode(IRTrainingNode):
    """Stage C: Per-term LoRA adapter slot training.

    Requires all base gates.  Sweeps every planned vocab-LoRA slot from the
    current churn requirement plan in a single execution.  For each slot the
    slot's vocabulary is activated, the corresponding LoRA adapter is
    installed/restored, the backbone + adapter are trained together for
    stageC_steps_per_slot steps on Berkeley payload rows, and the adapter
    state is snapshotted into the slot library.

    This turns the system from limited-vocabulary into anything-vocabulary:
    the backbone is pressured toward zero-hot generalization while each
    vocab LoRA absorbs term-specific detail.
    """

    node_id = "stage_c_lora"
    description = "Stage C: LoRA slot switching and per-term Berkeley training"
    required_gates = ["gate_pregestation", "gate_gestation", "gate_berkeley"]
    gpu_models = ["classifier"]
    model_attr = "classifier"
    optimizer_attrs = ["classifier_optimizer"]

    def __init__(self, cfg: ClassifierConfig) -> None:
        self.cfg = cfg

    def ir_input_ports(self) -> List[IRTensorPortSpec]:
        return [
            IRTensorPortSpec("berkeley_images", "Berkeley images", io="input", dtype="float32", shape="B x 3 x H x W", semantic="image_batch", detail="real-image specialization batch"),
            IRTensorPortSpec("berkeley_targets", "Berkeley targets", io="input", dtype="float32", shape="B x C", semantic="multilabel_targets", detail="slot-specific semantic targets"),
            IRTensorPortSpec("berkeley_masks", "Berkeley masks", io="input", dtype="float32", shape="B x 1 x H x W", semantic="segmentation_mask", detail="mask supervision targets"),
        ]

    def ir_output_ports(self) -> List[IRTensorPortSpec]:
        return [
            IRTensorPortSpec("classifier_logits", "Classifier logits", io="output", dtype="float32", shape="B x C", semantic="class_logits", detail="LoRA-specialized semantic predictions"),
            IRTensorPortSpec("classifier_lora_state", "LoRA slot state", io="output", dtype="float32", shape="slot tensors", semantic="adapter_state", detail="snapshotted slot parameters"),
        ]

    def ir_loss_terms(self) -> List[IRLossTermSpec]:
        return [
            IRLossTermSpec("stagec_lora_loss", "Stage C LoRA objective", kind="bce+cosine+mask_bce", optimizer_targets=["classifier_optimizer"], source_ports=["berkeley_images", "berkeley_targets", "berkeley_masks", "classifier_logits"], detail="Per-slot classifier supervision on Berkeley rows"),
        ]

    def ir_state_inputs(self) -> List[IRStateSpec]:
        return [
            IRStateSpec("label_embedding_bank", "Label embedding bank", role="semantic_teacher", detail="semantic cosine targets"),
            IRStateSpec("active_churn_terms", "Active churn terms", role="slot_router", detail="maps vocab groups onto LoRA slots"),
        ]

    def declare_training_mechanics(self) -> Dict[str, Any]:
        return {
            "module_family": "classifier",
            "module_label": "Stage C LoRA Round",
            "summary": "Per-term LoRA slots specialize the classifier on Berkeley rows without discarding the shared backbone.",
            "inputs": [
                {"id": "berkeley_refresh_loader", "label": "Berkeley refresh deck", "kind": "dataset", "detail": "real-image semantic supervision"},
                {"id": "classifier_model", "label": "Classifier + active LoRA", "kind": "model", "detail": "shared classifier with slot adapters"},
                {"id": "label_embedding_bank", "label": "Label embedding bank", "kind": "state", "detail": "semantic cosine target vectors"},
                {"id": "active_churn_terms", "label": "Active churn terms", "kind": "state", "detail": "term groups mapped onto LoRA slots"},
            ],
            "losses": [
                {"id": "loss_stagec_lora", "label": "BCE + cosine + mask BCE", "kind": "loss", "detail": "LoRA slot supervision objective"},
            ],
            "outputs": [
                {"id": "classifier_model", "label": "Classifier + active LoRA", "kind": "model", "detail": "adapter-updated classifier state"},
                {"id": "classifier_lora_slots", "label": "LoRA slot snapshots", "kind": "state", "detail": "snapshotted per-term adapter states"},
            ],
            "flows": [
                {"source": "berkeley_refresh_loader", "target": "self", "label": "real-image supervision", "kind": "consume"},
                {"source": "classifier_model", "target": "self", "label": "shared weights + active slot", "kind": "consume"},
                {"source": "label_embedding_bank", "target": "self", "label": "semantic targets", "kind": "condition"},
                {"source": "active_churn_terms", "target": "self", "label": "slot assignment", "kind": "condition"},
                {"source": "self", "target": "loss_stagec_lora", "label": "slot logits + mask logits", "kind": "predict"},
                {"source": "berkeley_refresh_loader", "target": "loss_stagec_lora", "label": "targets + masks", "kind": "supervise"},
                {"source": "label_embedding_bank", "target": "loss_stagec_lora", "label": "cosine anchors", "kind": "supervise"},
                {"source": "loss_stagec_lora", "target": "classifier_model", "label": "LoRA weight update", "kind": "optimize"},
                {"source": "self", "target": "classifier_lora_slots", "label": "snapshot trained slots", "kind": "emit"},
            ],
        }

    def execute(self, ctx: PipelineContext) -> None:
        if not self.cfg.lora_enabled or ctx.classifier is None or ctx.berkeley_refresh_loader is None:
            return

        from wav_ml_models import (
            ensure_tiny_classifier_lora_slot,
            install_tiny_classifier_lora,
            tiny_classifier_lora_snapshot,
            restore_tiny_classifier_lora_snapshot,
            set_tiny_classifier_lora_state,
        )
        from pipeline.nodes.vocab_node import activate_vocab_lora_slot, get_all_planned_lora_slots

        planned_slots = get_all_planned_lora_slots(ctx)

        # -- Install LoRA adapters once ------------------------------------
        install_tiny_classifier_lora(
            ctx.classifier,
            rank=int(self.cfg.stageC_lora_rank),
            alpha=float(self.cfg.stageC_lora_alpha),
        )

        weight_update_callback = make_runtime_weight_publish_callback(
            ctx,
            model_name="classifier",
            model=ctx.classifier,
            node_id=self.node_id,
        )

        # -- Sweep every planned slot --------------------------------------
        slots_trained = 0
        for slot_def in planned_slots:
            slot_signature = str(slot_def.get("signature", "")).strip()
            slot_name = str(slot_def.get("slot_name", f"vocab_{slot_signature}"))
            slot_terms = list(slot_def.get("terms") or [])
            if not slot_signature or not slot_terms:
                continue

            # Activate this slot's vocabulary in the pipeline context
            activate_vocab_lora_slot(ctx, slot_def)

            # Prepare LoRA slot
            ensure_tiny_classifier_lora_slot(ctx.classifier, slot_name=slot_name)
            existing_snapshot = (
                getattr(ctx, "lora_slot_snapshots", {}).get(str(slot_signature))
                or getattr(ctx, "lora_slot_snapshots", {}).get(str(slot_name))
            )
            if isinstance(existing_snapshot, dict):
                restore_tiny_classifier_lora_snapshot(ctx.classifier, existing_snapshot)

            # Backbone + LoRA train together
            set_tiny_classifier_lora_state(ctx.classifier, slot_name=slot_name, lora_only=False)
            ctx.lora_active_slot = str(slot_name)

            _run_classifier_refresh_epochs(
                classifier=ctx.classifier,
                optimizer=ctx.classifier_optimizer,
                loader=ctx.berkeley_refresh_loader,
                device=ctx.device,
                epochs=1,
                max_steps=self.cfg.stageC_steps_per_slot,
                amp_enabled=ctx.amp_enabled,
                amp_dtype=ctx.amp_dtype,
                grad_clip=self.cfg.grad_clip,
                grad_accum_steps=self.cfg.grad_accum_steps,
                semantic_cosine_weight=self.cfg.semantic_cosine_weight,
                grad_scaler=ctx.classifier_grad_scaler,
                channels_last=self.cfg.channels_last,
                stage_label=f"stageC_lora_{slot_name}",
                weight_update_callback=weight_update_callback,
                args=ctx.args,
                remap_targets_from_terms=True,
                active_class_names=list(ctx.class_names),
                source_class_names=list(ctx.supervised_class_names),
                stop_requested=ctx.stop_requested,
                pause_requested=ctx.paused,
                ipc_pump=getattr(ctx.viewer_proxy, "pump", None),
            )

            # Snapshot trained slot
            slot_snapshot = tiny_classifier_lora_snapshot(ctx.classifier, slot_name=slot_name)
            ctx.lora_slot_snapshots[str(slot_signature)] = dict(slot_snapshot)
            ctx.lora_slot_snapshots[str(slot_name)] = dict(slot_snapshot)

            library_entry = dict(getattr(ctx, "vocab_lora_library", {}).get(str(slot_signature), {}) or {})
            library_entry.update(
                {
                    "signature": str(slot_signature),
                    "slot_name": str(slot_name),
                    "terms": list(slot_terms),
                    "snapshot_present": True,
                    "trained_rounds": int(library_entry.get("trained_rounds", 0)) + 1,
                    "last_trained_round": int(getattr(ctx, "round_id", 0) or 0),
                }
            )
            ctx.vocab_lora_library[str(slot_signature)] = dict(library_entry)
            slots_trained += 1

            _log(
                f"[stageC] slot {slots_trained}/{len(planned_slots)} "
                f"name={slot_name} terms={len(slot_terms)} signature={slot_signature[:12]}"
            )

        # After sweeping all slots, leave the last trained slot active.
        # The 50 open vocab slots must never be used without a LoRA in place.
        # If no slots were trained, re-ensure the current vocab LoRA is active.
        if slots_trained == 0:
            ensure_vocab_lora_active(ctx, self.cfg)
        _log(
            f"[stageC] LoRA sweep complete: {slots_trained} slot(s) trained, "
            f"active_slot={ctx.lora_active_slot}"
        )


# ---------------------------------------------------------------------------
# Fake-class feedback node
# ---------------------------------------------------------------------------

class FakeClassFeedbackNode(IRTrainingNode):
    """Fake-class refresh: train the classifier to detect GAN outputs.

    Requires the generator to exist and all base gates to have passed.
    Uses discriminator confidence as a curriculum weight — samples the
    generator gets past the discriminator are the hardest/most instructive.

    The classifier learns a sentinel ``gan image`` class so that GAN artefacts
    in real-data batches do not pollute the semantic signal.
    """

    node_id = "stage_fake_feedback"
    description = "Fake-class feedback: train classifier to detect GAN outputs"
    required_gates = ["gate_pregestation", "gate_gestation"]
    gpu_models = ["classifier", "generator", "discriminator"]
    model_attr = "classifier"
    extra_model_attrs = ["generator", "discriminator"]
    optimizer_attrs = ["classifier_optimizer"]

    def __init__(self, cfg: ClassifierConfig) -> None:
        self.cfg = cfg

    def ir_input_ports(self) -> List[IRTensorPortSpec]:
        return [
            IRTensorPortSpec("fake_images", "Generated images", io="input", dtype="float32", shape="B x 3 x H x W", semantic="generated_image_batch", detail="conditioned GAN samples"),
            IRTensorPortSpec("fake_condition_targets", "Condition targets", io="input", dtype="float32", shape="B x C", semantic="condition_vectors", detail="conditioning vectors reused as supervision context"),
            IRTensorPortSpec("discriminator_confidence", "Discriminator confidence", io="input", dtype="float32", shape="B x 1", semantic="curriculum_weight", detail="hardness weighting from discriminator"),
        ]

    def ir_output_ports(self) -> List[IRTensorPortSpec]:
        return [
            IRTensorPortSpec("classifier_fake_logits", "Fake-class logits", io="output", dtype="float32", shape="B x C", semantic="class_logits", detail="classifier response to GAN samples"),
        ]

    def ir_loss_terms(self) -> List[IRLossTermSpec]:
        return [
            IRLossTermSpec("fake_class_loss", "Fake-class objective", kind="sentinel_bce", optimizer_targets=["classifier_optimizer"], source_ports=["fake_images", "fake_condition_targets", "discriminator_confidence", "classifier_fake_logits"], detail="BCE objective for GAN-image sentinel detection"),
        ]

    def ir_state_inputs(self) -> List[IRStateSpec]:
        return [
            IRStateSpec("payload_conditions", "Semantic condition bank", role="conditioning_bank", detail="semantic condition vectors used to sample fake images"),
        ]

    def declare_training_mechanics(self) -> Dict[str, Any]:
        return {
            "module_family": "classifier",
            "module_label": "Fake-Class Feedback",
            "summary": "GAN samples become a sentinel supervision signal so the classifier learns to reject generated artefacts.",
            "inputs": [
                {"id": "classifier_model", "label": "Classifier weights", "kind": "model", "detail": "shared TinyConv classifier"},
                {"id": "gan_generator_model", "label": "GAN generator", "kind": "model", "detail": "produces conditioned fake images"},
                {"id": "gan_discriminator_model", "label": "GAN discriminator", "kind": "model", "detail": "weights fake-sample hardness by confidence"},
                {"id": "semantic_condition_bank", "label": "Semantic condition bank", "kind": "state", "detail": "class-conditioned prompts for fake sampling"},
            ],
            "losses": [
                {"id": "loss_fake_class", "label": "Fake sentinel BCE", "kind": "loss", "detail": "classifier fake-image detection objective"},
            ],
            "outputs": [
                {"id": "classifier_model", "label": "Classifier weights", "kind": "model", "detail": "updated fake-detector-aware classifier state"},
            ],
            "flows": [
                {"source": "gan_generator_model", "target": "self", "label": "conditioned fake samples", "kind": "consume"},
                {"source": "gan_discriminator_model", "target": "self", "label": "confidence curriculum", "kind": "condition"},
                {"source": "semantic_condition_bank", "target": "self", "label": "conditioning vectors", "kind": "condition"},
                {"source": "classifier_model", "target": "self", "label": "trainable weights", "kind": "consume"},
                {"source": "self", "target": "loss_fake_class", "label": "fake logits", "kind": "predict"},
                {"source": "gan_discriminator_model", "target": "loss_fake_class", "label": "hardness weights", "kind": "supervise"},
                {"source": "semantic_condition_bank", "target": "loss_fake_class", "label": "sentinel targets", "kind": "supervise"},
                {"source": "loss_fake_class", "target": "classifier_model", "label": "AdamW update", "kind": "optimize"},
            ],
        }

    def should_run(self, ctx: PipelineContext) -> bool:
        if not super().should_run(ctx):
            return False
        return (
            self.cfg.fake_class_enabled
            and ctx.generator is not None
            and ctx.discriminator is not None
        )

    def execute(self, ctx: PipelineContext) -> None:
        ensure_vocab_lora_active(ctx, self.cfg)
        from pipeline.nodes.base import make_training_progress_callback
        preview_callback = make_classifier_step_preview_callback(ctx, self.node_id)
        weight_update_callback = make_runtime_weight_publish_callback(
            ctx,
            model_name="classifier",
            model=ctx.classifier,
            node_id=self.node_id,
        )
        fake_label_vector = _resolve_fake_feedback_label_vector(ctx)
        if fake_label_vector is None:
            _log("[fake-class] skipped: fake-sentinel label vector unavailable")
            return
        result = _run_fake_class_refresh_epochs(
            classifier=ctx.classifier,
            generator=ctx.generator,
            discriminator=ctx.discriminator,
            payload_conditions=list(getattr(ctx, "payload_conditions", []) or []),
            condition_num_classes=_infer_condition_vector_width(
                payload_conditions=list(getattr(ctx, "payload_conditions", []) or []),
                class_names=list(getattr(ctx, "class_names", []) or []),
            ),
            fake_label_vector=fake_label_vector,
            z_dim=max(8, int(getattr(ctx.generator, "z_dim", getattr(ctx.args, "generator_z_dim", 128)) or 128)),
            device=ctx.device,
            epochs=max(1, int(getattr(ctx.args, "generator_fake_feedback_epochs", 1) or 1)),
            steps_per_epoch=max(1, int(self.cfg.fake_class_steps)),
            batch_size=max(1, int(self.cfg.fake_class_batch_size)),
            lr=float(getattr(ctx.args, "generator_fake_feedback_lr", 6e-4) or 6e-4),
            weight_decay=float(getattr(ctx.args, "generator_fake_feedback_weight_decay", 1e-4) or 1e-4),
            lr_sine_cycles=float(getattr(ctx.args, "lr_sine_cycles", 1.0) or 1.0),
            lr_sine_frequency=float(getattr(ctx.args, "lr_sine_frequency", 0.0) or 0.0),
            lr_sine_tail_fraction=float(getattr(ctx.args, "lr_sine_tail_fraction", 0.15) or 0.15),
            lr_sine_min_scale=float(getattr(ctx.args, "lr_sine_min_scale", 0.0) or 0.0),
            amp_enabled=ctx.amp_enabled,
            amp_dtype=ctx.amp_dtype,
            channels_last=self.cfg.channels_last,
            grad_accum_steps=max(1, int(self.cfg.grad_accum_steps)),
            log_every=max(0, int(getattr(ctx.args, "generator_fake_feedback_log_every", 0) or 0)),
            include_condition_targets=bool(getattr(ctx.args, "generator_fake_feedback_include_condition_targets", True)),
            fake_vector_weight=float(getattr(ctx.args, "generator_fake_feedback_vector_weight", 1.0) or 1.0),
            condition_target_weight=float(
                getattr(
                    ctx.args,
                    "generator_fake_feedback_condition_weight",
                    self.cfg.fake_class_disc_weight,
                )
                or self.cfg.fake_class_disc_weight
            ),
            disc_conf_temperature=float(getattr(ctx.args, "generator_fake_feedback_disc_conf_temperature", 1.0) or 1.0),
            disc_conf_floor=float(getattr(ctx.args, "generator_fake_feedback_disc_conf_floor", 0.25) or 0.25),
            balance_disc_groups=bool(getattr(ctx.args, "generator_fake_feedback_disc_balance_groups", True)),
            step_preview_callback=preview_callback,
            progress_callback=make_training_progress_callback(
                ctx,
                self.node_id,
                "stage_fake_feedback",
                publish_loss=(preview_callback is None),
            ),
            stop_requested=ctx.stop_requested,
            weight_update_callback=weight_update_callback,
            seed=int(getattr(ctx.args, "seed", 0) or 0) + 19000 + (int(getattr(ctx, "round_id", 0)) * 17),
            grad_clip=float(self.cfg.grad_clip),
        )

        loss = float(result.get("loss", 0.0))
        ctx.log_metric("stagefake", "loss", loss)
        _log("[fake-class] feedback epoch complete")


# ---------------------------------------------------------------------------
# Gate replica sync node
# ---------------------------------------------------------------------------

class SyncGateReplicaNode(PipelineNode):
    """Keep the frozen gate_classifier in sync with the training classifier."""

    node_id = "sync_gate_replica"
    description = "Sync frozen gate_classifier from main classifier"
    runtime_object_type = "service"
    runtime_faculty = "housekeeping"
    gpu_models = ["classifier"]

    def __init__(self, cfg: ClassifierConfig) -> None:
        self.cfg = cfg

    def should_run(self, ctx: PipelineContext) -> bool:
        return ctx.classifier is not None

    def execute(self, ctx: PipelineContext) -> None:

        ctx.gate_classifier, info = _sync_gate_classifier_replica(
            source_classifier=ctx.classifier,
            gate_classifier=ctx.gate_classifier,
            gate_device=resolve_non_training_device(ctx),
            channels_last=self.cfg.channels_last,
        )
        if info.get("created"):
            _log(f"[gate-replica] created fresh replica on {info.get('device', 'cpu')}")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _term_to_slot_name(term: str) -> str:
    import re
    slug = re.sub(r"[^a-z0-9]+", "_", term.strip().lower()).strip("_")
    return slug or "slot"


def _build_gate_replica(model: nn.Module, ctx: PipelineContext) -> nn.Module:
    base = unwrap_compiled(model)
    target = resolve_non_training_device(ctx)
    if target.type == "cuda" and getattr(ctx, "gpu_residence", None) is not None:
        target = torch.device("cpu")
    replica = copy.deepcopy(base).to(device=target, dtype=torch.float32)
    replica.eval()
    freeze(replica)
    return replica


def _load_classifier_checkpoint(model: nn.Module, path: str, scope: str = "all") -> None:

    try:
        _apply_classifier_init(model, path, scope=str(scope))
        _log(f"[classifier] loaded checkpoint from {path} (scope={scope})")
    except Exception as exc:
        _log(f"[classifier] WARNING: could not load checkpoint {path!r}: {exc}")


def _log(msg: str) -> None:
    print(msg, flush=True)


def ensure_vocab_lora_active(ctx: PipelineContext, cfg: ClassifierConfig) -> Dict[str, Any]:
    """Ensure a LoRA adapter is installed and active whenever extra-vocab slots are in use.

    Design invariant: the 50 open/extra vocab slots must NEVER be used
    without a LoRA in place.  Only when labels are exclusively from
    the 101 built-in supervised vocabulary may the classifier operate
    without LoRA.  This helper enforces that invariant.
    """
    extra_terms = list(getattr(ctx, "active_extra_terms", []) or [])
    if not extra_terms:
        return {"needed": False, "reason": "no_extra_terms"}
    if ctx.classifier is None:
        return {"needed": True, "reason": "no_classifier", "installed": False}

    from wav_ml_models import (
        install_tiny_classifier_lora,
        ensure_tiny_classifier_lora_slot,
        set_tiny_classifier_lora_state,
        tiny_classifier_lora_modules,
        load_lora_slot_from_file,
    )
    from pipeline.nodes.vocab_node import _lora_library_dir

    # Step 1: ensure LoRA modules are installed on the classifier
    mods = tiny_classifier_lora_modules(ctx.classifier)
    if not mods:
        install_tiny_classifier_lora(
            ctx.classifier,
            rank=int(cfg.lora_rank),
            alpha=float(cfg.lora_alpha),
        )
        _log("[lora-guard] installed LoRA adapters (extra vocab active)")

    # Step 2: determine which slot should be active.
    # Always recompute the expected signature from the current extra_terms so
    # that a vocab change between stages (e.g. gestation → berkeley) is caught
    # immediately rather than silently reusing the stale gestation slot.
    expected_signature = hashlib.sha1(
        "|".join(str(t) for t in sorted(extra_terms)).encode("utf-8", errors="ignore")
    ).hexdigest()[:16]
    stored_signature = str(getattr(ctx, "vocab_lora_active_signature", "") or "").strip()
    if stored_signature and stored_signature != expected_signature:
        _log(
            f"[lora-guard] vocab changed (stored={stored_signature[:12]} "
            f"expected={expected_signature[:12]}) — switching LoRA slot"
        )
    signature = expected_signature
    ctx.vocab_lora_active_signature = signature

    slot_name = f"vocab_{signature}"
    stored_slot = str(getattr(ctx, "lora_active_slot", "") or "").strip()
    if stored_slot and stored_slot == slot_name:
        # Already on the right slot; keep the existing name (may be custom).
        slot_name = stored_slot

    # Step 3: ensure the slot exists, then load from library if available
    ensure_tiny_classifier_lora_slot(ctx.classifier, slot_name=slot_name)
    lib_dir = _lora_library_dir(ctx)
    if lib_dir is not None:
        slot_file = lib_dir / f"{slot_name}.pt"
        if slot_file.exists():
            load_lora_slot_from_file(ctx.classifier, slot_name, slot_file)

    # Step 4: activate — backbone + LoRA train together
    set_tiny_classifier_lora_state(ctx.classifier, slot_name=slot_name, lora_only=False)
    ctx.lora_active_slot = slot_name

    return {"needed": True, "installed": True, "slot_name": slot_name, "signature": signature}


# =========================================================================
# Functions extracted from wav_config_transformer_pipeline.py
# =========================================================================


def _sync_gate_classifier_replica(
    source_classifier: nn.Module,
    gate_classifier: Optional[nn.Module],
    gate_device: torch.device,
    channels_last: bool = False,
) -> Tuple[nn.Module, Dict[str, Any]]:
    def _sync_label_bank_state(source_base: nn.Module, gate_model: nn.Module) -> None:
        source_bank = getattr(source_base, "label_embed_bank", None)
        source_enabled_buf = getattr(source_base, "label_embed_enabled", None)
        source_enabled = bool(
            isinstance(source_enabled_buf, torch.Tensor)
            and int(source_enabled_buf.numel()) > 0
            and bool(int(source_enabled_buf.reshape(-1)[0].item()))
        )
        source_bank_valid = bool(
            source_enabled
            and isinstance(source_bank, torch.Tensor)
            and int(source_bank.ndim) == 2
            and int(source_bank.shape[0]) > 0
            and int(source_bank.shape[1]) > 0
        )
        if bool(source_bank_valid) and hasattr(gate_model, "set_label_embedding_bank"):
            gate_model.set_label_embedding_bank(
                source_bank.detach().to(device=gate_device, dtype=torch.float32),
                temperature=float(getattr(source_base, "embed_temperature", 10.0)),
            )
            return
        if hasattr(gate_model, "disable_label_embedding_bank"):
            gate_model.disable_label_embedding_bank()

    source_base = _unwrap_module_for_replica(source_classifier)
    created = False
    if gate_classifier is None:
        gate_classifier = copy.deepcopy(source_base)
        created = True
    gate_classifier = gate_classifier.to(device=gate_device, dtype=torch.float32)
    _sync_label_bank_state(source_base, gate_classifier)
    src_sd = source_base.state_dict()
    dst_keys = set(gate_classifier.state_dict().keys())
    if not dst_keys.issubset(src_sd.keys()):
        collapsed: Dict[str, torch.Tensor] = {}
        for k, v in src_sd.items():
            if ".slots." in k:
                continue
            plain = k.replace(".base.weight", ".weight").replace(".base.bias", ".bias")
            collapsed[plain] = v
        src_sd = collapsed
    gate_classifier.load_state_dict(src_sd, strict=not bool(created))
    gate_classifier.eval()
    for p in gate_classifier.parameters():
        p.requires_grad_(False)
    if bool(channels_last):
        gate_classifier = gate_classifier.to(memory_format=torch.channels_last)
    return gate_classifier, {
        "created": bool(created),
        "strict_load": bool(not bool(created)),
        "device": str(gate_device),
        "source_device": str(next(source_base.parameters()).device),
    }


def _apply_classifier_init(model: TinyConvClassifier, ckpt_path: str, scope: str):
    if not ckpt_path:
        return {"used": False, "loaded_keys": 0, "skipped_keys": 0, "path": ""}

    p = Path(ckpt_path)
    if not p.exists():
        raise FileNotFoundError(f"Classifier init checkpoint not found: {p}")

    blob = _torch_load_cpu(str(p))
    src_state = _strip_module_prefix(_extract_state_dict(blob))
    dst_state = model.state_dict()
    to_load = {}
    skipped = 0
    partial = 0

    for k, v in src_state.items():
        if scope == "features" and not k.startswith("features."):
            skipped += 1
            continue
        if k in dst_state and tuple(dst_state[k].shape) == tuple(v.shape):
            to_load[k] = v
        elif k in dst_state:
            patched = _try_partial_classifier_head_load(
                key=str(k),
                src_tensor=v,
                dst_tensor=dst_state[k],
            )
            if patched is not None:
                to_load[k] = patched
                partial += 1
            else:
                skipped += 1
        else:
            skipped += 1

    missing, unexpected = model.load_state_dict(to_load, strict=False)
    return {
        "used": True,
        "loaded_keys": len(to_load),
        "partial_keys": int(partial),
        "skipped_keys": skipped,
        "missing_after_load": len(missing),
        "unexpected_after_load": len(unexpected),
        "path": str(p.resolve()),
        "scope": scope,
    }


def _term_key(term: str) -> str:
    return re.sub(r"\s+", " ", str(term)).strip().lower()


def _remap_semantic_batch_to_active_vocab(
    yb: torch.Tensor,
    mb: torch.Tensor,
    batch_meta: Optional[Dict[str, Any]],
    *,
    active_class_names: Sequence[str],
    source_class_names: Optional[Sequence[str]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    meta = dict(batch_meta or {})
    terms_rows = list(meta.get("terms_rows") or [])
    if int(len(terms_rows)) != int(yb.shape[0]) or int(len(active_class_names)) <= 0:
        return yb, mb, meta
    active_names = [str(x) for x in list(active_class_names)]
    active_term_to_idx = {_term_key(name): int(i) for i, name in enumerate(active_names) if _term_key(name)}
    active_idx_to_term = {int(i): str(name) for i, name in enumerate(active_names)}
    source_names = list(source_class_names or [])
    if int(len(source_names)) != int(yb.shape[1]):
        source_names = active_names[: int(yb.shape[1])]
    source_idx_to_term = {
        int(i): str(name)
        for i, name in enumerate(source_names)
        if _term_key(name)
    }
    stack_list = list(meta.get("mask_stacks") or [])
    index_list = list(meta.get("mask_indices") or [])

    out_y_rows: List[torch.Tensor] = []
    out_m_rows: List[torch.Tensor] = []
    out_stack_rows: List[torch.Tensor] = []
    out_index_rows: List[torch.Tensor] = []

    for bi in range(int(yb.shape[0])):
        row_terms = [
            str(term)
            for term in list(terms_rows[int(bi)] or [])
            if _term_key(str(term))
        ]
        y_active_np = np.zeros((int(len(active_names)),), dtype=np.float32)
        for term in row_terms:
            dst_idx = int(active_term_to_idx.get(_term_key(term), -1))
            if 0 <= int(dst_idx) < int(y_active_np.size):
                y_active_np[int(dst_idx)] = 1.0
        src_positive = torch.nonzero(yb[int(bi)].detach().to(torch.float32) >= 0.5, as_tuple=False).reshape(-1).tolist()
        for src_idx in src_positive:
            src_term = str(source_idx_to_term.get(int(src_idx), "") or "")
            dst_idx = int(active_term_to_idx.get(_term_key(src_term), -1))
            if 0 <= int(dst_idx) < int(y_active_np.size):
                y_active_np[int(dst_idx)] = 1.0

        base_mask_np = np.asarray(mb[int(bi)].detach().to(torch.float32).cpu().numpy(), dtype=np.float32)
        if int(base_mask_np.ndim) == 3 and int(base_mask_np.shape[0]) == 1:
            base_mask_np = np.asarray(base_mask_np[0], dtype=np.float32)
        if int(base_mask_np.ndim) != 2:
            raise RuntimeError(f"Active-vocab remap requires [1,H,W] or [H,W] masks, got {tuple(base_mask_np.shape)}")
        height = int(base_mask_np.shape[0])
        width = int(base_mask_np.shape[1])

        mapped_stack_rows: List[np.ndarray] = []
        mapped_idx_rows: List[int] = []
        sample_stack = stack_list[int(bi)] if int(bi) < int(len(stack_list)) else None
        sample_idx = index_list[int(bi)] if int(bi) < int(len(index_list)) else None
        if sample_stack is not None and sample_idx is not None:
            stack_np = np.asarray(sample_stack, dtype=np.float32)
            idx_np = np.asarray(sample_idx, dtype=np.int64).reshape(-1)
            if int(stack_np.ndim) == 2:
                stack_np = stack_np[None, ...]
            pair_count = min(int(stack_np.shape[0]), int(idx_np.size)) if int(stack_np.ndim) == 3 else 0
            for si in range(int(pair_count)):
                src_term = str(source_idx_to_term.get(int(idx_np[int(si)]), "") or "")
                dst_idx = int(active_term_to_idx.get(_term_key(src_term), -1))
                if dst_idx < 0 or float(y_active_np[int(dst_idx)]) < 0.5:
                    continue
                mapped_stack_rows.append(np.asarray(stack_np[int(si)], dtype=np.float32))
                mapped_idx_rows.append(int(dst_idx))
        mapped_stack = (
            np.stack(mapped_stack_rows, axis=0).astype(np.float32, copy=False)
            if mapped_stack_rows
            else np.zeros((0, int(height), int(width)), dtype=np.float32)
        )
        mapped_idx = np.asarray(mapped_idx_rows, dtype=np.int64)

        # Only generate special masks for labels not already covered by the
        # wheel's mapped_stack — prevents global ones-fallback from overriding
        # the spatial masks that the wheel already stored for built-in terms.
        y_needs_special = np.zeros_like(y_active_np, dtype=np.float32)
        already_mapped = set(int(i) for i in mapped_idx_rows)
        for _ci in np.where(y_active_np >= 0.5)[0]:
            if int(_ci) not in already_mapped:
                y_needs_special[int(_ci)] = 1.0
        special_stack, special_idx = build_label_mask_stack(
            mixed_mask=base_mask_np,
            label_vec=y_needs_special,
            idx_to_term=active_idx_to_term,
            treat_mixed_mask_as_creation=True,
        )
        covered = set(np.asarray(mapped_idx, dtype=np.int64).reshape(-1).tolist())
        covered.update(np.asarray(special_idx, dtype=np.int64).reshape(-1).tolist())
        missing = [
            int(idx)
            for idx in np.where(np.asarray(y_active_np, dtype=np.float32) >= 0.5)[0].astype(np.int64).tolist()
            if int(idx) not in covered
        ]
        if missing:
            missing_y = np.zeros_like(y_active_np, dtype=np.float32)
            missing_y[np.asarray(missing, dtype=np.int64)] = 1.0
            creation_stack, creation_idx = build_creation_label_mask_stack(
                missing_y,
                height=int(height),
                width=int(width),
                creation_mask=base_mask_np,
            )
        else:
            creation_stack = np.zeros((0, int(height), int(width)), dtype=np.float32)
            creation_idx = np.zeros((0,), dtype=np.int64)
        remapped_stack, remapped_idx = combine_label_mask_stacks(
            y_active_np,
            (mapped_stack, mapped_idx),
            (special_stack, special_idx),
            (creation_stack, creation_idx),
            height=int(height),
            width=int(width),
            strict=True,
            idx_to_term=active_idx_to_term,
        )
        _nds_keep = [i for i in range(min(int(remapped_stack.shape[0]), int(remapped_idx.size))) if not _is_dataset_label_term(str((active_idx_to_term or {}).get(int(remapped_idx[i]), "")))]
        mixed_mask_np = _composite_mask_stack(remapped_stack[np.asarray(_nds_keep, dtype=np.int64)]) if _nds_keep else np.asarray(base_mask_np, dtype=np.float32)
        out_y_rows.append(torch.from_numpy(np.asarray(y_active_np, dtype=np.float32)))
        out_m_rows.append(torch.from_numpy(np.asarray(mixed_mask_np, dtype=np.float32)).unsqueeze(0))
        out_stack_rows.append(torch.from_numpy(np.asarray(remapped_stack, dtype=np.float32)))
        out_index_rows.append(torch.from_numpy(np.asarray(remapped_idx, dtype=np.int64)))

    meta["terms_rows"] = [list(row) for row in list(terms_rows)]
    meta["mask_stacks"] = out_stack_rows
    meta["mask_indices"] = out_index_rows
    y_out = torch.stack(out_y_rows, dim=0).to(device=yb.device, dtype=torch.float32)
    m_out = torch.stack(out_m_rows, dim=0).to(device=mb.device, dtype=torch.float32)
    return y_out, m_out, meta


def _run_classifier_refresh_epochs(
    classifier: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    grad_scaler: Any,
    stage_label: str,
    args: Any,
    max_steps: int = 0,
    min_steps: int = 0,
    lr_sine_cycles: float = 1.0,
    lr_sine_frequency: float = 0.0,
    lr_sine_tail_fraction: float = 0.15,
    lr_sine_min_scale: float = 0.0,
    amp_enabled: bool = False,
    amp_dtype: str = "float16",
    channels_last: bool = False,
    grad_accum_steps: int = 1,
    log_every: int = 0,
    max_seconds: float = 0.0,
    cache_x: Optional[torch.Tensor] = None,
    cache_y: Optional[torch.Tensor] = None,
    cache_m: Optional[torch.Tensor] = None,
    cache_batch_size: int = 0,
    active_classes: int = 0,
    semantic_mask_supervision_mode: str = "multihot_mix",
    vram_fraction: float = 0.35,
    activation_multiplier: float = 18.0,
    max_forward_batch_cap: int = 0,
    target_label_knockout_prob: float = 0.0,
    target_label_knockout_min_keep: int = 1,
    target_label_knockout_max_drop_frac: float = 0.5,
    target_label_knockout_seed: int = 0,
    step_preview_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    stop_requested: Optional[Callable[[], bool]] = None,
    weight_update_callback: Optional[Callable[[int], None]] = None,
    grad_clip: float = 1.0,
    semantic_soft_target_max: float = 0.0,
    semantic_cosine_weight: float = CLASSIFIER_SEMANTIC_COSINE_WEIGHT,
    remap_targets_from_terms: bool = False,
    active_class_names: Optional[Sequence[str]] = None,
    source_class_names: Optional[Sequence[str]] = None,
    pause_requested: Optional[Callable[[], bool]] = None,
    ipc_pump: Optional[Callable[[], None]] = None,
):
    if epochs <= 0:
        return {"ran": False, "loss": 0.0}

    # A prior transformer stage may have frozen this classifier for feature scoring.
    # Ensure refresh has trainable params before building autograd graph.
    if not any(bool(p.requires_grad) for p in classifier.parameters()):
        for p in classifier.parameters():
            p.requires_grad_(True)

    classifier.train()
    runtime_channels_last = bool(channels_last)
    if runtime_channels_last:
        classifier = classifier.to(memory_format=torch.channels_last)
    restore_cudnn_benchmark = None
    if device.type == "cuda" and hasattr(torch.backends, "cudnn"):
        try:
            restore_cudnn_benchmark = bool(torch.backends.cudnn.benchmark)
            if bool(restore_cudnn_benchmark):
                torch.backends.cudnn.benchmark = False
                _log("[berkeley-refresh] disabled cudnn benchmark for refresh stage to avoid one-sample workspace spikes")
        except Exception:
            restore_cudnn_benchmark = None
    amp_dtype_t = resolve_amp_dtype(amp_dtype) if amp_enabled else torch.float16
    use_scaler = bool(amp_enabled and device.type == "cuda" and amp_dtype_t == torch.float16)
    scaler = grad_scaler
    grad_accum_steps = max(1, int(grad_accum_steps))
    mode_key = re.sub(r"\s+", "_", str(semantic_mask_supervision_mode)).strip().lower()
    use_cache = (
        (cache_x is not None)
        and (cache_y is not None)
        and (cache_m is not None)
        and int(getattr(cache_x, "shape", [0])[0]) > 0
        and int(getattr(cache_y, "shape", [0])[0]) == int(getattr(cache_x, "shape", [0])[0])
        and int(getattr(cache_m, "shape", [0])[0]) == int(getattr(cache_x, "shape", [0])[0])
        and int(cache_batch_size) > 0
        and mode_key in ("", "multihot_mix", "multihot", "mixed")
        and bool(_refresh_cache_is_staging_safe(cache_x=cache_x, cache_y=cache_y, cache_m=cache_m, model_device=device))
    )
    cache_batch_size = max(1, int(cache_batch_size)) if use_cache else 0
    refresh_source = "cache" if use_cache else "loader"
    if bool(use_cache) and cache_x is not None and str(cache_x.device) != str(device):
        refresh_source = f"cache-staging:{str(cache_x.device)}->{str(device)}"
    knockout_prob = float(max(0.0, min(1.0, float(target_label_knockout_prob))))
    knockout_enabled = bool(knockout_prob > 0.0)
    knockout_rng = np.random.default_rng(max(0, int(target_label_knockout_seed)))
    knockout_rows_applied = 0
    knockout_labels_dropped = 0
    opt = optimizer
    min_steps = max(0, int(min_steps))
    if use_cache:
        cache_n = int(cache_x.shape[0])
        full_steps = max(1, int(math.ceil(float(cache_n) / float(cache_batch_size))))
        steps_per_epoch = int(full_steps if int(max_steps) <= 0 else max(1, int(max_steps)))
        steps_per_epoch = max(int(steps_per_epoch), int(min_steps))
    else:
        full_steps = max(1, int(len(loader)))
        steps_per_epoch = int(full_steps if int(max_steps) <= 0 else max(1, int(max_steps)))
        steps_per_epoch = max(int(steps_per_epoch), int(min_steps))
    updates_per_epoch = int(math.ceil(float(steps_per_epoch) / float(grad_accum_steps)))
    total_loss = 0.0
    n = 0
    t_start = time.time()
    if device.type == "cuda":
        _log(f"[berkeley-refresh] cuda preflight: {_cuda_mem_diag(device)}")
    total_target_steps = max(1, int(epochs) * steps_per_epoch)
    global_step = 0
    stop_now = False
    optimizer_step_count = 0
    logged_refresh_microbatch_cap = -1
    try:
        for _ in range(int(epochs)):
            step = 0
            opt.zero_grad(set_to_none=True)
            if use_cache:
                cache_n = int(cache_x.shape[0])
                cache_perm = torch.randperm(cache_n, device=cache_x.device)
                cache_pos = 0
            else:
                loader_iter = iter(loader)

            while step < steps_per_epoch:
                if stop_requested is not None:
                    try:
                        if bool(stop_requested()):
                            stop_now = True
                            break
                    except Exception:
                        pass
                # Batch-level pause: spin here so the GUI can pause/resume
                # between any two batches (and during dataloader pre-fetch waits).
                if pause_requested is not None:
                    try:
                        while bool(pause_requested()):
                            if stop_requested is not None:
                                try:
                                    if bool(stop_requested()):
                                        stop_now = True
                                        break
                                except Exception:
                                    pass
                            if stop_now:
                                break
                            if ipc_pump is not None:
                                try:
                                    ipc_pump()
                                except Exception:
                                    pass
                            time.sleep(0.05)
                    except Exception:
                        pass
                    if stop_now:
                        break
                if use_cache:
                    idx_parts: List[torch.Tensor] = []
                    need = int(cache_batch_size)
                    while need > 0:
                        if cache_pos >= cache_n:
                            cache_perm = torch.randperm(cache_n, device=cache_x.device)
                            cache_pos = 0
                        take = min(need, cache_n - cache_pos)
                        idx_parts.append(cache_perm[cache_pos : cache_pos + take])
                        cache_pos += take
                        need -= take
                    idx = idx_parts[0] if len(idx_parts) == 1 else torch.cat(idx_parts, dim=0)
                    xb = None
                    yb = None
                    mb = None
                    batch_meta = None
                else:
                    try:
                        xb, yb, mb, batch_meta = _unpack_masked_semantic_batch(next(loader_iter), context="berkeley refresh training")
                    except StopIteration:
                        loader_iter = iter(loader)
                        xb, yb, mb, batch_meta = _unpack_masked_semantic_batch(next(loader_iter), context="berkeley refresh training")
                if bool(remap_targets_from_terms) and isinstance(batch_meta, dict):
                    yb, mb, batch_meta = _remap_semantic_batch_to_active_vocab(
                        yb,
                        mb,
                        batch_meta,
                        active_class_names=list(active_class_names or []),
                        source_class_names=list(source_class_names or []),
                    )
                if not bool(use_cache):
                    xb, yb, mb = _expand_semantic_mask_supervision_batch(
                        xb=xb,
                        yb=yb,
                        mb=mb,
                        batch_meta=batch_meta,
                        mode=str(semantic_mask_supervision_mode),
                        context="berkeley refresh training",
                    )
                total_batch_n = int(idx.numel()) if use_cache else max(1, int(xb.shape[0]))
                slice_cap = _auto_berkeley_refresh_batch_size(
                    cache_x=(cache_x if use_cache else xb),
                    cache_y=(cache_y if use_cache else yb),
                    cache_m=(cache_m if use_cache else mb),
                    device=device,
                    vram_fraction=float(vram_fraction),
                    activation_multiplier=float(activation_multiplier),
                    max_cap=(int(max_forward_batch_cap) if int(max_forward_batch_cap) > 0 else int(total_batch_n)),
                )
                slice_cap = max(1, min(int(slice_cap), int(total_batch_n)))
                if int(slice_cap) != int(logged_refresh_microbatch_cap):
                    _log(
                        "[berkeley-refresh] microbatch cap: "
                        f"batch={int(total_batch_n)} cap={int(slice_cap)} source={str(refresh_source)}"
                    )
                    logged_refresh_microbatch_cap = int(slice_cap)
                step_loss_value = 0.0
                step_seen = 0
                while True:
                    try:
                        step_loss_value = 0.0
                        step_seen = 0
                        preview_items: list = []
                        for st in range(0, int(total_batch_n), int(slice_cap)):
                            ed = min(int(total_batch_n), int(st + slice_cap))
                            if use_cache:
                                idx_part = idx[st:ed]
                                with torch.no_grad():
                                    xb_part = cache_x.index_select(0, idx_part).detach()
                                    yb_part = cache_y.index_select(0, idx_part).detach()
                                    mb_part = cache_m.index_select(0, idx_part).detach()
                            else:
                                xb_part = xb[st:ed]
                                yb_part = yb[st:ed]
                                mb_part = mb[st:ed]
                            if xb_part.device != device:
                                if bool(runtime_channels_last):
                                    xb_part = xb_part.to(device=device, non_blocking=True, memory_format=torch.channels_last)
                                else:
                                    xb_part = xb_part.to(device, non_blocking=True)
                            if yb_part.device != device:
                                yb_part = yb_part.to(device, non_blocking=True)
                            if mb_part.device != device:
                                mb_part = mb_part.to(device, non_blocking=True)
                            if bool(knockout_enabled):
                                yb_part, rows_drop, labels_drop = _label_knockout_tensor_batch(
                                    yb=yb_part,
                                    rng=knockout_rng,
                                    prob=float(knockout_prob),
                                    min_keep=int(target_label_knockout_min_keep),
                                    max_drop_frac=float(target_label_knockout_max_drop_frac),
                                    threshold=0.5,
                                )
                                knockout_rows_applied += int(rows_drop)
                                knockout_labels_dropped += int(labels_drop)
                            if bool(runtime_channels_last) and xb_part.device == device:
                                xb_part = xb_part.contiguous(memory_format=torch.channels_last)
                            with autocast_context(device=device, enabled=amp_enabled, amp_dtype=amp_dtype_t):
                                out = _forward_classifier_outputs_require_mask(
                                    classifier=classifier,
                                    xb=xb_part,
                                    context="berkeley refresh training",
                                )
                                logits = out["logits"]
                            supervised_dim = int(yb_part.shape[1]) if int(yb_part.ndim) == 2 else int(logits.shape[1])
                            if int(active_classes) > 0:
                                supervised_dim = min(int(supervised_dim), int(active_classes))
                            supervised_dim = max(1, int(supervised_dim))
                            loss, _, _, _ = _classifier_supervision_loss(
                                classifier=classifier,
                                logits=logits,
                                y_multihot=yb_part,
                                supervised_dim=int(supervised_dim),
                                semantic_cosine_weight=float(semantic_cosine_weight),
                                semantic_soft_target_max=float(semantic_soft_target_max),
                            )
                            mask_loss, _ = _semantic_mask_bce_loss(
                                mask_logits=out["mask_logits"],
                                mask_targets=mb_part,
                                context="berkeley refresh training",
                            )
                            loss = (loss * float(CLASSIFIER_LOSS_SCALE)) + mask_loss
                            part_weight = float(ed - st) / float(max(1, int(total_batch_n)))
                            loss_to_backprop = (loss * float(part_weight)) / float(grad_accum_steps)
                            if use_scaler:
                                scaler.scale(loss_to_backprop).backward()
                            else:
                                loss_to_backprop.backward()
                            step_loss_value += float(loss.detach().item()) * float(ed - st)
                            step_seen += int(ed - st)
                            if step_preview_callback is not None and int(xb_part.shape[0]) > 0:
                                _probs_sup = torch.sigmoid(logits[:, : int(supervised_dim)].detach()).to(torch.float32)
                                _bl = float(loss.detach().to(torch.float32).item())
                                for _i in range(int(xb_part.shape[0])):
                                    preview_items.append({
                                        "img": xb_part[_i].detach().to(device="cpu", dtype=torch.float32),
                                        "probs": _probs_sup[_i].detach().to(device="cpu", dtype=torch.float32),
                                        "target_vec": yb_part[_i, : int(supervised_dim)].detach().to(device="cpu", dtype=torch.float32),
                                        "target_mask": mb_part[_i].detach().to(device="cpu", dtype=torch.float32),
                                        "detected_mask": torch.sigmoid(out["mask_logits"][_i].detach()).to(device="cpu", dtype=torch.float32),
                                        "batch_loss": _bl,
                                    })
                                del _probs_sup
                            del xb_part, yb_part, mb_part, out, logits, loss, mask_loss
                        break
                    except torch.OutOfMemoryError:
                        if device.type != "cuda":
                            raise
                        _log(
                            "[berkeley-refresh] cuda oom state: "
                            f"slice_cap={int(slice_cap)} batch={int(total_batch_n)} {_cuda_mem_diag(device)}"
                        )
                        opt.zero_grad(set_to_none=True)
                        gc.collect()
                        try:
                            torch.cuda.synchronize(device)
                        except Exception:
                            pass
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        try:
                            torch.cuda.ipc_collect()
                        except Exception:
                            pass
                        if bool(runtime_channels_last):
                            runtime_channels_last = False
                            classifier = classifier.to(memory_format=torch.contiguous_format)
                            _log("[berkeley-refresh] cuda oom; retrying with contiguous tensors")
                            continue
                        if int(slice_cap) <= 1:
                            raise
                        next_slice_cap = max(1, int(slice_cap // 2))
                        if int(next_slice_cap) > 1:
                            next_slice_cap = 1 << (int(next_slice_cap).bit_length() - 1)
                        slice_cap = int(next_slice_cap)
                        _log(
                            "[berkeley-refresh] cuda oom; retrying with smaller microbatch cap "
                            f"{int(slice_cap)}"
                        )
                do_step = ((step + 1) % grad_accum_steps == 0) or ((step + 1) == steps_per_epoch)
                if do_step:
                    if use_scaler:
                        scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(classifier.parameters(), float(grad_clip))
                    if use_scaler:
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()
                    optimizer_step_count += 1
                    if weight_update_callback is not None:
                        try:
                            weight_update_callback(int(optimizer_step_count))
                        except Exception:
                            pass
                    opt.zero_grad(set_to_none=True)
                total_loss += float(step_loss_value)
                n += int(step_seen)
                step += 1
                global_step += 1
                if int(log_every) > 0 and (global_step % int(log_every) == 0):
                    elapsed = max(1e-6, time.time() - t_start)
                    ips = float(n) / elapsed
                    _cur_loss = total_loss / max(1, n)
                    print(
                        f"[berkeley-refresh] step={global_step}/{total_target_steps} "
                        f"loss={_cur_loss:.4f} samples={n} samp_per_sec={ips:.1f}",
                        flush=True,
                    )
                    if progress_callback is not None:
                        try:
                            progress_callback({
                                "global_step": int(global_step),
                                "total_steps": int(total_target_steps),
                                "loss": float(_cur_loss),
                                "samples_per_sec": float(ips),
                            })
                        except (StageSkipForward, StageSkipBack, StageStopRequested):
                            raise
                        except Exception:
                            pass
                if step_preview_callback is not None and len(preview_items) > 0:
                    _avg_loss = float(total_loss / max(1, n))
                    _cb_batch = []
                    for _item in preview_items:
                        _cb_batch.append(
                            {
                                "global_step": int(global_step),
                                "total_steps": int(total_target_steps),
                                "img": _item["img"],
                                "probs": _item["probs"],
                                "target_vec": _item["target_vec"],
                                "target_mask": _item["target_mask"],
                                "detected_mask": _item["detected_mask"],
                                "loss": _avg_loss,
                                "batch_loss": float(_item["batch_loss"]),
                            }
                        )
                    try:
                        step_preview_callback(_cb_batch)
                    except Exception as e:
                        _log(f"[stage-opengl] C-step callback failed: {e}")
                if xb is not None:
                    del xb, yb, mb
                batch_meta = None  # drop mask_stack refs so pinned CUDA memory can be reclaimed
                if use_cache:
                    del idx
                if float(max_seconds) > 0.0 and (time.time() - t_start) >= float(max_seconds):
                    classifier.eval()
                    return {
                        "ran": True,
                        "loss": total_loss / max(1, n),
                        "samples": n,
                        "steps_per_epoch": int(steps_per_epoch),
                        "truncated_by_time": True,
                        "stopped_early": False,
                        "elapsed_sec": float(time.time() - t_start),
                        "source": refresh_source,
                        "target_knockout_rows_applied": int(knockout_rows_applied),
                        "target_knockout_labels_dropped": int(knockout_labels_dropped),
                    }
            if stop_now:
                break
    finally:
        if restore_cudnn_benchmark is not None and hasattr(torch.backends, "cudnn"):
            try:
                torch.backends.cudnn.benchmark = bool(restore_cudnn_benchmark)
            except Exception:
                pass
    classifier.eval()
    return {
        "ran": bool(n > 0),
        "loss": total_loss / max(1, n),
        "samples": n,
        "steps_per_epoch": int(steps_per_epoch),
        "truncated_by_time": False,
        "stopped_early": bool(stop_now),
        "elapsed_sec": float(time.time() - t_start),
        "source": refresh_source,
        "target_knockout_rows_applied": int(knockout_rows_applied),
        "target_knockout_labels_dropped": int(knockout_labels_dropped),
    }


def _run_fake_class_refresh_epochs(
    classifier: nn.Module,
    generator: nn.Module,
    discriminator: Optional[nn.Module],
    payload_conditions: Sequence[Any],
    condition_num_classes: int,
    fake_label_vector: Optional[torch.Tensor],
    z_dim: int,
    device: torch.device,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    lr_sine_cycles: float,
    lr_sine_frequency: float,
    lr_sine_tail_fraction: float,
    lr_sine_min_scale: float,
    amp_enabled: bool = False,
    amp_dtype: str = "float16",
    channels_last: bool = False,
    grad_accum_steps: int = 1,
    log_every: int = 0,
    include_condition_targets: bool = True,
    fake_vector_weight: float = 1.0,
    condition_target_weight: float = 0.35,
    disc_conf_temperature: float = 1.0,
    disc_conf_floor: float = 0.25,
    balance_disc_groups: bool = True,
    step_preview_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    stop_requested: Optional[Callable[[], bool]] = None,
    weight_update_callback: Optional[Callable[[int], None]] = None,
    seed: int = 0,
    grad_clip: float = 1.0,
):
    if epochs <= 0 or steps_per_epoch <= 0 or batch_size <= 0:
        return {"ran": False, "loss": 0.0, "samples": 0, "source": "fake_vector"}
    if generator is None or len(payload_conditions) <= 0:
        return {"ran": False, "loss": 0.0, "samples": 0, "source": "fake_vector"}
    if int(condition_num_classes) <= 0:
        return {"ran": False, "loss": 0.0, "samples": 0, "source": "fake_vector"}
    if fake_label_vector is None:
        return {"ran": False, "loss": 0.0, "samples": 0, "source": "fake_vector", "reason": "missing_fake_vector"}
    from wav_ml_models import TinyConvClassifier
    if not isinstance(classifier, TinyConvClassifier):
        return {
            "ran": False,
            "loss": 0.0,
            "samples": 0,
            "source": "fake_vector",
            "reason": f"unsupported_classifier:{type(classifier).__name__}",
        }

    if not any(bool(p.requires_grad) for p in classifier.parameters()):
        for p in classifier.parameters():
            p.requires_grad_(True)

    classifier.train()
    if channels_last:
        classifier = classifier.to(memory_format=torch.channels_last)
    generator_was_training = bool(generator.training)
    generator.eval()
    discriminator_was_training = False
    if discriminator is not None:
        discriminator_was_training = bool(discriminator.training)
        discriminator.eval()

    fake_vec = fake_label_vector.detach().to(device=device, dtype=torch.float32).reshape(-1)
    if int(fake_vec.numel()) <= 0:
        classifier.eval()
        generator.train(generator_was_training)
        if discriminator is not None:
            discriminator.train(discriminator_was_training)
        return {"ran": False, "loss": 0.0, "samples": 0, "source": "fake_vector", "reason": "empty_fake_vector"}
    fake_vec = F.normalize(fake_vec.unsqueeze(0), dim=1, eps=1e-6).squeeze(0)
    if int(classifier.embed_proj.out_features) != int(fake_vec.numel()):
        classifier.eval()
        generator.train(generator_was_training)
        if discriminator is not None:
            discriminator.train(discriminator_was_training)
        return {
            "ran": False,
            "loss": 0.0,
            "samples": 0,
            "source": "fake_vector",
            "reason": (
                f"fake_vector_dim_mismatch: vec={int(fake_vec.numel())} "
                f"embed_proj={int(classifier.embed_proj.out_features)}"
            ),
        }

    amp_dtype_t = resolve_amp_dtype(amp_dtype) if amp_enabled else torch.float16
    use_scaler = bool(amp_enabled and device.type == "cuda" and amp_dtype_t == torch.float16)
    scaler = make_grad_scaler(enabled=use_scaler)
    grad_accum_steps = max(1, int(grad_accum_steps))
    n_steps = max(1, int(steps_per_epoch))
    updates_per_epoch = int(math.ceil(float(n_steps) / float(grad_accum_steps)))
    opt = torch.optim.AdamW(classifier.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    rng = np.random.default_rng(seed)
    cond_bank_cpu = _payload_condition_bank_tensor(
        payload_targets=payload_conditions,
        num_classes=int(condition_num_classes),
    )
    if int(cond_bank_cpu.shape[0]) <= 0:
        classifier.eval()
        generator.train(generator_was_training)
        if discriminator is not None:
            discriminator.train(discriminator_was_training)
        return {"ran": False, "loss": 0.0, "samples": 0, "source": "fake_vector", "reason": "empty_condition_bank"}
    cond_bank = cond_bank_cpu.to(device=device, dtype=torch.float32)

    total_loss = 0.0
    n = 0
    global_step = 0
    fake_cos_sum = 0.0
    fake_cos_pass_sum = 0.0
    fake_cos_fail_sum = 0.0
    disc_conf_sum = 0.0
    disc_conf_pass_sum = 0.0
    disc_conf_fail_sum = 0.0
    cond_target_prob_sum = 0.0
    disc_pass_n = 0
    disc_fail_n = 0
    disc_logit_pass_sum = 0.0
    disc_logit_fail_sum = 0.0
    total_target_steps = int(max(1, int(epochs)) * int(n_steps))
    t_start = time.time()
    stop_now = False
    optimizer_step_count = 0
    opt.zero_grad(set_to_none=True)
    for _ in range(int(max(1, int(epochs)))):
        for step_idx in range(1, int(n_steps) + 1):
            if stop_requested is not None:
                try:
                    if bool(stop_requested()):
                        stop_now = True
                        break
                except Exception:
                    pass

            idx = rng.integers(0, int(cond_bank.shape[0]), size=max(1, int(batch_size)))
            idx_t = torch.as_tensor(idx, device=device, dtype=torch.long)
            cond = cond_bank.index_select(0, idx_t).to(device=device, dtype=torch.float32)
            z = torch.randn((int(cond.shape[0]), max(8, int(z_dim))), device=device)
            with torch.no_grad():
                with autocast_context(device=device, enabled=amp_enabled, amp_dtype=amp_dtype_t):
                    xb = generator(z, cond).to(torch.float32)
                disc_logits = None
                if discriminator is not None:
                    with autocast_context(device=device, enabled=amp_enabled, amp_dtype=amp_dtype_t):
                        disc_logits = discriminator(xb, cond).to(torch.float32)
            if channels_last:
                xb = xb.contiguous(memory_format=torch.channels_last)
            with autocast_context(device=device, enabled=amp_enabled, amp_dtype=amp_dtype_t):
                feat = classifier.extract_features(xb)
                z_norm = classifier.encode_semantic_from_features(feat)
                fake_cos = torch.sum(z_norm * fake_vec.unsqueeze(0).to(device=z_norm.device, dtype=z_norm.dtype), dim=1)
                fake_loss_per = 1.0 - fake_cos

                if bool(int(classifier.label_embed_enabled.item())) and int(classifier.label_embed_bank.shape[0]) > 0:
                    logits = classifier.semantic_logits_from_features(feat=feat)
                else:
                    logits = classifier.head[-1](feat)

                if int(logits.shape[1]) != int(cond.shape[1]):
                    raise RuntimeError(
                        "Condition/logit width mismatch in fake feedback: "
                        f"logits={tuple(logits.shape)} cond={tuple(cond.shape)}"
                    )
                if bool(include_condition_targets):
                    cond_prob = torch.sigmoid(logits.to(dtype=torch.float32))
                    cond_target = cond.to(dtype=torch.float32)
                    cond_loss_per = F.mse_loss(
                        cond_prob,
                        cond_target,
                        reduction="none",
                    ).mean(dim=1)
                else:
                    cond_loss_per = torch.zeros_like(fake_loss_per)

                loss_per_sample = (
                    (float(max(0.0, fake_vector_weight)) * fake_loss_per)
                    + (float(max(0.0, condition_target_weight)) * cond_loss_per)
                )
                if disc_logits is not None:
                    disc_det = disc_logits.detach().to(torch.float32)
                    pass_mask = disc_det >= 0.0
                    fail_mask = ~pass_mask
                    conf_temp = max(1e-4, float(disc_conf_temperature))
                    conf_floor = max(0.0, min(1.0, float(disc_conf_floor)))
                    disc_conf = torch.sigmoid(torch.abs(disc_det) / conf_temp)
                    conf_w = conf_floor + ((1.0 - conf_floor) * disc_conf)
                    if bool(balance_disc_groups):
                        pass_count = torch.clamp(pass_mask.to(torch.float32).sum(), min=1.0)
                        fail_count = torch.clamp(fail_mask.to(torch.float32).sum(), min=1.0)
                        w_pass = 0.5 / pass_count
                        w_fail = 0.5 / fail_count
                        group_w = torch.where(pass_mask, w_pass, w_fail).to(torch.float32)
                    else:
                        group_w = torch.ones_like(conf_w, dtype=torch.float32)
                    sample_w = group_w * conf_w
                    sample_w = sample_w / torch.clamp(sample_w.mean(), min=1e-6)
                    loss = (loss_per_sample.to(torch.float32) * sample_w.to(torch.float32)).mean().to(loss_per_sample.dtype)
                else:
                    pass_mask = None
                    disc_conf = None
                    loss = loss_per_sample.mean()
            loss = loss * float(CLASSIFIER_LOSS_SCALE)
            loss_to_backprop = loss / float(grad_accum_steps)
            if use_scaler:
                scaler.scale(loss_to_backprop).backward()
            else:
                loss_to_backprop.backward()

            do_step = ((int(step_idx) % int(grad_accum_steps)) == 0) or (int(step_idx) == int(n_steps))
            if do_step:
                if use_scaler:
                    scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(classifier.parameters(), float(grad_clip))
                if use_scaler:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                optimizer_step_count += 1
                if weight_update_callback is not None:
                    try:
                        weight_update_callback(int(optimizer_step_count))
                    except Exception:
                        pass
                opt.zero_grad(set_to_none=True)

            total_loss += float(loss.item()) * int(xb.shape[0])
            n += int(xb.shape[0])
            global_step += 1
            fake_cos_f = fake_cos.detach().to(torch.float32)
            fake_cos_sum += float(fake_cos_f.sum().item())
            probs = torch.sigmoid(logits.detach().to(torch.float32))
            if bool(include_condition_targets):
                target_count = torch.clamp(cond.sum(dim=1).to(torch.float32), min=1.0)
                target_prob = ((probs * cond).sum(dim=1) / target_count).mean()
                cond_target_prob_sum += float(target_prob.item()) * int(xb.shape[0])
            if disc_conf is not None:
                disc_conf_f = disc_conf.detach().to(torch.float32)
                disc_conf_sum += float(disc_conf_f.sum().item())
            else:
                disc_conf_f = None
            if pass_mask is not None:
                pass_count = int(pass_mask.to(torch.int64).sum().item())
                fail_count = int(int(xb.shape[0]) - pass_count)
                disc_pass_n += int(pass_count)
                disc_fail_n += int(fail_count)
                if pass_count > 0:
                    fake_cos_pass_sum += float(fake_cos_f[pass_mask].sum().item())
                if fail_count > 0:
                    fake_cos_fail_sum += float(fake_cos_f[~pass_mask].sum().item())
                if disc_conf_f is not None:
                    if pass_count > 0:
                        disc_conf_pass_sum += float(disc_conf_f[pass_mask].sum().item())
                    if fail_count > 0:
                        disc_conf_fail_sum += float(disc_conf_f[~pass_mask].sum().item())
                if disc_logits is not None:
                    if pass_count > 0:
                        disc_logit_pass_sum += float(disc_logits[pass_mask].sum().item())
                    if fail_count > 0:
                        disc_logit_fail_sum += float(disc_logits[~pass_mask].sum().item())

            if int(log_every) > 0 and (int(global_step) % int(log_every) == 0):
                elapsed = max(1e-6, time.time() - t_start)
                ips = float(n) / elapsed
                pass_rate = (float(disc_pass_n) / float(max(1, disc_pass_n + disc_fail_n))) if discriminator is not None else 0.0
                print(
                    f"[berkeley-fake-refresh] step={global_step}/{total_target_steps} "
                    f"loss={total_loss / max(1, n):.4f} samples={n} "
                    f"disc_pass_rate={pass_rate:.3f} fake_cos={fake_cos_sum / max(1, n):.3f} "
                    f"disc_conf={disc_conf_sum / max(1, n):.3f} "
                    f"samp_per_sec={ips:.1f}",
                    flush=True,
                )
                if progress_callback is not None:
                    try:
                        progress_callback(
                            {
                                "global_step": int(global_step),
                                "total_steps": int(total_target_steps),
                                "loss": float(total_loss / max(1, n)),
                                "samples_per_sec": float(ips),
                            }
                        )
                    except (StageSkipForward, StageSkipBack, StageStopRequested):
                        raise
                    except Exception:
                        pass
            if step_preview_callback is not None and int(xb.shape[0]) > 0:
                _avg_loss = float(total_loss / max(1, n))
                _batch_loss = float(loss.detach().to(torch.float32).item())
                _cb_batch = []
                for _i in range(int(xb.shape[0])):
                    try:
                        _cb_batch.append(
                            {
                                "global_step": int(global_step),
                                "total_steps": int(total_target_steps),
                                "img": xb[_i].detach().to(torch.float32).cpu(),
                                "probs": torch.sigmoid(logits[_i].detach()).to(torch.float32).cpu(),
                                "target_vec": cond[_i].detach().to(torch.float32).cpu(),
                                "loss": _avg_loss,
                                "batch_loss": _batch_loss,
                                "fake_cos": float(fake_cos_f[_i].detach().item()),
                                "disc_conf": (
                                    float(disc_conf_f[_i].detach().item())
                                    if disc_conf_f is not None
                                    else 0.0
                                ),
                                "disc_logit": (
                                    float(disc_logits[_i].detach().to(torch.float32).item())
                                    if disc_logits is not None
                                    else 0.0
                                ),
                                "disc_pass": (
                                    bool(pass_mask[_i].detach().item())
                                    if pass_mask is not None
                                    else False
                                ),
                            }
                        )
                    except Exception as e:
                        _log(f"[stage-opengl] fake-refresh callback item failed: {e}")
                try:
                    step_preview_callback(_cb_batch)
                except Exception as e:
                    _log(f"[stage-opengl] fake-refresh callback failed: {e}")
        if stop_now:
            break

    classifier.eval()
    generator.train(generator_was_training)
    if discriminator is not None:
        discriminator.train(discriminator_was_training)
    fake_cos_mean = (float(fake_cos_sum) / float(max(1, n))) if n > 0 else 0.0
    pass_total = max(1, disc_pass_n + disc_fail_n)
    disc_pass_rate = float(disc_pass_n) / float(pass_total)
    disc_fail_rate = float(disc_fail_n) / float(pass_total)
    fake_cos_pass = float(fake_cos_pass_sum) / float(max(1, disc_pass_n))
    fake_cos_fail = float(fake_cos_fail_sum) / float(max(1, disc_fail_n))
    disc_conf_mean = float(disc_conf_sum) / float(max(1, n))
    disc_conf_pass = float(disc_conf_pass_sum) / float(max(1, disc_pass_n))
    disc_conf_fail = float(disc_conf_fail_sum) / float(max(1, disc_fail_n))
    disc_logit_pass = float(disc_logit_pass_sum) / float(max(1, disc_pass_n))
    disc_logit_fail = float(disc_logit_fail_sum) / float(max(1, disc_fail_n))
    cond_target_prob_mean = float(cond_target_prob_sum) / float(max(1, n))
    return {
        "ran": bool(n > 0),
        "loss": total_loss / max(1, n),
        "samples": int(n),
        "stopped_early": bool(stop_now),
        "source": ("fake_vector_disc_feedback" if discriminator is not None else "fake_vector"),
        "elapsed_sec": float(time.time() - t_start),
        "fake_label_dim": int(fake_vec.numel()),
        "fake_cos_mean": float(fake_cos_mean),
        "fake_cos_pass": float(fake_cos_pass),
        "fake_cos_fail": float(fake_cos_fail),
        "disc_conf_mean": float(disc_conf_mean),
        "disc_conf_pass": float(disc_conf_pass),
        "disc_conf_fail": float(disc_conf_fail),
        "condition_target_prob_mean": float(cond_target_prob_mean),
        "disc_pass_samples": int(disc_pass_n),
        "disc_fail_samples": int(disc_fail_n),
        "disc_pass_rate": float(disc_pass_rate),
        "disc_fail_rate": float(disc_fail_rate),
        "disc_logit_pass": float(disc_logit_pass),
        "disc_logit_fail": float(disc_logit_fail),
    }


def _infer_condition_vector_width(payload_conditions: Sequence[Any], class_names: Sequence[str]) -> int:
    for row in payload_conditions:
        try:
            arr = np.asarray(row, dtype=np.float32).reshape(-1)
        except Exception:
            continue
        if int(arr.size) > 0:
            return int(arr.size)
    return max(1, int(len(class_names)))


def _resolve_fake_feedback_label_vector(ctx: PipelineContext) -> Optional[torch.Tensor]:
    cached_text = str(getattr(ctx, "fake_feedback_label_text", "") or "")
    cached_vec = getattr(ctx, "fake_feedback_label_vector", None)
    sentinel_text = str(getattr(ctx.args, "fake_image_sentinel_label", "GAN image") or "GAN image").strip() or "GAN image"
    if isinstance(cached_vec, torch.Tensor) and cached_text == sentinel_text and int(cached_vec.numel()) > 0:
        return cached_vec.detach().to(dtype=torch.float32, device="cpu")

    backend = str(getattr(ctx.args, "label_embedding_backend", "") or "").strip().lower()
    if backend != "sentence_transformers":
        _log(f"[fake-class] unsupported label embedding backend for fake sentinel: {backend!r}")
        return None

    model_name = str(getattr(ctx.args, "label_embedding_model", "") or "").strip()
    if not model_name:
        _log("[fake-class] missing label embedding model for fake sentinel encoding")
        return None

    try:
        from pipeline.nodes.label_embedding_node import _encode_texts_sentence_transformers

        vec_np = _encode_texts_sentence_transformers(
            [sentinel_text],
            model_name=model_name,
            device=ctx.device,
        )
    except Exception as exc:
        _log(f"[fake-class] could not encode fake sentinel label {sentinel_text!r}: {exc}")
        return None

    if not isinstance(vec_np, np.ndarray) or int(vec_np.ndim) != 2 or int(vec_np.shape[0]) <= 0 or int(vec_np.shape[1]) <= 0:
        _log(f"[fake-class] invalid fake sentinel embedding shape: {getattr(vec_np, 'shape', None)}")
        return None

    vec_t = torch.from_numpy(np.asarray(vec_np[0], dtype=np.float32)).detach().to(device="cpu")
    setattr(ctx, "fake_feedback_label_text", sentinel_text)
    setattr(ctx, "fake_feedback_label_vector", vec_t)
    return vec_t
