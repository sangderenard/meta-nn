"""
Berkeley Classifier Node — TinyConvClassifier training across all stages.

This file is the authoritative description of everything idiosyncratic to the
semantic/Berkeley classifier in this pipeline:

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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from pipeline.context import PipelineContext
from pipeline.graph import PipelineNode
from pipeline.nodes.base import (
    GatedNode,
    autocast_context,
    freeze,
    make_grad_scaler,
    module_device,
    resolve_amp_dtype,
    unwrap_compiled,
)


# ---------------------------------------------------------------------------
# Node config — all idiosyncrasies in one place
# ---------------------------------------------------------------------------

@dataclass
class BerkeleyClassifierConfig:
    """Every hyperparameter specific to the TinyConvClassifier."""

    # ---- architecture ---------------------------------------------------
    base_ch: int = 64
    max_ch: int = 384
    context_blocks: int = 8
    context_dropout: float = 0.05
    mask_decoder_channels: int = 0       # 0 = disabled

    # ---- label embedding ------------------------------------------------
    label_embedding_backend: str = "sentence_transformers"
    label_embedding_model: str = "all-MiniLM-L6-v2"
    label_embedding_dim: int = 0         # 0 = model native dim
    label_embedding_temperature: float = 10.0
    semantic_cosine_weight: float = 0.35

    # ---- LoRA -----------------------------------------------------------
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_enabled: bool = False
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
    stage0_loss_target: float = 0.35
    stage0_required_consecutive: int = 2

    # ---- Stage 1 — gestation (bootstrap primitives) ----------------------
    stage1_epochs: int = 5
    stage1_batch_size: int = 32
    stage1_loss_target: float = 0.30
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
    stageC_batch_size: int = 16

    # ---- Fake-class feedback -------------------------------------------
    fake_class_enabled: bool = True
    fake_class_steps: int = 32
    fake_class_batch_size: int = 16
    fake_class_disc_weight: float = 1.0   # discriminator-confidence weighting

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

    def __init__(self, cfg: BerkeleyClassifierConfig) -> None:
        self.cfg = cfg
        self._built = False

    def should_run(self, ctx: PipelineContext) -> bool:
        return not self._built

    def execute(self, ctx: PipelineContext) -> None:
        from wav_ml_models import TinyConvClassifier, maybe_compile_module, SinusoidalLRController, SinusoidalLROptions

        n_classes = len(ctx.class_names) if ctx.class_names else 1
        model = TinyConvClassifier(
            num_classes=n_classes,
            base_ch=self.cfg.base_ch,
            max_ch=self.cfg.max_ch,
            context_blocks=self.cfg.context_blocks,
            context_dropout=self.cfg.context_dropout,
            mask_decoder_channels=self.cfg.mask_decoder_channels,
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

        total_rounds = ctx.orchestration_cycles * ctx.orchestration_rounds
        lr_ctrl = SinusoidalLRController(
            optimizer=optimizer,
            total_steps=max(1, total_rounds),
            options=SinusoidalLROptions(
                cycles=self.cfg.lr_cycles,
                tail_fraction=self.cfg.lr_tail_fraction,
                min_scale=self.cfg.lr_min_scale,
            ),
        )

        ctx.classifier = model
        ctx.classifier_optimizer = optimizer
        ctx.classifier_lr_controller = lr_ctrl

        if self.cfg.amp:
            ctx.classifier_grad_scaler = make_grad_scaler(enabled=True)

        # Load checkpoint if available
        ckpt_path = self.cfg.classifier_init_ckpt or getattr(ctx.args, "classifier_init", "") or ""
        ckpt_scope = self.cfg.classifier_init_scope or "all"
        if str(ckpt_path).strip():
            _load_classifier_checkpoint(model, str(ckpt_path), scope=str(ckpt_scope))

        # Build frozen gate replica on CPU
        ctx.gate_classifier = _build_gate_replica(model, ctx)

        _log(f"[classifier] built: n_classes={n_classes} "
             f"base_ch={self.cfg.base_ch} max_ch={self.cfg.max_ch} "
             f"context_blocks={self.cfg.context_blocks}")
        self._built = True


# ---------------------------------------------------------------------------
# Stage 0 — Pre-gestation training node
# ---------------------------------------------------------------------------

class PregestationTrainNode(PipelineNode):
    """Stage 0: Train classifier on synthetic geometric direction+colour logic images.

    This is the very first stage; the classifier learns basic colour/direction/shape
    semantics from programmatically generated images before seeing any real data.
    Gate 0 passes when loss drops below stage0_loss_target for the required
    consecutive rounds.
    """

    node_id = "stage_0_pregestation"
    description = "Stage 0: pre-gestation classifier training (geometric logic)"

    def __init__(self, cfg: BerkeleyClassifierConfig) -> None:
        self.cfg = cfg

    def should_run(self, ctx: PipelineContext) -> bool:
        # Always attempt until gate passes; harmless to run again after passing.
        return ctx.pregestation_loader is not None and ctx.classifier is not None

    def execute(self, ctx: PipelineContext) -> None:
        from wav_config_transformer_pipeline import _run_berkeley_refresh_epochs

        result = _run_berkeley_refresh_epochs(
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
            args=ctx.args,
        )

        loss = float(result.get("loss", float("inf")))
        ctx.gate_pregestation.required_consecutive = self.cfg.stage0_required_consecutive
        ctx.gate_pregestation.record(
            round_id=ctx.round_id,
            metric=loss,
            threshold=self.cfg.stage0_loss_target,
            above=False,
        )
        ctx.log_metric("stage0", "loss", loss)
        _log(f"[stage0] loss={loss:.4f} gate={'PASS' if ctx.gate_pregestation.passed else 'hold'}")


# ---------------------------------------------------------------------------
# Stage 1 — Gestation training node
# ---------------------------------------------------------------------------

class GestationTrainNode(GatedNode):
    """Stage 1: Train classifier on bootstrap primitive symbols.

    Requires Gate 0 to have passed.  Teaches the classifier to recognise
    noise profile terms, basic colours, directions, and geometric structures
    before exposure to Berkeley SBD.

    Gate 1 passes when loss drops below stage1_loss_target.
    """

    node_id = "stage_1_gestation"
    description = "Stage 1: gestation classifier training (bootstrap primitives)"
    required_gates = ["gate_pregestation"]

    def __init__(self, cfg: BerkeleyClassifierConfig) -> None:
        self.cfg = cfg

    def execute(self, ctx: PipelineContext) -> None:
        from wav_config_transformer_pipeline import _run_berkeley_refresh_epochs

        result = _run_berkeley_refresh_epochs(
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
            args=ctx.args,
        )

        loss = float(result.get("loss", float("inf")))
        ctx.gate_gestation.required_consecutive = self.cfg.stage1_required_consecutive
        ctx.gate_gestation.record(
            round_id=ctx.round_id,
            metric=loss,
            threshold=self.cfg.stage1_loss_target,
            above=False,
        )
        ctx.log_metric("stage1", "loss", loss)
        _log(f"[stage1] loss={loss:.4f} gate={'PASS' if ctx.gate_gestation.passed else 'hold'}")


# ---------------------------------------------------------------------------
# Stage 2 — Berkeley SBD refresh training node
# ---------------------------------------------------------------------------

class BerkeleyRefreshTrainNode(GatedNode):
    """Stage 2: Full classifier refresh on Berkeley SBD + external payload images.

    Requires Gates 0+1.  Runs every N rounds (controlled by orchestrator edge
    condition).  This is the main real-image supervision stage.

    Gate 2 passes when confidence AND macro-F1 exceed their targets AND
    optionally loss drops below stage2_loss_target.
    """

    node_id = "stage_2_berkeley"
    description = "Stage 2: Berkeley SBD refresh (full multi-label classification)"
    required_gates = ["gate_pregestation", "gate_gestation"]

    def __init__(self, cfg: BerkeleyClassifierConfig) -> None:
        self.cfg = cfg

    def execute(self, ctx: PipelineContext) -> None:
        from wav_config_transformer_pipeline import _run_berkeley_refresh_epochs

        result = _run_berkeley_refresh_epochs(
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
            args=ctx.args,
        )

        loss = float(result.get("loss", float("inf")))
        ctx.log_metric("stage2", "loss", loss)
        _log(f"[stage2] loss={loss:.4f}")


# ---------------------------------------------------------------------------
# Stage C — LoRA slot training node
# ---------------------------------------------------------------------------

class LoRARoundNode(GatedNode):
    """Stage C: Per-term LoRA adapter slot training.

    Requires all base gates.  For each active churn-term group a dedicated
    LoRA slot is activated in the classifier, trained for stageC_steps_per_slot
    steps on Berkeley payload rows matching those terms, then snapshotted.

    This allows the classifier to specialise per vocabulary subset without
    catastrophic forgetting of the shared backbone.
    """

    node_id = "stage_c_lora"
    description = "Stage C: LoRA slot switching and per-term Berkeley training"
    required_gates = ["gate_pregestation", "gate_gestation", "gate_berkeley"]

    def __init__(self, cfg: BerkeleyClassifierConfig) -> None:
        self.cfg = cfg

    def execute(self, ctx: PipelineContext) -> None:
        if not self.cfg.lora_enabled:
            return

        from wav_config_transformer_pipeline import (
            _run_berkeley_refresh_epochs,
            ensure_tiny_classifier_lora_slot,
            install_tiny_classifier_lora,
            tiny_classifier_lora_snapshot,
            restore_tiny_classifier_lora_snapshot,
            set_tiny_classifier_lora_state,
        )

        # Cycle through churn-group terms with dedicated LoRA slots
        churn_terms = ctx.active_extra_terms[: self.cfg.lora_churn_slots]
        for term in churn_terms:
            slot_name = _term_to_slot_name(term)
            ensure_tiny_classifier_lora_slot(
                ctx.classifier,
                slot_name=slot_name,
                rank=self.cfg.stageC_lora_rank,
            )
            install_tiny_classifier_lora(ctx.classifier, slot_name=slot_name)

            if ctx.berkeley_refresh_loader is not None:
                _run_berkeley_refresh_epochs(
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
                    args=ctx.args,
                )

            # Snapshot this slot so it can be restored later
            ctx.lora_slot_snapshots[slot_name] = tiny_classifier_lora_snapshot(
                ctx.classifier, slot_name=slot_name
            )

        # Return to base (no active LoRA) after the round
        set_tiny_classifier_lora_state(ctx.classifier, active_slot="")
        ctx.lora_active_slot = ""
        _log(f"[stageC] LoRA round complete for {len(churn_terms)} churn slots")


# ---------------------------------------------------------------------------
# Fake-class feedback node
# ---------------------------------------------------------------------------

class FakeClassFeedbackNode(GatedNode):
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

    def __init__(self, cfg: BerkeleyClassifierConfig) -> None:
        self.cfg = cfg

    def should_run(self, ctx: PipelineContext) -> bool:
        if not super().should_run(ctx):
            return False
        return (
            self.cfg.fake_class_enabled
            and ctx.generator is not None
            and ctx.discriminator is not None
        )

    def execute(self, ctx: PipelineContext) -> None:
        from wav_config_transformer_pipeline import _run_fake_class_refresh_epochs

        _run_fake_class_refresh_epochs(
            classifier=ctx.classifier,
            optimizer=ctx.classifier_optimizer,
            generator=ctx.generator,
            discriminator=ctx.discriminator,
            device=ctx.device,
            steps=self.cfg.fake_class_steps,
            batch_size=self.cfg.fake_class_batch_size,
            disc_weight=self.cfg.fake_class_disc_weight,
            amp_enabled=ctx.amp_enabled,
            amp_dtype=ctx.amp_dtype,
            grad_scaler=ctx.classifier_grad_scaler,
            class_names=ctx.class_names,
            args=ctx.args,
        )
        _log("[fake-class] feedback epoch complete")


# ---------------------------------------------------------------------------
# Gate replica sync node
# ---------------------------------------------------------------------------

class SyncGateReplicaNode(PipelineNode):
    """Keep the frozen CPU gate_classifier in sync with the training classifier.

    The replica is a deepcopy on CPU used for gate evaluation so gradient
    updates on the main model do not interfere with evaluation runs.
    """

    node_id = "sync_gate_replica"
    description = "Sync frozen CPU gate_classifier from main classifier"

    def __init__(self, cfg: BerkeleyClassifierConfig) -> None:
        self.cfg = cfg

    def should_run(self, ctx: PipelineContext) -> bool:
        return ctx.classifier is not None

    def execute(self, ctx: PipelineContext) -> None:
        from wav_config_transformer_pipeline import _sync_gate_classifier_replica

        ctx.gate_classifier, info = _sync_gate_classifier_replica(
            source_classifier=ctx.classifier,
            gate_classifier=ctx.gate_classifier,
            gate_device=torch.device("cpu"),
            channels_last=self.cfg.channels_last,
        )
        if info.get("created"):
            _log("[gate-replica] created fresh CPU replica")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _term_to_slot_name(term: str) -> str:
    import re
    slug = re.sub(r"[^a-z0-9]+", "_", term.strip().lower()).strip("_")
    return slug or "slot"


def _build_gate_replica(model: nn.Module, ctx: PipelineContext) -> nn.Module:
    base = unwrap_compiled(model)
    replica = copy.deepcopy(base).to(device=torch.device("cpu"), dtype=torch.float32)
    replica.eval()
    freeze(replica)
    return replica


def _load_classifier_checkpoint(model: nn.Module, path: str, scope: str = "all") -> None:
    from wav_config_transformer_pipeline import _apply_classifier_init

    try:
        _apply_classifier_init(model, path, scope=str(scope))
        _log(f"[classifier] loaded checkpoint from {path} (scope={scope})")
    except Exception as exc:
        _log(f"[classifier] WARNING: could not load checkpoint {path!r}: {exc}")


def _log(msg: str) -> None:
    print(msg, flush=True)
