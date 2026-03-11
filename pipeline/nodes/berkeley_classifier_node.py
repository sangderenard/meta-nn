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
import gc
import math
import time
import numpy as np

# Module-level constants (originally in wav_config_transformer_pipeline)
CLASSIFIER_LOSS_SCALE = 1.0
CLASSIFIER_SEMANTIC_COSINE_WEIGHT = 0.35


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

    try:
        _apply_classifier_init(model, path, scope=str(scope))
        _log(f"[classifier] loaded checkpoint from {path} (scope={scope})")
    except Exception as exc:
        _log(f"[classifier] WARNING: could not load checkpoint {path!r}: {exc}")


def _log(msg: str) -> None:
    print(msg, flush=True)


# =========================================================================
# Functions extracted from wav_config_transformer_pipeline.py
# =========================================================================


def _sync_gate_classifier_replica(
    source_classifier: nn.Module,
    gate_classifier: Optional[nn.Module],
    gate_device: torch.device,
    channels_last: bool = False,
) -> Tuple[nn.Module, Dict[str, Any]]:
    source_base = _unwrap_module_for_replica(source_classifier)
    created = False
    if gate_classifier is None:
        gate_classifier = copy.deepcopy(source_base)
        created = True
    gate_classifier = gate_classifier.to(device=gate_device, dtype=torch.float32)
    gate_classifier.load_state_dict(source_base.state_dict(), strict=True)
    gate_classifier.eval()
    for p in gate_classifier.parameters():
        p.requires_grad_(False)
    if bool(channels_last):
        gate_classifier = gate_classifier.to(memory_format=torch.channels_last)
    return gate_classifier, {
        "created": bool(created),
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


def _run_berkeley_refresh_epochs(
    classifier: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    max_steps: int,
    min_steps: int,
    lr_sine_cycles: float,
    lr_sine_frequency: float,
    lr_sine_tail_fraction: float,
    lr_sine_min_scale: float,
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
    stop_requested: Optional[Callable[[], bool]] = None,
    grad_clip: float = 1.0,
    semantic_soft_target_max: float = 0.0,
    semantic_cosine_weight: float = CLASSIFIER_SEMANTIC_COSINE_WEIGHT,
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
    amp_dtype_t = _resolve_amp_dtype(amp_dtype) if amp_enabled else torch.float16
    use_scaler = bool(amp_enabled and device.type == "cuda" and amp_dtype_t == torch.float16)
    scaler = _make_grad_scaler(enabled=use_scaler)
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
    opt = torch.optim.AdamW(classifier.parameters(), lr=float(lr), weight_decay=float(weight_decay))
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
    lr_ctl = SinusoidalLRController(
        optimizer=opt,
        total_steps=max(1, int(epochs) * updates_per_epoch),
        options=SinusoidalLROptions(
            cycles=float(lr_sine_cycles),
            frequency=float(lr_sine_frequency),
            tail_fraction=float(lr_sine_tail_fraction),
            min_scale=float(lr_sine_min_scale),
        ),
    )
    total_loss = 0.0
    n = 0
    t_start = time.time()
    if device.type == "cuda":
        _log(f"[berkeley-refresh] cuda preflight: {_cuda_mem_diag(device)}")
    total_target_steps = max(1, int(epochs) * steps_per_epoch)
    global_step = 0
    stop_now = False
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
                            with _autocast_context(device=device, enabled=amp_enabled, amp_dtype_t=amp_dtype_t):
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
                    lr_ctl.step()
                    opt.zero_grad(set_to_none=True)
                total_loss += float(step_loss_value)
                n += int(step_seen)
                step += 1
                global_step += 1
                if int(log_every) > 0 and (global_step % int(log_every) == 0):
                    elapsed = max(1e-6, time.time() - t_start)
                    ips = float(n) / elapsed
                    print(
                        f"[berkeley-refresh] step={global_step}/{total_target_steps} "
                        f"loss={total_loss / max(1, n):.4f} samples={n} samp_per_sec={ips:.1f}",
                        flush=True,
                    )
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
    stop_requested: Optional[Callable[[], bool]] = None,
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

    amp_dtype_t = _resolve_amp_dtype(amp_dtype) if amp_enabled else torch.float16
    use_scaler = bool(amp_enabled and device.type == "cuda" and amp_dtype_t == torch.float16)
    scaler = _make_grad_scaler(enabled=use_scaler)
    grad_accum_steps = max(1, int(grad_accum_steps))
    n_steps = max(1, int(steps_per_epoch))
    updates_per_epoch = int(math.ceil(float(n_steps) / float(grad_accum_steps)))
    opt = torch.optim.AdamW(classifier.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    lr_ctl = SinusoidalLRController(
        optimizer=opt,
        total_steps=max(1, int(epochs) * max(1, int(updates_per_epoch))),
        options=SinusoidalLROptions(
            cycles=float(lr_sine_cycles),
            frequency=float(lr_sine_frequency),
            tail_fraction=float(lr_sine_tail_fraction),
            min_scale=float(lr_sine_min_scale),
        ),
    )

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
                with _autocast_context(device=device, enabled=amp_enabled, amp_dtype_t=amp_dtype_t):
                    xb = generator(z, cond).to(torch.float32)
                disc_logits = None
                if discriminator is not None:
                    with _autocast_context(device=device, enabled=amp_enabled, amp_dtype_t=amp_dtype_t):
                        disc_logits = discriminator(xb, cond).to(torch.float32)
            if channels_last:
                xb = xb.contiguous(memory_format=torch.channels_last)
            with _autocast_context(device=device, enabled=amp_enabled, amp_dtype_t=amp_dtype_t):
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
                lr_ctl.step()
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
