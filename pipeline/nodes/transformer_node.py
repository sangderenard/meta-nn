"""
Transformer Node — WavePatchTransformer training (Stage R) and config search.

This file is the authoritative description of everything idiosyncratic to the
WavePatchTransformer in this pipeline.

Model architecture choices owned here
--------------------------------------
  * WavePatchTransformer: d_model / nhead / num_layers / ff_mult / dropout
  * Optional deskew prefilter sub-module
  * Optional auxiliary filter bundles (transformer_filter_bundles)
  * torch.compile support

Loss composition owned here (Stage R)
--------------------------------------
  * Feature-score target loss  — pulls representations toward classifier's
    high-scoring regions (primary objective)
  * Rank loss                  — encourages diversity across the batch
  * Spurious penalty           — suppresses non-semantic texture shortcuts
  * Entropy regularisation     — prevents output collapse
  * High-bit / low-bit losses  — maintains signal structure across bit planes
  * Wave L1 reconstruction     — optional temporal fidelity term

Training schedule owned here
------------------------------
  * SinusoidalLR with configurable cycles
  * Degrade curriculum (progressively harder augmentations on inputs)
  * Gradient clipping and accumulation
  * AMP fp16 / bf16
  * channels_last support

Config search owned here
-------------------------
  * N random RenderConfig candidates scored against classifier feature metric
  * Optional scipy refinement
  * Best-config selection and persistence into ctx.render_config
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from pipeline.context import PipelineContext
from pipeline.graph import PipelineNode
from pipeline.nodes.base import GatedNode, make_grad_scaler


# ---------------------------------------------------------------------------
# Node config
# ---------------------------------------------------------------------------

@dataclass
class TransformerConfig:
    """Every hyperparameter specific to the WavePatchTransformer."""

    # ---- architecture ---------------------------------------------------
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    ff_mult: int = 4
    dropout: float = 0.1
    patch_size: int = 16          # pixels per patch (image side)
    image_size: int = 64          # rendered image side length in pixels
    chunk_samples: int = 0        # 0 = auto-derive from image_size + sample_rate

    # Deskew prefilter (learns to straighten slanted waveform renderings)
    deskew_enabled: bool = False
    deskew_channels: int = 16

    # Auxiliary filter bundles (secondary transformers per spectral band)
    filter_bundle_names: List[str] = field(default_factory=list)

    # ---- compile --------------------------------------------------------
    compile_model: bool = False
    compile_mode: str = "default"

    # ---- optimiser / LR -------------------------------------------------
    lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    grad_accum_steps: int = 1

    lr_cycles: float = 1.0
    lr_tail_fraction: float = 0.15
    lr_min_scale: float = 0.0

    # ---- AMP ------------------------------------------------------------
    amp: bool = False
    amp_dtype: str = "fp16"
    channels_last: bool = False

    # ---- Stage R loss weights -------------------------------------------
    feature_score_weight: float = 1.0
    rank_loss_weight: float = 0.10
    spurious_weight: float = 0.05
    entropy_weight: float = 0.01
    high_bit_weight: float = 0.20
    low_bit_weight: float = 0.10
    wave_l1_weight: float = 0.05

    # ---- Stage R training schedule --------------------------------------
    steps_per_round: int = 64
    degrade_curriculum_enabled: bool = True
    degrade_start_prob: float = 0.0
    degrade_end_prob: float = 0.5

    # ---- Config search --------------------------------------------------
    config_search_trials: int = 32
    config_search_epochs_per_trial: int = 1
    config_search_scipy_refine: bool = False

    # ---- Gate -----------------------------------------------------------
    gate_score_target: float = 0.60
    gate_entropy_min: float = 0.40
    gate_required_consecutive: int = 3

    # ---- Config search parameters ---------------------------------------
    seed: int = 42
    config_search_max_points: int = 32768
    config_search_batch_size: int = 16
    config_search_topk: int = 3
    config_search_threshold: float = 0.35
    config_search_w_topk: float = 0.60
    config_search_w_cov: float = 0.30
    config_search_w_mean: float = 0.10
    transformer_init_ckpt: str = ""
    sample_bits: int = 16


# ---------------------------------------------------------------------------
# Config search node (runs once at pipeline start)
# ---------------------------------------------------------------------------

class ConfigSearchNode(PipelineNode):
    """Find the best RenderConfig by scoring random candidates against the classifier.

    Runs once after the classifier has been built.  Persists the winning config
    into ctx.render_config.  Skipped if a config was loaded from a checkpoint.
    """

    node_id = "config_search"
    description = "Score random RenderConfig candidates via classifier feature metric"

    def __init__(self, cfg: TransformerConfig) -> None:
        self.cfg = cfg
        self._done = False

    def should_run(self, ctx: PipelineContext) -> bool:
        if self._done:
            return False
        if ctx.classifier is None:
            return False
        # Skip if render_config was already restored from a checkpoint
        return ctx.render_config is None

    def execute(self, ctx: PipelineContext) -> None:
        import numpy as np
        from wav_config_transformer_pipeline import (
            _score_config_with_classifier,
            _random_configs,
            _try_scipy_refine,
        )

        best_cfg = None
        best_score = float("-inf")
        # _random_configs(num_trials, rng, max_points) — not keyword-args-driven
        rng = np.random.default_rng(self.cfg.seed)
        max_points = self.cfg.config_search_max_points or 32768
        candidates = _random_configs(
            num_trials=self.cfg.config_search_trials,
            rng=rng,
            max_points=max_points,
        )

        for i, candidate in enumerate(candidates):
            # _score_config_with_classifier returns Dict[str, float]; key "score" is the composite
            score_result = _score_config_with_classifier(
                records=ctx.wav_records,
                indices=list(range(len(ctx.wav_records))),
                cfg=candidate,
                image_hw=(self.cfg.image_size, self.cfg.image_size),
                classifier=ctx.classifier,
                device=ctx.device,
                batch_size=self.cfg.config_search_batch_size,
                score_topk=self.cfg.config_search_topk,
                score_threshold=self.cfg.config_search_threshold,
                score_w_topk=self.cfg.config_search_w_topk,
                score_w_cov=self.cfg.config_search_w_cov,
                score_w_mean=self.cfg.config_search_w_mean,
                amp_enabled=ctx.amp_enabled,
                amp_dtype=str(ctx.amp_dtype or "float16"),
                channels_last=self.cfg.channels_last,
                active_classes=len(ctx.class_names),
            )
            score = float(score_result.get("score", float("-inf")))
            ctx.config_search_history.append({"trial": i, "score": score})
            if score > best_score:
                best_score = score
                best_cfg = candidate

        if self.cfg.config_search_scipy_refine and best_cfg is not None:
            best_cfg = _try_scipy_refine(
                base_cfg=best_cfg,
                classifier=ctx.classifier,
                streams=ctx.float_streams,
                device=ctx.device,
                args=ctx.args,
            ) or best_cfg

        ctx.render_config = best_cfg
        _log(f"[config-search] best_score={best_score:.4f} after {len(candidates)} trials")
        self._done = True


# ---------------------------------------------------------------------------
# Transformer build node
# ---------------------------------------------------------------------------

class BuildTransformerNode(PipelineNode):
    """Instantiate WavePatchTransformer and its optimizer.

    Runs once.  Requires render_config to be set (config search must precede this).
    """

    node_id = "build_transformer"
    description = "Instantiate WavePatchTransformer + optimizer"

    def __init__(self, cfg: TransformerConfig) -> None:
        self.cfg = cfg
        self._built = False

    def should_run(self, ctx: PipelineContext) -> bool:
        return not self._built and ctx.render_config is not None

    def execute(self, ctx: PipelineContext) -> None:
        from wav_ml_models import WavePatchTransformer, maybe_compile_module, SinusoidalLRController, SinusoidalLROptions

        model = WavePatchTransformer(
            d_model=self.cfg.d_model,
            nhead=self.cfg.nhead,
            num_layers=self.cfg.num_layers,
            ff_mult=self.cfg.ff_mult,
            dropout=self.cfg.dropout,
            image_size=self.cfg.image_size,
            patch_size=self.cfg.patch_size,
        ).to(ctx.device)

        if self.cfg.compile_model:
            model = maybe_compile_module(model, enabled=True, mode=self.cfg.compile_mode)

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

        ctx.transformer = model
        ctx.transformer_optimizer = optimizer
        ctx.transformer_lr_controller = lr_ctrl

        if self.cfg.amp:
            ctx.transformer_grad_scaler = make_grad_scaler(enabled=True)

        # Checkpoint load
        ckpt_path = self.cfg.transformer_init_ckpt or getattr(ctx.args, "resume_transformer", getattr(ctx.args, "transformer_init", "")) or ""
        if str(ckpt_path).strip():
            _load_transformer_checkpoint(model, str(ckpt_path))

        _log(f"[transformer] built: d_model={self.cfg.d_model} nhead={self.cfg.nhead} "
             f"num_layers={self.cfg.num_layers}")
        self._built = True


# ---------------------------------------------------------------------------
# Stage R — Transformer training node
# ---------------------------------------------------------------------------

class TransformerTrainNode(GatedNode):
    """Stage R: Train WavePatchTransformer to produce classifier-scoring outputs.

    Requires Gate 0 + Gate 1 (early gates).  The transformer generates image
    renderings from waveform chunks; it is trained so those renderings maximise
    the classifier feature score while maintaining wave-reconstruction fidelity.

    The degrade curriculum progressively applies augmentation distortions to
    the input chunks so the transformer learns robustness.

    Gate R (gate_transformer) passes when the combined feature-score + entropy
    metric exceeds its target for the required consecutive rounds.
    """

    node_id = "stage_r_transformer"
    description = "Stage R: WavePatchTransformer feature-score training"
    required_gates = ["gate_pregestation", "gate_gestation"]

    def __init__(self, cfg: TransformerConfig) -> None:
        self.cfg = cfg

    def should_run(self, ctx: PipelineContext) -> bool:
        if not super().should_run(ctx):
            return False
        return ctx.transformer is not None and ctx.classifier is not None

    def execute(self, ctx: PipelineContext) -> None:
        from wav_ml_models import train_transformer_feature_metric

        # Compute current degrade probability from curriculum schedule
        degrade_prob = _degrade_prob(
            ctx.total_rounds_completed,
            start=self.cfg.degrade_start_prob,
            end=self.cfg.degrade_end_prob,
            total_rounds=ctx.orchestration_cycles * ctx.orchestration_rounds,
            enabled=self.cfg.degrade_curriculum_enabled,
        )

        # train_transformer_feature_metric creates its own optimizer internally and
        # returns (trained_module, list_of_epoch_metric_dicts).  ctx.transformer_optimizer
        # is not consumed here — the function manages its own LR schedule per-call.
        trained_transformer, metrics_list = train_transformer_feature_metric(
            transformer=ctx.transformer,
            classifier=ctx.classifier,
            train_streams=ctx.float_streams,
            val_streams=ctx.float_streams,
            train_target_labels=None,
            cfg=ctx.render_config,
            sample_bits=self.cfg.sample_bits,
            image_hw=(self.cfg.image_size, self.cfg.image_size),
            device=ctx.device,
            epochs=1,
            steps_per_epoch=self.cfg.steps_per_round,
            lr=self.cfg.lr,
            lr_sine_cycles=self.cfg.lr_cycles,
            lr_sine_tail_fraction=self.cfg.lr_tail_fraction,
            lr_sine_min_scale=self.cfg.lr_min_scale,
            amp=ctx.amp_enabled,
            amp_dtype=str(ctx.amp_dtype or "float16"),
            channels_last=self.cfg.channels_last,
            grad_accum_steps=self.cfg.grad_accum_steps,
            grad_clip=self.cfg.grad_clip,
            degrade_inputs=self.cfg.degrade_curriculum_enabled and degrade_prob > 0,
            degrade_min_strength=degrade_prob,
            degrade_max_strength=degrade_prob,
            score_target_weight=self.cfg.feature_score_weight,
            score_rank_weight=self.cfg.rank_loss_weight,
            score_spurious_weight=self.cfg.spurious_weight,
            entropy_penalty_weight=self.cfg.entropy_weight,
            high_bit_penalty_weight=self.cfg.high_bit_weight,
            low_bit_penalty_weight=self.cfg.low_bit_weight,
            wave_l1_weight=self.cfg.wave_l1_weight,
        )

        ctx.transformer = trained_transformer
        last_m = (metrics_list or [{}])[-1]
        score = float(last_m.get("feature_score", 0.0))
        entropy = float(last_m.get("entropy", 0.0))
        combined = (score + entropy) / 2.0

        ctx.gate_transformer.required_consecutive = self.cfg.gate_required_consecutive
        ctx.gate_transformer.record(
            round_id=ctx.round_id,
            metric=combined,
            threshold=self.cfg.gate_score_target,
            above=True,
        )
        ctx.log_metric("stageR", "feature_score", score)
        ctx.log_metric("stageR", "entropy", entropy)

        _log(f"[stageR] score={score:.4f} entropy={entropy:.4f} "
             f"gate={'PASS' if ctx.gate_transformer.passed else 'hold'}")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _degrade_prob(
    step: int,
    start: float,
    end: float,
    total_rounds: int,
    enabled: bool,
) -> float:
    if not enabled or total_rounds <= 1:
        return float(start)
    t = min(1.0, float(step) / float(max(1, total_rounds - 1)))
    return float(start) + t * (float(end) - float(start))


def _load_transformer_checkpoint(model, path: str) -> None:
    from wav_config_transformer_pipeline import _apply_model_init, _torch_load_cpu

    try:
        ckpt = _torch_load_cpu(path)
        _apply_model_init(model, ckpt)
        _log(f"[transformer] loaded checkpoint from {path}")
    except Exception as exc:
        _log(f"[transformer] WARNING: could not load checkpoint {path!r}: {exc}")


def _log(msg: str) -> None:
    print(msg, flush=True)
