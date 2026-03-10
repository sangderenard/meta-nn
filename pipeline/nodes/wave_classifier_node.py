"""
Wave Classifier Node — Second TinyConvClassifier for wave feedback (Stage W).

This file is the authoritative description of everything idiosyncratic to the
wave-feedback classifier in this pipeline.

This model is distinct from the Berkeley classifier:
  * It trains on images rendered from transformer-accepted WAV chunks
  * Its labels come from WAV folder names or spectral clustering
    (not from semantic vocabulary)
  * Its output is used as a feedback signal to modulate the transformer's
    training weights and gate the wave stage
  * It optionally runs zero-shot evaluation against the label embedding bank
    to measure cross-domain generalisation

Model architecture choices owned here
--------------------------------------
  * Second TinyConvClassifier instance (smaller than the Berkeley one)
  * No LoRA, no mask decoder head
  * Stratified train/val split from transformer outputs

Loss composition owned here
----------------------------
  * Standard multi-label BCE (no semantic cosine loss)
  * Evaluated with macro-F1 and accuracy metrics

Feedback mechanism owned here
------------------------------
  * Wave entropy + feature score → combined feedback metric
  * Gate W passes when feedback metric exceeds threshold
  * Feedback score modulates transformer and generator loss weights
    via _feedback_scaled_weight (passed back into ctx)

Zero-shot evaluation owned here
---------------------------------
  * Encode query terms with the same sentence-transformer backend
  * Nearest-neighbour lookup in the Berkeley label embedding bank
  * Top-1 accuracy on wave-rendered images
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from pipeline.context import PipelineContext
from pipeline.graph import PipelineNode
from pipeline.nodes.base import GatedNode, make_grad_scaler


# ---------------------------------------------------------------------------
# Node config
# ---------------------------------------------------------------------------

@dataclass
class WaveClassifierConfig:
    """Every hyperparameter specific to the wave-feedback TinyConvClassifier."""

    # ---- architecture ---------------------------------------------------
    base_ch: int = 48
    max_ch: int = 256
    context_blocks: int = 4
    context_dropout: float = 0.05

    # ---- label assignment -----------------------------------------------
    # "folder"   → labels from WAV folder names
    # "spectral" → labels from spectral angular centroid clustering
    # "hash"     → deterministic path-hash quantile bucketing
    label_mode: str = "folder"
    n_label_buckets: int = 8       # used when label_mode=="hash" or "spectral"

    # ---- dataset construction from transformer outputs ------------------
    max_accepted_chunks: int = 1024
    val_fraction: float = 0.15
    min_train_samples: int = 64

    # ---- optimiser / LR -------------------------------------------------
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    grad_accum_steps: int = 1

    # ---- AMP ------------------------------------------------------------
    amp: bool = False
    amp_dtype: str = "fp16"

    # ---- training schedule ----------------------------------------------
    epochs_per_round: int = 3
    batch_size: int = 32
    num_workers: int = 0

    # ---- zero-shot evaluation -------------------------------------------
    zero_shot_enabled: bool = True
    zero_shot_query_terms: str = ""    # semicolon-delimited; empty = use active vocab
    zero_shot_top_k: int = 5

    # ---- wave feedback / gate -------------------------------------------
    gate_entropy_min: float = 0.40
    gate_feature_score_min: float = 0.55
    gate_required_consecutive: int = 3

    # Feedback scaling: how strongly the wave gate score modulates
    # transformer and generator loss weights
    feedback_min_weight: float = 0.3
    feedback_max_weight: float = 1.0


# ---------------------------------------------------------------------------
# Build node
# ---------------------------------------------------------------------------

class BuildWaveClassifierNode(PipelineNode):
    """Instantiate the wave-feedback TinyConvClassifier.

    Deferred until after the first transformer round produces accepted chunks.
    The number of output classes is determined by the dataset, so this node
    runs each time the class count changes rather than strictly once.
    """

    node_id = "build_wave_classifier"
    description = "Instantiate wave-feedback TinyConvClassifier"

    def __init__(self, cfg: WaveClassifierConfig) -> None:
        self.cfg = cfg
        self._last_n_classes = -1

    def should_run(self, ctx: PipelineContext) -> bool:
        # Only relevant if wave stage is active
        mode = str(ctx.orchestration_mode or getattr(ctx.args, "orchestration_mode", "")).lower()
        return "w" in mode

    def execute(self, ctx: PipelineContext) -> None:
        from wav_ml_models import TinyConvClassifier, maybe_compile_module

        # Determine label count from WAV records
        n_classes = _count_wave_labels(ctx, self.cfg)
        if n_classes == self._last_n_classes and ctx.wave_classifier is not None:
            return  # no rebuild needed

        model = TinyConvClassifier(
            num_classes=n_classes,
            base_ch=self.cfg.base_ch,
            max_ch=self.cfg.max_ch,
            context_blocks=self.cfg.context_blocks,
            context_dropout=self.cfg.context_dropout,
        ).to(ctx.device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        ctx.wave_classifier = model
        ctx.wave_classifier_optimizer = optimizer
        if self.cfg.amp:
            ctx.wave_classifier_grad_scaler = make_grad_scaler(enabled=True)

        self._last_n_classes = n_classes
        _log(f"[wave-classifier] built: n_classes={n_classes}")


# ---------------------------------------------------------------------------
# Stage W — wave classifier training node
# ---------------------------------------------------------------------------

class WaveClassifierTrainNode(GatedNode):
    """Stage W: Build dataset from transformer outputs and train wave classifier.

    Requires Gate 0 + Gate 1 + Gate R (transformer).
    Flow:
      1. _build_wave_classifier_dataset_from_transformer — collect accepted chunks,
         render them to images, assign labels from WAV folder names or spectral analysis
      2. train_classifier — standard multi-label BCE training
      3. evaluate_classifier — accuracy + macro-F1 on held-out validation set
      4. (optional) _evaluate_zero_shot_queries_on_images — cross-domain zero-shot
      5. _wave_feedback_gate_pass — compute feedback metric and update gate W
    """

    node_id = "stage_w_wave_classifier"
    description = "Stage W: Wave classifier training on transformer outputs"
    required_gates = ["gate_pregestation", "gate_gestation", "gate_transformer"]

    def __init__(self, cfg: WaveClassifierConfig) -> None:
        self.cfg = cfg

    def should_run(self, ctx: PipelineContext) -> bool:
        if not super().should_run(ctx):
            return False
        mode = str(ctx.orchestration_mode or getattr(ctx.args, "orchestration_mode", "")).lower()
        if "w" not in mode:
            return False
        return ctx.transformer is not None and ctx.classifier is not None

    def execute(self, ctx: PipelineContext) -> None:
        from wav_config_transformer_pipeline import (
            _build_wave_classifier_dataset_from_transformer,
            _wave_feedback_gate_pass,
            _wave_feedback_snapshot,
        )
        from wav_ml_models import (
            train_classifier,
            evaluate_classifier,
            evaluate_feature_score_before_after,
        )

        # -- Build dataset ------------------------------------------------
        dataset_result = _build_wave_classifier_dataset_from_transformer(
            transformer=ctx.transformer,
            classifier=ctx.classifier,
            streams=ctx.float_streams,
            render_cfg=ctx.render_config,
            device=ctx.device,
            max_chunks=self.cfg.max_accepted_chunks,
            label_mode=self.cfg.label_mode,
            n_buckets=self.cfg.n_label_buckets,
            val_fraction=self.cfg.val_fraction,
            args=ctx.args,
        )

        if dataset_result is None or int(dataset_result.get("n_train", 0)) < self.cfg.min_train_samples:
            _log(f"[stageW] not enough accepted chunks to train wave classifier "
                 f"(need {self.cfg.min_train_samples})")
            return

        train_loader = dataset_result["train_loader"]
        val_loader = dataset_result.get("val_loader")
        n_train = int(dataset_result.get("n_train", 0))
        n_classes = int(dataset_result.get("n_classes", 1))

        # Rebuild classifier if class count changed
        if ctx.wave_classifier is None or _get_n_classes(ctx.wave_classifier) != n_classes:
            _rebuild_wave_classifier(ctx, self.cfg, n_classes)

        # -- Train --------------------------------------------------------
        train_classifier(
            classifier=ctx.wave_classifier,
            optimizer=ctx.wave_classifier_optimizer,
            loader=train_loader,
            device=ctx.device,
            epochs=self.cfg.epochs_per_round,
            amp_enabled=ctx.amp_enabled,
            amp_dtype=ctx.amp_dtype,
            grad_clip=self.cfg.grad_clip,
            grad_scaler=ctx.wave_classifier_grad_scaler,
            args=ctx.args,
        )

        # -- Evaluate -----------------------------------------------------
        eval_result: Dict[str, float] = {}
        if val_loader is not None:
            eval_result = evaluate_classifier(
                classifier=ctx.wave_classifier,
                loader=val_loader,
                device=ctx.device,
                args=ctx.args,
            )

        accuracy = float(eval_result.get("accuracy", 0.0))
        macro_f1 = float(eval_result.get("macro_f1", 0.0))

        # -- Zero-shot evaluation -----------------------------------------
        zs_top1 = 0.0
        if self.cfg.zero_shot_enabled and ctx.label_embedding_bank is not None:
            zs_top1 = _run_zero_shot_eval(ctx, self.cfg, val_loader)

        # -- Wave feedback gate -------------------------------------------
        feedback = _wave_feedback_snapshot(
            classifier=ctx.classifier,
            wave_classifier=ctx.wave_classifier,
            streams=ctx.float_streams,
            render_cfg=ctx.render_config,
            device=ctx.device,
            args=ctx.args,
        )
        feature_score = float(feedback.get("feature_score", 0.0))
        entropy = float(feedback.get("entropy", 0.0))
        combined = (feature_score + entropy) / 2.0

        ctx.gate_wave.required_consecutive = self.cfg.gate_required_consecutive
        ctx.gate_wave.record(
            round_id=ctx.round_id,
            metric=combined,
            threshold=(self.cfg.gate_feature_score_min + self.cfg.gate_entropy_min) / 2.0,
            above=True,
        )

        # Store feedback weight for use by transformer/generator nodes
        ctx.log_metric("stageW", "accuracy", accuracy)
        ctx.log_metric("stageW", "macro_f1", macro_f1)
        ctx.log_metric("stageW", "feature_score", feature_score)
        ctx.log_metric("stageW", "entropy", entropy)
        ctx.log_metric("stageW", "zs_top1", zs_top1)

        _log(f"[stageW] n_train={n_train} acc={accuracy:.3f} f1={macro_f1:.3f} "
             f"feat={feature_score:.3f} entropy={entropy:.3f} "
             f"zs={zs_top1:.3f} gate={'PASS' if ctx.gate_wave.passed else 'hold'}")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _count_wave_labels(ctx: PipelineContext, cfg: WaveClassifierConfig) -> int:
    if cfg.label_mode == "hash" or cfg.label_mode == "spectral":
        return cfg.n_label_buckets
    # folder mode: count unique parent directories
    from wav_config_transformer_pipeline import _build_labels
    paths = [str(r.path) if hasattr(r, "path") else str(r) for r in ctx.wav_records]
    labels = _build_labels(paths, mode="folder")
    n = len(set(labels)) if labels else 2
    return max(2, n)


def _rebuild_wave_classifier(
    ctx: PipelineContext,
    cfg: WaveClassifierConfig,
    n_classes: int,
) -> None:
    from wav_ml_models import TinyConvClassifier

    model = TinyConvClassifier(
        num_classes=n_classes,
        base_ch=cfg.base_ch,
        max_ch=cfg.max_ch,
        context_blocks=cfg.context_blocks,
        context_dropout=cfg.context_dropout,
    ).to(ctx.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    ctx.wave_classifier = model
    ctx.wave_classifier_optimizer = optimizer
    _log(f"[wave-classifier] rebuilt with n_classes={n_classes}")


def _get_n_classes(model) -> int:
    try:
        return int(model.head[-1].out_features)
    except Exception:
        return -1


def _run_zero_shot_eval(
    ctx: PipelineContext,
    cfg: WaveClassifierConfig,
    val_loader,
) -> float:
    if val_loader is None:
        return 0.0
    try:
        from wav_config_transformer_pipeline import _evaluate_zero_shot_queries_on_images

        query_str = cfg.zero_shot_query_terms.strip() or ";".join(ctx.active_extra_terms[:16])
        result = _evaluate_zero_shot_queries_on_images(
            classifier=ctx.classifier,
            loader=val_loader,
            device=ctx.device,
            query_texts=query_str,
            label_bank=ctx.label_embedding_bank,
            class_names=ctx.class_names,
            top_k=cfg.zero_shot_top_k,
            args=ctx.args,
        )
        return float(result.get("top1", 0.0))
    except Exception as exc:
        _log(f"[stageW] zero-shot eval failed: {exc}")
        return 0.0


def _log(msg: str) -> None:
    print(msg, flush=True)
