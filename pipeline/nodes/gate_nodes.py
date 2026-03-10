"""
Gate Nodes — evaluation nodes that check whether a stage has passed its threshold.

Each gate node:
  1. Runs the relevant evaluation (classifier accuracy, feature score, etc.)
  2. Records the metric in the corresponding GateState
  3. Optionally logs to ctx.loss_logger / ctx.viewer_proxy

Gate checks are separate nodes so they appear explicitly in the graph as edges
and their conditions can be composed with other node conditions naturally.

Gates in this pipeline
-----------------------
  BerkeleyGateNode      — Gate 2: Berkeley confidence + macro-F1 after Stage 2 training
  TransformerGateNode   — Gate R: transformer feature score + entropy
  GeneratorGateNode     — Gate G: generator feature score quality
  WaveGateNode          — Gate W: wave entropy + feedback score
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from pipeline.context import PipelineContext
from pipeline.graph import PipelineNode
from pipeline.nodes.base import GatedNode


# ---------------------------------------------------------------------------
# Berkeley gate  (Gate 2)
# ---------------------------------------------------------------------------

@dataclass
class BerkeleyGateConfig:
    """Thresholds for the Berkeley SBD classifier gate."""

    # Minimum mean classification confidence (model certainty)
    confidence_target: float = 0.55

    # Minimum macro-averaged F1 score across all classes
    f1_target: float = 0.40

    # Optional: maximum cross-entropy loss (0.0 = disabled)
    loss_target: float = 0.0

    # Number of consecutive rounds above threshold before gate flips
    required_consecutive: int = 1

    # Batch size for gate evaluation pass
    eval_batch_size: int = 32


class BerkeleyGateNode(GatedNode):
    """Gate 2: Evaluate classifier confidence and F1 on Berkeley validation set.

    Requires Gate 0 + Gate 1.  Uses the frozen gate_classifier replica on CPU
    to avoid disrupting training-mode batch-norm statistics on the main model.

    The gate passes when both confidence AND macro-F1 exceed their targets
    for the required consecutive rounds.
    """

    node_id = "gate_berkeley"
    description = "Gate 2: Berkeley classifier confidence + macro-F1 evaluation"
    required_gates = ["gate_pregestation", "gate_gestation"]

    def __init__(self, cfg: BerkeleyGateConfig) -> None:
        self.cfg = cfg

    def should_run(self, ctx: PipelineContext) -> bool:
        if not super().should_run(ctx):
            return False
        return (
            ctx.payload_validation_loader is not None
            and ctx.gate_classifier is not None
        )

    def execute(self, ctx: PipelineContext) -> None:
        from wav_config_transformer_pipeline import _evaluate_berkeley_classifier_gate

        result = _evaluate_berkeley_classifier_gate(
            classifier=ctx.gate_classifier,
            loader=ctx.payload_validation_loader,
            device=ctx.device,
            class_names=ctx.class_names,
            args=ctx.args,
        )

        confidence = float(result.get("mean_confidence", 0.0))
        macro_f1 = float(result.get("macro_f1", 0.0))
        loss = float(result.get("loss", float("inf")))

        # Combined metric: harmonic mean of normalised confidence and F1
        conf_norm = min(1.0, confidence / max(1e-6, self.cfg.confidence_target))
        f1_norm = min(1.0, macro_f1 / max(1e-6, self.cfg.f1_target))
        combined = 2.0 * conf_norm * f1_norm / max(1e-6, conf_norm + f1_norm)

        # Both metrics must individually meet their targets
        passes = (
            confidence >= self.cfg.confidence_target
            and macro_f1 >= self.cfg.f1_target
            and (self.cfg.loss_target <= 0.0 or loss <= self.cfg.loss_target)
        )

        ctx.gate_berkeley.required_consecutive = self.cfg.required_consecutive
        if passes:
            ctx.gate_berkeley.record(
                round_id=ctx.round_id, metric=combined, threshold=0.99, above=True
            )
        else:
            # Record a sub-threshold metric to reset consecutive counter
            ctx.gate_berkeley.record(
                round_id=ctx.round_id, metric=0.0, threshold=0.99, above=True
            )

        ctx.log_metric("gate2", "confidence", confidence)
        ctx.log_metric("gate2", "macro_f1", macro_f1)
        ctx.log_metric("gate2", "loss", loss)

        _log(
            f"[gate2] conf={confidence:.3f}/{self.cfg.confidence_target:.2f} "
            f"f1={macro_f1:.3f}/{self.cfg.f1_target:.2f} "
            f"consecutive={ctx.gate_berkeley.consecutive_passes}/"
            f"{self.cfg.required_consecutive} "
            f"{'PASS' if ctx.gate_berkeley.passed else 'hold'}"
        )


# ---------------------------------------------------------------------------
# Transformer gate  (Gate R)
# ---------------------------------------------------------------------------

@dataclass
class TransformerGateConfig:
    """Thresholds for the transformer feature-score gate."""

    # Minimum combined (feature_score + entropy) / 2 metric
    score_target: float = 0.60

    # Individual minimums (both must be met)
    feature_score_min: float = 0.50
    entropy_min: float = 0.40

    required_consecutive: int = 3


class TransformerGateNode(GatedNode):
    """Gate R: Evaluate transformer output quality via classifier feature score.

    Requires Gate 0 + Gate 1.  Scores the transformer's current renderings
    and checks both the combined metric and the individual component thresholds.
    """

    node_id = "gate_transformer"
    description = "Gate R: transformer feature score + entropy evaluation"
    required_gates = ["gate_pregestation", "gate_gestation"]

    def __init__(self, cfg: TransformerGateConfig) -> None:
        self.cfg = cfg

    def should_run(self, ctx: PipelineContext) -> bool:
        if not super().should_run(ctx):
            return False
        return ctx.transformer is not None and ctx.classifier is not None

    def execute(self, ctx: PipelineContext) -> None:
        from wav_config_transformer_pipeline import _wave_feedback_snapshot

        result = _wave_feedback_snapshot(
            classifier=ctx.classifier,
            wave_classifier=ctx.wave_classifier,
            streams=ctx.float_streams,
            render_cfg=ctx.render_config,
            device=ctx.device,
            args=ctx.args,
        )

        feature_score = float(result.get("feature_score", 0.0))
        entropy = float(result.get("entropy", 0.0))
        combined = (feature_score + entropy) / 2.0

        passes = (
            combined >= self.cfg.score_target
            and feature_score >= self.cfg.feature_score_min
            and entropy >= self.cfg.entropy_min
        )

        ctx.gate_transformer.required_consecutive = self.cfg.required_consecutive
        threshold_metric = combined if passes else 0.0
        ctx.gate_transformer.record(
            round_id=ctx.round_id,
            metric=threshold_metric,
            threshold=self.cfg.score_target * 0.99,
            above=True,
        )

        ctx.log_metric("gateR", "feature_score", feature_score)
        ctx.log_metric("gateR", "entropy", entropy)

        _log(
            f"[gateR] score={feature_score:.3f} entropy={entropy:.3f} "
            f"combined={combined:.3f}/{self.cfg.score_target:.2f} "
            f"consecutive={ctx.gate_transformer.consecutive_passes}/"
            f"{self.cfg.required_consecutive} "
            f"{'PASS' if ctx.gate_transformer.passed else 'hold'}"
        )


# ---------------------------------------------------------------------------
# Generator gate  (Gate G)
# ---------------------------------------------------------------------------

@dataclass
class GeneratorGateConfig:
    """Thresholds for the GAN generator quality gate."""

    # Minimum feature score of generated images as judged by the classifier
    feature_score_target: float = 0.50
    required_consecutive: int = 2


class GeneratorGateNode(GatedNode):
    """Gate G: Evaluate generator output quality via classifier feature score.

    Requires all base gates + the generator to exist.
    Generates a small batch of images and scores them.
    """

    node_id = "gate_generator"
    description = "Gate G: generator feature score evaluation"
    required_gates = ["gate_pregestation", "gate_gestation", "gate_berkeley"]

    def __init__(self, cfg: GeneratorGateConfig) -> None:
        self.cfg = cfg

    def should_run(self, ctx: PipelineContext) -> bool:
        if not super().should_run(ctx):
            return False
        return ctx.generator is not None and ctx.classifier is not None

    def execute(self, ctx: PipelineContext) -> None:
        from wav_ml_models import evaluate_conditional_generator

        result = evaluate_conditional_generator(
            generator=ctx.generator,
            classifier=ctx.classifier,
            class_names=ctx.class_names,
            device=ctx.device,
            n_samples=32,
            args=ctx.args,
        )

        feat_score = float(result.get("feature_score", 0.0))

        ctx.gate_generator.required_consecutive = self.cfg.required_consecutive
        ctx.gate_generator.record(
            round_id=ctx.round_id,
            metric=feat_score,
            threshold=self.cfg.feature_score_target,
            above=True,
        )

        ctx.log_metric("gateG", "feature_score", feat_score)
        _log(
            f"[gateG] feat={feat_score:.3f}/{self.cfg.feature_score_target:.2f} "
            f"consecutive={ctx.gate_generator.consecutive_passes}/"
            f"{self.cfg.required_consecutive} "
            f"{'PASS' if ctx.gate_generator.passed else 'hold'}"
        )


# ---------------------------------------------------------------------------
# Wave feedback gate  (Gate W)
# ---------------------------------------------------------------------------

@dataclass
class WaveGateConfig:
    """Thresholds for the wave entropy + feedback gate."""

    entropy_min: float = 0.40
    feature_score_min: float = 0.55
    required_consecutive: int = 3


class WaveGateNode(GatedNode):
    """Gate W: Combined wave entropy + classifier feature score gate.

    Requires early gates + transformer gate.  Checks the wave-level feedback
    metric; used to conditionally unlock the wave classifier stage.
    """

    node_id = "gate_wave"
    description = "Gate W: wave entropy + feature score feedback evaluation"
    required_gates = ["gate_pregestation", "gate_gestation", "gate_transformer"]

    def __init__(self, cfg: WaveGateConfig) -> None:
        self.cfg = cfg

    def should_run(self, ctx: PipelineContext) -> bool:
        if not super().should_run(ctx):
            return False
        mode = str(ctx.orchestration_mode or getattr(ctx.args, "orchestration_mode", "")).lower()
        return "w" in mode and ctx.classifier is not None

    def execute(self, ctx: PipelineContext) -> None:
        from wav_config_transformer_pipeline import _wave_feedback_snapshot

        result = _wave_feedback_snapshot(
            classifier=ctx.classifier,
            wave_classifier=ctx.wave_classifier,
            streams=ctx.float_streams,
            render_cfg=ctx.render_config,
            device=ctx.device,
            args=ctx.args,
        )

        feature_score = float(result.get("feature_score", 0.0))
        entropy = float(result.get("entropy", 0.0))
        combined = (feature_score + entropy) / 2.0
        target = (self.cfg.feature_score_min + self.cfg.entropy_min) / 2.0

        passes = feature_score >= self.cfg.feature_score_min and entropy >= self.cfg.entropy_min
        metric = combined if passes else 0.0

        ctx.gate_wave.required_consecutive = self.cfg.required_consecutive
        ctx.gate_wave.record(
            round_id=ctx.round_id,
            metric=metric,
            threshold=target * 0.99,
            above=True,
        )

        ctx.log_metric("gateW", "feature_score", feature_score)
        ctx.log_metric("gateW", "entropy", entropy)
        _log(
            f"[gateW] feat={feature_score:.3f} entropy={entropy:.3f} "
            f"{'PASS' if ctx.gate_wave.passed else 'hold'}"
        )


# ---------------------------------------------------------------------------
# Checkpoint save node
# ---------------------------------------------------------------------------

class CheckpointSaveNode(PipelineNode):
    """Persist all models and pipeline state to disk.

    Runs at the end of every round.  Atomic write to avoid partial checkpoints.
    """

    node_id = "checkpoint_save"
    description = "Save pipeline checkpoint to disk"

    def __init__(self, save_every_n_rounds: int = 1) -> None:
        self.save_every_n_rounds = max(1, int(save_every_n_rounds))

    def should_run(self, ctx: PipelineContext) -> bool:
        return (ctx.round_id % self.save_every_n_rounds) == 0

    def execute(self, ctx: PipelineContext) -> None:
        from wav_config_transformer_pipeline import _save_training_segment_snapshot

        _save_training_segment_snapshot(
            output_dir=ctx.output_dir,
            classifier=ctx.classifier,
            transformer=ctx.transformer,
            generator=ctx.generator,
            discriminator=ctx.discriminator,
            wave_classifier=ctx.wave_classifier,
            render_config=ctx.render_config,
            class_names=ctx.class_names,
            label_embedding_bank=ctx.label_embedding_bank,
            gate_state={
                "pregestation": ctx.gate_pregestation.passed,
                "gestation": ctx.gate_gestation.passed,
                "berkeley": ctx.gate_berkeley.passed,
                "transformer": ctx.gate_transformer.passed,
                "generator": ctx.gate_generator.passed,
                "wave": ctx.gate_wave.passed,
            },
            metrics_history=ctx.metrics_history,
            cycle=ctx.cycle,
            round_id=ctx.round_id,
            args=ctx.args,
        )
        _log(f"[checkpoint] saved round={ctx.round_id}")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(msg, flush=True)
