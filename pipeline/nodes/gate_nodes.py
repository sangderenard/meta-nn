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
import copy
import json
import time
import numpy as np
import torch

# Module-level constants (originally in wav_config_transformer_pipeline)
CLASSIFIER_LOSS_SCALE = 1.0
CLASSIFIER_SEMANTIC_COSINE_WEIGHT = 0.35


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


# =========================================================================
# Functions extracted from wav_config_transformer_pipeline.py
# =========================================================================


def _wave_feedback_snapshot(
    wave_classifier_eval: Optional[Dict[str, Any]],
    wave_zero_shot_eval: Optional[Dict[str, Any]],
    zero_shot_weight: float,
) -> Dict[str, Any]:
    acc = None
    if isinstance(wave_classifier_eval, dict) and ("acc" in wave_classifier_eval):
        try:
            acc = float(wave_classifier_eval.get("acc", 0.0))
        except Exception:
            acc = None

    zs_top1 = None
    zs_ran = False
    if isinstance(wave_zero_shot_eval, dict) and bool(wave_zero_shot_eval.get("ran", False)):
        zs_ran = True
        try:
            zs_top1 = float(wave_zero_shot_eval.get("best_query_top1_rate", 0.0))
        except Exception:
            zs_top1 = None

    weighted_sum = 0.0
    weighted_den = 0.0
    if acc is not None:
        weighted_sum += float(acc)
        weighted_den += 1.0
    if zs_top1 is not None:
        z_w = max(0.0, float(zero_shot_weight))
        if z_w > 0.0:
            weighted_sum += z_w * float(zs_top1)
            weighted_den += z_w

    combined = None
    if weighted_den > 0.0:
        combined = float(weighted_sum / weighted_den)
        combined = float(max(0.0, min(1.0, combined)))

    return {
        "available": bool(weighted_den > 0.0),
        "wave_acc": (None if acc is None else float(max(0.0, min(1.0, acc)))),
        "zero_shot_ran": bool(zs_ran),
        "zero_shot_top1_rate": (None if zs_top1 is None else float(max(0.0, min(1.0, zs_top1)))),
        "combined_score": combined,
    }


def _feedback_scaled_weight(base_weight: float, feedback_score: Optional[float], boost: float) -> float:
    base = max(0.0, float(base_weight))
    b = max(0.0, float(boost))
    if feedback_score is None:
        return float(base)
    s = float(max(0.0, min(1.0, float(feedback_score))))
    return float(base * (1.0 + (b * (1.0 - s))))


def _wave_feedback_gate_pass(
    feedback: Dict[str, Any],
    min_acc: float,
    min_zero_shot_top1_rate: float,
) -> Tuple[bool, str]:
    if not bool(feedback.get("available", False)):
        return True, "warmup_no_wave_feedback"

    acc_floor = max(0.0, float(min_acc))
    zs_floor = max(0.0, float(min_zero_shot_top1_rate))
    acc = feedback.get("wave_acc", None)
    zs = feedback.get("zero_shot_top1_rate", None)
    if acc is not None and acc_floor > 0.0 and float(acc) < acc_floor:
        return False, f"wave_acc<{acc_floor:.3f}"
    if zs is not None and zs_floor > 0.0 and float(zs) < zs_floor:
        return False, f"wave_zs_top1<{zs_floor:.3f}"
    return True, "pass"


def _save_training_segment_snapshot(
    enabled: bool,
    out_dir: Path,
    objective_mode: str,
    run_tag: str,
    segment: str,
    best_cfg: Optional[RenderConfig],
    classifier: Optional[nn.Module] = None,
    transformer: Optional[nn.Module] = None,
    generator: Optional[nn.Module] = None,
    discriminator: Optional[nn.Module] = None,
    wave_classifier: Optional[nn.Module] = None,
    classifier_history: Optional[Sequence[Dict]] = None,
    transformer_history: Optional[Sequence[Dict]] = None,
    generator_history: Optional[Sequence[Dict]] = None,
    wave_classifier_history: Optional[Sequence[Dict]] = None,
    orchestration_history: Optional[Sequence[Dict]] = None,
    refresh_history: Optional[Sequence[Dict]] = None,
    gate_history: Optional[Sequence[Dict]] = None,
    gate_status: Optional[Dict] = None,
    extra: Optional[Dict] = None,
):
    if not bool(enabled):
        return

    payload = {
        "run_tag": run_tag,
        "objective_mode": str(objective_mode),
        "segment": str(segment),
        "timestamp": time.time(),
    }
    if best_cfg is not None:
        try:
            payload["best_cfg"] = best_cfg.to_dict()
        except Exception:
            pass
    if classifier is not None:
        payload["classifier_state"] = classifier.state_dict()
        payload["classifier_lora"] = _snapshot_classifier_lora(classifier)
    if transformer is not None:
        payload["transformer_state"] = transformer.state_dict()
    if generator is not None:
        payload["generator_state"] = generator.state_dict()
    if discriminator is not None:
        payload["discriminator_state"] = discriminator.state_dict()
    if wave_classifier is not None:
        payload["wave_classifier_state"] = wave_classifier.state_dict()
    if classifier_history is not None:
        payload["classifier_history"] = list(classifier_history)
    if transformer_history is not None:
        payload["transformer_history"] = list(transformer_history)
    if generator_history is not None:
        payload["generator_history"] = list(generator_history)
    if wave_classifier_history is not None:
        payload["wave_classifier_history"] = list(wave_classifier_history)
    if orchestration_history is not None:
        payload["orchestration_history"] = list(orchestration_history)
    if refresh_history is not None:
        payload["refresh_history"] = list(refresh_history)
    if gate_history is not None:
        payload["gate_history"] = list(gate_history)
    if isinstance(gate_status, dict):
        payload["gate_status"] = dict(gate_status)
    if isinstance(extra, dict):
        payload.update(extra)

    _save_pipeline_checkpoint(out_dir / "pipeline_checkpoint.pt", payload)
    if classifier is not None:
        torch.save(
            {
                "state_dict": classifier.state_dict(),
                "classifier_lora": _snapshot_classifier_lora(classifier),
            },
            out_dir / "classifier.pt",
        )
    if transformer is not None:
        torch.save({"state_dict": transformer.state_dict()}, out_dir / "transformer.pt")
    if generator is not None:
        torch.save({"state_dict": generator.state_dict()}, out_dir / "generator.pt")
    if discriminator is not None:
        torch.save({"state_dict": discriminator.state_dict()}, out_dir / "discriminator.pt")
    if wave_classifier is not None:
        torch.save({"state_dict": wave_classifier.state_dict()}, out_dir / "wave_classifier.pt")


def _evaluate_berkeley_classifier_gate(
    classifier: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_steps: int,
    active_classes: int = 0,
    amp_enabled: bool = False,
    amp_dtype: str = "float16",
    channels_last: bool = False,
    semantic_mask_supervision_mode: str = "multihot_mix",
    preview_sink: Optional[Dict[str, Any]] = None,
    semantic_cosine_weight: float = CLASSIFIER_SEMANTIC_COSINE_WEIGHT,
):
    classifier.eval()
    amp_dtype_t = _resolve_amp_dtype(amp_dtype) if amp_enabled else torch.float16
    runtime_amp_enabled = bool(amp_enabled and device.type == "cuda")
    runtime_channels_last = bool(channels_last)
    if bool(runtime_amp_enabled) and not _cuda_device_supports_amp_dtype(device=device, amp_dtype_t=amp_dtype_t):
        print(
            f"[berkeley-gate] disabling AMP on device={device} because amp_dtype={amp_dtype} is unsupported there",
            flush=True,
        )
        runtime_amp_enabled = False
    total_loss = 0.0
    n = 0
    logits_all = []
    targets_all = []
    steps = 0

    def _is_cuda_oom(exc: BaseException) -> bool:
        txt = str(exc).lower()
        if "out of memory" not in txt:
            return False
        return ("cuda" in txt) or ("cudnn" in txt) or ("cublas" in txt)

    batch_idx = 0
    mask_loss_total = 0.0
    for batch in loader:
        batch_idx += 1
        xb, yb, mb, batch_meta = _unpack_masked_semantic_batch(batch, context="berkeley gate evaluation")
        xb, yb, mb = _expand_semantic_mask_supervision_batch(
            xb=xb,
            yb=yb,
            mb=mb,
            batch_meta=batch_meta,
            mode=str(semantic_mask_supervision_mode),
            context="berkeley gate evaluation",
        )
        batch_n = int(xb.shape[0])
        eval_chunk = max(1, int(batch_n))

        while True:
            try:
                batch_loss = 0.0
                batch_seen = 0
                batch_logits: List[torch.Tensor] = []
                batch_targets: List[torch.Tensor] = []
                batch_mask_loss = 0.0
                for start in range(0, batch_n, eval_chunk):
                    stop = min(batch_n, start + eval_chunk)
                    xb_part = xb[start:stop]
                    yb_part = yb[start:stop]
                    mb_part = mb[start:stop]
                    if xb_part.device != device:
                        if runtime_channels_last:
                            xb_part = xb_part.to(device=device, non_blocking=True, memory_format=torch.channels_last)
                        else:
                            xb_part = xb_part.to(device, non_blocking=True)
                    elif runtime_channels_last:
                        xb_part = xb_part.contiguous(memory_format=torch.channels_last)
                    if yb_part.device != device:
                        yb_part = yb_part.to(device, non_blocking=True)
                    if mb_part.device != device:
                        mb_part = mb_part.to(device, non_blocking=True)
                    with torch.no_grad():
                        with _autocast_context(device=device, enabled=runtime_amp_enabled, amp_dtype_t=amp_dtype_t):
                            out = _forward_classifier_outputs_require_mask(
                                classifier=classifier,
                                xb=xb_part,
                                context="berkeley gate evaluation",
                            )
                            logits = out["logits"]
                        supervised_dim = int(yb_part.shape[1]) if int(yb_part.ndim) == 2 else int(logits.shape[1])
                        if int(active_classes) > 0:
                            supervised_dim = min(int(supervised_dim), int(active_classes))
                        supervised_dim = max(1, int(supervised_dim))
                        loss, logits_sup, y_sup, _ = _classifier_supervision_loss(
                            classifier=classifier,
                            logits=logits,
                            y_multihot=yb_part,
                            supervised_dim=int(supervised_dim),
                            semantic_cosine_weight=float(semantic_cosine_weight),
                        )
                        mask_loss, _ = _semantic_mask_bce_loss(
                            mask_logits=out["mask_logits"],
                            mask_targets=mb_part,
                            context="berkeley gate evaluation",
                        )
                        loss = (loss * float(CLASSIFIER_LOSS_SCALE)) + mask_loss
                    part_n = int(xb_part.shape[0])
                    batch_loss += float(loss.item()) * part_n
                    batch_mask_loss += float(mask_loss.item()) * part_n
                    batch_seen += part_n
                    batch_logits.append(logits_sup.detach().cpu())
                    batch_targets.append(y_sup.detach().cpu())
                    if isinstance(preview_sink, dict) and int(part_n) > 0:
                        try:
                            cap_i = int(part_n - 1)
                            cap_img = xb_part[cap_i].detach().to(torch.float32).cpu().numpy().astype(np.float32, copy=False)
                            cap_target = y_sup[cap_i].detach().to(torch.float32).cpu().numpy().astype(np.float32, copy=False)
                            cap_probs = torch.sigmoid(logits_sup[cap_i].to(torch.float32)).detach().cpu().numpy().astype(np.float32, copy=False)
                            preview_sink.clear()
                            preview_sink.update(
                                {
                                    "valid": True,
                                    "batch": int(batch_idx),
                                    "chunk_start": int(start),
                                    "chunk_stop": int(stop),
                                    "image": np.asarray(cap_img, dtype=np.float32),
                                    "target": np.asarray(cap_target, dtype=np.float32).reshape(-1),
                                    "probs": np.asarray(cap_probs, dtype=np.float32).reshape(-1),
                                }
                            )
                        except Exception:
                            pass
                    del xb_part, yb_part, mb_part, out, logits, loss, mask_loss, logits_sup, y_sup

                total_loss += float(batch_loss)
                mask_loss_total += float(batch_mask_loss)
                n += int(batch_seen)
                logits_all.extend(batch_logits)
                targets_all.extend(batch_targets)
                break
            except RuntimeError as e:
                if device.type == "cuda" and _is_cuda_backend_engine_error(e):
                    if bool(runtime_amp_enabled):
                        runtime_amp_enabled = False
                        torch.cuda.empty_cache()
                        print(
                            f"[berkeley-gate] backend engine selection failed on device={device}; retrying with AMP disabled",
                            flush=True,
                        )
                        continue
                    if bool(runtime_channels_last):
                        runtime_channels_last = False
                        classifier = classifier.to(memory_format=torch.contiguous_format)
                        xb = xb.contiguous()
                        torch.cuda.empty_cache()
                        print(
                            f"[berkeley-gate] backend engine selection failed on device={device}; retrying with contiguous tensors",
                            flush=True,
                        )
                        continue
                if device.type != "cuda" or (not _is_cuda_oom(e)) or int(eval_chunk) <= 1:
                    raise
                next_chunk = max(1, int(eval_chunk) // 2)
                if int(next_chunk) > 1:
                    next_chunk = 1 << (int(next_chunk).bit_length() - 1)
                torch.cuda.empty_cache()
                print(
                    f"[berkeley-gate] CUDA OOM at eval_chunk={eval_chunk}; retrying eval_chunk={next_chunk}",
                    flush=True,
                )
                eval_chunk = int(next_chunk)
        steps += 1
        if max_steps > 0 and steps >= int(max_steps):
            break

    if n <= 0:
        return {
            "loss": 0.0,
            "mask_bce": 0.0,
            "macro_f1": 0.0,
            "micro_f1": 0.0,
            "bit_acc": 0.0,
            "mean_confidence": 0.0,
            "num_samples": 0,
        }

    logits_cat = torch.cat(logits_all, dim=0)
    targets_cat = torch.cat(targets_all, dim=0)
    probs = torch.sigmoid(logits_cat)
    preds = (probs >= 0.5).float()
    t = targets_cat.float()

    tp = (preds * t).sum(dim=0)
    fp = (preds * (1.0 - t)).sum(dim=0)
    fn = ((1.0 - preds) * t).sum(dim=0)
    macro_f1 = torch.mean((2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)).item()
    tp_m = tp.sum()
    fp_m = fp.sum()
    fn_m = fn.sum()
    micro_f1 = ((2.0 * tp_m) / (2.0 * tp_m + fp_m + fn_m + 1e-8)).item()
    bit_acc = preds.eq(t).float().mean().item()
    mean_conf = torch.maximum(probs, 1.0 - probs).mean().item()
    return {
        "loss": total_loss / max(1, n),
        "mask_bce": mask_loss_total / max(1, n),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "bit_acc": float(bit_acc),
        "mean_confidence": float(mean_conf),
        "num_samples": int(n),
    }
