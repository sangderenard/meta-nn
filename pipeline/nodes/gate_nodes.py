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
  PregestationEvalNode  — Gate 0: pregestation validation loss on held-out logic rows
  GestationEvalNode     — Gate 1: gestation validation loss on held-out symbol rows
  BerkeleyGateNode      — Gate 2: Berkeley confidence + macro-F1 after Stage 2 training
  TransformerGateNode   — Gate R: transformer feature score + entropy
  GeneratorGateNode     — Gate G: generator feature score quality
  WaveGateNode          — Gate W: wave entropy + feedback score
"""
from __future__ import annotations

import copy
import json
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pipeline.context import PipelineContext
from pipeline.graph import PipelineNode
from pipeline.network_api import coerce_network_output
from pipeline.nodes.base import (
    GatedNode,
    autocast_context,
    cuda_supports_dtype,
    gpu_resident,
    resolve_amp_dtype,
    _save_pipeline_checkpoint,
)
from pipeline.nodes.save_restore_node import _classifier_checkpoint_metadata
from pipeline.nodes.data_nodes import (
    ensure_runtime_loader_contract,
    _expand_semantic_mask_supervision_batch,
    _forward_classifier_outputs_require_mask,
    _semantic_mask_bce_loss,
    _unpack_masked_semantic_batch,
)
from pipeline.nodes.speculative_node import _build_speculative_batch, _slice_speculative_batch_to_device
from pipeline.utils import (
    _classifier_supervision_loss,
    _is_cuda_backend_engine_error,
    _snapshot_classifier_lora,
)
from wav_ml_core import RenderConfig

# Module-level constants (originally in wav_config_transformer_pipeline)
CLASSIFIER_LOSS_SCALE = 1.0
CLASSIFIER_SEMANTIC_COSINE_WEIGHT = 0.35


def _gate_eval_model(ctx: PipelineContext, *, channels_last: bool = False) -> tuple[nn.Module, torch.device, str]:
    """Return the active recognition network for gate evaluation."""
    del channels_last
    model = getattr(ctx, "active_network", None)
    if model is None:
        raise RuntimeError("Gate evaluation requires an active_network")
    try:
        gate_device = next(model.parameters()).device
    except StopIteration:
        gate_device = torch.device("cpu")
    return model, gate_device, "active_network"


def _assigned_slot_confidence_probs(
    assignments: Sequence[Dict[str, Any]],
    slot_confidence: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """Project slot confidence onto vocab indices via slot→target assignments."""

    probs = torch.zeros(
        int(slot_confidence.shape[0]),
        int(vocab_size),
        device=slot_confidence.device,
        dtype=slot_confidence.dtype,
    )
    for b, sample in enumerate(list(assignments or [])):
        slot_to_target = list(sample.get("slot_to_target") or [])
        for slot_i, target_i in enumerate(slot_to_target):
            if target_i is None:
                continue
            ti = int(target_i)
            if 0 <= ti < int(vocab_size) and 0 <= int(slot_i) < int(slot_confidence.shape[1]):
                probs[b, ti] = torch.maximum(probs[b, ti], slot_confidence[b, int(slot_i)])
    return probs


def _evaluate_active_network_gate(
    *,
    ctx: PipelineContext,
    loader: DataLoader,
    device: torch.device,
    max_steps: int,
) -> Dict[str, Any]:
    model = getattr(ctx, "active_network", None)
    criterion = getattr(ctx, "active_network_criterion", None)
    if model is None or criterion is None:
        raise RuntimeError("Gate evaluation requires active_network and active_network_criterion")

    vocab_phrases = list(getattr(ctx, "class_names", []) or [])
    active_term_to_idx = dict(getattr(ctx, "semantic_term_to_idx", {}) or {})
    vocab_matrix = getattr(criterion, "vocab_matrix", None)
    if vocab_matrix is None:
        raise RuntimeError("Gate evaluation requires criterion.vocab_matrix")

    runtime_amp_enabled = bool(getattr(model, "_runtime_amp_enabled", False))
    amp_dtype = str(getattr(model, "_runtime_amp_dtype", "fp16") or "fp16")
    amp_dtype_t = resolve_amp_dtype(amp_dtype) if runtime_amp_enabled else torch.float16
    if runtime_amp_enabled and not cuda_supports_dtype(device=device, dtype=amp_dtype_t):
        print(
            f"[gate-eval] disabling AMP on device={device} because amp_dtype={amp_dtype} is unsupported there",
            flush=True,
        )
        runtime_amp_enabled = False
    if runtime_amp_enabled:
        for _p in model.parameters():
            if bool(_p.is_floating_point()):
                if _p.dtype == torch.float64:
                    print("[gate-eval] disabling AMP because active-network parameters are float64", flush=True)
                    runtime_amp_enabled = False
                break
    runtime_channels_last = bool(getattr(model, "_runtime_channels_last", False))
    eval_chunk_cap = int(getattr(model, "_runtime_microbatch_cap", 0) or 0)
    runtime_tensor_dtype: Optional[torch.dtype] = None
    for _p in model.parameters():
        if bool(_p.is_floating_point()):
            runtime_tensor_dtype = _p.dtype
            break

    model.eval()
    total_loss = 0.0
    mask_loss_total = 0.0
    n = 0
    steps = 0
    probs_all: List[torch.Tensor] = []
    targets_all: List[torch.Tensor] = []

    def _is_cuda_oom(exc: BaseException) -> bool:
        txt = str(exc).lower()
        if "out of memory" not in txt:
            return False
        return ("cuda" in txt) or ("cudnn" in txt) or ("cublas" in txt)

    for batch in loader:
        xb, mb, batch_meta = _unpack_masked_semantic_batch(batch, context="active-network gate evaluation")
        spec = _build_speculative_batch(
            xb,
            mb,
            batch_meta,
            vocab_phrases=vocab_phrases,
            criterion_vocab_matrix=vocab_matrix,
            active_term_to_idx=active_term_to_idx,
            device=torch.device("cpu"),
        )
        if spec is None:
            continue
        batch_n = int(spec["image"].shape[0])
        eval_chunk = int(eval_chunk_cap) if int(eval_chunk_cap) > 0 else int(batch_n)
        eval_chunk = max(1, min(int(eval_chunk), int(batch_n)))

        while True:
            try:
                batch_loss = 0.0
                batch_mask_loss = 0.0
                batch_seen = 0
                batch_probs: List[torch.Tensor] = []
                batch_targets: List[torch.Tensor] = []
                for start in range(0, int(batch_n), int(eval_chunk)):
                    stop = min(int(batch_n), int(start + eval_chunk))
                    spec_part = _slice_speculative_batch_to_device(
                        spec,
                        start=start,
                        stop=stop,
                        device=device,
                        tensor_dtype=runtime_tensor_dtype,
                        channels_last=runtime_channels_last,
                    )
                    with torch.no_grad():
                        with autocast_context(device=device, enabled=runtime_amp_enabled, amp_dtype=amp_dtype_t):
                            raw_out = (
                                model.forward_batch(spec_part["image"], spec_part["batch_vocab"])
                                if hasattr(model, "forward_batch") and callable(getattr(model, "forward_batch"))
                                else model(spec_part["image"], spec_part["batch_vocab"])
                            )
                            out = coerce_network_output(raw_out)
                            aux = out.aux if isinstance(out.aux, dict) else {}
                            loss_dict = criterion(
                                pred_vectors=out.slot_vectors,
                                pred_masks=out.slot_masks,
                                target_vectors=spec_part["target_vectors"],
                                target_masks=spec_part["target_masks"],
                                target_valid=spec_part["target_valid"],
                                present_mask=spec_part["present_mask"],
                                confidence_logits=out.slot_confidence,
                                slot_selection_logits=aux.get("slot_selection_logits"),
                            )

                    part_n = int(stop - start)
                    batch_loss += float(loss_dict["loss"].item()) * part_n
                    _mask_loss = loss_dict.get("mask_loss", 0.0)
                    if isinstance(_mask_loss, torch.Tensor):
                        _mask_loss = float(_mask_loss.item())
                    batch_mask_loss += float(_mask_loss) * part_n
                    batch_seen += part_n

                    assignments = loss_dict.get("assignments") or out.assignments or []
                    sample_probs = _assigned_slot_confidence_probs(
                        assignments=assignments,
                        slot_confidence=out.slot_confidence.sigmoid(),
                        vocab_size=len(vocab_phrases),
                    )
                    batch_probs.append(sample_probs.detach().cpu())
                    batch_targets.append(spec_part["present_mask"].detach().to(torch.float32).cpu())
                    del spec_part, raw_out, out, loss_dict

                total_loss += float(batch_loss)
                mask_loss_total += float(batch_mask_loss)
                n += int(batch_seen)
                probs_all.extend(batch_probs)
                targets_all.extend(batch_targets)
                break
            except RuntimeError as exc:
                if device.type == "cuda" and _is_cuda_backend_engine_error(exc):
                    if bool(runtime_amp_enabled):
                        runtime_amp_enabled = False
                        setattr(model, "_runtime_amp_enabled", False)
                        torch.cuda.empty_cache()
                        print("[gate-eval] backend engine selection failed; retrying with AMP disabled", flush=True)
                        continue
                    if bool(runtime_channels_last):
                        runtime_channels_last = False
                        setattr(model, "_runtime_channels_last", False)
                        model = model.to(memory_format=torch.contiguous_format)
                        xb = xb.contiguous()
                        torch.cuda.empty_cache()
                        print("[gate-eval] backend engine selection failed; retrying with contiguous tensors", flush=True)
                        continue
                if device.type != "cuda" or (not _is_cuda_oom(exc)) or int(eval_chunk) <= 1:
                    raise
                next_chunk = max(1, int(eval_chunk) // 2)
                if int(next_chunk) > 1:
                    next_chunk = 1 << (int(next_chunk).bit_length() - 1)
                if bool(runtime_channels_last):
                    runtime_channels_last = False
                    setattr(model, "_runtime_channels_last", False)
                    model = model.to(memory_format=torch.contiguous_format)
                torch.cuda.empty_cache()
                print(
                    f"[gate-eval] CUDA OOM at eval_chunk={eval_chunk}; retrying eval_chunk={next_chunk}",
                    flush=True,
                )
                eval_chunk = int(next_chunk)

        steps += 1
        if steps == 1 or steps % 10 == 0:
            running_loss = total_loss / max(1, n)
            limit_str = f"/{max_steps}" if max_steps > 0 else ""
            print(f"[gate-eval] step={steps}{limit_str} samples={n} loss={running_loss:.4f}", flush=True)
        del xb, mb, spec
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

    probs = torch.cat(probs_all, dim=0)
    targets = torch.cat(targets_all, dim=0).float()
    preds = (probs >= 0.5).float()
    tp = (preds * targets).sum(dim=0)
    fp = (preds * (1.0 - targets)).sum(dim=0)
    fn = ((1.0 - preds) * targets).sum(dim=0)
    macro_f1 = torch.mean((2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)).item()
    tp_m = tp.sum()
    fp_m = fp.sum()
    fn_m = fn.sum()
    micro_f1 = ((2.0 * tp_m) / (2.0 * tp_m + fp_m + fn_m + 1e-8)).item()
    bit_acc = preds.eq(targets).float().mean().item()
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


def _evaluate_loss_gate(
    *,
    ctx: PipelineContext,
    loader: Any,
    max_steps: int = 0,
    channels_last: bool = False,
    semantic_cosine_weight: float = CLASSIFIER_SEMANTIC_COSINE_WEIGHT,
) -> Dict[str, Any]:
    del semantic_cosine_weight
    gate_model, gate_device, gate_name = _gate_eval_model(ctx, channels_last=channels_last)
    with gpu_resident(ctx, [(gate_model, gate_name)], device=gate_device) if gate_device.type == "cuda" else nullcontext():
        return _evaluate_active_network_gate(
            ctx=ctx,
            loader=loader,
            device=gate_device,
            max_steps=max(0, int(max_steps)),
        )



class PregestationEvalNode(PipelineNode):
    """Gate 0: Evaluate held-out Stage-0 rows on the active network."""

    node_id = "gate_0_pregestation_eval"
    description = "Gate 0 Eval: pre-gestation validation loss"
    runtime_object_type = "evaluator"
    runtime_faculty = "gate"
    gpu_models = ["active_network"]

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    def should_run(self, ctx: PipelineContext) -> bool:
        return ctx.active_network is not None and ctx.active_network_criterion is not None

    def execute(self, ctx: PipelineContext) -> None:
        loader_info = ensure_runtime_loader_contract(ctx, consumer_id=self.node_id)
        max_steps = int(getattr(ctx.args, "gate_pregestation_eval_max_steps", 10) or 10)
        result = _evaluate_loss_gate(
            ctx=ctx,
            loader=loader_info["loader"],
            max_steps=max_steps,
            channels_last=False,
        )
        loss = float(result.get("loss", float("inf")))
        ctx.gate_pregestation.required_consecutive = int(getattr(self.cfg, "stage0_required_consecutive", 1))
        ctx.gate_pregestation.record(
            round_id=ctx.round_id,
            metric=loss,
            threshold=float(getattr(self.cfg, "stage0_loss_target", 0.0)),
            above=False,
        )
        ctx.log_metric("gate0", "loss", loss)
        _log(
            f"[gate0] loss={loss:.4f}/{float(getattr(self.cfg, 'stage0_loss_target', 0.0)):.4f} "
            f"consecutive={ctx.gate_pregestation.consecutive_passes}/{int(getattr(self.cfg, 'stage0_required_consecutive', 1))} "
            f"{'PASS' if ctx.gate_pregestation.passed else 'hold'}"
        )


class GestationEvalNode(GatedNode):
    """Gate 1: Evaluate held-out gestation rows on the active network."""

    node_id = "gate_1_gestation_eval"
    description = "Gate 1 Eval: gestation validation loss"
    runtime_object_type = "evaluator"
    runtime_faculty = "gate"
    required_gates = ["gate_pregestation"]
    gpu_models = ["active_network"]

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    def should_run(self, ctx: PipelineContext) -> bool:
        if ctx.is_gate_bypassed("gate_gestation"):
            return False
        if not super().should_run(ctx):
            return False
        return ctx.active_network is not None and ctx.active_network_criterion is not None

    def execute(self, ctx: PipelineContext) -> None:
        loader_info = ensure_runtime_loader_contract(ctx, consumer_id=self.node_id)
        max_steps = int(getattr(ctx.args, "gate_gestation_eval_max_steps", 10) or 10)
        result = _evaluate_loss_gate(
            ctx=ctx,
            loader=loader_info["loader"],
            max_steps=max_steps,
            channels_last=False,
        )
        loss = float(result.get("loss", float("inf")))
        ctx.gate_gestation.required_consecutive = int(getattr(self.cfg, "stage1_required_consecutive", 1))
        ctx.gate_gestation.record(
            round_id=ctx.round_id,
            metric=loss,
            threshold=float(getattr(self.cfg, "stage1_loss_target", 0.0)),
            above=False,
        )
        ctx.log_metric("gate1", "loss", loss)
        _log(
            f"[gate1] loss={loss:.4f}/{float(getattr(self.cfg, 'stage1_loss_target', 0.0)):.4f} "
            f"consecutive={ctx.gate_gestation.consecutive_passes}/{int(getattr(self.cfg, 'stage1_required_consecutive', 1))} "
            f"{'PASS' if ctx.gate_gestation.passed else 'hold'}"
        )


# ---------------------------------------------------------------------------
# Berkeley gate  (Gate 2)
# ---------------------------------------------------------------------------

@dataclass
class BerkeleyGateConfig:
    """Thresholds for the Berkeley SBD active-network gate."""

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
    """Gate 2: Evaluate active-network confidence and F1 on Berkeley validation set.

    Requires Gate 0 + Gate 1.  The gate passes when both confidence and
    macro-F1 exceed their targets for the required consecutive rounds.
    """

    node_id = "gate_berkeley"
    description = "Gate 2 Eval: active-network confidence + macro-F1"
    runtime_object_type = "evaluator"
    runtime_faculty = "gate"
    required_gates = ["gate_pregestation", "gate_gestation"]
    gpu_models = ["active_network"]

    def __init__(self, cfg: BerkeleyGateConfig, classifier_cfg: Any = None) -> None:
        self.cfg = cfg
        self.classifier_cfg = classifier_cfg

    def should_run(self, ctx: PipelineContext) -> bool:
        if not super().should_run(ctx):
            return False
        return ctx.active_network is not None and ctx.active_network_criterion is not None

    def execute(self, ctx: PipelineContext) -> None:
        loader_info = ensure_runtime_loader_contract(ctx, consumer_id=self.node_id)
        gate_model, gate_device, gate_name = _gate_eval_model(ctx, channels_last=False)
        with gpu_resident(ctx, [(gate_model, gate_name)], device=gate_device) if gate_device.type == "cuda" else nullcontext():
            result = _evaluate_active_network_gate(
                ctx=ctx,
                loader=loader_info["loader"],
                device=gate_device,
                max_steps=int(getattr(ctx.args, "gate_berkeley_eval_max_steps", 10) or 10),
            )

        confidence = float(result.get("mean_confidence", 0.0))
        macro_f1 = float(result.get("macro_f1", 0.0))
        loss = float(result.get("loss", float("inf")))

        conf_norm = min(1.0, confidence / max(1e-6, self.cfg.confidence_target))
        f1_norm = min(1.0, macro_f1 / max(1e-6, self.cfg.f1_target))
        combined = 2.0 * conf_norm * f1_norm / max(1e-6, conf_norm + f1_norm)
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

    # Minimum feature score required to pass gate R.
    score_target: float = 0.60

    # Feature score threshold (higher is better)
    feature_score_min: float = 0.50
    # Training entropy-excess threshold (lower is better). Negative disables it.
    max_trainer_entropy_excess: float = -1.0

    required_consecutive: int = 3

    # Evaluation parameters
    image_size: int = 128
    patch_size: int = 16
    chunk_samples: int = 0        # 0 = auto-derive
    eval_max_batches: int = 10
    channels_last: bool = False


class TransformerGateNode(GatedNode):
    """Gate R: Evaluate transformer output quality via feature score and trainer entropy excess.

    Requires Gate 0 + Gate 1.  Scores the transformer's current renderings
    and checks both the combined metric and the individual component thresholds.
    """

    node_id = "gate_transformer"
    description = "Gate R Eval: transformer feature score + trainer entropy excess"
    runtime_object_type = "evaluator"
    runtime_faculty = "gate"
    required_gates = ["gate_pregestation", "gate_gestation"]
    gpu_models = ["classifier"]

    def __init__(self, cfg: TransformerGateConfig) -> None:
        self.cfg = cfg

    def should_run(self, ctx: PipelineContext) -> bool:
        if not super().should_run(ctx):
            return False
        return ctx.transformer is not None and ctx.classifier is not None

    def execute(self, ctx: PipelineContext) -> None:
        from wav_ml_models import evaluate_feature_score_before_after
        from pipeline.utils import _resolve_synced_chunk_samples

        def _latest_stage_r_trainer_entropy_excess() -> float:
            rows = list(getattr(ctx, "metrics_history", []) or [])
            for row in reversed(rows):
                if str(row.get("stage", "")) != "stageR":
                    continue
                if str(row.get("key", "")) != "trainer_entropy_excess":
                    continue
                try:
                    return float(row.get("value", float("nan")))
                except Exception:
                    return float("nan")
            return float("nan")

        image_size = int(self.cfg.image_size)
        requested_chunks = int(self.cfg.chunk_samples or getattr(ctx.args, "chunk_samples", 0) or 1)
        if ctx.render_config is not None:
            chunk_samples, _ = _resolve_synced_chunk_samples(
                requested_chunk_samples=requested_chunks,
                patch_size=int(self.cfg.patch_size),
                cfg=ctx.render_config,
                image_hw=(image_size, image_size),
            )
        else:
            chunk_samples = max(1, requested_chunks)

        result = evaluate_feature_score_before_after(
            transformer=ctx.transformer,
            classifier=ctx.classifier,
            streams=ctx.float_streams,
            cfg=ctx.render_config,
            sample_bits=int(getattr(ctx.args, "sample_bits", 16) or 16),
            image_hw=(image_size, image_size),
            chunk_samples=int(chunk_samples),
            device=ctx.device,
            max_batches=int(self.cfg.eval_max_batches),
            amp=bool(ctx.amp_enabled),
            amp_dtype=str(ctx.amp_dtype or "float16"),
            channels_last=bool(self.cfg.channels_last),
        )

        feature_score = float(result.get("score_after", 0.0))
        trainer_entropy_excess = _latest_stage_r_trainer_entropy_excess()
        combined = float(feature_score)

        required_feature = max(float(self.cfg.score_target), float(self.cfg.feature_score_min))
        entropy_gate_enabled = float(self.cfg.max_trainer_entropy_excess) >= 0.0
        entropy_ok = (not entropy_gate_enabled) or (
            trainer_entropy_excess <= float(self.cfg.max_trainer_entropy_excess)
        )
        passes = (
            feature_score >= required_feature
            and entropy_ok
        )

        ctx.gate_transformer.required_consecutive = self.cfg.required_consecutive
        threshold_metric = feature_score if passes else 0.0
        ctx.gate_transformer.record(
            round_id=ctx.round_id,
            metric=threshold_metric,
            threshold=required_feature * 0.99,
            above=True,
        )

        ctx.log_metric("gateR", "feature_score", feature_score)
        if trainer_entropy_excess == trainer_entropy_excess:
            ctx.log_metric("gateR", "trainer_entropy_excess", trainer_entropy_excess)

        _log(
            f"[gateR] score={feature_score:.3f} trainer_entropy_excess={trainer_entropy_excess:.4f} "
            f"combined={combined:.3f} req_score={required_feature:.3f} "
            f"max_trainer_entropy_excess={self.cfg.max_trainer_entropy_excess:.4f} "
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
    description = "Gate G Eval: generator feature score"
    runtime_object_type = "evaluator"
    runtime_faculty = "gate"
    required_gates = ["gate_pregestation", "gate_gestation", "gate_berkeley"]
    gpu_models = ["generator", "classifier"]

    def __init__(self, cfg: GeneratorGateConfig, generator_cfg: Any = None) -> None:
        self.cfg = cfg
        self.generator_cfg = generator_cfg

    def should_run(self, ctx: PipelineContext) -> bool:
        if not super().should_run(ctx):
            return False
        return ctx.generator is not None and ctx.classifier is not None

    def execute(self, ctx: PipelineContext) -> None:
        from wav_ml_models import evaluate_conditional_generator

        g_cfg = self.generator_cfg
        image_size = int(getattr(g_cfg, "image_size", None) or getattr(ctx.args, "image_size", 128))
        z_dim = int(getattr(g_cfg, "z_dim", None) or getattr(ctx.args, "generator_z_dim", 128))
        active_term_to_idx = dict(ctx.semantic_term_to_idx)
        payload_loader = None
        _payload_bank = getattr(ctx, "payload_bank", None)
        _payload_bank_obj = getattr(_payload_bank, "bank", None)
        if _payload_bank_obj is not None and ctx.payload_conditions:
            from pipeline.nodes.data_nodes import build_stage_loaders
            from pipeline.semantic_wheel_cache import SemanticWheelPayloadDataset
            eval_ds = SemanticWheelPayloadDataset(
                bank=_payload_bank_obj,
                image_hw=(image_size, image_size),
            )
            payload_loader, _ = build_stage_loaders(
                dataset=eval_ds,
                name="generator_gate_eval",
                batch_size=32,
                num_workers=0,
                seed=int(getattr(ctx.args, "seed", 42) or 42),
                device_type=str(ctx.device.type),
                pin_memory=(ctx.device.type == "cuda"),
                prefetch_factor=2,
                shuffle_train=True,
            )

        result = evaluate_conditional_generator(
            generator=ctx.generator,
            classifier=ctx.classifier,
            payload_conditions=ctx.payload_conditions,
            payload_masks=ctx.payload_masks if ctx.payload_masks else [],
            num_classes=len(ctx.class_names) if ctx.class_names else 1,
            image_hw=(image_size, image_size),
            z_dim=z_dim,
            device=ctx.device,
            steps=1,
            batch_size=32,
            payload_loader=payload_loader,
            active_term_to_idx=active_term_to_idx,
        )

        feat_score = float(result.get("feature_score", result.get("target_prob", 0.0)))

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

    # Evaluation parameters
    image_size: int = 128
    patch_size: int = 16
    chunk_samples: int = 0        # 0 = auto-derive
    eval_max_batches: int = 10
    channels_last: bool = False


class WaveGateNode(GatedNode):
    """Gate W: Combined wave entropy + classifier feature score gate.

    Requires early gates + transformer gate.  Checks the wave-level feedback
    metric; used to conditionally unlock the wave classifier stage.
    """

    node_id = "gate_wave"
    description = "Gate W Eval: wave entropy + feature score feedback"
    runtime_object_type = "evaluator"
    runtime_faculty = "gate"
    required_gates = ["gate_pregestation", "gate_gestation", "gate_transformer"]
    gpu_models = ["classifier"]

    def __init__(self, cfg: WaveGateConfig) -> None:
        self.cfg = cfg

    def should_run(self, ctx: PipelineContext) -> bool:
        if not super().should_run(ctx):
            return False
        mode = str(ctx.orchestration_mode or getattr(ctx.args, "orchestration_mode", "")).lower()
        return "w" in mode and ctx.classifier is not None

    def execute(self, ctx: PipelineContext) -> None:
        from wav_ml_models import evaluate_feature_score_before_after
        from pipeline.utils import _resolve_synced_chunk_samples

        image_size = int(self.cfg.image_size)
        requested_chunks = int(self.cfg.chunk_samples or getattr(ctx.args, "chunk_samples", 0) or 1)
        if ctx.render_config is not None:
            chunk_samples, _ = _resolve_synced_chunk_samples(
                requested_chunk_samples=requested_chunks,
                patch_size=int(self.cfg.patch_size),
                cfg=ctx.render_config,
                image_hw=(image_size, image_size),
            )
        else:
            chunk_samples = max(1, requested_chunks)

        result = evaluate_feature_score_before_after(
            transformer=ctx.transformer,
            classifier=ctx.classifier,
            streams=ctx.float_streams,
            cfg=ctx.render_config,
            sample_bits=int(getattr(ctx.args, "sample_bits", 16) or 16),
            image_hw=(image_size, image_size),
            chunk_samples=int(chunk_samples),
            device=ctx.device,
            max_batches=int(self.cfg.eval_max_batches),
            amp=bool(ctx.amp_enabled),
            amp_dtype=str(ctx.amp_dtype or "float16"),
            channels_last=bool(self.cfg.channels_last),
        )

        feature_score = float(result.get("score_after", 0.0))
        entropy = float(result.get("hard_coverage_after", 0.0))
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
    runtime_object_type = "service"
    runtime_faculty = "persistence"

    def __init__(self, save_every_n_rounds: int = 1) -> None:
        self.save_every_n_rounds = max(1, int(save_every_n_rounds))

    @property
    def runtime_execution_policy(self) -> tuple:
        return ("periodic", {"period": self.save_every_n_rounds, "counter": "round_id"})

    def should_run(self, ctx: PipelineContext) -> bool:
        return (ctx.round_id % self.save_every_n_rounds) == 0

    def execute(self, ctx: PipelineContext) -> None:

        _save_training_segment_snapshot(
            enabled=True,
            out_dir=ctx.output_dir,
            objective_mode=getattr(ctx.args, "objective_mode", "berkeley_multilabel"),
            run_tag=ctx.run_tag,
            segment="checkpoint",
            best_cfg=ctx.render_config,
            classifier=ctx.classifier,
            transformer=ctx.transformer,
            generator=ctx.generator,
            discriminator=ctx.discriminator,
            wave_classifier=ctx.wave_classifier,
            orchestration_history=ctx.metrics_history,
            gate_status={
                "pregestation": ctx.gate_pregestation.passed,
                "gestation": ctx.gate_gestation.passed,
                "berkeley": ctx.gate_berkeley.passed,
                "transformer": ctx.gate_transformer.passed,
                "generator": ctx.gate_generator.passed,
                "wave": ctx.gate_wave.passed,
            },
            extra={
                "cycle": ctx.cycle,
                "round_id": ctx.round_id,
                "class_names": ctx.class_names,
            },
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
                **_classifier_checkpoint_metadata(classifier, ctx),
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
    active_term_to_idx: Optional[Dict[str, int]] = None,
    n_active_classes: int = 0,
):
    from pipeline.nodes.classifier_node import _yb_from_terms

    classifier.eval()
    amp_dtype_t = resolve_amp_dtype(amp_dtype) if amp_enabled else torch.float16
    runtime_amp_enabled = bool(amp_enabled and device.type == "cuda")
    runtime_channels_last = bool(channels_last)
    if bool(runtime_amp_enabled) and not cuda_supports_dtype(device=device, dtype=amp_dtype_t):
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
        xb, mb, batch_meta = _unpack_masked_semantic_batch(batch, context="berkeley gate evaluation")
        _tti = active_term_to_idx or {}
        _nac = int(n_active_classes) if int(n_active_classes) > 0 else len(_tti)
        _terms = list(batch_meta.get("terms_rows") or [[] for _ in range(int(xb.shape[0]))])
        yb = _yb_from_terms(_terms, _tti, _nac, xb.device)
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
                        with autocast_context(device=device, enabled=runtime_amp_enabled, amp_dtype=amp_dtype_t):
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
        if steps == 1 or steps % 10 == 0:
            running_loss = total_loss / max(1, n)
            limit_str = f"/{max_steps}" if max_steps > 0 else ""
            print(f"[gate-eval] step={steps}{limit_str} samples={n} loss={running_loss:.4f}", flush=True)
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
