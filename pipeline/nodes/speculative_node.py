"""
Speculative Network Node — pipeline integration for PrototypeAutoClassifier
(or any NetworkContract implementation).

Replaces the TinyConvClassifier training stages.  The training loop speaks
only the NetworkContract API; any conforming network drops in without changes
to this file.

Nodes exported
--------------
BuildSpeculativeNetNode        — instantiate PrototypeAutoClassifier, store on ctx
SpeculativePregestationNode    — stage-0 training on pregestation loader
SpeculativeGestationNode       — stage-1 training on gestation loader
SpeculativeBerkeleyNode        — stage-2 training on Berkeley loader

Helper functions
----------------
_build_speculative_batch       — convert standard (xb, mb, meta) loader tuple
                                 into the vocab-conditioned format the network needs
_run_speculative_net_epochs    — generic training loop for any stage
"""
from __future__ import annotations

import gc
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pipeline.context import PipelineContext
from pipeline.graph import PipelineNode
from pipeline.network_api import coerce_network_output
from pipeline.nodes.base import (
    GatedNode,
    IRLossTermSpec,
    IRStateSpec,
    IRTensorPortSpec,
    IRTrainingNode,
    _save_pipeline_checkpoint,
    autocast_context,
    cuda_supports_dtype,
    make_grad_scaler,
    make_training_progress_callback,
    make_runtime_weight_publish_callback,
    resolve_amp_dtype,
    StageSkipForward,
    StageSkipBack,
    StageStopRequested,
)
from pipeline.nodes.data_nodes import (
    _apply_network_dropout_rate,
    ensure_runtime_loader_contract,
    _unpack_masked_semantic_batch,
)
from pipeline.preview import make_speculative_step_preview_callback
from pipeline.utils import _is_cuda_backend_engine_error


# Global precision for the speculative network.  Change this one value to
# switch between float32 and float64 across both training and gate evaluation.
SPECULATIVE_NETWORK_DTYPE: str = "float64"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _tensor_nonfinite_stats(name: str, value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, torch.Tensor):
        return None
    mask = ~torch.isfinite(value.detach())
    nonfinite = int(mask.sum().item())
    if nonfinite <= 0:
        return None
    stats: Dict[str, Any] = {
        "name": str(name),
        "shape": tuple(int(x) for x in value.shape),
        "dtype": str(value.dtype),
        "nonfinite": int(nonfinite),
    }
    if int(value.ndim) >= 1 and int(value.shape[0]) > 0:
        try:
            per_sample = mask.reshape(int(value.shape[0]), -1).sum(dim=1).tolist()
            stats["per_sample"] = [int(x) for x in per_sample[:8]]
        except Exception:
            pass
    finite_vals = value.detach()[~mask]
    if int(finite_vals.numel()) > 0:
        try:
            stats["finite_min"] = float(finite_vals.min().item())
            stats["finite_max"] = float(finite_vals.max().item())
        except Exception:
            pass
    return stats


def _format_nonfinite_stats(stats: Sequence[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for item in list(stats or []):
        name = str(item.get("name", "?"))
        shape = tuple(item.get("shape") or ())
        nonfinite = int(item.get("nonfinite", 0))
        per_sample = item.get("per_sample")
        finite_min = item.get("finite_min")
        finite_max = item.get("finite_max")
        token = f"{name} shape={shape} nonfinite={nonfinite}"
        if isinstance(per_sample, list) and per_sample:
            token += f" per_sample={per_sample}"
        if finite_min is not None and finite_max is not None:
            token += f" finite=[{float(finite_min):.4g},{float(finite_max):.4g}]"
        parts.append(token)
    return "; ".join(parts)


def _collect_nonfinite_tensor_stats(named_values: Sequence[tuple[str, Any]]) -> List[Dict[str, Any]]:
    stats: List[Dict[str, Any]] = []
    for name, value in list(named_values or []):
        item = _tensor_nonfinite_stats(str(name), value)
        if isinstance(item, dict):
            stats.append(item)
    return stats


def _format_terms_rows_brief(
    terms_rows: Sequence[Sequence[str]],
    *,
    sample_offset: int = 0,
    max_rows: int = 4,
    max_terms: int = 6,
) -> str:
    rows = list(terms_rows or [])
    if not rows:
        return "samples=[]"
    parts: List[str] = []
    limit = max(1, int(max_rows))
    term_limit = max(1, int(max_terms))
    for local_idx, row in enumerate(rows[:limit]):
        terms = [str(x).strip() for x in list(row or []) if str(x).strip()]
        if int(len(terms)) > int(term_limit):
            terms = terms[:term_limit] + ["..."]
        label = "+".join(terms) if terms else "none"
        parts.append(f"{int(sample_offset) + int(local_idx)}:{label}")
    if int(len(rows)) > int(limit):
        parts.append("...")
    return "samples=[" + ", ".join(parts) + "]"


def _format_nonfinite_terms_context(
    stats: Sequence[Dict[str, Any]],
    terms_rows: Sequence[Sequence[str]],
    *,
    sample_offset: int = 0,
    max_rows: int = 4,
    max_terms: int = 6,
) -> str:
    rows = list(terms_rows or [])
    flagged: List[int] = []
    seen: set = set()
    for item in list(stats or []):
        per_sample = item.get("per_sample")
        if not isinstance(per_sample, list):
            continue
        for local_idx, count in enumerate(per_sample):
            if int(count) <= 0 or local_idx in seen:
                continue
            seen.add(local_idx)
            flagged.append(int(local_idx))
            if int(len(flagged)) >= int(max_rows):
                break
        if int(len(flagged)) >= int(max_rows):
            break
    if not flagged:
        return _format_terms_rows_brief(
            rows,
            sample_offset=sample_offset,
            max_rows=max_rows,
            max_terms=max_terms,
        )
    parts: List[str] = []
    term_limit = max(1, int(max_terms))
    for local_idx in flagged[: max(1, int(max_rows))]:
        row = rows[local_idx] if 0 <= int(local_idx) < int(len(rows)) else []
        terms = [str(x).strip() for x in list(row or []) if str(x).strip()]
        if int(len(terms)) > int(term_limit):
            terms = terms[:term_limit] + ["..."]
        label = "+".join(terms) if terms else "none"
        parts.append(f"{int(sample_offset) + int(local_idx)}:{label}")
    return "samples=[" + ", ".join(parts) + "]"


def _raise_nonfinite_error(
    *,
    stage_label: str,
    reason: str,
    stats: Sequence[Dict[str, Any]],
    terms_rows: Sequence[Sequence[str]],
    sample_offset: int = 0,
    extra: Optional[str] = None,
) -> None:
    message = f"[{stage_label}] {str(reason)}: {_format_nonfinite_stats(stats)}"
    context = _format_nonfinite_terms_context(
        stats,
        terms_rows,
        sample_offset=sample_offset,
    )
    if str(context).strip():
        message += f" | {context}"
    if extra:
        message += f" | {str(extra)}"
    raise _RejectSpeculativeBatch(message)


def _collect_nonfinite_param_names(module: nn.Module, *, grads: bool = False, max_items: int = 8) -> List[str]:
    out: List[str] = []
    for name, param in module.named_parameters():
        tensor = param.grad if grads else param.data
        if tensor is None:
            continue
        try:
            count = int((~torch.isfinite(tensor.detach())).sum().item())
        except Exception:
            count = 0
        if count > 0:
            out.append(f"{str(name)}={count}")
            if int(len(out)) >= int(max_items):
                break
    return out


def _snapshot_param_grads(module: nn.Module) -> List[tuple[nn.Parameter, Optional[torch.Tensor]]]:
    snapshots: List[tuple[nn.Parameter, Optional[torch.Tensor]]] = []
    for param in module.parameters():
        if not bool(param.requires_grad):
            continue
        grad = getattr(param, "grad", None)
        snapshots.append((param, None if grad is None else grad.detach().clone()))
    return snapshots


def _restore_param_grads(snapshots: Sequence[tuple[nn.Parameter, Optional[torch.Tensor]]]) -> None:
    for param, grad in list(snapshots or []):
        if grad is None:
            param.grad = None
            continue
        current_grad = getattr(param, "grad", None)
        if current_grad is None:
            param.grad = grad.detach().clone()
        else:
            current_grad.detach().copy_(grad)


def _scale_param_grads_(module: nn.Module, scale: float) -> None:
    scale = float(scale)
    if scale == 1.0:
        return
    for param in module.parameters():
        grad = getattr(param, "grad", None)
        if grad is not None:
            grad.detach().mul_(scale)


class _RejectSpeculativeBatch(RuntimeError):
    """Reject a single training batch without aborting the whole stage."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SpeculativeNetConfig:
    """Every hyperparameter for PrototypeAutoClassifier and its training."""

    # ---- Architecture ---------------------------------------------------
    hidden_dim: int = 256
    n_slots: int = 8
    # image_size is used by the mask head's upsampling target; should match
    # the spatial resolution of the images arriving from the loaders.
    image_size: int = 128
    # 3 for the standard RGB pipeline loaders; 4 for RGBA standalone mode.
    in_channels: int = 3
    row_dim: int = 128           # trainable row dimension per vocab phrase

    # ---- Vocabulary / embedding -----------------------------------------
    sentence_transformer_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ---- Loss Control Surface -------------------------------------------
    # Each family has:
    #   * enable_*     hard ablation switch for quick experiments
    #   * *_weight     coefficient in the total objective
    # A weight <= 0.0 also effectively disables the term.
    #
    # Core reconstruction / assignment losses
    enable_vector_loss: bool = True
    vector_weight: float = 1.0      # slot→matched-target embedding distance
    enable_mask_loss: bool = True
    mask_weight: float = 0.7        # per-slot mask reconstruction BCE
    dustbin_cost: float = 0.8       # one-to-one matcher dustbin penalty

    # Selection / certainty losses
    enable_selection_loss: bool = True
    selection_weight: float = 0.65  # per-slot cross-entropy over present items
    selection_temp: float = 6.0     # calibrated logsumexp temperature (display only)
    enable_confidence_loss: bool = True
    confidence_weight: float = 0.25 # slot confidence (1=real, 0=dustbin)

    # Sequential ordering / canvas-discipline losses
    enable_mask_order_loss: bool = True
    mask_order_weight: float = 0.25
    mask_order_margin: float = 0.02
    mask_order_target_volume_weight: float = 1.0
    mask_order_target_spread_weight: float = 0.75
    mask_order_pred_volume_weight: float = 1.0
    mask_order_pred_spread_weight: float = 0.75
    enable_residual_mask_loss: bool = True
    residual_mask_weight: float = 0.20
    residual_mask_detach_canvas: bool = True
    mask_fn_weight: float = 2.0      # false-negative weight in mask BCE (>1 = missing content penalised harder)

    # Hypergraph / state-rollout losses and priors
    hypergraph_prior_weight: float = 0.35
    duplicate_penalty: float = 1.25
    hypergraph_alpha: float = 0.5
    predictive_hypergraph_momentum: float = 0.80

    # ---- Optimiser ------------------------------------------------------
    lr: float = 5e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    # 0 = disable accumulation; -1 = one optimiser step per full loader pass.
    # Positive values step once per N loader batches.
    grad_accum_steps: int = -1
    # Per-stage overrides; -1 falls back to grad_accum_steps.
    stage0_grad_accum_steps: int = -1
    stage1_grad_accum_steps: int = -1
    stage2_grad_accum_steps: int = 100
    network_dropout: float = 0.0

    # ---- Runtime / memory -----------------------------------------------
    network_dtype: str = SPECULATIVE_NETWORK_DTYPE
    amp: bool = False
    amp_dtype: str = "fp16"
    channels_last: bool = False
    max_forward_batch_cap: int = 0

    # ---- Per-stage epoch counts -----------------------------------------
    stage0_epochs: int = 3   # pregestation
    stage1_epochs: int = 5   # gestation
    stage2_epochs: int = 2   # Berkeley

    # ---- Progress logging -----------------------------------------------
    # 0 = silent; N = print one line every N optimiser steps
    log_every: int = 50

    # ---- Checkpoint on optimizer step -----------------------------------
    # When True, write active_network.pt to disk after every optimizer step.
    # Intended for extreme gradient accumulation (ultra-large batches) where
    # each step is expensive and losing one would be costly.
    save_on_grad_step: bool = False


# ---------------------------------------------------------------------------
# Batch conversion helper
# ---------------------------------------------------------------------------


def _term_key(term: str) -> str:
    import re
    return re.sub(r"\s+", " ", str(term)).strip().lower()


def _build_speculative_batch(
    xb: torch.Tensor,
    mb: torch.Tensor,
    meta: Dict[str, Any],
    vocab_phrases: List[str],
    criterion_vocab_matrix: torch.Tensor,   # [V, D] from criterion.vocab_matrix
    active_term_to_idx: Dict[str, int],
    device: torch.device,
) -> Optional[Dict[str, Any]]:
    """Convert a standard loader batch into speculative-network inputs.

    Returns None if the batch is unusable (e.g. zero-size vocab_matrix).

    Output keys
    -----------
    image          [B, C, H, W]
    batch_vocab    List[List[str]]  — identical rows = ctx.class_names
    target_vectors [B, V, D]        — ST embeddings from criterion (no extra encode)
    target_masks   [B, V, H, W]     — per-term spatial masks
    present_mask   [B, V]           — 1.0 where term is active in this sample
    target_valid   [B, V] bool
    """
    B = int(xb.shape[0])
    V = len(vocab_phrases)
    if V == 0:
        return None

    # ---- present_mask [B, V] from terms_rows ----------------------------
    terms_rows = meta.get("terms_rows") or [[] for _ in range(B)]
    present_mask = torch.zeros(B, V, device=device)
    for b, terms in enumerate(terms_rows):
        for term in terms:
            vi = active_term_to_idx.get(_term_key(str(term)), -1)
            if 0 <= vi < V:
                present_mask[b, vi] = 1.0

    # ---- target_vectors [B, V, D] from criterion's vocab_matrix --------
    vocab_mat = criterion_vocab_matrix.to(device)  # [V, D]
    if vocab_mat.shape[0] != V:
        # vocab size mismatch — skip this batch silently
        return None
    target_vectors = vocab_mat.unsqueeze(0).expand(B, -1, -1)  # [B, V, D]

    # ---- target_masks [B, V, H, W] from per-term mask stacks -----------
    H, W = int(mb.shape[-2]), int(mb.shape[-1])
    target_masks = torch.zeros(B, V, H, W, device=device)

    mask_stacks = list(meta.get("mask_stacks") or [])
    mask_indices = list(meta.get("mask_indices") or [])

    for b in range(B):
        stk = mask_stacks[b] if b < len(mask_stacks) else None
        idx = mask_indices[b] if b < len(mask_indices) else None

        if stk is not None and idx is not None:
            stk_arr = np.asarray(stk, dtype=np.float32) if not isinstance(stk, np.ndarray) else stk
            stk_t = torch.from_numpy(stk_arr).to(device)
            if stk_t.ndim == 4:
                stk_t = stk_t[:, 0]           # [K, H, W]
            elif stk_t.ndim == 2:
                stk_t = stk_t.unsqueeze(0)    # [1, H, W]
            idx_list: List[int] = list(idx) if hasattr(idx, "__iter__") else [int(idx)]
            for k, vi in enumerate(idx_list):
                if 0 <= vi < V and k < int(stk_t.shape[0]):
                    m = stk_t[k]
                    if tuple(m.shape) != (H, W):
                        m = F.interpolate(
                            m[None, None], size=(H, W), mode="bilinear", align_corners=False
                        )[0, 0]
                    target_masks[b, vi] = m.clamp(0.0, 1.0)
        else:
            # Fallback: broadcast the aggregated mask for all present terms.
            agg = mb[b, 0] if mb.ndim == 4 else mb[b]
            mask_idx = present_mask[b].nonzero(as_tuple=True)[0]
            for vi in mask_idx.tolist():
                target_masks[b, int(vi)] = agg

    return {
        "image": xb,
        "batch_vocab": [vocab_phrases for _ in range(B)],
        "target_vectors": target_vectors,
        "target_masks": target_masks,
        "present_mask": present_mask,
        "target_valid": torch.ones(B, V, dtype=torch.bool, device=device),
    }


def _dedupe_terms(terms: Any) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for item in list(terms or []):
        txt = str(item).strip()
        if txt and txt not in seen:
            seen.add(txt)
            out.append(txt)
    return out


def _cosine_similarity_score(a: torch.Tensor, b: torch.Tensor) -> float:
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        return float("nan")
    if int(a.numel()) <= 0 or int(b.numel()) <= 0:
        return float("nan")
    a_n = F.normalize(a.reshape(1, -1).float(), dim=-1)
    b_n = F.normalize(b.reshape(1, -1).float(), dim=-1)
    return float((a_n * b_n).sum().item())


def _soft_mask_iou_score(pred_mask: torch.Tensor, target_mask: torch.Tensor) -> float:
    if not isinstance(pred_mask, torch.Tensor) or not isinstance(target_mask, torch.Tensor):
        return float("nan")
    pred = pred_mask.float().clamp(0.0, 1.0)
    target = target_mask.float().clamp(0.0, 1.0)
    inter = float((pred * target).sum().item())
    union = float(pred.sum().item() + target.sum().item() - inter)
    if union <= 1e-6:
        return 1.0
    return inter / union


def _build_slot_preview_rows(
    *,
    slot_vectors: torch.Tensor,
    slot_masks: torch.Tensor,
    target_vectors: torch.Tensor,
    target_masks: torch.Tensor,
    slot_assignments: List[Optional[int]],
    vocab_phrases: List[str],
    slot_confidence: torch.Tensor,
) -> List[str]:
    rows: List[str] = []
    n_slots = int(slot_masks.shape[0])
    for si in range(n_slots):
        assigned = slot_assignments[si] if si < len(slot_assignments) else None
        conf = float(slot_confidence[si].item()) if si < int(slot_confidence.shape[0]) else float("nan")
        if assigned is not None and 0 <= int(assigned) < int(target_vectors.shape[0]):
            target_idx = int(assigned)
            phrase = vocab_phrases[target_idx] if target_idx < len(vocab_phrases) else f"class_{target_idx}"
            label_score = _cosine_similarity_score(slot_vectors[si], target_vectors[target_idx])
            mask_score = _soft_mask_iou_score(slot_masks[si], target_masks[target_idx])
            rows.append(f"+s{si} {phrase} conf={conf:.2f} pick={label_score:.2f} mask={mask_score:.2f}")
        else:
            empty_score = max(0.0, min(1.0, 1.0 - float(slot_masks[si].mean().item())))
            rows.append(f"-s{si} dustbin conf={conf:.2f} empty={empty_score:.2f}")
    return rows


# ---------------------------------------------------------------------------
# Generic training loop
# ---------------------------------------------------------------------------


def _slice_speculative_batch_to_device(
    spec: Dict[str, Any],
    *,
    start: int,
    stop: int,
    device: torch.device,
    tensor_dtype: Optional[torch.dtype] = None,
    channels_last: bool = False,
) -> Dict[str, Any]:
    image = spec["image"][start:stop]
    image_to_dtype = tensor_dtype if isinstance(image, torch.Tensor) and bool(image.is_floating_point()) and tensor_dtype is not None else None
    if image.device != device or (image_to_dtype is not None and image.dtype != image_to_dtype):
        if channels_last:
            image = image.to(
                device=device,
                dtype=image_to_dtype,
                non_blocking=True,
                memory_format=torch.channels_last,
            )
        else:
            image = image.to(device=device, dtype=image_to_dtype, non_blocking=True)
    elif channels_last:
        image = image.contiguous(memory_format=torch.channels_last)

    out = {
        "image": image,
        "batch_vocab": list(spec["batch_vocab"][start:stop]),
    }
    for key in ("target_vectors", "target_masks", "present_mask", "target_valid"):
        value = spec[key][start:stop]
        if isinstance(value, torch.Tensor):
            value_to_dtype = tensor_dtype if bool(value.is_floating_point()) and tensor_dtype is not None else None
            if value.device != device or (value_to_dtype is not None and value.dtype != value_to_dtype):
                value = value.to(device=device, dtype=value_to_dtype, non_blocking=True)
        out[key] = value
    return out


def _is_cuda_oom(exc: BaseException) -> bool:
    txt = str(exc).lower()
    if "out of memory" not in txt:
        return False
    return ("cuda" in txt) or ("cudnn" in txt) or ("cublas" in txt)


def _criterion_loss_items(criterion: Any) -> List[Dict[str, Any]]:
    if hasattr(criterion, "loss_report_items") and callable(getattr(criterion, "loss_report_items")):
        try:
            items = list(criterion.loss_report_items())
            if items:
                return items
        except Exception:
            pass
    return [
        {"key": "vector_loss", "label": "vec", "enabled": True, "weight": 1.0},
        {"key": "mask_loss", "label": "mask", "enabled": True, "weight": 1.0},
        {"key": "selection_loss", "label": "sel", "enabled": True, "weight": 1.0},
        {"key": "confidence_loss", "label": "conf", "enabled": True, "weight": 1.0},
    ]


def _init_loss_metric_accumulators(
    criterion: Any,
) -> Dict[str, float]:
    return {
        str(item.get("key", "")): 0.0
        for item in _criterion_loss_items(criterion)
        if str(item.get("key", "")).strip()
    }


def _format_loss_rows(
    criterion: Any,
    loss_metrics: Dict[str, float],
    *,
    denom: float,
    per_row: int = 2,
) -> List[str]:
    items = _criterion_loss_items(criterion)
    tokens: List[str] = []
    safe_denom = max(1.0, float(denom))
    for item in items:
        key = str(item.get("key", "")).strip()
        if not key:
            continue
        label = str(item.get("label", key)).strip()
        enabled = bool(item.get("enabled", True))
        weight = float(item.get("weight", 1.0))
        value = float(loss_metrics.get(key, 0.0)) / safe_denom
        if enabled:
            tokens.append(f"{label}={value:.4f}@{weight:.2f}")
        else:
            tokens.append(f"{label}=off@{weight:.2f}")
    rows: List[str] = []
    for start in range(0, len(tokens), max(1, int(per_row))):
        rows.append(" ".join(tokens[start : start + max(1, int(per_row))]))
    return rows


def _speculative_diversity_bias(
    *,
    net: nn.Module,
    slot_selection_probs: Any,
    vocab_matrix: torch.Tensor,
    vocab_phrases: Sequence[str],
) -> torch.Tensor:
    if not isinstance(slot_selection_probs, torch.Tensor):
        return torch.zeros(0, device=vocab_matrix.device, dtype=vocab_matrix.dtype)
    if int(slot_selection_probs.ndim) != 3:
        return slot_selection_probs.new_zeros(slot_selection_probs.shape)

    slot_decoder = getattr(net, "slot_decoder", None)
    prior = getattr(slot_decoder, "hypergraph_prior", None)
    if prior is None:
        return slot_selection_probs.new_zeros(slot_selection_probs.shape)

    bsz, n_slots, vocab_size = [int(x) for x in slot_selection_probs.shape]
    if vocab_size <= 0:
        return slot_selection_probs.new_zeros(slot_selection_probs.shape)

    vocab_basis = vocab_matrix.to(
        device=slot_selection_probs.device,
        dtype=slot_selection_probs.dtype,
    )
    if int(vocab_basis.ndim) != 2 or int(vocab_basis.shape[0]) != int(vocab_size):
        return slot_selection_probs.new_zeros(slot_selection_probs.shape)
    vocab_basis = F.normalize(vocab_basis, dim=-1, eps=1e-6)

    hypergraph = getattr(net, "hypergraph", None)
    alpha = float(max(1e-6, getattr(net, "hypergraph_alpha", 0.5)))
    base_log_prior = slot_selection_probs.new_full((vocab_size,), -math.log(float(max(1, vocab_size))))
    pair_log_prior = slot_selection_probs.new_full(
        (vocab_size, vocab_size),
        -math.log(float(max(1, vocab_size))),
    )
    if hasattr(hypergraph, "vocab_statistics") and callable(getattr(hypergraph, "vocab_statistics")):
        try:
            stats = hypergraph.vocab_statistics(vocab_phrases, alpha=alpha)
            base_log_prior = stats.get("base_log_prior", base_log_prior).to(
                device=slot_selection_probs.device,
                dtype=slot_selection_probs.dtype,
            )
            pair_log_prior = stats.get("pair_log_prior", pair_log_prior).to(
                device=slot_selection_probs.device,
                dtype=slot_selection_probs.dtype,
            )
        except Exception:
            pass

    predictive_node_state = getattr(prior, "predictive_node_state", None)
    predictive_pair_state = getattr(prior, "predictive_pair_state", None)
    if isinstance(predictive_node_state, torch.Tensor):
        predictive_node = predictive_node_state.to(
            device=slot_selection_probs.device,
            dtype=slot_selection_probs.dtype,
        ).clamp(0.0, 1.0)
    else:
        predictive_node = slot_selection_probs.new_zeros(vocab_size)
    if isinstance(predictive_pair_state, torch.Tensor):
        predictive_pair = predictive_pair_state.to(
            device=slot_selection_probs.device,
            dtype=slot_selection_probs.dtype,
        ).clamp(0.0, 1.0)
    else:
        predictive_pair = slot_selection_probs.new_zeros(vocab_size, vocab_size)

    selected_mass = torch.cumsum(slot_selection_probs, dim=1) - slot_selection_probs
    selected_dist = selected_mass / selected_mass.sum(dim=-1, keepdim=True).clamp_min(1.0)

    base_prob = base_log_prior.exp().clamp(0.0, 1.0).view(1, 1, vocab_size)
    observed_pair_prob = pair_log_prior.exp().clamp(0.0, 1.0)
    predictive_node = predictive_node.view(1, 1, vocab_size)
    semantic_similarity = torch.matmul(vocab_basis, vocab_basis.transpose(0, 1)).clamp(0.0, 1.0)

    observed_pair_penalty = torch.einsum("bnv,vw->bnw", selected_dist, observed_pair_prob)
    predictive_pair_penalty = torch.einsum("bnv,vw->bnw", selected_dist, predictive_pair)
    semantic_redundancy = torch.einsum("bnv,vw->bnw", selected_dist, semantic_similarity)

    rarity_bonus = 1.0 - base_prob
    freshness_bonus = 1.0 - predictive_node

    return (
        (float(getattr(prior, "rarity_bonus_weight", 0.0)) * rarity_bonus)
        + (float(getattr(prior, "freshness_bonus_weight", 0.0)) * freshness_bonus)
        - (float(getattr(prior, "cooccurrence_penalty_weight", 0.0)) * observed_pair_penalty)
        - (float(getattr(prior, "predictive_pair_penalty_weight", 0.0)) * predictive_pair_penalty)
        - (float(getattr(prior, "semantic_similarity_penalty_weight", 0.0)) * semantic_redundancy)
    )


def _run_speculative_net_epochs(
    net: nn.Module,
    criterion: Any,           # PrototypeLoss — accessed via duck typing
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    vocab_phrases: List[str],
    active_term_to_idx: Dict[str, int],
    grad_clip: float = 1.0,
    grad_accum_steps: int = -1,
    network_dropout: float = 0.0,
    log_every: int = 50,
    stage_label: str = "speculative",
    step_preview_callback: Optional[Callable] = None,
    progress_callback: Optional[Callable] = None,
    stop_requested: Optional[Callable[[], bool]] = None,
    pause_requested: Optional[Callable[[], bool]] = None,
    ipc_pump: Optional[Callable[[], None]] = None,
    weight_update_callback: Optional[Callable[[int], None]] = None,
    grad_scaler: Optional[Any] = None,
    amp_enabled: bool = False,
    amp_dtype: str = "fp16",
    channels_last: bool = False,
    max_forward_batch_cap: int = 0,
    optimizer_step_save_callback: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """Train *net* for *epochs* passes over *loader*.

    Uses ``criterion.vocab_matrix`` (the [V, D] normalised ST matrix stored on
    the criterion) as target_vectors so no extra sentence-transformer encode
    runs during the training loop.
    """
    if epochs <= 0:
        return {"ran": False, "loss": 0.0}
    if loader is None:
        _log(f"[{stage_label}] skip: missing loader")
        return {"ran": False, "loss": 0.0, "reason": "missing_loader"}

    vocab_matrix: torch.Tensor = getattr(criterion, "vocab_matrix", None)  # type: ignore[assignment]
    if vocab_matrix is None:
        _log(f"[{stage_label}] skip: criterion has no vocab_matrix")
        return {"ran": False, "loss": 0.0, "reason": "no_vocab_matrix"}

    amp_dtype_t = resolve_amp_dtype(amp_dtype) if amp_enabled else torch.float16
    runtime_amp_enabled = bool(amp_enabled)
    if runtime_amp_enabled and not cuda_supports_dtype(device=device, dtype=amp_dtype_t):
        _log(
            f"[{stage_label}] disabling AMP on device={device} because amp_dtype={amp_dtype} is unsupported there"
        )
        runtime_amp_enabled = False
    runtime_tensor_dtype: Optional[torch.dtype] = None
    for _param in net.parameters():
        if bool(_param.is_floating_point()):
            runtime_tensor_dtype = _param.dtype
            break
    if runtime_tensor_dtype == torch.float64 and bool(runtime_amp_enabled):
        _log(f"[{stage_label}] disabling AMP because active-network parameters are float64")
        runtime_amp_enabled = False
        setattr(net, "_runtime_amp_enabled", False)
    runtime_channels_last = bool(channels_last)
    use_scaler = bool(
        grad_scaler is not None
        and runtime_amp_enabled
        and device.type == "cuda"
        and amp_dtype_t == torch.float16
    )

    net.train()
    network_dropout = float(max(0.0, min(0.95, network_dropout)))
    total_loss = 0.0
    n_samples = 0
    optimizer_steps = 0
    global_step = 0
    t_start = time.time()
    stop_now = False
    skipped_batches = 0
    last_grad_norm_value = float("nan")
    grad_accum_cfg = int(grad_accum_steps)
    accumulate_full_epoch = int(grad_accum_cfg) < 0
    target_grad_accum_batches = 1 if int(grad_accum_cfg) == 0 else max(1, int(grad_accum_cfg))
    pending_grad_batches = 0
    last_batch_terms_context = "samples=[]"
    grad_accum_desc = (
        "full-epoch"
        if bool(accumulate_full_epoch)
        else ("disabled" if int(grad_accum_cfg) == 0 else f"{int(target_grad_accum_batches)} batches/step")
    )
    _log(
        f"[{stage_label}] regulation grad_clip={float(grad_clip):.4f} "
        f"network_dropout={network_dropout:.4f} "
        f"grad_accum={grad_accum_desc}"
    )
    optimizer.zero_grad(set_to_none=True)

    def _finish_accumulated_step(*, batch_terms_context: str) -> bool:
        nonlocal pending_grad_batches
        nonlocal skipped_batches
        nonlocal optimizer_steps
        nonlocal last_grad_norm_value

        if int(pending_grad_batches) <= 0:
            return False

        if use_scaler:
            grad_scaler.unscale_(optimizer)
        _scale_param_grads_(net, 1.0 / float(max(1, int(pending_grad_batches))))

        bad_grad_names = _collect_nonfinite_param_names(net, grads=True)
        if bad_grad_names:
            optimizer.zero_grad(set_to_none=True)
            pending_grad_batches = 0
            skipped_batches += 1
            _log(
                f"[{stage_label}] rejecting accumulated step with non-finite gradients: "
                + ", ".join(bad_grad_names)
                + f" | {batch_terms_context}"
            )
            return False

        grad_norm = nn.utils.clip_grad_norm_(net.parameters(), float(grad_clip))
        grad_norm_value = float(grad_norm.detach().item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
        last_grad_norm_value = float(grad_norm_value)
        if not math.isfinite(grad_norm_value):
            optimizer.zero_grad(set_to_none=True)
            pending_grad_batches = 0
            skipped_batches += 1
            _log(
                f"[{stage_label}] rejecting accumulated step with non-finite gradient norm: "
                f"{grad_norm_value} | {batch_terms_context}"
            )
            return False

        param_snapshots = [
            (param, param.detach().clone())
            for param in net.parameters()
            if bool(param.requires_grad)
        ]
        if use_scaler:
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            optimizer.step()

        bad_param_names = _collect_nonfinite_param_names(net, grads=False)
        if bad_param_names:
            for param, snapshot in param_snapshots:
                param.data.copy_(snapshot)
            optimizer.state.clear()
            optimizer.zero_grad(set_to_none=True)
            pending_grad_batches = 0
            skipped_batches += 1
            _log(
                f"[{stage_label}] rejected accumulated optimizer step with non-finite parameters: "
                + ", ".join(bad_param_names)
                + f" | {batch_terms_context}"
            )
            return False

        optimizer_steps += 1
        if weight_update_callback is not None:
            try:
                weight_update_callback(optimizer_steps)
            except Exception:
                pass
        if optimizer_step_save_callback is not None:
            try:
                optimizer_step_save_callback()
            except Exception:
                pass
        optimizer.zero_grad(set_to_none=True)
        pending_grad_batches = 0
        return True

    for _epoch in range(epochs):
        if stop_now:
            break
        loader_iter = iter(loader)
        while True:
            if stop_requested is not None and stop_requested():
                stop_now = True
                break
            if ipc_pump is not None:
                try:
                    ipc_pump()
                except Exception:
                    pass

            try:
                raw_batch = next(loader_iter)
            except StopIteration:
                break

            try:
                xb, mb, meta = _unpack_masked_semantic_batch(raw_batch, context=stage_label)
            except Exception as exc:
                _log(f"[{stage_label}] batch unpack error: {exc}")
                continue

            spec = _build_speculative_batch(
                xb, mb, meta,
                vocab_phrases=vocab_phrases,
                criterion_vocab_matrix=vocab_matrix,
                active_term_to_idx=active_term_to_idx,
                device=torch.device("cpu"),
            )
            if spec is None:
                continue
            if network_dropout > 0.0:
                _apply_network_dropout_rate(net, network_dropout)
            terms_rows = list(meta.get("terms_rows") or [[] for _ in range(int(spec["image"].shape[0]))])
            total_batch_n = int(spec["image"].shape[0])
            slice_cap = int(max_forward_batch_cap) if int(max_forward_batch_cap) > 0 else int(total_batch_n)
            slice_cap = max(1, min(int(slice_cap), int(total_batch_n)))
            step_loss_value = 0.0
            step_seen = 0
            step_metrics = _init_loss_metric_accumulators(criterion)
            preview_bundle: Optional[Dict[str, Any]] = None
            predicted_selection_parts: List[torch.Tensor] = []
            predicted_confidence_parts: List[torch.Tensor] = []
            batch_terms_context = _format_terms_rows_brief(terms_rows)
            batch_rejection_message: Optional[str] = None
            batch_grad_snapshot = (
                _snapshot_param_grads(net)
                if int(pending_grad_batches) > 0
                else None
            )
            while True:
                try:
                    step_loss_value = 0.0
                    step_seen = 0
                    for key in step_metrics.keys():
                        step_metrics[key] = 0.0
                    preview_bundle = None
                    predicted_selection_parts = []
                    predicted_confidence_parts = []

                    for start in range(0, int(total_batch_n), int(slice_cap)):
                        stop = min(int(total_batch_n), int(start + slice_cap))
                        spec_part = _slice_speculative_batch_to_device(
                            spec,
                            start=start,
                            stop=stop,
                            device=device,
                            tensor_dtype=runtime_tensor_dtype,
                            channels_last=runtime_channels_last,
                        )
                        slice_terms_rows = terms_rows[start:stop]
                        input_nonfinite = _collect_nonfinite_tensor_stats(
                            [
                                ("image", spec_part.get("image")),
                                ("target_vectors", spec_part.get("target_vectors")),
                                ("target_masks", spec_part.get("target_masks")),
                                ("present_mask", spec_part.get("present_mask")),
                            ]
                        )
                        if input_nonfinite:
                            _raise_nonfinite_error(
                                stage_label=stage_label,
                                reason="rejecting batch with non-finite speculative inputs",
                                stats=input_nonfinite,
                                terms_rows=slice_terms_rows,
                                sample_offset=start,
                            )

                        try:
                            with autocast_context(
                                device=device,
                                enabled=runtime_amp_enabled,
                                amp_dtype=amp_dtype_t,
                            ):
                                raw_out = (
                                    net.forward_batch(spec_part["image"], spec_part["batch_vocab"])
                                    if hasattr(net, "forward_batch") and callable(getattr(net, "forward_batch"))
                                    else net(spec_part["image"], spec_part["batch_vocab"])
                                )
                                out = coerce_network_output(raw_out)
                                aux = out.aux if isinstance(out.aux, dict) else {}
                                forward_nonfinite = _collect_nonfinite_tensor_stats(
                                    [
                                        ("slot_vectors", out.slot_vectors),
                                        ("slot_masks", out.slot_masks),
                                        ("slot_confidence", out.slot_confidence),
                                        ("slot_selection_logits", aux.get("slot_selection_logits")),
                                        ("slot_selection_probs", aux.get("slot_selection_probs")),
                                        ("image_features", aux.get("image_features")),
                                        ("active_rows", aux.get("active_rows")),
                                        ("dynamic_layer", aux.get("dynamic_layer")),
                                        ("spatial_memory", aux.get("spatial_memory")),
                                        ("slot_hidden", aux.get("slot_hidden")),
                                        ("slot_prior_bias", aux.get("slot_prior_bias")),
                                        ("mask_canvas", aux.get("mask_canvas")),
                                    ]
                                )
                                if forward_nonfinite:
                                    _raise_nonfinite_error(
                                        stage_label=stage_label,
                                        reason="rejecting batch with non-finite active-network tensors before criterion",
                                        stats=forward_nonfinite,
                                        terms_rows=slice_terms_rows,
                                        sample_offset=start,
                                    )
                                slot_selection_logits = aux.get("slot_selection_logits")
                                adjusted_selection_logits = slot_selection_logits
                                if isinstance(slot_selection_logits, torch.Tensor):
                                    diversity_bias = _speculative_diversity_bias(
                                        net=net,
                                        slot_selection_probs=aux.get("slot_selection_probs"),
                                        vocab_matrix=vocab_matrix,
                                        vocab_phrases=vocab_phrases,
                                    )
                                    diversity_weight = float(
                                        max(
                                            0.0,
                                            getattr(getattr(net, "slot_decoder", None), "hypergraph_prior_weight", 0.0),
                                        )
                                    )
                                    if (
                                        float(diversity_weight) > 0.0
                                        and isinstance(diversity_bias, torch.Tensor)
                                        and tuple(diversity_bias.shape) == tuple(slot_selection_logits.shape)
                                    ):
                                        raw_selection_logits = slot_selection_logits
                                        slot_prior_bias = aux.get("slot_prior_bias")
                                        if (
                                            isinstance(slot_prior_bias, torch.Tensor)
                                            and tuple(slot_prior_bias.shape) == tuple(slot_selection_logits.shape)
                                        ):
                                            raw_selection_logits = slot_selection_logits - (
                                                float(diversity_weight)
                                                * slot_prior_bias.to(
                                                    device=slot_selection_logits.device,
                                                    dtype=slot_selection_logits.dtype,
                                                )
                                            )
                                        adjusted_selection_logits = raw_selection_logits + (
                                            float(diversity_weight) * diversity_bias
                                        )
                                loss_dict = criterion(
                                    pred_vectors=out.slot_vectors,
                                    pred_masks=out.slot_masks,
                                    target_vectors=spec_part["target_vectors"],
                                    target_masks=spec_part["target_masks"],
                                    target_valid=spec_part["target_valid"],
                                    present_mask=spec_part["present_mask"],
                                    confidence_logits=out.slot_confidence,
                                    slot_selection_logits=adjusted_selection_logits,
                                )
                                loss_nonfinite = _collect_nonfinite_tensor_stats(
                                    [
                                        ("loss", loss_dict.get("loss")),
                                        ("vector_loss", loss_dict.get("vector_loss")),
                                        ("mask_loss", loss_dict.get("mask_loss")),
                                        ("selection_loss", loss_dict.get("selection_loss")),
                                        ("confidence_loss", loss_dict.get("confidence_loss")),
                                        ("mask_order_loss", loss_dict.get("mask_order_loss")),
                                        ("residual_mask_loss", loss_dict.get("residual_mask_loss")),
                                    ]
                                )
                                if loss_nonfinite:
                                    _raise_nonfinite_error(
                                        stage_label=stage_label,
                                        reason="rejecting batch with non-finite speculative losses",
                                        stats=loss_nonfinite,
                                        terms_rows=slice_terms_rows,
                                        sample_offset=start,
                                    )
                                nonfinite_stats = loss_dict.get("nonfinite_stats")
                                if isinstance(nonfinite_stats, dict):
                                    output_nonfinite = int(
                                        max(
                                            int(nonfinite_stats.get("pred_vectors", 0)),
                                            int(nonfinite_stats.get("pred_masks", 0)),
                                            int(nonfinite_stats.get("confidence_logits", 0)),
                                            int(nonfinite_stats.get("slot_selection_logits", 0)),
                                        )
                                    )
                                    if output_nonfinite > 0:
                                        raise _RejectSpeculativeBatch(
                                            f"[{stage_label}] rejecting batch after sanitized non-finite "
                                            f"active-network outputs: {dict(nonfinite_stats)} | "
                                            f"{_format_terms_rows_brief(slice_terms_rows, sample_offset=start)}"
                                        )
                        except (StageSkipForward, StageSkipBack, StageStopRequested):
                            raise
                        except _RejectSpeculativeBatch:
                            raise
                        except Exception as exc:
                            raise _RejectSpeculativeBatch(
                                f"[{stage_label}] rejecting batch after criterion/forward error: {exc} | "
                                f"{_format_terms_rows_brief(slice_terms_rows, sample_offset=start)}"
                            ) from exc

                        part_n = int(stop - start)
                        part_weight = float(part_n) / float(max(1, int(total_batch_n)))
                        loss_to_backprop = loss_dict["loss"] * float(part_weight)
                        if not bool(torch.isfinite(loss_to_backprop.detach()).item()):
                            raise _RejectSpeculativeBatch(
                                f"[{stage_label}] rejecting batch with non-finite speculative loss after "
                                f"sanitization | {_format_terms_rows_brief(slice_terms_rows, sample_offset=start)}"
                            )
                        if use_scaler:
                            grad_scaler.scale(loss_to_backprop).backward()
                        else:
                            loss_to_backprop.backward()

                        step_loss_value += float(loss_dict["loss"].detach().item()) * float(part_n)
                        step_seen += int(part_n)
                        for key in step_metrics.keys():
                            value = loss_dict.get(key, 0.0)
                            if isinstance(value, torch.Tensor):
                                value = float(value.detach().item())
                            step_metrics[key] += float(value) * float(part_n)
                        selection_probs = aux.get("slot_selection_probs")
                        if isinstance(selection_probs, torch.Tensor):
                            predicted_selection_parts.append(selection_probs.detach().cpu())
                            predicted_confidence_parts.append(out.slot_confidence.detach().cpu())

                        if step_preview_callback is not None and int(stop) == int(total_batch_n):
                            diagnostics: List[Dict[str, Any]] = []
                            if hasattr(net, "collect_preview_diagnostics") and callable(getattr(net, "collect_preview_diagnostics")):
                                try:
                                    diagnostics = list(
                                        net.collect_preview_diagnostics(
                                            out,
                                            spec_part["batch_vocab"],
                                            terms_rows=terms_rows[start:stop],
                                        )
                                    )
                                except Exception:
                                    diagnostics = []
                            preview_bundle = {
                                "image": spec_part["image"].detach().cpu(),
                                "slot_vectors": out.slot_vectors.detach().cpu(),
                                "slot_masks": out.slot_masks.detach().cpu().sigmoid(),
                                "slot_confidence": out.slot_confidence.detach().cpu().sigmoid(),
                                "target_vectors": spec_part["target_vectors"].detach().cpu(),
                                "target_masks": spec_part["target_masks"].detach().cpu(),
                                "assignments": list(loss_dict.get("assignments") or []),
                                "batch_vocab": list(spec_part["batch_vocab"]),
                                "terms_rows": list(terms_rows[start:stop]),
                                "diagnostics": diagnostics,
                            }

                        del spec_part, raw_out, out, loss_dict
                    break
                except _RejectSpeculativeBatch as exc:
                    batch_rejection_message = str(exc)
                    if batch_grad_snapshot is None:
                        optimizer.zero_grad(set_to_none=True)
                    else:
                        _restore_param_grads(batch_grad_snapshot)
                    break
                except torch.OutOfMemoryError:
                    if device.type != "cuda":
                        raise
                    if batch_grad_snapshot is None:
                        optimizer.zero_grad(set_to_none=True)
                    else:
                        _restore_param_grads(batch_grad_snapshot)
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
                        net = net.to(memory_format=torch.contiguous_format)
                        setattr(net, "_runtime_channels_last", False)
                        _log(f"[{stage_label}] cuda oom; retrying with contiguous tensors")
                        continue
                    if int(slice_cap) <= 1:
                        raise
                    next_slice_cap = max(1, int(slice_cap // 2))
                    if int(next_slice_cap) > 1:
                        next_slice_cap = 1 << (int(next_slice_cap).bit_length() - 1)
                    slice_cap = int(next_slice_cap)
                    _log(f"[{stage_label}] cuda oom; retrying with smaller microbatch cap {int(slice_cap)}")
                except RuntimeError as exc:
                    if device.type == "cuda" and _is_cuda_backend_engine_error(exc):
                        if batch_grad_snapshot is None:
                            optimizer.zero_grad(set_to_none=True)
                        else:
                            _restore_param_grads(batch_grad_snapshot)
                        if bool(runtime_amp_enabled):
                            runtime_amp_enabled = False
                            use_scaler = False
                            setattr(net, "_runtime_amp_enabled", False)
                            torch.cuda.empty_cache()
                            _log(f"[{stage_label}] backend engine selection failed; retrying with AMP disabled")
                            continue
                        if bool(runtime_channels_last):
                            runtime_channels_last = False
                            net = net.to(memory_format=torch.contiguous_format)
                            setattr(net, "_runtime_channels_last", False)
                            torch.cuda.empty_cache()
                            _log(f"[{stage_label}] backend engine selection failed; retrying with contiguous tensors")
                            continue
                    if device.type == "cuda" and _is_cuda_oom(exc):
                        if batch_grad_snapshot is None:
                            optimizer.zero_grad(set_to_none=True)
                        else:
                            _restore_param_grads(batch_grad_snapshot)
                        gc.collect()
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        if bool(runtime_channels_last):
                            runtime_channels_last = False
                            net = net.to(memory_format=torch.contiguous_format)
                            setattr(net, "_runtime_channels_last", False)
                            _log(f"[{stage_label}] cuda oom; retrying with contiguous tensors")
                            continue
                        if int(slice_cap) > 1:
                            next_slice_cap = max(1, int(slice_cap // 2))
                            if int(next_slice_cap) > 1:
                                next_slice_cap = 1 << (int(next_slice_cap).bit_length() - 1)
                            slice_cap = int(next_slice_cap)
                            _log(f"[{stage_label}] cuda oom; retrying with smaller microbatch cap {int(slice_cap)}")
                            continue
                    raise

            if batch_rejection_message is not None:
                skipped_batches += 1
                _log(batch_rejection_message)
                del xb, mb, spec
                continue

            if (
                predicted_selection_parts
                and predicted_confidence_parts
                and hasattr(net, "observe_predictions")
                and callable(getattr(net, "observe_predictions"))
            ):
                try:
                    net.observe_predictions(
                        torch.cat(predicted_selection_parts, dim=0),
                        torch.cat(predicted_confidence_parts, dim=0),
                    )
                except Exception:
                    pass
            pending_grad_batches += 1
            last_batch_terms_context = batch_terms_context
            stepped_this_batch = False
            if (not bool(accumulate_full_epoch)) and int(pending_grad_batches) >= int(target_grad_accum_batches):
                stepped_this_batch = bool(_finish_accumulated_step(batch_terms_context=batch_terms_context))

            batch_n = int(total_batch_n)
            batch_loss = float(step_loss_value) / float(max(1, step_seen))
            total_loss += batch_loss * batch_n
            n_samples += batch_n
            global_step += 1
            avg_loss = total_loss / max(1, n_samples)
            elapsed = max(1e-6, time.time() - t_start)
            if hasattr(net, "observe_terms") and callable(getattr(net, "observe_terms")):
                try:
                    net.observe_terms(terms_rows, step=global_step)
                except Exception:
                    pass

            if step_preview_callback is not None and isinstance(preview_bundle, dict):
                payload_batch = []
                assignments = list(preview_bundle.get("assignments") or [])
                diagnostics = list(preview_bundle.get("diagnostics") or [])
                conf_t = preview_bundle["slot_confidence"]
                loss_rows = _format_loss_rows(
                    criterion,
                    step_metrics,
                    denom=float(batch_n),
                    per_row=2,
                )
                loss_scalars = {
                    str(key): float(value) / float(max(1, batch_n))
                    for key, value in step_metrics.items()
                }
                loss_scalars["batch_loss"] = float(batch_loss)
                loss_scalars["loss"] = float(avg_loss)
                for bi in range(min(batch_n, int(preview_bundle["image"].shape[0]), 4)):
                    slot_assignments = (
                        assignments[bi]["slot_to_target"]
                        if bi < len(assignments) and isinstance(assignments[bi], dict) else []
                    )
                    sample_diag = diagnostics[bi] if bi < len(diagnostics) and isinstance(diagnostics[bi], dict) else {}
                    terms_row = preview_bundle["terms_rows"][bi] if bi < len(preview_bundle["terms_rows"]) else []
                    target_labels = list(sample_diag.get("target_labels") or _dedupe_terms(terms_row))
                    decision_rows = [
                        f"loss={avg_loss:.4f} batch={batch_loss:.4f}",
                    ]
                    decision_rows.extend(loss_rows)
                    decision_rows.extend(list(sample_diag.get("decision_rows") or []))
                    slot_rows = _build_slot_preview_rows(
                        slot_vectors=preview_bundle["slot_vectors"][bi],
                        slot_masks=preview_bundle["slot_masks"][bi],
                        target_vectors=preview_bundle["target_vectors"][bi],
                        target_masks=preview_bundle["target_masks"][bi],
                        slot_assignments=slot_assignments,
                        vocab_phrases=vocab_phrases,
                        slot_confidence=conf_t[bi],
                    )
                    payload_batch.append({
                        "img": preview_bundle["image"][bi],
                        "slot_masks": preview_bundle["slot_masks"][bi],
                        "slot_confidence": conf_t[bi],
                        "assignments": slot_assignments,
                        "vocab_phrases": vocab_phrases,
                        "target_labels": target_labels or ["none"],
                        "decision_rows": decision_rows,
                        "slot_rows": slot_rows,
                        "hypergraph_summary": sample_diag.get("hypergraph_summary", {}),
                        "hypergraph_query": sample_diag.get("hypergraph_query", []),
                        "loss": avg_loss,
                        "batch_loss": batch_loss,
                        "global_step": global_step,
                        "total_steps": 0,
                        "loss_scalars": dict(loss_scalars),
                    })
                try:
                    step_preview_callback(payload_batch)
                except Exception as exc:
                    _log(f"[{stage_label}] preview callback error: {exc}")

            if int(log_every) > 0 and global_step % int(log_every) == 0:
                loss_rows = _format_loss_rows(
                    criterion,
                    step_metrics,
                    denom=float(batch_n),
                    per_row=3,
                )
                grad_txt = (
                    f"{last_grad_norm_value:.3g}/{float(grad_clip):.3g}"
                    if bool(stepped_this_batch) and math.isfinite(last_grad_norm_value)
                    else f"pending/{float(grad_clip):.3g}"
                )
                accum_txt = (
                    f"{int(pending_grad_batches)}/epoch"
                    if bool(accumulate_full_epoch)
                    else f"{int(pending_grad_batches)}/{int(target_grad_accum_batches)}"
                )
                _log(
                    f"[{stage_label}] step={global_step} "
                    f"loss={avg_loss:.4f} "
                    f"{' | '.join(loss_rows)} "
                    f"accum={accum_txt} "
                    f"grad={grad_txt} "
                    f"sps={n_samples / elapsed:.1f}"
                )

                if progress_callback is not None:
                    try:
                        progress_callback({
                            "global_step": global_step,
                            "total_steps": 0,
                            "loss": avg_loss,
                            "samples_per_sec": n_samples / elapsed,
                        })
                    except (StageSkipForward, StageSkipBack, StageStopRequested):
                        raise
                    except Exception:
                        pass

            # Clean up to free GPU memory promptly
            del xb, mb, spec
            if stop_now:
                break

        if stop_now:
            optimizer.zero_grad(set_to_none=True)
            pending_grad_batches = 0
            break
        if int(pending_grad_batches) > 0:
            _finish_accumulated_step(batch_terms_context=last_batch_terms_context)

    net.eval()
    return {
        "ran": True,
        "loss": total_loss / max(1, n_samples),
        "samples": n_samples,
        "steps": global_step,
        "optimizer_steps": optimizer_steps,
        "skipped_batches": skipped_batches,
        "elapsed_sec": time.time() - t_start,
    }


# ---------------------------------------------------------------------------
# Build node
# ---------------------------------------------------------------------------


class BuildSpeculativeNetNode(PipelineNode):
    """Instantiate PrototypeAutoClassifier and store it on ctx.active_network.

    Runs exactly once.  Requires ctx.class_names to be populated before this
    node executes (LabelEmbeddingNode or VocabNode must run first).
    """

    node_id = "build_classifier"
    description = "Build active recognition network"
    runtime_object_type = "builder"
    runtime_faculty = "build"
    gpu_models = ["active_network"]

    def __init__(self, cfg: SpeculativeNetConfig) -> None:
        self.cfg = cfg
        self._built = False

    @property
    def runtime_execution_policy(self) -> tuple:
        return ("once", {})

    def should_run(self, ctx: PipelineContext) -> bool:
        return not self._built

    def execute(self, ctx: PipelineContext) -> None:
        from speculative_network import (
            PrototypeAutoClassifier,
            PrototypeLoss,
            OneToOneMatcher,
            SentenceTransformerVocabBank,
            DynamicRowBank,
        )

        phrases = list(ctx.class_names or [])
        if not phrases:
            raise RuntimeError(
                "[speculative] ctx.class_names is empty — "
                "populate vocab before building the network"
            )

        _log(f"[speculative] encoding {len(phrases)} vocab phrases via sentence-transformers…")
        vocab_bank = SentenceTransformerVocabBank(
            phrases=phrases,
            model_name=self.cfg.sentence_transformer_model,
            device=str(ctx.device),
            normalize=True,
        )
        row_bank = DynamicRowBank(phrases=phrases, row_dim=self.cfg.row_dim)
        dtype_key = str(getattr(self.cfg, "network_dtype", "float64")).strip().lower()
        dtype_map = {
            "float64": torch.float64,
            "fp64": torch.float64,
            "double": torch.float64,
            "float32": torch.float32,
            "fp32": torch.float32,
            "float": torch.float32,
        }
        if dtype_key not in dtype_map:
            raise ValueError(
                f"[speculative] unsupported network_dtype={self.cfg.network_dtype!r}; "
                "expected one of float64/fp64/double/float32/fp32/float"
            )
        model_dtype = dtype_map[dtype_key]

        model = PrototypeAutoClassifier(
            vocab_bank=vocab_bank,
            row_bank=row_bank,
            hidden_dim=self.cfg.hidden_dim,
            n_slots=self.cfg.n_slots,
            image_size=self.cfg.image_size,
            in_channels=self.cfg.in_channels,
            selection_temp=self.cfg.selection_temp,
            hypergraph_prior_weight=self.cfg.hypergraph_prior_weight,
            duplicate_penalty=self.cfg.duplicate_penalty,
            hypergraph_alpha=self.cfg.hypergraph_alpha,
            network_dropout=self.cfg.network_dropout,
            predictive_hypergraph_momentum=self.cfg.predictive_hypergraph_momentum,
        ).to(device=ctx.device, dtype=model_dtype)
        if self.cfg.channels_last:
            model = model.to(memory_format=torch.channels_last)
        setattr(model, "_runtime_amp_enabled", bool(self.cfg.amp))
        setattr(model, "_runtime_amp_dtype", str(self.cfg.amp_dtype))
        setattr(model, "_runtime_channels_last", bool(self.cfg.channels_last))
        setattr(model, "_runtime_microbatch_cap", int(self.cfg.max_forward_batch_cap))
        setattr(model, "_runtime_grad_accum_steps", int(self.cfg.grad_accum_steps))
        setattr(model, "_runtime_network_dropout", float(max(0.0, min(0.95, self.cfg.network_dropout))))
        if hasattr(model, "set_preview_preferences") and callable(getattr(model, "set_preview_preferences")):
            try:
                model.set_preview_preferences(
                    output_panels=int(getattr(getattr(ctx, "args", None), "network_preview_output_panels", 2) or 2),
                    confidence_floor=float(getattr(getattr(ctx, "args", None), "network_preview_confidence_floor", 0.0) or 0.0),
                )
            except Exception:
                pass

        # Vocab matrix: normalised [V, D] ST vectors — lives on criterion as a buffer.
        vocab_matrix = F.normalize(
            torch.stack([vocab_bank.get_vector(p) for p in phrases], dim=0),
            dim=-1,
        ).to(dtype=model_dtype)
        matcher = OneToOneMatcher(dustbin_cost=self.cfg.dustbin_cost)
        criterion = PrototypeLoss(
            matcher=matcher,
            vocab_matrix=vocab_matrix,
            enable_vector_loss=self.cfg.enable_vector_loss,
            vector_weight=self.cfg.vector_weight,
            enable_mask_loss=self.cfg.enable_mask_loss,
            mask_weight=self.cfg.mask_weight,
            enable_selection_loss=self.cfg.enable_selection_loss,
            selection_weight=self.cfg.selection_weight,
            selection_temp=self.cfg.selection_temp,
            enable_confidence_loss=self.cfg.enable_confidence_loss,
            confidence_weight=self.cfg.confidence_weight,
            enable_mask_order_loss=self.cfg.enable_mask_order_loss,
            mask_order_weight=self.cfg.mask_order_weight,
            mask_order_margin=self.cfg.mask_order_margin,
            mask_order_target_volume_weight=self.cfg.mask_order_target_volume_weight,
            mask_order_target_spread_weight=self.cfg.mask_order_target_spread_weight,
            mask_order_pred_volume_weight=self.cfg.mask_order_pred_volume_weight,
            mask_order_pred_spread_weight=self.cfg.mask_order_pred_spread_weight,
            enable_residual_mask_loss=self.cfg.enable_residual_mask_loss,
            residual_mask_weight=self.cfg.residual_mask_weight,
            residual_mask_detach_canvas=self.cfg.residual_mask_detach_canvas,
            mask_fn_weight=self.cfg.mask_fn_weight,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        # The speculative network fully replaces the legacy classifier path.
        ctx.classifier = None
        ctx.classifier_optimizer = None
        ctx.classifier_grad_scaler = None
        ctx.classifier_lr_controller = None
        ctx.gate_classifier = None
        ctx.active_network = model
        ctx.active_network_optimizer = optimizer
        ctx.active_network_grad_scaler = make_grad_scaler(
            enabled=bool(
                self.cfg.amp
                and ctx.device.type == "cuda"
                and resolve_amp_dtype(self.cfg.amp_dtype) == torch.float16
            )
        ) if self.cfg.amp else None
        ctx.active_network_criterion = criterion
        resume_ckpt = ctx.resume_pipeline_ckpt if isinstance(ctx.resume_pipeline_ckpt, dict) else None
        if resume_ckpt is not None:
            if "active_network_state" in resume_ckpt:
                try:
                    model.load_state_dict(resume_ckpt["active_network_state"], strict=False)
                    _log("[speculative] resumed model state from pipeline checkpoint")
                except Exception as exc:
                    _log(f"[speculative] WARNING: could not resume model state: {exc}")
            if "active_network_optimizer_state" in resume_ckpt:
                try:
                    optimizer.load_state_dict(resume_ckpt["active_network_optimizer_state"])
                    _log("[speculative] resumed optimizer state from pipeline checkpoint")
                except Exception as exc:
                    _log(f"[speculative] WARNING: could not resume optimizer state: {exc}")
            scaler = getattr(ctx, "active_network_grad_scaler", None)
            if scaler is not None and "active_network_grad_scaler_state" in resume_ckpt:
                try:
                    scaler.load_state_dict(resume_ckpt["active_network_grad_scaler_state"])
                    _log("[speculative] resumed grad-scaler state from pipeline checkpoint")
                except Exception as exc:
                    _log(f"[speculative] WARNING: could not resume grad-scaler state: {exc}")

        _log(
            f"[speculative] built: vocab={len(phrases)} n_slots={self.cfg.n_slots} "
            f"hidden_dim={self.cfg.hidden_dim} in_channels={self.cfg.in_channels} "
            f"image_size={self.cfg.image_size}"
        )
        self._built = True


# ---------------------------------------------------------------------------
# Training stages
# ---------------------------------------------------------------------------


class _SpeculativeBaseTrainNode(IRTrainingNode):
    """Shared execute infrastructure for all speculative training stages."""

    gpu_models = ["active_network"]
    model_attr = "active_network"
    optimizer_attrs = ["active_network_optimizer"]

    def __init__(self, cfg: SpeculativeNetConfig) -> None:
        self.cfg = cfg

    def should_run(self, ctx: PipelineContext) -> bool:
        if not super().should_run(ctx):
            return False
        return ctx.active_network is not None and ctx.active_network_criterion is not None

    def _run(
        self,
        ctx: PipelineContext,
        epochs: int,
        stage_label: str,
        metric_stage: str,
        grad_accum_steps: int = -1,
    ) -> None:
        loader_info = ensure_runtime_loader_contract(ctx, consumer_id=self.node_id)
        preview_cb = make_speculative_step_preview_callback(ctx, self.node_id)
        weight_cb = make_runtime_weight_publish_callback(
            ctx,
            model_name="active_network",
            model=ctx.active_network,
            node_id=self.node_id,
        )

        step_save_cb: Optional[Callable[[], None]] = None
        if self.cfg.save_on_grad_step:
            _out_dir = getattr(ctx, "output_dir", None)
            if _out_dir is not None:
                _net = ctx.active_network
                _opt = ctx.active_network_optimizer
                _scaler = getattr(ctx, "active_network_grad_scaler", None)
                _label = stage_label
                def _make_step_save(_out_dir, _net, _opt, _scaler, _label, _ctx):
                    def _cb():
                        _path = _out_dir / "active_network.pt"
                        state_dict = _net.state_dict()
                        payload = {
                            "checkpoint_kind": "speculative_optimizer_step",
                            "timestamp": float(time.time()),
                            "segment": str(_label),
                            "round_id": int(getattr(_ctx, "round_id", 0) or 0),
                            "cycle": int(getattr(_ctx, "cycle", 0) or 0),
                            "total_rounds_completed": int(getattr(_ctx, "total_rounds_completed", 0) or 0),
                            "run_tag": str(getattr(_ctx, "run_tag", "") or ""),
                            "active_network_state": state_dict,
                            "state_dict": state_dict,
                        }
                        if _opt is not None:
                            try:
                                payload["active_network_optimizer_state"] = _opt.state_dict()
                            except Exception:
                                pass
                        if _scaler is not None and hasattr(_scaler, "state_dict"):
                            try:
                                payload["active_network_grad_scaler_state"] = _scaler.state_dict()
                            except Exception:
                                pass
                        _save_pipeline_checkpoint(_path, payload)
                        _log(f"[{_label}] mid-step checkpoint saved")
                    return _cb
                step_save_cb = _make_step_save(_out_dir, _net, _opt, _scaler, _label, ctx)

        result = _run_speculative_net_epochs(
            net=ctx.active_network,
            criterion=ctx.active_network_criterion,
            optimizer=ctx.active_network_optimizer,
            loader=loader_info["loader"],
            device=ctx.device,
            epochs=epochs,
            vocab_phrases=list(ctx.class_names),
            active_term_to_idx=dict(ctx.semantic_term_to_idx),
            grad_clip=self.cfg.grad_clip,
            grad_accum_steps=grad_accum_steps,
            network_dropout=self.cfg.network_dropout,
            log_every=self.cfg.log_every,
            stage_label=stage_label,
            step_preview_callback=preview_cb,
            progress_callback=make_training_progress_callback(
                ctx, self.node_id, stage_label,
                publish_loss=(preview_cb is None),
            ),
            stop_requested=ctx.stop_requested,
            pause_requested=ctx.paused,
            ipc_pump=getattr(ctx.viewer_proxy, "pump", None),
            weight_update_callback=weight_cb,
            grad_scaler=ctx.active_network_grad_scaler,
            amp_enabled=self.cfg.amp,
            amp_dtype=self.cfg.amp_dtype,
            channels_last=self.cfg.channels_last,
            max_forward_batch_cap=self.cfg.max_forward_batch_cap,
            optimizer_step_save_callback=step_save_cb,
        )
        loss = float(result.get("loss", float("inf")))
        ctx.log_metric(metric_stage, "loss", loss)
        _log(f"[{stage_label}] done — loss={loss:.4f}")


class SpeculativePregestationNode(_SpeculativeBaseTrainNode):
    """Stage 0: train the speculative network on the pregestation loader."""

    node_id = "stage_0_pregestation"
    description = "Stage 0: speculative network training (pregestation)"

    def ir_input_ports(self) -> List[IRTensorPortSpec]:
        return [
            IRTensorPortSpec("pregestation_images", "Pregestation images",
                             io="input", dtype="float32", shape="B x C x H x W",
                             semantic="image_batch", detail="synthetic logic render batch"),
        ]

    def ir_output_ports(self) -> List[IRTensorPortSpec]:
        return [
            IRTensorPortSpec("slot_masks", "Slot masks",
                             io="output", dtype="float32", shape="B x N x H x W",
                             semantic="mask_logits", detail="per-slot mask predictions"),
            IRTensorPortSpec("slot_confidence", "Slot confidence",
                             io="output", dtype="float32", shape="B x N",
                             semantic="confidence_logits", detail="per-slot belief scores"),
        ]

    def ir_loss_terms(self) -> List[IRLossTermSpec]:
        return [
            IRLossTermSpec("stage0_speculative_loss", "Stage 0 speculative objective",
                           kind="vector+mask+selection+confidence",
                           optimizer_targets=["active_network_optimizer"],
                           source_ports=["slot_masks", "slot_confidence"],
                           detail="one-to-one matched vector+mask+selection+confidence losses"),
        ]

    def ir_state_inputs(self) -> List[IRStateSpec]:
        return []

    def execute(self, ctx: PipelineContext) -> None:
        self._run(ctx, self.cfg.stage0_epochs, "stage0_speculative", "stage0",
                  grad_accum_steps=self.cfg.stage0_grad_accum_steps)


class SpeculativeGestationNode(_SpeculativeBaseTrainNode):
    """Stage 1: train the speculative network on the gestation loader."""

    node_id = "stage_1_gestation"
    description = "Stage 1: speculative network training (gestation)"
    required_gates = ["gate_pregestation"]

    def ir_input_ports(self) -> List[IRTensorPortSpec]:
        return [
            IRTensorPortSpec("gestation_images", "Gestation images",
                             io="input", dtype="float32", shape="B x C x H x W",
                             semantic="image_batch", detail="bootstrap primitive render batch"),
        ]

    def ir_output_ports(self) -> List[IRTensorPortSpec]:
        return [
            IRTensorPortSpec("slot_masks", "Slot masks",
                             io="output", dtype="float32", shape="B x N x H x W",
                             semantic="mask_logits", detail="per-slot mask predictions"),
            IRTensorPortSpec("slot_confidence", "Slot confidence",
                             io="output", dtype="float32", shape="B x N",
                             semantic="confidence_logits", detail="per-slot belief scores"),
        ]

    def ir_loss_terms(self) -> List[IRLossTermSpec]:
        return [
            IRLossTermSpec("stage1_speculative_loss", "Stage 1 speculative objective",
                           kind="vector+mask+selection+confidence",
                           optimizer_targets=["active_network_optimizer"],
                           source_ports=["slot_masks", "slot_confidence"],
                           detail="one-to-one matched vector+mask+selection+confidence losses"),
        ]

    def ir_state_inputs(self) -> List[IRStateSpec]:
        return []

    def execute(self, ctx: PipelineContext) -> None:
        self._run(ctx, self.cfg.stage1_epochs, "stage1_speculative", "stage1",
                  grad_accum_steps=self.cfg.stage1_grad_accum_steps)


class SpeculativeBerkeleyNode(_SpeculativeBaseTrainNode):
    """Stage 2: train the speculative network on the Berkeley refresh loader."""

    node_id = "stage_2_berkeley"
    description = "Stage 2: speculative network training (Berkeley refresh)"
    required_gates = ["gate_gestation"]

    def ir_input_ports(self) -> List[IRTensorPortSpec]:
        return [
            IRTensorPortSpec("berkeley_images", "Berkeley images",
                             io="input", dtype="float32", shape="B x C x H x W",
                             semantic="image_batch", detail="Berkeley SBD + payload batch"),
        ]

    def ir_output_ports(self) -> List[IRTensorPortSpec]:
        return [
            IRTensorPortSpec("slot_masks", "Slot masks",
                             io="output", dtype="float32", shape="B x N x H x W",
                             semantic="mask_logits", detail="per-slot mask predictions"),
            IRTensorPortSpec("slot_confidence", "Slot confidence",
                             io="output", dtype="float32", shape="B x N",
                             semantic="confidence_logits", detail="per-slot belief scores"),
        ]

    def ir_loss_terms(self) -> List[IRLossTermSpec]:
        return [
            IRLossTermSpec("stage2_speculative_loss", "Stage 2 speculative objective",
                           kind="vector+mask+selection+confidence",
                           optimizer_targets=["active_network_optimizer"],
                           source_ports=["slot_masks", "slot_confidence"],
                           detail="one-to-one matched vector+mask+selection+confidence losses"),
        ]

    def ir_state_inputs(self) -> List[IRStateSpec]:
        return []

    def execute(self, ctx: PipelineContext) -> None:
        self._run(ctx, self.cfg.stage2_epochs, "stage2_speculative", "stage2",
                  grad_accum_steps=self.cfg.stage2_grad_accum_steps)
