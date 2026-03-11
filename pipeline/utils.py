"""
Pipeline utilities — shared primitives used across all node modules.

This module owns functions that are truly cross-cutting and not specific
to any single model or data source:

  * Logging
  * AMP / autocast helpers
  * Device resolution and CUDA diagnostics
  * DataLoader performance tuning
  * Checkpoint I/O (torch load/save, state-dict helpers)
  * Classifier head geometry (expand/align output dims)
  * Classifier supervision loss (BCE + semantic cosine)
  * Label query encoding and nearest-label lookup
  * Image ↔ tensor conversions (CHW ↔ RGB-U8)
  * Text overlay rendering and preview I/O
  * Render-config helpers (source shape, sync, chunk resolution)

Everything here is stateless and operates on values passed in.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log(msg: str):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# AMP / device helpers
# ---------------------------------------------------------------------------

def _resolve_amp_dtype(amp_dtype: str) -> torch.dtype:
    key = str(amp_dtype).strip().lower()
    if key in ("fp16", "float16", "half"):
        return torch.float16
    if key in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported amp dtype: {amp_dtype!r}")


def _autocast_context(device: torch.device, enabled: bool, amp_dtype_t: torch.dtype):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type=device.type, dtype=amp_dtype_t)
    return nullcontext()


def _cuda_device_supports_amp_dtype(device: torch.device, amp_dtype_t: torch.dtype) -> bool:
    if device.type != "cuda":
        return True
    if amp_dtype_t != torch.bfloat16:
        return True
    try:
        major, _minor = torch.cuda.get_device_capability(device)
        return int(major) >= 8
    except Exception:
        return False


def _is_cuda_backend_engine_error(exc: BaseException) -> bool:
    txt = str(exc).lower()
    return (
        "unable to find an engine to execute this computation" in txt
        or "unable to find an engine" in txt
        or "find was unable to find an engine" in txt
    )


def _effective_dataloader_num_workers(num_workers: int, device: Optional[torch.device] = None) -> int:
    workers = max(0, int(num_workers))
    if workers <= 0:
        return 0
    device_type = ""
    if device is not None:
        try:
            device_type = str(device.type).strip().lower()
        except Exception:
            device_type = str(device).strip().lower()
    if os.name == "nt" and device_type == "cuda":
        return 0
    return workers


def _dataloader_perf_kwargs(num_workers: int, persistent_workers: bool, prefetch_factor: int):
    if int(num_workers) <= 0:
        return {}
    out = {"persistent_workers": bool(persistent_workers)}
    if int(prefetch_factor) > 0:
        out["prefetch_factor"] = int(prefetch_factor)
    return out


def _resolve_aux_compute_device(primary_device: Optional[torch.device] = None) -> torch.device:
    return torch.device("cpu")


def _refresh_cache_is_staging_safe(
    cache_x: Optional[torch.Tensor],
    cache_y: Optional[torch.Tensor],
    cache_m: Optional[torch.Tensor],
    model_device: torch.device,
) -> bool:
    tensors = [cache_x, cache_y, cache_m]
    real = [t for t in tensors if isinstance(t, torch.Tensor)]
    if len(real) <= 0:
        return False
    for t in real:
        try:
            if bool(t.requires_grad):
                return False
        except Exception:
            return False
    devs = {str(t.device) for t in real}
    if len(devs) != 1:
        return False
    cache_dev = next(iter(devs))
    if cache_dev == str(model_device):
        return True
    return cache_dev == "cpu"


def _unwrap_module_for_replica(module: nn.Module) -> nn.Module:
    wrapped = getattr(module, "_orig_mod", None)
    if isinstance(wrapped, nn.Module):
        return wrapped
    return module


def _make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _module_device_type(module: nn.Module) -> str:
    try:
        return next(module.parameters()).device.type
    except StopIteration:
        pass
    except Exception:
        return "cpu"
    try:
        return next(module.buffers()).device.type
    except StopIteration:
        return "cpu"
    except Exception:
        return "cpu"


def _cuda_mem_diag(device: torch.device) -> str:
    if device.type != "cuda":
        return "device=cpu"
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    except Exception:
        free_bytes, total_bytes = 0, 0
    try:
        allocated = int(torch.cuda.memory_allocated(device))
    except Exception:
        allocated = 0
    try:
        reserved = int(torch.cuda.memory_reserved(device))
    except Exception:
        reserved = 0
    mib = float(1024 * 1024)
    return (
        f"device={str(device)} "
        f"free_mb={float(free_bytes) / mib:.1f} total_mb={float(total_bytes) / mib:.1f} "
        f"allocated_mb={float(allocated) / mib:.1f} reserved_mb={float(reserved) / mib:.1f}"
    )


# ---------------------------------------------------------------------------
# Classifier head geometry
# ---------------------------------------------------------------------------

def _classifier_output_dim(model: nn.Module) -> int:
    from wav_ml_models import TinyConvClassifier
    if isinstance(model, TinyConvClassifier):
        last = model.head[-1]
        if isinstance(last, nn.Linear):
            return int(last.out_features)
    fc = getattr(model, "fc", None)
    if isinstance(fc, nn.Linear):
        return int(fc.out_features)
    raise RuntimeError(f"Unsupported classifier head for output-dim query: {type(model).__name__}")


def _expand_classifier_outputs(model: nn.Module, new_num_classes: int) -> nn.Module:
    from wav_ml_models import TinyConvClassifier
    target = max(1, int(new_num_classes))
    old_n = int(_classifier_output_dim(model))
    if old_n == target:
        return model
    if old_n > target:
        raise RuntimeError(f"Cannot shrink classifier outputs from {old_n} to {target}.")

    if isinstance(model, TinyConvClassifier):
        old_fc = model.head[-1]
        if not isinstance(old_fc, nn.Linear):
            raise RuntimeError("TinyConvClassifier head is missing final Linear layer.")
        new_fc = nn.Linear(int(old_fc.in_features), int(target))
        new_fc = new_fc.to(device=old_fc.weight.device, dtype=old_fc.weight.dtype)
        with torch.no_grad():
            new_fc.weight.zero_()
            if new_fc.bias is not None:
                new_fc.bias.zero_()
            new_fc.weight[:old_n].copy_(old_fc.weight.detach())
            if (new_fc.bias is not None) and (old_fc.bias is not None):
                new_fc.bias[:old_n].copy_(old_fc.bias.detach())
            if new_fc.bias is not None and old_n < target:
                new_fc.bias[old_n:].fill_(-4.0)
        model.head[-1] = new_fc
        return model

    fc = getattr(model, "fc", None)
    if isinstance(fc, nn.Linear):
        new_fc = nn.Linear(int(fc.in_features), int(target))
        new_fc = new_fc.to(device=fc.weight.device, dtype=fc.weight.dtype)
        with torch.no_grad():
            new_fc.weight.zero_()
            if new_fc.bias is not None:
                new_fc.bias.zero_()
            new_fc.weight[:old_n].copy_(fc.weight.detach())
            if (new_fc.bias is not None) and (fc.bias is not None):
                new_fc.bias[:old_n].copy_(fc.bias.detach())
            if new_fc.bias is not None and old_n < target:
                new_fc.bias[old_n:].fill_(-4.0)
        model.fc = new_fc
        return model

    raise RuntimeError(f"Unsupported classifier head for output expansion: {type(model).__name__}")


def _align_multilabel_targets_dim(y: torch.Tensor, out_dim: int) -> torch.Tensor:
    if y.ndim != 2:
        raise ValueError(f"Expected multilabel target tensor [B,C], got shape={tuple(y.shape)}")
    want = max(1, int(out_dim))
    got = int(y.shape[1])
    if got == want:
        return y
    if got < want:
        pad = torch.zeros((int(y.shape[0]), int(want - got)), device=y.device, dtype=y.dtype)
        return torch.cat([y, pad], dim=1)
    return y[:, :want]


def _set_feature_freeze(model, freeze: bool):
    for name, p in model.named_parameters():
        if name.startswith("features."):
            p.requires_grad_(not freeze)


def _make_classifier_from_ckpt_meta(
    model_name: str,
    num_classes: int,
    classifier_base_ch: int = 64,
    classifier_max_ch: int = 384,
    classifier_context_blocks: int = 8,
    classifier_context_dropout: float = 0.05,
):
    from wav_ml_models import TinyConvClassifier
    if model_name == "tiny":
        return TinyConvClassifier(
            num_classes=num_classes,
            base_ch=max(16, int(classifier_base_ch)),
            max_ch=max(32, int(classifier_max_ch)),
            context_blocks=max(0, int(classifier_context_blocks)),
            context_dropout=float(classifier_context_dropout),
            mask_decoder_channels=max(32, int(max(16, int(classifier_base_ch)) // 2)),
        )
    if model_name == "resnet18":
        from torchvision.models import resnet18
        m = resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    raise RuntimeError(f"Unsupported classifier model in checkpoint: {model_name}")


# ---------------------------------------------------------------------------
# Classifier supervision loss
# ---------------------------------------------------------------------------

CLASSIFIER_LOSS_SCALE = 1.0
CLASSIFIER_SEMANTIC_COSINE_WEIGHT = 0.35


def _build_semantic_supervision_targets(
    classifier: nn.Module,
    y_multihot: torch.Tensor,
    supervised_classes: int,
    semantic_target_temperature: float = 8.0,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    from wav_ml_models import TinyConvClassifier
    _ = semantic_target_temperature
    if not isinstance(classifier, TinyConvClassifier):
        return None
    if not bool(int(classifier.label_embed_enabled.item())):
        return None
    bank = classifier.label_embed_bank
    if bank is None or int(bank.ndim) != 2 or int(bank.shape[0]) <= 0 or int(bank.shape[1]) <= 0:
        return None
    sup = max(1, int(supervised_classes))
    y_sup = _align_multilabel_targets_dim(y_multihot, out_dim=int(sup)).to(dtype=torch.float32)
    bank_t = bank.to(device=y_sup.device, dtype=torch.float32)
    if int(bank_t.shape[0]) < int(sup):
        return None
    y_sem = _align_multilabel_targets_dim(y_multihot, out_dim=int(bank_t.shape[0])).to(dtype=torch.float32)
    y_sem = torch.clamp(y_sem, 0.0, 1.0)
    return y_sem, y_sup


def _embed_multihot_distribution(weights: torch.Tensor, bank_norm: torch.Tensor) -> torch.Tensor:
    emb = weights @ bank_norm
    emb_norm = torch.linalg.norm(emb, dim=1, keepdim=True)
    if bool(torch.any(emb_norm <= 1e-6).item()):
        prior = bank_norm.mean(dim=0, keepdim=True)
        fill_mask = (emb_norm <= 1e-6).to(dtype=torch.float32)
        emb = (emb * (1.0 - fill_mask)) + (prior * fill_mask)
        emb_norm = torch.linalg.norm(emb, dim=1, keepdim=True)
    return emb / torch.clamp(emb_norm, min=1e-6)


def _classifier_supervision_loss(
    classifier: nn.Module,
    logits: torch.Tensor,
    y_multihot: torch.Tensor,
    supervised_dim: int,
    semantic_cosine_weight: float = CLASSIFIER_SEMANTIC_COSINE_WEIGHT,
    semantic_soft_target_max: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    from wav_ml_models import TinyConvClassifier
    dim = max(1, int(supervised_dim))
    logits_sup = logits
    if int(logits_sup.shape[1]) > int(dim):
        logits_sup = logits_sup[:, : int(dim)]
    y_sup = _align_multilabel_targets_dim(y_multihot, out_dim=int(logits_sup.shape[1])).to(dtype=torch.float32)
    logits_sup_f = logits_sup.to(dtype=torch.float32)

    use_geom = False
    cos_loss = torch.zeros((), device=logits_sup_f.device, dtype=torch.float32)
    w_geom = float(max(0.0, semantic_cosine_weight))
    soft_max = float(max(0.0, semantic_soft_target_max))
    y_eff = y_sup
    if (
        (w_geom > 0.0 or soft_max > 0.0)
        and isinstance(classifier, TinyConvClassifier)
        and bool(int(classifier.label_embed_enabled.item()))
        and int(classifier.label_embed_bank.ndim) == 2
        and int(classifier.label_embed_bank.shape[0]) >= int(logits_sup_f.shape[1])
        and int(classifier.label_embed_bank.shape[1]) > 0
    ):
        bank = classifier.label_embed_bank[: int(logits_sup_f.shape[1]), :].to(device=logits_sup_f.device, dtype=torch.float32)
        bank = F.normalize(bank, dim=1, eps=1e-6)
        pred_probs = torch.sigmoid(logits_sup_f)
        pred_emb = _embed_multihot_distribution(weights=pred_probs, bank_norm=bank)
        tgt_emb = _embed_multihot_distribution(weights=y_sup, bank_norm=bank)
        if soft_max > 0.0:
            sim = torch.mm(tgt_emb, bank.t())
            neg_mask = y_sup < 0.5
            y_eff = torch.where(neg_mask, torch.clamp(sim, min=0.0, max=soft_max), y_sup)
        if w_geom > 0.0:
            cos_loss = (1.0 - torch.sum(pred_emb * tgt_emb, dim=1)).mean()
            use_geom = True

    bce = F.binary_cross_entropy_with_logits(logits_sup_f, y_eff, reduction="mean")
    total = bce + (float(w_geom) * cos_loss if use_geom else 0.0)
    return total, logits_sup, y_eff, {
        "bce": float(bce.detach().item()),
        "cos": float(cos_loss.detach().item()) if use_geom else 0.0,
        "used_geometry": bool(use_geom),
        "geometry_weight": float(w_geom if use_geom else 0.0),
    }


# ---------------------------------------------------------------------------
# Default class names (used by classifier init and vocab bootstrap)
# ---------------------------------------------------------------------------

def _default_class_names() -> List[str]:
    """Default semantic class vocabulary.

    This is the canonical starting vocabulary for new classifiers. It is
    intentionally not Berkeley-specific — it covers perceptual primitives,
    damage types, cardinal concepts, colour terms, and noise profiles.
    """
    class_names = [
        "noise", "pattern", "signal", "silence",
        "mixed noise and signal", "berkeley sbd dataset",
        "blur damage", "noise damage", "dropout damage",
        "quantization damage", "stride skew damage",
        "pulse", "rhythm", "change", "stable",
        "bright", "dark", "loud", "quiet", "smooth", "rough",
        "near", "far", "inside", "outside",
        "up", "down", "left", "right", "front", "back",
        "self", "other", "body", "face", "eye", "ear", "mouth", "hand",
        "touch", "voice", "breath", "pain", "comfort",
        "hunger", "thirst", "alert", "danger", "safety",
        "memory", "familiar", "unfamiliar", "attention", "surprise",
        "zero", "one", "two", "three", "four",
        "five", "six", "seven", "eight", "nine",
        "image", "sound", "motion", "edge", "shape", "texture",
        "low band", "mid band", "high band", "broadband", "narrowband",
        "burst", "echo", "object", "layout", "mask", "composite",
        "red", "green", "blue", "yellow", "cyan", "magenta",
        "brown", "black", "white", "gray", "grey",
        "white noise", "pink noise", "brown noise", "red noise",
        "blue noise", "violet noise", "gray noise",
        "gaussian white noise", "uniform white noise",
    ]
    return [str(x).strip() for x in class_names if str(x).strip()]


# Backward-compat alias
_default_berkeley_class_names = _default_class_names


def _load_classifier_for_metric(
    ckpt_path: str,
    device: torch.device,
    classifier_base_ch: int = 64,
    classifier_max_ch: int = 384,
    classifier_context_blocks: int = 8,
    classifier_context_dropout: float = 0.05,
    classifier_init_required: bool = False,
):
    """Load a classifier checkpoint for metric evaluation.

    Works with any checkpoint that stores class_names / model_name / state_dict.
    Falls back to a scratch classifier built from the default vocabulary when
    no checkpoint is available and *classifier_init_required* is False.
    """
    from pipeline.nodes.base import _torch_load_cpu, _strip_module_prefix, _extract_state_dict, _apply_state_dict

    def _build_scratch_classifier(fallback_reason: str):
        class_names = _default_class_names()
        num_classes = max(2, int(len(class_names)))
        model_name = "tiny"
        model = _make_classifier_from_ckpt_meta(
            model_name=model_name,
            num_classes=num_classes,
            classifier_base_ch=classifier_base_ch,
            classifier_max_ch=classifier_max_ch,
            classifier_context_blocks=classifier_context_blocks,
            classifier_context_dropout=classifier_context_dropout,
        )
        model = model.to(device)
        model.eval()
        info = {
            "path": "",
            "model_name": model_name,
            "num_classes": int(num_classes),
            "class_names": list(class_names),
            "loaded_keys": 0,
            "partial_keys": 0,
            "skipped_keys": 0,
            "missing_after_load": 0,
            "unexpected_after_load": 0,
            "init_mode": "scratch_fallback",
            "fallback_reason": str(fallback_reason),
        }
        return model, info

    path_raw = str(ckpt_path).strip()
    if not path_raw:
        if bool(classifier_init_required):
            raise RuntimeError(
                "Metric mode requires --classifier-init-ckpt when --classifier-init-required is set."
            )
        return _build_scratch_classifier("checkpoint_not_provided")

    p = Path(path_raw)
    if not p.exists():
        if bool(classifier_init_required):
            raise FileNotFoundError(f"Classifier checkpoint not found: {p}")
        return _build_scratch_classifier(f"checkpoint_missing:{path_raw}")

    try:
        blob = _torch_load_cpu(str(p))
    except Exception as e:
        if bool(classifier_init_required):
            raise
        return _build_scratch_classifier(f"checkpoint_unloadable:{type(e).__name__}:{e}")

    model_name = str(blob.get("model_name", "tiny"))
    class_names_raw = blob.get("class_names", [])
    if isinstance(class_names_raw, (list, tuple)):
        class_names = [str(x) for x in class_names_raw]
    else:
        class_names = []
    num_classes = int(blob.get("num_classes", len(class_names) if len(class_names) > 0 else 20))
    model = _make_classifier_from_ckpt_meta(
        model_name=model_name,
        num_classes=num_classes,
        classifier_base_ch=classifier_base_ch,
        classifier_max_ch=classifier_max_ch,
        classifier_context_blocks=classifier_context_blocks,
        classifier_context_dropout=classifier_context_dropout,
    )
    state = _strip_module_prefix(_extract_state_dict(blob))
    load_info = _apply_state_dict(
        model=model,
        src_state=state,
        source_name=f"classifier_metric_ckpt:{p}",
    )
    model = model.to(device)
    model.eval()

    info = {
        "path": str(p.resolve()),
        "model_name": model_name,
        "num_classes": num_classes,
        "class_names": class_names,
        "loaded_keys": int(load_info.get("loaded_keys", 0)),
        "partial_keys": int(load_info.get("partial_keys", 0)),
        "skipped_keys": int(load_info.get("skipped_keys", 0)),
        "missing_after_load": int(load_info.get("missing_after_load", 0)),
        "unexpected_after_load": int(load_info.get("unexpected_after_load", 0)),
        "init_mode": "checkpoint",
        "fallback_reason": "",
    }
    return model, info


# Backward-compat alias
_load_berkeley_classifier_for_metric = _load_classifier_for_metric


# ---------------------------------------------------------------------------
# Label query encoding and nearest-label lookup
# ---------------------------------------------------------------------------

def _encode_query_texts_for_bank(query_texts: Sequence[str], bank_info: Dict[str, Any], args, device: torch.device) -> np.ndarray:
    from pipeline.nodes.label_embedding_node import _encode_texts_sentence_transformers, _normalize_l2_rows_np
    if len(query_texts) <= 0:
        return np.zeros((0, max(1, int(bank_info.get("dim", 1)))), dtype=np.float32)
    backend = str(bank_info.get("backend_used", "")).strip().lower()
    if backend != "sentence_transformers":
        raise RuntimeError(
            "Only sentence-transformers semantic vectors are permitted for query encoding; "
            f"bank backend is {backend!r}."
        )
    emb = _encode_texts_sentence_transformers(
        query_texts,
        model_name=str(bank_info.get("model_name", str(args.label_embedding_model))),
        device=device,
    )
    bank_dim = max(1, int(bank_info.get("dim", emb.shape[1])))
    if int(emb.shape[1]) != bank_dim:
        raise RuntimeError(
            "Sentence-transformer geometry is configured as strict; "
            f"query dim ({int(emb.shape[1])}) must equal bank dim ({int(bank_dim)})."
        )
    return _normalize_l2_rows_np(emb.astype(np.float32, copy=False))


def _nearest_labels_for_queries(
    query_texts: Sequence[str],
    label_bank_np: np.ndarray,
    class_names: Sequence[str],
    label_texts: Sequence[str],
    topk: int,
    query_emb_np: np.ndarray,
) -> List[Dict[str, Any]]:
    from pipeline.nodes.label_embedding_node import _normalize_l2_rows_np
    if len(query_texts) <= 0:
        return []
    bank_arr = np.asarray(label_bank_np, dtype=np.float32)
    q_arr = np.asarray(query_emb_np, dtype=np.float32)
    if bank_arr.ndim != 2 or q_arr.ndim != 2:
        reason = f"shape_mismatch: query_shape={tuple(q_arr.shape)} label_shape={tuple(bank_arr.shape)}"
        return [{"query": str(qt), "matches": [], "reason": reason} for qt in query_texts]

    bank = _normalize_l2_rows_np(bank_arr)
    q = _normalize_l2_rows_np(q_arr)
    alignment_note = ""

    if int(q.shape[1]) != int(bank.shape[1]):
        if int(bank.shape[0]) == int(q.shape[1]) and int(bank.shape[1]) != int(q.shape[1]):
            bank = _normalize_l2_rows_np(bank.transpose(1, 0).astype(np.float32, copy=False))
            alignment_note = "label_bank_transposed"
            _log("[label-query] detected transposed label bank; auto-corrected for similarity lookup.")

    if int(q.shape[1]) != int(bank.shape[1]):
        reason = f"dim_mismatch: query_dim={int(q.shape[1])} label_dim={int(bank.shape[1])}"
        _log(f"[label-query] skipped nearest-label lookup ({reason}).")
        rows: List[Dict[str, Any]] = []
        for qt in query_texts:
            row = {"query": str(qt), "matches": [], "reason": reason}
            if alignment_note:
                row["alignment"] = alignment_note
            rows.append(row)
        return rows

    if int(bank.shape[0]) <= 0:
        return [{"query": str(qt), "matches": [], "reason": "empty_label_bank"} for qt in query_texts]

    sims = q @ bank.transpose(1, 0)
    k = max(1, min(int(topk), int(bank.shape[0])))
    rows: List[Dict[str, Any]] = []
    for i, qt in enumerate(query_texts):
        idx = np.argsort(-sims[i])[:k]
        hits: List[Dict[str, Any]] = []
        for j in idx.tolist():
            c = int(j)
            hits.append(
                {
                    "class_idx": c,
                    "class_name": str(class_names[c]) if c < len(class_names) else f"class_{c}",
                    "label_text": str(label_texts[c]) if c < len(label_texts) else "",
                    "score": float(sims[i, c]),
                }
            )
        row = {"query": str(qt), "matches": hits}
        if alignment_note:
            row["alignment"] = alignment_note
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Checkpoint model helpers (classifier-specific)
# ---------------------------------------------------------------------------

def _extract_deskew_prefilter_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    from pipeline.nodes.base import _strip_module_prefix
    src_state = _strip_module_prefix(model.state_dict())
    out: Dict[str, torch.Tensor] = {}
    for k, v in src_state.items():
        key = str(k)
        if key.startswith("aux_filter_bundles.") or ("prefilter_" in key) or ("post_residual_" in key):
            out[key] = v.detach().cpu()
    return out


def _parse_transformer_filter_bundle_names(raw: Any) -> List[str]:
    text = str(raw).strip()
    if not text:
        return []
    out: List[str] = []
    seen: set = set()
    for token in text.split(","):
        key = str(token).strip().lower()
        if (not key) or (key in ("none", "off", "0")):
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _snapshot_classifier_lora(model: Optional[nn.Module]) -> Dict[str, Any]:
    from wav_ml_models import TinyConvClassifier, tiny_classifier_lora_snapshot
    target = model
    if not isinstance(target, TinyConvClassifier):
        wrapped = getattr(model, "_orig_mod", None)
        if isinstance(wrapped, TinyConvClassifier):
            target = wrapped
    if not isinstance(target, TinyConvClassifier):
        return {"installed": False, "slot_names": []}
    return tiny_classifier_lora_snapshot(target)


def _restore_classifier_lora_from_blob(model: nn.Module, blob: Any, source_name: str) -> Dict[str, Any]:
    from wav_ml_models import TinyConvClassifier, restore_tiny_classifier_lora_snapshot
    target = model
    if not isinstance(target, TinyConvClassifier):
        wrapped = getattr(model, "_orig_mod", None)
        if isinstance(wrapped, TinyConvClassifier):
            target = wrapped
    if not isinstance(target, TinyConvClassifier):
        return {"used": False, "installed": False, "reason": f"unsupported_model:{type(model).__name__}", "source": source_name}
    snapshot = None
    if isinstance(blob, dict) and isinstance(blob.get("classifier_lora"), dict):
        snapshot = dict(blob.get("classifier_lora"))
    info = restore_tiny_classifier_lora_snapshot(target, snapshot)
    if not isinstance(info, dict):
        info = {"used": False, "installed": False, "reason": "invalid_restore_result"}
    info["source"] = str(source_name)
    return info


# ---------------------------------------------------------------------------
# Image ↔ tensor conversions
# ---------------------------------------------------------------------------

def _tensor_chw_to_rgb_u8(img: torch.Tensor) -> np.ndarray:
    x = img.detach().to(torch.float32)
    if x.ndim == 4:
        x = x[0]
    if x.ndim != 3:
        raise ValueError(f"Expected CHW image tensor, got shape={tuple(x.shape)}")
    if int(x.shape[0]) == 1:
        x = x.repeat(3, 1, 1)
    x = torch.clamp(x, 0.0, 1.0)
    arr = x.cpu().numpy().astype(np.float32, copy=False)
    arr = np.transpose(arr, (1, 2, 0))
    return np.round(arr * 255.0).astype(np.uint8)


def _rgb_u8_to_tensor_chw01(img_u8: np.ndarray) -> torch.Tensor:
    arr = np.asarray(img_u8, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3:
        raise ValueError(f"Expected HWC image array, got shape={tuple(arr.shape)}")
    if int(arr.shape[2]) == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif int(arr.shape[2]) > 3:
        arr = arr[:, :, :3]
    t = torch.from_numpy(arr.astype(np.float32, copy=False) / 255.0)
    return t.permute(2, 0, 1).contiguous()


def _overlay_text_rows_u8(image_u8: np.ndarray, header: str, rows: Sequence[str]) -> np.ndarray:
    out = np.asarray(image_u8, dtype=np.uint8).copy()
    text_rows: List[str] = []
    if str(header).strip():
        text_rows.append(str(header).strip())
    for r in rows:
        txt = str(r).strip()
        if txt:
            text_rows.append(txt)
    if len(text_rows) <= 0:
        return out
    try:
        from PIL import Image, ImageDraw, ImageFont
        im = Image.fromarray(out).convert("RGBA")
        font = ImageFont.load_default()
        pad = 2
        line_h = 11
        h = int(im.height)
        w = int(im.width)
        max_lines_per_col = max(1, (h - (pad * 2)) // line_h)
        n_cols = max(1, int(math.ceil(float(len(text_rows)) / float(max_lines_per_col))))
        col_w = max(1, w // n_cols)
        box_h = min(h, (pad * 2) + (line_h * max_lines_per_col))
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw_bg = ImageDraw.Draw(overlay)
        draw_bg.rectangle([(0, 0), (w - 1, box_h)], fill=(0, 0, 0, 110))
        im = Image.alpha_composite(im, overlay)
        draw = ImageDraw.Draw(im)
        for i, txt in enumerate(text_rows):
            col = int(i // max_lines_per_col)
            row = int(i % max_lines_per_col)
            x = int(pad + (col * col_w))
            y = int(pad + (row * line_h))
            if x >= (w - pad):
                break
            max_chars = max(16, int((col_w - (pad * 2)) // 5))
            color = (255, 225, 64, 255) if i == 0 else (240, 240, 240, 255)
            draw.text((x, y), str(txt)[:max_chars], fill=color, font=font)
        return np.asarray(im.convert("RGB"), dtype=np.uint8)
    except Exception:
        shade_h = min(int(out.shape[0]), 18 + (10 * len(text_rows)))
        out[:shade_h, :, :] = (out[:shade_h, :, :].astype(np.uint16) * 65 // 100).astype(np.uint8)
        return out


def _annotate_tensor_image(img: torch.Tensor, header: str, rows: Sequence[str]) -> torch.Tensor:
    base = _tensor_chw_to_rgb_u8(img)
    anno = _overlay_text_rows_u8(base, header=header, rows=rows)
    return _rgb_u8_to_tensor_chw01(anno)


def _concat_rgb_panels_h(images: Sequence[np.ndarray], pad: int = 3, fill: int = 0) -> np.ndarray:
    if len(images) <= 0:
        return np.full((8, 8, 3), int(fill) & 0xFF, dtype=np.uint8)
    imgs = [np.asarray(im, dtype=np.uint8) for im in images]
    h = max(int(im.shape[0]) for im in imgs)
    w = sum(int(im.shape[1]) for im in imgs) + (max(0, len(imgs) - 1) * int(max(0, pad)))
    out = np.full((int(h), int(w), 3), int(fill) & 0xFF, dtype=np.uint8)
    x = 0
    for i, im in enumerate(imgs):
        ih, iw = int(im.shape[0]), int(im.shape[1])
        y0 = max(0, (h - ih) // 2)
        out[y0 : y0 + ih, x : x + iw, :] = im
        x += iw
        if i + 1 < len(imgs):
            x += int(max(0, pad))
    return out


def _write_rgb_preview(path: Path, image_u8: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
        Image.fromarray(image_u8).save(str(path))
        return
    except Exception:
        pass
    ppm_path = path.with_suffix(".ppm")
    h, w = int(image_u8.shape[0]), int(image_u8.shape[1])
    with ppm_path.open("wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(image_u8.astype(np.uint8, copy=False).tobytes())


def _top_labels_from_probs(probs: np.ndarray, class_names: Sequence[str], topk: int) -> List[Dict]:
    if probs.size <= 0:
        return []
    k = max(1, min(int(topk), int(probs.size)))
    idx = np.argsort(-probs)[:k]
    out: List[Dict] = []
    for cls_idx in idx.tolist():
        c = int(cls_idx)
        name = str(class_names[c]) if c < len(class_names) else f"class_{c}"
        out.append({"class_idx": c, "class_name": name, "score": float(probs[c])})
    return out


def _format_top_label_lines(rows: Sequence[Dict], max_items: int = 3) -> List[str]:
    out: List[str] = []
    k = max(0, int(max_items))
    rows_list = list(rows)
    use_rows = rows_list[:k] if k > 0 else rows_list
    for r in use_rows:
        out.append(f"{r.get('class_name', '?')}:{float(r.get('score', 0.0)):.3f}")
    return out


def _semantic_target_entries_from_vector(
    cond_vec: Any,
    class_names: Sequence[str],
    max_items: int = 8,
    threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    arr = np.asarray(cond_vec, dtype=np.float32).reshape(-1)
    if int(arr.size) <= 0:
        return []
    t = float(threshold)
    idx = np.where(arr >= t)[0].astype(np.int64)
    if int(idx.size) <= 0:
        return []
    order = idx[np.argsort(-arr[idx])]
    k = max(1, int(max_items))
    out: List[Dict[str, Any]] = []
    for cls_idx in order[:k].tolist():
        c = int(cls_idx)
        out.append({
            "class_idx": c,
            "class_name": str(class_names[c]) if c < len(class_names) else f"class_{c}",
            "weight": float(arr[c]),
        })
    return out


def _format_target_line_from_condition(
    cond_vec: Any,
    class_names: Sequence[str],
    max_items: int = 4,
    threshold: float = 0.5,
) -> str:
    if isinstance(cond_vec, torch.Tensor):
        arr = cond_vec.detach().to(torch.float32).cpu().numpy().reshape(-1)
    else:
        arr = np.asarray(cond_vec, dtype=np.float32).reshape(-1)
    if int(arr.size) <= 0:
        return "target(0):none"
    t = float(threshold)
    idx = np.where(arr >= t)[0].astype(np.int64)
    if int(idx.size) <= 0:
        return "target(0):none"
    order = idx[np.argsort(-arr[idx])]
    k = int(order.size) if int(max_items) <= 0 else max(1, int(max_items))
    picks = order[:k].tolist()
    names = [str(class_names[int(i)]) if int(i) < len(class_names) else f"class_{int(i)}" for i in picks]
    more = f"+{int(order.size) - k}" if int(order.size) > k else ""
    return f"target({int(order.size)}):{','.join(names)}{more}"


# ---------------------------------------------------------------------------
# Render-config helpers
# ---------------------------------------------------------------------------

def _render_source_shape_for_stream(stream_len: int, cfg):
    from wav_ml_core import COLOR_MODE_MAP, COLOR_MODES
    downsample = max(1, int(cfg.downsample))
    width = max(1, int(cfg.width))
    base_len = max(1, (max(1, int(stream_len)) + downsample - 1) // downsample)
    if int(cfg.max_points) > 0:
        base_len = min(base_len, int(cfg.max_points))
    _, mapping = COLOR_MODE_MAP.get(cfg.color_mode, COLOR_MODE_MAP[COLOR_MODES[0][0]])
    special = bool(mapping.get("__special__") == "single_source_rgb_stride")
    max_len = (base_len + 2) // 3 if special else base_len
    total = int(math.ceil(float(max_len) / float(width)) * width)
    height = max(1, total // width)
    return {
        "downsample": int(downsample),
        "width": int(width),
        "base_len": int(base_len),
        "max_len": int(max_len),
        "total": int(total),
        "height": int(height),
        "mapping": mapping,
        "special": special,
    }


def _sync_render_width_with_embed_hw(cfg, image_hw: Tuple[int, int]) -> bool:
    target_w = max(1, int(image_hw[1]))
    if int(cfg.width) == int(target_w):
        return False
    cfg.width = int(target_w)
    return True


def _resolve_synced_chunk_samples(
    requested_chunk_samples: int,
    patch_size: int,
    cfg,
    image_hw: Tuple[int, int],
) -> Tuple[int, Dict[str, Any]]:
    from wav_ml_core import COLOR_MODE_MAP, COLOR_MODES
    requested = max(1, int(requested_chunk_samples))
    patch = max(1, int(patch_size))
    target_h = max(1, int(image_hw[0]))
    target_w = max(1, int(image_hw[1]))
    downsample = max(1, int(cfg.downsample))
    width = max(1, int(cfg.width))
    _, mapping = COLOR_MODE_MAP.get(cfg.color_mode, COLOR_MODE_MAP[COLOR_MODES[0][0]])
    special = bool(mapping.get("__special__") == "single_source_rgb_stride")
    if int(target_w) != int(width):
        cfg.width = int(target_w)
        width = int(target_w)
    aligned_h = max(patch, ((target_h + patch - 1) // patch) * patch)
    pixels = int(aligned_h) * int(width)
    if special:
        base_len = int(pixels) * 3
    else:
        base_len = int(pixels)
    chunk = int(base_len) * int(downsample)
    info = {
        "requested": int(requested),
        "resolved": int(chunk),
        "target_h": int(target_h),
        "aligned_h": int(aligned_h),
        "image_hw": [int(target_h), int(target_w)],
        "downsample": int(downsample),
        "special": bool(special),
        "patch_size": int(patch),
    }
    return int(chunk), info


def _resolve_shared_embed_image_size(args) -> Tuple[int, int]:
    h = max(8, int(getattr(args, "enforce_render_image_h", 0) or getattr(args, "image_size", 64)))
    w = max(8, int(getattr(args, "enforce_render_width", 0) or h))
    return (h, w)


def _build_enforced_render_config(args):
    from wav_ml_core import COLOR_MODE_MAP, COLOR_MODES, RenderConfig
    color_mode = str(args.enforce_render_color_mode)
    if color_mode not in COLOR_MODE_MAP:
        color_mode = COLOR_MODES[0][0]
    return RenderConfig(
        bitmode=str(args.enforce_render_bitmode),
        use_channels=str(args.enforce_render_use_channels),
        mono_pick=str(args.enforce_render_mono_pick),
        width=max(1, int(args.enforce_render_width)),
        downsample=max(1, int(args.enforce_render_downsample)),
        color_mode=color_mode,
        empty_fill=max(0, min(255, int(args.enforce_render_empty_fill))),
        bitmask_enable=bool(args.enforce_render_bitmask_enable),
        bitmask_low=int(args.enforce_render_bitmask_low),
        bitmask_high=int(args.enforce_render_bitmask_high),
        max_points=max(1024, int(args.enforce_render_max_points)),
    )


# ---------------------------------------------------------------------------
# Rendering and scoring helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _render_and_score_wave(
    wave_f32: np.ndarray,
    cfg,
    sample_bits: int,
    image_hw: Tuple[int, int],
    classifier: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: str,
    channels_last: bool,
):
    from wav_ml_core import render_mono_wave_to_tensor
    xb = torch.from_numpy(np.asarray(wave_f32, dtype=np.float32)[None, :]).to(device)
    amp_dtype_t = _resolve_amp_dtype(amp_dtype) if amp_enabled else torch.float16
    with _autocast_context(device=device, enabled=amp_enabled, amp_dtype_t=amp_dtype_t):
        img = render_mono_wave_to_tensor(xb, cfg=cfg, image_hw=image_hw, sample_bits=int(sample_bits))
        if channels_last:
            img = img.contiguous(memory_format=torch.channels_last)
        logits = classifier(img)
        probs = torch.sigmoid(logits[0]).to(torch.float32)
    return _tensor_chw_to_rgb_u8(img[0]), probs.detach().cpu().numpy().astype(np.float32, copy=False)


def _save_round_supervision_preview(
    preview_dir: Path,
    stage: str,
    cycle_id: int,
    round_id: int,
    sample_idx: int,
    clean_wave: np.ndarray,
    enhanced_wave: np.ndarray,
    cfg,
    sample_bits: int,
    image_hw: Tuple[int, int],
    classifier: nn.Module,
    class_names: Sequence[str],
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: str,
    channels_last: bool,
    topk: int,
    target_condition: Optional[Any] = None,
    extra: Optional[Dict] = None,
):
    clean_img, clean_probs = _render_and_score_wave(
        wave_f32=clean_wave, cfg=cfg, sample_bits=int(sample_bits),
        image_hw=image_hw, classifier=classifier, device=device,
        amp_enabled=amp_enabled, amp_dtype=amp_dtype, channels_last=channels_last,
    )
    out_img, out_probs = _render_and_score_wave(
        wave_f32=enhanced_wave, cfg=cfg, sample_bits=int(sample_bits),
        image_hw=image_hw, classifier=classifier, device=device,
        amp_enabled=amp_enabled, amp_dtype=amp_dtype, channels_last=channels_last,
    )
    panel = _concat_rgb_panels_h([clean_img, out_img], pad=4, fill=0)
    base = f"cycle_{int(cycle_id):04d}_round_{int(round_id):04d}_stage_{str(stage).lower()}_sample_{int(sample_idx):03d}"
    panel_path = preview_dir / f"{base}.png"
    meta_path = preview_dir / f"{base}.json"
    _write_rgb_preview(panel_path, panel)
    all_topk = max(1, min(int(clean_probs.size), len(class_names)))
    top_clean = _top_labels_from_probs(clean_probs, class_names=class_names, topk=int(all_topk))
    top_out = _top_labels_from_probs(out_probs, class_names=class_names, topk=int(all_topk))
    top_delta = _top_labels_from_probs((out_probs - clean_probs), class_names=class_names, topk=int(all_topk))
    payload: Dict = {
        "cycle": int(cycle_id), "round": int(round_id), "stage": str(stage),
        "sample_idx": int(sample_idx),
        "target_semantic": (
            _semantic_target_entries_from_vector(
                target_condition, class_names=class_names,
                max_items=max(1, len(class_names)), threshold=0.5,
            ) if target_condition is not None else []
        ),
        "top_labels_clean": top_clean, "top_labels_enhanced": top_out,
        "top_label_deltas": top_delta, "panel": str(panel_path),
    }
    if isinstance(extra, dict):
        payload["extra"] = dict(extra)
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


@torch.no_grad()
def _save_generator_supervision_preview(
    preview_dir: Path,
    cycle_id: int,
    round_id: int,
    generator: nn.Module,
    classifier: nn.Module,
    payload_conditions: Sequence[Any],
    num_classes: int,
    class_names: Sequence[str],
    image_hw: Tuple[int, int],
    z_dim: int,
    samples: int,
    topk: int,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: str,
    channels_last: bool,
    seed: int,
):
    from pipeline.nodes.data_nodes import _payload_condition_bank_tensor
    if len(payload_conditions) <= 0:
        return None
    cond_bank_cpu = _payload_condition_bank_tensor(
        payload_targets=payload_conditions, num_classes=int(num_classes),
    )
    if int(cond_bank_cpu.shape[0]) <= 0:
        return None
    n = max(1, int(samples))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, int(cond_bank_cpu.shape[0]), size=n)
    idx_t = torch.as_tensor(idx, device=device, dtype=torch.long)
    cond_bank = cond_bank_cpu.to(device=device, dtype=torch.float32)
    cond = cond_bank.index_select(0, idx_t).to(device=device, dtype=torch.float32)
    z = torch.randn((int(cond.shape[0]), max(8, int(z_dim))), device=device)
    was_gen_train = bool(generator.training)
    was_cls_train = bool(classifier.training)
    generator.eval()
    classifier.eval()
    amp_dtype_t = _resolve_amp_dtype(amp_dtype) if amp_enabled else torch.float16
    with _autocast_context(device=device, enabled=amp_enabled, amp_dtype_t=amp_dtype_t):
        fake = generator(z, cond).to(torch.float32)
        if channels_last:
            fake = fake.contiguous(memory_format=torch.channels_last)
        logits = classifier(fake)
        probs = torch.sigmoid(logits).to(torch.float32)
    if was_gen_train:
        generator.train()
    if was_cls_train:
        classifier.train()
    images = [_tensor_chw_to_rgb_u8(fake[i]) for i in range(int(fake.shape[0]))]
    panel = _concat_rgb_panels_h(images, pad=3, fill=0)
    base = f"cycle_{int(cycle_id):04d}_round_{int(round_id):04d}_stage_g_samples"
    panel_path = preview_dir / f"{base}.png"
    meta_path = preview_dir / f"{base}.json"
    _write_rgb_preview(panel_path, panel)
    rows_list: List[Dict] = []
    topk_all = max(1, min(len(class_names), int(probs.shape[1])))
    for i in range(int(fake.shape[0])):
        p = probs[i].detach().cpu().numpy().astype(np.float32, copy=False)
        target_cond = cond[i].detach().cpu().numpy().astype(np.float32, copy=False)
        rows_list.append({
            "sample": int(i),
            "target_semantic": _semantic_target_entries_from_vector(
                target_cond, class_names=class_names,
                max_items=max(1, len(class_names)), threshold=0.5,
            ),
            "top_labels": _top_labels_from_probs(p, class_names=class_names, topk=int(topk_all)),
        })
    payload = {
        "cycle": int(cycle_id), "round": int(round_id), "stage": "G",
        "panel": str(panel_path), "samples": rows_list,
    }
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def _hard_wipe_pipeline_caches(output_dir: Path, data_root: str, semantic_stage_cache_dir: str = "") -> Dict[str, Any]:
    out_dir = Path(output_dir)
    data_root_path = Path(str(data_root).strip() or "toys_to_survive_development/data/berkeley_sbd")
    targets: List[Path] = [
        out_dir / "accepted_wave_library",
        out_dir / "latent_wave_pool",
        out_dir / "training_supervision",
    ]
    semantic_stage_root = (
        Path(str(semantic_stage_cache_dir).strip())
        if str(semantic_stage_cache_dir).strip()
        else (out_dir / "semantic_stage_cache")
    )
    targets.append(semantic_stage_root)
    cache_root = data_root_path / "cache"
    if cache_root.exists():
        targets.append(cache_root / "semantic_mask_cache")
        for payload_dir in sorted(cache_root.glob("payload_bank_rgb*")):
            targets.append(Path(payload_dir))

    def _remove_path_retry(path: Path, retries: int = 6) -> Tuple[bool, str]:
        last_err = ""
        for attempt in range(max(1, int(retries))):
            try:
                if not path.exists():
                    return False, ""
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                return True, ""
            except FileNotFoundError:
                return False, ""
            except PermissionError as e:
                last_err = f"{type(e).__name__}: {e}"
                gc.collect()
                time.sleep(0.15 * float(attempt + 1))
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                break
        return False, str(last_err)

    info: Dict[str, Any] = {"requested": [str(p) for p in targets], "removed": [], "missing": [], "errors": []}
    for p in targets:
        if not p.exists():
            info["missing"].append(str(p))
            continue
        ok, err = _remove_path_retry(p)
        if bool(ok):
            info["removed"].append(str(p))
        else:
            info["errors"].append(f"{str(p)} ({str(err)})")
    return info


def _soft_reset_label_caches(output_dir: Path, data_root: str, semantic_stage_cache_dir: str = "") -> Dict[str, Any]:
    data_root_path = Path(str(data_root).strip() or "toys_to_survive_development/data/berkeley_sbd")
    targets: List[Path] = []
    semantic_stage_root = (
        Path(str(semantic_stage_cache_dir).strip())
        if str(semantic_stage_cache_dir).strip()
        else (Path(output_dir) / "semantic_stage_cache")
    )
    if semantic_stage_root.exists():
        targets.append(semantic_stage_root)
    cache_root = data_root_path / "cache"
    if cache_root.exists():
        for payload_dir in sorted(cache_root.glob("payload_bank_rgb*")):
            pd = Path(payload_dir)
            targets.extend([
                pd / "labels.npy", pd / "labels.tmp.npy",
                pd / "terms.json", pd / "sources.json",
                pd / "manifest.json", pd / "source_catalog.json",
                pd / "semantic_gate_labels",
            ])

    def _remove_path_retry(path: Path, retries: int = 6) -> Tuple[bool, str]:
        last_err = ""
        for attempt in range(max(1, int(retries))):
            try:
                if not path.exists():
                    return False, ""
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                return True, ""
            except FileNotFoundError:
                return False, ""
            except PermissionError as e:
                last_err = f"{type(e).__name__}: {e}"
                gc.collect()
                time.sleep(0.15 * float(attempt + 1))
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                break
        return False, str(last_err)

    info: Dict[str, Any] = {"requested": [str(p) for p in targets], "removed": [], "missing": [], "errors": []}
    for p in targets:
        if not p.exists():
            info["missing"].append(str(p))
            continue
        ok, err = _remove_path_retry(p)
        if bool(ok):
            info["removed"].append(str(p))
        else:
            info["errors"].append(f"{str(p)} ({str(err)})")
    return info


def _sanitize_cache_component(name: str) -> str:
    txt = re.sub(r"[^a-z0-9_\-]+", "_", str(name).strip().lower())
    txt = re.sub(r"_+", "_", txt).strip("_")
    return txt or "stage"


def _resolve_semantic_stage_cache_root(output_dir: Path, cache_dir: str, cache_nonce: str = "") -> Path:
    base = Path(str(cache_dir).strip()) if str(cache_dir).strip() else (Path(output_dir) / "semantic_stage_cache")
    if str(cache_nonce).strip():
        return base / f"session_{_sanitize_cache_component(str(cache_nonce))}"
    return base


def _semantic_stage_cache_dir(root_dir: Optional[Path], stage_name: str) -> str:
    if root_dir is None:
        return ""
    return str(Path(root_dir) / _sanitize_cache_component(stage_name))


def _resolve_semantic_stage_cache_cap_mb(global_cap_mb: int, specific_cap_mb: int) -> int:
    specific = int(specific_cap_mb)
    if int(specific) >= 0:
        return max(0, int(specific))
    return max(0, int(global_cap_mb))


def _log_bootstrap_dataset_cache(stage_name: str, dataset: Any) -> None:
    info = getattr(dataset, "persistent_cache_info", None)
    if not isinstance(info, dict) or not bool(info.get("enabled", False)):
        return
    _log(
        "[semantic-stage-cache] "
        f"stage={str(stage_name)} "
        f"desired={int(info.get('desired_rows', 0))} "
        f"cached={int(info.get('cached_rows', 0))} "
        f"cache_hit={1 if bool(info.get('cache_hit', False)) else 0} "
        f"rebuilt={1 if bool(info.get('rebuilt', False)) else 0} "
        f"generated={int(info.get('generated_rows', 0))} "
        f"max_mb={float(int(info.get('max_bytes', 0))) / (1024.0 * 1024.0):.1f} "
        f"dir={str(info.get('cache_dir', ''))}"
    )


# ---------------------------------------------------------------------------
# Payload cache index splitting
# ---------------------------------------------------------------------------

def _split_payload_cache_indices(
    source_rows: Sequence[str],
    seed: int,
    external_val_fraction: float = 0.20,
) -> Tuple[List[int], List[int], Dict[str, Dict[str, Any]]]:
    """Split payload rows into train/val index lists by source provenance."""
    n_rows = int(len(source_rows))
    by_source: Dict[str, List[int]] = {}
    for i in range(int(n_rows)):
        src = re.sub(r"\s+", " ", str(source_rows[int(i)])).strip().lower() or "unknown_source"
        by_source.setdefault(src, []).append(int(i))

    val_idx: List[int] = []
    train_idx: List[int] = []
    source_stats: Dict[str, Dict[str, Any]] = {}
    ext_frac = float(max(0.0, min(1.0, float(external_val_fraction))))
    base_seed = max(0, int(seed))
    for src in sorted(by_source.keys()):
        idxs = list(by_source.get(src, []))
        n_src = int(len(idxs))
        if n_src <= 0:
            continue
        src_key_seed = int(
            hashlib.sha256(f"{src}|{base_seed}".encode("utf-8")).digest()[0]
        )
        rng = np.random.default_rng(int(base_seed) + int(src_key_seed))
        arr = np.asarray(idxs, dtype=np.int64)
        rng.shuffle(arr)
        if src == "berkeley_sbd_train":
            v = np.zeros((0,), dtype=np.int64)
            t = arr
            rule = "gate=exclude_berkeley_train refresh=all_berkeley_train"
        elif src == "berkeley_sbd_val":
            v = arr
            t = np.zeros((0,), dtype=np.int64)
            rule = "gate=all_berkeley_val refresh=exclude_berkeley_val"
        else:
            take_val = int(round(float(n_src) * ext_frac))
            if float(ext_frac) > 0.0:
                take_val = max(1, int(take_val))
            take_val = max(0, min(int(n_src), int(take_val)))
            v = arr[: int(take_val)]
            t = arr[int(take_val) :]
            rule = f"external_split gate_fraction={ext_frac:.2f}"
        val_idx.extend([int(i) for i in v.tolist()])
        train_idx.extend([int(i) for i in t.tolist()])
        source_stats[str(src)] = {
            "rows": int(n_src),
            "gate_selected": int(v.size),
            "refresh_selected": int(t.size),
            "rule": str(rule),
        }
    val_idx = sorted(set(int(i) for i in val_idx if 0 <= int(i) < int(n_rows)))
    train_idx = sorted(set(int(i) for i in train_idx if 0 <= int(i) < int(n_rows)))
    return train_idx, val_idx, source_stats
