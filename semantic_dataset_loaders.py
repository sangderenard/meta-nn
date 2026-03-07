from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from torchvision.datasets import SBDataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def _norm_txt(x: str) -> str:
    return re.sub(r"\s+", " ", str(x)).strip().lower()


def normalize_vocab_terms(terms: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: set = set()
    for x in terms:
        t = re.sub(r"\s+", " ", str(x)).strip()
        if not t:
            continue
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out


@dataclass
class SemanticDiskRow:
    image_path: str
    label_vec: np.ndarray
    terms: List[str]
    source: str
    mask_path: str = ""
    layout: Optional[Dict[str, Any]] = None
    mask_array: Optional[np.ndarray] = None
    mask_stack_array: Optional[np.ndarray] = None
    mask_stack_indices: Optional[np.ndarray] = None


def _normalize_mask_array(mask: Any, height: int, width: int) -> np.ndarray:
    arr = np.asarray(mask, dtype=np.float32)
    if int(arr.ndim) == 3:
        arr = np.mean(arr, axis=0).astype(np.float32, copy=False)
    if int(arr.shape[0]) != int(height) or int(arr.shape[1]) != int(width):
        arr = np.asarray(
            TF.resize(
                Image.fromarray(np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8), mode="L"),
                [int(height), int(width)],
                interpolation=InterpolationMode.NEAREST,
            ),
            dtype=np.float32,
        )
    if float(np.max(arr)) > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)


def _normalize_attention_map(mask: Any, gamma: float = 1.0, blur_kernel: int = 0) -> np.ndarray:
    arr = np.asarray(mask, dtype=np.float32)
    if int(arr.ndim) != 2 or int(arr.size) <= 0:
        return np.zeros_like(np.asarray(arr, dtype=np.float32), dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    arr = np.maximum(arr, 0.0).astype(np.float32, copy=False)
    vmax = float(np.max(arr)) if int(arr.size) > 0 else 0.0
    if vmax > 1e-8:
        arr = (arr / float(vmax)).astype(np.float32, copy=False)
    else:
        return np.zeros_like(arr, dtype=np.float32)
    vmean = float(np.mean(arr)) if int(arr.size) > 0 else 0.0
    if vmean > 1e-8:
        arr = np.clip(arr / float(max(vmean * 2.0, 1.0)), 0.0, 1.0).astype(np.float32, copy=False)
    gm = max(0.35, float(gamma))
    if abs(gm - 1.0) > 1e-6:
        arr = np.power(np.clip(arr, 0.0, 1.0), gm).astype(np.float32, copy=False)
    kk = int(blur_kernel)
    if kk >= 3:
        kk = int(kk) | 1
        arr = np.asarray(
            F.avg_pool2d(torch.from_numpy(arr[None, None, ...]), kernel_size=int(kk), stride=1, padding=int(kk // 2))[0, 0].cpu().numpy(),
            dtype=np.float32,
        )
        vmax = float(np.max(arr)) if int(arr.size) > 0 else 0.0
        if vmax > 1e-8:
            arr = (arr / float(vmax)).astype(np.float32, copy=False)
    return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)


def _blend_attention_maps(maps: Sequence[Any], weights: Optional[Sequence[float]] = None, gamma: float = 1.0) -> np.ndarray:
    valid: List[np.ndarray] = []
    valid_weights: List[float] = []
    for i, m in enumerate(maps):
        arr = np.asarray(m, dtype=np.float32)
        if int(arr.ndim) != 2 or int(arr.size) <= 0:
            continue
        w = 1.0 if weights is None or i >= int(len(weights)) else float(weights[i])
        if w <= 0.0:
            continue
        norm = _normalize_attention_map(arr, gamma=1.0, blur_kernel=0)
        if float(np.max(norm)) <= 1e-8:
            continue
        valid.append(norm)
        valid_weights.append(float(w))
    if len(valid) <= 0:
        if len(maps) > 0:
            ref = np.asarray(maps[0], dtype=np.float32)
            return np.zeros_like(ref, dtype=np.float32)
        return np.zeros((0, 0), dtype=np.float32)
    acc = np.zeros_like(valid[0], dtype=np.float32)
    wsum = 0.0
    for arr, w in zip(valid, valid_weights):
        acc += float(w) * arr
        wsum += float(w)
    if wsum > 1e-8:
        acc = acc / float(wsum)
    return _normalize_attention_map(acc, gamma=float(gamma), blur_kernel=5)


def build_label_mask_stack(
    mixed_mask: Any,
    label_vec: Any,
    mask_stack_array: Optional[Any] = None,
    mask_stack_indices: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(label_vec, dtype=np.float32).reshape(-1)
    positive_idx = np.where(y >= 0.5)[0].astype(np.int64)
    mixed = _normalize_attention_map(
        _normalize_mask_array(mixed_mask, height=int(np.asarray(mixed_mask).shape[-2]), width=int(np.asarray(mixed_mask).shape[-1])),
        gamma=0.95,
        blur_kernel=3,
    )
    if mask_stack_array is not None and mask_stack_indices is not None:
        idx_arr = np.asarray(mask_stack_indices, dtype=np.int64).reshape(-1)
        stack_arr = np.asarray(mask_stack_array, dtype=np.float32)
        if int(stack_arr.ndim) == 2:
            stack_arr = stack_arr[None, ...]
        out_stack: List[np.ndarray] = []
        out_idx: List[int] = []
        for si, cls_idx in enumerate(idx_arr.tolist()):
            if si >= int(stack_arr.shape[0]):
                break
            out_stack.append(
                _normalize_attention_map(
                    _normalize_mask_array(
                        stack_arr[int(si)],
                        height=int(mixed.shape[0]),
                        width=int(mixed.shape[1]),
                    ),
                    gamma=0.95,
                    blur_kernel=3,
                )
            )
            out_idx.append(int(cls_idx))
        if len(out_stack) > 0:
            return (
                np.stack(out_stack, axis=0).astype(np.float32, copy=False),
                np.asarray(out_idx, dtype=np.int64),
            )
    if int(positive_idx.size) <= 0:
        return np.zeros((0, int(mixed.shape[0]), int(mixed.shape[1])), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    stack = np.repeat(mixed[None, :, :], int(positive_idx.size), axis=0).astype(np.float32, copy=False)
    return stack, np.asarray(positive_idx, dtype=np.int64)


def build_term_mask_stack_from_image(
    image: Any,
    label_vec: Any,
    idx_to_term: Optional[Dict[int, str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(label_vec, dtype=np.float32).reshape(-1)
    positive_idx = np.where(y >= 0.5)[0].astype(np.int64)
    if int(positive_idx.size) <= 0:
        chw = _image_to_chw01(image)
        return np.zeros((0, int(chw.shape[1]), int(chw.shape[2])), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    per_label_masks: List[np.ndarray] = []
    out_idx: List[int] = []
    fallback = infer_semantic_support_mask(image=image, terms=[])
    for cls_idx in positive_idx.tolist():
        term = ""
        if isinstance(idx_to_term, dict):
            term = re.sub(r"\s+", " ", str(idx_to_term.get(int(cls_idx), ""))).strip()
        mask = infer_semantic_support_mask(image=image, terms=([term] if term else []))
        if float(np.max(mask)) <= 1e-8:
            mask = np.asarray(fallback, dtype=np.float32)
        per_label_masks.append(np.asarray(mask, dtype=np.float32))
        out_idx.append(int(cls_idx))
    return np.stack(per_label_masks, axis=0).astype(np.float32, copy=False), np.asarray(out_idx, dtype=np.int64)


def semantic_mask_stack_collate(batch: Sequence[Any]) -> Any:
    if len(batch) <= 0:
        return default_collate(batch)
    first = batch[0]
    if not isinstance(first, (tuple, list)) or int(len(first)) < 5:
        return default_collate(batch)
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    ms: List[torch.Tensor] = []
    mask_stacks: List[torch.Tensor] = []
    mask_indices: List[torch.Tensor] = []
    for sample in batch:
        if not isinstance(sample, (tuple, list)) or int(len(sample)) < 5:
            raise RuntimeError("semantic_mask_stack_collate requires 5-tuple samples.")
        xs.append(sample[0])
        ys.append(sample[1])
        ms.append(sample[2])
        stack_t = sample[3] if torch.is_tensor(sample[3]) else torch.as_tensor(sample[3], dtype=torch.float32)
        idx_t = sample[4] if torch.is_tensor(sample[4]) else torch.as_tensor(sample[4], dtype=torch.long)
        mask_stacks.append(stack_t.to(dtype=torch.float32))
        mask_indices.append(idx_t.to(dtype=torch.long))
    return {
        "x": torch.stack(xs, dim=0),
        "y": torch.stack(ys, dim=0),
        "mask": torch.stack(ms, dim=0),
        "mask_stacks": mask_stacks,
        "mask_indices": mask_indices,
    }


def _image_to_chw01(image: Any) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if int(arr.ndim) == 3 and int(arr.shape[0]) in (1, 3, 4):
        chw = np.asarray(arr[:3, :, :], dtype=np.float32)
        if int(chw.shape[0]) == 1:
            chw = np.repeat(chw, 3, axis=0)
    elif int(arr.ndim) == 3 and int(arr.shape[2]) in (1, 3, 4):
        hwc = np.asarray(arr[:, :, :3], dtype=np.float32)
        if int(hwc.shape[2]) == 1:
            hwc = np.repeat(hwc, 3, axis=2)
        chw = np.transpose(hwc, (2, 0, 1)).astype(np.float32, copy=False)
    elif int(arr.ndim) == 2:
        gray = np.asarray(arr, dtype=np.float32)
        chw = np.repeat(gray[None, :, :], 3, axis=0).astype(np.float32, copy=False)
    else:
        raise RuntimeError(f"Unsupported image shape for semantic mask inference: {tuple(arr.shape)}")
    vmax = float(np.max(chw)) if int(chw.size) > 0 else 0.0
    vmin = float(np.min(chw)) if int(chw.size) > 0 else 0.0
    if vmax > 1.0:
        chw = chw / 255.0
    elif vmin < 0.0 and vmax <= 1.0:
        chw = (chw + 1.0) * 0.5
    return np.clip(chw, 0.0, 1.0).astype(np.float32, copy=False)


def _semantic_color_score_maps(chw: np.ndarray) -> Dict[str, np.ndarray]:
    arr = np.clip(np.asarray(chw, dtype=np.float32), 0.0, 1.0)
    if int(arr.ndim) != 3 or int(arr.shape[0]) < 3:
        return {}
    r = np.asarray(arr[0], dtype=np.float32)
    g = np.asarray(arr[1], dtype=np.float32)
    b = np.asarray(arr[2], dtype=np.float32)
    vmax = np.maximum.reduce([r, g, b]).astype(np.float32, copy=False)
    vmin = np.minimum.reduce([r, g, b]).astype(np.float32, copy=False)
    sat = np.clip(vmax - vmin, 0.0, 1.0).astype(np.float32, copy=False)

    def _norm01(x: np.ndarray) -> np.ndarray:
        xx = np.asarray(x, dtype=np.float32)
        hi = float(np.max(xx)) if int(xx.size) > 0 else 0.0
        if hi <= 1e-8:
            return np.zeros_like(xx, dtype=np.float32)
        return np.clip(xx / float(hi), 0.0, 1.0).astype(np.float32, copy=False)

    red = _norm01(np.clip(r - np.maximum(g, b), 0.0, 1.0) * np.clip(sat - 0.05, 0.0, 1.0))
    green = _norm01(np.clip(g - np.maximum(r, b), 0.0, 1.0) * np.clip(sat - 0.05, 0.0, 1.0))
    blue = _norm01(np.clip(b - np.maximum(r, g), 0.0, 1.0) * np.clip(sat - 0.05, 0.0, 1.0))
    yellow = _norm01(np.clip(np.minimum(r, g) - b, 0.0, 1.0) * np.clip(sat - 0.05, 0.0, 1.0))
    cyan = _norm01(np.clip(np.minimum(g, b) - r, 0.0, 1.0) * np.clip(sat - 0.05, 0.0, 1.0))
    magenta = _norm01(np.clip(np.minimum(r, b) - g, 0.0, 1.0) * np.clip(sat - 0.05, 0.0, 1.0))
    brown = _norm01(
        np.clip(r - g, 0.0, 1.0)
        * np.clip(g - b, 0.0, 1.0)
        * np.clip(vmax, 0.15, 0.75)
        * np.clip(0.85 - vmax, 0.0, 1.0)
    )
    black = _norm01(np.clip(0.22 - vmax, 0.0, 1.0))
    white = _norm01(np.clip(vmin - 0.78, 0.0, 1.0) * np.clip(0.20 - sat, 0.0, 1.0))
    gray = _norm01(np.clip(0.18 - sat, 0.0, 1.0) * np.clip(1.0 - np.abs(vmax - 0.5) * 2.2, 0.0, 1.0))
    return {
        "red": red,
        "green": green,
        "blue": blue,
        "yellow": yellow,
        "cyan": cyan,
        "magenta": magenta,
        "brown": brown,
        "black": black,
        "white": white,
        "gray": gray,
        "grey": gray,
    }


def detect_semantic_color_terms(
    image: Any,
    mask: Optional[Any] = None,
    coverage_threshold: float = 0.06,
    dominance_threshold: float = 0.22,
) -> List[str]:
    try:
        chw = _image_to_chw01(image)
    except Exception:
        return []
    maps = _semantic_color_score_maps(chw)
    if len(maps) <= 0:
        return []
    valid = None
    if mask is not None:
        try:
            valid = _normalize_mask_array(mask, height=int(chw.shape[1]), width=int(chw.shape[2]))
        except Exception:
            valid = None
    if valid is None or float(np.mean(valid)) <= 0.01:
        valid = np.ones((int(chw.shape[1]), int(chw.shape[2])), dtype=np.float32)
    valid_mask = np.asarray(valid, dtype=np.float32) >= 0.15
    valid_count = int(np.sum(valid_mask))
    if valid_count <= 0:
        valid_mask = np.ones((int(chw.shape[1]), int(chw.shape[2])), dtype=bool)
        valid_count = int(np.sum(valid_mask))

    out: List[str] = []
    for term in ["red", "green", "blue", "yellow", "cyan", "magenta", "brown", "black", "white", "gray", "grey"]:
        score = np.asarray(maps.get(term), dtype=np.float32)
        if int(score.size) <= 0:
            continue
        frac = float(np.mean(score[valid_mask] >= float(dominance_threshold))) if int(valid_count) > 0 else 0.0
        mean_score = float(np.mean(score[valid_mask])) if int(valid_count) > 0 else 0.0
        if frac >= float(coverage_threshold) or mean_score >= float(max(0.10, dominance_threshold * 0.72)):
            out.append(str(term))
    return normalize_vocab_terms(out)


def infer_semantic_support_mask(image: Any, terms: Optional[Sequence[str]] = None) -> np.ndarray:
    chw = _image_to_chw01(image)
    h = int(chw.shape[1])
    w = int(chw.shape[2])
    gray = np.clip(np.mean(np.asarray(chw[:3], dtype=np.float32), axis=0), 0.0, 1.0).astype(np.float32, copy=False)
    if int(gray.size) <= 0:
        return np.zeros((int(h), int(w)), dtype=np.float32)

    norm_terms = {_norm_txt(t) for t in (terms or []) if str(t).strip()}
    object_like = bool(
        ("object" in norm_terms)
        or any(str(t).startswith("digit ") for t in norm_terms)
        or any(str(t).startswith("letter ") for t in norm_terms)
        or any(str(t).startswith("pictogram ") for t in norm_terms)
    )
    color_terms = {
        str(t)
        for t in norm_terms
        if str(t)
        in {"red", "green", "blue", "yellow", "cyan", "magenta", "brown", "black", "white", "gray", "grey"}
    }
    localized_like = bool(
        bool(object_like)
        or len(color_terms) > 0
        or any(
            t in {
                "edge",
                "shape",
                "texture",
                "layout",
                "mask",
                "composite",
                "front",
                "back",
                "left",
                "right",
                "top",
                "bottom",
            }
            for t in norm_terms
        )
    )
    global_like = bool(
        (not bool(localized_like))
        and any(
            t in {
                "signal",
                "noise",
                "mixed noise and signal",
                "blur damage",
                "noise damage",
                "dropout damage",
                "quantization damage",
                "stride skew damage",
                "white",
                "black",
                "gray",
                "grey",
                "none",
            }
            for t in norm_terms
        )
    )

    border_parts = [gray[0:1, :], gray[-1:, :], gray[:, 0:1], gray[:, -1:]]
    border = np.concatenate([np.asarray(p, dtype=np.float32).reshape(-1) for p in border_parts], axis=0)
    bg = float(np.median(border)) if int(border.size) > 0 else float(np.median(gray))

    blur = F.avg_pool2d(torch.from_numpy(gray[None, None, ...]), kernel_size=5, stride=1, padding=2)[0, 0].cpu().numpy()
    diff_bg = np.abs(gray - float(bg)).astype(np.float32, copy=False)
    diff_local = np.abs(gray - np.asarray(blur, dtype=np.float32)).astype(np.float32, copy=False)
    gy, gx = np.gradient(gray.astype(np.float32, copy=False))
    grad = np.sqrt((gx * gx) + (gy * gy)).astype(np.float32, copy=False)

    def _norm_map(v: np.ndarray) -> np.ndarray:
        vmax = float(np.max(v)) if int(v.size) > 0 else 0.0
        if vmax <= 1e-8:
            return np.zeros_like(v, dtype=np.float32)
        return np.clip(np.asarray(v, dtype=np.float32) / float(vmax), 0.0, 1.0).astype(np.float32, copy=False)

    score = np.maximum.reduce([
        _norm_map(diff_bg),
        0.85 * _norm_map(diff_local),
        0.65 * _norm_map(grad),
    ]).astype(np.float32, copy=False)
    if len(color_terms) > 0:
        color_maps = _semantic_color_score_maps(chw)
        color_parts = [np.asarray(color_maps.get(str(term)), dtype=np.float32) for term in sorted(color_terms) if str(term) in color_maps]
        color_parts = [x for x in color_parts if int(x.size) > 0]
        if len(color_parts) > 0:
            color_score = np.maximum.reduce(color_parts).astype(np.float32, copy=False)
            color_score = _normalize_attention_map(color_score, gamma=0.82, blur_kernel=3)
            non_color_penalty = np.ones_like(color_score, dtype=np.float32)
            other_color_parts = [
                np.asarray(color_maps.get(str(term)), dtype=np.float32)
                for term in sorted(color_maps.keys())
                if str(term) not in color_terms
            ]
            other_color_parts = [x for x in other_color_parts if int(x.size) > 0]
            if len(other_color_parts) > 0:
                other_score = _normalize_attention_map(np.maximum.reduce(other_color_parts), gamma=1.0, blur_kernel=3)
                non_color_penalty = np.clip(1.0 - (0.55 * other_score), 0.15, 1.0).astype(np.float32, copy=False)
            score = _blend_attention_maps(
                [score * non_color_penalty, color_score * non_color_penalty],
                weights=[0.28, 1.45],
                gamma=0.82,
            ).astype(np.float32, copy=False)
    score_t = F.avg_pool2d(torch.from_numpy(score[None, None, ...]), kernel_size=5, stride=1, padding=2)
    score = _normalize_attention_map(score_t[0, 0].cpu().numpy(), gamma=0.92, blur_kernel=5)

    thr = max(0.16, float(np.percentile(score.reshape(-1), 72)) * 0.70)
    seed = (score >= float(thr)).astype(np.float32, copy=False)
    seed_t = F.avg_pool2d(torch.from_numpy(seed[None, None, ...]), kernel_size=5, stride=1, padding=2)
    seed = np.asarray(seed_t[0, 0].cpu().numpy(), dtype=np.float32)
    seed = _normalize_attention_map(seed, gamma=0.85, blur_kernel=3)

    coverage = float(np.mean(seed >= 0.40)) if int(seed.size) > 0 else 0.0
    if bool(object_like) and (coverage <= 0.01 or coverage >= 0.97):
        hi = float(np.percentile(gray.reshape(-1), 75))
        lo = float(np.percentile(gray.reshape(-1), 25))
        if abs(float(hi) - float(bg)) >= abs(float(lo) - float(bg)):
            refine = (gray >= float((hi + bg) * 0.5)).astype(np.float32, copy=False)
        else:
            refine = (gray <= float((lo + bg) * 0.5)).astype(np.float32, copy=False)
        refine_t = F.avg_pool2d(torch.from_numpy(refine[None, None, ...]), kernel_size=5, stride=1, padding=2)
        seed = _normalize_attention_map(refine_t[0, 0].cpu().numpy(), gamma=0.82, blur_kernel=5)
        coverage = float(np.mean(seed >= 0.40)) if int(seed.size) > 0 else 0.0

    if coverage <= 0.01:
        score_thr = max(0.10, float(np.percentile(score.reshape(-1), 55)) * 0.50)
        fallback = _normalize_attention_map(score * (score >= float(score_thr)).astype(np.float32, copy=False), gamma=0.95, blur_kernel=5)
        if float(np.max(fallback)) <= 1e-8:
            fallback = _normalize_attention_map(score, gamma=1.0, blur_kernel=7)
        seed = fallback

    if bool(object_like):
        mask = _blend_attention_maps([score, seed], weights=[0.65, 1.15], gamma=0.88)
    elif bool(global_like):
        local_grad = _normalize_attention_map((0.55 * diff_local) + (0.85 * grad), gamma=1.05, blur_kernel=5)
        mask = _blend_attention_maps([score, local_grad], weights=[0.70, 1.20], gamma=1.08)
    else:
        mask = _blend_attention_maps([score, seed], weights=[0.90, 0.75], gamma=0.98)

    if bool(object_like) and float(np.mean(mask >= 0.45)) <= 0.02:
        mask = _blend_attention_maps([mask, seed], weights=[0.65, 1.35], gamma=0.85)

    return _normalize_attention_map(mask, gamma=(0.82 if bool(object_like) else 1.05), blur_kernel=5)


@dataclass
class StageDatasetManifest:
    name: str
    dataset: Dataset
    batch_size: int
    seed: int
    num_workers: int
    device_type: str
    max_samples: int = 0
    ordered_indices: Optional[Sequence[int]] = None
    persistent_workers: bool = False
    prefetch_factor: int = 2


def effective_dataloader_num_workers(num_workers: int, device_type: str = "") -> int:
    workers = max(0, int(num_workers))
    if workers <= 0:
        return 0
    if os.name == "nt" and str(device_type).strip().lower() == "cuda":
        # Avoid Win32 shared-mapping allocation failures (error 1455) when
        # worker processes collate large semantic tensor batches on Windows.
        return 0
    return workers


def dataloader_perf_kwargs(num_workers: int, persistent_workers: bool, prefetch_factor: int) -> Dict[str, Any]:
    if int(num_workers) <= 0:
        return {}
    out = {"persistent_workers": bool(persistent_workers)}
    if int(prefetch_factor) > 0:
        out["prefetch_factor"] = int(prefetch_factor)
    return out


def build_loader_from_manifest(manifest: StageDatasetManifest) -> Tuple[Optional[DataLoader], int]:
    n = int(len(manifest.dataset))
    if n <= 0:
        return None, 0
    if manifest.ordered_indices is None:
        picks = np.arange(int(n), dtype=np.int64)
        rng = np.random.default_rng(int(manifest.seed))
        rng.shuffle(picks)
        if int(manifest.max_samples) > 0:
            picks = picks[: int(min(int(manifest.max_samples), int(picks.shape[0])))]
    else:
        picks = np.asarray([int(i) for i in manifest.ordered_indices if 0 <= int(i) < int(n)], dtype=np.int64)
        if int(manifest.max_samples) > 0 and int(picks.size) > int(manifest.max_samples):
            picks = picks[: int(manifest.max_samples)]
    if int(picks.size) <= 0:
        return None, 0
    ds_use: Dataset = manifest.dataset if int(picks.size) == int(n) else torch.utils.data.Subset(manifest.dataset, picks.tolist())
    collate_fn = semantic_mask_stack_collate if bool(getattr(manifest.dataset, "use_semantic_mask_stack_collate", False)) else None
    loader_num_workers = effective_dataloader_num_workers(
        num_workers=manifest.num_workers,
        device_type=manifest.device_type,
    )
    loader = DataLoader(
        ds_use,
        batch_size=max(1, int(manifest.batch_size)),
        shuffle=False,
        num_workers=int(loader_num_workers),
        pin_memory=(str(manifest.device_type).strip().lower() == "cuda"),
        drop_last=False,
        collate_fn=collate_fn,
        **dataloader_perf_kwargs(
            num_workers=int(loader_num_workers),
            persistent_workers=bool(manifest.persistent_workers),
            prefetch_factor=int(manifest.prefetch_factor),
        ),
    )
    return loader, int(picks.size)


@lru_cache(maxsize=32)
def _unit_interval_axis(length: int) -> np.ndarray:
    n = max(1, int(length))
    if n <= 1:
        return np.zeros((1,), dtype=np.float32)
    return np.linspace(0.0, 1.0, num=int(n), dtype=np.float32)


def augment_bootstrap_chw01(
    img: np.ndarray,
    seed: int,
    return_terms: bool = False,
    return_touch_mask: bool = False,
) -> Any:
    arr = np.asarray(img, dtype=np.float32)
    if int(arr.ndim) == 3 and int(arr.shape[0]) == 3:
        x = np.asarray(arr, dtype=np.float32, order="C")
    elif int(arr.ndim) == 3 and int(arr.shape[2]) == 3:
        x = np.transpose(arr, (2, 0, 1)).astype(np.float32, copy=False)
    elif int(arr.ndim) == 2:
        x = np.repeat(arr[None, :, :], 3, axis=0).astype(np.float32, copy=False)
    else:
        raise RuntimeError(f"Unsupported bootstrap row shape for augmentation: {tuple(arr.shape)}")
    x = np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)
    c, h, w = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    rng = np.random.default_rng(int(seed))
    applied_terms: List[str] = []
    touched = np.zeros((int(h), int(w)), dtype=np.float32)

    def _accumulate_touch(delta: Any, scale: float = 1.0, gamma: float = 1.0):
        nonlocal touched
        norm = _normalize_attention_map(delta, gamma=float(gamma), blur_kernel=3)
        if float(np.max(norm)) <= 1e-8:
            return
        touched = np.asarray(touched, dtype=np.float32) + (float(scale) * norm)

    def _mark_diff(prev_x: np.ndarray, next_x: np.ndarray, scale: float = 1.0):
        prev_g = np.mean(np.asarray(prev_x, dtype=np.float32), axis=0)
        next_g = np.mean(np.asarray(next_x, dtype=np.float32), axis=0)
        diff = np.abs(next_g - prev_g).astype(np.float32, copy=False)
        _accumulate_touch(diff, scale=float(scale), gamma=0.95)

    prev = np.asarray(x, dtype=np.float32).copy()
    shift_x = int(rng.integers(-max(1, w // 14), max(1, w // 14) + 1))
    shift_y = int(rng.integers(-max(1, h // 14), max(1, h // 14) + 1))
    if shift_x != 0:
        x = np.roll(x, shift=shift_x, axis=2)
    if shift_y != 0:
        x = np.roll(x, shift=shift_y, axis=1)
    if shift_x != 0 or shift_y != 0:
        _mark_diff(prev, x, scale=0.7)
    prev = np.asarray(x, dtype=np.float32).copy()
    if float(rng.random()) < 0.45:
        x = np.flip(x, axis=2).copy()
        _mark_diff(prev, x, scale=0.4)
        prev = np.asarray(x, dtype=np.float32).copy()
    if float(rng.random()) < 0.15:
        x = np.flip(x, axis=1).copy()
        _mark_diff(prev, x, scale=0.4)
        prev = np.asarray(x, dtype=np.float32).copy()

    if float(rng.random()) < 0.55:
        k = int(rng.choice(np.asarray([3, 5, 7], dtype=np.int32)))
        t = torch.from_numpy(np.asarray(x, dtype=np.float32)[None, ...])
        t = F.avg_pool2d(t, kernel_size=int(k), stride=1, padding=int(k // 2))
        x_blur = np.asarray(t[0].cpu().numpy(), dtype=np.float32)
        _mark_diff(prev, x_blur, scale=0.8)
        x = x_blur
        applied_terms.extend(normalize_vocab_terms(["blur damage", "signal"]))
        prev = np.asarray(x, dtype=np.float32).copy()
    if float(rng.random()) < 0.45:
        odd_shift = int(rng.integers(1, 8))
        x[:, 1::2, :] = np.roll(x[:, 1::2, :], shift=odd_shift, axis=2)
        stripe = np.zeros((h, w), dtype=np.float32)
        stripe[1::2, :] = 1.0
        _accumulate_touch(stripe, scale=0.7, gamma=1.15)
        applied_terms.extend(normalize_vocab_terms(["stride skew damage", "signal"]))
        prev = np.asarray(x, dtype=np.float32).copy()
    if float(rng.random()) < 0.35:
        keep = float(rng.uniform(0.76, 0.96))
        keep_mask = (rng.random((h, w), dtype=np.float32) < keep).astype(np.float32, copy=False)
        x = x * keep_mask[None, :, :]
        _accumulate_touch(1.0 - keep_mask, scale=1.0, gamma=0.90)
        applied_terms.extend(normalize_vocab_terms(["dropout damage", "signal"]))
        prev = np.asarray(x, dtype=np.float32).copy()
    if float(rng.random()) < 0.40:
        lv = int(rng.choice(np.asarray([4, 6, 8, 12], dtype=np.int32)))
        x_quant = np.round(x * float(lv - 1)) / float(max(1, lv - 1))
        _mark_diff(prev, x_quant, scale=0.7)
        x = x_quant
        applied_terms.extend(normalize_vocab_terms(["quantization damage", "signal"]))
        prev = np.asarray(x, dtype=np.float32).copy()

    x01 = _unit_interval_axis(int(w))[None, :]
    y01 = _unit_interval_axis(int(h))[:, None]
    fx = float(rng.uniform(1.2, 8.5))
    fy = float(rng.uniform(1.0, 7.5))
    ph = float(rng.uniform(0.0, 2.0 * np.pi))
    phase = np.empty((int(h), int(w)), dtype=np.float32)
    np.multiply(y01, np.float32(fy), out=phase)
    phase += np.float32(fx) * x01
    phase *= np.float32(2.0 * np.pi)
    phase += np.float32(ph)
    np.sin(phase, out=phase)
    phase *= np.float32(0.5)
    phase += np.float32(0.5)
    sig = phase
    noi = rng.random((h, w), dtype=np.float32)
    if float(rng.random()) < 0.5:
        mix_a = float(rng.uniform(0.18, 0.62))
        mix = np.clip((mix_a * sig) + ((1.0 - mix_a) * noi), 0.0, 1.0).astype(np.float32, copy=False)
    else:
        bits = int(rng.integers(2, 7))
        s_u16 = np.round(np.clip(sig, 0.0, 1.0) * 65535.0).astype(np.uint16, copy=False)
        n_u16 = np.round(np.clip(noi, 0.0, 1.0) * 65535.0).astype(np.uint16, copy=False)
        payload = ((n_u16 >> int(16 - bits)) & np.uint16((1 << bits) - 1)).astype(np.uint16, copy=False)
        keep_mask = np.uint16(0xFFFF ^ ((1 << bits) - 1))
        mix = np.clip(((s_u16 & keep_mask) | payload).astype(np.float32) / 65535.0, 0.0, 1.0).astype(np.float32, copy=False)
    blend = float(rng.uniform(0.08, 0.28))
    x_mix = np.clip(((1.0 - blend) * x) + (blend * mix[None, :, :]), 0.0, 1.0)
    _mark_diff(prev, x_mix, scale=0.6)
    x = x_mix
    applied_terms.extend(["mixed noise and signal", "noise", "signal"])
    prev = np.asarray(x, dtype=np.float32).copy()

    if float(rng.random()) < 0.65:
        std = float(rng.uniform(0.01, 0.08))
        noise_delta = (std * rng.standard_normal((c, h, w), dtype=np.float32)).astype(np.float32, copy=False)
        x_noise = np.clip(x + noise_delta, 0.0, 1.0)
        _mark_diff(prev, x_noise, scale=0.9)
        x = x_noise
        applied_terms.extend(normalize_vocab_terms(["noise damage", "noise", "mixed noise and signal", "signal"]))

    prev = np.asarray(x, dtype=np.float32).copy()
    if int(c) >= 3 and float(rng.random()) < 0.45:
        color_profiles = {
            "red": np.asarray([1.00, 0.18, 0.18], dtype=np.float32),
            "green": np.asarray([0.18, 1.00, 0.18], dtype=np.float32),
            "blue": np.asarray([0.18, 0.18, 1.00], dtype=np.float32),
            "yellow": np.asarray([1.00, 1.00, 0.20], dtype=np.float32),
            "cyan": np.asarray([0.18, 1.00, 1.00], dtype=np.float32),
            "magenta": np.asarray([1.00, 0.18, 1.00], dtype=np.float32),
            "brown": np.asarray([0.72, 0.44, 0.22], dtype=np.float32),
            "white": np.asarray([1.00, 1.00, 1.00], dtype=np.float32),
            "black": np.asarray([0.08, 0.08, 0.08], dtype=np.float32),
            "gray": np.asarray([0.55, 0.55, 0.55], dtype=np.float32),
        }
        color_key = str(rng.choice(np.asarray(list(color_profiles.keys()), dtype=object))).strip().lower()
        profile = color_profiles.get(color_key, np.asarray([1.0, 1.0, 1.0], dtype=np.float32))[:, None, None]
        luma = np.mean(np.asarray(x[:3], dtype=np.float32), axis=0, keepdims=True)
        tinted = np.clip(profile * np.clip(0.25 + (0.90 * luma), 0.0, 1.0), 0.0, 1.0).astype(np.float32, copy=False)
        alpha = float(rng.uniform(0.24, 0.58))
        x_tinted = np.clip(((1.0 - alpha) * x) + (alpha * tinted), 0.0, 1.0)
        _mark_diff(prev, x_tinted, scale=0.85)
        x = x_tinted
        applied_terms.extend(normalize_vocab_terms([str(color_key)]))
        if str(color_key) == "white":
            applied_terms.extend(normalize_vocab_terms(["bright"]))
        elif str(color_key) == "black":
            applied_terms.extend(normalize_vocab_terms(["dark"]))

    prev = np.asarray(x, dtype=np.float32).copy()
    if float(rng.random()) < 0.32:
        gx = np.zeros((h, w), dtype=np.float32)
        gy = np.zeros((h, w), dtype=np.float32)
        gray = np.mean(np.asarray(x[:3], dtype=np.float32), axis=0)
        gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
        edge = np.sqrt((gx * gx) + (gy * gy)).astype(np.float32, copy=False)
        if float(np.max(edge)) > 1e-8:
            edge = edge / float(np.max(edge))
        edge_rgb = np.repeat(edge[None, :, :], c, axis=0).astype(np.float32, copy=False)
        alpha = float(rng.uniform(0.20, 0.46))
        x_edge = np.clip(((1.0 - alpha) * x) + (alpha * edge_rgb), 0.0, 1.0)
        _mark_diff(prev, x_edge, scale=0.80)
        x = x_edge
        applied_terms.extend(normalize_vocab_terms(["edge", "shape", "rough"]))

    prev = np.asarray(x, dtype=np.float32).copy()
    if float(rng.random()) < 0.36:
        fx_t = float(rng.uniform(4.0, 16.0))
        fy_t = float(rng.uniform(4.0, 16.0))
        ph_t = float(rng.uniform(0.0, 2.0 * np.pi))
        tex = 0.5 + (0.25 * np.sin((2.0 * np.pi * ((fx_t * x01) + (0.35 * fy_t * y01))) + ph_t))
        tex += 0.25 * np.sin((2.0 * np.pi * ((0.45 * fx_t * x01) - (fy_t * y01))) + (ph_t * 0.7))
        tex = np.clip(tex, 0.0, 1.0).astype(np.float32, copy=False)
        tex_rgb = np.repeat(tex[None, :, :], c, axis=0).astype(np.float32, copy=False)
        alpha = float(rng.uniform(0.18, 0.42))
        x_tex = np.clip(x * (0.78 + (0.55 * tex_rgb)) + (alpha * tex_rgb), 0.0, 1.0)
        _mark_diff(prev, x_tex, scale=0.70)
        x = x_tex
        applied_terms.extend(normalize_vocab_terms(["texture", "pattern"]))
        if float(np.std(tex)) >= 0.18:
            applied_terms.extend(normalize_vocab_terms(["rough"]))
        else:
            applied_terms.extend(normalize_vocab_terms(["smooth"]))

    x_out = np.asarray(x, dtype=np.float32)
    touched = _normalize_attention_map(touched, gamma=1.05, blur_kernel=5)
    if bool(return_terms) and bool(return_touch_mask):
        return x_out, normalize_vocab_terms(applied_terms), touched
    if bool(return_terms):
        return x_out, normalize_vocab_terms(applied_terms)
    if bool(return_touch_mask):
        return x_out, touched
    return x_out


class BootstrapDynamicDataset(Dataset):
    def __init__(
        self,
        images: Sequence[np.ndarray],
        targets: Sequence[np.ndarray],
        total_rows: int,
        seed: int,
        augment: bool,
        expected_target_dim: int = 0,
        semantic_term_to_idx: Optional[Dict[str, int]] = None,
        augment_apply_terms: bool = True,
        return_masks: bool = False,
        return_mask_stack: bool = False,
        dataset_name: str = "bootstrap_dynamic",
    ):
        n = min(int(len(images)), int(len(targets)))
        if n <= 0:
            raise RuntimeError(f"{str(dataset_name)} requires non-empty images/targets.")
        self.images = [np.asarray(images[i], dtype=np.float32) for i in range(int(n))]
        self.targets = [np.asarray(targets[i], dtype=np.float32).reshape(-1) for i in range(int(n))]
        self.base_rows = int(n)
        self.total_rows = max(int(self.base_rows), int(total_rows))
        self.seed = int(seed)
        self.augment = bool(augment)
        self.return_masks = bool(return_masks)
        self.return_mask_stack = bool(return_mask_stack)
        self.use_semantic_mask_stack_collate = bool(self.return_mask_stack)
        y_sizes = sorted({int(v.size) for v in self.targets})
        if len(y_sizes) != 1:
            raise RuntimeError(f"{str(dataset_name)} target width mismatch: {y_sizes}")
        self.target_dim = int(y_sizes[0])
        if int(expected_target_dim) > 0 and int(self.target_dim) != int(expected_target_dim):
            raise RuntimeError(
                f"{str(dataset_name)} target width mismatch: got={int(self.target_dim)} expected={int(expected_target_dim)}"
            )
        self.semantic_term_to_idx = {
            re.sub(r"\s+", " ", str(k)).strip().lower(): int(v)
            for k, v in (semantic_term_to_idx.items() if isinstance(semantic_term_to_idx, dict) else [])
            if str(k).strip()
        }
        self.idx_to_term = {
            int(v): re.sub(r"\s+", " ", str(k)).strip()
            for k, v in self.semantic_term_to_idx.items()
            if 0 <= int(v) < int(self.target_dim)
        }
        self.augment_apply_terms = bool(augment_apply_terms)
        self.base_masks = []
        self.base_mask_stacks = []
        self.base_mask_stack_indices = []
        for i in range(int(self.base_rows)):
            tgt = np.asarray(self.targets[int(i)], dtype=np.float32).reshape(-1)
            active_terms = [str(self.idx_to_term[int(j)]) for j in np.where(tgt >= 0.5)[0].astype(np.int64).tolist() if int(j) in self.idx_to_term]
            base_mask = infer_semantic_support_mask(self.images[int(i)], terms=active_terms)
            self.base_masks.append(base_mask)
            if bool(self.return_mask_stack):
                mask_stack_np, mask_idx_np = build_term_mask_stack_from_image(
                    image=self.images[int(i)],
                    label_vec=tgt,
                    idx_to_term=self.idx_to_term,
                )
            else:
                mask_stack_np = np.zeros((0, int(base_mask.shape[0]), int(base_mask.shape[1])), dtype=np.float32)
                mask_idx_np = np.zeros((0,), dtype=np.int64)
            self.base_mask_stacks.append(mask_stack_np)
            self.base_mask_stack_indices.append(mask_idx_np)

    def __len__(self) -> int:
        return int(self.total_rows)

    def __getitem__(self, index: int):
        idx = int(index)
        if idx < 0:
            idx = int(self.total_rows) + idx
        if idx < 0 or idx >= int(self.total_rows):
            raise IndexError(idx)
        cycle = int(idx // max(1, int(self.base_rows)))
        off = int(idx % max(1, int(self.base_rows)))
        base_idx = int((off + (cycle * 17) + int(self.seed % max(1, int(self.base_rows)))) % int(self.base_rows))
        img = np.asarray(self.images[int(base_idx)], dtype=np.float32)
        tgt = np.asarray(self.targets[int(base_idx)], dtype=np.float32).copy()
        mask = np.asarray(self.base_masks[int(base_idx)], dtype=np.float32).copy()
        mask_stack_np = np.asarray(self.base_mask_stacks[int(base_idx)], dtype=np.float32).copy()
        mask_idx_np = np.asarray(self.base_mask_stack_indices[int(base_idx)], dtype=np.int64).copy()
        if bool(self.augment) and int(self.total_rows) > int(self.base_rows):
            aug_seed = int((int(self.seed) * 2654435761 + int(idx) * 1103515245 + int(base_idx) * 122949829) % (2**32 - 1))
            img, aug_terms, touched = augment_bootstrap_chw01(
                img=img,
                seed=int(aug_seed),
                return_terms=True,
                return_touch_mask=True,
            )
            aug_term_set = {
                re.sub(r"\s+", " ", str(term)).strip().lower()
                for term in (aug_terms if isinstance(aug_terms, list) else [])
                if str(term).strip()
            }
            global_aug = bool(
                aug_term_set.intersection(
                    {
                        "signal",
                        "noise",
                        "mixed noise and signal",
                        "blur damage",
                        "noise damage",
                        "dropout damage",
                        "quantization damage",
                        "stride skew damage",
                        "texture",
                        "pattern",
                    }
                )
            )
            mask = _blend_attention_maps(
                [np.asarray(mask, dtype=np.float32), np.asarray(touched, dtype=np.float32)],
                weights=([0.35, 1.25] if bool(global_aug) else [0.70, 0.95]),
                gamma=(1.08 if bool(global_aug) else 0.95),
            ).astype(np.float32, copy=False)
            if bool(self.augment_apply_terms) and int(len(self.semantic_term_to_idx)) > 0 and isinstance(aug_terms, list):
                for term in aug_terms:
                    tk = re.sub(r"\s+", " ", str(term)).strip().lower()
                    if not tk:
                        continue
                    ti = int(self.semantic_term_to_idx.get(tk, -1))
                    if 0 <= int(ti) < int(tgt.size):
                        tgt[int(ti)] = 1.0
            if bool(self.return_mask_stack):
                mask_stack_np, mask_idx_np = build_term_mask_stack_from_image(
                    image=img,
                    label_vec=tgt,
                    idx_to_term=self.idx_to_term,
                )
                if int(mask_stack_np.shape[0]) > 0 and int(mask_idx_np.size) > 0:
                    mixed_from_stack = np.max(np.asarray(mask_stack_np, dtype=np.float32), axis=0)
                    mask = _blend_attention_maps(
                        [np.asarray(mask, dtype=np.float32), np.asarray(mixed_from_stack, dtype=np.float32)],
                        weights=[0.55, 1.10],
                        gamma=0.92,
                    ).astype(np.float32, copy=False)
        x_t = torch.from_numpy(np.asarray(img, dtype=np.float32))
        y_t = torch.from_numpy(np.asarray(tgt, dtype=np.float32))
        if bool(self.return_masks):
            mask_t = torch.from_numpy(np.asarray(mask, dtype=np.float32)[None, ...])
            if bool(self.return_mask_stack):
                if int(mask_stack_np.shape[0]) <= 0 or int(mask_idx_np.size) <= 0:
                    mask_stack_np, mask_idx_np = build_term_mask_stack_from_image(
                        image=img,
                        label_vec=tgt,
                        idx_to_term=self.idx_to_term,
                    )
                if int(mask_stack_np.shape[0]) <= 0 or int(mask_idx_np.size) <= 0:
                    mask_stack_np, mask_idx_np = build_label_mask_stack(mixed_mask=mask, label_vec=tgt)
                return (
                    x_t,
                    y_t,
                    mask_t,
                    torch.from_numpy(np.asarray(mask_stack_np, dtype=np.float32)),
                    torch.from_numpy(np.asarray(mask_idx_np, dtype=np.int64)),
                )
            return x_t, y_t, mask_t
        return x_t, y_t


class DiskSemanticRowsDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[SemanticDiskRow],
        image_size: int,
        return_masks: bool = False,
        return_mask_stack: bool = False,
        degrade: bool = False,
        degrade_seed: int = 0,
        degrade_config: Optional[Dict[str, Any]] = None,
    ):
        self.rows = [
            SemanticDiskRow(
                image_path=str(r.image_path),
                label_vec=np.asarray(r.label_vec, dtype=np.float32).reshape(-1),
                terms=list(normalize_vocab_terms(r.terms)),
                source=str(r.source),
                mask_path=str(r.mask_path or ""),
                layout=dict(r.layout) if isinstance(r.layout, dict) else None,
                mask_array=(None if r.mask_array is None else np.asarray(r.mask_array, dtype=np.float32)),
                mask_stack_array=(None if r.mask_stack_array is None else np.asarray(r.mask_stack_array, dtype=np.float32)),
                mask_stack_indices=(None if r.mask_stack_indices is None else np.asarray(r.mask_stack_indices, dtype=np.int64)),
            )
            for r in rows
        ]
        self.image_size = max(8, int(image_size))
        self.return_masks = bool(return_masks)
        self.return_mask_stack = bool(return_mask_stack)
        self.use_semantic_mask_stack_collate = bool(self.return_mask_stack)
        self.degrade = bool(degrade)
        self.degrade_seed = int(degrade_seed)
        cfg = dict(degrade_config) if isinstance(degrade_config, dict) else {}
        self.degrade_config = {
            "blur_prob": float(cfg.get("blur_prob", 0.55)),
            "stride_skew_prob": float(cfg.get("stride_skew_prob", 0.40)),
            "dropout_prob": float(cfg.get("dropout_prob", 0.25)),
            "quantization_prob": float(cfg.get("quantization_prob", 0.30)),
            "noise_prob": float(cfg.get("noise_prob", 0.55)),
            "noise_std_min": float(cfg.get("noise_std_min", 0.01)),
            "noise_std_max": float(cfg.get("noise_std_max", 0.08)),
        }

    def __len__(self) -> int:
        return int(len(self.rows))

    def _load_rgb01(self, path: str) -> np.ndarray:
        with Image.open(str(path)) as im:
            rgb = im.convert("RGB")
            if int(rgb.size[0]) != int(self.image_size) or int(rgb.size[1]) != int(self.image_size):
                rgb = rgb.resize((int(self.image_size), int(self.image_size)), resample=Image.BILINEAR)
            arr = np.asarray(rgb, dtype=np.float32)
        arr = np.clip(arr / 255.0, 0.0, 1.0).astype(np.float32, copy=False)
        return np.transpose(arr, (2, 0, 1)).astype(np.float32, copy=False)

    def _load_mask01(self, row: SemanticDiskRow, h: int, w: int, image_rgb01: Optional[np.ndarray] = None) -> np.ndarray:
        inferred_mask = infer_semantic_support_mask(
            image=(image_rgb01 if image_rgb01 is not None else np.zeros((3, int(h), int(w)), dtype=np.float32)),
            terms=row.terms,
        )

        def _blend_with_inferred(base_mask: Any, explicit_weight: float = 0.90, inferred_weight: float = 1.10) -> np.ndarray:
            base_norm = _normalize_attention_map(
                _normalize_mask_array(base_mask, height=int(h), width=int(w)),
                gamma=0.95,
                blur_kernel=3,
            )
            inferred_norm = _normalize_attention_map(np.asarray(inferred_mask, dtype=np.float32), gamma=0.92, blur_kernel=5)
            if float(np.max(base_norm)) <= 1e-8:
                return inferred_norm
            if float(np.max(inferred_norm)) <= 1e-8:
                return base_norm
            return _blend_attention_maps(
                [base_norm, inferred_norm],
                weights=[float(explicit_weight), float(inferred_weight)],
                gamma=0.94,
            )

        if row.mask_array is not None:
            arr = np.asarray(row.mask_array, dtype=np.float32)
            if int(arr.ndim) == 3:
                arr = np.mean(arr, axis=0).astype(np.float32, copy=False)
            if int(arr.shape[0]) != int(h) or int(arr.shape[1]) != int(w):
                arr = np.asarray(
                    TF.resize(Image.fromarray(np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8), mode="L"), [int(h), int(w)], interpolation=InterpolationMode.NEAREST),
                    dtype=np.float32,
                )
            return _blend_with_inferred(arr / 255.0 if float(np.max(arr)) > 1.0 else arr)
        if str(row.mask_path).strip():
            mask_path = Path(str(row.mask_path))
            if mask_path.exists():
                if str(mask_path.suffix).strip().lower() == ".mat":
                    try:
                        from scipy.io import loadmat
                    except Exception as e:
                        raise RuntimeError(
                            f"scipy is required to read Berkeley SBD mask files: {mask_path} ({type(e).__name__}: {e})"
                        ) from e
                    blob = loadmat(str(mask_path), squeeze_me=False, struct_as_record=False)
                    seg = None
                    gtcls = blob.get("GTcls", None)
                    try:
                        seg = np.asarray(gtcls[0, 0].Segmentation, dtype=np.float32)
                    except Exception:
                        try:
                            seg = np.asarray(gtcls.Segmentation[0, 0], dtype=np.float32)
                        except Exception:
                            seg = None
                    if seg is None:
                        raise RuntimeError(f"Could not extract Berkeley segmentation from mask file: {mask_path}")
                    arr = (np.asarray(seg, dtype=np.float32) > 0.0).astype(np.float32, copy=False)
                    if int(arr.shape[1]) != int(w) or int(arr.shape[0]) != int(h):
                        arr = np.asarray(
                            TF.resize(Image.fromarray((arr * 255.0).astype(np.uint8), mode="L"), [int(h), int(w)], interpolation=InterpolationMode.NEAREST),
                            dtype=np.float32,
                        )
                    return _blend_with_inferred(arr / 255.0)
                with Image.open(str(mask_path)) as im:
                    gray = im.convert("L")
                    if int(gray.size[0]) != int(w) or int(gray.size[1]) != int(h):
                        gray = gray.resize((int(w), int(h)), resample=Image.NEAREST)
                    arr = np.asarray(gray, dtype=np.float32)
                return _blend_with_inferred(arr / 255.0)
        if isinstance(row.layout, dict):
            return _blend_with_inferred(build_layout_mask(row.layout, height=int(h), width=int(w)), explicit_weight=0.75, inferred_weight=1.15)
        return _normalize_attention_map(inferred_mask, gamma=0.92, blur_kernel=5)

    def _apply_degrade(self, x: np.ndarray, mask: Optional[np.ndarray], idx: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not bool(self.degrade):
            return x, mask
        rng = np.random.default_rng(int(self.degrade_seed) + (int(idx) * 104729))
        out = np.asarray(x, dtype=np.float32).copy()
        mask_out = None if mask is None else np.asarray(mask, dtype=np.float32).copy()
        c, h, w = int(out.shape[0]), int(out.shape[1]), int(out.shape[2])
        touch = np.zeros((int(h), int(w)), dtype=np.float32)

        def _accumulate_touch(delta: Any, scale: float = 1.0, gamma: float = 1.0):
            nonlocal touch
            norm = _normalize_attention_map(delta, gamma=float(gamma), blur_kernel=3)
            if float(np.max(norm)) <= 1e-8:
                return
            touch = np.asarray(touch, dtype=np.float32) + (float(scale) * norm)

        def _mark_diff(prev_x: np.ndarray, next_x: np.ndarray, scale: float = 1.0):
            prev_g = np.mean(np.asarray(prev_x, dtype=np.float32), axis=0)
            next_g = np.mean(np.asarray(next_x, dtype=np.float32), axis=0)
            _accumulate_touch(np.abs(next_g - prev_g).astype(np.float32, copy=False), scale=float(scale), gamma=0.95)

        if float(rng.random()) < float(self.degrade_config["blur_prob"]):
            prev = np.asarray(out, dtype=np.float32).copy()
            k = int(rng.choice(np.asarray([3, 5, 7], dtype=np.int32)))
            t = torch.from_numpy(out[None, ...])
            t = F.avg_pool2d(t, kernel_size=int(k), stride=1, padding=int(k // 2))
            out = np.asarray(t[0].cpu().numpy(), dtype=np.float32)
            _mark_diff(prev, out, scale=0.85)
        if float(rng.random()) < float(self.degrade_config["stride_skew_prob"]):
            prev = np.asarray(out, dtype=np.float32).copy()
            odd_shift = int(rng.integers(1, 8))
            out[:, 1::2, :] = np.roll(out[:, 1::2, :], shift=odd_shift, axis=2)
            stripe = np.zeros((h, w), dtype=np.float32)
            stripe[1::2, :] = 1.0
            _mark_diff(prev, out, scale=0.75)
            _accumulate_touch(stripe, scale=0.35, gamma=1.10)
            if mask_out is not None:
                mask_out[1::2, :] = np.roll(mask_out[1::2, :], shift=odd_shift, axis=1)
        if float(rng.random()) < float(self.degrade_config["dropout_prob"]):
            keep = float(rng.uniform(0.78, 0.96))
            keep_mask = (rng.random((h, w), dtype=np.float32) < keep).astype(np.float32, copy=False)
            out = out * keep_mask[None, :, :]
            _accumulate_touch(1.0 - keep_mask, scale=1.0, gamma=0.90)
        if float(rng.random()) < float(self.degrade_config["quantization_prob"]):
            prev = np.asarray(out, dtype=np.float32).copy()
            lv = int(rng.choice(np.asarray([4, 6, 8, 12], dtype=np.int32)))
            out = np.round(out * float(lv - 1)) / float(max(1, lv - 1))
            _mark_diff(prev, out, scale=0.75)
        if float(rng.random()) < float(self.degrade_config["noise_prob"]):
            prev = np.asarray(out, dtype=np.float32).copy()
            std = float(rng.uniform(float(self.degrade_config["noise_std_min"]), float(self.degrade_config["noise_std_max"])))
            out = out + (std * rng.standard_normal((c, h, w), dtype=np.float32)).astype(np.float32, copy=False)
            _mark_diff(prev, out, scale=0.90)
        out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)
        if mask_out is not None:
            touch = _normalize_attention_map(touch, gamma=1.05, blur_kernel=5)
            if float(np.max(touch)) > 1e-8:
                mask_out = _blend_attention_maps(
                    [np.asarray(mask_out, dtype=np.float32), np.asarray(touch, dtype=np.float32)],
                    weights=[0.55, 1.15],
                    gamma=0.98,
                ).astype(np.float32, copy=False)
            else:
                mask_out = _normalize_attention_map(mask_out, gamma=0.95, blur_kernel=3)
        return out, mask_out

    def __getitem__(self, index: int):
        row = self.rows[int(index)]
        x = self._load_rgb01(row.image_path)
        mask_np = None
        if bool(self.return_masks):
            mask_np = self._load_mask01(row, h=int(x.shape[1]), w=int(x.shape[2]), image_rgb01=x)
        x, mask_np = self._apply_degrade(x=x, mask=mask_np, idx=int(index))
        y = np.asarray(row.label_vec, dtype=np.float32).reshape(-1)
        if bool(self.return_masks):
            mask_stack_np = np.zeros((0, int(x.shape[1]), int(x.shape[2])), dtype=np.float32)
            mask_idx_np = np.zeros((0,), dtype=np.int64)
            if bool(self.return_mask_stack):
                if row.mask_stack_array is not None and row.mask_stack_indices is not None:
                    mask_stack_np, mask_idx_np = build_label_mask_stack(
                        mixed_mask=np.asarray(mask_np, dtype=np.float32),
                        label_vec=y,
                        mask_stack_array=row.mask_stack_array,
                        mask_stack_indices=row.mask_stack_indices,
                    )
                if int(mask_stack_np.shape[0]) <= 0 or int(mask_idx_np.size) <= 0:
                    inferred_stack_np, inferred_idx_np = build_term_mask_stack_from_image(
                        image=x,
                        label_vec=y,
                    )
                    if int(inferred_stack_np.shape[0]) > 0 and int(inferred_idx_np.size) > 0:
                        mask_stack_np, mask_idx_np = inferred_stack_np, inferred_idx_np
                        mixed_from_stack = np.max(np.asarray(mask_stack_np, dtype=np.float32), axis=0)
                        mask_np = _blend_attention_maps(
                            [np.asarray(mask_np, dtype=np.float32), np.asarray(mixed_from_stack, dtype=np.float32)],
                            weights=[0.60, 1.10],
                            gamma=0.94,
                        ).astype(np.float32, copy=False)
            x_t = torch.from_numpy(np.asarray(x, dtype=np.float32))
            y_t = torch.from_numpy(np.asarray(y, dtype=np.float32))
            mask_t = torch.from_numpy(np.asarray(mask_np, dtype=np.float32)[None, ...])
            if bool(self.return_mask_stack):
                if int(mask_stack_np.shape[0]) <= 0 or int(mask_idx_np.size) <= 0:
                    mask_stack_np, mask_idx_np = build_label_mask_stack(
                        mixed_mask=np.asarray(mask_np, dtype=np.float32),
                        label_vec=y,
                        mask_stack_array=row.mask_stack_array,
                        mask_stack_indices=row.mask_stack_indices,
                    )
                return (
                    x_t,
                    y_t,
                    mask_t,
                    torch.from_numpy(np.asarray(mask_stack_np, dtype=np.float32)),
                    torch.from_numpy(np.asarray(mask_idx_np, dtype=np.int64)),
                )
            return (x_t, y_t, mask_t)
        return (
            torch.from_numpy(np.asarray(x, dtype=np.float32)),
            torch.from_numpy(np.asarray(y, dtype=np.float32)),
        )


class OverrideTargetSubsetDataset(Dataset):
    def __init__(self, dataset: Dataset, indices: Sequence[int], override_targets: np.ndarray):
        self.dataset = dataset
        self.indices = [int(i) for i in indices]
        self.override_targets = np.asarray(override_targets, dtype=np.float32)
        if int(self.override_targets.ndim) != 2:
            raise RuntimeError(
                f"OverrideTargetSubsetDataset override_targets must be [N,C], got {tuple(self.override_targets.shape)}"
            )
        if int(len(self.indices)) != int(self.override_targets.shape[0]):
            raise RuntimeError(
                "OverrideTargetSubsetDataset length mismatch: "
                f"indices={int(len(self.indices))} targets={int(self.override_targets.shape[0])}"
            )

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, index: int):
        base = self.dataset[int(self.indices[int(index)])]
        tgt = torch.from_numpy(np.asarray(self.override_targets[int(index)], dtype=np.float32).reshape(-1))
        if isinstance(base, (tuple, list)):
            if len(base) >= 5:
                return base[0], tgt, base[2], base[3], base[4]
            if len(base) >= 3:
                return base[0], tgt, base[2]
            if len(base) >= 2:
                return base[0], tgt
        if isinstance(base, dict):
            out = dict(base)
            out["y"] = tgt
            return out
        raise RuntimeError(f"Unsupported base dataset sample for override: {type(base).__name__}")


def build_layout_mask(layout: Dict[str, Any], height: int, width: int) -> np.ndarray:
    h = max(1, int(height))
    w = max(1, int(width))
    mask = np.zeros((int(h), int(w)), dtype=np.float32)
    boxes = layout.get("boxes", None) if isinstance(layout, dict) else None
    if isinstance(boxes, list):
        for box in boxes:
            if not isinstance(box, (list, tuple)) or len(box) < 4:
                continue
            x0, y0, x1, y1 = [float(v) for v in box[:4]]
            if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.01:
                ix0 = int(np.floor(np.clip(x0, 0.0, 1.0) * float(w)))
                iy0 = int(np.floor(np.clip(y0, 0.0, 1.0) * float(h)))
                ix1 = int(np.ceil(np.clip(x1, 0.0, 1.0) * float(w)))
                iy1 = int(np.ceil(np.clip(y1, 0.0, 1.0) * float(h)))
            else:
                ix0 = int(np.floor(np.clip(x0, 0.0, float(w))))
                iy0 = int(np.floor(np.clip(y0, 0.0, float(h))))
                ix1 = int(np.ceil(np.clip(x1, 0.0, float(w))))
                iy1 = int(np.ceil(np.clip(y1, 0.0, float(h))))
            ix0 = max(0, min(int(w), int(ix0)))
            ix1 = max(0, min(int(w), int(ix1)))
            iy0 = max(0, min(int(h), int(iy0)))
            iy1 = max(0, min(int(h), int(iy1)))
            if ix1 > ix0 and iy1 > iy0:
                mask[int(iy0): int(iy1), int(ix0): int(ix1)] = 1.0
    elif str(layout.get("type", "")).strip().lower() == "quadrants":
        for q in layout.get("active_quadrants", [0, 1, 2, 3]):
            qi = int(q)
            x0 = 0 if qi in (0, 2) else (w // 2)
            x1 = (w // 2) if qi in (0, 2) else w
            y0 = 0 if qi in (0, 1) else (h // 2)
            y1 = (h // 2) if qi in (0, 1) else h
            mask[int(y0): int(y1), int(x0): int(x1)] = 1.0
    return mask.astype(np.float32, copy=False)


def _find_existing_image_path(root: Path, stem: str) -> Optional[Path]:
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"):
        cand = root / f"{str(stem)}{str(ext)}"
        if cand.exists():
            return cand
    return None


def _sidecar_mask_path(image_path: Path) -> str:
    candidates = [
        image_path.with_name(f"{image_path.stem}.mask.png"),
        image_path.with_name(f"{image_path.stem}_mask.png"),
        image_path.with_name(f"{image_path.stem}.mask.jpg"),
        image_path.with_name(f"{image_path.stem}_mask.jpg"),
    ]
    for cand in candidates:
        if cand.exists():
            return str(cand)
    return ""


def _sidecar_layout(image_path: Path) -> Optional[Dict[str, Any]]:
    candidates = [
        image_path.with_name(f"{image_path.stem}.layout.json"),
        image_path.with_name(f"{image_path.stem}_layout.json"),
        image_path.with_suffix(".layout.json"),
    ]
    for cand in candidates:
        if not cand.exists():
            continue
        try:
            blob = json.loads(cand.read_text(encoding="utf-8"))
            if isinstance(blob, dict):
                return blob
        except Exception:
            continue
    return None


def _folder_source_signature(root: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"root": str(root), "exists": bool(root.exists()), "datasets": []}
    if not root.exists():
        return out
    ds_dirs = [p for p in root.iterdir() if p.is_dir()]
    ds_dirs = sorted(ds_dirs, key=lambda p: p.name.lower())
    for ds_dir in ds_dirs:
        files = [p for p in ds_dir.rglob("*") if p.is_file() and str(p.suffix).strip().lower() in _IMAGE_SUFFIXES]
        if len(files) <= 0:
            continue
        labels = set()
        latest = 0.0
        for fp in files:
            try:
                latest = max(float(latest), float(fp.stat().st_mtime))
            except Exception:
                pass
            try:
                rel = fp.relative_to(ds_dir)
                if len(rel.parts) > 1:
                    lbl = re.sub(r"[_\-]+", " ", str(rel.parts[0])).strip()
                    if str(lbl):
                        labels.add(str(lbl))
            except Exception:
                pass
        out["datasets"].append(
            {
                "name": re.sub(r"[_\-]+", " ", str(ds_dir.name)).strip(),
                "files": int(len(files)),
                "latest_mtime": int(latest),
                "labels": sorted([str(x) for x in labels], key=lambda s: s.lower()),
            }
        )
    return out


def collect_semantic_disk_rows(
    data_root: str,
    class_names: Sequence[str],
    source_root: str = "",
) -> Tuple[List[SemanticDiskRow], Dict[str, Any]]:
    root = Path(str(data_root).strip() or "toys_to_survive_development/data/berkeley_sbd")
    class_lut = {_norm_txt(name): int(i) for i, name in enumerate(class_names)}
    n_classes = max(1, int(len(class_names)))
    rows: List[SemanticDiskRow] = []
    source_counts: Dict[str, int] = {}
    loaded_split_counts: Dict[str, int] = {"train": 0, "val": 0}
    missing_images = 0

    berkeley_dataset_idx = int(class_lut.get("berkeley sbd dataset", -1))
    object_idx = int(class_lut.get("object", -1))
    signal_idx = int(class_lut.get("signal", -1))

    def _augment_row_with_auto_color_terms(image_path: Path, label_vec: np.ndarray, terms_in: Sequence[str], mask_path: str = "") -> Tuple[np.ndarray, List[str]]:
        out_vec = np.asarray(label_vec, dtype=np.float32).reshape(-1).copy()
        out_terms = normalize_vocab_terms([str(x) for x in list(terms_in)])
        try:
            with Image.open(str(image_path)) as im:
                rgb = np.asarray(im.convert("RGB"), dtype=np.float32)
            mask_arr = None
            if str(mask_path).strip():
                try:
                    mp = Path(str(mask_path))
                    if mp.exists() and str(mp.suffix).strip().lower() != ".mat":
                        with Image.open(str(mp)) as mm:
                            mask_arr = np.asarray(mm.convert("L"), dtype=np.float32)
                            mask_arr = np.clip(mask_arr / 255.0, 0.0, 1.0).astype(np.float32, copy=False)
                except Exception:
                    mask_arr = None
            color_terms = detect_semantic_color_terms(image=rgb, mask=mask_arr)
        except Exception:
            color_terms = []
        if len(color_terms) <= 0:
            return out_vec, out_terms
        for term in color_terms:
            idx = int(class_lut.get(_norm_txt(term), -1))
            if 0 <= int(idx) < int(out_vec.size):
                out_vec[int(idx)] = 1.0
        return out_vec, normalize_vocab_terms(list(out_terms) + list(color_terms))

    split_specs = [("train", "berkeley_sbd_train"), ("val", "berkeley_sbd_val")]
    for split_name, source_key in split_specs:
        ds = SBDataset(root=str(root), image_set=split_name, mode="segmentation", download=False)
        label_path = root / "cache" / f"sbd_{split_name}_multilabel.npz"
        if not label_path.exists():
            raise RuntimeError(
                "Berkeley multilabel cache is missing for disk rows: "
                f"{label_path}. Generate label caches before running the pipeline."
            )
        with np.load(str(label_path), allow_pickle=False) as z:
            if "labels" not in z.files:
                raise RuntimeError(f"'labels' key missing in {label_path}")
            labels_split = np.asarray(z["labels"], dtype=np.float32)
        if int(labels_split.shape[0]) != int(len(ds.images)):
            raise RuntimeError(
                "Berkeley image/label row mismatch for disk rows: "
                f"split={split_name} images={int(len(ds.images))} labels={int(labels_split.shape[0])}"
            )
        for i, img_path in enumerate(ds.images):
            ip = Path(str(img_path))
            if not ip.exists():
                missing_images += 1
                continue
            yv = np.asarray(labels_split[int(i)], dtype=np.float32).reshape(-1)
            if int(yv.size) > int(n_classes):
                yv = yv[: int(n_classes)]
            elif int(yv.size) < int(n_classes):
                pad = np.zeros((int(n_classes) - int(yv.size),), dtype=np.float32)
                yv = np.concatenate([yv, pad], axis=0)
            if int(berkeley_dataset_idx) >= 0:
                yv[int(berkeley_dataset_idx)] = 1.0
            if int(object_idx) >= 0:
                yv[int(object_idx)] = 1.0
            if int(signal_idx) >= 0:
                yv[int(signal_idx)] = 1.0
            yv, row_terms = _augment_row_with_auto_color_terms(
                image_path=ip,
                label_vec=yv,
                terms_in=["berkeley sbd dataset", "object", "signal"],
                mask_path=(str(ds.masks[int(i)]) if int(i) < int(len(ds.masks)) else ""),
            )
            pos = np.where(np.asarray(yv, dtype=np.float32) > 0.5)[0].astype(np.int64).tolist()
            label_terms = [str(class_names[int(j)]) for j in pos if 0 <= int(j) < int(len(class_names))]
            terms = normalize_vocab_terms(list(row_terms) + list(label_terms))
            rows.append(
                SemanticDiskRow(
                    image_path=str(ip),
                    label_vec=np.asarray(yv, dtype=np.float32).reshape(-1),
                    terms=list(terms),
                    source=str(source_key),
                    mask_path=str(ds.masks[int(i)]) if int(i) < int(len(ds.masks)) else "",
                )
            )
            source_counts[str(source_key)] = int(source_counts.get(str(source_key), 0)) + 1
            loaded_split_counts[str(split_name)] = int(loaded_split_counts.get(str(split_name), 0)) + 1

    ext_root = Path(str(source_root).strip()) if str(source_root).strip() else (root / "payload_sources")
    ext_sig = _folder_source_signature(ext_root)
    external_unmapped_skipped = 0
    if ext_root.exists():
        ds_dirs = [p for p in ext_root.iterdir() if p.is_dir()]
        ds_dirs = sorted(ds_dirs, key=lambda p: p.name.lower())
        for ds_dir in ds_dirs:
            dataset_name = re.sub(r"[_\-]+", " ", str(ds_dir.name)).strip()
            if not str(dataset_name):
                continue
            files = [p for p in ds_dir.rglob("*") if p.is_file() and str(p.suffix).strip().lower() in _IMAGE_SUFFIXES]
            files = sorted(files, key=lambda p: str(p).lower())
            for fp in files:
                rel = fp.relative_to(ds_dir)
                label_term = ""
                if len(rel.parts) > 1:
                    label_term = re.sub(r"[_\-]+", " ", str(rel.parts[0])).strip()
                vec = np.zeros((int(n_classes),), dtype=np.float32)
                if int(berkeley_dataset_idx) >= 0:
                    vec[int(berkeley_dataset_idx)] = 1.0
                if int(signal_idx) >= 0:
                    vec[int(signal_idx)] = 1.0
                dataset_terms = [str(dataset_name)]
                if not str(dataset_name).strip().lower().endswith("dataset"):
                    dataset_terms.append(f"{dataset_name} dataset")
                specific_candidates = list(dataset_terms)
                if str(label_term):
                    specific_candidates.append(str(label_term))
                mapped_terms: List[str] = []
                specific_hits = 0
                specific_set = {str(x) for x in specific_candidates}
                for cand in (["berkeley sbd dataset", "signal"] + list(specific_candidates)):
                    key = _norm_txt(cand)
                    if key in class_lut:
                        cls_idx = int(class_lut[key])
                        vec[int(cls_idx)] = 1.0
                        mapped_terms.append(str(class_names[int(cls_idx)]))
                        if str(cand) in specific_set:
                            specific_hits += 1
                if int(specific_hits) <= 0:
                    external_unmapped_skipped += 1
                    continue
                vec, auto_terms = _augment_row_with_auto_color_terms(
                    image_path=fp,
                    label_vec=vec,
                    terms_in=mapped_terms,
                    mask_path=_sidecar_mask_path(fp),
                )
                rows.append(
                    SemanticDiskRow(
                        image_path=str(fp),
                        label_vec=np.asarray(vec, dtype=np.float32).reshape(-1),
                        terms=list(normalize_vocab_terms(auto_terms)),
                        source=str(dataset_name),
                        mask_path=_sidecar_mask_path(fp),
                        layout=_sidecar_layout(fp),
                    )
                )
                source_counts[str(dataset_name)] = int(source_counts.get(str(dataset_name), 0)) + 1

    info = {
        "available_rows": int(len(rows)),
        "available_train": int(loaded_split_counts.get("train", 0)),
        "available_val": int(loaded_split_counts.get("val", 0)),
        "available_external": int(max(0, int(len(rows)) - int(loaded_split_counts.get("train", 0)) - int(loaded_split_counts.get("val", 0)))),
        "source_counts": {str(k): int(v) for k, v in source_counts.items()},
        "missing_images": int(missing_images),
        "external_source_root": str(ext_root),
        "external_source_signature": ext_sig,
        "external_unmapped_skipped": int(external_unmapped_skipped),
    }
    return rows, info
