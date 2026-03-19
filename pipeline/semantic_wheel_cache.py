from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, Sampler

from pipeline.progress import interruptible_tqdm
from semantic_dataset_loaders import (
    SemanticDiskRow,
    _apply_degrade as _canonical_apply_degrade,
    _composite_mask_stack,
    _norm_txt,
    _resolve_processing_device,
    build_creation_label_mask_stack,
    build_label_mask_stack,
    build_layout_mask,
    build_term_mask_stack_from_image,
    build_term_mask_stacks_from_images,
    combine_label_mask_stacks,
    infer_semantic_support_mask,
    infer_semantic_support_masks,
    normalize_vocab_terms,
    term_mask_map_to_label_stack,
)


def _sanitize_component(name: str) -> str:
    txt = str(name).strip().lower()
    txt = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in txt)
    while "__" in txt:
        txt = txt.replace("__", "_")
    return txt.strip("_") or "wheel"


def _fit_pil_to_square(im: Image.Image, image_size: int, fill: int = 0) -> Image.Image:
    size = max(8, int(image_size))
    src_w, src_h = im.size
    if src_w <= 0 or src_h <= 0:
        return Image.new(im.mode, (size, size), color=fill)
    scale = 1.0
    if int(src_w) > int(size) or int(src_h) > int(size):
        scale = min(float(size) / float(src_w), float(size) / float(src_h))
    new_w = max(1, int(round(float(src_w) * float(scale))))
    new_h = max(1, int(round(float(src_h) * float(scale))))
    if int(new_w) != int(src_w) or int(new_h) != int(src_h):
        resample = Image.BILINEAR if im.mode != "L" else Image.NEAREST
        im = im.resize((int(new_w), int(new_h)), resample=resample)
    out = Image.new(im.mode, (size, size), color=fill)
    left = max(0, int((size - int(new_w)) // 2))
    top = max(0, int((size - int(new_h)) // 2))
    out.paste(im, (left, top))
    return out


def _load_fit_rgb_u8(path: str, image_size: int) -> np.ndarray:
    with Image.open(str(path)) as im:
        rgb = _fit_pil_to_square(im.convert("RGB"), image_size=int(image_size), fill=0)
        arr = np.asarray(rgb, dtype=np.uint8)
    return np.transpose(arr, (2, 0, 1)).astype(np.uint8, copy=False)


def _fit_mask_array(mask: np.ndarray, image_size: int) -> np.ndarray:
    """Resize a float32 [0,1] mask to (image_size, image_size).  No quantization."""
    arr = np.asarray(mask, dtype=np.float32)
    if int(arr.ndim) == 3:
        arr = np.mean(arr, axis=0).astype(np.float32, copy=False)
    arr = np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)
    size = max(8, int(image_size))
    h, w = int(arr.shape[0]), int(arr.shape[1])
    if h != size or w != size:
        arr = torch.nn.functional.interpolate(
            torch.from_numpy(arr[None, None, ...]),
            size=(size, size),
            mode='nearest',
        )[0, 0].numpy().astype(np.float32, copy=False)
    return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)


def _positive_label_bits(label_vec: np.ndarray) -> np.ndarray:
    return (np.asarray(label_vec, dtype=np.float32).reshape(-1) >= 0.5).astype(np.uint8, copy=False)


def _resolve_preload_workers(row_count: int, requested_workers: int = 0, max_cap: int = 32) -> int:
    if int(row_count) <= 1:
        return 1
    if int(requested_workers) > 0:
        return max(1, min(int(max_cap), int(row_count), int(requested_workers)))
    cpu_count = max(1, int(os.cpu_count() or 1))
    target = max(4, int(cpu_count) * 2)
    return max(1, min(int(max_cap), int(row_count), int(target)))


def _load_row_assets(row: SemanticDiskRow, image_size: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    return (
        _load_fit_rgb_u8(row.image_path, image_size=int(image_size)),
        _row_creation_mask(row, image_size=int(image_size)),
    )


def _preload_row_assets_batch(
    rows: Sequence[SemanticDiskRow],
    image_size: int,
    preload_workers: int = 0,
) -> Tuple[np.ndarray, List[Optional[np.ndarray]]]:
    if len(rows) <= 0:
        size = max(8, int(image_size))
        return np.zeros((0, 3, size, size), dtype=np.uint8), []
    workers = _resolve_preload_workers(len(rows), requested_workers=int(preload_workers))
    if int(workers) <= 1:
        loaded = [_load_row_assets(row, image_size=int(image_size)) for row in rows]
    else:
        with ThreadPoolExecutor(max_workers=int(workers)) as pool:
            loaded = list(pool.map(lambda row: _load_row_assets(row, image_size=int(image_size)), rows))
    image_u8_batch = np.stack([np.asarray(item[0], dtype=np.uint8) for item in loaded], axis=0).astype(np.uint8, copy=False)
    creation_masks_u8 = [item[1] for item in loaded]
    return image_u8_batch, creation_masks_u8


def _fit_image_array_u8(image: Any, image_size: int) -> np.ndarray:
    arr = np.asarray(image)
    if int(arr.ndim) == 3 and int(arr.shape[0]) in (1, 3, 4):
        chw = np.asarray(arr[:3], dtype=np.float32)
        if int(chw.shape[0]) == 1:
            chw = np.repeat(chw, 3, axis=0)
        hwc = np.transpose(chw, (1, 2, 0))
    elif int(arr.ndim) == 3 and int(arr.shape[2]) in (1, 3, 4):
        hwc = np.asarray(arr[..., :3], dtype=np.float32)
        if int(hwc.shape[2]) == 1:
            hwc = np.repeat(hwc, 3, axis=2)
    elif int(arr.ndim) == 2:
        gray = np.asarray(arr, dtype=np.float32)
        hwc = np.repeat(gray[:, :, None], 3, axis=2)
    else:
        raise RuntimeError(f"semantic wheel image row must be CHW/HWC/gray, got {tuple(arr.shape)}")
    if not np.issubdtype(np.asarray(arr).dtype, np.integer):
        hwc = np.clip(np.rint(np.clip(hwc, 0.0, 1.0) * 255.0), 0.0, 255.0).astype(np.uint8, copy=False)
    else:
        hwc = np.clip(hwc, 0.0, 255.0).astype(np.uint8, copy=False)
    fitted = _fit_pil_to_square(Image.fromarray(hwc, mode="RGB"), image_size=int(image_size), fill=0)
    out = np.asarray(fitted, dtype=np.uint8)
    return np.transpose(out, (2, 0, 1)).astype(np.uint8, copy=False)


def build_semantic_cache_entry(
    *,
    image: Any,
    label_vec: Any,
    image_size: int,
    terms: Optional[Sequence[str]] = None,
    mixed_mask: Optional[Any] = None,
    mask_stack: Optional[Any] = None,
    mask_indices: Optional[Any] = None,
) -> Dict[str, Any]:
    size = max(8, int(image_size))
    image_u8 = _fit_image_array_u8(image=image, image_size=size)
    image_f32 = np.clip(np.asarray(image_u8, dtype=np.float32) / 255.0, 0.0, 1.0).astype(np.float32, copy=False)
    label_arr = np.asarray(label_vec, dtype=np.float32).reshape(-1)
    normalized_terms = list(normalize_vocab_terms([str(x) for x in list(terms or [])]))
    if mixed_mask is None:
        mixed_mask_arr = infer_semantic_support_mask(image=image_f32, terms=normalized_terms)
    else:
        mixed_mask_arr = _fit_mask_array(np.asarray(mixed_mask, dtype=np.float32), image_size=size)
    stack_arr = np.asarray(mask_stack) if mask_stack is not None else np.zeros((0, size, size), dtype=np.float32)
    idx_arr = np.asarray(mask_indices, dtype=np.int64).reshape(-1) if mask_indices is not None else np.zeros((0,), dtype=np.int64)
    if int(stack_arr.ndim) == 2:
        stack_arr = stack_arr[None, ...]
    if int(stack_arr.ndim) == 3 and int(stack_arr.shape[0]) > 0:
        fitted_parts = [
            _fit_mask_array(np.asarray(stack_arr[int(i)], dtype=np.float32), image_size=size)
            for i in range(int(stack_arr.shape[0]))
        ]
        stack_arr = np.stack(fitted_parts, axis=0).astype(np.float32, copy=False)
    else:
        stack_arr = np.zeros((0, size, size), dtype=np.float32)
    pair_count = min(int(stack_arr.shape[0]), int(idx_arr.size))
    stack_arr = np.asarray(stack_arr[: int(pair_count)], dtype=np.float32)
    idx_arr = np.asarray(idx_arr[: int(pair_count)], dtype=np.int64)
    if int(stack_arr.shape[0]) > 0:
        mixed_mask_arr = _composite_mask_stack(stack_arr)
    return {
        "image_u8": np.asarray(image_u8, dtype=np.uint8),
        "label_vec_u8": _positive_label_bits(label_arr),
        "mixed_mask": np.clip(np.asarray(mixed_mask_arr, dtype=np.float32), 0.0, 1.0),
        "mask_stack": np.asarray(stack_arr, dtype=np.float32),
        "mask_indices": np.asarray(idx_arr, dtype=np.int32).reshape(-1),
        "terms": list(normalized_terms),
    }


@dataclass
class SemanticWheelCandidate:
    cache_key: str
    terms: List[str]
    source: str = ""


def _row_creation_mask(row: SemanticDiskRow, image_size: int) -> Optional[np.ndarray]:
    """Load or derive a float32 [0,1] creation mask for *row*.  No normalization."""
    size = max(8, int(image_size))
    if row.mask_array is not None:
        return _fit_mask_array(np.asarray(row.mask_array, dtype=np.float32), image_size=size)
    if str(row.mask_path).strip():
        mask_path = Path(str(row.mask_path))
        if mask_path.exists():
            if str(mask_path.suffix).strip().lower() in (".mat", ".npz"):
                npz_path = mask_path.with_suffix(".npz")
                mat_path = mask_path.with_suffix(".mat")
                seg = None
                try:
                    if npz_path.exists():
                        with np.load(str(npz_path), allow_pickle=False) as z:
                            seg = np.asarray(z["segmentation"], dtype=np.float32)
                    elif mat_path.exists():
                        from scipy.io import loadmat

                        blob = loadmat(str(mat_path), squeeze_me=False, struct_as_record=False)
                        gtcls = blob.get("GTcls", None)
                        try:
                            seg = np.asarray(gtcls[0, 0].Segmentation, dtype=np.float32)
                        except Exception:
                            try:
                                seg = np.asarray(gtcls.Segmentation[0, 0], dtype=np.float32)
                            except Exception:
                                seg = None
                except Exception:
                    seg = None
                if seg is not None:
                    seg01 = (np.asarray(seg, dtype=np.float32) > 0.0).astype(np.float32, copy=False)
                    return _fit_mask_array(seg01, image_size=size)
            try:
                with Image.open(str(mask_path)) as im:
                    gray = _fit_pil_to_square(im.convert("L"), image_size=size, fill=0)
                    arr = np.asarray(gray, dtype=np.float32)
                    if float(np.max(arr)) > 1.0:
                        arr = arr / 255.0
                    return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)
            except Exception:
                pass
    if isinstance(row.layout, dict):
        return np.clip(
            np.asarray(build_layout_mask(row.layout, height=size, width=size), dtype=np.float32),
            0.0, 1.0,
        ).astype(np.float32, copy=False)
    return None


def _sobel_edge_map(gray: np.ndarray) -> np.ndarray:
    """Fast Sobel gradient magnitude edge map, normalised to [0, 1]."""
    g = np.asarray(gray, dtype=np.float32)
    if int(g.ndim) != 2 or int(g.shape[0]) < 3 or int(g.shape[1]) < 3:
        return np.zeros_like(g)
    t = torch.from_numpy(g[None, None, ...])
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
    gx = F.conv2d(t, kx, padding=1)
    gy = F.conv2d(t, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-12)[0, 0].numpy()
    vmax = float(np.max(mag))
    if vmax > 1e-8:
        mag = mag / vmax
    return np.clip(mag, 0.0, 1.0).astype(np.float32, copy=False)


def _sobel_edge_map_torch(gray: torch.Tensor) -> torch.Tensor:
    g = gray.to(dtype=torch.float32)
    if int(g.ndim) != 2 or int(g.shape[0]) < 3 or int(g.shape[1]) < 3:
        return torch.zeros_like(g, dtype=torch.float32)
    t = g[None, None, ...]
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=g.device).reshape(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=g.device).reshape(1, 1, 3, 3)
    gx = F.conv2d(t, kx, padding=1)
    gy = F.conv2d(t, ky, padding=1)
    mag = torch.sqrt((gx * gx) + (gy * gy) + 1e-12)[0, 0]
    vmax = torch.amax(mag)
    if float(vmax.detach().cpu().item()) > 1e-8:
        mag = mag / vmax
    return torch.clamp(mag, 0.0, 1.0).to(dtype=torch.float32)


def _canny_edge_map(
    gray: np.ndarray,
    sigma: float = 1.4,
    low_ratio: float = 0.05,
    high_ratio: float = 0.15,
) -> np.ndarray:
    """Ultra-quality Canny-like edge detection with Gaussian smoothing,
    non-maximum suppression, and dual-threshold hysteresis.  Pure numpy+torch,
    no OpenCV dependency."""
    g = np.asarray(gray, dtype=np.float32)
    if int(g.ndim) != 2 or int(g.shape[0]) < 5 or int(g.shape[1]) < 5:
        return np.zeros_like(g)
    # Step 1: Gaussian blur
    ks = max(3, int(2 * int(round(3.0 * float(sigma))) + 1))
    ax = np.arange(-ks // 2 + 1, ks // 2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * float(sigma) ** 2)).astype(np.float32)
    kernel = kernel / float(np.sum(kernel))
    kt = torch.from_numpy(kernel[None, None, ...])
    t = torch.from_numpy(g[None, None, ...])
    smoothed = F.conv2d(t, kt, padding=ks // 2)[0, 0].numpy()
    # Step 2: Sobel gradients
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    st = torch.from_numpy(smoothed[None, None, ...])
    gx = F.conv2d(st, torch.from_numpy(kx[None, None, ...]), padding=1)[0, 0].numpy()
    gy = F.conv2d(st, torch.from_numpy(ky[None, None, ...]), padding=1)[0, 0].numpy()
    mag = np.sqrt(gx ** 2 + gy ** 2 + 1e-12).astype(np.float32)
    angle = np.arctan2(gy, gx)
    # Step 3: Non-maximum suppression
    h, w = mag.shape
    nms = np.zeros_like(mag)
    angle_deg = (np.degrees(angle) % 180.0).astype(np.float32)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            a = float(angle_deg[y, x])
            if (0.0 <= a < 22.5) or (157.5 <= a <= 180.0):
                n1, n2 = float(mag[y, x + 1]), float(mag[y, x - 1])
            elif 22.5 <= a < 67.5:
                n1, n2 = float(mag[y + 1, x + 1]), float(mag[y - 1, x - 1])
            elif 67.5 <= a < 112.5:
                n1, n2 = float(mag[y + 1, x]), float(mag[y - 1, x])
            else:
                n1, n2 = float(mag[y + 1, x - 1]), float(mag[y - 1, x + 1])
            if float(mag[y, x]) >= n1 and float(mag[y, x]) >= n2:
                nms[y, x] = float(mag[y, x])
    # Step 4: Dual-threshold hysteresis
    vmax = float(np.max(nms))
    if vmax < 1e-8:
        return np.zeros_like(g)
    high_thresh = float(high_ratio) * vmax
    low_thresh = float(low_ratio) * vmax
    strong = (nms >= high_thresh).astype(np.uint8)
    weak = ((nms >= low_thresh) & (nms < high_thresh)).astype(np.uint8)
    # Connect weak edges adjacent to strong edges
    out = np.asarray(strong, dtype=np.uint8).copy()
    changed = True
    while changed:
        changed = False
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if int(weak[y, x]) > 0 and int(out[y, x]) == 0:
                    if int(np.max(out[y - 1:y + 2, x - 1:x + 2])) > 0:
                        out[y, x] = 1
                        changed = True
    return np.clip(out.astype(np.float32), 0.0, 1.0).astype(np.float32, copy=False)


def _apply_degrade(
    x: np.ndarray,
    mask: Optional[np.ndarray],
    idx: int,
    seed: int,
    degrade_config: Optional[Dict[str, Any]] = None,
    processing_device: Optional[Any] = None,
    apply_vmean_compression: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, np.ndarray]]:
    """Delegates to the canonical _apply_degrade in semantic_dataset_loaders."""
    return _canonical_apply_degrade(
        x,
        seed=int(seed),
        idx=int(idx),
        mask=mask,
        degrade_config=degrade_config,
    )


def _build_clean_entry(
    row: SemanticDiskRow,
    image_size: int,
    idx_to_term: Dict[int, str],
    processing_device: Optional[Any] = None,
) -> Dict[str, Any]:
    return _build_clean_entries_batch(
        rows=[row],
        image_size=int(image_size),
        idx_to_term=idx_to_term,
        processing_device=processing_device,
    )[0]


def _build_clean_entries_batch(
    rows: Sequence[SemanticDiskRow],
    image_size: int,
    idx_to_term: Dict[int, str],
    processing_device: Optional[Any] = None,
    preload_workers: int = 0,
    progress_control: Any = None,
) -> List[Dict[str, Any]]:
    if len(rows) <= 0:
        return []
    size = max(8, int(image_size))
    image_u8_batch, creation_masks_u8 = _preload_row_assets_batch(
        rows=rows,
        image_size=size,
        preload_workers=int(preload_workers),
    )
    image_batch = np.clip(np.asarray(image_u8_batch, dtype=np.float32) / 255.0, 0.0, 1.0).astype(np.float32, copy=False)
    label_batch = np.stack(
        [np.asarray(row.label_vec, dtype=np.float32).reshape(-1) for row in rows],
        axis=0,
    ).astype(np.float32, copy=False)

    # Enrich each row with tonal/color terms using the same function pregestation uses,
    # then update label_batch so build_term_mask_stacks_from_images generates spatial
    # masks for those terms.
    _tonal_enrich = None
    try:
        from pipeline.nodes.vocab_node import _semantic_terms_with_tonal_masks as _tonal_enrich
    except Exception:
        pass
    term_to_idx: Dict[str, int] = {str(v).strip().lower(): int(k) for k, v in idx_to_term.items()}
    tonal_masks_per_row: List[Dict[str, np.ndarray]] = [{} for _ in rows]
    enriched_terms_per_row: List[List[str]] = [list(row.terms) for row in rows]
    if _tonal_enrich is not None:
        for _ri, _row in enumerate(rows):
            try:
                _enriched, _tmasks = _tonal_enrich(
                    terms=list(_row.terms),
                    image=image_u8_batch[int(_ri)],
                    image_size=size,
                )
                enriched_terms_per_row[int(_ri)] = list(_enriched)
                tonal_masks_per_row[int(_ri)] = dict(_tmasks)
                for _term in _enriched:
                    _t = str(_term).strip().lower()
                    _tidx = int(term_to_idx.get(_t, -1))
                    if 0 <= _tidx < int(label_batch.shape[1]):
                        label_batch[int(_ri), _tidx] = 1.0
            except Exception:
                pass

    fallback_masks = infer_semantic_support_masks(
        images=image_batch,
        terms_batch=enriched_terms_per_row,
        processing_device=processing_device,
    )
    heuristic_stacks, heuristic_indices = build_term_mask_stacks_from_images(
        images=image_batch,
        label_vecs=label_batch,
        idx_to_term=idx_to_term,
        processing_device=processing_device,
    )

    out: List[Dict[str, Any]] = []
    for row_idx, row in interruptible_tqdm(
        enumerate(rows),
        total=len(rows),
        desc="[wheel cache] building entries",
        unit="row",
        leave=False,
        dynamic_ncols=True,
        control=progress_control,
    ):
        label_vec = np.asarray(label_batch[int(row_idx)], dtype=np.float32)
        creation_mask_u8 = creation_masks_u8[int(row_idx)]
        creation_mask = np.asarray(creation_mask_u8, dtype=np.float32) if creation_mask_u8 is not None else None
        creation_stack = np.zeros((0, size, size), dtype=np.float32)
        creation_idx = np.zeros((0,), dtype=np.int64)
        if creation_mask is not None:
            creation_stack, creation_idx = build_creation_label_mask_stack(
                label_vec=label_vec,
                height=size,
                width=size,
                creation_mask=np.asarray(creation_mask, dtype=np.float32),
                processing_device=processing_device,
            )
        # Build tonal mask stack from per-image color/tonal detections
        _tmasks = tonal_masks_per_row[int(row_idx)]
        _tonal_slices: List[np.ndarray] = []
        _tonal_idxs: List[int] = []
        for _tterm, _tmask in _tmasks.items():
            _t = str(_tterm).strip().lower()
            _tidx = int(term_to_idx.get(_t, -1))
            if _tidx < 0:
                continue
            _tmask_np = np.asarray(_tmask, dtype=np.float32)
            if int(_tmask_np.ndim) == 2 and int(_tmask_np.size) > 0:
                _tonal_slices.append(_tmask_np)
                _tonal_idxs.append(_tidx)
        if _tonal_slices:
            tonal_stack = np.stack(_tonal_slices, axis=0).astype(np.float32, copy=False)
            tonal_idx = np.asarray(_tonal_idxs, dtype=np.int64)
        else:
            tonal_stack = np.zeros((0, size, size), dtype=np.float32)
            tonal_idx = np.zeros((0,), dtype=np.int64)
        heuristic_stack = np.asarray(heuristic_stacks[int(row_idx)], dtype=np.float32)
        heuristic_idx = np.asarray(heuristic_indices[int(row_idx)], dtype=np.int64)
        mask_stack, mask_indices = combine_label_mask_stacks(
            label_vec,
            (creation_stack, creation_idx),
            (tonal_stack, tonal_idx),
            (heuristic_stack, heuristic_idx),
            height=size,
            width=size,
            processing_device=processing_device,
            strict=True,
        )
        mixed_mask = (
            _composite_mask_stack(mask_stack, processing_device=processing_device)
            if int(mask_stack.shape[0]) > 0
            else np.zeros((size, size), dtype=np.float32)
        )
        mixed_mask = np.asarray(mixed_mask, dtype=np.float32)
        mask_stack = np.asarray(mask_stack, dtype=np.float32)
        out.append(
            {
                "image_u8": np.asarray(image_u8_batch[int(row_idx)], dtype=np.uint8),
                "label_vec_u8": _positive_label_bits(label_vec),
                "mixed_mask": np.clip(np.asarray(mixed_mask, dtype=np.float32), 0.0, 1.0),
                "mask_stack": np.asarray(mask_stack, dtype=np.float32),
                "mask_indices": np.asarray(mask_indices, dtype=np.int32).reshape(-1),
                "terms": list(normalize_vocab_terms(enriched_terms_per_row[int(row_idx)])),
            }
        )
    return out


def _build_deformed_entry(
    clean_entry: Dict[str, Any],
    variant_idx: int,
    base_row_position: int,
    seed: int,
    term_to_idx: Dict[str, int],
    degrade_config: Optional[Dict[str, Any]] = None,
    processing_device: Optional[Any] = None,
) -> Dict[str, Any]:
    image = np.clip(np.asarray(clean_entry["image_u8"], dtype=np.float32) / 255.0, 0.0, 1.0).astype(np.float32, copy=False)
    base_mask = np.asarray(clean_entry["mixed_mask"], dtype=np.float32)
    x, mask_out, term_masks = _apply_degrade(
        x=image,
        mask=base_mask,
        idx=int(base_row_position),
        seed=int(seed) + (int(variant_idx) + 1) * 7919,
        degrade_config=degrade_config,
        processing_device=processing_device,
    )
    label_vec = np.asarray(clean_entry["label_vec_u8"], dtype=np.uint8).astype(np.float32, copy=False)
    for term in normalize_vocab_terms(list(term_masks.keys())):
        ti = int(term_to_idx.get(_norm_txt(term), -1))
        if 0 <= int(ti) < int(label_vec.size):
            label_vec[int(ti)] = 1.0
    size = int(x.shape[1])
    base_stack = np.asarray(clean_entry["mask_stack"], dtype=np.float32)
    base_idx = np.asarray(clean_entry["mask_indices"], dtype=np.int64).reshape(-1)
    distortion_stack, distortion_idx = term_mask_map_to_label_stack(
        term_masks,
        label_vec,
        term_to_idx=term_to_idx,
        height=size,
        width=size,
        processing_device=processing_device,
    )
    mask_stack, mask_indices = combine_label_mask_stacks(
        label_vec,
        (base_stack, base_idx),
        (distortion_stack, distortion_idx),
        height=size,
        width=size,
        processing_device=processing_device,
    )
    mixed_mask = (
        _composite_mask_stack(mask_stack, processing_device=processing_device)
        if int(mask_stack.shape[0]) > 0
        else np.zeros((size, size), dtype=np.float32)
    )
    return {
        "image_u8": np.clip(np.rint(np.clip(x, 0.0, 1.0) * 255.0), 0.0, 255.0).astype(np.uint8, copy=False),
        "label_vec_u8": _positive_label_bits(label_vec),
        "mixed_mask": np.clip(np.asarray(mixed_mask, dtype=np.float32), 0.0, 1.0),
        "mask_stack": np.asarray(mask_stack, dtype=np.float32),
        "mask_indices": np.asarray(mask_indices, dtype=np.int32).reshape(-1),
        "terms": list(
            normalize_vocab_terms(
                list(clean_entry.get("terms") or [])
                + [str(x) for x in list(term_masks.keys())]
            )
        ),
    }


def _candidate_weight_map(candidates: Sequence[SemanticWheelCandidate], candidate_indices: Sequence[int]) -> Dict[int, float]:
    term_freq: Dict[str, int] = {}
    for row_idx in candidate_indices:
        row = candidates[int(row_idx)]
        keys = {
            _norm_txt(term)
            for term in normalize_vocab_terms([str(x) for x in list(row.terms)])
            if str(term).strip()
        }
        for key in keys:
            term_freq[str(key)] = int(term_freq.get(str(key), 0)) + 1
    out: Dict[int, float] = {}
    for row_idx in candidate_indices:
        row = candidates[int(row_idx)]
        keys = {
            _norm_txt(term)
            for term in normalize_vocab_terms([str(x) for x in list(row.terms)])
            if str(term).strip()
        }
        weight = 1.0
        for key in keys:
            weight += 1.0 / float(max(1, int(term_freq.get(str(key), 1))))
        out[int(row_idx)] = float(max(weight, 1e-4))
    return out


def _candidate_signature(candidates: Sequence[SemanticWheelCandidate], candidate_indices: Sequence[int]) -> str:
    hasher = hashlib.sha256()
    for row_idx in candidate_indices:
        row = candidates[int(row_idx)]
        payload = {
            "cache_key": str(row.cache_key),
            "terms": list(normalize_vocab_terms([str(x) for x in list(row.terms)])),
            "source": str(row.source),
        }
        hasher.update(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8"))
    return str(hasher.hexdigest())


def _weighted_deck_order(
    candidates: Sequence[SemanticWheelCandidate],
    candidate_indices: Sequence[int],
    seed: int,
    epoch: int,
) -> List[int]:
    items = [int(i) for i in candidate_indices]
    if len(items) <= 1:
        return items
    weights = _candidate_weight_map(candidates=candidates, candidate_indices=items)
    rng = np.random.default_rng(max(0, int(seed)) + (int(epoch) * 104729))
    u = np.clip(rng.random(len(items), dtype=np.float64), 1e-9, 1.0)
    keys = []
    for pos, row_idx in enumerate(items):
        w = float(max(1e-6, float(weights.get(int(row_idx), 1.0))))
        keys.append(float(math.log(float(u[int(pos)])) / w))
    order = np.argsort(np.asarray(keys, dtype=np.float64))[::-1].astype(np.int64).tolist()
    return [int(items[int(i)]) for i in order]


def _load_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        return dict(default)
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(blob, dict):
            return blob
    except Exception:
        pass
    return dict(default)


def _store_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _wheel_signature(config_blob: Dict[str, Any]) -> str:
    return str(hashlib.sha256(json.dumps(config_blob, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest())


def _chunk_payload(entries: Sequence[Dict[str, Any]], label_dim: int, image_size: int) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    n = int(len(entries))
    if int(n) <= 0:
        raise RuntimeError("semantic wheel chunk payload requires at least one entry")
    size = max(8, int(image_size))
    images = np.stack([np.asarray(entry["image_u8"], dtype=np.uint8) for entry in entries], axis=0).astype(np.uint8, copy=False)
    mixed_masks = np.stack([np.asarray(entry["mixed_mask"], dtype=np.float32) for entry in entries], axis=0).astype(np.float16, copy=False)
    label_bank_rows: List[np.ndarray] = []
    label_bank_lut: Dict[bytes, int] = {}
    label_refs: List[int] = []
    terms_bank_rows: List[str] = []
    terms_bank_lut: Dict[str, int] = {}
    terms_refs: List[int] = []
    mask_bank_rows: List[np.ndarray] = []
    mask_bank_lut: Dict[bytes, int] = {}
    assoc_offsets: List[int] = [0]
    assoc_mask_ids: List[int] = []
    assoc_label_indices: List[int] = []
    for entry in entries:
        terms_json = json.dumps(
            list(normalize_vocab_terms([str(x) for x in list(entry.get("terms") or [])])),
            ensure_ascii=True,
            separators=(",", ":"),
        )
        term_id = terms_bank_lut.get(str(terms_json), -1)
        if int(term_id) < 0:
            term_id = int(len(terms_bank_rows))
            terms_bank_lut[str(terms_json)] = int(term_id)
            terms_bank_rows.append(str(terms_json))
        terms_refs.append(int(term_id))

        label_vec = np.asarray(entry["label_vec_u8"], dtype=np.uint8).reshape(-1)
        if int(label_vec.size) != int(label_dim):
            raise RuntimeError(f"semantic wheel label width mismatch: got={int(label_vec.size)} expected={int(label_dim)}")
        label_key = bytes(label_vec.tobytes())
        label_id = label_bank_lut.get(label_key, -1)
        if int(label_id) < 0:
            label_id = int(len(label_bank_rows))
            label_bank_lut[label_key] = int(label_id)
            label_bank_rows.append(np.asarray(label_vec, dtype=np.uint8))
        label_refs.append(int(label_id))

        stack_f = np.asarray(entry["mask_stack"], dtype=np.float32)
        stack_idx = np.asarray(entry["mask_indices"], dtype=np.int32).reshape(-1)
        if int(stack_f.ndim) == 2:
            stack_f = stack_f[None, ...]
        if int(stack_f.ndim) != 3:
            stack_f = np.zeros((0, size, size), dtype=np.float32)
        pair_count = min(int(stack_f.shape[0]), int(stack_idx.size))
        for pos in range(int(pair_count)):
            mask_f16 = np.asarray(stack_f[int(pos)], dtype=np.float16)
            mask_key = bytes(mask_f16.tobytes())
            mask_id = mask_bank_lut.get(mask_key, -1)
            if int(mask_id) < 0:
                mask_id = int(len(mask_bank_rows))
                mask_bank_lut[mask_key] = int(mask_id)
                mask_bank_rows.append(mask_f16)
            assoc_mask_ids.append(int(mask_id))
            assoc_label_indices.append(int(stack_idx[int(pos)]))
        assoc_offsets.append(int(len(assoc_mask_ids)))

    label_bank = np.stack(label_bank_rows, axis=0).astype(np.uint8, copy=False) if len(label_bank_rows) > 0 else np.zeros((0, int(label_dim)), dtype=np.uint8)
    max_terms_len = max((len(str(x)) for x in terms_bank_rows), default=1)
    terms_bank = np.asarray(terms_bank_rows, dtype=f"<U{int(max_terms_len)}") if len(terms_bank_rows) > 0 else np.asarray([], dtype="<U1")
    mask_bank = np.stack(mask_bank_rows, axis=0).astype(np.float16, copy=False) if len(mask_bank_rows) > 0 else np.zeros((0, size, size), dtype=np.float16)
    payload = {
        "images": images,
        "mixed_masks": mixed_masks,
        "label_bank": label_bank,
        "label_refs": np.asarray(label_refs, dtype=np.int32),
        "terms_bank": terms_bank,
        "terms_refs": np.asarray(terms_refs, dtype=np.int32),
        "mask_bank": mask_bank,
        "assoc_offsets": np.asarray(assoc_offsets, dtype=np.int64),
        "assoc_mask_ids": np.asarray(assoc_mask_ids, dtype=np.int32),
        "assoc_label_indices": np.asarray(assoc_label_indices, dtype=np.int32),
    }
    raw_bytes = int(sum(int(arr.nbytes) for arr in payload.values()))
    info = {
        "rows": int(n),
        "unique_labels": int(label_bank.shape[0]),
        "unique_masks": int(mask_bank.shape[0]),
        "associations": int(len(assoc_mask_ids)),
        "raw_bytes": int(raw_bytes),
    }
    return payload, info


def _write_chunk(path: Path, payload: Dict[str, np.ndarray]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **payload)
    return int(path.stat().st_size) if path.exists() else 0


@dataclass
class SemanticWheelConfig:
    purpose: str
    cache_root: str
    image_size: int
    batch_size: int
    lookahead_batches: int
    seed: int
    deformations_per_clean: int = 0
    include_clean: bool = True
    explicit_max_bytes: int = 0
    sanity_cap_bytes: int = 8 * 1024 * 1024 * 1024
    allow_large_override: bool = False
    expiry_uses: int = 0
    max_base_rows: int = 0
    use_rare_term_deck: bool = True
    degrade_config: Optional[Dict[str, Any]] = None
    force_rebuild: bool = False
    processing_device: Optional[Any] = None
    preload_workers: int = 0
    progress_control: Optional[Any] = None


class SemanticWheelDataset(Dataset):
    def __init__(self, cache_dir: str, return_mask_stack: bool = False):
        self.cache_dir = Path(str(cache_dir))
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(f"semantic wheel manifest is missing: {manifest_path}")
        self.manifest = _load_json(manifest_path, default={})
        self.return_mask_stack = bool(return_mask_stack)
        self.use_semantic_mask_stack_collate = bool(self.return_mask_stack)
        self.image_size = int(self.manifest.get("image_size", 0))
        self.label_dim = int(self.manifest.get("label_dim", 0))
        self.chunk_rows = [int(x) for x in list(self.manifest.get("chunk_rows", []))]
        self.chunk_offsets: List[int] = [0]
        for count in self.chunk_rows:
            self.chunk_offsets.append(int(self.chunk_offsets[-1] + int(count)))
        self.total_rows = int(self.chunk_offsets[-1])
        self.lookahead_batches = int(self.manifest.get("lookahead_batches", 0))
        self._chunk_cache: Dict[int, Dict[str, np.ndarray]] = {}
        self._chunk_lru: List[int] = []
        self._max_cached_chunks = max(2, int(self.lookahead_batches) + 1)

    def __len__(self) -> int:
        return int(self.total_rows)

    def _chunk_path(self, chunk_idx: int) -> Path:
        return self.cache_dir / f"chunk_{int(chunk_idx):04d}.npz"

    def _load_chunk(self, chunk_idx: int) -> Dict[str, np.ndarray]:
        key = int(chunk_idx)
        cached = self._chunk_cache.get(int(key))
        if isinstance(cached, dict):
            if int(key) in self._chunk_lru:
                self._chunk_lru.remove(int(key))
            self._chunk_lru.append(int(key))
            return cached
        path = self._chunk_path(int(key))
        if not path.exists():
            raise RuntimeError(f"semantic wheel chunk is missing: {path}")
        with np.load(str(path), allow_pickle=False) as z:
            payload = {str(name): np.asarray(z[str(name)]) for name in z.files}
        self._chunk_cache[int(key)] = payload
        self._chunk_lru.append(int(key))
        while int(len(self._chunk_lru)) > int(self._max_cached_chunks):
            victim = int(self._chunk_lru.pop(0))
            self._chunk_cache.pop(int(victim), None)
        return payload

    def _locate(self, index: int) -> Tuple[int, int]:
        idx = int(index)
        if idx < 0:
            idx = int(self.total_rows) + idx
        if idx < 0 or idx >= int(self.total_rows):
            raise IndexError(idx)
        for chunk_idx in range(int(len(self.chunk_rows))):
            start = int(self.chunk_offsets[int(chunk_idx)])
            stop = int(self.chunk_offsets[int(chunk_idx) + 1])
            if int(start) <= int(idx) < int(stop):
                return int(chunk_idx), int(idx - start)
        raise IndexError(idx)

    def read_numpy_entry(self, index: int) -> Dict[str, np.ndarray]:
        chunk_idx, row_offset = self._locate(int(index))
        payload = self._load_chunk(int(chunk_idx))
        images = np.asarray(payload["images"], dtype=np.uint8)
        mixed_masks = np.asarray(payload["mixed_masks"], dtype=np.float32)
        label_bank = np.asarray(payload["label_bank"], dtype=np.uint8)
        label_refs = np.asarray(payload["label_refs"], dtype=np.int32).reshape(-1)
        terms_bank = np.asarray(payload.get("terms_bank", np.asarray([], dtype="<U1")))
        terms_refs = np.asarray(payload.get("terms_refs", np.zeros((0,), dtype=np.int32)), dtype=np.int32).reshape(-1)
        mask_bank = np.asarray(payload["mask_bank"], dtype=np.float32)
        assoc_offsets = np.asarray(payload["assoc_offsets"], dtype=np.int64).reshape(-1)
        assoc_mask_ids = np.asarray(payload["assoc_mask_ids"], dtype=np.int32).reshape(-1)
        assoc_label_indices = np.asarray(payload["assoc_label_indices"], dtype=np.int32).reshape(-1)
        label_ref = int(label_refs[int(row_offset)]) if 0 <= int(row_offset) < int(label_refs.size) else -1
        label_vec = (
            np.asarray(label_bank[int(label_ref)], dtype=np.uint8)
            if 0 <= int(label_ref) < int(label_bank.shape[0])
            else np.zeros((int(self.label_dim),), dtype=np.uint8)
        )
        term_ref = int(terms_refs[int(row_offset)]) if 0 <= int(row_offset) < int(terms_refs.size) else -1
        if 0 <= int(term_ref) < int(terms_bank.shape[0]):
            try:
                terms = list(normalize_vocab_terms(json.loads(str(terms_bank[int(term_ref)]))))
            except Exception:
                terms = []
        else:
            terms = []
        a0 = int(assoc_offsets[int(row_offset)]) if int(row_offset) < int(assoc_offsets.size) else 0
        a1 = int(assoc_offsets[int(row_offset) + 1]) if int(row_offset + 1) < int(assoc_offsets.size) else int(a0)
        mask_ids = np.asarray(assoc_mask_ids[int(a0): int(a1)], dtype=np.int32)
        label_indices = np.asarray(assoc_label_indices[int(a0): int(a1)], dtype=np.int32)
        stack_f32 = (
            np.asarray(mask_bank[mask_ids.tolist()], dtype=np.float32)
            if int(mask_ids.size) > 0 and int(mask_bank.shape[0]) > 0
            else np.zeros((0, int(self.image_size), int(self.image_size)), dtype=np.float32)
        )
        return {
            "image_u8": np.asarray(images[int(row_offset)], dtype=np.uint8),
            "label_vec_u8": np.asarray(label_vec, dtype=np.uint8),
            "mixed_mask": np.asarray(mixed_masks[int(row_offset)], dtype=np.float32),
            "mask_stack": np.asarray(stack_f32, dtype=np.float32),
            "mask_indices": np.asarray(label_indices, dtype=np.int32),
            "terms": list(terms),
        }

    def __getitem__(self, index: int):
        item = self.read_numpy_entry(int(index))
        x_t = torch.from_numpy(np.asarray(item["image_u8"], dtype=np.float32) / 255.0)
        y_t = torch.from_numpy(np.asarray(item["label_vec_u8"], dtype=np.float32))
        mask_t = torch.from_numpy(np.asarray(item["mixed_mask"], dtype=np.float32)).unsqueeze(0)
        if bool(self.return_mask_stack):
            stack_t = torch.from_numpy(np.asarray(item["mask_stack"], dtype=np.float32))
            idx_t = torch.from_numpy(np.asarray(item["mask_indices"], dtype=np.int64))
            return x_t, y_t, mask_t, stack_t, idx_t, list(item.get("terms") or [])
        return x_t, y_t, mask_t


class SemanticWheelPayloadBank:
    def __init__(self, dataset: SemanticWheelDataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return int(len(self.dataset))

    def get_image(self, index: int) -> np.ndarray:
        item = self.dataset.read_numpy_entry(int(index))
        return (np.asarray(item["image_u8"], dtype=np.float32) / 255.0).astype(np.float32, copy=False)

    def get_mask(self, index: int) -> np.ndarray:
        item = self.dataset.read_numpy_entry(int(index))
        return np.asarray(item["mixed_mask"], dtype=np.float32)


class SemanticWheelPayloadView(Sequence[np.ndarray]):
    def __init__(self, bank: SemanticWheelPayloadBank, kind: str):
        self.bank = bank
        self.kind = str(kind).strip().lower()

    def __len__(self) -> int:
        return int(len(self.bank))

    def __getitem__(self, index: int) -> np.ndarray:
        if str(self.kind) == "image":
            return self.bank.get_image(int(index))
        if str(self.kind) == "mask":
            return self.bank.get_mask(int(index))
        raise KeyError(self.kind)


class StatefulSequentialDeckSampler(Sampler[int]):
    def __init__(self, length: int):
        self.length = max(0, int(length))
        self.cursor = 0
        self.items_yielded: int = 0

    def __iter__(self):
        start = int(self.cursor)
        emitted = 0
        while int(emitted) < int(self.length):
            idx = int((int(start) + int(emitted)) % max(1, int(self.length)))
            emitted += 1
            self.items_yielded += 1
            self.cursor = int((idx + 1) % max(1, int(self.length)))
            yield int(idx)

    def __len__(self) -> int:
        return int(self.length)

    @property
    def all_items_seen(self) -> bool:
        """True when every item has been yielded at least once since last reset."""
        return int(self.items_yielded) >= int(self.length) and int(self.length) > 0

    def reset_items_yielded(self) -> None:
        self.items_yielded = 0


def ensure_semantic_candidate_cache(
    *,
    candidates: Sequence[SemanticWheelCandidate],
    candidate_indices: Sequence[int],
    build_entry_group: Callable[[int, int], Sequence[Dict[str, Any]]],
    build_entry_groups_batch: Optional[Callable[[Sequence[Tuple[int, int]]], Sequence[Sequence[Dict[str, Any]]]]] = None,
    label_dim: int,
    config: SemanticWheelConfig,
) -> Dict[str, Any]:
    if len(candidates) <= 0 or len(candidate_indices) <= 0:
        raise RuntimeError(f"{str(config.purpose)} requires non-empty semantic candidates")
    cache_root = Path(str(config.cache_root).strip())
    cache_root.mkdir(parents=True, exist_ok=True)
    purpose_key = _sanitize_component(config.purpose)
    build_config = {
        "format_version": 2,
        "purpose": str(config.purpose),
        "image_size": int(config.image_size),
        "batch_size": int(config.batch_size),
        "lookahead_batches": int(config.lookahead_batches),
        "deformations_per_clean": int(config.deformations_per_clean),
        "include_clean": bool(config.include_clean),
        "label_dim": int(label_dim),
        "explicit_max_bytes": int(config.explicit_max_bytes),
        "sanity_cap_bytes": int(config.sanity_cap_bytes),
        "allow_large_override": bool(config.allow_large_override),
        "max_base_rows": int(config.max_base_rows),
        "seed": int(config.seed),
        "use_rare_term_deck": bool(config.use_rare_term_deck),
    }
    signature = _wheel_signature(build_config)
    wheel_dir = cache_root / f"{purpose_key}_{str(signature)[:24]}"
    manifest_path = wheel_dir / "manifest.json"
    deck_state_path = wheel_dir / "deck_state.json"
    candidate_sig = _candidate_signature(candidates=candidates, candidate_indices=candidate_indices)
    manifest = _load_json(manifest_path, default={})
    manifest_ok = bool(
        not bool(config.force_rebuild)
        and
        manifest
        and str(manifest.get("signature", "")) == str(signature)
        and str(manifest.get("candidate_signature", "")) == str(candidate_sig)
        and int(manifest.get("label_dim", 0)) == int(label_dim)
        and int(manifest.get("image_size", 0)) == int(config.image_size)
    )
    if bool(manifest_ok):
        manifest["cache_hit"] = True
        manifest["lookahead_batches"] = int(config.lookahead_batches)
        _store_json(manifest_path, manifest)
        return {
            "cache_dir": str(wheel_dir),
            "manifest": str(manifest_path),
            "cache_hit": True,
            "dataset": SemanticWheelDataset(cache_dir=str(wheel_dir), return_mask_stack=True),
            "base_row_indices": [int(x) for x in list(manifest.get("base_row_indices", []))],
            "base_candidate_indices": [
                int(x)
                for x in list(manifest.get("base_candidate_indices", manifest.get("base_row_indices", [])))
            ],
            "info": dict(manifest),
        }

    wheel_dir.mkdir(parents=True, exist_ok=True)
    deck_state = _load_json(deck_state_path, default={})
    state_sig = str(deck_state.get("candidate_signature", ""))
    order = [int(x) for x in list(deck_state.get("order", []))] if state_sig == str(candidate_sig) else []
    epoch = int(deck_state.get("epoch", 0)) if state_sig == str(candidate_sig) else 0
    cursor = int(deck_state.get("cursor", 0)) if state_sig == str(candidate_sig) else 0
    candidate_list = [int(x) for x in list(candidate_indices)]
    if len(order) != int(len(candidate_list)) or sorted(order) != sorted(candidate_list):
        if bool(config.use_rare_term_deck):
            order = _weighted_deck_order(candidates=candidates, candidate_indices=candidate_list, seed=int(config.seed), epoch=int(epoch))
        else:
            rng = np.random.default_rng(max(0, int(config.seed)) + (int(epoch) * 104729))
            order = [int(x) for x in rng.permutation(np.asarray(candidate_list, dtype=np.int64)).tolist()]
        cursor = 0
    ordered_candidates = order[int(cursor):] + order[: int(cursor)]
    if int(config.max_base_rows) > 0:
        ordered_candidates = ordered_candidates[: int(config.max_base_rows)]
    if len(ordered_candidates) <= 0:
        raise RuntimeError(f"{str(config.purpose)} deck selection is empty")

    entries_per_clean = int(config.deformations_per_clean) + (1 if bool(config.include_clean) else 0)
    if int(entries_per_clean) <= 0:
        raise RuntimeError("semantic wheel requires at least one entry per clean row")

    def _build_group(spec: Tuple[int, int]) -> Tuple[int, int, List[Dict[str, Any]]]:
        base_row_idx, base_pos = spec
        group = [dict(x) for x in list(build_entry_group(int(base_row_idx), int(base_pos)))]
        return int(base_pos), int(base_row_idx), group

    def _group_specs(base_row_indices: Sequence[int]) -> List[Tuple[int, int]]:
        return [(int(base_row_idx), int(base_pos)) for base_pos, base_row_idx in enumerate(base_row_indices)]

    def _group_spec_batches(base_row_indices: Sequence[int]) -> List[List[Tuple[int, int]]]:
        specs = _group_specs(base_row_indices)
        chunk_size = max(1, int(config.batch_size))
        return [specs[i: i + int(chunk_size)] for i in range(0, int(len(specs)), int(chunk_size))]

    def _build_group_batch(spec_batch: Sequence[Tuple[int, int]]) -> List[Tuple[int, int, List[Dict[str, Any]]]]:
        if build_entry_groups_batch is None:
            return [_build_group(spec) for spec in spec_batch]
        groups = list(build_entry_groups_batch(spec_batch))
        if int(len(groups)) != int(len(spec_batch)):
            raise RuntimeError(
                f"{str(config.purpose)} batch entry builder returned {int(len(groups))} groups for {int(len(spec_batch))} specs"
            )
        out: List[Tuple[int, int, List[Dict[str, Any]]]] = []
        for spec, group in zip(spec_batch, groups):
            base_row_idx, base_pos = spec
            out.append((int(base_pos), int(base_row_idx), [dict(x) for x in list(group)]))
        return out

    # --- Config-only cap validation (no data work) ---
    effective_limit = int(config.explicit_max_bytes)
    if int(effective_limit) > int(config.sanity_cap_bytes) and not bool(config.allow_large_override):
        raise RuntimeError(
            f"{str(config.purpose)} requested wheel cap {float(effective_limit) / (1024.0 * 1024.0):.1f} MB "
            f"exceeds sanity cap {float(config.sanity_cap_bytes) / (1024.0 * 1024.0):.1f} MB. "
            "Enable the large-cache override to allow caps above the sanity guard."
        )
    if int(effective_limit) <= 0:
        if bool(config.allow_large_override):
            effective_limit = 0
        else:
            effective_limit = int(config.sanity_cap_bytes)

    # --- Single streaming pass: build → queue → write ---
    temp_dir = wheel_dir.with_name(f"{wheel_dir.name}_tmp")
    if temp_dir.exists():
        shutil.rmtree(str(temp_dir), ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    _SENTINEL = None
    # Queue depth of 2: lets one chunk serialize while the next builds
    write_queue: queue.Queue = queue.Queue(maxsize=2)

    # Writer thread state — collected as the single source of truth
    writer_state = {
        "chunk_idx": 0,
        "chunk_rows": [],
        "chunk_bytes": [],
        "total_raw_bytes": 0,
        "total_unique_labels": 0,
        "total_unique_masks": 0,
        "total_associations": 0,
        "error": None,
    }

    def _writer_loop() -> None:
        try:
            while True:
                item = write_queue.get()
                if item is _SENTINEL:
                    return
                entries_to_flush, = item
                if len(entries_to_flush) <= 0:
                    continue
                payload, chunk_info = _chunk_payload(
                    entries_to_flush,
                    label_dim=int(label_dim),
                    image_size=int(config.image_size),
                )
                on_disk = _write_chunk(
                    temp_dir / f"chunk_{int(writer_state['chunk_idx']):04d}.npz",
                    payload,
                )
                writer_state["chunk_rows"].append(int(len(entries_to_flush)))
                writer_state["chunk_bytes"].append(int(on_disk))
                writer_state["total_raw_bytes"] += int(chunk_info.get("raw_bytes", 0))
                writer_state["total_unique_labels"] += int(chunk_info.get("unique_labels", 0))
                writer_state["total_unique_masks"] += int(chunk_info.get("unique_masks", 0))
                writer_state["total_associations"] += int(chunk_info.get("associations", 0))
                writer_state["chunk_idx"] += 1
        except Exception as exc:
            writer_state["error"] = exc

    writer_thread = threading.Thread(target=_writer_loop, name="semantic-wheel-writer", daemon=True)
    writer_thread.start()

    selected_base_rows: List[int] = []
    current_entries: List[Dict[str, Any]] = []
    cap_exceeded = False

    spec_batches = _group_spec_batches(ordered_candidates)

    def _estimate_chunk_bytes(entries: Sequence[Dict[str, Any]]) -> int:
        return int(sum(
            int(np.asarray(e.get("image_u8", np.zeros(0)), dtype=np.uint8).nbytes)
            + int(np.asarray(e.get("mixed_mask", np.zeros(0)), dtype=np.float32).nbytes)
            + int(np.asarray(e.get("label_vec_u8", np.zeros(0)), dtype=np.uint8).nbytes)
            + int(np.asarray(e.get("mask_stack", np.zeros(0)), dtype=np.float32).nbytes)
            for e in entries
        ))

    _build_desc = f"[{config.purpose}] building & writing"
    for spec_batch in interruptible_tqdm(
        spec_batches,
        total=len(spec_batches),
        desc=_build_desc,
        unit="batch",
        leave=False,
        dynamic_ncols=True,
        control=config.progress_control,
    ):
        if cap_exceeded:
            break
        if writer_state["error"] is not None:
            raise writer_state["error"]
        group_batch = _build_group_batch(spec_batch)
        for _base_pos, base_row_idx, group in group_batch:
            if cap_exceeded:
                break
            current_entries.extend(group)
            while int(len(current_entries)) >= int(config.batch_size):
                chunk_to_write = current_entries[: int(config.batch_size)]
                current_entries = current_entries[int(config.batch_size):]
                if int(effective_limit) > 0:
                    est_bytes = _estimate_chunk_bytes(chunk_to_write)
                    if int(writer_state["total_raw_bytes"]) + int(est_bytes) > int(effective_limit):
                        cap_exceeded = True
                        break
                write_queue.put((chunk_to_write,))
            if not cap_exceeded:
                selected_base_rows.append(int(base_row_idx))

    # Flush remaining entries
    if len(current_entries) > 0 and not cap_exceeded:
        if int(effective_limit) > 0:
            est_bytes = _estimate_chunk_bytes(current_entries)
            if int(writer_state["total_raw_bytes"]) + int(est_bytes) > int(effective_limit):
                cap_exceeded = True
        if not cap_exceeded:
            write_queue.put((list(current_entries),))

    # Signal writer to finish and wait
    write_queue.put(_SENTINEL)
    writer_thread.join()

    if writer_state["error"] is not None:
        shutil.rmtree(str(temp_dir), ignore_errors=True)
        raise writer_state["error"]

    if cap_exceeded and len(selected_base_rows) > 0:
        actual_mb = float(writer_state["total_raw_bytes"]) / (1024.0 * 1024.0)
        cap_mb = float(effective_limit) / (1024.0 * 1024.0)
        tqdm.write(
            f"[{config.purpose}] wheel reached byte cap "
            f"({actual_mb:.1f} MB written, cap {cap_mb:.1f} MB). "
            f"Serving partial wheel with {int(len(selected_base_rows))} of "
            f"{int(len(ordered_candidates))} candidate rows."
        )

    if len(selected_base_rows) <= 0:
        shutil.rmtree(str(temp_dir), ignore_errors=True)
        raise RuntimeError(
            f"{str(config.purpose)} wheel produced zero rows at image_size={int(config.image_size)} "
            f"batch_size={int(config.batch_size)}"
        )

    chunk_rows = list(writer_state["chunk_rows"])
    chunk_bytes = list(writer_state["chunk_bytes"])
    total_raw_bytes = int(writer_state["total_raw_bytes"])

    final_manifest = {
        "version": 2,
        "signature": str(signature),
        "purpose": str(config.purpose),
        "candidate_signature": str(candidate_sig),
        "image_size": int(config.image_size),
        "label_dim": int(label_dim),
        "batch_size": int(config.batch_size),
        "lookahead_batches": int(config.lookahead_batches),
        "deformations_per_clean": int(config.deformations_per_clean),
        "include_clean": bool(config.include_clean),
        "entries_per_clean": int(entries_per_clean),
        "cache_hit": False,
        "explicit_max_bytes": int(config.explicit_max_bytes),
        "effective_max_bytes": int(effective_limit),
        "sanity_cap_bytes": int(config.sanity_cap_bytes),
        "allow_large_override": bool(config.allow_large_override),
        "total_raw_bytes": int(total_raw_bytes),
        "chunk_rows": [int(x) for x in chunk_rows],
        "chunk_bytes": [int(x) for x in chunk_bytes],
        "chunk_count": int(len(chunk_rows)),
        "total_rows": int(sum(chunk_rows)),
        "base_row_count": int(len(selected_base_rows)),
        "base_row_indices": [int(x) for x in selected_base_rows],
        "base_candidate_indices": [int(x) for x in selected_base_rows],
        "total_unique_labels": int(writer_state["total_unique_labels"]),
        "total_unique_masks": int(writer_state["total_unique_masks"]),
        "total_associations": int(writer_state["total_associations"]),
        "deck_epoch_start": int(epoch),
        "deck_cursor_start": int(cursor),
        "deck_use_rare_terms": bool(config.use_rare_term_deck),
    }
    _store_json(temp_dir / "manifest.json", final_manifest)
    if wheel_dir.exists():
        shutil.rmtree(str(wheel_dir), ignore_errors=True)
    temp_dir.replace(wheel_dir)

    new_cursor = int(cursor + len(selected_base_rows))
    new_epoch = int(epoch)
    new_order = list(order)
    if int(len(order)) > 0 and int(new_cursor) >= int(len(order)):
        new_epoch = int(epoch) + 1
        new_cursor = int(new_cursor % int(len(order)))
        if bool(config.use_rare_term_deck):
            new_order = _weighted_deck_order(candidates=candidates, candidate_indices=candidate_list, seed=int(config.seed), epoch=int(new_epoch))
        else:
            rng = np.random.default_rng(max(0, int(config.seed)) + (int(new_epoch) * 104729))
            new_order = [int(x) for x in rng.permutation(np.asarray(candidate_list, dtype=np.int64)).tolist()]
    deck_state_out = {
        "candidate_signature": str(candidate_sig),
        "epoch": int(new_epoch),
        "cursor": int(new_cursor),
        "order": [int(x) for x in new_order],
    }
    _store_json(deck_state_path, deck_state_out)

    return {
        "cache_dir": str(wheel_dir),
        "manifest": str(wheel_dir / "manifest.json"),
        "cache_hit": False,
        "dataset": SemanticWheelDataset(cache_dir=str(wheel_dir), return_mask_stack=True),
        "base_row_indices": [int(x) for x in selected_base_rows],
        "base_candidate_indices": [int(x) for x in selected_base_rows],
        "info": dict(final_manifest),
    }


def ensure_semantic_wheel_cache(
    rows: Sequence[SemanticDiskRow],
    candidate_indices: Sequence[int],
    class_names: Sequence[str],
    config: SemanticWheelConfig,
) -> Dict[str, Any]:
    if len(rows) <= 0 or len(candidate_indices) <= 0:
        raise RuntimeError(f"{str(config.purpose)} requires non-empty semantic rows")
    term_to_idx = {_norm_txt(str(name)): int(i) for i, name in enumerate(class_names) if str(name).strip()}
    idx_to_term = {int(i): str(name) for i, name in enumerate(class_names) if str(name).strip()}
    candidates: List[SemanticWheelCandidate] = []
    for row in interruptible_tqdm(
        rows,
        total=len(rows),
        desc=f"[{config.purpose}] hashing rows",
        unit="row",
        leave=False,
        dynamic_ncols=True,
        control=config.progress_control,
    ):
        payload = {
            "image_path": str(row.image_path),
            "mask_path": str(row.mask_path or ""),
            "terms": list(normalize_vocab_terms([str(x) for x in list(row.terms)])),
            "source": str(row.source),
            "labels": _positive_label_bits(np.asarray(row.label_vec, dtype=np.float32)).tolist(),
        }
        candidates.append(
            SemanticWheelCandidate(
                cache_key=json.dumps(payload, sort_keys=True, ensure_ascii=True),
                terms=list(normalize_vocab_terms([str(x) for x in list(row.terms)])),
                source=str(row.source),
            )
        )

    def _entry_group(base_row_idx: int, base_row_pos: int) -> List[Dict[str, Any]]:
        row = rows[int(base_row_idx)]
        clean = _build_clean_entry(
            row=row,
            image_size=int(config.image_size),
            idx_to_term=idx_to_term,
            processing_device=config.processing_device,
        )
        out: List[Dict[str, Any]] = []
        if bool(config.include_clean):
            out.append(dict(clean))
        for variant_idx in range(int(config.deformations_per_clean)):
            out.append(
                _build_deformed_entry(
                    clean_entry=clean,
                    variant_idx=int(variant_idx),
                    base_row_position=int(base_row_pos),
                    seed=int(config.seed),
                    term_to_idx=term_to_idx,
                    degrade_config=config.degrade_config,
                    processing_device=config.processing_device,
                )
            )
        return out

    def _entry_groups_batch(spec_batch: Sequence[Tuple[int, int]]) -> List[List[Dict[str, Any]]]:
        batch_rows = [rows[int(base_row_idx)] for base_row_idx, _base_row_pos in spec_batch]
        clean_entries = _build_clean_entries_batch(
            rows=batch_rows,
            image_size=int(config.image_size),
            idx_to_term=idx_to_term,
            processing_device=config.processing_device,
            preload_workers=int(config.preload_workers),
            progress_control=config.progress_control,
        )
        out_groups: List[List[Dict[str, Any]]] = []
        for clean, (base_row_idx, base_row_pos) in zip(clean_entries, spec_batch):
            out: List[Dict[str, Any]] = []
            if bool(config.include_clean):
                out.append(dict(clean))
            for variant_idx in range(int(config.deformations_per_clean)):
                out.append(
                    _build_deformed_entry(
                        clean_entry=clean,
                        variant_idx=int(variant_idx),
                        base_row_position=int(base_row_pos),
                        seed=int(config.seed),
                        term_to_idx=term_to_idx,
                        degrade_config=config.degrade_config,
                        processing_device=config.processing_device,
                    )
                )
            out_groups.append(out)
        return out_groups

    return ensure_semantic_candidate_cache(
        candidates=candidates,
        candidate_indices=candidate_indices,
        build_entry_group=_entry_group,
        build_entry_groups_batch=_entry_groups_batch,
        label_dim=int(len(class_names)),
        config=config,
    )
