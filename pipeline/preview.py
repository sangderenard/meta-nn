from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


def _to_np_float(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=torch.float32, device="cpu").numpy()
    return np.asarray(value, dtype=np.float32)


def _normalize_rgb_chw(image_like: Any) -> np.ndarray:
    arr = _to_np_float(image_like)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=0)
    elif arr.ndim == 3 and int(arr.shape[0]) == 1:
        arr = np.repeat(arr, 3, axis=0)
    elif arr.ndim == 3 and int(arr.shape[-1]) == 3 and int(arr.shape[0]) != 3:
        arr = arr.transpose(2, 0, 1)
    if arr.ndim != 3 or int(arr.shape[0]) < 3:
        raise ValueError(f"preview image must resolve to CHW with >=3 channels, got {tuple(arr.shape)}")
    return np.clip(np.asarray(arr[:3], dtype=np.float32), 0.0, 1.0)


def _normalize_mask_hw(mask_like: Any, height: int, width: int, fill: float) -> np.ndarray:
    if mask_like is None:
        return np.full((int(height), int(width)), float(fill), dtype=np.float32)
    arr = _to_np_float(mask_like)
    while arr.ndim > 2:
        arr = arr[0]
    if arr.ndim != 2:
        return np.full((int(height), int(width)), float(fill), dtype=np.float32)
    arr = np.clip(np.asarray(arr, dtype=np.float32), 0.0, 1.0)
    if tuple(arr.shape) == (int(height), int(width)):
        return arr
    src_h, src_w = int(arr.shape[0]), int(arr.shape[1])
    if src_h <= 0 or src_w <= 0:
        return np.full((int(height), int(width)), float(fill), dtype=np.float32)
    yi = np.floor(np.arange(int(height)) * src_h / max(1, int(height))).astype(np.int32).clip(0, src_h - 1)
    xi = np.floor(np.arange(int(width)) * src_w / max(1, int(width))).astype(np.int32).clip(0, src_w - 1)
    return np.ascontiguousarray(arr[yi[:, None], xi[None, :]], dtype=np.float32)


def _chw_to_u8_hwc(image_chw: np.ndarray) -> np.ndarray:
    arr = np.asarray(image_chw, dtype=np.float32)
    return np.ascontiguousarray(
        np.round(np.clip(arr.transpose(1, 2, 0), 0.0, 1.0) * 255.0).astype(np.uint8)
    )


def _format_target_line(target_vec: Any, class_names: Sequence[str], threshold: float = 0.5, max_items: int = 0) -> str:
    if target_vec is None:
        return "target:none"
    arr = _to_np_float(target_vec).reshape(-1)
    hits: List[str] = []
    for idx, value in enumerate(arr):
        if float(value) >= float(threshold):
            if 0 <= idx < len(class_names):
                hits.append(str(class_names[idx]))
            else:
                hits.append(f"class_{idx}")
            if int(max_items) > 0 and len(hits) >= int(max_items):
                break
    if not hits:
        return "target:none"
    return "target:" + ", ".join(hits)


def _format_target_lines(target_vec: Any, class_names: Sequence[str], threshold: float = 0.5) -> List[str]:
    """Return one line per active target label, for right-justified list display."""
    if target_vec is None:
        return ["target: none"]
    arr = _to_np_float(target_vec).reshape(-1)
    hits: List[str] = []
    for idx, value in enumerate(arr):
        if float(value) >= float(threshold):
            hits.append(str(class_names[idx]) if 0 <= idx < len(class_names) else f"class_{idx}")
    return hits if hits else ["none"]


def _format_top_lines(probs: Any, class_names: Sequence[str], topk: int = 0) -> List[str]:
    if probs is None:
        return []
    arr = _to_np_float(probs).reshape(-1)
    if arr.size <= 0:
        return []
    order = np.argsort(-arr) if int(topk) <= 0 else np.argsort(-arr)[: max(1, int(topk))]
    lines: List[str] = []
    for idx in order:
        label = str(class_names[int(idx)]) if 0 <= int(idx) < len(class_names) else f"class_{int(idx)}"
        lines.append(f"{label}:{float(arr[int(idx)]):.3f}")
    return lines


def _effective_batch_loss(payload_batch: Sequence[Dict[str, Any]]) -> Optional[float]:
    loss_value = float("nan")
    batch_loss_value = float("nan")
    for payload in payload_batch:
        if not math.isfinite(loss_value) and "loss" in payload:
            try:
                loss_value = float(payload.get("loss", float("nan")))
            except Exception:
                loss_value = float("nan")
        if "batch_loss" in payload:
            try:
                batch_loss_value = float(payload.get("batch_loss", float("nan")))
            except Exception:
                batch_loss_value = float("nan")
    if math.isfinite(batch_loss_value):
        return float(batch_loss_value)
    if math.isfinite(loss_value):
        return float(loss_value)
    return None


def build_classifier_preview_frames(
    payload_batch: Sequence[Dict[str, Any]],
    *,
    class_names: Sequence[str],
    cycle_id: int,
    round_id: int,
) -> Tuple[Optional[float], List[Dict[str, Any]]]:
    if not payload_batch:
        return None, []

    eff_loss = _effective_batch_loss(payload_batch)
    first = payload_batch[0]
    step_txt = (
        f"step={int(first.get('global_step', 0))}/{int(first.get('total_steps', 0))}"
        if ("global_step" in first or "total_steps" in first)
        else "step=0/0"
    )

    frames: List[Dict[str, Any]] = []
    for payload in payload_batch:
        image_like = payload.get("img", None)
        if image_like is None:
            continue
        try:
            image_chw = _normalize_rgb_chw(image_like)
        except Exception:
            continue
        height, width = int(image_chw.shape[1]), int(image_chw.shape[2])
        target_mask = _normalize_mask_hw(payload.get("target_mask", None), height=height, width=width, fill=1.0)
        detected_mask = _normalize_mask_hw(payload.get("detected_mask", None), height=height, width=width, fill=0.0)

        overlap = np.minimum(target_mask, detected_mask)
        detected_only = np.clip(detected_mask - target_mask, 0.0, 1.0)
        target_only = np.clip(target_mask - detected_mask, 0.0, 1.0)

        panel_target = np.concatenate([image_chw, target_mask[None, :, :]], axis=0)
        panel_diff = np.clip(
            (image_chw * 0.30) + (0.90 * np.stack([detected_only, overlap, target_only], axis=0)),
            0.0,
            1.0,
        )
        panel_detected = np.concatenate([image_chw, detected_mask[None, :, :]], axis=0)

        target_lines = _format_target_lines(payload.get("target_vec", None), class_names=class_names)
        top_lines = _format_top_lines(payload.get("probs", None), class_names=class_names)
        top_txt = top_lines[0] if top_lines else "n/a"

        loss_rows: List[str] = []
        loss_scalars: Dict[str, float] = {}
        for key, label in (("loss", "loss"), ("batch_loss", "batch")):
            if key not in payload:
                continue
            try:
                value = float(payload.get(key, float("nan")))
            except Exception:
                value = float("nan")
            if math.isfinite(value):
                loss_rows.append(f"{label}={value:.4f}")
                loss_scalars[key] = value

        diff_rows = [
            f"overlap={float(np.mean(overlap)):.3f}",
            f"target_only={float(np.mean(target_only)):.3f}",
            f"detected_only={float(np.mean(detected_only)):.3f}",
            f"mean_abs={float(np.mean(np.abs(target_mask - detected_mask))):.3f}",
        ]

        frames.append(
            {
                "images": [
                    _chw_to_u8_hwc(panel_target),
                    _chw_to_u8_hwc(panel_diff),
                    _chw_to_u8_hwc(panel_detected),
                ],
                "caption": f"[C] cycle={int(cycle_id)} round={int(round_id)} {step_txt} top={top_txt}",
                "titles": ["C target +mask", "C mask diff", "C detected +mask"],
                "step_txt": step_txt,
                "rows": [
                    target_lines,
                    diff_rows,
                    ["mask:detected"] + (top_lines or ["n/a"]),
                ],
                "loss_rows": loss_rows,
                "loss_scalars": loss_scalars,
            }
        )

    return eff_loss, frames


def make_classifier_step_preview_callback(ctx: Any, node_id: str):
    viewer = getattr(ctx, "viewer_proxy", None)
    enqueue_frame = getattr(viewer, "enqueue_frame", None) if viewer is not None else None
    publish_progress = getattr(ctx, "publish_node_progress", None)
    if not callable(enqueue_frame):
        return None

    def _callback(payload_batch: Sequence[Dict[str, Any]]) -> None:
        class_names = list(getattr(ctx, "class_names", []) or [])
        eff_loss, frames = build_classifier_preview_frames(
            payload_batch,
            class_names=class_names,
            cycle_id=int(getattr(ctx, "cycle", 0)),
            round_id=int(getattr(ctx, "round_id", 0)),
        )
        if callable(publish_progress) and eff_loss is not None and math.isfinite(float(eff_loss)):
            try:
                publish_progress(str(node_id), float(eff_loss))
            except Exception:
                pass
        for frame in frames:
            try:
                enqueue_frame(frame)
            except Exception:
                break

    return _callback
