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


# Lines that fit in the text panel at native resolution without any scaling.
# Derived from the known panel geometry: panel_h=256, header=20px, line=11px.
_PANEL_NATIVE_LINES: int = (256 - 20) // 11  # = 21


def _format_top_lines(
    probs: Any,
    class_names: Sequence[str],
    target_vec: Any = None,
    panel_lines: int = _PANEL_NATIVE_LINES,
) -> List[str]:
    """Return prediction lines for the score panel.

    All active targets (target_vec[i] >= 0.5) are listed first in
    score-descending order, prefixed with "!" so the viewer highlights them.
    Every target is shown regardless of its score — zeros and dead elements
    are the whole point.

    Non-target entries fill the remainder of *panel_lines* that targets did
    not use.  If targets alone meet or exceed *panel_lines*, no non-target
    entries are included and the renderer will scale the panel down to fit.
    Lines with a "!" prefix are rendered in amber by the viewer.
    """
    if probs is None:
        return []
    arr = _to_np_float(probs).reshape(-1)
    if arr.size <= 0:
        return []
    order = np.argsort(-arr)  # all indices, best score first

    target_set: set = set()
    if target_vec is not None:
        tvec = _to_np_float(target_vec).reshape(-1)
        for i, v in enumerate(tvec):
            if float(v) >= 0.5 and i < int(arr.size):
                target_set.add(int(i))

    lines: List[str] = []
    if target_set:
        # ALL targets, score-descending — zero-scoring entries appear at bottom
        for idx in order:
            if int(idx) in target_set:
                label = str(class_names[int(idx)]) if 0 <= int(idx) < len(class_names) else f"class_{int(idx)}"
                lines.append(f"!{label}:{float(arr[int(idx)]):.3f}")
        # Non-target budget = lines remaining before scaling would be needed
        non_target_budget = max(0, int(panel_lines) - len(lines))
        added = 0
        for idx in order:
            if added >= non_target_budget:
                break
            if int(idx) not in target_set:
                label = str(class_names[int(idx)]) if 0 <= int(idx) < len(class_names) else f"class_{int(idx)}"
                lines.append(f"{label}:{float(arr[int(idx)]):.3f}")
                added += 1
    else:
        for idx in order[:int(panel_lines)]:
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
    panel_lines: int = _PANEL_NATIVE_LINES,
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
    batch_size = len(payload_batch)
    for sample_idx, payload in enumerate(payload_batch):
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
        top_lines = _format_top_lines(
            payload.get("probs", None),
            class_names=class_names,
            target_vec=payload.get("target_vec", None),
            panel_lines=panel_lines,
        )
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
                "caption": f"[C] cycle={int(cycle_id)} round={int(round_id)} {step_txt} [{sample_idx+1}/{batch_size}] top={top_txt}",
                "titles": ["C target +mask", "C mask diff", "C detected +mask"],
                "step_txt": f"{step_txt} [{sample_idx+1}/{batch_size}]",
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


def _wave_batch_cpu(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        arr = value.detach().to(device="cpu", dtype=torch.float32)
    else:
        arr = torch.as_tensor(value, dtype=torch.float32)
    if arr.ndim == 1:
        arr = arr.unsqueeze(0)
    elif arr.ndim > 2:
        arr = arr.reshape(int(arr.shape[0]), -1)
    if arr.ndim != 2:
        raise ValueError(f"preview wave must resolve to [B, T], got {tuple(arr.shape)}")
    return arr


def build_transformer_preview_frames(
    payload_batch: Sequence[Dict[str, Any]],
    *,
    render_config: Any,
    image_hw: Tuple[int, int],
    sample_bits: int,
    class_names: Sequence[str],
    cycle_id: int,
    round_id: int,
) -> Tuple[Optional[float], List[Dict[str, Any]]]:
    if not payload_batch or render_config is None:
        return None, []

    from wav_ml_core import render_mono_wave_to_tensor

    eff_loss = _effective_batch_loss(payload_batch)
    first = payload_batch[0]
    step_txt = (
        f"step={int(first.get('step', 0))}/{int(first.get('steps_per_epoch', 0))}"
        if ("step" in first or "steps_per_epoch" in first)
        else "step=0/0"
    )

    valid_payloads: List[Dict[str, Any]] = []
    clean_rows: List[torch.Tensor] = []
    input_rows: List[torch.Tensor] = []
    output_rows: List[torch.Tensor] = []
    for payload in payload_batch:
        if any(payload.get(key, None) is None for key in ("x_clean", "x_in", "x_out")):
            continue
        try:
            clean_rows.append(_wave_batch_cpu(payload.get("x_clean")))
            input_rows.append(_wave_batch_cpu(payload.get("x_in")))
            output_rows.append(_wave_batch_cpu(payload.get("x_out")))
            valid_payloads.append(payload)
        except Exception:
            continue

    if not valid_payloads:
        return eff_loss, []

    with torch.no_grad():
        clean_images = render_mono_wave_to_tensor(
            torch.cat(clean_rows, dim=0),
            cfg=render_config,
            image_hw=image_hw,
            sample_bits=int(sample_bits),
        )
        input_images = render_mono_wave_to_tensor(
            torch.cat(input_rows, dim=0),
            cfg=render_config,
            image_hw=image_hw,
            sample_bits=int(sample_bits),
        )
        output_images = render_mono_wave_to_tensor(
            torch.cat(output_rows, dim=0),
            cfg=render_config,
            image_hw=image_hw,
            sample_bits=int(sample_bits),
        )

    frames: List[Dict[str, Any]] = []
    batch_size = len(valid_payloads)
    for sample_idx, payload in enumerate(valid_payloads):
        try:
            clean_chw = _normalize_rgb_chw(clean_images[sample_idx])
            input_chw = _normalize_rgb_chw(input_images[sample_idx])
            output_chw = _normalize_rgb_chw(output_images[sample_idx])
        except Exception:
            continue

        target_line = _format_target_line(payload.get("target_condition", None), class_names=class_names)
        metric_rows = [
            f"loss={float(payload.get('loss', float('nan'))):.4f}",
            f"score_target={float(payload.get('score_target', float('nan'))):.4f}",
            f"after={float(payload.get('score_after', float('nan'))):.4f}",
            f"gap={float(payload.get('score_gap', float('nan'))):.4f}",
        ]
        input_rows_txt = [
            f"degrade={float(payload.get('degrade_strength', 0.0)):.3f}",
            f"denoise={float(payload.get('denoise_l1', float('nan'))):.4f}",
            f"entropy={float(payload.get('entropy_excess', float('nan'))):.4f}",
            f"hi={float(payload.get('high_bits_l1', float('nan'))):.4f}",
            f"lo={float(payload.get('low_bits_l1', float('nan'))):.4f}",
        ]

        deskew_rows: List[str] = []
        deskew_preview = payload.get("deskew_preview", None)
        if isinstance(deskew_preview, dict):
            try:
                deskew_rows.append(
                    "deskew "
                    f"tgt={float(deskew_preview.get('target_skew', 0.0)):+.4f} "
                    f"pred={float(deskew_preview.get('pred_skew', 0.0)):+.4f} "
                    f"app={float(deskew_preview.get('applied_skew', 0.0)):+.4f} "
                    f"conf={float(deskew_preview.get('confidence', 0.0)):.3f}"
                )
                deskew_rows.append(
                    "deskew "
                    f"rem={float(deskew_preview.get('remaining_skew', 0.0)):+.4f} "
                    f"post={float(deskew_preview.get('post_residual_skew', 0.0)):+.4f}"
                )
            except Exception:
                deskew_rows = []
        if not deskew_rows:
            deskew_rows.append(
                "deskew "
                f"|pred|={float(payload.get('deskew_pred_abs', 0.0)):.4f} "
                f"|app|={float(payload.get('deskew_applied_abs', 0.0)):.4f} "
                f"conf={float(payload.get('deskew_confidence_mean', 0.0)):.3f}"
            )

        frame_loss = float(payload.get("loss", eff_loss if eff_loss is not None else float("nan")))
        loss_scalars: Dict[str, float] = {}
        if math.isfinite(frame_loss):
            loss_scalars["loss"] = frame_loss

        frames.append(
            {
                "images": [
                    _chw_to_u8_hwc(clean_chw),
                    _chw_to_u8_hwc(input_chw),
                    _chw_to_u8_hwc(output_chw),
                ],
                "caption": (
                    f"[R] cycle={int(cycle_id)} round={int(round_id)} {step_txt} "
                    f"[{sample_idx+1}/{batch_size}] loss={frame_loss:.4f}"
                ),
                "titles": ["R clean", "R input", "R output"],
                "step_txt": f"{step_txt} [{sample_idx+1}/{batch_size}]",
                "rows": [
                    [target_line, step_txt] + metric_rows[:2],
                    input_rows_txt,
                    metric_rows[2:] + deskew_rows,
                ],
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
        # Skip all preview work when preview is disabled at runtime.
        preview_check = getattr(ctx, "preview_enabled", None)
        if callable(preview_check) and not preview_check():
            return
        class_names = list(getattr(ctx, "class_names", []) or [])
        # Derive the native line capacity from the viewer's actual panel size
        # so the non-target budget tracks reality when panel dimensions change.
        _viewer = getattr(ctx, "viewer_proxy", None)
        _ph = int(getattr(_viewer, "panel_h", 256) or 256)
        _computed_panel_lines = max(1, (_ph - 20) // 11)
        eff_loss, frames = build_classifier_preview_frames(
            payload_batch,
            class_names=class_names,
            cycle_id=int(getattr(ctx, "cycle", 0)),
            round_id=int(getattr(ctx, "round_id", 0)),
            panel_lines=_computed_panel_lines,
        )
        for frame in frames:
            # Publish one loss entry per item so the graph ticks at the same
            # granularity as the scrub ring (one frame = one graph point).
            frame_loss = frame.get("loss_scalars", {}).get("batch_loss", None)
            if frame_loss is None:
                frame_loss = eff_loss
            if callable(publish_progress) and frame_loss is not None and math.isfinite(float(frame_loss)):
                try:
                    publish_progress(str(node_id), float(frame_loss))
                except Exception:
                    pass
            try:
                enqueue_frame(frame)
            except Exception:
                break

    return _callback


def make_transformer_step_preview_callback(ctx: Any, node_id: str):
    viewer = getattr(ctx, "viewer_proxy", None)
    enqueue_frame = getattr(viewer, "enqueue_frame", None) if viewer is not None else None
    publish_progress = getattr(ctx, "publish_node_progress", None)
    if not callable(enqueue_frame):
        return None

    def _callback(payload_batch: Sequence[Dict[str, Any]]) -> None:
        preview_check = getattr(ctx, "preview_enabled", None)
        if callable(preview_check) and not preview_check():
            return
        render_config = getattr(ctx, "render_config", None)
        args = getattr(ctx, "args", None)
        image_size = int(getattr(args, "image_size", 128) or 128)
        sample_bits = int(getattr(args, "sample_bits", 16) or 16)
        class_names = list(getattr(ctx, "class_names", []) or [])
        eff_loss, frames = build_transformer_preview_frames(
            payload_batch,
            render_config=render_config,
            image_hw=(image_size, image_size),
            sample_bits=sample_bits,
            class_names=class_names,
            cycle_id=int(getattr(ctx, "cycle", 0)),
            round_id=int(getattr(ctx, "round_id", 0)),
        )
        for frame in frames:
            frame_loss = frame.get("loss_scalars", {}).get("loss", None)
            if frame_loss is None:
                frame_loss = eff_loss
            if callable(publish_progress) and frame_loss is not None and math.isfinite(float(frame_loss)):
                try:
                    publish_progress(str(node_id), float(frame_loss))
                except Exception:
                    pass
            try:
                enqueue_frame(frame)
            except Exception:
                break

    return _callback
