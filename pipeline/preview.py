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
    arr = np.asarray(arr[:3], dtype=np.float32)
    arr_min = float(arr.min()) if int(arr.size) > 0 else 0.0
    arr_max = float(arr.max()) if int(arr.size) > 0 else 0.0
    if arr_min < -0.05 and arr_max <= 1.5:
        arr = (arr + 1.0) * 0.5
    elif arr_max > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


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


def _resize_image_hwc_u8(image_hwc: np.ndarray, height: int, width: int) -> np.ndarray:
    arr = np.asarray(image_hwc, dtype=np.uint8)
    if tuple(arr.shape[:2]) == (int(height), int(width)):
        return np.ascontiguousarray(arr)
    src_h, src_w = int(arr.shape[0]), int(arr.shape[1])
    yi = np.floor(np.arange(int(height)) * src_h / max(1, int(height))).astype(np.int32).clip(0, src_h - 1)
    xi = np.floor(np.arange(int(width)) * src_w / max(1, int(width))).astype(np.int32).clip(0, src_w - 1)
    return np.ascontiguousarray(arr[yi[:, None], xi[None, :], :], dtype=np.uint8)


def _normalize_preview_image_hwc_u8(
    image_like: Any,
    *,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> np.ndarray:
    arr = np.asarray(image_like)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3 and int(arr.shape[-1]) in (3, 4):
        arr = arr[:, :, :3]
    elif arr.ndim == 3 and int(arr.shape[0]) in (1, 3, 4):
        arr = _chw_to_u8_hwc(_normalize_rgb_chw(arr))
    else:
        raise ValueError(f"preview panel must resolve to HWC or CHW RGB, got {tuple(arr.shape)}")

    if arr.dtype != np.uint8:
        arr_f = np.asarray(arr, dtype=np.float32)
        arr_max = float(arr_f.max()) if int(arr_f.size) > 0 else 0.0
        if arr_max <= 1.5:
            arr = np.round(np.clip(arr_f, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            arr = np.round(np.clip(arr_f, 0.0, 255.0)).astype(np.uint8)

    arr = np.ascontiguousarray(arr[:, :, :3], dtype=np.uint8)
    if height is not None and width is not None:
        arr = _resize_image_hwc_u8(arr, int(height), int(width))
    return arr


def _tile_preview_images_hwc(images: Sequence[np.ndarray], height: int, width: int) -> np.ndarray:
    if not images:
        return np.zeros((int(height), int(width), 3), dtype=np.uint8)
    count = int(len(images))
    cols = max(1, math.ceil(math.sqrt(count)))
    rows = max(1, math.ceil(count / cols))
    cell_h = max(1, int(height) // rows)
    cell_w = max(1, int(width) // cols)
    canvas = np.zeros((int(height), int(width), 3), dtype=np.uint8)
    for idx, image in enumerate(images):
        r = idx // cols
        c = idx % cols
        y0 = r * cell_h
        x0 = c * cell_w
        y1 = min(int(height), y0 + cell_h)
        x1 = min(int(width), x0 + cell_w)
        tile = _normalize_preview_image_hwc_u8(image, height=(y1 - y0), width=(x1 - x0))
        canvas[y0:y1, x0:x1] = tile[: (y1 - y0), : (x1 - x0)]
    return canvas


def _normalize_preview_frame(frame_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce an arbitrary preview frame into the viewer's 3-panel schema."""

    out = dict(frame_dict)
    raw_images = list(out.get("images") or [])
    normalized_images: List[np.ndarray] = []
    if raw_images:
        base = _normalize_preview_image_hwc_u8(raw_images[0])
        panel_h, panel_w = int(base.shape[0]), int(base.shape[1])
        normalized_images.append(base)
        for image in raw_images[1:]:
            try:
                normalized_images.append(
                    _normalize_preview_image_hwc_u8(image, height=panel_h, width=panel_w)
                )
            except Exception:
                continue

        if len(normalized_images) == 1:
            normalized_images = [normalized_images[0], normalized_images[0], normalized_images[0]]
        elif len(normalized_images) == 2:
            normalized_images = [normalized_images[0], normalized_images[1], normalized_images[1]]
        elif len(normalized_images) > 3:
            normalized_images = [
                normalized_images[0],
                normalized_images[1],
                _tile_preview_images_hwc(normalized_images[2:], panel_h, panel_w),
            ]
        out["images"] = normalized_images[:3]

    titles = [str(x) for x in list(out.get("titles") or [])]
    rows = [[str(x) for x in list(section or [])] for section in list(out.get("rows") or [])[:3]]
    panel_count = max(3, len(out.get("images", [])))
    if len(titles) > 3:
        titles = titles[:2] + [f"Tiled x{max(1, len(raw_images) - 2)}"]
    while len(titles) < panel_count:
        titles.append(f"Panel {len(titles) + 1}")
    while len(rows) < panel_count:
        rows.append([])
    out["titles"] = titles[:3]
    out["rows"] = rows[:3]
    return out


def _dispatch_preview_frames(
    ctx: Any,
    node_id: str,
    frames: Sequence[Dict[str, Any]],
    *,
    eff_loss: Optional[float],
    preferred_loss_keys: Sequence[str],
) -> None:
    viewer = getattr(ctx, "viewer_proxy", None)
    enqueue_frame = getattr(viewer, "enqueue_frame", None) if viewer is not None else None
    publish_progress = getattr(ctx, "publish_node_progress", None)
    if not callable(enqueue_frame):
        return

    for frame in frames:
        frame = _normalize_preview_frame(frame)
        frame_loss = None
        loss_scalars = frame.get("loss_scalars", {})
        if isinstance(loss_scalars, dict):
            for key in preferred_loss_keys:
                if key in loss_scalars:
                    frame_loss = loss_scalars.get(key)
                    break
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
        _dispatch_preview_frames(
            ctx,
            str(node_id),
            frames,
            eff_loss=eff_loss,
            preferred_loss_keys=("batch_loss", "loss"),
        )

    return _callback


def make_transformer_step_preview_callback(ctx: Any, node_id: str):
    viewer = getattr(ctx, "viewer_proxy", None)
    enqueue_frame = getattr(viewer, "enqueue_frame", None) if viewer is not None else None
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
        _dispatch_preview_frames(
            ctx,
            str(node_id),
            frames,
            eff_loss=eff_loss,
            preferred_loss_keys=("loss",),
        )

    return _callback


def _normalize_generator_image_chw(image_like: Any) -> np.ndarray:
    """Convert a generator output tensor (CHW, possibly in [-1,1]) to a clipped [0,1] CHW array."""
    arr = _to_np_float(image_like)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and int(arr.shape[0]) != 3 and int(arr.shape[-1]) == 3:
        arr = arr.transpose(2, 0, 1)
    arr = arr[:3]
    # Remap [-1,1] → [0,1] if values are negative (tanh-activated output).
    if float(arr.min()) < -0.05:
        arr = (arr + 1.0) * 0.5
    return np.clip(np.asarray(arr, dtype=np.float32), 0.0, 1.0)


def build_generator_preview_frames(
    payload_batch: Sequence[Dict[str, Any]],
    *,
    class_names: Sequence[str],
    cycle_id: int,
    round_id: int,
) -> Tuple[Optional[float], List[Dict[str, Any]]]:
    """Build preview frames for Stage G (conditional GAN) training.

    Three-panel layout:
      0 – real target image (what the generator is conditioned on)
      1 – mask comparison: target mask (green) / fake mask (red) over dimmed target
      2 – generated fake image
    """
    if not payload_batch:
        return None, []

    first = payload_batch[0]
    step_txt = f"step={int(first.get('step', 0))}/{int(first.get('steps_per_epoch', 0))}"

    g_loss_avg = float(first.get("g_loss_avg", float("nan")))
    eff_loss: Optional[float] = float(g_loss_avg) if math.isfinite(g_loss_avg) else None

    frames: List[Dict[str, Any]] = []
    for payload in payload_batch:
        target_raw = payload.get("target_img", None)
        fake_raw = payload.get("fake_img", None)
        if target_raw is None or fake_raw is None:
            continue
        try:
            target_chw = _normalize_rgb_chw(target_raw)
            fake_chw = _normalize_generator_image_chw(fake_raw)
        except Exception:
            continue

        h, w = int(target_chw.shape[1]), int(target_chw.shape[2])
        target_mask = _normalize_mask_hw(payload.get("target_mask", None), height=h, width=w, fill=0.0)
        fake_mask = _normalize_mask_hw(payload.get("fake_mask", None), height=h, width=w, fill=0.0)

        # Panel 1: dimmed target with target mask (green) and fake mask (red) overlaid.
        dim = target_chw * 0.45
        mask_panel = np.clip(
            dim + np.stack([fake_mask * 0.55, target_mask * 0.55, np.zeros((h, w), dtype=np.float32)], axis=0),
            0.0, 1.0,
        )

        # Text rows
        target_labels = _format_target_lines(payload.get("target_condition", None), class_names=class_names)
        g_loss = float(payload.get("g_loss", float("nan")))
        d_loss = float(payload.get("d_loss", float("nan")))
        adv_loss = float(payload.get("adv_loss", float("nan")))
        target_prob = float(payload.get("target_prob_avg", float("nan")))

        metric_rows = []
        for k, v in [("g", g_loss), ("d", d_loss), ("adv", adv_loss), ("prob", target_prob)]:
            if math.isfinite(v):
                metric_rows.append(f"{k}={v:.4f}")

        loss_scalars: Dict[str, float] = {}
        if math.isfinite(g_loss):
            loss_scalars["loss"] = g_loss

        # Per-channel stats for the fake panel (fake_chw: [3,H,W] float32 in [0,1])
        ch_names = ("R", "G", "B")
        fake_rows = []
        for ci, cn in enumerate(ch_names):
            ch = fake_chw[ci]
            mu = float(ch.mean())
            sd = float(ch.std())
            lo = float(ch.min())
            hi = float(ch.max())
            fake_rows.append(f"{cn}: μ={mu:.2f} σ={sd:.2f} [{lo:.2f},{hi:.2f}]")

        frames.append({
            "images": [
                _chw_to_u8_hwc(target_chw),
                _chw_to_u8_hwc(mask_panel),
                _chw_to_u8_hwc(fake_chw),
            ],
            "caption": (
                f"[G] cycle={int(cycle_id)} round={int(round_id)} {step_txt}"
                + (f" g={g_loss:.4f}" if math.isfinite(g_loss) else "")
            ),
            "titles": ["G target", "G masks", "G fake"],
            "step_txt": step_txt,
            "rows": [
                target_labels,
                metric_rows,
                fake_rows,
            ],
            "loss_scalars": loss_scalars,
        })

    return eff_loss, frames


def make_generator_step_preview_callback(ctx: Any, node_id: str, preview_every: int = 10):
    """Return a step_preview_callback for Stage G GAN training.

    Throttled to one preview per *preview_every* calls so the viewer is not
    flooded (the GAN's step_preview_callback fires every generator step).
    """
    viewer = getattr(ctx, "viewer_proxy", None)
    enqueue_frame = getattr(viewer, "enqueue_frame", None) if viewer is not None else None
    if not callable(enqueue_frame):
        return None

    def _callback(payload_batch: Sequence[Dict[str, Any]]) -> None:
        preview_check = getattr(ctx, "preview_enabled", None)
        if callable(preview_check) and not preview_check():
            return
        class_names = list(getattr(ctx, "class_names", []) or [])
        eff_loss, frames = build_generator_preview_frames(
            payload_batch,
            class_names=class_names,
            cycle_id=int(getattr(ctx, "cycle", 0)),
            round_id=int(getattr(ctx, "round_id", 0)),
        )
        _dispatch_preview_frames(
            ctx,
            str(node_id),
            frames,
            eff_loss=eff_loss,
            preferred_loss_keys=("loss",),
        )

    return _callback


# ---------------------------------------------------------------------------
# Speculative-network preview  (NetworkContract / PrototypeAutoClassifier)
# ---------------------------------------------------------------------------


def _select_atlas_grid(slot_count: int, atlas_h: int, atlas_w: int) -> Tuple[int, int, int]:
    """Choose rows/cols that maximize square cell size inside the atlas."""

    best_rows = 1
    best_cols = max(1, int(slot_count))
    best_cell = 1
    best_used = 0
    for rows in range(1, max(1, int(slot_count)) + 1):
        cols = max(1, math.ceil(int(slot_count) / rows))
        cell = min(max(1, int(atlas_h) // rows), max(1, int(atlas_w) // cols))
        used = cell * cell * int(slot_count)
        if cell > best_cell or (cell == best_cell and used > best_used):
            best_rows = rows
            best_cols = cols
            best_cell = cell
            best_used = used
    return best_rows, best_cols, best_cell


def _build_slot_mask_atlas(
    slot_masks: np.ndarray,
    slot_confidence: Optional[np.ndarray],
    assignments: List,
    vocab_phrases: Sequence[str],
    *,
    atlas_h: int,
    atlas_w: int,
    confidence_floor: float = 0.0,
) -> np.ndarray:
    """Return a packed [3, atlas_h, atlas_w] mask atlas.

    Slots below *confidence_floor* are dropped.  Remaining slots are packed
    into square cells chosen to maximize visible mask size in the available
    atlas rectangle.
    """
    del vocab_phrases
    atlas_h = max(1, int(atlas_h))
    atlas_w = max(1, int(atlas_w))
    eligible: List[int] = []
    total_slots = int(slot_masks.shape[0])
    for i in range(total_slots):
        conf = float(slot_confidence[i]) if slot_confidence is not None and i < len(slot_confidence) else 0.0
        if conf >= float(confidence_floor):
            eligible.append(i)

    canvas = np.zeros((3, atlas_h, atlas_w), dtype=np.float32)
    if not eligible:
        return canvas

    rows, cols, cell = _select_atlas_grid(len(eligible), atlas_h=atlas_h, atlas_w=atlas_w)
    grid_h = rows * cell
    grid_w = cols * cell
    y_pad = max(0, (atlas_h - grid_h) // 2)
    x_pad = max(0, (atlas_w - grid_w) // 2)

    for packed_idx, slot_i in enumerate(eligible):
        row = packed_idx // cols
        col = packed_idx % cols
        y0 = y_pad + row * cell
        x0 = x_pad + col * cell
        y1 = min(atlas_h, y0 + cell)
        x1 = min(atlas_w, x0 + cell)

        mask = _normalize_mask_hw(slot_masks[slot_i], y1 - y0, x1 - x0, fill=0.0)
        conf = float(slot_confidence[slot_i]) if slot_confidence is not None and slot_i < len(slot_confidence) else 0.0
        assigned = assignments[slot_i] if slot_i < len(assignments) else None
        if assigned is not None:
            canvas[0, y0:y1, x0:x1] = np.maximum(canvas[0, y0:y1, x0:x1], mask * (0.10 + (1.0 - conf) * 0.20))
            canvas[1, y0:y1, x0:x1] = np.maximum(canvas[1, y0:y1, x0:x1], mask * (0.45 + conf * 0.55))
            canvas[2, y0:y1, x0:x1] = np.maximum(canvas[2, y0:y1, x0:x1], mask * 0.20)
        else:
            canvas[0, y0:y1, x0:x1] = np.maximum(canvas[0, y0:y1, x0:x1], mask * (0.35 + conf * 0.50))
            canvas[1, y0:y1, x0:x1] = np.maximum(canvas[1, y0:y1, x0:x1], mask * (0.06 + (1.0 - conf) * 0.18))
            canvas[2, y0:y1, x0:x1] = np.maximum(canvas[2, y0:y1, x0:x1], mask * 0.10)

    return np.clip(canvas, 0.0, 1.0)


def build_speculative_preview_frames(
    payload_batch: Sequence[Dict[str, Any]],
    *,
    cycle_id: int,
    round_id: int,
    output_panel_span: int = 2,
    confidence_floor: float = 0.0,
) -> Tuple[Optional[float], List[Dict[str, Any]]]:
    """Build preview frames for a NetworkContract step.

    Expected keys per payload item:
        img             [C, H, W]         — input image (RGB or RGBA)
        slot_masks      [N, H, W]         — sigmoid mask per slot
        slot_confidence [N]               — sigmoid confidence per slot, or None
        assignments     List[Optional[int]]
        vocab_phrases   List[str]
        target_labels   List[str]
        decision_rows   List[str]
        slot_rows       List[str]
        loss / batch_loss  float
        global_step / total_steps  int

    The network may claim one or two output panels.  With the default
    ``output_panel_span=2`` the input stays in panel 1 and the mask atlas is
    packed into a 2:1 rectangle spanning panels 2 and 3, then split into
    left/right square images for the viewer ring.
    """
    if not payload_batch:
        return None, []

    eff_loss = _effective_batch_loss(payload_batch)
    first = payload_batch[0]
    step_txt = (
        f"step={int(first.get('global_step', 0))}/{int(first.get('total_steps', 0))}"
        if ("global_step" in first or "total_steps" in first)
        else "step=?"
    )

    frames: List[Dict[str, Any]] = []

    for sample_idx, payload in enumerate(payload_batch):
        image_like = payload.get("img")
        if image_like is None:
            continue
        try:
            image_chw = _normalize_rgb_chw(image_like)
        except Exception:
            continue

        slot_masks_raw = payload.get("slot_masks")
        if slot_masks_raw is None:
            continue
        slot_masks_np = _to_np_float(slot_masks_raw)
        if slot_masks_np.ndim == 4:
            slot_masks_np = slot_masks_np[0]
        if slot_masks_np.ndim != 3:
            continue

        conf_raw = payload.get("slot_confidence")
        conf_np = _to_np_float(conf_raw).reshape(-1) if conf_raw is not None else None

        assignments: List = list(payload.get("assignments") or [])
        vocab_phrases: List[str] = list(payload.get("vocab_phrases") or [])

        H, W = int(image_chw.shape[1]), int(image_chw.shape[2])
        N = int(slot_masks_np.shape[0])
        resized = np.stack(
            [_normalize_mask_hw(slot_masks_np[i], H, W, fill=0.0) for i in range(N)],
            axis=0,
        )
        span = 1 if int(output_panel_span) <= 1 else 2
        atlas = _build_slot_mask_atlas(
            resized,
            slot_confidence=conf_np,
            assignments=assignments,
            vocab_phrases=vocab_phrases,
            atlas_h=H,
            atlas_w=W * span,
            confidence_floor=float(confidence_floor),
        )

        batch_loss = float(payload.get("batch_loss", payload.get("loss", float("nan"))))
        loss_scalars: Dict[str, float] = {}
        if math.isfinite(batch_loss):
            loss_scalars["batch_loss"] = batch_loss
        avg_loss = float(payload.get("loss", float("nan")))
        if math.isfinite(avg_loss):
            loss_scalars["loss"] = avg_loss
        raw_loss_scalars = payload.get("loss_scalars")
        if isinstance(raw_loss_scalars, dict):
            for key, value in raw_loss_scalars.items():
                try:
                    loss_scalars[str(key)] = float(value)
                except Exception:
                    continue

        target_rows = [str(x) for x in list(payload.get("target_labels") or []) if str(x).strip()]
        if not target_rows:
            target_rows = ["none"]

        decision_rows = [str(x) for x in list(payload.get("decision_rows") or []) if str(x).strip()]
        slot_rows = [str(x) for x in list(payload.get("slot_rows") or []) if str(x).strip()]
        if not slot_rows:
            for si in range(N):
                assigned = assignments[si] if si < len(assignments) else None
                phrase = (
                    vocab_phrases[assigned]
                    if (assigned is not None and assigned < len(vocab_phrases))
                    else "dustbin"
                )
                conf_str = (
                    f"{float(conf_np[si]):.2f}"
                    if conf_np is not None and si < len(conf_np)
                    else "?"
                )
                marker = "+" if assigned is not None else "-"
                slot_rows.append(f"{marker}{si}:{phrase}({conf_str})")

        active_count = 0
        if conf_np is not None:
            active_count = int(np.sum(conf_np >= float(confidence_floor)))
        else:
            active_count = int(N)
        decision_rows.append(f"tiles={active_count}/{N} conf>={float(confidence_floor):.2f}")

        images = [_chw_to_u8_hwc(image_chw)]
        titles = ["Input"]
        if span == 1:
            images.append(_chw_to_u8_hwc(atlas))
            titles.append(f"Mask Atlas x{active_count}")
        else:
            left = atlas[:, :, :W]
            right = atlas[:, :, W:]
            images.extend([_chw_to_u8_hwc(left), _chw_to_u8_hwc(right)])
            titles.extend([f"Mask Atlas L x{active_count}", "Mask Atlas R"])
        frames.append({
            "images": images,
            "caption": (
                f"[S] cycle={int(cycle_id)} round={int(round_id)} {step_txt} "
                f"[{sample_idx + 1}/{len(payload_batch)}]"
            ),
            "titles": titles,
            "step_txt": f"{step_txt} [{sample_idx + 1}/{len(payload_batch)}]",
            "rows": [
                target_rows,
                decision_rows,
                slot_rows,
            ],
            "loss_scalars": loss_scalars,
        })

    return eff_loss, frames


def make_speculative_step_preview_callback(ctx: Any, node_id: str) -> Optional[Any]:
    """Build a step callback that feeds speculative-network payloads to the viewer."""
    viewer = getattr(ctx, "viewer_proxy", None)
    enqueue_frame = getattr(viewer, "enqueue_frame", None) if viewer is not None else None
    if not callable(enqueue_frame):
        return None

    def _callback(payload_batch: Sequence[Dict[str, Any]]) -> None:
        preview_check = getattr(ctx, "preview_enabled", None)
        if callable(preview_check) and not preview_check():
            return
        prefs: Dict[str, Any] = {}
        network = getattr(ctx, "active_network", None)
        get_preview_preferences = getattr(network, "get_preview_preferences", None) if network is not None else None
        if callable(get_preview_preferences):
            try:
                raw_prefs = get_preview_preferences()
                if isinstance(raw_prefs, dict):
                    prefs = dict(raw_prefs)
            except Exception:
                prefs = {}
        eff_loss, frames = build_speculative_preview_frames(
            payload_batch,
            cycle_id=int(getattr(ctx, "cycle", 0)),
            round_id=int(getattr(ctx, "round_id", 0)),
            output_panel_span=int(
                prefs.get(
                    "output_panel_span",
                    getattr(getattr(ctx, "args", None), "network_preview_output_panels", 2),
                ) or 2
            ),
            confidence_floor=float(
                prefs.get(
                    "confidence_floor",
                    getattr(getattr(ctx, "args", None), "network_preview_confidence_floor", 0.0),
                ) or 0.0
            ),
        )
        _dispatch_preview_frames(
            ctx,
            str(node_id),
            frames,
            eff_loss=eff_loss,
            preferred_loss_keys=("batch_loss", "loss"),
        )

    return _callback
