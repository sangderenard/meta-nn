from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


_WEIGHT_MAP_BG = np.array([14, 16, 20], dtype=np.float32)
_WEIGHT_MAP_WARM = np.array([246, 127, 33], dtype=np.float32)
_WEIGHT_MAP_COOL = np.array([44, 140, 255], dtype=np.float32)
_WEIGHT_MAP_NEUTRAL = np.array([120, 196, 130], dtype=np.float32)
_WEIGHT_MAP_DRIFT = np.array([255, 52, 52], dtype=np.float32)
WEIGHT_IMAGE_MODES = ("parameter_groups", "architectural_tall", "architectural_wide")


def parameter_node_key(parameter_key: str) -> str:
    key = str(parameter_key or "").strip()
    if not key:
        return "<root>"
    parts = key.split(".")
    if len(parts) <= 1:
        return "<root>"
    return ".".join(parts[:-1]) or "<root>"


def snapshot_model_parameter_state(model: Any) -> Tuple[List[str], Dict[str, torch.Tensor]]:
    keys: List[str] = []
    state: Dict[str, torch.Tensor] = {}
    if model is None or not hasattr(model, "named_parameters"):
        return keys, state
    for name, param in model.named_parameters():
        keys.append(str(name))
        state[str(name)] = param.detach().to(device="cpu", dtype=torch.float32).clone()
    return keys, state


def snapshot_parameter_state_from_state_dict(
    state_dict: Dict[str, Any],
    parameter_keys: Optional[Sequence[str]] = None,
) -> Tuple[List[str], Dict[str, torch.Tensor]]:
    if not isinstance(state_dict, dict):
        return [], {}
    if parameter_keys is None:
        keys = [str(k) for k, v in state_dict.items() if torch.is_tensor(v) and torch.is_floating_point(v)]
    else:
        keys = [str(k) for k in parameter_keys]
    snap: Dict[str, torch.Tensor] = {}
    ordered: List[str] = []
    for key in keys:
        value = state_dict.get(key)
        if not torch.is_tensor(value) or not torch.is_floating_point(value):
            continue
        ordered.append(key)
        snap[key] = value.detach().to(device="cpu", dtype=torch.float32).clone()
    return ordered, snap


def summarize_parameter_nodes(
    state_dict: Dict[str, torch.Tensor],
    parameter_keys: Optional[Sequence[str]] = None,
    reference_state: Optional[Dict[str, torch.Tensor]] = None,
) -> List[Dict[str, Any]]:
    if parameter_keys is None:
        keys = sorted(str(k) for k in state_dict.keys())
    else:
        keys = [str(k) for k in parameter_keys]

    grouped: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for key in keys:
        tensor = state_dict.get(key)
        if tensor is None or not torch.is_tensor(tensor):
            continue
        if not torch.is_floating_point(tensor):
            continue
        flat = tensor.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        if flat.numel() <= 0:
            continue
        node_key = parameter_node_key(key)
        entry = grouped.get(node_key)
        if entry is None:
            entry = {
                "node": node_key,
                "count": 0,
                "sum": 0.0,
                "abs_sum": 0.0,
                "sq_sum": 0.0,
                "diff_abs_sum": 0.0,
                "keys": [],
            }
            grouped[node_key] = entry
            order.append(node_key)
        entry["count"] += int(flat.numel())
        entry["sum"] += float(flat.sum().item())
        entry["abs_sum"] += float(flat.abs().sum().item())
        entry["sq_sum"] += float((flat * flat).sum().item())
        entry["keys"].append(key)

        if isinstance(reference_state, dict):
            ref = reference_state.get(key)
            if torch.is_tensor(ref) and tuple(ref.shape) == tuple(tensor.shape):
                diff = (flat - ref.detach().to(device="cpu", dtype=torch.float32).reshape(-1)).abs()
                entry["diff_abs_sum"] += float(diff.sum().item())

    out: List[Dict[str, Any]] = []
    for node_key in order:
        entry = grouped[node_key]
        count = max(1, int(entry["count"]))
        out.append(
            {
                "node": str(node_key),
                "count": count,
                "mean": float(entry["sum"]) / float(count),
                "mean_abs": float(entry["abs_sum"]) / float(count),
                "rms": math.sqrt(max(0.0, float(entry["sq_sum"]) / float(count))),
                "diff_mean_abs": float(entry["diff_abs_sum"]) / float(count),
                "keys": list(entry["keys"]),
            }
        )
    return out


def _ordered_parameter_keys(
    state_dict: Dict[str, torch.Tensor],
    parameter_keys: Optional[Sequence[str]] = None,
) -> List[str]:
    if parameter_keys is None:
        return sorted(str(k) for k in state_dict.keys())
    return [str(k) for k in parameter_keys]


def _choose_unit_count(tensors: Sequence[torch.Tensor]) -> int:
    counts: List[int] = []
    for tensor in tensors:
        if not torch.is_tensor(tensor) or tensor.numel() <= 0:
            continue
        if tensor.ndim <= 0:
            counts.append(1)
        else:
            counts.append(max(1, int(tensor.shape[0])))
    if not counts:
        return 1
    freq: Dict[int, int] = {}
    for count in counts:
        freq[count] = freq.get(count, 0) + 1
    return max(freq.keys(), key=lambda value: (freq[value], value))


def _unit_rows_for_tensor(tensor: torch.Tensor, unit_count: int) -> Optional[torch.Tensor]:
    if not torch.is_tensor(tensor):
        return None
    flat = tensor.detach().to(device="cpu", dtype=torch.float32)
    if flat.numel() <= 0:
        return None
    if unit_count <= 1:
        return flat.reshape(1, -1)
    if flat.ndim == 0:
        return None
    if int(flat.shape[0]) == int(unit_count):
        return flat.reshape(int(unit_count), -1)
    if flat.ndim == 1 and int(flat.numel()) == int(unit_count):
        return flat.reshape(int(unit_count), 1)
    if int(flat.numel()) == int(unit_count):
        return flat.reshape(int(unit_count), 1)
    return None


def _short_layer_label(name: str) -> str:
    label = str(name or "").strip()
    if not label or label == "<root>":
        return "root"
    parts = label.split(".")
    if len(parts) >= 3 and parts[0] == "features" and parts[1].isdigit():
        return f"f{parts[1]}.{parts[2]}"
    if len(parts) >= 2 and parts[0] == "features" and parts[1].isdigit():
        return f"f{parts[1]}"
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return label


def infer_architecture_layers(
    state_dict: Dict[str, torch.Tensor],
    *,
    parameter_keys: Optional[Sequence[str]] = None,
    reference_state: Optional[Dict[str, torch.Tensor]] = None,
) -> List[Dict[str, Any]]:
    keys = _ordered_parameter_keys(state_dict, parameter_keys=parameter_keys)
    grouped: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []

    for key in keys:
        tensor = state_dict.get(key)
        if tensor is None or not torch.is_tensor(tensor) or not torch.is_floating_point(tensor):
            continue
        node = parameter_node_key(key)
        entry = grouped.get(node)
        if entry is None:
            entry = {"name": node, "keys": [], "tensors": []}
            grouped[node] = entry
            order.append(node)
        entry["keys"].append(key)
        entry["tensors"].append(tensor)

    layers: List[Dict[str, Any]] = []
    for node in order:
        entry = grouped[node]
        tensors = list(entry["tensors"])
        unit_count = max(1, int(_choose_unit_count(tensors)))

        sums = torch.zeros(unit_count, dtype=torch.float32)
        abs_sums = torch.zeros(unit_count, dtype=torch.float32)
        sq_sums = torch.zeros(unit_count, dtype=torch.float32)
        diff_abs_sums = torch.zeros(unit_count, dtype=torch.float32)
        elem_counts = torch.zeros(unit_count, dtype=torch.float32)
        used_keys: List[str] = []

        for key in entry["keys"]:
            tensor = state_dict.get(key)
            if tensor is None or not torch.is_tensor(tensor) or not torch.is_floating_point(tensor):
                continue
            rows = _unit_rows_for_tensor(tensor, unit_count)
            if rows is None:
                continue
            used_keys.append(key)
            sums += rows.sum(dim=1)
            abs_sums += rows.abs().sum(dim=1)
            sq_sums += (rows * rows).sum(dim=1)
            elem_counts += float(rows.shape[1])

            if isinstance(reference_state, dict):
                ref = reference_state.get(key)
                if torch.is_tensor(ref) and tuple(ref.shape) == tuple(tensor.shape):
                    ref_rows = _unit_rows_for_tensor(ref, unit_count)
                    if ref_rows is not None and tuple(ref_rows.shape) == tuple(rows.shape):
                        diff_abs_sums += (rows - ref_rows).abs().sum(dim=1)

        if not used_keys:
            continue

        safe_counts = torch.clamp(elem_counts, min=1.0)
        units: List[Dict[str, Any]] = []
        for idx in range(unit_count):
            count = float(safe_counts[idx].item())
            units.append(
                {
                    "index": int(idx),
                    "mean": float(sums[idx].item()) / count,
                    "mean_abs": float(abs_sums[idx].item()) / count,
                    "rms": math.sqrt(max(0.0, float(sq_sums[idx].item()) / count)),
                    "diff_mean_abs": float(diff_abs_sums[idx].item()) / count,
                    "count": int(max(1, round(float(elem_counts[idx].item())))),
                }
            )

        layers.append(
            {
                "name": str(node),
                "label": _short_layer_label(node),
                "unit_count": int(unit_count),
                "keys": used_keys,
                "units": units,
            }
        )
    return layers


def _node_color(
    stats: Dict[str, Any],
    *,
    max_abs_mean: float,
    max_energy: float,
    max_diff: float,
) -> np.ndarray:
    signed = float(stats.get("mean", 0.0)) / max(1e-12, float(max_abs_mean))
    energy = float(stats.get("mean_abs", stats.get("rms", 0.0))) / max(1e-12, float(max_energy))
    drift = float(stats.get("diff_mean_abs", 0.0)) / max(1e-12, float(max_diff))
    signed = max(-1.0, min(1.0, signed))
    energy = max(0.0, min(1.0, energy))
    drift = max(0.0, min(1.0, drift))

    warm = max(0.0, signed)
    cool = max(0.0, -signed)
    neutral = max(0.0, 1.0 - abs(signed))
    base = (warm * _WEIGHT_MAP_WARM) + (cool * _WEIGHT_MAP_COOL) + (neutral * _WEIGHT_MAP_NEUTRAL)
    intensity = 0.18 + (0.82 * energy)
    color = (_WEIGHT_MAP_BG * (1.0 - intensity)) + (base * intensity)
    if drift > 0.0:
        blend = 0.18 + (0.34 * drift)
        color = (color * (1.0 - blend)) + (_WEIGHT_MAP_DRIFT * blend)
    return np.clip(color, 0.0, 255.0)


def render_parameter_node_map(
    state_dict: Dict[str, torch.Tensor],
    *,
    parameter_keys: Optional[Sequence[str]] = None,
    reference_state: Optional[Dict[str, torch.Tensor]] = None,
    image_size: int = 256,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    stats = summarize_parameter_nodes(
        state_dict,
        parameter_keys=parameter_keys,
        reference_state=reference_state,
    )
    side_px = max(8, int(image_size))
    grid_side = max(1, int(math.ceil(math.sqrt(max(1, len(stats))))))
    grid = np.full((grid_side, grid_side, 3), _WEIGHT_MAP_BG.astype(np.uint8), dtype=np.uint8)

    if stats:
        max_abs_mean = max(max(abs(float(s["mean"])), float(s["mean_abs"])) for s in stats)
        max_energy = max(float(s["mean_abs"]) for s in stats)
        max_diff = max(float(s["diff_mean_abs"]) for s in stats)
        if max_diff <= 1e-12:
            max_diff = 1.0
        for idx, node_stats in enumerate(stats):
            gy = idx // grid_side
            gx = idx % grid_side
            grid[gy, gx] = _node_color(
                node_stats,
                max_abs_mean=max_abs_mean if max_abs_mean > 1e-12 else 1.0,
                max_energy=max_energy if max_energy > 1e-12 else 1.0,
                max_diff=max_diff,
            ).astype(np.uint8)

    try:
        from PIL import Image

        img = Image.fromarray(grid, mode="RGB").resize((side_px, side_px), Image.NEAREST)
        rgb = np.asarray(img, dtype=np.uint8)
    except Exception:
        ry = max(1, side_px // grid_side)
        rx = max(1, side_px // grid_side)
        rgb = np.repeat(np.repeat(grid, ry, axis=0), rx, axis=1)[:side_px, :side_px]
        if rgb.shape[0] != side_px or rgb.shape[1] != side_px:
            pad = np.full((side_px, side_px, 3), _WEIGHT_MAP_BG.astype(np.uint8), dtype=np.uint8)
            pad[: rgb.shape[0], : rgb.shape[1]] = rgb
            rgb = pad

    return rgb, {
        "node_count": int(len(stats)),
        "grid_side": int(grid_side),
        "stats": stats,
    }


def _resize_rgb_nearest(rgb: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    out_h = max(1, int(size_hw[0]))
    out_w = max(1, int(size_hw[1]))
    try:
        from PIL import Image

        img = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
        return np.asarray(img.resize((out_w, out_h), Image.NEAREST), dtype=np.uint8)
    except Exception:
        src = np.asarray(rgb, dtype=np.uint8)
        y_idx = np.floor(np.linspace(0, max(0, src.shape[0] - 1), out_h)).astype(np.int64)
        x_idx = np.floor(np.linspace(0, max(0, src.shape[1] - 1), out_w)).astype(np.int64)
        return src[y_idx][:, x_idx]


def _robust_scale(values: Sequence[float], percentile: float = 95.0) -> float:
    arr = np.asarray([abs(float(v)) for v in values], dtype=np.float32)
    if arr.size <= 0:
        return 1.0
    scale = float(np.percentile(arr, percentile))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(arr.max())
    if not np.isfinite(scale) or scale <= 1e-12:
        return 1.0
    return scale


def render_architectural_map(
    state_dict: Dict[str, torch.Tensor],
    *,
    parameter_keys: Optional[Sequence[str]] = None,
    reference_state: Optional[Dict[str, torch.Tensor]] = None,
    target_width: int = 256,
    target_height: int = 256,
    resample: str = "lanczos",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """One function for model image output.

    Builds a 1px-per-neuron raw grid, integer-upscales with *resample* to fill
    the target, and centres with black padding.  Target dimensions smaller than
    the neuron count are silently ignored (output may exceed target).  The grid
    height is forced odd so layer centering is symmetric.
    """
    layers = infer_architecture_layers(
        state_dict,
        parameter_keys=parameter_keys,
        reference_state=reference_state,
    )

    footer_h = 18
    max_unit_count = max((int(layer.get("unit_count", 0)) for layer in layers), default=1)
    # Raw height: 1 row per neuron in the widest layer, forced odd for centering
    raw_h = max_unit_count if max_unit_count % 2 == 1 else max_unit_count + 1
    row_capacity = raw_h

    # --- First pass: count unit-columns per layer ---
    raw_total_w = 0
    layer_layouts: List[Dict[str, Any]] = []
    for layer in layers:
        unit_count = int(layer.get("unit_count", 0))
        units = list(layer["units"])
        cols = max(1, int(math.ceil(float(max(1, unit_count)) / float(row_capacity))))
        abs_means = [abs(float(unit.get("mean", 0.0))) for unit in units]
        energies = [float(unit.get("mean_abs", unit.get("rms", 0.0))) for unit in units]
        diffs = [float(unit.get("diff_mean_abs", 0.0)) for unit in units]
        layer_layouts.append(
            {
                "name": str(layer["name"]),
                "label": str(layer["label"]),
                "unit_count": unit_count,
                "units": units,
                "cols": int(cols),
                "unit_x0": int(raw_total_w),
                "unit_x1": int(raw_total_w + cols),
                "max_abs_mean": _robust_scale(abs_means),
                "max_energy": _robust_scale(energies),
                "max_diff": _robust_scale(diffs),
            }
        )
        raw_total_w += cols

    raw_w = max(1, raw_total_w)

    # --- Build raw grid at 1 px per neuron ---
    raw = np.full((raw_h, raw_w, 3), _WEIGHT_MAP_BG.astype(np.uint8), dtype=np.uint8)
    for layer in layer_layouts:
        units = list(layer["units"])
        for idx, unit_stats in enumerate(units):
            col = idx // row_capacity
            pos_in_col = idx % row_capacity
            units_in_col = min(row_capacity, max(1, len(units) - col * row_capacity))
            y_offset = (row_capacity - units_in_col) // 2  # centred
            gx = int(layer["unit_x0"]) + col
            gy = y_offset + pos_in_col
            if 0 <= gy < raw_h and 0 <= gx < raw_w:
                raw[gy, gx] = _node_color(
                    unit_stats,
                    max_abs_mean=float(layer["max_abs_mean"]),
                    max_energy=float(layer["max_energy"]),
                    max_diff=float(layer["max_diff"]),
                ).astype(np.uint8)

    # --- Effective target: never smaller than neuron count ---
    eff_w = max(int(target_width), raw_w)
    eff_h = max(int(target_height), raw_h + footer_h)
    data_h = eff_h - footer_h

    # --- Integer scale: largest int factor that fits without exceeding target ---
    scale = max(1, min(data_h // raw_h, eff_w // raw_w))
    scaled_h = raw_h * scale
    scaled_w = raw_w * scale

    # --- Resize with parameterized algo (identity when scale == 1) ---
    if scale > 1:
        try:
            from PIL import Image
            _algo_map = {
                "lanczos": Image.LANCZOS,
                "nearest": Image.NEAREST,
                "bilinear": Image.BILINEAR,
                "bicubic": Image.BICUBIC,
            }
            algo = _algo_map.get(str(resample).lower(), Image.LANCZOS)
            scaled = np.asarray(
                Image.fromarray(raw).resize((scaled_w, scaled_h), algo),
                dtype=np.uint8,
            )
        except Exception:
            scaled = np.repeat(np.repeat(raw, scale, axis=0), scale, axis=1)
    else:
        scaled = raw

    # --- Centre in canvas with black padding ---
    canvas = np.zeros((eff_h, eff_w, 3), dtype=np.uint8)
    y0 = (data_h - scaled_h) // 2
    x0 = (eff_w - scaled_w) // 2
    canvas[y0:y0 + scaled_h, x0:x0 + scaled_w] = scaled

    # --- Footer labels ---
    try:
        from PIL import Image, ImageDraw, ImageFont
        im = Image.fromarray(canvas, mode="RGB")
        draw = ImageDraw.Draw(im)
        font = ImageFont.load_default()
        draw.rectangle([(0, data_h), (eff_w - 1, eff_h - 1)], fill=(8, 10, 14))
        draw.line([(0, data_h), (eff_w - 1, data_h)], fill=(52, 58, 68), width=1)

        last_label_right = -9999
        for ll in layer_layouts:
            lx0 = x0 + int(ll["unit_x0"]) * scale
            lx1 = x0 + int(ll["unit_x1"]) * scale
            lx1 = max(lx0 + 1, lx1)
            cx = (lx0 + lx1) // 2
            draw.line([(cx, data_h), (cx, eff_h - 1)], fill=(76, 84, 96), width=1)
            label = f"{ll['label']} [{int(ll['unit_count'])}]"
            label_w = len(label) * 6
            left = max(2, min(eff_w - label_w - 2, cx - (label_w // 2)))
            if left > last_label_right + 6:
                draw.text((left, data_h + 4), label, fill=(200, 208, 220), font=font)
                last_label_right = left + label_w
        canvas = np.asarray(im, dtype=np.uint8)
    except Exception:
        pass

    meta = {
        "layer_count": int(len(layer_layouts)),
        "layers": layer_layouts,
        "raw_width": int(raw_w),
        "raw_height": int(raw_h),
        "row_capacity": int(row_capacity),
        "scale": int(scale),
    }
    return canvas, meta


def render_weight_image(
    state_dict: Dict[str, torch.Tensor],
    *,
    parameter_keys: Optional[Sequence[str]] = None,
    reference_state: Optional[Dict[str, torch.Tensor]] = None,
    target_width: int = 256,
    target_height: int = 256,
    resample: str = "lanczos",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Single entry point for model weight visualisation."""
    return render_architectural_map(
        state_dict,
        parameter_keys=parameter_keys,
        reference_state=reference_state,
        target_width=target_width,
        target_height=target_height,
        resample=resample,
    )


def annotate_weight_map(
    rgb: np.ndarray,
    *,
    title: str = "",
    subtitle: str = "",
) -> np.ndarray:
    out = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8)).copy()
    try:
        from PIL import Image, ImageDraw, ImageFont

        im = Image.fromarray(out, mode="RGB")
        draw = ImageDraw.Draw(im)
        font = ImageFont.load_default()
        if title:
            draw.rectangle([(0, 0), (out.shape[1] - 1, 11)], fill=(10, 12, 16))
            draw.text((2, 1), str(title)[: max(8, out.shape[1] // 7)], fill=(225, 232, 240), font=font)
        if subtitle:
            y0 = max(0, out.shape[0] - 12)
            draw.rectangle([(0, y0), (out.shape[1] - 1, out.shape[0] - 1)], fill=(10, 12, 16))
            draw.text((2, y0 + 1), str(subtitle)[: max(8, out.shape[1] // 7)], fill=(160, 174, 190), font=font)
        out = np.asarray(im, dtype=np.uint8)
    except Exception:
        pass
    return out
