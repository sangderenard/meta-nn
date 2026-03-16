from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


WEIGHT_IMAGE_MODES = (
    "parameter_groups",
    "architectural_tall",
    "architectural_wide",
    "architectural_packed",
)

_MODE_TO_C = {
    "parameter_groups": 0,
    "architectural_tall": 1,
    "architectural_wide": 2,
    "architectural_packed": 3,
}

_WIMG_LIB: ctypes.CDLL | None = None


def _find_weight_image_library() -> str:
    lib_name = "nodus_loss_store"
    env_override = str(os.environ.get("NODUS_LOSS_STORE_LIB", "")).strip()
    if env_override:
        p = Path(env_override)
        if p.exists():
            return str(p)
    here = Path(__file__).resolve().parent
    if sys.platform == "win32":
        candidate = here / f"{lib_name}.dll"
    elif sys.platform == "darwin":
        candidate = here / f"lib{lib_name}.dylib"
    else:
        candidate = here / f"lib{lib_name}.so"

    if candidate.exists():
        return str(candidate)

    alt = here.parent / "native" / "build" / "Release" / candidate.name
    if alt.exists():
        return str(alt)

    found = ctypes.util.find_library(lib_name)
    if found:
        return found

    raise FileNotFoundError(
        "Cannot find nodus_loss_store shared library for C weight rendering."
    )


def _setup_weight_image_signatures(lib: ctypes.CDLL) -> None:
    if hasattr(lib, "nodus_weight_image_from_state_dict"):
        c_char_pp = ctypes.POINTER(ctypes.c_char_p)
        c_float_p = ctypes.POINTER(ctypes.c_float)
        c_float_pp = ctypes.POINTER(c_float_p)
        c_i32_p = ctypes.POINTER(ctypes.c_int32)
        c_u8_p = ctypes.POINTER(ctypes.c_uint8)
        c_u8_pp = ctypes.POINTER(c_u8_p)

        lib.nodus_weight_image_from_state_dict.argtypes = [
            c_char_pp,
            c_float_pp,
            c_i32_p,
            c_i32_p,
            ctypes.c_int32,
            c_float_pp,
            c_i32_p,
            ctypes.c_int,
            ctypes.c_int32,
            ctypes.c_int32,
            c_u8_pp,
            c_i32_p,
            c_i32_p,
        ]
        lib.nodus_weight_image_from_state_dict.restype = ctypes.c_int

        lib.nodus_weight_image_free.argtypes = [c_u8_p]
        lib.nodus_weight_image_free.restype = None


def _get_weight_image_lib() -> ctypes.CDLL:
    global _WIMG_LIB
    if _WIMG_LIB is None:
        _WIMG_LIB = ctypes.CDLL(_find_weight_image_library())
        _setup_weight_image_signatures(_WIMG_LIB)
    return _WIMG_LIB


def logical_parameter_node_key(parameter_key: str) -> str:
    key = str(parameter_key or "").strip()
    if not key:
        return "<root>"
    if ".slots." in key:
        prefix = key.split(".slots.", 1)[0].strip()
        if prefix:
            return prefix
    if ".base." in key:
        prefix = key.split(".base.", 1)[0].strip()
        if prefix:
            return prefix
    return parameter_node_key(key)


def _ordered_tensor_keys(
    state_dict: Dict[str, Any],
    parameter_keys: Optional[Sequence[str]] = None,
) -> List[str]:
    if parameter_keys is None:
        keys = [str(k) for k in state_dict.keys()]
    else:
        keys = [str(k) for k in parameter_keys]
    out: List[str] = []
    for key in keys:
        value = state_dict.get(key)
        if torch.is_tensor(value) and torch.is_floating_point(value) and int(value.numel()) > 0:
            out.append(str(key))
    return out


def _choose_unit_count_for_keys(state_dict: Dict[str, torch.Tensor], keys: Sequence[str]) -> int:
    counts: Dict[int, int] = {}
    for key in keys:
        value = state_dict.get(str(key))
        if not torch.is_tensor(value) or int(value.numel()) <= 0:
            continue
        shape0 = int(value.shape[0]) if int(value.ndim) > 0 else 1
        counts[shape0] = counts.get(shape0, 0) + 1
    if not counts:
        return 1
    return max(counts.items(), key=lambda item: (int(item[1]), int(item[0])))[0]


def _ordered_logical_layers(
    state_dict: Dict[str, torch.Tensor],
    parameter_keys: Optional[Sequence[str]] = None,
) -> "OrderedDict[str, List[str]]":
    layers: "OrderedDict[str, List[str]]" = OrderedDict()
    for key in _ordered_tensor_keys(state_dict, parameter_keys=parameter_keys):
        layer_name = logical_parameter_node_key(key)
        layers.setdefault(str(layer_name), []).append(str(key))
    return layers


def _estimate_architectural_grid(layers: Sequence[Dict[str, Any]]) -> Tuple[int, int]:
    unit_counts = [max(1, int(layer.get("unit_count", 1) or 1)) for layer in layers]
    if not unit_counts:
        return 1, 1
    raw_h = max(unit_counts)
    if (raw_h % 2) == 0:
        raw_h += 1
    raw_w = 0
    for unit_count in unit_counts:
        raw_w += max(1, int(np.ceil(float(unit_count) / float(raw_h))))
    return max(1, int(raw_w)), max(1, int(raw_h))


def _make_layer_entry(
    state_dict: Dict[str, torch.Tensor],
    *,
    layer_name: str,
    keys: Sequence[str],
    branch_id: str,
    flow_order: int,
    active: bool = True,
    rendered: bool = True,
    label: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "name": str(layer_name),
        "label": str(label or layer_name),
        "unit_count": int(_choose_unit_count_for_keys(state_dict, keys)),
        "keys": [str(k) for k in keys],
        "branch_id": str(branch_id),
        "flow_order": int(flow_order),
        "active": bool(active),
        "rendered": bool(rendered),
    }


def _generic_weight_render_spec(
    state_dict: Dict[str, torch.Tensor],
    parameter_keys: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    logical_layers = _ordered_logical_layers(state_dict, parameter_keys=parameter_keys)
    layers: List[Dict[str, Any]] = []
    for idx, (layer_name, keys) in enumerate(logical_layers.items()):
        layers.append(
            _make_layer_entry(
                state_dict,
                layer_name=str(layer_name),
                keys=keys,
                branch_id="main",
                flow_order=idx,
            )
        )
    return {
        "version": 1,
        "layout_name": "grouped_state_dict",
        "branches": [
            {"id": "main", "label": "main", "parent_id": "", "depth": 0, "order": 0, "kind": "linear"},
        ],
        "layers": layers,
        "suppressed_layers": [],
    }


def resolve_weight_render_spec(
    state_dict: Dict[str, torch.Tensor],
    *,
    parameter_keys: Optional[Sequence[str]] = None,
    weight_render_spec: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if isinstance(weight_render_spec, dict) and isinstance(weight_render_spec.get("layers"), list):
        layers: List[Dict[str, Any]] = []
        for idx, raw_layer in enumerate(list(weight_render_spec.get("layers", []))):
            if not isinstance(raw_layer, dict):
                continue
            keys = [
                str(key)
                for key in list(raw_layer.get("keys", []))
                if torch.is_tensor(state_dict.get(str(key)))
                and torch.is_floating_point(state_dict.get(str(key)))
                and int(state_dict.get(str(key)).numel()) > 0
            ]
            if not keys:
                continue
            layers.append(
                _make_layer_entry(
                    state_dict,
                    layer_name=str(raw_layer.get("name", f"layer_{idx}")),
                    label=str(raw_layer.get("label", raw_layer.get("name", f"layer_{idx}"))),
                    keys=keys,
                    branch_id=str(raw_layer.get("branch_id", "main") or "main"),
                    flow_order=int(raw_layer.get("flow_order", idx) or idx),
                    active=bool(raw_layer.get("active", True)),
                    rendered=bool(raw_layer.get("rendered", True)),
                )
            )
        if layers:
            return {
                "version": int(weight_render_spec.get("version", 1) or 1),
                "layout_name": str(weight_render_spec.get("layout_name", "provided_layout") or "provided_layout"),
                "branches": list(weight_render_spec.get("branches", []) or []),
                "layers": sorted(layers, key=lambda row: int(row.get("flow_order", 0))),
                "suppressed_layers": [str(name) for name in list(weight_render_spec.get("suppressed_layers", []) or [])],
                "semantic_bank_active": bool(weight_render_spec.get("semantic_bank_active", False)),
            }
    # Default path must stay model-agnostic for every training scenario.
    # Specialised layouts are only used when explicitly provided by caller.
    return _generic_weight_render_spec(state_dict, parameter_keys=parameter_keys)


def parameter_plan_from_render_spec(
    state_dict: Dict[str, torch.Tensor],
    *,
    parameter_keys: Optional[Sequence[str]] = None,
    weight_render_spec: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, str]]:
    spec = resolve_weight_render_spec(
        state_dict,
        parameter_keys=parameter_keys,
        weight_render_spec=weight_render_spec,
    )
    out: List[Tuple[str, str]] = []
    def _safe_branch_id(text: str) -> str:
        raw = str(text or "main")
        safe = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in raw)
        safe = safe.strip("_")
        return safe or "main"

    for layer in list(spec.get("layers", [])):
        if not bool(layer.get("rendered", True)):
            continue
        layer_name = str(layer.get("name", "layer") or "layer")
        branch_id = _safe_branch_id(str(layer.get("branch_id", "main") or "main"))
        for idx, state_key in enumerate(list(layer.get("keys", []))):
            if not torch.is_tensor(state_dict.get(str(state_key))):
                continue
            out.append((str(state_key), f"b{branch_id}.{layer_name}.p{idx:04d}"))
    if out:
        return out
    return [(key, key) for key in _ordered_tensor_keys(state_dict, parameter_keys=parameter_keys)]


def _render_weight_image_c(
    state_dict: Dict[str, torch.Tensor],
    *,
    parameter_keys: Optional[Sequence[str]] = None,
    parameter_plan: Optional[Sequence[Tuple[str, str]]] = None,
    reference_state: Optional[Dict[str, torch.Tensor]] = None,
    target_width: int = 256,
    target_height: int = 256,
    mode: str = "architectural_tall",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    mode_key = str(mode).strip().lower()
    if mode_key not in _MODE_TO_C:
        raise ValueError(f"Unsupported mode: {mode!r}")

    if parameter_plan is not None:
        plan = [(str(state_key), str(render_key)) for state_key, render_key in parameter_plan]
    elif parameter_keys is None:
        keys = sorted(str(k) for k in state_dict.keys())
        plan = [(key, key) for key in keys]
    else:
        plan = [(str(k), str(k)) for k in parameter_keys]

    packed_names_b: List[bytes] = []
    packed_arrays: List[np.ndarray] = []
    packed_shape0: List[int] = []
    packed_numel: List[int] = []
    packed_ref_arrays: List[np.ndarray | None] = []
    packed_ref_numel: List[int] = []

    for state_key, render_key in plan:
        tensor = state_dict.get(state_key)
        if tensor is None or not torch.is_tensor(tensor) or not torch.is_floating_point(tensor):
            continue

        t = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
        flat = t.reshape(-1)
        if int(flat.numel()) <= 0:
            continue

        packed_names_b.append(render_key.encode("utf-8", errors="ignore"))
        arr = flat.numpy()
        packed_arrays.append(arr)
        packed_numel.append(int(arr.size))
        packed_shape0.append(int(t.shape[0]) if int(t.ndim) > 0 else 1)

        ref_arr: np.ndarray | None = None
        ref_n = 0
        if isinstance(reference_state, dict):
            ref_t = reference_state.get(state_key)
            if torch.is_tensor(ref_t) and tuple(ref_t.shape) == tuple(t.shape):
                ref_np = (
                    ref_t.detach().to(device="cpu", dtype=torch.float32).contiguous().reshape(-1).numpy()
                )
                if int(ref_np.size) == int(arr.size):
                    ref_arr = ref_np
                    ref_n = int(ref_np.size)
        packed_ref_arrays.append(ref_arr)
        packed_ref_numel.append(ref_n)

    n = len(packed_arrays)
    if n <= 0:
        raise ValueError("No float tensors available for rendering")

    c_char_p_arr = ctypes.c_char_p * n
    c_float_p = ctypes.POINTER(ctypes.c_float)
    c_float_p_arr = c_float_p * n
    c_i32_arr = ctypes.c_int32 * n

    names_arr = c_char_p_arr(*packed_names_b)
    data_ptrs = [a.ctypes.data_as(c_float_p) for a in packed_arrays]
    data_arr = c_float_p_arr(*data_ptrs)
    numel_arr = c_i32_arr(*packed_numel)
    shape0_arr = c_i32_arr(*packed_shape0)

    ref_ptrs: List[Any] = []
    for ref_np in packed_ref_arrays:
        if ref_np is None:
            ref_ptrs.append(c_float_p())
        else:
            ref_ptrs.append(ref_np.ctypes.data_as(c_float_p))
    ref_data_arr = c_float_p_arr(*ref_ptrs)
    ref_numel_arr = c_i32_arr(*packed_ref_numel)

    out_rgb = ctypes.POINTER(ctypes.c_uint8)()
    out_w = ctypes.c_int32(0)
    out_h = ctypes.c_int32(0)

    lib = _get_weight_image_lib()
    rc = lib.nodus_weight_image_from_state_dict(
        names_arr,
        data_arr,
        numel_arr,
        shape0_arr,
        ctypes.c_int32(n),
        ref_data_arr,
        ref_numel_arr,
        ctypes.c_int(_MODE_TO_C[mode_key]),
        ctypes.c_int32(max(8, int(target_width))),
        ctypes.c_int32(max(8, int(target_height))),
        ctypes.byref(out_rgb),
        ctypes.byref(out_w),
        ctypes.byref(out_h),
    )
    if int(rc) != 0:
        raise RuntimeError("C weight renderer failed")

    w = int(out_w.value)
    h = int(out_h.value)
    if w <= 0 or h <= 0:
        if out_rgb:
            lib.nodus_weight_image_free(out_rgb)
        raise RuntimeError("C weight renderer returned invalid dimensions")

    byte_count = w * h * 3
    try:
        rgb_flat = np.ctypeslib.as_array(out_rgb, shape=(byte_count,)).copy()
    finally:
        lib.nodus_weight_image_free(out_rgb)

    rgb = rgb_flat.reshape(h, w, 3)

    meta: Dict[str, Any] = {
        "mode": mode_key,
        "width": int(w),
        "height": int(h),
        "transposed": bool(mode_key == "architectural_wide"),
        "packed": bool(mode_key == "architectural_packed"),
    }
    return rgb, meta


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


def render_parameter_node_map(
    state_dict: Dict[str, torch.Tensor],
    *,
    parameter_keys: Optional[Sequence[str]] = None,
    weight_render_spec: Optional[Dict[str, Any]] = None,
    reference_state: Optional[Dict[str, torch.Tensor]] = None,
    image_size: int = 256,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    render_spec = resolve_weight_render_spec(
        state_dict,
        parameter_keys=parameter_keys,
        weight_render_spec=weight_render_spec,
    )
    return _render_weight_image_c(
        state_dict,
        parameter_plan=parameter_plan_from_render_spec(
            state_dict,
            parameter_keys=parameter_keys,
            weight_render_spec=render_spec,
        ),
        reference_state=reference_state,
        target_width=max(8, int(image_size)),
        target_height=max(8, int(image_size)),
        mode="parameter_groups",
    )


def render_architectural_map(
    state_dict: Dict[str, torch.Tensor],
    *,
    parameter_keys: Optional[Sequence[str]] = None,
    weight_render_spec: Optional[Dict[str, Any]] = None,
    reference_state: Optional[Dict[str, torch.Tensor]] = None,
    target_width: int = 256,
    target_height: int = 256,
    resample: str = "lanczos",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    _ = resample
    render_spec = resolve_weight_render_spec(
        state_dict,
        parameter_keys=parameter_keys,
        weight_render_spec=weight_render_spec,
    )
    rgb, meta = _render_weight_image_c(
        state_dict,
        parameter_plan=parameter_plan_from_render_spec(
            state_dict,
            parameter_keys=parameter_keys,
            weight_render_spec=render_spec,
        ),
        reference_state=reference_state,
        target_width=target_width,
        target_height=target_height,
        mode="architectural_tall",
    )
    layers = list(render_spec.get("layers", []))
    raw_w, raw_h = _estimate_architectural_grid(layers)
    meta.update(
        {
            "layout_name": str(render_spec.get("layout_name", "grouped_state_dict")),
            "layer_count": int(len(layers)),
            "raw_width": int(raw_w),
            "raw_height": int(raw_h),
            "layers": layers,
            "branches": list(render_spec.get("branches", [])),
            "suppressed_layers": [str(name) for name in list(render_spec.get("suppressed_layers", []))],
            "semantic_bank_active": bool(render_spec.get("semantic_bank_active", False)),
        }
    )
    return rgb, meta


def render_weight_image(
    state_dict: Dict[str, torch.Tensor],
    *,
    parameter_keys: Optional[Sequence[str]] = None,
    weight_render_spec: Optional[Dict[str, Any]] = None,
    reference_state: Optional[Dict[str, torch.Tensor]] = None,
    target_width: int = 256,
    target_height: int = 256,
    mode: str = "architectural_tall",
    resample: str = "lanczos",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Single entry point for model weight visualisation."""
    _ = resample
    render_spec = resolve_weight_render_spec(
        state_dict,
        parameter_keys=parameter_keys,
        weight_render_spec=weight_render_spec,
    )
    rgb, meta = _render_weight_image_c(
        state_dict,
        parameter_plan=parameter_plan_from_render_spec(
            state_dict,
            parameter_keys=parameter_keys,
            weight_render_spec=render_spec,
        ),
        reference_state=reference_state,
        target_width=target_width,
        target_height=target_height,
        mode=mode,
    )
    layers = list(render_spec.get("layers", []))
    if str(mode).strip().lower() == "parameter_groups":
        meta.update(
            {
                "layout_name": str(render_spec.get("layout_name", "grouped_state_dict")),
                "node_count": int(len(layers)),
                "grid_side": int(np.ceil(np.sqrt(max(1, len(layers))))),
                "stats": [
                    {
                        "node": str(layer.get("name", "")),
                        "count": int(len(layer.get("keys", []))),
                        "keys": list(layer.get("keys", [])),
                        "branch_id": str(layer.get("branch_id", "main")),
                    }
                    for layer in layers
                ],
                "branches": list(render_spec.get("branches", [])),
                "suppressed_layers": [str(name) for name in list(render_spec.get("suppressed_layers", []))],
            }
        )
        return rgb, meta

    raw_w, raw_h = _estimate_architectural_grid(layers)
    meta.update(
        {
            "layout_name": str(render_spec.get("layout_name", "grouped_state_dict")),
            "layer_count": int(len(layers)),
            "raw_width": int(raw_w),
            "raw_height": int(raw_h),
            "layers": layers,
            "branches": list(render_spec.get("branches", [])),
            "suppressed_layers": [str(name) for name in list(render_spec.get("suppressed_layers", []))],
            "semantic_bank_active": bool(render_spec.get("semantic_bank_active", False)),
        }
    )
    return rgb, meta


def annotate_weight_map(
    rgb: np.ndarray,
    *,
    title: str = "",
    subtitle: str = "",
) -> np.ndarray:
    _ = title
    _ = subtitle
    return np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8)).copy()
