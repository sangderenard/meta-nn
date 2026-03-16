from __future__ import annotations

import ctypes
import ctypes.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


WEIGHT_IMAGE_MODES = ("parameter_groups", "architectural_tall", "architectural_wide")

_MODE_TO_C = {
    "parameter_groups": 0,
    "architectural_tall": 1,
    "architectural_wide": 2,
}

_WIMG_LIB: ctypes.CDLL | None = None


def _find_weight_image_library() -> str:
    lib_name = "nodus_loss_store"
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


def _render_weight_image_c(
    state_dict: Dict[str, torch.Tensor],
    *,
    parameter_keys: Optional[Sequence[str]] = None,
    reference_state: Optional[Dict[str, torch.Tensor]] = None,
    target_width: int = 256,
    target_height: int = 256,
    mode: str = "architectural_tall",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    mode_key = str(mode).strip().lower()
    if mode_key not in _MODE_TO_C:
        raise ValueError(f"Unsupported mode: {mode!r}")

    if parameter_keys is None:
        keys = sorted(str(k) for k in state_dict.keys())
    else:
        keys = [str(k) for k in parameter_keys]

    packed_names_b: List[bytes] = []
    packed_arrays: List[np.ndarray] = []
    packed_shape0: List[int] = []
    packed_numel: List[int] = []
    packed_ref_arrays: List[np.ndarray | None] = []
    packed_ref_numel: List[int] = []

    for key in keys:
        tensor = state_dict.get(key)
        if tensor is None or not torch.is_tensor(tensor) or not torch.is_floating_point(tensor):
            continue

        t = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
        flat = t.reshape(-1)
        if int(flat.numel()) <= 0:
            continue

        packed_names_b.append(key.encode("utf-8", errors="ignore"))
        arr = flat.numpy()
        packed_arrays.append(arr)
        packed_numel.append(int(arr.size))
        packed_shape0.append(int(t.shape[0]) if int(t.ndim) > 0 else 1)

        ref_arr: np.ndarray | None = None
        ref_n = 0
        if isinstance(reference_state, dict):
            ref_t = reference_state.get(key)
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
    reference_state: Optional[Dict[str, torch.Tensor]] = None,
    image_size: int = 256,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    return _render_weight_image_c(
        state_dict,
        parameter_keys=parameter_keys,
        reference_state=reference_state,
        target_width=max(8, int(image_size)),
        target_height=max(8, int(image_size)),
        mode="parameter_groups",
    )


def render_architectural_map(
    state_dict: Dict[str, torch.Tensor],
    *,
    parameter_keys: Optional[Sequence[str]] = None,
    reference_state: Optional[Dict[str, torch.Tensor]] = None,
    target_width: int = 256,
    target_height: int = 256,
    resample: str = "lanczos",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    _ = resample
    return _render_weight_image_c(
        state_dict,
        parameter_keys=parameter_keys,
        reference_state=reference_state,
        target_width=target_width,
        target_height=target_height,
        mode="architectural_tall",
    )


def render_weight_image(
    state_dict: Dict[str, torch.Tensor],
    *,
    parameter_keys: Optional[Sequence[str]] = None,
    reference_state: Optional[Dict[str, torch.Tensor]] = None,
    target_width: int = 256,
    target_height: int = 256,
    mode: str = "architectural_tall",
    resample: str = "lanczos",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Single entry point for model weight visualisation."""
    _ = resample
    return _render_weight_image_c(
        state_dict,
        parameter_keys=parameter_keys,
        reference_state=reference_state,
        target_width=target_width,
        target_height=target_height,
        mode=mode,
    )


def annotate_weight_map(
    rgb: np.ndarray,
    *,
    title: str = "",
    subtitle: str = "",
) -> np.ndarray:
    _ = title
    _ = subtitle
    return np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8)).copy()
