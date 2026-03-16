"""
nodus_loss_store.py -- Python ctypes binding using the native nodus_loss_store DLL.

This is the single authoritative loss-data store.  SaveRestoreNode writes via
``record()``.  Any reader (viewer, GUI, tests) reads via the query accessors.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# -- Locate the shared library ------------------------------------------

_LIB_NAME = "nodus_loss_store"

def _find_library() -> str:
    """Return the absolute path to the compiled shared library."""
    env_override = str(os.environ.get("NODUS_LOSS_STORE_LIB", "")).strip()
    if env_override:
        p = Path(env_override)
        if p.exists():
            return str(p)

    # 1. Check next to this file (pipeline/)
    here = Path(__file__).resolve().parent
    if sys.platform == "win32":
        candidate = here / f"{_LIB_NAME}.dll"
    elif sys.platform == "darwin":
        candidate = here / f"lib{_LIB_NAME}.dylib"
    else:
        candidate = here / f"lib{_LIB_NAME}.so"

    if candidate.exists():
        return str(candidate)

    # 2. Check native/build/Release (common dev layout)
    alt = here.parent / "native" / "build" / "Release" / (candidate.name)
    if alt.exists():
        return str(alt)

    # 3. System search
    found = ctypes.util.find_library(_LIB_NAME)
    if found:
        return found

    raise FileNotFoundError(
        f"Cannot find {_LIB_NAME} shared library.  "
        f"Build it with: cd native && cmake -B build && cmake --build build --config Release"
    )


_lib: ctypes.CDLL | None = None

def _get_lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(_find_library())
        _setup_signatures(_lib)
    return _lib


# -- C struct mirror ----------------------------------------------------

class CLossRecord(ctypes.Structure):
    _fields_ = [
        ("step",     ctypes.c_int32),
        ("round_id", ctypes.c_int32),
        ("loss",     ctypes.c_float),
        ("aux",      ctypes.c_float),
        ("ts",       ctypes.c_double),
    ]


class CGraphLineConfig(ctypes.Structure):
    _fields_ = [
        ("plot_x0",       ctypes.c_int32),
        ("plot_y0",       ctypes.c_int32),
        ("plot_w",        ctypes.c_int32),
        ("plot_h",        ctypes.c_int32),
        ("y_min",         ctypes.c_float),
        ("y_max",         ctypes.c_float),
        ("t_min",         ctypes.c_double),
        ("t_max",         ctypes.c_double),
        ("r",             ctypes.c_uint8),
        ("g",             ctypes.c_uint8),
        ("b",             ctypes.c_uint8),
        ("a",             ctypes.c_uint8),
        ("from_step",     ctypes.c_int32),
        ("use_time_axis", ctypes.c_int32),
    ]


# -- Signature declarations ---------------------------------------------

def _setup_signatures(lib: ctypes.CDLL) -> None:
    c_store_p = ctypes.c_void_p
    c_char_p  = ctypes.c_char_p

    lib.nodus_loss_store_create.argtypes  = [ctypes.c_int, ctypes.c_int]
    lib.nodus_loss_store_create.restype   = c_store_p

    lib.nodus_loss_store_destroy.argtypes = [c_store_p]
    lib.nodus_loss_store_destroy.restype  = None

    lib.nodus_loss_store_get_global.argtypes = []
    lib.nodus_loss_store_get_global.restype  = c_store_p

    lib.nodus_loss_store_record.argtypes  = [
        c_store_p, c_char_p, ctypes.c_float, ctypes.c_float,
        ctypes.c_int32, ctypes.c_double,
    ]
    lib.nodus_loss_store_record.restype   = ctypes.c_int32

    lib.nodus_loss_store_clear.argtypes   = [c_store_p]
    lib.nodus_loss_store_clear.restype    = None

    lib.nodus_loss_store_clear_channel.argtypes = [c_store_p, c_char_p]
    lib.nodus_loss_store_clear_channel.restype  = ctypes.c_int

    lib.nodus_loss_store_channel_count.argtypes = [c_store_p]
    lib.nodus_loss_store_channel_count.restype  = ctypes.c_int

    lib.nodus_loss_store_channel_name.argtypes  = [
        c_store_p, ctypes.c_int, c_char_p, ctypes.c_int,
    ]
    lib.nodus_loss_store_channel_name.restype   = ctypes.c_int

    lib.nodus_loss_store_channel_length.argtypes = [c_store_p, c_char_p]
    lib.nodus_loss_store_channel_length.restype  = ctypes.c_int32

    lib.nodus_loss_store_channel_cursor.argtypes = [c_store_p, c_char_p]
    lib.nodus_loss_store_channel_cursor.restype  = ctypes.c_int32

    lib.nodus_loss_store_query_since.argtypes = [
        c_store_p, c_char_p, ctypes.c_int32,
        ctypes.POINTER(CLossRecord), ctypes.c_int32,
    ]
    lib.nodus_loss_store_query_since.restype  = ctypes.c_int32

    lib.nodus_loss_store_latest.argtypes = [
        c_store_p, c_char_p, ctypes.POINTER(CLossRecord),
    ]
    lib.nodus_loss_store_latest.restype  = ctypes.c_int

    lib.nodus_loss_store_channel_data_ptr.argtypes = [
        c_store_p, c_char_p,
        ctypes.POINTER(ctypes.POINTER(CLossRecord)),
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.nodus_loss_store_channel_data_ptr.restype  = ctypes.c_int

    lib.nodus_loss_store_lock.argtypes   = [c_store_p]
    lib.nodus_loss_store_lock.restype    = None
    lib.nodus_loss_store_unlock.argtypes = [c_store_p]
    lib.nodus_loss_store_unlock.restype  = None

    # -- graph rendering --
    lib.nodus_loss_store_render_graph_line.argtypes = [
        c_store_p, c_char_p, ctypes.POINTER(CGraphLineConfig),
        ctypes.POINTER(ctypes.c_uint8),
    ]
    lib.nodus_loss_store_render_graph_line.restype = ctypes.c_int32

    lib.nodus_loss_store_channel_y_range.argtypes = [
        c_store_p, c_char_p, ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ]
    lib.nodus_loss_store_channel_y_range.restype = ctypes.c_int

    lib.nodus_loss_store_global_time_range.argtypes = [
        c_store_p,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ]
    lib.nodus_loss_store_global_time_range.restype = ctypes.c_int

    # -- bulk field extractors --
    lib.nodus_loss_store_get_loss_array.argtypes = [
        c_store_p, c_char_p, ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
    ]
    lib.nodus_loss_store_get_loss_array.restype = ctypes.c_int32

    lib.nodus_loss_store_get_ts_array.argtypes = [
        c_store_p, c_char_p, ctypes.c_int32,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int32,
    ]
    lib.nodus_loss_store_get_ts_array.restype = ctypes.c_int32

    lib.nodus_loss_store_get_step_array.argtypes = [
        c_store_p, c_char_p, ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32), ctypes.c_int32,
    ]
    lib.nodus_loss_store_get_step_array.restype = ctypes.c_int32

    # -- batch renderer --
    lib.nodus_loss_store_render_all_lines.argtypes = [
        c_store_p,
        ctypes.c_int32, ctypes.c_int32,  # plot_w, plot_h
        ctypes.c_float,                  # display_start_frac
        ctypes.c_int32,                  # num_channels
        ctypes.POINTER(ctypes.c_char_p), # channel_keys
        ctypes.POINTER(ctypes.c_uint8),  # colors (4 * num_channels)
        ctypes.POINTER(ctypes.c_uint8),  # out_rgba
        ctypes.POINTER(ctypes.c_float),  # out_y_min
        ctypes.POINTER(ctypes.c_float),  # out_y_max
        ctypes.POINTER(ctypes.c_double), # out_t_min
        ctypes.POINTER(ctypes.c_double), # out_t_max
    ]
    lib.nodus_loss_store_render_all_lines.restype = ctypes.c_int32

    # -- scrub ring --
    c_ring_p = ctypes.c_void_p
    c_uint8_p = ctypes.POINTER(ctypes.c_uint8)

    lib.nodus_scrub_ring_get_global.argtypes = []
    lib.nodus_scrub_ring_get_global.restype  = c_ring_p

    lib.nodus_scrub_ring_create.argtypes = []
    lib.nodus_scrub_ring_create.restype  = c_ring_p

    lib.nodus_scrub_ring_destroy.argtypes = [c_ring_p]
    lib.nodus_scrub_ring_destroy.restype  = None

    lib.nodus_scrub_ring_push.argtypes = [
        c_ring_p,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_double,  # step, round_id, ts
        ctypes.c_float, ctypes.c_char_p, ctypes.c_uint32, # loss, key, flags
        ctypes.c_uint32, ctypes.c_uint32,                 # image_w, image_h
        c_uint8_p, ctypes.c_uint32,                        # training_image, len
        c_uint8_p, ctypes.c_uint32,                        # output_image, len
        c_uint8_p, ctypes.c_uint32,                        # target_data, len
        c_uint8_p, c_uint8_p, c_uint8_p,                   # thumb0..2
    ]
    lib.nodus_scrub_ring_push.restype = ctypes.c_int32

    lib.nodus_scrub_ring_clear.argtypes = [c_ring_p]
    lib.nodus_scrub_ring_clear.restype  = None

    lib.nodus_scrub_ring_length.argtypes = [c_ring_p]
    lib.nodus_scrub_ring_length.restype  = ctypes.c_int32

    lib.nodus_scrub_ring_capacity.argtypes = [c_ring_p]
    lib.nodus_scrub_ring_capacity.restype  = ctypes.c_int32

    lib.nodus_scrub_ring_write_cursor.argtypes = [c_ring_p]
    lib.nodus_scrub_ring_write_cursor.restype  = ctypes.c_int32

    lib.nodus_scrub_ring_get_meta.argtypes = [
        c_ring_p, ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),   # out_step
        ctypes.POINTER(ctypes.c_int32),   # out_round_id
        ctypes.POINTER(ctypes.c_double),  # out_ts
        ctypes.POINTER(ctypes.c_float),   # out_loss
        ctypes.c_char_p, ctypes.c_int,    # out_channel_key, buflen
        ctypes.POINTER(ctypes.c_uint32),  # out_flags
        ctypes.POINTER(ctypes.c_uint32),  # out_image_w
        ctypes.POINTER(ctypes.c_uint32),  # out_image_h
        ctypes.POINTER(ctypes.c_uint32),  # out_target_len
        ctypes.POINTER(ctypes.c_uint32),  # out_output_w
        ctypes.POINTER(ctypes.c_uint32),  # out_output_h
    ]
    lib.nodus_scrub_ring_get_meta.restype = ctypes.c_int

    lib.nodus_scrub_ring_copy_training_image.argtypes = [
        c_ring_p, ctypes.c_int32, c_uint8_p, ctypes.c_uint32,
    ]
    lib.nodus_scrub_ring_copy_training_image.restype = ctypes.c_int32

    lib.nodus_scrub_ring_copy_target.argtypes = [
        c_ring_p, ctypes.c_int32, c_uint8_p, ctypes.c_uint32,
    ]
    lib.nodus_scrub_ring_copy_target.restype = ctypes.c_int32

    lib.nodus_scrub_ring_copy_output_image.argtypes = [
        c_ring_p, ctypes.c_int32, c_uint8_p, ctypes.c_uint32,
    ]
    lib.nodus_scrub_ring_copy_output_image.restype = ctypes.c_int32

    lib.nodus_scrub_ring_copy_thumbnail.argtypes = [
        c_ring_p, ctypes.c_int32, ctypes.c_int, c_uint8_p, ctypes.c_uint32,
    ]
    lib.nodus_scrub_ring_copy_thumbnail.restype = ctypes.c_int32

    lib.nodus_scrub_ring_reduce_output.argtypes = [
        c_ring_p, ctypes.c_int32, ctypes.c_uint32, ctypes.c_uint32,
    ]
    lib.nodus_scrub_ring_reduce_output.restype = ctypes.c_int

    lib.nodus_scrub_ring_reduce_training.argtypes = [
        c_ring_p, ctypes.c_int32, ctypes.c_uint32, ctypes.c_uint32,
    ]
    lib.nodus_scrub_ring_reduce_training.restype = ctypes.c_int

    lib.nodus_scrub_ring_lock.argtypes = [c_ring_p]
    lib.nodus_scrub_ring_lock.restype  = None

    lib.nodus_scrub_ring_unlock.argtypes = [c_ring_p]
    lib.nodus_scrub_ring_unlock.restype  = None

    # -- Composite cache signatures ----------------------------------------

    c_cache_p = ctypes.c_void_p
    c_frame_p = ctypes.c_void_p

    lib.nodus_composite_build_and_push.argtypes = [
        c_ring_p, ctypes.c_int32,            # ring, ring_index
        ctypes.c_uint32, ctypes.c_uint32,    # panel_w, panel_h
        c_cache_p,                            # cache
    ]
    lib.nodus_composite_build_and_push.restype = ctypes.c_int32

    lib.nodus_composite_cache_create.argtypes = [ctypes.c_int32]
    lib.nodus_composite_cache_create.restype  = c_cache_p

    lib.nodus_composite_cache_destroy.argtypes = [c_cache_p]
    lib.nodus_composite_cache_destroy.restype  = None

    lib.nodus_composite_cache_clear.argtypes = [c_cache_p]
    lib.nodus_composite_cache_clear.restype  = None

    lib.nodus_composite_cache_length.argtypes = [c_cache_p]
    lib.nodus_composite_cache_length.restype  = ctypes.c_int32

    lib.nodus_composite_cache_capacity.argtypes = [c_cache_p]
    lib.nodus_composite_cache_capacity.restype  = ctypes.c_int32

    lib.nodus_composite_cache_get_meta.argtypes = [
        c_cache_p, ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),   # out_step
        ctypes.POINTER(ctypes.c_int32),   # out_round_id
        ctypes.POINTER(ctypes.c_double),  # out_ts
        ctypes.POINTER(ctypes.c_float),   # out_loss
        ctypes.POINTER(ctypes.c_uint32),  # out_flags
        ctypes.POINTER(ctypes.c_uint32),  # out_panel_w
        ctypes.POINTER(ctypes.c_uint32),  # out_panel_h
        ctypes.POINTER(ctypes.c_int32),   # out_source_ring_cursor
    ]
    lib.nodus_composite_cache_get_meta.restype = ctypes.c_int

    lib.nodus_composite_cache_copy_panel.argtypes = [
        c_cache_p, ctypes.c_int32, ctypes.c_int,
        c_uint8_p, ctypes.c_uint32,
    ]
    lib.nodus_composite_cache_copy_panel.restype = ctypes.c_int32

    # -- Weight state store ----------------------------------------------

    c_weight_state_p = ctypes.c_void_p
    c_weight_image_p = ctypes.c_void_p
    c_char_pp = ctypes.POINTER(ctypes.c_char_p)
    c_float_p = ctypes.POINTER(ctypes.c_float)
    c_float_pp = ctypes.POINTER(c_float_p)
    c_i32_p = ctypes.POINTER(ctypes.c_int32)

    lib.nodus_weight_state_store_get_global.argtypes = []
    lib.nodus_weight_state_store_get_global.restype = c_weight_state_p

    lib.nodus_weight_state_store_publish_flat.argtypes = [
        c_weight_state_p,
        c_char_p,
        c_char_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        c_char_pp,
        c_float_pp,
        c_i32_p,
        c_i32_p,
        ctypes.c_int32,
    ]
    lib.nodus_weight_state_store_publish_flat.restype = ctypes.c_int

    lib.nodus_weight_state_store_get_meta.argtypes = [
        c_weight_state_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_uint64),
        c_char_p,
        ctypes.c_int,
        c_char_p,
        ctypes.c_int,
        c_char_p,
        ctypes.c_int,
    ]
    lib.nodus_weight_state_store_get_meta.restype = ctypes.c_int

    lib.nodus_weight_state_store_count.argtypes = [c_weight_state_p]
    lib.nodus_weight_state_store_count.restype = ctypes.c_int32

    lib.nodus_weight_state_store_get_meta_at.argtypes = [
        c_weight_state_p,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_uint64),
        c_char_p,
        ctypes.c_int,
        c_char_p,
        ctypes.c_int,
        c_char_p,
        ctypes.c_int,
    ]
    lib.nodus_weight_state_store_get_meta_at.restype = ctypes.c_int

    lib.nodus_weight_state_store_get_meta_for.argtypes = [
        c_weight_state_p,
        c_char_p,
        c_char_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_uint64),
        c_char_p,
        ctypes.c_int,
    ]
    lib.nodus_weight_state_store_get_meta_for.restype = ctypes.c_int

    # -- Weight image store ----------------------------------------------

    lib.nodus_weight_image_store_get_global.argtypes = []
    lib.nodus_weight_image_store_get_global.restype = c_weight_image_p

    lib.nodus_weight_image_store_length.argtypes = [c_weight_image_p]
    lib.nodus_weight_image_store_length.restype = ctypes.c_int32

    lib.nodus_weight_image_store_capacity.argtypes = [c_weight_image_p]
    lib.nodus_weight_image_store_capacity.restype = ctypes.c_int32

    lib.nodus_weight_image_store_set_limits.argtypes = [
        c_weight_image_p,
        ctypes.c_int32,
        ctypes.c_uint64,
    ]
    lib.nodus_weight_image_store_set_limits.restype = ctypes.c_int

    lib.nodus_weight_image_store_get_stats.argtypes = [
        c_weight_image_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.nodus_weight_image_store_get_stats.restype = ctypes.c_int

    lib.nodus_weight_image_store_configure_latest.argtypes = [
        c_weight_state_p,
        c_weight_image_p,
        ctypes.c_int,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.nodus_weight_image_store_configure_latest.restype = ctypes.c_int

    lib.nodus_weight_image_store_get_active_config.argtypes = [
        c_weight_image_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        c_char_p,
        ctypes.c_int,
        c_char_p,
        ctypes.c_int,
    ]
    lib.nodus_weight_image_store_get_active_config.restype = ctypes.c_int

    lib.nodus_weight_image_store_render_latest.argtypes = [
        c_weight_state_p,
        c_weight_image_p,
        ctypes.c_int,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.nodus_weight_image_store_render_latest.restype = ctypes.c_int

    lib.nodus_weight_image_store_measure_for.argtypes = [
        c_weight_state_p,
        c_char_p,
        c_char_p,
        ctypes.c_int,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.nodus_weight_image_store_measure_for.restype = ctypes.c_int

    lib.nodus_weight_image_store_render_for.argtypes = [
        c_weight_state_p,
        c_weight_image_p,
        c_char_p,
        c_char_p,
        ctypes.c_int,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.nodus_weight_image_store_render_for.restype = ctypes.c_int

    lib.nodus_weight_image_store_get_meta.argtypes = [
        c_weight_image_p,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint32),
        c_char_p,
        ctypes.c_int,
        c_char_p,
        ctypes.c_int,
        c_char_p,
        ctypes.c_int,
    ]
    lib.nodus_weight_image_store_get_meta.restype = ctypes.c_int

    lib.nodus_weight_image_store_copy_image.argtypes = [
        c_weight_image_p,
        ctypes.c_int32,
        c_uint8_p,
        ctypes.c_uint64,
    ]
    lib.nodus_weight_image_store_copy_image.restype = ctypes.c_int32

    lib.nodus_weight_image_store_mark_checkpoint.argtypes = [
        c_weight_image_p,
        ctypes.c_uint64,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.nodus_weight_image_store_mark_checkpoint.restype = ctypes.c_int


# -- Python dataclass for query results --

@dataclass(slots=True)
class LossRecord:
    step: int = 0
    round_id: int = 0
    loss: float = 0.0
    aux: float = 0.0
    ts: float = 0.0


# -- High-level wrapper -------------------------------------------------

class NodusLossStore:
    """Python wrapper around the native nodus_loss_store DLL.

    Usage::

        store = NodusLossStore(max_channels=64, max_records=100_000)
        store.record("train/total", loss=0.5, aux=0.0, round_id=1, ts=time.time())
        keys = store.channel_keys()
        data = store.query_since("train/total", from_step=0)
        store.close()
    """

    def __init__(self, max_channels: int = 64,
                 max_records: int = 100_000,
                 _raw_handle: ctypes.c_void_p | None = None):
        lib = _get_lib()
        self._lib = lib
        self._owns = _raw_handle is None
        if _raw_handle is not None:
            self._handle = _raw_handle
        else:
            self._handle = lib.nodus_loss_store_create(
                int(max_channels), int(max_records))
            if not self._handle:
                raise MemoryError("nodus_loss_store_create returned NULL")

    @classmethod
    def get_global(cls) -> "NodusLossStore":
        """Return the process-global singleton backed by OS shared memory.

        Every process that loads the DLL and calls this receives the SAME
        physical memory.  The returned wrapper does NOT own the handle --
        ``close()`` is a no-op.
        """
        lib = _get_lib()
        handle = lib.nodus_loss_store_get_global()
        if not handle:
            raise MemoryError("nodus_loss_store_get_global returned NULL")
        return cls(_raw_handle=handle)

    # -- lifecycle --

    def close(self) -> None:
        if self._handle and self._owns:
            self._lib.nodus_loss_store_destroy(self._handle)
        self._handle = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- write --

    def record(self, channel_key: str, loss: float, aux: float = 0.0,
               round_id: int = 0, ts: float = 0.0) -> int:
        """Append a loss record.  Returns the step index assigned."""
        step = self._lib.nodus_loss_store_record(
            self._handle,
            channel_key.encode("utf-8"),
            ctypes.c_float(loss),
            ctypes.c_float(aux),
            ctypes.c_int32(round_id),
            ctypes.c_double(ts),
        )
        return int(step)

    def clear(self) -> None:
        self._lib.nodus_loss_store_clear(self._handle)

    def clear_channel(self, channel_key: str) -> bool:
        rc = self._lib.nodus_loss_store_clear_channel(
            self._handle, channel_key.encode("utf-8"))
        return rc == 0

    # -- read: enumeration --

    def channel_count(self) -> int:
        return int(self._lib.nodus_loss_store_channel_count(self._handle))

    def channel_keys(self) -> List[str]:
        n = self.channel_count()
        keys: List[str] = []
        buf = ctypes.create_string_buffer(64)
        for i in range(n):
            rc = self._lib.nodus_loss_store_channel_name(
                self._handle, i, buf, 64)
            if rc >= 0:
                keys.append(buf.value.decode("utf-8"))
        return keys

    # -- read: per-channel --

    def channel_length(self, channel_key: str) -> int:
        return int(self._lib.nodus_loss_store_channel_length(
            self._handle, channel_key.encode("utf-8")))

    def channel_cursor(self, channel_key: str) -> int:
        return int(self._lib.nodus_loss_store_channel_cursor(
            self._handle, channel_key.encode("utf-8")))

    def query_since(self, channel_key: str,
                    from_step: int = 0,
                    max_records: int = 100_000) -> List[LossRecord]:
        buf = (CLossRecord * max_records)()
        n = self._lib.nodus_loss_store_query_since(
            self._handle,
            channel_key.encode("utf-8"),
            ctypes.c_int32(from_step),
            buf,
            ctypes.c_int32(max_records),
        )
        if n < 0:
            return []
        return [
            LossRecord(
                step=int(buf[i].step),
                round_id=int(buf[i].round_id),
                loss=float(buf[i].loss),
                aux=float(buf[i].aux),
                ts=float(buf[i].ts),
            )
            for i in range(n)
        ]

    def query_all(self, channel_key: str) -> List[LossRecord]:
        return self.query_since(channel_key, from_step=0)

    def latest(self, channel_key: str) -> Optional[LossRecord]:
        rec = CLossRecord()
        rc = self._lib.nodus_loss_store_latest(
            self._handle, channel_key.encode("utf-8"),
            ctypes.byref(rec))
        if rc != 0:
            return None
        return LossRecord(
            step=int(rec.step),
            round_id=int(rec.round_id),
            loss=float(rec.loss),
            aux=float(rec.aux),
            ts=float(rec.ts),
        )

    def summary(self) -> dict:
        """Per-channel summary matching the old LossAccumulator.summary() shape."""
        out = {}
        for ck in self.channel_keys():
            length = self.channel_length(ck)
            cursor = self.channel_cursor(ck)
            lat = self.latest(ck)
            out[ck] = {
                "length": length,
                "cursor": cursor,
                "latest_loss": lat.loss if lat else None,
                "latest_step": lat.step if lat else None,
            }
        return out

    # -- Bulk field extraction (flat arrays for tensor wrapping) --------

    def get_loss_array(self, channel_key: str, from_step: int = 0,
                       max_records: int = 100_000) -> "torch.Tensor":
        """Return a float32 CPU tensor of loss values for the channel."""
        import torch
        buf = (ctypes.c_float * max_records)()
        n = self._lib.nodus_loss_store_get_loss_array(
            self._handle, channel_key.encode("utf-8"),
            ctypes.c_int32(from_step), buf, ctypes.c_int32(max_records))
        if n <= 0:
            return torch.empty(0, dtype=torch.float32)
        return torch.frombuffer(bytearray(ctypes.string_at(buf, n * 4)),
                                dtype=torch.float32).clone()

    def get_ts_array(self, channel_key: str, from_step: int = 0,
                     max_records: int = 100_000) -> "torch.Tensor":
        """Return a float64 CPU tensor of timestamp values."""
        import torch
        buf = (ctypes.c_double * max_records)()
        n = self._lib.nodus_loss_store_get_ts_array(
            self._handle, channel_key.encode("utf-8"),
            ctypes.c_int32(from_step), buf, ctypes.c_int32(max_records))
        if n <= 0:
            return torch.empty(0, dtype=torch.float64)
        return torch.frombuffer(bytearray(ctypes.string_at(buf, n * 8)),
                                dtype=torch.float64).clone()

    def get_step_array(self, channel_key: str, from_step: int = 0,
                       max_records: int = 100_000) -> "torch.Tensor":
        """Return an int32 CPU tensor of step indices."""
        import torch
        buf = (ctypes.c_int32 * max_records)()
        n = self._lib.nodus_loss_store_get_step_array(
            self._handle, channel_key.encode("utf-8"),
            ctypes.c_int32(from_step), buf, ctypes.c_int32(max_records))
        if n <= 0:
            return torch.empty(0, dtype=torch.int32)
        return torch.frombuffer(bytearray(ctypes.string_at(buf, n * 4)),
                                dtype=torch.int32).clone()

    def to_device(self, channel_key: str, device: str = "cpu",
                  from_step: int = 0) -> Dict[str, "torch.Tensor"]:
        """Return dict of field tensors transferred to the given device.

        This copies data out of the C store into PyTorch tensors and moves
        them to the target device (e.g. ``"cuda:0"`` for GPU compute).
        """
        import torch
        return {
            "loss": self.get_loss_array(channel_key, from_step).to(device),
            "ts":   self.get_ts_array(channel_key, from_step).to(device),
            "step": self.get_step_array(channel_key, from_step).to(device),
        }

    # -- Y-range / time-range helpers ----------------------------------

    def channel_y_range(self, channel_key: str,
                        from_step: int = 0) -> Optional[Tuple[float, float]]:
        """Return (y_min, y_max) with 5% padding, or None if empty."""
        lo = ctypes.c_float(0.0)
        hi = ctypes.c_float(0.0)
        rc = self._lib.nodus_loss_store_channel_y_range(
            self._handle, channel_key.encode("utf-8"),
            ctypes.c_int32(from_step),
            ctypes.byref(lo), ctypes.byref(hi))
        if rc != 0:
            return None
        return (float(lo.value), float(hi.value))

    def global_time_range(self) -> Optional[Tuple[float, float]]:
        """Return (t_min, t_max) across all channels, or None."""
        lo = ctypes.c_double(0.0)
        hi = ctypes.c_double(0.0)
        rc = self._lib.nodus_loss_store_global_time_range(
            self._handle, ctypes.byref(lo), ctypes.byref(hi))
        if rc != 0:
            return None
        return (float(lo.value), float(hi.value))

    # -- Graph line renderer (C-side) ----------------------------------

    def render_graph_line(
        self,
        channel_key: str,
        plot_w: int,
        plot_h: int,
        y_min: float,
        y_max: float,
        t_min: float = 0.0,
        t_max: float = 0.0,
        color: Tuple[int, int, int, int] = (255, 255, 255, 255),
        from_step: int = 0,
        use_time_axis: bool = True,
    ) -> Optional["numpy.ndarray"]:
        """Render a loss channel into an RGBA overlay image (numpy HxWx4 uint8).

        The returned array has shape (plot_h, plot_w, 4) with pre-multiplied
        alpha.  The background is fully transparent -- ready for compositing
        over any background.
        """
        import numpy as np
        pw, ph = int(plot_w), int(plot_h)
        if pw <= 0 or ph <= 0:
            return None
        buf = (ctypes.c_uint8 * (pw * ph * 4))()
        ctypes.memset(buf, 0, pw * ph * 4)
        cfg = CGraphLineConfig(
            plot_x0=0, plot_y0=0, plot_w=pw, plot_h=ph,
            y_min=ctypes.c_float(y_min), y_max=ctypes.c_float(y_max),
            t_min=ctypes.c_double(t_min), t_max=ctypes.c_double(t_max),
            r=color[0], g=color[1], b=color[2], a=color[3],
            from_step=ctypes.c_int32(from_step),
            use_time_axis=1 if use_time_axis else 0,
        )
        n = self._lib.nodus_loss_store_render_graph_line(
            self._handle, channel_key.encode("utf-8"),
            ctypes.byref(cfg), buf)
        if n < 0:
            return None
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(ph, pw, 4).copy()
        return arr

    def render_graph_line_torch(
        self,
        channel_key: str,
        plot_w: int,
        plot_h: int,
        y_min: float,
        y_max: float,
        t_min: float = 0.0,
        t_max: float = 0.0,
        color: Tuple[int, int, int, int] = (255, 255, 255, 255),
        from_step: int = 0,
        use_time_axis: bool = True,
        device: str = "cpu",
    ) -> Optional["torch.Tensor"]:
        """Same as render_graph_line but returns a torch uint8 tensor (H,W,4).

        If device is ``"cuda:0"`` etc., the tensor is transferred to GPU.
        """
        import torch
        arr = self.render_graph_line(
            channel_key, plot_w, plot_h, y_min, y_max,
            t_min, t_max, color, from_step, use_time_axis)
        if arr is None:
            return None
        t = torch.from_numpy(arr)
        if device != "cpu":
            t = t.to(device)
        return t

    # -- Batch render (one C call for the entire graph) --

    def render_all_lines(
        self,
        channel_keys: List[str],
        colors_rgba: List[Tuple[int, int, int, int]],
        plot_w: int,
        plot_h: int,
        display_start_frac: float = 0.0,
    ) -> Optional[Dict]:
        """Render all requested channels onto one RGBA overlay in a single C call.

        Returns a dict with keys:
            overlay  - numpy uint8 array (plot_h, plot_w, 4) RGBA
            y_min    - float  (computed padded lower y bound)
            y_max    - float  (computed padded upper y bound)
            t_min    - float  (earliest wall-clock across channels, 0 if none)
            t_max    - float  (latest wall-clock across channels, 0 if none)
            segments - int    (total line segments drawn)
        Returns None if plot dimensions are invalid or no channels given.
        """
        import numpy as np
        pw, ph = int(plot_w), int(plot_h)
        nc = len(channel_keys)
        if pw <= 0 or ph <= 0 or nc == 0:
            return None
        if len(colors_rgba) < nc:
            return None

        # Build the C array of channel key pointers.
        c_keys = (ctypes.c_char_p * nc)()
        for i, k in enumerate(channel_keys):
            c_keys[i] = k.encode("utf-8")

        # Pack colours into a flat uint8 array.
        c_colors = (ctypes.c_uint8 * (nc * 4))()
        for i, (r, g, b, a) in enumerate(colors_rgba):
            c_colors[i * 4 + 0] = r & 0xFF
            c_colors[i * 4 + 1] = g & 0xFF
            c_colors[i * 4 + 2] = b & 0xFF
            c_colors[i * 4 + 3] = a & 0xFF

        buf = (ctypes.c_uint8 * (pw * ph * 4))()
        ctypes.memset(buf, 0, pw * ph * 4)

        y_lo = ctypes.c_float(0.0)
        y_hi = ctypes.c_float(0.0)
        t_lo = ctypes.c_double(0.0)
        t_hi = ctypes.c_double(0.0)

        seg = self._lib.nodus_loss_store_render_all_lines(
            self._handle,
            ctypes.c_int32(pw), ctypes.c_int32(ph),
            ctypes.c_float(display_start_frac),
            ctypes.c_int32(nc),
            c_keys, c_colors, buf,
            ctypes.byref(y_lo), ctypes.byref(y_hi),
            ctypes.byref(t_lo), ctypes.byref(t_hi),
        )
        if seg < 0:
            return None

        arr = np.frombuffer(buf, dtype=np.uint8).reshape(ph, pw, 4).copy()
        return {
            "overlay": arr,
            "y_min": float(y_lo.value),
            "y_max": float(y_hi.value),
            "t_min": float(t_lo.value),
            "t_max": float(t_hi.value),
            "segments": int(seg),
        }

    # -- PyTorch tensor accessors (zero-copy stride views) --

    def channel_tensors(self, channel_key: str) -> Optional[Dict[str, "torch.Tensor"]]:
        """Return dict of named 1-D tensors that VIEW the C memory directly.

        Uses implicit tensor-by-memory-stride: the underlying C array is
        a contiguous array of NodusLossRecord structs (24 bytes each).
        Each field is exposed as a 1-D tensor whose storage begins at the
        field's offset and whose stride equals ``sizeof(NodusLossRecord)``.

        The caller MUST hold the store lock while reading from these tensors
        (use ``lock()`` / ``unlock()`` or the ``locked()`` context manager).

        Returns None if the channel does not exist.

        Keys: ``step`` (int32), ``round_id`` (int32), ``loss`` (float32),
              ``aux`` (float32), ``ts`` (float64).
        """
        import torch

        ptr_out = ctypes.POINTER(CLossRecord)()
        len_out = ctypes.c_int32(0)
        rc = self._lib.nodus_loss_store_channel_data_ptr(
            self._handle,
            channel_key.encode("utf-8"),
            ctypes.byref(ptr_out),
            ctypes.byref(len_out),
        )
        if rc != 0 or len_out.value <= 0:
            return None

        n = int(len_out.value)
        base_addr = ctypes.addressof(ptr_out.contents)
        record_size = ctypes.sizeof(CLossRecord)  # 24 bytes

        # Field offsets inside NodusLossRecord (C struct, packed):
        #   step:     offset 0,  int32   (4 bytes)
        #   round_id: offset 4,  int32   (4 bytes)
        #   loss:     offset 8,  float32 (4 bytes)
        #   aux:      offset 12, float32 (4 bytes)
        #   ts:       offset 16, float64 (8 bytes)
        _fields = {
            "step":     (0,  torch.int32,   4),
            "round_id": (4,  torch.int32,   4),
            "loss":     (8,  torch.float32, 4),
            "aux":      (12, torch.float32, 4),
            "ts":       (16, torch.float64, 8),
        }

        tensors: Dict[str, torch.Tensor] = {}
        for name, (offset, dtype, elem_bytes) in _fields.items():
            field_addr = base_addr + offset
            # ctypes.cast to a flat byte buffer covering the strided range
            total_bytes = (n - 1) * record_size + elem_bytes
            buf = (ctypes.c_char * total_bytes).from_address(field_addr)
            # Wrap as a 1-D tensor with stride = record_size / elem_bytes
            storage = torch.frombuffer(buf, dtype=torch.uint8)
            # View via as_strided on a reinterpret of the storage
            flat = torch.frombuffer(buf, dtype=dtype, count=1)
            # Build the strided view manually
            t = torch.as_strided(
                torch.frombuffer(
                    (ctypes.c_char * total_bytes).from_address(field_addr),
                    dtype=dtype,
                ),
                size=(n,),
                stride=(record_size // elem_bytes,),
            )
            tensors[name] = t

        return tensors

    def locked(self):
        """Context manager for locking the store during tensor reads."""
        return _StoreLockCtx(self)

    def lock(self) -> None:
        self._lib.nodus_loss_store_lock(self._handle)

    def unlock(self) -> None:
        self._lib.nodus_loss_store_unlock(self._handle)


class _StoreLockCtx:
    __slots__ = ("_store",)

    def __init__(self, store: NodusLossStore):
        self._store = store

    def __enter__(self):
        self._store.lock()
        return self._store

    def __exit__(self, *exc):
        self._store.unlock()


# ======================================================================
#  Scrub Ring -- cross-process training-image & frame cache
# ======================================================================

# Mirror the C constants.
SCRUB_IMAGE_W      = 256
SCRUB_IMAGE_H      = 256
SCRUB_IMAGE_C      = 4
SCRUB_IMAGE_BYTES  = SCRUB_IMAGE_W * SCRUB_IMAGE_H * SCRUB_IMAGE_C
SCRUB_TARGET_BYTES = SCRUB_IMAGE_BYTES
SCRUB_THUMB_W      = 64
SCRUB_THUMB_H      = 64
SCRUB_THUMB_C      = 3
SCRUB_THUMB_BYTES  = SCRUB_THUMB_W * SCRUB_THUMB_H * SCRUB_THUMB_C
SCRUB_NUM_THUMBS   = 3
SCRUB_RING_CAPACITY = 1536
SCRUB_BLUE_ZONE     = 512

SCRUB_FLAG_HAS_IMAGE    = 0x01
SCRUB_FLAG_HAS_TARGET   = 0x02
SCRUB_FLAG_HAS_THUMBS   = 0x04
SCRUB_FLAG_HAS_GRADIENT = 0x08
SCRUB_FLAG_CHECKPOINT   = 0x10
SCRUB_FLAG_HAS_OUTPUT   = 0x20
SCRUB_FLAG_REDUCED      = 0x40


@dataclass(slots=True)
class ScrubMeta:
    step: int = 0
    round_id: int = 0
    ts: float = 0.0
    loss: float = 0.0
    channel_key: str = ""
    flags: int = 0
    image_w: int = 0
    image_h: int = 0
    target_len: int = 0
    output_w: int = 0
    output_h: int = 0


class NodusScrubRing:
    """Python wrapper around the native scrub ring shared memory."""

    def __init__(self, _raw_handle: ctypes.c_void_p | None = None):
        lib = _get_lib()
        self._lib = lib
        self._owns = _raw_handle is None
        if _raw_handle is not None:
            self._handle = _raw_handle
        else:
            self._handle = lib.nodus_scrub_ring_create()
            if not self._handle:
                raise MemoryError("nodus_scrub_ring_create returned NULL")

    @classmethod
    def get_global(cls) -> "NodusScrubRing":
        lib = _get_lib()
        handle = lib.nodus_scrub_ring_get_global()
        if not handle:
            raise MemoryError("nodus_scrub_ring_get_global returned NULL")
        return cls(_raw_handle=handle)

    def close(self) -> None:
        if self._handle and self._owns:
            self._lib.nodus_scrub_ring_destroy(self._handle)
        self._handle = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- write --

    def push(
        self,
        step: int,
        round_id: int,
        ts: float,
        loss: float,
        channel_key: str,
        flags: int = 0,
        image_w: int = 0,
        image_h: int = 0,
        training_image: "numpy.ndarray | None" = None,
        output_image: "numpy.ndarray | None" = None,
        target_data: "numpy.ndarray | None" = None,
        thumb0: "numpy.ndarray | None" = None,
        thumb1: "numpy.ndarray | None" = None,
        thumb2: "numpy.ndarray | None" = None,
    ) -> int:
        import numpy as np

        key_b = channel_key.encode("utf-8") if isinstance(channel_key, str) else channel_key

        img_ptr = None
        img_len = 0
        if training_image is not None:
            img = np.ascontiguousarray(training_image, dtype=np.uint8)
            img_ptr = img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
            img_len = img.nbytes

        out_ptr = None
        out_len = 0
        if output_image is not None:
            oimg = np.ascontiguousarray(output_image, dtype=np.uint8)
            out_ptr = oimg.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
            out_len = oimg.nbytes

        tgt_ptr = None
        tgt_len = 0
        if target_data is not None:
            tgt = np.ascontiguousarray(target_data, dtype=np.uint8)
            tgt_ptr = tgt.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
            tgt_len = tgt.nbytes

        t0_ptr = t1_ptr = t2_ptr = None
        if thumb0 is not None:
            t0 = np.ascontiguousarray(thumb0, dtype=np.uint8)
            t0_ptr = t0.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        if thumb1 is not None:
            t1 = np.ascontiguousarray(thumb1, dtype=np.uint8)
            t1_ptr = t1.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        if thumb2 is not None:
            t2 = np.ascontiguousarray(thumb2, dtype=np.uint8)
            t2_ptr = t2.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        return int(self._lib.nodus_scrub_ring_push(
            self._handle,
            ctypes.c_int32(step),
            ctypes.c_int32(round_id),
            ctypes.c_double(ts),
            ctypes.c_float(loss),
            key_b,
            ctypes.c_uint32(flags),
            ctypes.c_uint32(image_w),
            ctypes.c_uint32(image_h),
            img_ptr, ctypes.c_uint32(img_len),
            out_ptr, ctypes.c_uint32(out_len),
            tgt_ptr, ctypes.c_uint32(tgt_len),
            t0_ptr, t1_ptr, t2_ptr,
        ))

    def clear(self) -> None:
        self._lib.nodus_scrub_ring_clear(self._handle)

    # -- read --

    def length(self) -> int:
        return int(self._lib.nodus_scrub_ring_length(self._handle))

    def capacity(self) -> int:
        return int(self._lib.nodus_scrub_ring_capacity(self._handle))

    def write_cursor(self) -> int:
        return int(self._lib.nodus_scrub_ring_write_cursor(self._handle))

    def get_meta(self, index: int) -> Optional[ScrubMeta]:
        step = ctypes.c_int32()
        round_id = ctypes.c_int32()
        ts = ctypes.c_double()
        loss = ctypes.c_float()
        key_buf = ctypes.create_string_buffer(64)
        flags = ctypes.c_uint32()
        image_w = ctypes.c_uint32()
        image_h = ctypes.c_uint32()
        target_len = ctypes.c_uint32()
        output_w = ctypes.c_uint32()
        output_h = ctypes.c_uint32()

        rc = self._lib.nodus_scrub_ring_get_meta(
            self._handle, ctypes.c_int32(index),
            ctypes.byref(step), ctypes.byref(round_id), ctypes.byref(ts),
            ctypes.byref(loss), key_buf, 64,
            ctypes.byref(flags),
            ctypes.byref(image_w), ctypes.byref(image_h),
            ctypes.byref(target_len),
            ctypes.byref(output_w), ctypes.byref(output_h),
        )
        if rc != 0:
            return None
        return ScrubMeta(
            step=step.value,
            round_id=round_id.value,
            ts=ts.value,
            loss=loss.value,
            channel_key=key_buf.value.decode("utf-8", errors="replace"),
            flags=flags.value,
            image_w=image_w.value,
            image_h=image_h.value,
            target_len=target_len.value,
            output_w=output_w.value,
            output_h=output_h.value,
        )

    def copy_training_image(self, index: int) -> "Optional[numpy.ndarray]":
        import numpy as np
        buf = (ctypes.c_uint8 * SCRUB_IMAGE_BYTES)()
        n = self._lib.nodus_scrub_ring_copy_training_image(
            self._handle, ctypes.c_int32(index), buf,
            ctypes.c_uint32(SCRUB_IMAGE_BYTES))
        if n <= 0:
            return None
        return np.ctypeslib.as_array(buf)[:n].copy()

    def copy_training_image_shaped(self, index: int) -> "Optional[numpy.ndarray]":
        """Return training image as (H, W, C) uint8 array, or None."""
        import numpy as np
        meta = self.get_meta(index)
        if meta is None:
            return None
        raw = self.copy_training_image(index)
        if raw is None:
            return None
        h, w = meta.image_h, meta.image_w
        expected = h * w * SCRUB_IMAGE_C
        if len(raw) < expected:
            return None
        return raw[:expected].reshape(h, w, SCRUB_IMAGE_C)

    def copy_target(self, index: int) -> "Optional[numpy.ndarray]":
        import numpy as np
        buf = (ctypes.c_uint8 * SCRUB_TARGET_BYTES)()
        n = self._lib.nodus_scrub_ring_copy_target(
            self._handle, ctypes.c_int32(index), buf,
            ctypes.c_uint32(SCRUB_TARGET_BYTES))
        if n <= 0:
            return None
        return np.ctypeslib.as_array(buf)[:n].copy()

    def copy_output_image(self, index: int) -> "Optional[numpy.ndarray]":
        import numpy as np
        buf = (ctypes.c_uint8 * SCRUB_IMAGE_BYTES)()
        n = self._lib.nodus_scrub_ring_copy_output_image(
            self._handle, ctypes.c_int32(index), buf,
            ctypes.c_uint32(SCRUB_IMAGE_BYTES))
        if n <= 0:
            return None
        return np.ctypeslib.as_array(buf)[:n].copy()

    def copy_output_image_shaped(self, index: int) -> "Optional[numpy.ndarray]":
        """Return output image as (H, W, C) uint8 array, or None."""
        import numpy as np
        meta = self.get_meta(index)
        if meta is None or meta.output_w == 0 or meta.output_h == 0:
            return None
        raw = self.copy_output_image(index)
        if raw is None:
            return None
        h, w = meta.output_h, meta.output_w
        expected = h * w * SCRUB_IMAGE_C
        if len(raw) < expected:
            return None
        return raw[:expected].reshape(h, w, SCRUB_IMAGE_C)

    def copy_thumbnail(self, index: int, thumb_idx: int) -> "Optional[numpy.ndarray]":
        import numpy as np
        buf = (ctypes.c_uint8 * SCRUB_THUMB_BYTES)()
        n = self._lib.nodus_scrub_ring_copy_thumbnail(
            self._handle, ctypes.c_int32(index), thumb_idx,
            buf, ctypes.c_uint32(SCRUB_THUMB_BYTES))
        if n <= 0:
            return None
        return np.ctypeslib.as_array(buf)[:n].copy().reshape(
            SCRUB_THUMB_H, SCRUB_THUMB_W, SCRUB_THUMB_C)

    # -- quality reduction (GUI-side cache management) --

    def reduce_output(self, index: int,
                      target_w: int = 0, target_h: int = 0) -> bool:
        """Downsample the output image at `index` to target_w x target_h.

        Pass 0,0 to discard entirely.  The reduction is irreversible and
        happens in-place in shared memory.  Returns True on success.
        """
        rc = self._lib.nodus_scrub_ring_reduce_output(
            self._handle, ctypes.c_int32(index),
            ctypes.c_uint32(target_w), ctypes.c_uint32(target_h))
        return rc == 0

    def reduce_training(self, index: int,
                        target_w: int = 0, target_h: int = 0) -> bool:
        """Downsample the training image at `index` to target_w x target_h.

        Pass 0,0 to discard entirely.  Returns True on success.
        """
        rc = self._lib.nodus_scrub_ring_reduce_training(
            self._handle, ctypes.c_int32(index),
            ctypes.c_uint32(target_w), ctypes.c_uint32(target_h))
        return rc == 0

    # -- locking --

    def lock(self) -> None:
        self._lib.nodus_scrub_ring_lock(self._handle)

    def unlock(self) -> None:
        self._lib.nodus_scrub_ring_unlock(self._handle)

    def locked(self):
        return _RingLockCtx(self)


class _RingLockCtx:
    __slots__ = ("_ring",)

    def __init__(self, ring: NodusScrubRing):
        self._ring = ring

    def __enter__(self):
        self._ring.lock()
        return self._ring

    def __exit__(self, *exc):
        self._ring.unlock()


# ====================================================================
#  Composite cache constants
# ====================================================================

COMPOSITE_PANEL_MAX_W = 256
COMPOSITE_PANEL_MAX_H = 256
COMPOSITE_PANEL_C = 3
COMPOSITE_PANEL_MAX_BYTES = COMPOSITE_PANEL_MAX_W * COMPOSITE_PANEL_MAX_H * COMPOSITE_PANEL_C
COMPOSITE_CACHE_CAPACITY = 512


@dataclass(slots=True)
class CompositeFrameMeta:
    step: int = 0
    round_id: int = 0
    ts: float = 0.0
    loss: float = 0.0
    flags: int = 0
    panel_w: int = 0
    panel_h: int = 0
    source_ring_cursor: int = 0


class NodusCompositeCache:
    """GUI-side composite frame cache backed by native C ring.

    The cache holds resized RGB panels built from scrub ring entries.
    Create one per GUI process; it is *not* shared between processes.
    """

    def __init__(self, capacity: int = COMPOSITE_CACHE_CAPACITY):
        lib = _get_lib()
        self._lib = lib
        self._handle = lib.nodus_composite_cache_create(ctypes.c_int32(capacity))
        if not self._handle:
            raise RuntimeError("nodus_composite_cache_create returned NULL")

    def close(self):
        if self._handle:
            self._lib.nodus_composite_cache_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    # -- build helpers --

    def build_and_push(
        self,
        ring: NodusScrubRing,
        ring_index: int,
        panel_w: int,
        panel_h: int,
    ) -> int:
        """Build an RGB composite from scrub ring entry and push to cache.

        Returns the cache write cursor, -1 on error.
        """
        return int(self._lib.nodus_composite_build_and_push(
            ring._handle,
            ctypes.c_int32(ring_index),
            ctypes.c_uint32(panel_w),
            ctypes.c_uint32(panel_h),
            self._handle,
        ))

    # -- read --

    def length(self) -> int:
        return int(self._lib.nodus_composite_cache_length(self._handle))

    def capacity(self) -> int:
        return int(self._lib.nodus_composite_cache_capacity(self._handle))

    def clear(self):
        self._lib.nodus_composite_cache_clear(self._handle)

    def get_meta(self, index: int) -> Optional[CompositeFrameMeta]:
        step = ctypes.c_int32()
        round_id = ctypes.c_int32()
        ts = ctypes.c_double()
        loss = ctypes.c_float()
        flags = ctypes.c_uint32()
        panel_w = ctypes.c_uint32()
        panel_h = ctypes.c_uint32()
        src_cursor = ctypes.c_int32()

        rc = self._lib.nodus_composite_cache_get_meta(
            self._handle, ctypes.c_int32(index),
            ctypes.byref(step), ctypes.byref(round_id),
            ctypes.byref(ts), ctypes.byref(loss),
            ctypes.byref(flags),
            ctypes.byref(panel_w), ctypes.byref(panel_h),
            ctypes.byref(src_cursor),
        )
        if rc != 0:
            return None
        return CompositeFrameMeta(
            step=step.value,
            round_id=round_id.value,
            ts=ts.value,
            loss=loss.value,
            flags=flags.value,
            panel_w=panel_w.value,
            panel_h=panel_h.value,
            source_ring_cursor=src_cursor.value,
        )

    def copy_panel(self, index: int, panel_idx: int) -> "Optional[numpy.ndarray]":
        """Copy one RGB panel.  panel_idx: 0=target, 1=input, 2=output.

        Returns (H, W, 3) uint8 numpy array, or None.
        """
        import numpy as np
        buf = (ctypes.c_uint8 * COMPOSITE_PANEL_MAX_BYTES)()
        n = self._lib.nodus_composite_cache_copy_panel(
            self._handle, ctypes.c_int32(index), ctypes.c_int(panel_idx),
            buf, ctypes.c_uint32(COMPOSITE_PANEL_MAX_BYTES))
        if n <= 0:
            return None
        meta = self.get_meta(index)
        if meta is None:
            return None
        h, w = meta.panel_h, meta.panel_w
        expected = h * w * COMPOSITE_PANEL_C
        if n < expected:
            return None
        return np.ctypeslib.as_array(buf)[:expected].copy().reshape(h, w, COMPOSITE_PANEL_C)

    def copy_all_panels(self, index: int) -> "Optional[list]":
        """Return [target, input, output] as list of (H,W,3) arrays, or None."""
        panels = []
        for pi in range(3):
            p = self.copy_panel(index, pi)
            if p is None:
                return None
            panels.append(p)
        return panels


# ====================================================================
#  Weight State Store + Rendered Image Cache
# ====================================================================

WEIGHT_IMAGE_FLAG_CHECKPOINT = 0x01


@dataclass(slots=True)
class WeightStateMeta:
    publish_seq: int = 0
    generation: int = 0
    architecture_version: int = 0
    round_id: int = 0
    cycle: int = 0
    step: int = 0
    param_count: int = 0
    blob_bytes: int = 0
    model_name: str = ""
    node_id: str = ""
    blob_name: str = ""


@dataclass(slots=True)
class WeightImageMeta:
    image_seq: int = 0
    state_publish_seq: int = 0
    generation: int = 0
    architecture_version: int = 0
    round_id: int = 0
    cycle: int = 0
    step: int = 0
    width: int = 0
    height: int = 0
    channels: int = 0
    stride_bytes: int = 0
    mode: int = 0
    target_width: int = 0
    target_height: int = 0
    byte_count: int = 0
    flags: int = 0
    model_name: str = ""
    node_id: str = ""
    blob_name: str = ""


@dataclass(slots=True)
class WeightImageConfig:
    state_publish_seq: int = 0
    generation: int = 0
    architecture_version: int = 0
    round_id: int = 0
    cycle: int = 0
    step: int = 0
    mode: int = 0
    target_width: int = 0
    target_height: int = 0
    render_width: int = 0
    render_height: int = 0
    render_channels: int = 0
    render_stride_bytes: int = 0
    model_name: str = ""
    node_id: str = ""


@dataclass(slots=True)
class WeightImageStoreStats:
    max_entries: int = 0
    entry_count: int = 0
    max_total_bytes: int = 0
    total_bytes: int = 0


def _pack_weight_state_dict(
    state_dict: Dict[str, object],
    *,
    parameter_keys: Optional[List[str]] = None,
    parameter_plan: Optional[Sequence[Tuple[str, str]]] = None,
) -> Tuple[List[bytes], List["numpy.ndarray"], List[int], List[int]]:
    import numpy as np
    import torch

    if parameter_plan is not None:
        keys = [(str(state_key), str(render_key)) for state_key, render_key in parameter_plan]
    elif parameter_keys is None:
        keys = [
            (str(k), str(k))
            for k, v in state_dict.items()
            if torch.is_tensor(v) and torch.is_floating_point(v)
        ]
    else:
        keys = [(str(k), str(k)) for k in parameter_keys]

    packed_names: List[bytes] = []
    packed_arrays: List[np.ndarray] = []
    packed_numel: List[int] = []
    packed_shape0: List[int] = []

    for state_key, render_key in keys:
        value = state_dict.get(state_key)
        if value is None or (not torch.is_tensor(value)) or (not torch.is_floating_point(value)):
            continue
        tensor = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
        flat = tensor.reshape(-1)
        if int(flat.numel()) <= 0:
            continue
        arr = flat.numpy()
        packed_names.append(render_key.encode("utf-8", errors="ignore"))
        packed_arrays.append(arr)
        packed_numel.append(int(arr.size))
        packed_shape0.append(int(tensor.shape[0]) if int(tensor.ndim) > 0 else 1)

    return packed_names, packed_arrays, packed_numel, packed_shape0


class NodusWeightStateStore:
    """Cross-process store for the latest published floating-point model state."""

    def __init__(self, _raw_handle: ctypes.c_void_p):
        self._lib = _get_lib()
        self._handle = _raw_handle

    @classmethod
    def get_global(cls) -> "NodusWeightStateStore":
        lib = _get_lib()
        handle = lib.nodus_weight_state_store_get_global()
        if not handle:
            raise MemoryError("nodus_weight_state_store_get_global returned NULL")
        return cls(_raw_handle=handle)

    def publish_state_dict(
        self,
        state_dict: Dict[str, object],
        *,
        model_name: str,
        node_id: str = "",
        round_id: int = 0,
        cycle: int = 0,
        step: int = 0,
        generation: int = 0,
        architecture_version: int = 0,
        publish_seq: int = 0,
        parameter_keys: Optional[List[str]] = None,
        parameter_plan: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> Optional[WeightStateMeta]:
        packed_names, packed_arrays, packed_numel, packed_shape0 = _pack_weight_state_dict(
            state_dict,
            parameter_keys=parameter_keys,
            parameter_plan=parameter_plan,
        )
        count = len(packed_arrays)
        if count <= 0:
            return None

        c_char_p_arr = ctypes.c_char_p * count
        c_float_p = ctypes.POINTER(ctypes.c_float)
        c_float_p_arr = c_float_p * count
        c_i32_arr = ctypes.c_int32 * count

        names_arr = c_char_p_arr(*packed_names)
        data_arr = c_float_p_arr(*[arr.ctypes.data_as(c_float_p) for arr in packed_arrays])
        numel_arr = c_i32_arr(*packed_numel)
        shape0_arr = c_i32_arr(*packed_shape0)

        rc = self._lib.nodus_weight_state_store_publish_flat(
            self._handle,
            str(model_name).encode("utf-8"),
            str(node_id).encode("utf-8"),
            ctypes.c_int32(int(round_id)),
            ctypes.c_int32(int(cycle)),
            ctypes.c_int32(int(step)),
            ctypes.c_uint64(int(generation)),
            ctypes.c_uint64(int(architecture_version)),
            ctypes.c_uint64(int(publish_seq)),
            names_arr,
            data_arr,
            numel_arr,
            shape0_arr,
            ctypes.c_int32(count),
        )
        if int(rc) != 0:
            return None
        return self.get_meta()

    def _decode_meta(
        self,
        *,
        publish_seq: ctypes.c_uint64,
        generation: ctypes.c_uint64,
        architecture_version: ctypes.c_uint64,
        round_id: ctypes.c_int32,
        cycle: ctypes.c_int32,
        step: ctypes.c_int32,
        param_count: ctypes.c_int32,
        blob_bytes: ctypes.c_uint64,
        model_name,
        node_id,
        blob_name,
    ) -> WeightStateMeta:
        return WeightStateMeta(
            publish_seq=int(publish_seq.value),
            generation=int(generation.value),
            architecture_version=int(architecture_version.value),
            round_id=int(round_id.value),
            cycle=int(cycle.value),
            step=int(step.value),
            param_count=int(param_count.value),
            blob_bytes=int(blob_bytes.value),
            model_name=model_name.value.decode("utf-8", errors="replace"),
            node_id=node_id.value.decode("utf-8", errors="replace"),
            blob_name=blob_name.value.decode("utf-8", errors="replace"),
        )

    def get_meta(self) -> Optional[WeightStateMeta]:
        publish_seq = ctypes.c_uint64(0)
        generation = ctypes.c_uint64(0)
        architecture_version = ctypes.c_uint64(0)
        round_id = ctypes.c_int32(0)
        cycle = ctypes.c_int32(0)
        step = ctypes.c_int32(0)
        param_count = ctypes.c_int32(0)
        blob_bytes = ctypes.c_uint64(0)
        model_name = ctypes.create_string_buffer(64)
        node_id = ctypes.create_string_buffer(64)
        blob_name = ctypes.create_string_buffer(128)
        rc = self._lib.nodus_weight_state_store_get_meta(
            self._handle,
            ctypes.byref(publish_seq),
            ctypes.byref(generation),
            ctypes.byref(architecture_version),
            ctypes.byref(round_id),
            ctypes.byref(cycle),
            ctypes.byref(step),
            ctypes.byref(param_count),
            ctypes.byref(blob_bytes),
            model_name,
            len(model_name),
            node_id,
            len(node_id),
            blob_name,
            len(blob_name),
        )
        if int(rc) != 0:
            return None
        return self._decode_meta(
            publish_seq=publish_seq,
            generation=generation,
            architecture_version=architecture_version,
            round_id=round_id,
            cycle=cycle,
            step=step,
            param_count=param_count,
            blob_bytes=blob_bytes,
            model_name=model_name,
            node_id=node_id,
            blob_name=blob_name,
        )

    def count(self) -> int:
        return int(self._lib.nodus_weight_state_store_count(self._handle))

    def get_meta_at(self, index: int) -> Optional[WeightStateMeta]:
        publish_seq = ctypes.c_uint64(0)
        generation = ctypes.c_uint64(0)
        architecture_version = ctypes.c_uint64(0)
        round_id = ctypes.c_int32(0)
        cycle = ctypes.c_int32(0)
        step = ctypes.c_int32(0)
        param_count = ctypes.c_int32(0)
        blob_bytes = ctypes.c_uint64(0)
        model_name = ctypes.create_string_buffer(64)
        node_id = ctypes.create_string_buffer(64)
        blob_name = ctypes.create_string_buffer(128)
        rc = self._lib.nodus_weight_state_store_get_meta_at(
            self._handle,
            ctypes.c_int32(int(index)),
            ctypes.byref(publish_seq),
            ctypes.byref(generation),
            ctypes.byref(architecture_version),
            ctypes.byref(round_id),
            ctypes.byref(cycle),
            ctypes.byref(step),
            ctypes.byref(param_count),
            ctypes.byref(blob_bytes),
            model_name,
            len(model_name),
            node_id,
            len(node_id),
            blob_name,
            len(blob_name),
        )
        if int(rc) != 0:
            return None
        return self._decode_meta(
            publish_seq=publish_seq,
            generation=generation,
            architecture_version=architecture_version,
            round_id=round_id,
            cycle=cycle,
            step=step,
            param_count=param_count,
            blob_bytes=blob_bytes,
            model_name=model_name,
            node_id=node_id,
            blob_name=blob_name,
        )

    def get_meta_for(self, *, model_name: str, node_id: str = "") -> Optional[WeightStateMeta]:
        publish_seq = ctypes.c_uint64(0)
        generation = ctypes.c_uint64(0)
        architecture_version = ctypes.c_uint64(0)
        round_id = ctypes.c_int32(0)
        cycle = ctypes.c_int32(0)
        step = ctypes.c_int32(0)
        param_count = ctypes.c_int32(0)
        blob_bytes = ctypes.c_uint64(0)
        blob_name = ctypes.create_string_buffer(128)
        rc = self._lib.nodus_weight_state_store_get_meta_for(
            self._handle,
            str(model_name).encode("utf-8"),
            str(node_id).encode("utf-8"),
            ctypes.byref(publish_seq),
            ctypes.byref(generation),
            ctypes.byref(architecture_version),
            ctypes.byref(round_id),
            ctypes.byref(cycle),
            ctypes.byref(step),
            ctypes.byref(param_count),
            ctypes.byref(blob_bytes),
            blob_name,
            len(blob_name),
        )
        if int(rc) != 0:
            return None
        return WeightStateMeta(
            publish_seq=int(publish_seq.value),
            generation=int(generation.value),
            architecture_version=int(architecture_version.value),
            round_id=int(round_id.value),
            cycle=int(cycle.value),
            step=int(step.value),
            param_count=int(param_count.value),
            blob_bytes=int(blob_bytes.value),
            model_name=str(model_name),
            node_id=str(node_id),
            blob_name=blob_name.value.decode("utf-8", errors="replace"),
        )

    def list_meta(self) -> List[WeightStateMeta]:
        return [
            meta
            for meta in (self.get_meta_at(i) for i in range(max(0, int(self.count()))))
            if meta is not None
        ]


class NodusWeightImageStore:
    """Cross-process cache of GUI-rendered weight images."""

    def __init__(self, _raw_handle: ctypes.c_void_p):
        self._lib = _get_lib()
        self._handle = _raw_handle

    @classmethod
    def get_global(cls) -> "NodusWeightImageStore":
        lib = _get_lib()
        handle = lib.nodus_weight_image_store_get_global()
        if not handle:
            raise MemoryError("nodus_weight_image_store_get_global returned NULL")
        return cls(_raw_handle=handle)

    def length(self) -> int:
        return int(self._lib.nodus_weight_image_store_length(self._handle))

    def capacity(self) -> int:
        return int(self._lib.nodus_weight_image_store_capacity(self._handle))

    def set_limits(self, *, max_entries: int, max_total_bytes: int) -> bool:
        rc = self._lib.nodus_weight_image_store_set_limits(
            self._handle,
            ctypes.c_int32(int(max_entries)),
            ctypes.c_uint64(int(max_total_bytes)),
        )
        return int(rc) == 0

    def stats(self) -> Optional[WeightImageStoreStats]:
        max_entries = ctypes.c_int32(0)
        entry_count = ctypes.c_int32(0)
        max_total_bytes = ctypes.c_uint64(0)
        total_bytes = ctypes.c_uint64(0)
        rc = self._lib.nodus_weight_image_store_get_stats(
            self._handle,
            ctypes.byref(max_entries),
            ctypes.byref(entry_count),
            ctypes.byref(max_total_bytes),
            ctypes.byref(total_bytes),
        )
        if int(rc) != 0:
            return None
        return WeightImageStoreStats(
            max_entries=int(max_entries.value),
            entry_count=int(entry_count.value),
            max_total_bytes=int(max_total_bytes.value),
            total_bytes=int(total_bytes.value),
        )

    def configure_latest(
        self,
        state_store: NodusWeightStateStore,
        *,
        mode: int = 1,
        target_width: int = 256,
        target_height: int = 256,
    ) -> Optional[WeightImageConfig]:
        rc = self._lib.nodus_weight_image_store_configure_latest(
            state_store._handle,
            self._handle,
            ctypes.c_int(int(mode)),
            ctypes.c_int32(int(target_width)),
            ctypes.c_int32(int(target_height)),
        )
        if int(rc) != 0:
            return None
        return self.get_active_config()

    def get_active_config(self) -> Optional[WeightImageConfig]:
        state_publish_seq = ctypes.c_uint64(0)
        generation = ctypes.c_uint64(0)
        architecture_version = ctypes.c_uint64(0)
        round_id = ctypes.c_int32(0)
        cycle = ctypes.c_int32(0)
        step = ctypes.c_int32(0)
        mode = ctypes.c_int32(0)
        target_width = ctypes.c_int32(0)
        target_height = ctypes.c_int32(0)
        render_width = ctypes.c_int32(0)
        render_height = ctypes.c_int32(0)
        render_channels = ctypes.c_int32(0)
        render_stride_bytes = ctypes.c_int32(0)
        model_name = ctypes.create_string_buffer(64)
        node_id = ctypes.create_string_buffer(64)
        rc = self._lib.nodus_weight_image_store_get_active_config(
            self._handle,
            ctypes.byref(state_publish_seq),
            ctypes.byref(generation),
            ctypes.byref(architecture_version),
            ctypes.byref(round_id),
            ctypes.byref(cycle),
            ctypes.byref(step),
            ctypes.byref(mode),
            ctypes.byref(target_width),
            ctypes.byref(target_height),
            ctypes.byref(render_width),
            ctypes.byref(render_height),
            ctypes.byref(render_channels),
            ctypes.byref(render_stride_bytes),
            model_name,
            len(model_name),
            node_id,
            len(node_id),
        )
        if int(rc) != 0:
            return None
        return WeightImageConfig(
            state_publish_seq=int(state_publish_seq.value),
            generation=int(generation.value),
            architecture_version=int(architecture_version.value),
            round_id=int(round_id.value),
            cycle=int(cycle.value),
            step=int(step.value),
            mode=int(mode.value),
            target_width=int(target_width.value),
            target_height=int(target_height.value),
            render_width=int(render_width.value),
            render_height=int(render_height.value),
            render_channels=int(render_channels.value),
            render_stride_bytes=int(render_stride_bytes.value),
            model_name=model_name.value.decode("utf-8", errors="replace"),
            node_id=node_id.value.decode("utf-8", errors="replace"),
        )

    def render_latest(
        self,
        state_store: NodusWeightStateStore,
        *,
        mode: int = 1,
        target_width: int = 256,
        target_height: int = 256,
    ) -> bool:
        rc = self._lib.nodus_weight_image_store_render_latest(
            state_store._handle,
            self._handle,
            ctypes.c_int(int(mode)),
            ctypes.c_int32(int(target_width)),
            ctypes.c_int32(int(target_height)),
        )
        return int(rc) == 0

    def measure_for(
        self,
        state_store: NodusWeightStateStore,
        *,
        model_name: str,
        node_id: str = "",
        mode: int = 1,
        target_width: int = 256,
        target_height: int = 256,
    ) -> Optional[WeightImageConfig]:
        state_publish_seq = ctypes.c_uint64(0)
        generation = ctypes.c_uint64(0)
        architecture_version = ctypes.c_uint64(0)
        round_id = ctypes.c_int32(0)
        cycle = ctypes.c_int32(0)
        step = ctypes.c_int32(0)
        render_width = ctypes.c_int32(0)
        render_height = ctypes.c_int32(0)
        render_channels = ctypes.c_int32(0)
        render_stride_bytes = ctypes.c_int32(0)
        rc = self._lib.nodus_weight_image_store_measure_for(
            state_store._handle,
            str(model_name).encode("utf-8"),
            str(node_id).encode("utf-8"),
            ctypes.c_int(int(mode)),
            ctypes.c_int32(int(target_width)),
            ctypes.c_int32(int(target_height)),
            ctypes.byref(state_publish_seq),
            ctypes.byref(generation),
            ctypes.byref(architecture_version),
            ctypes.byref(round_id),
            ctypes.byref(cycle),
            ctypes.byref(step),
            ctypes.byref(render_width),
            ctypes.byref(render_height),
            ctypes.byref(render_channels),
            ctypes.byref(render_stride_bytes),
        )
        if int(rc) != 0:
            return None
        return WeightImageConfig(
            state_publish_seq=int(state_publish_seq.value),
            generation=int(generation.value),
            architecture_version=int(architecture_version.value),
            round_id=int(round_id.value),
            cycle=int(cycle.value),
            step=int(step.value),
            mode=int(mode),
            target_width=int(target_width),
            target_height=int(target_height),
            render_width=int(render_width.value),
            render_height=int(render_height.value),
            render_channels=int(render_channels.value),
            render_stride_bytes=int(render_stride_bytes.value),
            model_name=str(model_name),
            node_id=str(node_id),
        )

    def render_for(
        self,
        state_store: NodusWeightStateStore,
        *,
        model_name: str,
        node_id: str = "",
        mode: int = 1,
        target_width: int = 256,
        target_height: int = 256,
    ) -> bool:
        rc = self._lib.nodus_weight_image_store_render_for(
            state_store._handle,
            self._handle,
            str(model_name).encode("utf-8"),
            str(node_id).encode("utf-8"),
            ctypes.c_int(int(mode)),
            ctypes.c_int32(int(target_width)),
            ctypes.c_int32(int(target_height)),
        )
        return int(rc) == 0

    def get_meta(self, index: int) -> Optional[WeightImageMeta]:
        image_seq = ctypes.c_uint64(0)
        state_publish_seq = ctypes.c_uint64(0)
        generation = ctypes.c_uint64(0)
        architecture_version = ctypes.c_uint64(0)
        round_id = ctypes.c_int32(0)
        cycle = ctypes.c_int32(0)
        step = ctypes.c_int32(0)
        width = ctypes.c_int32(0)
        height = ctypes.c_int32(0)
        channels = ctypes.c_int32(0)
        stride_bytes = ctypes.c_int32(0)
        mode = ctypes.c_int32(0)
        target_width = ctypes.c_int32(0)
        target_height = ctypes.c_int32(0)
        byte_count = ctypes.c_uint64(0)
        flags = ctypes.c_uint32(0)
        model_name = ctypes.create_string_buffer(64)
        node_id = ctypes.create_string_buffer(64)
        blob_name = ctypes.create_string_buffer(128)
        rc = self._lib.nodus_weight_image_store_get_meta(
            self._handle,
            ctypes.c_int32(int(index)),
            ctypes.byref(image_seq),
            ctypes.byref(state_publish_seq),
            ctypes.byref(generation),
            ctypes.byref(architecture_version),
            ctypes.byref(round_id),
            ctypes.byref(cycle),
            ctypes.byref(step),
            ctypes.byref(width),
            ctypes.byref(height),
            ctypes.byref(channels),
            ctypes.byref(stride_bytes),
            ctypes.byref(mode),
            ctypes.byref(target_width),
            ctypes.byref(target_height),
            ctypes.byref(byte_count),
            ctypes.byref(flags),
            model_name,
            len(model_name),
            node_id,
            len(node_id),
            blob_name,
            len(blob_name),
        )
        if int(rc) != 0:
            return None
        return WeightImageMeta(
            image_seq=int(image_seq.value),
            state_publish_seq=int(state_publish_seq.value),
            generation=int(generation.value),
            architecture_version=int(architecture_version.value),
            round_id=int(round_id.value),
            cycle=int(cycle.value),
            step=int(step.value),
            width=int(width.value),
            height=int(height.value),
            channels=int(channels.value),
            stride_bytes=int(stride_bytes.value),
            mode=int(mode.value),
            target_width=int(target_width.value),
            target_height=int(target_height.value),
            byte_count=int(byte_count.value),
            flags=int(flags.value),
            model_name=model_name.value.decode("utf-8", errors="replace"),
            node_id=node_id.value.decode("utf-8", errors="replace"),
            blob_name=blob_name.value.decode("utf-8", errors="replace"),
        )

    def copy_image(self, index: int) -> "Optional[numpy.ndarray]":
        import numpy as np

        meta = self.get_meta(index)
        if meta is None or meta.byte_count <= 0:
            return None
        buf = (ctypes.c_uint8 * int(meta.byte_count))()
        n = self._lib.nodus_weight_image_store_copy_image(
            self._handle,
            ctypes.c_int32(int(index)),
            buf,
            ctypes.c_uint64(int(meta.byte_count)),
        )
        if int(n) <= 0:
            return None
        row_bytes = int(meta.stride_bytes) if int(meta.stride_bytes) > 0 else int(meta.width) * int(meta.channels)
        expected = int(meta.height) * row_bytes
        if int(n) < expected:
            return None
        flat = np.ctypeslib.as_array(buf)[:expected].copy()
        if row_bytes == int(meta.width) * int(meta.channels):
            return flat.reshape(int(meta.height), int(meta.width), int(meta.channels))
        rows = flat.reshape(int(meta.height), row_bytes)
        packed = rows[:, : int(meta.width) * int(meta.channels)]
        return packed.reshape(int(meta.height), int(meta.width), int(meta.channels))

    def latest_meta(self) -> Optional[WeightImageMeta]:
        length = self.length()
        if length <= 0:
            return None
        return self.get_meta(length - 1)

    def latest_image(self) -> "Optional[tuple[WeightImageMeta, numpy.ndarray]]":
        meta = self.latest_meta()
        if meta is None:
            return None
        rgb = self.copy_image(self.length() - 1)
        if rgb is None:
            return None
        return meta, rgb

    def mark_checkpoint(self, state_publish_seq: int, *, round_id: int, cycle: int) -> bool:
        rc = self._lib.nodus_weight_image_store_mark_checkpoint(
            self._handle,
            ctypes.c_uint64(int(state_publish_seq)),
            ctypes.c_int32(int(round_id)),
            ctypes.c_int32(int(cycle)),
        )
        return int(rc) == 0
