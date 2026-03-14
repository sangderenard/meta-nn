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
from typing import Dict, List, Optional, Tuple

# -- Locate the shared library ------------------------------------------

_LIB_NAME = "nodus_loss_store"

def _find_library() -> str:
    """Return the absolute path to the compiled shared library."""
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
                 max_records: int = 100_000):
        lib = _get_lib()
        self._lib = lib
        self._handle = lib.nodus_loss_store_create(
            int(max_channels), int(max_records))
        if not self._handle:
            raise MemoryError("nodus_loss_store_create returned NULL")

    # -- lifecycle --

    def close(self) -> None:
        if self._handle:
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
