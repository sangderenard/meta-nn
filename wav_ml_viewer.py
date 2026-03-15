import json
import math
import threading
import time
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from pipeline.nodus_loss_store import (
    NodusCompositeCache,
    NodusScrubRing,
    SCRUB_FLAG_HAS_IMAGE,
    SCRUB_FLAG_HAS_OUTPUT,
    SCRUB_FLAG_HAS_TARGET,
    SCRUB_FLAG_HAS_THUMBS,
    SCRUB_NUM_THUMBS,
    SCRUB_THUMB_H,
    SCRUB_THUMB_W,
)
from pipeline.plan_protocol import (
    MESSAGE_TYPE_EXECUTION_EVENT,
    MESSAGE_TYPE_PLAN_APPLY,
    MESSAGE_TYPE_PLAN_SNAPSHOT,
    MESSAGE_TYPE_RUN_CONTROL,
    MESSAGE_TYPE_RUNTIME_SNAPSHOT,
    MESSAGE_TYPE_WORKER_HELLO,
    ExecutionEventPayload,
    PlanApplyPayload,
    RunControlPayload,
    RuntimeSnapshotPayload,
    WorkerHelloPayload,
    is_protocol_envelope_message,
    make_envelope,
    parse_envelope,
)
from pipeline.weight_map import annotate_weight_map, render_weight_image, snapshot_parameter_state_from_state_dict

# ---------------------------------------------------------------------------
# Loss logging constants and binary record format
# ---------------------------------------------------------------------------

# Integer -> channel-key map used *only* when loading legacy v1 binary records.
_LEGACY_STAGE_NAMES: Dict[int, str] = {
    0: "cls",
    1: "gen",
    2: "disc",
    3: "trans",
    4: "wcls",
    5: "wcls_eval",
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


def _compose_weight_thumb_stack(tiles: Sequence[np.ndarray]) -> np.ndarray:
    valid_tiles = [np.ascontiguousarray(np.asarray(tile, dtype=np.uint8)) for tile in list(tiles)[:SCRUB_NUM_THUMBS]]
    if len(valid_tiles) != SCRUB_NUM_THUMBS:
        raise ValueError("expected a full scrub thumbnail tile stack")
    return np.concatenate(valid_tiles, axis=0)


def _extract_checkpoint_weight_state(
    payload: Dict[str, Any],
    model_name: Optional[str],
) -> Tuple[Optional[str], Optional[Dict[str, torch.Tensor]]]:
    if not isinstance(payload, dict):
        return None, None
    if model_name:
        state_key = f"{str(model_name)}_state"
        state_blob = payload.get(state_key)
        if isinstance(state_blob, dict):
            _keys, snap = snapshot_parameter_state_from_state_dict(state_blob)
            if snap:
                return str(model_name), snap
    for key, value in payload.items():
        if not str(key).endswith("_state") or not isinstance(value, dict):
            continue
        _keys, snap = snapshot_parameter_state_from_state_dict(value)
        if snap:
            return str(key[:-6]), snap
    if "state_dict" in payload and isinstance(payload["state_dict"], dict):
        _keys, snap = snapshot_parameter_state_from_state_dict(payload["state_dict"])
        if snap:
            return str(model_name or "state_dict"), snap
    return None, None


def _hsl_to_rgb(h: float, s: float, l: float) -> Tuple[int, int, int]:
    """Convert HSL (h in [0,360), s/l in [0,1]) to RGB (each 0-255)."""
    c = (1.0 - abs(2.0 * l - 1.0)) * s
    x = c * (1.0 - abs((h / 60.0) % 2.0 - 1.0))
    m = l - c / 2.0
    if h < 60:
        r1, g1, b1 = c, x, 0.0
    elif h < 120:
        r1, g1, b1 = x, c, 0.0
    elif h < 180:
        r1, g1, b1 = 0.0, c, x
    elif h < 240:
        r1, g1, b1 = 0.0, x, c
    elif h < 300:
        r1, g1, b1 = x, 0.0, c
    else:
        r1, g1, b1 = c, 0.0, x
    return (
        int(round((r1 + m) * 255)),
        int(round((g1 + m) * 255)),
        int(round((b1 + m) * 255)),
    )


def _channel_color(index: int, total: int) -> Tuple[int, int, int]:
    """Equal-angular-spacing colour from the HSL wheel.

    Uses the golden angle offset (137.508deg) so that any prefix of N channels
    is well-distributed even if total changes over time.
    """
    if total <= 0:
        total = 1
    hue = (index * 137.508) % 360.0
    return _hsl_to_rgb(hue, 0.72, 0.58)


# 20 bytes per record: step(i4) round(i2) stage(u1) pad(u1) loss(f4) aux(f4) ts(f4)
LOSS_RECORD_DTYPE = np.dtype([
    ("step",  "<i4"),
    ("round", "<i2"),
    ("stage", "<u1"),
    ("_pad",  "<u1"),
    ("loss",  "<f4"),
    ("aux",   "<f4"),
    ("ts",    "<f4"),
])

# Channel-key aware record for the extended binary log (v2).
# 52 bytes: step(i4) round(i2) stage(u1) pad(u1) loss(f4) aux(f4) ts(f4)
#           + channel_key (32-byte fixed UTF-8, null-padded)
LOSS_RECORD_V2_DTYPE = np.dtype([
    ("step",  "<i4"),
    ("round", "<i2"),
    ("stage", "<u1"),
    ("_pad",  "<u1"),
    ("loss",  "<f4"),
    ("aux",   "<f4"),
    ("ts",    "<f4"),
    ("channel_key", "S32"),
])


class _LossFileLogger:
    """Appends fixed-size binary loss records to a file for later analysis."""

    _FLUSH_EVERY = 200

    def __init__(self, path, start_time: float = 0.0):
        self._path = Path(path)
        self._file = None
        try:
            existing_bytes = int(self._path.stat().st_size) if self._path.exists() else 0
            # Detect existing format to count records correctly.
            v2_sz = LOSS_RECORD_V2_DTYPE.itemsize
            v1_sz = LOSS_RECORD_DTYPE.itemsize
            if existing_bytes > 0 and (existing_bytes % v2_sz == 0):
                self._counter = int(existing_bytes // v2_sz)
            else:
                self._counter = int(existing_bytes // v1_sz)
            self._file = open(self._path, "ab")
        except Exception as e:
            self._counter = 0
            print(f"[loss-logger] could not open {path}: {e}", flush=True)

    def log(self, round_idx: int, channel_key: str, loss: float, aux: float = 0.0):
        if self._file is None:
            return
        self._counter += 1
        ts = time.time()
        rec = np.zeros(1, dtype=LOSS_RECORD_V2_DTYPE)
        rec["step"][0]  = max(-(2**31), min(2**31 - 1, int(self._counter)))
        rec["round"][0] = max(-32768,   min(32767,     int(round_idx)))
        rec["stage"][0] = 0
        rec["loss"][0]  = float(loss) if math.isfinite(float(loss)) else float("nan")
        rec["aux"][0]   = float(aux)  if math.isfinite(float(aux))  else float("nan")
        rec["ts"][0]    = min(float(ts), 3.4e38)
        ck_bytes = str(channel_key).encode("utf-8")[:32]
        rec["channel_key"][0] = ck_bytes
        try:
            self._file.write(rec.tobytes())
            if self._counter % self._FLUSH_EVERY == 0:
                self._file.flush()
        except Exception:
            pass

    def flush(self):
        if self._file is not None:
            try:
                self._file.flush()
            except Exception:
                pass

    def close(self):
        if self._file is not None:
            try:
                self._file.flush()
                self._file.close()
            except Exception:
                pass
            self._file = None

    @staticmethod
    def load(path) -> np.ndarray:
        """Load all records from a loss_log.bin file.

        Detects v1 (20-byte) vs v2 (52-byte) format automatically.
        V1 records get their channel_key filled from the legacy stage map.
        """
        p = Path(path)
        if not p.exists() or p.stat().st_size == 0:
            return np.zeros(0, dtype=LOSS_RECORD_V2_DTYPE)
        data = p.read_bytes()
        sz = len(data)
        v2_n = sz // LOSS_RECORD_V2_DTYPE.itemsize
        v1_n = sz // LOSS_RECORD_DTYPE.itemsize
        # Prefer v2 if the file is evenly divisible by 52; fall back to v1.
        if v2_n > 0 and (sz % LOSS_RECORD_V2_DTYPE.itemsize == 0):
            return np.frombuffer(
                data[: v2_n * LOSS_RECORD_V2_DTYPE.itemsize],
                dtype=LOSS_RECORD_V2_DTYPE,
            ).copy()
        # v1 records -- upconvert to v2 with channel_key from legacy map.
        if v1_n > 0:
            v1 = np.frombuffer(
                data[: v1_n * LOSS_RECORD_DTYPE.itemsize],
                dtype=LOSS_RECORD_DTYPE,
            ).copy()
            v2 = np.zeros(v1_n, dtype=LOSS_RECORD_V2_DTYPE)
            for f in ("step", "round", "stage", "loss", "aux", "ts"):
                v2[f] = v1[f]
            for i, stage_int in enumerate(v1["stage"]):
                ck = _LEGACY_STAGE_NAMES.get(int(stage_int), f"stage_{int(stage_int)}")
                v2["channel_key"][i] = ck.encode("utf-8")[:32]
            return v2
        return np.zeros(0, dtype=LOSS_RECORD_V2_DTYPE)


def _wmap_thermal_rgb(t: np.ndarray) -> np.ndarray:
    """Map normalised float array t in [0,1] -> (H, W, 3) uint8 thermal colours.
    Ramp: black -> blue -> cyan -> green -> yellow -> red."""
    r = np.clip(t * 4.0 - 3.0, 0.0, 1.0)
    g = np.clip(np.minimum(t * 4.0 - 1.0, 3.0 - t * 4.0), 0.0, 1.0)
    b = np.clip(2.0 - t * 4.0, 0.0, 1.0)
    return np.round(np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def _wmap_flat_to_square(flat: torch.Tensor, sq: int) -> np.ndarray:
    """Flatten live weights -> (sq, sq, 3) thermal image, zero-padding to sq^2."""
    n = sq * sq
    arr = flat.cpu().numpy().astype(np.float32)
    if arr.shape[0] < n:
        arr = np.pad(arr, (0, n - arr.shape[0]))
    arr = arr[:n]
    lo, hi = float(arr.min()), float(arr.max())
    t = (arr - lo) / (hi - lo) if hi > lo + 1e-8 else np.zeros_like(arr)
    return _wmap_thermal_rgb(t.reshape(sq, sq))


def _wmap_diff_overlay(diff: torch.Tensor, sq: int) -> np.ndarray:
    """Absolute diff tensor -> (sq, sq) float32 in [0, 1] for overlay intensity."""
    n = sq * sq
    arr = diff.cpu().numpy().astype(np.float32)
    if arr.shape[0] < n:
        arr = np.pad(arr, (0, n - arr.shape[0]))
    arr = arr[:n]
    hi = float(arr.max())
    return (arr / hi if hi > 1e-8 else np.zeros_like(arr)).reshape(sq, sq)


def _wmap_apply_red(img: np.ndarray, diff: np.ndarray) -> np.ndarray:
    """Blend red onto img by diff saturation. img: (H,W,3) uint8, diff: (H,W) float32."""
    d = diff[..., np.newaxis]
    result = img.astype(np.float32) * (1.0 - d) + np.array([255, 0, 0], dtype=np.float32) * d
    return np.clip(result, 0, 255).astype(np.uint8)


def _wmap_build_flat(sd: Dict[str, Any], keys: List[str], model: Any) -> torch.Tensor:
    """Flatten state-dict values in the same param order as model.named_parameters().
    Missing keys are replaced with zeros of the correct shape."""
    chunks: List[torch.Tensor] = []
    param_map = dict(model.named_parameters())
    for k in keys:
        if k in sd:
            chunks.append(sd[k].detach().float().reshape(-1).cpu())
        elif k in param_map:
            chunks.append(torch.zeros(param_map[k].numel(), dtype=torch.float32))
    return torch.cat(chunks) if chunks else torch.zeros(1, dtype=torch.float32)


def _tensor_to_rgb_u8_image(x: torch.Tensor) -> np.ndarray:
    t = x.detach().to(device="cpu", dtype=torch.float32)
    if t.ndim == 4:
        t = t[0]
    if t.ndim == 2:
        t = t.unsqueeze(0)
    if t.ndim != 3:
        raise ValueError(f"Expected tensor image with ndim 2/3/4, got shape={tuple(t.shape)}")

    c = int(t.shape[0])
    if c <= 0:
        raise ValueError("Cannot visualize empty-channel tensor.")
    if c == 1:
        t = t.repeat(3, 1, 1)
    elif c == 2:
        t = torch.cat([t, t[:1]], dim=0)
    elif c > 4:
        t = t[:4]

    lo = float(t.min().item())
    hi = float(t.max().item())
    if hi <= lo + 1e-6:
        t = torch.zeros_like(t)
    elif lo < 0.0 or hi > 1.0:
        t = (t - lo) / (hi - lo)
    t = torch.clamp(t, 0.0, 1.0)
    img = (t.permute(1, 2, 0).contiguous().numpy() * 255.0).astype(np.uint8)
    return np.ascontiguousarray(img)




class _TransformerStatusOpenGLViewer:
    def __init__(
        self,
        enabled: bool,
        image_hw: Tuple[int, int],
        scale: int = 3,
        cycle_slots: int = 0,
        graph_h: int = 120,
        loss_store=None,
    ):
        self.enabled = bool(enabled)
        self.image_h = max(8, int(image_hw[0]))
        self.image_w = max(8, int(image_hw[1]))
        _ = max(1, int(scale))

        self.panel_w = max(8, min(256, int(self.image_w)))
        self.panel_h = max(8, min(256, int(self.image_h)))
        self.num_panels = 3
        self.top_bar_h = 84
        self.graph_h = max(0, int(graph_h))
        self.graph_toolbar_h = 22 if self.graph_h > 0 else 0
        self.graph_total_h = int(self.graph_h + self.graph_toolbar_h)
        # Extra pixels appended only to the right weight-map panel so it is taller than
        # the graph row; gives breathing room and avoids visually cutting off the map.
        self._weight_map_extra_h = 24
        # +2 columns: one sidebar on each side of the 3 main panels
        self.window_w = int(self.panel_w * (self.num_panels + 2))
        self.window_h = int(self.top_bar_h + (self.panel_h * 2) + self.graph_total_h + self._weight_map_extra_h)
        self._col_x = self.panel_w  # x-offset: main panels shift right by one column

        self._ready = False
        self._failed = False
        self._pygame = None
        self._gl = None
        self._textures = None
        self._stop_requested = False
        self._shutdown_save: Optional[bool] = None  # None=no shutdown, True=save, False=nosave
        self._launch_script: Optional[str] = None   # .bat to re-launch training
        self._output_dir: Optional[str] = None
        self._port_file_path: Optional[str] = None
        self._training_proc: Optional[Any] = None   # subprocess.Popen handle
        self._ipc_server_ref: Optional[Any] = None   # ViewerIPCServer backref

        self._last_present_t = 0.0
        # Slew: elapsed time controls how many scrub-ring entries are drained per pump().
        # Ring fill-depth drives target drain rate (fast when full, slow when empty).
        self._anim_frame_dt: float = 0.5
        self._anim_dt_slow: float = 0.5
        self._anim_dt_fast: float = 1.0 / 120.0
        self._anim_slew_tau: float = 0.25
        self._last_anim_t: float = 0.0
        self._last_slew_t: float = 0.0
        self._has_presented_frame = False
        self._top_bar_dirty = True
        self._panel_text_dirty = True

        self._caption = ""
        self._panel_titles = ["target", "input", "output"]
        self._panel_rows = [[], [], []]
        self._last_applied_frame_text: Dict[str, Any] = {
            "caption": "",
            "titles": list(self._panel_titles),
            "rows": [list(r) for r in self._panel_rows],
        }
        self._loss_display_rows: List[str] = []
        self._panel_text_rgb = [
            np.full((self.panel_h, self.panel_w, 3), 14, dtype=np.uint8),
            np.full((self.panel_h, self.panel_w, 3), 14, dtype=np.uint8),
            np.full((self.panel_h, self.panel_w, 3), 14, dtype=np.uint8),
        ]
        self._top_bar_rgb = np.full((self.top_bar_h, self.window_w, 3), 18, dtype=np.uint8)

        self._cycle_selected: List[bool] = []
        self._gate_override = False
        self._paused = False
        self._preview_enabled = True
        self._scrub_editor_enabled = True
        self._control_boxes: List[Tuple[str, int, Tuple[int, int, int, int]]] = []
        self._graph_control_boxes: List[Tuple[str, str, Tuple[int, int, int, int]]] = []
        self._graph_worker_hello: Dict[str, Any] = {}
        self._graph_plan_snapshot: Optional[Dict[str, Any]] = None
        self._graph_runtime_snapshot: Optional[Dict[str, Any]] = None
        self._graph_execution_events: deque = deque(maxlen=256)
        self._graph_x_axis_mode: str = "t"
        self._graph_history_mode: str = "recent"
        self.set_cycle_roster(total_cycles=int(cycle_slots))

        # The native C loss store is the single source of truth.
        # The viewer reads it directly -- no local deque copies.
        self._loss_store = loss_store  # NodusLossStore or None
        self._graph_dirty = False
        self._graph_rgb: Optional[np.ndarray] = (
            np.full((self.graph_total_h, self.window_w, 3), 14, dtype=np.uint8)
            if self.graph_total_h > 0 else None
        )

        # -- Shared-memory scrub ring + GUI-local composite cache --------------
        try:
            self._scrub_ring: Optional[NodusScrubRing] = NodusScrubRing.get_global()
        except Exception:
            self._scrub_ring = None
        self._composite_cache: Optional[NodusCompositeCache] = (
            NodusCompositeCache(capacity=512) if self._scrub_ring is not None else None
        )
        self._last_ring_cursor: int = 0
        # Text metadata for the most-recently consumed frame (set from IPC signal
        # or directly by same-process update()).
        self._ring_text_caption: str = ""
        self._ring_text_titles: List[str] = ["target", "input", "output"]
        self._ring_text_rows: List[List[str]] = [[], [], []]
        self._pending_frame_text_by_cursor: "OrderedDict[int, Dict[str, Any]]" = OrderedDict()
        _composite_text_capacity = (
            self._composite_cache.capacity() if self._composite_cache is not None else 512
        )
        self._composite_frame_text: deque = deque(maxlen=max(1, int(_composite_text_capacity)))
        self._composite_ring_cursors: deque = deque(maxlen=max(1, int(_composite_text_capacity)))

        # -- Sidebar state -----------------------------------------------------
        # -inf forces an immediate first render on the first _present() call.
        self._sidebar_dirty: bool = True
        self._preview_work_queue_ref: Optional[Any] = None
        self._weight_model_refs: Dict[str, Any] = {}
        # Optional per-model state-dicts for the red diff overlay:
        #   _weight_disk_states  - weights loaded from the on-disk checkpoint file
        #   _weight_ckpt_states  - weights from the pipeline checkpoint (to-be-merged)
        self._weight_disk_states: Dict[str, Optional[Dict[str, Any]]] = {}
        self._weight_ckpt_states: Dict[str, Optional[Dict[str, Any]]] = {}
        # -- Scrub / history ---------------------------------------------------
        # Weight-map snapshots accumulated at sidebar render rate (4 Hz).
        # _scrub_offset = 0 -> live; N -> show the snapshot N ticks back.
        self._weight_history_maxlen: int = 512
        self._weight_snapshot_deque: deque = deque(maxlen=self._weight_history_maxlen)
        # Loss series lengths recorded once per sidebar tick so the graph renderer
        # can map each snapshot -> x-pixel for the cache-region band and cursor.
        self._loss_count_at_snap_deque: deque = deque(maxlen=self._weight_history_maxlen)
        self._last_step_txt: str = ""
        self._scrub_offset: int = 0
        # Pipeline registers fn(scrub_offset) to handle "RESTORE STATE".
        self._on_restore_state: Optional[Callable] = None
        # Bounding box (x0,y0,x1,y1) in window coords for the restore button.
        self._restore_btn_window_rect: Optional[Tuple[int, int, int, int]] = None
        # Prev/Next checkpoint navigation button rects (window coords).
        self._prev_ckpt_btn_rect: Optional[Tuple[int, int, int, int]] = None
        self._next_ckpt_btn_rect: Optional[Tuple[int, int, int, int]] = None
        # Sparse weight-state snapshots: a small fixed count spread evenly across
        # the full visual cache so there are a couple of real restore points to
        # scrub to without copying enormous state_dicts constantly.
        # e.g. 4 snapshots across 512-frame cache -> stride 128 visual ticks
        #      @ 4 Hz sidebar rate ~= one snapshot every ~32 seconds.
        _weight_snap_count: int = 4
        self._weight_snap_stride: int = max(1, self._weight_history_maxlen // _weight_snap_count)
        self._weight_state_sparse_deque: deque = deque(maxlen=_weight_snap_count)
        self._snap_total: int = 0  # monotonically increasing visual snap counter
        # Directory for on-disk sparse weight backup files (set via set_weight_snap_dir).
        # Each sparse snapshot also saves a .pt file so restore works after restart.
        self._weight_snap_dir: Optional[Path] = None
        # Loss-count positions recorded at each disk save, for graph markers.
        # Unbounded list: accumulates all markers including pre-existing ones loaded at startup.
        self._disk_save_loss_counts: list = []
        # Checkpoint thumbnail registry -- ordered list of discovered checkpoint
        # entries beyond the in-memory cache.  Each entry is a dict with keys:
        #   round_id, cycle, thumb_path (Path or None), loss_counts ({sid:int})
        # Populated by register_checkpoint_thumbnail() and set_checkpoint_backup_dir().
        # Sorted oldest->newest (index 0 = oldest checkpoint).
        self._checkpoint_thumbs: List[Dict[str, Any]] = []
        self._checkpoint_thumb_cache: "OrderedDict[Tuple[int, int, str], np.ndarray]" = OrderedDict()
        self._checkpoint_thumb_pending: set[Tuple[int, int, str]] = set()
        self._checkpoint_thumb_results: deque = deque(maxlen=64)
        self._checkpoint_thumb_lock = threading.Lock()
        self._checkpoint_thumb_cache_limit: int = 48
        # Background thread for weight image + state_dict snapshot.
        # Main thread fires it and checks for completion; never blocks.
        self._weight_snap_thread: Optional[threading.Thread] = None
        self._weight_snap_result: Optional[Dict[str, Any]] = None
        # Which model to display in the right sidebar (None = first available).
        self._active_weight_model_name: Optional[str] = None
        # Delta logging: print param name when max|D| exceeds this threshold.
        self._weight_delta_threshold: float = 0.01
        # Previous sparse snapshot for delta comparison.
        self._prev_weight_snap: Dict[str, Dict[str, Any]] = {}

        # -- Pull-model data (populated by SaveRestoreNode responses) ------
        # Pending responses waiting to be processed.
        self._sr_response_queue: deque = deque(maxlen=256)
        # Channel registry from the last query_channel_list response.
        self._sr_channel_registry: List[Dict[str, Any]] = []
        # Interval between automatic data-pull queries (seconds).
        self._sr_poll_interval: float = 0.5
        self._sr_last_poll_t: float = 0.0
        # Weight-map data from SaveRestoreNode.
        self._sr_weight_models: List[str] = []
        self._sr_weight_active: Optional[str] = None
        # Cache browse data from SaveRestoreNode.
        self._sr_cache_entries: List[Dict[str, Any]] = []
        self._sr_cache_total: int = 0
        # Per-channel loss visibility (True = visible). Missing key -> visible.
        self._loss_channel_visible: Dict[str, bool] = {}
        # Legend hit-boxes for channel toggle clicks: list of (ck, x0, y0, x1, y1)
        # in *graph-local* pixel coords. Translated to window coords at click time.
        self._legend_hit_boxes: List[Tuple[str, int, int, int, int]] = []

        # Left sidebar: button panel (top) + scrub dial (bottom).
        # Right sidebar: weight map spans from top_bar to window bottom (covers graph row).
        # Fraction of the total loss history before which the graph is trimmed.
        # 0.0 = show everything; >0 = the graph's left edge starts here.
        # Set via trim_graph_to_first_checkpoint() after loading history + markers.
        self._graph_display_start_frac: float = 0.0

        self._cache_map_rgb   = np.full((self.panel_h, self.panel_w, 3), 14, dtype=np.uint8)
        self._frame_knob_rgb  = np.full((self.panel_h, self.panel_w, 3), 14, dtype=np.uint8)
        # weight_map is tall: fills the full right sidebar including the graph row, plus the
        # extra height strip below the graph (so the weight map extends further than the graph).
        _wm_h, _wm_w = self._weight_map_target_hw()
        self._weight_map_rgb  = np.full((_wm_h, _wm_w, 3), 14, dtype=np.uint8)

    def set_cycle_roster(self, total_cycles: int, selected: Optional[Sequence[bool]] = None):
        n = max(0, int(total_cycles))
        prev = list(self._cycle_selected)
        if n <= 0:
            self._cycle_selected = []
        else:
            out = [True] * n
            for i in range(min(len(prev), n)):
                out[i] = bool(prev[i])
            if selected is not None:
                for i, v in enumerate(list(selected)[:n]):
                    out[i] = bool(v)
            self._cycle_selected = out
        self._top_bar_dirty = True

    def selected_cycle_ids(self) -> List[int]:
        return [int(i + 1) for i, v in enumerate(self._cycle_selected) if bool(v)]

    def is_cycle_selected(self, cycle_local: int) -> bool:
        idx = int(cycle_local) - 1
        if idx < 0 or idx >= len(self._cycle_selected):
            return True
        return bool(self._cycle_selected[idx])

    def gate_override_enabled(self) -> bool:
        return bool(self._gate_override)

    def set_ipc_server(self, server: "ViewerIPCServer") -> None:
        """Store a back-reference to the IPC server for connection checks."""
        self._ipc_server_ref = server

    def set_launch_info(
        self,
        launch_script: Optional[str] = None,
        output_dir: Optional[str] = None,
        port_file_path: Optional[str] = None,
    ) -> None:
        if launch_script:
            self._launch_script = str(launch_script)
        if output_dir:
            self._output_dir = str(output_dir)
        if port_file_path:
            self._port_file_path = str(port_file_path)

    def has_training_connection(self) -> bool:
        srv = self._ipc_server_ref
        if srv is None:
            return False
        return getattr(srv, "has_connection", False)

    def shutdown_save(self) -> Optional[bool]:
        return self._shutdown_save

    def _start_training(self) -> None:
        """Launch the training bat/script as a subprocess."""
        import subprocess, os
        script = self._launch_script
        if not script:
            print("[viewer] no launch script configured; cannot start training", flush=True)
            return
        if self.has_training_connection():
            print("[viewer] training already connected; ignoring START", flush=True)
            return
        # Clear stop state so the next training run starts cleanly
        self._stop_requested = False
        self._shutdown_save = None
        self._top_bar_dirty = True
        try:
            if str(script).lower().endswith(".bat"):
                proc = subprocess.Popen(
                    ["cmd", "/c", script],
                    cwd=os.path.dirname(os.path.abspath(script)) or ".",
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                )
            else:
                import sys
                proc = subprocess.Popen(
                    [sys.executable, script],
                    cwd=os.path.dirname(os.path.abspath(script)) or ".",
                )
            self._training_proc = proc
            print(f"[viewer] launched training process (pid={proc.pid}): {script}", flush=True)
        except Exception as e:
            print(f"[viewer] failed to launch training: {e}", flush=True)

    def _init(self):
        if (not self.enabled) or self._ready or self._failed:
            return
        try:
            import pygame
            from OpenGL import GL

            pygame.display.init()
            pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
            pygame.display.set_mode((self.window_w, self.window_h), pygame.OPENGL | pygame.DOUBLEBUF)
            pygame.display.set_caption("Stage Status Viewer")

            GL.glViewport(0, 0, self.window_w, self.window_h)
            GL.glDisable(GL.GL_DEPTH_TEST)
            GL.glEnable(GL.GL_TEXTURE_2D)
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            GL.glClearColor(0.06, 0.06, 0.08, 1.0)

            tex = GL.glGenTextures(12)
            if isinstance(tex, int):
                tex = [int(tex)]
                while len(tex) < 12:
                    tex.append(int(GL.glGenTextures(1)))
            else:
                tex = [int(t) for t in list(tex)]
                while len(tex) < 12:
                    tex.append(int(GL.glGenTextures(1)))
            self._textures = {
                "img": [int(tex[0]), int(tex[1]), int(tex[2])],
                "text": [int(tex[3]), int(tex[4]), int(tex[5])],
                "bar": int(tex[6]),
                "graph": int(tex[7]),
                "cache_map":   int(tex[8]),
                "frame_knob":  int(tex[9]),
                "weight_knob": int(tex[10]),
                "weight_map":  int(tex[11]),
            }
            for tid in (
                self._textures["img"]
                + self._textures["text"]
                + [int(self._textures["bar"]),         int(self._textures["graph"]),
                   int(self._textures["cache_map"]),   int(self._textures["frame_knob"]),
                   int(self._textures["weight_knob"]), int(self._textures["weight_map"])]
            ):
                GL.glBindTexture(GL.GL_TEXTURE_2D, int(tid))
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)

            self._pygame = pygame
            self._gl = GL
            self._ready = True
            self._top_bar_dirty = True
            self._panel_text_dirty = True
            print(
                f"[transformer-viz] pygame+OpenGL ready ({self.window_w}x{self.window_h})",
                flush=True,
            )
        except Exception as e:
            self._failed = True
            self.enabled = False
            print(f"[transformer-viz] disabled: could not initialize pygame OpenGL viewer ({e})", flush=True)

    def _upload_texture(self, tex_id: int, img_rgb: np.ndarray):
        gl = self._gl
        h, w, _ = img_rgb.shape
        channels = int(img_rgb.shape[2]) if int(img_rgb.ndim) == 3 else 0
        if int(channels) == 4:
            internal_format = gl.GL_RGBA
            src_format = gl.GL_RGBA
        else:
            internal_format = gl.GL_RGB
            src_format = gl.GL_RGB
        gl.glBindTexture(gl.GL_TEXTURE_2D, int(tex_id))
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            internal_format,
            int(w),
            int(h),
            0,
            src_format,
            gl.GL_UNSIGNED_BYTE,
            np.ascontiguousarray(img_rgb),
        )

    def _resize_rgb_to_panel(self, img_rgb: np.ndarray) -> np.ndarray:
        h = int(img_rgb.shape[0])
        w = int(img_rgb.shape[1])
        if (h == int(self.panel_h)) and (w == int(self.panel_w)):
            return np.ascontiguousarray(img_rgb)
        t = torch.from_numpy(img_rgb.astype(np.float32, copy=False)).permute(2, 0, 1).unsqueeze(0)
        t = F.interpolate(t, size=(int(self.panel_h), int(self.panel_w)), mode="nearest")
        out = t.squeeze(0).permute(1, 2, 0).clamp(0.0, 255.0).to(torch.uint8).cpu().numpy()
        return np.ascontiguousarray(out)

    def _normalize_panel_text(
        self,
        panel_titles: Optional[Sequence[str]],
        panel_rows: Optional[Sequence[Sequence[str]]],
    ) -> Tuple[List[str], List[List[str]]]:
        titles = []
        rows = []
        for i in range(3):
            if panel_titles is not None and i < len(panel_titles):
                titles.append(str(panel_titles[i]))
            else:
                titles.append(self._panel_titles[i] if i < len(self._panel_titles) else "")
            cur_rows: List[str] = []
            if panel_rows is not None and i < len(panel_rows):
                src_rows = panel_rows[i]
                if isinstance(src_rows, (list, tuple)):
                    for r in src_rows:
                        txt = str(r).strip()
                        if txt:
                            cur_rows.append(txt)
            rows.append(cur_rows)
        return titles, rows

    def _render_panel_text(self, title: str, rows: Sequence[str]) -> np.ndarray:
        out = np.full((self.panel_h, self.panel_w, 3), 16, dtype=np.uint8)
        try:
            from PIL import Image, ImageDraw, ImageFont

            im = Image.fromarray(out).convert("RGB")
            draw = ImageDraw.Draw(im)
            font = ImageFont.load_default()

            draw.rectangle([(0, 0), (self.panel_w - 1, self.panel_h - 1)], fill=(20, 24, 30))
            draw.rectangle([(0, 0), (self.panel_w - 1, 15)], fill=(34, 42, 52))
            # Right-justify title
            _tw = font.getlength(str(title)) if hasattr(font, 'getlength') else len(str(title)) * 6
            draw.text((max(4, self.panel_w - 4 - int(_tw)), 2), str(title), fill=(255, 225, 70), font=font)
            draw.line([(0, 16), (self.panel_w - 1, 16)], fill=(70, 76, 88), width=1)
            y = 20
            for r in list(rows):
                txt = str(r)
                _rw = font.getlength(txt) if hasattr(font, "getlength") else len(txt) * 6
                draw.text((max(4, self.panel_w - 4 - int(_rw)), y), txt, fill=(230, 234, 240), font=font)
                y += 11
                if y >= (self.panel_h - 10):
                    break
            return np.asarray(im, dtype=np.uint8)
        except Exception:
            return out

    def _frame_text_meta(
        self,
        *,
        caption: str = "",
        titles: Optional[Sequence[str]] = None,
        rows: Optional[Sequence[Sequence[str]]] = None,
    ) -> Dict[str, Any]:
        norm_titles, norm_rows = self._normalize_panel_text(panel_titles=titles, panel_rows=rows)
        return {
            "caption": str(caption),
            "titles": list(norm_titles),
            "rows": [list(r) for r in norm_rows],
        }

    def _apply_frame_text_meta(self, meta: Optional[Dict[str, Any]]) -> None:
        if not isinstance(meta, dict):
            return
        new_caption = str(meta.get("caption", ""))
        new_titles = [str(x) for x in meta.get("titles", self._panel_titles)]
        new_rows = [list(r) for r in meta.get("rows", self._panel_rows)]
        self._ring_text_caption = new_caption
        self._ring_text_titles = list(new_titles)
        self._ring_text_rows = [list(r) for r in new_rows]
        self._last_applied_frame_text = {
            "caption": new_caption,
            "titles": list(new_titles),
            "rows": [list(r) for r in new_rows],
        }
        if new_titles != self._panel_titles or new_rows != self._panel_rows:
            self._panel_titles = new_titles
            self._panel_rows = new_rows
            self._panel_text_dirty = True
        if new_caption != self._caption:
            self._caption = new_caption
            self._top_bar_dirty = True

    def _clear_frame_text_cache(self) -> None:
        self._pending_frame_text_by_cursor.clear()
        self._composite_frame_text.clear()
        self._composite_ring_cursors.clear()

    def _prune_pending_frame_text(self, *, min_cursor: Optional[int] = None) -> None:
        if min_cursor is not None:
            while self._pending_frame_text_by_cursor:
                first_cursor = next(iter(self._pending_frame_text_by_cursor))
                if int(first_cursor) >= int(min_cursor):
                    break
                self._pending_frame_text_by_cursor.popitem(last=False)
        max_pending = max(
            64,
            (
                int(self._scrub_ring.capacity()) if self._scrub_ring is not None else 1536
            ) * 2,
        )
        while len(self._pending_frame_text_by_cursor) > max_pending:
            self._pending_frame_text_by_cursor.popitem(last=False)

    def _current_display_cache_index(self) -> Optional[int]:
        cache = self._composite_cache
        if cache is None:
            return None
        clen = cache.length()
        if clen <= 0:
            return None
        if self._scrub_offset == 0:
            return clen - 1
        if self._scrub_in_checkpoint_zone() and self._scrub_offset > self._cache_snap_count():
            return None
        off = max(1, min(clen, int(self._scrub_offset)))
        return max(0, min(clen - 1, clen - off))

    def _stage_frame_text(
        self,
        ring_cursor: Optional[int],
        *,
        caption: str = "",
        titles: Optional[Sequence[str]] = None,
        rows: Optional[Sequence[Sequence[str]]] = None,
    ) -> None:
        meta = self._frame_text_meta(caption=caption, titles=titles, rows=rows)
        if ring_cursor is None:
            self._apply_frame_text_meta(meta)
            return
        try:
            cursor = int(ring_cursor)
        except Exception:
            self._apply_frame_text_meta(meta)
            return
        if cursor < 0:
            self._apply_frame_text_meta(meta)
            return
        try:
            cache_idx = list(self._composite_ring_cursors).index(cursor)
        except ValueError:
            cache_idx = -1
        if cache_idx >= 0 and cache_idx < len(self._composite_frame_text):
            self._composite_frame_text[cache_idx] = meta
            if self._current_display_cache_index() == cache_idx:
                self._apply_frame_text_meta(meta)
            return
        self._pending_frame_text_by_cursor[cursor] = meta
        self._prune_pending_frame_text()

    def _cache_frame_text_for_cursor(self, ring_cursor: int) -> None:
        cursor = int(ring_cursor)
        meta = self._pending_frame_text_by_cursor.pop(cursor, None)
        if not isinstance(meta, dict):
            meta = {
                "caption": str(self._last_applied_frame_text.get("caption", "")),
                "titles": [str(x) for x in self._last_applied_frame_text.get("titles", self._panel_titles)],
                "rows": [list(r) for r in self._last_applied_frame_text.get("rows", self._panel_rows)],
            }
        self._composite_ring_cursors.append(cursor)
        self._composite_frame_text.append(meta)
        self._prune_pending_frame_text(min_cursor=cursor)

    def _apply_cached_frame_text(self, cache_index: int) -> None:
        if cache_index < 0 or cache_index >= len(self._composite_frame_text):
            return
        self._apply_frame_text_meta(self._composite_frame_text[cache_index])

    def _render_top_bar(self) -> np.ndarray:
        out = np.full((self.top_bar_h, self.window_w, 3), 18, dtype=np.uint8)
        self._control_boxes = []
        try:
            from PIL import Image, ImageDraw, ImageFont

            im = Image.fromarray(out).convert("RGB")
            draw = ImageDraw.Draw(im)
            font = ImageFont.load_default()

            def _draw_check(
                x: int,
                y: int,
                label: str,
                *,
                checked: bool,
                kind: str,
                idx: int = -1,
                fill_on: Tuple[int, int, int] = (52, 120, 66),
                fill_off: Tuple[int, int, int] = (36, 40, 44),
                text_on: Tuple[int, int, int] = (224, 230, 236),
                text_off: Tuple[int, int, int] = (150, 160, 172),
                outline: Tuple[int, int, int] = (186, 194, 204),
                enabled: bool = True,
            ) -> int:
                box = (int(x), int(y), int(x + 11), int(y + 11))
                cur_outline = outline if enabled else (94, 98, 108)
                cur_fill = fill_on if checked and enabled else (fill_off if enabled else (48, 52, 58))
                draw.rectangle([box[0], box[1], box[2], box[3]], outline=cur_outline, fill=cur_fill)
                if checked and enabled:
                    draw.line([(box[0] + 2, box[1] + 6), (box[0] + 5, box[1] + 9)], fill=(236, 244, 248), width=1)
                    draw.line([(box[0] + 5, box[1] + 9), (box[0] + 9, box[1] + 2)], fill=(236, 244, 248), width=1)
                draw.text((x + 15, y - 1), label, fill=(text_on if checked and enabled else text_off), font=font)
                self._control_boxes.append((kind, int(idx), box))
                return int(x + 15 + (len(label) * 7) + 10)

            draw.rectangle([(0, 0), (self.window_w - 1, self.top_bar_h - 1)], fill=(18, 22, 28))
            draw.line([(0, 17), (self.window_w - 1, 17)], fill=(42, 48, 58), width=1)
            draw.line([(0, 37), (self.window_w - 1, 37)], fill=(34, 40, 48), width=1)
            draw.line(
                [(0, self.top_bar_h - 1), (self.window_w - 1, self.top_bar_h - 1)],
                fill=(70, 76, 88),
                width=1,
            )
            cap = str(self._caption).strip()
            if cap:
                draw.text((6, 4), cap[: max(16, (self.window_w // 6) - 4)], fill=(230, 234, 240), font=font)

            connected = self.has_training_connection()
            stopping = self._shutdown_save is not None
            btn_y = 22
            btn_h = 13
            btn_right_margin = 6
            bx = self.window_w - btn_right_margin

            if connected and stopping:
                label_pending = "STOPPING..."
                lw_p = len(label_pending) * 7 + 10
                p_box = (bx - lw_p, btn_y, bx, btn_y + btn_h)
                draw.rectangle(
                    [p_box[0], p_box[1], p_box[2], p_box[3]],
                    outline=(120, 120, 60), fill=(80, 80, 30),
                )
                draw.text((p_box[0] + 5, btn_y + 1), label_pending, fill=(220, 220, 160), font=font)
            elif connected:
                label_stop = "STOP"
                lw_stop = len(label_stop) * 7 + 10
                stop_box = (bx - lw_stop, btn_y, bx, btn_y + btn_h)
                draw.rectangle(
                    [stop_box[0], stop_box[1], stop_box[2], stop_box[3]],
                    outline=(180, 60, 60), fill=(120, 36, 36),
                )
                draw.text((stop_box[0] + 5, btn_y + 1), label_stop, fill=(240, 200, 200), font=font)
                self._control_boxes.append(("stop_nosave", -1, stop_box))
                bx = stop_box[0] - 6

                label_save = "STOP+SAVE"
                lw_save = len(label_save) * 7 + 10
                save_box = (bx - lw_save, btn_y, bx, btn_y + btn_h)
                draw.rectangle(
                    [save_box[0], save_box[1], save_box[2], save_box[3]],
                    outline=(60, 160, 80), fill=(36, 100, 50),
                )
                draw.text((save_box[0] + 5, btn_y + 1), label_save, fill=(200, 240, 210), font=font)
                self._control_boxes.append(("stop_save", -1, save_box))
            elif self._launch_script:
                label_start = "START"
                lw_start = len(label_start) * 7 + 10
                start_box = (bx - lw_start, btn_y, bx, btn_y + btn_h)
                draw.rectangle(
                    [start_box[0], start_box[1], start_box[2], start_box[3]],
                    outline=(60, 100, 180), fill=(36, 64, 130),
                )
                draw.text((start_box[0] + 5, btn_y + 1), label_start, fill=(200, 220, 250), font=font)
                self._control_boxes.append(("start", -1, start_box))

            tx = 8
            row1_y = 22
            tx = _draw_check(
                tx,
                row1_y,
                "PAUSED" if self._paused else "PLAY",
                checked=bool(self._paused),
                kind="pause",
                fill_on=(180, 140, 40),
                fill_off=(36, 40, 44),
                text_on=(230, 220, 140),
                text_off=(140, 200, 140),
            )
            tx = _draw_check(
                tx,
                row1_y,
                "Preview",
                checked=bool(self._preview_enabled),
                kind="preview",
                fill_on=(52, 120, 66),
                fill_off=(120, 36, 36),
                text_on=(140, 200, 140),
                text_off=(200, 120, 120),
            )
            _draw_check(
                tx,
                row1_y,
                "Scrub Editor",
                checked=bool(self._scrub_editor_enabled),
                kind="scrub_editor",
                fill_on=(52, 120, 66),
                fill_off=(120, 36, 36),
                text_on=(140, 200, 140),
                text_off=(200, 120, 120),
                enabled=bool(self._preview_enabled),
            )

            row2_y = 42
            x = 8
            for i, is_on in enumerate(self._cycle_selected):
                token_w = 44
                if x + token_w >= (self.window_w - 175):
                    break
                x = _draw_check(
                    x,
                    row2_y,
                    f"C{i + 1}",
                    checked=bool(is_on),
                    kind="cycle",
                    idx=int(i),
                    fill_on=(52, 120, 66),
                    fill_off=(36, 40, 44),
                    text_on=(224, 230, 236),
                    text_off=(150, 160, 172),
                )

            _draw_check(
                max(x + 4, self.window_w - 166),
                row2_y,
                "Override gates",
                checked=bool(self._gate_override),
                kind="override",
                fill_on=(126, 84, 36),
                fill_off=(36, 40, 44),
                text_on=(230, 208, 170),
                text_off=(170, 160, 140),
            )

            active = self.selected_cycle_ids()
            active_txt = ",".join(str(i) for i in active) if len(active) > 0 else "none"
            draw.text(
                (6, self.top_bar_h - 14),
                f"cycles={active_txt} gate_override={1 if self._gate_override else 0}"
                f" preview={'on' if self._preview_enabled else 'off'}"
                f" scrub={'on' if self._scrub_editor_enabled else 'off'}"
                f"{' PAUSED' if self._paused else ''}",
                fill=(180, 188, 198),
                font=font,
            )
            if isinstance(self._graph_plan_snapshot, dict):
                node_count = len(self._graph_plan_snapshot.get("nodes", []))
                edge_count = len(self._graph_plan_snapshot.get("edges", []))
                plan_id = str(self._graph_plan_snapshot.get("plan_id", ""))[:18]
                draw.text(
                    (max(6, self.window_w - 360), 4),
                    f"plan={plan_id or 'none'} nodes={node_count} edges={edge_count}",
                    fill=(186, 206, 218),
                    font=font,
                )
            if isinstance(self._graph_runtime_snapshot, dict):
                exec_state = str(self._graph_runtime_snapshot.get("execution_state", ""))
                active_count = len(self._graph_runtime_snapshot.get("active_node_ids", []))
                draw.text(
                    (max(6, self.window_w - 360), self.top_bar_h - 14),
                    f"graph={exec_state or 'idle'} active={active_count}",
                    fill=(208, 214, 194),
                    font=font,
                )
            return np.asarray(im, dtype=np.uint8)
        except Exception:
            return out

    # -- Checkpoint navigation helpers ------------------------------------------

    def _scrub_offset_for_checkpoint(self, direction: int) -> Optional[int]:
        """Find the scrub offset for the next (direction=+1) or previous (-1) checkpoint.

        Checkpoint markers in ``_disk_save_loss_counts`` record per-sid loss-count
        at each disk save.  The ``_loss_count_at_snap_deque`` records the same per
        sidebar tick.  We find the tick whose loss-count is closest to a checkpoint
        marker, then translate that tick index to a scrub offset.

        Returns ``None`` if there is no checkpoint to jump to in that direction.
        """
        markers = self._disk_save_loss_counts
        if not markers:
            return None
        snap_list = list(self._loss_count_at_snap_deque)
        slen = len(snap_list)
        if slen == 0:
            return None

        # Identify a reference sid (the one with the most data).
        _lengths = self._loss_channel_lengths()
        ref_sid = max(_lengths, key=_lengths.get, default=None)
        if ref_sid is None:
            return None

        # Current position in loss-count space.
        cur_off = int(self._scrub_offset)
        if cur_off > 0 and cur_off <= slen:
            cur_idx = slen - cur_off
            cur_loss_n = int(snap_list[cur_idx].get(ref_sid, 0))
        else:
            # Live position = total length
            cur_loss_n = _lengths.get(ref_sid, 0)

        # Collect unique checkpoint loss-counts and sort them.
        ckpt_positions: List[int] = []
        for m in markers:
            v = m.get(ref_sid)
            if v is not None:
                ckpt_positions.append(int(v))
        if not ckpt_positions:
            return None
        ckpt_positions = sorted(set(ckpt_positions))

        # Find target checkpoint loss-count.
        target_n: Optional[int] = None
        if direction < 0:  # previous (earlier in time = smaller loss count)
            for cn in reversed(ckpt_positions):
                if cn < cur_loss_n - 1:
                    target_n = cn
                    break
        else:  # next (later in time = larger loss count)
            for cn in ckpt_positions:
                if cn > cur_loss_n + 1:
                    target_n = cn
                    break

        if target_n is None:
            return None

        # Find the snap index whose loss-count for ref_sid is closest to target_n.
        best_idx = 0
        best_dist = abs(int(snap_list[0].get(ref_sid, 0)) - target_n)
        for i, sc in enumerate(snap_list):
            d = abs(int(sc.get(ref_sid, 0)) - target_n)
            if d < best_dist:
                best_dist = d
                best_idx = i
        # Scrub offset: slen - idx  (offset 1 = newest; slen = oldest).
        return max(1, slen - best_idx)

    # -- Checkpoint thumbnail registry ------------------------------------------

    def register_checkpoint_thumbnail(
        self,
        round_id: int,
        cycle: int,
        thumb_path: Optional["Path"] = None,
        loss_counts: Optional[Dict[str, int]] = None,
        model_name: Optional[str] = None,
        checkpoint_path: Optional["Path"] = None,
    ) -> None:
        """Register a checkpoint entry for extended scrub navigation.

        Called at checkpoint-save time (via notify) and at startup when
        discovering existing checkpoints on disk.  Entries are kept sorted
        oldest->newest by (round_id, cycle).
        """
        # Insert in sorted order
        idx = len(self._checkpoint_thumbs)
        for i, e in enumerate(self._checkpoint_thumbs):
            if (e["round_id"], e["cycle"]) > (round_id, cycle):
                idx = i
                break
            if e["round_id"] == round_id and e["cycle"] == cycle:
                merged = dict(e)
                if thumb_path is not None:
                    merged["thumb_path"] = str(thumb_path)
                if checkpoint_path is not None:
                    merged["checkpoint_path"] = str(checkpoint_path)
                if loss_counts:
                    merged["loss_counts"] = dict(loss_counts)
                if model_name:
                    merged["model_name"] = str(model_name)
                self._checkpoint_thumbs[i] = merged
                return
        self._checkpoint_thumbs.insert(
            idx,
            {
                "round_id": int(round_id),
                "cycle": int(cycle),
                "thumb_path": (str(thumb_path) if thumb_path is not None else None),
                "checkpoint_path": (str(checkpoint_path) if checkpoint_path is not None else None),
                "loss_counts": dict(loss_counts or {}),
                "model_name": (str(model_name) if model_name else None),
            },
        )

    def _weight_map_target_hw(self) -> Tuple[int, int]:
        return (2 * self.panel_h + self.graph_total_h + self._weight_map_extra_h, self.panel_w)

    def _checkpoint_cache_key(self, entry: Dict[str, Any], model_name: Optional[str]) -> Tuple[int, int, str]:
        return (
            int(entry.get("round_id", 0)),
            int(entry.get("cycle", 0)),
            str(model_name or "").strip(),
        )

    def _target_checkpoint_model_name(self, entry: Dict[str, Any]) -> Optional[str]:
        name = str(self._active_weight_model_name or entry.get("model_name") or "").strip()
        return name or None

    def _remember_checkpoint_thumbnail(self, cache_key: Tuple[int, int, str], rgb: np.ndarray) -> np.ndarray:
        arr = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
        self._checkpoint_thumb_cache[cache_key] = arr
        self._checkpoint_thumb_cache.move_to_end(cache_key)
        while len(self._checkpoint_thumb_cache) > self._checkpoint_thumb_cache_limit:
            self._checkpoint_thumb_cache.popitem(last=False)
        return arr

    def _queue_checkpoint_thumbnail_render(self, entry: Dict[str, Any], model_name: Optional[str]) -> None:
        checkpoint_path = entry.get("checkpoint_path")
        if checkpoint_path is None or not Path(checkpoint_path).exists():
            return
        cache_key = self._checkpoint_cache_key(entry, model_name)
        with self._checkpoint_thumb_lock:
            if cache_key in self._checkpoint_thumb_pending:
                return
            self._checkpoint_thumb_pending.add(cache_key)
        target_hw = self._weight_map_target_hw()
        round_id = int(entry.get("round_id", 0))
        cycle = int(entry.get("cycle", 0))
        thumb_path = entry.get("thumb_path")
        if not thumb_path:
            thumb_path = str(Path(checkpoint_path).with_name(f"weight_thumb_r{round_id:06d}_c{cycle:04d}.png"))

        def _worker() -> None:
            result: Dict[str, Any] = {
                "cache_key": cache_key,
                "round_id": round_id,
                "cycle": cycle,
                "thumb_path": str(thumb_path),
            }
            try:
                payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                picked_model, state = _extract_checkpoint_weight_state(payload, model_name)
                if state is None:
                    raise ValueError(f"no usable *_state payload in {checkpoint_path}")
                _tgt_h, _tgt_w = target_hw
                rgb, _meta = render_weight_image(
                    state,
                    target_width=_tgt_w,
                    target_height=_tgt_h,
                )
                rgb = annotate_weight_map(
                    rgb,
                    title=f"{picked_model or 'weights'} r{round_id}",
                    subtitle="",
                )
                out_path = Path(thumb_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                from PIL import Image

                Image.fromarray(rgb, mode="RGB").save(out_path, format="PNG")
                out_path.with_suffix(".json").write_text(
                    json.dumps(
                        {
                            "model": str(picked_model or ""),
                            "round_id": int(round_id),
                            "cycle": int(cycle),
                            "mode": "architectural_tall",
                            "width": int(rgb.shape[1]),
                            "height": int(rgb.shape[0]),
                            "checkpoint_path": str(checkpoint_path),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                result["model_name"] = str(picked_model or "")
                result["rgb"] = rgb
            except Exception as exc:
                result["error"] = str(exc)
            finally:
                with self._checkpoint_thumb_lock:
                    self._checkpoint_thumb_results.append(result)
                    self._checkpoint_thumb_pending.discard(cache_key)

        threading.Thread(target=_worker, daemon=True).start()

    def _drain_checkpoint_thumbnail_results(self) -> None:
        changed = False
        while True:
            with self._checkpoint_thumb_lock:
                if not self._checkpoint_thumb_results:
                    break
                result = self._checkpoint_thumb_results.popleft()
            rgb = result.get("rgb")
            if rgb is not None:
                self._remember_checkpoint_thumbnail(result["cache_key"], rgb)
                for entry in self._checkpoint_thumbs:
                    if (
                        int(entry.get("round_id", 0)) == int(result.get("round_id", -1))
                        and int(entry.get("cycle", 0)) == int(result.get("cycle", -1))
                    ):
                        entry["thumb_path"] = str(result.get("thumb_path") or entry.get("thumb_path") or "")
                        if result.get("model_name"):
                            entry["model_name"] = str(result["model_name"])
                        break
                changed = True
        if changed:
            self._sidebar_dirty = True

    def _load_checkpoint_thumbnail(self, ckpt_idx: int) -> Optional[np.ndarray]:
        """Load a checkpoint thumbnail image and scale to the weight-map panel size.

        Returns a right-panel-sized RGB array.
        """
        if ckpt_idx < 0 or ckpt_idx >= len(self._checkpoint_thumbs):
            return None
        entry = self._checkpoint_thumbs[ckpt_idx]
        target_model = self._target_checkpoint_model_name(entry)
        cache_key = self._checkpoint_cache_key(entry, target_model)
        cached = self._checkpoint_thumb_cache.get(cache_key)
        if cached is not None:
            self._checkpoint_thumb_cache.move_to_end(cache_key)
            return np.ascontiguousarray(cached.copy())
        thumb_path = entry.get("thumb_path")
        thumb_model = str(entry.get("model_name") or "").strip()
        thumb_mode = ""
        target_h, target_w = self._weight_map_target_hw()
        if thumb_path is None or not Path(thumb_path).exists():
            self._queue_checkpoint_thumbnail_render(entry, target_model)
            return self._render_checkpoint_placeholder(entry, text="rendering thumbnail")
        sidecar = Path(thumb_path).with_suffix(".json")
        if sidecar.exists():
            try:
                side_meta = json.loads(sidecar.read_text(encoding="utf-8"))
                thumb_model = str(side_meta.get("model") or thumb_model).strip()
                thumb_mode = str(side_meta.get("mode") or "").strip().lower()
            except Exception:
                pass
        if thumb_mode != "architectural_tall":
            self._queue_checkpoint_thumbnail_render(entry, target_model)
            return self._render_checkpoint_placeholder(entry, text="refreshing thumbnail")
        if target_model and thumb_model and thumb_model != target_model:
            self._queue_checkpoint_thumbnail_render(entry, target_model)
            return self._render_checkpoint_placeholder(entry, text="rendering thumbnail")
        try:
            from PIL import Image
            img = Image.open(str(thumb_path)).convert("RGB").resize((target_w, target_h), Image.NEAREST)
            rgb = np.asarray(img, dtype=np.uint8)
            if thumb_model:
                entry["model_name"] = thumb_model
            return self._remember_checkpoint_thumbnail(cache_key, rgb)
        except Exception:
            self._queue_checkpoint_thumbnail_render(entry, target_model)
            return self._render_checkpoint_placeholder(entry, text="render failed")

    def _render_checkpoint_placeholder(self, entry: Dict[str, Any], *, text: str = "no thumbnail") -> np.ndarray:
        """Render a placeholder when no thumbnail file exists."""
        H, W = self._weight_map_target_hw()
        rgb = np.full((H, W, 3), (14, 16, 20), dtype=np.uint8)
        try:
            from PIL import Image, ImageDraw, ImageFont
            im = Image.fromarray(rgb)
            draw = ImageDraw.Draw(im)
            font = ImageFont.load_default()
            r = entry.get("round_id", "?")
            c = entry.get("cycle", "?")
            draw.text((4, 4), f"CKPT r{r} c{c}", fill=(200, 170, 40), font=font)
            draw.text((4, 20), str(text), fill=(80, 80, 100), font=font)
            rgb = np.asarray(im, dtype=np.uint8)
        except Exception:
            pass
        return rgb

    # -- Sidebar public API -----------------------------------------------------

    def set_queue_refs(self, work_queue: Any) -> None:
        """Register the preview work queue so the cache map can read its depth."""
        self._preview_work_queue_ref = work_queue

    def _history_snap_count(self) -> int:
        """Total scrub positions: in-memory cache + on-disk checkpoint thumbnails."""
        cc_len = self._composite_cache.length() if self._composite_cache else 0
        cache_len = max(
            len(self._weight_snapshot_deque),
            cc_len,
            len(self._loss_count_at_snap_deque),
        )
        return cache_len + len(self._checkpoint_thumbs)

    def _cache_snap_count(self) -> int:
        """In-memory cache positions only (without checkpoint thumbnails)."""
        cc_len = self._composite_cache.length() if self._composite_cache else 0
        return max(
            len(self._weight_snapshot_deque),
            cc_len,
            len(self._loss_count_at_snap_deque),
        )

    def _scrub_in_checkpoint_zone(self) -> bool:
        """True when the current scrub offset is beyond in-memory cache."""
        return self._scrub_offset > self._cache_snap_count()

    def _checkpoint_index_from_offset(self, offset: int) -> Optional[int]:
        """Map a scrub offset in the checkpoint zone to a _checkpoint_thumbs index.

        Checkpoint zone offsets start at cache_len + 1.
        Returns index into _checkpoint_thumbs (newest first) or None.
        """
        cache_len = self._cache_snap_count()
        if offset <= cache_len:
            return None
        # ckpt_offset 1 = most-recent checkpoint, N = oldest
        ckpt_offset = offset - cache_len
        n = len(self._checkpoint_thumbs)
        if n == 0 or ckpt_offset > n:
            return None
        # _checkpoint_thumbs is sorted oldest->newest, so index for newest-first:
        return n - ckpt_offset

    def _ring_index_from_cursor(self, ring_cursor: int) -> Optional[int]:
        ring = self._scrub_ring
        if ring is None:
            return None
        with ring.locked():
            write_cursor = int(ring.write_cursor())
            ring_len = int(ring.length())
        first_cursor = write_cursor - ring_len
        cursor = int(ring_cursor)
        if cursor < first_cursor or cursor >= write_cursor:
            return None
        return cursor - first_cursor

    def _weight_map_from_ring_cursor(self, ring_cursor: Optional[int]) -> Optional[np.ndarray]:
        if ring_cursor is None:
            return None
        ring_idx = self._ring_index_from_cursor(int(ring_cursor))
        if ring_idx is None:
            return None
        ring = self._scrub_ring
        if ring is None:
            return None
        meta = ring.get_meta(ring_idx)
        if meta is None or not (int(meta.flags) & int(SCRUB_FLAG_HAS_THUMBS)):
            return None
        tiles: List[np.ndarray] = []
        for thumb_idx in range(SCRUB_NUM_THUMBS):
            tile = ring.copy_thumbnail(ring_idx, thumb_idx)
            if tile is None:
                return None
            tiles.append(tile)
        rgb = _compose_weight_thumb_stack(tiles)
        title = str(self._active_weight_model_name or "").strip()
        if title:
            rgb = annotate_weight_map(rgb, title=title, subtitle="")
        return rgb

    def _checkpoint_index_for_loss_counts(self, loss_counts: Dict[str, int]) -> Optional[int]:
        if not isinstance(loss_counts, dict) or not self._checkpoint_thumbs:
            return None
        lengths = self._loss_channel_lengths()
        ref_sid = max(lengths, key=lengths.get, default=None)
        if ref_sid is None or ref_sid not in loss_counts:
            return None
        cursor_count = int(loss_counts.get(ref_sid, 0))
        best_idx: Optional[int] = None
        best_count = -1
        for idx, entry in enumerate(self._checkpoint_thumbs):
            ckpt_counts = entry.get("loss_counts") if isinstance(entry, dict) else None
            if not isinstance(ckpt_counts, dict) or ref_sid not in ckpt_counts:
                continue
            ckpt_count = int(ckpt_counts.get(ref_sid, 0))
            if ckpt_count <= cursor_count and ckpt_count >= best_count:
                best_idx = idx
                best_count = ckpt_count
        return best_idx

    def _record_history_snap(self, ring_cursor: Optional[int] = None) -> None:
        """Record a weight-map snapshot and loss-count position for the latest
        composite-cache entry.  Called once per composite built in _present()."""
        self._loss_count_at_snap_deque.append(
            self._loss_channel_lengths()
        )
        self._weight_snapshot_deque.append(
            {
                "ring_cursor": (int(ring_cursor) if ring_cursor is not None else None),
                "weight_map": np.ascontiguousarray(np.asarray(self._weight_map_rgb, dtype=np.uint8)).copy(),
            }
        )

    def _apply_composite_to_display(self, cache_index: int) -> None:
        """Upload the three RGB panels from the composite cache to OpenGL textures
        and apply the current ring text metadata."""
        cache = self._composite_cache
        if cache is None:
            return
        panels = cache.copy_all_panels(cache_index)
        if panels is None:
            return
        if self._textures is not None:
            for i, tid in enumerate(self._textures["img"]):
                self._upload_texture(int(tid), panels[i])
        self._apply_cached_frame_text(int(cache_index))

    def _apply_composite_at_offset(self, offset: int) -> None:
        """Display the composite cache entry at the given scrub offset."""
        cache = self._composite_cache
        if cache is None:
            return
        # If in checkpoint zone, no preview frame is available.
        if self._scrub_in_checkpoint_zone() and offset > self._cache_snap_count():
            return
        clen = cache.length()
        if clen <= 0:
            return
        off = max(1, min(clen, int(offset)))
        idx = max(0, min(clen - 1, clen - off))
        panels = cache.copy_all_panels(idx)
        if panels is None:
            return
        if self._textures is not None:
            for i, tid in enumerate(self._textures["img"]):
                self._upload_texture(int(tid), panels[i])
        self._apply_cached_frame_text(idx)


    def _history_weight_map_at_offset(self, offset: int) -> Optional[np.ndarray]:
        # Check checkpoint zone first
        ckpt_idx = self._checkpoint_index_from_offset(offset)
        if ckpt_idx is not None:
            return self._load_checkpoint_thumbnail(ckpt_idx)
        snaps = list(self._weight_snapshot_deque)
        slen = len(snaps)
        if slen <= 0:
            return None
        off = max(1, min(slen, int(offset)))
        idx = max(0, min(slen - 1, slen - off))
        blob = snaps[idx]
        if not isinstance(blob, dict):
            return None
        ring_rgb = self._weight_map_from_ring_cursor(blob.get("ring_cursor"))
        if ring_rgb is not None:
            return ring_rgb
        counts = list(self._loss_count_at_snap_deque)
        if 0 <= idx < len(counts):
            hist_ckpt_idx = self._checkpoint_index_for_loss_counts(counts[idx])
            if hist_ckpt_idx is not None:
                hist_ckpt = self._load_checkpoint_thumbnail(hist_ckpt_idx)
                if hist_ckpt is not None:
                    return hist_ckpt
        weight_map = blob.get("weight_map", None)
        if weight_map is None:
            return None
        return np.ascontiguousarray(np.asarray(weight_map, dtype=np.uint8))

    def set_training_graph_worker_hello(self, payload: Dict[str, Any]) -> None:
        self._graph_worker_hello = dict(payload or {})
        self._top_bar_dirty = True

    def set_training_graph_plan(self, payload: Dict[str, Any]) -> None:
        self._graph_plan_snapshot = dict(payload or {})
        self._top_bar_dirty = True
        self._graph_dirty = True

    def set_training_graph_runtime(self, payload: Dict[str, Any]) -> None:
        self._graph_runtime_snapshot = dict(payload or {})
        self._top_bar_dirty = True
        self._graph_dirty = True

    def append_training_graph_event(self, payload: Dict[str, Any]) -> None:
        self._graph_execution_events.append(dict(payload or {}))
        self._top_bar_dirty = True
        self._graph_dirty = True

    def set_weight_model_refs(
        self,
        models: Dict[str, Any],
        disk_states: Optional[Dict[str, Optional[Dict[str, Any]]]] = None,
        ckpt_states: Optional[Dict[str, Optional[Dict[str, Any]]]] = None,
    ) -> None:
        """Register the *currently active* nn.Module instances for weight-map display.

        Pass only models that are actually being run right now - the map shows
        exactly what is given.  Optionally supply state-dicts for the red diff
        overlay:
          disk_states  - {name: state_dict} loaded straight from the on-disk file
          ckpt_states  - {name: state_dict} from the pipeline checkpoint that
                         holds data to be integrated with the base model
        The red overlay pixel intensity = normalised |disk_weight - ckpt_weight|.
        """
        self._weight_model_refs = dict(models)
        self._weight_disk_states = dict(disk_states) if disk_states else {}
        self._weight_ckpt_states = dict(ckpt_states) if ckpt_states else {}
        self._sidebar_dirty = True

    def set_restore_state_callback(self, fn: "Callable[[int], None]") -> None:
        """Register a callback invoked when the user clicks RESTORE STATE.

        The callback receives the current scrub offset (positive integer = that
        many 4-Hz sidebar ticks in the past).  The pipeline should roll the model
        weights back to that checkpoint, invalidate any stale forward-cache data,
        and continue training from the restored state.  The viewer resets
        scrub_offset to 0 and clears the snapshot deque after invoking the callback.
        """
        self._on_restore_state = fn

    def set_active_weight_model(self, name: Optional[str]) -> None:
        """Set which model name is shown in the right sidebar weight image.

        Pass ``None`` to fall back to the first key in ``_weight_model_refs``.
        """
        self._active_weight_model_name = name if name is None else str(name)
        self._sidebar_dirty = True

    def set_weight_snap_dir(self, path) -> None:
        """Set the directory where per-snapshot weight .pt backup files are written.

        Files are named ``snap_{snap_total:08d}.pt`` and deleted automatically
        when the corresponding entry ages out of the sparse deque.
        """
        self._weight_snap_dir = Path(path) if path is not None else None

    def get_restore_state_dicts(self, offset: int) -> Optional[Dict[str, Dict]]:
        """Return the weight state-dicts for the snapshot at *offset* positions back.

        offset=1 -> most recent snapshot; offset=N -> Nth most recent (oldest = N where
        N == len(sparse deque)).  Mirrors the same indexing used by the scrub display.
        Returns ``None`` if the sparse deque is empty."""
        snaps = list(self._weight_state_sparse_deque)
        slen = len(snaps)
        if not snaps:
            return None
        off = max(1, min(slen, int(offset)))
        idx = max(0, min(slen - 1, slen - off))
        return snaps[idx]["states"]

    # -- Sidebar render methods -------------------------------------------------

    def _render_cache_map(self) -> np.ndarray:
        """Pixel-grid occupancy map: work queue (green) + frame buffer (blue)."""
        H, W = self.panel_h, self.panel_w
        out = np.full((H, W, 3), np.array([14, 16, 20], dtype=np.uint8), dtype=np.uint8)
        wq_cap  = 1024
        wq_used = int(self._preview_work_queue_ref.qsize()) if self._preview_work_queue_ref is not None else 0
        fb_cap  = (self._scrub_ring.capacity() if self._scrub_ring else 1536)
        fb_used = (self._scrub_ring.length() if self._scrub_ring else 0)
        margin  = 3
        label_h = 12
        section_h = max(1, (H - 2 * label_h - 3 * margin) // 2)
        grid_w = max(1, W - 2 * margin)

        def _slot_grid(used: int, cap: int, col_full: List[int], col_empty: List[int]) -> np.ndarray:
            g = np.empty((section_h, grid_w, 3), dtype=np.uint8)
            g[:] = col_empty
            if cap > 0:
                n = int(round(section_h * grid_w * min(used, cap) / cap))
                g.reshape(-1, 3)[:n] = col_full
            return g

        y0_wq = margin + label_h
        out[y0_wq: y0_wq + section_h, margin: margin + grid_w] = _slot_grid(
            wq_used, wq_cap, [72, 158, 100], [28, 32, 38])
        y0_fb = y0_wq + section_h + margin + label_h
        out[y0_fb: y0_fb + section_h, margin: margin + grid_w] = _slot_grid(
            fb_used, fb_cap, [90, 140, 210], [28, 32, 38])
        try:
            from PIL import Image, ImageDraw, ImageFont
            im = Image.fromarray(out)
            draw = ImageDraw.Draw(im)
            font = ImageFont.load_default()
            draw.text((margin, margin),                      f"work q  {wq_used}/{wq_cap}",  fill=(180, 220, 190), font=font)
            draw.text((margin, y0_wq + section_h + margin), f"scrub ring  {fb_used}/{fb_cap}", fill=(170, 200, 240), font=font)
            out = np.asarray(im, dtype=np.uint8)
        except Exception:
            pass
        return out

    def _render_knob(
        self,
        label: str,
        fill_frac: float = 0.0,
        color: Tuple[int, int, int] = (80, 160, 220),
    ) -> np.ndarray:
        """256x256 circular dial -- arc sweeps 300deg showing fill_frac [0, 1]."""
        H, W = self.panel_h, self.panel_w
        out = np.full((H, W, 3), np.array([14, 16, 20], dtype=np.uint8), dtype=np.uint8)
        try:
            from PIL import Image, ImageDraw, ImageFont
            im = Image.fromarray(out)
            draw = ImageDraw.Draw(im)
            font = ImageFont.load_default()
            label_h = 14
            margin = 8
            cx = W // 2
            cy = (H - label_h - margin) // 2 + margin
            r = min(cx - margin, cy - margin)
            if r < 4:
                return out
            # Outer circle body
            draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)],
                         fill=(20, 24, 30), outline=(55, 63, 78), width=2)
            # Inner ring
            ir = max(2, r - max(5, r // 5))
            draw.ellipse([(cx - ir, cy - ir), (cx + ir, cy + ir)],
                         outline=(42, 50, 62), width=1)
            # 12 tick marks
            for k in range(12):
                a = math.radians(-90.0 + k * 30.0)
                t0x = int(cx + (ir + 2) * math.cos(a))
                t0y = int(cy + (ir + 2) * math.sin(a))
                t1x = int(cx + (r - 1) * math.cos(a))
                t1y = int(cy + (r - 1) * math.sin(a))
                draw.line([(t0x, t0y), (t1x, t1y)],
                          fill=(95, 106, 122) if k % 3 == 0 else (48, 54, 66), width=1)
            # Progress arc: 300deg sweep starting at -210deg (just past bottom-left)
            sweep_deg = max(0.0, min(1.0, float(fill_frac))) * 300.0
            if sweep_deg > 0.5:
                arc_r = max(2, r - max(4, r // 6))
                draw.arc([(cx - arc_r, cy - arc_r), (cx + arc_r, cy + arc_r)],
                         start=-210, end=-210 + sweep_deg,
                         fill=color, width=max(3, arc_r // 5))
            # Centre dot
            dr = max(2, r // 10)
            draw.ellipse([(cx - dr, cy - dr), (cx + dr, cy + dr)], fill=(160, 170, 185))
            # Labels
            draw.text((4, H - label_h - 2), label[:W // 6], fill=(150, 160, 175), font=font)
            pct_str = f"{int(fill_frac * 100)}%"
            draw.text((W - len(pct_str) * 6 - 4, H - label_h - 2), pct_str, fill=color, font=font)
            out = np.asarray(im, dtype=np.uint8)
        except Exception:
            pass
        return out

    def _render_btn_panel(self) -> np.ndarray:
        """Left sidebar top panel: RESTORE STATE button and scrub status."""
        H, W = self.panel_h, self.panel_w
        out = np.full((H, W, 3), np.array([14, 16, 20], dtype=np.uint8), dtype=np.uint8)
        try:
            from PIL import Image, ImageDraw, ImageFont
            im = Image.fromarray(out)
            draw = ImageDraw.Draw(im)
            font = ImageFont.load_default()
            draw.rectangle([(0, 0), (W - 1, H - 1)], fill=(18, 22, 28))
            draw.text((4, 4), "Controls", fill=(140, 150, 170), font=font)
            draw.line([(0, 14), (W - 1, 14)], fill=(50, 56, 68), width=1)
            # RESTORE STATE button
            scrubbing = bool(self._scrub_offset > 0)
            btn_x0, btn_y0 = 8, 22
            btn_x1, btn_y1 = W - 8, 44
            btn_fill    = (90,  50, 160) if scrubbing else (36, 40, 48)
            btn_outline = (160, 120, 220) if scrubbing else (60, 66, 78)
            btn_text    = (230, 200, 255) if scrubbing else (80, 88, 100)
            draw.rectangle([(btn_x0, btn_y0), (btn_x1, btn_y1)],
                           fill=btn_fill, outline=btn_outline)
            lbl = "RESTORE STATE"
            lw = len(lbl) * 6
            draw.text(((W - lw) // 2, btn_y0 + 6), lbl, fill=btn_text, font=font)
            # Store window-space rect for click detection.
            self._restore_btn_window_rect = (
                btn_x0, btn_y0 + self.top_bar_h,
                btn_x1, btn_y1 + self.top_bar_h,
            )
            # Status info
            y_info = btn_y1 + 6
            total = self._history_snap_count()
            can_restore = callable(self._on_restore_state)
            in_ckpt_zone = self._scrub_in_checkpoint_zone()
            if scrubbing and in_ckpt_zone:
                ckpt_idx = self._checkpoint_index_from_offset(self._scrub_offset)
                if ckpt_idx is not None and ckpt_idx < len(self._checkpoint_thumbs):
                    ent = self._checkpoint_thumbs[ckpt_idx]
                    draw.text((4, y_info), f"ckpt r{ent['round_id']} c{ent['cycle']}",
                              fill=(200, 170, 40), font=font)
                else:
                    draw.text((4, y_info), f"checkpoint zone ({self._scrub_offset}/{total})",
                              fill=(200, 170, 40), font=font)
                draw.text((4, y_info + 12), "wheel \u2191\u2193 to navigate",
                          fill=(140, 120, 80), font=font)
                draw.text((4, y_info + 24), "showing weight thumbnail",
                          fill=(140, 120, 80), font=font)
            elif scrubbing:
                draw.text((4, y_info),      f"scrub: -{self._scrub_offset} / {total}",
                          fill=(180, 160, 220), font=font)
                draw.text((4, y_info + 12), "wheel \u2191\u2193 to navigate",
                          fill=(110, 100, 140), font=font)
                draw.text((4, y_info + 24), "click btn to restore & run" if can_restore else "restore unavailable",
                          fill=(110, 100, 140), font=font)
            else:
                draw.text((4, y_info),      f"live  (history: {total})",
                          fill=(80, 100, 80), font=font)
                draw.text((4, y_info + 12), "wheel \u2191\u2193 to scrub back",
                          fill=(60, 70, 60), font=font)

            # -- Prev / Next checkpoint buttons ----------------------------
            has_ckpts = len(self._disk_save_loss_counts) > 0
            nav_y0 = y_info + 36
            nav_h = 18
            half_w = (W - 24) // 2
            # PREV button
            p_x0, p_y0 = 8, nav_y0
            p_x1, p_y1 = 8 + half_w, nav_y0 + nav_h
            p_fill    = (50, 40, 80) if has_ckpts else (30, 32, 38)
            p_outline = (120, 100, 170) if has_ckpts else (50, 54, 62)
            p_text    = (180, 160, 220) if has_ckpts else (70, 74, 84)
            draw.rectangle([(p_x0, p_y0), (p_x1, p_y1)], fill=p_fill, outline=p_outline)
            plbl = "\u25c0 PREV"
            draw.text((p_x0 + 4, p_y0 + 3), plbl, fill=p_text, font=font)
            self._prev_ckpt_btn_rect = (
                p_x0, p_y0 + self.top_bar_h,
                p_x1, p_y1 + self.top_bar_h,
            )
            # NEXT button
            n_x0, n_y0 = W - 8 - half_w, nav_y0
            n_x1, n_y1 = W - 8, nav_y0 + nav_h
            n_fill    = (50, 40, 80) if has_ckpts else (30, 32, 38)
            n_outline = (120, 100, 170) if has_ckpts else (50, 54, 62)
            n_text    = (180, 160, 220) if has_ckpts else (70, 74, 84)
            draw.rectangle([(n_x0, n_y0), (n_x1, n_y1)], fill=n_fill, outline=n_outline)
            nlbl = "NEXT \u25b6"
            draw.text((n_x1 - len(nlbl) * 6 - 4, n_y0 + 3), nlbl, fill=n_text, font=font)
            self._next_ckpt_btn_rect = (
                n_x0, n_y0 + self.top_bar_h,
                n_x1, n_y1 + self.top_bar_h,
            )
            # Checkpoint count label between buttons
            n_ckpts = len(self._disk_save_loss_counts)
            if n_ckpts > 0:
                ck_lbl = f"{n_ckpts} ckpt{'s' if n_ckpts != 1 else ''}"
                draw.text(((W - len(ck_lbl) * 6) // 2, nav_y0 + nav_h + 2),
                          ck_lbl, fill=(130, 120, 160), font=font)

            # -- Step counter (below checkpoint nav) -----------------------
            step_txt = str(self._last_step_txt)
            if step_txt:
                draw.text((4, nav_y0 + nav_h + 16),
                          step_txt, fill=(220, 180, 90), font=font)

            # Loss display below step counter
            y_loss = nav_y0 + nav_h + 30
            if self._loss_display_rows:
                draw.line([(4, y_loss - 4), (W - 4, y_loss - 4)], fill=(50, 56, 68), width=1)
                for lr in self._loss_display_rows:
                    draw.text((4, y_loss), str(lr), fill=(220, 180, 90), font=font)
                    y_loss += 12

            # Cache summary below loss display
            if self._sr_cache_total > 0:
                y_cache = y_loss + 4
                if y_cache < H - 24:
                    draw.line([(4, y_cache - 2), (W - 4, y_cache - 2)], fill=(50, 56, 68), width=1)
                    draw.text((4, y_cache), f"cache: {self._sr_cache_total} entries",
                              fill=(100, 180, 180), font=font)
                    y_cache += 12
                    for ce in self._sr_cache_entries[:5]:
                        if y_cache >= H - 12:
                            break
                        ck = ce.get("channel_key", "?")
                        rnd = ce.get("round_id", 0)
                        draw.text((4, y_cache), f"  {ck} r{rnd}",
                                  fill=(80, 140, 140), font=font)
                        y_cache += 10
                    remaining = max(0, self._sr_cache_total - 5)
                    if remaining > 0 and y_cache < H - 12:
                        draw.text((4, y_cache), f"  +{remaining} more",
                                  fill=(60, 110, 110), font=font)

            out = np.asarray(im, dtype=np.uint8)
        except Exception:
            pass
        return out

    def _render_scrub_dial(self) -> np.ndarray:
        """Left sidebar bottom panel: circular dial showing scrub offset."""
        total = max(1, self._history_snap_count())
        offset = int(self._scrub_offset)
        fill_frac = float(offset) / float(total)
        in_ckpt_zone = self._scrub_in_checkpoint_zone()
        color = (200, 170, 40) if in_ckpt_zone else (170, 80, 220) if offset > 0 else (60, 80, 110)
        dial = self._render_knob("scrub", fill_frac=fill_frac, color=color)
        try:
            from PIL import Image, ImageDraw, ImageFont
            im = Image.fromarray(dial)
            draw = ImageDraw.Draw(im)
            font = ImageFont.load_default()
            if in_ckpt_zone:
                ckpt_idx = self._checkpoint_index_from_offset(offset)
                if ckpt_idx is not None and ckpt_idx < len(self._checkpoint_thumbs):
                    entry = self._checkpoint_thumbs[ckpt_idx]
                    txt = f"ckpt r{entry['round_id']}"
                else:
                    txt = "ckpt"
                draw.text(
                    (self.panel_w // 2 - len(txt) * 3, self.panel_h // 2 - 4),
                    txt, fill=(200, 170, 40), font=font,
                )
            else:
                txt = f"-{offset}" if offset > 0 else "live"
                draw.text(
                    (self.panel_w // 2 - len(txt) * 3, self.panel_h // 2 - 4),
                    txt, fill=(220, 210, 240), font=font,
                )
            dial = np.asarray(im, dtype=np.uint8)
        except Exception:
            pass
        return dial

    def _render_weight_map(self) -> np.ndarray:
        """Render weight image via the single render_weight_image function, crop to display."""
        H, W = self._weight_map_target_hw()
        out = np.full((H, W, 3), np.array([14, 16, 20], dtype=np.uint8), dtype=np.uint8)
        refs = self._weight_model_refs
        if not refs:
            try:
                from PIL import Image, ImageDraw, ImageFont
                im = Image.fromarray(out)
                draw = ImageDraw.Draw(im)
                font = ImageFont.load_default()
                draw.text((4, 4), "no weight refs\nset_weight_model_refs()", fill=(70, 78, 90), font=font)
                out = np.asarray(im, dtype=np.uint8)
            except Exception:
                pass
            return out

        _active_name = self._active_weight_model_name
        if _active_name is None or _active_name not in refs:
            _active_name = next(iter(refs))
        model = refs[_active_name]
        try:
            _sd_snap = {k: v.detach().float().cpu()
                        for k, v in model.state_dict().items()}
            param_keys = list(_sd_snap.keys())
            reference = self._weight_disk_states.get(_active_name)
            rgb, _meta = render_weight_image(
                _sd_snap,
                parameter_keys=param_keys,
                reference_state=reference,
                target_width=W,
                target_height=H,
            )
            # Crop to display area (image may exceed target when neuron count > target)
            crop_h = min(H, rgb.shape[0])
            crop_w = min(W, rgb.shape[1])
            out[:crop_h, :crop_w] = rgb[:crop_h, :crop_w]
        except Exception:
            pass
        return out

    def _draw_texture_px(self, tex_id: int, x0: int, y0: int, x1: int, y1: int):
        gl = self._gl
        xf0 = -1.0 + (2.0 * float(x0) / float(max(1, self.window_w)))
        xf1 = -1.0 + (2.0 * float(x1) / float(max(1, self.window_w)))
        yt = 1.0 - (2.0 * float(y0) / float(max(1, self.window_h)))
        yb = 1.0 - (2.0 * float(y1) / float(max(1, self.window_h)))
        gl.glBindTexture(gl.GL_TEXTURE_2D, int(tex_id))
        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(0.0, 1.0)
        gl.glVertex2f(float(xf0), float(yb))
        gl.glTexCoord2f(1.0, 1.0)
        gl.glVertex2f(float(xf1), float(yb))
        gl.glTexCoord2f(1.0, 0.0)
        gl.glVertex2f(float(xf1), float(yt))
        gl.glTexCoord2f(0.0, 0.0)
        gl.glVertex2f(float(xf0), float(yt))
        gl.glEnd()

    def _present(self, force: bool = False):
        if (not self._ready) or (not self.enabled) or self._stop_requested:
            return
        now = time.perf_counter()
        self._drain_checkpoint_thumbnail_results()

        # -- Drain new scrub ring entries into the composite cache ----------
        ring = self._scrub_ring
        cache = self._composite_cache
        drained_any = False
        if ring is not None and cache is not None:
            with ring.locked():
                ring_wc = ring.write_cursor()
                ring_len = ring.length()
            new_count = ring_wc - self._last_ring_cursor
            if new_count < 0:
                # Ring was cleared; reset tracking.
                cache.clear()
                self._clear_frame_text_cache()
                self._last_ring_cursor = ring_wc
                new_count = 0
            if new_count > ring_len:
                # Entries evicted past our tracking point; skip to valid range.
                self._last_ring_cursor = ring_wc - ring_len
                self._prune_pending_frame_text(min_cursor=self._last_ring_cursor)
                new_count = ring_len

            # Slew the drain rate: pending composite builds drive target interval.
            fill = float(min(new_count, 256)) / 256.0
            target_dt = self._anim_dt_fast + (self._anim_dt_slow - self._anim_dt_fast) * ((1.0 - fill) ** 2)
            _slew_elapsed = max(1e-4, now - self._last_slew_t)
            self._last_slew_t = now
            alpha = 1.0 - math.exp(-_slew_elapsed / max(1e-4, self._anim_slew_tau))
            self._anim_frame_dt += alpha * (target_dt - self._anim_frame_dt)

            while new_count > 0 and (now - self._last_anim_t) >= self._anim_frame_dt:
                source_cursor = int(self._last_ring_cursor)
                ring_idx = ring_len - new_count
                if 0 <= ring_idx < ring_len:
                    cache_cursor = cache.build_and_push(ring, ring_idx, self.panel_w, self.panel_h)
                    if cache_cursor >= 0:
                        self._cache_frame_text_for_cursor(source_cursor)
                        self._record_history_snap(source_cursor)
                        drained_any = True
                self._last_ring_cursor += 1
                new_count -= 1
                self._last_anim_t += self._anim_frame_dt
        else:
            self._last_slew_t = now

        if self._scrub_offset == 0:
            if drained_any and cache is not None:
                clen = cache.length()
                if clen > 0:
                    self._apply_composite_to_display(clen - 1)
        else:
            self._apply_composite_at_offset(self._scrub_offset)

        # Sidebar: btn_panel + scrub_dial every pump (cheap); weight snap is count-driven.
        self._cache_map_rgb  = self._render_btn_panel()
        self._frame_knob_rgb = self._render_scrub_dial()
        if self._scrub_offset == 0:
            if self._weight_snapshot_deque:
                live_blob = self._weight_snapshot_deque[-1]
                if isinstance(live_blob, dict):
                    live_wmap = self._weight_map_from_ring_cursor(live_blob.get("ring_cursor"))
                    if live_wmap is not None:
                        self._weight_map_rgb = live_wmap
            # Collect finished background snap if ready.
            if (self._weight_snap_thread is not None
                    and not self._weight_snap_thread.is_alive()
                    and self._weight_snap_result is not None):
                _res = self._weight_snap_result
                self._weight_snap_result = None
                self._weight_snap_thread = None
                _wmap = _res.get("weight_map")
                if _wmap is not None:
                    self._weight_map_rgb = _wmap
                    if self._weight_snapshot_deque:
                        _prev_blob = self._weight_snapshot_deque[-1]
                        self._weight_snapshot_deque[-1] = {
                            "ring_cursor": (_prev_blob.get("ring_cursor") if isinstance(_prev_blob, dict) else None),
                            "weight_map": _wmap.copy(),
                        }
                    else:
                        self._weight_snapshot_deque.append({"ring_cursor": None, "weight_map": _wmap.copy()})
                _sst = _res.get("states")
                if _sst is not None:
                    # Delete the file for the entry about to be evicted before it's gone.
                    _sd_maxlen = self._weight_state_sparse_deque.maxlen
                    if _sd_maxlen is not None and len(self._weight_state_sparse_deque) >= _sd_maxlen:
                        _evict = self._weight_state_sparse_deque[0]
                        _evict_file = _evict.get("file")
                        if _evict_file is not None:
                            try:
                                Path(_evict_file).unlink(missing_ok=True)
                            except Exception:
                                pass
                    if _sst:
                        self._prev_weight_snap = _sst
                    # Record loss-count position for the graph marker.
                    self._disk_save_loss_counts.append(
                        self._loss_channel_lengths()
                    )
                    self._weight_state_sparse_deque.append({
                        "snap_total": _res["snap_total"],
                        "states": _sst,
                        "file": _res.get("file"),
                    })
            # Fire a new background snap at the stride rate (only if previous finished).
            self._snap_total += 1
            if (self._snap_total % self._weight_snap_stride == 0
                    and self._weight_model_refs
                    and self._weight_snap_thread is None):
                _snap_total_capture = self._snap_total
                _model_refs_capture = dict(self._weight_model_refs)
                _prev_snap_capture  = dict(self._prev_weight_snap)
                _result_box: Dict[str, Any] = {}
                def _weight_snap_worker(
                        _refs=_model_refs_capture,
                        _st=_snap_total_capture,
                        _box=_result_box,
                        _prev=_prev_snap_capture,
                        _self=self) -> None:
                    _wmap = _self._render_weight_map()
                    _box["weight_map"] = _wmap
                    _box["snap_total"] = _st
                    _sparse: Dict[str, Any] = {}
                    for _sn, _sm in _refs.items():
                        try:
                            _sparse[_sn] = {
                                _k: _v.detach().cpu().clone()
                                for _k, _v in _sm.state_dict().items()
                            }
                        except Exception:
                            pass
                    _box["states"] = _sparse
                    # Log params whose weights shifted by more than threshold.
                    _thresh = _self._weight_delta_threshold
                    for _mn, _msd in _sparse.items():
                        _prev_msd = _prev.get(_mn)
                        if _prev_msd is None:
                            continue
                        for _pkey, _ptens in _msd.items():
                            _pprev = _prev_msd.get(_pkey)
                            if _pprev is None or _pprev.shape != _ptens.shape:
                                continue
                            try:
                                _d = (_ptens.float() - _pprev.float()).abs().max().item()
                                if _d > _thresh:
                                    print(f"[weight_delta] {_mn}.{_pkey}: max|\u0394|={_d:.4f}")
                            except Exception:
                                pass
                self._weight_snap_result = _result_box
                self._weight_snap_thread = threading.Thread(
                    target=_weight_snap_worker, daemon=True)
                self._weight_snap_thread.start()
        else:
            # Frozen: show the snapshot at the chosen offset.
            # _scrub_offset 1 = most-recent snapshot; _slen = oldest.
            weight_map = self._history_weight_map_at_offset(self._scrub_offset)
            if weight_map is not None:
                self._weight_map_rgb = weight_map
        self._sidebar_dirty = True

        dirty = bool(self._top_bar_dirty or self._panel_text_dirty or self._graph_dirty or self._sidebar_dirty)
        if (not dirty) and (not force):
            return

        if self._panel_text_dirty:
            for i in range(3):
                title = self._panel_titles[i] if i < len(self._panel_titles) else ""
                rows = self._panel_rows[i] if i < len(self._panel_rows) else []
                self._panel_text_rgb[i] = self._render_panel_text(title=title, rows=rows)
                self._upload_texture(int(self._textures["text"][i]), self._panel_text_rgb[i])
            self._panel_text_dirty = False

        if self._top_bar_dirty:
            self._top_bar_rgb = self._render_top_bar()
            self._upload_texture(int(self._textures["bar"]), self._top_bar_rgb)
            self._top_bar_dirty = False

        if self.graph_total_h > 0 and self._graph_dirty:
            self._graph_rgb = self._render_loss_graph()
            self._upload_texture(int(self._textures["graph"]), self._graph_rgb)
            self._graph_dirty = False

        if self._sidebar_dirty:
            self._upload_texture(int(self._textures["cache_map"]),  self._cache_map_rgb)
            self._upload_texture(int(self._textures["frame_knob"]), self._frame_knob_rgb)
            self._upload_texture(int(self._textures["weight_map"]), self._weight_map_rgb)
            self._sidebar_dirty = False

        gl = self._gl
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        self._draw_texture_px(
            int(self._textures["bar"]),
            0, 0, self.window_w, self.top_bar_h,
        )

        # Left sidebar column
        sb_y1 = int(self.top_bar_h + self.panel_h)
        sb_y2 = int(self.top_bar_h + 2 * self.panel_h)
        self._draw_texture_px(int(self._textures["cache_map"]),  0, self.top_bar_h, self._col_x, sb_y1)
        self._draw_texture_px(int(self._textures["frame_knob"]), 0, sb_y1,          self._col_x, sb_y2)

        # 3 main panels (offset right by one column)
        for i in range(3):
            x0 = int(self._col_x + i * self.panel_w)
            x1 = int(self._col_x + (i + 1) * self.panel_w)
            self._draw_texture_px(
                int(self._textures["text"][i]),
                x0, self.top_bar_h, x1, self.top_bar_h + self.panel_h,
            )
            self._draw_texture_px(
                int(self._textures["img"][i]),
                x0, self.top_bar_h + self.panel_h, x1, self.top_bar_h + (2 * self.panel_h),
            )

        # Right sidebar column - weight map spans the full content area including graph row.
        rx0 = int(self._col_x + self.num_panels * self.panel_w)
        self._draw_texture_px(int(self._textures["weight_map"]), rx0, self.top_bar_h, self.window_w, self.window_h)

        if self.graph_total_h > 0 and self._graph_rgb is not None:
            graph_y0 = int(self.top_bar_h + (2 * self.panel_h))
            # Graph only spans under the left 4 columns; right sidebar keeps the space.
            self._draw_texture_px(
                int(self._textures["graph"]),
                0, graph_y0, rx0, graph_y0 + self.graph_total_h,
            )

        gate_mode = "manual" if self._gate_override else "auto"
        self._pygame.display.set_caption(f"Stage Status Viewer | gate={gate_mode} | {self._caption}")
        self._pygame.display.flip()
        self._has_presented_frame = True

    def _handle_click(self, x: int, y: int) -> bool:
        xi = int(x)
        yi = int(y)
        if yi < 0 or yi >= int(self.top_bar_h):
            return False
        for kind, idx, box in self._control_boxes:
            if (xi >= int(box[0])) and (xi <= int(box[2])) and (yi >= int(box[1])) and (yi <= int(box[3])):
                if kind == "cycle":
                    if int(idx) >= 0 and int(idx) < len(self._cycle_selected):
                        self._cycle_selected[int(idx)] = not bool(self._cycle_selected[int(idx)])
                        self._top_bar_dirty = True
                        return True
                elif kind == "override":
                    self._gate_override = not bool(self._gate_override)
                    self._top_bar_dirty = True
                    return True
                elif kind == "pause":
                    self._paused = not bool(self._paused)
                    self._top_bar_dirty = True
                    return True
                elif kind == "preview":
                    self._preview_enabled = not bool(self._preview_enabled)
                    # Turning preview off auto-disables scrub editor
                    if not self._preview_enabled:
                        self._scrub_editor_enabled = False
                    self._top_bar_dirty = True
                    return True
                elif kind == "scrub_editor":
                    # Scrub editor requires preview to be on
                    if not self._preview_enabled:
                        pass  # can't enable scrub editor without preview
                    else:
                        self._scrub_editor_enabled = not bool(self._scrub_editor_enabled)
                    self._top_bar_dirty = True
                    return True
                elif kind == "stop_save":
                    self._shutdown_save = True
                    self._top_bar_dirty = True
                    print("[viewer] STOP+SAVE requested", flush=True)
                    return True
                elif kind == "stop_nosave":
                    self._shutdown_save = False
                    self._top_bar_dirty = True
                    print("[viewer] STOP (no save) requested", flush=True)
                    return True
                elif kind == "start":
                    self._start_training()
                    return True
        return False

    def _poll_events(self):
        if (not self._ready) or (self._pygame is None):
            return
        try:
            for event in self._pygame.event.get():
                if event.type == self._pygame.QUIT:
                    self._stop_requested = True
                    self.enabled = False
                    self.close()
                    return
                # Mouse-wheel: scrub weight / preview history.
                if event.type == self._pygame.MOUSEWHEEL:
                    if not self._scrub_editor_enabled:
                        continue
                    delta = int(getattr(event, "y", 0))
                    _snap_len = self._history_snap_count()
                    if _snap_len > 0:
                        # Upper bound is _snap_len (not _snap_len-1) so the oldest
                        # snapshot at index 0 is reachable (idx = _slen - _slen = 0).
                        self._scrub_offset = max(
                            0, min(_snap_len, self._scrub_offset - delta)
                        )
                        self._sidebar_dirty = True
                        self._graph_dirty = True  # redraw loss cursor at new scrub pos
                        self._present(force=True)
                    continue
                if event.type == self._pygame.MOUSEBUTTONDOWN and int(getattr(event, "button", 0)) == 1:
                    pos = getattr(event, "pos", None)
                    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                        xi, yi = int(pos[0]), int(pos[1])
                        # Restore-state button (left sidebar, button panel).
                        if self._restore_btn_window_rect is not None and self._scrub_offset > 0:
                            bx0, by0, bx1, by1 = self._restore_btn_window_rect
                            if bx0 <= xi <= bx1 and by0 <= yi <= by1:
                                if self._on_restore_state is not None:
                                    try:
                                        self._on_restore_state(int(self._scrub_offset))
                                    except Exception:
                                        pass
                                # Also send via IPC for remote restore
                                srv = self._ipc_server_ref
                                if srv is not None and getattr(srv, "has_connection", False):
                                    try:
                                        restore_msg = {
                                            "type": "restore",
                                            "offset": int(self._scrub_offset),
                                        }
                                        if self._scrub_in_checkpoint_zone():
                                            ckpt_idx = self._checkpoint_index_from_offset(self._scrub_offset)
                                            if ckpt_idx is not None and 0 <= ckpt_idx < len(self._checkpoint_thumbs):
                                                ent = self._checkpoint_thumbs[ckpt_idx]
                                                restore_msg["round_id"] = int(ent.get("round_id", 0))
                                                restore_msg["cycle"] = int(ent.get("cycle", 0))
                                        srv.send_query(restore_msg)
                                    except Exception:
                                        pass
                                self._scrub_offset = 0
                                self._weight_snapshot_deque.clear()
                                if self._composite_cache is not None:
                                    self._composite_cache.clear()
                                self._clear_frame_text_cache()
                                self._loss_count_at_snap_deque.clear()
                                self._sidebar_dirty = True
                                self._present(force=True)
                                continue
                        # Prev checkpoint button.
                        if self._prev_ckpt_btn_rect is not None:
                            bx0, by0, bx1, by1 = self._prev_ckpt_btn_rect
                            if bx0 <= xi <= bx1 and by0 <= yi <= by1:
                                new_off = self._scrub_offset_for_checkpoint(-1)
                                if new_off is not None:
                                    self._scrub_offset = new_off
                                    self._sidebar_dirty = True
                                    self._graph_dirty = True
                                    self._present(force=True)
                                continue
                        # Next checkpoint button.
                        if self._next_ckpt_btn_rect is not None:
                            bx0, by0, bx1, by1 = self._next_ckpt_btn_rect
                            if bx0 <= xi <= bx1 and by0 <= yi <= by1:
                                new_off = self._scrub_offset_for_checkpoint(+1)
                                if new_off is not None:
                                    self._scrub_offset = new_off
                                    self._sidebar_dirty = True
                                    self._graph_dirty = True
                                    self._present(force=True)
                                continue
                        # Graph toolbar + legend clicks.
                        graph_y0 = int(self.top_bar_h + 2 * self.panel_h)
                        if graph_y0 <= yi < graph_y0 + self.graph_total_h:
                            gx = xi
                            gy = yi - graph_y0
                            handled_graph_click = False
                            for kind, value, box in self._graph_control_boxes:
                                if box[0] <= gx <= box[2] and box[1] <= gy <= box[3]:
                                    if kind == "graph_axis" and value in {"t", "i"}:
                                        self._graph_x_axis_mode = str(value)
                                        self._graph_dirty = True
                                        handled_graph_click = True
                                    elif kind == "graph_history" and value in {"recent", "full"}:
                                        self._graph_history_mode = str(value)
                                        self._graph_dirty = True
                                        handled_graph_click = True
                                    if handled_graph_click:
                                        self._present(force=True)
                                        break
                            if handled_graph_click:
                                continue
                            for ck, lx0, ly0, lx1, ly1 in self._legend_hit_boxes:
                                if lx0 <= gx <= lx1 and ly0 <= gy <= ly1:
                                    cur = self._loss_channel_visible.get(ck, True)
                                    self._loss_channel_visible[ck] = not cur
                                    self._graph_dirty = True
                                    self._present(force=True)
                                    break
                            continue
                        # Top-bar toggle controls.
                        if self._handle_click(xi, yi):
                            self._present(force=True)
        except Exception:
            pass

    def pump(self):
        if not self.enabled:
            return
        self._init()
        if not self._ready:
            return
        self._poll_events()
        self._sr_poll_data()
        self._present(force=False)

    def notify_pipeline_checkpoint_saved(self) -> None:
        """Call this immediately after every _save_training_segment_snapshot call.

        Records the current loss-series lengths so the loss graph can draw a gold
        vertical marker at the exact loss position where each disk checkpoint
        was written.  Thread-safe: can be called from any thread.
        """
        snapshot = self._loss_channel_lengths()
        self._disk_save_loss_counts.append(snapshot)
        self._graph_dirty = True

    def notify_checkpoint_at_walltime(self, wall_ts: float) -> None:
        """Add a graph marker at the loss-series position nearest to wall_ts.

        Used at startup to place gold markers for checkpoint files that already
        existed on disk before this session (their mtime = wall_ts).  Finds the
        loss record whose timestamp is closest to wall_ts by binary-searching
        the C store ts arrays, then records the corresponding index into
        _disk_save_loss_counts so the graph renderer draws a gold marker there.
        """
        import bisect
        store = self._loss_store
        if store is None:
            return
        keys = self._loss_channel_keys()
        if not keys:
            self.notify_pipeline_checkpoint_saved()
            return
        snapshot: Dict[str, int] = {}
        for ck in keys:
            ts_tensor = store.get_ts_array(ck)
            if ts_tensor.numel() == 0:
                continue
            ts_list = ts_tensor.tolist()
            idx = bisect.bisect_left(ts_list, float(wall_ts))
            idx = max(0, min(len(ts_list) - 1, idx))
            snapshot[ck] = idx + 1  # 1-based count mirrors length convention
        if snapshot:
            self._disk_save_loss_counts.append(snapshot)
            self._graph_dirty = True

    def set_checkpoint_backup_dir(self, path) -> None:
        """Scan a backup directory for timestamped sub-dirs created by the batch launcher.

        Each sub-directory is expected to be named ``YYYYMMDD_HHMMSS`` and contain
        ``.pt`` files.  The directory's mtime (or the most-recently-modified .pt
        inside it) is used as the wall-clock timestamp.  A gold marker is placed
        on the loss graph at that time via ``notify_checkpoint_at_walltime``.

        Also discovers ``weight_thumb_r*_c*.png`` thumbnails so the extended
        scrub wheel can show checkpoint weight snapshots.
        """
        p = Path(path)
        if not p.is_dir():
            return
        _times: list = []
        for sub in sorted(p.iterdir()):
            if not sub.is_dir():
                continue
            pts = list(sub.glob("*.pt"))
            if not pts:
                continue
            checkpoint_path = sub / "pipeline_checkpoint.pt"
            if not checkpoint_path.exists():
                checkpoint_path = max(pts, key=lambda fp: fp.stat().st_mtime)
            # Use the newest .pt mtime in the sub-dir as the checkpoint time.
            t = max(f.stat().st_mtime for f in pts)
            _times.append(t)
            # Discover weight thumbnails and register for extended scrub.
            import re as _re
            found_thumb = False
            for thumb in sub.glob("weight_thumb_r*_c*.png"):
                m = _re.search(r"weight_thumb_r(\d+)_c(\d+)\.png$", thumb.name)
                if m:
                    found_thumb = True
                    rid = int(m.group(1))
                    cid = int(m.group(2))
                    model_name = None
                    sidecar = thumb.with_suffix(".json")
                    if sidecar.exists():
                        try:
                            meta = json.loads(sidecar.read_text(encoding="utf-8"))
                            model_name = meta.get("model")
                        except Exception:
                            model_name = None
                    self.register_checkpoint_thumbnail(
                        rid,
                        cid,
                        thumb_path=str(thumb),
                        model_name=model_name,
                        checkpoint_path=str(checkpoint_path),
                    )
            if (not found_thumb) and checkpoint_path.exists():
                try:
                    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                    rid = int(payload.get("round_id", 0)) if isinstance(payload, dict) else 0
                    cid = int(payload.get("cycle", 0)) if isinstance(payload, dict) else 0
                    model_name, _state = _extract_checkpoint_weight_state(payload if isinstance(payload, dict) else {}, self._active_weight_model_name)
                    self.register_checkpoint_thumbnail(
                        rid,
                        cid,
                        checkpoint_path=str(checkpoint_path),
                        model_name=model_name,
                    )
                except Exception:
                    pass
        for t in _times:
            self.notify_checkpoint_at_walltime(t)

    def trim_graph_to_first_checkpoint(self) -> None:
        """Set the graph's left edge to 5% before the earliest checkpoint marker.

        Call this *after* loading all historical loss data and setting checkpoint
        markers.  If no markers exist the graph shows everything (start_frac=0).
        """
        if not self._disk_save_loss_counts:
            self._graph_display_start_frac = 0.0
            return
        # Find the minimum fractional position across all markers.
        min_frac = 1.0
        _lengths = self._loss_channel_lengths()
        for snap in self._disk_save_loss_counts:
            for sid, count in snap.items():
                total = _lengths.get(sid, 0)
                if total == 0:
                    continue
                frac = float(count) / float(total)
                min_frac = min(min_frac, frac)
        # 5% left of the first checkpoint, clamped to 0.
        self._graph_display_start_frac = max(0.0, min_frac - 0.05)
        self._graph_dirty = True

    def stop_requested(self) -> bool:
        return bool(self._stop_requested)

    def update_loss(self, channel_key: str, loss: float, aux: float = 0.0, ts: float = 0.0):
        """Record a per-step loss value into the native C store.

        Prefer writing to the store directly; this thin wrapper exists for
        legacy callers that don't have a direct store reference.
        """
        if self.graph_h <= 0:
            return
        store = self._loss_store
        if store is not None:
            store.record(str(channel_key), float(loss), aux=float(aux), ts=float(ts) if float(ts) > 0.0 else time.time())
        self._graph_dirty = True

    # -- Helpers: read channel info from the C store --------------------

    def _loss_channel_keys(self) -> List[str]:
        """Return sorted channel keys from the native store."""
        store = self._loss_store
        if store is None:
            return []
        return sorted(store.channel_keys())

    def _loss_channel_length(self, ck: str) -> int:
        store = self._loss_store
        return store.channel_length(ck) if store else 0

    def _loss_channel_lengths(self) -> Dict[str, int]:
        """Return {channel_key: length} for every channel."""
        store = self._loss_store
        if store is None:
            return {}
        return {ck: store.channel_length(ck) for ck in store.channel_keys()}

    # -- Pull-model handlers (SaveRestoreNode IPC integration) ----------

    def _on_sr_notification(self, msg: dict) -> None:
        """Handle a lightweight notification from the SaveRestoreNode."""
        t = msg.get("type", "")
        if t == "notify_new_loss":
            # Loss data lives in the C store -- just mark redraw.
            self._graph_dirty = True
        elif t == "notify_new_result":
            pass  # Next poll will pull it
        elif t == "notify_checkpoint":
            self.notify_pipeline_checkpoint_saved()
            r = int(msg.get("round_id", 0))
            c = int(msg.get("cycle", 0))
            tp = msg.get("thumb_path")
            tm = msg.get("thumb_model")
            cp = msg.get("checkpoint_path")
            loss_counts = self._loss_channel_lengths()
            self.register_checkpoint_thumbnail(
                r,
                c,
                thumb_path=tp,
                loss_counts=loss_counts,
                model_name=tm,
                checkpoint_path=cp,
            )
        elif t == "notify_weight_map":
            self._sidebar_dirty = True
            model_name = msg.get("model")
            if model_name:
                self._active_weight_model_name = str(model_name)

    def _on_sr_response(self, msg: dict) -> None:
        """Handle a response to one of our queries from the SaveRestoreNode."""
        t = msg.get("type", "")

        if t == "resp_channel_list":
            self._sr_channel_registry = msg.get("channels", [])

        elif t == "resp_latest_result":
            ck = msg.get("channel_key", "")
            result = msg.get("result")
            if result is not None:
                caption = result.get("caption", ck)
                titles = result.get("titles")
                rows = result.get("rows")
                if caption:
                    self._ring_text_caption = str(caption)
                if titles:
                    self._ring_text_titles = [str(x) for x in titles]
                if rows:
                    self._ring_text_rows = [list(r) for r in rows]

        elif t == "resp_weight_map":
            wmap = msg.get("map")
            if wmap is not None:
                try:
                    rgb_list = wmap.get("rgb")
                    if rgb_list is not None:
                        rgb_arr = np.array(rgb_list, dtype=np.uint8)
                        self._weight_map_rgb = rgb_arr
                        self._sidebar_dirty = True
                    model_name = wmap.get("model")
                    if model_name:
                        self._active_weight_model_name = str(model_name)
                except Exception:
                    pass
            self._sr_weight_models = msg.get("models", [])
            self._sr_weight_active = msg.get("active")
            if self._sr_weight_active:
                self._active_weight_model_name = str(self._sr_weight_active)

        elif t == "resp_cache_browse":
            self._sr_cache_entries = msg.get("entries", [])
            self._sr_cache_total = int(msg.get("total", 0))

    def _sr_poll_data(self) -> None:
        """Periodically request fresh data from the SaveRestoreNode.

        Loss history is pulled incrementally per-channel using cursors so
        the graph can display data that was recorded in the training process.
        """
        now = time.time()
        if now - self._sr_last_poll_t < self._sr_poll_interval:
            return
        self._sr_last_poll_t = now

        # The C store may have new data -- always mark dirty.
        if self._loss_store is not None:
            self._graph_dirty = True

        srv = self._ipc_server_ref
        if srv is None or not getattr(srv, "has_connection", False):
            return

        srv.send_query({"type": "query_channel_list"})

        for ch_info in self._sr_channel_registry:
            ck = ch_info.get("key", "")
            if ch_info.get("has_result"):
                srv.send_query({
                    "type": "query_latest_result",
                    "channel_key": ck,
                })

        srv.send_query({"type": "query_cache_browse", "offset": 0, "limit": 50})

    def _graph_recent_start_frac(self) -> float:
        base = max(0.0, min(0.99, float(self._graph_display_start_frac)))
        if not self._loss_count_at_snap_deque:
            return base
        lengths = self._loss_channel_lengths()
        ref_sid = max(lengths, key=lengths.get, default=None)
        if ref_sid is None:
            return base
        oldest_counts = list(self._loss_count_at_snap_deque)[0]
        total_n = int(lengths.get(ref_sid, 0))
        start_n = int(oldest_counts.get(ref_sid, 0))
        if total_n <= 0 or start_n <= 0:
            return base
        return max(base, float(max(0, min(total_n, start_n))) / float(max(1, total_n)))

    def _graph_view_start_frac(self) -> float:
        base = max(0.0, min(0.99, float(self._graph_display_start_frac)))
        if str(self._graph_history_mode) == "recent":
            return self._graph_recent_start_frac()
        return base

    @staticmethod
    def _format_graph_elapsed(seconds: float) -> str:
        sec = max(0.0, float(seconds))
        if sec >= 3600.0:
            return f"{sec / 3600.0:.1f}h"
        if sec >= 60.0:
            return f"{sec / 60.0:.1f}m"
        return f"{sec:.0f}s"

    def _render_loss_graph(self) -> np.ndarray:
        try:
            from PIL import Image, ImageDraw, ImageFont
        except Exception:
            _gw = int(self._col_x + self.num_panels * self.panel_w)
            return np.full((self.graph_total_h, _gw, 3), 14, dtype=np.uint8)

        W_full = int(self._col_x + self.num_panels * self.panel_w)
        H = int(self.graph_total_h)
        toolbar_h = int(self.graph_toolbar_h)

        # Reserve the right portion for the pipeline graph canvas.
        pipeline_W = 0
        if isinstance(self._graph_plan_snapshot, dict) and self._graph_plan_snapshot.get("nodes"):
            pipeline_W = max(100, min(W_full // 3, 160))
        W = W_full - pipeline_W

        im = Image.new("RGB", (W_full, H), (14, 18, 22))
        loss_im = Image.new("RGB", (W, H), (14, 18, 22))
        draw = ImageDraw.Draw(loss_im)
        font = ImageFont.load_default()

        self._graph_control_boxes = []
        if toolbar_h > 0:
            draw.rectangle([(0, 0), (W - 1, toolbar_h - 1)], fill=(18, 22, 28))
            draw.line([(0, toolbar_h - 1), (W - 1, toolbar_h - 1)], fill=(46, 54, 66), width=1)
            draw.text((6, 6), "graph", fill=(176, 186, 198), font=font)

            def _toolbar_button(x: int, label: str, *, kind: str, value: str, active: bool) -> int:
                btn_w = max(18, len(label) * 7 + 10)
                box = (int(x), 4, int(x + btn_w), max(15, toolbar_h - 5))
                fill = (72, 108, 150) if active else (34, 40, 48)
                outline = (132, 170, 212) if active else (68, 74, 84)
                fg = (228, 236, 246) if active else (170, 178, 188)
                draw.rectangle([box[0], box[1], box[2], box[3]], fill=fill, outline=outline)
                draw.text((box[0] + 4, box[1] + 1), label, fill=fg, font=font)
                self._graph_control_boxes.append((kind, str(value), box))
                return int(box[2] + 6)

            tx = 46
            draw.text((tx, 6), "x", fill=(148, 158, 170), font=font)
            tx += 12
            tx = _toolbar_button(tx, "t", kind="graph_axis", value="t", active=(self._graph_x_axis_mode == "t"))
            tx = _toolbar_button(tx, "i", kind="graph_axis", value="i", active=(self._graph_x_axis_mode == "i"))
            tx += 6
            draw.text((tx, 6), "range", fill=(148, 158, 170), font=font)
            tx += 36
            tx = _toolbar_button(
                tx,
                "recent",
                kind="graph_history",
                value="recent",
                active=(self._graph_history_mode == "recent"),
            )
            _toolbar_button(
                tx,
                "full",
                kind="graph_history",
                value="full",
                active=(self._graph_history_mode == "full"),
            )

        # Plot margins
        mx0, mx1 = 52, W - 6
        my0, my1 = toolbar_h + 14, H - 18
        plot_w = max(1, mx1 - mx0)
        plot_h = max(1, my1 - my0)

        _sf = self._graph_view_start_frac()
        _view_span = max(1e-9, 1.0 - _sf)
        axis_mode = "t" if str(self._graph_x_axis_mode) == "t" else "i"

        def _view_x(data_frac: float) -> int:
            vf = (float(data_frac) - _sf) / _view_span
            return max(mx0, min(mx1, int(mx0 + vf * plot_w)))

        # -- Collect visible channel keys and colours -----------------------
        store = self._loss_store
        all_keys = self._loss_channel_keys() if store is not None else []
        vis_keys: List[str] = []
        vis_colors: List[Tuple[int, int, int, int]] = []
        n_all = len(all_keys)
        sorted_keys = sorted(all_keys)
        for ck in sorted_keys:
            if not self._loss_channel_visible.get(ck, True):
                continue
            idx = sorted_keys.index(ck)
            r, g, b = _channel_color(idx, n_all)
            vis_keys.append(ck)
            vis_colors.append((r, g, b, 200))

        lengths = self._loss_channel_lengths()
        ref_sid = max(lengths, key=lengths.get, default=None)
        ref_total = int(lengths.get(ref_sid, 0)) if ref_sid is not None else 0
        ref_ts = None

        # -- Render visible channel lines -----------------------------------
        result = None
        if store is not None and vis_keys and plot_w > 0 and plot_h > 0:
            if axis_mode == "i":
                # Index mode: records must be equally spaced on the x-axis regardless
                # of wall-clock time between them.  render_all_lines always uses time
                # when timestamps exist; use per-channel render_graph_line with
                # use_time_axis=False to get genuinely uniform spacing.
                _i_y_lo = float("inf")
                _i_y_hi = float("-inf")
                _i_from_steps: Dict[str, int] = {}
                for ck in vis_keys:
                    _i_n = self._loss_channel_length(ck)
                    _i_fs = int(_sf * _i_n)
                    _i_from_steps[ck] = _i_fs
                    _i_yr = store.channel_y_range(ck, from_step=_i_fs)
                    if _i_yr is not None:
                        _i_y_lo = min(_i_y_lo, float(_i_yr[0]))
                        _i_y_hi = max(_i_y_hi, float(_i_yr[1]))
                if math.isfinite(_i_y_lo) and math.isfinite(_i_y_hi) and _i_y_hi > _i_y_lo:
                    _i_overlay = Image.new("RGBA", (plot_w, plot_h), (0, 0, 0, 0))
                    _i_segments = 0
                    for _i_idx, ck in enumerate(vis_keys):
                        _i_arr = store.render_graph_line(
                            ck, plot_w, plot_h,
                            _i_y_lo, _i_y_hi,
                            from_step=int(_i_from_steps.get(ck, 0)),
                            color=vis_colors[_i_idx],
                            use_time_axis=False,
                        )
                        if _i_arr is None:
                            continue
                        _i_overlay = Image.alpha_composite(
                            _i_overlay, Image.fromarray(_i_arr, "RGBA")
                        )
                        _i_segments += 1
                    if _i_segments > 0:
                        result = {
                            "overlay": np.asarray(_i_overlay, dtype=np.uint8),
                            "y_min": float(_i_y_lo),
                            "y_max": float(_i_y_hi),
                            "t_min": 0.0,
                            "t_max": 0.0,
                            "segments": int(_i_segments),
                            "t_origin": 0.0,
                        }
            else:
                time_arrays: Dict[str, np.ndarray] = {}
                time_lo: Optional[float] = None
                time_hi: Optional[float] = None
                for ck in vis_keys:
                    ts_tensor = store.get_ts_array(ck)
                    if int(ts_tensor.numel()) <= 0:
                        continue
                    arr = ts_tensor.cpu().numpy()
                    time_arrays[ck] = arr
                    arr_lo = float(arr[0])
                    arr_hi = float(arr[-1])
                    time_lo = arr_lo if time_lo is None else min(time_lo, arr_lo)
                    time_hi = arr_hi if time_hi is None else max(time_hi, arr_hi)
                    if ck == ref_sid:
                        ref_ts = arr
                if time_arrays and time_lo is not None and time_hi is not None:
                    t_start = float(time_lo)
                    t_end = float(time_hi)
                    t_view_min = t_start + (_sf * (t_end - t_start))
                    if t_end <= t_view_min + 1e-9:
                        t_end = t_view_min + 1.0
                    y_lo = float("inf")
                    y_hi = float("-inf")
                    from_steps: Dict[str, int] = {}
                    for ck in vis_keys:
                        arr = time_arrays.get(ck)
                        if arr is None or arr.size <= 0:
                            continue
                        from_step = int(np.searchsorted(arr, t_view_min, side="left"))
                        from_steps[ck] = from_step
                        yr = store.channel_y_range(ck, from_step=from_step)
                        if yr is None:
                            continue
                        y_lo = min(y_lo, float(yr[0]))
                        y_hi = max(y_hi, float(yr[1]))
                    if y_hi > y_lo:
                        overlay_im = Image.new("RGBA", (plot_w, plot_h), (0, 0, 0, 0))
                        segments = 0
                        for i, ck in enumerate(vis_keys):
                            overlay_arr = store.render_graph_line(
                                ck,
                                plot_w,
                                plot_h,
                                y_lo,
                                y_hi,
                                t_min=t_view_min,
                                t_max=t_end,
                                color=vis_colors[i],
                                from_step=int(from_steps.get(ck, 0)),
                                use_time_axis=True,
                            )
                            if overlay_arr is None:
                                continue
                            overlay_im = Image.alpha_composite(overlay_im, Image.fromarray(overlay_arr, "RGBA"))
                            segments += 1
                        if segments > 0:
                            result = {
                                "overlay": np.asarray(overlay_im, dtype=np.uint8),
                                "y_min": float(y_lo),
                                "y_max": float(y_hi),
                                "t_min": float(t_view_min),
                                "t_max": float(t_end),
                                "segments": int(segments),
                                "t_origin": float(t_start),
                            }

        if result is None or result["segments"] == 0:
            draw.text((mx0, my0 + plot_h // 2 - 4), "no data yet", fill=(80, 88, 100), font=font)
            im.paste(loss_im, (0, 0))
            if pipeline_W > 0:
                try:
                    canvas_arr = self._render_pipeline_graph_canvas(pipeline_W, H)
                    im.paste(Image.fromarray(canvas_arr), (W, 0))
                except Exception:
                    pass
            return np.asarray(im, dtype=np.uint8)

        y_min = result["y_min"]
        y_max = result["y_max"]
        t_min_view = float(result.get("t_min", 0.0))
        t_max_view = float(result.get("t_max", 0.0))
        t_origin = float(result.get("t_origin", t_min_view))

        def _y_px(v: float) -> int:
            frac = (float(v) - y_min) / (y_max - y_min)
            return int(my1 - frac * plot_h)

        def _time_view_x(ts_value: float) -> int:
            if t_max_view <= t_min_view + 1e-9:
                return mx1
            frac = (float(ts_value) - t_min_view) / max(1e-9, t_max_view - t_min_view)
            return max(mx0, min(mx1, int(mx0 + frac * plot_w)))

        def _count_to_x(count_value: int) -> int:
            if axis_mode == "t":
                if ref_ts is None or len(ref_ts) == 0:
                    return mx1
                idx = max(0, min(len(ref_ts) - 1, int(count_value) - 1))
                return _time_view_x(float(ref_ts[idx]))
            total_n = max(1, int(ref_total))
            frac = float(max(0, min(total_n, int(count_value)))) / float(total_n)
            return _view_x(frac)

        # -- Cache-region and 2x-history overlay bands ----------------------
        try:
            if self._loss_count_at_snap_deque:
                _scl = list(self._loss_count_at_snap_deque)
                oldest_counts = _scl[0]
                if ref_sid is not None and ref_sid in oldest_counts:
                    cache_x0 = _count_to_x(int(oldest_counts[ref_sid]))
                    cache_w = max(0, mx1 - cache_x0)
                    hist_x0 = max(mx0, cache_x0 - cache_w * 2)
                    _ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                    _od = ImageDraw.Draw(_ov)
                    if hist_x0 < cache_x0:
                        _od.rectangle(
                            [(hist_x0, my0), (cache_x0, my1)], fill=(80, 60, 0, 38)
                        )
                    if cache_x0 < mx1:
                        _od.rectangle(
                            [(cache_x0, my0), (mx1, my1)], fill=(0, 80, 100, 50)
                        )
                    loss_im = Image.alpha_composite(loss_im.convert("RGBA"), _ov).convert("RGB")
                    draw = ImageDraw.Draw(loss_im)
        except Exception:
            pass

        # Horizontal gridlines + Y axis labels
        for frac, alpha in ((0.0, 1), (0.25, 0), (0.5, 0), (0.75, 0), (1.0, 1)):
            gy = int(my1 - frac * plot_h)
            draw.line([(mx0, gy), (mx1, gy)], fill=(34, 40, 50) if alpha == 0 else (55, 62, 74))
            lv = y_min + frac * (y_max - y_min)
            draw.text((2, gy - 5), f"{lv:.3f}", fill=(110, 120, 136), font=font)

        # -- Composite the C-rendered RGBA line overlay ---------------------
        overlay_arr = result["overlay"]  # (plot_h, plot_w, 4) uint8 RGBA
        overlay_img = Image.fromarray(overlay_arr, "RGBA")
        # Paste into the plot region of loss_im
        loss_rgba = loss_im.convert("RGBA")
        # The overlay covers (mx0..mx0+plot_w, my0..my0+plot_h)
        loss_rgba.paste(
            Image.alpha_composite(
                loss_rgba.crop((mx0, my0, mx0 + plot_w, my0 + plot_h)),
                overlay_img,
            ),
            (mx0, my0),
        )
        loss_im = loss_rgba.convert("RGB")
        draw = ImageDraw.Draw(loss_im)

        # -- Disk-save checkpoint markers -----------------------------------
        try:
            if self._disk_save_loss_counts:
                if ref_sid is not None:
                    for _dsc in self._disk_save_loss_counts:
                        if ref_sid in _dsc:
                            _dx = _count_to_x(int(_dsc[ref_sid]))
                            draw.line([(_dx, my0), (_dx, my1)], fill=(180, 150, 30), width=1)
                            draw.text((_dx + 1, my1 - 10), "v", fill=(200, 170, 40), font=font)
        except Exception:
            pass

        # -- Scrub cursor line ----------------------------------------------
        try:
            if self._scrub_offset > 0 and self._loss_count_at_snap_deque:
                _sl = list(self._loss_count_at_snap_deque)
                _slen = len(_sl)
                _off = min(int(self._scrub_offset), _slen)
                _cidx = _slen - _off
                if 0 <= _cidx < _slen:
                    _cc = _sl[_cidx]
                    if ref_sid is not None and ref_sid in _cc:
                        cx = _count_to_x(int(_cc[ref_sid]))
                        draw.line([(cx, my0), (cx, my1)], fill=(200, 150, 255), width=2)
                        draw.text(
                            (cx + 3, my0 + 2),
                            f"<{self._scrub_offset}",
                            fill=(200, 150, 255), font=font,
                        )
        except Exception:
            pass

        # Legend (horizontal, top of graph) -- clickable channel toggles
        lx = mx0
        _legend_boxes: List[Tuple[str, int, int, int, int]] = []
        _n_legend = len(sorted_keys)
        for _leg_idx, ck in enumerate(sorted_keys):
            ch_len = self._loss_channel_length(ck)
            if ch_len == 0:
                continue
            vis = self._loss_channel_visible.get(ck, True)
            color = _channel_color(_leg_idx, _n_legend)
            lat = store.latest(ck) if store is not None else None
            last_v = lat.loss if lat is not None else float("nan")
            label = f"{ck}={last_v:.4f}" if math.isfinite(last_v) else ck
            if vis:
                draw.rectangle([(lx, toolbar_h + 2), (lx + 7, toolbar_h + 9)], fill=color)
                draw.text((lx + 10, toolbar_h + 1), label, fill=color, font=font)
            else:
                dim = tuple(max(30, c // 3) for c in color)
                draw.rectangle([(lx, toolbar_h + 2), (lx + 7, toolbar_h + 9)], fill=dim, outline=(60, 60, 60))
                draw.text((lx + 10, toolbar_h + 1), label, fill=dim, font=font)
            entry_w = max(56, len(label) * 6 + 18)
            _legend_boxes.append((ck, lx, toolbar_h, lx + entry_w, toolbar_h + 12))
            lx += entry_w
            if lx > W - 60:
                break
        self._legend_hit_boxes = _legend_boxes

        # X-axis grid and labels
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            gx = int(mx0 + frac * plot_w)
            draw.line([(gx, my0), (gx, my1)], fill=(26, 30, 36))
            if axis_mode == "t":
                tick_ts = t_min_view + (frac * max(0.0, t_max_view - t_min_view))
                label = self._format_graph_elapsed(tick_ts - t_origin)
            else:
                start_idx = int(_sf * max(0, ref_total))
                label = str(int(start_idx + (frac * max(0, ref_total - start_idx))))
            draw.text((max(0, gx - (len(label) * 3)), my1 + 2), label, fill=(110, 120, 136), font=font)
        draw.text((mx1 - 36, H - 12), f"x={axis_mode}", fill=(124, 134, 146), font=font)

        im.paste(loss_im, (0, 0))

        # Composite pipeline graph canvas into the right portion when plan is loaded.
        if pipeline_W > 0:
            try:
                canvas_arr = self._render_pipeline_graph_canvas(pipeline_W, H)
                canvas_img = Image.fromarray(canvas_arr)
                im.paste(canvas_img, (W, 0))
            except Exception:
                pass

        return np.asarray(im, dtype=np.uint8)

    def _render_pipeline_graph_canvas(self, W: int, H: int) -> "np.ndarray":
        """Render pipeline node/edge graph from plan+runtime data.

        Returns an (H, W, 3) uint8 RGB image showing nodes grouped by group_id
        in horizontal lanes, colour-coded by runtime execution status.
        """
        out = np.full((H, W, 3), (14, 16, 20), dtype=np.uint8)
        plan = self._graph_plan_snapshot
        if not isinstance(plan, dict) or not plan.get("nodes"):
            return out
        try:
            from PIL import Image as _PGIL, ImageDraw as _PGDraw, ImageFont as _PGFont
        except Exception:
            return out
        try:
            runtime = self._graph_runtime_snapshot or {}
            active_ids: set = set(runtime.get("active_node_ids", []))
            # Build node-status map from execution events (most recent event per node wins).
            node_status: Dict[str, str] = {}
            for ev in self._graph_execution_events:
                nid = str(ev.get("node_id", ""))
                status = str(ev.get("status", ""))
                if nid and status:
                    node_status[nid] = status
            # Also fold in node_states from runtime snapshot when available.
            for ns in runtime.get("node_states", []):
                if isinstance(ns, dict):
                    nid = str(ns.get("node_id", ""))
                    st = str(ns.get("status", ""))
                    if nid and st:
                        node_status[nid] = st

            _STATUS_COLORS = {
                "active":    (0,   190, 220),
                "running":   (0,   190, 220),
                "ran":       (52,  168,  83),
                "completed": (52,  168,  83),
                "ok":        (52,  168,  83),
                "failed":    (210,  50,  50),
                "skipped":   (55,   62,  74),
                "pending":   (70,   80,  95),
            }

            # Group nodes by group_id preserving a preferred display order.
            _GROUP_ORDER = ["bootstrap", "build", "data", "vocab", "train", "gates", "housekeeping"]
            groups: Dict[str, List[Any]] = {}
            for n in plan.get("nodes", []):
                gid = str(n.get("group_id", "other"))
                if gid not in groups:
                    groups[gid] = []
                groups[gid].append(n)

            all_groups = [g for g in _GROUP_ORDER if g in groups]
            for g in sorted(groups.keys()):
                if g not in all_groups:
                    all_groups.append(g)

            if not all_groups:
                return out

            im = _PGIL.new("RGB", (W, H), (14, 16, 20))
            draw = _PGDraw.Draw(im)
            font = _PGFont.load_default()

            num_lanes = len(all_groups)
            lane_h = max(10, H // num_lanes)
            label_col_w = max(36, min(52, W // 4))
            node_area_w = W - label_col_w - 2

            exec_state = str(runtime.get("execution_state", ""))
            state_color = (80, 190, 80) if exec_state == "running" else (160, 170, 180)
            draw.text((2, 1), f"graph:{exec_state or 'idle'}", fill=state_color, font=font)

            for lane_idx, gid in enumerate(all_groups):
                lane_y0 = lane_idx * lane_h
                lane_y1 = min(H, lane_y0 + lane_h)
                bg = (20, 24, 30) if lane_idx % 2 == 0 else (17, 21, 26)
                draw.rectangle([(0, lane_y0), (W - 1, lane_y1 - 1)], fill=bg)

                # Group label (abbreviated)
                short_gid = gid[:7]
                draw.text((2, lane_y0 + max(0, (lane_h - 9) // 2)), short_gid,
                          fill=(110, 125, 145), font=font)

                lane_nodes = groups[gid]
                n_nodes = len(lane_nodes)
                if n_nodes == 0:
                    continue

                pad = 1
                node_w = max(6, (node_area_w - pad * (n_nodes + 1)) // n_nodes)
                node_h = max(5, lane_h - 4)

                for ni, node in enumerate(lane_nodes):
                    nid = str(node.get("node_id", ""))
                    nx0 = label_col_w + pad + ni * (node_w + pad)
                    nx1 = min(W - 1, nx0 + node_w)
                    ny0 = lane_y0 + 2
                    ny1 = min(lane_y1 - 1, ny0 + node_h)

                    if nid in active_ids:
                        color = _STATUS_COLORS["active"]
                    else:
                        st = node_status.get(nid, "pending")
                        color = _STATUS_COLORS.get(st, _STATUS_COLORS["pending"])

                    draw.rectangle([(nx0, ny0), (nx1, ny1)], fill=color, outline=(40, 48, 60))

                    # Abbreviated label (fits in node width)
                    lbl = str(node.get("label", nid))
                    # Remove common prefixes to shorten
                    for pfx in ("Stage ", "Build ", "Gate ", "Init ", "Vocab "):
                        if lbl.startswith(pfx):
                            lbl = lbl[len(pfx):]
                            break
                    char_w = max(1, node_w // 6)
                    short = lbl[:char_w] if char_w >= 1 else ""
                    if short:
                        draw.text((nx0 + 1, ny0 + 1), short, fill=(230, 238, 248), font=font)

            return np.asarray(im, dtype=np.uint8)
        except Exception:
            return out

    def enqueue_frame(self, frame_dict: dict):
        """Receive a frame dict — push images to scrub ring if present, store text."""
        ring_cursor: Optional[int] = None
        images = frame_dict.get("images")
        if (images is not None and isinstance(images, list) and len(images) >= 3
                and self._scrub_ring is not None):
            target_rgb = np.asarray(images[0], dtype=np.uint8)
            input_rgb = np.asarray(images[1], dtype=np.uint8)
            output_rgb = np.asarray(images[2], dtype=np.uint8)
            thumb_stack = _resize_rgb_nearest(
                self._weight_map_rgb,
                (SCRUB_THUMB_H * SCRUB_NUM_THUMBS, SCRUB_THUMB_W),
            )
            weight_tiles = tuple(
                np.ascontiguousarray(
                    thumb_stack[idx * SCRUB_THUMB_H : (idx + 1) * SCRUB_THUMB_H].copy()
                )
                for idx in range(SCRUB_NUM_THUMBS)
            )
            h, w = target_rgb.shape[0], target_rgb.shape[1]
            flags = SCRUB_FLAG_HAS_IMAGE | SCRUB_FLAG_HAS_TARGET | SCRUB_FLAG_HAS_OUTPUT | SCRUB_FLAG_HAS_THUMBS
            ring_cursor = self._scrub_ring.push(
                step=0, round_id=0, ts=time.time(), loss=0.0,
                channel_key="preview", flags=flags,
                image_w=w, image_h=h,
                training_image=input_rgb,
                output_image=output_rgb,
                target_data=target_rgb,
                thumb0=weight_tiles[0],
                thumb1=weight_tiles[1],
                thumb2=weight_tiles[2],
            )
        self._stage_frame_text(
            ring_cursor,
            caption=str(frame_dict.get("caption", "")),
            titles=frame_dict.get("titles", self._panel_titles),
            rows=frame_dict.get("rows", self._panel_rows),
        )

    def update(
        self,
        clean_img: torch.Tensor,
        input_img: torch.Tensor,
        output_img: torch.Tensor,
        caption: str,
        panel_titles: Optional[Sequence[str]] = None,
        panel_rows: Optional[Sequence[Sequence[str]]] = None,
        frame_losses: Optional[Dict[str, float]] = None,
    ):
        if not self.enabled:
            return
        self._init()
        if not self._ready:
            return
        self._poll_events()
        if self._stop_requested or (not self._ready):
            return

        ring = self._scrub_ring
        if ring is None:
            return

        target_rgb = _tensor_to_rgb_u8_image(clean_img)
        input_rgb = _tensor_to_rgb_u8_image(input_img)
        output_rgb = _tensor_to_rgb_u8_image(output_img)
        thumb_stack = _resize_rgb_nearest(
            self._weight_map_rgb,
            (SCRUB_THUMB_H * SCRUB_NUM_THUMBS, SCRUB_THUMB_W),
        )
        weight_tiles = tuple(
            np.ascontiguousarray(
                thumb_stack[idx * SCRUB_THUMB_H : (idx + 1) * SCRUB_THUMB_H].copy()
            )
            for idx in range(SCRUB_NUM_THUMBS)
        )
        h, w = target_rgb.shape[0], target_rgb.shape[1]

        flags = SCRUB_FLAG_HAS_IMAGE | SCRUB_FLAG_HAS_TARGET | SCRUB_FLAG_HAS_OUTPUT | SCRUB_FLAG_HAS_THUMBS
        ring_cursor = ring.push(
            step=0,
            round_id=0,
            ts=time.time(),
            loss=0.0,
            channel_key="viewer",
            flags=flags,
            image_w=w,
            image_h=h,
            training_image=input_rgb,
            output_image=output_rgb,
            target_data=target_rgb,
            thumb0=weight_tiles[0],
            thumb1=weight_tiles[1],
            thumb2=weight_tiles[2],
        )
        self._stage_frame_text(
            ring_cursor,
            caption=str(caption),
            titles=panel_titles,
            rows=panel_rows,
        )

    def close(self):
        if self._ready:
            try:
                if self._gl is not None and self._textures is not None:
                    tex_ids = (
                        list(self._textures.get("img", []))
                        + list(self._textures.get("text", []))
                        + [self._textures.get("bar", 0)]
                        + [self._textures.get("graph", 0)]
                    )
                    tex_ids = [int(t) for t in tex_ids if int(t) > 0]
                    if len(tex_ids) > 0:
                        self._gl.glDeleteTextures(tex_ids)
            except Exception:
                pass
            try:
                if self._pygame is not None:
                    self._pygame.display.quit()
                    self._pygame.quit()
            except Exception:
                pass
        self._ready = False
        self._textures = None
        self._pygame = None
        self._gl = None


# ---------------------------------------------------------------------------
# IPC layer -- decoupled GUI <-> training communication
# ---------------------------------------------------------------------------
# The viewer window runs in a separate process (launched by wav_ml_gui_main.py).
# The training pipeline talks to it over a localhost TCP connection managed by
# multiprocessing.connection (length-prefixed pickle).  Two classes:
#
#   ViewerIPCServer  -- runs in the GUI process; receives data, dispatches to viewer
#   ViewerIPCProxy   -- runs in the training process; same API as the viewer
#
# Message flow:
#   training -> GUI :  loss, frame, checkpoint_saved, cycle_roster, weight_map, ...
#   GUI -> training :  status (stop_requested, gate_override, cycle_selected)
# ---------------------------------------------------------------------------

_IPC_AUTHKEY = b'nodus_viewer_v1'


class ViewerIPCServer:
    """Runs in the GUI process alongside the real ``_TransformerStatusOpenGLViewer``.

    Accepts connections from training processes and dispatches incoming messages
    to the local viewer.  Periodically sends status updates back so the training
    process can detect stop requests, cycle selection changes, etc.
    """

    def __init__(
        self,
        viewer: "_TransformerStatusOpenGLViewer",
        port: int = 0,
        port_file: Optional[str] = None,
    ):
        from multiprocessing.connection import Listener

        self._viewer = viewer
        self._listener = Listener(
            ("localhost", int(port)), family="AF_INET", authkey=_IPC_AUTHKEY
        )
        self._port: int = self._listener.address[1]
        if port_file:
            Path(port_file).write_text(str(self._port), encoding="utf-8")
            print(
                f"[viewer-ipc] listening on port {self._port}, wrote {port_file}",
                flush=True,
            )
        else:
            print(f"[viewer-ipc] listening on port {self._port}", flush=True)

        self._conn: Optional[Any] = None
        self._accept_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._stopped = False
        self._last_status_t: float = 0.0

    @property
    def port(self) -> int:
        return self._port

    @property
    def has_connection(self) -> bool:
        with self._lock:
            return self._conn is not None

    def start(self) -> None:
        """Begin accepting connections in a background thread."""
        self._accept_thread = threading.Thread(
            target=self._accept_loop, name="viewer-ipc-accept", daemon=True
        )
        self._accept_thread.start()

    def _accept_loop(self) -> None:
        while not self._stopped:
            try:
                conn = self._listener.accept()
                with self._lock:
                    # Close any previous connection (e.g. previous pipeline run).
                    if self._conn is not None:
                        try:
                            self._conn.close()
                        except Exception:
                            pass
                    self._conn = conn
                print("[viewer-ipc] training process connected", flush=True)
                # Reset shutdown state for the new connection
                self._viewer._shutdown_save = None
                self._viewer._top_bar_dirty = True
            except OSError:
                break
            except Exception as e:
                if not self._stopped:
                    print(f"[viewer-ipc] accept error: {e}", flush=True)
                    time.sleep(0.5)

    def poll(self) -> None:
        """Non-blocking: drain inbound messages and send periodic status updates.

        Call this from the GUI main loop on every iteration.
        """
        with self._lock:
            conn = self._conn
        if conn is None:
            return

        # Drain all available messages from training -> GUI
        try:
            while conn.poll(0):
                msg = conn.recv()
                self._dispatch(msg)
        except (EOFError, OSError):
            with self._lock:
                self._conn = None
            # Reset shutdown state so the GUI is ready for a new training run
            self._viewer._shutdown_save = None
            self._viewer._top_bar_dirty = True
            print("[viewer-ipc] training process disconnected", flush=True)
            return
        except Exception as e:
            print(f"[viewer-ipc] recv error: {e}", flush=True)
            return

        # Send status back at most every 100 ms to avoid flooding
        now = time.time()
        if now - self._last_status_t < 0.1:
            return
        self._last_status_t = now
        try:
            is_stopping = (
                self._viewer.stop_requested()
                or self._viewer.shutdown_save() is not None
            )
            is_paused = bool(self._viewer._paused)
            is_preview = bool(self._viewer._preview_enabled)
            is_scrub_editor = bool(self._viewer._scrub_editor_enabled)
            status = {
                "type": "status",
                "stop_requested": is_stopping,
                "paused": is_paused,
                "gate_override": self._viewer.gate_override_enabled(),
                "preview_enabled": is_preview,
                "scrub_editor_enabled": is_scrub_editor,
                "cycle_selected": list(self._viewer._cycle_selected),
            }
            conn.send(status)
            cmd = "stop" if bool(status["stop_requested"]) else ("pause" if is_paused else "resume")
            run_control = RunControlPayload(
                command=cmd,
                selected_cycle_ids=self._viewer.selected_cycle_ids(),
                gate_override=bool(status["gate_override"]),
                preview_enabled=is_preview,
                scrub_editor_enabled=is_scrub_editor,
                metadata={
                    "source": "viewer_ipc_status_loop",
                    "save": self._viewer.shutdown_save(),
                },
            )
            conn.send(make_envelope(MESSAGE_TYPE_RUN_CONTROL, run_control).to_dict())
        except (EOFError, OSError):
            with self._lock:
                self._conn = None
            self._viewer._shutdown_save = None
            self._viewer._top_bar_dirty = True
        except Exception:
            pass

    def _dispatch(self, msg: dict) -> None:
        if is_protocol_envelope_message(msg):
            envelope, payload = parse_envelope(msg)
            if envelope.message_type == MESSAGE_TYPE_WORKER_HELLO:
                if hasattr(self._viewer, "set_training_graph_worker_hello"):
                    self._viewer.set_training_graph_worker_hello(payload.to_dict())
                # Clear stale stop state from any previous run so the new
                # training process is not immediately told to stop.
                self._viewer._stop_requested = False
                self._viewer._shutdown_save = None
                self._viewer._top_bar_dirty = True
                # Loss data is owned by the SaveRestoreNode.  Clear the
                # viewer's local graph caches so the pull-model repopulates
                # them from the authoritative source without duplicates.
                if self._viewer._loss_store is not None:
                    self._viewer._loss_store.clear()
                self._viewer._sr_channel_registry.clear()
                self._viewer._graph_dirty = True
            elif envelope.message_type == MESSAGE_TYPE_PLAN_SNAPSHOT:
                if hasattr(self._viewer, "set_training_graph_plan"):
                    self._viewer.set_training_graph_plan(payload.plan.to_dict())
            elif envelope.message_type == MESSAGE_TYPE_RUNTIME_SNAPSHOT:
                if hasattr(self._viewer, "set_training_graph_runtime"):
                    self._viewer.set_training_graph_runtime(payload.to_dict())
            elif envelope.message_type == MESSAGE_TYPE_EXECUTION_EVENT:
                if hasattr(self._viewer, "append_training_graph_event"):
                    self._viewer.append_training_graph_event(payload.to_dict())
            return

        t = msg.get("type")
        v = self._viewer
        if t == "loss":
            # Loss values are owned exclusively by the SaveRestoreNode.
            # The GUI obtains them via pull-model queries; ignore direct pushes.
            pass
        elif t == "frame_signal":
            stage_fn = getattr(v, "_stage_frame_text", None)
            if callable(stage_fn):
                stage_fn(
                    msg.get("ring_cursor"),
                    caption=str(msg.get("caption", "")),
                    titles=msg.get("titles", ["target", "input", "output"]),
                    rows=msg.get("rows", [[], [], []]),
                )
        elif t == "checkpoint_saved":
            v.notify_pipeline_checkpoint_saved()
        elif t == "checkpoint_at_walltime":
            v.notify_checkpoint_at_walltime(msg["wall_ts"])
        elif t == "cycle_roster":
            v.set_cycle_roster(msg["total_cycles"])
        elif t == "checkpoint_backup_dir":
            v.set_checkpoint_backup_dir(msg["path"])
        elif t == "trim_graph":
            v.trim_graph_to_first_checkpoint()
        elif t == "weight_map_rgb":
            # Training side pre-rendered the weight map image; paste it directly.
            v._weight_map_rgb = msg["rgb"]
            v._sidebar_dirty = True
        elif t == "weight_snapshot":
            v._weight_snapshot_deque.append(msg.get("weight_rgb"))
            v._loss_count_at_snap_deque.append(msg.get("loss_counts"))
        elif t == "exit":
            print("[viewer-ipc] training process exited cleanly", flush=True)
        # -- Pull-model: notifications from SaveRestoreNode ----------------
        elif t is not None and t.startswith("notify_"):
            handler = getattr(v, "_on_sr_notification", None)
            if callable(handler):
                try:
                    handler(msg)
                except Exception:
                    pass
        # -- Pull-model: responses to our queries --------------------------
        elif t is not None and t.startswith("resp_"):
            handler = getattr(v, "_on_sr_response", None)
            if callable(handler):
                try:
                    handler(msg)
                except Exception:
                    pass

    # -- Pull-model: send queries to the training side ---------------------

    def send_query(self, query: dict) -> None:
        """Send a query_* message to the training process.

        The response will arrive asynchronously via poll() -> _dispatch()
        and be routed to the viewer's _on_sr_response handler.
        """
        with self._lock:
            conn = self._conn
        if conn is None:
            return
        try:
            conn.send(query)
        except (EOFError, OSError):
            pass
        except Exception:
            pass

    def query_channel_list(self) -> None:
        self.send_query({"type": "query_channel_list"})

    def query_latest_result(self, channel_key: str) -> None:
        self.send_query({
            "type": "query_latest_result",
            "channel_key": str(channel_key),
        })

    def query_tm_summary(self) -> None:
        self.send_query({"type": "query_tm_summary"})

    def query_checkpoint_info(self) -> None:
        self.send_query({"type": "query_checkpoint_info"})

    def stop(self) -> None:
        self._stopped = True
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None
        try:
            self._listener.close()
        except Exception:
            pass


class ViewerIPCProxy:
    """Runs in the training process -- drop-in replacement for
    ``_TransformerStatusOpenGLViewer`` that forwards all calls over IPC to the
    standalone GUI process.

    The proxy presents the same public API so the pipeline code does not need to
    know whether the viewer is local or remote.
    """

    def __init__(
        self,
        port_file: Optional[str] = None,
        port: int = 0,
        timeout: float = 30.0,
        *,
        enabled: bool = True,
        image_hw: Tuple[int, int] = (64, 64),
        scale: int = 1,
        cycle_slots: int = 0,
    ):
        self.enabled = bool(enabled)
        self.image_h = max(8, int(image_hw[0]))
        self.image_w = max(8, int(image_hw[1]))
        self._stop_flag = False
        self._shutdown_save: Optional[bool] = None
        self._gate_override = False
        self._paused = False
        self._preview_enabled = True
        self._scrub_editor_enabled = True
        self._cycle_selected: List[bool] = [True] * max(0, cycle_slots)
        self._last_run_control = RunControlPayload(
            command="resume",
            selected_cycle_ids=[int(i + 1) for i in range(max(0, cycle_slots))],
            gate_override=False,
        )
        self._pending_plan_apply: Optional[PlanApplyPayload] = None
        self._conn: Optional[Any] = None
        self._connect_lock = threading.Lock()
        self._send_lock = threading.Lock()  # protects concurrent writes
        self._connected_once = False
        self._connection_lost = False
        self._port_file = str(port_file) if port_file else ""
        self._static_port = max(0, int(port))
        self._connect_timeout_s = max(0.0, float(timeout))
        self._reconnect_interval_s = 1.0
        self._next_reconnect_t = 0.0
        self._last_connected_port = 0
        self._on_restore: Optional[Callable] = None

        # Reference to SaveRestoreNode for pull-model query routing.
        # Set via set_save_restore_node() after construction.
        self._save_restore_node: Optional[Any] = None

        # Stub attributes that pipeline code may touch
        self._weight_snap_stride = 128
        self._weight_state_sparse_deque: deque = deque(maxlen=4)
        self._preview_work_queue_ref: Optional[Any] = None

        if not self.enabled:
            return

        if not self._connect(wait_for_port=bool(self._port_file), timeout=self._connect_timeout_s):
            self.enabled = False

        # Cross-process scrub ring handle (same shared memory as the GUI).
        self._scrub_ring: Optional[NodusScrubRing] = None
        if self.enabled:
            try:
                self._scrub_ring = NodusScrubRing.get_global()
            except Exception as e:
                print(f"[viewer-ipc] scrub ring init failed: {e}", flush=True)

    # -- internal helpers --------------------------------------------------

    def _resolve_port(
        self,
        *,
        wait_for_port: bool = False,
        timeout: float = 0.0,
        log_timeout: bool = False,
    ) -> int:
        actual_port = int(self._static_port)
        if self._port_file:
            pf = Path(self._port_file)
            deadline = time.time() + max(0.0, float(timeout))
            while True:
                if pf.exists():
                    try:
                        actual_port = int(pf.read_text(encoding="utf-8").strip())
                        break
                    except (ValueError, OSError):
                        pass
                if not wait_for_port or time.time() >= deadline:
                    if wait_for_port and log_timeout and actual_port <= 0:
                        print(
                            f"[viewer-ipc] timeout waiting for port file: {self._port_file}",
                            flush=True,
                        )
                    break
                time.sleep(0.25)
        return int(actual_port) if int(actual_port) > 0 else 0

    def _connect(
        self,
        *,
        wait_for_port: bool = False,
        timeout: float = 0.0,
        quiet: bool = False,
        reconnect: bool = False,
    ) -> bool:
        with self._connect_lock:
            if self._conn is not None:
                return True

            actual_port = self._resolve_port(
                wait_for_port=wait_for_port,
                timeout=timeout,
                log_timeout=not quiet,
            )
            if actual_port <= 0:
                if not quiet:
                    print("[viewer-ipc] no valid port", flush=True)
                return False

            import multiprocessing.connection as _mp_connection

            try:
                conn = _mp_connection.Client(
                    ("localhost", actual_port), family="AF_INET", authkey=_IPC_AUTHKEY
                )
            except Exception as e:
                if not quiet:
                    print(f"[viewer-ipc] connection failed: {e}", flush=True)
                return False

            prev = self._conn
            self._conn = conn
            if prev is not None and prev is not conn:
                try:
                    prev.close()
                except Exception:
                    pass
            self._connected_once = True
            self._connection_lost = False
            self._stop_flag = False
            self._shutdown_save = None
            self._paused = False
            self._preview_enabled = True
            self._scrub_editor_enabled = True
            self._last_connected_port = int(actual_port)
            self._next_reconnect_t = 0.0
            label = "reconnected" if reconnect else "connected"
            print(f"[viewer-ipc] {label} to GUI on port {actual_port}", flush=True)
            return True

    def _maybe_reconnect(self) -> None:
        if (not self.enabled) or self._conn is not None or (not self._connected_once):
            return
        now = time.time()
        if now < self._next_reconnect_t:
            return
        self._next_reconnect_t = now + self._reconnect_interval_s
        self._connect(reconnect=True, quiet=True)

    def _mark_connection_lost(self, reason: str = "") -> None:
        if self._connection_lost:
            return
        self._connection_lost = True
        self._stop_flag = False
        self._shutdown_save = None
        self._paused = False
        self._preview_enabled = False
        self._scrub_editor_enabled = False
        conn = self._conn
        self._conn = None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        self._next_reconnect_t = time.time() + self._reconnect_interval_s
        suffix = f": {reason}" if str(reason).strip() else ""
        print(
            f"[viewer-ipc] GUI connection lost; continuing without GUI control{suffix}; will retry",
            flush=True,
        )

    def _send(self, msg: dict) -> None:
        if not self.enabled:
            return
        if self._conn is None:
            self._maybe_reconnect()
            if self._conn is None:
                return
        with self._send_lock:
            try:
                self._conn.send(msg)
            except (EOFError, OSError):
                self._mark_connection_lost("send failed")
            except Exception:
                pass

    def _drain_status(self) -> None:
        """Non-blocking read of all available status messages from the GUI."""
        if not self.enabled:
            return
        if self._conn is None:
            self._maybe_reconnect()
            if self._conn is None:
                return
        if self._conn is None:
            return
        try:
            while self._conn.poll(0):
                msg = self._conn.recv()
                if is_protocol_envelope_message(msg):
                    envelope, payload = parse_envelope(msg)
                    if envelope.message_type == MESSAGE_TYPE_RUN_CONTROL:
                        self._last_run_control = payload
                        self._stop_flag = str(payload.command).lower() == "stop"
                        self._paused = str(payload.command).lower() == "pause"
                        self._gate_override = bool(payload.gate_override)
                        self._preview_enabled = bool(getattr(payload, "preview_enabled", True))
                        self._scrub_editor_enabled = bool(getattr(payload, "scrub_editor_enabled", True))
                        # Extract save preference from metadata
                        meta = getattr(payload, "metadata", {}) or {}
                        if self._stop_flag and "save" in meta:
                            self._shutdown_save = meta["save"]
                        if payload.selected_cycle_ids:
                            max_cycle = max(payload.selected_cycle_ids)
                            selected = [False] * max(max_cycle, len(self._cycle_selected))
                            for cycle_id in payload.selected_cycle_ids:
                                idx = int(cycle_id) - 1
                                if 0 <= idx < len(selected):
                                    selected[idx] = True
                            self._cycle_selected = selected
                    elif envelope.message_type == MESSAGE_TYPE_PLAN_APPLY:
                        self._pending_plan_apply = payload
                    continue

                t = msg.get("type")
                if t == "status":
                    self._stop_flag = msg.get("stop_requested", False)
                    self._gate_override = msg.get("gate_override", False)
                    self._paused = msg.get("paused", False)
                    self._preview_enabled = msg.get("preview_enabled", True)
                    self._scrub_editor_enabled = msg.get("scrub_editor_enabled", True)
                    cs = msg.get("cycle_selected")
                    if cs is not None:
                        self._cycle_selected = list(cs)
                    self._last_run_control = RunControlPayload(
                        command="stop" if bool(self._stop_flag) else ("pause" if bool(self._paused) else "resume"),
                        selected_cycle_ids=self.selected_cycle_ids(),
                        gate_override=bool(self._gate_override),
                        preview_enabled=bool(self._preview_enabled),
                        scrub_editor_enabled=bool(self._scrub_editor_enabled),
                        metadata={"source": "legacy_status"},
                    )
                elif t == "restore":
                    request_restore = getattr(self._save_restore_node, "request_restore", None)
                    if (
                        callable(request_restore)
                        and ("round_id" in msg)
                        and ("cycle" in msg)
                    ):
                        try:
                            request_restore(int(msg.get("round_id", 0)), int(msg.get("cycle", 0)))
                        except Exception:
                            pass
                    elif self._on_restore is not None:
                        try:
                            self._on_restore(int(msg.get("offset", 0)))
                        except Exception:
                            pass
                elif t is not None and t.startswith("query_"):
                    # Pull-model: route query to SaveRestoreNode, send response
                    self._handle_gui_query(msg)
        except (EOFError, OSError):
            self._mark_connection_lost("status read failed")
        except Exception:
            pass

    # -- public API (mirrors _TransformerStatusOpenGLViewer) ---------------

    def set_save_restore_node(self, node: Any) -> None:
        """Inject the SaveRestoreNode reference for pull-model query routing."""
        self._save_restore_node = node

    def _handle_gui_query(self, query: dict) -> None:
        """Route a query_* message from the GUI to the SaveRestoreNode."""
        node = self._save_restore_node
        if node is None:
            return
        handler = getattr(node, "handle_query", None)
        if not callable(handler):
            return
        try:
            response = handler(query)
            if response is not None:
                self._send(response)
        except Exception as exc:
            print(f"[viewer-ipc] query handler error: {exc}", flush=True)

    def send_notification(self, notification: dict) -> None:
        """Send a lightweight notify_* message to the GUI."""
        self._send(notification)

    def pump(self):
        self._drain_status()

    def stop_requested(self) -> bool:
        self._drain_status()
        return self._stop_flag

    def shutdown_save(self) -> Optional[bool]:
        """Return the save preference for the current shutdown, or None if no shutdown."""
        return self._shutdown_save

    def update_loss(self, channel_key: str, loss: float, aux: float = 0.0, ts: float = 0.0):
        # Loss values are owned exclusively by the SaveRestoreNode.
        # The GUI pulls them via the IPC query protocol; no direct push.
        pass

    def _current_weight_thumb_tiles(self) -> Tuple[Optional[Tuple[np.ndarray, ...]], Optional[str]]:
        node = self._save_restore_node
        if node is None:
            return None, None
        getter = getattr(node, "current_weight_thumb_tiles", None)
        if not callable(getter):
            return None, None
        try:
            payload = getter()
        except Exception:
            return None, None
        if not isinstance(payload, dict):
            return None, None
        tiles = payload.get("tiles")
        if not isinstance(tiles, tuple) and not isinstance(tiles, list):
            return None, None
        if len(tiles) != SCRUB_NUM_THUMBS:
            return None, None
        return (
            tuple(np.ascontiguousarray(np.asarray(tile, dtype=np.uint8)) for tile in tiles),
            (str(payload.get("model", "")).strip() or None),
        )

    def enqueue_frame(self, frame_dict: dict):
        """Push images to scrub ring if present, send text metadata via IPC."""
        if not self.preview_enabled():
            return
        images = frame_dict.get("images")
        ring = self._scrub_ring
        ring_cursor: Optional[int] = None
        if (images is not None and isinstance(images, list) and len(images) >= 3
                and ring is not None):
            target_rgb = np.asarray(images[0], dtype=np.uint8)
            input_rgb = np.asarray(images[1], dtype=np.uint8)
            output_rgb = np.asarray(images[2], dtype=np.uint8)
            weight_tiles, _weight_model = self._current_weight_thumb_tiles()
            h, w = target_rgb.shape[0], target_rgb.shape[1]
            flags = SCRUB_FLAG_HAS_IMAGE | SCRUB_FLAG_HAS_TARGET | SCRUB_FLAG_HAS_OUTPUT
            if weight_tiles is not None:
                flags |= SCRUB_FLAG_HAS_THUMBS
            ring_cursor = ring.push(
                step=0, round_id=0, ts=time.time(), loss=0.0,
                channel_key="preview", flags=flags,
                image_w=w, image_h=h,
                training_image=input_rgb,
                output_image=output_rgb,
                target_data=target_rgb,
                thumb0=(weight_tiles[0] if weight_tiles is not None else None),
                thumb1=(weight_tiles[1] if weight_tiles is not None else None),
                thumb2=(weight_tiles[2] if weight_tiles is not None else None),
            )
        self._send({
            "type": "frame_signal",
            "ring_cursor": (int(ring_cursor) if ring_cursor is not None else None),
            "caption": str(frame_dict.get("caption", "")),
            "titles": list(frame_dict.get("titles", ["target", "input", "output"])),
            "rows": [list(r) for r in frame_dict.get("rows", [[], [], []])],
        })

    def update(
        self,
        clean_img,
        input_img,
        output_img,
        caption: str,
        panel_titles=None,
        panel_rows=None,
        frame_losses=None,
    ):
        ring = self._scrub_ring
        if ring is None:
            return

        target_rgb = _tensor_to_rgb_u8_image(clean_img)
        input_rgb = _tensor_to_rgb_u8_image(input_img)
        output_rgb = _tensor_to_rgb_u8_image(output_img)
        weight_tiles, _weight_model = self._current_weight_thumb_tiles()
        h, w = target_rgb.shape[0], target_rgb.shape[1]

        flags = SCRUB_FLAG_HAS_IMAGE | SCRUB_FLAG_HAS_TARGET | SCRUB_FLAG_HAS_OUTPUT
        if weight_tiles is not None:
            flags |= SCRUB_FLAG_HAS_THUMBS
        ring_cursor = ring.push(
            step=0,
            round_id=0,
            ts=time.time(),
            loss=0.0,
            channel_key="viewer",
            flags=flags,
            image_w=w,
            image_h=h,
            training_image=input_rgb,
            output_image=output_rgb,
            target_data=target_rgb,
            thumb0=(weight_tiles[0] if weight_tiles is not None else None),
            thumb1=(weight_tiles[1] if weight_tiles is not None else None),
            thumb2=(weight_tiles[2] if weight_tiles is not None else None),
        )
        titles = list(panel_titles) if panel_titles else ["target", "input", "output"]
        rows = [list(r) for r in panel_rows] if panel_rows else [[], [], []]
        self._send({
            "type": "frame_signal",
            "ring_cursor": int(ring_cursor),
            "caption": str(caption),
            "titles": titles,
            "rows": rows,
        })

    def notify_pipeline_checkpoint_saved(self) -> None:
        self._send({"type": "checkpoint_saved"})

    def notify_checkpoint_at_walltime(self, wall_ts: float) -> None:
        self._send({"type": "checkpoint_at_walltime", "wall_ts": float(wall_ts)})

    def set_cycle_roster(self, total_cycles: int, selected=None):
        n = max(0, int(total_cycles))
        self._cycle_selected = [True] * n
        self._send({"type": "cycle_roster", "total_cycles": n})

    def set_queue_refs(self, work_queue) -> None:
        self._preview_work_queue_ref = work_queue

    def set_weight_model_refs(self, models, disk_states=None, ckpt_states=None) -> None:
        pass  # Weight map rendering not yet supported in IPC mode

    def set_restore_state_callback(self, fn) -> None:
        self._on_restore = fn

    def set_weight_snap_dir(self, path) -> None:
        pass  # Managed by the GUI side

    def set_checkpoint_backup_dir(self, path) -> None:
        self._send({"type": "checkpoint_backup_dir", "path": str(path)})

    def trim_graph_to_first_checkpoint(self) -> None:
        self._send({"type": "trim_graph"})

    def is_cycle_selected(self, cycle_local: int) -> bool:
        idx = int(cycle_local) - 1
        if idx < 0 or idx >= len(self._cycle_selected):
            return True
        return bool(self._cycle_selected[idx])

    def gate_override_enabled(self) -> bool:
        return self._gate_override

    def paused(self) -> bool:
        self._drain_status()
        return self._paused

    def preview_enabled(self) -> bool:
        self._drain_status()
        return self._preview_enabled

    def scrub_editor_enabled(self) -> bool:
        self._drain_status()
        return self._scrub_editor_enabled

    def selected_cycle_ids(self) -> List[int]:
        return [int(i + 1) for i, v in enumerate(self._cycle_selected) if bool(v)]

    def current_run_control(self) -> RunControlPayload:
        self._drain_status()
        return self._last_run_control

    def consume_pending_plan_apply(self) -> Optional[PlanApplyPayload]:
        """Return and clear any plan_apply payload queued from the GUI, or None."""
        self._drain_status()
        payload = self._pending_plan_apply
        self._pending_plan_apply = None
        return payload

    def get_restore_state_dicts(self, offset: int):
        return None  # Cross-process restore not yet implemented

    def send_worker_hello(
        self,
        payload: WorkerHelloPayload,
        *,
        session_id: str = "",
        plan_id: str = "",
        revision: int = 0,
    ) -> None:
        self._send(
            make_envelope(
                MESSAGE_TYPE_WORKER_HELLO,
                payload,
                session_id=session_id,
                worker_id=payload.worker_id,
                plan_id=plan_id,
                revision=revision,
            ).to_dict()
        )

    def send_plan_snapshot(
        self,
        plan,
        *,
        session_id: str = "",
        worker_id: str = "",
        revision: int = 0,
    ) -> None:
        self._send(
            make_envelope(
                MESSAGE_TYPE_PLAN_SNAPSHOT,
                {"plan": plan.to_dict() if hasattr(plan, "to_dict") else dict(plan)},
                session_id=session_id,
                worker_id=worker_id,
                plan_id=str(getattr(plan, "plan_id", "")),
                revision=revision,
            ).to_dict()
        )

    def send_runtime_snapshot(
        self,
        payload: RuntimeSnapshotPayload,
        *,
        session_id: str = "",
        worker_id: str = "",
        plan_id: str = "",
        revision: int = 0,
    ) -> None:
        self._send(
            make_envelope(
                MESSAGE_TYPE_RUNTIME_SNAPSHOT,
                payload,
                session_id=session_id,
                worker_id=worker_id or payload.worker_id,
                plan_id=plan_id,
                revision=revision,
            ).to_dict()
        )

    def send_execution_event(
        self,
        payload: ExecutionEventPayload,
        *,
        session_id: str = "",
        worker_id: str = "",
        plan_id: str = "",
        revision: int = 0,
    ) -> None:
        self._send(
            make_envelope(
                MESSAGE_TYPE_EXECUTION_EVENT,
                payload,
                session_id=session_id,
                worker_id=worker_id,
                plan_id=plan_id,
                revision=revision,
            ).to_dict()
        )

    def close(self) -> None:
        self._send({"type": "exit"})
        self.enabled = False
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
