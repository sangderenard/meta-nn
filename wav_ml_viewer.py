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
    WeightImageConfig,
    NodusWeightImageStore,
    NodusWeightStateStore,
    SCRUB_FLAG_HAS_IMAGE,
    SCRUB_FLAG_HAS_OUTPUT,
    SCRUB_FLAG_HAS_TARGET,
)
from pipeline.plan_protocol import (
    MESSAGE_TYPE_EXECUTION_EVENT,
    MESSAGE_TYPE_PLAN_APPLY,
    MESSAGE_TYPE_PLAN_SNAPSHOT,
    MESSAGE_TYPE_RUN_CONTROL,
    MESSAGE_TYPE_RUNTIME_SNAPSHOT,
    MESSAGE_TYPE_SCHEDULE_APPLY,
    MESSAGE_TYPE_WORKER_HELLO,
    ExecutionEventPayload,
    PlanApplyPayload,
    RunControlPayload,
    RuntimeSnapshotPayload,
    ScheduleApplyPayload,
    TrainingSchedule,
    WorkerHelloPayload,
    is_protocol_envelope_message,
    make_envelope,
    parse_envelope,
)
from pipeline.weight_map import annotate_weight_map
from pipeline.weight_image_cache import (
    WEIGHT_IMAGE_FLAG_CHECKPOINT,
    checkpoint_thumbnail_path,
    checkpoint_thumbnail_root,
    crop_weight_image_rgb,
    find_checkpoint_thumbnail,
    load_checkpoint_thumbnail,
    plan_weight_image_evictions,
    save_checkpoint_thumbnail,
)

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
        self.top_bar_h = 104
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
        self._backend_quit_requested = False  # QUIT button: stop backend only, keep GUI open
        self._shutdown_save: Optional[bool] = None  # None=no shutdown, True=save, False=nosave
        self._launch_script: Optional[str] = None   # .bat to re-launch training
        self._output_dir: Optional[str] = None
        self._port_file_path: Optional[str] = None
        self._training_proc: Optional[Any] = None   # subprocess.Popen handle
        self._ipc_server_ref: Optional[Any] = None   # ViewerIPCServer backref
        # START button safety gates:
        # - cooldown blocks rapid re-clicks
        # - launch_pending blocks duplicate starts before IPC connect arrives
        self._start_cooldown_s: float = 10.0
        self._start_cooldown_until: float = 0.0
        self._start_launch_grace_s: float = 30.0
        self._start_launch_pending_until: float = 0.0

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
        self._stage_selected: Dict[str, bool] = {}
        self._stage_roster: List[Tuple[str, str]] = []
        self._gate_override = False
        self._suppress_rebuild = False
        self._force_rebuild = False
        self._paused = True
        self._preview_enabled = True
        self._scrub_editor_enabled = True
        self._skip_forward_pending: bool = False
        self._skip_back_pending: bool = False
        self._save_now_pending: bool = False
        self._control_boxes: List[Tuple[str, Any, Tuple[int, int, int, int]]] = []
        self._graph_control_boxes: List[Tuple[str, str, Tuple[int, int, int, int]]] = []
        self._graph_worker_hello: Dict[str, Any] = {}
        self._graph_plan_snapshot: Optional[Dict[str, Any]] = None
        self._graph_runtime_snapshot: Optional[Dict[str, Any]] = None
        self._graph_execution_events: deque = deque(maxlen=256)
        self._graph_x_axis_mode: str = "t"
        self._graph_history_mode: str = "recent"
        self.set_cycle_roster(total_cycles=int(cycle_slots))
        self.set_stage_roster([
            ("stage_0_pregestation", "Pregest"),
            ("stage_1_gestation", "Gestation"),
            ("stage_2_berkeley", "Berkeley"),
            ("stage_r_transformer", "Transf"),
            ("stage_g_generator", "Generat"),
            ("stage_c_lora", "LoRA"),
            ("stage_w_wave_classifier", "WaveCls"),
        ])

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
        try:
            self._weight_state_store: Optional[NodusWeightStateStore] = NodusWeightStateStore.get_global()
        except Exception:
            self._weight_state_store = None
        try:
            self._weight_image_store: Optional[NodusWeightImageStore] = NodusWeightImageStore.get_global()
        except Exception:
            self._weight_image_store = None
        self._last_ring_cursor: int = 0
        # Text metadata for the most-recently consumed frame (set from IPC signal
        # or directly by same-process update()).
        self._ring_text_caption: str = ""
        self._ring_text_titles: List[str] = ["target", "input", "output"]
        self._ring_text_rows: List[List[str]] = [[], [], []]
        # Text is now stored directly in the C ring alongside images.
        # _composite_ring_cursors maps composite-cache index → ring cursor so
        # that _apply_cached_frame_text can call ring.read_text(cursor).
        _composite_text_capacity = (
            self._composite_cache.capacity() if self._composite_cache is not None else 512
        )
        self._composite_ring_cursors: deque = deque(maxlen=max(1, int(_composite_text_capacity)))

        # -- Sidebar state -----------------------------------------------------
        # -inf forces an immediate first render on the first _present() call.
        self._sidebar_dirty: bool = True
        self._preview_work_queue_ref: Optional[Any] = None
        # -- Scrub / history ---------------------------------------------------
        # GUI-local copies of rendered weight images for scrub history.
        self._weight_history_maxlen: int = 512
        self._weight_snapshot_deque: deque = deque(maxlen=self._weight_history_maxlen)
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
        # Loss-count positions recorded at each checkpoint save, for graph markers.
        self._disk_save_loss_counts: list = []
        self._shared_weight_render_thread: Optional[threading.Thread] = None
        self._shared_weight_render_result: Optional[Dict[str, Any]] = None
        self._last_weight_state_publish_seq: int = 0
        self._last_weight_image_seq: int = 0
        self._last_weight_render_sig: Optional[Tuple[int, int, int, int, int, int]] = None
        self._pending_checkpoint_weight_marks: Dict[int, Tuple[int, int]] = {}
        self._checkpoint_backup_dir: Optional[Path] = None
        self._checkpoint_live_dir: Optional[Path] = None
        self._weight_image_mode: int = 3
        self._weight_model_order: List[str] = []
        self._weight_snapshot_deques_by_model: Dict[str, deque] = {}
        self._weight_current_rgb_by_model: Dict[str, np.ndarray] = {}
        self._weight_current_meta_by_model: Dict[str, Dict[str, Any]] = {}
        self._weight_tab_hit_boxes: List[Tuple[str, Tuple[int, int, int, int]]] = []
        self._checkpoint_marker_records: List[Dict[str, Any]] = []
        self._checkpoint_weight_records: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
        self._checkpoint_weight_rgb_cache: Dict[Tuple[int, int, str], np.ndarray] = {}
        self._weight_measure_cfg_cache: Dict[Tuple[str, str, int, int, int], WeightImageConfig] = {}
        # Which model to display in the right sidebar (None = first available).
        self._active_weight_model_name: Optional[str] = None

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

    def set_stage_roster(self, roster) -> None:
        self._stage_roster = list(roster)
        for node_id, _label in roster:
            if node_id not in self._stage_selected:
                self._stage_selected[node_id] = True
        self._top_bar_dirty = True

    _GATE_PARENT_STAGE: Dict[str, str] = {
        "gate_0_pregestation_eval": "stage_0_pregestation",
        "gate_1_gestation_eval":    "stage_1_gestation",
        "gate_berkeley":            "stage_2_berkeley",
        "gate_transformer":         "stage_r_transformer",
        "gate_generator":           "stage_g_generator",
        "gate_wave":                "stage_w_wave_classifier",
    }

    _STAGE_SUPPORT_NODES: Dict[str, str] = {
        "build_flashcard_rows": "stage_g_generator",
        "stage_fake_feedback":  "stage_g_generator",
    }

    def is_node_selected(self, node_id: str) -> bool:
        nid = str(node_id)
        if not bool(self._stage_selected.get(nid, True)):
            return False
        parent = self._GATE_PARENT_STAGE.get(nid)
        if parent is not None:
            if not bool(self._stage_selected.get(parent, True)):
                return False
            if self._gate_override:
                return False
        support_parent = self._STAGE_SUPPORT_NODES.get(nid)
        if support_parent is not None:
            if not bool(self._stage_selected.get(support_parent, True)):
                return False
        return True

    def selected_node_ids(self) -> List[str]:
        return [nid for nid, v in self._stage_selected.items() if bool(v)]

    def gate_override_enabled(self) -> bool:
        return bool(self._gate_override)

    def suppress_rebuild_enabled(self) -> bool:
        return bool(self._suppress_rebuild)

    def force_rebuild_enabled(self) -> bool:
        return bool(self._force_rebuild)

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
        connected = bool(getattr(srv, "has_connection", False))
        if connected:
            # Connected means the launch completed; clear pending gate.
            self._start_launch_pending_until = 0.0
        return connected

    def _start_cooldown_remaining(self) -> float:
        return max(0.0, float(self._start_cooldown_until) - time.monotonic())

    def _is_training_proc_alive(self) -> bool:
        proc = self._training_proc
        if proc is None:
            return False
        try:
            return proc.poll() is None
        except Exception:
            return False

    def _has_launch_pending(self) -> bool:
        return time.monotonic() < float(self._start_launch_pending_until)

    def shutdown_save(self) -> Optional[bool]:
        return self._shutdown_save

    def _start_training(self) -> None:
        """Launch the training bat/script as a subprocess."""
        import subprocess, os
        script = self._launch_script
        if not script:
            print("[viewer] no launch script configured; cannot start training", flush=True)
            return
        cooldown_remaining = self._start_cooldown_remaining()
        if cooldown_remaining > 0.0:
            print(
                f"[viewer] START cooldown active ({cooldown_remaining:.1f}s remaining); ignoring START",
                flush=True,
            )
            self._top_bar_dirty = True
            return
        if self.has_training_connection():
            print("[viewer] training already connected; ignoring START", flush=True)
            self._start_cooldown_until = time.monotonic() + float(self._start_cooldown_s)
            self._top_bar_dirty = True
            return
        if self._is_training_proc_alive():
            print("[viewer] training process already running; ignoring START", flush=True)
            self._start_cooldown_until = time.monotonic() + float(self._start_cooldown_s)
            self._top_bar_dirty = True
            return
        if self._has_launch_pending():
            pending_left = max(0.0, float(self._start_launch_pending_until) - time.monotonic())
            print(
                f"[viewer] previous START still pending ({pending_left:.1f}s until timeout); ignoring START",
                flush=True,
            )
            self._start_cooldown_until = time.monotonic() + float(self._start_cooldown_s)
            self._top_bar_dirty = True
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
            now = time.monotonic()
            self._start_cooldown_until = now + float(self._start_cooldown_s)
            self._start_launch_pending_until = now + float(self._start_launch_grace_s)
            print(f"[viewer] launched training process (pid={proc.pid}): {script}", flush=True)
        except Exception as e:
            # Launch failed immediately; release pending gate so user can retry.
            self._start_launch_pending_until = 0.0
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

            font = ImageFont.load_default()
            line_h = 11
            header_h = 20
            # Render at the full height required so no line is ever clipped.
            native_h = max(self.panel_h, header_h + len(list(rows)) * line_h + 4)

            canvas = np.full((native_h, self.panel_w, 3), 16, dtype=np.uint8)
            im = Image.fromarray(canvas).convert("RGB")
            draw = ImageDraw.Draw(im)

            draw.rectangle([(0, 0), (self.panel_w - 1, native_h - 1)], fill=(20, 24, 30))
            draw.rectangle([(0, 0), (self.panel_w - 1, 15)], fill=(34, 42, 52))
            _tw = font.getlength(str(title)) if hasattr(font, 'getlength') else len(str(title)) * 6
            draw.text((max(4, self.panel_w - 4 - int(_tw)), 2), str(title), fill=(255, 225, 70), font=font)
            draw.line([(0, 16), (self.panel_w - 1, 16)], fill=(70, 76, 88), width=1)
            y = header_h
            for r in list(rows):
                txt = str(r)
                if txt.startswith("!"):
                    txt = txt[1:]
                    fill = (255, 210, 60)   # amber highlight for target entries
                else:
                    fill = (230, 234, 240)
                _rw = font.getlength(txt) if hasattr(font, "getlength") else len(txt) * 6
                draw.text((max(4, self.panel_w - 4 - int(_rw)), y), txt, fill=fill, font=font)
                y += line_h
            # Scale down to the fixed panel size only when the content exceeds it,
            # preserving all text rather than clipping.
            if native_h > self.panel_h:
                im = im.resize((self.panel_w, self.panel_h), Image.LANCZOS)
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
        self._composite_ring_cursors.clear()

    def _prune_pending_frame_text(self, *, min_cursor: int) -> None:
        while self._composite_ring_cursors and self._composite_ring_cursors[0] < min_cursor:
            self._composite_ring_cursors.popleft()

    def _current_display_cache_index(self) -> Optional[int]:
        cache = self._composite_cache
        if cache is None:
            return None
        clen = cache.length()
        if clen <= 0:
            return None
        if self._scrub_offset == 0:
            return clen - 1
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
        """Apply text for a just-arrived frame.  Reads from the ring when possible;
        falls back to the provided kwargs for the no-ring / no-text case."""
        meta = None
        if ring_cursor is not None:
            try:
                cursor = int(ring_cursor)
                if cursor >= 0 and self._scrub_ring is not None:
                    meta = self._scrub_ring.read_text(cursor)
            except Exception:
                pass
        if meta is None:
            meta = self._frame_text_meta(caption=caption, titles=titles, rows=rows)
        self._apply_frame_text_meta(meta)

    def _cache_frame_text_for_cursor(self, ring_cursor: int) -> None:
        """Record the ring cursor for a newly composited frame."""
        self._composite_ring_cursors.append(int(ring_cursor))

    def _apply_cached_frame_text(self, cache_index: int) -> None:
        cursors = list(self._composite_ring_cursors)
        if cache_index < 0 or cache_index >= len(cursors):
            return
        cursor = cursors[cache_index]
        meta = None
        if self._scrub_ring is not None:
            try:
                meta = self._scrub_ring.read_text(cursor)
            except Exception:
                pass
        if meta is not None:
            self._apply_frame_text_meta(meta)

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
                idx=-1,
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
                self._control_boxes.append((kind, idx, box))
                return int(x + 15 + (len(label) * 7) + 10)

            draw.rectangle([(0, 0), (self.window_w - 1, self.top_bar_h - 1)], fill=(18, 22, 28))
            draw.line([(0, 17), (self.window_w - 1, 17)], fill=(42, 48, 58), width=1)
            draw.line([(0, 37), (self.window_w - 1, 37)], fill=(34, 40, 48), width=1)
            draw.line([(0, 57), (self.window_w - 1, 57)], fill=(34, 40, 48), width=1)
            draw.line([(0, 77), (self.window_w - 1, 77)], fill=(28, 34, 42), width=1)
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

            row1_y = 22
            tx = 8
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

            # -- Right-side action buttons (right to left: STOP, STOP+SAVE, SAVE, >|, PLAY/PAUSE, |<) --
            def _draw_btn(bx, label, kind, fill, outline, text_fill):
                lw = len(label) * 7 + 10
                box = (bx - lw, btn_y, bx, btn_y + btn_h)
                draw.rectangle([box[0], box[1], box[2], box[3]], outline=outline, fill=fill)
                draw.text((box[0] + 5, btn_y + 1), label, fill=text_fill, font=font)
                self._control_boxes.append((kind, -1, box))
                return box[0] - 6  # next bx

            if connected and stopping:
                label_pending = "STOPPING..."
                lw_p = len(label_pending) * 7 + 10
                p_box = (bx - lw_p, btn_y, bx, btn_y + btn_h)
                draw.rectangle([p_box[0], p_box[1], p_box[2], p_box[3]], outline=(120, 120, 60), fill=(80, 80, 30))
                draw.text((p_box[0] + 5, btn_y + 1), label_pending, fill=(220, 220, 160), font=font)
            elif connected:
                bx = _draw_btn(bx, "QUIT",      "quit",        (80, 20, 20),   (140, 40, 40),  (255, 180, 180))
                bx -= 4  # gap before stop buttons
                bx = _draw_btn(bx, "STOP",      "stop_nosave", (120, 36, 36),  (180, 60, 60),  (240, 200, 200))
                bx = _draw_btn(bx, "STOP+SAVE", "stop_save",   (36, 100, 50),  (60, 160, 80),  (200, 240, 210))
                bx = _draw_btn(bx, "SAVE",      "save_now",    (30, 60, 140),  (60, 100, 200), (190, 210, 255))
                bx -= 4  # small gap before transport controls
                bx = _draw_btn(bx, ">|",        "skip_fwd",    (52, 52, 60),   (100, 104, 116),(200, 210, 224))
                bx = _draw_btn(bx, "PAUSE" if not self._paused else "PLAY",
                               "pause",          (60, 60, 70),   (100, 104, 116),(220, 224, 232))
                bx = _draw_btn(bx, "|<",        "skip_back",   (52, 52, 60),   (100, 104, 116),(200, 210, 224))
            elif not connected and self._launch_script:
                cooldown_remaining = self._start_cooldown_remaining()
                launch_pending = self._has_launch_pending()
                proc_alive = self._is_training_proc_alive()
                can_start = (cooldown_remaining <= 0.0) and (not launch_pending) and (not proc_alive)
                if launch_pending:
                    label_start = "STARTING..."
                elif proc_alive:
                    label_start = "RUNNING..."
                elif cooldown_remaining > 0.0:
                    label_start = f"START {int(math.ceil(cooldown_remaining))}s"
                else:
                    label_start = "START"
                start_outline = (60, 100, 180) if can_start else (86, 92, 104)
                start_fill = (36, 64, 130) if can_start else (48, 54, 64)
                start_text_fill = (200, 220, 250) if can_start else (170, 178, 188)
                _draw_btn(bx, label_start, "start" if can_start else "_start_disabled",
                          start_fill, start_outline, start_text_fill)

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

            _draw_check(
                max(x + 4, self.window_w - 330),
                row2_y,
                "Suppress rebuild",
                checked=bool(self._suppress_rebuild),
                kind="suppress_rebuild",
                fill_on=(60, 100, 120),
                fill_off=(36, 40, 44),
                text_on=(170, 210, 230),
                text_off=(150, 160, 172),
            )

            _draw_check(
                max(x + 4, self.window_w - 494),
                row2_y,
                "Force rebuild",
                checked=bool(self._force_rebuild),
                kind="force_rebuild",
                fill_on=(140, 60, 60),
                fill_off=(36, 40, 44),
                text_on=(240, 180, 180),
                text_off=(150, 160, 172),
            )

            row3_y = 62
            sx = 8
            for node_id, label in self._stage_roster:
                is_on = bool(self._stage_selected.get(node_id, True))
                sx = _draw_check(
                    sx,
                    row3_y,
                    label,
                    checked=is_on,
                    kind="stage",
                    idx=node_id,
                    fill_on=(44, 80, 130),
                    fill_off=(60, 36, 36),
                    text_on=(190, 210, 240),
                    text_off=(180, 140, 140),
                )

            active = self.selected_cycle_ids()
            active_txt = ",".join(str(i) for i in active) if len(active) > 0 else "none"
            desel_stages = [label for nid, label in self._stage_roster if not self._stage_selected.get(nid, True)]
            desel_txt = (",".join(desel_stages) if desel_stages else "none off")
            draw.text(
                (6, self.top_bar_h - 14),
                f"cycles={active_txt} skip={desel_txt} override={1 if self._gate_override else 0}"
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

    def _weight_map_target_hw(self) -> Tuple[int, int]:
        return (2 * self.panel_h + self.graph_total_h + self._weight_map_extra_h, self.panel_w)

    def _blank_weight_map_rgb(self) -> np.ndarray:
        target_h, target_w = self._weight_map_target_hw()
        return np.full((target_h, target_w, 3), 14, dtype=np.uint8)

    @staticmethod
    def _normalise_weight_model_name(name: Optional[str]) -> str:
        return str(name or "").strip()

    def _register_weight_model(self, name: Optional[str]) -> Optional[str]:
        model_name = self._normalise_weight_model_name(name)
        if not model_name:
            return None
        if model_name not in self._weight_model_order:
            self._weight_model_order.append(model_name)
        if model_name not in self._weight_snapshot_deques_by_model:
            self._weight_snapshot_deques_by_model[model_name] = deque(maxlen=self._weight_history_maxlen)
        if model_name not in self._weight_current_rgb_by_model:
            self._weight_current_rgb_by_model[model_name] = self._blank_weight_map_rgb()
        if self._active_weight_model_name is None:
            self._active_weight_model_name = model_name
            self._weight_snapshot_deque = self._weight_snapshot_deques_by_model[model_name]
        return model_name

    def _resolved_active_weight_model_name(self) -> Optional[str]:
        active = self._normalise_weight_model_name(self._active_weight_model_name)
        if active and active in self._weight_snapshot_deques_by_model:
            self._weight_snapshot_deque = self._weight_snapshot_deques_by_model[active]
            return active
        if self._weight_model_order:
            active = str(self._weight_model_order[0])
            self._active_weight_model_name = active
            self._weight_snapshot_deque = self._weight_snapshot_deques_by_model[active]
            return active
        return None

    def _remember_checkpoint_weight_record(
        self,
        *,
        round_id: int,
        cycle: int,
        model_name: Optional[str],
        generation: int = 0,
        architecture_version: int = 0,
        state_publish_seq: int = 0,
        node_id: str = "",
    ) -> None:
        model_key = self._register_weight_model(model_name)
        if model_key is None or int(round_id) <= 0:
            return
        self._checkpoint_weight_records[(int(round_id), int(cycle), model_key)] = {
            "round_id": int(round_id),
            "cycle": int(cycle),
            "model_name": str(model_key),
            "generation": int(generation),
            "architecture_version": int(architecture_version),
            "state_publish_seq": int(state_publish_seq),
            "node_id": str(node_id or ""),
        }

    def _checkpoint_offset_for_loss_counts(self, loss_counts: Dict[str, int]) -> Optional[int]:
        if not loss_counts:
            return None
        snap_list = list(self._loss_count_at_snap_deque)
        slen = len(snap_list)
        if slen <= 0:
            return None
        lengths = self._loss_channel_lengths()
        ref_sid = max(lengths, key=lengths.get, default=None)
        if ref_sid is None or ref_sid not in loss_counts:
            return None
        target_n = int(loss_counts.get(ref_sid, 0) or 0)
        if target_n <= 0:
            return None
        best_idx = 0
        best_dist = abs(int(snap_list[0].get(ref_sid, 0)) - target_n)
        for idx, snap in enumerate(snap_list):
            dist = abs(int(snap.get(ref_sid, 0)) - target_n)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        return max(1, slen - best_idx)

    def _checkpoint_marker_for_offset(self, offset: int) -> Optional[Dict[str, Any]]:
        target_offset = max(1, int(offset))
        for record in self._checkpoint_marker_records:
            counts = record.get("loss_counts")
            if not isinstance(counts, dict):
                continue
            marker_offset = self._checkpoint_offset_for_loss_counts(counts)
            if marker_offset is not None and int(marker_offset) == int(target_offset):
                return dict(record)
        return None

    def _checkpoint_record_for_model(
        self,
        marker: Optional[Dict[str, Any]],
        model_name: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(marker, dict):
            return None
        model_key = self._normalise_weight_model_name(model_name)
        if not model_key:
            return dict(marker)
        key = (
            int(marker.get("round_id", 0) or 0),
            int(marker.get("cycle", 0) or 0),
            model_key,
        )
        record = self._checkpoint_weight_records.get(key)
        if record is not None:
            merged = dict(marker)
            merged.update(record)
            return merged
        return dict(marker)

    def _find_checkpoint_image_store_index(
        self,
        *,
        round_id: int,
        cycle: int,
        model_name: str,
        generation: int = 0,
        architecture_version: int = 0,
    ) -> Optional[int]:
        store = self._weight_image_store
        if store is None:
            return None
        model_key = self._normalise_weight_model_name(model_name)
        for index in range(max(0, int(store.length())) - 1, -1, -1):
            meta = store.get_meta(index)
            if meta is None:
                continue
            if (int(meta.flags) & int(WEIGHT_IMAGE_FLAG_CHECKPOINT)) == 0:
                continue
            if int(meta.round_id) != int(round_id) or int(meta.cycle) != int(cycle):
                continue
            if self._normalise_weight_model_name(getattr(meta, "model_name", "")) != model_key:
                continue
            if int(generation) > 0 and int(getattr(meta, "generation", 0) or 0) != int(generation):
                continue
            if (
                int(architecture_version) > 0
                and int(getattr(meta, "architecture_version", 0) or 0) != int(architecture_version)
            ):
                continue
            return int(index)
        return None

    def _checkpoint_thumbnail_cache_key(
        self,
        *,
        round_id: int,
        cycle: int,
        model_name: str,
    ) -> Tuple[int, int, str]:
        return (int(round_id), int(cycle), self._normalise_weight_model_name(model_name))

    def _checkpoint_can_render_live_state(
        self,
        record: Optional[Dict[str, Any]],
        *,
        model_name: str,
    ) -> bool:
        if not self._has_shared_weight_pipeline():
            return False
        if not isinstance(record, dict):
            return False
        state_store = self._weight_state_store
        if state_store is None:
            return False
        node_id = str(record.get("node_id", "") or "")
        state_meta = None
        get_meta_for = getattr(state_store, "get_meta_for", None)
        if callable(get_meta_for):
            try:
                state_meta = get_meta_for(
                    model_name=str(model_name),
                    node_id=node_id,
                )
            except Exception:
                state_meta = None
        if state_meta is None:
            for candidate in self._weight_state_registry_entries():
                if self._normalise_weight_model_name(getattr(candidate, "model_name", "")) != self._normalise_weight_model_name(model_name):
                    continue
                if str(getattr(candidate, "node_id", "") or "") != node_id:
                    continue
                state_meta = candidate
                break
        if state_meta is None:
            return False
        if self._normalise_weight_model_name(getattr(state_meta, "model_name", "")) != self._normalise_weight_model_name(model_name):
            return False
        if node_id and str(getattr(state_meta, "node_id", "") or "") != node_id:
            return False
        if int(record.get("round_id", 0) or 0) > 0 and int(state_meta.round_id) != int(record.get("round_id", 0) or 0):
            return False
        if int(record.get("cycle", 0) or 0) > 0 and int(state_meta.cycle) != int(record.get("cycle", 0) or 0):
            return False
        if (
            int(record.get("state_publish_seq", 0) or 0) > 0
            and int(state_meta.publish_seq) != int(record.get("state_publish_seq", 0) or 0)
        ):
            return False
        if int(record.get("generation", 0) or 0) > 0 and int(getattr(state_meta, "generation", 0) or 0) != int(record.get("generation", 0) or 0):
            return False
        if (
            int(record.get("architecture_version", 0) or 0) > 0
            and int(getattr(state_meta, "architecture_version", 0) or 0) != int(record.get("architecture_version", 0) or 0)
        ):
            return False
        return True

    def _resolve_checkpoint_weight_rgb(
        self,
        marker: Optional[Dict[str, Any]],
        *,
        model_name: Optional[str],
    ) -> Optional[np.ndarray]:
        record = self._checkpoint_record_for_model(marker, model_name)
        if not isinstance(record, dict):
            return None
        model_key = self._normalise_weight_model_name(record.get("model_name") or model_name)
        round_id = int(record.get("round_id", 0) or 0)
        cycle = int(record.get("cycle", 0) or 0)
        if round_id <= 0 or not model_key:
            return None
        cache_key = self._checkpoint_thumbnail_cache_key(
            round_id=round_id,
            cycle=cycle,
            model_name=model_key,
        )
        cached = self._checkpoint_weight_rgb_cache.get(cache_key)
        if cached is not None:
            return np.ascontiguousarray(np.asarray(cached, dtype=np.uint8)).copy()
        root = self._checkpoint_thumbnail_root_path()
        path: Optional[Path] = None
        if root is not None:
            generation = int(record.get("generation", 0) or 0)
            architecture_version = int(record.get("architecture_version", 0) or 0)
            path = find_checkpoint_thumbnail(
                root,
                round_id=round_id,
                cycle=cycle,
                model_name=model_key,
                generation=(generation if generation > 0 else None),
                architecture_version=(architecture_version if architecture_version > 0 else None),
            )
        if path is None:
            index = self._find_checkpoint_image_store_index(
                round_id=round_id,
                cycle=cycle,
                model_name=model_key,
                generation=int(record.get("generation", 0) or 0),
                architecture_version=int(record.get("architecture_version", 0) or 0),
            )
            if index is not None:
                path = self._persist_checkpoint_weight_thumbnail(
                    int(index),
                    target_width=int(self.panel_w),
                    target_height=int(self._weight_map_target_hw()[0]),
                )
        if path is None and self._checkpoint_can_render_live_state(record, model_name=model_key):
            self._launch_shared_weight_render()
            index = self._find_checkpoint_image_store_index(
                round_id=round_id,
                cycle=cycle,
                model_name=model_key,
                generation=int(record.get("generation", 0) or 0),
                architecture_version=int(record.get("architecture_version", 0) or 0),
            )
            if index is not None:
                path = self._persist_checkpoint_weight_thumbnail(
                    int(index),
                    target_width=int(self.panel_w),
                    target_height=int(self._weight_map_target_hw()[0]),
                )
        if path is None:
            return None
        try:
            rgb = load_checkpoint_thumbnail(path)
        except Exception:
            return None
        cropped = crop_weight_image_rgb(
            rgb,
            target_width=int(self.panel_w),
            target_height=int(self._weight_map_target_hw()[0]),
        )
        self._checkpoint_weight_rgb_cache[cache_key] = cropped.copy()
        return cropped

    def _compose_weight_sidebar_rgb(
        self,
        base_rgb: np.ndarray,
        *,
        active_model: Optional[str],
        checkpoint_marker: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        out = crop_weight_image_rgb(
            np.ascontiguousarray(np.asarray(base_rgb, dtype=np.uint8)),
            target_width=int(self.panel_w),
            target_height=int(self._weight_map_target_hw()[0]),
        )
        self._weight_tab_hit_boxes = []
        model_names = [str(name) for name in self._weight_model_order if str(name)]
        if active_model and active_model not in model_names:
            model_names.append(str(active_model))
        if not model_names:
            return out
        try:
            from PIL import Image, ImageDraw, ImageFont

            im = Image.fromarray(out)
            draw = ImageDraw.Draw(im, "RGBA")
            font = ImageFont.load_default()
            tab_h = 18
            panel_w = int(out.shape[1])
            tab_x_margin = 4
            tab_y_start = 2

            # Layout pass: compute all tab boxes so we know total header height before drawing.
            tab_boxes: List[Tuple[str, Tuple[int, int, int, int]]] = []
            x = tab_x_margin
            y = tab_y_start
            for model_name in model_names:
                label = str(model_name)
                tab_w = max(30, len(label) * 7 + 12)
                if x + tab_w > panel_w - tab_x_margin and x > tab_x_margin:
                    # Wrap to next row (but don't wrap if this tab is the first on a row).
                    x = tab_x_margin
                    y += tab_h + 2
                clipped_w = min(tab_w, panel_w - tab_x_margin - x)
                box = (x, y, x + clipped_w, y + tab_h)
                tab_boxes.append((label, box))
                x = box[2] + tab_x_margin

            total_tab_h = y + tab_h + 3  # background height covers all rows
            draw.rectangle([(0, 0), (panel_w - 1, total_tab_h)], fill=(12, 16, 22, 208))

            for label, box in tab_boxes:
                is_active = str(label) == str(active_model or "")
                fill = (78, 112, 156, 230) if is_active else (28, 34, 44, 220)
                outline = (160, 206, 255, 255) if is_active else (76, 86, 98, 255)
                fg = (236, 244, 255, 255) if is_active else (170, 178, 188, 255)
                draw.rectangle([box[0], box[1], box[2], box[3]], fill=fill, outline=outline)
                draw.text((box[0] + 5, box[1] + 3), label, fill=fg, font=font)
                self._weight_tab_hit_boxes.append((label, box))

            status = "live"
            if isinstance(checkpoint_marker, dict) and int(checkpoint_marker.get("round_id", 0) or 0) > 0:
                status = f"checkpoint r{int(checkpoint_marker.get('round_id', 0))} c{int(checkpoint_marker.get('cycle', 0))}"
            active_label = str(active_model or "weights")
            draw.text((6, total_tab_h + 2), f"{active_label}  {status}", fill=(230, 236, 246, 255), font=font)
            return np.ascontiguousarray(np.asarray(im, dtype=np.uint8))
        except Exception:
            return out

    def _visible_weight_map_rgb(self) -> np.ndarray:
        active_model = self._resolved_active_weight_model_name()
        checkpoint_marker: Optional[Dict[str, Any]] = None
        if active_model is None and self._scrub_offset > 0:
            checkpoint_marker = self._checkpoint_marker_for_offset(self._scrub_offset)
            marker_model = self._normalise_weight_model_name(
                checkpoint_marker.get("model_name") if isinstance(checkpoint_marker, dict) else ""
            )
            if marker_model:
                active_model = self._register_weight_model(marker_model)
        base_rgb: Optional[np.ndarray] = None
        if self._scrub_offset > 0:
            checkpoint_marker = self._checkpoint_marker_for_offset(self._scrub_offset)
            if checkpoint_marker is not None and active_model is not None:
                base_rgb = self._resolve_checkpoint_weight_rgb(
                    checkpoint_marker,
                    model_name=active_model,
                )
            if base_rgb is None:
                base_rgb = self._history_weight_map_at_offset(
                    self._scrub_offset,
                    model_name=active_model,
                )
        elif active_model is not None:
            base_rgb = self._weight_current_rgb_by_model.get(active_model)
        if base_rgb is None:
            base_rgb = self._blank_weight_map_rgb()
        return self._compose_weight_sidebar_rgb(
            base_rgb,
            active_model=active_model,
            checkpoint_marker=checkpoint_marker,
        )

    def _weight_state_registry_entries(self) -> List[Any]:
        store = self._weight_state_store
        if store is None:
            return []
        list_meta = getattr(store, "list_meta", None)
        if callable(list_meta):
            try:
                metas = list_meta()
            except Exception:
                metas = []
            if metas:
                return list(metas)
        meta = store.get_meta()
        return [meta] if meta is not None else []

    def _find_weight_image_store_index_for_state(
        self,
        state_meta: Any,
        *,
        mode: Optional[int] = None,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        checkpoint_only: bool = False,
    ) -> Optional[int]:
        store = self._weight_image_store
        if store is None or state_meta is None:
            return None
        target_model = self._normalise_weight_model_name(getattr(state_meta, "model_name", ""))
        target_node = str(getattr(state_meta, "node_id", "") or "")
        target_publish_seq = int(getattr(state_meta, "publish_seq", 0) or 0)
        for index in range(max(0, int(store.length())) - 1, -1, -1):
            image_meta = store.get_meta(index)
            if image_meta is None:
                continue
            if int(getattr(image_meta, "state_publish_seq", 0) or 0) != target_publish_seq:
                continue
            if self._normalise_weight_model_name(getattr(image_meta, "model_name", "")) != target_model:
                continue
            if str(getattr(image_meta, "node_id", "") or "") != target_node:
                continue
            if checkpoint_only and (int(getattr(image_meta, "flags", 0) or 0) & int(WEIGHT_IMAGE_FLAG_CHECKPOINT)) == 0:
                continue
            if mode is not None and int(getattr(image_meta, "mode", 0) or 0) != int(mode):
                continue
            if target_width is not None and int(getattr(image_meta, "target_width", 0) or 0) != int(target_width):
                continue
            if target_height is not None and int(getattr(image_meta, "target_height", 0) or 0) != int(target_height):
                continue
            return int(index)
        return None

    def _weight_image_present_for_state(
        self,
        state_meta: Any,
        *,
        mode: int,
        target_width: int,
        target_height: int,
    ) -> bool:
        return self._find_weight_image_store_index_for_state(
            state_meta,
            mode=int(mode),
            target_width=int(target_width),
            target_height=int(target_height),
        ) is not None

    def _measure_weight_state_cfg(
        self,
        state_meta: Any,
        *,
        mode: int,
        target_width: int,
        target_height: int,
    ) -> Optional[WeightImageConfig]:
        if state_meta is None or self._weight_state_store is None or self._weight_image_store is None:
            return None
        key = (
            self._normalise_weight_model_name(getattr(state_meta, "model_name", "")),
            str(getattr(state_meta, "node_id", "") or ""),
            int(getattr(state_meta, "publish_seq", 0) or 0),
            int(mode),
            (int(target_width) << 16) ^ int(target_height),
        )
        cached = self._weight_measure_cfg_cache.get(key)
        if cached is not None:
            return cached
        try:
            cfg = self._weight_image_store.measure_for(
                self._weight_state_store,
                model_name=str(getattr(state_meta, "model_name", "") or ""),
                node_id=str(getattr(state_meta, "node_id", "") or ""),
                mode=int(mode),
                target_width=int(target_width),
                target_height=int(target_height),
            )
        except Exception:
            return None
        if cfg is not None:
            self._weight_measure_cfg_cache[key] = cfg
        return cfg

    # -- Sidebar public API -----------------------------------------------------

    def set_queue_refs(self, work_queue: Any) -> None:
        """Register the preview work queue so the cache map can read its depth."""
        self._preview_work_queue_ref = work_queue

    def _history_snap_count(self) -> int:
        cc_len = self._composite_cache.length() if self._composite_cache else 0
        active_model = self._resolved_active_weight_model_name()
        if active_model is not None:
            weight_len = len(self._weight_snapshot_deques_by_model.get(active_model, ()))
        else:
            weight_len = len(self._weight_snapshot_deque)
        return max(
            int(weight_len),
            cc_len,
            len(self._loss_count_at_snap_deque),
        )

    def _record_history_snap(self, _ring_cursor: Optional[int] = None) -> None:
        self._loss_count_at_snap_deque.append(self._loss_channel_lengths())
        active_model = self._resolved_active_weight_model_name()
        if active_model is None:
            return
        current_rgb = self._weight_current_rgb_by_model.get(active_model)
        if current_rgb is None:
            return
        snaps = self._weight_snapshot_deques_by_model.get(active_model)
        if snaps is None:
            return
        snaps.append(np.ascontiguousarray(np.asarray(current_rgb, dtype=np.uint8)).copy())
        self._weight_snapshot_deque = snaps

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

    def _history_weight_map_at_offset(
        self,
        offset: int,
        *,
        model_name: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        model_key = self._normalise_weight_model_name(model_name)
        if model_key:
            snaps = list(self._weight_snapshot_deques_by_model.get(model_key, ()))
        else:
            snaps = list(self._weight_snapshot_deque)
        slen = len(snaps)
        if slen <= 0:
            return None
        off = max(1, min(slen, int(offset)))
        idx = max(0, min(slen - 1, slen - off))
        return np.ascontiguousarray(np.asarray(snaps[idx], dtype=np.uint8))

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
        """Set which model name is shown in the right sidebar weight image."""
        if name is None:
            self._active_weight_model_name = None
        else:
            self._active_weight_model_name = self._register_weight_model(name)
        active = self._resolved_active_weight_model_name()
        if active is not None:
            self._sr_weight_active = str(active)
        self._sidebar_dirty = True

    def weight_image_spec(self) -> Dict[str, int]:
        target_h, target_w = self._weight_map_target_hw()
        return {
            "mode": int(self._weight_image_mode) | 0x100,
            "panel_crop_w": int(target_w),
            "panel_crop_h": int(target_h),
        }

    @staticmethod
    def _weight_render_signature(cfg: WeightImageConfig) -> Tuple[int, int, int, int, int, int]:
        return (
            int(cfg.state_publish_seq),
            int(cfg.mode),
            int(cfg.target_width),
            int(cfg.target_height),
            int(cfg.render_width),
            int(cfg.render_height),
        )

    def _apply_weight_image_store_limits(self, cfg: WeightImageConfig) -> None:
        store = self._weight_image_store
        if store is None:
            return
        set_limits = getattr(store, "set_limits", None)
        if not callable(set_limits):
            return
        row_bytes = int(getattr(cfg, "render_stride_bytes", 0) or 0)
        if row_bytes <= 0:
            row_bytes = int(cfg.render_width) * max(1, int(cfg.render_channels))
        image_bytes = row_bytes * max(1, int(cfg.render_height))
        max_entries = max(1, int(self._weight_history_maxlen))
        max_total_bytes = max(1, image_bytes) * max_entries
        self._persist_weight_image_evictions_for_limits(cfg)
        try:
            set_limits(max_entries=max_entries, max_total_bytes=max_total_bytes)
        except Exception:
            return

    def _checkpoint_thumbnail_root_path(self) -> Optional[Path]:
        if self._checkpoint_backup_dir is not None:
            return checkpoint_thumbnail_root(self._checkpoint_backup_dir)
        if self._checkpoint_live_dir is not None:
            return checkpoint_thumbnail_root(self._checkpoint_live_dir / "_weight_backup")
        return None

    def _persist_checkpoint_weight_thumbnail(
        self,
        index: int,
        *,
        target_width: int,
        target_height: int,
    ) -> Optional[Path]:
        store = self._weight_image_store
        root = self._checkpoint_thumbnail_root_path()
        if store is None or root is None:
            return None
        meta = store.get_meta(int(index))
        if meta is None:
            return None
        if (int(meta.flags) & int(WEIGHT_IMAGE_FLAG_CHECKPOINT)) == 0:
            return None
        rgb = store.copy_image(int(index))
        if rgb is None:
            return None
        cropped = crop_weight_image_rgb(
            rgb,
            target_width=int(target_width),
            target_height=int(target_height),
        )
        path = checkpoint_thumbnail_path(
            root,
            round_id=int(meta.round_id),
            cycle=int(meta.cycle),
            model_name=str(meta.model_name),
            generation=int(meta.generation),
            architecture_version=int(meta.architecture_version),
        )
        try:
            saved = save_checkpoint_thumbnail(path, cropped)
        except Exception:
            return None
        self._remember_checkpoint_weight_record(
            round_id=int(meta.round_id),
            cycle=int(meta.cycle),
            model_name=str(meta.model_name),
            generation=int(getattr(meta, "generation", 0) or 0),
            architecture_version=int(getattr(meta, "architecture_version", 0) or 0),
            state_publish_seq=int(getattr(meta, "state_publish_seq", 0) or 0),
            node_id=str(getattr(meta, "node_id", "") or ""),
        )
        self._checkpoint_weight_rgb_cache.pop(
            self._checkpoint_thumbnail_cache_key(
                round_id=int(meta.round_id),
                cycle=int(meta.cycle),
                model_name=str(meta.model_name),
            ),
            None,
        )
        return saved

    def _persist_weight_image_evictions(
        self,
        evict_indices: Sequence[int],
        *,
        target_width: int,
        target_height: int,
    ) -> None:
        for index in sorted({int(i) for i in evict_indices}):
            try:
                self._persist_checkpoint_weight_thumbnail(
                    int(index),
                    target_width=int(target_width),
                    target_height=int(target_height),
                )
            except Exception:
                continue

    def _persist_weight_image_evictions_for_limits(self, cfg: WeightImageConfig) -> None:
        store = self._weight_image_store
        if store is None:
            return
        row_bytes = int(getattr(cfg, "render_stride_bytes", 0) or 0)
        if row_bytes <= 0:
            row_bytes = int(cfg.render_width) * max(1, int(cfg.render_channels))
        image_bytes = row_bytes * max(1, int(cfg.render_height))
        max_entries = max(1, int(self._weight_history_maxlen))
        max_total_bytes = max(1, image_bytes) * max_entries
        entries = [store.get_meta(i) for i in range(max(0, int(store.length())))]
        evict_indices = plan_weight_image_evictions(
            [entry for entry in entries if entry is not None],
            max_entries=int(max_entries),
            max_total_bytes=int(max_total_bytes),
        )
        self._persist_weight_image_evictions(
            evict_indices,
            target_width=int(cfg.target_width),
            target_height=int(cfg.target_height),
        )

    def _persist_weight_image_evictions_for_render(self, cfg: WeightImageConfig) -> None:
        store = self._weight_image_store
        if store is None:
            return
        stats = store.stats()
        if stats is None:
            return
        row_bytes = int(getattr(cfg, "render_stride_bytes", 0) or 0)
        if row_bytes <= 0:
            row_bytes = int(cfg.render_width) * max(1, int(cfg.render_channels))
        incoming_bytes = row_bytes * max(1, int(cfg.render_height))
        entries = [store.get_meta(i) for i in range(max(0, int(store.length())))]
        evict_indices = plan_weight_image_evictions(
            [entry for entry in entries if entry is not None],
            max_entries=max(1, int(stats.max_entries)),
            max_total_bytes=max(0, int(stats.max_total_bytes)),
            incoming_bytes=max(0, int(incoming_bytes)),
        )
        self._persist_weight_image_evictions(
            evict_indices,
            target_width=int(cfg.target_width),
            target_height=int(cfg.target_height),
        )

    def _has_shared_weight_pipeline(self) -> bool:
        return (
            self._weight_state_store is not None
            and self._weight_image_store is not None
            and self.has_training_connection()
        )

    def _launch_shared_weight_render(self) -> None:
        if not self._has_shared_weight_pipeline():
            return
        if self._shared_weight_render_thread is not None:
            return
        state_store = self._weight_state_store
        image_store = self._weight_image_store
        if state_store is None or image_store is None:
            return
        spec = self.weight_image_spec()
        mode = int(spec.get("mode", self._weight_image_mode) or self._weight_image_mode)
        target_width = int(spec.get("panel_crop_w", self.panel_w) or self.panel_w)
        target_height = int(spec.get("panel_crop_h", self._weight_map_target_hw()[0]) or self._weight_map_target_hw()[0])
        registry_entries = list(self._weight_state_registry_entries())
        if not registry_entries:
            return
        for state_meta in registry_entries:
            self._register_weight_model(getattr(state_meta, "model_name", ""))
        active_model = self._resolved_active_weight_model_name()
        registry_entries.sort(
            key=lambda meta: (
                0 if self._normalise_weight_model_name(getattr(meta, "model_name", "")) == str(active_model or "") else 1,
                -int(getattr(meta, "publish_seq", 0) or 0),
            )
        )
        measured_cfgs: List[WeightImageConfig] = []
        pending_pairs: List[Tuple[Any, Optional[WeightImageConfig]]] = []
        for state_meta in registry_entries:
            cfg = self._measure_weight_state_cfg(
                state_meta,
                mode=int(mode),
                target_width=int(target_width),
                target_height=int(target_height),
            )
            if cfg is not None:
                measured_cfgs.append(cfg)
            if not self._weight_image_present_for_state(
                state_meta,
                mode=int(mode),
                target_width=int(target_width),
                target_height=int(target_height),
            ):
                pending_pairs.append((state_meta, cfg))
        limit_cfg = None
        if measured_cfgs:
            limit_cfg = max(
                measured_cfgs,
                key=lambda cfg: int(getattr(cfg, "render_stride_bytes", 0) or 0) * max(1, int(getattr(cfg, "render_height", 0) or 0)),
            )
        elif pending_pairs and pending_pairs[0][1] is not None:
            limit_cfg = pending_pairs[0][1]
        if limit_cfg is not None:
            self._apply_weight_image_store_limits(limit_cfg)
            self._persist_weight_image_evictions_for_render(limit_cfg)
        if not pending_pairs:
            return
        result_box: Dict[str, Any] = {}

        def _shared_weight_worker(
            _self=self,
            _state_store=state_store,
            _image_store=image_store,
            _pending_pairs=list(pending_pairs),
            _box=result_box,
        ) -> None:
            render_results: List[Dict[str, Any]] = []
            for _state_meta, _cfg in _pending_pairs:
                ok = _image_store.render_for(
                    _state_store,
                    model_name=str(getattr(_state_meta, "model_name", "") or ""),
                    node_id=str(getattr(_state_meta, "node_id", "") or ""),
                    mode=int(mode),
                    target_width=int(target_width),
                    target_height=int(target_height),
                )
                if not ok:
                    continue
                index = _self._find_weight_image_store_index_for_state(
                    _state_meta,
                    mode=int(mode),
                    target_width=int(target_width),
                    target_height=int(target_height),
                )
                if index is None:
                    continue
                image_meta = _image_store.get_meta(int(index))
                rgb = _image_store.copy_image(int(index))
                if image_meta is None or rgb is None:
                    continue
                subtitle = f"r{int(image_meta.round_id)} c{int(image_meta.cycle)} s{int(image_meta.step)}"
                render_results.append(
                    {
                        "state_publish_seq": int(getattr(_state_meta, "publish_seq", 0) or 0),
                        "model_name": str(getattr(_state_meta, "model_name", "") or ""),
                        "render_sig": (
                            int(getattr(_state_meta, "publish_seq", 0) or 0),
                            int(mode),
                            int(target_width),
                            int(target_height),
                            self._normalise_weight_model_name(getattr(_state_meta, "model_name", "")),
                            str(getattr(_state_meta, "node_id", "") or ""),
                        ),
                        "image_meta": image_meta,
                        "weight_map": annotate_weight_map(
                            np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8)),
                            title=str(getattr(image_meta, "model_name", "") or getattr(_state_meta, "model_name", "") or ""),
                            subtitle=subtitle,
                        ),
                        "measured_cfg": _cfg,
                    }
                )
            _box["ok"] = bool(render_results)
            _box["render_results"] = render_results

        self._shared_weight_render_result = result_box
        self._shared_weight_render_thread = threading.Thread(
            target=_shared_weight_worker,
            name="shared-weight-render",
            daemon=True,
        )
        self._shared_weight_render_thread.start()

    def _collect_shared_weight_render(self) -> None:
        thread = self._shared_weight_render_thread
        result = self._shared_weight_render_result
        if thread is None or thread.is_alive() or result is None:
            return
        self._shared_weight_render_thread = None
        self._shared_weight_render_result = None
        if not bool(result.get("ok")):
            return
        render_results = list(result.get("render_results") or [])
        if not render_results and result.get("image_meta") is not None and result.get("weight_map") is not None:
            render_results = [dict(result)]
        any_rendered = False
        for item in render_results:
            image_meta = item.get("image_meta")
            weight_map = item.get("weight_map")
            if image_meta is None or weight_map is None:
                continue
            self._last_weight_state_publish_seq = int(item.get("state_publish_seq", 0) or 0)
            self._last_weight_image_seq = int(getattr(image_meta, "image_seq", 0) or 0)
            render_sig = item.get("render_sig")
            if isinstance(render_sig, tuple):
                self._last_weight_render_sig = render_sig
            pending_ckpt = self._pending_checkpoint_weight_marks.pop(
                int(getattr(image_meta, "state_publish_seq", 0) or 0),
                None,
            )
            if pending_ckpt is not None:
                self._mark_checkpoint_weight_image(
                    int(getattr(image_meta, "state_publish_seq", 0) or 0),
                    int(pending_ckpt[0]),
                    int(pending_ckpt[1]),
                )
                self._remember_checkpoint_weight_record(
                    round_id=int(pending_ckpt[0]),
                    cycle=int(pending_ckpt[1]),
                    model_name=str(getattr(image_meta, "model_name", "") or item.get("model_name") or ""),
                    generation=int(getattr(image_meta, "generation", 0) or 0),
                    architecture_version=int(getattr(image_meta, "architecture_version", 0) or 0),
                    state_publish_seq=int(getattr(image_meta, "state_publish_seq", 0) or 0),
                    node_id=str(getattr(image_meta, "node_id", "") or ""),
                )
            model_name = self._register_weight_model(
                str(item.get("model_name") or getattr(image_meta, "model_name", "") or "")
            )
            weight_map = crop_weight_image_rgb(
                np.ascontiguousarray(np.asarray(weight_map, dtype=np.uint8)),
                target_width=int(getattr(image_meta, "target_width", 0) or weight_map.shape[1]),
                target_height=int(getattr(image_meta, "target_height", 0) or weight_map.shape[0]),
            )
            if model_name is not None:
                self._weight_current_rgb_by_model[model_name] = weight_map.copy()
                self._weight_current_meta_by_model[model_name] = {
                    "model_name": str(model_name),
                    "state_publish_seq": int(getattr(image_meta, "state_publish_seq", 0) or 0),
                    "image_seq": int(getattr(image_meta, "image_seq", 0) or 0),
                    "round_id": int(getattr(image_meta, "round_id", 0) or 0),
                    "cycle": int(getattr(image_meta, "cycle", 0) or 0),
                    "step": int(getattr(image_meta, "step", 0) or 0),
                    "generation": int(getattr(image_meta, "generation", 0) or 0),
                    "architecture_version": int(getattr(image_meta, "architecture_version", 0) or 0),
                    "node_id": str(getattr(image_meta, "node_id", "") or ""),
                    "mode": int(getattr(image_meta, "mode", 0) or 0),
                    "target_width": int(getattr(image_meta, "target_width", 0) or 0),
                    "target_height": int(getattr(image_meta, "target_height", 0) or 0),
                }
                snaps = self._weight_snapshot_deques_by_model.get(model_name)
                if snaps is not None:
                    snaps.append(weight_map.copy())
                    if str(self._resolved_active_weight_model_name() or "") == str(model_name):
                        self._weight_snapshot_deque = snaps
                any_rendered = True
        if any_rendered:
            self._sidebar_dirty = True

    def _mark_checkpoint_weight_image(self, state_publish_seq: int, round_id: int, cycle: int) -> None:
        store = self._weight_image_store
        if store is None or int(state_publish_seq) <= 0:
            return
        try:
            marked = store.mark_checkpoint(
                int(state_publish_seq),
                round_id=int(round_id),
                cycle=int(cycle),
            )
            if not marked:
                self._pending_checkpoint_weight_marks[int(state_publish_seq)] = (
                    int(round_id),
                    int(cycle),
                )
        except Exception:
            self._pending_checkpoint_weight_marks[int(state_publish_seq)] = (
                int(round_id),
                int(cycle),
            )

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
            if scrubbing:
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
        color = (170, 80, 220) if offset > 0 else (60, 80, 110)
        dial = self._render_knob("scrub", fill_frac=fill_frac, color=color)
        try:
            from PIL import Image, ImageDraw, ImageFont
            im = Image.fromarray(dial)
            draw = ImageDraw.Draw(im)
            font = ImageFont.load_default()
            txt = f"-{offset}" if offset > 0 else "live"
            draw.text(
                (self.panel_w // 2 - len(txt) * 3, self.panel_h // 2 - 4),
                txt, fill=(220, 210, 240), font=font,
            )
            dial = np.asarray(im, dtype=np.uint8)
        except Exception:
            pass
        return dial

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

        # Sidebar panels redraw every pump; live weight images arrive via shared memory.
        self._cache_map_rgb  = self._render_btn_panel()
        self._frame_knob_rgb = self._render_scrub_dial()
        if self._has_shared_weight_pipeline():
            self._collect_shared_weight_render()
            self._launch_shared_weight_render()
        self._weight_map_rgb = self._visible_weight_map_rgb()
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
                elif kind == "stage":
                    node_id = str(idx)
                    if node_id in self._stage_selected:
                        self._stage_selected[node_id] = not bool(self._stage_selected[node_id])
                        self._top_bar_dirty = True
                        return True
                elif kind == "override":
                    self._gate_override = not bool(self._gate_override)
                    self._top_bar_dirty = True
                    return True
                elif kind == "suppress_rebuild":
                    self._suppress_rebuild = not bool(self._suppress_rebuild)
                    if self._suppress_rebuild:
                        self._force_rebuild = False
                    self._top_bar_dirty = True
                    return True
                elif kind == "force_rebuild":
                    self._force_rebuild = not bool(self._force_rebuild)
                    if self._force_rebuild:
                        self._suppress_rebuild = False
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
                elif kind == "quit":
                    self._shutdown_save = False
                    self._backend_quit_requested = True
                    self._top_bar_dirty = True
                    print("[viewer] QUIT requested — stopping backend", flush=True)
                    proc = getattr(self, "_training_proc", None)
                    if proc is not None:
                        try:
                            proc.terminate()
                            print(f"[viewer] terminated training process (pid={proc.pid})", flush=True)
                        except Exception as e:
                            print(f"[viewer] terminate failed: {e}", flush=True)
                    return True
                elif kind == "save_now":
                    self._save_now_pending = True
                    self._top_bar_dirty = True
                    print("[viewer] SAVE requested", flush=True)
                    return True
                elif kind == "skip_fwd":
                    self._skip_forward_pending = True
                    self._top_bar_dirty = True
                    print("[viewer] skip forward requested", flush=True)
                    return True
                elif kind == "skip_back":
                    self._skip_back_pending = True
                    self._top_bar_dirty = True
                    print("[viewer] skip back requested", flush=True)
                    return True
                elif kind == "start":
                    self._start_training()
                    return True
        return False

    def _handle_weight_tab_click(self, x: int, y: int) -> bool:
        if not self._weight_tab_hit_boxes:
            return False
        rx0 = int(self._col_x + self.num_panels * self.panel_w)
        xi = int(x) - rx0
        yi = int(y) - int(self.top_bar_h)
        if xi < 0 or yi < 0:
            return False
        for model_name, box in self._weight_tab_hit_boxes:
            x0, y0, x1, y1 = box
            if x0 <= xi <= x1 and y0 <= yi <= y1:
                self.set_active_weight_model(model_name)
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
                                        srv.send_query({
                                            "type": "restore",
                                            "offset": int(self._scrub_offset),
                                        })
                                    except Exception:
                                        pass
                                self._scrub_offset = 0
                                self._weight_snapshot_deque.clear()
                                for snaps in self._weight_snapshot_deques_by_model.values():
                                    snaps.clear()
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
                        if self._handle_weight_tab_click(xi, yi):
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
        if self._start_cooldown_remaining() > 0.0 or self._has_launch_pending():
            self._top_bar_dirty = True
        self._poll_events()
        self._sr_poll_data()
        self._present(force=False)

    def notify_pipeline_checkpoint_saved(
        self,
        *,
        round_id: int = 0,
        cycle: int = 0,
        checkpoint_path: Optional[str] = None,
        weight_state_publish_seq: int = 0,
        weight_model: Optional[str] = None,
        weight_generation: int = 0,
        weight_architecture_version: int = 0,
        weight_node_id: str = "",
        weight_models: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        """Call this immediately after every _save_training_segment_snapshot call.

        Records the current loss-series lengths so the loss graph can draw a gold
        vertical marker at the exact loss position where each disk checkpoint
        was written.  Thread-safe: can be called from any thread.
        """
        model_records: List[Dict[str, Any]] = []
        for item in list(weight_models or []):
            if not isinstance(item, dict):
                continue
            model_name = self._normalise_weight_model_name(item.get("model"))
            if not model_name:
                continue
            model_records.append(
                {
                    "model_name": model_name,
                    "node_id": str(item.get("node_id", "") or ""),
                    "state_publish_seq": int(item.get("publish_seq", 0) or 0),
                    "generation": int(item.get("generation", 0) or 0),
                    "architecture_version": int(item.get("architecture_version", 0) or 0),
                    "round_id": int(item.get("round_id", 0) or 0),
                    "cycle": int(item.get("cycle", 0) or 0),
                    "step": int(item.get("step", 0) or 0),
                }
            )
        primary_model_name = self._normalise_weight_model_name(weight_model)
        if not model_records and primary_model_name:
            model_records.append(
                {
                    "model_name": primary_model_name,
                    "node_id": str(weight_node_id or ""),
                    "state_publish_seq": int(weight_state_publish_seq),
                    "generation": int(weight_generation),
                    "architecture_version": int(weight_architecture_version),
                    "round_id": int(round_id),
                    "cycle": int(cycle),
                    "step": 0,
                }
            )
        if not primary_model_name and model_records:
            primary_model_name = str(model_records[0].get("model_name", "") or "")
        snapshot = self._loss_channel_lengths()
        self._disk_save_loss_counts.append(snapshot)
        record = {
            "loss_counts": dict(snapshot),
            "round_id": int(round_id),
            "cycle": int(cycle),
            "checkpoint_path": str(checkpoint_path or ""),
            "state_publish_seq": int(weight_state_publish_seq),
            "model_name": str(primary_model_name),
            "generation": int(weight_generation),
            "architecture_version": int(weight_architecture_version),
            "node_id": str(weight_node_id or ""),
            "weight_models": [dict(item) for item in model_records],
        }
        self._checkpoint_marker_records.append(record)
        if model_records:
            for item in model_records:
                self._remember_checkpoint_weight_record(
                    round_id=int(round_id),
                    cycle=int(cycle),
                    model_name=str(item.get("model_name", "") or ""),
                    generation=int(item.get("generation", 0) or 0),
                    architecture_version=int(item.get("architecture_version", 0) or 0),
                    state_publish_seq=int(item.get("state_publish_seq", 0) or 0),
                    node_id=str(item.get("node_id", "") or ""),
                )
        else:
            self._remember_checkpoint_weight_record(
                round_id=int(round_id),
                cycle=int(cycle),
                model_name=weight_model,
                generation=int(weight_generation),
                architecture_version=int(weight_architecture_version),
                state_publish_seq=int(weight_state_publish_seq),
                node_id=str(weight_node_id or ""),
            )
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
            self._checkpoint_marker_records.append(
                {
                    "loss_counts": dict(snapshot),
                    "round_id": 0,
                    "cycle": 0,
                    "checkpoint_path": "",
                    "state_publish_seq": 0,
                    "model_name": "",
                    "generation": 0,
                    "architecture_version": 0,
                    "node_id": "",
                }
            )
            self._graph_dirty = True

    def set_checkpoint_backup_dir(self, path) -> None:
        """Scan a backup directory for timestamped sub-dirs created by the batch launcher.

        Each sub-directory is expected to be named ``YYYYMMDD_HHMMSS`` and contain
        ``.pt`` files.  The directory's mtime (or the most-recently-modified .pt
        inside it) is used as the wall-clock timestamp.  A gold marker is placed
        on the loss graph at that time via ``notify_checkpoint_at_walltime``.
        """
        p = Path(path)
        if not p.is_dir():
            return
        self._checkpoint_backup_dir = p
        self._checkpoint_weight_rgb_cache.clear()
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

    def backend_quit_requested(self) -> bool:
        return bool(self._backend_quit_requested)

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
            r = int(msg.get("round_id", 0))
            c = int(msg.get("cycle", 0))
            checkpoint_path = str(msg.get("checkpoint_path", "") or "").strip()
            weight_models = [item for item in list(msg.get("weight_models", []) or []) if isinstance(item, dict)]
            if checkpoint_path:
                try:
                    self._checkpoint_live_dir = Path(checkpoint_path).resolve().parent
                except Exception:
                    self._checkpoint_live_dir = Path(checkpoint_path).parent
            self.notify_pipeline_checkpoint_saved(
                round_id=int(r),
                cycle=int(c),
                checkpoint_path=checkpoint_path,
                weight_state_publish_seq=int(msg.get("weight_state_publish_seq", 0) or 0),
                weight_model=str(msg.get("weight_model", "") or ""),
                weight_generation=int(msg.get("weight_generation", 0) or 0),
                weight_architecture_version=int(msg.get("weight_architecture_version", 0) or 0),
                weight_node_id=str(msg.get("weight_node_id", "") or ""),
                weight_models=weight_models,
            )
            if weight_models:
                for item in weight_models:
                    model_name = self._register_weight_model(item.get("model"))
                    if self._active_weight_model_name is None and model_name is not None:
                        self.set_active_weight_model(model_name)
                    state_publish_seq = int(item.get("publish_seq", 0) or 0)
                    if state_publish_seq > 0:
                        self._mark_checkpoint_weight_image(state_publish_seq, r, c)
            else:
                state_publish_seq = int(msg.get("weight_state_publish_seq", 0) or 0)
                self._mark_checkpoint_weight_image(state_publish_seq, r, c)
        elif t == "notify_weight_state":
            self._sidebar_dirty = True
            for name in list(msg.get("known_models", []) or []):
                self._register_weight_model(name)
            model_name = self._register_weight_model(msg.get("model"))
            if self._active_weight_model_name is None and model_name is not None:
                self.set_active_weight_model(model_name)
            self._sr_weight_models = [str(name) for name in self._weight_model_order]
            self._sr_weight_active = self._normalise_weight_model_name(msg.get("model"))

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

        elif t == "resp_cache_browse":
            self._sr_cache_entries = msg.get("entries", [])
            self._sr_cache_total = int(msg.get("total", 0))

        elif t == "resp_weight_registry":
            models = []
            for entry in list(msg.get("models", []) or []):
                if not isinstance(entry, dict):
                    continue
                model_name = self._register_weight_model(entry.get("model"))
                if model_name is not None:
                    models.append(model_name)
            self._sr_weight_models = list(dict.fromkeys(models))
            self._sr_weight_active = self._normalise_weight_model_name(msg.get("active_model"))
            if self._active_weight_model_name is None and self._sr_weight_active:
                self.set_active_weight_model(self._sr_weight_active)

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
        srv.send_query({"type": "query_weight_registry"})

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
        snap_frac = float(max(0, min(total_n, start_n))) / float(max(1, total_n))
        # Never hide more than 85% of recorded data in recent mode —
        # ensures at least 15% of all training history is always shown.
        return max(base, min(snap_frac, 0.85))

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
                    # Compute the actual step number for position (_sf * _i_n),
                    # accounting for circular-buffer wrap: oldest_step = cursor - length.
                    _i_cursor = store.channel_cursor(ck)
                    _i_oldest_step = _i_cursor - _i_n
                    _i_fs = _i_oldest_step + int(_sf * _i_n)
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
                    # Skip records with ts<=0 when computing the time range —
                    # they have no meaningful wall-clock and would anchor the
                    # axis at epoch 0, compressing all real data to the right.
                    pos_mask = arr > 0.0
                    if pos_mask.any():
                        arr_lo = float(arr[pos_mask][0])
                        arr_hi = float(arr[pos_mask][-1])
                    else:
                        continue
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
                        # searchsorted gives a position index; convert to the actual
                        # step number so C's step-based filter is correct even when
                        # the circular buffer has wrapped.
                        _pos = int(np.searchsorted(arr, t_view_min, side="left"))
                        _cursor_t = store.channel_cursor(ck)
                        _oldest_t = _cursor_t - int(arr.size)
                        from_step = _oldest_t + _pos
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
            h, w = target_rgb.shape[0], target_rgb.shape[1]
            flags = SCRUB_FLAG_HAS_IMAGE | SCRUB_FLAG_HAS_TARGET | SCRUB_FLAG_HAS_OUTPUT
            ring_cursor = self._scrub_ring.push(
                step=0, round_id=0, ts=time.time(), loss=0.0,
                channel_key="preview", flags=flags,
                image_w=w, image_h=h,
                training_image=input_rgb,
                output_image=output_rgb,
                target_data=target_rgb,
                thumb0=None,
                thumb1=None,
                thumb2=None,
            )
            self._scrub_ring.write_text(
                ring_cursor,
                caption=str(frame_dict.get("caption", "")),
                titles=frame_dict.get("titles", self._panel_titles),
                rows=frame_dict.get("rows", self._panel_rows),
            )
        self._stage_frame_text(ring_cursor)

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
        h, w = target_rgb.shape[0], target_rgb.shape[1]

        flags = SCRUB_FLAG_HAS_IMAGE | SCRUB_FLAG_HAS_TARGET | SCRUB_FLAG_HAS_OUTPUT
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
            thumb0=None,
            thumb1=None,
            thumb2=None,
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
        self._last_sent_snapshot: Optional[tuple] = None

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
                self._last_sent_snapshot = None  # force first status send
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

        try:
            is_stopping = (
                self._viewer.stop_requested()
                or bool(getattr(self._viewer, "_backend_quit_requested", False))
                or self._viewer.shutdown_save() is not None
            )
            is_paused = bool(self._viewer._paused)
            is_preview = bool(self._viewer._preview_enabled)
            is_scrub_editor = bool(self._viewer._scrub_editor_enabled)
            weight_spec = {}
            get_weight_spec = getattr(self._viewer, "weight_image_spec", None)
            if callable(get_weight_spec):
                try:
                    weight_spec = dict(get_weight_spec() or {})
                except Exception:
                    weight_spec = {}
            skip_fwd = bool(self._viewer._skip_forward_pending)
            skip_back = bool(self._viewer._skip_back_pending)
            save_now = bool(self._viewer._save_now_pending)
            gate_override = self._viewer.gate_override_enabled()
            suppress_rebuild = self._viewer.suppress_rebuild_enabled()
            force_rebuild = self._viewer.force_rebuild_enabled()
            cycle_selected = list(self._viewer._cycle_selected)
            node_selected = dict(self._viewer._stage_selected)
            weight_mode = int(weight_spec.get("mode", 1) or 1)
            weight_cw = int(weight_spec.get("panel_crop_w", 0) or 0)
            weight_ch = int(weight_spec.get("panel_crop_h", 0) or 0)

            # Only send if state changed or a one-shot signal is pending.
            snapshot = (
                is_stopping, is_paused, gate_override, suppress_rebuild, force_rebuild, is_preview, is_scrub_editor,
                tuple(cycle_selected), tuple(sorted(node_selected.items())),
                weight_mode, weight_cw, weight_ch,
            )
            if snapshot == self._last_sent_snapshot and not skip_fwd and not skip_back and not save_now:
                return
            self._last_sent_snapshot = snapshot

            # Consume one-shot signals only after we've decided to send.
            if skip_fwd:
                self._viewer._skip_forward_pending = False
            if skip_back:
                self._viewer._skip_back_pending = False
            if save_now:
                self._viewer._save_now_pending = False

            status = {
                "type": "status",
                "stop_requested": is_stopping,
                "paused": is_paused,
                "gate_override": gate_override,
                "suppress_rebuild": suppress_rebuild,
                "force_rebuild": force_rebuild,
                "preview_enabled": is_preview,
                "scrub_editor_enabled": is_scrub_editor,
                "cycle_selected": cycle_selected,
                "node_selected": node_selected,
                "skip_forward": skip_fwd,
                "skip_back": skip_back,
                "save_now": save_now,
                "weight_image_mode": weight_mode,
                "weight_panel_crop_w": weight_cw,
                "weight_panel_crop_h": weight_ch,
            }
            conn.send(status)
            cmd = "stop" if bool(status["stop_requested"]) else ("pause" if is_paused else "resume")
            run_control = RunControlPayload(
                command=cmd,
                selected_cycle_ids=self._viewer.selected_cycle_ids(),
                selected_node_ids=self._viewer.selected_node_ids(),
                gate_override=bool(status["gate_override"]),
                suppress_rebuild=bool(status["suppress_rebuild"]),
                force_rebuild=bool(status["force_rebuild"]),
                preview_enabled=is_preview,
                scrub_editor_enabled=is_scrub_editor,
                metadata={
                    "source": "viewer_ipc_status_loop",
                    "save": self._viewer.shutdown_save(),
                    "weight_image_mode": int(weight_spec.get("mode", 1) or 1),
                    "weight_panel_crop_w": int(weight_spec.get("panel_crop_w", 0) or 0),
                    "weight_panel_crop_h": int(weight_spec.get("panel_crop_h", 0) or 0),
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
                self._viewer._backend_quit_requested = False
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
                stage_fn(msg.get("ring_cursor"))
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
        self._suppress_rebuild = False
        self._force_rebuild = False
        self._paused = True
        self._preview_enabled = True
        self._scrub_editor_enabled = True
        self._cycle_selected: List[bool] = [True] * max(0, cycle_slots)
        self._stage_selected: Dict[str, bool] = {}
        self._skip_forward_pending: bool = False
        self._skip_back_pending: bool = False
        self._save_now_pending: bool = False
        self._last_run_control = RunControlPayload(
            command="pause",
            selected_cycle_ids=[int(i + 1) for i in range(max(0, cycle_slots))],
            gate_override=False,
        )
        self._pending_plan_apply: Optional[PlanApplyPayload] = None
        self._pending_schedule: Optional[Any] = None  # ScheduleApplyPayload
        self._conn: Optional[Any] = None
        self._connect_lock = threading.Lock()
        self._send_lock = threading.Lock()  # protects concurrent writes
        self._recv_lock = threading.Lock()  # protects concurrent reads (bg pump vs stop_requested)
        self._connected_once = False
        self._connection_lost = False
        self._reconnect_failures = 0
        self._reconnect_give_up = 10  # stop training after N consecutive failures
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

        self._preview_work_queue_ref: Optional[Any] = None
        self._weight_image_mode: int = 3
        self._weight_panel_crop_w: int = max(8, int(self.image_w))
        self._weight_panel_crop_h: int = max(8, int(self.image_h))
        self._last_applied_weight_image_spec: Optional[Tuple[int, int, int]] = None

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
            self._reconnect_failures = 0
            self._stop_flag = False
            self._shutdown_save = None
            self._preview_enabled = True
            self._scrub_editor_enabled = True
            self._last_applied_weight_image_spec = None
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
        if self._connect(reconnect=True, quiet=True):
            self._reconnect_failures = 0
        else:
            self._reconnect_failures += 1
            if self._reconnect_failures >= self._reconnect_give_up:
                print(
                    f"[viewer-ipc] GUI unreachable after {self._reconnect_failures} "
                    f"reconnect attempts; requesting stop",
                    flush=True,
                )
                self._stop_flag = True
                self._shutdown_save = False

    def _mark_connection_lost(self, reason: str = "") -> None:
        if self._connection_lost:
            return
        self._connection_lost = True
        # Do NOT set _stop_flag here.  A transient socket blip races with the
        # reconnect thread: the training loop would see stop_requested()=True and
        # raise StageStopRequested before _connect() has a chance to clear it.
        # Do NOT touch _paused — preserve pause state so the backend does not
        # spontaneously resume on a momentary disconnect.
        self._shutdown_save = None
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

    def _update_weight_image_spec(
        self,
        *,
        mode: Any = None,
        panel_crop_w: Any = None,
        panel_crop_h: Any = None,
    ) -> None:
        try:
            if mode is not None:
                self._weight_image_mode = int(mode)
        except Exception:
            pass
        try:
            if panel_crop_w is not None:
                self._weight_panel_crop_w = max(8, int(panel_crop_w))
        except Exception:
            pass
        try:
            if panel_crop_h is not None:
                self._weight_panel_crop_h = max(8, int(panel_crop_h))
        except Exception:
            pass

    def _sync_weight_image_spec_to_store(self) -> None:
        node = self._save_restore_node
        if node is None:
            return
        spec_sig = (
            int(self._weight_image_mode),
            int(self._weight_panel_crop_w),
            int(self._weight_panel_crop_h),
        )
        if spec_sig == self._last_applied_weight_image_spec:
            return
        fn = getattr(node, "configure_runtime_weight_image", None)
        if not callable(fn):
            self._last_applied_weight_image_spec = spec_sig
            return
        try:
            applied = fn(
                mode=int(self._weight_image_mode),
                target_width=int(self._weight_panel_crop_w),
                target_height=int(self._weight_panel_crop_h),
            )
        except Exception:
            return
        if applied is None:
            return
        self._last_applied_weight_image_spec = spec_sig

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
        if not self._recv_lock.acquire(timeout=0.02):
            # Another drainer active; skip — cached state is fresh enough.
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
                        self._suppress_rebuild = bool(getattr(payload, "suppress_rebuild", False))
                        self._force_rebuild = bool(getattr(payload, "force_rebuild", False))
                        self._preview_enabled = bool(getattr(payload, "preview_enabled", True))
                        self._scrub_editor_enabled = bool(getattr(payload, "scrub_editor_enabled", True))
                        # Extract save preference from metadata
                        meta = getattr(payload, "metadata", {}) or {}
                        if self._stop_flag and "save" in meta:
                            self._shutdown_save = meta["save"]
                        self._update_weight_image_spec(
                            mode=meta.get("weight_image_mode"),
                            panel_crop_w=meta.get("weight_panel_crop_w"),
                            panel_crop_h=meta.get("weight_panel_crop_h"),
                        )
                        self._sync_weight_image_spec_to_store()
                        if payload.selected_cycle_ids:
                            max_cycle = max(payload.selected_cycle_ids)
                            selected = [False] * max(max_cycle, len(self._cycle_selected))
                            for cycle_id in payload.selected_cycle_ids:
                                idx = int(cycle_id) - 1
                                if 0 <= idx < len(selected):
                                    selected[idx] = True
                            self._cycle_selected = selected
                        sni_raw = getattr(payload, "selected_node_ids", None)
                        if sni_raw:  # empty list = "no explicit preference", same as selected_cycle_ids treatment
                            sni_set = set(sni_raw)
                            all_known = set(self._stage_selected.keys()) | sni_set
                            self._stage_selected = {nid: (nid in sni_set) for nid in all_known}
                    elif envelope.message_type == MESSAGE_TYPE_PLAN_APPLY:
                        self._pending_plan_apply = payload
                    elif envelope.message_type == MESSAGE_TYPE_SCHEDULE_APPLY:
                        self._pending_schedule = payload
                    continue

                t = msg.get("type")
                if t == "status":
                    self._stop_flag = msg.get("stop_requested", False)
                    self._gate_override = msg.get("gate_override", False)
                    self._suppress_rebuild = msg.get("suppress_rebuild", False)
                    self._force_rebuild = msg.get("force_rebuild", False)
                    self._paused = msg.get("paused", False)
                    self._preview_enabled = msg.get("preview_enabled", True)
                    self._scrub_editor_enabled = msg.get("scrub_editor_enabled", True)
                    cs = msg.get("cycle_selected")
                    if cs is not None:
                        self._cycle_selected = list(cs)
                    ns = msg.get("node_selected")
                    if isinstance(ns, dict):
                        self._stage_selected = {str(k): bool(v) for k, v in ns.items()}
                    if msg.get("skip_forward"):
                        self._skip_forward_pending = True
                    if msg.get("skip_back"):
                        self._skip_back_pending = True
                    if msg.get("save_now"):
                        self._save_now_pending = True
                    self._update_weight_image_spec(
                        mode=msg.get("weight_image_mode"),
                        panel_crop_w=msg.get("weight_panel_crop_w"),
                        panel_crop_h=msg.get("weight_panel_crop_h"),
                    )
                    self._last_run_control = RunControlPayload(
                        command="stop" if bool(self._stop_flag) else ("pause" if bool(self._paused) else "resume"),
                        selected_cycle_ids=self.selected_cycle_ids(),
                        gate_override=bool(self._gate_override),
                        preview_enabled=bool(self._preview_enabled),
                        scrub_editor_enabled=bool(self._scrub_editor_enabled),
                        metadata={
                            "source": "legacy_status",
                            "weight_image_mode": int(self._weight_image_mode),
                            "weight_panel_crop_w": int(self._weight_panel_crop_w),
                            "weight_panel_crop_h": int(self._weight_panel_crop_h),
                        },
                    )
                    self._sync_weight_image_spec_to_store()
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
        finally:
            self._recv_lock.release()

    # -- public API (mirrors _TransformerStatusOpenGLViewer) ---------------

    def set_save_restore_node(self, node: Any) -> None:
        """Inject the SaveRestoreNode reference for pull-model query routing."""
        self._save_restore_node = node
        self._sync_weight_image_spec_to_store()

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
        self._drain_status()  # keep receive buffer drained to prevent IPC deadlock
        return self._stop_flag

    def shutdown_save(self) -> Optional[bool]:
        """Return the save preference for the current shutdown, or None if no shutdown."""
        return self._shutdown_save

    def update_loss(self, channel_key: str, loss: float, aux: float = 0.0, ts: float = 0.0):
        # Loss values are owned exclusively by the SaveRestoreNode.
        # The GUI pulls them via the IPC query protocol; no direct push.
        pass

    def enqueue_frame(self, frame_dict: dict):
        """Push images and text to shared ring, signal GUI with just the cursor."""
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
            h, w = target_rgb.shape[0], target_rgb.shape[1]
            flags = SCRUB_FLAG_HAS_IMAGE | SCRUB_FLAG_HAS_TARGET | SCRUB_FLAG_HAS_OUTPUT
            ring_cursor = ring.push(
                step=0, round_id=0, ts=time.time(), loss=0.0,
                channel_key="preview", flags=flags,
                image_w=w, image_h=h,
                training_image=input_rgb,
                output_image=output_rgb,
                target_data=target_rgb,
                thumb0=None,
                thumb1=None,
                thumb2=None,
            )
            ring.write_text(
                ring_cursor,
                caption=str(frame_dict.get("caption", "")),
                titles=frame_dict.get("titles", ["target", "input", "output"]),
                rows=frame_dict.get("rows", [[], [], []]),
            )
        self._send({
            "type": "frame_signal",
            "ring_cursor": (int(ring_cursor) if ring_cursor is not None else None),
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
        h, w = target_rgb.shape[0], target_rgb.shape[1]

        flags = SCRUB_FLAG_HAS_IMAGE | SCRUB_FLAG_HAS_TARGET | SCRUB_FLAG_HAS_OUTPUT
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
            thumb0=None,
            thumb1=None,
            thumb2=None,
        )
        titles = list(panel_titles) if panel_titles else ["target", "input", "output"]
        rows = [list(r) for r in panel_rows] if panel_rows else [[], [], []]
        ring.write_text(ring_cursor, caption=str(caption), titles=titles, rows=rows)
        self._send({
            "type": "frame_signal",
            "ring_cursor": int(ring_cursor),
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

    def set_restore_state_callback(self, fn) -> None:
        self._on_restore = fn

    def set_checkpoint_backup_dir(self, path) -> None:
        self._send({"type": "checkpoint_backup_dir", "path": str(path)})

    def trim_graph_to_first_checkpoint(self) -> None:
        self._send({"type": "trim_graph"})

    def is_cycle_selected(self, cycle_local: int) -> bool:
        idx = int(cycle_local) - 1
        if idx < 0 or idx >= len(self._cycle_selected):
            return True
        return bool(self._cycle_selected[idx])

    _GATE_PARENT_STAGE: Dict[str, str] = {
        "gate_0_pregestation_eval": "stage_0_pregestation",
        "gate_1_gestation_eval":    "stage_1_gestation",
        "gate_berkeley":            "stage_2_berkeley",
        "gate_transformer":         "stage_r_transformer",
        "gate_generator":           "stage_g_generator",
        "gate_wave":                "stage_w_wave_classifier",
    }

    # Non-roster nodes that load data/run work on behalf of a parent training stage.
    # If the parent stage is deselected these nodes must also be skipped, regardless
    # of gate_override (unlike gate eval nodes they are NOT suppressed by gate_override
    # alone — only by the parent being off).
    _STAGE_SUPPORT_NODES: Dict[str, str] = {
        "build_flashcard_rows": "stage_g_generator",
        "stage_fake_feedback":  "stage_g_generator",
    }

    def is_node_selected(self, node_id: str) -> bool:
        self._drain_status()
        nid = str(node_id)
        if not bool(self._stage_selected.get(nid, True)):
            return False
        parent = self._GATE_PARENT_STAGE.get(nid)
        if parent is not None:
            if not bool(self._stage_selected.get(parent, True)):
                return False
            if self._gate_override:
                return False
        support_parent = self._STAGE_SUPPORT_NODES.get(nid)
        if support_parent is not None:
            if not bool(self._stage_selected.get(support_parent, True)):
                return False
        return True

    def selected_node_ids(self) -> List[str]:
        return [nid for nid, v in self._stage_selected.items() if bool(v)]

    def consume_skip_forward(self) -> bool:
        self._drain_status()
        if self._skip_forward_pending:
            self._skip_forward_pending = False
            return True
        return False

    def consume_skip_back(self) -> bool:
        self._drain_status()
        if self._skip_back_pending:
            self._skip_back_pending = False
            return True
        return False

    def consume_save_now(self) -> bool:
        self._drain_status()
        if self._save_now_pending:
            self._save_now_pending = False
            return True
        return False

    def gate_override_enabled(self) -> bool:
        return self._gate_override

    def suppress_rebuild_enabled(self) -> bool:
        return self._suppress_rebuild

    def force_rebuild_enabled(self) -> bool:
        return self._force_rebuild

    def paused(self) -> bool:
        return self._paused

    def preview_enabled(self) -> bool:
        return self._preview_enabled

    def scrub_editor_enabled(self) -> bool:
        return self._scrub_editor_enabled

    def gui_connected(self) -> bool:
        return bool(self.enabled and self._conn is not None)

    def weight_image_spec(self) -> Dict[str, int]:
        return {
            "mode": int(self._weight_image_mode) | 0x100,
            "panel_crop_w": int(self._weight_panel_crop_w),
            "panel_crop_h": int(self._weight_panel_crop_h),
        }

    def selected_cycle_ids(self) -> List[int]:
        return [int(i + 1) for i, v in enumerate(self._cycle_selected) if bool(v)]

    def current_run_control(self) -> RunControlPayload:
        return self._last_run_control

    def consume_pending_plan_apply(self) -> Optional[PlanApplyPayload]:
        """Return and clear any plan_apply payload queued from the GUI, or None."""
        self._drain_status()
        payload = self._pending_plan_apply
        self._pending_plan_apply = None
        return payload

    def consume_pending_schedule(self) -> Optional[ScheduleApplyPayload]:
        """Return and clear any schedule_apply payload queued from the GUI, or None."""
        self._drain_status()
        payload = self._pending_schedule
        self._pending_schedule = None
        return payload

    def send_schedule_apply(
        self,
        schedule: "TrainingSchedule",
        *,
        replace_current: bool = True,
        reason: str = "",
        session_id: str = "",
    ) -> None:
        """Send a ScheduleApplyPayload to the connected worker backend."""
        payload = ScheduleApplyPayload(
            schedule=schedule,
            replace_current=replace_current,
            reason=reason,
        )
        envelope = make_envelope(MESSAGE_TYPE_SCHEDULE_APPLY, payload, session_id=session_id)
        self._send(envelope.to_dict())

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
