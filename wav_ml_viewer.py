import math
import queue as _viewer_frame_queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

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

# ---------------------------------------------------------------------------
# Loss logging constants and binary record format
# ---------------------------------------------------------------------------
LOSS_STAGE_CLASSIFIER = 0
LOSS_STAGE_GENERATOR = 1
LOSS_STAGE_DISCRIMINATOR = 2
LOSS_STAGE_TRANSFORMER = 3
LOSS_STAGE_WAVE_CLASSIFIER = 4
LOSS_STAGE_WAVE_CLASSIFIER_EVAL = 5

_LOSS_STAGE_NAMES: Dict[int, str] = {
    LOSS_STAGE_CLASSIFIER: "cls",
    LOSS_STAGE_GENERATOR: "gen",
    LOSS_STAGE_DISCRIMINATOR: "disc",
    LOSS_STAGE_TRANSFORMER: "trans",
    LOSS_STAGE_WAVE_CLASSIFIER: "wcls",
    LOSS_STAGE_WAVE_CLASSIFIER_EVAL: "wcls_eval",
}

# Legacy fixed colours kept for backward-compat history loading only.
_LOSS_STAGE_COLORS_LEGACY: Dict[int, tuple] = {
    LOSS_STAGE_CLASSIFIER:          (80,  200, 220),
    LOSS_STAGE_GENERATOR:           (80,  200, 100),
    LOSS_STAGE_DISCRIMINATOR:       (240, 140,  40),
    LOSS_STAGE_TRANSFORMER:         (240, 220,  60),
    LOSS_STAGE_WAVE_CLASSIFIER:     (220,  80, 220),
    LOSS_STAGE_WAVE_CLASSIFIER_EVAL:(160, 120, 255),
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

    Uses the golden angle offset (137.508°) so that any prefix of N channels
    is well-distributed even if total changes over time.
    """
    if total <= 0:
        total = 1
    hue = (index * 137.508) % 360.0
    return _hsl_to_rgb(hue, 0.72, 0.58)


def _channel_key_from_stage_id(stage_id: int) -> str:
    """Convert legacy integer stage_id to a canonical channel key string."""
    return _LOSS_STAGE_NAMES.get(int(stage_id), f"stage_{stage_id}")


def make_channel_key(
    node_id: str = "",
    stage_id: int = -1,
    lora_slot: str = "",
) -> str:
    """Build a unique channel key for loss graph tracking.

    Format: ``<base>`` or ``<base>|<lora_slot>`` when a LoRA adapter is active.
    ``base`` is derived from node_id (preferred) or stage_id (fallback).
    """
    base = ""
    if node_id:
        base = str(node_id).strip().lower()
    elif stage_id >= 0:
        base = _channel_key_from_stage_id(stage_id)
    else:
        base = "unknown"
    lora = str(lora_slot).strip()
    if lora:
        return f"{base}|{lora}"
    return base

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
        # start_time retained for API compatibility but no longer used; ts is wall-clock.
        self._file = None
        try:
            existing_bytes = int(self._path.stat().st_size) if self._path.exists() else 0
            self._counter = int(existing_bytes // LOSS_RECORD_DTYPE.itemsize)
            self._file = open(self._path, "ab")
        except Exception as e:
            self._counter = 0
            print(f"[loss-logger] could not open {path}: {e}", flush=True)

    def log(self, round_idx: int, stage_id: int, loss: float, aux: float = 0.0):
        if self._file is None:
            return
        self._counter += 1
        ts = time.time()  # wall-clock Unix timestamp; consistent across sessions
        rec = np.zeros(1, dtype=LOSS_RECORD_DTYPE)
        rec["step"][0]  = max(-(2**31), min(2**31 - 1, int(self._counter)))
        rec["round"][0] = max(-32768,   min(32767,     int(round_idx)))
        rec["stage"][0] = int(stage_id) & 0xFF
        rec["loss"][0]  = float(loss) if math.isfinite(float(loss)) else float("nan")
        rec["aux"][0]   = float(aux)  if math.isfinite(float(aux))  else float("nan")
        rec["ts"][0]    = min(float(ts), 3.4e38)
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
        """Load all records from a loss_log.bin file into a structured numpy array."""
        p = Path(path)
        if not p.exists() or p.stat().st_size == 0:
            return np.zeros(0, dtype=LOSS_RECORD_DTYPE)
        data = p.read_bytes()
        n = len(data) // LOSS_RECORD_DTYPE.itemsize
        return np.frombuffer(data[: n * LOSS_RECORD_DTYPE.itemsize], dtype=LOSS_RECORD_DTYPE).copy()


def _wmap_thermal_rgb(t: np.ndarray) -> np.ndarray:
    """Map normalised float array t ∈ [0,1] → (H, W, 3) uint8 thermal colours.
    Ramp: black → blue → cyan → green → yellow → red."""
    r = np.clip(t * 4.0 - 3.0, 0.0, 1.0)
    g = np.clip(np.minimum(t * 4.0 - 1.0, 3.0 - t * 4.0), 0.0, 1.0)
    b = np.clip(2.0 - t * 4.0, 0.0, 1.0)
    return np.round(np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def _wmap_flat_to_square(flat: torch.Tensor, sq: int) -> np.ndarray:
    """Flatten live weights → (sq, sq, 3) thermal image, zero-padding to sq²."""
    n = sq * sq
    arr = flat.cpu().numpy().astype(np.float32)
    if arr.shape[0] < n:
        arr = np.pad(arr, (0, n - arr.shape[0]))
    arr = arr[:n]
    lo, hi = float(arr.min()), float(arr.max())
    t = (arr - lo) / (hi - lo) if hi > lo + 1e-8 else np.zeros_like(arr)
    return _wmap_thermal_rgb(t.reshape(sq, sq))


def _wmap_diff_overlay(diff: torch.Tensor, sq: int) -> np.ndarray:
    """Absolute diff tensor → (sq, sq) float32 in [0, 1] for overlay intensity."""
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
    ):
        self.enabled = bool(enabled)
        self.image_h = max(8, int(image_hw[0]))
        self.image_w = max(8, int(image_hw[1]))
        _ = max(1, int(scale))

        self.panel_w = max(8, min(256, int(self.image_w)))
        self.panel_h = max(8, min(256, int(self.image_h)))
        self.num_panels = 3
        self.top_bar_h = 56
        self.graph_h = max(0, int(graph_h))
        # +2 columns: one sidebar on each side of the 3 main panels
        self.window_w = int(self.panel_w * (self.num_panels + 2))
        self.window_h = int(self.top_bar_h + (self.panel_h * 2) + self.graph_h)
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
        # Blocking queue: enqueue_frame() blocks when full, back-pressuring the
        # preview worker and, through it, the training loop.  This ensures no
        # preview frames are ever dropped when the user is scanning scrollback.
        # 16384 frames × ~192 KB/frame ≈ 3 GB ceiling; tune as needed.
        self._frame_buffer: _viewer_frame_queue.Queue = _viewer_frame_queue.Queue(maxsize=16384)
        # Slew: elapsed time controls how many buffered frames are drained per pump().
        # Buffer fill-depth drives target drain rate (fast when full, slow when empty).
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
        self._loss_display_rows: List[str] = []
        self._panel_text_rgb = [
            np.full((self.panel_h, self.panel_w, 3), 14, dtype=np.uint8),
            np.full((self.panel_h, self.panel_w, 3), 14, dtype=np.uint8),
            np.full((self.panel_h, self.panel_w, 3), 14, dtype=np.uint8),
        ]
        self._top_bar_rgb = np.full((self.top_bar_h, self.window_w, 3), 18, dtype=np.uint8)

        self._cycle_selected: List[bool] = []
        self._gate_override = False
        self._control_boxes: List[Tuple[str, int, Tuple[int, int, int, int]]] = []
        self._graph_worker_hello: Dict[str, Any] = {}
        self._graph_plan_snapshot: Optional[Dict[str, Any]] = None
        self._graph_runtime_snapshot: Optional[Dict[str, Any]] = None
        self._graph_execution_events: deque = deque(maxlen=256)
        self.set_cycle_roster(total_cycles=int(cycle_slots))

        self._loss_graph_data: Dict[int, deque] = {}
        # Parallel wall-clock timestamps (time.time()) for each loss value, same maxlen.
        self._loss_graph_ts: Dict[int, deque] = {}
        self._graph_dirty = False
        self._graph_rgb: Optional[np.ndarray] = (
            np.full((self.graph_h, self.window_w, 3), 14, dtype=np.uint8)
            if self.graph_h > 0 else None
        )

        # ── Sidebar state ─────────────────────────────────────────────────────
        # -inf forces an immediate first render on the first _present() call.
        self._sidebar_dirty: bool = True
        self._preview_work_queue_ref: Optional[Any] = None
        self._weight_model_refs: Dict[str, Any] = {}
        # Optional per-model state-dicts for the red diff overlay:
        #   _weight_disk_states  – weights loaded from the on-disk checkpoint file
        #   _weight_ckpt_states  – weights from the pipeline checkpoint (to-be-merged)
        self._weight_disk_states: Dict[str, Optional[Dict[str, Any]]] = {}
        self._weight_ckpt_states: Dict[str, Optional[Dict[str, Any]]] = {}
        # ── Scrub / history ───────────────────────────────────────────────────
        # Weight-map snapshots accumulated at sidebar render rate (4 Hz).
        # _scrub_offset = 0 → live; N → show the snapshot N ticks back.
        self._weight_history_maxlen: int = 512
        self._weight_snapshot_deque: deque = deque(maxlen=self._weight_history_maxlen)
        # Played preview frames – paired 1-to-1 with weight_snapshot_deque so the
        # same index yields the co-occurring preview image at that point in time.
        self._played_frames_deque: deque = deque(maxlen=self._weight_history_maxlen)
        # Loss series lengths recorded once per sidebar tick so the graph renderer
        # can map each snapshot → x-pixel for the cache-region band and cursor.
        self._loss_count_at_snap_deque: deque = deque(maxlen=self._weight_history_maxlen)
        # Most-recently-uploaded preview frame (set each time a frame is consumed).
        self._last_displayed_frame: Optional[dict] = None
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
        # e.g. 4 snapshots across 512-frame cache → stride 128 visual ticks
        #      @ 4 Hz sidebar rate ≈ one snapshot every ~32 seconds.
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
        # Checkpoint thumbnail registry — ordered list of discovered checkpoint
        # entries beyond the in-memory cache.  Each entry is a dict with keys:
        #   round_id, cycle, thumb_path (Path or None), loss_counts ({sid:int})
        # Populated by register_checkpoint_thumbnail() and set_checkpoint_backup_dir().
        # Sorted oldest→newest (index 0 = oldest checkpoint).
        self._checkpoint_thumbs: List[Dict[str, Any]] = []
        # Background thread for weight image + state_dict snapshot.
        # Main thread fires it and checks for completion; never blocks.
        self._weight_snap_thread: Optional[threading.Thread] = None
        self._weight_snap_result: Optional[Dict[str, Any]] = None
        # Which model to display in the right sidebar (None = first available).
        self._active_weight_model_name: Optional[str] = None
        # Delta logging: print param name when max|Δ| exceeds this threshold.
        self._weight_delta_threshold: float = 0.01
        # Previous sparse snapshot for delta comparison.
        self._prev_weight_snap: Dict[str, Dict[str, Any]] = {}

        # ── Pull-model data (populated by SaveRestoreNode responses) ──────
        # Per-channel loss cursors: channel_key → highest step we have locally.
        self._sr_loss_cursors: Dict[str, int] = {}
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
        # Per-channel loss visibility (True = visible). Missing key → visible.
        self._loss_channel_visible: Dict[int, bool] = {}
        # Legend hit-boxes for channel toggle clicks: list of (sid, x0, y0, x1, y1)
        # in *graph-local* pixel coords. Translated to window coords at click time.
        self._legend_hit_boxes: List[Tuple[int, int, int, int, int]] = []

        # Left sidebar: button panel (top) + scrub dial (bottom).
        # Right sidebar: weight map spans from top_bar to window bottom (covers graph row).
        # Fraction of the total loss history before which the graph is trimmed.
        # 0.0 = show everything; >0 = the graph's left edge starts here.
        # Set via trim_graph_to_first_checkpoint() after loading history + markers.
        self._graph_display_start_frac: float = 0.0

        self._cache_map_rgb   = np.full((self.panel_h, self.panel_w, 3), 14, dtype=np.uint8)
        self._frame_knob_rgb  = np.full((self.panel_h, self.panel_w, 3), 14, dtype=np.uint8)
        # weight_map is tall: 2*panel_h + graph_h to fill full right sidebar including graph row.
        self._weight_map_rgb  = np.full((2 * self.panel_h + self.graph_h, self.panel_w, 3), 14, dtype=np.uint8)

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

    def _render_top_bar(self) -> np.ndarray:
        out = np.full((self.top_bar_h, self.window_w, 3), 18, dtype=np.uint8)
        self._control_boxes = []
        try:
            from PIL import Image, ImageDraw, ImageFont

            im = Image.fromarray(out).convert("RGB")
            draw = ImageDraw.Draw(im)
            font = ImageFont.load_default()

            draw.rectangle([(0, 0), (self.window_w - 1, self.top_bar_h - 1)], fill=(18, 22, 28))
            draw.line(
                [(0, self.top_bar_h - 1), (self.window_w - 1, self.top_bar_h - 1)],
                fill=(70, 76, 88),
                width=1,
            )
            cap = str(self._caption).strip()
            if cap:
                draw.text((6, 4), cap[: max(16, (self.window_w // 6) - 4)], fill=(230, 234, 240), font=font)

            x = 8
            y = 24
            for i, is_on in enumerate(self._cycle_selected):
                token_w = 42
                if x + token_w >= (self.window_w - 170):
                    break
                box = (x, y, x + 11, y + 11)
                fill = (52, 120, 66) if bool(is_on) else (36, 40, 44)
                draw.rectangle([box[0], box[1], box[2], box[3]], outline=(186, 194, 204), fill=fill)
                if bool(is_on):
                    draw.line([(box[0] + 2, box[1] + 6), (box[0] + 5, box[1] + 9)], fill=(236, 244, 248), width=1)
                    draw.line([(box[0] + 5, box[1] + 9), (box[0] + 9, box[1] + 2)], fill=(236, 244, 248), width=1)
                draw.text((x + 15, y - 1), f"C{i + 1}", fill=(224, 230, 236), font=font)
                self._control_boxes.append(("cycle", int(i), box))
                x += token_w

            ox = max(x + 4, self.window_w - 166)
            oy = y
            o_box = (ox, oy, ox + 11, oy + 11)
            o_fill = (126, 84, 36) if bool(self._gate_override) else (36, 40, 44)
            draw.rectangle([o_box[0], o_box[1], o_box[2], o_box[3]], outline=(186, 194, 204), fill=o_fill)
            if bool(self._gate_override):
                draw.line([(o_box[0] + 2, o_box[1] + 6), (o_box[0] + 5, o_box[1] + 9)], fill=(236, 244, 248), width=1)
                draw.line([(o_box[0] + 5, o_box[1] + 9), (o_box[0] + 9, o_box[1] + 2)], fill=(236, 244, 248), width=1)
            draw.text((ox + 15, oy - 1), "Override gates", fill=(230, 208, 170), font=font)
            self._control_boxes.append(("override", -1, o_box))

            # ── START / STOP buttons (right side of top bar, row y=4) ─────────
            connected = self.has_training_connection()
            stopping = self._shutdown_save is not None
            btn_y = 4
            btn_h = 13
            btn_right_margin = 6
            # Place buttons from right edge leftward.
            bx = self.window_w - btn_right_margin

            if connected and stopping:
                # Shutdown in progress — show status label
                label_pending = "STOPPING..."
                lw_p = len(label_pending) * 7 + 10
                p_box = (bx - lw_p, btn_y, bx, btn_y + btn_h)
                draw.rectangle(
                    [p_box[0], p_box[1], p_box[2], p_box[3]],
                    outline=(120, 120, 60), fill=(80, 80, 30),
                )
                draw.text((p_box[0] + 5, btn_y + 1), label_pending, fill=(220, 220, 160), font=font)
            elif connected:
                # STOP (no save) — red
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

                # STOP+SAVE — green
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
                # START — blue (only when disconnected and launch script available)
                label_start = "START"
                lw_start = len(label_start) * 7 + 10
                start_box = (bx - lw_start, btn_y, bx, btn_y + btn_h)
                draw.rectangle(
                    [start_box[0], start_box[1], start_box[2], start_box[3]],
                    outline=(60, 100, 180), fill=(36, 64, 130),
                )
                draw.text((start_box[0] + 5, btn_y + 1), label_start, fill=(200, 220, 250), font=font)
                self._control_boxes.append(("start", -1, start_box))

            active = self.selected_cycle_ids()
            active_txt = ",".join(str(i) for i in active) if len(active) > 0 else "none"
            draw.text(
                (6, self.top_bar_h - 14),
                f"cycles={active_txt} gate_override={1 if self._gate_override else 0}",
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

    # ── Checkpoint navigation helpers ──────────────────────────────────────────

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
        ref_sid = max(
            self._loss_graph_data.keys(),
            key=lambda s: len(self._loss_graph_data[s]),
            default=None,
        )
        if ref_sid is None:
            return None

        # Current position in loss-count space.
        cur_off = int(self._scrub_offset)
        if cur_off > 0 and cur_off <= slen:
            cur_idx = slen - cur_off
            cur_loss_n = int(snap_list[cur_idx].get(ref_sid, 0))
        else:
            # Live position = total length
            cur_loss_n = len(self._loss_graph_data.get(ref_sid, []))

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

    # ── Checkpoint thumbnail registry ──────────────────────────────────────────

    def register_checkpoint_thumbnail(
        self,
        round_id: int,
        cycle: int,
        thumb_path: Optional["Path"] = None,
        loss_counts: Optional[Dict[int, int]] = None,
    ) -> None:
        """Register a checkpoint entry for extended scrub navigation.

        Called at checkpoint-save time (via notify) and at startup when
        discovering existing checkpoints on disk.  Entries are kept sorted
        oldest→newest by (round_id, cycle).
        """
        entry: Dict[str, Any] = {
            "round_id": int(round_id),
            "cycle": int(cycle),
            "thumb_path": thumb_path,
            "loss_counts": loss_counts or {},
        }
        # Insert in sorted order
        idx = len(self._checkpoint_thumbs)
        for i, e in enumerate(self._checkpoint_thumbs):
            if (e["round_id"], e["cycle"]) > (round_id, cycle):
                idx = i
                break
            if e["round_id"] == round_id and e["cycle"] == cycle:
                # Update existing entry
                self._checkpoint_thumbs[i] = entry
                return
        self._checkpoint_thumbs.insert(idx, entry)

    def _load_checkpoint_thumbnail(self, ckpt_idx: int) -> Optional[np.ndarray]:
        """Load a checkpoint thumbnail image and scale to the weight-map panel size.

        Returns an (H, W, 3) uint8 RGB array suitable for the right sidebar,
        or None if no thumbnail is available.
        """
        if ckpt_idx < 0 or ckpt_idx >= len(self._checkpoint_thumbs):
            return None
        entry = self._checkpoint_thumbs[ckpt_idx]
        thumb_path = entry.get("thumb_path")
        if thumb_path is None or not Path(thumb_path).exists():
            return self._render_checkpoint_placeholder(entry)
        try:
            from PIL import Image
            img = Image.open(str(thumb_path)).convert("L")
            # Target size: weight_map panel dimensions
            H = 2 * self.panel_h + self.graph_h
            W = self.panel_w
            img = img.resize((W, W), Image.LANCZOS)  # square, then pad/crop to H
            grey = np.asarray(img, dtype=np.uint8)
            # Convert greyscale to RGB (all channels equal)
            rgb = np.zeros((H, W, 3), dtype=np.uint8)
            rgb[:, :] = (14, 16, 20)  # dark background
            # Centre the square thumbnail vertically
            y0 = max(0, (H - W) // 2)
            paste_h = min(W, H - y0)
            rgb[y0:y0 + paste_h, :, 0] = grey[:paste_h]
            rgb[y0:y0 + paste_h, :, 1] = grey[:paste_h]
            rgb[y0:y0 + paste_h, :, 2] = grey[:paste_h]
            # Add label overlay
            try:
                from PIL import Image as _PILImg, ImageDraw, ImageFont
                overlay = _PILImg.fromarray(rgb)
                draw = ImageDraw.Draw(overlay)
                font = ImageFont.load_default()
                r = entry.get("round_id", "?")
                c = entry.get("cycle", "?")
                draw.text((4, 4), f"CKPT r{r} c{c}", fill=(200, 170, 40), font=font)
                draw.text((4, H - 14), "weight snapshot", fill=(120, 120, 140), font=font)
                rgb = np.asarray(overlay, dtype=np.uint8)
            except Exception:
                pass
            return rgb
        except Exception:
            return self._render_checkpoint_placeholder(entry)

    def _render_checkpoint_placeholder(self, entry: Dict[str, Any]) -> np.ndarray:
        """Render a placeholder when no thumbnail file exists."""
        H = 2 * self.panel_h + self.graph_h
        W = self.panel_w
        rgb = np.full((H, W, 3), (14, 16, 20), dtype=np.uint8)
        try:
            from PIL import Image, ImageDraw, ImageFont
            im = Image.fromarray(rgb)
            draw = ImageDraw.Draw(im)
            font = ImageFont.load_default()
            r = entry.get("round_id", "?")
            c = entry.get("cycle", "?")
            draw.text((4, 4), f"CKPT r{r} c{c}", fill=(200, 170, 40), font=font)
            draw.text((4, 20), "no thumbnail", fill=(80, 80, 100), font=font)
            rgb = np.asarray(im, dtype=np.uint8)
        except Exception:
            pass
        return rgb

    # ── Sidebar public API ─────────────────────────────────────────────────────

    def set_queue_refs(self, work_queue: Any) -> None:
        """Register the preview work queue so the cache map can read its depth."""
        self._preview_work_queue_ref = work_queue

    def _history_snap_count(self) -> int:
        """Total scrub positions: in-memory cache + on-disk checkpoint thumbnails."""
        cache_len = max(
            len(self._weight_snapshot_deque),
            len(self._played_frames_deque),
            len(self._loss_count_at_snap_deque),
        )
        return cache_len + len(self._checkpoint_thumbs)

    def _cache_snap_count(self) -> int:
        """In-memory cache positions only (without checkpoint thumbnails)."""
        return max(
            len(self._weight_snapshot_deque),
            len(self._played_frames_deque),
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
        # _checkpoint_thumbs is sorted oldest→newest, so index for newest-first:
        return n - ckpt_offset

    def _clone_history_frame(self, frame: Optional[dict]) -> Optional[dict]:
        if not isinstance(frame, dict):
            return None
        images = frame.get("images", None)
        if not isinstance(images, list) or len(images) != 3:
            return None
        out_images: List[np.ndarray] = []
        for img in images:
            if img is None:
                return None
            out_images.append(np.ascontiguousarray(np.asarray(img, dtype=np.uint8)).copy())
        return {
            "images": out_images,
            "caption": str(frame.get("caption", "")),
            "titles": [str(x) for x in frame.get("titles", self._panel_titles)],
            "rows": [list(r) for r in frame.get("rows", self._panel_rows)],
            "step_txt": str(frame.get("step_txt", "")),
        }

    def _record_history_frame(self, frame: Optional[dict]) -> None:
        history_frame = self._clone_history_frame(frame)
        if history_frame is None:
            return
        self._played_frames_deque.append(history_frame)
        self._loss_count_at_snap_deque.append(
            {sid: len(dq) for sid, dq in self._loss_graph_data.items()}
        )
        self._weight_snapshot_deque.append(
            {"weight_map": np.ascontiguousarray(np.asarray(self._weight_map_rgb, dtype=np.uint8)).copy()}
        )

    def _history_frame_at_offset(self, offset: int) -> Optional[dict]:
        # If in checkpoint zone, no preview frame is available.
        if self._scrub_in_checkpoint_zone() and offset > self._cache_snap_count():
            return None
        history = list(self._played_frames_deque)
        hlen = len(history)
        if hlen <= 0:
            return None
        off = max(1, min(hlen, int(offset)))
        idx = max(0, min(hlen - 1, hlen - off))
        return history[idx]

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
        weight_map = blob.get("weight_map", None)
        if weight_map is None:
            return None
        return np.ascontiguousarray(np.asarray(weight_map, dtype=np.uint8))

    def _apply_frame_to_display(self, frame: Optional[dict]) -> None:
        if not isinstance(frame, dict):
            return
        imgs = frame.get("images", None)
        if isinstance(imgs, list) and len(imgs) == 3 and self._textures is not None:
            for i, tid in enumerate(self._textures["img"]):
                self._upload_texture(int(tid), np.asarray(imgs[i], dtype=np.uint8))
        new_caption = str(frame.get("caption", ""))
        new_titles = [str(x) for x in frame.get("titles", self._panel_titles)]
        new_rows = [list(r) for r in frame.get("rows", self._panel_rows)]
        if new_titles != self._panel_titles or new_rows != self._panel_rows:
            self._panel_titles = new_titles
            self._panel_rows = new_rows
            self._panel_text_dirty = True
        if new_caption != self._caption:
            self._caption = new_caption
            self._top_bar_dirty = True
        new_loss_rows = list(frame.get("loss_rows", self._loss_display_rows))
        if new_loss_rows != self._loss_display_rows:
            self._loss_display_rows = new_loss_rows
            self._sidebar_dirty = True
        new_step_txt = str(frame.get("step_txt", ""))
        if new_step_txt != self._last_step_txt:
            self._last_step_txt = new_step_txt
        self._last_displayed_frame = frame

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

        Pass only models that are actually being run right now – the map shows
        exactly what is given.  Optionally supply state-dicts for the red diff
        overlay:
          disk_states  – {name: state_dict} loaded straight from the on-disk file
          ckpt_states  – {name: state_dict} from the pipeline checkpoint that
                         holds data to be integrated with the base model
        The red overlay pixel intensity = normalised |disk_weight − ckpt_weight|.
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

    def set_weight_snap_dir(self, path) -> None:
        """Set the directory where per-snapshot weight .pt backup files are written.

        Files are named ``snap_{snap_total:08d}.pt`` and deleted automatically
        when the corresponding entry ages out of the sparse deque.
        """
        self._weight_snap_dir = Path(path) if path is not None else None

    def get_restore_state_dicts(self, offset: int) -> Optional[Dict[str, Dict]]:
        """Return the weight state-dicts for the snapshot at *offset* positions back.

        offset=1 → most recent snapshot; offset=N → Nth most recent (oldest = N where
        N == len(sparse deque)).  Mirrors the same indexing used by the scrub display.
        Returns ``None`` if the sparse deque is empty."""
        snaps = list(self._weight_state_sparse_deque)
        slen = len(snaps)
        if not snaps:
            return None
        off = max(1, min(slen, int(offset)))
        idx = max(0, min(slen - 1, slen - off))
        return snaps[idx]["states"]

    # ── Sidebar render methods ─────────────────────────────────────────────────

    def _render_cache_map(self) -> np.ndarray:
        """Pixel-grid occupancy map: work queue (green) + frame buffer (blue)."""
        H, W = self.panel_h, self.panel_w
        out = np.full((H, W, 3), np.array([14, 16, 20], dtype=np.uint8), dtype=np.uint8)
        wq_cap  = 1024
        wq_used = int(self._preview_work_queue_ref.qsize()) if self._preview_work_queue_ref is not None else 0
        fb_cap  = 16384
        fb_used = int(self._frame_buffer.qsize())
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
            draw.text((margin, y0_wq + section_h + margin), f"frame buf  {fb_used}/{fb_cap}", fill=(170, 200, 240), font=font)
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
        """256×256 circular dial — arc sweeps 300° showing fill_frac [0, 1]."""
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
            # Progress arc: 300° sweep starting at -210° (just past bottom-left)
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

            # ── Step counter ──────────────────────────────────────────────
            step_txt = str(self._last_step_txt)
            if step_txt:
                _stw = font.getlength(step_txt) if hasattr(font, "getlength") else len(step_txt) * 6
                draw.text((max(4, W - 4 - int(_stw)), y_info + 36),
                          step_txt, fill=(180, 200, 220), font=font)

            # ── Prev / Next checkpoint buttons ────────────────────────────
            has_ckpts = len(self._disk_save_loss_counts) > 0
            nav_y0 = y_info + 40
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

            # Loss display below checkpoint nav
            y_loss = nav_y0 + nav_h + 16
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
        """Per-model square weight images with optional disk-vs-checkpoint diff overlay.

        Layout (N active models, stacked vertically):
          Each model occupies a ``slot_h × W`` band, with a ``label_h`` strip below.

        Square image logic:
          sq = ceil(sqrt(n_params))  →  minimal-waste square for this model.
          • If sq*sq ≤ slot_h*W  (fits 1:1): draw at 1:1, centred, waste = sq²−n_params.
          • If sq*sq  > slot_h*W  (too big):  downscale sq×sq → slot_h×W with Lanczos
            (PIL LANCZOS preserves detail far better than stride-sampling).

        Red overlay (requires disk_states and ckpt_states passed to set_weight_model_refs):
          For each pixel, saturation of red = normalised |disk_weight − ckpt_weight|.
          If either state dict is absent the overlay is skipped silently."""
        H, W = 2 * self.panel_h + self.graph_h, self.panel_w
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

        # Only render the single active model – it gets the full height.
        _active_name = self._active_weight_model_name
        if _active_name is None or _active_name not in refs:
            _active_name = next(iter(refs))
        label_h = 12
        slot_h = max(1, H - label_h)
        y = 0

        for name, model in [(_active_name, refs[_active_name])]:
            n_params = 0
            mode_str = ""
            try:
                from PIL import Image as _PIL_Image, ImageDraw as _PIL_Draw, ImageFont as _PIL_Font

                # ── Flatten live parameters in named_parameters order ──────────
                # Use state_dict() for thread-safe snapshot — avoids racing
                # with optimizer.step() which mutates parameter tensors in-place.
                _sd_snap = {k: v.detach().float().reshape(-1).cpu()
                            for k, v in model.state_dict().items()}
                param_keys = list(_sd_snap.keys())
                live_flat = torch.cat(list(_sd_snap.values()))
                n_params = int(live_flat.numel())

                # ── Minimal-waste square side length ──────────────────────────
                sq = math.isqrt(n_params)
                if sq * sq < n_params:
                    sq += 1          # sq*sq is the smallest perfect-square ≥ n_params
                waste = sq * sq - n_params

                slot_px = slot_h * W
                needs_reduction = (sq * sq > slot_px)

                # ── Build base thermal image at sq×sq ─────────────────────────
                sq_img = _wmap_flat_to_square(live_flat, sq)  # (sq, sq, 3) uint8

                # ── Precompute diff map (sq×sq float32) if both states present ─
                disk_sd = self._weight_disk_states.get(name)
                ckpt_sd = self._weight_ckpt_states.get(name)
                diff_map: Optional[np.ndarray] = None
                if disk_sd is not None and ckpt_sd is not None:
                    disk_flat = _wmap_build_flat(disk_sd, param_keys, model)
                    ckpt_flat = _wmap_build_flat(ckpt_sd, param_keys, model)
                    diff_map = _wmap_diff_overlay((disk_flat - ckpt_flat).abs(), sq)

                # ── Place into slot ───────────────────────────────────────────
                if not needs_reduction:
                    # 1:1 – apply overlay then centre the sq×sq image in the slot
                    mode_str = f"1:1  waste={waste:,}"
                    if diff_map is not None:
                        sq_img = _wmap_apply_red(sq_img, diff_map)
                    cx = max(0, (W  - sq) // 2)
                    cy = max(0, (slot_h - sq) // 2)
                    draw_h = min(sq, slot_h)
                    draw_w = min(sq, W)
                    out[y + cy: y + cy + draw_h, cx: cx + draw_w] = sq_img[:draw_h, :draw_w]
                else:
                    # Lanczos downscale sq×sq → slot_h×W (detail-preserving),
                    # then apply the diff overlay on the downscaled images so the
                    # per-pixel diff is computed at display resolution.
                    mode_str = f"↓{sq}→{slot_h}×{W}"
                    slot_img = np.asarray(
                        _PIL_Image.fromarray(sq_img).resize((W, slot_h), _PIL_Image.LANCZOS),
                        dtype=np.uint8,
                    )
                    if diff_map is not None:
                        diff_small = np.asarray(
                            _PIL_Image.fromarray(
                                (diff_map * 255.0).astype(np.uint8)
                            ).resize((W, slot_h), _PIL_Image.LANCZOS),
                            dtype=np.float32,
                        ) / 255.0
                        slot_img = _wmap_apply_red(slot_img, diff_small)
                    out[y: y + slot_h, :W] = slot_img

            except Exception:
                pass

            # ── Label strip ───────────────────────────────────────────────────
            try:
                from PIL import Image as _PIL_Image, ImageDraw as _PIL_Draw, ImageFont as _PIL_Font
                ly = y + slot_h
                strip = _PIL_Image.fromarray(out[ly: ly + label_h, :W])
                draw = _PIL_Draw.Draw(strip)
                font = _PIL_Font.load_default()
                draw.rectangle([(0, 0), (W - 1, label_h - 1)], fill=(20, 24, 30))
                label = f"{name}  {n_params:,}  {mode_str}"
                draw.text((2, 1), label, fill=(140, 150, 165), font=font)
                out[ly: ly + label_h, :W] = np.asarray(strip, dtype=np.uint8)
            except Exception:
                pass

            y += slot_h + label_h
            if y >= H:
                break
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

        # Slew the drain rate: buffer fill-depth drives target frame interval.
        _pending = self._frame_buffer.qsize()
        fill = float(min(_pending, 256)) / 256.0
        target_dt = self._anim_dt_fast + (self._anim_dt_slow - self._anim_dt_fast) * ((1.0 - fill) ** 2)
        _slew_elapsed = max(1e-4, now - self._last_slew_t)
        self._last_slew_t = now
        alpha = 1.0 - math.exp(-_slew_elapsed / max(1e-4, self._anim_slew_tau))
        self._anim_frame_dt += alpha * (target_dt - self._anim_frame_dt)

        # Drain all frames due according to elapsed time.
        last_frame = None
        while not self._frame_buffer.empty() and (now - self._last_anim_t) >= self._anim_frame_dt:
            try:
                last_frame = self._frame_buffer.get_nowait()
            except _viewer_frame_queue.Empty:
                break
            for sid, lv in last_frame.get("losses", {}).items():
                self.update_loss(int(sid), float(lv))
            self._record_history_frame(last_frame)
            self._last_anim_t += self._anim_frame_dt
        if self._scrub_offset == 0:
            if last_frame is not None:
                self._apply_frame_to_display(last_frame)
        else:
            history_frame = self._history_frame_at_offset(self._scrub_offset)
            if history_frame is not None:
                self._apply_frame_to_display(history_frame)

        # Sidebar: btn_panel + scrub_dial every pump (cheap); weight snap is count-driven.
        self._cache_map_rgb  = self._render_btn_panel()
        self._frame_knob_rgb = self._render_scrub_dial()
        if self._scrub_offset == 0:
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
                        self._weight_snapshot_deque[-1] = {"weight_map": _wmap.copy()}
                    else:
                        self._weight_snapshot_deque.append({"weight_map": _wmap.copy()})
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
                        {sid: len(dq) for sid, dq in self._loss_graph_data.items()}
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

        if self.graph_h > 0 and self._graph_dirty:
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

        # Right sidebar column – weight map spans the full content area including graph row.
        rx0 = int(self._col_x + self.num_panels * self.panel_w)
        self._draw_texture_px(int(self._textures["weight_map"]), rx0, self.top_bar_h, self.window_w, self.window_h)

        if self.graph_h > 0 and self._graph_rgb is not None:
            graph_y0 = int(self.top_bar_h + (2 * self.panel_h))
            # Graph only spans under the left 4 columns; right sidebar keeps the space.
            self._draw_texture_px(
                int(self._textures["graph"]),
                0, graph_y0, rx0, graph_y0 + self.graph_h,
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
                                self._played_frames_deque.clear()
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
                        # Legend channel-toggle clicks (graph area).
                        graph_y0 = int(self.top_bar_h + 2 * self.panel_h)
                        if self._legend_hit_boxes and graph_y0 <= yi < graph_y0 + self.graph_h:
                            gx = xi
                            gy = yi - graph_y0
                            for sid, lx0, ly0, lx1, ly1 in self._legend_hit_boxes:
                                if lx0 <= gx <= lx1 and ly0 <= gy <= ly1:
                                    cur = self._loss_channel_visible.get(sid, True)
                                    self._loss_channel_visible[sid] = not cur
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
        vertical marker (\u25bc) at the exact loss position where each disk checkpoint
        was written.  Thread-safe: can be called from any thread.
        """
        snapshot = {sid: len(dq) for sid, dq in self._loss_graph_data.items()}
        self._disk_save_loss_counts.append(snapshot)
        self._graph_dirty = True

    def notify_checkpoint_at_walltime(self, wall_ts: float) -> None:
        """Add a graph marker at the loss-series position nearest to wall_ts.

        Used at startup to place gold markers for checkpoint files that already
        existed on disk before this session (their mtime = wall_ts).  Finds the
        loss record whose timestamp is closest to wall_ts by binary-searching the
        parallel _loss_graph_ts deques, then records the corresponding deque index
        into _disk_save_loss_counts so the graph renderer draws a gold \u25bc there.
        """
        import bisect
        if not self._loss_graph_ts:
            # No ts data loaded yet — fall back to marking at current position.
            self.notify_pipeline_checkpoint_saved()
            return
        snapshot: Dict[int, int] = {}
        for sid, ts_dq in self._loss_graph_ts.items():
            if not ts_dq:
                continue
            ts_list = list(ts_dq)
            idx = bisect.bisect_left(ts_list, float(wall_ts))
            idx = max(0, min(len(ts_list) - 1, idx))
            snapshot[sid] = idx + 1  # 1-based count mirrors len(dq) convention
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
            # Use the newest .pt mtime in the sub-dir as the checkpoint time.
            t = max(f.stat().st_mtime for f in pts)
            _times.append(t)
            # Discover weight thumbnails and register for extended scrub.
            import re as _re
            for thumb in sub.glob("weight_thumb_r*_c*.png"):
                m = _re.search(r"weight_thumb_r(\d+)_c(\d+)\.png$", thumb.name)
                if m:
                    rid = int(m.group(1))
                    cid = int(m.group(2))
                    self.register_checkpoint_thumbnail(rid, cid, thumb_path=str(thumb))
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
        for snap in self._disk_save_loss_counts:
            for sid, count in snap.items():
                dq = self._loss_graph_data.get(sid)
                if dq is None or len(dq) == 0:
                    continue
                frac = float(count) / float(len(dq))
                min_frac = min(min_frac, frac)
        # 5% left of the first checkpoint, clamped to 0.
        self._graph_display_start_frac = max(0.0, min_frac - 0.05)
        self._graph_dirty = True

    def stop_requested(self) -> bool:
        return bool(self._stop_requested)

    def update_loss(self, stage_id: int, loss: float, aux: float = 0.0, ts: float = 0.0):
        """Record a per-step loss value and mark the graph dirty for redraw.

        ts -- wall-clock Unix timestamp (time.time()) for this record.  Pass 0.0
              to use the current time (default for live updates).
        """
        if self.graph_h <= 0:
            return
        sid = int(stage_id)
        if sid not in self._loss_graph_data:
            self._loss_graph_data[sid] = deque(maxlen=100_000)
            self._loss_graph_ts[sid]   = deque(maxlen=100_000)
        v = float(loss)
        self._loss_graph_data[sid].append(v if math.isfinite(v) else float("nan"))
        self._loss_graph_ts[sid].append(float(ts) if float(ts) > 0.0 else time.time())
        self._graph_dirty = True

    # ── Pull-model handlers (SaveRestoreNode IPC integration) ──────────

    def _on_sr_notification(self, msg: dict) -> None:
        """Handle a lightweight notification from the SaveRestoreNode.

        Notifications tell us *new data is available* without pushing the
        data itself.  We queue a query to pull the actual values.
        """
        t = msg.get("type", "")
        if t == "notify_new_loss":
            ck = msg.get("channel_key", "")
            remote_len = int(msg.get("length", 0))
            local_cursor = self._sr_loss_cursors.get(ck, 0)
            if remote_len > local_cursor:
                # New data available — we'll pull it on the next _sr_poll cycle.
                self._graph_dirty = True
        elif t == "notify_new_result":
            # Mark that a new result is available for the given channel.
            pass  # Next poll will pull it
        elif t == "notify_checkpoint":
            # Checkpoint marker — record it for the gold marker on the graph.
            self.notify_pipeline_checkpoint_saved()
            # Register for extended scrub navigation
            r = int(msg.get("round_id", 0))
            c = int(msg.get("cycle", 0))
            tp = msg.get("thumb_path")
            loss_counts = {sid: len(dq) for sid, dq in self._loss_graph_data.items()}
            self.register_checkpoint_thumbnail(r, c, thumb_path=tp, loss_counts=loss_counts)
        elif t == "notify_weight_map":
            # Weight map updated — request fresh image on next poll.
            self._sidebar_dirty = True

    def _on_sr_response(self, msg: dict) -> None:
        """Handle a response to one of our queries from the SaveRestoreNode."""
        t = msg.get("type", "")

        if t == "resp_channel_list":
            self._sr_channel_registry = msg.get("channels", [])

        elif t == "resp_loss_history":
            ck = msg.get("channel_key", "")
            records = msg.get("records", [])
            if records and self.graph_h > 0:
                # Merge into local loss graph data (stage-id indexed for
                # backward compat; channel key → integer index mapping.)
                sid = self._sr_channel_to_sid(ck)
                if sid not in self._loss_graph_data:
                    self._loss_graph_data[sid] = deque(maxlen=100_000)
                    self._loss_graph_ts[sid] = deque(maxlen=100_000)
                for rec in records:
                    v = float(rec.get("loss", 0.0))
                    self._loss_graph_data[sid].append(
                        v if math.isfinite(v) else float("nan")
                    )
                    self._loss_graph_ts[sid].append(float(rec.get("ts", 0.0)))
                # Update cursor so we only pull new records next time
                if records:
                    self._sr_loss_cursors[ck] = int(records[-1].get("step", 0)) + 1
                self._graph_dirty = True

        elif t == "resp_latest_result":
            ck = msg.get("channel_key", "")
            result = msg.get("result")
            if result is not None:
                # Turn the latest result into a displayable frame.
                images = result.get("images")
                if images is not None and isinstance(images, (list, np.ndarray)):
                    frame = {
                        "images": images if isinstance(images, list) else [images],
                        "caption": result.get("caption", ck),
                        "titles": result.get("titles", [ck]),
                        "rows": result.get("rows", [[]]),
                        "losses": {},
                    }
                    self.enqueue_frame(frame)

        elif t == "resp_weight_map":
            wmap = msg.get("map")
            if wmap is not None:
                # Store the weight map RGB for right-panel rendering
                try:
                    rgb_list = wmap.get("rgb")
                    if rgb_list is not None:
                        rgb_arr = np.array(rgb_list, dtype=np.uint8)
                        self._weight_map_rgb = rgb_arr
                        self._sidebar_dirty = True
                except Exception:
                    pass
            # Store model names for weight model selector
            self._sr_weight_models = msg.get("models", [])
            self._sr_weight_active = msg.get("active")

        elif t == "resp_cache_browse":
            self._sr_cache_entries = msg.get("entries", [])
            self._sr_cache_total = int(msg.get("total", 0))

    def _sr_channel_to_sid(self, channel_key: str) -> int:
        """Map a channel key string to a stable integer for graph data keying.

        Legacy stage IDs 0-5 are preserved for known stage names.
        New channels get incrementing IDs starting at 100.
        """
        _REV = {v: k for k, v in _LOSS_STAGE_NAMES.items()}
        base = channel_key.split("|")[0] if "|" in channel_key else channel_key
        if base in _REV:
            return _REV[base]
        # Assign a stable integer by hashing the channel key into the 100+ range
        if not hasattr(self, "_sr_sid_map"):
            self._sr_sid_map: Dict[str, int] = {}
            self._sr_sid_next: int = 100
        if channel_key not in self._sr_sid_map:
            self._sr_sid_map[channel_key] = self._sr_sid_next
            self._sr_sid_next += 1
        return self._sr_sid_map[channel_key]

    def _sr_poll_data(self) -> None:
        """Periodically request fresh data from the SaveRestoreNode.

        Called from the main pump() loop.  Sends queries over IPC via the
        IPC server's send_query method.  Does nothing if no IPC server is
        attached.
        """
        now = time.time()
        if now - self._sr_last_poll_t < self._sr_poll_interval:
            return
        self._sr_last_poll_t = now

        srv = self._ipc_server_ref
        if srv is None or not getattr(srv, "has_connection", False):
            return

        # Request channel list to discover new channels
        srv.send_query({"type": "query_channel_list"})

        # Pull loss history for each known channel, only new records
        for ch_info in self._sr_channel_registry:
            ck = ch_info.get("key", "")
            if ch_info.get("has_loss"):
                cursor = self._sr_loss_cursors.get(ck, 0)
                srv.send_query({
                    "type": "query_loss_history",
                    "channel_key": ck,
                    "from_step": cursor,
                })
            # Pull latest result for display
            if ch_info.get("has_result"):
                srv.send_query({
                    "type": "query_latest_result",
                    "channel_key": ck,
                })

        # Request weight map for right-panel display
        srv.send_query({"type": "query_weight_map"})

        # Request cache summary for browsing
        srv.send_query({"type": "query_cache_browse", "offset": 0, "limit": 50})

    def _render_loss_graph(self) -> np.ndarray:
        try:
            from PIL import Image, ImageDraw, ImageFont
        except Exception:
            _gw = int(self._col_x + self.num_panels * self.panel_w)
            return np.full((self.graph_h, _gw, 3), 14, dtype=np.uint8)

        W_full = int(self._col_x + self.num_panels * self.panel_w)  # only under left 4 cols
        H = int(self.graph_h)

        # Reserve the right portion for the pipeline graph canvas when a plan is loaded.
        pipeline_W = 0
        if isinstance(self._graph_plan_snapshot, dict) and self._graph_plan_snapshot.get("nodes"):
            pipeline_W = max(100, min(W_full // 3, 160))
        W = W_full - pipeline_W

        im = Image.new("RGB", (W_full, H), (14, 18, 22))
        loss_im = Image.new("RGB", (W, H), (14, 18, 22))
        draw = ImageDraw.Draw(loss_im)
        font = ImageFont.load_default()

        # Plot margins
        mx0, mx1 = 52, W - 6
        my0, my1 = 14, H - 6
        plot_w = max(1, mx1 - mx0)
        plot_h = max(1, my1 - my0)

        # Display trimming: _graph_display_start_frac defines the left edge of the
        # visible window as a fraction of the full history.  All x-axis mapping
        # below uses _view_frac() which maps a data-fraction into the visible window.
        _sf = max(0.0, min(0.99, float(self._graph_display_start_frac)))
        _view_span = max(1e-9, 1.0 - _sf)

        def _view_x(data_frac: float) -> int:
            """Map a data-space fraction [0,1] to an x pixel, clamped to plot area."""
            vf = (float(data_frac) - _sf) / _view_span
            return max(mx0, min(mx1, int(mx0 + vf * plot_w)))

        # Compute global Y range from visible finite values only
        all_finite: List[float] = []
        for sid, dq in self._loss_graph_data.items():
            if not self._loss_channel_visible.get(sid, True):
                continue
            n_dq = len(dq)
            i_start = int(_sf * n_dq)
            all_finite.extend(v for v in list(dq)[i_start:] if math.isfinite(v) and v >= 0.0)

        if len(all_finite) < 2:
            draw.text((mx0, my0 + plot_h // 2 - 4), "no data yet", fill=(80, 88, 100), font=font)
            im.paste(loss_im, (0, 0))
            if pipeline_W > 0:
                try:
                    canvas_arr = self._render_pipeline_graph_canvas(pipeline_W, H)
                    im.paste(Image.fromarray(canvas_arr), (W, 0))
                except Exception:
                    pass
            return np.asarray(im, dtype=np.uint8)

        raw_min = min(all_finite)
        raw_max = max(all_finite)
        span = raw_max - raw_min
        y_min = max(0.0, raw_min - span * 0.05)
        y_max = raw_max + span * 0.05
        if y_max <= y_min + 1e-9:
            y_max = y_min + 1.0

        def _y_px(v: float) -> int:
            frac = (float(v) - y_min) / (y_max - y_min)
            return int(my1 - frac * plot_h)

        # ── Cache-region and 2×-history overlay bands ─────────────────────────
        # Map the scrub snapshot window onto the loss graph x-axis using the
        # per-tick loss-count records stored alongside each weight snapshot.
        try:
            if self._loss_count_at_snap_deque:
                _scl = list(self._loss_count_at_snap_deque)
                oldest_counts = _scl[0]
                ref_sid = max(
                    self._loss_graph_data.keys(),
                    key=lambda s: len(self._loss_graph_data[s]),
                    default=None,
                )
                if ref_sid is not None and ref_sid in oldest_counts:
                    total_n   = len(self._loss_graph_data[ref_sid])
                    cache_frac = float(max(0, int(oldest_counts[ref_sid]))) / float(max(1, total_n))
                    cache_x0  = _view_x(cache_frac)
                    cache_w   = max(0, mx1 - cache_x0)
                    hist_x0   = max(mx0, cache_x0 - cache_w * 2)
                    _ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                    _od = ImageDraw.Draw(_ov)
                    # 2× history region – dim amber tint
                    if hist_x0 < cache_x0:
                        _od.rectangle(
                            [(hist_x0, my0), (cache_x0, my1)], fill=(80, 60, 0, 38)
                        )
                    # Cache region – dim teal tint
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

        # Plot each series (trimmed to the visible window)
        for sid in sorted(self._loss_graph_data.keys()):
            if not self._loss_channel_visible.get(sid, True):
                continue
            dq = self._loss_graph_data[sid]
            if len(dq) == 0:
                continue
            all_vals = list(dq)
            n_all = len(all_vals)
            # Trim to visible range  (W used for the loss portion only)
            i_start = int(_sf * n_all)
            vals = all_vals[i_start:]
            n = len(vals)
            if n == 0:
                continue
            _sorted_sids = sorted(self._loss_graph_data.keys())
            _sid_idx = _sorted_sids.index(sid) if sid in _sorted_sids else sid
            color = _LOSS_STAGE_COLORS_LEGACY.get(sid, _channel_color(_sid_idx, len(_sorted_sids)))

            # Downsample to plot_w buckets via mean to avoid overdraw
            if n > plot_w:
                bucket = n / float(plot_w)
                downsampled: List[float] = []
                for bi in range(plot_w):
                    i0 = int(bi * bucket)
                    i1 = max(i0 + 1, int((bi + 1) * bucket))
                    chunk = [vals[i] for i in range(i0, min(i1, n)) if math.isfinite(vals[i])]
                    downsampled.append(sum(chunk) / len(chunk) if chunk else float("nan"))
                vals = downsampled
                n = plot_w

            pts: List[Optional[Tuple[int, int]]] = []
            for i, v in enumerate(vals):
                if not math.isfinite(v):
                    pts.append(None)
                    continue
                xp = int(mx0 + (float(i) / float(max(1, n - 1))) * float(plot_w))
                yp = max(my0, min(my1, _y_px(v)))
                pts.append((xp, yp))

            prev = None
            for pt in pts:
                if pt is not None and prev is not None:
                    draw.line([prev, pt], fill=color, width=1)
                prev = pt if pt is not None else None

        # ── Disk-save checkpoint markers ──────────────────────────────────────
        try:
            if self._disk_save_loss_counts:
                _ref_sid_d = max(
                    self._loss_graph_data.keys(),
                    key=lambda s: len(self._loss_graph_data[s]),
                    default=None,
                )
                if _ref_sid_d is not None:
                    _total_nd = len(self._loss_graph_data[_ref_sid_d])
                    for _dsc in self._disk_save_loss_counts:
                        if _ref_sid_d in _dsc:
                            _cn = max(0, min(_total_nd, int(_dsc[_ref_sid_d])))
                            _frac = float(_cn) / float(max(1, _total_nd))
                            _dx = _view_x(_frac)
                            draw.line([(_dx, my0), (_dx, my1)], fill=(180, 150, 30), width=1)
                            draw.text((_dx + 1, my1 - 10), "\u25bc", fill=(200, 170, 40), font=font)
        except Exception:
            pass

        # ── Scrub cursor line ──────────────────────────────────────────────────
        try:
            if self._scrub_offset > 0 and self._loss_count_at_snap_deque:
                _sl   = list(self._loss_count_at_snap_deque)
                _slen = len(_sl)
                _off  = min(int(self._scrub_offset), _slen)
                _cidx = _slen - _off
                if 0 <= _cidx < _slen:
                    _cc = _sl[_cidx]
                    ref_sid = max(
                        self._loss_graph_data.keys(),
                        key=lambda s: len(self._loss_graph_data[s]),
                        default=None,
                    )
                    if ref_sid is not None and ref_sid in _cc:
                        total_n  = len(self._loss_graph_data[ref_sid])
                        cursor_n = max(0, min(total_n, int(_cc[ref_sid])))
                        frac_c   = float(cursor_n) / float(max(1, total_n))
                        cx       = _view_x(frac_c)
                        draw.line([(cx, my0), (cx, my1)], fill=(200, 150, 255), width=2)
                        draw.text(
                            (cx + 3, my0 + 2),
                            f"\u25c2{self._scrub_offset}",
                            fill=(200, 150, 255), font=font,
                        )
        except Exception:
            pass

        # Legend (horizontal, top of graph) — clickable channel toggles
        lx = mx0
        _legend_sids = sorted(self._loss_graph_data.keys())
        _legend_boxes: List[Tuple[int, int, int, int, int]] = []
        # Build reverse map: sid → channel_key for dynamic (100+) channels
        _sid_to_ck: Dict[int, str] = {}
        if hasattr(self, "_sr_sid_map"):
            _sid_to_ck = {v: k for k, v in self._sr_sid_map.items()}
        for _leg_idx, sid in enumerate(_legend_sids):
            if len(self._loss_graph_data[sid]) == 0:
                continue
            vis = self._loss_channel_visible.get(sid, True)
            color = _LOSS_STAGE_COLORS_LEGACY.get(sid, _channel_color(_leg_idx, len(_legend_sids)))
            name = _LOSS_STAGE_NAMES.get(sid, _sid_to_ck.get(sid, f"ch{sid}"))
            dq = self._loss_graph_data[sid]
            last_v = next((v for v in reversed(dq) if math.isfinite(v)), float("nan"))
            label = f"{name}={last_v:.4f}" if math.isfinite(last_v) else name
            # Dim hidden channels
            if vis:
                draw.rectangle([(lx, 2), (lx + 7, 9)], fill=color)
                draw.text((lx + 10, 1), label, fill=color, font=font)
            else:
                dim = tuple(max(30, c // 3) for c in color)
                draw.rectangle([(lx, 2), (lx + 7, 9)], fill=dim, outline=(60, 60, 60))
                draw.text((lx + 10, 1), label, fill=dim, font=font)
            entry_w = max(56, len(label) * 6 + 18)
            _legend_boxes.append((sid, lx, 0, lx + entry_w, 12))
            lx += entry_w
            if lx > W - 60:
                break
        self._legend_hit_boxes = _legend_boxes

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
        """Thread-safe: push a fully-rendered frame dict into the display buffer.
        Blocks when the buffer is full, back-pressuring the preview worker and
        (through the work queue) the training loop.  Never drops frames."""
        self._frame_buffer.put(frame_dict)

    def update(
        self,
        clean_img: torch.Tensor,
        input_img: torch.Tensor,
        output_img: torch.Tensor,
        caption: str,
        panel_titles: Optional[Sequence[str]] = None,
        panel_rows: Optional[Sequence[Sequence[str]]] = None,
        frame_losses: Optional[Dict[int, float]] = None,
    ):
        if not self.enabled:
            return
        self._init()
        if not self._ready:
            return
        self._poll_events()
        if self._stop_requested or (not self._ready):
            return

        # Store losses in the frame so the graph advances in sync with each popped image
        clean_rgb = self._resize_rgb_to_panel(_tensor_to_rgb_u8_image(clean_img))
        in_rgb = self._resize_rgb_to_panel(_tensor_to_rgb_u8_image(input_img))
        out_rgb = self._resize_rgb_to_panel(_tensor_to_rgb_u8_image(output_img))
        titles, rows = self._normalize_panel_text(panel_titles=panel_titles, panel_rows=panel_rows)
        self._frame_buffer.put({
            "images": [clean_rgb, in_rgb, out_rgb],
            "caption": str(caption),
            "titles": titles,
            "rows": rows,
            "losses": dict(frame_losses) if frame_losses is not None else {},
        })

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
# IPC layer — decoupled GUI ↔ training communication
# ---------------------------------------------------------------------------
# The viewer window runs in a separate process (launched by wav_ml_gui_main.py).
# The training pipeline talks to it over a localhost TCP connection managed by
# multiprocessing.connection (length-prefixed pickle).  Two classes:
#
#   ViewerIPCServer  — runs in the GUI process; receives data, dispatches to viewer
#   ViewerIPCProxy   — runs in the training process; same API as the viewer
#
# Message flow:
#   training → GUI :  loss, frame, checkpoint_saved, cycle_roster, weight_map, ...
#   GUI → training :  status (stop_requested, gate_override, cycle_selected)
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

        # Drain all available messages from training → GUI
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
            status = {
                "type": "status",
                "stop_requested": is_stopping,
                "gate_override": self._viewer.gate_override_enabled(),
                "cycle_selected": list(self._viewer._cycle_selected),
            }
            conn.send(status)
            run_control = RunControlPayload(
                command="stop" if bool(status["stop_requested"]) else "resume",
                selected_cycle_ids=self._viewer.selected_cycle_ids(),
                gate_override=bool(status["gate_override"]),
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
            v.update_loss(
                msg["stage_id"], msg["loss"], msg.get("aux", 0.0), msg.get("ts", 0.0)
            )
        elif t == "frame":
            frame = msg["frame"]
            # Resize images to panel size if needed
            for i, img in enumerate(frame.get("images", [])):
                if (
                    img is not None
                    and (img.shape[0] != v.panel_h or img.shape[1] != v.panel_w)
                ):
                    frame["images"][i] = v._resize_rgb_to_panel(img)
            v.enqueue_frame(frame)
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
            v._played_frames_deque.append(msg.get("played_frame"))
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

        The response will arrive asynchronously via poll() → _dispatch()
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

    def query_loss_history(self, channel_key: str, from_step: int = 0) -> None:
        self.send_query({
            "type": "query_loss_history",
            "channel_key": str(channel_key),
            "from_step": int(from_step),
        })

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
    """Runs in the training process — drop-in replacement for
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
        self._cycle_selected: List[bool] = [True] * max(0, cycle_slots)
        self._last_run_control = RunControlPayload(
            command="resume",
            selected_cycle_ids=[int(i + 1) for i in range(max(0, cycle_slots))],
            gate_override=False,
        )
        self._pending_plan_apply: Optional[PlanApplyPayload] = None
        self._conn: Optional[Any] = None
        self._send_lock = threading.Lock()  # protects concurrent writes
        self._connected_once = False
        self._connection_lost = False
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

        # Resolve port from file (waits up to `timeout` seconds)
        actual_port = int(port)
        if port_file:
            pf = Path(port_file)
            deadline = time.time() + timeout
            while time.time() < deadline:
                if pf.exists():
                    try:
                        actual_port = int(pf.read_text(encoding="utf-8").strip())
                        break
                    except (ValueError, OSError):
                        pass
                time.sleep(0.25)
            else:
                print(
                    f"[viewer-ipc] timeout waiting for port file: {port_file}",
                    flush=True,
                )
                self.enabled = False
                return

        if actual_port <= 0:
            print("[viewer-ipc] no valid port", flush=True)
            self.enabled = False
            return

        from multiprocessing.connection import Client

        try:
            self._conn = Client(
                ("localhost", actual_port), family="AF_INET", authkey=_IPC_AUTHKEY
            )
            self._connected_once = True
            print(
                f"[viewer-ipc] connected to GUI on port {actual_port}", flush=True
            )
        except Exception as e:
            print(f"[viewer-ipc] connection failed: {e}", flush=True)
            self.enabled = False

    # ── internal helpers ──────────────────────────────────────────────────

    def _send(self, msg: dict) -> None:
        if not self.enabled or self._conn is None:
            return
        with self._send_lock:
            try:
                self._conn.send(msg)
            except (EOFError, OSError):
                self._connection_lost = True
                self.enabled = False
            except Exception:
                pass

    def _drain_status(self) -> None:
        """Non-blocking read of all available status messages from the GUI."""
        if not self.enabled or self._conn is None:
            return
        try:
            while self._conn.poll(0):
                msg = self._conn.recv()
                if is_protocol_envelope_message(msg):
                    envelope, payload = parse_envelope(msg)
                    if envelope.message_type == MESSAGE_TYPE_RUN_CONTROL:
                        self._last_run_control = payload
                        self._stop_flag = str(payload.command).lower() == "stop"
                        self._gate_override = bool(payload.gate_override)
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
                    cs = msg.get("cycle_selected")
                    if cs is not None:
                        self._cycle_selected = list(cs)
                    self._last_run_control = RunControlPayload(
                        command="stop" if bool(self._stop_flag) else "resume",
                        selected_cycle_ids=self.selected_cycle_ids(),
                        gate_override=bool(self._gate_override),
                        metadata={"source": "legacy_status"},
                    )
                elif t == "restore":
                    if self._on_restore is not None:
                        try:
                            self._on_restore(int(msg.get("offset", 0)))
                        except Exception:
                            pass
                elif t is not None and t.startswith("query_"):
                    # Pull-model: route query to SaveRestoreNode, send response
                    self._handle_gui_query(msg)
        except (EOFError, OSError):
            self._connection_lost = True
            self.enabled = False
        except Exception:
            pass

    # ── public API (mirrors _TransformerStatusOpenGLViewer) ───────────────

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
        if self._connection_lost:
            return True  # GUI closed or crashed
        return self._stop_flag

    def shutdown_save(self) -> Optional[bool]:
        """Return the save preference for the current shutdown, or None if no shutdown."""
        return self._shutdown_save

    def update_loss(self, stage_id: int, loss: float, aux: float = 0.0, ts: float = 0.0):
        self._send({
            "type": "loss",
            "stage_id": int(stage_id),
            "loss": float(loss),
            "aux": float(aux),
            "ts": float(ts),
        })

    def enqueue_frame(self, frame_dict: dict):
        self._send({"type": "frame", "frame": frame_dict})

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
        clean_rgb = _tensor_to_rgb_u8_image(clean_img)
        in_rgb = _tensor_to_rgb_u8_image(input_img)
        out_rgb = _tensor_to_rgb_u8_image(output_img)
        frame = {
            "images": [clean_rgb, in_rgb, out_rgb],
            "caption": str(caption),
            "titles": list(panel_titles) if panel_titles else ["target", "input", "output"],
            "rows": [list(r) for r in panel_rows] if panel_rows else [[], [], []],
            "losses": dict(frame_losses) if frame_losses else {},
        }
        self._send({"type": "frame", "frame": frame})

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
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
