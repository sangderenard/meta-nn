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

_LOSS_STAGE_COLORS: Dict[int, tuple] = {
    LOSS_STAGE_CLASSIFIER:          (80,  200, 220),  # cyan
    LOSS_STAGE_GENERATOR:           (80,  200, 100),  # green
    LOSS_STAGE_DISCRIMINATOR:       (240, 140,  40),  # orange
    LOSS_STAGE_TRANSFORMER:         (240, 220,  60),  # yellow
    LOSS_STAGE_WAVE_CLASSIFIER:     (220,  80, 220),  # magenta
    LOSS_STAGE_WAVE_CLASSIFIER_EVAL:(160, 120, 255),  # lavender
}

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
        self._scrub_offset: int = 0
        # Pipeline registers fn(scrub_offset) to handle "RESTORE STATE".
        self._on_restore_state: Optional[Callable] = None
        # Bounding box (x0,y0,x1,y1) in window coords for the restore button.
        self._restore_btn_window_rect: Optional[Tuple[int, int, int, int]] = None
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
            draw.text((4, 2), str(title)[:48], fill=(255, 225, 70), font=font)
            draw.line([(0, 16), (self.panel_w - 1, 16)], fill=(70, 76, 88), width=1)
            y = 20
            for r in list(rows)[: max(1, (self.panel_h - 20) // 11)]:
                draw.text((4, y), str(r)[:72], fill=(230, 234, 240), font=font)
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

    # ── Sidebar public API ─────────────────────────────────────────────────────

    def set_queue_refs(self, work_queue: Any) -> None:
        """Register the preview work queue so the cache map can read its depth."""
        self._preview_work_queue_ref = work_queue

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
            total = len(self._weight_snapshot_deque)
            if scrubbing:
                draw.text((4, y_info),      f"scrub: -{self._scrub_offset} / {total}",
                          fill=(180, 160, 220), font=font)
                draw.text((4, y_info + 12), "wheel \u2191\u2193 to navigate",
                          fill=(110, 100, 140), font=font)
                draw.text((4, y_info + 24), "click btn to restore & run",
                          fill=(110, 100, 140), font=font)
            else:
                draw.text((4, y_info),      f"live  (history: {total})",
                          fill=(80, 100, 80), font=font)
                draw.text((4, y_info + 12), "wheel \u2191\u2193 to scrub back",
                          fill=(60, 70, 60), font=font)
            out = np.asarray(im, dtype=np.uint8)
        except Exception:
            pass
        return out

    def _render_scrub_dial(self) -> np.ndarray:
        """Left sidebar bottom panel: circular dial showing scrub offset."""
        total = max(1, len(self._weight_snapshot_deque))
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
            self._last_anim_t += self._anim_frame_dt
        if last_frame is not None:
            imgs = last_frame.get("images", None)
            if isinstance(imgs, list) and len(imgs) == 3:
                for i, tid in enumerate(self._textures["img"]):
                    self._upload_texture(int(tid), np.asarray(imgs[i], dtype=np.uint8))
            self._caption = str(last_frame.get("caption", ""))
            self._panel_titles = [str(x) for x in last_frame.get("titles", self._panel_titles)]
            self._panel_rows = [list(r) for r in last_frame.get("rows", self._panel_rows)]
            self._panel_text_dirty = True
            self._top_bar_dirty = True
            self._last_displayed_frame = last_frame

        # Sidebar: btn_panel + scrub_dial every pump (cheap); weight snap is count-driven.
        self._cache_map_rgb  = self._render_btn_panel()
        self._frame_knob_rgb = self._render_scrub_dial()
        if self._scrub_offset == 0:
            _snap_imgs = (
                self._last_displayed_frame.get("images", None)
                if self._last_displayed_frame is not None else None
            )
            # Record loss series lengths so the graph can map this snap → x-pixel.
            self._loss_count_at_snap_deque.append(
                {sid: len(dq) for sid, dq in self._loss_graph_data.items()}
            )
            self._played_frames_deque.append({"images": _snap_imgs})
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
                    self._weight_snapshot_deque.append({"weight_map": _wmap.copy()})
                _sst = _res.get("states")
                if _sst:
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
                    if _sparse:
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
            _snaps  = list(self._weight_snapshot_deque)
            _slen   = len(_snaps)
            if _slen > 0:
                _off = min(int(self._scrub_offset), _slen)  # clamp: allow reaching index 0
                _idx = _slen - _off                          # 0 = oldest, _slen-1 = newest
                _idx = max(0, min(_slen - 1, _idx))         # hard safety clamp
                self._weight_map_rgb = np.asarray(_snaps[_idx]["weight_map"], dtype=np.uint8)
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
                    _snap_len = len(self._weight_snapshot_deque)
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
                                self._scrub_offset = 0
                                self._weight_snapshot_deque.clear()
                                self._played_frames_deque.clear()
                                self._sidebar_dirty = True
                                self._present(force=True)
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
        for dq in self._loss_graph_data.values():
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
            color = _LOSS_STAGE_COLORS.get(sid, (200, 200, 200))

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

        # Legend (horizontal, top of graph)
        lx = mx0
        for sid in sorted(_LOSS_STAGE_COLORS.keys()):
            if sid not in self._loss_graph_data or len(self._loss_graph_data[sid]) == 0:
                continue
            color = _LOSS_STAGE_COLORS[sid]
            name = _LOSS_STAGE_NAMES.get(sid, f"s{sid}")
            dq = self._loss_graph_data[sid]
            last_v = next((v for v in reversed(dq) if math.isfinite(v)), float("nan"))
            label = f"{name}={last_v:.4f}" if math.isfinite(last_v) else name
            draw.rectangle([(lx, 2), (lx + 7, 9)], fill=color)
            draw.text((lx + 10, 1), label, fill=color, font=font)
            lx += max(56, len(label) * 6 + 18)
            if lx > W - 60:
                break

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
                "ran":       (52,  168,  83),
                "completed": (52,  168,  83),
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
            status = {
                "type": "status",
                "stop_requested": self._viewer.stop_requested(),
                "gate_override": self._viewer.gate_override_enabled(),
                "cycle_selected": list(self._viewer._cycle_selected),
            }
            conn.send(status)
            run_control = RunControlPayload(
                command="stop" if bool(status["stop_requested"]) else "resume",
                selected_cycle_ids=self._viewer.selected_cycle_ids(),
                gate_override=bool(status["gate_override"]),
                metadata={"source": "viewer_ipc_status_loop"},
            )
            conn.send(make_envelope(MESSAGE_TYPE_RUN_CONTROL, run_control).to_dict())
        except (EOFError, OSError):
            with self._lock:
                self._conn = None
        except Exception:
            pass

    def _dispatch(self, msg: dict) -> None:
        if is_protocol_envelope_message(msg):
            envelope, payload = parse_envelope(msg)
            if envelope.message_type == MESSAGE_TYPE_WORKER_HELLO:
                if hasattr(self._viewer, "set_training_graph_worker_hello"):
                    self._viewer.set_training_graph_worker_hello(payload.to_dict())
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
        except (EOFError, OSError):
            self._connection_lost = True
            self.enabled = False
        except Exception:
            pass

    # ── public API (mirrors _TransformerStatusOpenGLViewer) ───────────────

    def pump(self):
        self._drain_status()

    def stop_requested(self) -> bool:
        self._drain_status()
        if self._connection_lost:
            return True  # GUI closed or crashed
        return self._stop_flag

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
