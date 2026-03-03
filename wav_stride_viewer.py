# wav_stride_viewer.py
# Simple "strided audio -> RGB image" WAV viewer for spotting file-like artifacts / errors.
#
# pip install numpy pillow
# python wav_stride_viewer.py

import math
import wave
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk


# ----------------------------
# WAV decoding helpers
# ----------------------------

def _sign_extend_24(u24: np.ndarray) -> np.ndarray:
    # u24 is int32 containing 24-bit values in [0, 2^24).
    # Sign-extend to int32 using xor/sub trick.
    sign_bit = 1 << 23
    return (u24 ^ sign_bit) - sign_bit


def decode_pcm(frames: bytes, nch_file: int, user_nch: int, mode: str):
    """
    Decode raw frame bytes into:
      - (N, C) float32 in [-1, 1] approximately
      - optional (N, C) int32 raw-ish sample values (for bit-window extraction)
      - optional sample bit depth for integer decode paths
    mode: one of {"auto", "8", "16", "24", "32", "float32"}.
    user_nch: interpret as 1 or 2 channels (for override experiments).
    nch_file: channels reported by the WAV header (used for auto path sanity).
    """
    if len(frames) == 0:
        return np.zeros((0, user_nch), dtype=np.float32), None, None, "No audio frames."

    # Determine bytes per sample for the chosen mode
    if mode == "8":
        bps = 1
        kind = "int"
        bits = 8
    elif mode == "16":
        bps = 2
        kind = "int"
        bits = 16
    elif mode == "24":
        bps = 3
        kind = "int"
        bits = 24
    elif mode == "32":
        bps = 4
        kind = "int"
        bits = 32
    elif mode == "float32":
        bps = 4
        kind = "float"
        bits = 32
    else:
        # "auto" is handled by caller using actual WAV sampwidth + a default interpretation
        raise ValueError("decode_pcm called with mode='auto' (should be resolved by caller)")

    stride = user_nch * bps
    n = len(frames) // stride
    trimmed = frames[: n * stride]

    if n == 0:
        return np.zeros((0, user_nch), dtype=np.float32), None, None, "Not enough bytes for chosen channel/bit settings."

    if kind == "float":
        data = np.frombuffer(trimmed, dtype="<f4").astype(np.float32, copy=False)
        data = data.reshape(-1, user_nch)
        # Clamp for display safety
        data = np.clip(data, -1.0, 1.0)
        return data, None, None, f"Decoded as float32, {user_nch}ch."
    else:
        if bits == 8:
            u8 = np.frombuffer(trimmed, dtype=np.uint8).astype(np.int16)
            # WAV 8-bit PCM is typically unsigned
            i = (u8 - 128).astype(np.int16)
            i = i.reshape(-1, user_nch)
            data = i.astype(np.float32) / 128.0
            return data, i.astype(np.int32), bits, f"Decoded as int8 (unsigned PCM assumed), {user_nch}ch."
        elif bits == 16:
            i = np.frombuffer(trimmed, dtype="<i2").reshape(-1, user_nch)
            data = i.astype(np.float32) / (2 ** 15)
            return data, i.astype(np.int32), bits, f"Decoded as int16, {user_nch}ch."
        elif bits == 24:
            b = np.frombuffer(trimmed, dtype=np.uint8).astype(np.int32)
            # Interpreting as little-endian 24-bit samples:
            # sample0: b0 + (b1<<8) + (b2<<16), etc.
            u24 = (b[0::3] | (b[1::3] << 8) | (b[2::3] << 16)).astype(np.int32)
            i24 = _sign_extend_24(u24)
            i24 = i24.reshape(-1, user_nch)
            data = i24.astype(np.float32) / (2 ** 23)
            return data, i24, bits, f"Decoded as int24, {user_nch}ch."
        elif bits == 32:
            i = np.frombuffer(trimmed, dtype="<i4").reshape(-1, user_nch)
            data = i.astype(np.float32) / (2 ** 31)
            return data, i, bits, f"Decoded as int32, {user_nch}ch."
        else:
            raise ValueError("Unsupported int bit depth.")


def pcm_to_u8(x: np.ndarray) -> np.ndarray:
    """x float32 in ~[-1,1] -> uint8 [0,255]."""
    x = np.clip(x, -1.0, 1.0)
    y = (x * 0.5 + 0.5) * 255.0
    return y.astype(np.uint8)


def _interleave_lr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.empty((a.size + b.size,), dtype=a.dtype)
    out[0::2] = a
    out[1::2] = b
    return out


def _pick_mono_stream(samples_f32: np.ndarray, samples_i32, mono_pick: str):
    c = samples_f32.shape[1]
    if c == 1:
        mono_f = samples_f32[:, 0]
        mono_i = samples_i32[:, 0] if samples_i32 is not None else None
        return mono_f, mono_i

    if mono_pick == "L":
        return samples_f32[:, 0], (samples_i32[:, 0] if samples_i32 is not None else None)
    if mono_pick == "R":
        return samples_f32[:, 1], (samples_i32[:, 1] if samples_i32 is not None else None)
    if mono_pick == "LR stride":
        mono_f = _interleave_lr(samples_f32[:, 0], samples_f32[:, 1])
        if samples_i32 is None:
            return mono_f, None
        mono_i = _interleave_lr(samples_i32[:, 0], samples_i32[:, 1])
        return mono_f, mono_i

    # Default: average L/R
    mono_f = 0.5 * (samples_f32[:, 0] + samples_f32[:, 1])
    if samples_i32 is None:
        return mono_f, None
    mono_i = ((samples_i32[:, 0].astype(np.int64) + samples_i32[:, 1].astype(np.int64)) // 2).astype(np.int32)
    return mono_f, mono_i


def _normalize_bit_window(bit_low: int, bit_high: int, sample_bits: int):
    max_bit = max(0, int(sample_bits) - 1)
    lo = max(0, min(int(bit_low), max_bit))
    hi = max(0, min(int(bit_high), max_bit))
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi, (hi - lo + 1)


def bit_window_to_u8(samples_i32: np.ndarray, sample_bits: int, bit_low: int, bit_high: int) -> np.ndarray:
    used_bits = int(bit_high) - int(bit_low) + 1
    full_mask = (1 << int(sample_bits)) - 1
    window_mask = (1 << used_bits) - 1

    unsigned = samples_i32.astype(np.int64) & full_mask
    window = (unsigned >> int(bit_low)) & window_mask
    if window_mask == 1:
        return (window.astype(np.uint8) * 255)

    scaled = np.round(window.astype(np.float32) * (255.0 / float(window_mask)))
    return scaled.astype(np.uint8)


def _score_contrast_geometry(stream_u8: np.ndarray, width: int):
    if width < 8:
        return float("inf"), 1.0, 1.0, 0

    usable = (stream_u8.size // width) * width
    if usable < width * 8:
        return float("inf"), 1.0, 1.0, 0

    arr = stream_u8[:usable].reshape(-1, width).astype(np.float32)
    h, w = arr.shape
    if h < 8 or w < 8:
        return float("inf"), 1.0, 1.0, h

    gx = np.abs(np.diff(arr, axis=1))
    gy = np.abs(np.diff(arr, axis=0))
    gx_c = gx[:-1, :]
    gy_c = gy[:, :-1]
    if gx_c.size == 0 or gy_c.size == 0:
        return float("inf"), 1.0, 1.0, h

    mag = gx_c + gy_c
    thresh = np.percentile(mag, 75.0)
    strong = mag >= thresh
    if np.any(strong):
        ax = gx_c[strong]
        ay = gy_c[strong]
    else:
        ax = gx_c.reshape(-1)
        ay = gy_c.reshape(-1)

    anis = np.minimum(ax, ay) / (np.maximum(ax, ay) + 1e-6)
    right_angle_dev = float(np.mean(anis))

    row_contrast = np.mean(gx, axis=1)
    if row_contrast.size >= 3:
        x = np.linspace(-1.0, 1.0, row_contrast.size, dtype=np.float32)
        coeff = np.polyfit(x, row_contrast, 2)
        fit = np.polyval(coeff, x)
        poly_dev = float(np.mean(np.abs(row_contrast - fit)) / (np.mean(np.abs(row_contrast)) + 1e-6))
    else:
        poly_dev = 1.0

    score = (0.7 * right_angle_dev) + (0.3 * poly_dev)
    return score, right_angle_dev, poly_dev, h


def autotune_width_from_stream(stream_u8: np.ndarray, requested_width: int, step: int, radius: int,
                               min_width: int = 16, max_width: int = 262144):
    if stream_u8.size < min_width * 8:
        return max(min_width, int(requested_width)), "Autotune skipped: not enough data."

    base = max(min_width, min(int(requested_width), max_width))
    step = max(1, int(step))
    radius = max(0, int(radius))

    candidates = []
    for k in range(-radius, radius + 1):
        w = base + (k * step)
        if min_width <= w <= max_width:
            candidates.append(w)
    if base not in candidates:
        candidates.append(base)
    candidates = sorted(set(candidates))

    best = None
    for w in candidates:
        score, right_angle_dev, poly_dev, h = _score_contrast_geometry(stream_u8, w)
        if not np.isfinite(score):
            continue
        if best is None or score < best[0]:
            best = (score, w, right_angle_dev, poly_dev, h)

    if best is None:
        return base, "Autotune skipped: no valid width candidate."

    score, w, right_angle_dev, poly_dev, h = best
    note = (
        f"Autotune width={w} (score={score:.4f}, right-angle={right_angle_dev:.4f}, "
        f"poly={poly_dev:.4f}, h={h})."
    )
    return w, note


# ----------------------------
# Image construction
# ----------------------------

COLOR_MODES = [
    # (label, required_audio_channels, mapping dict: {'R':src, 'G':src, 'B':src} where src in {None,'M','L','R'})
    ("Mono → Grayscale (R=G=B)", 1, {"R": "M", "G": "M", "B": "M"}),
    ("Mono → R only (G,B empty)", 1, {"R": "M", "G": None, "B": None}),
    ("Mono → G only (R,B empty)", 1, {"R": None, "G": "M", "B": None}),
    ("Mono → B only (R,G empty)", 1, {"R": None, "G": None, "B": "M"}),

    ("Stereo → R=L, G=R (B empty)", 2, {"R": "L", "G": "R", "B": None}),
    ("Stereo → R=L, B=R (G empty)", 2, {"R": "L", "G": None, "B": "R"}),
    ("Stereo → G=L, B=R (R empty)", 2, {"R": None, "G": "L", "B": "R"}),
    ("Stereo → R=R, G=L (B empty)", 2, {"R": "R", "G": "L", "B": None}),
    ("Single source → RGB via sample stride", 1, {"__special__": "single_source_rgb_stride"}),
]


def build_image(samples_f32: np.ndarray, width: int, downsample: int, color_mode_label: str, empty_fill: int,
                mono_pick: str, samples_i32=None, sample_bits: int = None,
                bitmask_enable: bool = False, bitmask_low: int = 0, bitmask_high: int = 7):
    """
    samples_f32: (N, C) float32
    width: pixels per row (stride)
    downsample: take every Nth sample
    color_mode_label: selected mode from COLOR_MODES
    empty_fill: 0..255 fill value for unused channels
    mono_pick: "L", "R", or "Mix" when user wants mono from stereo
    """
    if width < 1:
        width = 1
    if downsample < 1:
        downsample = 1

    N, C = samples_f32.shape
    if N == 0:
        img = np.full((1, 1, 3), empty_fill, dtype=np.uint8)
        return Image.fromarray(img, mode="RGB"), "No samples."

    # Downsample by simple decimation (good for visual pattern hunting)
    s_f = samples_f32[::downsample, :]
    s_i = samples_i32[::downsample, :] if samples_i32 is not None else None

    # Resolve mode
    mode = next((m for m in COLOR_MODES if m[0] == color_mode_label), None)
    if mode is None:
        mode = COLOR_MODES[0]
    _, need_ch, mapping = mode
    special = mapping.get("__special__")

    # Source signals (float + optional int mirrors for bit-window view)
    sources_f = {}
    sources_i = {}

    if special == "single_source_rgb_stride":
        src_f, src_i = _pick_mono_stream(s_f, s_i, mono_pick)
        sources_f["SRC"] = src_f
        if src_i is not None:
            sources_i["SRC"] = src_i
    else:
        if need_ch == 1:
            mono_f, mono_i = _pick_mono_stream(s_f, s_i, mono_pick)
            sources_f["M"] = mono_f
            if mono_i is not None:
                sources_i["M"] = mono_i
        else:
            # need stereo sources
            if C == 1:
                # Duplicate mono into L, R (still useful)
                Lf = s_f[:, 0]
                Rf = s_f[:, 0]
                Li = s_i[:, 0] if s_i is not None else None
                Ri = s_i[:, 0] if s_i is not None else None
            else:
                Lf = s_f[:, 0]
                Rf = s_f[:, 1]
                Li = s_i[:, 0] if s_i is not None else None
                Ri = s_i[:, 1] if s_i is not None else None

            sources_f["L"] = Lf
            sources_f["R"] = Rf
            if Li is not None and Ri is not None:
                sources_i["L"] = Li
                sources_i["R"] = Ri

    # Convert sources to uint8 streams
    can_use_bitmask = bool(bitmask_enable) and (sample_bits is not None) and (len(sources_i) > 0)
    if can_use_bitmask:
        mask_low_eff, mask_high_eff, _ = _normalize_bit_window(bitmask_low, bitmask_high, sample_bits)
    else:
        mask_low_eff, mask_high_eff = None, None

    sources_u8 = {}
    for k, vf in sources_f.items():
        vi = sources_i.get(k)
        if can_use_bitmask and vi is not None:
            sources_u8[k] = bit_window_to_u8(
                vi, sample_bits=sample_bits, bit_low=mask_low_eff, bit_high=mask_high_eff
            )
        else:
            sources_u8[k] = pcm_to_u8(vf.astype(np.float32, copy=False))

    # Map channel streams
    if special == "single_source_rgb_stride":
        base = sources_u8["SRC"]
        channel_streams = {
            "R": base[0::3],
            "G": base[1::3],
            "B": base[2::3],
        }
    else:
        channel_streams = {
            "R": (sources_u8[mapping.get("R")] if mapping.get("R") is not None else None),
            "G": (sources_u8[mapping.get("G")] if mapping.get("G") is not None else None),
            "B": (sources_u8[mapping.get("B")] if mapping.get("B") is not None else None),
        }

    # Pad to rectangle
    present_streams = [v for v in channel_streams.values() if v is not None]
    if len(present_streams) == 0:
        total = width
    else:
        max_len = max(int(v.size) for v in present_streams)
        total = int(math.ceil(max_len / width) * width) if max_len > 0 else width
    height = max(1, total // width)

    canvas = np.full((height, width, 3), int(empty_fill) & 0xFF, dtype=np.uint8)

    def place(channel_char: str):
        stream = channel_streams.get(channel_char)
        if stream is None:
            return
        if stream.size < total:
            pad = np.full((total - stream.size,), int(empty_fill) & 0xFF, dtype=np.uint8)
            stream = np.concatenate([stream, pad], axis=0)
        stream2d = stream[:total].reshape(height, width)
        idx = {"R": 0, "G": 1, "B": 2}[channel_char]
        canvas[:, :, idx] = stream2d

    place("R")
    place("G")
    place("B")

    if bitmask_enable:
        if can_use_bitmask:
            depth = (mask_high_eff - mask_low_eff + 1)
            bits_note = f"Bitmask view: bits {mask_low_eff}..{mask_high_eff} (depth={depth})."
        else:
            bits_note = "Bitmask requested, but decode mode has no integer bitplane (showing amplitude)."
    else:
        bits_note = "Amplitude view."

    extra_note = " RGB stride mode." if special == "single_source_rgb_stride" else ""

    return Image.fromarray(canvas, mode="RGB"), (
        f"Image: {width}px wide × {height}px tall (downsample={downsample}). "
        f"{bits_note}{extra_note}"
    )


# ----------------------------
# GUI
# ----------------------------

class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("WAV Stride Viewer (Audio → RGB)")

        self.wav_path = None
        self.wav_params = None
        self.frames = b""

        # Variables
        self.var_bitmode = tk.StringVar(value="auto")
        self.var_use_channels = tk.StringVar(value="auto")  # "auto", "1", "2"
        self.var_mono_pick = tk.StringVar(value="Mix")      # L/R/Mix
        self.var_width = tk.IntVar(value=2048)
        self.var_downsample = tk.IntVar(value=1)
        self.var_color = tk.StringVar(value=COLOR_MODES[4][0])  # default stereo mapping
        self.var_empty = tk.IntVar(value=0)
        self.var_bitmask_enable = tk.BooleanVar(value=False)
        self.var_mask_low = tk.IntVar(value=0)
        self.var_mask_high = tk.IntVar(value=7)
        self.var_autotune = tk.BooleanVar(value=False)
        self.var_autotune_span = tk.IntVar(value=6)
        self.var_autotune_step = tk.IntVar(value=16)
        self.var_fit = tk.BooleanVar(value=True)

        self.status = tk.StringVar(value="Load a WAV to begin.")

        self._render_after_id = None
        self._img_full = None
        self._img_tk = None

        self._build_ui()
        self._wire_events()

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=8)
        top.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)

        # File row
        file_row = ttk.Frame(top)
        file_row.grid(row=0, column=0, sticky="ew")
        file_row.columnconfigure(1, weight=1)

        ttk.Button(file_row, text="Load WAV…", command=self.load_wav).grid(row=0, column=0, padx=(0, 8))
        self.lbl_file = ttk.Label(file_row, text="(no file loaded)")
        self.lbl_file.grid(row=0, column=1, sticky="ew")

        # Controls
        controls = ttk.LabelFrame(top, text="Controls", padding=8)
        controls.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        for c in range(8):
            controls.columnconfigure(c, weight=1)

        r = 0
        ttk.Label(controls, text="Interpret bit width:").grid(row=r, column=0, sticky="w")
        ttk.Combobox(controls, textvariable=self.var_bitmode, state="readonly",
                     values=["auto", "8", "16", "24", "32", "float32"], width=10).grid(row=r, column=1, sticky="w")

        ttk.Label(controls, text="Use channels:").grid(row=r, column=2, sticky="w", padx=(12, 0))
        ttk.Combobox(controls, textvariable=self.var_use_channels, state="readonly",
                     values=["auto", "1", "2"], width=8).grid(row=r, column=3, sticky="w")

        ttk.Label(controls, text="Mono pick:").grid(row=r, column=4, sticky="w", padx=(12, 0))
        ttk.Combobox(controls, textvariable=self.var_mono_pick, state="readonly",
                     values=["Mix", "L", "R", "LR stride"], width=10).grid(row=r, column=5, sticky="w")

        ttk.Label(controls, text="Empty fill (0-255):").grid(row=r, column=6, sticky="w", padx=(12, 0))
        ttk.Spinbox(controls, from_=0, to=255, textvariable=self.var_empty, width=6).grid(row=r, column=7, sticky="w")

        r = 1
        ttk.Label(controls, text="Horizontal span (px):").grid(row=r, column=0, sticky="w", pady=(8, 0))
        ttk.Spinbox(controls, from_=16, to=262144, increment=16,
                    textvariable=self.var_width, width=10).grid(row=r, column=1, sticky="w", pady=(8, 0))

        ttk.Label(controls, text="Downsample (every Nth sample):").grid(row=r, column=2, sticky="w", padx=(12, 0), pady=(8, 0))
        ttk.Spinbox(controls, from_=1, to=1024, textvariable=self.var_downsample, width=8).grid(row=r, column=3, sticky="w", pady=(8, 0))

        ttk.Label(controls, text="Color mapping:").grid(row=r, column=4, sticky="w", padx=(12, 0), pady=(8, 0))
        ttk.Combobox(controls, textvariable=self.var_color, state="readonly",
                     values=[m[0] for m in COLOR_MODES], width=34).grid(row=r, column=5, columnspan=3, sticky="ew", pady=(8, 0))

        r = 2
        ttk.Checkbutton(controls, text="Fit to window", variable=self.var_fit).grid(row=r, column=0, sticky="w", pady=(8, 0))
        ttk.Checkbutton(controls, text="Use bitmask", variable=self.var_bitmask_enable).grid(
            row=r, column=1, sticky="w", padx=(12, 0), pady=(8, 0)
        )
        ttk.Label(controls, text="low bit:").grid(row=r, column=2, sticky="w", padx=(12, 0), pady=(8, 0))
        ttk.Spinbox(controls, from_=0, to=31, textvariable=self.var_mask_low, width=6).grid(
            row=r, column=3, sticky="w", pady=(8, 0)
        )
        ttk.Label(controls, text="high bit:").grid(row=r, column=4, sticky="w", padx=(12, 0), pady=(8, 0))
        ttk.Spinbox(controls, from_=0, to=31, textvariable=self.var_mask_high, width=6).grid(
            row=r, column=5, sticky="w", pady=(8, 0)
        )

        r = 3
        ttk.Checkbutton(controls, text="Autotune width", variable=self.var_autotune).grid(row=r, column=0, sticky="w", pady=(8, 0))
        ttk.Label(controls, text="± steps:").grid(row=r, column=1, sticky="w", padx=(12, 0), pady=(8, 0))
        ttk.Spinbox(controls, from_=0, to=64, textvariable=self.var_autotune_span, width=6).grid(row=r, column=2, sticky="w", pady=(8, 0))
        ttk.Label(controls, text="step px:").grid(row=r, column=3, sticky="w", padx=(12, 0), pady=(8, 0))
        ttk.Spinbox(controls, from_=1, to=4096, textvariable=self.var_autotune_step, width=8).grid(row=r, column=4, sticky="w", pady=(8, 0))

        # Viewer
        viewer = ttk.Frame(self.root, padding=8)
        viewer.grid(row=1, column=0, sticky="nsew")
        viewer.rowconfigure(0, weight=1)
        viewer.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(viewer, bg="#111111", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        yscroll = ttk.Scrollbar(viewer, orient="vertical", command=self.canvas.yview)
        xscroll = ttk.Scrollbar(viewer, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")

        # Status bar
        statusbar = ttk.Label(self.root, textvariable=self.status, anchor="w", padding=6)
        statusbar.grid(row=2, column=0, sticky="ew")

    def _wire_events(self):
        # Re-render on any control change
        for v in [
            self.var_bitmode, self.var_use_channels, self.var_mono_pick,
            self.var_width, self.var_downsample, self.var_color, self.var_empty,
            self.var_bitmask_enable, self.var_mask_low, self.var_mask_high,
            self.var_autotune, self.var_autotune_span,
            self.var_autotune_step, self.var_fit
        ]:
            v.trace_add("write", lambda *_: self.schedule_render())

        # Re-render on canvas resize if fitting
        self.canvas.bind("<Configure>", lambda e: self.schedule_render())

    def load_wav(self):
        path = filedialog.askopenfilename(
            title="Select WAV file",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            with wave.open(path, "rb") as wf:
                nch = wf.getnchannels()
                sw = wf.getsampwidth()
                fr = wf.getframerate()
                nf = wf.getnframes()
                frames = wf.readframes(nf)

            self.wav_path = path
            self.wav_params = (nch, sw, fr, nf)
            self.frames = frames
            self.lbl_file.config(text=path)
            self.status.set(f"Loaded: {nch}ch, {sw * 8}-bit, {fr} Hz, {nf} frames.")
            self.schedule_render()

        except wave.Error as e:
            messagebox.showerror("WAV read error", f"Could not read as PCM WAV:\n\n{e}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load WAV:\n\n{e}")

    def schedule_render(self):
        # Debounce re-renders when typing/spinning
        if self._render_after_id is not None:
            try:
                self.root.after_cancel(self._render_after_id)
            except Exception:
                pass
        self._render_after_id = self.root.after(120, self.render)

    def render(self):
        self._render_after_id = None

        if not self.frames or not self.wav_params:
            return

        nch_file, sampwidth, fr, nf = self.wav_params

        # Resolve "use channels"
        use_ch = self.var_use_channels.get()
        if use_ch == "auto":
            user_nch = nch_file if nch_file in (1, 2) else 2
        else:
            user_nch = int(use_ch)

        # Resolve bit mode
        bitmode = self.var_bitmode.get()
        actual_bits = sampwidth * 8
        # If auto: map sampwidth to mode (int unless 4 bytes where float is possible)
        if bitmode == "auto":
            if sampwidth == 1:
                mode = "8"
            elif sampwidth == 2:
                mode = "16"
            elif sampwidth == 3:
                mode = "24"
            elif sampwidth == 4:
                mode = "32"  # default; you can switch to float32 to probe
            else:
                mode = "16"
        else:
            mode = bitmode

        # Decode frames using chosen interpretation and chosen channel count (override experiments)
        try:
            samples, samples_i32, sample_bits, decode_note = decode_pcm(
                self.frames, nch_file=nch_file, user_nch=user_nch, mode=mode
            )
        except Exception as e:
            self.status.set(f"Decode failed: {e}")
            return

        width = int(self.var_width.get())
        downsample = int(self.var_downsample.get())
        empty_fill = int(self.var_empty.get())
        mono_pick = self.var_mono_pick.get()
        color_mode_label = self.var_color.get()
        bitmask_enable = bool(self.var_bitmask_enable.get())
        bitmask_low = int(self.var_mask_low.get())
        bitmask_high = int(self.var_mask_high.get())

        autotune_note = ""
        if bool(self.var_autotune.get()):
            try:
                span = int(self.var_autotune_span.get())
                step = int(self.var_autotune_step.get())
                s_f = samples[::max(1, downsample), :]
                s_i = samples_i32[::max(1, downsample), :] if samples_i32 is not None else None
                ref_f, ref_i = _pick_mono_stream(s_f, s_i, mono_pick)
                if bitmask_enable and (ref_i is not None) and (sample_bits is not None):
                    lo, hi, _ = _normalize_bit_window(bitmask_low, bitmask_high, sample_bits)
                    ref_stream = bit_window_to_u8(ref_i, sample_bits=sample_bits, bit_low=lo, bit_high=hi)
                else:
                    ref_stream = pcm_to_u8(ref_f.astype(np.float32, copy=False))

                width, autotune_note = autotune_width_from_stream(
                    ref_stream, requested_width=width, step=step, radius=span
                )
            except Exception as e:
                autotune_note = f"Autotune failed: {e}"

        # Build image
        try:
            img, build_note = build_image(
                samples_f32=samples,
                width=width,
                downsample=downsample,
                color_mode_label=color_mode_label,
                empty_fill=empty_fill,
                mono_pick=mono_pick,
                samples_i32=samples_i32,
                sample_bits=sample_bits,
                bitmask_enable=bitmask_enable,
                bitmask_low=bitmask_low,
                bitmask_high=bitmask_high,
            )
        except Exception as e:
            self.status.set(f"Build failed: {e}")
            return

        self._img_full = img

        # Display image
        fit = bool(self.var_fit.get())
        disp = img
        if fit:
            cw = max(1, self.canvas.winfo_width())
            ch = max(1, self.canvas.winfo_height())
            iw, ih = img.size
            s = min(cw / iw, ch / ih, 1.0)
            if s < 1.0:
                nw = max(1, int(iw * s))
                nh = max(1, int(ih * s))
                disp = img.resize((nw, nh), resample=Image.NEAREST)

        self._img_tk = ImageTk.PhotoImage(disp)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._img_tk)
        self.canvas.configure(scrollregion=(0, 0, disp.size[0], disp.size[1]))

        parts = [
            decode_note,
            f"WAV header: {nch_file}ch, {actual_bits}-bit, {fr} Hz",
        ]
        if autotune_note:
            parts.append(autotune_note)
        parts.append(build_note)
        self.status.set(" | ".join(parts))


def main():
    root = tk.Tk()
    # Make ttk look decent on Windows/Linux/macOS
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass

    app = App(root)
    root.minsize(900, 650)
    root.mainloop()


if __name__ == "__main__":
    main()
