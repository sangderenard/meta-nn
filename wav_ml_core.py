import math
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


COLOR_MODES = [
    ("Mono -> Grayscale (R=G=B)", 1, {"R": "M", "G": "M", "B": "M"}),
    ("Mono -> R only (G,B empty)", 1, {"R": "M", "G": None, "B": None}),
    ("Mono -> G only (R,B empty)", 1, {"R": None, "G": "M", "B": None}),
    ("Mono -> B only (R,G empty)", 1, {"R": None, "G": None, "B": "M"}),
    ("Stereo -> R=L, G=R (B empty)", 2, {"R": "L", "G": "R", "B": None}),
    ("Stereo -> R=L, B=R (G empty)", 2, {"R": "L", "G": None, "B": "R"}),
    ("Stereo -> G=L, B=R (R empty)", 2, {"R": None, "G": "L", "B": "R"}),
    ("Stereo -> R=R, G=L (B empty)", 2, {"R": "R", "G": "L", "B": None}),
    ("Single source -> RGB via sample stride", 1, {"__special__": "single_source_rgb_stride"}),
]
COLOR_MODE_MAP = {m[0]: (m[1], m[2]) for m in COLOR_MODES}


@dataclass
class WaveRecord:
    path: str
    nch: int
    sampwidth: int
    framerate: int
    nframes: int
    frames: bytes


@dataclass
class RenderConfig:
    bitmode: str = "auto"
    use_channels: str = "auto"  # auto / 1 / 2
    mono_pick: str = "Mix"      # Mix / L / R / LR stride
    width: int = 1024
    downsample: int = 1
    color_mode: str = COLOR_MODES[0][0]
    empty_fill: int = 0
    bitmask_enable: bool = False
    bitmask_low: int = 0
    bitmask_high: int = 7
    max_points: int = 262144

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def from_dict(x: Dict) -> "RenderConfig":
        return RenderConfig(**x)


def _sign_extend_24(u24: np.ndarray) -> np.ndarray:
    sign_bit = 1 << 23
    return (u24 ^ sign_bit) - sign_bit


def decode_pcm(frames: bytes, user_nch: int, mode: str):
    if len(frames) == 0:
        return np.zeros((0, user_nch), dtype=np.float32), None, None, "No audio frames."

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
        raise ValueError(f"Unsupported decode mode: {mode}")

    stride = user_nch * bps
    n = len(frames) // stride
    trimmed = frames[: n * stride]

    if n == 0:
        return np.zeros((0, user_nch), dtype=np.float32), None, None, "Not enough bytes for selected decode settings."

    if kind == "float":
        data = np.frombuffer(trimmed, dtype="<f4").astype(np.float32, copy=False).reshape(-1, user_nch)
        data = np.clip(data, -1.0, 1.0)
        return data, None, None, f"Decoded float32 {user_nch}ch"

    if bits == 8:
        u8 = np.frombuffer(trimmed, dtype=np.uint8).astype(np.int16)
        i = (u8 - 128).astype(np.int16).reshape(-1, user_nch)
        data = i.astype(np.float32) / 128.0
        return data, i.astype(np.int32), bits, f"Decoded int8 {user_nch}ch"
    if bits == 16:
        i = np.frombuffer(trimmed, dtype="<i2").reshape(-1, user_nch)
        data = i.astype(np.float32) / (2 ** 15)
        return data, i.astype(np.int32), bits, f"Decoded int16 {user_nch}ch"
    if bits == 24:
        b = np.frombuffer(trimmed, dtype=np.uint8).astype(np.int32)
        u24 = (b[0::3] | (b[1::3] << 8) | (b[2::3] << 16)).astype(np.int32)
        i24 = _sign_extend_24(u24).reshape(-1, user_nch)
        data = i24.astype(np.float32) / (2 ** 23)
        return data, i24, bits, f"Decoded int24 {user_nch}ch"
    if bits == 32:
        i = np.frombuffer(trimmed, dtype="<i4").reshape(-1, user_nch)
        data = i.astype(np.float32) / (2 ** 31)
        return data, i, bits, f"Decoded int32 {user_nch}ch"

    raise ValueError("Unsupported bit depth.")


def pcm_to_u8(x: np.ndarray) -> np.ndarray:
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
        mono_i = _interleave_lr(samples_i32[:, 0], samples_i32[:, 1]) if samples_i32 is not None else None
        return mono_f, mono_i

    mono_f = 0.5 * (samples_f32[:, 0] + samples_f32[:, 1])
    if samples_i32 is None:
        return mono_f, None
    mono_i = ((samples_i32[:, 0].astype(np.int64) + samples_i32[:, 1].astype(np.int64)) // 2).astype(np.int32)
    return mono_f, mono_i


def normalize_bit_window(bit_low: int, bit_high: int, sample_bits: int):
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


def build_image_array(
    samples_f32: np.ndarray,
    width: int,
    downsample: int,
    color_mode_label: str,
    empty_fill: int,
    mono_pick: str,
    samples_i32=None,
    sample_bits: Optional[int] = None,
    bitmask_enable: bool = False,
    bitmask_low: int = 0,
    bitmask_high: int = 7,
    max_points: int = 262144,
) -> Tuple[np.ndarray, str]:
    width = max(1, int(width))
    downsample = max(1, int(downsample))
    max_points = max(1024, int(max_points))

    n, c = samples_f32.shape
    if n == 0:
        return np.full((1, 1, 3), int(empty_fill) & 0xFF, dtype=np.uint8), "No samples"

    s_f = samples_f32[::downsample, :]
    s_i = samples_i32[::downsample, :] if samples_i32 is not None else None

    mode = COLOR_MODE_MAP.get(color_mode_label, COLOR_MODE_MAP[COLOR_MODES[0][0]])
    need_ch, mapping = mode
    special = mapping.get("__special__")

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
            if c == 1:
                lf = s_f[:, 0]
                rf = s_f[:, 0]
                li = s_i[:, 0] if s_i is not None else None
                ri = s_i[:, 0] if s_i is not None else None
            else:
                lf = s_f[:, 0]
                rf = s_f[:, 1]
                li = s_i[:, 0] if s_i is not None else None
                ri = s_i[:, 1] if s_i is not None else None
            sources_f["L"] = lf
            sources_f["R"] = rf
            if li is not None and ri is not None:
                sources_i["L"] = li
                sources_i["R"] = ri

    can_mask = bool(bitmask_enable) and (sample_bits is not None) and (len(sources_i) > 0)
    if can_mask:
        mask_low_eff, mask_high_eff, _ = normalize_bit_window(bitmask_low, bitmask_high, sample_bits)
    else:
        mask_low_eff, mask_high_eff = None, None

    sources_u8 = {}
    for k, vf in sources_f.items():
        vi = sources_i.get(k)
        if can_mask and vi is not None:
            stream = bit_window_to_u8(vi, sample_bits=sample_bits, bit_low=mask_low_eff, bit_high=mask_high_eff)
        else:
            stream = pcm_to_u8(vf.astype(np.float32, copy=False))
        if stream.size > max_points:
            stream = stream[:max_points]
        sources_u8[k] = stream

    if special == "single_source_rgb_stride":
        base = sources_u8["SRC"]
        channels = {"R": base[0::3], "G": base[1::3], "B": base[2::3]}
    else:
        channels = {
            "R": (sources_u8[mapping.get("R")] if mapping.get("R") is not None else None),
            "G": (sources_u8[mapping.get("G")] if mapping.get("G") is not None else None),
            "B": (sources_u8[mapping.get("B")] if mapping.get("B") is not None else None),
        }

    present = [v for v in channels.values() if v is not None]
    max_len = max((int(v.size) for v in present), default=1)
    total = int(math.ceil(max_len / width) * width)
    height = max(1, total // width)
    canvas = np.full((height, width, 3), int(empty_fill) & 0xFF, dtype=np.uint8)

    for key, idx in (("R", 0), ("G", 1), ("B", 2)):
        stream = channels.get(key)
        if stream is None:
            continue
        if stream.size < total:
            pad = np.full((total - stream.size,), int(empty_fill) & 0xFF, dtype=np.uint8)
            stream = np.concatenate([stream, pad], axis=0)
        canvas[:, :, idx] = stream[:total].reshape(height, width)

    if bitmask_enable and can_mask:
        note = f"bitmask bits {mask_low_eff}..{mask_high_eff}"
    elif bitmask_enable:
        note = "bitmask requested but unavailable in current decode mode"
    else:
        note = "amplitude"

    return canvas, note


def read_wav_record(path: str) -> WaveRecord:
    with wave.open(path, "rb") as wf:
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        fr = wf.getframerate()
        nf = wf.getnframes()
        frames = wf.readframes(nf)
    return WaveRecord(path=path, nch=nch, sampwidth=sw, framerate=fr, nframes=nf, frames=frames)


def discover_wavs(root_dir: str) -> List[str]:
    root = Path(root_dir)
    return sorted(str(p) for p in root.rglob("*.wav"))


def _resolve_decode_mode(sampwidth: int, bitmode: str) -> str:
    if bitmode != "auto":
        return bitmode
    if sampwidth == 1:
        return "8"
    if sampwidth == 2:
        return "16"
    if sampwidth == 3:
        return "24"
    if sampwidth == 4:
        return "32"
    return "16"


def render_record(record: WaveRecord, cfg: RenderConfig):
    use_ch = cfg.use_channels
    if use_ch == "auto":
        user_nch = record.nch if record.nch in (1, 2) else 2
    else:
        user_nch = int(use_ch)

    mode = _resolve_decode_mode(record.sampwidth, cfg.bitmode)
    samples, samples_i32, sample_bits, decode_note = decode_pcm(record.frames, user_nch=user_nch, mode=mode)

    image_u8, image_note = build_image_array(
        samples_f32=samples,
        width=cfg.width,
        downsample=cfg.downsample,
        color_mode_label=cfg.color_mode,
        empty_fill=cfg.empty_fill,
        mono_pick=cfg.mono_pick,
        samples_i32=samples_i32,
        sample_bits=sample_bits,
        bitmask_enable=cfg.bitmask_enable,
        bitmask_low=cfg.bitmask_low,
        bitmask_high=cfg.bitmask_high,
        max_points=cfg.max_points,
    )
    meta = {
        "decode_note": decode_note,
        "image_note": image_note,
        "sample_bits": sample_bits,
        "mode": mode,
        "user_nch": user_nch,
    }
    return image_u8, meta


def image_u8_to_tensor(image_u8: np.ndarray, image_hw: Tuple[int, int]) -> torch.Tensor:
    t = torch.from_numpy(image_u8).permute(2, 0, 1).float() / 255.0
    if image_hw is not None:
        h, w = int(image_hw[0]), int(image_hw[1])
        t = F.interpolate(t.unsqueeze(0), size=(h, w), mode="nearest").squeeze(0)
    return t


def record_to_tensor(record: WaveRecord, cfg: RenderConfig, image_hw: Tuple[int, int]) -> torch.Tensor:
    img_u8, _ = render_record(record, cfg)
    return image_u8_to_tensor(img_u8, image_hw=image_hw)


def _ste_floor(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.floor(x) - x).detach()


def _bit_window_from_mono_ste(wave_mono: torch.Tensor, sample_bits: int, bit_low: int, bit_high: int):
    lo, hi, depth = normalize_bit_window(bit_low, bit_high, sample_bits)
    # Match bit_window_to_u8 semantics exactly: quantize to signed PCM, reinterpret
    # as two's-complement unsigned, then slice the requested bit window.
    q_min = -(1 << (int(sample_bits) - 1))
    q_max = (1 << (int(sample_bits) - 1)) - 1
    q_scale = float(1 << (int(sample_bits) - 1))
    q_signed = torch.round(torch.clamp(wave_mono, -1.0, 1.0) * q_scale)
    q_signed = torch.clamp(q_signed, float(q_min), float(q_max)).to(torch.int64)

    full_mask = (1 << int(sample_bits)) - 1
    window_mask = (1 << int(depth)) - 1
    unsigned = torch.bitwise_and(q_signed, full_mask)
    window = torch.bitwise_and(torch.bitwise_right_shift(unsigned, int(lo)), window_mask)

    if depth == 1:
        hard = window.to(dtype=wave_mono.dtype)
    else:
        hard = window.to(dtype=wave_mono.dtype) / float(window_mask)
    hard = torch.clamp(hard, 0.0, 1.0)
    x01 = torch.clamp((wave_mono * 0.5) + 0.5, 0.0, 1.0)
    # Straight-through surrogate: preserve exact forward bit-window,
    # but keep a non-zero backward path to waveform amplitudes.
    return hard + (x01 - x01.detach()), lo, hi, depth


def render_mono_wave_to_tensor(
    wave_mono: torch.Tensor,
    cfg: RenderConfig,
    image_hw: Tuple[int, int],
    sample_bits: int = 16,
) -> torch.Tensor:
    if wave_mono.ndim != 2:
        raise ValueError("wave_mono must have shape [B, T]")
    bsz, n = wave_mono.shape
    if n == 0:
        return torch.zeros((bsz, 3, image_hw[0], image_hw[1]), dtype=wave_mono.dtype, device=wave_mono.device)

    downsample = max(1, int(cfg.downsample))
    width = max(1, int(cfg.width))
    empty_fill = float(int(cfg.empty_fill) & 0xFF) / 255.0

    stream = wave_mono[:, ::downsample]
    if cfg.max_points > 0 and stream.shape[1] > cfg.max_points:
        stream = stream[:, : cfg.max_points]

    if cfg.bitmask_enable:
        base, _, _, _ = _bit_window_from_mono_ste(stream, sample_bits, cfg.bitmask_low, cfg.bitmask_high)
    else:
        base = torch.clamp((stream * 0.5) + 0.5, 0.0, 1.0)

    _, mapping = COLOR_MODE_MAP.get(cfg.color_mode, COLOR_MODE_MAP[COLOR_MODES[0][0]])
    special = mapping.get("__special__")

    if special == "single_source_rgb_stride":
        channel_streams = {
            "R": base[:, 0::3],
            "G": base[:, 1::3],
            "B": base[:, 2::3],
        }
    else:
        l_stream = base
        r_stream = base
        m_stream = base
        source_map = {"L": l_stream, "R": r_stream, "M": m_stream}
        channel_streams = {
            "R": source_map.get(mapping.get("R")),
            "G": source_map.get(mapping.get("G")),
            "B": source_map.get(mapping.get("B")),
        }

    present = [v for v in channel_streams.values() if v is not None]
    max_len = max((v.shape[1] for v in present), default=1)
    total = int(math.ceil(float(max_len) / float(width)) * width)
    height = max(1, total // width)

    canvas = torch.full((bsz, 3, height, width), empty_fill, dtype=wave_mono.dtype, device=wave_mono.device)
    for ch_name, ch_idx in (("R", 0), ("G", 1), ("B", 2)):
        s = channel_streams.get(ch_name)
        if s is None:
            continue
        if s.shape[1] < total:
            s = F.pad(s, (0, total - s.shape[1]), value=empty_fill)
        canvas[:, ch_idx, :, :] = s[:, :total].reshape(bsz, height, width)

    h, w = int(image_hw[0]), int(image_hw[1])
    if canvas.shape[2] != h or canvas.shape[3] != w:
        canvas = F.interpolate(canvas, size=(h, w), mode="nearest")
    return canvas
