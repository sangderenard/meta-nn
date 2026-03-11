"""
Wave I/O — audio read/write, synthesis, and noise-profile utilities.

Functions here deal with the physical audio domain:

  * Saving / loading mono WAV files (PCM-16)
  * Fitting / padding waveforms to a target length
  * Procedural structured-tone generation (chirps, envelopes)
  * Coloured-noise synthesis (pink, brown, blue, …)
  * Noise-profile key sampling and spectral shaping
  * Library member path resolution and reinject-index loading
  * Multi-channel WAV decoding to float32 mono

These are used by both the wave-pool bootstrap in data_nodes and
the wave-classifier training loop — they are *not* model-specific.
"""
from __future__ import annotations

import json
import re
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Basic WAV I/O
# ---------------------------------------------------------------------------

def _save_mono_wav(path: Path, mono_f32: np.ndarray, framerate: int):
    """Write a mono float32 waveform as PCM-16 WAV."""
    y = np.clip(mono_f32, -1.0, 1.0)
    i16 = np.round(y * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(framerate))
        wf.writeframes(i16.tobytes())


def _fit_wave_length(x: np.ndarray, target_samples: int, rng: np.random.Generator):
    """Pad (tile) or crop *x* so it has exactly *target_samples* elements."""
    target = max(1, int(target_samples))
    if x.size == 0:
        return np.zeros((target,), dtype=np.float32)
    if x.size >= target:
        if x.size == target:
            return x.astype(np.float32, copy=False)
        start = int(rng.integers(0, x.size - target + 1))
        return x[start : start + target].astype(np.float32, copy=False)
    reps = int(np.ceil(float(target) / float(x.size)))
    y = np.tile(x, reps)[:target]
    return y.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Decoding helpers
# ---------------------------------------------------------------------------

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


def _decode_record_to_mono(record, cfg, max_points: int = 0):
    """Decode a WaveRecord to float32 mono according to *cfg*.

    Parameters
    ----------
    record : WaveRecord   (from wav_ml_core)
    cfg    : RenderConfig  (from wav_ml_core)
    """
    from wav_ml_core import decode_pcm

    if cfg.use_channels == "auto":
        user_nch = record.nch if record.nch in (1, 2) else 2
    else:
        user_nch = int(cfg.use_channels)

    mode = _resolve_decode_mode(record.sampwidth, cfg.bitmode)
    samples_f32, _samples_i32, sample_bits, _ = decode_pcm(
        record.frames, user_nch=user_nch, mode=mode,
    )
    if samples_f32.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32), (sample_bits if sample_bits is not None else 16)

    if samples_f32.shape[1] == 1:
        mono = samples_f32[:, 0]
    else:
        if cfg.mono_pick == "L":
            mono = samples_f32[:, 0]
        elif cfg.mono_pick == "R":
            mono = samples_f32[:, 1]
        elif cfg.mono_pick == "LR stride":
            out = np.empty((samples_f32.shape[0] * 2,), dtype=np.float32)
            out[0::2] = samples_f32[:, 0]
            out[1::2] = samples_f32[:, 1]
            mono = out
        else:
            mono = 0.5 * (samples_f32[:, 0] + samples_f32[:, 1])

    if max_points > 0 and mono.size > max_points:
        mono = mono[:max_points]
    bits = sample_bits if sample_bits is not None else (32 if mode == "float32" else 16)
    return mono.astype(np.float32, copy=False), int(bits)


# ---------------------------------------------------------------------------
# Structured-tone synthesis
# ---------------------------------------------------------------------------

def _synthesize_structured_wave(
    target_samples: int, framerate: int, rng: np.random.Generator,
) -> np.ndarray:
    """Generate a procedural multi-component chirp with envelope + tremolo."""
    n = max(256, int(target_samples))
    sr = max(1000, int(framerate))
    t = (np.arange(n, dtype=np.float32) / float(sr)).astype(np.float32, copy=False)
    y = np.zeros((n,), dtype=np.float32)

    n_comp = int(rng.integers(2, 6))
    dur = float(n) / float(sr)
    for _ in range(n_comp):
        f0 = float(rng.uniform(40.0, 1800.0))
        f1 = float(np.clip(f0 * float(rng.uniform(0.65, 1.6)), 20.0, 2400.0))
        phase0 = float(rng.uniform(0.0, 2.0 * np.pi))
        if bool(rng.random() < 0.5):
            k = (f1 - f0) / max(1e-6, dur)
            phase = (2.0 * np.pi * ((f0 * t) + (0.5 * k * t * t))) + phase0
        else:
            phase = (2.0 * np.pi * f0 * t) + phase0
        amp = float(rng.uniform(0.3, 1.0))
        y += (amp * np.sin(phase)).astype(np.float32, copy=False)

    attack = int(min(n, max(0, int(float(rng.uniform(0.01, 0.12)) * n))))
    release = int(min(n, max(0, int(float(rng.uniform(0.08, 0.35)) * n))))
    env = np.ones((n,), dtype=np.float32)
    if attack > 1:
        env[:attack] = np.linspace(0.0, 1.0, attack, endpoint=False, dtype=np.float32)
    if release > 1:
        env[-release:] *= np.linspace(1.0, 0.0, release, endpoint=True, dtype=np.float32)
    mod_f = float(rng.uniform(0.4, 8.0))
    mod_depth = float(rng.uniform(0.1, 0.65))
    mod = (1.0 - mod_depth) + (mod_depth * (0.5 * (1.0 + np.sin(
        (2.0 * np.pi * mod_f * t) + float(rng.uniform(0.0, 2.0 * np.pi)),
    ))))
    y = y * env * mod.astype(np.float32, copy=False)

    peak = float(np.max(np.abs(y))) if y.size > 0 else 0.0
    if peak > 1e-6:
        y = y / peak
    return y.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Noise-profile catalogue
# ---------------------------------------------------------------------------

def _latent_noise_profile_keys() -> List[str]:
    return [
        "uniform_white_noise",
        "gaussian_white_noise",
        "pink_noise",
        "brown_noise",
        "red_noise",
        "blue_noise",
        "violet_noise",
        "gray_noise",
    ]


def _sample_latent_noise_profile_key(rng: np.random.Generator) -> str:
    keys = _latent_noise_profile_keys()
    if len(keys) <= 0:
        return "gaussian_white_noise"
    weights = np.asarray(
        [0.20, 0.20, 0.15, 0.12, 0.10, 0.09, 0.07, 0.07], dtype=np.float64,
    )
    if int(weights.size) != int(len(keys)):
        return str(keys[int(rng.integers(0, len(keys)))])
    weights = weights / max(1e-12, float(np.sum(weights)))
    pick = int(rng.choice(np.arange(len(keys), dtype=np.int64), p=weights))
    return str(keys[int(pick)])


def _latent_noise_profile_spec(profile_key: str) -> Tuple[float, str]:
    key = re.sub(r"\s+", "_", str(profile_key)).strip().lower()
    lut: Dict[str, Tuple[float, str]] = {
        "uniform_white_noise": (0.0, "uniform"),
        "gaussian_white_noise": (0.0, "gaussian"),
        "pink_noise": (1.0, "gaussian"),
        "brown_noise": (2.0, "gaussian"),
        "red_noise": (1.8, "gaussian"),
        "blue_noise": (-1.0, "gaussian"),
        "violet_noise": (-2.0, "gaussian"),
        "gray_noise": (0.5, "gaussian"),
    }
    return tuple(lut.get(key, (0.0, "gaussian")))


def _synthesize_profiled_noise_wave(
    target_samples: int,
    framerate: int,
    profile_key: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Synthesise a coloured-noise waveform shaped by *profile_key*."""
    n = max(256, int(target_samples))
    sr = max(1000, int(framerate))
    beta, dist = _latent_noise_profile_spec(profile_key)
    if str(dist).strip().lower() == "uniform":
        base = rng.uniform(-1.0, 1.0, size=(n,)).astype(np.float32, copy=False)
    else:
        base = rng.standard_normal((n,)).astype(np.float32, copy=False)
    if abs(float(beta)) > 1e-6:
        spec = np.fft.rfft(base).astype(np.complex64, copy=False)
        freqs = np.fft.rfftfreq(int(n), d=(1.0 / float(sr))).astype(np.float32, copy=False)
        denom = np.maximum(freqs, 1.0).astype(np.float32, copy=False)
        shape = np.power(denom, -0.5 * float(beta)).astype(np.float32, copy=False)
        shape[0] = 0.0
        shaped = spec * shape.astype(np.complex64, copy=False)
        base = np.fft.irfft(shaped, n=int(n)).astype(np.float32, copy=False)
    base = base - float(np.mean(base))
    std = float(np.std(base)) if int(base.size) > 0 else 0.0
    if std > 1e-8:
        base = base / std
    return np.asarray(base, dtype=np.float32)


# ---------------------------------------------------------------------------
# Path / library helpers
# ---------------------------------------------------------------------------

def _latent_noise_profile_key_from_path(path: str) -> str:
    name = Path(str(path)).name.lower()
    m = re.search(r"latent_\d+_([a-z0-9_]+)\.wav$", name)
    if m is None:
        return ""
    key = re.sub(r"\s+", "_", str(m.group(1))).strip().lower()
    if key in set(_latent_noise_profile_keys()):
        return str(key)
    return ""


def _latent_pool_stream_kind(path: str) -> str:
    p = str(path).replace("\\", "/").lower()
    if "/latent_wave_pool/noise/" in p:
        return "noise"
    if "/latent_wave_pool/mix/" in p:
        return "mix"
    return "other"


def _latent_stream_semantic_terms_from_path(path: str) -> List[str]:
    from pipeline.nodes.vocab_node import (
        _normalize_vocab_terms,
        _semantic_noise_profile_terms,
    )
    p = str(path)
    kind = _latent_pool_stream_kind(p)
    profile_key = _latent_noise_profile_key_from_path(p)
    noise_terms = _semantic_noise_profile_terms(profile_key if profile_key else "gaussian_white_noise")
    if kind == "noise":
        return _normalize_vocab_terms(list(noise_terms))
    if kind == "mix":
        return _normalize_vocab_terms(["mixed noise and signal", "signal"] + list(noise_terms))
    return []


def _resolve_library_member_path(path_text: str, library_dir: Path) -> Optional[Path]:
    txt = str(path_text).strip()
    if not txt:
        return None
    try:
        root = library_dir.resolve()
    except Exception:
        root = Path(str(library_dir))
    p = Path(txt)
    candidates: List[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(Path.cwd() / p)
        candidates.append(library_dir / p)
    for c in candidates:
        if not c.exists():
            continue
        try:
            found = c.resolve()
        except Exception:
            continue
        if found == root or root in found.parents:
            return found
    return None


def _load_reinject_wave_paths(library_dir: Path, limit: int = 0) -> List[str]:
    idx_path = library_dir / "index.jsonl"
    if not idx_path.exists():
        return []

    out: List[str] = []
    with idx_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except Exception:
                continue
            fp = str(row.get("file", "")).strip()
            if not fp:
                continue
            found = _resolve_library_member_path(fp, library_dir=library_dir)
            if found is None:
                continue
            out.append(str(found))
            if limit > 0 and len(out) >= int(limit):
                break
    return out
