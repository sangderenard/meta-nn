# pipeline/noise_spectrum.py
#
# Noise colour spectrum classifier — standalone utility.
# Maps between noise vocabulary terms and internal profile keys, and classifies
# a raw sample array by measuring its power-law spectral slope (beta exponent).
#
# Profile key convention: snake_case of the canonical term name
# (e.g. "pink noise" -> "pink_noise").  The term->key mapping is therefore
# derivable from DEFAULT_VOCABULARY without a separate LUT, except for the two
# white-noise variants that share the "white noise" parent term.

from __future__ import annotations
from typing import Any, Dict, List, Sequence
import re

import numpy as np


def noise_term_to_profile_key(term: str) -> str:
    """Return the internal profile key for a noise vocabulary term, or '' if not a noise term."""
    key = re.sub(r"\s+", " ", str(term)).strip().lower()
    # White noise has two variants — prefer gaussian unless the caller specifies.
    if key in ("noise", "uniform white noise"):
        return "uniform_white_noise"
    if key == "white noise":
        return "gaussian_white_noise"
    if key.endswith(" noise"):
        # "pink noise" -> "pink_noise", etc.
        return key.replace(" ", "_")
    return ""


def noise_profile_key_to_terms(profile_key: str) -> List[str]:
    """Return the vocabulary terms that describe a given profile key.
    Falls back to ["noise"] for unknown keys."""
    k = re.sub(r"\s+", "_", str(profile_key)).strip().lower()
    if k == "uniform_white_noise":
        return ["noise", "white noise", "uniform white noise"]
    if k == "gaussian_white_noise":
        return ["noise", "white noise", "gaussian white noise"]
    if k.endswith("_noise"):
        term = k.replace("_", " ")
        return ["noise", term]
    return ["noise"]


def classify_noise_spectrum(sample: Any) -> List[str]:
    """Classify a raw sample (image array or 1-D signal) by its power-law
    spectral slope and return matching noise vocabulary terms."""
    from pipeline.vocabulary_defaults import NOISE_SPECTRUM_BETA

    arr = np.asarray(sample, dtype=np.float32)
    if int(arr.size) <= 0:
        return []
    # Flatten to greyscale
    if int(arr.ndim) == 3:
        if int(arr.shape[0]) in (1, 3, 4):
            gray = np.mean(np.asarray(arr[:3, ...], dtype=np.float32), axis=0)
        elif int(arr.shape[2]) in (1, 3, 4):
            gray = np.mean(np.asarray(arr[..., :3], dtype=np.float32), axis=2)
        else:
            gray = np.asarray(arr, dtype=np.float32).reshape(-1)
    else:
        gray = np.asarray(arr, dtype=np.float32)

    flat0 = np.asarray(gray, dtype=np.float64).reshape(-1)
    if int(flat0.size) < 32:
        return []

    # Reject flat / degenerate signals
    try:
        dyn = float(np.max(flat0) - np.min(flat0)) if int(flat0.size) > 0 else 0.0
        centered0 = flat0 - float(np.mean(flat0))
        var0 = float(np.mean(centered0 * centered0)) if int(flat0.size) > 0 else 0.0
        diff0 = np.diff(flat0) if int(flat0.size) > 1 else np.zeros((0,), dtype=np.float64)
        mean_abs_diff0 = float(np.mean(np.abs(diff0))) if int(diff0.size) > 0 else 0.0
        entropy01 = 0.0
        if dyn > 1e-12:
            norm0 = np.clip((flat0 - float(np.min(flat0))) / dyn, 0.0, 1.0)
            hist0, _ = np.histogram(norm0, bins=64, range=(0.0, 1.0))
            hist0 = np.asarray(hist0, dtype=np.float64)
            hist0 = hist0 / max(1.0, float(np.sum(hist0)))
            hist0 = hist0[hist0 > 0.0]
            if int(hist0.size) > 0:
                entropy01 = float(-np.sum(hist0 * np.log2(hist0)) / np.log2(64.0))
        if dyn < 1e-3 or var0 < 1e-8:
            return []
        if entropy01 < 0.08 and mean_abs_diff0 < 2e-3:
            return []
    except Exception:
        pass

    # Measure excess kurtosis (distinguishes gaussian vs uniform white noise)
    beta = 0.0
    excess_kurtosis = 0.0
    try:
        flat = np.asarray(gray, dtype=np.float64).reshape(-1)
        if int(flat.size) >= 32:
            centered = flat - float(np.mean(flat))
            var = float(np.mean(centered * centered))
            if var > 1e-12:
                m4 = float(np.mean((centered * centered) * (centered * centered)))
                excess_kurtosis = float((m4 / (var * var)) - 3.0)
    except Exception:
        excess_kurtosis = 0.0

    # Estimate spectral slope
    try:
        if int(np.asarray(gray).ndim) == 2:
            g = np.asarray(gray, dtype=np.float64)
            g = g - float(np.mean(g))
            h, w = int(g.shape[0]), int(g.shape[1])
            if h >= 8 and w >= 8:
                p = np.abs(np.fft.fft2(g)).astype(np.float64) ** 2
                fy = np.fft.fftfreq(h).astype(np.float64)[:, None]
                fx = np.fft.fftfreq(w).astype(np.float64)[None, :]
                r = np.sqrt((fx * fx) + (fy * fy)).reshape(-1)
                pow_flat = p.reshape(-1)
                mask = (r > 1e-6) & np.isfinite(pow_flat) & (pow_flat > 1e-20)
                if int(np.count_nonzero(mask)) >= 32:
                    x = np.log(r[mask])
                    y = np.log(pow_flat[mask])
                    slope = float(np.polyfit(x, y, 1)[0])
                    beta = -slope
                else:
                    v = g.reshape(-1)
                    v = v - float(np.mean(v))
                    spec = np.abs(np.fft.rfft(v)).astype(np.float64) ** 2
                    fr = np.fft.rfftfreq(int(v.size)).astype(np.float64)
                    mask1 = (fr > 1e-6) & np.isfinite(spec) & (spec > 1e-20)
                    if int(np.count_nonzero(mask1)) >= 16:
                        slope = float(np.polyfit(np.log(fr[mask1]), np.log(spec[mask1]), 1)[0])
                        beta = -slope
        else:
            v = np.asarray(gray, dtype=np.float64).reshape(-1)
            v = v - float(np.mean(v))
            spec = np.abs(np.fft.rfft(v)).astype(np.float64) ** 2
            fr = np.fft.rfftfreq(int(v.size)).astype(np.float64)
            mask = (fr > 1e-6) & np.isfinite(spec) & (spec > 1e-20)
            if int(np.count_nonzero(mask)) >= 16:
                slope = float(np.polyfit(np.log(fr[mask]), np.log(spec[mask]), 1)[0])
                beta = -slope
    except Exception:
        beta = 0.0

    # Match beta to nearest noise colour
    best_term = "white noise"
    best_dist = float("inf")
    for name, target_beta in NOISE_SPECTRUM_BETA.items():
        d = abs(float(beta) - float(target_beta))
        if d < best_dist:
            best_dist = d
            best_term = str(name)

    if best_term == "white noise":
        dist_term = "uniform white noise" if float(excess_kurtosis) <= -0.55 else "gaussian white noise"
        return ["noise", "white noise", str(dist_term)]
    return ["noise", str(best_term)]
