import argparse
import gc
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
import wave
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from wav_ml_core import (
    COLOR_MODE_MAP,
    COLOR_MODES,
    RenderConfig,
    WaveRecord,
    decode_pcm,
    discover_wavs,
    image_u8_to_tensor,
    normalize_bit_window,
    read_wav_record,
    render_record,
    render_mono_wave_to_tensor,
)
from wav_ml_models import (
    _TransformerStatusOpenGLViewer,
    ConditionalBitPlaneDiscriminator,
    ConditionalBitPlaneGenerator,
    SinusoidalLRController,
    SinusoidalLROptions,
    TinyConvClassifier,
    WavePatchTransformer,
    evaluate_conditional_generator,
    evaluate_classifier,
    evaluate_feature_score_before_after,
    evaluate_transformer_accuracy,
    multilabel_feature_score,
    configure_torch_runtime,
    maybe_compile_module,
    set_seed,
    train_conditional_generator_discriminator,
    train_classifier,
    train_transformer_feature_metric,
)

_ST_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}
GUI_STOP_EXIT_CODE = 42
CLASSIFIER_LOSS_SCALE = 10.0
CLASSIFIER_SEMANTIC_COSINE_WEIGHT = 0.35


def _log(msg: str):
    print(msg, flush=True)


def _resolve_amp_dtype(amp_dtype: str) -> torch.dtype:
    key = str(amp_dtype).strip().lower()
    if key in ("fp16", "float16", "half"):
        return torch.float16
    if key in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported amp dtype: {amp_dtype!r}")


def _autocast_context(device: torch.device, enabled: bool, amp_dtype_t: torch.dtype):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type=device.type, dtype=amp_dtype_t)
    return nullcontext()


def _dataloader_perf_kwargs(num_workers: int, persistent_workers: bool, prefetch_factor: int):
    if int(num_workers) <= 0:
        return {}
    out = {"persistent_workers": bool(persistent_workers)}
    if int(prefetch_factor) > 0:
        out["prefetch_factor"] = int(prefetch_factor)
    return out


def _make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _module_device_type(module: nn.Module) -> str:
    try:
        return next(module.parameters()).device.type
    except StopIteration:
        pass
    except Exception:
        return "cpu"
    try:
        return next(module.buffers()).device.type
    except StopIteration:
        return "cpu"
    except Exception:
        return "cpu"


def _torch_load_cpu(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Older torch versions may not support weights_only.
        return torch.load(path, map_location="cpu")
    except Exception:
        # Fallback for checkpoints that require legacy unpickling behavior.
        return torch.load(path, map_location="cpu", weights_only=False)


def _hard_wipe_pipeline_caches(output_dir: Path, berkeley_data_root: str) -> Dict[str, Any]:
    out_dir = Path(output_dir)
    berkeley_root = Path(str(berkeley_data_root).strip() or "toys_to_survive_development/data/berkeley_sbd")
    targets: List[Path] = [
        out_dir / "accepted_wave_library",
        out_dir / "latent_wave_pool",
        out_dir / "training_supervision",
    ]
    berkeley_cache_root = berkeley_root / "cache"
    if berkeley_cache_root.exists():
        for payload_dir in sorted(berkeley_cache_root.glob("payload_bank_rgb*")):
            targets.append(Path(payload_dir))

    def _remove_path_retry(path: Path, retries: int = 6) -> Tuple[bool, str]:
        last_err = ""
        for attempt in range(max(1, int(retries))):
            try:
                if not path.exists():
                    return False, ""
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                return True, ""
            except FileNotFoundError:
                return False, ""
            except PermissionError as e:
                last_err = f"{type(e).__name__}: {e}"
                gc.collect()
                time.sleep(0.15 * float(attempt + 1))
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                break
        return False, str(last_err)

    info: Dict[str, Any] = {
        "requested": [str(p) for p in targets],
        "removed": [],
        "missing": [],
        "errors": [],
    }
    for p in targets:
        if not p.exists():
            info["missing"].append(str(p))
            continue
        ok, err = _remove_path_retry(p)
        if bool(ok):
            info["removed"].append(str(p))
        else:
            info["errors"].append(f"{str(p)} ({str(err)})")
    return info


def _soft_reset_label_caches(output_dir: Path, berkeley_data_root: str) -> Dict[str, Any]:
    _ = Path(output_dir)
    berkeley_root = Path(str(berkeley_data_root).strip() or "toys_to_survive_development/data/berkeley_sbd")
    targets: List[Path] = []
    berkeley_cache_root = berkeley_root / "cache"
    if berkeley_cache_root.exists():
        for payload_dir in sorted(berkeley_cache_root.glob("payload_bank_rgb*")):
            pd = Path(payload_dir)
            targets.extend(
                [
                    pd / "labels.npy",
                    pd / "labels.tmp.npy",
                    pd / "terms.json",
                    pd / "sources.json",
                    pd / "manifest.json",
                    pd / "source_catalog.json",
                    pd / "semantic_gate_labels",
                ]
            )

    def _remove_path_retry(path: Path, retries: int = 6) -> Tuple[bool, str]:
        last_err = ""
        for attempt in range(max(1, int(retries))):
            try:
                if not path.exists():
                    return False, ""
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                return True, ""
            except FileNotFoundError:
                return False, ""
            except PermissionError as e:
                last_err = f"{type(e).__name__}: {e}"
                gc.collect()
                time.sleep(0.15 * float(attempt + 1))
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                break
        return False, str(last_err)

    info: Dict[str, Any] = {
        "requested": [str(p) for p in targets],
        "removed": [],
        "missing": [],
        "errors": [],
    }
    for p in targets:
        if not p.exists():
            info["missing"].append(str(p))
            continue
        ok, err = _remove_path_retry(p)
        if bool(ok):
            info["removed"].append(str(p))
        else:
            info["errors"].append(f"{str(p)} ({str(err)})")
    return info


def _normalize_l2_rows_np(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array for row normalization, got shape={tuple(x.shape)}")
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.clip(n, 1e-8, None)
    return (x / n).astype(np.float32, copy=False)


def _resolve_label_texts(class_names: Sequence[str], override_json: str) -> List[str]:
    base = [re.sub(r"\s+", " ", str(x)).strip() for x in class_names]
    base = [x if len(x) > 0 else "__empty_label__" for x in base]
    path = str(override_json).strip()
    if not path:
        return base
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Label text override JSON not found: {p}")
    blob = json.loads(p.read_text(encoding="utf-8"))
    out = list(base)
    if isinstance(blob, list):
        if len(blob) == len(out):
            out = [re.sub(r"\s+", " ", str(x)).strip() for x in blob]
        else:
            n = min(len(blob), len(out))
            for i in range(n):
                out[i] = re.sub(r"\s+", " ", str(blob[i])).strip()
        return [x if len(x) > 0 else "__empty_label__" for x in out]
    if isinstance(blob, dict):
        for i, name in enumerate(base):
            key_idx = str(i)
            if key_idx in blob:
                out[i] = re.sub(r"\s+", " ", str(blob[key_idx])).strip()
            elif name in blob:
                out[i] = re.sub(r"\s+", " ", str(blob[name])).strip()
        return [x if len(x) > 0 else "__empty_label__" for x in out]
    raise RuntimeError("Label text override JSON must be a list[str] or dict[str,str].")


def _encode_texts_sentence_transformers(texts: Sequence[str], model_name: str, device: torch.device) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError(
            "sentence-transformers backend requested but package is unavailable. "
            "Install with: pip install sentence-transformers"
        ) from e
    st_device = "cuda" if (device.type == "cuda" and torch.cuda.is_available()) else "cpu"
    cache_key = (str(model_name), str(st_device))
    model = _ST_MODEL_CACHE.get(cache_key, None)
    if model is None:
        model = SentenceTransformer(str(model_name), device=st_device)
        _ST_MODEL_CACHE[cache_key] = model
    vec = model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    if not isinstance(vec, np.ndarray):
        vec = np.asarray(vec, dtype=np.float32)
    return _normalize_l2_rows_np(vec.astype(np.float32, copy=False))


def _build_label_embedding_bank(
    class_names: Sequence[str],
    args,
    device: torch.device,
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    label_texts = _resolve_label_texts(class_names=class_names, override_json=str(args.label_text_json))
    backend_req = str(args.label_embedding_backend).strip().lower()
    if backend_req != "sentence_transformers":
        raise RuntimeError(
            "Only sentence-transformers semantic vectors are permitted. "
            f"Received --label-embedding-backend={backend_req!r}."
        )
    model_name = str(args.label_embedding_model).strip()
    emb = _encode_texts_sentence_transformers(label_texts, model_name=model_name, device=device)
    native_dim = int(emb.shape[1])

    target_dim = int(args.label_embedding_dim)
    if target_dim > 0 and target_dim != int(emb.shape[1]):
        raise RuntimeError(
            "Sentence-transformer geometry is configured as strict; "
            f"--label-embedding-dim must match model native dim ({int(emb.shape[1])}), got {int(target_dim)}."
        )

    info = {
        "enabled": True,
        "backend_requested": str(args.label_embedding_backend),
        "backend_used": "sentence_transformers",
        "model_name": model_name,
        "num_classes": int(len(class_names)),
        "native_dim": int(native_dim),
        "dim": int(emb.shape[1]),
        "temperature": float(args.label_embedding_temperature),
        "fallback_reason": "",
    }
    return emb.astype(np.float32, copy=False), label_texts, info


def _apply_label_embedding_bank_to_classifier(
    classifier: nn.Module,
    bank_np: Optional[np.ndarray],
    args,
) -> Dict[str, Any]:
    target = classifier
    if not isinstance(target, TinyConvClassifier):
        wrapped = getattr(classifier, "_orig_mod", None)
        if isinstance(wrapped, TinyConvClassifier):
            target = wrapped
        else:
            return {"applied": False, "reason": f"unsupported_model:{type(classifier).__name__}"}
    if not isinstance(target, TinyConvClassifier):
        return {"applied": False, "reason": f"unsupported_model:{type(classifier).__name__}"}
    if bank_np is None:
        target.disable_label_embedding_bank()
        return {"applied": False, "reason": "bank_none"}
    bank_arr = np.asarray(bank_np, dtype=np.float32)
    if int(bank_arr.ndim) != 2 or int(bank_arr.shape[0]) <= 0 or int(bank_arr.shape[1]) <= 0:
        return {"applied": False, "reason": f"invalid_bank_shape:{tuple(bank_arr.shape)}"}
    bank_arr = _normalize_l2_rows_np(bank_arr.astype(np.float32, copy=False))
    device = next(target.parameters()).device
    bank_t = torch.from_numpy(bank_arr).to(device=device)
    target.set_label_embedding_bank(bank_t, temperature=float(args.label_embedding_temperature))
    return {
        "applied": True,
        "classes": int(bank_t.shape[0]),
        "dim": int(bank_t.shape[1]),
        "temperature": float(args.label_embedding_temperature),
    }


def _parse_label_query_texts(raw: str) -> List[str]:
    s = str(raw).strip()
    if not s:
        return []
    toks = re.split(r"[|\n;,]+", s)
    out = [re.sub(r"\s+", " ", t).strip() for t in toks]
    return [t for t in out if len(t) > 0]


def _load_vocab_terms_json(path: str) -> List[str]:
    p_raw = str(path).strip()
    if not p_raw:
        return []
    p = Path(p_raw)
    if not p.exists():
        raise FileNotFoundError(f"Vocabulary term JSON not found: {p}")
    blob = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(blob, list):
        raw_terms = list(blob)
    elif isinstance(blob, dict):
        raw_terms = []
        for key in ("terms", "vocab", "words", "labels", "entries"):
            vals = blob.get(key, None)
            if isinstance(vals, list):
                raw_terms = list(vals)
                break
        if len(raw_terms) <= 0:
            raw_terms = list(blob.values())
    else:
        raise RuntimeError("Vocabulary term JSON must be a list[str] or dict containing a list field.")

    out: List[str] = []
    seen: set = set()
    for x in raw_terms:
        t = re.sub(r"\s+", " ", str(x)).strip()
        if not t:
            continue
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out


def _merge_vocab_terms(base_terms: Sequence[str], extra_terms: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: set = set()

    def _push(x: str):
        t = re.sub(r"\s+", " ", str(x)).strip()
        if not t:
            return
        k = t.lower()
        if k in seen:
            return
        seen.add(k)
        out.append(t)

    for t in base_terms:
        _push(str(t))
    for t in extra_terms:
        _push(str(t))
    return out


def _normalize_vocab_terms(terms: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: set = set()
    for x in terms:
        t = re.sub(r"\s+", " ", str(x)).strip()
        if not t:
            continue
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out


def _normalize_active_extra_terms_with_core(
    active_terms: Sequence[str],
    core_terms: Sequence[str],
    total_slots: int,
) -> List[str]:
    core = _normalize_vocab_terms(core_terms)
    active = _normalize_vocab_terms(active_terms)
    core_lc = {str(x).strip().lower() for x in core}
    tail = [t for t in active if str(t).strip().lower() not in core_lc]
    slots = max(int(len(core)), max(0, int(total_slots)))
    out = list(core) + list(tail)
    while len(out) < int(slots):
        out.append(f"semantic slot {int(len(out)) + 1}")
    return out[: int(slots)]


_DEFAULT_SEMANTIC_CORE_TERMS = [
    "none",
    "noise",
    "signal",
    "object",
    "mixed noise and signal",
    "white",
    "black",
    "blur damage",
    "noise damage",
    "dropout damage",
    "quantization damage",
    "stride skew damage",
    "gan image",
    "regurgitated content",
    "berkeley sbd dataset",
    "mnist dataset",
    "emnist dataset",
    "kmnist dataset",
]


def _default_semantic_core_terms() -> List[str]:
    return [str(x) for x in _DEFAULT_SEMANTIC_CORE_TERMS]


def _default_bootstrap_primitive_terms() -> List[str]:
    # Bootstrap/gestation vocabulary is strictly primitive internal semantics only:
    # no dataset marker terms.
    core_terms = _normalize_vocab_terms(_default_semantic_core_terms())
    color_terms = _normalize_vocab_terms(
        ["red", "green", "blue", "yellow", "cyan", "magenta", "brown", "gray", "grey"]
    )
    direction_terms = _normalize_vocab_terms(
        ["front", "back", "left", "right", "top", "bottom"]
    )
    dataset_terms = {
        "berkeley sbd dataset",
        "mnist dataset",
        "emnist dataset",
        "kmnist dataset",
    }
    out: List[str] = []
    for term in core_terms:
        key = re.sub(r"\s+", " ", str(term)).strip().lower()
        if not key:
            continue
        if key in dataset_terms:
            continue
        out.append(str(term))
    out.extend([str(t) for t in color_terms])
    out.extend([str(t) for t in direction_terms])
    return _normalize_vocab_terms(out)


def _semantic_kind_keys_for_class_names(class_names: Sequence[str]) -> List[str]:
    present = {
        re.sub(r"\s+", " ", str(name)).strip().lower()
        for name in class_names
        if str(name).strip()
    }
    alias_groups: Dict[str, List[str]] = {
        "none": ["none"],
        "noise": ["noise"],
        "signal": ["signal"],
        "object": ["object"],
        "mix": ["mixed noise and signal", "mix"],
        "mixed noise and signal": ["mixed noise and signal", "mix"],
        "white": ["white"],
        "black": ["black"],
        "blur damage": ["blur damage"],
        "noise damage": ["noise damage"],
        "dropout damage": ["dropout damage"],
        "quantization damage": ["quantization damage"],
        "stride skew damage": ["stride skew damage"],
        "gan image": ["gan image"],
        "regurgitated content": ["regurgitated content"],
        "berkeley sbd dataset": ["berkeley sbd dataset"],
        "mnist dataset": ["mnist dataset"],
        "emnist dataset": ["emnist dataset"],
        "kmnist dataset": ["kmnist dataset"],
    }
    out: List[str] = []
    for key, candidates in alias_groups.items():
        for cand in candidates:
            if str(cand).strip().lower() in present:
                out.append(str(key))
                break
    return out


def _find_semantic_term_index(class_names: Sequence[str], term: str) -> int:
    key = re.sub(r"\s+", " ", str(term)).strip().lower()
    if not key:
        return -1
    for i, name in enumerate(class_names):
        k = re.sub(r"\s+", " ", str(name)).strip().lower()
        if k == key:
            return int(i)
    return -1


def _semantic_term_index_map(class_names: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for i, name in enumerate(class_names):
        key = re.sub(r"\s+", " ", str(name)).strip().lower()
        if key and key not in out:
            out[key] = int(i)
    return out


def _semantic_damage_tags(term: str) -> List[str]:
    key = re.sub(r"\s+", " ", str(term)).strip().lower()
    if not key:
        return []
    lut: Dict[str, List[str]] = {
        "noise damage": ["noise damage", "noise", "mixed noise and signal", "signal"],
        "blur damage": ["blur damage", "signal"],
        "dropout damage": ["dropout damage", "signal"],
        "quantization damage": ["quantization damage", "signal"],
        "stride skew damage": ["stride skew damage", "signal"],
    }
    if key in lut:
        return _normalize_vocab_terms(lut[key])
    if key.endswith("damage"):
        return _normalize_vocab_terms([key, "signal"])
    return []


def _semantic_noise_family_terms() -> List[str]:
    return [
        "noise",
        "white noise",
        "pink noise",
        "brown noise",
        "red noise",
        "blue noise",
        "violet noise",
        "gray noise",
        "gaussian white noise",
        "uniform white noise",
    ]


def _semantic_noise_profile_key_from_term(term: str) -> str:
    key = re.sub(r"\s+", " ", str(term)).strip().lower()
    lut = {
        "noise": "uniform_white_noise",
        "white noise": "gaussian_white_noise",
        "uniform white noise": "uniform_white_noise",
        "gaussian white noise": "gaussian_white_noise",
        "pink noise": "pink_noise",
        "brown noise": "brown_noise",
        "red noise": "red_noise",
        "blue noise": "blue_noise",
        "violet noise": "violet_noise",
        "gray noise": "gray_noise",
        "grey noise": "gray_noise",
    }
    return str(lut.get(key, "")).strip().lower()


def _semantic_noise_profile_terms(profile_key: str) -> List[str]:
    key = re.sub(r"\s+", "_", str(profile_key)).strip().lower()
    lut: Dict[str, List[str]] = {
        "uniform_white_noise": ["noise", "white noise"],
        "gaussian_white_noise": ["noise", "white noise"],
        "pink_noise": ["noise", "pink noise"],
        "brown_noise": ["noise", "brown noise"],
        "red_noise": ["noise", "red noise"],
        "blue_noise": ["noise", "blue noise"],
        "violet_noise": ["noise", "violet noise"],
        "gray_noise": ["noise", "gray noise"],
    }
    if key not in lut:
        return ["noise"]
    return _normalize_vocab_terms(lut[key])


def _semantic_expand_inferred_tags(terms: Sequence[str]) -> List[str]:
    base = _normalize_vocab_terms([str(x) for x in list(terms)])
    out: List[str] = list(base)
    for term in base:
        key = re.sub(r"\s+", " ", str(term)).strip().lower()
        if not key:
            continue
        if key in ("mixed noise and signal", "mix"):
            out.extend(["mixed noise and signal", "noise", "signal"])
        if key == "noise":
            out.extend(["noise", "white noise"])
        if key.endswith("noise") and key != "noise":
            out.extend([key, "noise"])
        if key.endswith("damage"):
            out.extend(_semantic_damage_tags(key))
    return _normalize_vocab_terms(out)


def _semantic_noise_terms_from_spectrum_sample(sample: Any) -> List[str]:
    arr = np.asarray(sample, dtype=np.float32)
    if int(arr.size) <= 0:
        return ["noise"]
    if int(arr.ndim) == 3:
        if int(arr.shape[0]) in (1, 3, 4):
            gray = np.mean(np.asarray(arr[:3, ...], dtype=np.float32), axis=0)
        elif int(arr.shape[2]) in (1, 3, 4):
            gray = np.mean(np.asarray(arr[..., :3], dtype=np.float32), axis=2)
        else:
            gray = np.asarray(arr, dtype=np.float32).reshape(-1)
    else:
        gray = np.asarray(arr, dtype=np.float32)
    beta = 0.0
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
    noise_beta_lut = {
        "violet noise": -2.0,
        "blue noise": -1.0,
        "white noise": 0.0,
        "gray noise": 0.5,
        "pink noise": 1.0,
        "red noise": 1.8,
        "brown noise": 2.0,
    }
    best_term = "white noise"
    best_dist = float("inf")
    for name, target_beta in noise_beta_lut.items():
        d = abs(float(beta) - float(target_beta))
        if d < best_dist:
            best_dist = d
            best_term = str(name)
    return _normalize_vocab_terms(["noise", str(best_term)])


def _semantic_tags_for_symbol_term(term: str) -> List[str]:
    key = re.sub(r"\s+", " ", str(term)).strip().lower()
    if not key:
        return []
    berkeley_lut = getattr(_semantic_tags_for_symbol_term, "_berkeley_lut", None)
    if berkeley_lut is None:
        try:
            berkeley_lut = {str(x).strip().lower() for x in _default_berkeley_class_names()}
        except Exception:
            berkeley_lut = set()
        setattr(_semantic_tags_for_symbol_term, "_berkeley_lut", berkeley_lut)
    out: List[str] = []
    if key.startswith("digit "):
        out.extend(["mnist dataset", "signal"])
    elif key.startswith("letter "):
        out.extend(["emnist dataset", "signal"])
    elif key.startswith("pictogram "):
        out.extend(["kmnist dataset", "signal"])
    elif key in ("red", "green", "blue", "yellow", "cyan", "magenta", "brown", "gray", "grey"):
        out.extend([key, "signal"])
    elif key in ("front", "back", "left", "right", "top", "bottom"):
        out.extend([key, "object", "signal"])
    elif key in ("none",):
        out.extend(["none"])
    elif key in ("noise",):
        out.extend(_semantic_noise_profile_terms("uniform_white_noise"))
    elif key.endswith("noise"):
        profile_key = _semantic_noise_profile_key_from_term(key)
        if profile_key:
            out.extend(_semantic_noise_profile_terms(profile_key))
        else:
            out.extend([key, "noise"])
    elif key in ("mixed noise and signal", "mix"):
        out.extend(["mixed noise and signal", "signal"])
        out.extend(_semantic_noise_profile_terms("uniform_white_noise"))
    elif key in ("white", "black", "signal"):
        out.extend(["signal"])
    elif key == "spectrographic output":
        out.extend([key])
    elif key == "inverse spectrographic composition":
        out.extend([key])
    elif key == "pure tone construction":
        out.extend([key])
    elif key == "berkeley sbd dataset":
        out.extend(["berkeley sbd dataset", "object", "signal"])
    elif key in berkeley_lut:
        out.extend(["object", "berkeley sbd dataset", "signal"])
    elif key in ("mnist dataset", "emnist dataset", "kmnist dataset"):
        out.extend([key, "signal"])
    elif key == "regurgitated content":
        out.extend(["regurgitated content", "signal"])
    elif key == "gan image":
        out.extend(["gan image", "signal"])
    elif key.endswith("damage"):
        out.extend(_semantic_damage_tags(key))
    else:
        out.extend(["signal"])
    out.append(key)
    return _semantic_expand_inferred_tags(out)


def _merge_symbol_term_pools(
    pools: Sequence[Dict[str, List[np.ndarray]]],
    max_samples_per_term: int,
) -> Dict[str, List[np.ndarray]]:
    cap = max(1, int(max_samples_per_term))
    out: Dict[str, List[np.ndarray]] = {}
    for pool in pools:
        if not isinstance(pool, dict):
            continue
        for term, rows in pool.items():
            key = re.sub(r"\s+", " ", str(term)).strip().lower()
            if not key:
                continue
            dst = out.setdefault(key, [])
            for row in rows:
                if len(dst) >= cap:
                    break
                dst.append(np.asarray(row, dtype=np.float32))
    return out


def _build_symbol_term_origin_terms_map(
    official_pool: Dict[str, List[np.ndarray]],
    synthetic_pool: Dict[str, List[np.ndarray]],
    bootstrap_pool: Dict[str, List[np.ndarray]],
    bootstrap_origin_label: str,
) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    origin_txt = re.sub(r"\s+", " ", str(bootstrap_origin_label)).strip() or "internal bootstrap root vocab"

    def _append(term: str, tags: Sequence[str]):
        key = re.sub(r"\s+", " ", str(term)).strip().lower()
        if not key:
            return
        row = out.setdefault(key, [])
        seen = {str(x).strip().lower() for x in row}
        for t in _normalize_vocab_terms(tags):
            tl = str(t).strip().lower()
            if not tl or tl in seen:
                continue
            seen.add(tl)
            row.append(str(t))

    for term in official_pool.keys():
        key = re.sub(r"\s+", " ", str(term)).strip().lower()
        if not key:
            continue
        _append(
            key,
            _normalize_vocab_terms(
                ["official dataset pool", "symbol pool official"] + _semantic_tags_for_symbol_term(key)
            ),
        )
    for term in synthetic_pool.keys():
        key = re.sub(r"\s+", " ", str(term)).strip().lower()
        if not key:
            continue
        _append(
            key,
            _normalize_vocab_terms(
                ["synthetic semantic pool", "symbol pool synthetic"] + _semantic_tags_for_symbol_term(key)
            ),
        )
    for term in bootstrap_pool.keys():
        key = re.sub(r"\s+", " ", str(term)).strip().lower()
        if not key:
            continue
        _append(
            key,
            _normalize_vocab_terms(
                [
                    "symbol pool bootstrap",
                    "internal bootstrap material",
                    origin_txt,
                    f"{origin_txt}::{key}",
                ]
                + _semantic_tags_for_symbol_term(key)
            ),
        )
    return out


def _image_any_to_rgb_chw01(img: Any, image_size: int) -> np.ndarray:
    size = max(8, int(image_size))
    arr = None
    if torch.is_tensor(img):
        t = img.detach().to(torch.float32).cpu()
        if int(t.ndim) == 3 and int(t.shape[0]) in (1, 3, 4):
            t = t.unsqueeze(0)
            t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
            t = torch.clamp(t, 0.0, 1.0).squeeze(0)
            if int(t.shape[0]) == 1:
                t = t.repeat(3, 1, 1)
            elif int(t.shape[0]) > 3:
                t = t[:3, :, :]
            return t.numpy().astype(np.float32, copy=False)
        if int(t.ndim) == 2:
            arr = t.numpy().astype(np.float32, copy=False)
    if arr is None:
        if hasattr(img, "convert"):
            arr = np.asarray(img.convert("L"), dtype=np.float32)
        else:
            arr = np.asarray(img, dtype=np.float32)
    if int(arr.ndim) == 2:
        if float(np.max(arr)) > 1.0:
            arr = arr / 255.0
        arr = np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)
        t = torch.from_numpy(arr[None, None, ...]).to(torch.float32)
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
        g = torch.clamp(t[0, 0], 0.0, 1.0).cpu().numpy().astype(np.float32, copy=False)
        return np.repeat(g[None, :, :], 3, axis=0).astype(np.float32, copy=False)

    if int(arr.ndim) == 3:
        # Preserve true RGB when possible; support both CHW and HWC inputs.
        if int(arr.shape[0]) in (1, 3, 4) and int(arr.shape[1]) > 4 and int(arr.shape[2]) > 4:
            chw = np.asarray(arr[:3, :, :], dtype=np.float32)
            if int(chw.shape[0]) == 1:
                chw = np.repeat(chw, 3, axis=0)
        elif int(arr.shape[2]) in (1, 3, 4) and int(arr.shape[0]) > 4 and int(arr.shape[1]) > 4:
            hwc = np.asarray(arr[:, :, :3], dtype=np.float32)
            if int(hwc.shape[2]) == 1:
                hwc = np.repeat(hwc, 3, axis=2)
            chw = np.transpose(hwc, (2, 0, 1)).astype(np.float32, copy=False)
        else:
            raise RuntimeError(f"Unsupported 3D symbol image shape for conversion: {tuple(arr.shape)}")

        vmin = float(np.min(chw))
        vmax = float(np.max(chw))
        if vmax > 1.0:
            chw = chw / 255.0
        elif vmin < 0.0 and vmax <= 1.0:
            # Common generator range [-1, 1] -> [0, 1]
            chw = (chw + 1.0) * 0.5
        chw = np.clip(chw, 0.0, 1.0).astype(np.float32, copy=False)
        t = torch.from_numpy(chw[None, ...]).to(torch.float32)
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
        rgb = torch.clamp(t[0], 0.0, 1.0).cpu().numpy().astype(np.float32, copy=False)
        if int(rgb.shape[0]) == 1:
            rgb = np.repeat(rgb, 3, axis=0)
        elif int(rgb.shape[0]) > 3:
            rgb = rgb[:3, :, :]
        return np.asarray(rgb, dtype=np.float32)

    raise RuntimeError(f"Unsupported symbol image shape for conversion: {tuple(arr.shape)}")


def _semantic_tonal_tags_from_image(
    image: Any,
    image_size: int = 64,
    low_threshold: float = 0.15,
    high_threshold: float = 0.85,
    coverage_threshold: float = 0.25,
    gray_mean_tolerance: float = 0.12,
    gray_std_threshold: float = 0.12,
) -> List[str]:
    try:
        rgb = _image_any_to_rgb_chw01(image, image_size=max(8, int(image_size)))
    except Exception:
        return []
    arr = np.asarray(rgb, dtype=np.float32)
    if int(arr.ndim) != 3 or int(arr.shape[0]) <= 0:
        return []
    gray = np.clip(np.mean(arr[:3, :, :], axis=0), 0.0, 1.0).astype(np.float32, copy=False)
    flat = gray.reshape(-1)
    if int(flat.size) <= 0:
        return []
    lo_t = float(low_threshold)
    hi_t = float(high_threshold)
    cov_t = float(coverage_threshold)
    tags: List[str] = []
    low_frac = float(np.mean(flat <= lo_t))
    high_frac = float(np.mean(flat >= hi_t))
    mean_v = float(np.mean(flat))
    std_v = float(np.std(flat))
    if low_frac >= cov_t:
        tags.append("black")
    if high_frac >= cov_t:
        tags.append("white")
    if (abs(mean_v - 0.5) <= float(gray_mean_tolerance)) and (std_v <= float(gray_std_threshold)):
        tags.extend(["gray", "grey"])
    return _normalize_vocab_terms(tags)


def _semantic_terms_with_tonal_tags(
    terms: Sequence[str],
    image: Any,
    image_size: int = 64,
) -> List[str]:
    base = _normalize_vocab_terms([str(x) for x in list(terms)])
    tones = _semantic_tonal_tags_from_image(image=image, image_size=int(image_size))
    return _normalize_vocab_terms(list(base) + list(tones))


def _semantic_enrich_generated_terms_with_noise_spectrum(
    term_key: str,
    terms: Sequence[str],
    image: Any,
) -> List[str]:
    key = re.sub(r"\s+", " ", str(term_key)).strip().lower()
    out = _normalize_vocab_terms([str(x) for x in list(terms)])
    noise_related = bool(
        key == "noise"
        or key.endswith("noise")
        or key in ("mix", "mixed noise and signal", "noise damage")
    )
    if not noise_related:
        return out
    out.extend(_semantic_noise_terms_from_spectrum_sample(image))
    return _normalize_vocab_terms(out)


def _semantic_active_target_stats(
    rows: Sequence[Any],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "rows": 0,
        "min": 0,
        "mean": 0.0,
        "p50": 0.0,
        "max": 0,
    }
    if len(rows) <= 0:
        return out
    t = float(threshold)
    counts: List[int] = []
    for row in rows:
        arr = np.asarray(row, dtype=np.float32).reshape(-1)
        if int(arr.size) <= 0:
            counts.append(0)
            continue
        counts.append(int(np.count_nonzero(arr >= t)))
    if len(counts) <= 0:
        return out
    c_np = np.asarray(counts, dtype=np.int32)
    out["rows"] = int(c_np.size)
    out["min"] = int(c_np.min())
    out["mean"] = float(np.mean(c_np))
    out["p50"] = float(np.percentile(c_np, 50))
    out["max"] = int(c_np.max())
    return out


def _label_knockout_row_np(
    row: Any,
    rng: np.random.Generator,
    prob: float,
    min_keep: int,
    max_drop_frac: float,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, int]:
    arr = np.asarray(row, dtype=np.float32).reshape(-1).copy()
    p = float(max(0.0, min(1.0, float(prob))))
    keep_min = max(0, int(min_keep))
    frac = float(max(0.0, min(1.0, float(max_drop_frac))))
    if int(arr.size) <= 0 or p <= 0.0:
        return arr, 0
    if float(rng.random()) >= p:
        return arr, 0
    active = np.where(np.asarray(arr, dtype=np.float32) >= float(threshold))[0].astype(np.int64)
    n_active = int(active.size)
    if int(n_active) <= int(keep_min):
        return arr, 0
    max_drop_by_frac = int(math.floor(float(n_active) * float(frac)))
    max_drop = int(n_active - int(keep_min))
    if int(max_drop_by_frac) > 0:
        max_drop = min(int(max_drop), int(max_drop_by_frac))
    max_drop = max(1, int(max_drop))
    drop_n = int(rng.integers(1, int(max_drop) + 1))
    pick = rng.choice(active, size=int(drop_n), replace=False).astype(np.int64)
    arr[pick] = 0.0
    return arr.astype(np.float32, copy=False), int(drop_n)


def _label_knockout_rows_np(
    rows: Sequence[Any],
    prob: float,
    seed: int,
    min_keep: int,
    max_drop_frac: float,
    threshold: float = 0.5,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    rng = np.random.default_rng(max(0, int(seed)))
    out_rows: List[np.ndarray] = []
    rows_total = int(len(rows))
    rows_applied = 0
    labels_dropped = 0
    p = float(max(0.0, min(1.0, float(prob))))
    for row in rows:
        out_row, dropped = _label_knockout_row_np(
            row=row,
            rng=rng,
            prob=float(p),
            min_keep=int(min_keep),
            max_drop_frac=float(max_drop_frac),
            threshold=float(threshold),
        )
        out_rows.append(np.asarray(out_row, dtype=np.float32).reshape(-1))
        if int(dropped) > 0:
            rows_applied += 1
            labels_dropped += int(dropped)
    info = {
        "enabled": bool(float(p) > 0.0),
        "prob": float(p),
        "rows_total": int(rows_total),
        "rows_applied": int(rows_applied),
        "labels_dropped": int(labels_dropped),
        "min_keep": int(max(0, int(min_keep))),
        "max_drop_frac": float(max(0.0, min(1.0, float(max_drop_frac)))),
    }
    return out_rows, info


def _label_knockout_optional_rows_np(
    rows: Sequence[Any],
    prob: float,
    seed: int,
    min_keep: int,
    max_drop_frac: float,
    threshold: float = 0.5,
) -> Tuple[List[Any], Dict[str, Any]]:
    real_rows = [np.asarray(r, dtype=np.float32).reshape(-1) for r in rows if r is not None]
    knocked_rows, info = _label_knockout_rows_np(
        rows=real_rows,
        prob=float(prob),
        seed=int(seed),
        min_keep=int(min_keep),
        max_drop_frac=float(max_drop_frac),
        threshold=float(threshold),
    )
    out: List[Any] = []
    real_idx = 0
    for r in rows:
        if r is None:
            out.append(None)
            continue
        out.append(np.asarray(knocked_rows[int(real_idx)], dtype=np.float32).reshape(-1))
        real_idx += 1
    info = dict(info)
    info["rows_with_targets"] = int(len(real_rows))
    return out, info


def _label_knockout_tensor_batch(
    yb: torch.Tensor,
    rng: np.random.Generator,
    prob: float,
    min_keep: int,
    max_drop_frac: float,
    threshold: float = 0.5,
) -> Tuple[torch.Tensor, int, int]:
    if (not torch.is_tensor(yb)) or int(getattr(yb, "ndim", 0)) != 2 or int(yb.shape[0]) <= 0:
        return yb, 0, 0
    p = float(max(0.0, min(1.0, float(prob))))
    if float(p) <= 0.0:
        return yb, 0, 0
    keep_min = max(0, int(min_keep))
    frac = float(max(0.0, min(1.0, float(max_drop_frac))))
    out = yb.to(dtype=torch.float32).clone()
    rows_applied = 0
    labels_dropped = 0
    for bi in range(int(out.shape[0])):
        if float(rng.random()) >= float(p):
            continue
        row = out[int(bi)]
        active = torch.nonzero(row >= float(threshold), as_tuple=False).reshape(-1)
        n_active = int(active.numel())
        if int(n_active) <= int(keep_min):
            continue
        max_drop_by_frac = int(math.floor(float(n_active) * float(frac)))
        max_drop = int(n_active - int(keep_min))
        if int(max_drop_by_frac) > 0:
            max_drop = min(int(max_drop), int(max_drop_by_frac))
        max_drop = max(1, int(max_drop))
        drop_n = int(rng.integers(1, int(max_drop) + 1))
        pos = rng.choice(np.arange(int(n_active), dtype=np.int64), size=int(drop_n), replace=False).astype(np.int64)
        pos_t = torch.from_numpy(pos).to(device=active.device, dtype=torch.long)
        drop_idx = active.index_select(0, pos_t)
        row[drop_idx] = 0.0
        rows_applied += 1
        labels_dropped += int(drop_n)
    return out.to(dtype=yb.dtype), int(rows_applied), int(labels_dropped)


def _build_auto_symbol_term_pool(
    data_root: str,
    image_size: int,
    seed: int,
    max_samples_per_term: int,
    include_digits: bool = True,
    include_letters: bool = True,
    include_pictograms: bool = True,
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, Any]]:
    pool: Dict[str, List[np.ndarray]] = {}
    info: Dict[str, Any] = {
        "enabled": True,
        "available_terms": 0,
        "samples": 0,
        "digits_terms": 0,
        "letters_terms": 0,
        "pictogram_terms": 0,
        "reason": "",
    }
    max_k = max(1, int(max_samples_per_term))
    root = Path(str(data_root).strip() or "toys_to_survive_development/data/semantic_symbol_pool")
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(seed))
    try:
        from torchvision import datasets as tvds
    except Exception as e:
        info["enabled"] = False
        info["reason"] = f"torchvision_unavailable:{type(e).__name__}"
        return pool, info

    def _prefer_torchvision_mirror(dataset_cls: Any, preferred_url: str) -> None:
        try:
            current = [str(x).strip() for x in list(getattr(dataset_cls, "mirrors", [])) if str(x).strip()]
            pref = str(preferred_url).strip()
            if len(current) <= 0 or not pref:
                return
            ordered = [pref] + [u for u in current if str(u).strip() != pref]
            if ordered != current:
                dataset_cls.mirrors = ordered
        except Exception:
            return

    # Prefer the maintained S3 MNIST mirror over yann.lecun.com to avoid predictable 404 fallbacks.
    _prefer_torchvision_mirror(tvds.MNIST, "https://ossci-datasets.s3.amazonaws.com/mnist/")

    def _push(term: str, img_any: Any):
        key = re.sub(r"\s+", " ", str(term)).strip().lower()
        if not key:
            return
        rows = pool.setdefault(key, [])
        if len(rows) >= max_k:
            return
        rows.append(_image_any_to_rgb_chw01(img_any, image_size=max(8, int(image_size))))

    def _sample_dataset(dataset: Any, term_fn: Callable[[Any], str], dataset_term: str):
        n = int(len(dataset))
        if n <= 0:
            return
        order = np.arange(n, dtype=np.int64)
        rng.shuffle(order)
        for idx in order.tolist():
            try:
                x, y = dataset[int(idx)]
                term = str(term_fn(y)).strip()
                if not term:
                    continue
                _push(term, x)
                ds_term = re.sub(r"\s+", " ", str(dataset_term)).strip().lower()
                if ds_term:
                    _push(ds_term, x)
                _push("signal", x)
            except Exception:
                continue

    if bool(include_digits):
        try:
            ds_mnist = tvds.MNIST(root=str(root), train=True, download=True)
            _sample_dataset(ds_mnist, term_fn=lambda y: f"digit {int(y)}", dataset_term="mnist dataset")
        except Exception:
            pass
    if bool(include_letters):
        try:
            ds_letters = tvds.EMNIST(root=str(root), split="letters", train=True, download=True)
            _sample_dataset(
                ds_letters,
                term_fn=lambda y: f"letter {chr(ord('a') + max(0, min(25, int(y) - 1)))}",
                dataset_term="emnist dataset",
            )
        except Exception:
            pass
    if bool(include_pictograms):
        try:
            ds_kmnist = tvds.KMNIST(root=str(root), train=True, download=True)
            _sample_dataset(ds_kmnist, term_fn=lambda y: f"pictogram {int(y)}", dataset_term="kmnist dataset")
        except Exception:
            pass

    terms = list(pool.keys())
    info["available_terms"] = int(len(terms))
    info["samples"] = int(sum(len(v) for v in pool.values()))
    info["digits_terms"] = int(sum(1 for k in terms if str(k).startswith("digit ")))
    info["letters_terms"] = int(sum(1 for k in terms if str(k).startswith("letter ")))
    info["pictogram_terms"] = int(sum(1 for k in terms if str(k).startswith("pictogram ")))
    return pool, info


def _build_synthetic_semantic_symbol_pool(
    image_size: int,
    seed: int,
    max_samples_per_term: int,
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, Any]]:
    size = max(8, int(image_size))
    cap = max(1, int(max_samples_per_term))
    rng = np.random.default_rng(int(seed))
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    x01 = (xx / float(max(1, size - 1))).astype(np.float32, copy=False)
    y01 = (yy / float(max(1, size - 1))).astype(np.float32, copy=False)

    def _signal_pattern(phase: float = 0.0) -> np.ndarray:
        f0 = float(rng.uniform(2.0, 7.0))
        f1 = float(rng.uniform(2.0, 6.0))
        base = 0.5 + (0.5 * np.sin((2.0 * math.pi * ((f0 * x01) + (f1 * y01))) + float(phase)))
        env = 0.55 + (0.45 * np.sin((2.0 * math.pi * (x01 * 0.75)) + float(phase * 0.5)))
        return np.clip(base * env, 0.0, 1.0).astype(np.float32, copy=False)

    def _circle_object() -> np.ndarray:
        cx = float(size) * 0.5
        cy = float(size) * 0.5
        r = float(size) * 0.30
        d = np.sqrt(((xx - cx) ** 2) + ((yy - cy) ** 2))
        mask = (d <= r).astype(np.float32)
        bg = np.clip(0.15 + (0.10 * _signal_pattern(phase=0.9)), 0.0, 1.0)
        fg = np.clip(0.75 + (0.20 * _signal_pattern(phase=1.7)), 0.0, 1.0)
        return np.clip((bg * (1.0 - mask)) + (fg * mask), 0.0, 1.0).astype(np.float32, copy=False)

    def _box_blur(g: np.ndarray, k: int = 5) -> np.ndarray:
        kk = max(3, int(k) | 1)
        t = torch.from_numpy(np.asarray(g, dtype=np.float32)[None, None, ...])
        t = F.avg_pool2d(t, kernel_size=int(kk), stride=1, padding=int(kk // 2))
        return np.clip(t[0, 0].cpu().numpy().astype(np.float32, copy=False), 0.0, 1.0)

    def _quantize(g: np.ndarray, levels: int) -> np.ndarray:
        lv = max(2, int(levels))
        return np.clip(np.round(np.asarray(g, dtype=np.float32) * float(lv - 1)) / float(lv - 1), 0.0, 1.0).astype(
            np.float32, copy=False
        )

    def _to_rgb(g: np.ndarray) -> np.ndarray:
        gg = np.clip(np.asarray(g, dtype=np.float32), 0.0, 1.0)
        return np.repeat(gg[None, :, :], 3, axis=0).astype(np.float32, copy=False)

    def _term_image(term: str, sample_idx: int) -> np.ndarray:
        key = re.sub(r"\s+", " ", str(term)).strip().lower()
        phase = float(sample_idx) * 0.77
        if key == "white":
            g = np.ones((size, size), dtype=np.float32)
        elif key == "black":
            g = np.zeros((size, size), dtype=np.float32)
        elif key == "none":
            g = np.full((size, size), 0.50, dtype=np.float32)
        elif key == "noise":
            g = rng.random((size, size), dtype=np.float32)
        elif key == "signal":
            g = _signal_pattern(phase=phase)
        elif key == "object":
            g = _circle_object()
        elif key in ("mixed noise and signal", "mix"):
            g = np.clip((0.55 * _signal_pattern(phase=phase)) + (0.45 * rng.random((size, size), dtype=np.float32)), 0.0, 1.0)
        elif key == "blur damage":
            g = _box_blur(_signal_pattern(phase=phase), k=7)
        elif key == "noise damage":
            g = np.clip(_signal_pattern(phase=phase) + (0.20 * rng.standard_normal((size, size), dtype=np.float32)), 0.0, 1.0)
        elif key == "dropout damage":
            src = _signal_pattern(phase=phase)
            drop = (rng.random((size, size), dtype=np.float32) > 0.20).astype(np.float32, copy=False)
            g = np.clip(src * drop, 0.0, 1.0)
        elif key == "quantization damage":
            g = _quantize(_signal_pattern(phase=phase), levels=4)
        elif key == "stride skew damage":
            src = _signal_pattern(phase=phase)
            out = np.array(src, dtype=np.float32, copy=True)
            out[1::2, :] = np.roll(out[1::2, :], shift=3, axis=1)
            g = np.clip(out, 0.0, 1.0)
        elif key == "berkeley sbd dataset":
            g = np.clip((0.65 * _circle_object()) + (0.35 * _signal_pattern(phase=phase)), 0.0, 1.0)
        elif key == "mnist dataset":
            g = np.clip((0.75 * (x01 > 0.45).astype(np.float32)) + (0.25 * _signal_pattern(phase=phase)), 0.0, 1.0)
        elif key == "emnist dataset":
            g = np.clip((0.70 * (np.abs(x01 - y01) < 0.12).astype(np.float32)) + (0.30 * _signal_pattern(phase=phase)), 0.0, 1.0)
        elif key == "kmnist dataset":
            g = np.clip((0.70 * (np.abs((x01 + y01) - 1.0) < 0.12).astype(np.float32)) + (0.30 * _signal_pattern(phase=phase)), 0.0, 1.0)
        elif key == "regurgitated content":
            g = np.clip((0.60 * _signal_pattern(phase=phase)) + (0.40 * _signal_pattern(phase=phase + 1.8)), 0.0, 1.0)
        elif key == "gan image":
            checker = ((np.floor(xx / 8.0) + np.floor(yy / 8.0)) % 2.0).astype(np.float32, copy=False)
            g = np.clip((0.65 * checker) + (0.35 * _signal_pattern(phase=phase)), 0.0, 1.0)
        else:
            g = _signal_pattern(phase=phase)
        return _to_rgb(g)

    terms = _normalize_vocab_terms(_default_semantic_core_terms())
    pool: Dict[str, List[np.ndarray]] = {}
    for term in terms:
        key = re.sub(r"\s+", " ", str(term)).strip().lower()
        if not key:
            continue
        rows = pool.setdefault(key, [])
        for j in range(int(cap)):
            rows.append(_term_image(term=key, sample_idx=int(j)))

    info = {
        "enabled": True,
        "available_terms": int(len(pool)),
        "samples": int(sum(len(v) for v in pool.values())),
        "reason": "synthetic_core_semantic_pool",
    }
    return pool, info


def _build_internal_bootstrap_symbol_pool(
    data_root: str,
    image_size: int,
    seed: int,
    max_samples_per_term: int,
    origin_label: str = "internal bootstrap root vocab",
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, Any]]:
    size = max(8, int(image_size))
    # Keep multiple exemplars per term so gestation/gate coverage does not collapse to one image per primitive.
    cap = max(4, int(max_samples_per_term))
    rng = np.random.default_rng(int(seed))
    root = Path(str(data_root).strip() or "toys_to_survive_development/data/semantic_symbol_pool")
    out_dir = root / "internal_bootstrap_root_vocab"
    out_dir.mkdir(parents=True, exist_ok=True)

    pool: Dict[str, List[np.ndarray]] = {}
    info: Dict[str, Any] = {
        "enabled": True,
        "available_terms": 0,
        "samples": 0,
        "root_dir": str(out_dir),
        "manifest": str(out_dir / "manifest.json"),
        "origin_label": str(origin_label).strip(),
        "reason": "internal_bootstrap_root_vocab",
    }

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    x01 = (xx / float(max(1, size - 1))).astype(np.float32, copy=False)
    y01 = (yy / float(max(1, size - 1))).astype(np.float32, copy=False)

    def _slug(term: str) -> str:
        txt = re.sub(r"\s+", "_", str(term).strip().lower())
        txt = re.sub(r"[^a-z0-9_]+", "", txt)
        return txt or "term"

    def _to_rgb(g: np.ndarray) -> np.ndarray:
        gg = np.clip(np.asarray(g, dtype=np.float32), 0.0, 1.0)
        return np.repeat(gg[None, :, :], 3, axis=0).astype(np.float32, copy=False)

    def _resize_gray(g: np.ndarray) -> np.ndarray:
        arr = np.asarray(g, dtype=np.float32)
        if int(arr.ndim) != 2:
            raise RuntimeError(f"Expected 2D bootstrap image, got shape={tuple(arr.shape)}")
        t = torch.from_numpy(arr[None, None, ...]).to(torch.float32)
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
        return np.clip(t[0, 0].cpu().numpy().astype(np.float32, copy=False), 0.0, 1.0)

    def _write_sample(term: str, sample_idx: int, gray: np.ndarray):
        term_dir = out_dir / _slug(term)
        term_dir.mkdir(parents=True, exist_ok=True)
        npy_path = term_dir / f"sample_{int(sample_idx):03d}.npy"
        np.save(str(npy_path), np.asarray(gray, dtype=np.float32))
        png_path = term_dir / f"sample_{int(sample_idx):03d}.png"
        try:
            from PIL import Image

            u8 = np.round(np.clip(np.asarray(gray, dtype=np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)
            Image.fromarray(u8, mode="L").save(str(png_path))
        except Exception:
            pass

    def _push(term: str, img: np.ndarray):
        key = re.sub(r"\s+", " ", str(term)).strip().lower()
        if not key:
            return
        rows = pool.setdefault(key, [])
        if len(rows) >= cap:
            return
        gray = _resize_gray(img)
        gg = np.clip(np.asarray(gray, dtype=np.float32), 0.0, 1.0)
        if key == "red":
            rgb = np.stack([gg, 0.10 * gg, 0.10 * gg], axis=0).astype(np.float32, copy=False)
        elif key == "green":
            rgb = np.stack([0.10 * gg, gg, 0.10 * gg], axis=0).astype(np.float32, copy=False)
        elif key == "blue":
            rgb = np.stack([0.10 * gg, 0.10 * gg, gg], axis=0).astype(np.float32, copy=False)
        elif key == "yellow":
            rgb = np.stack([gg, gg, 0.12 * gg], axis=0).astype(np.float32, copy=False)
        elif key == "cyan":
            rgb = np.stack([0.12 * gg, gg, gg], axis=0).astype(np.float32, copy=False)
        elif key == "magenta":
            rgb = np.stack([gg, 0.12 * gg, gg], axis=0).astype(np.float32, copy=False)
        elif key == "brown":
            rgb = np.stack([0.70 * gg, 0.40 * gg, 0.20 * gg], axis=0).astype(np.float32, copy=False)
        else:
            rgb = _to_rgb(gg)
        rows.append(np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0).astype(np.float32, copy=False))

    def _signal_pattern(phase: float = 0.0) -> np.ndarray:
        fx = float(rng.uniform(2.0, 7.0))
        fy = float(rng.uniform(2.0, 6.0))
        g = 0.5 + (0.5 * np.sin((2.0 * math.pi * ((fx * x01) + (fy * y01))) + float(phase)))
        return np.clip(g, 0.0, 1.0).astype(np.float32, copy=False)

    def _circle_object() -> np.ndarray:
        cx = float(size) * 0.5
        cy = float(size) * 0.5
        r = float(size) * 0.30
        d = np.sqrt(((xx - cx) ** 2) + ((yy - cy) ** 2))
        mask = (d <= r).astype(np.float32, copy=False)
        bg = np.clip(0.14 + (0.10 * _signal_pattern(phase=0.9)), 0.0, 1.0)
        fg = np.clip(0.75 + (0.20 * _signal_pattern(phase=1.7)), 0.0, 1.0)
        return np.clip((bg * (1.0 - mask)) + (fg * mask), 0.0, 1.0).astype(np.float32, copy=False)

    def _box_blur(g: np.ndarray, k: int = 5) -> np.ndarray:
        kk = max(3, int(k) | 1)
        t = torch.from_numpy(np.asarray(g, dtype=np.float32)[None, None, ...])
        t = F.avg_pool2d(t, kernel_size=int(kk), stride=1, padding=int(kk // 2))
        return np.clip(t[0, 0].cpu().numpy().astype(np.float32, copy=False), 0.0, 1.0)

    def _quantize(g: np.ndarray, levels: int) -> np.ndarray:
        lv = max(2, int(levels))
        return np.clip(np.round(np.asarray(g, dtype=np.float32) * float(lv - 1)) / float(lv - 1), 0.0, 1.0).astype(
            np.float32, copy=False
        )

    def _stft_mag(signal: np.ndarray) -> np.ndarray:
        wave = np.asarray(signal, dtype=np.float32).reshape(-1)
        n_fft = max(32, int(size * 2))
        hop = max(4, int(n_fft // 8))
        if int(wave.size) < int(n_fft):
            pad = np.zeros((int(n_fft - int(wave.size)),), dtype=np.float32)
            wave = np.concatenate([wave, pad], axis=0)
        win = np.hanning(int(n_fft)).astype(np.float32, copy=False)
        cols: List[np.ndarray] = []
        for start in range(0, int(wave.size) - int(n_fft) + 1, int(hop)):
            seg = wave[start : start + int(n_fft)]
            mag = np.abs(np.fft.rfft((seg * win).astype(np.float32, copy=False))).astype(np.float32, copy=False)
            cols.append(mag)
        if len(cols) <= 0:
            cols = [np.zeros((int(n_fft // 2) + 1,), dtype=np.float32)]
        spec = np.stack(cols, axis=1).astype(np.float32, copy=False)
        spec = np.log1p(np.maximum(spec, 0.0))
        spec = spec - float(np.min(spec))
        den = float(np.max(spec))
        if den > 1e-8:
            spec = spec / den
        return _resize_gray(spec)

    def _pure_tone_wave(sample_idx: int) -> np.ndarray:
        n = max(1024, int(size * 18))
        sr = 16000.0
        t = (np.arange(n, dtype=np.float32) / float(sr)).astype(np.float32, copy=False)
        f0 = float(rng.uniform(120.0, 1400.0))
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        wave = np.sin((2.0 * math.pi * f0 * t) + phase)
        wave += 0.35 * np.sin((2.0 * math.pi * (f0 * 0.5) * t) + (phase * 0.4))
        if int(sample_idx) % 2 == 0:
            wave += 0.25 * np.sin((2.0 * math.pi * (f0 * 1.5) * t) + (phase * 1.2))
        env = np.hanning(n).astype(np.float32, copy=False)
        wave = wave.astype(np.float32, copy=False) * env
        peak = float(np.max(np.abs(wave))) if int(wave.size) > 0 else 0.0
        if peak > 1e-8:
            wave = wave / peak
        return wave.astype(np.float32, copy=False)

    def _spectrographic_image(sample_idx: int) -> np.ndarray:
        wave = _pure_tone_wave(sample_idx=int(sample_idx))
        return _stft_mag(wave)

    def _inverse_spectrographic_image(sample_idx: int) -> np.ndarray:
        spec = _spectrographic_image(sample_idx=int(sample_idx))
        inv = np.clip(1.0 - np.flipud(spec), 0.0, 1.0).astype(np.float32, copy=False)
        return np.clip((0.82 * inv) + (0.18 * rng.random((size, size), dtype=np.float32)), 0.0, 1.0)

    def _pure_tone_image(sample_idx: int) -> np.ndarray:
        phase = float(sample_idx) * 0.61
        fx = float(rng.uniform(1.5, 6.5))
        fy = float(rng.uniform(0.4, 2.0))
        g = 0.5 + (0.5 * np.sin((2.0 * math.pi * ((fx * x01) + (fy * y01))) + phase))
        return np.clip(g, 0.0, 1.0).astype(np.float32, copy=False)

    def _normalize_wave(wave: np.ndarray) -> np.ndarray:
        x = np.asarray(wave, dtype=np.float32).reshape(-1)
        if int(x.size) <= 0:
            return np.zeros((1024,), dtype=np.float32)
        x = x - float(np.mean(x))
        peak = float(np.max(np.abs(x)))
        if peak > 1e-8:
            x = x / peak
        return x.astype(np.float32, copy=False)

    def _white_noise_wave(sample_idx: int, distribution: str) -> np.ndarray:
        n = max(2048, int(size * 24))
        dist_key = str(distribution).strip().lower()
        if dist_key == "uniform":
            wave = rng.uniform(-1.0, 1.0, size=(n,)).astype(np.float32, copy=False)
        else:
            wave = rng.standard_normal((n,)).astype(np.float32, copy=False)
        # Keep deterministic sample-index variation while preserving class-level spectrum semantics.
        if int(sample_idx) % 2 == 0:
            wave = np.roll(wave, shift=int(17 + (sample_idx % 29)))
        return _normalize_wave(wave)

    def _colored_noise_wave(sample_idx: int, beta: float, distribution: str = "gaussian") -> np.ndarray:
        n = max(2048, int(size * 24))
        base = _white_noise_wave(sample_idx=int(sample_idx), distribution=str(distribution))
        spec = np.fft.rfft(base).astype(np.complex64, copy=False)
        freqs = np.fft.rfftfreq(int(n), d=(1.0 / 16000.0)).astype(np.float32, copy=False)
        denom = np.maximum(freqs, 1.0).astype(np.float32, copy=False)
        shape = np.power(denom, -0.5 * float(beta)).astype(np.float32, copy=False)
        shape[0] = 0.0
        shaped = spec * shape.astype(np.complex64, copy=False)
        wave = np.fft.irfft(shaped, n=int(n)).astype(np.float32, copy=False)
        return _normalize_wave(wave)

    def _spectral_noise_image(sample_idx: int, beta: float, distribution: str = "gaussian") -> np.ndarray:
        wave = _colored_noise_wave(sample_idx=int(sample_idx), beta=float(beta), distribution=str(distribution))
        return _stft_mag(wave)

    def _symbol_token_pattern(token: str, sample_idx: int, mode: str) -> np.ndarray:
        digest = hashlib.sha256(f"{token}|{int(sample_idx)}|{mode}".encode("utf-8")).digest()
        fx = 1.0 + (float(int(digest[0])) / 255.0) * 7.0
        fy = 1.0 + (float(int(digest[1])) / 255.0) * 7.0
        phase = (float(int(digest[2])) / 255.0) * (2.0 * math.pi)
        wave = 0.5 + (
            0.5
            * np.sin(
                (2.0 * math.pi * ((fx * x01) + (fy * y01)))
                + phase
                + (float(sample_idx) * 0.31)
            )
        )
        thresh = 0.46 + (float(int(digest[3])) / 255.0) * 0.12
        mask = (wave > thresh).astype(np.float32, copy=False)
        if str(mode).strip().lower() == "digit":
            cx = 0.5 + (((float(int(digest[4])) / 255.0) - 0.5) * 0.16)
            cy = 0.5 + (((float(int(digest[5])) / 255.0) - 0.5) * 0.16)
            rr = np.sqrt(((x01 - cx) ** 2) + ((y01 - cy) ** 2))
            ring = np.exp(-((rr - 0.28) ** 2) / 0.012).astype(np.float32, copy=False)
            out = (0.72 * mask) + (0.28 * ring)
        elif str(mode).strip().lower() == "letter":
            diag = (np.abs((x01 - y01) - ((float(int(digest[6])) / 255.0) - 0.5) * 0.35) < 0.08).astype(
                np.float32, copy=False
            )
            out = (0.66 * mask) + (0.34 * diag)
        else:
            cell = max(4, int(size // 14))
            checker = ((np.floor(xx / float(cell)) + np.floor(yy / float(cell))) % 2.0).astype(np.float32, copy=False)
            out = (0.62 * mask) + (0.38 * checker)
        return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

    def _mix_signal_with_noise_average(sig: np.ndarray, noi: np.ndarray, alpha: float) -> np.ndarray:
        a = max(0.0, min(1.0, float(alpha)))
        return np.clip((a * np.asarray(sig, dtype=np.float32)) + ((1.0 - a) * np.asarray(noi, dtype=np.float32)), 0.0, 1.0).astype(
            np.float32, copy=False
        )

    def _mix_signal_with_noise_pcm(sig: np.ndarray, noi: np.ndarray, noise_bits: int) -> np.ndarray:
        # Inner-bit PCM packing: keep high-order signal bits, inject low-order noise payload bits.
        s = np.clip(np.asarray(sig, dtype=np.float32), 0.0, 1.0)
        n = np.clip(np.asarray(noi, dtype=np.float32), 0.0, 1.0)
        bits = max(1, min(8, int(noise_bits)))
        s_u16 = np.round(s * 65535.0).astype(np.uint16, copy=False)
        n_u16 = np.round(n * 65535.0).astype(np.uint16, copy=False)
        payload = ((n_u16 >> int(16 - bits)) & np.uint16((1 << bits) - 1)).astype(np.uint16, copy=False)
        keep_mask = np.uint16(0xFFFF ^ ((1 << bits) - 1))
        packed = ((s_u16 & keep_mask) | payload).astype(np.uint16, copy=False)
        return np.clip(packed.astype(np.float32) / 65535.0, 0.0, 1.0).astype(np.float32, copy=False)

    berkeley_term_lc: set = set()

    def _term_gray(term: str, sample_idx: int) -> np.ndarray:
        key = re.sub(r"\s+", " ", str(term)).strip().lower()
        phase = float(sample_idx) * 0.77
        color_patterns: Dict[str, np.ndarray] = {
            "red": np.clip(x01, 0.0, 1.0).astype(np.float32, copy=False),
            "green": np.clip(y01, 0.0, 1.0).astype(np.float32, copy=False),
            "blue": np.clip(np.sqrt(((x01 - 0.5) ** 2) + ((y01 - 0.5) ** 2)) * 1.6, 0.0, 1.0).astype(np.float32, copy=False),
            "yellow": np.clip((x01 + y01) * 0.5, 0.0, 1.0).astype(np.float32, copy=False),
            "cyan": np.clip(np.abs(x01 - y01) * 1.4, 0.0, 1.0).astype(np.float32, copy=False),
            "magenta": ((np.floor(xx / 12.0) + np.floor(yy / 12.0)) % 2.0).astype(np.float32, copy=False),
            "brown": np.clip((0.55 * _signal_pattern(phase=phase)) + (0.45 * _box_blur(_signal_pattern(phase=phase + 0.8), k=9)), 0.0, 1.0).astype(np.float32, copy=False),
            "gray": np.full((size, size), 0.5, dtype=np.float32),
            "grey": np.full((size, size), 0.5, dtype=np.float32),
            "front": np.clip(1.0 - (np.sqrt(((x01 - 0.5) ** 2) + ((y01 - 0.5) ** 2)) * 1.8), 0.0, 1.0).astype(np.float32, copy=False),
            "back": np.clip(np.sqrt(((x01 - 0.5) ** 2) + ((y01 - 0.5) ** 2)) * 1.8, 0.0, 1.0).astype(np.float32, copy=False),
            "left": np.clip(1.0 - x01, 0.0, 1.0).astype(np.float32, copy=False),
            "right": np.clip(x01, 0.0, 1.0).astype(np.float32, copy=False),
            "top": np.clip(1.0 - y01, 0.0, 1.0).astype(np.float32, copy=False),
            "bottom": np.clip(y01, 0.0, 1.0).astype(np.float32, copy=False),
        }
        if key in color_patterns:
            return np.asarray(color_patterns[key], dtype=np.float32)
        if key.startswith("digit "):
            return _symbol_token_pattern(token=key, sample_idx=int(sample_idx), mode="digit")
        if key.startswith("letter "):
            return _symbol_token_pattern(token=key, sample_idx=int(sample_idx), mode="letter")
        if key.startswith("pictogram "):
            return _symbol_token_pattern(token=key, sample_idx=int(sample_idx), mode="pictogram")
        if key == "white":
            return np.ones((size, size), dtype=np.float32)
        if key == "black":
            return np.zeros((size, size), dtype=np.float32)
        if key == "none":
            return np.full((size, size), 0.5, dtype=np.float32)
        if key == "noise":
            return _spectral_noise_image(sample_idx=int(sample_idx), beta=0.0, distribution="uniform")
        if key == "white noise":
            return _spectral_noise_image(sample_idx=int(sample_idx), beta=0.0, distribution="gaussian")
        if key == "uniform white noise":
            return _spectral_noise_image(sample_idx=int(sample_idx), beta=0.0, distribution="uniform")
        if key == "gaussian white noise":
            return _spectral_noise_image(sample_idx=int(sample_idx), beta=0.0, distribution="gaussian")
        if key == "pink noise":
            return _spectral_noise_image(sample_idx=int(sample_idx), beta=1.0, distribution="gaussian")
        if key == "brown noise":
            return _spectral_noise_image(sample_idx=int(sample_idx), beta=2.0, distribution="gaussian")
        if key == "red noise":
            return _spectral_noise_image(sample_idx=int(sample_idx), beta=1.8, distribution="gaussian")
        if key == "blue noise":
            return _spectral_noise_image(sample_idx=int(sample_idx), beta=-1.0, distribution="gaussian")
        if key == "violet noise":
            return _spectral_noise_image(sample_idx=int(sample_idx), beta=-2.0, distribution="gaussian")
        if key == "gray noise":
            return _spectral_noise_image(sample_idx=int(sample_idx), beta=0.5, distribution="gaussian")
        if key == "signal":
            return _signal_pattern(phase=phase)
        if key == "object":
            return _circle_object()
        if key in ("mixed noise and signal", "mix"):
            sig = _signal_pattern(phase=phase)
            noi = rng.random((size, size), dtype=np.float32)
            if int(sample_idx) % 2 == 0:
                return _mix_signal_with_noise_average(sig=sig, noi=noi, alpha=float(rng.uniform(0.35, 0.65)))
            return _mix_signal_with_noise_pcm(sig=sig, noi=noi, noise_bits=int(rng.integers(2, 6)))
        if key == "blur damage":
            src = _signal_pattern(phase=phase)
            return _box_blur(src, k=int(5 + (2 * (int(sample_idx) % 3))))
        if key == "noise damage":
            sig = _signal_pattern(phase=phase)
            return np.clip(sig + (0.20 * rng.standard_normal((size, size), dtype=np.float32)), 0.0, 1.0)
        if key == "dropout damage":
            src = _signal_pattern(phase=phase)
            keep_p = 0.20 + (0.15 * float((int(sample_idx) % 3) / 2.0))
            keep = (rng.random((size, size), dtype=np.float32) > keep_p).astype(np.float32, copy=False)
            return np.clip(src * keep, 0.0, 1.0)
        if key == "quantization damage":
            levels = int([3, 4, 6, 8][int(sample_idx) % 4])
            return _quantize(_signal_pattern(phase=phase), levels=levels)
        if key == "stride skew damage":
            src = _signal_pattern(phase=phase)
            out = np.array(src, dtype=np.float32, copy=True)
            shift = int([2, 3, 5, 7][int(sample_idx) % 4])
            out[1::2, :] = np.roll(out[1::2, :], shift=shift, axis=1)
            return np.clip(out, 0.0, 1.0)
        if key == "berkeley sbd dataset":
            return np.clip((0.65 * _circle_object()) + (0.35 * _signal_pattern(phase=phase)), 0.0, 1.0)
        if key == "mnist dataset":
            return np.clip((0.75 * (x01 > 0.45).astype(np.float32)) + (0.25 * _signal_pattern(phase=phase)), 0.0, 1.0)
        if key == "emnist dataset":
            return np.clip((0.70 * (np.abs(x01 - y01) < 0.12).astype(np.float32)) + (0.30 * _signal_pattern(phase=phase)), 0.0, 1.0)
        if key == "kmnist dataset":
            return np.clip((0.70 * (np.abs((x01 + y01) - 1.0) < 0.12).astype(np.float32)) + (0.30 * _signal_pattern(phase=phase)), 0.0, 1.0)
        if key == "regurgitated content":
            return np.clip((0.60 * _signal_pattern(phase=phase)) + (0.40 * _signal_pattern(phase=phase + 1.8)), 0.0, 1.0)
        if key == "gan image":
            checker = ((np.floor(xx / 8.0) + np.floor(yy / 8.0)) % 2.0).astype(np.float32, copy=False)
            return np.clip((0.65 * checker) + (0.35 * _signal_pattern(phase=phase)), 0.0, 1.0)
        if key == "spectrographic output":
            return _spectrographic_image(sample_idx=int(sample_idx))
        if key == "inverse spectrographic composition":
            return _inverse_spectrographic_image(sample_idx=int(sample_idx))
        if key == "pure tone construction":
            return _pure_tone_image(sample_idx=int(sample_idx))
        if key in berkeley_term_lc:
            return np.clip((0.62 * _circle_object()) + (0.38 * _signal_pattern(phase=phase + 0.3)), 0.0, 1.0)
        return _signal_pattern(phase=phase)

    core_terms = _normalize_vocab_terms(_default_semantic_core_terms())
    try:
        berkeley_terms = _normalize_vocab_terms(_default_berkeley_class_names())
    except Exception:
        berkeley_terms = []
    berkeley_term_lc = {str(x).strip().lower() for x in berkeley_terms}
    bootstrap_primitive_terms = _normalize_vocab_terms(_default_bootstrap_primitive_terms())
    primary_terms = _normalize_vocab_terms(list(bootstrap_primitive_terms))

    for term in primary_terms:
        key = re.sub(r"\s+", " ", str(term)).strip().lower()
        for j in range(int(cap)):
            gray = _term_gray(term=key, sample_idx=int(j))
            _write_sample(term, sample_idx=int(j), gray=gray)
            _push(term, gray)

    manifest = {
        "seed": int(seed),
        "image_size": int(size),
        "max_samples_per_term": int(cap),
        "primary_terms": list(primary_terms),
        "core_terms": list(core_terms),
        "bootstrap_primitive_terms": list(bootstrap_primitive_terms),
        "berkeley_terms": list(berkeley_terms),
        "origin_label": str(origin_label).strip(),
        "pool_terms": sorted([str(k) for k in pool.keys()]),
        "term_counts": {str(k): int(len(v)) for k, v in pool.items()},
    }
    try:
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except Exception:
        pass

    info["available_terms"] = int(len(pool))
    info["samples"] = int(sum(len(v) for v in pool.values()))
    info["primary_terms"] = int(len(primary_terms))
    info["bootstrap_primitive_terms"] = int(len(bootstrap_primitive_terms))
    return pool, info


def _build_reference_flashcard_payload_rows(
    class_names: Sequence[str],
    condition_num_classes: int,
    supervised_num_classes: int,
    payload_images_base: Sequence[np.ndarray],
    payload_conditions_supervised_base: Sequence[np.ndarray],
    symbol_pool_by_term: Dict[str, List[np.ndarray]],
    image_size: int,
    per_term: int,
    seed: int,
    condition_vector_builder: Callable[[Sequence[str], Optional[np.ndarray]], np.ndarray],
    gan_image_provider: Optional[Callable[[int], List[np.ndarray]]] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, Any]]:
    c = max(1, int(condition_num_classes))
    per = max(0, int(per_term))
    if per <= 0:
        return [], [], {"enabled": False, "rows_added": 0, "reason": "per_term_zero"}
    if not callable(condition_vector_builder):
        raise RuntimeError("Flashcard condition builder is required; non-embedding fallback is disabled.")

    class_term_keys = {
        re.sub(r"\s+", " ", str(t)).strip().lower()
        for t in class_names
        if str(t).strip()
    }
    target_terms = [t for t in _default_semantic_core_terms() if str(t).strip().lower() in class_term_keys]
    if len(target_terms) <= 0:
        return [], [], {"enabled": False, "rows_added": 0, "reason": "no_core_terms_in_vocab"}

    size = max(8, int(image_size))
    if int(size) <= 0:
        if len(payload_images_base) > 0:
            sample = np.asarray(payload_images_base[0], dtype=np.float32)
            if int(sample.ndim) == 3:
                size = int(sample.shape[1])
        if int(size) <= 0:
            size = 256

    rng = np.random.default_rng(int(seed))
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    x01 = (xx / float(max(1, size - 1))).astype(np.float32, copy=False)
    y01 = (yy / float(max(1, size - 1))).astype(np.float32, copy=False)

    def _to_rgb(g: np.ndarray) -> np.ndarray:
        gg = np.clip(np.asarray(g, dtype=np.float32), 0.0, 1.0)
        return np.repeat(gg[None, :, :], 3, axis=0).astype(np.float32, copy=False)

    def _signal_pattern() -> np.ndarray:
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        fx = float(rng.uniform(2.0, 7.0))
        fy = float(rng.uniform(2.0, 6.0))
        g = 0.5 + (0.5 * np.sin((2.0 * math.pi * ((fx * x01) + (fy * y01))) + phase))
        return np.clip(g, 0.0, 1.0).astype(np.float32, copy=False)

    def _sample_white_noise_profile_key() -> str:
        return "gaussian_white_noise" if float(rng.random()) < 0.5 else "uniform_white_noise"

    def _noise_pattern(profile_key: str) -> np.ndarray:
        key = re.sub(r"\s+", "_", str(profile_key)).strip().lower()
        if key == "gaussian_white_noise":
            arr = rng.standard_normal((size, size), dtype=np.float32)
            lo = float(np.min(arr))
            hi = float(np.max(arr))
            den = max(1e-8, float(hi - lo))
            return np.clip((arr - lo) / den, 0.0, 1.0).astype(np.float32, copy=False)
        return np.clip(rng.random((size, size), dtype=np.float32), 0.0, 1.0).astype(np.float32, copy=False)

    def _mix_signal_with_noise_pcm(sig: np.ndarray, noi: np.ndarray, noise_bits: int) -> np.ndarray:
        # Compose signal + noise by replacing low-order PCM bits in the signal with noise payload bits.
        s = np.clip(np.asarray(sig, dtype=np.float32), 0.0, 1.0)
        n = np.clip(np.asarray(noi, dtype=np.float32), 0.0, 1.0)
        bits = max(1, min(8, int(noise_bits)))
        s_u16 = np.round(s * 65535.0).astype(np.uint16, copy=False)
        n_u16 = np.round(n * 65535.0).astype(np.uint16, copy=False)
        payload = ((n_u16 >> int(16 - bits)) & np.uint16((1 << bits) - 1)).astype(np.uint16, copy=False)
        keep_mask = np.uint16(0xFFFF ^ ((1 << bits) - 1))
        packed = ((s_u16 & keep_mask) | payload).astype(np.uint16, copy=False)
        return np.clip(packed.astype(np.float32) / 65535.0, 0.0, 1.0).astype(np.float32, copy=False)

    def _apply_damage(img_chw: np.ndarray, damage_term: str) -> np.ndarray:
        x = np.clip(np.asarray(img_chw, dtype=np.float32), 0.0, 1.0)
        key = re.sub(r"\s+", " ", str(damage_term)).strip().lower()
        if key == "blur damage":
            t = torch.from_numpy(x[None, ...]).to(torch.float32)
            t = F.avg_pool2d(t, kernel_size=5, stride=1, padding=2)
            return np.clip(t[0].cpu().numpy().astype(np.float32, copy=False), 0.0, 1.0)
        if key == "noise damage":
            n = (rng.standard_normal(x.shape).astype(np.float32) * 0.16).astype(np.float32, copy=False)
            return np.clip(x + n, 0.0, 1.0).astype(np.float32, copy=False)
        if key == "dropout damage":
            keep = (rng.random((1, int(x.shape[1]), int(x.shape[2])), dtype=np.float32) > 0.22).astype(np.float32, copy=False)
            return np.clip(x * keep, 0.0, 1.0).astype(np.float32, copy=False)
        if key == "quantization damage":
            levels = int(rng.integers(3, 7))
            return np.clip(np.round(x * float(levels - 1)) / float(levels - 1), 0.0, 1.0).astype(np.float32, copy=False)
        if key == "stride skew damage":
            out = np.array(x, dtype=np.float32, copy=True)
            shift = int(max(1, min(6, int(size // 24))))
            out[:, 1::2, :] = np.roll(out[:, 1::2, :], shift=shift, axis=2)
            return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)
        return x

    def _collect_symbol_rows(prefix: str) -> List[np.ndarray]:
        key_prefix = re.sub(r"\s+", " ", str(prefix)).strip().lower()
        rows: List[np.ndarray] = []
        for k, vals in symbol_pool_by_term.items():
            key = re.sub(r"\s+", " ", str(k)).strip().lower()
            if key == key_prefix or key.startswith(f"{key_prefix} "):
                rows.extend([np.asarray(v, dtype=np.float32) for v in vals])
        return rows

    object_seed_rows: List[Tuple[np.ndarray, np.ndarray]] = []
    n_obj = min(len(payload_images_base), len(payload_conditions_supervised_base))
    for i in range(int(n_obj)):
        object_seed_rows.append(
            (
                _image_any_to_rgb_chw01(payload_images_base[int(i)], image_size=int(size)),
                np.asarray(payload_conditions_supervised_base[int(i)], dtype=np.float32).reshape(-1),
            )
        )

    digit_rows = _collect_symbol_rows("digit")
    letter_rows = _collect_symbol_rows("letter")
    pict_rows = _collect_symbol_rows("pictogram")
    signal_rows = list(digit_rows) + list(letter_rows) + list(pict_rows) + [r[0] for r in object_seed_rows]
    if len(signal_rows) <= 0:
        signal_rows = [_to_rgb(_signal_pattern())]

    def _pick_signal() -> np.ndarray:
        idx = int(rng.integers(0, len(signal_rows)))
        return _image_any_to_rgb_chw01(signal_rows[idx], image_size=int(size))

    def _pick_object(require_supervised: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if len(object_seed_rows) <= 0:
            if bool(require_supervised):
                raise RuntimeError(
                    "Reference flashcard row requires Berkeley-supervised seed content, but payload object rows are empty."
                )
            return _pick_signal(), None
        idx = int(rng.integers(0, len(object_seed_rows)))
        img, sv = object_seed_rows[idx]
        return _image_any_to_rgb_chw01(img, image_size=int(size)), np.asarray(sv, dtype=np.float32).reshape(-1)

    def _pick_dataset_row(dataset_term: str) -> np.ndarray:
        key = re.sub(r"\s+", " ", str(dataset_term)).strip().lower()
        if key == "mnist dataset" and len(digit_rows) > 0:
            return _image_any_to_rgb_chw01(digit_rows[int(rng.integers(0, len(digit_rows)))], image_size=int(size))
        if key == "emnist dataset" and len(letter_rows) > 0:
            return _image_any_to_rgb_chw01(letter_rows[int(rng.integers(0, len(letter_rows)))], image_size=int(size))
        if key == "kmnist dataset" and len(pict_rows) > 0:
            return _image_any_to_rgb_chw01(pict_rows[int(rng.integers(0, len(pict_rows)))], image_size=int(size))
        if key == "berkeley sbd dataset":
            img, _ = _pick_object()
            return img
        return _pick_signal()

    cards_img: List[np.ndarray] = []
    cards_cond: List[np.ndarray] = []
    term_counts: Dict[str, int] = {}

    def _emit(term: str, img: np.ndarray, base_supervised: Optional[np.ndarray], extra_terms: Sequence[str]):
        key = re.sub(r"\s+", " ", str(term)).strip().lower()
        if not key:
            return
        img_rgb = _image_any_to_rgb_chw01(img, image_size=int(size))
        terms = _semantic_terms_with_tonal_tags(
            terms=(list(extra_terms) + [key]),
            image=img_rgb,
            image_size=int(size),
        )
        vec = np.asarray(condition_vector_builder(terms, base_supervised), dtype=np.float32).reshape(-1)
        if int(vec.size) != int(c):
            raise RuntimeError(f"flashcard condition builder width mismatch: got={int(vec.size)} expected={int(c)}")
        terms_lc = {
            re.sub(r"\s+", " ", str(t)).strip().lower()
            for t in terms
            if re.sub(r"\s+", " ", str(t)).strip()
        }
        requires_berkeley_supervision = bool(
            ("berkeley sbd dataset" in terms_lc) or ("object" in terms_lc) or (key in damage_terms)
        )
        if bool(requires_berkeley_supervision):
            if base_supervised is None:
                raise RuntimeError(
                    "Berkeley/object flashcard row emitted without supervised label vector. "
                    f"term={str(term)!r}"
                )
            sup_arr = np.asarray(base_supervised, dtype=np.float32).reshape(-1)
            sup_take = min(int(sup_arr.size), int(vec.size))
            if int(sup_take) <= 0:
                raise RuntimeError(
                    "Berkeley/object flashcard row has empty supervised label vector. "
                    f"term={str(term)!r}"
                )
            sup_clip = np.clip(np.asarray(sup_arr[: int(sup_take)], dtype=np.float32), 0.0, 1.0)
            vec_clip = np.clip(np.asarray(vec[: int(sup_take)], dtype=np.float32), 0.0, 1.0)
            missing = (sup_clip >= 0.5) & (vec_clip < 0.5)
            if bool(np.any(missing)):
                raise RuntimeError(
                    "Berkeley/object flashcard row dropped original supervised labels during conditioning. "
                    f"term={str(term)!r}"
                )
        cards_img.append(img_rgb)
        cards_cond.append(np.asarray(vec, dtype=np.float32).reshape(-1))
        term_counts[key] = int(term_counts.get(key, 0)) + 1

    damage_terms = {"blur damage", "noise damage", "dropout damage", "quantization damage", "stride skew damage"}
    for term in target_terms:
        term_key = re.sub(r"\s+", " ", str(term)).strip().lower()
        for _ in range(int(per)):
            if term_key == "none":
                _emit(term_key, _to_rgb(np.full((size, size), 0.50, dtype=np.float32)), None, ["none"])
                continue
            if term_key == "noise":
                noise_profile = _sample_white_noise_profile_key()
                noise_img = _noise_pattern(noise_profile)
                _emit(
                    term_key,
                    _to_rgb(noise_img),
                    None,
                    list(_semantic_noise_terms_from_spectrum_sample(noise_img)),
                )
                continue
            if term_key == "signal":
                _emit(term_key, _pick_signal(), None, ["signal"])
                continue
            if term_key == "mixed noise and signal":
                noise_profile = _sample_white_noise_profile_key()
                sig = _pick_signal()
                noise_img = _noise_pattern(noise_profile)
                noi = _to_rgb(noise_img)
                if int(rng.integers(0, 2)) == 0:
                    a = float(rng.uniform(0.35, 0.65))
                    mixed = np.clip((a * sig) + ((1.0 - a) * noi), 0.0, 1.0)
                else:
                    mixed = _mix_signal_with_noise_pcm(
                        sig=sig,
                        noi=noi,
                        noise_bits=int(rng.integers(2, 6)),
                    )
                _emit(
                    term_key,
                    mixed,
                    None,
                    ["signal", "mixed noise and signal"] + list(_semantic_noise_terms_from_spectrum_sample(noise_img)),
                )
                continue
            if term_key == "white":
                _emit(term_key, _to_rgb(np.ones((size, size), dtype=np.float32)), None, ["white", "signal"])
                continue
            if term_key == "black":
                _emit(term_key, _to_rgb(np.zeros((size, size), dtype=np.float32)), None, ["black", "signal"])
                continue
            if term_key == "object":
                obj_img, obj_sup = _pick_object(require_supervised=True)
                _emit(term_key, obj_img, obj_sup, ["object", "berkeley sbd dataset", "signal"])
                continue
            if term_key in ("mnist dataset", "emnist dataset", "kmnist dataset", "berkeley sbd dataset"):
                if term_key == "berkeley sbd dataset":
                    obj_img, obj_sup = _pick_object(require_supervised=True)
                    _emit(term_key, obj_img, obj_sup, [term_key, "object", "signal"])
                else:
                    ds_img = _pick_dataset_row(term_key)
                    _emit(term_key, ds_img, None, [term_key, "signal"])
                continue
            if term_key == "gan image":
                live_rows: List[np.ndarray] = []
                if callable(gan_image_provider):
                    try:
                        live_rows = [np.asarray(x, dtype=np.float32) for x in gan_image_provider(1)]
                    except Exception:
                        live_rows = []
                if len(live_rows) > 0:
                    _emit(term_key, _image_any_to_rgb_chw01(live_rows[0], image_size=int(size)), None, ["gan image", "signal"])
                continue
            if term_key == "regurgitated content":
                sig0 = _signal_pattern()
                sig1 = np.roll(sig0, shift=int(rng.integers(2, 9)), axis=1)
                _emit(
                    term_key,
                    _to_rgb(np.clip((0.55 * sig0) + (0.45 * sig1), 0.0, 1.0)),
                    None,
                    ["regurgitated content", "signal"],
                )
                continue
            if term_key in damage_terms:
                obj_img, obj_sup = _pick_object(require_supervised=True)
                dmg_extra_terms = list(_semantic_damage_tags(term_key))
                if term_key == "noise damage":
                    noise_delta = (rng.standard_normal(obj_img.shape).astype(np.float32) * 0.16).astype(np.float32, copy=False)
                    dmg = np.clip(np.asarray(obj_img, dtype=np.float32) + noise_delta, 0.0, 1.0).astype(np.float32, copy=False)
                    dmg_extra_terms.extend(_semantic_noise_terms_from_spectrum_sample(noise_delta))
                else:
                    dmg = _apply_damage(obj_img, term_key)
                _emit(
                    term_key,
                    dmg,
                    obj_sup,
                    list(dmg_extra_terms) + ["object", "berkeley sbd dataset"],
                )
                continue
            _emit(term_key, _pick_signal(), None, [term_key, "signal"])

    if len(cards_cond) > 1:
        perm = rng.permutation(np.arange(len(cards_cond), dtype=np.int64)).astype(np.int64).tolist()
        cards_img = [cards_img[int(i)] for i in perm]
        cards_cond = [cards_cond[int(i)] for i in perm]

    info = {
        "enabled": True,
        "rows_added": int(len(cards_cond)),
        "terms_targeted": int(len(target_terms)),
        "terms_covered": int(len(term_counts)),
        "per_term": int(per),
        "missing_terms": [t for t in target_terms if int(term_counts.get(str(t).strip().lower(), 0)) <= 0],
    }
    target_stats = _semantic_active_target_stats(cards_cond, threshold=0.5)
    info["target_active_min"] = int(target_stats.get("min", 0))
    info["target_active_mean"] = float(target_stats.get("mean", 0.0))
    info["target_active_p50"] = float(target_stats.get("p50", 0.0))
    info["target_active_max"] = int(target_stats.get("max", 0))
    return cards_img, cards_cond, info


def _rotate_active_extra_terms(
    active_terms: Sequence[str],
    pool_terms: Sequence[str],
    replace_count: int,
    seed: int,
    locked_prefix_count: int = 0,
    churn_cursor: int = 0,
    sweep_cycles: int = 0,
) -> Tuple[List[str], Dict[str, Any]]:
    active = [str(x) for x in active_terms]
    pool = _normalize_vocab_terms(pool_terms)
    k_replace = max(0, int(replace_count))
    locked = max(0, min(int(len(active)), int(locked_prefix_count)))
    sweep_n = max(0, int(sweep_cycles))
    cursor_in = max(0, int(churn_cursor))
    info: Dict[str, Any] = {
        "changed": False,
        "replaced": 0,
        "pool_terms": int(len(pool)),
        "active_terms": int(len(active)),
        "locked_prefix": int(locked),
        "replace_base": int(k_replace),
        "replace_auto": 0,
        "candidate_terms": 0,
        "sweep_cycles": int(sweep_n),
        "cursor_in": int(cursor_in),
        "cursor_out": int(cursor_in),
    }
    unlocked = int(len(active) - int(locked))
    if len(active) <= 0 or len(pool) <= len(active) or k_replace <= 0 or int(unlocked) <= 0:
        return active, info
    _ = int(seed)  # Kept for API compatibility; churn rotation is deterministic.
    active_lc = {str(x).strip().lower() for x in active}
    candidates = [t for t in pool if str(t).strip().lower() not in active_lc]
    info["candidate_terms"] = int(len(candidates))
    if len(candidates) <= 0:
        return active, info
    replace_n = min(int(k_replace), int(unlocked))
    if int(sweep_n) > 0:
        auto_n = int(math.ceil(float(len(candidates)) / float(max(1, int(sweep_n)))))
        info["replace_auto"] = int(auto_n)
        replace_n = max(int(replace_n), int(auto_n))
    replace_n = min(int(replace_n), int(unlocked), int(len(candidates)))
    if int(replace_n) <= 0:
        return active, info
    replace_space = [int(i) for i in range(int(locked), int(len(active)))]
    if len(replace_space) <= 0:
        return active, info
    slot_start = int(cursor_in) % int(len(replace_space))
    replace_idx = [int(replace_space[(slot_start + i) % int(len(replace_space))]) for i in range(int(replace_n))]
    picked_terms: List[str] = []
    ordered = sorted([str(t) for t in candidates], key=lambda s: s.lower())
    start = int(cursor_in) % int(len(ordered))
    picked_terms = [str(ordered[(start + i) % int(len(ordered))]) for i in range(int(replace_n))]
    info["cursor_out"] = int((start + int(replace_n)) % int(len(ordered)))
    changed = 0
    for pos, idx in enumerate(replace_idx):
        if pos >= len(picked_terms):
            break
        new_t = str(picked_terms[pos]).strip()
        if not new_t:
            continue
        if str(active[int(idx)]).strip().lower() == new_t.lower():
            continue
        active[int(idx)] = new_t
        changed += 1
    info["changed"] = bool(changed > 0)
    info["replaced"] = int(changed)
    return active, info


def _compute_gd_vocab_hash(
    supervised_class_names: Sequence[str],
    fixed_extra_terms: Sequence[str],
    condition_num_classes: int,
    args: Any,
    active_extra_terms: Optional[Sequence[str]] = None,
) -> Tuple[str, Dict[str, Any]]:
    hash_basis: Dict[str, Any] = {
        "supervised_class_names": [str(x) for x in supervised_class_names],
        "fixed_extra_terms": [str(x) for x in fixed_extra_terms],
        "condition_num_classes": int(condition_num_classes),
        "label_embedding_backend": str(getattr(args, "label_embedding_backend", "")),
        "label_embedding_model": str(getattr(args, "label_embedding_model", "")),
        "label_embedding_dim": int(getattr(args, "label_embedding_dim", 0)),
        "semantic_label_mode": "presence_v2",
        "generator": {
            "z_dim": int(getattr(args, "generator_z_dim", 0)),
            "base_ch": int(getattr(args, "generator_base_ch", 0)),
            "depth": int(getattr(args, "generator_depth", 0)),
            "min_ch": int(getattr(args, "generator_min_ch", 0)),
        },
        "discriminator": {
            "base_ch": int(getattr(args, "discriminator_base_ch", 0)),
            "depth": int(getattr(args, "discriminator_depth", 0)),
            "max_ch": int(getattr(args, "discriminator_max_ch", 0)),
        },
    }
    profile: Dict[str, Any] = dict(hash_basis)
    profile["active_extra_terms"] = [str(x) for x in (active_extra_terms or [])]
    blob = json.dumps(hash_basis, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(blob).hexdigest()[:24]
    return str(digest), profile


def _save_gd_vocab_library_snapshot(
    library_dir: Path,
    vocab_hash: str,
    condition_num_classes: int,
    generator: Optional[nn.Module],
    discriminator: Optional[nn.Module],
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    if generator is None or discriminator is None:
        return {"saved": False, "reason": "missing_generator_or_discriminator"}
    h = str(vocab_hash).strip()
    if not h:
        return {"saved": False, "reason": "empty_vocab_hash"}
    c = max(1, int(condition_num_classes))
    subdir = Path(library_dir) / f"{h}_c{int(c)}"
    subdir.mkdir(parents=True, exist_ok=True)
    gen_path = subdir / "generator.pt"
    disc_path = subdir / "discriminator.pt"
    meta_path = subdir / "meta.json"
    torch.save({"state_dict": generator.state_dict()}, gen_path)
    torch.save({"state_dict": discriminator.state_dict()}, disc_path)
    blob = dict(meta) if isinstance(meta, dict) else {}
    blob["vocab_hash"] = str(h)
    blob["condition_num_classes"] = int(c)
    blob["timestamp"] = float(time.time())
    meta_path.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    return {"saved": True, "dir": str(subdir), "generator": str(gen_path), "discriminator": str(disc_path)}


def _load_gd_vocab_library_snapshot(
    library_dir: Path,
    vocab_hash: str,
    generator: Optional[nn.Module],
    discriminator: Optional[nn.Module],
) -> Dict[str, Any]:
    if generator is None or discriminator is None:
        return {"loaded": False, "reason": "missing_generator_or_discriminator"}
    h = str(vocab_hash).strip()
    if not h:
        return {"loaded": False, "reason": "empty_vocab_hash"}
    root = Path(library_dir)
    if not root.exists():
        return {"loaded": False, "reason": "library_missing"}
    candidates = sorted(root.glob(f"{h}_c*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if len(candidates) <= 0:
        return {"loaded": False, "reason": "snapshot_not_found", "hash": str(h)}
    pick = candidates[0]
    gen_path = pick / "generator.pt"
    disc_path = pick / "discriminator.pt"
    if not gen_path.exists() or not disc_path.exists():
        return {"loaded": False, "reason": "snapshot_incomplete", "dir": str(pick)}
    gen_blob = _torch_load_cpu(str(gen_path))
    disc_blob = _torch_load_cpu(str(disc_path))
    gen_state = _extract_state_dict(gen_blob)
    disc_state = _extract_state_dict(disc_blob)
    gen_info = _apply_state_dict(generator, gen_state, source_name=f"gd_vocab_library:{pick.name}:generator")
    disc_info = _apply_state_dict(discriminator, disc_state, source_name=f"gd_vocab_library:{pick.name}:discriminator")
    return {
        "loaded": True,
        "dir": str(pick),
        "generator": gen_info,
        "discriminator": disc_info,
    }


def _encode_query_texts_for_bank(query_texts: Sequence[str], bank_info: Dict[str, Any], args, device: torch.device) -> np.ndarray:
    if len(query_texts) <= 0:
        return np.zeros((0, max(1, int(bank_info.get("dim", 1)))), dtype=np.float32)
    backend = str(bank_info.get("backend_used", "")).strip().lower()
    if backend != "sentence_transformers":
        raise RuntimeError(
            "Only sentence-transformers semantic vectors are permitted for query encoding; "
            f"bank backend is {backend!r}."
        )
    emb = _encode_texts_sentence_transformers(
        query_texts,
        model_name=str(bank_info.get("model_name", str(args.label_embedding_model))),
        device=device,
    )
    bank_dim = max(1, int(bank_info.get("dim", emb.shape[1])))
    if int(emb.shape[1]) != bank_dim:
        raise RuntimeError(
            "Sentence-transformer geometry is configured as strict; "
            f"query dim ({int(emb.shape[1])}) must equal bank dim ({int(bank_dim)})."
        )
    return _normalize_l2_rows_np(emb.astype(np.float32, copy=False))


def _nearest_labels_for_queries(
    query_texts: Sequence[str],
    label_bank_np: np.ndarray,
    class_names: Sequence[str],
    label_texts: Sequence[str],
    topk: int,
    query_emb_np: np.ndarray,
) -> List[Dict[str, Any]]:
    if len(query_texts) <= 0:
        return []
    bank_arr = np.asarray(label_bank_np, dtype=np.float32)
    q_arr = np.asarray(query_emb_np, dtype=np.float32)
    if bank_arr.ndim != 2 or q_arr.ndim != 2:
        reason = f"shape_mismatch: query_shape={tuple(q_arr.shape)} label_shape={tuple(bank_arr.shape)}"
        return [{"query": str(qt), "matches": [], "reason": reason} for qt in query_texts]

    bank = _normalize_l2_rows_np(bank_arr)
    q = _normalize_l2_rows_np(q_arr)
    alignment_note = ""

    if int(q.shape[1]) != int(bank.shape[1]):
        # Keep semantic space intact: first detect the common accidental transpose case.
        if int(bank.shape[0]) == int(q.shape[1]) and int(bank.shape[1]) != int(q.shape[1]):
            bank = _normalize_l2_rows_np(bank.transpose(1, 0).astype(np.float32, copy=False))
            alignment_note = "label_bank_transposed"
            _log("[label-query] detected transposed label bank; auto-corrected for similarity lookup.")

    if int(q.shape[1]) != int(bank.shape[1]):
        reason = f"dim_mismatch: query_dim={int(q.shape[1])} label_dim={int(bank.shape[1])}"
        _log(f"[label-query] skipped nearest-label lookup ({reason}).")
        rows: List[Dict[str, Any]] = []
        for qt in query_texts:
            row = {"query": str(qt), "matches": [], "reason": reason}
            if alignment_note:
                row["alignment"] = alignment_note
            rows.append(row)
        return rows

    if int(bank.shape[0]) <= 0:
        return [{"query": str(qt), "matches": [], "reason": "empty_label_bank"} for qt in query_texts]

    sims = q @ bank.transpose(1, 0)
    k = max(1, min(int(topk), int(bank.shape[0])))
    rows: List[Dict[str, Any]] = []
    for i, qt in enumerate(query_texts):
        idx = np.argsort(-sims[i])[:k]
        hits: List[Dict[str, Any]] = []
        for j in idx.tolist():
            c = int(j)
            hits.append(
                {
                    "class_idx": c,
                    "class_name": str(class_names[c]) if c < len(class_names) else f"class_{c}",
                    "label_text": str(label_texts[c]) if c < len(label_texts) else "",
                    "score": float(sims[i, c]),
                }
            )
        row = {"query": str(qt), "matches": hits}
        if alignment_note:
            row["alignment"] = alignment_note
        rows.append(row)
    return rows


@torch.inference_mode()
def _evaluate_zero_shot_queries_on_images(
    classifier: nn.Module,
    x: torch.Tensor,
    query_texts: Sequence[str],
    query_emb_np: np.ndarray,
    batch_size: int,
    device: torch.device,
    amp_enabled: bool = False,
    amp_dtype: str = "float16",
    channels_last: bool = False,
    topk: int = 3,
    max_samples: int = 128,
    max_report_samples: int = 8,
) -> Dict[str, Any]:
    if not isinstance(classifier, TinyConvClassifier):
        return {"ran": False, "reason": f"unsupported_classifier:{type(classifier).__name__}"}
    if x is None or int(x.ndim) != 4 or int(x.shape[0]) <= 0:
        return {"ran": False, "reason": "empty_input"}
    queries = [str(q) for q in query_texts if str(q).strip()]
    if len(queries) <= 0:
        return {"ran": False, "reason": "no_queries"}
    q_np = np.asarray(query_emb_np, dtype=np.float32)
    if q_np.ndim != 2 or int(q_np.shape[0]) != len(queries):
        return {
            "ran": False,
            "reason": (
                f"query_shape_mismatch: queries={len(queries)} "
                f"emb_shape={tuple(q_np.shape)}"
            ),
        }
    q_np = _normalize_l2_rows_np(q_np)
    q_dim = int(q_np.shape[1])
    emb_dim = int(classifier.embed_proj.out_features)
    if q_dim != emb_dim:
        return {
            "ran": False,
            "reason": f"query_dim_mismatch: query_dim={q_dim} embed_dim={emb_dim}",
        }

    use_amp = bool(amp_enabled and device.type == "cuda")
    amp_dtype_t = _resolve_amp_dtype(amp_dtype) if use_amp else torch.float16
    q_bank = torch.from_numpy(q_np).to(device=device, dtype=torch.float32)
    n_all = int(x.shape[0])
    n_use = int(n_all if int(max_samples) <= 0 else min(n_all, max(1, int(max_samples))))
    bsz = max(1, int(batch_size))
    q_n = int(q_bank.shape[0])
    report_cap = max(0, int(max_report_samples))
    topk_eff = max(1, min(int(topk), q_n))

    sum_scores = torch.zeros((q_n,), dtype=torch.float64)
    top1_counts = torch.zeros((q_n,), dtype=torch.int64)
    top1_score_sum = 0.0
    sample_rows: List[Dict[str, Any]] = []

    was_training = bool(classifier.training)
    classifier.eval()
    for i in range(0, n_use, bsz):
        xb = x[i : i + bsz].to(device, non_blocking=True)
        if channels_last:
            xb = xb.contiguous(memory_format=torch.channels_last)
        with _autocast_context(device=device, enabled=use_amp, amp_dtype_t=amp_dtype_t):
            feat = classifier.extract_features(xb)
            sims = classifier.semantic_logits_from_features(
                feat=feat,
                bank=q_bank,
                temperature=1.0,
            ).to(torch.float32)
        sum_scores += sims.sum(dim=0).detach().to("cpu", dtype=torch.float64)
        top1_vals, top1_idx = torch.max(sims, dim=1)
        top1_score_sum += float(top1_vals.sum().item())
        top1_counts += torch.bincount(top1_idx.detach().to("cpu"), minlength=q_n)

        if len(sample_rows) < report_cap:
            take = min(int(sims.shape[0]), int(report_cap - len(sample_rows)))
            vals, idxs = torch.topk(sims, k=topk_eff, dim=1)
            vals = vals.detach().to("cpu", dtype=torch.float32)
            idxs = idxs.detach().to("cpu", dtype=torch.int64)
            for r in range(take):
                hits = []
                for j in range(topk_eff):
                    q_idx = int(idxs[r, j].item())
                    hits.append(
                        {
                            "query_idx": int(q_idx),
                            "query": str(queries[q_idx]),
                            "score": float(vals[r, j].item()),
                        }
                    )
                sample_rows.append(
                    {
                        "sample_idx": int(i + r),
                        "topk": hits,
                    }
                )
    classifier.train(was_training)

    denom = float(max(1, n_use))
    mean_scores = (sum_scores / denom).numpy()
    top1_rate = (top1_counts.to(dtype=torch.float64) / denom).numpy()
    order = np.argsort(-mean_scores)
    query_rows: List[Dict[str, Any]] = []
    for q_idx in order.tolist():
        q_i = int(q_idx)
        query_rows.append(
            {
                "query_idx": int(q_i),
                "query": str(queries[q_i]),
                "mean_score": float(mean_scores[q_i]),
                "top1_rate": float(top1_rate[q_i]),
                "top1_count": int(top1_counts[q_i].item()),
            }
        )

    best = query_rows[0] if len(query_rows) > 0 else {}
    return {
        "ran": True,
        "samples": int(n_use),
        "num_queries": int(q_n),
        "topk": int(topk_eff),
        "mean_top1_score": float(top1_score_sum / denom),
        "best_query": str(best.get("query", "")),
        "best_query_mean_score": float(best.get("mean_score", 0.0)),
        "best_query_top1_rate": float(best.get("top1_rate", 0.0)),
        "query_rows": query_rows,
        "sample_rows": sample_rows,
    }


def _wave_feedback_snapshot(
    wave_classifier_eval: Optional[Dict[str, Any]],
    wave_zero_shot_eval: Optional[Dict[str, Any]],
    zero_shot_weight: float,
) -> Dict[str, Any]:
    acc = None
    if isinstance(wave_classifier_eval, dict) and ("acc" in wave_classifier_eval):
        try:
            acc = float(wave_classifier_eval.get("acc", 0.0))
        except Exception:
            acc = None

    zs_top1 = None
    zs_ran = False
    if isinstance(wave_zero_shot_eval, dict) and bool(wave_zero_shot_eval.get("ran", False)):
        zs_ran = True
        try:
            zs_top1 = float(wave_zero_shot_eval.get("best_query_top1_rate", 0.0))
        except Exception:
            zs_top1 = None

    weighted_sum = 0.0
    weighted_den = 0.0
    if acc is not None:
        weighted_sum += float(acc)
        weighted_den += 1.0
    if zs_top1 is not None:
        z_w = max(0.0, float(zero_shot_weight))
        if z_w > 0.0:
            weighted_sum += z_w * float(zs_top1)
            weighted_den += z_w

    combined = None
    if weighted_den > 0.0:
        combined = float(weighted_sum / weighted_den)
        combined = float(max(0.0, min(1.0, combined)))

    return {
        "available": bool(weighted_den > 0.0),
        "wave_acc": (None if acc is None else float(max(0.0, min(1.0, acc)))),
        "zero_shot_ran": bool(zs_ran),
        "zero_shot_top1_rate": (None if zs_top1 is None else float(max(0.0, min(1.0, zs_top1)))),
        "combined_score": combined,
    }


def _feedback_scaled_weight(base_weight: float, feedback_score: Optional[float], boost: float) -> float:
    base = max(0.0, float(base_weight))
    b = max(0.0, float(boost))
    if feedback_score is None:
        return float(base)
    s = float(max(0.0, min(1.0, float(feedback_score))))
    return float(base * (1.0 + (b * (1.0 - s))))


def _wave_feedback_gate_pass(
    feedback: Dict[str, Any],
    min_acc: float,
    min_zero_shot_top1_rate: float,
) -> Tuple[bool, str]:
    if not bool(feedback.get("available", False)):
        return True, "warmup_no_wave_feedback"

    acc_floor = max(0.0, float(min_acc))
    zs_floor = max(0.0, float(min_zero_shot_top1_rate))
    acc = feedback.get("wave_acc", None)
    zs = feedback.get("zero_shot_top1_rate", None)
    if acc is not None and acc_floor > 0.0 and float(acc) < acc_floor:
        return False, f"wave_acc<{acc_floor:.3f}"
    if zs is not None and zs_floor > 0.0 and float(zs) < zs_floor:
        return False, f"wave_zs_top1<{zs_floor:.3f}"
    return True, "pass"


def _sum_berkeley_refresh_samples(rows: Sequence[Dict[str, Any]]) -> int:
    total = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            samples_v = int(float(row.get("samples", 0)))
        except Exception:
            continue
        if samples_v <= 0:
            continue
        src = str(row.get("source", "")).strip().lower()
        if src == "fake_vector":
            continue
        total += int(samples_v)
    return int(max(0, total))


def _evaluate_berkeley_confidence_loss_gate(
    eval_stats: Optional[Dict[str, Any]],
    gate_config: Dict[str, Any],
    round_index: int,
    loss_history: Sequence[float],
    refresh_samples_seen: int = 0,
    refresh_samples_required: int = 0,
) -> Dict[str, Any]:
    _ = int(round_index)  # Reserved for future schedule variants; hard gate is time-invariant.
    _ = loss_history

    min_conf = max(0.0, float(gate_config.get("berkeley_min_confidence", 0.0)))
    min_macro_f1 = max(0.0, float(gate_config.get("berkeley_min_macro_f1", 0.0)))
    loss_floor = max(0.0, float(gate_config.get("berkeley_loss_min_floor", 0.0)))
    loss_target_cfg = float(gate_config.get("berkeley_loss_target", 0.0))
    loss_enabled = bool(loss_target_cfg > 0.0)
    loss_target = (None if not loss_enabled else float(max(loss_floor, loss_target_cfg)))

    refresh_required = max(0, int(refresh_samples_required))
    refresh_seen = max(0, int(refresh_samples_seen))
    refresh_coverage = (
        1.0
        if refresh_required <= 0
        else float(max(0.0, min(1.0, float(refresh_seen) / float(max(1, refresh_required)))))
    )
    refresh_pass = bool((refresh_required <= 0) or (refresh_seen >= refresh_required))

    if not isinstance(eval_stats, dict):
        confidence_pass = bool(min_conf <= 0.0)
        macro_f1_pass = bool(min_macro_f1 <= 0.0)
        loss_pass = bool(not loss_enabled)
        gate_pass = bool(refresh_pass and confidence_pass and macro_f1_pass and loss_pass)
        if not refresh_pass:
            reason = f"refresh<{refresh_required}"
        elif not confidence_pass:
            reason = f"conf<{min_conf:.3f}"
        elif not macro_f1_pass:
            reason = f"macro_f1<{min_macro_f1:.3f}"
        elif not loss_pass:
            reason = "loss_eval_missing"
        else:
            reason = "pass"
        mode = "hard_loss_full_refresh" if (loss_enabled or refresh_required > 0) else "confidence_only"
        return {
            "pass": bool(gate_pass),
            "mode": str(mode),
            "reason": str(reason),
            "confidence_pass": bool(confidence_pass),
            "macro_f1_pass": bool(macro_f1_pass),
            "loss_pass": bool(loss_pass),
            "refresh_pass": bool(refresh_pass),
            "refresh_samples_seen": int(refresh_seen),
            "refresh_samples_required": int(refresh_required),
            "refresh_coverage": float(refresh_coverage),
            "loss_target": loss_target,
            "confidence": None,
            "macro_f1": None,
            "loss": None,
        }

    conf = float(eval_stats.get("mean_confidence", 0.0))
    macro_f1 = float(eval_stats.get("macro_f1", 0.0))
    loss_v_raw = float(eval_stats.get("loss", float("nan")))
    loss_v = float(loss_v_raw) if math.isfinite(loss_v_raw) else float("nan")

    confidence_pass = (min_conf <= 0.0) or (conf >= min_conf)
    macro_f1_pass = (min_macro_f1 <= 0.0) or (macro_f1 >= min_macro_f1)
    if not loss_enabled:
        loss_pass = True
    elif not math.isfinite(loss_v):
        loss_pass = False
    else:
        loss_pass = bool(float(loss_v) <= float(loss_target))

    gate_pass = bool(refresh_pass and confidence_pass and macro_f1_pass and loss_pass)
    if not refresh_pass:
        reason = f"refresh<{refresh_required}"
    elif not confidence_pass:
        reason = f"conf<{min_conf:.3f}"
    elif not macro_f1_pass:
        reason = f"macro_f1<{min_macro_f1:.3f}"
    elif not loss_pass:
        reason = "loss_eval_missing" if (not math.isfinite(loss_v)) else f"loss>{float(loss_target):.4f}"
    else:
        reason = "pass"
    mode = "hard_loss_full_refresh" if (loss_enabled or refresh_required > 0) else "confidence_only"

    return {
        "pass": bool(gate_pass),
        "mode": str(mode),
        "reason": str(reason),
        "confidence_pass": bool(confidence_pass),
        "macro_f1_pass": bool(macro_f1_pass),
        "loss_pass": bool(loss_pass),
        "refresh_pass": bool(refresh_pass),
        "refresh_samples_seen": int(refresh_seen),
        "refresh_samples_required": int(refresh_required),
        "refresh_coverage": float(refresh_coverage),
        "loss_target": loss_target,
        "confidence": float(conf),
        "macro_f1": float(macro_f1),
        "loss": (None if not math.isfinite(loss_v) else float(max(0.0, loss_v))),
    }


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
        if "model_state_dict" in ckpt_obj and isinstance(ckpt_obj["model_state_dict"], dict):
            return ckpt_obj["model_state_dict"]
    if isinstance(ckpt_obj, dict):
        return ckpt_obj
    raise RuntimeError("Checkpoint does not contain a usable state dict.")


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]):
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            out[k[7:]] = v
        elif k.startswith("_orig_mod."):
            out[k[10:]] = v
        else:
            out[k] = v
    return out


def _is_expandable_classifier_head_key(key: str) -> bool:
    k = str(key)
    return k in ("head.2.weight", "head.2.bias", "fc.weight", "fc.bias")


def _try_partial_classifier_head_load(
    key: str,
    src_tensor: torch.Tensor,
    dst_tensor: torch.Tensor,
) -> Optional[torch.Tensor]:
    if not _is_expandable_classifier_head_key(key):
        return None
    if src_tensor.ndim != dst_tensor.ndim:
        return None
    if src_tensor.ndim == 2:
        if int(src_tensor.shape[1]) != int(dst_tensor.shape[1]):
            return None
        rows = min(int(src_tensor.shape[0]), int(dst_tensor.shape[0]))
        if rows <= 0:
            return None
        out = dst_tensor.detach().clone()
        out[:rows, :].copy_(src_tensor[:rows, :].to(device=dst_tensor.device, dtype=dst_tensor.dtype))
        return out
    if src_tensor.ndim == 1:
        rows = min(int(src_tensor.shape[0]), int(dst_tensor.shape[0]))
        if rows <= 0:
            return None
        out = dst_tensor.detach().clone()
        out[:rows].copy_(src_tensor[:rows].to(device=dst_tensor.device, dtype=dst_tensor.dtype))
        return out
    return None


def _apply_classifier_init(model: TinyConvClassifier, ckpt_path: str, scope: str):
    if not ckpt_path:
        return {"used": False, "loaded_keys": 0, "skipped_keys": 0, "path": ""}

    p = Path(ckpt_path)
    if not p.exists():
        raise FileNotFoundError(f"Classifier init checkpoint not found: {p}")

    blob = _torch_load_cpu(str(p))
    src_state = _strip_module_prefix(_extract_state_dict(blob))
    dst_state = model.state_dict()
    to_load = {}
    skipped = 0
    partial = 0

    for k, v in src_state.items():
        if scope == "features" and not k.startswith("features."):
            skipped += 1
            continue
        if k in dst_state and tuple(dst_state[k].shape) == tuple(v.shape):
            to_load[k] = v
        elif k in dst_state:
            patched = _try_partial_classifier_head_load(
                key=str(k),
                src_tensor=v,
                dst_tensor=dst_state[k],
            )
            if patched is not None:
                to_load[k] = patched
                partial += 1
            else:
                skipped += 1
        else:
            skipped += 1

    missing, unexpected = model.load_state_dict(to_load, strict=False)
    return {
        "used": True,
        "loaded_keys": len(to_load),
        "partial_keys": int(partial),
        "skipped_keys": skipped,
        "missing_after_load": len(missing),
        "unexpected_after_load": len(unexpected),
        "path": str(p.resolve()),
        "scope": scope,
    }


def _apply_model_init(model: nn.Module, ckpt_path: str):
    if not ckpt_path:
        return {"used": False, "loaded_keys": 0, "skipped_keys": 0, "path": ""}
    p = Path(ckpt_path)
    if not p.exists():
        return {"used": False, "loaded_keys": 0, "skipped_keys": 0, "path": str(p)}

    blob = _torch_load_cpu(str(p))
    src_state = _strip_module_prefix(_extract_state_dict(blob))
    dst_state = model.state_dict()
    to_load = {}
    skipped = 0
    partial = 0
    for k, v in src_state.items():
        if k in dst_state and tuple(dst_state[k].shape) == tuple(v.shape):
            to_load[k] = v
        elif k in dst_state:
            patched = _try_partial_classifier_head_load(
                key=str(k),
                src_tensor=v,
                dst_tensor=dst_state[k],
            )
            if patched is not None:
                to_load[k] = patched
                partial += 1
            else:
                skipped += 1
        else:
            skipped += 1
    missing, unexpected = model.load_state_dict(to_load, strict=False)
    return {
        "used": True,
        "loaded_keys": len(to_load),
        "partial_keys": int(partial),
        "skipped_keys": skipped,
        "missing_after_load": len(missing),
        "unexpected_after_load": len(unexpected),
        "path": str(p.resolve()),
    }


def _apply_state_dict(model: nn.Module, src_state: Dict[str, torch.Tensor], source_name: str):
    if not isinstance(src_state, dict):
        return {"used": False, "loaded_keys": 0, "skipped_keys": 0, "source": source_name}
    src_state = _strip_module_prefix(src_state)
    dst_state = model.state_dict()
    to_load = {}
    skipped = 0
    partial = 0
    for k, v in src_state.items():
        if k in dst_state and tuple(dst_state[k].shape) == tuple(v.shape):
            to_load[k] = v
        elif k in dst_state:
            patched = _try_partial_classifier_head_load(
                key=str(k),
                src_tensor=v,
                dst_tensor=dst_state[k],
            )
            if patched is not None:
                to_load[k] = patched
                partial += 1
            else:
                skipped += 1
        else:
            skipped += 1
    missing, unexpected = model.load_state_dict(to_load, strict=False)
    return {
        "used": True,
        "loaded_keys": len(to_load),
        "partial_keys": int(partial),
        "skipped_keys": skipped,
        "missing_after_load": len(missing),
        "unexpected_after_load": len(unexpected),
        "source": source_name,
    }


def _extract_deskew_prefilter_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    src_state = _strip_module_prefix(model.state_dict())
    out: Dict[str, torch.Tensor] = {}
    for k, v in src_state.items():
        key = str(k)
        if key.startswith("aux_filter_bundles.") or ("prefilter_" in key) or ("post_residual_" in key):
            out[key] = v.detach().cpu()
    return out


def _parse_transformer_filter_bundle_names(raw: Any) -> List[str]:
    text = str(raw).strip()
    if not text:
        return []
    out: List[str] = []
    seen: set = set()
    for token in text.split(","):
        key = str(token).strip().lower()
        if (not key) or (key in ("none", "off", "0")):
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_pipeline_checkpoint(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _save_training_segment_snapshot(
    enabled: bool,
    out_dir: Path,
    objective_mode: str,
    run_tag: str,
    segment: str,
    best_cfg: Optional[RenderConfig],
    classifier: Optional[nn.Module] = None,
    transformer: Optional[nn.Module] = None,
    generator: Optional[nn.Module] = None,
    discriminator: Optional[nn.Module] = None,
    wave_classifier: Optional[nn.Module] = None,
    classifier_history: Optional[Sequence[Dict]] = None,
    transformer_history: Optional[Sequence[Dict]] = None,
    generator_history: Optional[Sequence[Dict]] = None,
    wave_classifier_history: Optional[Sequence[Dict]] = None,
    orchestration_history: Optional[Sequence[Dict]] = None,
    refresh_history: Optional[Sequence[Dict]] = None,
    gate_history: Optional[Sequence[Dict]] = None,
    gate_status: Optional[Dict] = None,
    extra: Optional[Dict] = None,
):
    if not bool(enabled):
        return

    payload = {
        "run_tag": run_tag,
        "objective_mode": str(objective_mode),
        "segment": str(segment),
        "timestamp": time.time(),
    }
    if best_cfg is not None:
        try:
            payload["best_cfg"] = best_cfg.to_dict()
        except Exception:
            pass
    if classifier is not None:
        payload["classifier_state"] = classifier.state_dict()
    if transformer is not None:
        payload["transformer_state"] = transformer.state_dict()
    if generator is not None:
        payload["generator_state"] = generator.state_dict()
    if discriminator is not None:
        payload["discriminator_state"] = discriminator.state_dict()
    if wave_classifier is not None:
        payload["wave_classifier_state"] = wave_classifier.state_dict()
    if classifier_history is not None:
        payload["classifier_history"] = list(classifier_history)
    if transformer_history is not None:
        payload["transformer_history"] = list(transformer_history)
    if generator_history is not None:
        payload["generator_history"] = list(generator_history)
    if wave_classifier_history is not None:
        payload["wave_classifier_history"] = list(wave_classifier_history)
    if orchestration_history is not None:
        payload["orchestration_history"] = list(orchestration_history)
    if refresh_history is not None:
        payload["refresh_history"] = list(refresh_history)
    if gate_history is not None:
        payload["gate_history"] = list(gate_history)
    if isinstance(gate_status, dict):
        payload["gate_status"] = dict(gate_status)
    if isinstance(extra, dict):
        payload.update(extra)

    _save_pipeline_checkpoint(out_dir / "pipeline_checkpoint.pt", payload)
    if classifier is not None:
        torch.save({"state_dict": classifier.state_dict()}, out_dir / "classifier.pt")
    if transformer is not None:
        torch.save({"state_dict": transformer.state_dict()}, out_dir / "transformer.pt")
    if generator is not None:
        torch.save({"state_dict": generator.state_dict()}, out_dir / "generator.pt")
    if discriminator is not None:
        torch.save({"state_dict": discriminator.state_dict()}, out_dir / "discriminator.pt")
    if wave_classifier is not None:
        torch.save({"state_dict": wave_classifier.state_dict()}, out_dir / "wave_classifier.pt")


def _set_feature_freeze(model: TinyConvClassifier, freeze: bool):
    for name, p in model.named_parameters():
        if name.startswith("features."):
            p.requires_grad_(not freeze)


def _make_classifier_from_ckpt_meta(
    model_name: str,
    num_classes: int,
    classifier_base_ch: int = 64,
    classifier_max_ch: int = 384,
    classifier_context_blocks: int = 8,
    classifier_context_dropout: float = 0.05,
):
    if model_name == "tiny":
        return TinyConvClassifier(
            num_classes=num_classes,
            base_ch=max(16, int(classifier_base_ch)),
            max_ch=max(32, int(classifier_max_ch)),
            context_blocks=max(0, int(classifier_context_blocks)),
            context_dropout=float(classifier_context_dropout),
        )
    if model_name == "resnet18":
        from torchvision.models import resnet18

        m = resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    raise RuntimeError(f"Unsupported classifier model in checkpoint: {model_name}")


def _classifier_output_dim(model: nn.Module) -> int:
    if isinstance(model, TinyConvClassifier):
        last = model.head[-1]
        if isinstance(last, nn.Linear):
            return int(last.out_features)
    fc = getattr(model, "fc", None)
    if isinstance(fc, nn.Linear):
        return int(fc.out_features)
    raise RuntimeError(f"Unsupported classifier head for output-dim query: {type(model).__name__}")


def _expand_classifier_outputs(model: nn.Module, new_num_classes: int) -> nn.Module:
    target = max(1, int(new_num_classes))
    old_n = int(_classifier_output_dim(model))
    if old_n == target:
        return model
    if old_n > target:
        raise RuntimeError(f"Cannot shrink classifier outputs from {old_n} to {target}.")

    if isinstance(model, TinyConvClassifier):
        old_fc = model.head[-1]
        if not isinstance(old_fc, nn.Linear):
            raise RuntimeError("TinyConvClassifier head is missing final Linear layer.")
        new_fc = nn.Linear(int(old_fc.in_features), int(target))
        new_fc = new_fc.to(device=old_fc.weight.device, dtype=old_fc.weight.dtype)
        with torch.no_grad():
            new_fc.weight.zero_()
            if new_fc.bias is not None:
                new_fc.bias.zero_()
            new_fc.weight[:old_n].copy_(old_fc.weight.detach())
            if (new_fc.bias is not None) and (old_fc.bias is not None):
                new_fc.bias[:old_n].copy_(old_fc.bias.detach())
            if new_fc.bias is not None and old_n < target:
                new_fc.bias[old_n:].fill_(-4.0)
        model.head[-1] = new_fc
        return model

    fc = getattr(model, "fc", None)
    if isinstance(fc, nn.Linear):
        new_fc = nn.Linear(int(fc.in_features), int(target))
        new_fc = new_fc.to(device=fc.weight.device, dtype=fc.weight.dtype)
        with torch.no_grad():
            new_fc.weight.zero_()
            if new_fc.bias is not None:
                new_fc.bias.zero_()
            new_fc.weight[:old_n].copy_(fc.weight.detach())
            if (new_fc.bias is not None) and (fc.bias is not None):
                new_fc.bias[:old_n].copy_(fc.bias.detach())
            if new_fc.bias is not None and old_n < target:
                new_fc.bias[old_n:].fill_(-4.0)
        model.fc = new_fc
        return model

    raise RuntimeError(f"Unsupported classifier head for output expansion: {type(model).__name__}")


def _align_multilabel_targets_dim(y: torch.Tensor, out_dim: int) -> torch.Tensor:
    if y.ndim != 2:
        raise ValueError(f"Expected multilabel target tensor [B,C], got shape={tuple(y.shape)}")
    want = max(1, int(out_dim))
    got = int(y.shape[1])
    if got == want:
        return y
    if got < want:
        pad = torch.zeros((int(y.shape[0]), int(want - got)), device=y.device, dtype=y.dtype)
        return torch.cat([y, pad], dim=1)
    return y[:, :want]


def _build_semantic_supervision_targets(
    classifier: nn.Module,
    y_multihot: torch.Tensor,
    supervised_classes: int,
    semantic_target_temperature: float = 8.0,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    _ = semantic_target_temperature  # Retained for API compatibility; no centroid soft-target projection.
    if not isinstance(classifier, TinyConvClassifier):
        return None
    if not bool(int(classifier.label_embed_enabled.item())):
        return None
    bank = classifier.label_embed_bank
    if bank is None or int(bank.ndim) != 2 or int(bank.shape[0]) <= 0 or int(bank.shape[1]) <= 0:
        return None

    sup = max(1, int(supervised_classes))
    y_sup = _align_multilabel_targets_dim(y_multihot, out_dim=int(sup)).to(dtype=torch.float32)
    bank_t = bank.to(device=y_sup.device, dtype=torch.float32)
    if int(bank_t.shape[0]) < int(sup):
        return None
    # Use direct hard semantic supervision from provided multi-hot labels.
    # No centroid projection through label-text embeddings.
    y_sem = _align_multilabel_targets_dim(y_multihot, out_dim=int(bank_t.shape[0])).to(dtype=torch.float32)
    y_sem = torch.clamp(y_sem, 0.0, 1.0)
    return y_sem, y_sup


def _embed_multihot_distribution(weights: torch.Tensor, bank_norm: torch.Tensor) -> torch.Tensor:
    emb = weights @ bank_norm
    emb_norm = torch.linalg.norm(emb, dim=1, keepdim=True)
    if bool(torch.any(emb_norm <= 1e-6).item()):
        prior = bank_norm.mean(dim=0, keepdim=True)
        fill_mask = (emb_norm <= 1e-6).to(dtype=torch.float32)
        emb = (emb * (1.0 - fill_mask)) + (prior * fill_mask)
        emb_norm = torch.linalg.norm(emb, dim=1, keepdim=True)
    return emb / torch.clamp(emb_norm, min=1e-6)


def _classifier_supervision_loss(
    classifier: nn.Module,
    logits: torch.Tensor,
    y_multihot: torch.Tensor,
    supervised_dim: int,
    semantic_cosine_weight: float = CLASSIFIER_SEMANTIC_COSINE_WEIGHT,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    dim = max(1, int(supervised_dim))
    logits_sup = logits
    if int(logits_sup.shape[1]) > int(dim):
        logits_sup = logits_sup[:, : int(dim)]
    y_sup = _align_multilabel_targets_dim(y_multihot, out_dim=int(logits_sup.shape[1])).to(dtype=torch.float32)
    logits_sup_f = logits_sup.to(dtype=torch.float32)
    bce = F.binary_cross_entropy_with_logits(logits_sup_f, y_sup, reduction="mean")

    use_geom = False
    cos_loss = torch.zeros((), device=logits_sup_f.device, dtype=torch.float32)
    w_geom = float(max(0.0, semantic_cosine_weight))
    if (
        w_geom > 0.0
        and isinstance(classifier, TinyConvClassifier)
        and bool(int(classifier.label_embed_enabled.item()))
        and int(classifier.label_embed_bank.ndim) == 2
        and int(classifier.label_embed_bank.shape[0]) >= int(logits_sup_f.shape[1])
        and int(classifier.label_embed_bank.shape[1]) > 0
    ):
        bank = classifier.label_embed_bank[: int(logits_sup_f.shape[1]), :].to(device=logits_sup_f.device, dtype=torch.float32)
        bank = F.normalize(bank, dim=1, eps=1e-6)
        pred_probs = torch.sigmoid(logits_sup_f)
        pred_emb = _embed_multihot_distribution(weights=pred_probs, bank_norm=bank)
        tgt_emb = _embed_multihot_distribution(weights=y_sup, bank_norm=bank)
        cos_loss = (1.0 - torch.sum(pred_emb * tgt_emb, dim=1)).mean()
        use_geom = True

    total = bce + (float(w_geom) * cos_loss if use_geom else 0.0)
    return total, logits_sup, y_sup, {
        "bce": float(bce.detach().item()),
        "cos": float(cos_loss.detach().item()) if use_geom else 0.0,
        "used_geometry": bool(use_geom),
        "geometry_weight": float(w_geom if use_geom else 0.0),
    }


def _default_berkeley_class_names() -> List[str]:
    fallback = [
        "black",
        "white",
        "gray",
        "grey",
        "red",
        "green",
        "blue",
        "yellow",
        "cyan",
        "magenta",
        "brown",
        "noise",
        "white noise",
        "pink noise",
        "brown noise",
        "red noise",
        "blue noise",
        "violet noise",
        "gray noise",
        "uniform white noise",
        "gaussian white noise",
        "front",
        "back",
        "left",
        "right",
        "top",
        "bottom",
    ]
    try:
        from berkeley_sbd_pretrain import VOC20_CLASSES as voc20_classes
    except ModuleNotFoundError:
        try:
            from toys_to_survive_development.berkeley_sbd_pretrain import VOC20_CLASSES as voc20_classes
        except Exception:
            voc20_classes = fallback
    except Exception:
        voc20_classes = fallback
    names = [str(x).strip() for x in list(voc20_classes) if str(x).strip()]
    if len(names) < 2:
        return list(fallback)
    return names


def _load_berkeley_classifier_for_metric(
    ckpt_path: str,
    device: torch.device,
    classifier_base_ch: int = 64,
    classifier_max_ch: int = 384,
    classifier_context_blocks: int = 8,
    classifier_context_dropout: float = 0.05,
    classifier_init_required: bool = False,
):
    def _build_scratch_classifier(fallback_reason: str):
        class_names = _default_berkeley_class_names()
        num_classes = max(2, int(len(class_names)))
        model_name = "tiny"
        model = _make_classifier_from_ckpt_meta(
            model_name=model_name,
            num_classes=num_classes,
            classifier_base_ch=classifier_base_ch,
            classifier_max_ch=classifier_max_ch,
            classifier_context_blocks=classifier_context_blocks,
            classifier_context_dropout=classifier_context_dropout,
        )
        model = model.to(device)
        model.eval()
        info = {
            "path": "",
            "model_name": model_name,
            "num_classes": int(num_classes),
            "class_names": list(class_names),
            "loaded_keys": 0,
            "partial_keys": 0,
            "skipped_keys": 0,
            "missing_after_load": 0,
            "unexpected_after_load": 0,
            "init_mode": "scratch_fallback",
            "fallback_reason": str(fallback_reason),
        }
        return model, info

    path_raw = str(ckpt_path).strip()
    if not path_raw:
        if bool(classifier_init_required):
            raise RuntimeError(
                "Berkeley metric mode requires --classifier-init-ckpt when --classifier-init-required is set."
            )
        return _build_scratch_classifier("checkpoint_not_provided")

    p = Path(path_raw)
    if not p.exists():
        if bool(classifier_init_required):
            raise FileNotFoundError(f"Classifier checkpoint not found: {p}")
        return _build_scratch_classifier(f"checkpoint_missing:{path_raw}")

    try:
        blob = _torch_load_cpu(str(p))
    except Exception as e:
        if bool(classifier_init_required):
            raise
        return _build_scratch_classifier(f"checkpoint_unloadable:{type(e).__name__}:{e}")

    model_name = str(blob.get("model_name", "tiny"))
    class_names_raw = blob.get("class_names", [])
    if isinstance(class_names_raw, (list, tuple)):
        class_names = [str(x) for x in class_names_raw]
    else:
        class_names = []
    num_classes = int(blob.get("num_classes", len(class_names) if len(class_names) > 0 else 20))
    model = _make_classifier_from_ckpt_meta(
        model_name=model_name,
        num_classes=num_classes,
        classifier_base_ch=classifier_base_ch,
        classifier_max_ch=classifier_max_ch,
        classifier_context_blocks=classifier_context_blocks,
        classifier_context_dropout=classifier_context_dropout,
    )
    state = _strip_module_prefix(_extract_state_dict(blob))
    load_info = _apply_state_dict(
        model=model,
        src_state=state,
        source_name=f"berkeley_metric_ckpt:{p}",
    )
    model = model.to(device)
    model.eval()

    info = {
        "path": str(p.resolve()),
        "model_name": model_name,
        "num_classes": num_classes,
        "class_names": class_names,
        "loaded_keys": int(load_info.get("loaded_keys", 0)),
        "partial_keys": int(load_info.get("partial_keys", 0)),
        "skipped_keys": int(load_info.get("skipped_keys", 0)),
        "missing_after_load": int(load_info.get("missing_after_load", 0)),
        "unexpected_after_load": int(load_info.get("unexpected_after_load", 0)),
        "init_mode": "checkpoint",
        "fallback_reason": "",
    }
    return model, info


@torch.no_grad()
def _score_config_with_classifier(
    records: Sequence[WaveRecord],
    indices: Sequence[int],
    cfg: RenderConfig,
    image_hw: Tuple[int, int],
    classifier: nn.Module,
    device: torch.device,
    batch_size: int,
    score_topk: int,
    score_threshold: float,
    score_w_topk: float,
    score_w_cov: float,
    score_w_mean: float,
    amp_enabled: bool = False,
    amp_dtype: str = "float16",
    channels_last: bool = False,
    active_classes: int = 0,
):
    if len(indices) == 0:
        return {
            "score": float("-inf"),
            "topk_mean": 0.0,
            "hard_coverage": 0.0,
            "mean_prob": 0.0,
            "num_images": 0,
        }

    logits_all = []
    amp_dtype_t = _resolve_amp_dtype(amp_dtype) if amp_enabled else torch.float16
    for i in range(0, len(indices), max(1, int(batch_size))):
        batch_idx = indices[i : i + max(1, int(batch_size))]
        xb = []
        for idx in batch_idx:
            img_u8, _ = render_record(records[idx], cfg)
            xb.append(image_u8_to_tensor(img_u8, image_hw=image_hw))
        x = torch.stack(xb, dim=0).to(device, non_blocking=True)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with _autocast_context(device=device, enabled=amp_enabled, amp_dtype_t=amp_dtype_t):
            logits = classifier(x)
            if int(active_classes) > 0 and int(logits.shape[1]) > int(active_classes):
                logits = logits[:, : int(active_classes)]
        logits_all.append(logits.detach().cpu())

    logits_cat = torch.cat(logits_all, dim=0)
    probs = torch.sigmoid(logits_cat)
    ent = -(probs * torch.log2(torch.clamp(probs, 1e-8, 1.0)) + (1.0 - probs) * torch.log2(torch.clamp(1.0 - probs, 1e-8, 1.0)))
    s = multilabel_feature_score(
        logits_cat,
        topk=score_topk,
        threshold=score_threshold,
        w_topk=score_w_topk,
        w_cov=score_w_cov,
        w_mean=score_w_mean,
    )
    return {
        "score": float(s["score"].item()),
        "topk_mean": float(s["topk_mean"].item()),
        "hard_coverage": float(s["hard_coverage"].item()),
        "mean_prob": float(s["mean_prob"].item()),
        "mean_entropy": float(ent.mean().item()),
        "num_images": int(logits_cat.shape[0]),
    }


class _PayloadCacheMemmapDataset(Dataset):
    def __init__(
        self,
        images_path: str,
        labels_path: str,
        picks: np.ndarray,
        labels_index_mode: str = "global_pick",
    ):
        self._images_path = str(images_path)
        self._labels_path = str(labels_path)
        self._picks = np.asarray(picks, dtype=np.int64)
        mode = str(labels_index_mode).strip().lower()
        if mode not in ("global_pick", "local_dataset"):
            raise ValueError(
                "Unsupported labels_index_mode for _PayloadCacheMemmapDataset: "
                f"{labels_index_mode!r} (expected 'global_pick' or 'local_dataset')."
            )
        self._labels_index_mode = str(mode)
        self._images_mm = None
        self._labels_mm = None

    def __getstate__(self):
        # Windows DataLoader workers use spawn + pickle. Never pickle open memmaps.
        state = dict(self.__dict__)
        state["_images_mm"] = None
        state["_labels_mm"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._images_mm = None
        self._labels_mm = None

    def _ensure_open(self):
        if self._images_mm is None:
            self._images_mm = np.load(self._images_path, mmap_mode="r")
        if self._labels_mm is None:
            self._labels_mm = np.load(self._labels_path, mmap_mode="r")

    def __len__(self) -> int:
        return int(self._picks.shape[0])

    def __getitem__(self, i: int):
        self._ensure_open()
        row_i = int(i)
        j = int(self._picks[int(row_i)])
        x = np.asarray(self._images_mm[j], dtype=np.float32)
        if int(x.ndim) != 3:
            raise RuntimeError(f"Invalid Berkeley refresh cache image row shape: {tuple(x.shape)}")
        if int(x.shape[0]) != 3 and int(x.shape[2]) == 3:
            x = np.transpose(x, (2, 0, 1)).astype(np.float32, copy=False)
        if int(x.shape[0]) != 3:
            raise RuntimeError(f"Invalid Berkeley refresh cache image row shape: {tuple(x.shape)}")
        if str(self._labels_index_mode) == "local_dataset":
            label_row = int(row_i)
        else:
            label_row = int(j)
        if label_row < 0 or label_row >= int(self._labels_mm.shape[0]):
            raise IndexError(
                "Payload memmap label index out of bounds: "
                f"mode={self._labels_index_mode} label_row={int(label_row)} "
                f"labels_rows={int(self._labels_mm.shape[0])} "
                f"dataset_row={int(row_i)} pick={int(j)}"
            )
        y = np.asarray(self._labels_mm[label_row], dtype=np.float32).reshape(-1)
        return (
            torch.from_numpy(np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)),
            torch.from_numpy(np.clip(y, 0.0, 1.0).astype(np.float32, copy=False)),
        )


def _split_payload_cache_indices(
    source_rows: Sequence[str],
    seed: int,
    external_val_fraction: float = 0.20,
) -> Tuple[List[int], List[int], Dict[str, Dict[str, Any]]]:
    n_rows = int(len(source_rows))
    by_source: Dict[str, List[int]] = {}
    for i in range(int(n_rows)):
        src = re.sub(r"\s+", " ", str(source_rows[int(i)])).strip().lower() or "unknown_source"
        by_source.setdefault(src, []).append(int(i))

    val_idx: List[int] = []
    train_idx: List[int] = []
    source_stats: Dict[str, Dict[str, Any]] = {}
    ext_frac = float(max(0.0, min(1.0, float(external_val_fraction))))
    base_seed = max(0, int(seed))
    for src in sorted(by_source.keys()):
        idxs = list(by_source.get(src, []))
        n_src = int(len(idxs))
        if n_src <= 0:
            continue
        src_key_seed = int(
            hashlib.sha256(f"{src}|{base_seed}".encode("utf-8")).digest()[0]
        )
        rng = np.random.default_rng(int(base_seed) + int(src_key_seed))
        arr = np.asarray(idxs, dtype=np.int64)
        rng.shuffle(arr)
        if src == "berkeley_sbd_train":
            v = np.zeros((0,), dtype=np.int64)
            t = arr
            rule = "gate=exclude_berkeley_train refresh=all_berkeley_train"
        elif src == "berkeley_sbd_val":
            v = arr
            t = np.zeros((0,), dtype=np.int64)
            rule = "gate=all_berkeley_val refresh=exclude_berkeley_val"
        else:
            take_val = int(round(float(n_src) * ext_frac))
            if float(ext_frac) > 0.0:
                take_val = max(1, int(take_val))
            take_val = max(0, min(int(n_src), int(take_val)))
            v = arr[: int(take_val)]
            t = arr[int(take_val) :]
            rule = f"external_split gate_fraction={ext_frac:.2f}"
        val_idx.extend([int(i) for i in v.tolist()])
        train_idx.extend([int(i) for i in t.tolist()])
        source_stats[str(src)] = {
            "rows": int(n_src),
            "gate_selected": int(v.size),
            "refresh_selected": int(t.size),
            "rule": str(rule),
        }
    val_idx = sorted(set(int(i) for i in val_idx if 0 <= int(i) < int(n_rows)))
    train_idx = sorted(set(int(i) for i in train_idx if 0 <= int(i) < int(n_rows)))
    return train_idx, val_idx, source_stats


def _build_berkeley_refresh_loader(
    data_root: str,
    image_size: int,
    auto_install_scipy: bool,
    batch_size: int,
    num_workers: int,
    max_train: int,
    seed: int,
    device: torch.device,
    external_val_fraction: float = 0.20,
    validation_split_seed: int = 0,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
):
    del auto_install_scipy  # Legacy path intentionally disabled for refresh.
    root = Path(str(data_root).strip() or "toys_to_survive_development/data/berkeley_sbd")
    size = max(8, int(image_size))
    cache_dir = root / "cache" / f"payload_bank_rgb{int(size)}_v2"
    manifest_path = cache_dir / "manifest.json"
    images_path = cache_dir / "images.npy"
    labels_path = cache_dir / "labels.npy"
    required = [manifest_path, images_path, labels_path]
    missing = [str(p) for p in required if not p.exists()]
    if len(missing) > 0:
        raise RuntimeError(
            "Berkeley refresh requires pre-washed payload cache files and no longer falls back to legacy "
            f"prepare_sbd_multilabel. Missing: {missing}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to read Berkeley payload cache manifest: {manifest_path} ({e})") from e
    if int(manifest.get("version", -1)) != 2:
        raise RuntimeError(
            "Unsupported Berkeley payload cache version for refresh loader "
            f"(expected=2, got={manifest.get('version', None)})."
        )
    if int(manifest.get("image_size", -1)) != int(size):
        raise RuntimeError(
            "Berkeley payload cache image-size mismatch for refresh loader "
            f"(cache={manifest.get('image_size', None)} requested={int(size)})."
        )
    x_mem = np.load(str(images_path), mmap_mode="r")
    y_mem = np.load(str(labels_path), mmap_mode="r")
    if int(getattr(x_mem, "ndim", 0)) != 4 or int(getattr(y_mem, "ndim", 0)) != 2:
        raise RuntimeError(
            "Invalid Berkeley payload cache tensor shapes "
            f"(images={getattr(x_mem, 'shape', None)} labels={getattr(y_mem, 'shape', None)})."
        )
    n_rows = min(int(x_mem.shape[0]), int(y_mem.shape[0]))
    if n_rows <= 0:
        raise RuntimeError("Berkeley payload cache is empty for refresh loader.")

    sources_path = cache_dir / "sources.json"
    source_rows: List[str] = []
    if sources_path.exists():
        try:
            raw_sources = json.loads(sources_path.read_text(encoding="utf-8"))
            if isinstance(raw_sources, list):
                source_rows = [str(x) for x in raw_sources]
        except Exception:
            source_rows = []
    if int(len(source_rows)) < int(n_rows):
        source_rows.extend(["unknown_source"] * int(max(0, int(n_rows) - int(len(source_rows)))))
    if int(len(source_rows)) > int(n_rows):
        source_rows = source_rows[: int(n_rows)]

    split_seed = int(validation_split_seed) if int(validation_split_seed) != 0 else int(seed)
    train_idx, _, source_stats = _split_payload_cache_indices(
        source_rows=source_rows,
        seed=int(split_seed),
        external_val_fraction=float(external_val_fraction),
    )
    if int(len(train_idx)) <= 0:
        raise RuntimeError("Refresh loader has no non-validation rows after split.")
    idx = np.asarray(train_idx, dtype=np.int64)
    if int(max_train) > 0 and int(idx.shape[0]) > int(max_train):
        rng = np.random.default_rng(max(0, int(seed)))
        idx = rng.choice(idx, size=int(max_train), replace=False).astype(np.int64)
    else:
        rng = np.random.default_rng(max(0, int(seed)))
        rng.shuffle(idx)
    ds = _PayloadCacheMemmapDataset(
        images_path=str(images_path),
        labels_path=str(labels_path),
        picks=idx,
        labels_index_mode="global_pick",
    )
    loader = DataLoader(
        ds,
        batch_size=max(1, int(batch_size)),
        shuffle=True,
        num_workers=max(0, int(num_workers)),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        **_dataloader_perf_kwargs(
            num_workers=max(0, int(num_workers)),
            persistent_workers=bool(persistent_workers),
            prefetch_factor=int(prefetch_factor),
        ),
    )
    setattr(loader, "_refresh_source_stats", source_stats)
    return loader, int(len(ds))


def _build_berkeley_refresh_cache(
    loader: DataLoader,
    device: torch.device,
    cache_batches: int,
    cache_device: str,
    channels_last: bool,
):
    n_batches = max(0, int(cache_batches))
    mode = str(cache_device).strip().lower()
    if n_batches <= 0 or mode == "none":
        return None

    if mode == "cuda":
        target = torch.device("cuda:0" if device.type == "cuda" else "cpu")
    elif mode == "cpu":
        target = torch.device("cpu")
    else:
        target = torch.device("cuda:0" if device.type == "cuda" else "cpu")

    xs = []
    ys = []
    for i, (xb, yb) in enumerate(loader, start=1):
        xs.append(xb)
        ys.append(yb)
        if i >= n_batches:
            break

    if len(xs) == 0:
        return None

    x = torch.cat(xs, dim=0).contiguous()
    y = torch.cat(ys, dim=0).contiguous()
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)

    if target.type == "cuda":
        x = x.to(target, non_blocking=True)
        y = y.to(target, non_blocking=True)
    elif device.type == "cuda":
        x = x.pin_memory()
        y = y.pin_memory()

    return {
        "x": x,
        "y": y,
        "num_samples": int(y.shape[0]),
        "num_batches": len(xs),
        "device": target.type,
    }


def _auto_berkeley_refresh_batch_size(
    cache_x: torch.Tensor,
    cache_y: torch.Tensor,
    device: torch.device,
    vram_fraction: float,
    activation_multiplier: float,
    max_cap: int,
):
    if int(cache_x.shape[0]) <= 0:
        return 1
    max_cap = int(max_cap)
    hard_cap = int(cache_x.shape[0]) if max_cap <= 0 else min(int(cache_x.shape[0]), max_cap)
    if device.type != "cuda":
        return max(1, hard_cap)
    try:
        free_bytes, _ = torch.cuda.mem_get_info(device)
    except Exception:
        try:
            free_bytes, _ = torch.cuda.mem_get_info()
        except Exception:
            free_bytes = 0

    sample_bytes = int(cache_x[0].numel()) * int(cache_x.element_size())
    if cache_y is not None:
        sample_bytes += int(cache_y[0].numel()) * int(cache_y.element_size())
    sample_bytes = max(1, sample_bytes)
    mult = max(1.0, float(activation_multiplier))
    target = int(max(0.0, min(0.99, float(vram_fraction))) * float(max(0, free_bytes)))
    if target <= 0:
        return max(1, min(64, hard_cap))
    est = int(target // int(sample_bytes * mult))
    return max(1, min(est, hard_cap))


def _build_berkeley_gate_val_loader(
    data_root: str,
    image_size: int,
    auto_install_scipy: bool,
    batch_size: int,
    num_workers: int,
    max_val: int,
    seed: int,
    device: torch.device,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
):
    try:
        from berkeley_sbd_pretrain import prepare_sbd_multilabel
    except ModuleNotFoundError:
        from toys_to_survive_development.berkeley_sbd_pretrain import prepare_sbd_multilabel

    _, val_ds = prepare_sbd_multilabel(
        data_root=data_root,
        image_size=image_size,
        auto_install_scipy=auto_install_scipy,
    )
    sampler = None
    shuffle = False
    sample_count = int(len(val_ds))
    if max_val > 0 and len(val_ds) > int(max_val):
        sample_count = int(max_val)
        g = torch.Generator()
        g.manual_seed(max(0, int(seed)))
        sampler = torch.utils.data.RandomSampler(
            data_source=val_ds,
            replacement=False,
            num_samples=int(sample_count),
            generator=g,
        )
        shuffle = False

    loader = DataLoader(
        val_ds,
        batch_size=max(1, int(batch_size)),
        shuffle=bool(shuffle),
        sampler=sampler,
        num_workers=max(0, int(num_workers)),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        **_dataloader_perf_kwargs(
            num_workers=max(0, int(num_workers)),
            persistent_workers=bool(persistent_workers),
            prefetch_factor=int(prefetch_factor),
        ),
    )
    return loader, int(sample_count)


def _schedule_payload_gate_rows_by_active_terms(
    payload_terms_rows: Sequence[Sequence[str]],
    payload_global_picks: Sequence[int],
    payload_labels_mm: np.ndarray,
    supervised_class_names: Sequence[str],
    active_terms_lc: Sequence[str],
    active_vocab_lc: Sequence[str],
    require_all_supervised_active: bool = True,
    allow_unmapped_rows: bool = True,
) -> Tuple[List[int], List[int], List[List[str]], Dict[str, Any]]:
    n_rows = min(int(len(payload_terms_rows)), int(len(payload_global_picks)))
    active_order = list(
        dict.fromkeys(
            [
                re.sub(r"\s+", " ", str(t)).strip().lower()
                for t in active_terms_lc
                if re.sub(r"\s+", " ", str(t)).strip()
            ]
        )
    )
    active_set = set(active_order)
    active_vocab_set = {
        re.sub(r"\s+", " ", str(t)).strip().lower()
        for t in active_vocab_lc
        if re.sub(r"\s+", " ", str(t)).strip()
    }
    sup_name_lc = [
        re.sub(r"\s+", " ", str(x)).strip().lower()
        for x in supervised_class_names
        if re.sub(r"\s+", " ", str(x)).strip()
    ]
    order_lut = {str(t): int(i) for i, t in enumerate(active_order)}
    buckets: Dict[str, List[Tuple[int, int, List[str], List[str], int]]] = {}
    fallback_rows: List[Tuple[int, int, List[str], List[str], int]] = []
    skipped_no_active = 0
    skipped_inactive_supervised = 0
    selected_sup_positive_counts: List[int] = []
    selected_berkeley_rows = 0

    for ri in range(int(n_rows)):
        src_idx = int(payload_global_picks[int(ri)])
        if not (0 <= int(src_idx) < int(payload_labels_mm.shape[0])):
            raise RuntimeError(
                "Stage-2 payload gate supervised-label index out of bounds while scheduling rows: "
                f"row={int(ri)} pick={int(src_idx)} labels_rows={int(payload_labels_mm.shape[0])}"
            )
        raw_terms = payload_terms_rows[int(ri)]
        terms = _normalize_vocab_terms([str(x) for x in list(raw_terms)]) if isinstance(raw_terms, (list, tuple)) else []
        if len(terms) <= 0:
            raise RuntimeError(
                "Stage-2 payload gate encountered empty original terms row during scheduling; "
                f"row={int(ri)} global_pick={int(src_idx)}"
            )
        row_lc = {
            re.sub(r"\s+", " ", str(x)).strip().lower()
            for x in terms
            if re.sub(r"\s+", " ", str(x)).strip()
        }
        base_sup = np.asarray(payload_labels_mm[int(src_idx)], dtype=np.float32).reshape(-1)
        sup_take = min(int(len(sup_name_lc)), int(base_sup.size))
        sup_pos_idx = (
            np.where(np.asarray(base_sup[: int(sup_take)], dtype=np.float32) >= 0.5)[0].astype(np.int64).tolist()
            if int(sup_take) > 0
            else []
        )
        sup_terms_lc = []
        for sp in sup_pos_idx:
            if 0 <= int(sp) < int(len(sup_name_lc)):
                nm = str(sup_name_lc[int(sp)])
                if nm:
                    sup_terms_lc.append(nm)
        sup_terms_lc = list(dict.fromkeys(sup_terms_lc))
        missing_sup_vocab = [str(t) for t in sup_terms_lc if str(t) not in active_vocab_set]
        if len(missing_sup_vocab) > 0:
            raise RuntimeError(
                "Stage-2 payload gate row contains supervised Berkeley labels not present in active vocabulary. "
                f"row={int(ri)} pick={int(src_idx)} missing={','.join(missing_sup_vocab[:8])}"
            )
        row_is_berkeley = bool("berkeley sbd dataset" in row_lc)
        if bool(row_is_berkeley):
            if int(len(sup_terms_lc)) <= 0:
                raise RuntimeError(
                    "Stage-2 payload gate cannot pass Berkeley rows without supervised labels. "
                    f"row={int(ri)} pick={int(src_idx)}"
                )
            if bool(require_all_supervised_active) and any(str(t) not in active_set for t in sup_terms_lc):
                skipped_inactive_supervised += 1
                if not bool(allow_unmapped_rows):
                    continue
            selected_berkeley_rows += 1
            selected_sup_positive_counts.append(int(len(sup_terms_lc)))
        row_active_terms = [
            str(t)
            for t in active_order
            if (str(t) in row_lc) or (str(t) in set(sup_terms_lc))
        ]
        if int(len(row_active_terms)) <= 0:
            skipped_no_active += 1
            if not bool(allow_unmapped_rows):
                continue
            anchor = ""
        else:
            anchor = str(row_active_terms[0])
        row_payload = (
            int(ri),
            int(src_idx),
            list(terms),
            list(row_active_terms),
            int(len(sup_terms_lc)),
        )
        if anchor in active_set and int(len(anchor)) > 0:
            buckets.setdefault(str(anchor), []).append(row_payload)
        else:
            fallback_rows.append(row_payload)

    for key in list(buckets.keys()):
        rows = list(buckets.get(str(key), []))
        rows.sort(
            key=lambda row: (
                int(order_lut.get(str(row[3][0]) if len(row[3]) > 0 else str(key), 10**9)),
                -int(len(row[3])),
                -int(row[4]),
                int(row[0]),
            )
        )
        buckets[str(key)] = rows
    fallback_rows.sort(key=lambda row: (-int(len(row[3])), -int(row[4]), int(row[0])))

    ordered_rows: List[Tuple[int, int, List[str], List[str], int]] = []
    used_local: set = set()
    progress = True
    while bool(progress):
        progress = False
        for term in active_order:
            q = buckets.get(str(term), [])
            while int(len(q)) > 0 and int(q[0][0]) in used_local:
                q.pop(0)
            if int(len(q)) <= 0:
                continue
            row = q.pop(0)
            if int(row[0]) in used_local:
                continue
            used_local.add(int(row[0]))
            ordered_rows.append(row)
            progress = True
        if not bool(progress):
            break
    leftovers: List[Tuple[int, int, List[str], List[str], int]] = []
    for rows in buckets.values():
        leftovers.extend([row for row in rows if int(row[0]) not in used_local])
    leftovers.extend([row for row in fallback_rows if int(row[0]) not in used_local])
    leftovers.sort(
        key=lambda row: (
            int(order_lut.get(str(row[3][0]) if len(row[3]) > 0 else "", 10**9)),
            -int(len(row[3])),
            -int(row[4]),
            int(row[0]),
        )
    )
    ordered_rows.extend(leftovers)

    out_local_rows = [int(r[0]) for r in ordered_rows]
    out_global_picks = [int(r[1]) for r in ordered_rows]
    out_terms = [list(r[2]) for r in ordered_rows]
    info = {
        "rows_total": int(n_rows),
        "rows_selected": int(len(out_local_rows)),
        "rows_skipped_no_active": int(skipped_no_active),
        "rows_skipped_inactive_supervised": int(skipped_inactive_supervised),
        "active_terms": int(len(active_order)),
        "buckets": int(len([k for k, v in buckets.items() if int(len(v)) > 0])),
        "berkeley_rows_selected": int(selected_berkeley_rows),
        "berkeley_sup_min": int(np.min(selected_sup_positive_counts)) if int(len(selected_sup_positive_counts)) > 0 else 0,
        "berkeley_sup_mean": float(np.mean(selected_sup_positive_counts)) if int(len(selected_sup_positive_counts)) > 0 else 0.0,
        "berkeley_sup_max": int(np.max(selected_sup_positive_counts)) if int(len(selected_sup_positive_counts)) > 0 else 0,
        "mode": "deterministic_round_robin_by_active_terms",
    }
    return out_local_rows, out_global_picks, out_terms, info


def _schedule_semantic_gate_indices(
    targets: Sequence[np.ndarray],
    class_names: Sequence[str],
    active_terms: Sequence[str],
    seed: int,
    max_samples: int = 0,
    activation_threshold: float = 0.55,
) -> Tuple[List[int], Dict[str, Any]]:
    n = int(len(targets))
    expected_dim = max(1, int(len(class_names)))
    if n <= 0:
        return [], {
            "rows": 0,
            "target_dim_expected": int(expected_dim),
            "target_dim_min": 0,
            "target_dim_max": 0,
            "active_terms": 0,
            "active_term_hits": {},
            "bucketed_rows": 0,
            "selected_rows": 0,
            "max_samples": int(max_samples),
            "threshold": float(activation_threshold),
        }
    y_rows: List[np.ndarray] = []
    row_sizes: List[int] = []
    for i in range(int(n)):
        row = np.asarray(targets[int(i)], dtype=np.float32).reshape(-1)
        y_rows.append(row)
        row_sizes.append(int(row.size))
    bad_dims = sorted({int(sz) for sz in row_sizes if int(sz) != int(expected_dim)})
    if len(bad_dims) > 0:
        raise RuntimeError(
            "Semantic gate scheduler requires a single label width that matches the sentence-transformer "
            f"class width: expected={int(expected_dim)} got_dims={bad_dims}"
        )
    y = np.stack(y_rows, axis=0).astype(np.float32, copy=False)
    y = np.clip(y[:, : int(expected_dim)], 0.0, 1.0).astype(np.float32, copy=False)

    class_map = _semantic_term_index_map(class_names)
    active_terms_norm = _normalize_vocab_terms(active_terms)
    active_indices: List[int] = []
    for term in active_terms_norm:
        key = re.sub(r"\s+", " ", str(term)).strip().lower()
        if not key:
            continue
        idx = class_map.get(key, None)
        if idx is None:
            continue
        if int(idx) < 0 or int(idx) >= int(y.shape[1]):
            continue
        active_indices.append(int(idx))
    active_indices = list(dict.fromkeys(active_indices))

    _ = int(seed)  # Kept for API compatibility; scheduling is deterministic and staged.
    threshold = float(max(0.0, min(1.0, float(activation_threshold))))
    active_hits: Dict[str, int] = {}
    single_term_buckets: Dict[int, List[Tuple[float, int]]] = {int(idx): [] for idx in active_indices}
    signature_buckets: Dict[Tuple[int, ...], List[Tuple[float, int]]] = {}
    fallback_rows: List[Tuple[float, int]] = []
    rows_with_active_support = 0
    for idx in active_indices:
        name = str(class_names[int(idx)]) if int(idx) < int(len(class_names)) else f"class_{int(idx)}"
        active_hits[str(name)] = 0
    for ri in range(int(n)):
        row = np.asarray(y[int(ri)], dtype=np.float32).reshape(-1)
        support: List[int] = []
        for idx in active_indices:
            if float(row[int(idx)]) >= float(threshold):
                support.append(int(idx))
        if len(support) <= 0 and len(active_indices) > 0:
            vals = np.asarray(row[np.asarray(active_indices, dtype=np.int64)], dtype=np.float32).reshape(-1)
            if int(vals.size) > 0:
                top_local = int(np.argmax(vals))
                top_score = float(vals[int(top_local)])
                if float(top_score) > 1e-6:
                    support = [int(active_indices[int(top_local)])]
        if len(support) <= 0:
            fb_score = 0.0
            if len(active_indices) > 0:
                vals = np.asarray(row[np.asarray(active_indices, dtype=np.int64)], dtype=np.float32).reshape(-1)
                if int(vals.size) > 0:
                    fb_score = float(np.max(vals))
            fallback_rows.append((float(fb_score), int(ri)))
            continue
        rows_with_active_support += 1
        sig = tuple(sorted({int(x) for x in support}))
        sig_score = float(np.mean([float(row[int(ix)]) for ix in sig])) if len(sig) > 0 else 0.0
        signature_buckets.setdefault(sig, []).append((float(sig_score), int(ri)))
        for idx in sig:
            name = str(class_names[int(idx)]) if int(idx) < int(len(class_names)) else f"class_{int(idx)}"
            active_hits[str(name)] = int(active_hits.get(str(name), 0)) + 1
        if len(sig) == 1:
            s_idx = int(sig[0])
            single_term_buckets.setdefault(int(s_idx), []).append((float(row[int(s_idx)]), int(ri)))

    for idx in list(single_term_buckets.keys()):
        rows = list(single_term_buckets.get(int(idx), []))
        rows.sort(key=lambda x: (-float(x[0]), int(x[1])))
        single_term_buckets[int(idx)] = rows
    for sig in list(signature_buckets.keys()):
        rows = list(signature_buckets.get(sig, []))
        rows.sort(key=lambda x: (-float(x[0]), int(x[1])))
        signature_buckets[sig] = rows
    fallback_rows.sort(key=lambda x: (-float(x[0]), int(x[1])))

    ordered: List[int] = []
    used: set = set()
    stage_counts = {
        "single_term": 0,
        "multi_term": 0,
        "fallback": 0,
        "remaining": 0,
    }

    def _at_limit() -> bool:
        return bool(int(max_samples) > 0 and int(len(ordered)) >= int(max_samples))

    def _append_row(row_idx: int, stage_key: str) -> bool:
        r = int(row_idx)
        if r in used:
            return False
        ordered.append(int(r))
        used.add(int(r))
        stage_counts[str(stage_key)] = int(stage_counts.get(str(stage_key), 0)) + 1
        return True

    # Stage A: single-term groups (strict per-term round-robin).
    if len(active_indices) > 0:
        while not _at_limit():
            progressed = False
            for idx in active_indices:
                bucket = single_term_buckets.get(int(idx), [])
                while len(bucket) > 0 and int(bucket[0][1]) in used:
                    bucket.pop(0)
                if len(bucket) <= 0:
                    continue
                _, pick = bucket.pop(0)
                if _append_row(int(pick), stage_key="single_term"):
                    progressed = True
                if _at_limit():
                    break
            if not progressed:
                break

    # Stage B: multi-term signature groups (size-ascending, round-robin by signature).
    if not _at_limit():
        sig_groups = [sig for sig in signature_buckets.keys() if int(len(sig)) >= 2]
        size_values = sorted({int(len(sig)) for sig in sig_groups})
        for gsz in size_values:
            sigs = [sig for sig in sig_groups if int(len(sig)) == int(gsz)]
            sigs = sorted(
                sigs,
                key=lambda sig: tuple(
                    [
                        (
                            str(class_names[int(ix)]).strip().lower()
                            if int(ix) < int(len(class_names))
                            else f"class_{int(ix)}"
                        )
                        for ix in sig
                    ]
                ),
            )
            while not _at_limit():
                progressed = False
                for sig in sigs:
                    bucket = signature_buckets.get(sig, [])
                    while len(bucket) > 0 and int(bucket[0][1]) in used:
                        bucket.pop(0)
                    if len(bucket) <= 0:
                        continue
                    _, pick = bucket.pop(0)
                    if _append_row(int(pick), stage_key="multi_term"):
                        progressed = True
                    if _at_limit():
                        break
                if not progressed:
                    break
            if _at_limit():
                break

    # Stage C: fallback rows with weakest active support first promoted by score.
    if not _at_limit():
        for _, pick in fallback_rows:
            _append_row(int(pick), stage_key="fallback")
            if _at_limit():
                break

    # Stage D: deterministic remainder for full-deck stability.
    if not _at_limit():
        remaining = [int(i) for i in range(int(n)) if int(i) not in used]
        remaining = sorted(remaining)
        for pick in remaining:
            _append_row(int(pick), stage_key="remaining")
            if _at_limit():
                break

    if int(max_samples) > 0:
        ordered = ordered[: int(max_samples)]

    missing_terms: List[str] = []
    for idx in active_indices:
        name = str(class_names[int(idx)]) if int(idx) < int(len(class_names)) else f"class_{int(idx)}"
        if int(active_hits.get(str(name), 0)) <= 0:
            missing_terms.append(str(name))
    info = {
        "rows": int(n),
        "target_dim_expected": int(expected_dim),
        "target_dim_min": int(min(row_sizes)) if len(row_sizes) > 0 else 0,
        "target_dim_max": int(max(row_sizes)) if len(row_sizes) > 0 else 0,
        "active_terms": int(len(active_indices)),
        "active_term_hits": {str(k): int(v) for k, v in active_hits.items()},
        "missing_active_terms": list(missing_terms),
        "bucketed_rows": int(rows_with_active_support),
        "selected_rows": int(len(ordered)),
        "max_samples": int(max_samples),
        "threshold": float(threshold),
        "scheduler_mode": "deterministic_group_staged_v2",
        "stage_counts": {
            "single_term": int(stage_counts.get("single_term", 0)),
            "multi_term": int(stage_counts.get("multi_term", 0)),
            "fallback": int(stage_counts.get("fallback", 0)),
            "remaining": int(stage_counts.get("remaining", 0)),
        },
        "signature_groups": int(len(signature_buckets)),
    }
    return [int(i) for i in ordered], info


def _select_validation_indices(
    total_rows: int,
    seed: int,
    fraction: float = 0.25,
    min_count: int = 1,
    max_count: int = 0,
) -> List[int]:
    n = max(0, int(total_rows))
    if n <= 0:
        return []
    frac = float(max(0.0, min(1.0, float(fraction))))
    take = int(round(float(n) * frac))
    if int(min_count) > 0:
        take = max(int(take), int(min_count))
    if int(max_count) > 0:
        take = min(int(take), int(max_count))
    take = max(1, min(int(n), int(take)))
    idx = np.arange(int(n), dtype=np.int64)
    rng = np.random.default_rng(max(0, int(seed)))
    rng.shuffle(idx)
    return [int(i) for i in idx[: int(take)].tolist()]


def _build_payload_validation_gate_rows(
    data_root: str,
    image_size: int,
    seed: int,
    external_val_fraction: float = 0.20,
) -> Tuple[List[int], List[List[str]], Dict[str, Any]]:
    root = Path(str(data_root).strip() or "toys_to_survive_development/data/berkeley_sbd")
    size = max(8, int(image_size))
    cache_dir = root / "cache" / f"payload_bank_rgb{int(size)}_v2"
    manifest_path = cache_dir / "manifest.json"
    images_path = cache_dir / "images.npy"
    labels_path = cache_dir / "labels.npy"
    terms_path = cache_dir / "terms.json"
    sources_path = cache_dir / "sources.json"
    required = [manifest_path, images_path, labels_path, terms_path]
    missing = [str(p) for p in required if not p.exists()]
    if len(missing) > 0:
        raise RuntimeError(
            "Gate payload validation deck requires pre-washed payload cache files. "
            f"Missing: {missing}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed reading payload cache manifest for gate deck: {manifest_path} ({e})") from e
    if int(manifest.get("version", -1)) != 2:
        raise RuntimeError(
            "Unsupported payload cache version for gate deck "
            f"(expected=2, got={manifest.get('version', None)})."
        )
    if int(manifest.get("image_size", -1)) != int(size):
        raise RuntimeError(
            "Payload cache image-size mismatch for gate deck "
            f"(cache={manifest.get('image_size', None)} requested={int(size)})."
        )
    x_mem = np.load(str(images_path), mmap_mode="r")
    y_mem = np.load(str(labels_path), mmap_mode="r")
    if int(getattr(x_mem, "ndim", 0)) != 4 or int(getattr(y_mem, "ndim", 0)) != 2:
        raise RuntimeError(
            "Invalid payload cache tensor shapes for gate deck "
            f"(images={getattr(x_mem, 'shape', None)} labels={getattr(y_mem, 'shape', None)})."
        )
    terms_rows_raw: List[List[str]] = []
    try:
        raw_terms = json.loads(terms_path.read_text(encoding="utf-8"))
        if isinstance(raw_terms, list):
            for row in raw_terms:
                if isinstance(row, list):
                    terms_rows_raw.append(_normalize_vocab_terms([str(x) for x in row]))
                else:
                    terms_rows_raw.append([])
    except Exception as e:
        raise RuntimeError(f"Failed reading payload cache terms for gate deck: {terms_path} ({e})") from e

    n_rows = min(int(x_mem.shape[0]), int(y_mem.shape[0]), int(len(terms_rows_raw)))
    if n_rows <= 0:
        raise RuntimeError("Payload cache is empty for gate validation deck.")

    source_rows: List[str] = []
    if sources_path.exists():
        try:
            raw_sources = json.loads(sources_path.read_text(encoding="utf-8"))
            if isinstance(raw_sources, list):
                source_rows = [str(x) for x in raw_sources]
        except Exception:
            source_rows = []
    if int(len(source_rows)) < int(n_rows):
        source_rows.extend(["unknown_source"] * int(max(0, int(n_rows) - int(len(source_rows)))))
    if int(len(source_rows)) > int(n_rows):
        source_rows = source_rows[: int(n_rows)]

    _, gate_idx, source_stats = _split_payload_cache_indices(
        source_rows=source_rows,
        seed=int(seed),
        external_val_fraction=float(external_val_fraction),
    )
    selected_arr = np.asarray(gate_idx, dtype=np.int64)

    out_indices: List[int] = []
    out_terms: List[List[str]] = []
    for idx in selected_arr.tolist():
        if int(idx) < 0 or int(idx) >= int(n_rows):
            continue
        terms = list(terms_rows_raw[int(idx)]) if int(idx) < int(len(terms_rows_raw)) else []
        if len(terms) <= 0:
            raise RuntimeError(
                "Gate payload cache row is missing original terms; strict Stage-2 semantic gate requires "
                f"non-empty source terms. row_index={int(idx)}"
            )
        out_indices.append(int(idx))
        out_terms.append(list(terms))

    refresh_rows_total = int(sum(int(v.get("refresh_selected", 0)) for v in source_stats.values()))
    refresh_rows_berkeley_train = int(source_stats.get("berkeley_sbd_train", {}).get("refresh_selected", 0))
    refresh_rows_berkeley_val = int(source_stats.get("berkeley_sbd_val", {}).get("refresh_selected", 0))

    info = {
        "cache_dir": str(cache_dir),
        "manifest": str(manifest_path),
        "images_path": str(images_path),
        "labels_path": str(labels_path),
        "available_rows": int(n_rows),
        "selected_rows": int(len(out_indices)),
        "cache_label_dim": int(y_mem.shape[1]) if int(getattr(y_mem, "ndim", 0)) == 2 else 0,
        "selected_berkeley_val": int(source_stats.get("berkeley_sbd_val", {}).get("gate_selected", 0)),
        "selected_berkeley_train": int(source_stats.get("berkeley_sbd_train", {}).get("gate_selected", 0)),
        "refresh_rows_total": int(refresh_rows_total),
        "refresh_rows_berkeley_train": int(refresh_rows_berkeley_train),
        "refresh_rows_berkeley_val": int(refresh_rows_berkeley_val),
        "external_val_fraction": float(external_val_fraction),
        "source_stats": source_stats,
    }
    return out_indices, out_terms, info


def _build_gate_loader_from_arrays(
    images: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
    max_samples: int = 0,
    expected_target_dim: int = 0,
    ordered_indices: Optional[Sequence[int]] = None,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
) -> Tuple[Optional[DataLoader], int]:
    n = min(int(len(images)), int(len(targets)))
    if n <= 0:
        return None, 0
    x_rows: List[np.ndarray] = []
    for i in range(int(n)):
        arr = np.asarray(images[i], dtype=np.float32)
        if int(arr.ndim) == 3 and int(arr.shape[0]) == 3:
            row = arr
        elif int(arr.ndim) == 3 and int(arr.shape[2]) == 3:
            row = np.transpose(arr, (2, 0, 1)).astype(np.float32, copy=False)
        elif int(arr.ndim) == 2:
            row = np.repeat(arr[None, :, :], 3, axis=0).astype(np.float32, copy=False)
        else:
            raise RuntimeError(f"Unsupported gate image row shape: {tuple(arr.shape)}")
        x_rows.append(np.clip(row, 0.0, 1.0).astype(np.float32, copy=False))
    x_np = np.stack(x_rows, axis=0).astype(np.float32, copy=False)
    y_rows = [np.asarray(targets[i], dtype=np.float32).reshape(-1) for i in range(int(n))]
    y_sizes = [int(r.size) for r in y_rows]
    bad_dims = sorted({int(sz) for sz in y_sizes if int(sz) != int(y_sizes[0])}) if len(y_sizes) > 0 else []
    if len(bad_dims) > 0:
        dims = sorted({int(x) for x in y_sizes})
        raise RuntimeError(
            "Gate loader received mixed label widths; all rows must already match sentence-transformer semantic width. "
            f"got_dims={dims}"
        )
    y_dim = int(y_sizes[0]) if len(y_sizes) > 0 else 0
    if int(y_dim) <= 0:
        raise RuntimeError("Gate loader received empty label vectors.")
    if int(expected_target_dim) > 0 and int(y_dim) != int(expected_target_dim):
        raise RuntimeError(
            "Gate loader label width mismatch: "
            f"got={int(y_dim)} expected={int(expected_target_dim)}"
        )
    y_np = np.stack(y_rows, axis=0).astype(np.float32, copy=False)
    y_np = np.clip(y_np[:, : int(y_dim)], 0.0, 1.0).astype(np.float32, copy=False)

    if ordered_indices is None:
        picks = np.arange(int(n), dtype=np.int64)
        rng = np.random.default_rng(int(seed))
        rng.shuffle(picks)
        if int(max_samples) > 0:
            picks = picks[: int(min(int(max_samples), int(picks.shape[0])))]
    else:
        picks = np.asarray([int(i) for i in ordered_indices if 0 <= int(i) < int(n)], dtype=np.int64)
        if int(max_samples) > 0 and int(picks.size) > int(max_samples):
            picks = picks[: int(max_samples)]
    if int(picks.size) <= 0:
        return None, 0

    x_t = torch.from_numpy(x_np)
    y_t = torch.from_numpy(y_np)
    ds_full: Dataset = torch.utils.data.TensorDataset(x_t, y_t)
    ds_use: Dataset = torch.utils.data.Subset(ds_full, picks.tolist())
    loader = DataLoader(
        ds_use,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        **_dataloader_perf_kwargs(
            num_workers=max(0, int(num_workers)),
            persistent_workers=bool(persistent_workers),
            prefetch_factor=int(prefetch_factor),
        ),
    )
    return loader, int(picks.size)


def _build_gate_loader_from_dataset(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
    max_samples: int = 0,
    ordered_indices: Optional[Sequence[int]] = None,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
) -> Tuple[Optional[DataLoader], int]:
    n = int(len(dataset))
    if n <= 0:
        return None, 0

    if ordered_indices is None:
        picks = np.arange(int(n), dtype=np.int64)
        rng = np.random.default_rng(int(seed))
        rng.shuffle(picks)
        if int(max_samples) > 0:
            picks = picks[: int(min(int(max_samples), int(picks.shape[0])))]
    else:
        picks = np.asarray([int(i) for i in ordered_indices if 0 <= int(i) < int(n)], dtype=np.int64)
        if int(max_samples) > 0 and int(picks.size) > int(max_samples):
            picks = picks[: int(max_samples)]
    if int(picks.size) <= 0:
        return None, 0

    ds_use: Dataset = dataset if int(picks.size) == int(n) else torch.utils.data.Subset(dataset, picks.tolist())
    loader = DataLoader(
        ds_use,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        **_dataloader_perf_kwargs(
            num_workers=max(0, int(num_workers)),
            persistent_workers=bool(persistent_workers),
            prefetch_factor=int(prefetch_factor),
        ),
    )
    return loader, int(picks.size)


def _augment_bootstrap_chw01(
    img: np.ndarray,
    seed: int,
    return_terms: bool = False,
) -> Any:
    arr = np.asarray(img, dtype=np.float32)
    if int(arr.ndim) == 3 and int(arr.shape[0]) == 3:
        x = np.asarray(arr, dtype=np.float32, order="C")
    elif int(arr.ndim) == 3 and int(arr.shape[2]) == 3:
        x = np.transpose(arr, (2, 0, 1)).astype(np.float32, copy=False)
    elif int(arr.ndim) == 2:
        x = np.repeat(arr[None, :, :], 3, axis=0).astype(np.float32, copy=False)
    else:
        raise RuntimeError(f"Unsupported bootstrap row shape for augmentation: {tuple(arr.shape)}")
    x = np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)
    c, h, w = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    rng = np.random.default_rng(int(seed))

    applied_terms: List[str] = []

    # Spatial permutations.
    shift_x = int(rng.integers(-max(1, w // 14), max(1, w // 14) + 1))
    shift_y = int(rng.integers(-max(1, h // 14), max(1, h // 14) + 1))
    if shift_x != 0:
        x = np.roll(x, shift=shift_x, axis=2)
    if shift_y != 0:
        x = np.roll(x, shift=shift_y, axis=1)
    if float(rng.random()) < 0.45:
        x = np.flip(x, axis=2).copy()
    if float(rng.random()) < 0.15:
        x = np.flip(x, axis=1).copy()

    # Deformation/noise variants.
    if float(rng.random()) < 0.55:
        k = int(rng.choice(np.asarray([3, 5, 7], dtype=np.int32)))
        t = torch.from_numpy(np.asarray(x, dtype=np.float32)[None, ...])
        t = F.avg_pool2d(t, kernel_size=int(k), stride=1, padding=int(k // 2))
        x = np.asarray(t[0].cpu().numpy(), dtype=np.float32)
        applied_terms.extend(_semantic_damage_tags("blur damage"))
    if float(rng.random()) < 0.45:
        odd_shift = int(rng.integers(1, 8))
        x[:, 1::2, :] = np.roll(x[:, 1::2, :], shift=odd_shift, axis=2)
        applied_terms.extend(_semantic_damage_tags("stride skew damage"))
    if float(rng.random()) < 0.35:
        keep = float(rng.uniform(0.76, 0.96))
        mask = (rng.random((h, w), dtype=np.float32) < keep).astype(np.float32, copy=False)
        x = x * mask[None, :, :]
        applied_terms.extend(_semantic_damage_tags("dropout damage"))
    if float(rng.random()) < 0.40:
        lv = int(rng.choice(np.asarray([4, 6, 8, 12], dtype=np.int32)))
        x = np.round(x * float(lv - 1)) / float(lv - 1)
        applied_terms.extend(_semantic_damage_tags("quantization damage"))

    # Signal/noise tangling blend.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x01 = xx / float(max(1, w - 1))
    y01 = yy / float(max(1, h - 1))
    fx = float(rng.uniform(1.2, 8.5))
    fy = float(rng.uniform(1.0, 7.5))
    ph = float(rng.uniform(0.0, 2.0 * math.pi))
    sig = 0.5 + (0.5 * np.sin((2.0 * math.pi * ((fx * x01) + (fy * y01))) + ph))
    noi = rng.random((h, w), dtype=np.float32)
    noise_tags = _semantic_noise_terms_from_spectrum_sample(noi)
    if float(rng.random()) < 0.5:
        mix_a = float(rng.uniform(0.18, 0.62))
        mix = np.clip((mix_a * sig) + ((1.0 - mix_a) * noi), 0.0, 1.0).astype(np.float32, copy=False)
    else:
        bits = int(rng.integers(2, 7))
        s_u16 = np.round(np.clip(sig, 0.0, 1.0) * 65535.0).astype(np.uint16, copy=False)
        n_u16 = np.round(np.clip(noi, 0.0, 1.0) * 65535.0).astype(np.uint16, copy=False)
        payload = ((n_u16 >> int(16 - bits)) & np.uint16((1 << bits) - 1)).astype(np.uint16, copy=False)
        keep_mask = np.uint16(0xFFFF ^ ((1 << bits) - 1))
        mix = np.clip(((s_u16 & keep_mask) | payload).astype(np.float32) / 65535.0, 0.0, 1.0).astype(np.float32, copy=False)
    blend = float(rng.uniform(0.08, 0.28))
    x = np.clip(((1.0 - blend) * x) + (blend * mix[None, :, :]), 0.0, 1.0)
    applied_terms.extend(["mixed noise and signal", "noise", "signal"])
    applied_terms.extend(list(noise_tags))

    if float(rng.random()) < 0.65:
        std = float(rng.uniform(0.01, 0.08))
        noise_delta = (std * rng.standard_normal((c, h, w), dtype=np.float32)).astype(np.float32, copy=False)
        x = np.clip(x + noise_delta, 0.0, 1.0)
        applied_terms.extend(_semantic_damage_tags("noise damage"))
        applied_terms.extend(_semantic_noise_terms_from_spectrum_sample(noise_delta))
    x_out = np.asarray(x, dtype=np.float32)
    if bool(return_terms):
        return x_out, _normalize_vocab_terms(applied_terms)
    return x_out


class _BootstrapExpandedDataset(Dataset):
    def __init__(
        self,
        images: Sequence[np.ndarray],
        targets: Sequence[np.ndarray],
        total_rows: int,
        seed: int,
        augment: bool,
        expected_target_dim: int = 0,
        semantic_term_to_idx: Optional[Dict[str, int]] = None,
        augment_apply_terms: bool = True,
    ):
        n = min(int(len(images)), int(len(targets)))
        if n <= 0:
            raise RuntimeError("BootstrapExpandedDataset requires non-empty images/targets.")
        self.images = [np.asarray(images[i], dtype=np.float32) for i in range(int(n))]
        self.targets = [np.asarray(targets[i], dtype=np.float32).reshape(-1) for i in range(int(n))]
        self.base_rows = int(n)
        self.total_rows = max(int(self.base_rows), int(total_rows))
        self.seed = int(seed)
        self.augment = bool(augment)
        y_sizes = sorted({int(v.size) for v in self.targets})
        if len(y_sizes) != 1:
            raise RuntimeError(f"BootstrapExpandedDataset target width mismatch: {y_sizes}")
        self.target_dim = int(y_sizes[0])
        if int(expected_target_dim) > 0 and int(self.target_dim) != int(expected_target_dim):
            raise RuntimeError(
                "BootstrapExpandedDataset target width mismatch: "
                f"got={int(self.target_dim)} expected={int(expected_target_dim)}"
            )
        self.semantic_term_to_idx = {
            re.sub(r"\s+", " ", str(k)).strip().lower(): int(v)
            for k, v in (semantic_term_to_idx.items() if isinstance(semantic_term_to_idx, dict) else [])
            if str(k).strip()
        }
        self.augment_apply_terms = bool(augment_apply_terms)

    def __len__(self) -> int:
        return int(self.total_rows)

    def __getitem__(self, index: int):
        idx = int(index)
        if idx < 0:
            idx = int(self.total_rows) + idx
        if idx < 0 or idx >= int(self.total_rows):
            raise IndexError(idx)
        cycle = int(idx // max(1, int(self.base_rows)))
        off = int(idx % max(1, int(self.base_rows)))
        base_idx = int((off + (cycle * 17) + int(self.seed % max(1, int(self.base_rows)))) % int(self.base_rows))
        img = np.asarray(self.images[int(base_idx)], dtype=np.float32)
        tgt = np.asarray(self.targets[int(base_idx)], dtype=np.float32).copy()
        if bool(self.augment) and int(self.total_rows) > int(self.base_rows):
            aug_seed = int((int(self.seed) * 2654435761 + int(idx) * 1103515245 + int(base_idx) * 122949829) % (2**32 - 1))
            img, aug_terms = _augment_bootstrap_chw01(
                img=img,
                seed=int(aug_seed),
                return_terms=True,
            )
            if bool(self.augment_apply_terms) and int(len(self.semantic_term_to_idx)) > 0 and isinstance(aug_terms, list):
                for term in aug_terms:
                    tk = re.sub(r"\s+", " ", str(term)).strip().lower()
                    if not tk:
                        continue
                    ti = int(self.semantic_term_to_idx.get(tk, -1))
                    if 0 <= int(ti) < int(tgt.size):
                        tgt[int(ti)] = 1.0
        return (
            torch.from_numpy(np.asarray(img, dtype=np.float32)),
            torch.from_numpy(np.asarray(tgt, dtype=np.float32)),
        )


@torch.no_grad()
def _evaluate_berkeley_classifier_gate(
    classifier: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_steps: int,
    active_classes: int = 0,
    amp_enabled: bool = False,
    amp_dtype: str = "float16",
    channels_last: bool = False,
    preview_sink: Optional[Dict[str, Any]] = None,
):
    classifier.eval()
    amp_dtype_t = _resolve_amp_dtype(amp_dtype) if amp_enabled else torch.float16
    total_loss = 0.0
    n = 0
    logits_all = []
    targets_all = []
    steps = 0

    def _is_cuda_oom(exc: BaseException) -> bool:
        txt = str(exc).lower()
        if "out of memory" not in txt:
            return False
        return ("cuda" in txt) or ("cudnn" in txt) or ("cublas" in txt)

    batch_idx = 0
    for xb, yb in loader:
        batch_idx += 1
        xb = xb.to(device, non_blocking=True)
        if channels_last:
            xb = xb.contiguous(memory_format=torch.channels_last)
        yb = yb.to(device, non_blocking=True)
        batch_n = int(xb.shape[0])
        eval_chunk = max(1, int(batch_n))

        while True:
            try:
                batch_loss = 0.0
                batch_seen = 0
                batch_logits: List[torch.Tensor] = []
                batch_targets: List[torch.Tensor] = []
                for start in range(0, batch_n, eval_chunk):
                    stop = min(batch_n, start + eval_chunk)
                    xb_part = xb[start:stop]
                    yb_part = yb[start:stop]
                    with _autocast_context(device=device, enabled=amp_enabled, amp_dtype_t=amp_dtype_t):
                        logits = classifier(xb_part)
                    supervised_dim = int(yb_part.shape[1]) if int(yb_part.ndim) == 2 else int(logits.shape[1])
                    if int(active_classes) > 0:
                        supervised_dim = min(int(supervised_dim), int(active_classes))
                    supervised_dim = max(1, int(supervised_dim))
                    loss, logits_sup, y_sup, _ = _classifier_supervision_loss(
                        classifier=classifier,
                        logits=logits,
                        y_multihot=yb_part,
                        supervised_dim=int(supervised_dim),
                    )
                    loss = loss * float(CLASSIFIER_LOSS_SCALE)
                    part_n = int(xb_part.shape[0])
                    batch_loss += float(loss.item()) * part_n
                    batch_seen += part_n
                    batch_logits.append(logits_sup.detach().cpu())
                    batch_targets.append(y_sup.detach().cpu())
                    if isinstance(preview_sink, dict) and int(part_n) > 0:
                        try:
                            cap_i = int(part_n - 1)
                            cap_img = xb_part[cap_i].detach().to(torch.float32).cpu().numpy().astype(np.float32, copy=False)
                            cap_target = y_sup[cap_i].detach().to(torch.float32).cpu().numpy().astype(np.float32, copy=False)
                            cap_probs = torch.sigmoid(logits_sup[cap_i].to(torch.float32)).detach().cpu().numpy().astype(np.float32, copy=False)
                            preview_sink.clear()
                            preview_sink.update(
                                {
                                    "valid": True,
                                    "batch": int(batch_idx),
                                    "chunk_start": int(start),
                                    "chunk_stop": int(stop),
                                    "image": np.asarray(cap_img, dtype=np.float32),
                                    "target": np.asarray(cap_target, dtype=np.float32).reshape(-1),
                                    "probs": np.asarray(cap_probs, dtype=np.float32).reshape(-1),
                                }
                            )
                        except Exception:
                            pass

                total_loss += float(batch_loss)
                n += int(batch_seen)
                logits_all.extend(batch_logits)
                targets_all.extend(batch_targets)
                break
            except RuntimeError as e:
                if device.type != "cuda" or (not _is_cuda_oom(e)) or int(eval_chunk) <= 1:
                    raise
                next_chunk = max(1, int(eval_chunk) // 2)
                if int(next_chunk) > 1:
                    next_chunk = 1 << (int(next_chunk).bit_length() - 1)
                torch.cuda.empty_cache()
                print(
                    f"[berkeley-gate] CUDA OOM at eval_chunk={eval_chunk}; retrying eval_chunk={next_chunk}",
                    flush=True,
                )
                eval_chunk = int(next_chunk)
        steps += 1
        if max_steps > 0 and steps >= int(max_steps):
            break

    if n <= 0:
        return {
            "loss": 0.0,
            "macro_f1": 0.0,
            "micro_f1": 0.0,
            "bit_acc": 0.0,
            "mean_confidence": 0.0,
            "num_samples": 0,
        }

    logits_cat = torch.cat(logits_all, dim=0)
    targets_cat = torch.cat(targets_all, dim=0)
    probs = torch.sigmoid(logits_cat)
    preds = (probs >= 0.5).float()
    t = targets_cat.float()

    tp = (preds * t).sum(dim=0)
    fp = (preds * (1.0 - t)).sum(dim=0)
    fn = ((1.0 - preds) * t).sum(dim=0)
    macro_f1 = torch.mean((2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)).item()
    tp_m = tp.sum()
    fp_m = fp.sum()
    fn_m = fn.sum()
    micro_f1 = ((2.0 * tp_m) / (2.0 * tp_m + fp_m + fn_m + 1e-8)).item()
    bit_acc = preds.eq(t).float().mean().item()
    mean_conf = torch.maximum(probs, 1.0 - probs).mean().item()
    return {
        "loss": total_loss / max(1, n),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "bit_acc": float(bit_acc),
        "mean_confidence": float(mean_conf),
        "num_samples": int(n),
    }


def _run_berkeley_refresh_epochs(
    classifier: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    max_steps: int,
    min_steps: int,
    lr_sine_cycles: float,
    lr_sine_frequency: float,
    lr_sine_tail_fraction: float,
    lr_sine_min_scale: float,
    amp_enabled: bool = False,
    amp_dtype: str = "float16",
    channels_last: bool = False,
    grad_accum_steps: int = 1,
    log_every: int = 0,
    max_seconds: float = 0.0,
    cache_x: Optional[torch.Tensor] = None,
    cache_y: Optional[torch.Tensor] = None,
    cache_batch_size: int = 0,
    active_classes: int = 0,
    target_label_knockout_prob: float = 0.0,
    target_label_knockout_min_keep: int = 1,
    target_label_knockout_max_drop_frac: float = 0.5,
    target_label_knockout_seed: int = 0,
    step_preview_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    stop_requested: Optional[Callable[[], bool]] = None,
):
    if epochs <= 0:
        return {"ran": False, "loss": 0.0}

    # A prior transformer stage may have frozen this classifier for feature scoring.
    # Ensure refresh has trainable params before building autograd graph.
    if not any(bool(p.requires_grad) for p in classifier.parameters()):
        for p in classifier.parameters():
            p.requires_grad_(True)

    classifier.train()
    if channels_last:
        classifier = classifier.to(memory_format=torch.channels_last)
    amp_dtype_t = _resolve_amp_dtype(amp_dtype) if amp_enabled else torch.float16
    use_scaler = bool(amp_enabled and device.type == "cuda" and amp_dtype_t == torch.float16)
    scaler = _make_grad_scaler(enabled=use_scaler)
    grad_accum_steps = max(1, int(grad_accum_steps))
    use_cache = (
        (cache_x is not None)
        and (cache_y is not None)
        and int(getattr(cache_x, "shape", [0])[0]) > 0
        and int(getattr(cache_y, "shape", [0])[0]) == int(getattr(cache_x, "shape", [0])[0])
        and int(cache_batch_size) > 0
    )
    cache_batch_size = max(1, int(cache_batch_size)) if use_cache else 0
    refresh_source = "cache" if use_cache else "loader"
    knockout_prob = float(max(0.0, min(1.0, float(target_label_knockout_prob))))
    knockout_enabled = bool(knockout_prob > 0.0)
    knockout_rng = np.random.default_rng(max(0, int(target_label_knockout_seed)))
    knockout_rows_applied = 0
    knockout_labels_dropped = 0
    opt = torch.optim.AdamW(classifier.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    min_steps = max(0, int(min_steps))
    if use_cache:
        cache_n = int(cache_x.shape[0])
        full_steps = max(1, int(math.ceil(float(cache_n) / float(cache_batch_size))))
        steps_per_epoch = int(full_steps if int(max_steps) <= 0 else max(1, int(max_steps)))
        steps_per_epoch = max(int(steps_per_epoch), int(min_steps))
    else:
        full_steps = max(1, int(len(loader)))
        steps_per_epoch = int(full_steps if int(max_steps) <= 0 else max(1, int(max_steps)))
        steps_per_epoch = max(int(steps_per_epoch), int(min_steps))
    updates_per_epoch = int(math.ceil(float(steps_per_epoch) / float(grad_accum_steps)))
    lr_ctl = SinusoidalLRController(
        optimizer=opt,
        total_steps=max(1, int(epochs) * updates_per_epoch),
        options=SinusoidalLROptions(
            cycles=float(lr_sine_cycles),
            frequency=float(lr_sine_frequency),
            tail_fraction=float(lr_sine_tail_fraction),
            min_scale=float(lr_sine_min_scale),
        ),
    )
    total_loss = 0.0
    n = 0
    t_start = time.time()
    total_target_steps = max(1, int(epochs) * steps_per_epoch)
    global_step = 0
    stop_now = False
    for _ in range(int(epochs)):
        step = 0
        opt.zero_grad(set_to_none=True)
        if use_cache:
            cache_n = int(cache_x.shape[0])
            cache_perm = torch.randperm(cache_n, device=cache_x.device)
            cache_pos = 0
        else:
            loader_iter = iter(loader)

        while step < steps_per_epoch:
            if stop_requested is not None:
                try:
                    if bool(stop_requested()):
                        stop_now = True
                        break
                except Exception:
                    pass
            if use_cache:
                idx_parts: List[torch.Tensor] = []
                need = int(cache_batch_size)
                while need > 0:
                    if cache_pos >= cache_n:
                        cache_perm = torch.randperm(cache_n, device=cache_x.device)
                        cache_pos = 0
                    take = min(need, cache_n - cache_pos)
                    idx_parts.append(cache_perm[cache_pos : cache_pos + take])
                    cache_pos += take
                    need -= take
                idx = idx_parts[0] if len(idx_parts) == 1 else torch.cat(idx_parts, dim=0)
                xb = cache_x.index_select(0, idx)
                yb = cache_y.index_select(0, idx)
            else:
                try:
                    xb, yb = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(loader)
                    xb, yb = next(loader_iter)
            if xb.device != device:
                xb = xb.to(device, non_blocking=True)
            if yb.device != device:
                yb = yb.to(device, non_blocking=True)
            if bool(knockout_enabled):
                yb, rows_drop, labels_drop = _label_knockout_tensor_batch(
                    yb=yb,
                    rng=knockout_rng,
                    prob=float(knockout_prob),
                    min_keep=int(target_label_knockout_min_keep),
                    max_drop_frac=float(target_label_knockout_max_drop_frac),
                    threshold=0.5,
                )
                knockout_rows_applied += int(rows_drop)
                knockout_labels_dropped += int(labels_drop)
            if channels_last:
                xb = xb.contiguous(memory_format=torch.channels_last)
            with _autocast_context(device=device, enabled=amp_enabled, amp_dtype_t=amp_dtype_t):
                logits = classifier(xb)
            supervised_dim = int(yb.shape[1]) if int(yb.ndim) == 2 else int(logits.shape[1])
            if int(active_classes) > 0:
                supervised_dim = min(int(supervised_dim), int(active_classes))
            supervised_dim = max(1, int(supervised_dim))
            loss, logits_sup, y_sup, _ = _classifier_supervision_loss(
                classifier=classifier,
                logits=logits,
                y_multihot=yb,
                supervised_dim=int(supervised_dim),
            )
            loss = loss * float(CLASSIFIER_LOSS_SCALE)
            loss_to_backprop = loss / float(grad_accum_steps)
            if use_scaler:
                scaler.scale(loss_to_backprop).backward()
            else:
                loss_to_backprop.backward()
            do_step = ((step + 1) % grad_accum_steps == 0) or ((step + 1) == steps_per_epoch)
            if do_step:
                if use_scaler:
                    scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                if use_scaler:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                lr_ctl.step()
                opt.zero_grad(set_to_none=True)
            total_loss += float(loss.item()) * int(xb.shape[0])
            n += int(xb.shape[0])
            step += 1
            global_step += 1
            if int(log_every) > 0 and (global_step % int(log_every) == 0):
                elapsed = max(1e-6, time.time() - t_start)
                ips = float(n) / elapsed
                print(
                    f"[berkeley-refresh] step={global_step}/{total_target_steps} "
                    f"loss={total_loss / max(1, n):.4f} samples={n} samp_per_sec={ips:.1f}",
                    flush=True,
                )
            if step_preview_callback is not None:
                emit = False
                if int(log_every) > 0:
                    emit = ((int(global_step) % int(log_every) == 0) or (int(global_step) == int(total_target_steps)))
                else:
                    emit = (int(global_step) == int(total_target_steps))
                if emit and int(xb.shape[0]) > 0:
                    try:
                        step_preview_callback(
                            {
                                "global_step": int(global_step),
                                "total_steps": int(total_target_steps),
                                "img": xb[0].detach().to(torch.float32),
                                "probs": torch.sigmoid(logits_sup[0].detach()).to(torch.float32),
                                "target_vec": y_sup[0].detach().to(torch.float32),
                                "loss": float(total_loss / max(1, n)),
                                "batch_loss": float(loss.detach().to(torch.float32).item()),
                            }
                        )
                    except Exception as e:
                        _log(f"[stage-opengl] C-step callback failed: {e}")
            if float(max_seconds) > 0.0 and (time.time() - t_start) >= float(max_seconds):
                classifier.eval()
                return {
                    "ran": True,
                    "loss": total_loss / max(1, n),
                    "samples": n,
                    "steps_per_epoch": int(steps_per_epoch),
                    "truncated_by_time": True,
                    "stopped_early": False,
                    "elapsed_sec": float(time.time() - t_start),
                    "source": refresh_source,
                    "target_knockout_rows_applied": int(knockout_rows_applied),
                    "target_knockout_labels_dropped": int(knockout_labels_dropped),
                }
        if stop_now:
            break
    classifier.eval()
    return {
        "ran": bool(n > 0),
        "loss": total_loss / max(1, n),
        "samples": n,
        "steps_per_epoch": int(steps_per_epoch),
        "truncated_by_time": False,
        "stopped_early": bool(stop_now),
        "elapsed_sec": float(time.time() - t_start),
        "source": refresh_source,
        "target_knockout_rows_applied": int(knockout_rows_applied),
        "target_knockout_labels_dropped": int(knockout_labels_dropped),
    }


def _run_fake_class_refresh_epochs(
    classifier: nn.Module,
    generator: nn.Module,
    discriminator: Optional[nn.Module],
    payload_conditions: Sequence[Any],
    condition_num_classes: int,
    fake_label_vector: Optional[torch.Tensor],
    z_dim: int,
    device: torch.device,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    lr_sine_cycles: float,
    lr_sine_frequency: float,
    lr_sine_tail_fraction: float,
    lr_sine_min_scale: float,
    amp_enabled: bool = False,
    amp_dtype: str = "float16",
    channels_last: bool = False,
    grad_accum_steps: int = 1,
    log_every: int = 0,
    include_condition_targets: bool = True,
    fake_vector_weight: float = 1.0,
    condition_target_weight: float = 0.35,
    disc_conf_temperature: float = 1.0,
    disc_conf_floor: float = 0.25,
    balance_disc_groups: bool = True,
    step_preview_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    stop_requested: Optional[Callable[[], bool]] = None,
    seed: int = 0,
):
    if epochs <= 0 or steps_per_epoch <= 0 or batch_size <= 0:
        return {"ran": False, "loss": 0.0, "samples": 0, "source": "fake_vector"}
    if generator is None or len(payload_conditions) <= 0:
        return {"ran": False, "loss": 0.0, "samples": 0, "source": "fake_vector"}
    if int(condition_num_classes) <= 0:
        return {"ran": False, "loss": 0.0, "samples": 0, "source": "fake_vector"}
    if fake_label_vector is None:
        return {"ran": False, "loss": 0.0, "samples": 0, "source": "fake_vector", "reason": "missing_fake_vector"}
    if not isinstance(classifier, TinyConvClassifier):
        return {
            "ran": False,
            "loss": 0.0,
            "samples": 0,
            "source": "fake_vector",
            "reason": f"unsupported_classifier:{type(classifier).__name__}",
        }

    if not any(bool(p.requires_grad) for p in classifier.parameters()):
        for p in classifier.parameters():
            p.requires_grad_(True)

    classifier.train()
    if channels_last:
        classifier = classifier.to(memory_format=torch.channels_last)
    generator_was_training = bool(generator.training)
    generator.eval()
    discriminator_was_training = False
    if discriminator is not None:
        discriminator_was_training = bool(discriminator.training)
        discriminator.eval()

    fake_vec = fake_label_vector.detach().to(device=device, dtype=torch.float32).reshape(-1)
    if int(fake_vec.numel()) <= 0:
        classifier.eval()
        generator.train(generator_was_training)
        if discriminator is not None:
            discriminator.train(discriminator_was_training)
        return {"ran": False, "loss": 0.0, "samples": 0, "source": "fake_vector", "reason": "empty_fake_vector"}
    fake_vec = F.normalize(fake_vec.unsqueeze(0), dim=1, eps=1e-6).squeeze(0)
    if int(classifier.embed_proj.out_features) != int(fake_vec.numel()):
        classifier.eval()
        generator.train(generator_was_training)
        if discriminator is not None:
            discriminator.train(discriminator_was_training)
        return {
            "ran": False,
            "loss": 0.0,
            "samples": 0,
            "source": "fake_vector",
            "reason": (
                f"fake_vector_dim_mismatch: vec={int(fake_vec.numel())} "
                f"embed_proj={int(classifier.embed_proj.out_features)}"
            ),
        }

    amp_dtype_t = _resolve_amp_dtype(amp_dtype) if amp_enabled else torch.float16
    use_scaler = bool(amp_enabled and device.type == "cuda" and amp_dtype_t == torch.float16)
    scaler = _make_grad_scaler(enabled=use_scaler)
    grad_accum_steps = max(1, int(grad_accum_steps))
    n_steps = max(1, int(steps_per_epoch))
    updates_per_epoch = int(math.ceil(float(n_steps) / float(grad_accum_steps)))
    opt = torch.optim.AdamW(classifier.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    lr_ctl = SinusoidalLRController(
        optimizer=opt,
        total_steps=max(1, int(epochs) * max(1, int(updates_per_epoch))),
        options=SinusoidalLROptions(
            cycles=float(lr_sine_cycles),
            frequency=float(lr_sine_frequency),
            tail_fraction=float(lr_sine_tail_fraction),
            min_scale=float(lr_sine_min_scale),
        ),
    )

    rng = np.random.default_rng(seed)
    cond_bank_cpu = _payload_condition_bank_tensor(
        payload_targets=payload_conditions,
        num_classes=int(condition_num_classes),
    )
    if int(cond_bank_cpu.shape[0]) <= 0:
        classifier.eval()
        generator.train(generator_was_training)
        if discriminator is not None:
            discriminator.train(discriminator_was_training)
        return {"ran": False, "loss": 0.0, "samples": 0, "source": "fake_vector", "reason": "empty_condition_bank"}
    cond_bank = cond_bank_cpu.to(device=device, dtype=torch.float32)

    total_loss = 0.0
    n = 0
    global_step = 0
    fake_cos_sum = 0.0
    fake_cos_pass_sum = 0.0
    fake_cos_fail_sum = 0.0
    disc_conf_sum = 0.0
    disc_conf_pass_sum = 0.0
    disc_conf_fail_sum = 0.0
    cond_target_prob_sum = 0.0
    disc_pass_n = 0
    disc_fail_n = 0
    disc_logit_pass_sum = 0.0
    disc_logit_fail_sum = 0.0
    total_target_steps = int(max(1, int(epochs)) * int(n_steps))
    t_start = time.time()
    stop_now = False
    opt.zero_grad(set_to_none=True)
    for _ in range(int(max(1, int(epochs)))):
        for step_idx in range(1, int(n_steps) + 1):
            if stop_requested is not None:
                try:
                    if bool(stop_requested()):
                        stop_now = True
                        break
                except Exception:
                    pass

            idx = rng.integers(0, int(cond_bank.shape[0]), size=max(1, int(batch_size)))
            idx_t = torch.as_tensor(idx, device=device, dtype=torch.long)
            cond = cond_bank.index_select(0, idx_t).to(device=device, dtype=torch.float32)
            z = torch.randn((int(cond.shape[0]), max(8, int(z_dim))), device=device)
            with torch.no_grad():
                with _autocast_context(device=device, enabled=amp_enabled, amp_dtype_t=amp_dtype_t):
                    xb = generator(z, cond).to(torch.float32)
                disc_logits = None
                if discriminator is not None:
                    with _autocast_context(device=device, enabled=amp_enabled, amp_dtype_t=amp_dtype_t):
                        disc_logits = discriminator(xb, cond).to(torch.float32)
            if channels_last:
                xb = xb.contiguous(memory_format=torch.channels_last)
            with _autocast_context(device=device, enabled=amp_enabled, amp_dtype_t=amp_dtype_t):
                feat = classifier.extract_features(xb)
                z_norm = classifier.encode_semantic_from_features(feat)
                fake_cos = torch.sum(z_norm * fake_vec.unsqueeze(0).to(device=z_norm.device, dtype=z_norm.dtype), dim=1)
                fake_loss_per = 1.0 - fake_cos

                if bool(int(classifier.label_embed_enabled.item())) and int(classifier.label_embed_bank.shape[0]) > 0:
                    logits = classifier.semantic_logits_from_features(feat=feat)
                else:
                    logits = classifier.head[-1](feat)

                if int(logits.shape[1]) != int(cond.shape[1]):
                    raise RuntimeError(
                        "Condition/logit width mismatch in fake feedback: "
                        f"logits={tuple(logits.shape)} cond={tuple(cond.shape)}"
                    )
                if bool(include_condition_targets):
                    cond_prob = torch.sigmoid(logits.to(dtype=torch.float32))
                    cond_target = cond.to(dtype=torch.float32)
                    cond_loss_per = F.mse_loss(
                        cond_prob,
                        cond_target,
                        reduction="none",
                    ).mean(dim=1)
                else:
                    cond_loss_per = torch.zeros_like(fake_loss_per)

                loss_per_sample = (
                    (float(max(0.0, fake_vector_weight)) * fake_loss_per)
                    + (float(max(0.0, condition_target_weight)) * cond_loss_per)
                )
                if disc_logits is not None:
                    disc_det = disc_logits.detach().to(torch.float32)
                    pass_mask = disc_det >= 0.0
                    fail_mask = ~pass_mask
                    conf_temp = max(1e-4, float(disc_conf_temperature))
                    conf_floor = max(0.0, min(1.0, float(disc_conf_floor)))
                    disc_conf = torch.sigmoid(torch.abs(disc_det) / conf_temp)
                    conf_w = conf_floor + ((1.0 - conf_floor) * disc_conf)
                    if bool(balance_disc_groups):
                        pass_count = torch.clamp(pass_mask.to(torch.float32).sum(), min=1.0)
                        fail_count = torch.clamp(fail_mask.to(torch.float32).sum(), min=1.0)
                        w_pass = 0.5 / pass_count
                        w_fail = 0.5 / fail_count
                        group_w = torch.where(pass_mask, w_pass, w_fail).to(torch.float32)
                    else:
                        group_w = torch.ones_like(conf_w, dtype=torch.float32)
                    sample_w = group_w * conf_w
                    sample_w = sample_w / torch.clamp(sample_w.mean(), min=1e-6)
                    loss = (loss_per_sample.to(torch.float32) * sample_w.to(torch.float32)).mean().to(loss_per_sample.dtype)
                else:
                    pass_mask = None
                    disc_conf = None
                    loss = loss_per_sample.mean()
            loss = loss * float(CLASSIFIER_LOSS_SCALE)
            loss_to_backprop = loss / float(grad_accum_steps)
            if use_scaler:
                scaler.scale(loss_to_backprop).backward()
            else:
                loss_to_backprop.backward()

            do_step = ((int(step_idx) % int(grad_accum_steps)) == 0) or (int(step_idx) == int(n_steps))
            if do_step:
                if use_scaler:
                    scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                if use_scaler:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                lr_ctl.step()
                opt.zero_grad(set_to_none=True)

            total_loss += float(loss.item()) * int(xb.shape[0])
            n += int(xb.shape[0])
            global_step += 1
            fake_cos_f = fake_cos.detach().to(torch.float32)
            fake_cos_sum += float(fake_cos_f.sum().item())
            probs = torch.sigmoid(logits.detach().to(torch.float32))
            if bool(include_condition_targets):
                target_count = torch.clamp(cond.sum(dim=1).to(torch.float32), min=1.0)
                target_prob = ((probs * cond).sum(dim=1) / target_count).mean()
                cond_target_prob_sum += float(target_prob.item()) * int(xb.shape[0])
            if disc_conf is not None:
                disc_conf_f = disc_conf.detach().to(torch.float32)
                disc_conf_sum += float(disc_conf_f.sum().item())
            else:
                disc_conf_f = None
            if pass_mask is not None:
                pass_count = int(pass_mask.to(torch.int64).sum().item())
                fail_count = int(int(xb.shape[0]) - pass_count)
                disc_pass_n += int(pass_count)
                disc_fail_n += int(fail_count)
                if pass_count > 0:
                    fake_cos_pass_sum += float(fake_cos_f[pass_mask].sum().item())
                if fail_count > 0:
                    fake_cos_fail_sum += float(fake_cos_f[~pass_mask].sum().item())
                if disc_conf_f is not None:
                    if pass_count > 0:
                        disc_conf_pass_sum += float(disc_conf_f[pass_mask].sum().item())
                    if fail_count > 0:
                        disc_conf_fail_sum += float(disc_conf_f[~pass_mask].sum().item())
                if disc_logits is not None:
                    if pass_count > 0:
                        disc_logit_pass_sum += float(disc_logits[pass_mask].sum().item())
                    if fail_count > 0:
                        disc_logit_fail_sum += float(disc_logits[~pass_mask].sum().item())

            if int(log_every) > 0 and (int(global_step) % int(log_every) == 0):
                elapsed = max(1e-6, time.time() - t_start)
                ips = float(n) / elapsed
                pass_rate = (float(disc_pass_n) / float(max(1, disc_pass_n + disc_fail_n))) if discriminator is not None else 0.0
                print(
                    f"[berkeley-fake-refresh] step={global_step}/{total_target_steps} "
                    f"loss={total_loss / max(1, n):.4f} samples={n} "
                    f"disc_pass_rate={pass_rate:.3f} fake_cos={fake_cos_sum / max(1, n):.3f} "
                    f"disc_conf={disc_conf_sum / max(1, n):.3f} "
                    f"samp_per_sec={ips:.1f}",
                    flush=True,
                )
            if step_preview_callback is not None:
                emit = False
                if int(log_every) > 0:
                    emit = ((int(global_step) % int(log_every) == 0) or (int(global_step) == int(total_target_steps)))
                else:
                    emit = (int(global_step) == int(total_target_steps))
                if emit and int(xb.shape[0]) > 0:
                    try:
                        step_preview_callback(
                            {
                                "global_step": int(global_step),
                                "total_steps": int(total_target_steps),
                                "img": xb[0].detach().to(torch.float32),
                                "probs": torch.sigmoid(logits[0].detach()).to(torch.float32),
                                "target_vec": cond[0].detach().to(torch.float32),
                                "loss": float(total_loss / max(1, n)),
                                "batch_loss": float(loss.detach().to(torch.float32).item()),
                                "fake_cos": float(fake_cos_f[0].detach().item()),
                                "disc_conf": (
                                    float(disc_conf_f[0].detach().item())
                                    if disc_conf_f is not None
                                    else 0.0
                                ),
                                "disc_logit": (
                                    float(disc_logits[0].detach().to(torch.float32).item())
                                    if disc_logits is not None
                                    else 0.0
                                ),
                                "disc_pass": (
                                    bool(pass_mask[0].detach().item())
                                    if pass_mask is not None
                                    else False
                                ),
                            }
                        )
                    except Exception as e:
                        _log(f"[stage-opengl] fake-refresh callback failed: {e}")
        if stop_now:
            break

    classifier.eval()
    generator.train(generator_was_training)
    if discriminator is not None:
        discriminator.train(discriminator_was_training)
    fake_cos_mean = (float(fake_cos_sum) / float(max(1, n))) if n > 0 else 0.0
    pass_total = max(1, disc_pass_n + disc_fail_n)
    disc_pass_rate = float(disc_pass_n) / float(pass_total)
    disc_fail_rate = float(disc_fail_n) / float(pass_total)
    fake_cos_pass = float(fake_cos_pass_sum) / float(max(1, disc_pass_n))
    fake_cos_fail = float(fake_cos_fail_sum) / float(max(1, disc_fail_n))
    disc_conf_mean = float(disc_conf_sum) / float(max(1, n))
    disc_conf_pass = float(disc_conf_pass_sum) / float(max(1, disc_pass_n))
    disc_conf_fail = float(disc_conf_fail_sum) / float(max(1, disc_fail_n))
    disc_logit_pass = float(disc_logit_pass_sum) / float(max(1, disc_pass_n))
    disc_logit_fail = float(disc_logit_fail_sum) / float(max(1, disc_fail_n))
    cond_target_prob_mean = float(cond_target_prob_sum) / float(max(1, n))
    return {
        "ran": bool(n > 0),
        "loss": total_loss / max(1, n),
        "samples": int(n),
        "stopped_early": bool(stop_now),
        "source": ("fake_vector_disc_feedback" if discriminator is not None else "fake_vector"),
        "elapsed_sec": float(time.time() - t_start),
        "fake_label_dim": int(fake_vec.numel()),
        "fake_cos_mean": float(fake_cos_mean),
        "fake_cos_pass": float(fake_cos_pass),
        "fake_cos_fail": float(fake_cos_fail),
        "disc_conf_mean": float(disc_conf_mean),
        "disc_conf_pass": float(disc_conf_pass),
        "disc_conf_fail": float(disc_conf_fail),
        "condition_target_prob_mean": float(cond_target_prob_mean),
        "disc_pass_samples": int(disc_pass_n),
        "disc_fail_samples": int(disc_fail_n),
        "disc_pass_rate": float(disc_pass_rate),
        "disc_fail_rate": float(disc_fail_rate),
        "disc_logit_pass": float(disc_logit_pass),
        "disc_logit_fail": float(disc_logit_fail),
    }


def _sample_stream_chunks_with_labels(
    streams: Sequence[np.ndarray],
    labels: Sequence[int],
    metas: Sequence[Dict],
    batch_size: int,
    chunk_samples: int,
    rng: np.random.Generator,
):
    xb = np.zeros((batch_size, chunk_samples), dtype=np.float32)
    yb = np.zeros((batch_size,), dtype=np.int64)
    picked = []
    eligible = [int(i) for i, s in enumerate(streams) if int(np.asarray(s).size) >= int(chunk_samples)]
    if len(eligible) <= 0:
        raise RuntimeError(
            f"No streams are long enough for chunk_samples={int(chunk_samples)} "
            f"(stream_count={len(streams)})."
        )
    for i in range(batch_size):
        j = int(eligible[int(rng.integers(0, len(eligible)))])
        s = streams[j]
        yb[i] = int(labels[j])
        meta_row = dict(metas[j]) if isinstance(metas[j], dict) else {}
        meta_row["_stream_index"] = int(j)
        picked.append(meta_row)
        start = int(rng.integers(0, int(s.size) - int(chunk_samples) + 1))
        xb[i, :] = np.asarray(s[start : start + int(chunk_samples)], dtype=np.float32)
    return xb, yb, picked


def _save_mono_wav(path: Path, mono_f32: np.ndarray, framerate: int):
    y = np.clip(mono_f32, -1.0, 1.0)
    i16 = np.round(y * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(framerate))
        wf.writeframes(i16.tobytes())


def _fit_wave_length(x: np.ndarray, target_samples: int, rng: np.random.Generator):
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


def _synthesize_structured_wave(target_samples: int, framerate: int, rng: np.random.Generator) -> np.ndarray:
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
            # Linear chirp in phase form for cheap generation.
            k = (f1 - f0) / max(1e-6, dur)
            phase = (2.0 * np.pi * ((f0 * t) + (0.5 * k * t * t))) + phase0
        else:
            phase = (2.0 * np.pi * f0 * t) + phase0
        amp = float(rng.uniform(0.3, 1.0))
        y += (amp * np.sin(phase)).astype(np.float32, copy=False)

    # Envelope + tremolo keeps shapes nonstationary and less noise-like.
    attack = int(min(n, max(0, int(float(rng.uniform(0.01, 0.12)) * n))))
    release = int(min(n, max(0, int(float(rng.uniform(0.08, 0.35)) * n))))
    env = np.ones((n,), dtype=np.float32)
    if attack > 1:
        env[:attack] = np.linspace(0.0, 1.0, attack, endpoint=False, dtype=np.float32)
    if release > 1:
        env[-release:] *= np.linspace(1.0, 0.0, release, endpoint=True, dtype=np.float32)
    mod_f = float(rng.uniform(0.4, 8.0))
    mod_depth = float(rng.uniform(0.1, 0.65))
    mod = (1.0 - mod_depth) + (mod_depth * (0.5 * (1.0 + np.sin((2.0 * np.pi * mod_f * t) + float(rng.uniform(0.0, 2.0 * np.pi))))))
    y = y * env * mod.astype(np.float32, copy=False)

    peak = float(np.max(np.abs(y))) if y.size > 0 else 0.0
    if peak > 1e-6:
        y = y / peak
    return y.astype(np.float32, copy=False)


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
    weights = np.asarray([0.20, 0.20, 0.15, 0.12, 0.10, 0.09, 0.07, 0.07], dtype=np.float64)
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


def _latent_noise_profile_key_from_path(path: str) -> str:
    name = Path(str(path)).name.lower()
    m = re.search(r"latent_\d+_([a-z0-9_]+)\.wav$", name)
    if m is None:
        return ""
    key = re.sub(r"\s+", "_", str(m.group(1))).strip().lower()
    if key in set(_latent_noise_profile_keys()):
        return str(key)
    return ""


def _latent_stream_semantic_terms_from_path(path: str) -> List[str]:
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


def _load_reinject_wave_paths(library_dir: Path, limit: int = 0):
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


def _bootstrap_latent_wav_pool(
    out_dir: Path,
    seed: int,
    count: int,
    framerate: int,
    seconds: float,
    noise_std: float,
    reinject_dir: str,
    reinject_ratio: float,
    reinject_copy_gain: float,
    reinject_noise_gain: float,
    structured_ratio: float,
    structured_gain: float,
    structured_noise_gain: float,
):
    rng = np.random.default_rng(int(seed) + 424242)
    sample_count = max(256, int(float(framerate) * max(0.05, float(seconds))))
    pool_dir = out_dir / "latent_wave_pool"
    noise_dir = pool_dir / "noise"
    mix_dir = pool_dir / "mix"
    noise_dir.mkdir(parents=True, exist_ok=True)
    mix_dir.mkdir(parents=True, exist_ok=True)

    if reinject_dir:
        lib_dir = Path(reinject_dir)
    else:
        lib_dir = out_dir / "accepted_wave_library"
    reinject_paths = _load_reinject_wave_paths(lib_dir, limit=100000)

    rows = []
    mixed_count = 0
    structured_count = 0
    noise_profile_counts: Dict[str, int] = {}
    for i in range(max(1, int(count))):
        noise_profile = _sample_latent_noise_profile_key(rng)
        noise_raw = _synthesize_profiled_noise_wave(
            target_samples=int(sample_count),
            framerate=int(framerate),
            profile_key=str(noise_profile),
            rng=rng,
        )
        noise = np.asarray(noise_raw, dtype=np.float32) * float(noise_std)
        noise_terms = _semantic_noise_terms_from_spectrum_sample(noise_raw)
        y = noise.copy()
        src = ""
        used_mix = False

        if len(reinject_paths) > 0 and float(rng.random()) < float(reinject_ratio):
            src = reinject_paths[int(rng.integers(0, len(reinject_paths)))]
            try:
                rec = read_wav_record(src)
                mono, _ = _decode_record_to_mono(rec, cfg=RenderConfig(), max_points=0)
                ref = _fit_wave_length(mono, target_samples=sample_count, rng=rng)
                peak = float(np.max(np.abs(ref))) if ref.size > 0 else 0.0
                if peak > 1e-6:
                    ref = ref / peak
                y = (float(reinject_noise_gain) * noise) + (float(reinject_copy_gain) * ref)
                used_mix = True
                mixed_count += 1
            except Exception:
                src = ""
                y = noise
        elif float(rng.random()) < float(structured_ratio):
            proto = _synthesize_structured_wave(
                target_samples=sample_count,
                framerate=framerate,
                rng=rng,
            )
            y = (float(structured_noise_gain) * noise) + (float(structured_gain) * proto)
            src = "__structured_seed__"
            used_mix = True
            structured_count += 1

        y = np.clip(y, -1.0, 1.0)
        target_dir = mix_dir if used_mix else noise_dir
        prof_slug = re.sub(r"[^a-z0-9_]+", "_", str(noise_profile).strip().lower()).strip("_")
        if not prof_slug:
            prof_slug = "gaussian_white_noise"
        out_path = target_dir / f"latent_{i:05d}_{prof_slug}.wav"
        _save_mono_wav(out_path, y, framerate=framerate)
        rows.append(
            {
                "file": str(out_path),
                "mixed": bool(used_mix),
                "source": src,
                "noise_profile": str(prof_slug),
                "noise_terms": list(noise_terms),
            }
        )
        noise_profile_counts[str(prof_slug)] = int(noise_profile_counts.get(str(prof_slug), 0)) + 1

    index_path = pool_dir / "index.jsonl"
    with index_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    manifest = pool_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "seed": int(seed),
                "count": int(count),
                "framerate": int(framerate),
                "seconds": float(seconds),
                "sample_count": int(sample_count),
                "noise_std": float(noise_std),
                "reinject_dir": str(lib_dir),
                "reinject_ratio": float(reinject_ratio),
                "reinject_copy_gain": float(reinject_copy_gain),
                "reinject_noise_gain": float(reinject_noise_gain),
                "reinject_candidates": len(reinject_paths),
                "mixed_count": int(mixed_count),
                "structured_ratio": float(structured_ratio),
                "structured_gain": float(structured_gain),
                "structured_noise_gain": float(structured_noise_gain),
                "structured_count": int(structured_count),
                "noise_profiles": {str(k): int(v) for k, v in sorted(noise_profile_counts.items(), key=lambda kv: str(kv[0]))},
                "index": str(index_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "pool_dir": str(pool_dir),
        "manifest": str(manifest),
        "count": int(count),
        "reinject_dir": str(lib_dir),
        "reinject_candidates": len(reinject_paths),
        "mixed_count": int(mixed_count),
        "structured_count": int(structured_count),
        "index": str(index_path),
        "noise_profiles": {str(k): int(v) for k, v in sorted(noise_profile_counts.items(), key=lambda kv: str(kv[0]))},
    }


def _build_wave_classifier_dataset_from_transformer(
    streams: Sequence[np.ndarray],
    labels: Sequence[int],
    metas: Sequence[Dict],
    cfg: RenderConfig,
    transformer: nn.Module,
    berkeley_classifier: nn.Module,
    sample_bits: int,
    image_hw: Tuple[int, int],
    chunk_samples: int,
    total_samples: int,
    batch_size: int,
    device: torch.device,
    rng_seed: int,
    accept_score_threshold: float,
    accept_l1_threshold: float,
    library_dir: Path,
    library_limit: int,
    cycle_id: int,
    round_id: int,
    accepted_only: bool = False,
    run_tag: str = "",
    library_serial_start: int = 0,
    amp_enabled: bool = False,
    amp_dtype: str = "float16",
    channels_last: bool = False,
    pin_memory: bool = False,
    semantic_class_names: Optional[Sequence[str]] = None,
    semantic_label_bank: Optional[np.ndarray] = None,
    stream_target_labels: Optional[Sequence[Any]] = None,
):
    rng = np.random.default_rng(rng_seed)
    x_all = []
    y_all = []
    accepted_rows = []
    transformer.eval()
    berkeley_classifier.eval()
    amp_dtype_t = _resolve_amp_dtype(amp_dtype) if amp_enabled else torch.float16

    target_samples = max(1, int(total_samples))
    accepted_only = bool(accepted_only)
    # When accepted_only is enabled, keep drawing until we reach the requested
    # accepted-set size (or a bounded attempt budget).
    max_draw_samples = target_samples if (not accepted_only) else max(target_samples, target_samples * 8)

    drawn = 0
    kept = 0
    dirty_wave_categories: set = set()
    while drawn < max_draw_samples and ((kept < target_samples) if accepted_only else (drawn < target_samples)):
        if accepted_only:
            b = min(int(batch_size), max_draw_samples - drawn)
        else:
            b = min(int(batch_size), target_samples - drawn)
        if b <= 0:
            break
        xb_np, yb_np, picked = _sample_stream_chunks_with_labels(
            streams=streams,
            labels=labels,
            metas=metas,
            batch_size=b,
            chunk_samples=chunk_samples,
            rng=rng,
        )
        xb = torch.from_numpy(xb_np)
        if pin_memory and device.type == "cuda":
            xb = xb.pin_memory()
        xb = xb.to(device, non_blocking=True)
        with torch.no_grad():
            with _autocast_context(device=device, enabled=amp_enabled, amp_dtype_t=amp_dtype_t):
                xh = transformer(xb)
                imgs = render_mono_wave_to_tensor(xh, cfg=cfg, image_hw=image_hw, sample_bits=sample_bits)
                if channels_last:
                    imgs = imgs.contiguous(memory_format=torch.channels_last)
                logits = berkeley_classifier(imgs)
            probs = torch.sigmoid(logits)
            k = max(1, min(3, int(probs.shape[1])))
            per_score = torch.topk(probs, k=k, dim=1).values.mean(dim=1)
            per_l1 = torch.mean(torch.abs(xh - xb), dim=1)

        accept_mask = (per_score >= float(accept_score_threshold)) & (per_l1 <= float(accept_l1_threshold))
        if accepted_only:
            if bool(torch.any(accept_mask)):
                keep_idx = torch.nonzero(accept_mask, as_tuple=False).squeeze(1)
                keep_np = keep_idx.detach().cpu().numpy()
                x_all.append(imgs.index_select(0, keep_idx).detach().cpu())
                y_all.append(torch.from_numpy(yb_np[keep_np]))
                kept += int(keep_idx.numel())
        else:
            x_all.append(imgs.detach().cpu())
            y_all.append(torch.from_numpy(yb_np))
            kept += int(b)
        drawn += int(b)

        if library_limit > 0 and len(accepted_rows) < int(library_limit):
            library_dir.mkdir(parents=True, exist_ok=True)
            index_path = library_dir / "index.jsonl"
            with index_path.open("a", encoding="utf-8") as f:
                for i in range(b):
                    sc = float(per_score[i].item())
                    l1 = float(per_l1[i].item())
                    if sc >= float(accept_score_threshold) and l1 <= float(accept_l1_threshold):
                        serial = int(library_serial_start) + len(accepted_rows)
                        if run_tag:
                            fn = f"{run_tag}_cycle{cycle_id:03d}_round{round_id:03d}_{serial:08d}.wav"
                        else:
                            fn = f"cycle{cycle_id:03d}_round{round_id:03d}_{serial:08d}.wav"
                        source_path = str(picked[i].get("path", ""))
                        wave_origin_bucket = _wave_output_native_bucket(source_path)
                        out_dir = library_dir / "waves" / wave_origin_bucket
                        out_dir.mkdir(parents=True, exist_ok=True)
                        out_path = out_dir / fn
                        _save_mono_wav(out_path, xh[i].detach().cpu().numpy(), framerate=int(picked[i]["framerate"]))
                        sem_terms: List[str] = []
                        source_terms = picked[i].get("semantic_terms", [])
                        if isinstance(source_terms, list):
                            sem_terms.extend([str(x) for x in source_terms])
                        stream_idx = int(picked[i].get("_stream_index", -1))
                        if (
                            stream_target_labels is not None
                            and semantic_class_names is not None
                            and 0 <= int(stream_idx) < int(len(stream_target_labels))
                        ):
                            src_target = stream_target_labels[int(stream_idx)]
                            if src_target is not None:
                                src_arr = np.asarray(src_target, dtype=np.float32).reshape(-1)
                                src_take = min(int(src_arr.size), int(len(semantic_class_names)))
                                if int(src_take) > 0:
                                    src_pos = np.where(np.asarray(src_arr[: int(src_take)], dtype=np.float32) >= 0.5)[0]
                                    for ti in src_pos.tolist():
                                        if 0 <= int(ti) < int(len(semantic_class_names)):
                                            sem_terms.append(str(semantic_class_names[int(ti)]))
                        src_label_idx = int(yb_np[i])
                        if semantic_class_names is not None and 0 <= src_label_idx < len(semantic_class_names):
                            sem_terms.append(str(semantic_class_names[src_label_idx]))
                        if semantic_class_names is not None and int(probs.shape[1]) > 0:
                            p_row = probs[i].detach().to(torch.float32).cpu().numpy().reshape(-1)
                            p_take = min(int(p_row.size), int(len(semantic_class_names)))
                            if int(p_take) > 0:
                                p_clip = np.asarray(p_row[: int(p_take)], dtype=np.float32)
                                pred_pos = np.where(p_clip >= 0.20)[0].astype(np.int64).tolist()
                                if int(len(pred_pos)) <= 0:
                                    pred_pos = [int(np.argmax(p_clip))]
                                for pi in pred_pos:
                                    if 0 <= int(pi) < int(len(semantic_class_names)):
                                        sem_terms.append(str(semantic_class_names[int(pi)]))
                        sem_terms.extend(
                            [
                                "regurgitated content",
                                "wave native output",
                                f"wave origin {wave_origin_bucket.replace('_', ' ')}",
                            ]
                        )
                        generated_noise_terms = _semantic_noise_terms_from_spectrum_sample(
                            xh[i].detach().cpu().numpy().reshape(-1)
                        )
                        if wave_origin_bucket == "latent_noise":
                            sem_terms.extend(list(generated_noise_terms))
                        elif wave_origin_bucket in {"latent_mix", "accepted_loopback", "structured_seed"}:
                            sem_terms.extend(["mixed noise and signal", "noise", "signal"])
                            sem_terms.extend(list(generated_noise_terms))
                        else:
                            sem_terms.append("signal")
                        sem_terms = _semantic_expand_inferred_tags(sem_terms)
                        row = {
                            "file": str(out_path),
                            "label": int(yb_np[i]),
                            "score": sc,
                            "l1": l1,
                            "source": source_path,
                            "semantic_terms": list(sem_terms),
                            "wave_category": str(wave_origin_bucket),
                            "cycle": int(cycle_id),
                            "round": int(round_id),
                        }
                        f.write(json.dumps(row) + "\n")
                        accepted_rows.append(row)
                        dirty_wave_categories.add(str(wave_origin_bucket))
                    if len(accepted_rows) >= int(library_limit):
                        break

    for wave_category in sorted([str(x) for x in dirty_wave_categories]):
        _refresh_wave_library_semantic_centroid(
            library_dir=library_dir,
            wave_category=wave_category,
            semantic_class_names=semantic_class_names,
            semantic_label_bank=semantic_label_bank,
        )

    if len(x_all) == 0:
        x = torch.empty((0, 3, int(image_hw[0]), int(image_hw[1])), dtype=torch.float32)
        y = torch.empty((0,), dtype=torch.long)
    else:
        x = torch.cat(x_all, dim=0)
        y = torch.cat(y_all, dim=0).long()
        if accepted_only and int(x.shape[0]) > target_samples:
            x = x[:target_samples]
            y = y[:target_samples]
    if accepted_only and int(x.shape[0]) < target_samples:
        print(
            f"[wave-cls-dataset] accepted_only target={target_samples} got={int(x.shape[0])} "
            f"(drawn={drawn}, max_draw={max_draw_samples})",
            flush=True,
        )
    return x, y, accepted_rows


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


def _decode_record_to_mono(record: WaveRecord, cfg: RenderConfig, max_points: int = 0):
    if cfg.use_channels == "auto":
        user_nch = record.nch if record.nch in (1, 2) else 2
    else:
        user_nch = int(cfg.use_channels)

    mode = _resolve_decode_mode(record.sampwidth, cfg.bitmode)
    samples_f32, samples_i32, sample_bits, _ = decode_pcm(record.frames, user_nch=user_nch, mode=mode)
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


def _spectral_encoding_unit_coords(x: np.ndarray, sr: int, bins: int = 128) -> np.ndarray:
    _ = sr
    b = max(8, int(bins))
    if x.size < 64:
        return np.zeros((int(b),), dtype=np.float64)
    n = min(4096, x.size)
    n = int(2 ** np.floor(np.log2(max(64, n))))
    y = x[:n].astype(np.float64, copy=False)
    y = y * np.hanning(n)
    spec = np.abs(np.fft.rfft(y))
    if not np.all(np.isfinite(spec)):
        return np.zeros((int(b),), dtype=np.float64)
    if float(np.sum(spec)) <= 1e-12:
        return np.zeros((int(b),), dtype=np.float64)
    spec = np.log1p(spec)
    src = np.linspace(0.0, 1.0, int(spec.size), dtype=np.float64)
    dst = np.linspace(0.0, 1.0, int(b), dtype=np.float64)
    coords = np.interp(dst, src, spec).astype(np.float64, copy=False)
    coords = coords - float(np.mean(coords))
    nrm = float(np.linalg.norm(coords))
    if nrm <= 1e-12:
        return np.zeros((int(b),), dtype=np.float64)
    return (coords / nrm).astype(np.float64, copy=False)


def _cosine_medoid_projection_scores(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=np.float64)
    if int(arr.ndim) != 2 or int(arr.shape[0]) <= 0 or int(arr.shape[1]) <= 0:
        return np.zeros((0,), dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    safe = arr / np.clip(norms, 1e-12, None)
    sims = safe @ safe.transpose(1, 0)
    mean_sim = np.mean(sims, axis=1)
    medoid_idx = int(np.argmax(mean_sim))
    anchor = safe[int(medoid_idx)]
    return (safe @ anchor).astype(np.float64, copy=False)


def _deterministic_hash_quantile_labels(
    records: Sequence[WaveRecord],
    data_root: str,
    bins: int,
) -> Tuple[np.ndarray, List[str]]:
    n = int(len(records))
    k = max(2, min(int(n), int(max(2, int(bins)))))
    if n <= 0:
        return np.zeros((0,), dtype=np.int64), []
    root = Path(str(data_root)).resolve()
    key_rows: List[Tuple[int, int]] = []
    for ri, rec in enumerate(records):
        p = Path(rec.path).resolve()
        try:
            rel_txt = str(p.relative_to(root)).replace("\\", "/")
        except Exception:
            rel_txt = str(p).replace("\\", "/")
        digest = hashlib.sha256(rel_txt.strip().lower().encode("utf-8")).digest()
        order_key = int.from_bytes(digest[:8], byteorder="little", signed=False)
        key_rows.append((int(order_key), int(ri)))
    key_rows.sort(key=lambda kv: (int(kv[0]), int(kv[1])))
    labels = np.zeros((int(n),), dtype=np.int64)
    for rank, (_, row_idx) in enumerate(key_rows):
        labels[int(row_idx)] = int((int(rank) * int(k)) // max(1, int(n)))
    class_names = [f"split_hash_bin_{i}" for i in range(int(k))]
    return labels.astype(np.int64, copy=False), class_names


def _build_labels(
    records: Sequence[WaveRecord],
    data_root: str,
    label_mode: str,
    pseudo_classes: int,
    single_class_fallback: str = "spectral_cosine_quantile",
):
    root = Path(data_root).resolve()
    if label_mode == "folder":
        names: List[str] = []
        for rec in records:
            p = Path(rec.path).resolve()
            try:
                rel = p.relative_to(root)
                if len(rel.parts) >= 2:
                    names.append(rel.parts[0])
                else:
                    names.append(p.parent.name)
            except Exception:
                names.append(p.parent.name)
        uniq = sorted(set(names))
        if len(uniq) >= 2:
            lut = {n: i for i, n in enumerate(uniq)}
            labels = np.array([lut[n] for n in names], dtype=np.int64)
            return labels, uniq, "folder"

        fallback_mode = str(single_class_fallback).strip().lower()
        if fallback_mode in ("path_hash_quantile", "path_hash", "deterministic_path_hash"):
            labels, class_names = _deterministic_hash_quantile_labels(
                records=records,
                data_root=str(data_root),
                bins=int(pseudo_classes),
            )
            return labels, class_names, "folder_hash_quantile"

        _log("Folder labels are single-class; switching to pseudo labels from cosine-unified spectral encoding bins.")

    pseudo_classes = max(2, int(pseudo_classes))
    base_cfg = RenderConfig()
    enc_coords = []
    for rec in records:
        mono, _ = _decode_record_to_mono(rec, base_cfg, max_points=65536)
        enc_coords.append(_spectral_encoding_unit_coords(mono, sr=rec.framerate, bins=128))
    enc_np = np.asarray(enc_coords, dtype=np.float64)
    vals = _cosine_medoid_projection_scores(enc_np)
    if int(vals.size) != int(len(records)):
        vals = np.zeros((int(len(records)),), dtype=np.float64)
    q = np.linspace(0.0, 1.0, pseudo_classes + 1)
    edges = np.quantile(vals, q)
    if np.allclose(edges, edges[0]):
        rank = np.argsort(np.argsort(vals))
        labels = (rank * pseudo_classes) // max(1, len(vals))
    else:
        labels = np.searchsorted(edges[1:-1], vals, side="right")
    labels = labels.astype(np.int64)
    class_names = [f"pseudo_cos_bin_{i}" for i in range(int(labels.max()) + 1)]
    return labels, class_names, "spectral_cosine_quantile"


def _stratified_split(labels: np.ndarray, train_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []
    labels = np.asarray(labels, dtype=np.int64)
    for cls in sorted(set(labels.tolist())):
        idx = np.where(labels == cls)[0]
        idx = np.array(idx, copy=True)
        rng.shuffle(idx)
        cut = int(round(len(idx) * train_frac))
        cut = min(max(cut, 1), len(idx) - 1) if len(idx) > 1 else 1
        train_idx.extend(idx[:cut].tolist())
        val_idx.extend(idx[cut:].tolist())

    if len(val_idx) == 0:
        rng.shuffle(train_idx)
        val_idx = train_idx[-max(1, len(train_idx) // 5) :]
        train_idx = train_idx[: -len(val_idx)]

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def _sample_indices(indices: Sequence[int], limit: int, rng: np.random.Generator):
    idx = list(indices)
    if len(idx) <= 0:
        return []
    if limit <= 0 or len(idx) <= limit:
        out = [int(x) for x in idx]
        rng.shuffle(out)
        return out
    out = rng.choice(np.array(idx, dtype=np.int64), size=int(limit), replace=False).astype(np.int64).tolist()
    rng.shuffle(out)
    return [int(x) for x in out]


def _render_indices_to_tensors(
    records: Sequence[WaveRecord],
    labels: np.ndarray,
    indices: Sequence[int],
    cfg: RenderConfig,
    image_hw: Tuple[int, int],
):
    x_list = []
    y_list = []
    bits = []
    for idx in indices:
        img_u8, meta = render_record(records[idx], cfg)
        x_list.append(image_u8_to_tensor(img_u8, image_hw=image_hw))
        y_list.append(int(labels[idx]))
        if meta.get("sample_bits") is not None:
            bits.append(int(meta["sample_bits"]))
    x = torch.stack(x_list, dim=0)
    y = torch.tensor(y_list, dtype=torch.long)
    return x, y, bits


def _random_configs(num_trials: int, rng: np.random.Generator, max_points: int):
    color_choices = [m[0] for m in COLOR_MODES]
    widths = [256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
    downs = [1, 2, 3, 4, 6, 8]
    mono_choices = ["Mix", "L", "R", "LR stride"]

    cfgs = [
        RenderConfig(
            width=1024,
            downsample=1,
            color_mode=COLOR_MODES[0][0],
            mono_pick="Mix",
            bitmask_enable=False,
            max_points=max_points,
        )
    ]
    for _ in range(max(0, int(num_trials) - 1)):
        use_mask = bool(rng.random() < 0.5)
        lo = int(rng.integers(0, 12))
        hi = int(rng.integers(lo, min(16, lo + 9)))
        cfgs.append(
            RenderConfig(
                width=int(rng.choice(widths)),
                downsample=int(rng.choice(downs)),
                color_mode=str(rng.choice(color_choices)),
                mono_pick=str(rng.choice(mono_choices)),
                empty_fill=0,
                bitmask_enable=use_mask,
                bitmask_low=lo,
                bitmask_high=hi,
                max_points=max_points,
            )
        )
    return cfgs


def _try_scipy_refine(best_cfg: RenderConfig, max_points: int, enabled: bool):
    if not enabled:
        return [best_cfg]
    try:
        from scipy import optimize  # noqa: F401
    except Exception:
        _log("scipy is not installed; skipping scipy config refinement.")
        return [best_cfg]

    _log("scipy detected; running lightweight local config jitter around the current best.")
    rng = np.random.default_rng(777)
    width_jitter = [max(128, best_cfg.width + int(k) * 64) for k in range(-2, 3)]
    ds_jitter = [max(1, best_cfg.downsample + k) for k in (-1, 0, 1)]
    mask_lows = [max(0, best_cfg.bitmask_low + k) for k in (-2, -1, 0, 1, 2)]
    mask_highs = [max(0, best_cfg.bitmask_high + k) for k in (-2, -1, 0, 1, 2)]
    cfgs = []
    for _ in range(6):
        cfgs.append(
            RenderConfig(
                bitmode=best_cfg.bitmode,
                use_channels=best_cfg.use_channels,
                mono_pick=best_cfg.mono_pick,
                width=int(rng.choice(width_jitter)),
                downsample=int(rng.choice(ds_jitter)),
                color_mode=best_cfg.color_mode,
                empty_fill=best_cfg.empty_fill,
                bitmask_enable=best_cfg.bitmask_enable,
                bitmask_low=int(rng.choice(mask_lows)),
                bitmask_high=int(rng.choice(mask_highs)),
                max_points=max_points,
            )
        )
    return [best_cfg] + cfgs


def _prepare_streams(
    records: Sequence[WaveRecord],
    labels: np.ndarray,
    indices: Sequence[int],
    cfg: RenderConfig,
    max_points: int,
):
    streams = []
    ys = []
    bits = []
    meta = []
    for idx in indices:
        mono, bit_depth = _decode_record_to_mono(records[idx], cfg, max_points=max_points)
        if mono.size == 0:
            continue
        path_txt = str(records[idx].path)
        generated_terms = _latent_stream_semantic_terms_from_path(path_txt)
        streams.append(mono.astype(np.float32, copy=False))
        ys.append(int(labels[idx]))
        bits.append(int(bit_depth))
        meta.append(
            {
                "path": path_txt,
                "framerate": int(records[idx].framerate),
                "semantic_terms": list(generated_terms),
            }
        )
    return streams, ys, bits, meta


def _filter_stream_pool_for_chunk_samples(
    streams: Sequence[np.ndarray],
    labels: Sequence[int],
    metas: Sequence[Dict],
    chunk_samples: int,
    pool_name: str,
    target_labels: Optional[Sequence[Any]] = None,
):
    n_need = max(1, int(chunk_samples))
    keep = [int(i) for i, s in enumerate(streams) if int(np.asarray(s).size) >= n_need]
    if len(keep) <= 0:
        raise RuntimeError(
            f"{pool_name}: no streams are long enough for chunk_samples={n_need}. "
            f"stream_count={len(streams)}. "
            "Provide longer source WAVs or increase --latent-fallback-seconds."
        )
    out_streams = [np.asarray(streams[i], dtype=np.float32) for i in keep]
    out_labels = [int(labels[i]) for i in keep]
    out_metas = [dict(metas[i]) for i in keep]
    out_targets = None
    if target_labels is not None:
        out_targets = []
        for i in keep:
            tv = target_labels[i]
            if tv is None:
                out_targets.append(None)
            else:
                arr = np.asarray(tv, dtype=np.float32).reshape(-1)
                out_targets.append(arr.astype(np.float32, copy=False))
    info = {
        "pool": str(pool_name),
        "required_chunk_samples": int(n_need),
        "before": int(len(streams)),
        "after": int(len(out_streams)),
        "dropped": int(len(streams) - len(out_streams)),
    }
    return out_streams, out_labels, out_metas, out_targets, info


def _write_mono_wav(path: str, mono_f32: np.ndarray, framerate: int):
    y = np.clip(mono_f32, -1.0, 1.0)
    i16 = np.round(y * 32767.0).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(framerate))
        wf.writeframes(i16.tobytes())


def _append_accepted_rows_to_transformer_streams(
    rows: Sequence[Dict],
    seen_paths: set,
    train_streams: List[np.ndarray],
    train_stream_labels: List[int],
    train_meta: List[Dict],
    max_points: int,
    library_root: Optional[Path] = None,
    on_append_row: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> int:
    added = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        p = str(row.get("file", "")).strip()
        if not p:
            continue
        resolved = None
        if library_root is not None:
            resolved = _resolve_library_member_path(p, library_dir=library_root)
            if resolved is None:
                continue
            p = str(resolved)
        if p in seen_paths:
            continue
        try:
            rec = read_wav_record(p)
            mono, _ = _decode_record_to_mono(rec, cfg=RenderConfig(), max_points=max_points)
            if mono.size <= 0:
                continue
            train_streams.append(mono.astype(np.float32, copy=False))
            train_stream_labels.append(int(row.get("label", 0)))
            row_terms_raw = (
                [str(x) for x in row.get("semantic_terms", [])]
                if isinstance(row.get("semantic_terms", []), list)
                else []
            )
            row_terms_raw.extend(_latent_stream_semantic_terms_from_path(str(row.get("source", ""))))
            train_meta.append(
                {
                    "path": str(p),
                    "framerate": int(rec.framerate),
                    "semantic_terms": _semantic_expand_inferred_tags(row_terms_raw),
                }
            )
            if callable(on_append_row):
                try:
                    on_append_row(dict(row))
                except Exception:
                    pass
            seen_paths.add(p)
            added += 1
        except Exception:
            continue
    return int(added)


def _repair_accepted_library_index_semantics(library_dir: Path) -> Dict[str, Any]:
    idx_path = Path(library_dir) / "index.jsonl"
    if not idx_path.exists():
        return {
            "ran": False,
            "rows": 0,
            "updated": 0,
            "parse_failures": 0,
            "reason": "missing_index",
            "path": str(idx_path),
        }
    try:
        raw_lines = idx_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return {
            "ran": False,
            "rows": 0,
            "updated": 0,
            "parse_failures": 0,
            "reason": "read_failed",
            "path": str(idx_path),
        }

    out_lines: List[str] = []
    total_rows = 0
    updated_rows = 0
    parse_failures = 0
    for line in raw_lines:
        s = str(line).strip()
        if not s:
            continue
        total_rows += 1
        try:
            row = json.loads(s)
        except Exception:
            parse_failures += 1
            out_lines.append(s)
            continue
        if not isinstance(row, dict):
            out_lines.append(s)
            continue
        old_terms_raw = row.get("semantic_terms", [])
        old_terms = [str(x) for x in old_terms_raw] if isinstance(old_terms_raw, list) else []
        wave_cat = re.sub(r"\s+", " ", str(row.get("wave_category", ""))).strip().lower()
        terms = list(old_terms)
        src_terms = _latent_stream_semantic_terms_from_path(str(row.get("source", "")))
        if len(src_terms) > 0:
            terms.extend(list(src_terms))
        if wave_cat in {"latent_mix", "accepted_loopback", "structured_seed"}:
            terms.extend(["mixed noise and signal", "noise", "signal"])
        elif wave_cat == "latent_noise":
            terms.extend(["noise"])
        new_terms = _semantic_expand_inferred_tags(terms)
        old_norm = _normalize_vocab_terms(old_terms)
        if list(new_terms) != list(old_norm):
            updated_rows += 1
        row["semantic_terms"] = list(new_terms)
        out_lines.append(json.dumps(row, ensure_ascii=True))

    if int(updated_rows) > 0:
        tmp_path = idx_path.with_suffix(".jsonl.tmp")
        tmp_path.write_text(
            "\n".join(out_lines) + ("\n" if len(out_lines) > 0 else ""),
            encoding="utf-8",
        )
        tmp_path.replace(idx_path)
    return {
        "ran": True,
        "rows": int(total_rows),
        "updated": int(updated_rows),
        "parse_failures": int(parse_failures),
        "reason": "ok",
        "path": str(idx_path),
    }


def _latent_pool_stream_kind(path: str) -> str:
    p = str(path).replace("\\", "/").lower()
    if "/latent_wave_pool/noise/" in p:
        return "noise"
    if "/latent_wave_pool/mix/" in p:
        return "mix"
    return "other"


def _wave_output_native_bucket(path: str) -> str:
    p = str(path).replace("\\", "/").lower().strip()
    if not p:
        return "unknown"
    if p == "__structured_seed__":
        return "structured_seed"
    if "/latent_wave_pool/noise/" in p:
        return "latent_noise"
    if "/latent_wave_pool/mix/" in p:
        return "latent_mix"
    if "/accepted_wave_library/" in p:
        return "accepted_loopback"
    return "external_source"


def _wave_semantic_label_from_weighted_angular_centroid(
    term_counts: Dict[str, int],
    semantic_class_names: Optional[Sequence[str]],
    semantic_label_bank: Optional[np.ndarray],
) -> Tuple[str, float, int]:
    if semantic_class_names is None or semantic_label_bank is None:
        return "", 0.0, 0
    class_names = [str(x) for x in list(semantic_class_names)]
    if len(class_names) <= 0:
        return "", 0.0, 0
    bank_arr = np.asarray(semantic_label_bank, dtype=np.float32)
    if int(bank_arr.ndim) != 2:
        return "", 0.0, 0
    rows = min(int(bank_arr.shape[0]), int(len(class_names)))
    if rows <= 0:
        return "", 0.0, 0
    bank_norm = _normalize_l2_rows_np(bank_arr[: int(rows), :].astype(np.float32, copy=False))
    name_to_idx: Dict[str, int] = {}
    for i, name in enumerate(class_names[: int(rows)]):
        key = re.sub(r"\s+", " ", str(name)).strip().lower()
        if key and key not in name_to_idx:
            name_to_idx[key] = int(i)
    weights = np.zeros((int(rows),), dtype=np.float32)
    for term, cnt in term_counts.items():
        key = re.sub(r"\s+", " ", str(term)).strip().lower()
        if not key:
            continue
        idx = name_to_idx.get(key, None)
        if idx is None:
            continue
        w = float(max(0, int(cnt)))
        if w <= 0.0:
            continue
        weights[int(idx)] += float(w)
    active = np.where(weights > 0.0)[0].astype(np.int64)
    if int(active.size) <= 0:
        return "", 0.0, 0
    centroid = np.sum(bank_norm[active, :] * weights[active][:, None], axis=0)
    denom = float(np.linalg.norm(centroid))
    if denom <= 1e-12:
        return "", 0.0, 0
    centroid = (centroid / denom).astype(np.float32, copy=False)
    sims = np.matmul(bank_norm, centroid.astype(np.float32, copy=False))
    best_idx = int(np.argmax(sims))
    if best_idx < 0 or best_idx >= int(rows):
        return "", 0.0, int(active.size)
    return str(class_names[int(best_idx)]), float(sims[best_idx]), int(active.size)


def _refresh_wave_library_semantic_centroid(
    library_dir: Path,
    wave_category: str,
    semantic_class_names: Optional[Sequence[str]],
    semantic_label_bank: Optional[np.ndarray],
) -> Dict[str, Any]:
    index_path = library_dir / "index.jsonl"
    category_key = str(wave_category).strip() or "unknown"
    waves_dir = library_dir / "waves" / category_key
    waves_dir.mkdir(parents=True, exist_ok=True)
    out_path = waves_dir / "_semantic_centroid.json"
    info: Dict[str, Any] = {
        "refreshed": False,
        "reason": "",
        "index_path": str(index_path),
        "output_path": str(out_path),
        "wave_category": str(category_key),
    }
    if not index_path.exists():
        info["reason"] = "missing_index"
        return info
    term_counter: Counter = Counter()
    try:
        with index_path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    row = json.loads(s)
                except Exception:
                    continue
                row_category = str(row.get("wave_category", "")).strip() or "unknown"
                if str(row_category) != str(category_key):
                    continue
                terms = row.get("semantic_terms", [])
                if not isinstance(terms, list):
                    continue
                for t in terms:
                    key = re.sub(r"\s+", " ", str(t)).strip().lower()
                    if key:
                        term_counter[str(key)] += 1
    except Exception as ex:
        info["reason"] = f"scan_failed:{type(ex).__name__}"
        return info
    label, score, mapped = _wave_semantic_label_from_weighted_angular_centroid(
        term_counts={str(k): int(v) for k, v in term_counter.items()},
        semantic_class_names=semantic_class_names,
        semantic_label_bank=semantic_label_bank,
    )
    if not str(label).strip():
        if len(term_counter) > 0:
            label = str(term_counter.most_common(1)[0][0])
        else:
            label = "unknown"
        score = 0.0
    payload = {
        "updated_at": float(time.time()),
        "label": str(label),
        "score": float(score),
        "mapped_terms": int(mapped),
        "label_counts_top": [[str(k), int(v)] for k, v in term_counter.most_common(32)],
        "index_path": str(index_path),
        "wave_folder": str((Path("waves") / category_key).as_posix()),
        "wave_category": str(category_key),
        "centroid_mode": "weighted_angular_centroid_projection",
    }
    try:
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as ex:
        info["reason"] = f"write_failed:{type(ex).__name__}"
        return info
    info["refreshed"] = True
    info["reason"] = "ok"
    info["label"] = str(label)
    info["score"] = float(score)
    info["mapped_terms"] = int(mapped)
    return info


def _tensor_chw_to_rgb_u8(img: torch.Tensor) -> np.ndarray:
    x = img.detach().to(torch.float32)
    if x.ndim == 4:
        x = x[0]
    if x.ndim != 3:
        raise ValueError(f"Expected CHW image tensor, got shape={tuple(x.shape)}")
    if int(x.shape[0]) == 1:
        x = x.repeat(3, 1, 1)
    x = torch.clamp(x, 0.0, 1.0)
    arr = x.cpu().numpy().astype(np.float32, copy=False)
    arr = np.transpose(arr, (1, 2, 0))
    return np.round(arr * 255.0).astype(np.uint8)


def _rgb_u8_to_tensor_chw01(img_u8: np.ndarray) -> torch.Tensor:
    arr = np.asarray(img_u8, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3:
        raise ValueError(f"Expected HWC image array, got shape={tuple(arr.shape)}")
    if int(arr.shape[2]) == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif int(arr.shape[2]) > 3:
        arr = arr[:, :, :3]
    t = torch.from_numpy(arr.astype(np.float32, copy=False) / 255.0)
    return t.permute(2, 0, 1).contiguous()


def _overlay_text_rows_u8(image_u8: np.ndarray, header: str, rows: Sequence[str]) -> np.ndarray:
    out = np.asarray(image_u8, dtype=np.uint8).copy()
    text_rows: List[str] = []
    if str(header).strip():
        text_rows.append(str(header).strip())
    for r in rows:
        txt = str(r).strip()
        if txt:
            text_rows.append(txt)
    if len(text_rows) <= 0:
        return out
    try:
        from PIL import Image, ImageDraw, ImageFont

        im = Image.fromarray(out).convert("RGBA")
        font = ImageFont.load_default()
        pad = 2
        line_h = 11
        h = int(im.height)
        w = int(im.width)
        max_lines_per_col = max(1, (h - (pad * 2)) // line_h)
        n_cols = max(1, int(math.ceil(float(len(text_rows)) / float(max_lines_per_col))))
        col_w = max(1, w // n_cols)
        box_h = min(h, (pad * 2) + (line_h * max_lines_per_col))

        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw_bg = ImageDraw.Draw(overlay)
        draw_bg.rectangle([(0, 0), (w - 1, box_h)], fill=(0, 0, 0, 110))
        im = Image.alpha_composite(im, overlay)

        draw = ImageDraw.Draw(im)
        for i, txt in enumerate(text_rows):
            col = int(i // max_lines_per_col)
            row = int(i % max_lines_per_col)
            x = int(pad + (col * col_w))
            y = int(pad + (row * line_h))
            if x >= (w - pad):
                break
            max_chars = max(16, int((col_w - (pad * 2)) // 5))
            color = (255, 225, 64, 255) if i == 0 else (240, 240, 240, 255)
            draw.text((x, y), str(txt)[:max_chars], fill=color, font=font)
        return np.asarray(im.convert("RGB"), dtype=np.uint8)
    except Exception:
        shade_h = min(int(out.shape[0]), 18 + (10 * len(text_rows)))
        out[:shade_h, :, :] = (out[:shade_h, :, :].astype(np.uint16) * 65 // 100).astype(np.uint8)
        return out


def _annotate_tensor_image(img: torch.Tensor, header: str, rows: Sequence[str]) -> torch.Tensor:
    base = _tensor_chw_to_rgb_u8(img)
    anno = _overlay_text_rows_u8(base, header=header, rows=rows)
    return _rgb_u8_to_tensor_chw01(anno)


def _concat_rgb_panels_h(images: Sequence[np.ndarray], pad: int = 3, fill: int = 0) -> np.ndarray:
    if len(images) <= 0:
        return np.full((8, 8, 3), int(fill) & 0xFF, dtype=np.uint8)
    imgs = [np.asarray(im, dtype=np.uint8) for im in images]
    h = max(int(im.shape[0]) for im in imgs)
    w = sum(int(im.shape[1]) for im in imgs) + (max(0, len(imgs) - 1) * int(max(0, pad)))
    out = np.full((int(h), int(w), 3), int(fill) & 0xFF, dtype=np.uint8)
    x = 0
    for i, im in enumerate(imgs):
        ih, iw = int(im.shape[0]), int(im.shape[1])
        y0 = max(0, (h - ih) // 2)
        out[y0 : y0 + ih, x : x + iw, :] = im
        x += iw
        if i + 1 < len(imgs):
            x += int(max(0, pad))
    return out


def _write_rgb_preview(path: Path, image_u8: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        Image.fromarray(image_u8).save(str(path))
        return
    except Exception:
        pass
    ppm_path = path.with_suffix(".ppm")
    h, w = int(image_u8.shape[0]), int(image_u8.shape[1])
    with ppm_path.open("wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(image_u8.astype(np.uint8, copy=False).tobytes())


def _top_labels_from_probs(probs: np.ndarray, class_names: Sequence[str], topk: int) -> List[Dict]:
    if probs.size <= 0:
        return []
    k = max(1, min(int(topk), int(probs.size)))
    idx = np.argsort(-probs)[:k]
    out: List[Dict] = []
    for cls_idx in idx.tolist():
        c = int(cls_idx)
        name = str(class_names[c]) if c < len(class_names) else f"class_{c}"
        out.append(
            {
                "class_idx": c,
                "class_name": name,
                "score": float(probs[c]),
            }
        )
    return out


def _format_top_label_lines(rows: Sequence[Dict], max_items: int = 3) -> List[str]:
    out: List[str] = []
    k = max(0, int(max_items))
    rows_list = list(rows)
    if k <= 0:
        use_rows = rows_list
    else:
        use_rows = rows_list[:k]
    for r in use_rows:
        out.append(f"{r.get('class_name', '?')}:{float(r.get('score', 0.0)):.3f}")
    return out


def _semantic_target_entries_from_vector(
    cond_vec: Any,
    class_names: Sequence[str],
    max_items: int = 8,
    threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    arr = np.asarray(cond_vec, dtype=np.float32).reshape(-1)
    if int(arr.size) <= 0:
        return []
    t = float(threshold)
    idx = np.where(arr >= t)[0].astype(np.int64)
    if int(idx.size) <= 0:
        return []
    order = idx[np.argsort(-arr[idx])]
    k = max(1, int(max_items))
    out: List[Dict[str, Any]] = []
    for cls_idx in order[:k].tolist():
        c = int(cls_idx)
        out.append(
            {
                "class_idx": c,
                "class_name": str(class_names[c]) if c < len(class_names) else f"class_{c}",
                "weight": float(arr[c]),
            }
        )
    return out


def _format_target_line_from_condition(
    cond_vec: Any,
    class_names: Sequence[str],
    max_items: int = 4,
    threshold: float = 0.5,
) -> str:
    if isinstance(cond_vec, torch.Tensor):
        arr = cond_vec.detach().to(torch.float32).cpu().numpy().reshape(-1)
    else:
        arr = np.asarray(cond_vec, dtype=np.float32).reshape(-1)
    if int(arr.size) <= 0:
        return "target(0):none"
    t = float(threshold)
    idx = np.where(arr >= t)[0].astype(np.int64)
    if int(idx.size) <= 0:
        return "target(0):none"
    order = idx[np.argsort(-arr[idx])]
    if int(max_items) <= 0:
        k = int(order.size)
    else:
        k = max(1, int(max_items))
    picks = order[:k].tolist()
    names = [str(class_names[int(i)]) if int(i) < len(class_names) else f"class_{int(i)}" for i in picks]
    more = ""
    if int(order.size) > k:
        more = f"+{int(order.size) - k}"
    return f"target({int(order.size)}):{','.join(names)}{more}"


@torch.no_grad()
def _render_and_score_wave(
    wave_f32: np.ndarray,
    cfg: RenderConfig,
    sample_bits: int,
    image_hw: Tuple[int, int],
    classifier: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: str,
    channels_last: bool,
):
    xb = torch.from_numpy(np.asarray(wave_f32, dtype=np.float32)[None, :]).to(device)
    amp_dtype_t = _resolve_amp_dtype(amp_dtype) if amp_enabled else torch.float16
    with _autocast_context(device=device, enabled=amp_enabled, amp_dtype_t=amp_dtype_t):
        img = render_mono_wave_to_tensor(xb, cfg=cfg, image_hw=image_hw, sample_bits=int(sample_bits))
        if channels_last:
            img = img.contiguous(memory_format=torch.channels_last)
        logits = classifier(img)
        probs = torch.sigmoid(logits[0]).to(torch.float32)
    return _tensor_chw_to_rgb_u8(img[0]), probs.detach().cpu().numpy().astype(np.float32, copy=False)


def _save_round_supervision_preview(
    preview_dir: Path,
    stage: str,
    cycle_id: int,
    round_id: int,
    sample_idx: int,
    clean_wave: np.ndarray,
    enhanced_wave: np.ndarray,
    cfg: RenderConfig,
    sample_bits: int,
    image_hw: Tuple[int, int],
    classifier: nn.Module,
    class_names: Sequence[str],
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: str,
    channels_last: bool,
    topk: int,
    target_condition: Optional[Any] = None,
    extra: Optional[Dict] = None,
):
    clean_img, clean_probs = _render_and_score_wave(
        wave_f32=clean_wave,
        cfg=cfg,
        sample_bits=int(sample_bits),
        image_hw=image_hw,
        classifier=classifier,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        channels_last=channels_last,
    )
    out_img, out_probs = _render_and_score_wave(
        wave_f32=enhanced_wave,
        cfg=cfg,
        sample_bits=int(sample_bits),
        image_hw=image_hw,
        classifier=classifier,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        channels_last=channels_last,
    )
    panel = _concat_rgb_panels_h([clean_img, out_img], pad=4, fill=0)
    base = f"cycle_{int(cycle_id):04d}_round_{int(round_id):04d}_stage_{str(stage).lower()}_sample_{int(sample_idx):03d}"
    panel_path = preview_dir / f"{base}.png"
    meta_path = preview_dir / f"{base}.json"
    _write_rgb_preview(panel_path, panel)
    all_topk = max(1, min(int(clean_probs.size), len(class_names)))
    top_clean = _top_labels_from_probs(clean_probs, class_names=class_names, topk=int(all_topk))
    top_out = _top_labels_from_probs(out_probs, class_names=class_names, topk=int(all_topk))
    top_delta = _top_labels_from_probs((out_probs - clean_probs), class_names=class_names, topk=int(all_topk))
    payload: Dict = {
        "cycle": int(cycle_id),
        "round": int(round_id),
        "stage": str(stage),
        "sample_idx": int(sample_idx),
        "target_semantic": (
            _semantic_target_entries_from_vector(
                target_condition,
                class_names=class_names,
                max_items=max(1, len(class_names)),
                threshold=0.5,
            )
            if target_condition is not None
            else []
        ),
        "top_labels_clean": top_clean,
        "top_labels_enhanced": top_out,
        "top_label_deltas": top_delta,
        "panel": str(panel_path),
    }
    if isinstance(extra, dict):
        payload["extra"] = dict(extra)
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _expand_payload_conditions_with_semantic_bank(
    payload_conditions: Sequence[Any],
    condition_num_classes: int,
    supervised_num_classes: int,
    label_embedding_bank: Optional[np.ndarray],
    semantic_target_temperature: float = 8.0,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    c = max(1, int(condition_num_classes))
    n_rows = int(len(payload_conditions))
    if n_rows <= 0:
        return [], {"expanded": False, "reason": "empty_rows", "rows": 0, "condition_dim": int(c)}

    sup = max(1, min(int(supervised_num_classes), int(c)))
    base_rows: List[np.ndarray] = []
    sup_rows: List[np.ndarray] = []
    for i, row in enumerate(payload_conditions):
        arr = np.asarray(row, dtype=np.float32).reshape(-1)
        if int(arr.size) <= 0:
            raise RuntimeError(f"Semantic payload row {i} is empty.")
        if int(arr.size) > int(c):
            raise RuntimeError(
                "Semantic payload row width exceeds active semantic width; refusing truncation. "
                f"row={int(i)} row_dim={int(arr.size)} condition_dim={int(c)}"
            )

        if int(arr.size) >= int(c):
            arr_full = arr[: int(c)]
        else:
            pad = np.zeros((int(c) - int(arr.size),), dtype=np.float32)
            arr_full = np.concatenate([arr, pad], axis=0)
        if int(arr.size) >= int(sup):
            arr_sup = arr[: int(sup)]
        else:
            pad_sup = np.zeros((int(sup) - int(arr.size),), dtype=np.float32)
            arr_sup = np.concatenate([arr, pad_sup], axis=0)

        base_rows.append(np.clip(arr_full, 0.0, 1.0).astype(np.float32, copy=False))
        sup_rows.append(np.clip(arr_sup, 0.0, 1.0).astype(np.float32, copy=False))

    info: Dict[str, Any] = {
        "expanded": False,
        "reason": "",
        "rows": int(n_rows),
        "condition_dim": int(c),
        "supervised_dim": int(sup),
        "extra_dim": int(max(0, int(c) - int(sup))),
    }
    # Keep payload labels sparse and literal: preserve provided supervised slots and leave semantic extras at 0.
    # Semantic extras are added explicitly by term tags later in the pipeline.
    info["reason"] = "supervised_passthrough_full_width"
    info["expanded"] = bool(int(c) > int(sup))
    info["extra_mean"] = 0.0
    info["extra_max"] = 0.0
    return base_rows, info


def _payload_condition_bank_tensor(
    payload_targets: Sequence[Any],
    num_classes: int,
) -> torch.Tensor:
    n = int(len(payload_targets))
    c = max(1, int(num_classes))
    if n <= 0:
        return torch.zeros((0, c), dtype=torch.float32)
    bank = np.zeros((n, c), dtype=np.float32)
    for i in range(n):
        arr = np.asarray(payload_targets[i], dtype=np.float32).reshape(-1)
        if int(arr.size) <= 0:
            raise RuntimeError(f"Semantic condition row {i} is empty.")
        if int(arr.size) != int(c):
            raise RuntimeError(
                f"Semantic condition width mismatch at row {i}: got={int(arr.size)} expected={int(c)}"
            )
        bank[i, :] = arr
    return torch.from_numpy(bank.astype(np.float32, copy=False))


@torch.no_grad()
def _save_generator_supervision_preview(
    preview_dir: Path,
    cycle_id: int,
    round_id: int,
    generator: nn.Module,
    classifier: nn.Module,
    payload_conditions: Sequence[Any],
    num_classes: int,
    class_names: Sequence[str],
    image_hw: Tuple[int, int],
    z_dim: int,
    samples: int,
    topk: int,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: str,
    channels_last: bool,
    seed: int,
):
    if len(payload_conditions) <= 0:
        return None
    cond_bank_cpu = _payload_condition_bank_tensor(
        payload_targets=payload_conditions,
        num_classes=int(num_classes),
    )
    if int(cond_bank_cpu.shape[0]) <= 0:
        return None
    n = max(1, int(samples))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, int(cond_bank_cpu.shape[0]), size=n)
    idx_t = torch.as_tensor(idx, device=device, dtype=torch.long)
    cond_bank = cond_bank_cpu.to(device=device, dtype=torch.float32)
    cond = cond_bank.index_select(0, idx_t).to(device=device, dtype=torch.float32)
    z = torch.randn((int(cond.shape[0]), max(8, int(z_dim))), device=device)
    was_gen_train = bool(generator.training)
    was_cls_train = bool(classifier.training)
    generator.eval()
    classifier.eval()
    amp_dtype_t = _resolve_amp_dtype(amp_dtype) if amp_enabled else torch.float16
    with _autocast_context(device=device, enabled=amp_enabled, amp_dtype_t=amp_dtype_t):
        fake = generator(z, cond).to(torch.float32)
        if channels_last:
            fake = fake.contiguous(memory_format=torch.channels_last)
        logits = classifier(fake)
        probs = torch.sigmoid(logits).to(torch.float32)
    if was_gen_train:
        generator.train()
    if was_cls_train:
        classifier.train()

    images = [_tensor_chw_to_rgb_u8(fake[i]) for i in range(int(fake.shape[0]))]
    panel = _concat_rgb_panels_h(images, pad=3, fill=0)
    base = f"cycle_{int(cycle_id):04d}_round_{int(round_id):04d}_stage_g_samples"
    panel_path = preview_dir / f"{base}.png"
    meta_path = preview_dir / f"{base}.json"
    _write_rgb_preview(panel_path, panel)
    rows: List[Dict] = []
    topk_all = max(1, min(len(class_names), int(probs.shape[1])))
    for i in range(int(fake.shape[0])):
        p = probs[i].detach().cpu().numpy().astype(np.float32, copy=False)
        target_cond = cond[i].detach().cpu().numpy().astype(np.float32, copy=False)
        rows.append(
            {
                "sample": int(i),
                "target_semantic": _semantic_target_entries_from_vector(
                    target_cond,
                    class_names=class_names,
                    max_items=max(1, len(class_names)),
                    threshold=0.5,
                ),
                "top_labels": _top_labels_from_probs(p, class_names=class_names, topk=int(topk_all)),
            }
        )
    payload = {
        "cycle": int(cycle_id),
        "round": int(round_id),
        "stage": "G",
        "panel": str(panel_path),
        "samples": rows,
    }
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _render_source_shape_for_stream(stream_len: int, cfg: RenderConfig):
    downsample = max(1, int(cfg.downsample))
    width = max(1, int(cfg.width))
    base_len = max(1, (max(1, int(stream_len)) + downsample - 1) // downsample)
    if int(cfg.max_points) > 0:
        base_len = min(base_len, int(cfg.max_points))
    _, mapping = COLOR_MODE_MAP.get(cfg.color_mode, COLOR_MODE_MAP[COLOR_MODES[0][0]])
    special = bool(mapping.get("__special__") == "single_source_rgb_stride")
    max_len = (base_len + 2) // 3 if special else base_len
    total = int(math.ceil(float(max_len) / float(width)) * width)
    height = max(1, total // width)
    return {
        "downsample": int(downsample),
        "width": int(width),
        "base_len": int(base_len),
        "max_len": int(max_len),
        "total": int(total),
        "height": int(height),
        "mapping": mapping,
        "special": special,
    }


def _sync_render_width_with_embed_hw(cfg: RenderConfig, image_hw: Tuple[int, int]) -> bool:
    target_w = max(1, int(image_hw[1]))
    if int(cfg.width) == int(target_w):
        return False
    cfg.width = int(target_w)
    return True


def _resolve_synced_chunk_samples(
    requested_chunk_samples: int,
    patch_size: int,
    cfg: RenderConfig,
    image_hw: Tuple[int, int],
) -> Tuple[int, Dict[str, Any]]:
    requested = max(1, int(requested_chunk_samples))
    patch = max(1, int(patch_size))
    target_h = max(1, int(image_hw[0]))
    target_w = max(1, int(image_hw[1]))
    downsample = max(1, int(cfg.downsample))
    width = max(1, int(cfg.width))
    _, mapping = COLOR_MODE_MAP.get(cfg.color_mode, COLOR_MODE_MAP[COLOR_MODES[0][0]])
    special = bool(mapping.get("__special__") == "single_source_rgb_stride")

    base_target = int(target_h) * int(width)
    if special:
        base_target *= 3
    if int(cfg.max_points) > 0:
        base_target = min(int(base_target), int(cfg.max_points))
    base_target = max(1, int(base_target))

    low = int((base_target - 1) * downsample + 1)
    high = int(base_target * downsample)
    top_mult = int(high - (high % patch))
    if top_mult >= low:
        resolved = int(top_mult)
    else:
        fallback = int(high // patch) * int(patch)
        if fallback <= 0:
            fallback = int(patch)
        resolved = int(fallback)

    resolved = max(1, int(resolved))
    shape = _render_source_shape_for_stream(stream_len=int(resolved), cfg=cfg)
    needs_resize = (int(shape["height"]) != int(target_h)) or (int(shape["width"]) != int(target_w))
    info = {
        "requested": int(requested),
        "resolved": int(resolved),
        "patch_size": int(patch),
        "target_h": int(target_h),
        "target_w": int(target_w),
        "render_h": int(shape["height"]),
        "render_w": int(shape["width"]),
        "downsample": int(shape["downsample"]),
        "special_stride_mode": bool(special),
        "changed": bool(int(resolved) != int(requested)),
        "needs_resize": bool(needs_resize),
    }
    return int(resolved), info


def _resize_chw_nearest(img_chw: np.ndarray, height: int, width: int) -> np.ndarray:
    t = torch.from_numpy(img_chw.astype(np.float32, copy=False)).unsqueeze(0)
    out = F.interpolate(t, size=(max(1, int(height)), max(1, int(width))), mode="nearest")
    return out.squeeze(0).cpu().numpy().astype(np.float32, copy=False)


def _compose_payload_canvas(
    payload_images: Sequence[np.ndarray],
    payload_targets: Sequence[np.ndarray],
    num_classes: int,
    rng: np.random.Generator,
    target_h: int,
    target_w: int,
    min_payloads: int,
    max_payloads: int,
):
    n_payload = min(int(len(payload_images)), int(len(payload_targets)))
    if int(n_payload) <= 0:
        return None, None, 0
    if int(len(payload_images)) != int(len(payload_targets)):
        raise RuntimeError(
            "Payload canvas requires image/target parity: "
            f"images={int(len(payload_images))} targets={int(len(payload_targets))}"
        )
    lo = max(1, int(min_payloads))
    hi = max(lo, int(max_payloads))
    k = int(rng.integers(lo, hi + 1))
    if k <= 0:
        k = 1

    # Pick payload rows by maximizing new active-label coverage to avoid
    # repeatedly sampling dominant single-label rows from imbalanced banks.
    active_idx_per_row: List[np.ndarray] = []
    expected_dim = max(1, int(num_classes))
    for pidx in range(int(n_payload)):
        t_row = np.asarray(payload_targets[int(pidx)], dtype=np.float32).reshape(-1)
        if int(t_row.size) != int(expected_dim):
            raise RuntimeError(
                "Payload canvas target width mismatch; refusing positional truncation. "
                f"expected={int(expected_dim)} got={int(t_row.size)} row_pick={int(pidx)}"
            )
        active_idx = np.where(np.asarray(t_row, dtype=np.float32) >= 0.5)[0].astype(np.int64)
        active_idx_per_row.append(active_idx)

    replace = bool(int(k) > int(n_payload))
    selected: List[int] = []
    covered_terms: set = set()
    available = [int(i) for i in range(int(n_payload))]
    sample_cap = 192
    for _ in range(int(k)):
        if (not replace) and int(len(available)) <= 0:
            break
        if replace:
            pool = np.arange(int(n_payload), dtype=np.int64)
        else:
            pool = np.asarray(available, dtype=np.int64)
        if int(pool.size) <= 0:
            break
        if int(pool.size) > int(sample_cap):
            cand = rng.choice(pool, size=int(sample_cap), replace=False).astype(np.int64)
        else:
            cand = np.asarray(pool, dtype=np.int64)
        best_rows: List[int] = []
        best_key = (-1, -1)
        for ci in cand.tolist():
            act = active_idx_per_row[int(ci)]
            total_n = int(act.size)
            if int(total_n) <= 0:
                key = (0, 0)
            else:
                new_n = int(sum(1 for t in act.tolist() if int(t) not in covered_terms))
                key = (int(new_n), int(total_n))
            if key > best_key:
                best_key = key
                best_rows = [int(ci)]
            elif key == best_key:
                best_rows.append(int(ci))
        if int(len(best_rows)) <= 0:
            pick = int(cand[int(rng.integers(0, int(cand.size)))])
        else:
            pick = int(best_rows[int(rng.integers(0, int(len(best_rows))))])
        selected.append(int(pick))
        for ti in active_idx_per_row[int(pick)].tolist():
            covered_terms.add(int(ti))
        if not replace:
            available = [int(x) for x in available if int(x) != int(pick)]

    if int(len(selected)) <= 0:
        selected = [int(rng.integers(0, int(n_payload)))]
    idx = np.asarray(selected, dtype=np.int64)
    k_eff = int(idx.size)
    canvas = np.zeros((3, max(1, int(target_h)), max(1, int(target_w))), dtype=np.float32)
    rows = np.linspace(0, int(target_h), num=int(k_eff) + 1, dtype=np.int64)
    target_vec = np.zeros((max(1, int(num_classes)),), dtype=np.float32)
    used = 0
    for i, pidx in enumerate(idx.tolist()):
        r0 = int(rows[i])
        r1 = int(rows[i + 1])
        if r1 <= r0:
            continue
        patch = _resize_chw_nearest(payload_images[int(pidx)], height=(r1 - r0), width=int(target_w))
        canvas[:, r0:r1, :] = patch
        t = np.asarray(payload_targets[int(pidx)], dtype=np.float32).reshape(-1)
        if int(t.size) != int(target_vec.size):
            raise RuntimeError(
                "Payload canvas target width mismatch; refusing positional truncation. "
                f"expected={int(target_vec.size)} got={int(t.size)} row_pick={int(pidx)}"
            )
        target_vec = np.maximum(
            np.asarray(target_vec, dtype=np.float32),
            np.asarray(t, dtype=np.float32),
        ).astype(np.float32, copy=False)
        used += 1
    return canvas, target_vec.astype(np.float32, copy=False), int(used)


def _payload_canvas_to_base_sequence(canvas_chw: np.ndarray, shape_info: Dict, empty_fill: float):
    mapping = shape_info["mapping"]
    special = bool(shape_info["special"])
    base_len = int(shape_info["base_len"])
    max_len = int(shape_info["max_len"])
    total = int(shape_info["total"])

    if not special:
        streams = []
        if mapping.get("R") is not None:
            streams.append(canvas_chw[0].reshape(-1))
        if mapping.get("G") is not None:
            streams.append(canvas_chw[1].reshape(-1))
        if mapping.get("B") is not None:
            streams.append(canvas_chw[2].reshape(-1))
        if len(streams) == 0:
            flat = np.full((int(total),), float(empty_fill), dtype=np.float32)
        elif len(streams) == 1:
            flat = streams[0].astype(np.float32, copy=False)
        else:
            flat = np.mean(np.stack(streams, axis=0).astype(np.float32, copy=False), axis=0)
        if flat.size < max_len:
            pad = np.full((max_len - flat.size,), float(empty_fill), dtype=np.float32)
            flat = np.concatenate([flat, pad], axis=0)
        return flat[:base_len].astype(np.float32, copy=False)

    r = canvas_chw[0].reshape(-1).astype(np.float32, copy=False)
    g = canvas_chw[1].reshape(-1).astype(np.float32, copy=False)
    b = canvas_chw[2].reshape(-1).astype(np.float32, copy=False)
    if r.size < max_len:
        r = np.concatenate([r, np.full((max_len - r.size,), float(empty_fill), dtype=np.float32)], axis=0)
    if g.size < max_len:
        g = np.concatenate([g, np.full((max_len - g.size,), float(empty_fill), dtype=np.float32)], axis=0)
    if b.size < max_len:
        b = np.concatenate([b, np.full((max_len - b.size,), float(empty_fill), dtype=np.float32)], axis=0)
    idx = np.arange(base_len, dtype=np.int64)
    src_idx = idx // 3
    mod = idx % 3
    out = np.where(mod == 0, r[src_idx], np.where(mod == 1, g[src_idx], b[src_idx]))
    return out.astype(np.float32, copy=False)


def _embed_base_sequence_into_stream(
    stream_f32: np.ndarray,
    base_seq_01: np.ndarray,
    cfg: RenderConfig,
    sample_bits: int,
):
    x = np.asarray(stream_f32, dtype=np.float32).copy()
    shape = _render_source_shape_for_stream(stream_len=int(x.size), cfg=cfg)
    base_len = int(shape["base_len"])
    downsample = int(shape["downsample"])
    n_use = min(int(base_len), int(base_seq_01.size))
    if n_use <= 0:
        return x

    idx = (np.arange(n_use, dtype=np.int64) * downsample).astype(np.int64)
    idx = idx[idx < int(x.size)]
    n_use = int(idx.size)
    if n_use <= 0:
        return x

    base_vals = np.clip(base_seq_01[:n_use], 0.0, 1.0).astype(np.float32, copy=False)
    if bool(cfg.bitmask_enable):
        lo, hi, depth = normalize_bit_window(int(cfg.bitmask_low), int(cfg.bitmask_high), int(sample_bits))
        full_mask = (1 << int(sample_bits)) - 1
        window_mask = (1 << int(depth)) - 1
        q_min = -(1 << (int(sample_bits) - 1))
        q_max = (1 << (int(sample_bits) - 1)) - 1
        q_scale = float(1 << (int(sample_bits) - 1))

        current = np.clip(x[idx], -1.0, 1.0).astype(np.float32, copy=False)
        q_signed = np.round(current * q_scale).astype(np.int64)
        q_signed = np.clip(q_signed, q_min, q_max).astype(np.int64)

        desired = np.round(base_vals * float(window_mask)).astype(np.int64)
        desired = np.clip(desired, 0, window_mask).astype(np.int64)

        unsigned = np.bitwise_and(q_signed, full_mask)
        clear_mask = np.int64(~(window_mask << int(lo)))
        unsigned = np.bitwise_and(unsigned, clear_mask)
        unsigned = np.bitwise_or(unsigned, np.left_shift(desired, int(lo)))
        sign_cut = (1 << (int(sample_bits) - 1))
        signed = np.where(unsigned >= sign_cut, unsigned - (1 << int(sample_bits)), unsigned)
        x[idx] = np.clip(signed.astype(np.float32) / q_scale, -1.0, 1.0)
        return x.astype(np.float32, copy=False)

    x[idx] = (base_vals * 2.0) - 1.0
    x[idx] = np.clip(x[idx], -1.0, 1.0)
    return x.astype(np.float32, copy=False)


def _build_berkeley_payload_bank(
    data_root: str,
    image_size: int,
    auto_install_scipy: bool,
    max_samples: int,
    seed: int,
    source_root: str = "",
    force_cache_rebuild: bool = False,
):
    _ = bool(auto_install_scipy)  # Intentionally unused in this path; payload rebuild is SciPy-free.

    def _norm_txt(x: str) -> str:
        return re.sub(r"\s+", " ", str(x)).strip().lower()

    def _is_image_file(path: Path) -> bool:
        return str(path.suffix).strip().lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

    def _to_rgb_chw01(path: Path, size: int) -> np.ndarray:
        from PIL import Image

        with Image.open(str(path)) as im:
            rgb = im.convert("RGB")
            if int(rgb.size[0]) != int(size) or int(rgb.size[1]) != int(size):
                rgb = rgb.resize((int(size), int(size)), resample=Image.BILINEAR)
            arr = np.asarray(rgb, dtype=np.float32)
        if int(arr.ndim) != 3 or int(arr.shape[2]) != 3:
            raise RuntimeError(f"Unsupported RGB payload source image shape: {tuple(arr.shape)} ({path})")
        arr = np.clip(arr / 255.0, 0.0, 1.0).astype(np.float32, copy=False)
        return np.transpose(arr, (2, 0, 1)).astype(np.float32, copy=False)

    def _source_signature(root: Path) -> Dict[str, Any]:
        out: Dict[str, Any] = {"root": str(root), "exists": bool(root.exists()), "datasets": []}
        if not root.exists():
            return out
        ds_dirs = [p for p in root.iterdir() if p.is_dir()]
        ds_dirs = sorted(ds_dirs, key=lambda p: p.name.lower())
        for ds_dir in ds_dirs:
            files = [p for p in ds_dir.rglob("*") if p.is_file() and _is_image_file(p)]
            if len(files) <= 0:
                continue
            labels = set()
            latest = 0.0
            for fp in files:
                try:
                    latest = max(float(latest), float(fp.stat().st_mtime))
                except Exception:
                    pass
                try:
                    rel = fp.relative_to(ds_dir)
                    if len(rel.parts) > 1:
                        lbl = re.sub(r"[_\-]+", " ", str(rel.parts[0])).strip()
                        if str(lbl):
                            labels.add(str(lbl))
                except Exception:
                    pass
            out["datasets"].append(
                {
                    "name": re.sub(r"[_\-]+", " ", str(ds_dir.name)).strip(),
                    "files": int(len(files)),
                    "latest_mtime": int(latest),
                    "labels": sorted([str(x) for x in labels], key=lambda s: s.lower()),
                }
            )
        return out

    root = Path(str(data_root).strip() or "toys_to_survive_development/data/berkeley_sbd")
    size = max(8, int(image_size))
    class_names = _default_berkeley_class_names()
    n_classes = max(1, int(len(class_names)))
    class_lut = {_norm_txt(name): int(i) for i, name in enumerate(class_names)}
    ext_root = Path(str(source_root).strip()) if str(source_root).strip() else (root / "payload_sources")
    ext_sig = _source_signature(ext_root)

    cache_dir = root / "cache" / f"payload_bank_rgb{int(size)}_v2"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    images_path = cache_dir / "images.npy"
    labels_path = cache_dir / "labels.npy"
    terms_path = cache_dir / "terms.json"
    sources_path = cache_dir / "sources.json"
    source_catalog_path = cache_dir / "source_catalog.json"
    cache_version = 2

    cache_loaded = False
    images_np: Optional[np.ndarray] = None
    labels_np: Optional[np.ndarray] = None
    terms_rows: List[List[str]] = []
    source_rows: List[str] = []
    manifest: Dict[str, Any] = {}

    if not bool(force_cache_rebuild):
        try:
            if all(p.exists() for p in [manifest_path, images_path, labels_path, terms_path, sources_path]):
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                valid = True
                valid = valid and int(manifest.get("version", -1)) == int(cache_version)
                valid = valid and int(manifest.get("image_size", -1)) == int(size)
                valid = valid and int(manifest.get("num_classes", -1)) == int(n_classes)
                valid = valid and list(manifest.get("class_names", [])) == list(class_names)
                valid = valid and manifest.get("external_source_signature", {}) == ext_sig
                if valid:
                    images_np = np.load(str(images_path), mmap_mode="r")
                    labels_np = np.load(str(labels_path), mmap_mode="r")
                    terms_rows = json.loads(terms_path.read_text(encoding="utf-8"))
                    source_rows = json.loads(sources_path.read_text(encoding="utf-8"))
                    n_rows = int(images_np.shape[0]) if isinstance(images_np, np.ndarray) else 0
                    valid = valid and int(images_np.ndim) == 4 and int(images_np.shape[1]) == 3
                    valid = valid and int(images_np.shape[2]) == int(size) and int(images_np.shape[3]) == int(size)
                    valid = valid and int(labels_np.ndim) == 2 and int(labels_np.shape[1]) == int(n_classes)
                    valid = valid and int(labels_np.shape[0]) == int(n_rows)
                    valid = valid and int(len(terms_rows)) == int(n_rows)
                    valid = valid and int(len(source_rows)) == int(n_rows)
                    if bool(valid):
                        cache_loaded = True
        except Exception:
            cache_loaded = False

    if not bool(cache_loaded):
        _log(
            "Berkeley payload bank cache missing/stale; rebuilding from Berkeley SBD source files "
            f"into '{cache_dir}'."
        )
        row_specs: List[Tuple[str, np.ndarray, List[str], str]] = []
        terms_rows = []
        source_rows = []
        source_counts: Dict[str, int] = {}
        img_root = root / "img"
        split_specs = [
            ("train", root / "train.txt", root / "cache" / "sbd_train_multilabel.npz"),
            ("val", root / "val.txt", root / "cache" / "sbd_val_multilabel.npz"),
        ]
        loaded_split_counts: Dict[str, int] = {"train": 0, "val": 0}
        missing_images = 0
        decode_failures = 0
        for split_name, list_path, label_path in split_specs:
            if not list_path.exists():
                raise RuntimeError(f"Berkeley split list is missing for payload rebuild: {list_path}")
            if not label_path.exists():
                raise RuntimeError(
                    "Berkeley multilabel cache is missing for payload rebuild: "
                    f"{label_path}. Re-run berkeley_sbd_pretrain.py once to populate label caches."
                )
            ids = [str(x).strip() for x in list_path.read_text(encoding="utf-8").splitlines() if str(x).strip()]
            try:
                with np.load(str(label_path)) as z:
                    if "labels" not in z.files:
                        raise RuntimeError(f"'labels' key missing in {label_path}")
                    labels_split = np.asarray(z["labels"], dtype=np.float32)
            except Exception as e:
                raise RuntimeError(f"Failed reading Berkeley label cache {label_path}: {type(e).__name__}: {e}") from e
            if int(labels_split.ndim) != 2:
                raise RuntimeError(
                    "Berkeley label cache must be 2D [N,C]: "
                    f"path={label_path} shape={tuple(labels_split.shape)}"
                )
            if int(labels_split.shape[0]) != int(len(ids)):
                raise RuntimeError(
                    "Berkeley split list/label cache row mismatch during payload rebuild: "
                    f"split={split_name} list_rows={int(len(ids))} label_rows={int(labels_split.shape[0])}"
                )
            source_key = f"berkeley_sbd_{split_name}"
            for local_idx, stem in enumerate(ids):
                img_path = None
                for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"):
                    cand = img_root / f"{str(stem)}{str(ext)}"
                    if cand.exists():
                        img_path = cand
                        break
                if img_path is None:
                    missing_images += 1
                    continue
                try:
                    # Descriptor pass validates decode without keeping image tensors in RAM.
                    _ = _to_rgb_chw01(path=img_path, size=int(size))
                except Exception:
                    decode_failures += 1
                    continue
                yv = np.asarray(labels_split[int(local_idx)], dtype=np.float32).reshape(-1)
                if int(yv.size) >= int(n_classes):
                    yv = yv[: int(n_classes)]
                else:
                    pad = np.zeros((int(n_classes) - int(yv.size),), dtype=np.float32)
                    yv = np.concatenate([yv, pad], axis=0)
                pos = np.where(np.asarray(yv, dtype=np.float32) > 0.5)[0].astype(np.int64).tolist()
                label_terms = [str(class_names[int(i)]) for i in pos if 0 <= int(i) < int(len(class_names))]
                terms = _normalize_vocab_terms(
                    ["berkeley sbd dataset", "object", "signal"] + list(label_terms)
                )
                row_specs.append(
                    (
                        str(img_path),
                        np.asarray(yv, dtype=np.float32).reshape(-1),
                        list(terms),
                        str(source_key),
                    )
                )
                source_counts[str(source_key)] = int(source_counts.get(str(source_key), 0)) + 1
                loaded_split_counts[str(split_name)] = int(loaded_split_counts.get(str(split_name), 0)) + 1

        n_train = int(loaded_split_counts.get("train", 0))
        n_val = int(loaded_split_counts.get("val", 0))
        if int(n_train) <= 0 or int(n_val) <= 0:
            raise RuntimeError(
                "Berkeley payload rebuild produced empty core splits. "
                f"loaded_train={int(n_train)} loaded_val={int(n_val)} "
                f"missing_images={int(missing_images)} decode_failures={int(decode_failures)}"
            )

        if ext_root.exists():
            ds_dirs = [p for p in ext_root.iterdir() if p.is_dir()]
            ds_dirs = sorted(ds_dirs, key=lambda p: p.name.lower())
            for ds_dir in ds_dirs:
                dataset_name = re.sub(r"[_\-]+", " ", str(ds_dir.name)).strip()
                if not str(dataset_name):
                    continue
                files = [p for p in ds_dir.rglob("*") if p.is_file() and _is_image_file(p)]
                files = sorted(files, key=lambda p: str(p).lower())
                for fp in files:
                    try:
                        # Descriptor pass validates decode without keeping image tensors in RAM.
                        _ = _to_rgb_chw01(path=fp, size=int(size))
                        vec = np.zeros((int(n_classes),), dtype=np.float32)
                        rel = fp.relative_to(ds_dir)
                        label_term = ""
                        if len(rel.parts) > 1:
                            label_term = re.sub(r"[_\-]+", " ", str(rel.parts[0])).strip()
                        ref_terms = _normalize_vocab_terms(
                            [str(dataset_name), f"{dataset_name} dataset", "signal"] + ([str(label_term)] if str(label_term) else [])
                        )
                        for cand in [str(dataset_name), f"{dataset_name} dataset", str(label_term)]:
                            key = _norm_txt(cand)
                            if key in class_lut:
                                vec[int(class_lut[key])] = 1.0
                        row_specs.append(
                            (
                                str(fp),
                                np.asarray(vec, dtype=np.float32).reshape(-1),
                                list(ref_terms),
                                str(dataset_name),
                            )
                        )
                        source_counts[str(dataset_name)] = int(source_counts.get(str(dataset_name), 0)) + 1
                    except Exception:
                        decode_failures += 1
                        continue

        if len(row_specs) <= 0:
            return [], [], {"available": 0, "used": 0, "available_train": 0, "available_val": 0}, []

        rows_total = int(len(row_specs))
        images_tmp_path = cache_dir / "images.tmp.npy"
        labels_tmp_path = cache_dir / "labels.tmp.npy"
        for tmp_path in [images_tmp_path, labels_tmp_path]:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
        images_reused = False
        existing_images_mm: Optional[np.ndarray] = None
        if images_path.exists():
            try:
                existing_images_mm = np.load(str(images_path), mmap_mode="r")
                if (
                    int(getattr(existing_images_mm, "ndim", 0)) == 4
                    and int(existing_images_mm.shape[0]) == int(rows_total)
                    and int(existing_images_mm.shape[1]) == 3
                    and int(existing_images_mm.shape[2]) == int(size)
                    and int(existing_images_mm.shape[3]) == int(size)
                ):
                    images_reused = True
            except Exception:
                existing_images_mm = None
                images_reused = False

        terms_rows = []
        source_rows = []
        labels_mm = np.lib.format.open_memmap(
            str(labels_tmp_path),
            mode="w+",
            dtype=np.float32,
            shape=(int(rows_total), int(n_classes)),
        )
        if bool(images_reused):
            for wi, row in enumerate(row_specs):
                _, yv_row, terms_row, src_row = row
                labels_mm[int(wi), :] = np.asarray(yv_row, dtype=np.float32).reshape(-1)[: int(n_classes)]
                terms_rows.append(list(terms_row))
                source_rows.append(str(src_row))
                if ((int(wi) + 1) % 1024) == 0:
                    labels_mm.flush()
            labels_mm.flush()
            del labels_mm
            if labels_path.exists():
                labels_path.unlink()
            labels_tmp_path.replace(labels_path)
            images_np = existing_images_mm if existing_images_mm is not None else np.load(str(images_path), mmap_mode="r")
            labels_np = np.load(str(labels_path), mmap_mode="r")
        else:
            images_mm = np.lib.format.open_memmap(
                str(images_tmp_path),
                mode="w+",
                dtype=np.float16,
                shape=(int(rows_total), 3, int(size), int(size)),
            )
            for wi, row in enumerate(row_specs):
                img_path_txt, yv_row, terms_row, src_row = row
                x01 = _to_rgb_chw01(path=Path(str(img_path_txt)), size=int(size))
                images_mm[int(wi), :, :, :] = np.asarray(x01, dtype=np.float16, copy=False)
                labels_mm[int(wi), :] = np.asarray(yv_row, dtype=np.float32).reshape(-1)[: int(n_classes)]
                terms_rows.append(list(terms_row))
                source_rows.append(str(src_row))
                if ((int(wi) + 1) % 512) == 0:
                    images_mm.flush()
                    labels_mm.flush()
            images_mm.flush()
            labels_mm.flush()
            del images_mm
            del labels_mm
            if images_path.exists():
                images_path.unlink()
            if labels_path.exists():
                labels_path.unlink()
            images_tmp_path.replace(images_path)
            labels_tmp_path.replace(labels_path)
            images_np = np.load(str(images_path), mmap_mode="r")
            labels_np = np.load(str(labels_path), mmap_mode="r")

        source_catalog = {
            "generated_at": float(time.time()),
            "external_source_root": str(ext_root),
            "external_source_signature": ext_sig,
            "source_counts": {str(k): int(v) for k, v in source_counts.items()},
            "available_rows": int(rows_total),
            "available_train": int(n_train),
            "available_val": int(n_val),
            "missing_images": int(missing_images),
            "decode_failures": int(decode_failures),
            "images_reused": bool(images_reused),
        }
        manifest = {
            "version": int(cache_version),
            "image_size": int(size),
            "num_classes": int(n_classes),
            "class_names": list(class_names),
            "available_rows": int(rows_total),
            "available_train": int(n_train),
            "available_val": int(n_val),
            "available_external": int(max(0, int(rows_total) - int(n_train) - int(n_val))),
            "external_source_root": str(ext_root),
            "external_source_signature": ext_sig,
            "source_counts": {str(k): int(v) for k, v in source_counts.items()},
            "source_catalog": str(source_catalog_path),
            "generated_at": float(time.time()),
        }
        terms_path.write_text(json.dumps(terms_rows, ensure_ascii=True), encoding="utf-8")
        sources_path.write_text(json.dumps(source_rows, ensure_ascii=True), encoding="utf-8")
        source_catalog_path.write_text(json.dumps(source_catalog, indent=2), encoding="utf-8")
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    else:
        if source_catalog_path.exists():
            try:
                manifest = dict(manifest)
                manifest["source_catalog"] = str(source_catalog_path)
            except Exception:
                pass

    if images_np is None or labels_np is None:
        return [], [], {"available": 0, "used": 0, "available_train": 0, "available_val": 0}, []

    n_total = int(images_np.shape[0])
    if int(len(source_rows)) < int(n_total):
        source_rows = list(source_rows) + (["unknown_source"] * int(max(0, int(n_total) - int(len(source_rows)))))
    elif int(len(source_rows)) > int(n_total):
        source_rows = list(source_rows[: int(n_total)])

    def _source_supervised_stats(source_key: str) -> Dict[str, float]:
        key = re.sub(r"\s+", " ", str(source_key)).strip().lower()
        idx_rows = [
            int(i)
            for i in range(int(n_total))
            if re.sub(r"\s+", " ", str(source_rows[int(i)])).strip().lower() == key
        ]
        if len(idx_rows) <= 0:
            return {"rows": 0, "nonzero": 0, "min": 0, "mean": 0.0, "max": 0}
        idx_np = np.asarray(idx_rows, dtype=np.int64)
        y_np = np.asarray(labels_np[idx_np], dtype=np.float32)
        if int(y_np.ndim) == 1:
            y_np = y_np.reshape(1, -1)
        sup_take = min(int(n_classes), int(y_np.shape[1]))
        if int(sup_take) <= 0:
            return {"rows": int(y_np.shape[0]), "nonzero": 0, "min": 0, "mean": 0.0, "max": 0}
        pos_counts = np.count_nonzero(np.asarray(y_np[:, : int(sup_take)], dtype=np.float32) >= 0.5, axis=1).astype(np.int32)
        return {
            "rows": int(pos_counts.size),
            "nonzero": int(np.count_nonzero(pos_counts > 0)),
            "min": int(pos_counts.min()) if int(pos_counts.size) > 0 else 0,
            "mean": float(np.mean(pos_counts)) if int(pos_counts.size) > 0 else 0.0,
            "max": int(pos_counts.max()) if int(pos_counts.size) > 0 else 0,
        }

    train_lbl_stats = _source_supervised_stats("berkeley_sbd_train")
    val_lbl_stats = _source_supervised_stats("berkeley_sbd_val")
    if int(train_lbl_stats.get("rows", 0)) > 0 and int(train_lbl_stats.get("nonzero", 0)) < int(train_lbl_stats.get("rows", 0)):
        raise RuntimeError(
            "Berkeley payload cache train rows are missing supervised targets. "
            f"nonzero={int(train_lbl_stats.get('nonzero', 0))}/{int(train_lbl_stats.get('rows', 0))}"
        )
    if int(val_lbl_stats.get("rows", 0)) > 0 and int(val_lbl_stats.get("nonzero", 0)) < int(val_lbl_stats.get("rows", 0)):
        raise RuntimeError(
            "Berkeley payload cache val rows are missing supervised targets. "
            f"nonzero={int(val_lbl_stats.get('nonzero', 0))}/{int(val_lbl_stats.get('rows', 0))}"
        )
    _log(
        "[berkeley-label-sanity] "
        "source=payload_cache "
        f"train={int(train_lbl_stats.get('nonzero', 0))}/{int(train_lbl_stats.get('rows', 0))} "
        f"train_min/mean/max={int(train_lbl_stats.get('min', 0))}/"
        f"{float(train_lbl_stats.get('mean', 0.0)):.2f}/{int(train_lbl_stats.get('max', 0))} "
        f"val={int(val_lbl_stats.get('nonzero', 0))}/{int(val_lbl_stats.get('rows', 0))} "
        f"val_min/mean/max={int(val_lbl_stats.get('min', 0))}/"
        f"{float(val_lbl_stats.get('mean', 0.0)):.2f}/{int(val_lbl_stats.get('max', 0))}"
    )

    take = int(n_total) if int(max_samples) <= 0 else min(int(n_total), int(max_samples))
    rng = np.random.default_rng(seed)
    if int(take) >= int(n_total):
        picks = np.arange(int(n_total), dtype=np.int64)
        rng.shuffle(picks)
    else:
        picks = rng.choice(np.arange(int(n_total), dtype=np.int64), size=int(take), replace=False).astype(np.int64)

    out_images: List[np.ndarray] = []
    out_targets: List[np.ndarray] = []
    out_terms: List[List[str]] = []
    used_train = 0
    used_val = 0
    used_external = 0
    used_source_counts: Dict[str, int] = {}
    for idx in picks.tolist():
        img = np.asarray(images_np[int(idx)], dtype=np.float32)
        yv = np.asarray(labels_np[int(idx)], dtype=np.float32).reshape(-1)
        src = str(source_rows[int(idx)]) if int(idx) < int(len(source_rows)) else ""
        terms = (
            list(_normalize_vocab_terms(terms_rows[int(idx)]))
            if int(idx) < int(len(terms_rows)) and isinstance(terms_rows[int(idx)], list)
            else []
        )
        if len(terms) <= 0:
            raise RuntimeError(
                "Payload bank cache row is missing original terms; strict semantic pipeline requires "
                f"non-empty source terms. row_index={int(idx)} source={str(src)!r}"
            )
        out_images.append(np.clip(img, 0.0, 1.0).astype(np.float32, copy=False))
        out_targets.append(np.clip(yv, 0.0, 1.0).astype(np.float32, copy=False))
        out_terms.append(list(terms))
        used_source_counts[str(src)] = int(used_source_counts.get(str(src), 0)) + 1
        if str(src).lower() == "berkeley_sbd_train":
            used_train += 1
        elif str(src).lower() == "berkeley_sbd_val":
            used_val += 1
        else:
            used_external += 1

    info = {
        "available": int(n_total),
        "used": int(len(out_images)),
        "available_train": int(manifest.get("available_train", 0)),
        "available_val": int(manifest.get("available_val", 0)),
        "available_external": int(manifest.get("available_external", 0)),
        "used_train": int(used_train),
        "used_val": int(used_val),
        "used_external": int(used_external),
        "cache_dir": str(cache_dir),
        "cache_manifest": str(manifest_path),
        "cache_hit": bool(cache_loaded),
        "source_catalog": str(manifest.get("source_catalog", str(source_catalog_path))),
        "used_source_counts": {str(k): int(v) for k, v in used_source_counts.items()},
    }
    return out_images, out_targets, info, out_terms


def _imprint_latent_stream_targets(
    streams: Sequence[np.ndarray],
    metas: Sequence[Dict],
    payload_images: Sequence[np.ndarray],
    payload_targets: Sequence[np.ndarray],
    num_classes: int,
    chunk_samples: int,
    classifier: nn.Module,
    cfg: RenderConfig,
    sample_bits: int,
    image_hw: Tuple[int, int],
    device: torch.device,
    seed: int,
    steps: int,
    lr: float,
    l2_weight: float,
    min_target_classes: int,
    max_target_classes: int,
    kind_target_vectors: Optional[Dict[str, Any]] = None,
    semantic_term_to_idx: Optional[Dict[str, int]] = None,
    amp_enabled: bool = False,
    amp_dtype: str = "float16",
    channels_last: bool = False,
):
    if len(streams) != len(metas):
        raise RuntimeError(
            f"Latent imprint expects stream/meta parity, got streams={len(streams)} metas={len(metas)}"
        )

    rng = np.random.default_rng(seed)
    out_streams: List[np.ndarray] = []
    out_targets: List[Optional[np.ndarray]] = []
    mix_total = 0
    noise_total = 0
    imprinted = 0
    probs: List[float] = []
    target_class_count: List[int] = []
    payload_row_count: List[int] = []
    empty_fill = float(int(cfg.empty_fill) & 0xFF) / 255.0
    kind_vec_map: Dict[str, np.ndarray] = {}
    if isinstance(kind_target_vectors, dict):
        for key, row in kind_target_vectors.items():
            k = re.sub(r"\s+", " ", str(key)).strip().lower()
            if not k:
                continue
            arr = np.asarray(row, dtype=np.float32).reshape(-1)
            if int(arr.size) <= 0:
                continue
            if int(arr.size) > int(num_classes):
                raise RuntimeError(
                    "Latent kind target width exceeds configured class width; refusing truncation. "
                    f"kind={str(key)} row_dim={int(arr.size)} num_classes={int(num_classes)}"
                )
            if int(arr.size) >= int(num_classes):
                arr = arr[: int(num_classes)]
            else:
                pad = np.zeros((int(num_classes) - int(arr.size),), dtype=np.float32)
                arr = np.concatenate([arr, pad], axis=0)
            kind_vec_map[k] = np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)

    def _kind_target(kind_key: str) -> Optional[np.ndarray]:
        k = re.sub(r"\s+", " ", str(kind_key)).strip().lower()
        if k in kind_vec_map:
            return np.array(kind_vec_map[k], dtype=np.float32, copy=True)
        if "none" in kind_vec_map:
            return np.array(kind_vec_map["none"], dtype=np.float32, copy=True)
        return None

    def _terms_target_vector(terms_raw: Any) -> Optional[np.ndarray]:
        if not isinstance(semantic_term_to_idx, dict) or int(len(semantic_term_to_idx)) <= 0:
            return None
        if not isinstance(terms_raw, (list, tuple, set)):
            return None
        terms = _semantic_expand_inferred_tags([str(x) for x in terms_raw])
        if len(terms) <= 0:
            return None
        out = np.zeros((max(1, int(num_classes)),), dtype=np.float32)
        hit = 0
        for term in terms:
            key = re.sub(r"\s+", " ", str(term)).strip().lower()
            if not key:
                continue
            idx = int(semantic_term_to_idx.get(key, -1))
            if 0 <= int(idx) < int(out.size):
                out[int(idx)] = 1.0
                hit += 1
        if int(hit) <= 0:
            return None
        return np.asarray(out, dtype=np.float32)

    for s, m in zip(streams, metas):
        kind = _latent_pool_stream_kind(m.get("path", ""))
        meta_vec = _terms_target_vector(m.get("semantic_terms", []))
        if kind == "noise":
            noise_total += 1
            out_streams.append(np.asarray(s, dtype=np.float32))
            base_vec = _kind_target("noise")
            if base_vec is None:
                out_targets.append(meta_vec)
            elif meta_vec is None:
                out_targets.append(base_vec)
            else:
                out_targets.append(np.maximum(base_vec, meta_vec).astype(np.float32, copy=False))
            continue
        if kind != "mix":
            out_streams.append(np.asarray(s, dtype=np.float32))
            base_vec = _kind_target("none")
            if base_vec is None:
                out_targets.append(meta_vec)
            elif meta_vec is None:
                out_targets.append(base_vec)
            else:
                out_targets.append(np.maximum(base_vec, meta_vec).astype(np.float32, copy=False))
            continue

        mix_total += 1
        if int(num_classes) <= 0 or len(payload_images) <= 0:
            out_streams.append(np.asarray(s, dtype=np.float32))
            base_vec = _kind_target("mix")
            if base_vec is None:
                out_targets.append(meta_vec)
            elif meta_vec is None:
                out_targets.append(base_vec)
            else:
                out_targets.append(np.maximum(base_vec, meta_vec).astype(np.float32, copy=False))
            continue

        shape = _render_source_shape_for_stream(stream_len=int(max(1, s.size)), cfg=cfg)
        canvas, target_vec, n_target = _compose_payload_canvas(
            payload_images=payload_images,
            payload_targets=payload_targets,
            num_classes=int(num_classes),
            rng=rng,
            target_h=int(shape["height"]),
            target_w=int(shape["width"]),
            min_payloads=int(min_target_classes),
            max_payloads=int(max_target_classes),
        )
        if canvas is None or target_vec is None or float(np.sum(target_vec)) <= 0.0:
            raise RuntimeError("Latent mix-target imprint failed to build a valid payload canvas.")
        mix_vec = _kind_target("mix")
        if mix_vec is not None and int(np.asarray(mix_vec).size) == int(np.asarray(target_vec).size):
            target_vec = np.maximum(
                np.asarray(target_vec, dtype=np.float32).reshape(-1),
                np.asarray(mix_vec, dtype=np.float32).reshape(-1),
            ).astype(np.float32, copy=False)
        if meta_vec is not None and int(np.asarray(meta_vec).size) == int(np.asarray(target_vec).size):
            target_vec = np.maximum(
                np.asarray(target_vec, dtype=np.float32).reshape(-1),
                np.asarray(meta_vec, dtype=np.float32).reshape(-1),
            ).astype(np.float32, copy=False)
        base_seq = _payload_canvas_to_base_sequence(
            canvas_chw=canvas,
            shape_info=shape,
            empty_fill=float(empty_fill),
        )
        out_stream = _embed_base_sequence_into_stream(
            stream_f32=np.asarray(s, dtype=np.float32),
            base_seq_01=base_seq,
            cfg=cfg,
            sample_bits=int(sample_bits),
        )
        out_streams.append(np.clip(out_stream, -1.0, 1.0).astype(np.float32, copy=False))
        out_targets.append(np.asarray(target_vec, dtype=np.float32).reshape(-1))
        imprinted += 1
        with torch.no_grad():
            xb = torch.from_numpy(out_streams[-1][None, :]).to(device)
            amp_dtype_t = _resolve_amp_dtype(amp_dtype) if amp_enabled else torch.float16
            with _autocast_context(device=device, enabled=amp_enabled, amp_dtype_t=amp_dtype_t):
                img = render_mono_wave_to_tensor(xb, cfg=cfg, image_hw=image_hw, sample_bits=int(sample_bits))
                if channels_last:
                    img = img.contiguous(memory_format=torch.channels_last)
                logits = classifier(img)
            tgt = np.where(np.asarray(target_vec, dtype=np.float32).reshape(-1) > 0.5)[0].astype(np.int64).tolist()
            if len(tgt) <= 0:
                tgt = np.where(np.asarray(target_vec, dtype=np.float32).reshape(-1) > 0.0)[0].astype(np.int64).tolist()
            if len(tgt) > 0:
                probs.append(float(torch.sigmoid(logits[:, tgt]).mean().item()))
        payload_row_count.append(int(n_target))
        active_n = int(
            np.count_nonzero(np.asarray(target_vec, dtype=np.float32).reshape(-1) >= 0.5)
        )
        if int(active_n) <= 0:
            active_n = int(
                np.count_nonzero(np.asarray(target_vec, dtype=np.float32).reshape(-1) > 0.0)
            )
        target_class_count.append(int(active_n))

    info = {
        "streams": int(len(streams)),
        "mix_streams": int(mix_total),
        "noise_streams": int(noise_total),
        "imprinted_streams": int(imprinted),
        "target_supervised_streams": int(
            sum(
                1
                for t in out_targets
                if (t is not None) and (float(np.asarray(t, dtype=np.float32).reshape(-1).sum()) > 0.0)
            )
        ),
        "target_classes_min": (int(np.min(target_class_count)) if len(target_class_count) > 0 else 0),
        "mean_target_classes_per_stream": (float(np.mean(target_class_count)) if len(target_class_count) > 0 else 0.0),
        "target_classes_max": (int(np.max(target_class_count)) if len(target_class_count) > 0 else 0),
        "payload_rows_min": (int(np.min(payload_row_count)) if len(payload_row_count) > 0 else 0),
        "mean_payload_rows_per_stream": (float(np.mean(payload_row_count)) if len(payload_row_count) > 0 else 0.0),
        "payload_rows_max": (int(np.max(payload_row_count)) if len(payload_row_count) > 0 else 0),
        "mean_target_prob_after_imprint": (float(np.mean(probs)) if len(probs) > 0 else 0.0),
    }
    return out_streams, out_targets, info


def _build_enforced_render_config(args) -> RenderConfig:
    color_mode = str(args.enforce_render_color_mode)
    if color_mode not in COLOR_MODE_MAP:
        color_mode = COLOR_MODES[0][0]
    return RenderConfig(
        bitmode=str(args.enforce_render_bitmode),
        use_channels=str(args.enforce_render_use_channels),
        mono_pick=str(args.enforce_render_mono_pick),
        width=max(1, int(args.enforce_render_width)),
        downsample=max(1, int(args.enforce_render_downsample)),
        color_mode=color_mode,
        empty_fill=max(0, min(255, int(args.enforce_render_empty_fill))),
        bitmask_enable=bool(args.enforce_render_bitmask_enable),
        bitmask_low=int(args.enforce_render_bitmask_low),
        bitmask_high=int(args.enforce_render_bitmask_high),
        max_points=max(1024, int(args.enforce_render_max_points)),
    )


def parse_args():
    p = argparse.ArgumentParser(description="Config-search + classifier + waveform transformer training pipeline.")
    p.add_argument(
        "--wav-root",
        default="",
        help="Directory to recursively scan for .wav files. If omitted, latent fallback pool is generated.",
    )
    p.add_argument("--output-dir", default="toys_to_survive_development/wav_pipeline_runs/latest")
    p.add_argument(
        "--hard-wipe-caches",
        "--hard-wipe-cache",
        dest="hard_wipe_caches",
        action="store_true",
        help=(
            "Delete generated wave libraries and Berkeley payload-bank caches before startup "
            "(accepted_wave_library, latent_wave_pool, training_supervision, cache/payload_bank_rgb*)."
        ),
    )
    p.add_argument(
        "--soft-reset-labels",
        "--soft-reset-label-cache",
        dest="soft_reset_labels",
        action="store_true",
        help=(
            "Reset label/semantic cache artifacts while keeping cached images "
            "(payload labels/terms/sources/manifest/source_catalog and semantic_gate_labels)."
        ),
    )
    p.add_argument("--max-files", type=int, default=300)
    p.add_argument("--latent-fallback-count", type=int, default=192)
    p.add_argument("--latent-fallback-seconds", type=float, default=2.0)
    p.add_argument("--latent-fallback-rate", type=int, default=16000)
    p.add_argument("--latent-noise-std", type=float, default=0.20)
    p.add_argument(
        "--latent-reinject-dir",
        default="",
        help="Optional accepted-wave library dir to reinject from (expects index.jsonl). Defaults to <output-dir>/accepted_wave_library.",
    )
    p.add_argument("--latent-reinject-ratio", type=float, default=0.50)
    p.add_argument("--latent-reinject-copy-gain", type=float, default=0.70)
    p.add_argument("--latent-reinject-noise-gain", type=float, default=0.35)
    p.add_argument("--latent-structured-ratio", type=float, default=0.85)
    p.add_argument("--latent-structured-gain", type=float, default=0.80)
    p.add_argument("--latent-structured-noise-gain", type=float, default=0.20)
    p.add_argument(
        "--latent-berkeley-imprint-mix-targets",
        dest="latent_berkeley_imprint_mix_targets",
        action="store_true",
        help="When latent fallback is active in berkeley mode, write deterministic Berkeley image payloads into mix-stream target bits.",
    )
    p.add_argument(
        "--no-latent-berkeley-imprint-mix-targets",
        dest="latent_berkeley_imprint_mix_targets",
        action="store_false",
    )
    p.add_argument("--latent-berkeley-imprint-steps", type=int, default=12)
    p.add_argument("--latent-berkeley-imprint-lr", type=float, default=0.08)
    p.add_argument("--latent-berkeley-imprint-l2", type=float, default=0.02)
    p.add_argument("--latent-berkeley-imprint-min-target-classes", type=int, default=2)
    p.add_argument("--latent-berkeley-imprint-max-target-classes", type=int, default=4)
    p.add_argument(
        "--resume-from",
        default="",
        help="Optional prior run directory. If set, load checkpoints/config/history from there.",
    )
    p.add_argument(
        "--auto-resume",
        action="store_true",
        help="Auto-resume from current output-dir artifacts when present.",
    )
    p.add_argument(
        "--force-config-search",
        action="store_true",
        help="When resuming, ignore prior best config and run config search again.",
    )
    p.add_argument(
        "--checkpoint-every-round",
        type=int,
        default=1,
        help="Save pipeline checkpoint every N orchestration rounds (0 disables periodic snapshots).",
    )
    p.add_argument(
        "--checkpoint-after-training-segment",
        dest="checkpoint_after_training_segment",
        action="store_true",
        help="Save pipeline/model checkpoints immediately after each completed training segment.",
    )
    p.add_argument(
        "--no-checkpoint-after-training-segment",
        dest="checkpoint_after_training_segment",
        action="store_false",
    )
    p.add_argument(
        "--objective-mode",
        choices=["berkeley_multilabel", "wave_labels"],
        default="berkeley_multilabel",
        help="Search/optimize objective source. 'berkeley_multilabel' is the intended feature-presence metric mode.",
    )
    p.add_argument("--label-mode", choices=["folder", "spectral"], default="folder")
    p.add_argument("--pseudo-classes", type=int, default=4)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--amp", action="store_true", help="Enable autocast mixed precision on CUDA.")
    p.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--channels-last", action="store_true", help="Use NHWC memory format for image tensors/models.")
    p.add_argument("--compile-models", action="store_true", help="Use torch.compile for classifier/transformer models.")
    p.add_argument("--compile-mode", default="default")
    p.add_argument("--grad-accum-steps", type=int, default=1, help="Micro-batch accumulation factor.")
    p.add_argument("--pin-memory-wave-batches", action="store_true", help="Pin sampled wave batches before H2D copy.")
    p.add_argument("--cache-wave-cls-dataset-on-device", action="store_true", help="Cache rendered wave-classifier datasets on GPU.")
    p.add_argument("--cache-wave-streams-on-device", action="store_true", help="Cache waveform stream pools on GPU for vectorized batch sampling.")
    p.add_argument(
        "--stage-module-offload",
        dest="stage_module_offload",
        action="store_true",
        help="Move inactive stage modules (generator/discriminator/wave-classifier) to CPU between stages to free VRAM.",
    )
    p.add_argument("--no-stage-module-offload", dest="stage_module_offload", action="store_false")
    p.add_argument(
        "--stage-module-offload-empty-cache",
        dest="stage_module_offload_empty_cache",
        action="store_true",
        help="Run gc.collect()+torch.cuda.empty_cache() after staged CPU offload transitions.",
    )
    p.add_argument("--no-stage-module-offload-empty-cache", dest="stage_module_offload_empty_cache", action="store_false")
    p.add_argument("--no-cudnn-benchmark", dest="cudnn_benchmark", action="store_false")
    p.add_argument("--no-tf32", dest="allow_tf32", action="store_false")
    p.add_argument("--matmul-precision", choices=["high", "medium", "highest"], default="high")

    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--render-max-points", type=int, default=262144)
    p.add_argument(
        "--enforce-render-config",
        action="store_true",
        help="Bypass render config search/resume and use the fixed render config arguments below.",
    )
    p.add_argument("--enforce-render-bitmode", default="auto")
    p.add_argument("--enforce-render-use-channels", choices=["auto", "1", "2"], default="auto")
    p.add_argument("--enforce-render-mono-pick", choices=["Mix", "L", "R", "LR stride"], default="Mix")
    p.add_argument("--enforce-render-width", type=int, default=512)
    p.add_argument("--enforce-render-downsample", type=int, default=1)
    p.add_argument("--enforce-render-color-mode", default=COLOR_MODES[0][0], choices=[m[0] for m in COLOR_MODES])
    p.add_argument("--enforce-render-empty-fill", type=int, default=0)
    p.add_argument("--enforce-render-bitmask-enable", dest="enforce_render_bitmask_enable", action="store_true")
    p.add_argument("--no-enforce-render-bitmask-enable", dest="enforce_render_bitmask_enable", action="store_false")
    p.add_argument("--enforce-render-bitmask-low", type=int, default=0)
    p.add_argument("--enforce-render-bitmask-high", type=int, default=7)
    p.add_argument("--enforce-render-max-points", type=int, default=262144)

    p.add_argument("--config-trials", type=int, default=10)
    p.add_argument("--config-epochs", type=int, default=2)
    p.add_argument("--search-train-limit", type=int, default=96)
    p.add_argument("--search-val-limit", type=int, default=48)
    p.add_argument("--try-scipy-refine", action="store_true")
    p.add_argument("--score-topk", type=int, default=3)
    p.add_argument("--score-threshold", type=float, default=0.35)
    p.add_argument("--score-w-topk", type=float, default=0.60)
    p.add_argument("--score-w-cov", type=float, default=0.30)
    p.add_argument("--score-w-mean", type=float, default=0.10)

    p.add_argument("--classifier-epochs", type=int, default=6)
    p.add_argument("--classifier-batch-size", type=int, default=32)
    p.add_argument("--classifier-lr", type=float, default=2e-3)
    p.add_argument("--classifier-base-ch", type=int, default=64, help="Base channel width for image classifiers.")
    p.add_argument("--classifier-max-ch", type=int, default=384, help="Max channel cap for image classifiers.")
    p.add_argument("--classifier-context-blocks", type=int, default=8, help="Residual context blocks in image classifiers.")
    p.add_argument("--classifier-context-dropout", type=float, default=0.05, help="Dropout in classifier context blocks.")
    p.add_argument(
        "--lr-sine-cycles",
        type=float,
        default=1.0,
        help="Sinusoidal LR cycles. Set <=0 (with --lr-sine-frequency <=0) to keep constant base LR.",
    )
    p.add_argument("--lr-sine-frequency", type=float, default=0.0, help="Cycles per optimizer step; overrides --lr-sine-cycles when > 0.")
    p.add_argument("--lr-sine-tail-fraction", type=float, default=0.15)
    p.add_argument("--lr-sine-min-scale", type=float, default=0.0)
    p.add_argument(
        "--classifier-init-ckpt",
        default="",
        help="Optional pretrained classifier checkpoint (for warm-starting wave classifier).",
    )
    p.add_argument(
        "--classifier-init-required",
        dest="classifier_init_required",
        action="store_true",
        help=(
            "Require --classifier-init-ckpt to exist/load. "
            "When disabled, Berkeley objective can bootstrap classifier from scratch if checkpoint is absent."
        ),
    )
    p.add_argument("--no-classifier-init-required", dest="classifier_init_required", action="store_false")
    p.add_argument(
        "--classifier-init-scope",
        choices=["features", "all"],
        default="features",
        help="When loading init checkpoint, load only feature extractor or all matching keys.",
    )
    p.add_argument(
        "--classifier-freeze-features",
        action="store_true",
        help="Freeze classifier feature extractor during wave-domain classifier training.",
    )
    p.add_argument(
        "--label-embeddings",
        dest="label_embeddings_enabled",
        action="store_true",
        help="Use text-embedding label bank for classifier logits (replaces direct class-weight logits).",
    )
    p.add_argument("--no-label-embeddings", dest="label_embeddings_enabled", action="store_false")
    p.add_argument(
        "--label-embedding-backend",
        choices=["sentence_transformers"],
        default="sentence_transformers",
        help="Text embedding backend for label bank. Only sentence-transformers is supported.",
    )
    p.add_argument(
        "--label-embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformers model id used when backend is sentence_transformers.",
    )
    p.add_argument(
        "--label-embedding-dim",
        type=int,
        default=384,
        help="Label embedding size. Must match sentence-transformers model native dimension.",
    )
    p.add_argument("--label-embedding-temperature", type=float, default=10.0)
    p.add_argument(
        "--label-text-json",
        default="",
        help="Optional JSON override for class label text (list aligned by index or dict by class name/index).",
    )
    p.add_argument(
        "--label-query-texts",
        default="",
        help="Optional query text list (split by | ; , or newline) for nearest-label retrieval report.",
    )
    p.add_argument("--label-query-topk", type=int, default=5)
    p.add_argument(
        "--semantic-vocab-extra-texts",
        default="",
        help="Optional extra semantic vocabulary terms (split by | ; , or newline) appended to supervised labels.",
    )
    p.add_argument(
        "--semantic-vocab-extra-json",
        default="",
        help="Optional JSON file containing extra semantic vocabulary terms (list[str] or dict with list field).",
    )
    p.add_argument(
        "--semantic-vocab-extra-slots",
        type=int,
        default=0,
        help="Reserve this many extra semantic slots beyond supervised Berkeley classes (0 keeps legacy behavior).",
    )
    p.add_argument(
        "--semantic-vocab-churn-enabled",
        dest="semantic_vocab_churn_enabled",
        action="store_true",
        help="Rotate active extra semantic terms each cycle while keeping Berkeley supervised classes fixed.",
    )
    p.add_argument("--no-semantic-vocab-churn-enabled", dest="semantic_vocab_churn_enabled", action="store_false")
    p.add_argument(
        "--semantic-vocab-churn-replace-per-cycle",
        type=int,
        default=1,
        help="How many active extra semantic slots to replace each cycle when churn is enabled.",
    )
    p.add_argument(
        "--semantic-vocab-churn-sweep-cycles",
        type=int,
        default=0,
        help=(
            "If >0, auto-raise per-cycle churn replacement so the currently available non-active churn terms "
            "are swept through in roughly this many cycles (bounded by unlocked extra slots)."
        ),
    )
    p.add_argument(
        "--semantic-vocab-regurgitated-churn-enabled",
        dest="semantic_vocab_regurgitated_churn_enabled",
        action="store_true",
        help=(
            "When accepted/regurgitated waves are reingested, enqueue their semantic tags into a "
            "lifetime-limited churn reservoir used as an extra candidate source during semantic churn."
        ),
    )
    p.add_argument(
        "--no-semantic-vocab-regurgitated-churn-enabled",
        dest="semantic_vocab_regurgitated_churn_enabled",
        action="store_false",
    )
    p.add_argument(
        "--semantic-vocab-regurgitated-churn-prob",
        type=float,
        default=0.35,
        help="Probability per cycle that the regurgitated churn reservoir is sampled.",
    )
    p.add_argument(
        "--semantic-vocab-regurgitated-churn-lifetime",
        type=int,
        default=3,
        help="How many churn uses each regurgitated reservoir item survives before expiry.",
    )
    p.add_argument(
        "--semantic-vocab-regurgitated-churn-max-per-cycle",
        type=int,
        default=4,
        help="Maximum regurgitated reservoir items consumed (and lifetime-decremented) per churn cycle.",
    )
    p.add_argument(
        "--semantic-vocab-regurgitated-churn-capacity",
        type=int,
        default=512,
        help="Maximum number of regurgitated reservoir items retained.",
    )
    p.add_argument(
        "--semantic-vocab-text-condition-temperature",
        type=float,
        default=8.0,
        help="Temperature used when projecting text-term embeddings into semantic condition vectors.",
    )
    p.add_argument(
        "--semantic-vocab-text-condition-topk",
        type=int,
        default=0,
        help="Semantic-dimension pruning is disabled in strict mode; must be set to 0.",
    )
    p.add_argument(
        "--semantic-vocab-unknown-label-rows-per-cycle",
        type=int,
        default=8,
        help="Additional symbol payload rows per cycle sourced from unknown (out-of-vocabulary) term labels.",
    )
    p.add_argument(
        "--semantic-vocab-bootstrap-origin-label",
        default="internal bootstrap root vocab",
        help="Origin label text injected for bootstrap/unknown symbol conditioning vectors.",
    )
    p.add_argument(
        "--semantic-vocab-auto-symbol-pool",
        dest="semantic_vocab_auto_symbol_pool",
        action="store_true",
        help=(
            "Build semantic symbol pool from official datasets (MNIST/EMNIST/KMNIST) plus synthetic core semantic terms; "
            "used for churn slots and payload augmentation."
        ),
    )
    p.add_argument("--no-semantic-vocab-auto-symbol-pool", dest="semantic_vocab_auto_symbol_pool", action="store_false")
    p.add_argument(
        "--semantic-vocab-symbol-pool-root",
        default="toys_to_survive_development/data/semantic_symbol_pool",
        help="Local cache directory for auto-downloaded symbol datasets used by semantic churn.",
    )
    p.add_argument(
        "--semantic-vocab-symbol-samples-per-term",
        type=int,
        default=1,
        help="Max symbol examples retained per auto term (digit/letter/pictogram).",
    )
    p.add_argument(
        "--semantic-vocab-gestation-train-target-samples",
        type=int,
        default=0,
        help=(
            "Target Stage-1 gestation training rows after bootstrap augmentation expansion. "
            "0 disables expansion (use raw gestation split only)."
        ),
    )
    p.add_argument(
        "--semantic-vocab-gestation-val-target-samples",
        type=int,
        default=0,
        help=(
            "Target Gate-1 gestation validation rows after bootstrap augmentation expansion. "
            "0 uses max(raw_val_rows, round(train_target*0.2))."
        ),
    )
    p.add_argument(
        "--semantic-vocab-symbol-include-pictograms",
        dest="semantic_vocab_symbol_include_pictograms",
        action="store_true",
        help="Include KMNIST pictogram terms in the auto symbol pool when available.",
    )
    p.add_argument(
        "--no-semantic-vocab-symbol-include-pictograms",
        dest="semantic_vocab_symbol_include_pictograms",
        action="store_false",
    )
    p.add_argument(
        "--semantic-vocab-reference-flashcards",
        dest="semantic_vocab_reference_flashcards",
        action="store_true",
        help=(
            "Append synthetic/reference semantic flashcard rows to payload conditioning each cycle. "
            "Flashcards cover locked core terms and blend Berkeley object seeds with symbol/damage variants."
        ),
    )
    p.add_argument(
        "--no-semantic-vocab-reference-flashcards",
        dest="semantic_vocab_reference_flashcards",
        action="store_false",
    )
    p.add_argument(
        "--semantic-vocab-reference-flashcards-per-term",
        type=int,
        default=2,
        help="How many reference flashcard rows to synthesize per locked core semantic term.",
    )
    p.add_argument(
        "--gd-vocab-library-enabled",
        dest="gd_vocab_library_enabled",
        action="store_true",
        help="Persist and reuse generator/discriminator checkpoints keyed by semantic vocabulary hash.",
    )
    p.add_argument("--no-gd-vocab-library-enabled", dest="gd_vocab_library_enabled", action="store_false")
    p.add_argument(
        "--gd-vocab-library-dir",
        default="",
        help="Optional directory for hashed G/D vocabulary snapshots (default: <output-dir>/gd_vocab_library).",
    )
    p.add_argument(
        "--gd-vocab-library-autoload",
        dest="gd_vocab_library_autoload",
        action="store_true",
        help="Attempt to load matching hashed G/D snapshot when active semantic vocabulary changes.",
    )
    p.add_argument("--no-gd-vocab-library-autoload", dest="gd_vocab_library_autoload", action="store_false")
    p.add_argument("--berkeley-data-root", default="toys_to_survive_development/data/berkeley_sbd")
    p.add_argument(
        "--berkeley-image-size",
        type=int,
        default=0,
        help=(
            "0 means reuse --image-size; when >0, this value becomes the shared embed resolution "
            "for Berkeley + transformer/classifier rendering."
        ),
    )
    p.add_argument(
        "--berkeley-payload-max-samples",
        type=int,
        default=1024,
        help=(
            "Cap Berkeley payload-bank samples used by generator stages. "
            "Set 0 to use the legacy auto policy."
        ),
    )
    p.add_argument(
        "--berkeley-payload-source-root",
        default="",
        help=(
            "Optional folder-drop source root for payload ingestion. "
            "Each direct subfolder is treated as a dataset source label; defaults to "
            "<berkeley-data-root>/payload_sources."
        ),
    )
    p.add_argument(
        "--berkeley-payload-cache-rebuild",
        dest="berkeley_payload_cache_rebuild",
        action="store_true",
        help="Force rebuild of disk payload cache from Berkeley + payload source folders.",
    )
    p.add_argument("--no-berkeley-payload-cache-rebuild", dest="berkeley_payload_cache_rebuild", action="store_false")
    p.add_argument("--berkeley-auto-install-scipy", action="store_true")
    p.add_argument("--berkeley-refresh-every", type=int, default=0, help="Run Berkeley refresh every N config trials; 0=off.")
    p.add_argument(
        "--berkeley-refresh-round-every",
        type=int,
        default=0,
        help="In orchestration mode, run Berkeley refresh every N rounds; 0=off.",
    )
    p.add_argument("--berkeley-refresh-epochs", type=int, default=1)
    p.add_argument("--berkeley-refresh-lr", type=float, default=2e-4)
    p.add_argument("--berkeley-refresh-weight-decay", type=float, default=1e-4)
    p.add_argument("--berkeley-refresh-batch-size", type=int, default=32, help="Refresh train batch size. Set 0 to auto-size from GPU free memory when cache is enabled.")
    p.add_argument("--berkeley-refresh-loader-batch-size", type=int, default=128, help="Batch size used only while building Berkeley refresh cache/loader input queue.")
    p.add_argument("--berkeley-refresh-workers", type=int, default=2)
    p.add_argument("--berkeley-refresh-log-every", type=int, default=0, help="0 disables periodic Berkeley-refresh progress logs.")
    p.add_argument("--berkeley-refresh-max-seconds", type=float, default=0.0, help="0 disables wall-time cap per refresh call.")
    p.add_argument(
        "--berkeley-refresh-cache-batches",
        type=int,
        default=0,
        help="If >0, pre-cache this many Berkeley refresh batches for repeated GPU-fed refresh.",
    )
    p.add_argument(
        "--berkeley-refresh-cache-device",
        choices=["none", "auto", "cpu", "cuda"],
        default="none",
        help="Storage device for cached Berkeley refresh pool.",
    )
    p.add_argument("--berkeley-refresh-vram-fraction", type=float, default=0.90, help="Target fraction of free VRAM to fill for auto refresh batch.")
    p.add_argument("--berkeley-refresh-activation-mult", type=float, default=10.0, help="Activation overhead multiplier used for auto refresh batch sizing.")
    p.add_argument("--berkeley-refresh-max-batch-cap", type=int, default=0, help="Hard cap for auto refresh batch size (0 = no cap).")
    p.add_argument("--loader-persistent-workers", dest="loader_persistent_workers", action="store_true")
    p.add_argument("--no-loader-persistent-workers", dest="loader_persistent_workers", action="store_false")
    p.add_argument("--loader-prefetch-factor", type=int, default=4)
    p.add_argument("--berkeley-refresh-max-train", type=int, default=0, help="0 means full Berkeley train set.")
    p.add_argument("--berkeley-refresh-max-steps", type=int, default=0, help="0 means full epoch over refresh loader.")
    p.add_argument("--final-train-limit", type=int, default=0, help="0 means all.")
    p.add_argument("--final-val-limit", type=int, default=0, help="0 means all.")
    p.add_argument(
        "--classifier-subset-fresh-round-sampling",
        dest="classifier_subset_fresh_round_sampling",
        action="store_true",
        help="When --final-val-limit selects a subset, resample that subset for each round-level classifier metric pass.",
    )
    p.add_argument(
        "--no-classifier-subset-fresh-round-sampling",
        dest="classifier_subset_fresh_round_sampling",
        action="store_false",
    )

    p.add_argument("--transformer-epochs", type=int, default=6)
    p.add_argument("--transformer-steps", type=int, default=80)
    p.add_argument("--transformer-batch-size", type=int, default=16)
    p.add_argument(
        "--transformer-round-batch-size",
        type=int,
        default=0,
        help="If >0, batch size override for orchestration R-stage transformer updates (0 = use --transformer-batch-size).",
    )
    p.add_argument(
        "--joint-transformer-batch-size",
        type=int,
        default=0,
        help=(
            "If >0, batch size override for orchestration J-stage transformer updates. "
            "0 auto-resolves from R-stage batch with a conservative memory cap."
        ),
    )
    p.add_argument(
        "--transformer-eval-batch-size",
        type=int,
        default=0,
        help="If >0, batch size used for transformer stage evaluations/gates (0 = auto from active train batch).",
    )
    p.add_argument("--transformer-d-model", type=int, default=256)
    p.add_argument("--transformer-nhead", type=int, default=8)
    p.add_argument("--transformer-num-layers", type=int, default=8)
    p.add_argument("--transformer-ff-mult", type=int, default=4)
    p.add_argument("--transformer-dropout", type=float, default=0.10)
    p.add_argument(
        "--transformer-filter-bundles",
        type=str,
        default="deskew",
        help="Comma-separated pre/post filter bundle names wrapped around transformer (use 'none' to disable).",
    )
    p.add_argument(
        "--transformer-deskew-prefilter-max-skew",
        type=float,
        default=0.25,
        help="Max absolute skew predicted/applied by the deskew prefilter bundle.",
    )
    p.add_argument(
        "--transformer-deskew-prefilter-weight",
        type=float,
        default=0.20,
        help="Loss weight for deskew prefilter skew prediction.",
    )
    p.add_argument(
        "--transformer-deskew-residual-weight",
        type=float,
        default=0.10,
        help="Loss weight for deskew post-transform residual observer.",
    )
    p.add_argument(
        "--transformer-deskew-pre-token-mix",
        type=float,
        default=0.08,
        help="Bounded additive mix for deskew prefilter token residual injection into transformer input tokens.",
    )
    p.add_argument(
        "--transformer-deskew-post-token-mix",
        type=float,
        default=0.08,
        help="Bounded additive mix for deskew postfilter token residual injection before transformer patch_out.",
    )
    p.add_argument(
        "--transformer-deskew-token-residual-weight",
        type=float,
        default=0.01,
        help="L2 regularization weight for deskew token residual injections.",
    )
    p.add_argument("--transformer-log-every", type=int, default=0, help="If >0, print transformer stage timing/progress every N train steps.")
    p.add_argument(
        "--transformer-visualize-status",
        dest="transformer_visualize_status",
        action="store_true",
        help="Open pygame OpenGL viewer and blit transformer input/output images at each status report.",
    )
    p.add_argument("--no-transformer-visualize-status", dest="transformer_visualize_status", action="store_false")
    p.add_argument("--transformer-viz-scale", type=int, default=3, help="Display scale multiplier for status viewer.")
    p.add_argument("--transformer-cache-eval-batches", dest="transformer_cache_eval_batches", action="store_true")
    p.add_argument("--no-transformer-cache-eval-batches", dest="transformer_cache_eval_batches", action="store_false")
    p.add_argument("--transformer-degrade-inputs", dest="transformer_degrade_inputs", action="store_true")
    p.add_argument("--no-transformer-degrade-inputs", dest="transformer_degrade_inputs", action="store_false")
    p.add_argument("--transformer-degrade-min-strength", type=float, default=0.25)
    p.add_argument("--transformer-degrade-max-strength", type=float, default=0.95)
    p.add_argument("--transformer-degrade-noise-std-min", type=float, default=0.01)
    p.add_argument("--transformer-degrade-noise-std-max", type=float, default=0.14)
    p.add_argument("--transformer-degrade-dropout-max", type=float, default=0.25)
    p.add_argument("--transformer-degrade-quant-bits-min", type=int, default=3)
    p.add_argument("--transformer-degrade-quant-bits-max", type=int, default=10)
    p.add_argument(
        "--transformer-degrade-mode",
        choices=["full", "blur_stride"],
        default="full",
        help="Degradation pipeline mode. Use blur_stride to keep only blur + stride-skew effects.",
    )
    p.add_argument(
        "--transformer-degrade-domain",
        choices=["waveform", "bitwindow"],
        default="waveform",
        help="Apply degradation either to full waveform samples or only inside the configured bit window.",
    )
    p.add_argument(
        "--transformer-degrade-stride-skew-max",
        type=float,
        default=0.0,
        help="Max per-sample stride skew factor; 0 disables stride-skew warp.",
    )
    p.add_argument(
        "--transformer-degrade-vectorized-window",
        type=int,
        default=4,
        help="Prepare this many training steps at once using vectorized degradation (>=1).",
    )
    p.add_argument("--transformer-loss-entropy-weight", type=float, default=0.60)
    p.add_argument("--transformer-loss-high-bit-weight", type=float, default=0.50)
    p.add_argument("--transformer-loss-low-bit-weight", type=float, default=0.08)
    p.add_argument("--transformer-loss-wave-l1-weight", type=float, default=1.00)
    p.add_argument(
        "--transformer-loss-score-target-weight",
        type=float,
        default=1.50,
        help="Weight for keeping transformed-wave classifier feature score at/above clean-wave target.",
    )
    p.add_argument(
        "--transformer-loss-score-target-margin",
        type=float,
        default=0.02,
        help="Required improvement margin for target score (after vs before); loss penalizes shortfall.",
    )
    p.add_argument(
        "--transformer-loss-score-rank-weight",
        type=float,
        default=0.40,
        help="Additional signal-gain weight for target-vs-non-target ranking improvement.",
    )
    p.add_argument(
        "--transformer-loss-score-rank-margin",
        type=float,
        default=0.02,
        help="Margin used by ranking gap: target scores should exceed non-target max by this amount.",
    )
    p.add_argument(
        "--transformer-loss-score-spurious-weight",
        type=float,
        default=0.35,
        help="Additional signal-gain weight for suppressing non-target/spurious labels.",
    )
    p.add_argument(
        "--transformer-loss-score-spurious-threshold",
        type=float,
        default=0.35,
        help="Probability threshold used for soft counting spurious non-target activations.",
    )
    p.add_argument(
        "--transformer-loss-score-spurious-temp",
        type=float,
        default=0.20,
        help="Temperature for soft spurious-hit counting around spurious threshold.",
    )
    p.add_argument(
        "--transformer-loss-score-spurious-margin",
        type=float,
        default=0.00,
        help="Required reduction margin for spurious labels (after must be below before-margin).",
    )
    p.add_argument(
        "--target-label-knockout-prob",
        type=float,
        default=0.0,
        help=(
            "Per-row probability of randomly dropping a subset of active target labels during training-target assembly "
            "(0 disables label knockout)."
        ),
    )
    p.add_argument(
        "--target-label-knockout-max-drop-frac",
        type=float,
        default=0.50,
        help="Maximum fraction of active labels that may be dropped when knockout is applied to a row.",
    )
    p.add_argument(
        "--target-label-knockout-min-keep",
        type=int,
        default=1,
        help="Minimum number of active labels to preserve in a row after knockout.",
    )
    p.add_argument(
        "--transformer-accepted-preload-max",
        type=int,
        default=0,
        help="If >0, preload up to this many accepted-library waves into transformer train streams before rounds start.",
    )
    p.add_argument(
        "--transformer-accepted-append-max-per-round",
        type=int,
        default=0,
        help="If >0, append at most this many newly accepted waves per round into transformer train streams (0=all).",
    )
    p.add_argument(
        "--orchestration-mode",
        choices=["none", "sequential", "alternating", "staged_cgr", "staged_cgrw"],
        default="sequential",
        help=(
            "In berkeley_multilabel mode: training schedule "
            "(staged_cgr = C gate -> G gate -> R gate, "
            "staged_cgrw = C gate -> G gate -> R gate -> W feedback)."
        ),
    )
    p.add_argument("--orchestration-cycles", type=int, default=3)
    p.add_argument("--orchestration-rounds", type=int, default=2, help="Used when mode=alternating, staged_cgr, or staged_cgrw.")
    p.add_argument("--generator-epochs-per-round", type=int, default=1)
    p.add_argument("--generator-steps-per-round", type=int, default=120)
    p.add_argument("--generator-batch-size", type=int, default=64)
    p.add_argument(
        "--generator-grad-accum-steps",
        type=int,
        default=1,
        help="Micro-batch accumulation factor for generator/discriminator updates.",
    )
    p.add_argument(
        "--generator-classifier-batch-cap",
        type=int,
        default=0,
        help="If >0, chunk classifier forward passes in generator stage to this batch size cap.",
    )
    p.add_argument("--generator-z-dim", type=int, default=128)
    p.add_argument("--generator-base-ch", type=int, default=128)
    p.add_argument("--generator-depth", type=int, default=6)
    p.add_argument("--generator-min-ch", type=int, default=24)
    p.add_argument("--generator-lr", type=float, default=2e-4)
    p.add_argument("--discriminator-lr", type=float, default=2e-4)
    p.add_argument("--discriminator-base-ch", type=int, default=96)
    p.add_argument("--discriminator-depth", type=int, default=5)
    p.add_argument("--discriminator-max-ch", type=int, default=512)
    p.add_argument(
        "--discriminator-steps-per-generator-step",
        type=int,
        default=1,
        help="Number of discriminator updates per generator update (n_critic style).",
    )
    p.add_argument("--generator-log-every", type=int, default=20)
    p.add_argument("--generator-loss-adv-weight", type=float, default=0.50)
    p.add_argument("--generator-loss-cls-weight", type=float, default=1.50)
    p.add_argument(
        "--generator-loss-wave-weight",
        type=float,
        default=0.0,
        help="Additional generator loss weight from the differentiable wave/refine/render path.",
    )
    p.add_argument(
        "--generator-wave-score-margin",
        type=float,
        default=0.02,
        help="Required score margin for wave-coupled generator path (after vs before).",
    )
    p.add_argument(
        "--generator-wave-denoise-weight",
        type=float,
        default=0.25,
        help="Weight for wave-path denoise term in generator coupling loss.",
    )
    p.add_argument(
        "--generator-wave-carrier-blend",
        type=float,
        default=0.35,
        help="Blend ratio between sampled carrier waves and generator payload wave in coupling path.",
    )
    p.add_argument(
        "--generator-wave-strength",
        type=float,
        default=0.35,
        help="Degradation strength for the generator wave-coupling path.",
    )
    p.add_argument(
        "--generator-wave-stride-skew-max",
        type=float,
        default=0.18,
        help="Stride-skew max for generator wave-coupling degradation.",
    )
    p.add_argument(
        "--generator-wave-degrade-mode",
        choices=["full", "blur_stride"],
        default="blur_stride",
        help="Degradation mode used in generator wave-coupling path.",
    )
    p.add_argument("--generator-gate-min-target-prob", type=float, default=0.55)
    p.add_argument("--generator-gate-maintain-rounds", type=int, default=1)
    p.add_argument(
        "--generator-fake-feedback-enabled",
        dest="generator_fake_feedback_enabled",
        action="store_true",
        help=(
            "When staged_cgr/staged_cgrw is active, train classifier with generated images using a dedicated "
            "'GAN image' text-embedding vector and discriminator pass/fail balanced feedback."
        ),
    )
    p.add_argument(
        "--no-generator-fake-feedback-enabled",
        dest="generator_fake_feedback_enabled",
        action="store_false",
    )
    p.add_argument(
        "--fake-image-sentinel-label",
        type=str,
        default="GAN image",
        help="Text prompt encoded by the active label-embedding backend for fake-feedback vector supervision.",
    )
    p.add_argument("--generator-fake-feedback-epochs", type=int, default=1)
    p.add_argument("--generator-fake-feedback-steps-per-round", type=int, default=48)
    p.add_argument("--generator-fake-feedback-batch-size", type=int, default=32)
    p.add_argument("--generator-fake-feedback-lr", type=float, default=6e-4)
    p.add_argument("--generator-fake-feedback-weight-decay", type=float, default=1e-4)
    p.add_argument("--generator-fake-feedback-vector-weight", type=float, default=1.0)
    p.add_argument("--generator-fake-feedback-condition-weight", type=float, default=0.35)
    p.add_argument("--generator-fake-feedback-disc-conf-temperature", type=float, default=1.0)
    p.add_argument("--generator-fake-feedback-disc-conf-floor", type=float, default=0.25)
    p.add_argument(
        "--generator-fake-feedback-disc-balance-groups",
        dest="generator_fake_feedback_disc_balance_groups",
        action="store_true",
        help="Balance discriminator-positive and discriminator-negative generated groups while applying confidence weights.",
    )
    p.add_argument(
        "--no-generator-fake-feedback-disc-balance-groups",
        dest="generator_fake_feedback_disc_balance_groups",
        action="store_false",
    )
    p.add_argument("--generator-fake-feedback-log-every", type=int, default=0)
    p.add_argument(
        "--generator-fake-feedback-include-condition-targets",
        dest="generator_fake_feedback_include_condition_targets",
        action="store_true",
        help="Include generator condition targets alongside fake-vector supervision during fake-feedback refresh.",
    )
    p.add_argument(
        "--no-generator-fake-feedback-include-condition-targets",
        dest="generator_fake_feedback_include_condition_targets",
        action="store_false",
    )
    p.add_argument(
        "--joint-enabled",
        dest="joint_enabled",
        action="store_true",
        help="After C/G/R gates are satisfied, run coupled joint G+R updates each round.",
    )
    p.add_argument("--no-joint-enabled", dest="joint_enabled", action="store_false")
    p.add_argument("--joint-generator-epochs-per-round", type=int, default=1)
    p.add_argument("--joint-generator-steps-per-round", type=int, default=48)
    p.add_argument("--joint-transformer-epochs-per-round", type=int, default=1)
    p.add_argument("--joint-transformer-steps-per-round", type=int, default=40)
    p.add_argument("--transformer-epochs-per-round", type=int, default=1)
    p.add_argument("--transformer-steps-per-round", type=int, default=40)
    p.add_argument("--wave-cls-epochs-per-round", type=int, default=1)
    p.add_argument("--wave-cls-train-samples", type=int, default=768)
    p.add_argument("--wave-cls-val-samples", type=int, default=256)
    p.add_argument("--wave-cls-batch-size", type=int, default=32)
    p.add_argument("--wave-cls-lr", type=float, default=1e-3)
    p.add_argument("--wave-cls-log-every", type=int, default=0, help="If >0, emit wave-classifier preview/log every N train steps.")
    p.add_argument(
        "--wave-cls-semantic-mix-noise-prob",
        type=float,
        default=0.10,
        help="Per-sample probability of semantic mix/noise augmentation during wave-classifier training.",
    )
    p.add_argument(
        "--wave-cls-semantic-mix-noise-std",
        type=float,
        default=0.03,
        help="Gaussian noise std applied on mixed wave-classifier samples.",
    )
    p.add_argument(
        "--wave-cls-semantic-mix-blend-min",
        type=float,
        default=0.10,
        help="Lower blend bound for semantic mix augmentation.",
    )
    p.add_argument(
        "--wave-cls-semantic-mix-blend-max",
        type=float,
        default=0.35,
        help="Upper blend bound for semantic mix augmentation.",
    )
    p.add_argument(
        "--wave-zero-shot-query-texts",
        default="",
        help=(
            "Optional open-vocabulary text query list for wave-classifier zero-shot inference "
            "(split by | ; , or newline). When empty, falls back to --label-query-texts."
        ),
    )
    p.add_argument("--wave-zero-shot-topk", type=int, default=3, help="Per-sample top-k queries to report for wave zero-shot inference.")
    p.add_argument("--wave-zero-shot-max-samples", type=int, default=128, help="Max wave-validation samples used per zero-shot evaluation (0=all).")
    p.add_argument("--wave-zero-shot-report-samples", type=int, default=8, help="How many sample-level zero-shot rows to store.")
    p.add_argument(
        "--wave-cls-accepted-only",
        action="store_true",
        help="Train wave classifier on only accepted transformer outputs (score/l1 gated).",
    )
    p.add_argument(
        "--gate-gestation-loss-target",
        type=float,
        default=0.18,
        help=(
            "Hard gestation gate: maximum allowed average classifier loss on bootstrap-archetype-only rows. "
            "No downstream stage runs until this gate is satisfied."
        ),
    )
    p.add_argument(
        "--gate-gestation-maintain-rounds",
        type=int,
        default=1,
        help="Required consecutive gestation gate passes before Stage-2 validation gate can pass.",
    )
    p.add_argument(
        "--gate-gestation-val-max",
        type=int,
        default=0,
        help="Max gestation samples per gate eval (0=all bootstrap-archetype rows).",
    )
    p.add_argument(
        "--gate-gestation-batch-size",
        type=int,
        default=0,
        help=(
            "Batch size for gestation gate eval. "
            "0 uses --gate-berkeley-batch-size, then --berkeley-refresh-batch-size, then --classifier-batch-size."
        ),
    )
    p.add_argument(
        "--gate-gestation-eval-max-steps",
        type=int,
        default=0,
        help="Max gestation batches per eval (0=full gestation loader).",
    )
    p.add_argument(
        "--gate-berkeley-min-score",
        type=float,
        default=0.0,
        help="Legacy wave-score floor gate (kept for compatibility).",
    )
    p.add_argument(
        "--gate-berkeley-min-confidence",
        type=float,
        default=0.0,
        help="If >0, Stage-2 validation classifier mean-confidence gate (payload validation split).",
    )
    p.add_argument(
        "--gate-berkeley-min-macro-f1",
        type=float,
        default=0.0,
        help="If >0, Stage-2 validation classifier macro-F1 gate (payload validation split).",
    )
    p.add_argument(
        "--gate-berkeley-loss-target",
        type=float,
        default=0.12,
        help=(
            "If >0, explicit Stage-2 validation classifier loss target for loss-aware gate clamp. "
            "If <=0, target is adaptive from best recent loss * --gate-berkeley-loss-target-mult."
        ),
    )
    p.add_argument(
        "--gate-berkeley-loss-target-mult",
        type=float,
        default=1.15,
        help="Adaptive loss target multiplier over best recent Berkeley-val loss (>=1.0).",
    )
    p.add_argument(
        "--gate-berkeley-loss-min-floor",
        type=float,
        default=0.05,
        help="Lower bound for adaptive Berkeley-val loss target.",
    )
    p.add_argument(
        "--gate-berkeley-loss-thaw-rounds",
        type=int,
        default=2,
        help="Rounds where confidence/F1 can break ice before loss clamp ramps in.",
    )
    p.add_argument(
        "--gate-berkeley-loss-ramp-rounds",
        type=int,
        default=6,
        help="Rounds used to polynomially ramp loss influence after thaw.",
    )
    p.add_argument(
        "--gate-berkeley-loss-poly",
        type=float,
        default=2.0,
        help="Polynomial power for loss influence ramp (>=1).",
    )
    p.add_argument(
        "--gate-berkeley-loss-max-weight",
        type=float,
        default=0.65,
        help="Max loss weight in confidence/loss blended Berkeley gate score [0..1].",
    )
    p.add_argument(
        "--gate-berkeley-loss-clamp-memory-rounds",
        type=int,
        default=8,
        help="How many recent rounds to remember before releasing a loss clamp.",
    )
    p.add_argument(
        "--gate-berkeley-loss-release-rounds",
        type=int,
        default=3,
        help="Required consecutive good-loss rounds to release active loss clamp.",
    )
    p.add_argument(
        "--gate-berkeley-loss-blend-margin",
        type=float,
        default=0.02,
        help="Slack margin applied to blended confidence/loss clamp score.",
    )
    p.add_argument(
        "--gate-berkeley-maintain-rounds",
        type=int,
        default=1,
        help="Required consecutive Stage-2 validation gate passes before downstream is allowed.",
    )
    p.add_argument(
        "--gate-berkeley-val-max",
        type=int,
        default=0,
        help="Deprecated for Gate 2. Stage-2 validation gate always evaluates the full payload validation split.",
    )
    p.add_argument(
        "--gate-berkeley-batch-size",
        type=int,
        default=0,
        help=(
            "Batch size for Stage-2 validation gate evaluation loader. "
            "0 uses --berkeley-refresh-batch-size (or --classifier-batch-size when refresh batch is auto)."
        ),
    )
    p.add_argument(
        "--gate-berkeley-eval-max-steps",
        type=int,
        default=0,
        help="Max Stage-2 validation gate batches per eval after Gate 2 is already ready (0=full loader).",
    )
    p.add_argument(
        "--gate-total-token-schedule-enabled",
        dest="gate_total_token_schedule_enabled",
        action="store_true",
        help="Sort Stage-2 validation gate rows to prioritize active optional semantic tokens before gate decisions.",
    )
    p.add_argument("--no-gate-total-token-schedule-enabled", dest="gate_total_token_schedule_enabled", action="store_false")
    p.add_argument(
        "--gate-total-token-schedule-threshold",
        type=float,
        default=0.55,
        help="Activation threshold used while token-sorting Stage-2 validation gate rows.",
    )
    p.add_argument(
        "--gate-wave-min-entropy",
        type=float,
        default=0.12,
        help="Wave-material gate: minimum mean binary entropy on wave-rendered logits.",
    )
    p.add_argument(
        "--gate-wave-maintain-rounds",
        type=int,
        default=1,
        help="Required consecutive wave-entropy gate passes before transformer stage.",
    )
    p.add_argument(
        "--gate-transformer-min-score-after",
        type=float,
        default=0.0,
        help="If >0, transformer post-score gate threshold before wave-classifier stage.",
    )
    p.add_argument(
        "--gate-transformer-min-gain",
        type=float,
        default=-1e9,
        help="Transformer gain threshold before wave-classifier stage.",
    )
    p.add_argument(
        "--gate-transformer-maintain-rounds",
        type=int,
        default=1,
        help="Required consecutive transformer gate passes before wave-classifier stage.",
    )
    p.add_argument(
        "--gate-wave-feedback-min-acc",
        type=float,
        default=0.25,
        help=(
            "If >0, require previous round wave-classifier validation accuracy at/above this "
            "value before transformer gate can pass."
        ),
    )
    p.add_argument(
        "--gate-wave-feedback-min-zs-top1-rate",
        type=float,
        default=0.15,
        help=(
            "If >0 and zero-shot ran, require previous round wave zero-shot top1-rate at/above "
            "this value before transformer gate can pass."
        ),
    )
    p.add_argument(
        "--wave-feedback-zero-shot-weight",
        type=float,
        default=0.50,
        help="Weight of zero-shot top1-rate when building combined wave feedback score.",
    )
    p.add_argument(
        "--wave-feedback-transformer-loss-boost",
        type=float,
        default=0.50,
        help=(
            "Boost factor applied to transformer score-target loss weight when combined wave "
            "feedback score is weak."
        ),
    )
    p.add_argument(
        "--wave-feedback-generator-loss-boost",
        type=float,
        default=0.35,
        help=(
            "Boost factor applied to generator wave-coupling loss weight when combined wave "
            "feedback score is weak."
        ),
    )
    p.add_argument("--accept-score-threshold", type=float, default=0.40)
    p.add_argument("--accept-l1-threshold", type=float, default=0.20)
    p.add_argument("--library-max-per-round", type=int, default=24)
    p.add_argument("--chunk-samples", type=int, default=32768)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--transformer-lr", type=float, default=2e-4)
    p.add_argument("--max-delta", type=float, default=0.2)
    p.add_argument("--stream-max-points", type=int, default=262144)
    p.add_argument("--save-preview", type=int, default=2)
    p.add_argument(
        "--training-preview-enabled",
        dest="training_preview_enabled",
        action="store_true",
        help="Save image+label supervision previews during round updates.",
    )
    p.add_argument("--no-training-preview-enabled", dest="training_preview_enabled", action="store_false")
    p.add_argument("--training-preview-topk", type=int, default=6)
    p.add_argument("--training-preview-generator-samples", type=int, default=4)
    p.add_argument("--training-preview-every-round", type=int, default=1)
    p.add_argument(
        "--stage-opengl-preview-enabled",
        dest="stage_opengl_preview_enabled",
        action="store_true",
        help="Show live OpenGL previews across orchestration stages (C/G/R/J/W).",
    )
    p.add_argument("--no-stage-opengl-preview-enabled", dest="stage_opengl_preview_enabled", action="store_false")
    p.add_argument("--stage-opengl-preview-scale", type=int, default=3)
    p.add_argument("--stage-opengl-preview-every-round", type=int, default=1)
    p.add_argument(
        "--stage-opengl-preview-bootstrap",
        dest="stage_opengl_preview_bootstrap",
        action="store_true",
        help="Force an immediate OpenGL preview frame at startup to confirm window/display path.",
    )
    p.add_argument("--no-stage-opengl-preview-bootstrap", dest="stage_opengl_preview_bootstrap", action="store_false")
    p.add_argument(
        "--stage-opengl-preview-required",
        dest="stage_opengl_preview_required",
        action="store_true",
        help="Abort run if stage OpenGL preview cannot initialize.",
    )
    p.add_argument("--no-stage-opengl-preview-required", dest="stage_opengl_preview_required", action="store_false")
    p.set_defaults(
        cudnn_benchmark=True,
        allow_tf32=True,
        loader_persistent_workers=True,
        classifier_subset_fresh_round_sampling=True,
        transformer_cache_eval_batches=True,
        transformer_degrade_inputs=True,
        transformer_visualize_status=False,
        latent_berkeley_imprint_mix_targets=True,
        enforce_render_bitmask_enable=False,
        classifier_init_required=False,
        label_embeddings_enabled=True,
        berkeley_payload_cache_rebuild=False,
        semantic_vocab_churn_enabled=False,
        semantic_vocab_regurgitated_churn_enabled=True,
        semantic_vocab_text_condition_topk=0,
        semantic_vocab_unknown_label_rows_per_cycle=8,
        semantic_vocab_auto_symbol_pool=True,
        semantic_vocab_symbol_include_pictograms=True,
        semantic_vocab_reference_flashcards=True,
        gd_vocab_library_enabled=True,
        gd_vocab_library_autoload=True,
        training_preview_enabled=True,
        stage_opengl_preview_enabled=True,
        stage_opengl_preview_bootstrap=True,
        stage_opengl_preview_required=False,
        checkpoint_after_training_segment=False,
        stage_module_offload=True,
        stage_module_offload_empty_cache=True,
        joint_enabled=True,
        generator_fake_feedback_enabled=True,
        generator_fake_feedback_include_condition_targets=True,
        generator_fake_feedback_disc_balance_groups=True,
        gate_total_token_schedule_enabled=True,
    )
    args = p.parse_args()
    if int(args.semantic_vocab_text_condition_topk) != 0:
        raise RuntimeError(
            "Strict semantic parity mode requires --semantic-vocab-text-condition-topk=0 "
            "(no semantic-dimension pruning allowed)."
        )
    return args


def _resolve_shared_embed_image_size(args) -> int:
    image_size = max(1, int(args.image_size))
    berkeley_size = int(args.berkeley_image_size)
    if berkeley_size > 0:
        return max(1, berkeley_size)
    return image_size


def main():
    args = parse_args()
    if bool(args.stage_opengl_preview_enabled) and bool(args.transformer_visualize_status):
        # pygame uses a process-global display; keep one persistent viewer to avoid teardown conflicts.
        args.transformer_visualize_status = False
    transformer_filter_bundle_names = _parse_transformer_filter_bundle_names(args.transformer_filter_bundles)
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device if ("cuda" not in args.device or torch.cuda.is_available()) else "cpu")
    if ("cuda" in str(args.device).lower()) and (device.type != "cuda"):
        _log(f"Requested CUDA device '{args.device}' but CUDA is unavailable; falling back to CPU.")
    configure_torch_runtime(
        device=device,
        cudnn_benchmark=bool(args.cudnn_benchmark),
        allow_tf32=bool(args.allow_tf32),
        matmul_precision=str(args.matmul_precision),
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    grad_accum_steps = max(1, int(args.grad_accum_steps))
    force_payload_cache_rebuild_runtime = bool(args.berkeley_payload_cache_rebuild) or bool(getattr(args, "hard_wipe_caches", False)) or bool(getattr(args, "soft_reset_labels", False))
    stage_module_offload_runtime = bool(
        bool(args.stage_module_offload) and device.type == "cuda" and (not bool(args.compile_models))
    )
    if bool(args.stage_module_offload) and bool(args.compile_models) and device.type == "cuda":
        _log("VRAM: stage-module-offload disabled because --compile-models is enabled.")

    out_dir = Path(args.output_dir)
    semantic_cache_nonce = ""
    do_hard_wipe = bool(getattr(args, "hard_wipe_caches", False))
    do_soft_reset_labels = bool(getattr(args, "soft_reset_labels", False))
    if bool(do_hard_wipe):
        semantic_cache_nonce = str(int(time.time_ns()))
        wipe_info = _hard_wipe_pipeline_caches(
            output_dir=out_dir,
            berkeley_data_root=str(args.berkeley_data_root),
        )
        _log(
            "[hard-wipe-caches] "
            f"removed={int(len(wipe_info.get('removed', [])))} "
            f"missing={int(len(wipe_info.get('missing', [])))} "
            f"errors={int(len(wipe_info.get('errors', [])))}"
        )
        for p_txt in wipe_info.get("removed", []):
            _log(f"[hard-wipe-caches] removed: {str(p_txt)}")
        for e_txt in wipe_info.get("errors", []):
            _log(f"[hard-wipe-caches] error: {str(e_txt)}")
        if int(len(wipe_info.get("errors", []))) > 0:
            _log(
                "[hard-wipe-caches] warning: continuing with best-effort wipe; "
                "fresh semantic cache namespace enabled for this run."
            )
    elif bool(do_soft_reset_labels):
        semantic_cache_nonce = str(int(time.time_ns()))
        reset_info = _soft_reset_label_caches(
            output_dir=out_dir,
            berkeley_data_root=str(args.berkeley_data_root),
        )
        _log(
            "[soft-reset-labels] "
            f"removed={int(len(reset_info.get('removed', [])))} "
            f"missing={int(len(reset_info.get('missing', [])))} "
            f"errors={int(len(reset_info.get('errors', [])))}"
        )
        for p_txt in reset_info.get("removed", []):
            _log(f"[soft-reset-labels] removed: {str(p_txt)}")
        for e_txt in reset_info.get("errors", []):
            _log(f"[soft-reset-labels] error: {str(e_txt)}")
        if int(len(reset_info.get("errors", []))) > 0:
            _log(
                "[soft-reset-labels] warning: continuing with best-effort reset; "
                "fresh semantic cache namespace enabled for this run."
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Output dir: {out_dir}")
    _log(f"Device: {device}")
    _log(
        "Perf: "
        f"amp={amp_enabled}({args.amp_dtype}), channels_last={bool(args.channels_last)}, "
        f"compile={bool(args.compile_models)}({args.compile_mode}), grad_accum={grad_accum_steps}, "
        f"wave_stream_cache={bool(args.cache_wave_streams_on_device)}, "
        f"stage_module_offload={bool(stage_module_offload_runtime)}, "
        f"offload_empty_cache={bool(args.stage_module_offload_empty_cache)}, "
        f"tf32={bool(args.allow_tf32)}, cudnn_benchmark={bool(args.cudnn_benchmark)}"
    )
    _log(
        "Feature metric: "
        f"score_threshold={float(args.score_threshold):.4f}, "
        f"w_topk={float(args.score_w_topk):.3f}, w_cov={float(args.score_w_cov):.3f}, w_mean={float(args.score_w_mean):.3f}"
    )
    _log(
        "Label embeddings: "
        f"enabled={1 if bool(args.label_embeddings_enabled) else 0}, "
        f"backend={str(args.label_embedding_backend)}, "
        f"model={str(args.label_embedding_model)}, "
        f"dim={int(args.label_embedding_dim)}, "
        f"temp={float(args.label_embedding_temperature):.2f}"
    )
    if str(args.semantic_vocab_extra_texts).strip() or str(args.semantic_vocab_extra_json).strip():
        _log(
            "Semantic vocab extras: "
            f"texts={'set' if str(args.semantic_vocab_extra_texts).strip() else 'none'}, "
            f"json={str(args.semantic_vocab_extra_json).strip() or 'none'}"
        )
    _log(
        "Semantic vocab conditioning: "
        f"text_temp={float(args.semantic_vocab_text_condition_temperature):.2f}, "
        f"text_topk={int(args.semantic_vocab_text_condition_topk)}, "
        f"unknown_rows_per_cycle={int(args.semantic_vocab_unknown_label_rows_per_cycle)}, "
        f"bootstrap_origin='{str(args.semantic_vocab_bootstrap_origin_label)}'"
    )
    _log(
        "Target label knockout: "
        f"prob={float(args.target_label_knockout_prob):.3f}, "
        f"max_drop_frac={float(args.target_label_knockout_max_drop_frac):.3f}, "
        f"min_keep={int(args.target_label_knockout_min_keep)}"
    )
    _log(
        "Semantic vocab churn: "
        f"enabled={1 if bool(args.semantic_vocab_churn_enabled) else 0}, "
        f"replace_per_cycle={int(args.semantic_vocab_churn_replace_per_cycle)}, "
        f"sweep_cycles={int(args.semantic_vocab_churn_sweep_cycles)}"
    )
    _log(
        "Model capacity: "
        f"classifier(base={int(args.classifier_base_ch)},max={int(args.classifier_max_ch)},ctx={int(args.classifier_context_blocks)}), "
        f"transformer(d_model={int(args.transformer_d_model)},heads={int(args.transformer_nhead)},layers={int(args.transformer_num_layers)}), "
        f"generator(base={int(args.generator_base_ch)},depth={int(args.generator_depth)}), "
        f"discriminator(base={int(args.discriminator_base_ch)},depth={int(args.discriminator_depth)},max={int(args.discriminator_max_ch)})"
    )
    _log(
        "Transformer stage batch policy: "
        f"base={max(1, int(args.transformer_batch_size))}, "
        f"round_override={max(0, int(args.transformer_round_batch_size))}, "
        f"joint_override={max(0, int(args.joint_transformer_batch_size))}, "
        f"eval_override={max(0, int(args.transformer_eval_batch_size))}"
    )
    _log(
        "Transformer objective: signal(target+rank+spurious) + entropy + high-bit + low-bit + waveform-L1 "
        f"(w_score_target={float(args.transformer_loss_score_target_weight):.3f}, "
        f"score_margin={float(args.transformer_loss_score_target_margin):.3f}, "
        f"w_rank={float(args.transformer_loss_score_rank_weight):.3f}, "
        f"rank_margin={float(args.transformer_loss_score_rank_margin):.3f}, "
        f"w_spurious={float(args.transformer_loss_score_spurious_weight):.3f}, "
        f"spurious_thr={float(args.transformer_loss_score_spurious_threshold):.3f}, "
        f"spurious_temp={float(args.transformer_loss_score_spurious_temp):.3f}, "
        f"spurious_margin={float(args.transformer_loss_score_spurious_margin):.3f}, "
        f"w_entropy={float(args.transformer_loss_entropy_weight):.3f}, "
        f"w_hi={float(args.transformer_loss_high_bit_weight):.3f}, "
        f"w_wave={float(args.transformer_loss_wave_l1_weight):.3f}, "
        f"w_lo={float(args.transformer_loss_low_bit_weight):.3f})"
    )
    _log(
        "Degrade curriculum: "
        f"enabled={1 if bool(args.transformer_degrade_inputs) else 0}, "
        f"domain={str(args.transformer_degrade_domain)}, "
        f"mode={str(args.transformer_degrade_mode)}, "
        f"strength=[{float(args.transformer_degrade_min_strength):.3f},{float(args.transformer_degrade_max_strength):.3f}], "
        f"noise_std=[{float(args.transformer_degrade_noise_std_min):.3f},{float(args.transformer_degrade_noise_std_max):.3f}], "
        f"dropout_max={float(args.transformer_degrade_dropout_max):.3f}, "
        f"quant_bits=[{int(args.transformer_degrade_quant_bits_min)},{int(args.transformer_degrade_quant_bits_max)}], "
        f"stride_skew_max={float(args.transformer_degrade_stride_skew_max):.3f}, "
        f"vec_window={max(1, int(args.transformer_degrade_vectorized_window))}"
    )
    _log(
        "Transformer aux filters: "
        f"bundles={','.join(transformer_filter_bundle_names) if len(transformer_filter_bundle_names) > 0 else 'none'}, "
        f"deskew_max_skew={float(args.transformer_deskew_prefilter_max_skew):.3f}, "
        f"deskew_w_pre={float(args.transformer_deskew_prefilter_weight):.3f}, "
        f"deskew_w_post={float(args.transformer_deskew_residual_weight):.3f}, "
        f"deskew_mix_pre={float(args.transformer_deskew_pre_token_mix):.3f}, "
        f"deskew_mix_post={float(args.transformer_deskew_post_token_mix):.3f}, "
        f"deskew_token_l2={float(args.transformer_deskew_token_residual_weight):.4f}"
    )
    _log(
        "Latent Berkeley imprint: "
        f"enabled={1 if bool(args.latent_berkeley_imprint_mix_targets) else 0}, "
        f"steps={int(args.latent_berkeley_imprint_steps)}, "
        f"lr={float(args.latent_berkeley_imprint_lr):.4f}, "
        f"l2={float(args.latent_berkeley_imprint_l2):.4f}, "
        f"targets_per_mix=[{int(args.latent_berkeley_imprint_min_target_classes)},{int(args.latent_berkeley_imprint_max_target_classes)}]"
    )
    _log(
        "Transformer status viz: "
        f"enabled={1 if bool(args.transformer_visualize_status) else 0}, "
        f"scale={max(1, int(args.transformer_viz_scale))}"
    )
    _log(
        "Generator stage: "
        f"z_dim={int(args.generator_z_dim)}, batch={int(args.generator_batch_size)}, "
        f"steps={int(args.generator_steps_per_round)}, grad_accum={max(1, int(args.generator_grad_accum_steps))}, "
        f"cls_batch_cap={max(0, int(args.generator_classifier_batch_cap))}, "
        f"lr_g={float(args.generator_lr):.5f}, "
        f"lr_d={float(args.discriminator_lr):.5f}, gate_target_prob={float(args.generator_gate_min_target_prob):.4f}, "
        f"d_steps_per_g={max(1, int(args.discriminator_steps_per_generator_step))}, "
        f"gate_rounds={max(1, int(args.generator_gate_maintain_rounds))}, "
        f"w_adv={float(args.generator_loss_adv_weight):.3f}, "
        f"w_cls={float(args.generator_loss_cls_weight):.3f}, "
        f"w_wave={float(args.generator_loss_wave_weight):.3f}, "
        f"log_every={max(0, int(args.generator_log_every))}"
    )
    _log(
        "Generator fake feedback: "
        f"enabled={1 if bool(args.generator_fake_feedback_enabled) else 0}, "
        f"text='{str(args.fake_image_sentinel_label)}', "
        f"epochs={max(1, int(args.generator_fake_feedback_epochs))}, "
        f"steps={max(1, int(args.generator_fake_feedback_steps_per_round))}, "
        f"batch={max(1, int(args.generator_fake_feedback_batch_size))}, "
        f"lr={float(args.generator_fake_feedback_lr):.5f}, "
        f"w_vec={float(args.generator_fake_feedback_vector_weight):.3f}, "
        f"w_cond={float(args.generator_fake_feedback_condition_weight):.3f}, "
        f"disc_conf_temp={float(args.generator_fake_feedback_disc_conf_temperature):.3f}, "
        f"disc_conf_floor={float(args.generator_fake_feedback_disc_conf_floor):.3f}, "
        f"disc_balance={1 if bool(args.generator_fake_feedback_disc_balance_groups) else 0}, "
        f"include_targets={1 if bool(args.generator_fake_feedback_include_condition_targets) else 0}"
    )
    _log(
        "Stage OpenGL viz: "
        f"enabled={1 if bool(args.stage_opengl_preview_enabled) else 0}, "
        f"scale={max(1, int(args.stage_opengl_preview_scale))}, "
        f"every_round={max(1, int(args.stage_opengl_preview_every_round))}, "
        f"bootstrap={1 if bool(args.stage_opengl_preview_bootstrap) else 0}, "
        f"required={1 if bool(args.stage_opengl_preview_required) else 0}"
    )
    if bool(args.stage_opengl_preview_enabled):
        _log("Stage OpenGL viewer is active as the sole preview window (transformer status viewer disabled).")
        _log("Stage OpenGL reporting: one_example_per_stage=1, rotating_samples=1, labels_overlay=1")
    _log(
        "Joint stage: "
        f"enabled={1 if bool(args.joint_enabled) else 0}, "
        f"G_steps={int(args.joint_generator_steps_per_round)}, "
        f"R_steps={int(args.joint_transformer_steps_per_round)}, "
        f"wave_margin={float(args.generator_wave_score_margin):.3f}, "
        f"wave_denoise_w={float(args.generator_wave_denoise_weight):.3f}, "
        f"wave_blend={float(args.generator_wave_carrier_blend):.3f}, "
        f"wave_strength={float(args.generator_wave_strength):.3f}, "
        f"wave_mode={str(args.generator_wave_degrade_mode)}, "
        f"wave_stride_skew_max={float(args.generator_wave_stride_skew_max):.3f}"
    )
    _log(
        "Checkpoint policy: "
        f"every_round={int(args.checkpoint_every_round)}, "
        f"after_training_segment={1 if bool(args.checkpoint_after_training_segment) else 0}"
    )
    shared_embed_image_size = _resolve_shared_embed_image_size(args)
    image_hw_shared = (int(shared_embed_image_size), int(shared_embed_image_size))
    _log(
        "Embed resolution sync: "
        f"resolved={int(shared_embed_image_size)} "
        f"(image_size={int(args.image_size)}, berkeley_image_size={int(args.berkeley_image_size)})"
    )

    t0 = time.time()
    latent_fallback_info = None
    wav_data_root = str(args.wav_root).strip()
    wav_paths: List[str] = []

    if wav_data_root:
        wav_paths = discover_wavs(wav_data_root)
        if len(wav_paths) == 0:
            _log(f"No wav files found under: {wav_data_root}; switching to latent fallback.")

    if len(wav_paths) == 0:
        latent_fallback_info = _bootstrap_latent_wav_pool(
            out_dir=out_dir,
            seed=args.seed,
            count=max(8, int(args.latent_fallback_count)),
            framerate=args.latent_fallback_rate,
            seconds=args.latent_fallback_seconds,
            noise_std=args.latent_noise_std,
            reinject_dir=args.latent_reinject_dir,
            reinject_ratio=max(0.0, min(1.0, float(args.latent_reinject_ratio))),
            reinject_copy_gain=float(args.latent_reinject_copy_gain),
            reinject_noise_gain=float(args.latent_reinject_noise_gain),
            structured_ratio=max(0.0, min(1.0, float(args.latent_structured_ratio))),
            structured_gain=float(args.latent_structured_gain),
            structured_noise_gain=float(args.latent_structured_noise_gain),
        )
        wav_data_root = str(latent_fallback_info["pool_dir"])
        wav_paths = discover_wavs(wav_data_root)
        _log(
            "Latent fallback active: "
            f"root={wav_data_root}, wavs={len(wav_paths)}, mixed={latent_fallback_info['mixed_count']}, "
            f"structured={latent_fallback_info.get('structured_count', 0)}, "
            f"reinject_candidates={latent_fallback_info['reinject_candidates']}"
        )

    if len(wav_paths) == 0:
        raise RuntimeError("No wav files available after latent fallback bootstrap.")
    if args.max_files > 0:
        wav_paths = wav_paths[: args.max_files]
    _log(f"Loading {len(wav_paths)} wav files from: {wav_data_root}")

    records: List[WaveRecord] = []
    for p in wav_paths:
        try:
            records.append(read_wav_record(p))
        except Exception as e:
            _log(f"[skip] {p} ({e})")
    if len(records) < 8:
        shortfall = max(0, 8 - int(len(records)))
        _log(
            "Readable WAV pool is undersized; auto-supplementing from latent fallback: "
            f"have={len(records)} need=8 shortfall={int(shortfall)}"
        )
        latent_topup = _bootstrap_latent_wav_pool(
            out_dir=out_dir,
            seed=args.seed + 997,
            count=max(int(args.latent_fallback_count), int(shortfall)),
            framerate=args.latent_fallback_rate,
            seconds=args.latent_fallback_seconds,
            noise_std=args.latent_noise_std,
            reinject_dir=args.latent_reinject_dir,
            reinject_ratio=max(0.0, min(1.0, float(args.latent_reinject_ratio))),
            reinject_copy_gain=float(args.latent_reinject_copy_gain),
            reinject_noise_gain=float(args.latent_reinject_noise_gain),
            structured_ratio=max(0.0, min(1.0, float(args.latent_structured_ratio))),
            structured_gain=float(args.latent_structured_gain),
            structured_noise_gain=float(args.latent_structured_noise_gain),
        )
        topup_paths = discover_wavs(str(latent_topup.get("pool_dir", "")))
        seen_paths = {str(Path(r.path).resolve()).lower() for r in records}
        for p in topup_paths:
            if int(len(records)) >= 8:
                break
            rp = str(Path(str(p)).resolve()).lower()
            if rp in seen_paths:
                continue
            try:
                records.append(read_wav_record(p))
                seen_paths.add(rp)
            except Exception:
                continue
        if len(records) < 8:
            raise RuntimeError("Need at least 8 readable WAV files after latent top-up fallback.")

    resume_enabled = bool(args.auto_resume or str(args.resume_from).strip())
    resume_dir = Path(str(args.resume_from).strip()) if str(args.resume_from).strip() else out_dir
    resume_summary_path = resume_dir / "summary.json"
    resume_cfg_path = resume_dir / "best_render_config.json"
    resume_search_path = resume_dir / "search_results.json"
    resume_classifier_path = resume_dir / "classifier.pt"
    resume_transformer_path = resume_dir / "transformer.pt"
    resume_deskew_prefilter_path = resume_dir / "deskew_prefilter.pt"
    resume_wave_classifier_path = resume_dir / "wave_classifier.pt"
    resume_generator_path = resume_dir / "generator.pt"
    resume_discriminator_path = resume_dir / "discriminator.pt"
    resume_pipeline_ckpt_path = resume_dir / "pipeline_checkpoint.pt"

    resume_summary = _load_json(resume_summary_path) if resume_enabled else None
    resume_cfg = _load_json(resume_cfg_path) if resume_enabled else None
    resume_pipeline_ckpt = None
    if resume_enabled and resume_pipeline_ckpt_path.exists():
        try:
            resume_pipeline_ckpt = _torch_load_cpu(str(resume_pipeline_ckpt_path))
        except Exception:
            resume_pipeline_ckpt = None

    if resume_enabled:
        _log(
            "Resume mode: "
            f"dir={resume_dir}, summary={resume_summary_path.exists()}, "
            f"cfg={resume_cfg_path.exists()}, ckpt={resume_pipeline_ckpt_path.exists()}"
        )

    run_tag = time.strftime("%Y%m%d_%H%M%S")

    if args.objective_mode == "berkeley_multilabel":
        _log("Objective mode: berkeley_multilabel")
        label_mode = "folder" if args.label_mode == "folder" else "spectral"
        labels_for_split, split_class_names, label_source = _build_labels(
            records=records,
            data_root=wav_data_root,
            label_mode=label_mode,
            pseudo_classes=args.pseudo_classes,
            single_class_fallback="path_hash_quantile",
        )
        split_labels_from_pseudo_spectral = bool(str(label_source) in ("spectral_quantile", "spectral_cosine_quantile"))
        split_num_classes = len(set(labels_for_split.tolist()))
        if split_num_classes < 2:
            raise RuntimeError("Need at least 2 classes for data splitting.")
        if len(split_class_names) != split_num_classes:
            split_class_names = [f"wave_cls_{i}" for i in range(split_num_classes)]
        split_mix_class_index = _find_semantic_term_index(split_class_names, "mix")
        split_label_embedding_bank = None
        split_label_texts = [str(x) for x in split_class_names]
        split_label_embedding_info: Dict[str, Any] = {
            "enabled": False,
            "backend_requested": str(args.label_embedding_backend),
            "backend_used": "disabled",
            "model_name": str(args.label_embedding_model),
            "num_classes": int(split_num_classes),
            "native_dim": 0,
            "dim": 0,
            "temperature": float(args.label_embedding_temperature),
            "fallback_reason": "",
        }
        if bool(args.label_embeddings_enabled):
            split_label_embedding_bank, split_label_texts, split_label_embedding_info = _build_label_embedding_bank(
                class_names=split_class_names,
                args=args,
                device=device,
            )
        wave_zero_shot_query_source = "none"
        wave_zero_shot_single_probe_once = False
        wave_zero_shot_queries = _parse_label_query_texts(str(args.wave_zero_shot_query_texts))
        if len(wave_zero_shot_queries) <= 0:
            wave_zero_shot_queries = _parse_label_query_texts(str(args.label_query_texts))
            if len(wave_zero_shot_queries) > 0:
                wave_zero_shot_query_source = "label_query_texts"
        else:
            wave_zero_shot_query_source = "wave_zero_shot_query_texts"
        if len(wave_zero_shot_queries) <= 0:
            if bool(split_labels_from_pseudo_spectral):
                probe = [str(x) for x in split_label_texts if str(x).strip()]
                if len(probe) > 0:
                    wave_zero_shot_queries = [str(probe[0])]
                    wave_zero_shot_query_source = "split_label_texts_single_probe"
                    wave_zero_shot_single_probe_once = True
                    _log("[wave-zero-shot] pseudo split labels detected; running one blind label-text probe.")
            else:
                wave_zero_shot_queries = [str(x) for x in split_label_texts if str(x).strip()]
                wave_zero_shot_query_source = "split_label_texts_default"
        wave_zero_shot_query_emb = None
        wave_zero_shot_query_rows: List[Dict[str, Any]] = []
        if len(wave_zero_shot_queries) > 0 and bool(args.label_embeddings_enabled):
            wave_zero_shot_query_emb = _encode_query_texts_for_bank(
                query_texts=wave_zero_shot_queries,
                bank_info=split_label_embedding_info,
                args=args,
                device=device,
            )
            if split_label_embedding_bank is not None:
                wave_zero_shot_query_rows = _nearest_labels_for_queries(
                    query_texts=wave_zero_shot_queries,
                    label_bank_np=split_label_embedding_bank,
                    class_names=split_class_names,
                    label_texts=split_label_texts,
                    topk=max(1, int(args.wave_zero_shot_topk)),
                    query_emb_np=wave_zero_shot_query_emb,
                )
            _log(
                "[wave-zero-shot] query bank ready: "
                f"queries={len(wave_zero_shot_queries)}, "
                f"dim={int(wave_zero_shot_query_emb.shape[1])}, "
                f"source={wave_zero_shot_query_source}"
            )
        elif len(wave_zero_shot_queries) > 0:
            _log(
                "[wave-zero-shot] disabled: label embeddings are not active, "
                f"query_source={wave_zero_shot_query_source}"
            )
        if float(args.wave_cls_semantic_mix_noise_prob) > 0.0:
            _log(
                "[wave-cls] semantic mix augmentation: "
                f"prob={float(args.wave_cls_semantic_mix_noise_prob):.3f} "
                f"noise_std={float(args.wave_cls_semantic_mix_noise_std):.3f} "
                f"blend=[{float(args.wave_cls_semantic_mix_blend_min):.3f},{float(args.wave_cls_semantic_mix_blend_max):.3f}] "
                f"mix_label_index={int(split_mix_class_index)}"
            )
        train_idx, val_idx = _stratified_split(labels_for_split, train_frac=args.train_frac, seed=args.seed)
        _log(f"Split: train={len(train_idx)} val={len(val_idx)} (source={label_source})")

        search_train_idx = _sample_indices(train_idx, args.search_train_limit, rng)
        search_val_idx = _sample_indices(val_idx, args.search_val_limit, rng)
        final_train_idx = _sample_indices(train_idx, args.final_train_limit, rng)
        final_val_idx = _sample_indices(val_idx, args.final_val_limit, rng)
        search_indices = sorted(set(search_train_idx + search_val_idx))
        fresh_round_metric_subset = (
            bool(args.classifier_subset_fresh_round_sampling)
            and int(args.final_val_limit) > 0
            and len(val_idx) > int(args.final_val_limit)
        )
        fresh_round_metric_rng = np.random.default_rng(int(args.seed) + 49037)

        def _sample_round_classifier_val_indices() -> List[int]:
            if bool(fresh_round_metric_subset):
                return _sample_indices(val_idx, int(args.final_val_limit), fresh_round_metric_rng)
            return list(final_val_idx)

        if bool(fresh_round_metric_subset):
            _log(
                "Classifier subset freshness: "
                f"enabled=1 limit={int(args.final_val_limit)} val_total={len(val_idx)} resample=per_round"
            )

        image_hw = image_hw_shared
        candidates = _random_configs(num_trials=args.config_trials, rng=rng, max_points=args.render_max_points)

        classifier, classifier_init_info = _load_berkeley_classifier_for_metric(
            args.classifier_init_ckpt,
            device=device,
            classifier_base_ch=int(args.classifier_base_ch),
            classifier_max_ch=int(args.classifier_max_ch),
            classifier_context_blocks=int(args.classifier_context_blocks),
            classifier_context_dropout=float(args.classifier_context_dropout),
            classifier_init_required=bool(args.classifier_init_required),
        )
        if str(classifier_init_info.get("init_mode", "checkpoint")) != "checkpoint":
            _log(
                "[classifier-init] scratch fallback active: "
                f"reason={classifier_init_info.get('fallback_reason', 'unknown')} "
                f"classes={int(classifier_init_info.get('num_classes', 0))}"
            )
        else:
            _log(
                "[classifier-init] loaded checkpoint: "
                f"path={classifier_init_info.get('path', '')} "
                f"classes={int(classifier_init_info.get('num_classes', 0))} "
                f"loaded={int(classifier_init_info.get('loaded_keys', 0))} "
                f"partial={int(classifier_init_info.get('partial_keys', 0))} "
                f"skipped={int(classifier_init_info.get('skipped_keys', 0))}"
            )
        supervised_num_classes = int(classifier_init_info["num_classes"])
        supervised_class_names = list(classifier_init_info.get("class_names", []))
        if len(supervised_class_names) != supervised_num_classes:
            supervised_class_names = [f"berkeley_cls_{i}" for i in range(supervised_num_classes)]
        if not bool(args.label_embeddings_enabled):
            raise RuntimeError(
                "Hard semantic mode requires --label-embeddings in berkeley_multilabel objective."
            )
        semantic_extra_terms = _parse_label_query_texts(str(args.semantic_vocab_extra_texts))
        if str(args.semantic_vocab_extra_json).strip():
            semantic_extra_terms.extend(_load_vocab_terms_json(str(args.semantic_vocab_extra_json)))
        semantic_extra_terms = _normalize_vocab_terms(semantic_extra_terms)
        base_lc = {str(x).strip().lower() for x in supervised_class_names}
        semantic_core_terms = [
            t for t in _normalize_vocab_terms(_default_semantic_core_terms()) if str(t).strip().lower() not in base_lc
        ]
        semantic_extra_terms = [t for t in semantic_extra_terms if str(t).strip().lower() not in base_lc]
        semantic_extra_terms = _normalize_active_extra_terms_with_core(
            active_terms=semantic_extra_terms,
            core_terms=semantic_core_terms,
            total_slots=max(
                max(0, int(args.semantic_vocab_extra_slots)),
                int(len(semantic_extra_terms) + len(semantic_core_terms)),
            ),
        )
        semantic_core_extra_count = int(len(semantic_core_terms))
        semantic_class_names = list(supervised_class_names) + list(semantic_extra_terms)
        semantic_num_classes = max(1, int(len(semantic_class_names)))
        semantic_extra_count = int(max(0, int(semantic_num_classes - supervised_num_classes)))
        if semantic_extra_count > 0:
            _log(
                "[semantic-vocab] expanded classifier vocabulary: "
                f"supervised={supervised_num_classes} total={semantic_num_classes} extras={semantic_extra_count}"
            )
        if int(semantic_core_extra_count) > 0:
            _log(
                "[semantic-vocab] enforced core semantic terms: "
                f"{','.join(semantic_core_terms)} (locked={int(semantic_core_extra_count)})"
            )
        if int(args.semantic_vocab_extra_slots) > int(len(_normalize_vocab_terms(_parse_label_query_texts(str(args.semantic_vocab_extra_texts))))):
            _log(
                "[semantic-vocab] reserved extra slots: "
                f"slots={int(args.semantic_vocab_extra_slots)} seed_terms={int(len(semantic_extra_terms))}"
            )
        class_names = list(semantic_class_names)
        condition_num_classes = int(semantic_num_classes)
        num_classes = int(condition_num_classes)
        orchestration_mode = str(args.orchestration_mode).strip().lower()
        fake_feedback_enabled = bool(args.generator_fake_feedback_enabled) and (orchestration_mode in ("staged_cgr", "staged_cgrw"))
        fake_sentinel_label = str(args.fake_image_sentinel_label).strip() or "GAN image"
        fake_label_vector_t: Optional[torch.Tensor] = None
        score_active_classes = int(supervised_num_classes)
        label_embedding_bank = None
        label_texts = [str(x) for x in class_names]
        label_embedding_info: Dict[str, Any] = {
            "enabled": False,
            "backend_requested": str(args.label_embedding_backend),
            "backend_used": "disabled",
            "model_name": str(args.label_embedding_model),
            "num_classes": int(num_classes),
            "native_dim": 0,
            "dim": 0,
            "temperature": float(args.label_embedding_temperature),
            "fallback_reason": "",
        }
        label_query_rows: List[Dict[str, Any]] = []
        label_embedding_bank, label_texts, label_embedding_info = _build_label_embedding_bank(
            class_names=semantic_class_names,
            args=args,
            device=device,
        )
        label_embedding_info["supervised_num_classes"] = int(supervised_num_classes)
        label_embedding_info["semantic_num_classes"] = int(len(semantic_class_names))
        label_embedding_info["semantic_extra_count"] = int(semantic_extra_count)
        _semantic_text_condition_cache: Dict[Tuple[str, ...], np.ndarray] = {}
        _semantic_supervised_condition_cache: Dict[Tuple[int, ...], np.ndarray] = {}
        semantic_term_to_idx: Dict[str, int] = _semantic_term_index_map(class_names)
        _semantic_bank_norm_cache_key: Tuple[int, int, int] = (-1, -1, -1)
        _semantic_bank_norm_cache: Optional[np.ndarray] = None

        def _semantic_bank_norm_runtime() -> Optional[np.ndarray]:
            nonlocal _semantic_bank_norm_cache_key, _semantic_bank_norm_cache
            bank_np = np.asarray(label_embedding_bank, dtype=np.float32) if label_embedding_bank is not None else None
            if bank_np is None or int(bank_np.ndim) != 2:
                _semantic_bank_norm_cache_key = (-1, -1, -1)
                _semantic_bank_norm_cache = None
                return None
            rows = min(int(condition_num_classes), int(bank_np.shape[0]))
            cols = int(bank_np.shape[1]) if int(bank_np.ndim) == 2 else 0
            key = (int(rows), int(cols), int(bank_np.__array_interface__.get("data", (0,))[0]))
            if key != _semantic_bank_norm_cache_key:
                if rows <= 0 or cols <= 0:
                    _semantic_bank_norm_cache = None
                else:
                    _semantic_bank_norm_cache = _normalize_l2_rows_np(bank_np[: int(rows), :].astype(np.float32, copy=False))
                _semantic_bank_norm_cache_key = key
            return _semantic_bank_norm_cache

        def _semantic_condition_from_supervised_runtime(supervised_vec: Optional[np.ndarray]) -> np.ndarray:
            c = max(1, int(condition_num_classes))
            sup = max(1, min(int(supervised_num_classes), int(c)))
            zero = np.zeros((int(c),), dtype=np.float32)
            if supervised_vec is None:
                return zero
            arr = np.asarray(supervised_vec, dtype=np.float32).reshape(-1)
            if int(arr.size) <= 0:
                return zero
            arr_sup = np.zeros((int(sup),), dtype=np.float32)
            take = min(int(sup), int(arr.size))
            arr_sup[: int(take)] = np.clip(arr[: int(take)], 0.0, 1.0)
            key = tuple(np.where(arr_sup > 0.0)[0].astype(np.int64).tolist())
            if key in _semantic_supervised_condition_cache:
                return np.array(_semantic_supervised_condition_cache[key], dtype=np.float32, copy=True)
            out = np.zeros((int(c),), dtype=np.float32)
            out[: int(sup)] = np.clip(arr_sup, 0.0, 1.0)
            _semantic_supervised_condition_cache[key] = np.array(out, dtype=np.float32, copy=True)
            return np.array(out, dtype=np.float32, copy=True)

        def _semantic_condition_from_terms_runtime(
            terms: Sequence[str],
            base_supervised_vec: Optional[np.ndarray] = None,
            origin_terms: Optional[Sequence[str]] = None,
        ) -> np.ndarray:
            c = max(1, int(condition_num_classes))
            out = np.zeros((int(c),), dtype=np.float32)
            out = np.maximum(out, _semantic_condition_from_supervised_runtime(base_supervised_vec)).astype(np.float32, copy=False)
            # Presence labels are term-membership only. Origin metadata is intentionally excluded.
            clean_terms = _semantic_expand_inferred_tags([str(x) for x in list(terms)])
            if len(clean_terms) <= 0:
                return out

            exact_match_found = False
            for term in clean_terms:
                idx = int(semantic_term_to_idx.get(str(term).strip().lower(), -1))
                if 0 <= int(idx) < int(c):
                    out[int(idx)] = 1.0
                    exact_match_found = True
            if not bool(exact_match_found):
                alias_terms = _resolve_core_semantic_aliases(clean_terms)
                for term in alias_terms:
                    idx = int(semantic_term_to_idx.get(str(term).strip().lower(), -1))
                    if 0 <= int(idx) < int(c):
                        out[int(idx)] = 1.0
            return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

        semantic_kind_keys = _semantic_kind_keys_for_class_names(semantic_class_names)
        semantic_kind_target_vectors = {}
        for kind_key in semantic_kind_keys:
            kk = re.sub(r"\s+", " ", str(kind_key)).strip().lower()
            if kk in ("mix", "mixed noise and signal"):
                terms = ["mixed noise and signal", "mix", "signal", "noise"]
            elif kk == "regurgitated content":
                terms = ["regurgitated content", "mixed noise and signal", "signal"]
            else:
                terms = [str(kind_key)]
            semantic_kind_target_vectors[str(kind_key)] = _semantic_condition_from_terms_runtime(terms=terms)
        mix_condition_target_default = _semantic_condition_from_terms_runtime(
            terms=["mixed noise and signal", "signal", "noise"],
        )
        regurgitated_condition_target_default = _semantic_condition_from_terms_runtime(
            terms=["regurgitated content", "mixed noise and signal", "signal"],
        )
        _log(
            "[semantic-vocab] core kind vectors: "
            f"none={1 if 'none' in semantic_kind_target_vectors else 0} "
            f"noise={1 if 'noise' in semantic_kind_target_vectors else 0} "
            f"mix={1 if 'mix' in semantic_kind_target_vectors else 0} "
            f"signal={1 if 'signal' in semantic_kind_target_vectors else 0} "
            f"object={1 if 'object' in semantic_kind_target_vectors else 0}"
        )
        _log(
            "[label-embeddings] bank ready: "
            f"supervised={int(supervised_num_classes)} "
            f"semantic={int(label_embedding_info.get('num_classes', 0))} "
            f"dim={int(label_embedding_info.get('dim', 0))} "
            f"backend={label_embedding_info.get('backend_used', 'unknown')}"
        )
        active_extra_terms: List[str] = list(semantic_extra_terms)
        # Keep wave split labels isolated from semantic training churn.
        semantic_vocab_pool_terms: List[str] = _normalize_vocab_terms(
            list(active_extra_terms)
            + [
                str(fake_sentinel_label),
                "regurgitated content",
                "berkeley sbd dataset",
                "mnist dataset",
                "emnist dataset",
                "kmnist dataset",
            ]
        )
        semantic_vocab_pool_terms = [
            t for t in semantic_vocab_pool_terms if str(t).strip().lower() not in set(base_lc)
        ]
        semantic_churn_history: List[Dict[str, Any]] = []
        semantic_vocab_churn_cursor = 0
        semantic_core_lc = {str(x).strip().lower() for x in semantic_core_terms}
        regurgitated_churn_pool: List[Dict[str, Any]] = []
        regurgitated_churn_seen_paths: set = set()
        regurgitated_churn_seq = 0
        regurgitated_churn_last_info: Dict[str, Any] = {
            "enabled": bool(args.semantic_vocab_regurgitated_churn_enabled),
            "triggered": False,
            "pool_before": 0,
            "pool_after": 0,
            "used_items": 0,
            "expired_items": 0,
            "terms_added": 0,
            "reason": "startup",
        }
        gan_live_churn_pool: List[Dict[str, Any]] = []
        gan_live_churn_seq = 0
        gan_live_churn_last_info: Dict[str, Any] = {
            "enabled": True,
            "triggered": False,
            "harvested": 0,
            "used_items": 0,
            "expired_items": 0,
            "pool_before": 0,
            "pool_after": 0,
            "reason": "startup",
        }
        symbol_pool_by_term: Dict[str, List[np.ndarray]] = {}
        symbol_pool_origin_terms_by_term: Dict[str, List[str]] = {}
        symbol_pool_info: Dict[str, Any] = {"enabled": False, "available_terms": 0, "samples": 0}
        bootstrap_symbol_terms: set = set()
        gestation_symbol_terms: set = set()
        payload_images_base: List[np.ndarray] = []
        payload_conditions_supervised_base: List[np.ndarray] = []
        payload_condition_terms_supervised_base: List[List[str]] = []
        payload_symbol_aug_info: Dict[str, Any] = {"enabled": False, "rows_added": 0}
        payload_flashcard_info: Dict[str, Any] = {"enabled": False, "rows_added": 0}
        gd_vocab_library_dir = (
            Path(str(args.gd_vocab_library_dir).strip())
            if str(args.gd_vocab_library_dir).strip()
            else (out_dir / "gd_vocab_library")
        )
        active_gd_vocab_hash = ""
        active_gd_vocab_profile: Dict[str, Any] = {}
        query_texts = _parse_label_query_texts(str(args.label_query_texts))
        if len(query_texts) > 0:
            q_emb = _encode_query_texts_for_bank(
                query_texts=query_texts,
                bank_info=label_embedding_info,
                args=args,
                device=device,
            )
            label_query_rows = _nearest_labels_for_queries(
                query_texts=query_texts,
                label_bank_np=label_embedding_bank,
                class_names=semantic_class_names,
                label_texts=label_texts,
                topk=max(1, int(args.label_query_topk)),
                query_emb_np=q_emb,
            )
            for row in label_query_rows:
                if len(row.get("matches", [])) <= 0:
                    continue
                best = row["matches"][0]
                _log(
                    f"[label-query] '{row.get('query', '')}' -> "
                    f"{best.get('class_name', '')} ({best.get('score', 0.0):.4f})"
                )
        classifier_resume_info = {"used": False}
        if resume_enabled and resume_classifier_path.exists():
            classifier_resume_info = _apply_model_init(classifier, str(resume_classifier_path))
            _log(
                f"Resumed Berkeley classifier from {resume_classifier_path} "
                f"(loaded={classifier_resume_info.get('loaded_keys', 0)}, skipped={classifier_resume_info.get('skipped_keys', 0)})"
            )
        label_embed_apply_info = _apply_label_embedding_bank_to_classifier(
            classifier=classifier,
            bank_np=label_embedding_bank,
            args=args,
        )
        if bool(label_embed_apply_info.get("applied", False)):
            _log(
                "[label-embeddings] classifier active: "
                f"classes={label_embed_apply_info.get('classes', 0)} "
                f"dim={label_embed_apply_info.get('dim', 0)} "
                f"temp={float(label_embed_apply_info.get('temperature', 0.0)):.2f}"
            )
        if fake_feedback_enabled:
            if not bool(label_embed_apply_info.get("applied", False)):
                _log(
                    "Generator fake feedback disabled: classifier label embeddings are not active "
                    "(enable --label-embeddings and use TinyConvClassifier)."
                )
                fake_feedback_enabled = False
            else:
                fake_vec_np = _encode_query_texts_for_bank(
                    query_texts=[str(fake_sentinel_label)],
                    bank_info=label_embedding_info,
                    args=args,
                    device=device,
                )
                if int(fake_vec_np.shape[0]) <= 0:
                    _log("Generator fake feedback disabled: could not encode fake-sentinel text vector.")
                    fake_feedback_enabled = False
                else:
                    fake_label_vector_t = torch.from_numpy(fake_vec_np[0]).to(device=device, dtype=torch.float32)
                    _log(
                        "Generator fake feedback vector ready: "
                        f"text='{fake_sentinel_label}', dim={int(fake_label_vector_t.numel())}, "
                        f"backend={label_embedding_info.get('backend_used', 'unknown')}"
                    )
        if args.channels_last:
            classifier = classifier.to(memory_format=torch.channels_last)
        classifier = maybe_compile_module(
            classifier,
            enabled=bool(args.compile_models),
            mode=str(args.compile_mode),
        )

        refresh_loader = None
        refresh_cache = None
        refresh_count = 0
        refresh_cache_samples = 0
        refresh_run_batch_size = max(1, int(args.berkeley_refresh_batch_size))
        refresh_rows: List[Dict] = []
        berkeley_refresh_required_samples = 0
        refresh_init_configured = bool(int(args.berkeley_refresh_epochs) > 0)
        gate_external_val_fraction = 0.20
        gate_split_seed = int(args.seed) + 7089
        refresh_init_done = False

        def _init_refresh_resources(reason: str, samples_seen_for_startup: int):
            nonlocal refresh_loader, refresh_cache, refresh_count
            nonlocal refresh_cache_samples, refresh_run_batch_size
            nonlocal berkeley_refresh_required_samples, refresh_init_done
            if bool(refresh_init_done):
                return
            if not bool(refresh_init_configured):
                refresh_init_done = True
                return
            refresh_init_done = True
            refresh_loader_batch = (
                max(1, int(args.berkeley_refresh_loader_batch_size))
                if int(args.berkeley_refresh_batch_size) <= 0
                else max(1, int(args.berkeley_refresh_batch_size))
            )
            refresh_loader, refresh_count = _build_berkeley_refresh_loader(
                data_root=args.berkeley_data_root,
                image_size=int(shared_embed_image_size),
                auto_install_scipy=bool(args.berkeley_auto_install_scipy),
                batch_size=refresh_loader_batch,
                num_workers=args.berkeley_refresh_workers,
                max_train=0,
                seed=args.seed + 7300,
                device=device,
                external_val_fraction=float(gate_external_val_fraction),
                validation_split_seed=int(gate_split_seed),
                persistent_workers=bool(args.loader_persistent_workers),
                prefetch_factor=int(args.loader_prefetch_factor),
            )
            _log(
                "Berkeley refresh loader ready: "
                f"reason={str(reason)} "
                "source=semantic_payload_cache "
                f"search_every={args.berkeley_refresh_every}, round_every={args.berkeley_refresh_round_every}, "
                f"epochs={args.berkeley_refresh_epochs}, train={refresh_count}, full_non_validation_deck=1"
            )
            refresh_cache = None
            refresh_cache_samples = 0
            if int(args.berkeley_refresh_cache_batches) > 0:
                refresh_cache = _build_berkeley_refresh_cache(
                    loader=refresh_loader,
                    device=device,
                    cache_batches=int(args.berkeley_refresh_cache_batches),
                    cache_device=str(args.berkeley_refresh_cache_device),
                    channels_last=bool(args.channels_last),
                )
                if isinstance(refresh_cache, dict):
                    refresh_cache_samples = int(max(0, int(refresh_cache.get("num_samples", 0))))
                    _log(
                        "Berkeley refresh cache ready: "
                        f"samples={refresh_cache_samples}, "
                        f"batches={refresh_cache.get('num_batches', 0)}, "
                        f"device={refresh_cache.get('device', 'cpu')}"
                    )
            if int(args.berkeley_refresh_batch_size) <= 0:
                if isinstance(refresh_cache, dict):
                    refresh_run_batch_size = _auto_berkeley_refresh_batch_size(
                        cache_x=refresh_cache.get("x"),
                        cache_y=refresh_cache.get("y"),
                        device=device,
                        vram_fraction=float(args.berkeley_refresh_vram_fraction),
                        activation_multiplier=float(args.berkeley_refresh_activation_mult),
                        max_cap=int(args.berkeley_refresh_max_batch_cap),
                    )
                else:
                    refresh_run_batch_size = int(refresh_loader_batch)
                _log(f"Auto Berkeley refresh batch size: {refresh_run_batch_size}")
            else:
                refresh_run_batch_size = max(1, int(args.berkeley_refresh_batch_size))
            berkeley_refresh_required_samples = max(0, int(refresh_count))
            if berkeley_refresh_required_samples > 0:
                _log(
                    "Berkeley startup requirement: "
                    f"require_full_dataset_once={berkeley_refresh_required_samples} "
                    f"already_seen={int(samples_seen_for_startup)}"
                )
                if refresh_cache_samples > 0 and refresh_cache_samples < berkeley_refresh_required_samples:
                    _log(
                        "Berkeley refresh cache is a subset of train data; "
                        "startup pass will force loader mode until full-dataset requirement is met."
                    )

        berkeley_gate_loader = None
        berkeley_gate_loader_count = 0
        berkeley_gate_schedule_info: Dict[str, Any] = {"rows": 0, "selected_rows": 0}
        gestation_stage_loader = None
        gestation_stage_loader_count = 0
        gestation_stage_info: Dict[str, Any] = {"rows": 0, "selected_rows": 0}
        gestation_gate_loader = None
        gestation_gate_loader_count = 0
        gestation_gate_info: Dict[str, Any] = {"rows": 0, "selected_rows": 0}
        berkeley_loss_gate_enabled = (float(args.gate_berkeley_loss_target) > 0.0)
        total_gate_runtime_enabled = bool(
            float(args.gate_berkeley_min_confidence) > 0.0
            or float(args.gate_berkeley_min_macro_f1) > 0.0
            or bool(berkeley_loss_gate_enabled)
        )
        gestation_gate_runtime_enabled = bool(float(args.gate_gestation_loss_target) > 0.0)
        if bool(total_gate_runtime_enabled or gestation_gate_runtime_enabled):
            _log(
                "Semantic gates enabled: "
                f"gestation_loss_target={float(args.gate_gestation_loss_target):.4f}, "
                f"total_loss_target={float(args.gate_berkeley_loss_target):.4f}, "
                "stage1=gestation_train_only, gate1=gestation_val_only, "
                "stage2=non_validation_payload_train, gate2=stage2_validation_deck[payload_validation_split_only]."
            )

        best_cfg = None
        best_score = -1.0
        search_rows = []
        resume_search_rows = _load_json(resume_search_path) if (resume_enabled and resume_search_path.exists()) else None
        did_config_search = False

        if bool(args.enforce_render_config):
            best_cfg = _build_enforced_render_config(args)
            best_score = 0.0
            search_rows = [{"trial": 0, "score": 0.0, "forced": True, "cfg": best_cfg.to_dict()}]
            _log(f"Render config enforced by CLI: {best_cfg.to_dict()}")
        elif resume_enabled and (not args.force_config_search):
            if isinstance(resume_cfg, dict):
                try:
                    best_cfg = RenderConfig.from_dict(resume_cfg)
                    _log(f"Resumed best render config from {resume_cfg_path}")
                except Exception:
                    best_cfg = None
            if best_cfg is None and isinstance(resume_pipeline_ckpt, dict) and isinstance(resume_pipeline_ckpt.get("best_cfg"), dict):
                try:
                    best_cfg = RenderConfig.from_dict(resume_pipeline_ckpt.get("best_cfg"))
                    _log("Resumed best render config from pipeline checkpoint")
                except Exception:
                    best_cfg = None
            if isinstance(resume_summary, dict):
                best_score = float(resume_summary.get("best_search_score", best_score))
                prior_refresh = resume_summary.get("berkeley_refresh_history", [])
                if isinstance(prior_refresh, list):
                    refresh_rows.extend(prior_refresh)
            if isinstance(resume_search_rows, list):
                search_rows = list(resume_search_rows)

        berkeley_refresh_samples_seen = _sum_berkeley_refresh_samples(refresh_rows)
        _log(
            "Stage 2 refresh loader deferred until Gate 1 is ready "
            "(Stage 1 trains on gestation-only rows; Gate 1 evaluates gestation validation rows)."
        )

        if best_cfg is None:
            did_config_search = True
            _log(f"Config search started on {len(search_indices)} rendered items...")
            for i, cfg in enumerate(candidates, start=1):
                try:
                    metric = _score_config_with_classifier(
                        records=records,
                        indices=search_indices,
                        cfg=cfg,
                        image_hw=image_hw,
                        classifier=classifier,
                        device=device,
                        batch_size=args.classifier_batch_size,
                        score_topk=args.score_topk,
                        score_threshold=args.score_threshold,
                        score_w_topk=args.score_w_topk,
                        score_w_cov=args.score_w_cov,
                        score_w_mean=args.score_w_mean,
                        amp_enabled=amp_enabled,
                        amp_dtype=args.amp_dtype,
                        channels_last=bool(args.channels_last),
                        active_classes=int(score_active_classes),
                    )
                    score = float(metric["score"])
                    search_rows.append({"trial": i, "score": score, "cfg": cfg.to_dict(), **metric})
                    _log(
                        f"  [{i}/{len(candidates)}] score={score:.4f} topk={metric['topk_mean']:.4f} "
                        f"cov={metric['hard_coverage']:.4f}"
                    )
                    if score > best_score:
                        best_score = score
                        best_cfg = cfg

                    if (
                        refresh_loader is not None
                        and int(args.berkeley_refresh_every) > 0
                        and (i % int(args.berkeley_refresh_every) == 0)
                    ):
                        need_full_refresh_startup = (
                            int(berkeley_refresh_required_samples) > 0
                            and int(berkeley_refresh_samples_seen) < int(berkeley_refresh_required_samples)
                        )
                        refresh_cache_x = refresh_cache.get("x") if isinstance(refresh_cache, dict) else None
                        refresh_cache_y = refresh_cache.get("y") if isinstance(refresh_cache, dict) else None
                        refresh_cache_batch = int(refresh_run_batch_size)
                        if (
                            need_full_refresh_startup
                            and isinstance(refresh_cache, dict)
                            and int(refresh_cache_samples) > 0
                            and int(refresh_cache_samples) < int(berkeley_refresh_required_samples)
                        ):
                            _log(
                                "    refresh startup requirement active: forcing loader mode "
                                f"(cache_samples={refresh_cache_samples} < required={berkeley_refresh_required_samples})"
                            )
                            refresh_cache_x = None
                            refresh_cache_y = None
                            refresh_cache_batch = 0
                        if need_full_refresh_startup:
                            if (
                                refresh_cache_x is not None
                                and refresh_cache_y is not None
                                and int(refresh_cache_batch) > 0
                            ):
                                min_refresh_steps = int(
                                    math.ceil(
                                        float(int(refresh_cache_x.shape[0])) / float(max(1, int(refresh_cache_batch)))
                                    )
                                )
                            else:
                                min_refresh_steps = max(1, int(len(refresh_loader)))
                        else:
                            min_refresh_steps = 0
                        refresh_max_steps = int(args.berkeley_refresh_max_steps)
                        if need_full_refresh_startup and int(min_refresh_steps) > 0:
                            refresh_max_steps = int(min_refresh_steps)
                        ref = _run_berkeley_refresh_epochs(
                            classifier=classifier,
                            loader=refresh_loader,
                            device=device,
                            epochs=int(args.berkeley_refresh_epochs),
                            lr=float(args.berkeley_refresh_lr),
                            weight_decay=float(args.berkeley_refresh_weight_decay),
                            max_steps=int(refresh_max_steps),
                            min_steps=int(min_refresh_steps),
                            lr_sine_cycles=float(args.lr_sine_cycles),
                            lr_sine_frequency=float(args.lr_sine_frequency),
                            lr_sine_tail_fraction=float(args.lr_sine_tail_fraction),
                            lr_sine_min_scale=float(args.lr_sine_min_scale),
                            amp_enabled=amp_enabled,
                            amp_dtype=args.amp_dtype,
                            channels_last=bool(args.channels_last),
                            grad_accum_steps=grad_accum_steps,
                            log_every=int(args.berkeley_refresh_log_every),
                            max_seconds=float(args.berkeley_refresh_max_seconds),
                            cache_x=refresh_cache_x,
                            cache_y=refresh_cache_y,
                            cache_batch_size=int(refresh_cache_batch),
                            active_classes=int(score_active_classes),
                            target_label_knockout_prob=float(args.target_label_knockout_prob),
                            target_label_knockout_min_keep=int(args.target_label_knockout_min_keep),
                            target_label_knockout_max_drop_frac=float(args.target_label_knockout_max_drop_frac),
                            target_label_knockout_seed=int(args.seed) + 61001 + int(i),
                            step_preview_callback=None,
                        )
                        refresh_rows.append({"stage": "config_search", "trial": int(i), **ref})
                        berkeley_refresh_samples_seen += int(max(0, int(ref.get("samples", 0))))
                        _log(
                            "    refresh: "
                            f"loss={ref['loss']:.4f} samples={ref.get('samples', 0)} "
                            f"source={ref.get('source', 'loader')} "
                            f"startup_seen={berkeley_refresh_samples_seen}/{berkeley_refresh_required_samples} "
                            f"elapsed={ref.get('elapsed_sec', 0.0):.1f}s"
                        )
                        _save_training_segment_snapshot(
                            enabled=bool(args.checkpoint_after_training_segment),
                            out_dir=out_dir,
                            objective_mode=args.objective_mode,
                            run_tag=run_tag,
                            segment=f"config_refresh_trial_{int(i)}",
                            best_cfg=best_cfg,
                            classifier=classifier,
                            refresh_history=refresh_rows,
                            extra={"best_search_score": float(best_score)},
                        )
                except Exception as e:
                    _log(f"  [{i}/{len(candidates)}] failed: {e}")

            if best_cfg is None:
                raise RuntimeError("Config search failed to produce a usable config.")
        else:
            if bool(args.enforce_render_config):
                _log("Skipping config search because --enforce-render-config is active.")
            else:
                _log("Skipping config search due to resume config (use --force-config-search to override).")

        if args.try_scipy_refine and did_config_search:
            refine_candidates = _try_scipy_refine(best_cfg, max_points=args.render_max_points, enabled=True)
            if len(refine_candidates) > 1:
                _log("Running optional refinement candidates...")
                for j, cfg in enumerate(refine_candidates[1:], start=1):
                    try:
                        metric = _score_config_with_classifier(
                            records=records,
                            indices=search_indices,
                            cfg=cfg,
                            image_hw=image_hw,
                            classifier=classifier,
                            device=device,
                            batch_size=args.classifier_batch_size,
                            score_topk=args.score_topk,
                            score_threshold=args.score_threshold,
                            score_w_topk=args.score_w_topk,
                            score_w_cov=args.score_w_cov,
                            score_w_mean=args.score_w_mean,
                            amp_enabled=amp_enabled,
                            amp_dtype=args.amp_dtype,
                            channels_last=bool(args.channels_last),
                            active_classes=int(score_active_classes),
                        )
                        score = float(metric["score"])
                        search_rows.append({"trial": len(search_rows) + 1, "score": score, "cfg": cfg.to_dict(), **metric})
                        _log(f"  [refine {j}] score={score:.4f}")
                        if score > best_score:
                            best_score = score
                            best_cfg = cfg
                    except Exception as e:
                        _log(f"  [refine {j}] failed: {e}")

        _log(f"Best cfg score={best_score:.4f}: {best_cfg.to_dict()}")
        if _sync_render_width_with_embed_hw(best_cfg, image_hw=image_hw):
            _log(
                "Embed resolution sync: adjusted render width to shared embed width "
                f"(render_width={int(best_cfg.width)}, embed_w={int(image_hw[1])})"
            )
        cls_val_idx = _sample_round_classifier_val_indices()
        cls_val = _score_config_with_classifier(
            records=records,
            indices=cls_val_idx,
            cfg=best_cfg,
            image_hw=image_hw,
            classifier=classifier,
            device=device,
            batch_size=args.classifier_batch_size,
            score_topk=args.score_topk,
            score_threshold=args.score_threshold,
            score_w_topk=args.score_w_topk,
            score_w_cov=args.score_w_cov,
            score_w_mean=args.score_w_mean,
            amp_enabled=amp_enabled,
            amp_dtype=args.amp_dtype,
            channels_last=bool(args.channels_last),
            active_classes=int(score_active_classes),
        )
        if bool(args.enforce_render_config):
            best_score = float(cls_val.get("score", best_score))
        _log(
            f"Best-config val metric: score={cls_val['score']:.4f} "
            f"topk={cls_val['topk_mean']:.4f} coverage={cls_val['hard_coverage']:.4f}"
        )

        gate_config = {
            "gestation_loss_target": max(0.0, float(args.gate_gestation_loss_target)),
            "gestation_maintain_rounds": max(1, int(args.gate_gestation_maintain_rounds)),
            "berkeley_min_score": float(args.gate_berkeley_min_score),
            "berkeley_min_confidence": float(args.gate_berkeley_min_confidence),
            "berkeley_min_macro_f1": float(args.gate_berkeley_min_macro_f1),
            "berkeley_loss_target": float(args.gate_berkeley_loss_target),
            "berkeley_loss_target_mult": float(args.gate_berkeley_loss_target_mult),
            "berkeley_loss_min_floor": float(args.gate_berkeley_loss_min_floor),
            "berkeley_loss_thaw_rounds": max(0, int(args.gate_berkeley_loss_thaw_rounds)),
            "berkeley_loss_ramp_rounds": max(1, int(args.gate_berkeley_loss_ramp_rounds)),
            "berkeley_loss_poly": float(args.gate_berkeley_loss_poly),
            "berkeley_loss_max_weight": float(args.gate_berkeley_loss_max_weight),
            "berkeley_loss_clamp_memory_rounds": max(1, int(args.gate_berkeley_loss_clamp_memory_rounds)),
            "berkeley_loss_release_rounds": max(1, int(args.gate_berkeley_loss_release_rounds)),
            "berkeley_loss_blend_margin": float(args.gate_berkeley_loss_blend_margin),
            "berkeley_maintain_rounds": max(1, int(args.gate_berkeley_maintain_rounds)),
            "wave_min_entropy": float(args.gate_wave_min_entropy),
            "wave_maintain_rounds": max(1, int(args.gate_wave_maintain_rounds)),
            "generator_min_target_prob": float(args.generator_gate_min_target_prob),
            "generator_maintain_rounds": max(1, int(args.generator_gate_maintain_rounds)),
            "transformer_min_score_after": float(args.gate_transformer_min_score_after),
            "transformer_min_gain": float(args.gate_transformer_min_gain),
            "transformer_maintain_rounds": max(1, int(args.gate_transformer_maintain_rounds)),
            "wave_feedback_min_acc": float(args.gate_wave_feedback_min_acc),
            "wave_feedback_min_zs_top1_rate": float(args.gate_wave_feedback_min_zs_top1_rate),
            "wave_feedback_zero_shot_weight": float(args.wave_feedback_zero_shot_weight),
            "wave_feedback_transformer_loss_boost": float(args.wave_feedback_transformer_loss_boost),
            "wave_feedback_generator_loss_boost": float(args.wave_feedback_generator_loss_boost),
            "total_token_schedule_enabled": bool(args.gate_total_token_schedule_enabled),
            "total_token_schedule_threshold": float(args.gate_total_token_schedule_threshold),
        }
        _log(
            "Gate config: "
            f"gestation_loss_target={float(gate_config['gestation_loss_target']):.4f}, "
            f"gestation_maintain={int(gate_config['gestation_maintain_rounds'])}, "
            f"berkeley_loss_target={float(gate_config['berkeley_loss_target']):.4f}, "
            f"berkeley_min_conf={float(gate_config['berkeley_min_confidence']):.4f}, "
            f"berkeley_min_macro_f1={float(gate_config['berkeley_min_macro_f1']):.4f}, "
            f"berkeley_maintain={int(gate_config['berkeley_maintain_rounds'])}"
        )

        train_streams, train_stream_labels, bits_train, train_meta = _prepare_streams(
            records=records,
            labels=labels_for_split,
            indices=final_train_idx,
            cfg=best_cfg,
            max_points=args.stream_max_points,
        )
        val_streams, val_stream_labels, bits_val, val_meta = _prepare_streams(
            records=records,
            labels=labels_for_split,
            indices=final_val_idx,
            cfg=best_cfg,
            max_points=args.stream_max_points,
        )
        if len(train_streams) < 4 or len(val_streams) < 2:
            raise RuntimeError("Not enough decoded streams for transformer stage.")

        bits_all = bits_train + bits_val
        sample_bits = int(Counter(bits_all).most_common(1)[0][0]) if len(bits_all) > 0 else 16
        chunk_samples, chunk_sync_info = _resolve_synced_chunk_samples(
            requested_chunk_samples=int(args.chunk_samples),
            patch_size=int(args.patch_size),
            cfg=best_cfg,
            image_hw=image_hw,
        )
        _log(
            "Transformer chunk sync: "
            f"requested={chunk_sync_info['requested']} resolved={chunk_sync_info['resolved']} "
            f"patch={chunk_sync_info['patch_size']} render={chunk_sync_info['render_h']}x{chunk_sync_info['render_w']} "
            f"target={chunk_sync_info['target_h']}x{chunk_sync_info['target_w']} "
            f"downsample={chunk_sync_info['downsample']} special_stride_mode={1 if chunk_sync_info['special_stride_mode'] else 0} "
            f"resize_needed={1 if chunk_sync_info['needs_resize'] else 0}"
        )
        if bool(chunk_sync_info["needs_resize"]):
            raise RuntimeError(
                "Transformer chunk sync could not match embed resolution exactly "
                f"(render={chunk_sync_info['render_h']}x{chunk_sync_info['render_w']} "
                f"target={chunk_sync_info['target_h']}x{chunk_sync_info['target_w']}). "
                "Adjust --image-size, --enforce-render-width, --enforce-render-downsample, "
                "--enforce-render-max-points, or --patch-size."
            )
        orchestration_mode = str(args.orchestration_mode).strip().lower()
        if orchestration_mode in ("staged_cgr", "staged_cgrw") and (not bool(best_cfg.bitmask_enable)):
            raise RuntimeError(f"{orchestration_mode} requires bitmask-enabled render config (best cfg has bitmask_enable=0).")

        if latent_fallback_info is not None and bool(args.latent_berkeley_imprint_mix_targets):
            if not bool(best_cfg.bitmask_enable):
                raise RuntimeError(
                    "Latent Berkeley mix-target imprint requires bitmask-enabled render config "
                    "(best cfg has bitmask_enable=0)."
                )
            lo, hi, depth = normalize_bit_window(
                int(best_cfg.bitmask_low),
                int(best_cfg.bitmask_high),
                int(sample_bits),
            )
            _log(f"Latent Berkeley imprint bit window: [{lo}..{hi}] ({depth} bits)")

        payload_images: List[np.ndarray] = []
        payload_conditions: List[np.ndarray] = []
        payload_bank_info = {"available": 0, "used": 0}
        payload_gate_val_indices_base: List[int] = []
        payload_gate_val_terms_base: List[List[str]] = []
        payload_gate_val_info: Dict[str, Any] = {"available_rows": 0, "selected_rows": 0, "reason": "not_built"}
        payload_conditioning_info: Dict[str, Any] = {"expanded": False, "reason": "not_built"}
        need_payload_bank = bool(int(condition_num_classes) > 0) and (
            (
                latent_fallback_info is not None
                and bool(args.latent_berkeley_imprint_mix_targets)
            )
            or (orchestration_mode in ("staged_cgr", "staged_cgrw"))
            or bool(total_gate_runtime_enabled)
            or bool(gestation_gate_runtime_enabled)
        )
        if need_payload_bank:
            payload_size = int(shared_embed_image_size)
            payload_max_samples = int(args.berkeley_payload_max_samples)
            if payload_max_samples <= 0:
                payload_max_samples = max(256, min(4096, max(1, int(args.latent_fallback_count)) * 8))
            payload_images, payload_conditions, payload_bank_info, payload_condition_terms = _build_berkeley_payload_bank(
                data_root=args.berkeley_data_root,
                image_size=int(payload_size),
                auto_install_scipy=bool(args.berkeley_auto_install_scipy),
                max_samples=int(payload_max_samples),
                seed=args.seed + 7067,
                source_root=str(args.berkeley_payload_source_root),
                force_cache_rebuild=bool(force_payload_cache_rebuild_runtime),
            )
            _log(
                "Latent Berkeley payload bank: "
                f"available={payload_bank_info.get('available', 0)} "
                f"used={payload_bank_info.get('used', 0)} "
                f"train={payload_bank_info.get('used_train', 0)}/{payload_bank_info.get('available_train', 0)} "
                f"val={payload_bank_info.get('used_val', 0)}/{payload_bank_info.get('available_val', 0)} "
                f"ext={payload_bank_info.get('used_external', 0)}/{payload_bank_info.get('available_external', 0)} "
                f"cap={int(payload_max_samples)} image_size={payload_size}"
            )
            _log(
                "[berkeley-payload-cache] "
                f"cache_hit={1 if bool(payload_bank_info.get('cache_hit', False)) else 0} "
                f"manifest={payload_bank_info.get('cache_manifest', '')} "
                f"sources={payload_bank_info.get('source_catalog', '')}"
            )
            payload_gate_val_indices_base, payload_gate_val_terms_base, payload_gate_val_info = _build_payload_validation_gate_rows(
                data_root=args.berkeley_data_root,
                image_size=int(payload_size),
                seed=int(gate_split_seed),
                external_val_fraction=float(gate_external_val_fraction),
            )
            _log(
                "[gate-val-deck] "
                "source=payload_cache "
                f"available={int(payload_gate_val_info.get('available_rows', 0))} "
                f"selected={int(payload_gate_val_info.get('selected_rows', 0))} "
                f"berkeley_val={int(payload_gate_val_info.get('selected_berkeley_val', 0))} "
                f"berkeley_train={int(payload_gate_val_info.get('selected_berkeley_train', 0))} "
                f"refresh_nonval_total={int(payload_gate_val_info.get('refresh_rows_total', 0))} "
                f"refresh_berkeley_train={int(payload_gate_val_info.get('refresh_rows_berkeley_train', 0))} "
                f"cache_label_dim={int(payload_gate_val_info.get('cache_label_dim', 0))} "
                f"external_frac={float(payload_gate_val_info.get('external_val_fraction', 0.0)):.2f}"
            )
            if need_payload_bank and len(payload_images) <= 0:
                raise RuntimeError(
                    "Payload bank is required but no Berkeley payload images were prepared. "
                    "Check --berkeley-data-root and dataset availability."
                )
            payload_images_base = [np.asarray(x, dtype=np.float32) for x in payload_images]
            payload_conditions_supervised_base = []
            payload_condition_terms_supervised_base = []
            payload_supervised_pos_all: List[int] = []
            payload_supervised_pos_berkeley: List[int] = []
            payload_supervised_label_hits = np.zeros((max(1, int(supervised_num_classes)),), dtype=np.int64)
            payload_active_vocab_lc = {
                re.sub(r"\s+", " ", str(x)).strip().lower()
                for x in class_names
                if re.sub(r"\s+", " ", str(x)).strip()
            }
            if len(payload_conditions) > 0:
                for i, row in enumerate(payload_conditions):
                    arr = np.asarray(row, dtype=np.float32).reshape(-1)
                    if int(arr.size) <= 0:
                        raise RuntimeError(f"Semantic payload row {i} is empty.")
                    if int(arr.size) > int(supervised_num_classes):
                        raise RuntimeError(
                            "Payload supervised row width exceeds classifier supervised width; refusing truncation. "
                            f"row={int(i)} row_dim={int(arr.size)} supervised_dim={int(supervised_num_classes)}"
                        )
                    if int(arr.size) >= int(supervised_num_classes):
                        arr = arr[: int(supervised_num_classes)]
                    else:
                        pad = np.zeros((int(supervised_num_classes) - int(arr.size),), dtype=np.float32)
                        arr = np.concatenate([arr, pad], axis=0)
                    arr_clip = np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)
                    payload_conditions_supervised_base.append(np.asarray(arr_clip, dtype=np.float32))
                    row_terms: List[str] = []
                    if int(i) < int(len(payload_condition_terms)) and isinstance(payload_condition_terms[int(i)], list):
                        row_terms = _normalize_vocab_terms([str(x) for x in payload_condition_terms[int(i)]])
                    if len(row_terms) <= 0:
                        row_terms = ["berkeley sbd dataset", "object", "signal"]
                    payload_condition_terms_supervised_base.append(list(row_terms))
                    pos_idx = np.where(np.asarray(arr_clip, dtype=np.float32) >= 0.5)[0].astype(np.int64).tolist()
                    payload_supervised_pos_all.append(int(len(pos_idx)))
                    pos_label_terms_lc: List[str] = []
                    for pi in pos_idx:
                        if 0 <= int(pi) < int(payload_supervised_label_hits.size):
                            payload_supervised_label_hits[int(pi)] += 1
                        if 0 <= int(pi) < int(len(supervised_class_names)):
                            lbl = re.sub(r"\s+", " ", str(supervised_class_names[int(pi)])).strip().lower()
                            if lbl:
                                pos_label_terms_lc.append(str(lbl))
                    pos_label_terms_lc = list(dict.fromkeys(pos_label_terms_lc))
                    missing_active_vocab = [str(t) for t in pos_label_terms_lc if str(t) not in payload_active_vocab_lc]
                    if len(missing_active_vocab) > 0:
                        raise RuntimeError(
                            "Berkeley payload row has supervised labels outside active vocabulary. "
                            f"row={int(i)} missing={','.join(missing_active_vocab[:8])}"
                        )
                    row_terms_lc = {
                        re.sub(r"\s+", " ", str(x)).strip().lower()
                        for x in row_terms
                        if re.sub(r"\s+", " ", str(x)).strip()
                    }
                    if "berkeley sbd dataset" in row_terms_lc:
                        payload_supervised_pos_berkeley.append(int(len(pos_idx)))
                if int(len(payload_supervised_pos_berkeley)) > 0:
                    berk_pos_np = np.asarray(payload_supervised_pos_berkeley, dtype=np.int32)
                    if int(np.count_nonzero(berk_pos_np > 0)) < int(berk_pos_np.size):
                        raise RuntimeError(
                            "Berkeley payload rows cannot be label-empty in supervised target bank. "
                            f"nonzero={int(np.count_nonzero(berk_pos_np > 0))}/{int(berk_pos_np.size)}"
                        )
                all_pos_np = np.asarray(payload_supervised_pos_all, dtype=np.int32) if int(len(payload_supervised_pos_all)) > 0 else np.zeros((0,), dtype=np.int32)
                top_label_pairs = sorted(
                    [
                        (str(supervised_class_names[int(ci)]) if int(ci) < int(len(supervised_class_names)) else f"class_{int(ci)}", int(cnt))
                        for ci, cnt in enumerate(payload_supervised_label_hits.tolist())
                        if int(cnt) > 0
                    ],
                    key=lambda kv: (-int(kv[1]), str(kv[0]).lower()),
                )[:8]
                top_label_txt = "; ".join([f"{k}:{v}" for k, v in top_label_pairs]) if len(top_label_pairs) > 0 else "none"
                _log(
                    "[payload-supervised-targets] "
                    f"rows={int(all_pos_np.size)} "
                    f"all_min/mean/max={int(all_pos_np.min()) if int(all_pos_np.size) > 0 else 0}/"
                    f"{float(np.mean(all_pos_np)) if int(all_pos_np.size) > 0 else 0.0:.2f}/"
                    f"{int(all_pos_np.max()) if int(all_pos_np.size) > 0 else 0} "
                    f"berkeley_rows={int(len(payload_supervised_pos_berkeley))} "
                    f"berkeley_min/mean/max={int(np.min(payload_supervised_pos_berkeley)) if int(len(payload_supervised_pos_berkeley)) > 0 else 0}/"
                    f"{float(np.mean(payload_supervised_pos_berkeley)) if int(len(payload_supervised_pos_berkeley)) > 0 else 0.0:.2f}/"
                    f"{int(np.max(payload_supervised_pos_berkeley)) if int(len(payload_supervised_pos_berkeley)) > 0 else 0} "
                    f"top={top_label_txt}"
                )
            payload_conditions = []
            if int(semantic_extra_count) > 0 and bool(args.semantic_vocab_auto_symbol_pool):
                symbol_samples_per_term_eff = max(4, int(args.semantic_vocab_symbol_samples_per_term))
                symbol_pool_official, symbol_pool_info_official = _build_auto_symbol_term_pool(
                    data_root=str(args.semantic_vocab_symbol_pool_root),
                    image_size=int(payload_size),
                    seed=int(args.seed) + 8453,
                    max_samples_per_term=int(symbol_samples_per_term_eff),
                    include_digits=True,
                    include_letters=True,
                    include_pictograms=bool(args.semantic_vocab_symbol_include_pictograms),
                )
                symbol_pool_synth, symbol_pool_info_synth = _build_synthetic_semantic_symbol_pool(
                    image_size=int(payload_size),
                    seed=int(args.seed) + 9127,
                    max_samples_per_term=int(symbol_samples_per_term_eff),
                )
                symbol_pool_bootstrap, symbol_pool_info_bootstrap = _build_internal_bootstrap_symbol_pool(
                    data_root=str(args.semantic_vocab_symbol_pool_root),
                    image_size=int(payload_size),
                    seed=int(args.seed) + 9769,
                    max_samples_per_term=int(symbol_samples_per_term_eff),
                    origin_label=str(args.semantic_vocab_bootstrap_origin_label),
                )
                bootstrap_symbol_terms = {
                    re.sub(r"\s+", " ", str(k)).strip().lower()
                    for k in symbol_pool_bootstrap.keys()
                    if str(k).strip()
                }
                symbol_pool_by_term = _merge_symbol_term_pools(
                    pools=[symbol_pool_official, symbol_pool_synth, symbol_pool_bootstrap],
                    max_samples_per_term=int(symbol_samples_per_term_eff),
                )
                symbol_pool_origin_terms_by_term = _build_symbol_term_origin_terms_map(
                    official_pool=symbol_pool_official,
                    synthetic_pool=symbol_pool_synth,
                    bootstrap_pool=symbol_pool_bootstrap,
                    bootstrap_origin_label=str(args.semantic_vocab_bootstrap_origin_label),
                )
                digits_terms_total = int(
                    sum(
                        1
                        for k in symbol_pool_by_term.keys()
                        if re.sub(r"\s+", " ", str(k)).strip().lower().startswith("digit ")
                    )
                )
                letters_terms_total = int(
                    sum(
                        1
                        for k in symbol_pool_by_term.keys()
                        if re.sub(r"\s+", " ", str(k)).strip().lower().startswith("letter ")
                    )
                )
                pictogram_terms_total = int(
                    sum(
                        1
                        for k in symbol_pool_by_term.keys()
                        if re.sub(r"\s+", " ", str(k)).strip().lower().startswith("pictogram ")
                    )
                )
                symbol_pool_info = {
                    "enabled": True,
                    "available_terms": int(len(symbol_pool_by_term)),
                    "samples": int(sum(len(v) for v in symbol_pool_by_term.values())),
                    "samples_per_term_effective": int(symbol_samples_per_term_eff),
                    "official_terms": int(symbol_pool_info_official.get("available_terms", 0)),
                    "official_samples": int(symbol_pool_info_official.get("samples", 0)),
                    "synthetic_terms": int(symbol_pool_info_synth.get("available_terms", 0)),
                    "synthetic_samples": int(symbol_pool_info_synth.get("samples", 0)),
                    "bootstrap_terms": int(symbol_pool_info_bootstrap.get("available_terms", 0)),
                    "bootstrap_samples": int(symbol_pool_info_bootstrap.get("samples", 0)),
                    "bootstrap_root_dir": str(symbol_pool_info_bootstrap.get("root_dir", "")),
                    "bootstrap_origin_label": str(symbol_pool_info_bootstrap.get("origin_label", "")),
                    "digits_terms": int(digits_terms_total),
                    "letters_terms": int(letters_terms_total),
                    "pictogram_terms": int(pictogram_terms_total),
                    "official_digits_terms": int(symbol_pool_info_official.get("digits_terms", 0)),
                    "official_letters_terms": int(symbol_pool_info_official.get("letters_terms", 0)),
                    "official_pictogram_terms": int(symbol_pool_info_official.get("pictogram_terms", 0)),
                    "origin_terms_mapped": int(len(symbol_pool_origin_terms_by_term)),
                    "reason": str(symbol_pool_info_official.get("reason", "")),
                }
                if int(symbol_pool_info.get("available_terms", 0)) > 0:
                    semantic_vocab_pool_terms = _normalize_vocab_terms(
                        list(semantic_vocab_pool_terms) + list(symbol_pool_by_term.keys())
                    )
                gestation_symbol_terms = {
                    re.sub(r"\s+", " ", str(t)).strip().lower()
                    for t in _default_bootstrap_primitive_terms()
                    if str(t).strip()
                }
                if len(gestation_symbol_terms) <= 0:
                    gestation_symbol_terms = {
                        re.sub(r"\s+", " ", str(t)).strip().lower()
                        for t in _default_bootstrap_primitive_terms()
                        if str(t).strip()
                    }
                _log(
                    "[gestation-vocab] "
                    f"source=bootstrap_primitive_only terms={int(len(gestation_symbol_terms))}"
                )
                _log(
                    "[semantic-vocab] auto symbol pool: "
                    f"enabled={1 if bool(symbol_pool_info.get('enabled', False)) else 0} "
                    f"terms={int(symbol_pool_info.get('available_terms', 0))} "
                    f"samples={int(symbol_pool_info.get('samples', 0))} "
                    f"official_terms={int(symbol_pool_info.get('official_terms', 0))} "
                    f"synthetic_terms={int(symbol_pool_info.get('synthetic_terms', 0))} "
                    f"bootstrap_terms={int(symbol_pool_info.get('bootstrap_terms', 0))} "
                    f"bootstrap_samples={int(symbol_pool_info.get('bootstrap_samples', 0))} "
                    f"samples_per_term_eff={int(symbol_pool_info.get('samples_per_term_effective', 0))} "
                    f"bootstrap_origin={str(symbol_pool_info.get('bootstrap_origin_label', ''))} "
                    f"digits_total={int(symbol_pool_info.get('digits_terms', 0))} "
                    f"letters_total={int(symbol_pool_info.get('letters_terms', 0))} "
                    f"pictograms_total={int(symbol_pool_info.get('pictogram_terms', 0))} "
                    f"reason={str(symbol_pool_info.get('reason', ''))} "
                    f"bootstrap_dir={str(symbol_pool_info.get('bootstrap_root_dir', ''))}"
                )
            if need_payload_bank and len(payload_conditions) <= 0:
                if len(payload_conditions_supervised_base) <= 0:
                    raise RuntimeError("Semantic payload condition bank is empty.")
            if bool(need_payload_bank):
                payload_conditions, payload_conditioning_info = _expand_payload_conditions_with_semantic_bank(
                    payload_conditions=payload_conditions_supervised_base,
                    condition_num_classes=int(condition_num_classes),
                    supervised_num_classes=int(supervised_num_classes),
                    label_embedding_bank=label_embedding_bank,
                    semantic_target_temperature=8.0,
                )
                if len(payload_conditions) == len(payload_condition_terms_supervised_base):
                    payload_conditions = [
                        np.maximum(
                            np.asarray(payload_conditions[i], dtype=np.float32),
                            _semantic_condition_from_terms_runtime(
                                terms=payload_condition_terms_supervised_base[i],
                                origin_terms=[],
                            ),
                        ).astype(np.float32, copy=False)
                        for i in range(len(payload_conditions))
                    ]
                latent_payload_dims = sorted(
                    {
                        int(np.asarray(row, dtype=np.float32).reshape(-1).size)
                        for row in payload_conditions
                    }
                )
                bad_latent_payload_dims = [
                    int(d) for d in latent_payload_dims if int(d) != int(condition_num_classes)
                ]
                if len(bad_latent_payload_dims) > 0:
                    raise RuntimeError(
                        "Latent payload imprint requires full semantic-width targets: "
                        f"expected={int(condition_num_classes)} got_dims={bad_latent_payload_dims}"
                    )
                _log(
                    "[payload-latent-conditioning] "
                    f"rows={int(len(payload_conditions))} "
                    f"condition_dim={int(condition_num_classes)} "
                    f"reason={str(payload_conditioning_info.get('reason', 'unknown'))}"
                )
            if len(payload_conditions) != len(payload_images):
                raise RuntimeError(
                    "Payload image/condition bank mismatch: "
                    f"images={len(payload_images)} conditions={len(payload_conditions)}"
                )

        train_stream_target_labels: Optional[List[Any]] = None
        val_stream_target_labels: Optional[List[Any]] = None
        latent_targeting_info = {
            "enabled": False,
            "payload_bank": payload_bank_info,
            "payload_conditioning": dict(payload_conditioning_info),
        }
        if (
            latent_fallback_info is not None
            and bool(args.latent_berkeley_imprint_mix_targets)
            and int(condition_num_classes) > 0
            and len(payload_images) > 0
        ):
            train_streams, train_stream_target_labels, train_target_info = _imprint_latent_stream_targets(
                streams=train_streams,
                metas=train_meta,
                payload_images=payload_images,
                payload_targets=payload_conditions,
                num_classes=int(condition_num_classes),
                chunk_samples=int(chunk_samples),
                classifier=classifier,
                cfg=best_cfg,
                sample_bits=int(sample_bits),
                image_hw=image_hw,
                device=device,
                seed=args.seed + 111,
                steps=int(args.latent_berkeley_imprint_steps),
                lr=float(args.latent_berkeley_imprint_lr),
                l2_weight=float(args.latent_berkeley_imprint_l2),
                min_target_classes=int(args.latent_berkeley_imprint_min_target_classes),
                max_target_classes=int(args.latent_berkeley_imprint_max_target_classes),
                kind_target_vectors=semantic_kind_target_vectors,
                semantic_term_to_idx=semantic_term_to_idx,
                amp_enabled=amp_enabled,
                amp_dtype=args.amp_dtype,
                channels_last=bool(args.channels_last),
            )
            val_streams, val_stream_target_labels, val_target_info = _imprint_latent_stream_targets(
                streams=val_streams,
                metas=val_meta,
                payload_images=payload_images,
                payload_targets=payload_conditions,
                num_classes=int(condition_num_classes),
                chunk_samples=int(chunk_samples),
                classifier=classifier,
                cfg=best_cfg,
                sample_bits=int(sample_bits),
                image_hw=image_hw,
                device=device,
                seed=args.seed + 222,
                steps=int(args.latent_berkeley_imprint_steps),
                lr=float(args.latent_berkeley_imprint_lr),
                l2_weight=float(args.latent_berkeley_imprint_l2),
                min_target_classes=int(args.latent_berkeley_imprint_min_target_classes),
                max_target_classes=int(args.latent_berkeley_imprint_max_target_classes),
                kind_target_vectors=semantic_kind_target_vectors,
                semantic_term_to_idx=semantic_term_to_idx,
                amp_enabled=amp_enabled,
                amp_dtype=args.amp_dtype,
                channels_last=bool(args.channels_last),
            )
            latent_targeting_info = {
                "enabled": True,
                "payload_bank": payload_bank_info,
                "payload_conditioning": dict(payload_conditioning_info),
                "train": train_target_info,
                "val": val_target_info,
            }
            _log(
                "Latent Berkeley target imprint: "
                f"train_imprinted={train_target_info['imprinted_streams']}/{train_target_info['mix_streams']} "
                f"(targeted={train_target_info['target_supervised_streams']}, "
                f"targets(min/mean/max)="
                f"{int(train_target_info.get('target_classes_min', 0))}/"
                f"{float(train_target_info['mean_target_classes_per_stream']):.2f}/"
                f"{int(train_target_info.get('target_classes_max', 0))}, "
                f"mean_payload_rows={train_target_info.get('mean_payload_rows_per_stream', 0.0):.2f}, "
                f"mean_prob={train_target_info['mean_target_prob_after_imprint']:.4f}), "
                f"val_imprinted={val_target_info['imprinted_streams']}/{val_target_info['mix_streams']} "
                f"(targeted={val_target_info['target_supervised_streams']}, "
                f"targets(min/mean/max)="
                f"{int(val_target_info.get('target_classes_min', 0))}/"
                f"{float(val_target_info['mean_target_classes_per_stream']):.2f}/"
                f"{int(val_target_info.get('target_classes_max', 0))}, "
                f"mean_payload_rows={val_target_info.get('mean_payload_rows_per_stream', 0.0):.2f}, "
                f"mean_prob={val_target_info['mean_target_prob_after_imprint']:.4f})"
            )

        train_streams, train_stream_labels, train_meta, train_stream_target_labels, train_len_info = _filter_stream_pool_for_chunk_samples(
            streams=train_streams,
            labels=train_stream_labels,
            metas=train_meta,
            chunk_samples=int(chunk_samples),
            pool_name="transformer_train_streams",
            target_labels=train_stream_target_labels,
        )
        val_streams, val_stream_labels, val_meta, val_stream_target_labels, val_len_info = _filter_stream_pool_for_chunk_samples(
            streams=val_streams,
            labels=val_stream_labels,
            metas=val_meta,
            chunk_samples=int(chunk_samples),
            pool_name="transformer_val_streams",
            target_labels=val_stream_target_labels,
        )
        _log(
            "Transformer stream-length filter: "
            f"train kept={train_len_info['after']}/{train_len_info['before']} dropped={train_len_info['dropped']} "
            f"val kept={val_len_info['after']}/{val_len_info['before']} dropped={val_len_info['dropped']} "
            f"required_chunk={train_len_info['required_chunk_samples']}"
        )
        if len(train_streams) < 4 or len(val_streams) < 2:
            raise RuntimeError(
                "Not enough long streams after enforcing chunk length: "
                f"train={len(train_streams)} val={len(val_streams)} required_chunk={int(chunk_samples)}. "
                "Provide longer WAVs or increase --latent-fallback-seconds."
            )

        transformer = WavePatchTransformer(
            chunk_samples=chunk_samples,
            patch_size=int(args.patch_size),
            d_model=int(args.transformer_d_model),
            nhead=int(args.transformer_nhead),
            num_layers=int(args.transformer_num_layers),
            ff_mult=int(args.transformer_ff_mult),
            max_delta=float(args.max_delta),
            dropout=float(args.transformer_dropout),
            prefilter_max_skew=float(args.transformer_deskew_prefilter_max_skew),
            prefilter_enabled=bool("deskew" in transformer_filter_bundle_names),
            aux_filter_bundle_names=transformer_filter_bundle_names,
        ).to(device)
        baseline_eval_batch = max(
            1,
            (
                int(args.transformer_eval_batch_size)
                if int(args.transformer_eval_batch_size) > 0
                else int(args.transformer_batch_size)
            ),
        )

        baseline = evaluate_feature_score_before_after(
            transformer=transformer,
            classifier=classifier,
            streams=val_streams,
            cfg=best_cfg,
            sample_bits=sample_bits,
            image_hw=image_hw,
            chunk_samples=chunk_samples,
            device=device,
            max_batches=16,
            batch_size=int(baseline_eval_batch),
            score_topk=args.score_topk,
            score_threshold=args.score_threshold,
            score_w_topk=args.score_w_topk,
            score_w_cov=args.score_w_cov,
            score_w_mean=args.score_w_mean,
            amp=amp_enabled,
            amp_dtype=args.amp_dtype,
            channels_last=bool(args.channels_last),
            pin_memory=bool(args.pin_memory_wave_batches),
            stream_cache_on_device=bool(args.cache_wave_streams_on_device),
        )
        _log(
            f"Transformer baseline: before={baseline['score_before']:.4f} "
            f"after={baseline['score_after']:.4f} gain={baseline['score_gain']:.4f}"
        )

        mode = str(args.orchestration_mode).strip().lower()
        transformer_hist: List[Dict[str, float]] = []
        wave_classifier_hist: List[Dict[str, float]] = []
        generator_hist: List[Dict[str, float]] = []
        orchestration_rows: List[Dict] = []
        accepted_total: List[Dict] = []
        gate_history: List[Dict] = []
        wave_classifier: Optional[nn.Module] = None
        generator: Optional[nn.Module] = None
        discriminator: Optional[nn.Module] = None
        wave_classifier_init_info = {"used": False}
        wave_classifier_resume_info = {"used": False}
        generator_resume_info = {"used": False}
        discriminator_resume_info = {"used": False}
        transformer_resume_info = {"used": False}
        deskew_prefilter_resume_info = {"used": False}
        wave_classifier_last_eval = None
        wave_zero_shot_last_eval = None
        wave_zero_shot_history: List[Dict[str, Any]] = []
        training_preview_topk = max(1, int(args.training_preview_topk))
        training_preview_every = max(1, int(args.training_preview_every_round))
        training_preview_root = out_dir / "training_supervision"
        training_preview_manifest_path = training_preview_root / "manifest.jsonl"
        if bool(args.training_preview_enabled):
            training_preview_root.mkdir(parents=True, exist_ok=True)
        library_dir = out_dir / "accepted_wave_library"
        library_index_path = library_dir / "index.jsonl"
        accepted_repair_info = _repair_accepted_library_index_semantics(library_dir=library_dir)
        if bool(accepted_repair_info.get("ran", False)):
            _log(
                "[accepted-library] semantic repair: "
                f"rows={int(accepted_repair_info.get('rows', 0))} "
                f"updated={int(accepted_repair_info.get('updated', 0))} "
                f"parse_failures={int(accepted_repair_info.get('parse_failures', 0))} "
                f"path={str(accepted_repair_info.get('path', ''))}"
            )
        pipeline_checkpoint_path = out_dir / "pipeline_checkpoint.pt"
        semantic_vocab_churn_runtime = bool(args.semantic_vocab_churn_enabled) and int(semantic_extra_count) > 0
        semantic_vocab_pool_terms = _normalize_vocab_terms(semantic_vocab_pool_terms)

        def _sanitize_regurgitated_churn_terms(raw_terms: Sequence[str]) -> List[str]:
            if isinstance(raw_terms, str):
                terms = _normalize_vocab_terms([str(raw_terms)])
            elif isinstance(raw_terms, (list, tuple, set)):
                terms = _normalize_vocab_terms([str(x) for x in raw_terms])
            else:
                terms = []
            out: List[str] = []
            for term in terms:
                key = re.sub(r"\s+", " ", str(term)).strip().lower()
                if not key:
                    continue
                if key in base_lc:
                    continue
                if key in semantic_core_lc:
                    continue
                out.append(str(term))
            return _normalize_vocab_terms(out)

        def _register_regurgitated_churn_row(row: Dict[str, Any], source: str):
            nonlocal regurgitated_churn_seq
            if not bool(args.semantic_vocab_regurgitated_churn_enabled):
                return
            if not isinstance(row, dict):
                return
            path_txt = str(row.get("file", "")).strip()
            if not path_txt:
                return
            terms = []
            if isinstance(row.get("semantic_terms", []), list):
                terms.extend([str(x) for x in row.get("semantic_terms", [])])
            terms.extend(["regurgitated content", "mixed noise and signal", "signal"])
            terms = _sanitize_regurgitated_churn_terms(terms)
            if len(terms) <= 0:
                return
            life_default = max(1, int(args.semantic_vocab_regurgitated_churn_lifetime))
            for item in regurgitated_churn_pool:
                if str(item.get("path", "")).strip() == path_txt:
                    item["terms"] = _normalize_vocab_terms(list(item.get("terms", [])) + list(terms))
                    item["lifetime"] = max(int(item.get("lifetime", 0)), int(life_default))
                    item["source"] = str(source)
                    return
            regurgitated_churn_seq += 1
            regurgitated_churn_pool.append(
                {
                    "path": str(path_txt),
                    "terms": list(terms),
                    "lifetime": int(life_default),
                    "uses": 0,
                    "seq": int(regurgitated_churn_seq),
                    "source": str(source),
                }
            )
            regurgitated_churn_seen_paths.add(str(path_txt))
            cap = max(0, int(args.semantic_vocab_regurgitated_churn_capacity))
            if cap > 0 and len(regurgitated_churn_pool) > cap:
                regurgitated_churn_pool.sort(key=lambda d: int(d.get("seq", 0)))
                while len(regurgitated_churn_pool) > cap:
                    drop = regurgitated_churn_pool.pop(0)
                    regurgitated_churn_seen_paths.discard(str(drop.get("path", "")).strip())

        def _consume_regurgitated_churn_terms(cycle_seed: int) -> Tuple[List[str], Dict[str, Any]]:
            info: Dict[str, Any] = {
                "enabled": bool(args.semantic_vocab_regurgitated_churn_enabled),
                "triggered": False,
                "pool_before": int(len(regurgitated_churn_pool)),
                "pool_after": int(len(regurgitated_churn_pool)),
                "used_items": 0,
                "expired_items": 0,
                "terms_added": 0,
                "reason": "",
            }
            if not bool(args.semantic_vocab_regurgitated_churn_enabled):
                info["reason"] = "disabled"
                return [], info
            if len(regurgitated_churn_pool) <= 0:
                info["reason"] = "empty_pool"
                return [], info
            p_use = max(0.0, min(1.0, float(args.semantic_vocab_regurgitated_churn_prob)))
            if p_use <= 0.0:
                info["reason"] = "zero_probability"
                return [], info
            rng_reg = np.random.default_rng(int(cycle_seed))
            if float(rng_reg.random()) > p_use:
                info["reason"] = "probability_skip"
                return [], info
            alive_idx = [
                int(i)
                for i, item in enumerate(regurgitated_churn_pool)
                if int(item.get("lifetime", 0)) > 0 and len(_sanitize_regurgitated_churn_terms(item.get("terms", []))) > 0
            ]
            if len(alive_idx) <= 0:
                info["reason"] = "no_alive_items"
                regurgitated_churn_pool.clear()
                regurgitated_churn_seen_paths.clear()
                info["pool_after"] = 0
                return [], info
            take = min(
                max(1, int(args.semantic_vocab_regurgitated_churn_max_per_cycle)),
                int(len(alive_idx)),
            )
            picks = (
                rng_reg.choice(np.asarray(alive_idx, dtype=np.int64), size=int(take), replace=False)
                .astype(np.int64)
                .tolist()
            )
            info["triggered"] = True
            gathered_terms: List[str] = []
            for idx in picks:
                item = regurgitated_churn_pool[int(idx)]
                gathered_terms.extend(_sanitize_regurgitated_churn_terms(item.get("terms", [])))
                item["lifetime"] = int(item.get("lifetime", 0)) - 1
                item["uses"] = int(item.get("uses", 0)) + 1
            keep: List[Dict[str, Any]] = []
            expired = 0
            for item in regurgitated_churn_pool:
                if int(item.get("lifetime", 0)) > 0:
                    keep.append(item)
                else:
                    expired += 1
                    regurgitated_churn_seen_paths.discard(str(item.get("path", "")).strip())
            regurgitated_churn_pool[:] = keep
            gathered_terms = _sanitize_regurgitated_churn_terms(gathered_terms)
            info["used_items"] = int(len(picks))
            info["expired_items"] = int(expired)
            info["terms_added"] = int(len(gathered_terms))
            info["pool_after"] = int(len(regurgitated_churn_pool))
            if int(len(gathered_terms)) <= 0:
                info["reason"] = "no_valid_terms"
            else:
                info["reason"] = "ok"
            return gathered_terms, info

        def _register_live_gan_churn_row(image: np.ndarray, target: np.ndarray, source: str):
            nonlocal gan_live_churn_seq
            life_default = max(1, int(args.semantic_vocab_regurgitated_churn_lifetime))
            row = {
                "image": _image_any_to_rgb_chw01(image, image_size=int(shared_embed_image_size)),
                "target": np.asarray(target, dtype=np.float32).reshape(-1),
                "lifetime": int(life_default),
                "uses": 0,
                "seq": int(gan_live_churn_seq) + 1,
                "source": str(source),
            }
            gan_live_churn_seq = int(row["seq"])
            gan_live_churn_pool.append(row)
            cap = max(8, int(args.semantic_vocab_regurgitated_churn_capacity))
            if len(gan_live_churn_pool) > int(cap):
                gan_live_churn_pool.sort(key=lambda d: int(d.get("seq", 0)))
                del gan_live_churn_pool[: int(len(gan_live_churn_pool) - int(cap))]

        def _harvest_live_gan_churn_rows(cycle_seed: int, count: int, source: str) -> Dict[str, Any]:
            info: Dict[str, Any] = {
                "requested": int(max(0, int(count))),
                "harvested": 0,
                "reason": "",
            }
            n = max(0, int(count))
            if n <= 0:
                info["reason"] = "zero_requested"
                return info
            if generator is None:
                info["reason"] = "generator_unavailable"
                return info
            gen_device = torch.device("cuda" if _module_device_type(generator) == "cuda" else "cpu")
            cond_rows: List[np.ndarray] = []
            if len(payload_conditions) > 0:
                for row in payload_conditions:
                    arr = np.asarray(row, dtype=np.float32).reshape(-1)
                    if int(arr.size) == int(condition_num_classes):
                        cond_rows.append(arr.astype(np.float32, copy=False))
            if len(cond_rows) <= 0:
                # Do not synthesize fallback GAN prompts.
                # If no prompt-conditioning rows exist, churn stays dry.
                info["reason"] = "no_prompt_conditions"
                return info
            try:
                rng_gan = np.random.default_rng(int(cycle_seed) + 6113)
                picks = rng_gan.integers(0, len(cond_rows), size=n)
                cond_np = np.stack([cond_rows[int(i)] for i in picks], axis=0).astype(np.float32, copy=False)
                cond_t = torch.from_numpy(cond_np).to(device=gen_device, dtype=torch.float32)
                z = torch.randn((int(n), max(8, int(args.generator_z_dim))), device=gen_device)
                was_train = bool(generator.training)
                generator.eval()
                try:
                    with torch.no_grad():
                        fake = generator(z, cond_t).to(torch.float32).detach().cpu()
                finally:
                    if was_train:
                        generator.train()
                for i in range(int(fake.shape[0])):
                    base_cond = np.asarray(cond_np[int(i)], dtype=np.float32).reshape(-1)
                    img_np = fake[int(i)].numpy()
                    tone_terms = _semantic_tonal_tags_from_image(
                        image=img_np,
                        image_size=int(shared_embed_image_size),
                    )
                    target_vec = np.array(base_cond, dtype=np.float32, copy=True)
                    if len(tone_terms) > 0:
                        addon_vec = np.asarray(
                            _semantic_condition_from_terms_runtime(
                                terms=tone_terms,
                                origin_terms=["live generator churn", "tonal tag enrichment"],
                            ),
                            dtype=np.float32,
                        ).reshape(-1)
                        if int(addon_vec.size) != int(base_cond.size):
                            info["reason"] = "target_dim_mismatch"
                            return info
                        target_vec = np.maximum(target_vec, addon_vec).astype(np.float32, copy=False)
                    _register_live_gan_churn_row(
                        image=img_np,
                        target=target_vec,
                        source=str(source),
                    )
                info["harvested"] = int(fake.shape[0])
                info["reason"] = "ok"
            except Exception as e:
                info["reason"] = f"harvest_error:{type(e).__name__}"
            return info

        def _consume_live_gan_churn_rows(cycle_seed: int, requested: int) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, Any]]:
            nonlocal gan_live_churn_last_info
            info: Dict[str, Any] = {
                "enabled": True,
                "triggered": False,
                "harvested": 0,
                "used_items": 0,
                "expired_items": 0,
                "pool_before": int(len(gan_live_churn_pool)),
                "pool_after": int(len(gan_live_churn_pool)),
                "reason": "",
            }
            need = max(0, int(requested))
            if need <= 0:
                info["reason"] = "zero_requested"
                gan_live_churn_last_info = dict(info)
                return [], [], info
            alive = [i for i, item in enumerate(gan_live_churn_pool) if int(item.get("lifetime", 0)) > 0]
            if len(alive) < int(need):
                harvest_need = int(max(int(need), int(need * 2) - int(len(alive))))
                harvest_info = _harvest_live_gan_churn_rows(
                    cycle_seed=int(cycle_seed),
                    count=int(harvest_need),
                    source="generator_live",
                )
                info["harvested"] = int(harvest_info.get("harvested", 0))
                if int(info["harvested"]) <= 0 and len(alive) <= 0:
                    info["reason"] = str(harvest_info.get("reason", "harvest_empty"))
                    gan_live_churn_last_info = dict(info)
                    return [], [], info
            alive = [i for i, item in enumerate(gan_live_churn_pool) if int(item.get("lifetime", 0)) > 0]
            if len(alive) <= 0:
                info["reason"] = "empty_pool"
                gan_live_churn_last_info = dict(info)
                return [], [], info
            rng_gan = np.random.default_rng(int(cycle_seed) + 6199)
            take = int(min(int(need), int(len(alive))))
            picks = (
                rng_gan.choice(np.asarray(alive, dtype=np.int64), size=int(take), replace=False)
                .astype(np.int64)
                .tolist()
            )
            out_images: List[np.ndarray] = []
            out_targets: List[np.ndarray] = []
            info["triggered"] = True
            for idx in picks:
                item = gan_live_churn_pool[int(idx)]
                out_images.append(np.asarray(item.get("image"), dtype=np.float32))
                out_targets.append(np.asarray(item.get("target"), dtype=np.float32).reshape(-1))
                item["lifetime"] = int(item.get("lifetime", 0)) - 1
                item["uses"] = int(item.get("uses", 0)) + 1
            keep: List[Dict[str, Any]] = []
            expired = 0
            for item in gan_live_churn_pool:
                if int(item.get("lifetime", 0)) > 0:
                    keep.append(item)
                else:
                    expired += 1
            gan_live_churn_pool[:] = keep
            info["used_items"] = int(len(out_images))
            info["expired_items"] = int(expired)
            info["pool_after"] = int(len(gan_live_churn_pool))
            info["reason"] = "ok" if int(len(out_images)) > 0 else "no_rows"
            gan_live_churn_last_info = dict(info)
            return out_images, out_targets, info

        def _append_default_mix_targets(target_rows: Optional[List[Any]], add_count: int):
            n = max(0, int(add_count))
            if target_rows is None or n <= 0:
                return
            base_vec = regurgitated_condition_target_default
            if base_vec is None:
                base_vec = mix_condition_target_default
            if base_vec is None:
                target_rows.extend([None] * int(n))
                return
            for _ in range(int(n)):
                target_rows.append(np.array(base_vec, dtype=np.float32, copy=True))

        def _build_symbol_payload_rows_for_active_terms(cycle_seed: int) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, Any]]:
            if int(semantic_extra_count) <= 0:
                return [], [], {"enabled": False, "rows_added": 0, "matched_terms": 0}
            rng_sym = np.random.default_rng(int(cycle_seed))
            images_out: List[np.ndarray] = []
            conds_out: List[np.ndarray] = []
            matched = 0
            embedded_rows = 0
            unknown_rows = 0
            per_term = max(1, int(args.semantic_vocab_symbol_samples_per_term))
            origin_label = str(args.semantic_vocab_bootstrap_origin_label).strip() or "internal bootstrap root vocab"
            origin_label_lc = origin_label.lower()
            class_term_index = _semantic_term_index_map(class_names)
            active_term_lc = {re.sub(r"\s+", " ", str(t)).strip().lower() for t in active_extra_terms}
            unknown_term_seen: set = set()
            unknown_bootstrap_seen: set = set()
            unknown_other_seen: set = set()
            origin_row_counts: Dict[str, int] = {"bootstrap": 0, "official": 0, "synthetic": 0, "unknown": 0}
            gan_live_rows_added = 0
            gan_live_info: Dict[str, Any] = {
                "triggered": False,
                "harvested": 0,
                "used_items": 0,
                "expired_items": 0,
                "pool_before": int(len(gan_live_churn_pool)),
                "pool_after": int(len(gan_live_churn_pool)),
                "reason": "inactive",
            }

            def _origin_terms_for_symbol_key(key: str) -> List[str]:
                terms = list(symbol_pool_origin_terms_by_term.get(str(key), []))
                if str(key) in bootstrap_symbol_terms:
                    terms = list(terms) + [
                        "symbol pool bootstrap",
                        "internal bootstrap material",
                        origin_label,
                        f"{origin_label}::{str(key)}",
                    ]
                return _normalize_vocab_terms(terms)

            def _origin_source_for_row(origin_terms: Sequence[str]) -> str:
                keys = {re.sub(r"\s+", " ", str(x)).strip().lower() for x in origin_terms if str(x).strip()}
                if ("symbol pool bootstrap" in keys) or (origin_label_lc in keys):
                    return "bootstrap"
                if any(k.startswith(f"{origin_label_lc}::") for k in keys):
                    return "bootstrap"
                if ("symbol pool official" in keys) or ("official dataset pool" in keys):
                    return "official"
                if ("symbol pool synthetic" in keys) or ("synthetic semantic pool" in keys):
                    return "synthetic"
                return "unknown"

            for term in active_extra_terms:
                key = re.sub(r"\s+", " ", str(term)).strip().lower()
                rows = symbol_pool_by_term.get(key, [])
                if key == "gan image":
                    live_images, live_targets, live_info = _consume_live_gan_churn_rows(
                        cycle_seed=int(cycle_seed) + 401,
                        requested=int(per_term),
                    )
                    gan_live_info = dict(live_info)
                    for li, lt in zip(live_images, live_targets):
                        img = _image_any_to_rgb_chw01(li, image_size=int(shared_embed_image_size))
                        vec = np.asarray(lt, dtype=np.float32).reshape(-1)
                        if int(vec.size) != int(condition_num_classes):
                            continue
                        tone_terms = _semantic_tonal_tags_from_image(
                            image=img,
                            image_size=int(shared_embed_image_size),
                        )
                        if len(tone_terms) > 0:
                            tone_vec = np.asarray(
                                _semantic_condition_from_terms_runtime(
                                    terms=tone_terms,
                                    origin_terms=["live generator churn", "tonal tag enrichment"],
                                ),
                                dtype=np.float32,
                            ).reshape(-1)
                            if int(tone_vec.size) == int(vec.size):
                                vec = np.maximum(vec, tone_vec).astype(np.float32, copy=False)
                        images_out.append(img)
                        conds_out.append(np.clip(vec, 0.0, 1.0).astype(np.float32, copy=False))
                        embedded_rows += 1
                        gan_live_rows_added += 1
                        origin_row_counts["synthetic"] = int(origin_row_counts.get("synthetic", 0)) + 1
                    if int(len(live_images)) > 0:
                        matched += 1
                    # Never backfill GAN rows from static/synthetic symbol pools.
                    continue
                if len(rows) <= 0:
                    continue
                matched += 1
                if len(rows) <= per_term:
                    picks = list(range(len(rows)))
                else:
                    picks = rng_sym.choice(np.arange(len(rows), dtype=np.int64), size=per_term, replace=False).astype(np.int64).tolist()
                for pi in picks:
                    img = _image_any_to_rgb_chw01(rows[int(pi)], image_size=int(shared_embed_image_size))
                    images_out.append(img)
                    semantic_terms = _semantic_terms_with_tonal_tags(
                        terms=_semantic_tags_for_symbol_term(key),
                        image=img,
                        image_size=int(shared_embed_image_size),
                    )
                    semantic_terms = _semantic_enrich_generated_terms_with_noise_spectrum(
                        term_key=key,
                        terms=semantic_terms,
                        image=img,
                    )
                    origin_terms = _origin_terms_for_symbol_key(key)
                    vec = _semantic_condition_from_terms_runtime(
                        terms=semantic_terms,
                        origin_terms=origin_terms,
                    )
                    conds_out.append(np.asarray(vec, dtype=np.float32).reshape(-1))
                    embedded_rows += 1
                    source_key = _origin_source_for_row(origin_terms)
                    origin_row_counts[source_key] = int(origin_row_counts.get(source_key, 0)) + 1
            unknown_cap = max(0, int(args.semantic_vocab_unknown_label_rows_per_cycle))
            if int(unknown_cap) > 0:
                unknown_terms_all = [
                    str(k)
                    for k in symbol_pool_by_term.keys()
                    if re.sub(r"\s+", " ", str(k)).strip().lower() not in class_term_index
                    and re.sub(r"\s+", " ", str(k)).strip().lower() not in active_term_lc
                ]
                unknown_bootstrap = [
                    t
                    for t in unknown_terms_all
                    if re.sub(r"\s+", " ", str(t)).strip().lower() in bootstrap_symbol_terms
                ]
                unknown_other = [
                    t
                    for t in unknown_terms_all
                    if re.sub(r"\s+", " ", str(t)).strip().lower() not in bootstrap_symbol_terms
                ]
                rng_sym.shuffle(unknown_bootstrap)
                rng_sym.shuffle(unknown_other)
                unknown_terms = list(unknown_bootstrap) + list(unknown_other)
                for term_txt in unknown_terms:
                    if int(unknown_rows) >= int(unknown_cap):
                        break
                    key = re.sub(r"\s+", " ", str(term_txt)).strip().lower()
                    rows = symbol_pool_by_term.get(key, [])
                    if len(rows) <= 0:
                        continue
                    take = min(len(rows), max(1, int(min(per_term, unknown_cap - unknown_rows))))
                    if len(rows) <= take:
                        picks = list(range(len(rows)))[: int(take)]
                    else:
                        picks = rng_sym.choice(np.arange(len(rows), dtype=np.int64), size=int(take), replace=False).astype(np.int64).tolist()
                    for pi in picks:
                        if int(unknown_rows) >= int(unknown_cap):
                            break
                        img = _image_any_to_rgb_chw01(rows[int(pi)], image_size=int(shared_embed_image_size))
                        semantic_terms = _semantic_terms_with_tonal_tags(
                            terms=_semantic_tags_for_symbol_term(key),
                            image=img,
                            image_size=int(shared_embed_image_size),
                        )
                        semantic_terms = _semantic_enrich_generated_terms_with_noise_spectrum(
                            term_key=key,
                            terms=semantic_terms,
                            image=img,
                        )
                        origin_terms = _normalize_vocab_terms(
                            list(_origin_terms_for_symbol_key(key))
                            + [
                                "unknown semantic label",
                                "out of active semantic vocabulary",
                                f"unknown semantic term::{key}",
                            ]
                        )
                        vec = _semantic_condition_from_terms_runtime(
                            terms=semantic_terms,
                            origin_terms=origin_terms,
                        )
                        images_out.append(img)
                        conds_out.append(np.asarray(vec, dtype=np.float32).reshape(-1))
                        embedded_rows += 1
                        unknown_rows += 1
                        unknown_term_seen.add(str(key))
                        if str(key) in bootstrap_symbol_terms:
                            unknown_bootstrap_seen.add(str(key))
                        else:
                            unknown_other_seen.add(str(key))
                        source_key = _origin_source_for_row(origin_terms)
                        origin_row_counts[source_key] = int(origin_row_counts.get(source_key, 0)) + 1
            if len(conds_out) > 1:
                perm = rng_sym.permutation(np.arange(len(conds_out), dtype=np.int64)).astype(np.int64).tolist()
                images_out = [images_out[int(i)] for i in perm]
                conds_out = [conds_out[int(i)] for i in perm]
            info = {
                "enabled": True,
                "rows_added": int(len(conds_out)),
                "matched_terms": int(matched),
                "terms_total": int(len(active_extra_terms)),
                "tagged_rows": int(embedded_rows),
                "gan_live_rows_added": int(gan_live_rows_added),
                "gan_live_info": dict(gan_live_info),
                "unknown_rows_added": int(unknown_rows),
                "unknown_terms_sampled": sorted([str(x) for x in unknown_term_seen]),
                "unknown_bootstrap_terms_sampled": sorted([str(x) for x in unknown_bootstrap_seen]),
                "unknown_other_terms_sampled": sorted([str(x) for x in unknown_other_seen]),
                "origin_row_counts": {str(k): int(v) for k, v in origin_row_counts.items()},
                "origin_terms_mapped_terms": int(len(symbol_pool_origin_terms_by_term)),
            }
            target_stats = _semantic_active_target_stats(conds_out, threshold=0.5)
            info["target_active_min"] = int(target_stats.get("min", 0))
            info["target_active_mean"] = float(target_stats.get("mean", 0.0))
            info["target_active_p50"] = float(target_stats.get("p50", 0.0))
            info["target_active_max"] = int(target_stats.get("max", 0))
            return images_out, conds_out, info

        def _maybe_restore_gd_vocab_snapshot(reason: str):
            if (not bool(args.gd_vocab_library_enabled)) or (not bool(args.gd_vocab_library_autoload)):
                return {"loaded": False, "reason": "gd_library_disabled"}
            if generator is None or discriminator is None:
                return {"loaded": False, "reason": "gd_modules_unavailable"}
            if not str(active_gd_vocab_hash).strip():
                return {"loaded": False, "reason": "gd_vocab_hash_empty"}
            load_info = _load_gd_vocab_library_snapshot(
                library_dir=gd_vocab_library_dir,
                vocab_hash=str(active_gd_vocab_hash),
                generator=generator,
                discriminator=discriminator,
            )
            if bool(load_info.get("loaded", False)):
                _log(
                    "[gd-vocab-library] restored: "
                    f"reason={str(reason)} hash={str(active_gd_vocab_hash)} "
                    f"dir={load_info.get('dir', '')}"
                )
            return load_info

        def _activate_semantic_vocab(
            cycle_id: int,
            cycle_local: int,
            global_round: int,
            reason: str,
            allow_churn: bool,
            autoload_gd: bool,
        ):
            nonlocal active_extra_terms
            nonlocal semantic_vocab_churn_cursor
            nonlocal class_names, label_embedding_bank, label_texts, label_embedding_info
            nonlocal fake_label_vector_t, payload_images, payload_conditions
            nonlocal payload_conditioning_info, payload_symbol_aug_info, payload_flashcard_info
            nonlocal active_gd_vocab_hash, active_gd_vocab_profile
            nonlocal semantic_term_to_idx
            nonlocal semantic_kind_target_vectors
            nonlocal mix_condition_target_default
            nonlocal regurgitated_condition_target_default
            nonlocal regurgitated_churn_last_info
            active_extra_terms = _normalize_active_extra_terms_with_core(
                active_terms=active_extra_terms,
                core_terms=semantic_core_terms,
                total_slots=int(semantic_extra_count),
            )
            churn_info = {"changed": False, "replaced": 0}
            regurg_terms_used: List[str] = []
            regurg_info: Dict[str, Any] = {
                "enabled": bool(args.semantic_vocab_regurgitated_churn_enabled),
                "triggered": False,
                "pool_before": int(len(regurgitated_churn_pool)),
                "pool_after": int(len(regurgitated_churn_pool)),
                "used_items": 0,
                "expired_items": 0,
                "terms_added": 0,
                "reason": "inactive",
            }
            if bool(allow_churn) and bool(semantic_vocab_churn_runtime):
                regurg_terms_used, regurg_info = _consume_regurgitated_churn_terms(
                    cycle_seed=int(args.seed) + (int(cycle_local) * 1811) + (int(global_round) * 43),
                )
                pool_terms_effective = _normalize_vocab_terms(
                    list(semantic_vocab_pool_terms) + list(regurg_terms_used)
                )
                active_extra_terms, churn_info = _rotate_active_extra_terms(
                    active_terms=active_extra_terms,
                    pool_terms=pool_terms_effective,
                    replace_count=max(1, int(args.semantic_vocab_churn_replace_per_cycle)),
                    seed=int(args.seed) + (int(cycle_local) * 1733) + (int(global_round) * 37),
                    locked_prefix_count=int(semantic_core_extra_count),
                    churn_cursor=int(semantic_vocab_churn_cursor),
                    sweep_cycles=int(args.semantic_vocab_churn_sweep_cycles),
                )
                semantic_vocab_churn_cursor = int(churn_info.get("cursor_out", semantic_vocab_churn_cursor))
            regurgitated_churn_last_info = dict(regurg_info)
            active_semantic_names = list(supervised_class_names) + list(active_extra_terms)
            class_names = list(active_semantic_names)
            semantic_term_to_idx = _semantic_term_index_map(class_names)
            label_embedding_bank, label_texts, label_embedding_info = _build_label_embedding_bank(
                class_names=active_semantic_names,
                args=args,
                device=device,
            )
            label_embedding_info["supervised_num_classes"] = int(supervised_num_classes)
            label_embedding_info["semantic_num_classes"] = int(len(active_semantic_names))
            label_embedding_info["semantic_extra_count"] = int(semantic_extra_count)
            _semantic_text_condition_cache.clear()
            _semantic_supervised_condition_cache.clear()
            semantic_kind_keys = _semantic_kind_keys_for_class_names(active_semantic_names)
            semantic_kind_target_vectors = {}
            for kind_key in semantic_kind_keys:
                kk = re.sub(r"\s+", " ", str(kind_key)).strip().lower()
                if kk in ("mix", "mixed noise and signal"):
                    terms = ["mixed noise and signal", "mix", "signal", "noise"]
                elif kk == "regurgitated content":
                    terms = ["regurgitated content", "mixed noise and signal", "signal"]
                else:
                    terms = [str(kind_key)]
                semantic_kind_target_vectors[str(kind_key)] = _semantic_condition_from_terms_runtime(terms=terms)
            mix_condition_target_default = _semantic_condition_from_terms_runtime(
                terms=["mixed noise and signal", "signal", "noise"],
            )
            regurgitated_condition_target_default = _semantic_condition_from_terms_runtime(
                terms=["regurgitated content", "mixed noise and signal", "signal"],
            )
            _apply_label_embedding_bank_to_classifier(
                classifier=classifier,
                bank_np=label_embedding_bank,
                args=args,
            )
            if fake_feedback_enabled:
                try:
                    fake_vec_np = _encode_query_texts_for_bank(
                        query_texts=[str(fake_sentinel_label)],
                        bank_info=label_embedding_info,
                        args=args,
                        device=device,
                    )
                    if int(fake_vec_np.shape[0]) > 0:
                        fake_label_vector_t = torch.from_numpy(fake_vec_np[0]).to(device=device, dtype=torch.float32)
                except Exception:
                    pass

            payload_images = list(payload_images_base)
            if len(payload_conditions_supervised_base) > 0:
                payload_conditions, payload_conditioning_info = _expand_payload_conditions_with_semantic_bank(
                    payload_conditions=payload_conditions_supervised_base,
                    condition_num_classes=int(condition_num_classes),
                    supervised_num_classes=int(supervised_num_classes),
                    label_embedding_bank=label_embedding_bank,
                    semantic_target_temperature=8.0,
                )
                if len(payload_conditions) == len(payload_condition_terms_supervised_base):
                    payload_conditions = [
                        np.maximum(
                            np.asarray(payload_conditions[i], dtype=np.float32),
                            _semantic_condition_from_terms_runtime(
                                terms=payload_condition_terms_supervised_base[i],
                                origin_terms=[],
                            ),
                        ).astype(np.float32, copy=False)
                        for i in range(len(payload_conditions))
                    ]
            else:
                payload_conditions = []
                payload_conditioning_info = {
                    "expanded": False,
                    "reason": "no_supervised_payload_base",
                    "rows": 0,
                    "condition_dim": int(condition_num_classes),
                }
            sym_images, sym_conds, payload_symbol_aug_info = _build_symbol_payload_rows_for_active_terms(
                cycle_seed=int(args.seed) + (int(cycle_local) * 2903) + (int(global_round) * 53),
            )
            if len(sym_images) > 0 and len(sym_images) == len(sym_conds):
                payload_images.extend(sym_images)
                payload_conditions.extend(sym_conds)
            payload_flashcard_info = {"enabled": False, "rows_added": 0}
            if bool(args.semantic_vocab_reference_flashcards):
                def _flashcard_gan_provider(count: int) -> List[np.ndarray]:
                    imgs_live, _, _ = _consume_live_gan_churn_rows(
                        cycle_seed=int(args.seed) + (int(cycle_local) * 4441) + (int(global_round) * 83),
                        requested=max(1, int(count)),
                    )
                    return [np.asarray(x, dtype=np.float32) for x in imgs_live]

                flash_images, flash_conds, payload_flashcard_info = _build_reference_flashcard_payload_rows(
                    class_names=class_names,
                    condition_num_classes=int(condition_num_classes),
                    supervised_num_classes=int(supervised_num_classes),
                    payload_images_base=payload_images_base,
                    payload_conditions_supervised_base=payload_conditions_supervised_base,
                    symbol_pool_by_term=symbol_pool_by_term,
                    image_size=int(shared_embed_image_size),
                    per_term=max(0, int(args.semantic_vocab_reference_flashcards_per_term)),
                    seed=int(args.seed) + (int(cycle_local) * 3251) + (int(global_round) * 67),
                    condition_vector_builder=lambda terms, base_sup: _semantic_condition_from_terms_runtime(
                        terms=terms,
                        base_supervised_vec=base_sup,
                        origin_terms=[],
                    ),
                    gan_image_provider=_flashcard_gan_provider,
                )
                if len(flash_images) > 0 and len(flash_images) == len(flash_conds):
                    payload_images.extend(flash_images)
                    payload_conditions.extend(flash_conds)
            payload_target_knockout_info: Dict[str, Any] = {
                "enabled": bool(float(args.target_label_knockout_prob) > 0.0),
                "prob": float(max(0.0, min(1.0, float(args.target_label_knockout_prob)))),
                "rows_total": int(len(payload_conditions)),
                "rows_applied": 0,
                "labels_dropped": 0,
                "min_keep": int(max(0, int(args.target_label_knockout_min_keep))),
                "max_drop_frac": float(max(0.0, min(1.0, float(args.target_label_knockout_max_drop_frac)))),
            }
            if bool(payload_target_knockout_info.get("enabled", False)) and int(len(payload_conditions)) > 0:
                payload_conditions, payload_target_knockout_info = _label_knockout_rows_np(
                    rows=payload_conditions,
                    prob=float(args.target_label_knockout_prob),
                    seed=int(args.seed) + (int(cycle_local) * 4349) + (int(global_round) * 97),
                    min_keep=int(args.target_label_knockout_min_keep),
                    max_drop_frac=float(args.target_label_knockout_max_drop_frac),
                    threshold=0.5,
                )
                _log(
                    "[target-knockout] "
                    "scope=payload_active "
                    f"rows={int(payload_target_knockout_info.get('rows_total', 0))} "
                    f"applied={int(payload_target_knockout_info.get('rows_applied', 0))} "
                    f"dropped={int(payload_target_knockout_info.get('labels_dropped', 0))} "
                    f"prob={float(payload_target_knockout_info.get('prob', 0.0)):.3f}"
                )

            active_gd_vocab_hash, active_gd_vocab_profile = _compute_gd_vocab_hash(
                supervised_class_names=supervised_class_names,
                fixed_extra_terms=semantic_core_terms,
                condition_num_classes=int(condition_num_classes),
                args=args,
                active_extra_terms=active_extra_terms,
            )
            row = {
                "cycle": int(cycle_id),
                "cycle_local": int(cycle_local),
                "global_round": int(global_round),
                "reason": str(reason),
                "allow_churn": bool(allow_churn),
                "churn_changed": bool(churn_info.get("changed", False)),
                "churn_replaced": int(churn_info.get("replaced", 0)),
                "churn_cursor_in": int(churn_info.get("cursor_in", 0)),
                "churn_cursor_out": int(churn_info.get("cursor_out", 0)),
                "churn_replace_base": int(churn_info.get("replace_base", 0)),
                "churn_replace_auto": int(churn_info.get("replace_auto", 0)),
                "churn_candidate_terms": int(churn_info.get("candidate_terms", 0)),
                "churn_sweep_cycles": int(churn_info.get("sweep_cycles", 0)),
                "active_extra_terms": list(active_extra_terms),
                "vocab_hash": str(active_gd_vocab_hash),
                "payload_rows": int(len(payload_conditions)),
                "payload_images": int(len(payload_images)),
                "symbol_rows_added": int(payload_symbol_aug_info.get("rows_added", 0)),
                "symbol_gan_live_rows_added": int(payload_symbol_aug_info.get("gan_live_rows_added", 0)),
                "symbol_unknown_rows_added": int(payload_symbol_aug_info.get("unknown_rows_added", 0)),
                "symbol_target_active_min": int(payload_symbol_aug_info.get("target_active_min", 0)),
                "symbol_target_active_mean": float(payload_symbol_aug_info.get("target_active_mean", 0.0)),
                "symbol_target_active_max": int(payload_symbol_aug_info.get("target_active_max", 0)),
                "symbol_origin_row_counts": dict(payload_symbol_aug_info.get("origin_row_counts", {})),
                "symbol_unknown_terms_sampled": list(payload_symbol_aug_info.get("unknown_terms_sampled", [])),
                "flashcard_rows_added": int(payload_flashcard_info.get("rows_added", 0)),
                "flashcard_target_active_min": int(payload_flashcard_info.get("target_active_min", 0)),
                "flashcard_target_active_mean": float(payload_flashcard_info.get("target_active_mean", 0.0)),
                "flashcard_target_active_max": int(payload_flashcard_info.get("target_active_max", 0)),
                "target_knockout": dict(payload_target_knockout_info),
                "regurgitated_churn": dict(regurg_info),
                "regurgitated_terms_added": int(len(regurg_terms_used)),
                "gan_live_churn": dict(payload_symbol_aug_info.get("gan_live_info", {})),
            }
            semantic_churn_history.append(row)
            _log(
                "[semantic-vocab] active set: "
                f"cycle={int(cycle_id)} reason={str(reason)} "
                f"extras={int(len(active_extra_terms))} changed={1 if bool(churn_info.get('changed', False)) else 0} "
                f"replaced={int(churn_info.get('replaced', 0))} "
                f"replace_base={int(churn_info.get('replace_base', 0))} "
                f"replace_auto={int(churn_info.get('replace_auto', 0))} "
                f"cursor={int(churn_info.get('cursor_in', 0))}->{int(churn_info.get('cursor_out', 0))} "
                f"payload={int(len(payload_conditions))} symbols={int(payload_symbol_aug_info.get('rows_added', 0))} "
                f"gan_live={int(payload_symbol_aug_info.get('gan_live_rows_added', 0))} "
                f"unknown_symbols={int(payload_symbol_aug_info.get('unknown_rows_added', 0))} "
                f"symbol_targets(min/mean/max)="
                f"{int(payload_symbol_aug_info.get('target_active_min', 0))}/"
                f"{float(payload_symbol_aug_info.get('target_active_mean', 0.0)):.2f}/"
                f"{int(payload_symbol_aug_info.get('target_active_max', 0))} "
                f"flashcards={int(payload_flashcard_info.get('rows_added', 0))} "
                f"flash_targets(min/mean/max)="
                f"{int(payload_flashcard_info.get('target_active_min', 0))}/"
                f"{float(payload_flashcard_info.get('target_active_mean', 0.0)):.2f}/"
                f"{int(payload_flashcard_info.get('target_active_max', 0))} "
                f"regurg_used={int(regurg_info.get('used_items', 0))} "
                f"regurg_terms={int(regurg_info.get('terms_added', 0))} "
                f"regurg_pool={int(regurg_info.get('pool_after', len(regurgitated_churn_pool)))} "
                f"hash={str(active_gd_vocab_hash)}"
            )
            if bool(autoload_gd):
                _maybe_restore_gd_vocab_snapshot(reason=f"{reason}:cycle={int(cycle_id)}")

        _activate_semantic_vocab(
            cycle_id=0,
            cycle_local=0,
            global_round=int(len(orchestration_rows)),
            reason="startup",
            allow_churn=False,
            autoload_gd=False,
        )
        gestation_gate_samples_seen = 0
        gestation_gate_samples_required = 0
        total_gate_samples_seen = 0
        total_gate_samples_required = 0
        gestation_gate_eval_indices: Optional[List[int]] = None
        gestation_gate_eval_seed = int(args.seed) + 1777
        gestation_gate_preview_images: List[np.ndarray] = []
        gestation_gate_preview_targets: List[np.ndarray] = []
        total_gate_preview_images: List[np.ndarray] = []
        total_gate_preview_targets: List[np.ndarray] = []

        def _rebuild_semantic_gate_loaders(reason: str, cycle_seed: int, include_payload_val: bool = True):
            nonlocal gestation_stage_loader, gestation_stage_loader_count, gestation_stage_info
            nonlocal gestation_gate_loader, gestation_gate_loader_count, gestation_gate_info
            nonlocal berkeley_gate_loader, berkeley_gate_loader_count, berkeley_gate_schedule_info
            nonlocal gestation_gate_samples_seen, gestation_gate_samples_required
            nonlocal total_gate_samples_seen, total_gate_samples_required
            nonlocal gestation_gate_eval_indices
            nonlocal gestation_gate_preview_images, gestation_gate_preview_targets
            nonlocal total_gate_preview_images, total_gate_preview_targets
            if int(condition_num_classes) <= 0:
                gestation_stage_loader = None
                gestation_stage_loader_count = 0
                gestation_gate_loader = None
                berkeley_gate_loader = None
                gestation_stage_info = {"rows": 0, "selected_rows": 0, "reason": "condition_num_classes<=0"}
                gestation_gate_loader_count = 0
                berkeley_gate_loader_count = 0
                gestation_gate_info = {"rows": 0, "selected_rows": 0, "reason": "condition_num_classes<=0"}
                berkeley_gate_schedule_info = {"rows": 0, "selected_rows": 0, "reason": "condition_num_classes<=0"}
                gestation_gate_samples_seen = 0
                gestation_gate_samples_required = 0
                total_gate_samples_seen = 0
                total_gate_samples_required = 0
                total_gate_preview_images = []
                total_gate_preview_targets = []
                return

            gate_loader_batch = int(args.gate_berkeley_batch_size)
            if gate_loader_batch <= 0:
                gate_loader_batch = int(args.berkeley_refresh_batch_size)
            if gate_loader_batch <= 0:
                gate_loader_batch = int(args.classifier_batch_size)
            gate_loader_batch = max(1, int(gate_loader_batch))

            gestation_batch = int(args.gate_gestation_batch_size)
            if gestation_batch <= 0:
                gestation_batch = int(gate_loader_batch)
            gestation_batch = max(1, int(gestation_batch))

            origin_label = re.sub(r"\s+", " ", str(args.semantic_vocab_bootstrap_origin_label)).strip() or "internal bootstrap root vocab"
            gestation_images: List[np.ndarray] = []
            gestation_targets: List[np.ndarray] = []
            gestation_term_keys: set = set()
            gestation_term_counts: Dict[str, int] = {}
            gestation_term_indices: Dict[str, List[int]] = {}
            gestation_fallback_rebuild = False

            required_gestation_terms = sorted([str(x) for x in gestation_symbol_terms], key=lambda s: s.lower())
            if len(required_gestation_terms) <= 0:
                required_gestation_terms = sorted(
                    [re.sub(r"\s+", " ", str(x)).strip().lower() for x in _default_bootstrap_primitive_terms() if str(x).strip()],
                    key=lambda s: s.lower(),
                )

            def _append_gestation_rows_for_term(term_key: str, rows: Sequence[np.ndarray]):
                key = re.sub(r"\s+", " ", str(term_key)).strip().lower()
                if not key or len(rows) <= 0:
                    return
                origin_terms = _normalize_vocab_terms(
                    [
                        "symbol pool bootstrap",
                        "internal bootstrap material",
                        str(origin_label),
                        f"{str(origin_label)}::{str(key)}",
                    ]
                )
                for row_img in rows:
                    img_rgb = _image_any_to_rgb_chw01(row_img, image_size=int(shared_embed_image_size))
                    semantic_terms = _semantic_terms_with_tonal_tags(
                        terms=_semantic_tags_for_symbol_term(str(key)),
                        image=img_rgb,
                        image_size=int(shared_embed_image_size),
                    )
                    semantic_terms = _semantic_enrich_generated_terms_with_noise_spectrum(
                        term_key=str(key),
                        terms=semantic_terms,
                        image=img_rgb,
                    )
                    gestation_images.append(img_rgb)
                    gestation_targets.append(
                        np.asarray(
                            _semantic_condition_from_terms_runtime(
                                terms=semantic_terms,
                                origin_terms=origin_terms,
                            ),
                            dtype=np.float32,
                        ).reshape(-1)
                    )
                    gestation_term_counts[key] = int(gestation_term_counts.get(key, 0)) + 1
                    gestation_term_indices.setdefault(str(key), []).append(int(len(gestation_images) - 1))
                gestation_term_keys.add(str(key))

            for term_key in required_gestation_terms:
                rows = symbol_pool_by_term.get(str(term_key), [])
                _append_gestation_rows_for_term(term_key=term_key, rows=rows)
            missing_terms = [
                str(t) for t in required_gestation_terms
                if int(gestation_term_counts.get(str(t), 0)) <= 0
            ]
            if len(missing_terms) > 0:
                gestation_fallback_rebuild = True
                fallback_bootstrap, _ = _build_internal_bootstrap_symbol_pool(
                    data_root=str(args.semantic_vocab_symbol_pool_root),
                    image_size=int(shared_embed_image_size),
                    seed=int(cycle_seed) + 149,
                    max_samples_per_term=max(4, int(args.semantic_vocab_symbol_samples_per_term)),
                    origin_label=str(args.semantic_vocab_bootstrap_origin_label),
                )
                for term_key in list(missing_terms):
                    rows = fallback_bootstrap.get(str(term_key), [])
                    if len(rows) <= 0 and len(fallback_bootstrap) > 0:
                        rows = fallback_bootstrap.get(str(term_key).strip().lower(), [])
                    if len(rows) <= 0:
                        continue
                    _append_gestation_rows_for_term(term_key=term_key, rows=rows)
            missing_terms_after = [
                str(t) for t in required_gestation_terms
                if int(gestation_term_counts.get(str(t), 0)) <= 0
            ]
            _log(
                "[gestation-archetypes] "
                "source=fixed_core_vocab "
                f"terms={int(len(required_gestation_terms))} "
                f"rows={int(len(gestation_images))} "
                f"missing={int(len(missing_terms_after))} "
                f"fallback_rebuild={1 if bool(gestation_fallback_rebuild) else 0}"
            )
            if len(missing_terms_after) > 0:
                raise RuntimeError(
                    "Gestation bootstrap coverage missing required terms: "
                    + ",".join([str(x) for x in missing_terms_after[:24]])
                )
            gestation_total_rows = int(len(gestation_images))
            if int(gestation_total_rows) <= 0:
                raise RuntimeError("No gestation rows available for Stage 1 / Gate 1.")
            expected_gate_dim = max(1, int(condition_num_classes))
            gestation_dims_all = sorted({int(np.asarray(row, dtype=np.float32).reshape(-1).size) for row in gestation_targets})
            bad_gestation_dims = [int(d) for d in gestation_dims_all if int(d) != int(expected_gate_dim)]
            if len(bad_gestation_dims) > 0:
                raise RuntimeError(
                    "Gestation gate targets must already be full semantic width: "
                    f"expected={int(expected_gate_dim)} got_dims={bad_gestation_dims}"
                )
            val_idx_seeded: List[int] = []
            for term_key in required_gestation_terms:
                key = re.sub(r"\s+", " ", str(term_key)).strip().lower()
                if not key:
                    continue
                term_rows = [int(i) for i in list(gestation_term_indices.get(str(key), [])) if 0 <= int(i) < int(gestation_total_rows)]
                if int(len(term_rows)) <= 0:
                    continue
                if int(len(term_rows)) == 1:
                    val_idx_seeded.append(int(term_rows[0]))
                    continue
                term_seed = int(hashlib.sha256(f"{int(gestation_gate_eval_seed)}|{str(key)}".encode("utf-8")).hexdigest()[:8], 16)
                term_rng = np.random.default_rng(int(term_seed))
                term_arr = np.asarray(term_rows, dtype=np.int64)
                term_rng.shuffle(term_arr)
                val_take = max(1, int(round(float(term_arr.size) * 0.25)))
                val_take = max(1, min(int(val_take), int(term_arr.size) - 1))
                val_idx_seeded.extend([int(i) for i in term_arr[: int(val_take)].tolist()])
            gestation_gate_eval_indices = sorted(set(int(i) for i in val_idx_seeded if 0 <= int(i) < int(gestation_total_rows)))
            valid_gestation_val_idx = [
                int(i) for i in list(gestation_gate_eval_indices)
                if 0 <= int(i) < int(gestation_total_rows)
            ]
            if int(len(valid_gestation_val_idx)) <= 0:
                raise RuntimeError("Gestation validation split produced zero rows; cannot evaluate Gate 1.")
            val_idx_set = {int(i) for i in valid_gestation_val_idx}
            gestation_train_idx = [int(i) for i in range(int(gestation_total_rows)) if int(i) not in val_idx_set]
            if int(len(gestation_train_idx)) <= 0:
                raise RuntimeError(
                    "Stage 1 gestation split produced zero training rows; "
                    "increase gestation source rows or reduce gestation validation coverage."
                )
            gestation_train_images = [np.asarray(gestation_images[int(i)], dtype=np.float32) for i in gestation_train_idx]
            gestation_train_targets = [np.asarray(gestation_targets[int(i)], dtype=np.float32).reshape(-1) for i in gestation_train_idx]
            gestation_target_knockout_info: Dict[str, Any] = {
                "enabled": bool(float(args.target_label_knockout_prob) > 0.0),
                "prob": float(max(0.0, min(1.0, float(args.target_label_knockout_prob)))),
                "rows_total": int(len(gestation_train_targets)),
                "rows_applied": 0,
                "labels_dropped": 0,
                "min_keep": int(max(0, int(args.target_label_knockout_min_keep))),
                "max_drop_frac": float(max(0.0, min(1.0, float(args.target_label_knockout_max_drop_frac)))),
            }
            if bool(gestation_target_knockout_info.get("enabled", False)) and int(len(gestation_train_targets)) > 0:
                gestation_train_targets, gestation_target_knockout_info = _label_knockout_rows_np(
                    rows=gestation_train_targets,
                    prob=float(args.target_label_knockout_prob),
                    seed=int(args.seed) + int(cycle_seed) + 22241,
                    min_keep=int(args.target_label_knockout_min_keep),
                    max_drop_frac=float(args.target_label_knockout_max_drop_frac),
                    threshold=0.5,
                )
                _log(
                    "[target-knockout] "
                    "scope=gestation_stage1_train "
                    f"rows={int(gestation_target_knockout_info.get('rows_total', 0))} "
                    f"applied={int(gestation_target_knockout_info.get('rows_applied', 0))} "
                    f"dropped={int(gestation_target_knockout_info.get('labels_dropped', 0))} "
                    f"prob={float(gestation_target_knockout_info.get('prob', 0.0)):.3f}"
                )
            gestation_eval_images = [np.asarray(gestation_images[int(i)], dtype=np.float32) for i in valid_gestation_val_idx]
            gestation_eval_targets = [np.asarray(gestation_targets[int(i)], dtype=np.float32).reshape(-1) for i in valid_gestation_val_idx]
            gest_train_stats = _semantic_active_target_stats(gestation_train_targets, threshold=0.5)
            gest_val_stats = _semantic_active_target_stats(gestation_eval_targets, threshold=0.5)
            _log(
                "[gestation-target-stats] "
                f"train_rows={int(gest_train_stats.get('rows', 0))} "
                f"train_min/mean/max={int(gest_train_stats.get('min', 0))}/"
                f"{float(gest_train_stats.get('mean', 0.0)):.2f}/"
                f"{int(gest_train_stats.get('max', 0))} "
                f"val_rows={int(gest_val_stats.get('rows', 0))} "
                f"val_min/mean/max={int(gest_val_stats.get('min', 0))}/"
                f"{float(gest_val_stats.get('mean', 0.0)):.2f}/"
                f"{int(gest_val_stats.get('max', 0))}"
            )
            raw_gestation_train_rows = int(len(gestation_train_images))
            raw_gestation_val_rows = int(len(gestation_eval_images))
            gestation_train_target_rows_req = int(args.semantic_vocab_gestation_train_target_samples)
            gestation_train_target_rows = int(raw_gestation_train_rows)
            if int(gestation_train_target_rows_req) > 0:
                gestation_train_target_rows = max(int(raw_gestation_train_rows), int(gestation_train_target_rows_req))
            gestation_val_target_rows_req = int(args.semantic_vocab_gestation_val_target_samples)
            if int(gestation_val_target_rows_req) > 0:
                gestation_val_target_rows = max(int(raw_gestation_val_rows), int(gestation_val_target_rows_req))
            else:
                gestation_val_target_rows = max(
                    int(raw_gestation_val_rows),
                    int(math.ceil(float(max(1, int(gestation_train_target_rows))) * 0.20)),
                )
            gestation_stage_dataset = _BootstrapExpandedDataset(
                images=gestation_train_images,
                targets=gestation_train_targets,
                total_rows=int(gestation_train_target_rows),
                seed=int(args.seed) + 1703 + int(cycle_seed),
                augment=bool(int(gestation_train_target_rows) > int(raw_gestation_train_rows)),
                expected_target_dim=int(expected_gate_dim),
                semantic_term_to_idx=semantic_term_to_idx,
                augment_apply_terms=True,
            )
            gestation_gate_dataset = _BootstrapExpandedDataset(
                images=gestation_eval_images,
                targets=gestation_eval_targets,
                total_rows=int(gestation_val_target_rows),
                seed=int(gestation_gate_eval_seed) + int(cycle_seed),
                augment=bool(int(gestation_val_target_rows) > int(raw_gestation_val_rows)),
                expected_target_dim=int(expected_gate_dim),
                semantic_term_to_idx=semantic_term_to_idx,
                augment_apply_terms=True,
            )
            gestation_gate_preview_images = []
            gestation_gate_preview_targets = []
            gest_preview_cap = min(int(len(gestation_gate_dataset)), 64)
            for preview_i in range(int(gest_preview_cap)):
                try:
                    gi, gt = gestation_gate_dataset[int(preview_i)]
                    if torch.is_tensor(gi):
                        gi_np = gi.detach().to(torch.float32).cpu().numpy().astype(np.float32, copy=False)
                    else:
                        gi_np = np.asarray(gi, dtype=np.float32)
                    if torch.is_tensor(gt):
                        gt_np = gt.detach().to(torch.float32).cpu().numpy().astype(np.float32, copy=False).reshape(-1)
                    else:
                        gt_np = np.asarray(gt, dtype=np.float32).reshape(-1)
                    gestation_gate_preview_images.append(np.asarray(gi_np, dtype=np.float32))
                    gestation_gate_preview_targets.append(np.asarray(gt_np, dtype=np.float32).reshape(-1))
                except Exception:
                    continue
            gestation_stage_batch = int(args.berkeley_refresh_batch_size)
            if int(gestation_stage_batch) <= 0:
                gestation_stage_batch = int(gate_loader_batch)
            gestation_stage_batch = max(1, int(gestation_stage_batch))
            gestation_stage_loader, gestation_stage_loader_count = _build_gate_loader_from_dataset(
                dataset=gestation_stage_dataset,
                batch_size=int(gestation_stage_batch),
                num_workers=max(0, int(args.berkeley_refresh_workers)),
                device=device,
                seed=int(args.seed) + 1703,
                max_samples=0,
                ordered_indices=None,
                persistent_workers=bool(args.loader_persistent_workers),
                prefetch_factor=int(args.loader_prefetch_factor),
            )
            if gestation_stage_loader is None or int(gestation_stage_loader_count) <= 0:
                raise RuntimeError("Stage 1 gestation loader is empty; cannot run gestation training.")
            if int(len(gestation_gate_preview_targets)) > 0:
                active_counts = np.asarray(
                    [
                        int(np.count_nonzero(np.asarray(row, dtype=np.float32).reshape(-1) >= 0.5))
                        for row in gestation_gate_preview_targets
                    ],
                    dtype=np.int32,
                )
                _log(
                    "[gestation-target-sparsity] "
                    f"rows={int(active_counts.size)} "
                    f"min={int(active_counts.min())} "
                    f"p50={int(np.percentile(active_counts, 50))} "
                    f"max={int(active_counts.max())}"
                )
            gestation_gate_order = [int(i) for i in range(int(len(gestation_gate_dataset)))]
            gestation_val_max_eff = int(args.gate_gestation_val_max)
            if int(gestation_val_max_eff) > 0 and int(gestation_val_max_eff) < int(len(gestation_gate_order)):
                _log(
                    "[gestation-gate] overriding --gate-gestation-val-max "
                    f"from {int(gestation_val_max_eff)} to {int(len(gestation_gate_order))} "
                    "to preserve complete primitive-tag coverage."
                )
                gestation_val_max_eff = int(len(gestation_gate_order))
            gestation_dims = sorted({int(np.asarray(row, dtype=np.float32).reshape(-1).size) for row in gestation_eval_targets})
            gestation_gate_loader, gestation_gate_loader_count = _build_gate_loader_from_dataset(
                dataset=gestation_gate_dataset,
                batch_size=int(gestation_batch),
                num_workers=max(0, int(args.berkeley_refresh_workers)),
                device=device,
                seed=int(gestation_gate_eval_seed),
                max_samples=int(gestation_val_max_eff),
                ordered_indices=gestation_gate_order,
                persistent_workers=bool(args.loader_persistent_workers),
                prefetch_factor=int(args.loader_prefetch_factor),
            )
            if gestation_gate_loader is None or int(gestation_gate_loader_count) <= 0:
                raise RuntimeError("Gate 1 gestation validation loader is empty; cannot evaluate Gate 1.")
            gestation_stage_info = {
                "rows": int(raw_gestation_train_rows),
                "selected_rows": int(gestation_stage_loader_count),
                "expanded_rows": int(gestation_train_target_rows),
                "batch_size": int(gestation_stage_batch),
                "target_dim": int(expected_gate_dim),
                "reason": str(reason),
            }
            gestation_gate_info = {
                "rows": int(len(gestation_images)),
                "selected_rows": int(gestation_gate_loader_count),
                "validation_rows": int(raw_gestation_val_rows),
                "validation_rows_expanded": int(gestation_val_target_rows),
                "training_rows": int(raw_gestation_train_rows),
                "training_rows_expanded": int(gestation_train_target_rows),
                "batch_size": int(gestation_batch),
                "target_dim": int(expected_gate_dim),
                "reason": str(reason),
            }
            gestation_gate_samples_seen = 0
            gestation_gate_samples_required = int(gestation_gate_loader_count)
            _log(
                "[gestation-split] "
                f"train={int(raw_gestation_train_rows)} "
                f"train_expanded={int(gestation_train_target_rows)} "
                f"gate_val={int(raw_gestation_val_rows)} "
                f"gate_val_expanded={int(gestation_val_target_rows)} "
                f"terms={int(len(required_gestation_terms))}"
            )

            payload_enabled = bool(include_payload_val)
            payload_semantic_cache_hit = False
            payload_semantic_cache_path = ""
            payload_targets_sched_np: Optional[np.ndarray] = None
            payload_rows_in_total = 0
            payload_stage2_picks_np = np.zeros((0,), dtype=np.int64)
            payload_stage2_terms: List[List[str]] = []
            if bool(payload_enabled) and int(len(payload_gate_val_terms_base)) > 0:
                if int(len(payload_gate_val_indices_base)) != int(len(payload_gate_val_terms_base)):
                    raise RuntimeError(
                        "Payload gate deck index/terms length mismatch: "
                        f"indices={int(len(payload_gate_val_indices_base))} terms={int(len(payload_gate_val_terms_base))}"
                    )
                active_vocab_lc = {
                    re.sub(r"\s+", " ", str(x)).strip().lower()
                    for x in class_names
                    if re.sub(r"\s+", " ", str(x)).strip()
                }
                active_churn_terms_raw = _normalize_vocab_terms([str(x) for x in list(active_extra_terms)])
                active_churn_terms_lc = []
                for t in active_churn_terms_raw:
                    tk = re.sub(r"\s+", " ", str(t)).strip().lower()
                    if not tk:
                        continue
                    if tk not in active_vocab_lc:
                        continue
                    if re.fullmatch(r"semantic slot \d+", tk):
                        continue
                    active_churn_terms_lc.append(str(tk))
                active_churn_terms_lc = list(dict.fromkeys(active_churn_terms_lc))
                if int(len(active_churn_terms_lc)) <= 0:
                    raise RuntimeError(
                        "Stage-2 payload gate requires at least one active churn term present in current vocabulary; "
                        "none were available after normalization."
                    )
                supervised_terms_lc = []
                for sup_name in supervised_class_names:
                    sk = re.sub(r"\s+", " ", str(sup_name)).strip().lower()
                    if not sk:
                        continue
                    if sk not in active_vocab_lc:
                        raise RuntimeError(
                            "Stage-2 payload gate requires supervised Berkeley labels to remain inside active vocabulary. "
                            f"Missing label term: {str(sup_name)}"
                        )
                    supervised_terms_lc.append(str(sk))
                active_stage2_terms_lc = list(dict.fromkeys(list(active_vocab_lc)))
                if int(len(active_stage2_terms_lc)) <= 0:
                    raise RuntimeError(
                        "Stage-2 payload gate has no active terms after resolving current active vocabulary."
                    )
                payload_labels_cache_path = str(payload_gate_val_info.get("labels_path", "")).strip()
                payload_labels_mm = None
                if str(payload_labels_cache_path):
                    try:
                        payload_labels_mm = np.load(str(payload_labels_cache_path), mmap_mode="r")
                    except Exception:
                        payload_labels_mm = None
                if payload_labels_mm is None:
                    raise RuntimeError(
                        "Stage-2 payload gate requires original supervised labels cache for strict anchoring, "
                        "but labels cache could not be loaded."
                    )
                selected_local_rows, selected_global_picks, payload_stage2_terms, stage2_row_sched_info = _schedule_payload_gate_rows_by_active_terms(
                    payload_terms_rows=payload_gate_val_terms_base,
                    payload_global_picks=payload_gate_val_indices_base,
                    payload_labels_mm=payload_labels_mm,
                    supervised_class_names=supervised_class_names,
                    active_terms_lc=active_stage2_terms_lc,
                    active_vocab_lc=list(active_vocab_lc),
                    require_all_supervised_active=False,
                    allow_unmapped_rows=True,
                )
                if int(len(selected_local_rows)) <= 0:
                    raise RuntimeError(
                        "Stage-2 payload gate found zero rows after deterministic active-term scheduling. "
                        f"active_churn_terms={int(len(active_churn_terms_lc))} "
                        f"active_stage2_terms={int(len(active_stage2_terms_lc))} "
                        f"rows_skipped_no_active={int(stage2_row_sched_info.get('rows_skipped_no_active', 0))} "
                        f"rows_skipped_inactive_supervised={int(stage2_row_sched_info.get('rows_skipped_inactive_supervised', 0))} "
                        f"payload_rows={int(len(payload_gate_val_terms_base))}"
                    )
                _log(
                    "[stage2-row-scheduler] "
                    f"mode={str(stage2_row_sched_info.get('mode', 'unknown'))} "
                    f"rows_total={int(stage2_row_sched_info.get('rows_total', 0))} "
                    f"selected={int(stage2_row_sched_info.get('rows_selected', 0))} "
                    f"active_terms={int(stage2_row_sched_info.get('active_terms', 0))} "
                    f"skipped_no_active={int(stage2_row_sched_info.get('rows_skipped_no_active', 0))} "
                    f"skipped_inactive_supervised={int(stage2_row_sched_info.get('rows_skipped_inactive_supervised', 0))}"
                )
                if int(stage2_row_sched_info.get("berkeley_rows_selected", 0)) > 0:
                    _log(
                        "[stage2-berkeley-targets] "
                        f"rows={int(stage2_row_sched_info.get('berkeley_rows_selected', 0))} "
                        f"min/mean/max={int(stage2_row_sched_info.get('berkeley_sup_min', 0))}/"
                        f"{float(stage2_row_sched_info.get('berkeley_sup_mean', 0.0)):.2f}/"
                        f"{int(stage2_row_sched_info.get('berkeley_sup_max', 0))} "
                        f"skipped_inactive_supervised={int(stage2_row_sched_info.get('rows_skipped_inactive_supervised', 0))}"
                    )
                payload_stage2_picks_np = np.asarray(selected_global_picks, dtype=np.int64)
                cache_dir_txt = str(payload_gate_val_info.get("cache_dir", "")).strip()
                cache_identity = {
                    "supervised_class_names": [str(x) for x in supervised_class_names],
                    "semantic_core_terms": [str(x) for x in semantic_core_terms],
                    "condition_num_classes": int(condition_num_classes),
                    "label_embedding_backend": str(label_embedding_info.get("backend_used", "")),
                    "label_embedding_model": str(label_embedding_info.get("model_name", "")),
                    "label_embedding_dim": int(label_embedding_info.get("dim", 0)),
                    "semantic_label_mode": "presence_v3_supervised_anchor",
                    "active_churn_terms_lc": list(active_churn_terms_lc),
                    "active_stage2_terms_lc": list(active_stage2_terms_lc),
                    "hard_wipe_nonce": str(semantic_cache_nonce),
                }
                class_blob = json.dumps(cache_identity, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
                terms_blob = json.dumps(payload_stage2_terms, ensure_ascii=True, separators=(",", ":"))
                cache_key = hashlib.sha256((class_blob + "|" + terms_blob).encode("utf-8")).hexdigest()[:24]
                if str(cache_dir_txt):
                    sem_dir = Path(str(cache_dir_txt)) / "semantic_gate_labels"
                    sem_dir.mkdir(parents=True, exist_ok=True)
                    cache_file = sem_dir / f"valdeck_semantic_{cache_key}.npy"
                    payload_semantic_cache_path = str(cache_file)
                    if cache_file.exists():
                        try:
                            cached = np.load(str(cache_file), mmap_mode="r")
                            if (
                                int(getattr(cached, "ndim", 0)) == 2
                                and int(cached.shape[0]) == int(len(payload_stage2_picks_np))
                                and int(cached.shape[1]) == int(expected_gate_dim)
                            ):
                                cached_arr = np.asarray(cached, dtype=np.float32)
                                cached_valid = True
                                if payload_labels_mm is not None:
                                    for out_ri in range(int(cached_arr.shape[0])):
                                        src_idx = int(payload_stage2_picks_np[int(out_ri)])
                                        if not (0 <= int(src_idx) < int(payload_labels_mm.shape[0])):
                                            cached_valid = False
                                            break
                                        base_sup = np.asarray(payload_labels_mm[int(src_idx)], dtype=np.float32).reshape(-1)
                                        sup_take = min(int(supervised_num_classes), int(base_sup.size), int(expected_gate_dim))
                                        if int(sup_take) <= 0:
                                            continue
                                        base_sup_clip = np.clip(np.asarray(base_sup[: int(sup_take)], dtype=np.float32), 0.0, 1.0)
                                        vec_sup = np.clip(np.asarray(cached_arr[int(out_ri), : int(sup_take)], dtype=np.float32), 0.0, 1.0)
                                        missing_mask = (base_sup_clip >= 0.5) & (vec_sup < 0.5)
                                        if bool(np.any(missing_mask)):
                                            cached_valid = False
                                            break
                                if bool(cached_valid):
                                    payload_targets_sched_np = np.asarray(cached_arr, dtype=np.float32)
                                    payload_semantic_cache_hit = True
                                else:
                                    payload_semantic_cache_hit = False
                                    _log(
                                        "[semantic-gate-cache] stale cache dropped: "
                                        f"path={str(cache_file)} reason=missing_supervised_anchor"
                                    )
                        except Exception:
                            payload_semantic_cache_hit = False

                if not bool(payload_semantic_cache_hit):
                    if payload_labels_mm is None:
                        raise RuntimeError(
                            "Stage-2 payload gate requires original supervised labels cache for strict anchoring, "
                            "but labels cache could not be loaded."
                        )
                    mat = np.zeros((int(len(payload_stage2_terms)), int(expected_gate_dim)), dtype=np.float32)
                    for out_ri, terms in enumerate(payload_stage2_terms):
                        src_idx = int(payload_stage2_picks_np[int(out_ri)])
                        if not (0 <= int(src_idx) < int(payload_labels_mm.shape[0])):
                            raise RuntimeError(
                                "Stage-2 payload gate supervised-label index out of bounds: "
                                f"row={int(out_ri)} pick={int(src_idx)} labels_rows={int(payload_labels_mm.shape[0])}"
                            )
                        base_sup = np.asarray(payload_labels_mm[int(src_idx)], dtype=np.float32).reshape(-1)
                        if int(base_sup.size) <= 0:
                            raise RuntimeError(
                                "Stage-2 payload gate supervised label row is empty: "
                                f"row={int(out_ri)} pick={int(src_idx)}"
                            )
                        vec = np.asarray(
                            _semantic_condition_from_terms_runtime(
                                terms=terms,
                                base_supervised_vec=base_sup,
                                origin_terms=[],
                            ),
                            dtype=np.float32,
                        ).reshape(-1)
                        if int(vec.size) != int(expected_gate_dim):
                            raise RuntimeError(
                                "Payload gate semantic vector width mismatch: "
                                f"row={int(out_ri)} got={int(vec.size)} expected={int(expected_gate_dim)}"
                            )
                        sup_take = min(int(supervised_num_classes), int(base_sup.size), int(expected_gate_dim))
                        if int(sup_take) > 0:
                            base_sup_clip = np.clip(np.asarray(base_sup[: int(sup_take)], dtype=np.float32), 0.0, 1.0)
                            vec_sup = np.clip(np.asarray(vec[: int(sup_take)], dtype=np.float32), 0.0, 1.0)
                            missing_mask = (base_sup_clip >= 0.5) & (vec_sup < 0.5)
                            if bool(np.any(missing_mask)):
                                raise RuntimeError(
                                    "Stage-2 payload gate semantic vector dropped original supervised labels; "
                                    f"row={int(out_ri)} pick={int(src_idx)}"
                                )
                        mat[int(out_ri), :] = np.clip(vec, 0.0, 1.0).astype(np.float32, copy=False)
                    payload_targets_sched_np = np.asarray(mat, dtype=np.float32)
                    if str(payload_semantic_cache_path):
                        try:
                            np.save(str(payload_semantic_cache_path), np.asarray(payload_targets_sched_np, dtype=np.float32))
                        except Exception:
                            pass
                if int(payload_targets_sched_np.shape[0]) != int(len(payload_stage2_picks_np)):
                    raise RuntimeError(
                        "Payload gate semantic cache row mismatch: "
                        f"labels={int(payload_targets_sched_np.shape[0])} picks={int(len(payload_stage2_picks_np))}"
                    )
                payload_rows_in_total = int(payload_targets_sched_np.shape[0])

            stage2_symbol_images: List[np.ndarray] = []
            stage2_symbol_targets: List[np.ndarray] = []
            stage2_symbol_term_keys: set = set()
            if bool(payload_enabled) and int(len(symbol_pool_by_term)) > 0:
                for term_key in sorted([str(x) for x in symbol_pool_by_term.keys()], key=lambda s: s.lower()):
                    key = re.sub(r"\s+", " ", str(term_key)).strip().lower()
                    if not key or key in gestation_symbol_terms:
                        continue
                    rows = symbol_pool_by_term.get(key, [])
                    if len(rows) <= 0:
                        continue
                    stage2_symbol_term_keys.add(str(key))
                    origin_terms = _normalize_vocab_terms(
                        list(symbol_pool_origin_terms_by_term.get(str(key), []))
                        + [
                            "symbol pool stage2",
                            "non-gestation semantic pool",
                            f"symbol pool stage2::{str(key)}",
                        ]
                    )
                    for row_img in rows:
                        img_rgb = _image_any_to_rgb_chw01(row_img, image_size=int(shared_embed_image_size))
                        semantic_terms = _semantic_terms_with_tonal_tags(
                            terms=_semantic_tags_for_symbol_term(str(key)),
                            image=img_rgb,
                            image_size=int(shared_embed_image_size),
                        )
                        semantic_terms = _semantic_enrich_generated_terms_with_noise_spectrum(
                            term_key=str(key),
                            terms=semantic_terms,
                            image=img_rgb,
                        )
                        stage2_symbol_images.append(img_rgb)
                        stage2_symbol_targets.append(
                            np.asarray(
                                _semantic_condition_from_terms_runtime(
                                    terms=semantic_terms,
                                    origin_terms=origin_terms,
                                ),
                                dtype=np.float32,
                            ).reshape(-1)
                        )
            stage2_symbol_targets_sched_np = (
                np.stack([np.asarray(y, dtype=np.float32).reshape(-1) for y in stage2_symbol_targets], axis=0).astype(np.float32, copy=False)
                if int(len(stage2_symbol_targets)) > 0
                else np.zeros((0, int(expected_gate_dim)), dtype=np.float32)
            )
            stage2_symbol_dims = sorted({int(np.asarray(row, dtype=np.float32).reshape(-1).size) for row in stage2_symbol_targets})
            bad_stage2_dims = [int(d) for d in stage2_symbol_dims if int(d) != int(expected_gate_dim)]
            if len(bad_stage2_dims) > 0:
                raise RuntimeError(
                    "Stage-2 symbol targets must already be full semantic width: "
                    f"expected={int(expected_gate_dim)} got_dims={bad_stage2_dims}"
                )
            stage2_target_stats = _semantic_active_target_stats(stage2_symbol_targets, threshold=0.5)
            if int(stage2_target_stats.get("rows", 0)) > 0:
                _log(
                    "[stage2-symbol-target-stats] "
                    f"rows={int(stage2_target_stats.get('rows', 0))} "
                    f"min/mean/max={int(stage2_target_stats.get('min', 0))}/"
                    f"{float(stage2_target_stats.get('mean', 0.0)):.2f}/"
                    f"{int(stage2_target_stats.get('max', 0))}"
                )
            stage2_symbol_rows_in_total = int(stage2_symbol_targets_sched_np.shape[0])

            gestation_targets_sched_np = (
                np.stack([np.asarray(y, dtype=np.float32).reshape(-1) for y in gestation_eval_targets], axis=0).astype(np.float32, copy=False)
                if int(len(gestation_eval_targets)) > 0
                else np.zeros((0, int(expected_gate_dim)), dtype=np.float32)
            )
            total_target_parts: List[np.ndarray] = []
            if payload_targets_sched_np is not None and int(payload_targets_sched_np.shape[0]) > 0:
                total_target_parts.append(np.asarray(payload_targets_sched_np, dtype=np.float32))
            if int(len(total_target_parts)) <= 0:
                total_targets_sched_np = np.zeros((0, int(expected_gate_dim)), dtype=np.float32)
            elif int(len(total_target_parts)) == 1:
                total_targets_sched_np = np.asarray(total_target_parts[0], dtype=np.float32)
            else:
                total_targets_sched_np = np.concatenate(total_target_parts, axis=0).astype(np.float32, copy=False)

            _log(
                "[semantic-gate-width] "
                f"target_dim={int(expected_gate_dim)} "
                f"payload_enabled={1 if bool(payload_enabled) else 0} "
                f"payload_terms_rows={int(len(payload_gate_val_terms_base))} "
                f"payload_rows_active={int(payload_rows_in_total)} "
                f"stage2_symbol_rows={int(stage2_symbol_rows_in_total)} "
                f"stage2_symbol_terms={int(len(stage2_symbol_term_keys))} "
                f"payload_semantic_cache_hit={1 if bool(payload_semantic_cache_hit) else 0} "
                f"gestation_dims={gestation_dims}"
            )
            total_datasets: List[Dataset] = []
            if bool(payload_enabled) and int(payload_rows_in_total) > 0:
                payload_images_path = str(payload_gate_val_info.get("images_path", "")).strip()
                if (not payload_images_path) or (not str(payload_semantic_cache_path).strip()):
                    raise RuntimeError("Payload gate requires images_path and semantic label cache path.")
                payload_dataset = _PayloadCacheMemmapDataset(
                    images_path=str(payload_images_path),
                    labels_path=str(payload_semantic_cache_path),
                    picks=np.asarray(payload_stage2_picks_np, dtype=np.int64),
                    labels_index_mode="local_dataset",
                )
                total_datasets.append(payload_dataset)

            total_rows = int(total_targets_sched_np.shape[0]) if int(getattr(total_targets_sched_np, "ndim", 0)) == 2 else 0
            if int(len(total_datasets)) <= 0 or int(total_rows) <= 0:
                berkeley_gate_loader = None
                berkeley_gate_loader_count = 0
                berkeley_gate_schedule_info = {
                    "rows": 0,
                    "selected_rows": 0,
                    "payload_val_rows": int(payload_rows_in_total),
                    "excluded_stage2_symbol_rows": int(stage2_symbol_rows_in_total),
                    "excluded_stage2_symbol_terms": int(len(stage2_symbol_term_keys)),
                    "excluded_gestation_val_rows": int(len(gestation_eval_targets)),
                    "active_terms": 0,
                    "reason": f"{str(reason)}:empty_stage2_validation_deck",
                }
                total_gate_samples_seen = 0
                total_gate_samples_required = 0
                total_gate_preview_images = []
                total_gate_preview_targets = []
                _log(
                    "[semantic-gate-loader] "
                    f"reason={str(reason)} "
                    f"stage1_gestation_train={int(gestation_stage_loader_count)}/{int(len(gestation_train_images))} "
                    f"gestation={int(gestation_gate_loader_count)}/{int(len(gestation_eval_images))} "
                    f"stage2_val=0/0 "
                    f"excluded_stage2_symbols={int(stage2_symbol_rows_in_total)} "
                    "active_terms=0 bucketed=0"
                )
                return
            total_dataset: Dataset = total_datasets[0] if int(len(total_datasets)) == 1 else torch.utils.data.ConcatDataset(total_datasets)
            total_gate_preview_images = []
            total_gate_preview_targets = []
            total_preview_cap = min(int(total_rows), 64)
            for preview_i in range(int(total_preview_cap)):
                try:
                    sample_row = total_dataset[int(preview_i)]
                    if not (isinstance(sample_row, (tuple, list)) and int(len(sample_row)) >= 2):
                        continue
                    img_raw, target_raw = sample_row[0], sample_row[1]
                    if torch.is_tensor(img_raw):
                        img_np = img_raw.detach().to(torch.float32).cpu().numpy().astype(np.float32, copy=False)
                    else:
                        img_np = np.asarray(img_raw, dtype=np.float32)
                    if torch.is_tensor(target_raw):
                        target_np = target_raw.detach().to(torch.float32).cpu().numpy().astype(np.float32, copy=False).reshape(-1)
                    else:
                        target_np = np.asarray(target_raw, dtype=np.float32).reshape(-1)
                    if int(target_np.size) != int(expected_gate_dim):
                        continue
                    total_gate_preview_images.append(
                        _image_any_to_rgb_chw01(img_np, image_size=int(shared_embed_image_size))
                    )
                    total_gate_preview_targets.append(np.asarray(target_np, dtype=np.float32).reshape(-1))
                except Exception:
                    continue
            total_order: Optional[List[int]] = None
            berkeley_gate_schedule_info = {
                "rows": int(total_rows),
                "selected_rows": int(total_rows),
                "payload_val_rows": int(payload_rows_in_total),
                "excluded_stage2_symbol_rows": int(stage2_symbol_rows_in_total),
                "excluded_stage2_symbol_terms": int(len(stage2_symbol_term_keys)),
                "excluded_gestation_val_rows": int(len(gestation_eval_targets)),
                "target_dim": int(condition_num_classes),
                "active_terms": 0,
                "reason": str(reason),
            }
            if bool(args.gate_total_token_schedule_enabled):
                core_sched_lc = {re.sub(r"\s+", " ", str(t)).strip().lower() for t in semantic_core_terms}
                skip_sched_lc = {
                    "berkeley sbd dataset",
                    "mnist dataset",
                    "emnist dataset",
                    "kmnist dataset",
                    "gan image",
                    "regurgitated content",
                }
                active_terms_stage2_sched = []
                for term in supervised_class_names:
                    tk = re.sub(r"\s+", " ", str(term)).strip().lower()
                    if not tk:
                        continue
                    active_terms_stage2_sched.append(str(term))
                for term in active_extra_terms:
                    tk = re.sub(r"\s+", " ", str(term)).strip().lower()
                    if not tk:
                        continue
                    if tk in core_sched_lc or tk in skip_sched_lc:
                        continue
                    if re.fullmatch(r"semantic slot \d+", tk):
                        continue
                    active_terms_stage2_sched.append(str(term))
                active_terms_stage2_sched = list(dict.fromkeys(active_terms_stage2_sched))
                if len(active_terms_stage2_sched) <= 0:
                    active_terms_stage2_sched = list(class_names)
                total_order, berkeley_gate_schedule_info = _schedule_semantic_gate_indices(
                    targets=total_targets_sched_np,
                    class_names=class_names,
                    active_terms=active_terms_stage2_sched,
                    seed=int(cycle_seed) + 1931,
                    max_samples=0,
                    activation_threshold=float(args.gate_total_token_schedule_threshold),
                )
                berkeley_gate_schedule_info = dict(berkeley_gate_schedule_info)
                berkeley_gate_schedule_info["reason"] = str(reason)
                try:
                    hits_map = berkeley_gate_schedule_info.get("active_term_hits", {})
                    if isinstance(hits_map, dict) and len(hits_map) > 0:
                        top_hits = sorted(
                            [(str(k), int(v)) for k, v in hits_map.items()],
                            key=lambda kv: (-int(kv[1]), str(kv[0]).lower()),
                        )[:8]
                        stage_counts = berkeley_gate_schedule_info.get("stage_counts", {})
                        stage_counts_txt = ""
                        if isinstance(stage_counts, dict):
                            stage_counts_txt = ",".join(
                                [
                                    f"{str(k)}:{int(v)}"
                                    for k, v in sorted(stage_counts.items(), key=lambda kv: str(kv[0]).lower())
                                ]
                            )
                        missing_terms = berkeley_gate_schedule_info.get("missing_active_terms", [])
                        missing_txt = ""
                        if isinstance(missing_terms, list) and len(missing_terms) > 0:
                            missing_txt = ",".join([str(x) for x in missing_terms[:8]])
                        _log(
                            "[semantic-gate-scheduler] "
                            f"mode={str(berkeley_gate_schedule_info.get('scheduler_mode', 'unknown'))} "
                            f"active_terms_churn={int(len(active_extra_terms))} "
                            f"active_terms_supervised={int(len(supervised_class_names))} "
                            f"active_terms_sched={int(len(active_terms_stage2_sched))} "
                            f"top_hits={'; '.join([f'{k}:{v}' for k, v in top_hits])} "
                            f"stages={stage_counts_txt or 'n/a'} "
                            f"missing={missing_txt or 'none'}"
                        )
                except Exception:
                    pass
            berkeley_gate_loader, berkeley_gate_loader_count = _build_gate_loader_from_dataset(
                dataset=total_dataset,
                batch_size=int(gate_loader_batch),
                num_workers=max(0, int(args.berkeley_refresh_workers)),
                device=device,
                seed=int(args.seed) + 2017,
                max_samples=0,
                ordered_indices=total_order,
                persistent_workers=bool(args.loader_persistent_workers),
                prefetch_factor=int(args.loader_prefetch_factor),
            )
            total_gate_samples_seen = 0
            total_gate_samples_required = int(berkeley_gate_loader_count)
            _log(
                "[semantic-gate-loader] "
                f"reason={str(reason)} "
                f"stage1_gestation_train={int(gestation_stage_loader_count)}/{int(len(gestation_train_images))} "
                f"gestation={int(gestation_gate_loader_count)}/{int(len(gestation_eval_images))} "
                f"stage2_val={int(berkeley_gate_loader_count)}/{int(total_rows)} "
                f"payload_val={int(payload_rows_in_total)} "
                f"excluded_stage2_symbols={int(stage2_symbol_rows_in_total)} "
                f"active_terms={int(berkeley_gate_schedule_info.get('active_terms', 0))} "
                f"bucketed={int(berkeley_gate_schedule_info.get('bucketed_rows', 0))}"
            )

        _rebuild_semantic_gate_loaders(
            reason="startup",
            cycle_seed=int(args.seed),
            include_payload_val=False,
        )

        def _semantic_vocab_runtime_snapshot() -> Dict[str, Any]:
            return {
                "supervised_class_names": list(supervised_class_names),
                "active_extra_terms": list(active_extra_terms),
                "semantic_extra_count": int(semantic_extra_count),
                "semantic_core_terms": list(semantic_core_terms),
                "semantic_core_extra_count": int(semantic_core_extra_count),
                "semantic_text_condition_temperature": float(args.semantic_vocab_text_condition_temperature),
                "semantic_text_condition_topk": int(args.semantic_vocab_text_condition_topk),
                "semantic_unknown_label_rows_per_cycle": int(args.semantic_vocab_unknown_label_rows_per_cycle),
                "semantic_bootstrap_origin_label": str(args.semantic_vocab_bootstrap_origin_label),
                "semantic_kind_vectors_present": sorted([str(k) for k in semantic_kind_target_vectors.keys()]),
                "semantic_default_core_terms": list(_default_semantic_core_terms()),
                "semantic_vocab_pool_terms": list(semantic_vocab_pool_terms),
                "semantic_vocab_churn_cursor": int(semantic_vocab_churn_cursor),
                "semantic_vocab_churn_sweep_cycles": int(args.semantic_vocab_churn_sweep_cycles),
                "semantic_churn_history": list(semantic_churn_history),
                "regurgitated_churn_last_info": dict(regurgitated_churn_last_info),
                "regurgitated_churn_pool_items": [
                    {
                        "path": str(item.get("path", "")),
                        "terms": list(item.get("terms", [])) if isinstance(item.get("terms", []), list) else [],
                        "lifetime": int(item.get("lifetime", 0)),
                        "uses": int(item.get("uses", 0)),
                        "seq": int(item.get("seq", 0)),
                        "source": str(item.get("source", "")),
                    }
                    for item in regurgitated_churn_pool
                ],
                "gan_live_churn_last_info": dict(gan_live_churn_last_info),
                "gan_live_churn_pool_items": [
                    {
                        "lifetime": int(item.get("lifetime", 0)),
                        "uses": int(item.get("uses", 0)),
                        "seq": int(item.get("seq", 0)),
                        "source": str(item.get("source", "")),
                    }
                    for item in gan_live_churn_pool
                ],
                "symbol_pool_info": dict(symbol_pool_info),
                "symbol_pool_origin_terms_by_term": {
                    str(k): list(v)
                    for k, v in symbol_pool_origin_terms_by_term.items()
                },
                "payload_conditioning_info": dict(payload_conditioning_info),
                "payload_symbol_aug_info": dict(payload_symbol_aug_info),
                "payload_flashcard_info": dict(payload_flashcard_info),
                "payload_images_base_count": int(len(payload_images_base)),
                "payload_conditions_base_count": int(len(payload_conditions_supervised_base)),
                "payload_condition_terms_base_count": int(len(payload_condition_terms_supervised_base)),
                "payload_gate_validation_rows": int(len(payload_gate_val_terms_base)),
                "payload_gate_validation_info": dict(payload_gate_val_info),
                "payload_images_active_count": int(len(payload_images)),
                "payload_conditions_active_count": int(len(payload_conditions)),
                "gestation_stage_loader_info": dict(gestation_stage_info),
                "gestation_gate_loader_info": dict(gestation_gate_info),
                "total_gate_loader_info": dict(berkeley_gate_schedule_info),
                "gd_vocab_library": {
                    "enabled": bool(args.gd_vocab_library_enabled),
                    "autoload": bool(args.gd_vocab_library_autoload),
                    "dir": str(gd_vocab_library_dir),
                    "active_hash": str(active_gd_vocab_hash),
                    "active_profile": dict(active_gd_vocab_profile),
                },
            }

        def _decorate_checkpoint_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
            out = dict(payload) if isinstance(payload, dict) else {}
            out["semantic_vocab_runtime"] = _semantic_vocab_runtime_snapshot()
            out["active_gd_vocab_hash"] = str(active_gd_vocab_hash)
            out["active_gd_vocab_profile"] = dict(active_gd_vocab_profile)
            return out

        def _offload_inactive_stage_models(reason: str):
            nonlocal generator, discriminator, wave_classifier
            if (not bool(stage_module_offload_runtime)) or (device.type != "cuda"):
                return
            moved: List[str] = []
            if generator is not None and _module_device_type(generator) == "cuda":
                generator = generator.to(torch.device("cpu"))
                moved.append("generator")
            if discriminator is not None and _module_device_type(discriminator) == "cuda":
                discriminator = discriminator.to(torch.device("cpu"))
                moved.append("discriminator")
            if wave_classifier is not None and _module_device_type(wave_classifier) == "cuda":
                wave_classifier = wave_classifier.to(torch.device("cpu"))
                moved.append("wave_classifier")
            if len(moved) <= 0:
                return
            if bool(args.stage_module_offload_empty_cache):
                gc.collect()
                torch.cuda.empty_cache()
            _log(f"[vram-offload] moved {', '.join(moved)} to cpu ({reason})")

        def _cleanup_cuda_allocator():
            if device.type != "cuda":
                return
            if not bool(args.stage_module_offload_empty_cache):
                return
            gc.collect()
            torch.cuda.empty_cache()

        def _is_cuda_oom(exc: BaseException) -> bool:
            txt = str(exc).lower()
            if "out of memory" not in txt:
                return False
            return ("cuda" in txt) or ("cudnn" in txt) or ("cublas" in txt)

        def _resolve_transformer_stage_train_batch(stage_tag: str) -> int:
            stage_key = str(stage_tag).strip().upper()
            base_batch = max(1, int(args.transformer_batch_size))
            round_batch_cli = int(args.transformer_round_batch_size)
            round_batch = max(1, int(round_batch_cli)) if int(round_batch_cli) > 0 else int(base_batch)
            joint_batch_cli = int(args.joint_transformer_batch_size)
            if stage_key == "J":
                if int(joint_batch_cli) > 0:
                    return max(1, int(joint_batch_cli))
                # Joint updates run after generator/discriminator activity; default to a smaller micro-batch.
                auto_joint = max(1, int(round_batch) // 2)
                if (not bool(stage_module_offload_runtime)) and int(auto_joint) > 1:
                    auto_joint = max(1, int(auto_joint) // 2)
                return int(auto_joint)
            if stage_key == "R":
                return int(round_batch)
            return int(base_batch)

        def _resolve_transformer_stage_eval_batch(train_batch: int) -> int:
            eval_batch_cli = int(args.transformer_eval_batch_size)
            if int(eval_batch_cli) > 0:
                return max(1, int(eval_batch_cli))
            return max(1, int(train_batch))

        def _train_transformer_stage_with_retry(
            stage_tag: str,
            epochs: int,
            steps_per_epoch: int,
            score_target_weight: float,
            score_target_margin: float,
            seed: int,
            step_preview_callback: Optional[Callable[[Dict[str, Any]], None]],
        ) -> Tuple[nn.Module, List[Dict[str, float]], int]:
            nonlocal transformer
            stage_key = str(stage_tag).strip().upper()
            train_batch = max(1, int(_resolve_transformer_stage_train_batch(stage_key)))
            vec_window = max(1, int(args.transformer_degrade_vectorized_window))
            cache_eval_batches = bool(args.transformer_cache_eval_batches)
            attempt = 0
            while True:
                attempt += 1
                if attempt == 1:
                    _log(
                        f"[transformer-memory] stage={stage_key} "
                        f"train_batch={int(train_batch)} vec_window={int(vec_window)} "
                        f"cache_eval={1 if bool(cache_eval_batches) else 0}"
                    )
                else:
                    _log(
                        f"[transformer-memory] stage={stage_key} retry#{int(attempt)} "
                        f"train_batch={int(train_batch)} vec_window={int(vec_window)} "
                        f"cache_eval={1 if bool(cache_eval_batches) else 0}"
                    )
                try:
                    train_target_labels_stage = train_stream_target_labels
                    if (
                        train_stream_target_labels is not None
                        and float(args.target_label_knockout_prob) > 0.0
                    ):
                        train_target_labels_stage, train_knockout_info = _label_knockout_optional_rows_np(
                            rows=train_stream_target_labels,
                            prob=float(args.target_label_knockout_prob),
                            seed=int(seed) + (int(attempt) * 101) + (ord(stage_key[0]) if stage_key else 0),
                            min_keep=int(args.target_label_knockout_min_keep),
                            max_drop_frac=float(args.target_label_knockout_max_drop_frac),
                            threshold=0.5,
                        )
                        _log(
                            "[target-knockout] "
                            "scope=transformer_train "
                            f"stage={stage_key} "
                            f"rows={int(train_knockout_info.get('rows_total', 0))} "
                            f"with_targets={int(train_knockout_info.get('rows_with_targets', 0))} "
                            f"applied={int(train_knockout_info.get('rows_applied', 0))} "
                            f"dropped={int(train_knockout_info.get('labels_dropped', 0))} "
                            f"prob={float(train_knockout_info.get('prob', 0.0)):.3f}"
                        )
                    transformer, stage_hist = train_transformer_feature_metric(
                        transformer=transformer,
                        classifier=classifier,
                        train_streams=train_streams,
                        train_target_labels=train_target_labels_stage,
                        val_streams=val_streams,
                        cfg=best_cfg,
                        sample_bits=sample_bits,
                        image_hw=image_hw,
                        device=device,
                        epochs=max(1, int(epochs)),
                        steps_per_epoch=max(1, int(steps_per_epoch)),
                        batch_size=int(train_batch),
                        chunk_samples=chunk_samples,
                        lr=args.transformer_lr,
                        lr_sine_cycles=args.lr_sine_cycles,
                        lr_sine_frequency=args.lr_sine_frequency,
                        lr_sine_tail_fraction=args.lr_sine_tail_fraction,
                        lr_sine_min_scale=args.lr_sine_min_scale,
                        score_topk=args.score_topk,
                        score_threshold=args.score_threshold,
                        score_w_topk=args.score_w_topk,
                        score_w_cov=args.score_w_cov,
                        score_w_mean=args.score_w_mean,
                        degrade_inputs=bool(args.transformer_degrade_inputs),
                        degrade_min_strength=float(args.transformer_degrade_min_strength),
                        degrade_max_strength=float(args.transformer_degrade_max_strength),
                        degrade_noise_std_min=float(args.transformer_degrade_noise_std_min),
                        degrade_noise_std_max=float(args.transformer_degrade_noise_std_max),
                        degrade_dropout_max=float(args.transformer_degrade_dropout_max),
                        degrade_quant_bits_min=int(args.transformer_degrade_quant_bits_min),
                        degrade_quant_bits_max=int(args.transformer_degrade_quant_bits_max),
                        degrade_vectorized_window=max(1, int(vec_window)),
                        degrade_mode=str(args.transformer_degrade_mode),
                        degrade_domain=str(args.transformer_degrade_domain),
                        degrade_stride_skew_max=float(args.transformer_degrade_stride_skew_max),
                        deskew_prefilter_weight=float(args.transformer_deskew_prefilter_weight),
                        deskew_residual_weight=float(args.transformer_deskew_residual_weight),
                        deskew_pre_token_mix=float(args.transformer_deskew_pre_token_mix),
                        deskew_post_token_mix=float(args.transformer_deskew_post_token_mix),
                        deskew_token_residual_weight=float(args.transformer_deskew_token_residual_weight),
                        entropy_penalty_weight=float(args.transformer_loss_entropy_weight),
                        high_bit_penalty_weight=float(args.transformer_loss_high_bit_weight),
                        low_bit_penalty_weight=float(args.transformer_loss_low_bit_weight),
                        wave_l1_weight=float(args.transformer_loss_wave_l1_weight),
                        score_target_weight=float(score_target_weight),
                        score_target_margin=float(score_target_margin),
                        score_rank_weight=float(args.transformer_loss_score_rank_weight),
                        score_rank_margin=float(args.transformer_loss_score_rank_margin),
                        score_spurious_weight=float(args.transformer_loss_score_spurious_weight),
                        score_spurious_threshold=float(args.transformer_loss_score_spurious_threshold),
                        score_spurious_temp=float(args.transformer_loss_score_spurious_temp),
                        score_spurious_margin=float(args.transformer_loss_score_spurious_margin),
                        amp=amp_enabled,
                        amp_dtype=args.amp_dtype,
                        channels_last=bool(args.channels_last),
                        grad_accum_steps=grad_accum_steps,
                        pin_memory=bool(args.pin_memory_wave_batches),
                        stream_cache_on_device=bool(args.cache_wave_streams_on_device),
                        log_every_steps=max(0, int(args.transformer_log_every)),
                        step_preview_callback=step_preview_callback,
                        stop_requested=_poll_gui_stop,
                        cache_eval_batches=bool(cache_eval_batches),
                        visualize_status=bool(args.transformer_visualize_status),
                        visualize_scale=max(1, int(args.transformer_viz_scale)),
                        seed=int(seed),
                    )
                    return transformer, stage_hist, int(train_batch)
                except RuntimeError as e:
                    if device.type != "cuda" or (not _is_cuda_oom(e)):
                        raise
                    prev_batch = int(train_batch)
                    prev_vec = int(vec_window)
                    prev_cache = bool(cache_eval_batches)
                    if bool(cache_eval_batches):
                        cache_eval_batches = False
                    elif int(vec_window) > 1:
                        vec_window = max(1, int(vec_window) // 2)
                    elif int(train_batch) > 1:
                        train_batch = max(1, int(train_batch) // 2)
                    else:
                        _log(
                            f"[transformer-memory] stage={stage_key} "
                            f"CUDA OOM at minimal settings (train_batch={int(train_batch)}); aborting."
                        )
                        raise
                    _cleanup_cuda_allocator()
                    _log(
                        f"[transformer-memory] stage={stage_key} CUDA OOM "
                        f"(train_batch={int(prev_batch)} vec_window={int(prev_vec)} cache_eval={1 if bool(prev_cache) else 0}); "
                        f"retrying with train_batch={int(train_batch)} vec_window={int(vec_window)} "
                        f"cache_eval={1 if bool(cache_eval_batches) else 0}"
                    )

        def _evaluate_transformer_stage_with_retry(
            stage_tag: str,
            batch_size_hint: int,
            max_batches: int,
        ) -> Tuple[Dict[str, float], int]:
            stage_key = str(stage_tag).strip().upper()
            eval_batch = max(1, int(_resolve_transformer_stage_eval_batch(batch_size_hint)))
            while True:
                try:
                    eval_stats = evaluate_feature_score_before_after(
                        transformer=transformer,
                        classifier=classifier,
                        streams=val_streams,
                        cfg=best_cfg,
                        sample_bits=sample_bits,
                        image_hw=image_hw,
                        chunk_samples=chunk_samples,
                        device=device,
                        max_batches=max(1, int(max_batches)),
                        batch_size=int(eval_batch),
                        score_topk=args.score_topk,
                        score_threshold=args.score_threshold,
                        score_w_topk=args.score_w_topk,
                        score_w_cov=args.score_w_cov,
                        score_w_mean=args.score_w_mean,
                        amp=amp_enabled,
                        amp_dtype=args.amp_dtype,
                        channels_last=bool(args.channels_last),
                        pin_memory=bool(args.pin_memory_wave_batches),
                        stream_cache_on_device=bool(args.cache_wave_streams_on_device),
                    )
                    return eval_stats, int(eval_batch)
                except RuntimeError as e:
                    if device.type != "cuda" or (not _is_cuda_oom(e)) or int(eval_batch) <= 1:
                        raise
                    prev_eval_batch = int(eval_batch)
                    eval_batch = max(1, int(eval_batch) // 2)
                    _cleanup_cuda_allocator()
                    _log(
                        f"[transformer-memory] stage={stage_key} eval CUDA OOM "
                        f"(batch={int(prev_eval_batch)}); retrying with batch={int(eval_batch)}"
                    )

        if resume_enabled:
            if isinstance(resume_summary, dict):
                if isinstance(resume_summary.get("transformer_history"), list):
                    transformer_hist.extend(list(resume_summary.get("transformer_history", [])))
                if isinstance(resume_summary.get("wave_classifier_history"), list):
                    wave_classifier_hist.extend(list(resume_summary.get("wave_classifier_history", [])))
                elif isinstance(resume_summary.get("classifier_history"), list):
                    wave_classifier_hist.extend(list(resume_summary.get("classifier_history", [])))
                if isinstance(resume_summary.get("generator_history"), list):
                    generator_hist.extend(list(resume_summary.get("generator_history", [])))
                if isinstance(resume_summary.get("orchestration_history"), list):
                    orchestration_rows.extend(list(resume_summary.get("orchestration_history", [])))
                if resume_summary.get("wave_classifier_last_eval") is not None:
                    wave_classifier_last_eval = resume_summary.get("wave_classifier_last_eval")
                if resume_summary.get("wave_zero_shot_last_eval") is not None:
                    wave_zero_shot_last_eval = resume_summary.get("wave_zero_shot_last_eval")
                if isinstance(resume_summary.get("wave_zero_shot_history"), list):
                    wave_zero_shot_history.extend(list(resume_summary.get("wave_zero_shot_history", [])))
                if isinstance(resume_summary.get("berkeley_refresh_history"), list) and len(refresh_rows) == 0:
                    refresh_rows.extend(list(resume_summary.get("berkeley_refresh_history", [])))
                if isinstance(resume_summary.get("gate_history"), list):
                    gate_history.extend(list(resume_summary.get("gate_history", [])))
            if isinstance(resume_pipeline_ckpt, dict):
                if isinstance(resume_pipeline_ckpt.get("transformer_history"), list):
                    transformer_hist = list(resume_pipeline_ckpt.get("transformer_history", transformer_hist))
                if isinstance(resume_pipeline_ckpt.get("wave_classifier_history"), list):
                    wave_classifier_hist = list(resume_pipeline_ckpt.get("wave_classifier_history", wave_classifier_hist))
                if isinstance(resume_pipeline_ckpt.get("generator_history"), list):
                    generator_hist = list(resume_pipeline_ckpt.get("generator_history", generator_hist))
                if isinstance(resume_pipeline_ckpt.get("orchestration_history"), list):
                    orchestration_rows = list(resume_pipeline_ckpt.get("orchestration_history", orchestration_rows))
                if isinstance(resume_pipeline_ckpt.get("refresh_history"), list):
                    refresh_rows = list(resume_pipeline_ckpt.get("refresh_history", refresh_rows))
                if isinstance(resume_pipeline_ckpt.get("gate_history"), list):
                    gate_history = list(resume_pipeline_ckpt.get("gate_history", gate_history))
                if resume_pipeline_ckpt.get("wave_classifier_last_eval") is not None:
                    wave_classifier_last_eval = resume_pipeline_ckpt.get("wave_classifier_last_eval")
                if resume_pipeline_ckpt.get("wave_zero_shot_last_eval") is not None:
                    wave_zero_shot_last_eval = resume_pipeline_ckpt.get("wave_zero_shot_last_eval")
                if isinstance(resume_pipeline_ckpt.get("wave_zero_shot_history"), list):
                    wave_zero_shot_history = list(resume_pipeline_ckpt.get("wave_zero_shot_history", wave_zero_shot_history))
            resume_semantic_runtime = None
            if isinstance(resume_summary, dict) and isinstance(resume_summary.get("semantic_vocab_runtime"), dict):
                resume_semantic_runtime = dict(resume_summary.get("semantic_vocab_runtime"))
            if isinstance(resume_pipeline_ckpt, dict) and isinstance(resume_pipeline_ckpt.get("semantic_vocab_runtime"), dict):
                resume_semantic_runtime = dict(resume_pipeline_ckpt.get("semantic_vocab_runtime"))
            if isinstance(resume_semantic_runtime, dict):
                prior_hist = resume_semantic_runtime.get("semantic_churn_history", [])
                if isinstance(prior_hist, list) and len(prior_hist) > 0:
                    semantic_churn_history = list(prior_hist) + list(semantic_churn_history)
                prior_terms = _normalize_vocab_terms(resume_semantic_runtime.get("active_extra_terms", []))
                if int(semantic_extra_count) > 0 and len(prior_terms) == int(semantic_extra_count):
                    active_extra_terms = _normalize_active_extra_terms_with_core(
                        active_terms=prior_terms,
                        core_terms=semantic_core_terms,
                        total_slots=int(semantic_extra_count),
                    )
                prior_pool = _normalize_vocab_terms(resume_semantic_runtime.get("semantic_vocab_pool_terms", []))
                if len(prior_pool) > 0:
                    semantic_vocab_pool_terms = _normalize_vocab_terms(
                        list(semantic_vocab_pool_terms) + list(prior_pool)
                    )
                try:
                    semantic_vocab_churn_cursor = max(0, int(resume_semantic_runtime.get("semantic_vocab_churn_cursor", 0)))
                except Exception:
                    semantic_vocab_churn_cursor = 0
                prior_regurg_pool = resume_semantic_runtime.get("regurgitated_churn_pool_items", [])
                if isinstance(prior_regurg_pool, list) and len(prior_regurg_pool) > 0:
                    regurgitated_churn_pool = []
                    regurgitated_churn_seen_paths.clear()
                    regurgitated_churn_seq = 0
                    for item in prior_regurg_pool:
                        if not isinstance(item, dict):
                            continue
                        path_txt = str(item.get("path", "")).strip()
                        terms = _sanitize_regurgitated_churn_terms(item.get("terms", []))
                        life = max(0, int(item.get("lifetime", 0)))
                        uses = max(0, int(item.get("uses", 0)))
                        seq = max(0, int(item.get("seq", 0)))
                        if not path_txt or life <= 0 or len(terms) <= 0:
                            continue
                        regurgitated_churn_pool.append(
                            {
                                "path": str(path_txt),
                                "terms": list(terms),
                                "lifetime": int(life),
                                "uses": int(uses),
                                "seq": int(seq),
                                "source": str(item.get("source", "resume")),
                            }
                        )
                        regurgitated_churn_seen_paths.add(str(path_txt))
                        regurgitated_churn_seq = max(int(regurgitated_churn_seq), int(seq))
                prior_regurg_info = resume_semantic_runtime.get("regurgitated_churn_last_info", None)
                if isinstance(prior_regurg_info, dict):
                    regurgitated_churn_last_info = dict(prior_regurg_info)
                _activate_semantic_vocab(
                    cycle_id=0,
                    cycle_local=0,
                    global_round=int(len(orchestration_rows)),
                    reason="resume_restore",
                    allow_churn=False,
                    autoload_gd=False,
                )
                _rebuild_semantic_gate_loaders(
                    reason="resume_restore",
                    cycle_seed=int(args.seed) + (int(len(orchestration_rows)) * 13),
                    include_payload_val=False,
                )

        if resume_enabled and resume_transformer_path.exists():
            transformer_resume_info = _apply_model_init(transformer, str(resume_transformer_path))
            _log(
                f"Resumed transformer from {resume_transformer_path} "
                f"(loaded={transformer_resume_info.get('loaded_keys', 0)}, skipped={transformer_resume_info.get('skipped_keys', 0)})"
            )
        elif isinstance(resume_pipeline_ckpt, dict) and isinstance(resume_pipeline_ckpt.get("transformer_state"), dict):
            transformer_resume_info = _apply_state_dict(
                transformer,
                resume_pipeline_ckpt.get("transformer_state"),
                source_name="pipeline_checkpoint",
            )
            _log(
                "Resumed transformer from pipeline checkpoint "
                f"(loaded={transformer_resume_info.get('loaded_keys', 0)}, skipped={transformer_resume_info.get('skipped_keys', 0)})"
            )
        if resume_enabled and resume_deskew_prefilter_path.exists():
            deskew_prefilter_resume_info = _apply_model_init(transformer, str(resume_deskew_prefilter_path))
            _log(
                f"Resumed deskew prefilter from {resume_deskew_prefilter_path} "
                f"(loaded={deskew_prefilter_resume_info.get('loaded_keys', 0)}, "
                f"skipped={deskew_prefilter_resume_info.get('skipped_keys', 0)})"
            )
        elif isinstance(resume_pipeline_ckpt, dict) and isinstance(resume_pipeline_ckpt.get("deskew_prefilter_state"), dict):
            deskew_prefilter_resume_info = _apply_state_dict(
                transformer,
                resume_pipeline_ckpt.get("deskew_prefilter_state"),
                source_name="pipeline_checkpoint_deskew_prefilter",
            )
            _log(
                "Resumed deskew prefilter from pipeline checkpoint "
                f"(loaded={deskew_prefilter_resume_info.get('loaded_keys', 0)}, "
                f"skipped={deskew_prefilter_resume_info.get('skipped_keys', 0)})"
            )
        transformer = maybe_compile_module(
            transformer,
            enabled=bool(args.compile_models),
            mode=str(args.compile_mode),
        )
        if mode in ("staged_cgr", "staged_cgrw"):
            generator = ConditionalBitPlaneGenerator(
                num_classes=int(condition_num_classes),
                image_hw=image_hw,
                z_dim=max(8, int(args.generator_z_dim)),
                base_ch=max(32, int(args.generator_base_ch)),
                depth=max(2, int(args.generator_depth)),
                min_ch=max(8, int(args.generator_min_ch)),
            ).to(device)
            discriminator = ConditionalBitPlaneDiscriminator(
                num_classes=int(condition_num_classes),
                base_ch=max(16, int(args.discriminator_base_ch)),
                depth=max(2, int(args.discriminator_depth)),
                max_ch=max(16, int(args.discriminator_max_ch)),
            ).to(device)
            if resume_enabled and resume_generator_path.exists():
                generator_resume_info = _apply_model_init(generator, str(resume_generator_path))
                _log(
                    f"Resumed generator from {resume_generator_path} "
                    f"(loaded={generator_resume_info.get('loaded_keys', 0)}, skipped={generator_resume_info.get('skipped_keys', 0)})"
                )
            elif isinstance(resume_pipeline_ckpt, dict) and isinstance(resume_pipeline_ckpt.get("generator_state"), dict):
                generator_resume_info = _apply_state_dict(
                    generator,
                    resume_pipeline_ckpt.get("generator_state"),
                    source_name="pipeline_checkpoint",
                )
                _log(
                    "Resumed generator from pipeline checkpoint "
                    f"(loaded={generator_resume_info.get('loaded_keys', 0)}, skipped={generator_resume_info.get('skipped_keys', 0)})"
                )
            if resume_enabled and resume_discriminator_path.exists():
                discriminator_resume_info = _apply_model_init(discriminator, str(resume_discriminator_path))
                _log(
                    f"Resumed discriminator from {resume_discriminator_path} "
                    f"(loaded={discriminator_resume_info.get('loaded_keys', 0)}, skipped={discriminator_resume_info.get('skipped_keys', 0)})"
                )
            elif isinstance(resume_pipeline_ckpt, dict) and isinstance(resume_pipeline_ckpt.get("discriminator_state"), dict):
                discriminator_resume_info = _apply_state_dict(
                    discriminator,
                    resume_pipeline_ckpt.get("discriminator_state"),
                    source_name="pipeline_checkpoint",
                )
                _log(
                    "Resumed discriminator from pipeline checkpoint "
                    f"(loaded={discriminator_resume_info.get('loaded_keys', 0)}, skipped={discriminator_resume_info.get('skipped_keys', 0)})"
                )
            generator = maybe_compile_module(
                generator,
                enabled=bool(args.compile_models),
                mode=str(args.compile_mode),
            )
            discriminator = maybe_compile_module(
                discriminator,
                enabled=bool(args.compile_models),
                mode=str(args.compile_mode),
            )
            _maybe_restore_gd_vocab_snapshot(reason="post_init")

        cycle_offset = 0
        for row in orchestration_rows:
            if isinstance(row, dict):
                try:
                    cycle_offset = max(cycle_offset, int(float(row.get("cycle", 0))))
                except Exception:
                    pass

        library_serial = 0
        if library_index_path.exists():
            try:
                with library_index_path.open("r", encoding="utf-8") as f:
                    library_serial = sum(1 for line in f if line.strip())
            except Exception:
                library_serial = 0

        berkeley_refresh_samples_seen = _sum_berkeley_refresh_samples(refresh_rows)

        gestation_gate_streak = 0
        berkeley_gate_streak = 0
        wave_gate_streak = 0
        generator_gate_streak = 0
        transformer_gate_streak = 0
        gestation_gate_loss_history: List[float] = []
        berkeley_gate_loss_history: List[float] = []
        berkeley_loss_target_value = max(0.0, float(gate_config.get("berkeley_loss_target", 0.0)))
        berkeley_loss_target_unmet = bool(berkeley_loss_target_value > 0.0)
        if isinstance(resume_summary, dict):
            gs = resume_summary.get("gate_status", {})
            if isinstance(gs, dict):
                gestation_gate_streak = int(gs.get("gestation_streak", gestation_gate_streak))
                berkeley_gate_streak = int(gs.get("berkeley_streak", berkeley_gate_streak))
                wave_gate_streak = int(gs.get("wave_streak", wave_gate_streak))
                generator_gate_streak = int(gs.get("generator_streak", generator_gate_streak))
                transformer_gate_streak = int(gs.get("transformer_streak", transformer_gate_streak))
        if isinstance(resume_pipeline_ckpt, dict):
            gs = resume_pipeline_ckpt.get("gate_status", {})
            if isinstance(gs, dict):
                gestation_gate_streak = int(gs.get("gestation_streak", gestation_gate_streak))
                berkeley_gate_streak = int(gs.get("berkeley_streak", berkeley_gate_streak))
                wave_gate_streak = int(gs.get("wave_streak", wave_gate_streak))
                generator_gate_streak = int(gs.get("generator_streak", generator_gate_streak))
                transformer_gate_streak = int(gs.get("transformer_streak", transformer_gate_streak))
        for row in orchestration_rows:
            if not isinstance(row, dict):
                continue
            ev = row.get("berkeley_classifier_eval", None)
            if not isinstance(ev, dict):
                continue
            try:
                lv = float(ev.get("loss", float("nan")))
            except Exception:
                continue
            if math.isfinite(lv):
                berkeley_gate_loss_history.append(float(max(0.0, lv)))
            gev = row.get("gestation_classifier_eval", None)
            if isinstance(gev, dict):
                try:
                    glv = float(gev.get("loss", float("nan")))
                except Exception:
                    glv = float("nan")
                if math.isfinite(glv):
                    gestation_gate_loss_history.append(float(max(0.0, glv)))

        resume_loss_target_prev = None
        if isinstance(gate_history, list):
            for row in reversed(gate_history):
                if not isinstance(row, dict):
                    continue
                raw_prev = row.get("berkeley_loss_target", None)
                if raw_prev is None:
                    continue
                try:
                    prev_v = float(raw_prev)
                except Exception:
                    continue
                if math.isfinite(prev_v):
                    resume_loss_target_prev = float(prev_v)
                    break
        if (
            resume_loss_target_prev is not None
            and math.isfinite(float(resume_loss_target_prev))
            and abs(float(resume_loss_target_prev) - float(berkeley_loss_target_value)) > 1e-9
        ):
            _log(
                "Berkeley gate loss target changed on resume: "
                f"previous={float(resume_loss_target_prev):.4f} current={float(berkeley_loss_target_value):.4f}. "
                "Using current CLI value and resetting Berkeley gate streak/history."
            )
            berkeley_gate_streak = 0
            berkeley_gate_loss_history.clear()

        def _evaluate_two_stage_semantic_gates(round_index: int) -> Dict[str, Any]:
            nonlocal gestation_gate_streak, berkeley_gate_streak
            nonlocal gestation_gate_loss_history, berkeley_gate_loss_history
            nonlocal berkeley_loss_target_unmet
            nonlocal gestation_gate_samples_seen, gestation_gate_samples_required
            nonlocal total_gate_samples_seen, total_gate_samples_required
            nonlocal refresh_loader

            gate_active_classes = int(condition_num_classes) if int(condition_num_classes) > 0 else int(score_active_classes)

            gestation_enabled = bool(float(gate_config.get("gestation_loss_target", 0.0)) > 0.0)
            gestation_eval = None
            if bool(gestation_enabled) and gestation_gate_loader is not None:
                _cleanup_cuda_allocator()
                gestation_eval = _evaluate_berkeley_classifier_gate(
                    classifier=classifier,
                    loader=gestation_gate_loader,
                    device=device,
                    max_steps=int(args.gate_gestation_eval_max_steps),
                    active_classes=int(gate_active_classes),
                    amp_enabled=amp_enabled,
                    amp_dtype=args.amp_dtype,
                    channels_last=bool(args.channels_last),
                )
            if isinstance(gestation_eval, dict):
                seen_now = int(max(0, int(gestation_eval.get("num_samples", 0))))
                gestation_gate_samples_seen = int(gestation_gate_samples_seen) + int(seen_now)
                if int(gestation_gate_samples_required) > 0:
                    gestation_gate_samples_seen = min(int(gestation_gate_samples_seen), int(gestation_gate_samples_required))
                try:
                    glv = float(gestation_eval.get("loss", float("nan")))
                except Exception:
                    glv = float("nan")
                if math.isfinite(glv):
                    gestation_gate_loss_history.append(float(max(0.0, glv)))
            gestation_gate_cfg = {
                "berkeley_min_confidence": 0.0,
                "berkeley_min_macro_f1": 0.0,
                "berkeley_loss_target": float(gate_config.get("gestation_loss_target", 0.0)),
                "berkeley_loss_min_floor": 0.0,
            }
            gestation_detail = _evaluate_berkeley_confidence_loss_gate(
                eval_stats=gestation_eval,
                gate_config=gestation_gate_cfg,
                round_index=int(round_index),
                loss_history=gestation_gate_loss_history,
                refresh_samples_seen=int(gestation_gate_samples_seen),
                refresh_samples_required=int(gestation_gate_samples_required),
            )
            if not bool(gestation_enabled):
                gestation_detail = {
                    "pass": True,
                    "mode": "disabled",
                    "reason": "disabled",
                    "loss_target": None,
                    "loss": None,
                    "refresh_samples_seen": int(gestation_gate_samples_seen),
                    "refresh_samples_required": int(gestation_gate_samples_required),
                }
            gestation_pass = bool(gestation_detail.get("pass", True))
            if bool(gestation_pass):
                gestation_gate_streak += 1
            else:
                gestation_gate_streak = 0
            gestation_ready = bool(gestation_gate_streak >= int(gate_config.get("gestation_maintain_rounds", 1)))

            stage2_required = bool(refresh_init_configured)
            stage2_ready = (not bool(stage2_required)) or (refresh_loader is not None)
            if not bool(stage2_required):
                stage2_mode = "not_required"
            elif bool(stage2_ready):
                stage2_mode = "ready"
            else:
                stage2_mode = "waiting_for_gate_1"
            if bool(gestation_ready) and bool(stage2_required) and (not bool(stage2_ready)):
                _init_refresh_resources(
                    reason=f"initial_stage_2:round={int(round_index)}",
                    samples_seen_for_startup=int(berkeley_refresh_samples_seen),
                )
                if refresh_loader is not None:
                    _rebuild_semantic_gate_loaders(
                        reason=f"initial_stage_2_ready:round={int(round_index)}",
                        cycle_seed=int(args.seed) + (int(round_index) * 421),
                        include_payload_val=True,
                    )
                stage2_ready = (not bool(stage2_required)) or (refresh_loader is not None)
                if bool(stage2_ready):
                    stage2_mode = "ready"
                else:
                    stage2_mode = "refresh_loader_unavailable"

            total_enabled = bool(
                float(gate_config.get("berkeley_min_confidence", 0.0)) > 0.0
                or float(gate_config.get("berkeley_min_macro_f1", 0.0)) > 0.0
                or float(gate_config.get("berkeley_loss_target", 0.0)) > 0.0
            )
            total_ready_before_eval = bool(berkeley_gate_streak >= int(gate_config["berkeley_maintain_rounds"]))
            total_eval_max_steps = int(args.gate_berkeley_eval_max_steps)
            if (not bool(total_ready_before_eval)) and berkeley_gate_loader is not None:
                total_eval_max_steps = max(1, int(len(berkeley_gate_loader)))
            total_eval = None
            if bool(gestation_ready) and bool(stage2_ready) and bool(total_enabled) and berkeley_gate_loader is not None:
                _cleanup_cuda_allocator()
                total_eval = _evaluate_berkeley_classifier_gate(
                    classifier=classifier,
                    loader=berkeley_gate_loader,
                    device=device,
                    max_steps=int(total_eval_max_steps),
                    active_classes=int(gate_active_classes),
                    amp_enabled=amp_enabled,
                    amp_dtype=args.amp_dtype,
                    channels_last=bool(args.channels_last),
                )
            if isinstance(total_eval, dict):
                seen_now = int(max(0, int(total_eval.get("num_samples", 0))))
                total_gate_samples_seen = int(total_gate_samples_seen) + int(seen_now)
                if int(total_gate_samples_required) > 0:
                    total_gate_samples_seen = min(int(total_gate_samples_seen), int(total_gate_samples_required))
                try:
                    tlv = float(total_eval.get("loss", float("nan")))
                except Exception:
                    tlv = float("nan")
                if math.isfinite(tlv):
                    berkeley_gate_loss_history.append(float(max(0.0, tlv)))
                    if berkeley_loss_target_value > 0.0:
                        berkeley_loss_target_unmet = bool(float(tlv) > float(berkeley_loss_target_value))
            if not bool(total_enabled):
                total_detail = {
                    "pass": True,
                    "mode": "disabled",
                    "reason": "disabled",
                    "loss_target": None,
                    "loss": None,
                    "refresh_samples_seen": int(total_gate_samples_seen),
                    "refresh_samples_required": int(total_gate_samples_required),
                }
                total_pass = True
            elif not bool(gestation_ready):
                total_detail = {
                    "pass": False,
                    "mode": "blocked_by_gestation",
                    "reason": "gestation_gate",
                    "loss_target": gate_config.get("berkeley_loss_target", None),
                    "loss": None,
                    "refresh_samples_seen": int(total_gate_samples_seen),
                    "refresh_samples_required": int(total_gate_samples_required),
                }
                total_pass = False
            elif not bool(stage2_ready):
                total_detail = {
                    "pass": False,
                    "mode": "blocked_by_initial_stage_2",
                    "reason": "initial_stage_2",
                    "loss_target": gate_config.get("berkeley_loss_target", None),
                    "loss": None,
                    "refresh_samples_seen": int(total_gate_samples_seen),
                    "refresh_samples_required": int(total_gate_samples_required),
                }
                total_pass = False
            else:
                total_detail = _evaluate_berkeley_confidence_loss_gate(
                    eval_stats=total_eval,
                    gate_config=gate_config,
                    round_index=int(round_index),
                    loss_history=berkeley_gate_loss_history,
                    refresh_samples_seen=int(total_gate_samples_seen),
                    refresh_samples_required=int(total_gate_samples_required),
                )
                total_pass = bool(total_detail.get("pass", True))
            if bool(total_pass):
                berkeley_gate_streak += 1
            else:
                berkeley_gate_streak = 0
            total_ready = bool(berkeley_gate_streak >= int(gate_config["berkeley_maintain_rounds"]))
            try:
                g_loss = float(gestation_eval.get("loss", float("nan"))) if isinstance(gestation_eval, dict) else float("nan")
            except Exception:
                g_loss = float("nan")
            try:
                g_conf = (
                    float(gestation_eval.get("mean_confidence", float("nan")))
                    if isinstance(gestation_eval, dict)
                    else float("nan")
                )
            except Exception:
                g_conf = float("nan")
            try:
                g_macro_f1 = (
                    float(gestation_eval.get("macro_f1", float("nan")))
                    if isinstance(gestation_eval, dict)
                    else float("nan")
                )
            except Exception:
                g_macro_f1 = float("nan")
            try:
                g_samples = (
                    int(max(0, int(gestation_eval.get("num_samples", 0))))
                    if isinstance(gestation_eval, dict)
                    else 0
                )
            except Exception:
                g_samples = 0
            try:
                t_loss = float(total_eval.get("loss", float("nan"))) if isinstance(total_eval, dict) else float("nan")
            except Exception:
                t_loss = float("nan")
            try:
                t_conf = (
                    float(total_eval.get("mean_confidence", float("nan")))
                    if isinstance(total_eval, dict)
                    else float("nan")
                )
            except Exception:
                t_conf = float("nan")
            try:
                t_macro_f1 = (
                    float(total_eval.get("macro_f1", float("nan")))
                    if isinstance(total_eval, dict)
                    else float("nan")
                )
            except Exception:
                t_macro_f1 = float("nan")
            try:
                t_samples = (
                    int(max(0, int(total_eval.get("num_samples", 0))))
                    if isinstance(total_eval, dict)
                    else 0
                )
            except Exception:
                t_samples = 0
            if not bool(gestation_ready):
                phase_label = "s1_g1_only"
            elif not bool(stage2_ready):
                phase_label = "s2_only"
            else:
                phase_label = "s2_g2"
            _log(
                "[gates] "
                f"round={int(round_index)} "
                f"phase={str(phase_label)} "
                f"gestation_pass={1 if bool(gestation_pass) else 0} "
                f"gestation_ready={1 if bool(gestation_ready) else 0} "
                f"gestation_mode={str(gestation_detail.get('mode', ''))} "
                f"gestation_reason={str(gestation_detail.get('reason', ''))} "
                f"gestation_loss={(g_loss if math.isfinite(g_loss) else float('nan')):.4f} "
                f"gestation_conf={(g_conf if math.isfinite(g_conf) else float('nan')):.4f} "
                f"gestation_macro_f1={(g_macro_f1 if math.isfinite(g_macro_f1) else float('nan')):.4f} "
                f"gestation_target={gestation_detail.get('loss_target', None)} "
                f"gestation_samples={int(g_samples)} "
                f"gestation_cov={int(gestation_detail.get('refresh_samples_seen', 0))}/{int(gestation_detail.get('refresh_samples_required', 0))} "
                f"stage2_ready={1 if bool(stage2_ready) else 0} "
                f"stage2_mode={str(stage2_mode)} "
                f"g2_pass={1 if bool(total_pass) else 0} "
                f"g2_ready={1 if bool(total_ready) else 0} "
                f"g2_mode={str(total_detail.get('mode', ''))} "
                f"g2_reason={str(total_detail.get('reason', ''))} "
                f"g2_loss={(t_loss if math.isfinite(t_loss) else float('nan')):.4f} "
                f"g2_conf={(t_conf if math.isfinite(t_conf) else float('nan')):.4f} "
                f"g2_macro_f1={(t_macro_f1 if math.isfinite(t_macro_f1) else float('nan')):.4f} "
                f"g2_target={total_detail.get('loss_target', None)} "
                f"g2_samples={int(t_samples)} "
                f"g2_cov={int(total_detail.get('refresh_samples_seen', 0))}/{int(total_detail.get('refresh_samples_required', 0))}"
            )
            return {
                "gestation_eval": gestation_eval,
                "gestation_detail": dict(gestation_detail),
                "gestation_pass": bool(gestation_pass),
                "gestation_ready": bool(gestation_ready),
                "initial_stage_2_ready": bool(stage2_ready),
                "initial_stage_2_mode": str(stage2_mode),
                "total_eval": total_eval,
                "total_detail": dict(total_detail),
                "total_pass": bool(total_pass),
                "total_ready": bool(total_ready),
            }

        initial_gate_eval = _evaluate_two_stage_semantic_gates(round_index=max(0, int(len(orchestration_rows))))
        initial_gestation_eval = initial_gate_eval.get("gestation_eval", None)
        initial_gestation_detail = dict(initial_gate_eval.get("gestation_detail", {}))
        initial_gestation_pass = bool(initial_gate_eval.get("gestation_pass", True))
        initial_stage_2_ready = bool(initial_gate_eval.get("initial_stage_2_ready", (refresh_loader is not None)))
        initial_stage_2_mode = str(
            initial_gate_eval.get(
                "initial_stage_2_mode",
                ("ready" if bool(initial_stage_2_ready) else "waiting_for_gate_1"),
            )
        )
        initial_berkeley_eval = initial_gate_eval.get("total_eval", None)
        initial_berkeley_gate_detail = dict(initial_gate_eval.get("total_detail", {}))
        initial_berkeley_gate_pass = bool(initial_gate_eval.get("total_pass", True))
        _log(
            "Initial Stage 1 -> Gate 1: "
            f"pass={1 if initial_gestation_pass else 0} "
            f"mode={initial_gestation_detail.get('mode', 'hard_loss_full_refresh')} "
            f"reason={initial_gestation_detail.get('reason', 'pass')} "
            f"conf={(float(initial_gestation_eval.get('mean_confidence', float('nan'))) if isinstance(initial_gestation_eval, dict) else float('nan')):.4f} "
            f"macro_f1={(float(initial_gestation_eval.get('macro_f1', float('nan'))) if isinstance(initial_gestation_eval, dict) else float('nan')):.4f} "
            f"loss={(float(initial_gestation_eval.get('loss', float('nan'))) if isinstance(initial_gestation_eval, dict) else float('nan')):.4f} "
            f"target={initial_gestation_detail.get('loss_target', None)} "
            f"samples={int(initial_gestation_eval.get('num_samples', 0)) if isinstance(initial_gestation_eval, dict) else 0} "
            f"coverage={int(initial_gestation_detail.get('refresh_samples_seen', 0))}/"
            f"{int(initial_gestation_detail.get('refresh_samples_required', 0))}"
        )
        _log("Stage 1 trains on gestation-only rows; Gate 1 evaluates gestation validation rows.")
        _log(
            "Initial Stage 2: "
            f"ready={1 if bool(initial_stage_2_ready) else 0} "
            f"mode={initial_stage_2_mode}"
        )
        conf_init = None if (initial_berkeley_eval is None) else initial_berkeley_eval.get("mean_confidence", None)
        f1_init = None if (initial_berkeley_eval is None) else initial_berkeley_eval.get("macro_f1", None)
        loss_init = None if (initial_berkeley_eval is None) else initial_berkeley_eval.get("loss", None)
        _log(
            "Gate 2: "
            f"pass={1 if initial_berkeley_gate_pass else 0} "
            f"mode={initial_berkeley_gate_detail.get('mode', 'hard_loss_full_refresh')} "
            f"reason={initial_berkeley_gate_detail.get('reason', 'pass')} "
            f"conf={(float(conf_init) if conf_init is not None else float('nan')):.4f} "
            f"macro_f1={(float(f1_init) if f1_init is not None else float('nan')):.4f} "
            f"loss={(float(loss_init) if loss_init is not None else float('nan')):.4f} "
            f"loss_target={initial_berkeley_gate_detail.get('loss_target', None)} "
            f"coverage={int(initial_berkeley_gate_detail.get('refresh_samples_seen', 0))}/"
            f"{int(initial_berkeley_gate_detail.get('refresh_samples_required', 0))} "
            f"stage2_validation_rows={int(berkeley_gate_schedule_info.get('selected_rows', 0))}"
        )
        _log("Pipeline order: Stage 1 -> Gate 1 -> Stage 2 -> Gate 2")
        initial_wave_gate_pass = (
            (float(cls_val.get("mean_entropy", 0.0)) >= gate_config["wave_min_entropy"])
            and (float(cls_val["score"]) >= gate_config["berkeley_min_score"])
        )
        if initial_wave_gate_pass:
            wave_gate_streak = max(wave_gate_streak, 1)
        else:
            wave_gate_streak = 0

        stage_opengl_every = max(1, int(args.stage_opengl_preview_every_round))
        stage_opengl_step_driven = bool(args.stage_opengl_preview_enabled)
        stage_opengl_viewer = _TransformerStatusOpenGLViewer(
            enabled=bool(args.stage_opengl_preview_enabled),
            image_hw=image_hw,
            scale=max(1, int(args.stage_opengl_preview_scale)),
            cycle_slots=max(0, int(args.orchestration_cycles)),
        )
        stage_preview_cursor: Dict[str, int] = {}
        gui_stop_requested = False
        gui_stop_logged = False

        def _poll_gui_stop(note: str = "") -> bool:
            nonlocal gui_stop_requested, gui_stop_logged
            if gui_stop_requested:
                return True
            if not bool(args.stage_opengl_preview_enabled):
                return False
            try:
                stage_opengl_viewer.pump()
            except Exception as e:
                _log(f"[stage-opengl] event pump failed: {e}")
            if stage_opengl_viewer.stop_requested():
                gui_stop_requested = True
                if not gui_stop_logged:
                    tail = f" ({note})" if str(note).strip() else ""
                    _log(f"[stage-opengl] close requested{tail}; stopping at next safe boundary.")
                    gui_stop_logged = True
            return gui_stop_requested

        def _sync_stage_cycle_roster(total_cycles: int):
            if not bool(args.stage_opengl_preview_enabled):
                return
            try:
                stage_opengl_viewer.set_cycle_roster(total_cycles=max(0, int(total_cycles)))
            except Exception as e:
                _log(f"[stage-opengl] cycle roster sync failed: {e}")

        def _stage_cycle_selected(cycle_local: int) -> bool:
            if not bool(args.stage_opengl_preview_enabled):
                return True
            try:
                return bool(stage_opengl_viewer.is_cycle_selected(int(cycle_local)))
            except Exception:
                return True

        def _stage_gate_override_enabled() -> bool:
            if not bool(args.stage_opengl_preview_enabled):
                return False
            try:
                return bool(stage_opengl_viewer.gate_override_enabled())
            except Exception:
                return False

        if bool(args.stage_opengl_preview_enabled) and bool(args.stage_opengl_preview_bootstrap):
            try:
                bootstrap_done = False
                if bool(gestation_gate_runtime_enabled) and len(gestation_gate_preview_images) > 0:
                    pick = int(max(0, int(args.seed)) % len(gestation_gate_preview_images))
                    stage_preview_cursor["GEST"] = (pick + 1) % len(gestation_gate_preview_images)
                    img_np = np.asarray(gestation_gate_preview_images[pick], dtype=np.float32)
                    x_boot = torch.from_numpy(img_np)
                    target_line = "target:none"
                    if pick < len(gestation_gate_preview_targets):
                        target_line = _format_target_line_from_condition(
                            gestation_gate_preview_targets[pick],
                            class_names=class_names,
                            max_items=0,
                            threshold=0.5,
                        )
                    stage_opengl_viewer.update(
                        clean_img=x_boot,
                        input_img=x_boot,
                        output_img=x_boot,
                        caption=f"[BOOT GEST] sample={pick}",
                        panel_titles=["BOOT GEST target", "BOOT GEST input", "BOOT GEST out"],
                        panel_rows=[[target_line], [f"sample={pick}"], [f"sample={pick}"]],
                    )
                    bootstrap_done = True
                elif (not bool(gestation_gate_runtime_enabled)) and len(payload_images) > 0:
                    pick = int(max(0, int(args.seed)) % len(payload_images))
                    stage_preview_cursor["C"] = (pick + 1) % len(payload_images)
                    img_np = np.asarray(payload_images[pick], dtype=np.float32)
                    x_boot = torch.from_numpy(img_np)
                    target_line = "target:none"
                    if pick < len(payload_conditions):
                        target_line = _format_target_line_from_condition(
                            payload_conditions[pick],
                            class_names=class_names,
                            max_items=0,
                            threshold=0.5,
                        )
                    stage_opengl_viewer.update(
                        clean_img=x_boot,
                        input_img=x_boot,
                        output_img=x_boot,
                        caption=f"[BOOT C] sample={pick}",
                        panel_titles=["BOOT C target", "BOOT C input", "BOOT C out"],
                        panel_rows=[[target_line], [f"sample={pick}"], [f"sample={pick}"]],
                    )
                    bootstrap_done = True
                elif len(val_streams) > 0:
                    s_boot = np.asarray(val_streams[0], dtype=np.float32)
                    stage_preview_cursor["R"] = 1 % max(1, len(val_streams))
                    s_boot = _fit_wave_length(
                        x=s_boot,
                        target_samples=int(chunk_samples),
                        rng=rng,
                    )
                    with torch.no_grad():
                        xb_boot = torch.from_numpy(s_boot[None, :]).to(device)
                        img_boot = render_mono_wave_to_tensor(
                            xb_boot, cfg=best_cfg, image_hw=image_hw, sample_bits=int(sample_bits)
                        )
                    stage_opengl_viewer.update(
                        clean_img=img_boot[0],
                        input_img=img_boot[0],
                        output_img=img_boot[0],
                        caption="[BOOT R] sample=0",
                        panel_titles=["BOOT R target", "BOOT R input", "BOOT R out"],
                        panel_rows=[["sample=0"], ["sample=0"], ["sample=0"]],
                    )
                    bootstrap_done = True
                _log(f"[stage-opengl] bootstrap={'ok' if bootstrap_done else 'skipped'}")
            except Exception as e:
                _log(f"[stage-opengl] bootstrap failed: {e}")
        if bool(args.stage_opengl_preview_enabled) and bool(args.stage_opengl_preview_required) and (not bool(stage_opengl_viewer.enabled)):
            raise RuntimeError("Stage OpenGL preview required but viewer failed to initialize.")
        if bool(stage_opengl_step_driven):
            _log("[stage-opengl] live mode: per-step callback previews only (round OpenGL snapshots disabled)")

        def _chunk_wave_for_preview(stream: np.ndarray) -> np.ndarray:
            s = np.asarray(stream, dtype=np.float32)
            return _fit_wave_length(
                x=s,
                target_samples=int(chunk_samples),
                rng=rng,
            ).astype(np.float32, copy=False)

        def _append_training_preview_manifest(payload: Optional[Dict]):
            if not bool(args.training_preview_enabled):
                return
            if not isinstance(payload, dict):
                return
            try:
                with training_preview_manifest_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(payload) + "\n")
            except Exception:
                pass

        def _stage_gl_emit_standard(
            stage_key: str,
            cycle_id: int,
            round_id: int,
            step_txt: str,
            clean_img: torch.Tensor,
            input_img: torch.Tensor,
            output_img: torch.Tensor,
            probs_in: np.ndarray,
            probs_out: np.ndarray,
            label_names: Sequence[str],
            target_line_override: Optional[str] = None,
            target_extra_rows: Optional[Sequence[str]] = None,
            extra_out_rows: Optional[Sequence[str]] = None,
        ):
            if _poll_gui_stop(note=f"{stage_key}-emit"):
                return
            if (not bool(args.stage_opengl_preview_enabled)) or (not bool(stage_opengl_viewer.enabled)):
                return
            p_in = np.asarray(probs_in, dtype=np.float32).reshape(-1)
            p_out = np.asarray(probs_out, dtype=np.float32).reshape(-1)
            all_topk_in = max(1, min(int(p_in.size), len(label_names)))
            all_topk_out = max(1, min(int(p_out.size), len(label_names)))
            top_in = _top_labels_from_probs(p_in, class_names=label_names, topk=int(all_topk_in))
            top_out = _top_labels_from_probs(p_out, class_names=label_names, topk=int(all_topk_out))
            if isinstance(target_line_override, str) and str(target_line_override).strip():
                target_line = str(target_line_override)
            else:
                target_line = "target:none"
            in_lines = _format_top_label_lines(top_in, max_items=0)
            out_lines = _format_top_label_lines(top_out, max_items=0)
            target_rows: List[str] = [target_line, str(step_txt)]
            if target_extra_rows is not None:
                for r in target_extra_rows:
                    txt = str(r).strip()
                    if txt:
                        target_rows.append(txt)
            extra_rows: List[str] = []
            if extra_out_rows is not None:
                for r in extra_out_rows:
                    txt = str(r).strip()
                    if txt:
                        extra_rows.append(txt)
            in_rows = in_lines if len(in_lines) > 0 else ["n/a"]
            out_rows = [target_line] + extra_rows + (out_lines if len(out_lines) > 0 else ["n/a"])
            top_txt = "n/a"
            if len(out_lines) > 0:
                top_txt = out_lines[0]
            elif len(in_lines) > 0:
                top_txt = in_lines[0]
            stage_opengl_viewer.update(
                clean_img=clean_img,
                input_img=input_img,
                output_img=output_img,
                caption=f"[{stage_key}] cycle={int(cycle_id)} round={int(round_id)} {step_txt} top={top_txt}",
                panel_titles=[f"{stage_key} target", f"{stage_key} input", f"{stage_key} out"],
                panel_rows=[target_rows, in_rows, out_rows],
            )
            _log(f"[stage-opengl] {stage_key} {step_txt} top={top_txt}")

        def _make_c_step_callback(cycle_id: int, round_id: int):
            def _cb(payload: Dict[str, Any]):
                if _poll_gui_stop(note="C-step"):
                    return
                try:
                    img = payload.get("img")
                    probs = payload.get("probs")
                    target_vec = payload.get("target_vec")
                    if img is None or probs is None or target_vec is None:
                        return
                    step_txt = f"step={int(payload.get('global_step', 0))}/{int(payload.get('total_steps', 0))}"
                    target_line = _format_target_line_from_condition(
                        target_vec,
                        class_names=class_names,
                        max_items=0,
                        threshold=0.5,
                    )
                    loss_rows: List[str] = []
                    if "loss" in payload:
                        try:
                            loss_rows.append(f"loss={float(payload.get('loss', 0.0)):.4f}")
                        except Exception:
                            pass
                    if "batch_loss" in payload:
                        try:
                            loss_rows.append(f"batch={float(payload.get('batch_loss', 0.0)):.4f}")
                        except Exception:
                            pass
                    p = probs.detach().to(torch.float32).cpu().numpy().astype(np.float32, copy=False).reshape(-1)
                    _stage_gl_emit_standard(
                        stage_key="C",
                        cycle_id=int(cycle_id),
                        round_id=int(round_id),
                        step_txt=step_txt,
                        clean_img=img,
                        input_img=img,
                        output_img=img,
                        probs_in=p,
                        probs_out=p,
                        label_names=class_names,
                        target_line_override=target_line,
                        target_extra_rows=loss_rows,
                    )
                except Exception as e:
                    _log(f"[stage-opengl] C live preview failed: {e}")
            return _cb

        def _make_g_step_callback(stage_key: str, cycle_id: int, round_id: int):
            stage_local = str(stage_key).strip().upper()
            def _cb(payload: Dict[str, Any]):
                if _poll_gui_stop(note=f"{stage_local}-step"):
                    return
                try:
                    img = payload.get("fake_img")
                    target_img = payload.get("target_img")
                    probs = payload.get("probs")
                    if img is None or probs is None:
                        return
                    if target_img is None:
                        target_img = img
                    step_txt = f"step={int(payload.get('step', 0))}/{int(payload.get('steps_per_epoch', 0))}"
                    p = probs.detach().to(torch.float32).cpu().numpy().astype(np.float32, copy=False).reshape(-1)
                    target_cond = payload.get("target_condition")
                    target_line = None
                    if target_cond is not None:
                        target_line = _format_target_line_from_condition(
                            target_cond,
                            class_names=class_names,
                            max_items=0,
                            threshold=0.5,
                        )
                    target_rows: List[str] = []
                    for k, lbl in (
                        ("g_loss", "g"),
                        ("d_loss", "d"),
                        ("adv_loss", "adv"),
                        ("wave_loss", "wave"),
                    ):
                        if k in payload:
                            try:
                                target_rows.append(f"{lbl}={float(payload.get(k, 0.0)):.4f}")
                            except Exception:
                                pass
                    deskew_rows: List[str] = []
                    deskew_preview = payload.get("deskew_preview", None)
                    if isinstance(deskew_preview, dict):
                        try:
                            tgt = float(deskew_preview.get("target_skew", 0.0))
                            pred = float(deskew_preview.get("pred_skew", 0.0))
                            app = float(deskew_preview.get("applied_skew", 0.0))
                            rem = float(deskew_preview.get("remaining_skew", 0.0))
                            post = float(deskew_preview.get("post_residual_skew", 0.0))
                            conf = float(deskew_preview.get("confidence", 0.0))
                            deskew_rows.append(
                                f"deskew sample tgt={tgt:+.4f} pred={pred:+.4f} app={app:+.4f} conf={conf:.3f}"
                            )
                            deskew_rows.append(f"deskew sample rem={rem:+.4f} post={post:+.4f}")
                        except Exception:
                            pass
                    try:
                        deskew_rows.append(
                            "deskew avg "
                            f"|tgt|={float(payload.get('deskew_target_abs', 0.0)):.4f} "
                            f"|pred|={float(payload.get('deskew_pred_abs', 0.0)):.4f} "
                            f"|app|={float(payload.get('deskew_applied_abs', 0.0)):.4f} "
                            f"|rem|={float(payload.get('deskew_remaining_abs', 0.0)):.4f} "
                            f"|post|={float(payload.get('deskew_post_abs', 0.0)):.4f} "
                            f"conf={float(payload.get('deskew_confidence_mean', 0.0)):.3f}"
                        )
                    except Exception:
                        pass
                    _stage_gl_emit_standard(
                        stage_key=stage_local,
                        cycle_id=int(cycle_id),
                        round_id=int(round_id),
                        step_txt=step_txt,
                        clean_img=target_img,
                        input_img=img,
                        output_img=img,
                        probs_in=p,
                        probs_out=p,
                        label_names=class_names,
                        target_line_override=target_line,
                        target_extra_rows=target_rows,
                        extra_out_rows=deskew_rows,
                    )
                except Exception as e:
                    _log(f"[stage-opengl] {stage_local} live preview failed: {e}")
            return _cb

        def _make_r_step_callback(stage_key: str, cycle_id: int, round_id: int):
            stage_local = str(stage_key).strip().upper()
            def _cb(payload: Dict[str, Any]):
                if _poll_gui_stop(note=f"{stage_local}-step"):
                    return
                try:
                    x_clean = payload.get("x_clean")
                    x_in = payload.get("x_in")
                    x_out = payload.get("x_out")
                    if x_clean is None or x_in is None or x_out is None:
                        return
                    step_txt = f"step={int(payload.get('step', 0))}/{int(payload.get('steps_per_epoch', 0))}"
                    with torch.no_grad():
                        xc = x_clean.to(device=device, dtype=torch.float32)
                        xi = x_in.to(device=device, dtype=torch.float32)
                        xo = x_out.to(device=device, dtype=torch.float32)
                        img_clean = render_mono_wave_to_tensor(xc, cfg=best_cfg, image_hw=image_hw, sample_bits=int(sample_bits))
                        img_in = render_mono_wave_to_tensor(xi, cfg=best_cfg, image_hw=image_hw, sample_bits=int(sample_bits))
                        img_out = render_mono_wave_to_tensor(xo, cfg=best_cfg, image_hw=image_hw, sample_bits=int(sample_bits))
                        cls_in = img_in
                        cls_out = img_out
                        if bool(args.channels_last):
                            cls_in = cls_in.contiguous(memory_format=torch.channels_last)
                            cls_out = cls_out.contiguous(memory_format=torch.channels_last)
                        p_in = torch.sigmoid(classifier(cls_in))[0].detach().to(torch.float32).cpu().numpy().astype(np.float32, copy=False)
                        p_out = torch.sigmoid(classifier(cls_out))[0].detach().to(torch.float32).cpu().numpy().astype(np.float32, copy=False)
                    target_cond = payload.get("target_condition")
                    target_line = None
                    if target_cond is not None:
                        target_line = _format_target_line_from_condition(
                            target_cond,
                            class_names=class_names,
                            max_items=0,
                            threshold=0.5,
                        )
                    loss_rows: List[str] = []
                    for k, lbl in (
                        ("loss", "loss"),
                        ("denoise_l1", "denoise"),
                        ("high_bits_l1", "hi"),
                        ("low_bits_l1", "lo"),
                        ("score_after", "after"),
                        ("score_gap", "gap"),
                    ):
                        if k in payload:
                            try:
                                loss_rows.append(f"{lbl}={float(payload.get(k, 0.0)):.4f}")
                            except Exception:
                                pass
                    deskew_rows: List[str] = []
                    deskew_preview = payload.get("deskew_preview", None)
                    if isinstance(deskew_preview, dict):
                        try:
                            tgt = float(deskew_preview.get("target_skew", 0.0))
                            pred = float(deskew_preview.get("pred_skew", 0.0))
                            app = float(deskew_preview.get("applied_skew", 0.0))
                            rem = float(deskew_preview.get("remaining_skew", 0.0))
                            post = float(deskew_preview.get("post_residual_skew", 0.0))
                            conf = float(deskew_preview.get("confidence", 0.0))
                            deskew_rows.append(
                                f"deskew sample tgt={tgt:+.4f} pred={pred:+.4f} app={app:+.4f} conf={conf:.3f}"
                            )
                            deskew_rows.append(f"deskew sample rem={rem:+.4f} post={post:+.4f}")
                        except Exception:
                            pass
                    try:
                        deskew_rows.append(
                            "deskew avg "
                            f"|tgt|={float(payload.get('deskew_target_abs', 0.0)):.4f} "
                            f"|pred|={float(payload.get('deskew_pred_abs', 0.0)):.4f} "
                            f"|app|={float(payload.get('deskew_applied_abs', 0.0)):.4f} "
                            f"|rem|={float(payload.get('deskew_remaining_abs', 0.0)):.4f} "
                            f"|post|={float(payload.get('deskew_post_abs', 0.0)):.4f} "
                            f"conf={float(payload.get('deskew_confidence_mean', 0.0)):.3f}"
                        )
                    except Exception:
                        pass
                    _stage_gl_emit_standard(
                        stage_key=stage_local,
                        cycle_id=int(cycle_id),
                        round_id=int(round_id),
                        step_txt=step_txt,
                        clean_img=img_clean[0],
                        input_img=img_in[0],
                        output_img=img_out[0],
                        probs_in=p_in,
                        probs_out=p_out,
                        label_names=class_names,
                        target_line_override=target_line,
                        target_extra_rows=loss_rows,
                        extra_out_rows=deskew_rows,
                    )
                except Exception as e:
                    _log(f"[stage-opengl] {stage_local} live preview failed: {e}")
            return _cb

        def _make_w_step_callback(cycle_id: int, round_id: int):
            def _cb(payload: Dict[str, Any]):
                if _poll_gui_stop(note="W-step"):
                    return
                try:
                    img = payload.get("img")
                    probs = payload.get("probs")
                    target_class = int(payload.get("target_class", -1))
                    if img is None or probs is None:
                        return
                    step_txt = f"step={int(payload.get('step', 0))}/{int(payload.get('steps_per_epoch', 0))}"
                    if int(target_class) >= 0 and int(target_class) < len(split_class_names):
                        target_line = f"target:{str(split_class_names[int(target_class)])}"
                    else:
                        target_line = "target:none"
                    loss_rows: List[str] = []
                    if "train_loss" in payload:
                        try:
                            loss_rows.append(f"loss={float(payload.get('train_loss', 0.0)):.4f}")
                        except Exception:
                            pass
                    if "batch_loss" in payload:
                        try:
                            loss_rows.append(f"batch={float(payload.get('batch_loss', 0.0)):.4f}")
                        except Exception:
                            pass
                    p = probs.detach().to(torch.float32).cpu().numpy().astype(np.float32, copy=False).reshape(-1)
                    _stage_gl_emit_standard(
                        stage_key="W",
                        cycle_id=int(cycle_id),
                        round_id=int(round_id),
                        step_txt=step_txt,
                        clean_img=img,
                        input_img=img,
                        output_img=img,
                        probs_in=p,
                        probs_out=p,
                        label_names=split_class_names,
                        target_line_override=target_line,
                        target_extra_rows=loss_rows,
                    )
                except Exception as e:
                    _log(f"[stage-opengl] W live preview failed: {e}")
            return _cb

        def _run_transformer_round_preview(stage: str, cycle_id: int, round_id: int, global_round: int, extra: Optional[Dict] = None):
            if _poll_gui_stop(note=f"{stage}-round-preview"):
                return
            do_file = bool(args.training_preview_enabled) and (int(global_round) % int(training_preview_every) == 0)
            do_gl = bool(args.stage_opengl_preview_enabled) and (int(global_round) % int(stage_opengl_every) == 0)
            stage_key = str(stage).strip().upper()
            # Keep semantic chain + C-stage round snapshots active in live mode.
            if bool(stage_opengl_step_driven) and str(stage_key) not in ("S1", "G1", "S2", "G2", "C"):
                do_gl = False
            if (not do_file) and (not do_gl):
                return

            def _next_stage_index(key: str, total: int, seed_term: int) -> int:
                n = int(total)
                if n <= 0:
                    return 0
                k = str(key).upper()
                if k not in stage_preview_cursor:
                    stage_preview_cursor[k] = int(seed_term) % n
                idx = int(stage_preview_cursor[k]) % n
                stage_preview_cursor[k] = (idx + 1) % n
                return idx

            def _render_semantic_standard(
                *,
                stage_emit_key: str,
                img_np: np.ndarray,
                target_vec: Optional[np.ndarray],
                step_txt: str,
                detail_rows: Sequence[str],
            ) -> None:
                x0 = torch.from_numpy(np.asarray(img_np, dtype=np.float32))
                with torch.no_grad():
                    x = torch.from_numpy(np.asarray(img_np, dtype=np.float32)[None, ...]).to(device=device, dtype=torch.float32)
                    if bool(args.channels_last):
                        x = x.contiguous(memory_format=torch.channels_last)
                    probs = torch.sigmoid(classifier(x)).to(torch.float32)
                p0 = probs[0].detach().cpu().numpy().astype(np.float32, copy=False)
                if target_vec is not None:
                    target_line = _format_target_line_from_condition(
                        target_vec,
                        class_names=class_names,
                        max_items=0,
                        threshold=0.5,
                    )
                else:
                    target_line = "target:none"
                _stage_gl_emit_standard(
                    stage_key=str(stage_emit_key),
                    cycle_id=int(cycle_id),
                    round_id=int(round_id),
                    step_txt=str(step_txt),
                    clean_img=x0,
                    input_img=x0,
                    output_img=x0,
                    probs_in=p0,
                    probs_out=p0,
                    label_names=class_names,
                    target_line_override=target_line,
                    target_extra_rows=[str(x) for x in list(detail_rows)],
                    extra_out_rows=[str(x) for x in list(detail_rows)],
                )

            if stage_key == "S1":
                if do_gl and stage_opengl_viewer.enabled:
                    if int(len(gestation_gate_preview_images)) <= 0:
                        _log("[stage-opengl] STAGE 1 preview skipped: no gestation samples.")
                    else:
                        pick = _next_stage_index(
                            "S1",
                            len(gestation_gate_preview_images),
                            seed_term=(int(args.seed) + (int(cycle_id) * 1103) + (int(round_id) * 173) + int(global_round)),
                        )
                        target_vec = None
                        if 0 <= int(pick) < int(len(gestation_gate_preview_targets)):
                            target_vec = np.asarray(gestation_gate_preview_targets[int(pick)], dtype=np.float32).reshape(-1)
                        s_rows = [
                            f"rows={int(gestation_gate_info.get('rows', len(gestation_gate_preview_images)))}",
                            f"selected={int(gestation_gate_info.get('selected_rows', len(gestation_gate_preview_images)))}",
                            "source=fixed_core_vocab",
                        ]
                        _render_semantic_standard(
                            stage_emit_key="S1",
                            img_np=np.asarray(gestation_gate_preview_images[int(pick)], dtype=np.float32),
                            target_vec=target_vec,
                            step_txt=f"sample={int(pick)}",
                            detail_rows=s_rows,
                        )
                return

            if stage_key == "G1":
                if do_gl and stage_opengl_viewer.enabled:
                    gate1_eval = extra.get("gestation_classifier_eval", None) if isinstance(extra, dict) else None
                    gate1_detail = extra.get("gestation_gate_detail", None) if isinstance(extra, dict) else None
                    gate1_pass = bool(extra.get("gestation_gate_pass", False)) if isinstance(extra, dict) else False
                    if int(len(gestation_gate_preview_images)) <= 0:
                        _log("[stage-opengl] GATE 1 preview skipped: no gestation samples.")
                    else:
                        pick = _next_stage_index(
                            "G1",
                            len(gestation_gate_preview_images),
                            seed_term=(int(args.seed) + (int(cycle_id) * 1301) + (int(round_id) * 211) + int(global_round)),
                        )
                        target_vec = None
                        if 0 <= int(pick) < int(len(gestation_gate_preview_targets)):
                            target_vec = np.asarray(gestation_gate_preview_targets[int(pick)], dtype=np.float32).reshape(-1)
                        g_rows: List[str] = [f"pass={1 if bool(gate1_pass) else 0}"]
                        if isinstance(gate1_detail, dict):
                            g_rows.append(f"reason={str(gate1_detail.get('reason', ''))}")
                            if gate1_detail.get("loss_target", None) is not None:
                                try:
                                    g_rows.append(f"target={float(gate1_detail.get('loss_target', 0.0)):.4f}")
                                except Exception:
                                    pass
                            g_rows.append(
                                f"cov={int(gate1_detail.get('refresh_samples_seen', 0))}/"
                                f"{int(gate1_detail.get('refresh_samples_required', 0))}"
                            )
                        if isinstance(gate1_eval, dict):
                            try:
                                g_rows.append(f"loss={float(gate1_eval.get('loss', float('nan'))):.4f}")
                            except Exception:
                                pass
                        g1_step_txt = f"sample={int(pick)}"
                        if isinstance(gate1_eval, dict):
                            try:
                                g1_loss = float(gate1_eval.get("loss", float("nan")))
                                if math.isfinite(g1_loss):
                                    g1_step_txt = f"{g1_step_txt} loss={g1_loss:.4f}"
                            except Exception:
                                pass
                        _render_semantic_standard(
                            stage_emit_key="G1",
                            img_np=np.asarray(gestation_gate_preview_images[int(pick)], dtype=np.float32),
                            target_vec=target_vec,
                            step_txt=g1_step_txt,
                            detail_rows=g_rows,
                        )
                return

            if stage_key == "S2":
                if do_gl and stage_opengl_viewer.enabled:
                    if int(len(payload_images)) <= 0:
                        _log("[stage-opengl] STAGE 2 preview skipped: no non-validation samples.")
                    else:
                        pick = _next_stage_index(
                            "S2",
                            len(payload_images),
                            seed_term=(int(args.seed) + (int(cycle_id) * 1601) + (int(round_id) * 239) + int(global_round)),
                        )
                        target_vec = None
                        if 0 <= int(pick) < int(len(payload_conditions)):
                            target_vec = np.asarray(payload_conditions[int(pick)], dtype=np.float32).reshape(-1)
                        s2_ready = bool(extra.get("initial_stage_2_ready", False)) if isinstance(extra, dict) else False
                        s2_mode = str(extra.get("initial_stage_2_mode", "")) if isinstance(extra, dict) else ""
                        s_rows = [
                            f"ready={1 if bool(s2_ready) else 0}",
                            f"mode={s2_mode}",
                            f"rows={int(len(payload_images))}",
                            "source=non_validation_total_set",
                        ]
                        _render_semantic_standard(
                            stage_emit_key="S2",
                            img_np=np.asarray(payload_images[int(pick)], dtype=np.float32),
                            target_vec=target_vec,
                            step_txt=f"sample={int(pick)}",
                            detail_rows=s_rows,
                        )
                return

            if stage_key == "G2":
                if do_gl and stage_opengl_viewer.enabled:
                    gate2_eval = extra.get("berkeley_classifier_eval", None) if isinstance(extra, dict) else None
                    gate2_detail = extra.get("berkeley_gate_detail", None) if isinstance(extra, dict) else None
                    gate2_pass = bool(extra.get("berkeley_gate_pass", False)) if isinstance(extra, dict) else False
                    if int(len(total_gate_preview_images)) <= 0:
                        _log("[stage-opengl] GATE 2 preview skipped: no total-gate samples.")
                    else:
                        pick = _next_stage_index(
                            "G2",
                            len(total_gate_preview_images),
                            seed_term=(int(args.seed) + (int(cycle_id) * 1801) + (int(round_id) * 271) + int(global_round)),
                        )
                        target_vec = None
                        if 0 <= int(pick) < int(len(total_gate_preview_targets)):
                            target_vec = np.asarray(total_gate_preview_targets[int(pick)], dtype=np.float32).reshape(-1)
                        g2_rows: List[str] = [f"pass={1 if bool(gate2_pass) else 0}"]
                        if isinstance(gate2_detail, dict):
                            g2_rows.append(f"mode={str(gate2_detail.get('mode', ''))}")
                            g2_rows.append(f"reason={str(gate2_detail.get('reason', ''))}")
                            if gate2_detail.get("loss_target", None) is not None:
                                try:
                                    g2_rows.append(f"target={float(gate2_detail.get('loss_target', 0.0)):.4f}")
                                except Exception:
                                    pass
                            g2_rows.append(
                                f"cov={int(gate2_detail.get('refresh_samples_seen', 0))}/"
                                f"{int(gate2_detail.get('refresh_samples_required', 0))}"
                            )
                        if isinstance(gate2_eval, dict):
                            try:
                                g2_rows.append(f"loss={float(gate2_eval.get('loss', float('nan'))):.4f}")
                            except Exception:
                                pass
                            try:
                                g2_rows.append(f"conf={float(gate2_eval.get('mean_confidence', float('nan'))):.4f}")
                            except Exception:
                                pass
                            try:
                                g2_rows.append(f"macro_f1={float(gate2_eval.get('macro_f1', float('nan'))):.4f}")
                            except Exception:
                                pass
                        g2_step_txt = f"sample={int(pick)}"
                        if isinstance(gate2_eval, dict):
                            try:
                                g2_loss = float(gate2_eval.get("loss", float("nan")))
                                if math.isfinite(g2_loss):
                                    g2_step_txt = f"{g2_step_txt} loss={g2_loss:.4f}"
                            except Exception:
                                pass
                        _render_semantic_standard(
                            stage_emit_key="G2",
                            img_np=np.asarray(total_gate_preview_images[int(pick)], dtype=np.float32),
                            target_vec=target_vec,
                            step_txt=g2_step_txt,
                            detail_rows=g2_rows,
                        )
                return

            if stage_key == "C":
                if do_gl and stage_opengl_viewer.enabled:
                    try:
                        stage_idx = -1
                        stage_record_idx = -1
                        stage_metric: Dict[str, Any] = {}
                        if isinstance(extra, dict):
                            try:
                                stage_idx = int(extra.get("stage_c_stream_index", -1))
                            except Exception:
                                stage_idx = -1
                            try:
                                stage_record_idx = int(extra.get("stage_c_record_index", -1))
                            except Exception:
                                stage_record_idx = -1
                            if isinstance(extra.get("stage_c_metric", None), dict):
                                stage_metric = dict(extra.get("stage_c_metric", {}))
                        if not (0 <= int(stage_idx) < int(len(val_streams))) and (0 <= int(stage_record_idx) < int(len(records))):
                            rec_path = str(getattr(records[int(stage_record_idx)], "path", "")).strip().lower()
                            if rec_path:
                                for i_meta, meta in enumerate(val_meta):
                                    p_meta = str(meta.get("path", "")).strip().lower() if isinstance(meta, dict) else ""
                                    if p_meta and p_meta == rec_path:
                                        stage_idx = int(i_meta)
                                        break
                        if not (0 <= int(stage_idx) < int(len(val_streams))) and int(len(val_streams)) > 0:
                            stage_idx = _next_stage_index(
                                "C",
                                len(val_streams),
                                seed_term=(int(args.seed) + (int(cycle_id) * 1901) + (int(round_id) * 307) + int(global_round)),
                            )
                        if not (0 <= int(stage_idx) < int(len(val_streams))):
                            _log("[stage-opengl] C preview skipped: no stage-c validation stream.")
                        else:
                            clean_wave = _chunk_wave_for_preview(val_streams[int(stage_idx)])
                            with torch.no_grad():
                                xb = torch.from_numpy(clean_wave[None, :]).to(device)
                                img = render_mono_wave_to_tensor(
                                    xb, cfg=best_cfg, image_hw=image_hw, sample_bits=int(sample_bits)
                                )
                                cls_img = img
                                if bool(args.channels_last):
                                    cls_img = cls_img.contiguous(memory_format=torch.channels_last)
                                probs = torch.sigmoid(classifier(cls_img)).to(torch.float32)
                            x0 = img[0].detach().to(torch.float32).cpu()
                            p0 = probs[0].detach().cpu().numpy().astype(np.float32, copy=False)
                            target_line = "target:none"
                            if (
                                val_stream_target_labels is not None
                                and 0 <= int(stage_idx) < int(len(val_stream_target_labels))
                                and val_stream_target_labels[int(stage_idx)] is not None
                            ):
                                target_line = _format_target_line_from_condition(
                                    val_stream_target_labels[int(stage_idx)],
                                    class_names=class_names,
                                    max_items=0,
                                    threshold=0.5,
                                )
                            detail_rows: List[str] = [f"stream_idx={int(stage_idx)}"]
                            if isinstance(stage_metric, dict):
                                for key_txt, lbl in (
                                    ("score", "score"),
                                    ("topk_mean", "topk"),
                                    ("hard_coverage", "coverage"),
                                    ("mean_entropy", "entropy"),
                                ):
                                    if key_txt in stage_metric:
                                        try:
                                            detail_rows.append(f"{lbl}={float(stage_metric.get(key_txt, 0.0)):.4f}")
                                        except Exception:
                                            pass
                            if int(len(detail_rows)) <= 1:
                                detail_rows.append("stats=unavailable")
                            _stage_gl_emit_standard(
                                stage_key="C",
                                cycle_id=int(cycle_id),
                                round_id=int(round_id),
                                step_txt=f"sample={int(stage_idx)} src=stage",
                                clean_img=x0,
                                input_img=x0,
                                output_img=x0,
                                probs_in=p0,
                                probs_out=p0,
                                label_names=class_names,
                                target_line_override=target_line,
                                target_extra_rows=detail_rows,
                                extra_out_rows=detail_rows,
                            )
                    except Exception as e:
                        _log(f"[stage-opengl] classifier preview failed: {e}")
                # C-stage preview must not fall through to unrelated generic preview paths.
                return

            if (
                do_gl
                and stage_opengl_viewer.enabled
                and (stage_key in ("G", "J"))
                and (generator is not None)
                and (len(payload_conditions) > 0)
                and (len(payload_images) > 0)
            ):
                try:
                    pick = _next_stage_index(
                        stage_key,
                        len(payload_conditions),
                        seed_term=(int(args.seed) + (int(cycle_id) * 3001) + (int(round_id) * 431) + int(global_round)),
                    )
                    cond_vec = np.asarray(payload_conditions[pick], dtype=np.float32).reshape(-1)
                    cond = torch.from_numpy(cond_vec[None, :]).to(device=device, dtype=torch.float32)
                    z = torch.randn((1, max(8, int(args.generator_z_dim))), device=device)
                    was_gen_train = bool(generator.training)
                    was_cls_train = bool(classifier.training)
                    generator.eval()
                    classifier.eval()
                    try:
                        target_np = np.asarray(payload_images[pick], dtype=np.float32, order="C")
                        target_img = torch.from_numpy(target_np[None, ...]).to(device=device, dtype=torch.float32)
                        if bool(args.channels_last):
                            target_img = target_img.contiguous(memory_format=torch.channels_last)
                        amp_dtype_t = _resolve_amp_dtype(args.amp_dtype) if amp_enabled else torch.float16
                        with _autocast_context(device=device, enabled=amp_enabled, amp_dtype_t=amp_dtype_t):
                            fake = generator(z, cond).to(torch.float32)
                            if bool(args.channels_last):
                                fake = fake.contiguous(memory_format=torch.channels_last)
                            logits_fake = classifier(fake)
                            probs_fake = torch.sigmoid(logits_fake).to(torch.float32)
                            logits_target = classifier(target_img)
                            probs_target = torch.sigmoid(logits_target).to(torch.float32)
                    finally:
                        if was_gen_train:
                            generator.train()
                        if was_cls_train:
                            classifier.train()
                    p_fake = probs_fake[0].detach().cpu().numpy().astype(np.float32, copy=False)
                    p_target = probs_target[0].detach().cpu().numpy().astype(np.float32, copy=False)
                    top_fake = _top_labels_from_probs(
                        p_fake,
                        class_names=class_names,
                        topk=max(1, min(len(class_names), int(p_fake.shape[0]))),
                    )
                    top_target = _top_labels_from_probs(
                        p_target,
                        class_names=class_names,
                        topk=max(1, min(len(class_names), int(p_target.shape[0]))),
                    )
                    top_fake_lines = _format_top_label_lines(top_fake, max_items=0)
                    top_target_lines = _format_top_label_lines(top_target, max_items=0)
                    target_line = _format_target_line_from_condition(
                        cond_vec,
                        class_names=class_names,
                        max_items=0,
                        threshold=0.5,
                    )
                    tgt_rows = _semantic_target_entries_from_vector(
                        cond_vec,
                        class_names=class_names,
                        max_items=int(num_classes),
                        threshold=0.5,
                    )
                    tgt_ids = [int(r.get("class_idx", -1)) for r in tgt_rows if int(r.get("class_idx", -1)) >= 0]
                    tgt_prob_fake = 0.0
                    tgt_prob_target = 0.0
                    if len(tgt_ids) > 0:
                        tgt_prob_fake = float(np.mean([float(p_fake[int(i)]) for i in tgt_ids if int(i) < int(p_fake.shape[0])]))
                        tgt_prob_target = float(
                            np.mean([float(p_target[int(i)]) for i in tgt_ids if int(i) < int(p_target.shape[0])])
                        )
                    target_rows = [target_line, f"target_ref_prob:{tgt_prob_target:.3f}"] + top_target_lines
                    input_rows = top_fake_lines if len(top_fake_lines) > 0 else ["n/a"]
                    out_rows = [target_line, f"fake_target_prob:{tgt_prob_fake:.3f}"] + input_rows
                    top_txt = "n/a" if len(top_fake_lines) <= 0 else top_fake_lines[0]
                    stage_opengl_viewer.update(
                        clean_img=target_img[0],
                        input_img=fake[0],
                        output_img=fake[0],
                        caption=f"[{stage_key}] cycle={int(cycle_id)} round={int(round_id)} sample={int(pick)} top={top_txt}",
                        panel_titles=[f"{stage_key} target ref", f"{stage_key} fake", f"{stage_key} summary"],
                        panel_rows=[target_rows, input_rows, out_rows],
                    )
                    _log(f"[stage-opengl] {stage_key} sample={int(pick)} top={top_txt}")
                except Exception as e:
                    _log(f"[stage-opengl] generator preview failed: {e}")

            if len(val_streams) <= 0:
                return
            sample_idx = _next_stage_index(
                stage_key,
                len(val_streams),
                seed_term=(int(args.seed) + (int(cycle_id) * 1009) + (int(round_id) * 131) + int(global_round)),
            )
            clean_wave = _chunk_wave_for_preview(val_streams[sample_idx])
            target_condition = None
            if val_stream_target_labels is not None and sample_idx < len(val_stream_target_labels):
                tv = val_stream_target_labels[sample_idx]
                if tv is not None:
                    target_condition = np.asarray(tv, dtype=np.float32).reshape(-1)
            with torch.no_grad():
                xb = torch.from_numpy(clean_wave[None, :]).to(device)
                enhanced_wave = transformer(xb).detach().cpu().numpy()[0].astype(np.float32, copy=False)

            if do_gl and stage_opengl_viewer.enabled and (stage_key not in ("G", "J", "C")):
                try:
                    with torch.no_grad():
                        x_clean_t = torch.from_numpy(clean_wave[None, :]).to(device)
                        x_out_t = torch.from_numpy(enhanced_wave[None, :]).to(device)
                        img_clean = render_mono_wave_to_tensor(
                            x_clean_t, cfg=best_cfg, image_hw=image_hw, sample_bits=int(sample_bits)
                        )
                        img_out = render_mono_wave_to_tensor(
                            x_out_t, cfg=best_cfg, image_hw=image_hw, sample_bits=int(sample_bits)
                        )
                        cls_clean = img_clean
                        cls_out = img_out
                        if bool(args.channels_last):
                            cls_clean = cls_clean.contiguous(memory_format=torch.channels_last)
                            cls_out = cls_out.contiguous(memory_format=torch.channels_last)
                        probs_clean = torch.sigmoid(classifier(cls_clean)).to(torch.float32)[0].detach().cpu().numpy().astype(np.float32, copy=False)
                        probs_out = torch.sigmoid(classifier(cls_out)).to(torch.float32)[0].detach().cpu().numpy().astype(np.float32, copy=False)
                    top_clean = _top_labels_from_probs(
                        probs_clean,
                        class_names=class_names,
                        topk=max(1, min(len(class_names), int(probs_clean.shape[0]))),
                    )
                    top_out = _top_labels_from_probs(
                        probs_out,
                        class_names=class_names,
                        topk=max(1, min(len(class_names), int(probs_out.shape[0]))),
                    )
                    top_clean_lines = _format_top_label_lines(top_clean, max_items=0)
                    top_out_lines = _format_top_label_lines(top_out, max_items=0)
                    if target_condition is not None:
                        target_line = _format_target_line_from_condition(
                            target_condition,
                            class_names=class_names,
                            max_items=0,
                            threshold=0.5,
                        )
                    else:
                        target_line = "target:none"
                    clean_rows = [target_line] + (top_clean_lines if len(top_clean_lines) > 0 else ["n/a"])
                    input_rows = top_clean_lines if len(top_clean_lines) > 0 else ["n/a"]
                    out_rows = [target_line] + (top_out_lines if len(top_out_lines) > 0 else ["n/a"])
                    top_txt = "n/a" if len(top_out_lines) <= 0 else top_out_lines[0]
                    stage_opengl_viewer.update(
                        clean_img=img_clean[0],
                        input_img=img_clean[0],
                        output_img=img_out[0],
                        caption=f"[{stage_key}] cycle={int(cycle_id)} round={int(round_id)} sample={int(sample_idx)} top={top_txt}",
                        panel_titles=[f"{stage_key} clean", f"{stage_key} input", f"{stage_key} out"],
                        panel_rows=[clean_rows, input_rows, out_rows],
                    )
                    _log(f"[stage-opengl] {stage_key} sample={int(sample_idx)} top={top_txt}")
                except Exception as e:
                    _log(f"[stage-opengl] waveform preview failed: {e}")

            if not do_file:
                return
            payload = _save_round_supervision_preview(
                preview_dir=training_preview_root,
                stage=str(stage),
                cycle_id=int(cycle_id),
                round_id=int(round_id),
                sample_idx=int(sample_idx),
                clean_wave=clean_wave,
                enhanced_wave=enhanced_wave,
                cfg=best_cfg,
                sample_bits=int(sample_bits),
                image_hw=image_hw,
                classifier=classifier,
                class_names=class_names,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=args.amp_dtype,
                channels_last=bool(args.channels_last),
                topk=int(training_preview_topk),
                target_condition=target_condition,
                extra=extra,
            )
            if isinstance(payload, dict):
                clean_top = payload.get("top_labels_clean", [])
                out_top = payload.get("top_labels_enhanced", [])
                if isinstance(clean_top, list) and len(clean_top) > 0 and isinstance(out_top, list) and len(out_top) > 0:
                    _log(
                        "Training preview "
                        f"[{stage}] cycle={int(cycle_id)} round={int(round_id)} sample={int(sample_idx)} "
                        f"{clean_top[0].get('class_name', '?')}:{clean_top[0].get('score', 0.0):.3f}"
                        f" -> {out_top[0].get('class_name', '?')}:{out_top[0].get('score', 0.0):.3f}"
                    )
            _append_training_preview_manifest(payload)

        if mode == "none":
            _poll_gui_stop(note="mode-none")
            gestation_gate_ready = (
                gestation_gate_streak >= gate_config["gestation_maintain_rounds"]
            )
            berkeley_gate_ready = (
                berkeley_gate_streak >= gate_config["berkeley_maintain_rounds"]
            )
            wave_gate_ready = wave_gate_streak >= gate_config["wave_maintain_rounds"]
            if gestation_gate_ready and berkeley_gate_ready:
                if _poll_gui_stop(note="mode-none-transformer-start"):
                    _log("Transformer training skipped due to GUI close request.")
                elif wave_gate_ready:
                    _offload_inactive_stage_models(reason="mode-none-transformer")
                    transformer, run_hist, _ = _train_transformer_stage_with_retry(
                        stage_tag="R",
                        epochs=int(args.transformer_epochs),
                        steps_per_epoch=int(args.transformer_steps),
                        score_target_weight=float(args.transformer_loss_score_target_weight),
                        score_target_margin=float(args.transformer_loss_score_target_margin),
                        seed=int(args.seed + 9000),
                        step_preview_callback=_make_r_step_callback("R", cycle_id=0, round_id=0),
                    )
                    transformer_hist.extend(run_hist)
                    _save_training_segment_snapshot(
                        enabled=bool(args.checkpoint_after_training_segment),
                        out_dir=out_dir,
                        objective_mode=args.objective_mode,
                        run_tag=run_tag,
                        segment="mode_none_transformer_train",
                        best_cfg=best_cfg,
                        classifier=classifier,
                        transformer=transformer,
                        transformer_history=transformer_hist,
                        wave_classifier_history=wave_classifier_hist,
                        orchestration_history=orchestration_rows,
                        refresh_history=refresh_rows,
                        gate_history=gate_history,
                        gate_status={
                            "gestation_streak": int(gestation_gate_streak),
                            "berkeley_streak": int(berkeley_gate_streak),
                            "wave_streak": int(wave_gate_streak),
                            "generator_streak": int(generator_gate_streak),
                            "transformer_streak": int(transformer_gate_streak),
                        },
                        extra={
                            "sample_bits": int(sample_bits),
                            "chunk_samples": int(chunk_samples),
                            "patch_size": int(args.patch_size),
                        },
                    )
                elif not wave_gate_ready:
                    _log(
                        "Transformer training skipped by wave-entropy gate "
                        f"(streak={wave_gate_streak}/{gate_config['wave_maintain_rounds']}, "
                        f"min_entropy={gate_config['wave_min_entropy']:.4f})."
                    )
            elif not bool(gestation_gate_ready):
                _log(
                    "Transformer training skipped by gestation gate "
                    f"(streak={gestation_gate_streak}/{gate_config['gestation_maintain_rounds']}, "
                    f"loss_target={float(gate_config['gestation_loss_target']):.4f})."
                )
            else:
                _log(
                    "Transformer training skipped by Stage-2 validation gate "
                    f"(streak={berkeley_gate_streak}/{gate_config['berkeley_maintain_rounds']}, "
                    f"min_conf={gate_config['berkeley_min_confidence']:.4f}, "
                    f"min_macro_f1={gate_config['berkeley_min_macro_f1']:.4f})."
                )
            _save_pipeline_checkpoint(
                pipeline_checkpoint_path,
                _decorate_checkpoint_payload(
                    {
                        "run_tag": run_tag,
                        "objective_mode": args.objective_mode,
                        "orchestration_mode": mode,
                        "last_cycle": cycle_offset,
                        "last_round": 0,
                        "best_cfg": best_cfg.to_dict(),
                        "classifier_state": classifier.state_dict(),
                        "transformer_state": transformer.state_dict(),
                        "generator_state": (generator.state_dict() if generator is not None else None),
                        "discriminator_state": (discriminator.state_dict() if discriminator is not None else None),
                        "wave_classifier_state": None,
                        "classifier_init_info": classifier_init_info,
                        "classifier_resume_info": classifier_resume_info,
                        "transformer_resume_info": transformer_resume_info,
                        "generator_resume_info": generator_resume_info,
                        "discriminator_resume_info": discriminator_resume_info,
                        "wave_classifier_init_info": wave_classifier_init_info,
                        "wave_classifier_resume_info": wave_classifier_resume_info,
                        "transformer_history": transformer_hist,
                        "generator_history": generator_hist,
                        "wave_classifier_history": wave_classifier_hist,
                        "orchestration_history": orchestration_rows,
                        "refresh_history": refresh_rows,
                        "gate_history": gate_history,
                        "gate_status": {
                            "gestation_streak": int(gestation_gate_streak),
                            "berkeley_streak": int(berkeley_gate_streak),
                            "wave_streak": int(wave_gate_streak),
                            "generator_streak": int(generator_gate_streak),
                            "transformer_streak": int(transformer_gate_streak),
                        },
                        "wave_classifier_last_eval": wave_classifier_last_eval,
                        "wave_zero_shot_query_source": str(wave_zero_shot_query_source),
                        "wave_zero_shot_queries": list(wave_zero_shot_queries),
                        "wave_zero_shot_query_rows": list(wave_zero_shot_query_rows),
                        "wave_zero_shot_last_eval": wave_zero_shot_last_eval,
                        "wave_zero_shot_history": list(wave_zero_shot_history),
                        "accepted_library_count": library_serial,
                        "sample_bits": int(sample_bits),
                        "chunk_samples": int(chunk_samples),
                        "patch_size": int(args.patch_size),
                        "timestamp": time.time(),
                    }
                ),
            )
        elif mode in ("staged_cgr", "staged_cgrw"):
            if generator is None or discriminator is None:
                raise RuntimeError(f"{mode} mode requires generator/discriminator initialization.")

            def _save_staged_round_checkpoint(last_cycle: int, last_round: int):
                _save_pipeline_checkpoint(
                    pipeline_checkpoint_path,
                    _decorate_checkpoint_payload(
                        {
                            "run_tag": run_tag,
                            "objective_mode": args.objective_mode,
                            "orchestration_mode": mode,
                            "last_cycle": int(last_cycle),
                            "last_round": int(last_round),
                            "best_cfg": best_cfg.to_dict(),
                            "classifier_state": classifier.state_dict(),
                            "transformer_state": transformer.state_dict(),
                            "generator_state": generator.state_dict(),
                            "discriminator_state": discriminator.state_dict(),
                            "wave_classifier_state": (wave_classifier.state_dict() if wave_classifier is not None else None),
                            "classifier_init_info": classifier_init_info,
                            "classifier_resume_info": classifier_resume_info,
                            "transformer_resume_info": transformer_resume_info,
                            "generator_resume_info": generator_resume_info,
                            "discriminator_resume_info": discriminator_resume_info,
                            "wave_classifier_init_info": wave_classifier_init_info,
                            "wave_classifier_resume_info": wave_classifier_resume_info,
                            "transformer_history": transformer_hist,
                            "generator_history": generator_hist,
                            "wave_classifier_history": wave_classifier_hist,
                            "orchestration_history": orchestration_rows,
                            "refresh_history": refresh_rows,
                            "gate_history": gate_history,
                            "gate_status": {
                                "gestation_streak": int(gestation_gate_streak),
                                "berkeley_streak": int(berkeley_gate_streak),
                                "wave_streak": int(wave_gate_streak),
                                "generator_streak": int(generator_gate_streak),
                                "transformer_streak": int(transformer_gate_streak),
                            },
                            "wave_classifier_last_eval": wave_classifier_last_eval,
                            "wave_zero_shot_query_source": str(wave_zero_shot_query_source),
                            "wave_zero_shot_queries": list(wave_zero_shot_queries),
                            "wave_zero_shot_query_rows": list(wave_zero_shot_query_rows),
                            "wave_zero_shot_last_eval": wave_zero_shot_last_eval,
                            "wave_zero_shot_history": list(wave_zero_shot_history),
                            "accepted_library_count": int(library_serial),
                            "sample_bits": int(sample_bits),
                            "chunk_samples": int(chunk_samples),
                            "patch_size": int(args.patch_size),
                            "timestamp": time.time(),
                        }
                    ),
                )
                if bool(args.gd_vocab_library_enabled):
                    _save_gd_vocab_library_snapshot(
                        library_dir=gd_vocab_library_dir,
                        vocab_hash=str(active_gd_vocab_hash),
                        condition_num_classes=int(condition_num_classes),
                        generator=generator,
                        discriminator=discriminator,
                        meta={
                            "run_tag": str(run_tag),
                            "source": "staged_round_checkpoint",
                            "cycle": int(last_cycle),
                            "round": int(last_round),
                            "active_extra_terms": list(active_extra_terms),
                            "semantic_churn_history_size": int(len(semantic_churn_history)),
                            "symbol_pool_info": dict(symbol_pool_info),
                            "vocab_profile": dict(active_gd_vocab_profile),
                        },
                    )

            def _run_staged_fake_feedback(stage_tag: str, cycle_id: int, round_id: int, global_round: int):
                if (not bool(fake_feedback_enabled)) or (fake_label_vector_t is None):
                    return None
                if generator is None:
                    return None
                if len(payload_conditions) <= 0:
                    return None
                fb = _run_fake_class_refresh_epochs(
                    classifier=classifier,
                    generator=generator,
                    discriminator=discriminator,
                    payload_conditions=payload_conditions,
                    condition_num_classes=int(condition_num_classes),
                    fake_label_vector=fake_label_vector_t,
                    z_dim=max(8, int(args.generator_z_dim)),
                    device=device,
                    epochs=max(1, int(args.generator_fake_feedback_epochs)),
                    steps_per_epoch=max(1, int(args.generator_fake_feedback_steps_per_round)),
                    batch_size=max(1, int(args.generator_fake_feedback_batch_size)),
                    lr=float(args.generator_fake_feedback_lr),
                    weight_decay=float(args.generator_fake_feedback_weight_decay),
                    lr_sine_cycles=float(args.lr_sine_cycles),
                    lr_sine_frequency=float(args.lr_sine_frequency),
                    lr_sine_tail_fraction=float(args.lr_sine_tail_fraction),
                    lr_sine_min_scale=float(args.lr_sine_min_scale),
                    amp_enabled=amp_enabled,
                    amp_dtype=args.amp_dtype,
                    channels_last=bool(args.channels_last),
                    grad_accum_steps=grad_accum_steps,
                    log_every=max(0, int(args.generator_fake_feedback_log_every)),
                    include_condition_targets=bool(args.generator_fake_feedback_include_condition_targets),
                    fake_vector_weight=float(args.generator_fake_feedback_vector_weight),
                    condition_target_weight=float(args.generator_fake_feedback_condition_weight),
                    disc_conf_temperature=float(args.generator_fake_feedback_disc_conf_temperature),
                    disc_conf_floor=float(args.generator_fake_feedback_disc_conf_floor),
                    balance_disc_groups=bool(args.generator_fake_feedback_disc_balance_groups),
                    step_preview_callback=None,
                    stop_requested=_poll_gui_stop,
                    seed=int(args.seed + 19000 + (int(global_round) * 17)),
                )
                refresh_rows.append({"stage": f"{str(stage_tag)}_fake_feedback", "cycle": cycle_id, "round": round_id, **fb})
                _log(
                    "  fake-feedback refresh: "
                    f"loss={float(fb.get('loss', 0.0)):.4f} "
                    f"samples={int(fb.get('samples', 0))} "
                    f"disc_pass={float(fb.get('disc_pass_rate', 0.0)):.3f} "
                    f"disc_fail={float(fb.get('disc_fail_rate', 0.0)):.3f} "
                    f"disc_conf={float(fb.get('disc_conf_mean', 0.0)):.3f} "
                    f"fake_cos={float(fb.get('fake_cos_mean', 0.0)):.3f}"
                )
                return fb

            run_wave_stage = bool(mode == "staged_cgrw")
            if run_wave_stage and wave_classifier is None:
                wave_classifier = TinyConvClassifier(
                    num_classes=split_num_classes,
                    base_ch=int(args.classifier_base_ch),
                    max_ch=int(args.classifier_max_ch),
                    context_blocks=int(args.classifier_context_blocks),
                    context_dropout=float(args.classifier_context_dropout),
                )
                wave_classifier_init_info = _apply_classifier_init(
                    model=wave_classifier,
                    ckpt_path=args.classifier_init_ckpt,
                    scope=args.classifier_init_scope,
                )
                if resume_enabled and resume_wave_classifier_path.exists():
                    wave_classifier_resume_info = _apply_model_init(wave_classifier, str(resume_wave_classifier_path))
                    _log(
                        f"Resumed wave classifier from {resume_wave_classifier_path} "
                        f"(loaded={wave_classifier_resume_info.get('loaded_keys', 0)}, "
                        f"skipped={wave_classifier_resume_info.get('skipped_keys', 0)})"
                    )
                elif isinstance(resume_pipeline_ckpt, dict) and isinstance(resume_pipeline_ckpt.get("wave_classifier_state"), dict):
                    wave_classifier_resume_info = _apply_state_dict(
                        wave_classifier,
                        resume_pipeline_ckpt.get("wave_classifier_state"),
                        source_name="pipeline_checkpoint",
                    )
                    _log(
                        "Resumed wave classifier from pipeline checkpoint "
                        f"(loaded={wave_classifier_resume_info.get('loaded_keys', 0)}, "
                        f"skipped={wave_classifier_resume_info.get('skipped_keys', 0)})"
                    )
                wave_label_embed_apply_info = _apply_label_embedding_bank_to_classifier(
                    classifier=wave_classifier,
                    bank_np=(None if bool(split_labels_from_pseudo_spectral) else split_label_embedding_bank),
                    args=args,
                )
                if bool(split_labels_from_pseudo_spectral):
                    _log("[label-embeddings] wave classifier split bank disabled for spectral pseudo labels.")
                if bool(wave_label_embed_apply_info.get("applied", False)):
                    _log(
                        "[label-embeddings] wave classifier active: "
                        f"classes={wave_label_embed_apply_info.get('classes', 0)} "
                        f"dim={wave_label_embed_apply_info.get('dim', 0)} "
                        f"temp={float(wave_label_embed_apply_info.get('temperature', 0.0)):.2f}"
                    )
                if args.classifier_freeze_features:
                    _set_feature_freeze(wave_classifier, freeze=True)
                if args.channels_last:
                    wave_classifier = wave_classifier.to(memory_format=torch.channels_last)
                wave_classifier = maybe_compile_module(
                    wave_classifier,
                    enabled=bool(args.compile_models),
                    mode=str(args.compile_mode),
                )

            accepted_stream_seen: set = set()
            if run_wave_stage:
                preload_cap = max(0, int(args.transformer_accepted_preload_max))
                if preload_cap > 0 and library_index_path.exists():
                    preload_rows = []
                    try:
                        with library_index_path.open("r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    preload_rows.append(json.loads(line))
                                except Exception:
                                    continue
                    except Exception:
                        preload_rows = []
                    if len(preload_rows) > preload_cap:
                        preload_rows = preload_rows[-preload_cap:]
                    preload_added = _append_accepted_rows_to_transformer_streams(
                        rows=preload_rows,
                        seen_paths=accepted_stream_seen,
                        train_streams=train_streams,
                        train_stream_labels=train_stream_labels,
                        train_meta=train_meta,
                        max_points=int(args.stream_max_points),
                        library_root=library_dir,
                        on_append_row=lambda r: _register_regurgitated_churn_row(r, source="accepted_preload"),
                    )
                    if preload_added > 0:
                        if train_stream_target_labels is not None:
                            _append_default_mix_targets(train_stream_target_labels, int(preload_added))
                        _log(
                            "Preloaded accepted-library streams for transformer: "
                            f"added={preload_added} total_train_streams={len(train_streams)}"
                        )

            def _run_wave_stage_after_transformer(round_row: Dict[str, Any], cycle_id: int, round_id: int, global_round: int):
                nonlocal wave_classifier, wave_classifier_last_eval, wave_zero_shot_last_eval, library_serial, transformer_gate_streak
                if (not run_wave_stage) or (wave_classifier is None):
                    return round_row

                x_wave_train, y_wave_train, accepted_rows = _build_wave_classifier_dataset_from_transformer(
                    streams=train_streams,
                    labels=train_stream_labels,
                    metas=train_meta,
                    cfg=best_cfg,
                    transformer=transformer,
                    berkeley_classifier=classifier,
                    sample_bits=sample_bits,
                    image_hw=image_hw,
                    chunk_samples=chunk_samples,
                    total_samples=max(1, int(args.wave_cls_train_samples)),
                    batch_size=max(1, int(args.wave_cls_batch_size)),
                    device=device,
                    rng_seed=args.seed + 12000 + (global_round * 11),
                    accept_score_threshold=float(args.accept_score_threshold),
                    accept_l1_threshold=float(args.accept_l1_threshold),
                    library_dir=library_dir,
                    library_limit=max(0, int(args.library_max_per_round)),
                    cycle_id=cycle_id,
                    round_id=round_id,
                    accepted_only=bool(args.wave_cls_accepted_only),
                    run_tag=run_tag,
                    library_serial_start=library_serial,
                    amp_enabled=amp_enabled,
                    amp_dtype=args.amp_dtype,
                    channels_last=bool(args.channels_last),
                    pin_memory=bool(args.pin_memory_wave_batches),
                    semantic_class_names=class_names,
                    semantic_label_bank=label_embedding_bank,
                    stream_target_labels=train_stream_target_labels,
                )
                x_wave_val, y_wave_val, _ = _build_wave_classifier_dataset_from_transformer(
                    streams=val_streams,
                    labels=val_stream_labels,
                    metas=val_meta,
                    cfg=best_cfg,
                    transformer=transformer,
                    berkeley_classifier=classifier,
                    sample_bits=sample_bits,
                    image_hw=image_hw,
                    chunk_samples=chunk_samples,
                    total_samples=max(1, int(args.wave_cls_val_samples)),
                    batch_size=max(1, int(args.wave_cls_batch_size)),
                    device=device,
                    rng_seed=args.seed + 12100 + (global_round * 11),
                    accept_score_threshold=float(args.accept_score_threshold),
                    accept_l1_threshold=float(args.accept_l1_threshold),
                    library_dir=library_dir,
                    library_limit=0,
                    cycle_id=cycle_id,
                    round_id=round_id,
                    accepted_only=bool(args.wave_cls_accepted_only),
                    run_tag=run_tag,
                    library_serial_start=library_serial,
                    amp_enabled=amp_enabled,
                    amp_dtype=args.amp_dtype,
                    channels_last=bool(args.channels_last),
                    pin_memory=bool(args.pin_memory_wave_batches),
                    semantic_class_names=class_names,
                    semantic_label_bank=label_embedding_bank,
                    stream_target_labels=val_stream_target_labels,
                )

                accepted_total.extend(accepted_rows)
                library_serial += len(accepted_rows)
                append_cap = max(0, int(args.transformer_accepted_append_max_per_round))
                to_append = accepted_rows if append_cap <= 0 else accepted_rows[:append_cap]
                appended_streams = _append_accepted_rows_to_transformer_streams(
                    rows=to_append,
                    seen_paths=accepted_stream_seen,
                    train_streams=train_streams,
                    train_stream_labels=train_stream_labels,
                    train_meta=train_meta,
                    max_points=int(args.stream_max_points),
                    library_root=library_dir,
                    on_append_row=lambda r: _register_regurgitated_churn_row(r, source="accepted_round_append"),
                )
                if appended_streams > 0 and train_stream_target_labels is not None:
                    _append_default_mix_targets(train_stream_target_labels, int(appended_streams))
                round_row["transformer_train_stream_add"] = int(appended_streams)
                round_row["transformer_train_stream_total"] = int(len(train_streams))
                if appended_streams > 0:
                    _log(
                        "  transformer stream pool extended from accepted library: "
                        f"+{appended_streams} (total={len(train_streams)})"
                    )

                round_row["wave_train_samples"] = int(x_wave_train.shape[0])
                round_row["wave_val_samples"] = int(x_wave_val.shape[0])
                round_row["accepted_library"] = len(accepted_rows)
                can_train_wave = (
                    int(x_wave_train.shape[0]) >= 2
                    and int(x_wave_val.shape[0]) >= 2
                    and int(torch.unique(y_wave_train).numel()) >= 2
                )
                if can_train_wave:
                    wave_classifier, round_wave_hist = train_classifier(
                        model=wave_classifier,
                        x_train=x_wave_train,
                        y_train=y_wave_train,
                        x_val=x_wave_val,
                        y_val=y_wave_val,
                        device=device,
                        epochs=max(1, int(args.wave_cls_epochs_per_round)),
                        batch_size=max(1, int(args.wave_cls_batch_size)),
                        lr=float(args.wave_cls_lr),
                        lr_sine_cycles=args.lr_sine_cycles,
                        lr_sine_frequency=args.lr_sine_frequency,
                        lr_sine_tail_fraction=args.lr_sine_tail_fraction,
                        lr_sine_min_scale=args.lr_sine_min_scale,
                        amp=amp_enabled,
                        amp_dtype=args.amp_dtype,
                        channels_last=bool(args.channels_last),
                        grad_accum_steps=grad_accum_steps,
                        cache_dataset_on_device=bool(args.cache_wave_cls_dataset_on_device),
                        seed=args.seed + 13000 + global_round,
                        log_every_steps=max(0, int(args.wave_cls_log_every)),
                        step_preview_callback=_make_w_step_callback(cycle_id=int(cycle_id), round_id=int(round_id)),
                        stop_requested=_poll_gui_stop,
                        semantic_mix_noise_prob=float(args.wave_cls_semantic_mix_noise_prob),
                        semantic_mix_noise_std=float(args.wave_cls_semantic_mix_noise_std),
                        semantic_mix_blend_min=float(args.wave_cls_semantic_mix_blend_min),
                        semantic_mix_blend_max=float(args.wave_cls_semantic_mix_blend_max),
                        semantic_mix_class_index=int(split_mix_class_index),
                    )
                    for row in round_wave_hist:
                        rr = dict(row)
                        rr["cycle"] = float(cycle_id)
                        rr["round"] = float(round_id)
                        wave_classifier_hist.append(rr)
                    wave_classifier_last_eval = evaluate_classifier(
                        model=wave_classifier,
                        x=x_wave_val,
                        y=y_wave_val,
                        batch_size=max(1, int(args.wave_cls_batch_size)),
                        device=device,
                        amp=amp_enabled,
                        amp_dtype=args.amp_dtype,
                        channels_last=bool(args.channels_last),
                    )
                    round_row["wave_classifier_val"] = wave_classifier_last_eval
                    run_wave_zero_shot = (
                        wave_zero_shot_query_emb is not None
                        and len(wave_zero_shot_queries) > 0
                    )
                    if run_wave_zero_shot and bool(wave_zero_shot_single_probe_once) and len(wave_zero_shot_history) > 0:
                        run_wave_zero_shot = False
                    if run_wave_zero_shot:
                        wave_zero_shot_last_eval = _evaluate_zero_shot_queries_on_images(
                            classifier=wave_classifier,
                            x=x_wave_val,
                            query_texts=wave_zero_shot_queries,
                            query_emb_np=wave_zero_shot_query_emb,
                            batch_size=max(1, int(args.wave_cls_batch_size)),
                            device=device,
                            amp_enabled=amp_enabled,
                            amp_dtype=args.amp_dtype,
                            channels_last=bool(args.channels_last),
                            topk=max(1, int(args.wave_zero_shot_topk)),
                            max_samples=int(args.wave_zero_shot_max_samples),
                            max_report_samples=int(args.wave_zero_shot_report_samples),
                        )
                        round_row["wave_zero_shot"] = {
                            "ran": bool(wave_zero_shot_last_eval.get("ran", False)),
                            "reason": str(wave_zero_shot_last_eval.get("reason", "")),
                            "samples": int(wave_zero_shot_last_eval.get("samples", 0)),
                            "num_queries": int(wave_zero_shot_last_eval.get("num_queries", 0)),
                            "mean_top1_score": float(wave_zero_shot_last_eval.get("mean_top1_score", 0.0)),
                            "best_query": str(wave_zero_shot_last_eval.get("best_query", "")),
                            "best_query_mean_score": float(wave_zero_shot_last_eval.get("best_query_mean_score", 0.0)),
                            "best_query_top1_rate": float(wave_zero_shot_last_eval.get("best_query_top1_rate", 0.0)),
                        }
                        wave_zero_shot_history.append(
                            {
                                "cycle": float(cycle_id),
                                "round": float(round_id),
                                **round_row["wave_zero_shot"],
                            }
                        )
                    _log(
                        f"  wave classifier val: acc={wave_classifier_last_eval['acc']:.4f} "
                        f"loss={wave_classifier_last_eval['loss']:.4f} accepted={len(accepted_rows)}"
                    )
                else:
                    round_row["wave_classifier_val"] = None
                    round_row["wave_classifier_skipped"] = True
                    round_row["wave_unique_train_labels"] = int(torch.unique(y_wave_train).numel())
                    _log(
                        f"  wave classifier skipped (samples={x_wave_train.shape[0]}/{x_wave_val.shape[0]}, "
                        f"unique_train_labels={round_row['wave_unique_train_labels']})"
                    )

                feedback_post = _wave_feedback_snapshot(
                    wave_classifier_eval=wave_classifier_last_eval,
                    wave_zero_shot_eval=wave_zero_shot_last_eval,
                    zero_shot_weight=float(gate_config["wave_feedback_zero_shot_weight"]),
                )
                feedback_post_pass, feedback_post_reason = _wave_feedback_gate_pass(
                    feedback=feedback_post,
                    min_acc=float(gate_config["wave_feedback_min_acc"]),
                    min_zero_shot_top1_rate=float(gate_config["wave_feedback_min_zs_top1_rate"]),
                )
                round_row["wave_feedback_post"] = feedback_post
                round_row["wave_feedback_gate_post_pass"] = bool(feedback_post_pass)
                round_row["wave_feedback_gate_post_reason"] = str(feedback_post_reason)
                if not feedback_post_pass:
                    transformer_gate_streak = 0
                    round_row["transformer_streak"] = int(transformer_gate_streak)
                    round_row["transformer_ready"] = False
                    _log(f"  wave feedback gate reset transformer streak ({feedback_post_reason})")
                return round_row

            cycles = max(1, int(args.orchestration_cycles))
            rounds_per_cycle = max(1, int(args.orchestration_rounds))
            _sync_stage_cycle_roster(total_cycles=cycles)
            gate_override_boot = bool(_stage_gate_override_enabled())
            _log(
                f"Orchestration: mode={mode}, cycles={cycles}, rounds_per_cycle={rounds_per_cycle}, "
                f"generator_steps_per_round={int(args.generator_steps_per_round)}, "
                f"transformer_steps_per_round={int(args.transformer_steps_per_round)}, "
                f"wave_stage={1 if run_wave_stage else 0}, "
                f"resume_cycle_offset={cycle_offset}, "
                f"gate_override={1 if gate_override_boot else 0}"
            )

            global_round = len(orchestration_rows)
            stop_orchestration = False
            for cycle_local in range(1, cycles + 1):
                if stop_orchestration:
                    break
                cycle_id = cycle_offset + cycle_local
                if not _stage_cycle_selected(int(cycle_local)):
                    _log(f"[cycle {cycle_id}/{cycle_offset + cycles}] skipped by stage roster selection.")
                    continue
                _activate_semantic_vocab(
                    cycle_id=int(cycle_id),
                    cycle_local=int(cycle_local),
                    global_round=int(global_round),
                    reason="cycle_start",
                    allow_churn=True,
                    autoload_gd=True,
                )
                _rebuild_semantic_gate_loaders(
                    reason=f"cycle_start:{int(cycle_id)}",
                    cycle_seed=int(args.seed) + (int(cycle_id) * 421) + int(global_round),
                    include_payload_val=bool(gestation_gate_streak >= gate_config["gestation_maintain_rounds"]),
                )
                for round_id in range(1, rounds_per_cycle + 1):
                    if _poll_gui_stop(note=f"cycle={int(cycle_id)} round={int(round_id)} start"):
                        stop_orchestration = True
                        break
                    global_round += 1
                    gate_override_active = bool(_stage_gate_override_enabled())
                    _log(f"[cycle {cycle_id}/{cycle_offset + cycles} round {round_id}/{rounds_per_cycle}]")
                    _offload_inactive_stage_models(
                        reason=f"cycle={int(cycle_id)} round={int(round_id)} start"
                    )
                    gestation_ready_before_stage1 = bool(
                        gestation_gate_streak >= gate_config["gestation_maintain_rounds"]
                    )
                    did_gestation_stage1_round = False
                    if (not bool(gestation_ready_before_stage1)) and (gestation_stage_loader is not None):
                        gestation_stage_steps = max(1, int(len(gestation_stage_loader)))
                        ref_stage1 = _run_berkeley_refresh_epochs(
                            classifier=classifier,
                            loader=gestation_stage_loader,
                            device=device,
                            epochs=int(args.berkeley_refresh_epochs),
                            lr=float(args.berkeley_refresh_lr),
                            weight_decay=float(args.berkeley_refresh_weight_decay),
                            max_steps=int(gestation_stage_steps),
                            min_steps=int(gestation_stage_steps),
                            lr_sine_cycles=float(args.lr_sine_cycles),
                            lr_sine_frequency=float(args.lr_sine_frequency),
                            lr_sine_tail_fraction=float(args.lr_sine_tail_fraction),
                            lr_sine_min_scale=float(args.lr_sine_min_scale),
                            amp_enabled=amp_enabled,
                            amp_dtype=args.amp_dtype,
                            channels_last=bool(args.channels_last),
                            grad_accum_steps=grad_accum_steps,
                            log_every=int(args.berkeley_refresh_log_every),
                            max_seconds=float(args.berkeley_refresh_max_seconds),
                            cache_x=None,
                            cache_y=None,
                            cache_batch_size=0,
                            active_classes=int(condition_num_classes) if int(condition_num_classes) > 0 else int(score_active_classes),
                            target_label_knockout_prob=float(args.target_label_knockout_prob),
                            target_label_knockout_min_keep=int(args.target_label_knockout_min_keep),
                            target_label_knockout_max_drop_frac=float(args.target_label_knockout_max_drop_frac),
                            target_label_knockout_seed=int(args.seed) + (int(cycle_id) * 10007) + (int(round_id) * 97) + 1,
                            step_preview_callback=_make_c_step_callback(cycle_id=int(cycle_id), round_id=int(round_id)),
                            stop_requested=_poll_gui_stop,
                        )
                        refresh_rows.append({"stage": "stage1_gestation", "cycle": cycle_id, "round": round_id, **ref_stage1})
                        _log(
                            "  Stage 1 gestation refresh: "
                            f"loss={ref_stage1['loss']:.4f} samples={ref_stage1.get('samples', 0)} "
                            f"steps={int(ref_stage1.get('steps_per_epoch', 0))} "
                            f"elapsed={ref_stage1.get('elapsed_sec', 0.0):.1f}s"
                        )
                        did_gestation_stage1_round = True
                        _save_training_segment_snapshot(
                            enabled=bool(args.checkpoint_after_training_segment),
                            out_dir=out_dir,
                            objective_mode=args.objective_mode,
                            run_tag=run_tag,
                            segment=f"stage1_gestation_cycle_{int(cycle_id)}_round_{int(round_id)}",
                            best_cfg=best_cfg,
                            classifier=classifier,
                            transformer=transformer,
                            generator=generator,
                            discriminator=discriminator,
                            transformer_history=transformer_hist,
                            generator_history=generator_hist,
                            wave_classifier_history=wave_classifier_hist,
                            orchestration_history=orchestration_rows,
                            refresh_history=refresh_rows,
                            gate_history=gate_history,
                            gate_status={
                                "gestation_streak": int(gestation_gate_streak),
                                "berkeley_streak": int(berkeley_gate_streak),
                                "wave_streak": int(wave_gate_streak),
                                "generator_streak": int(generator_gate_streak),
                                "transformer_streak": int(transformer_gate_streak),
                            },
                            extra={
                                "sample_bits": int(sample_bits),
                                "chunk_samples": int(chunk_samples),
                                "patch_size": int(args.patch_size),
                                "stage1_rows": int(gestation_stage_loader_count),
                            },
                        )
                    if bool(did_gestation_stage1_round):
                        _cleanup_cuda_allocator()
                    gestation_refresh_ready = bool(gestation_gate_streak >= gate_config["gestation_maintain_rounds"])
                    if bool(gestation_refresh_ready) and (refresh_loader is None):
                        _init_refresh_resources(
                            reason=f"gestation_unlocked:cycle={int(cycle_id)} round={int(round_id)}",
                            samples_seen_for_startup=int(berkeley_refresh_samples_seen),
                        )

                    refresh_only_loss_mode = bool(berkeley_loss_target_value > 0.0 and bool(berkeley_loss_target_unmet))
                    did_refresh_round = False
                    berkeley_ready_before_refresh = bool(
                        (not bool(total_gate_runtime_enabled))
                        or (berkeley_gate_streak >= gate_config["berkeley_maintain_rounds"])
                    )
                    force_full_stage2_until_gate2 = bool(
                        refresh_loader is not None
                        and bool(gestation_refresh_ready)
                        and bool(total_gate_runtime_enabled)
                        and (not bool(berkeley_ready_before_refresh))
                    )
                    run_refresh_this_round = (
                        refresh_loader is not None
                        and (
                            bool(force_full_stage2_until_gate2)
                            or (
                                int(args.berkeley_refresh_round_every) > 0
                                and (global_round % int(args.berkeley_refresh_round_every) == 0)
                            )
                            or bool(refresh_only_loss_mode)
                        )
                    )
                    if run_refresh_this_round:
                        need_full_refresh_startup = (
                            int(berkeley_refresh_required_samples) > 0
                            and int(berkeley_refresh_samples_seen) < int(berkeley_refresh_required_samples)
                        )
                        need_full_refresh_loss = bool(refresh_only_loss_mode)
                        need_full_refresh_gate2 = bool(force_full_stage2_until_gate2)
                        refresh_cache_x = refresh_cache.get("x") if isinstance(refresh_cache, dict) else None
                        refresh_cache_y = refresh_cache.get("y") if isinstance(refresh_cache, dict) else None
                        refresh_cache_batch = int(refresh_run_batch_size)
                        if (
                            need_full_refresh_startup
                            and isinstance(refresh_cache, dict)
                            and int(refresh_cache_samples) > 0
                            and int(refresh_cache_samples) < int(berkeley_refresh_required_samples)
                        ):
                            _log(
                                "  Berkeley refresh startup requirement active: forcing loader mode "
                                f"(cache_samples={refresh_cache_samples} < required={berkeley_refresh_required_samples})"
                            )
                            refresh_cache_x = None
                            refresh_cache_y = None
                            refresh_cache_batch = 0
                        if need_full_refresh_gate2:
                            if (
                                refresh_cache_x is not None
                                or refresh_cache_y is not None
                                or int(refresh_cache_batch) > 0
                            ):
                                _log(
                                    "  Stage 2 pre-Gate2 lock: forcing full non-validation deck refresh "
                                    "(cache disabled until Gate 2 is ready)."
                                )
                            refresh_cache_x = None
                            refresh_cache_y = None
                            refresh_cache_batch = 0
                        if need_full_refresh_loss:
                            if (
                                refresh_cache_x is not None
                                or refresh_cache_y is not None
                                or int(refresh_cache_batch) > 0
                            ):
                                _log(
                                    "  Berkeley loss target unmet: forcing full-dataset loader refresh "
                                    f"(latest_target={berkeley_loss_target_value:.4f})"
                                )
                            refresh_cache_x = None
                            refresh_cache_y = None
                            refresh_cache_batch = 0
                        if need_full_refresh_startup or need_full_refresh_loss or need_full_refresh_gate2:
                            min_refresh_steps = max(1, int(len(refresh_loader)))
                        else:
                            min_refresh_steps = 0
                        refresh_max_steps = int(args.berkeley_refresh_max_steps)
                        if (need_full_refresh_startup or need_full_refresh_loss or need_full_refresh_gate2) and int(min_refresh_steps) > 0:
                            refresh_max_steps = int(min_refresh_steps)
                        ref = _run_berkeley_refresh_epochs(
                            classifier=classifier,
                            loader=refresh_loader,
                            device=device,
                            epochs=int(args.berkeley_refresh_epochs),
                            lr=float(args.berkeley_refresh_lr),
                            weight_decay=float(args.berkeley_refresh_weight_decay),
                            max_steps=int(refresh_max_steps),
                            min_steps=int(min_refresh_steps),
                            lr_sine_cycles=float(args.lr_sine_cycles),
                            lr_sine_frequency=float(args.lr_sine_frequency),
                            lr_sine_tail_fraction=float(args.lr_sine_tail_fraction),
                            lr_sine_min_scale=float(args.lr_sine_min_scale),
                            amp_enabled=amp_enabled,
                            amp_dtype=args.amp_dtype,
                            channels_last=bool(args.channels_last),
                            grad_accum_steps=grad_accum_steps,
                            log_every=int(args.berkeley_refresh_log_every),
                            max_seconds=float(args.berkeley_refresh_max_seconds),
                            cache_x=refresh_cache_x,
                            cache_y=refresh_cache_y,
                            cache_batch_size=int(refresh_cache_batch),
                            active_classes=int(score_active_classes),
                            target_label_knockout_prob=float(args.target_label_knockout_prob),
                            target_label_knockout_min_keep=int(args.target_label_knockout_min_keep),
                            target_label_knockout_max_drop_frac=float(args.target_label_knockout_max_drop_frac),
                            target_label_knockout_seed=int(args.seed) + (int(cycle_id) * 10007) + (int(round_id) * 97) + 2,
                            step_preview_callback=_make_c_step_callback(cycle_id=int(cycle_id), round_id=int(round_id)),
                            stop_requested=_poll_gui_stop,
                        )
                        refresh_rows.append({"stage": "orchestration", "cycle": cycle_id, "round": round_id, **ref})
                        berkeley_refresh_samples_seen += int(max(0, int(ref.get("samples", 0))))
                        _log(
                            "  Berkeley refresh: "
                            f"loss={ref['loss']:.4f} samples={ref.get('samples', 0)} "
                            f"source={ref.get('source', 'loader')} "
                            f"startup_seen={berkeley_refresh_samples_seen}/{berkeley_refresh_required_samples} "
                            f"elapsed={ref.get('elapsed_sec', 0.0):.1f}s"
                        )
                        did_refresh_round = True
                        ref_loss = float(ref.get("loss", float("nan")))
                        if berkeley_loss_target_value > 0.0:
                            berkeley_loss_target_unmet = bool((not math.isfinite(ref_loss)) or (ref_loss > berkeley_loss_target_value))
                        _save_training_segment_snapshot(
                            enabled=bool(args.checkpoint_after_training_segment),
                            out_dir=out_dir,
                            objective_mode=args.objective_mode,
                            run_tag=run_tag,
                            segment=f"orchestration_refresh_cycle_{int(cycle_id)}_round_{int(round_id)}",
                            best_cfg=best_cfg,
                            classifier=classifier,
                            transformer=transformer,
                            generator=generator,
                            discriminator=discriminator,
                            transformer_history=transformer_hist,
                            generator_history=generator_hist,
                            wave_classifier_history=wave_classifier_hist,
                            orchestration_history=orchestration_rows,
                            refresh_history=refresh_rows,
                            gate_history=gate_history,
                            gate_status={
                                "gestation_streak": int(gestation_gate_streak),
                                "berkeley_streak": int(berkeley_gate_streak),
                                "wave_streak": int(wave_gate_streak),
                                "generator_streak": int(generator_gate_streak),
                                "transformer_streak": int(transformer_gate_streak),
                            },
                            extra={
                                "sample_bits": int(sample_bits),
                                "chunk_samples": int(chunk_samples),
                                "patch_size": int(args.patch_size),
                            },
                        )
                        if need_full_refresh_loss and bool(berkeley_loss_target_unmet) and (not bool(gate_override_active)):
                            berkeley_gate_streak = 0
                            wave_gate_streak = 0
                            _log(
                                "  full-dataset refresh-only round: "
                                f"loss={ref_loss:.4f} target={berkeley_loss_target_value:.4f} "
                                "(downstream skipped)"
                            )
                            round_row = {
                                "cycle": cycle_id,
                                "round": round_id,
                                "stage": "C",
                                "berkeley_refresh_only": True,
                                "berkeley_refresh_loss": (None if not math.isfinite(ref_loss) else float(max(0.0, ref_loss))),
                                "berkeley_refresh_loss_target": float(berkeley_loss_target_value),
                                "downstream_skipped": "berkeley_loss_refresh_only",
                                "berkeley_gate_pass": False,
                                "berkeley_streak": int(berkeley_gate_streak),
                                "berkeley_ready": False,
                                "wave_entropy_streak": int(wave_gate_streak),
                                "wave_entropy_ready": False,
                                "berkeley_gate_detail": {
                                    "mode": "hard_loss_full_refresh",
                                    "reason": f"loss>{berkeley_loss_target_value:.4f}",
                                    "loss_pass": False,
                                    "loss_target": float(berkeley_loss_target_value),
                                    "loss": (None if not math.isfinite(ref_loss) else float(max(0.0, ref_loss))),
                                    "refresh_pass": True,
                                    "refresh_samples_seen": int(berkeley_refresh_samples_seen),
                                    "refresh_samples_required": int(berkeley_refresh_required_samples),
                                },
                            }
                            gate_history.append(
                                {
                                    "cycle": cycle_id,
                                    "round": round_id,
                                    "gestation_pass": bool(gestation_gate_streak >= gate_config["gestation_maintain_rounds"]),
                                    "gestation_streak": int(gestation_gate_streak),
                                    "gestation_ready": bool(gestation_gate_streak >= gate_config["gestation_maintain_rounds"]),
                                    "berkeley_pass": False,
                                    "berkeley_gate_mode": "hard_loss_full_refresh",
                                    "berkeley_gate_reason": f"loss>{berkeley_loss_target_value:.4f}",
                                    "berkeley_loss_pass": False,
                                    "berkeley_loss_target": float(berkeley_loss_target_value),
                                    "berkeley_refresh_pass": True,
                                    "berkeley_refresh_seen": int(berkeley_refresh_samples_seen),
                                    "berkeley_refresh_required": int(berkeley_refresh_required_samples),
                                    "berkeley_streak": int(berkeley_gate_streak),
                                    "berkeley_ready": False,
                                    "wave_pass": False,
                                    "wave_streak": int(wave_gate_streak),
                                    "wave_ready": False,
                                    "generator_pass": None,
                                    "generator_streak": int(generator_gate_streak),
                                    "generator_ready": None,
                                    "transformer_pass": None,
                                    "transformer_streak": int(transformer_gate_streak),
                                    "transformer_ready": None,
                                }
                            )
                            orchestration_rows.append(round_row)
                            stage_chain_ready = bool(gestation_gate_streak >= gate_config["gestation_maintain_rounds"])
                            _run_transformer_round_preview(
                                stage="S1",
                                cycle_id=int(cycle_id),
                                round_id=int(round_id),
                                global_round=int(global_round),
                                extra={
                                    "gestation_classifier_eval": round_row.get("gestation_classifier_eval", None),
                                    "gestation_gate_detail": round_row.get("gestation_gate_detail", None),
                                    "gestation_gate_pass": round_row.get("gestation_gate_pass", None),
                                },
                            )
                            _run_transformer_round_preview(
                                stage="G1",
                                cycle_id=int(cycle_id),
                                round_id=int(round_id),
                                global_round=int(global_round),
                                extra={
                                    "gestation_classifier_eval": round_row.get("gestation_classifier_eval", None),
                                    "gestation_gate_detail": round_row.get("gestation_gate_detail", None),
                                    "gestation_gate_pass": round_row.get("gestation_gate_pass", None),
                                },
                            )
                            if bool(stage_chain_ready):
                                _run_transformer_round_preview(
                                    stage="S2",
                                    cycle_id=int(cycle_id),
                                    round_id=int(round_id),
                                    global_round=int(global_round),
                                    extra={
                                        "initial_stage_2_ready": False,
                                        "initial_stage_2_mode": "refresh_only_loss_clamp",
                                    },
                                )
                                _run_transformer_round_preview(
                                    stage="G2",
                                    cycle_id=int(cycle_id),
                                    round_id=int(round_id),
                                    global_round=int(global_round),
                                    extra={
                                        "berkeley_classifier_eval": round_row.get("berkeley_classifier_eval", None),
                                        "berkeley_gate_detail": round_row.get("berkeley_gate_detail", None),
                                        "berkeley_gate_pass": round_row.get("berkeley_gate_pass", None),
                                    },
                                )
                                _run_transformer_round_preview(
                                    stage="C",
                                    cycle_id=int(cycle_id),
                                    round_id=int(round_id),
                                    global_round=int(global_round),
                                    extra={
                                        "downstream_skipped": "berkeley_loss_refresh_only",
                                        "gestation_classifier_eval": round_row.get("gestation_classifier_eval", None),
                                        "gestation_gate_detail": round_row.get("gestation_gate_detail", None),
                                        "berkeley_classifier_eval": round_row.get("berkeley_classifier_eval", None),
                                        "berkeley_gate_detail": round_row.get("berkeley_gate_detail", None),
                                        "stage_c_stream_index": -1,
                                        "stage_c_record_index": -1,
                                        "stage_c_metric": {},
                                    },
                                )
                            if int(args.checkpoint_every_round) > 0 and (global_round % int(args.checkpoint_every_round) == 0):
                                _save_staged_round_checkpoint(last_cycle=cycle_id, last_round=round_id)
                            if _poll_gui_stop(note=f"cycle={int(cycle_id)} round={int(round_id)} stage=C"):
                                stop_orchestration = True
                                break
                            continue
                        elif need_full_refresh_loss and bool(berkeley_loss_target_unmet) and bool(gate_override_active):
                            _log(
                                "  gate override active: continuing downstream despite refresh-only loss clamp "
                                f"(loss={ref_loss:.4f} target={berkeley_loss_target_value:.4f})"
                            )
                    if did_refresh_round:
                        _cleanup_cuda_allocator()
                    if _poll_gui_stop(note=f"cycle={int(cycle_id)} round={int(round_id)} post-refresh"):
                        stop_orchestration = True
                        break

                    berkeley_round_metric_idx = _sample_round_classifier_val_indices()
                    berkeley_round_metric = _score_config_with_classifier(
                        records=records,
                        indices=berkeley_round_metric_idx,
                        cfg=best_cfg,
                        image_hw=image_hw,
                        classifier=classifier,
                        device=device,
                        batch_size=args.classifier_batch_size,
                        score_topk=args.score_topk,
                        score_threshold=args.score_threshold,
                        score_w_topk=args.score_w_topk,
                        score_w_cov=args.score_w_cov,
                        score_w_mean=args.score_w_mean,
                        amp_enabled=amp_enabled,
                        amp_dtype=args.amp_dtype,
                        channels_last=bool(args.channels_last),
                        active_classes=int(score_active_classes),
                    )
                    two_stage_gate_eval = _evaluate_two_stage_semantic_gates(round_index=int(global_round))
                    gestation_round_eval = two_stage_gate_eval.get("gestation_eval", None)
                    gestation_gate_eval = dict(two_stage_gate_eval.get("gestation_detail", {}))
                    gestation_pass = bool(two_stage_gate_eval.get("gestation_pass", True))
                    gestation_ready = bool(two_stage_gate_eval.get("gestation_ready", True))
                    stage2_mode = str(
                        two_stage_gate_eval.get(
                            "initial_stage_2_mode",
                            ("ready" if bool(two_stage_gate_eval.get("initial_stage_2_ready", False)) else "waiting_for_gate_1"),
                        )
                    )
                    berkeley_round_eval = two_stage_gate_eval.get("total_eval", None)
                    berkeley_gate_eval = dict(two_stage_gate_eval.get("total_detail", {}))
                    berkeley_pass = bool(two_stage_gate_eval.get("total_pass", True))
                    berkeley_ready = bool(two_stage_gate_eval.get("total_ready", True))

                    wave_pass = (
                        (float(berkeley_round_metric.get("mean_entropy", 0.0)) >= gate_config["wave_min_entropy"])
                        and (float(berkeley_round_metric["score"]) >= gate_config["berkeley_min_score"])
                    )
                    if wave_pass:
                        wave_gate_streak += 1
                    else:
                        wave_gate_streak = 0
                    wave_ready = wave_gate_streak >= gate_config["wave_maintain_rounds"]

                    round_row: Dict = {
                        "cycle": cycle_id,
                        "round": round_id,
                        "berkeley_metric": berkeley_round_metric,
                        "gestation_classifier_eval": gestation_round_eval,
                        "gestation_gate_detail": dict(gestation_gate_eval),
                        "gestation_gate_pass": bool(gestation_pass),
                        "gestation_streak": int(gestation_gate_streak),
                        "gestation_ready": bool(gestation_ready),
                        "initial_stage_2_ready": bool(two_stage_gate_eval.get("initial_stage_2_ready", False)),
                        "initial_stage_2_mode": str(stage2_mode),
                        "berkeley_classifier_eval": berkeley_round_eval,
                        "berkeley_gate_detail": dict(berkeley_gate_eval),
                        "berkeley_gate_pass": bool(berkeley_pass),
                        "berkeley_streak": int(berkeley_gate_streak),
                        "berkeley_ready": bool(berkeley_ready),
                        "wave_entropy_gate_pass": bool(wave_pass),
                        "wave_entropy_streak": int(wave_gate_streak),
                        "wave_entropy_ready": bool(wave_ready),
                        "gate_override": bool(gate_override_active),
                    }
                    wave_feedback_ctx = _wave_feedback_snapshot(
                        wave_classifier_eval=wave_classifier_last_eval,
                        wave_zero_shot_eval=wave_zero_shot_last_eval,
                        zero_shot_weight=float(gate_config["wave_feedback_zero_shot_weight"]),
                    )
                    feedback_transformer_score_weight = _feedback_scaled_weight(
                        base_weight=float(args.transformer_loss_score_target_weight),
                        feedback_score=wave_feedback_ctx.get("combined_score", None),
                        boost=float(gate_config["wave_feedback_transformer_loss_boost"]),
                    )
                    feedback_generator_wave_weight = _feedback_scaled_weight(
                        base_weight=float(args.generator_loss_wave_weight),
                        feedback_score=wave_feedback_ctx.get("combined_score", None),
                        boost=float(gate_config["wave_feedback_generator_loss_boost"]),
                    )
                    round_row["wave_feedback_pre"] = wave_feedback_ctx
                    round_row["feedback_weights"] = {
                        "generator_wave_weight": float(feedback_generator_wave_weight),
                        "transformer_score_target_weight": float(feedback_transformer_score_weight),
                    }
                    _run_transformer_round_preview(
                        stage="S1",
                        cycle_id=int(cycle_id),
                        round_id=int(round_id),
                        global_round=int(global_round),
                        extra={
                            "gestation_classifier_eval": round_row.get("gestation_classifier_eval", None),
                            "gestation_gate_detail": round_row.get("gestation_gate_detail", None),
                            "gestation_gate_pass": round_row.get("gestation_gate_pass", None),
                        },
                    )
                    _run_transformer_round_preview(
                        stage="G1",
                        cycle_id=int(cycle_id),
                        round_id=int(round_id),
                        global_round=int(global_round),
                        extra={
                            "gestation_classifier_eval": round_row.get("gestation_classifier_eval", None),
                            "gestation_gate_detail": round_row.get("gestation_gate_detail", None),
                            "gestation_gate_pass": round_row.get("gestation_gate_pass", None),
                        },
                    )
                    if bool(gestation_ready):
                        _run_transformer_round_preview(
                            stage="S2",
                            cycle_id=int(cycle_id),
                            round_id=int(round_id),
                            global_round=int(global_round),
                            extra={
                                "initial_stage_2_ready": round_row.get("initial_stage_2_ready", False),
                                "initial_stage_2_mode": round_row.get("initial_stage_2_mode", ""),
                            },
                        )
                    if bool(gestation_ready) and bool(two_stage_gate_eval.get("initial_stage_2_ready", False)):
                        _run_transformer_round_preview(
                            stage="G2",
                            cycle_id=int(cycle_id),
                            round_id=int(round_id),
                            global_round=int(global_round),
                            extra={
                                "berkeley_classifier_eval": round_row.get("berkeley_classifier_eval", None),
                                "berkeley_gate_detail": round_row.get("berkeley_gate_detail", None),
                                "berkeley_gate_pass": round_row.get("berkeley_gate_pass", None),
                            },
                        )

                    if (not bool(gate_override_active)) and (not gestation_ready or not berkeley_ready or not wave_ready):
                        round_row["stage"] = "C"
                        if not bool(gestation_ready):
                            round_row["downstream_skipped"] = "gestation_gate"
                        elif not bool(berkeley_ready):
                            round_row["downstream_skipped"] = "berkeley_gate"
                        else:
                            round_row["downstream_skipped"] = "wave_entropy_gate"
                        conf_txt = (
                            f"gest_reason={gestation_gate_eval.get('reason', 'pass')}, "
                            f"gest_cov={int(gestation_gate_eval.get('refresh_samples_seen', 0))}/"
                            f"{int(gestation_gate_eval.get('refresh_samples_required', 0))}, "
                            f"stage2_ready={1 if bool(two_stage_gate_eval.get('initial_stage_2_ready', False)) else 0}, "
                            f"stage2_mode={stage2_mode}, "
                            f"total_mode={berkeley_gate_eval.get('mode', 'hard_loss_full_refresh')}, "
                            f"total_reason={berkeley_gate_eval.get('reason', 'pass')}, "
                            f"total_cov={int(berkeley_gate_eval.get('refresh_samples_seen', 0))}/"
                            f"{int(berkeley_gate_eval.get('refresh_samples_required', 0))}, "
                        )
                        if berkeley_round_eval is not None:
                            conf_txt = (
                                f"conf={berkeley_round_eval['mean_confidence']:.4f}, "
                                f"macro_f1={berkeley_round_eval['macro_f1']:.4f}, "
                                f"loss={float(berkeley_round_eval.get('loss', 0.0)):.4f}, "
                                f"{conf_txt}"
                            )
                        _log(
                            "  downstream skipped by pre-generator gates "
                            f"({conf_txt}"
                            f"wave_entropy={berkeley_round_metric.get('mean_entropy', 0.0):.4f}, "
                            f"gest_streak={gestation_gate_streak}/{gate_config['gestation_maintain_rounds']}, "
                            f"berk_streak={berkeley_gate_streak}/{gate_config['berkeley_maintain_rounds']}, "
                            f"wave_streak={wave_gate_streak}/{gate_config['wave_maintain_rounds']})"
                        )
                        gate_history.append(
                            {
                                "cycle": cycle_id,
                                "round": round_id,
                                "gestation_pass": bool(gestation_pass),
                                "gestation_gate_reason": str(gestation_gate_eval.get("reason", "pass")),
                                "gestation_loss_target": gestation_gate_eval.get("loss_target", None),
                                "gestation_streak": int(gestation_gate_streak),
                                "gestation_ready": bool(gestation_ready),
                                "berkeley_pass": bool(berkeley_pass),
                                "berkeley_gate_mode": str(berkeley_gate_eval.get("mode", "hard_loss_full_refresh")),
                                "berkeley_gate_reason": str(berkeley_gate_eval.get("reason", "pass")),
                                "berkeley_loss_pass": bool(berkeley_gate_eval.get("loss_pass", True)),
                                "berkeley_loss_target": berkeley_gate_eval.get("loss_target", None),
                                "berkeley_refresh_pass": bool(berkeley_gate_eval.get("refresh_pass", True)),
                                "berkeley_refresh_seen": int(berkeley_gate_eval.get("refresh_samples_seen", 0)),
                                "berkeley_refresh_required": int(berkeley_gate_eval.get("refresh_samples_required", 0)),
                                "berkeley_streak": int(berkeley_gate_streak),
                                "berkeley_ready": bool(berkeley_ready),
                                "wave_pass": bool(wave_pass),
                                "wave_streak": int(wave_gate_streak),
                                "wave_ready": bool(wave_ready),
                                "generator_pass": None,
                                "generator_streak": int(generator_gate_streak),
                                "generator_ready": None,
                                "transformer_pass": None,
                                "transformer_streak": int(transformer_gate_streak),
                                "transformer_ready": None,
                                "gate_override": bool(gate_override_active),
                            }
                        )
                        orchestration_rows.append(round_row)
                        if bool(gestation_ready) and bool(two_stage_gate_eval.get("initial_stage_2_ready", False)):
                            _run_transformer_round_preview(
                                stage="C",
                                cycle_id=int(cycle_id),
                                round_id=int(round_id),
                                global_round=int(global_round),
                                extra={
                                    "downstream_skipped": str(round_row.get("downstream_skipped", "")),
                                    "gestation_classifier_eval": round_row.get("gestation_classifier_eval", None),
                                    "gestation_gate_detail": round_row.get("gestation_gate_detail", None),
                                    "berkeley_classifier_eval": round_row.get("berkeley_classifier_eval", None),
                                    "berkeley_gate_detail": round_row.get("berkeley_gate_detail", None),
                                    "stage_c_stream_index": -1,
                                    "stage_c_record_index": (
                                        int(berkeley_round_metric_idx[0])
                                        if isinstance(berkeley_round_metric_idx, (list, tuple, np.ndarray))
                                        and len(berkeley_round_metric_idx) > 0
                                        else -1
                                    ),
                                    "stage_c_metric": (
                                        dict(berkeley_round_metric)
                                        if isinstance(berkeley_round_metric, dict)
                                        else {}
                                    ),
                                },
                            )
                        if int(args.checkpoint_every_round) > 0 and (global_round % int(args.checkpoint_every_round) == 0):
                            _save_staged_round_checkpoint(last_cycle=cycle_id, last_round=round_id)
                        if _poll_gui_stop(note=f"cycle={int(cycle_id)} round={int(round_id)} stage=C"):
                            stop_orchestration = True
                            break
                        continue
                    if bool(gate_override_active) and ((not gestation_ready) or (not berkeley_ready) or (not wave_ready)):
                        _log(
                            "  gate override active: bypassing pre-generator gates "
                            f"(gest_ready={int(gestation_ready)} berk_ready={int(berkeley_ready)} wave_ready={int(wave_ready)})"
                        )

                    generator_ready = generator_gate_streak >= gate_config["generator_maintain_rounds"]
                    transformer_ready_pre = transformer_gate_streak >= gate_config["transformer_maintain_rounds"]
                    generator_ready_effective = bool(generator_ready or bool(gate_override_active))
                    transformer_ready_pre_effective = bool(transformer_ready_pre or bool(gate_override_active))
                    if bool(args.joint_enabled) and bool(generator_ready_effective) and bool(transformer_ready_pre_effective):
                        round_row["stage"] = "J"
                        generator, discriminator, joint_gen_hist = train_conditional_generator_discriminator(
                            generator=generator,
                            discriminator=discriminator,
                            classifier=classifier,
                            payload_images=payload_images,
                            payload_conditions=payload_conditions,
                            num_classes=int(condition_num_classes),
                            image_hw=image_hw,
                            device=device,
                            epochs=max(1, int(args.joint_generator_epochs_per_round)),
                            steps_per_epoch=max(1, int(args.joint_generator_steps_per_round)),
                            batch_size=max(1, int(args.generator_batch_size)),
                            disc_steps_per_gen_step=max(1, int(args.discriminator_steps_per_generator_step)),
                            z_dim=max(8, int(args.generator_z_dim)),
                            lr_g=float(args.generator_lr),
                            lr_d=float(args.discriminator_lr),
                            w_adv=float(args.generator_loss_adv_weight),
                            w_cls=float(args.generator_loss_cls_weight),
                            w_wave=float(feedback_generator_wave_weight),
                            wave_margin=float(args.generator_wave_score_margin),
                            wave_denoise_weight=float(args.generator_wave_denoise_weight),
                            wave_carrier_blend=float(args.generator_wave_carrier_blend),
                            wave_strength=float(args.generator_wave_strength),
                            wave_stride_skew_max=float(args.generator_wave_stride_skew_max),
                            wave_degrade_mode=str(args.generator_wave_degrade_mode),
                            transformer_for_wave=transformer,
                            wave_cfg=best_cfg,
                            wave_sample_bits=int(sample_bits),
                            wave_chunk_samples=int(chunk_samples),
                            wave_reference_streams=train_streams,
                            log_every_steps=max(0, int(args.generator_log_every)),
                            step_preview_callback=_make_g_step_callback("J", cycle_id=int(cycle_id), round_id=int(round_id)),
                            stop_requested=_poll_gui_stop,
                            amp=amp_enabled,
                            amp_dtype=args.amp_dtype,
                            channels_last=bool(args.channels_last),
                            grad_accum_steps=max(1, int(args.generator_grad_accum_steps)),
                            classifier_forward_batch_cap=max(0, int(args.generator_classifier_batch_cap)),
                            seed=args.seed + 15400 + global_round,
                        )
                        for row in joint_gen_hist:
                            rr = dict(row)
                            rr["cycle"] = float(cycle_id)
                            rr["round"] = float(round_id)
                            rr["stage"] = "J"
                            generator_hist.append(rr)
                        joint_generator_eval = evaluate_conditional_generator(
                            generator=generator,
                            classifier=classifier,
                            payload_conditions=payload_conditions,
                            num_classes=int(condition_num_classes),
                            image_hw=image_hw,
                            z_dim=max(8, int(args.generator_z_dim)),
                            device=device,
                            steps=max(8, int(args.gate_berkeley_eval_max_steps)),
                            batch_size=max(1, int(args.generator_batch_size)),
                            amp=amp_enabled,
                            amp_dtype=args.amp_dtype,
                            channels_last=bool(args.channels_last),
                            discriminator=discriminator,
                            seed=args.seed + 15500 + global_round,
                        )
                        generator_pass = (
                            float(joint_generator_eval["target_prob"]) >= gate_config["generator_min_target_prob"]
                        )
                        if generator_pass:
                            generator_gate_streak += 1
                        else:
                            generator_gate_streak = 0
                        generator_ready = generator_gate_streak >= gate_config["generator_maintain_rounds"]
                        generator_ready_effective = bool(generator_ready or bool(gate_override_active))
                        round_row["generator_eval"] = joint_generator_eval
                        round_row["generator_gate_pass"] = bool(generator_pass)
                        round_row["generator_streak"] = int(generator_gate_streak)
                        round_row["generator_ready"] = bool(generator_ready)
                        round_row["generator_ready_effective"] = bool(generator_ready_effective)
                        joint_fake_feedback = _run_staged_fake_feedback(
                            stage_tag="J",
                            cycle_id=int(cycle_id),
                            round_id=int(round_id),
                            global_round=int(global_round),
                        )
                        if isinstance(joint_fake_feedback, dict):
                            round_row["fake_feedback"] = dict(joint_fake_feedback)
                        if bool(args.training_preview_enabled) and (int(global_round) % int(training_preview_every) == 0):
                            gen_preview = _save_generator_supervision_preview(
                                preview_dir=training_preview_root,
                                cycle_id=int(cycle_id),
                                round_id=int(round_id),
                                generator=generator,
                                classifier=classifier,
                                payload_conditions=payload_conditions,
                                num_classes=int(condition_num_classes),
                                class_names=class_names,
                                image_hw=image_hw,
                                z_dim=max(8, int(args.generator_z_dim)),
                                samples=max(1, int(args.training_preview_generator_samples)),
                                topk=int(training_preview_topk),
                                device=device,
                                amp_enabled=amp_enabled,
                                amp_dtype=args.amp_dtype,
                                channels_last=bool(args.channels_last),
                                seed=int(args.seed + 17100 + global_round),
                            )
                            _append_training_preview_manifest(gen_preview)

                        _offload_inactive_stage_models(
                            reason=f"cycle={int(cycle_id)} round={int(round_id)} joint-transformer"
                        )
                        transformer, joint_round_hist, joint_train_batch = _train_transformer_stage_with_retry(
                            stage_tag="J",
                            epochs=max(1, int(args.joint_transformer_epochs_per_round)),
                            steps_per_epoch=max(1, int(args.joint_transformer_steps_per_round)),
                            score_target_weight=float(feedback_transformer_score_weight),
                            score_target_margin=float(args.transformer_loss_score_target_margin),
                            seed=int(args.seed + 9200 + global_round),
                            step_preview_callback=_make_r_step_callback("J", cycle_id=int(cycle_id), round_id=int(round_id)),
                        )
                        for row in joint_round_hist:
                            rr = dict(row)
                            rr["cycle"] = float(cycle_id)
                            rr["round"] = float(round_id)
                            rr["stage"] = "J"
                            transformer_hist.append(rr)
                        joint_feature_eval, joint_eval_batch = _evaluate_transformer_stage_with_retry(
                            stage_tag="J",
                            batch_size_hint=int(joint_train_batch),
                            max_batches=20,
                        )
                        wave_feedback_gate_ok, wave_feedback_gate_reason = _wave_feedback_gate_pass(
                            feedback=wave_feedback_ctx,
                            min_acc=float(gate_config["wave_feedback_min_acc"]),
                            min_zero_shot_top1_rate=float(gate_config["wave_feedback_min_zs_top1_rate"]),
                        )
                        transformer_pass = (
                            (float(joint_feature_eval["score_after"]) >= gate_config["transformer_min_score_after"])
                            and (float(joint_feature_eval["score_gain"]) >= gate_config["transformer_min_gain"])
                            and bool(wave_feedback_gate_ok)
                        )
                        if transformer_pass:
                            transformer_gate_streak += 1
                        else:
                            transformer_gate_streak = 0
                        transformer_ready = transformer_gate_streak >= gate_config["transformer_maintain_rounds"]
                        transformer_ready_effective = bool(transformer_ready or bool(gate_override_active))
                        round_row["transformer_eval"] = joint_feature_eval
                        round_row["transformer_train_batch_size"] = int(joint_train_batch)
                        round_row["transformer_eval_batch_size"] = int(joint_eval_batch)
                        round_row["transformer_gate_pass"] = bool(transformer_pass)
                        round_row["transformer_streak"] = int(transformer_gate_streak)
                        round_row["transformer_ready"] = bool(transformer_ready)
                        round_row["transformer_ready_effective"] = bool(transformer_ready_effective)
                        round_row["wave_feedback_gate_pass"] = bool(wave_feedback_gate_ok)
                        round_row["wave_feedback_gate_reason"] = str(wave_feedback_gate_reason)

                        _log(
                            "  joint stage: "
                            f"g_target={joint_generator_eval['target_prob']:.4f} "
                            f"r_after={joint_feature_eval['score_after']:.4f} "
                            f"r_gain={joint_feature_eval['score_gain']:.4f} "
                            f"(g_streak={generator_gate_streak}/{gate_config['generator_maintain_rounds']}, "
                            f"r_streak={transformer_gate_streak}/{gate_config['transformer_maintain_rounds']})"
                        )
                        gate_history.append(
                            {
                                "cycle": cycle_id,
                                "round": round_id,
                                "gestation_pass": bool(gestation_pass),
                                "gestation_streak": int(gestation_gate_streak),
                                "gestation_ready": bool(gestation_ready),
                                "berkeley_pass": bool(berkeley_pass),
                                "berkeley_streak": int(berkeley_gate_streak),
                                "berkeley_ready": bool(berkeley_ready),
                                "wave_pass": bool(wave_pass),
                                "wave_streak": int(wave_gate_streak),
                                "wave_ready": bool(wave_ready),
                                "generator_pass": bool(generator_pass),
                                "generator_streak": int(generator_gate_streak),
                                "generator_ready": bool(generator_ready),
                                "transformer_pass": bool(transformer_pass),
                                "transformer_streak": int(transformer_gate_streak),
                                "transformer_ready": bool(transformer_ready),
                                "gate_override": bool(gate_override_active),
                                "wave_feedback_gate_pass": bool(wave_feedback_gate_ok),
                                "wave_feedback_gate_reason": str(wave_feedback_gate_reason),
                            }
                        )
                        _save_training_segment_snapshot(
                            enabled=bool(args.checkpoint_after_training_segment),
                            out_dir=out_dir,
                            objective_mode=args.objective_mode,
                            run_tag=run_tag,
                            segment=f"round_joint_cycle_{int(cycle_id)}_round_{int(round_id)}",
                            best_cfg=best_cfg,
                            classifier=classifier,
                            transformer=transformer,
                            generator=generator,
                            discriminator=discriminator,
                            transformer_history=transformer_hist,
                            generator_history=generator_hist,
                            wave_classifier_history=wave_classifier_hist,
                            orchestration_history=orchestration_rows,
                            refresh_history=refresh_rows,
                            gate_history=gate_history,
                            gate_status={
                                "gestation_streak": int(gestation_gate_streak),
                                "berkeley_streak": int(berkeley_gate_streak),
                                "wave_streak": int(wave_gate_streak),
                                "generator_streak": int(generator_gate_streak),
                                "transformer_streak": int(transformer_gate_streak),
                            },
                            extra={
                                "sample_bits": int(sample_bits),
                                "chunk_samples": int(chunk_samples),
                                "patch_size": int(args.patch_size),
                                "accepted_library_count": int(library_serial),
                                "joint_generator_eval": joint_generator_eval,
                                "joint_transformer_eval": joint_feature_eval,
                            },
                        )
                        if bool(transformer_ready_effective) and run_wave_stage:
                            round_row = _run_wave_stage_after_transformer(
                                round_row=round_row,
                                cycle_id=int(cycle_id),
                                round_id=int(round_id),
                                global_round=int(global_round),
                            )
                            if len(gate_history) > 0 and isinstance(gate_history[-1], dict):
                                gate_history[-1]["wave_feedback_post"] = round_row.get("wave_feedback_post", {})
                                gate_history[-1]["wave_feedback_gate_post_pass"] = bool(round_row.get("wave_feedback_gate_post_pass", True))
                                gate_history[-1]["wave_feedback_gate_post_reason"] = str(round_row.get("wave_feedback_gate_post_reason", ""))
                            _run_transformer_round_preview(
                                stage="W",
                                cycle_id=int(cycle_id),
                                round_id=int(round_id),
                                global_round=int(global_round),
                                extra={"wave_classifier_val": round_row.get("wave_classifier_val", None)},
                            )
                        else:
                            if (
                                generator is not None
                                and device.type == "cuda"
                                and _module_device_type(generator) != "cuda"
                            ):
                                generator = generator.to(device)
                            _run_transformer_round_preview(
                                stage="J",
                                cycle_id=int(cycle_id),
                                round_id=int(round_id),
                                global_round=int(global_round),
                                extra={"joint_generator_eval": joint_generator_eval, "joint_transformer_eval": joint_feature_eval},
                            )
                        orchestration_rows.append(round_row)
                        if int(args.checkpoint_every_round) > 0 and (global_round % int(args.checkpoint_every_round) == 0):
                            _save_staged_round_checkpoint(last_cycle=cycle_id, last_round=round_id)
                        if _poll_gui_stop(note=f"cycle={int(cycle_id)} round={int(round_id)} stage=J"):
                            stop_orchestration = True
                            break
                        continue

                    if not bool(generator_ready_effective):
                        generator, discriminator, round_gen_hist = train_conditional_generator_discriminator(
                            generator=generator,
                            discriminator=discriminator,
                            classifier=classifier,
                            payload_images=payload_images,
                            payload_conditions=payload_conditions,
                            num_classes=int(condition_num_classes),
                            image_hw=image_hw,
                            device=device,
                            epochs=max(1, int(args.generator_epochs_per_round)),
                            steps_per_epoch=max(1, int(args.generator_steps_per_round)),
                            batch_size=max(1, int(args.generator_batch_size)),
                            disc_steps_per_gen_step=max(1, int(args.discriminator_steps_per_generator_step)),
                            z_dim=max(8, int(args.generator_z_dim)),
                            lr_g=float(args.generator_lr),
                            lr_d=float(args.discriminator_lr),
                            w_adv=float(args.generator_loss_adv_weight),
                            w_cls=float(args.generator_loss_cls_weight),
                            w_wave=float(feedback_generator_wave_weight),
                            wave_margin=float(args.generator_wave_score_margin),
                            wave_denoise_weight=float(args.generator_wave_denoise_weight),
                            wave_carrier_blend=float(args.generator_wave_carrier_blend),
                            wave_strength=float(args.generator_wave_strength),
                            wave_stride_skew_max=float(args.generator_wave_stride_skew_max),
                            wave_degrade_mode=str(args.generator_wave_degrade_mode),
                            transformer_for_wave=transformer,
                            wave_cfg=best_cfg,
                            wave_sample_bits=int(sample_bits),
                            wave_chunk_samples=int(chunk_samples),
                            wave_reference_streams=train_streams,
                            log_every_steps=max(0, int(args.generator_log_every)),
                            step_preview_callback=_make_g_step_callback("G", cycle_id=int(cycle_id), round_id=int(round_id)),
                            stop_requested=_poll_gui_stop,
                            amp=amp_enabled,
                            amp_dtype=args.amp_dtype,
                            channels_last=bool(args.channels_last),
                            grad_accum_steps=max(1, int(args.generator_grad_accum_steps)),
                            classifier_forward_batch_cap=max(0, int(args.generator_classifier_batch_cap)),
                            seed=args.seed + 15000 + global_round,
                        )
                        for row in round_gen_hist:
                            rr = dict(row)
                            rr["cycle"] = float(cycle_id)
                            rr["round"] = float(round_id)
                            generator_hist.append(rr)
                        generator_eval = evaluate_conditional_generator(
                            generator=generator,
                            classifier=classifier,
                            payload_conditions=payload_conditions,
                            num_classes=int(condition_num_classes),
                            image_hw=image_hw,
                            z_dim=max(8, int(args.generator_z_dim)),
                            device=device,
                            steps=max(8, int(args.gate_berkeley_eval_max_steps)),
                            batch_size=max(1, int(args.generator_batch_size)),
                            amp=amp_enabled,
                            amp_dtype=args.amp_dtype,
                            channels_last=bool(args.channels_last),
                            discriminator=discriminator,
                            seed=args.seed + 15100 + global_round,
                        )
                        generator_pass = float(generator_eval["target_prob"]) >= gate_config["generator_min_target_prob"]
                        if generator_pass:
                            generator_gate_streak += 1
                        else:
                            generator_gate_streak = 0
                        generator_ready = generator_gate_streak >= gate_config["generator_maintain_rounds"]
                        round_row["stage"] = "G"
                        round_row["generator_eval"] = generator_eval
                        round_row["generator_gate_pass"] = bool(generator_pass)
                        round_row["generator_streak"] = int(generator_gate_streak)
                        round_row["generator_ready"] = bool(generator_ready)
                        round_fake_feedback = _run_staged_fake_feedback(
                            stage_tag="G",
                            cycle_id=int(cycle_id),
                            round_id=int(round_id),
                            global_round=int(global_round),
                        )
                        if isinstance(round_fake_feedback, dict):
                            round_row["fake_feedback"] = dict(round_fake_feedback)
                        round_row["downstream_skipped"] = "generator_gate" if (not generator_ready and (not bool(gate_override_active))) else None
                        _log(
                            "  generator gate: "
                            f"target_prob={generator_eval['target_prob']:.4f} "
                            f"min={gate_config['generator_min_target_prob']:.4f} "
                            f"streak={generator_gate_streak}/{gate_config['generator_maintain_rounds']}"
                        )
                        gate_history.append(
                            {
                                "cycle": cycle_id,
                                "round": round_id,
                                "berkeley_pass": bool(berkeley_pass),
                                "berkeley_streak": int(berkeley_gate_streak),
                                "berkeley_ready": bool(berkeley_ready),
                                "wave_pass": bool(wave_pass),
                                "wave_streak": int(wave_gate_streak),
                                "wave_ready": bool(wave_ready),
                                "generator_pass": bool(generator_pass),
                                "generator_streak": int(generator_gate_streak),
                                "generator_ready": bool(generator_ready),
                                "gate_override": bool(gate_override_active),
                                "transformer_pass": None,
                                "transformer_streak": int(transformer_gate_streak),
                                "transformer_ready": None,
                            }
                        )
                        _save_training_segment_snapshot(
                            enabled=bool(args.checkpoint_after_training_segment),
                            out_dir=out_dir,
                            objective_mode=args.objective_mode,
                            run_tag=run_tag,
                            segment=f"round_generator_cycle_{int(cycle_id)}_round_{int(round_id)}",
                            best_cfg=best_cfg,
                            classifier=classifier,
                            transformer=transformer,
                            generator=generator,
                            discriminator=discriminator,
                            transformer_history=transformer_hist,
                            generator_history=generator_hist,
                            wave_classifier_history=wave_classifier_hist,
                            orchestration_history=orchestration_rows,
                            refresh_history=refresh_rows,
                            gate_history=gate_history,
                            gate_status={
                                "gestation_streak": int(gestation_gate_streak),
                                "berkeley_streak": int(berkeley_gate_streak),
                                "wave_streak": int(wave_gate_streak),
                                "generator_streak": int(generator_gate_streak),
                                "transformer_streak": int(transformer_gate_streak),
                            },
                            extra={
                                "sample_bits": int(sample_bits),
                                "chunk_samples": int(chunk_samples),
                                "patch_size": int(args.patch_size),
                            },
                        )
                        if bool(args.training_preview_enabled) and (int(global_round) % int(training_preview_every) == 0):
                            gen_preview = _save_generator_supervision_preview(
                                preview_dir=training_preview_root,
                                cycle_id=int(cycle_id),
                                round_id=int(round_id),
                                generator=generator,
                                classifier=classifier,
                                payload_conditions=payload_conditions,
                                num_classes=int(condition_num_classes),
                                class_names=class_names,
                                image_hw=image_hw,
                                z_dim=max(8, int(args.generator_z_dim)),
                                samples=max(1, int(args.training_preview_generator_samples)),
                                topk=int(training_preview_topk),
                                device=device,
                                amp_enabled=amp_enabled,
                                amp_dtype=args.amp_dtype,
                                channels_last=bool(args.channels_last),
                                seed=int(args.seed + 17000 + global_round),
                            )
                            _append_training_preview_manifest(gen_preview)
                            if isinstance(gen_preview, dict):
                                rows = gen_preview.get("samples", [])
                                if isinstance(rows, list) and len(rows) > 0:
                                    top = rows[0].get("top_labels", [])
                                    if isinstance(top, list) and len(top) > 0:
                                        _log(
                                            "Generator sample preview "
                                            f"cycle={int(cycle_id)} round={int(round_id)} "
                                            f"top={top[0].get('class_name', '?')}:{top[0].get('score', 0.0):.3f}"
                                        )
                        _run_transformer_round_preview(
                            stage="G",
                            cycle_id=int(cycle_id),
                            round_id=int(round_id),
                            global_round=int(global_round),
                            extra={"generator_eval": generator_eval},
                        )
                        orchestration_rows.append(round_row)
                        if int(args.checkpoint_every_round) > 0 and (global_round % int(args.checkpoint_every_round) == 0):
                            _save_staged_round_checkpoint(last_cycle=cycle_id, last_round=round_id)
                        if _poll_gui_stop(note=f"cycle={int(cycle_id)} round={int(round_id)} stage=G"):
                            stop_orchestration = True
                            break
                        continue

                    round_row["stage"] = "R"
                    round_row["generator_gate_pass"] = True
                    round_row["generator_streak"] = int(generator_gate_streak)
                    round_row["generator_ready"] = True
                    _offload_inactive_stage_models(
                        reason=f"cycle={int(cycle_id)} round={int(round_id)} transformer"
                    )
                    transformer, round_hist, round_train_batch = _train_transformer_stage_with_retry(
                        stage_tag="R",
                        epochs=max(1, int(args.transformer_epochs_per_round)),
                        steps_per_epoch=max(1, int(args.transformer_steps_per_round)),
                        score_target_weight=float(feedback_transformer_score_weight),
                        score_target_margin=float(args.transformer_loss_score_target_margin),
                        seed=int(args.seed + 9000 + global_round),
                        step_preview_callback=_make_r_step_callback("R", cycle_id=int(cycle_id), round_id=int(round_id)),
                    )
                    for row in round_hist:
                        rr = dict(row)
                        rr["cycle"] = float(cycle_id)
                        rr["round"] = float(round_id)
                        transformer_hist.append(rr)
                    _save_training_segment_snapshot(
                        enabled=bool(args.checkpoint_after_training_segment),
                        out_dir=out_dir,
                        objective_mode=args.objective_mode,
                        run_tag=run_tag,
                        segment=f"round_transformer_cycle_{int(cycle_id)}_round_{int(round_id)}",
                        best_cfg=best_cfg,
                        classifier=classifier,
                        transformer=transformer,
                        generator=generator,
                        discriminator=discriminator,
                        transformer_history=transformer_hist,
                        generator_history=generator_hist,
                        wave_classifier_history=wave_classifier_hist,
                        orchestration_history=orchestration_rows,
                        refresh_history=refresh_rows,
                        gate_history=gate_history,
                        gate_status={
                            "gestation_streak": int(gestation_gate_streak),
                            "berkeley_streak": int(berkeley_gate_streak),
                            "wave_streak": int(wave_gate_streak),
                            "generator_streak": int(generator_gate_streak),
                            "transformer_streak": int(transformer_gate_streak),
                        },
                        extra={
                            "sample_bits": int(sample_bits),
                            "chunk_samples": int(chunk_samples),
                            "patch_size": int(args.patch_size),
                            "accepted_library_count": int(library_serial),
                        },
                    )

                    feature_eval, round_eval_batch = _evaluate_transformer_stage_with_retry(
                        stage_tag="R",
                        batch_size_hint=int(round_train_batch),
                        max_batches=20,
                    )
                    _log(
                        f"  transformer score: before={feature_eval['score_before']:.4f} "
                        f"after={feature_eval['score_after']:.4f} gain={feature_eval['score_gain']:.4f}"
                    )

                    wave_feedback_gate_ok, wave_feedback_gate_reason = _wave_feedback_gate_pass(
                        feedback=wave_feedback_ctx,
                        min_acc=float(gate_config["wave_feedback_min_acc"]),
                        min_zero_shot_top1_rate=float(gate_config["wave_feedback_min_zs_top1_rate"]),
                    )
                    transformer_pass = (
                        (float(feature_eval["score_after"]) >= gate_config["transformer_min_score_after"])
                        and (float(feature_eval["score_gain"]) >= gate_config["transformer_min_gain"])
                        and bool(wave_feedback_gate_ok)
                    )
                    if transformer_pass:
                        transformer_gate_streak += 1
                    else:
                        transformer_gate_streak = 0
                    transformer_ready = (
                        transformer_gate_streak >= gate_config["transformer_maintain_rounds"]
                    )
                    transformer_ready_effective = bool(transformer_ready or bool(gate_override_active))
                    round_row["transformer_eval"] = feature_eval
                    round_row["transformer_train_batch_size"] = int(round_train_batch)
                    round_row["transformer_eval_batch_size"] = int(round_eval_batch)
                    round_row["transformer_gate_pass"] = bool(transformer_pass)
                    round_row["transformer_streak"] = int(transformer_gate_streak)
                    round_row["transformer_ready"] = bool(transformer_ready)
                    round_row["transformer_ready_effective"] = bool(transformer_ready_effective)
                    round_row["wave_feedback_gate_pass"] = bool(wave_feedback_gate_ok)
                    round_row["wave_feedback_gate_reason"] = str(wave_feedback_gate_reason)

                    gate_history.append(
                        {
                            "cycle": cycle_id,
                            "round": round_id,
                            "gestation_pass": bool(gestation_pass),
                            "gestation_streak": int(gestation_gate_streak),
                            "gestation_ready": bool(gestation_ready),
                            "berkeley_pass": bool(berkeley_pass),
                            "berkeley_streak": int(berkeley_gate_streak),
                            "berkeley_ready": bool(berkeley_ready),
                            "wave_pass": bool(wave_pass),
                            "wave_streak": int(wave_gate_streak),
                            "wave_ready": bool(wave_ready),
                            "generator_pass": True,
                            "generator_streak": int(generator_gate_streak),
                            "generator_ready": True,
                            "transformer_pass": bool(transformer_pass),
                            "transformer_streak": int(transformer_gate_streak),
                            "transformer_ready": bool(transformer_ready),
                            "gate_override": bool(gate_override_active),
                            "wave_feedback_gate_pass": bool(wave_feedback_gate_ok),
                            "wave_feedback_gate_reason": str(wave_feedback_gate_reason),
                        }
                    )
                    if bool(gate_override_active) and (not bool(transformer_ready)):
                        _log(
                            "  gate override active: bypassing transformer gate "
                            f"(streak={transformer_gate_streak}/{gate_config['transformer_maintain_rounds']})"
                        )
                    if not bool(transformer_ready_effective):
                        round_row["downstream_skipped"] = "transformer_gate"
                        _log(
                            "  transformer gate not met "
                            f"(after={feature_eval['score_after']:.4f} min_after={gate_config['transformer_min_score_after']:.4f}, "
                            f"gain={feature_eval['score_gain']:.4f} min_gain={gate_config['transformer_min_gain']:.4f}, "
                            f"streak={transformer_gate_streak}/{gate_config['transformer_maintain_rounds']}, "
                            f"wave_feedback={wave_feedback_gate_reason})"
                        )
                        _run_transformer_round_preview(
                            stage="R",
                            cycle_id=int(cycle_id),
                            round_id=int(round_id),
                            global_round=int(global_round),
                            extra={"transformer_eval": feature_eval},
                        )
                    elif run_wave_stage:
                        round_row = _run_wave_stage_after_transformer(
                            round_row=round_row,
                            cycle_id=int(cycle_id),
                            round_id=int(round_id),
                            global_round=int(global_round),
                        )
                        if len(gate_history) > 0 and isinstance(gate_history[-1], dict):
                            gate_history[-1]["wave_feedback_post"] = round_row.get("wave_feedback_post", {})
                            gate_history[-1]["wave_feedback_gate_post_pass"] = bool(round_row.get("wave_feedback_gate_post_pass", True))
                            gate_history[-1]["wave_feedback_gate_post_reason"] = str(round_row.get("wave_feedback_gate_post_reason", ""))
                        _run_transformer_round_preview(
                            stage="W",
                            cycle_id=int(cycle_id),
                            round_id=int(round_id),
                            global_round=int(global_round),
                            extra={"wave_classifier_val": round_row.get("wave_classifier_val", None)},
                        )
                    else:
                        _run_transformer_round_preview(
                            stage="R",
                            cycle_id=int(cycle_id),
                            round_id=int(round_id),
                            global_round=int(global_round),
                            extra={"transformer_eval": feature_eval},
                        )
                    orchestration_rows.append(round_row)
                    if int(args.checkpoint_every_round) > 0 and (global_round % int(args.checkpoint_every_round) == 0):
                        _save_staged_round_checkpoint(last_cycle=cycle_id, last_round=round_id)
                    if _poll_gui_stop(note=f"cycle={int(cycle_id)} round={int(round_id)} stage=R"):
                        stop_orchestration = True
                        break
                if stop_orchestration:
                    break
        else:
            wave_classifier = TinyConvClassifier(
                num_classes=split_num_classes,
                base_ch=int(args.classifier_base_ch),
                max_ch=int(args.classifier_max_ch),
                context_blocks=int(args.classifier_context_blocks),
                context_dropout=float(args.classifier_context_dropout),
            )
            wave_classifier_init_info = _apply_classifier_init(
                model=wave_classifier,
                ckpt_path=args.classifier_init_ckpt,
                scope=args.classifier_init_scope,
            )
            if resume_enabled and resume_wave_classifier_path.exists():
                wave_classifier_resume_info = _apply_model_init(wave_classifier, str(resume_wave_classifier_path))
                _log(
                    f"Resumed wave classifier from {resume_wave_classifier_path} "
                    f"(loaded={wave_classifier_resume_info.get('loaded_keys', 0)}, "
                    f"skipped={wave_classifier_resume_info.get('skipped_keys', 0)})"
                )
            elif isinstance(resume_pipeline_ckpt, dict) and isinstance(resume_pipeline_ckpt.get("wave_classifier_state"), dict):
                wave_classifier_resume_info = _apply_state_dict(
                    wave_classifier,
                    resume_pipeline_ckpt.get("wave_classifier_state"),
                    source_name="pipeline_checkpoint",
                )
                _log(
                    "Resumed wave classifier from pipeline checkpoint "
                    f"(loaded={wave_classifier_resume_info.get('loaded_keys', 0)}, "
                    f"skipped={wave_classifier_resume_info.get('skipped_keys', 0)})"
                )
            wave_label_embed_apply_info = _apply_label_embedding_bank_to_classifier(
                classifier=wave_classifier,
                bank_np=(None if bool(split_labels_from_pseudo_spectral) else split_label_embedding_bank),
                args=args,
            )
            if bool(split_labels_from_pseudo_spectral):
                _log("[label-embeddings] wave classifier split bank disabled for spectral pseudo labels.")
            if bool(wave_label_embed_apply_info.get("applied", False)):
                _log(
                    "[label-embeddings] wave classifier active: "
                    f"classes={wave_label_embed_apply_info.get('classes', 0)} "
                    f"dim={wave_label_embed_apply_info.get('dim', 0)} "
                    f"temp={float(wave_label_embed_apply_info.get('temperature', 0.0)):.2f}"
                )
            if args.classifier_freeze_features:
                _set_feature_freeze(wave_classifier, freeze=True)
            if args.channels_last:
                wave_classifier = wave_classifier.to(memory_format=torch.channels_last)
            wave_classifier = maybe_compile_module(
                wave_classifier,
                enabled=bool(args.compile_models),
                mode=str(args.compile_mode),
            )

            cycles = max(1, int(args.orchestration_cycles))
            rounds_per_cycle = 1 if mode == "sequential" else max(1, int(args.orchestration_rounds))
            _sync_stage_cycle_roster(total_cycles=cycles)
            gate_override_boot = bool(_stage_gate_override_enabled())
            _log(
                f"Orchestration: mode={mode}, cycles={cycles}, rounds_per_cycle={rounds_per_cycle}, "
                f"transformer_steps_per_round={args.transformer_steps_per_round}, "
                f"wave_cls_samples={args.wave_cls_train_samples}/{args.wave_cls_val_samples}, "
                f"accepted_to_transformer_preload={int(args.transformer_accepted_preload_max)}, "
                f"accepted_to_transformer_append={int(args.transformer_accepted_append_max_per_round)}, "
                f"accepted_only={bool(args.wave_cls_accepted_only)}, "
                f"resume_cycle_offset={cycle_offset}, library_serial_start={library_serial}, "
                f"gate_override={1 if gate_override_boot else 0}"
            )

            accepted_stream_seen: set = set()
            preload_cap = max(0, int(args.transformer_accepted_preload_max))
            if preload_cap > 0 and library_index_path.exists():
                preload_rows = []
                try:
                    with library_index_path.open("r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                preload_rows.append(json.loads(line))
                            except Exception:
                                continue
                except Exception:
                    preload_rows = []
                if len(preload_rows) > preload_cap:
                    preload_rows = preload_rows[-preload_cap:]
                preload_added = _append_accepted_rows_to_transformer_streams(
                    rows=preload_rows,
                    seen_paths=accepted_stream_seen,
                    train_streams=train_streams,
                    train_stream_labels=train_stream_labels,
                    train_meta=train_meta,
                    max_points=int(args.stream_max_points),
                    library_root=library_dir,
                    on_append_row=lambda r: _register_regurgitated_churn_row(r, source="accepted_preload"),
                )
                if preload_added > 0:
                    if train_stream_target_labels is not None:
                        _append_default_mix_targets(train_stream_target_labels, int(preload_added))
                    _log(
                        "Preloaded accepted-library streams for transformer: "
                        f"added={preload_added} total_train_streams={len(train_streams)}"
                    )

            global_round = len(orchestration_rows)
            stop_orchestration = False
            for cycle_local in range(1, cycles + 1):
                if stop_orchestration:
                    break
                cycle_id = cycle_offset + cycle_local
                if not _stage_cycle_selected(int(cycle_local)):
                    _log(f"[cycle {cycle_id}/{cycle_offset + cycles}] skipped by stage roster selection.")
                    continue
                _activate_semantic_vocab(
                    cycle_id=int(cycle_id),
                    cycle_local=int(cycle_local),
                    global_round=int(global_round),
                    reason="cycle_start",
                    allow_churn=True,
                    autoload_gd=True,
                )
                _rebuild_semantic_gate_loaders(
                    reason=f"cycle_start:{int(cycle_id)}",
                    cycle_seed=int(args.seed) + (int(cycle_id) * 421) + int(global_round),
                    include_payload_val=bool(gestation_gate_streak >= gate_config["gestation_maintain_rounds"]),
                )
                for round_id in range(1, rounds_per_cycle + 1):
                    if _poll_gui_stop(note=f"cycle={int(cycle_id)} round={int(round_id)} start"):
                        stop_orchestration = True
                        break
                    global_round += 1
                    gate_override_active = bool(_stage_gate_override_enabled())
                    _log(f"[cycle {cycle_id}/{cycle_offset + cycles} round {round_id}/{rounds_per_cycle}]")
                    _offload_inactive_stage_models(
                        reason=f"cycle={int(cycle_id)} round={int(round_id)} start"
                    )
                    gestation_ready_before_stage1 = bool(
                        gestation_gate_streak >= gate_config["gestation_maintain_rounds"]
                    )
                    did_gestation_stage1_round = False
                    if (not bool(gestation_ready_before_stage1)) and (gestation_stage_loader is not None):
                        gestation_stage_steps = max(1, int(len(gestation_stage_loader)))
                        ref_stage1 = _run_berkeley_refresh_epochs(
                            classifier=classifier,
                            loader=gestation_stage_loader,
                            device=device,
                            epochs=int(args.berkeley_refresh_epochs),
                            lr=float(args.berkeley_refresh_lr),
                            weight_decay=float(args.berkeley_refresh_weight_decay),
                            max_steps=int(gestation_stage_steps),
                            min_steps=int(gestation_stage_steps),
                            lr_sine_cycles=float(args.lr_sine_cycles),
                            lr_sine_frequency=float(args.lr_sine_frequency),
                            lr_sine_tail_fraction=float(args.lr_sine_tail_fraction),
                            lr_sine_min_scale=float(args.lr_sine_min_scale),
                            amp_enabled=amp_enabled,
                            amp_dtype=args.amp_dtype,
                            channels_last=bool(args.channels_last),
                            grad_accum_steps=grad_accum_steps,
                            log_every=int(args.berkeley_refresh_log_every),
                            max_seconds=float(args.berkeley_refresh_max_seconds),
                            cache_x=None,
                            cache_y=None,
                            cache_batch_size=0,
                            active_classes=int(condition_num_classes) if int(condition_num_classes) > 0 else int(score_active_classes),
                            target_label_knockout_prob=float(args.target_label_knockout_prob),
                            target_label_knockout_min_keep=int(args.target_label_knockout_min_keep),
                            target_label_knockout_max_drop_frac=float(args.target_label_knockout_max_drop_frac),
                            target_label_knockout_seed=int(args.seed) + (int(cycle_id) * 10007) + (int(round_id) * 97) + 1,
                            step_preview_callback=_make_c_step_callback(cycle_id=int(cycle_id), round_id=int(round_id)),
                            stop_requested=_poll_gui_stop,
                        )
                        refresh_rows.append({"stage": "stage1_gestation", "cycle": cycle_id, "round": round_id, **ref_stage1})
                        _log(
                            "  Stage 1 gestation refresh: "
                            f"loss={ref_stage1['loss']:.4f} samples={ref_stage1.get('samples', 0)} "
                            f"steps={int(ref_stage1.get('steps_per_epoch', 0))} "
                            f"elapsed={ref_stage1.get('elapsed_sec', 0.0):.1f}s"
                        )
                        did_gestation_stage1_round = True
                        _save_training_segment_snapshot(
                            enabled=bool(args.checkpoint_after_training_segment),
                            out_dir=out_dir,
                            objective_mode=args.objective_mode,
                            run_tag=run_tag,
                            segment=f"stage1_gestation_cycle_{int(cycle_id)}_round_{int(round_id)}",
                            best_cfg=best_cfg,
                            classifier=classifier,
                            transformer=transformer,
                            generator=generator,
                            discriminator=discriminator,
                            wave_classifier=wave_classifier,
                            transformer_history=transformer_hist,
                            generator_history=generator_hist,
                            wave_classifier_history=wave_classifier_hist,
                            orchestration_history=orchestration_rows,
                            refresh_history=refresh_rows,
                            gate_history=gate_history,
                            gate_status={
                                "gestation_streak": int(gestation_gate_streak),
                                "berkeley_streak": int(berkeley_gate_streak),
                                "wave_streak": int(wave_gate_streak),
                                "generator_streak": int(generator_gate_streak),
                                "transformer_streak": int(transformer_gate_streak),
                            },
                            extra={
                                "sample_bits": int(sample_bits),
                                "chunk_samples": int(chunk_samples),
                                "patch_size": int(args.patch_size),
                                "stage1_rows": int(gestation_stage_loader_count),
                            },
                        )
                    if bool(did_gestation_stage1_round):
                        _cleanup_cuda_allocator()
                    gestation_refresh_ready = bool(gestation_gate_streak >= gate_config["gestation_maintain_rounds"])
                    if bool(gestation_refresh_ready) and (refresh_loader is None):
                        _init_refresh_resources(
                            reason=f"gestation_unlocked:cycle={int(cycle_id)} round={int(round_id)}",
                            samples_seen_for_startup=int(berkeley_refresh_samples_seen),
                        )

                    refresh_only_loss_mode = bool(berkeley_loss_target_value > 0.0 and bool(berkeley_loss_target_unmet))
                    did_refresh_round = False
                    berkeley_ready_before_refresh = bool(
                        (not bool(total_gate_runtime_enabled))
                        or (berkeley_gate_streak >= gate_config["berkeley_maintain_rounds"])
                    )
                    force_full_stage2_until_gate2 = bool(
                        refresh_loader is not None
                        and bool(gestation_refresh_ready)
                        and bool(total_gate_runtime_enabled)
                        and (not bool(berkeley_ready_before_refresh))
                    )
                    run_refresh_this_round = (
                        refresh_loader is not None
                        and (
                            bool(force_full_stage2_until_gate2)
                            or (
                                int(args.berkeley_refresh_round_every) > 0
                                and (global_round % int(args.berkeley_refresh_round_every) == 0)
                            )
                            or bool(refresh_only_loss_mode)
                        )
                    )
                    if run_refresh_this_round:
                        need_full_refresh_startup = (
                            int(berkeley_refresh_required_samples) > 0
                            and int(berkeley_refresh_samples_seen) < int(berkeley_refresh_required_samples)
                        )
                        need_full_refresh_loss = bool(refresh_only_loss_mode)
                        need_full_refresh_gate2 = bool(force_full_stage2_until_gate2)
                        refresh_cache_x = refresh_cache.get("x") if isinstance(refresh_cache, dict) else None
                        refresh_cache_y = refresh_cache.get("y") if isinstance(refresh_cache, dict) else None
                        refresh_cache_batch = int(refresh_run_batch_size)
                        if (
                            need_full_refresh_startup
                            and isinstance(refresh_cache, dict)
                            and int(refresh_cache_samples) > 0
                            and int(refresh_cache_samples) < int(berkeley_refresh_required_samples)
                        ):
                            _log(
                                "  Berkeley refresh startup requirement active: forcing loader mode "
                                f"(cache_samples={refresh_cache_samples} < required={berkeley_refresh_required_samples})"
                            )
                            refresh_cache_x = None
                            refresh_cache_y = None
                            refresh_cache_batch = 0
                        if need_full_refresh_gate2:
                            if (
                                refresh_cache_x is not None
                                or refresh_cache_y is not None
                                or int(refresh_cache_batch) > 0
                            ):
                                _log(
                                    "  Stage 2 pre-Gate2 lock: forcing full non-validation deck refresh "
                                    "(cache disabled until Gate 2 is ready)."
                                )
                            refresh_cache_x = None
                            refresh_cache_y = None
                            refresh_cache_batch = 0
                        if need_full_refresh_loss:
                            if (
                                refresh_cache_x is not None
                                or refresh_cache_y is not None
                                or int(refresh_cache_batch) > 0
                            ):
                                _log(
                                    "  Berkeley loss target unmet: forcing full-dataset loader refresh "
                                    f"(latest_target={berkeley_loss_target_value:.4f})"
                                )
                            refresh_cache_x = None
                            refresh_cache_y = None
                            refresh_cache_batch = 0
                        if need_full_refresh_startup or need_full_refresh_loss or need_full_refresh_gate2:
                            min_refresh_steps = max(1, int(len(refresh_loader)))
                        else:
                            min_refresh_steps = 0
                        refresh_max_steps = int(args.berkeley_refresh_max_steps)
                        if (need_full_refresh_startup or need_full_refresh_loss or need_full_refresh_gate2) and int(min_refresh_steps) > 0:
                            refresh_max_steps = int(min_refresh_steps)
                        ref = _run_berkeley_refresh_epochs(
                            classifier=classifier,
                            loader=refresh_loader,
                            device=device,
                            epochs=int(args.berkeley_refresh_epochs),
                            lr=float(args.berkeley_refresh_lr),
                            weight_decay=float(args.berkeley_refresh_weight_decay),
                            max_steps=int(refresh_max_steps),
                            min_steps=int(min_refresh_steps),
                            lr_sine_cycles=float(args.lr_sine_cycles),
                            lr_sine_frequency=float(args.lr_sine_frequency),
                            lr_sine_tail_fraction=float(args.lr_sine_tail_fraction),
                            lr_sine_min_scale=float(args.lr_sine_min_scale),
                            amp_enabled=amp_enabled,
                            amp_dtype=args.amp_dtype,
                            channels_last=bool(args.channels_last),
                            grad_accum_steps=grad_accum_steps,
                            log_every=int(args.berkeley_refresh_log_every),
                            max_seconds=float(args.berkeley_refresh_max_seconds),
                            cache_x=refresh_cache_x,
                            cache_y=refresh_cache_y,
                            cache_batch_size=int(refresh_cache_batch),
                            active_classes=int(score_active_classes),
                            target_label_knockout_prob=float(args.target_label_knockout_prob),
                            target_label_knockout_min_keep=int(args.target_label_knockout_min_keep),
                            target_label_knockout_max_drop_frac=float(args.target_label_knockout_max_drop_frac),
                            target_label_knockout_seed=int(args.seed) + (int(cycle_id) * 10007) + (int(round_id) * 97) + 2,
                            step_preview_callback=_make_c_step_callback(cycle_id=int(cycle_id), round_id=int(round_id)),
                            stop_requested=_poll_gui_stop,
                        )
                        refresh_rows.append({"stage": "orchestration", "cycle": cycle_id, "round": round_id, **ref})
                        berkeley_refresh_samples_seen += int(max(0, int(ref.get("samples", 0))))
                        _log(
                            "  Berkeley refresh: "
                            f"loss={ref['loss']:.4f} samples={ref.get('samples', 0)} "
                            f"source={ref.get('source', 'loader')} "
                            f"startup_seen={berkeley_refresh_samples_seen}/{berkeley_refresh_required_samples} "
                            f"elapsed={ref.get('elapsed_sec', 0.0):.1f}s"
                        )
                        did_refresh_round = True
                        ref_loss = float(ref.get("loss", float("nan")))
                        if berkeley_loss_target_value > 0.0:
                            berkeley_loss_target_unmet = bool((not math.isfinite(ref_loss)) or (ref_loss > berkeley_loss_target_value))
                        _save_training_segment_snapshot(
                            enabled=bool(args.checkpoint_after_training_segment),
                            out_dir=out_dir,
                            objective_mode=args.objective_mode,
                            run_tag=run_tag,
                            segment=f"orchestration_refresh_cycle_{int(cycle_id)}_round_{int(round_id)}",
                            best_cfg=best_cfg,
                            classifier=classifier,
                            transformer=transformer,
                            generator=generator,
                            discriminator=discriminator,
                            wave_classifier=wave_classifier,
                            transformer_history=transformer_hist,
                            generator_history=generator_hist,
                            wave_classifier_history=wave_classifier_hist,
                            orchestration_history=orchestration_rows,
                            refresh_history=refresh_rows,
                            gate_history=gate_history,
                            gate_status={
                                "gestation_streak": int(gestation_gate_streak),
                                "berkeley_streak": int(berkeley_gate_streak),
                                "wave_streak": int(wave_gate_streak),
                                "generator_streak": int(generator_gate_streak),
                                "transformer_streak": int(transformer_gate_streak),
                            },
                            extra={
                                "sample_bits": int(sample_bits),
                                "chunk_samples": int(chunk_samples),
                                "patch_size": int(args.patch_size),
                            },
                        )
                        if need_full_refresh_loss and bool(berkeley_loss_target_unmet) and (not bool(gate_override_active)):
                            berkeley_gate_streak = 0
                            wave_gate_streak = 0
                            _log(
                                "  full-dataset refresh-only round: "
                                f"loss={ref_loss:.4f} target={berkeley_loss_target_value:.4f} "
                                "(downstream skipped)"
                            )
                            round_row = {
                                "cycle": cycle_id,
                                "round": round_id,
                                "stage": "C",
                                "berkeley_refresh_only": True,
                                "berkeley_refresh_loss": (None if not math.isfinite(ref_loss) else float(max(0.0, ref_loss))),
                                "berkeley_refresh_loss_target": float(berkeley_loss_target_value),
                                "downstream_skipped": "berkeley_loss_refresh_only",
                                "berkeley_gate_pass": False,
                                "berkeley_streak": int(berkeley_gate_streak),
                                "berkeley_ready": False,
                                "wave_entropy_streak": int(wave_gate_streak),
                                "wave_entropy_ready": False,
                                "berkeley_gate_detail": {
                                    "mode": "hard_loss_full_refresh",
                                    "reason": f"loss>{berkeley_loss_target_value:.4f}",
                                    "loss_pass": False,
                                    "loss_target": float(berkeley_loss_target_value),
                                    "loss": (None if not math.isfinite(ref_loss) else float(max(0.0, ref_loss))),
                                    "refresh_pass": True,
                                    "refresh_samples_seen": int(berkeley_refresh_samples_seen),
                                    "refresh_samples_required": int(berkeley_refresh_required_samples),
                                },
                            }
                            gate_history.append(
                                {
                                    "cycle": cycle_id,
                                    "round": round_id,
                                    "gestation_pass": bool(gestation_gate_streak >= gate_config["gestation_maintain_rounds"]),
                                    "gestation_streak": int(gestation_gate_streak),
                                    "gestation_ready": bool(gestation_gate_streak >= gate_config["gestation_maintain_rounds"]),
                                    "berkeley_pass": False,
                                    "berkeley_gate_mode": "hard_loss_full_refresh",
                                    "berkeley_gate_reason": f"loss>{berkeley_loss_target_value:.4f}",
                                    "berkeley_loss_pass": False,
                                    "berkeley_loss_target": float(berkeley_loss_target_value),
                                    "berkeley_refresh_pass": True,
                                    "berkeley_refresh_seen": int(berkeley_refresh_samples_seen),
                                    "berkeley_refresh_required": int(berkeley_refresh_required_samples),
                                    "berkeley_streak": int(berkeley_gate_streak),
                                    "berkeley_ready": False,
                                    "wave_pass": False,
                                    "wave_streak": int(wave_gate_streak),
                                    "wave_ready": False,
                                    "generator_pass": None,
                                    "generator_streak": int(generator_gate_streak),
                                    "generator_ready": None,
                                    "transformer_pass": None,
                                    "transformer_streak": int(transformer_gate_streak),
                                    "transformer_ready": None,
                                }
                            )
                            orchestration_rows.append(round_row)
                            _run_transformer_round_preview(
                                stage="S1",
                                cycle_id=int(cycle_id),
                                round_id=int(round_id),
                                global_round=int(global_round),
                                extra={
                                    "gestation_classifier_eval": round_row.get("gestation_classifier_eval", None),
                                    "gestation_gate_detail": round_row.get("gestation_gate_detail", None),
                                    "gestation_gate_pass": round_row.get("gestation_gate_pass", None),
                                },
                            )
                            _run_transformer_round_preview(
                                stage="G1",
                                cycle_id=int(cycle_id),
                                round_id=int(round_id),
                                global_round=int(global_round),
                                extra={
                                    "gestation_classifier_eval": round_row.get("gestation_classifier_eval", None),
                                    "gestation_gate_detail": round_row.get("gestation_gate_detail", None),
                                    "gestation_gate_pass": round_row.get("gestation_gate_pass", None),
                                },
                            )
                            stage_chain_ready = bool(gestation_gate_streak >= gate_config["gestation_maintain_rounds"])
                            if bool(stage_chain_ready):
                                _run_transformer_round_preview(
                                    stage="S2",
                                    cycle_id=int(cycle_id),
                                    round_id=int(round_id),
                                    global_round=int(global_round),
                                    extra={
                                        "initial_stage_2_ready": False,
                                        "initial_stage_2_mode": "refresh_only_loss_clamp",
                                    },
                                )
                                _run_transformer_round_preview(
                                    stage="G2",
                                    cycle_id=int(cycle_id),
                                    round_id=int(round_id),
                                    global_round=int(global_round),
                                    extra={
                                        "berkeley_classifier_eval": round_row.get("berkeley_classifier_eval", None),
                                        "berkeley_gate_detail": round_row.get("berkeley_gate_detail", None),
                                        "berkeley_gate_pass": round_row.get("berkeley_gate_pass", None),
                                    },
                                )
                                _run_transformer_round_preview(
                                    stage="C",
                                    cycle_id=int(cycle_id),
                                    round_id=int(round_id),
                                    global_round=int(global_round),
                                    extra={
                                        "downstream_skipped": "berkeley_loss_refresh_only",
                                        "gestation_classifier_eval": round_row.get("gestation_classifier_eval", None),
                                        "gestation_gate_detail": round_row.get("gestation_gate_detail", None),
                                        "berkeley_classifier_eval": round_row.get("berkeley_classifier_eval", None),
                                        "berkeley_gate_detail": round_row.get("berkeley_gate_detail", None),
                                        "stage_c_stream_index": -1,
                                        "stage_c_record_index": -1,
                                        "stage_c_metric": {},
                                    },
                                )
                            if _poll_gui_stop(note=f"cycle={int(cycle_id)} round={int(round_id)} stage=C"):
                                stop_orchestration = True
                                break
                            continue
                        elif need_full_refresh_loss and bool(berkeley_loss_target_unmet) and bool(gate_override_active):
                            _log(
                                "  gate override active: continuing downstream despite refresh-only loss clamp "
                                f"(loss={ref_loss:.4f} target={berkeley_loss_target_value:.4f})"
                            )
                    if did_refresh_round:
                        _cleanup_cuda_allocator()
                    if _poll_gui_stop(note=f"cycle={int(cycle_id)} round={int(round_id)} post-refresh"):
                        stop_orchestration = True
                        break

                    berkeley_round_metric_idx = _sample_round_classifier_val_indices()
                    berkeley_round_metric = _score_config_with_classifier(
                        records=records,
                        indices=berkeley_round_metric_idx,
                        cfg=best_cfg,
                        image_hw=image_hw,
                        classifier=classifier,
                        device=device,
                        batch_size=args.classifier_batch_size,
                        score_topk=args.score_topk,
                        score_threshold=args.score_threshold,
                        score_w_topk=args.score_w_topk,
                        score_w_cov=args.score_w_cov,
                        score_w_mean=args.score_w_mean,
                        amp_enabled=amp_enabled,
                        amp_dtype=args.amp_dtype,
                        channels_last=bool(args.channels_last),
                        active_classes=int(score_active_classes),
                    )
                    two_stage_gate_eval = _evaluate_two_stage_semantic_gates(round_index=int(global_round))
                    gestation_round_eval = two_stage_gate_eval.get("gestation_eval", None)
                    gestation_gate_eval = dict(two_stage_gate_eval.get("gestation_detail", {}))
                    gestation_pass = bool(two_stage_gate_eval.get("gestation_pass", True))
                    gestation_ready = bool(two_stage_gate_eval.get("gestation_ready", True))
                    stage2_mode = str(
                        two_stage_gate_eval.get(
                            "initial_stage_2_mode",
                            ("ready" if bool(two_stage_gate_eval.get("initial_stage_2_ready", False)) else "waiting_for_gate_1"),
                        )
                    )
                    berkeley_round_eval = two_stage_gate_eval.get("total_eval", None)
                    berkeley_gate_eval = dict(two_stage_gate_eval.get("total_detail", {}))
                    berkeley_pass = bool(two_stage_gate_eval.get("total_pass", True))
                    berkeley_ready = bool(two_stage_gate_eval.get("total_ready", True))

                    wave_pass = (
                        (float(berkeley_round_metric.get("mean_entropy", 0.0)) >= gate_config["wave_min_entropy"])
                        and (float(berkeley_round_metric["score"]) >= gate_config["berkeley_min_score"])
                    )
                    if wave_pass:
                        wave_gate_streak += 1
                    else:
                        wave_gate_streak = 0
                    wave_ready = wave_gate_streak >= gate_config["wave_maintain_rounds"]

                    round_row: Dict = {
                        "cycle": cycle_id,
                        "round": round_id,
                        "berkeley_metric": berkeley_round_metric,
                        "gestation_classifier_eval": gestation_round_eval,
                        "gestation_gate_detail": dict(gestation_gate_eval),
                        "gestation_gate_pass": bool(gestation_pass),
                        "gestation_streak": int(gestation_gate_streak),
                        "gestation_ready": bool(gestation_ready),
                        "initial_stage_2_ready": bool(two_stage_gate_eval.get("initial_stage_2_ready", False)),
                        "initial_stage_2_mode": str(stage2_mode),
                        "berkeley_classifier_eval": berkeley_round_eval,
                        "berkeley_gate_detail": dict(berkeley_gate_eval),
                        "berkeley_gate_pass": bool(berkeley_pass),
                        "berkeley_streak": int(berkeley_gate_streak),
                        "berkeley_ready": bool(berkeley_ready),
                        "wave_entropy_gate_pass": bool(wave_pass),
                        "wave_entropy_streak": int(wave_gate_streak),
                        "wave_entropy_ready": bool(wave_ready),
                        "gate_override": bool(gate_override_active),
                    }
                    wave_feedback_ctx = _wave_feedback_snapshot(
                        wave_classifier_eval=wave_classifier_last_eval,
                        wave_zero_shot_eval=wave_zero_shot_last_eval,
                        zero_shot_weight=float(gate_config["wave_feedback_zero_shot_weight"]),
                    )
                    feedback_transformer_score_weight = _feedback_scaled_weight(
                        base_weight=float(args.transformer_loss_score_target_weight),
                        feedback_score=wave_feedback_ctx.get("combined_score", None),
                        boost=float(gate_config["wave_feedback_transformer_loss_boost"]),
                    )
                    round_row["wave_feedback_pre"] = wave_feedback_ctx
                    round_row["feedback_weights"] = {
                        "transformer_score_target_weight": float(feedback_transformer_score_weight),
                    }
                    _run_transformer_round_preview(
                        stage="S1",
                        cycle_id=int(cycle_id),
                        round_id=int(round_id),
                        global_round=int(global_round),
                        extra={
                            "gestation_classifier_eval": round_row.get("gestation_classifier_eval", None),
                            "gestation_gate_detail": round_row.get("gestation_gate_detail", None),
                            "gestation_gate_pass": round_row.get("gestation_gate_pass", None),
                        },
                    )
                    _run_transformer_round_preview(
                        stage="G1",
                        cycle_id=int(cycle_id),
                        round_id=int(round_id),
                        global_round=int(global_round),
                        extra={
                            "gestation_classifier_eval": round_row.get("gestation_classifier_eval", None),
                            "gestation_gate_detail": round_row.get("gestation_gate_detail", None),
                            "gestation_gate_pass": round_row.get("gestation_gate_pass", None),
                        },
                    )
                    if bool(gestation_ready):
                        _run_transformer_round_preview(
                            stage="S2",
                            cycle_id=int(cycle_id),
                            round_id=int(round_id),
                            global_round=int(global_round),
                            extra={
                                "initial_stage_2_ready": round_row.get("initial_stage_2_ready", False),
                                "initial_stage_2_mode": round_row.get("initial_stage_2_mode", ""),
                            },
                        )
                    if bool(gestation_ready) and bool(two_stage_gate_eval.get("initial_stage_2_ready", False)):
                        _run_transformer_round_preview(
                            stage="G2",
                            cycle_id=int(cycle_id),
                            round_id=int(round_id),
                            global_round=int(global_round),
                            extra={
                                "berkeley_classifier_eval": round_row.get("berkeley_classifier_eval", None),
                                "berkeley_gate_detail": round_row.get("berkeley_gate_detail", None),
                                "berkeley_gate_pass": round_row.get("berkeley_gate_pass", None),
                            },
                        )

                    if (not bool(gate_override_active)) and (not gestation_ready or not berkeley_ready or not wave_ready):
                        if not bool(gestation_ready):
                            round_row["downstream_skipped"] = "gestation_gate"
                        elif not bool(berkeley_ready):
                            round_row["downstream_skipped"] = "berkeley_gate"
                        else:
                            round_row["downstream_skipped"] = "wave_entropy_gate"
                        conf_txt = (
                            f"gest_reason={gestation_gate_eval.get('reason', 'pass')}, "
                            f"gest_cov={int(gestation_gate_eval.get('refresh_samples_seen', 0))}/"
                            f"{int(gestation_gate_eval.get('refresh_samples_required', 0))}, "
                            f"stage2_ready={1 if bool(two_stage_gate_eval.get('initial_stage_2_ready', False)) else 0}, "
                            f"stage2_mode={stage2_mode}, "
                            f"total_mode={berkeley_gate_eval.get('mode', 'hard_loss_full_refresh')}, "
                            f"total_reason={berkeley_gate_eval.get('reason', 'pass')}, "
                            f"total_cov={int(berkeley_gate_eval.get('refresh_samples_seen', 0))}/"
                            f"{int(berkeley_gate_eval.get('refresh_samples_required', 0))}, "
                        )
                        if berkeley_round_eval is not None:
                            conf_txt = (
                                f"conf={berkeley_round_eval['mean_confidence']:.4f}, "
                                f"macro_f1={berkeley_round_eval['macro_f1']:.4f}, "
                                f"loss={float(berkeley_round_eval.get('loss', 0.0)):.4f}, "
                                f"{conf_txt}"
                            )
                        _log(
                            "  downstream skipped by pre-transformer gates "
                            f"({conf_txt}"
                            f"wave_entropy={berkeley_round_metric.get('mean_entropy', 0.0):.4f}, "
                            f"gest_streak={gestation_gate_streak}/{gate_config['gestation_maintain_rounds']}, "
                            f"berk_streak={berkeley_gate_streak}/{gate_config['berkeley_maintain_rounds']}, "
                            f"wave_streak={wave_gate_streak}/{gate_config['wave_maintain_rounds']})"
                        )
                        gate_history.append(
                            {
                                "cycle": cycle_id,
                                "round": round_id,
                                "gestation_pass": bool(gestation_pass),
                                "gestation_gate_reason": str(gestation_gate_eval.get("reason", "pass")),
                                "gestation_loss_target": gestation_gate_eval.get("loss_target", None),
                                "gestation_streak": int(gestation_gate_streak),
                                "gestation_ready": bool(gestation_ready),
                                "berkeley_pass": bool(berkeley_pass),
                                "berkeley_gate_mode": str(berkeley_gate_eval.get("mode", "hard_loss_full_refresh")),
                                "berkeley_gate_reason": str(berkeley_gate_eval.get("reason", "pass")),
                                "berkeley_loss_pass": bool(berkeley_gate_eval.get("loss_pass", True)),
                                "berkeley_loss_target": berkeley_gate_eval.get("loss_target", None),
                                "berkeley_refresh_pass": bool(berkeley_gate_eval.get("refresh_pass", True)),
                                "berkeley_refresh_seen": int(berkeley_gate_eval.get("refresh_samples_seen", 0)),
                                "berkeley_refresh_required": int(berkeley_gate_eval.get("refresh_samples_required", 0)),
                                "berkeley_streak": int(berkeley_gate_streak),
                                "berkeley_ready": bool(berkeley_ready),
                                "wave_pass": bool(wave_pass),
                                "wave_streak": int(wave_gate_streak),
                                "wave_ready": bool(wave_ready),
                                "generator_pass": None,
                                "generator_streak": int(generator_gate_streak),
                                "generator_ready": None,
                                "transformer_pass": None,
                                "transformer_streak": int(transformer_gate_streak),
                                "transformer_ready": None,
                                "gate_override": bool(gate_override_active),
                            }
                        )
                        orchestration_rows.append(round_row)
                        if bool(gestation_ready) and bool(two_stage_gate_eval.get("initial_stage_2_ready", False)):
                            _run_transformer_round_preview(
                                stage="C",
                                cycle_id=int(cycle_id),
                                round_id=int(round_id),
                                global_round=int(global_round),
                                extra={
                                    "downstream_skipped": str(round_row.get("downstream_skipped", "")),
                                    "gestation_classifier_eval": round_row.get("gestation_classifier_eval", None),
                                    "gestation_gate_detail": round_row.get("gestation_gate_detail", None),
                                    "berkeley_classifier_eval": round_row.get("berkeley_classifier_eval", None),
                                    "berkeley_gate_detail": round_row.get("berkeley_gate_detail", None),
                                    "stage_c_stream_index": -1,
                                    "stage_c_record_index": (
                                        int(berkeley_round_metric_idx[0])
                                        if isinstance(berkeley_round_metric_idx, (list, tuple, np.ndarray))
                                        and len(berkeley_round_metric_idx) > 0
                                        else -1
                                    ),
                                    "stage_c_metric": (
                                        dict(berkeley_round_metric)
                                        if isinstance(berkeley_round_metric, dict)
                                        else {}
                                    ),
                                },
                            )
                        if int(args.checkpoint_every_round) > 0 and (global_round % int(args.checkpoint_every_round) == 0):
                            _save_pipeline_checkpoint(
                                pipeline_checkpoint_path,
                                _decorate_checkpoint_payload(
                                    {
                                        "run_tag": run_tag,
                                        "objective_mode": args.objective_mode,
                                        "orchestration_mode": mode,
                                        "last_cycle": int(cycle_id),
                                        "last_round": int(round_id),
                                        "best_cfg": best_cfg.to_dict(),
                                        "classifier_state": classifier.state_dict(),
                                        "transformer_state": transformer.state_dict(),
                                        "generator_state": (generator.state_dict() if generator is not None else None),
                                        "discriminator_state": (discriminator.state_dict() if discriminator is not None else None),
                                        "wave_classifier_state": (wave_classifier.state_dict() if wave_classifier is not None else None),
                                        "classifier_init_info": classifier_init_info,
                                        "classifier_resume_info": classifier_resume_info,
                                        "transformer_resume_info": transformer_resume_info,
                                        "generator_resume_info": generator_resume_info,
                                        "discriminator_resume_info": discriminator_resume_info,
                                        "wave_classifier_init_info": wave_classifier_init_info,
                                        "wave_classifier_resume_info": wave_classifier_resume_info,
                                        "transformer_history": transformer_hist,
                                        "generator_history": generator_hist,
                                        "wave_classifier_history": wave_classifier_hist,
                                        "orchestration_history": orchestration_rows,
                                        "refresh_history": refresh_rows,
                                        "gate_history": gate_history,
                                        "gate_status": {
                                            "gestation_streak": int(gestation_gate_streak),
                                            "berkeley_streak": int(berkeley_gate_streak),
                                            "wave_streak": int(wave_gate_streak),
                                            "generator_streak": int(generator_gate_streak),
                                            "transformer_streak": int(transformer_gate_streak),
                                        },
                                        "wave_classifier_last_eval": wave_classifier_last_eval,
                                        "wave_zero_shot_query_source": str(wave_zero_shot_query_source),
                                        "wave_zero_shot_queries": list(wave_zero_shot_queries),
                                        "wave_zero_shot_query_rows": list(wave_zero_shot_query_rows),
                                        "wave_zero_shot_last_eval": wave_zero_shot_last_eval,
                                        "wave_zero_shot_history": list(wave_zero_shot_history),
                                        "accepted_library_count": int(library_serial),
                                        "sample_bits": int(sample_bits),
                                        "chunk_samples": int(chunk_samples),
                                        "patch_size": int(args.patch_size),
                                        "timestamp": time.time(),
                                    }
                                ),
                            )
                        if _poll_gui_stop(note=f"cycle={int(cycle_id)} round={int(round_id)} stage=C"):
                            stop_orchestration = True
                            break
                        continue
                    if bool(gate_override_active) and ((not gestation_ready) or (not berkeley_ready) or (not wave_ready)):
                        _log(
                            "  gate override active: bypassing pre-transformer gates "
                            f"(gest_ready={int(gestation_ready)} berk_ready={int(berkeley_ready)} wave_ready={int(wave_ready)})"
                        )

                    _offload_inactive_stage_models(
                        reason=f"cycle={int(cycle_id)} round={int(round_id)} transformer"
                    )
                    transformer, round_hist, round_train_batch = _train_transformer_stage_with_retry(
                        stage_tag="R",
                        epochs=max(1, int(args.transformer_epochs_per_round)),
                        steps_per_epoch=max(1, int(args.transformer_steps_per_round)),
                        score_target_weight=float(feedback_transformer_score_weight),
                        score_target_margin=float(args.transformer_loss_score_target_margin),
                        seed=int(args.seed + 9000 + global_round),
                        step_preview_callback=_make_r_step_callback("R", cycle_id=int(cycle_id), round_id=int(round_id)),
                    )
                    for row in round_hist:
                        rr = dict(row)
                        rr["cycle"] = float(cycle_id)
                        rr["round"] = float(round_id)
                        transformer_hist.append(rr)
                    _save_training_segment_snapshot(
                        enabled=bool(args.checkpoint_after_training_segment),
                        out_dir=out_dir,
                        objective_mode=args.objective_mode,
                        run_tag=run_tag,
                        segment=f"round_transformer_cycle_{int(cycle_id)}_round_{int(round_id)}",
                        best_cfg=best_cfg,
                        classifier=classifier,
                        transformer=transformer,
                        generator=generator,
                        discriminator=discriminator,
                        wave_classifier=wave_classifier,
                        transformer_history=transformer_hist,
                        generator_history=generator_hist,
                        wave_classifier_history=wave_classifier_hist,
                        orchestration_history=orchestration_rows,
                        refresh_history=refresh_rows,
                        gate_history=gate_history,
                        gate_status={
                            "gestation_streak": int(gestation_gate_streak),
                            "berkeley_streak": int(berkeley_gate_streak),
                            "wave_streak": int(wave_gate_streak),
                            "generator_streak": int(generator_gate_streak),
                            "transformer_streak": int(transformer_gate_streak),
                        },
                        extra={
                            "sample_bits": int(sample_bits),
                            "chunk_samples": int(chunk_samples),
                            "patch_size": int(args.patch_size),
                            "accepted_library_count": int(library_serial),
                        },
                    )

                    feature_eval, round_eval_batch = _evaluate_transformer_stage_with_retry(
                        stage_tag="R",
                        batch_size_hint=int(round_train_batch),
                        max_batches=20,
                    )
                    _log(
                        f"  transformer score: before={feature_eval['score_before']:.4f} "
                        f"after={feature_eval['score_after']:.4f} gain={feature_eval['score_gain']:.4f}"
                    )

                    wave_feedback_gate_ok, wave_feedback_gate_reason = _wave_feedback_gate_pass(
                        feedback=wave_feedback_ctx,
                        min_acc=float(gate_config["wave_feedback_min_acc"]),
                        min_zero_shot_top1_rate=float(gate_config["wave_feedback_min_zs_top1_rate"]),
                    )
                    transformer_pass = (
                        (float(feature_eval["score_after"]) >= gate_config["transformer_min_score_after"])
                        and (float(feature_eval["score_gain"]) >= gate_config["transformer_min_gain"])
                        and bool(wave_feedback_gate_ok)
                    )
                    if transformer_pass:
                        transformer_gate_streak += 1
                    else:
                        transformer_gate_streak = 0
                    transformer_ready = (
                        transformer_gate_streak >= gate_config["transformer_maintain_rounds"]
                    )
                    transformer_ready_effective = bool(transformer_ready or bool(gate_override_active))
                    round_row["transformer_eval"] = feature_eval
                    round_row["transformer_train_batch_size"] = int(round_train_batch)
                    round_row["transformer_eval_batch_size"] = int(round_eval_batch)
                    round_row["transformer_gate_pass"] = bool(transformer_pass)
                    round_row["transformer_streak"] = int(transformer_gate_streak)
                    round_row["transformer_ready"] = bool(transformer_ready)
                    round_row["transformer_ready_effective"] = bool(transformer_ready_effective)
                    round_row["wave_feedback_gate_pass"] = bool(wave_feedback_gate_ok)
                    round_row["wave_feedback_gate_reason"] = str(wave_feedback_gate_reason)

                    gate_history.append(
                        {
                            "cycle": cycle_id,
                            "round": round_id,
                            "gestation_pass": bool(gestation_pass),
                            "gestation_streak": int(gestation_gate_streak),
                            "gestation_ready": bool(gestation_ready),
                            "berkeley_pass": bool(berkeley_pass),
                            "berkeley_streak": int(berkeley_gate_streak),
                            "berkeley_ready": bool(berkeley_ready),
                            "wave_pass": bool(wave_pass),
                            "wave_streak": int(wave_gate_streak),
                            "wave_ready": bool(wave_ready),
                            "generator_pass": None,
                            "generator_streak": int(generator_gate_streak),
                            "generator_ready": None,
                            "transformer_pass": bool(transformer_pass),
                            "transformer_streak": int(transformer_gate_streak),
                            "transformer_ready": bool(transformer_ready),
                            "gate_override": bool(gate_override_active),
                            "wave_feedback_gate_pass": bool(wave_feedback_gate_ok),
                            "wave_feedback_gate_reason": str(wave_feedback_gate_reason),
                        }
                    )
                    _run_transformer_round_preview(
                        stage="R",
                        cycle_id=int(cycle_id),
                        round_id=int(round_id),
                        global_round=int(global_round),
                        extra={"transformer_eval": feature_eval},
                    )

                    if bool(gate_override_active) and (not bool(transformer_ready)):
                        _log(
                            "  gate override active: bypassing transformer gate "
                            f"(streak={transformer_gate_streak}/{gate_config['transformer_maintain_rounds']})"
                        )
                    if not bool(transformer_ready_effective):
                        round_row["downstream_skipped"] = "transformer_gate"
                        _log(
                            "  wave-classifier stage skipped by transformer gate "
                            f"(after={feature_eval['score_after']:.4f} min_after={gate_config['transformer_min_score_after']:.4f}, "
                            f"gain={feature_eval['score_gain']:.4f} min_gain={gate_config['transformer_min_gain']:.4f}, "
                            f"streak={transformer_gate_streak}/{gate_config['transformer_maintain_rounds']}, "
                            f"wave_feedback={wave_feedback_gate_reason})"
                        )
                        orchestration_rows.append(round_row)
                        if int(args.checkpoint_every_round) > 0 and (global_round % int(args.checkpoint_every_round) == 0):
                            _save_pipeline_checkpoint(
                                pipeline_checkpoint_path,
                                _decorate_checkpoint_payload(
                                    {
                                        "run_tag": run_tag,
                                        "objective_mode": args.objective_mode,
                                        "orchestration_mode": mode,
                                        "last_cycle": int(cycle_id),
                                        "last_round": int(round_id),
                                        "best_cfg": best_cfg.to_dict(),
                                        "classifier_state": classifier.state_dict(),
                                        "transformer_state": transformer.state_dict(),
                                        "generator_state": (generator.state_dict() if generator is not None else None),
                                        "discriminator_state": (discriminator.state_dict() if discriminator is not None else None),
                                        "wave_classifier_state": (wave_classifier.state_dict() if wave_classifier is not None else None),
                                        "classifier_init_info": classifier_init_info,
                                        "classifier_resume_info": classifier_resume_info,
                                        "transformer_resume_info": transformer_resume_info,
                                        "generator_resume_info": generator_resume_info,
                                        "discriminator_resume_info": discriminator_resume_info,
                                        "wave_classifier_init_info": wave_classifier_init_info,
                                        "wave_classifier_resume_info": wave_classifier_resume_info,
                                        "transformer_history": transformer_hist,
                                        "generator_history": generator_hist,
                                        "wave_classifier_history": wave_classifier_hist,
                                        "orchestration_history": orchestration_rows,
                                        "refresh_history": refresh_rows,
                                        "gate_history": gate_history,
                                        "gate_status": {
                                            "gestation_streak": int(gestation_gate_streak),
                                            "berkeley_streak": int(berkeley_gate_streak),
                                            "wave_streak": int(wave_gate_streak),
                                            "generator_streak": int(generator_gate_streak),
                                            "transformer_streak": int(transformer_gate_streak),
                                        },
                                        "wave_classifier_last_eval": wave_classifier_last_eval,
                                        "wave_zero_shot_query_source": str(wave_zero_shot_query_source),
                                        "wave_zero_shot_queries": list(wave_zero_shot_queries),
                                        "wave_zero_shot_query_rows": list(wave_zero_shot_query_rows),
                                        "wave_zero_shot_last_eval": wave_zero_shot_last_eval,
                                        "wave_zero_shot_history": list(wave_zero_shot_history),
                                        "accepted_library_count": int(library_serial),
                                        "sample_bits": int(sample_bits),
                                        "chunk_samples": int(chunk_samples),
                                        "patch_size": int(args.patch_size),
                                        "timestamp": time.time(),
                                    }
                                ),
                            )
                        if _poll_gui_stop(note=f"cycle={int(cycle_id)} round={int(round_id)} stage=R"):
                            stop_orchestration = True
                            break
                        continue

                    x_wave_train, y_wave_train, accepted_rows = _build_wave_classifier_dataset_from_transformer(
                        streams=train_streams,
                        labels=train_stream_labels,
                        metas=train_meta,
                        cfg=best_cfg,
                        transformer=transformer,
                        berkeley_classifier=classifier,
                        sample_bits=sample_bits,
                        image_hw=image_hw,
                        chunk_samples=chunk_samples,
                        total_samples=max(1, int(args.wave_cls_train_samples)),
                        batch_size=max(1, int(args.wave_cls_batch_size)),
                        device=device,
                        rng_seed=args.seed + 12000 + (global_round * 11),
                        accept_score_threshold=float(args.accept_score_threshold),
                        accept_l1_threshold=float(args.accept_l1_threshold),
                        library_dir=library_dir,
                        library_limit=max(0, int(args.library_max_per_round)),
                        cycle_id=cycle_id,
                        round_id=round_id,
                        accepted_only=bool(args.wave_cls_accepted_only),
                        run_tag=run_tag,
                        library_serial_start=library_serial,
                        amp_enabled=amp_enabled,
                        amp_dtype=args.amp_dtype,
                        channels_last=bool(args.channels_last),
                        pin_memory=bool(args.pin_memory_wave_batches),
                        semantic_class_names=class_names,
                        semantic_label_bank=label_embedding_bank,
                        stream_target_labels=train_stream_target_labels,
                    )
                    x_wave_val, y_wave_val, _ = _build_wave_classifier_dataset_from_transformer(
                        streams=val_streams,
                        labels=val_stream_labels,
                        metas=val_meta,
                        cfg=best_cfg,
                        transformer=transformer,
                        berkeley_classifier=classifier,
                        sample_bits=sample_bits,
                        image_hw=image_hw,
                        chunk_samples=chunk_samples,
                        total_samples=max(1, int(args.wave_cls_val_samples)),
                        batch_size=max(1, int(args.wave_cls_batch_size)),
                        device=device,
                        rng_seed=args.seed + 12100 + (global_round * 11),
                        accept_score_threshold=float(args.accept_score_threshold),
                        accept_l1_threshold=float(args.accept_l1_threshold),
                        library_dir=library_dir,
                        library_limit=0,
                        cycle_id=cycle_id,
                        round_id=round_id,
                        accepted_only=bool(args.wave_cls_accepted_only),
                        run_tag=run_tag,
                        library_serial_start=library_serial,
                        amp_enabled=amp_enabled,
                        amp_dtype=args.amp_dtype,
                        channels_last=bool(args.channels_last),
                        pin_memory=bool(args.pin_memory_wave_batches),
                        semantic_class_names=class_names,
                        semantic_label_bank=label_embedding_bank,
                        stream_target_labels=val_stream_target_labels,
                    )
                    accepted_total.extend(accepted_rows)
                    library_serial += len(accepted_rows)
                    append_cap = max(0, int(args.transformer_accepted_append_max_per_round))
                    to_append = accepted_rows if append_cap <= 0 else accepted_rows[:append_cap]
                    appended_streams = _append_accepted_rows_to_transformer_streams(
                        rows=to_append,
                        seen_paths=accepted_stream_seen,
                        train_streams=train_streams,
                        train_stream_labels=train_stream_labels,
                        train_meta=train_meta,
                        max_points=int(args.stream_max_points),
                        library_root=library_dir,
                        on_append_row=lambda r: _register_regurgitated_churn_row(r, source="accepted_round_append"),
                    )
                    if appended_streams > 0 and train_stream_target_labels is not None:
                        _append_default_mix_targets(train_stream_target_labels, int(appended_streams))
                    round_row["transformer_train_stream_add"] = int(appended_streams)
                    round_row["transformer_train_stream_total"] = int(len(train_streams))
                    if appended_streams > 0:
                        _log(
                            "  transformer stream pool extended from accepted library: "
                            f"+{appended_streams} (total={len(train_streams)})"
                        )

                    round_row["wave_train_samples"] = int(x_wave_train.shape[0])
                    round_row["wave_val_samples"] = int(x_wave_val.shape[0])
                    round_row["accepted_library"] = len(accepted_rows)
                    can_train_wave = (
                        int(x_wave_train.shape[0]) >= 2
                        and int(x_wave_val.shape[0]) >= 2
                        and int(torch.unique(y_wave_train).numel()) >= 2
                    )
                    if can_train_wave:
                        wave_classifier, round_wave_hist = train_classifier(
                            model=wave_classifier,
                            x_train=x_wave_train,
                            y_train=y_wave_train,
                            x_val=x_wave_val,
                            y_val=y_wave_val,
                            device=device,
                            epochs=max(1, int(args.wave_cls_epochs_per_round)),
                            batch_size=max(1, int(args.wave_cls_batch_size)),
                            lr=float(args.wave_cls_lr),
                            lr_sine_cycles=args.lr_sine_cycles,
                            lr_sine_frequency=args.lr_sine_frequency,
                            lr_sine_tail_fraction=args.lr_sine_tail_fraction,
                            lr_sine_min_scale=args.lr_sine_min_scale,
                            amp=amp_enabled,
                            amp_dtype=args.amp_dtype,
                            channels_last=bool(args.channels_last),
                            grad_accum_steps=grad_accum_steps,
                            cache_dataset_on_device=bool(args.cache_wave_cls_dataset_on_device),
                            seed=args.seed + 13000 + global_round,
                            log_every_steps=max(0, int(args.wave_cls_log_every)),
                            step_preview_callback=_make_w_step_callback(cycle_id=int(cycle_id), round_id=int(round_id)),
                            stop_requested=_poll_gui_stop,
                            semantic_mix_noise_prob=float(args.wave_cls_semantic_mix_noise_prob),
                            semantic_mix_noise_std=float(args.wave_cls_semantic_mix_noise_std),
                            semantic_mix_blend_min=float(args.wave_cls_semantic_mix_blend_min),
                            semantic_mix_blend_max=float(args.wave_cls_semantic_mix_blend_max),
                            semantic_mix_class_index=int(split_mix_class_index),
                        )
                        for row in round_wave_hist:
                            rr = dict(row)
                            rr["cycle"] = float(cycle_id)
                            rr["round"] = float(round_id)
                            wave_classifier_hist.append(rr)
                        wave_classifier_last_eval = evaluate_classifier(
                            model=wave_classifier,
                            x=x_wave_val,
                            y=y_wave_val,
                            batch_size=max(1, int(args.wave_cls_batch_size)),
                            device=device,
                            amp=amp_enabled,
                            amp_dtype=args.amp_dtype,
                            channels_last=bool(args.channels_last),
                        )
                        round_row["wave_classifier_val"] = wave_classifier_last_eval
                        run_wave_zero_shot = (
                            wave_zero_shot_query_emb is not None
                            and len(wave_zero_shot_queries) > 0
                        )
                        if run_wave_zero_shot and bool(wave_zero_shot_single_probe_once) and len(wave_zero_shot_history) > 0:
                            run_wave_zero_shot = False
                        if run_wave_zero_shot:
                            wave_zero_shot_last_eval = _evaluate_zero_shot_queries_on_images(
                                classifier=wave_classifier,
                                x=x_wave_val,
                                query_texts=wave_zero_shot_queries,
                                query_emb_np=wave_zero_shot_query_emb,
                                batch_size=max(1, int(args.wave_cls_batch_size)),
                                device=device,
                                amp_enabled=amp_enabled,
                                amp_dtype=args.amp_dtype,
                                channels_last=bool(args.channels_last),
                                topk=max(1, int(args.wave_zero_shot_topk)),
                                max_samples=int(args.wave_zero_shot_max_samples),
                                max_report_samples=int(args.wave_zero_shot_report_samples),
                            )
                            round_row["wave_zero_shot"] = {
                                "ran": bool(wave_zero_shot_last_eval.get("ran", False)),
                                "reason": str(wave_zero_shot_last_eval.get("reason", "")),
                                "samples": int(wave_zero_shot_last_eval.get("samples", 0)),
                                "num_queries": int(wave_zero_shot_last_eval.get("num_queries", 0)),
                                "mean_top1_score": float(wave_zero_shot_last_eval.get("mean_top1_score", 0.0)),
                                "best_query": str(wave_zero_shot_last_eval.get("best_query", "")),
                                "best_query_mean_score": float(wave_zero_shot_last_eval.get("best_query_mean_score", 0.0)),
                                "best_query_top1_rate": float(wave_zero_shot_last_eval.get("best_query_top1_rate", 0.0)),
                            }
                            wave_zero_shot_history.append(
                                {
                                    "cycle": float(cycle_id),
                                    "round": float(round_id),
                                    **round_row["wave_zero_shot"],
                                }
                            )
                        _log(
                            f"  wave classifier val: acc={wave_classifier_last_eval['acc']:.4f} "
                            f"loss={wave_classifier_last_eval['loss']:.4f} accepted={len(accepted_rows)}"
                        )
                        if isinstance(round_row.get("wave_zero_shot"), dict):
                            if bool(round_row["wave_zero_shot"].get("ran", False)):
                                _log(
                                    "  wave zero-shot: "
                                    f"top='{round_row['wave_zero_shot'].get('best_query', '')}' "
                                    f"mean={float(round_row['wave_zero_shot'].get('best_query_mean_score', 0.0)):.4f} "
                                    f"top1_rate={float(round_row['wave_zero_shot'].get('best_query_top1_rate', 0.0)):.3f} "
                                    f"samples={int(round_row['wave_zero_shot'].get('samples', 0))}"
                                )
                            else:
                                _log(
                                    "  wave zero-shot skipped: "
                                    f"reason={round_row['wave_zero_shot'].get('reason', 'unknown')}"
                                )
                        _save_training_segment_snapshot(
                            enabled=bool(args.checkpoint_after_training_segment),
                            out_dir=out_dir,
                            objective_mode=args.objective_mode,
                            run_tag=run_tag,
                            segment=f"round_wave_classifier_cycle_{int(cycle_id)}_round_{int(round_id)}",
                            best_cfg=best_cfg,
                            classifier=classifier,
                            transformer=transformer,
                            generator=generator,
                            discriminator=discriminator,
                            wave_classifier=wave_classifier,
                            transformer_history=transformer_hist,
                            generator_history=generator_hist,
                            wave_classifier_history=wave_classifier_hist,
                            orchestration_history=orchestration_rows,
                            refresh_history=refresh_rows,
                            gate_history=gate_history,
                            gate_status={
                                "gestation_streak": int(gestation_gate_streak),
                                "berkeley_streak": int(berkeley_gate_streak),
                                "wave_streak": int(wave_gate_streak),
                                "generator_streak": int(generator_gate_streak),
                                "transformer_streak": int(transformer_gate_streak),
                            },
                            extra={
                                "sample_bits": int(sample_bits),
                                "chunk_samples": int(chunk_samples),
                                "patch_size": int(args.patch_size),
                                "accepted_library_count": int(library_serial),
                            },
                        )
                    else:
                        round_row["wave_classifier_val"] = None
                        round_row["wave_classifier_skipped"] = True
                        round_row["wave_unique_train_labels"] = int(torch.unique(y_wave_train).numel())
                        _log(
                            f"  wave classifier skipped (samples={x_wave_train.shape[0]}/{x_wave_val.shape[0]}, "
                            f"unique_train_labels={round_row['wave_unique_train_labels']})"
                        )
                    feedback_post = _wave_feedback_snapshot(
                        wave_classifier_eval=wave_classifier_last_eval,
                        wave_zero_shot_eval=wave_zero_shot_last_eval,
                        zero_shot_weight=float(gate_config["wave_feedback_zero_shot_weight"]),
                    )
                    feedback_post_pass, feedback_post_reason = _wave_feedback_gate_pass(
                        feedback=feedback_post,
                        min_acc=float(gate_config["wave_feedback_min_acc"]),
                        min_zero_shot_top1_rate=float(gate_config["wave_feedback_min_zs_top1_rate"]),
                    )
                    round_row["wave_feedback_post"] = feedback_post
                    round_row["wave_feedback_gate_post_pass"] = bool(feedback_post_pass)
                    round_row["wave_feedback_gate_post_reason"] = str(feedback_post_reason)
                    if len(gate_history) > 0 and isinstance(gate_history[-1], dict):
                        gate_history[-1]["wave_feedback_post"] = feedback_post
                        gate_history[-1]["wave_feedback_gate_post_pass"] = bool(feedback_post_pass)
                        gate_history[-1]["wave_feedback_gate_post_reason"] = str(feedback_post_reason)
                    if not feedback_post_pass:
                        transformer_gate_streak = 0
                        round_row["transformer_streak"] = int(transformer_gate_streak)
                        round_row["transformer_ready"] = False
                        _log(f"  wave feedback gate reset transformer streak ({feedback_post_reason})")
                    _run_transformer_round_preview(
                        stage="W",
                        cycle_id=int(cycle_id),
                        round_id=int(round_id),
                        global_round=int(global_round),
                        extra={"wave_classifier_val": round_row.get("wave_classifier_val", None)},
                    )
                    orchestration_rows.append(round_row)

                    if int(args.checkpoint_every_round) > 0 and (global_round % int(args.checkpoint_every_round) == 0):
                        _save_pipeline_checkpoint(
                            pipeline_checkpoint_path,
                            _decorate_checkpoint_payload(
                                {
                                    "run_tag": run_tag,
                                    "objective_mode": args.objective_mode,
                                    "orchestration_mode": mode,
                                    "last_cycle": int(cycle_id),
                                    "last_round": int(round_id),
                                    "best_cfg": best_cfg.to_dict(),
                                    "classifier_state": classifier.state_dict(),
                                    "transformer_state": transformer.state_dict(),
                                    "generator_state": (generator.state_dict() if generator is not None else None),
                                    "discriminator_state": (discriminator.state_dict() if discriminator is not None else None),
                                    "wave_classifier_state": (wave_classifier.state_dict() if wave_classifier is not None else None),
                                    "classifier_init_info": classifier_init_info,
                                    "classifier_resume_info": classifier_resume_info,
                                    "transformer_resume_info": transformer_resume_info,
                                    "generator_resume_info": generator_resume_info,
                                    "discriminator_resume_info": discriminator_resume_info,
                                    "wave_classifier_init_info": wave_classifier_init_info,
                                    "wave_classifier_resume_info": wave_classifier_resume_info,
                                    "transformer_history": transformer_hist,
                                    "generator_history": generator_hist,
                                    "wave_classifier_history": wave_classifier_hist,
                                    "orchestration_history": orchestration_rows,
                                    "refresh_history": refresh_rows,
                                    "gate_history": gate_history,
                                    "gate_status": {
                                        "gestation_streak": int(gestation_gate_streak),
                                        "berkeley_streak": int(berkeley_gate_streak),
                                        "wave_streak": int(wave_gate_streak),
                                        "generator_streak": int(generator_gate_streak),
                                        "transformer_streak": int(transformer_gate_streak),
                                    },
                                    "wave_classifier_last_eval": wave_classifier_last_eval,
                                    "wave_zero_shot_query_source": str(wave_zero_shot_query_source),
                                    "wave_zero_shot_queries": list(wave_zero_shot_queries),
                                    "wave_zero_shot_query_rows": list(wave_zero_shot_query_rows),
                                    "wave_zero_shot_last_eval": wave_zero_shot_last_eval,
                                    "wave_zero_shot_history": list(wave_zero_shot_history),
                                    "accepted_library_count": int(library_serial),
                                    "sample_bits": int(sample_bits),
                                    "chunk_samples": int(chunk_samples),
                                    "patch_size": int(args.patch_size),
                                    "timestamp": time.time(),
                                }
                            ),
                        )
                    if _poll_gui_stop(note=f"cycle={int(cycle_id)} round={int(round_id)} stage=W"):
                        stop_orchestration = True
                        break
                if stop_orchestration:
                    break

        if _poll_gui_stop(note="final-eval"):
            final_eval = {
                "score_before": 0.0,
                "score_after": 0.0,
                "score_gain": 0.0,
                "hard_coverage_before": 0.0,
                "hard_coverage_after": 0.0,
                "stopped_early": True,
            }
            _log("Transformer final evaluation skipped due to GUI close request.")
        else:
            _offload_inactive_stage_models(reason="final-transformer-eval")
            final_eval, final_eval_batch = _evaluate_transformer_stage_with_retry(
                stage_tag="FINAL",
                batch_size_hint=int(_resolve_transformer_stage_train_batch("R")),
                max_batches=20,
            )
            _log(
                f"Transformer final: before={final_eval['score_before']:.4f} "
                f"after={final_eval['score_after']:.4f} gain={final_eval['score_gain']:.4f} "
                f"(eval_batch={int(final_eval_batch)})"
            )

        final_last_cycle = int(cycle_offset)
        final_last_round = 0
        if len(orchestration_rows) > 0 and isinstance(orchestration_rows[-1], dict):
            last_row = orchestration_rows[-1]
            try:
                final_last_cycle = int(last_row.get("cycle", final_last_cycle))
            except Exception:
                pass
            try:
                final_last_round = int(last_row.get("round", final_last_round))
            except Exception:
                pass

        _save_pipeline_checkpoint(
            out_dir / "pipeline_checkpoint.pt",
            _decorate_checkpoint_payload(
                {
                    "run_tag": run_tag,
                    "objective_mode": args.objective_mode,
                    "orchestration_mode": mode,
                    "last_cycle": int(final_last_cycle if mode != "none" else cycle_offset),
                    "last_round": int(final_last_round if mode in ("alternating", "staged_cgr", "staged_cgrw") else 0),
                    "best_cfg": best_cfg.to_dict(),
                    "classifier_state": classifier.state_dict(),
                    "transformer_state": transformer.state_dict(),
                    "deskew_prefilter_state": _extract_deskew_prefilter_state(transformer),
                    "generator_state": (generator.state_dict() if generator is not None else None),
                    "discriminator_state": (discriminator.state_dict() if discriminator is not None else None),
                    "wave_classifier_state": (wave_classifier.state_dict() if wave_classifier is not None else None),
                    "classifier_init_info": classifier_init_info,
                    "classifier_resume_info": classifier_resume_info,
                    "transformer_resume_info": transformer_resume_info,
                    "deskew_prefilter_resume_info": deskew_prefilter_resume_info,
                    "generator_resume_info": generator_resume_info,
                    "discriminator_resume_info": discriminator_resume_info,
                    "wave_classifier_init_info": wave_classifier_init_info,
                    "wave_classifier_resume_info": wave_classifier_resume_info,
                    "transformer_history": transformer_hist,
                    "generator_history": generator_hist,
                    "wave_classifier_history": wave_classifier_hist,
                    "orchestration_history": orchestration_rows,
                    "refresh_history": refresh_rows,
                    "gate_history": gate_history,
                    "gate_status": {
                        "gestation_streak": int(gestation_gate_streak),
                        "berkeley_streak": int(berkeley_gate_streak),
                        "wave_streak": int(wave_gate_streak),
                        "generator_streak": int(generator_gate_streak),
                        "transformer_streak": int(transformer_gate_streak),
                    },
                    "wave_classifier_last_eval": wave_classifier_last_eval,
                    "accepted_library_count": int(library_serial),
                    "latent_berkeley_targeting": latent_targeting_info,
                    "transformer_eval": final_eval,
                    "gui_stop_requested": bool(gui_stop_requested),
                    "label_embedding_info": label_embedding_info,
                    "label_texts": label_texts,
                    "label_query_rows": label_query_rows,
                    "wave_label_embedding_info": split_label_embedding_info,
                    "wave_label_texts": split_label_texts,
                    "wave_zero_shot_query_source": str(wave_zero_shot_query_source),
                    "wave_zero_shot_queries": list(wave_zero_shot_queries),
                    "wave_zero_shot_query_rows": list(wave_zero_shot_query_rows),
                    "wave_zero_shot_last_eval": wave_zero_shot_last_eval,
                    "wave_zero_shot_history": list(wave_zero_shot_history),
                    "sample_bits": int(sample_bits),
                    "chunk_samples": int(chunk_samples),
                    "patch_size": int(args.patch_size),
                    "timestamp": time.time(),
                }
            ),
        )

        classifier_path = out_dir / "classifier.pt"
        transformer_path = out_dir / "transformer.pt"
        deskew_prefilter_path = out_dir / "deskew_prefilter.pt"
        generator_path = out_dir / "generator.pt"
        discriminator_path = out_dir / "discriminator.pt"
        wave_classifier_path = out_dir / "wave_classifier.pt"
        label_embedding_bank_path = out_dir / "label_embedding_bank.pt"
        wave_label_embedding_bank_path = out_dir / "wave_label_embedding_bank.pt"
        cfg_path = out_dir / "best_render_config.json"
        summary_path = out_dir / "summary.json"
        search_path = out_dir / "search_results.json"
        orchestration_path = out_dir / "orchestration_history.json"

        torch.save(
            {
                "state_dict": classifier.state_dict(),
                "num_classes": num_classes,
                "class_names": class_names,
                "label_texts": label_texts,
                "label_embedding_info": label_embedding_info,
                "model_name": classifier_init_info.get("model_name", "tiny"),
            },
            classifier_path,
        )
        torch.save(
            {"state_dict": transformer.state_dict(), "chunk_samples": chunk_samples, "patch_size": int(args.patch_size)},
            transformer_path,
        )
        torch.save(
            {
                "state_dict": _extract_deskew_prefilter_state(transformer),
                "chunk_samples": int(chunk_samples),
                "patch_size": int(args.patch_size),
                "source_transformer": str(transformer_path),
            },
            deskew_prefilter_path,
        )
        if generator is not None:
            torch.save(
                {
                    "state_dict": generator.state_dict(),
                    "num_classes": int(condition_num_classes),
                    "z_dim": max(8, int(args.generator_z_dim)),
                    "image_hw": list(image_hw),
                },
                generator_path,
            )
        if discriminator is not None:
            torch.save(
                {
                    "state_dict": discriminator.state_dict(),
                    "num_classes": int(condition_num_classes),
                },
                discriminator_path,
            )
        final_gd_library_save_info: Dict[str, Any] = {"saved": False, "reason": "gd_library_disabled"}
        if bool(args.gd_vocab_library_enabled):
            final_gd_library_save_info = _save_gd_vocab_library_snapshot(
                library_dir=gd_vocab_library_dir,
                vocab_hash=str(active_gd_vocab_hash),
                condition_num_classes=int(condition_num_classes),
                generator=generator,
                discriminator=discriminator,
                meta={
                    "run_tag": str(run_tag),
                    "source": "final_artifacts",
                    "cycle": int(final_last_cycle),
                    "round": int(final_last_round),
                    "active_extra_terms": list(active_extra_terms),
                    "semantic_churn_history_size": int(len(semantic_churn_history)),
                    "vocab_profile": dict(active_gd_vocab_profile),
                },
            )
            if bool(final_gd_library_save_info.get("saved", False)):
                _log(
                    "[gd-vocab-library] saved final snapshot: "
                    f"hash={str(active_gd_vocab_hash)} dir={final_gd_library_save_info.get('dir', '')}"
                )
        if wave_classifier is not None:
            torch.save(
                {
                    "state_dict": wave_classifier.state_dict(),
                    "num_classes": split_num_classes,
                    "class_names": split_class_names,
                    "label_texts": split_label_texts,
                    "label_embedding_info": split_label_embedding_info,
                    "model_name": "tiny",
                    "init_info": wave_classifier_init_info,
                },
                wave_classifier_path,
            )
        if label_embedding_bank is not None:
            torch.save(
                {
                    "embeddings": torch.from_numpy(np.asarray(label_embedding_bank, dtype=np.float32)),
                    "class_names": list(class_names),
                    "label_texts": list(label_texts),
                    "info": dict(label_embedding_info),
                    "queries": list(label_query_rows),
                },
                label_embedding_bank_path,
            )
        if split_label_embedding_bank is not None:
            torch.save(
                {
                    "embeddings": torch.from_numpy(np.asarray(split_label_embedding_bank, dtype=np.float32)),
                    "class_names": list(split_class_names),
                    "label_texts": list(split_label_texts),
                    "info": dict(split_label_embedding_info),
                },
                wave_label_embedding_bank_path,
            )

        cfg_path.write_text(json.dumps(best_cfg.to_dict(), indent=2), encoding="utf-8")
        search_path.write_text(json.dumps(search_rows, indent=2), encoding="utf-8")
        if len(orchestration_rows) > 0:
            orchestration_path.write_text(json.dumps(orchestration_rows, indent=2), encoding="utf-8")

        preview_scores_path = out_dir / "previews" / "preview_label_scores.json"
        if args.save_preview > 0:
            preview_dir = out_dir / "previews"
            preview_dir.mkdir(parents=True, exist_ok=True)
            transformer.eval()
            classifier.eval()
            preview_scores: List[Dict] = []
            with torch.no_grad():
                for i in range(min(int(args.save_preview), len(val_streams))):
                    s = val_streams[i]
                    x_np = _fit_wave_length(
                        x=np.asarray(s, dtype=np.float32),
                        target_samples=int(chunk_samples),
                        rng=rng,
                    )
                    xb = torch.from_numpy(x_np[None, :]).to(device)
                    xh_t = transformer(xb)
                    xh = xh_t.detach().cpu().numpy()[0]
                    _write_mono_wav(str(preview_dir / f"sample_{i:02d}_original.wav"), x_np, framerate=val_meta[i]["framerate"])
                    _write_mono_wav(str(preview_dir / f"sample_{i:02d}_enhanced.wav"), xh, framerate=val_meta[i]["framerate"])
                    img_in = render_mono_wave_to_tensor(
                        xb,
                        cfg=best_cfg,
                        image_hw=image_hw,
                        sample_bits=int(sample_bits),
                    )
                    img_out = render_mono_wave_to_tensor(
                        xh_t,
                        cfg=best_cfg,
                        image_hw=image_hw,
                        sample_bits=int(sample_bits),
                    )
                    if args.channels_last:
                        img_in = img_in.contiguous(memory_format=torch.channels_last)
                        img_out = img_out.contiguous(memory_format=torch.channels_last)
                    logits_in = classifier(img_in)
                    logits_out = classifier(img_out)
                    probs_in = torch.sigmoid(logits_in[0]).detach().cpu().numpy().astype(np.float32, copy=False)
                    probs_out = torch.sigmoid(logits_out[0]).detach().cpu().numpy().astype(np.float32, copy=False)
                    topk = max(1, min(int(probs_out.shape[0]), len(class_names)))
                    top_idx = np.argsort(-probs_out)[:topk]
                    label_rows: List[Dict] = []
                    for cls_idx in top_idx.tolist():
                        c = int(cls_idx)
                        name = str(class_names[c]) if c < len(class_names) else f"class_{c}"
                        label_rows.append(
                            {
                                "class_idx": c,
                                "class_name": name,
                                "clean_score": float(probs_in[c]),
                                "enhanced_score": float(probs_out[c]),
                                "delta": float(probs_out[c] - probs_in[c]),
                            }
                        )
                    target_semantic = []
                    if val_stream_target_labels is not None and i < len(val_stream_target_labels):
                        tv = val_stream_target_labels[i]
                        if tv is not None:
                            target_semantic = _semantic_target_entries_from_vector(
                                tv,
                                class_names=class_names,
                                max_items=max(1, len(class_names)),
                                threshold=0.5,
                            )
                    preview_scores.append(
                        {
                            "sample": int(i),
                            "stream_meta_path": str(val_meta[i].get("path", "")),
                            "target_semantic": target_semantic,
                            "labels": label_rows,
                        }
                    )
                    label_txt = ", ".join(
                        [
                            f"{row['class_name']}:{row['clean_score']:.3f}->{row['enhanced_score']:.3f}"
                            for row in label_rows[:4]
                        ]
                    )
                    _log(f"Preview sample {i:02d} label scores: {label_txt}")
            preview_scores_path.write_text(json.dumps(preview_scores, indent=2), encoding="utf-8")
            _log(f"Preview label-score report: {preview_scores_path}")

        wave_zero_shot_report_path = out_dir / "wave_zero_shot_report.json"
        wave_zero_shot_report = {
            "query_source": str(wave_zero_shot_query_source),
            "queries": list(wave_zero_shot_queries),
            "query_rows": list(wave_zero_shot_query_rows),
            "last_eval": wave_zero_shot_last_eval,
            "history": list(wave_zero_shot_history),
        }
        wave_zero_shot_report_path.write_text(json.dumps(wave_zero_shot_report, indent=2), encoding="utf-8")

        summary = {
            "args": vars(args),
            "run_tag": run_tag,
            "resume_enabled": bool(resume_enabled),
            "resume_dir": str(resume_dir) if resume_enabled else "",
            "objective_mode": args.objective_mode,
            "orchestration_mode": mode,
            "wav_data_root": wav_data_root,
            "latent_fallback": latent_fallback_info,
            "latent_berkeley_targeting": latent_targeting_info,
            "num_records": len(records),
            "num_classes": num_classes,
            "condition_num_classes": int(condition_num_classes),
            "class_names": class_names,
            "label_texts": label_texts,
            "label_embedding_info": label_embedding_info,
            "label_query_rows": label_query_rows,
            "semantic_vocab_runtime": _semantic_vocab_runtime_snapshot(),
            "semantic_churn_history": list(semantic_churn_history),
            "active_gd_vocab_hash": str(active_gd_vocab_hash),
            "active_gd_vocab_profile": dict(active_gd_vocab_profile),
            "gd_vocab_library_last_save": dict(final_gd_library_save_info),
            "fake_feedback": {
                "enabled": bool(fake_feedback_enabled),
                "text": str(fake_sentinel_label),
                "vector_dim": (int(fake_label_vector_t.numel()) if fake_label_vector_t is not None else 0),
                "include_condition_targets": bool(args.generator_fake_feedback_include_condition_targets),
                "vector_weight": float(args.generator_fake_feedback_vector_weight),
                "condition_weight": float(args.generator_fake_feedback_condition_weight),
            },
            "wave_label_num_classes": split_num_classes,
            "wave_label_class_names": split_class_names,
            "wave_label_texts": split_label_texts,
            "wave_label_embedding_info": split_label_embedding_info,
            "wave_zero_shot_query_source": str(wave_zero_shot_query_source),
            "wave_zero_shot_queries": list(wave_zero_shot_queries),
            "wave_zero_shot_query_rows": list(wave_zero_shot_query_rows),
            "wave_zero_shot_last_eval": wave_zero_shot_last_eval,
            "wave_zero_shot_history": list(wave_zero_shot_history),
            "label_source_for_split": label_source,
            "train_count": len(train_idx),
            "val_count": len(val_idx),
            "best_cfg": best_cfg.to_dict(),
            "best_search_score": best_score,
            "classifier_init": classifier_init_info,
            "classifier_resume": classifier_resume_info,
            "classifier_init_trials": [{"mode": "berkeley_multilabel"}],
            "classifier_val": cls_val,
            "berkeley_refresh_history": refresh_rows,
            "gate_config": gate_config,
            "gate_history": gate_history,
            "gate_status": {
                "gestation_streak": int(gestation_gate_streak),
                "berkeley_streak": int(berkeley_gate_streak),
                "wave_streak": int(wave_gate_streak),
                "generator_streak": int(generator_gate_streak),
                "transformer_streak": int(transformer_gate_streak),
            },
            "transformer_baseline": baseline,
            "transformer_eval": final_eval,
            "transformer_history": transformer_hist,
            "transformer_resume": transformer_resume_info,
            "deskew_prefilter_resume": deskew_prefilter_resume_info,
            "generator_resume": generator_resume_info,
            "discriminator_resume": discriminator_resume_info,
            "generator_history": generator_hist,
            "orchestration_history": orchestration_rows,
            "classifier_history": wave_classifier_hist,
            "wave_classifier_init": wave_classifier_init_info,
            "wave_classifier_resume": wave_classifier_resume_info,
            "wave_classifier_history": wave_classifier_hist,
            "wave_classifier_last_eval": wave_classifier_last_eval,
            "accepted_library_count": int(library_serial),
            "accepted_library_index": str(library_index_path) if library_index_path.exists() else "",
            "gui_stop_requested": bool(gui_stop_requested),
            "elapsed_sec": time.time() - t0,
            "artifacts": {
                "classifier": str(classifier_path),
                "transformer": str(transformer_path),
                "deskew_prefilter": str(deskew_prefilter_path) if deskew_prefilter_path.exists() else "",
                "generator": str(generator_path) if generator is not None else "",
                "discriminator": str(discriminator_path) if discriminator is not None else "",
                "wave_classifier": str(wave_classifier_path) if wave_classifier is not None else "",
                "label_embedding_bank": str(label_embedding_bank_path) if label_embedding_bank_path.exists() else "",
                "wave_label_embedding_bank": str(wave_label_embedding_bank_path) if wave_label_embedding_bank_path.exists() else "",
                "config": str(cfg_path),
                "search_results": str(search_path),
                "orchestration_history": str(orchestration_path) if orchestration_path.exists() else "",
                "accepted_library_index": str(library_index_path) if library_index_path.exists() else "",
                "latent_manifest": str(latent_fallback_info["manifest"]) if latent_fallback_info else "",
                "preview_label_scores": str(preview_scores_path) if preview_scores_path.exists() else "",
                "wave_zero_shot_report": str(wave_zero_shot_report_path),
                "training_supervision_dir": str(training_preview_root) if training_preview_root.exists() else "",
                "training_supervision_manifest": str(training_preview_manifest_path) if training_preview_manifest_path.exists() else "",
                "pipeline_checkpoint": str(out_dir / "pipeline_checkpoint.pt"),
                "gd_vocab_library_dir": str(gd_vocab_library_dir),
            },
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        try:
            stage_opengl_viewer.close()
        except Exception:
            pass
        _log(f"Finished. Summary: {summary_path}")
        if bool(gui_stop_requested):
            _log(f"GUI stop requested; exiting with code {int(GUI_STOP_EXIT_CODE)}.")
            return int(GUI_STOP_EXIT_CODE)
        return 0

    label_mode = "folder" if args.label_mode == "folder" else "spectral"
    labels, class_names, label_source = _build_labels(
        records=records,
        data_root=wav_data_root,
        label_mode=label_mode,
        pseudo_classes=args.pseudo_classes,
    )
    num_classes = len(set(labels.tolist()))
    if num_classes < 2:
        raise RuntimeError("Need at least 2 classes for classifier training.")
    _log(f"Labels: {label_source}, classes={num_classes}")
    label_embedding_bank = None
    label_texts = [str(x) for x in class_names]
    label_embedding_info: Dict[str, Any] = {
        "enabled": False,
        "backend_requested": str(args.label_embedding_backend),
        "backend_used": "disabled",
        "model_name": str(args.label_embedding_model),
        "num_classes": int(num_classes),
        "native_dim": 0,
        "dim": 0,
        "temperature": float(args.label_embedding_temperature),
        "fallback_reason": "",
    }
    label_query_rows: List[Dict[str, Any]] = []
    if bool(args.label_embeddings_enabled):
        label_embedding_bank, label_texts, label_embedding_info = _build_label_embedding_bank(
            class_names=class_names,
            args=args,
            device=device,
        )
        _log(
            "[label-embeddings] bank ready: "
            f"classes={int(label_embedding_info.get('num_classes', 0))} "
            f"dim={int(label_embedding_info.get('dim', 0))} "
            f"backend={label_embedding_info.get('backend_used', 'unknown')}"
        )
        query_texts = _parse_label_query_texts(str(args.label_query_texts))
        if len(query_texts) > 0:
            q_emb = _encode_query_texts_for_bank(
                query_texts=query_texts,
                bank_info=label_embedding_info,
                args=args,
                device=device,
            )
            label_query_rows = _nearest_labels_for_queries(
                query_texts=query_texts,
                label_bank_np=label_embedding_bank,
                class_names=class_names,
                label_texts=label_texts,
                topk=max(1, int(args.label_query_topk)),
                query_emb_np=q_emb,
            )
            for row in label_query_rows:
                if len(row.get("matches", [])) <= 0:
                    continue
                best = row["matches"][0]
                _log(
                    f"[label-query] '{row.get('query', '')}' -> "
                    f"{best.get('class_name', '')} ({best.get('score', 0.0):.4f})"
                )

    train_idx, val_idx = _stratified_split(labels, train_frac=args.train_frac, seed=args.seed)
    _log(f"Split: train={len(train_idx)} val={len(val_idx)}")

    search_train_idx = _sample_indices(train_idx, args.search_train_limit, rng)
    search_val_idx = _sample_indices(val_idx, args.search_val_limit, rng)
    _log(f"Search subset: train={len(search_train_idx)} val={len(search_val_idx)}")

    image_hw = image_hw_shared

    candidates = _random_configs(num_trials=args.config_trials, rng=rng, max_points=args.render_max_points)
    best_cfg = None
    best_score = -1.0
    search_rows = []
    init_info_search = []
    resume_search_rows = _load_json(resume_search_path) if (resume_enabled and resume_search_path.exists()) else None
    did_config_search = False

    if bool(args.enforce_render_config):
        best_cfg = _build_enforced_render_config(args)
        best_score = 0.0
        search_rows = [{"trial": 0, "score": 0.0, "forced": True, "cfg": best_cfg.to_dict()}]
        _log(f"Render config enforced by CLI: {best_cfg.to_dict()}")
    elif resume_enabled and (not args.force_config_search):
        if isinstance(resume_cfg, dict):
            try:
                best_cfg = RenderConfig.from_dict(resume_cfg)
                _log(f"Resumed best render config from {resume_cfg_path}")
            except Exception:
                best_cfg = None
        if best_cfg is None and isinstance(resume_pipeline_ckpt, dict) and isinstance(resume_pipeline_ckpt.get("best_cfg"), dict):
            try:
                best_cfg = RenderConfig.from_dict(resume_pipeline_ckpt.get("best_cfg"))
                _log("Resumed best render config from pipeline checkpoint")
            except Exception:
                best_cfg = None
        if isinstance(resume_summary, dict):
            best_score = float(resume_summary.get("best_search_score", best_score))
            if isinstance(resume_summary.get("classifier_init_trials"), list):
                init_info_search = list(resume_summary.get("classifier_init_trials", init_info_search))
        if isinstance(resume_search_rows, list):
            search_rows = list(resume_search_rows)

    if best_cfg is None:
        did_config_search = True
        _log("Config search started...")
        for i, cfg in enumerate(candidates, start=1):
            try:
                xtr, ytr, _ = _render_indices_to_tensors(records, labels, search_train_idx, cfg, image_hw=image_hw)
                xva, yva, _ = _render_indices_to_tensors(records, labels, search_val_idx, cfg, image_hw=image_hw)
                model = TinyConvClassifier(
                    num_classes=num_classes,
                    base_ch=int(args.classifier_base_ch),
                    max_ch=int(args.classifier_max_ch),
                    context_blocks=int(args.classifier_context_blocks),
                    context_dropout=float(args.classifier_context_dropout),
                )
                init_info = _apply_classifier_init(model, args.classifier_init_ckpt, args.classifier_init_scope)
                _apply_label_embedding_bank_to_classifier(
                    classifier=model,
                    bank_np=label_embedding_bank,
                    args=args,
                )
                if args.classifier_freeze_features:
                    _set_feature_freeze(model, freeze=True)
                model, _ = train_classifier(
                    model=model,
                    x_train=xtr,
                    y_train=ytr,
                    x_val=xva,
                    y_val=yva,
                    device=device,
                    epochs=args.config_epochs,
                    batch_size=args.classifier_batch_size,
                    lr=args.classifier_lr,
                    lr_sine_cycles=args.lr_sine_cycles,
                    lr_sine_frequency=args.lr_sine_frequency,
                    lr_sine_tail_fraction=args.lr_sine_tail_fraction,
                    lr_sine_min_scale=args.lr_sine_min_scale,
                    amp=amp_enabled,
                    amp_dtype=args.amp_dtype,
                    channels_last=bool(args.channels_last),
                    grad_accum_steps=grad_accum_steps,
                    seed=args.seed + i,
                )
                val_stats = evaluate_classifier(
                    model,
                    xva,
                    yva,
                    batch_size=args.classifier_batch_size,
                    device=device,
                    amp=amp_enabled,
                    amp_dtype=args.amp_dtype,
                    channels_last=bool(args.channels_last),
                )
                score = float(val_stats["acc"])
                row = {"trial": i, "score": score, "cfg": cfg.to_dict(), "val_loss": float(val_stats["loss"])}
                search_rows.append(row)
                init_info_search.append({"trial": i, **init_info})
                _log(f"  [{i}/{len(candidates)}] val_acc={score:.4f} cfg={cfg.to_dict()}")
                if score > best_score:
                    best_score = score
                    best_cfg = cfg
            except Exception as e:
                _log(f"  [{i}/{len(candidates)}] failed: {e}")
    else:
        if bool(args.enforce_render_config):
            _log("Skipping config search because --enforce-render-config is active.")
        else:
            _log("Skipping config search due to resume config (use --force-config-search to override).")

    if best_cfg is None:
        raise RuntimeError("Config search failed to produce a usable config.")

    if args.try_scipy_refine and did_config_search:
        refine_candidates = _try_scipy_refine(best_cfg, max_points=args.render_max_points, enabled=True)
        if len(refine_candidates) > 1:
            _log("Running optional refinement candidates...")
            for j, cfg in enumerate(refine_candidates[1:], start=1):
                try:
                    xtr, ytr, _ = _render_indices_to_tensors(records, labels, search_train_idx, cfg, image_hw=image_hw)
                    xva, yva, _ = _render_indices_to_tensors(records, labels, search_val_idx, cfg, image_hw=image_hw)
                    model = TinyConvClassifier(
                        num_classes=num_classes,
                        base_ch=int(args.classifier_base_ch),
                        max_ch=int(args.classifier_max_ch),
                        context_blocks=int(args.classifier_context_blocks),
                        context_dropout=float(args.classifier_context_dropout),
                    )
                    init_info = _apply_classifier_init(model, args.classifier_init_ckpt, args.classifier_init_scope)
                    _apply_label_embedding_bank_to_classifier(
                        classifier=model,
                        bank_np=label_embedding_bank,
                        args=args,
                    )
                    if args.classifier_freeze_features:
                        _set_feature_freeze(model, freeze=True)
                    model, _ = train_classifier(
                        model=model,
                        x_train=xtr,
                        y_train=ytr,
                        x_val=xva,
                        y_val=yva,
                        device=device,
                        epochs=max(1, args.config_epochs),
                        batch_size=args.classifier_batch_size,
                        lr=args.classifier_lr,
                        lr_sine_cycles=args.lr_sine_cycles,
                        lr_sine_frequency=args.lr_sine_frequency,
                        lr_sine_tail_fraction=args.lr_sine_tail_fraction,
                        lr_sine_min_scale=args.lr_sine_min_scale,
                        amp=amp_enabled,
                        amp_dtype=args.amp_dtype,
                        channels_last=bool(args.channels_last),
                        grad_accum_steps=grad_accum_steps,
                        seed=args.seed + 999 + j,
                    )
                    val_stats = evaluate_classifier(
                        model,
                        xva,
                        yva,
                        batch_size=args.classifier_batch_size,
                        device=device,
                        amp=amp_enabled,
                        amp_dtype=args.amp_dtype,
                        channels_last=bool(args.channels_last),
                    )
                    score = float(val_stats["acc"])
                    search_rows.append({"trial": len(search_rows) + 1, "score": score, "cfg": cfg.to_dict()})
                    init_info_search.append({"trial": len(search_rows), **init_info})
                    _log(f"  [refine {j}] val_acc={score:.4f} cfg={cfg.to_dict()}")
                    if score > best_score:
                        best_score = score
                        best_cfg = cfg
                except Exception as e:
                    _log(f"  [refine {j}] failed: {e}")

    _log(f"Best cfg val_acc={best_score:.4f}: {best_cfg.to_dict()}")
    if _sync_render_width_with_embed_hw(best_cfg, image_hw=image_hw):
        _log(
            "Embed resolution sync: adjusted render width to shared embed width "
            f"(render_width={int(best_cfg.width)}, embed_w={int(image_hw[1])})"
        )

    final_train_idx = _sample_indices(train_idx, args.final_train_limit, rng)
    final_val_idx = _sample_indices(val_idx, args.final_val_limit, rng)
    _log(f"Final classifier subset: train={len(final_train_idx)} val={len(final_val_idx)}")

    x_train, y_train, bits_train = _render_indices_to_tensors(records, labels, final_train_idx, best_cfg, image_hw=image_hw)
    x_val, y_val, bits_val = _render_indices_to_tensors(records, labels, final_val_idx, best_cfg, image_hw=image_hw)
    bits_all = bits_train + bits_val
    sample_bits = int(Counter(bits_all).most_common(1)[0][0]) if len(bits_all) > 0 else 16
    _log(f"Sample bits for transformer renderer: {sample_bits}")

    classifier = TinyConvClassifier(
        num_classes=num_classes,
        base_ch=int(args.classifier_base_ch),
        max_ch=int(args.classifier_max_ch),
        context_blocks=int(args.classifier_context_blocks),
        context_dropout=float(args.classifier_context_dropout),
    )
    classifier_init_info = _apply_classifier_init(classifier, args.classifier_init_ckpt, args.classifier_init_scope)
    classifier_resume_info = {"used": False}
    if resume_enabled and resume_classifier_path.exists():
        classifier_resume_info = _apply_model_init(classifier, str(resume_classifier_path))
        _log(
            f"Resumed classifier from {resume_classifier_path} "
            f"(loaded={classifier_resume_info.get('loaded_keys', 0)}, skipped={classifier_resume_info.get('skipped_keys', 0)})"
        )
    elif isinstance(resume_pipeline_ckpt, dict) and isinstance(resume_pipeline_ckpt.get("classifier_state"), dict):
        classifier_resume_info = _apply_state_dict(
            classifier,
            resume_pipeline_ckpt.get("classifier_state"),
            source_name="pipeline_checkpoint",
        )
        _log(
            "Resumed classifier from pipeline checkpoint "
            f"(loaded={classifier_resume_info.get('loaded_keys', 0)}, skipped={classifier_resume_info.get('skipped_keys', 0)})"
        )
    if classifier_init_info["used"]:
        _log(
            f"Classifier init loaded {classifier_init_info['loaded_keys']} keys "
            f"(scope={classifier_init_info['scope']}, skipped={classifier_init_info['skipped_keys']})."
        )
    label_embed_apply_info = _apply_label_embedding_bank_to_classifier(
        classifier=classifier,
        bank_np=label_embedding_bank,
        args=args,
    )
    if bool(label_embed_apply_info.get("applied", False)):
        _log(
            "[label-embeddings] classifier active: "
            f"classes={label_embed_apply_info.get('classes', 0)} "
            f"dim={label_embed_apply_info.get('dim', 0)} "
            f"temp={float(label_embed_apply_info.get('temperature', 0.0)):.2f}"
        )
    if args.classifier_freeze_features:
        _set_feature_freeze(classifier, freeze=True)
        _log("Classifier feature extractor frozen.")
    if args.channels_last:
        classifier = classifier.to(memory_format=torch.channels_last)
    classifier = maybe_compile_module(
        classifier,
        enabled=bool(args.compile_models),
        mode=str(args.compile_mode),
    )
    classifier_hist: List[Dict[str, float]] = []
    if isinstance(resume_summary, dict) and isinstance(resume_summary.get("classifier_history"), list):
        classifier_hist.extend(list(resume_summary.get("classifier_history", [])))
    elif isinstance(resume_pipeline_ckpt, dict) and isinstance(resume_pipeline_ckpt.get("classifier_history"), list):
        classifier_hist.extend(list(resume_pipeline_ckpt.get("classifier_history", [])))

    classifier, run_classifier_hist = train_classifier(
        model=classifier,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        device=device,
        epochs=args.classifier_epochs,
        batch_size=args.classifier_batch_size,
        lr=args.classifier_lr,
        lr_sine_cycles=args.lr_sine_cycles,
        lr_sine_frequency=args.lr_sine_frequency,
        lr_sine_tail_fraction=args.lr_sine_tail_fraction,
        lr_sine_min_scale=args.lr_sine_min_scale,
        amp=amp_enabled,
        amp_dtype=args.amp_dtype,
        channels_last=bool(args.channels_last),
        grad_accum_steps=grad_accum_steps,
        seed=args.seed + 5000,
    )
    classifier_hist.extend(run_classifier_hist)
    _save_training_segment_snapshot(
        enabled=bool(args.checkpoint_after_training_segment),
        out_dir=out_dir,
        objective_mode=args.objective_mode,
        run_tag=run_tag,
        segment="final_classifier_train",
        best_cfg=best_cfg,
        classifier=classifier,
        classifier_history=classifier_hist,
        extra={"best_search_score": float(best_score)},
    )
    cls_val = evaluate_classifier(
        classifier,
        x_val,
        y_val,
        batch_size=args.classifier_batch_size,
        device=device,
        amp=amp_enabled,
        amp_dtype=args.amp_dtype,
        channels_last=bool(args.channels_last),
    )
    _log(f"Classifier val: acc={cls_val['acc']:.4f} loss={cls_val['loss']:.4f}")

    train_streams, train_stream_labels, _, train_meta = _prepare_streams(
        records=records,
        labels=labels,
        indices=final_train_idx,
        cfg=best_cfg,
        max_points=args.stream_max_points,
    )
    val_streams, val_stream_labels, _, val_meta = _prepare_streams(
        records=records,
        labels=labels,
        indices=final_val_idx,
        cfg=best_cfg,
        max_points=args.stream_max_points,
    )
    if len(train_streams) < 4 or len(val_streams) < 2:
        raise RuntimeError("Not enough decoded streams for transformer stage.")

    chunk_samples, chunk_sync_info = _resolve_synced_chunk_samples(
        requested_chunk_samples=int(args.chunk_samples),
        patch_size=int(args.patch_size),
        cfg=best_cfg,
        image_hw=image_hw,
    )
    _log(
        "Transformer chunk sync: "
        f"requested={chunk_sync_info['requested']} resolved={chunk_sync_info['resolved']} "
        f"patch={chunk_sync_info['patch_size']} render={chunk_sync_info['render_h']}x{chunk_sync_info['render_w']} "
        f"target={chunk_sync_info['target_h']}x{chunk_sync_info['target_w']} "
        f"downsample={chunk_sync_info['downsample']} special_stride_mode={1 if chunk_sync_info['special_stride_mode'] else 0} "
        f"resize_needed={1 if chunk_sync_info['needs_resize'] else 0}"
    )
    if bool(chunk_sync_info["needs_resize"]):
        raise RuntimeError(
            "Transformer chunk sync could not match embed resolution exactly "
            f"(render={chunk_sync_info['render_h']}x{chunk_sync_info['render_w']} "
            f"target={chunk_sync_info['target_h']}x{chunk_sync_info['target_w']}). "
            "Adjust --image-size, --enforce-render-width, --enforce-render-downsample, "
            "--enforce-render-max-points, or --patch-size."
        )

    train_streams, train_stream_labels, train_meta, _, train_len_info = _filter_stream_pool_for_chunk_samples(
        streams=train_streams,
        labels=train_stream_labels,
        metas=train_meta,
        chunk_samples=int(chunk_samples),
        pool_name="transformer_train_streams",
        target_labels=None,
    )
    val_streams, val_stream_labels, val_meta, _, val_len_info = _filter_stream_pool_for_chunk_samples(
        streams=val_streams,
        labels=val_stream_labels,
        metas=val_meta,
        chunk_samples=int(chunk_samples),
        pool_name="transformer_val_streams",
        target_labels=None,
    )
    _log(
        "Transformer stream-length filter: "
        f"train kept={train_len_info['after']}/{train_len_info['before']} dropped={train_len_info['dropped']} "
        f"val kept={val_len_info['after']}/{val_len_info['before']} dropped={val_len_info['dropped']} "
        f"required_chunk={train_len_info['required_chunk_samples']}"
    )
    if len(train_streams) < 4 or len(val_streams) < 2:
        raise RuntimeError(
            "Not enough long streams after enforcing chunk length: "
            f"train={len(train_streams)} val={len(val_streams)} required_chunk={int(chunk_samples)}. "
            "Provide longer WAVs or increase --latent-fallback-seconds."
        )

    transformer = WavePatchTransformer(
        chunk_samples=chunk_samples,
        patch_size=int(args.patch_size),
        d_model=int(args.transformer_d_model),
        nhead=int(args.transformer_nhead),
        num_layers=int(args.transformer_num_layers),
        ff_mult=int(args.transformer_ff_mult),
        max_delta=float(args.max_delta),
        dropout=float(args.transformer_dropout),
        prefilter_max_skew=float(args.transformer_deskew_prefilter_max_skew),
        prefilter_enabled=bool("deskew" in transformer_filter_bundle_names),
        aux_filter_bundle_names=transformer_filter_bundle_names,
    )
    transformer = transformer.to(device)
    transformer_resume_info = {"used": False}
    deskew_prefilter_resume_info = {"used": False}
    if resume_enabled and resume_transformer_path.exists():
        transformer_resume_info = _apply_model_init(transformer, str(resume_transformer_path))
        _log(
            f"Resumed transformer from {resume_transformer_path} "
            f"(loaded={transformer_resume_info.get('loaded_keys', 0)}, skipped={transformer_resume_info.get('skipped_keys', 0)})"
        )
    elif isinstance(resume_pipeline_ckpt, dict) and isinstance(resume_pipeline_ckpt.get("transformer_state"), dict):
        transformer_resume_info = _apply_state_dict(
            transformer,
            resume_pipeline_ckpt.get("transformer_state"),
            source_name="pipeline_checkpoint",
        )
        _log(
            "Resumed transformer from pipeline checkpoint "
            f"(loaded={transformer_resume_info.get('loaded_keys', 0)}, skipped={transformer_resume_info.get('skipped_keys', 0)})"
        )
    if resume_enabled and resume_deskew_prefilter_path.exists():
        deskew_prefilter_resume_info = _apply_model_init(transformer, str(resume_deskew_prefilter_path))
        _log(
            f"Resumed deskew prefilter from {resume_deskew_prefilter_path} "
            f"(loaded={deskew_prefilter_resume_info.get('loaded_keys', 0)}, "
            f"skipped={deskew_prefilter_resume_info.get('skipped_keys', 0)})"
        )
    elif isinstance(resume_pipeline_ckpt, dict) and isinstance(resume_pipeline_ckpt.get("deskew_prefilter_state"), dict):
        deskew_prefilter_resume_info = _apply_state_dict(
            transformer,
            resume_pipeline_ckpt.get("deskew_prefilter_state"),
            source_name="pipeline_checkpoint_deskew_prefilter",
        )
        _log(
            "Resumed deskew prefilter from pipeline checkpoint "
            f"(loaded={deskew_prefilter_resume_info.get('loaded_keys', 0)}, "
            f"skipped={deskew_prefilter_resume_info.get('skipped_keys', 0)})"
        )
    transformer = maybe_compile_module(
        transformer,
        enabled=bool(args.compile_models),
        mode=str(args.compile_mode),
    )

    baseline = evaluate_transformer_accuracy(
        transformer=transformer,
        classifier=classifier,
        streams=val_streams,
        labels=val_stream_labels,
        cfg=best_cfg,
        sample_bits=sample_bits,
        image_hw=image_hw,
        chunk_samples=chunk_samples,
        device=device,
        max_batches=16,
        batch_size=args.transformer_batch_size,
        amp=amp_enabled,
        amp_dtype=args.amp_dtype,
        channels_last=bool(args.channels_last),
        pin_memory=bool(args.pin_memory_wave_batches),
        stream_cache_on_device=bool(args.cache_wave_streams_on_device),
    )
    _log(f"Transformer baseline val: before={baseline['acc_before']:.4f} after={baseline['acc_after']:.4f}")

    transformer_hist: List[Dict[str, float]] = []
    if isinstance(resume_summary, dict) and isinstance(resume_summary.get("transformer_history"), list):
        transformer_hist.extend(list(resume_summary.get("transformer_history", [])))
    elif isinstance(resume_pipeline_ckpt, dict) and isinstance(resume_pipeline_ckpt.get("transformer_history"), list):
        transformer_hist.extend(list(resume_pipeline_ckpt.get("transformer_history", [])))

    transformer, run_transformer_hist = train_transformer_feature_metric(
        transformer=transformer,
        classifier=classifier,
        train_streams=train_streams,
        train_target_labels=None,
        val_streams=val_streams,
        cfg=best_cfg,
        sample_bits=sample_bits,
        image_hw=image_hw,
        device=device,
        epochs=args.transformer_epochs,
        steps_per_epoch=args.transformer_steps,
        batch_size=args.transformer_batch_size,
        chunk_samples=chunk_samples,
        lr=args.transformer_lr,
        lr_sine_cycles=args.lr_sine_cycles,
        lr_sine_frequency=args.lr_sine_frequency,
        lr_sine_tail_fraction=args.lr_sine_tail_fraction,
        lr_sine_min_scale=args.lr_sine_min_scale,
        score_topk=args.score_topk,
        score_threshold=args.score_threshold,
        score_w_topk=args.score_w_topk,
        score_w_cov=args.score_w_cov,
        score_w_mean=args.score_w_mean,
        degrade_inputs=bool(args.transformer_degrade_inputs),
        degrade_min_strength=float(args.transformer_degrade_min_strength),
        degrade_max_strength=float(args.transformer_degrade_max_strength),
        degrade_noise_std_min=float(args.transformer_degrade_noise_std_min),
        degrade_noise_std_max=float(args.transformer_degrade_noise_std_max),
        degrade_dropout_max=float(args.transformer_degrade_dropout_max),
        degrade_quant_bits_min=int(args.transformer_degrade_quant_bits_min),
        degrade_quant_bits_max=int(args.transformer_degrade_quant_bits_max),
        degrade_vectorized_window=max(1, int(args.transformer_degrade_vectorized_window)),
        degrade_mode=str(args.transformer_degrade_mode),
        degrade_domain=str(args.transformer_degrade_domain),
        degrade_stride_skew_max=float(args.transformer_degrade_stride_skew_max),
        deskew_prefilter_weight=float(args.transformer_deskew_prefilter_weight),
        deskew_residual_weight=float(args.transformer_deskew_residual_weight),
        deskew_pre_token_mix=float(args.transformer_deskew_pre_token_mix),
        deskew_post_token_mix=float(args.transformer_deskew_post_token_mix),
        deskew_token_residual_weight=float(args.transformer_deskew_token_residual_weight),
        entropy_penalty_weight=float(args.transformer_loss_entropy_weight),
        high_bit_penalty_weight=float(args.transformer_loss_high_bit_weight),
        low_bit_penalty_weight=float(args.transformer_loss_low_bit_weight),
        wave_l1_weight=float(args.transformer_loss_wave_l1_weight),
        score_target_weight=float(args.transformer_loss_score_target_weight),
        score_target_margin=float(args.transformer_loss_score_target_margin),
        score_rank_weight=float(args.transformer_loss_score_rank_weight),
        score_rank_margin=float(args.transformer_loss_score_rank_margin),
        score_spurious_weight=float(args.transformer_loss_score_spurious_weight),
        score_spurious_threshold=float(args.transformer_loss_score_spurious_threshold),
        score_spurious_temp=float(args.transformer_loss_score_spurious_temp),
        score_spurious_margin=float(args.transformer_loss_score_spurious_margin),
        amp=amp_enabled,
        amp_dtype=args.amp_dtype,
        channels_last=bool(args.channels_last),
        grad_accum_steps=grad_accum_steps,
        pin_memory=bool(args.pin_memory_wave_batches),
        stream_cache_on_device=bool(args.cache_wave_streams_on_device),
        log_every_steps=max(0, int(args.transformer_log_every)),
        cache_eval_batches=bool(args.transformer_cache_eval_batches),
        visualize_status=bool(args.transformer_visualize_status),
        visualize_scale=max(1, int(args.transformer_viz_scale)),
        seed=args.seed + 9000,
    )
    transformer_hist.extend(run_transformer_hist)
    _save_training_segment_snapshot(
        enabled=bool(args.checkpoint_after_training_segment),
        out_dir=out_dir,
        objective_mode=args.objective_mode,
        run_tag=run_tag,
        segment="final_transformer_train",
        best_cfg=best_cfg,
        classifier=classifier,
        transformer=transformer,
        classifier_history=classifier_hist,
        transformer_history=transformer_hist,
        extra={
            "sample_bits": int(sample_bits),
            "chunk_samples": int(chunk_samples),
            "patch_size": int(args.patch_size),
            "best_search_score": float(best_score),
        },
    )

    final_eval = evaluate_transformer_accuracy(
        transformer=transformer,
        classifier=classifier,
        streams=val_streams,
        labels=val_stream_labels,
        cfg=best_cfg,
        sample_bits=sample_bits,
        image_hw=image_hw,
        chunk_samples=chunk_samples,
        device=device,
        max_batches=20,
        batch_size=args.transformer_batch_size,
        amp=amp_enabled,
        amp_dtype=args.amp_dtype,
        channels_last=bool(args.channels_last),
        pin_memory=bool(args.pin_memory_wave_batches),
        stream_cache_on_device=bool(args.cache_wave_streams_on_device),
    )
    _log(f"Transformer final val: before={final_eval['acc_before']:.4f} after={final_eval['acc_after']:.4f}")

    _save_pipeline_checkpoint(
        out_dir / "pipeline_checkpoint.pt",
        {
            "run_tag": run_tag,
            "objective_mode": args.objective_mode,
            "best_cfg": best_cfg.to_dict(),
            "classifier_state": classifier.state_dict(),
            "transformer_state": transformer.state_dict(),
            "deskew_prefilter_state": _extract_deskew_prefilter_state(transformer),
            "classifier_init_info": classifier_init_info,
            "classifier_resume_info": classifier_resume_info,
            "transformer_resume_info": transformer_resume_info,
            "deskew_prefilter_resume_info": deskew_prefilter_resume_info,
            "classifier_history": classifier_hist,
            "transformer_history": transformer_hist,
            "classifier_val": cls_val,
            "transformer_eval": final_eval,
            "label_embedding_info": label_embedding_info,
            "label_texts": label_texts,
            "label_query_rows": label_query_rows,
            "sample_bits": int(sample_bits),
            "chunk_samples": int(chunk_samples),
            "patch_size": int(args.patch_size),
            "timestamp": time.time(),
        },
    )

    classifier_path = out_dir / "classifier.pt"
    transformer_path = out_dir / "transformer.pt"
    deskew_prefilter_path = out_dir / "deskew_prefilter.pt"
    label_embedding_bank_path = out_dir / "label_embedding_bank.pt"
    cfg_path = out_dir / "best_render_config.json"
    summary_path = out_dir / "summary.json"
    search_path = out_dir / "search_results.json"

    torch.save(
        {
            "state_dict": classifier.state_dict(),
            "num_classes": num_classes,
            "class_names": class_names,
            "label_texts": label_texts,
            "label_embedding_info": label_embedding_info,
        },
        classifier_path,
    )
    torch.save({"state_dict": transformer.state_dict(), "chunk_samples": chunk_samples, "patch_size": int(args.patch_size)}, transformer_path)
    torch.save(
        {
            "state_dict": _extract_deskew_prefilter_state(transformer),
            "chunk_samples": int(chunk_samples),
            "patch_size": int(args.patch_size),
            "source_transformer": str(transformer_path),
        },
        deskew_prefilter_path,
    )
    if label_embedding_bank is not None:
        torch.save(
            {
                "embeddings": torch.from_numpy(np.asarray(label_embedding_bank, dtype=np.float32)),
                "class_names": list(class_names),
                "label_texts": list(label_texts),
                "info": dict(label_embedding_info),
                "queries": list(label_query_rows),
            },
            label_embedding_bank_path,
        )
    cfg_path.write_text(json.dumps(best_cfg.to_dict(), indent=2), encoding="utf-8")
    search_path.write_text(json.dumps(search_rows, indent=2), encoding="utf-8")

    if args.save_preview > 0:
        preview_dir = out_dir / "previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        transformer.eval()
        with torch.no_grad():
            for i in range(min(int(args.save_preview), len(val_streams))):
                s = val_streams[i]
                x_np = _fit_wave_length(
                    x=np.asarray(s, dtype=np.float32),
                    target_samples=int(chunk_samples),
                    rng=rng,
                )
                xb = torch.from_numpy(x_np[None, :]).to(device)
                xh = transformer(xb).detach().cpu().numpy()[0]
                _write_mono_wav(str(preview_dir / f"sample_{i:02d}_original.wav"), x_np, framerate=val_meta[i]["framerate"])
                _write_mono_wav(str(preview_dir / f"sample_{i:02d}_enhanced.wav"), xh, framerate=val_meta[i]["framerate"])

    summary = {
        "args": vars(args),
        "run_tag": run_tag,
        "resume_enabled": bool(resume_enabled),
        "resume_dir": str(resume_dir) if resume_enabled else "",
        "wav_data_root": wav_data_root,
        "latent_fallback": latent_fallback_info,
        "num_records": len(records),
        "num_classes": num_classes,
        "class_names": class_names,
        "label_texts": label_texts,
        "label_embedding_info": label_embedding_info,
        "label_query_rows": label_query_rows,
        "label_source": label_source,
        "train_count": len(train_idx),
        "val_count": len(val_idx),
        "best_cfg": best_cfg.to_dict(),
        "best_search_score": best_score,
        "classifier_init": classifier_init_info,
        "classifier_resume": classifier_resume_info,
        "classifier_init_trials": init_info_search,
        "classifier_val": cls_val,
        "transformer_eval": final_eval,
        "classifier_history": classifier_hist,
        "transformer_history": transformer_hist,
        "transformer_resume": transformer_resume_info,
        "deskew_prefilter_resume": deskew_prefilter_resume_info,
        "elapsed_sec": time.time() - t0,
        "artifacts": {
            "classifier": str(classifier_path),
            "transformer": str(transformer_path),
            "deskew_prefilter": str(deskew_prefilter_path) if deskew_prefilter_path.exists() else "",
            "label_embedding_bank": str(label_embedding_bank_path) if label_embedding_bank_path.exists() else "",
            "config": str(cfg_path),
            "search_results": str(search_path),
            "latent_manifest": str(latent_fallback_info["manifest"]) if latent_fallback_info else "",
            "pipeline_checkpoint": str(out_dir / "pipeline_checkpoint.pt"),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log(f"Finished. Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(int(main() or 0))

