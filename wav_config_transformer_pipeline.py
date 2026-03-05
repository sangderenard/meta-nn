import argparse
import gc
import hashlib
import json
import math
import os
import re
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
from torch.utils.data import DataLoader

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
        out.extend(["noise"])
    elif key.endswith("noise"):
        out.extend([key, "noise"])
    elif key in ("mixed noise and signal", "mix"):
        out.extend(["mixed noise and signal", "signal", "noise"])
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
        out.extend(["regurgitated content", "mixed noise and signal", "signal"])
    elif key == "gan image":
        out.extend(["gan image", "signal"])
    elif key.endswith("damage"):
        out.extend([key, "signal"])
    else:
        out.extend(["signal"])
    out.append(key)
    return _normalize_vocab_terms(out)


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
                dst.append(np.asarray(row, dtype=np.float32, copy=False))
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
        if int(t.ndim) == 3 and int(t.shape[0]) in (1, 3):
            t = t.unsqueeze(0)
            t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
            t = torch.clamp(t, 0.0, 1.0).squeeze(0)
            if int(t.shape[0]) == 1:
                t = t.repeat(3, 1, 1)
            return t.numpy().astype(np.float32, copy=False)
        if int(t.ndim) == 2:
            arr = t.numpy().astype(np.float32, copy=False)
    if arr is None:
        if hasattr(img, "convert"):
            arr = np.asarray(img.convert("L"), dtype=np.float32)
        else:
            arr = np.asarray(img, dtype=np.float32)
    if int(arr.ndim) == 3:
        if int(arr.shape[2]) >= 3:
            arr = arr[:, :, :3].mean(axis=2)
        else:
            arr = arr[:, :, 0]
    if int(arr.ndim) != 2:
        raise RuntimeError(f"Unsupported symbol image shape for conversion: {tuple(arr.shape)}")
    if float(np.max(arr)) > 1.0:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)
    t = torch.from_numpy(arr[None, None, ...]).to(torch.float32)
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    g = torch.clamp(t[0, 0], 0.0, 1.0).cpu().numpy().astype(np.float32, copy=False)
    return np.repeat(g[None, :, :], 3, axis=0).astype(np.float32, copy=False)


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
        t = torch.from_numpy(np.asarray(g, dtype=np.float32, copy=False)[None, None, ...])
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
    cap = max(1, int(max_samples_per_term))
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
        rows.append(_to_rgb(_resize_gray(img)))

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
        t = torch.from_numpy(np.asarray(g, dtype=np.float32, copy=False)[None, None, ...])
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
            return np.asarray(color_patterns[key], dtype=np.float32, copy=False)
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
            return np.clip((0.55 * _signal_pattern(phase=phase)) + (0.45 * rng.random((size, size), dtype=np.float32)), 0.0, 1.0)
        if key == "blur damage":
            return _box_blur(_signal_pattern(phase=phase), k=7)
        if key == "noise damage":
            return np.clip(_signal_pattern(phase=phase) + (0.20 * rng.standard_normal((size, size), dtype=np.float32)), 0.0, 1.0)
        if key == "dropout damage":
            src = _signal_pattern(phase=phase)
            keep = (rng.random((size, size), dtype=np.float32) > 0.20).astype(np.float32, copy=False)
            return np.clip(src * keep, 0.0, 1.0)
        if key == "quantization damage":
            return _quantize(_signal_pattern(phase=phase), levels=4)
        if key == "stride skew damage":
            src = _signal_pattern(phase=phase)
            out = np.array(src, dtype=np.float32, copy=True)
            out[1::2, :] = np.roll(out[1::2, :], shift=3, axis=1)
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
    extra_terms = _normalize_vocab_terms(["spectrographic output", "inverse spectrographic composition", "pure tone construction"])
    primary_terms = _normalize_vocab_terms(list(core_terms) + list(berkeley_terms) + list(extra_terms))

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

    def _noise_pattern() -> np.ndarray:
        return np.clip(rng.random((size, size), dtype=np.float32), 0.0, 1.0).astype(np.float32, copy=False)

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
                rows.extend([np.asarray(v, dtype=np.float32, copy=False) for v in vals])
        return rows

    object_seed_rows: List[Tuple[np.ndarray, np.ndarray]] = []
    n_obj = min(len(payload_images_base), len(payload_conditions_supervised_base))
    for i in range(int(n_obj)):
        object_seed_rows.append(
            (
                _image_any_to_rgb_chw01(payload_images_base[int(i)], image_size=int(size)),
                np.asarray(payload_conditions_supervised_base[int(i)], dtype=np.float32, copy=False).reshape(-1),
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

    def _pick_object() -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if len(object_seed_rows) <= 0:
            return _pick_signal(), None
        idx = int(rng.integers(0, len(object_seed_rows)))
        img, sv = object_seed_rows[idx]
        return _image_any_to_rgb_chw01(img, image_size=int(size)), np.asarray(sv, dtype=np.float32, copy=False).reshape(-1)

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
        terms = list(extra_terms) + [key]
        vec = np.asarray(condition_vector_builder(terms, base_supervised), dtype=np.float32).reshape(-1)
        if int(vec.size) != int(c):
            raise RuntimeError(f"flashcard condition builder width mismatch: got={int(vec.size)} expected={int(c)}")
        cards_img.append(_image_any_to_rgb_chw01(img, image_size=int(size)))
        cards_cond.append(np.asarray(vec, dtype=np.float32, copy=False).reshape(-1))
        term_counts[key] = int(term_counts.get(key, 0)) + 1

    damage_terms = {"blur damage", "noise damage", "dropout damage", "quantization damage", "stride skew damage"}
    for term in target_terms:
        term_key = re.sub(r"\s+", " ", str(term)).strip().lower()
        for _ in range(int(per)):
            if term_key == "none":
                _emit(term_key, _to_rgb(np.full((size, size), 0.50, dtype=np.float32)), None, ["none"])
                continue
            if term_key == "noise":
                _emit(term_key, _to_rgb(_noise_pattern()), None, ["noise"])
                continue
            if term_key == "signal":
                _emit(term_key, _pick_signal(), None, ["signal"])
                continue
            if term_key == "mixed noise and signal":
                sig = _pick_signal()
                noi = _to_rgb(_noise_pattern())
                a = float(rng.uniform(0.35, 0.65))
                _emit(term_key, np.clip((a * sig) + ((1.0 - a) * noi), 0.0, 1.0), None, ["signal", "noise", "mixed noise and signal"])
                continue
            if term_key == "white":
                _emit(term_key, _to_rgb(np.ones((size, size), dtype=np.float32)), None, ["white", "signal"])
                continue
            if term_key == "black":
                _emit(term_key, _to_rgb(np.zeros((size, size), dtype=np.float32)), None, ["black", "signal"])
                continue
            if term_key == "object":
                obj_img, obj_sup = _pick_object()
                _emit(term_key, obj_img, obj_sup, ["object", "berkeley sbd dataset", "signal"])
                continue
            if term_key in ("mnist dataset", "emnist dataset", "kmnist dataset", "berkeley sbd dataset"):
                if term_key == "berkeley sbd dataset":
                    obj_img, obj_sup = _pick_object()
                    _emit(term_key, obj_img, obj_sup, [term_key, "object", "signal"])
                else:
                    ds_img = _pick_dataset_row(term_key)
                    _emit(term_key, ds_img, None, [term_key, "signal"])
                continue
            if term_key == "gan image":
                chk = ((np.floor(xx / 8.0) + np.floor(yy / 8.0)) % 2.0).astype(np.float32, copy=False)
                sig = _signal_pattern()
                _emit(term_key, _to_rgb(np.clip((0.65 * chk) + (0.35 * sig), 0.0, 1.0)), None, ["gan image", "signal"])
                continue
            if term_key == "regurgitated content":
                sig0 = _signal_pattern()
                sig1 = np.roll(sig0, shift=int(rng.integers(2, 9)), axis=1)
                _emit(
                    term_key,
                    _to_rgb(np.clip((0.55 * sig0) + (0.45 * sig1), 0.0, 1.0)),
                    None,
                    ["regurgitated content", "mixed noise and signal", "signal"],
                )
                continue
            if term_key in damage_terms:
                obj_img, obj_sup = _pick_object()
                dmg = _apply_damage(obj_img, term_key)
                _emit(term_key, dmg, obj_sup, [term_key, "object", "berkeley sbd dataset", "signal"])
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
    return cards_img, cards_cond, info


def _rotate_active_extra_terms(
    active_terms: Sequence[str],
    pool_terms: Sequence[str],
    replace_count: int,
    seed: int,
    locked_prefix_count: int = 0,
) -> Tuple[List[str], Dict[str, Any]]:
    active = [str(x) for x in active_terms]
    pool = _normalize_vocab_terms(pool_terms)
    k_replace = max(0, int(replace_count))
    locked = max(0, min(int(len(active)), int(locked_prefix_count)))
    info: Dict[str, Any] = {
        "changed": False,
        "replaced": 0,
        "pool_terms": int(len(pool)),
        "active_terms": int(len(active)),
        "locked_prefix": int(locked),
    }
    unlocked = int(len(active) - int(locked))
    if len(active) <= 0 or len(pool) <= len(active) or k_replace <= 0 or int(unlocked) <= 0:
        return active, info
    rng = np.random.default_rng(int(seed))
    replace_n = min(int(k_replace), int(unlocked))
    replace_space = np.arange(int(locked), len(active), dtype=np.int64)
    replace_idx = rng.choice(replace_space, size=replace_n, replace=False).astype(np.int64).tolist()
    active_lc = {str(x).strip().lower() for x in active}
    candidates = [t for t in pool if str(t).strip().lower() not in active_lc]
    if len(candidates) <= 0:
        return active, info
    rng.shuffle(candidates)
    changed = 0
    for pos, idx in enumerate(replace_idx):
        if pos >= len(candidates):
            break
        new_t = str(candidates[pos]).strip()
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
    active_extra_terms: Sequence[str],
    condition_num_classes: int,
    args: Any,
) -> Tuple[str, Dict[str, Any]]:
    profile: Dict[str, Any] = {
        "supervised_class_names": [str(x) for x in supervised_class_names],
        "active_extra_terms": [str(x) for x in active_extra_terms],
        "condition_num_classes": int(condition_num_classes),
        "label_embedding_backend": str(getattr(args, "label_embedding_backend", "")),
        "label_embedding_model": str(getattr(args, "label_embedding_model", "")),
        "label_embedding_dim": int(getattr(args, "label_embedding_dim", 0)),
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
    blob = json.dumps(profile, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
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

    base = F.normalize(bank_t[: int(sup), :], dim=1, eps=1e-6)
    y_embed = y_sup @ base
    y_norm = torch.linalg.norm(y_embed, dim=1, keepdim=True)
    if bool(torch.any(y_norm <= 1e-6).item()):
        prior = base.mean(dim=0, keepdim=True)
        fill_mask = (y_norm <= 1e-6).to(dtype=torch.float32)
        y_embed = (y_embed * (1.0 - fill_mask)) + (prior * fill_mask)
        y_norm = torch.linalg.norm(y_embed, dim=1, keepdim=True)
    y_embed = y_embed / torch.clamp(y_norm, min=1e-6)

    bank_full = F.normalize(bank_t, dim=1, eps=1e-6)
    t = max(0.1, float(semantic_target_temperature))
    soft_logits = y_embed @ bank_full.transpose(0, 1)
    y_sem = torch.sigmoid(soft_logits * t).to(dtype=torch.float32)
    return y_sem, y_sup


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


def _build_berkeley_refresh_loader(
    data_root: str,
    image_size: int,
    auto_install_scipy: bool,
    batch_size: int,
    num_workers: int,
    max_train: int,
    seed: int,
    device: torch.device,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
):
    try:
        from berkeley_sbd_pretrain import prepare_sbd_multilabel
    except ModuleNotFoundError:
        from toys_to_survive_development.berkeley_sbd_pretrain import prepare_sbd_multilabel

    train_ds, _ = prepare_sbd_multilabel(
        data_root=data_root,
        image_size=image_size,
        auto_install_scipy=auto_install_scipy,
    )
    sampler = None
    shuffle = True
    sample_count = int(len(train_ds))
    if max_train > 0 and len(train_ds) > int(max_train):
        sample_count = int(max_train)
        g = torch.Generator()
        g.manual_seed(max(0, int(seed)))
        sampler = torch.utils.data.RandomSampler(
            data_source=train_ds,
            replacement=False,
            num_samples=int(sample_count),
            generator=g,
        )
        shuffle = False

    loader = DataLoader(
        train_ds,
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


def _schedule_semantic_gate_indices(
    targets: Sequence[np.ndarray],
    class_names: Sequence[str],
    active_terms: Sequence[str],
    seed: int,
    max_samples: int = 0,
    activation_threshold: float = 0.55,
) -> Tuple[List[int], Dict[str, Any]]:
    n = int(len(targets))
    if n <= 0:
        return [], {
            "rows": 0,
            "active_terms": 0,
            "active_term_hits": {},
            "bucketed_rows": 0,
            "selected_rows": 0,
            "max_samples": int(max_samples),
            "threshold": float(activation_threshold),
        }
    if isinstance(targets, np.ndarray):
        y = np.asarray(targets, dtype=np.float32)
    else:
        y = np.stack([np.asarray(row, dtype=np.float32).reshape(-1) for row in targets], axis=0).astype(np.float32, copy=False)
    if int(y.ndim) != 2:
        y = np.asarray(y, dtype=np.float32).reshape(int(n), -1)

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

    rng = np.random.default_rng(int(seed))
    threshold = float(max(0.0, min(1.0, float(activation_threshold))))
    buckets: Dict[int, List[int]] = {}
    active_hits: Dict[str, int] = {}
    for idx in active_indices:
        vals = np.asarray(y[:, int(idx)], dtype=np.float32).reshape(-1)
        picks = np.where(vals >= float(threshold))[0].astype(np.int64)
        if int(picks.size) <= 0:
            top_take = min(int(n), max(6, int(max(1, n) // 12)))
            if int(top_take) > 0:
                cand = np.argsort(vals)[::-1][: int(top_take)].astype(np.int64)
                cand = cand[vals[cand] > 1e-6]
                picks = cand.astype(np.int64, copy=False)
        picks_l = picks.tolist()
        rng.shuffle(picks_l)
        buckets[int(idx)] = [int(i) for i in picks_l]
        name = str(class_names[int(idx)]) if int(idx) < int(len(class_names)) else f"class_{int(idx)}"
        active_hits[name] = int(len(picks_l))

    ordered: List[int] = []
    used: set = set()
    if len(buckets) > 0:
        while True:
            progressed = False
            for idx in active_indices:
                row_bucket = buckets.get(int(idx), [])
                while len(row_bucket) > 0 and int(row_bucket[0]) in used:
                    row_bucket.pop(0)
                if len(row_bucket) <= 0:
                    continue
                pick = int(row_bucket.pop(0))
                if pick in used:
                    continue
                ordered.append(int(pick))
                used.add(int(pick))
                progressed = True
                if int(max_samples) > 0 and int(len(ordered)) >= int(max_samples):
                    break
            if (not progressed) or (int(max_samples) > 0 and int(len(ordered)) >= int(max_samples)):
                break

    remaining = [int(i) for i in range(int(n)) if int(i) not in used]
    rng.shuffle(remaining)
    ordered.extend(remaining)
    if int(max_samples) > 0:
        ordered = ordered[: int(max_samples)]

    info = {
        "rows": int(n),
        "active_terms": int(len(active_indices)),
        "active_term_hits": {str(k): int(v) for k, v in active_hits.items()},
        "bucketed_rows": int(len(used)),
        "selected_rows": int(len(ordered)),
        "max_samples": int(max_samples),
        "threshold": float(threshold),
    }
    return [int(i) for i in ordered], info


def _build_gate_loader_from_arrays(
    images: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
    max_samples: int = 0,
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
    y_dim = max(1, int(max(int(r.size) for r in y_rows)))
    y_np = np.zeros((int(n), int(y_dim)), dtype=np.float32)
    for i, row in enumerate(y_rows):
        take = min(int(y_dim), int(row.size))
        y_np[int(i), : int(take)] = row[: int(take)]
    y_np = np.clip(y_np, 0.0, 1.0).astype(np.float32, copy=False)

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

    for xb, yb in loader:
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
                    sem_targets = _build_semantic_supervision_targets(
                        classifier=classifier,
                        y_multihot=yb_part,
                        supervised_classes=int(supervised_dim),
                    )
                    if sem_targets is not None:
                        y_sem, y_sup = sem_targets
                        logits_sup = logits[:, : int(y_sup.shape[1])]
                    else:
                        logits_sup = logits
                        if int(logits_sup.shape[1]) > int(supervised_dim):
                            logits_sup = logits_sup[:, : int(supervised_dim)]
                        y_sup = _align_multilabel_targets_dim(yb_part, out_dim=int(logits_sup.shape[1])).to(dtype=torch.float32)

                    with _autocast_context(device=device, enabled=amp_enabled, amp_dtype_t=amp_dtype_t):
                        if sem_targets is not None:
                            pred = torch.sigmoid(logits.to(dtype=torch.float32))
                            tgt = y_sem.to(dtype=torch.float32)
                            loss = F.mse_loss(pred, tgt)
                        else:
                            pred = torch.sigmoid(logits_sup.to(dtype=torch.float32))
                            tgt = y_sup.to(dtype=torch.float32)
                            loss = F.mse_loss(pred, tgt)
                    part_n = int(xb_part.shape[0])
                    batch_loss += float(loss.item()) * part_n
                    batch_seen += part_n
                    batch_logits.append(logits_sup.detach().cpu())
                    batch_targets.append(y_sup.detach().cpu())

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
            if channels_last:
                xb = xb.contiguous(memory_format=torch.channels_last)
            with _autocast_context(device=device, enabled=amp_enabled, amp_dtype_t=amp_dtype_t):
                logits = classifier(xb)
            supervised_dim = int(yb.shape[1]) if int(yb.ndim) == 2 else int(logits.shape[1])
            if int(active_classes) > 0:
                supervised_dim = min(int(supervised_dim), int(active_classes))
            supervised_dim = max(1, int(supervised_dim))
            sem_targets = _build_semantic_supervision_targets(
                classifier=classifier,
                y_multihot=yb,
                supervised_classes=int(supervised_dim),
            )
            if sem_targets is not None:
                y_sem, y_sup = sem_targets
                logits_sup = logits[:, : int(y_sup.shape[1])]
            else:
                logits_sup = logits
                if int(logits_sup.shape[1]) > int(supervised_dim):
                    logits_sup = logits_sup[:, : int(supervised_dim)]
                y_sup = _align_multilabel_targets_dim(yb, out_dim=int(logits_sup.shape[1])).to(dtype=torch.float32)
            with _autocast_context(device=device, enabled=amp_enabled, amp_dtype_t=amp_dtype_t):
                if sem_targets is not None:
                    pred = torch.sigmoid(logits.to(dtype=torch.float32))
                    tgt = y_sem.to(dtype=torch.float32)
                    loss = F.mse_loss(pred, tgt)
                else:
                    pred = torch.sigmoid(logits_sup.to(dtype=torch.float32))
                    tgt = y_sup.to(dtype=torch.float32)
                    loss = F.mse_loss(pred, tgt)
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
        picked.append(metas[j])
        start = int(rng.integers(0, int(s.size) - int(chunk_samples) + 1))
        xb[i, :] = np.asarray(s[start : start + int(chunk_samples)], dtype=np.float32, copy=False)
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
            p = Path(fp)
            candidates = []
            if p.is_absolute():
                candidates.append(p)
            else:
                candidates.append(Path.cwd() / p)
                candidates.append(library_dir / p)
            found = None
            for c in candidates:
                if c.exists():
                    found = c.resolve()
                    break
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
    for i in range(max(1, int(count))):
        noise = (rng.standard_normal(sample_count).astype(np.float32) * float(noise_std))
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
        out_path = target_dir / f"latent_{i:05d}.wav"
        _save_mono_wav(out_path, y, framerate=framerate)
        rows.append({"file": str(out_path), "mixed": bool(used_mix), "source": src})

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
                        out_path = library_dir / fn
                        _save_mono_wav(out_path, xh[i].detach().cpu().numpy(), framerate=int(picked[i]["framerate"]))
                        sem_terms = ["regurgitated content", "mixed noise and signal", "signal"]
                        if semantic_class_names is not None and int(probs.shape[1]) > 0:
                            k_sem = max(1, min(6, int(probs.shape[1]), int(len(semantic_class_names))))
                            top_vals, top_idx = torch.topk(probs[i], k=k_sem, dim=0)
                            for score_t, idx_t in zip(top_vals.tolist(), top_idx.tolist()):
                                score = float(score_t)
                                if score < 0.20:
                                    continue
                                idx = int(idx_t)
                                if 0 <= idx < len(semantic_class_names):
                                    sem_terms.append(str(semantic_class_names[idx]))
                        sem_terms = _normalize_vocab_terms(sem_terms)
                        row = {
                            "file": str(out_path),
                            "label": int(yb_np[i]),
                            "score": sc,
                            "l1": l1,
                            "source": picked[i]["path"],
                            "semantic_terms": list(sem_terms),
                            "cycle": int(cycle_id),
                            "round": int(round_id),
                        }
                        f.write(json.dumps(row) + "\n")
                        accepted_rows.append(row)
                    if len(accepted_rows) >= int(library_limit):
                        break

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


def _spectral_centroid(x: np.ndarray, sr: int) -> float:
    if x.size < 64:
        return 0.0
    n = min(4096, x.size)
    n = int(2 ** np.floor(np.log2(max(64, n))))
    y = x[:n].astype(np.float64, copy=False)
    y = y * np.hanning(n)
    spec = np.abs(np.fft.rfft(y))
    denom = float(np.sum(spec))
    if denom <= 1e-12:
        return 0.0
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sr))
    return float(np.sum(freqs * spec) / denom)


def _build_labels(records: Sequence[WaveRecord], data_root: str, label_mode: str, pseudo_classes: int):
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

        _log("Folder labels are single-class; switching to pseudo labels from spectral centroid bins.")

    pseudo_classes = max(2, int(pseudo_classes))
    base_cfg = RenderConfig()
    centroids = []
    for rec in records:
        mono, _ = _decode_record_to_mono(rec, base_cfg, max_points=65536)
        centroids.append(_spectral_centroid(mono, sr=rec.framerate))
    vals = np.array(centroids, dtype=np.float64)
    q = np.linspace(0.0, 1.0, pseudo_classes + 1)
    edges = np.quantile(vals, q)
    if np.allclose(edges, edges[0]):
        rank = np.argsort(np.argsort(vals))
        labels = (rank * pseudo_classes) // max(1, len(vals))
    else:
        labels = np.searchsorted(edges[1:-1], vals, side="right")
    labels = labels.astype(np.int64)
    class_names = [f"pseudo_bin_{i}" for i in range(int(labels.max()) + 1)]
    return labels, class_names, "spectral_quantile"


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
        streams.append(mono.astype(np.float32, copy=False))
        ys.append(int(labels[idx]))
        bits.append(int(bit_depth))
        meta.append({"path": records[idx].path, "framerate": int(records[idx].framerate)})
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
    out_streams = [np.asarray(streams[i], dtype=np.float32, copy=False) for i in keep]
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
    on_append_row: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> int:
    added = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        p = str(row.get("file", "")).strip()
        if not p or p in seen_paths:
            continue
        try:
            rec = read_wav_record(p)
            mono, _ = _decode_record_to_mono(rec, cfg=RenderConfig(), max_points=max_points)
            if mono.size <= 0:
                continue
            train_streams.append(mono.astype(np.float32, copy=False))
            train_stream_labels.append(int(row.get("label", 0)))
            train_meta.append(
                {
                    "path": str(p),
                    "framerate": int(rec.framerate),
                    "semantic_terms": list(row.get("semantic_terms", [])) if isinstance(row.get("semantic_terms", []), list) else [],
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


def _latent_pool_stream_kind(path: str) -> str:
    p = str(path).replace("\\", "/").lower()
    if "/latent_wave_pool/noise/" in p:
        return "noise"
    if "/latent_wave_pool/mix/" in p:
        return "mix"
    return "other"


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
        idx = np.where(arr > 0.0)[0].astype(np.int64)
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
        return "target:none"
    t = float(threshold)
    idx = np.where(arr >= t)[0].astype(np.int64)
    if int(idx.size) <= 0:
        idx = np.where(arr > 0.0)[0].astype(np.int64)
    if int(idx.size) <= 0:
        return "target:none"
    order = idx[np.argsort(-arr[idx])]
    k = max(1, int(max_items))
    picks = order[:k].tolist()
    names = [str(class_names[int(i)]) if int(i) < len(class_names) else f"class_{int(i)}" for i in picks]
    more = ""
    if int(order.size) > k:
        more = f"+{int(order.size) - k}"
    return f"target:{','.join(names)}{more}"


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
    if int(c) <= int(sup):
        info["reason"] = "no_extra_dims"
        return base_rows, info
    if label_embedding_bank is None:
        info["reason"] = "no_label_embedding_bank"
        return base_rows, info

    bank = np.asarray(label_embedding_bank, dtype=np.float32)
    if int(bank.ndim) != 2 or int(bank.shape[1]) <= 0:
        info["reason"] = f"invalid_bank_shape:{tuple(bank.shape)}"
        return base_rows, info
    if int(bank.shape[0]) < int(c):
        info["reason"] = f"bank_class_count_lt_condition_dim:{int(bank.shape[0])}<{int(c)}"
        return base_rows, info

    y_sup = torch.from_numpy(np.stack(sup_rows, axis=0).astype(np.float32, copy=False))
    bank_t = torch.from_numpy(bank[: int(c), :].astype(np.float32, copy=False))
    base = F.normalize(bank_t[: int(sup), :], dim=1, eps=1e-6)
    y_embed = y_sup @ base
    y_norm = torch.linalg.norm(y_embed, dim=1, keepdim=True)
    if bool(torch.any(y_norm <= 1e-6).item()):
        prior = base.mean(dim=0, keepdim=True)
        fill = (y_norm <= 1e-6).to(dtype=torch.float32)
        y_embed = (y_embed * (1.0 - fill)) + (prior * fill)
        y_norm = torch.linalg.norm(y_embed, dim=1, keepdim=True)
    y_embed = y_embed / torch.clamp(y_norm, min=1e-6)
    bank_full = F.normalize(bank_t, dim=1, eps=1e-6)
    t = max(0.1, float(semantic_target_temperature))
    y_sem = torch.sigmoid((y_embed @ bank_full.transpose(0, 1)) * t).to(torch.float32)
    y_sem_np = y_sem.cpu().numpy().astype(np.float32, copy=False)

    out_rows: List[np.ndarray] = [np.clip(y_sem_np[i], 0.0, 1.0).astype(np.float32, copy=False) for i in range(int(n_rows))]

    extras = np.stack([row[int(sup) : int(c)] for row in out_rows], axis=0).astype(np.float32, copy=False)
    info["expanded"] = True
    info["reason"] = "semantic_bank_full"
    info["extra_mean"] = float(extras.mean()) if extras.size > 0 else 0.0
    info["extra_max"] = float(extras.max()) if extras.size > 0 else 0.0
    return out_rows, info


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
    if len(payload_images) <= 0:
        return None, None, 0
    lo = max(1, int(min_payloads))
    hi = max(lo, int(max_payloads))
    k = int(rng.integers(lo, hi + 1))
    if k <= 0:
        k = 1

    idx = rng.choice(np.arange(len(payload_images)), size=int(k), replace=(k > len(payload_images))).astype(np.int64)
    canvas = np.zeros((3, max(1, int(target_h)), max(1, int(target_w))), dtype=np.float32)
    rows = np.linspace(0, int(target_h), num=int(k) + 1, dtype=np.int64)
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
        d = min(int(target_vec.size), int(t.size))
        if d > 0:
            target_vec[:d] = np.maximum(target_vec[:d], t[:d])
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
    try:
        from berkeley_sbd_pretrain import prepare_sbd_multilabel
    except ModuleNotFoundError:
        from toys_to_survive_development.berkeley_sbd_pretrain import prepare_sbd_multilabel

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
        train_ds, val_ds = prepare_sbd_multilabel(
            data_root=str(root),
            image_size=int(size),
            auto_install_scipy=bool(auto_install_scipy),
        )
        n_train = int(len(train_ds))
        n_val = int(len(val_ds))
        all_images: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        terms_rows = []
        source_rows = []
        source_counts: Dict[str, int] = {}

        for split_name, ds in [("train", train_ds), ("val", val_ds)]:
            source_key = f"berkeley_sbd_{split_name}"
            for local_idx in range(int(len(ds))):
                try:
                    x, y = ds[int(local_idx)]
                    x01 = torch.clamp((x.float() * 0.5) + 0.5, 0.0, 1.0).cpu().numpy().astype(np.float32, copy=False)
                    yv = torch.clamp(y.float(), 0.0, 1.0).cpu().numpy().astype(np.float32, copy=False).reshape(-1)
                    if int(yv.size) >= int(n_classes):
                        yv = yv[: int(n_classes)]
                    else:
                        pad = np.zeros((int(n_classes) - int(yv.size),), dtype=np.float32)
                        yv = np.concatenate([yv, pad], axis=0)
                    pos = np.where(yv > 0.5)[0].astype(np.int64).tolist()
                    label_terms = [str(class_names[int(i)]) for i in pos if 0 <= int(i) < int(len(class_names))]
                    terms = _normalize_vocab_terms(
                        [str(split_name), "berkeley sbd dataset", "object", "signal"] + list(label_terms)
                    )
                    all_images.append(np.asarray(x01, dtype=np.float32, copy=False))
                    all_labels.append(np.asarray(yv, dtype=np.float32, copy=False))
                    terms_rows.append(list(terms))
                    source_rows.append(str(source_key))
                    source_counts[str(source_key)] = int(source_counts.get(str(source_key), 0)) + 1
                except Exception:
                    continue

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
                        img = _to_rgb_chw01(path=fp, size=int(size))
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
                        all_images.append(np.asarray(img, dtype=np.float32, copy=False))
                        all_labels.append(np.asarray(vec, dtype=np.float32, copy=False))
                        terms_rows.append(list(ref_terms))
                        source_rows.append(str(dataset_name))
                        source_counts[str(dataset_name)] = int(source_counts.get(str(dataset_name), 0)) + 1
                    except Exception:
                        continue

        if len(all_images) <= 0:
            return [], [], {"available": 0, "used": 0, "available_train": 0, "available_val": 0}, []

        images_np = np.stack(all_images, axis=0).astype(np.float16, copy=False)
        labels_np = np.stack(all_labels, axis=0).astype(np.float32, copy=False)
        source_catalog = {
            "generated_at": float(time.time()),
            "external_source_root": str(ext_root),
            "external_source_signature": ext_sig,
            "source_counts": {str(k): int(v) for k, v in source_counts.items()},
            "available_rows": int(images_np.shape[0]),
            "available_train": int(n_train),
            "available_val": int(n_val),
        }
        manifest = {
            "version": int(cache_version),
            "image_size": int(size),
            "num_classes": int(n_classes),
            "class_names": list(class_names),
            "available_rows": int(images_np.shape[0]),
            "available_train": int(n_train),
            "available_val": int(n_val),
            "available_external": int(max(0, int(images_np.shape[0]) - int(n_train) - int(n_val))),
            "external_source_root": str(ext_root),
            "external_source_signature": ext_sig,
            "source_counts": {str(k): int(v) for k, v in source_counts.items()},
            "source_catalog": str(source_catalog_path),
            "generated_at": float(time.time()),
        }
        np.save(str(images_path), np.asarray(images_np, dtype=np.float16))
        np.save(str(labels_path), np.asarray(labels_np, dtype=np.float32))
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
            terms = _normalize_vocab_terms([str(src), "signal"])
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

    for s, m in zip(streams, metas):
        kind = _latent_pool_stream_kind(m.get("path", ""))
        if kind == "noise":
            noise_total += 1
            out_streams.append(np.asarray(s, dtype=np.float32))
            out_targets.append(_kind_target("noise"))
            continue
        if kind != "mix":
            out_streams.append(np.asarray(s, dtype=np.float32))
            out_targets.append(_kind_target("none"))
            continue

        mix_total += 1
        if int(num_classes) <= 0 or len(payload_images) <= 0:
            out_streams.append(np.asarray(s, dtype=np.float32))
            out_targets.append(_kind_target("mix"))
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
        target_class_count.append(int(n_target))

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
        "mean_target_classes_per_stream": (float(np.mean(target_class_count)) if len(target_class_count) > 0 else 0.0),
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
        default=0.28,
        help=(
            "Hard gestation gate: maximum allowed average classifier loss on bootstrap-archetype-only rows. "
            "No downstream stage runs until this gate is satisfied."
        ),
    )
    p.add_argument(
        "--gate-gestation-maintain-rounds",
        type=int,
        default=1,
        help="Required consecutive gestation gate passes before total-dataset gate can pass.",
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
        help="If >0, total-dataset classifier mean-confidence gate (evaluated on payload+bootstrap gate set).",
    )
    p.add_argument(
        "--gate-berkeley-min-macro-f1",
        type=float,
        default=0.0,
        help="If >0, total-dataset classifier macro-F1 gate (evaluated on payload+bootstrap gate set).",
    )
    p.add_argument(
        "--gate-berkeley-loss-target",
        type=float,
        default=0.0,
        help=(
            "If >0, explicit total-dataset classifier loss target for loss-aware gate clamp. "
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
        help="Required consecutive total-dataset gate passes before downstream is allowed.",
    )
    p.add_argument(
        "--gate-berkeley-val-max",
        type=int,
        default=0,
        help="Max total-dataset gate samples per eval (0=all payload+bootstrap rows).",
    )
    p.add_argument(
        "--gate-berkeley-batch-size",
        type=int,
        default=0,
        help=(
            "Batch size for total-dataset gate evaluation loader. "
            "0 uses --berkeley-refresh-batch-size (or --classifier-batch-size when refresh batch is auto)."
        ),
    )
    p.add_argument(
        "--gate-berkeley-eval-max-steps",
        type=int,
        default=0,
        help="Max total-dataset gate batches per eval (0=full loader).",
    )
    p.add_argument(
        "--gate-total-token-schedule-enabled",
        dest="gate_total_token_schedule_enabled",
        action="store_true",
        help="Sort total-dataset gate rows to prioritize active optional semantic tokens before gate decisions.",
    )
    p.add_argument("--no-gate-total-token-schedule-enabled", dest="gate_total_token_schedule_enabled", action="store_false")
    p.add_argument(
        "--gate-total-token-schedule-threshold",
        type=float,
        default=0.55,
        help="Activation threshold used while token-sorting total-dataset gate rows.",
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
    stage_module_offload_runtime = bool(
        bool(args.stage_module_offload) and device.type == "cuda" and (not bool(args.compile_models))
    )
    if bool(args.stage_module_offload) and bool(args.compile_models) and device.type == "cuda":
        _log("VRAM: stage-module-offload disabled because --compile-models is enabled.")

    out_dir = Path(args.output_dir)
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
            count=args.latent_fallback_count,
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
        raise RuntimeError("Need at least 8 readable WAV files.")

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
        )
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
        wave_zero_shot_queries = _parse_label_query_texts(str(args.wave_zero_shot_query_texts))
        if len(wave_zero_shot_queries) <= 0:
            wave_zero_shot_queries = _parse_label_query_texts(str(args.label_query_texts))
            if len(wave_zero_shot_queries) > 0:
                wave_zero_shot_query_source = "label_query_texts"
        else:
            wave_zero_shot_query_source = "wave_zero_shot_query_texts"
        if len(wave_zero_shot_queries) <= 0:
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
            bank_norm = _semantic_bank_norm_runtime()
            if bank_norm is None or int(bank_norm.shape[0]) < int(sup):
                return zero
            base = bank_norm[: int(sup), :]
            y_embed = arr_sup @ base
            norm = float(np.linalg.norm(y_embed))
            if norm <= 1e-8:
                y_embed = np.mean(base, axis=0).astype(np.float32, copy=False)
                norm = float(np.linalg.norm(y_embed))
            y_embed = (y_embed / max(1e-8, norm)).astype(np.float32, copy=False)
            sims = y_embed @ bank_norm.transpose(1, 0)
            temp = max(0.1, float(args.semantic_vocab_text_condition_temperature))
            vec = (1.0 / (1.0 + np.exp(-np.clip(sims * temp, -20.0, 20.0)))).astype(np.float32, copy=False)
            out = np.clip(vec[: int(c)], 0.0, 1.0).astype(np.float32, copy=False)
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
            clean_terms = _normalize_vocab_terms(
                [str(x) for x in list(terms) + (list(origin_terms) if origin_terms is not None else [])]
            )
            if len(clean_terms) <= 0:
                return out
            bank_norm = _semantic_bank_norm_runtime()
            if bank_norm is None or int(bank_norm.shape[0]) <= 0:
                return out
            cache_key = tuple([str(x).strip().lower() for x in clean_terms])
            term_vec = _semantic_text_condition_cache.get(cache_key, None)
            if term_vec is None:
                q = _encode_query_texts_for_bank(
                    query_texts=clean_terms,
                    bank_info=label_embedding_info,
                    args=args,
                    device=device,
                )
                if int(q.ndim) != 2 or int(q.shape[0]) <= 0:
                    term_vec = np.zeros((int(c),), dtype=np.float32)
                else:
                    q_mean = np.mean(_normalize_l2_rows_np(q.astype(np.float32, copy=False)), axis=0).astype(np.float32, copy=False)
                    q_mean = (q_mean / max(1e-8, float(np.linalg.norm(q_mean)))).astype(np.float32, copy=False)
                    sims = q_mean @ bank_norm.transpose(1, 0)
                    temp = max(0.1, float(args.semantic_vocab_text_condition_temperature))
                    vec = (1.0 / (1.0 + np.exp(-np.clip(sims * temp, -20.0, 20.0)))).astype(np.float32, copy=False)
                    term_vec = np.clip(vec[: int(c)], 0.0, 1.0).astype(np.float32, copy=False)
                _semantic_text_condition_cache[cache_key] = np.array(term_vec, dtype=np.float32, copy=True)
            out = np.maximum(out, np.asarray(term_vec, dtype=np.float32).reshape(-1)[: int(c)]).astype(np.float32, copy=False)
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
        semantic_vocab_pool_terms: List[str] = _normalize_vocab_terms(
            list(active_extra_terms)
            + list(split_class_names)
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
        symbol_pool_by_term: Dict[str, List[np.ndarray]] = {}
        symbol_pool_origin_terms_by_term: Dict[str, List[str]] = {}
        symbol_pool_info: Dict[str, Any] = {"enabled": False, "available_terms": 0, "samples": 0}
        bootstrap_symbol_terms: set = set()
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
        if int(args.berkeley_refresh_epochs) > 0 and (
            int(args.berkeley_refresh_every) > 0 or int(args.berkeley_refresh_round_every) > 0
        ):
            refresh_size = int(shared_embed_image_size)
            refresh_loader_batch = (
                max(1, int(args.berkeley_refresh_loader_batch_size))
                if int(args.berkeley_refresh_batch_size) <= 0
                else max(1, int(args.berkeley_refresh_batch_size))
            )
            refresh_loader, refresh_count = _build_berkeley_refresh_loader(
                data_root=args.berkeley_data_root,
                image_size=refresh_size,
                auto_install_scipy=bool(args.berkeley_auto_install_scipy),
                batch_size=refresh_loader_batch,
                num_workers=args.berkeley_refresh_workers,
                max_train=args.berkeley_refresh_max_train,
                seed=args.seed + 7300,
                device=device,
                persistent_workers=bool(args.loader_persistent_workers),
                prefetch_factor=int(args.loader_prefetch_factor),
            )
            _log(
                "Berkeley refresh loader ready: "
                f"search_every={args.berkeley_refresh_every}, round_every={args.berkeley_refresh_round_every}, "
                f"epochs={args.berkeley_refresh_epochs}, train={refresh_count}"
            )
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

        berkeley_gate_loader = None
        berkeley_gate_loader_count = 0
        berkeley_gate_schedule_info: Dict[str, Any] = {"rows": 0, "selected_rows": 0}
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
                "total_dataset=bootstrap+payload(all datasets)."
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

        berkeley_refresh_required_samples = max(0, int(refresh_count))
        if bool(total_gate_runtime_enabled):
            if int(berkeley_refresh_required_samples) > 0:
                _log(
                    "Berkeley refresh startup requirement disabled for semantic total-dataset gate; "
                    "coverage gating now tracks bootstrap+payload rows."
                )
            berkeley_refresh_required_samples = 0
        berkeley_refresh_samples_seen = _sum_berkeley_refresh_samples(refresh_rows)
        if berkeley_refresh_required_samples > 0:
            _log(
                "Berkeley startup requirement: "
                f"require_full_dataset_once={berkeley_refresh_required_samples} "
                f"already_seen={berkeley_refresh_samples_seen}"
            )
            if refresh_cache_samples > 0 and refresh_cache_samples < berkeley_refresh_required_samples:
                _log(
                    "Berkeley refresh cache is a subset of train data; "
                    "startup pass will force loader mode until full-dataset requirement is met."
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
                force_cache_rebuild=bool(args.berkeley_payload_cache_rebuild),
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
            if need_payload_bank and len(payload_images) <= 0:
                raise RuntimeError(
                    "Payload bank is required but no Berkeley payload images were prepared. "
                    "Check --berkeley-data-root and dataset availability."
                )
            payload_images_base = [np.asarray(x, dtype=np.float32, copy=False) for x in payload_images]
            payload_conditions_supervised_base = []
            payload_condition_terms_supervised_base = []
            if len(payload_conditions) > 0:
                for i, row in enumerate(payload_conditions):
                    arr = np.asarray(row, dtype=np.float32).reshape(-1)
                    if int(arr.size) <= 0:
                        raise RuntimeError(f"Semantic payload row {i} is empty.")
                    if int(arr.size) >= int(supervised_num_classes):
                        arr = arr[: int(supervised_num_classes)]
                    else:
                        pad = np.zeros((int(supervised_num_classes) - int(arr.size),), dtype=np.float32)
                        arr = np.concatenate([arr, pad], axis=0)
                    payload_conditions_supervised_base.append(np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False))
                    row_terms: List[str] = []
                    if int(i) < int(len(payload_condition_terms)) and isinstance(payload_condition_terms[int(i)], list):
                        row_terms = _normalize_vocab_terms([str(x) for x in payload_condition_terms[int(i)]])
                    if len(row_terms) <= 0:
                        row_terms = ["berkeley sbd dataset", "object", "signal"]
                    payload_condition_terms_supervised_base.append(list(row_terms))
            payload_conditions = []
            if int(semantic_extra_count) > 0 and bool(args.semantic_vocab_auto_symbol_pool):
                symbol_pool_official, symbol_pool_info_official = _build_auto_symbol_term_pool(
                    data_root=str(args.semantic_vocab_symbol_pool_root),
                    image_size=int(payload_size),
                    seed=int(args.seed) + 8453,
                    max_samples_per_term=max(1, int(args.semantic_vocab_symbol_samples_per_term)),
                    include_digits=True,
                    include_letters=True,
                    include_pictograms=bool(args.semantic_vocab_symbol_include_pictograms),
                )
                symbol_pool_synth, symbol_pool_info_synth = _build_synthetic_semantic_symbol_pool(
                    image_size=int(payload_size),
                    seed=int(args.seed) + 9127,
                    max_samples_per_term=max(1, int(args.semantic_vocab_symbol_samples_per_term)),
                )
                symbol_pool_bootstrap, symbol_pool_info_bootstrap = _build_internal_bootstrap_symbol_pool(
                    data_root=str(args.semantic_vocab_symbol_pool_root),
                    image_size=int(payload_size),
                    seed=int(args.seed) + 9769,
                    max_samples_per_term=max(1, int(args.semantic_vocab_symbol_samples_per_term)),
                    origin_label=str(args.semantic_vocab_bootstrap_origin_label),
                )
                bootstrap_symbol_terms = {
                    re.sub(r"\s+", " ", str(k)).strip().lower()
                    for k in symbol_pool_bootstrap.keys()
                    if str(k).strip()
                }
                symbol_pool_by_term = _merge_symbol_term_pools(
                    pools=[symbol_pool_official, symbol_pool_synth, symbol_pool_bootstrap],
                    max_samples_per_term=max(1, int(args.semantic_vocab_symbol_samples_per_term)),
                )
                symbol_pool_origin_terms_by_term = _build_symbol_term_origin_terms_map(
                    official_pool=symbol_pool_official,
                    synthetic_pool=symbol_pool_synth,
                    bootstrap_pool=symbol_pool_bootstrap,
                    bootstrap_origin_label=str(args.semantic_vocab_bootstrap_origin_label),
                )
                symbol_pool_info = {
                    "enabled": True,
                    "available_terms": int(len(symbol_pool_by_term)),
                    "samples": int(sum(len(v) for v in symbol_pool_by_term.values())),
                    "official_terms": int(symbol_pool_info_official.get("available_terms", 0)),
                    "official_samples": int(symbol_pool_info_official.get("samples", 0)),
                    "synthetic_terms": int(symbol_pool_info_synth.get("available_terms", 0)),
                    "synthetic_samples": int(symbol_pool_info_synth.get("samples", 0)),
                    "bootstrap_terms": int(symbol_pool_info_bootstrap.get("available_terms", 0)),
                    "bootstrap_samples": int(symbol_pool_info_bootstrap.get("samples", 0)),
                    "bootstrap_root_dir": str(symbol_pool_info_bootstrap.get("root_dir", "")),
                    "bootstrap_origin_label": str(symbol_pool_info_bootstrap.get("origin_label", "")),
                    "digits_terms": int(symbol_pool_info_official.get("digits_terms", 0)),
                    "letters_terms": int(symbol_pool_info_official.get("letters_terms", 0)),
                    "pictogram_terms": int(symbol_pool_info_official.get("pictogram_terms", 0)),
                    "origin_terms_mapped": int(len(symbol_pool_origin_terms_by_term)),
                    "reason": str(symbol_pool_info_official.get("reason", "")),
                }
                if int(symbol_pool_info.get("available_terms", 0)) > 0:
                    semantic_vocab_pool_terms = _normalize_vocab_terms(
                        list(semantic_vocab_pool_terms) + list(symbol_pool_by_term.keys())
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
                    f"bootstrap_origin={str(symbol_pool_info.get('bootstrap_origin_label', ''))} "
                    f"digits={int(symbol_pool_info.get('digits_terms', 0))} "
                    f"letters={int(symbol_pool_info.get('letters_terms', 0))} "
                    f"pictograms={int(symbol_pool_info.get('pictogram_terms', 0))} "
                    f"reason={str(symbol_pool_info.get('reason', ''))} "
                    f"bootstrap_dir={str(symbol_pool_info.get('bootstrap_root_dir', ''))}"
                )
            if need_payload_bank and len(payload_conditions) <= 0:
                if len(payload_conditions_supervised_base) <= 0:
                    raise RuntimeError("Semantic payload condition bank is empty.")
            if len(payload_conditions) != len(payload_images):
                payload_conditions = list(payload_conditions_supervised_base)
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
                f"mean_targets={train_target_info['mean_target_classes_per_stream']:.2f}, "
                f"mean_prob={train_target_info['mean_target_prob_after_imprint']:.4f}), "
                f"val_imprinted={val_target_info['imprinted_streams']}/{val_target_info['mix_streams']} "
                f"(targeted={val_target_info['target_supervised_streams']}, "
                f"mean_targets={val_target_info['mean_target_classes_per_stream']:.2f}, "
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
            if int(semantic_extra_count) <= 0 or len(symbol_pool_by_term) <= 0:
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
                if len(rows) <= 0:
                    continue
                matched += 1
                if len(rows) <= per_term:
                    picks = list(range(len(rows)))
                else:
                    picks = rng_sym.choice(np.arange(len(rows), dtype=np.int64), size=per_term, replace=False).astype(np.int64).tolist()
                for pi in picks:
                    img = np.asarray(rows[int(pi)], dtype=np.float32, copy=False)
                    images_out.append(img)
                    semantic_terms = _semantic_tags_for_symbol_term(key)
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
                        img = np.asarray(rows[int(pi)], dtype=np.float32, copy=False)
                        semantic_terms = _semantic_tags_for_symbol_term(key)
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
                "unknown_rows_added": int(unknown_rows),
                "unknown_terms_sampled": sorted([str(x) for x in unknown_term_seen]),
                "unknown_bootstrap_terms_sampled": sorted([str(x) for x in unknown_bootstrap_seen]),
                "unknown_other_terms_sampled": sorted([str(x) for x in unknown_other_seen]),
                "origin_row_counts": {str(k): int(v) for k, v in origin_row_counts.items()},
                "origin_terms_mapped_terms": int(len(symbol_pool_origin_terms_by_term)),
            }
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
            nonlocal class_names, label_embedding_bank, label_texts, label_embedding_info
            nonlocal fake_label_vector_t, payload_images, payload_conditions
            nonlocal payload_conditioning_info, payload_symbol_aug_info, payload_flashcard_info
            nonlocal active_gd_vocab_hash, active_gd_vocab_profile
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
                )
            regurgitated_churn_last_info = dict(regurg_info)
            active_semantic_names = list(supervised_class_names) + list(active_extra_terms)
            class_names = list(active_semantic_names)
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
                            np.asarray(payload_conditions[i], dtype=np.float32, copy=False),
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
                )
                if len(flash_images) > 0 and len(flash_images) == len(flash_conds):
                    payload_images.extend(flash_images)
                    payload_conditions.extend(flash_conds)

            active_gd_vocab_hash, active_gd_vocab_profile = _compute_gd_vocab_hash(
                supervised_class_names=supervised_class_names,
                active_extra_terms=active_extra_terms,
                condition_num_classes=int(condition_num_classes),
                args=args,
            )
            row = {
                "cycle": int(cycle_id),
                "cycle_local": int(cycle_local),
                "global_round": int(global_round),
                "reason": str(reason),
                "allow_churn": bool(allow_churn),
                "churn_changed": bool(churn_info.get("changed", False)),
                "churn_replaced": int(churn_info.get("replaced", 0)),
                "active_extra_terms": list(active_extra_terms),
                "vocab_hash": str(active_gd_vocab_hash),
                "payload_rows": int(len(payload_conditions)),
                "payload_images": int(len(payload_images)),
                "symbol_rows_added": int(payload_symbol_aug_info.get("rows_added", 0)),
                "symbol_unknown_rows_added": int(payload_symbol_aug_info.get("unknown_rows_added", 0)),
                "symbol_origin_row_counts": dict(payload_symbol_aug_info.get("origin_row_counts", {})),
                "symbol_unknown_terms_sampled": list(payload_symbol_aug_info.get("unknown_terms_sampled", [])),
                "flashcard_rows_added": int(payload_flashcard_info.get("rows_added", 0)),
                "regurgitated_churn": dict(regurg_info),
                "regurgitated_terms_added": int(len(regurg_terms_used)),
            }
            semantic_churn_history.append(row)
            _log(
                "[semantic-vocab] active set: "
                f"cycle={int(cycle_id)} reason={str(reason)} "
                f"extras={int(len(active_extra_terms))} changed={1 if bool(churn_info.get('changed', False)) else 0} "
                f"replaced={int(churn_info.get('replaced', 0))} "
                f"payload={int(len(payload_conditions))} symbols={int(payload_symbol_aug_info.get('rows_added', 0))} "
                f"unknown_symbols={int(payload_symbol_aug_info.get('unknown_rows_added', 0))} "
                f"flashcards={int(payload_flashcard_info.get('rows_added', 0))} "
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

        def _rebuild_semantic_gate_loaders(reason: str, cycle_seed: int):
            nonlocal gestation_gate_loader, gestation_gate_loader_count, gestation_gate_info
            nonlocal berkeley_gate_loader, berkeley_gate_loader_count, berkeley_gate_schedule_info
            nonlocal gestation_gate_samples_seen, gestation_gate_samples_required
            nonlocal total_gate_samples_seen, total_gate_samples_required
            if int(condition_num_classes) <= 0:
                gestation_gate_loader = None
                berkeley_gate_loader = None
                gestation_gate_loader_count = 0
                berkeley_gate_loader_count = 0
                gestation_gate_info = {"rows": 0, "selected_rows": 0, "reason": "condition_num_classes<=0"}
                berkeley_gate_schedule_info = {"rows": 0, "selected_rows": 0, "reason": "condition_num_classes<=0"}
                gestation_gate_samples_seen = 0
                gestation_gate_samples_required = 0
                total_gate_samples_seen = 0
                total_gate_samples_required = 0
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
            for term_key in sorted([str(x) for x in bootstrap_symbol_terms], key=lambda s: s.lower()):
                rows = symbol_pool_by_term.get(str(term_key), [])
                if len(rows) <= 0:
                    continue
                origin_terms = _normalize_vocab_terms(
                    [
                        "symbol pool bootstrap",
                        "internal bootstrap material",
                        str(origin_label),
                        f"{str(origin_label)}::{str(term_key)}",
                    ]
                )
                semantic_terms = _semantic_tags_for_symbol_term(str(term_key))
                for row_img in rows:
                    gestation_images.append(_image_any_to_rgb_chw01(row_img, image_size=int(shared_embed_image_size)))
                    gestation_targets.append(
                        np.asarray(
                            _semantic_condition_from_terms_runtime(
                                terms=semantic_terms,
                                origin_terms=origin_terms,
                            ),
                            dtype=np.float32,
                        ).reshape(-1)
                    )
            if len(gestation_images) <= 0:
                fallback_bootstrap, _ = _build_internal_bootstrap_symbol_pool(
                    data_root=str(args.semantic_vocab_symbol_pool_root),
                    image_size=int(shared_embed_image_size),
                    seed=int(cycle_seed) + 149,
                    max_samples_per_term=max(1, int(args.semantic_vocab_symbol_samples_per_term)),
                    origin_label=str(args.semantic_vocab_bootstrap_origin_label),
                )
                for term_key in sorted([str(x) for x in fallback_bootstrap.keys()], key=lambda s: s.lower()):
                    rows = fallback_bootstrap.get(str(term_key), [])
                    if len(rows) <= 0:
                        continue
                    origin_terms = _normalize_vocab_terms(
                        [
                            "symbol pool bootstrap",
                            "internal bootstrap material",
                            str(origin_label),
                            f"{str(origin_label)}::{str(term_key)}",
                        ]
                    )
                    semantic_terms = _semantic_tags_for_symbol_term(str(term_key))
                    for row_img in rows:
                        gestation_images.append(_image_any_to_rgb_chw01(row_img, image_size=int(shared_embed_image_size)))
                        gestation_targets.append(
                            np.asarray(
                                _semantic_condition_from_terms_runtime(
                                    terms=semantic_terms,
                                    origin_terms=origin_terms,
                                ),
                                dtype=np.float32,
                            ).reshape(-1)
                        )
            gestation_gate_loader, gestation_gate_loader_count = _build_gate_loader_from_arrays(
                images=gestation_images,
                targets=gestation_targets,
                batch_size=int(gestation_batch),
                num_workers=max(0, int(args.berkeley_refresh_workers)),
                device=device,
                seed=int(cycle_seed) + 1777,
                max_samples=int(args.gate_gestation_val_max),
                ordered_indices=None,
                persistent_workers=bool(args.loader_persistent_workers),
                prefetch_factor=int(args.loader_prefetch_factor),
            )
            gestation_gate_info = {
                "rows": int(len(gestation_images)),
                "selected_rows": int(gestation_gate_loader_count),
                "batch_size": int(gestation_batch),
                "reason": str(reason),
            }
            gestation_gate_samples_seen = 0
            gestation_gate_samples_required = int(gestation_gate_loader_count)

            total_images: List[np.ndarray] = [np.asarray(x, dtype=np.float32, copy=False) for x in payload_images]
            total_targets: List[np.ndarray] = [np.asarray(y, dtype=np.float32, copy=False).reshape(-1) for y in payload_conditions]
            total_images.extend([np.asarray(x, dtype=np.float32, copy=False) for x in gestation_images])
            total_targets.extend([np.asarray(y, dtype=np.float32, copy=False).reshape(-1) for y in gestation_targets])
            total_order: Optional[List[int]] = None
            berkeley_gate_schedule_info = {
                "rows": int(len(total_targets)),
                "selected_rows": int(len(total_targets)),
                "active_terms": 0,
                "reason": str(reason),
            }
            if bool(args.gate_total_token_schedule_enabled):
                total_order, berkeley_gate_schedule_info = _schedule_semantic_gate_indices(
                    targets=total_targets,
                    class_names=class_names,
                    active_terms=active_extra_terms,
                    seed=int(cycle_seed) + 1931,
                    max_samples=int(args.gate_berkeley_val_max),
                    activation_threshold=float(args.gate_total_token_schedule_threshold),
                )
                berkeley_gate_schedule_info = dict(berkeley_gate_schedule_info)
                berkeley_gate_schedule_info["reason"] = str(reason)
            berkeley_gate_loader, berkeley_gate_loader_count = _build_gate_loader_from_arrays(
                images=total_images,
                targets=total_targets,
                batch_size=int(gate_loader_batch),
                num_workers=max(0, int(args.berkeley_refresh_workers)),
                device=device,
                seed=int(cycle_seed) + 2017,
                max_samples=int(args.gate_berkeley_val_max),
                ordered_indices=total_order,
                persistent_workers=bool(args.loader_persistent_workers),
                prefetch_factor=int(args.loader_prefetch_factor),
            )
            total_gate_samples_seen = 0
            total_gate_samples_required = int(berkeley_gate_loader_count)
            _log(
                "[semantic-gate-loader] "
                f"reason={str(reason)} "
                f"gestation={int(gestation_gate_loader_count)}/{int(len(gestation_images))} "
                f"total={int(berkeley_gate_loader_count)}/{int(len(total_targets))} "
                f"active_terms={int(berkeley_gate_schedule_info.get('active_terms', 0))} "
                f"bucketed={int(berkeley_gate_schedule_info.get('bucketed_rows', 0))}"
            )

        _rebuild_semantic_gate_loaders(reason="startup", cycle_seed=int(args.seed))

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
                "payload_images_active_count": int(len(payload_images)),
                "payload_conditions_active_count": int(len(payload_conditions)),
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
                    transformer, stage_hist = train_transformer_feature_metric(
                        transformer=transformer,
                        classifier=classifier,
                        train_streams=train_streams,
                        train_target_labels=train_stream_target_labels,
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

            total_enabled = bool(
                float(gate_config.get("berkeley_min_confidence", 0.0)) > 0.0
                or float(gate_config.get("berkeley_min_macro_f1", 0.0)) > 0.0
                or float(gate_config.get("berkeley_loss_target", 0.0)) > 0.0
            )
            total_eval = None
            if bool(gestation_ready) and bool(total_enabled) and berkeley_gate_loader is not None:
                _cleanup_cuda_allocator()
                total_eval = _evaluate_berkeley_classifier_gate(
                    classifier=classifier,
                    loader=berkeley_gate_loader,
                    device=device,
                    max_steps=int(args.gate_berkeley_eval_max_steps),
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
            return {
                "gestation_eval": gestation_eval,
                "gestation_detail": dict(gestation_detail),
                "gestation_pass": bool(gestation_pass),
                "gestation_ready": bool(gestation_ready),
                "total_eval": total_eval,
                "total_detail": dict(total_detail),
                "total_pass": bool(total_pass),
                "total_ready": bool(total_ready),
            }

        initial_gate_eval = _evaluate_two_stage_semantic_gates(round_index=max(0, int(len(orchestration_rows))))
        initial_gestation_eval = initial_gate_eval.get("gestation_eval", None)
        initial_gestation_detail = dict(initial_gate_eval.get("gestation_detail", {}))
        initial_gestation_pass = bool(initial_gate_eval.get("gestation_pass", True))
        initial_berkeley_eval = initial_gate_eval.get("total_eval", None)
        initial_berkeley_gate_detail = dict(initial_gate_eval.get("total_detail", {}))
        initial_berkeley_gate_pass = bool(initial_gate_eval.get("total_pass", True))
        _log(
            "Initial gestation gate: "
            f"pass={1 if initial_gestation_pass else 0} "
            f"reason={initial_gestation_detail.get('reason', 'pass')} "
            f"loss={(float(initial_gestation_eval.get('loss', float('nan'))) if isinstance(initial_gestation_eval, dict) else float('nan')):.4f} "
            f"target={initial_gestation_detail.get('loss_target', None)} "
            f"coverage={int(initial_gestation_detail.get('refresh_samples_seen', 0))}/"
            f"{int(initial_gestation_detail.get('refresh_samples_required', 0))}"
        )
        conf_init = None if (initial_berkeley_eval is None) else initial_berkeley_eval.get("mean_confidence", None)
        f1_init = None if (initial_berkeley_eval is None) else initial_berkeley_eval.get("macro_f1", None)
        loss_init = None if (initial_berkeley_eval is None) else initial_berkeley_eval.get("loss", None)
        _log(
            "Initial total-dataset gate: "
            f"pass={1 if initial_berkeley_gate_pass else 0} "
            f"mode={initial_berkeley_gate_detail.get('mode', 'hard_loss_full_refresh')} "
            f"reason={initial_berkeley_gate_detail.get('reason', 'pass')} "
            f"conf={(float(conf_init) if conf_init is not None else float('nan')):.4f} "
            f"macro_f1={(float(f1_init) if f1_init is not None else float('nan')):.4f} "
            f"loss={(float(loss_init) if loss_init is not None else float('nan')):.4f} "
            f"loss_target={initial_berkeley_gate_detail.get('loss_target', None)} "
            f"coverage={int(initial_berkeley_gate_detail.get('refresh_samples_seen', 0))}/"
            f"{int(initial_berkeley_gate_detail.get('refresh_samples_required', 0))} "
            f"token_schedule_rows={int(berkeley_gate_schedule_info.get('selected_rows', 0))}"
        )
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
                if len(payload_images) > 0:
                    pick = int(max(0, int(args.seed)) % len(payload_images))
                    stage_preview_cursor["C"] = (pick + 1) % len(payload_images)
                    img_np = np.asarray(payload_images[pick], dtype=np.float32)
                    x_boot = torch.from_numpy(img_np)
                    target_line = "target:none"
                    if pick < len(payload_conditions):
                        target_line = _format_target_line_from_condition(
                            payload_conditions[pick],
                            class_names=class_names,
                            max_items=4,
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
                        max_items=4,
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
                            max_items=4,
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
                            max_items=4,
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
            if bool(stage_opengl_step_driven):
                do_gl = False
            if (not do_file) and (not do_gl):
                return

            stage_key = str(stage).strip().upper()

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

            if do_gl and stage_opengl_viewer.enabled and (stage_key == "C") and (len(payload_images) > 0):
                try:
                    pick = _next_stage_index(
                        "C",
                        len(payload_images),
                        seed_term=(int(args.seed) + (int(cycle_id) * 2713) + (int(round_id) * 337) + int(global_round)),
                    )
                    img_np = np.asarray(payload_images[pick], dtype=np.float32)
                    x0 = torch.from_numpy(img_np)
                    x = torch.from_numpy(img_np[None, ...]).to(device=device, dtype=torch.float32)
                    if bool(args.channels_last):
                        x = x.contiguous(memory_format=torch.channels_last)
                    with torch.no_grad():
                        probs = torch.sigmoid(classifier(x)).to(torch.float32)
                    p0 = probs[0].detach().cpu().numpy().astype(np.float32, copy=False)
                    top = _top_labels_from_probs(p0, class_names=class_names, topk=max(1, min(len(class_names), int(p0.shape[0]))))
                    top_lines = _format_top_label_lines(top, max_items=0)
                    target_line = "target:none"
                    if pick < len(payload_conditions):
                        target_line = _format_target_line_from_condition(
                            payload_conditions[pick],
                            class_names=class_names,
                            max_items=4,
                            threshold=0.5,
                        )
                    in_rows = top_lines if len(top_lines) > 0 else ["n/a"]
                    out_rows = [target_line] + in_rows
                    top_txt = "n/a" if len(top_lines) <= 0 else top_lines[0]
                    stage_opengl_viewer.update(
                        clean_img=x0,
                        input_img=x0,
                        output_img=x0,
                        caption=f"[C] cycle={int(cycle_id)} round={int(round_id)} sample={int(pick)} top={top_txt}",
                        panel_titles=["C target", "C classifier", "C summary"],
                        panel_rows=[[target_line], in_rows, out_rows],
                    )
                    _log(f"[stage-opengl] C sample={int(pick)} top={top_txt}")
                except Exception as e:
                    _log(f"[stage-opengl] classifier preview failed: {e}")

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
                        max_items=4,
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
                            max_items=4,
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
                    "Transformer training skipped by total-dataset gate "
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
                    bank_np=split_label_embedding_bank,
                    args=args,
                )
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
                    if (
                        wave_zero_shot_query_emb is not None
                        and len(wave_zero_shot_queries) > 0
                    ):
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

                    refresh_only_loss_mode = bool(berkeley_loss_target_value > 0.0 and bool(berkeley_loss_target_unmet))
                    did_refresh_round = False
                    run_refresh_this_round = (
                        refresh_loader is not None
                        and (
                            (
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
                        if need_full_refresh_startup or need_full_refresh_loss:
                            min_refresh_steps = max(1, int(len(refresh_loader)))
                        else:
                            min_refresh_steps = 0
                        refresh_max_steps = int(args.berkeley_refresh_max_steps)
                        if (need_full_refresh_startup or need_full_refresh_loss) and int(min_refresh_steps) > 0:
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
                            _run_transformer_round_preview(
                                stage="C",
                                cycle_id=int(cycle_id),
                                round_id=int(round_id),
                                global_round=int(global_round),
                                extra={"downstream_skipped": "berkeley_loss_refresh_only"},
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
                        _run_transformer_round_preview(
                            stage="C",
                            cycle_id=int(cycle_id),
                            round_id=int(round_id),
                            global_round=int(global_round),
                            extra={"downstream_skipped": str(round_row.get("downstream_skipped", ""))},
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
                bank_np=split_label_embedding_bank,
                args=args,
            )
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

                    refresh_only_loss_mode = bool(berkeley_loss_target_value > 0.0 and bool(berkeley_loss_target_unmet))
                    did_refresh_round = False
                    run_refresh_this_round = (
                        refresh_loader is not None
                        and (
                            (
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
                        if need_full_refresh_startup or need_full_refresh_loss:
                            min_refresh_steps = max(1, int(len(refresh_loader)))
                        else:
                            min_refresh_steps = 0
                        refresh_max_steps = int(args.berkeley_refresh_max_steps)
                        if (need_full_refresh_startup or need_full_refresh_loss) and int(min_refresh_steps) > 0:
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
                                stage="C",
                                cycle_id=int(cycle_id),
                                round_id=int(round_id),
                                global_round=int(global_round),
                                extra={"downstream_skipped": "berkeley_loss_refresh_only"},
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
                        _run_transformer_round_preview(
                            stage="C",
                            cycle_id=int(cycle_id),
                            round_id=int(round_id),
                            global_round=int(global_round),
                            extra={"downstream_skipped": str(round_row.get("downstream_skipped", ""))},
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
                        if (
                            wave_zero_shot_query_emb is not None
                            and len(wave_zero_shot_queries) > 0
                        ):
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
                                max_items=16,
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
