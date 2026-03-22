#!/usr/bin/env python3
"""
nodus_server.py -- HTTP server that exposes the nodus shared-memory data stores.

Connects to the same cross-process shared memory as the training pipeline and
serves the data as JSON / PNG over HTTP.  Most endpoints are read-only; the
server only writes lightweight runtime-control flags used to prioritize web
work over training and to honor cooperative shutdown requests.

Usage:
    python nodus_server.py [--host 127.0.0.1] [--port 7272]

Endpoints:
    GET /                                   HTML index / API listing
    GET /api/status                         JSON  overall store status
    GET /api/model/interface/meta          JSON  unified lease/latest model descriptor
    GET /api/loss/channels                  JSON  all loss channels + summary
    GET /api/loss/channel/<key>/records     JSON  records (query: from_step, max)
    GET /api/loss/channel/<key>/latest      JSON  most recent record
    GET /api/loss/graph.png                 PNG   rendered loss graph
                                                  (query: w, h, start, bg)
    GET /api/scrub/info                     JSON  scrub ring info
    GET /api/scrub/latest/meta              JSON  newest entry metadata
    GET /api/scrub/latest/training.png      PNG   newest training image
    GET /api/scrub/latest/output.png        PNG   newest output image
    GET /api/scrub/latest/target.png        PNG   newest target image
    GET /api/scrub/<idx>/meta               JSON  entry metadata by index
    GET /api/scrub/<idx>/training.png       PNG   training image by index
    GET /api/scrub/<idx>/output.png         PNG   output image by index
    GET /api/scrub/<idx>/target.png         PNG   target image by index
    GET /api/weight/meta                    JSON  latest weight-state snapshot
    GET /api/weight/meta/all                JSON  all retained weight-state entries
    GET /api/weight/image/info              JSON  image-cache stats
    GET /api/weight/image/latest.png        PNG   newest rendered weight image
    GET /api/weight/image/<idx>.png         PNG   weight image by index
    GET /api/runtime/control                JSON  runtime-control flags
    GET /api/model/latest/classifier/meta   JSON  latest classifier labels + input shape
    POST /api/model/latest/classifier/infer JSON  latest classifier inference on an uploaded image
    POST /api/model/interface/prepare       JSON  unified prepare/export for lease/latest models
    POST /api/model/interface/infer         JSON  unified classifier inference for lease/latest models
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import dataclasses
import io
import json
import re
import struct
import sys
import threading
import time
import zlib
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlparse

# ---------------------------------------------------------------------------
# Minimal stdlib PNG encoder (no Pillow dependency)
# ---------------------------------------------------------------------------

def _png_chunk(tag: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def _encode_png_rgb(width: int, height: int, rgb: bytes) -> bytes:
    """Encode raw RGB bytes (width*height*3) as a PNG file."""
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    stride = width * 3
    raw = b"".join(b"\x00" + rgb[y * stride:(y + 1) * stride] for y in range(height))
    idat = _png_chunk(b"IDAT", zlib.compress(raw, 1))
    iend = _png_chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


def _encode_png_rgba(width: int, height: int, rgba: bytes) -> bytes:
    """Encode raw RGBA bytes (width*height*4) as a PNG file."""
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
    stride = width * 4
    raw = b"".join(b"\x00" + rgba[y * stride:(y + 1) * stride] for y in range(height))
    idat = _png_chunk(b"IDAT", zlib.compress(raw, 1))
    iend = _png_chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


def _composite_rgba_on_black(width: int, height: int, rgba: bytes) -> bytes:
    """Composite RGBA overlay onto a black background, return RGB bytes."""
    import array
    out = array.array("B", b"\x00" * (width * height * 3))
    for i in range(width * height):
        r, g, b, a = rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2], rgba[i * 4 + 3]
        if a:
            alpha = a / 255.0
            out[i * 3]     = int(r * alpha)
            out[i * 3 + 1] = int(g * alpha)
            out[i * 3 + 2] = int(b * alpha)
    return bytes(out)


# ---------------------------------------------------------------------------
# Lazy store accessors
# ---------------------------------------------------------------------------

def _get_stores():
    """Return (loss_store, scrub_ring, weight_state_store, weight_image_store, runtime_control).

    Imported lazily so the module can be imported without the DLL present.
    Stores are the cross-process global singletons.
    """
    from pipeline.nodus_loss_store import (
        NodusLossStore,
        NodusRuntimeControlStore,
        NodusScrubRing,
        NodusWeightStateStore,
        NodusWeightImageStore,
    )
    loss   = NodusLossStore.get_global()
    scrub  = NodusScrubRing.get_global()
    wstate = NodusWeightStateStore.get_global()
    wimage = NodusWeightImageStore.get_global()
    control = NodusRuntimeControlStore.get_global()
    return loss, scrub, wstate, wimage, control


# Cached after first successful load.
_STORES: tuple | None = None

# ---------------------------------------------------------------------------
# Lease store (optional — enabled via --lease-store-dir)
# ---------------------------------------------------------------------------

_LEASE_STORE = None   # type: Optional["pipeline.lease_store.LeaseStore"]
_LEASE_STORE_LOCK = threading.Lock()
_WEB_DATASET = None   # type: Optional["pipeline.web_dataset.WebDataset"]

# ---------------------------------------------------------------------------
# Checkpoint directory (optional — enables /api/model/latest/*)
# ---------------------------------------------------------------------------

_CHECKPOINT_DIR: Optional[str] = None
# Cache: model_name → (mtime, onnx_bytes)
_ONNX_CACHE: dict = {}
_ONNX_CACHE_LOCK = threading.Lock()
_LATEST_CLASSIFIER_CACHE: dict = {}
_LATEST_CLASSIFIER_CACHE_LOCK = threading.Lock()

_LATEST_MODEL_ALLOWED = {
    "classifier", "transformer", "generator", "discriminator", "wave_classifier",
}
_MODEL_CLASS_TO_NAME = {
    "TinyConvClassifier": "classifier",
    "ConditionalBitPlaneGenerator": "generator",
    "ConditionalBitPlaneDiscriminator": "discriminator",
    "WavePatchTransformer": "transformer",
    "DeskewFilterBundle": "wave_classifier",
}
_MODEL_NAME_TO_CLASS = {value: key for key, value in _MODEL_CLASS_TO_NAME.items()}
_ALLOWED_MODEL_CLASSES = set(_MODEL_CLASS_TO_NAME.keys())


def _lease_store():
    return _LEASE_STORE


def _web_dataset():
    return _WEB_DATASET


def _stores():
    global _STORES
    if _STORES is None:
        _STORES = _get_stores()
    return _STORES


def _loss():
    return _stores()[0]


def _scrub():
    return _stores()[1]


def _wstate():
    return _stores()[2]


def _wimage():
    return _stores()[3]


def _control():
    return _stores()[4]


def _model_name_for_class(model_class: str) -> str:
    return _MODEL_CLASS_TO_NAME.get(str(model_class or "").strip(), "")


def _model_class_for_name(model_name: str) -> str:
    return _MODEL_NAME_TO_CLASS.get(str(model_name or "").strip(), "")


def _normalise_model_selection(model_name: str = "", model_class: str = "") -> tuple[str, str]:
    resolved_name = str(model_name or "").strip()
    resolved_class = str(model_class or "").strip()
    if resolved_class and resolved_class not in _ALLOWED_MODEL_CLASSES:
        raise ValueError(f"model_class must be one of {sorted(_ALLOWED_MODEL_CLASSES)}")
    if resolved_name and resolved_name not in _LATEST_MODEL_ALLOWED:
        raise ValueError(f"model_name must be one of {sorted(_LATEST_MODEL_ALLOWED)}")
    if not resolved_name and resolved_class:
        resolved_name = _model_name_for_class(resolved_class)
    if not resolved_class and resolved_name:
        resolved_class = _model_class_for_name(resolved_name)
    if not resolved_name and not resolved_class:
        resolved_name = "classifier"
        resolved_class = _model_class_for_name(resolved_name)
    if not resolved_name:
        raise ValueError("could not infer model_name from model_class")
    if not resolved_class:
        raise ValueError("could not infer model_class from model_name")
    return resolved_name, resolved_class


def _normalise_input_shape(raw: object, default: object = None) -> list[int]:
    fallback = list(default) if isinstance(default, (list, tuple)) else [1, 3, 64, 64]
    if isinstance(raw, str):
        parts = [p.strip() for p in str(raw).split(",") if p.strip()]
        values = [int(p) for p in parts] if parts else list(fallback)
    elif isinstance(raw, (list, tuple)):
        values = [int(x) for x in list(raw)]
    else:
        values = list(fallback)
    if len(values) != 4:
        values = list(fallback)
    if int(values[0]) <= 0:
        values[0] = 1
    if int(values[1]) <= 0:
        values[1] = 3
    if int(values[2]) <= 0:
        values[2] = 64
    if int(values[3]) <= 0:
        values[3] = 64
    return [int(x) for x in values]


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _json(obj) -> bytes:
    return json.dumps(obj, allow_nan=False).encode()


def _dataclass_dict(obj) -> dict:
    return dataclasses.asdict(obj)


def _runtime_control_state_dict() -> dict:
    try:
        state = _control().get_state()
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    if state is None:
        return {"ok": False, "error": "runtime control state unavailable"}
    return {
        "ok": True,
        "active_service_count": int(getattr(state, "active_service_count", 0)),
        "service_enter_count": int(getattr(state, "service_enter_count", 0)),
        "service_exit_count": int(getattr(state, "service_exit_count", 0)),
        "exit_requested": bool(getattr(state, "exit_requested", False)),
        "last_service_ts": float(getattr(state, "last_service_ts", 0.0)),
        "exit_ts": float(getattr(state, "exit_ts", 0.0)),
        "last_source": str(getattr(state, "last_source", "") or ""),
        "exit_reason": str(getattr(state, "exit_reason", "") or ""),
    }


@contextmanager
def _service_priority(source: str):
    control = None
    try:
        control = _control()
    except Exception:
        control = None
    token = str(source or "service")
    if control is not None:
        try:
            control.begin_service(token)
        except Exception:
            control = None
    try:
        yield
    finally:
        if control is not None:
            try:
                control.end_service(token)
            except Exception:
                pass


def _extract_checkpoint_state_dict(blob: object, preferred_keys: tuple[str, ...] = ()) -> dict:
    if isinstance(blob, dict):
        for key in preferred_keys:
            candidate = blob.get(key)
            if isinstance(candidate, dict):
                blob = candidate
                break
        else:
            blob = blob.get("state_dict", blob)
    if isinstance(blob, dict) and isinstance(blob.get("model_state_dict"), dict):
        blob = blob["model_state_dict"]
    if not isinstance(blob, dict):
        raise RuntimeError("checkpoint does not contain a state_dict payload")
    return dict(blob)


def _normalise_label_list(raw: object, count: int, prefix: str) -> list[str]:
    items: list[str]
    if isinstance(raw, (list, tuple)):
        items = [str(x) for x in list(raw)]
    elif isinstance(raw, dict):
        tmp = ["" for _ in range(max(0, int(count)))]
        for key, value in raw.items():
            try:
                idx = int(key)
            except Exception:
                continue
            if 0 <= idx < len(tmp):
                tmp[idx] = str(value)
        items = tmp
    else:
        items = []
    items = [str(x).strip() for x in items]
    target = max(1, int(count))
    if len(items) < target:
        items.extend(f"{prefix}_{i}" for i in range(len(items), target))
    if len(items) > target:
        items = items[:target]
    return items


def _infer_tiny_classifier_ctor_kwargs(state_dict: dict, fallback_num_classes: int) -> dict:
    import torch

    def _shape(key: str) -> tuple[int, ...]:
        value = state_dict.get(key)
        if torch.is_tensor(value):
            return tuple(int(x) for x in value.shape)
        return ()

    c0 = _shape("features.0.weight")
    c1 = _shape("features.4.weight")
    c2 = _shape("features.8.weight")
    c3 = _shape("features.12.weight")
    head = _shape("head.2.weight")
    bank = _shape("label_embed_bank")
    mask = _shape("mask_head.0.weight")

    context_indices = set()
    for key in state_dict.keys():
        m = re.match(r"features\.(\d+)\.conv1\.weight$", str(key))
        if m:
            context_indices.add(int(m.group(1)))

    num_classes = int(fallback_num_classes)
    if len(head) >= 1 and int(head[0]) > 0:
        num_classes = int(head[0])
    if len(bank) >= 1 and int(bank[0]) > 0:
        num_classes = max(int(num_classes), int(bank[0]))
    num_classes = max(1, int(num_classes))

    base_ch = int(c0[0]) if len(c0) >= 1 else 64
    max_ch = max(
        int(base_ch),
        int(c1[0]) if len(c1) >= 1 else int(base_ch),
        int(c2[0]) if len(c2) >= 1 else int(base_ch),
        int(c3[0]) if len(c3) >= 1 else int(base_ch),
    )
    return {
        "num_classes": int(num_classes),
        "base_ch": int(base_ch),
        "max_ch": int(max_ch),
        "context_blocks": int(len(context_indices)),
        "context_dropout": 0.05,
        "mask_decoder_channels": int(mask[0]) if len(mask) >= 1 else 0,
    }


def _checkpoint_signature(paths: list[object]) -> tuple:
    sig = []
    for path in paths:
        if path is None:
            continue
        try:
            if path.exists():
                st = path.stat()
                sig.append((str(path), int(st.st_mtime_ns), int(st.st_size)))
        except Exception:
            continue
    return tuple(sig)


def _load_model_from_checkpoint_blob(checkpoint: object, model_class_name: str):
    import torch
    import wav_ml_models as _models
    from wav_ml_models import prime_tiny_classifier_label_bank_for_state_dict

    if model_class_name not in _ALLOWED_MODEL_CLASSES:
        raise ValueError(f"model_class must be one of {sorted(_ALLOWED_MODEL_CLASSES)}")

    cls = getattr(_models, model_class_name)
    ctor_kwargs = {}
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("ctor_kwargs"), dict):
        ctor_kwargs = dict(checkpoint.get("ctor_kwargs") or {})
    state_dict = _extract_checkpoint_state_dict(checkpoint, preferred_keys=("state_dict",))

    if model_class_name == "TinyConvClassifier" and not ctor_kwargs:
        fallback_num_classes = int(checkpoint.get("num_classes", 0) or 0) if isinstance(checkpoint, dict) else 0
        ctor_kwargs = _infer_tiny_classifier_ctor_kwargs(state_dict, fallback_num_classes)

    if ctor_kwargs:
        model = cls(**ctor_kwargs)
    else:
        try:
            model = cls()
        except TypeError as exc:
            raise RuntimeError(
                f"Cannot auto-instantiate {model_class_name}; include ctor_kwargs in the checkpoint or request"
            ) from exc

    if model_class_name == "TinyConvClassifier":
        temperature = float(checkpoint.get("label_embedding_temperature", 10.0) or 10.0) if isinstance(checkpoint, dict) else 10.0
        prime_tiny_classifier_label_bank_for_state_dict(model, state_dict, temperature=temperature)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, state_dict, dict(ctor_kwargs)


def _build_classifier_bundle(
    *,
    cache_key: str,
    signature: tuple,
    classifier_blob: object,
    pipeline_blob: object,
    checkpoint_dir: str,
    classifier_path: str,
    pipeline_checkpoint_path: str = "",
    source: str = "latest",
    lease_id: str = "",
    collection_id: str = "",
    slot_name: str = "",
    generation: int = 0,
) -> dict:
    import torch
    from wav_ml_models import TinyConvClassifier, prime_tiny_classifier_label_bank_for_state_dict

    with _LATEST_CLASSIFIER_CACHE_LOCK:
        cached = _LATEST_CLASSIFIER_CACHE.get(cache_key)
        if cached is not None and cached.get("signature") == signature:
            return cached["bundle"]

        if classifier_blob:
            state_dict = _extract_checkpoint_state_dict(classifier_blob, preferred_keys=("state_dict",))
        else:
            state_dict = _extract_checkpoint_state_dict(pipeline_blob, preferred_keys=("classifier_state",))

        meta_blob: dict = {}
        if isinstance(pipeline_blob, dict):
            meta_blob.update(pipeline_blob)
        if isinstance(classifier_blob, dict):
            meta_blob.update(classifier_blob)

        num_classes = int(meta_blob.get("num_classes", 0) or 0)
        if num_classes <= 0:
            ctor_probe = _infer_tiny_classifier_ctor_kwargs(state_dict, 1)
            num_classes = int(ctor_probe.get("num_classes", 1))
        ctor_kwargs = meta_blob.get("ctor_kwargs")
        if not isinstance(ctor_kwargs, dict):
            ctor_kwargs = _infer_tiny_classifier_ctor_kwargs(state_dict, num_classes)
        ctor_kwargs = dict(ctor_kwargs)
        ctor_kwargs["num_classes"] = max(1, int(ctor_kwargs.get("num_classes", num_classes)))
        num_classes = int(ctor_kwargs["num_classes"])

        model = TinyConvClassifier(**ctor_kwargs)
        prime_tiny_classifier_label_bank_for_state_dict(
            model,
            state_dict,
            temperature=float(meta_blob.get("label_embedding_temperature", 10.0) or 10.0),
        )
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        class_names = _normalise_label_list(meta_blob.get("class_names"), num_classes, "class")
        label_texts = _normalise_label_list(meta_blob.get("label_texts", class_names), num_classes, "label")
        input_shape = _normalise_input_shape(meta_blob.get("input_shape"), [1, 3, 64, 64])

        bundle = {
            "source": str(source or "latest"),
            "model_name": "classifier",
            "model_class": "TinyConvClassifier",
            "model": model,
            "class_names": class_names,
            "label_texts": label_texts,
            "input_shape": input_shape,
            "checkpoint_dir": str(checkpoint_dir or ""),
            "classifier_path": str(classifier_path or ""),
            "pipeline_checkpoint_path": str(pipeline_checkpoint_path or ""),
            "mtime": float(max((item[1] for item in signature), default=0) / 1_000_000_000.0),
            "num_classes": int(num_classes),
            "ctor_kwargs": dict(ctor_kwargs),
            "lease_id": str(lease_id or ""),
            "collection_id": str(collection_id or ""),
            "slot_name": str(slot_name or ""),
            "generation": int(generation or 0),
        }
        _LATEST_CLASSIFIER_CACHE[cache_key] = {"signature": signature, "bundle": bundle}
        return bundle


def _load_latest_classifier_bundle() -> dict:
    if not _CHECKPOINT_DIR:
        raise RuntimeError("checkpoint dir not configured")

    import pathlib
    import torch

    ckpt_dir = pathlib.Path(_CHECKPOINT_DIR)
    classifier_path = ckpt_dir / "classifier.pt"
    pipeline_path = ckpt_dir / "pipeline_checkpoint.pt"
    if not classifier_path.exists() and not pipeline_path.exists():
        raise FileNotFoundError("classifier.pt or pipeline_checkpoint.pt not found")

    classifier_blob = (
        torch.load(str(classifier_path), map_location="cpu", weights_only=False)
        if classifier_path.exists()
        else {}
    )
    pipeline_blob = (
        torch.load(str(pipeline_path), map_location="cpu", weights_only=False)
        if pipeline_path.exists()
        else {}
    )
    signature = _checkpoint_signature([classifier_path, pipeline_path])
    return _build_classifier_bundle(
        cache_key="latest",
        signature=signature,
        classifier_blob=classifier_blob,
        pipeline_blob=pipeline_blob,
        checkpoint_dir=str(ckpt_dir),
        classifier_path=str(classifier_path if classifier_path.exists() else pipeline_path),
        pipeline_checkpoint_path=str(pipeline_path) if pipeline_path.exists() else "",
        source="latest",
    )


def _lease_model_artifacts(lease_id: str) -> dict:
    import pathlib

    lease_id = str(lease_id or "").strip()
    if not lease_id:
        raise ValueError("lease_id required")
    ls = _lease_store()
    if ls is None:
        raise RuntimeError("lease store not configured")
    lease = ls.get_lease(lease_id)
    if lease is None:
        raise FileNotFoundError("lease not found")
    weights_dir = pathlib.Path(ls._dir) / "active_weights" / lease.collection_id
    return {
        "lease_store": ls,
        "lease": lease,
        "weights_dir": weights_dir,
        "weights_path": weights_dir / "weights.pt",
        "onnx_path": weights_dir / "model.onnx",
    }


def _load_lease_classifier_bundle(lease_id: str) -> dict:
    import torch

    artifacts = _lease_model_artifacts(lease_id)
    weights_path = artifacts["weights_path"]
    if not weights_path.exists():
        raise FileNotFoundError("weights.pt not found for this lease collection")
    checkpoint = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    model_class = str(checkpoint.get("model_class", "TinyConvClassifier") or "TinyConvClassifier").strip()
    if model_class != "TinyConvClassifier":
        raise RuntimeError(f"lease model is {model_class}, not TinyConvClassifier")
    lease = artifacts["lease"]
    signature = _checkpoint_signature([weights_path])
    return _build_classifier_bundle(
        cache_key=f"lease:{lease_id}",
        signature=signature,
        classifier_blob=checkpoint,
        pipeline_blob={},
        checkpoint_dir=str(artifacts["weights_dir"]),
        classifier_path=str(weights_path),
        source="lease",
        lease_id=str(lease_id),
        collection_id=str(lease.collection_id),
        slot_name=str(getattr(lease, "slot_name", "") or ""),
        generation=int(getattr(lease, "generation", 0) or 0),
    )


def _decode_image_b64_to_rgb(image_b64: str):
    if not image_b64:
        raise ValueError("image_b64 required")
    payload = str(image_b64).strip()
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    raw_bytes = base64.b64decode(payload)
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow not installed on server; cannot decode image") from exc
    img = Image.open(io.BytesIO(raw_bytes))
    return img.convert("RGB")


def _prepare_classifier_input_tensor(image_b64: str, input_shape: list[int]):
    import numpy as np
    import torch
    from PIL import Image

    img = _decode_image_b64_to_rgb(image_b64)
    width = max(1, int(input_shape[3]))
    height = max(1, int(input_shape[2]))
    resized = img.resize((width, height), resample=Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32)
    if int(arr.ndim) != 3 or int(arr.shape[2]) < 3:
        raise ValueError("decoded image is not RGB")
    chw = np.transpose(arr[:, :, :3] / 255.0, (2, 0, 1))[None, ...]
    tensor = torch.from_numpy(chw)
    meta = {
        "original_width": int(img.size[0]),
        "original_height": int(img.size[1]),
        "resized_width": int(width),
        "resized_height": int(height),
    }
    return tensor, meta


def _classifier_bundle_meta_payload(bundle: dict) -> dict:
    payload = {
        "source": str(bundle.get("source", "latest") or "latest"),
        "model_name": str(bundle.get("model_name", "classifier") or "classifier"),
        "model_class": str(bundle.get("model_class", "TinyConvClassifier") or "TinyConvClassifier"),
        "checkpoint_dir": str(bundle.get("checkpoint_dir", "") or ""),
        "classifier_path": str(bundle.get("classifier_path", "") or ""),
        "pipeline_checkpoint_path": str(bundle.get("pipeline_checkpoint_path", "") or ""),
        "modified": float(bundle.get("mtime", 0.0) or 0.0),
        "input_shape": list(bundle.get("input_shape") or [1, 3, 64, 64]),
        "num_classes": int(bundle.get("num_classes", 0) or 0),
        "class_names": list(bundle.get("class_names") or []),
        "label_texts": list(bundle.get("label_texts") or []),
        "ctor_kwargs": dict(bundle.get("ctor_kwargs") or {}),
        "server_infer_supported": True,
    }
    lease_id = str(bundle.get("lease_id", "") or "")
    if lease_id:
        payload["lease_id"] = lease_id
    collection_id = str(bundle.get("collection_id", "") or "")
    if collection_id:
        payload["collection_id"] = collection_id
    slot_name = str(bundle.get("slot_name", "") or "")
    if slot_name:
        payload["slot_name"] = slot_name
    generation = int(bundle.get("generation", 0) or 0)
    if generation:
        payload["generation"] = generation
    return payload


def _run_classifier_inference(bundle: dict, image_b64: str, topk: int) -> dict:
    import numpy as np
    import torch

    input_tensor, image_meta = _prepare_classifier_input_tensor(
        image_b64=image_b64,
        input_shape=list(bundle["input_shape"]),
    )
    model = bundle["model"]
    with torch.inference_mode():
        logits = model(input_tensor.to(dtype=torch.float32))
        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        logits_np = logits.detach().cpu().numpy().reshape(-1)
    class_names = list(bundle["class_names"])
    label_texts = list(bundle["label_texts"])
    scores = []
    for idx, score in enumerate(probs.tolist()):
        scores.append({
            "index": int(idx),
            "label": str(class_names[idx]) if idx < len(class_names) else f"class_{idx}",
            "label_text": str(label_texts[idx]) if idx < len(label_texts) else (
                str(class_names[idx]) if idx < len(class_names) else f"class_{idx}"
            ),
            "score": float(score),
            "logit": float(logits_np[idx]) if idx < int(len(logits_np)) else 0.0,
        })
    order = np.argsort(-probs)[: max(1, min(int(topk), int(probs.size)))]
    result = _classifier_bundle_meta_payload(bundle)
    result.update({
        "image": image_meta,
        "top_scores": [scores[int(i)] for i in order.tolist()],
        "scores": scores,
    })
    return result


def _export_model_to_onnx_bytes(model, input_shape: list[int], opset: int, service_source: str) -> bytes:
    import torch

    dummy_input = torch.randn(*_normalise_input_shape(input_shape))
    buf = io.BytesIO()
    with _service_priority(service_source):
        torch.onnx.export(
            model,
            dummy_input,
            buf,
            opset_version=int(opset),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
    return buf.getvalue()


def _export_collection_onnx(collection_id: str, model_class_name: str, input_shape: object, opset_version: int, *, lease_store=None) -> dict:
    import torch

    collection_id = str(collection_id or "").strip()
    model_class_name = str(model_class_name or "").strip()
    if not collection_id or not model_class_name:
        raise ValueError("collection_id and model_class are required")
    if model_class_name not in _ALLOWED_MODEL_CLASSES:
        raise ValueError(f"model_class must be one of {sorted(_ALLOWED_MODEL_CLASSES)}")

    ls = lease_store if lease_store is not None else _lease_store()
    if ls is None:
        raise RuntimeError("lease store not configured")
    weights_dir = __import__("pathlib").Path(ls._dir) / "active_weights" / collection_id
    pt_path = weights_dir / "weights.pt"
    if not pt_path.exists():
        raise FileNotFoundError("weights.pt not found for this collection")

    checkpoint = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    model, _state_dict, _ctor_kwargs = _load_model_from_checkpoint_blob(checkpoint, model_class_name)
    resolved_shape = _normalise_input_shape(
        input_shape,
        checkpoint.get("input_shape", [1, 3, 64, 64]) if isinstance(checkpoint, dict) else [1, 3, 64, 64],
    )
    onnx_bytes = _export_model_to_onnx_bytes(
        model,
        resolved_shape,
        int(opset_version or 17),
        f"lease_onnx_export:{collection_id}",
    )
    onnx_path = weights_dir / "model.onnx"
    onnx_path.write_bytes(onnx_bytes)
    return {
        "ok": True,
        "collection_id": collection_id,
        "model_class": model_class_name,
        "onnx_path": str(onnx_path),
        "onnx_size_bytes": len(onnx_bytes),
        "opset_version": int(opset_version or 17),
        "input_shape": resolved_shape,
    }


def _build_model_interface_descriptor(
    *,
    source: str,
    lease_id: str = "",
    model_name: str = "",
    model_class: str = "",
    input_shape: object = None,
    ensure_onnx: bool = False,
    opset_version: int = 17,
) -> dict:
    resolved_source = str(source or "latest").strip().lower()
    if resolved_source not in {"latest", "lease"}:
        raise ValueError("source must be 'latest' or 'lease'")

    if resolved_source == "latest":
        resolved_name, resolved_class = _normalise_model_selection(model_name, model_class)
        if not _CHECKPOINT_DIR:
            raise RuntimeError("checkpoint dir not configured")
        descriptor = {
            "source": "latest",
            "model_name": resolved_name,
            "model_class": resolved_class,
            "input_shape": _normalise_input_shape(input_shape),
            "pt_url": f"/api/model/latest/{resolved_name}.pt",
            "onnx_url": (
                f"/api/model/latest/{resolved_name}.onnx"
                f"?model_class={resolved_class}&input_shape={','.join(str(x) for x in _normalise_input_shape(input_shape))}"
            ),
            "onnx_ready": True,
            "server_infer_supported": False,
            "train_with_lease": False,
        }
        if resolved_name == "classifier" and resolved_class == "TinyConvClassifier":
            bundle = _load_latest_classifier_bundle()
            descriptor.update(_classifier_bundle_meta_payload(bundle))
            descriptor["meta_url"] = "/api/model/latest/classifier/meta"
            descriptor["infer_url"] = "/api/model/interface/infer"
            descriptor["onnx_url"] = (
                "/api/model/latest/classifier.onnx"
                f"?model_class=TinyConvClassifier&input_shape={','.join(str(x) for x in descriptor['input_shape'])}"
            )
        return descriptor

    artifacts = _lease_model_artifacts(lease_id)
    weights_path = artifacts["weights_path"]
    if not weights_path.exists():
        raise FileNotFoundError("weights.pt not found for this lease collection")

    import torch

    checkpoint = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    checkpoint_model_class = str(checkpoint.get("model_class", "") or "").strip() if isinstance(checkpoint, dict) else ""
    resolved_name, resolved_class = _normalise_model_selection(
        model_name,
        model_class or checkpoint_model_class,
    )
    if ensure_onnx:
        _export_collection_onnx(
            str(artifacts["lease"].collection_id),
            resolved_class,
            input_shape,
            int(opset_version or 17),
            lease_store=artifacts["lease_store"],
        )

    resolved_shape = _normalise_input_shape(
        input_shape,
        checkpoint.get("input_shape", [1, 3, 64, 64]) if isinstance(checkpoint, dict) else [1, 3, 64, 64],
    )
    lease = artifacts["lease"]
    descriptor = {
        "source": "lease",
        "lease_id": str(lease_id),
        "collection_id": str(lease.collection_id),
        "slot_name": str(getattr(lease, "slot_name", "") or ""),
        "generation": int(getattr(lease, "generation", 0) or 0),
        "model_name": resolved_name,
        "model_class": resolved_class,
        "input_shape": resolved_shape,
        "weights_pt_url": f"/api/web/lease/{lease_id}/weights",
        "onnx_url": f"/api/web/lease/{lease_id}/weights.onnx",
        "onnx_ready": bool(artifacts["onnx_path"].exists()),
        "dataset_url": f"/api/web/lease/{lease_id}/dataset",
        "server_infer_supported": False,
        "train_with_lease": True,
    }
    if resolved_name == "classifier" and resolved_class == "TinyConvClassifier":
        bundle = _load_lease_classifier_bundle(lease_id)
        descriptor.update(_classifier_bundle_meta_payload(bundle))
        descriptor["infer_url"] = "/api/model/interface/infer"
    return descriptor


# ---------------------------------------------------------------------------
# Route handler
# ---------------------------------------------------------------------------

# Colour palette for graph rendering (RGBA).
_GRAPH_COLORS = [
    (255, 128,  64, 255),
    ( 64, 200, 255, 255),
    (128, 255, 128, 255),
    (255, 220,  60, 255),
    (200,  80, 255, 255),
    (255,  80, 120, 255),
    ( 80, 255, 200, 255),
    (255, 180,  80, 255),
]


class NodusHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the nodus data server."""

    server_version = "NodusServer/1.0"
    protocol_version = "HTTP/1.1"

    # ------------------------------------------------------------------ #
    #  Entry point                                                         #
    # ------------------------------------------------------------------ #

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path.rstrip("/") or "/"
        qs     = parse_qs(parsed.query, keep_blank_values=False)
        try:
            self._dispatch(path, qs)
        except Exception as exc:
            self._send(500, "application/json", _json({"error": str(exc)}))

    def do_POST(self):
        parsed = urlparse(self.path)
        path   = parsed.path.rstrip("/") or "/"
        length = int(self.headers.get("Content-Length", 0) or 0)
        body   = self.rfile.read(length) if length > 0 else b""
        try:
            self._dispatch_post(path, body)
        except Exception as exc:
            self._send(500, "application/json", _json({"error": str(exc)}))

    def do_DELETE(self):
        parsed = urlparse(self.path)
        path   = parsed.path.rstrip("/") or "/"
        try:
            self._dispatch_delete(path)
        except Exception as exc:
            self._send(500, "application/json", _json({"error": str(exc)}))

    def do_OPTIONS(self):
        # CORS preflight — browsers send this before cross-origin POST/DELETE
        self._send(204, "text/plain", b"")

    def log_message(self, fmt, *args):
        # Keep output minimal.
        sys.stderr.write(f"[nodus] {self.address_string()} {fmt % args}\n")

    # ------------------------------------------------------------------ #
    #  Routing                                                             #
    # ------------------------------------------------------------------ #

    def _dispatch(self, path: str, qs: dict):
        # Root index
        if path == "/":
            return self._handle_index()

        if path == "/train":
            return self._handle_train_ui()

        # -- Latest model (from checkpoint dir) --
        if path == "/api/model/latest":
            return self._handle_model_latest_list()

        if path == "/api/model/latest/classifier/meta":
            return self._handle_model_latest_classifier_meta()

        m = re.fullmatch(r"/api/model/latest/([^/]+)\.onnx", path)
        if m:
            return self._handle_model_latest_onnx(m.group(1), qs)

        m = re.fullmatch(r"/api/model/latest/([^/]+)\.pt", path)
        if m:
            return self._handle_model_latest_pt(m.group(1))

        if path == "/api/status":
            return self._handle_status()

        if path == "/api/runtime/control":
            return self._handle_runtime_control_status()

        # -- Loss store --
        if path == "/api/loss/channels":
            return self._handle_loss_channels()

        m = re.fullmatch(r"/api/loss/channel/(.+)/records", path)
        if m:
            return self._handle_loss_records(m.group(1), qs)

        m = re.fullmatch(r"/api/loss/channel/(.+)/latest", path)
        if m:
            return self._handle_loss_latest(m.group(1))

        if path == "/api/loss/graph.png":
            return self._handle_loss_graph(qs)

        # -- Scrub ring --
        if path == "/api/scrub/info":
            return self._handle_scrub_info()

        if path == "/api/scrub/latest/meta":
            return self._handle_scrub_meta("latest")
        if path == "/api/scrub/latest/training.png":
            return self._handle_scrub_image("latest", "training")
        if path == "/api/scrub/latest/output.png":
            return self._handle_scrub_image("latest", "output")
        if path == "/api/scrub/latest/target.png":
            return self._handle_scrub_image("latest", "target")

        m = re.fullmatch(r"/api/scrub/(\d+)/meta", path)
        if m:
            return self._handle_scrub_meta(int(m.group(1)))

        m = re.fullmatch(r"/api/scrub/(\d+)/training\.png", path)
        if m:
            return self._handle_scrub_image(int(m.group(1)), "training")

        m = re.fullmatch(r"/api/scrub/(\d+)/output\.png", path)
        if m:
            return self._handle_scrub_image(int(m.group(1)), "output")

        m = re.fullmatch(r"/api/scrub/(\d+)/target\.png", path)
        if m:
            return self._handle_scrub_image(int(m.group(1)), "target")

        # -- Weight stores --
        if path == "/api/weight/meta":
            return self._handle_weight_meta_latest()

        if path == "/api/weight/meta/all":
            return self._handle_weight_meta_all()

        if path == "/api/weight/image/info":
            return self._handle_weight_image_info()

        if path == "/api/weight/image/latest.png":
            return self._handle_weight_image("latest")

        m = re.fullmatch(r"/api/weight/image/(\d+)\.png", path)
        if m:
            return self._handle_weight_image(int(m.group(1)))

        # -- Web dataset / lease weights --
        if path == "/api/web/dataset/status":
            return self._handle_web_dataset_status()

        m = re.fullmatch(r"/api/web/dataset/([^/]+)/([^/]+)/image\.png", path)
        if m:
            return self._handle_web_sample_image(m.group(1), m.group(2))

        m = re.fullmatch(r"/api/web/lease/([^/]+)/weights", path)
        if m:
            return self._handle_web_lease_weights(m.group(1))

        m = re.fullmatch(r"/api/web/lease/([^/]+)/weights\.onnx", path)
        if m:
            return self._handle_web_lease_weights_onnx(m.group(1))

        m = re.fullmatch(r"/api/web/lease/([^/]+)/dataset", path)
        if m:
            return self._handle_web_lease_dataset(m.group(1))

        # -- Lease system --
        if path == "/api/lease/status":
            return self._handle_lease_status()

        m = re.fullmatch(r"/api/lease/collection/([^/]+)", path)
        if m:
            return self._handle_lease_collection_detail(m.group(1))

        m = re.fullmatch(r"/api/lease/([^/]+)", path)
        if m:
            return self._handle_lease_detail(m.group(1))

        # -- Unified model interface (lease or latest) --
        if path == "/api/model/interface/meta":
            return self._handle_model_interface_meta(qs)

        self._send(404, "application/json", _json({"error": "not found", "path": path}))

    # ------------------------------------------------------------------ #
    #  Handlers                                                            #
    # ------------------------------------------------------------------ #

    def _handle_index(self):
        html = _INDEX_HTML
        self._send(200, "text/html; charset=utf-8", html.encode())

    def _handle_train_ui(self):
        import pathlib
        html_path = pathlib.Path(__file__).parent / "native" / "nodus_train.html"
        if not html_path.exists():
            self._send(404, "text/plain", b"nodus_train.html not found")
            return
        self._send(200, "text/html; charset=utf-8", html_path.read_bytes())

    # ------------------------------------------------------------------ #
    #  Latest model handlers (served from --checkpoint-dir)                #
    # ------------------------------------------------------------------ #

    def _handle_model_latest_list(self):
        if not _CHECKPOINT_DIR:
            self._send(503, "application/json",
                       _json({"error": "checkpoint dir not configured (start with --checkpoint-dir)"}))
            return
        import pathlib
        ckpt_dir = pathlib.Path(_CHECKPOINT_DIR)
        available = {}
        for name in _LATEST_MODEL_ALLOWED:
            pt_path = ckpt_dir / f"{name}.pt"
            if pt_path.exists():
                st = pt_path.stat()
                available[name] = {
                    "pt_size_bytes": st.st_size,
                    "modified": st.st_mtime,
                    "onnx_url": f"/api/model/latest/{name}.onnx",
                    "pt_url": f"/api/model/latest/{name}.pt",
                }
        self._send(200, "application/json", _json({"checkpoint_dir": _CHECKPOINT_DIR, "models": available}))

    def _handle_model_latest_pt(self, name: str):
        if not _CHECKPOINT_DIR:
            self._send(503, "application/json",
                       _json({"error": "checkpoint dir not configured"}))
            return
        if name not in _LATEST_MODEL_ALLOWED:
            self._send(400, "application/json",
                       _json({"error": f"model must be one of {sorted(_LATEST_MODEL_ALLOWED)}",
                               "given": name}))
            return
        import pathlib
        pt_path = pathlib.Path(_CHECKPOINT_DIR) / f"{name}.pt"
        if not pt_path.exists():
            self._send(404, "application/json", _json({"error": f"{name}.pt not found"}))
            return
        self._send(200, "application/octet-stream", pt_path.read_bytes())

    def _handle_model_latest_classifier_meta(self):
        try:
            bundle = _load_latest_classifier_bundle()
        except Exception as exc:
            self._send(503, "application/json", _json({"error": str(exc)}))
            return
        self._send(200, "application/json", _json({
            "checkpoint_dir": bundle["checkpoint_dir"],
            "classifier_path": bundle["classifier_path"],
            "pipeline_checkpoint_path": bundle["pipeline_checkpoint_path"],
            "modified": bundle["mtime"],
            "input_shape": list(bundle["input_shape"]),
            "num_classes": int(bundle["num_classes"]),
            "class_names": list(bundle["class_names"]),
            "label_texts": list(bundle["label_texts"]),
            "ctor_kwargs": dict(bundle["ctor_kwargs"]),
        }))

    def _handle_model_latest_classifier_infer(self, body: bytes):
        try:
            req = json.loads(body or b"{}")
        except Exception:
            self._send(400, "application/json", _json({"error": "invalid JSON body"}))
            return
        image_b64 = str(req.get("image_b64") or "").strip()
        topk = max(1, int(req.get("topk", 12) or 12))
        if not image_b64:
            self._send(400, "application/json", _json({"error": "image_b64 required"}))
            return
        with _service_priority("latest_classifier_infer"):
            try:
                bundle = _load_latest_classifier_bundle()
                result = _run_classifier_inference(bundle, image_b64, topk)
            except Exception as exc:
                self._send(500, "application/json", _json({"error": f"classifier inference failed: {exc}"}))
                return
        self._send(200, "application/json", _json(result))

    def _handle_model_latest_onnx(self, name: str, qs: dict):
        """On-demand ONNX conversion of the latest checkpoint.

        Caches the .onnx bytes keyed on .pt mtime — only re-exports when the
        checkpoint file changes.
        """
        if not _CHECKPOINT_DIR:
            self._send(503, "application/json",
                       _json({"error": "checkpoint dir not configured (start with --checkpoint-dir)"}))
            return
        if name not in _LATEST_MODEL_ALLOWED:
            self._send(400, "application/json",
                       _json({"error": f"model must be one of {sorted(_LATEST_MODEL_ALLOWED)}",
                               "given": name}))
            return

        import pathlib
        pt_path = pathlib.Path(_CHECKPOINT_DIR) / f"{name}.pt"
        if not pt_path.exists():
            self._send(404, "application/json", _json({"error": f"{name}.pt not found"}))
            return

        current_mtime = pt_path.stat().st_mtime

        # Check cache
        with _ONNX_CACHE_LOCK:
            cached = _ONNX_CACHE.get(name)
            if cached is not None and cached[0] == current_mtime:
                self._send(200, "application/octet-stream", cached[1])
                return

        # Must convert — parse optional query params
        model_class = (qs.get("model_class") or [""])[0].strip()
        input_shape_str = (qs.get("input_shape") or [""])[0].strip()
        opset = int((qs.get("opset") or ["17"])[0])

        try:
            import torch
            import wav_ml_models as _models
            from wav_ml_models import prime_tiny_classifier_label_bank_for_state_dict

            checkpoint = torch.load(str(pt_path), map_location="cpu", weights_only=False)

            # Infer model class from name if not explicitly given
            _CLASS_HINTS = {
                "classifier": "TinyConvClassifier",
                "transformer": "WavePatchTransformer",
                "generator": "ConditionalBitPlaneGenerator",
                "discriminator": "ConditionalBitPlaneDiscriminator",
                "wave_classifier": "DeskewFilterBundle",
            }
            if not model_class:
                model_class = _CLASS_HINTS.get(name, "")
            if not model_class:
                self._send(400, "application/json",
                           _json({"error": "cannot infer model_class — pass ?model_class=ClassName"}))
                return

            _ALLOWED = {
                "TinyConvClassifier", "ConditionalBitPlaneGenerator",
                "ConditionalBitPlaneDiscriminator", "WavePatchTransformer",
                "DeskewFilterBundle",
            }
            if model_class not in _ALLOWED:
                self._send(400, "application/json",
                           _json({"error": f"model_class must be one of {sorted(_ALLOWED)}"}))
                return

            if name == "classifier" or model_class == "TinyConvClassifier":
                with _service_priority("latest_classifier_onnx"):
                    bundle = _load_latest_classifier_bundle()
                    model = bundle["model"]
            else:
                ctor_kwargs = {}
                state_dict = checkpoint
                if isinstance(checkpoint, dict):
                    ctor_kwargs = checkpoint.get("ctor_kwargs", {})
                    state_dict = checkpoint.get("state_dict", checkpoint)
                    if "model_state_dict" in state_dict:
                        state_dict = state_dict["model_state_dict"]
                    if not model_class:
                        model_class = checkpoint.get("model_class", "")
                cls = getattr(_models, model_class)
                model = cls(**ctor_kwargs) if ctor_kwargs else cls()
                if isinstance(state_dict, dict):
                    model.load_state_dict(state_dict, strict=False)
            model.eval()

            if input_shape_str:
                input_shape = [int(d) for d in input_shape_str.split(",")]
            else:
                input_shape = list(bundle["input_shape"]) if name == "classifier" or model_class == "TinyConvClassifier" else [1, 3, 64, 64]
            dummy_input = torch.randn(*input_shape)

            buf = io.BytesIO()
            with _service_priority(f"latest_model_onnx:{name}"):
                torch.onnx.export(
                    model, dummy_input, buf,
                    opset_version=opset,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                )
            onnx_bytes = buf.getvalue()

            with _ONNX_CACHE_LOCK:
                _ONNX_CACHE[name] = (current_mtime, onnx_bytes)

            self._send(200, "application/octet-stream", onnx_bytes)
        except Exception as exc:
            self._send(500, "application/json",
                       _json({"error": f"ONNX conversion failed: {exc}"}))

    def _handle_status(self):
        try:
            ls  = _loss()
            sr  = _scrub()
            wst = _wstate()
            wim = _wimage()
            runtime_control = _runtime_control_state_dict()
            body = _json({
                "ok": True,
                "loss_channels": ls.channel_count(),
                "scrub_length": sr.length(),
                "scrub_capacity": sr.capacity(),
                "weight_image_count": wim.length(),
                "runtime_control": runtime_control,
            })
        except Exception as exc:
            body = _json({"ok": False, "error": str(exc)})
        self._send(200, "application/json", body)

    def _handle_runtime_control_status(self):
        self._send(200, "application/json", _json(_runtime_control_state_dict()))

    # ---- Unified model interface (interrupt-pause-resume) ----

    def _handle_model_interface_meta(self, qs: dict):
        """GET /api/model/interface/meta

        Returns the descriptor for either the latest checkpoint or a lease model.
        The descriptor includes input_shape, class labels, and infer_url so the
        client knows where to send inference requests.

        Query params:
            source       "latest" (default) or "lease"
            lease_id     required when source=lease
            model_name   optional (default "classifier")
            model_class  optional (default inferred from name)
            input_shape  optional comma-separated e.g. "1,3,64,64"
        """
        source = (qs.get("source") or ["latest"])[0].strip().lower() or "latest"
        lease_id = (qs.get("lease_id") or [""])[0].strip()
        model_name = (qs.get("model_name") or [""])[0].strip()
        model_class = (qs.get("model_class") or [""])[0].strip()
        input_shape_str = (qs.get("input_shape") or [""])[0].strip()
        input_shape = [int(x) for x in input_shape_str.split(",") if x.strip()] if input_shape_str else None
        try:
            descriptor = _build_model_interface_descriptor(
                source=source,
                lease_id=lease_id,
                model_name=model_name,
                model_class=model_class,
                input_shape=input_shape,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            self._send(404 if isinstance(exc, FileNotFoundError) else 400,
                       "application/json", _json({"error": str(exc)}))
            return
        except Exception as exc:
            self._send(500, "application/json", _json({"error": str(exc)}))
            return
        self._send(200, "application/json", _json(descriptor))

    def _handle_model_interface_prepare(self, body: bytes):
        """POST /api/model/interface/prepare

        Ensures the model is ready for client use (ONNX export if requested) and
        returns the full descriptor.  Does NOT run inference or pause training.

        Body JSON:
            source        "latest" or "lease"
            lease_id      required when source=lease
            model_name    optional
            model_class   optional
            input_shape   optional list e.g. [1,3,64,64]
            ensure_onnx   bool (default false) — trigger ONNX export now
            opset_version int (default 17)
        """
        try:
            req = json.loads(body or b"{}")
        except Exception:
            self._send(400, "application/json", _json({"error": "invalid JSON body"}))
            return
        source = str(req.get("source") or "latest").strip().lower() or "latest"
        lease_id = str(req.get("lease_id") or "").strip()
        model_name = str(req.get("model_name") or "").strip()
        model_class = str(req.get("model_class") or "").strip()
        input_shape = req.get("input_shape") or None
        ensure_onnx = bool(req.get("ensure_onnx", False))
        opset_version = int(req.get("opset_version") or 17)
        try:
            with _service_priority("model_interface_prepare"):
                descriptor = _build_model_interface_descriptor(
                    source=source,
                    lease_id=lease_id,
                    model_name=model_name,
                    model_class=model_class,
                    input_shape=input_shape,
                    ensure_onnx=ensure_onnx,
                    opset_version=opset_version,
                )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            self._send(404 if isinstance(exc, FileNotFoundError) else 400,
                       "application/json", _json({"error": str(exc)}))
            return
        except Exception as exc:
            self._send(500, "application/json", _json({"error": str(exc)}))
            return
        self._send(200, "application/json", _json(descriptor))

    def _handle_model_interface_infer(self, body: bytes):
        """POST /api/model/interface/infer

        Interrupts training (via service priority), runs classifier inference on
        the supplied image, then allows training to resume.

        Body JSON:
            source       "latest" (default) or "lease"
            lease_id     required when source=lease
            model_name   optional
            model_class  optional
            image_b64    base64 PNG/JPEG (required)
            topk         int (default 12)
        """
        try:
            req = json.loads(body or b"{}")
        except Exception:
            self._send(400, "application/json", _json({"error": "invalid JSON body"}))
            return
        source = str(req.get("source") or "latest").strip().lower() or "latest"
        lease_id = str(req.get("lease_id") or "").strip()
        model_name = str(req.get("model_name") or "").strip()
        model_class = str(req.get("model_class") or "").strip()
        image_b64 = str(req.get("image_b64") or "").strip()
        topk = max(1, int(req.get("topk") or 12))
        if not image_b64:
            self._send(400, "application/json", _json({"error": "image_b64 required"}))
            return
        service_tag = f"model_interface_infer:{source}"
        with _service_priority(service_tag):
            try:
                if source == "lease":
                    bundle = _load_lease_classifier_bundle(lease_id)
                else:
                    bundle = _load_latest_classifier_bundle()
                result = _run_classifier_inference(bundle, image_b64, topk)
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                self._send(404 if isinstance(exc, FileNotFoundError) else 400,
                           "application/json", _json({"error": str(exc)}))
                return
            except Exception as exc:
                self._send(500, "application/json", _json({"error": f"inference failed: {exc}"}))
                return
        self._send(200, "application/json", _json(result))

    # ---- Loss store ----

    def _handle_loss_channels(self):
        ls = _loss()
        channels = []
        with ls.locked():
            for key in ls.channel_keys():
                lat = ls.latest(key)
                channels.append({
                    "key": key,
                    "length": ls.channel_length(key),
                    "cursor": ls.channel_cursor(key),
                    "latest_loss": lat.loss if lat else None,
                    "latest_step": lat.step if lat else None,
                })
        self._send(200, "application/json", _json({"channels": channels}))

    def _handle_loss_records(self, key: str, qs: dict):
        ls = _loss()
        from_step  = int(qs.get("from_step", ["0"])[0])
        max_rec    = int(qs.get("max", ["100000"])[0])
        records = ls.query_since(key, from_step=from_step, max_records=max_rec)
        body = _json({
            "key": key,
            "count": len(records),
            "records": [_dataclass_dict(r) for r in records],
        })
        self._send(200, "application/json", body)

    def _handle_loss_latest(self, key: str):
        ls = _loss()
        rec = ls.latest(key)
        if rec is None:
            self._send(404, "application/json", _json({"error": "channel not found or empty"}))
            return
        self._send(200, "application/json", _json(_dataclass_dict(rec)))

    def _handle_loss_graph(self, qs: dict):
        ls = _loss()
        w   = max(1, int(qs.get("w", ["800"])[0]))
        h   = max(1, int(qs.get("h", ["200"])[0]))
        start = float(qs.get("start", ["0.0"])[0])
        bg_str = qs.get("bg", ["0,0,0"])[0]

        with ls.locked():
            keys = ls.channel_keys()
            if not keys:
                blank = b"\x00" * (w * h * 3)
                self._send(200, "image/png", _encode_png_rgb(w, h, blank))
                return

            colors = [_GRAPH_COLORS[i % len(_GRAPH_COLORS)] for i in range(len(keys))]
            result = ls.render_all_lines(keys, colors, w, h,
                                         display_start_frac=max(0.0, min(1.0, start)))
        if result is None:
            blank = b"\x00" * (w * h * 3)
            self._send(200, "image/png", _encode_png_rgb(w, h, blank))
            return

        overlay = result["overlay"]  # numpy (h, w, 4) uint8
        rgba_bytes = overlay.tobytes()

        try:
            bg_parts = [max(0, min(255, int(v))) for v in bg_str.split(",")]
            br, bg, bb = (bg_parts + [0, 0, 0])[:3]
        except Exception:
            br, bg, bb = 0, 0, 0

        import array
        rgb = array.array("B", b"\x00" * (w * h * 3))
        for i in range(w * h):
            r, g, b_, a = (rgba_bytes[i * 4], rgba_bytes[i * 4 + 1],
                           rgba_bytes[i * 4 + 2], rgba_bytes[i * 4 + 3])
            if a:
                alpha = a / 255.0
                inv = 1.0 - alpha
                rgb[i * 3]     = int(r * alpha + br * inv)
                rgb[i * 3 + 1] = int(g * alpha + bg * inv)
                rgb[i * 3 + 2] = int(b_ * alpha + bb * inv)
            else:
                rgb[i * 3]     = br
                rgb[i * 3 + 1] = bg
                rgb[i * 3 + 2] = bb

        self._send(200, "image/png", _encode_png_rgb(w, h, bytes(rgb)))

    # ---- Scrub ring ----

    def _handle_scrub_info(self):
        sr = _scrub()
        with sr.locked():
            body = _json({
                "length": sr.length(),
                "capacity": sr.capacity(),
                "write_cursor": sr.write_cursor(),
            })
        self._send(200, "application/json", body)

    def _handle_scrub_meta(self, index):
        sr = _scrub()
        with sr.locked():
            idx = self._resolve_scrub_index(sr, index)
            if idx is None:
                self._send(404, "application/json", _json({"error": "empty ring"}))
                return
            meta = sr.get_meta(idx)
        if meta is None:
            self._send(404, "application/json", _json({"error": "index out of range"}))
            return
        d = _dataclass_dict(meta)
        d["index"] = idx
        self._send(200, "application/json", _json(d))

    def _handle_scrub_image(self, index, kind: str):
        sr = _scrub()
        with sr.locked():
            idx = self._resolve_scrub_index(sr, index)
            if idx is None:
                self._send(404, "application/json", _json({"error": "empty ring"}))
                return
            meta = sr.get_meta(idx)
            if meta is None:
                self._send(404, "application/json", _json({"error": "index out of range"}))
                return

            if kind == "training":
                raw = sr.copy_training_image(idx)
                w, h = meta.image_w, meta.image_h
                channels = 4
            elif kind == "output":
                raw = sr.copy_output_image(idx)
                w, h = meta.output_w, meta.output_h
                channels = 4
            elif kind == "target":
                raw = sr.copy_target(idx)
                w, h = meta.image_w, meta.image_h
                channels = 4
            else:
                self._send(400, "application/json", _json({"error": "unknown image kind"}))
                return

        if raw is None or len(raw) == 0:
            self._send(404, "application/json", _json({"error": "image not present"}))
            return

        if w == 0 or h == 0:
            from pipeline.nodus_loss_store import SCRUB_IMAGE_W, SCRUB_IMAGE_H
            w, h = SCRUB_IMAGE_W, SCRUB_IMAGE_H

        rgba = bytes(raw[:w * h * channels])
        self._send(200, "image/png", _encode_png_rgba(w, h, rgba))

    @staticmethod
    def _resolve_scrub_index(sr, index) -> Optional[int]:
        """Resolve 'latest' or integer index to a logical ring index."""
        length = sr.length()
        if length == 0:
            return None
        if index == "latest":
            return length - 1
        idx = int(index)
        if idx < 0 or idx >= length:
            return None
        return idx

    # ---- Weight stores ----

    def _handle_weight_meta_latest(self):
        wst = _wstate()
        meta = wst.get_meta()
        if meta is None:
            self._send(404, "application/json", _json({"error": "no weight state published"}))
            return
        self._send(200, "application/json", _json(_dataclass_dict(meta)))

    def _handle_weight_meta_all(self):
        wst = _wstate()
        with wst.locked():
            count = wst.count()
            entries = []
            for i in range(count):
                meta = wst.get_meta_at(i)
                if meta is not None:
                    entries.append(_dataclass_dict(meta))
        self._send(200, "application/json", _json({"count": count, "entries": entries}))

    def _handle_weight_image_info(self):
        wim = _wimage()
        stats = wim.stats()
        if stats is None:
            self._send(500, "application/json", _json({"error": "could not read stats"}))
            return
        self._send(200, "application/json", _json(_dataclass_dict(stats)))

    def _handle_weight_image(self, index):
        wim = _wimage()
        with wim.locked():
            count = wim.length()
            if count == 0:
                self._send(404, "application/json", _json({"error": "no weight images cached"}))
                return

            if index == "latest":
                idx = count - 1
            else:
                idx = int(index)
                if idx < 0 or idx >= count:
                    self._send(404, "application/json", _json({"error": "index out of range"}))
                    return

            meta = wim.get_meta(idx)
            if meta is None:
                self._send(404, "application/json", _json({"error": "metadata unavailable"}))
                return

            raw = wim.copy_image(idx)

        if raw is None or len(raw) == 0:
            self._send(404, "application/json", _json({"error": "image data unavailable"}))
            return

        w, h, c = meta.width, meta.height, meta.channels
        flat = bytes(raw.reshape(-1))
        if c == 3:
            self._send(200, "image/png", _encode_png_rgb(w, h, flat))
        elif c == 4:
            self._send(200, "image/png", _encode_png_rgba(w, h, flat))
        else:
            self._send(500, "application/json",
                       _json({"error": f"unexpected channel count {c}"}))

    # ------------------------------------------------------------------ #
    #  Lease routes (GET)                                                  #
    # ------------------------------------------------------------------ #

    def _handle_lease_status(self):
        ls = _lease_store()
        if ls is None:
            self._send(503, "application/json",
                       _json({"error": "lease store not configured (start with --lease-store-dir)"}))
            return
        ls.expire_stale_leases()
        self._send(200, "application/json", _json({
            "slots": ls.slot_status(),
            "locked": ls.locked_slots(),
        }))

    def _handle_lease_collection_detail(self, collection_id: str):
        ls = _lease_store()
        if ls is None:
            self._send(503, "application/json", _json({"error": "lease store not configured"}))
            return
        detail = ls.collection_detail(collection_id)
        if detail is None:
            self._send(404, "application/json", _json({"error": "collection not found"}))
            return
        self._send(200, "application/json", _json(detail))

    def _handle_lease_detail(self, lease_id: str):
        ls = _lease_store()
        if ls is None:
            self._send(503, "application/json", _json({"error": "lease store not configured"}))
            return
        lease = ls.get_lease(lease_id)
        if lease is None:
            self._send(404, "application/json", _json({"error": "lease not found"}))
            return
        import dataclasses as _dc
        self._send(200, "application/json", _json(_dc.asdict(lease)))

    # ------------------------------------------------------------------ #
    #  Web dataset + lease weight handlers (GET)                          #
    # ------------------------------------------------------------------ #

    def _handle_web_dataset_status(self):
        wd = _web_dataset()
        if wd is None:
            self._send(503, "application/json", _json({"error": "web dataset not configured"}))
            return
        self._send(200, "application/json", _json(wd.status()))

    def _handle_web_sample_image(self, slot_name: str, sample_id: str):
        wd = _web_dataset()
        if wd is None:
            self._send(503, "application/json", _json({"error": "web dataset not configured"}))
            return
        sample = wd.get_sample(slot_name, sample_id)
        if sample is None:
            self._send(404, "application/json", _json({"error": "sample not found"}))
            return
        rgba = wd.load_rgba(slot_name, sample_id)
        if rgba is None:
            self._send(404, "application/json", _json({"error": "image data not found"}))
            return
        import struct as _struct, zlib as _zlib
        # Encode float32 [H,W,4] → uint8 RGBA PNG
        h, w = int(rgba.shape[0]), int(rgba.shape[1])
        rgba_u8 = (rgba.clip(0, 1) * 255).astype("uint8").tobytes()
        self._send(200, "image/png", _encode_png_rgba(w, h, rgba_u8))

    def _handle_web_lease_weights(self, lease_id: str):
        ls = _lease_store()
        if ls is None:
            self._send(503, "application/json", _json({"error": "lease store not configured"}))
            return
        lease = ls.get_lease(lease_id)
        if lease is None:
            self._send(404, "application/json", _json({"error": "lease not found"}))
            return
        import dataclasses as _dc
        weights_path = (
            __import__("pathlib").Path(ls._dir)
            / "active_weights" / lease.collection_id / "weights.pt"
        )
        if not weights_path.exists():
            self._send(503, "application/json",
                       _json({"error": "weights not yet exported for this collection",
                               "collection_id": lease.collection_id}))
            return
        body = weights_path.read_bytes()
        self._send(200, "application/octet-stream", body)

    def _handle_web_lease_weights_onnx(self, lease_id: str):
        """Serve a previously-exported ONNX file for a lease."""
        ls = _lease_store()
        if ls is None:
            self._send(503, "application/json", _json({"error": "lease store not configured"}))
            return
        lease = ls.get_lease(lease_id)
        if lease is None:
            self._send(404, "application/json", _json({"error": "lease not found"}))
            return
        onnx_path = (
            __import__("pathlib").Path(ls._dir)
            / "active_weights" / lease.collection_id / "model.onnx"
        )
        if not onnx_path.exists():
            self._send(404, "application/json",
                       _json({"error": "ONNX model not found — POST /api/onnx/export first",
                               "collection_id": lease.collection_id}))
            return
        body = onnx_path.read_bytes()
        self._send(200, "application/octet-stream", body)

    def _handle_onnx_export(self, ls, body: bytes):
        """Export a .pt checkpoint to ONNX via torch.onnx.export.

        Body JSON::

            {
              "collection_id": "<id>",
              "model_class": "TinyConvClassifier" | "ConditionalBitPlaneGenerator" | ...,
              "input_shape": [1, 3, 64, 64],     // optional, default varies by class
              "opset_version": 17                   // optional, default 17
            }
        """
        if ls is None:
            self._send(503, "application/json", _json({"error": "lease store not configured"}))
            return
        try:
            req = json.loads(body or b"{}")
        except Exception:
            self._send(400, "application/json", _json({"error": "invalid JSON body"}))
            return
        collection_id = str(req.get("collection_id") or "").strip()
        model_class_name = str(req.get("model_class") or "").strip()
        if not collection_id or not model_class_name:
            self._send(400, "application/json",
                       _json({"error": "collection_id and model_class are required"}))
            return

        _ALLOWED_CLASSES = {
            "TinyConvClassifier", "ConditionalBitPlaneGenerator",
            "ConditionalBitPlaneDiscriminator", "WavePatchTransformer",
            "DeskewFilterBundle",
        }
        if model_class_name not in _ALLOWED_CLASSES:
            self._send(400, "application/json",
                       _json({"error": f"model_class must be one of {sorted(_ALLOWED_CLASSES)}",
                               "given": model_class_name}))
            return

        weights_dir = __import__("pathlib").Path(ls._dir) / "active_weights" / collection_id
        pt_path = weights_dir / "weights.pt"
        if not pt_path.exists():
            self._send(404, "application/json",
                       _json({"error": "weights.pt not found for this collection"}))
            return

        try:
            import torch
            import wav_ml_models as _models

            checkpoint = torch.load(str(pt_path), map_location="cpu", weights_only=False)
            cls = getattr(_models, model_class_name)

            # Try to infer constructor args from checkpoint metadata
            ctor_kwargs = {}
            if isinstance(checkpoint, dict) and "ctor_kwargs" in checkpoint:
                ctor_kwargs = checkpoint["ctor_kwargs"]
            state_dict = _extract_checkpoint_state_dict(checkpoint, preferred_keys=("state_dict",))

            if model_class_name == "TinyConvClassifier":
                if not isinstance(ctor_kwargs, dict) or not ctor_kwargs:
                    ctor_kwargs = _infer_tiny_classifier_ctor_kwargs(
                        state_dict,
                        int(getattr(checkpoint, "get", lambda *_args, **_kwargs: 0)("num_classes", 0) if hasattr(checkpoint, "get") else 0),
                    )
            model = cls(**ctor_kwargs) if ctor_kwargs else cls.__new__(cls)
            if not ctor_kwargs:
                try:
                    model = cls()
                except TypeError:
                    self._send(400, "application/json",
                               _json({"error": f"Cannot auto-instantiate {model_class_name} — "
                                                f"include ctor_kwargs in checkpoint or request"}))
                    return
            if isinstance(state_dict, dict):
                if model_class_name == "TinyConvClassifier":
                    prime_tiny_classifier_label_bank_for_state_dict(
                        model,
                        state_dict,
                        temperature=float(getattr(checkpoint, "get", lambda *_args, **_kwargs: 10.0)("label_embedding_temperature", 10.0) if hasattr(checkpoint, "get") else 10.0),
                    )
                model.load_state_dict(state_dict, strict=False)
            model.eval()

            input_shape = req.get("input_shape") or [1, 3, 64, 64]
            input_shape = [int(d) for d in input_shape]
            dummy_input = torch.randn(*input_shape)
            opset = int(req.get("opset_version") or 17)

            onnx_path = weights_dir / "model.onnx"
            with _service_priority(f"lease_onnx_export:{collection_id}"):
                torch.onnx.export(
                    model, dummy_input, str(onnx_path),
                    opset_version=opset,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                )
            onnx_size = onnx_path.stat().st_size
            self._send(200, "application/json", _json({
                "ok": True,
                "collection_id": collection_id,
                "model_class": model_class_name,
                "onnx_path": str(onnx_path),
                "onnx_size_bytes": onnx_size,
                "opset_version": opset,
                "input_shape": input_shape,
            }))
        except Exception as exc:
            self._send(500, "application/json",
                       _json({"error": f"ONNX export failed: {exc}"}))

    def _handle_web_lease_dataset(self, lease_id: str):
        ls = _lease_store()
        wd = _web_dataset()
        if ls is None or wd is None:
            self._send(503, "application/json", _json({"error": "lease store or web dataset not configured"}))
            return
        lease = ls.get_lease(lease_id)
        if lease is None:
            self._send(404, "application/json", _json({"error": "lease not found"}))
            return
        import dataclasses as _dc
        samples = wd.samples_for_collection(lease.slot_name, lease.collection_id)
        self._send(200, "application/json", _json({
            "collection_id": lease.collection_id,
            "slot_name": lease.slot_name,
            "generation": lease.generation,
            "samples": [_dc.asdict(s) for s in samples],
        }))

    # ------------------------------------------------------------------ #
    #  POST routing + handlers                                             #
    # ------------------------------------------------------------------ #

    def _dispatch_post(self, path: str, body: bytes):
        ls = _lease_store()

        if path == "/api/model/latest/classifier/infer":
            return self._handle_model_latest_classifier_infer(body)

        # ONNX export from a .pt checkpoint
        if path == "/api/onnx/export":
            return self._handle_onnx_export(ls, body)

        # Browser-submitted training sample
        m = re.fullmatch(r"/api/web/dataset/([^/]+)", path)
        if m:
            return self._handle_web_dataset_submit(m.group(1), body)

        # Open a new collection (lock a slot, issue leases)
        if path == "/api/lease/collection":
            return self._handle_lease_open_collection(ls, body)

        # Return a lease with optional gradient payload
        m = re.fullmatch(r"/api/lease/([^/]+)/gradients", path)
        if m:
            return self._handle_lease_return(ls, m.group(1), body)

        # Force-expire a collection (admin unlock)
        m = re.fullmatch(r"/api/lease/collection/([^/]+)/expire", path)
        if m:
            return self._handle_lease_expire_collection(ls, m.group(1))

        # Mark a collection as applied (pipeline consumed the gradients)
        m = re.fullmatch(r"/api/lease/collection/([^/]+)/apply", path)
        if m:
            return self._handle_lease_mark_applied(ls, m.group(1))

        # -- Unified model interface (lease or latest) --
        if path == "/api/model/interface/prepare":
            return self._handle_model_interface_prepare(body)

        if path == "/api/model/interface/infer":
            return self._handle_model_interface_infer(body)

        self._send(404, "application/json", _json({"error": "not found", "path": path}))

    def _handle_web_dataset_submit(self, slot_name: str, body: bytes):
        """Accept a browser-submitted labeled image.

        Body must be JSON::

            {
              "image_b64": "<base64 PNG/JPEG bytes>",
              "labels":    ["cat", "dog", ...],
              "client_hint": "optional string"
            }

        The image is decoded, converted to RGBA float32, and stored.
        If the image has no alpha channel a fully opaque mask is synthesised.
        """
        wd = _web_dataset()
        if wd is None:
            self._send(503, "application/json", _json({"error": "web dataset not configured"}))
            return
        try:
            req = json.loads(body or b"{}")
        except Exception:
            self._send(400, "application/json", _json({"error": "invalid JSON body"}))
            return
        image_b64 = req.get("image_b64") or ""
        labels = list(req.get("labels") or [])
        client_hint = str(req.get("client_hint") or "")
        if not image_b64:
            self._send(400, "application/json", _json({"error": "image_b64 required"}))
            return
        try:
            import base64, io as _io
            import numpy as _np
            raw_bytes = base64.b64decode(image_b64)
            # Decode image using stdlib struct + a minimal PNG/JPEG reader.
            # We rely on PIL if available; otherwise return an error with instructions.
            try:
                from PIL import Image as _PilImage
                img = _PilImage.open(_io.BytesIO(raw_bytes))
                img_rgba = img.convert("RGBA")
                rgba_arr = _np.asarray(img_rgba, dtype=_np.float32) / 255.0  # [H,W,4]
            except ImportError:
                self._send(503, "application/json",
                           _json({"error": "Pillow not installed on server; cannot decode image"}))
                return
        except Exception as exc:
            self._send(400, "application/json", _json({"error": f"image decode failed: {exc}"}))
            return
        try:
            sample_id = wd.add_sample(
                slot_name=str(slot_name),
                rgba=rgba_arr,
                labels=[str(l) for l in labels],
                client_hint=client_hint,
            )
        except Exception as exc:
            self._send(500, "application/json", _json({"error": f"store failed: {exc}"}))
            return
        self._send(200, "application/json", _json({
            "sample_id": sample_id,
            "slot_name": str(slot_name),
            "labels": labels,
            "width": int(rgba_arr.shape[1]),
            "height": int(rgba_arr.shape[0]),
        }))

    def _handle_lease_open_collection(self, ls, body: bytes):
        if ls is None:
            self._send(503, "application/json", _json({"error": "lease store not configured"}))
            return
        try:
            req = json.loads(body or b"{}")
        except Exception:
            self._send(400, "application/json", _json({"error": "invalid JSON body"}))
            return
        slot_name = str(req.get("slot_name") or "").strip()
        if not slot_name:
            self._send(400, "application/json", _json({"error": "slot_name required"}))
            return
        generation   = int(req.get("generation", 0))
        count        = max(1, int(req.get("count", 1)))
        ttl_seconds  = float(req.get("ttl_seconds", 3600))
        deadline_sec = req.get("deadline_seconds")
        note         = str(req.get("note", ""))
        coll, leases = ls.open_collection(
            slot_name=slot_name,
            generation=generation,
            count=count,
            ttl_seconds=ttl_seconds,
            deadline_seconds=float(deadline_sec) if deadline_sec is not None else None,
            note=note,
        )
        import dataclasses as _dc
        self._send(200, "application/json", _json({
            "collection": _dc.asdict(coll),
            "leases": [_dc.asdict(l) for l in leases],
        }))

    def _handle_lease_return(self, ls, lease_id: str, body: bytes):
        if ls is None:
            self._send(503, "application/json", _json({"error": "lease store not configured"}))
            return
        # Body may be raw npz bytes (application/octet-stream) or empty
        gradient_data = body if body else None
        lease, completed_coll = ls.return_lease(lease_id, gradient_data)
        if lease is None:
            self._send(404, "application/json",
                       _json({"error": "lease not found or already resolved"}))
            return
        import dataclasses as _dc
        self._send(200, "application/json", _json({
            "lease": _dc.asdict(lease),
            "collection_complete": completed_coll is not None,
            "collection_id": lease.collection_id,
        }))

    def _handle_lease_expire_collection(self, ls, collection_id: str):
        if ls is None:
            self._send(503, "application/json", _json({"error": "lease store not configured"}))
            return
        n = ls.force_expire_collection(collection_id)
        self._send(200, "application/json", _json({
            "collection_id": collection_id,
            "expired_count": n,
        }))

    def _handle_lease_mark_applied(self, ls, collection_id: str):
        if ls is None:
            self._send(503, "application/json", _json({"error": "lease store not configured"}))
            return
        ok = ls.mark_applied(collection_id)
        self._send(200, "application/json", _json({
            "collection_id": collection_id,
            "applied": ok,
        }))

    # ------------------------------------------------------------------ #
    #  DELETE routing + handlers                                           #
    # ------------------------------------------------------------------ #

    def _dispatch_delete(self, path: str):
        ls = _lease_store()

        # Abandon a lease (client gives up, no gradient)
        m = re.fullmatch(r"/api/lease/([^/]+)", path)
        if m:
            return self._handle_lease_abandon(ls, m.group(1))

        self._send(404, "application/json", _json({"error": "not found", "path": path}))

    def _handle_lease_abandon(self, ls, lease_id: str):
        if ls is None:
            self._send(503, "application/json", _json({"error": "lease store not configured"}))
            return
        ok = ls.abandon_lease(lease_id)
        if not ok:
            self._send(404, "application/json",
                       _json({"error": "lease not found or already resolved"}))
            return
        self._send(200, "application/json", _json({"abandoned": True, "lease_id": lease_id}))

    # ------------------------------------------------------------------ #
    #  Response helper                                                     #
    # ------------------------------------------------------------------ #

    def _send(self, code: int, content_type: str, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# HTML index page
# ---------------------------------------------------------------------------

_INDEX_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Nodus Data Server</title>
<style>
  body { font-family: monospace; background: #111; color: #ccc; padding: 2em; }
  h1   { color: #fa0; }
  h2   { color: #8cf; margin-top: 1.5em; }
  a    { color: #6df; }
  table { border-collapse: collapse; margin: 0.5em 0; }
  td, th { border: 1px solid #444; padding: 0.3em 0.8em; text-align: left; }
  th   { background: #222; color: #8cf; }
  code { background: #222; padding: 0.1em 0.4em; border-radius: 3px; }
</style>
</head>
<body>
<h1>Nodus Data Server</h1>
<p>Read-only HTTP interface to the nodus shared-memory training data stores.</p>
<p><strong><a href="/train">&#9654; Open Browser Training Console</a></strong></p>

<h2>Status</h2>
<table>
  <tr><th>Endpoint</th><th>Description</th></tr>
  <tr><td><a href="/api/status">/api/status</a></td><td>Overall store status (JSON)</td></tr>
  <tr><td><a href="/api/runtime/control">/api/runtime/control</a></td><td>Runtime-control service priority + exit flags</td></tr>
</table>

<h2>Loss Store</h2>
<table>
  <tr><th>Endpoint</th><th>Description</th></tr>
  <tr><td><a href="/api/loss/channels">/api/loss/channels</a></td><td>All channels + summary</td></tr>
  <tr><td>/api/loss/channel/&lt;key&gt;/records?from_step=0&amp;max=10000</td><td>Records for a channel</td></tr>
  <tr><td>/api/loss/channel/&lt;key&gt;/latest</td><td>Most recent record</td></tr>
  <tr><td><a href="/api/loss/graph.png?w=800&amp;h=200">/api/loss/graph.png</a>
      &nbsp;<small>?w=800&amp;h=200&amp;start=0.0&amp;bg=0,0,0</small></td>
      <td>Rendered loss graph (PNG)</td></tr>
</table>

<h2>Scrub Ring</h2>
<table>
  <tr><th>Endpoint</th><th>Description</th></tr>
  <tr><td><a href="/api/scrub/info">/api/scrub/info</a></td><td>Ring length / capacity / cursor</td></tr>
  <tr><td><a href="/api/scrub/latest/meta">/api/scrub/latest/meta</a></td><td>Newest entry metadata</td></tr>
  <tr><td><a href="/api/scrub/latest/training.png">/api/scrub/latest/training.png</a></td><td>Newest training image</td></tr>
  <tr><td><a href="/api/scrub/latest/output.png">/api/scrub/latest/output.png</a></td><td>Newest output image</td></tr>
  <tr><td><a href="/api/scrub/latest/target.png">/api/scrub/latest/target.png</a></td><td>Newest target image</td></tr>
  <tr><td>/api/scrub/&lt;idx&gt;/meta</td><td>Entry metadata by index</td></tr>
  <tr><td>/api/scrub/&lt;idx&gt;/training.png</td><td>Training image by index</td></tr>
  <tr><td>/api/scrub/&lt;idx&gt;/output.png</td><td>Output image by index</td></tr>
  <tr><td>/api/scrub/&lt;idx&gt;/target.png</td><td>Target image by index</td></tr>
</table>

<h2>Weight Stores</h2>
<table>
  <tr><th>Endpoint</th><th>Description</th></tr>
  <tr><td><a href="/api/weight/meta">/api/weight/meta</a></td><td>Latest weight-state snapshot</td></tr>
  <tr><td><a href="/api/weight/meta/all">/api/weight/meta/all</a></td><td>All retained weight-state entries</td></tr>
  <tr><td><a href="/api/weight/image/info">/api/weight/image/info</a></td><td>Image-cache stats</td></tr>
  <tr><td><a href="/api/weight/image/latest.png">/api/weight/image/latest.png</a></td><td>Newest rendered weight image</td></tr>
  <tr><td>/api/weight/image/&lt;idx&gt;.png</td><td>Weight image by index</td></tr>
</table>

<h2>Browser Training Data</h2>
<p>Requires <code>--lease-store-dir</code>.  Browser clients submit labeled RGBA images
(alpha = mask).  Samples accumulate until the <em>WebLeaseNode</em> pipeline stage
claims them into a collection.</p>
<table>
  <tr><th>Method</th><th>Endpoint</th><th>Description</th></tr>
  <tr><td>POST</td><td>/api/web/dataset/&lt;slot&gt;</td><td>Submit labeled image — body: <code>{"image_b64":"&lt;base64 PNG/JPEG&gt;","labels":[...],"client_hint":"..."}</code></td></tr>
  <tr><td>GET</td><td><a href="/api/web/dataset/status">/api/web/dataset/status</a></td><td>Pending sample counts per slot</td></tr>
  <tr><td>GET</td><td>/api/web/dataset/&lt;slot&gt;/&lt;sample_id&gt;/image.png</td><td>Preview a submitted image (RGBA)</td></tr>
  <tr><td>GET</td><td>/api/web/lease/&lt;lease_id&gt;/weights</td><td>Download model weights for a lease (PyTorch .pt binary)</td></tr>
  <tr><td>GET</td><td>/api/web/lease/&lt;lease_id&gt;/weights.onnx</td><td>Download model weights as ONNX (must export first)</td></tr>
  <tr><td>GET</td><td>/api/web/lease/&lt;lease_id&gt;/dataset</td><td>JSON list of dataset samples assigned to this lease</td></tr>
</table>

<h2>ONNX Export</h2>
<p>Requires <code>--lease-store-dir</code>. Converts a <code>weights.pt</code> checkpoint into <code>model.onnx</code> for browser/native ONNX Runtime inference.</p>
<table>
  <tr><th>Method</th><th>Endpoint</th><th>Description</th></tr>
  <tr><td>POST</td><td>/api/onnx/export</td><td>Export .pt to ONNX &mdash; body: <code>{"collection_id":"...","model_class":"TinyConvClassifier","input_shape":[1,3,64,64],"opset_version":17}</code></td></tr>
</table>

<h2>Latest Model (Live Weights)</h2>
<p>Requires <code>--checkpoint-dir</code>. Serves the freshest model weights from the pipeline&rsquo;s
output directory. ONNX conversion is done on-demand and cached until the <code>.pt</code> file changes.</p>
<table>
  <tr><th>Method</th><th>Endpoint</th><th>Description</th></tr>
  <tr><td>GET</td><td><a href="/api/model/latest">/api/model/latest</a></td><td>List available model checkpoints with timestamps</td></tr>
  <tr><td>GET</td><td>/api/model/latest/classifier/meta</td><td>Latest classifier labels, texts, and input shape</td></tr>
  <tr><td>POST</td><td>/api/model/latest/classifier/infer</td><td>Run latest classifier on <code>{"image_b64":"..."}</code> and return the full labeled score vector</td></tr>
  <tr><td>GET</td><td>/api/model/latest/&lt;name&gt;.onnx</td><td>On-demand ONNX &mdash; <code>?model_class=&amp;input_shape=1,3,64,64&amp;opset=17</code></td></tr>
  <tr><td>GET</td><td>/api/model/latest/&lt;name&gt;.pt</td><td>Raw PyTorch checkpoint</td></tr>
</table>

<h2>Distributed Gradient Leases</h2>
<p>Requires <code>--lease-store-dir</code>.  A <em>collection</em> is a batch of leases
for one network slot at a fixed generation.  The slot is <strong>locked</strong> (local
training paused) while any lease is outstanding.</p>
<table>
  <tr><th>Method</th><th>Endpoint</th><th>Description</th></tr>
  <tr><td>GET</td><td><a href="/api/lease/status">/api/lease/status</a></td><td>All slots with lockout state</td></tr>
  <tr><td>GET</td><td>/api/lease/collection/&lt;id&gt;</td><td>Collection detail + lease list</td></tr>
  <tr><td>GET</td><td>/api/lease/&lt;lease_id&gt;</td><td>Single lease detail</td></tr>
  <tr><td>POST</td><td>/api/lease/collection</td><td>Open collection, issue leases — body: <code>{"slot_name","generation","count","ttl_seconds","deadline_seconds","note"}</code></td></tr>
  <tr><td>POST</td><td>/api/lease/&lt;lease_id&gt;/gradients</td><td>Return lease with optional gradient blob (raw npz bytes body)</td></tr>
  <tr><td>POST</td><td>/api/lease/collection/&lt;id&gt;/expire</td><td>Force-expire all active leases, unlock slot immediately</td></tr>
  <tr><td>POST</td><td>/api/lease/collection/&lt;id&gt;/apply</td><td>Mark collection as applied (gradients consumed by pipeline)</td></tr>
  <tr><td>DELETE</td><td>/api/lease/&lt;lease_id&gt;</td><td>Abandon lease (no gradient returned)</td></tr>
</table>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Serve nodus shared-memory data over HTTP.")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7272,
                        help="TCP port (default: 7272)")
    parser.add_argument("--lease-store-dir", default="",
                        help="Directory for distributed gradient lease state. "
                             "If set, enables /api/lease/* endpoints.")
    parser.add_argument("--checkpoint-dir", default="",
                        help="Pipeline output directory containing latest .pt checkpoints. "
                             "If set, enables /api/model/latest/* endpoints that serve "
                             "the freshest weights with on-demand ONNX conversion.")
    args = parser.parse_args()

    # Eagerly load the shared-memory stores so startup errors surface immediately.
    print(f"[nodus] Loading shared-memory stores ...", flush=True)
    try:
        loss, scrub, wst, wim, control = _get_stores()
        global _STORES
        _STORES = (loss, scrub, wst, wim, control)
        print(f"[nodus] Connected: {loss.channel_count()} loss channel(s), "
              f"scrub ring {scrub.length()}/{scrub.capacity()}, "
              f"{wim.length()} weight image(s), "
              f"service_active={control.active_services()}.", flush=True)
    except Exception as exc:
        print(f"[nodus] WARNING: Could not load stores: {exc}", flush=True)
        print(f"[nodus] Server will still start; store errors will be "
              f"reported per-request.", flush=True)

    # Optionally init the lease store + web dataset.
    global _LEASE_STORE, _WEB_DATASET
    if args.lease_store_dir:
        try:
            from pipeline.lease_store import LeaseStore
            from pipeline.web_dataset import WebDataset
            _LEASE_STORE = LeaseStore(args.lease_store_dir)
            _WEB_DATASET = WebDataset(args.lease_store_dir)
            print(f"[nodus] Lease store + web dataset: {args.lease_store_dir}", flush=True)
            # Background thread to expire stale leases every 60 s
            def _expiry_loop():
                import time as _time
                while True:
                    _time.sleep(60)
                    try:
                        _LEASE_STORE.expire_stale_leases()
                    except Exception:
                        pass
            t = threading.Thread(target=_expiry_loop, daemon=True, name="lease-expiry")
            t.start()
        except Exception as exc:
            print(f"[nodus] WARNING: Could not init lease store: {exc}", flush=True)

    # Optionally set checkpoint directory for latest-model endpoints.
    global _CHECKPOINT_DIR
    if args.checkpoint_dir:
        _CHECKPOINT_DIR = args.checkpoint_dir
        print(f"[nodus] Checkpoint dir (latest model): {_CHECKPOINT_DIR}", flush=True)

    server = ThreadingHTTPServer((args.host, args.port), NodusHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"[nodus] Serving at {url}", flush=True)
    try:
        control = _control()
    except Exception:
        control = None
    if control is not None:
        try:
            control.clear_exit_requested()
        except Exception:
            pass
        def _shutdown_watch():
            while True:
                time.sleep(0.5)
                try:
                    state = control.get_state()
                except Exception:
                    state = None
                if state is None:
                    continue
                if bool(getattr(state, "exit_requested", False)):
                    reason = str(getattr(state, "exit_reason", "") or "runtime_exit")
                    print(f"[nodus] Exit requested via runtime control ({reason}); shutting down.", flush=True)
                    try:
                        server.shutdown()
                    except Exception:
                        pass
                    return
        threading.Thread(target=_shutdown_watch, daemon=True, name="runtime-control-watch").start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[nodus] Shutting down.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
