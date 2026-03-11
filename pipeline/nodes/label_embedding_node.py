"""
Label Embedding Node — sentence-transformer semantic embedding bank.

This file owns everything specific to the semantic embedding machinery:

  * Which sentence-transformer model is used (all-MiniLM-L6-v2 by default)
  * How class names are resolved from the active vocabulary
  * Optional JSON override file for label text
  * Temperature of the cosine similarity head
  * L2-normalisation and embedding bank installation into TinyConvClassifier
  * Caching of the SentenceTransformer model across calls (_ST_MODEL_CACHE)

The embedding bank runs after every vocab churn event because the class_names
list changes and the bank dimensions must match.  The node is cheap (CPU-side
sentence-transformer inference) so re-running each round is acceptable.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from pipeline.context import PipelineContext
from pipeline.graph import PipelineNode


# ---------------------------------------------------------------------------
# Node config
# ---------------------------------------------------------------------------

@dataclass
class LabelEmbeddingConfig:
    """Every hyperparameter specific to the semantic label embedding bank."""

    # ---- sentence-transformer backend -----------------------------------
    backend: str = "sentence_transformers"     # currently the only supported backend
    model_name: str = "all-MiniLM-L6-v2"      # HuggingFace model name

    # Optional JSON file that overrides class-name display text before encoding.
    # Format: list[str] (same length as class_names) or dict[str|int, str].
    label_text_json: str = ""

    # Embedding dimensionality.  0 = use model's native dimensionality.
    # Must match model native dim if > 0.
    target_dim: int = 0

    # Cosine similarity temperature applied inside TinyConvClassifier.
    # Higher → sharper distributions; lower → softer.
    temperature: float = 10.0


# ---------------------------------------------------------------------------
# Embedding bank build node (re-runs after each vocab churn)
# ---------------------------------------------------------------------------

class BuildLabelEmbeddingNode(PipelineNode):
    """Build L2-normalised label embedding bank from active class_names.

    Encodes ctx.class_names via sentence-transformers and installs the
    resulting bank into ctx.classifier (and ctx.wave_classifier if present)
    via TinyConvClassifier.set_label_embedding_bank().

    The bank is also stored in ctx.label_embedding_bank as a numpy array for
    use by zero-shot evaluation and semantic supervision loss.

    Runs whenever class_names has changed (after vocab churn) or when the
    classifier has been freshly built.
    """

    node_id = "build_label_embedding"
    description = "Build sentence-transformer label embedding bank"

    def __init__(self, cfg: LabelEmbeddingConfig) -> None:
        self.cfg = cfg
        self._last_class_names_hash: int = -1

    def should_run(self, ctx: PipelineContext) -> bool:
        if not ctx.class_names or ctx.classifier is None:
            return False
        # Re-run if class_names changed since last build
        current_hash = hash(tuple(ctx.class_names))
        return current_hash != self._last_class_names_hash

    def execute(self, ctx: PipelineContext) -> None:

        bank_np, label_texts, info = _build_label_embedding_bank(
            class_names=ctx.class_names,
            args=_make_embedding_args(self.cfg, ctx),
            device=ctx.device,
        )

        ctx.label_embedding_bank = bank_np
        ctx.label_texts = label_texts
        ctx.label_embedding_info = info

        # Install into the main classifier
        _apply_label_embedding_bank_to_classifier(
            classifier=ctx.classifier,
            bank_np=bank_np,
            args=_make_embedding_args(self.cfg, ctx),
        )

        # Install into the wave classifier if present
        if ctx.wave_classifier is not None:
            _apply_label_embedding_bank_to_classifier(
                classifier=ctx.wave_classifier,
                bank_np=bank_np,
                args=_make_embedding_args(self.cfg, ctx),
            )

        self._last_class_names_hash = hash(tuple(ctx.class_names))

        _log(
            f"[label-embedding] built: n_classes={len(ctx.class_names)} "
            f"dim={int(bank_np.shape[1])} model={self.cfg.model_name!r} "
            f"temp={self.cfg.temperature}"
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

class _EmbeddingArgs:
    """Minimal args-like object so we can call the existing functions
    without passing the entire argparse.Namespace."""

    def __init__(self, cfg: LabelEmbeddingConfig, ctx: PipelineContext) -> None:
        self.label_text_json = cfg.label_text_json
        self.label_embedding_backend = cfg.backend
        self.label_embedding_model = cfg.model_name
        self.label_embedding_dim = cfg.target_dim
        self.label_embedding_temperature = cfg.temperature
        # Forward any remaining attrs from ctx.args
        self._base = ctx.args

    def __getattr__(self, name: str):
        return getattr(self._base, name)


def _make_embedding_args(cfg: LabelEmbeddingConfig, ctx: PipelineContext) -> Any:
    return _EmbeddingArgs(cfg, ctx)


def _log(msg: str) -> None:
    print(msg, flush=True)


# =========================================================================
# Functions extracted from wav_config_transformer_pipeline.py
# =========================================================================

_ST_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}


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
    from wav_ml_models import TinyConvClassifier

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
