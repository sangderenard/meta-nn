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
from typing import Any, Dict, List, Optional

import numpy as np
import torch

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
        from wav_config_transformer_pipeline import (
            _build_label_embedding_bank,
            _apply_label_embedding_bank_to_classifier,
        )

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
