"""
Vocabulary Node — semantic vocabulary management and churn rotation.

This file is the authoritative description of all vocabulary-related
orchestration in the pipeline.

What "vocabulary" means here
------------------------------
The classifier is trained against a fixed-width multi-hot target vector.
Each dimension corresponds to a semantic "class name" (e.g. "white noise",
"red", "front", "edge").  The vocabulary is partitioned into:

  core_terms     — always-present, locked terms (noise colour profiles, etc.)
  extra_terms    — rotating pool of richer semantic concepts

The VocabChurnNode rotates the extra_terms pool on a configurable cycle so the
classifier is exposed to a wider range of semantic concepts over a long run
without exceeding the fixed target-vector width.

Responsibilities owned here
-----------------------------
  * Loading vocab terms from JSON files or defaults
  * Merging core + extra terms into the active class_names list
  * Per-cycle churn: replace N extra terms with fresh candidates from the pool
  * Rebuilding the label embedding bank after each churn event
  * Propagating updated class_names → ctx.semantic_term_to_idx
  * Rebuilding payload conditions after vocab changes
  * Rebuilding symbol pool and flashcard rows after vocab changes
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.context import PipelineContext
from pipeline.graph import PipelineNode
from pipeline.nodes.base import OneTimeNode


# ---------------------------------------------------------------------------
# Node config
# ---------------------------------------------------------------------------

@dataclass
class VocabConfig:
    """Every hyperparameter specific to vocabulary management."""

    # ---- vocab sources --------------------------------------------------
    # Path to a JSON file of extra semantic terms (list[str] or dict).
    extra_terms_json: str = ""

    # Additional terms supplied directly on the CLI / config
    extra_terms_inline: List[str] = field(default_factory=list)

    # Number of semantic extra slots added on top of the supervised Berkeley labels.
    total_slots: int = 64

    # ---- churn schedule -------------------------------------------------
    # Number of extra_terms to replace per churn event
    churn_n: int = 4

    # Churn happens every N vocab rotation cycles (outer loop cycles)
    churn_every_n_cycles: int = 1

    # When True, newly churned terms are appended to the end of the
    # extra list instead of replacing random positions
    churn_append_mode: bool = False

    # ---- symbol pools ---------------------------------------------------
    # "auto"      → load from official image datasets (MNIST, EMNIST, …)
    # "synthetic" → render synthetic geometric symbols
    # "internal"  → load from internal bootstrap data root
    symbol_pool_mode: str = "synthetic"
    symbol_pool_samples_per_term: int = 16

    # ---- flashcard rows -------------------------------------------------
    flashcard_enabled: bool = True
    flashcard_rows_per_term: int = 4

    # ---- removed seed terms (never added to churn) ----------------------
    removed_seed_terms: List[str] = field(default_factory=list)

    # ---- symbol pool / flashcard parameters -------------------------
    seed: int = 0
    image_size: int = 128
    symbol_pool_root: str = ""
    symbol_include_pictograms: bool = True
    symbol_bootstrap_origin_label: str = "internal bootstrap root vocab"
    extra_terms_cli: str = ""
    classifier_init_ckpt: str = ""


# ---------------------------------------------------------------------------
# Initial vocab setup node (runs once at pipeline start)
# ---------------------------------------------------------------------------

class InitVocabNode(OneTimeNode):
    """Load and assemble the initial vocabulary at pipeline start.

    Merges defaults + JSON file + inline terms, trims to total_slots,
    and populates ctx.class_names, ctx.core_terms, ctx.active_extra_terms,
    and ctx.semantic_term_to_idx.
    """

    node_id = "init_vocab"
    description = "Load and assemble initial semantic vocabulary"

    def __init__(self, cfg: VocabConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def _execute_once(self, ctx: PipelineContext) -> None:
        from wav_config_transformer_pipeline import (
            _default_semantic_core_terms,
            _load_vocab_terms_json,
            _normalize_active_extra_terms_with_core,
            _semantic_term_index_map,
        )

        supervised = _resolve_supervised_class_names(ctx)
        core = _default_semantic_core_terms()

        extra: List[str] = list(self.cfg.extra_terms_inline)
        if str(self.cfg.extra_terms_json).strip():
            extra = extra + _load_vocab_terms_json(self.cfg.extra_terms_json)

        # Also pick up terms from CLI / config if present
        cli_extra = self.cfg.extra_terms_cli or (
            getattr(ctx.args, "semantic_vocab_extra_texts", None)
            or getattr(ctx.args, "extra_semantic_terms", "")
            or ""
        )
        if str(cli_extra).strip():
            from wav_config_transformer_pipeline import _parse_label_query_texts
            extra = extra + _parse_label_query_texts(cli_extra)

        active_extra = _normalize_active_extra_terms_with_core(
            active_terms=extra,
            core_terms=core,
            total_slots=self.cfg.total_slots,
        )

        ctx.supervised_class_names = list(supervised)
        ctx.core_terms = list(core)
        ctx.active_extra_terms = list(active_extra)
        ctx.class_names = list(supervised) + list(active_extra)
        ctx.semantic_term_to_idx = _semantic_term_index_map(ctx.class_names)

        _log(
            f"[vocab-init] classes={len(ctx.class_names)} "
            f"supervised={len(ctx.supervised_class_names)} "
            f"extras={len(ctx.active_extra_terms)}"
        )


# ---------------------------------------------------------------------------
# Vocab activation / churn node (runs every cycle)
# ---------------------------------------------------------------------------

class VocabChurnNode(PipelineNode):
    """Rotate extra vocabulary terms once per churn cycle.

    Selects N replacement terms from the full pool of available candidates
    and updates ctx.class_names and ctx.semantic_term_to_idx.

    After churn, dependent resources (label bank, payload conditions,
    symbol pool, flashcard rows) are rebuilt in subsequent nodes.
    """

    node_id = "vocab_churn"
    description = "Rotate extra vocab terms (churn) on schedule"

    def __init__(self, cfg: VocabConfig) -> None:
        self.cfg = cfg

    def should_run(self, ctx: PipelineContext) -> bool:
        if self.cfg.churn_n <= 0:
            return False
        return (ctx.vocab_rotation_cycle % max(1, self.cfg.churn_every_n_cycles)) == 0

    def execute(self, ctx: PipelineContext) -> None:
        from wav_config_transformer_pipeline import (
            _rotate_active_extra_terms,
            _semantic_term_index_map,
        )

        old_terms = list(ctx.active_extra_terms)

        new_active, churn_info = _rotate_active_extra_terms(
            active_terms=ctx.active_extra_terms,
            pool_terms=_build_full_term_pool(ctx, self.cfg),
            replace_count=self.cfg.churn_n,
            seed=self.cfg.seed,
            locked_prefix_count=int(len(ctx.core_terms)),
            churn_cursor=int(ctx.vocab_rotation_cycle),
            sweep_cycles=int(max(0, self.cfg.churn_every_n_cycles)),
        )

        ctx.active_extra_terms = list(new_active)
        ctx.class_names = list(ctx.supervised_class_names) + list(new_active)
        ctx.semantic_term_to_idx = _semantic_term_index_map(ctx.class_names)
        ctx.vocab_rotation_cycle += 1

        changed = [t for t in new_active if t not in old_terms]
        _log(
            f"[vocab-churn] cycle={ctx.vocab_rotation_cycle} "
            f"replaced={int(churn_info.get('replaced', len(changed)))} "
            f"changed={changed[:4]}{'…' if len(changed) > 4 else ''}"
        )


# ---------------------------------------------------------------------------
# Symbol pool build node
# ---------------------------------------------------------------------------

class BuildSymbolPoolNode(PipelineNode):
    """Build or rebuild the symbol image pool after vocab changes.

    The symbol pool provides per-term reference images used in Stage 0/1
    data generation and GAN conditioning.
    """

    node_id = "build_symbol_pool"
    description = "Build per-term symbol image pool"

    def __init__(self, cfg: VocabConfig) -> None:
        self.cfg = cfg

    def execute(self, ctx: PipelineContext) -> None:
        from wav_config_transformer_pipeline import (
            _build_auto_symbol_term_pool,
            _build_synthetic_semantic_symbol_pool,
            _build_internal_bootstrap_symbol_pool,
        )

        mode = str(self.cfg.symbol_pool_mode).strip().lower()
        if mode == "auto":
            pool, info = _build_auto_symbol_term_pool(
                data_root=self.cfg.symbol_pool_root,
                image_size=self.cfg.image_size,
                seed=self.cfg.seed,
                max_samples_per_term=self.cfg.symbol_pool_samples_per_term,
                include_pictograms=self.cfg.symbol_include_pictograms,
            )
        elif mode == "internal":
            pool, info = _build_internal_bootstrap_symbol_pool(
                data_root=self.cfg.symbol_pool_root,
                image_size=self.cfg.image_size,
                seed=self.cfg.seed,
                max_samples_per_term=self.cfg.symbol_pool_samples_per_term,
                origin_label=self.cfg.symbol_bootstrap_origin_label,
            )
        else:  # "synthetic"
            pool, info = _build_synthetic_semantic_symbol_pool(
                image_size=self.cfg.image_size,
                seed=self.cfg.seed,
                max_samples_per_term=self.cfg.symbol_pool_samples_per_term,
            )

        ctx.symbol_pool = pool
        _log(
            f"[symbol-pool] built mode={mode} terms={len(pool) if pool else 0} "
            f"samples={int(info.get('samples', 0)) if isinstance(info, dict) else 0}"
        )


# ---------------------------------------------------------------------------
# Flashcard rows build node
# ---------------------------------------------------------------------------

class BuildFlashcardRowsNode(PipelineNode):
    """Build per-term reference flashcard rows for GAN payload conditioning.

    Flashcard rows are pre-composed (image, condition_vector) pairs — one set
    per active semantic term — used as supervised targets during GAN training
    to anchor the generator's conditional representation.
    """

    node_id = "build_flashcard_rows"
    description = "Build per-term GAN conditioning flashcard rows"

    def __init__(self, cfg: VocabConfig) -> None:
        self.cfg = cfg

    def should_run(self, ctx: PipelineContext) -> bool:
        return self.cfg.flashcard_enabled and ctx.payload_bank is not None

    def execute(self, ctx: PipelineContext) -> None:
        import numpy as np
        from wav_config_transformer_pipeline import _build_reference_flashcard_payload_rows

        # _build_reference_flashcard_payload_rows(
        #   class_names, condition_num_classes, supervised_num_classes,
        #   payload_images_base, payload_conditions_supervised_base,
        #   symbol_pool_by_term, image_size, per_term, seed,
        #   condition_vector_builder, gan_image_provider=None
        # ) -> (flashcard_images, flashcard_conditions, info_dict)
        #
        # condition_vector_builder: Callable[[Sequence[str], Optional[np.ndarray]], np.ndarray]
        # Build a multi-hot condition vector over ctx.class_names for the given terms.
        class_names_local = list(ctx.class_names)
        name_to_idx = {n.lower(): i for i, n in enumerate(class_names_local)}

        def _condition_builder(terms, embedding=None):
            vec = np.zeros(len(class_names_local), dtype=np.float32)
            for t in (terms or []):
                idx = name_to_idx.get(str(t).lower())
                if idx is not None:
                    vec[idx] = 1.0
            return vec

        flashcard_images, flashcard_conditions, info = _build_reference_flashcard_payload_rows(
            class_names=class_names_local,
            condition_num_classes=len(class_names_local),
            supervised_num_classes=len(ctx.supervised_class_names),
            payload_images_base=ctx.payload_bank if ctx.payload_bank is not None else [],
            payload_conditions_supervised_base=ctx.payload_conditions,
            symbol_pool_by_term=ctx.symbol_pool or {},
            image_size=self.cfg.image_size,
            per_term=self.cfg.flashcard_rows_per_term,
            seed=self.cfg.seed,
            condition_vector_builder=_condition_builder,
        )

        ctx.flashcard_rows = list(zip(flashcard_images, flashcard_conditions))
        _log(f"[flashcard] built {len(ctx.flashcard_rows)} rows "
             f"(info: {info.get('rows_added', 0)} added)")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_full_term_pool(ctx: PipelineContext, cfg: VocabConfig) -> List[str]:
    """Assemble the full pool of candidate extra terms for churn selection."""
    from wav_config_transformer_pipeline import (
        _default_bootstrap_primitive_terms,
        _load_vocab_terms_json,
        _normalize_vocab_terms,
        _merge_vocab_terms,
        _removed_semantic_seed_terms,
    )

    base = _default_bootstrap_primitive_terms()
    extra: List[str] = list(cfg.extra_terms_inline)
    if str(cfg.extra_terms_json).strip():
        extra = extra + _load_vocab_terms_json(cfg.extra_terms_json)

    removed = set(_removed_semantic_seed_terms() + list(cfg.removed_seed_terms))
    merged = _merge_vocab_terms(base, extra)
    return _normalize_vocab_terms([t for t in merged if t.lower() not in removed])


def _resolve_supervised_class_names(ctx: PipelineContext) -> List[str]:
    from wav_config_transformer_pipeline import _default_berkeley_class_names, _torch_load_cpu

    ckpt_path = str(getattr(ctx.args, "classifier_init_ckpt", "") or "").strip()
    if ckpt_path:
        p = Path(ckpt_path)
        if p.exists():
            try:
                blob = _torch_load_cpu(str(p))
                raw = blob.get("class_names", [])
                if isinstance(raw, (list, tuple)):
                    names = [str(x).strip() for x in raw if str(x).strip()]
                    if names:
                        return names
                num_classes = int(blob.get("num_classes", 0))
                if num_classes > 0:
                    return [f"berkeley_cls_{i}" for i in range(num_classes)]
            except Exception as exc:
                _log(f"[vocab-init] WARNING: could not read classifier init metadata from {p}: {exc}")
    return [str(x).strip() for x in _default_berkeley_class_names() if str(x).strip()]


def _log(msg: str) -> None:
    print(msg, flush=True)
