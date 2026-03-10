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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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


def _build_reference_flashcard_payload_rows(
    class_names: Sequence[str],
    condition_num_classes: int,
    supervised_num_classes: int,
    payload_images_base: Sequence["np.ndarray"],
    payload_conditions_supervised_base: Sequence["np.ndarray"],
    symbol_pool_by_term: Dict[str, List["np.ndarray"]],
    image_size: int,
    per_term: int,
    seed: int,
    condition_vector_builder: Callable[[Sequence[str], Optional["np.ndarray"]], "np.ndarray"],
    gan_image_provider: Optional[Callable[[int], List["np.ndarray"]]] = None,
) -> Tuple[List["np.ndarray"], List["np.ndarray"], Dict[str, Any]]:
    import math
    import re
    import numpy as np
    import torch
    import torch.nn.functional as F

    from wav_config_transformer_pipeline import (
        _default_semantic_core_terms,
        _find_semantic_term_index,
        _image_any_to_rgb_chw01,
        _normalize_vocab_terms,
        _semantic_active_target_stats,
        _semantic_damage_tags,
        _semantic_noise_profile_key_from_term,
        _semantic_noise_profile_terms,
        _semantic_noise_terms_from_spectrum_sample,
        _semantic_terms_with_tonal_tags,
    )

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
    berkeley_idx = int(_find_semantic_term_index(class_names, "berkeley sbd dataset"))

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
        object_seed_rows.append((
            _image_any_to_rgb_chw01(payload_images_base[int(i)], image_size=int(size)),
            np.asarray(payload_conditions_supervised_base[int(i)], dtype=np.float32).reshape(-1),
        ))

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
                raise RuntimeError("Reference flashcard row requires Berkeley-supervised seed content, but payload object rows are empty.")
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

    damage_terms = {"blur damage", "noise damage", "dropout damage", "quantization damage", "stride skew damage"}

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
                raise RuntimeError(f"Berkeley/object flashcard row emitted without supervised label vector. term={str(term)!r}")
            sup_arr = np.asarray(base_supervised, dtype=np.float32).reshape(-1)
            sup_take = min(int(sup_arr.size), int(vec.size))
            if int(sup_take) <= 0:
                raise RuntimeError(f"Berkeley/object flashcard row has empty supervised label vector. term={str(term)!r}")
            sup_clip = np.clip(np.asarray(sup_arr[: int(sup_take)], dtype=np.float32), 0.0, 1.0)
            vec_clip = np.clip(np.asarray(vec[: int(sup_take)], dtype=np.float32), 0.0, 1.0)
            missing = (sup_clip >= 0.5) & (vec_clip < 0.5)
            if bool(np.any(missing)):
                raise RuntimeError(f"Berkeley/object flashcard row dropped original supervised labels during conditioning. term={str(term)!r}")
            if int(berkeley_idx) < 0:
                raise RuntimeError("Berkeley/object flashcard row cannot be emitted because 'berkeley sbd dataset' is missing from semantic class names.")
            if int(vec.size) <= int(berkeley_idx) or float(vec[int(berkeley_idx)]) < 0.5:
                raise RuntimeError(f"Berkeley/object flashcard row is missing required 'berkeley sbd dataset' target flag. term={str(term)!r}")
        cards_img.append(img_rgb)
        cards_cond.append(np.asarray(vec, dtype=np.float32).reshape(-1))
        term_counts[key] = int(term_counts.get(key, 0)) + 1

    for term in target_terms:
        term_key = re.sub(r"\s+", " ", str(term)).strip().lower()
        for _ in range(int(per)):
            if term_key == "none":
                _emit(term_key, _to_rgb(np.full((size, size), 0.50, dtype=np.float32)), None, ["none"])
                continue
            if term_key == "noise":
                noise_profile = _sample_white_noise_profile_key()
                noise_img = _noise_pattern(noise_profile)
                _emit(term_key, _to_rgb(noise_img), None, list(_semantic_noise_terms_from_spectrum_sample(noise_img)))
                continue
            if term_key.endswith("noise"):
                noise_profile = _semantic_noise_profile_key_from_term(term_key)
                if noise_profile:
                    noise_img = _noise_pattern(noise_profile)
                    _emit(
                        term_key,
                        _to_rgb(noise_img),
                        None,
                        _normalize_vocab_terms(
                            list(_semantic_noise_profile_terms(noise_profile))
                            + list(_semantic_noise_terms_from_spectrum_sample(noise_img))
                        ),
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
                    mixed = _mix_signal_with_noise_pcm(sig=sig, noi=noi, noise_bits=int(rng.integers(2, 6)))
                _emit(term_key, mixed, None, ["signal", "mixed noise and signal"] + list(_semantic_noise_terms_from_spectrum_sample(noise_img)))
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
                _emit(term_key, _to_rgb(np.clip((0.55 * sig0) + (0.45 * sig1), 0.0, 1.0)), None, ["regurgitated content", "signal"])
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
                _emit(term_key, dmg, obj_sup, list(dmg_extra_terms) + ["object", "berkeley sbd dataset"])
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
