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


# =========================================================================
# Functions extracted from wav_config_transformer_pipeline.py
# =========================================================================

PREGESTATION_MODE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "direction_color": {
        "name": "direction_color",
        "include_diagonal": True,
        "diagonal_d_temp_threshold": 1.05,
        "diagonal_fraction_at_max": 0.50,
        "diagonal_max_d_temp": 3.0,
        "include_depth": False,
        "depth_occluder": None,
        "depth_labels": (),
    },
    "direction_color_depth": {
        "name": "direction_color_depth",
        "include_diagonal": True,
        "diagonal_d_temp_threshold": 1.05,
        "diagonal_fraction_at_max": 0.50,
        "diagonal_max_d_temp": 3.0,
        "include_depth": True,
        "depth_occluder": "cross",
        "depth_labels": ("front", "behind"),
    },
}

_DEFAULT_SEMANTIC_CORE_TERMS: List[str] = [
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

_REMOVED_SEMANTIC_SEED_TERMS = [
    "none",
    "noise",
    "signal",
    "object",
    "mixed noise and signal",
    "mnist dataset",
    "emnist dataset",
    "kmnist dataset",
]


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


def _default_semantic_core_terms() -> List[str]:
    return [str(x) for x in _DEFAULT_SEMANTIC_CORE_TERMS]


def _removed_semantic_seed_terms() -> List[str]:
    return [str(x) for x in _REMOVED_SEMANTIC_SEED_TERMS]


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
    structure_terms = _normalize_vocab_terms(
        ["pattern", "edge", "shape", "texture", "bright", "dark", "smooth", "rough", "object", "signal"]
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
    out.extend([str(t) for t in structure_terms])
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
        "white noise": ["white noise", "gaussian white noise", "uniform white noise"],
        "pink noise": ["pink noise"],
        "brown noise": ["brown noise"],
        "red noise": ["red noise"],
        "blue noise": ["blue noise"],
        "violet noise": ["violet noise"],
        "gray noise": ["gray noise", "grey noise"],
        "gaussian white noise": ["gaussian white noise", "white noise"],
        "uniform white noise": ["uniform white noise", "white noise"],
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
    _REMOVED_SEMANTIC_SEED_TERMS = [
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


    _DEFAULT_SEMANTIC_CORE_TERMS: List[str] = []


    def _default_semantic_core_terms() -> List[str]:
        return [str(x) for x in _DEFAULT_SEMANTIC_CORE_TERMS]


    def _removed_semantic_seed_terms() -> List[str]:
        return [str(x) for x in _REMOVED_SEMANTIC_SEED_TERMS]
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
    return _REMOVED_SEMANTIC_SEED_TERMS


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
        "uniform_white_noise": ["noise", "white noise", "uniform white noise"],
        "gaussian_white_noise": ["noise", "white noise", "gaussian white noise"],
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
            out.extend(_semantic_noise_profile_terms("uniform_white_noise"))
        if key.endswith("noise") and key != "noise":
            profile_key = _semantic_noise_profile_key_from_term(key)
            if profile_key:
                out.extend(_semantic_noise_profile_terms(profile_key))
            else:
                out.extend([key, "noise"])
        if key.endswith("damage"):
            out.extend(_semantic_damage_tags(key))
    return _normalize_vocab_terms(out)


def _semantic_noise_terms_from_spectrum_sample(sample: Any) -> List[str]:
    arr = np.asarray(sample, dtype=np.float32)
    if int(arr.size) <= 0:
        return []
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
    if best_term == "white noise":
        dist_term = "uniform white noise" if float(excess_kurtosis) <= -0.55 else "gaussian white noise"
        return _normalize_vocab_terms(["noise", "white noise", str(dist_term)])
    return _normalize_vocab_terms(["noise", str(best_term)])


def _semantic_tags_for_symbol_term(term: str) -> List[str]:
    key = re.sub(r"\s+", " ", str(term)).strip().lower()
    if not key:
        return []
    berkeley_lut = getattr(_semantic_tags_for_symbol_term, "_berkeley_lut", None)
    if berkeley_lut is None:
        try:
            berkeley_lut = {
                str(x).strip().lower()
                for x in _default_berkeley_class_names()
                if str(x).strip()
            }
            # Keep only object-like Berkeley labels here; dataset tags and symbol namespaces
            # have dedicated handlers below.
            berkeley_lut = {
                str(x)
                for x in berkeley_lut
                if str(x)
                not in {
                    "berkeley sbd dataset",
                    "mnist dataset",
                    "emnist dataset",
                    "kmnist dataset",
                    "object",
                    "signal",
                }
                and (not bool(re.fullmatch(r"digit \d+", str(x))))
                and (not bool(re.fullmatch(r"letter [a-z]", str(x))))
                and (not bool(re.fullmatch(r"pictogram \d+", str(x))))
            }
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
        tags.extend(["black", "dark"])
    if high_frac >= cov_t:
        tags.extend(["white", "bright"])
    if (abs(mean_v - 0.5) <= float(gray_mean_tolerance)) and (std_v <= float(gray_std_threshold)):
        tags.extend(["gray", "grey"])
    # warm/cool color temperature: compare mean R vs mean B channel balance
    if int(arr.shape[0]) >= 3:
        _r_mean = float(np.mean(arr[0]))
        _b_mean = float(np.mean(arr[2]))
        _temp_diff = _r_mean - _b_mean
        if _temp_diff >= 0.10:
            tags.append("warm")
        elif _temp_diff <= -0.10:
            tags.append("cool")
    tags.extend(detect_semantic_color_terms(rgb))
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

    def _spectral_noise(beta: float = 0.0, distribution: str = "gaussian") -> np.ndarray:
        if str(distribution).strip().lower() == "uniform":
            base = rng.uniform(-1.0, 1.0, size=(size, size)).astype(np.float32, copy=False)
        else:
            base = rng.standard_normal((size, size), dtype=np.float32)
        if abs(float(beta)) > 1e-6:
            spec = np.fft.fft2(base).astype(np.complex64, copy=False)
            fy = np.fft.fftfreq(int(size)).astype(np.float32)[:, None]
            fx = np.fft.fftfreq(int(size)).astype(np.float32)[None, :]
            rr = np.sqrt((fx * fx) + (fy * fy)).astype(np.float32, copy=False)
            rr = np.maximum(rr, np.float32(1.0 / float(max(1, size)))).astype(np.float32, copy=False)
            shape = np.power(rr, np.float32(-0.5 * float(beta))).astype(np.float32, copy=False)
            shape[0, 0] = np.float32(0.0)
            base = np.fft.ifft2(spec * shape.astype(np.complex64, copy=False)).real.astype(np.float32, copy=False)
        base = base - float(np.min(base))
        hi = float(np.max(base))
        if hi > 1e-8:
            base = base / hi
        return np.clip(base, 0.0, 1.0).astype(np.float32, copy=False)

    def _to_rgb(g: np.ndarray) -> np.ndarray:
        gg = np.clip(np.asarray(g, dtype=np.float32), 0.0, 1.0)
        return np.repeat(gg[None, :, :], 3, axis=0).astype(np.float32, copy=False)

    def _colorize(g: np.ndarray, rgb: Tuple[float, float, float]) -> np.ndarray:
        gg = np.clip(np.asarray(g, dtype=np.float32), 0.0, 1.0)
        tint = np.asarray(rgb, dtype=np.float32).reshape(3, 1, 1)
        return np.clip(gg[None, :, :] * tint, 0.0, 1.0).astype(np.float32, copy=False)

    def _dot_mask(cx01: float, cy01: float, radius01: float, edge01: float = 0.018) -> np.ndarray:
        dx = x01 - np.float32(cx01)
        dy = y01 - np.float32(cy01)
        dist = np.sqrt((dx * dx) + (dy * dy)).astype(np.float32, copy=False)
        radius = np.float32(max(0.01, float(radius01)))
        edge = np.float32(max(0.004, float(edge01)))
        return np.clip((radius - dist) / edge, 0.0, 1.0).astype(np.float32, copy=False)

    def _colored_dot_scene(term_key: str, sample_idx: int, target_rgb: Tuple[float, float, float]) -> np.ndarray:
        local_seed = int(abs(hash((str(term_key), int(sample_idx), int(seed), int(size)))) % (2**32 - 1))
        local_rng = np.random.default_rng(local_seed)
        bg = np.clip(0.05 + (0.08 * _box_blur(_signal_pattern(phase=0.21 * float(sample_idx)), k=9)), 0.0, 1.0)
        scene = np.repeat(bg[None, :, :], 3, axis=0).astype(np.float32, copy=False)
        anchors = [
            (0.18, 0.18), (0.50, 0.18), (0.82, 0.18),
            (0.18, 0.50), (0.50, 0.50), (0.82, 0.50),
            (0.18, 0.82), (0.50, 0.82), (0.82, 0.82),
        ]
        local_rng.shuffle(anchors)
        target_count = int(local_rng.integers(2, 5))
        tint = np.asarray(target_rgb, dtype=np.float32).reshape(3, 1, 1)
        for cx, cy in anchors[:target_count]:
            rr = float(local_rng.uniform(0.075, 0.12))
            mask = _dot_mask(
                cx01=float(cx) + float(local_rng.uniform(-0.035, 0.035)),
                cy01=float(cy) + float(local_rng.uniform(-0.035, 0.035)),
                radius01=rr,
                edge01=float(local_rng.uniform(0.010, 0.022)),
            )
            dot_gain = np.float32(local_rng.uniform(0.78, 1.0))
            scene = np.clip(scene + ((dot_gain * mask)[None, :, :] * tint), 0.0, 1.0)
        distractor_palette = [
            np.asarray([1.00, 0.52, 0.08], dtype=np.float32),
            np.asarray([0.52, 1.00, 0.12], dtype=np.float32),
            np.asarray([0.38, 0.72, 1.00], dtype=np.float32),
            np.asarray([0.72, 0.40, 1.00], dtype=np.float32),
        ]
        for cx, cy in anchors[target_count: target_count + int(local_rng.integers(2, 5))]:
            mask = _dot_mask(
                cx01=float(cx) + float(local_rng.uniform(-0.05, 0.05)),
                cy01=float(cy) + float(local_rng.uniform(-0.05, 0.05)),
                radius01=float(local_rng.uniform(0.028, 0.045)),
                edge01=float(local_rng.uniform(0.008, 0.015)),
            )
            distractor_rgb = distractor_palette[int(local_rng.integers(0, len(distractor_palette)))].reshape(3, 1, 1)
            distractor_gain = np.float32(local_rng.uniform(0.22, 0.42))
            scene = np.clip(scene + ((distractor_gain * mask)[None, :, :] * distractor_rgb), 0.0, 1.0)
        scene = np.clip(scene + (0.015 * local_rng.standard_normal(scene.shape, dtype=np.float32)), 0.0, 1.0)
        return scene.astype(np.float32, copy=False)

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
        elif key == "white noise":
            g = _spectral_noise(beta=0.0, distribution="gaussian")
        elif key == "uniform white noise":
            g = _spectral_noise(beta=0.0, distribution="uniform")
        elif key == "gaussian white noise":
            g = _spectral_noise(beta=0.0, distribution="gaussian")
        elif key == "pink noise":
            g = _spectral_noise(beta=1.0, distribution="gaussian")
        elif key == "brown noise":
            g = _spectral_noise(beta=2.0, distribution="gaussian")
        elif key == "red noise":
            g = _spectral_noise(beta=1.8, distribution="gaussian")
        elif key == "blue noise":
            g = _spectral_noise(beta=-1.0, distribution="gaussian")
        elif key == "violet noise":
            g = _spectral_noise(beta=-2.0, distribution="gaussian")
        elif key in ("gray noise", "grey noise"):
            g = _spectral_noise(beta=0.5, distribution="gaussian")
        elif key == "red":
            return _colored_dot_scene(key, sample_idx=int(sample_idx), target_rgb=(1.0, 0.18, 0.18))
        elif key == "green":
            return _colored_dot_scene(key, sample_idx=int(sample_idx), target_rgb=(0.18, 1.0, 0.24))
        elif key == "blue":
            return _colored_dot_scene(key, sample_idx=int(sample_idx), target_rgb=(0.18, 0.42, 1.0))
        elif key == "yellow":
            return _colored_dot_scene(key, sample_idx=int(sample_idx), target_rgb=(1.0, 0.96, 0.20))
        elif key == "cyan":
            return _colored_dot_scene(key, sample_idx=int(sample_idx), target_rgb=(0.18, 0.96, 1.0))
        elif key == "magenta":
            return _colored_dot_scene(key, sample_idx=int(sample_idx), target_rgb=(1.0, 0.24, 0.92))
        elif key == "brown":
            return _colored_dot_scene(key, sample_idx=int(sample_idx), target_rgb=(0.62, 0.42, 0.20))
        elif key in ("gray", "grey"):
            g = np.full((size, size), 0.50, dtype=np.float32)
        elif key == "pattern":
            g = np.clip(
                0.5
                + (0.25 * np.sin((2.0 * math.pi * ((4.0 * x01) + phase))))
                + (0.25 * np.sin((2.0 * math.pi * ((5.0 * y01) - (phase * 0.7))))),
                0.0,
                1.0,
            )
        elif key == "shape":
            diamond = (np.abs(x01 - 0.5) + np.abs(y01 - 0.5) <= 0.28).astype(np.float32, copy=False)
            g = np.clip((0.18 * _signal_pattern(phase=phase)) + (0.82 * diamond), 0.0, 1.0)
        elif key == "edge":
            ring = np.logical_and(np.abs(x01 - 0.5) + np.abs(y01 - 0.5) <= 0.32, np.abs(x01 - 0.5) + np.abs(y01 - 0.5) >= 0.25)
            g = np.clip((0.10 * _signal_pattern(phase=phase)) + (0.90 * ring.astype(np.float32, copy=False)), 0.0, 1.0)
        elif key == "texture":
            fine = np.sin((2.0 * math.pi * ((13.0 * x01) + (7.0 * y01))) + phase)
            coarse = np.sin((2.0 * math.pi * ((3.0 * x01) - (4.0 * y01))) - (phase * 0.6))
            g = np.clip(0.5 + (0.25 * fine) + (0.25 * coarse), 0.0, 1.0)
        elif key == "bright":
            g = np.clip(0.72 + (0.22 * _signal_pattern(phase=phase)), 0.0, 1.0)
        elif key == "dark":
            g = np.clip(0.04 + (0.22 * _signal_pattern(phase=phase)), 0.0, 1.0)
        elif key == "smooth":
            g = _box_blur(_signal_pattern(phase=phase), k=11)
        elif key == "rough":
            g = np.clip(0.5 + (0.30 * _signal_pattern(phase=phase)) + (0.26 * rng.standard_normal((size, size), dtype=np.float32)), 0.0, 1.0)
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
        if key == "pattern":
            cell = max(4, int(size // 10))
            checker = ((np.floor(xx / float(cell)) + np.floor(yy / float(cell))) % 2.0).astype(np.float32, copy=False)
            return np.clip((0.55 * checker) + (0.45 * _signal_pattern(phase=phase)), 0.0, 1.0)
        if key == "shape":
            circle = _circle_object()
            rim = np.clip(circle - _box_blur(circle, k=11), 0.0, 1.0)
            return np.clip((0.78 * circle) + (0.22 * rim), 0.0, 1.0)
        if key == "edge":
            circle = _circle_object()
            rim = np.clip(circle - _box_blur(circle, k=9), 0.0, 1.0)
            diag = (np.abs(x01 - y01) < 0.05).astype(np.float32, copy=False)
            return np.clip((0.72 * rim) + (0.28 * diag), 0.0, 1.0)
        if key == "texture":
            fine = 0.5 + (0.25 * np.sin((2.0 * math.pi * ((10.0 * x01) + (7.0 * y01))) + phase))
            fine += 0.25 * np.sin((2.0 * math.pi * ((5.0 * x01) - (11.0 * y01))) + (phase * 0.7))
            return np.clip(fine, 0.0, 1.0).astype(np.float32, copy=False)
        if key == "bright":
            return np.clip(0.80 + (0.18 * _signal_pattern(phase=phase)), 0.0, 1.0)
        if key == "dark":
            return np.clip(0.06 + (0.16 * _signal_pattern(phase=phase)), 0.0, 1.0)
        if key == "smooth":
            return _box_blur(_signal_pattern(phase=phase), k=11)
        if key == "rough":
            return np.clip((0.55 * _signal_pattern(phase=phase)) + (0.45 * rng.random((size, size), dtype=np.float32)), 0.0, 1.0)
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


# ---------------------------------------------------------------------------
# Pregestation mode configuration registry
# ---------------------------------------------------------------------------
# Each entry describes one training curriculum "stage" for the pre-gestation
# image generator.  Sub-round index 0 uses the first entry, sub-round 1 uses
# the second, etc.  Add new entries to extend the curriculum.
#
# Keys per mode config:
#   include_diagonal            bool   – generate corner-position samples when d_temp is high
#   diagonal_d_temp_threshold   float  – d_temp at which diagonal samples start appearing
#   diagonal_fraction_at_max    float  – fraction of per_combo samples that are diagonal at max d_temp
#   diagonal_max_d_temp         float  – d_temp value where diagonal_fraction reaches its maximum
#   include_depth               bool   – add cross-occluder + front/behind labeling
#   depth_occluder              str|None – "cross" or None
#   depth_labels                tuple  – (front_label, behind_label) strings


def _build_pregestation_logic_rows(
    image_size: int,
    seed: int,
    samples_per_combo: int,
    active_terms_lc: Optional[Sequence[str]] = None,
    circle_radius_temperature: float = 1.0,
    circle_displacement_temperature: float = 1.0,
    mode: str = "direction_color",
) -> Tuple[List[np.ndarray], List[np.ndarray], List[List[str]], Dict[str, Any]]:
    """Build pre-gestation training images according to *mode*.

    Modes (see PREGESTATION_MODE_CONFIGS):
      "direction_color"       – colored circles at cardinal positions; diagonal
                                placements with two direction labels appear when
                                circle_displacement_temperature is high.
      "direction_color_depth" – same curriculum plus a full-length cross occluder
                                that adds front / behind depth labeling.  The
                                object-presence mask is always the FULL disk
                                regardless of how much of the circle is visible.
    """
    size = max(16, int(image_size))
    per_combo = max(1, int(samples_per_combo))
    r_temp = float(max(0.1, float(circle_radius_temperature)))
    d_temp = float(max(0.1, float(circle_displacement_temperature)))
    mode = str(mode).strip().lower() or "direction_color"
    mode_cfg = PREGESTATION_MODE_CONFIGS.get(mode, PREGESTATION_MODE_CONFIGS["direction_color"])
    _include_depth = bool(mode_cfg.get("include_depth", False))
    _depth_labels: Tuple[str, ...] = tuple(str(lbl) for lbl in mode_cfg.get("depth_labels", ("front", "behind")))
    _include_diagonal = bool(mode_cfg.get("include_diagonal", True))
    _diag_thresh = float(mode_cfg.get("diagonal_d_temp_threshold", 1.05))
    _diag_max_temp = float(mode_cfg.get("diagonal_max_d_temp", 3.0))
    _diag_frac_max = float(mode_cfg.get("diagonal_fraction_at_max", 0.50))

    active = {
        re.sub(r"\s+", " ", str(x)).strip().lower()
        for x in (active_terms_lc or [])
        if str(x).strip()
    }
    all_colors = ["red", "green", "blue", "yellow", "cyan", "magenta", "brown", "white", "black", "gray"]
    all_dirs = ["up", "down", "left", "right"]
    colors = [c for c in all_colors if (not active or c in active)]
    directions = [d for d in all_dirs if (not active or d in active)]
    if len(colors) <= 0 or len(directions) <= 0:
        return [], [], [], {"enabled": False, "rows": 0, "reason": "missing_color_or_direction_terms"}

    rng = np.random.default_rng(int(seed))
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    x01 = (xx / float(max(1, size - 1))).astype(np.float32, copy=False)
    y01 = (yy / float(max(1, size - 1))).astype(np.float32, copy=False)

    # Anchor displacement from center (0.5, 0.5).  Temperature scales the offset so
    # d_temp=1.0 → default displacement; d_temp<1 → closer to center; d_temp>1 → farther.
    _base_offset = 0.32
    _d = float(_base_offset * d_temp)
    anchors = {
        "up":    (0.50, max(0.05, 0.50 - _d)),
        "down":  (0.50, min(0.95, 0.50 + _d)),
        "left":  (max(0.05, 0.50 - _d), 0.50),
        "right": (min(0.95, 0.50 + _d), 0.50),
    }

    # Diagonal anchors: equidistant from center as cardinal anchors so all
    # placements inhabit a single isotropic ring.
    _diag_offset = float(_d / float(np.sqrt(2.0)))
    dirs_set = set(directions)

    # Diagonal fraction: rises from 0 at d_temp=threshold to diag_frac_max at
    # d_temp=diagonal_max_d_temp.  At high temperatures the model must distinguish
    # circles that fall near image corners as belonging to TWO direction quadrants.
    _diag_frac = 0.0
    if _include_diagonal and d_temp > _diag_thresh:
        _diag_range = max(1e-6, _diag_max_temp - _diag_thresh)
        _diag_frac = min(_diag_frac_max, _diag_frac_max * (d_temp - _diag_thresh) / _diag_range)

    # Build the list of (direction_labels, cx_base, cy_base) for diagonal samples.
    diagonal_combos: List[Tuple[List[str], float, float]] = []
    if _diag_frac > 0.0:
        if "up" in dirs_set and "left" in dirs_set:
            diagonal_combos.append((["up", "left"],
                                    max(0.05, 0.5 - _diag_offset),
                                    max(0.05, 0.5 - _diag_offset)))
        if "up" in dirs_set and "right" in dirs_set:
            diagonal_combos.append((["up", "right"],
                                    min(0.95, 0.5 + _diag_offset),
                                    max(0.05, 0.5 - _diag_offset)))
        if "down" in dirs_set and "left" in dirs_set:
            diagonal_combos.append((["down", "left"],
                                    max(0.05, 0.5 - _diag_offset),
                                    min(0.95, 0.5 + _diag_offset)))
        if "down" in dirs_set and "right" in dirs_set:
            diagonal_combos.append((["down", "right"],
                                    min(0.95, 0.5 + _diag_offset),
                                    min(0.95, 0.5 + _diag_offset)))
    diag_per_combo = max(1, int(per_combo * _diag_frac)) if _diag_frac > 0.0 else 0

    palette = {
        "red": (1.00, 0.18, 0.18),
        "green": (0.18, 1.00, 0.22),
        "blue": (0.18, 0.40, 1.00),
        "yellow": (1.00, 0.96, 0.20),
        "cyan": (0.18, 0.96, 1.00),
        "magenta": (1.00, 0.24, 0.92),
        "brown": (0.62, 0.42, 0.20),
        "white": (0.96, 0.96, 0.96),
        "black": (0.10, 0.10, 0.10),
        "gray": (0.56, 0.56, 0.56),
    }

    def _soft_disk(cx: float, cy: float, radius: float, edge: float) -> np.ndarray:
        dist = np.sqrt(((x01 - np.float32(cx)) ** 2) + ((y01 - np.float32(cy)) ** 2)).astype(np.float32, copy=False)
        return np.clip((np.float32(radius) - dist) / np.float32(max(1e-4, edge)), 0.0, 1.0).astype(np.float32, copy=False)

    # -- Cross occluder (depth mode only) ------------------------------------
    # Full-length cross whose arms pass through all four cardinal anchor regions.
    # arm_half controls half-width of each bar in normalized image coordinates.
    _cross_arm_half = 0.055
    _cross_alpha = np.clip(
        (np.abs(y01 - 0.5) < _cross_arm_half).astype(np.float32)
        + (np.abs(x01 - 0.5) < _cross_arm_half).astype(np.float32),
        0.0, 1.0,
    )  # shape (H, W), in [0, 1]
    _cross_color_rgb = np.asarray([0.50, 0.50, 0.50], dtype=np.float32).reshape(3, 1, 1)

    out_images: List[np.ndarray] = []
    out_masks: List[np.ndarray] = []
    out_mask_stacks: List[np.ndarray] = []
    out_terms: List[List[str]] = []

    def _build_samples(dir_labels: List[str], cx0: float, cy0: float, n: int) -> None:
        """Generate n samples per color for the given anchor and direction labels."""
        for color in colors:
            rgb = np.asarray(palette[str(color)], dtype=np.float32).reshape(3, 1, 1)
            for _ in range(n):
                # Background with faint central guide lines.
                bg = np.full((3, size, size), 0.035, dtype=np.float32)
                guide_x = np.exp(-((x01 - 0.5) ** 2) / 0.0015).astype(np.float32, copy=False)
                guide_y = np.exp(-((y01 - 0.5) ** 2) / 0.0015).astype(np.float32, copy=False)
                guide = np.clip(0.025 * (guide_x + guide_y), 0.0, 0.05).astype(np.float32, copy=False)
                bg = np.clip(bg + guide[None, :, :], 0.0, 1.0)
                jitter_x = float(rng.uniform(-0.025, 0.025))
                jitter_y = float(rng.uniform(-0.025, 0.025))
                # r_temp scales the base radius range: temp=1 → [0.085, 0.115].
                _r_lo = float(0.085 * r_temp)
                _r_hi = float(0.115 * r_temp)
                radius = float(rng.uniform(_r_lo, _r_hi))
                edge = float(rng.uniform(0.014, 0.024))
                disk = _soft_disk(cx=float(cx0 + jitter_x), cy=float(cy0 + jitter_y),
                                  radius=radius, edge=edge)
                disk_3d = disk[None, :, :]

                if _include_depth and len(_depth_labels) >= 2:
                    # 50 / 50 front vs behind assignment per sample.
                    depth = _depth_labels[0] if rng.random() < 0.5 else _depth_labels[1]
                    cross_3d = _cross_alpha[None, :, :]
                    if depth == _depth_labels[0]:
                        # "front": composite cross onto bg, then circle on top.
                        layer = bg * (1.0 - cross_3d) + _cross_color_rgb * cross_3d
                        img = np.clip(layer * (1.0 - disk_3d) + rgb * disk_3d,
                                      0.0, 1.0).astype(np.float32, copy=False)
                    else:
                        # "behind": composite circle onto bg, then cross on top.
                        # The mask below still records the full disk — "the thing
                        # is HERE even though the cross is covering part of it."
                        layer = bg * (1.0 - disk_3d) + rgb * disk_3d
                        img = np.clip(layer * (1.0 - cross_3d) + _cross_color_rgb * cross_3d,
                                      0.0, 1.0).astype(np.float32, copy=False)
                    extra_terms: List[str] = [depth]
                else:
                    img = np.clip(bg + disk_3d * rgb, 0.0, 1.0).astype(np.float32, copy=False)
                    extra_terms = []

                # Object permanence policy: every element gets its FULL spatial
                # extent as its mask, regardless of compositing order.  All
                # per-element masks are stacked and stored in the cache so that
                # no ground-truth information is discarded.  The composite
                # normalization of the stack is the single mask passed to the
                # dataloader at training time.

                bg_terms = ["dark", "signal", "shape"]
                circle_terms = list(dir_labels) + [str(color), "object", "signal", "shape"] + extra_terms
                bg_mask = np.ones((size, size), dtype=np.float32)
                disk_mask = disk.astype(np.float32, copy=False)

                if _include_depth and len(_depth_labels) >= 2:
                    cross_terms = ["gray", "shape", "signal"]
                    cross_mask = _cross_alpha.astype(np.float32, copy=False)
                    elem_masks = [bg_mask, disk_mask, cross_mask]
                    merged = _normalize_vocab_terms(bg_terms + circle_terms + cross_terms)
                else:
                    elem_masks = [bg_mask, disk_mask]
                    merged = _normalize_vocab_terms(bg_terms + circle_terms)

                # Composite mask: additive-sum all per-element masks, normalized
                # to [0, 1].  This is the mask the dataloader sees at training time.
                elem_stack = np.stack(elem_masks, axis=0)  # [k, H, W]
                composite_mask = _composite_mask_stack(elem_stack)

                out_images.append(img)
                out_masks.append(composite_mask)
                out_mask_stacks.append(elem_stack)
                out_terms.append(merged)

    # Cardinal combos (always generated).
    for direction in directions:
        cx0, cy0 = anchors[str(direction)]
        _build_samples([str(direction)], cx0, cy0, per_combo)

    # Diagonal combos (generated when displacement temperature is high enough).
    for (dlabels, dcx, dcy) in diagonal_combos:
        _build_samples(dlabels, dcx, dcy, diag_per_combo)

    info = {
        "enabled": True,
        "rows": int(len(out_images)),
        "mode": str(mode),
        "colors": list(colors),
        "directions": list(directions),
        "samples_per_combo": int(per_combo),
        "combos": int(len(colors) * len(directions)),
        "diagonal_combos": int(len(diagonal_combos)),
        "diagonal_samples_per_combo": int(diag_per_combo),
        "diagonal_fraction": float(_diag_frac),
        "include_depth": bool(_include_depth),
        "circle_radius_temperature": float(r_temp),
        "circle_displacement_temperature": float(d_temp),
        "reason": f"formal_logic_{mode}",
    }
    return out_images, out_masks, out_mask_stacks, out_terms, info


# Recognized color term names whose score maps can be extracted by
# _semantic_color_score_maps.  "grey" is an alias for "gray" and both are
# included so either spelling in the term list triggers mask extraction.


def _enrich_pregestation_stack_with_observed_color_masks(
    image_chw01: np.ndarray,
    elem_stack: np.ndarray,
    term_row: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Extend a geometric mask stack with pixel-observed color masks.

    For each color term present in *term_row* that belongs to the known
    colorimetric vocabulary, extract the corresponding per-pixel color score
    map from *image_chw01* and append it to *elem_stack* as an additional
    mask slot.  The output ``extended_stack`` has shape ``[k + k', H, W]``
    where ``k'`` is the number of matched color terms.  The composite
    normalization is recomputed from the extended stack.

    This solves the denormalization problem cleanly: because the raw
    per-element geometric masks are available in *elem_stack* we never need
    to invert the normalized composite.  We simply append to the raw stack
    and let ``_composite_mask_stack`` renormalize the combined result.

    Returns ``(extended_stack, composite)`` — terms are not modified.
    """
    chw = np.asarray(image_chw01, dtype=np.float32)
    color_maps = _semantic_color_score_maps(chw)
    norm_terms = {re.sub(r"\s+", " ", str(t)).strip().lower() for t in term_row}

    observed: List[np.ndarray] = []
    for color_name in _PREGESTATION_OBSERVED_COLOR_TERMS:
        if str(color_name) not in norm_terms:
            continue
        cmap = color_maps.get(str(color_name))
        if cmap is None:
            continue
        score = np.asarray(cmap, dtype=np.float32)
        vmax = float(np.max(score))
        if vmax > 1e-8:
            observed.append(np.clip(score / float(vmax), 0.0, 1.0).astype(np.float32, copy=False))

    if len(observed) == 0:
        return elem_stack, _composite_mask_stack(elem_stack)

    observed_np = np.stack(observed, axis=0)  # [k', H, W]
    extended = np.concatenate([elem_stack, observed_np], axis=0)  # [k + k', H, W]
    return extended, _composite_mask_stack(extended)


from pipeline.nodes.vocab_node import _build_reference_flashcard_payload_rows  # migration shim


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
