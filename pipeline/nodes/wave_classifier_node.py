"""
Wave Classifier Node — Second TinyConvClassifier for wave feedback (Stage W).

This file is the authoritative description of everything idiosyncratic to the
wave-feedback classifier in this pipeline.

This model is distinct from the Berkeley classifier:
  * It trains on images rendered from transformer-accepted WAV chunks
  * Its labels come from WAV folder names or spectral clustering
    (not from semantic vocabulary)
  * Its output is used as a feedback signal to modulate the transformer's
    training weights and gate the wave stage
  * It optionally runs zero-shot evaluation against the label embedding bank
    to measure cross-domain generalisation

Model architecture choices owned here
--------------------------------------
  * Second TinyConvClassifier instance (smaller than the Berkeley one)
  * No LoRA, no mask decoder head
  * Stratified train/val split from transformer outputs

Loss composition owned here
----------------------------
  * Standard multi-label BCE (no semantic cosine loss)
  * Evaluated with macro-F1 and accuracy metrics

Feedback mechanism owned here
------------------------------
  * Wave entropy + feature score → combined feedback metric
  * Gate W passes when feedback metric exceeds threshold
  * Feedback score modulates transformer and generator loss weights
    via _feedback_scaled_weight (passed back into ctx)

Zero-shot evaluation owned here
---------------------------------
  * Encode query terms with the same sentence-transformer backend
  * Nearest-neighbour lookup in the Berkeley label embedding bank
  * Top-1 accuracy on wave-rendered images
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from pipeline.context import PipelineContext
from pipeline.graph import PipelineNode
from pipeline.nodes.base import GatedNode, make_grad_scaler
from pipeline.nodes.label_embedding_node import _normalize_l2_rows_np
from pipeline.nodes.vocab_node import _semantic_expand_inferred_tags, _semantic_noise_terms_from_spectrum_sample
from pipeline.wave_io import _decode_record_to_mono, _save_mono_wav
import numpy as np
from wav_ml_core import RenderConfig, WaveRecord, read_wav_record, render_mono_wave_to_tensor


# ---------------------------------------------------------------------------
# Node config
# ---------------------------------------------------------------------------

@dataclass
class WaveClassifierConfig:
    """Every hyperparameter specific to the wave-feedback TinyConvClassifier."""

    # ---- architecture ---------------------------------------------------
    base_ch: int = 48
    max_ch: int = 256
    context_blocks: int = 4
    context_dropout: float = 0.05

    # ---- label assignment -----------------------------------------------
    # "folder"   → labels from WAV folder names
    # "spectral" → labels from spectral angular centroid clustering
    # "hash"     → deterministic path-hash quantile bucketing
    label_mode: str = "folder"
    n_label_buckets: int = 8       # used when label_mode=="hash" or "spectral"

    # ---- dataset construction from transformer outputs ------------------
    max_accepted_chunks: int = 1024
    val_fraction: float = 0.15
    min_train_samples: int = 64

    # ---- optimiser / LR -------------------------------------------------
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    grad_accum_steps: int = 1

    # ---- AMP ------------------------------------------------------------
    amp: bool = False
    amp_dtype: str = "fp16"

    # ---- training schedule ----------------------------------------------
    epochs_per_round: int = 3
    batch_size: int = 32
    num_workers: int = 0

    # ---- zero-shot evaluation -------------------------------------------
    zero_shot_enabled: bool = True
    zero_shot_query_terms: str = ""    # semicolon-delimited; empty = use active vocab
    zero_shot_top_k: int = 5

    # ---- wave feedback / gate -------------------------------------------
    gate_entropy_min: float = 0.40
    gate_feature_score_min: float = 0.55
    gate_required_consecutive: int = 3

    # Feedback scaling: how strongly the wave gate score modulates
    # transformer and generator loss weights
    feedback_min_weight: float = 0.3
    feedback_max_weight: float = 1.0


# ---------------------------------------------------------------------------
# Build node
# ---------------------------------------------------------------------------

class BuildWaveClassifierNode(PipelineNode):
    """Instantiate the wave-feedback TinyConvClassifier.

    Deferred until after the first transformer round produces accepted chunks.
    The number of output classes is determined by the dataset, so this node
    runs each time the class count changes rather than strictly once.
    """

    node_id = "build_wave_classifier"
    description = "Instantiate wave-feedback TinyConvClassifier"

    def __init__(self, cfg: WaveClassifierConfig) -> None:
        self.cfg = cfg
        self._last_n_classes = -1

    def should_run(self, ctx: PipelineContext) -> bool:
        # Only relevant if wave stage is active
        mode = str(ctx.orchestration_mode or getattr(ctx.args, "orchestration_mode", "")).lower()
        return "w" in mode

    def execute(self, ctx: PipelineContext) -> None:
        from wav_ml_models import TinyConvClassifier, maybe_compile_module

        # Determine label count from WAV records
        n_classes = _count_wave_labels(ctx, self.cfg)
        if n_classes == self._last_n_classes and ctx.wave_classifier is not None:
            return  # no rebuild needed

        model = TinyConvClassifier(
            num_classes=n_classes,
            base_ch=self.cfg.base_ch,
            max_ch=self.cfg.max_ch,
            context_blocks=self.cfg.context_blocks,
            context_dropout=self.cfg.context_dropout,
        ).to(ctx.device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        ctx.wave_classifier = model
        ctx.wave_classifier_optimizer = optimizer
        if self.cfg.amp:
            ctx.wave_classifier_grad_scaler = make_grad_scaler(enabled=True)

        self._last_n_classes = n_classes
        _log(f"[wave-classifier] built: n_classes={n_classes}")


# ---------------------------------------------------------------------------
# Stage W — wave classifier training node
# ---------------------------------------------------------------------------

class WaveClassifierTrainNode(GatedNode):
    """Stage W: Build dataset from transformer outputs and train wave classifier.

    Requires Gate 0 + Gate 1 + Gate R (transformer).
    Flow:
      1. _build_wave_classifier_dataset_from_transformer — collect accepted chunks,
         render them to images, assign labels from WAV folder names or spectral analysis
      2. train_classifier — standard multi-label BCE training
      3. evaluate_classifier — accuracy + macro-F1 on held-out validation set
      4. (optional) _evaluate_zero_shot_queries_on_images — cross-domain zero-shot
      5. _wave_feedback_gate_pass — compute feedback metric and update gate W
    """

    node_id = "stage_w_wave_classifier"
    description = "Stage W: Wave classifier training on transformer outputs"
    required_gates = ["gate_pregestation", "gate_gestation", "gate_transformer"]

    def __init__(self, cfg: WaveClassifierConfig) -> None:
        self.cfg = cfg

    def should_run(self, ctx: PipelineContext) -> bool:
        if not super().should_run(ctx):
            return False
        mode = str(ctx.orchestration_mode or getattr(ctx.args, "orchestration_mode", "")).lower()
        if "w" not in mode:
            return False
        return ctx.transformer is not None and ctx.classifier is not None

    def execute(self, ctx: PipelineContext) -> None:
        from pipeline.nodes.gate_nodes import (
            _wave_feedback_gate_pass,
            _wave_feedback_snapshot,
        )
        from wav_ml_models import (
            train_classifier,
            evaluate_classifier,
            evaluate_feature_score_before_after,
        )

        # -- Build dataset ------------------------------------------------
        dataset_result = _build_wave_classifier_dataset_from_transformer(
            transformer=ctx.transformer,
            classifier=ctx.classifier,
            streams=ctx.float_streams,
            render_cfg=ctx.render_config,
            device=ctx.device,
            max_chunks=self.cfg.max_accepted_chunks,
            label_mode=self.cfg.label_mode,
            n_buckets=self.cfg.n_label_buckets,
            val_fraction=self.cfg.val_fraction,
            args=ctx.args,
        )

        if dataset_result is None or int(dataset_result.get("n_train", 0)) < self.cfg.min_train_samples:
            _log(f"[stageW] not enough accepted chunks to train wave classifier "
                 f"(need {self.cfg.min_train_samples})")
            return

        train_loader = dataset_result["train_loader"]
        val_loader = dataset_result.get("val_loader")
        n_train = int(dataset_result.get("n_train", 0))
        n_classes = int(dataset_result.get("n_classes", 1))

        # Rebuild classifier if class count changed
        if ctx.wave_classifier is None or _get_n_classes(ctx.wave_classifier) != n_classes:
            _rebuild_wave_classifier(ctx, self.cfg, n_classes)

        # -- Train --------------------------------------------------------
        train_classifier(
            classifier=ctx.wave_classifier,
            optimizer=ctx.wave_classifier_optimizer,
            loader=train_loader,
            device=ctx.device,
            epochs=self.cfg.epochs_per_round,
            amp_enabled=ctx.amp_enabled,
            amp_dtype=ctx.amp_dtype,
            grad_clip=self.cfg.grad_clip,
            grad_scaler=ctx.wave_classifier_grad_scaler,
            args=ctx.args,
        )

        # -- Evaluate -----------------------------------------------------
        eval_result: Dict[str, float] = {}
        if val_loader is not None:
            eval_result = evaluate_classifier(
                classifier=ctx.wave_classifier,
                loader=val_loader,
                device=ctx.device,
                args=ctx.args,
            )

        accuracy = float(eval_result.get("accuracy", 0.0))
        macro_f1 = float(eval_result.get("macro_f1", 0.0))

        # -- Zero-shot evaluation -----------------------------------------
        zs_top1 = 0.0
        if self.cfg.zero_shot_enabled and ctx.label_embedding_bank is not None:
            zs_top1 = _run_zero_shot_eval(ctx, self.cfg, val_loader)

        # -- Wave feedback gate -------------------------------------------
        feedback = _wave_feedback_snapshot(
            classifier=ctx.classifier,
            wave_classifier=ctx.wave_classifier,
            streams=ctx.float_streams,
            render_cfg=ctx.render_config,
            device=ctx.device,
            args=ctx.args,
        )
        feature_score = float(feedback.get("feature_score", 0.0))
        entropy = float(feedback.get("entropy", 0.0))
        combined = (feature_score + entropy) / 2.0

        ctx.gate_wave.required_consecutive = self.cfg.gate_required_consecutive
        ctx.gate_wave.record(
            round_id=ctx.round_id,
            metric=combined,
            threshold=(self.cfg.gate_feature_score_min + self.cfg.gate_entropy_min) / 2.0,
            above=True,
        )

        # Store feedback weight for use by transformer/generator nodes
        ctx.log_metric("stageW", "accuracy", accuracy)
        ctx.log_metric("stageW", "macro_f1", macro_f1)
        ctx.log_metric("stageW", "feature_score", feature_score)
        ctx.log_metric("stageW", "entropy", entropy)
        ctx.log_metric("stageW", "zs_top1", zs_top1)

        _log(f"[stageW] n_train={n_train} acc={accuracy:.3f} f1={macro_f1:.3f} "
             f"feat={feature_score:.3f} entropy={entropy:.3f} "
             f"zs={zs_top1:.3f} gate={'PASS' if ctx.gate_wave.passed else 'hold'}")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _count_wave_labels(ctx: PipelineContext, cfg: WaveClassifierConfig) -> int:
    records = list(getattr(ctx, "wav_records", []) or [])
    if len(records) <= 0:
        return max(2, int(cfg.n_label_buckets))
    if str(cfg.label_mode).strip().lower() in {"hash", "spectral"}:
        return max(2, int(cfg.n_label_buckets))
    data_root = _resolve_wave_data_root(ctx, records)
    labels, class_names, _ = _build_labels(
        records=records,
        data_root=data_root,
        label_mode=str(cfg.label_mode).strip().lower() or "folder",
        pseudo_classes=max(2, int(cfg.n_label_buckets)),
    )
    if int(len(class_names)) > 0:
        return max(2, int(len(class_names)))
    if labels is None:
        return 2
    return max(2, int(np.unique(np.asarray(labels, dtype=np.int64)).size))


def _rebuild_wave_classifier(
    ctx: PipelineContext,
    cfg: WaveClassifierConfig,
    n_classes: int,
) -> None:
    from wav_ml_models import TinyConvClassifier

    model = TinyConvClassifier(
        num_classes=n_classes,
        base_ch=cfg.base_ch,
        max_ch=cfg.max_ch,
        context_blocks=cfg.context_blocks,
        context_dropout=cfg.context_dropout,
    ).to(ctx.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    ctx.wave_classifier = model
    ctx.wave_classifier_optimizer = optimizer
    _log(f"[wave-classifier] rebuilt with n_classes={n_classes}")


def _get_n_classes(model) -> int:
    try:
        return int(model.head[-1].out_features)
    except Exception:
        return -1


def _run_zero_shot_eval(
    ctx: PipelineContext,
    cfg: WaveClassifierConfig,
    val_loader,
) -> float:
    if val_loader is None:
        return 0.0
    try:

        query_str = cfg.zero_shot_query_terms.strip() or ";".join(ctx.active_extra_terms[:16])
        result = _evaluate_zero_shot_queries_on_images(
            classifier=ctx.classifier,
            loader=val_loader,
            device=ctx.device,
            query_texts=query_str,
            label_bank=ctx.label_embedding_bank,
            class_names=ctx.class_names,
            top_k=cfg.zero_shot_top_k,
            args=ctx.args,
        )
        return float(result.get("top1", 0.0))
    except Exception as exc:
        _log(f"[stageW] zero-shot eval failed: {exc}")
        return 0.0


def _log(msg: str) -> None:
    print(msg, flush=True)


# =========================================================================
# Functions extracted from wav_config_transformer_pipeline.py
# =========================================================================


def _resolve_wave_data_root(ctx: PipelineContext, records: Sequence[WaveRecord]) -> str:
    wav_dir = str(getattr(ctx.args, "wav_root", "") or getattr(ctx.args, "wav_dir", "") or "").strip()
    if wav_dir:
        return wav_dir
    latent_dir = getattr(ctx, "latent_wav_pool_dir", None)
    if latent_dir is not None:
        return str(latent_dir)
    if len(records) > 0:
        try:
            first = Path(_record_path(records[0])).resolve()
            return str(first.parent.parent if first.parent != first.anchor else first.parent)
        except Exception:
            pass
    return "."


def _record_path(record: Any) -> str:
    if hasattr(record, "path"):
        return str(record.path)
    return str(record)


def _ensure_wave_record(record: Any) -> WaveRecord:
    if isinstance(record, WaveRecord):
        return record
    if hasattr(record, "path") and hasattr(record, "frames") and hasattr(record, "framerate"):
        return record  # duck-typed WaveRecord
    return read_wav_record(str(record))


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
    records: Sequence[Any],
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
        p = Path(_record_path(rec)).resolve()
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
    from wav_ml_models import TinyConvClassifier, _autocast_context, _resolve_amp_dtype

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
        with _autocast_context(device=device, use_amp=use_amp, amp_dtype_t=amp_dtype_t):
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
    from wav_ml_models import _autocast_context, _resolve_amp_dtype

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
            with _autocast_context(device=device, use_amp=amp_enabled, amp_dtype_t=amp_dtype_t):
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


def _build_labels(
    records: Sequence[Any],
    data_root: str,
    label_mode: str,
    pseudo_classes: int,
    single_class_fallback: str = "spectral_cosine_quantile",
):
    root = Path(data_root).resolve()
    if label_mode == "folder":
        names: List[str] = []
        for rec in records:
            p = Path(_record_path(rec)).resolve()
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
    for rec_any in records:
        rec = _ensure_wave_record(rec_any)
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
