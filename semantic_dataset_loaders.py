from __future__ import annotations

import gzip
import math
import pickle
import queue
import json
import hashlib
import os
import re
import shutil
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data._utils.collate import default_collate
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from pipeline.filesystem_emergency import raise_if_filesystem_space_emergency
from pipeline.nodes.interrupts import StageStopRequested
from pipeline.progress import interruptible_tqdm


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
_SEMANTIC_DISK_ROWS_CACHE_LOCK = threading.Lock()
_SEMANTIC_DISK_ROWS_CACHE: Dict[str, Tuple[List["SemanticDiskRow"], Dict[str, Any]]] = {}
_SEMANTIC_COLOR_TERMS: Tuple[str, ...] = (
    "bright",
    "dark",
    "warm",
    "cool",
    "red",
    "orange",
    "green",
    "blue",
    "yellow",
    "cyan",
    "magenta",
    "brown",
    "black",
    "white",
    "gray",
    "neutral",
    "edge",
)
_INGESTED_ITEM_MASK_TERMS = frozenset(("signal", "object"))
_DATASET_LABEL_SUFFIX = " dataset"


def _directory_size_bytes(path: Path) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    total = 0
    try:
        for child in root.rglob("*"):
            try:
                if child.is_file():
                    total += int(child.stat().st_size)
            except Exception:
                continue
    except Exception:
        return 0
    return int(total)


def _remove_tree(path: Path) -> None:
    root = Path(path)
    if not root.exists():
        return
    shutil.rmtree(str(root), ignore_errors=False)


def _reset_cache_dir(path: Path) -> None:
    root = Path(path)
    if root.exists():
        _remove_tree(root)
    root.mkdir(parents=True, exist_ok=True)


def _cache_control_file(cache_root: Path) -> Path:
    return Path(cache_root) / ".cache_control.json"


def _load_cache_control(cache_root: Path) -> Dict[str, Any]:
    path = _cache_control_file(cache_root)
    if not path.exists():
        return {"loop_cursor": 0}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {"loop_cursor": 0}


def _store_cache_control(cache_root: Path, payload: Dict[str, Any], progress_control: Any = None) -> None:
    root = Path(cache_root)
    try:
        root.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise_if_filesystem_space_emergency(
            progress_control,
            exc,
            note="semantic stage cache control write",
            write_path=root,
        )
        return
    path = _cache_control_file(root)
    try:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except Exception as exc:
        raise_if_filesystem_space_emergency(
            progress_control,
            exc,
            note="semantic stage cache control write",
            write_path=path,
        )
        pass


def _cache_dir_mtime_ns(path: Path) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    try:
        return int(root.stat().st_mtime_ns)
    except Exception:
        return 0


def _loop_slot_dir(cache_root: Path, slot_idx: int) -> Path:
    return Path(cache_root) / f"loop_slot_{int(slot_idx):04d}"


@lru_cache(maxsize=8192)
def _norm_txt(x: str) -> str:
    return re.sub(r"\s+", " ", str(x)).strip().lower()


def normalize_vocab_terms(terms: Sequence[str]) -> List[str]:
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


class DatasetTermRegistry:
    """Discovers unique terms from raw data in encounter order.

    Assigns local indices 0..N-1 with NO relationship to any vocabulary list.
    The index order is purely the order in which unique terms are first seen.
    """

    def __init__(self) -> None:
        self._term_to_idx: Dict[str, int] = {}
        self._idx_to_term: Dict[int, str] = {}

    def register(self, term: str) -> int:
        """Register a term.  Returns its local index (stable once assigned)."""
        key = _norm_txt(str(term))
        if not key:
            raise ValueError("DatasetTermRegistry: cannot register an empty term")
        idx = self._term_to_idx.get(key)
        if idx is not None:
            return idx
        idx = len(self._term_to_idx)
        self._term_to_idx[key] = idx
        self._idx_to_term[idx] = key
        return idx

    def register_many(self, terms: Sequence[str]) -> List[int]:
        """Register multiple terms.  Returns their local indices."""
        return [self.register(t) for t in terms if _norm_txt(str(t))]

    @property
    def term_to_idx(self) -> Dict[str, int]:
        return dict(self._term_to_idx)

    @property
    def idx_to_term(self) -> Dict[int, str]:
        return dict(self._idx_to_term)

    @property
    def local_vocab(self) -> List[str]:
        """All registered terms in encounter order."""
        return [self._idx_to_term[i] for i in range(len(self._idx_to_term))]

    def __len__(self) -> int:
        return len(self._term_to_idx)

    def __contains__(self, term: str) -> bool:
        return _norm_txt(str(term)) in self._term_to_idx

    def extra_terms_beyond_supervised(self, supervised_names: Sequence[str]) -> List[str]:
        """Return terms in this registry that are NOT in the supervised set.

        These are the terms the churn/LoRA system needs to find slots for.
        """
        supervised_set = {_norm_txt(str(s)) for s in supervised_names if _norm_txt(str(s))}
        return [t for t in self.local_vocab if t not in supervised_set]


def targets_from_terms(
    terms_rows: Sequence[Sequence[str]],
    term_to_idx: Dict[str, int],
    n_classes: int,
) -> List[np.ndarray]:
    """Canonical multi-hot target builder.  ALL loaders MUST use this.

    Parameters
    ----------
    terms_rows : per-image lists of semantic term strings.
    term_to_idx : mapping from normalised term string → class index.
    n_classes : width of the target vector.

    Returns
    -------
    List of float32 multi-hot vectors, one per row.
    """
    out: List[np.ndarray] = []
    nc = max(1, int(n_classes))
    dropped: List[str] = []
    for terms in terms_rows:
        y = np.zeros(nc, dtype=np.float32)
        for term in terms:
            key = re.sub(r"\s+", " ", str(term)).strip().lower()
            if not key:
                continue
            idx = term_to_idx.get(key, -1)
            if 0 <= idx < nc:
                y[idx] = 1.0
            else:
                dropped.append(key)
        out.append(y)
    if dropped:
        raise ValueError(
            f"targets_from_terms: {len(dropped)} term(s) not in active vocabulary "
            f"(silent label filtering is forbidden). "
            f"First dropped: {dropped[:20]}"
        )
    return out


def merge_terms_with_mask_indices(
    terms: Sequence[str],
    mask_indices: Any,
    registry: Optional["DatasetTermRegistry"] = None,
) -> List[str]:
    merged = list(normalize_vocab_terms([str(x) for x in list(terms or [])]))
    if registry is None:
        return merged
    idx_to_term = registry.idx_to_term
    extra_terms: List[str] = []
    idx_arr = np.asarray(mask_indices, dtype=np.int64).reshape(-1)
    for cls_idx in idx_arr.tolist():
        term = idx_to_term.get(int(cls_idx))
        if str(term or "").strip():
            extra_terms.append(str(term))
    if len(extra_terms) <= 0:
        return merged
    return list(normalize_vocab_terms(list(merged) + extra_terms))


def convert_sbd_mat_to_npz(root, progress_control: Any = None) -> None:
    """One-time conversion of {root}/cls/*.mat → {root}/cls/*.npz.

    After this, scipy.io.loadmat is no longer needed to read Berkeley SBD masks.
    Idempotent: skips any .mat file that already has a matching .npz.
    """
    from pathlib import Path as _Path
    root_path = _Path(str(root))
    cls_dirs = []
    for cls_dir in (
        root_path / "cls",
        root_path / "dataset" / "cls",
        root_path / "benchmark_RELEASE" / "dataset" / "cls",
    ):
        if cls_dir.exists() and cls_dir not in cls_dirs:
            cls_dirs.append(cls_dir)
    if not cls_dirs:
        return
    needs = []
    for cls_dir in cls_dirs:
        needs.extend(
            [
                f
                for f in sorted(cls_dir.glob("*.mat"))
                if not (cls_dir / f"{f.stem}.npz").exists()
            ]
        )
    if not needs:
        return
    try:
        import scipy.io as sio
        import numpy as _np
        print(f"[mat2npz] converting {len(needs)} Berkeley .mat masks → .npz (one-time) ...", flush=True)
        errors = 0
        for mat_path in needs:
            try:
                blob = sio.loadmat(str(mat_path), squeeze_me=False, struct_as_record=False)
                gtcls = blob.get("GTcls", None)
                seg = None
                try:
                    seg = _np.asarray(gtcls[0, 0].Segmentation, dtype=_np.uint8)
                except Exception:
                    try:
                        seg = _np.asarray(gtcls.Segmentation[0, 0], dtype=_np.uint8)
                    except Exception:
                        pass
                if seg is not None:
                    npz_path = mat_path.with_suffix(".npz")
                    try:
                        _np.savez_compressed(str(npz_path), segmentation=seg)
                    except Exception as exc:
                        raise_if_filesystem_space_emergency(
                            progress_control,
                            exc,
                            note="berkeley mat-to-npz conversion",
                            write_path=npz_path,
                        )
                        raise
            except StageStopRequested:
                raise
            except Exception:
                errors += 1
        print(f"[mat2npz] done. errors={errors}", flush=True)
    except ImportError:
        print("[mat2npz] scipy not available; mat→npz conversion skipped", flush=True)


def _read_sbd_split_file(root, split_name: str):
    """Read Berkeley SBD split file and return (image_paths, mask_paths).

    Does not import scipy or torchvision — just reads the .txt index file.
    Prefers train_noval.txt for the train split (excludes val overlap).
    Supports the flattened repo layout (root/{train,val}.txt + root/img + root/cls),
    the legacy root/dataset layout, and benchmark_RELEASE/dataset.
    Returns: (list[Path], list[str])
    """
    from pathlib import Path as _Path

    root = _Path(str(root))
    layout_candidates = [
        (root / "dataset", root),
        (root, root),
        (root / "benchmark_RELEASE" / "dataset", root / "benchmark_RELEASE" / "dataset"),
    ]
    seen_layout_keys = set()

    for split_dir, asset_root in layout_candidates:
        layout_key = (str(split_dir), str(asset_root))
        if layout_key in seen_layout_keys:
            continue
        seen_layout_keys.add(layout_key)

        split_candidates = []
        if split_name == "train":
            split_candidates.append(split_dir / "train_noval.txt")
        split_candidates.append(split_dir / f"{split_name}.txt")
        split_file = next((c for c in split_candidates if c.exists()), None)
        if split_file is None:
            continue

        img_dir = asset_root / "img"
        cls_dir = asset_root / "cls"
        if not img_dir.exists() and (split_dir / "img").exists():
            img_dir = split_dir / "img"
        if not cls_dir.exists() and (split_dir / "cls").exists():
            cls_dir = split_dir / "cls"
        if not img_dir.exists():
            continue

        lines = [
            ln.strip()
            for ln in split_file.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        image_paths = []
        mask_paths = []
        for name in lines:
            stem = str(name).strip()
            img_p = img_dir / f"{stem}.jpg"
            if not img_p.exists():
                continue
            mask_p = cls_dir / f"{stem}.mat"
            if not mask_p.exists():
                mask_npz = cls_dir / f"{stem}.npz"
                mask_p = mask_npz if mask_npz.exists() else mask_p
            image_paths.append(img_p)
            mask_paths.append(str(mask_p) if mask_p.exists() else "")
        if image_paths:
            return image_paths, mask_paths

    return [], []


@dataclass
class SemanticDiskRow:
    image_path: str
    terms: List[str]
    source: str
    mask_path: str = ""
    layout: Optional[Dict[str, Any]] = None
    mask_array: Optional[np.ndarray] = None
    mask_stack_array: Optional[np.ndarray] = None
    mask_stack_indices: Optional[np.ndarray] = None
    mask_cache_file: str = ""


def _clone_semantic_disk_row(row: SemanticDiskRow) -> SemanticDiskRow:
    return SemanticDiskRow(
        image_path=str(row.image_path),
        terms=list(row.terms),
        source=str(row.source),
        mask_path=str(row.mask_path or ""),
        layout=(dict(row.layout) if isinstance(row.layout, dict) else None),
        mask_array=(None if row.mask_array is None else np.asarray(row.mask_array, dtype=np.float32).copy()),
        mask_stack_array=(
            None if row.mask_stack_array is None else np.asarray(row.mask_stack_array, dtype=np.float32).copy()
        ),
        mask_stack_indices=(
            None if row.mask_stack_indices is None else np.asarray(row.mask_stack_indices, dtype=np.int64).copy()
        ),
        mask_cache_file=str(row.mask_cache_file or ""),
    )


def _clone_semantic_disk_rows(rows: Sequence[SemanticDiskRow]) -> List[SemanticDiskRow]:
    return [_clone_semantic_disk_row(row) for row in rows]


def _semantic_disk_rows_cache_key(data_root: str, class_names: Sequence[str], source_root: str = "") -> str:
    norm_classes = [re.sub(r"\s+", " ", str(name)).strip().lower() for name in class_names]
    payload = {
        "data_root": str(Path(str(data_root).strip() or "data/berkeley_sbd").resolve()),
        "source_root": str(Path(str(source_root).strip()).resolve()) if str(source_root).strip() else "",
        "class_names": norm_classes,
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _resolve_semantic_startup_threads(work_items: int, max_cap: int = 16) -> int:
    if int(work_items) <= 1:
        return 1
    cpu_count = max(1, int(os.cpu_count() or 1))
    return max(1, min(int(max_cap), int(cpu_count), int(work_items)))


def _ordered_thread_map(
    items: Sequence[Any],
    worker_fn: Any,
    max_workers: int,
    desc: str = "processing",
    progress_control: Any = None,
) -> List[Any]:
    count = int(len(items))
    if count <= 0:
        return []
    workers = max(1, min(int(max_workers), int(count)))
    if workers <= 1:
        return [
            worker_fn(item)
            for item in interruptible_tqdm(
                items,
                desc=desc,
                unit="row",
                leave=False,
                dynamic_ncols=True,
                control=progress_control,
            )
        ]
    results: List[Any] = [None] * count
    pool = ThreadPoolExecutor(max_workers=int(workers), thread_name_prefix="semantic-row-build")
    try:
        future_to_index = {pool.submit(worker_fn, item): int(i) for i, item in enumerate(items)}
        with interruptible_tqdm(
            total=count,
            desc=desc,
            unit="row",
            leave=False,
            dynamic_ncols=True,
            control=progress_control,
        ) as pbar:
            for future in as_completed(future_to_index):
                idx = int(future_to_index[future])
                results[idx] = future.result()
                pbar.update(1)
    except BaseException:
        # Cancel pending futures immediately so shutdown doesn't block on thousands
        # of queued items (e.g. when StageStopRequested is raised mid-collection).
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    return results


class _ThreadedPrefetchIterator:
    """Prefetch iterator that does NOT hold a strong reference to ``self``
    inside the worker thread.

    ``threading.Thread(target=self._run)`` stores a bound method, keeping
    ``self`` alive as long as the thread runs.  If the consumer replaces the
    iterator mid-epoch (e.g. when steps_per_epoch < len(loader)), the old
    iterator can never be garbage-collected, and its prefetched batches — which
    may contain large pinned-memory tensors — accumulate until the process
    runs out of RAM.

    Fix: pass the three shared objects (queue, stop-event, inner iterator) as
    plain arguments to a static worker function.  The thread holds no reference
    to ``self``, so CPython's reference-counting can free ``self`` as soon as
    the last external reference is dropped, triggering ``__del__`` → ``close()``
    which signals the worker to exit.
    """

    def __init__(self, loader: Any, max_prefetch_batches: int):
        self._loader_iter = iter(loader)
        self._queue: "queue.Queue[Tuple[str, Any]]" = queue.Queue(maxsize=max(2, int(max_prefetch_batches)))
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=_ThreadedPrefetchIterator._worker,
            args=(self._queue, self._stop_event, self._loader_iter),
            daemon=True,
        )
        self._thread.start()

    @staticmethod
    def _worker(q: "queue.Queue[Tuple[str, Any]]", stop: threading.Event, loader_iter: Any) -> None:
        try:
            for batch in loader_iter:
                while not stop.is_set():
                    try:
                        q.put(("batch", batch), timeout=0.05)
                        break
                    except queue.Full:
                        continue
                if stop.is_set():
                    break
            else:
                # Normal exhaustion — signal consumer.
                while not stop.is_set():
                    try:
                        q.put(("stop", None), timeout=0.05)
                        break
                    except queue.Full:
                        continue
        except BaseException as e:
            if not stop.is_set():
                try:
                    q.put(("error", e), timeout=1.0)
                except Exception:
                    pass

    def __iter__(self):
        return self

    def __next__(self):
        kind, payload = self._queue.get()
        if kind == "batch":
            return payload
        if kind == "error":
            raise payload
        raise StopIteration

    def close(self) -> None:
        """Signal the worker thread to exit and drain buffered batches."""
        self._stop_event.set()
        # Drain so a thread blocked on put() can unblock and exit cleanly.
        try:
            while True:
                self._queue.get_nowait()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()


class ThreadedPrefetchLoader:
    def __init__(self, loader: Any, max_prefetch_batches: int = 2):
        self._loader = loader
        self.max_prefetch_batches = max(2, int(max_prefetch_batches))
        self.prefetch_mode = "threaded"

    def __iter__(self):
        return _ThreadedPrefetchIterator(self._loader, max_prefetch_batches=int(self.max_prefetch_batches))

    def __len__(self):
        return len(self._loader)

    def __getattr__(self, name: str):
        return getattr(self._loader, name)


def maybe_wrap_loader_with_threaded_prefetch(
    loader: Any,
    requested_workers: int,
    effective_workers: int,
    device_type: str,
    prefetch_factor: int,
) -> Any:
    if loader is None:
        return None
    if int(requested_workers) <= 0:
        return loader
    if int(effective_workers) > 0:
        return loader
    if os.name != "nt":
        return loader
    if str(device_type).strip().lower() != "cuda":
        return loader
    return ThreadedPrefetchLoader(loader, max_prefetch_batches=max(2, int(prefetch_factor)))


def _normalize_mask_array(mask: Any, height: int, width: int) -> np.ndarray:
    arr = np.asarray(mask, dtype=np.float32)
    if int(arr.ndim) == 3:
        arr = np.mean(arr, axis=0).astype(np.float32, copy=False)
    if float(np.max(arr)) > 1.0:
        arr = arr / 255.0
    if int(arr.shape[0]) != int(height) or int(arr.shape[1]) != int(width):
        arr = torch.nn.functional.interpolate(
            torch.from_numpy(arr[None, None, ...]),
            size=(int(height), int(width)),
            mode='nearest',
        )[0, 0].numpy().astype(np.float32, copy=False)
    return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)



def _composite_mask_stack(stack: Any, processing_device: Optional[Any] = None) -> np.ndarray:
    """Label-density composite: sum of per-label masks, normalized by vmax.

    This is the ONE canonical mask compositing method.  Individual masks are
    assumed to be in [0, 1].  The sum is divided by its maximum so that the
    densest pixel reaches 1.0.  No clipping, no gating, no gamma.
    """
    arr = np.asarray(stack, dtype=np.float32)
    if int(arr.ndim) != 3 or int(arr.shape[0]) <= 0:
        return np.zeros((0, 0) if int(arr.ndim) < 2 else (int(arr.shape[-2]), int(arr.shape[-1])), dtype=np.float32)
    composite = np.sum(arr, axis=0).astype(np.float32, copy=False)
    vmax = float(np.max(composite))
    if vmax > 1e-8:
        composite = (composite / vmax).astype(np.float32, copy=False)
    else:
        composite = np.zeros_like(composite, dtype=np.float32)
    return composite


def _positive_label_indices(label_vec: Any) -> np.ndarray:
    return np.where(np.asarray(label_vec, dtype=np.float32).reshape(-1) >= 0.5)[0].astype(np.int64)


def _normalize_stack_row(
    mask: Any,
    *,
    height: int,
    width: int,
    processing_device: Optional[Any] = None,
) -> np.ndarray:
    return _normalize_mask_array(mask, height=int(height), width=int(width))


def _normalize_mask_array_batch(stack: np.ndarray, height: int, width: int) -> np.ndarray:
    """Batch version of _normalize_mask_array for [N, H, W] float32 arrays.

    Fast path when all masks are already the correct spatial size — pure numpy,
    no Python loop.  Falls back to per-element PIL resize only when dimensions
    differ (uncommon during training where a fixed image_size is used).
    """
    arr = np.asarray(stack, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None]
    n = int(arr.shape[0])
    if n == 0:
        return np.zeros((0, max(0, int(height)), max(0, int(width))), dtype=np.float32)
    cur_h, cur_w = int(arr.shape[-2]), int(arr.shape[-1])
    if cur_h != int(height) or cur_w != int(width):
        # Per-element PIL resize — only when truly needed
        resized = np.empty((n, int(height), int(width)), dtype=np.float32)
        for i in range(n):
            resized[i] = _normalize_mask_array(arr[i], height=int(height), width=int(width))
        return resized
    # Fast path: correct size, just clamp range
    out = arr / 255.0 if float(np.max(arr)) > 1.0 else arr
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def _normalize_stack_row_batch(
    stack: np.ndarray,
    *,
    height: int,
    width: int,
    processing_device: Optional[Any] = None,
    apply_attention_normalization: bool = False,
) -> np.ndarray:
    """Batch version of _normalize_stack_row for [N, H, W] arrays.

    Returns [N, H, W] — each slice is resize-scaled to [0, 1].  Attention
    normalization (vmean/gamma) is intentionally NOT applied per-slice; it
    belongs on the final composite, not on individual masks before summing.
    Pass apply_attention_normalization=True to re-enable the per-slice
    attention path (e.g. for visualization).
    """
    clamped = _normalize_mask_array_batch(stack, height=int(height), width=int(width))
    if not bool(apply_attention_normalization):
        return clamped
    return clamped


def term_mask_map_to_label_stack(
    term_mask_map: Optional[Dict[str, Any]],
    label_vec: Any,
    *,
    registry: Optional["DatasetTermRegistry"] = None,
    height: int = 0,
    width: int = 0,
    processing_device: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if not isinstance(term_mask_map, dict) or not term_mask_map:
        return np.zeros((0, int(max(0, height)), int(max(0, width))), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    raw_masks: List[np.ndarray] = []
    indices: List[int] = []
    for raw_term, raw_mask in term_mask_map.items():
        tk = _norm_txt(str(raw_term))
        if not tk:
            continue
        ci = registry.register(tk) if registry is not None else -1
        if ci < 0:
            continue
        arr = np.asarray(raw_mask, dtype=np.float32)
        if int(arr.ndim) != 2 or int(arr.size) <= 0:
            continue
        if int(height) <= 0 or int(width) <= 0:
            height = int(arr.shape[0])
            width = int(arr.shape[1])
        raw_masks.append(arr)
        indices.append(int(ci))
    if len(raw_masks) <= 0:
        return np.zeros((0, int(max(0, height)), int(max(0, width))), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    # Batch-normalise all collected masks at once
    stacked = np.stack(raw_masks, axis=0)  # [K, H, W]
    normed = _normalize_stack_row_batch(
        stacked,
        height=int(height),
        width=int(width),
        processing_device=processing_device,
    )  # [K, H, W]
    vmax_per = np.max(normed.reshape(len(raw_masks), -1), axis=1)
    keep = np.where(vmax_per > 1e-8)[0]
    if len(keep) == 0:
        return np.zeros((0, int(max(0, height)), int(max(0, width))), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return normed[keep].astype(np.float32, copy=False), np.array([indices[int(i)] for i in keep], dtype=np.int64)


def combine_label_mask_stacks(
    label_vec: Any,
    *parts: Tuple[Any, Any],
    height: int = 0,
    width: int = 0,
    fallback_creation_mask: Optional[Any] = None,
    processing_device: Optional[Any] = None,
    strict: bool = False,
    idx_to_term: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    positive_idx = _positive_label_indices(label_vec)
    if int(positive_idx.size) <= 0:
        return np.zeros((0, int(max(0, height)), int(max(0, width))), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    if (int(height) <= 0 or int(width) <= 0) and fallback_creation_mask is not None:
        ref = np.asarray(fallback_creation_mask, dtype=np.float32)
        if int(ref.ndim) >= 2:
            height = int(ref.shape[-2])
            width = int(ref.shape[-1])

    rows: List[np.ndarray] = []
    indices: List[int] = []
    covered: set[int] = set()

    for raw_stack, raw_indices in parts:
        if raw_stack is None or raw_indices is None:
            continue
        stack_arr = np.asarray(raw_stack, dtype=np.float32)
        idx_arr = np.asarray(raw_indices, dtype=np.int64).reshape(-1)
        if int(stack_arr.ndim) == 2:
            stack_arr = stack_arr[None, ...]
        if int(stack_arr.ndim) != 3 or int(stack_arr.shape[0]) <= 0 or int(idx_arr.size) <= 0:
            continue
        if int(height) <= 0 or int(width) <= 0:
            height = int(stack_arr.shape[-2])
            width = int(stack_arr.shape[-1])
        pair_count = min(int(stack_arr.shape[0]), int(idx_arr.size))
        # Batch-normalise all slices in this part at once, then filter
        batch_norm = _normalize_stack_row_batch(
            stack_arr[:int(pair_count)],
            height=int(height),
            width=int(width),
            processing_device=processing_device,
        )  # [pair_count, H, W]
        vmax_per = np.max(batch_norm.reshape(int(pair_count), -1), axis=1)  # [pair_count]
        neg = np.where(np.asarray(idx_arr[:int(pair_count)], dtype=np.int64) < 0)[0]
        if int(neg.size) > 0:
            raise ValueError(
                f"combine_label_mask_stacks: received negative indices at "
                f"positions {neg.tolist()} — upstream must assign every term an index"
            )
        valid_mask = vmax_per > 1e-8
        for si in np.where(valid_mask)[0]:
            cls_idx = int(idx_arr[int(si)])
            rows.append(batch_norm[int(si)])
            indices.append(cls_idx)
            covered.add(cls_idx)

    missing = [int(ci) for ci in positive_idx.tolist() if int(ci) not in covered]
    if missing:
        _h = int(max(1, height))
        _w = int(max(1, width))
        for ci in missing:
            word = str((idx_to_term or {}).get(int(ci), f"<unknown idx {ci}>"))
            if _is_always_true_term(word):
                # Always-true terms (dataset labels, "object", "signal")
                # intentionally receive a whole-image mask.
                rows.append(np.ones((_h, _w), dtype=np.float32))
                indices.append(ci)
            else:
                raise ValueError(
                    f"combine_label_mask_stacks: label idx={ci} ('{word}') is positive "
                    f"but has no spatial mask and is not an always-true category — "
                    f"every labelled term must have a mask"
                )
    if len(rows) <= 0:
        return np.zeros((0, int(max(0, height)), int(max(0, width))), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(rows, axis=0).astype(np.float32, copy=False), np.asarray(indices, dtype=np.int64)



def elem_stacks_to_label_stacks(
    elem_stack: np.ndarray,
    elem_term_lists: Sequence[Sequence[str]],
    label_vec: np.ndarray,
    term_to_idx: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert per-element spatial masks into per-label mask stacks.

    Each element mask is replicated for every known label index that
    appears in that element's term list. The result is a ``[K, H, W]``
    stack paired with a ``[K]`` int64 index array suitable for direct
    lookup by ``_expand_semantic_mask_supervision_batch``.
    """
    stack = np.asarray(elem_stack, dtype=np.float32)
    if int(stack.ndim) == 2:
        stack = stack[None, ...]
    out_masks: List[np.ndarray] = []
    out_idx: List[int] = []
    h = int(stack.shape[-2]) if int(stack.ndim) >= 2 else 0
    w = int(stack.shape[-1]) if int(stack.ndim) >= 2 else 0
    covered: set[int] = set()
    n_elems = min(int(stack.shape[0]), int(len(elem_term_lists)))
    if n_elems > 0:
        batch_norm = _normalize_stack_row_batch(stack[:n_elems], height=int(h), width=int(w))
        for ei in range(n_elems):
            mask_e = batch_norm[ei]
            for term in elem_term_lists[int(ei)]:
                tk = _norm_txt(str(term))
                if not tk:
                    continue
                ci = int(term_to_idx.get(tk, -1))
                if ci < 0:
                    raise ValueError(
                        f"elem_stacks_to_label_stacks: term {tk!r} has no index in the "
                        f"provided mapping — every term in the data must have an index"
                    )
                out_masks.append(mask_e)
                out_idx.append(ci)
                covered.add(int(ci))
    if len(out_masks) == 0:
        h, w = int(stack.shape[-2]), int(stack.shape[-1])
        return np.zeros((0, h, w), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(out_masks, axis=0).astype(np.float32, copy=False), np.asarray(out_idx, dtype=np.int64)


def _is_dataset_label_term(term: str) -> bool:
    return bool(_norm_txt(str(term)).endswith(_DATASET_LABEL_SUFFIX))


def _is_always_true_term(term: str) -> bool:
    """Return True for terms that are always true for every image in their dataset.

    These are dataset-level labels (e.g. "berkeley sbd dataset") and ingested
    item signals ("signal", "object").  They should receive whole-image masks
    on first sight rather than being silently dropped or patched downstream.
    """
    normed = _norm_txt(str(term))
    return bool(normed.endswith(_DATASET_LABEL_SUFFIX) or normed in _INGESTED_ITEM_MASK_TERMS)


def build_combined_mask_stacks(
    label_vec: Any,
    registry: "DatasetTermRegistry",
    height: int,
    width: int,
    *,
    elem_stack: Optional[Any] = None,
    elem_term_lists: Optional[Sequence[Sequence[str]]] = None,
    heuristic_stack: Optional[Any] = None,
    heuristic_idx: Optional[Any] = None,
    tonal_masks: Optional[Dict[str, Any]] = None,
    extra_parts: Optional[Sequence[Tuple[Any, Any]]] = None,
    processing_device: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Unified per-row mask-building pipeline.

    Combines explicit-element, heuristic, tonal, and any caller-supplied
    ``extra_parts`` mask sources into ``(mask_stack, mask_indices)``.  Each
    label receives only the spatial masks that directly correspond to it.
    Compositing is deferred to batch-preparation time on the training device.
    """
    y = np.asarray(label_vec, dtype=np.float32).reshape(-1)
    h = int(max(1, height))
    w = int(max(1, width))
    term_to_idx = registry.term_to_idx
    idx_to_term = registry.idx_to_term

    parts: List[Tuple[Any, Any]] = []
    has_elem = elem_stack is not None and elem_term_lists is not None

    # -- 1. Explicit element stacks (pregestation synthetic geometry) --
    if has_elem:
        explicit_stack, explicit_idx = elem_stacks_to_label_stacks(
            elem_stack=elem_stack,
            elem_term_lists=elem_term_lists,
            label_vec=y,
            term_to_idx=term_to_idx,
        )
        parts.append((explicit_stack, explicit_idx))

    # -- 4. Tonal masks --
    if tonal_masks:
        _tonal_slices: List[np.ndarray] = []
        _tonal_idxs: List[int] = []
        for _tterm, _tmask in tonal_masks.items():
            _tk = str(_tterm).strip().lower()
            if not _tk:
                continue
            _tidx = registry.register(_tk)
            _tmask_np = np.asarray(_tmask, dtype=np.float32)
            if int(_tmask_np.ndim) == 2 and int(_tmask_np.size) > 0:
                _tonal_slices.append(_tmask_np)
                _tonal_idxs.append(_tidx)
        if _tonal_slices:
            parts.append((
                np.stack(_tonal_slices, axis=0).astype(np.float32, copy=False),
                np.asarray(_tonal_idxs, dtype=np.int64),
            ))

    # -- 5. Heuristic stacks (from image analysis, pre-computed) --
    if heuristic_stack is not None and heuristic_idx is not None:
        parts.append((
            np.asarray(heuristic_stack, dtype=np.float32),
            np.asarray(heuristic_idx, dtype=np.int64),
        ))

    # -- 6. Extra caller-supplied (stack, idx) pairs --
    if extra_parts:
        for ep_stack, ep_idx in extra_parts:
            if ep_stack is not None and ep_idx is not None:
                parts.append((
                    np.asarray(ep_stack, dtype=np.float32),
                    np.asarray(ep_idx, dtype=np.int64),
                ))

    # -- 7. Combine --
    merged_stack, merged_idx = combine_label_mask_stacks(
        y,
        *parts,
        height=h,
        width=w,
        processing_device=processing_device,
        strict=True,
        idx_to_term=registry.idx_to_term,
    )

    return (
        np.asarray(merged_stack, dtype=np.float32),
        np.asarray(merged_idx, dtype=np.int64),
    )


def build_term_mask_stack_from_image(
    image: Any,
    label_vec: Any,
    registry: Optional["DatasetTermRegistry"] = None,
    term_mask_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    stacks, indices = build_term_mask_stacks_from_images(
        images=np.asarray(image, dtype=np.float32)[None, ...],
        label_vecs=np.asarray(label_vec, dtype=np.float32).reshape(1, -1),
        registry=registry,
        term_mask_overrides_batch=[term_mask_overrides] if isinstance(term_mask_overrides, dict) else None,
    )
    return stacks[0], indices[0]


def _single_label_whole_image_mask(label_vec: Any, *, height: int, width: int) -> Optional[np.ndarray]:
    positive_idx = _positive_label_indices(label_vec)
    if int(positive_idx.size) != 1 or int(height) <= 0 or int(width) <= 0:
        return None
    return np.ones((int(height), int(width)), dtype=np.float32)


def assemble_semantic_mask_layers(
    *,
    image: Any,
    label_vec: Any,
    registry: Optional["DatasetTermRegistry"] = None,
    original_parts: Optional[Sequence[Tuple[Any, Any]]] = None,
    deformation_term_masks: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    chw = _image_to_chw01(image)
    h = int(chw.shape[1])
    w = int(chw.shape[2])
    y = np.asarray(label_vec, dtype=np.float32).reshape(-1).copy()

    parts: List[Tuple[Any, Any]] = []

    detected_stack, detected_idx = build_term_mask_stack_from_image(
        image=chw,
        label_vec=y,
        registry=registry,
    )
    if int(detected_stack.shape[0]) > 0 and int(detected_idx.size) > 0:
        parts.append((detected_stack, detected_idx))

    final_stack, final_idx = combine_label_mask_stacks(
        y,
        *parts,
        height=int(h),
        width=int(w),
        fallback_creation_mask=None,
    ) if len(parts) > 0 else (
        np.zeros((0, int(h), int(w)), dtype=np.float32),
        np.zeros((0,), dtype=np.int64),
    )
    return y, np.asarray(final_stack, dtype=np.float32), np.asarray(final_idx, dtype=np.int64)


def semantic_mask_stack_collate(batch: Sequence[Any]) -> Any:
    if len(batch) <= 0:
        return default_collate(batch)
    first = batch[0]
    if not isinstance(first, (tuple, list)) or int(len(first)) < 5:
        return default_collate(batch)
    xs: List[torch.Tensor] = []
    ms: List[torch.Tensor] = []
    mask_stacks: List[torch.Tensor] = []
    mask_indices: List[torch.Tensor] = []
    terms_rows: List[List[str]] = []
    for sample in batch:
        if not isinstance(sample, (tuple, list)) or int(len(sample)) < 5:
            raise RuntimeError("semantic_mask_stack_collate requires 5-tuple samples: (x, mask, stack, idx, terms).")
        xs.append(sample[0])
        ms.append(sample[1])
        stack_t = sample[2] if torch.is_tensor(sample[2]) else torch.as_tensor(sample[2])
        idx_t = sample[3] if torch.is_tensor(sample[3]) else torch.as_tensor(sample[3], dtype=torch.long)
        mask_stacks.append(stack_t)
        mask_indices.append(idx_t.to(dtype=torch.long))
        if int(len(sample)) >= 5 and isinstance(sample[4], (list, tuple)):
            terms_rows.append(normalize_vocab_terms([str(x) for x in list(sample[4])]))
        else:
            terms_rows.append([])
    return {
        "x": torch.stack(xs, dim=0),
        "mask": torch.stack(ms, dim=0),
        "mask_stacks": mask_stacks,
        "mask_indices": mask_indices,
        "terms_rows": terms_rows,
    }


def _image_to_chw01(image: Any) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if int(arr.ndim) == 3 and int(arr.shape[0]) in (1, 3, 4):
        chw = np.asarray(arr[:3, :, :], dtype=np.float32)
        if int(chw.shape[0]) == 1:
            chw = np.repeat(chw, 3, axis=0)
    elif int(arr.ndim) == 3 and int(arr.shape[2]) in (1, 3, 4):
        hwc = np.asarray(arr[:, :, :3], dtype=np.float32)
        if int(hwc.shape[2]) == 1:
            hwc = np.repeat(hwc, 3, axis=2)
        chw = np.transpose(hwc, (2, 0, 1)).astype(np.float32, copy=False)
    elif int(arr.ndim) == 2:
        gray = np.asarray(arr, dtype=np.float32)
        chw = np.repeat(gray[None, :, :], 3, axis=0).astype(np.float32, copy=False)
    else:
        raise RuntimeError(f"Unsupported image shape for semantic mask inference: {tuple(arr.shape)}")
    vmax = float(np.max(chw)) if int(chw.size) > 0 else 0.0
    vmin = float(np.min(chw)) if int(chw.size) > 0 else 0.0
    if vmax > 1.0:
        chw = chw / 255.0
    elif vmin < 0.0 and vmax <= 1.0:
        chw = (chw + 1.0) * 0.5
    return np.clip(chw, 0.0, 1.0).astype(np.float32, copy=False)


def _image_batch_to_bchw01(images: Any) -> np.ndarray:
    arr = np.asarray(images, dtype=np.float32)
    if int(arr.ndim) == 2:
        return _image_to_chw01(arr)[None, ...]
    if int(arr.ndim) == 3:
        if int(arr.shape[0]) in (1, 3, 4) or int(arr.shape[2]) in (1, 3, 4):
            return _image_to_chw01(arr)[None, ...]
        bgray = np.repeat(arr[:, None, :, :], 3, axis=1).astype(np.float32, copy=False)
        vmax = np.max(bgray, axis=(1, 2, 3), keepdims=True) if int(bgray.size) > 0 else np.zeros((int(arr.shape[0]), 1, 1, 1), dtype=np.float32)
        vmin = np.min(bgray, axis=(1, 2, 3), keepdims=True) if int(bgray.size) > 0 else np.zeros((int(arr.shape[0]), 1, 1, 1), dtype=np.float32)
        if float(np.max(vmax)) > 1.0:
            bgray = bgray / 255.0
        elif float(np.min(vmin)) < 0.0 and float(np.max(vmax)) <= 1.0:
            bgray = (bgray + 1.0) * 0.5
        return np.clip(bgray, 0.0, 1.0).astype(np.float32, copy=False)
    if int(arr.ndim) == 4 and int(arr.shape[1]) in (1, 3, 4):
        bchw = np.asarray(arr[:, :3, :, :], dtype=np.float32)
        if int(bchw.shape[1]) == 1:
            bchw = np.repeat(bchw, 3, axis=1)
    elif int(arr.ndim) == 4 and int(arr.shape[3]) in (1, 3, 4):
        bhwc = np.asarray(arr[:, :, :, :3], dtype=np.float32)
        if int(bhwc.shape[3]) == 1:
            bhwc = np.repeat(bhwc, 3, axis=3)
        bchw = np.transpose(bhwc, (0, 3, 1, 2)).astype(np.float32, copy=False)
    else:
        raise RuntimeError(f"Unsupported image batch shape for semantic mask inference: {tuple(arr.shape)}")
    vmax = np.max(bchw, axis=(1, 2, 3), keepdims=True) if int(bchw.size) > 0 else np.zeros((int(bchw.shape[0]), 1, 1, 1), dtype=np.float32)
    vmin = np.min(bchw, axis=(1, 2, 3), keepdims=True) if int(bchw.size) > 0 else np.zeros((int(bchw.shape[0]), 1, 1, 1), dtype=np.float32)
    if float(np.max(vmax)) > 1.0:
        bchw = bchw / 255.0
    elif float(np.min(vmin)) < 0.0 and float(np.max(vmax)) <= 1.0:
        bchw = (bchw + 1.0) * 0.5
    return np.clip(bchw, 0.0, 1.0).astype(np.float32, copy=False)

def build_observed_color_stack_from_terms(
    image: Any,
    term_row: Sequence[str] = (),
    processing_device: Optional[Any] = None,
) -> Tuple[np.ndarray, List[str]]:
    bchw = _image_batch_to_bchw01(np.asarray(image, dtype=np.float32))
    h = int(bchw.shape[2]) if int(bchw.ndim) == 4 else 0
    w = int(bchw.shape[3]) if int(bchw.ndim) == 4 else 0
    resolved = torch.device("cpu")
    if processing_device is not None:
        try:
            resolved = processing_device if isinstance(processing_device, torch.device) else torch.device(str(processing_device))
            if resolved.type == "cuda" and not torch.cuda.is_available():
                resolved = torch.device("cpu")
        except Exception:
            resolved = torch.device("cpu")
    arr = torch.as_tensor(bchw, dtype=torch.float32, device=resolved)
    arr = torch.clamp(arr, 0.0, 1.0)
    if int(arr.ndim) != 4 or int(arr.shape[1]) < 3:
        return np.zeros((0, h, w), dtype=np.float32), []

    r = arr[:, 0]
    g = arr[:, 1]
    b = arr[:, 2]
    vmax = torch.maximum(torch.maximum(r, g), b)
    vmin = torch.minimum(torch.minimum(r, g), b)
    sat = torch.clamp(vmax - vmin, 0.0, 1.0)
    luma = torch.mean(arr[:, :3], dim=1)

    def _norm01_batch(x: torch.Tensor) -> torch.Tensor:
        hi = torch.amax(x, dim=(1, 2), keepdim=True)
        valid = hi > 1e-8
        safe_hi = torch.where(valid, hi, torch.ones_like(hi))
        return torch.where(valid, torch.clamp(x / safe_hi, 0.0, 1.0), torch.zeros_like(x))

    # --- HSV decomposition for hue-angle-based chromatic binning ---
    # chroma (= HSV value * HSV saturation = vmax - vmin) already computed as `sat`
    safe_delta = torch.where(sat > 1e-8, sat, torch.ones_like(sat))
    r_is_max = (r >= g) & (r >= b)
    g_is_max = (g > r) & (g >= b)
    # hue in [0, 6) segments matching standard HSV
    h_r = (g - b) / safe_delta          # segment when r is max: [-1, 1)
    h_g = (b - r) / safe_delta + 2.0    # segment when g is max: [1, 3)
    h_b = (r - g) / safe_delta + 4.0    # segment when b is max: [3, 5)
    hue_norm = torch.where(r_is_max, h_r, torch.where(g_is_max, h_g, h_b))
    hue_norm = hue_norm % 6.0
    hue_deg = hue_norm * 60.0           # [0, 360) degrees
    hue_deg = torch.where(sat > 1e-8, hue_deg, torch.zeros_like(hue_deg))
    # HSV saturation (normalised by value) — gates achromatic pixels out of hue scoring
    safe_vmax = torch.where(vmax > 1e-8, vmax, torch.ones_like(vmax))
    hsv_sat = torch.where(vmax > 1e-8, sat / safe_vmax, torch.zeros_like(vmax))

    _PI = torch.tensor(math.pi, dtype=torch.float32, device=resolved)

    def _hue_bell(center_deg: float, half_width_deg: float) -> torch.Tensor:
        """Cosine bell centred on `center_deg`, zero at ±half_width_deg. Handles 0°/360° wrap."""
        diff = torch.abs(hue_deg - center_deg)
        diff = torch.minimum(diff, 360.0 - diff)
        t = torch.clamp(diff / half_width_deg, 0.0, 1.0)
        return 0.5 * (1.0 + torch.cos(t * _PI))

    def _chroma_mask(center_deg: float, half_width_deg: float, sat_lo: float = 0.15) -> torch.Tensor:
        """Hue bell weighted by a smooth HSV-saturation gate and pixel value (brightness)."""
        gate = torch.clamp((hsv_sat - sat_lo) / 0.15, 0.0, 1.0)
        return _hue_bell(center_deg, half_width_deg) * gate * vmax

    color_maps: Dict[str, torch.Tensor] = {
        # --- Tonal (hue-independent) ---
        "bright":  _norm01_batch(torch.clamp(luma, 0.0, 1.0)),
        "dark":    _norm01_batch(torch.clamp(1.0 - luma, 0.0, 1.0)),
        "warm":    _norm01_batch(torch.clamp((r + 0.5 * g) - b, 0.0, 1.0) * torch.clamp(sat, 0.0, 1.0)),
        "cool":    _norm01_batch(torch.clamp((b + 0.5 * g) - r, 0.0, 1.0) * torch.clamp(sat, 0.0, 1.0)),
        # --- Chromatic: hue-angle binning via cosine bell on [0°, 360°) ---
        # Each color is credited only when the pixel's actual HSV hue falls within
        # that color's angular window, so a pixel that is orange (hue ≈ 30°) will
        # not bleed into the red bin (centred at 0°).
        "red":     _norm01_batch(_chroma_mask(  0.0, 22.0)),   # 338°–22°  (wraps)
        "orange":  _norm01_batch(_chroma_mask( 30.0, 17.0)),   # 13°–47°
        "yellow":  _norm01_batch(_chroma_mask( 60.0, 22.0)),   # 38°–82°
        "green":   _norm01_batch(_chroma_mask(120.0, 50.0)),   # 70°–170°
        "cyan":    _norm01_batch(_chroma_mask(180.0, 25.0)),   # 155°–205°
        "blue":    _norm01_batch(_chroma_mask(240.0, 42.0)),   # 198°–282°
        "magenta": _norm01_batch(_chroma_mask(300.0, 33.0)),   # 267°–333°
        # Brown: orange-amber hue, dark, moderately saturated
        "brown":   _norm01_batch(
            _hue_bell(25.0, 22.0)
            * torch.clamp((hsv_sat - 0.25) / 0.15, 0.0, 1.0)
            * torch.clamp((vmax - 0.15) / 0.10, 0.0, 1.0)
            * torch.clamp((0.62 - vmax) / 0.15, 0.0, 1.0)
        ),
        # --- Achromatic: based on chroma / value, not hue ---
        "black":   _norm01_batch(torch.clamp(0.22 - vmax, 0.0, 1.0)),
        "white":   _norm01_batch(torch.clamp(vmin - 0.78, 0.0, 1.0) * torch.clamp(0.20 - sat, 0.0, 1.0)),
        "gray":    _norm01_batch(torch.clamp(0.18 - sat, 0.0, 1.0) * torch.clamp(1.0 - torch.abs(vmax - 0.5) * 2.2, 0.0, 1.0)),
        "neutral": _norm01_batch(torch.clamp(0.25 - sat, 0.0, 1.0) * torch.clamp(vmax - 0.05, 0.0, 1.0)),
    }
    gx = torch.zeros_like(luma)
    gy = torch.zeros_like(luma)
    gx[:, :, 1:-1] = luma[:, :, 2:] - luma[:, :, :-2]
    gy[:, 1:-1, :] = luma[:, 2:, :] - luma[:, :-2, :]
    color_maps["edge"] = _norm01_batch(torch.sqrt((gx * gx) + (gy * gy)))

    observed_masks: List[np.ndarray] = []
    observed_terms: List[str] = []
    for color_name in _SEMANTIC_COLOR_TERMS:
        score_t = color_maps.get(str(color_name))
        if score_t is None or int(score_t.ndim) != 3 or int(score_t.shape[0]) <= 0:
            continue
        score = np.asarray(score_t[0].detach().cpu().numpy(), dtype=np.float32)
        vmax = float(np.max(score))
        if vmax <= 1e-8:
            continue
        observed_masks.append(np.clip(score / vmax, 0.0, 1.0).astype(np.float32, copy=False))
        observed_terms.append(str(color_name))

    if len(observed_masks) <= 0:
        return np.zeros((0, h, w), dtype=np.float32), []
    return np.stack(observed_masks, axis=0).astype(np.float32, copy=False), list(observed_terms)


def _normalize_semantic_terms_batch(
    terms_batch: Optional[Sequence[Optional[Sequence[str]]]],
    batch_size: int,
) -> List[Sequence[str]]:
    if terms_batch is None:
        return [[] for _ in range(int(max(0, batch_size)))]
    term_rows = list(terms_batch)
    if int(len(term_rows)) < int(batch_size):
        term_rows.extend([[] for _ in range(int(batch_size) - int(len(term_rows)))])
    elif int(len(term_rows)) > int(batch_size):
        term_rows = term_rows[: int(batch_size)]
    return term_rows


def build_term_mask_stacks_from_images(
    images: Any,
    label_vecs: Any,
    term_mask_overrides_batch: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    processing_device: Optional[Any] = None,
    registry: Optional[DatasetTermRegistry] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    bchw = _image_batch_to_bchw01(images)
    bsz = int(bchw.shape[0]) if int(bchw.ndim) == 4 else 0
    h = int(bchw.shape[2]) if int(bchw.ndim) == 4 else 0
    w = int(bchw.shape[3]) if int(bchw.ndim) == 4 else 0
    y = np.asarray(label_vecs, dtype=np.float32)
    if int(y.ndim) == 1:
        y = y.reshape(1, -1)
    if int(bsz) <= 0 or int(y.shape[0]) != int(bsz):
        return [np.zeros((0, int(h), int(w)), dtype=np.float32) for _ in range(int(max(0, bsz)))], [np.zeros((0,), dtype=np.int64) for _ in range(int(max(0, bsz)))]

    override_rows = list(term_mask_overrides_batch) if term_mask_overrides_batch is not None else [None for _ in range(int(bsz))]
    if int(len(override_rows)) < int(bsz):
        override_rows.extend([None for _ in range(int(bsz) - int(len(override_rows)))])

    out_stack: List[np.ndarray] = []
    out_idx: List[np.ndarray] = []
    for row_idx in range(int(bsz)):
        observed_stack, observed_terms = build_observed_color_stack_from_terms(
            image=bchw[int(row_idx)],
            processing_device=processing_device,
        ) if registry is not None else (
            np.zeros((0, int(h), int(w)), dtype=np.float32),
            [],
        )

        row_masks: List[np.ndarray] = []
        row_indices: List[int] = []
        seen_idx: set[int] = set()

        observed_count = min(int(observed_stack.shape[0]), int(len(observed_terms)))
        for pos in range(int(observed_count)):
            _obs_tk = _norm_txt(str(observed_terms[int(pos)]))
            if not _obs_tk:
                continue
            cls_idx = registry.register(_obs_tk)
            row_masks.append(np.asarray(observed_stack[int(pos)], dtype=np.float32))
            row_indices.append(int(cls_idx))
            seen_idx.add(int(cls_idx))

        override_row = override_rows[int(row_idx)]
        if isinstance(override_row, dict) and override_row:
            override_stack, override_idx = term_mask_map_to_label_stack(
                override_row,
                y[int(row_idx)],
                registry=registry,
                height=int(h),
                width=int(w),
                processing_device=processing_device,
            )
            override_count = min(int(override_stack.shape[0]), int(override_idx.size))
            for pos in range(int(override_count)):
                cls_idx = int(override_idx[int(pos)])
                if int(cls_idx) in seen_idx:
                    continue
                row_masks.append(np.asarray(override_stack[int(pos)], dtype=np.float32))
                row_indices.append(int(cls_idx))
                seen_idx.add(int(cls_idx))

        # -- Graduate always-true terms to whole-image masks on first sight --
        # Terms like "berkeley sbd dataset", "object", "signal" are true for
        # every image in their dataset.  They have no spatial localisation so
        # they get a full-image mask intentionally, not as a lazy fallback.
        if int(h) > 0 and int(w) > 0 and registry is not None:
            _idx_to_term_row = registry.idx_to_term
            y_row = y[int(row_idx)]
            for ci in range(int(y_row.shape[0])):
                if float(y_row[int(ci)]) < 0.5:
                    continue
                if int(ci) in seen_idx:
                    continue
                term_name = _idx_to_term_row.get(int(ci), "")
                if not _is_always_true_term(term_name):
                    continue
                row_masks.append(np.ones((int(h), int(w)), dtype=np.float32))
                row_indices.append(int(ci))
                seen_idx.add(int(ci))

        if len(row_masks) <= 0:
            out_stack.append(np.zeros((0, int(h), int(w)), dtype=np.float32))
            out_idx.append(np.zeros((0,), dtype=np.int64))
            continue
        out_stack.append(np.stack(row_masks, axis=0).astype(np.float32, copy=False))
        out_idx.append(np.asarray(row_indices, dtype=np.int64))
    return out_stack, out_idx


@dataclass
class StageDatasetManifest:
    name: str
    dataset: Dataset
    batch_size: int
    seed: int
    num_workers: int
    device_type: str
    max_samples: int = 0
    ordered_indices: Optional[Sequence[int]] = None
    persistent_workers: bool = False
    prefetch_factor: int = 2
    pin_memory: bool = False
    shuffle: bool = False
    sampler: Optional[Sampler] = None


def effective_dataloader_num_workers(num_workers: int, device_type: str = "") -> int:
    workers = max(0, int(num_workers))
    if workers <= 0:
        return 0
    if os.name == "nt" and str(device_type).strip().lower() == "cuda":
        # Avoid Win32 shared-mapping allocation failures (error 1455) when
        # worker processes collate large semantic tensor batches on Windows.
        return 0
    return workers


def dataloader_perf_kwargs(num_workers: int, persistent_workers: bool, prefetch_factor: int) -> Dict[str, Any]:
    if int(num_workers) <= 0:
        return {}
    out = {"persistent_workers": bool(persistent_workers)}
    if int(prefetch_factor) > 0:
        out["prefetch_factor"] = int(prefetch_factor)
    return out


def build_loader_from_manifest(manifest: StageDatasetManifest) -> Tuple[Optional[DataLoader], int]:
    def _dataset_uses_semantic_mask_stack_collate(ds: Dataset) -> bool:
        seen_ids = set()
        cur = ds
        while cur is not None and id(cur) not in seen_ids:
            seen_ids.add(id(cur))
            if bool(getattr(cur, "use_semantic_mask_stack_collate", False)):
                return True
            cur = getattr(cur, "dataset", None)
        return False

    n = int(len(manifest.dataset))
    if n <= 0:
        return None, 0
    if manifest.ordered_indices is None:
        picks = np.arange(int(n), dtype=np.int64)
        rng = np.random.default_rng(int(manifest.seed))
        rng.shuffle(picks)
        if int(manifest.max_samples) > 0:
            picks = picks[: int(min(int(manifest.max_samples), int(picks.shape[0])))]
    else:
        picks = np.asarray([int(i) for i in manifest.ordered_indices if 0 <= int(i) < int(n)], dtype=np.int64)
        if int(manifest.max_samples) > 0 and int(picks.size) > int(manifest.max_samples):
            picks = picks[: int(manifest.max_samples)]
    if int(picks.size) <= 0:
        return None, 0
    if manifest.sampler is not None and int(picks.size) != int(n):
        raise RuntimeError(
            "StageDatasetManifest sampler requires a full-dataset view; "
            "subset selection via ordered_indices/max_samples is not compatible."
        )
    ds_use: Dataset = manifest.dataset if int(picks.size) == int(n) else torch.utils.data.Subset(manifest.dataset, picks.tolist())
    collate_fn = semantic_mask_stack_collate if _dataset_uses_semantic_mask_stack_collate(ds_use) else None
    loader_num_workers = effective_dataloader_num_workers(
        num_workers=manifest.num_workers,
        device_type=manifest.device_type,
    )
    loader_generator = None
    use_shuffle = bool(manifest.shuffle) and manifest.sampler is None
    if bool(use_shuffle):
        loader_generator = torch.Generator()
        loader_generator.manual_seed(max(0, int(manifest.seed)))
    loader = DataLoader(
        ds_use,
        batch_size=max(1, int(manifest.batch_size)),
        shuffle=bool(use_shuffle),
        sampler=manifest.sampler,
        num_workers=int(loader_num_workers),
        pin_memory=bool(manifest.pin_memory),
        drop_last=False,
        collate_fn=collate_fn,
        generator=loader_generator,
        **dataloader_perf_kwargs(
            num_workers=int(loader_num_workers),
            persistent_workers=bool(manifest.persistent_workers),
            prefetch_factor=int(manifest.prefetch_factor),
        ),
    )
    loader = maybe_wrap_loader_with_threaded_prefetch(
        loader=loader,
        requested_workers=int(manifest.num_workers),
        effective_workers=int(loader_num_workers),
        device_type=str(manifest.device_type),
        prefetch_factor=int(manifest.prefetch_factor),
    )
    return loader, int(picks.size)


@lru_cache(maxsize=32)
def _unit_interval_axis(length: int) -> np.ndarray:
    n = max(1, int(length))
    if n <= 1:
        return np.zeros((1,), dtype=np.float32)
    return np.linspace(0.0, 1.0, num=int(n), dtype=np.float32)




def _apply_degrade(
    x: np.ndarray,
    *,
    seed: int,
    idx: int = 0,
    mask: Optional[np.ndarray] = None,
    degrade_config: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, np.ndarray]]:
    """Canonical image degradation pipeline — ONE place for all degrade logic.

    Applies probabilistic degradations to a CHW float32 image in [0, 1].

    Returns (degraded_image, mask_out, term_masks):
      - degraded_image: CHW float32 in [0, 1]
      - mask_out: spatial mask with spatial transforms propagated (None if not given)
      - term_masks: Dict[str, ndarray] of per-term raw accumulated pixel-delta masks.

    Masks in term_masks are RAW accumulated deltas — NO internal normalization.
    Callers normalize via vmax division before compositing.

    Term labeling is honest to what each operation does — "signal" is never added by degrade:
      Spatial transforms label only their own name (shift / horizontal flip / vertical flip).
      Damage labels only the damage type (blur damage, stride skew damage, etc.).
      Noise labels: noise damage, noise, mixed noise and signal (source always has signal).
      Color tinting labels the color; also bright if white, dark if black.
      Edge operations label the operation name and edge.
    The source image's signal labels come from its own construction, not from degrade.
    Excluded by design: texture synthesis, any internal normalization.
    """
    import os as _os
    cfg_raw = dict(degrade_config) if isinstance(degrade_config, dict) else {}
    cfg = {
        # Spatial transforms — applied independently; not always desired
        "shift_prob":               float(cfg_raw.get("shift_prob", 0.35)),
        "hflip_prob":               float(cfg_raw.get("hflip_prob", 0.40)),
        "vflip_prob":               float(cfg_raw.get("vflip_prob", 0.15)),
        # Damage — rarer, each type distinct
        "blur_prob":                float(cfg_raw.get("blur_prob", 0.35)),
        "stride_skew_prob":         float(cfg_raw.get("stride_skew_prob", 0.20)),
        "dropout_prob":             float(cfg_raw.get("dropout_prob", 0.20)),
        "quantization_prob":        float(cfg_raw.get("quantization_prob", 0.22)),
        "noise_prob":               float(cfg_raw.get("noise_prob", 0.45)),
        "noise_std_min":            float(cfg_raw.get("noise_std_min", 0.01)),
        "noise_std_max":            float(cfg_raw.get("noise_std_max", 0.08)),
        # Semantic modifications
        "color_prob":               float(cfg_raw.get("color_prob", 0.32)),
        "edge_highlight_prob":      float(cfg_raw.get("edge_highlight_prob", 0.28)),
        "edge_highlight_blend_min": float(cfg_raw.get("edge_highlight_blend_min", 0.05)),
        "edge_highlight_blend_max": float(cfg_raw.get("edge_highlight_blend_max", 0.25)),
        "edge_highlight_ultra":     bool(cfg_raw.get("edge_highlight_ultra",
                                         str(_os.environ.get("EDGE_HIGHLIGHT_ULTRA", "0")).strip() == "1")),
        "edge_blur_prob":           float(cfg_raw.get("edge_blur_prob", 0.22)),
        "edge_blur_kernel":         int(cfg_raw.get("edge_blur_kernel", 7)),
        "edge_blur_spread":         int(cfg_raw.get("edge_blur_spread", 5)),
    }
    rng = np.random.default_rng(int(seed) + (int(idx) * 104729))
    arr = np.asarray(x, dtype=np.float32)
    if int(arr.ndim) == 3 and int(arr.shape[0]) == 3:
        out = np.asarray(arr, dtype=np.float32, order="C")
    elif int(arr.ndim) == 3 and int(arr.shape[2]) == 3:
        out = np.transpose(arr, (2, 0, 1)).astype(np.float32, copy=False)
    elif int(arr.ndim) == 2:
        out = np.repeat(arr[None, :, :], 3, axis=0).astype(np.float32, copy=False)
    else:
        out = np.asarray(arr, dtype=np.float32, order="C")
    out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)
    c, h, w = int(out.shape[0]), int(out.shape[1]), int(out.shape[2])
    mask_out = None if mask is None else np.asarray(mask, dtype=np.float32).copy()
    term_masks: Dict[str, np.ndarray] = {}

    def _accum(terms: Sequence[str], delta: np.ndarray) -> None:
        d = np.asarray(delta, dtype=np.float32)
        if int(d.ndim) == 3:
            d = np.mean(d, axis=0)
        d = np.maximum(d, 0.0)
        if float(np.max(d)) <= 1e-8:
            return
        for t in normalize_vocab_terms([str(s) for s in terms]):
            tk = _norm_txt(t)
            prev = term_masks.get(tk)
            if prev is None:
                term_masks[tk] = d.copy()
            else:
                term_masks[tk] = np.asarray(prev, dtype=np.float32) + d

    # Shift (roll)
    if float(rng.random()) < float(cfg["shift_prob"]):
        shift_x = int(rng.integers(-max(1, w // 14), max(1, w // 14) + 1))
        shift_y = int(rng.integers(-max(1, h // 14), max(1, h // 14) + 1))
        if shift_x != 0 or shift_y != 0:
            prev = out.copy()
            if shift_x != 0:
                out = np.roll(out, shift=shift_x, axis=2)
                if mask_out is not None:
                    mask_out = np.roll(mask_out, shift=shift_x, axis=1)
            if shift_y != 0:
                out = np.roll(out, shift=shift_y, axis=1)
                if mask_out is not None:
                    mask_out = np.roll(mask_out, shift=shift_y, axis=0)
            _accum(["shift"], np.abs(np.mean(out, axis=0) - np.mean(prev, axis=0)))

    # Horizontal flip
    if float(rng.random()) < float(cfg["hflip_prob"]):
        prev = out.copy()
        out = np.flip(out, axis=2).copy()
        if mask_out is not None:
            mask_out = np.flip(mask_out, axis=1).copy()
        _accum(["horizontal flip"], np.abs(np.mean(out, axis=0) - np.mean(prev, axis=0)))

    # Vertical flip
    if float(rng.random()) < float(cfg["vflip_prob"]):
        prev = out.copy()
        out = np.flip(out, axis=1).copy()
        if mask_out is not None:
            mask_out = np.flip(mask_out, axis=0).copy()
        _accum(["vertical flip"], np.abs(np.mean(out, axis=0) - np.mean(prev, axis=0)))

    # Blur damage
    if float(rng.random()) < float(cfg["blur_prob"]):
        prev = out.copy()
        k = int(rng.choice(np.asarray([3, 5, 7], dtype=np.int32)))
        t = torch.from_numpy(out[None, ...])
        t = F.avg_pool2d(t, kernel_size=int(k), stride=1, padding=int(k // 2))
        out = np.asarray(t[0].cpu().numpy(), dtype=np.float32)
        _accum(["blur damage"], np.abs(np.mean(out, axis=0) - np.mean(prev, axis=0)))

    # Stride-skew damage
    if float(rng.random()) < float(cfg["stride_skew_prob"]):
        odd_shift = int(rng.integers(1, 8))
        out[:, 1::2, :] = np.roll(out[:, 1::2, :], shift=odd_shift, axis=2)
        if mask_out is not None:
            mask_out[1::2, :] = np.roll(mask_out[1::2, :], shift=odd_shift, axis=1)
        stripe = np.zeros((h, w), dtype=np.float32)
        stripe[1::2, :] = 1.0
        _accum(["stride skew damage"], stripe)

    # Dropout damage
    if float(rng.random()) < float(cfg["dropout_prob"]):
        keep = float(rng.uniform(0.78, 0.96))
        keep_mask = (rng.random((h, w), dtype=np.float32) < keep).astype(np.float32, copy=False)
        out = out * keep_mask[None, :, :]
        _accum(["dropout damage"], 1.0 - keep_mask)

    # Quantization damage
    if float(rng.random()) < float(cfg["quantization_prob"]):
        prev = out.copy()
        lv = int(rng.choice(np.asarray([4, 6, 8, 12], dtype=np.int32)))
        out = np.round(out * float(lv - 1)) / float(max(1, lv - 1))
        _accum(["quantization damage"], np.abs(np.mean(out, axis=0) - np.mean(prev, axis=0)))

    # Noise — auto-includes "noise" + "mixed noise and signal" since noise on a signal is always mixed
    if float(rng.random()) < float(cfg["noise_prob"]):
        prev = out.copy()
        std = float(rng.uniform(float(cfg["noise_std_min"]), float(cfg["noise_std_max"])))
        noise_arr = (std * rng.standard_normal((c, h, w), dtype=np.float32)).astype(np.float32, copy=False)
        out = np.clip(out + noise_arr, 0.0, 1.0)
        _accum(["noise damage", "noise", "mixed noise and signal"],
               np.abs(np.mean(out, axis=0) - np.mean(prev, axis=0)))

    # Color tinting
    if int(c) >= 3 and float(rng.random()) < float(cfg["color_prob"]):
        prev = out.copy()
        _COLOR_PROFILES = {
            "red":     np.asarray([1.00, 0.18, 0.18], dtype=np.float32),
            "green":   np.asarray([0.18, 1.00, 0.18], dtype=np.float32),
            "blue":    np.asarray([0.18, 0.18, 1.00], dtype=np.float32),
            "yellow":  np.asarray([1.00, 1.00, 0.20], dtype=np.float32),
            "cyan":    np.asarray([0.18, 1.00, 1.00], dtype=np.float32),
            "magenta": np.asarray([1.00, 0.18, 1.00], dtype=np.float32),
            "brown":   np.asarray([0.72, 0.44, 0.22], dtype=np.float32),
            "white":   np.asarray([1.00, 1.00, 1.00], dtype=np.float32),
            "black":   np.asarray([0.08, 0.08, 0.08], dtype=np.float32),
            "gray":    np.asarray([0.55, 0.55, 0.55], dtype=np.float32),
        }
        color_key = str(rng.choice(np.asarray(list(_COLOR_PROFILES.keys()), dtype=object))).strip().lower()
        profile = _COLOR_PROFILES.get(color_key, np.asarray([1.0, 1.0, 1.0], dtype=np.float32))[:, None, None]
        luma = np.mean(out[:3], axis=0, keepdims=True)
        tinted = np.clip(profile * np.clip(0.25 + (0.90 * luma), 0.0, 1.0), 0.0, 1.0).astype(np.float32, copy=False)
        alpha = float(rng.uniform(0.24, 0.58))
        out = np.clip(((1.0 - alpha) * out) + (alpha * tinted), 0.0, 1.0)
        tint_terms = [color_key]
        if color_key == "white":
            tint_terms.append("bright")
        elif color_key == "black":
            tint_terms.append("dark")
        _accum(tint_terms, np.abs(np.mean(out, axis=0) - np.mean(prev, axis=0)))

    # Edge highlight
    if float(rng.random()) < float(cfg["edge_highlight_prob"]):
        from pipeline.semantic_wheel_cache import _canny_edge_map, _sobel_edge_map
        prev = out.copy()
        gray = np.mean(out[:3], axis=0).astype(np.float32, copy=False)
        edge_map = _canny_edge_map(gray) if bool(cfg["edge_highlight_ultra"]) else _sobel_edge_map(gray)
        blend = float(rng.uniform(float(cfg["edge_highlight_blend_min"]), float(cfg["edge_highlight_blend_max"])))
        out = np.clip(out + (float(blend) * edge_map[None, :, :]), 0.0, 1.0)
        _accum(["edge highlight", "edge"], edge_map)

    # Edge blur
    if float(rng.random()) < float(cfg["edge_blur_prob"]):
        from pipeline.semantic_wheel_cache import _sobel_edge_map
        gray = np.mean(out[:3], axis=0).astype(np.float32, copy=False)
        edge_mask = _sobel_edge_map(gray)
        blur_k = int(cfg["edge_blur_kernel"])
        spread_k = int(cfg["edge_blur_spread"])
        out_t = torch.from_numpy(out[None, ...]).to(torch.float32)
        blurred = F.avg_pool2d(out_t, kernel_size=blur_k, stride=1, padding=blur_k // 2)[0].numpy().astype(np.float32)
        em_t = torch.from_numpy(edge_mask[None, None, ...]).to(torch.float32)
        spread_mask = F.avg_pool2d(em_t, kernel_size=spread_k, stride=1, padding=spread_k // 2)[0, 0].numpy().astype(np.float32)
        spread_mask = spread_mask / max(float(np.max(spread_mask)), 1e-8)
        out = np.clip(out * (1.0 - spread_mask[None]) + blurred * spread_mask[None], 0.0, 1.0).astype(np.float32, copy=False)
        _accum(["edge blur", "edge"], spread_mask)

    out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)
    return out, mask_out, term_masks


def augment_bootstrap_chw01(
    img: np.ndarray,
    seed: int,
    return_terms: bool = False,
    return_touch_mask: bool = False,
    return_term_masks: bool = False,
) -> Any:
    """Bootstrap image augmentation — delegates to canonical _apply_degrade."""
    out, _, raw_term_masks = _apply_degrade(np.asarray(img, dtype=np.float32), seed=int(seed))
    if not bool(return_terms) and not bool(return_touch_mask) and not bool(return_term_masks):
        return out
    terms = list(raw_term_masks.keys())
    if not bool(return_touch_mask) and not bool(return_term_masks):
        return out, terms
    all_vals = [np.asarray(v, dtype=np.float32) for v in raw_term_masks.values() if float(np.max(v)) > 1e-8]
    if len(all_vals) > 0:
        touch_sum = np.sum(np.stack(all_vals, axis=0), axis=0).astype(np.float32, copy=False)
        vmax = float(np.max(touch_sum))
        touched = np.clip(touch_sum / vmax, 0.0, 1.0).astype(np.float32, copy=False) if vmax > 1e-8 else touch_sum
    else:
        touched = np.zeros((int(out.shape[1]), int(out.shape[2])), dtype=np.float32)
    if bool(return_terms) and bool(return_touch_mask) and bool(return_term_masks):
        return out, terms, touched, raw_term_masks
    if bool(return_terms) and bool(return_touch_mask):
        return out, terms, touched
    if bool(return_terms):
        return out, terms
    if bool(return_touch_mask):
        return out, touched
    return out



class DiskSemanticRowsDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[SemanticDiskRow],
        image_size: int,
        return_masks: bool = False,
        return_mask_stack: bool = False,
        degrade: bool = False,
        degrade_seed: int = 0,
        degrade_config: Optional[Dict[str, Any]] = None,
        class_names: Optional[Sequence[str]] = None,
    ):
        self.rows = [
            SemanticDiskRow(
                image_path=str(r.image_path),
                terms=list(normalize_vocab_terms(r.terms)),
                source=str(r.source),
                mask_path=str(r.mask_path or ""),
                layout=dict(r.layout) if isinstance(r.layout, dict) else None,
                mask_array=(None if r.mask_array is None else np.asarray(r.mask_array, dtype=np.float32)),
                mask_stack_array=(None if r.mask_stack_array is None else np.asarray(r.mask_stack_array, dtype=np.float32)),
                mask_stack_indices=(None if r.mask_stack_indices is None else np.asarray(r.mask_stack_indices, dtype=np.int64)),
                mask_cache_file=str(getattr(r, "mask_cache_file", "") or ""),
            )
            for r in rows
        ]
        self.image_size = max(8, int(image_size))
        self.return_masks = bool(return_masks)
        self.return_mask_stack = bool(return_mask_stack)
        self.use_semantic_mask_stack_collate = bool(self.return_mask_stack)
        self.degrade = bool(degrade)
        self.degrade_seed = int(degrade_seed)
        self.class_names = [str(name) for name in list(class_names or []) if str(name).strip()]
        self.registry = DatasetTermRegistry()
        self.registry.register_many(self.class_names)
        cfg = dict(degrade_config) if isinstance(degrade_config, dict) else {}
        self.degrade_config = {
            "blur_prob": float(cfg.get("blur_prob", 0.55)),
            "stride_skew_prob": float(cfg.get("stride_skew_prob", 0.40)),
            "dropout_prob": float(cfg.get("dropout_prob", 0.25)),
            "quantization_prob": float(cfg.get("quantization_prob", 0.30)),
            "noise_prob": float(cfg.get("noise_prob", 0.55)),
            "noise_std_min": float(cfg.get("noise_std_min", 0.01)),
            "noise_std_max": float(cfg.get("noise_std_max", 0.08)),
            "edge_highlight_prob": float(cfg.get("edge_highlight_prob", 0.35)),
            "edge_highlight_blend_min": float(cfg.get("edge_highlight_blend_min", 0.05)),
            "edge_highlight_blend_max": float(cfg.get("edge_highlight_blend_max", 0.25)),
            "edge_highlight_ultra": bool(cfg.get("edge_highlight_ultra", str(os.environ.get("EDGE_HIGHLIGHT_ULTRA", "0")).strip() == "1")),
            "edge_blur_prob": float(cfg.get("edge_blur_prob", 0.30)),
            "edge_blur_kernel": int(cfg.get("edge_blur_kernel", 7)),
            "edge_blur_spread": int(cfg.get("edge_blur_spread", 5)),
        }

    def __len__(self) -> int:
        return int(len(self.rows))

    def _load_rgb01(self, path: str) -> np.ndarray:
        with Image.open(str(path)) as im:
            rgb = im.convert("RGB")
            if int(rgb.size[0]) != int(self.image_size) or int(rgb.size[1]) != int(self.image_size):
                rgb = rgb.resize((int(self.image_size), int(self.image_size)), resample=Image.BILINEAR)
            arr = np.asarray(rgb, dtype=np.float32)
        arr = np.clip(arr / 255.0, 0.0, 1.0).astype(np.float32, copy=False)
        return np.transpose(arr, (2, 0, 1)).astype(np.float32, copy=False)

    def _load_creation_mask01(self, row: SemanticDiskRow, h: int, w: int) -> Optional[np.ndarray]:
        if row.mask_array is not None:
            arr = np.asarray(row.mask_array, dtype=np.float32)
            if int(arr.ndim) == 3:
                arr = np.mean(arr, axis=0).astype(np.float32, copy=False)
            return _normalize_mask_array(arr, height=int(h), width=int(w))
        if str(row.mask_path).strip():
            mask_path = Path(str(row.mask_path))
            if mask_path.exists():
                if str(mask_path.suffix).strip().lower() in (".mat", ".npz"):
                    _npz_path = mask_path.with_suffix(".npz")
                    _mat_path = mask_path.with_suffix(".mat")
                    try:
                        if _npz_path.exists():
                            _d = np.load(str(_npz_path), allow_pickle=False)
                            seg = np.asarray(_d["segmentation"], dtype=np.float32)
                        elif _mat_path.exists():
                            try:
                                from scipy.io import loadmat
                            except Exception as e:
                                raise RuntimeError(
                                    f"scipy is required to read Berkeley SBD mask files: {_mat_path} ({type(e).__name__}: {e})"
                                ) from e
                            blob = loadmat(str(_mat_path), squeeze_me=False, struct_as_record=False)
                            seg = None
                            gtcls = blob.get("GTcls", None)
                            try:
                                seg = np.asarray(gtcls[0, 0].Segmentation, dtype=np.float32)
                            except Exception:
                                try:
                                    seg = np.asarray(gtcls.Segmentation[0, 0], dtype=np.float32)
                                except Exception:
                                    seg = None
                            if seg is None:
                                raise RuntimeError(f"Could not extract Berkeley segmentation from mask file: {_mat_path}")
                        else:
                            seg = None
                    except RuntimeError:
                        raise
                    except Exception:
                        seg = None
                    if seg is not None:
                        arr = (np.asarray(seg, dtype=np.float32) > 0.0).astype(np.float32, copy=False)
                        return _normalize_mask_array(arr, height=int(h), width=int(w))
                    # fall through to next branch if seg is None
                with Image.open(str(mask_path)) as im:
                    gray = im.convert("L")
                    if int(gray.size[0]) != int(w) or int(gray.size[1]) != int(h):
                        gray = gray.resize((int(w), int(h)), resample=Image.NEAREST)
                    arr = np.asarray(gray, dtype=np.float32) / 255.0
                return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)
        if isinstance(row.layout, dict):
            return np.clip(
                np.asarray(build_layout_mask(row.layout, height=int(h), width=int(w)), dtype=np.float32),
                0.0, 1.0,
            ).astype(np.float32, copy=False)
        return None

    def _load_mask01(self, row: SemanticDiskRow, h: int, w: int, image_rgb01: Optional[np.ndarray] = None) -> np.ndarray:
        creation_mask = self._load_creation_mask01(row, h=int(h), w=int(w))
        if creation_mask is not None:
            return np.asarray(creation_mask, dtype=np.float32)
        single_mask = _single_label_whole_image_mask(
            targets_from_terms([row.terms], self.term_to_idx, len(self.class_names))[0],
            height=int(h), width=int(w),
        )
        if single_mask is not None:
            return np.asarray(single_mask, dtype=np.float32)
        return np.zeros((int(h), int(w)), dtype=np.float32)

    def _apply_degrade(self, x: np.ndarray, mask: Optional[np.ndarray], idx: int) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, np.ndarray]]:
        if not bool(self.degrade):
            return x, mask, {}
        return _apply_degrade(
            x,
            seed=int(self.degrade_seed),
            idx=int(idx),
            mask=mask,
            degrade_config=self.degrade_config,
        )

    def __getitem__(self, index: int):
        row = self.rows[int(index)]
        x = self._load_rgb01(row.image_path)
        y = targets_from_terms([row.terms], self.term_to_idx, len(self.class_names))[0]
        if bool(self.return_masks):
            base_mask_np = self._load_mask01(row, h=int(x.shape[1]), w=int(x.shape[2]), image_rgb01=x)
            cached_bundle = _load_row_mask_cache_bundle(getattr(row, "mask_cache_file", ""))
            cached_stack_array = row.mask_stack_array
            cached_stack_indices = row.mask_stack_indices
            if isinstance(cached_bundle, dict):
                if cached_stack_array is None:
                    cached_stack_array = cached_bundle.get("mask_stack")
                if cached_stack_indices is None:
                    cached_stack_indices = cached_bundle.get("mask_indices")
            x, base_mask_np, degrade_term_masks = self._apply_degrade(x=x, mask=base_mask_np, idx=int(index))
            y, mask_stack_np, mask_idx_np = assemble_semantic_mask_layers(
                image=x,
                label_vec=y,
                registry=self.registry,
                original_parts=[(cached_stack_array, cached_stack_indices)],
                deformation_term_masks=degrade_term_masks,
            )
            x_t = torch.from_numpy(np.asarray(x, dtype=np.float32))
            y_t = torch.from_numpy(np.asarray(y, dtype=np.float32))
            h_img = int(x.shape[1]) if int(np.asarray(x).ndim) >= 3 else int(self.image_size)
            w_img = int(x.shape[2]) if int(np.asarray(x).ndim) >= 3 else int(self.image_size)
            mask_t = torch.zeros(1, h_img, w_img, dtype=torch.float32)
            if bool(self.return_mask_stack):
                return (
                    x_t,
                    y_t,
                    mask_t,
                    torch.from_numpy(np.asarray(mask_stack_np, dtype=np.float32)),
                    torch.from_numpy(np.asarray(mask_idx_np, dtype=np.int64)),
                )
            return (x_t, y_t, mask_t)
        return (
            torch.from_numpy(np.asarray(x, dtype=np.float32)),
            torch.from_numpy(np.asarray(y, dtype=np.float32)),
        )


class OverrideTargetSubsetDataset(Dataset):
    def __init__(self, dataset: Dataset, indices: Sequence[int], override_targets: np.ndarray):
        self.dataset = dataset
        self.indices = [int(i) for i in indices]
        self.override_targets = np.asarray(override_targets, dtype=np.float32)
        if int(self.override_targets.ndim) != 2:
            raise RuntimeError(
                f"OverrideTargetSubsetDataset override_targets must be [N,C], got {tuple(self.override_targets.shape)}"
            )
        if int(len(self.indices)) != int(self.override_targets.shape[0]):
            raise RuntimeError(
                "OverrideTargetSubsetDataset length mismatch: "
                f"indices={int(len(self.indices))} targets={int(self.override_targets.shape[0])}"
            )

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, index: int):
        base = self.dataset[int(self.indices[int(index)])]
        tgt = torch.from_numpy(np.asarray(self.override_targets[int(index)], dtype=np.float32).reshape(-1))
        if isinstance(base, (tuple, list)):
            if len(base) >= 5:
                return base[0], tgt, base[2], base[3], base[4]
            if len(base) >= 3:
                return base[0], tgt, base[2]
            if len(base) >= 2:
                return base[0], tgt
        if isinstance(base, dict):
            out = dict(base)
            out["y"] = tgt
            return out
        raise RuntimeError(f"Unsupported base dataset sample for override: {type(base).__name__}")


def build_layout_mask(layout: Dict[str, Any], height: int, width: int) -> np.ndarray:
    h = max(1, int(height))
    w = max(1, int(width))
    mask = np.zeros((int(h), int(w)), dtype=np.float32)
    boxes = layout.get("boxes", None) if isinstance(layout, dict) else None
    if isinstance(boxes, list):
        for box in boxes:
            if not isinstance(box, (list, tuple)) or len(box) < 4:
                continue
            x0, y0, x1, y1 = [float(v) for v in box[:4]]
            if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.01:
                ix0 = int(np.floor(np.clip(x0, 0.0, 1.0) * float(w)))
                iy0 = int(np.floor(np.clip(y0, 0.0, 1.0) * float(h)))
                ix1 = int(np.ceil(np.clip(x1, 0.0, 1.0) * float(w)))
                iy1 = int(np.ceil(np.clip(y1, 0.0, 1.0) * float(h)))
            else:
                ix0 = int(np.floor(np.clip(x0, 0.0, float(w))))
                iy0 = int(np.floor(np.clip(y0, 0.0, float(h))))
                ix1 = int(np.ceil(np.clip(x1, 0.0, float(w))))
                iy1 = int(np.ceil(np.clip(y1, 0.0, float(h))))
            ix0 = max(0, min(int(w), int(ix0)))
            ix1 = max(0, min(int(w), int(ix1)))
            iy0 = max(0, min(int(h), int(iy0)))
            iy1 = max(0, min(int(h), int(iy1)))
            if ix1 > ix0 and iy1 > iy0:
                mask[int(iy0): int(iy1), int(ix0): int(ix1)] = 1.0
    elif str(layout.get("type", "")).strip().lower() == "quadrants":
        for q in layout.get("active_quadrants", [0, 1, 2, 3]):
            qi = int(q)
            x0 = 0 if qi in (0, 2) else (w // 2)
            x1 = (w // 2) if qi in (0, 2) else w
            y0 = 0 if qi in (0, 1) else (h // 2)
            y1 = (h // 2) if qi in (0, 1) else h
            mask[int(y0): int(y1), int(x0): int(x1)] = 1.0
    return mask.astype(np.float32, copy=False)


def _find_existing_image_path(root: Path, stem: str) -> Optional[Path]:
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"):
        cand = root / f"{str(stem)}{str(ext)}"
        if cand.exists():
            return cand
    return None


def _sidecar_mask_path(image_path: Path) -> str:
    candidates = [
        image_path.with_name(f"{image_path.stem}.mask.png"),
        image_path.with_name(f"{image_path.stem}_mask.png"),
        image_path.with_name(f"{image_path.stem}.mask.jpg"),
        image_path.with_name(f"{image_path.stem}_mask.jpg"),
    ]
    for cand in candidates:
        if cand.exists():
            return str(cand)
    return ""


def _sidecar_layout(image_path: Path) -> Optional[Dict[str, Any]]:
    candidates = [
        image_path.with_name(f"{image_path.stem}.layout.json"),
        image_path.with_name(f"{image_path.stem}_layout.json"),
        image_path.with_suffix(".layout.json"),
    ]
    for cand in candidates:
        if not cand.exists():
            continue
        try:
            blob = json.loads(cand.read_text(encoding="utf-8"))
            if isinstance(blob, dict):
                return blob
        except Exception:
            continue
    return None


def _mask_cache_file_path(cache_root: Path, image_path: Path, terms: Sequence[str], mask_path: str = "", layout: Optional[Dict[str, Any]] = None) -> Path:
    key_blob = json.dumps(
        {
            "image_path": str(image_path),
            "image_mtime": (int(image_path.stat().st_mtime) if image_path.exists() else 0),
            "mask_path": str(mask_path),
            "mask_mtime": (int(Path(str(mask_path)).stat().st_mtime) if str(mask_path).strip() and Path(str(mask_path)).exists() else 0),
            "terms": list(normalize_vocab_terms([str(x) for x in terms])),
            "layout": layout if isinstance(layout, dict) else None,
            "cache_version": 6,
        },
        sort_keys=True,
        ensure_ascii=True,
    )
    digest = __import__("hashlib").sha256(key_blob.encode("utf-8")).hexdigest()
    return cache_root / f"{digest}.npz"


def _load_mask_cache_npz(cache_file: Path) -> Optional[Dict[str, np.ndarray]]:
    if not cache_file.exists():
        return None
    try:
        with np.load(str(cache_file), allow_pickle=False) as z:
            stack = np.asarray(z["mask_stack"], dtype=np.float32)
            indices = np.asarray(z["mask_indices"], dtype=np.int64)
        return {
            "mask_stack": stack,
            "mask_indices": indices,
        }
    except Exception:
        return None


def _load_row_mask_cache_bundle(cache_file: str) -> Optional[Dict[str, np.ndarray]]:
    path = Path(str(cache_file).strip()) if str(cache_file).strip() else None
    if path is None:
        return None
    return _load_mask_cache_npz(path)


def _build_and_store_row_mask_cache(
    cache_root: Path,
    image_path: Path,
    registry: Optional["DatasetTermRegistry"],
    terms: Sequence[str],
    mask_path: str = "",
    layout: Optional[Dict[str, Any]] = None,
    progress_control: Any = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], bool]:
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise_if_filesystem_space_emergency(
            progress_control,
            exc,
            note="semantic mask cache write",
            write_path=cache_root,
        )
        raise
    cache_file = _mask_cache_file_path(
        cache_root=cache_root,
        image_path=image_path,
        terms=terms,
        mask_path=mask_path,
        layout=layout,
    )
    cached = _load_mask_cache_npz(cache_file)
    if isinstance(cached, dict):
        return cached.get("mask_stack"), cached.get("mask_indices"), False
    with Image.open(str(image_path)) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.float32)
    chw = np.transpose(np.clip(rgb / 255.0, 0.0, 1.0).astype(np.float32, copy=False), (2, 0, 1)).astype(np.float32, copy=False)
    h = int(chw.shape[1])
    w = int(chw.shape[2])

    explicit_mask = None
    if str(mask_path).strip():
        mp = Path(str(mask_path))
        if mp.exists():
            if str(mp.suffix).strip().lower() in (".mat", ".npz"):
                _npz_path = mp.with_suffix(".npz")
                _mat_path = mp.with_suffix(".mat")
                try:
                    if _npz_path.exists():
                        _data = np.load(str(_npz_path), allow_pickle=False)
                        seg = np.asarray(_data["segmentation"], dtype=np.float32)
                        explicit_mask = (seg > 0.0).astype(np.float32, copy=False)
                    elif _mat_path.exists():
                        from scipy.io import loadmat
                        blob = loadmat(str(_mat_path), squeeze_me=False, struct_as_record=False)
                        seg = None
                        gtcls = blob.get("GTcls", None)
                        try:
                            seg = np.asarray(gtcls[0, 0].Segmentation, dtype=np.float32)
                        except Exception:
                            try:
                                seg = np.asarray(gtcls.Segmentation[0, 0], dtype=np.float32)
                            except Exception:
                                seg = None
                        if seg is not None:
                            explicit_mask = (np.asarray(seg, dtype=np.float32) > 0.0).astype(np.float32, copy=False)
                except Exception:
                    explicit_mask = None
            else:
                with Image.open(str(mp)) as mm:
                    explicit_mask = np.asarray(mm.convert("L"), dtype=np.float32)
            if explicit_mask is not None:
                explicit_mask = _normalize_mask_array(explicit_mask, height=int(h), width=int(w))
                _vm = float(np.max(explicit_mask))
                explicit_mask = (explicit_mask / _vm).astype(np.float32) if _vm > 1e-8 else np.zeros_like(explicit_mask, dtype=np.float32)
    if explicit_mask is None and isinstance(layout, dict):
        explicit_mask = np.maximum(np.asarray(build_layout_mask(layout, height=int(h), width=int(w)), dtype=np.float32), 0.0)
        _vm = float(np.max(explicit_mask))
        explicit_mask = (explicit_mask / _vm).astype(np.float32) if _vm > 1e-8 else np.zeros_like(explicit_mask, dtype=np.float32)

    _t2i = registry.term_to_idx if registry is not None else {}
    label_vec = targets_from_terms([list(terms)], _t2i, max(len(_t2i), 1))[0]
    _, inferred_stack, inferred_idx = assemble_semantic_mask_layers(
        image=chw,
        label_vec=label_vec,
        registry=registry,
    )
    try:
        np.savez_compressed(
            str(cache_file),
            mask_stack=np.asarray(inferred_stack, dtype=np.float32),
            mask_indices=np.asarray(inferred_idx, dtype=np.int64),
        )
    except Exception as exc:
        raise_if_filesystem_space_emergency(
            progress_control,
            exc,
            note="semantic mask cache write",
            write_path=cache_file,
        )
        pass
    return np.asarray(inferred_stack, dtype=np.float32), np.asarray(inferred_idx, dtype=np.int64), True


def _resolve_semantic_mask_cache_root(root: Path) -> Path:
    return root / "cache" / "semantic_mask_cache"


def _materialize_semantic_row_mask_cache(
    rows: Sequence[SemanticDiskRow],
    class_names: Sequence[str],
    cache_root: Path,
    max_workers: int = 0,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    info = {
        "mask_cache_enabled": bool(len(rows) > 0),
        "mask_cache_root": str(cache_root),
        "mask_cache_rows": int(len(rows)),
        "mask_cache_hits": 0,
        "mask_cache_writes": 0,
        "mask_cache_failures": 0,
        "mask_cache_threads": 0,
        "mask_cache_seconds": 0.0,
    }
    if int(len(rows)) <= 0:
        return info
    _registry = DatasetTermRegistry()
    _registry.register_many(list(class_names))
    workers = int(max_workers)
    if workers <= 0:
        workers = min(8, max(1, (os.cpu_count() or 1)))
    info["mask_cache_threads"] = int(workers)

    def _job(row_index: int) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray], bool]:
        row = rows[int(row_index)]
        cache_file = _mask_cache_file_path(
            cache_root=cache_root,
            image_path=Path(str(row.image_path)),
            terms=row.terms,
            mask_path=str(row.mask_path or ""),
            layout=(dict(row.layout) if isinstance(row.layout, dict) else None),
        )
        if cache_file.exists():
            return int(row_index), None, None, False
        stack, indices, created = _build_and_store_row_mask_cache(
            cache_root=cache_root,
            image_path=Path(str(row.image_path)),
            registry=_registry,
            terms=row.terms,
            mask_path=str(row.mask_path or ""),
            layout=(dict(row.layout) if isinstance(row.layout, dict) else None),
            progress_control=None,
        )
        return int(row_index), stack, indices, bool(created)

    if int(workers) <= 1:
        for row_index in range(int(len(rows))):
            try:
                ridx, stack, indices, created = _job(row_index)
                row = rows[int(ridx)]
                row.mask_cache_file = str(
                    _mask_cache_file_path(
                        cache_root=cache_root,
                        image_path=Path(str(row.image_path)),
                        terms=row.terms,
                        mask_path=str(row.mask_path or ""),
                        layout=(dict(row.layout) if isinstance(row.layout, dict) else None),
                    )
                )
                row.mask_array = None
                row.mask_stack_array = None
                row.mask_stack_indices = None
                if bool(created):
                    info["mask_cache_writes"] = int(info["mask_cache_writes"]) + 1
                else:
                    info["mask_cache_hits"] = int(info["mask_cache_hits"]) + 1
            except StageStopRequested:
                raise
            except Exception:
                info["mask_cache_failures"] = int(info["mask_cache_failures"]) + 1
        info["mask_cache_seconds"] = float(max(0.0, time.perf_counter() - t0))
        return info

    with ThreadPoolExecutor(max_workers=int(workers), thread_name_prefix="semantic-mask-cache") as pool:
        pending: Dict[Any, int] = {}
        submit_cursor = 0
        max_inflight = max(int(workers), int(workers) * 2)

        while submit_cursor < int(len(rows)) and int(len(pending)) < int(max_inflight):
            fut = pool.submit(_job, submit_cursor)
            pending[fut] = int(submit_cursor)
            submit_cursor += 1

        while len(pending) > 0:
            done_batch = next(as_completed(list(pending.keys())))
            pending.pop(done_batch, None)
            try:
                ridx, stack, indices, created = done_batch.result()
                row = rows[int(ridx)]
                row.mask_cache_file = str(
                    _mask_cache_file_path(
                        cache_root=cache_root,
                        image_path=Path(str(row.image_path)),
                        terms=row.terms,
                        mask_path=str(row.mask_path or ""),
                        layout=(dict(row.layout) if isinstance(row.layout, dict) else None),
                    )
                )
                row.mask_array = None
                row.mask_stack_array = None
                row.mask_stack_indices = None
                if bool(created):
                    info["mask_cache_writes"] = int(info["mask_cache_writes"]) + 1
                else:
                    info["mask_cache_hits"] = int(info["mask_cache_hits"]) + 1
            except StageStopRequested:
                raise
            except Exception:
                info["mask_cache_failures"] = int(info["mask_cache_failures"]) + 1
            while submit_cursor < int(len(rows)) and int(len(pending)) < int(max_inflight):
                fut = pool.submit(_job, submit_cursor)
                pending[fut] = int(submit_cursor)
                submit_cursor += 1
    info["mask_cache_seconds"] = float(max(0.0, time.perf_counter() - t0))
    return info


def _folder_source_signature(root: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"root": str(root), "exists": bool(root.exists()), "datasets": []}
    if not root.exists():
        return out
    ds_dirs = [p for p in root.iterdir() if p.is_dir()]
    ds_dirs = sorted(ds_dirs, key=lambda p: p.name.lower())
    for ds_dir in ds_dirs:
        files = [p for p in ds_dir.rglob("*") if p.is_file() and str(p.suffix).strip().lower() in _IMAGE_SUFFIXES]
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


def _disk_rows_cache_path(data_root: str, cache_key: str) -> Path:
    """Fixed path for the on-disk row cache, keyed by logical scan parameters."""
    key_hash = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:16]
    root = Path(str(data_root).strip() or "data/berkeley_sbd")
    return root / "cache" / f"semantic_disk_rows_{key_hash}.pkl.gz"


def _disk_rows_freshness_sig(root: Path, ext_root: Path) -> str:
    """Cheap string fingerprint of on-disk data; changes when files are added/replaced."""
    parts: List[str] = []
    for split_name in ("train", "val"):
        lp = root / "cache" / f"sbd_{split_name}_multilabel.npz"
        try:
            st = lp.stat()
            parts.append(f"{split_name}:{st.st_mtime:.0f}:{st.st_size}")
        except Exception:
            parts.append(f"{split_name}:missing")
    if ext_root.exists():
        try:
            ds_dirs = sorted(
                [p for p in ext_root.iterdir() if p.is_dir()],
                key=lambda p: p.name.lower(),
            )
            for ds_dir in ds_dirs:
                files = [
                    p for p in ds_dir.rglob("*")
                    if p.is_file() and str(p.suffix).strip().lower() in _IMAGE_SUFFIXES
                ]
                latest = max((fp.stat().st_mtime for fp in files), default=0.0)
                parts.append(f"ext:{ds_dir.name}:{len(files)}:{latest:.0f}")
        except Exception:
            parts.append("ext:sig_error")
    return "|".join(parts)


def collect_semantic_disk_rows(
    data_root: str,
    class_names: Sequence[str],
    source_root: str = "",
    progress_control: Any = None,
) -> Tuple[List[SemanticDiskRow], Dict[str, Any]]:
    root = Path(str(data_root).strip() or "data/berkeley_sbd")
    cache_key = _semantic_disk_rows_cache_key(
        data_root=str(root),
        class_names=class_names,
        source_root=str(source_root),
    )
    with _SEMANTIC_DISK_ROWS_CACHE_LOCK:
        cached_payload = _SEMANTIC_DISK_ROWS_CACHE.get(cache_key)
    if cached_payload is not None:
        cached_rows, cached_info = cached_payload
        info_out = dict(cached_info)
        info_out["inprocess_cache_hit"] = True
        return _clone_semantic_disk_rows(cached_rows), info_out

    t0 = time.perf_counter()

    # --- Disk cache check (survives process restarts) ---
    _ext_root_pre = Path(str(source_root).strip()) if str(source_root).strip() else (root / "payload_sources")
    _freshness_sig = _disk_rows_freshness_sig(root, _ext_root_pre)
    _disk_cache_file = _disk_rows_cache_path(str(root), cache_key)
    if _disk_cache_file.exists():
        try:
            with gzip.open(str(_disk_cache_file), "rb") as _f:
                _bundle = pickle.load(_f)
            if _bundle.get("freshness_sig") == _freshness_sig:
                _rows = _bundle["rows"]
                _info = dict(_bundle["info"])
                _info["disk_cache_hit"] = True
                _info["inprocess_cache_hit"] = False
                _info["row_build_seconds"] = float(max(0.0, time.perf_counter() - t0))
                with _SEMANTIC_DISK_ROWS_CACHE_LOCK:
                    _SEMANTIC_DISK_ROWS_CACHE[cache_key] = (_clone_semantic_disk_rows(_rows), dict(_info))
                print(
                    f"[collect-disk-rows] disk cache hit — {len(_rows)} rows"
                    f" in {_info['row_build_seconds']:.3f}s",
                    flush=True,
                )
                return _clone_semantic_disk_rows(_rows), _info
        except Exception:
            pass  # stale / corrupt — fall through to full scan

    class_lut = {_norm_txt(name): int(i) for i, name in enumerate(class_names)}
    n_classes = max(1, int(len(class_names)))
    rows: List[SemanticDiskRow] = []
    source_counts: Dict[str, int] = {}
    loaded_split_counts: Dict[str, int] = {"train": 0, "val": 0}
    missing_images = 0
    startup_row_threads = 1

    berkeley_dataset_idx = int(class_lut.get("berkeley sbd dataset", -1))
    object_idx = int(class_lut.get("object", -1))
    signal_idx = int(class_lut.get("signal", -1))
    try:
        from berkeley_sbd_pretrain import VOC20_CLASSES as _VOC20_CLASSES
        voc20_names = [str(x) for x in list(_VOC20_CLASSES)]
    except Exception:
        voc20_names = []

    convert_sbd_mat_to_npz(root, progress_control=progress_control)
    split_specs = [("train", "berkeley_sbd_train"), ("val", "berkeley_sbd_val")]
    for split_name, source_key in split_specs:
        _split_images, _mask_paths = _read_sbd_split_file(root, split_name)
        label_path = root / "cache" / f"sbd_{split_name}_multilabel.npz"
        if not label_path.exists():
            try:
                from berkeley_sbd_pretrain import _labels_from_segmentation_masks, _load_sbd_split
                print(f"[collect-disk-rows] label cache missing — auto-building for split={split_name}...", flush=True)
                _ds = _load_sbd_split(root, image_set=split_name, download=False)
                _labels_from_segmentation_masks(_ds, label_path, progress_control=progress_control)
            except StageStopRequested:
                raise
            except Exception as _build_exc:
                raise RuntimeError(
                    "Berkeley multilabel cache is missing and could not be auto-built: "
                    f"{label_path}. Error: {_build_exc}"
                ) from _build_exc
        with np.load(str(label_path), allow_pickle=False) as z:
            if "labels" not in z.files:
                raise RuntimeError(f"'labels' key missing in {label_path}")
            labels_split = np.asarray(z["labels"], dtype=np.float32)
        if int(labels_split.shape[0]) != int(len(_split_images)):
            raise RuntimeError(
                "Berkeley image/label row mismatch for disk rows: "
                f"split={split_name} images={int(len(_split_images))} labels={int(labels_split.shape[0])}"
            )
        split_specs_rows: List[Tuple[Path, np.ndarray, str, str, str, List[str]]] = []
        split_missing = 0
        for i, img_path in enumerate(_split_images):
            ip = Path(str(img_path))
            if not ip.exists():
                split_missing += 1
                continue
            yv = np.zeros((int(n_classes),), dtype=np.float32)
            voc_vec = np.asarray(labels_split[int(i)], dtype=np.float32).reshape(-1)
            positive_voc_terms: List[str] = []
            for voc_idx in np.flatnonzero(voc_vec > 0.5):
                if 0 <= int(voc_idx) < int(len(voc20_names)):
                    positive_voc_terms.append(str(voc20_names[int(voc_idx)]))
            if int(berkeley_dataset_idx) >= 0:
                yv[int(berkeley_dataset_idx)] = 1.0
            if int(object_idx) >= 0:
                yv[int(object_idx)] = 1.0
            if int(signal_idx) >= 0:
                yv[int(signal_idx)] = 1.0
            split_specs_rows.append(
                (
                    ip,
                    np.asarray(yv, dtype=np.float32).reshape(-1).copy(),
                    str(_mask_paths[int(i)]) if int(i) < int(len(_mask_paths)) else "",
                    str(source_key),
                    str(split_name),
                    positive_voc_terms,
                )
            )
        missing_images += int(split_missing)

        split_workers = _resolve_semantic_startup_threads(len(split_specs_rows))
        startup_row_threads = max(int(startup_row_threads), int(split_workers))

        def _build_berkeley_row(spec: Tuple[Path, np.ndarray, str, str, str, List[str]]) -> SemanticDiskRow:
            ip, yv_base, mask_path_local, source_key_local, _, voc_terms = spec
            base_terms = ["berkeley sbd dataset", "object", "signal"] + [str(t) for t in voc_terms]
            terms = list(normalize_vocab_terms(base_terms))
            return SemanticDiskRow(
                image_path=str(ip),
                terms=terms,
                source=str(source_key_local),
                mask_path=str(mask_path_local),
            )

        for row in _ordered_thread_map(
            split_specs_rows,
            _build_berkeley_row,
            max_workers=int(split_workers),
            desc=f"[berkeley/{split_name}] loading rows",
            progress_control=progress_control,
        ):
            rows.append(row)
            source_counts[str(source_key)] = int(source_counts.get(str(source_key), 0)) + 1
            loaded_split_counts[str(split_name)] = int(loaded_split_counts.get(str(split_name), 0)) + 1

    ext_root = Path(str(source_root).strip()) if str(source_root).strip() else (root / "payload_sources")
    ext_sig = _folder_source_signature(ext_root)
    external_unmapped_skipped = 0
    if ext_root.exists():
        ds_dirs = [p for p in ext_root.iterdir() if p.is_dir()]
        ds_dirs = sorted(ds_dirs, key=lambda p: p.name.lower())
        external_specs: List[Tuple[Path, str, str]] = []
        for ds_dir in ds_dirs:
            dataset_name = re.sub(r"[_\-]+", " ", str(ds_dir.name)).strip()
            if not str(dataset_name):
                continue
            files = [p for p in ds_dir.rglob("*") if p.is_file() and str(p.suffix).strip().lower() in _IMAGE_SUFFIXES]
            files = sorted(files, key=lambda p: str(p).lower())
            for fp in files:
                rel = fp.relative_to(ds_dir)
                label_term = ""
                if len(rel.parts) > 1:
                    label_term = re.sub(r"[_\-]+", " ", str(rel.parts[0])).strip()
                external_specs.append((fp, str(dataset_name), str(label_term)))

        external_workers = _resolve_semantic_startup_threads(len(external_specs))
        startup_row_threads = max(int(startup_row_threads), int(external_workers))

        def _build_external_row(spec: Tuple[Path, str, str]) -> Tuple[Optional[SemanticDiskRow], int]:
            fp, dataset_name_local, label_term_local = spec
            dataset_terms = [str(dataset_name_local)]
            if not str(dataset_name_local).strip().lower().endswith("dataset"):
                dataset_terms.append(f"{dataset_name_local} dataset")
            specific_candidates = list(dataset_terms)
            if str(label_term_local):
                specific_candidates.append(str(label_term_local))
            mapped_terms: List[str] = []
            specific_hits = 0
            specific_set = {str(x) for x in specific_candidates}
            for cand in (["berkeley sbd dataset", "signal"] + list(specific_candidates)):
                key = _norm_txt(cand)
                if key in class_lut:
                    mapped_terms.append(str(class_names[int(class_lut[key])]))
                    if str(cand) in specific_set:
                        specific_hits += 1
            if int(specific_hits) <= 0:
                return None, 1
            terms = list(normalize_vocab_terms(mapped_terms))
            return SemanticDiskRow(
                image_path=str(fp),
                terms=terms,
                source=str(dataset_name_local),
                mask_path=_sidecar_mask_path(fp),
                layout=_sidecar_layout(fp),
            ), 0

        for row, unmapped_skip in _ordered_thread_map(
            external_specs,
            _build_external_row,
            max_workers=int(external_workers),
            desc="[external] loading rows",
            progress_control=progress_control,
        ):
            external_unmapped_skipped += int(unmapped_skip)
            if row is None:
                continue
            rows.append(row)
            source_counts[str(row.source)] = int(source_counts.get(str(row.source), 0)) + 1

    info = {
        "available_rows": int(len(rows)),
        "available_train": int(loaded_split_counts.get("train", 0)),
        "available_val": int(loaded_split_counts.get("val", 0)),
        "available_external": int(max(0, int(len(rows)) - int(loaded_split_counts.get("train", 0)) - int(loaded_split_counts.get("val", 0)))),
        "source_counts": {str(k): int(v) for k, v in source_counts.items()},
        "missing_images": int(missing_images),
        "external_source_root": str(ext_root),
        "external_source_signature": ext_sig,
        "external_unmapped_skipped": int(external_unmapped_skipped),
        "row_build_threads": int(startup_row_threads),
        "row_build_seconds": float(max(0.0, time.perf_counter() - t0)),
        "inprocess_cache_hit": False,
        "mask_cache_enabled": False,
        "mask_cache_hits": 0,
        "mask_cache_writes": 0,
        "mask_cache_failures": 0,
        "mask_cache_seconds": 0.0,
        "mask_cache_threads": 0,
    }
    info["disk_cache_hit"] = False
    with _SEMANTIC_DISK_ROWS_CACHE_LOCK:
        _SEMANTIC_DISK_ROWS_CACHE[cache_key] = (_clone_semantic_disk_rows(rows), dict(info))
    # --- Write disk cache ---
    try:
        _disk_cache_file.parent.mkdir(parents=True, exist_ok=True)
        _bundle = {"freshness_sig": _freshness_sig, "rows": rows, "info": info}
        with gzip.open(str(_disk_cache_file), "wb", compresslevel=1) as _f:
            pickle.dump(_bundle, _f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[collect-disk-rows] disk cache written → {_disk_cache_file}", flush=True)
    except Exception as _write_exc:
        raise_if_filesystem_space_emergency(
            progress_control,
            _write_exc,
            note="semantic disk-row cache write",
            write_path=_disk_cache_file,
        )
        print(f"[collect-disk-rows] disk cache write failed: {_write_exc}", flush=True)
    return rows, info
