from __future__ import annotations

import queue
import json
import hashlib
import os
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data._utils.collate import default_collate
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
_SEMANTIC_DISK_ROWS_CACHE_LOCK = threading.Lock()
_SEMANTIC_DISK_ROWS_CACHE: Dict[str, Tuple[List["SemanticDiskRow"], Dict[str, Any]]] = {}
_SEMANTIC_COLOR_TERMS: Tuple[str, ...] = (
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


def _store_cache_control(cache_root: Path, payload: Dict[str, Any]) -> None:
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    path = _cache_control_file(root)
    try:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
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


def convert_sbd_mat_to_npz(root) -> None:
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
                    _np.savez_compressed(str(mat_path.with_suffix(".npz")), segmentation=seg)
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
    label_vec: np.ndarray
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
        label_vec=np.asarray(row.label_vec, dtype=np.float32).reshape(-1).copy(),
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
) -> List[Any]:
    count = int(len(items))
    if count <= 0:
        return []
    workers = max(1, min(int(max_workers), int(count)))
    if workers <= 1:
        return [
            worker_fn(item)
            for item in tqdm(items, desc=desc, unit="row", leave=False, dynamic_ncols=True)
        ]
    results: List[Any] = [None] * count
    with ThreadPoolExecutor(max_workers=int(workers), thread_name_prefix="semantic-row-build") as pool:
        future_to_index = {pool.submit(worker_fn, item): int(i) for i, item in enumerate(items)}
        with tqdm(total=count, desc=desc, unit="row", leave=False, dynamic_ncols=True) as pbar:
            for future in as_completed(future_to_index):
                idx = int(future_to_index[future])
                results[idx] = future.result()
                pbar.update(1)
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


def _normalize_attention_map(mask: Any, gamma: float = 1.0, blur_kernel: int = 0, apply_vmean_compression: bool = False, apply_binarize: bool = False) -> np.ndarray:
    arr = np.asarray(mask, dtype=np.float32)
    if int(arr.ndim) != 2 or int(arr.size) <= 0:
        return np.zeros_like(np.asarray(arr, dtype=np.float32), dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    arr = np.maximum(arr, 0.0).astype(np.float32, copy=False)
    vmax = float(np.max(arr)) if int(arr.size) > 0 else 0.0
    if vmax > 1e-8:
        arr = (arr / float(vmax)).astype(np.float32, copy=False)
    else:
        return np.zeros_like(arr, dtype=np.float32)
    if bool(apply_vmean_compression):
        vmean = float(np.mean(arr)) if int(arr.size) > 0 else 0.0
        if vmean > 1e-8:
            arr = np.clip(arr / float(max(vmean * 2.0, 1.0)), 0.0, 1.0).astype(np.float32, copy=False)
    if bool(apply_binarize):
        return (arr > 0.5).astype(np.float32, copy=False)
    gm = max(0.35, float(gamma))
    if abs(gm - 1.0) > 1e-6:
        arr = np.power(np.clip(arr, 0.0, 1.0), gm).astype(np.float32, copy=False)
    kk = int(blur_kernel)
    if kk >= 3:
        kk = int(kk) | 1
        arr = np.asarray(
            F.avg_pool2d(torch.from_numpy(arr[None, None, ...]), kernel_size=int(kk), stride=1, padding=int(kk // 2))[0, 0].cpu().numpy(),
            dtype=np.float32,
        )
        vmax = float(np.max(arr)) if int(arr.size) > 0 else 0.0
        if vmax > 1e-8:
            arr = (arr / float(vmax)).astype(np.float32, copy=False)
    return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)


def _blend_attention_maps(maps: Sequence[Any], weights: Optional[Sequence[float]] = None, gamma: float = 1.0, apply_vmean_compression: bool = False, apply_binarize: bool = False) -> np.ndarray:
    valid: List[np.ndarray] = []
    valid_weights: List[float] = []
    for i, m in enumerate(maps):
        arr = np.asarray(m, dtype=np.float32)
        if int(arr.ndim) != 2 or int(arr.size) <= 0:
            continue
        w = 1.0 if weights is None or i >= int(len(weights)) else float(weights[i])
        if w <= 0.0:
            continue
        norm = _normalize_attention_map(arr, gamma=1.0, blur_kernel=0, apply_vmean_compression=bool(apply_vmean_compression))
        if float(np.max(norm)) <= 1e-8:
            continue
        valid.append(norm)
        valid_weights.append(float(w))
    if len(valid) <= 0:
        if len(maps) > 0:
            ref = np.asarray(maps[0], dtype=np.float32)
            return np.zeros_like(ref, dtype=np.float32)
        return np.zeros((0, 0), dtype=np.float32)
    acc = np.zeros_like(valid[0], dtype=np.float32)
    wsum = 0.0
    for arr, w in zip(valid, valid_weights):
        acc += float(w) * arr
        wsum += float(w)
    if wsum > 1e-8:
        acc = acc / float(wsum)
    return _normalize_attention_map(acc, gamma=float(gamma), blur_kernel=5, apply_vmean_compression=bool(apply_vmean_compression), apply_binarize=bool(apply_binarize))


def _composite_mask_stack(stack: Any, processing_device: Optional[Any] = None) -> np.ndarray:
    """Additive-sum all per-label masks, then normalize to [0, 1] by vmax only.

    Individual masks must already be in [0, 1] (via _normalize_stack_row /
    _normalize_stack_row_batch).  The sum is divided by its maximum so that
    pixels covered by the most labels reach 1.0; no vmean/gamma distortion is
    applied here.  Attention-style normalization belongs on the final
    mixed_mask stored in the cache entry, not on the raw stack composite.
    """
    arr = np.asarray(stack, dtype=np.float32)
    if int(arr.ndim) != 3 or int(arr.shape[0]) <= 0:
        return np.zeros((0, 0) if int(arr.ndim) < 2 else (int(arr.shape[-2]), int(arr.shape[-1])), dtype=np.float32)
    composite = np.sum(arr, axis=0).astype(np.float32, copy=False)
    vmax = float(np.max(composite))
    if vmax > 1e-8:
        composite = np.clip(composite / vmax, 0.0, 1.0).astype(np.float32, copy=False)
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
    apply_attention_normalization: bool = False,
) -> np.ndarray:
    clamped = _normalize_mask_array(mask, height=int(height), width=int(width))
    if not bool(apply_attention_normalization):
        return clamped
    resolved = _resolve_processing_device(processing_device)
    if resolved is None:
        return _normalize_attention_map(
            clamped,
            gamma=1.0,
            blur_kernel=0,
            apply_vmean_compression=True,
        )
    return np.asarray(
        _normalize_attention_map_batch_torch(clamped, gamma=1.0, blur_kernel=0, device=resolved, apply_vmean_compression=True).detach().cpu().numpy(),
        dtype=np.float32,
    )


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
    resolved = _resolve_processing_device(processing_device)
    if resolved is None:
        return _normalize_attention_map_batch(clamped, gamma=1.0, blur_kernel=0, apply_vmean_compression=True)
    return np.asarray(
        _normalize_attention_map_batch_torch(clamped, gamma=1.0, blur_kernel=0, device=resolved, apply_vmean_compression=True).detach().cpu().numpy(),
        dtype=np.float32,
    )


def build_creation_label_mask_stack(
    label_vec: Any,
    *,
    height: int = 0,
    width: int = 0,
    creation_mask: Optional[Any] = None,
    processing_device: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    positive_idx = _positive_label_indices(label_vec)
    if creation_mask is not None:
        base = np.asarray(creation_mask, dtype=np.float32)
        if int(base.ndim) == 3:
            base = np.mean(base, axis=0).astype(np.float32, copy=False)
        if int(height) <= 0 or int(width) <= 0:
            height = int(base.shape[-2]) if int(base.ndim) >= 2 else int(height)
            width = int(base.shape[-1]) if int(base.ndim) >= 2 else int(width)
        base_mask = _normalize_stack_row(
            base,
            height=int(height),
            width=int(width),
            processing_device=processing_device,
        )
    else:
        return np.zeros((0, int(max(0, height)), int(max(0, width))), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    if int(positive_idx.size) <= 0:
        return np.zeros((0, int(base_mask.shape[0]), int(base_mask.shape[1])), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    stack = np.repeat(base_mask[None, :, :], int(positive_idx.size), axis=0).astype(np.float32, copy=False)
    return stack, np.asarray(positive_idx, dtype=np.int64)


def term_mask_map_to_label_stack(
    term_mask_map: Optional[Dict[str, Any]],
    label_vec: Any,
    *,
    term_to_idx: Optional[Dict[str, int]] = None,
    idx_to_term: Optional[Dict[int, str]] = None,
    height: int = 0,
    width: int = 0,
    processing_device: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if not isinstance(term_mask_map, dict) or not term_mask_map:
        return np.zeros((0, int(max(0, height)), int(max(0, width))), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    lut = {
        _norm_txt(str(k)): int(v)
        for k, v in ((term_to_idx or {}).items())
        if str(k).strip()
    }
    if not lut and isinstance(idx_to_term, dict):
        lut = {
            _norm_txt(str(term)): int(idx)
            for idx, term in idx_to_term.items()
            if str(term).strip()
        }
    positive_set = set(_positive_label_indices(label_vec).tolist())
    raw_masks: List[np.ndarray] = []
    indices: List[int] = []
    for raw_term, raw_mask in term_mask_map.items():
        ci = int(lut.get(_norm_txt(str(raw_term)), -1))
        if ci < 0 or ci not in positive_set:
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
        valid_mask = (np.asarray(idx_arr[:int(pair_count)], dtype=np.int64) >= 0) & (vmax_per > 1e-8)
        for si in np.where(valid_mask)[0]:
            cls_idx = int(idx_arr[int(si)])
            rows.append(batch_norm[int(si)])
            indices.append(cls_idx)
            covered.add(cls_idx)

    missing = [int(ci) for ci in positive_idx.tolist() if int(ci) not in covered]
    if missing and bool(strict):
        raise ValueError(
            f"combine_label_mask_stacks: {len(missing)} positive label(s) have no spatial mask. "
            f"Missing label indices: {missing[:8]}{'…' if len(missing) > 8 else ''}"
        )
    if len(rows) <= 0:
        return np.zeros((0, int(max(0, height)), int(max(0, width))), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(rows, axis=0).astype(np.float32, copy=False), np.asarray(indices, dtype=np.int64)


def flatten_density_to_transmissivity(stack: Any) -> np.ndarray:
    """Max-projection of a label stack: each pixel shows peak single-label coverage.

    Represents transmissivity — which pixels are blocked/covered by any label at
    all, ignoring how many labels overlap (density is lost).  Suitable for
    visualization, attention gating, or future training modes where occlusion
    without density is the semantic.  Do NOT use for density-based training.
    """
    arr = np.asarray(stack, dtype=np.float32)
    if int(arr.ndim) != 3 or int(arr.shape[0]) <= 0:
        return np.zeros((0, 0) if int(arr.ndim) < 2 else (int(arr.shape[-2]), int(arr.shape[-1])), dtype=np.float32)
    flat = np.max(arr, axis=0).astype(np.float32, copy=False)
    return _normalize_attention_map(flat, gamma=1.0, blur_kernel=0)


def depth_map_from_mask_stack(stack: Any, label_indices: Optional[Sequence[int]] = None) -> np.ndarray:
    """Depth-weighted composite of a label stack.

    Labels at a lower stack position (earlier in the label list = higher semantic
    priority / foreground) contribute more strongly than later labels.  Weight
    for position i in a stack of N is (N - i) / N, so position 0 → weight 1.0
    and the last position → weight 1/N.  Optionally pass ``label_indices`` (a
    sequence of raw label indices) to remap the depth ordering to the original
    label-list position rather than the stack-row position.

    Suitable for layered spatial attention or future training modes where label
    hierarchy encodes scene depth.  Do NOT use for density-based training.
    """
    arr = np.asarray(stack, dtype=np.float32)
    if int(arr.ndim) != 3 or int(arr.shape[0]) <= 0:
        return np.zeros((0, 0) if int(arr.ndim) < 2 else (int(arr.shape[-2]), int(arr.shape[-1])), dtype=np.float32)
    n = int(arr.shape[0])
    if label_indices is not None and int(len(label_indices)) == n:
        order = [int(x) for x in label_indices]
        rank = {idx: pos for pos, idx in enumerate(sorted(set(order)))}
        max_rank = max(rank.values()) if rank else 0
        weights = np.array(
            [float(max_rank - rank.get(order[i], max_rank)) / float(max(1, max_rank)) for i in range(n)],
            dtype=np.float32,
        )
    else:
        weights = np.linspace(1.0, 1.0 / float(max(1, n)), n, dtype=np.float32)
    weighted = np.sum(arr * weights[:, None, None], axis=0).astype(np.float32, copy=False)
    return _normalize_attention_map(weighted, gamma=1.0, blur_kernel=0)


def elem_stacks_to_label_stacks(
    elem_stack: np.ndarray,
    elem_term_lists: Sequence[Sequence[str]],
    label_vec: np.ndarray,
    term_to_idx: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert per-element spatial masks into per-label mask stacks.

    Each element mask is replicated for every positive label index that
    appears in that element's term list.  The result is a ``[K, H, W]``
    stack paired with a ``[K]`` int64 index array suitable for direct
    lookup by ``_expand_semantic_mask_supervision_batch``.
    """
    y = np.asarray(label_vec, dtype=np.float32).reshape(-1)
    positive_set = set(np.where(y >= 0.5)[0].tolist())
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
        # Batch-scale all element masks to [0, 1] at once (no attention normalization).
        batch_norm = _normalize_stack_row_batch(stack[:n_elems], height=int(h), width=int(w))
        for ei in range(n_elems):
            mask_e = batch_norm[ei]
            for term in elem_term_lists[int(ei)]:
                tk = _norm_txt(str(term))  # cached
                ci = int(term_to_idx.get(tk, -1))
                if ci < 0 or ci not in positive_set:
                    continue
                out_masks.append(mask_e)
                out_idx.append(ci)
                covered.add(int(ci))
    if len(out_masks) == 0:
        h, w = int(stack.shape[-2]), int(stack.shape[-1])
        return np.zeros((0, h, w), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(out_masks, axis=0).astype(np.float32, copy=False), np.asarray(out_idx, dtype=np.int64)


def _is_dataset_label_term(term: str) -> bool:
    return bool(_norm_txt(str(term)).endswith(_DATASET_LABEL_SUFFIX))


def _resolve_creation_mask(
    mixed_mask: Any,
    *,
    height: int,
    width: int,
    mask_stack_array: Optional[Any] = None,
    processing_device: Optional[Any] = None,
) -> np.ndarray:
    mixed_arr = np.asarray(mixed_mask, dtype=np.float32)
    if int(mixed_arr.ndim) >= 2 and float(np.max(mixed_arr)) > 1e-8:
        return _normalize_stack_row(
            mixed_arr,
            height=int(height),
            width=int(width),
            processing_device=processing_device,
            apply_attention_normalization=True,
        )
    if mask_stack_array is None:
        return np.zeros((int(max(0, height)), int(max(0, width))), dtype=np.float32)
    stack_arr = np.asarray(mask_stack_array, dtype=np.float32)
    if int(stack_arr.ndim) == 2:
        stack_arr = stack_arr[None, ...]
    if int(stack_arr.ndim) != 3 or int(stack_arr.shape[0]) <= 0:
        return np.zeros((int(max(0, height)), int(max(0, width))), dtype=np.float32)
    if int(height) <= 0 or int(width) <= 0:
        height = int(stack_arr.shape[-2])
        width = int(stack_arr.shape[-1])
    return _normalize_stack_row(
        _composite_mask_stack(stack_arr, processing_device=processing_device),
        height=int(height),
        width=int(width),
        processing_device=processing_device,
        apply_attention_normalization=True,
    )


def _build_special_label_mask_stack(
    label_vec: Any,
    *,
    idx_to_term: Optional[Dict[int, str]] = None,
    height: int,
    width: int,
    creation_mask: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    positive_idx = _positive_label_indices(label_vec)
    if int(positive_idx.size) <= 0 or int(height) <= 0 or int(width) <= 0:
        return np.zeros((0, int(max(0, height)), int(max(0, width))), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    if not isinstance(idx_to_term, dict) or not idx_to_term:
        return np.zeros((0, int(height), int(width)), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    creation = np.asarray(creation_mask, dtype=np.float32) if creation_mask is not None else np.zeros((int(height), int(width)), dtype=np.float32)
    creation_ok = int(creation.ndim) == 2 and float(np.max(creation)) > 1e-8
    full_frame = np.ones((int(height), int(width)), dtype=np.float32)
    rows: List[np.ndarray] = []
    indices: List[int] = []
    for cls_idx in positive_idx.tolist():
        term_key = _norm_txt(str(idx_to_term.get(int(cls_idx), "")))
        if not term_key:
            continue
        if term_key in _INGESTED_ITEM_MASK_TERMS:
            if not creation_ok:
                continue
            rows.append(creation)
            indices.append(int(cls_idx))
            continue
        if _is_dataset_label_term(term_key):
            rows.append(full_frame)
            indices.append(int(cls_idx))
    if len(rows) <= 0:
        return np.zeros((0, int(height), int(width)), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(rows, axis=0).astype(np.float32, copy=False), np.asarray(indices, dtype=np.int64)


def _composite_non_dataset_label_stack(
    mask_stack: Any,
    mask_indices: Any,
    *,
    idx_to_term: Optional[Dict[int, str]] = None,
    height: int = 0,
    width: int = 0,
    fallback_mask: Optional[Any] = None,
    processing_device: Optional[Any] = None,
) -> np.ndarray:
    stack_arr = np.asarray(mask_stack, dtype=np.float32)
    idx_arr = np.asarray(mask_indices, dtype=np.int64).reshape(-1)
    if int(stack_arr.ndim) == 2:
        stack_arr = stack_arr[None, ...]
    if int(stack_arr.ndim) == 3 and int(stack_arr.shape[0]) > 0:
        if int(height) <= 0 or int(width) <= 0:
            height = int(stack_arr.shape[-2])
            width = int(stack_arr.shape[-1])
        pair_count = min(int(stack_arr.shape[0]), int(idx_arr.size))
        keep: List[int] = []
        for si in range(int(pair_count)):
            term_name = str(idx_to_term.get(int(idx_arr[int(si)]), "")) if isinstance(idx_to_term, dict) else ""
            if not _is_dataset_label_term(term_name):
                keep.append(int(si))
        if keep:
            return np.asarray(
                _composite_mask_stack(stack_arr[np.asarray(keep, dtype=np.int64)], processing_device=processing_device),
                dtype=np.float32,
            )
    if fallback_mask is not None and int(height) > 0 and int(width) > 0:
        return _normalize_stack_row(
            fallback_mask,
            height=int(height),
            width=int(width),
            processing_device=processing_device,
        )
    return np.zeros((int(max(0, height)), int(max(0, width))), dtype=np.float32)


def build_label_mask_stack(
    mixed_mask: Any,
    label_vec: Any,
    mask_stack_array: Optional[Any] = None,
    mask_stack_indices: Optional[Any] = None,
    *,
    idx_to_term: Optional[Dict[int, str]] = None,
    treat_mixed_mask_as_creation: bool = False,
    processing_device: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(label_vec, dtype=np.float32).reshape(-1)
    mixed_arr = np.asarray(mixed_mask, dtype=np.float32)
    h = int(mixed_arr.shape[-2]) if int(mixed_arr.ndim) >= 2 else 0
    w = int(mixed_arr.shape[-1]) if int(mixed_arr.ndim) >= 2 else 0
    if (int(h) <= 0 or int(w) <= 0) and mask_stack_array is not None:
        stack_ref = np.asarray(mask_stack_array, dtype=np.float32)
        if int(stack_ref.ndim) == 2:
            h = int(stack_ref.shape[-2])
            w = int(stack_ref.shape[-1])
        elif int(stack_ref.ndim) == 3 and int(stack_ref.shape[0]) > 0:
            h = int(stack_ref.shape[-2])
            w = int(stack_ref.shape[-1])
    creation_mask = (
        _resolve_creation_mask(
            mixed_mask,
            height=int(h),
            width=int(w),
            mask_stack_array=mask_stack_array,
            processing_device=processing_device,
        )
        if bool(treat_mixed_mask_as_creation)
        else np.zeros((int(max(0, h)), int(max(0, w))), dtype=np.float32)
    )
    special_stack, special_idx = _build_special_label_mask_stack(
        y,
        idx_to_term=idx_to_term,
        height=int(h),
        width=int(w),
        creation_mask=creation_mask,
    )
    parts: List[Tuple[Any, Any]] = []
    if mask_stack_array is not None and mask_stack_indices is not None:
        parts.append((mask_stack_array, mask_stack_indices))
    if int(special_stack.shape[0]) > 0 and int(special_idx.size) > 0:
        parts.append((special_stack, special_idx))
    if parts:
        out_stack, out_idx = combine_label_mask_stacks(
            y,
            *parts,
            height=int(h),
            width=int(w),
            processing_device=processing_device,
        )
        if int(out_stack.shape[0]) > 0 and int(out_idx.size) > 0:
            return out_stack, out_idx
    return np.zeros((0, int(max(0, h)), int(max(0, w))), dtype=np.float32), np.zeros((0,), dtype=np.int64)


def build_term_mask_stack_from_image(
    image: Any,
    label_vec: Any,
    idx_to_term: Optional[Dict[int, str]] = None,
    term_mask_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    stacks, indices = build_term_mask_stacks_from_images(
        images=np.asarray(image, dtype=np.float32)[None, ...],
        label_vecs=np.asarray(label_vec, dtype=np.float32).reshape(1, -1),
        idx_to_term=idx_to_term,
        term_mask_overrides_batch=[term_mask_overrides] if isinstance(term_mask_overrides, dict) else None,
    )
    return stacks[0], indices[0]


def augment_label_vec_with_detected_color_terms(
    *,
    image: Any,
    label_vec: Any,
    term_to_idx: Optional[Dict[str, int]] = None,
    valid_mask: Optional[Any] = None,
) -> np.ndarray:
    y = np.asarray(label_vec, dtype=np.float32).reshape(-1).copy()
    if int(y.size) <= 0 or not isinstance(term_to_idx, dict) or not term_to_idx:
        return y
    detected_terms = detect_semantic_color_terms(image=image, mask=valid_mask)
    for term in normalize_vocab_terms([str(x) for x in list(detected_terms)]):
        ti = int(term_to_idx.get(_norm_txt(term), -1))
        if 0 <= int(ti) < int(y.size):
            y[int(ti)] = 1.0
    return y


def _single_label_whole_image_mask(label_vec: Any, *, height: int, width: int) -> Optional[np.ndarray]:
    positive_idx = _positive_label_indices(label_vec)
    if int(positive_idx.size) != 1 or int(height) <= 0 or int(width) <= 0:
        return None
    return np.ones((int(height), int(width)), dtype=np.float32)


def assemble_semantic_mask_layers(
    *,
    image: Any,
    label_vec: Any,
    idx_to_term: Optional[Dict[int, str]] = None,
    term_to_idx: Optional[Dict[str, int]] = None,
    original_mixed_mask: Optional[Any] = None,
    original_parts: Optional[Sequence[Tuple[Any, Any]]] = None,
    deformation_term_masks: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    chw = _image_to_chw01(image)
    h = int(chw.shape[1])
    w = int(chw.shape[2])
    y = np.asarray(label_vec, dtype=np.float32).reshape(-1).copy()

    parts: List[Tuple[Any, Any]] = []
    for raw_stack, raw_idx in (list(original_parts) if original_parts is not None else []):
        parts.append((raw_stack, raw_idx))

    has_original_parts = any(
        raw_stack is not None and raw_idx is not None and int(np.asarray(raw_idx).size) > 0
        for raw_stack, raw_idx in parts
    )
    creation_seed_mask = np.asarray(original_mixed_mask, dtype=np.float32) if original_mixed_mask is not None else np.zeros((int(h), int(w)), dtype=np.float32)
    if float(np.max(np.asarray(creation_seed_mask, dtype=np.float32))) <= 1e-8 and has_original_parts:
        stacked_parts: List[np.ndarray] = []
        for raw_stack, _raw_idx in parts:
            part_arr = np.asarray(raw_stack, dtype=np.float32)
            if int(part_arr.ndim) == 2:
                part_arr = part_arr[None, ...]
            if int(part_arr.ndim) == 3 and int(part_arr.shape[0]) > 0:
                stacked_parts.append(
                    _normalize_stack_row_batch(
                        part_arr,
                        height=int(h),
                        width=int(w),
                    )
                )
        if stacked_parts:
            creation_seed_mask = _composite_mask_stack(np.concatenate(stacked_parts, axis=0))
    if original_mixed_mask is not None or float(np.max(np.asarray(creation_seed_mask, dtype=np.float32))) > 1e-8:
        original_stack, original_idx = build_label_mask_stack(
            mixed_mask=np.asarray(creation_seed_mask, dtype=np.float32),
            label_vec=y,
            idx_to_term=idx_to_term,
            treat_mixed_mask_as_creation=True,
        )
        if int(original_stack.shape[0]) > 0 and int(original_idx.size) > 0:
            parts.append((original_stack, original_idx))

    if isinstance(deformation_term_masks, dict) and deformation_term_masks and isinstance(term_to_idx, dict):
        for term in normalize_vocab_terms([str(x) for x in list(deformation_term_masks.keys())]):
            ti = int(term_to_idx.get(_norm_txt(term), -1))
            if 0 <= int(ti) < int(y.size):
                y[int(ti)] = 1.0
        deformation_stack, deformation_idx = term_mask_map_to_label_stack(
            deformation_term_masks,
            y,
            term_to_idx=term_to_idx,
            height=int(h),
            width=int(w),
        )
        if int(deformation_stack.shape[0]) > 0 and int(deformation_idx.size) > 0:
            parts.append((deformation_stack, deformation_idx))

    pre_detect_stack, pre_detect_idx = combine_label_mask_stacks(
        y,
        *parts,
        height=int(h),
        width=int(w),
        fallback_creation_mask=None,
    ) if len(parts) > 0 else (
        np.zeros((0, int(h), int(w)), dtype=np.float32),
        np.zeros((0,), dtype=np.int64),
    )
    detect_mask = _composite_non_dataset_label_stack(
        pre_detect_stack,
        pre_detect_idx,
        idx_to_term=idx_to_term,
        height=int(h),
        width=int(w),
        fallback_mask=original_mixed_mask,
    ) if int(pre_detect_stack.shape[0]) > 0 else (
        _normalize_stack_row(original_mixed_mask, height=int(h), width=int(w))
        if original_mixed_mask is not None
        else None
    )
    y = augment_label_vec_with_detected_color_terms(
        image=chw,
        label_vec=y,
        term_to_idx=term_to_idx,
        valid_mask=detect_mask,
    )
    detected_stack, detected_idx = build_term_mask_stack_from_image(
        image=chw,
        label_vec=y,
        idx_to_term=idx_to_term,
    ) if isinstance(idx_to_term, dict) and idx_to_term else (
        np.zeros((0, int(h), int(w)), dtype=np.float32),
        np.zeros((0,), dtype=np.int64),
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
    mixed_mask = _composite_non_dataset_label_stack(
        final_stack,
        final_idx,
        idx_to_term=idx_to_term,
        height=int(h),
        width=int(w),
        fallback_mask=original_mixed_mask,
    ) if int(final_stack.shape[0]) > 0 else (
        _normalize_stack_row(original_mixed_mask, height=int(h), width=int(w))
        if original_mixed_mask is not None
        else np.zeros((int(h), int(w)), dtype=np.float32)
    )
    return y, np.asarray(mixed_mask, dtype=np.float32), np.asarray(final_stack, dtype=np.float32), np.asarray(final_idx, dtype=np.int64)


def semantic_mask_stack_collate(batch: Sequence[Any]) -> Any:
    if len(batch) <= 0:
        return default_collate(batch)
    first = batch[0]
    if not isinstance(first, (tuple, list)) or int(len(first)) < 5:
        return default_collate(batch)
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    ms: List[torch.Tensor] = []
    mask_stacks: List[torch.Tensor] = []
    mask_indices: List[torch.Tensor] = []
    terms_rows: List[List[str]] = []
    for sample in batch:
        if not isinstance(sample, (tuple, list)) or int(len(sample)) < 5:
            raise RuntimeError("semantic_mask_stack_collate requires 5+-tuple samples.")
        xs.append(sample[0])
        ys.append(sample[1])
        ms.append(sample[2])
        stack_t = sample[3] if torch.is_tensor(sample[3]) else torch.as_tensor(sample[3])
        idx_t = sample[4] if torch.is_tensor(sample[4]) else torch.as_tensor(sample[4], dtype=torch.long)
        mask_stacks.append(stack_t)
        mask_indices.append(idx_t.to(dtype=torch.long))
        if int(len(sample)) >= 6 and isinstance(sample[5], (list, tuple)):
            terms_rows.append(normalize_vocab_terms([str(x) for x in list(sample[5])]))
        else:
            terms_rows.append([])
    return {
        "x": torch.stack(xs, dim=0),
        "y": torch.stack(ys, dim=0),
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


def _resolve_processing_device(processing_device: Optional[Any]) -> Optional[torch.device]:
    if processing_device is None:
        return None
    if isinstance(processing_device, torch.device):
        return processing_device
    text = str(processing_device).strip().lower()
    if not text or text == "none":
        return None
    if text == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        return torch.device(str(processing_device))
    except Exception:
        return None


def _normalize_attention_map_batch_torch(
    mask: Any,
    *,
    gamma: float = 1.0,
    blur_kernel: int = 0,
    device: Optional[Any] = None,
    apply_vmean_compression: bool = False,
    apply_binarize: bool = False,
) -> torch.Tensor:
    resolved = _resolve_processing_device(device) or torch.device("cpu")
    arr = mask if torch.is_tensor(mask) else torch.as_tensor(mask, dtype=torch.float32, device=resolved)
    if torch.is_tensor(arr):
        arr = arr.to(device=resolved, dtype=torch.float32)
    squeeze = False
    if int(arr.ndim) == 2:
        arr = arr.unsqueeze(0)
        squeeze = True
    if int(arr.ndim) != 3 or int(arr.numel()) <= 0:
        out = torch.zeros_like(arr, dtype=torch.float32)
        return out[0] if bool(squeeze) and int(out.ndim) == 3 and int(out.shape[0]) > 0 else out
    arr = torch.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = torch.clamp(arr, min=0.0)
    vmax = torch.amax(arr, dim=(1, 2), keepdim=True)
    valid = vmax > 1e-8
    arr = torch.where(valid, arr / torch.where(valid, vmax, torch.ones_like(vmax)), torch.zeros_like(arr))
    if bool(apply_vmean_compression):
        vmean = torch.mean(arr, dim=(1, 2), keepdim=True)
        denom = torch.clamp(vmean * 2.0, min=1.0)
        arr = torch.where(vmean > 1e-8, torch.clamp(arr / denom, 0.0, 1.0), arr)
    if bool(apply_binarize):
        arr = (arr > 0.5).to(dtype=torch.float32)
        return arr[0] if bool(squeeze) else arr
    gm = max(0.35, float(gamma))
    if abs(gm - 1.0) > 1e-6:
        arr = torch.pow(torch.clamp(arr, 0.0, 1.0), gm)
    kk = int(blur_kernel)
    if kk >= 3:
        kk = int(kk) | 1
        arr = F.avg_pool2d(arr[:, None, :, :], kernel_size=int(kk), stride=1, padding=int(kk // 2))[:, 0]
        vmax = torch.amax(arr, dim=(1, 2), keepdim=True)
        valid = vmax > 1e-8
        arr = torch.where(valid, arr / torch.where(valid, vmax, torch.ones_like(vmax)), torch.zeros_like(arr))
    arr = torch.clamp(arr, 0.0, 1.0).to(dtype=torch.float32)
    return arr[0] if bool(squeeze) else arr


def _blend_attention_maps_batch_torch(
    maps: Sequence[Any],
    *,
    weights: Optional[Sequence[float]] = None,
    gamma: float = 1.0,
    device: Optional[Any] = None,
    apply_vmean_compression: bool = False,
    apply_binarize: bool = False,
) -> torch.Tensor:
    resolved = _resolve_processing_device(device) or torch.device("cpu")
    valid: List[torch.Tensor] = []
    valid_weights: List[float] = []
    ref_shape: Optional[Tuple[int, int, int]] = None
    for i, m in enumerate(maps):
        arr = m if torch.is_tensor(m) else torch.as_tensor(m, dtype=torch.float32, device=resolved)
        arr = arr.to(device=resolved, dtype=torch.float32)
        if int(arr.ndim) == 2:
            arr = arr.unsqueeze(0)
        if int(arr.ndim) != 3 or int(arr.numel()) <= 0:
            continue
        if ref_shape is None:
            ref_shape = (int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2]))
        w = 1.0 if weights is None or i >= int(len(weights)) else float(weights[i])
        if w <= 0.0:
            continue
        norm = _normalize_attention_map_batch_torch(arr, gamma=1.0, blur_kernel=0, device=resolved, apply_vmean_compression=bool(apply_vmean_compression))
        if float(torch.amax(norm).detach().cpu().item()) <= 1e-8:
            continue
        valid.append(norm)
        valid_weights.append(float(w))
    if len(valid) <= 0:
        if ref_shape is None:
            return torch.zeros((0, 0, 0), dtype=torch.float32, device=resolved)
        return torch.zeros(ref_shape, dtype=torch.float32, device=resolved)
    weights_t = torch.as_tensor(valid_weights, dtype=torch.float32, device=resolved).view(-1, 1, 1, 1)
    stack = torch.stack(valid, dim=0).to(dtype=torch.float32)
    acc = torch.sum(weights_t * stack, dim=0)
    wsum = float(torch.sum(torch.as_tensor(valid_weights, dtype=torch.float32)).item())
    if wsum > 1e-8:
        acc = acc / float(wsum)
    return _normalize_attention_map_batch_torch(acc, gamma=float(gamma), blur_kernel=5, device=resolved, apply_vmean_compression=bool(apply_vmean_compression), apply_binarize=bool(apply_binarize))


def _semantic_color_score_maps_batch_torch(
    chw_batch: Any,
    *,
    device: Optional[Any] = None,
) -> Dict[str, torch.Tensor]:
    resolved = _resolve_processing_device(device) or torch.device("cpu")
    arr = torch.as_tensor(_image_batch_to_bchw01(chw_batch), dtype=torch.float32, device=resolved)
    arr = torch.clamp(arr, 0.0, 1.0)
    if int(arr.ndim) != 4 or int(arr.shape[1]) < 3:
        return {}
    r = arr[:, 0]
    g = arr[:, 1]
    b = arr[:, 2]
    vmax = torch.maximum(torch.maximum(r, g), b)
    vmin = torch.minimum(torch.minimum(r, g), b)
    sat = torch.clamp(vmax - vmin, 0.0, 1.0)

    def _norm01_batch(x: torch.Tensor) -> torch.Tensor:
        hi = torch.amax(x, dim=(1, 2), keepdim=True)
        valid = hi > 1e-8
        return torch.where(valid, torch.clamp(x / torch.where(valid, hi, torch.ones_like(hi)), 0.0, 1.0), torch.zeros_like(x))

    red = _norm01_batch(torch.clamp(r - torch.maximum(g, b), 0.0, 1.0) * torch.clamp(sat - 0.05, 0.0, 1.0))
    green = _norm01_batch(torch.clamp(g - torch.maximum(r, b), 0.0, 1.0) * torch.clamp(sat - 0.05, 0.0, 1.0))
    blue = _norm01_batch(torch.clamp(b - torch.maximum(r, g), 0.0, 1.0) * torch.clamp(sat - 0.05, 0.0, 1.0))
    yellow = _norm01_batch(torch.clamp(torch.minimum(r, g) - b, 0.0, 1.0) * torch.clamp(sat - 0.05, 0.0, 1.0))
    cyan = _norm01_batch(torch.clamp(torch.minimum(g, b) - r, 0.0, 1.0) * torch.clamp(sat - 0.05, 0.0, 1.0))
    magenta = _norm01_batch(torch.clamp(torch.minimum(r, b) - g, 0.0, 1.0) * torch.clamp(sat - 0.05, 0.0, 1.0))
    brown = _norm01_batch(
        torch.clamp(r - g, 0.0, 1.0)
        * torch.clamp(g - b, 0.0, 1.0)
        * torch.clamp(vmax, 0.15, 0.75)
        * torch.clamp(0.85 - vmax, 0.0, 1.0)
    )
    orange = _norm01_batch(
        torch.clamp(r - b - 0.05, 0.0, 1.0)
        * torch.clamp(r - g - 0.08, 0.0, 1.0)
        * torch.clamp(g - b - 0.02, 0.0, 1.0)
        * torch.clamp(sat - 0.10, 0.0, 1.0)
    )
    black = _norm01_batch(torch.clamp(0.22 - vmax, 0.0, 1.0))
    white = _norm01_batch(torch.clamp(vmin - 0.78, 0.0, 1.0) * torch.clamp(0.20 - sat, 0.0, 1.0))
    gray = _norm01_batch(torch.clamp(0.18 - sat, 0.0, 1.0) * torch.clamp(1.0 - torch.abs(vmax - 0.5) * 2.2, 0.0, 1.0))
    luma = torch.mean(arr[:, :3], dim=1)
    gx = torch.zeros_like(luma)
    gy = torch.zeros_like(luma)
    gx[:, :, 1:-1] = luma[:, :, 2:] - luma[:, :, :-2]
    gy[:, 1:-1, :] = luma[:, 2:, :] - luma[:, :-2, :]
    edge = _norm01_batch(torch.sqrt((gx * gx) + (gy * gy)))
    return {
        "red": red,
        "orange": orange,
        "green": green,
        "blue": blue,
        "yellow": yellow,
        "cyan": cyan,
        "magenta": magenta,
        "brown": brown,
        "black": black,
        "white": white,
        "gray": gray,
        "edge": edge,
    }


def _semantic_color_mask_from_score_batch_torch(
    scores: Any,
    *,
    device: Optional[Any] = None,
) -> torch.Tensor:
    resolved = _resolve_processing_device(device) or torch.device("cpu")
    arr = scores if torch.is_tensor(scores) else torch.as_tensor(scores, dtype=torch.float32, device=resolved)
    arr = arr.to(device=resolved, dtype=torch.float32)
    squeeze = False
    if int(arr.ndim) == 2:
        arr = arr.unsqueeze(0)
        squeeze = True
    n = int(arr.shape[0])
    if int(n) <= 0 or int(arr.numel()) <= 0:
        return arr[0] if bool(squeeze) else arr
    flat = arr.view(int(n), -1)
    try:
        thr = torch.quantile(flat, q=0.82, dim=1)
    except Exception:
        thr = torch.as_tensor(np.percentile(flat.detach().cpu().numpy(), 82, axis=1), dtype=torch.float32, device=resolved)
    thr = torch.clamp(thr * 0.75, min=0.20)
    exact = (arr >= thr[:, None, None]).to(dtype=torch.float32) * arr
    out = _normalize_attention_map_batch_torch(exact, gamma=0.78, blur_kernel=1, device=resolved)
    return out[0] if bool(squeeze) else out


def _avg_pool2d_batch(batch_hw: np.ndarray, kernel_size: int) -> np.ndarray:
    kk = max(1, int(kernel_size))
    if kk <= 1:
        return np.asarray(batch_hw, dtype=np.float32)
    pooled = F.avg_pool2d(
        torch.from_numpy(np.asarray(batch_hw, dtype=np.float32)[:, None, :, :]),
        kernel_size=int(kk),
        stride=1,
        padding=int(kk // 2),
    )
    return np.asarray(pooled[:, 0].cpu().numpy(), dtype=np.float32)


def _normalize_attention_map_batch(mask: Any, gamma: float = 1.0, blur_kernel: int = 0, apply_vmean_compression: bool = False, apply_binarize: bool = False) -> np.ndarray:
    arr = np.asarray(mask, dtype=np.float32)
    squeeze = False
    if int(arr.ndim) == 2:
        arr = arr[None, ...]
        squeeze = True
    if int(arr.ndim) != 3 or int(arr.size) <= 0:
        out = np.zeros_like(np.asarray(arr, dtype=np.float32), dtype=np.float32)
        return out[0] if bool(squeeze) and int(out.ndim) == 3 and int(out.shape[0]) > 0 else out
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    arr = np.maximum(arr, 0.0).astype(np.float32, copy=False)
    vmax = np.max(arr, axis=(1, 2), keepdims=True) if int(arr.size) > 0 else np.zeros((int(arr.shape[0]), 1, 1), dtype=np.float32)
    valid = vmax > 1e-8
    arr = np.where(valid, arr / np.where(valid, vmax, 1.0), 0.0).astype(np.float32, copy=False)
    if bool(apply_vmean_compression):
        vmean = np.mean(arr, axis=(1, 2), keepdims=True) if int(arr.size) > 0 else np.zeros((int(arr.shape[0]), 1, 1), dtype=np.float32)
        arr = np.where(vmean > 1e-8, np.clip(arr / np.maximum(vmean * 2.0, 1.0), 0.0, 1.0), arr).astype(np.float32, copy=False)
    if bool(apply_binarize):
        arr = (arr > 0.5).astype(np.float32, copy=False)
        return np.asarray(arr[0], dtype=np.float32) if bool(squeeze) else arr
    gm = max(0.35, float(gamma))
    if abs(gm - 1.0) > 1e-6:
        arr = np.power(np.clip(arr, 0.0, 1.0), gm).astype(np.float32, copy=False)
    kk = int(blur_kernel)
    if kk >= 3:
        kk = int(kk) | 1
        arr = _avg_pool2d_batch(arr, kernel_size=int(kk))
        vmax = np.max(arr, axis=(1, 2), keepdims=True) if int(arr.size) > 0 else np.zeros((int(arr.shape[0]), 1, 1), dtype=np.float32)
        valid = vmax > 1e-8
        arr = np.where(valid, arr / np.where(valid, vmax, 1.0), 0.0).astype(np.float32, copy=False)
    arr = np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)
    return np.asarray(arr[0], dtype=np.float32) if bool(squeeze) else arr


def _blend_attention_maps_batch(maps: Sequence[Any], weights: Optional[Sequence[float]] = None, gamma: float = 1.0, apply_vmean_compression: bool = False, apply_binarize: bool = False) -> np.ndarray:
    valid: List[np.ndarray] = []
    valid_weights: List[float] = []
    ref_shape: Optional[Tuple[int, int, int]] = None
    for i, m in enumerate(maps):
        arr = np.asarray(m, dtype=np.float32)
        if int(arr.ndim) == 2:
            arr = arr[None, ...]
        if int(arr.ndim) != 3 or int(arr.size) <= 0:
            continue
        if ref_shape is None:
            ref_shape = (int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2]))
        w = 1.0 if weights is None or i >= int(len(weights)) else float(weights[i])
        if w <= 0.0:
            continue
        norm = _normalize_attention_map_batch(arr, gamma=1.0, blur_kernel=0, apply_vmean_compression=bool(apply_vmean_compression))
        if float(np.max(norm)) <= 1e-8:
            continue
        valid.append(np.asarray(norm, dtype=np.float32))
        valid_weights.append(float(w))
    if len(valid) <= 0:
        if ref_shape is None:
            return np.zeros((0, 0, 0), dtype=np.float32)
        return np.zeros(ref_shape, dtype=np.float32)
    weights_np = np.asarray(valid_weights, dtype=np.float32).reshape(-1, 1, 1, 1)
    stack = np.stack(valid, axis=0).astype(np.float32, copy=False)
    acc = np.sum(weights_np * stack, axis=0).astype(np.float32, copy=False)
    wsum = float(np.sum(np.asarray(valid_weights, dtype=np.float32)))
    if wsum > 1e-8:
        acc = acc / float(wsum)
    return _normalize_attention_map_batch(acc, gamma=float(gamma), blur_kernel=5, apply_vmean_compression=bool(apply_vmean_compression), apply_binarize=bool(apply_binarize))


def _semantic_color_score_maps(chw: np.ndarray) -> Dict[str, np.ndarray]:
    maps = _semantic_color_score_maps_batch(np.asarray(chw, dtype=np.float32)[None, ...])
    return {str(k): np.asarray(v[0], dtype=np.float32) for k, v in maps.items()}


def _semantic_color_score_maps_batch(chw_batch: Any) -> Dict[str, np.ndarray]:
    arr = np.clip(_image_batch_to_bchw01(chw_batch), 0.0, 1.0).astype(np.float32, copy=False)
    if int(arr.ndim) != 4 or int(arr.shape[1]) < 3:
        return {}
    r = np.asarray(arr[:, 0], dtype=np.float32)
    g = np.asarray(arr[:, 1], dtype=np.float32)
    b = np.asarray(arr[:, 2], dtype=np.float32)
    vmax = np.maximum.reduce([r, g, b]).astype(np.float32, copy=False)
    vmin = np.minimum.reduce([r, g, b]).astype(np.float32, copy=False)
    sat = np.clip(vmax - vmin, 0.0, 1.0).astype(np.float32, copy=False)

    def _norm01_batch(x: np.ndarray) -> np.ndarray:
        xx = np.asarray(x, dtype=np.float32)
        hi = np.max(xx, axis=(1, 2), keepdims=True) if int(xx.size) > 0 else np.zeros((int(xx.shape[0]), 1, 1), dtype=np.float32)
        valid = hi > 1e-8
        return np.where(valid, np.clip(xx / np.where(valid, hi, 1.0), 0.0, 1.0), 0.0).astype(np.float32, copy=False)

    red = _norm01_batch(np.clip(r - np.maximum(g, b), 0.0, 1.0) * np.clip(sat - 0.05, 0.0, 1.0))
    green = _norm01_batch(np.clip(g - np.maximum(r, b), 0.0, 1.0) * np.clip(sat - 0.05, 0.0, 1.0))
    blue = _norm01_batch(np.clip(b - np.maximum(r, g), 0.0, 1.0) * np.clip(sat - 0.05, 0.0, 1.0))
    yellow = _norm01_batch(np.clip(np.minimum(r, g) - b, 0.0, 1.0) * np.clip(sat - 0.05, 0.0, 1.0))
    cyan = _norm01_batch(np.clip(np.minimum(g, b) - r, 0.0, 1.0) * np.clip(sat - 0.05, 0.0, 1.0))
    magenta = _norm01_batch(np.clip(np.minimum(r, b) - g, 0.0, 1.0) * np.clip(sat - 0.05, 0.0, 1.0))
    brown = _norm01_batch(
        np.clip(r - g, 0.0, 1.0)
        * np.clip(g - b, 0.0, 1.0)
        * np.clip(vmax, 0.15, 0.75)
        * np.clip(0.85 - vmax, 0.0, 1.0)
    )
    orange = _norm01_batch(
        np.clip(r - b - 0.05, 0.0, 1.0)
        * np.clip(r - g - 0.08, 0.0, 1.0)
        * np.clip(g - b - 0.02, 0.0, 1.0)
        * np.clip(sat - 0.10, 0.0, 1.0)
    )
    black = _norm01_batch(np.clip(0.22 - vmax, 0.0, 1.0))
    white = _norm01_batch(np.clip(vmin - 0.78, 0.0, 1.0) * np.clip(0.20 - sat, 0.0, 1.0))
    gray = _norm01_batch(np.clip(0.18 - sat, 0.0, 1.0) * np.clip(1.0 - np.abs(vmax - 0.5) * 2.2, 0.0, 1.0))
    # Fast Sobel-like edge detector across batch
    luma = np.mean(arr[:, :3], axis=1)  # (B, H, W)
    gx = np.zeros_like(luma)
    gy = np.zeros_like(luma)
    gx[:, :, 1:-1] = luma[:, :, 2:] - luma[:, :, :-2]
    gy[:, 1:-1, :] = luma[:, 2:, :] - luma[:, :-2, :]
    edge = _norm01_batch(np.sqrt(gx * gx + gy * gy).astype(np.float32, copy=False))
    return {
        "red": red,
        "orange": orange,
        "green": green,
        "blue": blue,
        "yellow": yellow,
        "cyan": cyan,
        "magenta": magenta,
        "brown": brown,
        "black": black,
        "white": white,
        "gray": gray,
        "edge": edge,
    }


def detect_semantic_color_terms(
    image: Any,
    mask: Optional[Any] = None,
    coverage_threshold: float = 0.06,
    dominance_threshold: float = 0.22,
) -> List[str]:
    try:
        chw = _image_to_chw01(image)
    except Exception:
        return []
    maps = _semantic_color_score_maps(chw)
    if len(maps) <= 0:
        return []
    valid = None
    if mask is not None:
        try:
            valid = _normalize_mask_array(mask, height=int(chw.shape[1]), width=int(chw.shape[2]))
        except Exception:
            valid = None
    if valid is None or float(np.mean(valid)) <= 0.01:
        valid = np.ones((int(chw.shape[1]), int(chw.shape[2])), dtype=np.float32)
    valid_mask = np.asarray(valid, dtype=np.float32) >= 0.15
    valid_count = int(np.sum(valid_mask))
    if valid_count <= 0:
        valid_mask = np.ones((int(chw.shape[1]), int(chw.shape[2])), dtype=bool)
        valid_count = int(np.sum(valid_mask))

    out: List[str] = []
    for term in ["red", "orange", "green", "blue", "yellow", "cyan", "magenta", "brown", "black", "white", "gray", "edge"]:
        score = np.asarray(maps.get(term), dtype=np.float32)
        if int(score.size) <= 0:
            continue
        frac = float(np.mean(score[valid_mask] >= float(dominance_threshold))) if int(valid_count) > 0 else 0.0
        mean_score = float(np.mean(score[valid_mask])) if int(valid_count) > 0 else 0.0
        if frac >= float(coverage_threshold) or mean_score >= float(max(0.10, dominance_threshold * 0.72)):
            out.append(str(term))
    return normalize_vocab_terms(out)


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


def _semantic_color_mask_from_score(color_score: Any) -> np.ndarray:
    score = np.asarray(color_score, dtype=np.float32)
    if int(score.ndim) != 2 or int(score.size) <= 0:
        return np.zeros((0, 0), dtype=np.float32)
    thr = max(0.20, float(np.percentile(score.reshape(-1), 82)) * 0.75)
    exact = (score >= float(thr)).astype(np.float32, copy=False) * score
    return _normalize_attention_map(exact, gamma=0.78, blur_kernel=1)


def _semantic_color_mask_from_score_batch(scores: Any) -> np.ndarray:
    """Batch version of _semantic_color_mask_from_score for [N, H, W] arrays.

    Computes the 82nd-percentile threshold per image simultaneously using
    np.percentile over the spatial axes, then thresholds and normalises all
    images in one pass — no Python loop over N.
    """
    arr = np.asarray(scores, dtype=np.float32)
    squeeze = False
    if arr.ndim == 2:
        arr = arr[None]
        squeeze = True
    n = int(arr.shape[0])
    if n == 0 or int(arr.size) == 0:
        return arr[0] if squeeze else arr
    flat = arr.reshape(n, -1)  # [N, H*W]
    # percentile per image — [N]
    thr = np.percentile(flat, 82, axis=1).astype(np.float32) * 0.75
    thr = np.maximum(0.20, thr)  # [N]
    # broadcast threshold over [N, H, W]
    exact = (arr >= thr[:, None, None]) * arr  # [N, H, W]
    out = _normalize_attention_map_batch(exact, gamma=0.78, blur_kernel=1)
    return np.asarray(out[0], dtype=np.float32) if squeeze else out


def infer_semantic_support_masks(
    images: Any,
    terms_batch: Optional[Sequence[Optional[Sequence[str]]]] = None,
    processing_device: Optional[Any] = None,
) -> np.ndarray:
    bchw = _image_batch_to_bchw01(images)
    bsz = int(bchw.shape[0]) if int(bchw.ndim) == 4 else 0
    h = int(bchw.shape[2]) if int(bchw.ndim) == 4 else 0
    w = int(bchw.shape[3]) if int(bchw.ndim) == 4 else 0
    if int(bsz) <= 0 or int(h) <= 0 or int(w) <= 0:
        return np.zeros((int(max(0, bsz)), int(max(0, h)), int(max(0, w))), dtype=np.float32)
    term_rows = _normalize_semantic_terms_batch(terms_batch, batch_size=int(bsz))
    resolved = _resolve_processing_device(processing_device)
    if resolved is not None:
        color_maps_t = _semantic_color_score_maps_batch_torch(bchw, device=resolved)
        score_acc_t = torch.zeros((int(bsz), int(h), int(w)), dtype=torch.float32, device=resolved)
        _color_term_set = set(_SEMANTIC_COLOR_TERMS)
        for term_key, term_scores_t in color_maps_t.items():
            if term_key not in _color_term_set:
                continue
            wants = torch.as_tensor(
                [term_key in {_norm_txt(t) for t in (row or []) if str(t).strip()} for row in term_rows],
                dtype=torch.bool,
                device=resolved,
            )
            if not bool(torch.any(wants).item()):
                continue
            score_acc_t[wants] = torch.maximum(score_acc_t[wants], term_scores_t[wants])
        has_signal = torch.amax(score_acc_t.view(int(bsz), -1), dim=1) > 1e-8
        out_t = torch.zeros((int(bsz), int(h), int(w)), dtype=torch.float32, device=resolved)
        if bool(torch.any(has_signal).item()):
            out_t[has_signal] = _semantic_color_mask_from_score_batch_torch(score_acc_t[has_signal], device=resolved)
        return np.asarray(out_t.detach().cpu().numpy(), dtype=np.float32)
    color_maps = _semantic_color_score_maps_batch(bchw)
    # Accumulate max color score per image across relevant terms — no per-image loop.
    # For each color term present in color_maps, find which images request it, then
    # max-accumulate that term's [N, H, W] score map into those image slots at once.
    score_acc = np.zeros((int(bsz), int(h), int(w)), dtype=np.float32)
    _color_term_set = set(_SEMANTIC_COLOR_TERMS)
    for term_key, term_scores in color_maps.items():
        if term_key not in _color_term_set:
            continue
        # which images mention this color term — build boolean mask [N]
        wants = np.array(
            [term_key in {_norm_txt(t) for t in (row or []) if str(t).strip()} for row in term_rows],
            dtype=bool,
        )
        if not np.any(wants):
            continue
        # max-accumulate across the entire [N, H, W] slice in one numpy op
        score_acc[wants] = np.maximum(score_acc[wants], np.asarray(term_scores, dtype=np.float32)[wants])
    # Apply batch threshold+normalise only to images that have any signal
    has_signal = np.max(score_acc.reshape(int(bsz), -1), axis=1) > 1e-8  # [N]
    out = np.zeros((int(bsz), int(h), int(w)), dtype=np.float32)
    if np.any(has_signal):
        out[has_signal] = _semantic_color_mask_from_score_batch(score_acc[has_signal])
    return out.astype(np.float32, copy=False)


def infer_semantic_support_mask(image: Any, terms: Optional[Sequence[str]] = None) -> np.ndarray:
    return np.asarray(
        infer_semantic_support_masks(images=np.asarray(image, dtype=np.float32)[None, ...], terms_batch=[terms or []])[0],
        dtype=np.float32,
    )


def build_term_mask_stacks_from_images(
    images: Any,
    label_vecs: Any,
    idx_to_term: Optional[Dict[int, str]] = None,
    term_mask_overrides_batch: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    processing_device: Optional[Any] = None,
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

    resolved = _resolve_processing_device(processing_device)
    color_maps_t = _semantic_color_score_maps_batch_torch(bchw, device=resolved) if resolved is not None else {}
    color_maps = _semantic_color_score_maps_batch(bchw) if resolved is None else {}
    override_rows = list(term_mask_overrides_batch) if term_mask_overrides_batch is not None else [None for _ in range(int(bsz))]
    if int(len(override_rows)) < int(bsz):
        override_rows.extend([None for _ in range(int(bsz) - int(len(override_rows)))])

    masks_per_image: List[List[np.ndarray]] = [[] for _ in range(int(bsz))]
    idx_per_image: List[List[int]] = [[] for _ in range(int(bsz))]

    # Pre-build cls→term_key map once (avoids re-sub inside every loop iteration)
    cls_to_term_key: Dict[int, str] = {}
    if isinstance(idx_to_term, dict):
        for ci, tname in idx_to_term.items():
            cls_to_term_key[int(ci)] = _norm_txt(str(tname))

    # active[n, c] = True when image n has class c positive — computed for the whole batch at once
    active = y >= 0.5  # [N, C]
    n_classes = int(y.shape[1]) if int(y.ndim) >= 2 else 0

    # ---- Color-map path: loop over terms (~12), not over images (N can be large) ----
    for term_key, term_scores in (color_maps_t.items() if resolved is not None else color_maps.items()):
        # Which class indices map to this color term?
        cls_for_term = [ci for ci, tk in cls_to_term_key.items() if tk == term_key and int(ci) < int(n_classes)]
        if not cls_for_term:
            continue
        term_scores_arr = term_scores if resolved is not None else np.asarray(term_scores, dtype=np.float32)  # [N, H, W]
        for cls_idx in cls_for_term:
            # Which images have this class active? — vectorised boolean index
            active_imgs = np.where(active[:, int(cls_idx)])[0]  # [K]
            if len(active_imgs) == 0:
                continue
            # Compute masks for ALL active images in one batch call, no Python loop
            if resolved is not None:
                active_imgs_t = torch.as_tensor(active_imgs.tolist(), dtype=torch.long, device=resolved)
                batch_masks_t = _semantic_color_mask_from_score_batch_torch(term_scores_arr[active_imgs_t], device=resolved)  # [K, H, W]
                vmax_t = torch.amax(batch_masks_t.view(len(active_imgs), -1), dim=1)
                keep = torch.nonzero(vmax_t > 1e-8, as_tuple=False).view(-1).detach().cpu().numpy().tolist()
                for li in keep:
                    gi = int(active_imgs[int(li)])
                    masks_per_image[gi].append(np.asarray(batch_masks_t[int(li)].detach().cpu().numpy(), dtype=np.float32))
                    idx_per_image[gi].append(int(cls_idx))
            else:
                batch_masks = _semantic_color_mask_from_score_batch(term_scores_arr[active_imgs])  # [K, H, W]
                vmax = np.max(batch_masks.reshape(len(active_imgs), -1), axis=1)  # [K]
                for li in np.where(vmax > 1e-8)[0]:
                    gi = int(active_imgs[li])
                    masks_per_image[gi].append(batch_masks[int(li)])
                    idx_per_image[gi].append(int(cls_idx))

    # ---- Override path: rare per-image user-supplied masks, stays per-image ----
    has_overrides = any(isinstance(r, dict) and r for r in override_rows)
    if has_overrides:
        for row_idx in range(int(bsz)):
            if not isinstance(override_rows[int(row_idx)], dict) or not override_rows[int(row_idx)]:
                continue
            override_lut = {
                _norm_txt(str(k)): np.asarray(v, dtype=np.float32)
                for k, v in override_rows[int(row_idx)].items()
                if str(k).strip() and v is not None
            }
            positive_idx = np.where(active[int(row_idx)])[0]
            for cls_idx in positive_idx.tolist():
                term_key = cls_to_term_key.get(int(cls_idx), "")
                if term_key not in override_lut:
                    continue
                # skip if already contributed by color_map path above
                if any(int(ei) == int(cls_idx) for ei in idx_per_image[int(row_idx)]):
                    continue
                mask = _normalize_attention_map(override_lut[term_key], gamma=0.85, blur_kernel=1)
                if float(np.max(mask)) <= 1e-8:
                    continue
                masks_per_image[int(row_idx)].append(np.asarray(mask, dtype=np.float32))
                idx_per_image[int(row_idx)].append(int(cls_idx))

    out_stack: List[np.ndarray] = []
    out_idx: List[np.ndarray] = []
    for row_idx in range(int(bsz)):
        if len(masks_per_image[int(row_idx)]) <= 0:
            out_stack.append(np.zeros((0, int(h), int(w)), dtype=np.float32))
            out_idx.append(np.zeros((0,), dtype=np.int64))
            continue
        out_stack.append(np.stack(masks_per_image[int(row_idx)], axis=0).astype(np.float32, copy=False))
        out_idx.append(np.asarray(idx_per_image[int(row_idx)], dtype=np.int64))
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
    Callers normalize via _normalize_attention_map / _normalize_attention_map_batch.

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


def _cache_encode_image_u8(img: np.ndarray) -> np.ndarray:
    arr = np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0)
    return np.round(arr * 255.0).astype(np.uint8, copy=False)


def _cache_decode_image_u8(img: np.ndarray) -> np.ndarray:
    return (np.asarray(img, dtype=np.float32) / 255.0).astype(np.float32, copy=False)


def _cache_encode_mask_f16(mask: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0).astype(np.float16, copy=False)


def _cache_decode_mask_f16(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask, dtype=np.float32)


def _cache_encode_target_f16(target: np.ndarray) -> np.ndarray:
    return np.asarray(target, dtype=np.float16)


def _cache_decode_target_f16(target: np.ndarray) -> np.ndarray:
    return np.asarray(target, dtype=np.float32)


def _estimate_target_positive_count(targets: Sequence[np.ndarray], max_rows: int = 16) -> float:
    if len(targets) <= 0:
        return 1.0
    counts: List[int] = []
    for row in list(targets)[: max(1, int(max_rows))]:
        arr = np.asarray(row, dtype=np.float32).reshape(-1)
        counts.append(max(1, int(np.count_nonzero(arr >= 0.5))))
    if len(counts) <= 0:
        return 1.0
    return float(max(1.0, float(np.mean(np.asarray(counts, dtype=np.float32)))))


def _update_signature_with_array(hasher: "hashlib._Hash", arr: np.ndarray) -> None:
    arr_np = np.asarray(arr, dtype=np.float32)
    hasher.update(np.asarray(arr_np.shape, dtype=np.int64).tobytes())
    hasher.update(_cache_encode_image_u8(arr_np).tobytes())


def _bootstrap_dataset_signature(
    images: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    total_rows: int,
    seed: int,
    augment: bool,
    augment_apply_terms: bool,
    return_masks: bool,
    return_mask_stack: bool,
    dataset_name: str,
) -> str:
    hasher = hashlib.sha256()
    payload = {
        "dataset_name": str(dataset_name),
        "base_rows": int(min(len(images), len(targets))),
        "total_rows": int(total_rows),
        "seed": int(seed),
        "augment": bool(augment),
        "augment_apply_terms": bool(augment_apply_terms),
        "return_masks": bool(return_masks),
        "return_mask_stack": bool(return_mask_stack),
    }
    hasher.update(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8"))
    n = min(int(len(images)), int(len(targets)))
    for i in range(int(n)):
        _update_signature_with_array(hasher, np.asarray(images[int(i)], dtype=np.float32))
        tgt = np.asarray(targets[int(i)], dtype=np.float32).reshape(-1)
        hasher.update(np.asarray(tgt.shape, dtype=np.int64).tobytes())
        hasher.update(_cache_encode_target_f16(tgt).tobytes())
    return str(hasher.hexdigest())


class BootstrapDynamicDataset(Dataset):
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
        return_masks: bool = False,
        return_mask_stack: bool = False,
        dataset_name: str = "bootstrap_dynamic",
        persistent_cache_dir: str = "",
        persistent_cache_max_rows: int = 0,
        persistent_cache_rebuild: bool = False,
        persistent_cache_max_bytes: int = 0,
        persistent_cache_overflow_strategy: str = "loop",
        persistent_cache_slot_lifespan: int = 0,
        base_masks: Optional[Sequence[np.ndarray]] = None,
    ):
        n = min(int(len(images)), int(len(targets)))
        if n <= 0:
            raise RuntimeError(f"{str(dataset_name)} requires non-empty images/targets.")
        self.images = [np.asarray(images[i], dtype=np.float32) for i in range(int(n))]
        self.targets = [np.asarray(targets[i], dtype=np.float32).reshape(-1) for i in range(int(n))]
        self.base_rows = int(n)
        self.total_rows = max(int(self.base_rows), int(total_rows))
        self.seed = int(seed)
        self.augment = bool(augment)
        self.return_masks = bool(return_masks)
        self.return_mask_stack = bool(return_mask_stack)
        self.use_semantic_mask_stack_collate = bool(self.return_mask_stack)
        y_sizes = sorted({int(v.size) for v in self.targets})
        if len(y_sizes) != 1:
            raise RuntimeError(f"{str(dataset_name)} target width mismatch: {y_sizes}")
        self.target_dim = int(y_sizes[0])
        if int(expected_target_dim) > 0 and int(self.target_dim) != int(expected_target_dim):
            raise RuntimeError(
                f"{str(dataset_name)} target width mismatch: got={int(self.target_dim)} expected={int(expected_target_dim)}"
            )
        self.semantic_term_to_idx = {
            re.sub(r"\s+", " ", str(k)).strip().lower(): int(v)
            for k, v in (semantic_term_to_idx.items() if isinstance(semantic_term_to_idx, dict) else [])
            if str(k).strip()
        }
        self.idx_to_term = {
            int(v): re.sub(r"\s+", " ", str(k)).strip()
            for k, v in self.semantic_term_to_idx.items()
            if 0 <= int(v) < int(self.target_dim)
        }
        self.augment_apply_terms = bool(augment_apply_terms)
        self.base_masks: List[Optional[np.ndarray]] = [None] * int(self.base_rows)
        if base_masks is not None:
            for _bm_i in range(min(int(self.base_rows), int(len(base_masks)))):
                if base_masks[_bm_i] is not None:
                    self.base_masks[_bm_i] = np.asarray(base_masks[_bm_i], dtype=np.float32)
        self.base_mask_stacks: List[Optional[np.ndarray]] = [None] * int(self.base_rows)
        self.base_mask_stack_indices: List[Optional[np.ndarray]] = [None] * int(self.base_rows)
        self.dataset_name = str(dataset_name)
        self.persistent_cache_dir = str(persistent_cache_dir).strip()
        self.persistent_cache_max_rows = int(persistent_cache_max_rows)
        self.persistent_cache_rebuild = bool(persistent_cache_rebuild)
        self.persistent_cache_max_bytes = max(0, int(persistent_cache_max_bytes))
        strategy_key = re.sub(r"\s+", "_", str(persistent_cache_overflow_strategy)).strip().lower()
        self.persistent_cache_overflow_strategy = strategy_key if strategy_key in ("loop", "evict") else "loop"
        self.persistent_cache_slot_lifespan = max(0, int(persistent_cache_slot_lifespan))
        self.persistent_cache_signature = _bootstrap_dataset_signature(
            images=self.images,
            targets=self.targets,
            total_rows=int(self.total_rows),
            seed=int(self.seed),
            augment=bool(self.augment),
            augment_apply_terms=bool(self.augment_apply_terms),
            return_masks=bool(self.return_masks),
            return_mask_stack=bool(self.return_mask_stack),
            dataset_name=str(self.dataset_name),
        )
        self.cached_rows = 0
        # In-memory eager cache: populated at construction for non-augmented datasets
        # so that __getitem__ is a pure list lookup with zero recomputation per epoch.
        self._eager_row_cache: List[Optional[tuple]] = [None] * int(self.total_rows)
        self._eager_row_cache_ready = False
        self._cache_manifest_path = ""
        self._cache_images_path = ""
        self._cache_targets_path = ""
        self._cache_masks_path = ""
        self._cache_mask_stacks_path = ""
        self._cache_mask_indices_path = ""
        self._cache_mask_offsets_path = ""
        self._cache_images_mm = None
        self._cache_targets_mm = None
        self._cache_masks_mm = None
        self._cache_mask_stacks_mm = None
        self._cache_mask_indices_mm = None
        self._cache_mask_offsets_mm = None
        self.persistent_cache_info: Dict[str, Any] = {
            "enabled": False,
            "cache_dir": str(self.persistent_cache_dir),
            "signature": str(self.persistent_cache_signature),
            "desired_rows": 0,
            "cached_rows": 0,
            "cache_hit": False,
            "rebuilt": False,
            "generated_rows": 0,
            "max_bytes": int(self.persistent_cache_max_bytes),
            "overflow_strategy": str(self.persistent_cache_overflow_strategy),
            "estimated_bytes": 0,
            "loop_slots": 0,
            "loop_slot": -1,
            "slot_lifespan": int(self.persistent_cache_slot_lifespan),
        }
        self._initialize_persistent_cache()
        # For non-augmented datasets with no disk cache: pre-materialise every row
        # once now so each subsequent __getitem__ is a zero-cost list lookup.
        if not bool(self.augment) and int(self.cached_rows) < int(self.total_rows):
            for _ei in tqdm(
                range(int(self.total_rows)),
                desc=f"[{self.dataset_name}] materializing",
                unit="row",
                leave=False,
                dynamic_ncols=True,
            ):
                self._eager_row_cache[_ei] = self._materialize_numpy_row(_ei)
            self._eager_row_cache_ready = True

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_cache_images_mm"] = None
        state["_cache_targets_mm"] = None
        state["_cache_masks_mm"] = None
        state["_cache_mask_stacks_mm"] = None
        state["_cache_mask_indices_mm"] = None
        state["_cache_mask_offsets_mm"] = None
        # Don't ship the eager cache across process boundaries — workers re-derive
        # on demand via _materialize_numpy_row (num_workers=0 on Windows/CUDA anyway).
        state["_eager_row_cache"] = [None] * int(len(state.get("_eager_row_cache", [])))
        state["_eager_row_cache_ready"] = False
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._cache_images_mm = None
        self._cache_targets_mm = None
        self._cache_masks_mm = None
        self._cache_mask_stacks_mm = None
        self._cache_mask_indices_mm = None
        self._cache_mask_offsets_mm = None
        self._eager_row_cache_ready = False

    def _desired_cached_rows(self) -> int:
        if not self.persistent_cache_dir:
            return 0
        cap = int(self.persistent_cache_max_rows)
        if int(cap) <= 0:
            cap = int(self.total_rows)
        return max(0, min(int(self.total_rows), int(cap)))

    def _estimated_persistent_cache_bytes(self, desired_rows: int) -> int:
        rows = max(0, int(desired_rows))
        if rows <= 0 or int(len(self.images)) <= 0:
            return 0
        sample_img = np.asarray(self.images[0], dtype=np.float32)
        if int(sample_img.ndim) != 3:
            raise RuntimeError(f"{str(self.dataset_name)} cache sizing requires CHW image rows; got {tuple(sample_img.shape)}")
        img_bytes = int(rows) * int(sample_img.size)
        tgt_bytes = int(rows) * int(self.target_dim) * int(np.dtype(np.float16).itemsize)
        mask_bytes = 0
        if bool(self.return_masks):
            mask_bytes = int(rows) * int(sample_img.shape[1]) * int(sample_img.shape[2])
        stack_bytes = 0
        if bool(self.return_mask_stack):
            avg_positive = _estimate_target_positive_count(self.targets)
            stack_rows = int(rows * max(1.0, float(avg_positive)))
            stack_bytes = int(stack_rows) * int(sample_img.shape[1]) * int(sample_img.shape[2])
            stack_bytes += int(stack_rows) * int(np.dtype(np.int64).itemsize)
            stack_bytes += int(rows + 1) * int(np.dtype(np.int64).itemsize)
        return int(img_bytes + tgt_bytes + mask_bytes + stack_bytes + 4096)

    def _ensure_persistent_cache_open(self) -> None:
        if int(self.cached_rows) <= 0:
            return
        if self._cache_images_mm is None and str(self._cache_images_path).strip():
            self._cache_images_mm = np.load(str(self._cache_images_path), mmap_mode="r")
        if self._cache_targets_mm is None and str(self._cache_targets_path).strip():
            self._cache_targets_mm = np.load(str(self._cache_targets_path), mmap_mode="r")
        if bool(self.return_masks) and self._cache_masks_mm is None and str(self._cache_masks_path).strip():
            self._cache_masks_mm = np.load(str(self._cache_masks_path), mmap_mode="r")
        if bool(self.return_mask_stack) and self._cache_mask_stacks_mm is None and str(self._cache_mask_stacks_path).strip():
            self._cache_mask_stacks_mm = np.load(str(self._cache_mask_stacks_path), mmap_mode="r")
        if bool(self.return_mask_stack) and self._cache_mask_indices_mm is None and str(self._cache_mask_indices_path).strip():
            self._cache_mask_indices_mm = np.load(str(self._cache_mask_indices_path), mmap_mode="r")
        if bool(self.return_mask_stack) and self._cache_mask_offsets_mm is None and str(self._cache_mask_offsets_path).strip():
            self._cache_mask_offsets_mm = np.load(str(self._cache_mask_offsets_path), mmap_mode="r")

    def _load_existing_persistent_cache(self, manifest_path: Path, desired_rows: int) -> int:
        if not manifest_path.exists() or bool(self.persistent_cache_rebuild):
            return 0
        images_path = manifest_path.with_name("images.npy")
        targets_path = manifest_path.with_name("targets.npy")
        masks_path = manifest_path.with_name("masks.npy")
        mask_stacks_path = manifest_path.with_name("mask_stacks.npy")
        mask_indices_path = manifest_path.with_name("mask_indices.npy")
        mask_offsets_path = manifest_path.with_name("mask_offsets.npy")
        if not images_path.exists() or not targets_path.exists():
            return 0
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return 0
        manifest_version = int(manifest.get("version", -1))
        if int(manifest_version) not in (3,):
            return 0
        if str(manifest.get("signature", "")) != str(self.persistent_cache_signature):
            return 0
        if int(manifest.get("target_dim", -1)) != int(self.target_dim):
            return 0
        if bool(self.return_mask_stack) and int(manifest_version) < 3:
            return 0
        cached_rows = int(manifest.get("cached_rows", 0))
        if int(cached_rows) <= 0:
            return 0
        try:
            mm_images = np.load(str(images_path), mmap_mode="r")
            mm_targets = np.load(str(targets_path), mmap_mode="r")
            if int(getattr(mm_images, "ndim", 0)) != 4 or int(getattr(mm_targets, "ndim", 0)) != 2:
                return 0
            usable = min(int(cached_rows), int(mm_images.shape[0]), int(mm_targets.shape[0]), int(desired_rows))
            if int(usable) <= 0:
                return 0
            if bool(self.return_masks):
                if not masks_path.exists():
                    return 0
                mm_masks = np.load(str(masks_path), mmap_mode="r")
                if int(getattr(mm_masks, "ndim", 0)) != 3:
                    return 0
                usable = min(int(usable), int(mm_masks.shape[0]))
                self._cache_masks_mm = mm_masks
                self._cache_masks_path = str(masks_path)
            if bool(self.return_mask_stack):
                if not mask_stacks_path.exists() or not mask_indices_path.exists() or not mask_offsets_path.exists():
                    return 0
                mm_mask_stacks = np.load(str(mask_stacks_path), mmap_mode="r")
                mm_mask_indices = np.load(str(mask_indices_path), mmap_mode="r")
                mm_mask_offsets = np.load(str(mask_offsets_path), mmap_mode="r")
                if int(getattr(mm_mask_stacks, "ndim", 0)) != 3 or int(getattr(mm_mask_indices, "ndim", 0)) != 1 or int(getattr(mm_mask_offsets, "ndim", 0)) != 1:
                    return 0
                if int(mm_mask_offsets.shape[0]) < int(usable + 1):
                    return 0
                self._cache_mask_stacks_mm = mm_mask_stacks
                self._cache_mask_indices_mm = mm_mask_indices
                self._cache_mask_offsets_mm = mm_mask_offsets
                self._cache_mask_stacks_path = str(mask_stacks_path)
                self._cache_mask_indices_path = str(mask_indices_path)
                self._cache_mask_offsets_path = str(mask_offsets_path)
            self._cache_images_mm = mm_images
            self._cache_targets_mm = mm_targets
            self._cache_images_path = str(images_path)
            self._cache_targets_path = str(targets_path)
            self._cache_manifest_path = str(manifest_path)
            self.cached_rows = int(usable)
            self.persistent_cache_info.update(
                {
                    "enabled": True,
                    "desired_rows": int(desired_rows),
                    "cached_rows": int(usable),
                    "cache_hit": bool(int(usable) >= int(desired_rows)),
                    "rebuilt": False,
                    "generated_rows": 0,
                    "manifest": str(manifest_path),
                }
            )
            return int(usable)
        except Exception:
            self._cache_images_mm = None
            self._cache_targets_mm = None
            self._cache_masks_mm = None
            return 0

    def _write_persistent_cache(self, cache_dir: Path, desired_rows: int) -> int:
        _reset_cache_dir(cache_dir)
        self._cache_images_mm = None
        self._cache_targets_mm = None
        self._cache_masks_mm = None
        self._cache_mask_stacks_mm = None
        self._cache_mask_indices_mm = None
        self._cache_mask_offsets_mm = None
        images_rows: List[np.ndarray] = []
        target_rows: List[np.ndarray] = []
        mask_rows: List[np.ndarray] = []
        mask_stack_rows: List[np.ndarray] = []
        mask_index_rows: List[np.ndarray] = []
        mask_offsets: List[int] = [0]
        for idx in tqdm(
            range(int(desired_rows)),
            desc=f"[{self.dataset_name}] building cache",
            unit="row",
            leave=False,
            dynamic_ncols=True,
        ):
            img_np, tgt_np, mask_np, mask_stack_np, mask_idx_np = self._materialize_numpy_row(int(idx))
            images_rows.append(_cache_encode_image_u8(img_np))
            target_rows.append(_cache_encode_target_f16(tgt_np))
            if bool(self.return_masks):
                mask_rows.append(_cache_encode_mask_f16(mask_np))
            if bool(self.return_mask_stack):
                stack_f16 = _cache_encode_mask_f16(mask_stack_np) if int(np.asarray(mask_stack_np).size) > 0 else np.zeros((0, int(img_np.shape[1]), int(img_np.shape[2])), dtype=np.float16)
                idx_np = np.asarray(mask_idx_np, dtype=np.int64).reshape(-1)
                if int(stack_f16.shape[0]) != int(idx_np.size):
                    raise RuntimeError(f"{str(self.dataset_name)} mask stack cache mismatch: stack_rows={int(stack_f16.shape[0])} idx={int(idx_np.size)}")
                mask_stack_rows.append(np.asarray(stack_f16, dtype=np.float16))
                mask_index_rows.append(idx_np)
                mask_offsets.append(int(mask_offsets[-1] + int(stack_f16.shape[0])))
        if int(len(images_rows)) <= 0:
            return 0
        images_np = np.stack(images_rows, axis=0).astype(np.uint8, copy=False)
        targets_np = np.stack(target_rows, axis=0).astype(np.float16, copy=False)
        masks_np = np.stack(mask_rows, axis=0).astype(np.float16, copy=False) if bool(self.return_masks) else None
        if bool(self.return_mask_stack):
            if int(mask_offsets[-1]) > 0:
                mask_stacks_np = np.concatenate(mask_stack_rows, axis=0).astype(np.float16, copy=False)
                mask_indices_np = np.concatenate(mask_index_rows, axis=0).astype(np.int64, copy=False)
            else:
                mask_stacks_np = np.zeros((0, int(images_np.shape[2]), int(images_np.shape[3])), dtype=np.float16)
                mask_indices_np = np.zeros((0,), dtype=np.int64)
            mask_offsets_np = np.asarray(mask_offsets, dtype=np.int64)
        else:
            mask_stacks_np = None
            mask_indices_np = None
            mask_offsets_np = None
        images_path = cache_dir / "images.npy"
        targets_path = cache_dir / "targets.npy"
        masks_path = cache_dir / "masks.npy"
        mask_stacks_path = cache_dir / "mask_stacks.npy"
        mask_indices_path = cache_dir / "mask_indices.npy"
        mask_offsets_path = cache_dir / "mask_offsets.npy"
        manifest_path = cache_dir / "manifest.json"
        np.save(str(images_path), images_np)
        np.save(str(targets_path), targets_np)
        if masks_np is not None:
            np.save(str(masks_path), masks_np)
        elif masks_path.exists():
            masks_path.unlink()
        if mask_stacks_np is not None and mask_indices_np is not None and mask_offsets_np is not None:
            np.save(str(mask_stacks_path), mask_stacks_np)
            np.save(str(mask_indices_path), mask_indices_np)
            np.save(str(mask_offsets_path), mask_offsets_np)
        else:
            for orphan_path in (mask_stacks_path, mask_indices_path, mask_offsets_path):
                if orphan_path.exists():
                    orphan_path.unlink()
        manifest = {
            "version": 3,
            "dataset_name": str(self.dataset_name),
            "signature": str(self.persistent_cache_signature),
            "cached_rows": int(images_np.shape[0]),
            "target_dim": int(self.target_dim),
            "return_masks": bool(self.return_masks),
            "return_mask_stack": bool(self.return_mask_stack),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self._cache_images_mm = None
        self._cache_targets_mm = None
        self._cache_masks_mm = None
        self._cache_mask_stacks_mm = None
        self._cache_mask_indices_mm = None
        self._cache_mask_offsets_mm = None
        self._cache_images_path = str(images_path)
        self._cache_targets_path = str(targets_path)
        self._cache_masks_path = str(masks_path if masks_np is not None else "")
        self._cache_mask_stacks_path = str(mask_stacks_path if mask_stacks_np is not None else "")
        self._cache_mask_indices_path = str(mask_indices_path if mask_indices_np is not None else "")
        self._cache_mask_offsets_path = str(mask_offsets_path if mask_offsets_np is not None else "")
        self._cache_manifest_path = str(manifest_path)
        self.cached_rows = int(images_np.shape[0])
        self.persistent_cache_info.update(
            {
                "enabled": True,
                "desired_rows": int(desired_rows),
                "cached_rows": int(self.cached_rows),
                "cache_hit": False,
                "rebuilt": True,
                "generated_rows": int(self.cached_rows),
                "manifest": str(manifest_path),
            }
        )
        return int(self.cached_rows)

    def _initialize_loop_persistent_cache(self, cache_root: Path, desired_rows: int) -> bool:
        loop_root = Path(cache_root) / "_loop_pool"
        loop_root.mkdir(parents=True, exist_ok=True)
        estimated_bytes = int(self._estimated_persistent_cache_bytes(int(desired_rows)))
        self.persistent_cache_info["estimated_bytes"] = int(estimated_bytes)
        cap_bytes = max(0, int(self.persistent_cache_max_bytes))
        slot_count = 1
        if int(cap_bytes) > 0 and int(estimated_bytes) > 0:
            slot_count = max(1, int(cap_bytes // max(1, int(estimated_bytes))))
        self.persistent_cache_info["loop_slots"] = int(slot_count)
        lifespan = max(0, int(self.persistent_cache_slot_lifespan))
        control = _load_cache_control(loop_root)
        slot_use_counts: Dict[str, int] = {
            str(k): max(0, int(v))
            for k, v in control.get("slot_use_counts", {}).items()
        }
        for slot_idx in range(int(slot_count)):
            slot_dir = _loop_slot_dir(loop_root, slot_idx)
            manifest_path = slot_dir / "manifest.json"
            # Skip expired slots (lifespan > 0 and use count has reached the limit).
            uses = int(slot_use_counts.get(str(slot_idx), 0))
            if int(lifespan) > 0 and int(uses) >= int(lifespan):
                continue
            loaded_rows = self._load_existing_persistent_cache(manifest_path=manifest_path, desired_rows=int(desired_rows))
            if int(loaded_rows) >= int(desired_rows):
                # Increment the use count for the chosen slot.
                slot_use_counts[str(slot_idx)] = int(uses) + 1
                control["slot_use_counts"] = slot_use_counts
                _store_cache_control(loop_root, control)
                self.persistent_cache_info["cache_dir"] = str(slot_dir)
                self.persistent_cache_info["loop_slot"] = int(slot_idx)
                self.persistent_cache_info["slot_uses"] = int(uses) + 1
                self.persistent_cache_info["slot_lifespan"] = int(lifespan)
                self.persistent_cache_info["pool_bytes"] = int(_directory_size_bytes(loop_root))
                return True
        cursor = int(control.get("loop_cursor", 0))
        slot_idx = int(cursor % max(1, int(slot_count)))
        control["loop_cursor"] = int((slot_idx + 1) % max(1, int(slot_count)))
        control["slot_count"] = int(slot_count)
        # Reset use count for the slot being rebuilt.
        slot_use_counts[str(slot_idx)] = 0
        control["slot_use_counts"] = slot_use_counts
        _store_cache_control(loop_root, control)
        slot_dir = _loop_slot_dir(loop_root, slot_idx)
        self._write_persistent_cache(cache_dir=slot_dir, desired_rows=int(desired_rows))
        self.persistent_cache_info["cache_dir"] = str(slot_dir)
        self.persistent_cache_info["loop_slot"] = int(slot_idx)
        self.persistent_cache_info["slot_uses"] = 0
        self.persistent_cache_info["slot_lifespan"] = int(lifespan)
        self.persistent_cache_info["pool_bytes"] = int(_directory_size_bytes(loop_root))
        return True

    def _initialize_evicting_persistent_cache(self, cache_root: Path, desired_rows: int) -> bool:
        estimated_bytes = int(self._estimated_persistent_cache_bytes(int(desired_rows)))
        self.persistent_cache_info["estimated_bytes"] = int(estimated_bytes)
        cache_dir = Path(cache_root) / str(self.persistent_cache_signature)[:24]
        manifest_path = cache_dir / "manifest.json"
        loaded_rows = self._load_existing_persistent_cache(manifest_path=manifest_path, desired_rows=int(desired_rows))
        if int(loaded_rows) >= int(desired_rows):
            self.persistent_cache_info["cache_dir"] = str(cache_dir)
            self.persistent_cache_info["pool_bytes"] = int(_directory_size_bytes(cache_root))
            return True
        cap_bytes = max(0, int(self.persistent_cache_max_bytes))
        if int(cap_bytes) > 0:
            pool_bytes = int(_directory_size_bytes(cache_root))
            current_dir_bytes = int(_directory_size_bytes(cache_dir)) if cache_dir.exists() else 0
            candidate_dirs = [p for p in Path(cache_root).iterdir() if p.is_dir() and p != cache_dir]
            candidate_dirs = sorted(candidate_dirs, key=_cache_dir_mtime_ns)
            while int(pool_bytes - current_dir_bytes + estimated_bytes) > int(cap_bytes) and len(candidate_dirs) > 0:
                victim = candidate_dirs.pop(0)
                victim_bytes = int(_directory_size_bytes(victim))
                try:
                    _remove_tree(victim)
                    pool_bytes -= int(victim_bytes)
                except Exception:
                    break
        self._write_persistent_cache(cache_dir=cache_dir, desired_rows=int(desired_rows))
        self.persistent_cache_info["cache_dir"] = str(cache_dir)
        self.persistent_cache_info["pool_bytes"] = int(_directory_size_bytes(cache_root))
        return True

    def _initialize_persistent_cache(self) -> None:
        desired_rows = int(self._desired_cached_rows())
        self.persistent_cache_info["desired_rows"] = int(desired_rows)
        if int(desired_rows) <= 0:
            return
        cache_root = Path(str(self.persistent_cache_dir)).resolve()
        cache_root.mkdir(parents=True, exist_ok=True)
        if str(self.persistent_cache_overflow_strategy) == "loop":
            self._initialize_loop_persistent_cache(cache_root=cache_root, desired_rows=int(desired_rows))
            return
        self._initialize_evicting_persistent_cache(cache_root=cache_root, desired_rows=int(desired_rows))

    def _materialize_numpy_row(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = int(index)
        if idx < 0:
            idx = int(self.total_rows) + idx
        if idx < 0 or idx >= int(self.total_rows):
            raise IndexError(idx)
        if bool(self._eager_row_cache_ready) and self._eager_row_cache[idx] is not None:
            return self._eager_row_cache[idx]  # type: ignore[return-value]
        cycle = int(idx // max(1, int(self.base_rows)))
        off = int(idx % max(1, int(self.base_rows)))
        base_idx = int((off + (cycle * 17) + int(self.seed % max(1, int(self.base_rows)))) % int(self.base_rows))
        img = np.asarray(self.images[int(base_idx)], dtype=np.float32)
        tgt = np.asarray(self.targets[int(base_idx)], dtype=np.float32).copy()
        exact_term_masks: Dict[str, Any] = {}
        if bool(self.return_masks):
            mask_base, mask_stack_base, mask_idx_base = self._get_base_mask_bundle(int(base_idx))
            mask = np.asarray(mask_base, dtype=np.float32).copy()
            mask_stack_np = np.asarray(mask_stack_base, dtype=np.float32).copy()
            mask_idx_np = np.asarray(mask_idx_base, dtype=np.int64).copy()
        else:
            mask = np.zeros((int(img.shape[1]), int(img.shape[2])), dtype=np.float32)
            mask_stack_np = np.zeros((0, int(img.shape[1]), int(img.shape[2])), dtype=np.float32)
            mask_idx_np = np.zeros((0,), dtype=np.int64)
        if bool(self.augment) and int(self.total_rows) > int(self.base_rows):
            aug_seed = int((int(self.seed) * 2654435761 + int(idx) * 1103515245 + int(base_idx) * 122949829) % (2**32 - 1))
            img, aug_terms, touched, aug_term_masks = augment_bootstrap_chw01(
                img=img,
                seed=int(aug_seed),
                return_terms=True,
                return_touch_mask=True,
                return_term_masks=True,
            )
            aug_term_set = {
                re.sub(r"\s+", " ", str(term)).strip().lower()
                for term in (aug_terms if isinstance(aug_terms, list) else [])
                if str(term).strip()
            }
            global_aug = bool(
                aug_term_set.intersection(
                    {
                        "signal",
                        "noise",
                        "mixed noise and signal",
                        "blur damage",
                        "noise damage",
                        "dropout damage",
                        "quantization damage",
                        "stride skew damage",
                        "edge highlight",
                        "edge blur",
                        "texture",
                        "pattern",
                    }
                )
            )
            mask = _blend_attention_maps(
                [np.asarray(mask, dtype=np.float32), np.asarray(touched, dtype=np.float32)],
                weights=([0.35, 1.25] if bool(global_aug) else [0.70, 0.95]),
                gamma=(1.08 if bool(global_aug) else 0.95),
            ).astype(np.float32, copy=False)
            if bool(self.augment_apply_terms) and int(len(self.semantic_term_to_idx)) > 0 and isinstance(aug_terms, list):
                for term in aug_terms:
                    tk = re.sub(r"\s+", " ", str(term)).strip().lower()
                    if not tk:
                        continue
                    ti = int(self.semantic_term_to_idx.get(tk, -1))
                    if 0 <= int(ti) < int(tgt.size):
                        tgt[int(ti)] = 1.0
        if bool(self.return_masks):
            tgt, mask, mask_stack_np, mask_idx_np = assemble_semantic_mask_layers(
                image=img,
                label_vec=tgt,
                idx_to_term=self.idx_to_term,
                term_to_idx=self.semantic_term_to_idx,
                original_mixed_mask=np.asarray(mask, dtype=np.float32),
                original_parts=[(mask_stack_np, mask_idx_np)],
                deformation_term_masks=aug_term_masks if bool(self.augment) and int(self.total_rows) > int(self.base_rows) else None,
            )
        return (
            np.asarray(img, dtype=np.float32),
            np.asarray(tgt, dtype=np.float32),
            np.asarray(mask, dtype=np.float32),
            np.asarray(mask_stack_np, dtype=np.float32),
            np.asarray(mask_idx_np, dtype=np.int64),
        )

    def _materialize_cached_row(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._ensure_persistent_cache_open()
        if self._cache_images_mm is None or self._cache_targets_mm is None:
            raise RuntimeError(f"{str(self.dataset_name)} persistent cache is unavailable.")
        img = _cache_decode_image_u8(self._cache_images_mm[int(index)])
        tgt = _cache_decode_target_f16(self._cache_targets_mm[int(index)]).reshape(-1)
        if bool(self.return_masks) and self._cache_masks_mm is not None:
            mask = _cache_decode_mask_f16(self._cache_masks_mm[int(index)])
        else:
            mask = np.zeros((int(img.shape[1]), int(img.shape[2])), dtype=np.float32)
        if bool(self.return_mask_stack):
            if self._cache_mask_offsets_mm is None or self._cache_mask_indices_mm is None or self._cache_mask_stacks_mm is None:
                mask_stack_np, mask_idx_np = build_label_mask_stack(
                    mixed_mask=np.asarray(mask, dtype=np.float32),
                    label_vec=tgt,
                    idx_to_term=self.idx_to_term,
                    treat_mixed_mask_as_creation=bool(float(np.max(np.asarray(mask, dtype=np.float32))) > 1e-8),
                )
            else:
                start = int(self._cache_mask_offsets_mm[int(index)])
                stop = int(self._cache_mask_offsets_mm[int(index) + 1])
                if int(stop) > int(start):
                    mask_stack_np = _cache_decode_mask_f16(self._cache_mask_stacks_mm[int(start):int(stop)])
                    mask_idx_np = np.asarray(self._cache_mask_indices_mm[int(start):int(stop)], dtype=np.int64)
                else:
                    mask_stack_np = np.zeros((0, int(mask.shape[0]), int(mask.shape[1])), dtype=np.float32)
                    mask_idx_np = np.zeros((0,), dtype=np.int64)
                if int(mask_stack_np.shape[0]) <= 0 or int(mask_idx_np.size) <= 0:
                    mask_stack_np, mask_idx_np = build_label_mask_stack(
                        mixed_mask=np.asarray(mask, dtype=np.float32),
                        label_vec=tgt,
                        idx_to_term=self.idx_to_term,
                        treat_mixed_mask_as_creation=bool(float(np.max(np.asarray(mask, dtype=np.float32))) > 1e-8),
                    )
            if float(np.max(np.asarray(mask, dtype=np.float32))) <= 1e-8 and int(mask_stack_np.shape[0]) > 0 and int(mask_idx_np.size) > 0:
                mask = _composite_non_dataset_label_stack(
                    mask_stack_np,
                    mask_idx_np,
                    idx_to_term=self.idx_to_term,
                    height=int(img.shape[1]),
                    width=int(img.shape[2]),
                    fallback_mask=mask,
                )
        else:
            mask_stack_np = np.zeros((0, int(mask.shape[0]), int(mask.shape[1])), dtype=np.float32)
            mask_idx_np = np.zeros((0,), dtype=np.int64)
        return (
            np.asarray(img, dtype=np.float32),
            np.asarray(tgt, dtype=np.float32),
            np.asarray(mask, dtype=np.float32),
            np.asarray(mask_stack_np, dtype=np.float32),
            np.asarray(mask_idx_np, dtype=np.int64),
        )

    def _get_base_mask_bundle(self, base_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = int(base_idx)
        cached_mask = self.base_masks[int(idx)]
        cached_stack = self.base_mask_stacks[int(idx)]
        cached_stack_idx = self.base_mask_stack_indices[int(idx)]
        if cached_mask is None:
            # Derive composite mask from the element stack if the builder
            # provided one. For a single-label sample with no explicit mask,
            # treat the whole image as the source mask.
            if cached_stack is not None and int(np.asarray(cached_stack).ndim) == 3 and int(np.asarray(cached_stack).shape[0]) > 0:
                cached_mask = _composite_mask_stack(np.asarray(cached_stack, dtype=np.float32))
            else:
                img_ref = np.asarray(self.images[int(idx)], dtype=np.float32)
                cached_mask = _single_label_whole_image_mask(
                    self.targets[int(idx)],
                    height=int(img_ref.shape[1]),
                    width=int(img_ref.shape[2]),
                )
                if cached_mask is None:
                    cached_mask = np.zeros((int(img_ref.shape[1]), int(img_ref.shape[2])), dtype=np.float32)
            self.base_masks[int(idx)] = np.asarray(cached_mask, dtype=np.float32)
        if bool(self.return_mask_stack) and (cached_stack is None or cached_stack_idx is None):
            tgt = np.asarray(self.targets[int(idx)], dtype=np.float32).reshape(-1)
            cached_stack, cached_stack_idx = build_label_mask_stack(
                mixed_mask=np.asarray(cached_mask, dtype=np.float32),
                label_vec=tgt,
                idx_to_term=self.idx_to_term,
                treat_mixed_mask_as_creation=bool(float(np.max(np.asarray(cached_mask, dtype=np.float32))) > 1e-8),
            )
            self.base_mask_stacks[int(idx)] = np.asarray(cached_stack, dtype=np.float32)
            self.base_mask_stack_indices[int(idx)] = np.asarray(cached_stack_idx, dtype=np.int64)
        mask_np = np.asarray(self.base_masks[int(idx)], dtype=np.float32)
        if bool(self.return_mask_stack):
            stack_np = np.asarray(self.base_mask_stacks[int(idx)], dtype=np.float32)
            stack_idx_np = np.asarray(self.base_mask_stack_indices[int(idx)], dtype=np.int64)
            if float(np.max(mask_np)) <= 1e-8 and int(stack_np.shape[0]) > 0 and int(stack_idx_np.size) > 0:
                mask_np = _composite_non_dataset_label_stack(
                    stack_np,
                    stack_idx_np,
                    idx_to_term=self.idx_to_term,
                    height=int(mask_np.shape[0]),
                    width=int(mask_np.shape[1]),
                    fallback_mask=mask_np,
                )
                self.base_masks[int(idx)] = np.asarray(mask_np, dtype=np.float32)
        else:
            stack_np = np.zeros((0, int(mask_np.shape[0]), int(mask_np.shape[1])), dtype=np.float32)
            stack_idx_np = np.zeros((0,), dtype=np.int64)
        return mask_np, stack_np, stack_idx_np

    def __len__(self) -> int:
        return int(self.total_rows)

    def __getitem__(self, index: int):
        idx = int(index)
        if idx < 0:
            idx = int(self.total_rows) + idx
        if idx < 0 or idx >= int(self.total_rows):
            raise IndexError(idx)
        if int(idx) < int(self.cached_rows):
            img, tgt, mask, mask_stack_np, mask_idx_np = self._materialize_cached_row(int(idx))
        else:
            img, tgt, mask, mask_stack_np, mask_idx_np = self._materialize_numpy_row(int(idx))
        x_t = torch.from_numpy(np.asarray(img, dtype=np.float32))
        y_t = torch.from_numpy(np.asarray(tgt, dtype=np.float32))
        if bool(self.return_masks):
            mask_t = torch.from_numpy(np.asarray(mask, dtype=np.float32)[None, ...])
            if bool(self.return_mask_stack):
                if int(mask_stack_np.shape[0]) <= 0 or int(mask_idx_np.size) <= 0:
                    mask_stack_np, mask_idx_np = build_label_mask_stack(
                        mixed_mask=np.asarray(mask, dtype=np.float32),
                        label_vec=tgt,
                        idx_to_term=self.idx_to_term,
                        treat_mixed_mask_as_creation=bool(float(np.max(np.asarray(mask, dtype=np.float32))) > 1e-8),
                    )
                return (
                    x_t,
                    y_t,
                    mask_t,
                    torch.from_numpy(np.asarray(mask_stack_np, dtype=np.float32)),
                    torch.from_numpy(np.asarray(mask_idx_np, dtype=np.int64)),
                )
            return x_t, y_t, mask_t
        return x_t, y_t


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
                label_vec=np.asarray(r.label_vec, dtype=np.float32).reshape(-1),
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
        self.term_to_idx = {
            _norm_txt(str(name)): int(i)
            for i, name in enumerate(self.class_names)
            if str(name).strip()
        }
        self.idx_to_term = {
            int(i): str(name)
            for i, name in enumerate(self.class_names)
            if str(name).strip()
        }
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
        cached_bundle = _load_row_mask_cache_bundle(getattr(row, "mask_cache_file", ""))
        if isinstance(cached_bundle, dict) and cached_bundle.get("mixed_mask", None) is not None:
            arr = np.asarray(cached_bundle.get("mixed_mask"), dtype=np.float32)
            if int(arr.ndim) == 3:
                arr = np.mean(arr, axis=0).astype(np.float32, copy=False)
            return _normalize_mask_array(arr, height=int(h), width=int(w))
        single_mask = _single_label_whole_image_mask(row.label_vec, height=int(h), width=int(w))
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
        y = np.asarray(row.label_vec, dtype=np.float32).reshape(-1)
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
            y, mask_np, mask_stack_np, mask_idx_np = assemble_semantic_mask_layers(
                image=x,
                label_vec=y,
                idx_to_term=self.idx_to_term,
                term_to_idx=self.term_to_idx,
                original_mixed_mask=np.asarray(base_mask_np, dtype=np.float32),
                original_parts=[(cached_stack_array, cached_stack_indices)],
                deformation_term_masks=degrade_term_masks,
            )
            x_t = torch.from_numpy(np.asarray(x, dtype=np.float32))
            y_t = torch.from_numpy(np.asarray(y, dtype=np.float32))
            mask_t = torch.from_numpy(np.asarray(mask_np, dtype=np.float32)[None, ...])
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
            "cache_version": 4,
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
            mixed = np.asarray(z["mixed_mask"], dtype=np.float32)
            stack = np.asarray(z["mask_stack"], dtype=np.float32)
            indices = np.asarray(z["mask_indices"], dtype=np.int64)
        return {
            "mixed_mask": mixed,
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
    label_vec: np.ndarray,
    idx_to_term: Optional[Dict[int, str]],
    terms: Sequence[str],
    mask_path: str = "",
    layout: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], bool]:
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_file = _mask_cache_file_path(
        cache_root=cache_root,
        image_path=image_path,
        terms=terms,
        mask_path=mask_path,
        layout=layout,
    )
    cached = _load_mask_cache_npz(cache_file)
    if isinstance(cached, dict):
        return cached.get("mixed_mask"), cached.get("mask_stack"), cached.get("mask_indices"), False
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
                explicit_mask = _normalize_attention_map(_normalize_mask_array(explicit_mask, height=int(h), width=int(w)), gamma=0.95, blur_kernel=3)
    if explicit_mask is None and isinstance(layout, dict):
        explicit_mask = _normalize_attention_map(build_layout_mask(layout, height=int(h), width=int(w)), gamma=0.95, blur_kernel=3)

    term_to_idx = {
        _norm_txt(str(term)): int(idx)
        for idx, term in ((idx_to_term or {}).items() if isinstance(idx_to_term, dict) else [])
        if str(term).strip()
    }
    _, mixed_mask, inferred_stack, inferred_idx = assemble_semantic_mask_layers(
        image=chw,
        label_vec=label_vec,
        idx_to_term=idx_to_term,
        term_to_idx=term_to_idx,
        original_mixed_mask=np.asarray(explicit_mask, dtype=np.float32) if explicit_mask is not None else None,
        original_parts=None,
        deformation_term_masks=None,
    )
    mixed_mask = _normalize_attention_map(np.asarray(mixed_mask, dtype=np.float32), gamma=0.92, blur_kernel=5)
    try:
        np.savez_compressed(
            str(cache_file),
            mixed_mask=np.asarray(mixed_mask, dtype=np.float32),
            mask_stack=np.asarray(inferred_stack, dtype=np.float32),
            mask_indices=np.asarray(inferred_idx, dtype=np.int64),
        )
    except Exception:
        pass
    return np.asarray(mixed_mask, dtype=np.float32), np.asarray(inferred_stack, dtype=np.float32), np.asarray(inferred_idx, dtype=np.int64), True


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
    idx_to_term = {int(i): str(name) for i, name in enumerate(class_names)}
    workers = int(max_workers)
    if workers <= 0:
        workers = min(8, max(1, (os.cpu_count() or 1)))
    info["mask_cache_threads"] = int(workers)

    def _job(row_index: int) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], bool]:
        row = rows[int(row_index)]
        cache_file = _mask_cache_file_path(
            cache_root=cache_root,
            image_path=Path(str(row.image_path)),
            terms=row.terms,
            mask_path=str(row.mask_path or ""),
            layout=(dict(row.layout) if isinstance(row.layout, dict) else None),
        )
        if cache_file.exists():
            return int(row_index), None, None, None, False
        mixed, stack, indices, created = _build_and_store_row_mask_cache(
            cache_root=cache_root,
            image_path=Path(str(row.image_path)),
            label_vec=np.asarray(row.label_vec, dtype=np.float32).reshape(-1),
            idx_to_term=idx_to_term,
            terms=row.terms,
            mask_path=str(row.mask_path or ""),
            layout=(dict(row.layout) if isinstance(row.layout, dict) else None),
        )
        return int(row_index), mixed, stack, indices, bool(created)

    if int(workers) <= 1:
        for row_index in range(int(len(rows))):
            try:
                ridx, mixed, stack, indices, created = _job(row_index)
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
                ridx, mixed, stack, indices, created = done_batch.result()
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


def collect_semantic_disk_rows(
    data_root: str,
    class_names: Sequence[str],
    source_root: str = "",
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
        voc20_names = [
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "dining table",
            "dog",
            "horse",
            "motorbike",
            "person",
            "potted plant",
            "sheep",
            "sofa",
            "train",
            "tv monitor",
        ]
    voc20_to_class_idx = {
        int(voc_idx): int(class_lut[key])
        for voc_idx, key in enumerate([_norm_txt(name) for name in voc20_names])
        if key in class_lut
    }

    def _augment_row_with_auto_color_terms(image_path: Path, label_vec: np.ndarray, terms_in: Sequence[str], mask_path: str = "") -> Tuple[np.ndarray, List[str]]:
        out_vec = np.asarray(label_vec, dtype=np.float32).reshape(-1).copy()
        out_terms = normalize_vocab_terms([str(x) for x in list(terms_in)])
        try:
            with Image.open(str(image_path)) as im:
                rgb = np.asarray(im.convert("RGB"), dtype=np.float32)
            mask_arr = None
            if str(mask_path).strip():
                try:
                    mp = Path(str(mask_path))
                    if mp.exists() and str(mp.suffix).strip().lower() != ".mat":
                        with Image.open(str(mp)) as mm:
                            mask_arr = np.asarray(mm.convert("L"), dtype=np.float32)
                            mask_arr = np.clip(mask_arr / 255.0, 0.0, 1.0).astype(np.float32, copy=False)
                except Exception:
                    mask_arr = None
            color_terms = detect_semantic_color_terms(image=rgb, mask=mask_arr)
        except Exception:
            color_terms = []
        if len(color_terms) <= 0:
            return out_vec, out_terms
        for term in color_terms:
            idx = int(class_lut.get(_norm_txt(term), -1))
            if 0 <= int(idx) < int(out_vec.size):
                out_vec[int(idx)] = 1.0
        return out_vec, normalize_vocab_terms(list(out_terms) + list(color_terms))

    convert_sbd_mat_to_npz(root)
    split_specs = [("train", "berkeley_sbd_train"), ("val", "berkeley_sbd_val")]
    for split_name, source_key in split_specs:
        _split_images, _mask_paths = _read_sbd_split_file(root, split_name)
        label_path = root / "cache" / f"sbd_{split_name}_multilabel.npz"
        if not label_path.exists():
            try:
                from berkeley_sbd_pretrain import _labels_from_segmentation_masks, _load_sbd_split
                print(f"[collect-disk-rows] label cache missing — auto-building for split={split_name}...", flush=True)
                _ds = _load_sbd_split(root, image_set=split_name, download=False)
                _labels_from_segmentation_masks(_ds, label_path)
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
        split_specs_rows: List[Tuple[Path, np.ndarray, str, str, str]] = []
        split_missing = 0
        for i, img_path in enumerate(_split_images):
            ip = Path(str(img_path))
            if not ip.exists():
                split_missing += 1
                continue
            yv = np.zeros((int(n_classes),), dtype=np.float32)
            voc_vec = np.asarray(labels_split[int(i)], dtype=np.float32).reshape(-1)
            for voc_idx in np.flatnonzero(voc_vec > 0.5):
                cls_idx = int(voc20_to_class_idx.get(int(voc_idx), -1))
                if 0 <= int(cls_idx) < int(n_classes):
                    yv[int(cls_idx)] = 1.0
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
                )
            )
        missing_images += int(split_missing)

        split_workers = _resolve_semantic_startup_threads(len(split_specs_rows))
        startup_row_threads = max(int(startup_row_threads), int(split_workers))

        def _build_berkeley_row(spec: Tuple[Path, np.ndarray, str, str, str]) -> SemanticDiskRow:
            ip, yv_base, mask_path_local, source_key_local, _ = spec
            yv_local, row_terms = _augment_row_with_auto_color_terms(
                image_path=ip,
                label_vec=yv_base,
                terms_in=["berkeley sbd dataset", "object", "signal"],
                mask_path=str(mask_path_local),
            )
            pos = np.where(np.asarray(yv_local, dtype=np.float32) > 0.5)[0].astype(np.int64).tolist()
            label_terms = [str(class_names[int(j)]) for j in pos if 0 <= int(j) < int(len(class_names))]
            terms = normalize_vocab_terms(list(row_terms) + list(label_terms))
            return SemanticDiskRow(
                image_path=str(ip),
                label_vec=np.asarray(yv_local, dtype=np.float32).reshape(-1),
                terms=list(terms),
                source=str(source_key_local),
                mask_path=str(mask_path_local),
            )

        for row in _ordered_thread_map(split_specs_rows, _build_berkeley_row, max_workers=int(split_workers), desc=f"[berkeley/{split_name}] loading rows"):
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
            vec = np.zeros((int(n_classes),), dtype=np.float32)
            if int(berkeley_dataset_idx) >= 0:
                vec[int(berkeley_dataset_idx)] = 1.0
            if int(signal_idx) >= 0:
                vec[int(signal_idx)] = 1.0
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
                    cls_idx = int(class_lut[key])
                    vec[int(cls_idx)] = 1.0
                    mapped_terms.append(str(class_names[int(cls_idx)]))
                    if str(cand) in specific_set:
                        specific_hits += 1
            if int(specific_hits) <= 0:
                return None, 1
            vec, auto_terms = _augment_row_with_auto_color_terms(
                image_path=fp,
                label_vec=vec,
                terms_in=mapped_terms,
                mask_path=_sidecar_mask_path(fp),
            )
            return SemanticDiskRow(
                image_path=str(fp),
                label_vec=np.asarray(vec, dtype=np.float32).reshape(-1),
                terms=list(normalize_vocab_terms(auto_terms)),
                source=str(dataset_name_local),
                mask_path=_sidecar_mask_path(fp),
                layout=_sidecar_layout(fp),
            ), 0

        for row, unmapped_skip in _ordered_thread_map(external_specs, _build_external_row, max_workers=int(external_workers), desc="[external] loading rows"):
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
    with _SEMANTIC_DISK_ROWS_CACHE_LOCK:
        _SEMANTIC_DISK_ROWS_CACHE[cache_key] = (_clone_semantic_disk_rows(rows), dict(info))
    return rows, info
