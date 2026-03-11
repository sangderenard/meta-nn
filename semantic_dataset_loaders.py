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

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
_SEMANTIC_DISK_ROWS_CACHE_LOCK = threading.Lock()
_SEMANTIC_DISK_ROWS_CACHE: Dict[str, Tuple[List["SemanticDiskRow"], Dict[str, Any]]] = {}


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
    cls_dir = _Path(str(root)) / "cls"
    if not cls_dir.exists():
        return
    needs = [f for f in sorted(cls_dir.glob("*.mat")) if not (cls_dir / f"{f.stem}.npz").exists()]
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
                    _np.savez_compressed(str(cls_dir / f"{mat_path.stem}.npz"), segmentation=seg)
            except Exception:
                errors += 1
        print(f"[mat2npz] done. errors={errors}", flush=True)
    except ImportError:
        print("[mat2npz] scipy not available; mat→npz conversion skipped", flush=True)


def _read_sbd_split_file(root, split_name: str):
    """Read Berkeley SBD split file and return (image_paths, mask_stems).

    Does not import scipy or torchvision — just reads the .txt index file.
    Prefers train_noval.txt for the train split (excludes val overlap).
    Returns: (list of Path, list of str stem names)
    """
    from pathlib import Path as _Path
    root = _Path(str(root))
    dataset_dir = root / "dataset"
    candidates = []
    if split_name == "train":
        candidates.append(dataset_dir / "train_noval.txt")
    candidates.append(dataset_dir / f"{split_name}.txt")
    split_file = next((c for c in candidates if c.exists()), None)
    if split_file is None:
        return [], []
    img_dir = root / "img"
    cls_dir = root / "cls"
    lines = [ln.strip() for ln in split_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    image_paths = []
    mask_stems = []
    for name in lines:
        img_p = img_dir / f"{name}.jpg"
        if img_p.exists():
            image_paths.append(img_p)
            mask_stems.append(str(name))
    return image_paths, mask_stems


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
        "data_root": str(Path(str(data_root).strip() or "toys_to_survive_development/data/berkeley_sbd").resolve()),
        "source_root": str(Path(str(source_root).strip()).resolve()) if str(source_root).strip() else "",
        "class_names": norm_classes,
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _resolve_semantic_startup_threads(work_items: int, max_cap: int = 16) -> int:
    if int(work_items) <= 1:
        return 1
    cpu_count = max(1, int(os.cpu_count() or 1))
    return max(1, min(int(max_cap), int(cpu_count), int(work_items)))


def _ordered_thread_map(items: Sequence[Any], worker_fn: Any, max_workers: int) -> List[Any]:
    count = int(len(items))
    if count <= 0:
        return []
    workers = max(1, min(int(max_workers), int(count)))
    if workers <= 1:
        return [worker_fn(item) for item in items]
    results: List[Any] = [None] * count
    with ThreadPoolExecutor(max_workers=int(workers), thread_name_prefix="semantic-row-build") as pool:
        future_to_index = {pool.submit(worker_fn, item): int(i) for i, item in enumerate(items)}
        for future in as_completed(future_to_index):
            idx = int(future_to_index[future])
            results[idx] = future.result()
    return results


class _ThreadedPrefetchIterator:
    def __init__(self, loader: Any, max_prefetch_batches: int):
        self._loader_iter = iter(loader)
        self._queue: "queue.Queue[Tuple[str, Any]]" = queue.Queue(maxsize=max(2, int(max_prefetch_batches)))
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            for batch in self._loader_iter:
                self._queue.put(("batch", batch))
            self._queue.put(("stop", None))
        except BaseException as e:
            self._queue.put(("error", e))

    def __iter__(self):
        return self

    def __next__(self):
        kind, payload = self._queue.get()
        if kind == "batch":
            return payload
        if kind == "error":
            raise payload
        raise StopIteration


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
    if int(arr.shape[0]) != int(height) or int(arr.shape[1]) != int(width):
        arr = np.asarray(
            TF.resize(
                Image.fromarray(np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8), mode="L"),
                [int(height), int(width)],
                interpolation=InterpolationMode.NEAREST,
            ),
            dtype=np.float32,
        )
    if float(np.max(arr)) > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)


def _normalize_attention_map(mask: Any, gamma: float = 1.0, blur_kernel: int = 0) -> np.ndarray:
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
    vmean = float(np.mean(arr)) if int(arr.size) > 0 else 0.0
    if vmean > 1e-8:
        arr = np.clip(arr / float(max(vmean * 2.0, 1.0)), 0.0, 1.0).astype(np.float32, copy=False)
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


def _blend_attention_maps(maps: Sequence[Any], weights: Optional[Sequence[float]] = None, gamma: float = 1.0) -> np.ndarray:
    valid: List[np.ndarray] = []
    valid_weights: List[float] = []
    for i, m in enumerate(maps):
        arr = np.asarray(m, dtype=np.float32)
        if int(arr.ndim) != 2 or int(arr.size) <= 0:
            continue
        w = 1.0 if weights is None or i >= int(len(weights)) else float(weights[i])
        if w <= 0.0:
            continue
        norm = _normalize_attention_map(arr, gamma=1.0, blur_kernel=0)
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
    return _normalize_attention_map(acc, gamma=float(gamma), blur_kernel=5)


def _composite_mask_stack(stack: Any) -> np.ndarray:
    """Additive-sum all per-label masks, then normalize to [0, 1].

    Every mask in the stack contributes proportionally to the sum, so
    pixels covered by more labels receive higher values (density semantics).
    This is the single canonical path for collapsing a label stack into a
    composite mask — all training callers must go through here.
    """
    arr = np.asarray(stack, dtype=np.float32)
    if int(arr.ndim) != 3 or int(arr.shape[0]) <= 0:
        return np.zeros((0, 0) if int(arr.ndim) < 2 else (int(arr.shape[-2]), int(arr.shape[-1])), dtype=np.float32)
    composite = np.sum(arr, axis=0).astype(np.float32, copy=False)
    return _normalize_attention_map(composite, gamma=1.0, blur_kernel=0)


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


def build_label_mask_stack(
    mixed_mask: Any,
    label_vec: Any,
    mask_stack_array: Optional[Any] = None,
    mask_stack_indices: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(label_vec, dtype=np.float32).reshape(-1)
    positive_idx = np.where(y >= 0.5)[0].astype(np.int64)
    mixed = _normalize_attention_map(
        _normalize_mask_array(mixed_mask, height=int(np.asarray(mixed_mask).shape[-2]), width=int(np.asarray(mixed_mask).shape[-1])),
        gamma=0.95,
        blur_kernel=3,
    )
    if mask_stack_array is not None and mask_stack_indices is not None:
        idx_arr = np.asarray(mask_stack_indices, dtype=np.int64).reshape(-1)
        stack_arr = np.asarray(mask_stack_array, dtype=np.float32)
        if int(stack_arr.ndim) == 2:
            stack_arr = stack_arr[None, ...]
        out_stack: List[np.ndarray] = []
        out_idx: List[int] = []
        for si, cls_idx in enumerate(idx_arr.tolist()):
            if si >= int(stack_arr.shape[0]):
                break
            out_stack.append(
                _normalize_attention_map(
                    _normalize_mask_array(
                        stack_arr[int(si)],
                        height=int(mixed.shape[0]),
                        width=int(mixed.shape[1]),
                    ),
                    gamma=0.95,
                    blur_kernel=3,
                )
            )
            out_idx.append(int(cls_idx))
        if len(out_stack) > 0:
            return (
                np.stack(out_stack, axis=0).astype(np.float32, copy=False),
                np.asarray(out_idx, dtype=np.int64),
            )
    if int(positive_idx.size) <= 0:
        return np.zeros((0, int(mixed.shape[0]), int(mixed.shape[1])), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    stack = np.repeat(mixed[None, :, :], int(positive_idx.size), axis=0).astype(np.float32, copy=False)
    return stack, np.asarray(positive_idx, dtype=np.int64)


def build_term_mask_stack_from_image(
    image: Any,
    label_vec: Any,
    idx_to_term: Optional[Dict[int, str]] = None,
    term_mask_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(label_vec, dtype=np.float32).reshape(-1)
    positive_idx = np.where(y >= 0.5)[0].astype(np.int64)
    if int(positive_idx.size) <= 0:
        chw = _image_to_chw01(image)
        return np.zeros((0, int(chw.shape[1]), int(chw.shape[2])), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    chw = _image_to_chw01(image)
    color_maps = _semantic_color_score_maps(chw)
    override_lut = {
        _norm_txt(str(k)): np.asarray(v, dtype=np.float32)
        for k, v in (term_mask_overrides.items() if isinstance(term_mask_overrides, dict) else [])
        if str(k).strip() and v is not None
    }
    per_label_masks: List[np.ndarray] = []
    out_idx: List[int] = []
    fallback = infer_semantic_support_mask(image=chw, terms=[])
    for cls_idx in positive_idx.tolist():
        term = ""
        if isinstance(idx_to_term, dict):
            term = re.sub(r"\s+", " ", str(idx_to_term.get(int(cls_idx), ""))).strip()
        term_key = _norm_txt(term)
        mask = None
        if term_key in override_lut:
            mask = _normalize_attention_map(np.asarray(override_lut[term_key], dtype=np.float32), gamma=0.85, blur_kernel=1)
        elif term_key in color_maps:
            color_score = np.asarray(color_maps.get(term_key), dtype=np.float32)
            thr = max(0.20, float(np.percentile(color_score.reshape(-1), 82)) * 0.75) if int(color_score.size) > 0 else 1.0
            color_exact = (color_score >= float(thr)).astype(np.float32, copy=False) * np.asarray(color_score, dtype=np.float32)
            mask = _normalize_attention_map(color_exact, gamma=0.78, blur_kernel=1)
        if mask is None:
            mask = infer_semantic_support_mask(image=chw, terms=([term] if term else []))
        if float(np.max(mask)) <= 1e-8:
            mask = np.asarray(fallback, dtype=np.float32)
        per_label_masks.append(np.asarray(mask, dtype=np.float32))
        out_idx.append(int(cls_idx))
    return np.stack(per_label_masks, axis=0).astype(np.float32, copy=False), np.asarray(out_idx, dtype=np.int64)


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
    for sample in batch:
        if not isinstance(sample, (tuple, list)) or int(len(sample)) < 5:
            raise RuntimeError("semantic_mask_stack_collate requires 5-tuple samples.")
        xs.append(sample[0])
        ys.append(sample[1])
        ms.append(sample[2])
        stack_t = sample[3] if torch.is_tensor(sample[3]) else torch.as_tensor(sample[3], dtype=torch.float32)
        idx_t = sample[4] if torch.is_tensor(sample[4]) else torch.as_tensor(sample[4], dtype=torch.long)
        mask_stacks.append(stack_t.to(dtype=torch.float32))
        mask_indices.append(idx_t.to(dtype=torch.long))
    return {
        "x": torch.stack(xs, dim=0),
        "y": torch.stack(ys, dim=0),
        "mask": torch.stack(ms, dim=0),
        "mask_stacks": mask_stacks,
        "mask_indices": mask_indices,
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


def _semantic_color_score_maps(chw: np.ndarray) -> Dict[str, np.ndarray]:
    arr = np.clip(np.asarray(chw, dtype=np.float32), 0.0, 1.0)
    if int(arr.ndim) != 3 or int(arr.shape[0]) < 3:
        return {}
    r = np.asarray(arr[0], dtype=np.float32)
    g = np.asarray(arr[1], dtype=np.float32)
    b = np.asarray(arr[2], dtype=np.float32)
    vmax = np.maximum.reduce([r, g, b]).astype(np.float32, copy=False)
    vmin = np.minimum.reduce([r, g, b]).astype(np.float32, copy=False)
    sat = np.clip(vmax - vmin, 0.0, 1.0).astype(np.float32, copy=False)

    def _norm01(x: np.ndarray) -> np.ndarray:
        xx = np.asarray(x, dtype=np.float32)
        hi = float(np.max(xx)) if int(xx.size) > 0 else 0.0
        if hi <= 1e-8:
            return np.zeros_like(xx, dtype=np.float32)
        return np.clip(xx / float(hi), 0.0, 1.0).astype(np.float32, copy=False)

    red = _norm01(np.clip(r - np.maximum(g, b), 0.0, 1.0) * np.clip(sat - 0.05, 0.0, 1.0))
    green = _norm01(np.clip(g - np.maximum(r, b), 0.0, 1.0) * np.clip(sat - 0.05, 0.0, 1.0))
    blue = _norm01(np.clip(b - np.maximum(r, g), 0.0, 1.0) * np.clip(sat - 0.05, 0.0, 1.0))
    yellow = _norm01(np.clip(np.minimum(r, g) - b, 0.0, 1.0) * np.clip(sat - 0.05, 0.0, 1.0))
    cyan = _norm01(np.clip(np.minimum(g, b) - r, 0.0, 1.0) * np.clip(sat - 0.05, 0.0, 1.0))
    magenta = _norm01(np.clip(np.minimum(r, b) - g, 0.0, 1.0) * np.clip(sat - 0.05, 0.0, 1.0))
    brown = _norm01(
        np.clip(r - g, 0.0, 1.0)
        * np.clip(g - b, 0.0, 1.0)
        * np.clip(vmax, 0.15, 0.75)
        * np.clip(0.85 - vmax, 0.0, 1.0)
    )
    # orange: r dominant over b, r > g by margin, g > b (distinguishes from pure red)
    orange = _norm01(
        np.clip(r - b - 0.05, 0.0, 1.0)
        * np.clip(r - g - 0.08, 0.0, 1.0)
        * np.clip(g - b - 0.02, 0.0, 1.0)
        * np.clip(sat - 0.10, 0.0, 1.0)
    )
    black = _norm01(np.clip(0.22 - vmax, 0.0, 1.0))
    white = _norm01(np.clip(vmin - 0.78, 0.0, 1.0) * np.clip(0.20 - sat, 0.0, 1.0))
    gray = _norm01(np.clip(0.18 - sat, 0.0, 1.0) * np.clip(1.0 - np.abs(vmax - 0.5) * 2.2, 0.0, 1.0))
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
        "grey": gray,
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
    for term in ["red", "orange", "green", "blue", "yellow", "cyan", "magenta", "brown", "black", "white", "gray", "grey"]:
        score = np.asarray(maps.get(term), dtype=np.float32)
        if int(score.size) <= 0:
            continue
        frac = float(np.mean(score[valid_mask] >= float(dominance_threshold))) if int(valid_count) > 0 else 0.0
        mean_score = float(np.mean(score[valid_mask])) if int(valid_count) > 0 else 0.0
        if frac >= float(coverage_threshold) or mean_score >= float(max(0.10, dominance_threshold * 0.72)):
            out.append(str(term))
    return normalize_vocab_terms(out)


def infer_semantic_support_mask(image: Any, terms: Optional[Sequence[str]] = None) -> np.ndarray:
    # ============================================================
    # WARNING: THIS FUNCTION IS NOT READY FOR PREGESTATION TRAINING
    # ============================================================
    # The non_color_penalty branch actively suppresses pixels that
    # match colors OTHER than the target — which will corrupt depth-mode
    # rows by penalizing the gray cross occluder (a labeled element).
    # The border-median BG estimation, seed-growing, and object_like /
    # global_like branching all introduce heuristics that can produce
    # false spatial information when the ground-truth geometry is already
    # known exactly from the builder masks.
    # This function is appropriate for the Berkeley/real-image payload path
    # where no ground-truth geometry exists.  Before using it for any
    # synthetic training path, audit the non_color_penalty suppression,
    # the object_like coverage fallback, and the cross-color penalty
    # weighting to ensure they will not degrade clean builder-generated rows.
    # ============================================================
    chw = _image_to_chw01(image)
    h = int(chw.shape[1])
    w = int(chw.shape[2])
    gray = np.clip(np.mean(np.asarray(chw[:3], dtype=np.float32), axis=0), 0.0, 1.0).astype(np.float32, copy=False)
    if int(gray.size) <= 0:
        return np.zeros((int(h), int(w)), dtype=np.float32)

    norm_terms = {_norm_txt(t) for t in (terms or []) if str(t).strip()}
    object_like = bool(
        ("object" in norm_terms)
        or any(str(t).startswith("digit ") for t in norm_terms)
        or any(str(t).startswith("letter ") for t in norm_terms)
        or any(str(t).startswith("pictogram ") for t in norm_terms)
    )
    color_terms = {
        str(t)
        for t in norm_terms
        if str(t)
        in {"red", "green", "blue", "yellow", "cyan", "magenta", "brown", "black", "white", "gray", "grey"}
    }
    localized_like = bool(
        bool(object_like)
        or len(color_terms) > 0
        or any(
            t in {
                "edge",
                "shape",
                "texture",
                "layout",
                "mask",
                "composite",
                "front",
                "back",
                "left",
                "right",
                "top",
                "bottom",
            }
            for t in norm_terms
        )
    )
    global_like = bool(
        (not bool(localized_like))
        and any(
            t in {
                "signal",
                "noise",
                "mixed noise and signal",
                "blur damage",
                "noise damage",
                "dropout damage",
                "quantization damage",
                "stride skew damage",
                "white",
                "black",
                "gray",
                "grey",
                "none",
            }
            for t in norm_terms
        )
    )

    border_parts = [gray[0:1, :], gray[-1:, :], gray[:, 0:1], gray[:, -1:]]
    border = np.concatenate([np.asarray(p, dtype=np.float32).reshape(-1) for p in border_parts], axis=0)
    bg = float(np.median(border)) if int(border.size) > 0 else float(np.median(gray))

    blur = F.avg_pool2d(torch.from_numpy(gray[None, None, ...]), kernel_size=5, stride=1, padding=2)[0, 0].cpu().numpy()
    diff_bg = np.abs(gray - float(bg)).astype(np.float32, copy=False)
    diff_local = np.abs(gray - np.asarray(blur, dtype=np.float32)).astype(np.float32, copy=False)
    gy, gx = np.gradient(gray.astype(np.float32, copy=False))
    grad = np.sqrt((gx * gx) + (gy * gy)).astype(np.float32, copy=False)

    def _norm_map(v: np.ndarray) -> np.ndarray:
        vmax = float(np.max(v)) if int(v.size) > 0 else 0.0
        if vmax <= 1e-8:
            return np.zeros_like(v, dtype=np.float32)
        return np.clip(np.asarray(v, dtype=np.float32) / float(vmax), 0.0, 1.0).astype(np.float32, copy=False)

    score = np.maximum.reduce([
        _norm_map(diff_bg),
        0.85 * _norm_map(diff_local),
        0.65 * _norm_map(grad),
    ]).astype(np.float32, copy=False)
    if len(color_terms) > 0:
        color_maps = _semantic_color_score_maps(chw)
        color_parts = [np.asarray(color_maps.get(str(term)), dtype=np.float32) for term in sorted(color_terms) if str(term) in color_maps]
        color_parts = [x for x in color_parts if int(x.size) > 0]
        if len(color_parts) > 0:
            color_score = np.maximum.reduce(color_parts).astype(np.float32, copy=False)
            color_score = _normalize_attention_map(color_score, gamma=0.82, blur_kernel=3)
            non_color_penalty = np.ones_like(color_score, dtype=np.float32)
            other_color_parts = [
                np.asarray(color_maps.get(str(term)), dtype=np.float32)
                for term in sorted(color_maps.keys())
                if str(term) not in color_terms
            ]
            other_color_parts = [x for x in other_color_parts if int(x.size) > 0]
            if len(other_color_parts) > 0:
                other_score = _normalize_attention_map(np.maximum.reduce(other_color_parts), gamma=1.0, blur_kernel=3)
                non_color_penalty = np.clip(1.0 - (0.55 * other_score), 0.15, 1.0).astype(np.float32, copy=False)
            score = _blend_attention_maps(
                [score * non_color_penalty, color_score * non_color_penalty],
                weights=[0.28, 1.45],
                gamma=0.82,
            ).astype(np.float32, copy=False)
    score_t = F.avg_pool2d(torch.from_numpy(score[None, None, ...]), kernel_size=5, stride=1, padding=2)
    score = _normalize_attention_map(score_t[0, 0].cpu().numpy(), gamma=0.92, blur_kernel=5)

    thr = max(0.16, float(np.percentile(score.reshape(-1), 72)) * 0.70)
    seed = (score >= float(thr)).astype(np.float32, copy=False)
    seed_t = F.avg_pool2d(torch.from_numpy(seed[None, None, ...]), kernel_size=5, stride=1, padding=2)
    seed = np.asarray(seed_t[0, 0].cpu().numpy(), dtype=np.float32)
    seed = _normalize_attention_map(seed, gamma=0.85, blur_kernel=3)

    coverage = float(np.mean(seed >= 0.40)) if int(seed.size) > 0 else 0.0
    if bool(object_like) and (coverage <= 0.01 or coverage >= 0.97):
        hi = float(np.percentile(gray.reshape(-1), 75))
        lo = float(np.percentile(gray.reshape(-1), 25))
        if abs(float(hi) - float(bg)) >= abs(float(lo) - float(bg)):
            refine = (gray >= float((hi + bg) * 0.5)).astype(np.float32, copy=False)
        else:
            refine = (gray <= float((lo + bg) * 0.5)).astype(np.float32, copy=False)
        refine_t = F.avg_pool2d(torch.from_numpy(refine[None, None, ...]), kernel_size=5, stride=1, padding=2)
        seed = _normalize_attention_map(refine_t[0, 0].cpu().numpy(), gamma=0.82, blur_kernel=5)
        coverage = float(np.mean(seed >= 0.40)) if int(seed.size) > 0 else 0.0

    if coverage <= 0.01:
        score_thr = max(0.10, float(np.percentile(score.reshape(-1), 55)) * 0.50)
        fallback = _normalize_attention_map(score * (score >= float(score_thr)).astype(np.float32, copy=False), gamma=0.95, blur_kernel=5)
        if float(np.max(fallback)) <= 1e-8:
            fallback = _normalize_attention_map(score, gamma=1.0, blur_kernel=7)
        seed = fallback

    if bool(object_like):
        mask = _blend_attention_maps([score, seed], weights=[0.65, 1.15], gamma=0.88)
    elif bool(global_like):
        local_grad = _normalize_attention_map((0.55 * diff_local) + (0.85 * grad), gamma=1.05, blur_kernel=5)
        mask = _blend_attention_maps([score, local_grad], weights=[0.70, 1.20], gamma=1.08)
    else:
        mask = _blend_attention_maps([score, seed], weights=[0.90, 0.75], gamma=0.98)

    if bool(object_like) and float(np.mean(mask >= 0.45)) <= 0.02:
        mask = _blend_attention_maps([mask, seed], weights=[0.65, 1.35], gamma=0.85)

    return _normalize_attention_map(mask, gamma=(0.82 if bool(object_like) else 1.05), blur_kernel=5)


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
    ds_use: Dataset = manifest.dataset if int(picks.size) == int(n) else torch.utils.data.Subset(manifest.dataset, picks.tolist())
    collate_fn = semantic_mask_stack_collate if bool(getattr(manifest.dataset, "use_semantic_mask_stack_collate", False)) else None
    loader_num_workers = effective_dataloader_num_workers(
        num_workers=manifest.num_workers,
        device_type=manifest.device_type,
    )
    loader = DataLoader(
        ds_use,
        batch_size=max(1, int(manifest.batch_size)),
        shuffle=False,
        num_workers=int(loader_num_workers),
        pin_memory=(str(manifest.device_type).strip().lower() == "cuda"),
        drop_last=False,
        collate_fn=collate_fn,
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


def augment_bootstrap_chw01(
    img: np.ndarray,
    seed: int,
    return_terms: bool = False,
    return_touch_mask: bool = False,
    return_term_masks: bool = False,
) -> Any:
    arr = np.asarray(img, dtype=np.float32)
    if int(arr.ndim) == 3 and int(arr.shape[0]) == 3:
        x = np.asarray(arr, dtype=np.float32, order="C")
    elif int(arr.ndim) == 3 and int(arr.shape[2]) == 3:
        x = np.transpose(arr, (2, 0, 1)).astype(np.float32, copy=False)
    elif int(arr.ndim) == 2:
        x = np.repeat(arr[None, :, :], 3, axis=0).astype(np.float32, copy=False)
    else:
        raise RuntimeError(f"Unsupported bootstrap row shape for augmentation: {tuple(arr.shape)}")
    x = np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)
    c, h, w = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    rng = np.random.default_rng(int(seed))
    applied_terms: List[str] = []
    touched = np.zeros((int(h), int(w)), dtype=np.float32)
    term_masks: Dict[str, np.ndarray] = {}

    def _accumulate_touch(delta: Any, scale: float = 1.0, gamma: float = 1.0):
        nonlocal touched
        norm = _normalize_attention_map(delta, gamma=float(gamma), blur_kernel=3)
        if float(np.max(norm)) <= 1e-8:
            return
        touched = np.asarray(touched, dtype=np.float32) + (float(scale) * norm)
        return np.asarray(norm, dtype=np.float32)

    def _accumulate_term_mask(terms: Sequence[str], delta: Any, scale: float = 1.0, gamma: float = 1.0):
        norm = _accumulate_touch(delta, scale=float(scale), gamma=float(gamma))
        if norm is None or float(np.max(norm)) <= 1e-8:
            return
        for term in normalize_vocab_terms([str(t) for t in list(terms)]):
            tk = _norm_txt(term)
            prev_mask = np.asarray(term_masks.get(tk, np.zeros((int(h), int(w)), dtype=np.float32)), dtype=np.float32)
            term_masks[tk] = np.maximum(prev_mask, np.asarray(norm, dtype=np.float32)).astype(np.float32, copy=False)

    def _mark_diff(prev_x: np.ndarray, next_x: np.ndarray, scale: float = 1.0):
        prev_g = np.mean(np.asarray(prev_x, dtype=np.float32), axis=0)
        next_g = np.mean(np.asarray(next_x, dtype=np.float32), axis=0)
        diff = np.abs(next_g - prev_g).astype(np.float32, copy=False)
        return _accumulate_touch(diff, scale=float(scale), gamma=0.95)

    prev = np.asarray(x, dtype=np.float32).copy()
    shift_x = int(rng.integers(-max(1, w // 14), max(1, w // 14) + 1))
    shift_y = int(rng.integers(-max(1, h // 14), max(1, h // 14) + 1))
    if shift_x != 0:
        x = np.roll(x, shift=shift_x, axis=2)
    if shift_y != 0:
        x = np.roll(x, shift=shift_y, axis=1)
    if shift_x != 0 or shift_y != 0:
        _mark_diff(prev, x, scale=0.7)
    prev = np.asarray(x, dtype=np.float32).copy()
    if float(rng.random()) < 0.45:
        x = np.flip(x, axis=2).copy()
        _mark_diff(prev, x, scale=0.4)
        prev = np.asarray(x, dtype=np.float32).copy()
    if float(rng.random()) < 0.15:
        x = np.flip(x, axis=1).copy()
        _mark_diff(prev, x, scale=0.4)
        prev = np.asarray(x, dtype=np.float32).copy()

    if float(rng.random()) < 0.55:
        k = int(rng.choice(np.asarray([3, 5, 7], dtype=np.int32)))
        t = torch.from_numpy(np.asarray(x, dtype=np.float32)[None, ...])
        t = F.avg_pool2d(t, kernel_size=int(k), stride=1, padding=int(k // 2))
        x_blur = np.asarray(t[0].cpu().numpy(), dtype=np.float32)
        _accumulate_term_mask(["blur damage", "signal"], np.abs(np.mean(x_blur[:3], axis=0) - np.mean(prev[:3], axis=0)), scale=0.8, gamma=0.95)
        x = x_blur
        applied_terms.extend(normalize_vocab_terms(["blur damage", "signal"]))
        prev = np.asarray(x, dtype=np.float32).copy()
    if float(rng.random()) < 0.45:
        odd_shift = int(rng.integers(1, 8))
        x[:, 1::2, :] = np.roll(x[:, 1::2, :], shift=odd_shift, axis=2)
        stripe = np.zeros((h, w), dtype=np.float32)
        stripe[1::2, :] = 1.0
        _accumulate_term_mask(["stride skew damage", "signal"], stripe, scale=0.7, gamma=1.15)
        applied_terms.extend(normalize_vocab_terms(["stride skew damage", "signal"]))
        prev = np.asarray(x, dtype=np.float32).copy()
    if float(rng.random()) < 0.35:
        keep = float(rng.uniform(0.76, 0.96))
        keep_mask = (rng.random((h, w), dtype=np.float32) < keep).astype(np.float32, copy=False)
        x = x * keep_mask[None, :, :]
        _accumulate_term_mask(["dropout damage", "signal"], 1.0 - keep_mask, scale=1.0, gamma=0.90)
        applied_terms.extend(normalize_vocab_terms(["dropout damage", "signal"]))
        prev = np.asarray(x, dtype=np.float32).copy()
    if float(rng.random()) < 0.40:
        lv = int(rng.choice(np.asarray([4, 6, 8, 12], dtype=np.int32)))
        x_quant = np.round(x * float(lv - 1)) / float(max(1, lv - 1))
        _accumulate_term_mask(["quantization damage", "signal"], np.abs(np.mean(x_quant[:3], axis=0) - np.mean(prev[:3], axis=0)), scale=0.7, gamma=0.95)
        x = x_quant
        applied_terms.extend(normalize_vocab_terms(["quantization damage", "signal"]))
        prev = np.asarray(x, dtype=np.float32).copy()

    x01 = _unit_interval_axis(int(w))[None, :]
    y01 = _unit_interval_axis(int(h))[:, None]
    fx = float(rng.uniform(1.2, 8.5))
    fy = float(rng.uniform(1.0, 7.5))
    ph = float(rng.uniform(0.0, 2.0 * np.pi))
    phase = np.empty((int(h), int(w)), dtype=np.float32)
    np.multiply(y01, np.float32(fy), out=phase)
    phase += np.float32(fx) * x01
    phase *= np.float32(2.0 * np.pi)
    phase += np.float32(ph)
    np.sin(phase, out=phase)
    phase *= np.float32(0.5)
    phase += np.float32(0.5)
    sig = phase
    noi = rng.random((h, w), dtype=np.float32)
    if float(rng.random()) < 0.5:
        mix_a = float(rng.uniform(0.18, 0.62))
        mix = np.clip((mix_a * sig) + ((1.0 - mix_a) * noi), 0.0, 1.0).astype(np.float32, copy=False)
    else:
        bits = int(rng.integers(2, 7))
        s_u16 = np.round(np.clip(sig, 0.0, 1.0) * 65535.0).astype(np.uint16, copy=False)
        n_u16 = np.round(np.clip(noi, 0.0, 1.0) * 65535.0).astype(np.uint16, copy=False)
        payload = ((n_u16 >> int(16 - bits)) & np.uint16((1 << bits) - 1)).astype(np.uint16, copy=False)
        keep_mask = np.uint16(0xFFFF ^ ((1 << bits) - 1))
        mix = np.clip(((s_u16 & keep_mask) | payload).astype(np.float32) / 65535.0, 0.0, 1.0).astype(np.float32, copy=False)
    blend = float(rng.uniform(0.08, 0.28))
    x_mix = np.clip(((1.0 - blend) * x) + (blend * mix[None, :, :]), 0.0, 1.0)
    signal_support = np.ones((int(h), int(w)), dtype=np.float32)
    noise_support = np.ones((int(h), int(w)), dtype=np.float32)
    _accumulate_term_mask(["signal"], signal_support, scale=1.0, gamma=1.0)
    _accumulate_term_mask(["noise", "mixed noise and signal"], noise_support, scale=1.0, gamma=1.0)
    _accumulate_touch(np.abs(np.mean(x_mix[:3], axis=0) - np.mean(prev[:3], axis=0)), scale=0.6, gamma=0.95)
    x = x_mix
    applied_terms.extend(["mixed noise and signal", "noise", "signal"])
    prev = np.asarray(x, dtype=np.float32).copy()

    if float(rng.random()) < 0.65:
        std = float(rng.uniform(0.01, 0.08))
        noise_delta = (std * rng.standard_normal((c, h, w), dtype=np.float32)).astype(np.float32, copy=False)
        x_noise = np.clip(x + noise_delta, 0.0, 1.0)
        _accumulate_term_mask(["noise damage", "noise", "mixed noise and signal", "signal"], np.mean(np.abs(noise_delta), axis=0), scale=0.9, gamma=0.90)
        x = x_noise
        applied_terms.extend(normalize_vocab_terms(["noise damage", "noise", "mixed noise and signal", "signal"]))

    prev = np.asarray(x, dtype=np.float32).copy()
    if int(c) >= 3 and float(rng.random()) < 0.45:
        color_profiles = {
            "red": np.asarray([1.00, 0.18, 0.18], dtype=np.float32),
            "green": np.asarray([0.18, 1.00, 0.18], dtype=np.float32),
            "blue": np.asarray([0.18, 0.18, 1.00], dtype=np.float32),
            "yellow": np.asarray([1.00, 1.00, 0.20], dtype=np.float32),
            "cyan": np.asarray([0.18, 1.00, 1.00], dtype=np.float32),
            "magenta": np.asarray([1.00, 0.18, 1.00], dtype=np.float32),
            "brown": np.asarray([0.72, 0.44, 0.22], dtype=np.float32),
            "white": np.asarray([1.00, 1.00, 1.00], dtype=np.float32),
            "black": np.asarray([0.08, 0.08, 0.08], dtype=np.float32),
            "gray": np.asarray([0.55, 0.55, 0.55], dtype=np.float32),
        }
        color_key = str(rng.choice(np.asarray(list(color_profiles.keys()), dtype=object))).strip().lower()
        profile = color_profiles.get(color_key, np.asarray([1.0, 1.0, 1.0], dtype=np.float32))[:, None, None]
        luma = np.mean(np.asarray(x[:3], dtype=np.float32), axis=0, keepdims=True)
        tinted = np.clip(profile * np.clip(0.25 + (0.90 * luma), 0.0, 1.0), 0.0, 1.0).astype(np.float32, copy=False)
        alpha = float(rng.uniform(0.24, 0.58))
        x_tinted = np.clip(((1.0 - alpha) * x) + (alpha * tinted), 0.0, 1.0)
        _accumulate_term_mask([str(color_key)], np.abs(np.mean(x_tinted[:3], axis=0) - np.mean(prev[:3], axis=0)), scale=0.85, gamma=0.90)
        x = x_tinted
        applied_terms.extend(normalize_vocab_terms([str(color_key)]))
        if str(color_key) == "white":
            applied_terms.extend(normalize_vocab_terms(["bright"]))
        elif str(color_key) == "black":
            applied_terms.extend(normalize_vocab_terms(["dark"]))

    prev = np.asarray(x, dtype=np.float32).copy()
    if float(rng.random()) < 0.32:
        gx = np.zeros((h, w), dtype=np.float32)
        gy = np.zeros((h, w), dtype=np.float32)
        gray = np.mean(np.asarray(x[:3], dtype=np.float32), axis=0)
        gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
        edge = np.sqrt((gx * gx) + (gy * gy)).astype(np.float32, copy=False)
        if float(np.max(edge)) > 1e-8:
            edge = edge / float(np.max(edge))
        edge_rgb = np.repeat(edge[None, :, :], c, axis=0).astype(np.float32, copy=False)
        alpha = float(rng.uniform(0.20, 0.46))
        x_edge = np.clip(((1.0 - alpha) * x) + (alpha * edge_rgb), 0.0, 1.0)
        _accumulate_term_mask(["edge", "shape", "rough"], edge, scale=0.80, gamma=0.90)
        x = x_edge
        applied_terms.extend(normalize_vocab_terms(["edge", "shape", "rough"]))

    prev = np.asarray(x, dtype=np.float32).copy()
    if float(rng.random()) < 0.36:
        fx_t = float(rng.uniform(4.0, 16.0))
        fy_t = float(rng.uniform(4.0, 16.0))
        ph_t = float(rng.uniform(0.0, 2.0 * np.pi))
        tex = 0.5 + (0.25 * np.sin((2.0 * np.pi * ((fx_t * x01) + (0.35 * fy_t * y01))) + ph_t))
        tex += 0.25 * np.sin((2.0 * np.pi * ((0.45 * fx_t * x01) - (fy_t * y01))) + (ph_t * 0.7))
        tex = np.clip(tex, 0.0, 1.0).astype(np.float32, copy=False)
        tex_rgb = np.repeat(tex[None, :, :], c, axis=0).astype(np.float32, copy=False)
        alpha = float(rng.uniform(0.18, 0.42))
        x_tex = np.clip(x * (0.78 + (0.55 * tex_rgb)) + (alpha * tex_rgb), 0.0, 1.0)
        tex_terms = ["texture", "pattern"]
        if float(np.std(tex)) >= 0.18:
            tex_terms.append("rough")
        else:
            tex_terms.append("smooth")
        _accumulate_term_mask(tex_terms, tex, scale=0.70, gamma=0.90)
        x = x_tex
        applied_terms.extend(normalize_vocab_terms(["texture", "pattern"]))
        if float(np.std(tex)) >= 0.18:
            applied_terms.extend(normalize_vocab_terms(["rough"]))
        else:
            applied_terms.extend(normalize_vocab_terms(["smooth"]))

    x_out = np.asarray(x, dtype=np.float32)
    touched = _normalize_attention_map(touched, gamma=1.05, blur_kernel=5)
    if bool(return_terms) and bool(return_touch_mask) and bool(return_term_masks):
        term_masks = {str(k): _normalize_attention_map(v, gamma=0.85, blur_kernel=1) for k, v in term_masks.items() if float(np.max(v)) > 1e-8}
        return x_out, normalize_vocab_terms(applied_terms), touched, term_masks
    if bool(return_terms) and bool(return_touch_mask):
        return x_out, normalize_vocab_terms(applied_terms), touched
    if bool(return_terms):
        return x_out, normalize_vocab_terms(applied_terms)
    if bool(return_touch_mask):
        return x_out, touched
    return x_out


def _cache_encode_image_u8(img: np.ndarray) -> np.ndarray:
    arr = np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0)
    return np.round(arr * 255.0).astype(np.uint8, copy=False)


def _cache_decode_image_u8(img: np.ndarray) -> np.ndarray:
    return (np.asarray(img, dtype=np.float32) / 255.0).astype(np.float32, copy=False)


def _cache_encode_mask_u8(mask: np.ndarray) -> np.ndarray:
    arr = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    return np.round(arr * 255.0).astype(np.uint8, copy=False)


def _cache_decode_mask_u8(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask, dtype=np.float32) / 255.0).astype(np.float32, copy=False)


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

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_cache_images_mm"] = None
        state["_cache_targets_mm"] = None
        state["_cache_masks_mm"] = None
        state["_cache_mask_stacks_mm"] = None
        state["_cache_mask_indices_mm"] = None
        state["_cache_mask_offsets_mm"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._cache_images_mm = None
        self._cache_targets_mm = None
        self._cache_masks_mm = None
        self._cache_mask_stacks_mm = None
        self._cache_mask_indices_mm = None
        self._cache_mask_offsets_mm = None

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
        for idx in range(int(desired_rows)):
            img_np, tgt_np, mask_np, mask_stack_np, mask_idx_np = self._materialize_numpy_row(int(idx))
            images_rows.append(_cache_encode_image_u8(img_np))
            target_rows.append(_cache_encode_target_f16(tgt_np))
            if bool(self.return_masks):
                mask_rows.append(_cache_encode_mask_u8(mask_np))
            if bool(self.return_mask_stack):
                stack_u8 = _cache_encode_mask_u8(mask_stack_np) if int(np.asarray(mask_stack_np).size) > 0 else np.zeros((0, int(img_np.shape[1]), int(img_np.shape[2])), dtype=np.uint8)
                idx_np = np.asarray(mask_idx_np, dtype=np.int64).reshape(-1)
                if int(stack_u8.shape[0]) != int(idx_np.size):
                    raise RuntimeError(f"{str(self.dataset_name)} mask stack cache mismatch: stack_rows={int(stack_u8.shape[0])} idx={int(idx_np.size)}")
                mask_stack_rows.append(np.asarray(stack_u8, dtype=np.uint8))
                mask_index_rows.append(idx_np)
                mask_offsets.append(int(mask_offsets[-1] + int(stack_u8.shape[0])))
        if int(len(images_rows)) <= 0:
            return 0
        images_np = np.stack(images_rows, axis=0).astype(np.uint8, copy=False)
        targets_np = np.stack(target_rows, axis=0).astype(np.float16, copy=False)
        masks_np = np.stack(mask_rows, axis=0).astype(np.uint8, copy=False) if bool(self.return_masks) else None
        if bool(self.return_mask_stack):
            if int(mask_offsets[-1]) > 0:
                mask_stacks_np = np.concatenate(mask_stack_rows, axis=0).astype(np.uint8, copy=False)
                mask_indices_np = np.concatenate(mask_index_rows, axis=0).astype(np.int64, copy=False)
            else:
                mask_stacks_np = np.zeros((0, int(images_np.shape[2]), int(images_np.shape[3])), dtype=np.uint8)
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
            if bool(self.return_mask_stack):
                exact_term_masks = dict(aug_term_masks) if isinstance(aug_term_masks, dict) else {}
                base_signal_terms = ["signal", "object"]
                for base_term in base_signal_terms:
                    if _norm_txt(base_term) not in exact_term_masks and float(np.max(mask_base)) > 1e-8:
                        exact_term_masks[_norm_txt(base_term)] = np.asarray(mask_base, dtype=np.float32)
                if "mixed noise and signal" in aug_term_set and float(np.max(mask_base)) > 1e-8:
                    exact_term_masks["mixed noise and signal"] = np.maximum(
                        np.asarray(exact_term_masks.get("noise", touched), dtype=np.float32),
                        np.asarray(mask_base, dtype=np.float32),
                    ).astype(np.float32, copy=False)
                mask_stack_np, mask_idx_np = build_term_mask_stack_from_image(
                    image=img,
                    label_vec=tgt,
                    idx_to_term=self.idx_to_term,
                    term_mask_overrides=exact_term_masks,
                )
                if int(mask_stack_np.shape[0]) > 0 and int(mask_idx_np.size) > 0:
                    mixed_from_stack = _composite_mask_stack(mask_stack_np)
                    mask = _blend_attention_maps(
                        [np.asarray(mask, dtype=np.float32), np.asarray(mixed_from_stack, dtype=np.float32)],
                        weights=[0.55, 1.10],
                        gamma=0.92,
                    ).astype(np.float32, copy=False)
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
            mask = _cache_decode_mask_u8(self._cache_masks_mm[int(index)])
        else:
            mask = np.zeros((int(img.shape[1]), int(img.shape[2])), dtype=np.float32)
        if bool(self.return_mask_stack):
            if self._cache_mask_offsets_mm is None or self._cache_mask_indices_mm is None or self._cache_mask_stacks_mm is None:
                mask_stack_np, mask_idx_np = build_term_mask_stack_from_image(
                    image=img,
                    label_vec=tgt,
                    idx_to_term=self.idx_to_term,
                )
                if int(mask_stack_np.shape[0]) <= 0 or int(mask_idx_np.size) <= 0:
                    mask_stack_np, mask_idx_np = build_label_mask_stack(mixed_mask=mask, label_vec=tgt)
            else:
                start = int(self._cache_mask_offsets_mm[int(index)])
                stop = int(self._cache_mask_offsets_mm[int(index) + 1])
                if int(stop) > int(start):
                    mask_stack_np = _cache_decode_mask_u8(self._cache_mask_stacks_mm[int(start):int(stop)])
                    mask_idx_np = np.asarray(self._cache_mask_indices_mm[int(start):int(stop)], dtype=np.int64)
                else:
                    mask_stack_np = np.zeros((0, int(mask.shape[0]), int(mask.shape[1])), dtype=np.float32)
                    mask_idx_np = np.zeros((0,), dtype=np.int64)
                if int(mask_stack_np.shape[0]) <= 0 or int(mask_idx_np.size) <= 0:
                    mask_stack_np, mask_idx_np = build_label_mask_stack(mixed_mask=mask, label_vec=tgt)
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
            tgt = np.asarray(self.targets[int(idx)], dtype=np.float32).reshape(-1)
            active_terms = [
                str(self.idx_to_term[int(j)])
                for j in np.where(tgt >= 0.5)[0].astype(np.int64).tolist()
                if int(j) in self.idx_to_term
            ]
            cached_mask = infer_semantic_support_mask(self.images[int(idx)], terms=active_terms)
            self.base_masks[int(idx)] = np.asarray(cached_mask, dtype=np.float32)
        if bool(self.return_mask_stack) and (cached_stack is None or cached_stack_idx is None):
            tgt = np.asarray(self.targets[int(idx)], dtype=np.float32).reshape(-1)
            cached_stack, cached_stack_idx = build_term_mask_stack_from_image(
                image=self.images[int(idx)],
                label_vec=tgt,
                idx_to_term=self.idx_to_term,
            )
            self.base_mask_stacks[int(idx)] = np.asarray(cached_stack, dtype=np.float32)
            self.base_mask_stack_indices[int(idx)] = np.asarray(cached_stack_idx, dtype=np.int64)
        mask_np = np.asarray(self.base_masks[int(idx)], dtype=np.float32)
        if bool(self.return_mask_stack):
            stack_np = np.asarray(self.base_mask_stacks[int(idx)], dtype=np.float32)
            stack_idx_np = np.asarray(self.base_mask_stack_indices[int(idx)], dtype=np.int64)
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
                    mask_stack_np, mask_idx_np = build_term_mask_stack_from_image(
                        image=img,
                        label_vec=tgt,
                            idx_to_term=self.idx_to_term,
                    )
                if int(mask_stack_np.shape[0]) <= 0 or int(mask_idx_np.size) <= 0:
                    mask_stack_np, mask_idx_np = build_label_mask_stack(mixed_mask=mask, label_vec=tgt)
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
        cfg = dict(degrade_config) if isinstance(degrade_config, dict) else {}
        self.degrade_config = {
            "blur_prob": float(cfg.get("blur_prob", 0.55)),
            "stride_skew_prob": float(cfg.get("stride_skew_prob", 0.40)),
            "dropout_prob": float(cfg.get("dropout_prob", 0.25)),
            "quantization_prob": float(cfg.get("quantization_prob", 0.30)),
            "noise_prob": float(cfg.get("noise_prob", 0.55)),
            "noise_std_min": float(cfg.get("noise_std_min", 0.01)),
            "noise_std_max": float(cfg.get("noise_std_max", 0.08)),
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

    def _load_mask01(self, row: SemanticDiskRow, h: int, w: int, image_rgb01: Optional[np.ndarray] = None) -> np.ndarray:
        inferred_mask = infer_semantic_support_mask(
            image=(image_rgb01 if image_rgb01 is not None else np.zeros((3, int(h), int(w)), dtype=np.float32)),
            terms=row.terms,
        )

        def _exact_mask(base_mask: Any) -> np.ndarray:
            return _normalize_attention_map(
                _normalize_mask_array(base_mask, height=int(h), width=int(w)),
                gamma=0.95,
                blur_kernel=1,
            )

        if row.mask_array is not None:
            arr = np.asarray(row.mask_array, dtype=np.float32)
            if int(arr.ndim) == 3:
                arr = np.mean(arr, axis=0).astype(np.float32, copy=False)
            if int(arr.shape[0]) != int(h) or int(arr.shape[1]) != int(w):
                arr = np.asarray(
                    TF.resize(Image.fromarray(np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8), mode="L"), [int(h), int(w)], interpolation=InterpolationMode.NEAREST),
                    dtype=np.float32,
                )
            return _exact_mask(arr / 255.0 if float(np.max(arr)) > 1.0 else arr)
        cached_bundle = _load_row_mask_cache_bundle(getattr(row, "mask_cache_file", ""))
        if isinstance(cached_bundle, dict) and cached_bundle.get("mixed_mask", None) is not None:
            arr = np.asarray(cached_bundle.get("mixed_mask"), dtype=np.float32)
            if int(arr.ndim) == 3:
                arr = np.mean(arr, axis=0).astype(np.float32, copy=False)
            return _exact_mask(arr)
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
                        if int(arr.shape[1]) != int(w) or int(arr.shape[0]) != int(h):
                            arr = np.asarray(
                                TF.resize(Image.fromarray((arr * 255.0).astype(np.uint8), mode="L"), [int(h), int(w)], interpolation=InterpolationMode.NEAREST),
                                dtype=np.float32,
                            )
                        return _exact_mask(arr / 255.0)
                    # fall through to next branch if seg is None
                with Image.open(str(mask_path)) as im:
                    gray = im.convert("L")
                    if int(gray.size[0]) != int(w) or int(gray.size[1]) != int(h):
                        gray = gray.resize((int(w), int(h)), resample=Image.NEAREST)
                    arr = np.asarray(gray, dtype=np.float32)
                return _exact_mask(arr / 255.0)
        if isinstance(row.layout, dict):
            return _exact_mask(build_layout_mask(row.layout, height=int(h), width=int(w)))
        return _normalize_attention_map(inferred_mask, gamma=0.92, blur_kernel=5)

    def _apply_degrade(self, x: np.ndarray, mask: Optional[np.ndarray], idx: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not bool(self.degrade):
            return x, mask
        rng = np.random.default_rng(int(self.degrade_seed) + (int(idx) * 104729))
        out = np.asarray(x, dtype=np.float32).copy()
        mask_out = None if mask is None else np.asarray(mask, dtype=np.float32).copy()
        c, h, w = int(out.shape[0]), int(out.shape[1]), int(out.shape[2])
        touch = np.zeros((int(h), int(w)), dtype=np.float32)

        def _accumulate_touch(delta: Any, scale: float = 1.0, gamma: float = 1.0):
            nonlocal touch
            norm = _normalize_attention_map(delta, gamma=float(gamma), blur_kernel=3)
            if float(np.max(norm)) <= 1e-8:
                return
            touch = np.asarray(touch, dtype=np.float32) + (float(scale) * norm)

        def _mark_diff(prev_x: np.ndarray, next_x: np.ndarray, scale: float = 1.0):
            prev_g = np.mean(np.asarray(prev_x, dtype=np.float32), axis=0)
            next_g = np.mean(np.asarray(next_x, dtype=np.float32), axis=0)
            _accumulate_touch(np.abs(next_g - prev_g).astype(np.float32, copy=False), scale=float(scale), gamma=0.95)

        if float(rng.random()) < float(self.degrade_config["blur_prob"]):
            prev = np.asarray(out, dtype=np.float32).copy()
            k = int(rng.choice(np.asarray([3, 5, 7], dtype=np.int32)))
            t = torch.from_numpy(out[None, ...])
            t = F.avg_pool2d(t, kernel_size=int(k), stride=1, padding=int(k // 2))
            out = np.asarray(t[0].cpu().numpy(), dtype=np.float32)
            _mark_diff(prev, out, scale=0.85)
        if float(rng.random()) < float(self.degrade_config["stride_skew_prob"]):
            prev = np.asarray(out, dtype=np.float32).copy()
            odd_shift = int(rng.integers(1, 8))
            out[:, 1::2, :] = np.roll(out[:, 1::2, :], shift=odd_shift, axis=2)
            stripe = np.zeros((h, w), dtype=np.float32)
            stripe[1::2, :] = 1.0
            _mark_diff(prev, out, scale=0.75)
            _accumulate_touch(stripe, scale=0.35, gamma=1.10)
            if mask_out is not None:
                mask_out[1::2, :] = np.roll(mask_out[1::2, :], shift=odd_shift, axis=1)
        if float(rng.random()) < float(self.degrade_config["dropout_prob"]):
            keep = float(rng.uniform(0.78, 0.96))
            keep_mask = (rng.random((h, w), dtype=np.float32) < keep).astype(np.float32, copy=False)
            out = out * keep_mask[None, :, :]
            _accumulate_touch(1.0 - keep_mask, scale=1.0, gamma=0.90)
        if float(rng.random()) < float(self.degrade_config["quantization_prob"]):
            prev = np.asarray(out, dtype=np.float32).copy()
            lv = int(rng.choice(np.asarray([4, 6, 8, 12], dtype=np.int32)))
            out = np.round(out * float(lv - 1)) / float(max(1, lv - 1))
            _mark_diff(prev, out, scale=0.75)
        if float(rng.random()) < float(self.degrade_config["noise_prob"]):
            prev = np.asarray(out, dtype=np.float32).copy()
            std = float(rng.uniform(float(self.degrade_config["noise_std_min"]), float(self.degrade_config["noise_std_max"])))
            out = out + (std * rng.standard_normal((c, h, w), dtype=np.float32)).astype(np.float32, copy=False)
            _mark_diff(prev, out, scale=0.90)
        out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)
        if mask_out is not None:
            touch = _normalize_attention_map(touch, gamma=1.05, blur_kernel=5)
            if float(np.max(touch)) > 1e-8:
                mask_out = _blend_attention_maps(
                    [np.asarray(mask_out, dtype=np.float32), np.asarray(touch, dtype=np.float32)],
                    weights=[0.55, 1.15],
                    gamma=0.98,
                ).astype(np.float32, copy=False)
            else:
                mask_out = _normalize_attention_map(mask_out, gamma=0.95, blur_kernel=3)
        return out, mask_out

    def __getitem__(self, index: int):
        row = self.rows[int(index)]
        x = self._load_rgb01(row.image_path)
        mask_np = None
        if bool(self.return_masks):
            mask_np = self._load_mask01(row, h=int(x.shape[1]), w=int(x.shape[2]), image_rgb01=x)
        x, mask_np = self._apply_degrade(x=x, mask=mask_np, idx=int(index))
        y = np.asarray(row.label_vec, dtype=np.float32).reshape(-1)
        if bool(self.return_masks):
            mask_stack_np = np.zeros((0, int(x.shape[1]), int(x.shape[2])), dtype=np.float32)
            mask_idx_np = np.zeros((0,), dtype=np.int64)
            if bool(self.return_mask_stack):
                cached_bundle = _load_row_mask_cache_bundle(getattr(row, "mask_cache_file", ""))
                cached_stack_array = row.mask_stack_array
                cached_stack_indices = row.mask_stack_indices
                if isinstance(cached_bundle, dict):
                    if cached_stack_array is None:
                        cached_stack_array = cached_bundle.get("mask_stack")
                    if cached_stack_indices is None:
                        cached_stack_indices = cached_bundle.get("mask_indices")
                if cached_stack_array is not None and cached_stack_indices is not None:
                    mask_stack_np, mask_idx_np = build_label_mask_stack(
                        mixed_mask=np.asarray(mask_np, dtype=np.float32),
                        label_vec=y,
                        mask_stack_array=cached_stack_array,
                        mask_stack_indices=cached_stack_indices,
                    )
                if int(mask_stack_np.shape[0]) <= 0 or int(mask_idx_np.size) <= 0:
                    inferred_stack_np, inferred_idx_np = build_term_mask_stack_from_image(
                        image=x,
                        label_vec=y,
                    )
                    if int(inferred_stack_np.shape[0]) > 0 and int(inferred_idx_np.size) > 0:
                        mask_stack_np, mask_idx_np = inferred_stack_np, inferred_idx_np
                        mixed_from_stack = _composite_mask_stack(mask_stack_np)
                        mask_np = _blend_attention_maps(
                            [np.asarray(mask_np, dtype=np.float32), np.asarray(mixed_from_stack, dtype=np.float32)],
                            weights=[0.60, 1.10],
                            gamma=0.94,
                        ).astype(np.float32, copy=False)
            x_t = torch.from_numpy(np.asarray(x, dtype=np.float32))
            y_t = torch.from_numpy(np.asarray(y, dtype=np.float32))
            mask_t = torch.from_numpy(np.asarray(mask_np, dtype=np.float32)[None, ...])
            if bool(self.return_mask_stack):
                if int(mask_stack_np.shape[0]) <= 0 or int(mask_idx_np.size) <= 0:
                    mask_stack_np, mask_idx_np = build_label_mask_stack(
                        mixed_mask=np.asarray(mask_np, dtype=np.float32),
                        label_vec=y,
                        mask_stack_array=cached_stack_array,
                        mask_stack_indices=cached_stack_indices,
                    )
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
            "cache_version": 3,
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

    # Rebuild with term-localized inference from the actual image. Color terms remain color-only.
    inferred_stack, inferred_idx = build_term_mask_stack_from_image(image=chw, label_vec=label_vec, idx_to_term=idx_to_term)
    if int(inferred_stack.shape[0]) <= 0 or int(inferred_idx.size) <= 0:
        mixed = infer_semantic_support_mask(image=chw, terms=terms)
        inferred_stack, inferred_idx = build_label_mask_stack(mixed_mask=mixed, label_vec=label_vec)
    if explicit_mask is not None and int(inferred_stack.shape[0]) > 0:
        blended_stack: List[np.ndarray] = []
        for i, _cls_idx in enumerate(inferred_idx.tolist()):
            mask_i = np.asarray(inferred_stack[int(i)], dtype=np.float32)
            blended_i = np.asarray(mask_i, dtype=np.float32) * np.asarray(explicit_mask, dtype=np.float32)
            blended_i = _normalize_attention_map(blended_i, gamma=0.90, blur_kernel=1)
            if float(np.max(blended_i)) <= 1e-8:
                blended_i = np.asarray(explicit_mask, dtype=np.float32)
            blended_stack.append(np.asarray(blended_i, dtype=np.float32))
        inferred_stack = np.stack(blended_stack, axis=0).astype(np.float32, copy=False)
    mixed_mask = _composite_mask_stack(inferred_stack) if int(inferred_stack.shape[0]) > 0 else infer_semantic_support_mask(image=chw, terms=terms)
    if explicit_mask is not None:
        mixed_mask = np.asarray(explicit_mask, dtype=np.float32)
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
    root = Path(str(data_root).strip() or "toys_to_survive_development/data/berkeley_sbd")
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
        _split_images, _mask_stems = _read_sbd_split_file(root, split_name)
        label_path = root / "cache" / f"sbd_{split_name}_multilabel.npz"
        if not label_path.exists():
            raise RuntimeError(
                "Berkeley multilabel cache is missing for disk rows: "
                f"{label_path}. Generate label caches before running the pipeline."
            )
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
            yv = np.asarray(labels_split[int(i)], dtype=np.float32).reshape(-1)
            if int(yv.size) > int(n_classes):
                yv = yv[: int(n_classes)]
            elif int(yv.size) < int(n_classes):
                pad = np.zeros((int(n_classes) - int(yv.size),), dtype=np.float32)
                yv = np.concatenate([yv, pad], axis=0)
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
                    str(root / "cls" / f"{_mask_stems[int(i)]}.mat") if int(i) < int(len(_mask_stems)) else "",
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

        for row in _ordered_thread_map(split_specs_rows, _build_berkeley_row, max_workers=int(split_workers)):
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

        for row, unmapped_skip in _ordered_thread_map(external_specs, _build_external_row, max_workers=int(external_workers)):
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
    }
    cache_info = _materialize_semantic_row_mask_cache(
        rows=rows,
        class_names=class_names,
        cache_root=_resolve_semantic_mask_cache_root(root),
    )
    info.update(cache_info)
    with _SEMANTIC_DISK_ROWS_CACHE_LOCK:
        _SEMANTIC_DISK_ROWS_CACHE[cache_key] = (_clone_semantic_disk_rows(rows), dict(info))
    return rows, info
