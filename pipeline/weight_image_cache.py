from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, List, Optional

import numpy as np
from PIL import Image


CHECKPOINT_THUMBNAIL_DIRNAME = "_checkpoint_thumbnails"
WEIGHT_IMAGE_FLAG_CHECKPOINT = 0x01


def _checkpoint_model_slug(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(model_name or "").strip()).strip("_") or "model"


def crop_weight_image_rgb(
    rgb: np.ndarray,
    *,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
    if arr.ndim != 3 or int(arr.shape[2]) != 3:
        raise ValueError("expected RGB image with shape (H, W, 3)")
    target_w = max(1, int(target_width))
    target_h = max(1, int(target_height))
    src_h = int(arr.shape[0])
    src_w = int(arr.shape[1])
    if src_h == target_h and src_w == target_w:
        return arr.copy()

    out = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    copy_w = min(src_w, target_w)
    copy_h = min(src_h, target_h)

    src_x0 = max(0, (src_w - copy_w) // 2)
    src_y0 = max(0, (src_h - copy_h) // 2)
    dst_x0 = max(0, (target_w - copy_w) // 2)
    dst_y0 = max(0, (target_h - copy_h) // 2)

    out[dst_y0 : dst_y0 + copy_h, dst_x0 : dst_x0 + copy_w] = arr[
        src_y0 : src_y0 + copy_h,
        src_x0 : src_x0 + copy_w,
    ]
    return out


def plan_weight_image_evictions(
    entries: Iterable[Any],
    *,
    max_entries: int,
    max_total_bytes: int,
    incoming_bytes: int = 0,
    clear_all: bool = False,
) -> List[int]:
    work = [
        {
            "orig_index": int(i),
            "byte_count": max(0, int(getattr(entry, "byte_count", 0) or 0)),
            "flags": int(getattr(entry, "flags", 0) or 0),
        }
        for i, entry in enumerate(entries)
    ]
    if bool(clear_all):
        return [int(item["orig_index"]) for item in work]

    limit_entries = max(1, int(max_entries))
    limit_bytes = max(0, int(max_total_bytes))
    incoming = max(0, int(incoming_bytes))
    total_bytes = sum(int(item["byte_count"]) for item in work)
    evicted: List[int] = []

    def _pick_evict_index() -> int:
        for pos, item in enumerate(work):
            if (int(item["flags"]) & int(WEIGHT_IMAGE_FLAG_CHECKPOINT)) == 0:
                return pos
        return 0 if work else -1

    while True:
        if incoming > 0:
            needs_trim = (
                (len(work) >= limit_entries and len(work) > 0)
                or (
                    limit_bytes > 0
                    and len(work) > 0
                    and (total_bytes + incoming) > limit_bytes
                )
            )
        else:
            needs_trim = (
                len(work) > limit_entries
                or (
                    len(work) > 1
                    and limit_bytes > 0
                    and total_bytes > limit_bytes
                )
            )
        if not needs_trim:
            break
        pick = _pick_evict_index()
        if pick < 0:
            break
        item = work.pop(pick)
        total_bytes = max(0, total_bytes - int(item["byte_count"]))
        evicted.append(int(item["orig_index"]))
    return evicted


def checkpoint_thumbnail_root(base_dir: Path) -> Path:
    return Path(base_dir) / CHECKPOINT_THUMBNAIL_DIRNAME


def checkpoint_thumbnail_path(
    root_dir: Path,
    *,
    round_id: int,
    cycle: int,
    model_name: str,
    generation: int,
    architecture_version: int,
) -> Path:
    model_slug = _checkpoint_model_slug(model_name)
    return (
        Path(root_dir)
        / (
            f"r{int(round_id):06d}_c{int(cycle):04d}"
            f"_{model_slug}_g{int(generation):06d}"
            f"_a{int(architecture_version) & 0xFFFFFFFFFFFFFFFF:016x}.png"
        )
    )


def save_checkpoint_thumbnail(path: Path, rgb: np.ndarray) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8)), mode="RGB").save(dest)
    return dest


def find_checkpoint_thumbnail(
    root_dir: Path,
    *,
    round_id: int,
    cycle: int,
    model_name: str,
    generation: Optional[int] = None,
    architecture_version: Optional[int] = None,
) -> Optional[Path]:
    root = Path(root_dir)
    if not root.exists():
        return None
    if generation is not None and architecture_version is not None:
        exact = checkpoint_thumbnail_path(
            root,
            round_id=int(round_id),
            cycle=int(cycle),
            model_name=str(model_name),
            generation=int(generation),
            architecture_version=int(architecture_version),
        )
        if exact.exists():
            return exact
    model_slug = _checkpoint_model_slug(model_name)
    pattern = f"r{int(round_id):06d}_c{int(cycle):04d}_{model_slug}_g*_a*.png"
    matches = sorted(root.glob(pattern))
    if not matches:
        return None
    return matches[-1]


def load_checkpoint_thumbnail(path: Path) -> np.ndarray:
    with Image.open(Path(path)) as im:
        return np.ascontiguousarray(np.asarray(im.convert("RGB"), dtype=np.uint8))
