"""Smoke test: semantic wheel cache build, round-trip, and cap guard."""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from pipeline.nodes.data_nodes import (
    _expand_semantic_mask_supervision_batch,
    _unpack_masked_semantic_batch,
)
from pipeline.semantic_wheel_cache import (
    SemanticWheelCandidate,
    SemanticWheelConfig,
    SemanticWheelDataset,
    build_semantic_cache_entry,
    ensure_semantic_candidate_cache,
    ensure_semantic_wheel_cache,
)
from semantic_dataset_loaders import SemanticDiskRow, semantic_mask_stack_collate


def _ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def _make_rows(root: Path, n_rows: int = 3) -> tuple[list[SemanticDiskRow], list[str]]:
    class_names = [
        "berkeley sbd dataset",
        "object",
        "signal",
        "red object",
        "noise damage",
    ]
    rows: list[SemanticDiskRow] = []
    for i in range(n_rows):
        img = np.zeros((12, 10, 3), dtype=np.uint8)
        img[..., 0] = 64 + (i * 40)
        img[2:8, 3:9, 1] = 200
        mask = np.zeros((12, 10), dtype=np.uint8)
        mask[2:8, 3:9] = 255
        img_path = root / f"image_{i}.png"
        mask_path = root / f"mask_{i}.png"
        Image.fromarray(img, mode="RGB").save(img_path)
        Image.fromarray(mask, mode="L").save(mask_path)
        y = np.zeros((len(class_names),), dtype=np.float32)
        y[0] = 1.0
        y[1] = 1.0
        y[2] = 1.0
        if i % 2 == 0:
            y[3] = 1.0
        rows.append(
            SemanticDiskRow(
                image_path=str(img_path),
                label_vec=y,
                terms=["berkeley sbd dataset", "object", "signal", "red object"],
                source="berkeley_sbd_train",
                mask_path=str(mask_path),
            )
        )
    return rows, class_names


def test_roundtrip() -> None:
    print("\n--- test_roundtrip ---")
    with tempfile.TemporaryDirectory(dir=".") as td:
        root = Path(td)
        rows, class_names = _make_rows(root)
        cfg = SemanticWheelConfig(
            purpose="smoke",
            cache_root=str(root / "cache"),
            image_size=16,
            batch_size=2,
            lookahead_batches=2,
            seed=7,
            deformations_per_clean=1,
            include_clean=True,
            sanity_cap_bytes=64 * 1024 * 1024,
            allow_large_override=False,
            expiry_uses=2,
        )
        built = ensure_semantic_wheel_cache(
            rows=rows,
            candidate_indices=[0, 1, 2],
            class_names=class_names,
            config=cfg,
        )
        ds = SemanticWheelDataset(cache_dir=str(built["cache_dir"]), return_mask_stack=True)
        assert len(ds) == 6, len(ds)
        sample = ds.read_numpy_entry(0)
        assert sample["image_u8"].dtype == np.uint8
        assert sample["mixed_mask"].dtype == np.float32
        assert sample["mask_stack"].dtype == np.float32
        _ok("wheel dataset round-trips float32 mask payloads")

        loader = DataLoader(
            ds,
            batch_size=2,
            shuffle=False,
            collate_fn=semantic_mask_stack_collate,
        )
        batch = next(iter(loader))
        xb, yb, mb, meta = _unpack_masked_semantic_batch(batch, context="semantic-wheel-smoke")
        assert xb.dtype == torch.float32
        assert yb.dtype == torch.float32
        assert mb.dtype == torch.float32
        assert float(torch.amax(mb).item()) <= 1.0
        assert float(torch.amin(mb).item()) >= 0.0
        assert meta["mask_stacks"][0].dtype == torch.uint8
        _ok("dataloader unpack normalizes uint8 masks to float01")

        x_exp, y_exp, m_exp = _expand_semantic_mask_supervision_batch(
            xb=xb,
            yb=yb,
            mb=mb,
            batch_meta=meta,
            mode="single_label_passes",
            context="semantic-wheel-smoke",
        )
        assert x_exp.dtype == torch.float32
        assert y_exp.dtype == torch.float32
        assert m_exp.dtype == torch.float32
        assert int(x_exp.shape[0]) >= int(xb.shape[0])
        _ok("single-label expansion consumes cached uint8 mask stacks")


def test_sanity_cap_guard() -> None:
    print("\n--- test_sanity_cap_guard ---")
    with tempfile.TemporaryDirectory(dir=".") as td:
        root = Path(td)
        rows, class_names = _make_rows(root, n_rows=2)
        cfg = SemanticWheelConfig(
            purpose="smoke_guard",
            cache_root=str(root / "cache"),
            image_size=16,
            batch_size=2,
            lookahead_batches=1,
            seed=5,
            deformations_per_clean=1,
            include_clean=True,
            sanity_cap_bytes=256,
            allow_large_override=False,
        )
        try:
            ensure_semantic_wheel_cache(
                rows=rows,
                candidate_indices=[0, 1],
                class_names=class_names,
                config=cfg,
            )
        except RuntimeError as exc:
            assert "sanity cap" in str(exc).lower()
            _ok("sanity cap hard-fails without override")
            return
        raise AssertionError("expected sanity cap guard to raise")


def test_generic_candidate_deck_rotation() -> None:
    print("\n--- test_generic_candidate_deck_rotation ---")
    with tempfile.TemporaryDirectory(dir=".") as td:
        root = Path(td)
        label_dim = 5
        images: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        candidates: list[SemanticWheelCandidate] = []
        for i in range(5):
            img = np.zeros((3, 12, 12), dtype=np.float32)
            img[0, :, :] = float(i + 1) / 8.0
            img[1, 2:10, 2:10] = 1.0
            y = np.zeros((label_dim,), dtype=np.float32)
            y[i] = 1.0
            images.append(img)
            targets.append(y)
            candidates.append(
                SemanticWheelCandidate(
                    cache_key=f"candidate-{i}",
                    terms=[f"term-{i}"],
                    source="synthetic",
                )
            )

        def _entry_group(base_idx: int, _base_pos: int) -> list[dict]:
            return [
                build_semantic_cache_entry(
                    image=images[int(base_idx)],
                    label_vec=targets[int(base_idx)],
                    image_size=16,
                )
            ]

        cfg = SemanticWheelConfig(
            purpose="generic_deck",
            cache_root=str(root / "cache"),
            image_size=16,
            batch_size=2,
            lookahead_batches=1,
            seed=11,
            deformations_per_clean=0,
            include_clean=True,
            max_base_rows=2,
            sanity_cap_bytes=64 * 1024 * 1024,
            allow_large_override=False,
            expiry_uses=1,
            use_rare_term_deck=False,
        )
        first = ensure_semantic_candidate_cache(
            candidates=candidates,
            candidate_indices=list(range(5)),
            build_entry_group=_entry_group,
            label_dim=label_dim,
            config=cfg,
        )
        second = ensure_semantic_candidate_cache(
            candidates=candidates,
            candidate_indices=list(range(5)),
            build_entry_group=_entry_group,
            label_dim=label_dim,
            config=cfg,
        )
        first_rows = [int(x) for x in list(first.get("base_candidate_indices") or [])]
        second_rows = [int(x) for x in list(second.get("base_candidate_indices") or [])]
        assert int(len(first_rows)) == 2
        assert int(len(second_rows)) == 2
        assert set(first_rows).isdisjoint(set(second_rows))
        _ok("generic candidate cache rotates bounded deck slices across rebuilds")


def test_generic_candidate_cache_expiry_respected() -> None:
    print("\n--- test_generic_candidate_cache_expiry_respected ---")
    with tempfile.TemporaryDirectory(dir=".") as td:
        root = Path(td)
        label_dim = 3
        candidates = [
            SemanticWheelCandidate(cache_key=f"candidate-{i}", terms=[f"term-{i}"], source="synthetic")
            for i in range(4)
        ]

        def _entry_group(base_idx: int, _base_pos: int) -> list[dict]:
            img = np.zeros((3, 8, 8), dtype=np.float32)
            img[0, :, :] = float(base_idx + 1) / 8.0
            y = np.zeros((label_dim,), dtype=np.float32)
            y[int(base_idx) % int(label_dim)] = 1.0
            return [build_semantic_cache_entry(image=img, label_vec=y, image_size=8)]

        cfg = SemanticWheelConfig(
            purpose="expiry_probe",
            cache_root=str(root / "cache"),
            image_size=8,
            batch_size=2,
            lookahead_batches=1,
            seed=17,
            deformations_per_clean=0,
            include_clean=True,
            max_base_rows=2,
            sanity_cap_bytes=16 * 1024 * 1024,
            allow_large_override=False,
            expiry_uses=2,
            use_rare_term_deck=False,
        )
        first = ensure_semantic_candidate_cache(
            candidates=candidates,
            candidate_indices=list(range(4)),
            build_entry_group=_entry_group,
            label_dim=label_dim,
            config=cfg,
        )
        second = ensure_semantic_candidate_cache(
            candidates=candidates,
            candidate_indices=list(range(4)),
            build_entry_group=_entry_group,
            label_dim=label_dim,
            config=cfg,
        )
        third = ensure_semantic_candidate_cache(
            candidates=candidates,
            candidate_indices=list(range(4)),
            build_entry_group=_entry_group,
            label_dim=label_dim,
            config=cfg,
        )
        assert bool(first.get("cache_hit", False)) is False
        assert bool(second.get("cache_hit", False)) is True
        assert bool(third.get("cache_hit", False)) is False
        _ok("expiry_uses is enforced for semantic wheel cache hits")


def test_generic_candidate_cache_prunes_old_variants_and_tmp_dirs() -> None:
    print("\n--- test_generic_candidate_cache_prunes_old_variants_and_tmp_dirs ---")
    with tempfile.TemporaryDirectory(dir=".") as td:
        root = Path(td)
        cache_root = root / "cache"
        label_dim = 2
        candidates = [
            SemanticWheelCandidate(cache_key=f"candidate-{i}", terms=[f"term-{i}"], source="synthetic")
            for i in range(3)
        ]

        def _entry_group(base_idx: int, _base_pos: int) -> list[dict]:
            img = np.zeros((3, 8, 8), dtype=np.float32)
            img[1, :, :] = float(base_idx + 1) / 6.0
            y = np.zeros((label_dim,), dtype=np.float32)
            y[int(base_idx) % int(label_dim)] = 1.0
            return [build_semantic_cache_entry(image=img, label_vec=y, image_size=8)]

        cfg_a = SemanticWheelConfig(
            purpose="prune_probe",
            cache_root=str(cache_root),
            image_size=8,
            batch_size=2,
            lookahead_batches=1,
            seed=3,
            deformations_per_clean=0,
            include_clean=True,
            sanity_cap_bytes=16 * 1024 * 1024,
            allow_large_override=False,
            expiry_uses=0,
            use_rare_term_deck=False,
        )
        first = ensure_semantic_candidate_cache(
            candidates=candidates,
            candidate_indices=list(range(3)),
            build_entry_group=_entry_group,
            label_dim=label_dim,
            config=cfg_a,
        )
        stale_tmp = cache_root / "prune_probe_deadbeef_tmp"
        stale_tmp.mkdir(parents=True, exist_ok=True)
        old_ts = float(time.time()) - 900.0
        os.utime(stale_tmp, (old_ts, old_ts))
        cfg_b = SemanticWheelConfig(
            purpose="prune_probe",
            cache_root=str(cache_root),
            image_size=8,
            batch_size=2,
            lookahead_batches=1,
            seed=9,
            deformations_per_clean=0,
            include_clean=True,
            sanity_cap_bytes=16 * 1024 * 1024,
            allow_large_override=False,
            expiry_uses=0,
            use_rare_term_deck=False,
        )
        second = ensure_semantic_candidate_cache(
            candidates=candidates,
            candidate_indices=list(range(3)),
            build_entry_group=_entry_group,
            label_dim=label_dim,
            config=cfg_b,
        )
        remaining = sorted(
            p.name
            for p in cache_root.iterdir()
            if p.is_dir() and p.name.startswith("prune_probe_")
        )
        assert remaining == [Path(str(second["cache_dir"])).name], remaining
        assert Path(str(first["cache_dir"])).name not in remaining
        assert stale_tmp.exists() is False
        _ok("stale same-purpose wheel dirs and tmp leftovers are pruned")


if __name__ == "__main__":
    test_roundtrip()
    test_sanity_cap_guard()
    test_generic_candidate_deck_rotation()
    test_generic_candidate_cache_expiry_respected()
    test_generic_candidate_cache_prunes_old_variants_and_tmp_dirs()
    print("\nALL TESTS PASSED")
