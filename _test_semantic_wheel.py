"""Smoke test: semantic wheel cache build, round-trip, and cap guard."""
from __future__ import annotations

import tempfile
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
    SemanticWheelConfig,
    SemanticWheelDataset,
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
        assert sample["mixed_mask_u8"].dtype == np.uint8
        assert sample["mask_stack_u8"].dtype == np.uint8
        _ok("wheel dataset round-trips uint8 payloads")

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


if __name__ == "__main__":
    test_roundtrip()
    test_sanity_cap_guard()
    print("\nALL TESTS PASSED")
