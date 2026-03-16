from __future__ import annotations

import copy
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import pipeline.semantic_wheel_cache as semantic_wheel_cache
from pipeline.context import PipelineContext
from pipeline.nodes.classifier_node import _sync_gate_classifier_replica
from pipeline.nodes.vocab_node import (
    _default_bootstrap_primitive_terms,
    VocabChurnNode,
    VocabConfig,
)
from pipeline.semantic_wheel_cache import (
    SemanticWheelCandidate,
    SemanticWheelConfig,
    SemanticWheelDataset,
    build_semantic_cache_entry,
    ensure_semantic_candidate_cache,
)
from semantic_dataset_loaders import (
    build_term_mask_stacks_from_images,
    collect_semantic_disk_rows,
    infer_semantic_support_masks,
)
from wav_ml_models import TinyConvClassifier, prime_tiny_classifier_label_bank_for_state_dict


def _ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def test_gate_replica_sync_realigns_label_bank() -> None:
    print("\n--- test_gate_replica_sync_realigns_label_bank ---")
    source = TinyConvClassifier(
        num_classes=4,
        base_ch=16,
        max_ch=64,
        context_blocks=0,
    )
    stale_gate = copy.deepcopy(source)
    source.set_label_embedding_bank(
        torch.randn(4, 13, dtype=torch.float32),
        temperature=7.5,
    )

    synced, info = _sync_gate_classifier_replica(
        source_classifier=source,
        gate_classifier=stale_gate,
        gate_device=torch.device("cpu"),
        channels_last=False,
    )

    assert info["created"] is False, info
    assert info["strict_load"] is True, info
    assert tuple(synced.label_embed_bank.shape) == (4, 13), tuple(synced.label_embed_bank.shape)
    assert bool(int(synced.label_embed_enabled.item())) is True
    assert int(synced.embed_proj.out_features) == 13
    assert torch.allclose(synced.label_embed_bank, source.label_embed_bank)
    assert torch.allclose(synced.embed_proj.weight, source.embed_proj.weight)
    assert torch.allclose(synced.embed_proj.bias, source.embed_proj.bias)
    _ok("gate replica sync tolerates stale label-bank buffers and lands on the source state")


def test_checkpoint_prime_realigns_label_bank_before_load() -> None:
    print("\n--- test_checkpoint_prime_realigns_label_bank_before_load ---")
    source = TinyConvClassifier(
        num_classes=4,
        base_ch=16,
        max_ch=64,
        context_blocks=0,
    )
    source.set_label_embedding_bank(
        torch.randn(4, 13, dtype=torch.float32),
        temperature=6.0,
    )
    target = TinyConvClassifier(
        num_classes=4,
        base_ch=16,
        max_ch=64,
        context_blocks=0,
    )

    primed = prime_tiny_classifier_label_bank_for_state_dict(
        target,
        source.state_dict(),
        temperature=6.0,
    )
    assert primed is True
    target.load_state_dict(source.state_dict(), strict=False)

    assert tuple(target.label_embed_bank.shape) == (4, 13), tuple(target.label_embed_bank.shape)
    assert bool(int(target.label_embed_enabled.item())) is True
    assert int(target.embed_proj.out_features) == 13
    assert torch.allclose(target.label_embed_bank, source.label_embed_bank)
    assert torch.allclose(target.embed_proj.weight, source.embed_proj.weight)
    assert torch.allclose(target.embed_proj.bias, source.embed_proj.bias)
    _ok("checkpoint priming reshapes stale classifiers before state_dict restore")


def test_collect_semantic_disk_rows_remaps_voc_bits_by_name() -> None:
    print("\n--- test_collect_semantic_disk_rows_remaps_voc_bits_by_name ---")
    with tempfile.TemporaryDirectory(dir=".") as td:
        root = Path(td)
        (root / "img").mkdir(parents=True, exist_ok=True)
        (root / "cache").mkdir(parents=True, exist_ok=True)
        (root / "train.txt").write_text("sample_train\n", encoding="utf-8")
        (root / "val.txt").write_text("sample_val\n", encoding="utf-8")

        for stem, fill in (("sample_train", 64), ("sample_val", 192)):
            image = np.full((10, 10, 3), fill_value=int(fill), dtype=np.uint8)
            Image.fromarray(image, mode="RGB").save(root / "img" / f"{stem}.jpg")

        train_labels = np.zeros((1, 20), dtype=np.float32)
        train_labels[0, 0] = 1.0
        train_labels[0, 6] = 1.0
        val_labels = np.zeros((1, 20), dtype=np.float32)
        val_labels[0, 14] = 1.0
        np.savez(root / "cache" / "sbd_train_multilabel.npz", labels=train_labels)
        np.savez(root / "cache" / "sbd_val_multilabel.npz", labels=val_labels)

        class_names = [
            "berkeley sbd dataset",
            "object",
            "signal",
            "car",
            "person",
        ]
        rows, info = collect_semantic_disk_rows(str(root), class_names)

        assert int(info["available_rows"]) == 2, info
        train_row = next(row for row in rows if str(row.source) == "berkeley_sbd_train")
        val_row = next(row for row in rows if str(row.source) == "berkeley_sbd_val")

        assert float(train_row.label_vec[0]) == 1.0
        assert float(train_row.label_vec[1]) == 1.0
        assert float(train_row.label_vec[2]) == 1.0
        assert float(train_row.label_vec[3]) == 1.0
        assert float(train_row.label_vec[4]) == 0.0
        assert float(val_row.label_vec[4]) == 1.0
        _ok("Berkeley multilabel rows remap VOC cache bits by class name instead of raw position")


def test_voc20_terms_enter_only_via_churn() -> None:
    print("\n--- test_voc20_terms_enter_only_via_churn ---")
    fixed_supervised = [f"fixed-{i}" for i in range(151)]
    ctx = PipelineContext()
    ctx.supervised_class_names = list(fixed_supervised)
    ctx.core_terms = ["object", "signal"]
    ctx.active_extra_terms = ["object", "signal", "semantic slot 3", "semantic slot 4"]
    ctx.class_names = list(ctx.supervised_class_names) + list(ctx.active_extra_terms)
    node = VocabChurnNode(VocabConfig(churn_n=2, churn_every_n_cycles=1, seed=0))

    bootstrap_terms = set(_default_bootstrap_primitive_terms())
    assert "aeroplane" not in bootstrap_terms

    node.execute(ctx)

    assert ctx.supervised_class_names == fixed_supervised
    assert ctx.class_names[: len(fixed_supervised)] == fixed_supervised
    assert ctx.active_extra_terms[0:2] == ["object", "signal"]
    assert "aeroplane" in ctx.active_extra_terms
    _ok("VOC20 names are injected only at churn time and do not edit the fixed supervised slice")


def test_semantic_candidate_cache_batch_build_preserves_order() -> None:
    print("\n--- test_semantic_candidate_cache_batch_build_preserves_order ---")
    with tempfile.TemporaryDirectory(dir=".") as td:
        root = Path(td)
        label_dim = 4
        images: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        candidates: list[SemanticWheelCandidate] = []
        for i in range(label_dim):
            img = np.zeros((3, 12, 12), dtype=np.float32)
            img[0, :, :] = float(i + 1) / 10.0
            img[1, 2:10, 2:10] = 1.0
            y = np.zeros((label_dim,), dtype=np.float32)
            y[i] = 1.0
            images.append(img)
            targets.append(y)
            candidates.append(
                SemanticWheelCandidate(
                    cache_key=f"batch-{i}",
                    terms=[f"term-{i}"],
                    source="synthetic",
                )
            )

        row_calls = 0
        batch_calls = 0
        batch_sizes: list[int] = []

        def _entry_group(base_idx: int, _base_pos: int) -> list[dict]:
            nonlocal row_calls
            row_calls += 1
            return []

        def _entry_groups_batch(spec_batch: list[tuple[int, int]]) -> list[list[dict]]:
            nonlocal batch_calls
            batch_calls += 1
            batch_sizes.append(int(len(spec_batch)))
            out: list[list[dict]] = []
            for base_idx, _base_pos in spec_batch:
                out.append(
                    [
                        build_semantic_cache_entry(
                            image=images[int(base_idx)],
                            label_vec=targets[int(base_idx)],
                            image_size=16,
                        )
                    ]
                )
            return out

        cfg = SemanticWheelConfig(
            purpose="batch_order",
            cache_root=str(root / "cache"),
            image_size=16,
            batch_size=2,
            lookahead_batches=1,
            seed=19,
            deformations_per_clean=0,
            include_clean=True,
            max_base_rows=4,
            sanity_cap_bytes=64 * 1024 * 1024,
            allow_large_override=False,
            expiry_uses=1,
            use_rare_term_deck=False,
        )

        built = ensure_semantic_candidate_cache(
            candidates=candidates,
            candidate_indices=list(range(label_dim)),
            build_entry_group=_entry_group,
            build_entry_groups_batch=_entry_groups_batch,
            label_dim=label_dim,
            config=cfg,
        )

        assert int(row_calls) == 0, row_calls
        assert int(batch_calls) >= 1, batch_calls
        assert max(batch_sizes) >= 2, batch_sizes

        ds = SemanticWheelDataset(cache_dir=str(built["cache_dir"]), return_mask_stack=True)
        selected = [int(x) for x in list(built.get("base_candidate_indices", []))]
        assert int(len(ds)) == int(len(selected)), (len(ds), selected)
        for row_idx, candidate_idx in enumerate(selected):
            entry = ds.read_numpy_entry(int(row_idx))
            observed = int(np.argmax(np.asarray(entry["label_vec_u8"], dtype=np.uint8)))
            assert observed == int(candidate_idx), (row_idx, observed, candidate_idx, selected)
        _ok("semantic candidate cache dispatches ordered multi-image batch builders without reordering rows")


def test_semantic_support_masks_torch_path_matches_default() -> None:
    print("\n--- test_semantic_support_masks_torch_path_matches_default ---")
    images = np.zeros((2, 3, 12, 12), dtype=np.float32)
    images[0, 0, :, :] = 1.0
    images[1, 1, :, :] = 1.0
    terms = [["red", "signal"], ["green", "signal"]]

    base = infer_semantic_support_masks(images=images, terms_batch=terms)
    torch_cpu = infer_semantic_support_masks(
        images=images,
        terms_batch=terms,
        processing_device=torch.device("cpu"),
    )

    assert tuple(base.shape) == tuple(torch_cpu.shape), (base.shape, torch_cpu.shape)
    assert np.allclose(base, torch_cpu, atol=1e-5), float(np.max(np.abs(base - torch_cpu)))
    _ok("torch-backed semantic support mask path matches default CPU implementation")


def test_term_mask_stacks_torch_path_matches_default() -> None:
    print("\n--- test_term_mask_stacks_torch_path_matches_default ---")
    images = np.zeros((2, 3, 10, 10), dtype=np.float32)
    images[0, 0, :, :] = 1.0
    images[1, 2, :, :] = 1.0
    labels = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    idx_to_term = {0: "red", 1: "blue", 2: "signal"}

    base_stacks, base_indices = build_term_mask_stacks_from_images(
        images=images,
        label_vecs=labels,
        idx_to_term=idx_to_term,
    )
    torch_stacks, torch_indices = build_term_mask_stacks_from_images(
        images=images,
        label_vecs=labels,
        idx_to_term=idx_to_term,
        processing_device=torch.device("cpu"),
    )

    assert len(base_stacks) == len(torch_stacks) == 2
    for i in range(2):
        assert np.array_equal(np.asarray(base_indices[i], dtype=np.int64), np.asarray(torch_indices[i], dtype=np.int64)), (i, base_indices[i], torch_indices[i])
        assert np.allclose(np.asarray(base_stacks[i], dtype=np.float32), np.asarray(torch_stacks[i], dtype=np.float32), atol=1e-5), i
    _ok("torch-backed term-mask stack builder matches default CPU implementation")


if __name__ == "__main__":
    test_gate_replica_sync_realigns_label_bank()
    test_checkpoint_prime_realigns_label_bank_before_load()
    test_collect_semantic_disk_rows_remaps_voc_bits_by_name()
    test_voc20_terms_enter_only_via_churn()
    test_semantic_candidate_cache_batch_build_preserves_order()
    test_semantic_support_masks_torch_path_matches_default()
    test_term_mask_stacks_torch_path_matches_default()
    print("\nALL TESTS PASSED")
