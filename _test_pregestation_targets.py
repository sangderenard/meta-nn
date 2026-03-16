"""Smoke test: pregestation target enrichment and shuffled train manifests."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset

from pipeline.nodes.data_nodes import _build_stage_loader_pair
from pipeline.semantic_wheel_cache import StatefulSequentialDeckSampler
from pipeline.nodes.vocab_node import (
    _build_pregestation_logic_rows,
    _semantic_terms_with_tonal_tags,
)
from semantic_dataset_loaders import (
    StageDatasetManifest,
    assemble_semantic_mask_layers,
    build_label_mask_stack,
    build_loader_from_manifest,
    combine_label_mask_stacks,
    elem_stacks_to_label_stacks,
)


def _ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def test_pregestation_targets_include_observed_terms() -> None:
    print("\n--- test_pregestation_targets_include_observed_terms ---")
    active_terms = [
        "up",
        "down",
        "left",
        "right",
        "white",
        "black",
        "gray",
        "signal",
        "shape",
        "object",
        "bright",
        "dark",
    ]
    images, masks, mask_stacks, elem_term_lists, term_rows, _info = _build_pregestation_logic_rows(
        image_size=32,
        seed=7,
        samples_per_combo=1,
        active_terms_lc=active_terms,
        mode="direction_color",
    )
    chosen = next((i for i, row in enumerate(term_rows) if "white" in row), -1)
    assert chosen >= 0, "expected a white pregestation row"
    enriched_terms = _semantic_terms_with_tonal_tags(
        terms=term_rows[int(chosen)],
        image=images[int(chosen)],
        image_size=32,
    )
    added_terms = [
        str(term)
        for term in enriched_terms
        if str(term) not in {str(x) for x in term_rows[int(chosen)]}
    ]
    assert len(added_terms) > 0, (term_rows[int(chosen)], enriched_terms)
    _ok("pregestation target rows include image-observed tonal terms")

    class_names = [
        "up",
        "down",
        "left",
        "right",
        "white",
        "black",
        "gray",
        "signal",
        "shape",
        "object",
        "bright",
        "dark",
        "warm",
        "cool",
    ]
    idx_to_term = {int(i): str(name) for i, name in enumerate(class_names)}
    term_to_idx = {str(name).strip().lower(): int(i) for i, name in enumerate(class_names)}
    y = np.zeros((len(class_names),), dtype=np.float32)
    for term in enriched_terms:
        idx = int(term_to_idx.get(str(term).strip().lower(), -1))
        if idx >= 0:
            y[int(idx)] = 1.0
    explicit_stack, explicit_idx = elem_stacks_to_label_stacks(
        elem_stack=mask_stacks[int(chosen)],
        elem_term_lists=elem_term_lists[int(chosen)],
        label_vec=y,
        term_to_idx=term_to_idx,
    )
    fallback_stack, fallback_idx = build_label_mask_stack(
        mixed_mask=np.asarray(masks[int(chosen)], dtype=np.float32),
        label_vec=y,
        idx_to_term=idx_to_term,
        treat_mixed_mask_as_creation=True,
    )
    merged_stack, merged_idx = combine_label_mask_stacks(
        y,
        (explicit_stack, explicit_idx),
        (fallback_stack, fallback_idx),
        height=int(masks[int(chosen)].shape[0]),
        width=int(masks[int(chosen)].shape[1]),
        fallback_creation_mask=np.asarray(masks[int(chosen)], dtype=np.float32),
    )
    merged_idx_set = set(np.asarray(merged_idx, dtype=np.int64).tolist())
    assert int(term_to_idx["signal"]) in merged_idx_set
    assert int(term_to_idx["object"]) in merged_idx_set
    assert set(np.asarray(fallback_idx, dtype=np.int64).tolist()) == {int(term_to_idx["signal"]), int(term_to_idx["object"])}
    _ok("pregestation fallback masks retain ingested-item coverage for signal/object")


def test_stage_manifest_shuffle_with_ordered_subset() -> None:
    print("\n--- test_stage_manifest_shuffle_with_ordered_subset ---")
    ds = TensorDataset(torch.arange(12, dtype=torch.int64))
    manifest = StageDatasetManifest(
        name="shuffle_smoke",
        dataset=ds,
        batch_size=12,
        seed=11,
        num_workers=0,
        device_type="cpu",
        ordered_indices=list(range(12)),
        shuffle=True,
    )
    loader, count = build_loader_from_manifest(manifest)
    assert loader is not None
    assert count == 12
    first = next(iter(loader))[0].tolist()
    second = next(iter(loader))[0].tolist()
    assert first != second, (first, second)
    _ok("ordered split subsets still shuffle across train iterations")


def test_special_label_masks_use_ingested_mask_semantics() -> None:
    print("\n--- test_special_label_masks_use_ingested_mask_semantics ---")
    class_names = ["signal", "object", "mnist dataset"]
    idx_to_term = {int(i): str(name) for i, name in enumerate(class_names)}
    term_to_idx = {str(name).strip().lower(): int(i) for i, name in enumerate(class_names)}
    label_vec = np.ones((len(class_names),), dtype=np.float32)
    mixed_mask = np.zeros((8, 8), dtype=np.float32)
    mixed_mask[2:6, 1:5] = 1.0
    special_stack, special_idx = build_label_mask_stack(
        mixed_mask=mixed_mask,
        label_vec=label_vec,
        idx_to_term=idx_to_term,
        treat_mixed_mask_as_creation=True,
    )
    mask_by_idx = {
        int(special_idx[i]): np.asarray(special_stack[i], dtype=np.float32)
        for i in range(int(len(special_idx)))
    }
    assert set(mask_by_idx.keys()) == {0, 1, 2}, mask_by_idx.keys()
    assert float(np.max(mask_by_idx[0])) > 0.9
    assert float(np.mean(mask_by_idx[0])) < 0.9
    assert float(np.max(mask_by_idx[1])) > 0.9
    assert float(np.mean(mask_by_idx[1])) < 0.9
    assert np.allclose(mask_by_idx[2], np.ones_like(mixed_mask))
    image = np.zeros((1, 8, 8), dtype=np.float32)
    _, assembled_mask, assembled_stack, assembled_idx = assemble_semantic_mask_layers(
        image=image,
        label_vec=label_vec,
        idx_to_term=idx_to_term,
        term_to_idx=term_to_idx,
        original_mixed_mask=mixed_mask,
    )
    assert set(np.asarray(assembled_idx, dtype=np.int64).tolist()) >= {0, 1, 2}
    assert float(np.mean(np.asarray(assembled_mask, dtype=np.float32))) < 0.9
    assert float(np.max(np.asarray(assembled_mask, dtype=np.float32))) > 0.9
    assert int(np.asarray(assembled_stack, dtype=np.float32).shape[0]) >= 3
    _ok("signal/object reuse item mask while dataset labels stay full-frame")


def test_stateful_sequential_deck_sampler_continues_partial_pass() -> None:
    print("\n--- test_stateful_sequential_deck_sampler_continues_partial_pass ---")
    sampler = StatefulSequentialDeckSampler(6)
    first_iter = iter(sampler)
    first_two = [next(first_iter), next(first_iter)]
    resumed = list(iter(sampler))
    assert first_two == [0, 1], first_two
    assert resumed == [2, 3, 4, 5, 0, 1], resumed
    _ok("stateful sequential deck sampler resumes from the next unseen row")


def test_chunked_stage_loader_uses_sequential_subset_access() -> None:
    print("\n--- test_chunked_stage_loader_uses_sequential_subset_access ---")

    class _ChunkedIndexDataset(Dataset):
        def __init__(self, values):
            self.values = [int(v) for v in values]
            self.chunk_rows = [2, 2, 2, 2]
            self.lookahead_batches = 2

        def __len__(self):
            return int(len(self.values))

        def read_numpy_entry(self, index: int):
            return {"value": int(self.values[int(index)])}

        def __getitem__(self, index: int):
            return torch.tensor(int(self.values[int(index)]), dtype=torch.int64)

    ds = _ChunkedIndexDataset(range(8))
    loader, eval_loader = _build_stage_loader_pair(
        dataset=ds,
        name="chunked_smoke",
        batch_size=4,
        num_workers=0,
        seed=17,
        device_type="cpu",
        train_indices=[2, 4, 6, 7],
        eval_indices=[1, 3],
        prefetch_factor=2,
        shuffle_train=True,
    )
    assert loader is not None
    assert eval_loader is not None
    first = next(iter(loader)).tolist()
    second = next(iter(loader)).tolist()
    eval_batch = next(iter(eval_loader)).tolist()
    assert first == [2, 4, 6, 7], first
    assert second == [2, 4, 6, 7], second
    assert eval_batch == [1, 3], eval_batch
    _ok("chunked semantic-wheel subsets stay sequential instead of randomizing chunk access")


if __name__ == "__main__":
    test_pregestation_targets_include_observed_terms()
    test_stage_manifest_shuffle_with_ordered_subset()
    test_special_label_masks_use_ingested_mask_semantics()
    test_stateful_sequential_deck_sampler_continues_partial_pass()
    test_chunked_stage_loader_uses_sequential_subset_access()
    print("\nALL TESTS PASSED")
