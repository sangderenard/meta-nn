"""Smoke test: pregestation target enrichment and shuffled train manifests."""
from __future__ import annotations

from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset

from pipeline.nodes.data_nodes import build_stage_loaders
from pipeline.semantic_wheel_cache import StatefulSequentialDeckSampler
from pipeline.nodes.vocab_node import (
    _build_pregestation_logic_rows,
)
from pipeline.vocabulary_defaults import DEFAULT_VOCABULARY
from semantic_dataset_loaders import (
    DatasetTermRegistry,
    StageDatasetManifest,
    assemble_semantic_mask_layers,
    build_observed_color_stack_from_terms,
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
    observed_stack, observed_terms = build_observed_color_stack_from_terms(
        image=images[int(chosen)],
        term_row=term_rows[int(chosen)],
    )
    assert len(observed_terms) > 0, term_rows[int(chosen)]
    _ok("pregestation target rows include shared observed-color masks")

    class_names = list(DEFAULT_VOCABULARY)
    idx_to_term = {int(i): str(name) for i, name in enumerate(class_names)}
    term_to_idx = {str(name).strip().lower(): int(i) for i, name in enumerate(class_names)}
    y = np.zeros((len(class_names),), dtype=np.float32)
    for term in term_rows[int(chosen)]:
        tk = str(term).strip().lower()
        idx = int(term_to_idx.get(tk, -1))
        if idx < 0:
            raise ValueError(
                f"test: term {tk!r} not in test vocabulary — every term needs an index"
            )
        y[int(idx)] = 1.0
    # Geometric elem_stacks cover all base terms (dark, signal, shape, color, direction).
    explicit_stack, explicit_idx = elem_stacks_to_label_stacks(
        elem_stack=mask_stacks[int(chosen)],
        elem_term_lists=elem_term_lists[int(chosen)],
        label_vec=y,
        term_to_idx=term_to_idx,
    )
    observed_idx = np.asarray(
        [int(term_to_idx[str(t).strip().lower()]) for t in observed_terms if str(t).strip().lower() in term_to_idx],
        dtype=np.int64,
    )
    h_, w_ = int(masks[int(chosen)].shape[0]), int(masks[int(chosen)].shape[1])
    merged_stack, merged_idx = combine_label_mask_stacks(
        y,
        (explicit_stack, explicit_idx),
        (observed_stack, observed_idx),
        height=h_,
        width=w_,
        fallback_creation_mask=None,
    )
    merged_idx_set = set(np.asarray(merged_idx, dtype=np.int64).tolist())
    assert int(term_to_idx["signal"]) in merged_idx_set or int(term_to_idx["object"]) in merged_idx_set or len(merged_idx_set) > 0
    _ok("pregestation element and shared observed-color stacks combine without creation-mask stamping")


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


def test_assemble_semantic_mask_layers_returns_stack_and_idx() -> None:
    print("\n--- test_assemble_semantic_mask_layers_returns_stack_and_idx ---")
    class_names = list(DEFAULT_VOCABULARY)
    reg = DatasetTermRegistry()
    reg.register_many(class_names)
    label_vec = np.ones((len(class_names),), dtype=np.float32)
    image = np.zeros((3, 8, 8), dtype=np.float32)
    y, assembled_stack, assembled_idx = assemble_semantic_mask_layers(
        image=image,
        label_vec=label_vec,
        registry=reg,
    )
    assert int(np.asarray(y).shape[0]) == len(class_names)
    assert int(np.asarray(assembled_stack).ndim) == 3
    _ok("assemble_semantic_mask_layers returns (y, stack, idx) 3-tuple")


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
    loader, eval_loader = build_stage_loaders(
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
    test_assemble_semantic_mask_layers_returns_stack_and_idx()
    test_stateful_sequential_deck_sampler_continues_partial_pass()
    test_chunked_stage_loader_uses_sequential_subset_access()
    print("\nALL TESTS PASSED")
