"""Smoke test: pregestation target enrichment and shuffled train manifests."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import TensorDataset

from pipeline.nodes.vocab_node import (
    _build_pregestation_logic_rows,
    _semantic_terms_with_tonal_tags,
)
from semantic_dataset_loaders import (
    StageDatasetManifest,
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
    added_idx = int(term_to_idx[added_terms[0]])
    assert added_idx in set(np.asarray(merged_idx, dtype=np.int64).tolist())
    assert int(np.asarray(merged_stack).shape[0]) >= int(np.count_nonzero(y >= 0.5))
    _ok("pregestation mask stacks retain added heuristic target labels")


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


if __name__ == "__main__":
    test_pregestation_targets_include_observed_terms()
    test_stage_manifest_shuffle_with_ordered_subset()
    print("\nALL TESTS PASSED")
