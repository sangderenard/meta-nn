import numpy as np

from pipeline.semantic_wheel_cache import (
    SemanticWheelCandidate,
    SemanticWheelConfig,
    SemanticWheelDataset,
    build_semantic_cache_entry,
    ensure_semantic_candidate_cache,
)
from semantic_dataset_loaders import DatasetTermRegistry, normalize_vocab_terms


def test_semantic_wheel_terms_sidecar_is_written_and_used(tmp_path):
    candidates = [
        SemanticWheelCandidate(cache_key="r0", terms=["cat", "signal"], source="unit"),
        SemanticWheelCandidate(cache_key="r1", terms=["dog", "signal"], source="unit"),
        SemanticWheelCandidate(cache_key="r2", terms=["bird", "signal"], source="unit"),
    ]
    candidate_indices = [0, 1, 2]

    registry = DatasetTermRegistry()
    for c in candidates:
        registry.register_many(c.terms)

    def _entry_group(base_row_idx: int, _base_row_pos: int):
        val = float(base_row_idx + 1) / 10.0
        image = np.full((3, 8, 8), val, dtype=np.float32)
        mask_stack = np.ones((1, 8, 8), dtype=np.float32)
        mask_indices = np.asarray([0], dtype=np.int64)
        return [
            build_semantic_cache_entry(
                image=image,
                image_size=8,
                terms=candidates[int(base_row_idx)].terms,
                mask_stack=mask_stack,
                mask_indices=mask_indices,
            )
        ]

    result = ensure_semantic_candidate_cache(
        candidates=candidates,
        candidate_indices=candidate_indices,
        build_entry_group=_entry_group,
        build_entry_groups_batch=None,
        registry=registry,
        config=SemanticWheelConfig(
            purpose="unit_sidecar",
            cache_root=str(tmp_path),
            image_size=8,
            batch_size=2,
            lookahead_batches=1,
            seed=7,
            deformations_per_clean=0,
            include_clean=True,
            use_rare_term_deck=False,
        ),
    )

    cache_dir = result["cache_dir"]
    ds = SemanticWheelDataset(cache_dir=cache_dir, return_mask_stack=False)
    manifest_sidecar = str(ds.manifest.get("terms_sidecar", ""))
    assert manifest_sidecar
    assert (ds.cache_dir / manifest_sidecar).exists()

    expected_terms = [
        list(normalize_vocab_terms(candidates[int(row_idx)].terms))
        for row_idx in list(result.get("base_row_indices") or [])
    ]
    actual_terms = [list(ds.read_terms_entry(i)) for i in range(int(len(ds)))]
    assert actual_terms == expected_terms

    for i in range(int(len(ds))):
        item_terms = list(ds.read_numpy_entry(i).get("terms") or [])
        assert item_terms == expected_terms[int(i)]

    iter_terms = [list(t) for t in ds.iter_terms()]
    assert iter_terms == expected_terms
