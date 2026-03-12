"""Smoke test: precomputed per-label mask stacks from element stacks."""
import numpy as np
from semantic_dataset_loaders import (
    BootstrapDynamicDataset,
    _composite_mask_stack,
    elem_stacks_to_label_stacks,
)

size = 256
n_classes = 10
term_to_idx = {f"term_{i}": i for i in range(n_classes)}

images = [np.random.rand(3, size, size).astype(np.float32) for _ in range(4)]
targets = [np.zeros(n_classes, dtype=np.float32) for _ in range(4)]
for i in range(4):
    targets[i][i % n_classes] = 1.0
    targets[i][(i + 1) % n_classes] = 1.0

elem_stacks = [
    np.stack([
        np.ones((size, size), dtype=np.float32),
        np.random.rand(size, size).astype(np.float32),
    ])
    for _ in range(4)
]
# bg covers term_0,term_1; disk covers term_2,term_3
elem_term_lists_all = [[["term_0", "term_1"], ["term_2", "term_3"]]] * 4
composite_masks = [_composite_mask_stack(s) for s in elem_stacks]

# Precompute per-label stacks
label_stacks = []
label_indices = []
for i in range(4):
    ls, li = elem_stacks_to_label_stacks(
        elem_stack=elem_stacks[i],
        elem_term_lists=elem_term_lists_all[i],
        label_vec=targets[i],
        term_to_idx=term_to_idx,
    )
    label_stacks.append(ls)
    label_indices.append(li)
    print(f"  row {i}: label_stack={tuple(ls.shape)} indices={li.tolist()}")

ds = BootstrapDynamicDataset(
    images=images,
    targets=targets,
    total_rows=4,
    seed=42,
    augment=False,
    expected_target_dim=n_classes,
    semantic_term_to_idx=term_to_idx,
    return_masks=True,
    return_mask_stack=True,
    dataset_name="test_precomputed",
    base_masks=composite_masks,
)
for i in range(4):
    ds.base_mask_stacks[i] = label_stacks[i]
    ds.base_mask_stack_indices[i] = label_indices[i]

for i in range(4):
    x, y, m, stack, idx = ds[i]
    assert x.shape == (3, size, size), f"x shape {x.shape}"
    assert y.shape == (n_classes,), f"y shape {y.shape}"
    assert m.shape == (1, size, size), f"m shape {m.shape}"
    assert stack.shape[0] > 0, f"stack empty at {i}"
    assert idx.shape[0] == stack.shape[0], f"idx/stack mismatch at {i}"
    print(f"  sample {i}: stack={tuple(stack.shape)} idx={idx.tolist()} OK")

print("ALL TESTS PASSED")
