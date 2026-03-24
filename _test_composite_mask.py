"""Smoke test: precomputed per-label mask stacks from element stacks."""
import numpy as np
from pipeline.vocabulary_defaults import DEFAULT_VOCABULARY
from semantic_dataset_loaders import (
    _composite_mask_stack,
    elem_stacks_to_label_stacks,
)

size = 256
class_names = list(DEFAULT_VOCABULARY)
n_classes = len(class_names)
term_to_idx = {str(t).strip().lower(): i for i, t in enumerate(class_names)}

# Pick four real terms for the two element groups
group_a = [class_names[0], class_names[1]]
group_b = [class_names[2], class_names[3]]

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
elem_term_lists_all = [[group_a, group_b]] * 4

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

for i in range(4):
    stack = label_stacks[i]
    idx = label_indices[i]
    composite = _composite_mask_stack(stack)
    assert stack.shape[0] > 0, f"stack empty at {i}"
    assert idx.shape[0] == stack.shape[0], f"idx/stack mismatch at {i}"
    assert composite.shape == (size, size), f"composite shape {composite.shape}"
    print(f"  sample {i}: stack={tuple(stack.shape)} idx={idx.tolist()} OK")

print("ALL TESTS PASSED")
