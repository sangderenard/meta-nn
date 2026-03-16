from __future__ import annotations

import numpy as np
import torch
from pipeline.weight_map import (
    render_parameter_node_map,
    render_weight_image,
)


def test_parameter_groups_mode_renders_via_c_backend():
    state = {
        "block.weight": torch.tensor([[1.0, -2.0], [0.5, 1.5]], dtype=torch.float32),
        "block.bias": torch.tensor([0.25, -0.75], dtype=torch.float32),
        "head.weight": torch.tensor([[0.5, -0.5]], dtype=torch.float32),
    }

    rgb, meta = render_parameter_node_map(
        state,
        parameter_keys=["block.weight", "block.bias", "head.weight"],
        image_size=32,
    )
    assert rgb.shape == (32, 32, 3)
    assert meta["mode"] == "parameter_groups"
    assert meta["width"] == 32
    assert meta["height"] == 32
    assert int(np.count_nonzero(rgb)) > 0

def test_architectural_modes_place_layers_horizontally_and_transpose_cleanly():
    state = {
        "stem.weight": torch.tensor(
            [[0.1, -0.2], [0.3, 0.4], [-0.1, 0.2]],
            dtype=torch.float32,
        ),
        "stem.bias": torch.tensor([0.01, -0.02, 0.03], dtype=torch.float32),
        "head.weight": torch.tensor(
            [[0.5, -0.6, 0.7], [-0.3, 0.2, 0.1]],
            dtype=torch.float32,
        ),
        "head.bias": torch.tensor([0.2, -0.25], dtype=torch.float32),
    }

    tall_rgb, tall_meta = render_weight_image(
        state,
        target_width=64,
        target_height=64,
        mode="architectural_tall",
    )
    wide_rgb, wide_meta = render_weight_image(
        state,
        target_width=64,
        target_height=64,
        mode="architectural_wide",
    )

    assert tall_meta["mode"] == "architectural_tall"
    assert wide_meta["mode"] == "architectural_wide"
    assert bool(tall_meta["transposed"]) is False
    assert bool(wide_meta["transposed"]) is True
    assert int(np.abs(tall_rgb.astype(np.int32) - wide_rgb.astype(np.int32)).sum()) > 0
