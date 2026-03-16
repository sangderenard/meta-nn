from __future__ import annotations

import numpy as np
import torch
from pipeline.weight_map import (
    parameter_plan_from_render_spec,
    render_parameter_node_map,
    render_weight_image,
    resolve_weight_render_spec,
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


def test_architectural_packed_mode_renders_and_reports_flag():
    state = {
        "seq.0.weight": torch.randn(128, 64, dtype=torch.float32),
        "seq.1.weight": torch.randn(32, 128, dtype=torch.float32),
        "seq.2.weight": torch.randn(512, 32, dtype=torch.float32),
        "seq.3.weight": torch.randn(16, 512, dtype=torch.float32),
    }

    packed_rgb, packed_meta = render_weight_image(
        state,
        target_width=96,
        target_height=96,
        mode="architectural_packed",
    )

    assert packed_meta["mode"] == "architectural_packed"
    assert bool(packed_meta.get("packed", False)) is True
    assert bool(packed_meta.get("transposed", False)) is False
    assert packed_rgb.shape[2] == 3


def test_generic_render_spec_collapses_wrapper_paths_and_preserves_order():
    state = {
        "encoder.block0.base.weight": torch.randn(4, 3, dtype=torch.float32),
        "encoder.block0.base.bias": torch.randn(4, dtype=torch.float32),
        "encoder.block0.slots.alpha.down.weight": torch.randn(2, 3, dtype=torch.float32),
        "encoder.block0.slots.alpha.up.weight": torch.randn(4, 2, dtype=torch.float32),
        "head.weight": torch.randn(2, 4, dtype=torch.float32),
    }

    spec = resolve_weight_render_spec(state)
    assert str(spec.get("layout_name", "")) == "grouped_state_dict"
    layer_names = [str(row.get("name", "")) for row in list(spec.get("layers", []))]
    assert layer_names == ["encoder.block0", "head"]

    plan = parameter_plan_from_render_spec(state, weight_render_spec=spec)
    render_keys = [str(render_key) for _, render_key in plan]
    assert render_keys[0].startswith("encoder.block0.p")
    assert render_keys[-1].startswith("head.p")
    assert len(render_keys) == 5
