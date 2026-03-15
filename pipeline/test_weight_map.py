from __future__ import annotations

import json

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from pipeline.nodes.save_restore_node import SaveRestoreNode
from pipeline.weight_map import (
    annotate_weight_map,
    render_parameter_node_map,
    render_weight_image,
    snapshot_parameter_state_from_state_dict,
    summarize_parameter_nodes,
)


class _ToyModel(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int) -> None:
        super().__init__()
        self.block = nn.Linear(in_dim, hid_dim)
        self.head = nn.Linear(hid_dim, out_dim)


def test_summarize_parameter_nodes_groups_weight_and_bias_into_one_node():
    state = {
        "block.weight": torch.tensor([[1.0, -2.0], [0.5, 1.5]], dtype=torch.float32),
        "block.bias": torch.tensor([0.25, -0.75], dtype=torch.float32),
        "head.weight": torch.tensor([[0.5, -0.5]], dtype=torch.float32),
    }

    stats = summarize_parameter_nodes(
        state,
        parameter_keys=["block.weight", "block.bias", "head.weight"],
    )

    assert [row["node"] for row in stats] == ["block", "head"]
    assert stats[0]["count"] == 6
    assert stats[1]["count"] == 2

    rgb, meta = render_parameter_node_map(
        state,
        parameter_keys=["block.weight", "block.bias", "head.weight"],
        image_size=32,
    )
    assert rgb.shape == (32, 32, 3)
    assert meta["node_count"] == 2
    assert int(np.count_nonzero(rgb)) > 0


def test_save_weight_thumbnail_uses_active_model_only_and_256px(tmp_path):
    node = SaveRestoreNode()
    classifier = _ToyModel(4, 3, 2)
    generator = _ToyModel(6, 5, 4)

    with torch.no_grad():
        classifier.block.weight.fill_(0.1)
        classifier.block.bias.fill_(0.2)
        classifier.head.weight.fill_(0.3)
        classifier.head.bias.fill_(0.4)
        generator.block.weight.fill_(1.0)
        generator.block.bias.fill_(0.5)
        generator.head.weight.fill_(-0.75)
        generator.head.bias.fill_(0.25)

    node.weight_tracker.register_base("classifier", classifier)
    node.weight_tracker.register_base("generator", generator)

    with torch.no_grad():
        generator.block.weight.add_(0.25)
        generator.head.bias.sub_(0.15)

    payload = {
        "classifier_state": classifier.state_dict(),
        "generator_state": generator.state_dict(),
    }
    node.weight_tracker.set_active("generator")

    info = node._save_weight_thumbnail(
        tmp_path,
        {"classifier": classifier, "generator": generator},
        payload=payload,
        round_id=7,
        cycle=2,
    )

    assert info is not None
    assert info["model"] == "generator"

    thumb_path = tmp_path / "weight_thumb_r000007_c0002.png"
    sidecar_path = tmp_path / "weight_thumb_r000007_c0002.json"
    assert thumb_path.exists()
    assert sidecar_path.exists()

    loaded = np.asarray(Image.open(thumb_path).convert("RGB"), dtype=np.uint8)
    assert loaded.shape[1] == 256

    meta = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert meta["model"] == "generator"
    assert meta["width"] == 256
    assert meta["mode"] == "architectural_tall"
    assert meta["height"] == loaded.shape[0]

    parameter_keys, saved_state = snapshot_parameter_state_from_state_dict(
        payload["generator_state"],
        [name for name, _ in generator.named_parameters()],
    )
    expected_rgb, expected_meta = render_weight_image(
        saved_state,
        parameter_keys=parameter_keys,
        reference_state=node.weight_tracker.base_state("generator"),
        target_width=256,
        target_height=256,
    )
    expected_rgb = annotate_weight_map(
        expected_rgb,
        title="generator r7",
        subtitle="",
    )

    assert np.array_equal(loaded, expected_rgb)


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
    )
    wide_rgb, wide_meta = render_weight_image(
        state,
        target_width=64,
        target_height=64,
    )

    assert tall_meta["layer_count"] == 2
    assert tall_meta["layers"][0]["name"] == "stem"
    assert tall_meta["layers"][1]["name"] == "head"
