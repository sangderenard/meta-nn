import numpy as np
import torch

from pipeline.preview import build_classifier_preview_frames
from wav_ml_viewer import _TransformerStatusOpenGLViewer


def test_build_classifier_preview_frames_formats_masks_and_scores():
    payload_batch = [
        {
            "global_step": 3,
            "total_steps": 9,
            "img": torch.tensor(
                [
                    [[0.1, 0.2], [0.3, 0.4]],
                    [[0.5, 0.6], [0.7, 0.8]],
                    [[0.2, 0.3], [0.4, 0.5]],
                ],
                dtype=torch.float32,
            ),
            "probs": torch.tensor([0.1, 0.9, 0.4], dtype=torch.float32),
            "target_vec": torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32),
            "target_mask": torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
            "detected_mask": torch.tensor([[1.0, 1.0], [0.0, 0.0]], dtype=torch.float32),
            "loss": 0.75,
            "batch_loss": 0.5,
        }
    ]

    eff_loss, frames = build_classifier_preview_frames(
        payload_batch,
        class_names=["zero", "one", "two"],
        cycle_id=2,
        round_id=7,
    )

    assert eff_loss == 0.5
    assert len(frames) == 1
    frame = frames[0]
    assert frame["caption"].startswith("[C] cycle=2 round=7 step=3/9")
    assert frame["titles"] == ["C target +mask", "C mask diff", "C detected +mask"]
    assert frame["rows"][0][0] == "target:one, two"
    assert "one:0.900" in frame["rows"][2]
    assert frame["images"][0].shape == (2, 2, 4)
    assert frame["images"][1].shape == (2, 2, 3)
    assert frame["images"][2].shape == (2, 2, 4)
