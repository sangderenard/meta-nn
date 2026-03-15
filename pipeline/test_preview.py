import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from pipeline.nodus_loss_store import SCRUB_FLAG_HAS_THUMBS
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
    assert frame["rows"][0] == ["one", "two"]
    assert "one:0.900" in frame["rows"][2]
    assert frame["images"][0].shape == (2, 2, 4)
    assert frame["images"][1].shape == (2, 2, 3)
    assert frame["images"][2].shape == (2, 2, 4)


class _FakeCompositeCache:
    def __init__(self, length: int = 2):
        self._panel = np.zeros((8, 8, 3), dtype=np.uint8)
        self._length = int(length)

    def copy_all_panels(self, _index):
        return [self._panel.copy(), self._panel.copy(), self._panel.copy()]

    def length(self):
        return self._length


def test_viewer_applies_frame_text_when_matching_image_is_presented():
    viewer = _TransformerStatusOpenGLViewer(enabled=False, image_hw=(8, 8), graph_h=0)
    viewer._composite_cache = _FakeCompositeCache(length=2)

    viewer._stage_frame_text(
        10,
        caption="frame one",
        titles=["one-a", "one-b", "one-c"],
        rows=[["a"], ["b"], ["c"]],
    )
    viewer._cache_frame_text_for_cursor(10)
    viewer._stage_frame_text(
        11,
        caption="frame two",
        titles=["two-a", "two-b", "two-c"],
        rows=[["x"], ["y"], ["z"]],
    )
    viewer._cache_frame_text_for_cursor(11)

    viewer._apply_composite_to_display(0)
    assert viewer._caption == "frame one"
    assert viewer._panel_titles == ["one-a", "one-b", "one-c"]

    viewer._apply_composite_to_display(1)
    assert viewer._caption == "frame two"
    assert viewer._panel_titles == ["two-a", "two-b", "two-c"]


def test_viewer_backfills_late_frame_text_by_ring_cursor():
    viewer = _TransformerStatusOpenGLViewer(enabled=False, image_hw=(8, 8), graph_h=0)
    viewer._composite_cache = _FakeCompositeCache(length=1)

    viewer._cache_frame_text_for_cursor(25)
    viewer._apply_composite_to_display(0)
    assert viewer._caption == ""

    viewer._stage_frame_text(
        25,
        caption="late frame",
        titles=["late-a", "late-b", "late-c"],
        rows=[["l1"], ["l2"], ["l3"]],
    )

    assert viewer._caption == "late frame"
    assert viewer._panel_titles == ["late-a", "late-b", "late-c"]


def test_viewer_builds_weight_map_from_ring_thumbnail_tiles():
    class _FakeMeta:
        def __init__(self, flags: int):
            self.flags = flags

    class _FakeRing:
        def __init__(self):
            self._write_cursor = 6
            self._length = 2
            self._tiles = [
                np.full((64, 64, 3), 10, dtype=np.uint8),
                np.full((64, 64, 3), 40, dtype=np.uint8),
                np.full((64, 64, 3), 90, dtype=np.uint8),
            ]

        def locked(self):
            class _Ctx:
                def __enter__(self_inner):
                    return self

                def __exit__(self_inner, *exc):
                    return False

            return _Ctx()

        def write_cursor(self):
            return self._write_cursor

        def length(self):
            return self._length

        def get_meta(self, _index):
            return _FakeMeta(SCRUB_FLAG_HAS_THUMBS)

        def copy_thumbnail(self, _index, thumb_idx):
            return self._tiles[thumb_idx].copy()

    viewer = _TransformerStatusOpenGLViewer(enabled=False, image_hw=(8, 8), graph_h=10)
    viewer._scrub_ring = _FakeRing()
    viewer._active_weight_model_name = "toy"

    rgb = viewer._weight_map_from_ring_cursor(5)

    assert rgb is not None
    assert rgb.shape == (2 * viewer.panel_h + viewer.graph_total_h, viewer.panel_w, 3)
    assert int(np.count_nonzero(rgb)) > 0


def test_viewer_renders_missing_checkpoint_thumbnail_from_checkpoint_file(tmp_path):
    viewer = _TransformerStatusOpenGLViewer(enabled=False, image_hw=(8, 8), graph_h=10)
    viewer._active_weight_model_name = "classifier"

    ckpt_path = tmp_path / "pipeline_checkpoint.pt"
    torch.save(
        {
            "round_id": 7,
            "cycle": 2,
            "classifier_state": {
                "fc.weight": torch.tensor([[0.1, -0.2], [0.3, 0.4]], dtype=torch.float32),
                "fc.bias": torch.tensor([0.01, -0.02], dtype=torch.float32),
            },
        },
        ckpt_path,
    )

    viewer.register_checkpoint_thumbnail(
        7,
        2,
        checkpoint_path=str(ckpt_path),
        model_name="classifier",
    )

    placeholder = viewer._load_checkpoint_thumbnail(0)
    assert placeholder is not None

    thumb_path = None
    for _ in range(60):
        viewer._drain_checkpoint_thumbnail_results()
        thumb_path = viewer._checkpoint_thumbs[0].get("thumb_path")
        if thumb_path and Path(thumb_path).exists():
            break
        time.sleep(0.05)

    assert thumb_path is not None
    assert Path(thumb_path).exists()

    rgb = viewer._load_checkpoint_thumbnail(0)
    assert rgb is not None
    assert rgb.shape == (2 * viewer.panel_h + viewer.graph_total_h, viewer.panel_w, 3)


def test_viewer_replaces_legacy_checkpoint_thumbnail_with_architectural_render(tmp_path):
    viewer = _TransformerStatusOpenGLViewer(enabled=False, image_hw=(8, 8), graph_h=10)
    viewer._active_weight_model_name = "classifier"

    ckpt_path = tmp_path / "pipeline_checkpoint.pt"
    thumb_path = tmp_path / "weight_thumb_r000007_c0002.png"
    torch.save(
        {
            "round_id": 7,
            "cycle": 2,
            "classifier_state": {
                "fc.weight": torch.tensor([[0.1, -0.2], [0.3, 0.4]], dtype=torch.float32),
                "fc.bias": torch.tensor([0.01, -0.02], dtype=torch.float32),
            },
        },
        ckpt_path,
    )
    Image.fromarray(np.full((16, 16, 3), 200, dtype=np.uint8), mode="RGB").save(thumb_path, format="PNG")
    thumb_path.with_suffix(".json").write_text(
        '{"model":"classifier","mode":"parameter_groups","width":16,"height":16}',
        encoding="utf-8",
    )

    viewer.register_checkpoint_thumbnail(
        7,
        2,
        thumb_path=str(thumb_path),
        checkpoint_path=str(ckpt_path),
        model_name="classifier",
    )

    placeholder = viewer._load_checkpoint_thumbnail(0)
    assert placeholder is not None

    refreshed = None
    for _ in range(60):
        viewer._drain_checkpoint_thumbnail_results()
        sidecar = json.loads(thumb_path.with_suffix(".json").read_text(encoding="utf-8"))
        if str(sidecar.get("mode")) == "architectural_tall":
            refreshed = viewer._load_checkpoint_thumbnail(0)
            break
        time.sleep(0.05)

    assert refreshed is not None
    assert refreshed.shape == (2 * viewer.panel_h + viewer.graph_total_h, viewer.panel_w, 3)
