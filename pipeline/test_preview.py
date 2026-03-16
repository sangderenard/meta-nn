from types import SimpleNamespace
from tempfile import TemporaryDirectory

import numpy as np
import torch

from pipeline.preview import build_classifier_preview_frames
from pipeline.weight_image_cache import (
    checkpoint_thumbnail_path,
    checkpoint_thumbnail_root,
    crop_weight_image_rgb,
    save_checkpoint_thumbnail,
)
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


def test_viewer_enqueue_frame_does_not_push_weight_tiles():
    class _RecordingRing:
        def __init__(self):
            self.calls = []

        def push(self, **kwargs):
            self.calls.append(dict(kwargs))
            return 12

        def capacity(self):
            return 8

    viewer = _TransformerStatusOpenGLViewer(enabled=False, image_hw=(8, 8), graph_h=0)
    viewer._scrub_ring = _RecordingRing()

    viewer.enqueue_frame(
        {
            "images": [
                np.zeros((4, 4, 3), dtype=np.uint8),
                np.zeros((4, 4, 3), dtype=np.uint8),
                np.zeros((4, 4, 3), dtype=np.uint8),
            ],
            "caption": "frame",
            "titles": ["a", "b", "c"],
            "rows": [[], [], []],
        }
    )

    assert len(viewer._scrub_ring.calls) == 1
    call = viewer._scrub_ring.calls[0]
    assert call["thumb0"] is None
    assert call["thumb1"] is None
    assert call["thumb2"] is None


def test_shared_weight_render_rerenders_on_config_change_and_uses_target_dims():
    class _FakeStateStore:
        def get_meta(self):
            return SimpleNamespace(publish_seq=7, model_name="toy")

    class _FakeImageStore:
        def __init__(self):
            self.calls = []
            self.limit_calls = []
            self._cfg = None
            self._image_seq = 0
            self._max_entries = 512

        def get_active_config(self):
            return self._cfg

        def length(self):
            return 0

        def stats(self):
            return SimpleNamespace(max_entries=self._max_entries, max_total_bytes=0)

        def get_meta(self, _index):
            return None

        def set_limits(self, *, max_entries, max_total_bytes):
            self.limit_calls.append((int(max_entries), int(max_total_bytes)))
            return True

        def render_latest(self, _state_store, *, mode, target_width, target_height):
            self.calls.append((int(mode), int(target_width), int(target_height)))
            self._image_seq += 1
            return True

        def latest_image(self):
            meta = SimpleNamespace(
                image_seq=self._image_seq,
                state_publish_seq=7,
                round_id=1,
                cycle=2,
                step=3,
                model_name="toy",
            )
            rgb = np.full((6, 5, 3), 64, dtype=np.uint8)
            return meta, rgb

    viewer = _TransformerStatusOpenGLViewer(enabled=False, image_hw=(8, 8), graph_h=0)
    viewer._ipc_server_ref = SimpleNamespace(has_connection=True)
    viewer._weight_state_store = _FakeStateStore()
    image_store = _FakeImageStore()
    image_store._max_entries = viewer._weight_history_maxlen
    viewer._weight_image_store = image_store

    image_store._cfg = SimpleNamespace(
        state_publish_seq=7,
        mode=1,
        target_width=100,
        target_height=80,
        render_width=150,
        render_height=120,
        render_channels=3,
        render_stride_bytes=450,
    )
    viewer._launch_shared_weight_render()
    viewer._shared_weight_render_thread.join(timeout=2.0)
    viewer._collect_shared_weight_render()

    image_store._cfg = SimpleNamespace(
        state_publish_seq=7,
        mode=1,
        target_width=140,
        target_height=90,
        render_width=210,
        render_height=140,
        render_channels=3,
        render_stride_bytes=630,
    )
    viewer._launch_shared_weight_render()
    viewer._shared_weight_render_thread.join(timeout=2.0)
    viewer._collect_shared_weight_render()

    assert image_store.calls == [(1, 100, 80), (1, 140, 90)]
    assert image_store.limit_calls == [
        (viewer._weight_history_maxlen, 450 * 120 * viewer._weight_history_maxlen),
        (viewer._weight_history_maxlen, 630 * 140 * viewer._weight_history_maxlen),
    ]
    assert len(viewer._weight_snapshot_deque) == 2
    assert viewer._weight_current_rgb_by_model["toy"].shape == (90, 140, 3)


def test_crop_weight_image_rgb_center_crops_and_pads():
    src = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)

    cropped = crop_weight_image_rgb(src, target_width=2, target_height=2)
    assert cropped.shape == (2, 2, 3)
    np.testing.assert_array_equal(cropped, src[1:3, 2:4, :])

    padded = crop_weight_image_rgb(src[:2, :2, :], target_width=4, target_height=4)
    assert padded.shape == (4, 4, 3)
    np.testing.assert_array_equal(padded[1:3, 1:3, :], src[:2, :2, :])


def test_viewer_resolves_checkpoint_weight_image_from_disk():
    viewer = _TransformerStatusOpenGLViewer(enabled=False, image_hw=(8, 8), graph_h=0)
    viewer._loss_store = SimpleNamespace(
        channel_keys=lambda: ["loss"],
        channel_length=lambda _ck: 5,
    )
    viewer._loss_count_at_snap_deque.append({"loss": 5})
    viewer.set_active_weight_model("generator")

    with TemporaryDirectory() as tmp_dir:
        root = checkpoint_thumbnail_root(tmp_dir)
        rgb = np.full(viewer._weight_map_target_hw() + (3,), 96, dtype=np.uint8)
        save_checkpoint_thumbnail(
            checkpoint_thumbnail_path(
                root,
                round_id=3,
                cycle=2,
                model_name="generator",
                generation=1,
                architecture_version=7,
            ),
            rgb,
        )
        viewer.set_checkpoint_backup_dir(tmp_dir)
        viewer.notify_pipeline_checkpoint_saved(
            round_id=3,
            cycle=2,
            weight_model="generator",
            weight_generation=1,
            weight_architecture_version=7,
        )
        viewer._scrub_offset = 1
        marker = viewer._checkpoint_marker_for_offset(1)
        resolved = viewer._resolve_checkpoint_weight_rgb(marker, model_name="generator")

    assert resolved is not None
    assert resolved.shape == viewer._weight_map_target_hw() + (3,)
    assert int(resolved[0, 0, 0]) == 96


def test_viewer_tracks_weight_registry_and_active_tab():
    viewer = _TransformerStatusOpenGLViewer(enabled=False, image_hw=(8, 8), graph_h=0)

    viewer._on_sr_response(
        {
            "type": "resp_weight_registry",
            "models": [
                {"model": "generator"},
                {"model": "discriminator"},
            ],
            "active_model": "generator",
        }
    )

    assert viewer._weight_model_order == ["generator", "discriminator"]
    assert viewer._active_weight_model_name == "generator"

    viewer.set_active_weight_model("discriminator")

    assert viewer._resolved_active_weight_model_name() == "discriminator"
    assert viewer._weight_snapshot_deque is viewer._weight_snapshot_deques_by_model["discriminator"]
