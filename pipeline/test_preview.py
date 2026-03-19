from types import SimpleNamespace
from tempfile import TemporaryDirectory

import numpy as np
import torch

from pipeline.preview import (
    build_classifier_preview_frames,
    build_transformer_preview_frames,
    make_transformer_step_preview_callback,
)
from pipeline.weight_image_cache import (
    checkpoint_thumbnail_path,
    checkpoint_thumbnail_root,
    crop_weight_image_rgb,
    save_checkpoint_thumbnail,
)
from wav_ml_core import RenderConfig
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


def test_build_transformer_preview_frames_renders_wave_triplet():
    cfg = RenderConfig(width=32, downsample=1, max_points=32)
    payload_batch = [
        {
            "step": 2,
            "steps_per_epoch": 8,
            "x_clean": torch.linspace(-1.0, 1.0, steps=32, dtype=torch.float32).unsqueeze(0),
            "x_in": torch.zeros((1, 32), dtype=torch.float32),
            "x_out": torch.linspace(1.0, -1.0, steps=32, dtype=torch.float32).unsqueeze(0),
            "loss": 0.4,
            "score_target": 0.25,
            "score_after": 0.6,
            "score_gap": 0.1,
            "denoise_l1": 0.2,
            "high_bits_l1": 0.05,
            "low_bits_l1": 0.03,
            "entropy_excess": 0.02,
            "degrade_strength": 0.3,
        }
    ]

    eff_loss, frames = build_transformer_preview_frames(
        payload_batch,
        render_config=cfg,
        image_hw=(16, 16),
        sample_bits=16,
        class_names=["zero", "one"],
        cycle_id=1,
        round_id=2,
    )

    assert eff_loss == 0.4
    assert len(frames) == 1
    frame = frames[0]
    assert frame["caption"].startswith("[R] cycle=1 round=2 step=2/8")
    assert frame["titles"] == ["R clean", "R input", "R output"]
    assert frame["rows"][0][0] == "target:none"
    assert "degrade=0.300" in frame["rows"][1]
    assert "after=0.6000" in frame["rows"][2]
    assert frame["images"][0].shape == (16, 16, 3)
    assert frame["images"][1].shape == (16, 16, 3)
    assert frame["images"][2].shape == (16, 16, 3)


def test_transformer_preview_callback_enqueues_frames_and_publishes_loss():
    class _FakeViewer:
        def __init__(self):
            self.frames = []

        def enqueue_frame(self, frame):
            self.frames.append(frame)

    published = []
    ctx = SimpleNamespace(
        viewer_proxy=_FakeViewer(),
        render_config=RenderConfig(width=32, downsample=1, max_points=32),
        class_names=["zero", "one"],
        cycle=4,
        round_id=5,
        args=SimpleNamespace(image_size=16, sample_bits=16),
        preview_enabled=lambda: True,
        publish_node_progress=lambda node_id, loss: published.append((node_id, loss)),
    )
    callback = make_transformer_step_preview_callback(ctx, "stage_r_transformer")

    callback(
        [
            {
                "step": 1,
                "steps_per_epoch": 4,
                "x_clean": torch.linspace(-1.0, 1.0, steps=32, dtype=torch.float32).unsqueeze(0),
                "x_in": torch.zeros((1, 32), dtype=torch.float32),
                "x_out": torch.linspace(1.0, -1.0, steps=32, dtype=torch.float32).unsqueeze(0),
                "loss": 0.25,
                "score_target": 0.2,
                "score_after": 0.5,
                "score_gap": 0.1,
                "denoise_l1": 0.15,
                "high_bits_l1": 0.04,
                "low_bits_l1": 0.02,
                "entropy_excess": 0.01,
                "degrade_strength": 0.4,
            }
        ]
    )

    assert len(ctx.viewer_proxy.frames) == 1
    assert published == [("stage_r_transformer", 0.25)]


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
        def __init__(self):
            self._entries = [
                SimpleNamespace(
                    publish_seq=7,
                    model_name="generator",
                    node_id="stage_g",
                    generation=1,
                    architecture_version=11,
                    round_id=1,
                    cycle=2,
                    step=3,
                ),
                SimpleNamespace(
                    publish_seq=8,
                    model_name="discriminator",
                    node_id="stage_g",
                    generation=1,
                    architecture_version=11,
                    round_id=1,
                    cycle=2,
                    step=4,
                ),
            ]

        def get_meta(self):
            return self._entries[-1]

        def list_meta(self):
            return list(self._entries)

        def get_meta_for(self, *, model_name, node_id=""):
            for entry in self._entries:
                if str(entry.model_name) != str(model_name):
                    continue
                if str(entry.node_id) != str(node_id or ""):
                    continue
                return entry
            return None

    class _FakeImageStore:
        def __init__(self):
            self.calls = []
            self.limit_calls = []
            self._image_seq = 0
            self._max_entries = 512
            self._entries = []

        def _find_entry_index(self, *, model_name, node_id, state_publish_seq):
            for index, (meta, _rgb) in enumerate(self._entries):
                if int(meta.state_publish_seq) != int(state_publish_seq):
                    continue
                if str(meta.model_name) != str(model_name):
                    continue
                if str(meta.node_id) != str(node_id):
                    continue
                return index
            return None

        def length(self):
            return len(self._entries)

        def stats(self):
            return SimpleNamespace(max_entries=self._max_entries, max_total_bytes=0)

        def get_meta(self, index):
            if index < 0 or index >= len(self._entries):
                return None
            return self._entries[int(index)][0]

        def set_limits(self, *, max_entries, max_total_bytes):
            self.limit_calls.append((int(max_entries), int(max_total_bytes)))
            return True

        def measure_for(self, state_store, *, model_name, node_id="", mode, target_width, target_height):
            state_meta = state_store.get_meta_for(model_name=str(model_name), node_id=str(node_id or ""))
            if state_meta is None:
                return None
            render_w = int(target_width) + 50
            render_h = int(target_height) + 40
            return SimpleNamespace(
                state_publish_seq=int(state_meta.publish_seq),
                generation=int(state_meta.generation),
                architecture_version=int(state_meta.architecture_version),
                round_id=int(state_meta.round_id),
                cycle=int(state_meta.cycle),
                step=int(state_meta.step),
                mode=int(mode),
                target_width=int(target_width),
                target_height=int(target_height),
                render_width=int(render_w),
                render_height=int(render_h),
                render_channels=3,
                render_stride_bytes=int(render_w * 3),
                model_name=str(model_name),
                node_id=str(node_id or ""),
            )

        def render_for(self, state_store, *, model_name, node_id="", mode, target_width, target_height):
            self.calls.append((str(model_name), int(mode), int(target_width), int(target_height)))
            state_meta = state_store.get_meta_for(model_name=str(model_name), node_id=str(node_id or ""))
            if state_meta is None:
                return False
            cfg = self.measure_for(
                state_store,
                model_name=str(model_name),
                node_id=str(node_id or ""),
                mode=int(mode),
                target_width=int(target_width),
                target_height=int(target_height),
            )
            if cfg is None:
                return False
            self._image_seq += 1
            fill = 64 if str(model_name) == "generator" else 96
            rgb = np.full((int(cfg.render_height), int(cfg.render_width), 3), fill, dtype=np.uint8)
            meta = SimpleNamespace(
                image_seq=self._image_seq,
                state_publish_seq=int(state_meta.publish_seq),
                generation=int(state_meta.generation),
                architecture_version=int(state_meta.architecture_version),
                round_id=int(state_meta.round_id),
                cycle=int(state_meta.cycle),
                step=int(state_meta.step),
                width=int(cfg.render_width),
                height=int(cfg.render_height),
                channels=3,
                stride_bytes=int(cfg.render_stride_bytes),
                mode=int(mode),
                target_width=int(target_width),
                target_height=int(target_height),
                byte_count=int(cfg.render_stride_bytes) * int(cfg.render_height),
                flags=0,
                model_name=str(model_name),
                node_id=str(node_id or ""),
                blob_name=f"blob_{self._image_seq}",
            )
            existing = self._find_entry_index(
                model_name=str(model_name),
                node_id=str(node_id or ""),
                state_publish_seq=int(state_meta.publish_seq),
            )
            if existing is not None:
                self._entries.pop(existing)
            self._entries.append((meta, rgb))
            return True

        def copy_image(self, index):
            if index < 0 or index >= len(self._entries):
                return None
            return self._entries[int(index)][1].copy()

    viewer = _TransformerStatusOpenGLViewer(enabled=False, image_hw=(8, 8), graph_h=0)
    viewer._ipc_server_ref = SimpleNamespace(has_connection=True)
    viewer._weight_state_store = _FakeStateStore()
    image_store = _FakeImageStore()
    image_store._max_entries = viewer._weight_history_maxlen
    viewer._weight_image_store = image_store
    viewer.weight_image_spec = lambda: {"mode": 1, "panel_crop_w": 100, "panel_crop_h": 80}

    viewer._launch_shared_weight_render()
    viewer._shared_weight_render_thread.join(timeout=2.0)
    viewer._collect_shared_weight_render()

    viewer.weight_image_spec = lambda: {"mode": 1, "panel_crop_w": 140, "panel_crop_h": 90}
    viewer._launch_shared_weight_render()
    viewer._shared_weight_render_thread.join(timeout=2.0)
    viewer._collect_shared_weight_render()

    assert image_store.calls == [
        ("generator", 1, 100, 80),
        ("discriminator", 1, 100, 80),
        ("generator", 1, 140, 90),
        ("discriminator", 1, 140, 90),
    ]
    assert image_store.limit_calls == [
        (viewer._weight_history_maxlen, 450 * 120 * viewer._weight_history_maxlen),
        (viewer._weight_history_maxlen, 570 * 130 * viewer._weight_history_maxlen),
    ]
    assert len(viewer._weight_snapshot_deques_by_model["generator"]) == 2
    assert len(viewer._weight_snapshot_deques_by_model["discriminator"]) == 2
    assert viewer._weight_current_rgb_by_model["generator"].shape == (90, 140, 3)
    assert viewer._weight_current_rgb_by_model["discriminator"].shape == (90, 140, 3)
    assert viewer._weight_current_meta_by_model["generator"]["mode"] == 1
    assert viewer._weight_current_meta_by_model["discriminator"]["target_width"] == 140


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


def test_viewer_checkpoint_notification_tracks_all_weight_models():
    viewer = _TransformerStatusOpenGLViewer(enabled=False, image_hw=(8, 8), graph_h=0)

    viewer.notify_pipeline_checkpoint_saved(
        round_id=5,
        cycle=3,
        weight_models=[
            {
                "model": "generator",
                "node_id": "stage_g",
                "publish_seq": 21,
                "generation": 2,
                "architecture_version": 17,
            },
            {
                "model": "discriminator",
                "node_id": "stage_g",
                "publish_seq": 22,
                "generation": 2,
                "architecture_version": 19,
            },
        ],
    )

    gen = viewer._checkpoint_weight_records[(5, 3, "generator")]
    disc = viewer._checkpoint_weight_records[(5, 3, "discriminator")]
    assert gen["state_publish_seq"] == 21
    assert disc["state_publish_seq"] == 22
    assert gen["generation"] == 2
    assert disc["architecture_version"] == 19
