from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from pipeline.nodes.classifier_node import (
    ClassifierConfig,
    FakeClassFeedbackNode,
    _run_fake_class_refresh_epochs,
)


def test_fake_feedback_node_uses_current_refresh_signature():
    captured = {}
    fake_vec = torch.ones(8, dtype=torch.float32)

    class _FakeViewer:
        def enqueue_frame(self, _frame):
            pass

        def send_execution_event(self, _payload):
            pass

    ctx = SimpleNamespace(
        classifier=torch.nn.Linear(1, 1),
        generator=SimpleNamespace(z_dim=64),
        discriminator=object(),
        device=torch.device("cpu"),
        amp_enabled=False,
        amp_dtype="float16",
        class_names=["one", "two"],
        payload_conditions=[np.array([1.0, 0.0], dtype=np.float32)],
        payload_masks=[np.ones((8, 8), dtype=np.float32)],
        payload_terms=[["one", "berkeley sbd dataset"]],
        args=SimpleNamespace(
            fake_image_sentinel_label="GAN image",
            label_embedding_backend="sentence_transformers",
            label_embedding_model="dummy-model",
            generator_fake_feedback_epochs=2,
            generator_fake_feedback_lr=0.001,
            generator_fake_feedback_weight_decay=0.0001,
            lr_sine_cycles=1.5,
            lr_sine_frequency=0.25,
            lr_sine_tail_fraction=0.2,
            lr_sine_min_scale=0.1,
            generator_fake_feedback_log_every=3,
            generator_fake_feedback_include_condition_targets=True,
            generator_fake_feedback_vector_weight=1.0,
            generator_fake_feedback_condition_weight=0.35,
            generator_fake_feedback_disc_conf_temperature=1.0,
            generator_fake_feedback_disc_conf_floor=0.25,
            generator_fake_feedback_disc_balance_groups=True,
            seed=7,
        ),
        stop_requested=lambda: False,
        viewer_proxy=_FakeViewer(),
        publish_node_progress=lambda *_args, **_kwargs: None,
        log_metric=lambda *_args, **_kwargs: None,
        round_id=4,
    )

    cfg = ClassifierConfig(fake_class_enabled=True, fake_class_steps=11, fake_class_batch_size=5, grad_clip=1.25)
    node = FakeClassFeedbackNode(cfg)

    def _capture_refresh(**kwargs):
        captured["kwargs"] = kwargs
        return {"loss": 0.5}

    with patch("pipeline.nodes.classifier_node.ensure_vocab_lora_active", lambda *_args, **_kwargs: None), patch(
        "pipeline.nodes.classifier_node.make_runtime_weight_publish_callback", lambda *_args, **_kwargs: None
    ), patch(
        "pipeline.nodes.classifier_node._resolve_fake_feedback_label_vector", lambda _ctx: fake_vec
    ), patch(
        "pipeline.nodes.classifier_node._run_fake_class_refresh_epochs",
        _capture_refresh,
    ):
        node.execute(ctx)

    kwargs = captured["kwargs"]
    assert "optimizer" not in kwargs
    assert len(kwargs["payload_conditions"]) == len(ctx.payload_conditions)
    assert np.allclose(kwargs["payload_conditions"][0], ctx.payload_conditions[0])
    assert len(kwargs["payload_masks"]) == len(ctx.payload_masks)
    assert np.allclose(kwargs["payload_masks"][0], ctx.payload_masks[0])
    assert len(kwargs["payload_terms"]) == len(ctx.payload_terms)
    assert kwargs["payload_terms"][0] == ctx.payload_terms[0]
    assert kwargs["condition_num_classes"] == 2
    assert torch.equal(kwargs["fake_label_vector"], fake_vec)
    assert kwargs["z_dim"] == 64
    assert kwargs["epochs"] == 2
    assert kwargs["steps_per_epoch"] == 11
    assert kwargs["batch_size"] == 5
    assert kwargs["progress_callback"] is not None


def test_fake_refresh_requires_mask_bundle():
    fake_vec = torch.ones(8, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="payload_masks"):
        _run_fake_class_refresh_epochs(
            classifier=torch.nn.Linear(1, 1),
            generator=object(),
            discriminator=None,
            payload_conditions=[np.array([1.0, 0.0], dtype=np.float32)],
            payload_masks=[],
            payload_terms=[["one"]],
            condition_num_classes=2,
            fake_label_vector=fake_vec,
            z_dim=8,
            device=torch.device("cpu"),
            epochs=1,
            steps_per_epoch=1,
            batch_size=1,
            lr=1e-3,
            weight_decay=0.0,
            lr_sine_cycles=1.0,
            lr_sine_frequency=0.0,
            lr_sine_tail_fraction=0.0,
            lr_sine_min_scale=0.0,
        )


def test_fake_refresh_requires_row_provenance_bundle():
    fake_vec = torch.ones(8, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="payload_terms"):
        _run_fake_class_refresh_epochs(
            classifier=torch.nn.Linear(1, 1),
            generator=object(),
            discriminator=None,
            payload_conditions=[np.array([1.0, 0.0], dtype=np.float32)],
            payload_masks=[np.ones((8, 8), dtype=np.float32)],
            payload_terms=[],
            condition_num_classes=2,
            fake_label_vector=fake_vec,
            z_dim=8,
            device=torch.device("cpu"),
            epochs=1,
            steps_per_epoch=1,
            batch_size=1,
            lr=1e-3,
            weight_decay=0.0,
            lr_sine_cycles=1.0,
            lr_sine_frequency=0.0,
            lr_sine_tail_fraction=0.0,
            lr_sine_min_scale=0.0,
        )


def test_fake_refresh_preview_bundle_keeps_masks_attached():
    from wav_ml_models import TinyConvClassifier

    captured_batches = []

    class _DummyGenerator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.z_dim = 8

        def forward(self, z, cond):
            batch = int(z.shape[0])
            return torch.sigmoid(torch.randn((batch, 3, 32, 32), device=z.device, dtype=torch.float32))

    class _DummyDiscriminator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.last_mask = None

        def forward(self, x, cond, mask):
            self.last_mask = mask.detach().cpu()
            return torch.zeros((int(x.shape[0]),), device=x.device, dtype=torch.float32)

    classifier = TinyConvClassifier(
        num_classes=2,
        base_ch=16,
        max_ch=32,
        context_blocks=0,
        mask_decoder_channels=16,
    )
    generator = _DummyGenerator()
    discriminator = _DummyDiscriminator()
    fake_vec = torch.ones(int(classifier.embed_proj.out_features), dtype=torch.float32)

    result = _run_fake_class_refresh_epochs(
        classifier=classifier,
        generator=generator,
        discriminator=discriminator,
        payload_conditions=[np.array([1.0, 0.0], dtype=np.float32)],
        payload_masks=[np.ones((32, 32), dtype=np.float32)],
        payload_terms=[["one", "berkeley sbd dataset"]],
        condition_num_classes=2,
        fake_label_vector=fake_vec,
        z_dim=8,
        device=torch.device("cpu"),
        epochs=1,
        steps_per_epoch=1,
        batch_size=1,
        lr=1e-3,
        weight_decay=0.0,
        lr_sine_cycles=1.0,
        lr_sine_frequency=0.0,
        lr_sine_tail_fraction=0.0,
        lr_sine_min_scale=0.0,
        step_preview_callback=lambda batch: captured_batches.append(batch),
    )

    assert result["ran"] is True
    assert captured_batches
    preview_item = captured_batches[0][0]
    assert preview_item["payload_row_idx"] == 0
    assert preview_item["payload_terms"] == ["one", "berkeley sbd dataset"]
    assert preview_item["target_mask"] is not None
    assert preview_item["detected_mask"] is not None
    assert preview_item["mask_source"] == "classifier_detected"
    assert discriminator.last_mask is not None
    assert tuple(discriminator.last_mask.shape) == (1, 1, 32, 32)
