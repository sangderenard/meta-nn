from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from pipeline.nodes.classifier_node import ClassifierConfig, FakeClassFeedbackNode


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
    assert kwargs["condition_num_classes"] == 2
    assert torch.equal(kwargs["fake_label_vector"], fake_vec)
    assert kwargs["z_dim"] == 64
    assert kwargs["epochs"] == 2
    assert kwargs["steps_per_epoch"] == 11
    assert kwargs["batch_size"] == 5
    assert kwargs["progress_callback"] is not None
