from types import SimpleNamespace

import torch

from pipeline.nodes.save_restore_node import SaveRestoreNode


class _FakeWeightStateStore:
    def __init__(self):
        self.calls = []
        self._last_meta = None

    def publish_state_dict(self, state_dict, **kwargs):
        self.calls.append(dict(kwargs))
        self._last_meta = SimpleNamespace(
            publish_seq=len(self.calls),
            generation=int(kwargs.get("generation", 0)),
            architecture_version=int(kwargs.get("architecture_version", 0)),
            round_id=int(kwargs.get("round_id", 0)),
            cycle=int(kwargs.get("cycle", 0)),
            step=int(kwargs.get("step", 0)),
            param_count=len(state_dict),
            blob_bytes=1,
            model_name=str(kwargs.get("model_name", "")),
            node_id=str(kwargs.get("node_id", "")),
            blob_name="blob",
        )
        return self._last_meta

    def get_meta(self):
        return self._last_meta


class _FakeWeightImageStore:
    def get_active_config(self):
        return None

    def configure_latest(self, *_args, **_kwargs):
        return None

    def length(self):
        return 0

    def capacity(self):
        return 1

    def stats(self):
        return SimpleNamespace(max_total_bytes=0)

    def get_meta(self, _index):
        return None


def test_publish_runtime_weight_state_tracks_generation_on_architecture_change():
    sr = SaveRestoreNode()
    sr.weight_state_store = _FakeWeightStateStore()
    sr.weight_image_store = _FakeWeightImageStore()

    model = torch.nn.Sequential(
        torch.nn.Linear(4, 3),
        torch.nn.ReLU(),
        torch.nn.Linear(3, 2),
    )
    notification1 = sr.publish_runtime_weight_state(
        "classifier",
        model,
        node_id="stage_0_pregestation",
        round_id=1,
        cycle=2,
        step=3,
    )
    notification2 = sr.publish_runtime_weight_state(
        "classifier",
        model,
        node_id="stage_0_pregestation",
        round_id=1,
        cycle=2,
        step=4,
    )

    wider = torch.nn.Sequential(
        torch.nn.Linear(4, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 2),
    )
    notification3 = sr.publish_runtime_weight_state(
        "classifier",
        wider,
        node_id="stage_0_pregestation",
        round_id=1,
        cycle=2,
        step=5,
    )

    assert notification1 is not None
    assert notification2 is not None
    assert notification3 is not None
    assert sr.weight_state_store.calls[0]["generation"] == 1
    assert sr.weight_state_store.calls[1]["generation"] == 1
    assert sr.weight_state_store.calls[2]["generation"] == 2
    assert sr.weight_state_store.calls[0]["architecture_version"] == sr.weight_state_store.calls[1]["architecture_version"]
    assert sr.weight_state_store.calls[2]["architecture_version"] != sr.weight_state_store.calls[1]["architecture_version"]
    assert notification3["generation"] == 2


def test_make_checkpoint_notification_includes_all_runtime_weight_models():
    sr = SaveRestoreNode()
    generator_meta = SimpleNamespace(
        publish_seq=21,
        generation=2,
        architecture_version=17,
        round_id=5,
        cycle=3,
        step=11,
        model_name="generator",
        node_id="stage_g",
    )
    discriminator_meta = SimpleNamespace(
        publish_seq=22,
        generation=2,
        architecture_version=19,
        round_id=5,
        cycle=3,
        step=12,
        model_name="discriminator",
        node_id="stage_g",
    )
    sr._last_weight_state_meta = discriminator_meta
    sr._weight_model_registry = {
        ("generator", "stage_g"): generator_meta,
        ("discriminator", "stage_g"): discriminator_meta,
    }

    notification = sr.make_checkpoint_notification(round_id=5, cycle=3)

    assert notification["weight_model"] == "discriminator"
    assert {(entry["model"], entry["publish_seq"]) for entry in notification["weight_models"]} == {
        ("generator", 21),
        ("discriminator", 22),
    }
