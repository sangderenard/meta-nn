from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import speculative_network as sn
import pipeline.nodes.speculative_node as speculative_node
from pipeline.nodes.speculative_node import BuildSpeculativeNetNode, SpeculativeNetConfig


class _FakeVocabBank:
    def __init__(self, phrases, model_name="unused", device=None, normalize=True):
        self.phrases = list(phrases)
        self.dim = 8
        self._vectors = {}
        for idx, phrase in enumerate(self.phrases):
            vec = torch.zeros(self.dim, dtype=torch.float32)
            vec[idx % self.dim] = 1.0
            vec[(idx + 1) % self.dim] = 0.5
            self._vectors[str(phrase)] = F.normalize(vec, dim=0)

    def get_vector(self, phrase):
        return self._vectors[str(phrase)].clone()


def _dropout_modules(model: nn.Module):
    return [m for m in model.modules() if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d))]


def _zero_module_params(module: nn.Module) -> None:
    for param in module.parameters():
        with torch.no_grad():
            param.zero_()


def test_prototype_auto_classifier_installs_dropout_modules_when_enabled():
    phrases = ["alpha", "beta", "gamma"]
    model = sn.PrototypeAutoClassifier(
        vocab_bank=_FakeVocabBank(phrases),
        row_bank=sn.DynamicRowBank(phrases=phrases, row_dim=16),
        hidden_dim=32,
        n_slots=4,
        image_size=32,
        network_dropout=0.10,
    )

    dropouts = _dropout_modules(model)
    assert dropouts, "expected speculative network to expose dropout modules"
    assert all(abs(float(mod.p) - 0.10) < 1e-6 for mod in dropouts)


def test_build_speculative_net_node_propagates_network_dropout(monkeypatch):
    monkeypatch.setattr(sn, "SentenceTransformerVocabBank", _FakeVocabBank)
    cfg = SpeculativeNetConfig(
        image_size=32,
        hidden_dim=32,
        row_dim=16,
        n_slots=4,
        network_dropout=0.10,
        network_dtype="float32",
    )
    ctx = SimpleNamespace(
        class_names=["alpha", "beta", "gamma"],
        device=torch.device("cpu"),
        args=SimpleNamespace(
            network_preview_output_panels=2,
            network_preview_confidence_floor=0.0,
        ),
        resume_pipeline_ckpt=None,
    )

    BuildSpeculativeNetNode(cfg).execute(ctx)

    dropouts = _dropout_modules(ctx.active_network)
    assert dropouts, "builder should create an active network with dropout modules"
    assert all(abs(float(mod.p) - 0.10) < 1e-6 for mod in dropouts)
    assert abs(float(getattr(ctx.active_network, "_runtime_network_dropout", 0.0)) - 0.10) < 1e-6


def test_hypergraph_prior_prefers_novel_dissimilar_terms():
    prior = sn.ParameterizedHypergraphPrior(vocab_size=3, hidden_dim=4, dropout_p=0.0)
    _zero_module_params(prior.feature_mlp)
    _zero_module_params(prior.context_fuse)
    _zero_module_params(prior.bias_head)
    with torch.no_grad():
        prior.term_embeddings.zero_()
        prior.predictive_node_state.copy_(torch.tensor([0.90, 0.65, 0.05], dtype=torch.float32))
        prior.predictive_pair_state.copy_(
            torch.tensor(
                [
                    [1.00, 0.85, 0.02],
                    [0.85, 1.00, 0.10],
                    [0.02, 0.10, 1.00],
                ],
                dtype=torch.float32,
            )
        )

    bias = prior(
        base_log_prior=torch.log(torch.tensor([0.80, 0.35, 0.05], dtype=torch.float32)),
        pair_log_prior=torch.log(
            torch.tensor(
                [
                    [1.00, 0.90, 0.05],
                    [0.90, 1.00, 0.10],
                    [0.05, 0.10, 1.00],
                ],
                dtype=torch.float32,
            )
        ),
        selected_mass=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
        selected_context=torch.zeros(1, 4, dtype=torch.float32),
        semantic_similarity=torch.tensor(
            [
                [1.00, 0.95, 0.05],
                [0.95, 1.00, 0.10],
                [0.05, 0.10, 1.00],
            ],
            dtype=torch.float32,
        ),
    )

    # With term 0 already selected, term 2 should be preferred over the
    # frequent/co-occurring/semantically similar term 1.
    assert float(bias[0, 2]) > float(bias[0, 1])


class _DummySpeculativeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward_batch(self, image, batch_vocab):
        batch_n = int(image.shape[0])
        slot_vectors = self.weight.view(1, 1, 1).expand(batch_n, 1, 1)
        slot_masks = self.weight.view(1, 1, 1, 1).expand(batch_n, 1, 1, 1)
        slot_confidence = self.weight.view(1, 1).expand(batch_n, 1)
        return {
            "slot_vectors": slot_vectors,
            "slot_masks": slot_masks,
            "slot_confidence": slot_confidence,
            "aux": {},
        }


class _DummySpeculativeCriterion:
    def __init__(self):
        self.vocab_matrix = torch.zeros(1, 1, dtype=torch.float32)

    def __call__(
        self,
        *,
        pred_vectors,
        pred_masks,
        target_vectors,
        target_masks,
        target_valid,
        present_mask,
        confidence_logits,
        slot_selection_logits,
    ):
        loss = (
            (pred_vectors - target_vectors).pow(2).mean()
            + (pred_masks - target_masks).pow(2).mean()
            + confidence_logits.pow(2).mean()
        )
        zero = loss.detach() * 0.0
        return {
            "loss": loss,
            "vector_loss": loss.detach(),
            "mask_loss": zero,
            "selection_loss": zero,
            "confidence_loss": zero,
            "assignments": [],
        }


def _fake_speculative_batch(
    xb,
    mb,
    meta,
    *,
    vocab_phrases,
    criterion_vocab_matrix,
    active_term_to_idx,
    device,
):
    batch_n = int(xb.shape[0])
    return {
        "image": xb.to(device=device, dtype=torch.float32),
        "batch_vocab": [["alpha"] for _ in range(batch_n)],
        "target_vectors": torch.zeros(batch_n, 1, 1, device=device, dtype=torch.float32),
        "target_masks": torch.zeros(batch_n, 1, 1, 1, device=device, dtype=torch.float32),
        "present_mask": torch.ones(batch_n, 1, device=device, dtype=torch.float32),
        "target_valid": torch.ones(batch_n, 1, device=device, dtype=torch.bool),
    }


def _make_speculative_loader(num_batches: int, batch_size: int = 2):
    return [
        {
            "x": torch.ones(batch_size, 1, 1, 1, dtype=torch.float32),
            "mask": torch.zeros(batch_size, 1, 1, dtype=torch.float32),
            "terms_rows": [["alpha"] for _ in range(batch_size)],
        }
        for _ in range(num_batches)
    ]


@pytest.mark.parametrize(
    ("grad_accum_steps", "epochs", "expected_optimizer_steps"),
    [
        (0, 2, 6),
        (2, 2, 4),
        (-1, 2, 2),
    ],
)
def test_run_speculative_net_epochs_respects_grad_accum_steps(
    monkeypatch,
    grad_accum_steps,
    epochs,
    expected_optimizer_steps,
):
    monkeypatch.setattr(speculative_node, "_build_speculative_batch", _fake_speculative_batch)

    net = _DummySpeculativeNet()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    criterion = _DummySpeculativeCriterion()
    optimizer_updates = []

    result = speculative_node._run_speculative_net_epochs(
        net=net,
        criterion=criterion,
        optimizer=optimizer,
        loader=_make_speculative_loader(num_batches=3, batch_size=2),
        device=torch.device("cpu"),
        epochs=epochs,
        vocab_phrases=["alpha"],
        active_term_to_idx={"alpha": 0},
        grad_clip=1.0,
        grad_accum_steps=grad_accum_steps,
        log_every=0,
        stage_label="spec-test",
        weight_update_callback=optimizer_updates.append,
    )

    assert result["ran"] is True
    assert int(result["steps"]) == 3 * epochs
    assert int(result["optimizer_steps"]) == expected_optimizer_steps
    assert optimizer_updates == list(range(1, expected_optimizer_steps + 1))
