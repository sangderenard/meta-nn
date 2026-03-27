"""
Standalone training test: TinyConvClassifier + HypergraphNet

Verifies:
  - HypergraphNet attaches to TinyConvClassifier cleanly
  - forward_with_aux emits both logits and hypergraph_logits
  - Combined loss back-propagates through both networks
  - All four independent training-mode configurations work
  - observe_hypergraph registers neurons and populates edge data
  - Loss is finite and the combined system can take gradient steps

Run from the repo root:
    python _test_hypergraph_net.py
"""
from __future__ import annotations

import sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# ---------------------------------------------------------------------------
# Repo imports
# ---------------------------------------------------------------------------
from pipeline.vocabulary_defaults import DEFAULT_VOCABULARY
from pipeline.word_cooccurrence_hypergraph import HypergraphNet, WordCooccurrenceHypergraph
from wav_ml_models import TinyConvClassifier


# ---------------------------------------------------------------------------
# Constants matching the live repo configuration
# ---------------------------------------------------------------------------
NUM_CLASSES       = 151   # 101 supervised + 50 dynamic (from architecture snapshot)
POOLED_FEAT_DIM   = 256   # c3 with base_ch=64, max_ch=384
HIDDEN_WIDTH      = 256
MAX_HYPEREDGE_LEN = 32    # declared max terms per row for the vocab extension net
BATCH             = 4
IMAGE_H = IMAGE_W = 64


def _ok(msg: str) -> None:
    print(f"  [ok] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Synthetic term rows
# ---------------------------------------------------------------------------

def _make_term_rows(vocab: list[str], n_rows: int = 120, seed: int = 0) -> list[list[str]]:
    """
    Generate synthetic co-occurrence rows from the vocabulary.
    Each row picks 2–8 terms at random (simulating real dataset hyperedges).
    """
    rng = random.Random(seed)
    rows = []
    for _ in range(n_rows):
        k = rng.randint(2, 8)
        rows.append(rng.sample(vocab, min(k, len(vocab))))
    return rows


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _combined_loss(
    out: dict,
    targets: torch.Tensor,
) -> torch.Tensor:
    """BCE on both logit heads, summed."""
    loss = F.binary_cross_entropy_with_logits(out["logits"], targets)
    if "hypergraph_logits" in out:
        loss = loss + F.binary_cross_entropy_with_logits(out["hypergraph_logits"], targets)
    return loss


def _one_step(
    classifier: TinyConvClassifier,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    targets: torch.Tensor,
    active_hg_keys: list[str] | None = None,
) -> float:
    optimizer.zero_grad()
    out = classifier.forward_with_aux(images, active_hg_keys)
    loss = _combined_loss(out, targets)
    loss.backward()
    optimizer.step()
    return float(loss.item())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_attach_and_forward() -> None:
    print("test_attach_and_forward")

    classifier = TinyConvClassifier(
        num_classes=NUM_CLASSES,
        base_ch=64,
        max_ch=384,
        context_blocks=2,   # fewer blocks for speed
    )
    hg_net = HypergraphNet(
        input_width=POOLED_FEAT_DIM,
        hidden_width=HIDDEN_WIDTH,
        output_width=NUM_CLASSES,
        max_hyperedge_length=MAX_HYPEREDGE_LEN,
    )
    classifier.attach_hypergraph_net(hg_net)

    assert classifier.hypergraph_net is hg_net, "hypergraph_net not attached"
    assert "hypergraph_net" in dict(classifier.named_modules()), "not in module tree"
    _ok("attach_hypergraph_net registers as a submodule")

    images = torch.randn(BATCH, 3, IMAGE_H, IMAGE_W)
    out = classifier.forward_with_aux(images)
    assert "logits" in out, "missing logits"
    assert "hypergraph_logits" in out, "missing hypergraph_logits"
    assert out["logits"].shape == (BATCH, NUM_CLASSES)
    assert out["hypergraph_logits"].shape == (BATCH, NUM_CLASSES)
    _ok("forward_with_aux emits logits and hypergraph_logits with correct shapes")


def test_observe_hypergraph() -> None:
    print("test_observe_hypergraph")

    extra_vocab = [f"extra_term_{i}" for i in range(50)]
    full_vocab = DEFAULT_VOCABULARY + extra_vocab
    term_rows = _make_term_rows(full_vocab, n_rows=120)

    hg = WordCooccurrenceHypergraph(term_rows)
    n_unique = len(hg.unique_edges)
    assert len(hg) == 120
    assert n_unique > 0
    _ok(f"hypergraph built: {len(hg)} rows, {n_unique} unique edges, {len(hg.nodes)} nodes")

    hg_net = HypergraphNet(
        input_width=POOLED_FEAT_DIM,
        hidden_width=HIDDEN_WIDTH,
        output_width=NUM_CLASSES,
        max_hyperedge_length=MAX_HYPEREDGE_LEN,
    )
    hg_net.observe_hypergraph(hg)

    assert len(hg_net.neuron_library) == len(hg.nodes), \
        f"expected {len(hg.nodes)} neurons, got {len(hg_net.neuron_library)}"
    assert len(hg_net._edge_data) == n_unique
    _ok(f"observe_hypergraph registered {len(hg_net.neuron_library)} neurons and {len(hg_net._edge_data)} edges")

    # Weight composition should now use real edge data (not stub)
    w = hg_net._compose_dynamic_head_weight()
    assert w.shape == (MAX_HYPEREDGE_LEN, MAX_HYPEREDGE_LEN)
    assert w is not hg_net._stub_head_weight, "should not fall back to stub after observe"
    assert torch.isfinite(w).all(), "composed weight contains non-finite values"
    _ok("_compose_dynamic_head_weight returns finite (mhl × mhl) tensor after observe")


def test_combined_training() -> None:
    print("test_combined_training")

    extra_vocab = [f"extra_term_{i}" for i in range(50)]
    full_vocab = DEFAULT_VOCABULARY + extra_vocab
    term_rows = _make_term_rows(full_vocab, n_rows=120)
    hg = WordCooccurrenceHypergraph(term_rows)

    classifier = TinyConvClassifier(
        num_classes=NUM_CLASSES,
        base_ch=64,
        max_ch=384,
        context_blocks=2,
    )
    hg_net = HypergraphNet(
        input_width=POOLED_FEAT_DIM,
        hidden_width=HIDDEN_WIDTH,
        output_width=NUM_CLASSES,
        max_hyperedge_length=MAX_HYPEREDGE_LEN,
    )
    hg_net.observe_hypergraph(hg)
    classifier.attach_hypergraph_net(hg_net)

    optimizer = Adam(classifier.parameters(), lr=1e-3)
    images  = torch.randn(BATCH, 3, IMAGE_H, IMAGE_W)
    targets = torch.zeros(BATCH, NUM_CLASSES)
    # Randomly activate a handful of classes per sample
    rng = random.Random(42)
    for b in range(BATCH):
        for c in rng.sample(range(NUM_CLASSES), 5):
            targets[b, c] = 1.0

    active_keys = rng.sample(full_vocab, 10)

    losses = []
    for step in range(5):
        loss = _one_step(classifier, optimizer, images, targets, active_keys)
        assert torch.isfinite(torch.tensor(loss)), f"non-finite loss at step {step}"
        losses.append(loss)

    _ok(f"5 steps completed, losses: {[f'{l:.4f}' for l in losses]}")
    # Not asserting strict decrease — 5 steps with random data is noisy —
    # but at least it must not explode.
    assert losses[-1] < losses[0] * 10, "loss exploded"
    _ok("loss is stable")


def test_training_mode_configurations() -> None:
    print("test_training_mode_configurations")

    classifier = TinyConvClassifier(num_classes=NUM_CLASSES, base_ch=64, max_ch=384, context_blocks=2)
    hg_net = HypergraphNet(
        input_width=POOLED_FEAT_DIM,
        hidden_width=HIDDEN_WIDTH,
        output_width=NUM_CLASSES,
        max_hyperedge_length=MAX_HYPEREDGE_LEN,
    )
    classifier.attach_hypergraph_net(hg_net)

    images  = torch.randn(BATCH, 3, IMAGE_H, IMAGE_W)
    targets = torch.zeros(BATCH, NUM_CLASSES)
    targets[:, :5] = 1.0

    def _backbone_params():
        return [p for n, p in classifier.named_parameters() if "hypergraph_net" not in n]

    def _hgnet_params():
        return list(hg_net.parameters())

    configs = [
        ("both train",         True,  True),
        ("backbone only",      True,  False),
        ("hypernet only",      False, True),
        ("neither (inference)", False, False),
    ]

    for label, backbone_train, hg_train in configs:
        classifier.set_backbone_training(backbone_train)
        classifier.set_hypergraph_net_training(hg_train)

        bp_grad = any(p.requires_grad for p in _backbone_params())
        hp_grad = any(p.requires_grad for p in _hgnet_params())
        assert bp_grad == backbone_train, f"{label}: backbone grad mismatch"
        assert hp_grad == hg_train,       f"{label}: hypernet grad mismatch"

        # Forward + backward must not crash regardless of mode
        out = classifier.forward_with_aux(images)
        loss = _combined_loss(out, targets)
        if backbone_train or hg_train:
            loss.backward()

        _ok(f"{label}: forward+backward OK, backbone_grad={bp_grad}, hgnet_grad={hp_grad}")


def test_device_consistency() -> None:
    print("test_device_consistency")

    if not torch.cuda.is_available():
        print("  [skip] CUDA not available")
        return

    device = torch.device("cuda")
    classifier = TinyConvClassifier(num_classes=NUM_CLASSES, base_ch=64, max_ch=384, context_blocks=2)
    hg_net = HypergraphNet(
        input_width=POOLED_FEAT_DIM,
        hidden_width=HIDDEN_WIDTH,
        output_width=NUM_CLASSES,
        max_hyperedge_length=MAX_HYPEREDGE_LEN,
    )

    extra_vocab = [f"extra_term_{i}" for i in range(10)]
    hg_net.observe_hypergraph(WordCooccurrenceHypergraph(_make_term_rows(DEFAULT_VOCABULARY + extra_vocab, n_rows=20)))
    classifier.attach_hypergraph_net(hg_net)
    classifier.to(device)

    images = torch.randn(BATCH, 3, IMAGE_H, IMAGE_W, device=device)
    out = classifier.forward_with_aux(images)
    assert out["logits"].device.type == "cuda"
    assert out["hypergraph_logits"].device.type == "cuda"
    _ok("all outputs on CUDA after .to(device)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)

    test_attach_and_forward()
    test_observe_hypergraph()
    test_combined_training()
    test_training_mode_configurations()
    test_device_consistency()

    print("\nAll tests passed.")
