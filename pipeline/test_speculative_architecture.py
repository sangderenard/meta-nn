from __future__ import annotations

import torch

from pipeline.network_api import coerce_network_output
from speculative_network import DynamicRowBank, PrototypeAutoClassifier


class _FakeVocabBank:
    def __init__(self, phrases: list[str], dim: int = 32) -> None:
        self.phrases = list(phrases)
        self._dim = int(dim)
        self._vectors = {}
        for idx, phrase in enumerate(self.phrases):
            vec = torch.zeros(self._dim, dtype=torch.float32)
            vec[idx % self._dim] = 1.0
            self._vectors[str(phrase)] = vec

    @property
    def dim(self) -> int:
        return self._dim

    def get_vector(self, phrase: str, device: torch.device | None = None) -> torch.Tensor:
        vec = self._vectors[str(phrase)]
        return vec if device is None else vec.to(device)


def test_speculative_network_forward_batch_preserves_contract_with_multiscale_mask_head():
    phrases = ["a", "b", "c", "d"]
    vocab_bank = _FakeVocabBank(phrases, dim=32)
    row_bank = DynamicRowBank(phrases=phrases, row_dim=64)
    model = PrototypeAutoClassifier(
        vocab_bank=vocab_bank,
        row_bank=row_bank,
        hidden_dim=128,
        n_slots=4,
        image_size=64,
        in_channels=4,
    )

    image = torch.rand(2, 4, 64, 64, dtype=torch.float32)
    vocab = [list(phrases), list(phrases)]
    out = coerce_network_output(model.forward_batch(image, vocab))

    assert out.slot_vectors.shape == (2, 4, 32)
    assert out.slot_masks.shape == (2, 4, 64, 64)
    assert out.slot_confidence.shape == (2, 4)
    assert torch.isfinite(out.slot_vectors).all()
    assert torch.isfinite(out.slot_masks).all()
    assert torch.isfinite(out.slot_confidence).all()

    aux = dict(out.aux)
    spatial_memory = aux.get("spatial_memory")
    selection_logits = aux.get("slot_selection_logits")
    assert isinstance(spatial_memory, torch.Tensor)
    assert spatial_memory.ndim == 3
    assert spatial_memory.shape[0] == 2
    assert spatial_memory.shape[2] == 128
    assert isinstance(selection_logits, torch.Tensor)
    assert selection_logits.shape == (2, 4, len(phrases))
