"""
Network Contract — the standard API for any pluggable recognition network
in the sequential learning pipeline.

Design principle
----------------
The pipeline is not about *these* networks.  It is a sequential learning
system that must work with *any* network.  This module defines the single
data surface (``NetworkOutput``) and the protocol (``NetworkContract``) that
any conforming network must satisfy.  Training loops, gate nodes, and the
preview system speak only this contract — they never reference architecture-
specific attributes.

NetworkOutput fields
--------------------
slot_vectors   [B, N, D]    Per-slot predicted embedding (un-normalised).
slot_masks     [B, N, H, W] Per-slot mask logits (pre-sigmoid).
slot_confidence [B, N]      Per-slot confidence logits (pre-sigmoid).
assignments    List[Dict]   One-to-one slot→vocab bindings (from matcher).
aux            Dict         Anything extra the implementation wants to surface.

Confidence semantics
--------------------
Each slot can express how much it believes it found a real vocabulary
concept:

  sigmoid(slot_confidence[b, i]) ≈ 1  →  "I found something real here."
  sigmoid(slot_confidence[b, i]) ≈ 0  →  "Dustbin — nothing useful."

Training target: 1.0 for slots assigned to a present vocabulary item by
the one-to-one matcher; 0.0 for dustbin slots.  This lets gates and
previews threshold on certainty independently of mask quality, and lets
the network learn explicit "I don't know" behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Protocol, runtime_checkable

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Standard output surface
# ---------------------------------------------------------------------------


@dataclass
class NetworkOutput:
    """Structured result of one NetworkContract.forward_batch call.

    All tensors live on the same device as the input image.  Callers should
    apply .sigmoid() to mask / confidence logits where probabilities are
    needed — the raw logits are stored here so the loss function receives
    gradients through them.
    """

    # Per-slot predicted embedding vectors — [B, N, D], NOT normalised.
    # The training criterion normalises internally; callers should not.
    slot_vectors: torch.Tensor

    # Per-slot spatial mask — [B, N, H, W], pre-sigmoid logits.
    # sigmoid(slot_masks[b, i]) gives the per-pixel probability that slot i
    # covers the vocabulary concept it was assigned to.
    slot_masks: torch.Tensor

    # Per-slot confidence — [B, N], pre-sigmoid logits.
    # Each slot declares how sure it is that it found something real.
    # High = real concept found.  Low = background / dustbin.
    # Gates and previews can threshold on .sigmoid() independently of masks.
    slot_confidence: torch.Tensor

    # One dict per sample in the batch.  Populated by the training criterion
    # after one-to-one matching; empty list when called without a criterion
    # (e.g. inference-only forward pass).
    # Schema per entry:
    #   "slot_to_target": List[Optional[int]]  — vocab index per slot, None = dustbin
    #   "valid_target_count": int
    assignments: List[Dict[str, Any]] = field(default_factory=list)

    # Arbitrary tensors / scalars the network implementation wants to surface.
    # Useful for: feature maps, attention weights, gate diagnostics.
    aux: Dict[str, Any] = field(default_factory=dict)


def coerce_network_output(raw: Any) -> NetworkOutput:
    """Convert a contract return value into a canonical NetworkOutput."""

    if isinstance(raw, NetworkOutput):
        return raw
    if not isinstance(raw, dict):
        raise TypeError(f"Network output must be NetworkOutput or dict, got {type(raw).__name__}")

    slot_vectors = raw.get("slot_vectors", raw.get("pred_vectors"))
    slot_masks = raw.get("slot_masks", raw.get("pred_masks"))
    slot_confidence = raw.get("slot_confidence", raw.get("confidence_logits"))
    if not isinstance(slot_vectors, torch.Tensor):
        raise TypeError("Network output is missing tensor field 'slot_vectors'/'pred_vectors'")
    if not isinstance(slot_masks, torch.Tensor):
        raise TypeError("Network output is missing tensor field 'slot_masks'/'pred_masks'")
    if not isinstance(slot_confidence, torch.Tensor):
        raise TypeError("Network output is missing tensor field 'slot_confidence'/'confidence_logits'")

    assignments = raw.get("assignments")
    aux = raw.get("aux")
    if not isinstance(assignments, list):
        assignments = []
    if not isinstance(aux, dict):
        aux = {
            k: v
            for k, v in raw.items()
            if k
            not in {
                "slot_vectors",
                "pred_vectors",
                "slot_masks",
                "pred_masks",
                "slot_confidence",
                "confidence_logits",
                "assignments",
                "aux",
            }
        }

    return NetworkOutput(
        slot_vectors=slot_vectors,
        slot_masks=slot_masks,
        slot_confidence=slot_confidence,
        assignments=assignments,
        aux=aux,
    )


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class NetworkContract(Protocol):
    """Protocol any pluggable recognition network must satisfy.

    The pipeline receives an already-constructed object and verifies
    conformance via ``isinstance(obj, NetworkContract)`` at build time.
    It calls only ``forward_batch`` and the standard ``nn.Module`` interface
    at train / inference time — it never reads architecture-specific fields.

    Minimum implementation checklist
    ---------------------------------
    * ``vocab_phrases`` property   → List[str]
    * ``n_slots`` property         → int
    * ``forward_batch(image, vocab)`` → NetworkOutput  (or dict with same keys)
    * ``parameters()``, ``train()``, ``eval()``, ``to()``

    How to satisfy this protocol
    ----------------------------
    Subclass ``nn.Module`` and add ``vocab_phrases``, ``n_slots``, and
    ``forward_batch``.  Python's ``@runtime_checkable`` protocol checks
    attribute existence at runtime; plain instance attributes satisfy
    ``@property`` slots in the protocol, so ``self.n_slots = 8`` in
    ``__init__`` is sufficient.

    Example skeleton::

        class MyNet(nn.Module):
            def __init__(self, phrases, n_slots, ...):
                super().__init__()
                self.vocab_phrases = list(phrases)
                self.n_slots = n_slots
                ...

            def forward_batch(self, image, vocab):
                out = self.forward(image, vocab)
                return NetworkOutput(
                    slot_vectors=out["vectors"],
                    slot_masks=out["masks"],
                    slot_confidence=out["confidence"],
                )
    """

    @property
    def vocab_phrases(self) -> List[str]:
        """All vocabulary phrases this network was built for."""
        ...

    @property
    def n_slots(self) -> int:
        """Number of output slots (concepts attended to per forward pass)."""
        ...

    def forward_batch(
        self,
        image: torch.Tensor,
        vocab: List[List[str]],
    ) -> "NetworkOutput":
        """Run one forward pass; return structured, device-consistent output.

        Args:
            image: ``[B, C, H, W]`` on the network's device.  ``C`` may be 3
                   (RGB) or 4 (RGBA); implementations should handle both.
            vocab: ``B`` per-sample vocabulary lists — typically identical
                   rows taken from ``ctx.class_names``.

        Returns:
            ``NetworkOutput`` with ``slot_vectors``, ``slot_masks``, and
            ``slot_confidence`` populated.  ``assignments`` may be left empty
            (the training loop fills it via the criterion).
        """
        ...

    def parameters(self) -> Iterator[nn.Parameter]:
        ...

    def train(self, mode: bool = True) -> "NetworkContract":
        ...

    def eval(self) -> "NetworkContract":
        ...

    def to(self, *args: Any, **kwargs: Any) -> "NetworkContract":
        ...
