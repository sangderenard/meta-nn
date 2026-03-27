from __future__ import annotations

import torch

from speculative_network import OneToOneMatcher, PrototypeLoss


def test_matcher_sanitizes_nonfinite_slot_vectors_and_uses_dustbin():
    matcher = OneToOneMatcher(dustbin_cost=1.25)
    pred_vectors = torch.tensor(
        [[[1.0, 0.0], [float("nan"), float("inf")]]],
        dtype=torch.float32,
    )
    target_vectors = torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)
    target_valid = torch.tensor([[True]], dtype=torch.bool)

    result = matcher(pred_vectors, target_vectors, target_valid)

    assert torch.isfinite(result["pair_cost"]).all()
    assert result["assignments"][0]["slot_to_target"] == [0, None]
    assert result["dustbin_mask"][0].tolist() == [False, True]
    assert float(result["assign_matrix"][0, 0, 0].item()) == 1.0


def test_prototype_loss_stays_finite_when_predictions_contain_nan():
    matcher = OneToOneMatcher(dustbin_cost=1.25)
    criterion = PrototypeLoss(
        matcher=matcher,
        vocab_matrix=torch.eye(2, dtype=torch.float32),
    )
    pred_vectors = torch.tensor(
        [[[1.0, 0.0], [float("nan"), 0.0]]],
        dtype=torch.float32,
    )
    pred_masks = torch.tensor(
        [[
            [[0.0, 0.0], [0.0, 0.0]],
            [[float("nan"), 0.0], [0.0, 0.0]],
        ]],
        dtype=torch.float32,
    )
    target_vectors = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]]],
        dtype=torch.float32,
    )
    target_masks = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
    target_valid = torch.tensor([[True, True]], dtype=torch.bool)
    present_mask = torch.tensor([[True, False]], dtype=torch.bool)
    confidence_logits = torch.tensor([[0.0, float("nan")]], dtype=torch.float32)
    slot_selection_logits = torch.tensor(
        [[[0.1, float("nan")], [0.0, 0.0]]],
        dtype=torch.float32,
    )

    result = criterion(
        pred_vectors=pred_vectors,
        pred_masks=pred_masks,
        target_vectors=target_vectors,
        target_masks=target_masks,
        target_valid=target_valid,
        present_mask=present_mask,
        confidence_logits=confidence_logits,
        slot_selection_logits=slot_selection_logits,
    )

    assert torch.isfinite(result["loss"])
    assert result["nonfinite_stats"]["pred_vectors"] == 1
    assert result["nonfinite_stats"]["pred_masks"] == 1
    assert result["nonfinite_stats"]["confidence_logits"] == 1
    assert result["nonfinite_stats"]["slot_selection_logits"] == 1
