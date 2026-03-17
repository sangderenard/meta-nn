from types import SimpleNamespace

import numpy as np
import torch

from pipeline.context import PipelineContext
from pipeline.nodes.classifier_node import _remap_semantic_batch_to_active_vocab
from pipeline.nodes.vocab_node import (
    activate_vocab_lora_slot,
    register_churn_requirement,
    select_active_vocab_lora_slot,
)
from wav_ml_models import (
    TinyConvClassifier,
    install_tiny_classifier_lora,
    restore_tiny_classifier_lora_snapshot,
    set_tiny_classifier_lora_state,
    tiny_classifier_lora_named_modules,
    tiny_classifier_lora_snapshot,
)


def _ok(msg: str) -> None:
    print(f"[ok] {msg}")


def test_lora_snapshot_roundtrip() -> None:
    print("\n--- test_lora_snapshot_roundtrip ---")
    model = TinyConvClassifier(num_classes=8, base_ch=16, max_ch=32, context_blocks=0, mask_decoder_channels=8)
    install_tiny_classifier_lora(model, rank=4, alpha=8.0)
    set_tiny_classifier_lora_state(model, slot_name="vocab_demo", lora_only=False)
    with torch.no_grad():
        for _, mod in tiny_classifier_lora_named_modules(model):
            slot = mod.slots["vocab_demo"]
            for param in slot.parameters():
                param.add_(torch.full_like(param, 0.125))
    snap = tiny_classifier_lora_snapshot(model, slot_name="vocab_demo")

    clone = TinyConvClassifier(num_classes=8, base_ch=16, max_ch=32, context_blocks=0, mask_decoder_channels=8)
    info = restore_tiny_classifier_lora_snapshot(clone, snap)
    assert bool(info.get("used", False)), info
    assert str(info.get("active_slot", "")) == "vocab_demo", info
    snap_clone = tiny_classifier_lora_snapshot(clone, slot_name="vocab_demo")
    for module_name, module_slots in snap["slot_weights"].items():
        clone_slots = snap_clone["slot_weights"].get(module_name, {})
        assert "vocab_demo" in clone_slots, (module_name, clone_slots.keys())
        for tensor_name, tensor in module_slots["vocab_demo"].items():
            assert torch.allclose(tensor, clone_slots["vocab_demo"][tensor_name]), (module_name, tensor_name)
    _ok("LoRA slot weights round-trip through snapshot/restore")


def test_lora_slot_inherits_base_placement() -> None:
    print("\n--- test_lora_slot_inherits_base_placement ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    model = TinyConvClassifier(num_classes=8, base_ch=16, max_ch=32, context_blocks=0, mask_decoder_channels=8)
    model = model.to(device=device, dtype=dtype)
    install_tiny_classifier_lora(model, rank=4, alpha=8.0)
    set_tiny_classifier_lora_state(model, slot_name="placement_demo", lora_only=False)

    for module_name, mod in tiny_classifier_lora_named_modules(model):
        slot = mod.slots["placement_demo"]
        for param in slot.parameters():
            assert param.device == mod.weight.device, (module_name, param.device, mod.weight.device)
            assert param.dtype == mod.weight.dtype, (module_name, param.dtype, mod.weight.dtype)
    _ok("LoRA slots inherit wrapped module device and dtype")


def test_vocab_plan_select_and_activate() -> None:
    print("\n--- test_vocab_plan_select_and_activate ---")
    ctx = PipelineContext(args=SimpleNamespace())
    ctx.supervised_class_names = ["signal", "object"]
    ctx.active_extra_terms = ["warm", "cool", "legacy a", "legacy b", "legacy c"]
    ctx.class_names = list(ctx.supervised_class_names) + list(ctx.active_extra_terms)
    ctx.vocab_lora_max_terms = 5

    plan = register_churn_requirement(
        ctx=ctx,
        required_terms=["signal", "warm", "cool", "alpha", "beta", "gamma", "delta", "epsilon"],
        term_rows=[
            ["signal", "alpha", "beta"],
            ["signal", "gamma", "delta"],
            ["signal", "epsilon", "alpha"],
        ],
        source="berkeley_refresh",
        stage_label="stage2_berkeley",
        max_terms_per_slot=5,
    )
    assert int(plan["slot_count"]) >= 2, plan
    assert str(ctx.vocab_lora_latest_plan_signature) == str(plan["signature"]), ctx.vocab_lora_latest_plan_signature

    slot = select_active_vocab_lora_slot(ctx)
    assert isinstance(slot, dict) and slot, slot
    info = activate_vocab_lora_slot(ctx, slot)
    assert str(info["signature"]) == str(slot["signature"]), info
    assert int(len(ctx.active_extra_terms)) == 5, ctx.active_extra_terms
    assert set(slot.get("terms", [])).issubset(set(ctx.active_extra_terms)), ctx.active_extra_terms
    _ok("Churn plan splits oversized vocab requirement into activatable slots")


def test_remap_semantic_batch_to_active_vocab() -> None:
    print("\n--- test_remap_semantic_batch_to_active_vocab ---")
    yb = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    mb = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    mb[:, :, 2:6, 1:5] = 1.0
    batch_meta = {
        "terms_rows": [["signal", "alpha", "mnist dataset"]],
        "mask_stacks": [],
        "mask_indices": [],
    }
    y_out, m_out, meta_out = _remap_semantic_batch_to_active_vocab(
        yb,
        mb,
        batch_meta,
        active_class_names=["signal", "object", "alpha", "mnist dataset", "legacy slot"],
        source_class_names=["signal", "cat"],
    )
    y_np = y_out.detach().cpu().numpy()
    assert tuple(y_np.shape) == (1, 5), y_np.shape
    assert float(y_np[0, 0]) > 0.5
    assert float(y_np[0, 2]) > 0.5
    assert float(y_np[0, 3]) > 0.5
    assert float(np.max(m_out.detach().cpu().numpy())) > 0.9
    remapped_idx = meta_out["mask_indices"][0].detach().cpu().numpy().astype(np.int64).tolist()
    assert 0 in remapped_idx, remapped_idx
    assert 2 in remapped_idx, remapped_idx
    assert 3 in remapped_idx, remapped_idx
    _ok("Active-vocab remap rebuilds targets and masks from terms rows")


if __name__ == "__main__":
    test_lora_snapshot_roundtrip()
    test_lora_slot_inherits_base_placement()
    test_vocab_plan_select_and_activate()
    test_remap_semantic_batch_to_active_vocab()
    print("\nALL TESTS PASSED")
