from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from pipeline.context import PipelineContext
from pipeline.nodes.classifier_node import _yb_from_terms
from pipeline.nodes.data_nodes import (
    LabelMaskDropoutConfig,
    _apply_label_mask_dropout,
    _expand_semantic_mask_supervision_batch,
    build_stage_loaders,
)
from pipeline.nodes.generator_node import _FlashcardDataset
from pipeline.nodes.vocab_node import (
    activate_vocab_lora_slot,
    build_stage_vocab_lora_execution_plan,
    capture_vocab_baseline_state,
    clear_flashcard_stage_state,
    register_churn_requirement,
    reset_vocab_stage_state,
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
from pipeline.vocabulary_defaults import DEFAULT_VOCABULARY


def _ok(msg: str) -> None:
    print(f"[ok] {msg}")


class _OrderedScalarDataset(Dataset):
    def __init__(self, values):
        self._values = [int(v) for v in list(values)]

    def __len__(self) -> int:
        return int(len(self._values))

    def __getitem__(self, idx: int):
        return torch.tensor(int(self._values[int(idx)]), dtype=torch.long)


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
    ctx.supervised_class_names = list(DEFAULT_VOCABULARY)
    ctx.active_extra_terms = ["extra warm a", "extra cool b", "legacy term a", "legacy term b", "legacy term c"]
    ctx.class_names = list(ctx.supervised_class_names) + list(ctx.active_extra_terms)
    ctx.vocab_lora_max_terms = 5

    plan = register_churn_requirement(
        ctx=ctx,
        required_terms=["signal", "extra warm a", "extra cool b", "dataset alpha", "dataset beta", "dataset gamma", "dataset delta", "dataset epsilon"],
        term_rows=[
            ["signal", "dataset alpha", "dataset beta"],
            ["signal", "dataset gamma", "dataset delta"],
            ["signal", "dataset epsilon", "dataset alpha"],
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


def test_yb_from_terms() -> None:
    print("\n--- test_yb_from_terms ---")
    active_term_to_idx = {"signal": 0, "object": 1, "dataset alpha": 2, "dataset beta": 3, "dataset gamma": 4}
    terms_rows = [["signal", "dataset alpha", "dataset beta"]]
    yb = _yb_from_terms(
        terms_rows=terms_rows,
        active_term_to_idx=active_term_to_idx,
        n_active_classes=5,
        device=torch.device("cpu"),
    )
    y_np = yb.detach().cpu().numpy()
    assert tuple(y_np.shape) == (1, 5), y_np.shape
    assert float(y_np[0, 0]) > 0.5   # signal
    assert float(y_np[0, 1]) < 0.5   # object — not in terms
    assert float(y_np[0, 2]) > 0.5   # dataset alpha
    assert float(y_np[0, 3]) > 0.5   # dataset beta
    assert float(y_np[0, 4]) < 0.5   # dataset gamma — not in terms
    _ok("_yb_from_terms builds multi-hot targets from terms rows")


def test_stage_local_execution_plan_ignores_hijacked_live_signature() -> None:
    print("\n--- test_stage_local_execution_plan_ignores_hijacked_live_signature ---")
    ctx = PipelineContext(args=SimpleNamespace())
    ctx.supervised_class_names = list(DEFAULT_VOCABULARY)
    ctx.active_extra_terms = ["extra person", "extra car", "extra cat", "extra sheep", "extra slot 5"]
    ctx.class_names = list(ctx.supervised_class_names) + list(ctx.active_extra_terms)
    ctx.vocab_lora_max_terms = 5
    capture_vocab_baseline_state(ctx)

    payload_plan = register_churn_requirement(
        ctx=ctx,
        required_terms=["berkeley sbd dataset", "object", "signal", "extra person", "extra car", "extra cat", "extra sheep"],
        term_rows=[
            ["berkeley sbd dataset", "object", "signal", "extra person"],
            ["berkeley sbd dataset", "object", "signal", "extra cat"],
            ["berkeley sbd dataset", "object", "signal", "extra sheep"],
        ],
        source="payload_bank",
        stage_label="payload_bank",
        max_terms_per_slot=5,
    )
    assert str(ctx.vocab_lora_latest_plan_signature) == str(payload_plan["signature"]), ctx.vocab_lora_latest_plan_signature

    flashcard_plan = register_churn_requirement(
        ctx=ctx,
        required_terms=["flash digit 0", "flash digit 1", "flash letter a", "flash letter b", "flash mnist", "flash emnist"],
        term_rows=[
            ["flash digit 0", "flash mnist"],
            ["flash letter a", "flash emnist"],
        ],
        source="flashcard_symbol_pool",
        stage_label="flashcard_requirements",
        max_terms_per_slot=5,
    )
    assert int(flashcard_plan["required_extra_term_count"]) > 0, flashcard_plan
    assert str(ctx.vocab_lora_latest_plan_signature) == str(flashcard_plan["signature"]), ctx.vocab_lora_latest_plan_signature

    execution_plan = build_stage_vocab_lora_execution_plan(
        ctx=ctx,
        term_rows=[
            ["berkeley sbd dataset", "object", "signal", "extra person"],
            ["berkeley sbd dataset", "object", "signal", "extra cat"],
            ["berkeley sbd dataset", "object", "signal", "extra sheep"],
            ["flash digit 0", "flash mnist"],
            ["flash letter a", "flash emnist"],
        ],
        source="payload_stage_g_generator",
        stage_label="stage_g_generator",
    )
    slots = list(execution_plan.get("slots") or [])
    assert int(len(slots)) >= 2, execution_plan
    all_slot_terms = set()
    for slot in slots:
        all_slot_terms.update(slot.get("terms", []))
    assert "extra sheep" in all_slot_terms, all_slot_terms
    assert "flash digit 0" in all_slot_terms, all_slot_terms
    ordered = list(execution_plan.get("ordered_row_indices") or [])
    assert sorted(ordered) == [0, 1, 2, 3, 4], ordered
    assert sum(int(slot.get("row_count", 0)) for slot in slots) == 5, slots

    _ok("stage-local execution planning ignores a hijacked live signature")


def test_stage_local_execution_plan_keeps_rows_vocab_aligned() -> None:
    print("\n--- test_stage_local_execution_plan_keeps_rows_vocab_aligned ---")
    ctx = PipelineContext(args=SimpleNamespace())
    ctx.supervised_class_names = list(DEFAULT_VOCABULARY)
    ctx.active_extra_terms = ["extra person", "extra cat", "flash digit 0", "flash letter a", "extra slot 5"]
    ctx.class_names = list(ctx.supervised_class_names) + list(ctx.active_extra_terms)
    ctx.vocab_lora_max_terms = 2
    capture_vocab_baseline_state(ctx)

    term_rows = [
        ["signal", "extra person"],
        ["signal", "extra cat"],
        ["signal", "flash digit 0"],
        ["signal", "flash letter a"],
    ]
    execution_plan = build_stage_vocab_lora_execution_plan(
        ctx=ctx,
        term_rows=term_rows,
        source="payload_stage_g_generator",
        stage_label="stage_g_generator",
    )
    slots = list(execution_plan.get("slots") or [])
    slot_groups = list(execution_plan.get("slot_groups") or [])
    assert int(len(slot_groups)) >= 2, execution_plan

    for group in slot_groups:
        slot = dict(slots[int(group["slot_index"])])
        activate_vocab_lora_slot(ctx, slot)
        for row_idx in list(group.get("row_indices") or []):
            row_terms = list(term_rows[int(row_idx)])
            missing = [
                term for term in row_terms
                if str(term).strip().lower() not in ctx.semantic_term_to_idx
            ]
            assert missing == [], (slot.get("terms"), row_terms, missing)
    reset_vocab_stage_state(ctx)
    _ok("stage-local execution plan keeps slot rows aligned with the active vocab")


def test_stage_loader_preserves_subset_order_when_unshuffled() -> None:
    print("\n--- test_stage_loader_preserves_subset_order_when_unshuffled ---")
    ds = _OrderedScalarDataset([0, 1, 2, 3, 4, 5])
    subset = Subset(ds, [4, 1, 5, 2])
    loader, _ = build_stage_loaders(
        dataset=subset,
        name="ordered_subset_test",
        batch_size=2,
        num_workers=0,
        device_type="cpu",
        shuffle_train=False,
    )
    got = []
    for batch in loader:
        got.extend(int(x) for x in batch.reshape(-1).tolist())
    assert got == [4, 1, 5, 2], got
    _ok("build_stage_loaders preserves subset order when shuffle_train=False")


def test_reset_vocab_stage_state_restores_baseline() -> None:
    print("\n--- test_reset_vocab_stage_state_restores_baseline ---")
    ctx = PipelineContext(args=SimpleNamespace())
    ctx.supervised_class_names = list(DEFAULT_VOCABULARY)
    ctx.active_extra_terms = ["extra warm a", "extra cool b", "legacy term a"]
    ctx.class_names = list(ctx.supervised_class_names) + list(ctx.active_extra_terms)
    ctx.vocab_lora_max_terms = 3
    capture_vocab_baseline_state(ctx)

    plan = register_churn_requirement(
        ctx=ctx,
        required_terms=["signal", "dataset alpha", "dataset beta", "dataset gamma", "dataset delta"],
        term_rows=[["signal", "dataset alpha"], ["signal", "dataset beta"], ["signal", "dataset gamma"], ["signal", "dataset delta"]],
        source="stage_c_lora",
        stage_label="stage_c_lora",
        max_terms_per_slot=3,
    )
    slot = dict((plan.get("slots") or [])[0])
    activate_vocab_lora_slot(ctx, slot)
    ctx.vocab_lora_latest_plan_signature = str(plan.get("signature", ""))
    ctx.vocab_lora_plan_slot_cursor = 2
    ctx.vocab_churn_activation_pending = True

    reset_vocab_stage_state(ctx)
    assert ctx.active_extra_terms == ["extra warm a", "extra cool b", "legacy term a"], ctx.active_extra_terms
    assert ctx.class_names == list(DEFAULT_VOCABULARY) + ["extra warm a", "extra cool b", "legacy term a"], ctx.class_names
    expected_term_to_idx = {str(t).strip().lower(): i for i, t in enumerate(list(DEFAULT_VOCABULARY) + ["extra warm a", "extra cool b", "legacy term a"])}
    assert ctx.semantic_term_to_idx == expected_term_to_idx, ctx.semantic_term_to_idx
    assert str(ctx.lora_active_slot) == "", ctx.lora_active_slot
    assert str(ctx.vocab_lora_active_signature) == "", ctx.vocab_lora_active_signature
    assert list(ctx.vocab_lora_active_terms) == [], ctx.vocab_lora_active_terms
    assert str(ctx.vocab_lora_latest_plan_signature) == "", ctx.vocab_lora_latest_plan_signature
    assert int(ctx.vocab_lora_plan_slot_cursor) == 0, ctx.vocab_lora_plan_slot_cursor
    assert bool(ctx.vocab_churn_activation_pending) is False, ctx.vocab_churn_activation_pending
    _ok("reset_vocab_stage_state restores the baseline vocabulary state")


def test_flashcard_dataset_emits_full_frame_masks_and_aligned_indices() -> None:
    print("\n--- test_flashcard_dataset_emits_full_frame_masks_and_aligned_indices ---")
    ctx = PipelineContext(args=SimpleNamespace())
    ctx.supervised_class_names = list(DEFAULT_VOCABULARY)
    ctx.active_extra_terms = ["flash digit 0", "flash mnist"]
    ctx.class_names = list(ctx.supervised_class_names) + list(ctx.active_extra_terms)
    n_classes = len(ctx.class_names)
    ctx.semantic_term_to_idx = {str(t).strip().lower(): i for i, t in enumerate(ctx.class_names)}

    signal_idx = ctx.semantic_term_to_idx["signal"]
    flash_digit_idx = ctx.semantic_term_to_idx["flash digit 0"]
    flash_mnist_idx = ctx.semantic_term_to_idx["flash mnist"]

    condition = np.zeros(n_classes, dtype=np.float32)
    condition[signal_idx] = 1.0
    condition[flash_digit_idx] = 1.0
    condition[flash_mnist_idx] = 1.0

    sample = {
        "image": np.ones((3, 8, 8), dtype=np.float32),
        "condition": condition,
        "mask": np.ones((1, 8, 8), dtype=np.float32),
        "terms": ["signal", "flash digit 0", "flash mnist"],
        "mask_mode": "full_frame_per_label",
    }
    ds = _FlashcardDataset(samples=[sample], image_hw=(8, 8), active_term_to_idx=ctx.semantic_term_to_idx)
    img, mask, stack, idx, terms = ds[0]
    assert tuple(img.shape) == (3, 8, 8), img.shape
    assert torch.allclose(mask, torch.ones((1, 8, 8), dtype=torch.float32)), mask
    assert tuple(stack.shape) == (3, 8, 8), stack.shape
    assert torch.allclose(stack, torch.ones((3, 8, 8), dtype=torch.float32)), stack
    assert sorted(idx.tolist()) == sorted([signal_idx, flash_digit_idx, flash_mnist_idx]), idx
    assert terms == ["signal", "flash digit 0", "flash mnist"], terms
    _ok("flashcard dataset emits full-frame masks and aligned mask indices")


def test_flashcard_label_dropout_preserves_full_frame_contract() -> None:
    print("\n--- test_flashcard_label_dropout_preserves_full_frame_contract ---")
    ctx = PipelineContext(args=SimpleNamespace())
    ctx.supervised_class_names = list(DEFAULT_VOCABULARY)
    ctx.active_extra_terms = ["flash digit 0", "flash mnist"]
    ctx.class_names = list(ctx.supervised_class_names) + list(ctx.active_extra_terms)
    n_classes = len(ctx.class_names)
    ctx.semantic_term_to_idx = {str(t).strip().lower(): i for i, t in enumerate(ctx.class_names)}

    signal_idx = ctx.semantic_term_to_idx["signal"]
    flash_digit_idx = ctx.semantic_term_to_idx["flash digit 0"]
    flash_mnist_idx = ctx.semantic_term_to_idx["flash mnist"]

    condition = np.zeros(n_classes, dtype=np.float32)
    condition[signal_idx] = 1.0
    condition[flash_digit_idx] = 1.0
    condition[flash_mnist_idx] = 1.0

    sample = {
        "image": np.ones((3, 8, 8), dtype=np.float32),
        "condition": condition,
        "mask": np.ones((1, 8, 8), dtype=np.float32),
        "terms": ["signal", "flash digit 0", "flash mnist"],
        "mask_mode": "full_frame_per_label",
    }
    ds = _FlashcardDataset(samples=[sample], image_hw=(8, 8), active_term_to_idx=ctx.semantic_term_to_idx)
    xb, mb, stack, idx, terms = ds[0]
    yb = _yb_from_terms([terms], ctx.semantic_term_to_idx, n_classes, torch.device("cpu"))
    cfg = LabelMaskDropoutConfig(drop_rate=1.0, min_keep_labels=1, max_drop_frac=1.0)
    yb_drop, stacks_drop, indices_drop = _apply_label_mask_dropout(
        yb=yb,
        stack_list=[stack],
        index_list=[idx],
        cfg=cfg,
        rng=np.random.default_rng(123),
    )
    assert int((yb_drop[0] >= 0.5).sum().item()) == 1, yb_drop
    assert tuple(stacks_drop[0].shape) == (1, 8, 8), stacks_drop[0].shape
    assert tuple(indices_drop[0].shape) == (1,), indices_drop[0].shape

    _, _, mb_drop = _expand_semantic_mask_supervision_batch(
        xb=xb.unsqueeze(0),
        yb=yb_drop,
        mb=mb.unsqueeze(0),
        batch_meta={
            "mask_stacks": stacks_drop,
            "mask_indices": indices_drop,
            "terms_rows": [terms],
        },
        mode="multihot_mix",
        context="flashcard_dropout_test",
        dropout_cfg=None,
        dropout_rng=None,
    )
    assert torch.allclose(mb_drop, torch.ones((1, 1, 8, 8), dtype=torch.float32)), mb_drop
    _ok("flashcard label dropout keeps the full-frame supervision contract intact")


def test_clear_flashcard_stage_state_resets_stale_rows() -> None:
    print("\n--- test_clear_flashcard_stage_state_resets_stale_rows ---")
    ctx = PipelineContext(args=SimpleNamespace())
    ctx.flashcard_rows = [{"image": np.zeros((3, 4, 4), dtype=np.float32)}]
    ctx.flashcard_row_terms = [["stale term"]]
    clear_flashcard_stage_state(ctx)
    assert ctx.flashcard_rows == [], ctx.flashcard_rows
    assert ctx.flashcard_row_terms == [], ctx.flashcard_row_terms
    _ok("flashcard stage state reset clears stale rows")


if __name__ == "__main__":
    test_lora_snapshot_roundtrip()
    test_lora_slot_inherits_base_placement()
    test_vocab_plan_select_and_activate()
    test_yb_from_terms()
    test_stage_local_execution_plan_ignores_hijacked_live_signature()
    test_stage_local_execution_plan_keeps_rows_vocab_aligned()
    test_stage_loader_preserves_subset_order_when_unshuffled()
    test_reset_vocab_stage_state_restores_baseline()
    test_flashcard_dataset_emits_full_frame_masks_and_aligned_indices()
    test_flashcard_label_dropout_preserves_full_frame_contract()
    test_clear_flashcard_stage_state_resets_stale_rows()
    print("\nALL TESTS PASSED")
