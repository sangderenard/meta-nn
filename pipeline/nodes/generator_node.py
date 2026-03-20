"""
Generator + Discriminator Nodes — Conditional GAN (Stage G).

This file is the authoritative description of everything idiosyncratic to the
conditional GAN pair in this pipeline.

Model architecture choices owned here
--------------------------------------
  ConditionalBitPlaneGenerator
    * z_dim  — latent noise dimensionality
    * depth  — number of upsampling blocks
    * base_ch — initial channel count
    * Conditioning mechanism: multi-hot semantic class vector projected and
      concatenated to the latent before each block

  ConditionalBitPlaneDiscriminator
    * depth  — number of downsampling blocks
    * base_ch — initial channel count
    * Patch-level discrimination (not image-level)
    * Class conditioning: same projection scheme as generator

Loss composition owned here (Stage G)
--------------------------------------
  * Adversarial loss (non-saturating generator heuristic)
  * Classifier feature score loss (generated images should score well)
  * Wave reconstruction loss (generated bit-plane matches target wave bits)
  * Discriminator: real/fake + gradient penalty (R1)
  * Vocabulary-snapshot library: save G+D weights keyed to current vocab hash
    so that vocab rotations can quickly restore a prior adapted state

Training schedule owned here
------------------------------
  * Separate LR for generator and discriminator
  * D steps per G step ratio
  * AMP support
  * Generator gate: feature score of generated images must exceed threshold

Joint mode
----------
  When the orchestrator sets ``joint_mode=True`` on this node the generator
  and discriminator are trained simultaneously with the transformer in a
  single combined backward pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from pipeline.context import PipelineContext
from pipeline.graph import PipelineNode
from pipeline.nodes.base import (
    IRLossTermSpec,
    IRStateSpec,
    IRTrainingNode,
    IRTensorPortSpec,
    make_runtime_weight_publish_callback,
    make_grad_scaler,
)
import hashlib
import json
import time
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Flashcard dataset helper
# ---------------------------------------------------------------------------

class _FlashcardDataset(Dataset):
    """Wraps pre-built CHW float32 flashcard images with per-slot rebuilt conditions.

    Returns ``(img [3,H,W] float32, cond [C] float32, mask [1,H,W] float32)``
    where mask is all-zeros (flashcard rows carry no spatial annotation).
    """

    def __init__(
        self,
        images: list,
        conditions: list,
        num_classes: int,
        image_hw: tuple,
    ) -> None:
        self._h = int(image_hw[0])
        self._w = int(image_hw[1])
        n = min(len(images), len(conditions))
        c = max(1, int(num_classes))
        cond_arr = np.zeros((n, c), dtype=np.float32)
        for i in range(n):
            vec = np.asarray(conditions[i], dtype=np.float32).reshape(-1)
            end = min(int(vec.size), c)
            cond_arr[i, :end] = vec[:end]
        self._conds = torch.from_numpy(cond_arr)
        self._imgs = [np.asarray(img, dtype=np.float32) for img in images[:n]]
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int):
        img = torch.from_numpy(self._imgs[int(idx)])
        # Accept both CHW (3,H,W) and HWC (H,W,3)
        if img.ndim == 3 and img.shape[-1] == 3:
            img = img.permute(2, 0, 1).contiguous()
        if tuple(img.shape[-2:]) != (self._h, self._w):
            img = F.interpolate(
                img.unsqueeze(0), size=(self._h, self._w), mode="nearest"
            ).squeeze(0)
        zero_mask = torch.zeros((1, self._h, self._w), dtype=torch.float32)
        return (
            img.clamp(0.0, 1.0).contiguous(),
            self._conds[int(idx)],
            zero_mask,
        )


# ---------------------------------------------------------------------------
# Node config
# ---------------------------------------------------------------------------

@dataclass
class GeneratorConfig:
    """Every hyperparameter specific to the conditional GAN pair."""

    # ---- generator architecture -----------------------------------------
    z_dim: int = 128
    g_depth: int = 4
    g_base_ch: int = 64
    g_max_ch: int = 512
    image_size: int = 128           # output image side length in pixels
    g_mask_decoder_channels: int = 64  # >0 enables generator mask head (required for conditional training)

    # ---- discriminator architecture -------------------------------------
    d_depth: int = 4
    d_base_ch: int = 64
    d_max_ch: int = 512
    d_patch_size: int = 8           # discriminator receptive field patch size

    # ---- optimiser / LR -------------------------------------------------
    g_lr: float = 1e-4
    d_lr: float = 4e-4              # discriminator trains faster
    g_weight_decay: float = 0.0
    d_weight_decay: float = 0.0
    g_beta1: float = 0.0            # Adam β₁ (0 recommended for GANs)
    g_beta2: float = 0.99
    d_beta1: float = 0.0
    d_beta2: float = 0.99

    # ---- AMP ------------------------------------------------------------
    amp: bool = False
    amp_dtype: str = "fp16"

    # ---- training schedule ----------------------------------------------
    steps_per_round: int = 64
    batch_size: int = 64
    d_steps_per_g_step: int = 1     # how many D updates per G update

    # ---- loss weights ---------------------------------------------------
    # Generator losses
    adv_weight: float = 1.0         # adversarial (non-saturating)
    feature_score_weight: float = 0.5  # classifier feature score guidance
    wave_recon_weight: float = 0.1  # bit-plane reconstruction fidelity
    mask_weight: float = 0.0        # mask BCE supervision
    outside_mask_weight: float = 0.0  # L1 reconstruction outside mask
    disc_mask_weight: float = 0.0   # discriminator mask head adversarial

    # Anti-collapse losses
    # diversity_weight > 0 adds a penalty when batch-level per-pixel std
    # falls below diversity_target_std — directly fights mode collapse.
    diversity_weight: float = 0.0
    diversity_target_std: float = 0.15

    # R1 gradient penalty weight on discriminator (0 = disabled)
    r1_weight: float = 10.0

    # Discriminator instance noise: add Gaussian noise (std) to D inputs
    # so D cannot dominate G early; set to e.g. 0.05–0.10 to help escape.
    d_instance_noise_std: float = 0.0

    # ---- vocab-snapshot library -----------------------------------------
    vocab_snapshot_enabled: bool = True
    vocab_snapshot_dir: str = ""    # empty → {output_dir}/gd_vocab_library

    # ---- gate -----------------------------------------------------------
    gate_feature_score_target: float = 0.50
    gate_required_consecutive: int = 2

    # ---- compile --------------------------------------------------------
    compile_model: bool = False

    # ---- joint mode (G+R simultaneous) ----------------------------------
    joint_mode: bool = False

    # ---- Checkpoint init -----------------------------------------------
    generator_init_ckpt: str = ""
    discriminator_init_ckpt: str = ""


# ---------------------------------------------------------------------------
# Build node (runs once)
# ---------------------------------------------------------------------------

class BuildGANNode(PipelineNode):
    """Instantiate ConditionalBitPlaneGenerator + Discriminator and their optimizers.

    Skipped when orchestration mode does not include GAN stages
    (checked via ctx.args.orchestration_mode).
    """

    node_id = "build_gan"
    description = "Instantiate ConditionalBitPlaneGenerator + Discriminator"
    runtime_object_type = "builder"
    runtime_faculty = "build"
    gpu_models = ["generator", "discriminator"]

    def __init__(self, cfg: GeneratorConfig) -> None:
        self.cfg = cfg
        self._built = False

    @property
    def runtime_execution_policy(self) -> tuple:
        return ("once", {})

    def should_run(self, ctx: PipelineContext) -> bool:
        if self._built:
            return False
        mode = str(ctx.orchestration_mode or getattr(ctx.args, "orchestration_mode", "staged_cgrw")).lower()
        return "g" in mode  # any mode containing 'g' uses the GAN

    def execute(self, ctx: PipelineContext) -> None:
        from wav_ml_models import (
            ConditionalBitPlaneGenerator,
            ConditionalBitPlaneDiscriminator,
            maybe_compile_module,
        )

        n_classes = len(ctx.class_names) if ctx.class_names else 1

        generator = ConditionalBitPlaneGenerator(
            num_classes=n_classes,
            image_hw=(self.cfg.image_size, self.cfg.image_size),
            z_dim=self.cfg.z_dim,
            depth=self.cfg.g_depth,
            base_ch=self.cfg.g_base_ch,
            min_ch=max(8, int(getattr(self.cfg, "g_min_ch", 12) or 12)),
            mask_decoder_channels=int(self.cfg.g_mask_decoder_channels),
        ).to(ctx.device)

        discriminator = ConditionalBitPlaneDiscriminator(
            num_classes=n_classes,
            depth=self.cfg.d_depth,
            base_ch=self.cfg.d_base_ch,
            max_ch=self.cfg.d_max_ch,
        ).to(ctx.device)

        if self.cfg.compile_model:
            generator = maybe_compile_module(generator, enabled=True)
            discriminator = maybe_compile_module(discriminator, enabled=True)

        g_optimizer = torch.optim.Adam(
            generator.parameters(),
            lr=self.cfg.g_lr,
            betas=(self.cfg.g_beta1, self.cfg.g_beta2),
            weight_decay=self.cfg.g_weight_decay,
        )
        d_optimizer = torch.optim.Adam(
            discriminator.parameters(),
            lr=self.cfg.d_lr,
            betas=(self.cfg.d_beta1, self.cfg.d_beta2),
            weight_decay=self.cfg.d_weight_decay,
        )

        ctx.generator = generator
        ctx.discriminator = discriminator
        ctx.generator_optimizer = g_optimizer
        ctx.discriminator_optimizer = d_optimizer

        if self.cfg.amp:
            ctx.generator_grad_scaler = make_grad_scaler(enabled=True)
            ctx.discriminator_grad_scaler = make_grad_scaler(enabled=True)

        resume_ckpt = ctx.resume_pipeline_ckpt if isinstance(ctx.resume_pipeline_ckpt, dict) else None
        if resume_ckpt is not None:
            if "generator_state" in resume_ckpt:
                try:
                    generator.load_state_dict(resume_ckpt["generator_state"], strict=False)
                    _log("[GAN] resumed generator model state from pipeline checkpoint")
                except Exception as exc:
                    _log(f"[GAN] WARNING: could not resume generator model state: {exc}")
            if "discriminator_state" in resume_ckpt:
                try:
                    discriminator.load_state_dict(resume_ckpt["discriminator_state"], strict=False)
                    _log("[GAN] resumed discriminator model state from pipeline checkpoint")
                except Exception as exc:
                    _log(f"[GAN] WARNING: could not resume discriminator model state: {exc}")
            if "generator_optimizer_state" in resume_ckpt:
                try:
                    g_optimizer.load_state_dict(resume_ckpt["generator_optimizer_state"])
                    _log("[GAN] resumed generator optimizer state from pipeline checkpoint")
                except Exception as exc:
                    _log(f"[GAN] WARNING: could not resume generator optimizer state: {exc}")
            if "discriminator_optimizer_state" in resume_ckpt:
                try:
                    d_optimizer.load_state_dict(resume_ckpt["discriminator_optimizer_state"])
                    _log("[GAN] resumed discriminator optimizer state from pipeline checkpoint")
                except Exception as exc:
                    _log(f"[GAN] WARNING: could not resume discriminator optimizer state: {exc}")
            g_scaler = ctx.generator_grad_scaler
            if g_scaler is not None and "generator_grad_scaler_state" in resume_ckpt:
                try:
                    g_scaler.load_state_dict(resume_ckpt["generator_grad_scaler_state"])
                    _log("[GAN] resumed generator grad-scaler state from pipeline checkpoint")
                except Exception as exc:
                    _log(f"[GAN] WARNING: could not resume generator grad-scaler state: {exc}")
            d_scaler = ctx.discriminator_grad_scaler
            if d_scaler is not None and "discriminator_grad_scaler_state" in resume_ckpt:
                try:
                    d_scaler.load_state_dict(resume_ckpt["discriminator_grad_scaler_state"])
                    _log("[GAN] resumed discriminator grad-scaler state from pipeline checkpoint")
                except Exception as exc:
                    _log(f"[GAN] WARNING: could not resume discriminator grad-scaler state: {exc}")

        # Checkpoint load
        g_ckpt = self.cfg.generator_init_ckpt or getattr(ctx.args, "generator_init", "") or ""
        d_ckpt = self.cfg.discriminator_init_ckpt or getattr(ctx.args, "discriminator_init", "") or ""
        if str(g_ckpt).strip():
            _load_gan_checkpoint(generator, str(g_ckpt), label="generator")
        if str(d_ckpt).strip():
            _load_gan_checkpoint(discriminator, str(d_ckpt), label="discriminator")

        # Try to restore vocab-snapshot if a prior run saved one
        if self.cfg.vocab_snapshot_enabled:
            _try_restore_vocab_snapshot(ctx, generator, discriminator, self.cfg)

        _log(f"[GAN] built: z_dim={self.cfg.z_dim} g_depth={self.cfg.g_depth} "
             f"d_depth={self.cfg.d_depth} n_classes={n_classes}")
        self._built = True


# ---------------------------------------------------------------------------
# Stage G — GAN training node
# ---------------------------------------------------------------------------

class GeneratorTrainNode(IRTrainingNode):
    """Stage G: Train conditional GAN pair.

    Requires all base gates (pre-gestation, gestation, Berkeley).
    The generator learns to produce images that:
      1. Fool the discriminator (adversarial loss)
      2. Score well on the classifier's feature metric (semantic guidance)
      3. Reconstruct the target bit-plane pattern (wave reconstruction)

    The discriminator uses R1 gradient penalty for training stability.
    After each round the vocab snapshot library is updated so that rotating
    the vocabulary can quickly restore a prior adapted G+D state.
    """

    node_id = "stage_g_generator"
    description = "Stage G: Conditional GAN training (generator + discriminator)"
    required_gates = ["gate_pregestation", "gate_gestation", "gate_berkeley"]
    gpu_models = ["generator", "discriminator", "classifier"]
    model_attr = "generator"
    extra_model_attrs = ["discriminator", "classifier"]
    optimizer_attrs = ["generator_optimizer", "discriminator_optimizer"]

    def __init__(self, cfg: GeneratorConfig) -> None:
        self.cfg = cfg

    def ir_input_ports(self) -> List[IRTensorPortSpec]:
        return [
            IRTensorPortSpec(
                "payload_images",
                "Payload images",
                io="input",
                dtype="float32",
                shape="B x 3 x H x W",
                semantic="real_image_batch",
                detail="real semantic payload images used for adversarial and reconstruction supervision",
            ),
            IRTensorPortSpec(
                "payload_conditions",
                "Payload conditions",
                io="input",
                dtype="float32",
                shape="B x C",
                semantic="condition_vectors",
                detail="multihot semantic conditioning vectors",
            ),
            IRTensorPortSpec(
                "payload_masks",
                "Payload masks",
                io="input",
                dtype="float32",
                shape="B x 1 x H x W",
                semantic="spatial_mask_batch",
                detail="optional spatial targets aligned with the payload bank",
            ),
        ]

    def ir_output_ports(self) -> List[IRTensorPortSpec]:
        return [
            IRTensorPortSpec(
                "generated_images",
                "Generated images",
                io="output",
                dtype="float32",
                shape="B x 3 x H x W",
                semantic="generated_image_batch",
                detail="conditioned samples emitted by the GAN generator",
            ),
            IRTensorPortSpec(
                "discriminator_patch_logits",
                "Discriminator patch logits",
                io="output",
                dtype="float32",
                shape="B x 1 x h x w",
                semantic="patch_discriminator_logits",
                detail="critic scores over real and generated image patches",
            ),
        ]

    def ir_loss_terms(self) -> List[IRLossTermSpec]:
        return [
            IRLossTermSpec(
                "stage_g_generator_loss",
                "Stage G generator objective",
                kind="adversarial+classifier_guidance+wave_reconstruction",
                optimizer_targets=["generator_optimizer"],
                source_ports=["payload_images", "payload_conditions", "generated_images"],
                detail="Generator objective combining adversarial pressure, classifier guidance, and wave reconstruction fidelity",
            ),
            IRLossTermSpec(
                "stage_g_discriminator_loss",
                "Stage G discriminator objective",
                kind="real_fake+r1_penalty",
                optimizer_targets=["discriminator_optimizer"],
                source_ports=["payload_images", "generated_images", "discriminator_patch_logits"],
                detail="Discriminator objective over real/fake patches with R1 stabilization",
            ),
        ]

    def ir_state_inputs(self) -> List[IRStateSpec]:
        return [
            IRStateSpec(
                "vocab_snapshot_library",
                "Vocab snapshot library",
                role="checkpoint_bank",
                detail="keyed G+D snapshot bank for restoring vocab-specific GAN states",
            ),
        ]

    def ir_contract_notes(self) -> str:
        return (
            "PyTorch Stage-G trainer with a dual-model contract. The current helper "
            "creates its own inner optimizers per call, but the IR still declares "
            "generator/discriminator ownership, tensor surfaces, and optimization targets."
        )

    def declare_training_mechanics(self) -> Dict[str, Any]:
        return {
            "module_family": "gan",
            "module_label": "Stage G Generator",
            "summary": "Generator and discriminator co-train against payload supervision, classifier guidance, and adversarial pressure.",
            "inputs": [
                {"id": "semantic_payload_bank", "label": "Semantic payload bank", "kind": "dataset", "detail": "real target images for adversarial refresh"},
                {"id": "semantic_condition_bank", "label": "Condition bank", "kind": "state", "detail": "multihot conditioning vectors"},
                {"id": "semantic_payload_masks", "label": "Payload masks", "kind": "dataset", "detail": "optional spatial supervision masks"},
                {"id": "gan_generator_model", "label": "GAN generator", "kind": "model", "detail": "conditioned image synthesizer"},
                {"id": "gan_discriminator_model", "label": "GAN discriminator", "kind": "model", "detail": "real-vs-fake critic with stability penalty"},
                {"id": "classifier_model", "label": "Classifier teacher", "kind": "model", "detail": "semantic guidance signal for generated samples"},
            ],
            "losses": [
                {"id": "loss_stage_g_generator", "label": "adv + classifier guidance + wave recon", "kind": "loss", "detail": "generator composite objective"},
                {"id": "loss_stage_g_discriminator", "label": "real/fake discrimination + R1", "kind": "loss", "detail": "discriminator objective"},
            ],
            "outputs": [
                {"id": "gan_generator_model", "label": "GAN generator", "kind": "model", "detail": "updated generator weights"},
                {"id": "gan_discriminator_model", "label": "GAN discriminator", "kind": "model", "detail": "updated discriminator weights"},
            ],
            "flows": [
                {"source": "semantic_payload_bank", "target": "self", "label": "real image targets", "kind": "consume"},
                {"source": "semantic_condition_bank", "target": "self", "label": "conditioning vectors", "kind": "condition"},
                {"source": "semantic_payload_masks", "target": "self", "label": "spatial targets", "kind": "consume"},
                {"source": "gan_generator_model", "target": "self", "label": "generator weights", "kind": "consume"},
                {"source": "gan_discriminator_model", "target": "self", "label": "critic weights", "kind": "consume"},
                {"source": "classifier_model", "target": "self", "label": "semantic teacher", "kind": "condition"},
                {"source": "self", "target": "loss_stage_g_generator", "label": "fake samples", "kind": "predict"},
                {"source": "semantic_payload_bank", "target": "loss_stage_g_generator", "label": "real targets", "kind": "supervise"},
                {"source": "semantic_condition_bank", "target": "loss_stage_g_generator", "label": "conditioning targets", "kind": "supervise"},
                {"source": "classifier_model", "target": "loss_stage_g_generator", "label": "feature guidance", "kind": "supervise"},
                {"source": "self", "target": "loss_stage_g_discriminator", "label": "real vs fake logits", "kind": "predict"},
                {"source": "semantic_payload_bank", "target": "loss_stage_g_discriminator", "label": "real samples", "kind": "supervise"},
                {"source": "loss_stage_g_generator", "target": "gan_generator_model", "label": "optimizer step", "kind": "optimize"},
                {"source": "loss_stage_g_discriminator", "target": "gan_discriminator_model", "label": "optimizer step", "kind": "optimize"},
            ],
        }

    def should_run(self, ctx: PipelineContext) -> bool:
        if not super().should_run(ctx):
            return False
        return ctx.generator is not None and ctx.discriminator is not None

    def execute(self, ctx: PipelineContext) -> None:
        from wav_ml_models import train_conditional_generator_discriminator
        from pipeline.semantic_wheel_cache import SemanticWheelPayloadDataset
        from pipeline.preview import make_generator_step_preview_callback
        from pipeline.nodes.base import make_training_progress_callback
        from pipeline.nodes.vocab_node import activate_vocab_lora_slot, get_all_planned_lora_slots

        if not ctx.payload_masks:
            _log("[stageG] WARNING: payload_masks is empty — generator cannot train without spatial masks")
            return

        generator_step_callback = make_runtime_weight_publish_callback(
            ctx,
            model_name="generator",
            model=ctx.generator,
            node_id=self.node_id,
        )
        discriminator_step_callback = make_runtime_weight_publish_callback(
            ctx,
            model_name="discriminator",
            model=ctx.discriminator,
            node_id=self.node_id,
        )
        _step_preview_cb = make_generator_step_preview_callback(ctx, self.node_id)
        _progress_cb = make_training_progress_callback(
            ctx, self.node_id, "stage_g_generator",
            publish_loss=(_step_preview_cb is None),
        )

        planned_slots = get_all_planned_lora_slots(ctx)

        steps_per_slot = max(1, self.cfg.steps_per_round // max(1, len(planned_slots)))

        image_hw = (self.cfg.image_size, self.cfg.image_size)
        _payload_bank = ctx.payload_bank
        _payload_bank_obj = getattr(_payload_bank, "bank", None)  # SemanticWheelPayloadBank

        all_metrics: List[Dict[str, Any]] = []
        slots_trained = 0

        for slot_def in planned_slots:
            slot_signature = str(slot_def.get("signature", "")).strip()
            slot_name = str(slot_def.get("slot_name", f"vocab_{slot_signature}"))
            slot_terms = list(slot_def.get("terms") or [])
            if not slot_signature or not slot_terms:
                continue

            # Activate this slot's vocabulary and install its LoRA on the classifier.
            activate_vocab_lora_slot(ctx, slot_def)

            # class_names and n_classes are read AFTER activation so they reflect
            # the slot's vocabulary (slot terms are added to ctx.class_names here).
            current_class_names = list(ctx.class_names or [])
            current_n_classes = max(1, len(current_class_names))

            _log(
                f"[stageG] slot {slot_name} ({len(slot_terms)} terms): "
                f"{len(ctx.payload_bank or [])} rows, {steps_per_slot} steps"
            )

            if _payload_bank_obj is not None:
                from torch.utils.data import ConcatDataset, DataLoader, RandomSampler, Subset
                from pipeline.nodes.data_nodes import rebuild_conditions_from_terms
                _slot_conditions = rebuild_conditions_from_terms(
                    payload_terms=ctx.payload_terms or [],
                    class_names=current_class_names,
                ) if ctx.payload_terms else ctx.payload_conditions
                _ds = SemanticWheelPayloadDataset(
                    bank=_payload_bank_obj,
                    conditions=_slot_conditions,
                    num_classes=current_n_classes,
                    image_hw=image_hw,
                )
                # ---- Slot-affinity filtering (CNC tool-changer) ----
                # Select payload rows whose extra terms match the active slot.
                # Rows with no extra terms (supervised-only) are included in
                # every slot so the model keeps seeing diverse imagery.
                _slot_extra_keys = {str(t).strip().lower() for t in slot_terms}
                _supervised_keys = {str(n).strip().lower() for n in list(getattr(ctx, "supervised_class_names", []))}
                _all_pt = ctx.payload_terms or []
                _affine_idx: list = []
                _neutral_idx: list = []
                for _ri, _row_t in enumerate(_all_pt):
                    if _ri >= len(_ds):
                        break
                    _row_extra = {str(t).strip().lower() for t in _row_t} - _supervised_keys
                    if not _row_extra:
                        _neutral_idx.append(_ri)
                    elif _row_extra & _slot_extra_keys:
                        _affine_idx.append(_ri)
                    # rows with extra terms NOT in this slot → skip for this slot
                _slot_idx = _affine_idx + _neutral_idx
                if _slot_idx and len(_slot_idx) < len(_ds):
                    _ds = Subset(_ds, _slot_idx)
                    _log(
                        f"[stageG] slot affinity: {len(_affine_idx)} matching + "
                        f"{len(_neutral_idx)} neutral of {len(_all_pt)} rows"
                    )
                # Merge flashcard rows only when this slot covers flashcard terms.
                _fc_terms = list(getattr(ctx, "flashcard_row_terms", []) or [])
                _fc_rows  = list(getattr(ctx, "flashcard_rows", []) or [])
                if _fc_terms and _fc_rows:
                    _fc_has_affinity = any(
                        any(str(t).strip().lower() in _slot_extra_keys for t in row_t)
                        for row_t in _fc_terms
                    )
                    if _fc_has_affinity:
                        _fc_conds = rebuild_conditions_from_terms(
                            payload_terms=_fc_terms,
                            class_names=current_class_names,
                        )
                        _fc_imgs = [img for img, _cond in _fc_rows]
                        _fc_ds = _FlashcardDataset(
                            images=_fc_imgs,
                            conditions=_fc_conds,
                            num_classes=current_n_classes,
                            image_hw=image_hw,
                        )
                        _ds = ConcatDataset([_ds, _fc_ds])
                        _log(
                            f"[stageG] merged {len(_fc_ds)} flashcard rows into slot "
                            f"{slot_name} dataset (total {len(_ds)} rows)"
                        )
                _slot_loader = DataLoader(
                    _ds,
                    batch_size=self.cfg.batch_size,
                    sampler=RandomSampler(_ds, replacement=True),
                    num_workers=1,
                    pin_memory=True,
                    prefetch_factor=2,
                    persistent_workers=True,
                )
            else:
                _slot_loader = None

            trained_g, trained_d, metrics_list = train_conditional_generator_discriminator(
                generator=ctx.generator,
                discriminator=ctx.discriminator,
                classifier=ctx.classifier,
                payload_images=ctx.payload_bank,
                payload_conditions=ctx.payload_conditions,
                payload_masks=ctx.payload_masks,
                num_classes=current_n_classes,
                image_hw=image_hw,
                payload_loader=_slot_loader,
                device=ctx.device,
                batch_size=self.cfg.batch_size,
                steps_per_epoch=steps_per_slot,
                disc_steps_per_gen_step=self.cfg.d_steps_per_g_step,
                z_dim=self.cfg.z_dim,
                lr_g=self.cfg.g_lr,
                lr_d=self.cfg.d_lr,
                g_opt=ctx.generator_optimizer,
                d_opt=ctx.discriminator_optimizer,
                w_adv=self.cfg.adv_weight,
                w_cls=self.cfg.feature_score_weight,
                w_wave=self.cfg.wave_recon_weight,
                w_mask=self.cfg.mask_weight,
                w_outside_mask=self.cfg.outside_mask_weight,
                w_disc_mask=self.cfg.disc_mask_weight,
                w_diversity=self.cfg.diversity_weight,
                diversity_target_std=self.cfg.diversity_target_std,
                d_instance_noise_std=self.cfg.d_instance_noise_std,
                amp=ctx.amp_enabled,
                amp_dtype=str(ctx.amp_dtype or "float16"),
                channels_last=False,
                log_every_steps=50,
                step_preview_callback=_step_preview_cb,
                progress_callback=_progress_cb,
                stop_requested=ctx.stop_requested,
                pause_requested=ctx.paused,
                ipc_pump=getattr(ctx.viewer_proxy, "pump", None),
                generator_step_callback=generator_step_callback,
                discriminator_step_callback=discriminator_step_callback,
            )

            ctx.generator = trained_g
            ctx.discriminator = trained_d
            all_metrics.extend(metrics_list or [])
            slots_trained += 1

        if slots_trained == 0:
            _log("[stageG] WARNING: no slots trained — all payload rows skipped")
            return

        last_m = (all_metrics or [{}])[-1]
        g_loss = float(last_m.get("g_loss", float("inf")))
        d_loss = float(last_m.get("d_loss", float("inf")))
        feat_score = float(last_m.get("feature_score", last_m.get("g_target_prob", 0.0)))

        ctx.gate_generator.required_consecutive = self.cfg.gate_required_consecutive
        ctx.gate_generator.record(
            round_id=ctx.round_id,
            metric=feat_score,
            threshold=self.cfg.gate_feature_score_target,
            above=True,
        )
        ctx.log_metric("stageG", "g_loss", g_loss)
        ctx.log_metric("stageG", "d_loss", d_loss)
        ctx.log_metric("stageG", "feature_score", feat_score)

        if self.cfg.vocab_snapshot_enabled:
            _save_vocab_snapshot(ctx, self.cfg)

        _log(
            f"[stageG] {slots_trained} slot(s) — "
            f"g_loss={g_loss:.4f} d_loss={d_loss:.4f} "
            f"feat={feat_score:.4f} gate={'PASS' if ctx.gate_generator.passed else 'hold'}"
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_gan_checkpoint(model, path: str, label: str) -> None:
    from pipeline.nodes.base import _apply_model_init, _torch_load_cpu

    try:
        ckpt = _torch_load_cpu(path)
        _apply_model_init(model, ckpt)
        _log(f"[{label}] loaded checkpoint from {path}")
    except Exception as exc:
        _log(f"[{label}] WARNING: could not load checkpoint {path!r}: {exc}")


def _try_restore_vocab_snapshot(
    ctx: PipelineContext,
    generator,
    discriminator,
    cfg: GeneratorConfig,
) -> None:

    try:
        vocab_hash, _profile = _compute_gd_vocab_hash(
            supervised_class_names=ctx.class_names,
            fixed_extra_terms=ctx.active_extra_terms,
            condition_num_classes=max(1, int(len(ctx.class_names))),
            args=ctx.args,
            active_extra_terms=ctx.active_extra_terms,
        )
        snap_dir = str(cfg.vocab_snapshot_dir).strip() or str(ctx.output_dir / "gd_vocab_library")
        snap = _load_gd_vocab_library_snapshot(snap_dir, vocab_hash, generator, discriminator)
        if bool(snap.get("loaded", False)):
            _log(f"[GAN] restored vocab snapshot for hash {vocab_hash[:8]}")
    except Exception as exc:
        _log(f"[GAN] vocab snapshot restore skipped: {exc}")


def _save_vocab_snapshot(ctx: PipelineContext, cfg: GeneratorConfig) -> None:

    try:
        vocab_hash, profile = _compute_gd_vocab_hash(
            supervised_class_names=ctx.class_names,
            fixed_extra_terms=ctx.active_extra_terms,
            condition_num_classes=max(1, int(len(ctx.class_names))),
            args=ctx.args,
            active_extra_terms=ctx.active_extra_terms,
        )
        snap_dir = str(cfg.vocab_snapshot_dir).strip() or str(ctx.output_dir / "gd_vocab_library")
        _save_gd_vocab_library_snapshot(
            snap_dir,
            vocab_hash,
            condition_num_classes=max(1, int(len(ctx.class_names))),
            generator=ctx.generator,
            discriminator=ctx.discriminator,
            meta=profile,
        )
    except Exception as exc:
        _log(f"[GAN] vocab snapshot save failed: {exc}")


def _log(msg: str) -> None:
    print(msg, flush=True)


# =========================================================================
# Functions extracted from wav_config_transformer_pipeline.py
# =========================================================================


def _compute_gd_vocab_hash(
    supervised_class_names: Sequence[str],
    fixed_extra_terms: Sequence[str],
    condition_num_classes: int,
    args: Any,
    active_extra_terms: Optional[Sequence[str]] = None,
) -> Tuple[str, Dict[str, Any]]:
    hash_basis: Dict[str, Any] = {
        "supervised_class_names": [str(x) for x in supervised_class_names],
        "fixed_extra_terms": [str(x) for x in fixed_extra_terms],
        "condition_num_classes": int(condition_num_classes),
        "label_embedding_backend": str(getattr(args, "label_embedding_backend", "")),
        "label_embedding_model": str(getattr(args, "label_embedding_model", "")),
        "label_embedding_dim": int(getattr(args, "label_embedding_dim", 0)),
        "semantic_label_mode": "presence_v2",
        "generator": {
            "z_dim": int(getattr(args, "generator_z_dim", 0)),
            "base_ch": int(getattr(args, "generator_base_ch", 0)),
            "depth": int(getattr(args, "generator_depth", 0)),
            "min_ch": int(getattr(args, "generator_min_ch", 0)),
        },
        "discriminator": {
            "base_ch": int(getattr(args, "discriminator_base_ch", 0)),
            "depth": int(getattr(args, "discriminator_depth", 0)),
            "max_ch": int(getattr(args, "discriminator_max_ch", 0)),
        },
    }
    profile: Dict[str, Any] = dict(hash_basis)
    profile["active_extra_terms"] = [str(x) for x in (active_extra_terms or [])]
    blob = json.dumps(hash_basis, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(blob).hexdigest()[:24]
    return str(digest), profile


def _save_gd_vocab_library_snapshot(
    library_dir: Path,
    vocab_hash: str,
    condition_num_classes: int,
    generator: Optional[nn.Module],
    discriminator: Optional[nn.Module],
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    if generator is None or discriminator is None:
        return {"saved": False, "reason": "missing_generator_or_discriminator"}
    h = str(vocab_hash).strip()
    if not h:
        return {"saved": False, "reason": "empty_vocab_hash"}
    c = max(1, int(condition_num_classes))
    subdir = Path(library_dir) / f"{h}_c{int(c)}"
    subdir.mkdir(parents=True, exist_ok=True)
    gen_path = subdir / "generator.pt"
    disc_path = subdir / "discriminator.pt"
    meta_path = subdir / "meta.json"
    torch.save({"state_dict": generator.state_dict()}, gen_path)
    torch.save({"state_dict": discriminator.state_dict()}, disc_path)
    blob = dict(meta) if isinstance(meta, dict) else {}
    blob["vocab_hash"] = str(h)
    blob["condition_num_classes"] = int(c)
    blob["timestamp"] = float(time.time())
    meta_path.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    return {"saved": True, "dir": str(subdir), "generator": str(gen_path), "discriminator": str(disc_path)}


def _load_gd_vocab_library_snapshot(
    library_dir: Path,
    vocab_hash: str,
    generator: Optional[nn.Module],
    discriminator: Optional[nn.Module],
) -> Dict[str, Any]:
    if generator is None or discriminator is None:
        return {"loaded": False, "reason": "missing_generator_or_discriminator"}
    h = str(vocab_hash).strip()
    if not h:
        return {"loaded": False, "reason": "empty_vocab_hash"}
    root = Path(library_dir)
    if not root.exists():
        return {"loaded": False, "reason": "library_missing"}
    candidates = sorted(root.glob(f"{h}_c*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if len(candidates) <= 0:
        return {"loaded": False, "reason": "snapshot_not_found", "hash": str(h)}
    pick = candidates[0]
    gen_path = pick / "generator.pt"
    disc_path = pick / "discriminator.pt"
    if not gen_path.exists() or not disc_path.exists():
        return {"loaded": False, "reason": "snapshot_incomplete", "dir": str(pick)}
    gen_blob = _torch_load_cpu(str(gen_path))
    disc_blob = _torch_load_cpu(str(disc_path))
    gen_state = _extract_state_dict(gen_blob)
    disc_state = _extract_state_dict(disc_blob)
    gen_info = _apply_state_dict(generator, gen_state, source_name=f"gd_vocab_library:{pick.name}:generator")
    disc_info = _apply_state_dict(discriminator, disc_state, source_name=f"gd_vocab_library:{pick.name}:discriminator")
    return {
        "loaded": True,
        "dir": str(pick),
        "generator": gen_info,
        "discriminator": disc_info,
    }
