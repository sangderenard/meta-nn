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
from typing import Any, Dict, List, Optional

import torch

from pipeline.context import PipelineContext
from pipeline.graph import PipelineNode
from pipeline.nodes.base import GatedNode, make_grad_scaler


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
    image_size: int = 64            # output image side length in pixels

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
    d_steps_per_g_step: int = 1     # how many D updates per G update

    # ---- loss weights ---------------------------------------------------
    # Generator losses
    adv_weight: float = 1.0         # adversarial (non-saturating)
    feature_score_weight: float = 0.5  # classifier feature score guidance
    wave_recon_weight: float = 0.1  # bit-plane reconstruction fidelity

    # Discriminator losses
    r1_weight: float = 10.0         # R1 gradient penalty coefficient

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

    def __init__(self, cfg: GeneratorConfig) -> None:
        self.cfg = cfg
        self._built = False

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
            z_dim=self.cfg.z_dim,
            depth=self.cfg.g_depth,
            base_ch=self.cfg.g_base_ch,
            max_ch=self.cfg.g_max_ch,
            image_size=self.cfg.image_size,
        ).to(ctx.device)

        discriminator = ConditionalBitPlaneDiscriminator(
            num_classes=n_classes,
            depth=self.cfg.d_depth,
            base_ch=self.cfg.d_base_ch,
            max_ch=self.cfg.d_max_ch,
            image_size=self.cfg.image_size,
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

class GeneratorTrainNode(GatedNode):
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

    def __init__(self, cfg: GeneratorConfig) -> None:
        self.cfg = cfg

    def should_run(self, ctx: PipelineContext) -> bool:
        if not super().should_run(ctx):
            return False
        return ctx.generator is not None and ctx.discriminator is not None

    def execute(self, ctx: PipelineContext) -> None:
        from wav_ml_models import train_conditional_generator_discriminator

        # train_conditional_generator_discriminator creates its own optimizers internally
        # and returns (trained_generator, trained_discriminator, list_of_epoch_metric_dicts).
        trained_g, trained_d, metrics_list = train_conditional_generator_discriminator(
            generator=ctx.generator,
            discriminator=ctx.discriminator,
            classifier=ctx.classifier,
            payload_images=ctx.payload_bank,
            payload_conditions=ctx.payload_conditions,
            payload_masks=ctx.payload_masks if ctx.payload_masks else [],
            num_classes=len(ctx.class_names) if ctx.class_names else 1,
            image_hw=(self.cfg.image_size, self.cfg.image_size),
            device=ctx.device,
            steps_per_epoch=self.cfg.steps_per_round,
            disc_steps_per_gen_step=self.cfg.d_steps_per_g_step,
            z_dim=self.cfg.z_dim,
            lr_g=self.cfg.g_lr,
            lr_d=self.cfg.d_lr,
            w_adv=self.cfg.adv_weight,
            w_cls=self.cfg.feature_score_weight,
            w_wave=self.cfg.wave_recon_weight,
            amp=ctx.amp_enabled,
            amp_dtype=str(ctx.amp_dtype or "float16"),
            channels_last=False,
        )

        ctx.generator = trained_g
        ctx.discriminator = trained_d
        last_m = (metrics_list or [{}])[-1]
        g_loss = float(last_m.get("g_loss", float("inf")))
        d_loss = float(last_m.get("d_loss", float("inf")))
        feat_score = float(last_m.get("feature_score", 0.0))

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

        # Update vocab-keyed library snapshot
        if self.cfg.vocab_snapshot_enabled:
            _save_vocab_snapshot(ctx, self.cfg)

        _log(f"[stageG] g_loss={g_loss:.4f} d_loss={d_loss:.4f} "
             f"feat={feat_score:.4f} gate={'PASS' if ctx.gate_generator.passed else 'hold'}")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_gan_checkpoint(model, path: str, label: str) -> None:
    from wav_config_transformer_pipeline import _apply_model_init, _torch_load_cpu

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
    from wav_config_transformer_pipeline import (
        _load_gd_vocab_library_snapshot,
        _compute_gd_vocab_hash,
    )

    try:
        vocab_hash = _compute_gd_vocab_hash(ctx.class_names, ctx.active_extra_terms)
        snap_dir = str(cfg.vocab_snapshot_dir).strip() or str(ctx.output_dir / "gd_vocab_library")
        snap = _load_gd_vocab_library_snapshot(snap_dir, vocab_hash)
        if snap is not None:
            generator.load_state_dict(snap["generator"], strict=False)
            discriminator.load_state_dict(snap["discriminator"], strict=False)
            _log(f"[GAN] restored vocab snapshot for hash {vocab_hash[:8]}")
    except Exception as exc:
        _log(f"[GAN] vocab snapshot restore skipped: {exc}")


def _save_vocab_snapshot(ctx: PipelineContext, cfg: GeneratorConfig) -> None:
    from wav_config_transformer_pipeline import (
        _save_gd_vocab_library_snapshot,
        _compute_gd_vocab_hash,
    )

    try:
        vocab_hash = _compute_gd_vocab_hash(ctx.class_names, ctx.active_extra_terms)
        snap_dir = str(cfg.vocab_snapshot_dir).strip() or str(ctx.output_dir / "gd_vocab_library")
        _save_gd_vocab_library_snapshot(
            snap_dir, vocab_hash,
            generator=ctx.generator,
            discriminator=ctx.discriminator,
        )
    except Exception as exc:
        _log(f"[GAN] vocab snapshot save failed: {exc}")


def _log(msg: str) -> None:
    print(msg, flush=True)
