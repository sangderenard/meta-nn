"""
Data Nodes — dataset and loader construction for every pipeline stage.

Each node owns the idiosyncrasies of its data source:

  WavePoolNode          — discover WAV files; fall back to synthetic latent pool
  PregestationDataNode  — Stage 0: synthetic geometric direction+colour images
  GestationDataNode     — Stage 1: bootstrap primitive symbol images
  BerkeleyDataNode      — Stage 2: Berkeley SBD + external payload images
  BerkeleyPayloadNode   — build/load the on-disk payload image bank

Data ownership
--------------
  WavePoolNode        → ctx.wav_records, ctx.float_streams, ctx.stream_paths
  PregestationDataNode→ ctx.pregestation_loader, ctx.pregestation_dataset
  GestationDataNode   → ctx.gestation_loader, ctx.gestation_dataset
  BerkeleyDataNode    → ctx.berkeley_refresh_loader, ctx.berkeley_gate_val_loader
  BerkeleyPayloadNode → ctx.payload_bank, ctx.payload_masks, ctx.payload_conditions, ctx.payload_bank_ready
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.context import PipelineContext
from pipeline.graph import PipelineNode
from pipeline.nodes.base import GatedNode, OneTimeNode
import math
import torch


# ---------------------------------------------------------------------------
# Wave pool node
# ---------------------------------------------------------------------------

@dataclass
class WavePoolConfig:
    """Idiosyncrasies of WAV file discovery and the synthetic latent pool fallback."""

    # Glob pattern or directory for real WAV files
    wav_dir: str = ""

    # Latent pool — used when no real WAVs are found
    latent_pool_dir: str = ""
    latent_pool_size: int = 64          # synthetic files to generate
    latent_pool_sample_rate: int = 22050
    latent_pool_duration_s: float = 5.0
    latent_pool_noise_profiles: List[str] = field(default_factory=list)

    # Stream decoding
    max_streams: int = 0                # 0 = all
    chunk_samples: int = 0              # 0 = auto from render config
    min_stream_seconds: float = 0.5

    # Latent pool generation parameters
    seed: int = 42
    latent_pool_noise_std: float = 0.5
    latent_reinject_dir: str = ""
    latent_reinject_ratio: float = 0.0
    latent_reinject_copy_gain: float = 0.9
    latent_reinject_noise_gain: float = 0.1
    latent_structured_ratio: float = 0.25
    latent_structured_gain: float = 0.5
    latent_structured_noise_gain: float = 0.1


class WavePoolNode(PipelineNode):
    """Discover WAV files and decode them to float32 mono streams.

    Falls back to a synthetic latent noise pool when no WAV files are found.
    Populates ctx.wav_records, ctx.float_streams, ctx.stream_paths.
    """

    node_id = "wave_pool"
    description = "Discover WAV files and decode to float32 streams"

    def __init__(self, cfg: WavePoolConfig) -> None:
        self.cfg = cfg
        self._done = False

    def should_run(self, ctx: PipelineContext) -> bool:
        return not self._done

    def execute(self, ctx: PipelineContext) -> None:
        from wav_ml_core import discover_wavs, read_wav_record

        # Resolve WAV directory from node config or ctx.args
        wav_dir = str(self.cfg.wav_dir).strip() or str(getattr(ctx.args, "wav_dir", "")).strip()

        # Discover real WAV files
        records = []
        if wav_dir:
            records = list(discover_wavs(wav_dir))
            if self.cfg.max_streams > 0:
                records = records[: self.cfg.max_streams]

        # Fall back to synthetic pool
        # _bootstrap_latent_wav_pool(out_dir, seed, count, framerate, seconds, noise_std,
        #   reinject_dir, reinject_ratio, reinject_copy_gain, reinject_noise_gain,
        #   structured_ratio, structured_gain, structured_noise_gain)
        # returns Dict with key "pool_dir"
        if not records:
            pool_info = _bootstrap_latent_wav_pool(
                out_dir=self.cfg.latent_pool_dir or str(ctx.output_dir / "latent_wave_pool"),
                seed=self.cfg.seed,
                count=self.cfg.latent_pool_size,
                framerate=self.cfg.latent_pool_sample_rate,
                seconds=self.cfg.latent_pool_duration_s,
                noise_std=self.cfg.latent_pool_noise_std,
                reinject_dir=self.cfg.latent_reinject_dir,
                reinject_ratio=self.cfg.latent_reinject_ratio,
                reinject_copy_gain=self.cfg.latent_reinject_copy_gain,
                reinject_noise_gain=self.cfg.latent_reinject_noise_gain,
                structured_ratio=self.cfg.latent_structured_ratio,
                structured_gain=self.cfg.latent_structured_gain,
                structured_noise_gain=self.cfg.latent_structured_noise_gain,
            )
            pool_dir = str(pool_info["pool_dir"])
            ctx.latent_wav_pool_dir = Path(pool_dir)
            records = list(discover_wavs(pool_dir))
            _log(f"[wave-pool] using synthetic latent pool ({len(records)} files) at {pool_dir}")
        else:
            _log(f"[wave-pool] discovered {len(records)} real WAV files")

        # Decode to mono float32 streams directly via read_wav_record.
        # _prepare_streams from the monolith requires a RenderConfig that is not
        # available yet at this stage (config search runs after WavePoolNode).
        chunk_samples = int(self.cfg.chunk_samples)
        raw_streams: list = []
        paths: list = []
        for rec in records:
            try:
                wav = read_wav_record(rec)
                if wav is not None and getattr(wav, "float_data", None) is not None and len(wav.float_data) > 0:
                    raw_streams.append(wav.float_data)
                    paths.append(str(rec))
            except Exception as exc:
                _log(f"[wave-pool] WARNING: could not decode {rec}: {exc}")
        streams = raw_streams

        streams, paths = _filter_stream_pool_for_chunk_samples(
            streams=streams,
            paths=paths,
            chunk_samples=chunk_samples,
            min_seconds=self.cfg.min_stream_seconds,
            args=ctx.args,
        )

        ctx.wav_records = list(records)
        ctx.float_streams = list(streams)
        ctx.stream_paths = list(paths)
        self._done = True
        _log(f"[wave-pool] {len(ctx.float_streams)} decodable streams")


# ---------------------------------------------------------------------------
# Pre-gestation data node  (Stage 0)
# ---------------------------------------------------------------------------

@dataclass
class PregestationDataConfig:
    """Idiosyncrasies of the synthetic geometric direction+colour logic images."""

    # Number of samples generated per term×direction×colour combination
    samples_per_combo: int = 64

    # Image resolution (must match classifier input)
    image_size: int = 64

    # Batch and loader settings
    batch_size: int = 32
    num_workers: int = 0

    # On-disk cache cap (MB) — reserved for future use by dataset loaders
    cache_mb: int = 256

    # Sub-round generation mode sequence — each mode is one call to the monolith
    # helper; results are concatenated into a single loader.
    # Valid values: "direction_color", "symbol", "noise_texture"
    mode_sequence: List[str] = field(
        default_factory=lambda: ["direction_color", "symbol", "noise_texture"]
    )

    # circle_displacement_temperature controls geometric warp intensity
    displacement_temperature: float = 0.4

    # circle_radius_temperature controls symbol radius variation
    circle_radius_temperature: float = 1.0

    # RNG seed for reproducible generation
    seed: int = 42


class PregestationDataNode(PipelineNode):
    """Build the Stage 0 dataset: synthetic geometric direction+colour images.

    The dataset is generated programmatically from active semantic terms and
    their geometric/colour interpretations.  A loop-pool disk cache avoids
    re-generating the same images every round.

    Runs every round — the loader is rebuilt when vocab changes.
    """

    node_id = "pregestation_data"
    description = "Build Stage 0 pregestation dataset (synthetic geometric logic)"

    def __init__(self, cfg: PregestationDataConfig) -> None:
        self.cfg = cfg
        self._last_vocab_hash: int = -1

    def should_run(self, ctx: PipelineContext) -> bool:
        return bool(ctx.class_names)

    def execute(self, ctx: PipelineContext) -> None:
        current_hash = hash(tuple(ctx.class_names))
        if current_hash == self._last_vocab_hash and ctx.pregestation_loader is not None:
            return  # vocab unchanged; reuse existing loader

        from wav_config_transformer_pipeline import (
            _build_pregestation_logic_rows,
            _resolve_semantic_stage_cache_root,
        )
        from semantic_dataset_loaders import (
            build_loader_from_manifest,
            StageDatasetManifest,
        )

        # ctx.semantic_cache_nonce is set by the orchestrator at startup —
        # no ctx.args fallback needed.
        _resolve_semantic_stage_cache_root(
            output_dir=ctx.output_dir,
            cache_dir=ctx.semantic_stage_cache_dir,
            cache_nonce=ctx.semantic_cache_nonce or "",
        )

        # _build_pregestation_logic_rows(image_size, seed, samples_per_combo,
        #   active_terms_lc, circle_radius_temperature,
        #   circle_displacement_temperature, mode)
        # → (images, targets, terms_list, info) per single mode.
        # Loop over mode_sequence and concatenate.
        active_terms_lc = [t.lower() for t in ctx.class_names]
        all_images: list = []
        all_targets: list = []
        for mode in self.cfg.mode_sequence:
            imgs, tgts, _terms, _info = _build_pregestation_logic_rows(
                image_size=self.cfg.image_size,
                seed=self.cfg.seed,
                samples_per_combo=self.cfg.samples_per_combo,
                active_terms_lc=active_terms_lc,
                circle_radius_temperature=self.cfg.circle_radius_temperature,
                circle_displacement_temperature=self.cfg.displacement_temperature,
                mode=mode,
            )
            all_images.extend(imgs)
            all_targets.extend(tgts)

        ctx.pregestation_logic_rows = {"images": all_images, "targets": all_targets}

        if not all_images:
            _log("[pregestation-data] WARNING: no images built; skipping loader")
            return

        manifest = StageDatasetManifest(
            stage_name="pregestation",
            images=all_images,
            targets=all_targets,
        )
        loader = build_loader_from_manifest(
            manifest=manifest,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=True,
        )
        ctx.pregestation_loader = loader
        self._last_vocab_hash = current_hash
        _log(f"[pregestation-data] {len(all_images)} images "
             f"batch_size={self.cfg.batch_size}")


# ---------------------------------------------------------------------------
# Gestation data node  (Stage 1)
# ---------------------------------------------------------------------------

@dataclass
class GestationDataConfig:
    """Idiosyncrasies of the bootstrap primitive symbol dataset."""

    image_size: int = 64
    batch_size: int = 32
    num_workers: int = 0
    samples_per_term: int = 32
    cache_mb: int = 128


class GestationDataNode(PipelineNode):
    """Build the Stage 1 dataset: bootstrap primitive symbol images.

    Uses the symbol_pool built by BuildSymbolPoolNode.
    Rebuilt when vocab changes.
    """

    node_id = "gestation_data"
    description = "Build Stage 1 gestation dataset (bootstrap primitive symbols)"

    def __init__(self, cfg: GestationDataConfig) -> None:
        self.cfg = cfg
        self._last_vocab_hash: int = -1

    def should_run(self, ctx: PipelineContext) -> bool:
        return bool(ctx.class_names)

    def execute(self, ctx: PipelineContext) -> None:
        current_hash = hash(tuple(ctx.class_names))
        if current_hash == self._last_vocab_hash and ctx.gestation_loader is not None:
            return

        from semantic_dataset_loaders import (
            BootstrapDynamicDataset,
            build_loader_from_manifest,
            StageDatasetManifest,
        )

        symbol_pool = ctx.symbol_pool or {}

        # Flatten symbol pool into (image, target) pairs
        images, targets = _flatten_symbol_pool(
            symbol_pool=symbol_pool,
            class_names=ctx.class_names,
            semantic_term_to_idx=ctx.semantic_term_to_idx,
            samples_per_term=self.cfg.samples_per_term,
        )

        if not images:
            _log("[gestation-data] WARNING: no symbol images; gestation loader skipped")
            return

        manifest = StageDatasetManifest(
            stage_name="gestation",
            images=images,
            targets=targets,
        )
        loader = build_loader_from_manifest(
            manifest=manifest,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=True,
        )
        ctx.gestation_loader = loader
        self._last_vocab_hash = current_hash
        _log(f"[gestation-data] {len(images)} images batch_size={self.cfg.batch_size}")


# ---------------------------------------------------------------------------
# Berkeley payload bank node (runs once)
# ---------------------------------------------------------------------------

@dataclass
class BerkeleyPayloadConfig:
    """Idiosyncrasies of the Berkeley SBD payload image bank."""

    berkeley_data_root: str = ""     # empty → default location
    payload_bank_dir: str = ""       # empty → {berkeley_root}/cache/payload_bank_rgb...

    # Image resolution for payload rendering
    image_size: int = 64

    # Batch size for building the bank
    build_batch_size: int = 32
    build_num_workers: int = 0

    # Maximum number of images to load into the bank
    max_images: int = 0              # 0 = all available

    # Cache cap for payload bank disk storage (MB)
    cache_mb: int = 2048

    # Force rebuild of on-disk cache (ignores cached files)
    force_cache_rebuild: bool = False

    # Payload bank construction parameters
    seed: int = 42
    auto_install_scipy: bool = False


class BerkeleyPayloadNode(OneTimeNode):
    """Build or load the on-disk Berkeley SBD payload image bank.

    The payload bank is a pre-rendered collection of Berkeley SBD images
    indexed by semantic labels.  It feeds the GAN generator (as supervised
    targets) and Stage 2 classifier training.
    """

    node_id = "berkeley_payload"
    description = "Build/load Berkeley SBD payload image bank"

    def __init__(self, cfg: BerkeleyPayloadConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def _execute_once(self, ctx: PipelineContext) -> None:

        berkeley_root = (
            str(self.cfg.berkeley_data_root).strip()
            or ctx.berkeley_data_root
            or str(getattr(ctx.args, "berkeley_data_root", ""))
        )

        # _build_berkeley_payload_bank(data_root, image_size, auto_install_scipy,
        #   max_samples, seed, source_root="", force_cache_rebuild=False)
        # returns (out_images, out_targets, info, out_terms, out_masks)
        out_images, _out_targets, _info, _out_terms, out_masks = _build_berkeley_payload_bank(
            data_root=berkeley_root,
            image_size=self.cfg.image_size,
            auto_install_scipy=self.cfg.auto_install_scipy,
            max_samples=self.cfg.max_images or 0,
            seed=self.cfg.seed,
            source_root=str(self.cfg.payload_bank_dir).strip(),
            force_cache_rebuild=self.cfg.force_cache_rebuild,
        )

        ctx.payload_bank = out_images
        ctx.payload_masks = list(out_masks) if out_masks else []
        ctx.payload_bank_ready = bool(out_images)

        # Build payload conditions tensor
        if out_images:
            conditions = _expand_payload_conditions_with_semantic_bank(
                payload_bank=out_images,
                class_names=ctx.class_names,
                label_embedding_bank=ctx.label_embedding_bank,
                args=ctx.args,
            )
            ctx.payload_conditions = conditions
            _log(f"[berkeley-payload] bank ready: {len(conditions)} condition rows")
        else:
            _log("[berkeley-payload] WARNING: payload bank not available")


# ---------------------------------------------------------------------------
# Berkeley refresh data node  (Stage 2)
# ---------------------------------------------------------------------------

@dataclass
class BerkeleyDataConfig:
    """Idiosyncrasies of the Berkeley SBD Stage 2 refresh loader."""

    batch_size: int = 16
    num_workers: int = 0
    prefetch_factor: int = 0

    # How many Berkeley refresh batches to pre-cache in memory/GPU
    prebuild_batches: int = 0      # 0 = stream on-the-fly

    # How many rounds between full Berkeley refresh loader rebuilds
    rebuild_every_n_rounds: int = 4

    # Gate 2 validation loader settings
    gate_val_batch_size: int = 32
    gate_val_num_workers: int = 0


class BerkeleyDataNode(GatedNode):
    """Build the Stage 2 Berkeley SBD refresh DataLoader.

    Requires Gate 0 (pre-gestation).  Rebuilt every rebuild_every_n_rounds rounds
    to cycle through the Berkeley dataset with fresh shuffles and updated
    semantic label assignments.
    """

    node_id = "berkeley_data"
    description = "Build Stage 2 Berkeley SBD refresh DataLoader"
    required_gates = ["gate_pregestation"]

    def __init__(self, cfg: BerkeleyDataConfig) -> None:
        self.cfg = cfg
        self._last_build_round: int = -1

    def should_run(self, ctx: PipelineContext) -> bool:
        if not super().should_run(ctx):
            return False
        if ctx.berkeley_refresh_loader is None:
            return True
        rounds_since = ctx.total_rounds_completed - self._last_build_round
        return rounds_since >= self.cfg.rebuild_every_n_rounds

    def execute(self, ctx: PipelineContext) -> None:

        loader = _build_berkeley_refresh_loader(
            payload_bank=ctx.payload_bank,
            class_names=ctx.class_names,
            semantic_term_to_idx=ctx.semantic_term_to_idx,
            label_embedding_bank=ctx.label_embedding_bank,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            prefetch_factor=self.cfg.prefetch_factor,
            device=ctx.device,
            args=ctx.args,
        )

        gate_val_loader = _build_berkeley_gate_val_loader(
            payload_bank=ctx.payload_bank,
            class_names=ctx.class_names,
            semantic_term_to_idx=ctx.semantic_term_to_idx,
            label_embedding_bank=ctx.label_embedding_bank,
            batch_size=self.cfg.gate_val_batch_size,
            num_workers=self.cfg.gate_val_num_workers,
            args=ctx.args,
        )

        # Optional: pre-cache N batches into memory for fast iteration
        if self.cfg.prebuild_batches > 0 and loader is not None:
            cache = _build_berkeley_refresh_cache(
                loader=loader,
                n_batches=self.cfg.prebuild_batches,
                device=ctx.device,
            )
            ctx.berkeley_cache = cache
        else:
            ctx.berkeley_cache = None

        ctx.berkeley_refresh_loader = loader
        ctx.berkeley_gate_val_loader = gate_val_loader
        self._last_build_round = ctx.total_rounds_completed

        _log(f"[berkeley-data] refresh loader rebuilt at round {ctx.total_rounds_completed}")


# ---------------------------------------------------------------------------
# Payload validation dataset node
# ---------------------------------------------------------------------------

class PayloadValidationDataNode(GatedNode):
    """Build the payload gate validation dataset (Gate 2 evaluation deck).

    Requires Gate 0.  Built once and reused.  Provides a fixed held-out subset
    of Berkeley payload images for the Berkeley confidence/F1 gate check.
    """

    node_id = "payload_validation_data"
    description = "Build payload gate validation dataset (Gate 2 deck)"
    required_gates = ["gate_pregestation"]

    def __init__(self) -> None:
        self._built = False

    def should_run(self, ctx: PipelineContext) -> bool:
        if not super().should_run(ctx):
            return False
        return not self._built and ctx.payload_bank is not None

    def execute(self, ctx: PipelineContext) -> None:

        dataset = _build_payload_validation_gate_dataset(
            payload_bank=ctx.payload_bank,
            class_names=ctx.class_names,
            semantic_term_to_idx=ctx.semantic_term_to_idx,
            label_embedding_bank=ctx.label_embedding_bank,
            args=ctx.args,
        )

        loader = _build_gate_loader_from_dataset(
            dataset=dataset,
            batch_size=32,
            num_workers=0,
        )

        ctx.payload_validation_dataset = dataset
        ctx.payload_validation_loader = loader
        self._built = True
        _log(f"[payload-val-data] built {len(dataset) if dataset else 0} rows")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _bootstrap_latent_wav_pool(
    out_dir: Any,
    seed: int,
    count: int,
    framerate: int,
    seconds: float,
    noise_std: float,
    reinject_dir: str,
    reinject_ratio: float,
    reinject_copy_gain: float,
    reinject_noise_gain: float,
    structured_ratio: float,
    structured_gain: float,
    structured_noise_gain: float,
) -> Dict[str, Any]:
    import json
    import re

    import numpy as np

    from wav_config_transformer_pipeline import (
        RenderConfig,
        _decode_record_to_mono,
        _fit_wave_length,
        _load_reinject_wave_paths,
        _sample_latent_noise_profile_key,
        _save_mono_wav,
        _semantic_noise_terms_from_spectrum_sample,
        _synthesize_profiled_noise_wave,
        _synthesize_structured_wave,
        read_wav_record,
    )

    out_root = Path(out_dir)
    rng = np.random.default_rng(int(seed) + 424242)
    sample_count = max(256, int(float(framerate) * max(0.05, float(seconds))))
    pool_dir = out_root / 'latent_wave_pool'
    noise_dir = pool_dir / 'noise'
    mix_dir = pool_dir / 'mix'
    noise_dir.mkdir(parents=True, exist_ok=True)
    mix_dir.mkdir(parents=True, exist_ok=True)

    if reinject_dir:
        lib_dir = Path(reinject_dir)
    else:
        lib_dir = out_root / 'accepted_wave_library'
    reinject_paths = _load_reinject_wave_paths(lib_dir, limit=100000)

    rows = []
    mixed_count = 0
    structured_count = 0
    noise_profile_counts: Dict[str, int] = {}
    for i in range(max(1, int(count))):
        noise_profile = _sample_latent_noise_profile_key(rng)
        noise_raw = _synthesize_profiled_noise_wave(
            target_samples=int(sample_count),
            framerate=int(framerate),
            profile_key=str(noise_profile),
            rng=rng,
        )
        noise = np.asarray(noise_raw, dtype=np.float32) * float(noise_std)
        noise_terms = _semantic_noise_terms_from_spectrum_sample(noise_raw)
        y = noise.copy()
        src = ''
        used_mix = False

        if len(reinject_paths) > 0 and float(rng.random()) < float(reinject_ratio):
            src = reinject_paths[int(rng.integers(0, len(reinject_paths)))]
            try:
                rec = read_wav_record(src)
                mono, _ = _decode_record_to_mono(rec, cfg=RenderConfig(), max_points=0)
                ref = _fit_wave_length(mono, target_samples=sample_count, rng=rng)
                peak = float(np.max(np.abs(ref))) if ref.size > 0 else 0.0
                if peak > 1e-6:
                    ref = ref / peak
                y = (float(reinject_noise_gain) * noise) + (float(reinject_copy_gain) * ref)
                used_mix = True
                mixed_count += 1
            except Exception:
                src = ''
                y = noise
        elif float(rng.random()) < float(structured_ratio):
            proto = _synthesize_structured_wave(
                target_samples=sample_count,
                framerate=framerate,
                rng=rng,
            )
            y = (float(structured_noise_gain) * noise) + (float(structured_gain) * proto)
            src = '__structured_seed__'
            used_mix = True
            structured_count += 1

        y = np.clip(y, -1.0, 1.0)
        target_dir = mix_dir if used_mix else noise_dir
        prof_slug = re.sub(r'[^a-z0-9_]+', '_', str(noise_profile).strip().lower()).strip('_')
        if not prof_slug:
            prof_slug = 'gaussian_white_noise'
        out_path = target_dir / f'latent_{i:05d}_{prof_slug}.wav'
        _save_mono_wav(out_path, y, framerate=framerate)
        rows.append(
            {
                'file': str(out_path),
                'mixed': bool(used_mix),
                'source': src,
                'noise_profile': str(prof_slug),
                'noise_terms': list(noise_terms),
            }
        )
        noise_profile_counts[str(prof_slug)] = int(noise_profile_counts.get(str(prof_slug), 0)) + 1

    index_path = pool_dir / 'index.jsonl'
    with index_path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + '\n')

    manifest = pool_dir / 'manifest.json'
    manifest.write_text(
        json.dumps(
            {
                'seed': int(seed),
                'count': int(count),
                'framerate': int(framerate),
                'seconds': float(seconds),
                'sample_count': int(sample_count),
                'noise_std': float(noise_std),
                'reinject_dir': str(lib_dir),
                'reinject_ratio': float(reinject_ratio),
                'reinject_copy_gain': float(reinject_copy_gain),
                'reinject_noise_gain': float(reinject_noise_gain),
                'reinject_candidates': len(reinject_paths),
                'mixed_count': int(mixed_count),
                'structured_ratio': float(structured_ratio),
                'structured_gain': float(structured_gain),
                'structured_noise_gain': float(structured_noise_gain),
                'structured_count': int(structured_count),
                'noise_profiles': {str(k): int(v) for k, v in sorted(noise_profile_counts.items(), key=lambda kv: str(kv[0]))},
                'index': str(index_path),
            },
            indent=2,
        ),
        encoding='utf-8',
    )
    return {
        'pool_dir': str(pool_dir),
        'manifest': str(manifest),
        'count': int(count),
        'reinject_dir': str(lib_dir),
        'reinject_candidates': len(reinject_paths),
        'mixed_count': int(mixed_count),
        'structured_count': int(structured_count),
        'index': str(index_path),
        'noise_profiles': {str(k): int(v) for k, v in sorted(noise_profile_counts.items(), key=lambda kv: str(kv[0]))},
    }


def _flatten_symbol_pool(
    symbol_pool: Dict[str, Any],
    class_names: List[str],
    semantic_term_to_idx: Dict[str, int],
    samples_per_term: int,
) -> tuple:
    import numpy as np

    images = []
    targets = []
    n_classes = len(class_names)

    for term, term_images in symbol_pool.items():
        term_lc = str(term).strip().lower()
        idx = semantic_term_to_idx.get(term_lc, -1)
        if idx < 0:
            continue
        term_imgs = list(term_images)[:samples_per_term]
        for img in term_imgs:
            target = np.zeros(n_classes, dtype=np.float32)
            target[idx] = 1.0
            images.append(img)
            targets.append(target)

    return images, targets


def _log(msg: str) -> None:
    print(msg, flush=True)


# =========================================================================
# Functions extracted from wav_config_transformer_pipeline.py
# =========================================================================


def _sum_berkeley_refresh_samples(rows: Sequence[Dict[str, Any]]) -> int:
    total = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            samples_v = int(float(row.get("samples", 0)))
        except Exception:
            continue
        if samples_v <= 0:
            continue
        src = str(row.get("source", "")).strip().lower()
        if src == "fake_vector":
            continue
        total += int(samples_v)
    return int(max(0, total))


def _evaluate_berkeley_confidence_loss_gate(
    eval_stats: Optional[Dict[str, Any]],
    gate_config: Dict[str, Any],
    round_index: int,
    loss_history: Sequence[float],
    refresh_samples_seen: int = 0,
    refresh_samples_required: int = 0,
) -> Dict[str, Any]:
    _ = int(round_index)  # Reserved for future schedule variants; hard gate is time-invariant.
    _ = loss_history

    min_conf = max(0.0, float(gate_config.get("berkeley_min_confidence", 0.0)))
    min_macro_f1 = max(0.0, float(gate_config.get("berkeley_min_macro_f1", 0.0)))
    loss_floor = max(0.0, float(gate_config.get("berkeley_loss_min_floor", 0.0)))
    loss_target_cfg = float(gate_config.get("berkeley_loss_target", 0.0))
    loss_enabled = bool(loss_target_cfg > 0.0)
    loss_target = (None if not loss_enabled else float(max(loss_floor, loss_target_cfg)))

    refresh_required = max(0, int(refresh_samples_required))
    refresh_seen = max(0, int(refresh_samples_seen))
    refresh_coverage = (
        1.0
        if refresh_required <= 0
        else float(max(0.0, min(1.0, float(refresh_seen) / float(max(1, refresh_required)))))
    )
    refresh_pass = bool((refresh_required <= 0) or (refresh_seen >= refresh_required))

    if not isinstance(eval_stats, dict):
        confidence_pass = bool(min_conf <= 0.0)
        macro_f1_pass = bool(min_macro_f1 <= 0.0)
        loss_pass = bool(not loss_enabled)
        gate_pass = bool(refresh_pass and confidence_pass and macro_f1_pass and loss_pass)
        if not refresh_pass:
            reason = f"refresh<{refresh_required}"
        elif not confidence_pass:
            reason = f"conf<{min_conf:.3f}"
        elif not macro_f1_pass:
            reason = f"macro_f1<{min_macro_f1:.3f}"
        elif not loss_pass:
            reason = "loss_eval_missing"
        else:
            reason = "pass"
        mode = "hard_loss_full_refresh" if (loss_enabled or refresh_required > 0) else "confidence_only"
        return {
            "pass": bool(gate_pass),
            "mode": str(mode),
            "reason": str(reason),
            "confidence_pass": bool(confidence_pass),
            "macro_f1_pass": bool(macro_f1_pass),
            "loss_pass": bool(loss_pass),
            "refresh_pass": bool(refresh_pass),
            "refresh_samples_seen": int(refresh_seen),
            "refresh_samples_required": int(refresh_required),
            "refresh_coverage": float(refresh_coverage),
            "loss_target": loss_target,
            "confidence": None,
            "macro_f1": None,
            "loss": None,
        }

    conf = float(eval_stats.get("mean_confidence", 0.0))
    macro_f1 = float(eval_stats.get("macro_f1", 0.0))
    loss_v_raw = float(eval_stats.get("loss", float("nan")))
    loss_v = float(loss_v_raw) if math.isfinite(loss_v_raw) else float("nan")

    confidence_pass = (min_conf <= 0.0) or (conf >= min_conf)
    macro_f1_pass = (min_macro_f1 <= 0.0) or (macro_f1 >= min_macro_f1)
    if not loss_enabled:
        loss_pass = True
    elif not math.isfinite(loss_v):
        loss_pass = False
    else:
        loss_pass = bool(float(loss_v) <= float(loss_target))

    gate_pass = bool(refresh_pass and confidence_pass and macro_f1_pass and loss_pass)
    if not refresh_pass:
        reason = f"refresh<{refresh_required}"
    elif not confidence_pass:
        reason = f"conf<{min_conf:.3f}"
    elif not macro_f1_pass:
        reason = f"macro_f1<{min_macro_f1:.3f}"
    elif not loss_pass:
        reason = "loss_eval_missing" if (not math.isfinite(loss_v)) else f"loss>{float(loss_target):.4f}"
    else:
        reason = "pass"
    mode = "hard_loss_full_refresh" if (loss_enabled or refresh_required > 0) else "confidence_only"

    return {
        "pass": bool(gate_pass),
        "mode": str(mode),
        "reason": str(reason),
        "confidence_pass": bool(confidence_pass),
        "macro_f1_pass": bool(macro_f1_pass),
        "loss_pass": bool(loss_pass),
        "refresh_pass": bool(refresh_pass),
        "refresh_samples_seen": int(refresh_seen),
        "refresh_samples_required": int(refresh_required),
        "refresh_coverage": float(refresh_coverage),
        "loss_target": loss_target,
        "confidence": float(conf),
        "macro_f1": float(macro_f1),
        "loss": (None if not math.isfinite(loss_v) else float(max(0.0, loss_v))),
    }


def _build_berkeley_refresh_loader(
    data_root: str,
    image_size: int,
    auto_install_scipy: bool,
    batch_size: int,
    num_workers: int,
    max_train: int,
    seed: int,
    device: torch.device,
    external_val_fraction: float = 0.20,
    validation_split_seed: int = 0,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    return_mask_stack: bool = False,
):
    del auto_install_scipy
    rows, rows_info = collect_semantic_disk_rows(
        data_root=str(data_root),
        class_names=_default_berkeley_class_names(),
        source_root="",
    )
    n_rows = int(len(rows))
    if n_rows <= 0:
        raise RuntimeError("Berkeley disk row collection is empty for refresh loader.")
    source_rows = [str(r.source) for r in rows]
    split_seed = int(validation_split_seed) if int(validation_split_seed) != 0 else int(seed)
    train_idx, _, source_stats = _split_payload_cache_indices(
        source_rows=source_rows,
        seed=int(split_seed),
        external_val_fraction=float(external_val_fraction),
    )
    if int(len(train_idx)) <= 0:
        raise RuntimeError("Refresh loader has no non-validation rows after split.")
    idx = np.asarray(train_idx, dtype=np.int64)
    if int(max_train) > 0 and int(idx.shape[0]) > int(max_train):
        rng = np.random.default_rng(max(0, int(seed)))
        idx = rng.choice(idx, size=int(max_train), replace=False).astype(np.int64)
    else:
        rng = np.random.default_rng(max(0, int(seed)))
        rng.shuffle(idx)
    selected_rows = [rows[int(i)] for i in idx.tolist()]
    ds = DiskSemanticRowsDataset(
        rows=selected_rows,
        image_size=int(image_size),
        return_masks=True,
        return_mask_stack=bool(return_mask_stack),
        degrade=True,
        degrade_seed=int(seed),
    )
    loader_num_workers = _effective_dataloader_num_workers(num_workers=num_workers, device=device)
    loader = DataLoader(
        ds,
        batch_size=max(1, int(batch_size)),
        shuffle=True,
        num_workers=int(loader_num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=(semantic_mask_stack_collate if bool(getattr(ds, "use_semantic_mask_stack_collate", False)) else None),
        **_dataloader_perf_kwargs(
            num_workers=int(loader_num_workers),
            persistent_workers=bool(persistent_workers),
            prefetch_factor=int(prefetch_factor),
        ),
    )
    loader = maybe_wrap_loader_with_threaded_prefetch(
        loader=loader,
        requested_workers=int(num_workers),
        effective_workers=int(loader_num_workers),
        device_type=str(device.type),
        prefetch_factor=int(prefetch_factor),
    )
    setattr(loader, "_refresh_source_stats", source_stats)
    setattr(loader, "_refresh_rows_info", rows_info)
    return loader, int(len(ds))


def _build_berkeley_refresh_cache(
    loader: DataLoader,
    device: torch.device,
    cache_batches: int,
    cache_device: str,
    channels_last: bool,
):
    n_batches = max(0, int(cache_batches))
    mode = str(cache_device).strip().lower()
    if n_batches <= 0 or mode == "none":
        return None

    if mode == "cuda":
        target = torch.device(str(device))
    else:
        target = torch.device("cpu")

    xs = []
    ys = []
    ms = []
    for i, batch in enumerate(loader, start=1):
        xb, yb, mb, _ = _unpack_masked_semantic_batch(batch, context="berkeley refresh cache")
        xs.append(xb)
        ys.append(yb)
        ms.append(mb)
        if i >= n_batches:
            break

    if len(xs) == 0:
        return None

    x = torch.cat(xs, dim=0).contiguous().detach()
    y = torch.cat(ys, dim=0).contiguous().detach()
    m = torch.cat(ms, dim=0).contiguous().detach()
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)

    estimated_bytes = int(x.numel()) * int(x.element_size())
    estimated_bytes += int(y.numel()) * int(y.element_size())
    estimated_bytes += int(m.numel()) * int(m.element_size())
    target_effective = torch.device(str(target))
    target_free_bytes = 0
    if target_effective.type == "cuda":
        try:
            target_free_bytes, _ = torch.cuda.mem_get_info(target_effective)
        except Exception:
            try:
                target_free_bytes, _ = torch.cuda.mem_get_info()
            except Exception:
                target_free_bytes = 0
        # PyTorch's caching allocator holds reserved-but-not-allocated blocks that
        # CUDA considers occupied but are freely reusable by the allocator.  Add
        # this cache back so the staging guard uses effective usable VRAM, not
        # just the CUDA-reported free bytes.
        try:
            _pt_cache = max(0, int(torch.cuda.memory_reserved(target_effective)) - int(torch.cuda.memory_allocated(target_effective)))
            target_free_bytes = int(target_free_bytes) + _pt_cache
        except Exception:
            pass
        if int(target_free_bytes) > 0 and int(estimated_bytes) >= int(0.25 * float(target_free_bytes)):
            target_effective = torch.device("cpu")

    if target_effective.type == "cuda":
        x = x.to(target_effective, non_blocking=True)
        y = y.to(target_effective, non_blocking=True)
        m = m.to(target_effective, non_blocking=True)
    elif device.type == "cuda":
        x = x.pin_memory()
        y = y.pin_memory()
        m = m.pin_memory()

    return {
        "x": x,
        "y": y,
        "m": m,
        "num_samples": int(y.shape[0]),
        "num_batches": len(xs),
        "device": str(target_effective),
        "requested_device": str(target),
        "estimated_bytes": int(estimated_bytes),
        "target_free_bytes": int(target_free_bytes),
        "staging_only": True,
        "cross_device_staging": bool(target_effective.type == "cuda" and str(target_effective) != str(device)),
    }


def _auto_berkeley_refresh_batch_size(
    cache_x: torch.Tensor,
    cache_y: torch.Tensor,
    cache_m: Optional[torch.Tensor],
    device: torch.device,
    vram_fraction: float,
    activation_multiplier: float,
    max_cap: int,
):
    if int(cache_x.shape[0]) <= 0:
        return 1
    max_cap = int(max_cap)
    hard_cap = int(cache_x.shape[0]) if max_cap <= 0 else min(int(cache_x.shape[0]), max_cap)
    if device.type != "cuda":
        return max(1, hard_cap)
    try:
        free_bytes, _ = torch.cuda.mem_get_info(device)
    except Exception:
        try:
            free_bytes, _ = torch.cuda.mem_get_info()
        except Exception:
            free_bytes = 0
    # PyTorch's caching allocator keeps reserved-but-idle blocks that CUDA marks
    # as occupied.  Add that back so the microbatch limit sees actual usable VRAM
    # rather than just CUDA-free bytes, which ignores allocator cache entirely.
    try:
        _pt_cache = max(0, int(torch.cuda.memory_reserved(device)) - int(torch.cuda.memory_allocated(device)))
        free_bytes = int(free_bytes) + _pt_cache
    except Exception:
        pass

    sample_bytes = int(cache_x[0].numel()) * int(cache_x.element_size())
    if cache_y is not None:
        sample_bytes += int(cache_y[0].numel()) * int(cache_y.element_size())
    if cache_m is not None:
        sample_bytes += int(cache_m[0].numel()) * int(cache_m.element_size())
    sample_bytes = max(1, sample_bytes)
    mult = max(1.0, float(activation_multiplier))
    target = int(max(0.0, min(0.99, float(vram_fraction))) * float(max(0, free_bytes)))
    if target <= 0:
        return max(1, min(64, hard_cap))
    est = int(target // int(sample_bytes * mult))
    return max(1, min(est, hard_cap))


def _build_berkeley_gate_val_loader(
    data_root: str,
    image_size: int,
    auto_install_scipy: bool,
    batch_size: int,
    num_workers: int,
    max_val: int,
    seed: int,
    device: torch.device,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    return_mask_stack: bool = False,
):
    del auto_install_scipy
    rows, _ = collect_semantic_disk_rows(
        data_root=str(data_root),
        class_names=_default_berkeley_class_names(),
        source_root="",
    )
    val_rows = [r for r in rows if str(r.source).strip().lower() == "berkeley_sbd_val"]
    val_ds = DiskSemanticRowsDataset(
        rows=val_rows,
        image_size=int(image_size),
        return_masks=True,
        return_mask_stack=bool(return_mask_stack),
        degrade=False,
        degrade_seed=int(seed),
    )
    sampler = None
    shuffle = False
    sample_count = int(len(val_ds))
    if max_val > 0 and len(val_ds) > int(max_val):
        sample_count = int(max_val)
        g = torch.Generator()
        g.manual_seed(max(0, int(seed)))
        sampler = torch.utils.data.RandomSampler(
            data_source=val_ds,
            replacement=False,
            num_samples=int(sample_count),
            generator=g,
        )
        shuffle = False

    loader_num_workers = _effective_dataloader_num_workers(num_workers=num_workers, device=device)
    loader = DataLoader(
        val_ds,
        batch_size=max(1, int(batch_size)),
        shuffle=bool(shuffle),
        sampler=sampler,
        num_workers=int(loader_num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=(semantic_mask_stack_collate if bool(getattr(val_ds, "use_semantic_mask_stack_collate", False)) else None),
        **_dataloader_perf_kwargs(
            num_workers=int(loader_num_workers),
            persistent_workers=bool(persistent_workers),
            prefetch_factor=int(prefetch_factor),
        ),
    )
    loader = maybe_wrap_loader_with_threaded_prefetch(
        loader=loader,
        requested_workers=int(num_workers),
        effective_workers=int(loader_num_workers),
        device_type=str(device.type),
        prefetch_factor=int(prefetch_factor),
    )
    return loader, int(sample_count)


def _schedule_payload_gate_rows_by_active_terms(
    payload_terms_rows: Sequence[Sequence[str]],
    payload_global_picks: Sequence[int],
    payload_labels_mm: np.ndarray,
    supervised_class_names: Sequence[str],
    active_terms_lc: Sequence[str],
    active_vocab_lc: Sequence[str],
    require_all_supervised_active: bool = True,
    allow_unmapped_rows: bool = True,
) -> Tuple[List[int], List[int], List[List[str]], Dict[str, Any]]:
    n_rows = min(int(len(payload_terms_rows)), int(len(payload_global_picks)))
    active_order = list(
        dict.fromkeys(
            [
                re.sub(r"\s+", " ", str(t)).strip().lower()
                for t in active_terms_lc
                if re.sub(r"\s+", " ", str(t)).strip()
            ]
        )
    )
    active_set = set(active_order)
    active_vocab_set = {
        re.sub(r"\s+", " ", str(t)).strip().lower()
        for t in active_vocab_lc
        if re.sub(r"\s+", " ", str(t)).strip()
    }
    sup_name_lc = [
        re.sub(r"\s+", " ", str(x)).strip().lower()
        for x in supervised_class_names
        if re.sub(r"\s+", " ", str(x)).strip()
    ]
    order_lut = {str(t): int(i) for i, t in enumerate(active_order)}
    buckets: Dict[str, List[Tuple[int, int, List[str], List[str], int]]] = {}
    fallback_rows: List[Tuple[int, int, List[str], List[str], int]] = []
    skipped_no_active = 0
    skipped_inactive_supervised = 0
    selected_sup_positive_counts: List[int] = []
    selected_berkeley_rows = 0

    for ri in range(int(n_rows)):
        src_idx = int(payload_global_picks[int(ri)])
        if not (0 <= int(src_idx) < int(payload_labels_mm.shape[0])):
            raise RuntimeError(
                "Stage-2 payload gate supervised-label index out of bounds while scheduling rows: "
                f"row={int(ri)} pick={int(src_idx)} labels_rows={int(payload_labels_mm.shape[0])}"
            )
        raw_terms = payload_terms_rows[int(ri)]
        terms = _normalize_vocab_terms([str(x) for x in list(raw_terms)]) if isinstance(raw_terms, (list, tuple)) else []
        if len(terms) <= 0:
            raise RuntimeError(
                "Stage-2 payload gate encountered empty original terms row during scheduling; "
                f"row={int(ri)} global_pick={int(src_idx)}"
            )
        row_lc = {
            re.sub(r"\s+", " ", str(x)).strip().lower()
            for x in terms
            if re.sub(r"\s+", " ", str(x)).strip()
        }
        base_sup = np.asarray(payload_labels_mm[int(src_idx)], dtype=np.float32).reshape(-1)
        sup_take = min(int(len(sup_name_lc)), int(base_sup.size))
        sup_pos_idx = (
            np.where(np.asarray(base_sup[: int(sup_take)], dtype=np.float32) >= 0.5)[0].astype(np.int64).tolist()
            if int(sup_take) > 0
            else []
        )
        sup_terms_lc = []
        for sp in sup_pos_idx:
            if 0 <= int(sp) < int(len(sup_name_lc)):
                nm = str(sup_name_lc[int(sp)])
                if nm:
                    sup_terms_lc.append(nm)
        sup_terms_lc = list(dict.fromkeys(sup_terms_lc))
        missing_sup_vocab = [str(t) for t in sup_terms_lc if str(t) not in active_vocab_set]
        if len(missing_sup_vocab) > 0:
            raise RuntimeError(
                "Stage-2 payload gate row contains supervised Berkeley labels not present in active vocabulary. "
                f"row={int(ri)} pick={int(src_idx)} missing={','.join(missing_sup_vocab[:8])}"
            )
        row_is_berkeley = bool("berkeley sbd dataset" in row_lc)
        if bool(row_is_berkeley):
            if int(len(sup_terms_lc)) <= 0:
                raise RuntimeError(
                    "Stage-2 payload gate cannot pass Berkeley rows without supervised labels. "
                    f"row={int(ri)} pick={int(src_idx)}"
                )
            if bool(require_all_supervised_active) and any(str(t) not in active_set for t in sup_terms_lc):
                skipped_inactive_supervised += 1
                if not bool(allow_unmapped_rows):
                    continue
            selected_berkeley_rows += 1
            selected_sup_positive_counts.append(int(len(sup_terms_lc)))
        row_active_terms = [
            str(t)
            for t in active_order
            if (str(t) in row_lc) or (str(t) in set(sup_terms_lc))
        ]
        if int(len(row_active_terms)) <= 0:
            skipped_no_active += 1
            if not bool(allow_unmapped_rows):
                continue
            anchor = ""
        else:
            anchor = str(row_active_terms[0])
        row_payload = (
            int(ri),
            int(src_idx),
            list(terms),
            list(row_active_terms),
            int(len(sup_terms_lc)),
        )
        if anchor in active_set and int(len(anchor)) > 0:
            buckets.setdefault(str(anchor), []).append(row_payload)
        else:
            fallback_rows.append(row_payload)

    for key in list(buckets.keys()):
        rows = list(buckets.get(str(key), []))
        rows.sort(
            key=lambda row: (
                int(order_lut.get(str(row[3][0]) if len(row[3]) > 0 else str(key), 10**9)),
                -int(len(row[3])),
                -int(row[4]),
                int(row[0]),
            )
        )
        buckets[str(key)] = rows
    fallback_rows.sort(key=lambda row: (-int(len(row[3])), -int(row[4]), int(row[0])))

    ordered_rows: List[Tuple[int, int, List[str], List[str], int]] = []
    used_local: set = set()
    progress = True
    while bool(progress):
        progress = False
        for term in active_order:
            q = buckets.get(str(term), [])
            while int(len(q)) > 0 and int(q[0][0]) in used_local:
                q.pop(0)
            if int(len(q)) <= 0:
                continue
            row = q.pop(0)
            if int(row[0]) in used_local:
                continue
            used_local.add(int(row[0]))
            ordered_rows.append(row)
            progress = True
        if not bool(progress):
            break
    leftovers: List[Tuple[int, int, List[str], List[str], int]] = []
    for rows in buckets.values():
        leftovers.extend([row for row in rows if int(row[0]) not in used_local])
    leftovers.extend([row for row in fallback_rows if int(row[0]) not in used_local])
    leftovers.sort(
        key=lambda row: (
            int(order_lut.get(str(row[3][0]) if len(row[3]) > 0 else "", 10**9)),
            -int(len(row[3])),
            -int(row[4]),
            int(row[0]),
        )
    )
    ordered_rows.extend(leftovers)

    out_local_rows = [int(r[0]) for r in ordered_rows]
    out_global_picks = [int(r[1]) for r in ordered_rows]
    out_terms = [list(r[2]) for r in ordered_rows]
    info = {
        "rows_total": int(n_rows),
        "rows_selected": int(len(out_local_rows)),
        "rows_skipped_no_active": int(skipped_no_active),
        "rows_skipped_inactive_supervised": int(skipped_inactive_supervised),
        "active_terms": int(len(active_order)),
        "buckets": int(len([k for k, v in buckets.items() if int(len(v)) > 0])),
        "berkeley_rows_selected": int(selected_berkeley_rows),
        "berkeley_sup_min": int(np.min(selected_sup_positive_counts)) if int(len(selected_sup_positive_counts)) > 0 else 0,
        "berkeley_sup_mean": float(np.mean(selected_sup_positive_counts)) if int(len(selected_sup_positive_counts)) > 0 else 0.0,
        "berkeley_sup_max": int(np.max(selected_sup_positive_counts)) if int(len(selected_sup_positive_counts)) > 0 else 0,
        "mode": "deterministic_round_robin_by_active_terms",
    }
    return out_local_rows, out_global_picks, out_terms, info


def _schedule_semantic_gate_indices(
    targets: Sequence[np.ndarray],
    class_names: Sequence[str],
    active_terms: Sequence[str],
    seed: int,
    max_samples: int = 0,
    activation_threshold: float = 0.55,
) -> Tuple[List[int], Dict[str, Any]]:
    n = int(len(targets))
    expected_dim = max(1, int(len(class_names)))
    if n <= 0:
        return [], {
            "rows": 0,
            "target_dim_expected": int(expected_dim),
            "target_dim_min": 0,
            "target_dim_max": 0,
            "active_terms": 0,
            "active_term_hits": {},
            "bucketed_rows": 0,
            "selected_rows": 0,
            "max_samples": int(max_samples),
            "threshold": float(activation_threshold),
        }
    y_rows: List[np.ndarray] = []
    row_sizes: List[int] = []
    for i in range(int(n)):
        row = np.asarray(targets[int(i)], dtype=np.float32).reshape(-1)
        y_rows.append(row)
        row_sizes.append(int(row.size))
    bad_dims = sorted({int(sz) for sz in row_sizes if int(sz) != int(expected_dim)})
    if len(bad_dims) > 0:
        raise RuntimeError(
            "Semantic gate scheduler requires a single label width that matches the sentence-transformer "
            f"class width: expected={int(expected_dim)} got_dims={bad_dims}"
        )
    y = np.stack(y_rows, axis=0).astype(np.float32, copy=False)
    y = np.clip(y[:, : int(expected_dim)], 0.0, 1.0).astype(np.float32, copy=False)

    class_map = _semantic_term_index_map(class_names)
    active_terms_norm = _normalize_vocab_terms(active_terms)
    active_indices: List[int] = []
    for term in active_terms_norm:
        key = re.sub(r"\s+", " ", str(term)).strip().lower()
        if not key:
            continue
        idx = class_map.get(key, None)
        if idx is None:
            continue
        if int(idx) < 0 or int(idx) >= int(y.shape[1]):
            continue
        active_indices.append(int(idx))
    active_indices = list(dict.fromkeys(active_indices))

    _ = int(seed)  # Kept for API compatibility; scheduling is deterministic and staged.
    threshold = float(max(0.0, min(1.0, float(activation_threshold))))
    active_hits: Dict[str, int] = {}
    single_term_buckets: Dict[int, List[Tuple[float, int]]] = {int(idx): [] for idx in active_indices}
    signature_buckets: Dict[Tuple[int, ...], List[Tuple[float, int]]] = {}
    fallback_rows: List[Tuple[float, int]] = []
    rows_with_active_support = 0
    for idx in active_indices:
        name = str(class_names[int(idx)]) if int(idx) < int(len(class_names)) else f"class_{int(idx)}"
        active_hits[str(name)] = 0
    for ri in range(int(n)):
        row = np.asarray(y[int(ri)], dtype=np.float32).reshape(-1)
        support: List[int] = []
        for idx in active_indices:
            if float(row[int(idx)]) >= float(threshold):
                support.append(int(idx))
        if len(support) <= 0 and len(active_indices) > 0:
            vals = np.asarray(row[np.asarray(active_indices, dtype=np.int64)], dtype=np.float32).reshape(-1)
            if int(vals.size) > 0:
                top_local = int(np.argmax(vals))
                top_score = float(vals[int(top_local)])
                if float(top_score) > 1e-6:
                    support = [int(active_indices[int(top_local)])]
        if len(support) <= 0:
            fb_score = 0.0
            if len(active_indices) > 0:
                vals = np.asarray(row[np.asarray(active_indices, dtype=np.int64)], dtype=np.float32).reshape(-1)
                if int(vals.size) > 0:
                    fb_score = float(np.max(vals))
            fallback_rows.append((float(fb_score), int(ri)))
            continue
        rows_with_active_support += 1
        sig = tuple(sorted({int(x) for x in support}))
        sig_score = float(np.mean([float(row[int(ix)]) for ix in sig])) if len(sig) > 0 else 0.0
        signature_buckets.setdefault(sig, []).append((float(sig_score), int(ri)))
        for idx in sig:
            name = str(class_names[int(idx)]) if int(idx) < int(len(class_names)) else f"class_{int(idx)}"
            active_hits[str(name)] = int(active_hits.get(str(name), 0)) + 1
        if len(sig) == 1:
            s_idx = int(sig[0])
            single_term_buckets.setdefault(int(s_idx), []).append((float(row[int(s_idx)]), int(ri)))

    for idx in list(single_term_buckets.keys()):
        rows = list(single_term_buckets.get(int(idx), []))
        rows.sort(key=lambda x: (-float(x[0]), int(x[1])))
        single_term_buckets[int(idx)] = rows
    for sig in list(signature_buckets.keys()):
        rows = list(signature_buckets.get(sig, []))
        rows.sort(key=lambda x: (-float(x[0]), int(x[1])))
        signature_buckets[sig] = rows
    fallback_rows.sort(key=lambda x: (-float(x[0]), int(x[1])))

    ordered: List[int] = []
    used: set = set()
    stage_counts = {
        "single_term": 0,
        "multi_term": 0,
        "fallback": 0,
        "remaining": 0,
    }

    def _at_limit() -> bool:
        return bool(int(max_samples) > 0 and int(len(ordered)) >= int(max_samples))

    def _append_row(row_idx: int, stage_key: str) -> bool:
        r = int(row_idx)
        if r in used:
            return False
        ordered.append(int(r))
        used.add(int(r))
        stage_counts[str(stage_key)] = int(stage_counts.get(str(stage_key), 0)) + 1
        return True

    # Stage A: single-term groups (strict per-term round-robin).
    if len(active_indices) > 0:
        while not _at_limit():
            progressed = False
            for idx in active_indices:
                bucket = single_term_buckets.get(int(idx), [])
                while len(bucket) > 0 and int(bucket[0][1]) in used:
                    bucket.pop(0)
                if len(bucket) <= 0:
                    continue
                _, pick = bucket.pop(0)
                if _append_row(int(pick), stage_key="single_term"):
                    progressed = True
                if _at_limit():
                    break
            if not progressed:
                break

    # Stage B: multi-term signature groups (size-ascending, round-robin by signature).
    if not _at_limit():
        sig_groups = [sig for sig in signature_buckets.keys() if int(len(sig)) >= 2]
        size_values = sorted({int(len(sig)) for sig in sig_groups})
        for gsz in size_values:
            sigs = [sig for sig in sig_groups if int(len(sig)) == int(gsz)]
            sigs = sorted(
                sigs,
                key=lambda sig: tuple(
                    [
                        (
                            str(class_names[int(ix)]).strip().lower()
                            if int(ix) < int(len(class_names))
                            else f"class_{int(ix)}"
                        )
                        for ix in sig
                    ]
                ),
            )
            while not _at_limit():
                progressed = False
                for sig in sigs:
                    bucket = signature_buckets.get(sig, [])
                    while len(bucket) > 0 and int(bucket[0][1]) in used:
                        bucket.pop(0)
                    if len(bucket) <= 0:
                        continue
                    _, pick = bucket.pop(0)
                    if _append_row(int(pick), stage_key="multi_term"):
                        progressed = True
                    if _at_limit():
                        break
                if not progressed:
                    break
            if _at_limit():
                break

    # Stage C: fallback rows with weakest active support first promoted by score.
    if not _at_limit():
        for _, pick in fallback_rows:
            _append_row(int(pick), stage_key="fallback")
            if _at_limit():
                break

    # Stage D: deterministic remainder for full-deck stability.
    if not _at_limit():
        remaining = [int(i) for i in range(int(n)) if int(i) not in used]
        remaining = sorted(remaining)
        for pick in remaining:
            _append_row(int(pick), stage_key="remaining")
            if _at_limit():
                break

    if int(max_samples) > 0:
        ordered = ordered[: int(max_samples)]

    missing_terms: List[str] = []
    for idx in active_indices:
        name = str(class_names[int(idx)]) if int(idx) < int(len(class_names)) else f"class_{int(idx)}"
        if int(active_hits.get(str(name), 0)) <= 0:
            missing_terms.append(str(name))
    info = {
        "rows": int(n),
        "target_dim_expected": int(expected_dim),
        "target_dim_min": int(min(row_sizes)) if len(row_sizes) > 0 else 0,
        "target_dim_max": int(max(row_sizes)) if len(row_sizes) > 0 else 0,
        "active_terms": int(len(active_indices)),
        "active_term_hits": {str(k): int(v) for k, v in active_hits.items()},
        "missing_active_terms": list(missing_terms),
        "bucketed_rows": int(rows_with_active_support),
        "selected_rows": int(len(ordered)),
        "max_samples": int(max_samples),
        "threshold": float(threshold),
        "scheduler_mode": "deterministic_group_staged_v2",
        "stage_counts": {
            "single_term": int(stage_counts.get("single_term", 0)),
            "multi_term": int(stage_counts.get("multi_term", 0)),
            "fallback": int(stage_counts.get("fallback", 0)),
            "remaining": int(stage_counts.get("remaining", 0)),
        },
        "signature_groups": int(len(signature_buckets)),
    }
    return [int(i) for i in ordered], info


def _select_validation_indices(
    total_rows: int,
    seed: int,
    fraction: float = 0.25,
    min_count: int = 1,
    max_count: int = 0,
) -> List[int]:
    n = max(0, int(total_rows))
    if n <= 0:
        return []
    frac = float(max(0.0, min(1.0, float(fraction))))
    take = int(round(float(n) * frac))
    if int(min_count) > 0:
        take = max(int(take), int(min_count))
    if int(max_count) > 0:
        take = min(int(take), int(max_count))
    take = max(1, min(int(n), int(take)))
    idx = np.arange(int(n), dtype=np.int64)
    rng = np.random.default_rng(max(0, int(seed)))
    rng.shuffle(idx)
    return [int(i) for i in idx[: int(take)].tolist()]


def _build_payload_validation_gate_rows(
    data_root: str,
    image_size: int,
    seed: int,
    external_val_fraction: float = 0.20,
) -> Tuple[List[int], List[List[str]], Dict[str, Any]]:
    root = Path(str(data_root).strip() or "toys_to_survive_development/data/berkeley_sbd")
    size = max(8, int(image_size))
    cache_dir = root / "cache" / f"payload_bank_rgb{int(size)}_v2"
    manifest_path = cache_dir / "manifest.json"
    images_path = cache_dir / "images.npy"
    labels_path = cache_dir / "labels.npy"
    terms_path = cache_dir / "terms.json"
    sources_path = cache_dir / "sources.json"
    required = [manifest_path, images_path, labels_path, terms_path]
    missing = [str(p) for p in required if not p.exists()]
    if len(missing) > 0:
        raise RuntimeError(
            "Gate payload validation deck requires payload cache artifacts. "
            f"Missing: {missing}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed reading payload cache manifest for gate deck: {manifest_path} ({e})") from e
    if int(manifest.get("version", -1)) != 2:
        raise RuntimeError(
            "Unsupported payload cache version for gate deck "
            f"(expected=2, got={manifest.get('version', None)})."
        )
    if int(manifest.get("image_size", -1)) != int(size):
        raise RuntimeError(
            "Payload cache image-size mismatch for gate deck "
            f"(cache={manifest.get('image_size', None)} requested={int(size)})."
        )
    x_mem = np.load(str(images_path), mmap_mode="r")
    y_mem = np.load(str(labels_path), mmap_mode="r")
    if int(getattr(x_mem, "ndim", 0)) != 4 or int(getattr(y_mem, "ndim", 0)) != 2:
        raise RuntimeError(
            "Invalid payload cache tensor shapes for gate deck "
            f"(images={getattr(x_mem, 'shape', None)} labels={getattr(y_mem, 'shape', None)})."
        )
    terms_rows_raw: List[List[str]] = []
    try:
        raw_terms = json.loads(terms_path.read_text(encoding="utf-8"))
        if isinstance(raw_terms, list):
            for row in raw_terms:
                if isinstance(row, list):
                    terms_rows_raw.append(_normalize_vocab_terms([str(x) for x in row]))
                else:
                    terms_rows_raw.append([])
    except Exception as e:
        raise RuntimeError(f"Failed reading payload cache terms for gate deck: {terms_path} ({e})") from e

    n_rows = min(int(x_mem.shape[0]), int(y_mem.shape[0]), int(len(terms_rows_raw)))
    if n_rows <= 0:
        raise RuntimeError("Payload cache is empty for gate validation deck.")

    source_rows: List[str] = []
    if sources_path.exists():
        try:
            raw_sources = json.loads(sources_path.read_text(encoding="utf-8"))
            if isinstance(raw_sources, list):
                source_rows = [str(x) for x in raw_sources]
        except Exception:
            source_rows = []
    if int(len(source_rows)) < int(n_rows):
        source_rows.extend(["unknown_source"] * int(max(0, int(n_rows) - int(len(source_rows)))))
    if int(len(source_rows)) > int(n_rows):
        source_rows = source_rows[: int(n_rows)]

    _, gate_idx, source_stats = _split_payload_cache_indices(
        source_rows=source_rows,
        seed=int(seed),
        external_val_fraction=float(external_val_fraction),
    )
    selected_arr = np.asarray(gate_idx, dtype=np.int64)

    out_indices: List[int] = []
    out_terms: List[List[str]] = []
    for idx in selected_arr.tolist():
        if int(idx) < 0 or int(idx) >= int(n_rows):
            continue
        terms = list(terms_rows_raw[int(idx)]) if int(idx) < int(len(terms_rows_raw)) else []
        if len(terms) <= 0:
            raise RuntimeError(
                "Gate payload cache row is missing original terms; strict Stage-2 semantic gate requires "
                f"non-empty source terms. row_index={int(idx)}"
            )
        out_indices.append(int(idx))
        out_terms.append(list(terms))

    refresh_rows_total = int(sum(int(v.get("refresh_selected", 0)) for v in source_stats.values()))
    refresh_rows_berkeley_train = int(source_stats.get("berkeley_sbd_train", {}).get("refresh_selected", 0))
    refresh_rows_berkeley_val = int(source_stats.get("berkeley_sbd_val", {}).get("refresh_selected", 0))

    info = {
        "cache_dir": str(cache_dir),
        "manifest": str(manifest_path),
        "images_path": str(images_path),
        "labels_path": str(labels_path),
        "available_rows": int(n_rows),
        "selected_rows": int(len(out_indices)),
        "cache_label_dim": int(y_mem.shape[1]) if int(getattr(y_mem, "ndim", 0)) == 2 else 0,
        "selected_berkeley_val": int(source_stats.get("berkeley_sbd_val", {}).get("gate_selected", 0)),
        "selected_berkeley_train": int(source_stats.get("berkeley_sbd_train", {}).get("gate_selected", 0)),
        "refresh_rows_total": int(refresh_rows_total),
        "refresh_rows_berkeley_train": int(refresh_rows_berkeley_train),
        "refresh_rows_berkeley_val": int(refresh_rows_berkeley_val),
        "external_val_fraction": float(external_val_fraction),
        "source_stats": source_stats,
    }
    return out_indices, out_terms, info


def _build_payload_validation_gate_dataset(
    data_root: str,
    image_size: int,
    seed: int,
    external_val_fraction: float = 0.20,
    source_root: str = "",
    return_mask_stack: bool = False,
) -> Tuple[Optional[Dataset], np.ndarray, List[List[str]], Dict[str, Any]]:
    rows, rows_info = collect_semantic_disk_rows(
        data_root=str(data_root),
        class_names=_default_berkeley_class_names(),
        source_root=str(source_root),
    )
    if int(len(rows)) <= 0:
        info = dict(rows_info)
        info.update({"selected_rows": 0, "label_dim": 0, "reason": "empty_disk_rows"})
        return None, np.zeros((0, 0), dtype=np.float32), [], info
    source_rows = [str(r.source) for r in rows]
    _, gate_idx, source_stats = _split_payload_cache_indices(
        source_rows=source_rows,
        seed=int(seed),
        external_val_fraction=float(external_val_fraction),
    )
    selected_rows = [rows[int(i)] for i in gate_idx if 0 <= int(i) < int(len(rows))]
    terms_rows = [list(_normalize_vocab_terms(r.terms)) for r in selected_rows]
    label_rows = [np.asarray(r.label_vec, dtype=np.float32).reshape(-1) for r in selected_rows]
    label_dim = int(label_rows[0].size) if int(len(label_rows)) > 0 else 0
    labels_np = (
        np.stack(label_rows, axis=0).astype(np.float32, copy=False)
        if int(len(label_rows)) > 0
        else np.zeros((0, int(label_dim)), dtype=np.float32)
    )
    ds = DiskSemanticRowsDataset(
        rows=selected_rows,
        image_size=int(image_size),
        return_masks=True,
        return_mask_stack=bool(return_mask_stack),
        degrade=False,
        degrade_seed=int(seed),
    )
    refresh_rows_total = int(sum(int(v.get("refresh_selected", 0)) for v in source_stats.values()))
    refresh_rows_berkeley_train = int(source_stats.get("berkeley_sbd_train", {}).get("refresh_selected", 0))
    refresh_rows_berkeley_val = int(source_stats.get("berkeley_sbd_val", {}).get("refresh_selected", 0))
    info = {
        "available_rows": int(rows_info.get("available_rows", len(rows))),
        "selected_rows": int(len(selected_rows)),
        "label_dim": int(label_dim),
        "selected_berkeley_val": int(source_stats.get("berkeley_sbd_val", {}).get("gate_selected", 0)),
        "selected_berkeley_train": int(source_stats.get("berkeley_sbd_train", {}).get("gate_selected", 0)),
        "refresh_rows_total": int(refresh_rows_total),
        "refresh_rows_berkeley_train": int(refresh_rows_berkeley_train),
        "refresh_rows_berkeley_val": int(refresh_rows_berkeley_val),
        "external_val_fraction": float(external_val_fraction),
        "source_stats": source_stats,
        "cache_dir": str(Path(str(data_root).strip() or "toys_to_survive_development/data/berkeley_sbd") / "cache"),
        "data_source": "disk_dataset_rows",
        "source_root": str(source_root),
        "mask_cache_hits": int(rows_info.get("mask_cache_hits", 0)),
        "mask_cache_writes": int(rows_info.get("mask_cache_writes", 0)),
        "mask_cache_failures": int(rows_info.get("mask_cache_failures", 0)),
        "mask_cache_seconds": float(rows_info.get("mask_cache_seconds", 0.0)),
        "mask_cache_threads": int(rows_info.get("mask_cache_threads", 0)),
        "row_build_threads": int(rows_info.get("row_build_threads", 0)),
        "row_build_seconds": float(rows_info.get("row_build_seconds", 0.0)),
        "inprocess_cache_hit": bool(rows_info.get("inprocess_cache_hit", False)),
    }
    return ds, labels_np, terms_rows, info


def _build_gate_loader_from_arrays(
    images: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
    max_samples: int = 0,
    expected_target_dim: int = 0,
    ordered_indices: Optional[Sequence[int]] = None,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
) -> Tuple[Optional[DataLoader], int]:
    n = min(int(len(images)), int(len(targets)))
    if n <= 0:
        return None, 0
    x_rows: List[np.ndarray] = []
    for i in range(int(n)):
        arr = np.asarray(images[i], dtype=np.float32)
        if int(arr.ndim) == 3 and int(arr.shape[0]) == 3:
            row = arr
        elif int(arr.ndim) == 3 and int(arr.shape[2]) == 3:
            row = np.transpose(arr, (2, 0, 1)).astype(np.float32, copy=False)
        elif int(arr.ndim) == 2:
            row = np.repeat(arr[None, :, :], 3, axis=0).astype(np.float32, copy=False)
        else:
            raise RuntimeError(f"Unsupported gate image row shape: {tuple(arr.shape)}")
        x_rows.append(np.clip(row, 0.0, 1.0).astype(np.float32, copy=False))
    x_np = np.stack(x_rows, axis=0).astype(np.float32, copy=False)
    y_rows = [np.asarray(targets[i], dtype=np.float32).reshape(-1) for i in range(int(n))]
    y_sizes = [int(r.size) for r in y_rows]
    bad_dims = sorted({int(sz) for sz in y_sizes if int(sz) != int(y_sizes[0])}) if len(y_sizes) > 0 else []
    if len(bad_dims) > 0:
        dims = sorted({int(x) for x in y_sizes})
        raise RuntimeError(
            "Gate loader received mixed label widths; all rows must already match sentence-transformer semantic width. "
            f"got_dims={dims}"
        )
    y_dim = int(y_sizes[0]) if len(y_sizes) > 0 else 0
    if int(y_dim) <= 0:
        raise RuntimeError("Gate loader received empty label vectors.")
    if int(expected_target_dim) > 0 and int(y_dim) != int(expected_target_dim):
        raise RuntimeError(
            "Gate loader label width mismatch: "
            f"got={int(y_dim)} expected={int(expected_target_dim)}"
        )
    y_np = np.stack(y_rows, axis=0).astype(np.float32, copy=False)
    y_np = np.clip(y_np[:, : int(y_dim)], 0.0, 1.0).astype(np.float32, copy=False)

    if ordered_indices is None:
        picks = np.arange(int(n), dtype=np.int64)
        rng = np.random.default_rng(int(seed))
        rng.shuffle(picks)
        if int(max_samples) > 0:
            picks = picks[: int(min(int(max_samples), int(picks.shape[0])))]
    else:
        picks = np.asarray([int(i) for i in ordered_indices if 0 <= int(i) < int(n)], dtype=np.int64)
        if int(max_samples) > 0 and int(picks.size) > int(max_samples):
            picks = picks[: int(max_samples)]
    if int(picks.size) <= 0:
        return None, 0

    x_t = torch.from_numpy(x_np)
    y_t = torch.from_numpy(y_np)
    ds_full: Dataset = torch.utils.data.TensorDataset(x_t, y_t)
    ds_use: Dataset = torch.utils.data.Subset(ds_full, picks.tolist())
    loader_num_workers = _effective_dataloader_num_workers(num_workers=num_workers, device=device)
    loader = DataLoader(
        ds_use,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=int(loader_num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        **_dataloader_perf_kwargs(
            num_workers=int(loader_num_workers),
            persistent_workers=bool(persistent_workers),
            prefetch_factor=int(prefetch_factor),
        ),
    )
    loader = maybe_wrap_loader_with_threaded_prefetch(
        loader=loader,
        requested_workers=int(num_workers),
        effective_workers=int(loader_num_workers),
        device_type=str(device.type),
        prefetch_factor=int(prefetch_factor),
    )
    return loader, int(picks.size)


def _build_gate_loader_from_dataset(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
    max_samples: int = 0,
    ordered_indices: Optional[Sequence[int]] = None,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
) -> Tuple[Optional[DataLoader], int]:
    n = int(len(dataset))
    if n <= 0:
        return None, 0

    if ordered_indices is None:
        picks = np.arange(int(n), dtype=np.int64)
        rng = np.random.default_rng(int(seed))
        rng.shuffle(picks)
        if int(max_samples) > 0:
            picks = picks[: int(min(int(max_samples), int(picks.shape[0])))]
    else:
        picks = np.asarray([int(i) for i in ordered_indices if 0 <= int(i) < int(n)], dtype=np.int64)
        if int(max_samples) > 0 and int(picks.size) > int(max_samples):
            picks = picks[: int(max_samples)]
    if int(picks.size) <= 0:
        return None, 0

    ds_use: Dataset = dataset if int(picks.size) == int(n) else torch.utils.data.Subset(dataset, picks.tolist())
    collate_fn = semantic_mask_stack_collate if bool(getattr(dataset, "use_semantic_mask_stack_collate", False)) else None
    loader_num_workers = _effective_dataloader_num_workers(num_workers=num_workers, device=device)
    loader = DataLoader(
        ds_use,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=int(loader_num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=collate_fn,
        **_dataloader_perf_kwargs(
            num_workers=int(loader_num_workers),
            persistent_workers=bool(persistent_workers),
            prefetch_factor=int(prefetch_factor),
        ),
    )
    loader = maybe_wrap_loader_with_threaded_prefetch(
        loader=loader,
        requested_workers=int(num_workers),
        effective_workers=int(loader_num_workers),
        device_type=str(device.type),
        prefetch_factor=int(prefetch_factor),
    )
    return loader, int(picks.size)


def _unpack_masked_semantic_batch(batch: Any, context: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    meta: Dict[str, Any] = {}
    if isinstance(batch, dict):
        xb = batch.get("x")
        yb = batch.get("y")
        mb = batch.get("mask")
        if "mask_stacks" in batch:
            meta["mask_stacks"] = list(batch.get("mask_stacks") or [])
        if "mask_indices" in batch:
            meta["mask_indices"] = list(batch.get("mask_indices") or [])
    elif isinstance(batch, (tuple, list)) and int(len(batch)) >= 3:
        xb, yb, mb = batch[0], batch[1], batch[2]
        if int(len(batch)) >= 5:
            meta["mask_stacks"] = batch[3]
            meta["mask_indices"] = batch[4]
    else:
        raise RuntimeError(f"{str(context)} requires (image, target, mask) batches.")
    if not (torch.is_tensor(xb) and torch.is_tensor(yb) and torch.is_tensor(mb)):
        raise RuntimeError(f"{str(context)} received non-tensor batch members.")
    if int(mb.ndim) == 3:
        mb = mb.unsqueeze(1)
    elif int(mb.ndim) != 4:
        raise RuntimeError(f"{str(context)} requires mask tensors with shape [B,1,H,W]; got {tuple(mb.shape)}")
    if int(xb.shape[0]) != int(yb.shape[0]) or int(xb.shape[0]) != int(mb.shape[0]):
        raise RuntimeError(
            f"{str(context)} batch size mismatch: x={int(xb.shape[0])} y={int(yb.shape[0])} m={int(mb.shape[0])}"
        )
    return xb, yb, mb.to(dtype=torch.float32), meta


def _expand_semantic_mask_supervision_batch(
    xb: torch.Tensor,
    yb: torch.Tensor,
    mb: torch.Tensor,
    batch_meta: Optional[Dict[str, Any]],
    mode: str,
    context: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mode_key = re.sub(r"\s+", "_", str(mode)).strip().lower()
    if mode_key in ("", "multihot_mix", "multihot", "mixed"):
        return xb, yb, mb
    if mode_key not in ("single_label_passes", "single_passes", "single_label"):
        raise RuntimeError(f"{str(context)} received unsupported semantic mask supervision mode: {mode!r}")
    if int(xb.shape[0]) <= 0:
        return xb, yb, mb
    stack_list = []
    index_list = []
    if isinstance(batch_meta, dict):
        stack_list = list(batch_meta.get("mask_stacks") or [])
        index_list = list(batch_meta.get("mask_indices") or [])
    x_rows: List[torch.Tensor] = []
    y_rows: List[torch.Tensor] = []
    m_rows: List[torch.Tensor] = []
    for bi in range(int(xb.shape[0])):
        y_row = yb[bi].detach().to(torch.float32)
        active = torch.nonzero(y_row >= 0.5, as_tuple=False).reshape(-1)
        if int(active.numel()) <= 0:
            x_rows.append(xb[bi])
            y_rows.append(y_row)
            m_rows.append(mb[bi])
            continue
        sample_stack = None
        sample_idx = None
        if int(bi) < int(len(stack_list)):
            sample_stack = stack_list[int(bi)]
        if int(bi) < int(len(index_list)):
            sample_idx = index_list[int(bi)]
        stack_lookup: Dict[int, torch.Tensor] = {}
        if sample_stack is not None and sample_idx is not None:
            stack_t = sample_stack if torch.is_tensor(sample_stack) else torch.as_tensor(sample_stack, dtype=torch.float32)
            idx_t = sample_idx if torch.is_tensor(sample_idx) else torch.as_tensor(sample_idx, dtype=torch.long)
            stack_t = stack_t.to(device=xb.device, dtype=torch.float32)
            idx_t = idx_t.to(device=xb.device, dtype=torch.long).reshape(-1)
            if int(stack_t.ndim) == 2:
                stack_t = stack_t.unsqueeze(0)
            n_pairs = min(int(stack_t.shape[0]), int(idx_t.numel()))
            for si in range(int(n_pairs)):
                stack_lookup[int(idx_t[si].item())] = stack_t[si]
        for lbl in active.tolist():
            y_single = torch.zeros_like(y_row)
            y_single[int(lbl)] = 1.0
            mask_single = stack_lookup.get(int(lbl), mb[bi])
            if int(mask_single.ndim) == 2:
                mask_single = mask_single.unsqueeze(0)
            x_rows.append(xb[bi])
            y_rows.append(y_single)
            m_rows.append(mask_single.to(dtype=torch.float32))
    if len(x_rows) <= 0:
        return xb, yb, mb
    return torch.stack(x_rows, dim=0), torch.stack(y_rows, dim=0), torch.stack(m_rows, dim=0)


def _forward_classifier_outputs_require_mask(classifier: nn.Module, xb: torch.Tensor, context: str) -> Dict[str, torch.Tensor]:
    if hasattr(classifier, "forward_with_aux") and callable(getattr(classifier, "forward_with_aux")):
        out = classifier.forward_with_aux(xb)
    else:
        out = {"logits": classifier(xb)}
    logits = out.get("logits")
    if not isinstance(logits, torch.Tensor):
        raise RuntimeError(f"{str(context)} classifier forward did not return logits tensor.")
    mask_logits = out.get("mask_logits")
    if not isinstance(mask_logits, torch.Tensor):
        raise RuntimeError(f"{str(context)} requires a classifier mask head and mask logits output.")
    out["mask_logits"] = mask_logits
    return out


def _semantic_mask_bce_loss(mask_logits: torch.Tensor, mask_targets: torch.Tensor, context: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if int(mask_logits.ndim) == 3:
        mask_logits = mask_logits.unsqueeze(1)
    if int(mask_targets.ndim) == 3:
        mask_targets = mask_targets.unsqueeze(1)
    if int(mask_logits.ndim) != 4 or int(mask_targets.ndim) != 4:
        raise RuntimeError(
            f"{str(context)} mask tensors must be [B,1,H,W]; got logits={tuple(mask_logits.shape)} targets={tuple(mask_targets.shape)}"
        )
    if tuple(mask_logits.shape[-2:]) != tuple(mask_targets.shape[-2:]):
        mask_logits = F.interpolate(mask_logits, size=tuple(mask_targets.shape[-2:]), mode="bilinear", align_corners=False)
    if int(mask_logits.shape[1]) != int(mask_targets.shape[1]):
        if int(mask_logits.shape[1]) == 1 and int(mask_targets.shape[1]) > 1:
            mask_targets = mask_targets[:, :1, :, :]
        elif int(mask_targets.shape[1]) == 1 and int(mask_logits.shape[1]) > 1:
            mask_logits = mask_logits[:, :1, :, :]
        else:
            raise RuntimeError(
                f"{str(context)} mask channel mismatch: logits={int(mask_logits.shape[1])} targets={int(mask_targets.shape[1])}"
            )
    mask_targets = torch.clamp(mask_targets.to(dtype=torch.float32), 0.0, 1.0)
    return F.binary_cross_entropy_with_logits(mask_logits.to(dtype=torch.float32), mask_targets), mask_logits


@torch.no_grad()


def _filter_stream_pool_for_chunk_samples(
    streams: Sequence[np.ndarray],
    labels: Sequence[int],
    metas: Sequence[Dict],
    chunk_samples: int,
    pool_name: str,
    target_labels: Optional[Sequence[Any]] = None,
):
    n_need = max(1, int(chunk_samples))
    keep = [int(i) for i, s in enumerate(streams) if int(np.asarray(s).size) >= n_need]
    if len(keep) <= 0:
        raise RuntimeError(
            f"{pool_name}: no streams are long enough for chunk_samples={n_need}. "
            f"stream_count={len(streams)}. "
            "Provide longer source WAVs or increase --latent-fallback-seconds."
        )
    out_streams = [np.asarray(streams[i], dtype=np.float32) for i in keep]
    out_labels = [int(labels[i]) for i in keep]
    out_metas = [dict(metas[i]) for i in keep]
    out_targets = None
    if target_labels is not None:
        out_targets = []
        for i in keep:
            tv = target_labels[i]
            if tv is None:
                out_targets.append(None)
            else:
                arr = np.asarray(tv, dtype=np.float32).reshape(-1)
                out_targets.append(arr.astype(np.float32, copy=False))
    info = {
        "pool": str(pool_name),
        "required_chunk_samples": int(n_need),
        "before": int(len(streams)),
        "after": int(len(out_streams)),
        "dropped": int(len(streams) - len(out_streams)),
    }
    return out_streams, out_labels, out_metas, out_targets, info


def _expand_payload_conditions_with_semantic_bank(
    payload_conditions: Sequence[Any],
    condition_num_classes: int,
    supervised_num_classes: int,
    label_embedding_bank: Optional[np.ndarray],
    semantic_target_temperature: float = 8.0,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    c = max(1, int(condition_num_classes))
    n_rows = int(len(payload_conditions))
    if n_rows <= 0:
        return [], {"expanded": False, "reason": "empty_rows", "rows": 0, "condition_dim": int(c)}

    sup = max(1, min(int(supervised_num_classes), int(c)))
    base_rows: List[np.ndarray] = []
    sup_rows: List[np.ndarray] = []
    for i, row in enumerate(payload_conditions):
        arr = np.asarray(row, dtype=np.float32).reshape(-1)
        if int(arr.size) <= 0:
            raise RuntimeError(f"Semantic payload row {i} is empty.")
        if int(arr.size) > int(c):
            raise RuntimeError(
                "Semantic payload row width exceeds active semantic width; refusing truncation. "
                f"row={int(i)} row_dim={int(arr.size)} condition_dim={int(c)}"
            )

        if int(arr.size) >= int(c):
            arr_full = arr[: int(c)]
        else:
            pad = np.zeros((int(c) - int(arr.size),), dtype=np.float32)
            arr_full = np.concatenate([arr, pad], axis=0)
        if int(arr.size) >= int(sup):
            arr_sup = arr[: int(sup)]
        else:
            pad_sup = np.zeros((int(sup) - int(arr.size),), dtype=np.float32)
            arr_sup = np.concatenate([arr, pad_sup], axis=0)

        base_rows.append(np.clip(arr_full, 0.0, 1.0).astype(np.float32, copy=False))
        sup_rows.append(np.clip(arr_sup, 0.0, 1.0).astype(np.float32, copy=False))

    info: Dict[str, Any] = {
        "expanded": False,
        "reason": "",
        "rows": int(n_rows),
        "condition_dim": int(c),
        "supervised_dim": int(sup),
        "extra_dim": int(max(0, int(c) - int(sup))),
    }
    # Keep payload labels sparse and literal: preserve provided supervised slots and leave semantic extras at 0.
    # Semantic extras are added explicitly by term tags later in the pipeline.
    info["reason"] = "supervised_passthrough_full_width"
    info["expanded"] = bool(int(c) > int(sup))
    info["extra_mean"] = 0.0
    info["extra_max"] = 0.0
    return base_rows, info


def _payload_condition_bank_tensor(
    payload_targets: Sequence[Any],
    num_classes: int,
) -> torch.Tensor:
    n = int(len(payload_targets))
    c = max(1, int(num_classes))
    if n <= 0:
        return torch.zeros((0, c), dtype=torch.float32)
    bank = np.zeros((n, c), dtype=np.float32)
    for i in range(n):
        arr = np.asarray(payload_targets[i], dtype=np.float32).reshape(-1)
        if int(arr.size) <= 0:
            raise RuntimeError(f"Semantic condition row {i} is empty.")
        if int(arr.size) != int(c):
            raise RuntimeError(
                f"Semantic condition width mismatch at row {i}: got={int(arr.size)} expected={int(c)}"
            )
        bank[i, :] = arr
    return torch.from_numpy(bank.astype(np.float32, copy=False))


@torch.no_grad()


def _build_berkeley_payload_bank(
    data_root: str,
    image_size: int,
    auto_install_scipy: bool,
    max_samples: int,
    seed: int,
    source_root: str = "",
    force_cache_rebuild: bool = False,
):
    _ = bool(auto_install_scipy)
    _ = bool(force_cache_rebuild)
    root = Path(str(data_root).strip() or "toys_to_survive_development/data/berkeley_sbd")
    size = max(8, int(image_size))
    rows, rows_info = collect_semantic_disk_rows(
        data_root=str(data_root),
        class_names=_default_berkeley_class_names(),
        source_root=str(source_root),
    )
    if int(len(rows)) <= 0:
        return [], [], {"available": 0, "used": 0, "available_train": 0, "available_val": 0}, [], []

    labels_np = np.stack([np.asarray(r.label_vec, dtype=np.float32).reshape(-1) for r in rows], axis=0).astype(np.float32, copy=False)
    terms_rows = [list(_normalize_vocab_terms(r.terms)) for r in rows]
    source_rows = [str(r.source) for r in rows]
    class_names = _default_berkeley_class_names()
    n_classes = max(1, int(len(class_names)))
    class_lut = {re.sub(r"\s+", " ", str(name)).strip().lower(): int(i) for i, name in enumerate(class_names)}
    berkeley_dataset_idx = int(class_lut.get("berkeley sbd dataset", -1))
    n_total = int(len(rows))

    def _source_supervised_stats(source_key: str) -> Dict[str, float]:
        key = re.sub(r"\s+", " ", str(source_key)).strip().lower()
        idx_rows = [
            int(i)
            for i in range(int(n_total))
            if re.sub(r"\s+", " ", str(source_rows[int(i)])).strip().lower() == key
        ]
        if len(idx_rows) <= 0:
            return {"rows": 0, "nonzero": 0, "min": 0, "mean": 0.0, "max": 0}
        idx_np = np.asarray(idx_rows, dtype=np.int64)
        y_np = np.asarray(labels_np[idx_np], dtype=np.float32)
        if int(y_np.ndim) == 1:
            y_np = y_np.reshape(1, -1)
        sup_take = min(int(n_classes), int(y_np.shape[1]))
        if int(sup_take) <= 0:
            return {"rows": int(y_np.shape[0]), "nonzero": 0, "min": 0, "mean": 0.0, "max": 0}
        pos_counts = np.count_nonzero(np.asarray(y_np[:, : int(sup_take)], dtype=np.float32) >= 0.5, axis=1).astype(np.int32)
        return {
            "rows": int(pos_counts.size),
            "nonzero": int(np.count_nonzero(pos_counts > 0)),
            "min": int(pos_counts.min()) if int(pos_counts.size) > 0 else 0,
            "mean": float(np.mean(pos_counts)) if int(pos_counts.size) > 0 else 0.0,
            "max": int(pos_counts.max()) if int(pos_counts.size) > 0 else 0,
        }

    train_lbl_stats = _source_supervised_stats("berkeley_sbd_train")
    val_lbl_stats = _source_supervised_stats("berkeley_sbd_val")
    if int(train_lbl_stats.get("rows", 0)) > 0 and int(train_lbl_stats.get("nonzero", 0)) < int(train_lbl_stats.get("rows", 0)):
        raise RuntimeError(
            "Berkeley payload cache train rows are missing supervised targets. "
            f"nonzero={int(train_lbl_stats.get('nonzero', 0))}/{int(train_lbl_stats.get('rows', 0))}"
        )
    if int(val_lbl_stats.get("rows", 0)) > 0 and int(val_lbl_stats.get("nonzero", 0)) < int(val_lbl_stats.get("rows", 0)):
        raise RuntimeError(
            "Berkeley payload cache val rows are missing supervised targets. "
            f"nonzero={int(val_lbl_stats.get('nonzero', 0))}/{int(val_lbl_stats.get('rows', 0))}"
        )
    _log(
        "[berkeley-label-sanity] "
        "source=disk_dataset_rows "
        f"train={int(train_lbl_stats.get('nonzero', 0))}/{int(train_lbl_stats.get('rows', 0))} "
        f"train_min/mean/max={int(train_lbl_stats.get('min', 0))}/"
        f"{float(train_lbl_stats.get('mean', 0.0)):.2f}/{int(train_lbl_stats.get('max', 0))} "
        f"val={int(val_lbl_stats.get('nonzero', 0))}/{int(val_lbl_stats.get('rows', 0))} "
        f"val_min/mean/max={int(val_lbl_stats.get('min', 0))}/"
        f"{float(val_lbl_stats.get('mean', 0.0)):.2f}/{int(val_lbl_stats.get('max', 0))}"
    )

    take = int(n_total) if int(max_samples) <= 0 else min(int(n_total), int(max_samples))
    rng = np.random.default_rng(seed)
    if int(take) >= int(n_total):
        picks = np.arange(int(n_total), dtype=np.int64)
        rng.shuffle(picks)
    else:
        picks = rng.choice(np.arange(int(n_total), dtype=np.int64), size=int(take), replace=False).astype(np.int64)

    out_targets: List[np.ndarray] = []
    out_terms: List[List[str]] = []
    selected_rows: List[Any] = []
    used_train = 0
    used_val = 0
    used_external = 0
    used_source_counts: Dict[str, int] = {}
    for idx in picks.tolist():
        yv = np.asarray(labels_np[int(idx)], dtype=np.float32).reshape(-1)
        src = str(source_rows[int(idx)]) if int(idx) < int(len(source_rows)) else ""
        terms = (
            list(_normalize_vocab_terms(terms_rows[int(idx)]))
            if int(idx) < int(len(terms_rows)) and isinstance(terms_rows[int(idx)], list)
            else []
        )
        if len(terms) <= 0:
            raise RuntimeError(
                "Payload bank cache row is missing original terms; strict semantic pipeline requires "
                f"non-empty source terms. row_index={int(idx)} source={str(src)!r}"
            )
        terms_lc = {
            re.sub(r"\s+", " ", str(t)).strip().lower()
            for t in terms
            if re.sub(r"\s+", " ", str(t)).strip()
        }
        if "berkeley sbd dataset" not in terms_lc:
            raise RuntimeError(
                "Payload row missing required 'berkeley sbd dataset' term tag. "
                f"row_index={int(idx)} source={str(src)!r}"
            )
        if int(yv.size) <= int(berkeley_dataset_idx) or float(yv[int(berkeley_dataset_idx)]) < 0.5:
            raise RuntimeError(
                "Payload row missing required 'berkeley sbd dataset' target bit. "
                f"row_index={int(idx)} source={str(src)!r}"
            )
        out_targets.append(np.clip(yv, 0.0, 1.0).astype(np.float32, copy=False))
        out_terms.append(list(terms))
        selected_rows.append(rows[int(idx)])
        used_source_counts[str(src)] = int(used_source_counts.get(str(src), 0)) + 1
        if str(src).lower() == "berkeley_sbd_train":
            used_train += 1
        elif str(src).lower() == "berkeley_sbd_val":
            used_val += 1
        else:
            used_external += 1

    payload_bank = _LazyDiskSemanticPayloadBank(
        rows=selected_rows,
        image_size=int(size),
        seed=int(seed),
    )
    out_images = _LazyDiskSemanticPayloadView(payload_bank, kind="image")
    out_masks = _LazyDiskSemanticPayloadView(payload_bank, kind="mask")

    info = {
        "available": int(n_total),
        "used": int(len(selected_rows)),
        "available_train": int(rows_info.get("available_train", 0)),
        "available_val": int(rows_info.get("available_val", 0)),
        "available_external": int(rows_info.get("available_external", 0)),
        "used_train": int(used_train),
        "used_val": int(used_val),
        "used_external": int(used_external),
        "cache_dir": str(root / "cache"),
        "cache_manifest": "",
        "cache_hit": False,
        "source_catalog": "",
        "used_source_counts": {str(k): int(v) for k, v in used_source_counts.items()},
        "mask_cache_hits": int(rows_info.get("mask_cache_hits", 0)),
        "mask_cache_writes": int(rows_info.get("mask_cache_writes", 0)),
        "mask_cache_failures": int(rows_info.get("mask_cache_failures", 0)),
        "mask_cache_seconds": float(rows_info.get("mask_cache_seconds", 0.0)),
        "mask_cache_threads": int(rows_info.get("mask_cache_threads", 0)),
        "row_build_threads": int(rows_info.get("row_build_threads", 0)),
        "row_build_seconds": float(rows_info.get("row_build_seconds", 0.0)),
        "inprocess_cache_hit": bool(rows_info.get("inprocess_cache_hit", False)),
    }
    return out_images, out_targets, info, out_terms, out_masks
