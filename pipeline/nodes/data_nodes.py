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
        from wav_config_transformer_pipeline import _filter_stream_pool_for_chunk_samples
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
        from wav_config_transformer_pipeline import (
            _build_berkeley_payload_bank,
            _expand_payload_conditions_with_semantic_bank,
            _payload_condition_bank_tensor,
        )

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
        from wav_config_transformer_pipeline import (
            _build_berkeley_refresh_loader,
            _build_berkeley_gate_val_loader,
            _build_berkeley_refresh_cache,
        )

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
        from wav_config_transformer_pipeline import (
            _build_payload_validation_gate_dataset,
            _build_gate_loader_from_dataset,
        )

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
