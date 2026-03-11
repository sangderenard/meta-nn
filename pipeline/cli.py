"""
CLI argument parsing for the nodus training pipeline.

Extracted from wav_config_transformer_pipeline.py to keep the monolith slim
and allow the standalone graph runner (pipeline_runner.py) to share the
same argument definitions.
"""
from __future__ import annotations

import argparse

from wav_ml_core import COLOR_MODE_MAP, COLOR_MODES, RenderConfig


def _build_enforced_render_config(args) -> RenderConfig:
    color_mode = str(args.enforce_render_color_mode)
    if color_mode not in COLOR_MODE_MAP:
        color_mode = COLOR_MODES[0][0]
    return RenderConfig(
        bitmode=str(args.enforce_render_bitmode),
        use_channels=str(args.enforce_render_use_channels),
        mono_pick=str(args.enforce_render_mono_pick),
        width=max(1, int(args.enforce_render_width)),
        downsample=max(1, int(args.enforce_render_downsample)),
        color_mode=color_mode,
        empty_fill=max(0, min(255, int(args.enforce_render_empty_fill))),
        bitmask_enable=bool(args.enforce_render_bitmask_enable),
        bitmask_low=int(args.enforce_render_bitmask_low),
        bitmask_high=int(args.enforce_render_bitmask_high),
        max_points=max(1024, int(args.enforce_render_max_points)),
    )




def parse_args():
    p = argparse.ArgumentParser(description="Config-search + classifier + waveform transformer training pipeline.")
    p.add_argument(
        "--wav-root",
        default="",
        help="Directory to recursively scan for .wav files. If omitted, latent fallback pool is generated.",
    )
    p.add_argument("--output-dir", default="toys_to_survive_development/wav_pipeline_runs/latest")
    p.add_argument(
        "--hard-wipe-caches",
        "--hard-wipe-cache",
        dest="hard_wipe_caches",
        action="store_true",
        help=(
            "Delete generated wave libraries and Berkeley payload-bank caches before startup "
            "(accepted_wave_library, latent_wave_pool, training_supervision, cache/payload_bank_rgb*)."
        ),
    )
    p.add_argument(
        "--soft-reset-labels",
        "--soft-reset-label-cache",
        dest="soft_reset_labels",
        action="store_true",
        help=(
            "Reset label/semantic cache artifacts while keeping cached images "
            "(payload labels/terms/sources/manifest/source_catalog and semantic_gate_labels)."
        ),
    )
    p.add_argument("--max-files", type=int, default=300)
    p.add_argument("--latent-fallback-count", type=int, default=192)
    p.add_argument("--latent-fallback-seconds", type=float, default=2.0)
    p.add_argument("--latent-fallback-rate", type=int, default=16000)
    p.add_argument("--latent-noise-std", type=float, default=0.20)
    p.add_argument(
        "--latent-reinject-dir",
        default="",
        help="Optional accepted-wave library dir to reinject from (expects index.jsonl). Defaults to <output-dir>/accepted_wave_library.",
    )
    p.add_argument("--latent-reinject-ratio", type=float, default=0.50)
    p.add_argument("--latent-reinject-copy-gain", type=float, default=0.70)
    p.add_argument("--latent-reinject-noise-gain", type=float, default=0.35)
    p.add_argument("--latent-structured-ratio", type=float, default=0.85)
    p.add_argument("--latent-structured-gain", type=float, default=0.80)
    p.add_argument("--latent-structured-noise-gain", type=float, default=0.20)
    p.add_argument(
        "--latent-berkeley-imprint-mix-targets",
        dest="latent_berkeley_imprint_mix_targets",
        action="store_true",
        help="When latent fallback is active in berkeley mode, write deterministic Berkeley image payloads into mix-stream target bits.",
    )
    p.add_argument(
        "--no-latent-berkeley-imprint-mix-targets",
        dest="latent_berkeley_imprint_mix_targets",
        action="store_false",
    )
    p.add_argument("--latent-berkeley-imprint-steps", type=int, default=12)
    p.add_argument("--latent-berkeley-imprint-lr", type=float, default=0.08)
    p.add_argument("--latent-berkeley-imprint-l2", type=float, default=0.02)
    p.add_argument("--latent-berkeley-imprint-min-target-classes", type=int, default=2)
    p.add_argument("--latent-berkeley-imprint-max-target-classes", type=int, default=4)
    p.add_argument(
        "--resume-from",
        default="",
        help="Optional prior run directory. If set, load checkpoints/config/history from there.",
    )
    p.add_argument(
        "--auto-resume",
        action="store_true",
        help="Auto-resume from current output-dir artifacts when present.",
    )
    p.add_argument(
        "--force-config-search",
        action="store_true",
        help="When resuming, ignore prior best config and run config search again.",
    )
    p.add_argument(
        "--checkpoint-every-round",
        type=int,
        default=1,
        help="Save pipeline checkpoint every N orchestration rounds (0 disables periodic snapshots).",
    )
    p.add_argument(
        "--checkpoint-after-training-segment",
        dest="checkpoint_after_training_segment",
        action="store_true",
        help="Save pipeline/model checkpoints immediately after each completed training segment.",
    )
    p.add_argument(
        "--no-checkpoint-after-training-segment",
        dest="checkpoint_after_training_segment",
        action="store_false",
    )
    p.add_argument(
        "--objective-mode",
        choices=["berkeley_multilabel", "wave_labels"],
        default="berkeley_multilabel",
        help="Search/optimize objective source. 'berkeley_multilabel' is the intended feature-presence metric mode.",
    )
    p.add_argument("--label-mode", choices=["folder", "spectral"], default="folder")
    p.add_argument("--pseudo-classes", type=int, default=4)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--amp", action="store_true", help="Enable autocast mixed precision on CUDA.")
    p.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--channels-last", action="store_true", help="Use NHWC memory format for image tensors/models.")
    p.add_argument("--compile-models", action="store_true", help="Use torch.compile for classifier/transformer models.")
    p.add_argument("--compile-mode", default="default")
    p.add_argument("--grad-accum-steps", type=int, default=1, help="Micro-batch accumulation factor.")
    p.add_argument("--classifier-grad-clip", type=float, default=1.0, help="Gradient clip norm for classifier optimizer steps.")
    p.add_argument("--classifier-semantic-soft-target-max", type=float, default=0.0, help="Max soft target for negative classes in classifier BCE, set by cosine similarity to active label centroid.")
    p.add_argument("--classifier-semantic-cosine-weight", type=float, default=0.35, help="Weight for semantic cosine embedding loss in classifier supervision.")
    p.add_argument("--generator-grad-clip", type=float, default=1.0, help="Gradient clip norm for generator optimizer steps.")
    p.add_argument("--discriminator-grad-clip", type=float, default=1.0, help="Gradient clip norm for discriminator optimizer steps.")
    p.add_argument("--transformer-grad-clip", type=float, default=1.0, help="Gradient clip norm for transformer optimizer steps.")
    p.add_argument("--pin-memory-wave-batches", action="store_true", help="Pin sampled wave batches before H2D copy.")
    p.add_argument("--cache-wave-cls-dataset-on-device", action="store_true", help="Cache rendered wave-classifier datasets on GPU.")
    p.add_argument("--cache-wave-streams-on-device", action="store_true", help="Cache waveform stream pools on GPU for vectorized batch sampling.")
    p.add_argument(
        "--stage-module-offload",
        dest="stage_module_offload",
        action="store_true",
        help="Move inactive stage modules (generator/discriminator/wave-classifier) to CPU between stages to free VRAM.",
    )
    p.add_argument("--no-stage-module-offload", dest="stage_module_offload", action="store_false")
    p.add_argument(
        "--stage-module-offload-empty-cache",
        dest="stage_module_offload_empty_cache",
        action="store_true",
        help="Run gc.collect()+torch.cuda.empty_cache() after staged CPU offload transitions.",
    )
    p.add_argument("--no-stage-module-offload-empty-cache", dest="stage_module_offload_empty_cache", action="store_false")
    p.add_argument("--no-cudnn-benchmark", dest="cudnn_benchmark", action="store_false")
    p.add_argument("--no-tf32", dest="allow_tf32", action="store_false")
    p.add_argument("--matmul-precision", choices=["high", "medium", "highest"], default="high")

    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--render-max-points", type=int, default=262144)
    p.add_argument(
        "--enforce-render-config",
        action="store_true",
        help="Bypass render config search/resume and use the fixed render config arguments below.",
    )
    p.add_argument("--enforce-render-bitmode", default="auto")
    p.add_argument("--enforce-render-use-channels", choices=["auto", "1", "2"], default="auto")
    p.add_argument("--enforce-render-mono-pick", choices=["Mix", "L", "R", "LR stride"], default="Mix")
    p.add_argument("--enforce-render-width", type=int, default=512)
    p.add_argument("--enforce-render-downsample", type=int, default=1)
    p.add_argument("--enforce-render-color-mode", default=COLOR_MODES[0][0], choices=[m[0] for m in COLOR_MODES])
    p.add_argument("--enforce-render-empty-fill", type=int, default=0)
    p.add_argument("--enforce-render-bitmask-enable", dest="enforce_render_bitmask_enable", action="store_true")
    p.add_argument("--no-enforce-render-bitmask-enable", dest="enforce_render_bitmask_enable", action="store_false")
    p.add_argument("--enforce-render-bitmask-low", type=int, default=0)
    p.add_argument("--enforce-render-bitmask-high", type=int, default=7)
    p.add_argument("--enforce-render-max-points", type=int, default=262144)

    p.add_argument("--config-trials", type=int, default=10)
    p.add_argument("--config-epochs", type=int, default=2)
    p.add_argument("--search-train-limit", type=int, default=96)
    p.add_argument("--search-val-limit", type=int, default=48)
    p.add_argument("--try-scipy-refine", action="store_true")
    p.add_argument("--score-topk", type=int, default=3)
    p.add_argument("--score-threshold", type=float, default=0.35)
    p.add_argument("--score-w-topk", type=float, default=0.60)
    p.add_argument("--score-w-cov", type=float, default=0.30)
    p.add_argument("--score-w-mean", type=float, default=0.10)

    p.add_argument("--classifier-epochs", type=int, default=6)
    p.add_argument("--classifier-batch-size", type=int, default=32)
    p.add_argument("--classifier-lr", type=float, default=2e-3)
    p.add_argument("--classifier-base-ch", type=int, default=64, help="Base channel width for image classifiers.")
    p.add_argument("--classifier-max-ch", type=int, default=384, help="Max channel cap for image classifiers.")
    p.add_argument("--classifier-context-blocks", type=int, default=8, help="Residual context blocks in image classifiers.")
    p.add_argument("--classifier-context-dropout", type=float, default=0.05, help="Dropout in classifier context blocks.")
    p.add_argument(
        "--lr-sine-cycles",
        type=float,
        default=1.0,
        help="Sinusoidal LR cycles. Set <=0 (with --lr-sine-frequency <=0) to keep constant base LR.",
    )
    p.add_argument("--lr-sine-frequency", type=float, default=0.0, help="Cycles per optimizer step; overrides --lr-sine-cycles when > 0.")
    p.add_argument("--lr-sine-tail-fraction", type=float, default=0.15)
    p.add_argument("--lr-sine-min-scale", type=float, default=0.0)
    p.add_argument(
        "--classifier-init-ckpt",
        default="",
        help="Optional pretrained classifier checkpoint (for warm-starting wave classifier).",
    )
    p.add_argument(
        "--classifier-init-required",
        dest="classifier_init_required",
        action="store_true",
        help=(
            "Require --classifier-init-ckpt to exist/load. "
            "When disabled, Berkeley objective can bootstrap classifier from scratch if checkpoint is absent."
        ),
    )
    p.add_argument("--no-classifier-init-required", dest="classifier_init_required", action="store_false")
    p.add_argument(
        "--classifier-init-scope",
        choices=["features", "all"],
        default="features",
        help="When loading init checkpoint, load only feature extractor or all matching keys.",
    )
    p.add_argument(
        "--classifier-freeze-features",
        action="store_true",
        help="Freeze classifier feature extractor during wave-domain classifier training.",
    )
    p.add_argument(
        "--label-embeddings",
        dest="label_embeddings_enabled",
        action="store_true",
        help="Use text-embedding label bank for classifier logits (replaces direct class-weight logits).",
    )
    p.add_argument("--no-label-embeddings", dest="label_embeddings_enabled", action="store_false")
    p.add_argument(
        "--label-embedding-backend",
        choices=["sentence_transformers"],
        default="sentence_transformers",
        help="Text embedding backend for label bank. Only sentence-transformers is supported.",
    )
    p.add_argument(
        "--label-embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformers model id used when backend is sentence_transformers.",
    )
    p.add_argument(
        "--label-embedding-dim",
        type=int,
        default=384,
        help="Label embedding size. Must match sentence-transformers model native dimension.",
    )
    p.add_argument("--label-embedding-temperature", type=float, default=10.0)
    p.add_argument(
        "--label-text-json",
        default="",
        help="Optional JSON override for class label text (list aligned by index or dict by class name/index).",
    )
    p.add_argument(
        "--label-query-texts",
        default="",
        help="Optional query text list (split by | ; , or newline) for nearest-label retrieval report.",
    )
    p.add_argument("--label-query-topk", type=int, default=5)
    p.add_argument(
        "--semantic-vocab-extra-texts",
        default="",
        help="Optional extra semantic vocabulary terms (split by | ; , or newline) appended to supervised labels.",
    )
    p.add_argument(
        "--semantic-vocab-extra-json",
        default="",
        help="Optional JSON file containing extra semantic vocabulary terms (list[str] or dict with list field).",
    )
    p.add_argument(
        "--semantic-vocab-extra-slots",
        type=int,
        default=50,
        help="Reserve this many churn-local semantic slots beyond the invariant supervised vocabulary.",
    )
    p.add_argument(
        "--semantic-vocab-churn-enabled",
        dest="semantic_vocab_churn_enabled",
        action="store_true",
        help="Rotate active extra semantic terms each cycle while keeping Berkeley supervised classes fixed.",
    )
    p.add_argument("--no-semantic-vocab-churn-enabled", dest="semantic_vocab_churn_enabled", action="store_false")
    p.add_argument(
        "--semantic-vocab-churn-replace-per-cycle",
        type=int,
        default=1,
        help="How many active extra semantic slots to replace each cycle when churn is enabled.",
    )
    p.add_argument(
        "--semantic-vocab-churn-sweep-cycles",
        type=int,
        default=0,
        help=(
            "If >0, auto-raise per-cycle churn replacement so the currently available non-active churn terms "
            "are swept through in roughly this many cycles (bounded by unlocked extra slots)."
        ),
    )
    p.add_argument(
        "--semantic-vocab-regurgitated-churn-enabled",
        dest="semantic_vocab_regurgitated_churn_enabled",
        action="store_true",
        help=(
            "When accepted/regurgitated waves are reingested, enqueue their semantic tags into a "
            "lifetime-limited churn reservoir used as an extra candidate source during semantic churn."
        ),
    )
    p.add_argument(
        "--no-semantic-vocab-regurgitated-churn-enabled",
        dest="semantic_vocab_regurgitated_churn_enabled",
        action="store_false",
    )
    p.add_argument(
        "--semantic-vocab-regurgitated-churn-prob",
        type=float,
        default=0.35,
        help="Probability per cycle that the regurgitated churn reservoir is sampled.",
    )
    p.add_argument(
        "--semantic-vocab-regurgitated-churn-lifetime",
        type=int,
        default=3,
        help="How many churn uses each regurgitated reservoir item survives before expiry.",
    )
    p.add_argument(
        "--semantic-vocab-regurgitated-churn-max-per-cycle",
        type=int,
        default=4,
        help="Maximum regurgitated reservoir items consumed (and lifetime-decremented) per churn cycle.",
    )
    p.add_argument(
        "--semantic-vocab-regurgitated-churn-capacity",
        type=int,
        default=512,
        help="Maximum number of regurgitated reservoir items retained.",
    )
    p.add_argument(
        "--semantic-vocab-deferred-churn-prob",
        type=float,
        default=0.35,
        help=(
            "Per-cycle probability that deferred non-basic terms (e.g., Stage-2 deferrals) are used as a churn "
            "source when selecting next active extras."
        ),
    )
    p.add_argument(
        "--semantic-vocab-deferred-churn-max-per-cycle",
        type=int,
        default=8,
        help="Maximum deferred terms consumed from the churn queue per cycle when deferred source is selected.",
    )
    p.add_argument(
        "--semantic-vocab-text-condition-temperature",
        type=float,
        default=8.0,
        help="Temperature used when projecting text-term embeddings into semantic condition vectors.",
    )
    p.add_argument(
        "--semantic-vocab-text-condition-topk",
        type=int,
        default=0,
        help="Semantic-dimension pruning is disabled in strict mode; must be set to 0.",
    )
    p.add_argument(
        "--semantic-vocab-unknown-label-rows-per-cycle",
        type=int,
        default=0,
        help=(
            "Out-of-vocabulary symbol terms are deferred into churn candidates; this controls how many are sampled "
            "per cycle for churn-queueing (not direct classifier payload admission)."
        ),
    )
    p.add_argument(
        "--semantic-vocab-bootstrap-origin-label",
        default="internal bootstrap root vocab",
        help="Origin label text injected for bootstrap/unknown symbol conditioning vectors.",
    )
    p.add_argument(
        "--semantic-vocab-auto-symbol-pool",
        dest="semantic_vocab_auto_symbol_pool",
        action="store_true",
        help=(
            "Build semantic symbol pool from official datasets (MNIST/EMNIST/KMNIST) plus synthetic bootstrap semantic terms; "
            "used for churn slots and payload augmentation."
        ),
    )
    p.add_argument("--no-semantic-vocab-auto-symbol-pool", dest="semantic_vocab_auto_symbol_pool", action="store_false")
    p.add_argument(
        "--semantic-vocab-symbol-pool-root",
        default="toys_to_survive_development/data/semantic_symbol_pool",
        help="Local cache directory for auto-downloaded symbol datasets used by semantic churn.",
    )
    p.add_argument(
        "--semantic-vocab-symbol-samples-per-term",
        type=int,
        default=1,
        help="Max symbol examples retained per auto term (digit/letter/pictogram).",
    )
    p.add_argument(
        "--semantic-vocab-gestation-train-target-samples",
        type=int,
        default=0,
        help=(
            "Target Stage-1 gestation training rows after bootstrap augmentation expansion. "
            "0 disables expansion (use raw gestation split only)."
        ),
    )
    p.add_argument(
        "--semantic-vocab-gestation-val-target-samples",
        type=int,
        default=0,
        help=(
            "Target Gate-1 gestation validation rows after bootstrap augmentation expansion. "
            "0 uses max(raw_val_rows, round(train_target*0.2))."
        ),
    )
    p.add_argument(
        "--semantic-vocab-pregestation-enabled",
        dest="semantic_vocab_pregestation_enabled",
        action="store_true",
        help="Run a pre-gestation formal-logic stage with color+direction synthetic scenes before gestation refresh.",
    )
    p.add_argument("--no-semantic-vocab-pregestation-enabled", dest="semantic_vocab_pregestation_enabled", action="store_false")
    p.add_argument(
        "--semantic-vocab-pregestation-samples-per-combo",
        type=int,
        default=4,
        help="How many formal color+direction logic scenes to synthesize per color/direction combination for pre-gestation.",
    )
    p.add_argument(
        "--semantic-vocab-symbol-include-pictograms",
        dest="semantic_vocab_symbol_include_pictograms",
        action="store_true",
        help="Include KMNIST pictogram terms in the auto symbol pool when available.",
    )
    p.add_argument(
        "--no-semantic-vocab-symbol-include-pictograms",
        dest="semantic_vocab_symbol_include_pictograms",
        action="store_false",
    )
    p.add_argument(
        "--semantic-vocab-reference-flashcards",
        dest="semantic_vocab_reference_flashcards",
        action="store_true",
        help=(
            "Append synthetic/reference semantic flashcard rows to payload conditioning each cycle. "
            "Flashcards cover currently active semantic terms and blend Berkeley object seeds with symbol/damage variants."
        ),
    )
    p.add_argument(
        "--no-semantic-vocab-reference-flashcards",
        dest="semantic_vocab_reference_flashcards",
        action="store_false",
    )
    p.add_argument(
        "--semantic-vocab-reference-flashcards-per-term",
        type=int,
        default=2,
        help="How many reference flashcard rows to synthesize per active semantic term.",
    )
    p.add_argument(
        "--semantic-stage-cache-enabled",
        dest="semantic_stage_cache_enabled",
        action="store_true",
        help=(
            "Persist pre-generated semantic stage rows to disk and reuse them on later boots before falling back "
            "to live synthesis/augmentation."
        ),
    )
    p.add_argument("--no-semantic-stage-cache-enabled", dest="semantic_stage_cache_enabled", action="store_false")
    p.add_argument(
        "--semantic-stage-cache-dir",
        default="",
        help="Optional root directory for semantic stage caches (default: <output-dir>/semantic_stage_cache).",
    )
    p.add_argument(
        "--semantic-stage-cache-max-rows",
        type=int,
        default=512,
        help=(
            "Maximum number of pre-generated rows to cache per semantic stage dataset. "
            "0 means cache the full dataset length for that stage."
        ),
    )
    p.add_argument(
        "--semantic-stage-cache-max-mb",
        type=int,
        default=0,
        help=(
            "Hard disk cap in MB for each semantic stage cache pool. "
            "0 disables disk-cap enforcement for that stage cache."
        ),
    )
    p.add_argument(
        "--pregestation-stage-cache-max-mb",
        type=int,
        default=-1,
        help=(
            "Hard disk cap in MB for the pregestation stage cache pool. "
            "Negative values inherit --semantic-stage-cache-max-mb; 0 disables the pregestation disk cap."
        ),
    )
    p.add_argument(
        "--gestation-stage-cache-max-mb",
        type=int,
        default=-1,
        help=(
            "Hard disk cap in MB for each gestation stage cache pool. "
            "Negative values inherit --semantic-stage-cache-max-mb; 0 disables the gestation disk cap."
        ),
    )
    p.add_argument(
        "--semantic-stage-cache-overflow-strategy",
        choices=["loop", "evict"],
        default="loop",
        help="When the semantic stage cache hits its disk cap, either reuse bounded loop slots or evict older cache entries.",
    )
    p.add_argument(
        "--semantic-stage-cache-slot-lifespan",
        type=int,
        default=0,
        dest="semantic_stage_cache_slot_lifespan",
        help=(
            "Maximum number of times a loop-pool slot may be reused before it is treated as expired and rebuilt. "
            "0 (default) disables expiry ΓÇö slots are recycled indefinitely."
        ),
    )
    p.add_argument(
        "--semantic-stage-cache-rebuild",
        dest="semantic_stage_cache_rebuild",
        action="store_true",
        help="Force rebuild of semantic stage disk caches for this run.",
    )
    p.add_argument("--no-semantic-stage-cache-rebuild", dest="semantic_stage_cache_rebuild", action="store_false")
    p.add_argument(
        "--gd-vocab-library-enabled",
        dest="gd_vocab_library_enabled",
        action="store_true",
        help="Persist and reuse generator/discriminator checkpoints keyed by semantic vocabulary hash.",
    )
    p.add_argument("--no-gd-vocab-library-enabled", dest="gd_vocab_library_enabled", action="store_false")
    p.add_argument(
        "--gd-vocab-library-dir",
        default="",
        help="Optional directory for hashed G/D vocabulary snapshots (default: <output-dir>/gd_vocab_library).",
    )
    p.add_argument(
        "--gd-vocab-library-autoload",
        dest="gd_vocab_library_autoload",
        action="store_true",
        help="Attempt to load matching hashed G/D snapshot when active semantic vocabulary changes.",
    )
    p.add_argument("--no-gd-vocab-library-autoload", dest="gd_vocab_library_autoload", action="store_false")
    p.add_argument("--berkeley-data-root", default="toys_to_survive_development/data/berkeley_sbd")
    p.add_argument(
        "--berkeley-image-size",
        type=int,
        default=0,
        help=(
            "0 means reuse --image-size; when >0, this value becomes the shared embed resolution "
            "for Berkeley + transformer/classifier rendering."
        ),
    )
    p.add_argument(
        "--berkeley-payload-max-samples",
        type=int,
        default=1024,
        help=(
            "Cap Berkeley payload-bank samples used by generator stages. "
            "Set 0 to use the legacy auto policy."
        ),
    )
    p.add_argument(
        "--berkeley-payload-source-root",
        default="",
        help=(
            "Optional folder-drop source root for payload ingestion. "
            "Each direct subfolder is treated as a dataset source label; defaults to "
            "<berkeley-data-root>/payload_sources."
        ),
    )
    p.add_argument(
        "--berkeley-payload-cache-rebuild",
        dest="berkeley_payload_cache_rebuild",
        action="store_true",
        help="Force rebuild of disk payload cache from Berkeley + payload source folders.",
    )
    p.add_argument("--no-berkeley-payload-cache-rebuild", dest="berkeley_payload_cache_rebuild", action="store_false")
    p.add_argument("--berkeley-auto-install-scipy", action="store_true")
    p.add_argument("--berkeley-refresh-every", type=int, default=0, help="Run Berkeley refresh every N config trials; 0=off.")
    p.add_argument(
        "--berkeley-refresh-round-every",
        type=int,
        default=0,
        help="In orchestration mode, run Berkeley refresh every N rounds; 0=off.",
    )
    p.add_argument("--berkeley-refresh-epochs", type=int, default=1)
    p.add_argument("--berkeley-refresh-lr", type=float, default=2e-4)
    p.add_argument("--berkeley-refresh-weight-decay", type=float, default=1e-4)
    p.add_argument("--berkeley-refresh-batch-size", type=int, default=32, help="Refresh train batch size. Set 0 to auto-size from GPU free memory when cache is enabled.")
    p.add_argument("--berkeley-refresh-loader-batch-size", type=int, default=128, help="Batch size used only while building Berkeley refresh cache/loader input queue.")
    p.add_argument("--berkeley-refresh-workers", type=int, default=2)
    p.add_argument("--berkeley-refresh-log-every", type=int, default=0, help="0 disables periodic Berkeley-refresh progress logs.")
    p.add_argument("--berkeley-refresh-max-seconds", type=float, default=0.0, help="0 disables wall-time cap per refresh call.")
    p.add_argument(
        "--semantic-mask-supervision-mode",
        choices=["multihot_mix", "single_label_passes"],
        default="multihot_mix",
        help="Use multihot mixed-mask supervision or expand active labels into single-label mask-supervised passes.",
    )
    p.add_argument(
        "--berkeley-refresh-cache-batches",
        type=int,
        default=0,
        help="If >0, pre-cache this many Berkeley refresh batches for repeated GPU-fed refresh.",
    )
    p.add_argument(
        "--berkeley-refresh-cache-device",
        choices=["none", "auto", "cpu", "cuda"],
        default="none",
        help="Storage device for cached Berkeley refresh pool.",
    )
    p.add_argument("--berkeley-refresh-vram-fraction", type=float, default=0.90, help="Target fraction of free VRAM to fill for auto refresh batch.")
    p.add_argument("--berkeley-refresh-activation-mult", type=float, default=10.0, help="Activation overhead multiplier used for auto refresh batch sizing.")
    p.add_argument("--berkeley-refresh-max-batch-cap", type=int, default=0, help="Hard cap for auto refresh batch size (0 = no cap).")
    p.add_argument("--stage-c-lora-enabled", dest="stage_c_lora_enabled", action="store_true", help="Enable Stage C LoRA-only classifier refresh on churn rows.")
    p.add_argument("--no-stage-c-lora-enabled", dest="stage_c_lora_enabled", action="store_false")
    p.add_argument("--stage-c-lora-rank", type=int, default=8, help="LoRA rank used for Stage C classifier overlays.")
    p.add_argument("--stage-c-lora-alpha", type=float, default=16.0, help="LoRA alpha scaling used for Stage C classifier overlays.")
    p.add_argument("--stage-c-lora-max-terms", type=int, default=50, help="Maximum churn terms assigned to a single Stage C LoRA slot before scheduler switches slots.")
    p.add_argument("--loader-persistent-workers", dest="loader_persistent_workers", action="store_true")
    p.add_argument("--no-loader-persistent-workers", dest="loader_persistent_workers", action="store_false")
    p.add_argument("--loader-prefetch-factor", type=int, default=4)
    p.add_argument("--berkeley-refresh-max-train", type=int, default=0, help="0 means full Berkeley train set.")
    p.add_argument("--berkeley-refresh-max-steps", type=int, default=0, help="0 means full epoch over refresh loader.")
    p.add_argument("--final-train-limit", type=int, default=0, help="0 means all.")
    p.add_argument("--final-val-limit", type=int, default=0, help="0 means all.")
    p.add_argument(
        "--classifier-subset-fresh-round-sampling",
        dest="classifier_subset_fresh_round_sampling",
        action="store_true",
        help="When --final-val-limit selects a subset, resample that subset for each round-level classifier metric pass.",
    )
    p.add_argument(
        "--no-classifier-subset-fresh-round-sampling",
        dest="classifier_subset_fresh_round_sampling",
        action="store_false",
    )

    p.add_argument("--transformer-epochs", type=int, default=6)
    p.add_argument("--transformer-steps", type=int, default=80)
    p.add_argument("--transformer-batch-size", type=int, default=16)
    p.add_argument(
        "--transformer-round-batch-size",
        type=int,
        default=0,
        help="If >0, batch size override for orchestration R-stage transformer updates (0 = use --transformer-batch-size).",
    )
    p.add_argument(
        "--joint-transformer-batch-size",
        type=int,
        default=0,
        help=(
            "If >0, batch size override for orchestration J-stage transformer updates. "
            "0 auto-resolves from R-stage batch with a conservative memory cap."
        ),
    )
    p.add_argument(
        "--transformer-eval-batch-size",
        type=int,
        default=0,
        help="If >0, batch size used for transformer stage evaluations/gates (0 = auto from active train batch).",
    )
    p.add_argument("--transformer-d-model", type=int, default=256)
    p.add_argument("--transformer-nhead", type=int, default=8)
    p.add_argument("--transformer-num-layers", type=int, default=8)
    p.add_argument("--transformer-ff-mult", type=int, default=4)
    p.add_argument("--transformer-dropout", type=float, default=0.10)
    p.add_argument(
        "--transformer-filter-bundles",
        type=str,
        default="deskew",
        help="Comma-separated pre/post filter bundle names wrapped around transformer (use 'none' to disable).",
    )
    p.add_argument(
        "--transformer-deskew-prefilter-max-skew",
        type=float,
        default=0.25,
        help="Max absolute skew predicted/applied by the deskew prefilter bundle.",
    )
    p.add_argument(
        "--transformer-deskew-prefilter-weight",
        type=float,
        default=0.20,
        help="Loss weight for deskew prefilter skew prediction.",
    )
    p.add_argument(
        "--transformer-deskew-residual-weight",
        type=float,
        default=0.10,
        help="Loss weight for deskew post-transform residual observer.",
    )
    p.add_argument(
        "--transformer-deskew-pre-token-mix",
        type=float,
        default=0.08,
        help="Bounded additive mix for deskew prefilter token residual injection into transformer input tokens.",
    )
    p.add_argument(
        "--transformer-deskew-post-token-mix",
        type=float,
        default=0.08,
        help="Bounded additive mix for deskew postfilter token residual injection before transformer patch_out.",
    )
    p.add_argument(
        "--transformer-deskew-token-residual-weight",
        type=float,
        default=0.01,
        help="L2 regularization weight for deskew token residual injections.",
    )
    p.add_argument("--transformer-log-every", type=int, default=0, help="If >0, print transformer stage timing/progress every N train steps.")
    p.add_argument(
        "--transformer-visualize-status",
        dest="transformer_visualize_status",
        action="store_true",
        help="Open pygame OpenGL viewer and blit transformer input/output images at each status report.",
    )
    p.add_argument("--no-transformer-visualize-status", dest="transformer_visualize_status", action="store_false")
    p.add_argument("--transformer-viz-scale", type=int, default=3, help="Display scale multiplier for status viewer.")
    p.add_argument("--transformer-cache-eval-batches", dest="transformer_cache_eval_batches", action="store_true")
    p.add_argument("--no-transformer-cache-eval-batches", dest="transformer_cache_eval_batches", action="store_false")
    p.add_argument("--transformer-degrade-inputs", dest="transformer_degrade_inputs", action="store_true")
    p.add_argument("--no-transformer-degrade-inputs", dest="transformer_degrade_inputs", action="store_false")
    p.add_argument("--transformer-degrade-min-strength", type=float, default=0.25)
    p.add_argument("--transformer-degrade-max-strength", type=float, default=0.95)
    p.add_argument("--transformer-degrade-noise-std-min", type=float, default=0.01)
    p.add_argument("--transformer-degrade-noise-std-max", type=float, default=0.14)
    p.add_argument("--transformer-degrade-dropout-max", type=float, default=0.25)
    p.add_argument("--transformer-degrade-quant-bits-min", type=int, default=3)
    p.add_argument("--transformer-degrade-quant-bits-max", type=int, default=10)
    p.add_argument(
        "--transformer-degrade-mode",
        choices=["full", "blur_stride"],
        default="full",
        help="Degradation pipeline mode. Use blur_stride to keep only blur + stride-skew effects.",
    )
    p.add_argument(
        "--transformer-degrade-domain",
        choices=["waveform", "bitwindow"],
        default="waveform",
        help="Apply degradation either to full waveform samples or only inside the configured bit window.",
    )
    p.add_argument(
        "--transformer-degrade-stride-skew-max",
        type=float,
        default=0.0,
        help="Max per-sample stride skew factor; 0 disables stride-skew warp.",
    )
    p.add_argument(
        "--transformer-degrade-vectorized-window",
        type=int,
        default=4,
        help="Prepare this many training steps at once using vectorized degradation (>=1).",
    )
    p.add_argument("--transformer-loss-entropy-weight", type=float, default=0.60)
    p.add_argument("--transformer-loss-high-bit-weight", type=float, default=0.50)
    p.add_argument("--transformer-loss-low-bit-weight", type=float, default=0.08)
    p.add_argument("--transformer-loss-wave-l1-weight", type=float, default=1.00)
    p.add_argument(
        "--transformer-loss-score-target-weight",
        type=float,
        default=1.50,
        help="Weight for keeping transformed-wave classifier feature score at/above clean-wave target.",
    )
    p.add_argument(
        "--transformer-loss-score-target-margin",
        type=float,
        default=0.02,
        help="Required improvement margin for target score (after vs before); loss penalizes shortfall.",
    )
    p.add_argument(
        "--transformer-loss-score-rank-weight",
        type=float,
        default=0.40,
        help="Additional signal-gain weight for target-vs-non-target ranking improvement.",
    )
    p.add_argument(
        "--transformer-loss-score-rank-margin",
        type=float,
        default=0.02,
        help="Margin used by ranking gap: target scores should exceed non-target max by this amount.",
    )
    p.add_argument(
        "--transformer-loss-score-spurious-weight",
        type=float,
        default=0.35,
        help="Additional signal-gain weight for suppressing non-target/spurious labels.",
    )
    p.add_argument(
        "--transformer-loss-score-spurious-threshold",
        type=float,
        default=0.35,
        help="Probability threshold used for soft counting spurious non-target activations.",
    )
    p.add_argument(
        "--transformer-loss-score-spurious-temp",
        type=float,
        default=0.20,
        help="Temperature for soft spurious-hit counting around spurious threshold.",
    )
    p.add_argument(
        "--transformer-loss-score-spurious-margin",
        type=float,
        default=0.00,
        help="Required reduction margin for spurious labels (after must be below before-margin).",
    )
    p.add_argument(
        "--target-label-knockout-prob",
        type=float,
        default=0.0,
        help=(
            "Per-row probability of randomly dropping a subset of active target labels during training-target assembly "
            "(0 disables label knockout)."
        ),
    )
    p.add_argument(
        "--target-label-knockout-max-drop-frac",
        type=float,
        default=0.50,
        help="Maximum fraction of active labels that may be dropped when knockout is applied to a row.",
    )
    p.add_argument(
        "--target-label-knockout-min-keep",
        type=int,
        default=1,
        help="Minimum number of active labels to preserve in a row after knockout.",
    )
    p.add_argument(
        "--transformer-accepted-preload-max",
        type=int,
        default=0,
        help="If >0, preload up to this many accepted-library waves into transformer train streams before rounds start.",
    )
    p.add_argument(
        "--transformer-accepted-append-max-per-round",
        type=int,
        default=0,
        help="If >0, append at most this many newly accepted waves per round into transformer train streams (0=all).",
    )
    p.add_argument(
        "--orchestration-mode",
        choices=["none", "sequential", "alternating", "staged_cgr", "staged_cgrw"],
        default="sequential",
        help=(
            "In berkeley_multilabel mode: training schedule "
            "(staged_cgr = C gate -> G gate -> R gate, "
            "staged_cgrw = C gate -> G gate -> R gate -> W feedback)."
        ),
    )
    p.add_argument("--orchestration-cycles", type=int, default=3)
    p.add_argument("--orchestration-rounds", type=int, default=2, help="Used when mode=alternating, staged_cgr, or staged_cgrw.")
    p.add_argument("--generator-epochs-per-round", type=int, default=1)
    p.add_argument("--generator-steps-per-round", type=int, default=120)
    p.add_argument("--generator-batch-size", type=int, default=64)
    p.add_argument(
        "--generator-grad-accum-steps",
        type=int,
        default=1,
        help="Micro-batch accumulation factor for generator/discriminator updates.",
    )
    p.add_argument(
        "--generator-classifier-batch-cap",
        type=int,
        default=0,
        help="If >0, chunk classifier forward passes in generator stage to this batch size cap.",
    )
    p.add_argument("--generator-z-dim", type=int, default=128)
    p.add_argument("--generator-base-ch", type=int, default=128)
    p.add_argument("--generator-depth", type=int, default=6)
    p.add_argument("--generator-min-ch", type=int, default=24)
    p.add_argument("--generator-lr", type=float, default=2e-4)
    p.add_argument("--discriminator-lr", type=float, default=2e-4)
    p.add_argument("--discriminator-base-ch", type=int, default=96)
    p.add_argument("--discriminator-depth", type=int, default=5)
    p.add_argument("--discriminator-max-ch", type=int, default=512)
    p.add_argument(
        "--discriminator-steps-per-generator-step",
        type=int,
        default=1,
        help="Number of discriminator updates per generator update (n_critic style).",
    )
    p.add_argument("--generator-log-every", type=int, default=20)
    p.add_argument("--generator-loss-adv-weight", type=float, default=0.50)
    p.add_argument("--generator-loss-cls-weight", type=float, default=1.50)
    p.add_argument("--generator-loss-mask-weight", type=float, default=1.00)
    p.add_argument("--generator-loss-disc-mask-weight", type=float, default=0.50)
    p.add_argument("--generator-loss-outside-mask-weight", type=float, default=0.25)
    p.add_argument(
        "--generator-loss-wave-weight",
        type=float,
        default=0.0,
        help="Additional generator loss weight from the differentiable wave/refine/render path.",
    )
    p.add_argument(
        "--generator-wave-score-margin",
        type=float,
        default=0.02,
        help="Required score margin for wave-coupled generator path (after vs before).",
    )
    p.add_argument(
        "--generator-wave-denoise-weight",
        type=float,
        default=0.25,
        help="Weight for wave-path denoise term in generator coupling loss.",
    )
    p.add_argument(
        "--generator-wave-carrier-blend",
        type=float,
        default=0.35,
        help="Blend ratio between sampled carrier waves and generator payload wave in coupling path.",
    )
    p.add_argument(
        "--generator-wave-strength",
        type=float,
        default=0.35,
        help="Degradation strength for the generator wave-coupling path.",
    )
    p.add_argument(
        "--generator-wave-stride-skew-max",
        type=float,
        default=0.18,
        help="Stride-skew max for generator wave-coupling degradation.",
    )
    p.add_argument(
        "--generator-wave-degrade-mode",
        choices=["full", "blur_stride"],
        default="blur_stride",
        help="Degradation mode used in generator wave-coupling path.",
    )
    p.add_argument("--generator-mask-threshold", type=float, default=0.5)
    p.add_argument("--generator-gate-min-target-prob", type=float, default=0.55)
    p.add_argument("--generator-gate-maintain-rounds", type=int, default=1)
    p.add_argument(
        "--generator-fake-feedback-enabled",
        dest="generator_fake_feedback_enabled",
        action="store_true",
        help=(
            "When staged_cgr/staged_cgrw is active, train classifier with generated images using a dedicated "
            "'GAN image' text-embedding vector and discriminator pass/fail balanced feedback."
        ),
    )
    p.add_argument(
        "--no-generator-fake-feedback-enabled",
        dest="generator_fake_feedback_enabled",
        action="store_false",
    )
    p.add_argument(
        "--fake-image-sentinel-label",
        type=str,
        default="GAN image",
        help="Text prompt encoded by the active label-embedding backend for fake-feedback vector supervision.",
    )
    p.add_argument("--generator-fake-feedback-epochs", type=int, default=1)
    p.add_argument("--generator-fake-feedback-steps-per-round", type=int, default=48)
    p.add_argument("--generator-fake-feedback-batch-size", type=int, default=32)
    p.add_argument("--generator-fake-feedback-lr", type=float, default=6e-4)
    p.add_argument("--generator-fake-feedback-weight-decay", type=float, default=1e-4)
    p.add_argument("--generator-fake-feedback-vector-weight", type=float, default=1.0)
    p.add_argument("--generator-fake-feedback-condition-weight", type=float, default=0.35)
    p.add_argument("--generator-fake-feedback-disc-conf-temperature", type=float, default=1.0)
    p.add_argument("--generator-fake-feedback-disc-conf-floor", type=float, default=0.25)
    p.add_argument(
        "--generator-fake-feedback-disc-balance-groups",
        dest="generator_fake_feedback_disc_balance_groups",
        action="store_true",
        help="Balance discriminator-positive and discriminator-negative generated groups while applying confidence weights.",
    )
    p.add_argument(
        "--no-generator-fake-feedback-disc-balance-groups",
        dest="generator_fake_feedback_disc_balance_groups",
        action="store_false",
    )
    p.add_argument("--generator-fake-feedback-log-every", type=int, default=0)
    p.add_argument(
        "--generator-fake-feedback-include-condition-targets",
        dest="generator_fake_feedback_include_condition_targets",
        action="store_true",
        help="Include generator condition targets alongside fake-vector supervision during fake-feedback refresh.",
    )
    p.add_argument(
        "--no-generator-fake-feedback-include-condition-targets",
        dest="generator_fake_feedback_include_condition_targets",
        action="store_false",
    )
    p.add_argument(
        "--joint-enabled",
        dest="joint_enabled",
        action="store_true",
        help="After C/G/R gates are satisfied, run coupled joint G+R updates each round.",
    )
    p.add_argument("--no-joint-enabled", dest="joint_enabled", action="store_false")
    p.add_argument("--joint-generator-epochs-per-round", type=int, default=1)
    p.add_argument("--joint-generator-steps-per-round", type=int, default=48)
    p.add_argument("--joint-transformer-epochs-per-round", type=int, default=1)
    p.add_argument("--joint-transformer-steps-per-round", type=int, default=40)
    p.add_argument("--transformer-epochs-per-round", type=int, default=1)
    p.add_argument("--transformer-steps-per-round", type=int, default=40)
    p.add_argument("--wave-cls-epochs-per-round", type=int, default=1)
    p.add_argument("--wave-cls-train-samples", type=int, default=768)
    p.add_argument("--wave-cls-val-samples", type=int, default=256)
    p.add_argument("--wave-cls-batch-size", type=int, default=32)
    p.add_argument("--wave-cls-lr", type=float, default=1e-3)
    p.add_argument("--wave-cls-log-every", type=int, default=0, help="If >0, emit wave-classifier preview/log every N train steps.")
    p.add_argument(
        "--wave-cls-semantic-mix-noise-prob",
        type=float,
        default=0.10,
        help="Per-sample probability of semantic mix/noise augmentation during wave-classifier training.",
    )
    p.add_argument(
        "--wave-cls-semantic-mix-noise-std",
        type=float,
        default=0.03,
        help="Gaussian noise std applied on mixed wave-classifier samples.",
    )
    p.add_argument(
        "--wave-cls-semantic-mix-blend-min",
        type=float,
        default=0.10,
        help="Lower blend bound for semantic mix augmentation.",
    )
    p.add_argument(
        "--wave-cls-semantic-mix-blend-max",
        type=float,
        default=0.35,
        help="Upper blend bound for semantic mix augmentation.",
    )
    p.add_argument(
        "--wave-zero-shot-query-texts",
        default="",
        help=(
            "Optional open-vocabulary text query list for wave-classifier zero-shot inference "
            "(split by | ; , or newline). When empty, falls back to --label-query-texts."
        ),
    )
    p.add_argument("--wave-zero-shot-topk", type=int, default=3, help="Per-sample top-k queries to report for wave zero-shot inference.")
    p.add_argument("--wave-zero-shot-max-samples", type=int, default=128, help="Max wave-validation samples used per zero-shot evaluation (0=all).")
    p.add_argument("--wave-zero-shot-report-samples", type=int, default=8, help="How many sample-level zero-shot rows to store.")
    p.add_argument(
        "--wave-cls-accepted-only",
        action="store_true",
        help="Train wave classifier on only accepted transformer outputs (score/l1 gated).",
    )
    p.add_argument(
        "--gate-pregestation-loss-target",
        type=float,
        default=1.5,
        help=(
            "Hard pre-gestation gate: maximum allowed average classifier loss on the Stage-0 formal logic color-direction pairs set. "
            "Stage 1 gestation does not begin until this gate is satisfied."
        ),
    )
    p.add_argument(
        "--gate-pregestation-maintain-rounds",
        type=int,
        default=1,
        help="Required consecutive pre-gestation gate passes before Stage 1 gestation can begin.",
    )
    p.add_argument(
        "--pregestation-stage-batch-size",
        type=int,
        default=8,
        help=(
            "Batch size for Stage-0 pre-gestation formal-logic training and its matching gate loader. "
            "Values <=0 fall back to --gate-gestation-batch-size, then --gate-berkeley-batch-size, then --classifier-batch-size."
        ),
    )
    p.add_argument(
        "--pregestation-stage-deck-passes",
        type=int,
        default=4,
        help="How many shuffled passes over the Stage-0 loader to run per pre-gestation round.",
    )
    p.add_argument(
        "--pregestation-stage-min-steps",
        type=int,
        default=128,
        help="Minimum Stage-0 pre-gestation optimizer steps per round even if the deck is smaller.",
    )
    p.add_argument(
        "--pregestation-circle-radius-temperature",
        type=float,
        default=1.0,
        dest="pregestation_circle_radius_temperature",
        help=(
            "Temperature multiplier for the pre-gestation color-circle radius. "
            "1.0 = default range [0.085, 0.115]; values <1 produce smaller circles, >1 larger circles."
        ),
    )
    p.add_argument(
        "--pregestation-circle-displacement-temperature",
        type=float,
        default=1.0,
        dest="pregestation_circle_displacement_temperature",
        help=(
            "Base temperature multiplier for the offset of the color circle anchor from center. "
            "1.0 = default 0.32 normalized-image-unit displacement; <1 pushes anchors toward center, >1 pushes farther out. "
            "The effective temperature is further scaled by the undulating schedule controlled by "
            "--pregestation-temp-trend, --pregestation-temp-oscillation-amplitude, and "
            "--pregestation-temp-oscillation-period."
        ),
    )
    p.add_argument(
        "--pregestation-temp-trend",
        type=float,
        default=1.5,
        dest="pregestation_temp_trend",
        help=(
            "Maximum additional temperature multiplier added as the pregestation gate streak approaches "
            "the maintain threshold.  At streak=0 the trend contribution is 0; at streak=maintain_rounds "
            "it reaches this value.  E.g. 1.5 means the base displacement is doubled when the model is "
            "close to passing the gate.  Defaults to 1.5."
        ),
    )
    p.add_argument(
        "--pregestation-temp-oscillation-amplitude",
        type=float,
        default=0.4,
        dest="pregestation_temp_oscillation_amplitude",
        help=(
            "Amplitude of the sine-wave oscillation overlaid on the temperature trend.  Positive peaks "
            "raise the effective displacement temperature (harder ΓåÆ more diagonal samples); negative peaks "
            "lower it (easier ΓåÆ pure cardinal placements).  Defaults to 0.4."
        ),
    )
    p.add_argument(
        "--pregestation-temp-oscillation-period",
        type=int,
        default=4,
        dest="pregestation_temp_oscillation_period",
        help=(
            "Number of pregestation data rebuilds per complete sine-wave cycle.  Lower values produce "
            "faster easy/hard alternation; higher values create longer stable phases.  Defaults to 4."
        ),
    )
    p.add_argument(
        "--pregestation-sub-rounds",
        type=int,
        default=1,
        dest="pregestation_sub_rounds",
        help=(
            "Number of sequential sub-rounds within Stage 0 pre-gestation. "
            "The model must pass the gate 'maintain' consecutive times per sub-round and "
            "climb through all sub-rounds before Stage 1 gestation begins. "
            "Default 1 = single level, backward-compatible with prior behavior."
        ),
    )
    p.add_argument(
        "--gate-gestation-loss-target",
        type=float,
        default=0.18,
        help=(
            "Hard gestation gate: maximum allowed average classifier loss on bootstrap-archetype-only rows. "
            "No downstream stage runs until this gate is satisfied."
        ),
    )
    p.add_argument(
        "--gate-gestation-maintain-rounds",
        type=int,
        default=1,
        help="Required consecutive gestation gate passes before Stage-2 validation gate can pass.",
    )
    p.add_argument(
        "--gate-gestation-val-max",
        type=int,
        default=0,
        help="Max gestation samples per gate eval (0=all bootstrap-archetype rows).",
    )
    p.add_argument(
        "--gate-gestation-batch-size",
        type=int,
        default=0,
        help=(
            "Batch size for gestation gate eval. "
            "0 uses --gate-berkeley-batch-size, then --berkeley-refresh-batch-size, then --classifier-batch-size."
        ),
    )
    p.add_argument(
        "--gate-gestation-eval-max-steps",
        type=int,
        default=0,
        help="Max gestation batches per eval (0=full gestation loader).",
    )
    p.add_argument(
        "--gate-berkeley-min-score",
        type=float,
        default=0.0,
        help="Legacy wave-score floor gate (kept for compatibility).",
    )
    p.add_argument(
        "--gate-berkeley-min-confidence",
        type=float,
        default=0.0,
        help="If >0, Stage-2 validation classifier mean-confidence gate (payload validation split).",
    )
    p.add_argument(
        "--gate-berkeley-min-macro-f1",
        type=float,
        default=0.0,
        help="If >0, Stage-2 validation classifier macro-F1 gate (payload validation split).",
    )
    p.add_argument(
        "--gate-berkeley-loss-target",
        type=float,
        default=0.12,
        help=(
            "If >0, explicit Stage-2 validation classifier loss target for loss-aware gate clamp. "
            "If <=0, target is adaptive from best recent loss * --gate-berkeley-loss-target-mult."
        ),
    )
    p.add_argument(
        "--gate-berkeley-loss-target-mult",
        type=float,
        default=1.15,
        help="Adaptive loss target multiplier over best recent Berkeley-val loss (>=1.0).",
    )
    p.add_argument(
        "--gate-berkeley-loss-min-floor",
        type=float,
        default=0.05,
        help="Lower bound for adaptive Berkeley-val loss target.",
    )
    p.add_argument(
        "--gate-berkeley-loss-thaw-rounds",
        type=int,
        default=2,
        help="Rounds where confidence/F1 can break ice before loss clamp ramps in.",
    )
    p.add_argument(
        "--gate-berkeley-loss-ramp-rounds",
        type=int,
        default=6,
        help="Rounds used to polynomially ramp loss influence after thaw.",
    )
    p.add_argument(
        "--gate-berkeley-loss-poly",
        type=float,
        default=2.0,
        help="Polynomial power for loss influence ramp (>=1).",
    )
    p.add_argument(
        "--gate-berkeley-loss-max-weight",
        type=float,
        default=0.65,
        help="Max loss weight in confidence/loss blended Berkeley gate score [0..1].",
    )
    p.add_argument(
        "--gate-berkeley-loss-clamp-memory-rounds",
        type=int,
        default=8,
        help="How many recent rounds to remember before releasing a loss clamp.",
    )
    p.add_argument(
        "--gate-berkeley-loss-release-rounds",
        type=int,
        default=3,
        help="Required consecutive good-loss rounds to release active loss clamp.",
    )
    p.add_argument(
        "--gate-berkeley-loss-blend-margin",
        type=float,
        default=0.02,
        help="Slack margin applied to blended confidence/loss clamp score.",
    )
    p.add_argument(
        "--gate-berkeley-maintain-rounds",
        type=int,
        default=1,
        help="Required consecutive Stage-2 validation gate passes before downstream is allowed.",
    )
    p.add_argument(
        "--gate-berkeley-val-max",
        type=int,
        default=0,
        help="Deprecated for Gate 2. Stage-2 validation gate always evaluates the full payload validation split.",
    )
    p.add_argument(
        "--gate-berkeley-batch-size",
        type=int,
        default=0,
        help=(
            "Batch size for Stage-2 validation gate evaluation loader. "
            "0 uses --berkeley-refresh-batch-size (or --classifier-batch-size when refresh batch is auto)."
        ),
    )
    p.add_argument(
        "--gate-berkeley-eval-max-steps",
        type=int,
        default=0,
        help="Max Stage-2 validation gate batches per eval after Gate 2 is already ready (0=full loader).",
    )
    p.add_argument(
        "--gate-total-token-schedule-enabled",
        dest="gate_total_token_schedule_enabled",
        action="store_true",
        help="Sort Stage-2 validation gate rows to prioritize active optional semantic tokens before gate decisions.",
    )
    p.add_argument("--no-gate-total-token-schedule-enabled", dest="gate_total_token_schedule_enabled", action="store_false")
    p.add_argument(
        "--gate-total-token-schedule-threshold",
        type=float,
        default=0.55,
        help="Activation threshold used while token-sorting Stage-2 validation gate rows.",
    )
    p.add_argument(
        "--gate-wave-min-entropy",
        type=float,
        default=0.12,
        help="Wave-material gate: minimum mean binary entropy on wave-rendered logits.",
    )
    p.add_argument(
        "--gate-wave-maintain-rounds",
        type=int,
        default=1,
        help="Required consecutive wave-entropy gate passes before transformer stage.",
    )
    p.add_argument(
        "--gate-transformer-min-score-after",
        type=float,
        default=0.0,
        help="If >0, transformer post-score gate threshold before wave-classifier stage.",
    )
    p.add_argument(
        "--gate-transformer-min-gain",
        type=float,
        default=-1e9,
        help="Transformer gain threshold before wave-classifier stage.",
    )
    p.add_argument(
        "--gate-transformer-maintain-rounds",
        type=int,
        default=1,
        help="Required consecutive transformer gate passes before wave-classifier stage.",
    )
    p.add_argument(
        "--gate-wave-feedback-min-acc",
        type=float,
        default=0.25,
        help=(
            "If >0, require previous round wave-classifier validation accuracy at/above this "
            "value before transformer gate can pass."
        ),
    )
    p.add_argument(
        "--gate-wave-feedback-min-zs-top1-rate",
        type=float,
        default=0.15,
        help=(
            "If >0 and zero-shot ran, require previous round wave zero-shot top1-rate at/above "
            "this value before transformer gate can pass."
        ),
    )
    p.add_argument(
        "--wave-feedback-zero-shot-weight",
        type=float,
        default=0.50,
        help="Weight of zero-shot top1-rate when building combined wave feedback score.",
    )
    p.add_argument(
        "--wave-feedback-transformer-loss-boost",
        type=float,
        default=0.50,
        help=(
            "Boost factor applied to transformer score-target loss weight when combined wave "
            "feedback score is weak."
        ),
    )
    p.add_argument(
        "--wave-feedback-generator-loss-boost",
        type=float,
        default=0.35,
        help=(
            "Boost factor applied to generator wave-coupling loss weight when combined wave "
            "feedback score is weak."
        ),
    )
    p.add_argument("--accept-score-threshold", type=float, default=0.40)
    p.add_argument("--accept-l1-threshold", type=float, default=0.20)
    p.add_argument("--library-max-per-round", type=int, default=24)
    p.add_argument("--chunk-samples", type=int, default=32768)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--transformer-lr", type=float, default=2e-4)
    p.add_argument("--max-delta", type=float, default=0.2)
    p.add_argument("--stream-max-points", type=int, default=262144)
    p.add_argument("--save-preview", type=int, default=2)
    p.add_argument(
        "--training-preview-enabled",
        dest="training_preview_enabled",
        action="store_true",
        help="Save image+label supervision previews during round updates.",
    )
    p.add_argument("--no-training-preview-enabled", dest="training_preview_enabled", action="store_false")
    p.add_argument("--training-preview-topk", type=int, default=6)
    p.add_argument("--training-preview-generator-samples", type=int, default=4)
    p.add_argument("--training-preview-every-round", type=int, default=1)
    p.add_argument(
        "--stage-opengl-preview-enabled",
        dest="stage_opengl_preview_enabled",
        action="store_true",
        help="Show live OpenGL previews across orchestration stages (C/G/R/J/W).",
    )
    p.add_argument("--no-stage-opengl-preview-enabled", dest="stage_opengl_preview_enabled", action="store_false")
    p.add_argument("--stage-opengl-preview-scale", type=int, default=3)
    p.add_argument("--stage-opengl-preview-every-round", type=int, default=1)
    p.add_argument(
        "--stage-opengl-preview-bootstrap",
        dest="stage_opengl_preview_bootstrap",
        action="store_true",
        help="Force an immediate OpenGL preview frame at startup to confirm window/display path.",
    )
    p.add_argument("--no-stage-opengl-preview-bootstrap", dest="stage_opengl_preview_bootstrap", action="store_false")
    p.add_argument(
        "--stage-opengl-preview-required",
        dest="stage_opengl_preview_required",
        action="store_true",
        help="Abort run if stage OpenGL preview cannot initialize.",
    )
    p.add_argument("--no-stage-opengl-preview-required", dest="stage_opengl_preview_required", action="store_false")
    p.add_argument(
        "--viewer-port-file",
        default=None,
        help="Path to a file containing the IPC port of a running standalone GUI "
             "(wav_ml_gui_main.py).  When set, the pipeline connects to the external "
             "GUI via ViewerIPCProxy instead of creating its own PyGame window.",
    )
    p.set_defaults(
        cudnn_benchmark=True,
        allow_tf32=True,
        loader_persistent_workers=True,
        classifier_subset_fresh_round_sampling=True,
        transformer_cache_eval_batches=True,
        transformer_degrade_inputs=True,
        transformer_visualize_status=False,
        latent_berkeley_imprint_mix_targets=True,
        enforce_render_bitmask_enable=False,
        classifier_init_required=False,
        label_embeddings_enabled=True,
        berkeley_payload_cache_rebuild=False,
        semantic_vocab_churn_enabled=False,
        semantic_vocab_regurgitated_churn_enabled=True,
        semantic_vocab_deferred_churn_prob=0.35,
        semantic_vocab_deferred_churn_max_per_cycle=8,
        semantic_vocab_text_condition_topk=0,
        semantic_vocab_unknown_label_rows_per_cycle=0,
        semantic_vocab_auto_symbol_pool=False,
        semantic_vocab_pregestation_enabled=True,
        semantic_vocab_symbol_include_pictograms=True,
        semantic_vocab_reference_flashcards=False,
        semantic_stage_cache_enabled=True,
        semantic_stage_cache_rebuild=False,
        gd_vocab_library_enabled=True,
        gd_vocab_library_autoload=True,
        training_preview_enabled=True,
        stage_opengl_preview_enabled=True,
        stage_opengl_preview_bootstrap=True,
        stage_opengl_preview_required=False,
        checkpoint_after_training_segment=False,
        stage_c_lora_enabled=True,
        stage_module_offload=True,
        stage_module_offload_empty_cache=True,
        joint_enabled=True,
        generator_fake_feedback_enabled=True,
        generator_fake_feedback_include_condition_targets=True,
        generator_fake_feedback_disc_balance_groups=True,
        gate_total_token_schedule_enabled=True,
    )
    args = p.parse_args()
    if int(args.semantic_vocab_text_condition_topk) != 0:
        raise RuntimeError(
            "Strict semantic parity mode requires --semantic-vocab-text-condition-topk=0 "
            "(no semantic-dimension pruning allowed)."
        )
    return args




def _resolve_shared_embed_image_size(args) -> int:
    image_size = max(1, int(args.image_size))
    berkeley_size = int(args.berkeley_image_size)
    if berkeley_size > 0:
        return max(1, berkeley_size)
    return image_size



