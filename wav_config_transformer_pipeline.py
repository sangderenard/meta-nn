import argparse
import copy
import gc
import hashlib
import json
import math
import os
import queue as _preview_queue_module
import re
import shutil
import sys
import threading
import time
import wave
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from wav_ml_core import (
    COLOR_MODE_MAP,
    COLOR_MODES,
    RenderConfig,
    WaveRecord,
    decode_pcm,
    discover_wavs,
    image_u8_to_tensor,
    normalize_bit_window,
    read_wav_record,
    render_record,
    render_mono_wave_to_tensor,
)
from wav_ml_models import (
    _TransformerStatusOpenGLViewer,
    _LossFileLogger,
    _tensor_to_rgb_u8_image,
    LOSS_STAGE_CLASSIFIER,
    LOSS_STAGE_DISCRIMINATOR,
    LOSS_STAGE_GENERATOR,
    LOSS_STAGE_TRANSFORMER,
    LOSS_STAGE_WAVE_CLASSIFIER,
    LOSS_STAGE_WAVE_CLASSIFIER_EVAL,
)
from wav_ml_viewer import ViewerIPCProxy
from wav_ml_models import (
    ConditionalBitPlaneDiscriminator,
    ConditionalBitPlaneGenerator,
    SinusoidalLRController,
    SinusoidalLROptions,
    TinyConvClassifier,
    WavePatchTransformer,
    evaluate_conditional_generator,
    evaluate_classifier,
    evaluate_feature_score_before_after,
    evaluate_transformer_accuracy,
    multilabel_feature_score,
    configure_torch_runtime,
    ensure_tiny_classifier_lora_slot,
    install_tiny_classifier_lora,
    maybe_compile_module,
    restore_tiny_classifier_lora_snapshot,
    set_seed,
    set_tiny_classifier_lora_state,
    tiny_classifier_lora_snapshot,
    tiny_classifier_lora_modules,
    train_conditional_generator_discriminator,
    train_classifier,
    train_transformer_feature_metric,
)
try:
    from semantic_dataset_loaders import (
        BootstrapDynamicDataset,
        DiskSemanticRowsDataset,
        OverrideTargetSubsetDataset,
        StageDatasetManifest,
        _composite_mask_stack,
        _semantic_color_score_maps,
        build_label_mask_stack,
        build_loader_from_manifest,
        collect_semantic_disk_rows,
        detect_semantic_color_terms,
        infer_semantic_support_mask,
        maybe_wrap_loader_with_threaded_prefetch,
        semantic_mask_stack_collate,
    )
except ModuleNotFoundError:
    from toys_to_survive_development.semantic_dataset_loaders import (
        BootstrapDynamicDataset,
        DiskSemanticRowsDataset,
        OverrideTargetSubsetDataset,
        StageDatasetManifest,
        _composite_mask_stack,
        _semantic_color_score_maps,
        build_label_mask_stack,
        build_loader_from_manifest,
        collect_semantic_disk_rows,
        detect_semantic_color_terms,
        infer_semantic_support_mask,
        maybe_wrap_loader_with_threaded_prefetch,
        semantic_mask_stack_collate,
    )

GUI_STOP_EXIT_CODE = 42
CLASSIFIER_LOSS_SCALE = 1.0
CLASSIFIER_SEMANTIC_COSINE_WEIGHT = 0.35


def _log(msg: str):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Thin main() — delegates to the pipeline graph runner / orchestrator.
#
# The original 12 000-line inline training loop has been replaced by the
# graph-based execution engine in pipeline.orchestrator.run().
# To run in graph mode use::
#
#     python wav_pipeline_graph.py [args]
#
# This entry point is kept for backward-compatibility with scripts and
# batch launchers that invoke  ``python wav_config_transformer_pipeline.py``.
# ---------------------------------------------------------------------------

def main():
    from pipeline.cli import parse_args, _resolve_shared_embed_image_size
    from pipeline.orchestrator import run
    from pathlib import Path

    args = parse_args()

    output_dir = Path(str(getattr(args, "output_dir", "output")).strip() or "output")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        run(args=args, output_dir=output_dir)
    except KeyboardInterrupt:
        _log("[main] interrupted by user")
        return GUI_STOP_EXIT_CODE
    except Exception as exc:
        import traceback
        _log(f"[main] FATAL: {exc}")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(int(main() or 0))


# -------------------------------------------------------------------------
# Lazy re-exports: functions moved to pipeline/ modules.
# Avoids circular imports by deferring the import until first access.
# -------------------------------------------------------------------------
_EXTRACTED_SYMBOLS = {
    "_build_enforced_render_config": "pipeline.cli",
    "_resolve_shared_embed_image_size": "pipeline.cli",
    "parse_args": "pipeline.cli",
    "_apply_model_init": "pipeline.nodes.base",
    "_apply_state_dict": "pipeline.nodes.base",
    "_extract_state_dict": "pipeline.nodes.base",
    "_is_expandable_classifier_head_key": "pipeline.nodes.base",
    "_load_json": "pipeline.nodes.base",
    "_save_pipeline_checkpoint": "pipeline.nodes.base",
    "_strip_module_prefix": "pipeline.nodes.base",
    "_torch_load_cpu": "pipeline.nodes.base",
    "_try_partial_classifier_head_load": "pipeline.nodes.base",
    "_apply_classifier_init": "pipeline.nodes.berkeley_classifier_node",
    "_run_berkeley_refresh_epochs": "pipeline.nodes.berkeley_classifier_node",
    "_run_fake_class_refresh_epochs": "pipeline.nodes.berkeley_classifier_node",
    "_sync_gate_classifier_replica": "pipeline.nodes.berkeley_classifier_node",
    "_auto_berkeley_refresh_batch_size": "pipeline.nodes.data_nodes",
    "_build_berkeley_gate_val_loader": "pipeline.nodes.data_nodes",
    "_build_berkeley_payload_bank": "pipeline.nodes.data_nodes",
    "_build_berkeley_refresh_cache": "pipeline.nodes.data_nodes",
    "_build_berkeley_refresh_loader": "pipeline.nodes.data_nodes",
    "_build_gate_loader_from_arrays": "pipeline.nodes.data_nodes",
    "_build_gate_loader_from_dataset": "pipeline.nodes.data_nodes",
    "_build_payload_validation_gate_dataset": "pipeline.nodes.data_nodes",
    "_build_payload_validation_gate_rows": "pipeline.nodes.data_nodes",
    "_evaluate_berkeley_confidence_loss_gate": "pipeline.nodes.data_nodes",
    "_expand_payload_conditions_with_semantic_bank": "pipeline.nodes.data_nodes",
    "_expand_semantic_mask_supervision_batch": "pipeline.nodes.data_nodes",
    "_filter_stream_pool_for_chunk_samples": "pipeline.nodes.data_nodes",
    "_forward_classifier_outputs_require_mask": "pipeline.nodes.data_nodes",
    "_payload_condition_bank_tensor": "pipeline.nodes.data_nodes",
    "_schedule_payload_gate_rows_by_active_terms": "pipeline.nodes.data_nodes",
    "_schedule_semantic_gate_indices": "pipeline.nodes.data_nodes",
    "_select_validation_indices": "pipeline.nodes.data_nodes",
    "_semantic_mask_bce_loss": "pipeline.nodes.data_nodes",
    "_sum_berkeley_refresh_samples": "pipeline.nodes.data_nodes",
    "_unpack_masked_semantic_batch": "pipeline.nodes.data_nodes",
    "_evaluate_berkeley_classifier_gate": "pipeline.nodes.gate_nodes",
    "_feedback_scaled_weight": "pipeline.nodes.gate_nodes",
    "_save_training_segment_snapshot": "pipeline.nodes.gate_nodes",
    "_wave_feedback_gate_pass": "pipeline.nodes.gate_nodes",
    "_wave_feedback_snapshot": "pipeline.nodes.gate_nodes",
    "_compute_gd_vocab_hash": "pipeline.nodes.generator_node",
    "_load_gd_vocab_library_snapshot": "pipeline.nodes.generator_node",
    "_save_gd_vocab_library_snapshot": "pipeline.nodes.generator_node",
    "_ST_MODEL_CACHE": "pipeline.nodes.label_embedding_node",
    "_apply_label_embedding_bank_to_classifier": "pipeline.nodes.label_embedding_node",
    "_build_label_embedding_bank": "pipeline.nodes.label_embedding_node",
    "_encode_texts_sentence_transformers": "pipeline.nodes.label_embedding_node",
    "_normalize_l2_rows_np": "pipeline.nodes.label_embedding_node",
    "_resolve_label_texts": "pipeline.nodes.label_embedding_node",
    "_random_configs": "pipeline.nodes.transformer_node",
    "_score_config_with_classifier": "pipeline.nodes.transformer_node",
    "_try_scipy_refine": "pipeline.nodes.transformer_node",
    "_DEFAULT_SEMANTIC_CORE_TERMS": "pipeline.nodes.vocab_node",
    "_REMOVED_SEMANTIC_SEED_TERMS": "pipeline.nodes.vocab_node",
    "_build_auto_symbol_term_pool": "pipeline.nodes.vocab_node",
    "_build_internal_bootstrap_symbol_pool": "pipeline.nodes.vocab_node",
    "_build_pregestation_logic_rows": "pipeline.nodes.vocab_node",
    "_build_symbol_term_origin_terms_map": "pipeline.nodes.vocab_node",
    "_build_synthetic_semantic_symbol_pool": "pipeline.nodes.vocab_node",
    "_default_bootstrap_primitive_terms": "pipeline.nodes.vocab_node",
    "_default_semantic_core_terms": "pipeline.nodes.vocab_node",
    "_enrich_pregestation_stack_with_observed_color_masks": "pipeline.nodes.vocab_node",
    "_find_semantic_term_index": "pipeline.nodes.vocab_node",
    "_image_any_to_rgb_chw01": "pipeline.nodes.vocab_node",
    "_label_knockout_optional_rows_np": "pipeline.nodes.vocab_node",
    "_label_knockout_row_np": "pipeline.nodes.vocab_node",
    "_label_knockout_rows_np": "pipeline.nodes.vocab_node",
    "_label_knockout_tensor_batch": "pipeline.nodes.vocab_node",
    "_load_vocab_terms_json": "pipeline.nodes.vocab_node",
    "_merge_symbol_term_pools": "pipeline.nodes.vocab_node",
    "_merge_vocab_terms": "pipeline.nodes.vocab_node",
    "_normalize_active_extra_terms_with_core": "pipeline.nodes.vocab_node",
    "_normalize_vocab_terms": "pipeline.nodes.vocab_node",
    "_parse_label_query_texts": "pipeline.nodes.vocab_node",
    "_removed_semantic_seed_terms": "pipeline.nodes.vocab_node",
    "_rotate_active_extra_terms": "pipeline.nodes.vocab_node",
    "_semantic_active_target_stats": "pipeline.nodes.vocab_node",
    "_semantic_damage_tags": "pipeline.nodes.vocab_node",
    "_semantic_enrich_generated_terms_with_noise_spectrum": "pipeline.nodes.vocab_node",
    "_semantic_expand_inferred_tags": "pipeline.nodes.vocab_node",
    "_semantic_kind_keys_for_class_names": "pipeline.nodes.vocab_node",
    "_semantic_noise_family_terms": "pipeline.nodes.vocab_node",
    "_semantic_noise_profile_key_from_term": "pipeline.nodes.vocab_node",
    "_semantic_noise_profile_terms": "pipeline.nodes.vocab_node",
    "_semantic_noise_terms_from_spectrum_sample": "pipeline.nodes.vocab_node",
    "_semantic_tags_for_symbol_term": "pipeline.nodes.vocab_node",
    "_semantic_term_index_map": "pipeline.nodes.vocab_node",
    "_semantic_terms_with_tonal_tags": "pipeline.nodes.vocab_node",
    "_semantic_tonal_tags_from_image": "pipeline.nodes.vocab_node",
    "_build_labels": "pipeline.nodes.wave_classifier_node",
    "_build_wave_classifier_dataset_from_transformer": "pipeline.nodes.wave_classifier_node",
    "_evaluate_zero_shot_queries_on_images": "pipeline.nodes.wave_classifier_node",
}


def __getattr__(name):
    if name in _EXTRACTED_SYMBOLS:
        import importlib
        _mod = importlib.import_module(_EXTRACTED_SYMBOLS[name])
        _val = getattr(_mod, name)
        globals()[name] = _val
        return _val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
