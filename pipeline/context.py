"""
PipelineContext — the single source of truth shared by all graph nodes.

Previously this state was scattered across thousands of lines of local variables
inside main().  It now lives in one place so every node can read/write it cleanly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Gate state
# ---------------------------------------------------------------------------

@dataclass
class GateState:
    """Pass/fail tracking for one pipeline gate.

    Each gate has a *required_consecutive_passes* threshold; the gate only
    flips to ``passed=True`` once that many rounds in a row have cleared
    the metric target.
    """

    name: str = ""
    passed: bool = False
    consecutive_passes: int = 0
    required_consecutive: int = 1
    last_metric: float = float("nan")
    history: List[Tuple[int, float]] = field(default_factory=list)  # (round_id, metric)

    def record(self, round_id: int, metric: float, threshold: float, above: bool = False) -> bool:
        """Record *metric* for *round_id*.

        ``above=True``  → gate passes when metric  > threshold (e.g. accuracy/F1).
        ``above=False`` → gate passes when metric  < threshold (e.g. loss).

        Returns True if the gate is now passed.
        """
        self.last_metric = float(metric)
        self.history.append((int(round_id), float(metric)))
        met = (float(metric) > float(threshold)) if above else (float(metric) < float(threshold))
        if met:
            self.consecutive_passes += 1
        else:
            self.consecutive_passes = 0
        if not self.passed and self.consecutive_passes >= self.required_consecutive:
            self.passed = True
        return self.passed

    def reset(self) -> None:
        self.passed = False
        self.consecutive_passes = 0
        self.last_metric = float("nan")
        self.history.clear()


# ---------------------------------------------------------------------------
# Main context
# ---------------------------------------------------------------------------

@dataclass
class PipelineContext:
    """All mutable shared state for the training pipeline.

    Nodes read their inputs from this object and write their outputs back.
    The orchestrator creates one context, wires the graph, and passes it to
    every ``execute_sequence`` call through the entire run.
    """

    # ---- parsed CLI arguments (argparse.Namespace) ----------------------
    args: Any = None

    # ---- runtime / device -----------------------------------------------
    device: Optional[torch.device] = None
    output_dir: Optional[Path] = None
    raise_on_node_failure: bool = True

    # ---- AMP ------------------------------------------------------------
    amp_enabled: bool = False
    amp_dtype: Optional[torch.dtype] = None

    # ---- models ---------------------------------------------------------
    #  Each model slot is populated by its node's first execution.

    # Main semantic / Berkeley classifier (TinyConvClassifier)
    classifier: Optional[nn.Module] = None
    classifier_optimizer: Optional[torch.optim.Optimizer] = None
    classifier_lr_controller: Optional[Any] = None
    classifier_grad_scaler: Optional[Any] = None

    # Frozen CPU replica used for gate evaluation without disrupting gradients
    gate_classifier: Optional[nn.Module] = None

    # Patch-based wav→image transformer (WavePatchTransformer)
    transformer: Optional[nn.Module] = None
    transformer_optimizer: Optional[torch.optim.Optimizer] = None
    transformer_lr_controller: Optional[Any] = None
    transformer_grad_scaler: Optional[Any] = None

    # Conditional GAN generator (ConditionalBitPlaneGenerator)
    generator: Optional[nn.Module] = None
    generator_optimizer: Optional[torch.optim.Optimizer] = None
    generator_grad_scaler: Optional[Any] = None

    # Conditional GAN discriminator (ConditionalBitPlaneDiscriminator)
    discriminator: Optional[nn.Module] = None
    discriminator_optimizer: Optional[torch.optim.Optimizer] = None
    discriminator_grad_scaler: Optional[Any] = None

    # Second TinyConvClassifier trained on transformer-accepted wave chunks
    wave_classifier: Optional[nn.Module] = None
    wave_classifier_optimizer: Optional[torch.optim.Optimizer] = None
    wave_classifier_grad_scaler: Optional[Any] = None

    # ---- vocabulary / embedding -----------------------------------------
    supervised_class_names: List[str] = field(default_factory=list)
    class_names: List[str] = field(default_factory=list)
    active_extra_terms: List[str] = field(default_factory=list)
    core_terms: List[str] = field(default_factory=list)
    label_embedding_bank: Optional[Any] = None       # np.ndarray [n_classes, embed_dim]
    label_texts: List[str] = field(default_factory=list)
    label_embedding_info: Dict[str, Any] = field(default_factory=dict)
    semantic_term_to_idx: Dict[str, int] = field(default_factory=dict)

    # ---- gate states ----------------------------------------------------
    gate_pregestation: GateState = field(
        default_factory=lambda: GateState(name="pregestation")
    )
    gate_gestation: GateState = field(
        default_factory=lambda: GateState(name="gestation")
    )
    gate_berkeley: GateState = field(
        default_factory=lambda: GateState(name="berkeley")
    )
    gate_wave: GateState = field(
        default_factory=lambda: GateState(name="wave")
    )
    gate_generator: GateState = field(
        default_factory=lambda: GateState(name="generator")
    )
    gate_transformer: GateState = field(
        default_factory=lambda: GateState(name="transformer")
    )

    # ---- WAV / stream data ----------------------------------------------
    wav_records: List[Any] = field(default_factory=list)      # List[WaveRecord]
    float_streams: List[Any] = field(default_factory=list)    # decoded mono float32 arrays
    stream_paths: List[str] = field(default_factory=list)
    render_config: Optional[Any] = None                       # RenderConfig
    accepted_wave_library: List[Any] = field(default_factory=list)

    # ---- stage data loaders / datasets ----------------------------------
    # Stage 0 – pre-gestation (synthetic geometric logic images)
    pregestation_loader: Optional[Any] = None
    pregestation_dataset: Optional[Any] = None

    # Stage 1 – gestation (bootstrap primitive symbol images)
    gestation_loader: Optional[Any] = None
    gestation_dataset: Optional[Any] = None

    # Stage 2 – Berkeley SBD full refresh
    berkeley_refresh_loader: Optional[Any] = None
    berkeley_gate_val_loader: Optional[Any] = None
    berkeley_cache: Optional[Any] = None                      # pre-loaded batches

    # Payload validation dataset (Gate 2 deck)
    payload_validation_dataset: Optional[Any] = None
    payload_validation_loader: Optional[Any] = None

    # ---- Berkeley payload bank ------------------------------------------
    payload_bank: Optional[Any] = None
    payload_conditions: List[Any] = field(default_factory=list)
    payload_masks: List[Any] = field(default_factory=list)
    payload_bank_ready: bool = False

    # ---- symbol / flashcard pools ---------------------------------------
    symbol_pool: Optional[Dict[str, Any]] = None
    flashcard_rows: List[Any] = field(default_factory=list)
    pregestation_logic_rows: Optional[Any] = None            # cached geometric images

    # ---- wave pool ------------------------------------------------------
    latent_wav_pool_dir: Optional[Path] = None

    # ---- LoRA adapter state (classifier) --------------------------------
    lora_slot_snapshots: Dict[str, Any] = field(default_factory=dict)
    lora_active_slot: str = ""

    # ---- metrics / history ----------------------------------------------
    metrics_history: List[Dict[str, Any]] = field(default_factory=list)
    round_rows: List[Dict[str, Any]] = field(default_factory=list)
    config_search_history: List[Dict[str, Any]] = field(default_factory=list)

    # ---- orchestration loop state ---------------------------------------
    cycle: int = 0
    round_id: int = 0
    total_rounds_completed: int = 0
    vocab_rotation_cycle: int = 0
    last_node_statuses: Dict[str, str] = field(default_factory=dict)

    # ---- orchestration configuration (plan-driven) ----------------------
    # These mirror the top-level CLI flags; stored here so nodes can read
    # them without needing to fall back to ctx.args in plan-driven mode.
    orchestration_mode: str = "staged_cgrw"
    orchestration_cycles: int = 1
    orchestration_rounds: int = 1

    # ---- viewer / preview -----------------------------------------------
    viewer_proxy: Optional[Any] = None
    loss_logger: Optional[Any] = None

    # ---- checkpoint metadata --------------------------------------------
    checkpoint_path: Optional[Path] = None
    resumed_from_checkpoint: bool = False
    resume_dir: Optional[Path] = None
    resume_summary: Optional[Dict[str, Any]] = None
    resume_pipeline_ckpt: Optional[Dict[str, Any]] = None
    run_tag: str = ""
    semantic_cache_nonce: str = ""
    graph_plan: Optional[Any] = None
    graph_plan_path: Optional[Path] = None
    plan_id: str = ""
    worker_id: str = ""
    session_id: str = ""
    runtime_snapshot_path: Optional[Path] = None

    # ---- misc flags -----------------------------------------------------
    berkeley_data_root: str = ""
    semantic_stage_cache_dir: str = ""

    # ------------------------------------------------------------------
    # Convenience predicates (used as edge conditions in the graph)
    # ------------------------------------------------------------------

    def all_base_gates_passed(self) -> bool:
        """True once Stage 0, 1, and 2 have all cleared."""
        return (
            self.gate_pregestation.passed
            and self.gate_gestation.passed
            and self.gate_berkeley.passed
        )

    def early_gates_passed(self) -> bool:
        """True once Stage 0 and Stage 1 have cleared (Stage 2 not required yet)."""
        return self.gate_pregestation.passed and self.gate_gestation.passed

    def generator_and_classifier_ready(self) -> bool:
        return (
            self.all_base_gates_passed()
            and self.generator is not None
            and self.discriminator is not None
        )

    def wave_stage_ready(self) -> bool:
        return (
            self.all_base_gates_passed()
            and self.gate_transformer.passed
            and self.transformer is not None
        )

    def gate_override_enabled(self) -> bool:
        proxy = self.viewer_proxy
        if proxy is None:
            return False
        fn = getattr(proxy, "gate_override_enabled", None)
        if not callable(fn):
            return False
        try:
            return bool(fn())
        except Exception:
            return False

    def is_cycle_selected(self, cycle_id: int) -> bool:
        proxy = self.viewer_proxy
        if proxy is None:
            return True
        fn = getattr(proxy, "is_cycle_selected", None)
        if not callable(fn):
            return True
        try:
            return bool(fn(int(cycle_id)))
        except Exception:
            return True

    def selected_cycle_ids(self) -> List[int]:
        proxy = self.viewer_proxy
        if proxy is None:
            return []
        fn = getattr(proxy, "selected_cycle_ids", None)
        if not callable(fn):
            return []
        try:
            return [int(x) for x in fn()]
        except Exception:
            return []

    def stop_requested(self) -> bool:
        proxy = self.viewer_proxy
        if proxy is None:
            return False
        fn = getattr(proxy, "stop_requested", None)
        if not callable(fn):
            return False
        try:
            return bool(fn())
        except Exception:
            return False

    def log_metric(self, stage: str, key: str, value: float) -> None:
        self.metrics_history.append(
            {"cycle": self.cycle, "round": self.round_id, "stage": stage, "key": key, "value": float(value)}
        )
