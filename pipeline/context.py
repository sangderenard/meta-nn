"""
PipelineContext — the single source of truth shared by all graph nodes.

Previously this state was scattered across thousands of lines of local variables
inside main().  It now lives in one place so every node can read/write it cleanly.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Channel-key builders: every loss series is identified by a short string.
# ---------------------------------------------------------------------------

# Known node-IDs that produce loss values.
_NODE_CHANNEL_KEYS: Dict[str, str] = {
    "stage_0_pregestation":    "preg",
    "stage_1_gestation":      "gest",
    "stage_2_berkeley":       "berk",
    "stage_fake_feedback":    "fakefb",
    "gate_0_pregestation_eval": "preg_eval",
    "gate_1_gestation_eval":  "gest_eval",
    "gate_berkeley":          "berk_eval",
    "stage_r_transformer":    "trans",
    "stage_g_generator":      "gen",
    "stage_w_wave_classifier": "wcls",
}


def _channel_key_for_node(node_id: str) -> Optional[str]:
    key = str(node_id or "").strip().lower()
    return _NODE_CHANNEL_KEYS.get(key)


# Metric (stage_label, metric_key) → channel key.
_METRIC_CHANNEL_KEYS: Dict[Tuple[str, str], str] = {
    ("stage0", "loss"): "preg",
    ("stage1", "loss"): "gest",
    ("stage2", "loss"): "berk",
    ("stagefake", "loss"): "fakefb",
    ("gate0", "loss"):  "preg_eval",
    ("gate1", "loss"):  "gest_eval",
    ("gate2", "loss"):  "berk_eval",
    ("stageg", "g_loss"): "gen",
    ("stageg", "d_loss"): "disc",
    ("stager", "loss"):   "trans",
    ("stagew", "loss"):       "wcls",
    ("stagew", "train_loss"): "wcls",
    ("stagew", "val_loss"):   "wcls_eval",
}


def _channel_key_for_metric(stage: str, key: str) -> Optional[str]:
    stage_key = str(stage or "").strip().lower()
    metric_key = str(key or "").strip().lower()
    return _METRIC_CHANNEL_KEYS.get((stage_key, metric_key))


# ---------------------------------------------------------------------------
# Signal store — simple read/write numeric state atoms
# ---------------------------------------------------------------------------

class SignalStore:
    """Runtime registry of named numeric signals.

    Each signal holds a single ``float`` value.  Condition expressions
    access the store through the normal dotted-accessor path on the
    ``PipelineContext``::

        ctx.signals.gate_pregestation_passed   # → 0.0 or 1.0

    Attribute-style reads are supported so condition_expr can resolve
    ``signals.xxx`` without special-casing.  Writes go through
    :meth:`write` (or the ``[]`` operator).
    """

    __slots__ = ("_values",)

    def __init__(self) -> None:
        object.__setattr__(self, "_values", {})

    # -- read --
    def read(self, signal_id: str) -> float:
        return self._values.get(str(signal_id), 0.0)

    def __getattr__(self, name: str) -> float:
        if name.startswith("_"):
            raise AttributeError(name)
        return self._values.get(name, 0.0)

    def __getitem__(self, key: str) -> float:
        return self._values.get(str(key), 0.0)

    # -- write --
    def write(self, signal_id: str, value: float) -> None:
        self._values[str(signal_id)] = float(value)

    def __setitem__(self, key: str, value: float) -> None:
        self._values[str(key)] = float(value)

    # -- bulk --
    def snapshot(self) -> Dict[str, float]:
        return dict(self._values)

    def load(self, values: Dict[str, float]) -> None:
        for k, v in values.items():
            self._values[str(k)] = float(v)

    def init_from_records(self, records: Any) -> None:
        """Seed initial values from a list of ``SignalRecord``."""
        for rec in records:
            sid = str(getattr(rec, "signal_id", ""))
            if sid:
                self._values.setdefault(sid, float(getattr(rec, "initial_value", 0.0)))

    def __repr__(self) -> str:
        pairs = ", ".join(f"{k}={v}" for k, v in sorted(self._values.items()))
        return f"SignalStore({pairs})"

    def __len__(self) -> int:
        return len(self._values)

    def __contains__(self, key: str) -> bool:
        return str(key) in self._values


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
    non_training_device: Optional[torch.device] = None
    non_training_device_preference: str = "auto"
    output_dir: Optional[Path] = None
    raise_on_node_failure: bool = True

    # ---- signals --------------------------------------------------------
    #  Simple read/write numeric atoms.  Condition expressions and predicate
    #  graphs reference signals by ID.  Nodes write at runtime.
    signals: SignalStore = field(default_factory=SignalStore)

    # ---- predicate graphs -----------------------------------------------
    #  Registry of live PredicateGraph objects keyed by graph_id.
    #  Populated from the plan at startup; edges look up their graph here.
    predicate_graphs: Dict[str, Any] = field(default_factory=dict)

    # ---- AMP ------------------------------------------------------------
    amp_enabled: bool = False
    amp_dtype: Optional[torch.dtype] = None

    # ---- GPU residence management ---------------------------------------
    #  Populated by the orchestrator when --stage-module-offload is active.
    #  Nodes use ``ctx.gpu_residence.require(model, name, device)`` to
    #  guarantee a model is on the GPU inside a managed context-manager.
    gpu_residence: Optional[Any] = None  # GPUResidenceManager (from pipeline.nodes.base)
    semantic_tensor_workload: Optional[Any] = None

    # ---- models ---------------------------------------------------------
    #  Each model slot is populated by its node's first execution.

    # Main semantic classifier (TinyConvClassifier)
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
    pregestation_eval_loader: Optional[Any] = None
    pregestation_eval_dataset: Optional[Any] = None

    # Stage 1 – gestation (bootstrap primitive symbol images)
    gestation_loader: Optional[Any] = None
    gestation_dataset: Optional[Any] = None
    gestation_eval_loader: Optional[Any] = None
    gestation_eval_dataset: Optional[Any] = None

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
    vocab_lora_library: Dict[str, Any] = field(default_factory=dict)
    vocab_lora_plan_cache: Dict[str, Any] = field(default_factory=dict)
    vocab_lora_requirement_history: List[Dict[str, Any]] = field(default_factory=list)
    vocab_lora_pending_terms: List[str] = field(default_factory=list)
    vocab_lora_active_signature: str = ""
    vocab_lora_active_terms: List[str] = field(default_factory=list)
    vocab_lora_max_terms: int = 0
    vocab_lora_latest_plan_signature: str = ""
    vocab_lora_plan_slot_cursor: int = 0

    # ---- metrics / history ----------------------------------------------
    metrics_history: List[Dict[str, Any]] = field(default_factory=list)
    round_rows: List[Dict[str, Any]] = field(default_factory=list)
    config_search_history: List[Dict[str, Any]] = field(default_factory=list)

    # ---- training schedule (GUI plan builder) ----------------------------
    #  When a TrainingSchedule is active, the orchestrator walks its rows
    #  sequentially.  Each row specifies cycles, rounds, active stages, and
    #  per-node config overrides.  schedule_index = -1 means pre-start.
    schedule: Optional[Any] = None              # TrainingSchedule | None
    schedule_index: int = -1                    # -1 = pre-start, 0+ = active row
    schedule_row_label: str = ""
    schedule_active_stages: Optional[Any] = None  # frozenset[str] | None (None = all)

    # ---- per-node config overrides (set from schedule row) ---------------
    #  Maps node_id → {field_name: value}.  Applied by each node's execute()
    #  before running.  Cleared and replaced at the start of each schedule row.
    node_config_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # ---- orchestration loop state ---------------------------------------
    cycle: int = 0
    round_id: int = 0
    total_rounds_completed: int = 0
    vocab_rotation_cycle: int = 0
    vocab_lora_plan_registered_cycle: int = -1  # orchestrator cycle when plan_signature was last set
    vocab_lora_plan_registered_round: int = -1   # round_id when plan_signature was last set
    vocab_churn_activation_pending: bool = False  # set by activating sources; consumed+cleared by VocabChurnNode
    last_node_statuses: Dict[str, str] = field(default_factory=dict)
    last_execution_trace: List[Dict[str, Any]] = field(default_factory=list)
    last_program_trace: List[Dict[str, Any]] = field(default_factory=list)
    edge_reaction_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    graph_layers: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # ---- orchestration configuration (plan-driven) ----------------------
    # These mirror the top-level CLI flags; stored here so nodes can read
    # them without needing to fall back to ctx.args in plan-driven mode.
    orchestration_mode: str = "staged_cgrw"
    orchestration_cycles: int = 1
    orchestration_rounds: int = 1

    # ---- cycle gates (IR-driven) ----------------------------------------
    #  Populated by the orchestrator from plan cycle edges via CycleGate.from_plan().
    #  The interpreter consults these at hold points for repeat/exhaust decisions.
    cycle_gates: List[Any] = field(default_factory=list)

    # ---- graph node references (used by condition_expr) ------------------
    data: Optional[Any] = None           # DataNode instance
    preg_cfg: Optional[Any] = None       # PregestationDataConfig
    gest_cfg: Optional[Any] = None       # GestationDataConfig
    bdata_cfg: Optional[Any] = None      # BerkeleyDataConfig

    # ---- viewer / preview -----------------------------------------------
    viewer_proxy: Optional[Any] = None
    loss_logger: Optional[Any] = None
    save_restore_node: Optional[Any] = None

    # ---- checkpoint metadata --------------------------------------------
    checkpoint_path: Optional[Path] = None
    resumed_from_checkpoint: bool = False
    resume_dir: Optional[Path] = None
    resume_summary: Optional[Dict[str, Any]] = None
    resume_pipeline_ckpt: Optional[Dict[str, Any]] = None
    startup_restore_pending: bool = False
    startup_restore_round: int = 0
    startup_restore_cycle: int = 0
    shutdown_save_pending: bool = False
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
        """True once Stage 0, 1, and 2 have all effectively cleared."""
        return (
            self.gate_effectively_passed("gate_pregestation")
            and self.gate_effectively_passed("gate_gestation")
            and self.gate_effectively_passed("gate_berkeley")
        )

    def early_gates_passed(self) -> bool:
        """True when Stage 0 and Stage 1 gates are both effectively passed."""
        return (
            self.gate_effectively_passed("gate_pregestation")
            and self.gate_effectively_passed("gate_gestation")
        )

    def stage_2_berkeley_selected(self) -> bool:
        """True when stage_2_berkeley is selected in the GUI (or no GUI is attached)."""
        return self.is_node_selected("stage_2_berkeley")

    # Mapping from gate context-attribute name → the training stage node_id whose
    # execution produces that gate's value.  Used by is_gate_bypassed() and
    # gate_effectively_passed().
    _GATE_PREREQUISITE_STAGE: ClassVar[Dict[str, str]] = {
        "gate_pregestation": "stage_0_pregestation",
        "gate_gestation":    "stage_1_gestation",
        "gate_berkeley":     "stage_2_berkeley",
        "gate_transformer":  "stage_r_transformer",
        "gate_generator":    "stage_g_generator",
        "gate_wave":         "stage_w_wave_classifier",
    }

    def is_gate_bypassed(self, gate_attr: str) -> bool:
        """True when this gate's result should be treated as satisfied without
        the gate having actually passed:

        - gate_override is active (explicit GUI override), OR
        - the training stage that produces this gate is deselected (user skipped
          the prerequisite — implicitly accepting the gate as met).
        """
        if self.gate_override_enabled():
            return True
        stage_id = self._GATE_PREREQUISITE_STAGE.get(str(gate_attr))
        if stage_id is not None:
            try:
                return not bool(self.is_node_selected(stage_id))
            except Exception:
                pass
        return False

    def gate_effectively_passed(self, gate_attr: str) -> bool:
        """Canonical single-gate satisfaction check.

        A gate is effectively passed when it has actually passed OR is bypassed
        (gate_override active, or the prerequisite training stage is deselected).
        This is the single source of truth — all composite gate predicates should
        compose from this method.
        """
        gate = getattr(self, gate_attr, None)
        if gate is not None and getattr(gate, "passed", False):
            return True
        return self.is_gate_bypassed(gate_attr)

    def generator_and_classifier_ready(self) -> bool:
        return (
            self.all_base_gates_passed()
            and self.generator is not None
            and self.discriminator is not None
        )

    def wave_stage_ready(self) -> bool:
        return (
            self.all_base_gates_passed()
            and self.gate_effectively_passed("gate_transformer")
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

    def is_node_selected(self, node_id: str) -> bool:
        # Schedule-row active_stages is a hard gate: if set and non-empty,
        # only listed nodes may run.  The GUI's per-node toggle can still
        # deselect within those; it cannot add back what the row excluded.
        sas = self.schedule_active_stages
        if sas is not None and len(sas) > 0:
            if str(node_id) not in sas:
                return False
        proxy = self.viewer_proxy
        if proxy is None:
            return True
        fn = getattr(proxy, "is_node_selected", None)
        if not callable(fn):
            return True
        try:
            return bool(fn(str(node_id)))
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

    def consume_skip_forward(self) -> bool:
        proxy = self.viewer_proxy
        if proxy is None:
            return False
        fn = getattr(proxy, "consume_skip_forward", None)
        if not callable(fn):
            return False
        try:
            return bool(fn())
        except Exception:
            return False

    def consume_skip_back(self) -> bool:
        proxy = self.viewer_proxy
        if proxy is None:
            return False
        fn = getattr(proxy, "consume_skip_back", None)
        if not callable(fn):
            return False
        try:
            return bool(fn())
        except Exception:
            return False

    def save_now_requested(self) -> bool:
        proxy = self.viewer_proxy
        if proxy is None:
            return False
        fn = getattr(proxy, "consume_save_now", None)
        if not callable(fn):
            return False
        try:
            return bool(fn())
        except Exception:
            return False

    def shutdown_save(self) -> bool:
        """Return True when the GUI requested stop-with-save (default True)."""
        proxy = self.viewer_proxy
        if proxy is None:
            return True
        fn = getattr(proxy, "shutdown_save", None)
        if not callable(fn):
            return True
        try:
            val = fn()
            if val is None:
                return True
            return bool(val)
        except Exception:
            return True

    def publish_loss(self, channel_key: str, loss: float, aux: float = 0.0) -> None:
        ck = str(channel_key)
        value = float(loss)
        aux_value = float(aux)
        now = time.time()

        lora_slot = getattr(self, "lora_active_slot", "")
        if lora_slot:
            ck = f"{ck}|{lora_slot}"

        logger = self.loss_logger
        if logger is not None:
            try:
                logger.log(int(self.round_id), ck, value, aux=aux_value)
                flush = getattr(logger, "flush", None)
                if callable(flush):
                    flush()
            except Exception:
                pass

        # Route to SaveRestoreNode — the single source of truth for loss data.
        # The GUI pulls values from the accumulator via the IPC pull-model;
        # no direct push to the viewer proxy.
        sr = self.save_restore_node
        if sr is not None:
            try:
                sr.record_loss(
                    channel_key=ck, loss=value, aux=aux_value,
                    round_id=int(self.round_id), ts=now,
                )
                proxy = self.viewer_proxy
                if proxy is not None:
                    send_notif = getattr(proxy, "send_notification", None)
                    if callable(send_notif):
                        send_notif(sr.make_loss_notification(ck))
            except Exception:
                pass

    def publish_node_progress(self, node_id: str, loss: float, aux: float = 0.0) -> None:
        ck = _channel_key_for_node(node_id)
        if ck is None:
            return
        self.publish_loss(ck, float(loss), aux=float(aux))

    def log_metric(self, stage: str, key: str, value: float) -> None:
        value_f = float(value)
        self.metrics_history.append(
            {"cycle": self.cycle, "round": self.round_id, "stage": stage, "key": key, "value": value_f}
        )
        ck = _channel_key_for_metric(stage, key)
        if ck is not None:
            self.publish_loss(ck, value_f)

    # ------------------------------------------------------------------
    # Runtime controls (Pause / Preview / Scrub Editor)
    # ------------------------------------------------------------------

    def paused(self) -> bool:
        """True when the GUI has requested a pause (graph should spin-wait)."""
        proxy = self.viewer_proxy
        if proxy is None:
            return False
        fn = getattr(proxy, "paused", None)
        if not callable(fn):
            return False
        try:
            return bool(fn())
        except Exception:
            return False

    def preview_enabled(self) -> bool:
        """True when preview image generation is allowed."""
        proxy = self.viewer_proxy
        if proxy is None:
            return True
        fn = getattr(proxy, "preview_enabled", None)
        if not callable(fn):
            return True
        try:
            return bool(fn())
        except Exception:
            return True

    def scrub_editor_enabled(self) -> bool:
        """True when the scrub editor (history retention / checkpointing) is on."""
        proxy = self.viewer_proxy
        if proxy is None:
            return True
        fn = getattr(proxy, "scrub_editor_enabled", None)
        if not callable(fn):
            return True
        try:
            return bool(fn())
        except Exception:
            return True
