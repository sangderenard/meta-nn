"""
SaveRestoreNode — unified checkpoint save / restore / seed-bank / replay node.

Consolidates all persistence responsibilities:
  1. **Checkpoint saving** — periodic model + optimizer state persistence
     (replaces the former CheckpointSaveNode in gate_nodes.py).
  2. **Seed bank** — deterministic seed management for reproducible replay.
     Records the RNG state *before* every training round so that a restore
     can re-seed identically.
  3. **Training material cache** — per-channel (model+weights/lora)
     recording of the training inputs that were consumed since the last
     checkpoint.  Retired to disk when the in-memory budget is exceeded.
  4. **Restore replay** — given a checkpoint, re-seeds the RNG and replays
     the cached training material sequence so the model arrives at exactly
     the same parameter state as the moment the user scrubbed back to.

Channel keys
~~~~~~~~~~~~
Every loss channel and training-material stream is keyed by a canonical
string: ``<node_id>`` or ``<node_id>|<lora_slot>``.  This tag is carried
through the cache so each model+weights configuration has its own
independent history.
"""
from __future__ import annotations

import gc
import json
import math
import os
import random
import shutil
import struct
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from pipeline.context import PipelineContext
from pipeline.graph import PipelineNode
from pipeline.nodes.base import _save_pipeline_checkpoint


def _log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# IPC Query / Response protocol constants
# ---------------------------------------------------------------------------
# These message types flow over the existing viewer IPC connection.
# The GUI sends queries; the training side (via ViewerIPCProxy → SaveRestoreNode)
# produces responses.  Lightweight tick-notifications travel training → GUI to
# signal that new data is available without pushing the data itself.

# Query types (GUI → Training)
QUERY_CHANNEL_LIST = "query_channel_list"
QUERY_LOSS_HISTORY = "query_loss_history"
QUERY_LATEST_RESULT = "query_latest_result"
QUERY_TM_SUMMARY = "query_tm_summary"
QUERY_CHECKPOINT_INFO = "query_checkpoint_info"
QUERY_WEIGHT_MAP = "query_weight_map"
QUERY_CACHE_BROWSE = "query_cache_browse"

# Response types (Training → GUI)
RESP_CHANNEL_LIST = "resp_channel_list"
RESP_LOSS_HISTORY = "resp_loss_history"
RESP_LATEST_RESULT = "resp_latest_result"
RESP_TM_SUMMARY = "resp_tm_summary"
RESP_CHECKPOINT_INFO = "resp_checkpoint_info"
RESP_WEIGHT_MAP = "resp_weight_map"
RESP_CACHE_BROWSE = "resp_cache_browse"

# Notification types (Training → GUI, lightweight)
NOTIFY_NEW_LOSS = "notify_new_loss"
NOTIFY_NEW_RESULT = "notify_new_result"
NOTIFY_CHECKPOINT = "notify_checkpoint"
NOTIFY_WEIGHT_MAP = "notify_weight_map"


# ---------------------------------------------------------------------------
# Seed Bank
# ---------------------------------------------------------------------------

@dataclass
class SeedSnapshot:
    """Captures the full RNG state at a point in time."""
    round_id: int = 0
    cycle: int = 0
    python_state: Any = None      # random.getstate()
    numpy_state: Any = None       # np.random.get_state()
    torch_cpu_state: Any = None   # torch.random.get_rng_state()
    torch_cuda_states: Optional[Dict[int, Any]] = None  # {device_ordinal: state}

    def capture(self, round_id: int = 0, cycle: int = 0) -> "SeedSnapshot":
        self.round_id = int(round_id)
        self.cycle = int(cycle)
        self.python_state = random.getstate()
        self.numpy_state = np.random.get_state()
        self.torch_cpu_state = torch.random.get_rng_state()
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            self.torch_cuda_states = {
                i: torch.cuda.get_rng_state(i) for i in range(n)
            }
        return self

    def restore(self) -> None:
        """Re-seed all RNG engines to the captured state."""
        if self.python_state is not None:
            random.setstate(self.python_state)
        if self.numpy_state is not None:
            np.random.set_state(self.numpy_state)
        if self.torch_cpu_state is not None:
            torch.random.set_rng_state(self.torch_cpu_state)
        if self.torch_cuda_states is not None and torch.cuda.is_available():
            for dev_idx, state in self.torch_cuda_states.items():
                try:
                    torch.cuda.set_rng_state(state, int(dev_idx))
                except Exception:
                    pass

    def to_saveable(self) -> Dict[str, Any]:
        """Serialise to a dict that torch.save can handle."""
        blob: Dict[str, Any] = {
            "round_id": self.round_id,
            "cycle": self.cycle,
            "python_state": self.python_state,
        }
        # numpy state is a tuple (str, ndarray, int, int, float) — keep as-is
        if self.numpy_state is not None:
            blob["numpy_state"] = self.numpy_state
        if self.torch_cpu_state is not None:
            blob["torch_cpu_state"] = self.torch_cpu_state
        if self.torch_cuda_states is not None:
            blob["torch_cuda_states"] = {
                int(k): v for k, v in self.torch_cuda_states.items()
            }
        return blob

    @classmethod
    def from_saveable(cls, blob: Dict[str, Any]) -> "SeedSnapshot":
        ss = cls()
        ss.round_id = int(blob.get("round_id", 0))
        ss.cycle = int(blob.get("cycle", 0))
        ss.python_state = blob.get("python_state")
        ss.numpy_state = blob.get("numpy_state")
        ss.torch_cpu_state = blob.get("torch_cpu_state")
        cuda = blob.get("torch_cuda_states")
        if isinstance(cuda, dict):
            ss.torch_cuda_states = {int(k): v for k, v in cuda.items()}
        return ss


class SeedBank:
    """Ring buffer of per-round RNG snapshots.

    Keeps the most recent ``maxlen`` snapshots in memory and optionally
    persists to a directory for post-crash recovery.
    """

    def __init__(self, maxlen: int = 256, persist_dir: Optional[Path] = None):
        self._maxlen = max(1, int(maxlen))
        self._ring: deque[SeedSnapshot] = deque(maxlen=self._maxlen)
        self._persist_dir = persist_dir
        if persist_dir is not None:
            Path(persist_dir).mkdir(parents=True, exist_ok=True)

    def capture(self, round_id: int, cycle: int) -> SeedSnapshot:
        snap = SeedSnapshot().capture(round_id=round_id, cycle=cycle)
        self._ring.append(snap)
        if self._persist_dir is not None:
            try:
                path = self._persist_dir / f"seed_r{round_id:06d}_c{cycle:04d}.pt"
                torch.save(snap.to_saveable(), path)
                self._evict_old_files()
            except Exception as exc:
                _log(f"[seed-bank] persist failed: {exc}")
        return snap

    def latest(self) -> Optional[SeedSnapshot]:
        return self._ring[-1] if self._ring else None

    def find(self, round_id: int, cycle: int) -> Optional[SeedSnapshot]:
        for snap in reversed(self._ring):
            if snap.round_id == round_id and snap.cycle == cycle:
                return snap
        # Fallback: try loading from disk
        if self._persist_dir is not None:
            path = self._persist_dir / f"seed_r{round_id:06d}_c{cycle:04d}.pt"
            if path.exists():
                try:
                    blob = torch.load(path, map_location="cpu", weights_only=False)
                    return SeedSnapshot.from_saveable(blob)
                except Exception:
                    pass
        return None

    def _evict_old_files(self) -> None:
        if self._persist_dir is None:
            return
        files = sorted(self._persist_dir.glob("seed_r*.pt"))
        while len(files) > self._maxlen * 2:
            try:
                files[0].unlink()
            except Exception:
                pass
            files.pop(0)

    def save_bank(self, path: Path) -> None:
        """Persist the entire ring buffer to a single file."""
        torch.save(
            [snap.to_saveable() for snap in self._ring],
            path,
        )

    def load_bank(self, path: Path) -> None:
        """Restore ring buffer from a previously saved bank file."""
        if not path.exists():
            return
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
            for blob in data:
                self._ring.append(SeedSnapshot.from_saveable(blob))
        except Exception as exc:
            _log(f"[seed-bank] load failed: {exc}")


# ---------------------------------------------------------------------------
# Training Material Cache
# ---------------------------------------------------------------------------

@dataclass
class TrainingMaterialEntry:
    """One training batch/input recorded for potential replay."""
    channel_key: str = ""
    round_id: int = 0
    step: int = 0
    data: Optional[Dict[str, Any]] = None
    disk_path: Optional[Path] = None
    size_bytes: int = 0

    def evict_to_disk(self, base_dir: Path) -> None:
        """Move in-memory data to disk, freeing RAM."""
        if self.data is None:
            return
        dest = base_dir / self.channel_key.replace("|", "_")
        dest.mkdir(parents=True, exist_ok=True)
        fname = f"tm_r{self.round_id:06d}_s{self.step:08d}.pt"
        fpath = dest / fname
        try:
            torch.save(self.data, fpath)
            self.disk_path = fpath
            self.data = None
        except Exception as exc:
            _log(f"[tm-cache] evict failed: {exc}")

    def load_from_disk(self) -> Optional[Dict[str, Any]]:
        if self.data is not None:
            return self.data
        if self.disk_path is not None and self.disk_path.exists():
            try:
                return torch.load(self.disk_path, map_location="cpu", weights_only=False)
            except Exception:
                pass
        return None

    def delete_disk(self) -> None:
        if self.disk_path is not None:
            try:
                self.disk_path.unlink(missing_ok=True)
            except Exception:
                pass
            self.disk_path = None


class TrainingMaterialCache:
    """Per-channel cache of training inputs consumed since the last checkpoint.

    When the in-memory budget is exceeded, oldest entries are retired to disk.
    On checkpoint save, the cache records a manifest so restore-replay can
    reconstruct the exact training sequence.
    """

    def __init__(
        self,
        max_memory_bytes: int = 512 * 1024 * 1024,  # 512 MB default
        max_disk_bytes: int = 2 * 1024 * 1024 * 1024,  # 2 GB default
        base_dir: Optional[Path] = None,
    ):
        self._max_mem = max(0, int(max_memory_bytes))
        self._max_disk = max(0, int(max_disk_bytes))
        self._base_dir = base_dir
        if base_dir is not None:
            Path(base_dir).mkdir(parents=True, exist_ok=True)
        # channel_key → ordered list of entries since last checkpoint
        self._channels: Dict[str, List[TrainingMaterialEntry]] = {}
        self._mem_used: int = 0
        self._disk_used: int = 0

    def record(self, channel_key: str, round_id: int, step: int,
               data: Dict[str, Any]) -> None:
        """Record a training batch for this channel."""
        entry = TrainingMaterialEntry(
            channel_key=str(channel_key),
            round_id=int(round_id),
            step=int(step),
            data=data,
        )
        # Estimate size
        entry.size_bytes = self._estimate_size(data)
        self._mem_used += entry.size_bytes

        if channel_key not in self._channels:
            self._channels[channel_key] = []
        self._channels[channel_key].append(entry)

        # Enforce memory budget
        self._enforce_memory_budget()

    def _estimate_size(self, data: Dict[str, Any]) -> int:
        total = 0
        for v in data.values():
            if isinstance(v, torch.Tensor):
                total += v.nelement() * v.element_size()
            elif isinstance(v, np.ndarray):
                total += v.nbytes
            else:
                total += 64  # rough estimate for scalars/strings
        return max(64, total)

    def _enforce_memory_budget(self) -> None:
        """Evict oldest in-memory entries to disk until under budget."""
        if self._base_dir is None or self._mem_used <= self._max_mem:
            return
        # Collect all in-memory entries across channels, sort by age
        all_entries: List[TrainingMaterialEntry] = []
        for entries in self._channels.values():
            for e in entries:
                if e.data is not None:
                    all_entries.append(e)
        # Sort by (round_id, step) ascending — oldest first
        all_entries.sort(key=lambda e: (e.round_id, e.step))

        for entry in all_entries:
            if self._mem_used <= self._max_mem:
                break
            sz = entry.size_bytes
            entry.evict_to_disk(self._base_dir)
            self._mem_used -= sz
            self._disk_used += sz

        # Enforce disk budget: delete oldest disk entries
        self._enforce_disk_budget()

    def _enforce_disk_budget(self) -> None:
        if self._disk_used <= self._max_disk:
            return
        for entries in self._channels.values():
            for entry in list(entries):
                if self._disk_used <= self._max_disk:
                    return
                if entry.disk_path is not None:
                    self._disk_used -= entry.size_bytes
                    entry.delete_disk()
                    entries.remove(entry)

    def clear_before_checkpoint(self) -> None:
        """Drop all entries — called after a checkpoint has been saved.

        The checkpoint itself contains the model state, so the cache only
        needs material *since* the last checkpoint for replay.
        """
        for entries in self._channels.values():
            for entry in entries:
                entry.delete_disk()
        self._channels.clear()
        self._mem_used = 0
        self._disk_used = 0

    def save_manifest(self, path: Path) -> None:
        """Write a JSON manifest listing all cached entries per channel."""
        manifest: Dict[str, List[Dict[str, Any]]] = {}
        for ck, entries in self._channels.items():
            manifest[ck] = [
                {
                    "round_id": e.round_id,
                    "step": e.step,
                    "has_data": e.data is not None,
                    "disk_path": str(e.disk_path) if e.disk_path else None,
                    "size_bytes": e.size_bytes,
                }
                for e in entries
            ]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def load_manifest(self, path: Path) -> None:
        """Reload entry metadata from a manifest (data will need disk load)."""
        if not path.exists():
            return
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        for ck, entries_data in manifest.items():
            self._channels.setdefault(ck, [])
            for ed in entries_data:
                dp = ed.get("disk_path")
                entry = TrainingMaterialEntry(
                    channel_key=ck,
                    round_id=int(ed.get("round_id", 0)),
                    step=int(ed.get("step", 0)),
                    disk_path=Path(dp) if dp else None,
                    size_bytes=int(ed.get("size_bytes", 0)),
                )
                self._channels[ck].append(entry)
                if entry.disk_path is not None:
                    self._disk_used += entry.size_bytes

    def replay_entries(self, channel_key: str) -> List[Dict[str, Any]]:
        """Yield all cached training batches for a channel in order.

        Loads from disk as needed.\n        """
        result: List[Dict[str, Any]] = []
        for entry in self._channels.get(channel_key, []):
            data = entry.data if entry.data is not None else entry.load_from_disk()
            if data is not None:
                result.append(data)
        return result

    def channel_keys(self) -> List[str]:
        return list(self._channels.keys())

    def channel_entry_count(self, channel_key: str) -> int:
        return len(self._channels.get(channel_key, []))

    def total_entries(self) -> int:
        return sum(len(v) for v in self._channels.values())


# ---------------------------------------------------------------------------
# Loss Accumulator — per-channel loss history (replaces viewer-side storage)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class LossRecord:
    step: int = 0
    round_id: int = 0
    loss: float = 0.0
    aux: float = 0.0
    ts: float = 0.0


class LossAccumulator:
    """Thread-safe per-channel loss history.

    The training loop calls ``record()`` on every step.  The GUI pulls
    data via ``query_since()`` specifying the cursor (step index) it last
    saw, receiving only the new records.
    """

    def __init__(self, maxlen: int = 100_000):
        self._maxlen = max(1, int(maxlen))
        self._channels: Dict[str, deque] = {}  # channel_key → deque[LossRecord]
        self._lock = threading.Lock()
        # Monotonic step counter per channel — always increases, used as cursor.
        self._step_counters: Dict[str, int] = {}

    def record(self, channel_key: str, loss: float, aux: float = 0.0,
               round_id: int = 0, ts: float = 0.0) -> int:
        """Append a loss record.  Returns the step index assigned."""
        ck = str(channel_key)
        with self._lock:
            if ck not in self._channels:
                self._channels[ck] = deque(maxlen=self._maxlen)
                self._step_counters[ck] = 0
            step = self._step_counters[ck]
            self._step_counters[ck] = step + 1
            v = float(loss) if math.isfinite(float(loss)) else float("nan")
            rec = LossRecord(
                step=step,
                round_id=int(round_id),
                loss=v,
                aux=float(aux),
                ts=float(ts) if float(ts) > 0.0 else time.time(),
            )
            self._channels[ck].append(rec)
        return step

    def channel_keys(self) -> List[str]:
        with self._lock:
            return list(self._channels.keys())

    def channel_length(self, channel_key: str) -> int:
        with self._lock:
            dq = self._channels.get(str(channel_key))
            return len(dq) if dq is not None else 0

    def query_since(self, channel_key: str, from_step: int = 0) -> List[Dict[str, Any]]:
        """Return loss records with step >= from_step as lightweight dicts."""
        ck = str(channel_key)
        result: List[Dict[str, Any]] = []
        with self._lock:
            dq = self._channels.get(ck)
            if dq is None:
                return result
            for rec in dq:
                if rec.step >= int(from_step):
                    result.append({
                        "step": rec.step,
                        "round_id": rec.round_id,
                        "loss": rec.loss,
                        "aux": rec.aux,
                        "ts": rec.ts,
                    })
        return result

    def query_all(self, channel_key: str) -> List[Dict[str, Any]]:
        return self.query_since(channel_key, from_step=0)

    def latest(self, channel_key: str) -> Optional[Dict[str, Any]]:
        ck = str(channel_key)
        with self._lock:
            dq = self._channels.get(ck)
            if dq and len(dq) > 0:
                rec = dq[-1]
                return {"step": rec.step, "round_id": rec.round_id,
                        "loss": rec.loss, "aux": rec.aux, "ts": rec.ts}
        return None

    def summary(self) -> Dict[str, Dict[str, Any]]:
        """Per-channel summary: length, latest loss, step cursor."""
        with self._lock:
            out: Dict[str, Dict[str, Any]] = {}
            for ck, dq in self._channels.items():
                last = dq[-1] if dq else None
                out[ck] = {
                    "length": len(dq),
                    "cursor": self._step_counters.get(ck, 0),
                    "latest_loss": last.loss if last else None,
                    "latest_step": last.step if last else None,
                }
            return out

    def clear(self) -> None:
        with self._lock:
            self._channels.clear()
            self._step_counters.clear()


# ---------------------------------------------------------------------------
# Result Store — detached training outputs for GUI pull
# ---------------------------------------------------------------------------

class ResultStore:
    """Retains the most recent N detached training results per channel.

    Training nodes call ``store()`` with output tensors/images.  All tensors
    are ``.detach().cpu()``-ed so no gradient graph is retained.  The GUI
    queries for the latest result per channel when it wants to refresh its
    display panels.
    """

    def __init__(self, maxlen_per_channel: int = 16):
        self._maxlen = max(1, int(maxlen_per_channel))
        self._channels: Dict[str, deque] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _detach_value(v: Any) -> Any:
        if isinstance(v, torch.Tensor):
            return v.detach().cpu()
        if isinstance(v, np.ndarray):
            return v.copy()
        if isinstance(v, dict):
            return {k: ResultStore._detach_value(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return type(v)(ResultStore._detach_value(x) for x in v)
        return v

    def store(self, channel_key: str, result: Dict[str, Any],
              round_id: int = 0, step: int = 0) -> None:
        """Store a detached copy of runtime training output."""
        ck = str(channel_key)
        detached = {
            "_round_id": int(round_id),
            "_step": int(step),
            "_ts": time.time(),
        }
        for k, v in result.items():
            detached[k] = self._detach_value(v)
        with self._lock:
            if ck not in self._channels:
                self._channels[ck] = deque(maxlen=self._maxlen)
            self._channels[ck].append(detached)

    def latest(self, channel_key: str) -> Optional[Dict[str, Any]]:
        ck = str(channel_key)
        with self._lock:
            dq = self._channels.get(ck)
            if dq and len(dq) > 0:
                return dq[-1]
        return None

    def latest_serialisable(self, channel_key: str) -> Optional[Dict[str, Any]]:
        """Return the latest result in a form that can be pickled over IPC.

        Tensors are converted to numpy; everything else is kept as-is.
        """
        raw = self.latest(channel_key)
        if raw is None:
            return None
        out: Dict[str, Any] = {}
        for k, v in raw.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.numpy()
            else:
                out[k] = v
        return out

    def channel_keys(self) -> List[str]:
        with self._lock:
            return list(self._channels.keys())

    def channel_depth(self, channel_key: str) -> int:
        with self._lock:
            dq = self._channels.get(str(channel_key))
            return len(dq) if dq else 0


# ---------------------------------------------------------------------------
# Weight Tracker
# ---------------------------------------------------------------------------

class WeightTracker:
    """Captures detached model weight snapshots and renders weight-map images.

    Maintains a base tensor (initial weights at load time) and the latest
    snapshot.  The difference ``|current − base|`` is rendered as an RGB
    image suitable for the viewer's right-panel weight display.

    Thread-safe: snapshot capture and image rendering can happen from any
    thread.
    """

    def __init__(self, image_hw: Tuple[int, int] = (128, 128)) -> None:
        self._lock = threading.Lock()
        self._image_h, self._image_w = max(8, int(image_hw[0])), max(8, int(image_hw[1]))
        # Per-model storage: model_name → {base: flat_tensor, latest: flat_tensor}
        self._models: Dict[str, Dict[str, torch.Tensor]] = {}
        self._active_model: Optional[str] = None

    def register_base(self, model_name: str, model: nn.Module) -> None:
        """Capture base (initial) weights for a model.  Called once at init."""
        flat = self._flatten_params(model)
        with self._lock:
            entry = self._models.setdefault(model_name, {})
            entry["base"] = flat
            if self._active_model is None:
                self._active_model = model_name

    def capture(self, model_name: str, model: nn.Module) -> None:
        """Capture the current weight state (detached, CPU)."""
        flat = self._flatten_params(model)
        with self._lock:
            entry = self._models.setdefault(model_name, {})
            entry["latest"] = flat

    def set_active(self, model_name: str) -> None:
        with self._lock:
            self._active_model = model_name

    def model_names(self) -> List[str]:
        with self._lock:
            return list(self._models.keys())

    def render_weight_map_rgb(self, model_name: Optional[str] = None) -> Optional[np.ndarray]:
        """Render an (H, W, 3) uint8 RGB image of |current − base| for the named model.

        Returns None if base or latest is not yet captured.
        The delta magnitude is normalised to [0,255] and mapped to a heat
        palette (blue → red → yellow).
        """
        with self._lock:
            name = model_name or self._active_model
            if name is None or name not in self._models:
                return None
            entry = self._models[name]
            base = entry.get("base")
            latest = entry.get("latest")
        if base is None or latest is None:
            return None
        n = min(len(base), len(latest))
        if n == 0:
            return None
        delta = (latest[:n] - base[:n]).abs()
        # Normalise to [0, 1]
        dmax = delta.max().item()
        if dmax < 1e-12:
            normed = torch.zeros(n)
        else:
            normed = delta / dmax
        # Reshape to (H, W) by padding/truncating and wrapping
        total_px = self._image_h * self._image_w
        if n < total_px:
            padded = torch.zeros(total_px)
            padded[:n] = normed
        else:
            padded = normed[:total_px]
        grid = padded.reshape(self._image_h, self._image_w).numpy()
        # Heat palette: 0 → dark blue, 0.5 → red, 1.0 → yellow
        rgb = np.zeros((self._image_h, self._image_w, 3), dtype=np.uint8)
        r = np.clip(grid * 2.0, 0.0, 1.0)
        g = np.clip(grid * 2.0 - 1.0, 0.0, 1.0)
        b = np.clip(1.0 - grid * 2.0, 0.0, 1.0)
        rgb[:, :, 0] = (r * 255).astype(np.uint8)
        rgb[:, :, 1] = (g * 255).astype(np.uint8)
        rgb[:, :, 2] = (b * 255).astype(np.uint8)
        return rgb

    def render_serialisable(self, model_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Return the weight map as a serialisable dict for IPC transfer."""
        rgb = self.render_weight_map_rgb(model_name)
        if rgb is None:
            return None
        return {
            "model": model_name or self._active_model,
            "height": rgb.shape[0],
            "width": rgb.shape[1],
            "rgb": rgb.tolist(),
        }

    @staticmethod
    def _flatten_params(model: nn.Module) -> torch.Tensor:
        """Flatten all model parameters into a single detached CPU tensor."""
        parts = []
        for p in model.parameters():
            parts.append(p.detach().cpu().float().reshape(-1))
        if not parts:
            return torch.zeros(0)
        return torch.cat(parts)


# ---------------------------------------------------------------------------
# The Node
# ---------------------------------------------------------------------------

class SaveRestoreNode(PipelineNode):
    """Unified checkpoint save / restore / seed-bank / replay node.

    Replaces the former CheckpointSaveNode with expanded responsibilities:

    **Save mode** (default, runs at end of every Nth round):
      - Captures RNG seed state into the seed bank
      - Persists all model weights + optimizer states
      - Saves the training material cache manifest
      - Notifies the viewer of the checkpoint marker

    **Restore mode** (triggered by viewer RESTORE STATE or programmatic call):
      - Loads the target checkpoint weights
      - Restores the seed bank state to that checkpoint's RNG snapshot
      - Replays the training material cache to bring the model to the
        exact state at the restoration point
      - Returns control to the orchestrator

    The node_id remains ``checkpoint_save`` for graph compatibility.
    """

    node_id = "checkpoint_save"
    description = "Save/restore checkpoints with seed bank and training replay"
    runtime_object_type = "service"
    runtime_faculty = "persistence"

    def __init__(
        self,
        save_every_n_rounds: int = 1,
        seed_bank_maxlen: int = 256,
        tm_max_memory_mb: int = 512,
        tm_max_disk_mb: int = 2048,
    ) -> None:
        self.save_every_n_rounds = max(1, int(save_every_n_rounds))
        self._seed_bank_maxlen = max(1, int(seed_bank_maxlen))
        self._tm_max_memory = int(tm_max_memory_mb) * 1024 * 1024
        self._tm_max_disk = int(tm_max_disk_mb) * 1024 * 1024

        self.seed_bank: Optional[SeedBank] = None
        self.training_cache: Optional[TrainingMaterialCache] = None

        # -- Pull-model data stores (GUI queries these) -------------------
        self.loss_accumulator: LossAccumulator = LossAccumulator()
        self.result_store: ResultStore = ResultStore()

        # Pending restore request: (round_id, cycle) to restore to.
        self._pending_restore: Optional[Tuple[int, int]] = None
        # Replay callbacks: channel_key → callable that accepts a batch dict
        # and performs one training step.  Registered by the orchestrator.
        self._replay_callbacks: Dict[str, Callable[[Dict[str, Any]], None]] = {}
        # Snapshot of the classifier lora helper (injected)
        self._snapshot_lora_fn: Optional[Callable] = None

        # -- Weight tracking (base tensor + per-checkpoint delta image) ----
        self.weight_tracker: WeightTracker = WeightTracker()

    def initialise(self, ctx: PipelineContext) -> None:
        """Called once by the orchestrator after context is set up."""
        out_dir = ctx.output_dir
        if out_dir is None:
            return

        seed_dir = out_dir / "seed_bank"
        self.seed_bank = SeedBank(
            maxlen=self._seed_bank_maxlen,
            persist_dir=seed_dir,
        )
        # Try loading existing bank from checkpoint
        bank_path = out_dir / "seed_bank.pt"
        self.seed_bank.load_bank(bank_path)

        tm_dir = out_dir / "training_material_cache"
        self.training_cache = TrainingMaterialCache(
            max_memory_bytes=self._tm_max_memory,
            max_disk_bytes=self._tm_max_disk,
            base_dir=tm_dir,
        )
        # Try loading existing manifest
        manifest_path = out_dir / "tm_manifest.json"
        self.training_cache.load_manifest(manifest_path)

        # Register base weights for weight-tracking delta image
        _models = {
            "classifier": ctx.classifier,
            "transformer": ctx.transformer,
            "generator": ctx.generator,
            "discriminator": ctx.discriminator,
            "wave_classifier": ctx.wave_classifier,
        }
        for name, model in _models.items():
            if model is not None:
                try:
                    self.weight_tracker.register_base(name, model)
                except Exception:
                    pass

    @property
    def runtime_execution_policy(self) -> tuple:
        return ("periodic", {"period": self.save_every_n_rounds})

    def declare_subnodes(self) -> list:
        return [
            {"subnode_id": "seed_bank", "kind": "state_store", "label": "Seed Bank", "order": 0},
            {"subnode_id": "training_cache", "kind": "state_store", "label": "Training Material Cache", "order": 1},
            {"subnode_id": "loss_accumulator", "kind": "metric_store", "label": "Loss Accumulator", "order": 2},
            {"subnode_id": "result_store", "kind": "metric_store", "label": "Result Store", "order": 3},
            {"subnode_id": "weight_tracker", "kind": "diagnostic", "label": "Weight Tracker", "order": 4},
        ]

    def should_run(self, ctx: PipelineContext) -> bool:
        # Always run if a restore is pending
        if self._pending_restore is not None:
            return True
        return (ctx.round_id % self.save_every_n_rounds) == 0

    def execute(self, ctx: PipelineContext) -> None:
        # Handle pending restore first
        if self._pending_restore is not None:
            self._execute_restore(ctx)
            return

        # Normal save path
        self._execute_save(ctx)

    def _execute_save(self, ctx: PipelineContext) -> None:
        """Save checkpoint: seed snapshot, model weights, cache manifest."""
        out_dir = ctx.output_dir
        if out_dir is None:
            return

        # 1. Capture seed state for this round
        if self.seed_bank is not None:
            self.seed_bank.capture(
                round_id=ctx.round_id,
                cycle=ctx.cycle,
            )

        # 2. Build the checkpoint payload
        snapshot_lora = self._get_lora_snapshot(ctx)

        payload: Dict[str, Any] = {
            "run_tag": ctx.run_tag,
            "objective_mode": str(getattr(ctx.args, "objective_mode", "berkeley_multilabel")),
            "segment": "checkpoint",
            "timestamp": time.time(),
            "cycle": ctx.cycle,
            "round_id": ctx.round_id,
            "total_rounds_completed": ctx.total_rounds_completed,
            "class_names": ctx.class_names,
        }
        if ctx.render_config is not None:
            try:
                payload["best_cfg"] = ctx.render_config.to_dict()
            except Exception:
                pass
        # Gate status
        payload["gate_status"] = {
            "pregestation": ctx.gate_pregestation.passed,
            "gestation": ctx.gate_gestation.passed,
            "berkeley": ctx.gate_berkeley.passed,
            "transformer": ctx.gate_transformer.passed,
            "generator": ctx.gate_generator.passed,
            "wave": ctx.gate_wave.passed,
        }
        if ctx.metrics_history:
            payload["orchestration_history"] = list(ctx.metrics_history)
        # Lora state
        if ctx.lora_slot_snapshots:
            payload["lora_slot_snapshots"] = dict(ctx.lora_slot_snapshots)
        payload["lora_active_slot"] = ctx.lora_active_slot

        # Model state dicts
        _models = {
            "classifier": ctx.classifier,
            "transformer": ctx.transformer,
            "generator": ctx.generator,
            "discriminator": ctx.discriminator,
            "wave_classifier": ctx.wave_classifier,
        }
        for name, model in _models.items():
            if model is not None:
                payload[f"{name}_state"] = model.state_dict()
        if snapshot_lora is not None:
            payload["classifier_lora"] = snapshot_lora

        # 3. Atomic write
        _save_pipeline_checkpoint(out_dir / "pipeline_checkpoint.pt", payload)

        # Write per-model files
        for name, model in _models.items():
            if model is not None:
                extra = {}
                if name == "classifier" and snapshot_lora is not None:
                    extra["classifier_lora"] = snapshot_lora
                torch.save(
                    {"state_dict": model.state_dict(), **extra},
                    out_dir / f"{name}.pt",
                )

        # 4. Save seed bank
        if self.seed_bank is not None:
            self.seed_bank.save_bank(out_dir / "seed_bank.pt")

        # 5. Save training material manifest and clear for next segment
        if self.training_cache is not None:
            self.training_cache.save_manifest(out_dir / "tm_manifest.json")
            self.training_cache.clear_before_checkpoint()

        # 6. Notify viewer (lightweight notification only — GUI pulls details)
        viewer = ctx.viewer_proxy
        if viewer is not None:
            notification = self.make_checkpoint_notification(ctx.round_id, ctx.cycle)
            send_fn = getattr(viewer, "_send", None)
            if callable(send_fn):
                try:
                    send_fn(notification)
                except Exception:
                    pass

        # 7. Capture weight snapshots for delta rendering
        for name, model in _models.items():
            if model is not None:
                try:
                    self.weight_tracker.capture(name, model)
                except Exception:
                    pass

        _log(f"[checkpoint] saved round={ctx.round_id} cycle={ctx.cycle}")

    def _execute_restore(self, ctx: PipelineContext) -> None:
        """Restore to a checkpoint and replay training material."""
        target_round, target_cycle = self._pending_restore
        self._pending_restore = None
        out_dir = ctx.output_dir
        if out_dir is None:
            _log("[restore] ERROR: no output_dir set, cannot restore")
            return

        _log(f"[restore] restoring to round={target_round} cycle={target_cycle}")

        # 1. Load the checkpoint
        ckpt_path = out_dir / "pipeline_checkpoint.pt"
        if not ckpt_path.exists():
            _log("[restore] ERROR: no checkpoint file found")
            return

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            _log(f"[restore] ERROR loading checkpoint: {exc}")
            return

        # 2. Restore model weights
        _models = {
            "classifier": ctx.classifier,
            "transformer": ctx.transformer,
            "generator": ctx.generator,
            "discriminator": ctx.discriminator,
            "wave_classifier": ctx.wave_classifier,
        }
        for name, model in _models.items():
            sd_key = f"{name}_state"
            if model is not None and sd_key in ckpt:
                try:
                    model.load_state_dict(ckpt[sd_key], strict=False)
                    _log(f"[restore] loaded {name} weights")
                except Exception as exc:
                    _log(f"[restore] WARNING: {name} weight load failed: {exc}")

        # 3. Restore seed state
        if self.seed_bank is not None:
            seed_snap = self.seed_bank.find(target_round, target_cycle)
            if seed_snap is not None:
                seed_snap.restore()
                _log(f"[restore] RNG state restored to r={target_round} c={target_cycle}")
            else:
                _log("[restore] WARNING: no seed snapshot found for target, RNG not restored")

        # 4. Replay training material
        if self.training_cache is not None:
            for ck in self.training_cache.channel_keys():
                cb = self._replay_callbacks.get(ck)
                if cb is None:
                    _log(f"[restore] no replay callback for channel {ck!r}, skipping")
                    continue
                batches = self.training_cache.replay_entries(ck)
                if not batches:
                    continue
                _log(f"[restore] replaying {len(batches)} batches on channel {ck!r}")
                for batch_data in batches:
                    try:
                        cb(batch_data)
                    except Exception as exc:
                        _log(f"[restore] replay batch error on {ck!r}: {exc}")
                        break

        # 5. Clear cache after replay (we're now at the replayed state)
        if self.training_cache is not None:
            self.training_cache.clear_before_checkpoint()

        _log(f"[restore] replay complete, control returned to orchestrator")

    def _get_lora_snapshot(self, ctx: PipelineContext) -> Optional[Any]:
        if self._snapshot_lora_fn is not None and ctx.classifier is not None:
            try:
                return self._snapshot_lora_fn(ctx.classifier)
            except Exception:
                pass
        return None

    # -- Public API for external callers ----------------------------------

    def request_restore(self, round_id: int, cycle: int) -> None:
        """Schedule a restore for the next execute() call."""
        self._pending_restore = (int(round_id), int(cycle))

    def register_replay_callback(
        self,
        channel_key: str,
        callback: Callable[[Dict[str, Any]], None],
    ) -> None:
        """Register a function that replays one training batch for a channel.

        The callback receives the same dict that was originally passed to
        ``training_cache.record()``.
        """
        self._replay_callbacks[str(channel_key)] = callback

    def record_training_material(
        self,
        channel_key: str,
        round_id: int,
        step: int,
        data: Dict[str, Any],
    ) -> None:
        """Record a training batch for potential replay."""
        if self.training_cache is not None:
            self.training_cache.record(
                channel_key=str(channel_key),
                round_id=int(round_id),
                step=int(step),
                data=data,
            )

    def capture_seed(self, round_id: int, cycle: int) -> Optional[SeedSnapshot]:
        """Explicitly capture seed state (normally done in _execute_save)."""
        if self.seed_bank is not None:
            return self.seed_bank.capture(round_id, cycle)
        return None

    def set_lora_snapshot_fn(self, fn: Callable) -> None:
        self._snapshot_lora_fn = fn

    # -- Pull-model: record data (called by training nodes) ---------------

    def record_loss(self, channel_key: str, loss: float, aux: float = 0.0,
                    round_id: int = 0, ts: float = 0.0) -> int:
        """Record a loss value.  Returns the step index.

        Also emits a lightweight notification dict (caller can forward
        to the viewer proxy).
        """
        return self.loss_accumulator.record(
            channel_key=channel_key, loss=loss, aux=aux,
            round_id=round_id, ts=ts,
        )

    def store_result(self, channel_key: str, result: Dict[str, Any],
                     round_id: int = 0, step: int = 0) -> None:
        """Store a detached training output for GUI pull."""
        self.result_store.store(
            channel_key=channel_key, result=result,
            round_id=round_id, step=step,
        )

    # -- Pull-model: query handler (called via IPC from GUI) --------------

    def handle_query(self, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Dispatch a GUI query and return a response dict, or None.

        The ViewerIPCProxy calls this when it receives a query_* message
        from the GUI.  The response is sent back over the same connection.
        """
        qtype = query.get("type", "")

        if qtype == QUERY_CHANNEL_LIST:
            return self._resp_channel_list()
        elif qtype == QUERY_LOSS_HISTORY:
            return self._resp_loss_history(query)
        elif qtype == QUERY_LATEST_RESULT:
            return self._resp_latest_result(query)
        elif qtype == QUERY_TM_SUMMARY:
            return self._resp_tm_summary()
        elif qtype == QUERY_CHECKPOINT_INFO:
            return self._resp_checkpoint_info()
        elif qtype == QUERY_WEIGHT_MAP:
            return self._resp_weight_map(query)
        elif qtype == QUERY_CACHE_BROWSE:
            return self._resp_cache_browse(query)
        return None

    def _resp_channel_list(self) -> Dict[str, Any]:
        loss_keys = self.loss_accumulator.channel_keys()
        result_keys = self.result_store.channel_keys()
        tm_keys = self.training_cache.channel_keys() if self.training_cache else []
        all_keys = sorted(set(loss_keys) | set(result_keys) | set(tm_keys))
        return {
            "type": RESP_CHANNEL_LIST,
            "channels": [
                {
                    "key": ck,
                    "has_loss": ck in loss_keys,
                    "loss_length": self.loss_accumulator.channel_length(ck),
                    "has_result": ck in result_keys,
                    "result_depth": self.result_store.channel_depth(ck),
                    "has_tm": ck in tm_keys,
                    "tm_entries": (self.training_cache.channel_entry_count(ck)
                                   if self.training_cache else 0),
                }
                for ck in all_keys
            ],
        }

    def _resp_loss_history(self, query: Dict[str, Any]) -> Dict[str, Any]:
        ck = str(query.get("channel_key", ""))
        from_step = int(query.get("from_step", 0))
        records = self.loss_accumulator.query_since(ck, from_step)
        return {
            "type": RESP_LOSS_HISTORY,
            "channel_key": ck,
            "from_step": from_step,
            "records": records,
        }

    def _resp_latest_result(self, query: Dict[str, Any]) -> Dict[str, Any]:
        ck = str(query.get("channel_key", ""))
        result = self.result_store.latest_serialisable(ck)
        return {
            "type": RESP_LATEST_RESULT,
            "channel_key": ck,
            "result": result,  # None if nothing stored yet
        }

    def _resp_tm_summary(self) -> Dict[str, Any]:
        if self.training_cache is None:
            return {"type": RESP_TM_SUMMARY, "channels": {}, "total": 0}
        keys = self.training_cache.channel_keys()
        channels = {
            ck: self.training_cache.channel_entry_count(ck)
            for ck in keys
        }
        return {
            "type": RESP_TM_SUMMARY,
            "channels": channels,
            "total": self.training_cache.total_entries(),
        }

    def _resp_checkpoint_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "type": RESP_CHECKPOINT_INFO,
            "seed_bank_size": len(self.seed_bank._ring) if self.seed_bank else 0,
            "latest_seed": None,
        }
        if self.seed_bank:
            latest = self.seed_bank.latest()
            if latest:
                info["latest_seed"] = {
                    "round_id": latest.round_id,
                    "cycle": latest.cycle,
                }
        return info

    def _resp_weight_map(self, query: Dict[str, Any]) -> Dict[str, Any]:
        model_name = query.get("model") or None
        data = self.weight_tracker.render_serialisable(model_name)
        return {
            "type": RESP_WEIGHT_MAP,
            "models": self.weight_tracker.model_names(),
            "active": self.weight_tracker._active_model,
            "map": data,
        }

    def _resp_cache_browse(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Return a browsable summary of the training material cache.

        Supports pagination via ``offset`` and ``limit`` query fields,
        and optional ``channel_key`` filtering.
        """
        if self.training_cache is None:
            return {"type": RESP_CACHE_BROWSE, "entries": [], "total": 0}
        ck_filter = query.get("channel_key", "")
        offset = max(0, int(query.get("offset", 0)))
        limit = max(1, min(200, int(query.get("limit", 50))))
        if ck_filter:
            keys = [ck_filter]
        else:
            keys = self.training_cache.channel_keys()
        entries: List[Dict[str, Any]] = []
        for ck in keys:
            items = self.training_cache.replay_entries(ck)
            for idx, item in enumerate(items):
                entries.append({
                    "channel_key": ck,
                    "index": idx,
                    "round_id": item.get("round_id", 0) if isinstance(item, dict) else 0,
                    "step": item.get("step", 0) if isinstance(item, dict) else 0,
                    "keys": list(item.keys()) if isinstance(item, dict) else [],
                })
        total = len(entries)
        page = entries[offset:offset + limit]
        return {"type": RESP_CACHE_BROWSE, "entries": page, "total": total}

    # -- Notification helpers (lightweight, no data payload) ---------------

    def make_loss_notification(self, channel_key: str) -> Dict[str, Any]:
        """Build a lightweight tick notification for the GUI."""
        return {
            "type": NOTIFY_NEW_LOSS,
            "channel_key": str(channel_key),
            "length": self.loss_accumulator.channel_length(channel_key),
        }

    def make_result_notification(self, channel_key: str) -> Dict[str, Any]:
        return {
            "type": NOTIFY_NEW_RESULT,
            "channel_key": str(channel_key),
        }

    def make_checkpoint_notification(self, round_id: int, cycle: int) -> Dict[str, Any]:
        return {
            "type": NOTIFY_CHECKPOINT,
            "round_id": int(round_id),
            "cycle": int(cycle),
            "loss_summary": self.loss_accumulator.summary(),
        }
