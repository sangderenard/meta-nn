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
from pipeline.nodus_loss_store import SCRUB_NUM_THUMBS, SCRUB_THUMB_H, SCRUB_THUMB_W
from pipeline.nodes.base import _save_pipeline_checkpoint
from pipeline.weight_map import (
    annotate_weight_map,
    render_weight_image,
    snapshot_model_parameter_state,
    snapshot_parameter_state_from_state_dict,
)


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


# Weight thumbnails are rendered as compact active-model diagnostics for the
# viewer right panel, so they use a fixed square footprint by default.
_WEIGHT_THUMB_SIDE = max(8, int(os.getenv("WEIGHT_THUMB_SIDE", "256")))
_WEIGHT_THUMB_PNG_COMPRESS_LEVEL = max(
    0, min(9, int(os.getenv("WEIGHT_THUMB_PNG_COMPRESS_LEVEL", "0")))
)
_WEIGHT_LIVE_TILE_STACK_H = SCRUB_THUMB_H * SCRUB_NUM_THUMBS


def _resize_rgb_nearest(rgb: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    out_h = max(1, int(size_hw[0]))
    out_w = max(1, int(size_hw[1]))
    try:
        from PIL import Image

        img = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
        return np.asarray(img.resize((out_w, out_h), Image.NEAREST), dtype=np.uint8)
    except Exception:
        src = np.asarray(rgb, dtype=np.uint8)
        y_idx = np.floor(np.linspace(0, max(0, src.shape[0] - 1), out_h)).astype(np.int64)
        x_idx = np.floor(np.linspace(0, max(0, src.shape[1] - 1), out_w)).astype(np.int64)
        return src[y_idx][:, x_idx]


def _weight_thumb_tiles_from_rgb(rgb: np.ndarray) -> Tuple[np.ndarray, ...]:
    src = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
    if src.ndim != 3 or src.shape[2] != 3:
        raise ValueError("weight thumb source must be an RGB image")

    scale = min(
        float(SCRUB_THUMB_W) / float(max(1, src.shape[1])),
        float(_WEIGHT_LIVE_TILE_STACK_H) / float(max(1, src.shape[0])),
    )
    scaled_w = max(1, int(round(float(src.shape[1]) * scale)))
    scaled_h = max(1, int(round(float(src.shape[0]) * scale)))
    scaled = _resize_rgb_nearest(src, (scaled_h, scaled_w))

    canvas = np.full(
        (_WEIGHT_LIVE_TILE_STACK_H, SCRUB_THUMB_W, 3),
        (14, 16, 20),
        dtype=np.uint8,
    )
    y0 = max(0, (_WEIGHT_LIVE_TILE_STACK_H - scaled_h) // 2)
    x0 = max(0, (SCRUB_THUMB_W - scaled_w) // 2)
    canvas[y0 : y0 + scaled_h, x0 : x0 + scaled_w] = scaled
    return tuple(
        np.ascontiguousarray(canvas[idx * SCRUB_THUMB_H : (idx + 1) * SCRUB_THUMB_H].copy())
        for idx in range(SCRUB_NUM_THUMBS)
    )


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

    def __init__(self, image_hw: Tuple[int, int] = (256, 256)) -> None:
        self._lock = threading.Lock()
        self._image_h, self._image_w = max(8, int(image_hw[0])), max(8, int(image_hw[1]))
        # Per-model storage: model_name → {parameter_keys, base, latest, live_tiles}
        self._models: Dict[str, Dict[str, Any]] = {}
        self._active_model: Optional[str] = None

    def register_base(self, model_name: str, model: nn.Module) -> None:
        """Capture base (initial) weights for a model.  Called once at init."""
        parameter_keys, base_state = snapshot_model_parameter_state(model)
        with self._lock:
            entry = self._models.setdefault(model_name, {})
            entry["parameter_keys"] = list(parameter_keys)
            entry["base"] = dict(base_state)
            if self._active_model is None:
                self._active_model = model_name

    def capture(self, model_name: str, model: nn.Module) -> None:
        """Capture the current weight state (detached, CPU)."""
        parameter_keys, latest_state = snapshot_model_parameter_state(model)
        with self._lock:
            entry = self._models.setdefault(model_name, {})
            if parameter_keys:
                entry["parameter_keys"] = list(parameter_keys)
            entry["latest"] = dict(latest_state)

    def set_active(self, model_name: str) -> None:
        with self._lock:
            self._active_model = model_name

    def model_names(self) -> List[str]:
        with self._lock:
            return list(self._models.keys())

    def active_model(self) -> Optional[str]:
        with self._lock:
            return self._active_model

    def base_state(self, model_name: str) -> Optional[Dict[str, torch.Tensor]]:
        with self._lock:
            entry = self._models.get(str(model_name), {})
            base = entry.get("base")
            return dict(base) if isinstance(base, dict) else None

    def refresh_live_tiles(self, model_name: Optional[str] = None) -> Optional[Tuple[np.ndarray, ...]]:
        with self._lock:
            name = model_name or self._active_model
            if name is None or name not in self._models:
                return None
            entry = self._models[name]
            parameter_keys = list(entry.get("parameter_keys", []) or [])
            base = entry.get("base")
            latest = entry.get("latest")
        if not isinstance(latest, dict):
            return None
        rgb, _meta = render_weight_image(
            latest,
            parameter_keys=parameter_keys,
            reference_state=base if isinstance(base, dict) else None,
            target_width=SCRUB_THUMB_W,
            target_height=SCRUB_THUMB_H,
        )
        tiles = _weight_thumb_tiles_from_rgb(rgb)
        with self._lock:
            entry = self._models.setdefault(str(name), {})
            entry["live_tiles"] = tuple(np.ascontiguousarray(tile.copy()) for tile in tiles)
        return tiles

    def live_thumb_tiles(self, model_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self._lock:
            name = model_name or self._active_model
            if name is None or name not in self._models:
                return None
            tiles = self._models[name].get("live_tiles")
            if not isinstance(tiles, tuple) or len(tiles) != SCRUB_NUM_THUMBS:
                return None
            return {
                "model": str(name),
                "tiles": tuple(np.ascontiguousarray(np.asarray(tile, dtype=np.uint8)).copy() for tile in tiles),
            }

    def render_weight_map_rgb(self, model_name: Optional[str] = None) -> Optional[np.ndarray]:
        """Render an architectural-tall active-model map for the named model."""
        with self._lock:
            name = model_name or self._active_model
            if name is None or name not in self._models:
                return None
            entry = self._models[name]
            parameter_keys = list(entry.get("parameter_keys", []) or [])
            base = entry.get("base")
            latest = entry.get("latest")
        if not isinstance(latest, dict):
            return None
        rgb, meta = render_weight_image(
            latest,
            parameter_keys=parameter_keys,
            reference_state=base if isinstance(base, dict) else None,
            target_width=self._image_w,
            target_height=self._image_h,
        )
        return annotate_weight_map(
            rgb,
            title=str(name),
            subtitle="",
        )

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
        from pipeline.nodus_loss_store import NodusLossStore as _NLS
        self.loss_store: _NLS = _NLS.get_global()
        self.result_store: ResultStore = ResultStore()

        # Pending restore request: (round_id, cycle) to restore to.
        self._pending_restore: Optional[Tuple[int, int]] = None
        self._force_save_pending: bool = False
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

        # Load historical loss data from binary log files so the pull-model
        # can serve the full history (including previous sessions) from round 0.
        self._load_historical_losses(out_dir)

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

    def _load_historical_losses(self, out_dir: Path) -> None:
        """Replay binary loss-log files into the LossAccumulator.

        This ensures the pull-model serves complete history to the GUI,
        including data from previous training sessions.
        """
        try:
            from wav_ml_viewer import _LossFileLogger
        except Exception:
            return
        loaded = 0
        for fname in ("loss_log_prev.bin", "loss_log.bin"):
            fpath = out_dir / fname
            try:
                recs = _LossFileLogger.load(fpath)
                for rec in recs:
                    ck_raw = rec["channel_key"]
                    ck = ck_raw.decode("utf-8").rstrip("\x00") if isinstance(ck_raw, (bytes, np.bytes_)) else str(ck_raw)
                    if not ck:
                        ck = f"stage_{int(rec['stage'])}"
                    loss_val = float(rec["loss"])
                    if not math.isfinite(loss_val):
                        continue
                    self.loss_store.record(
                        channel_key=ck,
                        loss=loss_val,
                        aux=float(rec["aux"]) if math.isfinite(float(rec["aux"])) else 0.0,
                        round_id=int(rec["round"]),
                        ts=float(rec["ts"]),
                    )
                    loaded += 1
            except Exception as exc:
                _log(f"[save-restore] could not load {fname}: {exc}")
        if loaded > 0:
            _log(f"[save-restore] loaded {loaded} historical loss records into accumulator")

    @property
    def runtime_execution_policy(self) -> tuple:
        return ("periodic", {"period": self.save_every_n_rounds, "counter": "round_id", "restore_overrides": True})

    def declare_subnodes(self) -> list:
        return [
            {"subnode_id": "seed_bank", "kind": "state_store", "label": "Seed Bank", "order": 0},
            {"subnode_id": "training_cache", "kind": "state_store", "label": "Training Material Cache", "order": 1},
            {"subnode_id": "loss_store", "kind": "metric_store", "label": "Nodus Loss Store (native)", "order": 2},
            {"subnode_id": "result_store", "kind": "metric_store", "label": "Result Store", "order": 3},
            {"subnode_id": "weight_tracker", "kind": "diagnostic", "label": "Weight Tracker", "order": 4},
        ]

    def should_run(self, ctx: PipelineContext) -> bool:
        # Always run if a restore is pending
        if self._pending_restore is not None or self._force_save_pending:
            return True
        return (ctx.round_id % self.save_every_n_rounds) == 0

    def execute(self, ctx: PipelineContext) -> None:
        # Handle pending restore first
        if self._pending_restore is not None:
            self._execute_restore(ctx)
            return

        if self._force_save_pending:
            try:
                self._execute_save(ctx)
            finally:
                self._force_save_pending = False
            return

        # Scrub editor off → skip all checkpoint storage
        if not ctx.scrub_editor_enabled():
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
            payload["metrics_history"] = list(ctx.metrics_history)
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
        _optimizers = {
            "classifier": ctx.classifier_optimizer,
            "transformer": ctx.transformer_optimizer,
            "generator": ctx.generator_optimizer,
            "discriminator": ctx.discriminator_optimizer,
            "wave_classifier": ctx.wave_classifier_optimizer,
        }
        _lr_controllers = {
            "classifier": ctx.classifier_lr_controller,
            "transformer": ctx.transformer_lr_controller,
        }
        _grad_scalers = {
            "classifier": ctx.classifier_grad_scaler,
            "transformer": ctx.transformer_grad_scaler,
            "generator": ctx.generator_grad_scaler,
            "discriminator": ctx.discriminator_grad_scaler,
            "wave_classifier": ctx.wave_classifier_grad_scaler,
        }
        for name, model in _models.items():
            if model is not None:
                payload[f"{name}_state"] = model.state_dict()
        for name, optimizer in _optimizers.items():
            if optimizer is not None:
                try:
                    payload[f"{name}_optimizer_state"] = optimizer.state_dict()
                except Exception:
                    pass
        for name, controller in _lr_controllers.items():
            if controller is not None and hasattr(controller, "state_dict"):
                try:
                    payload[f"{name}_lr_controller_state"] = controller.state_dict()
                except Exception:
                    pass
        for name, scaler in _grad_scalers.items():
            if scaler is not None and hasattr(scaler, "state_dict"):
                try:
                    payload[f"{name}_grad_scaler_state"] = scaler.state_dict()
                except Exception:
                    pass
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

        # 3b. Save the active-model node thumbnail that matches the saved checkpoint payload.
        thumb_info = self._save_weight_thumbnail(
            out_dir,
            _models,
            payload=payload,
            round_id=ctx.round_id,
            cycle=ctx.cycle,
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
            notification = self.make_checkpoint_notification(
                ctx.round_id,
                ctx.cycle,
                thumb_path=(thumb_info or {}).get("path"),
                thumb_model=(thumb_info or {}).get("model"),
                checkpoint_path=(out_dir / "pipeline_checkpoint.pt"),
            )
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

    def _save_weight_thumbnail(
        self,
        out_dir: Path,
        models: Dict[str, Any],
        *,
        payload: Dict[str, Any],
        round_id: int,
        cycle: int,
    ) -> Optional[Dict[str, Any]]:
        """Render and save a 256px active-model architectural thumbnail for the viewer."""
        try:
            active_name = self.weight_tracker.active_model()
            if not active_name or models.get(active_name) is None:
                for name, model in models.items():
                    if model is not None:
                        active_name = str(name)
                        break
            if not active_name:
                return None

            active_model = models.get(active_name)
            if active_model is None:
                return None

            state_key = f"{active_name}_state"
            state_blob = payload.get(state_key)
            if not isinstance(state_blob, dict):
                state_blob = active_model.state_dict()

            parameter_keys, saved_state = snapshot_parameter_state_from_state_dict(
                state_blob,
                [name for name, _ in active_model.named_parameters()],
            )
            if not saved_state:
                return None

            rgb, meta = render_weight_image(
                saved_state,
                parameter_keys=parameter_keys,
                reference_state=self.weight_tracker.base_state(active_name),
                target_width=_WEIGHT_THUMB_SIDE,
                target_height=_WEIGHT_THUMB_SIDE,
            )
            rgb = annotate_weight_map(
                rgb,
                title=f"{active_name} r{int(round_id)}",
                subtitle="",
            )

            from PIL import Image

            img = Image.fromarray(rgb, mode="RGB")
            thumb_path = out_dir / f"weight_thumb_r{round_id:06d}_c{cycle:04d}.png"
            img.save(
                str(thumb_path),
                format="PNG",
                compress_level=_WEIGHT_THUMB_PNG_COMPRESS_LEVEL,
            )
            sidecar_path = thumb_path.with_suffix(".json")
            sidecar_path.write_text(
                json.dumps(
                    {
                        "model": str(active_name),
                        "round_id": int(round_id),
                        "cycle": int(cycle),
                        "layer_count": int(meta.get("layer_count", 0)),
                        "mode": "architectural_tall",
                        "width": int(img.size[0]),
                        "height": int(img.size[1]),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            return {
                "path": str(thumb_path),
                "model": str(active_name),
                "layer_count": int(meta.get("layer_count", 0)),
            }
        except Exception as exc:
            _log(f"[checkpoint] weight thumbnail failed: {exc}")
        return None

    def _load_checkpoint_for_restore(
        self,
        out_dir: Path,
        target_round: int,
        target_cycle: int,
    ) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
        candidate_paths: List[Path] = []
        live_ckpt = out_dir / "pipeline_checkpoint.pt"
        if live_ckpt.exists():
            candidate_paths.append(live_ckpt)

        backup_root = out_dir / "_weight_backup"
        if backup_root.is_dir():
            backup_dirs = sorted(
                [p for p in backup_root.iterdir() if p.is_dir()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for backup_dir in backup_dirs:
                ckpt_path = backup_dir / "pipeline_checkpoint.pt"
                if ckpt_path.exists():
                    candidate_paths.append(ckpt_path)

        fallback_path: Optional[Path] = None
        fallback_ckpt: Optional[Dict[str, Any]] = None
        for ckpt_path in candidate_paths:
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            except Exception as exc:
                _log(f"[restore] WARNING: could not load checkpoint candidate {ckpt_path}: {exc}")
                continue
            if fallback_path is None and isinstance(ckpt, dict):
                fallback_path = ckpt_path
                fallback_ckpt = ckpt
            if (
                isinstance(ckpt, dict)
                and int(ckpt.get("round_id", -1)) == int(target_round)
                and int(ckpt.get("cycle", -1)) == int(target_cycle)
            ):
                return ckpt_path, ckpt
        return fallback_path, fallback_ckpt

    def prepare_startup_restore(self, ctx: PipelineContext) -> None:
        """Arm the startup resume target from context state during graph execution."""
        if not bool(getattr(ctx, "startup_restore_pending", False)):
            return
        target_round = int(getattr(ctx, "startup_restore_round", 0) or 0)
        target_cycle = int(getattr(ctx, "startup_restore_cycle", 0) or 0)
        if target_round <= 0:
            resume_ckpt = ctx.resume_pipeline_ckpt if isinstance(ctx.resume_pipeline_ckpt, dict) else {}
            target_round = int(resume_ckpt.get("round_id", 0) or 0)
            target_cycle = int(resume_ckpt.get("cycle", 0) or 0)
        if target_round <= 0:
            raise RuntimeError("startup restore is pending but no checkpoint round/cycle is available")
        self.request_restore(target_round, target_cycle)

    def prepare_shutdown_save(self, ctx: PipelineContext) -> None:
        """Force a checkpoint save on the next execute() during shutdown."""
        if not bool(getattr(ctx, "shutdown_save_pending", False)):
            return
        self._force_save_pending = True

    def prepare_runtime_checkpoint(self, ctx: PipelineContext) -> None:
        """Arm restore/save control requests before the checkpoint node runs."""
        if bool(getattr(ctx, "shutdown_save_pending", False)):
            self.prepare_shutdown_save(ctx)
            return
        if bool(getattr(ctx, "startup_restore_pending", False)):
            self.prepare_startup_restore(ctx)

    @staticmethod
    def _prime_cycle_gate_state(ctx: PipelineContext, total_rounds_completed: int) -> None:
        cycle_gates = list(getattr(ctx, "cycle_gates", None) or [])
        for gate in cycle_gates:
            max_iterations = int(getattr(gate, "max_iterations", 0) or 0)
            if max_iterations > 0:
                gate.iteration = min(int(total_rounds_completed), max_iterations)
            else:
                gate.iteration = int(total_rounds_completed)

    def _execute_restore(self, ctx: PipelineContext) -> None:
        """Restore to a checkpoint and replay training material."""
        target_round, target_cycle = self._pending_restore
        self._pending_restore = None
        startup_restore = bool(getattr(ctx, "startup_restore_pending", False))
        out_dir = ctx.output_dir
        if out_dir is None:
            if startup_restore:
                raise RuntimeError("startup restore failed: output_dir is unavailable")
            _log("[restore] ERROR: no output_dir set, cannot restore")
            return

        _log(f"[restore] restoring to round={target_round} cycle={target_cycle}")

        # 1. Load the checkpoint
        ckpt_path, ckpt = self._load_checkpoint_for_restore(out_dir, target_round, target_cycle)
        if ckpt_path is None or ckpt is None:
            if startup_restore:
                raise RuntimeError("startup restore failed: no checkpoint file found")
            _log("[restore] ERROR: no checkpoint file found")
            return
        _log(f"[restore] loading checkpoint payload from: {ckpt_path}")
        live_ckpt_path = out_dir / "pipeline_checkpoint.pt"
        # Replay is only meaningful when we loaded the exact live checkpoint AND its
        # round/cycle matches the target.  A fallback to the wrong checkpoint must NOT
        # replay: replaying material from round 100 onto a model restored to round 100
        # (when the user wanted round 5) would silently continue training instead of
        # rewinding, which is confusing and data-corrupting.
        _ckpt_round = int(ckpt.get("round_id", -1))
        _ckpt_cycle = int(ckpt.get("cycle", -1))
        _is_exact_target = (_ckpt_round == int(target_round) and _ckpt_cycle == int(target_cycle))
        try:
            replay_allowed = _is_exact_target and (ckpt_path.resolve() == live_ckpt_path.resolve())
        except Exception:
            replay_allowed = _is_exact_target and (str(ckpt_path) == str(live_ckpt_path))
        if not _is_exact_target:
            if startup_restore:
                raise RuntimeError(
                    "startup restore failed: checkpoint round/cycle did not match the requested resume target"
                )
            _log(
                f"[restore] WARNING: checkpoint has round={_ckpt_round} cycle={_ckpt_cycle}, "
                f"target was round={target_round} cycle={target_cycle}; "
                "restoring best-available fallback, replay disabled"
            )

        ctx.cycle = max(0, int(ckpt.get("cycle", target_cycle) or target_cycle))
        ctx.round_id = max(0, int(ckpt.get("round_id", target_round) or target_round))
        try:
            ctx.total_rounds_completed = max(
                0,
                int(ckpt.get("total_rounds_completed", ctx.total_rounds_completed) or ctx.total_rounds_completed),
            )
        except Exception:
            pass

        # 2. Restore model weights
        _models = {
            "classifier": ctx.classifier,
            "transformer": ctx.transformer,
            "generator": ctx.generator,
            "discriminator": ctx.discriminator,
            "wave_classifier": ctx.wave_classifier,
        }
        _optimizers = {
            "classifier": ctx.classifier_optimizer,
            "transformer": ctx.transformer_optimizer,
            "generator": ctx.generator_optimizer,
            "discriminator": ctx.discriminator_optimizer,
            "wave_classifier": ctx.wave_classifier_optimizer,
        }
        _lr_controllers = {
            "classifier": ctx.classifier_lr_controller,
            "transformer": ctx.transformer_lr_controller,
        }
        _grad_scalers = {
            "classifier": ctx.classifier_grad_scaler,
            "transformer": ctx.transformer_grad_scaler,
            "generator": ctx.generator_grad_scaler,
            "discriminator": ctx.discriminator_grad_scaler,
            "wave_classifier": ctx.wave_classifier_grad_scaler,
        }
        for name, model in _models.items():
            sd_key = f"{name}_state"
            if model is not None and sd_key in ckpt:
                try:
                    model.load_state_dict(ckpt[sd_key], strict=False)
                    _log(f"[restore] loaded {name} weights")
                except Exception as exc:
                    _log(f"[restore] WARNING: {name} weight load failed: {exc}")

        # 2b. Restore optimizer, LR-controller, and grad-scaler state
        for name, optimizer in _optimizers.items():
            state_key = f"{name}_optimizer_state"
            if optimizer is None or state_key not in ckpt:
                continue
            try:
                optimizer.load_state_dict(ckpt[state_key])
                _log(f"[restore] loaded {name} optimizer state")
            except Exception as exc:
                _log(f"[restore] WARNING: {name} optimizer state load failed: {exc}")
        for name, controller in _lr_controllers.items():
            state_key = f"{name}_lr_controller_state"
            if controller is None or state_key not in ckpt or not hasattr(controller, "load_state_dict"):
                continue
            try:
                controller.load_state_dict(ckpt[state_key])
                _log(f"[restore] loaded {name} LR-controller state")
            except Exception as exc:
                _log(f"[restore] WARNING: {name} LR-controller state load failed: {exc}")
        for name, scaler in _grad_scalers.items():
            state_key = f"{name}_grad_scaler_state"
            if scaler is None or state_key not in ckpt or not hasattr(scaler, "load_state_dict"):
                continue
            try:
                scaler.load_state_dict(ckpt[state_key])
                _log(f"[restore] loaded {name} grad-scaler state")
            except Exception as exc:
                _log(f"[restore] WARNING: {name} grad-scaler state load failed: {exc}")

        # 3. Restore seed state
        if self.seed_bank is not None:
            seed_snap = self.seed_bank.find(target_round, target_cycle)
            if seed_snap is not None:
                seed_snap.restore()
                _log(f"[restore] RNG state restored to r={target_round} c={target_cycle}")
            else:
                if startup_restore:
                    raise RuntimeError(
                        "startup restore failed: no matching seed snapshot was found for the resume target"
                    )
                _log("[restore] WARNING: no seed snapshot found for target, RNG not restored")

        # 4. Replay training material
        if self.training_cache is not None:
            if replay_allowed:
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
            elif self.training_cache.total_entries() > 0:
                _log(
                    "[restore] selected checkpoint came from backup history; "
                    "skipping live training-material replay cache"
                )

        # 5. Clear cache after replay (we're now at the replayed state)
        if self.training_cache is not None:
            self.training_cache.clear_before_checkpoint()

        self._prime_cycle_gate_state(ctx, int(getattr(ctx, "total_rounds_completed", 0) or 0))
        if startup_restore:
            ctx.startup_restore_pending = False
            ctx.resume_pipeline_ckpt = None

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

    def update_runtime_weight_map(self, model_name: str, model: Any) -> Optional[Dict[str, Any]]:
        """Capture the active runtime model so the GUI right panel stays current."""
        if not model_name or model is None:
            return None
        try:
            self.weight_tracker.set_active(str(model_name))
            self.weight_tracker.capture(str(model_name), model)
            self.weight_tracker.refresh_live_tiles(str(model_name))
        except Exception:
            return None
        return self.make_weight_map_notification(model_name)

    def current_weight_thumb_tiles(self, model_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        return self.weight_tracker.live_thumb_tiles(model_name)

    # -- Pull-model: record data (called by training nodes) ---------------

    def record_loss(self, channel_key: str, loss: float, aux: float = 0.0,
                    round_id: int = 0, ts: float = 0.0) -> int:
        """Record a loss value.  Returns the step index.

        Also emits a lightweight notification dict (caller can forward
        to the viewer proxy).
        """
        return self.loss_store.record(
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
        loss_keys = self.loss_store.channel_keys()
        result_keys = self.result_store.channel_keys()
        tm_keys = self.training_cache.channel_keys() if self.training_cache else []
        all_keys = sorted(set(loss_keys) | set(result_keys) | set(tm_keys))
        return {
            "type": RESP_CHANNEL_LIST,
            "channels": [
                {
                    "key": ck,
                    "has_loss": ck in loss_keys,
                    "loss_length": self.loss_store.channel_length(ck),
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
        recs = self.loss_store.query_since(ck, from_step)
        records = [
            {"step": r.step, "round_id": r.round_id,
             "loss": r.loss, "aux": r.aux, "ts": r.ts}
            for r in recs
        ]
        return {
            "type": RESP_LOSS_HISTORY,
            "channel_key": ck,
            "from_step": from_step,
            "records": records,
        }

    def _resp_latest_result(self, query: Dict[str, Any]) -> Dict[str, Any]:
        ck = str(query.get("channel_key", ""))
        result = self.result_store.latest_serialisable(ck)
        # Strip binary/image data — images travel via scrub ring, not IPC.
        if result is not None:
            result = {k: v for k, v in result.items()
                      if not isinstance(v, (np.ndarray, bytes, bytearray))}
        return {
            "type": RESP_LATEST_RESULT,
            "channel_key": ck,
            "result": result,  # text/scalar metadata only
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
            "active": self.weight_tracker.active_model(),
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
            "length": self.loss_store.channel_length(channel_key),
        }

    def make_result_notification(self, channel_key: str) -> Dict[str, Any]:
        return {
            "type": NOTIFY_NEW_RESULT,
            "channel_key": str(channel_key),
        }

    def make_checkpoint_notification(
        self,
        round_id: int,
        cycle: int,
        *,
        thumb_path=None,
        thumb_model=None,
        checkpoint_path=None,
    ) -> Dict[str, Any]:
        d = {
            "type": NOTIFY_CHECKPOINT,
            "round_id": int(round_id),
            "cycle": int(cycle),
            "loss_summary": self.loss_store.summary(),
        }
        if thumb_path is not None:
            d["thumb_path"] = str(thumb_path)
        if thumb_model is not None:
            d["thumb_model"] = str(thumb_model)
        if checkpoint_path is not None:
            d["checkpoint_path"] = str(checkpoint_path)
        return d

    def make_weight_map_notification(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        return {
            "type": NOTIFY_WEIGHT_MAP,
            "model": str(model_name or self.weight_tracker.active_model() or ""),
        }
