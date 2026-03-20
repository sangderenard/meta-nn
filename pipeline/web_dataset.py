"""
pipeline/web_dataset.py  --  Browser-submitted labeled image/mask dataset.

Users submit RGBA images via the HTTP endpoint
``POST /api/web/dataset/{slot_name}``.  The alpha channel encodes the mask.
Each submission carries an arbitrary list of label strings.

Samples accumulate here, unordered, until the WebLeaseNode pulls them into a
collection.  Once consumed by a collection, samples are marked so they are not
re-consumed on the next round.

Storage layout
--------------
    {store_dir}/web_dataset/{slot_name}/{sample_id}.json    -- metadata
    {store_dir}/web_dataset/{slot_name}/{sample_id}_rgba.npy  -- float32 [H,W,4]
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class WebSample:
    sample_id: str
    slot_name: str
    submitted_at: float
    labels: List[str]
    width: int
    height: int
    client_hint: str = ""
    consumed_by: Optional[str] = None   # collection_id that claimed this sample


class WebDataset:
    """Thread-safe, disk-backed store for browser-submitted image+mask samples."""

    def __init__(self, store_dir: str):
        self._root = Path(str(store_dir)) / "web_dataset"
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _slot_dir(self, slot_name: str) -> Path:
        d = self._root / str(slot_name)
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def add_sample(
        self,
        slot_name: str,
        rgba: np.ndarray,
        labels: List[str],
        client_hint: str = "",
    ) -> str:
        """Store a sample.  *rgba* must be float32 [H, W, 4], values in [0,1].

        Returns the new sample_id.
        """
        if rgba.ndim != 3 or rgba.shape[2] != 4:
            raise ValueError(f"rgba must be [H,W,4], got {tuple(rgba.shape)}")
        rgba = np.ascontiguousarray(rgba, dtype=np.float32)
        sample_id = str(uuid.uuid4())
        slot_dir = self._slot_dir(str(slot_name))
        meta = WebSample(
            sample_id=sample_id,
            slot_name=str(slot_name),
            submitted_at=time.time(),
            labels=[str(l) for l in labels],
            width=int(rgba.shape[1]),
            height=int(rgba.shape[0]),
            client_hint=str(client_hint),
        )
        with self._lock:
            np.save(str(slot_dir / f"{sample_id}_rgba.npy"), rgba)
            (slot_dir / f"{sample_id}.json").write_text(
                json.dumps(asdict(meta), indent=2), encoding="utf-8"
            )
        return sample_id

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def pending_samples(self, slot_name: str) -> List[WebSample]:
        """Return all samples not yet consumed by a collection."""
        with self._lock:
            return [s for s in self._load_slot(str(slot_name)) if s.consumed_by is None]

    def pending_count(self, slot_name: str) -> int:
        return len(self.pending_samples(str(slot_name)))

    def get_sample(self, slot_name: str, sample_id: str) -> Optional[WebSample]:
        with self._lock:
            path = self._slot_dir(str(slot_name)) / f"{sample_id}.json"
            if not path.exists():
                return None
            try:
                return WebSample(**json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                return None

    def load_rgba(self, slot_name: str, sample_id: str) -> Optional[np.ndarray]:
        """Load the RGBA array for a sample, or None if not found."""
        path = self._slot_dir(str(slot_name)) / f"{sample_id}_rgba.npy"
        if not path.exists():
            return None
        return np.load(str(path))

    def slot_names(self) -> List[str]:
        """All slot names that have at least one stored sample."""
        with self._lock:
            return [d.name for d in self._root.iterdir() if d.is_dir()]

    def status(self) -> Dict[str, dict]:
        """Return per-slot sample counts."""
        with self._lock:
            result: Dict[str, dict] = {}
            for slot in self.slot_names():
                samples = self._load_slot(slot)
                result[slot] = {
                    "total": len(samples),
                    "pending": sum(1 for s in samples if s.consumed_by is None),
                    "consumed": sum(1 for s in samples if s.consumed_by is not None),
                }
            return result

    # ------------------------------------------------------------------
    # Consumption
    # ------------------------------------------------------------------

    def claim_pending(self, slot_name: str, collection_id: str) -> List[WebSample]:
        """Mark all pending samples for *slot_name* as consumed by *collection_id*.

        Returns the claimed samples.  Safe to call multiple times — already-
        consumed samples are not re-claimed.
        """
        with self._lock:
            claimed: List[WebSample] = []
            for sample in self._load_slot(str(slot_name)):
                if sample.consumed_by is None:
                    sample.consumed_by = str(collection_id)
                    self._save_meta(str(slot_name), sample)
                    claimed.append(sample)
            return claimed

    def samples_for_collection(self, slot_name: str, collection_id: str) -> List[WebSample]:
        """Return all samples consumed by *collection_id*."""
        with self._lock:
            return [
                s for s in self._load_slot(str(slot_name))
                if s.consumed_by == str(collection_id)
            ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_slot(self, slot_name: str) -> List[WebSample]:
        """Load all sample metadata for a slot.  Must be called under self._lock."""
        slot_dir = self._root / str(slot_name)
        if not slot_dir.exists():
            return []
        samples: List[WebSample] = []
        for f in sorted(slot_dir.glob("*.json")):
            try:
                samples.append(WebSample(**json.loads(f.read_text(encoding="utf-8"))))
            except Exception:
                pass
        return samples

    def _save_meta(self, slot_name: str, sample: WebSample) -> None:
        """Persist sample metadata.  Must be called under self._lock."""
        path = self._slot_dir(str(slot_name)) / f"{sample.sample_id}.json"
        path.write_text(json.dumps(asdict(sample), indent=2), encoding="utf-8")
