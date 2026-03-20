"""
pipeline/lease_store.py  --  Distributed gradient lease system.

Manages the lifecycle of network leases issued to remote clients (e.g. web
browsers running ONNX/WebGPU inference+training).  A *collection* is a batch
of leases for the same network slot at the same generation.  While any lease
in a collection is outstanding the slot is **locked** — the local training
pipeline should not update its weights until group gradient data returns.

State machines
--------------
Collection:  open → [complete | expired] → applied
Lease:       active → [returned | abandoned | expired]

Lockout rule
  A slot is locked while any collection for it has status ``"open"``.

Gradient accumulation
  Each returning lease may optionally attach a gradient blob (raw bytes of a
  numpy .npz file).  When a collection completes or expires, the blobs are
  queued on disk.  The pipeline pops them via ``pop_ready_gradients(slot)``,
  which marks collections as ``"applied"`` and returns the raw bytes.

Persistence
  All state is written to ``{store_dir}/collections/``, ``leases/``, and
  ``gradients/``.  The in-memory lock index is rebuilt on startup by scanning
  collection files, so the store survives server and pipeline restarts.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_LEASE_STORE_VERSION = 1


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class LeaseRecord:
    lease_id: str
    collection_id: str
    slot_name: str
    generation: int
    issued_at: float
    expires_at: float
    status: str          # "active" | "returned" | "abandoned" | "expired"
    client_hint: str = ""
    has_gradient: bool = False


@dataclass
class CollectionRecord:
    collection_id: str
    slot_name: str
    generation: int
    target_count: int
    returned_count: int = 0    # leases resolved (any terminal status)
    gradient_count: int = 0    # leases that came back WITH gradient data
    status: str = "open"       # "open" | "complete" | "expired" | "applied"
    issued_at: float = 0.0
    deadline: Optional[float] = None
    note: str = ""


# ---------------------------------------------------------------------------
# Lease store
# ---------------------------------------------------------------------------

class LeaseStore:
    """Thread-safe, disk-backed store for network leases and gradient blobs."""

    def __init__(self, store_dir: str):
        self._dir = Path(str(store_dir))
        self._dir.mkdir(parents=True, exist_ok=True)
        (self._dir / "collections").mkdir(exist_ok=True)
        (self._dir / "leases").mkdir(exist_ok=True)
        (self._dir / "gradients").mkdir(exist_ok=True)
        self._lock = threading.RLock()
        # slot_name → set of open collection IDs  (fast lockout check)
        self._locked_slots: Dict[str, set] = {}
        # in-memory caches (authoritative after _rebuild_index)
        self._collections: Dict[str, CollectionRecord] = {}
        self._leases: Dict[str, LeaseRecord] = {}
        self._rebuild_index()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_index(self) -> None:
        """Scan all on-disk state and rebuild in-memory index.  Called once at
        startup; also safe to call if on-disk state has drifted."""
        with self._lock:
            self._locked_slots.clear()
            self._collections.clear()
            self._leases.clear()
            for f in sorted((self._dir / "collections").glob("*.json")):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    coll = CollectionRecord(**data)
                    self._collections[coll.collection_id] = coll
                    if coll.status == "open":
                        self._locked_slots.setdefault(coll.slot_name, set()).add(
                            coll.collection_id
                        )
                except Exception:
                    pass
            for f in sorted((self._dir / "leases").glob("*.json")):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    lease = LeaseRecord(**data)
                    self._leases[lease.lease_id] = lease
                except Exception:
                    pass

    def _save_collection(self, coll: CollectionRecord) -> None:
        path = self._dir / "collections" / f"{coll.collection_id}.json"
        path.write_text(json.dumps(asdict(coll), indent=2), encoding="utf-8")

    def _save_lease(self, lease: LeaseRecord) -> None:
        path = self._dir / "leases" / f"{lease.lease_id}.json"
        path.write_text(json.dumps(asdict(lease), indent=2), encoding="utf-8")

    def _collection_leases_locked(self, collection_id: str) -> List[LeaseRecord]:
        """Return all leases for a collection.  Must be called under self._lock."""
        return [l for l in self._leases.values() if l.collection_id == collection_id]

    def _maybe_complete_collection(self, coll: CollectionRecord) -> None:
        """If all leases are resolved, mark collection complete and unlock slot.
        Must be called under self._lock."""
        if coll.status != "open":
            return
        leases = self._collection_leases_locked(coll.collection_id)
        if all(l.status != "active" for l in leases):
            coll.status = "complete"
            self._locked_slots.get(coll.slot_name, set()).discard(coll.collection_id)
            self._save_collection(coll)

    # ------------------------------------------------------------------
    # Lockout queries
    # ------------------------------------------------------------------

    def is_locked(self, slot_name: str) -> bool:
        """Return True if *slot_name* currently has outstanding leases."""
        with self._lock:
            return bool(self._locked_slots.get(str(slot_name)))

    def locked_slots(self) -> Dict[str, List[str]]:
        """Return {slot_name: [open_collection_id, ...]} for all locked slots."""
        with self._lock:
            return {k: list(v) for k, v in self._locked_slots.items() if v}

    # ------------------------------------------------------------------
    # Collection lifecycle
    # ------------------------------------------------------------------

    def open_collection(
        self,
        slot_name: str,
        generation: int,
        count: int,
        ttl_seconds: float = 3600.0,
        deadline_seconds: Optional[float] = None,
        note: str = "",
    ) -> Tuple[CollectionRecord, List[LeaseRecord]]:
        """Open a new collection for *slot_name*, issuing *count* leases.

        The slot is immediately locked until all leases are resolved or the
        collection is force-expired.

        Parameters
        ----------
        slot_name:
            Identifies the network being leased (e.g. ``"vocab_cat_dog"``).
        generation:
            Monotonic version of the network snapshot being distributed.
            Callers should increment this each time the master weights change.
        count:
            Number of leases to issue (how many clients will receive the
            network).
        ttl_seconds:
            Per-lease time-to-live.  Leases not returned within this window
            are eligible for expiry via ``expire_stale_leases()``.
        deadline_seconds:
            If set, the entire collection is force-expired after this many
            seconds even if not all leases have returned.  Useful as a
            hard unlock guarantee.
        note:
            Free-text annotation stored with the collection.
        """
        with self._lock:
            coll_id = str(uuid.uuid4())
            now = time.time()
            coll = CollectionRecord(
                collection_id=coll_id,
                slot_name=str(slot_name),
                generation=int(generation),
                target_count=max(1, int(count)),
                issued_at=now,
                deadline=(now + float(deadline_seconds)) if deadline_seconds is not None else None,
                note=str(note),
            )
            (self._dir / "gradients" / coll_id).mkdir(parents=True, exist_ok=True)
            self._save_collection(coll)
            self._collections[coll_id] = coll
            self._locked_slots.setdefault(str(slot_name), set()).add(coll_id)

            expires_at = now + float(ttl_seconds)
            leases: List[LeaseRecord] = []
            for _ in range(max(1, int(count))):
                lease_id = str(uuid.uuid4())
                lease = LeaseRecord(
                    lease_id=lease_id,
                    collection_id=coll_id,
                    slot_name=str(slot_name),
                    generation=int(generation),
                    issued_at=now,
                    expires_at=expires_at,
                    status="active",
                )
                self._save_lease(lease)
                self._leases[lease_id] = lease
                leases.append(lease)
            return coll, leases

    def get_collection(self, collection_id: str) -> Optional[CollectionRecord]:
        with self._lock:
            return self._collections.get(str(collection_id))

    def get_collection_leases(self, collection_id: str) -> List[LeaseRecord]:
        with self._lock:
            return self._collection_leases_locked(str(collection_id))

    def get_lease(self, lease_id: str) -> Optional[LeaseRecord]:
        with self._lock:
            return self._leases.get(str(lease_id))

    # ------------------------------------------------------------------
    # Lease resolution
    # ------------------------------------------------------------------

    def return_lease(
        self, lease_id: str, gradient_data: Optional[bytes] = None
    ) -> Tuple[Optional[LeaseRecord], Optional[CollectionRecord]]:
        """Mark a lease as returned, optionally with gradient bytes.

        Returns ``(lease, collection)`` where *collection* is non-None only if
        this return caused the collection to complete (all leases resolved).
        """
        with self._lock:
            lease = self._leases.get(str(lease_id))
            if lease is None or lease.status != "active":
                return None, None
            now = time.time()
            if float(lease.expires_at) < now:
                lease.status = "expired"
            else:
                lease.status = "returned"
                if gradient_data:
                    grad_path = (
                        self._dir / "gradients" / lease.collection_id / f"{lease_id}.npz"
                    )
                    grad_path.write_bytes(gradient_data)
                    lease.has_gradient = True
            self._save_lease(lease)

            coll = self._collections.get(lease.collection_id)
            if coll is not None and coll.status == "open":
                coll.returned_count += 1
                if lease.has_gradient:
                    coll.gradient_count += 1
                self._maybe_complete_collection(coll)
                just_completed = coll.status == "complete"
            else:
                just_completed = False

            return lease, (coll if just_completed else None)

    def abandon_lease(self, lease_id: str) -> bool:
        """Mark a lease as abandoned (client gave up without returning gradients)."""
        with self._lock:
            lease = self._leases.get(str(lease_id))
            if lease is None or lease.status != "active":
                return False
            lease.status = "abandoned"
            self._save_lease(lease)
            coll = self._collections.get(lease.collection_id)
            if coll is not None and coll.status == "open":
                coll.returned_count += 1
                self._maybe_complete_collection(coll)
            return True

    def expire_stale_leases(self) -> Dict[str, int]:
        """Expire any active leases past their ``expires_at`` time, and any
        collections past their ``deadline``.

        Returns ``{slot_name: expired_lease_count}`` for affected slots.
        Safe to call on a background timer.
        """
        with self._lock:
            now = time.time()
            expired_by_slot: Dict[str, int] = {}
            for lease in list(self._leases.values()):
                if lease.status == "active" and float(lease.expires_at) < now:
                    lease.status = "expired"
                    self._save_lease(lease)
                    expired_by_slot[lease.slot_name] = (
                        expired_by_slot.get(lease.slot_name, 0) + 1
                    )
                    coll = self._collections.get(lease.collection_id)
                    if coll is not None and coll.status == "open":
                        coll.returned_count += 1
                        self._maybe_complete_collection(coll)
            # Force-expire collections past their deadline
            for coll in list(self._collections.values()):
                if (
                    coll.status == "open"
                    and coll.deadline is not None
                    and float(coll.deadline) < now
                ):
                    n = self._force_expire_collection_locked(coll)
                    expired_by_slot[coll.slot_name] = (
                        expired_by_slot.get(coll.slot_name, 0) + n
                    )
            return expired_by_slot

    def force_expire_collection(self, collection_id: str) -> int:
        """Force-expire all active leases in a collection, immediately unlocking
        the slot.  Returns the number of leases that were still active."""
        with self._lock:
            coll = self._collections.get(str(collection_id))
            if coll is None or coll.status != "open":
                return 0
            return self._force_expire_collection_locked(coll)

    def _force_expire_collection_locked(self, coll: CollectionRecord) -> int:
        """Must be called under self._lock."""
        expired = 0
        for lease in self._collection_leases_locked(coll.collection_id):
            if lease.status == "active":
                lease.status = "expired"
                coll.returned_count += 1
                self._save_lease(lease)
                expired += 1
        coll.status = "expired"
        self._locked_slots.get(coll.slot_name, set()).discard(coll.collection_id)
        self._save_collection(coll)
        return expired

    # ------------------------------------------------------------------
    # Gradient consumption (pipeline side)
    # ------------------------------------------------------------------

    def pop_ready_gradients(
        self, slot_name: str
    ) -> List[Tuple[str, List[bytes]]]:
        """Return all completed/expired collections for *slot_name* with their
        gradient blobs, marking them as ``"applied"``.

        Returns ``[(collection_id, [npz_bytes, ...])]``.  The caller is
        responsible for aggregating the gradient blobs and applying them.
        """
        with self._lock:
            result: List[Tuple[str, List[bytes]]] = []
            for coll in list(self._collections.values()):
                if coll.slot_name != str(slot_name):
                    continue
                if coll.status not in ("complete", "expired"):
                    continue
                grad_dir = self._dir / "gradients" / coll.collection_id
                blobs: List[bytes] = []
                if grad_dir.exists():
                    for gf in sorted(grad_dir.glob("*.npz")):
                        blobs.append(gf.read_bytes())
                result.append((coll.collection_id, blobs))
                coll.status = "applied"
                self._save_collection(coll)
            return result

    def mark_applied(self, collection_id: str) -> bool:
        """Mark a specific collection as applied without consuming gradients."""
        with self._lock:
            coll = self._collections.get(str(collection_id))
            if coll is None or coll.status not in ("complete", "expired"):
                return False
            coll.status = "applied"
            self._save_collection(coll)
            return True

    # ------------------------------------------------------------------
    # Status / introspection
    # ------------------------------------------------------------------

    def slot_status(self) -> Dict[str, dict]:
        """Return a per-slot status dict suitable for JSON serialisation."""
        with self._lock:
            slots: Dict[str, dict] = {}
            for coll in self._collections.values():
                s = coll.slot_name
                if s not in slots:
                    slots[s] = {
                        "slot_name": s,
                        "locked": False,
                        "open_collections": [],
                        "recent_collections": [],
                    }
                if coll.status == "open":
                    slots[s]["locked"] = True
                    slots[s]["open_collections"].append(coll.collection_id)
                else:
                    slots[s]["recent_collections"].append(
                        {"collection_id": coll.collection_id, "status": coll.status}
                    )
            return slots

    def all_collections(self) -> List[CollectionRecord]:
        with self._lock:
            return list(self._collections.values())

    def collection_detail(self, collection_id: str) -> Optional[dict]:
        """Return a collection + its leases as a JSON-ready dict, or None."""
        with self._lock:
            coll = self._collections.get(str(collection_id))
            if coll is None:
                return None
            leases = self._collection_leases_locked(collection_id)
            return {
                **asdict(coll),
                "leases": [asdict(l) for l in leases],
            }
