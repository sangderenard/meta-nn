"""
pipeline/nodes/web_lease_node.py  --  Web Lease Stage

Leases the current trained model to remote web clients, waits a configurable
duration for gradient contributions, then either incorporates them or short-
circuits.

Stage contract
--------------
1.  Checks ``ctx.lease_store_dir`` is set and the target model is available on
    ctx.  If either is missing the stage is a no-op.
2.  Checks whether there is pending browser-submitted data in the
    ``WebDataset``.  If ``require_dataset=True`` and there is none, skips.
3.  Opens a ``LeaseStore`` collection for the configured slot → the slot is
    **locked** from this point.  The pipeline would normally not continue
    training that slot while locked; this stage is the one that blocks instead.
4.  Exports model weights to
    ``{lease_store_dir}/active_weights/{collection_id}/weights.pt`` so the
    server can serve them at ``GET /api/web/lease/{lease_id}/weights``.
5.  Claims pending WebDataset samples into the collection so the server can
    serve them at ``GET /api/web/lease/{lease_id}/dataset``.
6.  Blocks, polling every ``poll_interval_seconds``, until:
      a. All leases have returned (collection complete), or
      b. ``wait_deadline_seconds`` has elapsed, or
      c. ``ctx.stop_requested()`` signals shutdown.
    In case (b)/(c) the collection is force-expired, unlocking the slot.
7.  Pops gradient blobs from the LeaseStore.  If any arrived AND
    ``apply_contributions=True``:
      - Loads each blob as a numpy ``.npz`` with keys matching
        ``state_dict`` parameter names.
      - Averages the deltas across blobs.
      - Adds ``learning_rate * mean_delta`` to each matching parameter.
    The weight file is then refreshed on disk.
8.  Returns a metrics dict.  The orchestrator continues normally.

Browser client contract (future)
---------------------------------
The browser receives model weights via GET /api/web/lease/{id}/weights and
dataset samples via GET /api/web/lease/{id}/dataset.  After local
training (WebAssembly, ONNX, etc.) it POSTs a numpy ``.npz`` to
``POST /api/web/lease/{id}/gradients`` where each key is a ``state_dict``
parameter name and each value is the weight **delta**
(trained_param - original_param).  The delta format keeps the server side
ignorant of the optimiser used on the browser.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.graph import PipelineNode
from pipeline.context import PipelineContext


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class WebLeaseConfig:
    node_id: str = "stage_web_lease"

    # Which slot / model to lease
    slot_name: str = "web_lease"
    # "classifier" | "generator" | "transformer" — which ctx attribute holds
    # the nn.Module to export.  "classifier" → ctx.classifier_model, etc.
    target_model: str = "classifier"

    # How many concurrent browser leases to issue per collection
    lease_count: int = 1

    # Per-lease TTL: a browser that doesn't return within this window is expired
    lease_ttl_seconds: float = 3600.0

    # Hard outer deadline for the entire wait.  After this the stage force-
    # expires all outstanding leases and moves on regardless of contributions.
    wait_deadline_seconds: float = 3600.0

    # How often to wake up and check collection status while blocking
    poll_interval_seconds: float = 30.0

    # Minimum number of contributions required before applying.
    # 0 = apply whatever arrives (even zero → short-circuit).
    min_contributions: int = 0

    # Whether to actually apply averaged weight deltas to the model.
    # Set to False to collect data without modifying anything (dry-run).
    apply_contributions: bool = True

    # Learning rate multiplier applied to the averaged delta before adding to
    # the model weights.
    contribution_lr: float = 1.0

    # If True, skip the stage entirely when no browser-submitted samples are
    # pending (nothing for clients to train on).
    require_dataset: bool = False

    # ONNX auto-export: when set, the node will also torch.onnx.export the
    # model so browsers can load it directly without a separate API call.
    # Provide the input shape as a list of ints, e.g. [1, 3, 64, 64].
    # None / empty list disables auto-export.
    onnx_input_shape: Optional[list] = None

    enabled: bool = True


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

_MODEL_ATTR: Dict[str, str] = {
    "classifier":  "classifier_model",
    "generator":   "generator_model",
    "transformer": "transformer_model",
}


class WebLeaseNode(PipelineNode):
    """Pipeline stage that leases the model to web clients and waits for work."""

    def __init__(self, cfg: Optional[WebLeaseConfig] = None):
        self.cfg = cfg or WebLeaseConfig()

    @property
    def node_id(self) -> str:
        return str(self.cfg.node_id)

    @property
    def description(self) -> str:
        return (
            f"WebLeaseNode[{self.cfg.slot_name}] model={self.cfg.target_model} "
            f"leases={self.cfg.lease_count} deadline={self.cfg.wait_deadline_seconds}s"
        )

    def should_run(self, ctx: PipelineContext) -> bool:
        if not self.cfg.enabled:
            return False
        if not str(getattr(ctx, "lease_store_dir", "") or "").strip():
            return False
        return True

    def execute(self, ctx: PipelineContext) -> Dict[str, Any]:
        import torch
        from pipeline.lease_store import LeaseStore
        from pipeline.web_dataset import WebDataset
        from pipeline.nodes.save_restore_node import _classifier_checkpoint_metadata

        def _log(msg: str) -> None:
            print(f"[web-lease] {msg}", flush=True)

        lease_store_dir = str(ctx.lease_store_dir or "").strip()
        if not lease_store_dir:
            return {"ran": False, "reason": "no_lease_store_dir"}

        ls = LeaseStore(lease_store_dir)
        wd = WebDataset(lease_store_dir)

        # ----------------------------------------------------------------
        # Resolve target model
        # ----------------------------------------------------------------
        model_attr = _MODEL_ATTR.get(str(self.cfg.target_model), str(self.cfg.target_model))
        model = getattr(ctx, model_attr, None)
        if model is None:
            _log(f"target model '{self.cfg.target_model}' not found on ctx ({model_attr}); skipping")
            return {"ran": False, "reason": "model_not_available"}

        # ----------------------------------------------------------------
        # Dataset gate
        # ----------------------------------------------------------------
        if self.cfg.require_dataset:
            pending = wd.pending_count(self.cfg.slot_name)
            if pending == 0:
                _log("require_dataset=True but no pending web samples; skipping")
                return {"ran": False, "reason": "no_dataset_samples"}

        # ----------------------------------------------------------------
        # Determine generation (use round counter if available)
        # ----------------------------------------------------------------
        generation = int(getattr(ctx, "round_index", 0) or 0)

        # ----------------------------------------------------------------
        # Open collection → slot is now locked
        # ----------------------------------------------------------------
        _log(
            f"opening collection: slot={self.cfg.slot_name} gen={generation} "
            f"leases={self.cfg.lease_count} deadline={self.cfg.wait_deadline_seconds}s"
        )
        coll, leases = ls.open_collection(
            slot_name=self.cfg.slot_name,
            generation=generation,
            count=self.cfg.lease_count,
            ttl_seconds=self.cfg.lease_ttl_seconds,
            deadline_seconds=self.cfg.wait_deadline_seconds,
            note=f"model={self.cfg.target_model}",
        )
        collection_id = coll.collection_id
        _log(f"collection {collection_id[:8]}… issued {len(leases)} lease(s)")

        # ----------------------------------------------------------------
        # Export weights → available to server at /api/web/lease/{id}/weights
        # ----------------------------------------------------------------
        weights_dir = Path(lease_store_dir) / "active_weights" / collection_id
        weights_dir.mkdir(parents=True, exist_ok=True)
        weights_path = weights_dir / "weights.pt"
        try:
            sd = model.state_dict()
            payload = {
                "state_dict": {k: v.cpu() for k, v in sd.items()},
                "model_class": type(model).__name__,
            }
            if str(self.cfg.target_model) == "classifier":
                payload.update(_classifier_checkpoint_metadata(model, ctx))
            # Include constructor kwargs if the model exposes them
            if hasattr(model, "ctor_kwargs"):
                payload["ctor_kwargs"] = model.ctor_kwargs
            torch.save(payload, str(weights_path))
            _log(f"weights exported: {weights_path}")

            # Auto ONNX export if configured
            if self.cfg.onnx_input_shape:
                try:
                    onnx_path = weights_dir / "model.onnx"
                    dummy = torch.randn(*[int(d) for d in self.cfg.onnx_input_shape])
                    model.eval()
                    torch.onnx.export(
                        model, dummy, str(onnx_path),
                        opset_version=17,
                        input_names=["input"],
                        output_names=["output"],
                        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                    )
                    _log(f"ONNX auto-exported: {onnx_path} ({onnx_path.stat().st_size} bytes)")
                except Exception as onnx_exc:
                    _log(f"WARNING: ONNX auto-export failed: {onnx_exc}")

        except Exception as exc:
            _log(f"WARNING: could not export weights: {exc}")

        # ----------------------------------------------------------------
        # Claim pending web dataset samples into this collection
        # ----------------------------------------------------------------
        claimed = wd.claim_pending(self.cfg.slot_name, collection_id)
        _log(f"claimed {len(claimed)} web dataset sample(s) for this collection")

        # ----------------------------------------------------------------
        # Block until completion, deadline, or stop
        # ----------------------------------------------------------------
        deadline = time.time() + float(self.cfg.wait_deadline_seconds)
        contributions = 0

        _log(f"waiting up to {self.cfg.wait_deadline_seconds}s for contributions …")
        while True:
            # Check pipeline stop signal
            if ctx.stop_requested():
                _log("stop requested; force-expiring collection")
                ls.force_expire_collection(collection_id)
                return {"ran": True, "reason": "stopped", "contributions": 0}

            # Check deadline
            if time.time() >= deadline:
                _log("deadline reached; force-expiring remaining leases")
                ls.force_expire_collection(collection_id)
                break

            # Re-read collection status from disk (may have been updated by server)
            ls._rebuild_index()
            current_coll = ls.get_collection(collection_id)
            if current_coll is not None and current_coll.status != "open":
                _log(
                    f"collection resolved: status={current_coll.status} "
                    f"returned={current_coll.returned_count}/{current_coll.target_count} "
                    f"gradients={current_coll.gradient_count}"
                )
                contributions = current_coll.gradient_count
                break

            remaining = max(0.0, deadline - time.time())
            _log(
                f"  waiting … remaining={remaining:.0f}s "
                f"returned={getattr(current_coll, 'returned_count', '?')}/"
                f"{self.cfg.lease_count}"
            )
            time.sleep(min(float(self.cfg.poll_interval_seconds), remaining + 1))

        # Re-read final contribution count
        final_coll = ls.get_collection(collection_id)
        if final_coll is not None:
            contributions = final_coll.gradient_count

        # ----------------------------------------------------------------
        # Pop gradient blobs and optionally apply
        # ----------------------------------------------------------------
        ready = ls.pop_ready_gradients(self.cfg.slot_name)
        blobs: List[bytes] = []
        for _cid, _blobs in ready:
            if _cid == collection_id:
                blobs.extend(_blobs)

        _log(f"contributions received: {len(blobs)}")

        if len(blobs) < max(1, self.cfg.min_contributions):
            _log(
                f"contributions {len(blobs)} < min_contributions {self.cfg.min_contributions}; "
                f"short-circuit (no weight update)"
            )
        elif self.cfg.apply_contributions and blobs:
            self._apply_deltas(model, blobs, self.cfg.contribution_lr, _log)

        # ----------------------------------------------------------------
        # Refresh on-disk weights after potential update
        # ----------------------------------------------------------------
        try:
            sd = model.state_dict()
            payload = {
                "state_dict": {k: v.cpu() for k, v in sd.items()},
                "model_class": type(model).__name__,
            }
            if str(self.cfg.target_model) == "classifier":
                payload.update(_classifier_checkpoint_metadata(model, ctx))
            if hasattr(model, "ctor_kwargs"):
                payload["ctor_kwargs"] = model.ctor_kwargs
            torch.save(payload, str(weights_path))
        except Exception:
            pass

        # Mark collection applied
        ls.mark_applied(collection_id)

        return {
            "ran": True,
            "collection_id": collection_id,
            "leases_issued": len(leases),
            "dataset_samples": len(claimed),
            "contributions": len(blobs),
            "applied": self.cfg.apply_contributions and len(blobs) >= max(1, self.cfg.min_contributions),
        }

    # ------------------------------------------------------------------
    # Gradient application
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_deltas(model, blobs: List[bytes], lr: float, log_fn) -> None:
        """Average weight deltas from all blobs and add to model parameters.

        Each blob is a numpy ``.npz`` where keys are ``state_dict`` param names
        and values are float32 delta arrays (trained_weights - original_weights).
        Unrecognised keys are silently skipped.
        """
        import io
        import numpy as np
        import torch

        sd = model.state_dict()
        accum: Dict[str, Any] = {}
        counts: Dict[str, int] = {}

        for raw in blobs:
            try:
                npz = np.load(io.BytesIO(raw), allow_pickle=False)
                for key in npz.files:
                    if key not in sd:
                        continue
                    delta = torch.from_numpy(np.asarray(npz[key], dtype=np.float32))
                    if delta.shape != sd[key].shape:
                        continue
                    if key not in accum:
                        accum[key] = delta.clone()
                        counts[key] = 1
                    else:
                        accum[key] += delta
                        counts[key] += 1
            except Exception as exc:
                log_fn(f"WARNING: could not parse gradient blob: {exc}")

        if not accum:
            log_fn("no usable gradient data in blobs")
            return

        applied = 0
        with torch.no_grad():
            for key, delta_sum in accum.items():
                mean_delta = delta_sum / max(1, counts[key])
                sd[key].add_(mean_delta.to(sd[key].device) * lr)
                applied += 1
        model.load_state_dict(sd)
        log_fn(f"applied averaged deltas to {applied} parameter(s) (lr={lr})")
