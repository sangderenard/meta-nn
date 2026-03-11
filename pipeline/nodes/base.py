"""
Base helpers shared across all pipeline nodes.

Provides:
  - GatedNode  — skip automatically until a set of GateState flags are True
  - OneShotNode — run exactly once, skip on subsequent rounds
  - _resolve_amp / _make_scaler — AMP helpers used by every training node
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable, List, Optional, Sequence

import torch
import torch.nn as nn

from pipeline.graph import PipelineNode
from pipeline.context import PipelineContext, GateState
import json


# ---------------------------------------------------------------------------
# Gated base
# ---------------------------------------------------------------------------

class GatedNode(PipelineNode):
    """A node that skips itself until all specified gates have passed.

    Usage::

        class MyNode(GatedNode):
            required_gates = ["gate_pregestation", "gate_gestation"]
            node_id = "my_node"

            def execute(self, ctx):
                ...
    """

    required_gates: Sequence[str] = []

    def should_run(self, ctx: PipelineContext) -> bool:
        for gate_attr in self.required_gates:
            gate: GateState = getattr(ctx, gate_attr, None)
            if gate is None or not gate.passed:
                return False
        return True


# ---------------------------------------------------------------------------
# One-shot base
# ---------------------------------------------------------------------------

class OneTimeNode(PipelineNode):
    """Runs exactly once per pipeline run (not per round).

    After the first successful execution the node marks itself done and
    all subsequent calls to ``should_run`` return False.
    """

    def __init__(self) -> None:
        self._done = False

    def should_run(self, ctx: PipelineContext) -> bool:
        return not self._done

    def execute(self, ctx: PipelineContext) -> None:
        self._execute_once(ctx)
        self._done = True

    def _execute_once(self, ctx: PipelineContext) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# AMP helpers (shared by all training nodes)
# ---------------------------------------------------------------------------

def resolve_amp_dtype(amp_dtype_str: str) -> torch.dtype:
    key = str(amp_dtype_str).strip().lower()
    if key in ("fp16", "float16", "half"):
        return torch.float16
    if key in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported amp dtype: {amp_dtype_str!r}")


def autocast_context(
    device: torch.device,
    enabled: bool,
    amp_dtype: torch.dtype,
):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type=device.type, dtype=amp_dtype)
    return nullcontext()


def make_grad_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def cuda_supports_dtype(device: torch.device, dtype: torch.dtype) -> bool:
    if device.type != "cuda":
        return True
    if dtype != torch.bfloat16:
        return True
    try:
        major, _ = torch.cuda.get_device_capability(device)
        return int(major) >= 8
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def unwrap_compiled(module: nn.Module) -> nn.Module:
    """Strip torch.compile wrapper if present."""
    orig = getattr(module, "_orig_mod", None)
    if isinstance(orig, nn.Module):
        return orig
    return module


def module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        pass
    try:
        return next(module.buffers()).device
    except StopIteration:
        return torch.device("cpu")


def freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()


def unfreeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(True)
    module.train()


# =========================================================================
# Functions extracted from wav_config_transformer_pipeline.py
# =========================================================================


def _torch_load_cpu(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Older torch versions may not support weights_only.
        return torch.load(path, map_location="cpu")
    except Exception:
        # Fallback for checkpoints that require legacy unpickling behavior.
        return torch.load(path, map_location="cpu", weights_only=False)


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
        if "model_state_dict" in ckpt_obj and isinstance(ckpt_obj["model_state_dict"], dict):
            return ckpt_obj["model_state_dict"]
    if isinstance(ckpt_obj, dict):
        return ckpt_obj
    raise RuntimeError("Checkpoint does not contain a usable state dict.")


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]):
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            out[k[7:]] = v
        elif k.startswith("_orig_mod."):
            out[k[10:]] = v
        else:
            out[k] = v
    return out


def _is_expandable_classifier_head_key(key: str) -> bool:
    k = str(key)
    return k in ("head.2.weight", "head.2.bias", "fc.weight", "fc.bias")


def _try_partial_classifier_head_load(
    key: str,
    src_tensor: torch.Tensor,
    dst_tensor: torch.Tensor,
) -> Optional[torch.Tensor]:
    if not _is_expandable_classifier_head_key(key):
        return None
    if src_tensor.ndim != dst_tensor.ndim:
        return None
    if src_tensor.ndim == 2:
        if int(src_tensor.shape[1]) != int(dst_tensor.shape[1]):
            return None
        rows = min(int(src_tensor.shape[0]), int(dst_tensor.shape[0]))
        if rows <= 0:
            return None
        out = dst_tensor.detach().clone()
        out[:rows, :].copy_(src_tensor[:rows, :].to(device=dst_tensor.device, dtype=dst_tensor.dtype))
        return out
    if src_tensor.ndim == 1:
        rows = min(int(src_tensor.shape[0]), int(dst_tensor.shape[0]))
        if rows <= 0:
            return None
        out = dst_tensor.detach().clone()
        out[:rows].copy_(src_tensor[:rows].to(device=dst_tensor.device, dtype=dst_tensor.dtype))
        return out
    return None


def _apply_model_init(model: nn.Module, ckpt_path: str):
    if not ckpt_path:
        return {"used": False, "loaded_keys": 0, "skipped_keys": 0, "path": ""}
    p = Path(ckpt_path)
    if not p.exists():
        return {"used": False, "loaded_keys": 0, "skipped_keys": 0, "path": str(p)}

    blob = _torch_load_cpu(str(p))
    lora_restore_info = _restore_classifier_lora_from_blob(model, blob, str(p))
    src_state = _strip_module_prefix(_extract_state_dict(blob))
    dst_state = model.state_dict()
    to_load = {}
    skipped = 0
    partial = 0
    for k, v in src_state.items():
        if k in dst_state and tuple(dst_state[k].shape) == tuple(v.shape):
            to_load[k] = v
        elif k in dst_state:
            patched = _try_partial_classifier_head_load(
                key=str(k),
                src_tensor=v,
                dst_tensor=dst_state[k],
            )
            if patched is not None:
                to_load[k] = patched
                partial += 1
            else:
                skipped += 1
        else:
            skipped += 1
    missing, unexpected = model.load_state_dict(to_load, strict=False)
    return {
        "used": True,
        "loaded_keys": len(to_load),
        "partial_keys": int(partial),
        "skipped_keys": skipped,
        "missing_after_load": len(missing),
        "unexpected_after_load": len(unexpected),
        "path": str(p.resolve()),
        "lora_restore": lora_restore_info,
    }


def _apply_state_dict(
    model: nn.Module,
    src_state: Dict[str, torch.Tensor],
    source_name: str,
    classifier_lora: Optional[Dict[str, Any]] = None,
):
    lora_restore_info = _restore_classifier_lora_from_blob(
        model,
        ({"classifier_lora": dict(classifier_lora)} if isinstance(classifier_lora, dict) else None),
        source_name,
    )
    if not isinstance(src_state, dict):
        return {
            "used": False,
            "loaded_keys": 0,
            "skipped_keys": 0,
            "source": source_name,
            "lora_restore": lora_restore_info,
        }
    src_state = _strip_module_prefix(src_state)
    dst_state = model.state_dict()
    to_load = {}
    skipped = 0
    partial = 0
    for k, v in src_state.items():
        if k in dst_state and tuple(dst_state[k].shape) == tuple(v.shape):
            to_load[k] = v
        elif k in dst_state:
            patched = _try_partial_classifier_head_load(
                key=str(k),
                src_tensor=v,
                dst_tensor=dst_state[k],
            )
            if patched is not None:
                to_load[k] = patched
                partial += 1
            else:
                skipped += 1
        else:
            skipped += 1
    missing, unexpected = model.load_state_dict(to_load, strict=False)
    return {
        "used": True,
        "loaded_keys": len(to_load),
        "partial_keys": int(partial),
        "skipped_keys": skipped,
        "missing_after_load": len(missing),
        "unexpected_after_load": len(unexpected),
        "source": source_name,
        "lora_restore": lora_restore_info,
    }


def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_pipeline_checkpoint(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, tmp_path)
    except Exception as _ckpt_err:
        # Disk full or write error — clean up partial file and warn, but do not crash training.
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        print(f"[checkpoint] WARNING: save failed (disk full?), skipping: {path.name} — {_ckpt_err}", flush=True)
        return
    try:
        os.replace(tmp_path, path)
    except Exception as _mv_err:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        print(f"[checkpoint] WARNING: rename failed, skipping: {path.name} — {_mv_err}", flush=True)
