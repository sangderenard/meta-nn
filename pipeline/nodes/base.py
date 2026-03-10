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
