from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

import torch
import torch.nn as nn

from pipeline.context import PipelineContext
from pipeline.nodes.base import gpu_resident, module_device, resolve_non_training_device


class SemanticTensorWorkload(nn.Module):
    """Lightweight resident wrapper for GPU-eligible semantic preprocessing.

    The workload itself is not a trainable model. It exists so the same staged
    GPU residence manager used for training modules can also pin semantic data
    prep kernels on-device under an explicit name.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_resident_token", torch.zeros((1,), dtype=torch.float32), persistent=False)

    @property
    def processing_device(self) -> torch.device:
        return module_device(self)


def ensure_semantic_tensor_workload(ctx: PipelineContext) -> SemanticTensorWorkload:
    workload = getattr(ctx, "semantic_tensor_workload", None)
    if isinstance(workload, SemanticTensorWorkload):
        return workload
    workload = SemanticTensorWorkload()
    setattr(ctx, "semantic_tensor_workload", workload)
    return workload


@contextmanager
def semantic_processing_device(
    ctx: PipelineContext,
    *,
    enabled: bool,
) -> Iterator[Optional[torch.device]]:
    if not bool(enabled):
        yield None
        return
    device = resolve_non_training_device(ctx)
    if device.type != "cuda":
        yield None
        return
    workload = ensure_semantic_tensor_workload(ctx)
    with gpu_resident(ctx, [(workload, "semantic_tensor_workload")], device=device):
        yield workload.processing_device
