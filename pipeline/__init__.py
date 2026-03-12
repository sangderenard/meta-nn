"""Pipeline graph package for the wav ML training system."""
from .graph import PipelineNode, PipelineEdge, PipelineGraph
from .context import PipelineContext, GateState
from .plan_protocol import (
    DEFAULT_PLAN_FILENAME,
    DEFAULT_RUNTIME_SNAPSHOT_FILENAME,
    TrainingGraphPlan,
    ProtocolEnvelope,
)


def build_training_graph_from_plan(*args, **kwargs):
    from .orchestrator import build_training_graph_from_plan as _impl

    return _impl(*args, **kwargs)

__all__ = [
    "PipelineNode",
    "PipelineEdge",
    "PipelineGraph",
    "PipelineContext",
    "GateState",
    "DEFAULT_PLAN_FILENAME",
    "DEFAULT_RUNTIME_SNAPSHOT_FILENAME",
    "TrainingGraphPlan",
    "ProtocolEnvelope",
    "build_training_graph_from_plan",
]
