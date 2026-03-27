from __future__ import annotations

import unittest
from types import SimpleNamespace

from pipeline.graph import PipelineGraph, PipelineNode
from pipeline.graph_layers import build_execution_program


class _SourceNode(PipelineNode):
    def __init__(self, node_id: str) -> None:
        self._node_id = str(node_id)

    @property
    def node_id(self) -> str:
        return self._node_id

    def execute(self, ctx) -> None:
        return None


class _TargetNeedsProvisionNode(PipelineNode):
    @property
    def node_id(self) -> str:
        return "target"

    def execute(self, ctx) -> None:
        if not bool(getattr(ctx, "loader_ready", False)):
            raise RuntimeError("target executed without provisioned loader")


class ProgramEdgeActivationTests(unittest.TestCase):
    def test_conditional_on_traverse_still_fires_with_unconditional_sibling_edge(self) -> None:
        graph = PipelineGraph(name="program-edge-activation")
        graph.add_node(_SourceNode("data_node"))
        graph.add_node(_SourceNode("gate_node"))
        graph.add_node(_TargetNeedsProvisionNode())

        graph.add_edge(
            "data_node",
            "target",
            condition=lambda ctx: bool(getattr(ctx, "rebuild_due", False)),
            condition_id="data.rebuild_due",
            on_traverse=lambda ctx: setattr(ctx, "loader_ready", True),
            label="provides:loader",
        )
        graph.add_edge("data_node", "target", label="per_round")
        graph.add_edge(
            "gate_node",
            "target",
            condition=lambda ctx: bool(getattr(ctx, "gate_ready", False)),
            condition_id="gates.ready",
            label="after_gate",
        )

        ctx = SimpleNamespace(
            rebuild_due=True,
            gate_ready=True,
            loader_ready=False,
            is_node_selected=lambda _node_id: True,
            viewer_proxy=None,
            raise_on_node_failure=True,
        )

        statuses = graph.execute_program(ctx, execution_program=build_execution_program(graph))

        self.assertEqual(statuses["target"], "ran")
        self.assertTrue(ctx.loader_ready)


if __name__ == "__main__":
    unittest.main()
