from __future__ import annotations

import unittest
from types import SimpleNamespace

from pipeline.graph import PipelineGraph, PipelineNode
from pipeline.nodes.interrupts import StageStopRequested
from pipeline.progress import interruptible_tqdm


class _Control:
    def __init__(self, *, stop: bool = False, save: bool = False):
        self._stop = bool(stop)
        self._save = bool(save)
        self.pump_calls = 0

    def stop_requested(self) -> bool:
        return self._stop

    def shutdown_save(self) -> bool:
        return self._save

    def paused(self) -> bool:
        return False

    def pump(self) -> None:
        self.pump_calls += 1


class _PauseThenResumeControl(_Control):
    def __init__(self):
        super().__init__(stop=False, save=False)
        self._pause_checks = 0

    def paused(self) -> bool:
        self._pause_checks += 1
        return self._pause_checks < 3


class _ProgressStopNode(PipelineNode):
    @property
    def node_id(self) -> str:
        return "stopper"

    def execute(self, ctx) -> None:
        for _ in interruptible_tqdm(range(4), control=ctx, disable=True):
            pass


class ProgressTests(unittest.TestCase):
    def test_interruptible_tqdm_raises_stop_requested(self) -> None:
        control = _Control(stop=True, save=False)
        with self.assertRaises(StageStopRequested) as excinfo:
            list(interruptible_tqdm(range(3), control=control, disable=True))
        self.assertFalse(excinfo.exception.save_requested)

    def test_interruptible_tqdm_pause_pumps_until_resume(self) -> None:
        control = _PauseThenResumeControl()
        self.assertEqual(list(interruptible_tqdm(range(1), control=control, disable=True, pause_poll_s=0.01)), [0])
        self.assertGreaterEqual(control.pump_calls, 1)

    def test_graph_marks_stop_save_interrupt_before_reraising(self) -> None:
        graph = PipelineGraph(name="progress-stop")
        graph.add_node(_ProgressStopNode())
        ctx = SimpleNamespace(
            stop_requested=lambda: True,
            shutdown_save=lambda: True,
            paused=lambda: False,
            viewer_proxy=None,
            raise_on_node_failure=True,
        )
        with self.assertRaises(StageStopRequested) as excinfo:
            graph.execute_sequence(ctx)
        self.assertTrue(excinfo.exception.save_requested)
        self.assertEqual(ctx.last_node_statuses["stopper"], "interrupted:stop_save")
        self.assertEqual(ctx.last_execution_trace[-1]["status"], "interrupted:stop_save")


if __name__ == "__main__":
    unittest.main()
