"""
ViewerIPCNode — first-class pipeline graph node representing the IPC
connection between the training process and the standalone GUI viewer.

This node owns the viewer proxy reference and is responsible for:
  1. Pumping IPC messages (draining inbound from GUI, sending outbound).
  2. Forwarding pull-model queries to the SaveRestoreNode.
  3. Relaying restore requests from the GUI to the SaveRestoreNode.
  4. Sending checkpoint/loss/result notifications to the GUI.
  5. Emitting training-material and result data into the SaveRestoreNode
     when called by training nodes.

By representing the IPC channel as a graph node, it becomes visible in
Mermaid diagrams, plan IR, and runtime snapshots — conforming to the
overarching IR+Mermaid+Graph Runtime philosophy.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Optional

from pipeline.context import PipelineContext
from pipeline.graph import PipelineNode


def _log(msg: str) -> None:
    print(msg, flush=True)


class ViewerIPCNode(PipelineNode):
    """Graph node that owns and drives the viewer IPC connection.

    This node runs every cycle (it never skips).  Its ``execute()``
    pumps the IPC connection, processes pending GUI queries, and
    forwards any restore request to the SaveRestoreNode.

    Attributes set after construction via ``wire()``:
        viewer_proxy  — the ViewerIPCProxy instance
        save_restore  — the SaveRestoreNode for query routing and restore
    """

    node_id = "viewer_ipc"
    description = "IPC bridge to standalone GUI viewer"

    def __init__(self) -> None:
        self._viewer_proxy: Optional[Any] = None
        self._save_restore: Optional[Any] = None
        self._pump_thread: Optional[threading.Thread] = None
        self._thread_stop = threading.Event()
        self._pump_interval: float = 0.05  # 20 Hz background pump

    # -- Wiring -----------------------------------------------------------

    def wire(
        self,
        viewer_proxy: Any,
        save_restore: Optional[Any] = None,
    ) -> None:
        """Inject runtime references after graph construction.

        Called by the orchestrator once context.viewer_proxy and the
        SaveRestoreNode are available.
        """
        self._viewer_proxy = viewer_proxy
        self._save_restore = save_restore
        # Connect the proxy to the save-restore node's query handler
        if save_restore is not None:
            _set_sr = getattr(viewer_proxy, "set_save_restore_node", None)
            if callable(_set_sr):
                _set_sr(save_restore)

    # -- PipelineNode protocol --------------------------------------------

    @property
    def runtime_object_type(self) -> str:
        return "service"

    @property
    def runtime_faculty(self) -> str:
        return "ipc"

    @property
    def runtime_execution_policy(self) -> tuple:
        return ("always", {})

    def should_run(self, ctx: PipelineContext) -> bool:
        return self._viewer_proxy is not None

    def execute(self, ctx: PipelineContext) -> None:
        """Pump IPC, process queries, and relay notifications."""
        proxy = self._viewer_proxy
        if proxy is None:
            return
        pump = getattr(proxy, "pump", None)
        if callable(pump):
            pump()

    # -- Background pump thread -------------------------------------------

    def start_background_pump(self) -> None:
        """Start a daemon thread that pumps IPC at _pump_interval Hz.

        This ensures the GUI stays responsive even between graph execution
        passes (e.g. during long training steps).
        """
        if self._pump_thread is not None:
            return
        self._thread_stop.clear()
        self._pump_thread = threading.Thread(
            target=self._bg_pump_loop, name="ipc-pump", daemon=True,
        )
        self._pump_thread.start()
        _log("[viewer_ipc] background pump thread started")

    def stop_background_pump(self) -> None:
        self._thread_stop.set()
        t = self._pump_thread
        if t is not None:
            t.join(timeout=2.0)
            self._pump_thread = None

    def _bg_pump_loop(self) -> None:
        while not self._thread_stop.is_set():
            proxy = self._viewer_proxy
            if proxy is not None:
                pump = getattr(proxy, "pump", None)
                if callable(pump):
                    try:
                        pump()
                    except Exception:
                        pass
            self._thread_stop.wait(self._pump_interval)

    # -- Convenience: forward notifications to GUI ------------------------

    def notify_loss(self, channel_key: str) -> None:
        """Send a lightweight loss-available notification to the GUI."""
        sr = self._save_restore
        proxy = self._viewer_proxy
        if sr is None or proxy is None:
            return
        send = getattr(proxy, "send_notification", None)
        if callable(send):
            send(sr.make_loss_notification(channel_key))

    def notify_result(self, channel_key: str) -> None:
        sr = self._save_restore
        proxy = self._viewer_proxy
        if sr is None or proxy is None:
            return
        send = getattr(proxy, "send_notification", None)
        if callable(send):
            send(sr.make_result_notification(channel_key))

    def notify_checkpoint(self, round_id: int, cycle: int) -> None:
        sr = self._save_restore
        proxy = self._viewer_proxy
        if sr is None or proxy is None:
            return
        send = getattr(proxy, "send_notification", None)
        if callable(send):
            send(sr.make_checkpoint_notification(round_id, cycle))

    # -- Runtime control accessors (edge faculty) -------------------------

    def paused(self) -> bool:
        fn = getattr(self._viewer_proxy, "paused", None)
        return bool(fn()) if callable(fn) else False

    def preview_enabled(self) -> bool:
        fn = getattr(self._viewer_proxy, "preview_enabled", None)
        return bool(fn()) if callable(fn) else True

    def scrub_editor_enabled(self) -> bool:
        fn = getattr(self._viewer_proxy, "scrub_editor_enabled", None)
        return bool(fn()) if callable(fn) else True
