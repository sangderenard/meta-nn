"""Wraps the script-style cross-process runtime-control-store smoke test for pytest."""
from __future__ import annotations

import subprocess
import sys
import time

from pipeline.nodus_loss_store import NodusRuntimeControlStore


def test_runtime_control_store_cross_process():
    store = NodusRuntimeControlStore.get_global()
    store.clear()
    try:
        store.begin_service("parent", ts=time.time())
        store.set_exit_requested(True, reason="parent_exit", ts=time.time())

        child_code = (
            "from pipeline.nodus_loss_store import NodusRuntimeControlStore; "
            "import time; "
            "s=NodusRuntimeControlStore.get_global(); "
            "st=s.get_state(); "
            "print(st)"
        )
        result = subprocess.run(
            [sys.executable, "-c", child_code],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0, f"Child failed: {result.stderr}"
        assert "active_service_count=1" in result.stdout
        assert "exit_requested=True" in result.stdout
    finally:
        store.clear()
