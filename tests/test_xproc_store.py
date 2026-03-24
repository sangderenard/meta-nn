"""Wraps the script-style cross-process shared-memory smoke test for pytest."""
from __future__ import annotations

import subprocess
import sys
import time

from pipeline.nodus_loss_store import NodusLossStore


def test_xproc_loss_store():
    store = NodusLossStore.get_global()
    store.clear()
    try:
        store.record("test/xproc", loss=0.42, aux=0.0, round_id=1, ts=time.time())

        child_code = (
            "from pipeline.nodus_loss_store import NodusLossStore; "
            "s=NodusLossStore.get_global(); "
            "rec=s.latest('test/xproc'); "
            "print(f'CHILD READ: {rec}')"
        )
        result = subprocess.run(
            [sys.executable, "-c", child_code],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0, f"Child failed: {result.stderr}"
        assert "0.41999" in result.stdout, f"Expected loss value in: {result.stdout}"
    finally:
        store.clear()
