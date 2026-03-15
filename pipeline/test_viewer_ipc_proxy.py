import numpy as np
import multiprocessing.connection as _mp_connection
from pathlib import Path
from tempfile import TemporaryDirectory

from pipeline.nodus_loss_store import SCRUB_FLAG_HAS_THUMBS
from pipeline.plan_protocol import (
    MESSAGE_TYPE_RUN_CONTROL,
    RunControlPayload,
    make_envelope,
)
from wav_ml_viewer import ViewerIPCProxy


class _QueuedConn:
    def __init__(self, messages):
        self._messages = list(messages)
        self.sent = []
        self.closed = False

    def poll(self, _timeout=0):
        return bool(self._messages)

    def recv(self):
        if not self._messages:
            raise EOFError
        return self._messages.pop(0)

    def send(self, msg):
        self.sent.append(msg)

    def close(self):
        self.closed = True


class _RecvEOFConn:
    def __init__(self):
        self.closed = False

    def poll(self, _timeout=0):
        return True

    def recv(self):
        raise EOFError

    def close(self):
        self.closed = True


class _SendEOFConn:
    def __init__(self):
        self.closed = False

    def send(self, _msg):
        raise EOFError

    def close(self):
        self.closed = True


def _make_proxy() -> ViewerIPCProxy:
    proxy = ViewerIPCProxy(enabled=False, cycle_slots=2)
    proxy.enabled = True
    return proxy


class _RecordingRing:
    def __init__(self):
        self.calls = []

    def push(self, **kwargs):
        self.calls.append(dict(kwargs))
        return 17


def test_viewer_ipc_proxy_explicit_stop_request_survives():
    stop_msg = make_envelope(
        MESSAGE_TYPE_RUN_CONTROL,
        RunControlPayload(
            command="stop",
            selected_cycle_ids=[1, 2],
            gate_override=False,
            metadata={"save": False},
        ),
    ).to_dict()
    proxy = _make_proxy()
    proxy._conn = _QueuedConn([stop_msg])

    assert proxy.stop_requested() is True
    assert proxy.shutdown_save() is False


def test_viewer_ipc_proxy_status_disconnect_does_not_look_like_stop():
    proxy = _make_proxy()
    proxy._conn = _RecvEOFConn()

    assert proxy.stop_requested() is False
    assert proxy._connection_lost is True
    assert proxy.enabled is True
    assert proxy._conn is None
    assert proxy.shutdown_save() is None


def test_viewer_ipc_proxy_send_disconnect_does_not_look_like_stop():
    proxy = _make_proxy()
    proxy._conn = _SendEOFConn()

    proxy.send_notification({"type": "checkpoint_saved"})

    assert proxy.stop_requested() is False
    assert proxy._connection_lost is True
    assert proxy.enabled is True
    assert proxy._conn is None


def test_viewer_ipc_proxy_auto_reconnects_after_disconnect():
    proxy = _make_proxy()
    proxy._connected_once = True
    proxy._connection_lost = True
    proxy._reconnect_interval_s = 0.0

    with TemporaryDirectory() as tmp_dir:
        port_file = Path(tmp_dir) / ".viewer_port"
        port_file.write_text("61999", encoding="utf-8")
        proxy._port_file = str(port_file)
        reconnect_conn = _QueuedConn([])
        calls = []
        orig_client = _mp_connection.Client

        def _fake_client(address, family, authkey):
            calls.append((address, family, authkey))
            return reconnect_conn

        _mp_connection.Client = _fake_client
        try:
            proxy.pump()
        finally:
            _mp_connection.Client = orig_client

    assert proxy._conn is reconnect_conn
    assert proxy._connection_lost is False
    assert proxy._last_connected_port == 61999
    assert calls == [(("localhost", 61999), "AF_INET", b"nodus_viewer_v1")]


def test_viewer_ipc_proxy_pushes_weight_tiles_into_scrub_ring():
    class _FakeSaveRestore:
        def current_weight_thumb_tiles(self):
            tile = np.full((64, 64, 3), 12, dtype=np.uint8)
            return {"model": "classifier", "tiles": (tile, tile.copy(), tile.copy())}

    proxy = _make_proxy()
    proxy._scrub_ring = _RecordingRing()
    proxy.set_save_restore_node(_FakeSaveRestore())

    proxy.enqueue_frame(
        {
            "images": [
                np.zeros((4, 4, 3), dtype=np.uint8),
                np.zeros((4, 4, 3), dtype=np.uint8),
                np.zeros((4, 4, 3), dtype=np.uint8),
            ],
            "caption": "frame",
            "titles": ["a", "b", "c"],
            "rows": [[], [], []],
        }
    )

    assert len(proxy._scrub_ring.calls) == 1
    call = proxy._scrub_ring.calls[0]
    assert int(call["flags"]) & int(SCRUB_FLAG_HAS_THUMBS)
    assert call["thumb0"] is not None
    assert call["thumb1"] is not None
    assert call["thumb2"] is not None
