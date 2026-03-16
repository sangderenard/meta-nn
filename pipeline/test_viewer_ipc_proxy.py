import numpy as np
import multiprocessing.connection as _mp_connection
from pathlib import Path
from tempfile import TemporaryDirectory

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


def test_viewer_ipc_proxy_does_not_push_weight_tiles_into_scrub_ring():
    proxy = _make_proxy()
    proxy._scrub_ring = _RecordingRing()

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
    assert call["thumb0"] is None
    assert call["thumb1"] is None
    assert call["thumb2"] is None


def test_viewer_ipc_proxy_syncs_weight_image_spec_from_status():
    class _FakeSaveRestore:
        def __init__(self):
            self.calls = []

        def configure_runtime_weight_image(self, *, mode, target_width, target_height):
            self.calls.append((int(mode), int(target_width), int(target_height)))
            return None

    proxy = _make_proxy()
    fake_sr = _FakeSaveRestore()
    proxy.set_save_restore_node(fake_sr)
    proxy._conn = _QueuedConn(
        [
            {
                "type": "status",
                "stop_requested": False,
                "paused": False,
                "gate_override": False,
                "preview_enabled": True,
                "scrub_editor_enabled": True,
                "cycle_selected": [True, True],
                "weight_image_mode": 1,
                "weight_panel_crop_w": 123,
                "weight_panel_crop_h": 234,
            }
        ]
    )

    spec = proxy.weight_image_spec()

    assert spec == {"mode": 1, "panel_crop_w": 123, "panel_crop_h": 234}
    assert fake_sr.calls == [(1, 123, 234)]


def test_viewer_ipc_proxy_retries_weight_image_spec_until_it_applies():
    class _FakeSaveRestore:
        def __init__(self):
            self.calls = []
            self.results = [None, object()]

        def configure_runtime_weight_image(self, *, mode, target_width, target_height):
            self.calls.append((int(mode), int(target_width), int(target_height)))
            return self.results.pop(0)

    proxy = _make_proxy()
    proxy._weight_image_mode = 1
    proxy._weight_panel_crop_w = 123
    proxy._weight_panel_crop_h = 234
    fake_sr = _FakeSaveRestore()

    proxy.set_save_restore_node(fake_sr)
    assert proxy._last_applied_weight_image_spec is None

    proxy._sync_weight_image_spec_to_store()
    assert proxy._last_applied_weight_image_spec == (1, 123, 234)
    assert fake_sr.calls == [(1, 123, 234), (1, 123, 234)]
