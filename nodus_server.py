#!/usr/bin/env python3
"""
nodus_server.py -- HTTP server that exposes the nodus shared-memory data stores.

Connects to the same cross-process shared memory as the training pipeline and
serves the data as JSON / PNG over HTTP.  The server is read-only; it never
writes to the shared memory.

Usage:
    python nodus_server.py [--host 127.0.0.1] [--port 7272]

Endpoints:
    GET /                                   HTML index / API listing
    GET /api/status                         JSON  overall store status
    GET /api/loss/channels                  JSON  all loss channels + summary
    GET /api/loss/channel/<key>/records     JSON  records (query: from_step, max)
    GET /api/loss/channel/<key>/latest      JSON  most recent record
    GET /api/loss/graph.png                 PNG   rendered loss graph
                                                  (query: w, h, start, bg)
    GET /api/scrub/info                     JSON  scrub ring info
    GET /api/scrub/latest/meta              JSON  newest entry metadata
    GET /api/scrub/latest/training.png      PNG   newest training image
    GET /api/scrub/latest/output.png        PNG   newest output image
    GET /api/scrub/latest/target.png        PNG   newest target image
    GET /api/scrub/<idx>/meta               JSON  entry metadata by index
    GET /api/scrub/<idx>/training.png       PNG   training image by index
    GET /api/scrub/<idx>/output.png         PNG   output image by index
    GET /api/scrub/<idx>/target.png         PNG   target image by index
    GET /api/weight/meta                    JSON  latest weight-state snapshot
    GET /api/weight/meta/all                JSON  all retained weight-state entries
    GET /api/weight/image/info              JSON  image-cache stats
    GET /api/weight/image/latest.png        PNG   newest rendered weight image
    GET /api/weight/image/<idx>.png         PNG   weight image by index
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import io
import json
import re
import struct
import sys
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlparse

# ---------------------------------------------------------------------------
# Minimal stdlib PNG encoder (no Pillow dependency)
# ---------------------------------------------------------------------------

def _png_chunk(tag: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def _encode_png_rgb(width: int, height: int, rgb: bytes) -> bytes:
    """Encode raw RGB bytes (width*height*3) as a PNG file."""
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    stride = width * 3
    raw = b"".join(b"\x00" + rgb[y * stride:(y + 1) * stride] for y in range(height))
    idat = _png_chunk(b"IDAT", zlib.compress(raw, 1))
    iend = _png_chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


def _encode_png_rgba(width: int, height: int, rgba: bytes) -> bytes:
    """Encode raw RGBA bytes (width*height*4) as a PNG file."""
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
    stride = width * 4
    raw = b"".join(b"\x00" + rgba[y * stride:(y + 1) * stride] for y in range(height))
    idat = _png_chunk(b"IDAT", zlib.compress(raw, 1))
    iend = _png_chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


def _composite_rgba_on_black(width: int, height: int, rgba: bytes) -> bytes:
    """Composite RGBA overlay onto a black background, return RGB bytes."""
    import array
    out = array.array("B", b"\x00" * (width * height * 3))
    for i in range(width * height):
        r, g, b, a = rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2], rgba[i * 4 + 3]
        if a:
            alpha = a / 255.0
            out[i * 3]     = int(r * alpha)
            out[i * 3 + 1] = int(g * alpha)
            out[i * 3 + 2] = int(b * alpha)
    return bytes(out)


# ---------------------------------------------------------------------------
# Lazy store accessors
# ---------------------------------------------------------------------------

def _get_stores():
    """Return (loss_store, scrub_ring, weight_state_store, weight_image_store).

    Imported lazily so the module can be imported without the DLL present.
    Stores are the cross-process global singletons.
    """
    from pipeline.nodus_loss_store import (
        NodusLossStore,
        NodusScrubRing,
        NodusWeightStateStore,
        NodusWeightImageStore,
    )
    loss   = NodusLossStore.get_global()
    scrub  = NodusScrubRing.get_global()
    wstate = NodusWeightStateStore.get_global()
    wimage = NodusWeightImageStore.get_global()
    return loss, scrub, wstate, wimage


# Cached after first successful load.
_STORES: tuple | None = None


def _stores():
    global _STORES
    if _STORES is None:
        _STORES = _get_stores()
    return _STORES


def _loss():
    return _stores()[0]


def _scrub():
    return _stores()[1]


def _wstate():
    return _stores()[2]


def _wimage():
    return _stores()[3]


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _json(obj) -> bytes:
    return json.dumps(obj, allow_nan=False).encode()


def _dataclass_dict(obj) -> dict:
    return dataclasses.asdict(obj)


# ---------------------------------------------------------------------------
# Route handler
# ---------------------------------------------------------------------------

# Colour palette for graph rendering (RGBA).
_GRAPH_COLORS = [
    (255, 128,  64, 255),
    ( 64, 200, 255, 255),
    (128, 255, 128, 255),
    (255, 220,  60, 255),
    (200,  80, 255, 255),
    (255,  80, 120, 255),
    ( 80, 255, 200, 255),
    (255, 180,  80, 255),
]


class NodusHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the nodus data server."""

    server_version = "NodusServer/1.0"
    protocol_version = "HTTP/1.1"

    # ------------------------------------------------------------------ #
    #  Entry point                                                         #
    # ------------------------------------------------------------------ #

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path.rstrip("/") or "/"
        qs     = parse_qs(parsed.query, keep_blank_values=False)

        try:
            self._dispatch(path, qs)
        except Exception as exc:
            body = _json({"error": str(exc)})
            self._send(500, "application/json", body)

    def log_message(self, fmt, *args):
        # Keep output minimal.
        sys.stderr.write(f"[nodus] {self.address_string()} {fmt % args}\n")

    # ------------------------------------------------------------------ #
    #  Routing                                                             #
    # ------------------------------------------------------------------ #

    def _dispatch(self, path: str, qs: dict):
        # Root index
        if path == "/":
            return self._handle_index()

        if path == "/api/status":
            return self._handle_status()

        # -- Loss store --
        if path == "/api/loss/channels":
            return self._handle_loss_channels()

        m = re.fullmatch(r"/api/loss/channel/(.+)/records", path)
        if m:
            return self._handle_loss_records(m.group(1), qs)

        m = re.fullmatch(r"/api/loss/channel/(.+)/latest", path)
        if m:
            return self._handle_loss_latest(m.group(1))

        if path == "/api/loss/graph.png":
            return self._handle_loss_graph(qs)

        # -- Scrub ring --
        if path == "/api/scrub/info":
            return self._handle_scrub_info()

        if path == "/api/scrub/latest/meta":
            return self._handle_scrub_meta("latest")
        if path == "/api/scrub/latest/training.png":
            return self._handle_scrub_image("latest", "training")
        if path == "/api/scrub/latest/output.png":
            return self._handle_scrub_image("latest", "output")
        if path == "/api/scrub/latest/target.png":
            return self._handle_scrub_image("latest", "target")

        m = re.fullmatch(r"/api/scrub/(\d+)/meta", path)
        if m:
            return self._handle_scrub_meta(int(m.group(1)))

        m = re.fullmatch(r"/api/scrub/(\d+)/training\.png", path)
        if m:
            return self._handle_scrub_image(int(m.group(1)), "training")

        m = re.fullmatch(r"/api/scrub/(\d+)/output\.png", path)
        if m:
            return self._handle_scrub_image(int(m.group(1)), "output")

        m = re.fullmatch(r"/api/scrub/(\d+)/target\.png", path)
        if m:
            return self._handle_scrub_image(int(m.group(1)), "target")

        # -- Weight stores --
        if path == "/api/weight/meta":
            return self._handle_weight_meta_latest()

        if path == "/api/weight/meta/all":
            return self._handle_weight_meta_all()

        if path == "/api/weight/image/info":
            return self._handle_weight_image_info()

        if path == "/api/weight/image/latest.png":
            return self._handle_weight_image("latest")

        m = re.fullmatch(r"/api/weight/image/(\d+)\.png", path)
        if m:
            return self._handle_weight_image(int(m.group(1)))

        self._send(404, "application/json", _json({"error": "not found", "path": path}))

    # ------------------------------------------------------------------ #
    #  Handlers                                                            #
    # ------------------------------------------------------------------ #

    def _handle_index(self):
        html = _INDEX_HTML
        self._send(200, "text/html; charset=utf-8", html.encode())

    def _handle_status(self):
        try:
            ls  = _loss()
            sr  = _scrub()
            wst = _wstate()
            wim = _wimage()
            body = _json({
                "ok": True,
                "loss_channels": ls.channel_count(),
                "scrub_length": sr.length(),
                "scrub_capacity": sr.capacity(),
                "weight_image_count": wim.length(),
            })
        except Exception as exc:
            body = _json({"ok": False, "error": str(exc)})
        self._send(200, "application/json", body)

    # ---- Loss store ----

    def _handle_loss_channels(self):
        ls = _loss()
        channels = []
        for key in ls.channel_keys():
            lat = ls.latest(key)
            channels.append({
                "key": key,
                "length": ls.channel_length(key),
                "cursor": ls.channel_cursor(key),
                "latest_loss": lat.loss if lat else None,
                "latest_step": lat.step if lat else None,
            })
        self._send(200, "application/json", _json({"channels": channels}))

    def _handle_loss_records(self, key: str, qs: dict):
        ls = _loss()
        from_step  = int(qs.get("from_step", ["0"])[0])
        max_rec    = int(qs.get("max", ["100000"])[0])
        records = ls.query_since(key, from_step=from_step, max_records=max_rec)
        body = _json({
            "key": key,
            "count": len(records),
            "records": [_dataclass_dict(r) for r in records],
        })
        self._send(200, "application/json", body)

    def _handle_loss_latest(self, key: str):
        ls = _loss()
        rec = ls.latest(key)
        if rec is None:
            self._send(404, "application/json", _json({"error": "channel not found or empty"}))
            return
        self._send(200, "application/json", _json(_dataclass_dict(rec)))

    def _handle_loss_graph(self, qs: dict):
        ls = _loss()
        w   = max(1, int(qs.get("w", ["800"])[0]))
        h   = max(1, int(qs.get("h", ["200"])[0]))
        start = float(qs.get("start", ["0.0"])[0])
        bg_str = qs.get("bg", ["0,0,0"])[0]

        keys = ls.channel_keys()
        if not keys:
            # Return a blank black PNG.
            blank = b"\x00" * (w * h * 3)
            self._send(200, "image/png", _encode_png_rgb(w, h, blank))
            return

        # Extend palette cyclically.
        colors = [_GRAPH_COLORS[i % len(_GRAPH_COLORS)] for i in range(len(keys))]
        result = ls.render_all_lines(keys, colors, w, h,
                                     display_start_frac=max(0.0, min(1.0, start)))
        if result is None:
            blank = b"\x00" * (w * h * 3)
            self._send(200, "image/png", _encode_png_rgb(w, h, blank))
            return

        overlay = result["overlay"]  # numpy (h, w, 4) uint8
        rgba_bytes = overlay.tobytes()

        try:
            bg_parts = [max(0, min(255, int(v))) for v in bg_str.split(",")]
            br, bg, bb = (bg_parts + [0, 0, 0])[:3]
        except Exception:
            br, bg, bb = 0, 0, 0

        import array
        rgb = array.array("B", b"\x00" * (w * h * 3))
        for i in range(w * h):
            r, g, b_, a = (rgba_bytes[i * 4], rgba_bytes[i * 4 + 1],
                           rgba_bytes[i * 4 + 2], rgba_bytes[i * 4 + 3])
            if a:
                alpha = a / 255.0
                inv = 1.0 - alpha
                rgb[i * 3]     = int(r * alpha + br * inv)
                rgb[i * 3 + 1] = int(g * alpha + bg * inv)
                rgb[i * 3 + 2] = int(b_ * alpha + bb * inv)
            else:
                rgb[i * 3]     = br
                rgb[i * 3 + 1] = bg
                rgb[i * 3 + 2] = bb

        self._send(200, "image/png", _encode_png_rgb(w, h, bytes(rgb)))

    # ---- Scrub ring ----

    def _handle_scrub_info(self):
        sr = _scrub()
        body = _json({
            "length": sr.length(),
            "capacity": sr.capacity(),
            "write_cursor": sr.write_cursor(),
        })
        self._send(200, "application/json", body)

    def _handle_scrub_meta(self, index):
        sr = _scrub()
        idx = self._resolve_scrub_index(sr, index)
        if idx is None:
            self._send(404, "application/json", _json({"error": "empty ring"}))
            return
        meta = sr.get_meta(idx)
        if meta is None:
            self._send(404, "application/json", _json({"error": "index out of range"}))
            return
        d = _dataclass_dict(meta)
        d["index"] = idx
        self._send(200, "application/json", _json(d))

    def _handle_scrub_image(self, index, kind: str):
        sr = _scrub()
        idx = self._resolve_scrub_index(sr, index)
        if idx is None:
            self._send(404, "application/json", _json({"error": "empty ring"}))
            return
        meta = sr.get_meta(idx)
        if meta is None:
            self._send(404, "application/json", _json({"error": "index out of range"}))
            return

        if kind == "training":
            raw = sr.copy_training_image(idx)
            w, h = meta.image_w, meta.image_h
            channels = 4
        elif kind == "output":
            raw = sr.copy_output_image(idx)
            w, h = meta.output_w, meta.output_h
            channels = 4
        elif kind == "target":
            raw = sr.copy_target(idx)
            # Target is stored as RGBA same dimensions as training image.
            w, h = meta.image_w, meta.image_h
            channels = 4
        else:
            self._send(400, "application/json", _json({"error": "unknown image kind"}))
            return

        if raw is None or len(raw) == 0:
            self._send(404, "application/json", _json({"error": "image not present"}))
            return

        if w == 0 or h == 0:
            # Fall back to known scrub image size.
            from pipeline.nodus_loss_store import SCRUB_IMAGE_W, SCRUB_IMAGE_H
            w, h = SCRUB_IMAGE_W, SCRUB_IMAGE_H

        rgba = bytes(raw[:w * h * channels])
        self._send(200, "image/png", _encode_png_rgba(w, h, rgba))

    @staticmethod
    def _resolve_scrub_index(sr, index) -> Optional[int]:
        """Resolve 'latest' or integer index to a logical ring index."""
        length = sr.length()
        if length == 0:
            return None
        if index == "latest":
            return length - 1
        idx = int(index)
        if idx < 0 or idx >= length:
            return None
        return idx

    # ---- Weight stores ----

    def _handle_weight_meta_latest(self):
        wst = _wstate()
        meta = wst.get_meta()
        if meta is None:
            self._send(404, "application/json", _json({"error": "no weight state published"}))
            return
        self._send(200, "application/json", _json(_dataclass_dict(meta)))

    def _handle_weight_meta_all(self):
        wst = _wstate()
        count = wst.count()
        entries = []
        for i in range(count):
            meta = wst.get_meta_at(i)
            if meta is not None:
                entries.append(_dataclass_dict(meta))
        self._send(200, "application/json", _json({"count": count, "entries": entries}))

    def _handle_weight_image_info(self):
        wim = _wimage()
        stats = wim.stats()
        if stats is None:
            self._send(500, "application/json", _json({"error": "could not read stats"}))
            return
        self._send(200, "application/json", _json(_dataclass_dict(stats)))

    def _handle_weight_image(self, index):
        wim = _wimage()
        count = wim.length()
        if count == 0:
            self._send(404, "application/json", _json({"error": "no weight images cached"}))
            return

        if index == "latest":
            idx = count - 1
        else:
            idx = int(index)
            if idx < 0 or idx >= count:
                self._send(404, "application/json", _json({"error": "index out of range"}))
                return

        meta = wim.get_meta(idx)
        if meta is None:
            self._send(404, "application/json", _json({"error": "metadata unavailable"}))
            return

        raw = wim.copy_image(idx)
        if raw is None or len(raw) == 0:
            self._send(404, "application/json", _json({"error": "image data unavailable"}))
            return

        w, h, c = meta.width, meta.height, meta.channels
        flat = bytes(raw.reshape(-1))
        if c == 3:
            self._send(200, "image/png", _encode_png_rgb(w, h, flat))
        elif c == 4:
            self._send(200, "image/png", _encode_png_rgba(w, h, flat))
        else:
            self._send(500, "application/json",
                       _json({"error": f"unexpected channel count {c}"}))

    # ------------------------------------------------------------------ #
    #  Response helper                                                     #
    # ------------------------------------------------------------------ #

    def _send(self, code: int, content_type: str, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# HTML index page
# ---------------------------------------------------------------------------

_INDEX_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Nodus Data Server</title>
<style>
  body { font-family: monospace; background: #111; color: #ccc; padding: 2em; }
  h1   { color: #fa0; }
  h2   { color: #8cf; margin-top: 1.5em; }
  a    { color: #6df; }
  table { border-collapse: collapse; margin: 0.5em 0; }
  td, th { border: 1px solid #444; padding: 0.3em 0.8em; text-align: left; }
  th   { background: #222; color: #8cf; }
  code { background: #222; padding: 0.1em 0.4em; border-radius: 3px; }
</style>
</head>
<body>
<h1>Nodus Data Server</h1>
<p>Read-only HTTP interface to the nodus shared-memory training data stores.</p>

<h2>Status</h2>
<table>
  <tr><th>Endpoint</th><th>Description</th></tr>
  <tr><td><a href="/api/status">/api/status</a></td><td>Overall store status (JSON)</td></tr>
</table>

<h2>Loss Store</h2>
<table>
  <tr><th>Endpoint</th><th>Description</th></tr>
  <tr><td><a href="/api/loss/channels">/api/loss/channels</a></td><td>All channels + summary</td></tr>
  <tr><td>/api/loss/channel/&lt;key&gt;/records?from_step=0&amp;max=10000</td><td>Records for a channel</td></tr>
  <tr><td>/api/loss/channel/&lt;key&gt;/latest</td><td>Most recent record</td></tr>
  <tr><td><a href="/api/loss/graph.png?w=800&amp;h=200">/api/loss/graph.png</a>
      &nbsp;<small>?w=800&amp;h=200&amp;start=0.0&amp;bg=0,0,0</small></td>
      <td>Rendered loss graph (PNG)</td></tr>
</table>

<h2>Scrub Ring</h2>
<table>
  <tr><th>Endpoint</th><th>Description</th></tr>
  <tr><td><a href="/api/scrub/info">/api/scrub/info</a></td><td>Ring length / capacity / cursor</td></tr>
  <tr><td><a href="/api/scrub/latest/meta">/api/scrub/latest/meta</a></td><td>Newest entry metadata</td></tr>
  <tr><td><a href="/api/scrub/latest/training.png">/api/scrub/latest/training.png</a></td><td>Newest training image</td></tr>
  <tr><td><a href="/api/scrub/latest/output.png">/api/scrub/latest/output.png</a></td><td>Newest output image</td></tr>
  <tr><td><a href="/api/scrub/latest/target.png">/api/scrub/latest/target.png</a></td><td>Newest target image</td></tr>
  <tr><td>/api/scrub/&lt;idx&gt;/meta</td><td>Entry metadata by index</td></tr>
  <tr><td>/api/scrub/&lt;idx&gt;/training.png</td><td>Training image by index</td></tr>
  <tr><td>/api/scrub/&lt;idx&gt;/output.png</td><td>Output image by index</td></tr>
  <tr><td>/api/scrub/&lt;idx&gt;/target.png</td><td>Target image by index</td></tr>
</table>

<h2>Weight Stores</h2>
<table>
  <tr><th>Endpoint</th><th>Description</th></tr>
  <tr><td><a href="/api/weight/meta">/api/weight/meta</a></td><td>Latest weight-state snapshot</td></tr>
  <tr><td><a href="/api/weight/meta/all">/api/weight/meta/all</a></td><td>All retained weight-state entries</td></tr>
  <tr><td><a href="/api/weight/image/info">/api/weight/image/info</a></td><td>Image-cache stats</td></tr>
  <tr><td><a href="/api/weight/image/latest.png">/api/weight/image/latest.png</a></td><td>Newest rendered weight image</td></tr>
  <tr><td>/api/weight/image/&lt;idx&gt;.png</td><td>Weight image by index</td></tr>
</table>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Serve nodus shared-memory data over HTTP (read-only).")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7272,
                        help="TCP port (default: 7272)")
    args = parser.parse_args()

    # Eagerly load the stores so startup errors surface immediately.
    print(f"[nodus] Loading shared-memory stores ...", flush=True)
    try:
        loss, scrub, wst, wim = _get_stores()
        _STORES_ref = (loss, scrub, wst, wim)
        global _STORES
        _STORES = _STORES_ref
        print(f"[nodus] Connected: {loss.channel_count()} loss channel(s), "
              f"scrub ring {scrub.length()}/{scrub.capacity()}, "
              f"{wim.length()} weight image(s).", flush=True)
    except Exception as exc:
        print(f"[nodus] WARNING: Could not load stores: {exc}", flush=True)
        print(f"[nodus] Server will still start; store errors will be "
              f"reported per-request.", flush=True)

    server = ThreadingHTTPServer((args.host, args.port), NodusHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"[nodus] Serving at {url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[nodus] Shutting down.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
