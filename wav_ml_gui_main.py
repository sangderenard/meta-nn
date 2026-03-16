#!/usr/bin/env python
"""Standalone GUI viewer for nodus training visualization.

Launch this *before* the training pipeline to get an immediate viewer window
with historical loss graphs and checkpoint markers loaded from leftover files.
The training process connects via IPC to deliver live
updates (loss values, preview frames, checkpoint notifications).

Usage (manual):
    python wav_ml_gui_main.py --output-dir path/to/output --image-size 64 --scale 1

The batch launcher (run_iterative_wav_pipeline.bat) starts this automatically
and passes the port-file path so the pipeline can find the IPC port.
"""

import argparse
import math
import time
from pathlib import Path
from multiprocessing.connection import Client


def _probe_existing_viewer(port_file: Path) -> bool:
    """Return True when an existing standalone GUI is reachable via the port file."""
    try:
        if not port_file.exists():
            return False
        raw = port_file.read_text(encoding="utf-8").strip()
        if not raw:
            return False
        port = int(raw)
        conn = Client(("localhost", port), family="AF_INET", authkey=b"nodus_viewer_v1")
        conn.close()
        return True
    except Exception:
        return False


def _load_history(viewer, out_dir: Path, loss_store=None) -> None:
    """Scan the output directory and load historical data into the viewer.

    When *loss_store* is provided (a NodusLossStore instance), historical
    loss values are written directly into the native store instead of the
    viewer's local deques.
    """
    from pipeline.plan_protocol import (
        DEFAULT_PLAN_FILENAME,
        DEFAULT_RUNTIME_SNAPSHOT_FILENAME,
        TrainingGraphPlan,
        is_protocol_envelope_message,
        parse_envelope,
    )
    from wav_ml_viewer import (
        _LossFileLogger,
    )

    def _record(ck: str, v: float, ts: float = 0.0):
        if loss_store is not None:
            loss_store.record(ck, v, ts=ts)
        else:
            viewer.update_loss(ck, v, ts=ts)

    loaded = 0

    # ── summary.json (high-level per-round history) ───────────────────────
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        try:
            import json
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            for row in data.get("berkeley_refresh_history", []):
                try:
                    v = float(row.get("loss", float("nan")))
                    if math.isfinite(v):
                        _record("berk", v)
                        loaded += 1
                except Exception:
                    pass
            for row in data.get("generator_history", []):
                try:
                    gv = float(row.get("g_loss", float("nan")))
                    dv = float(row.get("d_loss", float("nan")))
                    if math.isfinite(gv):
                        _record("gen", gv)
                        loaded += 1
                    if math.isfinite(dv):
                        _record("disc", dv)
                        loaded += 1
                except Exception:
                    pass
            for row in data.get("transformer_history", []):
                try:
                    v = float(row.get("train_loss", float("nan")))
                    if math.isfinite(v):
                        _record("trans", v)
                        loaded += 1
                except Exception:
                    pass
            for row in data.get("wave_classifier_history", []):
                try:
                    v = float(row.get("train_loss", float(row.get("loss", float("nan")))))
                    if math.isfinite(v):
                        _record("wcls", v)
                        loaded += 1
                except Exception:
                    pass
                try:
                    ev = float(row.get("val_loss", float("nan")))
                    if math.isfinite(ev):
                        _record("wcls_eval", ev)
                        loaded += 1
                except Exception:
                    pass
            if loaded > 0:
                print(f"[gui] loaded {loaded} values from summary.json", flush=True)
        except Exception as e:
            print(f"[gui] could not load summary.json: {e}", flush=True)

    # -- training_graph_plan.json (logical graph plan) --------------------
    plan_path = out_dir / DEFAULT_PLAN_FILENAME
    if plan_path.exists():
        try:
            plan = TrainingGraphPlan.load_json(plan_path)
            if hasattr(viewer, "set_training_graph_plan"):
                viewer.set_training_graph_plan(plan.to_dict())
            print(
                f"[gui] loaded {DEFAULT_PLAN_FILENAME} "
                f"({len(plan.nodes)} nodes, {len(plan.edges)} edges)",
                flush=True,
            )
        except Exception as e:
            print(f"[gui] could not load {DEFAULT_PLAN_FILENAME}: {e}", flush=True)

    # -- graph_runtime_snapshot.json (latest worker runtime state) --------
    runtime_path = out_dir / DEFAULT_RUNTIME_SNAPSHOT_FILENAME
    if runtime_path.exists():
        try:
            import json

            runtime_blob = json.loads(runtime_path.read_text(encoding="utf-8"))
            if is_protocol_envelope_message(runtime_blob):
                _, payload = parse_envelope(runtime_blob)
                if hasattr(viewer, "set_training_graph_runtime"):
                    viewer.set_training_graph_runtime(payload.to_dict())
                print(
                    f"[gui] loaded {DEFAULT_RUNTIME_SNAPSHOT_FILENAME} "
                    f"(state={payload.execution_state})",
                    flush=True,
                )
        except Exception as e:
            print(f"[gui] could not load {DEFAULT_RUNTIME_SNAPSHOT_FILENAME}: {e}", flush=True)

    # ── loss_log_prev.bin (binary records from previous session) ──────────
    prev_path = out_dir / "loss_log_prev.bin"
    try:
        recs = _LossFileLogger.load(prev_path)
        for rec in recs:
            ck = rec["channel_key"].decode("utf-8").rstrip("\x00") if rec["channel_key"] else ""
            if not ck:
                ck = f"stage_{int(rec['stage'])}"
            _record(ck, float(rec["loss"]), ts=float(rec["ts"]))
        if len(recs) > 0:
            print(f"[gui] loaded {len(recs)} records from loss_log_prev.bin", flush=True)
    except Exception as e:
        print(f"[gui] could not load loss_log_prev.bin: {e}", flush=True)

    # ── loss_log.bin (current/most recent session, may still be accumulating) ─
    cur_path = out_dir / "loss_log.bin"
    try:
        recs = _LossFileLogger.load(cur_path)
        for rec in recs:
            ck = rec["channel_key"].decode("utf-8").rstrip("\x00") if rec["channel_key"] else ""
            if not ck:
                ck = f"stage_{int(rec['stage'])}"
            _record(ck, float(rec["loss"]), ts=float(rec["ts"]))
        if len(recs) > 0:
            print(f"[gui] loaded {len(recs)} records from loss_log.bin", flush=True)
    except Exception:
        pass  # May not exist yet

    # ── Checkpoint markers from .pt files ─────────────────────────────────
    ckpt_candidates = [
        out_dir / "pipeline_checkpoint.pt",
        out_dir / "classifier.pt",
        out_dir / "transformer.pt",
        out_dir / "generator.pt",
        out_dir / "discriminator.pt",
        out_dir / "wave_classifier.pt",
    ]
    seen_times: list = []
    for cp in ckpt_candidates:
        if not cp.exists():
            continue
        mt = cp.stat().st_mtime
        if not any(abs(mt - t) < 1.0 for t in seen_times):
            seen_times.append(mt)
            viewer.notify_checkpoint_at_walltime(mt)
    if seen_times:
        print(f"[gui] placed {len(seen_times)} checkpoint marker(s) on graph", flush=True)

    # ── Batch-launcher checkpoint backup directory ────────────────────────
    bk_dir = out_dir / "_weight_backup"
    if bk_dir.is_dir():
        viewer.set_checkpoint_backup_dir(bk_dir)
        print(f"[gui] scanned batch backup dir: {bk_dir}", flush=True)

    # ── Trim graph left edge to 5% before earliest checkpoint marker ──────
    viewer.trim_graph_to_first_checkpoint()


def main():
    parser = argparse.ArgumentParser(
        description="Nodus training viewer — standalone GUI entry point")
    parser.add_argument("--output-dir", required=True,
                        help="Training output directory to scan for history")
    parser.add_argument("--image-size", type=int, default=64,
                        help="Image H and W in pixels (square)")
    parser.add_argument("--scale", type=int, default=3,
                        help="Display scale factor")
    parser.add_argument("--cycle-slots", type=int, default=0,
                        help="Number of orchestration cycle toggle slots")
    parser.add_argument("--port", type=int, default=0,
                        help="IPC listen port (0 = auto-assign)")
    parser.add_argument("--port-file", default=None,
                        help="Write the actual IPC port number to this file")
    parser.add_argument("--launch-script", default=None,
                        help="Path to the training launcher script (.bat) for the START button")
    args = parser.parse_args()

    image_hw = (int(args.image_size), int(args.image_size))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    port_file = Path(args.port_file) if args.port_file else (out_dir / ".viewer_port")

    # Refuse to launch a duplicate GUI when an existing viewer is already reachable.
    if _probe_existing_viewer(port_file):
        print(f"[gui] existing viewer detected via {port_file}; skipping duplicate launch", flush=True)
        return

    # Import viewer after arg parse so the window opens as fast as possible
    from wav_ml_viewer import _TransformerStatusOpenGLViewer, ViewerIPCServer
    from pipeline.nodus_loss_store import NodusLossStore

    # Create the native loss store — single authoritative source for all loss data.
    loss_store = NodusLossStore.get_global()

    viewer = _TransformerStatusOpenGLViewer(
        enabled=True,
        image_hw=image_hw,
        scale=max(1, int(args.scale)),
        cycle_slots=max(0, int(args.cycle_slots)),
        loss_store=loss_store,
    )

    # Load history from leftover files before any training connects
    _load_history(viewer, out_dir, loss_store=loss_store)

    # Start IPC server for training processes to connect
    server = ViewerIPCServer(viewer, port=int(args.port), port_file=str(port_file))
    server.start()

    # Give the viewer a back-reference to the server (for connection state)
    # and launch info (for the START button).
    viewer.set_ipc_server(server)
    viewer.set_launch_info(
        launch_script=args.launch_script,
        output_dir=str(out_dir),
        port_file_path=str(port_file),
    )

    print("[gui] viewer ready, waiting for training process...", flush=True)

    # Main event loop — pumps the viewer and drains IPC messages
    try:
        while True:
            viewer.pump()
            server.poll()
            if viewer.stop_requested():
                break
            time.sleep(0.002)  # ~500 Hz poll; pump() self-throttles via slew
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        viewer.close()
        # Clean up port file
        try:
            if port_file.exists():
                port_file.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()
