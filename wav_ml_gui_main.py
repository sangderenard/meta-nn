#!/usr/bin/env python
"""Standalone GUI viewer for nodus training visualization.

Launch this *before* the training pipeline to get an immediate viewer window
with historical loss graphs, checkpoint markers, and weight-backup scans loaded
from leftover files.  The training process connects via IPC to deliver live
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


def _load_history(viewer, out_dir: Path) -> None:
    """Scan the output directory and load historical data into the viewer."""
    from pipeline.plan_protocol import (
        DEFAULT_PLAN_FILENAME,
        DEFAULT_RUNTIME_SNAPSHOT_FILENAME,
        TrainingGraphPlan,
        is_protocol_envelope_message,
        parse_envelope,
    )
    from wav_ml_viewer import (
        LOSS_STAGE_CLASSIFIER,
        LOSS_STAGE_GENERATOR,
        LOSS_STAGE_DISCRIMINATOR,
        LOSS_STAGE_TRANSFORMER,
        LOSS_STAGE_WAVE_CLASSIFIER,
        LOSS_STAGE_WAVE_CLASSIFIER_EVAL,
        _LossFileLogger,
    )

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
                        viewer.update_loss(LOSS_STAGE_CLASSIFIER, v)
                        loaded += 1
                except Exception:
                    pass
            for row in data.get("generator_history", []):
                try:
                    gv = float(row.get("g_loss", float("nan")))
                    dv = float(row.get("d_loss", float("nan")))
                    if math.isfinite(gv):
                        viewer.update_loss(LOSS_STAGE_GENERATOR, gv)
                        loaded += 1
                    if math.isfinite(dv):
                        viewer.update_loss(LOSS_STAGE_DISCRIMINATOR, dv)
                        loaded += 1
                except Exception:
                    pass
            for row in data.get("transformer_history", []):
                try:
                    v = float(row.get("train_loss", float("nan")))
                    if math.isfinite(v):
                        viewer.update_loss(LOSS_STAGE_TRANSFORMER, v)
                        loaded += 1
                except Exception:
                    pass
            for row in data.get("wave_classifier_history", []):
                try:
                    v = float(row.get("train_loss", float(row.get("loss", float("nan")))))
                    if math.isfinite(v):
                        viewer.update_loss(LOSS_STAGE_WAVE_CLASSIFIER, v)
                        loaded += 1
                except Exception:
                    pass
                try:
                    ev = float(row.get("val_loss", float("nan")))
                    if math.isfinite(ev):
                        viewer.update_loss(LOSS_STAGE_WAVE_CLASSIFIER_EVAL, ev)
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
            viewer.update_loss(int(rec["stage"]), float(rec["loss"]), ts=float(rec["ts"]))
        if len(recs) > 0:
            print(f"[gui] loaded {len(recs)} records from loss_log_prev.bin", flush=True)
    except Exception as e:
        print(f"[gui] could not load loss_log_prev.bin: {e}", flush=True)

    # ── loss_log.bin (current/most recent session, may still be accumulating) ─
    cur_path = out_dir / "loss_log.bin"
    try:
        recs = _LossFileLogger.load(cur_path)
        for rec in recs:
            viewer.update_loss(int(rec["stage"]), float(rec["loss"]), ts=float(rec["ts"]))
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

    # ── Batch-launcher weight backup directory ────────────────────────────
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
    args = parser.parse_args()

    # Import viewer after arg parse so the window opens as fast as possible
    from wav_ml_viewer import _TransformerStatusOpenGLViewer, ViewerIPCServer

    image_hw = (int(args.image_size), int(args.image_size))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    viewer = _TransformerStatusOpenGLViewer(
        enabled=True,
        image_hw=image_hw,
        scale=max(1, int(args.scale)),
        cycle_slots=max(0, int(args.cycle_slots)),
    )

    # Load history from leftover files before any training connects
    _load_history(viewer, out_dir)

    # Start IPC server for training processes to connect
    port_file = args.port_file or str(out_dir / ".viewer_port")
    server = ViewerIPCServer(viewer, port=int(args.port), port_file=port_file)
    server.start()

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
            pf = Path(port_file)
            if pf.exists():
                pf.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()
