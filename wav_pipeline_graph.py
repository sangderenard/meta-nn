"""
wav_pipeline_graph.py — graph-based entry point for the wav ML training pipeline.

This is a thin shim that:
  1. Parses CLI arguments (reusing parse_args from the original pipeline)
  2. Sets up the output directory
  3. Hands off to pipeline.orchestrator.run()

The 23 000-line monolith (wav_config_transformer_pipeline.py) still exists and
provides the underlying utility functions and training primitives that each node
delegates to.  The graph layer sits on top and replaces only main() and the ad-hoc
orchestration logic.

Usage
-----
  python wav_pipeline_graph.py [same CLI args as before]

To print the graph topology without running:
  python wav_pipeline_graph.py --graph-summary

To print a legend of node IDs and descriptions:
  python wav_pipeline_graph.py --graph-legend
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Probe-import scipy.io BEFORE any torch/pipeline import to prevent Windows
# DLL ordering conflict that causes STATUS_HEAP_CORRUPTION (0xC0000374).
try:
    import scipy.io as _scipy_io_probe  # noqa: F401
    del _scipy_io_probe
except ImportError:
    pass


def main() -> None:
    graph_cli, passthrough_argv = _parse_graph_cli(sys.argv)

    # -- Quick flags that don't need the full argparse ----------------------
    if graph_cli.graph_summary:
        _print_graph_summary()
        return

    if graph_cli.graph_legend:
        _print_graph_legend()
        return

    # -- Phase 3: plan-driven bootstrap ------------------------------------
    if str(graph_cli.from_plan).strip():
        _run_from_plan(
            plan_path=Path(str(graph_cli.from_plan).strip()),
            override_output_dir=str(graph_cli.from_plan_output_dir or "").strip() or None,
        )
        return

    # -- Full argument parsing (reuse existing parse_args) -----------------
    original_argv = list(sys.argv)
    try:
        sys.argv = list(passthrough_argv)
        from pipeline.cli import parse_args
        args = parse_args()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[wav_pipeline_graph] ERROR: argument parsing failed: {exc}", flush=True)
        sys.exit(1)
    finally:
        sys.argv = original_argv

    # -- Output directory --------------------------------------------------
    output_dir = Path(str(getattr(args, "output_dir", "output")).strip() or "output")
    output_dir.mkdir(parents=True, exist_ok=True)

    if graph_cli.write_graph_plan or graph_cli.graph_plan_only:
        graph_plan_path = (
            Path(str(graph_cli.write_graph_plan).strip())
            if str(graph_cli.write_graph_plan or "").strip()
            else (output_dir / "training_graph_plan.json")
        )
        _write_graph_plan(args=args, output_dir=output_dir, plan_path=graph_plan_path)
        if graph_cli.graph_plan_only:
            return

    # -- Optional: print topology before running --------------------------
    if bool(getattr(args, "print_graph", False)):
        _print_graph_summary(args=args)

    # -- Run ---------------------------------------------------------------
    from pipeline.orchestrator import run
    try:
        run(args=args, output_dir=output_dir)
    except KeyboardInterrupt:
        print("\n[wav_pipeline_graph] interrupted by user", flush=True)
        sys.exit(0)
    except Exception as exc:
        import traceback
        print(f"[wav_pipeline_graph] FATAL: {exc}", flush=True)
        traceback.print_exc()
        sys.exit(1)


def _parse_graph_cli(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--graph-summary", action="store_true")
    parser.add_argument("--graph-legend", action="store_true")
    parser.add_argument("--write-graph-plan", default="")
    parser.add_argument("--graph-plan-only", action="store_true")
    # Phase 3: plan-driven bootstrap — start worker from a saved plan file
    # without needing all 470 original CLI args.
    parser.add_argument("--from-plan", default="",
                        help="Path to training_graph_plan.json; start worker from saved plan.")
    parser.add_argument("--from-plan-output-dir", default="",
                        help="Override output directory when starting from a saved plan.")
    opts, remaining = parser.parse_known_args(list(argv[1:]))
    return opts, [argv[0], *remaining]


def _write_graph_plan(args, output_dir: Path, plan_path: Path) -> None:
    from pipeline.orchestrator import build_training_graph_plan

    plan = build_training_graph_plan(args=args, output_dir=output_dir)
    plan.save_json(plan_path)
    print(
        f"[wav_pipeline_graph] graph plan written: {plan_path} "
        f"({len(plan.nodes)} nodes, {len(plan.edges)} edges)",
        flush=True,
    )


def _run_from_plan(plan_path: Path, override_output_dir: str = None) -> None:
    """Phase 3: start the worker from a saved TrainingGraphPlan.

    Loads hyperparameters from *plan_path* and reconstructs the pipeline graph
    from the plan's config_blobs without needing all 470 original CLI args.
    The plan's worker_hints supply the remaining runtime parameters (device,
    cycles, rounds, output directory).

    An optional *override_output_dir* lets the caller redirect output without
    editing the plan file.
    """
    import types
    from pipeline.plan_protocol import TrainingGraphPlan
    from pipeline.orchestrator import run

    print(f"[wav_pipeline_graph] loading plan from {plan_path}", flush=True)
    plan = TrainingGraphPlan.load_json(plan_path)
    hints = plan.worker_hints or {}

    output_dir_str = (
        override_output_dir
        or str(hints.get("output_dir", "output")).strip()
        or "output"
    )
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build a minimal args namespace from plan's worker_hints.
    # The plan carries the node config blobs; args only provide runtime knobs.
    args = types.SimpleNamespace(
        output_dir=str(output_dir),
        device=str(hints.get("device_preference", "auto")),
        orchestration_mode=str(hints.get("orchestration_mode", "staged_cgrw")),
        orchestration_cycles=int(hints.get("orchestration_cycles", 1)),
        orchestration_rounds=int(hints.get("orchestration_rounds", 1)),
        cycles=int(hints.get("orchestration_cycles", 1)),
        rounds_per_cycle=int(hints.get("orchestration_rounds", 1)),
    )

    print(
        f"[wav_pipeline_graph] plan-driven bootstrap: "
        f"device={args.device} cycles={args.cycles} rounds={args.rounds_per_cycle} "
        f"output={output_dir}",
        flush=True,
    )

    try:
        run(args=args, output_dir=output_dir, initial_plan=plan)
    except KeyboardInterrupt:
        print("\n[wav_pipeline_graph] interrupted by user", flush=True)
        sys.exit(0)
    except Exception as exc:
        import traceback
        print(f"[wav_pipeline_graph] FATAL (from-plan): {exc}", flush=True)
        traceback.print_exc()
        sys.exit(1)


def _print_graph_summary(args=None) -> None:
    """Print the wired graph topology and exit."""
    from pipeline.orchestrator import build_pipeline_graph, _build_configs_from_args

    if args is None:
        # Use all defaults
        class _FakeArgs:
            def __getattr__(self, _):
                return None
        args = _FakeArgs()

    cfg = _build_configs_from_args(args)
    graph = build_pipeline_graph(
        classifier_cfg=cfg["classifier"],
        transformer_cfg=cfg["transformer"],
        generator_cfg=cfg["generator"],
        wave_cfg=cfg["wave"],
        vocab_cfg=cfg["vocab"],
        embedding_cfg=cfg["embedding"],
        wave_pool_cfg=cfg["wave_pool"],
        berkeley_gate_cfg=cfg["berkeley_gate"],
        transformer_gate_cfg=cfg["transformer_gate"],
        generator_gate_cfg=cfg["generator_gate"],
        wave_gate_cfg=cfg["wave_gate"],
        save_every_n_rounds=1,
        berkeley_refresh_every_n_rounds=4,
    )
    print(graph.summary())


def _print_graph_legend() -> None:
    """Print a table of node_id → description for all registered nodes."""
    from pipeline.orchestrator import build_pipeline_graph, _build_configs_from_args

    class _FakeArgs:
        def __getattr__(self, _):
            return None

    cfg = _build_configs_from_args(_FakeArgs())
    graph = build_pipeline_graph(
        classifier_cfg=cfg["classifier"],
        transformer_cfg=cfg["transformer"],
        generator_cfg=cfg["generator"],
        wave_cfg=cfg["wave"],
        vocab_cfg=cfg["vocab"],
        embedding_cfg=cfg["embedding"],
        wave_pool_cfg=cfg["wave_pool"],
        berkeley_gate_cfg=cfg["berkeley_gate"],
        transformer_gate_cfg=cfg["transformer_gate"],
        generator_gate_cfg=cfg["generator_gate"],
        wave_gate_cfg=cfg["wave_gate"],
        save_every_n_rounds=1,
        berkeley_refresh_every_n_rounds=4,
    )

    print("\nNode legend:")
    print(f"  {'node_id':<45} description")
    print("  " + "-" * 80)
    for nid, node in sorted(graph._nodes.items()):
        print(f"  {nid:<45} {node.description}")
    print()


if __name__ == "__main__":
    main()
