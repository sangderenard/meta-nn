from __future__ import annotations

import errno
import tempfile
from pathlib import Path

import numpy as np

from pipeline.context import PipelineContext
from pipeline.nodes.data_nodes import _save_raw_stage_cache
from pipeline.nodes.interrupts import StageStopRequested
from pipeline.semantic_wheel_cache import _write_chunk


def _ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def _build_context(root: Path) -> tuple[PipelineContext, dict[str, Path]]:
    output_dir = root / "output"
    data_root = root / "berkeley"
    stage_cache_dir = output_dir / "semantic_stage_cache"
    paths = {
        "output_dir": output_dir,
        "data_root": data_root,
        "stage_cache_dir": stage_cache_dir,
        "accepted_wave_library": output_dir / "accepted_wave_library",
        "latent_wave_pool": output_dir / "latent_wave_pool",
        "training_supervision": output_dir / "training_supervision",
        "semantic_mask_cache": data_root / "cache" / "semantic_mask_cache",
        "semantic_wheels": data_root / "cache" / "semantic_wheels",
        "payload_bank": data_root / "cache" / "payload_bank_rgb16",
        "semantic_disk_rows": data_root / "cache" / "semantic_disk_rows_deadbeef.pkl.gz",
    }
    ctx = PipelineContext(
        output_dir=output_dir,
        berkeley_data_root=str(data_root),
        semantic_stage_cache_dir=str(stage_cache_dir),
    )
    return ctx, paths


def _touch_file(path: Path, payload: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _populate_ephemeral_targets(paths: dict[str, Path]) -> list[Path]:
    created: list[Path] = []
    for key in (
        "accepted_wave_library",
        "latent_wave_pool",
        "training_supervision",
        "stage_cache_dir",
        "semantic_mask_cache",
        "semantic_wheels",
        "payload_bank",
    ):
        marker = paths[key] / "marker.bin"
        _touch_file(marker)
        created.append(paths[key])
    _touch_file(paths["semantic_disk_rows"], payload=b"cache")
    created.append(paths["semantic_disk_rows"])
    return created


def _assert_purged(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if path.exists()]
    assert not missing, missing


def test_context_handler_purges_and_requests_save() -> None:
    print("\n--- test_context_handler_purges_and_requests_save ---")
    with tempfile.TemporaryDirectory(dir=".") as td:
        root = Path(td)
        ctx, paths = _build_context(root)
        created = _populate_ephemeral_targets(paths)
        try:
            ctx.handle_filesystem_space_emergency(
                OSError(errno.ENOSPC, "No space left on device"),
                note="unit test emergency",
                write_path=paths["stage_cache_dir"] / "payload.bin",
            )
        except StageStopRequested as exc:
            assert bool(exc.save_requested)
        else:
            raise AssertionError("expected StageStopRequested")
        assert bool(ctx.filesystem_space_emergency_pending)
        assert bool(ctx.shutdown_save_pending)
        assert "unit test emergency" in str(ctx.filesystem_space_emergency_reason)
        assert len(ctx.filesystem_space_emergency_cleanup.get("removed", [])) >= 8
        assert len(ctx.filesystem_space_emergency_cleanup.get("errors", [])) == 0
        _assert_purged(created)
        _ok("context handler purges dataloader caches and requests shutdown save")


def test_semantic_wheel_write_chunk_enospc_triggers_shutdown() -> None:
    print("\n--- test_semantic_wheel_write_chunk_enospc_triggers_shutdown ---")
    import pipeline.semantic_wheel_cache as swc

    with tempfile.TemporaryDirectory(dir=".") as td:
        root = Path(td)
        ctx, paths = _build_context(root)
        created = _populate_ephemeral_targets(paths)
        original = swc.np.savez_compressed

        def _fail_savez(*_args, **_kwargs):
            raise OSError(errno.ENOSPC, "No space left on device")

        swc.np.savez_compressed = _fail_savez
        try:
            try:
                _write_chunk(
                    root / "write_target" / "chunk_0000.npz",
                    {"images": np.zeros((1, 3, 4, 4), dtype=np.uint8)},
                    progress_control=ctx,
                )
            except StageStopRequested as exc:
                assert bool(exc.save_requested)
            else:
                raise AssertionError("expected StageStopRequested")
        finally:
            swc.np.savez_compressed = original
        assert bool(ctx.filesystem_space_emergency_pending)
        _assert_purged(created)
        _ok("semantic wheel chunk writer routes ENOSPC through shared shutdown hook")


def test_raw_stage_cache_enospc_triggers_shutdown() -> None:
    print("\n--- test_raw_stage_cache_enospc_triggers_shutdown ---")
    import pipeline.nodes.data_nodes as data_nodes

    with tempfile.TemporaryDirectory(dir=".") as td:
        root = Path(td)
        ctx, paths = _build_context(root)
        created = _populate_ephemeral_targets(paths)
        original = data_nodes.gzip.open

        def _fail_open(*_args, **_kwargs):
            raise OSError(errno.ENOSPC, "No space left on device")

        data_nodes.gzip.open = _fail_open
        try:
            try:
                _save_raw_stage_cache(
                    root / "raw_cache",
                    "deadbeef",
                    {"array": np.zeros((2, 2), dtype=np.float32)},
                    progress_control=ctx,
                )
            except StageStopRequested as exc:
                assert bool(exc.save_requested)
            else:
                raise AssertionError("expected StageStopRequested")
        finally:
            data_nodes.gzip.open = original
        assert bool(ctx.filesystem_space_emergency_pending)
        _assert_purged(created)
        _ok("raw stage cache writer routes ENOSPC through shared shutdown hook")


if __name__ == "__main__":
    test_context_handler_purges_and_requests_save()
    test_semantic_wheel_write_chunk_enospc_triggers_shutdown()
    test_raw_stage_cache_enospc_triggers_shutdown()
    print("\nFILESYSTEM_EMERGENCY_TESTS_PASSED")
