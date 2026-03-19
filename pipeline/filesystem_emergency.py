from __future__ import annotations

import errno
import gc
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pipeline.nodes.interrupts import StageStopRequested


_DISK_FULL_PATTERNS: Tuple[str, ...] = (
    "no space left on device",
    "not enough space on the disk",
    "there is not enough space on the disk",
    "disk full",
    "filesystem full",
    "no usable temporary directory",
)


def _progress_control_root(control: Any) -> Any:
    nested = getattr(control, "progress_control", None)
    return nested if nested is not None else control


def _iter_exception_chain(exc: BaseException) -> Iterable[BaseException]:
    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        yield cur
        nxt = getattr(cur, "__cause__", None)
        if nxt is None:
            nxt = getattr(cur, "__context__", None)
        cur = nxt if isinstance(nxt, BaseException) else None


def is_filesystem_space_error(exc: BaseException) -> bool:
    for item in _iter_exception_chain(exc):
        if isinstance(item, OSError):
            try:
                if int(getattr(item, "errno", -1) or -1) == int(errno.ENOSPC):
                    return True
            except Exception:
                pass
            try:
                if int(getattr(item, "winerror", -1) or -1) == 112:
                    return True
            except Exception:
                pass
        txt = str(item).strip().lower()
        if any(pattern in txt for pattern in _DISK_FULL_PATTERNS):
            return True
    return False


def _remove_path_retry(path: Path, retries: int = 6) -> Tuple[bool, str]:
    last_err = ""
    for attempt in range(max(1, int(retries))):
        try:
            if not path.exists():
                return False, ""
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            return True, ""
        except FileNotFoundError:
            return False, ""
        except PermissionError as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            gc.collect()
            time.sleep(0.15 * float(attempt + 1))
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            break
    return False, str(last_err)


def dataloader_ephemeral_cache_targets(
    *,
    output_dir: Any = None,
    berkeley_data_root: str = "",
    semantic_stage_cache_dir: str = "",
    extra_paths: Optional[Sequence[Any]] = None,
) -> List[Path]:
    targets: List[Path] = []
    out_dir = Path(output_dir) if output_dir is not None else None
    if out_dir is not None:
        targets.extend(
            [
                out_dir / "accepted_wave_library",
                out_dir / "latent_wave_pool",
                out_dir / "training_supervision",
                Path(str(semantic_stage_cache_dir).strip()) if str(semantic_stage_cache_dir).strip()
                else (out_dir / "semantic_stage_cache"),
            ]
        )
    data_root_path = Path(str(berkeley_data_root).strip() or "data/berkeley_sbd")
    cache_root = data_root_path / "cache"
    if cache_root.exists():
        targets.append(cache_root / "semantic_mask_cache")
        targets.append(cache_root / "semantic_wheels")
        targets.extend(sorted(cache_root.glob("payload_bank_rgb*")))
        targets.extend(sorted(cache_root.glob("semantic_disk_rows_*.pkl.gz")))
    for extra in list(extra_paths or []):
        if extra is None:
            continue
        try:
            txt = str(extra).strip()
        except Exception:
            txt = ""
        if txt:
            targets.append(Path(txt))
    deduped: List[Path] = []
    seen: set[str] = set()
    for target in targets:
        key = str(target)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(target)
    return deduped


def purge_dataloader_ephemeral_caches(
    *,
    output_dir: Any = None,
    berkeley_data_root: str = "",
    semantic_stage_cache_dir: str = "",
    extra_paths: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    targets = dataloader_ephemeral_cache_targets(
        output_dir=output_dir,
        berkeley_data_root=berkeley_data_root,
        semantic_stage_cache_dir=semantic_stage_cache_dir,
        extra_paths=extra_paths,
    )
    info: Dict[str, Any] = {
        "requested": [str(p) for p in targets],
        "removed": [],
        "missing": [],
        "errors": [],
    }
    for path in targets:
        if not path.exists():
            info["missing"].append(str(path))
            continue
        ok, err = _remove_path_retry(path)
        if bool(ok):
            info["removed"].append(str(path))
        else:
            info["errors"].append(f"{str(path)} ({str(err)})")
    return info


def raise_if_filesystem_space_emergency(
    control: Any,
    exc: BaseException,
    *,
    note: str = "",
    write_path: Any = None,
) -> None:
    if not is_filesystem_space_error(exc):
        return
    root = _progress_control_root(control)
    handler = getattr(root, "handle_filesystem_space_emergency", None)
    if callable(handler):
        handler(exc, note=note, write_path=write_path)
        return
    path_txt = str(write_path).strip() if write_path is not None else ""
    msg = str(note).strip() or "Filesystem space emergency during dataloader cache write"
    if path_txt:
        msg = f"{msg} [{path_txt}]"
    raise StageStopRequested(msg, save_requested=True)
