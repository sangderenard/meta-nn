from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Optional

from tqdm import tqdm

from pipeline.nodes.interrupts import StageStopRequested


_CURRENT_PROGRESS_CONTROL: ContextVar[Any] = ContextVar("pipeline_progress_control", default=None)
_DATALOADER_PATCHED = False


def current_progress_control() -> Any:
    try:
        return _CURRENT_PROGRESS_CONTROL.get()
    except Exception:
        return None


@contextmanager
def register_progress_control(control: Any):
    token = _CURRENT_PROGRESS_CONTROL.set(control)
    try:
        yield control
    finally:
        _CURRENT_PROGRESS_CONTROL.reset(token)


def _resolve_progress_control(control: Any = None) -> Any:
    return control if control is not None else current_progress_control()


def _progress_control_root(control: Any) -> Any:
    control = _resolve_progress_control(control)
    nested = getattr(control, "progress_control", None)
    return nested if nested is not None else control


def _control_callable(control: Any, name: str) -> Optional[Any]:
    root = _progress_control_root(control)
    if root is None:
        return None
    fn = getattr(root, name, None)
    return fn if callable(fn) else None


def _control_pump(control: Any) -> Optional[Any]:
    root = _progress_control_root(control)
    if root is None:
        return None
    fn = getattr(root, "pump", None)
    if callable(fn):
        return fn
    proxy = getattr(root, "viewer_proxy", None)
    fn = getattr(proxy, "pump", None)
    return fn if callable(fn) else None


def control_callable(name: str, control: Any = None) -> Optional[Any]:
    return _control_callable(control, name)


def control_pump(control: Any = None) -> Optional[Any]:
    return _control_pump(control)


def control_stop_requested(control: Any = None, *, stop_requested: Any = None) -> bool:
    fn = stop_requested if callable(stop_requested) else _control_callable(control, "stop_requested")
    if fn is None:
        return False
    try:
        return bool(fn())
    except Exception:
        return False


def stop_save_requested(control: Any = None) -> bool:
    fn = _control_callable(control, "shutdown_save")
    if fn is None:
        return False
    try:
        return bool(fn())
    except Exception:
        return False


def raise_if_stop_requested(control: Any = None, *, stop_requested: Any = None) -> None:
    if not control_stop_requested(control, stop_requested=stop_requested):
        return
    save_requested = stop_save_requested(control)
    raise StageStopRequested(
        "GUI stop+save requested" if save_requested else "GUI stop requested",
        save_requested=save_requested,
    )


def wait_for_resume(
    control: Any = None,
    *,
    pause_requested: Any = None,
    stop_requested: Any = None,
    pump: Any = None,
    pause_poll_s: float = 0.05,
) -> bool:
    pause_fn = pause_requested if callable(pause_requested) else _control_callable(control, "paused")
    stop_fn = stop_requested if callable(stop_requested) else _control_callable(control, "stop_requested")
    pump_fn = pump if callable(pump) else _control_pump(control)
    poll_s = max(0.01, float(pause_poll_s))
    while True:
        if control_stop_requested(control, stop_requested=stop_fn):
            return True
        if pause_fn is None:
            return False
        try:
            paused = bool(pause_fn())
        except Exception:
            return False
        if not paused:
            return False
        if pump_fn is not None:
            try:
                pump_fn()
            except Exception:
                pass
        time.sleep(poll_s)


def check_control(control: Any = None, *, pause_poll_s: float = 0.05) -> None:
    while True:
        raise_if_stop_requested(control)
        pause_fn = _control_callable(control, "paused")
        if pause_fn is None:
            return
        try:
            paused = bool(pause_fn())
        except Exception:
            return
        if not paused:
            return
        pump_fn = _control_pump(control)
        if pump_fn is not None:
            try:
                pump_fn()
            except Exception:
                pass
        time.sleep(max(0.01, float(pause_poll_s)))


class InterruptibleIterator:
    """Generic iterator wrapper that honors the registered progress control."""

    def __init__(self, iterator, *, control: Any = None, pause_poll_s: float = 0.05):
        self._iterator = iterator
        self.progress_control = _resolve_progress_control(control)
        self.pause_poll_s = max(0.01, float(pause_poll_s))

    def __iter__(self):
        return self

    def __next__(self):
        check_control(self.progress_control, pause_poll_s=self.pause_poll_s)
        item = next(self._iterator)
        check_control(self.progress_control, pause_poll_s=self.pause_poll_s)
        return item


def _install_interruptible_dataloader_patch() -> None:
    global _DATALOADER_PATCHED
    if _DATALOADER_PATCHED:
        return
    try:
        from torch.utils.data import DataLoader
    except Exception:
        return

    original_iter = getattr(DataLoader, "__iter__", None)
    if not callable(original_iter):
        return

    def _interruptible_iter(self):
        iterator = original_iter(self)
        control = current_progress_control()
        if control is None:
            return iterator
        return InterruptibleIterator(iterator, control=control)

    DataLoader.__iter__ = _interruptible_iter
    _DATALOADER_PATCHED = True


_install_interruptible_dataloader_patch()


class InterruptibleTqdm(tqdm):
    """tqdm wrapper that honors pipeline pause/stop controls."""

    def __init__(self, *args, control: Any = None, pause_poll_s: float = 0.05, **kwargs):
        self.progress_control = _resolve_progress_control(control)
        self.pause_poll_s = max(0.01, float(pause_poll_s))
        super().__init__(*args, **kwargs)

    def _raise_stop(self) -> None:
        raise_if_stop_requested(self.progress_control)

    def check_control(self) -> None:
        check_control(self.progress_control, pause_poll_s=self.pause_poll_s)

    def __enter__(self):
        super().__enter__()
        self.check_control()
        return self

    def __iter__(self):
        iterable = self.iterable  # set by tqdm.__init__ when an iterable is passed
        if iterable is None:
            # Manual progress-bar mode (tqdm(total=N)) — not an iterator, fall through
            # to parent so callers that mistakenly iterate get a clear tqdm error.
            yield from super().__iter__()
            return
        try:
            for item in iterable:
                self.check_control()
                yield item
                # Post-item check: stop/pause is caught here after the loop body
                # runs, before the next item is fetched and the bar increments.
                self.check_control()
                super().update(1)
        finally:
            try:
                self.close()
            except Exception:
                pass

    def update(self, n=1):
        self.check_control()
        return super().update(n)


def interruptible_tqdm(*args, control: Any = None, pause_poll_s: float = 0.05, **kwargs) -> InterruptibleTqdm:
    return InterruptibleTqdm(*args, control=control, pause_poll_s=pause_poll_s, **kwargs)
