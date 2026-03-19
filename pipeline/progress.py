from __future__ import annotations

import time
from typing import Any, Optional

from tqdm import tqdm

from pipeline.nodes.interrupts import StageStopRequested


def _progress_control_root(control: Any) -> Any:
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


class InterruptibleTqdm(tqdm):
    """tqdm wrapper that honors pipeline pause/stop controls."""

    def __init__(self, *args, control: Any = None, pause_poll_s: float = 0.05, **kwargs):
        self.progress_control = control
        self.pause_poll_s = max(0.01, float(pause_poll_s))
        super().__init__(*args, **kwargs)

    def _raise_stop(self) -> None:
        stop_fn = _control_callable(self.progress_control, "stop_requested")
        if stop_fn is None:
            return
        try:
            stop_requested = bool(stop_fn())
        except Exception:
            return
        if not stop_requested:
            return
        save_requested = False
        save_fn = _control_callable(self.progress_control, "shutdown_save")
        if save_fn is not None:
            try:
                save_requested = bool(save_fn())
            except Exception:
                save_requested = False
        raise StageStopRequested(
            "GUI stop+save requested" if save_requested else "GUI stop requested",
            save_requested=save_requested,
        )

    def check_control(self) -> None:
        pause_fn = _control_callable(self.progress_control, "paused")
        pump_fn = _control_pump(self.progress_control)
        while True:
            self._raise_stop()
            if pause_fn is None:
                return
            try:
                paused = bool(pause_fn())
            except Exception:
                return
            if not paused:
                return
            if pump_fn is not None:
                try:
                    pump_fn()
                except Exception:
                    pass
            time.sleep(self.pause_poll_s)

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
