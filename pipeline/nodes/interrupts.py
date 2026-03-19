"""Lightweight stage-interrupt exceptions with no pipeline dependencies."""


class StageStopRequested(Exception):
    """Raised by long-running stage work when the GUI requested stop/stop+save."""

    def __init__(self, message: str = "GUI stop requested", *, save_requested: bool = False):
        super().__init__(message)
        self.save_requested = bool(save_requested)


class StageSkipForward(Exception):
    """Raised by a training loop when the GUI requests skipping to the next stage."""


class StageSkipBack(Exception):
    """Raised by a training loop when the GUI requests returning to the previous stage."""
