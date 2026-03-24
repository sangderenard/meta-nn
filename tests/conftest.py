"""Shared fixtures for the meta-nn test suite."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_work_dir(tmp_path: Path) -> Path:
    """Return a temporary working directory rooted inside pytest's tmp_path."""
    return tmp_path
