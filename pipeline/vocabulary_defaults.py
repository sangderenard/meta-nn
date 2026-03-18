# pipeline/vocabulary_defaults.py
#
# Loads the program-default vocabulary configuration from
# pipeline/config/default_vocabulary.json.
#
# Nothing here should be hardcoded.  If you need to change the default
# vocabulary or noise physics, edit the JSON file.

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List

_CONFIG_PATH = Path(__file__).parent / "config" / "default_vocabulary.json"

def _load() -> dict:
    return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))

_config = _load()

DEFAULT_VOCABULARY: List[str] = [str(t).strip() for t in _config["vocabulary"] if str(t).strip()]
NOISE_SPECTRUM_BETA: Dict[str, float] = {
    str(k): float(v)
    for k, v in _config["noise_spectrum_beta"].items()
    if not str(k).startswith("_")
}
