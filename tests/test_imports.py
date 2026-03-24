"""Wraps the script-style _test_imports smoke test for pytest."""
from __future__ import annotations

import importlib


MODULES = [
    "pipeline.utils",
    "pipeline.wave_io",
    "pipeline.nodes.base",
    "pipeline.nodes.vocab_node",
    "pipeline.nodes.data_nodes",
    "pipeline.nodes.classifier_node",
    "pipeline.nodes.generator_node",
    "pipeline.nodes.transformer_node",
    "pipeline.nodes.wave_classifier_node",
    "pipeline.nodes.gate_nodes",
    "pipeline.orchestrator",
]


def test_all_pipeline_modules_import():
    failures = []
    for name in MODULES:
        try:
            mod = importlib.import_module(name)
            n = len([x for x in dir(mod) if not x.startswith("__")])
            assert n > 0, f"{name} exported 0 symbols"
        except Exception as exc:
            failures.append(f"{name}: {exc}")
    assert not failures, "Failed imports:\n" + "\n".join(failures)
