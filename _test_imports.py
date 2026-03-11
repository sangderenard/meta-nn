"""Quick smoke test: import every pipeline module and count symbols."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

modules = [
    "pipeline.utils",
    "pipeline.wave_io",
    "pipeline.nodes.base",
    "pipeline.nodes.vocab_node",
    "pipeline.nodes.data_nodes",
    "pipeline.nodes.berkeley_classifier_node",
    "pipeline.nodes.generator_node",
    "pipeline.nodes.transformer_node",
    "pipeline.nodes.wave_classifier_node",
    "pipeline.nodes.gate_nodes",
    "pipeline.orchestrator",
]

import importlib

ok = 0
fail = 0
for name in modules:
    try:
        mod = importlib.import_module(name)
        n = len([x for x in dir(mod) if not x.startswith("__")])
        print(f"  OK  {name} ({n} symbols)")
        ok += 1
    except Exception as exc:
        print(f"  FAIL {name}: {exc}")
        fail += 1

print(f"\n{ok} imported, {fail} failed")
if fail:
    sys.exit(1)
print("All pipeline modules imported successfully!")
