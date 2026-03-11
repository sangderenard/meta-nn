"""Update the generated Mermaid flowcharts in README.md."""
from __future__ import annotations

from pipeline.graph_layers import main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(["--write-readme"]))
