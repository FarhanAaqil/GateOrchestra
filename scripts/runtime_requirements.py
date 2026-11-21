"""Print the runtime dependencies from pyproject.toml, one per line (for pip-audit in CI)."""

from __future__ import annotations

from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    print("\n".join(project["dependencies"]))
