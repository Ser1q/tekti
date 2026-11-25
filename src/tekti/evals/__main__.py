"""CLI entry point for ``python -m tekti.evals``."""

from __future__ import annotations

from tekti.evals.score import main

if __name__ == "__main__":
    raise SystemExit(main())
