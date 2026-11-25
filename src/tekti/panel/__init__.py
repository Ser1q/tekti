"""Review panel: five angles, each run by several model families in parallel.

    review(doc, requirements, facts, cfg) -> list[RawFinding]

After a call, `runner.last_dropped` lists reviewers that failed ("lens/model: error")
and `runner.last_stats` holds validation counts. Use `dropped()` / `stats()` to read
them, since rebinding at module level would not show through a plain re-export.
"""

from __future__ import annotations

from tekti.panel import runner
from tekti.panel.prompts import system_prompt, user_prompt
from tekti.panel.runner import PanelConfig, PanelStats, review
from tekti.panel.schema import ReviewerOutput

# Same list object as runner.last_dropped (it is cleared in place, never rebound).
last_dropped = runner.last_dropped


def dropped() -> list[str]:
    """Reviewers dropped in the last `review` call."""
    return list(runner.last_dropped)


def stats() -> PanelStats:
    """Validation counts from the last `review` call."""
    return runner.last_stats


__all__ = [
    "PanelConfig",
    "PanelStats",
    "ReviewerOutput",
    "dropped",
    "last_dropped",
    "review",
    "stats",
    "system_prompt",
    "user_prompt",
]
