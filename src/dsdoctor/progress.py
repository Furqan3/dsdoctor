"""Progress reporting, for the runs that are long enough to need it.

At 600 images every check finishes before you look up. At 100,000 the image
integrity scan and the card's fingerprint each read every byte on disk, and a
tool that prints nothing for four minutes is indistinguishable from one that
has hung - so people kill it, and an audit nobody waits for catches nothing.

Two rules keep this from becoming noise:

  * nothing is drawn unless stderr is a terminal. Piping `--format json` into
    another program, or running in CI, must produce exactly the bytes that
    program expects and no control codes.
  * progress goes to stderr, never stdout. stdout carries the report.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager


def _interactive() -> bool:
    try:
        return sys.stderr.isatty()
    except Exception:            # pragma: no cover - exotic stream objects
        return False


@contextmanager
def spinner(label: str, enabled: bool = True):
    """A live status line for one long step, or nothing at all."""
    if not (enabled and _interactive()):
        yield lambda _msg: None
        return
    try:
        from rich.console import Console
    except ImportError:          # pragma: no cover - rich is a hard dependency
        yield lambda _msg: None
        return

    console = Console(stderr=True)
    with console.status(f"[bold]{label}", spinner="dots") as status:
        yield lambda msg: status.update(f"[bold]{label}[/bold] {msg}")


def track(iterable, label: str, total: int | None = None,
          enabled: bool = True, min_items: int = 2000):
    """Wrap an iterable in a progress bar when it is big enough to matter.

    Below `min_items` the bar costs more attention than the wait does, so the
    iterable is returned untouched.
    """
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None
    if not (enabled and _interactive()) or (total or 0) < min_items:
        return iterable
    try:
        from tqdm import tqdm
    except ImportError:          # pragma: no cover - tqdm is a hard dependency
        return iterable
    return tqdm(iterable, total=total, desc=label, unit="file",
                file=sys.stderr, leave=False)
