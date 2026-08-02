"""Per-entry flock, mirroring the katic pipeline's .locks idiom.

Two runs must not collide on one entry's directory.
"""
from __future__ import annotations

import fcntl
from contextlib import contextmanager
from pathlib import Path

from .root import ensure, lock_dir


@contextmanager
def entry_lock(entry_id: str, blocking: bool = True):
    ensure(lock_dir())
    path: Path = lock_dir() / f"{entry_id}.lock"
    flags = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
    with path.open("w") as fh:
        fcntl.flock(fh, flags)
        try:
            yield path
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
