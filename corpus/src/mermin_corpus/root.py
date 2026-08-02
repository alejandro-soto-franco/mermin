"""Corpus root resolution and the ASF-EX1 mount assertion.

The root is injected by MERMIN_CORPUS_ROOT so tests run against a temporary
directory with no drive attached. When the resolved root lies under /mnt, the
volume is asserted before any write, because /mnt/ASF-EX1 is an ordinary
directory on the root btrfs volume whenever the drive is absent, and writing
there would silently fill the system disk.
"""
from __future__ import annotations

import os
from pathlib import Path

from .errors import CorpusMountError

DEFAULT_ROOT = Path("/mnt/ASF-EX1/mermin-corpus")
EX1_MOUNT = Path("/mnt/ASF-EX1")
EX1_UUID = "32d726f6-68d8-495f-b40c-e364c66b8ce5"


def corpus_root() -> Path:
    return Path(os.environ.get("MERMIN_CORPUS_ROOT", str(DEFAULT_ROOT)))


def assert_mounted(path: Path) -> None:
    """Raise unless `path` is safe to write to.

    Paths outside /mnt carry no mount contract and pass unconditionally.
    """
    if not str(path).startswith("/mnt/"):
        return

    if not os.path.ismount(EX1_MOUNT):
        raise CorpusMountError(
            f"{EX1_MOUNT} is not mounted. Writing to {path} would land on the "
            f"root volume. Mount ASF-EX1, or set MERMIN_CORPUS_ROOT."
        )

    device = Path("/dev/disk/by-uuid") / EX1_UUID
    if not device.exists():
        raise CorpusMountError(f"No block device with UUID {EX1_UUID}")

    if os.stat(EX1_MOUNT).st_dev != os.stat(device).st_rdev:
        raise CorpusMountError(
            f"{EX1_MOUNT} is mounted from a device whose UUID is not {EX1_UUID}"
        )


def ensure(path: Path) -> Path:
    assert_mounted(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def entry_dir(partition: str, entry_id: str) -> Path:
    return corpus_root() / partition / entry_id


def lock_dir() -> Path:
    return corpus_root() / ".locks"
