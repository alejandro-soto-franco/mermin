"""Streaming sha256 over files and directory trees.

A directory hash folds in each relative path as well as its contents, so a
rename changes the digest. OME-Zarr entries are directories, which is why the
tree form exists.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

CHUNK = 1024 * 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def sha256_tree(path: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(str(f.relative_to(path)).encode())
        h.update(b"\0")
        with f.open("rb") as fh:
            for block in iter(lambda: fh.read(CHUNK), b""):
                h.update(block)
    return h.hexdigest()


def sha256_of(path: Path) -> str:
    return sha256_tree(path) if path.is_dir() else sha256_file(path)


def size_of(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
