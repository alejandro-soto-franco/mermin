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
    """Digest of a directory tree, folding in each file's relative path.

    The path is length-prefixed and the content contributes a fixed-width
    per-file digest, so no arrangement of names and bytes can produce the
    same stream as a different tree.
    """
    h = hashlib.sha256()
    for f in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = str(f.relative_to(path)).encode()
        h.update(len(rel).to_bytes(8, "big"))
        h.update(rel)
        h.update(bytes.fromhex(sha256_file(f)))
    return h.hexdigest()


def sha256_of(path: Path) -> str:
    return sha256_tree(path) if path.is_dir() else sha256_file(path)


def size_of(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
