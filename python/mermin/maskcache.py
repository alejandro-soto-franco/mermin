"""Content-addressed storage for nuclear masks.

Segmentation is the one step in the pipeline that is expensive and, under
cellpose, not reproducible. Everything downstream of the nuclear mask is
deterministic given that mask, so caching the mask alone makes every number
after it repeatable.
"""

import hashlib
import json
import os
import tempfile
import warnings
from pathlib import Path

import numpy as np

from mermin.backends import SegmentationBackend

# Part of every key. Bumping it invalidates every stored mask, which is what to
# do when the stored representation or the key inputs change.
CACHE_FORMAT_VERSION = "1"


def _framed(digest: "hashlib._Hash", field: bytes) -> None:
    """Feed one length-prefixed field to the digest.

    The length prefix is what stops two different field sequences hashing the
    same. Concatenating unframed fields is a collision, not a theoretical one.
    """
    digest.update(len(field).to_bytes(8, "big"))
    digest.update(field)


def mask_cache_key(plane: np.ndarray, backend: SegmentationBackend) -> str:
    """The cache key for one plane segmented by one backend at one configuration."""
    plane = np.ascontiguousarray(plane, dtype=np.float64)
    digest = hashlib.sha256()
    _framed(digest, CACHE_FORMAT_VERSION.encode())
    _framed(digest, repr(plane.shape).encode())
    _framed(digest, plane.dtype.str.encode())
    _framed(digest, plane.tobytes())
    _framed(digest, backend.name.encode())
    _framed(digest, backend.version().encode())
    _framed(
        digest,
        json.dumps(backend.config(), sort_keys=True, separators=(",", ":")).encode(),
    )
    return digest.hexdigest()


def _entry_path(cache_dir: str | Path, key: str) -> Path:
    return Path(cache_dir) / f"{key}.npz"


def load_mask(cache_dir: str | Path, key: str) -> np.ndarray | None:
    """The stored mask, or None when there is no usable entry.

    An unreadable entry is a miss rather than an exception: a corrupt cache
    must not break an analysis whose inputs are fine.
    """
    path = _entry_path(cache_dir, key)
    if not path.is_file():
        return None
    try:
        with np.load(path) as stored:
            mask = stored["mask"]
    except Exception as error:
        warnings.warn(
            f"ignoring an unreadable mask cache entry at {path}: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    if mask.ndim != 2:
        warnings.warn(
            f"ignoring a mask cache entry at {path} holding a {mask.ndim}D array",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return mask.astype(np.int32, copy=False)


def store_mask(cache_dir: str | Path, key: str, mask: np.ndarray) -> None:
    """Write a mask into the cache, staging first and swapping into place.

    A killed run must never leave a half-written mask that a later run reads as
    a hit, so the entry appears atomically or not at all.
    """
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    handle, staged = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        # The file object matters: passing a path to savez_compressed appends
        # `.npz` to it, which would leave the staged file somewhere other than
        # where os.replace is told to look.
        with os.fdopen(handle, "wb") as file:
            np.savez_compressed(file, mask=np.asarray(mask, dtype=np.int32))
        os.replace(staged, _entry_path(directory, key))
    except BaseException:
        Path(staged).unlink(missing_ok=True)
        raise
