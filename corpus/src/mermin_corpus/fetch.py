"""Fetch dispatch across the corpus source kinds.

Raw bytes are always retained. The only destructive action here is overwriting
an entry's own raw directory under `force`, and a recorded hash that disagrees
with what regenerates or re-downloads is an error rather than an update.
"""
from __future__ import annotations

import datetime as _dt
import shutil
from pathlib import Path

import numpy as np
import tifffile

from .errors import FetchError
from .hashing import sha256_of, size_of
from .lock import entry_lock
from .manifest import Entry, Manifest, validate_entry
from .phantoms import generate
from .root import ensure, entry_dir


def raw_path(entry: Entry) -> Path:
    if entry.source.get("kind") == "local":
        return Path(entry.source["path"])
    return entry_dir(entry.partition, entry.id) / "raw"


def _today() -> str:
    return _dt.datetime.now(_dt.timezone.utc).date().isoformat()


def _record(manifest: Manifest, entry: Entry, target: Path) -> None:
    digest = sha256_of(target)
    recorded = entry.provenance.get("raw_sha256") or ""
    if recorded and recorded != digest:
        raise FetchError(
            f"{entry.id}: sha256 {digest} disagrees with the recorded {recorded}. "
            f"The upstream artefact changed, or the local copy is damaged."
        )
    manifest.update_provenance(
        entry.id,
        raw_sha256=digest,
        raw_bytes=size_of(target),
        access_date=_today(),
        status="fetched",
    )
    manifest.save()


def _fetch_generated(entry: Entry, dest: Path) -> Path:
    result = generate(entry.source["generator"], int(entry.source["seed"]))
    ensure(dest)
    out = dest / f"{entry.id}.tif"
    tifffile.imwrite(
        out,
        result.image,
        imagej=True,
        metadata={"axes": result.axes},
        resolution=(1.0 / result.pixel_size_um, 1.0 / result.pixel_size_um),
    )
    return out


def _fetch_local(entry: Entry) -> Path:
    src = Path(entry.source["path"])
    if not src.exists():
        raise FetchError(f"{entry.id}: local path {src} does not exist")
    return src


def fetch_entry(manifest: Manifest, entry_id: str, force: bool = False) -> Path:
    entry = manifest.get(entry_id)
    validate_entry(entry)
    kind = entry.source["kind"]

    with entry_lock(entry_id):
        if kind == "local":
            target = _fetch_local(entry)
            _record(manifest, entry, target)
            return target

        dest = raw_path(entry)
        already = dest.exists() and any(dest.iterdir())
        if already and not force:
            return next(iter(sorted(dest.iterdir())))

        if already and force:
            shutil.rmtree(dest)

        if kind == "generated":
            out = _fetch_generated(entry, dest)
        else:
            raise FetchError(f"{entry.id}: source kind {kind} is not implemented yet")

        _record(manifest, entry, dest)
        return out
