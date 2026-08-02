"""Fetch dispatch across the corpus source kinds.

Raw bytes are always retained. The only destructive action here is overwriting
an entry's own raw directory under `force`, and a recorded hash that disagrees
with what regenerates or re-downloads is an error rather than an update.
"""
from __future__ import annotations

import datetime as _dt
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx
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


def _download(url: str, dest_file: Path) -> None:
    ensure(dest_file.parent)
    with httpx.Client(follow_redirects=True, timeout=120.0) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with dest_file.open("wb") as fh:
                for chunk in response.iter_bytes(chunk_size=1 << 16):
                    fh.write(chunk)


def _url_filename(url: str) -> str:
    name = Path(unquote(urlparse(url).path)).name
    if not name:
        raise FetchError(f"cannot derive a filename from {url}")
    return name


def _fetch_http(entry: Entry, dest: Path) -> Path:
    out = dest / _url_filename(entry.source["url"])
    _download(entry.source["url"], out)
    return out


def _fetch_zip_index(entry: Entry, dest: Path) -> Path:
    members = list(entry.source.get("members", []))
    if not members:
        raise FetchError(f"{entry.id}: zip-index requires a non-empty members list")

    ensure(dest)
    resolved_dest = dest.resolve()
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "archive.zip"
        _download(entry.source["url"], archive)
        with zipfile.ZipFile(archive) as zf:
            available = set(zf.namelist())
            missing = [m for m in members if m not in available]
            if missing:
                raise FetchError(f"{entry.id}: members absent from the archive: {missing}")
            for member in members:
                candidate = (resolved_dest / member).resolve()
                if not candidate.is_relative_to(resolved_dest):
                    raise FetchError(f"{entry.id}: member {member!r} escapes the destination")
                ensure(candidate.parent)
                with zf.open(member) as src, candidate.open("wb") as fh:
                    shutil.copyfileobj(src, fh)
    files = sorted(p for p in dest.rglob("*") if p.is_file())
    if not files:
        raise FetchError(f"{entry.id}: the archive yielded no files")
    return files[0]


_FETCHERS = {
    "generated": _fetch_generated,
    "http": _fetch_http,
    "zip-index": _fetch_zip_index,
}


def fetch_entry(manifest: Manifest, entry_id: str, force: bool = False) -> Path:
    entry = manifest.get(entry_id)
    validate_entry(entry)
    kind = entry.source["kind"]

    with entry_lock(entry_id):
        if kind == "local":
            target = _fetch_local(entry)
            _record(manifest, entry, target)
            return target

        fetcher = _FETCHERS.get(kind)
        if fetcher is None:
            raise FetchError(f"{entry.id}: source kind {kind} is not implemented yet")

        dest = raw_path(entry)
        if dest.exists() and any(dest.iterdir()) and not force:
            return next(iter(sorted(dest.iterdir())))

        # Fetch into staging first, so an existing artefact is only ever
        # replaced by a complete one. A failed fetch leaves the old bytes
        # untouched rather than deleting them and raising.
        staging = dest.parent / f".{dest.name}.incoming"
        if staging.exists():
            shutil.rmtree(staging)
        out = fetcher(entry, staging)
        relative = out.relative_to(staging)

        if dest.exists():
            shutil.rmtree(dest)
        ensure(dest.parent)
        staging.rename(dest)

        _record(manifest, entry, dest)
        return dest / relative
