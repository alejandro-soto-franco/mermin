"""TOML manifest for the corpus, loaded and written through tomlkit.

tomlkit is used rather than tomllib plus a writer because the manifest is
edited in place by fetch and probe, and its comments and key order are part of
the document.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomlkit

from .errors import ManifestError
from .root import corpus_root

PARTITIONS = {"commercial-safe", "eval-only", "private"}
ACCESS = {"OPEN", "REG", "PRIVATE"}
KINDS = {"generated", "local", "http", "zip-index", "ome-zarr"}
STATUSES = {"pending", "fetched", "probed", "golden"}


@dataclass
class Entry:
    id: str
    rung: int
    partition: str
    access: str
    licence: str
    attribution: str
    doi: str
    retain_raw: bool
    source: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)
    roles: dict[str, Any] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


def validate_entry(entry: Entry) -> None:
    if entry.partition not in PARTITIONS:
        raise ManifestError(f"{entry.id}: partition {entry.partition!r} is not one of {sorted(PARTITIONS)}")
    if entry.access not in ACCESS:
        raise ManifestError(f"{entry.id}: access {entry.access!r} is not one of {sorted(ACCESS)}")
    if entry.access == "PRIVATE" and entry.partition != "private":
        raise ManifestError(f"{entry.id}: PRIVATE access requires the private partition")
    if entry.partition == "private" and entry.access != "PRIVATE":
        raise ManifestError(f"{entry.id}: the private partition requires PRIVATE access")
    if not entry.retain_raw:
        raise ManifestError(f"{entry.id}: retain_raw must be true; this corpus never purges raw")

    kind = entry.source.get("kind")
    if kind not in KINDS:
        raise ManifestError(f"{entry.id}: source kind {kind!r} is not one of {sorted(KINDS)}")
    if kind in {"http", "zip-index", "ome-zarr"} and not entry.source.get("url"):
        raise ManifestError(f"{entry.id}: source kind {kind} requires a url")
    if kind == "local" and not entry.source.get("path"):
        raise ManifestError(f"{entry.id}: source kind local requires a path")
    if kind == "generated" and not entry.source.get("generator"):
        raise ManifestError(f"{entry.id}: source kind generated requires a generator")

    status = entry.provenance.get("status")
    if status not in STATUSES:
        raise ManifestError(f"{entry.id}: status {status!r} is not one of {sorted(STATUSES)}")


class Manifest:
    def __init__(self, path: Path, doc: tomlkit.TOMLDocument):
        self.path = path
        self._doc = doc
        self.schema = int(doc.get("schema", 0))
        self.entries = [self._entry(t) for t in doc.get("entry", [])]

    @staticmethod
    def _entry(table: Any) -> Entry:
        return Entry(
            id=str(table["id"]),
            rung=int(table["rung"]),
            partition=str(table["partition"]),
            access=str(table["access"]),
            licence=str(table["licence"]),
            attribution=str(table.get("attribution", "")),
            doi=str(table.get("doi", "")),
            retain_raw=bool(table.get("retain_raw", True)),
            source=dict(table.get("source", {})),
            expected=dict(table.get("expected", {})),
            roles=dict(table.get("roles", {})),
            policy=dict(table.get("policy", {})),
            provenance=dict(table.get("provenance", {})),
        )

    def get(self, entry_id: str) -> Entry:
        for e in self.entries:
            if e.id == entry_id:
                return e
        raise ManifestError(f"unknown entry {entry_id!r}")

    def _table(self, entry_id: str) -> Any:
        for t in self._doc["entry"]:
            if str(t["id"]) == entry_id:
                return t
        raise ManifestError(f"unknown entry {entry_id!r}")

    def update_provenance(self, entry_id: str, **fields: Any) -> None:
        table = self._table(entry_id)["provenance"]
        for key, value in fields.items():
            table[key] = value
        self.get(entry_id).provenance.update(fields)

    def set_expected(self, entry_id: str, expected: dict[str, Any]) -> None:
        table = self._table(entry_id)["expected"]
        for key, value in expected.items():
            table[key] = value
        self.get(entry_id).expected.update(expected)

    def save(self) -> None:
        self.path.write_text(tomlkit.dumps(self._doc))


def manifest_path() -> Path:
    return corpus_root() / "manifest.toml"


def load(path: Path | None = None) -> Manifest:
    p = path or manifest_path()
    if not p.exists():
        raise ManifestError(f"no manifest at {p}")
    return Manifest(p, tomlkit.parse(p.read_text()))
