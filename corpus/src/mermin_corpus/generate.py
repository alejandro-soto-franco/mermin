"""Golden-record generation over the corpus's reachable entries.

Only the entries that complete `mermin.analyze()` under
`segmentation="threshold"`, the deterministic backend, are golden-able: a
Cellpose result is not reproducible enough to pin, and Cellpose must not be
installed in this environment regardless. Eight of the corpus's thirteen
manifest entries reach `analyze()` today; the module-level `GOLDEN_ENTRIES`
table is the complete, deliberate list, measured by the phase 4 corpus
survey, not everything the manifest happens to contain. An entry that starts
failing, or a new manifest entry, does not silently drop out of or into the
golden set just because it exists in the manifest.

Two of `idr0021-1884807` and `idr0062-6001240`'s `pixel_size_um` arguments
are stated assumptions, not measurements: both OME-Zarr stores carry no
calibration a reader can parse, so `1.0` is a probe value rather than the
file's own scale. `idr0047-4496763`'s `pixel_size_um=1.0` is different: that
one is the manifest's own authoritative reading. `GOLDEN_ENTRIES` records
which is which, and `build_entry_record` writes it into the golden's
`invocation` block precisely so a reader of the golden never has to guess.

**Goldens are Python JSON, not portable JSON.** `correlations.correlation_length`
is legitimately infinite when a field has fewer than three cells to
correlate (`mermin.pipeline.analyze`'s early-return branch), and
`json.dump`'s default `allow_nan=True` writes that as the bare token
`Infinity`. That token round-trips through `json.load` back to
`float("inf")` in Python, which is all a golden ever needs, but `Infinity`
is not valid JSON for any non-Python reader. This is a deliberate choice,
not an oversight: a sentinel string would need its own decode path on read,
and that decode path would still have to reject a NaN smuggled in under the
same sentinel, for no benefit over the token this module already emits and
`mermin_corpus.goldens.compare` already understands (`_close` treats
`inf == inf` as equal and any NaN as a difference, never as equal to
itself). If this ever needs to travel to a non-Python consumer, translate at
the boundary; the goldens on disk stay exactly what `json.dump` writes.
"""
from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .errors import GenerateError
from .goldens import Difference, build_record, compare
from .manifest import Manifest
from .root import entry_dir

#: entry id -> keyword arguments passed to `mermin.analyze()` beyond
#: `segmentation="threshold"`, and whether `pixel_size_um` is a stated
#: assumption rather than a measurement. Values measured by the phase 4
#: corpus survey (Q1/Q2); do not re-derive these by hand, they encode which
#: reader-metadata gaps required a probe value and which channel mapping
#: the manifest's own roles predict.
GOLDEN_ENTRIES: dict[str, dict[str, Any]] = {
    "phantom-uniform": {"kwargs": {}, "pixel_size_um_assumed": False},
    "phantom-defect-pair": {"kwargs": {}, "pixel_size_um_assumed": False},
    "phantom-radial": {"kwargs": {}, "pixel_size_um_assumed": False},
    "phantom-hexatic": {"kwargs": {}, "pixel_size_um_assumed": False},
    "montano-hvf-2026-04": {"kwargs": {}, "pixel_size_um_assumed": False},
    "idr0021-1884807": {
        "kwargs": {"pixel_size_um": 1.0},
        "pixel_size_um_assumed": True,
    },
    "idr0062-6001240": {
        "kwargs": {"pixel_size_um": 1.0},
        "pixel_size_um_assumed": True,
    },
    "idr0047-4496763": {
        "kwargs": {"channels": {"nuclear": 2, "fibre": 0}, "pixel_size_um": 1.0},
        "pixel_size_um_assumed": False,
    },
}

_ENVIRONMENT_PACKAGES = ("numpy", "scipy", "scikit-image", "polars", "bioio")


def goldens_dir() -> Path:
    """`corpus/goldens`, resolved relative to this file rather than the
    current working directory, so the generator writes to the same place
    whether it is run from the repository root or from inside `corpus/`."""
    return Path(__file__).resolve().parent.parent.parent / "goldens"


def golden_path(entry_id: str) -> Path:
    return goldens_dir() / f"{entry_id}.json"


def _environment() -> dict[str, str]:
    """Installed versions, read through `importlib.metadata` rather than
    hardcoded, so a dependency bump shows up as an `environment` difference
    the next time `--check` runs, instead of silently going unrecorded.

    `mermin` is not a distribution on this worktree's `PYTHONPATH` (no
    wheel built, no `.dist-info`), so `importlib.metadata.version("mermin")`
    would raise; `mermin.__version__` already carries the correct fallback
    for that case (`mermin/__init__.py`), so it is used directly rather than
    reimplemented here.
    """
    import mermin

    env = {"mermin": mermin.__version__}
    for package in _ENVIRONMENT_PACKAGES:
        env[package] = version(package)
    return env


def _artefact_path(entry: Any) -> Path:
    """The artefact a probed entry resolves to.

    This reads `probe.py`'s own output (`meta.json`'s `artefacts[0]`, the
    same file it calls `summary`) rather than re-walking the corpus
    directory tree itself: `probe.probe_entry` already decided, for every
    entry regardless of source kind, which candidate is canonical. Re-deriving
    that choice here would be a second resolver liable to disagree with the
    first.
    """
    meta_path = entry_dir(entry.partition, entry.id) / "meta.json"
    if not meta_path.exists():
        raise GenerateError(
            f"{entry.id}: no meta.json at {meta_path}; probe this entry first"
        )
    meta = json.loads(meta_path.read_text())
    artefacts = meta.get("artefacts") or []
    if not artefacts:
        raise GenerateError(f"{entry.id}: meta.json at {meta_path} lists no artefacts")
    return Path(artefacts[0]["path"])


def build_entry_record(manifest: Manifest, entry_id: str) -> dict[str, Any]:
    """Analyse `entry_id` and build its golden record. Writes nothing."""
    if entry_id not in GOLDEN_ENTRIES:
        raise GenerateError(
            f"{entry_id!r} is not one of the goldenable entries: "
            f"{sorted(GOLDEN_ENTRIES)}"
        )
    import mermin

    spec = GOLDEN_ENTRIES[entry_id]
    entry = manifest.get(entry_id)
    path = _artefact_path(entry)

    result = mermin.analyze(path, segmentation="threshold", **spec["kwargs"])

    invocation = {
        "segmentation": "threshold",
        "pixel_size_um": spec["kwargs"].get("pixel_size_um"),
        "pixel_size_um_assumed": spec["pixel_size_um_assumed"],
        "channels": spec["kwargs"].get("channels"),
    }
    return build_record(
        result, entry=entry_id, invocation=invocation, environment=_environment()
    )


def generate_entry(manifest: Manifest, entry_id: str) -> Path:
    """Write `entry_id`'s golden record to `corpus/goldens/<entry_id>.json`.

    `json.dump(..., indent=2, sort_keys=True)` plus a trailing newline, so a
    regeneration that changes nothing produces a byte-identical file and a
    real change produces a minimal diff.
    """
    record = build_entry_record(manifest, entry_id)
    out = golden_path(entry_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return out


def check_entry(manifest: Manifest, entry_id: str) -> list[Difference]:
    """Every way `entry_id`'s current analysis differs from its committed
    golden. Regenerates in memory; nothing on disk is written or read back
    beyond the golden file itself."""
    out = golden_path(entry_id)
    if not out.exists():
        raise GenerateError(f"{entry_id}: no golden at {out}; run without --check first")
    golden = json.loads(out.read_text())
    current = build_entry_record(manifest, entry_id)
    return compare(golden, current)
