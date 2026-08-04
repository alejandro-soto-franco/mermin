"""Hermetic golden: `mermin.analyze()` pinned against a phantom generated in
process, needing no ASF-EX1 mount and no `tomlkit`.

`corpus/goldens/*.json` (the eight real-corpus goldens `mermin_corpus.generate`
produces) can only be regenerated or checked with the corpus volume mounted,
and importing `mermin_corpus.generate` itself needs `tomlkit` (via
`.manifest`), a `mermin-corpus`-only dependency absent from CI's `floors` and
`python-tests` jobs and from the standard `mermin` test environment
(`tests/test_corpus_goldens.py`'s own module docstring documents this same
gap for the real goldens). So the determinism claim
`mermin-py/pyproject.toml`'s `scikit-image` floor comment makes -- that
`watershed`'s boundary tie-breaking is stable at and above 0.25.2 -- was
checked by nothing in either job: the golden suite there is both
`-m "not corpus"`-deselected and `tomlkit`-skipped.

This module closes that gap. `mermin_corpus.phantoms` imports only numpy and
`mermin_corpus.goldens` imports only the standard library (verified by
reading both files: neither imports `.manifest` or anything that does), so a
golden built from a phantom generated here in process from its own seed needs
no drive and no `tomlkit`, and can therefore run under `floors` and
`python-tests` both, exactly where the real corpus goldens cannot.

The seed comes from `corpus/manifest.seed.toml`, read with `tomllib`
(standard library since Python 3.11, distinct from `tomlkit`, which is the
dependency this module exists to avoid), the same source
`tests/test_phantom_closed_form.py` already reads a seed from -- not invented
here a second time.

`HERMETIC_ENTRIES` is deliberately a second table, not an extension of
`mermin_corpus.generate.GOLDEN_ENTRIES`: a phantom generated in process is not
a manifest entry, and `GOLDEN_ENTRIES` cannot hold one without importing
`.manifest`, the dependency this whole module exists to avoid.
"""
from __future__ import annotations

import json
import tomllib
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .errors import GenerateError
from .goldens import Difference, build_record, compare
from .phantoms import generate

#: entry id -> the phantom generator and `analyze()` keyword arguments beyond
#: `segmentation="threshold"`. Mirrors `mermin_corpus.generate.GOLDEN_ENTRIES`
#: in shape, deliberately, for the same reason every table like it in this
#: project does: measured, not invented, and read by both the generator and
#: the test that regenerates against it.
HERMETIC_ENTRIES: dict[str, dict[str, Any]] = {
    "phantom-uniform-hermetic": {"generator": "uniform_director", "kwargs": {}},
}

#: Same package list as `mermin_corpus.generate._ENVIRONMENT_PACKAGES`,
#: duplicated rather than imported: importing `.generate` pulls in
#: `.manifest` and therefore `tomlkit`, which is exactly what this module
#: must not require.
_ENVIRONMENT_PACKAGES = (
    "numpy",
    "scipy",
    "scikit-image",
    "polars",
    "bioio",
    "bioio-ome-tiff",
    "bioio-ome-zarr",
    "bioio-tifffile",
)

#: `corpus/manifest.seed.toml`, resolved relative to this file. Four parents
#: up from `corpus/src/mermin_corpus/hermetic.py` is the repository root.
_SEED_MANIFEST = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "corpus"
    / "manifest.seed.toml"
)


def hermetic_goldens_dir() -> Path:
    """`tests/hermetic_goldens`, resolved relative to this file rather than
    the current working directory, mirroring `generate.goldens_dir`. Kept
    under `tests/`, not `corpus/goldens/`: the latter is the real, drive-
    backed goldens' directory, and `tests/test_corpus_goldens.py` globs it
    directly (`ENTRY_IDS = sorted(p.stem for p in _GOLDENS_DIR.glob("*.json"))`)
    to build its own entry list, so a file placed there would be swept into
    that suite as if it were a ninth manifest entry, with no manifest entry
    to back it.
    """
    return Path(__file__).resolve().parent.parent.parent.parent / "tests" / "hermetic_goldens"


def hermetic_golden_path(entry_id: str) -> Path:
    return hermetic_goldens_dir() / f"{entry_id}.json"


def _seed_for(generator: str) -> int:
    with _SEED_MANIFEST.open("rb") as f:
        manifest = tomllib.load(f)
    entry = next(
        (e for e in manifest["entry"] if e["source"].get("generator") == generator),
        None,
    )
    if entry is None:
        raise GenerateError(
            f"no {_SEED_MANIFEST} entry has source.generator == {generator!r}"
        )
    return int(entry["source"]["seed"])


def _environment() -> dict[str, str]:
    """Installed versions, read through `importlib.metadata`, mirroring
    `mermin_corpus.generate._environment` (duplicated, not imported, for the
    reason `_ENVIRONMENT_PACKAGES` above states)."""
    import mermin

    env = {"mermin": mermin.__version__}
    for package in _ENVIRONMENT_PACKAGES:
        env[package] = version(package)
    return env


def build_hermetic_record(entry_id: str, *, tmp_dir: Path) -> dict[str, Any]:
    """Generate `entry_id`'s phantom in process, analyse it, and build its
    golden record with the same `build_record` the corpus goldens use.

    Writes only inside `tmp_dir`, which the caller owns (a pytest `tmp_path`
    fixture, or a throwaway directory at CLI regeneration time): this
    function itself touches no drive and needs no `tomlkit`.
    """
    if entry_id not in HERMETIC_ENTRIES:
        raise GenerateError(
            f"{entry_id!r} is not one of the hermetic entries: "
            f"{sorted(HERMETIC_ENTRIES)}"
        )
    import tifffile

    import mermin

    spec = HERMETIC_ENTRIES[entry_id]
    seed = _seed_for(spec["generator"])
    phantom = generate(spec["generator"], seed)

    # Same ImageJ-flavoured TIFF convention `mermin_corpus.fetch._fetch_generated`
    # writes for the real, drive-backed phantom entries, so this record is
    # built from the identical construction the corpus goldens are, only
    # in process rather than read off ASF-EX1.
    path = tmp_dir / f"{entry_id}.tif"
    tifffile.imwrite(
        path,
        phantom.image,
        imagej=True,
        metadata={"axes": phantom.axes, "unit": "micron"},
        resolution=(1.0 / phantom.pixel_size_um, 1.0 / phantom.pixel_size_um),
    )

    result = mermin.analyze(path, segmentation="threshold", **spec["kwargs"])

    invocation = {
        "segmentation": "threshold",
        "pixel_size_um": spec["kwargs"].get("pixel_size_um"),
        "pixel_size_um_assumed": False,
        "channels": spec["kwargs"].get("channels"),
        # Not part of the real corpus goldens' invocation shape: recorded here
        # so a reader can reproduce the exact phantom from the golden alone,
        # without cross-referencing `corpus/manifest.seed.toml` separately.
        "generator": spec["generator"],
        "seed": seed,
    }
    return build_record(
        result, entry=entry_id, invocation=invocation, environment=_environment()
    )


def generate_hermetic_entry(entry_id: str) -> Path:
    """Write `entry_id`'s hermetic golden record to
    `tests/hermetic_goldens/<entry_id>.json`.

    Same `json.dump(..., indent=2, sort_keys=True)` plus a trailing newline
    as `mermin_corpus.generate.generate_entry`, so a regeneration that
    changes nothing produces a byte-identical file.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        record = build_hermetic_record(entry_id, tmp_dir=Path(tmp))
    out = hermetic_golden_path(entry_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return out


def check_hermetic_entry(entry_id: str) -> list[Difference]:
    """Every way `entry_id`'s current in-process analysis differs from its
    committed hermetic golden. Regenerates in memory; nothing on disk is
    written or read back beyond the golden file itself."""
    import tempfile

    out = hermetic_golden_path(entry_id)
    if not out.exists():
        raise GenerateError(
            f"{entry_id}: no hermetic golden at {out}; run without --check first"
        )
    golden = json.loads(out.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        current = build_hermetic_record(entry_id, tmp_dir=Path(tmp))
    return compare(golden, current)
