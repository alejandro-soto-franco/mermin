"""Reader-level inspection of a corpus artefact.

This module imports no mermin code. Its output describes what a reader sees,
which is what makes it evidence for the ingest design rather than a restatement
of mermin's own assumptions.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from bioio import BioImage

from .errors import ProbeError
from .manifest import Manifest
from .root import ensure, entry_dir

_EMISSION = re.compile(r'id="wavelength"[^>]*value="([\d.]+)"')
_CALIBRATION_X = re.compile(r'id="spatial-calibration-x"[^>]*value="([\d.]+)"')
_CALIBRATION_STATE = re.compile(r'id="spatial-calibration-state"[^>]*value="([^"]+)"')
_CALIBRATION_UNITS = re.compile(r'id="spatial-calibration-units"[^>]*value="([^"]+)"')
_MICRON_UNITS = {"micron", "microns", "um", "µm"}
_TIFF_SUFFIXES = {".tif", ".tiff"}


def _imagej_labels(path: Path) -> list[str]:
    """ImageJ `Labels` as a list, however tifffile round-tripped them.

    A single label comes back as a bare string rather than a list, and
    iterating that directly walks it character by character.
    """
    try:
        import tifffile

        with tifffile.TiffFile(path) as tif:
            labels = (tif.imagej_metadata or {}).get("Labels") or []
    except Exception:
        return []
    if isinstance(labels, str):
        labels = [labels]
    return [str(label) for label in labels]


def _metamorph_pixel_size_um(path: Path) -> float | None:
    """Pixel size from the MetaMorph spatial calibration, when it is enabled.

    Every plane of a MetaMorph stack carries the same calibration, so the
    first label that states one is enough.
    """
    for label in _imagej_labels(path):
        state = _CALIBRATION_STATE.search(label)
        if not state or state.group(1).lower() != "on":
            continue
        units = _CALIBRATION_UNITS.search(label)
        if not units or not units.group(1).lower().startswith("micron"):
            continue
        value = _CALIBRATION_X.search(label)
        if value and float(value.group(1)) > 0:
            return float(value.group(1))
    return None


def _pixel_size_um_from_tiff(path: Path) -> float | None:
    """Pixel size in microns read directly from the TIFF resolution tags.

    bioio-tifffile's own `physical_pixel_sizes` cannot be trusted here: its
    `_get_pixel_size` maps an unset `ResolutionUnit` (RESUNIT.NONE) to a
    microns scalar of 1 by convention, so an uncalibrated file with the
    default 1/1 resolution tag reads back as "1.0 micron per pixel" rather
    than "no calibration". It also keys `imagej_metadata["unit"]` directly, so
    an ImageJ file that never wrote a `unit` field (as this corpus's own
    phantom generator does) raises inside bioio and is swallowed into None
    even when a real resolution tag is present. Reading the tags directly and
    only trusting an explicit micron-family unit string avoids both.

    When no TIFF-tag calibration is present, this falls back to the MetaMorph
    spatial calibration embedded in the ImageJ Labels (see
    `_metamorph_pixel_size_um`), which is where the real Montano batch keeps
    its pixel size; the standard tags are unset there.
    """
    try:
        import tifffile
    except ImportError:
        return _metamorph_pixel_size_um(path)
    try:
        with tifffile.TiffFile(path) as tif:
            if not tif.is_imagej:
                return _metamorph_pixel_size_um(path)
            meta = tif.imagej_metadata or {}
            unit = str(meta.get("unit") or "").lower()
            if unit not in _MICRON_UNITS:
                return _metamorph_pixel_size_um(path)
            page = tif.pages[0]
            if "XResolution" not in page.tags:
                return _metamorph_pixel_size_um(path)
            num, den = page.tags["XResolution"].value
            if not num:
                return _metamorph_pixel_size_um(path)
            return float(den) / float(num)
    except Exception:
        return _metamorph_pixel_size_um(path)


def _pixel_size_um(path: Path, image: BioImage) -> float | None:
    if path.suffix.lower() in _TIFF_SUFFIXES:
        return _pixel_size_um_from_tiff(path)
    sizes = image.physical_pixel_sizes
    if sizes is None:
        # bioio-ome-zarr returns None outright, rather than a sizes object
        # with unset fields, when the store's transform carries no scale it
        # can parse (for example a bare `coordinateTransformations` list with
        # no "scale" entry). Uncalibrated is a legitimate reader outcome.
        return None
    for value in (sizes.X, sizes.Y):
        if value:
            return float(value)
    return None


def _emission_nm(path: Path) -> list[float | None]:
    """Per-plane emission wavelength from the MetaMorph XML in ImageJ Labels.

    One entry per label, `None` where a label carried no parseable wavelength,
    so a partial parse cannot be mistaken for a file with fewer planes. The
    Montano batch stores its channel identity here and nowhere else, and its
    plane order is not constant across the batch.
    """
    out: list[float | None] = []
    for label in _imagej_labels(path):
        match = _EMISSION.search(label)
        out.append(float(match.group(1)) if match else None)
    return out


def probe_file(path: Path) -> dict[str, Any]:
    """Reader-level inspection of one artefact.

    Every bioio call below can fail lazily rather than at `BioImage(path)`:
    dims, dtype and channel access all trigger the reader's own metadata
    parsing on first touch, and a reader that cannot make sense of a
    particular store's dialect (an unsupported format, or a metadata shape
    it does not expect) raises whatever exception it likes, not a
    ProbeError. Only the BioImage construction and its attribute access are
    wrapped so such a failure is reported against this artefact rather than
    crashing the caller's batch; `_pixel_size_um` and `_emission_nm` are
    called outside the wrapper so a bug in either of those helpers surfaces
    as itself, rather than being mistaken for a missing reader plugin.
    """
    try:
        image = BioImage(path)
        axes = "".join(image.dims.order)
        shape = [int(n) for n in image.dims.shape]
        dtype = str(image.dtype)
        size_c = int(image.dims.C)
        channel_names = list(image.channel_names or [])
        reader = type(image.reader).__module__.split(".")[0]
    except Exception as exc:
        raise ProbeError(
            f"cannot read {path}: {exc}. Likely a missing bioio reader plugin, "
            f"or a reader that cannot parse this store's metadata dialect; see "
            f"https://github.com/bioio-devs/bioio for the plugin list."
        ) from exc

    squeezed = [(a, n) for a, n in zip(axes, shape) if n > 1 or a in "CYX"]
    return {
        "path": str(path),
        "axes": "".join(a for a, _ in squeezed),
        "shape": [n for _, n in squeezed],
        "dtype": dtype,
        "size_c": size_c,
        "pixel_size_um": _pixel_size_um(path, image),
        "channel_names": channel_names,
        "emission_nm": _emission_nm(path) if path.suffix.lower() in _TIFF_SUFFIXES else [],
        "reader": reader,
    }


_ZARR_STORE_MARKERS = (".zgroup", ".zarray")


def _artefact_candidates(target: Path) -> list[Path]:
    """Every artefact directly readable from `target`.

    A plain file is the one candidate. A directory is either an OME-Zarr
    store root, identified by carrying `.zgroup` or `.zarray` directly (in
    which case the whole directory is the one candidate, since a reader
    needs the store root rather than one of its chunk files), or a plain
    directory of files, possibly nested: a zip-index fetch preserves each
    member's archive subpath (Task 6), so a multi-field source like BBBC021
    puts its files one directory below `raw/`, inside a plate subdirectory
    rather than directly in it. Recursion stops at the first zarr store root
    found along a branch; it never descends into one.
    """
    if target.is_file():
        return [target]
    if any((target / marker).exists() for marker in _ZARR_STORE_MARKERS):
        return [target]
    found: list[Path] = []
    for child in sorted(target.iterdir()):
        if child.name.startswith("."):
            continue
        found.extend(_artefact_candidates(child))
    return found


_MAX_PROBED_ARTEFACTS = 200


def _emission_sort_key(seq: tuple) -> tuple:
    """A total order over `emission_nm` tuples that tolerates `None` entries.

    Plain tuple comparison raises when a `None` meets a `float`, which a
    partially parsed label can produce. Mapping `None` to a sentinel that
    sorts before every float keeps ordering deterministic without losing it.
    """
    return tuple((0, 0.0) if x is None else (1, x) for x in seq)


def probe_entry(manifest: Manifest, entry_id: str) -> dict[str, Any]:
    """Probe every artefact belonging to `entry_id`, up to a cap.

    A batch entry can register dozens of sibling files whose facts vary (a
    mixed channel count, or emission metadata present on some files and
    absent on others); probing only the first candidate, as an earlier
    version of this function did, silently recorded that one file's facts as
    though they held for the whole entry. Every candidate is probed instead,
    capped at `_MAX_PROBED_ARTEFACTS` since metadata reads are cheap but not
    free, and the manifest's `expected` block gains fields that describe the
    variation across the whole entry rather than asserting a false uniformity.
    """
    from .fetch import raw_path

    entry = manifest.get(entry_id)
    target = raw_path(entry)
    if not target.exists():
        raise ProbeError(
            f"{entry_id}: nothing fetched at {target}. Run fetch for this entry first."
        )
    candidates = sorted(_artefact_candidates(target))
    if not candidates:
        raise ProbeError(f"{entry_id}: nothing fetched at {target}")

    truncated = len(candidates) > _MAX_PROBED_ARTEFACTS
    artefacts = [probe_file(c) for c in candidates[:_MAX_PROBED_ARTEFACTS]]
    summary = artefacts[0]

    out_dir = ensure(entry_dir(entry.partition, entry.id))
    meta = {
        "summary": summary,
        "artefacts": artefacts,
        "n_artefacts": len(candidates),
        "truncated": truncated,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))

    axes_seen = {a["axes"] for a in artefacts}
    dtype_seen = {a["dtype"] for a in artefacts}
    shape_seen = sorted({tuple(a["shape"]) for a in artefacts})
    size_c_seen = sorted({a["size_c"] for a in artefacts})
    emission_seen = sorted(
        {tuple(a["emission_nm"]) for a in artefacts}, key=_emission_sort_key
    )
    uniform = (
        len(axes_seen) == 1
        and len(dtype_seen) == 1
        and len(shape_seen) == 1
        and len(size_c_seen) == 1
    )

    expected: dict[str, Any] = {
        "axes": summary["axes"],
        "shape": summary["shape"],
        "dtype": summary["dtype"],
        "size_c": summary["size_c"],
        "pixel_size_um": summary["pixel_size_um"],
        "reader": summary["reader"],
        "n_artefacts": len(candidates),
        "uniform": uniform,
        "size_c_observed": size_c_seen,
        "shape_observed": [list(s) for s in shape_seen],
    }
    if emission_seen != [()]:
        expected["emission_nm_observed"] = [list(e) for e in emission_seen]

    manifest.set_expected(entry_id, expected)
    manifest.update_provenance(entry_id, status="probed")
    manifest.save()

    result = dict(summary)
    result["n_artefacts"] = len(candidates)
    result["truncated"] = truncated
    return result
