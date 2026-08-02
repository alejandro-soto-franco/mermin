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

from .manifest import Manifest
from .root import ensure, entry_dir

_EMISSION = re.compile(r'id="wavelength"[^>]*value="([\d.]+)"')
_MICRON_UNITS = {"micron", "microns", "um", "µm"}
_TIFF_SUFFIXES = {".tif", ".tiff"}


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
    """
    try:
        import tifffile
    except ImportError:
        return None
    try:
        with tifffile.TiffFile(path) as tif:
            if not tif.is_imagej:
                return None
            meta = tif.imagej_metadata or {}
            unit = str(meta.get("unit") or "").lower()
            if unit not in _MICRON_UNITS:
                return None
            page = tif.pages[0]
            if "XResolution" not in page.tags:
                return None
            num, den = page.tags["XResolution"].value
            if not num:
                return None
            return float(den) / float(num)
    except Exception:
        return None


def _pixel_size_um(path: Path, image: BioImage) -> float | None:
    if path.suffix.lower() in _TIFF_SUFFIXES:
        return _pixel_size_um_from_tiff(path)
    sizes = image.physical_pixel_sizes
    for value in (sizes.X, sizes.Y):
        if value:
            return float(value)
    return None


def _emission_nm(path: Path) -> list[float]:
    """MetaMorph emission wavelengths, one per plane, from the ImageJ Labels.

    The Montano batch stores its channel identity here and nowhere else, and its
    plane order is not constant across the batch.
    """
    try:
        import tifffile
    except ImportError:
        return []
    try:
        with tifffile.TiffFile(path) as tif:
            labels = (tif.imagej_metadata or {}).get("Labels") or []
            return [float(m.group(1)) for label in labels
                    if (m := _EMISSION.search(str(label)))]
    except Exception:
        return []


def probe_file(path: Path) -> dict[str, Any]:
    image = BioImage(path)
    axes = "".join(image.dims.order)
    shape = [int(n) for n in image.dims.shape]
    squeezed = [(a, n) for a, n in zip(axes, shape) if n > 1 or a in "CYX"]
    info: dict[str, Any] = {
        "path": str(path),
        "axes": "".join(a for a, _ in squeezed),
        "shape": [n for _, n in squeezed],
        "dtype": str(image.dtype),
        "size_c": int(image.dims.C),
        "pixel_size_um": _pixel_size_um(path, image),
        "channel_names": list(image.channel_names or []),
        "emission_nm": _emission_nm(path) if path.suffix.lower() in _TIFF_SUFFIXES else [],
        "reader": type(image.reader).__module__.split(".")[0],
    }
    return info


def probe_entry(manifest: Manifest, entry_id: str) -> dict[str, Any]:
    from .fetch import raw_path

    entry = manifest.get(entry_id)
    target = raw_path(entry)
    candidates = sorted(p for p in ([target] if target.is_file() else target.iterdir()))
    if not candidates:
        raise FileNotFoundError(f"{entry_id}: nothing fetched at {target}")

    info = probe_file(candidates[0])
    info["n_artefacts"] = len(candidates)

    out_dir = ensure(entry_dir(entry.partition, entry.id))
    (out_dir / "meta.json").write_text(json.dumps(info, indent=2, sort_keys=True))

    manifest.set_expected(entry_id, {
        "axes": info["axes"],
        "shape": info["shape"],
        "dtype": info["dtype"],
        "size_c": info["size_c"],
        "pixel_size_um": info["pixel_size_um"] or 0.0,
        "reader": info["reader"],
    })
    manifest.update_provenance(entry_id, status="probed")
    manifest.save()
    return info
