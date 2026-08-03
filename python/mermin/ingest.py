"""Open a microscopy file, resolve channel roles, and return normalised planes.

This replaces two things that were wrong on real data: `io.py`'s hardcoded
`{"dapi": 0, "vimentin": 1}` channel order, and `pipeline.py`'s default
`pixel_size_um = 0.345`. Channel order is not constant within a batch, and
0.345 was a superseded estimate against a measured 0.6897, so every
length-derived quantity came out a factor of two low. `pixel_size_um` has no
default here: the resolution order is the explicit argument, then file
metadata, and otherwise a `PixelSizeError` naming the file, never a silent
guess.

The pixel-size and emission logic below reproduces what phase 1's corpus
prober (`corpus/src/mermin_corpus/probe.py`) proved against ten real files.
It is not imported from there: `corpus/**` is excluded from the maturin
wheel build in `mermin-py/pyproject.toml`, so it never ships, but the two
lessons it earned carry over unchanged:

- `bioio-tifffile`'s own `physical_pixel_sizes` cannot be trusted for TIFFs.
  Its reader maps an unset `ResolutionUnit` to a microns scalar of 1 by
  convention, so an uncalibrated file with the default 1/1 resolution tag
  reads back as "1.0 micron per pixel" rather than "no calibration". Reading
  the TIFF resolution tags directly, and only trusting an explicit micron
  unit string, avoids that.
- `tifffile` round-trips a single ImageJ `Labels` entry as a bare string
  rather than a list. Iterating that directly walks it character by
  character and silently finds nothing.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from bioio import BioImage

from mermin.roles import RoleResolution, resolve_roles

_EMISSION = re.compile(r'id="wavelength"[^>]*value="([\d.]+)"')
_CALIBRATION_X = re.compile(r'id="spatial-calibration-x"[^>]*value="([\d.]+)"')
_CALIBRATION_STATE = re.compile(r'id="spatial-calibration-state"[^>]*value="([^"]+)"')
_CALIBRATION_UNITS = re.compile(r'id="spatial-calibration-units"[^>]*value="([^"]+)"')
_MICRON_UNITS = {"micron", "microns", "um", "µm"}
_TIFF_SUFFIXES = {".tif", ".tiff"}

_LOWER_PERCENTILE = 1.0
_UPPER_PERCENTILE = 99.5

_SUPPORTED_PROJECTIONS = {"single", "max", "mean"}


class PixelSizeError(Exception):
    """No pixel size was supplied and none could be read from the file."""


@dataclass(frozen=True)
class LoadedImage:
    planes: dict[str, np.ndarray]
    pixel_size_um: float
    roles: dict[str, RoleResolution]
    projection: str
    axes: str
    dtype: str
    channel_names: list[str]
    emission_nm: list[float | None]


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

    Falls back to the MetaMorph spatial calibration embedded in the ImageJ
    `Labels` when the standard tags carry no explicit micron unit.
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


def _pixel_size_um_from_bioio(image: BioImage) -> float | None:
    """Pixel size for non-TIFF formats, via bioio's own metadata.

    Not used for TIFFs: `bioio-tifffile` maps an unset `ResolutionUnit` to a
    microns scalar of 1, so it can never report an uncalibrated TIFF.
    """
    sizes = image.physical_pixel_sizes
    if sizes is None:
        return None
    for value in (sizes.X, sizes.Y):
        if value:
            return float(value)
    return None


def _resolve_pixel_size_um(
    path: Path, image: BioImage, explicit: float | None
) -> float:
    if explicit is not None:
        return float(explicit)
    if path.suffix.lower() in _TIFF_SUFFIXES:
        value = _pixel_size_um_from_tiff(path)
    else:
        value = _pixel_size_um_from_bioio(image)
    if value is None:
        raise PixelSizeError(
            f"{path} carries no pixel size in its metadata; pass one "
            f"explicitly, for example open_image(path, pixel_size_um=0.69)."
        )
    return value


def _emission_nm(path: Path, n_channels: int) -> list[float | None]:
    """Per-channel emission wavelength from the MetaMorph XML in ImageJ Labels.

    One entry per label, `None` where a label carried no parseable
    wavelength, so a partial parse cannot be mistaken for a file with fewer
    channels. The labels are written one per plane in acquisition order; that
    coincides with channel order only when the file carries a single Z and a
    single timepoint, which is what a MetaMorph channel stack is. When the
    label count does not match the channel count, the emission cannot be
    trusted to line up with channel index, so it is dropped rather than
    misaligned, and role resolution falls through to name matching or the
    position fallback instead; a `UserWarning` names both counts so the drop
    is visible rather than silent.
    """
    if path.suffix.lower() not in _TIFF_SUFFIXES:
        return [None] * n_channels
    out: list[float | None] = []
    for label in _imagej_labels(path):
        match = _EMISSION.search(label)
        out.append(float(match.group(1)) if match else None)
    if len(out) != n_channels:
        if out:
            warnings.warn(
                f"{len(out)} ImageJ labels for {n_channels} channels, so "
                f"per-channel emission cannot be read; falling back to "
                f"channel names",
                UserWarning,
                stacklevel=3,
            )
        return [None] * n_channels
    return out


def _squeezed_axes(image: BioImage) -> str:
    """The source file's own axis letters, dropping size-1 dims other than C/Y/X.

    bioio always reports a full `TCZYX` order regardless of what the file
    actually carries; this reflects back only the axes the source data has,
    which is what the corpus's measured `CYX` and `CZYX` facts describe.
    """
    order = "".join(image.dims.order)
    shape = [int(n) for n in image.dims.shape]
    return "".join(a for a, n in zip(order, shape) if n > 1 or a in "CYX")


def _select_plane(
    image: BioImage, index: int, projection: str, z: int, t: int
) -> np.ndarray:
    """The `YX` plane for one channel, under the given projection policy.

    `"single"` takes the plane at `z`/`t` directly. `"max"`/`"mean"` read the
    whole Z stack for the channel at `t` and reduce over Z, before any
    normalisation, so the percentile stretch in `_normalise` sees the
    projected plane rather than one slice of it. When the file has no Z axis,
    or Z has extent 1, a reduction over one plane is that plane, so all three
    modes return the single plane rather than raising: `idr0047` (Z=25) and
    `idr0062` (Z=236) make this a real branch, not a hypothetical one, but a
    `CYX` file with no Z axis at all must still work under `projection="max"`
    without the caller special-casing it.
    """
    z_size = int(image.dims.Z)
    if projection == "single" or z_size <= 1:
        return image.get_image_data("YX", C=index, Z=z, T=t)
    stack = image.get_image_data("ZYX", C=index, T=t)
    if projection == "max":
        return stack.max(axis=0)
    return stack.mean(axis=0)


def _normalise(plane: np.ndarray) -> np.ndarray:
    """Convert to native-endian float64 and percentile-normalise to [0, 1]."""
    values = plane.astype(np.float64)
    lo, hi = np.percentile(values, (_LOWER_PERCENTILE, _UPPER_PERCENTILE))
    if hi - lo > 0:
        return np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    return np.zeros_like(values)


def open_image(
    path: str | Path,
    *,
    channels: dict[str, int] | None = None,
    pixel_size_um: float | None = None,
    projection: str = "single",
    z: int = 0,
    t: int = 0,
) -> LoadedImage:
    """Open a microscopy file, resolve roles, and return normalised planes.

    `pixel_size_um` has no default: it is the explicit argument if given,
    then metadata read from the file, then a `PixelSizeError` naming the
    file. A wrong length scale propagates silently into every length-derived
    quantity downstream, so an absent calibration stops the run rather than
    guessing one.
    """
    if projection not in _SUPPORTED_PROJECTIONS:
        raise ValueError(
            f"unsupported projection {projection!r}; supported: "
            f"{sorted(_SUPPORTED_PROJECTIONS)}"
        )

    path = Path(path)
    image = BioImage(path)

    channel_names = [str(n) for n in (image.channel_names or [])]
    emission_nm = _emission_nm(path, len(channel_names))
    roles = resolve_roles(channel_names, emission_nm, explicit=channels)
    resolved_pixel_size_um = _resolve_pixel_size_um(path, image, pixel_size_um)

    planes = {
        role_name: _normalise(
            _select_plane(image, resolution.index, projection, z, t)
        )
        for role_name, resolution in roles.items()
    }

    return LoadedImage(
        planes=planes,
        pixel_size_um=resolved_pixel_size_um,
        roles=roles,
        projection=projection,
        axes=_squeezed_axes(image),
        dtype=str(image.dtype),
        channel_names=channel_names,
        emission_nm=emission_nm,
    )
