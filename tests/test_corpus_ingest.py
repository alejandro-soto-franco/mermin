"""Corpus-backed role resolution suite.

Each corpus entry's probe record (`corpus/src/mermin_corpus/probe.py`, as
recorded in `manifest.toml` and each entry's `meta.json`) predicts how
`resolve_roles` will assign `nuclear` and `fibre` for that file's actual
channel identity. This suite opens the real files and asserts the resolver
reaches exactly the mechanism and indices the probe record predicts, so a
regression in `mermin.roles` or `mermin.ingest` shows up against real data,
not only the synthetic fixtures in `test_roles.py` and `test_ingest.py`.

Skipped entirely, with a stated reason, when the corpus volume is not
mounted. Marked `corpus` (declared in `mermin-py/pyproject.toml`; mirrored in
`tests/conftest.py::pytest_configure` since a bare `pytest tests/` run from
the repository root has no root-level pytest config to read the former from).

The manifest-presence check runs before importing `mermin.ingest` (which
pulls in `bioio`), so the module skips cleanly on a machine that has neither
the corpus volume nor the heavy imaging dependencies installed, rather than
erroring on an unrelated `ModuleNotFoundError`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

MANIFEST_PATH = Path("/mnt/ASF-EX1/mermin-corpus/manifest.toml")

pytestmark = pytest.mark.corpus

if not MANIFEST_PATH.exists():
    pytest.skip(
        f"{MANIFEST_PATH} not mounted; corpus-backed suite skipped",
        allow_module_level=True,
    )

import tomllib

from mermin.ingest import PixelSizeError, open_image
from mermin.roles import UnresolvableRoleError


def _manifest() -> dict:
    with MANIFEST_PATH.open("rb") as f:
        return tomllib.load(f)


def _entry_artefact_path(entry_id: str) -> Path:
    """The on-disk artefact path for a manifest entry, via its `meta.json`.

    Reads the manifest for the entry's partition, then the `meta.json` that
    sits beside the entry's data for the artefact path the phase 1 prober
    recorded, rather than hardcoding a corpus layout here.
    """
    manifest = _manifest()
    entry = next((e for e in manifest["entry"] if e["id"] == entry_id), None)
    if entry is None:
        raise KeyError(f"{entry_id!r} not found in {MANIFEST_PATH}")
    root = Path(manifest["root"])
    meta_path = root / entry["partition"] / entry_id / "meta.json"
    with meta_path.open() as f:
        meta = json.load(f)
    return Path(meta["artefacts"][0]["path"])


MONTANO_DIR = Path(
    "/home/alejandrosotofranco/mermin/mermin-tests/data/montano/"
    "postmeno-vestrogen-hVF-2026-04"
)


# --- montano-hvf-2026-04 (registered in place; private, not in the manifest's
# --- meta.json artefact scheme, so its paths are hardcoded per the task) ---


def test_montano_c01_dv_resolves_nuclear_and_fibre_by_emission():
    img = open_image(MONTANO_DIR / "c01_dv.tif")
    assert img.pixel_size_um == pytest.approx(0.69)
    assert img.roles["nuclear"].mechanism == "emission"
    assert img.roles["fibre"].mechanism == "emission"
    assert img.roles["nuclear"].index == 0
    assert img.roles["fibre"].index == 1


def test_montano_c08_dsv_resolves_nuclear_1_fibre_2_by_emission():
    # THE DEFECT THE WHOLE PROJECT EXISTS TO FIX: this file's channels are
    # emission-ordered [525.0, 470.0, 666.0], not [470.0, 666.0, ...] as in
    # c01_dv.tif. The old hardcoded {"dapi": 0, "vimentin": 1} channel order
    # (see mermin.ingest's module docstring) would have picked channel 0
    # (525.0 nm, neither role) as nuclear and channel 1 (470.0 nm, the real
    # nuclear channel) as fibre. The emission-based resolver must instead
    # find the true minimum (470.0 nm, index 1) as nuclear and the true
    # maximum (666.0 nm, index 2) as fibre, regardless of acquisition order.
    img = open_image(MONTANO_DIR / "c08_dsv.tif")
    assert img.pixel_size_um == pytest.approx(0.69)
    assert img.roles["nuclear"].mechanism == "emission"
    assert img.roles["fibre"].mechanism == "emission"
    assert img.roles["nuclear"].index == 1
    assert img.roles["fibre"].index == 2


def test_montano_calibration_reference_is_unresolvable():
    # jm008.3_..._w4.tif is the batch's pixel-size calibration reference, a
    # single-channel file with no emission metadata, not an analysis target.
    # It must raise rather than pairing its one channel with itself.
    calibration_ref = (
        MONTANO_DIR
        / "jm008.3_ifg2vg3p_tgfbe2response_2hr_C05_sx_1_sy_1_w4.tif"
    )
    with pytest.raises(UnresolvableRoleError, match="one channel"):
        open_image(calibration_ref)


# --- idr0021-1884807: raw wavelength channel names, no readable pixel size ---


def test_idr0021_raises_pixel_size_error_without_an_explicit_value():
    path = _entry_artefact_path("idr0021-1884807")
    with pytest.raises(PixelSizeError, match="no pixel size"):
        open_image(path)


def test_idr0021_resolves_nuclear_0_fibre_2_by_emission():
    path = _entry_artefact_path("idr0021-1884807")
    img = open_image(path, pixel_size_um=1.0)
    assert img.channel_names == ["442.0", "525.0", "615.0"]
    assert img.dtype == ">f4"
    assert img.roles["nuclear"].mechanism == "emission"
    assert img.roles["fibre"].mechanism == "emission"
    assert img.roles["nuclear"].index == 0
    assert img.roles["fibre"].index == 2


# --- idr0062-6001240: LaminB1/Dapi, CZYX with Z=236 ---


def test_idr0062_resolves_fibre_0_nuclear_1_by_name():
    path = _entry_artefact_path("idr0062-6001240")
    img = open_image(path, pixel_size_um=1.0)
    assert img.channel_names == ["LaminB1", "Dapi"]
    assert img.axes == "CZYX"
    assert img.roles["nuclear"].mechanism == "name"
    assert img.roles["fibre"].mechanism == "name"
    assert img.roles["fibre"].index == 0
    assert img.roles["nuclear"].index == 1


# --- idr0047-4496763: dye names, not targets, so fibre is unknowable ---


def test_idr0047_dye_names_raise_rather_than_guess_a_fibre_channel():
    path = _entry_artefact_path("idr0047-4496763")
    # Role resolution runs before pixel-size resolution in `open_image`, so
    # this raises without needing an explicit pixel size, even though this
    # file's own metadata does carry one (1.0 um/px, per its meta.json).
    with pytest.raises(UnresolvableRoleError, match="explicit mapping"):
        open_image(path)


# --- the four phantom-* entries: synthetic names, positional fallback ---

PHANTOM_IDS = [
    "phantom-uniform",
    "phantom-defect-pair",
    "phantom-radial",
    "phantom-hexatic",
]


@pytest.mark.parametrize("entry_id", PHANTOM_IDS)
def test_phantom_resolves_by_position_with_a_warning(entry_id):
    path = _entry_artefact_path(entry_id)
    with pytest.warns(UserWarning, match="position"):
        img = open_image(path)
    assert img.roles["nuclear"].mechanism == "position"
    assert img.roles["fibre"].mechanism == "position"
    assert img.roles["nuclear"].index == 0
    assert img.roles["fibre"].index == 1
    # Read from the TIFF's own metadata with no explicit argument.
    assert img.pixel_size_um == pytest.approx(0.5)
