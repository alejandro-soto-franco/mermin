import json

import numpy as np
import pytest
import tifffile

from mermin_corpus.errors import ProbeError
from mermin_corpus.probe import _pixel_size_um, probe_entry, probe_file

MANIFEST = """\
schema = 1
root = "PLACEHOLDER"

[[entry]]
id = "phantom-uniform"
rung = 0
partition = "commercial-safe"
access = "OPEN"
licence = "CC0-1.0"
attribution = "generated"
doi = ""
retain_raw = true
[entry.source]
kind = "generated"
generator = "uniform_director"
seed = 1
[entry.expected]
axes = ""
shape = []
dtype = ""
size_c = 0
pixel_size_um = 0.0
reader = ""
[entry.roles]
nuclear = { by = "index", value = 0 }
fibre = { by = "index", value = 1 }
[entry.policy]
projection = "single"
z = 0
t = 0
[entry.provenance]
access_date = ""
raw_sha256 = ""
raw_bytes = 0
status = "pending"
"""


def test_probe_file_reports_axes_shape_and_dtype(tmp_path):
    p = tmp_path / "two.tif"
    tifffile.imwrite(p, np.zeros((2, 32, 32), dtype=np.uint16),
                     imagej=True, metadata={"axes": "CYX"})
    info = probe_file(p)
    assert info["axes"] == "CYX"
    assert info["shape"] == [2, 32, 32]
    assert info["dtype"] == "uint16"
    assert info["size_c"] == 2
    assert info["reader"]


def test_probe_file_reports_pixel_size_when_present(tmp_path):
    p = tmp_path / "cal.tif"
    tifffile.imwrite(p, np.zeros((1, 8, 8), dtype=np.uint16), imagej=True,
                     metadata={"axes": "CYX", "unit": "micron"},
                     resolution=(1 / 0.69, 1 / 0.69))
    assert probe_file(p)["pixel_size_um"] == pytest.approx(0.69, rel=1e-6)


def test_probe_file_reports_none_when_uncalibrated(tmp_path):
    p = tmp_path / "raw.tif"
    tifffile.imwrite(p, np.zeros((1, 8, 8), dtype=np.uint16))
    assert probe_file(p)["pixel_size_um"] is None


def test_probe_entry_writes_meta_and_fills_expected(tmp_path, monkeypatch):
    monkeypatch.setenv("MERMIN_CORPUS_ROOT", str(tmp_path))
    from mermin_corpus.fetch import fetch_entry
    from mermin_corpus.manifest import load

    p = tmp_path / "manifest.toml"
    p.write_text(MANIFEST.replace("PLACEHOLDER", str(tmp_path)))
    fetch_entry(load(p), "phantom-uniform")

    info = probe_entry(load(p), "phantom-uniform")
    meta = tmp_path / "commercial-safe" / "phantom-uniform" / "meta.json"
    meta_doc = json.loads(meta.read_text())
    assert meta_doc["summary"]["axes"] == "CYX"
    assert meta_doc["n_artefacts"] == 1
    assert meta_doc["truncated"] is False
    assert len(meta_doc["artefacts"]) == 1
    assert info["size_c"] == 2

    e = load(p).get("phantom-uniform")
    assert e.expected["axes"] == "CYX"
    assert e.expected["size_c"] == 2
    assert e.expected["n_artefacts"] == 1
    assert e.expected["uniform"] is True
    assert e.expected["size_c_observed"] == [2]
    assert e.provenance["status"] == "probed"


def test_emission_survives_a_single_label_written_as_a_bare_string(tmp_path):
    label = '<MetaData><PlaneInfo><prop id="wavelength" type="float" value="470"/></PlaneInfo></MetaData>'
    p = tmp_path / "one.tif"
    tifffile.imwrite(p, np.zeros((1, 8, 8), dtype=np.uint16), imagej=True,
                     metadata={"axes": "CYX", "Labels": [label]})
    assert probe_file(p)["emission_nm"] == [470.0]


def test_emission_is_length_preserving_when_a_label_does_not_parse(tmp_path):
    good = '<MetaData><PlaneInfo><prop id="wavelength" type="float" value="666"/></PlaneInfo></MetaData>'
    p = tmp_path / "partial.tif"
    tifffile.imwrite(p, np.zeros((2, 8, 8), dtype=np.uint16), imagej=True,
                     metadata={"axes": "CYX", "Labels": [good, "no wavelength here"]})
    assert probe_file(p)["emission_nm"] == [666.0, None]


def test_metamorph_spatial_calibration_is_read_when_tags_are_absent(tmp_path):
    label = (
        '<MetaData><PlaneInfo>'
        '<prop id="spatial-calibration-state" type="bool" value="on"/>'
        '<prop id="spatial-calibration-x" type="float" value="0.69"/>'
        '<prop id="spatial-calibration-units" type="string" value="Microns"/>'
        '</PlaneInfo></MetaData>'
    )
    p = tmp_path / "mm.tif"
    tifffile.imwrite(p, np.zeros((1, 8, 8), dtype=np.uint16), imagej=True,
                     metadata={"axes": "CYX", "Labels": [label]})
    assert probe_file(p)["pixel_size_um"] == pytest.approx(0.69)


def test_probe_entry_skips_dotfiles_when_choosing_a_candidate(tmp_path, monkeypatch):
    """A registered local directory may carry a .gitkeep placeholder that kept
    the gitignored directory tracked before real data was dropped in; it must
    never be the file probe reads.
    """
    monkeypatch.setenv("MERMIN_CORPUS_ROOT", str(tmp_path))
    from mermin_corpus.fetch import fetch_entry
    from mermin_corpus.manifest import load

    data_dir = tmp_path / "local-data"
    data_dir.mkdir()
    (data_dir / ".gitkeep").write_bytes(b"")
    tifffile.imwrite(data_dir / "a.tif", np.zeros((2, 8, 8), dtype=np.uint16),
                     imagej=True, metadata={"axes": "CYX"})

    manifest = f"""\
schema = 1
root = "{tmp_path}"

[[entry]]
id = "local-with-gitkeep"
rung = 4
partition = "private"
access = "PRIVATE"
licence = "collaborator data, not redistributable"
attribution = "test"
doi = ""
retain_raw = true
[entry.source]
kind = "local"
path = "{data_dir}"
[entry.expected]
axes = ""
shape = []
dtype = ""
size_c = 0
pixel_size_um = 0.0
reader = ""
[entry.roles]
nuclear = {{ by = "index", value = 0 }}
fibre = {{ by = "index", value = 1 }}
[entry.policy]
projection = "single"
z = 0
t = 0
[entry.provenance]
access_date = ""
raw_sha256 = ""
raw_bytes = 0
status = "pending"
"""
    p = tmp_path / "manifest.toml"
    p.write_text(manifest)
    fetch_entry(load(p), "local-with-gitkeep")

    info = probe_entry(load(p), "local-with-gitkeep")
    assert info["path"].endswith("a.tif")
    assert info["axes"] == "CYX"


def test_pixel_size_um_is_none_when_the_reader_reports_no_sizes_object(tmp_path):
    """bioio-ome-zarr returns physical_pixel_sizes = None outright, rather
    than a sizes object with unset X/Y, when a store's transform carries no
    parseable scale. Reachable against a real IDR store missing
    coordinateTransformations scale entries.
    """
    p = tmp_path / "no-tiff-suffix.zarr"
    p.mkdir()

    class FakeImage:
        physical_pixel_sizes = None

    assert _pixel_size_um(p, FakeImage()) is None


def test_probe_file_raises_probe_error_for_an_unreadable_format(tmp_path):
    """A format with no installed bioio reader plugin must surface as a
    ProbeError so the CLI's per-entry handling records it and continues the
    batch, rather than an uncaught bioio exception crashing the whole run.
    """
    p = tmp_path / "unreadable.czi"
    p.write_bytes(b"not a real czi file")
    with pytest.raises(ProbeError, match="missing bioio reader plugin"):
        probe_file(p)


def test_probe_file_wraps_a_failure_that_surfaces_only_on_dims_access(tmp_path, monkeypatch):
    """A reader can construct BioImage successfully and only fail once
    `.dims` is touched, which is lazy in bioio. Reachable against a real
    NGFF v0.3 store: bioio-ome-zarr assumes each multiscales axis is a dict
    with a "name" key, but v0.3 stores a plain list of axis name strings,
    so `ax["name"]` raises TypeError on first `.dims` access, after
    `BioImage(path)` has already returned.
    """
    import mermin_corpus.probe as probe_module

    class ExplodesOnDims:
        def __init__(self, path):
            pass

        @property
        def dims(self):
            raise TypeError("string indices must be integers, not 'str'")

    monkeypatch.setattr(probe_module, "BioImage", ExplodesOnDims)
    p = tmp_path / "old-style-axes.zarr"
    p.mkdir()
    with pytest.raises(ProbeError, match="cannot parse this store's metadata dialect"):
        probe_file(p)


def test_probe_entry_finds_files_nested_below_the_target_directory(tmp_path, monkeypatch):
    """A zip-index fetch preserves each member's archive subpath (Task 6), so
    a multi-field source like BBBC021 puts its files one directory below
    `raw/`, inside a plate subdirectory, rather than directly in it.
    `iterdir()` alone would hand the reader that subdirectory itself, which
    no bioio plugin can open. Reachable against a real BBBC021 fetch.
    """
    monkeypatch.setenv("MERMIN_CORPUS_ROOT", str(tmp_path))
    from mermin_corpus.fetch import fetch_entry
    from mermin_corpus.manifest import load

    data_dir = tmp_path / "local-data"
    plate_dir = data_dir / "Week4_27481"
    plate_dir.mkdir(parents=True)
    tifffile.imwrite(plate_dir / "a.tif", np.zeros((2, 8, 8), dtype=np.uint16),
                     imagej=True, metadata={"axes": "CYX"})

    manifest = f"""\
schema = 1
root = "{tmp_path}"

[[entry]]
id = "local-nested"
rung = 4
partition = "private"
access = "PRIVATE"
licence = "collaborator data, not redistributable"
attribution = "test"
doi = ""
retain_raw = true
[entry.source]
kind = "local"
path = "{data_dir}"
[entry.expected]
axes = ""
shape = []
dtype = ""
size_c = 0
pixel_size_um = 0.0
reader = ""
[entry.roles]
nuclear = {{ by = "index", value = 0 }}
fibre = {{ by = "index", value = 1 }}
[entry.policy]
projection = "single"
z = 0
t = 0
[entry.provenance]
access_date = ""
raw_sha256 = ""
raw_bytes = 0
status = "pending"
"""
    p = tmp_path / "manifest.toml"
    p.write_text(manifest)
    fetch_entry(load(p), "local-nested")

    info = probe_entry(load(p), "local-nested")
    assert info["path"].endswith("a.tif")
    assert info["axes"] == "CYX"


def test_probe_entry_treats_a_zarr_store_directory_as_one_candidate(tmp_path, monkeypatch):
    """An OME-Zarr store is itself a directory of chunk files. The reader
    needs the store root, not one of its chunk files, so recursing past
    `.zgroup`/`.zarray` the way BBBC021's plate subdirectory is recursed
    into would hand bioio a raw chunk file and it would fail to open it.
    Reachable against any real ome-zarr fetch; caught by probing a fetched
    IDR store before this fix landed.
    """
    monkeypatch.setenv("MERMIN_CORPUS_ROOT", str(tmp_path))
    from mermin_corpus.probe import _artefact_candidates

    raw = tmp_path / "raw"
    store = raw / "6001240.zarr"
    chunk_dir = store / "0" / "0"
    chunk_dir.mkdir(parents=True)
    (store / ".zgroup").write_text('{"zarr_format": 2}')
    (store / ".zattrs").write_text("{}")
    (chunk_dir / "0.0.0.0.0").write_bytes(b"\x00" * 16)

    assert _artefact_candidates(raw) == [store]


@pytest.mark.drive
def test_drive_reprobe_matches_the_recorded_expected_for_variable_entries():
    """Re-probing the live corpus must reproduce what is already recorded.

    This is the automated check for the branch's central claim: that
    montano-hvf-2026-04's `expected` block now records the batch's channel
    variation across all forty-nine files, rather than one file's facts
    standing in for the whole entry.
    """
    import os
    from pathlib import Path

    if not os.path.ismount(Path("/mnt/ASF-EX1")):
        pytest.skip("ASF-EX1 is not mounted")

    from mermin_corpus.manifest import load, manifest_path

    entry_ids = ["montano-hvf-2026-04", "bbbc021-week4-27481", "phantom-uniform"]
    before = {eid: dict(load(manifest_path()).get(eid).expected) for eid in entry_ids}

    for eid in entry_ids:
        probe_entry(load(manifest_path()), eid)

    after = {eid: dict(load(manifest_path()).get(eid).expected) for eid in entry_ids}
    for eid in entry_ids:
        assert after[eid] == before[eid], (
            f"{eid}: re-probing disagrees with the recorded expected block"
        )

    montano = after["montano-hvf-2026-04"]
    assert montano["uniform"] is False
    assert montano["size_c_observed"] == [1, 2, 3]


def test_probe_imports_no_mermin_code():
    import subprocess
    import sys

    code = (
        "import sys, mermin_corpus.probe;"
        "bad=[m for m in sys.modules if m=='mermin' or m.startswith('mermin.')];"
        "print(bad); assert not bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
