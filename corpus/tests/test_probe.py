import json

import numpy as np
import pytest
import tifffile

from mermin_corpus.probe import probe_entry, probe_file

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
    assert json.loads(meta.read_text())["axes"] == "CYX"
    assert info["size_c"] == 2

    e = load(p).get("phantom-uniform")
    assert e.expected["axes"] == "CYX"
    assert e.expected["size_c"] == 2
    assert e.provenance["status"] == "probed"


def test_probe_imports_no_mermin_code():
    import subprocess
    import sys

    code = (
        "import sys, mermin_corpus.probe;"
        "bad=[m for m in sys.modules if m=='mermin' or m.startswith('mermin.')];"
        "print(bad); assert not bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
