import json

import pytest

from mermin_corpus.fetch import fetch_entry
from mermin_corpus.manifest import load

MANIFEST = """\
schema = 1
root = "PLACEHOLDER"

[[entry]]
id = "idr-test"
rung = 1
partition = "commercial-safe"
access = "OPEN"
licence = "CC-BY-4.0"
attribution = "test"
doi = ""
retain_raw = true
[entry.source]
kind = "ome-zarr"
url = "https://example.test/6001240.zarr"
level = 1
[entry.expected]
axes = ""
[entry.roles]
nuclear = { by = "channel_name", value = "DAPI" }
fibre = { by = "channel_name", value = "AF647" }
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

ZATTRS = {
    "multiscales": [{"datasets": [{"path": "0"}, {"path": "1"}], "axes": [
        {"name": "c"}, {"name": "y"}, {"name": "x"}]}],
    "omero": {"channels": [{"label": "DAPI"}, {"label": "AF647"}]},
}


@pytest.fixture()
def fake_store(monkeypatch, tmp_path):
    base = "https://example.test/6001240.zarr"
    files = {
        f"{base}/.zgroup": json.dumps({"zarr_format": 2}).encode(),
        f"{base}/.zattrs": json.dumps(ZATTRS).encode(),
        f"{base}/1/.zarray": json.dumps(
            {"shape": [2, 4, 4], "chunks": [1, 4, 4], "zarr_format": 2,
             "dtype": "<u2", "compressor": None, "fill_value": 0, "order": "C",
             "filters": None}
        ).encode(),
        f"{base}/1/0.0.0": b"\x00" * 32,
        f"{base}/1/1.0.0": b"\x01" * 32,
        f"{base}/0/.zarray": json.dumps({"shape": [2, 8, 8]}).encode(),
        f"{base}/0/0.0.0": b"\x02" * 128,
    }

    def fake_get(url, **kwargs):
        class R:
            status_code = 200 if url in files else 404
            content = files.get(url, b"")

            def raise_for_status(self):
                if self.status_code != 200:
                    raise RuntimeError(f"404 {url}")

        return R()

    monkeypatch.setattr("mermin_corpus.fetch._zarr_get", lambda url: files[url])
    return base


def test_ome_zarr_mirrors_only_the_named_level(tmp_path, monkeypatch, fake_store):
    monkeypatch.setenv("MERMIN_CORPUS_ROOT", str(tmp_path))
    p = tmp_path / "manifest.toml"
    p.write_text(MANIFEST.replace("PLACEHOLDER", str(tmp_path)))

    fetch_entry(load(p), "idr-test")

    store = tmp_path / "commercial-safe" / "idr-test" / "raw" / "6001240.zarr"
    assert (store / ".zattrs").exists()
    assert (store / ".zgroup").exists()
    assert (store / "1" / ".zarray").exists()
    assert (store / "1" / "0.0.0").exists()
    assert not (store / "0").exists()
    assert load(p).get("idr-test").provenance["status"] == "fetched"


def test_zarr_listing_prefers_consolidated_metadata(monkeypatch):
    from mermin_corpus import fetch

    consolidated = json.dumps({"metadata": {
        "1/.zarray": {"shape": [2, 4, 4], "chunks": [1, 4, 4]},
    }}).encode()
    calls = []

    def fake_get(url):
        calls.append(url)
        if url.endswith("/.zmetadata"):
            return consolidated
        raise AssertionError(f"should not have fetched {url}")

    monkeypatch.setattr(fetch, "_zarr_get", fake_get)
    keys = fetch._zarr_listing("https://example.test/s.zarr", "1")
    assert keys == ["1/.zarray", "1/0.0.0", "1/1.0.0"]
    assert calls == ["https://example.test/s.zarr/.zmetadata"]


def test_zarr_listing_falls_back_to_the_level_zarray(monkeypatch):
    from mermin_corpus import fetch

    def fake_get(url):
        if url.endswith("/.zmetadata"):
            raise RuntimeError("404")
        return json.dumps({"shape": [1, 8, 8], "chunks": [1, 4, 8]}).encode()

    monkeypatch.setattr(fetch, "_zarr_get", fake_get)
    assert fetch._zarr_listing("https://example.test/s.zarr", "2") == [
        "2/.zarray", "2/0.0.0", "2/0.1.0",
    ]
