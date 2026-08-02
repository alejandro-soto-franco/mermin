import io
import zipfile

import pytest

from mermin_corpus.fetch import fetch_entry
from mermin_corpus.manifest import load

TEMPLATE = """\
schema = 1
root = "PLACEHOLDER"

[[entry]]
id = "ENTRY_ID"
rung = 2
partition = "commercial-safe"
access = "OPEN"
licence = "CC-BY-4.0"
attribution = "test"
doi = ""
retain_raw = true
[entry.source]
kind = "KIND"
url = "URL"
MEMBERS
[entry.expected]
axes = ""
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


def _manifest(tmp_path, entry_id, kind, url, members=""):
    p = tmp_path / "manifest.toml"
    p.write_text(
        TEMPLATE.replace("PLACEHOLDER", str(tmp_path))
        .replace("ENTRY_ID", entry_id)
        .replace("KIND", kind)
        .replace("URL", url)
        .replace("MEMBERS", members)
    )
    return p


class FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload
        self.status_code = 200
        self.headers = {"content-length": str(len(payload))}

    def raise_for_status(self):
        return None

    def iter_bytes(self, chunk_size=65536):
        for i in range(0, len(self._payload), chunk_size):
            yield self._payload[i : i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture()
def fake_http(monkeypatch):
    store: dict[str, bytes] = {}

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def stream(self, method, url, **k):
            return FakeResponse(store[url])

    monkeypatch.setattr("mermin_corpus.fetch.httpx.Client", FakeClient)
    return store


def test_http_entry_downloads_to_raw(tmp_path, monkeypatch, fake_http):
    monkeypatch.setenv("MERMIN_CORPUS_ROOT", str(tmp_path))
    fake_http["https://example.test/a.tif"] = b"TIFFBYTES"
    p = _manifest(tmp_path, "ome-sample", "http", "https://example.test/a.tif")
    out = fetch_entry(load(p), "ome-sample")
    assert out.read_bytes() == b"TIFFBYTES"
    assert out.name == "a.tif"
    assert load(p).get("ome-sample").provenance["status"] == "fetched"


def test_zip_index_extracts_only_named_members(tmp_path, monkeypatch, fake_http):
    monkeypatch.setenv("MERMIN_CORPUS_ROOT", str(tmp_path))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("plate/keep_w1.tif", b"DAPI")
        z.writestr("plate/keep_w2.tif", b"TUBULIN")
        z.writestr("plate/drop_w1.tif", b"NO")
    fake_http["https://example.test/plate.zip"] = buf.getvalue()

    members = 'members = ["plate/keep_w1.tif", "plate/keep_w2.tif"]'
    p = _manifest(tmp_path, "bbbc021-w1", "zip-index", "https://example.test/plate.zip", members)
    fetch_entry(load(p), "bbbc021-w1")

    raw = tmp_path / "commercial-safe" / "bbbc021-w1" / "raw"
    got = sorted(f.name for f in raw.rglob("*.tif"))
    assert got == ["keep_w1.tif", "keep_w2.tif"]


def test_zip_index_rejects_a_traversal_member(tmp_path, monkeypatch, fake_http):
    monkeypatch.setenv("MERMIN_CORPUS_ROOT", str(tmp_path))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("../escape.tif", b"NO")
    fake_http["https://example.test/evil.zip"] = buf.getvalue()

    members = 'members = ["../escape.tif"]'
    p = _manifest(tmp_path, "evil", "zip-index", "https://example.test/evil.zip", members)
    from mermin_corpus.errors import FetchError

    with pytest.raises(FetchError, match="escapes"):
        fetch_entry(load(p), "evil")
