import numpy as np
import pytest
import tifffile

from mermin_corpus.errors import FetchError
from mermin_corpus.fetch import fetch_entry, raw_path
from mermin_corpus.manifest import load

GENERATED = """\
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
seed = 20260802
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

[[entry]]
id = "montano-hvf"
rung = 4
partition = "private"
access = "PRIVATE"
licence = "collaborator, not redistributable"
attribution = "Josh Montano, hVF batch 2026-04"
doi = ""
retain_raw = true
[entry.source]
kind = "local"
path = "LOCALPATH"
[entry.expected]
axes = ""
[entry.roles]
nuclear = { by = "emission_nm", value = 470 }
fibre = { by = "emission_nm", value = 666 }
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


@pytest.fixture()
def corpus(tmp_path, monkeypatch):
    monkeypatch.setenv("MERMIN_CORPUS_ROOT", str(tmp_path))
    local = tmp_path / "source-data"
    local.mkdir()
    tifffile.imwrite(local / "a.tif", np.zeros((2, 16, 16), dtype=np.uint16))
    p = tmp_path / "manifest.toml"
    p.write_text(GENERATED.replace("PLACEHOLDER", str(tmp_path)).replace("LOCALPATH", str(local)))
    return p


def test_generated_entry_writes_a_tiff_and_records_provenance(corpus):
    m = load(corpus)
    out = fetch_entry(m, "phantom-uniform")
    assert out.exists()
    assert out.suffix == ".tif"
    e = load(corpus).get("phantom-uniform")
    assert e.provenance["status"] == "fetched"
    assert len(e.provenance["raw_sha256"]) == 64
    assert e.provenance["raw_bytes"] > 0
    assert e.provenance["access_date"]


def test_generated_entry_is_reproducible(corpus):
    m = load(corpus)
    fetch_entry(m, "phantom-uniform")
    first = load(corpus).get("phantom-uniform").provenance["raw_sha256"]
    fetch_entry(load(corpus), "phantom-uniform", force=True)
    assert load(corpus).get("phantom-uniform").provenance["raw_sha256"] == first


def test_local_entry_is_registered_in_place_without_copying(corpus, tmp_path):
    m = load(corpus)
    out = fetch_entry(m, "montano-hvf")
    assert out == tmp_path / "source-data"
    assert not (tmp_path / "private" / "montano-hvf" / "raw").exists()
    e = load(corpus).get("montano-hvf")
    assert e.provenance["status"] == "fetched"
    assert e.provenance["raw_bytes"] > 0


def test_local_entry_fails_when_the_path_is_missing(corpus, tmp_path):
    import shutil
    shutil.rmtree(tmp_path / "source-data")
    with pytest.raises(FetchError, match="does not exist"):
        fetch_entry(load(corpus), "montano-hvf")


def test_refetch_without_force_is_a_no_op(corpus):
    m = load(corpus)
    fetch_entry(m, "phantom-uniform")
    path = raw_path(load(corpus).get("phantom-uniform")) / "phantom-uniform.tif"
    stamp = path.stat().st_mtime_ns
    fetch_entry(load(corpus), "phantom-uniform")
    assert path.stat().st_mtime_ns == stamp


def test_hash_mismatch_on_refetch_raises(corpus):
    m = load(corpus)
    fetch_entry(m, "phantom-uniform")
    m2 = load(corpus)
    m2.update_provenance("phantom-uniform", raw_sha256="0" * 64)
    m2.save()
    with pytest.raises(FetchError, match="sha256"):
        fetch_entry(load(corpus), "phantom-uniform", force=True)


def test_unsupported_kind_does_not_destroy_existing_raw(corpus, tmp_path, monkeypatch):
    m = load(corpus)
    fetch_entry(m, "phantom-uniform")
    raw = tmp_path / "commercial-safe" / "phantom-uniform" / "raw"
    before = sorted(p.name for p in raw.iterdir())
    assert before

    # Every valid kind now has a fetcher, so the guard is reached by removing
    # one from the registry rather than by naming an unimplemented kind.
    from mermin_corpus import fetch as fetch_mod

    monkeypatch.delitem(fetch_mod._FETCHERS, "generated")

    with pytest.raises(FetchError, match="not implemented"):
        fetch_entry(load(corpus), "phantom-uniform", force=True)

    assert sorted(p.name for p in raw.iterdir()) == before


def test_a_failing_fetch_leaves_the_previous_artefact_intact(corpus, tmp_path, monkeypatch):
    m = load(corpus)
    fetch_entry(m, "phantom-uniform")
    raw = tmp_path / "commercial-safe" / "phantom-uniform" / "raw"
    before = sorted(p.name for p in raw.iterdir())
    original = (raw / before[0]).read_bytes()

    from mermin_corpus import fetch as fetch_mod

    def boom(entry, dest):
        raise RuntimeError("fetch failed halfway")

    monkeypatch.setitem(fetch_mod._FETCHERS, "generated", boom)

    with pytest.raises(RuntimeError, match="halfway"):
        fetch_entry(load(corpus), "phantom-uniform", force=True)

    assert sorted(p.name for p in raw.iterdir()) == before
    assert (raw / before[0]).read_bytes() == original
