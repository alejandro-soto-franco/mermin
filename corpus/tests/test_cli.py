import pytest

from mermin_corpus.cli import main

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


@pytest.fixture()
def corpus(tmp_path, monkeypatch):
    monkeypatch.setenv("MERMIN_CORPUS_ROOT", str(tmp_path))
    (tmp_path / "manifest.toml").write_text(MANIFEST.replace("PLACEHOLDER", str(tmp_path)))
    return tmp_path


def test_status_lists_entries(corpus, capsys):
    assert main(["status"]) == 0
    out = capsys.readouterr().out
    assert "phantom-uniform" in out
    assert "pending" in out


def test_fetch_then_probe_by_rung(corpus, capsys):
    assert main(["fetch", "--rung", "0"]) == 0
    assert main(["probe", "--rung", "0"]) == 0
    capsys.readouterr()
    main(["status"])
    assert "probed" in capsys.readouterr().out


def test_fetch_requires_a_selector(corpus):
    with pytest.raises(SystemExit):
        main(["fetch"])


def test_unknown_id_exits_nonzero(corpus, capsys):
    assert main(["fetch", "--id", "nope"]) == 1
    assert "unknown entry" in capsys.readouterr().err
