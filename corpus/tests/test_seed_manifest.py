from pathlib import Path

from mermin_corpus.manifest import load, validate_entry

SEED = Path(__file__).parent.parent / "manifest.seed.toml"


def test_seed_manifest_validates():
    m = load(SEED)
    assert m.schema == 1
    assert m.entries
    for e in m.entries:
        validate_entry(e)


def test_every_rung_is_represented():
    rungs = {e.rung for e in load(SEED).entries}
    assert rungs == {0, 1, 2, 3, 4}


def test_restricted_sources_are_not_commercial_safe():
    for e in load(SEED).entries:
        if "astrazeneca" in e.licence.lower() or e.access == "PRIVATE":
            assert e.partition in {"eval-only", "private"}


def test_all_phantom_generators_are_registered():
    from mermin_corpus.phantoms import GENERATORS

    registered = {
        e.source["generator"] for e in load(SEED).entries if e.source["kind"] == "generated"
    }
    assert registered == set(GENERATORS)
