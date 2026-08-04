"""Configure pytest to find mermin modules, and share corpus-lookup helpers.

`tests/` has no `__init__.py`, so it is not a package: a bare `pytest tests/`
run from the repository root cannot `from tests.test_corpus_ingest import
...`, since nothing named `tests` is importable (confirmed: that import
raises `ModuleNotFoundError: No module named 'tests'` under this
invocation). `conftest.py` itself has no such problem, since pytest always
makes it importable from any test module in the same directory, so the
corpus-manifest helpers shared by `test_corpus_ingest.py` and
`test_corpus_segmentation.py` live here instead of in either suite.
"""
import json
import sys
import tomllib
from pathlib import Path

# Add the python directory to the path so we can import mermin
python_dir = Path(__file__).parent.parent / "python"
sys.path.insert(0, str(python_dir))

# The golden suite (test_corpus_goldens.py) consumes the corpus tooling
# (mermin_corpus.goldens, .invariants, .generate) directly, the same way it
# consumes mermin itself. mermin_corpus/__init__.py is import-light, so
# putting this on the path costs nothing for the hermetic (non-corpus) tests
# that never import anything under mermin_corpus.
corpus_src_dir = Path(__file__).parent.parent / "corpus" / "src"
sys.path.insert(0, str(corpus_src_dir))

MANIFEST_PATH = Path("/mnt/ASF-EX1/mermin-corpus/manifest.toml")


def pytest_configure(config):
    """Register markers declared canonically in mermin-py/pyproject.toml.

    That file is the package's own pytest config, but `tests/` sits at the
    repository root with no root-level pyproject.toml or pytest.ini for
    pytest's own config discovery to find, so a bare `pytest tests/` never
    reads it. Mirroring the declaration here is what actually silences the
    unknown-mark warning for that invocation.
    """
    config.addinivalue_line(
        "markers", "corpus: requires the ASF-EX1 mermin-corpus volume mounted"
    )


def require_corpus() -> None:
    """Skip the calling module, with a reason naming the volume, unless the
    ASF-EX1 corpus manifest is present.

    Call at module level before importing anything that needs the volume or
    the heavy imaging dependencies it implies, so an unmounted drive skips
    cleanly rather than failing on an unrelated `ModuleNotFoundError`.
    """
    if not MANIFEST_PATH.exists():
        import pytest

        pytest.skip(
            f"{MANIFEST_PATH} not mounted; corpus-backed suite skipped",
            allow_module_level=True,
        )


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
