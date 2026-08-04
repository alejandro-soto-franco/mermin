"""Hermetic golden suite: `mermin.analyze()` against a phantom generated in
process, pinned by `tests/hermetic_goldens/*.json`.

This is the driveless, `tomlkit`-free sibling of `test_corpus_goldens.py`.
`mermin_corpus.hermetic` (see its module docstring) needs neither the ASF-EX1
mount nor `tomlkit`, so unlike `test_corpus_goldens.py`'s golden comparison
(`test_matches_golden`, gated on both), the test below is not marked
`corpus` and runs under CI's `floors` and `python-tests` jobs, and under this
project's standard `pytest tests/` invocation, regardless of which drive or
optional dependency either happens to have.

It exists to close a gap the final whole-branch review found: the
`scikit-image>=0.25.2` floor `mermin-py/pyproject.toml` declares rests on a
determinism claim (`skimage.segmentation.watershed`'s boundary tie-breaking
is stable at and above 0.25.2) that no job actually checked, because the
suite that could check it (`test_corpus_goldens.py`) is both
`-m "not corpus"`-deselected and `tomlkit`-skipped in exactly the jobs that
verify the floor. A comparator that never runs where the floor is enforced
does not enforce anything.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip(
    "mermin._native", reason="mermin._native is not built in this checkout"
)

from mermin_corpus.goldens import build_record, compare
from mermin_corpus.hermetic import HERMETIC_ENTRIES, build_hermetic_record, hermetic_golden_path

ENTRY_IDS = sorted(HERMETIC_ENTRIES)


@pytest.mark.parametrize("entry_id", ENTRY_IDS)
def test_matches_hermetic_golden(entry_id, tmp_path):
    """`analyze()` on this in-process phantom, today, must reproduce its
    committed hermetic golden.

    Same `environment`-kind-is-reported-not-failed split as
    `test_corpus_goldens.test_matches_golden`: a recorded dependency version
    explaining a difference elsewhere is not itself a regression, but every
    other difference is.
    """
    golden = json.loads(hermetic_golden_path(entry_id).read_text())
    current = build_hermetic_record(entry_id, tmp_dir=tmp_path)

    diffs = compare(golden, current)
    real = [d for d in diffs if d.kind != "environment"]
    assert not real, f"{entry_id}: {len(real)} golden mismatch(es):\n" + "\n".join(
        f"  {d}" for d in real
    )


def test_hermetic_golden_files_exist_for_every_entry():
    """A `HERMETIC_ENTRIES` id with no committed golden would otherwise pass
    the parametrized test above by never running it at all (a bad id used to
    build `ENTRY_IDS` would just make the loop shorter, not fail)."""
    for entry_id in ENTRY_IDS:
        assert hermetic_golden_path(entry_id).exists(), (
            f"{entry_id}: no committed golden at {hermetic_golden_path(entry_id)}"
        )
