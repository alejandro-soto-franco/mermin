"""Golden-record suite: `mermin.analyze()` against every reachable corpus
entry, pinned by `corpus/goldens/*.json`.

**A golden is generated, never hand-written.** These are produced by
`mermin_corpus.generate` (`python -m mermin_corpus.cli goldens`) from a real
analysis run, committed as the record of what `analyze()` produced on that
entry at that commit. Editing a golden by hand to make a failing test pass is
the defect this phase exists to catch: it hides a regression instead of
reporting one. A genuine, reviewed change in behaviour is recorded by
regenerating the golden from a fresh run, never by hand-editing the file this
suite reads.

**A golden mismatch and an invariant violation are different kinds of
failure**, and are kept in separate tests so a reader sees which one fired
without opening a traceback: `test_matches_golden` reports drift against a
past run, which may be an intended change; `test_invariants_hold` reports a
violation of a property that must hold of any result, whatever the goldens
say (`mermin_corpus.invariants`, phase 4's other pillar).

`_entry_artefact_path` and `MANIFEST_PATH` come from `tests/conftest.py`,
shared with `test_corpus_ingest.py` and `test_corpus_segmentation.py` for the
reason given there: `tests/` has no `__init__.py`, so a bare `pytest tests/`
run cannot import a sibling test module by its dotted path.

Unlike those two modules, this one is not corpus-only: the acceptance test
below (`test_comparator_reports_...`) operates on a golden file and an
in-memory copy, needs no drive and no `mermin` import, and must run in CI,
where the corpus volume is never mounted. So the drive requirement is not a
module-level `require_corpus()` skip (that would skip this file's CI-bound
tests too); it is a plain `pytest.skip` inside the `analysed` fixture, which
only the two `corpus`-marked, golden-backed tests below depend on.

`mermin` and `mermin_corpus.generate`/`.goldens`/`.invariants` are all
imported at module level regardless: `mermin/__init__.py` is a lazy
`__getattr__` shim (no heavy submodule import until an attribute like
`analyze` is actually touched), and `mermin_corpus.generate` pulls in only
`tomlkit` (via `.manifest`) at import time, not `bioio` or `mermin` itself.
Neither costs anything to import on a machine with no corpus volume, and
`mermin.analyze` is never actually called except inside the `analysed`
fixture, which the `corpus`-marked tests alone depend on.
"""
from __future__ import annotations

import copy
import json
import warnings
from typing import Any

import pytest

from conftest import MANIFEST_PATH, _entry_artefact_path

import mermin

from mermin_corpus.generate import GOLDEN_ENTRIES, _environment, golden_path
from mermin_corpus.goldens import LOOSE, TIGHT, build_record, compare
from mermin_corpus.invariants import check_invariants

# The eight entries the phase 4 corpus survey found reachable through
# `analyze()` under the deterministic threshold backend. `GOLDEN_ENTRIES` (the
# generator's own table, imported rather than re-typed here) is the single
# place that list is written down; this suite parametrises over its keys
# rather than carrying a second copy that could drift from it.
ENTRY_IDS = sorted(GOLDEN_ENTRIES)


def _load_golden(entry_id: str) -> dict[str, Any]:
    return json.loads(golden_path(entry_id).read_text())


@pytest.fixture(scope="session")
def analysed():
    """A cache of `(golden, result)` keyed by entry id, shared across every
    test in the session.

    `montano-hvf-2026-04` alone is 1495 cells and 12264 defects; running
    `analyze()` on it once per test rather than once per session would be
    the slow way to find that out. `test_matches_golden` and
    `test_invariants_hold` both draw from this cache, so each entry is
    analysed at most once per `pytest` invocation regardless of which, or
    how many, tests touch it.

    The invocation is read from the golden's own recorded `invocation`
    block, not from `GOLDEN_ENTRIES["kwargs"]`: the golden on disk is the
    source of truth for how it was produced, and this way a golden always
    documents the exact call that can reproduce it.

    Skips (not module-level: see the module docstring) unless the ASF-EX1
    corpus manifest is mounted, since every caller of this fixture needs the
    drive.
    """
    if not MANIFEST_PATH.exists():
        pytest.skip(f"{MANIFEST_PATH} not mounted; corpus-backed suite skipped")

    cache: dict[str, tuple[dict[str, Any], Any]] = {}

    def _get(entry_id: str) -> tuple[dict[str, Any], Any]:
        if entry_id not in cache:
            golden = _load_golden(entry_id)
            invocation = golden["invocation"]
            path = _entry_artefact_path(entry_id)
            result = mermin.analyze(
                path,
                segmentation=invocation["segmentation"],
                pixel_size_um=invocation["pixel_size_um"],
                channels=invocation["channels"],
            )
            cache[entry_id] = (golden, result)
        return cache[entry_id]

    return _get


@pytest.mark.corpus
@pytest.mark.parametrize("entry_id", ENTRY_IDS)
def test_matches_golden(entry_id, analysed):
    """`analyze()` on this entry, today, must reproduce its committed
    golden.

    Every `Difference` whose `kind` is `environment` is reported through
    `warnings.warn` rather than failing: a recorded dependency version
    explaining a difference elsewhere is not itself a regression. Every
    other difference fails the test, with the full list of them as the
    assertion message so a developer sees every field path, both values and
    the tolerance in one read, not just the first mismatch.
    """
    golden, result = analysed(entry_id)
    current = build_record(
        result, entry=entry_id, invocation=golden["invocation"], environment=_environment()
    )
    diffs = compare(golden, current)

    environmental = [d for d in diffs if d.kind == "environment"]
    for d in environmental:
        warnings.warn(f"{entry_id}: {d}")

    real = [d for d in diffs if d.kind != "environment"]
    assert not real, f"{entry_id}: {len(real)} golden mismatch(es):\n" + "\n".join(
        f"  {d}" for d in real
    )


@pytest.mark.corpus
@pytest.mark.parametrize("entry_id", ENTRY_IDS)
def test_invariants_hold(entry_id, analysed):
    """Properties that must hold of `analyze()`'s result on this entry
    whatever the golden says. A failure here is a defect in the code, not a
    drift to review against a past run; that is why it is a separate test
    from `test_matches_golden` rather than folded into it.
    """
    _golden, result = analysed(entry_id)
    violations = check_invariants(result)
    assert not violations, f"{entry_id}: {len(violations)} invariant violation(s):\n" + "\n".join(
        f"  {v.check}: {v.detail}" for v in violations
    )


# --- the acceptance test: a comparator that cannot fail certifies whatever
# --- it is pointed at, so this must demonstrate that it can. It reads a real
# --- golden file and a modified in-memory copy, touching neither the drive
# --- nor `analyze()`, so it is not marked `corpus` and runs in CI. ---


def _phantom_uniform_golden() -> dict[str, Any]:
    return json.loads(golden_path("phantom-uniform").read_text())


def test_comparator_reports_an_exact_field_perturbation():
    golden = _phantom_uniform_golden()
    current = copy.deepcopy(golden)
    current["counts"]["cells"] = golden["counts"]["cells"] + 1

    diffs = compare(golden, current)

    matches = [d for d in diffs if d.path == "counts.cells"]
    assert len(matches) == 1, diffs
    assert matches[0].kind == "exact"
    assert matches[0].golden == golden["counts"]["cells"]
    assert matches[0].current == current["counts"]["cells"]


def test_comparator_reports_a_tight_numeric_perturbation():
    # `frank` is pinned tight (TIGHT = 1e-12): it comes from the structure
    # tensor of the fibre plane alone and never touches a segmentation mask,
    # so it reproduces to the last bit. A change many orders of magnitude
    # above TIGHT must be reported.
    golden = _phantom_uniform_golden()
    current = copy.deepcopy(golden)
    original = golden["numerics"]["frank"]["bend"]
    current["numerics"]["frank"]["bend"] = original * (1 + 1e4 * TIGHT)

    diffs = compare(golden, current)

    matches = [d for d in diffs if d.path == "numerics.frank.bend"]
    assert len(matches) == 1, diffs
    assert matches[0].kind == "tolerance"
    assert matches[0].tolerance == TIGHT


def test_comparator_reports_a_loose_numeric_perturbation_outside_tolerance():
    # `cells` derives from contours and centroids, so it does touch the mask
    # (LOOSE = 1e-3, the tolerance that absorbs a boundary-slivers change in
    # watershed tie-breaking between skimage versions). A change well outside
    # that tolerance is still a real difference and must be reported.
    golden = _phantom_uniform_golden()
    current = copy.deepcopy(golden)
    original = golden["numerics"]["cells"]["area"]["mean"]
    current["numerics"]["cells"]["area"]["mean"] = original * (1 + 10 * LOOSE)

    diffs = compare(golden, current)

    matches = [d for d in diffs if d.path == "numerics.cells.area.mean"]
    assert len(matches) == 1, diffs
    assert matches[0].kind == "tolerance"
    assert matches[0].tolerance == LOOSE


def test_comparator_reports_no_difference_within_the_loose_tolerance():
    # The other half of the same claim: a mask-dependent field moving by
    # less than LOOSE is exactly the boundary-sliver noise the tolerance
    # exists to absorb, so it must report nothing. Without this test, a
    # LOOSE tolerance that silently ignored everything (a comparator that
    # never checks `cells` at all) would look identical to the test above.
    golden = _phantom_uniform_golden()
    current = copy.deepcopy(golden)
    original = golden["numerics"]["cells"]["area"]["mean"]
    current["numerics"]["cells"]["area"]["mean"] = original * (1 + 0.1 * LOOSE)

    diffs = compare(golden, current)

    assert not any(d.path == "numerics.cells.area.mean" for d in diffs)
