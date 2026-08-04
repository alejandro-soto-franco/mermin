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

**`mermin_corpus.generate` is imported lazily, not at module level, for the
same reason.** `mermin_corpus.goldens` and `.invariants` (imported below,
unconditionally) touch nothing past the standard library. `.generate` is
different: it imports `.manifest`, which imports `tomlkit`, a
`mermin-corpus` dependency that is not part of the standard `mermin` test
environment and is absent from CI's `python-tests` and `floors` jobs. A
module-level `from mermin_corpus.generate import ...` on such a machine
raises `ModuleNotFoundError` during collection, before pytest has any chance
to skip: an error, not a skip, exactly the failure mode phase 1 already
ruled out for an unmounted drive. The `corpus_tooling` fixture below imports
it at test setup instead and skips with a stated reason if it is not
installed, so this file behaves the same way for an absent optional
dependency as it already does for an absent drive: `mermin` itself is
unaffected (`mermin/__init__.py` is a lazy `__getattr__` shim, so importing
it plainly costs nothing either).
"""
from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path
from typing import Any

import pytest

from conftest import MANIFEST_PATH, _entry_artefact_path

import mermin

from mermin_corpus.goldens import LOOSE, TIGHT, build_record, compare
from mermin_corpus.invariants import check_invariants

_GOLDENS_DIR = Path(__file__).resolve().parent.parent / "corpus" / "goldens"

# The entries currently pinned by a committed golden, read off disk rather
# than from `mermin_corpus.generate.GOLDEN_ENTRIES`. `@pytest.mark.parametrize`
# below runs at collection time, before any fixture gets a chance to skip, so
# this list must not need `tomlkit` (or anything else `generate` pulls in) to
# exist at all; the committed `corpus/goldens/*.json` files are already the
# source of truth for which entries are golden-able, so nothing is lost by
# reading the directory instead of the generator's table.
ENTRY_IDS = sorted(p.stem for p in _GOLDENS_DIR.glob("*.json"))


@pytest.fixture(scope="session")
def corpus_tooling():
    """`mermin_corpus.generate`, imported here rather than at module level.

    Skips with a stated reason if `tomlkit` (a `mermin-corpus` dependency,
    not a `mermin` one) is not installed, rather than letting collection
    fail with `ModuleNotFoundError`. See the module docstring.
    """
    try:
        import mermin_corpus.generate as generate
    except ImportError as exc:
        pytest.skip(
            f"mermin_corpus.generate unavailable ({exc}); corpus-backed "
            "suite skipped"
        )
    return generate


@pytest.fixture(scope="session")
def analysed(corpus_tooling):
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

    Depends on `corpus_tooling` first, so a missing `tomlkit` skips before
    the drive is even checked: the absent tooling, not the drive, is what
    gates these tests when both `test_matches_golden` and
    `test_invariants_hold` need `analyze()`'s output regardless of whether
    either individually needs `generate` past this point.
    """
    if not MANIFEST_PATH.exists():
        pytest.skip(f"{MANIFEST_PATH} not mounted; corpus-backed suite skipped")

    generate = corpus_tooling
    cache: dict[str, tuple[dict[str, Any], Any]] = {}

    def _get(entry_id: str) -> tuple[dict[str, Any], Any]:
        if entry_id not in cache:
            golden = json.loads(generate.golden_path(entry_id).read_text())
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
def test_matches_golden(entry_id, analysed, corpus_tooling):
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
        result,
        entry=entry_id,
        invocation=golden["invocation"],
        environment=corpus_tooling._environment(),
    )
    diffs = compare(golden, current)

    environmental = [d for d in diffs if d.kind == "environment"]
    for d in environmental:
        warnings.warn(f"{entry_id}: {d}")

    real = [d for d in diffs if d.kind != "environment"]
    assert not real, f"{entry_id}: {len(real)} golden mismatch(es):\n" + "\n".join(
        f"  {d}" for d in real
    )


@pytest.mark.parametrize("entry_id", ENTRY_IDS)
def test_golden_invocation_matches_the_generator_table(entry_id, corpus_tooling):
    """`test_matches_golden` reads `invocation` from the golden and passes
    that same dict to both `analyze()` and `build_record`, so its
    `invocation` section is compared against itself and can never fail: a
    golden hand-edited to disagree with `mermin_corpus.generate.GOLDEN_ENTRIES`
    (the table that is supposed to have produced it) would still pass there.

    This test closes that gap by deriving the expected `invocation` from
    `GOLDEN_ENTRIES` independently -- the same four-field shape
    `generate.build_entry_record` writes, reconstructed here rather than
    imported from it, since there is no pure function in `generate.py` that
    returns just the invocation without running a full `analyze()` -- and
    comparing it to what the committed golden actually records. Needs
    `corpus_tooling` (so `tomlkit`-absent still skips cleanly) but not the
    drive: `GOLDEN_ENTRIES` and the golden files are both already checked
    in, so this runs even when ASF-EX1 is not mounted.
    """
    generate = corpus_tooling
    golden = json.loads(generate.golden_path(entry_id).read_text())
    spec = generate.GOLDEN_ENTRIES[entry_id]
    expected_invocation = {
        "segmentation": "threshold",
        "pixel_size_um": spec["kwargs"].get("pixel_size_um"),
        "pixel_size_um_assumed": spec["pixel_size_um_assumed"],
        "channels": spec["kwargs"].get("channels"),
    }
    assert golden["invocation"] == expected_invocation


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
    # Reads straight off `_GOLDENS_DIR`, not through
    # `mermin_corpus.generate.golden_path`: this test must not need
    # `tomlkit` any more than `ENTRY_IDS` above does, since it is the whole
    # reason this suite is not corpus-only.
    return json.loads((_GOLDENS_DIR / "phantom-uniform.json").read_text())


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
