"""Golden records over the corpus, and the comparator that reads them.

This module never imports `mermin`. It reads an `AnalysisResult` by attribute,
so a golden can be built and compared without the analysis package installed,
and so the corpus tooling stays independent of the code it measures.

`numerics.fields` summarises `theta`, `coherence` and `optimal_sigma` with a
mean, a standard deviation and a mean over each of the field's four spatial
quadrants (`_quadrant_means`). A per-quadrant mean was chosen over two other
options considered: a hash of the raw array is exact and would catch any
rearrangement, but a hash tells a reader nothing about what moved and is
brittle to last-bit floating-point noise, which this module otherwise treats
as absorbed rather than as a defect (see `TIGHT`/`LOOSE` below); dumping the
full field is exact and legible but unreadable as a diff and would bloat the
golden by orders of magnitude on the corpus's largest entry (4015x4015).
Quadrant means are cheap, round-trip through JSON as four ordinary floats,
and localise a change to a quarter of the image in a diff a person can read.
They also close a real gap: a full spatial reversal (`field[::-1, ::-1]`)
leaves the whole-field mean and standard deviation unchanged but swaps
opposite quadrants, so unless two opposite quadrant means coincide exactly,
the reversal now surfaces as a difference.
"""

from dataclasses import dataclass
from typing import Any

import math

SCHEMA_VERSION = "1"

# Relative tolerances. The split is measured rather than stylistic.
#
# `frank` and the field summaries come from the structure tensor of the fibre
# plane and never touch a segmentation mask, so they reproduce to the last bit
# on every supported scikit-image version. They are pinned tightly.
#
# The per-cell aggregates, `correlations` and `ldg_params` derive from
# contours and centroids, so they do touch the mask.
# `skimage.segmentation.watershed` changed its boundary tie-breaking between
# 0.24 and 0.25 (5 pixels in 65536 moved on a phantom, 31 in 65536 on real
# IDR data), which is why an earlier version of this constant was `1e-3`: the
# stated intent was to absorb that boundary drift across the declared
# scikit-image range.
#
# That intent does not survive measurement. Below the floor
# (`scikit-image>=0.25.2`, the lowest release the watershed change is stable
# at), the real cross-version change on these mask-dependent quantities is
# 6-40% relative -- two to three orders of magnitude past anything `1e-3`
# could absorb, so the old constant was never doing the job its comment
# claimed. At and above the floor, every reachable corpus entry reproduces
# its golden bit-exactly (0.25.2 and 0.26.0 measured, relative delta
# `0.000e+00` on every mask-dependent quantity, across all eight entries). A
# loose tolerance no longer has a version boundary to absorb, so what `LOOSE`
# actually guards against now is floating-point non-associativity across
# machines: a different BLAS/LAPACK build, thread count, or SIMD path
# reordering the same reduction. That risk is real but unmeasured on this
# box, unlike the version boundary it replaces. `1e-6` leaves six orders of
# magnitude of headroom over the floating-point noise this pipeline is known
# to produce elsewhere (a ~7.3e-12 `convexity` ceiling rounding artefact,
# measured on idr0062 and montano in the phase 4 corpus survey) while
# remaining far tighter than any real change measured here, so it no longer
# blinds the goldens to a change the way `1e-3` did.
TIGHT = 1e-12
LOOSE = 1e-6

# Cell-identifier columns excluded from the per-column numeric summary:
# `label` is an arbitrary watershed-assigned integer with no physical
# meaning of its own (its min/max/mean say nothing about the segmentation),
# and it is already pinned exactly by `schema` (present/absent) and by
# `counts.cells` (row count). Every other column `result.cells` carries is
# summarised, not a fixed subset: `build_record` reads the frame's own
# column list, so a new numeric column added to the pipeline is captured the
# moment it appears rather than needing this module taught its name.
_CELL_IDENTIFIER_COLUMNS = frozenset({"label"})


@dataclass(frozen=True)
class Difference:
    path: str
    golden: Any
    current: Any
    kind: str
    tolerance: float | None = None

    def __str__(self) -> str:
        if self.kind == "environment":
            return f"{self.path}: recorded {self.golden!r}, now {self.current!r}"
        return (
            f"{self.path}: expected {self.golden!r}, got {self.current!r}"
            + (f" (relative tolerance {self.tolerance})" if self.tolerance else "")
        )


def _quadrant_means(field: Any) -> dict[str, float]:
    """Mean of each of a 2D field's four spatial quadrants.

    See the module docstring for why quadrant means and not a hash or a full
    dump. Integer floor division splits an odd dimension so every pixel
    lands in exactly one quadrant; that is a deterministic, reproducible
    split, not a design requirement of the four sizes matching.
    """
    ny, nx = field.shape
    my, mx = ny // 2, nx // 2
    return {
        "top_left": float(field[:my, :mx].mean()),
        "top_right": float(field[:my, mx:].mean()),
        "bottom_left": float(field[my:, :mx].mean()),
        "bottom_right": float(field[my:, mx:].mean()),
    }


def _summarise(frame, column: str) -> dict[str, float] | None:
    """Min/max/mean of a column, or `None` for an empty frame.

    `polars.Series.min()` on an empty column returns `None`, not an empty
    aggregate, so `float(None)` would crash uninformatively on a corpus tile
    whose segmentation found nothing. `None` is returned explicitly instead,
    and `compare` treats it like any other value: two zero-cell records
    compare equal, a zero-cell record against a populated one differs.
    """
    if frame.height == 0:
        return None
    series = frame[column]
    return {
        "min": float(series.min()),
        "max": float(series.max()),
        "mean": float(series.mean()),
    }


def build_record(
    result: Any,
    *,
    entry: str,
    invocation: dict[str, Any],
    environment: dict[str, str],
) -> dict[str, Any]:
    """A golden record for one analysed entry."""
    theta = result.fields["theta"]
    coherence = result.fields["coherence"]
    optimal_sigma = result.fields["optimal_sigma"]

    cells: dict[str, Any] = {}
    for column in result.cells.columns:
        if column in _CELL_IDENTIFIER_COLUMNS:
            continue
        cells[column] = _summarise(result.cells, column)

    r_bins = result.correlations.get("r_bins", [])
    g_values = result.correlations.get("g_values", [])

    # Charges as detected, not as truth: `detect_defects` over-counts against
    # every phantom's own ground truth (see the phase 4 corpus survey), so
    # this is pinning known-current-but-wrong behaviour on purpose, the same
    # way `counts.defects` already does. Compared exactly, not under
    # `LOOSE`: a defect is present or it is not, and its charge is a
    # discrete, quantised quantity (a multiple of 1/k, checked separately by
    # `mermin_corpus.invariants`), so there is no boundary-pixel noise here
    # to absorb the way there is in a per-cell area or perimeter. Deliberately
    # left out of `_NUMERIC_TOLERANCE` below, exactly like `schema`, so it
    # flows through the exact-comparison path with no comparator change: a
    # sign flip on every charge, or a change to any one charge, must fail.
    charges = [float(d["charge"]) for d in result.defects]

    return {
        "schema_version": SCHEMA_VERSION,
        "entry": entry,
        # The column set of `cells`, not its summarised contents: if a column
        # disappears from the pipeline's output on both sides, no golden built
        # from the per-column summary loop alone would ever notice. This is
        # an unrecognised top-level section as far as `compare` is concerned,
        # so it is already compared exactly with no comparator change
        # required.
        "schema": sorted(result.cells.columns),
        "environment": dict(environment),
        "invocation": dict(invocation),
        "ingest": {
            "roles": {
                role: {"index": r["index"], "mechanism": r["mechanism"]}
                for role, r in result.ingest["roles"].items()
            },
            "pixel_size_um": result.ingest["pixel_size_um"],
            "projection": result.ingest["projection"],
        },
        "segmentation": {
            "backend": result.segmentation["backend"],
            "version": result.segmentation["version"],
            "config": dict(result.segmentation["config"]),
            "mechanism": result.segmentation["mechanism"],
        },
        "counts": {
            "nuclei": int(result.segmentation["n_nuclei"]),
            "cells": int(len(result.cells)),
            "defects": int(len(result.defects)),
        },
        "numerics": {
            "frank": {k: float(v) for k, v in result.frank.items()},
            "fields": {
                "shape": list(theta.shape),
                # `optimal_sigma` is a per-pixel field (mermin-orient's
                # multiscale structure tensor picks the coherence-maximising
                # scale at every pixel independently), the same shape as
                # `theta` and `coherence`, not a single scalar chosen once
                # for the whole image. It is summarised the same way they
                # are rather than cast through `float()`, which raised on
                # every real (non-`fake_result`) analysis: `float()` only
                # accepts a 0-dimensional array.
                "optimal_sigma_mean": float(optimal_sigma.mean()),
                "optimal_sigma_std": float(optimal_sigma.std()),
                # Mean and standard deviation alone are invariant under any
                # measure-preserving rearrangement of a field's pixels,
                # including a full spatial reversal (`field[::-1, ::-1]`) of
                # exactly the arrays `analyze()` hands the caller. Per-
                # quadrant means close that gap; see the module docstring
                # for why quadrant means rather than a hash or a full dump.
                "optimal_sigma_quadrants": _quadrant_means(optimal_sigma),
                "theta_mean": float(theta.mean()),
                "theta_std": float(theta.std()),
                "theta_quadrants": _quadrant_means(theta),
                "coherence_mean": float(coherence.mean()),
                "coherence_std": float(coherence.std()),
                "coherence_quadrants": _quadrant_means(coherence),
            },
            "correlations": {
                "correlation_length": float(result.correlations["correlation_length"]),
                "n_bins": len(r_bins),
                # The curve itself, not only its length: `n_bins` and
                # `correlation_length` can both hold steady while `G_k(r)`
                # changes shape. `analyze()` takes the `len(labels) < 3`
                # fallback with both lists empty, in which case these are
                # `None` rather than an aggregate of nothing.
                # `len(...)`, not bare truthiness: `_native.orientational_
                # correlation` returns `r_bins`/`g_values` as numpy arrays on
                # every real analysis (only the `len(labels) < 3` fallback in
                # `analyze()` uses plain empty lists), and `bool()` on a
                # multi-element ndarray raises rather than testing emptiness.
                "g_values": {
                    "min": float(min(g_values)),
                    "max": float(max(g_values)),
                    "mean": float(sum(g_values) / len(g_values)),
                } if len(g_values) else None,
                "r_bins": {
                    "first": float(r_bins[0]),
                    "last": float(r_bins[-1]),
                } if len(r_bins) else None,
            },
            "ldg_params": {k: float(v) for k, v in result.ldg_params.items()},
            "cells": cells,
            # Minimum content needed to catch a sign flip, a changed charge
            # value, or per-defect drift, none of which move `counts.defects`
            # (a count, not a charge). `charge_sum` is the quantity the
            # README states a number for (phantom-radial's 95 detections
            # summing to +4.5); `charge_min`/`charge_max` catch a change
            # confined to one end of the distribution that a sum alone could
            # still cancel out. `None`, not `0.0`, for min/max on zero
            # defects: a golden with no defects and a golden whose one
            # defect happens to have charge exactly 0 must not compare
            # equal. `charge_sum` stays `0.0` on zero defects: the sum of no
            # charges is genuinely zero, not missing information.
            "defects": {
                "charge_sum": float(sum(charges)),
                "charge_min": float(min(charges)) if charges else None,
                "charge_max": float(max(charges)) if charges else None,
            },
        },
    }


class _Missing:
    """Sentinel for an absent key, distinct from any value a record could
    legitimately hold (including `None`).

    A plain `object()` sentinel reprs as `<object object at 0x...>`, which is
    exactly what a caller sees in the message for a missing section or
    field: `Difference.__str__` formats both sides with `!r`, and that is
    the acceptance-critical path where a message most needs to be readable,
    not a raw memory address. `__repr__` overridden so `expected {...}, got
    <missing>` reads as what happened rather than an implementation detail.
    """

    def __repr__(self) -> str:
        return "<missing>"


_MISSING = _Missing()


def _close(golden: Any, current: Any, tolerance: float) -> bool:
    if isinstance(golden, bool) or isinstance(current, bool):
        return golden == current
    if not isinstance(golden, (int, float)) or not isinstance(current, (int, float)):
        return golden == current
    # NaN never compares equal, including to itself. A NaN where a number was
    # is a defect, so it must surface as a difference rather than pass.
    if math.isnan(golden) or math.isnan(current):
        return False
    if math.isinf(golden) or math.isinf(current):
        return golden == current
    return math.isclose(golden, current, rel_tol=tolerance, abs_tol=0.0)


def _walk(golden: Any, current: Any, path: str, kind: str, tolerance: float | None,
          out: list[Difference]) -> None:
    if isinstance(golden, dict):
        if not isinstance(current, dict):
            out.append(Difference(path, golden, current, kind, tolerance))
            return
        for key in sorted(set(golden) | set(current)):
            _walk(
                golden.get(key, _MISSING),
                current.get(key, _MISSING),
                f"{path}.{key}" if path else str(key),
                kind,
                tolerance,
                out,
            )
        return
    if golden is _MISSING or current is _MISSING:
        out.append(Difference(path, golden, current, kind, tolerance))
        return
    if tolerance is None:
        # An exact comparison also rejects a numeric type change even when the
        # values are equal by value (3 == 3.0): a golden's `int()` cast
        # regressing to a float is a schema defect, and JSON round-trips int
        # and float distinctly, so this costs nothing to check. The tolerance
        # path below is untouched: comparing an int to a float there is
        # legitimate, since both sides are cast through `float()`.
        if (
            kind == "exact"
            and isinstance(golden, (int, float))
            and isinstance(current, (int, float))
            and type(golden) is not type(current)
        ):
            out.append(Difference(path, golden, current, kind, None))
            return
        if golden != current:
            out.append(Difference(path, golden, current, kind, None))
        return
    if not _close(golden, current, tolerance):
        out.append(Difference(path, golden, current, kind, tolerance))


# Which tolerance each branch of `numerics` takes. Anything not named here is
# compared exactly, so a new field added to a record fails loudly rather than
# being silently unchecked. `defects` is deliberately absent: a detected
# charge is discrete and quantised, not a mask-boundary-sensitive statistic,
# so it takes the same exact path as an unrecognised section rather than a
# named tolerance.
_NUMERIC_TOLERANCE = {
    "frank": TIGHT,
    "fields": TIGHT,
    "correlations": LOOSE,
    "ldg_params": LOOSE,
    "cells": LOOSE,
}


_KNOWN_SECTIONS = {
    "environment", "numerics", "schema_version", "entry", "invocation",
    "ingest", "segmentation", "counts",
}


def compare(golden: dict[str, Any], current: dict[str, Any]) -> list[Difference]:
    """Every way `current` differs from `golden`.

    An `environment` difference is reported with kind `environment` so a caller
    can note it without failing: a recorded dependency version explains a
    difference elsewhere, it is not one itself.

    The top-level keys of both records are unioned, the same principle already
    applied inside `numerics`: a section neither known to this function nor
    present in the other record is a difference in its own right, so a section
    added to `build_record` without being taught to `compare` fails loudly
    rather than going unchecked from the moment it exists.
    """
    out: list[Difference] = []

    _walk(golden.get("environment", {}), current.get("environment", {}),
          "environment", "environment", None, out)

    for section in ("schema_version", "entry", "invocation", "ingest",
                    "segmentation", "counts"):
        _walk(golden.get(section, _MISSING), current.get(section, _MISSING),
              section, "exact", None, out)

    g_num = golden.get("numerics", {})
    c_num = current.get("numerics", {})
    if isinstance(g_num, dict) and isinstance(c_num, dict):
        for branch in sorted(set(g_num) | set(c_num)):
            _walk(
                g_num.get(branch, _MISSING),
                c_num.get(branch, _MISSING),
                f"numerics.{branch}",
                "tolerance",
                _NUMERIC_TOLERANCE.get(branch),
                out,
            )
    else:
        # A malformed record (`numerics` not a mapping, e.g. `None`) would
        # otherwise crash `set(g_num)` with an uninformative TypeError. Every
        # other type mismatch in this module produces a clean `Difference`;
        # this one should too.
        _walk(g_num, c_num, "numerics", "exact", None, out)

    for key in sorted((set(golden) | set(current)) - _KNOWN_SECTIONS):
        _walk(golden.get(key, _MISSING), current.get(key, _MISSING),
              key, "exact", None, out)

    return out
