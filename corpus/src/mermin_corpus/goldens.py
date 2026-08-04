"""Golden records over the corpus, and the comparator that reads them.

This module never imports `mermin`. It reads an `AnalysisResult` by attribute,
so a golden can be built and compared without the analysis package installed,
and so the corpus tooling stays independent of the code it measures. It does
import `numpy`, already a declared dependency of `mermin-corpus` itself (see
`corpus/pyproject.toml`) and, transitively, of `mermin` (a per-pixel field is
a numpy array), so this costs nothing new: the circular statistics below
need vectorised trigonometric ufuncs, and a pure-Python loop over a
4015x4015 field would cost tens of seconds it does not need to.

`numerics.fields` summarises `theta`, `coherence` and `optimal_sigma` with a
mean, a dispersion statistic and a statistic over each of the field's four
spatial quadrants. A per-quadrant statistic was chosen over two other
options considered: a hash of the raw array is exact and would catch any
rearrangement, but a hash tells a reader nothing about what moved and is
brittle to last-bit floating-point noise, which this module otherwise treats
as absorbed rather than as a defect (see `TIGHT`/`LOOSE` below); dumping the
full field is exact and legible but unreadable as a diff and would bloat the
golden by orders of magnitude on the corpus's largest entry (4015x4015).
Quadrant statistics are cheap, round-trip through JSON as four ordinary
floats, and localise a change to a quarter of the image in a diff a person
can read. They also close a real gap: a full spatial reversal
(`field[::-1, ::-1]`) leaves the whole-field mean and dispersion unchanged
but swaps opposite quadrants, so unless two opposite quadrants coincide
exactly, the reversal now surfaces as a difference.

`theta` is a director angle identified modulo pi, not an ordinary linear
quantity (`mermin-orient/src/structure_tensor.rs` wraps it into `[0, pi)`;
so do `mermin-shape/src/minkowski.rs`'s `elongation_angle` and
`mermin-orient/src/cell_orientation.rs`'s `nuclear_angle`, both orientations
of an axis that is only defined up to sign). An arithmetic mean of such an
angle is not merely fragile, it is the wrong statistic: averaging a value
just under pi with a value just over 0 -- physically the same orientation
-- produces something near pi/2. `_circular_mean_and_resultant` uses the
convention this project already establishes elsewhere
(`cell_orientations` in `cell_orientation.rs`, and
`tests/test_phantom_closed_form.py`'s `test_uniform_director_recovers_the_
known_angle`): double the angle before averaging as a unit complex number,
then halve the mean's argument back, `0.5 * angle(mean(exp(2j * theta)))`.
Doubling maps the pi-periodic identification onto an ordinary
2*pi-periodic one where an average is well-defined; halving maps back. Its
companion, the resultant length `abs(mean(exp(2j * theta)))`, replaces a
standard deviation for these quantities: 1.0 for a perfectly concentrated
angle, 0.0 for a uniformly spread one. This is used, not a purpose-built
circular standard deviation, for the same reason: one convention across the
project is worth more than a marginally more familiar number.
"""

from dataclasses import dataclass
from typing import Any

import math

import numpy as np

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
#
# An all-declared-floors run against this `1e-6` surfaced one field at
# 1.652809e-06, just outside it: `montano-hvf-2026-04`'s
# `numerics.cells.elongation_angle.mean`, ten orders of magnitude above
# every other field's floors-run delta (1e-15 to 1e-16, machine epsilon).
# Part of that was a wrong statistic rather than noise for `LOOSE` to
# absorb: a plain arithmetic mean of a modulo-pi angle jumps when a cell's
# angle crosses the wrap point, so `_circular_mean_and_resultant` replaced
# it for every angular quantity here (`theta`, `elongation_angle`,
# `nuclear_angle`). That collapsed seven of the eight entries to 1e-14 or
# below.
#
# It did not fix `montano-hvf-2026-04`, and the residual is worth stating
# rather than quietly absorbing. That entry's angular aggregates still move
# at all-floors, by 7.2e-6 on the mean and 5.1e-6 on the resultant length,
# which is larger than the pre-fix figure rather than smaller. The cause is
# conditioning rather than the formula: its 1495 cells have a resultant
# length of 0.22, meaning almost no preferred orientation, and the direction
# of a circular mean is poorly determined when the resultant is small. A
# summary that is barely defined cannot be pinned tightly, and no choice of
# tolerance makes it well defined.
#
# So `LOOSE` stays at `1e-6`, and that one entry's angular aggregates are
# expected to differ on a dependency set other than the one its golden
# records in `environment`. Loosening to `1e-5` to cover it would blind
# every other field, all of which reproduce to 1e-14, to a change ten
# thousand times larger than any they exhibit.
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

# `cells` columns holding an orientation angle identified modulo pi, not an
# ordinary linear quantity: confirmed from the Rust source, not assumed.
# `elongation_from_w1_tensor` (`mermin-shape/src/minkowski.rs`) documents
# `elongation_angle` as "orientation of major axis in `[0, pi)` radians" and
# wraps it there explicitly; `fit_nuclear_ellipse`
# (`mermin-orient/src/cell_orientation.rs`) documents `nuclear_angle`
# identically and wraps it the same way. Both are eigenvector/ellipse-axis
# orientations with no preferred sign, the same identification `theta`
# carries (see the module docstring), so both are summarised with
# `_circular_summarise` rather than `_summarise`.
_CELL_ANGULAR_COLUMNS = frozenset({"elongation_angle", "nuclear_angle"})


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


def _circular_mean_and_resultant(values: Any) -> tuple[float, float]:
    """Circular mean and resultant length of an angle identified modulo pi.

    See the module docstring for the doubling convention and why it is
    correct here (`theta`, `elongation_angle`, `nuclear_angle`). `values`
    must be non-empty; callers with a possibly-empty column or field handle
    that themselves; e.g. `_circular_summarise` short-circuits before this
    is called. The mean is wrapped into `[0, pi)` via Python's `%`, matching
    the domain every angular quantity is already stored in on the Rust
    side, so a golden's circular mean is directly comparable to the raw
    field or column it summarises.
    """
    z = complex(np.mean(np.exp(2j * np.asarray(values, dtype=np.float64))))
    mean = (0.5 * math.atan2(z.imag, z.real)) % math.pi
    resultant_length = abs(z)
    return float(mean), float(resultant_length)


def _quadrant_slices(field: Any) -> dict[str, Any]:
    """A 2D field's four spatial quadrants as array slices.

    Integer floor division splits an odd dimension so every pixel lands in
    exactly one quadrant; that is a deterministic, reproducible split, not
    a design requirement of the four sizes matching.
    """
    ny, nx = field.shape
    my, mx = ny // 2, nx // 2
    return {
        "top_left": field[:my, :mx],
        "top_right": field[:my, mx:],
        "bottom_left": field[my:, :mx],
        "bottom_right": field[my:, mx:],
    }


def _quadrant_means(field: Any) -> dict[str, float]:
    """Linear mean of each of a 2D field's four spatial quadrants.

    For `coherence` and `optimal_sigma` only: ordinary linear quantities.
    `theta`, identified modulo pi, needs `_circular_quadrant_means` instead,
    or the same wrap-around error the module docstring describes for the
    whole field shows up per quadrant. See the module docstring for why a
    quadrant statistic at all, rather than a hash or a full dump.
    """
    return {name: float(sub.mean()) for name, sub in _quadrant_slices(field).items()}


def _circular_quadrant_means(field: Any) -> dict[str, float]:
    """Circular mean (`_circular_mean_and_resultant`) of each of a 2D
    field's four spatial quadrants, for an angle identified modulo pi
    (`theta`)."""
    return {
        name: _circular_mean_and_resultant(sub.ravel())[0]
        for name, sub in _quadrant_slices(field).items()
    }


def _summarise(frame, column: str) -> dict[str, float] | None:
    """Min/max/mean of a linear column, or `None` for an empty frame.

    `polars.Series.min()` on an empty column returns `None`, not an empty
    aggregate, so `float(None)` would crash uninformatively on a corpus tile
    whose segmentation found nothing. `None` is returned explicitly instead,
    and `compare` treats it like any other value: two zero-cell records
    compare equal, a zero-cell record against a populated one differs.

    For an angular column (`_CELL_ANGULAR_COLUMNS`), use
    `_circular_summarise` instead: min/max/mean of a modulo-pi angle are
    not meaningful statistics, only the least unstable ones, and this
    module now has the correct ones available.
    """
    if frame.height == 0:
        return None
    series = frame[column]
    return {
        "min": float(series.min()),
        "max": float(series.max()),
        "mean": float(series.mean()),
    }


def _circular_summarise(frame, column: str) -> dict[str, float] | None:
    """Circular mean and resultant length of a `cells` column holding an
    angle identified modulo pi (`elongation_angle`, `nuclear_angle`), in
    place of the linear min/max/mean `_summarise` uses for every other
    column. `None` for an empty frame, for the same reason `_summarise`
    returns `None`: there is no aggregate of nothing, and `compare` treats
    `None` uniformly regardless of which function produced it.
    """
    if frame.height == 0:
        return None
    mean, resultant_length = _circular_mean_and_resultant(frame[column].to_numpy())
    return {"mean": mean, "resultant_length": resultant_length}


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

    # Computed once, not inline twice: `theta` is up to 4015x4015 (montano),
    # and `_circular_mean_and_resultant` does a full pass over it.
    theta_mean, theta_resultant_length = _circular_mean_and_resultant(theta.ravel())

    cells: dict[str, Any] = {}
    for column in result.cells.columns:
        if column in _CELL_IDENTIFIER_COLUMNS:
            continue
        if column in _CELL_ANGULAR_COLUMNS:
            cells[column] = _circular_summarise(result.cells, column)
        else:
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
                # `theta` is a director angle identified modulo pi (see the
                # module docstring), so its mean and dispersion use
                # `_circular_mean_and_resultant` rather than `theta.mean()`/
                # `theta.std()`: a plain arithmetic mean is not merely
                # fragile for this quantity, it is the wrong statistic, and
                # `theta_resultant_length` (1.0 concentrated, 0.0 spread)
                # replaces a standard deviation rather than inventing a
                # circular one.
                "theta_mean": theta_mean,
                "theta_resultant_length": theta_resultant_length,
                "theta_quadrants": _circular_quadrant_means(theta),
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
