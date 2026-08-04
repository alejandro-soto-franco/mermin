"""Invariants that must hold over any `AnalysisResult`, independent of the
golden records.

A golden mismatch may be an intended change to the analysis; a violation of
one of these is a defect in the code, whatever the goldens say. Each check
below traces to a fact established by a survey of the corpus before this plan
was written, not to a property inferred from a golden itself.

Like `goldens.py`, this module never imports `mermin`: it reads an
`AnalysisResult` by attribute, so it can run with no drive, no built Rust
extension, and no `mermin` on the path. `check_invariants` reports every
violation it finds rather than raising on the first, so a caller sees the
whole picture in one pass. It does not defend against a malformed result
(a missing attribute or key raises `AttributeError`/`KeyError` as normal
attribute access would); that is a programming error in the caller, not an
invariant to report on.
"""

from dataclasses import dataclass
from typing import Any

import math

#: Absolute tolerance for comparing a defect charge to the nearest multiple of
#: `1/k`. Charges arrive as floats, so an exact comparison would reject a
#: value that is correct up to floating-point rounding.
CHARGE_TOLERANCE = 1e-9

#: The smallest area observed anywhere in the corpus was 0.5 square pixels;
#: 0.25 sits below every real value and above zero, so a degenerate contour
#: surfaces here rather than propagating into the Landau-de Gennes fit.
MIN_AREA = 0.25


@dataclass(frozen=True)
class Violation:
    check: str
    detail: str


def _check_coherence_range(result: Any) -> Violation | None:
    """`fields["coherence"]` is `|psi_k|` by construction, so it must lie in
    [0, 1]. A value outside that range means the orientation computation
    itself is wrong."""
    coherence = result.fields["coherence"]
    low = float(coherence.min())
    high = float(coherence.max())
    if low < 0.0 or high > 1.0:
        return Violation(
            "coherence_range",
            f"coherence must lie in [0, 1]; observed range [{low!r}, {high!r}]",
        )
    return None


def _check_area_floor(result: Any) -> Violation | None:
    """A cell area of zero pixels, in this data, does not exist: 0.25 square
    pixels is a floor a degenerate contour must fail rather than sneak
    through."""
    areas = [float(a) for a in result.cells["area"]]
    offenders = [a for a in areas if a < MIN_AREA]
    if offenders:
        return Violation(
            "area_floor",
            f"{len(offenders)} of {len(areas)} cell areas fall below the "
            f"{MIN_AREA!r} square pixel floor; smallest observed "
            f"{min(offenders)!r}",
        )
    return None


def _check_cell_count_matches_nuclei(result: Any) -> Violation | None:
    """A nucleus whose watershed catchment came out empty would otherwise
    vanish from `result.cells` with no error, so the two counts must agree
    exactly."""
    n_nuclei = int(result.segmentation["n_nuclei"])
    n_cells = len(result.cells)
    if n_cells != n_nuclei:
        return Violation(
            "cell_count_matches_nuclei",
            f"segmentation found {n_nuclei!r} nuclei but {n_cells!r} became "
            "cells",
        )
    return None


def _check_correlation_length_never_nan(result: Any) -> Violation | None:
    """Infinity is a legitimate correlation length for a uniform director
    field. NaN is not a result at all."""
    value = float(result.correlations["correlation_length"])
    if math.isnan(value):
        return Violation(
            "correlation_length_finite_or_infinite",
            "correlation_length is NaN, which is not a defined length",
        )
    return None


def _check_defect_charges_are_multiples_of_1_over_k(
    result: Any, k: int
) -> Violation | None:
    """A holonomy-derived defect charge cannot take any value other than a
    multiple of `1/k`."""
    unit = 1.0 / k
    offenders = []
    for defect in result.defects:
        charge = float(defect["charge"])
        nearest = round(charge / unit)
        if abs(charge - nearest * unit) > CHARGE_TOLERANCE:
            offenders.append(charge)
    if offenders:
        return Violation(
            "defect_charge_multiple_of_1_over_k",
            f"{len(offenders)} of {len(result.defects)} defect charges are "
            f"not a multiple of 1/{k!r} within tolerance "
            f"{CHARGE_TOLERANCE!r}: {offenders!r}",
        )
    return None


def check_invariants(result: Any, *, k: int = 2) -> list[Violation]:
    """Every way `result` violates an invariant that must hold whatever the
    goldens say.

    A result with no cells has nothing for the area floor or the cell/nucleus
    count to check, and a result with no defects has nothing for the charge
    check to check: both are vacuously satisfied rather than treated as
    violations, since an empty result is not itself a defect.
    """
    checks = (
        _check_coherence_range(result),
        _check_area_floor(result),
        _check_cell_count_matches_nuclei(result),
        _check_correlation_length_never_nan(result),
        _check_defect_charges_are_multiples_of_1_over_k(result, k),
    )
    return [violation for violation in checks if violation is not None]
