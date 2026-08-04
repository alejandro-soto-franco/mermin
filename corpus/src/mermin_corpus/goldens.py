"""Golden records over the corpus, and the comparator that reads them.

This module never imports `mermin`. It reads an `AnalysisResult` by attribute,
so a golden can be built and compared without the analysis package installed,
and so the corpus tooling stays independent of the code it measures.
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
# The per-cell aggregates, `correlations` and `ldg_params` derive from contours
# and centroids, so they do touch the mask, and
# `skimage.segmentation.watershed` changed its boundary tie-breaking between
# 0.24 and 0.25: 5 pixels in 65536 moved on a phantom and 31 in 65536 on real
# IDR data, always boundary slivers between adjacent nuclei. A loose tolerance
# absorbs that without hiding a real change.
TIGHT = 1e-12
LOOSE = 1e-3

_CELL_COLUMNS = ("area", "perimeter", "shape_index", "elongation")


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

    cells: dict[str, Any] = {}
    for column in _CELL_COLUMNS:
        if column in result.cells.columns:
            cells[column] = _summarise(result.cells, column)

    r_bins = result.correlations.get("r_bins", [])
    g_values = result.correlations.get("g_values", [])

    return {
        "schema_version": SCHEMA_VERSION,
        "entry": entry,
        # The column set of `cells`, not its summarised contents: if a column
        # disappears from the pipeline's output on both sides, no golden built
        # from _CELL_COLUMNS alone would ever notice. This is an unrecognised
        # top-level section as far as `compare` is concerned, so it is already
        # compared exactly with no comparator change required.
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
                "optimal_sigma": float(result.fields["optimal_sigma"]),
                "theta_mean": float(theta.mean()),
                "theta_std": float(theta.std()),
                "coherence_mean": float(coherence.mean()),
                "coherence_std": float(coherence.std()),
            },
            "correlations": {
                "correlation_length": float(result.correlations["correlation_length"]),
                "n_bins": len(r_bins),
                # The curve itself, not only its length: `n_bins` and
                # `correlation_length` can both hold steady while `G_k(r)`
                # changes shape. `analyze()` takes the `len(labels) < 3`
                # fallback with both lists empty, in which case these are
                # `None` rather than an aggregate of nothing.
                "g_values": {
                    "min": float(min(g_values)),
                    "max": float(max(g_values)),
                    "mean": float(sum(g_values) / len(g_values)),
                } if g_values else None,
                "r_bins": {
                    "first": float(r_bins[0]),
                    "last": float(r_bins[-1]),
                } if r_bins else None,
            },
            "ldg_params": {k: float(v) for k, v in result.ldg_params.items()},
            "cells": cells,
        },
    }


_MISSING = object()


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
# being silently unchecked.
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
