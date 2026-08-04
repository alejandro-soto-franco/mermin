"""Segmentation backends.

Nuclear segmentation sits behind a protocol so the pipeline can run with or
without cellpose installed, and so a caller can tell afterwards which
implementation produced a mask. `ThresholdBackend` needs nothing beyond scipy
and scikit-image, which mermin already requires, and is deterministic.
"""

import importlib.metadata
import importlib.util
import warnings
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
from scipy import ndimage
from skimage import feature, filters, segmentation

from mermin.errors import BackendUnavailableError, SegmentationError

# Bumped whenever the threshold algorithm changes in a way that moves a mask.
# It is part of the backend version, so it takes part in the cache key and an
# algorithm change invalidates stored masks rather than silently reusing them.
THRESHOLD_ALGORITHM_VERSION = "1"


@runtime_checkable
class SegmentationBackend(Protocol):
    """One nuclear segmenter.

    `config` returns everything that changes the mask and `version` identifies
    the implementation those settings select, so the pair keys a cache entry.
    """

    name: str

    def version(self) -> str: ...

    def config(self) -> dict[str, Any]: ...

    def segment(self, plane: np.ndarray) -> np.ndarray: ...


def _check_plane(plane: np.ndarray) -> np.ndarray:
    plane = np.asarray(plane, dtype=np.float64)
    if plane.ndim != 2:
        raise SegmentationError(
            f"segmentation takes one 2D plane, got shape {plane.shape}"
        )
    return plane


def _remove_small_objects(foreground: np.ndarray, min_size: int) -> np.ndarray:
    """Drop connected components holding fewer than `min_size` pixels.

    Written out rather than taken from `skimage.morphology`, whose
    `remove_small_objects` moved this boundary in 0.26: `min_size` is rewritten
    to `max_size` and compared with `<=`, so a component of exactly `min_size`
    pixels is kept on 0.25 and dropped on 0.26, and `max_size` does not exist
    before 0.26 at all. This backend's masks carry goldens, so they cannot move
    when a dependency is upgraded.
    """
    labelled, _count = ndimage.label(foreground)
    sizes = np.bincount(labelled.ravel())
    keep = sizes >= min_size
    keep[0] = False
    return keep[labelled]


@dataclass
class ThresholdBackend:
    """Gaussian smoothing, an Otsu threshold and a marker-controlled watershed.

    Deterministic: the same plane gives a byte-identical mask on every run,
    which is what lets goldens and the byte-identity suite hold. Seeds are
    sorted by row and then column before labels are assigned, so numbering does
    not depend on the order `peak_local_max` happens to return.
    """

    sigma: float = 2.0
    min_size: int = 25
    min_distance: int = 5

    name = "threshold"

    def version(self) -> str:
        return f"threshold-{THRESHOLD_ALGORITHM_VERSION}"

    def config(self) -> dict[str, Any]:
        return {
            "sigma": self.sigma,
            "min_size": self.min_size,
            "min_distance": self.min_distance,
        }

    def segment(self, plane: np.ndarray) -> np.ndarray:
        plane = _check_plane(plane)
        empty = np.zeros(plane.shape, dtype=np.int32)

        smoothed = ndimage.gaussian_filter(plane, self.sigma)
        if smoothed.max() <= smoothed.min():
            return empty

        foreground = smoothed > filters.threshold_otsu(smoothed)
        foreground = _remove_small_objects(foreground, min_size=self.min_size)
        if not foreground.any():
            return empty

        distance = ndimage.distance_transform_edt(foreground)
        peaks = feature.peak_local_max(
            distance, min_distance=self.min_distance, labels=foreground
        )
        if len(peaks) == 0:
            return empty

        markers = np.zeros(plane.shape, dtype=np.int32)
        for index, (row, col) in enumerate(sorted(map(tuple, peaks)), start=1):
            markers[row, col] = index

        labels = segmentation.watershed(-distance, markers, mask=foreground)
        return labels.astype(np.int32)


MIN_CELLPOSE_MAJOR = 4


def cellpose_version() -> str | None:
    """The installed cellpose version, or None when it is absent.

    Answered from distribution metadata rather than by importing cellpose,
    which would pull torch and cost seconds on a call that only selects a
    backend.
    """
    if importlib.util.find_spec("cellpose") is None:
        return None
    try:
        return importlib.metadata.version("cellpose")
    except importlib.metadata.PackageNotFoundError:
        return None


def _major(version: str) -> int:
    try:
        return int(version.split(".")[0])
    except ValueError:
        return -1


@dataclass
class CellposeBackend:
    """Cellpose 4, the Cellpose-SAM release.

    Version 3 is not supported through a second code path. Its model class,
    `models.Cellpose`, does not exist in version 4, its `eval` returned four
    values where version 4 returns three, and its `channels` argument is now
    ignored. One API, and a stated error for anything else.
    """

    pretrained_model: str = "cpsam_v2"
    diameter: float | None = None
    gpu: bool = False
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    min_size: int = 15

    name = "cellpose"

    def __post_init__(self) -> None:
        installed = cellpose_version()
        if installed is None:
            raise BackendUnavailableError(
                "the cellpose backend needs cellpose, which is not installed. "
                "Install it with the optional extra: mermin[cellpose]. "
                "The threshold backend needs no extra and is always available."
            )
        if _major(installed) < MIN_CELLPOSE_MAJOR:
            raise BackendUnavailableError(
                f"the cellpose backend needs cellpose {MIN_CELLPOSE_MAJOR} or "
                f"newer; cellpose {installed} is installed. Version 3 has no "
                "models.CellposeModel and its eval returns a different number "
                "of values."
            )
        self._installed_version = installed

    def version(self) -> str:
        return f"cellpose-{self._installed_version}"

    def config(self) -> dict[str, Any]:
        return {
            "pretrained_model": self.pretrained_model,
            "diameter": self.diameter,
            "gpu": self.gpu,
            "flow_threshold": self.flow_threshold,
            "cellprob_threshold": self.cellprob_threshold,
            "min_size": self.min_size,
        }

    def segment(self, plane: np.ndarray) -> np.ndarray:
        plane = _check_plane(plane)

        # Imported here, never at module scope: this line pulls torch.
        from cellpose import models

        model = models.CellposeModel(
            gpu=self.gpu, pretrained_model=self.pretrained_model
        )
        # Three values, not four. `channels` is ignored by version 4, so it is
        # not passed, and a bare 2D array is converted to three channels by
        # cellpose itself.
        masks, _flows, _styles = model.eval(
            plane,
            diameter=self.diameter,
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
            min_size=self.min_size,
        )
        return np.asarray(masks, dtype=np.int32)


def _looks_like_a_backend(spec: Any) -> bool:
    return all(
        hasattr(spec, attribute)
        for attribute in ("name", "version", "config", "segment")
    )


def resolve_backend(spec: Any = "auto") -> tuple[SegmentationBackend, str]:
    """Turn a backend specification into a backend and the mechanism that chose it.

    Mirrors the role resolver: a weaker mechanism is permitted, is warned
    about, and is recorded, so a caller can tell a chosen backend from a
    fallback after the fact.
    """
    if isinstance(spec, str):
        if spec == "threshold":
            return ThresholdBackend(), "explicit"
        if spec == "cellpose":
            return CellposeBackend(), "explicit"
        if spec == "auto":
            installed = cellpose_version()
            if installed is not None and _major(installed) >= MIN_CELLPOSE_MAJOR:
                return CellposeBackend(), "auto"
            reason = (
                "cellpose is not installed"
                if installed is None
                else f"cellpose {installed} is older than {MIN_CELLPOSE_MAJOR}.0"
            )
            warnings.warn(
                f"segmentation backend 'auto' selected 'threshold' because "
                f"{reason}. Install the cellpose extra, mermin[cellpose], to "
                "use it, or pass segmentation='threshold' to silence this.",
                RuntimeWarning,
                stacklevel=2,
            )
            return ThresholdBackend(), "auto-fallback"
        raise SegmentationError(
            f"unknown segmentation backend {spec!r}; expected 'auto', "
            "'cellpose', 'threshold', or a SegmentationBackend instance"
        )
    if _looks_like_a_backend(spec):
        return spec, "instance"
    raise SegmentationError(
        f"segmentation must be 'auto', 'cellpose', 'threshold', or a "
        f"SegmentationBackend instance, got {type(spec).__name__}"
    )
