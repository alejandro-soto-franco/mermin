"""Segmentation backends.

Nuclear segmentation sits behind a protocol so the pipeline can run with or
without cellpose installed, and so a caller can tell afterwards which
implementation produced a mask. `ThresholdBackend` needs nothing beyond scipy
and scikit-image, which mermin already requires, and is deterministic.
"""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
from scipy import ndimage
from skimage import feature, filters, segmentation

from mermin.errors import SegmentationError

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
