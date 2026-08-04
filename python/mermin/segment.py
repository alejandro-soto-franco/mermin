"""Cell segmentation: a backend protocol for nuclei, watershed for cell bodies."""

from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage
from skimage import measure, segmentation, morphology

from mermin.backends import SegmentationBackend, resolve_backend
from mermin.maskcache import load_mask, mask_cache_key, store_mask


def segment_nuclei_with_provenance(
    plane: np.ndarray,
    backend: str | SegmentationBackend = "auto",
    *,
    mask_cache: str | Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Segment nuclei, returning the mask and a record of how it was produced.

    Args:
        plane: 2D array, the normalised nuclear channel.
        backend: "auto", "cellpose", "threshold", or a SegmentationBackend.
        mask_cache: Directory for cached masks. None writes nothing.

    Returns:
        (mask, provenance) where mask is a 2D int32 instance mask with 0 as
        background, and provenance names the backend, its version, its
        configuration, the mechanism that selected it, the cache outcome and
        the nucleus count.
    """
    resolved, mechanism = resolve_backend(backend)

    outcome = "disabled"
    mask = None
    key = None
    if mask_cache is not None:
        key = mask_cache_key(plane, resolved)
        mask = load_mask(mask_cache, key)
        outcome = "hit" if mask is not None else "miss"

    if mask is None:
        mask = resolved.segment(plane)
        if mask_cache is not None and key is not None:
            store_mask(mask_cache, key, mask)

    provenance = {
        "backend": resolved.name,
        "version": resolved.version(),
        "config": resolved.config(),
        "mechanism": mechanism,
        "cache": outcome,
        "n_nuclei": int(np.count_nonzero(np.unique(mask))),
    }
    return mask, provenance


def segment_nuclei(
    plane: np.ndarray,
    backend: str | SegmentationBackend = "auto",
    *,
    mask_cache: str | Path | None = None,
) -> np.ndarray:
    """Segment nuclei from the nuclear channel. See
    `segment_nuclei_with_provenance` for the record of how the mask was made.
    """
    return segment_nuclei_with_provenance(plane, backend, mask_cache=mask_cache)[0]


def segment_cell_bodies(
    vimentin: np.ndarray,
    nuclear_mask: np.ndarray,
) -> np.ndarray:
    """Segment cell bodies using marker-controlled watershed on vimentin.

    Seeds are the nuclear centroids. Energy landscape is the inverted
    distance transform of the thresholded vimentin channel.

    Args:
        vimentin: 2D array, normalized vimentin channel.
        nuclear_mask: 2D integer array from segment_nuclei.

    Returns:
        2D integer array: cell body instance mask (same labels as nuclear_mask).
    """
    markers = nuclear_mask.copy()

    # Energy landscape: inverted vimentin intensity
    thresh = np.percentile(vimentin[vimentin > 0], 20) if np.any(vimentin > 0) else 0.1
    foreground = vimentin > thresh
    # `closing`, not the deprecated `binary_closing`: `mode="ignore"` is
    # load-bearing, not cosmetic. `binary_closing` treats pixels outside the
    # image as True for its erosion half and False for its dilation half
    # (its only mode, added in 0.23); `closing`'s default border mode is
    # "reflect", which mirrors nearby data instead. `mode="ignore"` reproduces
    # `binary_closing`'s fixed border exactly (confirmed against the 0.26
    # source: both convert "ignore" to the same fixed border value), so
    # foreground touching the edge of the frame closes the same way it always
    # did. See tests/test_segment_cell_bodies.py for the border-touching
    # fixture that pins this.
    foreground = morphology.closing(foreground, morphology.disk(3), mode="ignore")

    distance = ndimage.distance_transform_edt(foreground)
    energy = -distance

    cell_mask = segmentation.watershed(energy, markers=markers, mask=foreground)
    return cell_mask.astype(np.int32)


def extract_contours(cell_mask: np.ndarray, pixel_size: float = 1.0):
    """Extract boundary contours from a cell body mask.

    Args:
        cell_mask: 2D integer array from segment_cell_bodies.
        pixel_size: Physical size of one pixel in um.

    Returns:
        dict mapping label -> Nx2 numpy array of boundary points in physical units.
    """
    labels = np.unique(cell_mask)
    labels = labels[labels > 0]

    contours = {}
    for label in labels:
        binary = (cell_mask == label).astype(np.uint8)
        found = measure.find_contours(binary, 0.5)
        if found:
            longest = max(found, key=len)
            # Convert (row, col) to (x, y) in physical units
            contours[int(label)] = longest[:, ::-1] * pixel_size

    return contours


def build_neighbor_graph(centroids: np.ndarray):
    """Build Delaunay triangulation neighbor graph from cell centroids.

    Args:
        centroids: Nx2 array of (x, y) centroid positions.

    Returns:
        (triangulation, adjacency) where adjacency is a dict mapping
        cell_index -> set of neighbor indices.
    """
    from scipy.spatial import Delaunay

    tri = Delaunay(centroids)
    adjacency: dict[int, set[int]] = {i: set() for i in range(len(centroids))}

    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                adjacency[simplex[i]].add(simplex[j])
                adjacency[simplex[j]].add(simplex[i])

    return tri, adjacency
