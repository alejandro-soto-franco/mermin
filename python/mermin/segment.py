"""Cell segmentation: Cellpose for nuclei, watershed for cell bodies."""

import numpy as np
from scipy import ndimage
from skimage import measure, segmentation, morphology


def segment_nuclei(dapi: np.ndarray, cellpose_model: str = "nuclei", diameter: float | None = None):
    """Segment nuclei from DAPI channel using Cellpose.

    Args:
        dapi: 2D array, normalized DAPI channel.
        cellpose_model: Cellpose model name.
        diameter: Expected nuclear diameter in pixels. None for auto-detect.

    Returns:
        2D integer array: instance segmentation mask (0 = background).
    """
    from cellpose import models

    model = models.Cellpose(model_type=cellpose_model, gpu=False)
    masks, _, _, _ = model.eval([dapi], diameter=diameter, channels=[0, 0])
    return masks[0].astype(np.int32)


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
    foreground = morphology.binary_closing(foreground, morphology.disk(3))

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
