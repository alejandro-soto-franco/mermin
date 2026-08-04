"""segment_cell_bodies: the closing step's border behaviour.

`skimage.morphology.binary_closing` is deprecated since scikit-image 0.26 and
removed in 0.28, in favour of `closing`. Its default border mode does not
match `binary_closing`'s, so `segment_cell_bodies` calls `closing(...,
mode="ignore")` to keep the border-touching pixels behaving the same way they
always did. This file pins that against a fixture with foreground touching
the frame edge, so a future migration cannot move it silently, in the same
spirit as `test_remove_small_objects_keeps_exactly_min_size_drops_one_less`
in test_backends.py.
"""

import numpy as np
import warnings

from skimage import morphology

from mermin.segment import segment_cell_bodies


def _watershed_input(foreground: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A vimentin plane and a matching single-nucleus marker for `foreground`.

    `segment_cell_bodies` thresholds `vimentin` at the 20th percentile of its
    own positive pixels, so a uniform intensity is unusable: the threshold
    equals every pixel's value and none of them clears a strict `>`. A low,
    uniform filler block (below the threshold) makes up more than 20% of the
    positive pixels so the percentile lands on the filler value, and
    `foreground` is set well above it, so `foreground` alone survives
    thresholding.
    """
    vimentin = np.zeros(foreground.shape, dtype=np.float64)
    vimentin[5:-5, 5:-5] = 0.5  # filler, below the threshold
    vimentin[foreground] = 5.0
    nuclear_mask = np.zeros(foreground.shape, dtype=np.int32)
    rows, cols = np.nonzero(foreground)
    nuclear_mask[rows[0], cols[0]] = 1
    return vimentin, nuclear_mask


def test_closing_border_mode_matches_binary_closing_on_border_touching_foreground():
    """The exact fixture from segment.py's call site: `disk(3)`, foreground
    touching every edge of the frame. Pins `closing(..., mode="ignore")`
    against the deprecated `binary_closing` it replaces, byte for byte."""
    foreground = np.zeros((40, 40), dtype=bool)
    foreground[19:21, :] = True  # touches the left and right borders
    foreground[:, 19:21] = True  # touches the top and bottom borders

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        expected = morphology.binary_closing(foreground, morphology.disk(3))
    actual = morphology.closing(foreground, morphology.disk(3), mode="ignore")

    assert actual.dtype == expected.dtype
    assert np.array_equal(actual, expected)


def test_segment_cell_bodies_does_not_warn_and_keeps_the_border_mask():
    """The end-to-end call: no FutureWarning, and the border-touching
    foreground survives closing rather than being eroded by the frame edge."""
    foreground = np.zeros((40, 40), dtype=bool)
    foreground[19:21, :] = True
    foreground[:, 19:21] = True

    vimentin, nuclear_mask = _watershed_input(foreground)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        cell_mask = segment_cell_bodies(vimentin, nuclear_mask)

    assert cell_mask.dtype == np.int32
    # The border-touching arms of the cross are not eroded away by the frame
    # edge: cells remain assigned along row/col 0 and the last row/col.
    assert cell_mask[0, 20] != 0
    assert cell_mask[20, 0] != 0
    assert cell_mask[-1, 20] != 0
    assert cell_mask[20, -1] != 0
