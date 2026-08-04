"""Segmentation backend protocol and the threshold implementation."""

import numpy as np
import pytest

from mermin.backends import SegmentationBackend, ThresholdBackend, _remove_small_objects
from mermin.errors import MerminError, SegmentationError


def discs(shape=(120, 120), centres=((30, 30), (30, 90), (90, 30), (90, 90)), radius=12):
    """A plane holding disjoint bright discs on a dark background."""
    plane = np.zeros(shape, dtype=np.float64)
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    for r, c in centres:
        plane[(rows - r) ** 2 + (cols - c) ** 2 <= radius**2] = 1.0
    return plane


def big_and_small_squares(shape=(200, 120)):
    """A plane holding two disjoint bright squares of different sizes.

    An elongated rectangle's distance transform has a flat ridge along its
    long axis, which `peak_local_max` (with a finite `min_distance`) breaks
    into several seeds rather than one - that shape does not isolate the
    sort. A square's distance transform is a single pyramid with one
    interior maximum, so each blob here contributes exactly one seed.

    The big square (rows 100-180, centre row 140) has a larger inscribed
    radius, hence higher distance-transform intensity, than the small one
    (rows 10-30, centre row 20). `peak_local_max` returns coordinates in
    intensity-descending order, so its raw order visits the big square
    first even though it sits at the higher row. Sorting seeds by
    `(row, col)` must reverse that and put the small square's label first.
    Verified directly: with the sort removed, `peak_local_max` returns
    `[(140, 60), (20, 30)]` on this fixture - big square first.
    """
    plane = np.zeros(shape, dtype=np.float64)
    plane[100:181, 20:101] = 1.0  # big: 81 x 81, centre (140, 60)
    plane[10:31, 20:41] = 1.0  # small: 21 x 21, centre (20, 30)
    return plane


class TestThresholdBackend:
    def test_recovers_every_disc(self):
        mask = ThresholdBackend().segment(discs())
        labels = np.unique(mask)
        assert labels[0] == 0
        assert len(labels) - 1 == 4

    def test_labels_sit_on_the_discs(self):
        centres = ((30, 30), (30, 90), (90, 30), (90, 90))
        mask = ThresholdBackend().segment(discs(centres=centres))
        found = set()
        for r, c in centres:
            label = int(mask[r, c])
            assert label > 0, f"centre {(r, c)} was not segmented"
            found.add(label)
        assert len(found) == 4, "two centres share a label"

    def test_mask_is_int32_two_dimensional(self):
        mask = ThresholdBackend().segment(discs())
        assert mask.dtype == np.int32
        assert mask.ndim == 2
        assert mask.shape == (120, 120)

    def test_repeated_runs_are_byte_identical(self):
        plane = discs()
        first = ThresholdBackend().segment(plane)
        second = ThresholdBackend().segment(plane)
        assert first.tobytes() == second.tobytes()

    def test_label_numbering_follows_row_then_column(self):
        """Labels come from seeds sorted by (row, col), never from
        `peak_local_max` ordering, so numbering is stable across skimage
        versions.

        This fixture does not discriminate the sort from a no-op: these
        three discs already sit in row-major order, so `peak_local_max`
        happens to return them in exactly that order regardless of whether
        the implementation sorts. Kept because it still documents the
        intended numbering; `test_seed_sort_reverses_peak_local_max_order`
        below is the one that actually exercises the sort."""
        centres = ((20, 90), (20, 20), (90, 20))
        mask = ThresholdBackend().segment(discs(centres=centres))
        assert int(mask[20, 20]) == 1
        assert int(mask[20, 90]) == 2
        assert int(mask[90, 20]) == 3

    def test_seed_sort_reverses_peak_local_max_order(self):
        """`peak_local_max` visits the big square first (higher
        distance-transform intensity), but sorting seeds by `(row, col)`
        must put the small square - at the lower row - first instead.
        A fixture where the two orderings agree cannot tell a working sort
        from a removed one."""
        mask = ThresholdBackend().segment(big_and_small_squares())
        assert int(mask[20, 30]) == 1  # small square: lower row, label 1
        assert int(mask[140, 60]) == 2  # big square: higher row, label 2

    def test_empty_plane_gives_an_empty_mask(self):
        mask = ThresholdBackend().segment(np.zeros((64, 64), dtype=np.float64))
        assert mask.shape == (64, 64)
        assert mask.dtype == np.int32
        assert not mask.any()

    def test_non_two_dimensional_input_raises(self):
        with pytest.raises(SegmentationError, match="2D"):
            ThresholdBackend().segment(np.zeros((3, 16, 16)))

    def test_config_carries_every_parameter_that_changes_the_mask(self):
        assert ThresholdBackend(sigma=1.5, min_size=30, min_distance=7).config() == {
            "sigma": 1.5,
            "min_size": 30,
            "min_distance": 7,
        }

    def test_version_and_name(self):
        backend = ThresholdBackend()
        assert backend.name == "threshold"
        assert isinstance(backend.version(), str)
        assert backend.version()

    def test_satisfies_the_protocol(self):
        assert isinstance(ThresholdBackend(), SegmentationBackend)


def test_segmentation_error_is_a_mermin_error():
    assert issubclass(SegmentationError, MerminError)


def test_remove_small_objects_keeps_exactly_min_size_drops_one_less():
    """Pins the boundary at `min_size` itself, version-independent.

    `skimage.morphology.remove_small_objects` moved this boundary between
    0.25 and 0.26 (`< min_size` became `<= max_size`), which is exactly why
    `_remove_small_objects` is written out rather than delegated. Areas are
    exact rectangles so the pixel counts are known, not incidental."""
    foreground = np.zeros((10, 20), dtype=bool)
    foreground[0:5, 0:5] = True  # 5 x 5 = 25 pixels: exactly min_size
    foreground[0:4, 10:16] = True  # 4 x 6 = 24 pixels: min_size - 1

    kept = _remove_small_objects(foreground, min_size=25)

    expected = np.zeros_like(foreground)
    expected[0:5, 0:5] = True
    assert np.array_equal(kept, expected)
