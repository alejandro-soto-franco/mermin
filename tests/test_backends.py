"""Segmentation backend protocol and the threshold implementation."""

import numpy as np
import pytest

from mermin.backends import SegmentationBackend, ThresholdBackend
from mermin.errors import MerminError, SegmentationError


def discs(shape=(120, 120), centres=((30, 30), (30, 90), (90, 30), (90, 90)), radius=12):
    """A plane holding disjoint bright discs on a dark background."""
    plane = np.zeros(shape, dtype=np.float64)
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    for r, c in centres:
        plane[(rows - r) ** 2 + (cols - c) ** 2 <= radius**2] = 1.0
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
        versions."""
        centres = ((20, 90), (20, 20), (90, 20))
        mask = ThresholdBackend().segment(discs(centres=centres))
        assert int(mask[20, 20]) == 1
        assert int(mask[20, 90]) == 2
        assert int(mask[90, 20]) == 3

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
