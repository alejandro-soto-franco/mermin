"""Segmentation over real corpus entries. Requires the ASF-EX1 volume.

`_entry_artefact_path` and the module-level skip come from `conftest.py`
rather than from `test_corpus_ingest.py`: `tests/` has no `__init__.py`, so a
bare `pytest tests/` run cannot import that module by its dotted path
(`tests.test_corpus_ingest`), and `conftest.py` is where both suites' shared
corpus-lookup helpers live instead.
"""

import importlib.util

import numpy as np
import pytest

from conftest import _entry_artefact_path, require_corpus

pytestmark = pytest.mark.corpus

require_corpus()

from mermin.backends import ThresholdBackend
from mermin.ingest import open_image
from mermin.pipeline import analyze
from mermin.segment import segment_nuclei_with_provenance

ENTRY = "phantom-defect-pair"


@pytest.fixture(scope="module")
def nuclear_plane():
    return open_image(_entry_artefact_path(ENTRY)).planes["nuclear"]


class TestThresholdOnRealData:
    def test_recovers_a_plausible_share_of_the_blobs(self, nuclear_plane):
        """The phantom scatters 60 blobs at uniform random positions with no
        minimum separation, so overlapping ones merge and the recoverable
        count is below 60. The band is the assertion; 60 exactly would be
        false. Measured on this entry (fixed seed 20260802, so this is not
        run-to-run noise): 50 nuclei recovered.
        """
        _mask, provenance = segment_nuclei_with_provenance(
            nuclear_plane, ThresholdBackend()
        )
        assert 25 <= provenance["n_nuclei"] <= 60

    def test_the_mask_matches_the_plane(self, nuclear_plane):
        mask, _provenance = segment_nuclei_with_provenance(
            nuclear_plane, ThresholdBackend()
        )
        assert mask.shape == nuclear_plane.shape
        assert mask.dtype == np.int32
        assert mask.min() == 0

    def test_two_runs_are_byte_identical(self, nuclear_plane):
        """Determinism on a real file rather than on synthetic discs. This is
        the property phase 6's byte-identity suite builds on."""
        first = ThresholdBackend().segment(nuclear_plane)
        second = ThresholdBackend().segment(nuclear_plane)
        assert first.tobytes() == second.tobytes()

    def test_the_cache_reproduces_the_uncached_mask(self, tmp_path, nuclear_plane):
        """Enabling the cache must never change a number."""
        uncached, _ = segment_nuclei_with_provenance(nuclear_plane, ThresholdBackend())
        _first, first_prov = segment_nuclei_with_provenance(
            nuclear_plane, ThresholdBackend(), mask_cache=tmp_path
        )
        cached, second_prov = segment_nuclei_with_provenance(
            nuclear_plane, ThresholdBackend(), mask_cache=tmp_path
        )
        assert first_prov["cache"] == "miss"
        assert second_prov["cache"] == "hit"
        assert cached.tobytes() == uncached.tobytes()

    def test_analyze_runs_end_to_end_on_the_threshold_backend(self):
        """The integration check: a real file, through the whole pipeline,
        with no cellpose and no explicit pixel size."""
        result = analyze(_entry_artefact_path(ENTRY), segmentation="threshold")
        assert result.segmentation["backend"] == "threshold"
        assert result.segmentation["mechanism"] == "explicit"
        assert result.segmentation["n_nuclei"] > 0
        assert result.ingest["pixel_size_um"] == pytest.approx(0.5)
        assert len(result.cells) > 0


@pytest.mark.skipif(
    importlib.util.find_spec("cellpose") is None,
    reason="cellpose is not installed; it is an optional extra",
)
class TestCellposeOnRealData:
    def test_the_cellpose_backend_runs_end_to_end(self, nuclear_plane):
        """Skipped wherever cellpose is absent, which includes CI. The first
        run downloads a model checkpoint.
        """
        from mermin.backends import CellposeBackend

        _mask, provenance = segment_nuclei_with_provenance(
            nuclear_plane, CellposeBackend()
        )
        assert provenance["backend"] == "cellpose"
        assert provenance["version"].startswith("cellpose-4")
        assert provenance["n_nuclei"] > 0
