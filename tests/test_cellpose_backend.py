"""`CellposeBackend.segment`, exercised for real against a synthetic image.

Every cellpose test elsewhere in this suite (`test_backends.py`,
`test_segment_dispatch.py`) monkeypatches `cellpose_version` to simulate
cellpose being installed, so what they check is the availability gate, the
version comparison and the import hygiene - the wrapper around `segment`,
never the call into cellpose itself. `test_corpus_segmentation.py` does call
it for real, but that suite is marked `corpus` and needs the ASF-EX1 volume,
so it never runs in CI.

This is the test that closes that gap: it imports cellpose and calls
`model.eval`, which is exactly where the 0.4.0 defect lived - `models.
Cellpose` did not exist in cellpose 4, and its `eval` returned four values
where version 4 returns three. Nothing here needs the ASF-EX1 corpus, so it
carries no `corpus` mark and runs wherever cellpose is installed, including
the dedicated CI job that installs `mermin[cellpose]`.

Skipped cleanly when cellpose is absent. Where cellpose is importable, this
test must run and must be able to fail: a job that installs cellpose and then
skips this test is worse than no job, because it reports green.
"""

import importlib.util

import numpy as np
import pytest


def _nuclear_stain(shape=(256, 256), count=14, seed=20260804):
    """Gaussian blobs on a dark field, standing in for a nuclear stain.

    In the style of `corpus/src/mermin_corpus/phantoms.py::_nuclei`, but not
    imported from there: this test must not depend on the corpus package,
    which the hermetic cellpose CI job does not install. Blobs are fewer and
    larger than the phantom's (radius 10-16px against a 256px frame, rather
    than 3-6px) and kept clear of the frame edge, both aimed at giving
    Cellpose-SAM - trained on real microscopy, not on this - the best
    plausible chance of recognising something in a crude synthetic image.
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]]
    field = np.zeros(shape, dtype=np.float64)
    margin = 40
    cx = rng.uniform(margin, shape[1] - margin, count)
    cy = rng.uniform(margin, shape[0] - margin, count)
    radius = rng.uniform(10.0, 16.0, count)
    for i in range(count):
        field += np.exp(
            -(((x - cx[i]) ** 2 + (y - cy[i]) ** 2) / (2.0 * radius[i] ** 2))
        )
    field = np.clip(field, 0.0, 1.0) + rng.normal(0.0, 0.01, shape)
    return (np.clip(field, 0.0, 1.0) * 255.0).astype(np.uint8)


@pytest.mark.skipif(
    importlib.util.find_spec("cellpose") is None,
    reason="cellpose is not installed; it is an optional extra",
)
class TestCellposeBackendRunsForReal:
    def test_segment_returns_the_api_contract(self):
        """The valuable assertions here are the ones that would have caught
        the 0.4.0 break: `CellposeModel` accepts the keywords the backend
        passes it, `eval` accepts the four keywords the backend passes and
        returns exactly the three values the backend unpacks, the
        `cpsam_v2` checkpoint name resolves and downloads, and what comes
        back is a 2D int32 mask with 0 as background.

        Asserts only that something was found, not how much. Observed
        directly against this fixture (cellpose 4.2.1.1, CPU, three
        repeated runs): a stable 9 of the 14 blobs recovered, every time.
        That count is not pinned here - deep-net inference is not
        guaranteed bit-identical across hardware or a cellpose point
        release the way the rest of this suite's masks are, so a fixed
        figure would eventually fail for reasons unrelated to a real
        regression. `n_nuclei > 0` is the part of that observation safe to
        rely on everywhere this test runs.
        """
        from mermin.backends import CellposeBackend

        plane = _nuclear_stain()
        mask = CellposeBackend().segment(plane)

        assert mask.shape == plane.shape
        assert mask.ndim == 2
        assert mask.dtype == np.int32
        assert mask.min() == 0
        assert mask.max() > 0, "found no nuclei at all in a fixture built to hold 14"

    def test_segment_nuclei_with_provenance_runs_the_same_path(self):
        """The dispatch layer `analyze` actually calls, not `segment`
        directly, so that a break confined to `segment_nuclei_with_provenance`
        - rather than to `CellposeBackend.segment` itself - is also caught."""
        from mermin.backends import CellposeBackend
        from mermin.segment import segment_nuclei_with_provenance

        plane = _nuclear_stain()
        mask, provenance = segment_nuclei_with_provenance(plane, CellposeBackend())

        assert mask.dtype == np.int32
        assert provenance["backend"] == "cellpose"
        assert provenance["version"].startswith("cellpose-4")
        assert provenance["n_nuclei"] == int(np.count_nonzero(np.unique(mask)))
