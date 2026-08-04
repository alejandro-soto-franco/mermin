"""Hermetic phantom test: `analyze()` against a phantom generated in process.

Generates from `mermin_corpus.phantoms`, writes it to a `tmp_path` TIFF, and
asserts only the closed-form properties the phantom's own construction
guarantees. This is deliberately not the golden suite: it needs no corpus
volume, so it is not marked `corpus` and runs in CI, where the golden suite
(`test_corpus_goldens.py`, gated on the ASF-EX1 mount) does not.

Nothing here asserts anything about defect detection. `_native.detect_defects`
does not recover the phantoms' known defect counts or charges; see the
"Analysis Pipeline" caveat in README.md and the measurement recorded in
`.superpowers/sdd/2026-08-04-phase4-goldens/task-5-report.md`.
"""
import tomllib
from pathlib import Path

import numpy as np
import pytest
import tifffile

pytest.importorskip(
    "mermin._native", reason="mermin._native is not built in this checkout"
)

from mermin.pipeline import analyze
from mermin_corpus.phantoms import generate

SEED_MANIFEST = Path(__file__).parent.parent / "corpus" / "manifest.seed.toml"

# Measured directly (see test_uniform_director_recovers_the_known_angle's
# docstring): the circular-mean recovery error for this phantom and seed is
# 0.0024 rad. This tolerance carries roughly an 8x margin over that, not a
# value picked to make the assertion pass.
DIRECTOR_TOLERANCE_RAD = 0.02


def _seed_for(generator: str) -> int:
    """The seed `corpus/manifest.seed.toml` records for a phantom generator.

    Reads the source-controlled seed manifest directly via `tomllib` rather
    than `mermin_corpus.manifest.load()`, which defaults to the ASF-EX1
    volume's live `manifest.toml` and would make this test depend on the
    drive it exists to run without.
    """
    with SEED_MANIFEST.open("rb") as f:
        manifest = tomllib.load(f)
    entry = next(
        e for e in manifest["entry"] if e["source"].get("generator") == generator
    )
    return int(entry["source"]["seed"])


def _write_phantom_tiff(path, result):
    """Write a phantom to an ImageJ-flavoured TIFF carrying axes and pixel
    size metadata, the same convention `mermin_corpus.fetch._fetch_generated`
    uses so the corpus volume's own phantom TIFFs read back identically.

    The phantom carries no wavelength metadata, so role resolution needs an
    explicit `channels` mapping at `analyze()`; only pixel size is read back
    from this file's own metadata.
    """
    tifffile.imwrite(
        path,
        result.image,
        imagej=True,
        metadata={"axes": result.axes, "unit": "micron"},
        resolution=(1.0 / result.pixel_size_um, 1.0 / result.pixel_size_um),
    )
    return path


def test_uniform_director_recovers_the_known_angle(tmp_path):
    """`uniform_director`'s truth carries a single constant angle by
    construction. The correct statistic for a nematic director defined
    modulo pi is the circular mean of `2 * theta`, halved and wrapped back
    into that same modulo-pi identification; a plain mean of `theta` is
    wrong for an angle with that identification and would pass or fail for
    the wrong reason.

    Measured for this phantom and seed: the circular-mean recovery error is
    0.0024 rad (about 0.14 degrees). `DIRECTOR_TOLERANCE_RAD` (0.02 rad)
    carries roughly an 8x margin over that measurement.
    """
    seed = _seed_for("uniform_director")
    result = generate("uniform_director", seed)
    path = _write_phantom_tiff(tmp_path / "uniform_director.tif", result)

    analysis = analyze(
        path, channels={"nuclear": 0, "fibre": 1}, segmentation="threshold"
    )

    theta = analysis.fields["theta"]
    circular_mean = 0.5 * np.angle(np.mean(np.exp(2j * theta)))
    known_angle = result.truth["theta"]

    # Both are defined modulo pi; wrap the difference into (-pi/2, pi/2]
    # before comparing, rather than the raw angles themselves.
    error = (circular_mean - known_angle + np.pi / 2) % np.pi - np.pi / 2
    assert abs(error) < DIRECTOR_TOLERANCE_RAD

    assert analysis.ingest["pixel_size_um"] == pytest.approx(result.pixel_size_um)
    assert len(analysis.cells) > 0
