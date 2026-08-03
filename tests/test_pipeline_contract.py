import inspect
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

from mermin.ingest import open_image


def _write_emission_tif(path, wavelengths):
    """A CYX tif whose ImageJ `Labels` carry a real wavelength per channel."""
    data = np.zeros((len(wavelengths), 16, 16), dtype=np.uint16)
    labels = [
        f'<MetaData><PlaneInfo><prop id="wavelength" type="float" value="{v}"/>'
        f"</PlaneInfo></MetaData>"
        for v in wavelengths
    ]
    tifffile.imwrite(path, data, imagej=True, metadata={"axes": "CYX", "Labels": labels})
    return path


def _write_unlabelled_tif(path):
    """A CYX tif with no channel metadata at all, so roles fall to position."""
    tifffile.imwrite(
        path, np.zeros((2, 16, 16), dtype=np.uint16), imagej=True,
        metadata={"axes": "CYX"},
    )
    return path


def _ingest_provenance(loaded):
    """Mirrors the `ingest={...}` construction in `mermin.pipeline.analyze`
    (mermin/pipeline.py, end of `analyze`), built from a real `LoadedImage`.

    `analyze` itself needs the compiled `mermin._native` extension, which is
    not built in this checkout; the task that added this test forbids
    stubbing it. `open_image` needs no extension, and it produces exactly
    the `roles`/`pixel_size_um`/`projection` data `analyze` feeds into
    `ingest`, so this exercises the real shape against real ingest output
    rather than only asserting a field name exists.
    """
    return {
        "roles": {
            role: {
                "index": resolution.index,
                "mechanism": resolution.mechanism,
                "evidence": resolution.evidence,
            }
            for role, resolution in loaded.roles.items()
        },
        "pixel_size_um": loaded.pixel_size_um,
        "projection": loaded.projection,
    }


def test_analyze_requires_no_default_pixel_size():
    from mermin.pipeline import analyze

    sig = inspect.signature(analyze)
    assert sig.parameters["pixel_size_um"].default is None


def test_io_module_is_gone():
    with pytest.raises(ModuleNotFoundError):
        import mermin.io  # noqa: F401


def test_analyze_exposes_ingest_provenance():
    from mermin.pipeline import AnalysisResult

    assert "ingest" in AnalysisResult.__dataclass_fields__


def test_ingest_provenance_distinguishes_measured_roles_from_a_guess(tmp_path):
    """`ingest`'s entire purpose is letting a caller tell a measured role
    identity from a positional guess after the fact, so assert the shape
    that lets it: per-role `index`/`mechanism`/`evidence`, plus the resolved
    `pixel_size_um` and `projection`, and that `mechanism` actually differs
    between a file with real emission metadata and one with none.
    """
    measured = open_image(
        _write_emission_tif(tmp_path / "measured.tif", (470.0, 666.0)),
        pixel_size_um=0.69,
    )
    with pytest.warns(UserWarning, match="position"):
        guessed = open_image(
            _write_unlabelled_tif(tmp_path / "guessed.tif"), pixel_size_um=0.69
        )

    for loaded, expected_mechanism in ((measured, "emission"), (guessed, "position")):
        ingest = _ingest_provenance(loaded)
        assert set(ingest) == {"roles", "pixel_size_um", "projection"}
        assert ingest["pixel_size_um"] == pytest.approx(0.69)
        assert ingest["projection"] == "single"
        assert set(ingest["roles"]) == {"nuclear", "fibre"}
        for role in ("nuclear", "fibre"):
            entry = ingest["roles"][role]
            assert set(entry) == {"index", "mechanism", "evidence"}
            assert isinstance(entry["index"], int)
            assert entry["mechanism"] == expected_mechanism

    # The whole point: a caller can tell these two apart afterwards.
    assert (
        _ingest_provenance(measured)["roles"]["nuclear"]["mechanism"]
        != _ingest_provenance(guessed)["roles"]["nuclear"]["mechanism"]
    )


def test_experiment_does_not_default_the_pixel_size():
    from mermin.pipeline import Experiment

    assert Experiment.__dataclass_fields__["pixel_size_um"].default is None


def test_analyze_and_experiment_reachable_from_package_root():
    import mermin
    from mermin.pipeline import Experiment, analyze

    assert mermin.analyze is analyze
    assert mermin.Experiment is Experiment
    assert "analyze" in mermin.__all__
    assert "Experiment" in mermin.__all__


def test_bare_import_needs_nothing_heavy():
    """`import mermin` alone must not drag in tifffile, scipy, scikit-image,
    cellpose or bioio. Run in a fresh subprocess so modules other test files
    already imported in this session cannot mask a regression."""
    repo_root = Path(__file__).parent.parent
    code = (
        "import sys; sys.path.insert(0, 'python'); import mermin; "
        "heavy = {'scipy', 'skimage', 'cellpose', 'bioio', 'tifffile'}; "
        "loaded = heavy & sys.modules.keys(); "
        "assert not loaded, sorted(loaded)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
