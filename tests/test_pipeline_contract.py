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
    from mermin.pipeline import _ingest_provenance

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


def test_experiment_exposes_the_same_optional_fields_as_analyze_with_the_same_defaults():
    import inspect

    from mermin.pipeline import Experiment, analyze

    sig = inspect.signature(analyze)
    for name in ("channels", "projection", "z", "t"):
        assert name in Experiment.__dataclass_fields__
        assert (
            Experiment.__dataclass_fields__[name].default
            == sig.parameters[name].default
        )


def test_experiment_run_passes_channels_projection_z_t_to_analyze(monkeypatch):
    """`Experiment.run()` used to call `analyze(p, pixel_size_um=...)` and
    stop, so a file needing an explicit channel mapping, or a non-default
    Z/T selection, was unreachable from the batch API. Assert every one of
    the four fields actually reaches `analyze` for every path in the batch.
    """
    import mermin.pipeline as pipeline_module

    captured = []

    def fake_analyze(path, **kwargs):
        captured.append((path, kwargs))
        return object()

    monkeypatch.setattr(pipeline_module, "analyze", fake_analyze)

    experiment = pipeline_module.Experiment(
        pixel_size_um=0.69,
        channels={"nuclear": 0, "fibre": 1},
        projection="max",
        z=2,
        t=1,
    )
    experiment.add_condition("ctrl", ["a.tif", "b.tif"])
    experiment.run()

    assert len(captured) == 2
    for path, kwargs in captured:
        assert kwargs["channels"] == {"nuclear": 0, "fibre": 1}
        assert kwargs["pixel_size_um"] == pytest.approx(0.69)
        assert kwargs["projection"] == "max"
        assert kwargs["z"] == 2
        assert kwargs["t"] == 1


def test_analyze_and_experiment_reachable_from_package_root():
    import mermin
    from mermin.pipeline import Experiment, analyze

    assert mermin.analyze is analyze
    assert mermin.Experiment is Experiment
    assert "analyze" in mermin.__all__
    assert "Experiment" in mermin.__all__


def test_summary_omits_internal_katic_when_the_column_is_absent():
    # `analyze()` never constructs `internal_katic_k2`, so `summary()` used
    # to report a mean of exactly 0.000 unconditionally, misleadingly
    # implying the value had been measured.
    import polars as pl

    from mermin.pipeline import AnalysisResult

    result = AnalysisResult(
        cells=pl.DataFrame({"label": [1, 2]}),
        fields={},
        defects=[],
        correlations={},
        frank={"ratio": 1.5},
        ldg_params={},
        persistence={"pairs": []},
        ingest={},
        segmentation={},
    )
    summary = result.summary()
    assert "psi_2" not in summary
    assert "2 cells" in summary
    assert "Frank ratio = 1.50" in summary


def test_summary_reports_internal_katic_when_the_column_is_present():
    import polars as pl

    from mermin.pipeline import AnalysisResult

    result = AnalysisResult(
        cells=pl.DataFrame({"internal_katic_k2": [0.1, 0.5, 0.9]}),
        fields={},
        defects=[],
        correlations={},
        frank={"ratio": 1.0},
        ldg_params={},
        persistence={"pairs": []},
        ingest={},
        segmentation={},
    )
    assert "mean |psi_2| = 0.500" in result.summary()


def test_analyze_takes_segmentation_and_mask_cache():
    from mermin.pipeline import analyze

    parameters = inspect.signature(analyze).parameters
    assert parameters["segmentation"].default == "auto"
    assert parameters["mask_cache"].default is None


def test_analyze_no_longer_takes_cellpose_diameter():
    """Removed rather than deprecated: backend parameters belong on the
    backend."""
    from mermin.pipeline import analyze

    assert "cellpose_diameter" not in inspect.signature(analyze).parameters


def test_experiment_carries_the_segmentation_settings():
    from mermin.pipeline import Experiment

    experiment = Experiment()
    assert experiment.segmentation == "auto"
    assert experiment.mask_cache is None


def test_analysis_result_carries_segmentation_provenance():
    from dataclasses import fields

    from mermin.pipeline import AnalysisResult

    assert "segmentation" in {f.name for f in fields(AnalysisResult)}


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
