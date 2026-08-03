import inspect
import subprocess
import sys
from pathlib import Path

import pytest


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
