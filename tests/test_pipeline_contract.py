import inspect

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
