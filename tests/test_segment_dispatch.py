"""segment_nuclei dispatch, provenance and cache behaviour."""

import numpy as np
import pytest

from mermin.segment import segment_nuclei, segment_nuclei_with_provenance


class CountingBackend:
    """A backend that records how many times it actually segmented."""

    name = "counting"

    def __init__(self):
        self.calls = 0

    def version(self):
        return "counting-1"

    def config(self):
        return {"answer": 42}

    def segment(self, plane):
        self.calls += 1
        mask = np.zeros(plane.shape, dtype=np.int32)
        mask[:4, :4] = 1
        mask[-4:, -4:] = 2
        return mask


@pytest.fixture
def plane():
    rng = np.random.default_rng(1)
    return rng.random((32, 32))


def test_segment_nuclei_returns_the_mask(plane):
    backend = CountingBackend()
    mask = segment_nuclei(plane, backend)
    assert mask.dtype == np.int32
    assert backend.calls == 1


def test_provenance_records_the_backend_and_the_mechanism(plane):
    backend = CountingBackend()
    _mask, provenance = segment_nuclei_with_provenance(plane, backend)
    assert provenance["backend"] == "counting"
    assert provenance["version"] == "counting-1"
    assert provenance["config"] == {"answer": 42}
    assert provenance["mechanism"] == "instance"
    assert provenance["cache"] == "disabled"
    assert provenance["n_nuclei"] == 2


def test_auto_fallback_is_recorded_as_such(plane, monkeypatch):
    from mermin import backends as backends_module

    monkeypatch.setattr(backends_module, "cellpose_version", lambda: None)
    with pytest.warns(RuntimeWarning):
        _mask, provenance = segment_nuclei_with_provenance(plane, "auto")
    assert provenance["backend"] == "threshold"
    assert provenance["mechanism"] == "auto-fallback"


class TestCache:
    def test_a_second_call_hits_and_does_not_re_enter_the_backend(self, tmp_path, plane):
        backend = CountingBackend()
        first, first_prov = segment_nuclei_with_provenance(
            plane, backend, mask_cache=tmp_path
        )
        second, second_prov = segment_nuclei_with_provenance(
            plane, backend, mask_cache=tmp_path
        )
        assert backend.calls == 1
        assert first_prov["cache"] == "miss"
        assert second_prov["cache"] == "hit"
        assert np.array_equal(first, second)

    def test_a_changed_config_misses(self, tmp_path, plane):
        class Other(CountingBackend):
            def config(self):
                return {"answer": 43}

        segment_nuclei_with_provenance(plane, CountingBackend(), mask_cache=tmp_path)
        other = Other()
        _mask, provenance = segment_nuclei_with_provenance(
            plane, other, mask_cache=tmp_path
        )
        assert provenance["cache"] == "miss"
        assert other.calls == 1

    def test_a_changed_plane_misses(self, tmp_path, plane):
        backend = CountingBackend()
        segment_nuclei_with_provenance(plane, backend, mask_cache=tmp_path)
        other = plane.copy()
        other[0, 0] += 1.0
        _mask, provenance = segment_nuclei_with_provenance(
            other, backend, mask_cache=tmp_path
        )
        assert provenance["cache"] == "miss"
        assert backend.calls == 2
