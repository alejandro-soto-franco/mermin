"""Content-addressed nuclear mask cache."""

from unittest.mock import patch

import numpy as np
import pytest

from mermin.backends import ThresholdBackend
from mermin.maskcache import load_mask, mask_cache_key, store_mask

PINNED_KEY = "e929725ec6ddb15c628b8ee3b8a19167dd654bca312aa8281f73fcadfec57134"


@pytest.fixture
def plane():
    rng = np.random.default_rng(0)
    return rng.random((32, 32))


class TestKey:
    def test_same_inputs_give_the_same_key(self, plane):
        backend = ThresholdBackend()
        assert mask_cache_key(plane, backend) == mask_cache_key(plane, backend)

    def test_a_changed_plane_changes_the_key(self, plane):
        other = plane.copy()
        other[0, 0] += 1.0
        backend = ThresholdBackend()
        assert mask_cache_key(plane, backend) != mask_cache_key(other, backend)

    def test_a_changed_config_changes_the_key(self, plane):
        assert mask_cache_key(plane, ThresholdBackend(sigma=2.0)) != mask_cache_key(
            plane, ThresholdBackend(sigma=2.5)
        )

    def test_a_changed_shape_changes_the_key(self):
        flat = np.arange(16, dtype=np.float64)
        backend = ThresholdBackend()
        assert mask_cache_key(flat.reshape(4, 4), backend) != mask_cache_key(
            flat.reshape(2, 8), backend
        )

    def test_the_derivation_is_pinned(self):
        """A fixed input has a fixed digest, so a change to the key derivation
        shows up as a failing test rather than as silently missed cache
        entries."""
        plane = np.zeros((4, 4), dtype=np.float64)
        key = mask_cache_key(plane, ThresholdBackend())
        assert len(key) == 64
        assert key == PINNED_KEY

    def test_framing_prevents_a_field_boundary_collision(self, plane):
        """Two backends whose name and version concatenate to the same bytes
        under naive framing must still get different keys.

        `name="ab"`, `version()="c"` and `name="a"`, `version()="bc"`
        concatenate to the identical byte string `b"abc"` without a length
        prefix. The length prefix is what tells them apart.
        """

        class _Backend:
            def __init__(self, name, version):
                self.name = name
                self._version = version

            def version(self):
                return self._version

            def config(self):
                return {}

        left = _Backend("ab", "c")
        right = _Backend("a", "bc")
        assert mask_cache_key(plane, left) != mask_cache_key(plane, right)


class TestRoundTrip:
    def test_store_then_load_returns_the_mask(self, tmp_path, plane):
        mask = ThresholdBackend().segment(plane)
        key = mask_cache_key(plane, ThresholdBackend())
        store_mask(tmp_path, key, mask)
        loaded = load_mask(tmp_path, key)
        assert loaded is not None
        assert loaded.dtype == np.int32
        assert np.array_equal(loaded, mask)

    def test_a_missing_entry_is_none(self, tmp_path):
        assert load_mask(tmp_path, "0" * 64) is None

    def test_a_missing_directory_is_none(self, tmp_path):
        assert load_mask(tmp_path / "absent", "0" * 64) is None

    def test_store_creates_the_directory(self, tmp_path):
        target = tmp_path / "nested" / "cache"
        store_mask(target, "a" * 64, np.zeros((4, 4), dtype=np.int32))
        assert load_mask(target, "a" * 64) is not None

    def test_a_corrupt_entry_is_a_warned_miss(self, tmp_path):
        key = "b" * 64
        store_mask(tmp_path, key, np.zeros((4, 4), dtype=np.int32))
        entry = next(tmp_path.iterdir())
        entry.write_bytes(b"not an npz file")
        with pytest.warns(RuntimeWarning):
            assert load_mask(tmp_path, key) is None

    def test_no_temporary_files_are_left_behind(self, tmp_path):
        store_mask(tmp_path, "c" * 64, np.zeros((4, 4), dtype=np.int32))
        assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]

    def test_a_failed_write_leaves_no_entry_and_no_temporary_file(self, tmp_path):
        """Staging-then-swap must hold under a write failure partway through,
        not merely call os.replace on the happy path."""
        key = "d" * 64

        with patch(
            "mermin.maskcache.np.savez_compressed", side_effect=RuntimeError("boom")
        ):
            with pytest.raises(RuntimeError, match="boom"):
                store_mask(tmp_path, key, np.zeros((4, 4), dtype=np.int32))

        assert load_mask(tmp_path, key) is None
        assert list(tmp_path.iterdir()) == []
