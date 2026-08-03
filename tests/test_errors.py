"""The shared `MerminError` base and its reachability from the package root.

`PixelSizeError` (`mermin.ingest`) and `RoleError` (`mermin.roles`) used to
share no base, so catching everything `open_image` or `analyze` can raise
needed four names imported from two different modules, and only
`PixelSizeError` was reachable from `mermin` itself.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from mermin.errors import MerminError
from mermin.ingest import PixelSizeError
from mermin.roles import AmbiguousRoleError, RoleError, UnresolvableRoleError


def test_pixel_size_error_is_a_mermin_error():
    assert issubclass(PixelSizeError, MerminError)


def test_role_error_is_a_mermin_error():
    assert issubclass(RoleError, MerminError)


def test_role_error_subclasses_are_unchanged():
    assert issubclass(AmbiguousRoleError, RoleError)
    assert issubclass(UnresolvableRoleError, RoleError)


def test_all_four_names_are_reachable_from_the_package_root():
    import mermin

    assert mermin.MerminError is MerminError
    assert mermin.RoleError is RoleError
    assert mermin.AmbiguousRoleError is AmbiguousRoleError
    assert mermin.UnresolvableRoleError is UnresolvableRoleError


def test_all_four_names_are_declared_in_all():
    import mermin

    for name in ("MerminError", "RoleError", "AmbiguousRoleError", "UnresolvableRoleError"):
        assert name in mermin.__all__


def test_a_single_except_mermin_error_catches_both_families():
    def _raise(exc):
        raise exc("boom")

    for exc_cls in (PixelSizeError, RoleError, AmbiguousRoleError, UnresolvableRoleError):
        with pytest.raises(MerminError):
            _raise(exc_cls)


def test_bare_import_still_needs_nothing_heavy():
    """Reaching `MerminError`/`RoleError` via the package root must not pull
    in `bioio` (which `mermin.ingest` imports at module level): `RoleError`
    lives in `mermin.roles`, which has no heavy imports of its own, and
    `MerminError` lives in `mermin.errors`, which has none at all.
    """
    repo_root = Path(__file__).parent.parent
    code = (
        "import sys; sys.path.insert(0, 'python'); import mermin; "
        "mermin.RoleError; mermin.MerminError; "
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
