"""mermin: k-atic alignment analysis of fluorescence microscopy."""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    # The installed wheel's version is dynamic (`dynamic = ["version"]` in
    # mermin-py/pyproject.toml): maturin reads it from mermin-py's Cargo.toml,
    # which takes it from `[workspace.package] version` in the root Cargo.toml.
    # That one place is the single source of truth for an installed package.
    __version__ = _version("mermin")
except PackageNotFoundError:
    # Not installed, e.g. this checkout on `sys.path` with no wheel built.
    # Kept in step with `[workspace.package] version` in Cargo.toml by hand.
    __version__ = "0.5.0"

__all__ = [
    "open_image",
    "PixelSizeError",
    "MerminError",
    "RoleError",
    "AmbiguousRoleError",
    "UnresolvableRoleError",
    "segment_nuclei",
    "segment_nuclei_with_provenance",
    "segment_cell_bodies",
    "extract_contours",
    "build_neighbor_graph",
    "SegmentationBackend",
    "ThresholdBackend",
    "CellposeBackend",
    "resolve_backend",
    "SegmentationError",
    "BackendUnavailableError",
    "analyze",
    "Experiment",
]


def __getattr__(name: str):
    """Lazy import of submodules to avoid importing heavy dependencies unnecessarily."""
    if name == "open_image" or name == "PixelSizeError":
        from mermin.ingest import PixelSizeError, open_image
        return locals()[name]
    elif name == "MerminError":
        from mermin.errors import MerminError
        return MerminError
    elif name in ("SegmentationError", "BackendUnavailableError"):
        from mermin.errors import BackendUnavailableError, SegmentationError
        return locals()[name]
    elif name in ("RoleError", "AmbiguousRoleError", "UnresolvableRoleError"):
        from mermin.roles import AmbiguousRoleError, RoleError, UnresolvableRoleError
        return locals()[name]
    elif name in (
        "segment_nuclei",
        "segment_nuclei_with_provenance",
        "segment_cell_bodies",
        "extract_contours",
        "build_neighbor_graph",
    ):
        from mermin.segment import (
            segment_nuclei,
            segment_nuclei_with_provenance,
            segment_cell_bodies,
            extract_contours,
            build_neighbor_graph,
        )
        return locals()[name]
    elif name in ("SegmentationBackend", "ThresholdBackend", "CellposeBackend", "resolve_backend"):
        from mermin.backends import (
            CellposeBackend,
            SegmentationBackend,
            ThresholdBackend,
            resolve_backend,
        )
        return locals()[name]
    elif name in ("analyze", "Experiment"):
        from mermin.pipeline import Experiment, analyze
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
