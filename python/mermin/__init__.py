"""mermin: k-atic alignment analysis of fluorescence microscopy."""

__version__ = "0.1.0"

__all__ = [
    "load_tiff",
    "discover_tiffs",
    "segment_nuclei",
    "segment_cell_bodies",
    "extract_contours",
    "build_neighbor_graph",
]


def __getattr__(name: str):
    """Lazy import of submodules to avoid importing heavy dependencies unnecessarily."""
    if name == "load_tiff" or name == "discover_tiffs":
        from mermin.io import load_tiff, discover_tiffs
        return locals()[name]
    elif name in ("segment_nuclei", "segment_cell_bodies", "extract_contours", "build_neighbor_graph"):
        from mermin.segment import (
            segment_nuclei,
            segment_cell_bodies,
            extract_contours,
            build_neighbor_graph,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
