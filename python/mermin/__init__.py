"""mermin: k-atic alignment analysis of fluorescence microscopy."""

__version__ = "0.1.0"

__all__ = [
    "open_image",
    "PixelSizeError",
    "segment_nuclei",
    "segment_cell_bodies",
    "extract_contours",
    "build_neighbor_graph",
    "analyze",
    "Experiment",
]


def __getattr__(name: str):
    """Lazy import of submodules to avoid importing heavy dependencies unnecessarily."""
    if name == "open_image" or name == "PixelSizeError":
        from mermin.ingest import PixelSizeError, open_image
        return locals()[name]
    elif name in ("segment_nuclei", "segment_cell_bodies", "extract_contours", "build_neighbor_graph"):
        from mermin.segment import (
            segment_nuclei,
            segment_cell_bodies,
            extract_contours,
            build_neighbor_graph,
        )
        return locals()[name]
    elif name in ("analyze", "Experiment"):
        from mermin.pipeline import Experiment, analyze
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
