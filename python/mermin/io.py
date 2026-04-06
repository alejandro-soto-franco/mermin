"""TIFF loading, channel separation, and preprocessing."""

import numpy as np
import tifffile
from pathlib import Path


def load_tiff(path: str | Path, channels: dict[str, int] | None = None):
    """Load a multi-frame TIFF and separate channels.

    Args:
        path: Path to the TIFF file.
        channels: Mapping of channel name to frame index.
            Default: {"dapi": 0, "vimentin": 1}.

    Returns:
        dict mapping channel names to 2D numpy arrays (float64, normalized to [0, 1]).
    """
    if channels is None:
        channels = {"dapi": 0, "vimentin": 1}

    with tifffile.TiffFile(path) as tif:
        pages = tif.pages
        result = {}
        for name, idx in channels.items():
            if idx >= len(pages):
                raise ValueError(f"Frame {idx} not found in {path} (has {len(pages)} frames)")
            raw = pages[idx].asarray().astype(np.float64)
            # Percentile normalization
            p1, p99 = np.percentile(raw, 1), np.percentile(raw, 99.5)
            if p99 - p1 > 0:
                normalized = np.clip((raw - p1) / (p99 - p1), 0.0, 1.0)
            else:
                normalized = np.zeros_like(raw)
            result[name] = normalized
        return result


def discover_tiffs(directory: str | Path, pattern: str = "*.tif") -> list[Path]:
    """Find all TIFF files in a directory."""
    return sorted(Path(directory).glob(pattern))
