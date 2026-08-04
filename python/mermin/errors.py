"""The shared exception base for everything ingest and role resolution raise.

`PixelSizeError` (`mermin.ingest`) and `RoleError` (`mermin.roles`) shared no
base, so catching everything a call to `open_image` or `analyze` can raise
needed four names imported from two different modules. `MerminError` lets a
caller write one `except mermin.MerminError` instead.

Kept to a single class with no imports of its own, so that reaching it
through the package root's lazy `__getattr__` never pulls in anything heavier
than this module.
"""
from __future__ import annotations


class MerminError(Exception):
    """Base class for every exception mermin itself raises."""


class SegmentationError(MerminError):
    """Segmentation could not be carried out as asked."""


class BackendUnavailableError(SegmentationError):
    """A named segmentation backend is not installed, or is too old."""
