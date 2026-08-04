"""Exceptions raised by the corpus tooling."""


class CorpusError(Exception):
    """Base class for every corpus failure."""


class CorpusMountError(CorpusError):
    """The corpus root is under /mnt but the expected volume is not mounted."""


class ManifestError(CorpusError):
    """A manifest entry is missing a field or contradicts itself."""


class FetchError(CorpusError):
    """A fetch failed, or produced bytes that disagree with a recorded hash."""


class ProbeError(CorpusError):
    """An artefact could not be probed, usually because it was never fetched."""


class GenerateError(CorpusError):
    """A golden could not be generated or checked."""
