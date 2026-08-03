"""Channel role resolution from whatever identity a file actually carries.

The corpus probe table showed channel identity arrives in three distinct
forms, which earlier designs treated as one: real biological labels
(`LaminB1`, `1-DAPI`), wavelengths used as names (`442.0`), and synthetic
placeholders bioio invents when a file carries nothing (`Channel:0:0`).
Only the first two are metadata.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

SYNTHETIC_NAME = re.compile(r"^Channel:\d+:\d+$")

NUCLEAR_TERMS = ("dapi", "hoechst", "dna", "nucleus", "nuclei")
# Lamins are intermediate filaments, the same family as vimentin, and the
# nuclear lamina is a filamentous meshwork, so a lamin stain is a fibre
# target rather than a nuclear one. Placing it under nuclear would make
# idr0062's `LaminB1, Dapi` ambiguous and leave that entry with no fibre.
FIBRE_TERMS = (
    "vimentin", "tubulin", "actin", "phalloidin", "lamin", "cy5", "af647",
)

# DAPI emits around 460 nm. Anything at or below this bound is nuclear.
NUCLEAR_MAX_NM = 500.0


class RoleError(Exception):
    """Channel roles could not be resolved."""


class AmbiguousRoleError(RoleError):
    """More than one channel claims the same role."""


class UnresolvableRoleError(RoleError):
    """No mechanism can assign both roles to distinct channels."""


@dataclass(frozen=True)
class RoleResolution:
    role: str
    index: int
    mechanism: str  # explicit | emission | name | position
    evidence: str


def _as_wavelength(name: str) -> float | None:
    try:
        return float(name)
    except (TypeError, ValueError):
        return None


def _usable_names(names: list[str]) -> list[str | None]:
    return [
        None if (n is None or SYNTHETIC_NAME.match(str(n))) else str(n) for n in names
    ]


def _match_terms(names: list[str | None], terms: tuple[str, ...]) -> list[int]:
    return [
        i
        for i, n in enumerate(names)
        if n is not None and any(t in n.lower() for t in terms)
    ]


def _from_emission(values: list[float | None]) -> dict[str, RoleResolution] | None:
    present = [(i, v) for i, v in enumerate(values) if v is not None]
    if len(present) < 2:
        return None
    nuclear = min(present, key=lambda p: p[1])
    fibre = max(present, key=lambda p: p[1])
    if nuclear[0] == fibre[0] or nuclear[1] > NUCLEAR_MAX_NM:
        return None
    return {
        "nuclear": RoleResolution("nuclear", nuclear[0], "emission", f"{nuclear[1]} nm"),
        "fibre": RoleResolution("fibre", fibre[0], "emission", f"{fibre[1]} nm"),
    }


def resolve_roles(
    channel_names: list[str],
    emission_nm: list[float | None],
    explicit: dict[str, int] | None = None,
) -> dict[str, RoleResolution]:
    """Assign `nuclear` and `fibre` to distinct channel indices."""
    if explicit:
        missing = {"nuclear", "fibre"} - set(explicit)
        if missing:
            raise RoleError(f"explicit mapping is missing {sorted(missing)}")
        if explicit["nuclear"] == explicit["fibre"]:
            raise RoleError("explicit mapping assigns both roles to one channel")
        return {
            role: RoleResolution(role, index, "explicit", "caller supplied")
            for role, index in explicit.items()
        }

    names = _usable_names(list(channel_names))
    count = len(names)
    if count < 2:
        raise UnresolvableRoleError(
            f"cannot assign two roles to one channel: {count} channel(s) present "
            f"with no distinguishing metadata"
        )

    # A numeric name IS a wavelength, so it feeds the emission path.
    numeric = [_as_wavelength(n) if n is not None else None for n in names]
    emission = list(emission_nm) if any(v is not None for v in emission_nm) else numeric
    resolved = _from_emission(emission)
    if resolved is not None:
        return resolved

    nuclear_hits = _match_terms(names, NUCLEAR_TERMS)
    fibre_hits = _match_terms(names, FIBRE_TERMS)
    if len(nuclear_hits) > 1:
        raise AmbiguousRoleError(
            f"channels {nuclear_hits} all match the nuclear vocabulary"
        )
    if len(fibre_hits) > 1:
        raise AmbiguousRoleError(f"channels {fibre_hits} all match the fibre vocabulary")
    if len(nuclear_hits) == 1 and len(fibre_hits) == 1 and nuclear_hits != fibre_hits:
        return {
            "nuclear": RoleResolution(
                "nuclear", nuclear_hits[0], "name", str(names[nuclear_hits[0]])
            ),
            "fibre": RoleResolution(
                "fibre", fibre_hits[0], "name", str(names[fibre_hits[0]])
            ),
        }

    warnings.warn(
        f"no channel metadata resolved a role; assuming position, "
        f"nuclear=0 and fibre=1 of {count} channels",
        UserWarning,
        stacklevel=2,
    )
    return {
        "nuclear": RoleResolution("nuclear", 0, "position", "assumed"),
        "fibre": RoleResolution("fibre", 1, "position", "assumed"),
    }
