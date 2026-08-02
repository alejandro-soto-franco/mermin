"""Rung 0 phantoms whose director field is known in closed form.

Each generator returns a two-channel uint16 image: channel 0 stands in for the
nuclear stain, channel 1 for the fibre stain whose texture carries the director.
The fibre channel is a sinusoidal grating whose local phase follows the target
angle field, which is what a structure tensor recovers.

`truth` records what the analysis must produce, so these entries assert in
closed form rather than against a pinned golden.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

SIZE = 256
PIXEL_SIZE_UM = 0.5
PERIOD_PX = 8.0


@dataclass
class PhantomResult:
    image: np.ndarray
    axes: str
    pixel_size_um: float
    truth: dict
    theta: np.ndarray


def _grid() -> tuple[np.ndarray, np.ndarray]:
    c = np.arange(SIZE) - SIZE / 2.0
    return np.meshgrid(c, c, indexing="xy")


def _nuclei(rng: np.random.Generator) -> np.ndarray:
    """Scattered elliptical blobs standing in for a nuclear stain."""
    x, y = _grid()
    field = np.zeros_like(x)
    count = 60
    cx = rng.uniform(-SIZE / 2.0, SIZE / 2.0, count)
    cy = rng.uniform(-SIZE / 2.0, SIZE / 2.0, count)
    sx = rng.uniform(3.0, 6.0, count)
    sy = rng.uniform(3.0, 6.0, count)
    for i in range(count):
        field += np.exp(-(((x - cx[i]) / sx[i]) ** 2 + ((y - cy[i]) / sy[i]) ** 2))
    return np.clip(field + rng.normal(0.0, 0.01, field.shape), 0.0, 1.0)


def _render(theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Grating whose wavevector is perpendicular to the director `theta`."""
    x, y = _grid()
    phase = 2.0 * np.pi * (x * np.sin(theta) - y * np.cos(theta)) / PERIOD_PX
    fibre = 0.5 * (1.0 + np.sin(phase))
    fibre = np.clip(fibre + rng.normal(0.0, 0.02, fibre.shape), 0.0, 1.0)

    nuclei = _nuclei(rng)

    stack = np.stack([nuclei, fibre])
    return (stack * 65535.0).astype(np.uint16)


def uniform_director(seed: int) -> PhantomResult:
    rng = np.random.default_rng(seed)
    angle = float(rng.uniform(0.0, np.pi))
    theta = np.full((SIZE, SIZE), angle)
    return PhantomResult(
        _render(theta, rng), "CYX", PIXEL_SIZE_UM,
        {"field": "uniform", "theta": angle, "n_defects": 0, "charges": [], "total_charge": 0},
        theta,
    )


def defect_pair(seed: int) -> PhantomResult:
    rng = np.random.default_rng(seed)
    x, y = _grid()
    sep = SIZE / 4.0
    theta = (
        0.5 * np.arctan2(y, x - sep)
        - 0.5 * np.arctan2(y, x + sep)
    )
    return PhantomResult(
        _render(theta, rng), "CYX", PIXEL_SIZE_UM,
        {
            "field": "defect_pair",
            "n_defects": 2,
            "charges": [0.5, -0.5],
            "total_charge": 0.0,
            "positions_px": [[SIZE / 2 + sep, SIZE / 2], [SIZE / 2 - sep, SIZE / 2]],
        },
        theta,
    )


def radial_defect(seed: int) -> PhantomResult:
    rng = np.random.default_rng(seed)
    x, y = _grid()
    theta = np.arctan2(y, x)
    return PhantomResult(
        _render(theta, rng), "CYX", PIXEL_SIZE_UM,
        {
            "field": "radial",
            "n_defects": 1,
            "charges": [1.0],
            "total_charge": 1.0,
            "positions_px": [[SIZE / 2, SIZE / 2]],
        },
        theta,
    )


def hexatic_lattice(seed: int) -> PhantomResult:
    rng = np.random.default_rng(seed)
    x, y = _grid()
    theta = np.zeros_like(x)
    for m in range(6):
        a = m * np.pi / 3.0
        theta += np.cos(2.0 * np.pi * (x * np.cos(a) + y * np.sin(a)) / 24.0)
    theta = np.angle(np.exp(1j * theta)) / 6.0
    return PhantomResult(
        _render(theta, rng), "CYX", PIXEL_SIZE_UM,
        {"field": "hexatic", "k": 6, "n_defects": 0, "charges": [], "total_charge": 0.0},
        theta,
    )


GENERATORS: dict[str, Callable[[int], PhantomResult]] = {
    "uniform_director": uniform_director,
    "defect_pair": defect_pair,
    "radial_defect": radial_defect,
    "hexatic_lattice": hexatic_lattice,
}


def generate(name: str, seed: int) -> PhantomResult:
    return GENERATORS[name](seed)
