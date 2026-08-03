import numpy as np
import pytest

from mermin_corpus.phantoms import GENERATORS, generate


def test_all_generators_produce_two_channel_uint16():
    for name in GENERATORS:
        r = generate(name, seed=20260802)
        assert r.image.dtype == np.uint16
        assert r.axes == "CYX"
        assert r.image.shape[0] == 2
        assert r.pixel_size_um > 0


def test_generation_is_deterministic_under_a_seed():
    a = generate("defect_pair", seed=7).image
    b = generate("defect_pair", seed=7).image
    assert np.array_equal(a, b)


def test_uniform_director_truth_records_a_single_angle():
    r = generate("uniform_director", seed=1)
    assert r.truth["total_charge"] == 0
    assert r.truth["n_defects"] == 0
    assert 0.0 <= r.truth["theta"] < np.pi


def test_defect_pair_truth_is_two_half_charges_summing_to_zero():
    r = generate("defect_pair", seed=1)
    assert r.truth["n_defects"] == 2
    assert r.truth["charges"] == [0.5, -0.5]
    assert r.truth["total_charge"] == 0.0


def test_radial_defect_truth_is_a_single_plus_one():
    r = generate("radial_defect", seed=1)
    assert r.truth["charges"] == [1.0]
    assert r.truth["total_charge"] == 1.0


def test_hexatic_lattice_truth_records_k_six():
    r = generate("hexatic_lattice", seed=1)
    assert r.truth["k"] == 6
    assert r.truth["total_charge"] == 0.0


def test_unknown_generator_raises():
    with pytest.raises(ValueError, match="nope"):
        generate("nope", seed=1)


def _loop_winding(theta: np.ndarray, col: int, row: int, r: int) -> float:
    """Nematic winding around a square pixel loop, in units of 2 pi.

    Differences are wrapped into (-pi/2, pi/2] because a director is defined
    modulo pi, so a vector wrap into (-pi, pi] would give the wrong charge.
    """
    pts = []
    pts += [(row - r, col + i) for i in range(-r, r)]
    pts += [(row + i, col + r) for i in range(-r, r)]
    pts += [(row + r, col + i) for i in range(r, -r, -1)]
    pts += [(row + i, col - r) for i in range(r, -r, -1)]
    vals = np.array([theta[p, q] for p, q in pts])
    d = np.diff(np.concatenate([vals, vals[:1]]))
    d = (d + np.pi / 2.0) % np.pi - np.pi / 2.0
    return float(d.sum() / (2.0 * np.pi))


def test_measured_winding_matches_the_recorded_charges():
    half = 128  # SIZE // 2
    sep = 64    # SIZE // 4

    r = generate("defect_pair", seed=3)
    assert _loop_winding(r.theta, half + sep, half, 20) == pytest.approx(0.5, abs=0.05)
    assert _loop_winding(r.theta, half - sep, half, 20) == pytest.approx(-0.5, abs=0.05)

    r = generate("radial_defect", seed=3)
    assert _loop_winding(r.theta, half, half, 30) == pytest.approx(1.0, abs=0.05)

    for name in ("uniform_director", "hexatic_lattice"):
        r = generate(name, seed=3)
        assert _loop_winding(r.theta, half + 40, half - 30, 15) == pytest.approx(0.0, abs=0.05)
