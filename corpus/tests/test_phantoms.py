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
    with pytest.raises(KeyError):
        generate("nope", seed=1)
