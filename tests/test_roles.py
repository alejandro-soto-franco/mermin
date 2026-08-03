import pytest

from mermin.roles import (
    AmbiguousRoleError,
    UnresolvableRoleError,
    resolve_roles,
)


def test_explicit_mapping_wins_outright():
    r = resolve_roles(["Dapi", "Vimentin"], [], explicit={"nuclear": 1, "fibre": 0})
    assert r["nuclear"].index == 1
    assert r["fibre"].index == 0
    assert r["nuclear"].mechanism == "explicit"


def test_vocabulary_matches_case_insensitively():
    # idr0062 names its channels LaminB1 and Dapi. Lamin is an intermediate
    # filament, so it is the fibre channel here, not a second nuclear one.
    r = resolve_roles(["LaminB1", "Dapi"], [])
    assert r["nuclear"].index == 1
    assert r["fibre"].index == 0
    assert r["nuclear"].mechanism == "name"


def test_vocabulary_matches_a_prefixed_label():
    # idr0047 prefixes fluorophores with an acquisition order.
    r = resolve_roles(["3-CY5", "5-TMR", "1-DAPI", "7-TRANS"], [])
    assert r["nuclear"].index == 2
    assert r["fibre"].index == 0


def test_numeric_names_are_treated_as_emission():
    # idr0021 names its channels by wavelength.
    r = resolve_roles(["442.0", "525.0", "615.0"], [])
    assert r["nuclear"].index == 0
    assert r["fibre"].index == 2
    assert r["nuclear"].mechanism == "emission"


def test_synthetic_names_are_never_metadata():
    # bioio invents these when a file carries no channel information.
    r = resolve_roles(["Channel:0:0", "Channel:0:1"], [470.0, 666.0])
    assert r["nuclear"].mechanism == "emission"
    assert r["nuclear"].index == 0
    assert r["fibre"].index == 1


def test_emission_resolves_the_two_montano_orderings():
    two = resolve_roles(["Channel:0:0", "Channel:0:1"], [470.0, 666.0])
    assert (two["nuclear"].index, two["fibre"].index) == (0, 1)
    three = resolve_roles(
        ["Channel:0:0", "Channel:0:1", "Channel:0:2"], [525.0, 470.0, 666.0]
    )
    assert (three["nuclear"].index, three["fibre"].index) == (1, 2)


def test_positional_fallback_warns():
    with pytest.warns(UserWarning, match="position"):
        r = resolve_roles(["Channel:0:0", "Channel:0:1"], [])
    assert r["nuclear"].index == 0
    assert r["fibre"].index == 1
    assert r["nuclear"].mechanism == "position"


def test_a_single_channel_cannot_supply_two_roles():
    # The Montano batch's calibration reference is single channel with no
    # emission metadata. Pairing a channel with itself is the defect this
    # rewrite exists to remove.
    with pytest.raises(UnresolvableRoleError, match="one channel"):
        resolve_roles(["Channel:0:0"], [])


def test_two_channels_matching_one_role_is_an_error():
    with pytest.raises(AmbiguousRoleError, match="nuclear"):
        resolve_roles(["DAPI", "Hoechst"], [])


def test_emission_ignores_none_entries():
    r = resolve_roles(["Channel:0:0", "Channel:0:1", "Channel:0:2"], [470.0, None, 666.0])
    assert (r["nuclear"].index, r["fibre"].index) == (0, 2)
