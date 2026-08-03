import numpy as np
import pytest
import tifffile

from mermin.ingest import PixelSizeError, open_image


def _write(path, data, **meta):
    tifffile.imwrite(path, data, imagej=True, metadata={"axes": "CYX", **meta})
    return path


def test_pixel_size_from_metadata(tmp_path):
    p = tmp_path / "cal.tif"
    tifffile.imwrite(
        p, np.zeros((2, 16, 16), dtype=np.uint16), imagej=True,
        metadata={"axes": "CYX", "unit": "micron"}, resolution=(1 / 0.5, 1 / 0.5),
    )
    assert open_image(p, channels={"nuclear": 0, "fibre": 1}).pixel_size_um == pytest.approx(0.5)


def test_explicit_pixel_size_overrides_metadata(tmp_path):
    p = tmp_path / "cal.tif"
    tifffile.imwrite(
        p, np.zeros((2, 16, 16), dtype=np.uint16), imagej=True,
        metadata={"axes": "CYX", "unit": "micron"}, resolution=(1 / 0.5, 1 / 0.5),
    )
    img = open_image(p, channels={"nuclear": 0, "fibre": 1}, pixel_size_um=0.69)
    assert img.pixel_size_um == pytest.approx(0.69)


def test_missing_pixel_size_raises_rather_than_defaulting(tmp_path):
    p = _write(tmp_path / "raw.tif", np.zeros((2, 16, 16), dtype=np.uint16))
    with pytest.raises(PixelSizeError, match="no pixel size"):
        open_image(p, channels={"nuclear": 0, "fibre": 1})


def test_planes_are_native_endian_float64(tmp_path):
    p = tmp_path / "be.tif"
    tifffile.imwrite(p, np.zeros((2, 16, 16), dtype=">f4"))
    img = open_image(p, channels={"nuclear": 0, "fibre": 1}, pixel_size_um=1.0)
    for plane in img.planes.values():
        assert plane.dtype == np.float64
        assert plane.dtype.byteorder in ("=", "|")


def test_planes_are_percentile_normalised(tmp_path):
    data = np.zeros((2, 32, 32), dtype=np.uint16)
    data[0, 8:24, 8:24] = 4000
    p = _write(tmp_path / "n.tif", data)
    img = open_image(p, channels={"nuclear": 0, "fibre": 1}, pixel_size_um=1.0)
    assert img.planes["nuclear"].min() >= 0.0
    assert img.planes["nuclear"].max() <= 1.0


def test_roles_carry_their_mechanism(tmp_path):
    p = _write(tmp_path / "r.tif", np.zeros((2, 16, 16), dtype=np.uint16))
    img = open_image(p, channels={"nuclear": 1, "fibre": 0}, pixel_size_um=1.0)
    assert img.roles["nuclear"].mechanism == "explicit"
    assert img.roles["nuclear"].index == 1


def test_projection_is_recorded(tmp_path):
    p = _write(tmp_path / "p.tif", np.zeros((2, 16, 16), dtype=np.uint16))
    img = open_image(p, channels={"nuclear": 0, "fibre": 1}, pixel_size_um=1.0)
    assert img.projection == "single"


def _write_zstack(path):
    # ImageJ hyperstacks require TZCYXS axis order on write.
    # Z=0 and Z=1 share a hot square (200, then 50) so max and mean disagree
    # there rather than by a uniform scale factor; Z=2 carries a disjoint
    # square (100) so single(z=0), max and mean are each a distinguishable
    # pattern after percentile normalisation, not merely a rescaled copy.
    data = np.zeros((3, 2, 8, 8), dtype=np.uint16)
    data[0, 0, 0:4, 0:4] = 200
    data[1, 0, 0:4, 0:4] = 50
    data[2, 0, 4:8, 4:8] = 100
    tifffile.imwrite(
        path, data, imagej=True,
        metadata={"axes": "ZCYX", "unit": "micron"}, resolution=(1, 1),
    )
    return path


def test_projection_max_and_mean_disagree_with_single_and_each_other(tmp_path):
    p = _write_zstack(tmp_path / "zstack.tif")
    channels = {"nuclear": 0, "fibre": 1}
    single = open_image(p, channels=channels, pixel_size_um=1.0, projection="single", z=0)
    maxp = open_image(p, channels=channels, pixel_size_um=1.0, projection="max")
    meanp = open_image(p, channels=channels, pixel_size_um=1.0, projection="mean")

    assert not np.array_equal(single.planes["nuclear"], maxp.planes["nuclear"])
    assert not np.array_equal(maxp.planes["nuclear"], meanp.planes["nuclear"])
    assert maxp.projection == "max"
    assert meanp.projection == "mean"


def test_projection_single_returns_the_named_plane(tmp_path):
    p = _write_zstack(tmp_path / "zstack.tif")
    channels = {"nuclear": 0, "fibre": 1}
    z0 = open_image(p, channels=channels, pixel_size_um=1.0, projection="single", z=0)
    z2 = open_image(p, channels=channels, pixel_size_um=1.0, projection="single", z=2)
    assert z0.projection == "single"
    assert not np.array_equal(z0.planes["nuclear"], z2.planes["nuclear"])
    # z=0's hot square sits top-left; z=2's disjoint square sits bottom-right.
    assert z0.planes["nuclear"][0, 0] > z0.planes["nuclear"][4, 4]
    assert z2.planes["nuclear"][4, 4] > z2.planes["nuclear"][0, 0]


def test_unsupported_projection_raises(tmp_path):
    p = _write(tmp_path / "p.tif", np.zeros((2, 16, 16), dtype=np.uint16))
    with pytest.raises(ValueError, match="unsupported projection"):
        open_image(p, channels={"nuclear": 0, "fibre": 1}, pixel_size_um=1.0, projection="bogus")


def test_projection_with_no_z_axis_returns_the_plane_rather_than_raising(tmp_path):
    data = np.zeros((2, 16, 16), dtype=np.uint16)
    data[0, 4:12, 4:12] = 4000
    p = _write(tmp_path / "no_z.tif", data)
    channels = {"nuclear": 0, "fibre": 1}
    single = open_image(p, channels=channels, pixel_size_um=1.0, projection="single")
    maxp = open_image(p, channels=channels, pixel_size_um=1.0, projection="max")
    meanp = open_image(p, channels=channels, pixel_size_um=1.0, projection="mean")
    assert np.array_equal(single.planes["nuclear"], maxp.planes["nuclear"])
    assert np.array_equal(maxp.planes["nuclear"], meanp.planes["nuclear"])


def test_emission_length_mismatch_warns_and_falls_through(tmp_path):
    p = tmp_path / "mismatch.tif"
    data = np.zeros((3, 16, 16), dtype=np.uint16)
    labels = [
        f'<MetaData><PlaneInfo><prop id="wavelength" type="float" value="{v}"/>'
        f"</PlaneInfo></MetaData>"
        for v in (400.0, 500.0, 600.0, 700.0)
    ]
    tifffile.imwrite(p, data, imagej=True, metadata={"axes": "CYX", "Labels": labels})

    with pytest.warns(UserWarning) as record:
        img = open_image(p, pixel_size_um=1.0)

    messages = [str(w.message) for w in record]
    assert any("4 ImageJ labels for 3 channels" in m for m in messages)
    assert any("position" in m for m in messages)
    assert img.roles["nuclear"].mechanism == "position"
    assert img.roles["nuclear"].index == 0
    assert img.roles["fibre"].index == 1
