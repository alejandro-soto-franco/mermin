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
    # A bare `tifffile.imwrite` with no axes metadata reads back through
    # bioio as a single channel with a Z extent of 2, not two channels, so
    # this must go through `_write` like every other fixture here: without
    # it, `channels={"fibre": 1}` names a channel that does not exist and
    # role resolution now rejects it rather than reading the wrong plane.
    p = _write(tmp_path / "be.tif", np.zeros((2, 16, 16), dtype=">f4"))
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


def _write_ome_zarr(path, *, unit=None, scale=None, channels=("nuclear", "fibre")):
    """A minimal two-channel NGFF v0.4 store, `CYX`, 8x8, all zero.

    `unit` sets the X and Y axes' declared unit; omitting it leaves the axis
    with no `unit` key at all, which is the NGFF convention for "this axis
    is uncalibrated", not a missing conversion factor of 1. `scale` sets the
    dataset's `coordinateTransformations` scale for X and Y; omitting it
    omits the transform entirely, as an unwritten dataset would.
    """
    import zarr

    data = np.zeros((len(channels), 8, 8), dtype=np.uint16)
    group = zarr.open_group(store=str(path), mode="w", zarr_format=2)
    arr = group.create_array("0", shape=data.shape, dtype=data.dtype, chunks=data.shape)
    arr[:] = data

    def _axis(name, kind):
        axis = {"name": name, "type": kind}
        if kind == "space" and unit is not None:
            axis["unit"] = unit
        return axis

    axes = [_axis("c", "channel"), _axis("y", "space"), _axis("x", "space")]
    dataset = {"path": "0"}
    if scale is not None:
        dataset["coordinateTransformations"] = [
            {"type": "scale", "scale": [1.0, scale, scale]}
        ]
    multiscales = [{"axes": axes, "datasets": [dataset], "version": "0.4"}]
    group.attrs["multiscales"] = multiscales
    group.attrs["omero"] = {"channels": [{"label": c} for c in channels]}
    return path


@pytest.mark.parametrize(
    "unit, scale",
    [
        ("micrometer", 0.69),
        ("nanometer", 690.0),
        ("millimeter", 0.00069),
        ("centimeter", 0.000069),
        ("meter", 0.00000069),
    ],
)
def test_ome_zarr_pixel_size_converts_the_recognised_units(tmp_path, unit, scale):
    # bioio's own `physical_pixel_sizes` returns the raw scale for OME-Zarr
    # without ever consulting the axis's declared unit, so a store whose
    # scale is expressed in anything other than micrometres would otherwise
    # come back off by orders of magnitude.
    p = _write_ome_zarr(tmp_path / f"{unit}.zarr", unit=unit, scale=scale)
    img = open_image(p, channels={"nuclear": 0, "fibre": 1})
    assert img.pixel_size_um == pytest.approx(0.69, rel=1e-6)


@pytest.mark.parametrize("unit", ["micron", "um", "nm", "mm", "cm", "m"])
def test_ome_zarr_pixel_size_accepts_the_unit_aliases(tmp_path, unit):
    p = _write_ome_zarr(tmp_path / f"alias_{unit}.zarr", unit=unit, scale=1.0)
    img = open_image(p, channels={"nuclear": 0, "fibre": 1})
    assert img.pixel_size_um > 0.0


def test_ome_zarr_uncalibrated_raises_rather_than_reporting_one(tmp_path):
    # NGFF's own convention for "no calibration" is a scale of 1.0 with no
    # unit declared on the axis. bioio's `physical_pixel_sizes` takes that
    # scale at face value and reports 1.0 micron per pixel; this is the
    # silent-guess defect I2 exists to close.
    p = _write_ome_zarr(tmp_path / "uncal.zarr", unit=None, scale=1.0)
    with pytest.raises(PixelSizeError, match="no pixel size"):
        open_image(p, channels={"nuclear": 0, "fibre": 1})


def test_ome_zarr_missing_transform_raises(tmp_path):
    p = _write_ome_zarr(tmp_path / "no_transform.zarr", unit="micrometer", scale=None)
    with pytest.raises(PixelSizeError, match="no pixel size"):
        open_image(p, channels={"nuclear": 0, "fibre": 1})


def test_ome_zarr_unrecognised_unit_raises(tmp_path):
    p = _write_ome_zarr(tmp_path / "bogus_unit.zarr", unit="furlong", scale=0.69)
    with pytest.raises(PixelSizeError, match="no pixel size"):
        open_image(p, channels={"nuclear": 0, "fibre": 1})


def test_ome_zarr_explicit_pixel_size_overrides_metadata(tmp_path):
    p = _write_ome_zarr(tmp_path / "cal.zarr", unit="nanometer", scale=690.0)
    img = open_image(p, channels={"nuclear": 0, "fibre": 1}, pixel_size_um=1.23)
    assert img.pixel_size_um == pytest.approx(1.23)


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
