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
