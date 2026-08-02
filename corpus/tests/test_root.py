import os
from pathlib import Path

import pytest

from mermin_corpus.errors import CorpusMountError
from mermin_corpus.root import assert_mounted, corpus_root, ensure, entry_dir

EX1_UUID = "32d726f6-68d8-495f-b40c-e364c66b8ce5"


def test_default_root_is_ex1():
    os.environ.pop("MERMIN_CORPUS_ROOT", None)
    assert corpus_root() == Path("/mnt/ASF-EX1/mermin-corpus")


def test_env_overrides_root(tmp_path, monkeypatch):
    monkeypatch.setenv("MERMIN_CORPUS_ROOT", str(tmp_path))
    assert corpus_root() == tmp_path


def test_assert_mounted_is_a_noop_outside_mnt(tmp_path):
    # A tmp root carries no mount contract, so this must not raise.
    assert_mounted(tmp_path)


def test_assert_mounted_rejects_an_unmounted_mnt_path(monkeypatch):
    monkeypatch.setattr("os.path.ismount", lambda p: False)
    with pytest.raises(CorpusMountError, match="not mounted"):
        assert_mounted(Path("/mnt/ASF-EX1/mermin-corpus"))


def test_assert_mounted_rejects_the_wrong_device(monkeypatch):
    monkeypatch.setattr("os.path.ismount", lambda p: str(p) == "/mnt/ASF-EX1")

    real_stat = os.stat

    class FakeStat:
        st_dev = 1
        st_rdev = 999

    monkeypatch.setattr("os.stat", lambda p, *a, **k: FakeStat())
    with pytest.raises(CorpusMountError, match="UUID"):
        assert_mounted(Path("/mnt/ASF-EX1/mermin-corpus"))
    monkeypatch.setattr("os.stat", real_stat)


def test_ensure_refuses_to_mkdir_when_unmounted(monkeypatch):
    monkeypatch.setattr("os.path.ismount", lambda p: False)
    with pytest.raises(CorpusMountError):
        ensure(Path("/mnt/ASF-EX1/mermin-corpus/commercial-safe/x/raw"))


def test_ensure_creates_under_a_tmp_root(tmp_path):
    target = tmp_path / "commercial-safe" / "x" / "raw"
    assert ensure(target) == target
    assert target.is_dir()


def test_entry_dir_composes_partition_and_id(tmp_path, monkeypatch):
    monkeypatch.setenv("MERMIN_CORPUS_ROOT", str(tmp_path))
    assert entry_dir("eval-only", "bbbc021-w1") == tmp_path / "eval-only" / "bbbc021-w1"


@pytest.mark.drive
def test_real_ex1_passes_the_assertion():
    root = Path("/mnt/ASF-EX1")
    if not os.path.ismount(root):
        pytest.skip("ASF-EX1 is not mounted")
    assert_mounted(root / "mermin-corpus")
