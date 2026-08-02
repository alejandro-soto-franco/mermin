import multiprocessing as mp
import time

import pytest

from mermin_corpus.lock import entry_lock


def _hold(root, started, release):
    import os
    os.environ["MERMIN_CORPUS_ROOT"] = root
    from mermin_corpus.lock import entry_lock
    with entry_lock("busy"):
        started.set()
        release.wait(timeout=10)


def test_lock_is_exclusive_across_processes(tmp_path, monkeypatch):
    monkeypatch.setenv("MERMIN_CORPUS_ROOT", str(tmp_path))
    ctx = mp.get_context("spawn")
    started, release = ctx.Event(), ctx.Event()
    proc = ctx.Process(target=_hold, args=(str(tmp_path), started, release))
    proc.start()
    try:
        assert started.wait(timeout=10)
        with pytest.raises(BlockingIOError):
            with entry_lock("busy", blocking=False):
                pass
    finally:
        release.set()
        proc.join(timeout=10)


def test_lock_is_reacquirable_after_release(tmp_path, monkeypatch):
    monkeypatch.setenv("MERMIN_CORPUS_ROOT", str(tmp_path))
    with entry_lock("free", blocking=False):
        pass
    with entry_lock("free", blocking=False):
        pass
