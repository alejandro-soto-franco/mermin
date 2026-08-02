import hashlib

from mermin_corpus.hashing import sha256_file, sha256_tree, size_of


def test_sha256_file_matches_hashlib(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"mermin" * 1000)
    assert sha256_file(p) == hashlib.sha256(b"mermin" * 1000).hexdigest()


def test_sha256_tree_is_order_independent(tmp_path):
    a = tmp_path / "a"
    a.mkdir()
    (a / "z.bin").write_bytes(b"z")
    (a / "m.bin").write_bytes(b"m")
    first = sha256_tree(a)

    b = tmp_path / "b"
    b.mkdir()
    (b / "m.bin").write_bytes(b"m")
    (b / "z.bin").write_bytes(b"z")
    assert sha256_tree(b) == first


def test_sha256_tree_changes_when_a_name_changes(tmp_path):
    a = tmp_path / "a"
    (a / "sub").mkdir(parents=True)
    (a / "sub" / "x.bin").write_bytes(b"x")
    before = sha256_tree(a)
    (a / "sub" / "x.bin").rename(a / "sub" / "y.bin")
    assert sha256_tree(a) != before


def test_size_of_sums_a_tree(tmp_path):
    a = tmp_path / "a"
    (a / "sub").mkdir(parents=True)
    (a / "sub" / "x.bin").write_bytes(b"1234")
    (a / "y.bin").write_bytes(b"12")
    assert size_of(a) == 6
