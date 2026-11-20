import zipfile

import pytest

from evals.fetch_bird import safe_extract


def test_safe_extract_blocks_zip_slip(tmp_path):
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("../escaped.txt", "x")
    with pytest.raises(ValueError, match="Unsafe path"):
        safe_extract(z, tmp_path / "out")
    assert not (tmp_path / "escaped.txt").exists()


def test_safe_extract_extracts_and_skips_macosx(tmp_path):
    z = tmp_path / "ok.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("a/b.txt", "hi")
        f.writestr("__MACOSX/junk", "x")
    safe_extract(z, tmp_path / "out")
    assert (tmp_path / "out/a/b.txt").read_text() == "hi"
    assert not (tmp_path / "out/__MACOSX").exists()
