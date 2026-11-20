"""Download + extract BIRD dev into data/bird_raw (resumable, zip-slip safe).

    python -m evals.fetch_bird          # ~350 MB download, ~1 GB extracted
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"


def safe_extract(zip_path: Path, dest: Path) -> None:
    """Extract, refusing members that would land outside `dest` (zip-slip) and skipping
    macOS resource junk. The archive is untrusted data."""
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path) as z:
        for m in z.infolist():
            if m.filename.startswith("__MACOSX/"):
                continue
            target = (dest / m.filename).resolve()
            if dest != target and dest not in target.parents:
                raise ValueError(f"Unsafe path in archive: {m.filename}")
            z.extract(m, dest)


def download(url: str, out: Path) -> None:
    """Resume a partial download if one exists."""
    have = out.stat().st_size if out.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        if have and r.status != 206:  # server ignored Range: start over
            have = 0
        with open(out, "ab" if have else "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)


def main(root: str = "data/bird_raw") -> None:
    root_p = Path(root)
    root_p.mkdir(parents=True, exist_ok=True)
    zip_path = root_p / "dev.zip"
    if not any((root_p / "extracted").rglob("dev.json")):
        print("Downloading BIRD dev ...")
        download(URL, zip_path)
        safe_extract(zip_path, root_p / "extracted")
    inner = next((root_p / "extracted").rglob("dev_databases.zip"), None)
    if inner and not (inner.parent / "dev_databases").exists():
        print("Extracting databases ...")
        safe_extract(inner, inner.parent)
    print("BIRD ready under", root_p)


if __name__ == "__main__":
    main(*sys.argv[1:2])
