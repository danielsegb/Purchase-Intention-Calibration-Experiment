from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

URL = (
    "https://archive.ics.uci.edu/static/public/468/"
    "online%2Bshoppers%2Bpurchasing%2Bintention%2Bdataset.zip"
)
EXPECTED_SHA256 = "b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9"
OUT_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_FILE = OUT_DIR / "online_shoppers_intention.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading official UCI dataset from:\n{URL}")
    with urllib.request.urlopen(URL) as response:
        payload = response.read()

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        candidates = [n for n in zf.namelist() if n.lower().endswith("online_shoppers_intention.csv")]
        if not candidates:
            raise RuntimeError(f"CSV not found in UCI archive. Files: {zf.namelist()}")
        with zf.open(candidates[0]) as src, OUT_FILE.open("wb") as dst:
            dst.write(src.read())

    digest = sha256(OUT_FILE)
    print(f"Saved: {OUT_FILE}")
    print(f"SHA-256: {digest}")
    if digest != EXPECTED_SHA256:
        raise RuntimeError(
            "Downloaded CSV hash differs from the file used for the reported experiment. "
            "Do not proceed without checking whether UCI has changed the file."
        )
    print("Dataset hash matches the reported experiment.")


if __name__ == "__main__":
    main()
