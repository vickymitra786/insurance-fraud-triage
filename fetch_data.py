"""Download the public vehicle-claim fraud table (Oracle "carclaims") and rebuild it as one CSV.

Author: Vivek Mitra

Source: a public, no-login mirror on the Hugging Face Hub (dataset "FeatEng/Data",
folder "shivamb__vehicle-claim-fraud-detection"). The mirror stores the 15,420 rows
in two parquet files. Each row keeps its original row number in "__index_level_0__",
so sorting by that column restores the original order. The row number plus one is the
original PolicyNumber column (a running id with no information, so it is not a feature).

The download is pinned to one revision and checked against SHA-256 hashes.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RAW = DATA / "raw"
OUT_CSV = DATA / "carclaims.csv"

REVISION = "eb82fee0a992bccbf63cc667f2b044cd396c9a5c"
BASE = f"https://huggingface.co/datasets/FeatEng/Data/resolve/{REVISION}/shivamb__vehicle-claim-fraud-detection"
FILES = {
    "train-00000-of-00001.parquet": "8c39f5b5b048b4bf8a3eb1e80eb4f236587f584d309cd0cca9ba4f34aaa024e2",
    "test-00000-of-00001.parquet": "a75cd4e8cbaf81408317fd10cd128c2fcc122e20be6f38ee85370c09c1ff9b39",
}

EXPECTED_ROWS = 15420
EXPECTED_FRAUD = 923


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, digest in FILES.items():
        dst = RAW / name
        if not dst.exists() or sha256(dst) != digest:
            print(f"  downloading {name}")
            req = urllib.request.Request(f"{BASE}/{name}", headers={"User-Agent": "research-download/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as f:
                f.write(r.read())
        got = sha256(dst)
        if got != digest:
            sys.exit(f"hash mismatch for {name}: {got}")


def rebuild() -> pd.DataFrame:
    parts = [pd.read_parquet(RAW / n) for n in FILES]
    d = pd.concat(parts, ignore_index=True).sort_values("__index_level_0__").reset_index(drop=True)
    assert d["__index_level_0__"].tolist() == list(range(EXPECTED_ROWS)), "row numbers are not 0..15419"
    d = d.rename(columns={"__FeatEng_target__": "FraudFound_P"})
    d.insert(0, "PolicyNumber", d.pop("__index_level_0__") + 1)
    assert len(d) == EXPECTED_ROWS and int(d["FraudFound_P"].sum()) == EXPECTED_FRAUD
    return d


def main() -> pd.DataFrame:
    download()
    d = rebuild()
    d.to_csv(OUT_CSV, index=False)
    print(f"  {len(d):,} claims, {int(d.FraudFound_P.sum())} frauds -> {OUT_CSV.relative_to(HERE)}")
    return d


if __name__ == "__main__":
    main()
