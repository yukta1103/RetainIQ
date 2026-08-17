"""Fetch the raw Olist CSVs into data/raw/.

Idempotent: a file whose size already matches the manifest is skipped, so
re-running the pipeline costs nothing.
"""

from __future__ import annotations

import sys

import requests

from retainiq.config import HF_BASE_URL, HF_REPO, RAW_DIR
from retainiq.data.schema import ALL_TABLES, TableSpec

CHUNK = 1 << 20  # 1 MiB


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,.1f} {unit}" if unit != "B" else f"{n:,} B"
        n /= 1024.0
    return str(n)


def download_table(spec: TableSpec, *, force: bool = False) -> bool:
    """Download one table. Returns True if bytes were fetched, False if cached."""
    dest = RAW_DIR / spec.filename

    if dest.exists() and not force:
        actual = dest.stat().st_size
        if actual == spec.expected_bytes:
            print(f"  [cached] {spec.filename:<45} {_human(actual)}")
            return False
        print(
            f"  [stale ] {spec.filename} is {actual:,} B, expected "
            f"{spec.expected_bytes:,} B — re-downloading"
        )

    url = f"{HF_BASE_URL}/{spec.filename}"
    tmp = dest.with_suffix(dest.suffix + ".part")

    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        written = 0
        next_mark = 20  # report at 20/40/60/80%, not on every 1 MiB chunk
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(CHUNK):
                fh.write(chunk)
                written += len(chunk)
                pct = 100.0 * written / spec.expected_bytes
                if pct >= next_mark:
                    print(f"  [get   ] {spec.filename:<45} {pct:5.1f}%")
                    next_mark += 20

    # Integrity gate: a silent partial download would poison every downstream
    # number, so fail loudly rather than proceeding with truncated data.
    if written != spec.expected_bytes:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"{spec.filename}: downloaded {written:,} B but manifest expects "
            f"{spec.expected_bytes:,} B. Refusing to use a corrupt file."
        )

    tmp.replace(dest)
    print(f"  [ok    ] {spec.filename:<45} {_human(written)}")
    return True


def download_all(*, force: bool = False) -> None:
    print(f"Source: huggingface.co/datasets/{HF_REPO}")
    print(f"Target: {RAW_DIR}\n")

    fetched = 0
    for spec in ALL_TABLES:
        if download_table(spec, force=force):
            fetched += 1

    total = sum(t.expected_bytes for t in ALL_TABLES)
    print(f"\n{len(ALL_TABLES)} tables present ({_human(total)} total), {fetched} newly downloaded.")


if __name__ == "__main__":  # pragma: no cover
    download_all(force="--force" in sys.argv)
