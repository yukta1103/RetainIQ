"""Download the raw Olist CSVs. Usage: python scripts/download_data.py [--force]"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retainiq.data.download import download_all  # noqa: E402

if __name__ == "__main__":
    download_all(force="--force" in sys.argv)
