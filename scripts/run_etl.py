"""Run the full ETL. Usage: python scripts/run_etl.py [--quiet]"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retainiq.pipeline.run_etl import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
