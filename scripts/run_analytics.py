"""Run Phase 2 analytics. Usage: python scripts/run_analytics.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retainiq.analytics.run_analytics import run  # noqa: E402

if __name__ == "__main__":
    run()
