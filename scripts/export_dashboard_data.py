"""Export slim dashboard tables. Usage: python scripts/export_dashboard_data.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retainiq.dashboard_export import run  # noqa: E402

if __name__ == "__main__":
    run()
