"""Train and evaluate CLV models. Usage: python scripts/train_clv.py [--no-sim]"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retainiq.models.train_clv import run  # noqa: E402

if __name__ == "__main__":
    run(run_simulation="--no-sim" not in sys.argv)
