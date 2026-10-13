"""Compatibility entry point: launch the principal API, including /videos."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    from scripts.run_api import main as run_api

    run_api(argv)


if __name__ == "__main__":
    main()
