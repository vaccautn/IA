"""Launch the VACCA Vision API server.

Usage:
    .venv/Scripts/python scripts/run_api.py                  # default port 8001
    .venv/Scripts/python scripts/run_api.py --port 3000      # custom port
    .venv/Scripts/python scripts/run_api.py --reload         # dev mode
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="VACCA Vision API server")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev)")
    args = parser.parse_args()

    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    base_url = f"http://{args.host}:{args.port}"
    logger.info("Starting VACCA Vision API at %s", base_url)
    logger.info("API docs: %s/docs", base_url)
    logger.info("Test UI: %s/ui", base_url)
    logger.info("Health: %s/health", base_url)

    uvicorn.run(
        "vacca_api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
