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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="VACCA Vision API server")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev)")
    parser.add_argument("--open", action="store_true", help="Open the local video UI")
    args = parser.parse_args(argv)

    import uvicorn
    from vacca_video.policies import require_single_server_process

    config = uvicorn.Config("vacca_api.main:app", host=args.host, port=args.port,
                            reload=args.reload, log_level="info")
    require_single_server_process(config.workers)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    url_host = f"[{args.host}]" if ":" in args.host else args.host
    base_url = f"http://{url_host}:{args.port}"
    logger.info("Starting VACCA Vision API at %s", base_url)
    logger.info("API docs: %s/docs", base_url)
    logger.info("Test UI: %s/ui", base_url)
    logger.info("Health: %s/health", base_url)

    if args.open:
        from vacca_api.browser import run_with_browser

        browser_host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(args.host, args.host)
        if ":" in browser_host:
            browser_host = f"[{browser_host}]"
        run_with_browser(host=args.host, port=args.port, reload=args.reload,
                         url=f"http://{browser_host}:{args.port}/videos/")
    else:
        uvicorn.run(
            "vacca_api.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info",
        )


if __name__ == "__main__":
    main()
