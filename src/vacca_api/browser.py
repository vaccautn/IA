"""Bounded auto-open tied to the serving Uvicorn instance, including reload."""

import asyncio
from contextlib import suppress
import logging
from multiprocessing import get_context
from urllib.parse import urlsplit, urlunsplit
import webbrowser

import uvicorn
from uvicorn.main import STARTUP_FAILURE
from uvicorn.supervisors import ChangeReload

from vacca_video.policies import require_single_server_process

logger = logging.getLogger(__name__)
BROWSER_READY_TIMEOUT_SECONDS = 120
BROWSER_READY_POLL_SECONDS = 0.1


class BrowserServer(uvicorn.Server):
    def __init__(self, config, url, opened):
        super().__init__(config)
        self.browser_url = url
        self.browser_opened = opened

    async def open_when_ready(self, deadline):
        loop = asyncio.get_running_loop()
        while not self.should_exit and not self.browser_opened.value:
            remaining = deadline - loop.time()
            if remaining <= 0:
                logger.warning("Browser readiness wait expired; open %s manually", self.browser_url)
                return
            if self.started:
                # A shared flag prevents another window on reload/worker restart.
                with self.browser_opened.get_lock():
                    if self.browser_opened.value:
                        return
                    self.browser_opened.value = True
                try:
                    self.browser_url = self.listening_browser_url()
                    if not webbrowser.open(self.browser_url):
                        logger.warning("Browser could not open %s; open it manually", self.browser_url)
                except Exception as exc:
                    logger.warning("Browser opening failed: %s; open %s manually", type(exc).__name__, self.browser_url)
                return
            await asyncio.sleep(min(BROWSER_READY_POLL_SECONDS, remaining))

    def listening_browser_url(self):
        parts = urlsplit(self.browser_url)
        if parts.port != 0:
            return self.browser_url
        # These are the actual asyncio listeners created by Uvicorn, including
        # those created from a reload child's inherited bound sockets.
        for listener in self.servers:
            for socket in listener.sockets or ():
                port = socket.getsockname()[1]
                if port:
                    host = parts.netloc.rsplit(":", 1)[0]
                    return urlunsplit(parts._replace(netloc=f"{host}:{port}"))
        raise RuntimeError("The serving instance has no assigned listener port")

    async def serve(self, sockets=None):
        deadline = asyncio.get_running_loop().time() + BROWSER_READY_TIMEOUT_SECONDS
        watcher = asyncio.create_task(self.open_when_ready(deadline))
        try:
            # Uvicorn sets started only after lifespan AND listening sockets.
            await super().serve(sockets=sockets)
        finally:
            watcher.cancel()
            with suppress(asyncio.CancelledError):
                await watcher


def run_with_browser(*, host, port, reload, url):
    config = uvicorn.Config("vacca_api.main:app", host=host, port=port,
                            reload=reload, log_level="info")
    require_single_server_process(config.workers)
    opened = get_context("spawn").Value("b", False)
    server = BrowserServer(config, url, opened)
    try:
        if config.should_reload:
            socket = config.bind_socket()
            try:
                ChangeReload(config, target=server.run, sockets=[socket]).run()
            finally:
                socket.close()
        else:
            server.run()
    except KeyboardInterrupt:
        pass  # Match Uvicorn's normal signal/launcher behavior.
    if not server.started and not config.should_reload and config.workers == 1:
        raise SystemExit(STARTUP_FAILURE)
