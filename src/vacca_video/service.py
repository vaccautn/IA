"""Video resources owned by the principal API lifespan, not by a mounted app."""

import asyncio
from importlib.util import find_spec
import logging
import os

from .policies import ROOT, WORKER_SHUTDOWN_JOIN_SECONDS
from .store import Store
from .worker import Worker

logger = logging.getLogger(__name__)
UNAVAILABLE_DETAIL = "El servicio de video no está disponible. Consultar el registro del servicio."


def check_runtime():
    # Probe optional packages without importing OpenCV, Torch or loading weights.
    for package in ("cv2", "ultralytics"):
        if find_spec(package) is None:
            raise ImportError(f"Missing video runtime: {package}")


class VideoResources:
    def __init__(self, directory=None, worker_factory=None):
        self.directory = directory or os.environ.get("VACCA_VIDEO_DATA", ROOT / "outputs/web")
        self.worker_factory = worker_factory or Worker
        self.store = None
        self.worker = None
        self.started = False
        self.available = False
        self.upload_lock = asyncio.Lock()

    def prepare(self):
        self.store = Store(self.directory)
        self.worker = self.worker_factory(self.store)

    def start(self, start_worker=True):
        try:
            check_runtime()
            if self.store is None:
                self.prepare()
            if start_worker:
                # Startup and recovery must both succeed before any job is claimed.
                self.worker.ready.clear()
                with self.store.connect() as conn:
                    self.store.recover(conn)
                    self.worker.thread.start()
                    self.started = True
                self.worker.ready.set()
            self.available = True
        except Exception as exc:
            logger.error("Video startup unavailable: %s", type(exc).__name__)
            self.close()

    def close(self):
        self.available = False
        if self.worker is not None:
            self.worker.stop.set()
            self.worker.ready.set()
            if self.started:
                self.worker.thread.join(timeout=WORKER_SHUTDOWN_JOIN_SECONDS)
                self.started = False
