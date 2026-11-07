"""Video resources owned by the principal API lifespan, not by a mounted app."""

import asyncio
from importlib.util import find_spec
import logging
import os
import threading

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
        self._admission_lock = threading.Lock()
        self._available = False
        self._worker_faulted = False
        self.upload_lock = asyncio.Lock()
        self.finish_video = None
        self.track_video = None
        self.finish_passages = None

    @property
    def available(self):
        with self._admission_lock:
            return self._available

    @available.setter
    def available(self, value):
        with self._admission_lock:
            self._available = value

    @property
    def worker_faulted(self):
        with self._admission_lock:
            return self._worker_faulted

    @property
    def history_available(self):
        with self._admission_lock:
            return self.store is not None and (self._available or self._worker_faulted)

    def _on_worker_fault(self):
        with self._admission_lock:
            self._worker_faulted = True
            self._available = False

    def add_job_if_available(
        self, job_id, filename, idempotency_key=None, source_sha256=None, source_size=None
    ):
        with self._admission_lock:
            if not self._available:
                return False
            self.store.add(job_id, filename, idempotency_key, source_sha256, source_size)
            return True

    def _recover_tracking(self):
        # Optional metadata: a failure here must not block detection startup.
        from .tracking_job import recover_interrupted

        try:
            recover_interrupted(self.store.directory)
        except OSError as exc:
            logger.error("Video tracking recovery skipped: %s", type(exc).__name__)

    def prepare(self):
        self.store = Store(self.directory)
        self.worker = self.worker_factory(self.store)

    def start(self, start_worker=True):
        try:
            check_runtime()
            if self.store is None:
                self.prepare()
            self.worker.finish_video = self.finish_video
            self.worker.track_video = self.track_video
            self.worker.finish_passages = self.finish_passages
            self.worker.on_fault = self._on_worker_fault
            if start_worker:
                # Startup and recovery must both succeed before any job is claimed.
                self.worker.ready.clear()
                with self.store.connect() as conn:
                    self.store.recover(conn)
                    if self.track_video is not None:
                        self._recover_tracking()
                    self.worker.thread.start()
                    self.started = True
                self.available = True
                self.worker.ready.set()
            else:
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
