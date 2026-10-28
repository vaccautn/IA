"""One local job at a time, with inference isolated in a CLI subprocess."""

import math
import os
import subprocess
import sys
import threading
import logging

from .policies import (
    DEFAULT_MAX_FRAMES,
    ROOT,
    WORKER_POLL_SECONDS,
    WORKER_TERMINATE_GRACE_SECONDS,
    WORKER_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)


class Worker:
    def __init__(self, store):
        self.store = store
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.ready.set()
        self._health_lock = threading.Lock()
        self._health = "starting"
        self.on_fault = None
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.finish_video = None
        self.track_video = None

    @property
    def health(self):
        with self._health_lock:
            return self._health

    def _set_health(self, health):
        with self._health_lock:
            self._health = health

    def _fault(self, error):
        with self._health_lock:
            if self._health == "faulted":
                return
            self._health = "faulted"
        logger.error("Video worker stopped after storage failure: %s", type(error).__name__)
        if self.on_fault is not None:
            self.on_fault()

    def loop(self):
        try:
            self._run_loop()
        except Exception as exc:
            self._fault(exc)
        if self.health != "faulted":
            self._set_health("stopped")

    def _run_loop(self):
        # Principal startup may defer claims until recovery commits successfully.
        self.ready.wait()
        if self.stop.is_set():
            return
        self._set_health("healthy")
        while not self.stop.is_set():
            try:
                job_id = self.store.claim()
            except Exception as exc:
                self._fault(exc)
                return
            if job_id is None:
                self.stop.wait(WORKER_POLL_SECONDS)
                continue
            directory = self.store.directory / job_id
            status = "completed"
            error = None
            try:
                command = [
                    sys.executable,
                    "-u",
                    str(ROOT / "scripts/process_video.py"),
                    str(directory / "input.mp4"),
                    "--output",
                    str(directory / "result"),
                    "--max-frames",
                    str(DEFAULT_MAX_FRAMES),
                    "--preview-limit",
                    str(DEFAULT_MAX_FRAMES),
                    "--no-video",
                ]
                with (directory / "worker.log").open("w", encoding="utf-8") as log:
                    with subprocess.Popen(
                        command,
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                    ) as proc:
                        # Preserve the existing polling budget: 600 seconds / 0.5.
                        polls = math.ceil(WORKER_TIMEOUT_SECONDS / WORKER_POLL_SECONDS)
                        for _ in range(polls):
                            if proc.poll() is not None:
                                break
                            if self.stop.wait(WORKER_POLL_SECONDS):
                                break
                        if proc.poll() is None:
                            proc.terminate()
                            try:
                                proc.wait(timeout=WORKER_TERMINATE_GRACE_SECONDS)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                                proc.wait()
                            raise RuntimeError(
                                "Procesamiento interrumpido o límite de 10 minutos alcanzado."
                            )
                        if proc.returncode:
                            raise RuntimeError(
                                "No se pudo analizar el archivo. Verificá que sea un video MP4 válido y que haya espacio disponible."
                            )
                if self.finish_video is not None:
                    error = self.finish_video(self.store.directory, job_id, self.stop)
                    if self.stop.is_set():
                        raise RuntimeError("El servicio se interrumpió antes de publicar el resultado.")
            except Exception as exc:
                status = "failed"
                error = str(exc)
            try:
                self.store.update(job_id, status, error)
            except Exception as exc:
                self._fault(exc)
                return
            if status == "completed" and self.track_video is not None:
                # Optional and after publication: detections are never held back
                # and a tracking failure never changes the job outcome.
                try:
                    self.track_video(directory, self.stop)
                except Exception as exc:
                    logger.error("Video tracking hook failed: %s", type(exc).__name__)
