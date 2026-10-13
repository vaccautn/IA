"""One local job at a time, with inference isolated in a CLI subprocess."""

import math
import os
import subprocess
import sys
import threading

from .policies import (
    DEFAULT_MAX_FRAMES,
    ROOT,
    WORKER_POLL_SECONDS,
    WORKER_TERMINATE_GRACE_SECONDS,
    WORKER_TIMEOUT_SECONDS,
)


class Worker:
    def __init__(self, store):
        self.store = store
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.ready.set()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.finish_video = None

    def loop(self):
        # Principal startup may defer claims until recovery commits successfully.
        self.ready.wait()
        while not self.stop.is_set():
            job_id = self.store.claim()
            if job_id is None:
                self.stop.wait(WORKER_POLL_SECONDS)
                continue
            directory = self.store.directory / job_id
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
                optional_error = None
                if self.finish_video is not None:
                    optional_error = self.finish_video(self.store.directory, job_id, self.stop)
                    if self.stop.is_set():
                        raise RuntimeError("El servicio se interrumpió antes de publicar el resultado.")
                self.store.update(job_id, "completed", optional_error)
            except Exception as exc:
                self.store.update(job_id, "failed", str(exc))
