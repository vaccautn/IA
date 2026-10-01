"""Standalone, single-user video service. Run on loopback with one worker only."""

from contextlib import asynccontextmanager, contextmanager
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import threading
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

ROOT = Path(__file__).resolve().parents[2]
MAX_UPLOAD = 50 * 1024 * 1024
MAX_FRAMES = 300
RESERVE = 50 * 1024 * 1024


class Store:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.db = self.directory / "jobs.sqlite3"
        with self.connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, name TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL, error TEXT)"
            )

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.db, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def list(self):
        with self.connect() as conn:
            return [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM jobs ORDER BY created_at DESC LIMIT 100"
                )
            ]

    def get(self, job_id):
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "Análisis no encontrado")
        return dict(row)

    def add(self, job_id, name):
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO jobs VALUES (?, ?, 'pending', ?, NULL)",
                (job_id, name, datetime.now(timezone.utc).isoformat()),
            )

    def update(self, job_id, status, error=None):
        with self.connect() as conn:
            conn.execute(
                "UPDATE jobs SET status=?, error=? WHERE id=?", (status, error, job_id)
            )

    def recover(self):
        with self.connect() as conn:
            conn.execute(
                "UPDATE jobs SET status='failed', error='El servicio se interrumpió. Volvé a cargar el video para reintentar.' WHERE status='processing'"
            )

    def claim(self):
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT id FROM jobs WHERE status='pending' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE jobs SET status='processing' WHERE id=?", (row["id"],)
                )
                return row["id"]


class Worker:
    def __init__(self, store):
        self.store = store
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def loop(self):
        while not self.stop.is_set():
            job_id = self.store.claim()
            if job_id is None:
                self.stop.wait(0.5)
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
                    str(MAX_FRAMES),
                    "--preview-limit",
                    str(MAX_FRAMES),
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
                        for _ in range(1200):
                            if proc.poll() is not None:
                                break
                            if self.stop.wait(0.5):
                                break
                        if proc.poll() is None:
                            proc.terminate()
                            try:
                                proc.wait(timeout=5)
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
                self.store.update(job_id, "completed")
            except Exception as exc:
                self.store.update(job_id, "failed", str(exc))


def records(directory):
    path = directory / "result/detections.jsonl"
    if not path.exists():
        return []
    result = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                result.append(json.loads(line))
            except ValueError:
                break  # Worker may be in the middle of writing its last line.
    return result


def create_app(directory=None, start_worker=True):
    store = Store(directory or os.environ.get("VACCA_VIDEO_DATA", ROOT / "outputs/web"))
    worker = Worker(store)

    @asynccontextmanager
    async def lifespan(app):
        if start_worker:
            store.recover()
            worker.thread.start()
        yield
        worker.stop.set()
        if start_worker:
            worker.thread.join(timeout=10)

    app = FastAPI(
        title="VACCA Video · prototipo independiente",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.store = store
    upload_lock = asyncio.Lock()
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "testserver"]
    )

    @app.middleware("http")
    async def local_origin(request, call_next):
        origin = request.headers.get("origin")
        if origin and origin != str(request.base_url).rstrip("/"):
            return JSONResponse({"detail": "Origen no permitido"}, status_code=403)
        if request.method == "POST":
            async with upload_lock:
                return await call_next(request)
        return await call_next(request)

    def detail(job):
        directory = store.directory / job["id"]
        frames = records(directory)
        summary_path = directory / "result/summary.json"
        summary = None
        if job["status"] in ("completed", "failed") and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return dict(
            job,
            frames_analyzed=len(frames),
            summary=summary,
            schema_version="vacca-video-job-v1",
        )

    @app.get("/api/v1/analyses")
    def list_jobs():
        return [detail(job) for job in store.list()]

    @app.post("/api/v1/analyses", status_code=202)
    async def upload(
        request: Request, filename: str = Query(min_length=1, max_length=200)
    ):
        if not filename.lower().endswith(".mp4"):
            raise HTTPException(415, "Elegí un video MP4.")
        with store.connect() as conn:
            active = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN ('pending','processing')"
            ).fetchone()[0]
        if active >= 3:
            raise HTTPException(
                429, "Hay tres videos en espera. Esperá a que terminen."
            )
        if shutil.disk_usage(store.directory).free < MAX_UPLOAD + RESERVE:
            raise HTTPException(
                507,
                "No hay espacio suficiente. Liberá espacio antes de cargar otro video.",
            )
        job_id = uuid4().hex
        directory = store.directory / job_id
        directory.mkdir()
        target = directory / "input.mp4"
        size = 0
        try:
            with target.open("xb") as stream:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_UPLOAD:
                        raise HTTPException(413, "El límite por video es 50 MB.")
                    stream.write(chunk)
            with target.open("rb") as stream:
                header = stream.read(32)
            if size < 12 or b"ftyp" not in header:
                raise HTTPException(415, "El archivo no tiene una cabecera MP4 válida.")
            store.add(job_id, filename)
        except BaseException:
            target.unlink(missing_ok=True)
            directory.rmdir()
            raise
        return detail(store.get(job_id))

    @app.get("/api/v1/analyses/{job_id}")
    def get_job(job_id: str):
        return detail(store.get(job_id))

    @app.get("/api/v1/analyses/{job_id}/frames")
    def get_frames(
        job_id: str, offset: int = Query(0, ge=0), limit: int = Query(30, ge=1, le=100)
    ):
        store.get(job_id)
        frames = records(store.directory / job_id)
        return {
            "items": [
                dict(frame, image_url=f"/api/v1/analyses/{job_id}/frames/{i}/image")
                for i, frame in enumerate(frames[offset : offset + limit], start=offset)
            ],
            "total": len(frames),
        }

    @app.get("/api/v1/analyses/{job_id}/frames/{index}/image")
    def image(job_id: str, index: int):
        store.get(job_id)
        if index < 0 or index >= MAX_FRAMES:
            raise HTTPException(404)
        path = store.directory / job_id / "result/frames" / f"{index:06d}.jpg"
        if not path.is_file():
            raise HTTPException(404, "Frame todavía no disponible")
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/api/v1/analyses/{job_id}/results")
    def download(job_id: str):
        job = store.get(job_id)
        path = store.directory / job_id / "result/detections.jsonl"
        if job["status"] not in ("completed", "failed") or not path.exists():
            raise HTTPException(409, "Los resultados todavía no están disponibles")
        return FileResponse(
            path, filename=f"{job_id}.jsonl", media_type="application/x-ndjson"
        )

    web = ROOT / "web/dist"
    if web.is_dir():
        app.mount("/", StaticFiles(directory=web, html=True), name="web")
    return app
