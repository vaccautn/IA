"""Reusable local video routes/UI; the principal API owns runtime resources."""

from contextlib import asynccontextmanager
from ipaddress import ip_address
import json
import shutil
import subprocess as subprocess  # Compatibility for existing worker callers.
from urllib.parse import urlsplit
from uuid import uuid4

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .policies import (
    DEFAULT_MAX_FRAMES as MAX_FRAMES,
    ROOT,
    WEB_DISK_RESERVE_BYTES as RESERVE,
    WEB_MAX_ACTIVE_JOBS,
    WEB_MAX_UPLOAD_BYTES as MAX_UPLOAD,
)
from .store import JobNotFound, Store
from .worker import Worker as Worker
from .schemas import VideoErrorResponse, FramePageResponse, JobResponse
from .service import UNAVAILABLE_DETAIL, VideoResources


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


def video_store(request: Request) -> Store:
    resources = getattr(request.app.state, "video", None)
    if resources is None or not resources.available:
        raise HTTPException(503, UNAVAILABLE_DETAIL)
    return resources.store


def create_router():
    router = APIRouter(responses={503: {"model": VideoErrorResponse}})

    def detail(store, job):
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

    @router.get(
        "/api/v1/analyses", response_model=list[JobResponse],
        response_model_exclude_unset=True,
    )
    def list_jobs(store: Store = Depends(video_store)):
        return [detail(store, job) for job in store.list()]

    @router.post(
        "/api/v1/analyses", status_code=202, response_model=JobResponse,
        response_model_exclude_unset=True,
        responses={code: {"model": VideoErrorResponse} for code in (413, 415, 429, 507)},
    )
    async def upload(
        request: Request, filename: str = Query(min_length=1, max_length=200),
        store: Store = Depends(video_store),
    ):
        if not filename.lower().endswith(".mp4"):
            raise HTTPException(415, "Elegí un video MP4.")
        with store.connect() as conn:
            active = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN ('pending','processing')"
            ).fetchone()[0]
        if active >= WEB_MAX_ACTIVE_JOBS:
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
        return detail(store, store.get(job_id))

    @router.get(
        "/api/v1/analyses/{job_id}", response_model=JobResponse,
        response_model_exclude_unset=True, responses={404: {"model": VideoErrorResponse}},
    )
    def get_job(job_id: str, store: Store = Depends(video_store)):
        return detail(store, store.get(job_id))

    @router.get(
        "/api/v1/analyses/{job_id}/frames", response_model=FramePageResponse,
        response_model_exclude_unset=True, responses={404: {"model": VideoErrorResponse}},
    )
    def get_frames(
        job_id: str, offset: int = Query(0, ge=0), limit: int = Query(30, ge=1, le=100),
        store: Store = Depends(video_store),
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

    @router.get("/api/v1/analyses/{job_id}/frames/{index}/image")
    def image(job_id: str, index: int, store: Store = Depends(video_store)):
        store.get(job_id)
        if index < 0 or index >= MAX_FRAMES:
            raise HTTPException(404)
        path = store.directory / job_id / "result/frames" / f"{index:06d}.jpg"
        if not path.is_file():
            raise HTTPException(404, "Frame todavía no disponible")
        return FileResponse(path, media_type="image/jpeg")

    @router.get("/api/v1/analyses/{job_id}/results")
    def download(job_id: str, store: Store = Depends(video_store)):
        job = store.get(job_id)
        path = store.directory / job_id / "result/detections.jsonl"
        if job["status"] not in ("completed", "failed") or not path.exists():
            raise HTTPException(409, "Los resultados todavía no están disponibles")
        return FileResponse(
            path, filename=f"{job_id}.jsonl", media_type="application/x-ndjson"
        )

    return router


def is_loopback(host):
    try:
        address = ip_address(host)
        return address.is_loopback or (
            address.version == 6 and address.ipv4_mapped is not None
            and address.ipv4_mapped.is_loopback
        )
    except ValueError:
        return False


class VideoLocalMiddleware:
    """Restrict only video data/UI; never gate image or BCS inference."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        root_path = scope.get("root_path", "")
        if root_path and (path == root_path or path.startswith(root_path + "/")):
            path = path[len(root_path):]
        protected = any(path == prefix or path.startswith(prefix + "/")
                        for prefix in ("/api/v1/analyses", "/videos"))
        if scope["type"] != "http" or not protected:
            return await self.app(scope, receive, send)
        request = Request(scope)
        try:
            host = urlsplit("http://" + request.headers.get("host", "")).hostname
        except ValueError:
            host = None
        if not host or not (host == "localhost" or is_loopback(host)):
            return await PlainTextResponse("Invalid host header", status_code=400)(scope, receive, send)
        if not request.client or not is_loopback(request.client.host):
            return await JSONResponse({"detail": "Acceso permitido solo desde esta computadora"}, status_code=403)(scope, receive, send)
        if any(name in request.headers for name in (
            "forwarded", "x-forwarded-for", "x-forwarded-host", "x-forwarded-proto"
        )):
            return await JSONResponse({"detail": "El acceso a video no admite proxies"}, status_code=403)(scope, receive, send)
        origin = request.headers.get("origin")
        base = urlsplit(str(request.base_url))
        if origin and origin != f"{base.scheme}://{base.netloc}":
            return await JSONResponse({"detail": "Origen no permitido"}, status_code=403)(scope, receive, send)
        resources = getattr(request.app.state, "video", None)
        if request.method == "POST" and path.startswith("/api/v1/analyses") and resources:
            async with resources.upload_lock:
                return await self.app(scope, receive, send)
        return await self.app(scope, receive, send)


class VideoUI:
    def __init__(self, directory):
        self.directory = directory
        self.static = StaticFiles(directory=directory, html=True, check_dir=False)

    async def __call__(self, scope, receive, send):
        if not (self.directory / "index.html").is_file():
            return await JSONResponse({"detail": "La interfaz de video no está compilada. Ejecutar npm run build desde IA/web."}, status_code=503)(scope, receive, send)
        return await self.static(scope, receive, send)


def configure_video(app, web_directory=None):
    app.include_router(create_router())
    app.add_middleware(VideoLocalMiddleware)

    @app.exception_handler(JobNotFound)
    async def job_not_found(request, exc):
        return JSONResponse({"detail": "Análisis no encontrado"}, status_code=404)

    @app.get("/videos", include_in_schema=False)
    def videos_redirect():
        return RedirectResponse("/videos/", status_code=307)

    app.mount("/videos", VideoUI(web_directory or ROOT / "web/dist"), name="videos")


def create_app(directory=None, start_worker=True):
    """Small contract-test composition; production starts vacca_api.main only."""
    resources = VideoResources(directory, worker_factory=Worker)
    resources.prepare()
    if not start_worker:
        resources.available = True

    @asynccontextmanager
    async def lifespan(app):
        resources.start(start_worker=start_worker)
        try:
            yield
        finally:
            resources.close()

    app = FastAPI(lifespan=lifespan)
    app.state.video = resources
    app.state.store = resources.store  # Existing contract-test compatibility.
    configure_video(app)
    return app
