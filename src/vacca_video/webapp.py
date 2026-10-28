"""Reusable local video routes/UI; the principal API owns runtime resources."""

from contextlib import asynccontextmanager
from ipaddress import ip_address
import json
import shutil
import subprocess as subprocess  # Compatibility for existing worker callers.
from urllib.parse import urlsplit
from uuid import uuid4

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse, StreamingResponse
import tempfile
import zipfile
from fastapi.staticfiles import StaticFiles
from pydantic import TypeAdapter, ValidationError
from pydantic_core import PydanticSerializationError

from .policies import (
    DEFAULT_MAX_FRAMES as MAX_FRAMES,
    ROOT,
    WEB_DISK_RESERVE_BYTES as RESERVE,
    WEB_MAX_ACTIVE_JOBS,
    WEB_MAX_UPLOAD_BYTES as MAX_UPLOAD,
)
from .store import JobNotFound, Store
from .worker import Worker as Worker
from .schemas import (
    AnalysisSummaryResponse,
    VideoErrorResponse,
    FramePageResponse,
    JobResponse,
    TrackListResponse,
    sanitize_optional_candidate,
)
from .service import UNAVAILABLE_DETAIL, VideoResources
from .artifacts import (
    BCS_WRITE_ERROR,
    TRACK_MEDIA_TYPES,
    frame_artifact,
    job_artifact,
    read_summary,
    read_tracking_json,
    tracking_artifact,
)
from .candidate import CANDIDATE_ARTIFACT
from .tracking_job import TRACKING_STATUS

SUMMARY_UNAVAILABLE_DETAIL = (
    "No se pudo leer el resumen guardado; las detecciones y el historial siguen disponibles."
)
_JOB_RESPONSE_LIST_ADAPTER = TypeAdapter(list[JobResponse])


def _ensure_job_response_serializable(response):
    """Preflight the same JSON serializer and list wrapper used by FastAPI."""
    models = _JOB_RESPONSE_LIST_ADAPTER.validate_python([response])
    _JOB_RESPONSE_LIST_ADAPTER.dump_json(
        models, by_alias=True, exclude_unset=True, ensure_ascii=False
    )


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
    if resources is None or not resources.history_available:
        raise HTTPException(503, UNAVAILABLE_DETAIL)
    return resources.store


def highlighted_frame(frames, job_id):
    """Return the first frame maximizing count then mean confidence."""
    best = None
    best_rank = None
    for index, frame in enumerate(frames):
        if type(frame) is not dict:
            continue
        detections = frame.get("detections", [])
        if type(detections) is not list or any(type(item) is not dict for item in detections):
            continue
        scores = [item.get("confidence") for item in detections]
        if any(type(score) not in (int, float) or not 0 <= score <= 1 for score in scores):
            continue
        count = frame.get("detection_count")
        if type(count) is not int or count < 0:
            continue
        mean = sum(scores) / len(scores) if scores else None
        rank = (count, mean if mean is not None else -1)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best = dict(frame, sample_index=index, mean_confidence=mean,
                        image_url=f"/api/v1/analyses/{job_id}/frames/{index}/image")
    return best


TRACKS_SCHEMA_VERSION = "vacca-video-tracks-v1"
TRACKING_FAILURE_REASONS = {
    "interrupted", "timeout", "process_failed", "summary_missing", "disabled_by_configuration",
}
TRACKING_SUMMARY_FIELDS = (
    "identity_semantics", "tracker", "model_sha256", "fps", "width", "height", "frames_read",
    "frames_sampled", "stop_reason", "config", "warnings", "notes",
)
# Disk paths stay private; the API exposes job-scoped URLs instead.
TRACK_PRIVATE_FIELDS = ("clip", "best_original", "best_crop")


def _track_entry(job_id, item):
    if type(item) is not dict or type(item.get("track_id")) is not int or item["track_id"] < 0:
        raise ValueError("Invalid track entry")
    track_id = item["track_id"]
    prefix = f"/api/v1/analyses/{job_id}/tracks/{track_id}"
    clip = item.get("clip")
    clip_name = clip.rsplit("/", 1)[-1] if type(clip) is str else None
    entry = {key: value for key, value in item.items() if key not in TRACK_PRIVATE_FIELDS}
    entry.update(
        clip_url=f"{prefix}/clip" if clip_name in ("clip.webm", "clip.mp4") else None,
        clip_media_type=TRACK_MEDIA_TYPES.get(clip_name) if clip_name in ("clip.webm", "clip.mp4") else None,
        best_crop_url=f"{prefix}/best-crop",
        best_original_url=f"{prefix}/best-original",
    )
    return entry


def tracks_detail(root, job):
    """Describe a job's tracks; tracking problems never surface as HTTP errors."""
    job_id = job["id"]
    response = dict(schema_version=TRACKS_SCHEMA_VERSION, tracks=[])
    if job["status"] in ("pending", "processing"):
        return dict(response, status="pending")
    if job["status"] != "completed":
        return dict(response, status="unavailable", reason="video_not_completed")
    try:
        recorded = read_tracking_json(root, job_id, TRACKING_STATUS)
    except FileNotFoundError:
        return dict(response, status="unavailable", reason="not_tracked")
    except (OSError, UnicodeError, ValueError, RecursionError):
        return dict(response, status="failed", reason="status_unreadable")
    state = recorded.get("status") if type(recorded) is dict else None
    reason = recorded.get("reason") if type(recorded) is dict else None
    reason = reason if reason in TRACKING_FAILURE_REASONS else "unknown"
    if state == "running":
        return dict(response, status="pending")
    if state == "disabled":
        return dict(response, status="disabled", reason=reason)
    if state != "completed":
        return dict(response, status="failed", reason=reason)
    try:
        summary = read_tracking_json(root, job_id, "summary.json")
        if type(summary) is not dict or type(summary.get("tracks")) is not list:
            raise ValueError("Invalid tracking summary")
        response.update({key: summary[key] for key in TRACKING_SUMMARY_FIELDS if key in summary})
        response.update(status="completed",
                        tracks=[_track_entry(job_id, item) for item in summary["tracks"]])
        # Preflight the serializer so one bad artifact cannot become a 500.
        TrackListResponse.model_validate(response).model_dump_json()
    except (OSError, UnicodeError, ValueError, ValidationError, PydanticSerializationError,
            RecursionError):
        return dict(schema_version=TRACKS_SCHEMA_VERSION, tracks=[], status="failed",
                    reason="summary_unreadable")
    return response


def video_admission_store(request: Request) -> Store:
    resources = getattr(request.app.state, "video", None)
    if resources is None or not resources.available:
        raise HTTPException(503, UNAVAILABLE_DETAIL)
    return resources.store


def create_router():
    router = APIRouter(responses={503: {"model": VideoErrorResponse}})

    def detail(store, job):
        directory = store.directory / job["id"]
        frames = records(directory)
        highlight = None
        highlight_error = False
        if job["status"] in ("completed", "failed"):
            try:
                highlight = highlighted_frame(frames, job["id"])
            except (KeyError, TypeError, ValueError, OverflowError, RecursionError):
                highlight_error = True
        summary = None
        summary_error = None
        if job["status"] in ("completed", "failed"):
            try:
                raw_summary = read_summary(store.directory, job["id"])
                if type(raw_summary) is not dict:
                    raise ValueError("Summary must be an object")
                summary = sanitize_optional_candidate(raw_summary)
                if "bcs" in summary and job["status"] != "completed":
                    summary["bcs"] = dict(status="not_evaluated", reason="video_not_completed")
                elif job.get("error") == BCS_WRITE_ERROR:
                    summary["bcs"] = dict(status="failed", reason="metadata_write_failed")
                AnalysisSummaryResponse.model_validate(summary)
            except (OSError, UnicodeError, ValueError, ValidationError, RecursionError):
                summary = None
                summary_error = SUMMARY_UNAVAILABLE_DETAIL
        response = dict(
            job,
            frames_analyzed=len(frames),
            summary=summary,
            highlighted_frame=None if highlight_error else highlight,
            schema_version="vacca-video-job-v1",
        )
        if summary_error is not None:
            response["summary_error"] = summary_error
        try:
            _ensure_job_response_serializable(response)
        except (
            ValidationError,
            PydanticSerializationError,
            UnicodeError,
            ValueError,
            RecursionError,
        ):
            if highlight is not None and not highlight_error:
                response["highlighted_frame"] = None
                highlight_error = True
                try:
                    _ensure_job_response_serializable(response)
                    return response
                except (ValidationError, PydanticSerializationError, UnicodeError,
                        ValueError, RecursionError):
                    pass
            if summary is None:
                raise
            response["summary"] = None
            response["summary_error"] = SUMMARY_UNAVAILABLE_DETAIL
            # If the job fields themselves are not renderable, let that error
            # surface rather than disguising it as a corrupt summary.
            _ensure_job_response_serializable(response)
        return response

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
        openapi_extra={
            "requestBody": {
                "required": True,
                # QuickTime (.mov, phone recordings) shares the MP4 ISO box layout.
                "content": {media: {"schema": {"type": "string", "format": "binary"}}
                            for media in ("video/mp4", "video/quicktime")},
            },
        },
    )
    async def upload(
        request: Request, filename: str = Query(min_length=1, max_length=200),
        store: Store = Depends(video_admission_store),
    ):
        if not filename.lower().endswith((".mp4", ".mov")):
            raise HTTPException(415, "Elegí un video MP4 o MOV.")
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
            if not request.app.state.video.add_job_if_available(job_id, filename):
                raise HTTPException(503, UNAVAILABLE_DETAIL)
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

    @router.get(
        "/api/v1/analyses/{job_id}/frames/{index}/image", response_class=FileResponse,
        responses={200: {
            "description": "Annotated JPEG; attachment only when download=true, otherwise no Content-Disposition header.",
            "content": {"image/jpeg": {"schema": {"type": "string", "format": "binary"}}},
            "headers": {"Content-Disposition": {
                "description": "Present only with download=true: attachment; filename=frame-{job_id}-{index+1}.jpg",
                "schema": {"type": "string"},
            }},
        }},
    )
    def image(job_id: str, index: int, download: bool = False,
              store: Store = Depends(video_store)):
        store.get(job_id)
        if index < 0 or index >= MAX_FRAMES:
            raise HTTPException(404)
        try:
            path = frame_artifact(store.directory, job_id, index)
            if not path.is_file():
                raise ValueError("Missing frame")
        except ValueError:
            raise HTTPException(404, "Frame todavía no disponible")
        return FileResponse(
            path, media_type="image/jpeg",
            filename=f"frame-{job_id}-{index + 1}.jpg" if download else None,
        )

    @router.get(
        "/api/v1/analyses/{job_id}/frames.zip", response_class=StreamingResponse,
        responses={200: {
            "description": "ZIP of available annotated frames from a completed or failed analysis.",
            "content": {"application/zip": {"schema": {"type": "string", "format": "binary"}}},
            "headers": {"Content-Disposition": {
                "description": "attachment; filename=frames-{job_id}.zip",
                "schema": {"type": "string"},
            }},
        }},
    )
    def download_frames(job_id: str, store: Store = Depends(video_store)):
        job = store.get(job_id)
        if job["status"] not in ("completed", "failed"):
            raise HTTPException(409, "Esperá a que termine el análisis")
        try:
            paths = [frame_artifact(store.directory, job_id, index)
                     for index in range(MAX_FRAMES)]
            paths = [path for path in paths if path.is_file()]
        except ValueError:
            raise HTTPException(404, "Frames unavailable") from None
        if not paths:
            raise HTTPException(409, "No hay frames disponibles para descargar")
        try:
            if shutil.disk_usage(store.directory).free < RESERVE + sum(p.stat().st_size for p in paths):
                raise HTTPException(507, "No hay espacio suficiente para preparar el archivo ZIP.")
        except OSError:
            raise HTTPException(507, "No se pudo reservar espacio para el archivo ZIP.") from None
        try:
            archive = tempfile.SpooledTemporaryFile(max_size=1024 * 1024)
        except (OSError, MemoryError):
            raise HTTPException(507, "No se pudo preparar el archivo ZIP.") from None
        try:
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
                for path in paths:
                    bundle.write(path, arcname=path.name)
            archive.seek(0)
        except (OSError, MemoryError):
            archive.close()
            raise HTTPException(507, "No se pudo preparar el archivo ZIP.") from None
        except BaseException:
            archive.close()
            raise

        def chunks():
            try:
                while chunk := archive.read(64 * 1024):
                    yield chunk
            finally:
                archive.close()

        class ArchiveResponse(StreamingResponse):
            async def __call__(self, scope, receive, send):
                try:
                    await super().__call__(scope, receive, send)
                finally:
                    archive.close()

        return ArchiveResponse(
            chunks(), media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="frames-{job_id}.zip"'},
        )

    @router.get(
        "/api/v1/analyses/{job_id}/candidate/image", response_class=FileResponse,
        responses={200: {
            "description": "Original candidate PNG, without a Content-Disposition header.",
            "content": {"image/png": {"schema": {"type": "string", "format": "binary"}}},
        }},
    )
    def candidate_image(job_id: str, store: Store = Depends(video_store)):
        store.get(job_id)
        try:
            path = job_artifact(store.directory, job_id, CANDIDATE_ARTIFACT)
            if not path.is_file():
                raise ValueError("Missing candidate")
        except ValueError:
            raise HTTPException(404, "Imagen candidata no disponible") from None
        return FileResponse(path, media_type="image/png")

    @router.get(
        "/api/v1/analyses/{job_id}/results", response_class=FileResponse,
        responses={200: {
            "description": "NDJSON download: one frame record per line, not a single JSON object; may contain partial results from a failed analysis.",
            "content": {"application/x-ndjson": {"schema": {"type": "string", "format": "binary"}}},
            "headers": {"Content-Disposition": {
                "description": "attachment; filename={job_id}.jsonl",
                "schema": {"type": "string"},
            }},
        }},
    )
    def download(job_id: str, store: Store = Depends(video_store)):
        job = store.get(job_id)
        path = store.directory / job_id / "result/detections.jsonl"
        if job["status"] not in ("completed", "failed") or not path.exists():
            raise HTTPException(409, "Los resultados todavía no están disponibles")
        return FileResponse(
            path, filename=f"{job_id}.jsonl", media_type="application/x-ndjson"
        )

    @router.get(
        "/api/v1/analyses/{job_id}/tracks", response_model=TrackListResponse,
        response_model_exclude_unset=True, responses={404: {"model": VideoErrorResponse}},
    )
    def get_tracks(job_id: str, store: Store = Depends(video_store)):
        return tracks_detail(store.directory, store.get(job_id))

    def track_file(store, job_id, track_id, names):
        job = store.get(job_id)
        try:
            recorded = read_tracking_json(store.directory, job_id, TRACKING_STATUS)
        except (OSError, UnicodeError, ValueError, RecursionError):
            recorded = None
        # Interrupted runs may leave partial clips; only finished tracking is served.
        if job["status"] != "completed" or type(recorded) is not dict or recorded.get("status") != "completed":
            raise HTTPException(404, "Trayectoria no disponible")
        for name in names:
            try:
                path = tracking_artifact(store.directory, job_id, name, track_id)
            except ValueError:
                raise HTTPException(404, "Trayectoria no disponible") from None
            if path.is_file():
                return FileResponse(path, media_type=TRACK_MEDIA_TYPES[name])
        raise HTTPException(404, "Trayectoria no disponible")

    def binary(description, *media_types):
        return {200: {
            "description": description,
            "content": {media: {"schema": {"type": "string", "format": "binary"}}
                        for media in media_types},
        }, 404: {"model": VideoErrorResponse}}

    @router.get(
        "/api/v1/analyses/{job_id}/tracks/{track_id}/clip", response_class=FileResponse,
        responses=binary("Clip 320x320 of one trajectory; samples without detection are labelled grey frames.",
                         "video/webm", "video/mp4"),
    )
    def track_clip(job_id: str, track_id: int, store: Store = Depends(video_store)):
        return track_file(store, job_id, track_id, ("clip.webm", "clip.mp4"))

    @router.get(
        "/api/v1/analyses/{job_id}/tracks/{track_id}/best-crop", response_class=FileResponse,
        responses=binary("Padded rectangular crop of the selected frame; it may include other animals.",
                         "image/png"),
    )
    def track_best_crop(job_id: str, track_id: int, store: Store = Depends(video_store)):
        return track_file(store, job_id, track_id, ("best-crop.png",))

    @router.get(
        "/api/v1/analyses/{job_id}/tracks/{track_id}/best-original", response_class=FileResponse,
        responses=binary("Original full frame selected for the trajectory, without annotations.",
                         "image/jpeg"),
    )
    def track_best_original(job_id: str, track_id: int, store: Store = Depends(video_store)):
        return track_file(store, job_id, track_id, ("best-original.jpg",))

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
