"""Bridge the CLI worker to the owning API loop; never load a third model."""

import asyncio
from concurrent.futures import TimeoutError as FutureTimeout
from concurrent.futures import CancelledError as FutureCancelled
import logging
from time import monotonic

from fastapi import HTTPException

from vacca_vision import ImageValidationError, validate_image_bytes
from .artifacts import BCS_WRITE_ERROR, job_artifact, publish_summary, read_summary
from .candidate import CANDIDATE_ARTIFACT, POLICY
from .schemas import CandidateResponse, reject_candidate

logger = logging.getLogger(__name__)
BCS_WAIT_SECONDS = 60


def candidate_input(root, job_id):
    summary = read_summary(root, job_id)
    if summary.get("status") != "completed" or summary.get("stop_reason") not in {"frame_limit", "end_of_source"}:
        return summary, None, "video_not_completed"
    if "candidate" not in summary:
        return summary, None, "legacy_no_candidate"
    if summary["candidate"] is None:
        if summary.get("candidate_rejection_reason") in {"invalid_candidate", "candidate_write_failed"}:
            return summary, None, summary["candidate_rejection_reason"]
        return summary, None, "no_eligible_frame"
    candidate = CandidateResponse.model_validate(summary["candidate"])
    if summary.get("selection_policy") != POLICY:
        raise ValueError("Invalid selection policy")
    with job_artifact(root, job_id, CANDIDATE_ARTIFACT).open("rb") as stream:
        image = stream.read(10 * 1024 * 1024 + 1)
    metadata = validate_image_bytes(image)
    if not image.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Candidate encoding does not match PNG metadata")
    if (metadata.width, metadata.height) != (candidate.image_width, candidate.image_height):
        raise ValueError("Candidate dimensions do not match")
    return summary, image, None


class VideoBCSCoordinator:
    def __init__(self, loop, evaluate, max_wait=BCS_WAIT_SECONDS):
        self.loop = loop
        self.evaluate = evaluate
        self.max_wait = max_wait
        self.tasks = set()
        self.closing = False

    async def outcome(self, image):
        task = asyncio.current_task()
        self.tasks.add(task)
        try:
            if self.closing:
                return dict(status="not_evaluated", reason="shutdown")
            try:
                result = await self.evaluate(image, internal=True)
            except HTTPException as exc:
                if exc.status_code == 400:
                    status, reason = "invalid_input", "image_rejected"
                elif exc.status_code == 503:
                    status = "busy" if exc.detail == "Inference capacity is busy; retry shortly" else "unavailable"
                    reason = "capacity_busy" if status == "busy" else "capability_unavailable"
                else:
                    status, reason = "failed", "inference_failed"
                return dict(status=status, reason=reason)
            except Exception as exc:
                logger.error("Video BCS failed: %s", type(exc).__name__)
                return dict(status="failed", reason="inference_failed")
            return dict(status="completed", reason=None,
                        bcs_category=result.bcs_category, model_status=result.model_status,
                        package_id=result.package_id, inference_time_ms=result.inference_time_ms,
                        lineage=getattr(result, "lineage", None))
        finally:
            self.tasks.discard(task)

    def finish(self, root, job_id, stop):
        """Runs on the worker thread; only bounded waiting occurs here."""
        if stop.is_set() or self.closing:
            raise RuntimeError("El servicio se interrumpió antes de completar BCS.")
        try:
            summary, image, reason = candidate_input(root, job_id)
        except (OSError, ValueError, ImageValidationError):
            try:
                summary = read_summary(root, job_id)
            except (OSError, ValueError):
                return BCS_WRITE_ERROR
            summary = reject_candidate(summary)
            image, reason = None, "invalid_candidate"
        if reason == "legacy_no_candidate":
            return None  # Older jobs are never retroactively evaluated.
        if image is None:
            outcome = dict(status="invalid_input" if reason == "invalid_candidate" else "not_evaluated", reason=reason)
        else:
            future = asyncio.run_coroutine_threadsafe(self.outcome(image), self.loop)
            deadline = monotonic() + self.max_wait
            try:
                while True:
                    if stop.is_set() or self.closing:
                        raise RuntimeError("El servicio se interrumpió antes de completar BCS.")
                    try:
                        outcome = future.result(timeout=min(0.1, max(0.001, deadline - monotonic())))
                        break
                    except FutureTimeout:
                        if monotonic() >= deadline:
                            outcome = dict(status="not_evaluated", reason="evaluation_timeout")
                            break
                    except FutureCancelled:
                        raise RuntimeError("El servicio se interrumpió antes de completar BCS.") from None
            finally:
                if not future.done():
                    future.cancel()
        if stop.is_set() or self.closing:
            raise RuntimeError("El servicio se interrumpió antes de completar BCS.")
        summary["bcs"] = outcome
        try:
            publish_summary(root, job_id, summary)
        except (OSError, ValueError) as exc:
            logger.error("Video BCS result not saved: %s", type(exc).__name__)
            return BCS_WRITE_ERROR
        return None

    async def close(self):
        self.closing = True
        for task in tuple(self.tasks):
            task.cancel()
        if self.tasks:
            await asyncio.gather(*tuple(self.tasks), return_exceptions=True)
