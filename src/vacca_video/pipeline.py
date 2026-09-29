"""Incremental processing with injectable sources, detectors and output sinks."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Callable, Iterable, Protocol


@dataclass(frozen=True)
class Frame:
    index: int
    timestamp_ms: float
    image: Any
    timestamp_source: str = "decoder"


@dataclass(frozen=True)
class SamplingConfig:
    sample_fps: float = 5.0
    max_frames: int | None = 300

    def __post_init__(self) -> None:
        if not math.isfinite(self.sample_fps) or not 0 < self.sample_fps <= 120:
            raise ValueError("sample_fps debe estar entre 0 y 120 (sin incluir 0)")
        if self.max_frames is not None and (
            isinstance(self.max_frames, bool)
            or not isinstance(self.max_frames, int)
            or self.max_frames < 1
        ):
            raise ValueError("max_frames debe ser positivo o None")


class FrameDetector(Protocol):
    def detect(self, image: Any) -> dict[str, Any]: ...


class ResultSink(Protocol):
    def write(self, frame: Frame, result: dict[str, Any]) -> None: ...


class TimestampSampler:
    """Select at most one decoded frame per requested temporal interval."""

    def __init__(self, fps: float) -> None:
        self.interval_ms = 1000 / fps
        self.next_ms = 0.0

    def accepts(self, timestamp_ms: float) -> bool:
        if timestamp_ms + 1e-6 < self.next_ms:
            return False
        self.next_ms = (
            math.floor((timestamp_ms + 1e-6) / self.interval_ms) + 1
        ) * self.interval_ms
        return True


def process_frames(
    frames: Iterable[Frame],
    detector: FrameDetector,
    sink: ResultSink,
    config: SamplingConfig,
    summary: dict[str, Any],
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    """Stream results without retaining the full video or detection history.

    Resource lifetime belongs to the caller. On failure/interrupt, summary keeps
    partial statistics and the exception propagates so it cannot look successful.
    """
    sampler = TimestampSampler(config.sample_fps)
    started = perf_counter()
    summary.update(
        status="processing",
        config=asdict(config),
        frames_read=0,
        frames_analyzed=0,
        frames_with_cows=0,
        max_cows_in_frame=0,
        inference_ms_total=0.0,
        timestamp_fallback_frames=0,
        last_timestamp_ms=None,
        stop_reason=None,
    )
    previous_time = -1.0
    previous_index = -1
    try:
        for frame in frames:
            if (
                not math.isfinite(frame.timestamp_ms)
                or frame.timestamp_ms < 0
                or frame.timestamp_ms <= previous_time
                or frame.index <= previous_index
            ):
                raise ValueError("Frames fuera de orden o con tiempo inválido")
            previous_time, previous_index = frame.timestamp_ms, frame.index
            summary["frames_read"] += 1
            summary["last_timestamp_ms"] = round(frame.timestamp_ms, 3)
            if frame.timestamp_source != "decoder":
                summary["timestamp_fallback_frames"] += 1
            if not sampler.accepts(frame.timestamp_ms):
                continue
            result = detector.detect(frame.image)
            result = {
                **result,
                "schema_version": "vacca-video-frame-v1",
                "processing_id": summary["processing_id"],
                "model_version": summary["model"]["name"],
                "frame_index": frame.index,
                "timestamp_ms": round(frame.timestamp_ms, 3),
                "timestamp_source": frame.timestamp_source,
            }
            sink.write(frame, result)
            count = result["detection_count"]
            summary["frames_analyzed"] += 1
            summary["frames_with_cows"] += int(count > 0)
            summary["max_cows_in_frame"] = max(summary["max_cows_in_frame"], count)
            summary["inference_ms_total"] += result["inference_time_ms"]
            if progress is not None:
                progress(summary)
            if (
                config.max_frames is not None
                and summary["frames_analyzed"] >= config.max_frames
            ):
                summary["stop_reason"] = "frame_limit"
                break
        else:
            summary["stop_reason"] = "end_of_source"
        if not summary["frames_analyzed"]:
            raise ValueError("El video no contiene frames decodificables")
        summary["status"] = "completed"
    except KeyboardInterrupt:
        summary.update(status="cancelled", stop_reason="interrupted")
        raise
    except Exception as exc:
        summary.update(status="failed", stop_reason="error", error=str(exc))
        raise
    finally:
        elapsed = perf_counter() - started
        analyzed = summary["frames_analyzed"]
        summary["processing_wall_seconds"] = round(elapsed, 3)
        summary["processing_fps"] = round(analyzed / elapsed, 3) if elapsed else 0
        summary["mean_inference_ms"] = (
            round(summary["inference_ms_total"] / analyzed, 3) if analyzed else None
        )
        summary["inference_ms_total"] = round(summary["inference_ms_total"], 3)
