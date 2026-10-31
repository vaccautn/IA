"""Incremental JSONL/video plus a bounded, offline HTML presentation."""

from __future__ import annotations

import json
import logging
from importlib.resources import files
from pathlib import Path

from .pipeline import Frame
from .policies import CLI_PREVIEW_LIMIT
from .candidate import CANDIDATE_ARTIFACT, POLICY, CandidateSelector
from .artifacts import publish_initial_summary

logger = logging.getLogger(__name__)


def json_text(value) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False)


class PresentationSink:
    def __init__(
        self,
        output: Path,
        source: dict,
        sample_fps: float,
        preview_limit: int = CLI_PREVIEW_LIMIT,
        write_video: bool = True,
    ) -> None:
        if not 1 <= preview_limit <= 1000:
            raise ValueError("preview_limit debe estar entre 1 y 1000")
        self.output = Path(output)
        self.source = source
        self.sample_fps = min(sample_fps, source["fps"])
        self.preview_limit = preview_limit
        self.write_video = write_video
        self.preview_count = 0
        self.video = None
        self.records = None
        self.preview = None
        self.candidate = CandidateSelector()
        self.sample_ordinal = 0
        scale = min(1.0, 960 / source["width"])
        # MPEG-4 codecs generally need even dimensions.
        self.size = (
            max(2, int(source["width"] * scale) // 2 * 2),
            max(2, int(source["height"] * scale) // 2 * 2),
        )

    def __enter__(self):
        import cv2

        self.output.mkdir(parents=True, exist_ok=False)
        try:
            (self.output / "frames").mkdir()
            self.records = (self.output / "detections.jsonl").open(
                "w", encoding="utf-8"
            )
            self.preview = (self.output / "frames.js").open("w", encoding="utf-8")
            self.preview.write("window.VACCA_FRAMES = [\n")
            if self.write_video:
                self.video = cv2.VideoWriter(
                    str(self.output / "annotated.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    self.sample_fps,
                    self.size,
                )
                if not self.video.isOpened():
                    raise ValueError(
                        "El codec MP4 no está disponible; usar --no-video para HTML/JSON"
                    )
            return self
        except BaseException:
            self.close()
            raise

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        if self.video is not None:
            self.video.release()
            self.video = None
        if self.records is not None:
            self.records.close()
            self.records = None
        if self.preview is not None:
            self.preview.write("\n];\n")
            self.preview.close()
            self.preview = None

    def write(self, frame: Frame, result: dict) -> None:
        import cv2

        self.candidate.consider(frame, result, self.sample_ordinal)
        self.sample_ordinal += 1
        annotated = cv2.resize(frame.image, self.size)
        scale_x = self.size[0] / result["image_width"]
        scale_y = self.size[1] / result["image_height"]
        for detection in result["detections"]:
            bbox = detection["bbox"]
            x1 = round(
                (bbox["x_center"] - bbox["width"] / 2) * result["image_width"] * scale_x
            )
            y1 = round(
                (bbox["y_center"] - bbox["height"] / 2)
                * result["image_height"]
                * scale_y
            )
            x2 = round(
                (bbox["x_center"] + bbox["width"] / 2) * result["image_width"] * scale_x
            )
            y2 = round(
                (bbox["y_center"] + bbox["height"] / 2)
                * result["image_height"]
                * scale_y
            )
            x1, x2 = (max(0, min(self.size[0] - 1, x)) for x in (x1, x2))
            y1, y2 = (max(0, min(self.size[1] - 1, y)) for y in (y1, y2))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (83, 224, 142), 2)
            cv2.putText(
                annotated,
                f"vaca {detection['confidence']:.0%}",
                (x1, max(18, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (83, 224, 142),
                2,
            )
        cv2.rectangle(
            annotated,
            (0, self.size[1] - 32),
            (self.size[0], self.size[1]),
            (22, 32, 28),
            -1,
        )
        cv2.putText(
            annotated,
            f"VACCA | t={frame.timestamp_ms / 1000:.2f}s | visibles={result['detection_count']} | frame={frame.index}",
            (10, self.size[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (240, 245, 242),
            1,
        )
        if self.video is not None:
            self.video.write(annotated)
        if self.preview_count < self.preview_limit:
            relative = f"frames/{self.preview_count:06d}.jpg"
            if not cv2.imwrite(str(self.output / relative), annotated):
                raise OSError("No se pudo guardar el frame de presentación")
            preview = {
                "image": relative,
                "timestamp_ms": result["timestamp_ms"],
                "frame_index": frame.index,
                "detection_count": result["detection_count"],
                "inference_time_ms": result["inference_time_ms"],
            }
            if self.preview_count:
                self.preview.write(",\n")
            self.preview.write(json_text(preview))
            self.preview_count += 1
        self.records.write(json_text(result) + "\n")
        self.records.flush()

    def finalize(self, summary: dict) -> None:
        """Close the MP4/JS before publishing a report, including partial failures."""
        self.close()
        selected = self.candidate.metadata
        candidate_write_failed = False
        if selected is not None:
            import cv2

            path = self.output / CANDIDATE_ARTIFACT
            try:
                if not cv2.imwrite(str(path), self.candidate.image):
                    raise OSError("Could not save original candidate image")
            except (cv2.error, OSError, TypeError, ValueError) as exc:
                logger.warning("Original candidate persistence failed: %s", type(exc).__name__)
                selected = None
                candidate_write_failed = True
                try:
                    path.unlink(missing_ok=True)
                except OSError as cleanup_error:
                    logger.warning("Partial candidate cleanup failed: %s", type(cleanup_error).__name__)
            else:
                selected = dict(
                    selected, artifact=CANDIDATE_ARTIFACT, encoding="PNG",
                    image_width=self.candidate.image.shape[1],
                    image_height=self.candidate.image.shape[0],
                    detector_model=summary.get("model"),
                )
        if candidate_write_failed:
            summary["candidate_rejection_reason"] = "candidate_write_failed"
        summary.update(
            selection_policy=POLICY, candidate=selected,
            bcs=dict(status="not_evaluated", reason=(
                "video_not_completed" if summary.get("status") != "completed" else
                "candidate_write_failed" if candidate_write_failed else
                "web_api_only" if selected else "no_eligible_frame"
            )),
        )
        summary.update(
            preview_frames=self.preview_count,
            preview_limit=self.preview_limit,
            preview_truncated=summary.get("frames_analyzed", 0) > self.preview_count,
            annotated_video="annotated.mp4" if self.write_video else None,
            annotated_video_fps=self.sample_fps if self.write_video else None,
            count_semantics="visible_cows_per_frame_not_unique_animals",
        )
        report = files("vacca_video").joinpath("report.html").read_text(encoding="utf-8")
        (self.output / "report.html").write_text(report, encoding="utf-8")
        publish_initial_summary(self.output, summary)
