"""Framework-neutral, one-frame selection over existing sampled detections."""

from copy import deepcopy
import math

POLICY = "exactly_one_detected_cow_highest_confidence_first_tie"
CANDIDATE_ARTIFACT = "candidate.png"


class CandidateSelector:
    def __init__(self):
        self.image = None
        self.metadata = None

    def consider(self, frame, result, sample_ordinal):
        detections = result.get("detections", [])
        if (len(detections) != 1 or result.get("detection_count") != 1
                or result.get("cow_detected") is not True):
            return
        cow = detections[0]
        confidence = cow.get("confidence")
        if (cow.get("class_name") != "cow" or isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not math.isfinite(confidence) or not 0 <= confidence <= 1):
            return
        if self.metadata is not None and confidence <= self.metadata["detector_confidence"]:
            return
        self.image = frame.image.copy()
        self.metadata = dict(
            frame_index=frame.index, sample_ordinal=sample_ordinal,
            timestamp_ms=frame.timestamp_ms, timestamp_source=frame.timestamp_source,
            detector_confidence=confidence, target_index=0,
            original_bbox=deepcopy({key: cow[key] for key in ("x1", "y1", "x2", "y2", "bbox")}),
        )
