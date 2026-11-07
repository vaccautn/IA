"""Small thread-safe operational counters for the prototype API."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import threading
from typing import Literal

Capability = Literal["detect", "bcs"]
SERVICE_IMPACTING_FAILURE_RATE_REVIEW_THRESHOLDS = (0.01, 0.02, 0.05)


@dataclass(slots=True)
class CapabilityMetricSnapshot:
    requests: int = 0
    client_rejections: int = 0
    busy_rejections: int = 0
    server_runtime_failures: int = 0
    inference_attempts: int = 0
    inference_successes: int = 0
    inference_failures: int = 0
    request_wall_time_ms_total: float = 0.0
    last_request_wall_time_ms: float | None = None
    non_inference_wall_time_ms_total: float = 0.0
    last_non_inference_wall_time_ms: float | None = None
    successful_inference_ms: float = 0.0
    failed_inference_ms: float = 0.0
    last_successful_inference_ms: float | None = None
    last_failed_inference_ms: float | None = None


class PrototypeMetrics:
    """Keep bounded, path-free counters for each inference capability."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._measurement_window_started_at_utc = datetime.now(timezone.utc).isoformat()
        self._values = {
            "detect": CapabilityMetricSnapshot(),
            "bcs": CapabilityMetricSnapshot(),
        }

    def request(self, capability: Capability) -> None:
        with self._lock:
            self._values[capability].requests += 1

    def client_rejection(self, capability: Capability, request_wall_ms: float) -> None:
        with self._lock:
            value = self._values[capability]
            value.client_rejections += 1
            self._record_wall(value, request_wall_ms, 0.0)

    def server_runtime_failure(self, capability: Capability, request_wall_ms: float) -> None:
        with self._lock:
            value = self._values[capability]
            value.server_runtime_failures += 1
            self._record_wall(value, request_wall_ms, 0.0)

    def inference_success(self, capability: Capability, inference_ms: float, request_wall_ms: float) -> None:
        with self._lock:
            value = self._values[capability]
            value.inference_attempts += 1
            value.inference_successes += 1
            self._record_wall(value, request_wall_ms, inference_ms)
            value.successful_inference_ms += inference_ms
            value.last_successful_inference_ms = inference_ms

    def inference_failure(self, capability: Capability, inference_ms: float, request_wall_ms: float) -> None:
        with self._lock:
            value = self._values[capability]
            value.inference_attempts += 1
            value.inference_failures += 1
            self._record_wall(value, request_wall_ms, inference_ms)
            value.failed_inference_ms += inference_ms
            value.last_failed_inference_ms = inference_ms

    def busy(self, capability: Capability, request_wall_ms: float) -> None:
        with self._lock:
            value = self._values[capability]
            value.busy_rejections += 1
            self._record_wall(value, request_wall_ms, 0.0)

    @staticmethod
    def _record_wall(
        value: CapabilityMetricSnapshot, request_wall_ms: float, inference_ms: float
    ) -> None:
        value.request_wall_time_ms_total += request_wall_ms
        value.last_request_wall_time_ms = request_wall_ms
        overhead = max(0.0, request_wall_ms - inference_ms)
        value.non_inference_wall_time_ms_total += overhead
        value.last_non_inference_wall_time_ms = overhead

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            result = {
                "measurement_window_started_at_utc": self._measurement_window_started_at_utc,
                **{name: asdict(value) for name, value in self._values.items()},
            }
            for data in (result["detect"], result["bcs"]):
                eligible = data["requests"] - data["client_rejections"]
                failures = (
                    data["busy_rejections"]
                    + data["server_runtime_failures"]
                    + data["inference_failures"]
                )
                data["eligible_operational_requests"] = eligible
                data["service_impacting_failures"] = failures
                data["service_impacting_failure_rate"] = (
                    failures / eligible if eligible > 0 else None
                )
                data["service_impacting_failure_rate_review_thresholds"] = list(
                    SERVICE_IMPACTING_FAILURE_RATE_REVIEW_THRESHOLDS
                )
            return result
