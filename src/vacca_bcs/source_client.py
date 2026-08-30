from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol
from urllib.parse import urlsplit

import requests

SCHEMA_VERSION = "bcs-source-v1"
SOURCE_PATH = "/api/bcs-source-v1"
_SOURCE_KEYS = {"schema_version", "rows"}
_ROW_KEYS = {"evaluation_id", "session_id", "animal_id", "valor_cc", "evidence"}
_EVIDENCE_KEYS = {"evidence_id", "storage_key"}
DEFAULT_MAX_RESPONSE_BYTES = 64 * 1024 * 1024


class BCSSourceClientError(Exception):
    pass


class BCSSourceConfigurationError(BCSSourceClientError):
    pass


class BCSSourceTransportError(BCSSourceClientError):
    pass


class BCSSourceHTTPError(BCSSourceClientError):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"bcs source request returned HTTP {status_code}")


class BCSSourceJSONError(BCSSourceClientError):
    pass


class BCSSourceResponseTooLargeError(BCSSourceClientError):
    pass


class BCSSourceContractError(BCSSourceClientError):
    pass


@dataclass(frozen=True, slots=True)
class BCSSourceEvidence:
    evidence_id: int
    storage_key: str

    def __post_init__(self) -> None:
        if type(self.evidence_id) is not int or self.evidence_id <= 0:
            raise BCSSourceContractError("evidence_id must be a positive integer")
        if type(self.storage_key) is not str:
            raise BCSSourceContractError("storage_key must be a string")


@dataclass(frozen=True, slots=True)
class BCSSourceEvaluationRow:
    evaluation_id: int
    session_id: int
    animal_id: int
    valor_cc: int
    evidence: tuple[BCSSourceEvidence, ...]

    def __post_init__(self) -> None:
        for name in ("evaluation_id", "session_id", "animal_id"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise BCSSourceContractError(f"{name} must be a positive integer")
        if type(self.valor_cc) is not int or not 1 <= self.valor_cc <= 5:
            raise BCSSourceContractError(
                "valor_cc must be an integer in the range 1..5"
            )
        evidence = tuple(self.evidence)
        if any(type(item) is not BCSSourceEvidence for item in evidence):
            raise BCSSourceContractError("evidence must contain source evidence values")
        if [item.evidence_id for item in evidence] != sorted(
            {item.evidence_id for item in evidence}
        ):
            raise BCSSourceContractError(
                "evidence must be strictly ordered by evidence_id"
            )
        object.__setattr__(self, "evidence", evidence)


@dataclass(frozen=True, slots=True)
class BCSSourceExport:
    schema_version: str
    rows: tuple[BCSSourceEvaluationRow, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise BCSSourceContractError("schema_version must be bcs-source-v1")
        rows = tuple(self.rows)
        if any(type(row) is not BCSSourceEvaluationRow for row in rows):
            raise BCSSourceContractError("rows must contain source evaluation values")
        evaluation_ids = [row.evaluation_id for row in rows]
        evidence_ids = [item.evidence_id for row in rows for item in row.evidence]
        if evaluation_ids != sorted(set(evaluation_ids)):
            raise BCSSourceContractError(
                "rows must be strictly ordered by evaluation_id"
            )
        if len(evaluation_ids) != len(set(evaluation_ids)):
            raise BCSSourceContractError("duplicate evaluation_id")
        if len(evidence_ids) != len(set(evidence_ids)):
            raise BCSSourceContractError("duplicate evidence_id")
        object.__setattr__(self, "rows", rows)


class HTTPTransport(Protocol):
    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout: float,
        stream: bool,
        allow_redirects: bool,
    ) -> Any: ...


def _require_object(value: Any, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise BCSSourceContractError(f"{name} must be a JSON object")
    return value


def _require_keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise BCSSourceContractError(f"{name} has an invalid field set")


def _parse_export(payload: Any) -> BCSSourceExport:
    source = _require_object(payload, "source export")
    _require_keys(source, _SOURCE_KEYS, "source export")
    if source["schema_version"] != SCHEMA_VERSION:
        raise BCSSourceContractError("schema_version must be bcs-source-v1")
    if type(source["rows"]) is not list:
        raise BCSSourceContractError("rows must be a JSON array")

    rows: list[BCSSourceEvaluationRow] = []
    for raw_row in source["rows"]:
        row = _require_object(raw_row, "evaluation row")
        _require_keys(row, _ROW_KEYS, "evaluation row")
        if type(row["evidence"]) is not list:
            raise BCSSourceContractError("evidence must be a JSON array")
        evidence_values: list[BCSSourceEvidence] = []
        for raw_evidence in row["evidence"]:
            evidence = _require_object(raw_evidence, "evidence")
            _require_keys(evidence, _EVIDENCE_KEYS, "evidence")
            evidence_values.append(
                BCSSourceEvidence(evidence["evidence_id"], evidence["storage_key"])
            )
        rows.append(
            BCSSourceEvaluationRow(
                row["evaluation_id"],
                row["session_id"],
                row["animal_id"],
                row["valor_cc"],
                tuple(evidence_values),
            )
        )
    return BCSSourceExport(SCHEMA_VERSION, tuple(rows))


def _close_response(response: Any) -> None:
    try:
        response.close()
    except Exception:
        pass


def _read_response_body(response: Any, maximum: int) -> bytes:
    try:
        declared = response.headers.get("Content-Length")
        if declared is not None:
            if type(declared) is not str or not declared.isdigit():
                raise BCSSourceTransportError("invalid Content-Length response header")
            if int(declared) > maximum:
                raise BCSSourceResponseTooLargeError(
                    "bcs source response exceeds configured maximum"
                )
        body = bytearray()
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if type(chunk) is not bytes:
                raise BCSSourceTransportError(
                    "bcs source response contained an invalid chunk"
                )
            body.extend(chunk)
            if len(body) > maximum:
                raise BCSSourceResponseTooLargeError(
                    "bcs source response exceeds configured maximum"
                )
        return bytes(body)
    except BCSSourceClientError:
        raise
    except Exception as exc:
        raise BCSSourceTransportError(
            f"bcs source transport failed with {type(exc).__name__}"
        ) from None
    finally:
        _close_response(response)


class BCSSourceClient:
    def __init__(
        self,
        base_url: str,
        bearer_token: str,
        timeout: float,
        transport: HTTPTransport | None = None,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    ) -> None:
        self._base_url = self._normalize_base_url(base_url)
        if (
            type(bearer_token) is not str
            or not bearer_token.strip()
            or any(char.isspace() for char in bearer_token)
        ):
            raise BCSSourceConfigurationError("bearer token must be a non-empty token")
        if (
            type(timeout) not in (int, float)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise BCSSourceConfigurationError("timeout must be finite and positive")
        if type(max_response_bytes) is not int or max_response_bytes <= 0:
            raise BCSSourceConfigurationError(
                "max_response_bytes must be a positive integer"
            )
        self._bearer_token = bearer_token
        self._timeout = float(timeout)
        self._max_response_bytes = max_response_bytes
        self._transport = transport if transport is not None else requests.Session()

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        if type(base_url) is not str or not base_url.strip():
            raise BCSSourceConfigurationError("base URL must be non-empty")
        value = base_url.strip().rstrip("/")
        if any(char.isspace() for char in value):
            raise BCSSourceConfigurationError("base URL is malformed")
        try:
            parsed = urlsplit(value)
            hostname = parsed.hostname
            _ = parsed.port
        except ValueError:
            raise BCSSourceConfigurationError("base URL is malformed") from None
        if (
            not hostname
            or not parsed.netloc
            or parsed.netloc.endswith(":")
            or parsed.username
            or parsed.password
        ):
            raise BCSSourceConfigurationError("base URL is malformed")
        if (
            parsed.path not in ("", "/")
            or "?" in value
            or "#" in value
            or parsed.scheme.lower() not in {"http", "https"}
        ):
            raise BCSSourceConfigurationError("base URL is malformed")
        if parsed.scheme.lower() == "http" and hostname.lower() not in {
            "localhost",
            "127.0.0.1",
            "::1",
        }:
            raise BCSSourceConfigurationError("insecure remote base URL is not allowed")
        return value

    def __repr__(self) -> str:
        return f"{type(self).__name__}(base_url={self._base_url!r}, timeout={self._timeout!r})"

    def fetch(self) -> BCSSourceExport:
        try:
            response = self._transport.get(
                f"{self._base_url}{SOURCE_PATH}",
                headers={
                    "Authorization": f"Bearer {self._bearer_token}",
                    "Accept": "application/json",
                },
                timeout=self._timeout,
                stream=True,
                allow_redirects=False,
            )
            status_code = response.status_code
        except Exception as exc:
            raise BCSSourceTransportError(
                f"bcs source transport failed with {type(exc).__name__}"
            ) from None
        if type(status_code) is not int or not 200 <= status_code < 300:
            _close_response(response)
            raise BCSSourceHTTPError(status_code)
        try:
            payload = json.loads(
                _read_response_body(response, self._max_response_bytes)
            )
        except BCSSourceClientError:
            raise
        except (TypeError, ValueError):
            raise BCSSourceJSONError("bcs source response was not valid JSON") from None
        return _parse_export(payload)
