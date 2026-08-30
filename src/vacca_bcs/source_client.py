from __future__ import annotations

import json
import hashlib
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
DEFAULT_MAX_IMAGE_BYTES = 10 * 1024 * 1024
_SIGNED_URL_KEYS = {
    "schema_version",
    "evidence_id",
    "signed_url",
    "expires_in_seconds",
}


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


@dataclass(frozen=True, slots=True)
class BCSEvidencePayload:
    evidence_id: int
    payload: bytes
    sha256: str

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(evidence_id={self.evidence_id!r}, "
            f"size_bytes={len(self.payload)!r}, sha256={self.sha256!r})"
        )


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


def _reject_duplicate_members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BCSSourceContractError("duplicate JSON object member")
        result[key] = value
    return result


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
            if (
                type(declared) is not str
                or not declared.isascii()
                or not declared.isdigit()
            ):
                raise BCSSourceTransportError("invalid Content-Length response header")
            if int(declared) > maximum:
                raise BCSSourceResponseTooLargeError(
                    "bcs source response exceeds configured maximum"
                )
        declared_length = None if declared is None else int(declared)
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
        if declared_length is not None and len(body) != declared_length:
            raise BCSSourceTransportError(
                "response length did not match Content-Length"
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


def _request_response(
    transport: HTTPTransport, url: str, headers: Mapping[str, str], timeout: float
) -> Any:
    response = None
    try:
        response = transport.get(
            url,
            headers=headers,
            timeout=timeout,
            stream=True,
            allow_redirects=False,
        )
        status_code = response.status_code
    except Exception as exc:
        _close_response(response)
        raise BCSSourceTransportError(
            f"bcs source transport failed with {type(exc).__name__}"
        ) from None
    if type(status_code) is not int:
        _close_response(response)
        raise BCSSourceTransportError("invalid HTTP status code")
    if not 200 <= status_code < 300:
        _close_response(response)
        raise BCSSourceHTTPError(status_code)
    return response


def _decode_json_response(response: Any, maximum: int) -> Any:
    try:
        return json.loads(
            _read_response_body(response, maximum),
            object_pairs_hook=_reject_duplicate_members,
        )
    except BCSSourceClientError:
        raise
    except RecursionError:
        raise BCSSourceContractError("JSON nesting is too deep") from None
    except (TypeError, ValueError):
        raise BCSSourceJSONError("bcs source response was not valid JSON") from None


def _validate_signed_url(value: str) -> None:
    if not value or any(char.isspace() or char in r"\@#" for char in value):
        raise BCSSourceContractError("signed_url is malformed")
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError:
        raise BCSSourceContractError("signed_url is malformed") from None
    if (
        not hostname
        or not parsed.netloc
        or parsed.netloc.endswith(":")
        or parsed.username
        or parsed.password
        or not parsed.query
        or parsed.scheme.lower() not in {"http", "https"}
    ):
        raise BCSSourceContractError("signed_url is malformed")
    if parsed.scheme.lower() == "http" and hostname.lower() not in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        raise BCSSourceContractError("insecure remote signed_url is not allowed")


def _parse_signed_response(payload: Any, evidence_id: int) -> str:
    signed = _require_object(payload, "signed evidence response")
    _require_keys(signed, _SIGNED_URL_KEYS, "signed evidence response")
    if signed["schema_version"] != SCHEMA_VERSION:
        raise BCSSourceContractError("schema_version must be bcs-source-v1")
    if type(signed["evidence_id"]) is not int or signed["evidence_id"] != evidence_id:
        raise BCSSourceContractError("signed evidence_id does not match request")
    if (
        type(signed["expires_in_seconds"]) is not int
        or not 1 <= signed["expires_in_seconds"] <= 86400
    ):
        raise BCSSourceContractError(
            "expires_in_seconds must be an integer in the range 1..86400"
        )
    signed_url = signed["signed_url"]
    if type(signed_url) is not str:
        raise BCSSourceContractError("signed_url must be a string")
    _validate_signed_url(signed_url)
    return signed_url


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
        self._owns_transport = transport is None

    def close(self) -> None:
        if self._owns_transport:
            _close_response(self._transport)

    def __enter__(self) -> BCSSourceClient:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        if type(base_url) is not str or not base_url.strip():
            raise BCSSourceConfigurationError("base URL must be non-empty")
        value = base_url.strip()
        if any(char.isspace() or char == "\\" for char in value):
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
            or "@" in value
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
        return value[:-1] if parsed.path == "/" else value

    def __repr__(self) -> str:
        return f"{type(self).__name__}(base_url={self._base_url!r}, timeout={self._timeout!r})"

    def fetch(self) -> BCSSourceExport:
        response = _request_response(
            self._transport,
            f"{self._base_url}{SOURCE_PATH}",
            {
                "Authorization": f"Bearer {self._bearer_token}",
                "Accept": "application/json",
            },
            self._timeout,
        )
        return _parse_export(_decode_json_response(response, self._max_response_bytes))


class BCSEvidenceMaterializer(BCSSourceClient):
    def __init__(
        self,
        backend_base_url: str,
        bearer_token: str,
        timeout: float,
        *,
        backend_transport: HTTPTransport | None = None,
        download_transport: HTTPTransport | None = None,
        max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
    ) -> None:
        super().__init__(backend_base_url, bearer_token, timeout, backend_transport)
        if type(max_image_bytes) is not int or max_image_bytes <= 0:
            raise BCSSourceConfigurationError(
                "max_image_bytes must be a positive integer"
            )
        self._max_image_bytes = max_image_bytes
        self._download_transport = (
            download_transport if download_transport is not None else requests.Session()
        )
        self._owns_download_transport = download_transport is None

    def close(self) -> None:
        super().close()
        if self._owns_download_transport:
            _close_response(self._download_transport)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(backend_base_url={self._base_url!r}, "
            f"timeout={self._timeout!r}, max_image_bytes={self._max_image_bytes!r})"
        )

    def materialize(self, evidence_id: int) -> BCSEvidencePayload:
        if type(evidence_id) is not int or evidence_id <= 0:
            raise BCSSourceContractError("evidence_id must be a positive integer")
        signed_response = _request_response(
            self._transport,
            f"{self._base_url}{SOURCE_PATH}/evidence/{evidence_id}/signed-url",
            {
                "Authorization": f"Bearer {self._bearer_token}",
                "Accept": "application/json",
            },
            self._timeout,
        )
        signed_url = _parse_signed_response(
            _decode_json_response(signed_response, 64 * 1024), evidence_id
        )
        download_response = _request_response(
            self._download_transport, signed_url, {}, self._timeout
        )
        payload = _read_response_body(download_response, self._max_image_bytes)
        if not payload:
            raise BCSSourceContractError("evidence payload must not be empty")
        return BCSEvidencePayload(
            evidence_id=evidence_id,
            payload=payload,
            sha256=hashlib.sha256(payload).hexdigest(),
        )
