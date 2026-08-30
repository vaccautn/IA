from __future__ import annotations

import hashlib
import json

import requests
import pytest
import vacca_bcs.source_client as source_client_module

from vacca_bcs.source_client import (
    BCSEvidenceMaterializer,
    BCSSourceResponseTooLargeError,
    BCSSourceClient,
    BCSSourceConfigurationError,
    BCSSourceContractError,
    BCSSourceHTTPError,
    BCSSourceJSONError,
    BCSSourceTransportError,
)


class FakeResponse:
    def __init__(
        self, payload, status_code=200, *, headers=None, body=None, chunks=None
    ):
        self.status_code = status_code
        self.headers = headers or {}
        self.closed = False
        self.iter_content_calls = 0
        self._chunks = (
            chunks
            if chunks is not None
            else [
                body
                if body is not None
                else (
                    b"not-json"
                    if isinstance(payload, Exception)
                    else payload
                    if isinstance(payload, bytes)
                    else json.dumps(payload).encode()
                )
            ]
        )

    def iter_content(self, chunk_size):
        self.iter_content_calls += 1
        return iter(self._chunks)

    def close(self):
        self.closed = True


class BrokenStatusResponse:
    def __init__(self):
        self.closed = False

    @property
    def status_code(self):
        raise RuntimeError("secret-token")

    def close(self):
        self.closed = True


class BrokenHeadersResponse:
    def __init__(self):
        self.closed = False

    status_code = 200

    @property
    def headers(self):
        raise RuntimeError("secret-token")

    def close(self):
        self.closed = True


class FakeTransport:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls, self.close_calls = response, error, [], 0

    def get(self, url, *, headers, timeout, stream, allow_redirects):
        self.calls.append((url, headers, timeout, stream, allow_redirects))
        if self.error:
            raise self.error
        return self.response

    def close(self):
        self.close_calls += 1


_DEFAULT = object()


def export_payload():
    return {
        "schema_version": "bcs-source-v1",
        "rows": [
            {
                "evaluation_id": 2,
                "session_id": 7,
                "animal_id": 31,
                "valor_cc": 4,
                "evidence": [
                    {"evidence_id": 5, "storage_key": "same-key"},
                    {"evidence_id": 8, "storage_key": ""},
                ],
            },
            {
                "evaluation_id": 9,
                "session_id": 8,
                "animal_id": 32,
                "valor_cc": 1,
                "evidence": [{"evidence_id": 10, "storage_key": "same-key"}],
            },
        ],
    }


def client_for(
    payload=_DEFAULT,
    *,
    status_code=200,
    error=None,
    response_kwargs=None,
    max_response_bytes=None,
):
    transport = FakeTransport(
        FakeResponse(
            export_payload() if payload is _DEFAULT else payload,
            status_code,
            **(response_kwargs or {}),
        ),
        error,
    )
    kwargs = (
        {} if max_response_bytes is None else {"max_response_bytes": max_response_bytes}
    )
    return BCSSourceClient(
        "http://127.0.0.1:8000/", "secret-token", 2.5, transport, **kwargs
    ), transport


def test_fetch_parses_immutable_nested_export_without_materializing_keys():
    client, _ = client_for()

    exported = client.fetch()

    assert exported.schema_version == "bcs-source-v1"
    assert exported.rows[0].evidence[1].storage_key == ""
    assert (
        exported.rows[0].evidence[0].storage_key
        == exported.rows[1].evidence[0].storage_key
    )
    assert isinstance(exported.rows, tuple)
    assert isinstance(exported.rows[0].evidence, tuple)
    with pytest.raises(AttributeError):
        exported.rows = ()


def test_fetch_uses_exact_path_bearer_header_and_timeout():
    client, transport = client_for()

    client.fetch()

    assert transport.calls == [
        (
            "http://127.0.0.1:8000/api/bcs-source-v1",
            {"Authorization": "Bearer secret-token", "Accept": "application/json"},
            2.5,
            True,
            False,
        )
    ]


def test_http_and_transport_failures_are_typed_without_sensitive_data():
    for error, expected in [
        (None, BCSSourceHTTPError),
        (requests.Timeout("secret-token"), BCSSourceTransportError),
        (requests.ConnectionError("secret-token"), BCSSourceTransportError),
    ]:
        client, transport = client_for(
            payload={"error": "do not disclose"}, status_code=503, error=error
        )
        with pytest.raises(expected) as failure:
            client.fetch()
        assert "secret-token" not in str(failure.value)
        assert "do not disclose" not in str(failure.value)
        if error is None:
            assert transport.response.closed


def test_invalid_json_is_typed_and_payload_is_not_disclosed():
    client, transport = client_for(payload=ValueError("secret-token"))

    with pytest.raises(BCSSourceJSONError) as failure:
        client.fetch()

    assert "secret-token" not in str(failure.value)
    assert transport.response.closed


def test_redirect_status_is_not_followed():
    client, transport = client_for(status_code=302)

    with pytest.raises(BCSSourceHTTPError):
        client.fetch()

    assert transport.response.closed
    assert transport.calls[0][3:] == (True, False)


def test_declared_or_chunked_response_over_limit_fails_without_disclosure():
    cases = [
        {
            "response_kwargs": {
                "body": b"secret-token",
                "headers": {"Content-Length": "12"},
            }
        },
        {"response_kwargs": {"chunks": [b"safe", b"secret-token"]}},
    ]
    for case in cases:
        client, transport = client_for(max_response_bytes=8, **case)
        with pytest.raises(BCSSourceResponseTooLargeError) as failure:
            client.fetch()
        assert "secret-token" not in str(failure.value)
        if "headers" in case["response_kwargs"]:
            assert transport.response.iter_content_calls == 0


def test_content_length_must_match_streamed_bytes():
    client, transport = client_for(
        response_kwargs={"body": b"{}", "headers": {"Content-Length": "1"}}
    )

    with pytest.raises(BCSSourceTransportError):
        client.fetch()

    assert transport.response.closed


@pytest.mark.parametrize("content_length", ["-1", "invalid", 1])
def test_invalid_content_length_fails_closed(content_length):
    client, transport = client_for(
        response_kwargs={"body": b"{}", "headers": {"Content-Length": content_length}}
    )

    with pytest.raises(BCSSourceTransportError):
        client.fetch()

    assert transport.response.closed


@pytest.mark.parametrize(
    "body",
    [
        b'{"schema_version":"bcs-source-v1","schema_version":"bcs-source-v1","rows":[]}',
        b'{"schema_version":"bcs-source-v1","rows":[{"evaluation_id":1,"evaluation_id":1,"session_id":1,"animal_id":1,"valor_cc":1,"evidence":[]}]}',
        b'{"schema_version":"bcs-source-v1","rows":[{"evaluation_id":1,"session_id":1,"animal_id":1,"valor_cc":1,"evidence":[{"evidence_id":1,"storage_key":"a","storage_key":"secret-token"}]}]}',
    ],
)
def test_duplicate_json_members_at_every_nesting_level_are_rejected(body):
    client, transport = client_for(response_kwargs={"body": body})

    with pytest.raises(BCSSourceContractError) as failure:
        client.fetch()

    assert "secret-token" not in str(failure.value)
    assert transport.response.closed


def test_deep_json_failure_is_a_sanitized_contract_error():
    client, transport = client_for(response_kwargs={"body": b"[" * 1100 + b"]" * 1100})

    with pytest.raises(BCSSourceContractError) as failure:
        client.fetch()

    assert "secret-token" not in str(failure.value)
    assert transport.response.closed


def test_response_closes_when_status_or_headers_fail():
    for response in (BrokenStatusResponse(), BrokenHeadersResponse()):
        transport = FakeTransport(response=response)
        client = BCSSourceClient("https://api.test", "secret-token", 1, transport)
        with pytest.raises(BCSSourceTransportError) as failure:
            client.fetch()
        assert "secret-token" not in str(failure.value)
        assert response.closed


def test_response_closes_when_streaming_fails():
    client, transport = client_for(response_kwargs={"chunks": [b"safe", object()]})

    with pytest.raises(BCSSourceTransportError):
        client.fetch()

    assert transport.response.closed


@pytest.mark.parametrize("status_code", [True, False, "200", object()])
def test_status_code_must_be_a_strict_integer(status_code):
    client, transport = client_for(status_code=status_code)

    with pytest.raises(BCSSourceTransportError) as failure:
        client.fetch()

    assert "secret-token" not in str(failure.value)
    assert transport.response.closed


def test_client_closes_only_owned_sessions_and_supports_context_manager(monkeypatch):
    injected = FakeTransport()
    BCSSourceClient("https://api.test", "token", 1, injected).close()
    assert injected.close_calls == 0

    owned = FakeTransport()
    monkeypatch.setattr(source_client_module.requests, "Session", lambda: owned)
    with BCSSourceClient("https://api.test", "token", 1):
        pass
    assert owned.close_calls == 1


@pytest.mark.parametrize("payload", [None, [], {"schema_version": "other", "rows": []}])
def test_malformed_or_wrong_version_payload_fails_closed(payload):
    client, _ = client_for(payload)
    with pytest.raises(BCSSourceContractError):
        client.fetch()


def test_empty_export_and_empty_evidence_are_valid():
    empty = {"schema_version": "bcs-source-v1", "rows": []}
    row_without_evidence = {
        "schema_version": "bcs-source-v1",
        "rows": [
            {
                "evaluation_id": 1,
                "session_id": 2,
                "animal_id": 3,
                "valor_cc": 1,
                "evidence": [],
            }
        ],
    }

    assert client_for(empty)[0].fetch().rows == ()
    assert client_for(row_without_evidence)[0].fetch().rows[0].evidence == ()


@pytest.mark.parametrize(
    "field,value",
    [
        ("valor_cc", 3.0),
        ("valor_cc", True),
        ("valor_cc", 0),
        ("valor_cc", 6),
        ("evaluation_id", True),
        *[
            (field, value)
            for field in ("evaluation_id", "session_id", "animal_id")
            for value in (0, -1)
        ],
    ],
)
def test_scores_and_ids_are_strictly_validated(field, value):
    payload = export_payload()
    payload["rows"][0][field] = value
    client, _ = client_for(payload)
    with pytest.raises(BCSSourceContractError):
        client.fetch()


@pytest.mark.parametrize("value", [0, -1])
def test_evidence_ids_must_be_positive(value):
    payload = export_payload()
    payload["rows"][0]["evidence"][0]["evidence_id"] = value
    with pytest.raises(BCSSourceContractError):
        client_for(payload)[0].fetch()


@pytest.mark.parametrize("rows", [[9, 2], [2, 2]])
def test_evaluation_order_or_duplicate_ids_fail(rows):
    payload = export_payload()
    payload["rows"] = [
        dict(payload["rows"][index], evaluation_id=value)
        for index, value in enumerate(rows)
    ]
    client, _ = client_for(payload)
    with pytest.raises(BCSSourceContractError):
        client.fetch()


def test_evidence_order_and_duplicate_ids_fail():
    for evidence in [
        [
            {"evidence_id": 8, "storage_key": "a"},
            {"evidence_id": 5, "storage_key": "b"},
        ],
        [
            {"evidence_id": 5, "storage_key": "a"},
            {"evidence_id": 5, "storage_key": "b"},
        ],
    ]:
        payload = export_payload()
        payload["rows"][0]["evidence"] = evidence
        client, _ = client_for(payload)
        with pytest.raises(BCSSourceContractError):
            client.fetch()

    payload = export_payload()
    payload["rows"][1]["evidence"] = [{"evidence_id": 5, "storage_key": "other-key"}]
    with pytest.raises(BCSSourceContractError):
        client_for(payload)[0].fetch()


def test_storage_key_must_be_a_string():
    payload = export_payload()
    payload["rows"][0]["evidence"][0]["storage_key"] = None
    client, _ = client_for(payload)
    with pytest.raises(BCSSourceContractError):
        client.fetch()


@pytest.mark.parametrize(
    "base_url,token,timeout",
    [
        ("", "token", 1),
        ("ftp://example.test", "token", 1),
        ("https://", "token", 1),
        ("https://api.test/path", "token", 1),
        (r"https://api.test\escape", "token", 1),
        ("https://api.test?", "token", 1),
        ("https://api.test#", "token", 1),
        ("https://@api.test", "token", 1),
        ("https://:@api.test", "token", 1),
        ("https://user:pass@api.test", "token", 1),
        ("https://api.test//", "token", 1),
        ("https://[::1", "token", 1),
        ("https://api.test:bad", "token", 1),
        ("https://api.test:", "token", 1),
        ("http://example.test", "token", 1),
        ("https://api.test", "", 1),
        ("https://api.test", "token", 0),
    ],
)
def test_client_rejects_unsafe_configuration(base_url, token, timeout):
    with pytest.raises(BCSSourceConfigurationError):
        BCSSourceClient(base_url, token, timeout, FakeTransport())


@pytest.mark.parametrize(
    "base_url", ["http://localhost/", "http://127.0.0.1:8000", "http://[::1]"]
)
def test_loopback_http_is_allowed(base_url):
    BCSSourceClient(base_url, "token", 1, FakeTransport())


def test_client_repr_does_not_disclose_token():
    client, _ = client_for()
    assert "secret-token" not in repr(client)


@pytest.mark.parametrize("max_response_bytes", [0, -1, True])
def test_client_rejects_invalid_response_limit(max_response_bytes):
    with pytest.raises(BCSSourceConfigurationError):
        BCSSourceClient(
            "https://api.test",
            "token",
            1,
            FakeTransport(),
            max_response_bytes=max_response_bytes,
        )


def signed_url_body(
    evidence_id=7,
    signed_url="https://r2.example/object?sig=a@b",
    expiry=3600,
    schema_version="bcs-source-v1",
):
    return json.dumps(
        {
            "schema_version": schema_version,
            "evidence_id": evidence_id,
            "signed_url": signed_url,
            "expires_in_seconds": expiry,
        }
    ).encode()


def materializer_for(
    signed_response=None,
    download_response=None,
    *,
    max_image_bytes=None,
):
    backend = FakeTransport(signed_response or FakeResponse(signed_url_body()))
    download = FakeTransport(download_response or FakeResponse(b"image-bytes"))
    kwargs = {} if max_image_bytes is None else {"max_image_bytes": max_image_bytes}
    return (
        BCSEvidenceMaterializer(
            "https://backend.example/",
            "backend-token",
            2.5,
            backend_transport=backend,
            download_transport=download,
            **kwargs,
        ),
        backend,
        download,
    )


def test_materialize_resolves_signed_url_and_keeps_credentials_separate():
    client, backend, download = materializer_for()

    result = client.materialize(7)

    assert result.evidence_id == 7
    assert result.payload == b"image-bytes"
    assert result.sha256 == hashlib.sha256(b"image-bytes").hexdigest()
    assert backend.calls[0] == (
        "https://backend.example/api/bcs-source-v1/evidence/7/signed-url",
        {"Authorization": "Bearer backend-token", "Accept": "application/json"},
        2.5,
        True,
        False,
    )
    assert download.calls[0] == (
        "https://r2.example/object?sig=a@b",
        {},
        2.5,
        True,
        False,
    )
    assert "backend-token" not in repr(client)
    assert "sig=abc" not in repr(result)
    assert "image-bytes" not in repr(result)


@pytest.mark.parametrize("evidence_id", [True, 0, -1, 1.5])
def test_materialize_requires_positive_strict_evidence_id(evidence_id):
    client, backend, _ = materializer_for()
    with pytest.raises(BCSSourceContractError):
        client.materialize(evidence_id)
    assert backend.calls == []


@pytest.mark.parametrize(
    "body",
    [
        signed_url_body(evidence_id=8),
        signed_url_body(schema_version="other"),
        json.dumps({"schema_version": "bcs-source-v1", "evidence_id": 7}).encode(),
        signed_url_body(signed_url="http://remote.example/object?sig=abc"),
        signed_url_body(signed_url=1),
        signed_url_body(signed_url="https://user:pass@r2.example/object?sig=abc"),
        signed_url_body(signed_url=r"https://r2.example/object\escape?sig=abc"),
        signed_url_body(signed_url="https://r2.example/object#fragment"),
        signed_url_body(signed_url="https://r2.example/object"),
        signed_url_body(signed_url="https://r2.example:bad/object?sig=abc"),
        signed_url_body(expiry=0),
    ],
)
def test_signed_response_domain_is_strict(body):
    client, _, download = materializer_for(FakeResponse(body))
    with pytest.raises(BCSSourceContractError):
        client.materialize(7)
    assert download.calls == []


def test_loopback_http_signed_url_is_allowed():
    local, _, local_download = materializer_for(
        FakeResponse(signed_url_body(signed_url="http://127.0.0.1:9000/object?sig=abc"))
    )
    assert local.materialize(7).payload == b"image-bytes"
    assert local_download.calls[0][0].startswith("http://127.0.0.1:9000/")


def test_signed_and_download_http_failures_are_closed_and_sanitized():
    backend_response = FakeResponse(b"secret-token", status_code=503)
    client, _, _ = materializer_for(backend_response)
    with pytest.raises(BCSSourceHTTPError) as failure:
        client.materialize(7)
    assert backend_response.closed
    assert "secret-token" not in str(failure.value)
    download_response = FakeResponse(b"secret-token", status_code=302)
    client, _, _ = materializer_for(download_response=download_response)
    with pytest.raises(BCSSourceHTTPError) as failure:
        client.materialize(7)
    assert download_response.closed
    assert "secret-token" not in str(failure.value)


@pytest.mark.parametrize(
    "status_code,headers", [(206, {}), (200, {"Content-Range": "bytes 0-5/10"})]
)
def test_partial_downloads_are_rejected_before_payload_return(status_code, headers):
    response = FakeResponse(b"partial", status_code=status_code, headers=headers)
    client, _, _ = materializer_for(download_response=response)
    with pytest.raises(BCSSourceHTTPError):
        client.materialize(7)
    assert response.closed
    assert response.iter_content_calls == 0


def test_download_declared_limit_rejects_before_reading():
    response = FakeResponse(b"secret-token", headers={"Content-Length": "12"})
    client, _, _ = materializer_for(download_response=response, max_image_bytes=8)
    with pytest.raises(BCSSourceResponseTooLargeError) as failure:
        client.materialize(7)
    assert response.closed
    assert response.iter_content_calls == 0
    assert "secret-token" not in str(failure.value)


def test_download_chunk_limit_and_framing_reject_and_close():
    cases = [
        (
            FakeResponse(b"", chunks=[b"safe", b"secret-token"]),
            BCSSourceResponseTooLargeError,
        ),
        (
            FakeResponse(b"image", headers={"Content-Length": "4"}),
            BCSSourceTransportError,
        ),
        (FakeResponse(b""), BCSSourceContractError),
    ]
    for response, error_type in cases:
        client, _, _ = materializer_for(download_response=response, max_image_bytes=8)
        with pytest.raises(error_type) as failure:
            client.materialize(7)
        assert response.closed
        assert "secret-token" not in str(failure.value)


def test_materializer_closes_only_owned_sessions(monkeypatch):
    injected_backend, injected_download = FakeTransport(), FakeTransport()
    BCSEvidenceMaterializer(
        "https://backend.example",
        "token",
        1,
        backend_transport=injected_backend,
        download_transport=injected_download,
    ).close()
    assert injected_backend.close_calls == injected_download.close_calls == 0
    owned_download = FakeTransport()
    monkeypatch.setattr(
        source_client_module.requests, "Session", lambda: owned_download
    )
    with BCSEvidenceMaterializer(
        "https://backend.example", "token", 1, backend_transport=injected_backend
    ):
        pass
    assert owned_download.close_calls == 1


def test_invalid_image_limit_creates_no_owned_session(monkeypatch):
    created = []
    monkeypatch.setattr(
        source_client_module.requests, "Session", lambda: created.append(True)
    )
    with pytest.raises(BCSSourceConfigurationError):
        BCSEvidenceMaterializer(
            "https://backend.example", "token", 1, max_image_bytes=0
        )
    assert created == []
