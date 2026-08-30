from __future__ import annotations

import json

import requests
import pytest

from vacca_bcs.source_client import (
    BCSSourceClient,
    BCSSourceConfigurationError,
    BCSSourceContractError,
    BCSSourceHTTPError,
    BCSSourceJSONError,
    BCSSourceResponseTooLargeError,
    BCSSourceTransportError,
)


class FakeResponse:
    def __init__(
        self, payload, status_code=200, *, headers=None, body=None, chunks=None
    ):
        self.status_code = status_code
        self.headers = headers or {}
        self.closed = False
        self._chunks = (
            chunks
            if chunks is not None
            else [
                body
                if body is not None
                else (
                    b"not-json"
                    if isinstance(payload, Exception)
                    else json.dumps(payload).encode()
                )
            ]
        )

    def iter_content(self, chunk_size):
        return iter(self._chunks)

    def close(self):
        self.closed = True


class FakeTransport:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def get(self, url, *, headers, timeout, stream, allow_redirects):
        self.calls.append((url, headers, timeout, stream, allow_redirects))
        if self.error:
            raise self.error
        return self.response


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
        client, _ = client_for(
            payload={"error": "do not disclose"}, status_code=503, error=error
        )
        with pytest.raises(expected) as failure:
            client.fetch()
        assert "secret-token" not in str(failure.value)
        assert "do not disclose" not in str(failure.value)


def test_invalid_json_is_typed_and_payload_is_not_disclosed():
    client, _ = client_for(payload=ValueError("secret-token"))

    with pytest.raises(BCSSourceJSONError) as failure:
        client.fetch()

    assert "secret-token" not in str(failure.value)


def test_redirect_status_is_not_followed():
    client, transport = client_for(status_code=302)

    with pytest.raises(BCSSourceHTTPError):
        client.fetch()

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
        client, _ = client_for(max_response_bytes=8, **case)
        with pytest.raises(BCSSourceResponseTooLargeError) as failure:
            client.fetch()
        assert "secret-token" not in str(failure.value)


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
        ("https://api.test?", "token", 1),
        ("https://api.test#", "token", 1),
        ("https://user:pass@api.test", "token", 1),
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
