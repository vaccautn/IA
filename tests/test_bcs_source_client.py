from __future__ import annotations

import requests
import pytest

from vacca_bcs.source_client import (
    BCSSourceClient,
    BCSSourceConfigurationError,
    BCSSourceContractError,
    BCSSourceHTTPError,
    BCSSourceJSONError,
    BCSSourceTransportError,
)


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.status_code = status_code
        self.payload = payload

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class FakeTransport:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def get(self, url, *, headers, timeout):
        self.calls.append((url, headers, timeout))
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


def client_for(payload=_DEFAULT, *, status_code=200, error=None):
    transport = FakeTransport(
        FakeResponse(export_payload() if payload is _DEFAULT else payload, status_code), error
    )
    return BCSSourceClient("http://127.0.0.1:8000/", "secret-token", 2.5, transport), transport


def test_fetch_parses_immutable_nested_export_without_materializing_keys():
    client, _ = client_for()

    exported = client.fetch()

    assert exported.schema_version == "bcs-source-v1"
    assert exported.rows[0].evidence[1].storage_key == ""
    assert exported.rows[0].evidence[0].storage_key == exported.rows[1].evidence[0].storage_key
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
        )
    ]


def test_http_and_transport_failures_are_typed_without_sensitive_data():
    for error, expected in [
        (None, BCSSourceHTTPError),
        (requests.Timeout("secret-token"), BCSSourceTransportError),
        (requests.ConnectionError("secret-token"), BCSSourceTransportError),
    ]:
        client, _ = client_for(payload={"error": "do not disclose"}, status_code=503, error=error)
        with pytest.raises(expected) as failure:
            client.fetch()
        assert "secret-token" not in str(failure.value)
        assert "do not disclose" not in str(failure.value)


def test_invalid_json_is_typed_and_payload_is_not_disclosed():
    client, _ = client_for(payload=ValueError("secret-token"))

    with pytest.raises(BCSSourceJSONError) as failure:
        client.fetch()

    assert "secret-token" not in str(failure.value)


@pytest.mark.parametrize("payload", [None, [], {"schema_version": "other", "rows": []}])
def test_malformed_or_wrong_version_payload_fails_closed(payload):
    client, _ = client_for(payload)
    with pytest.raises(BCSSourceContractError):
        client.fetch()


@pytest.mark.parametrize(
    "field,value",
    [("valor_cc", 3.0), ("valor_cc", True), ("valor_cc", 0), ("valor_cc", 6), ("evaluation_id", True)],
)
def test_scores_and_ids_are_strictly_validated(field, value):
    payload = export_payload()
    payload["rows"][0][field] = value
    client, _ = client_for(payload)
    with pytest.raises(BCSSourceContractError):
        client.fetch()


@pytest.mark.parametrize("rows", [[9, 2], [2, 2]])
def test_evaluation_order_or_duplicate_ids_fail(rows):
    payload = export_payload()
    payload["rows"] = [dict(payload["rows"][index], evaluation_id=value) for index, value in enumerate(rows)]
    client, _ = client_for(payload)
    with pytest.raises(BCSSourceContractError):
        client.fetch()


def test_evidence_order_and_duplicate_ids_fail():
    for evidence in [
        [{"evidence_id": 8, "storage_key": "a"}, {"evidence_id": 5, "storage_key": "b"}],
        [{"evidence_id": 5, "storage_key": "a"}, {"evidence_id": 5, "storage_key": "b"}],
    ]:
        payload = export_payload()
        payload["rows"][0]["evidence"] = evidence
        client, _ = client_for(payload)
        with pytest.raises(BCSSourceContractError):
            client.fetch()


def test_storage_key_must_be_a_string():
    payload = export_payload()
    payload["rows"][0]["evidence"][0]["storage_key"] = None
    client, _ = client_for(payload)
    with pytest.raises(BCSSourceContractError):
        client.fetch()


@pytest.mark.parametrize(
    "base_url,token,timeout",
    [("", "token", 1), ("ftp://example.test", "token", 1), ("https://", "token", 1), ("https://api.test", "", 1), ("https://api.test", "token", 0)],
)
def test_client_rejects_unsafe_configuration(base_url, token, timeout):
    with pytest.raises(BCSSourceConfigurationError):
        BCSSourceClient(base_url, token, timeout, FakeTransport())


def test_client_repr_does_not_disclose_token():
    client, _ = client_for()
    assert "secret-token" not in repr(client)
