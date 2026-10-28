"""Generated principal API contracts, without entering lifespan or loading models."""

from pathlib import Path

import pytest

from vacca_api import main


def test_raw_mp4_request_and_typed_job_response():
    operation = main.app.openapi()["paths"]["/api/v1/analyses"]["post"]
    body = operation["requestBody"]
    assert body["required"] is True
    assert body["content"] == {
        "video/mp4": {"schema": {"type": "string", "format": "binary"}},
        "video/quicktime": {"schema": {"type": "string", "format": "binary"}},
    }
    filename = next(p for p in operation["parameters"] if p["name"] == "filename")
    assert filename["in"] == "query" and filename["required"] is True
    assert filename["schema"]["minLength"] == 1
    assert filename["schema"]["maxLength"] == 200
    assert operation["responses"]["202"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/JobResponse"
    }
    for code in (413, 415, 429, 507, 503):
        assert set(operation["responses"][str(code)]["content"]) == {"application/json"}


@pytest.mark.parametrize("suffix,media_type", [
    ("/frames/{index}/image", "image/jpeg"),
    ("/candidate/image", "image/png"),
    ("/frames.zip", "application/zip"),
    ("/results", "application/x-ndjson"),
])
def test_artifact_success_is_binary_not_json_and_errors_remain_json(suffix, media_type):
    operation = main.app.openapi()["paths"]["/api/v1/analyses/{job_id}" + suffix]["get"]
    content = operation["responses"]["200"]["content"]
    assert set(content) == {media_type}
    assert content[media_type]["schema"] == {"type": "string", "format": "binary"}
    for code in ("503", "422"):
        assert set(operation["responses"][code]["content"]) == {"application/json"}
    if suffix == "/frames/{index}/image":
        download = next(p for p in operation["parameters"] if p["name"] == "download")
        assert download["in"] == "query"
        assert download["schema"]["type"] == "boolean"
        assert download["schema"]["default"] is False


@pytest.mark.parametrize("endpoint,dto", [("/detect", "DetectResponse"), ("/bcs", "BCSResponse")])
def test_image_multipart_and_response_dtos_are_preserved(endpoint, dto):
    schema = main.app.openapi()
    operation = schema["paths"][endpoint]["post"]
    content = operation["requestBody"]["content"]
    assert set(content) == {"multipart/form-data"}
    body_name = content["multipart/form-data"]["schema"]["$ref"].rsplit("/", 1)[1]
    body = schema["components"]["schemas"][body_name]
    assert body["required"] == ["file"]
    assert body["properties"]["file"]["type"] == "string"
    assert operation["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": f"#/components/schemas/{dto}"
    }
    for code in ("400", "413", "500", "503", "422"):
        assert set(operation["responses"][code]["content"]) == {"application/json"}


@pytest.mark.parametrize("guide", ["api.md", "arquitectura.md"])
def test_canonical_guides_include_eligible_video_lazy_bcs_trigger(guide):
    text = (Path(__file__).resolve().parents[1] / "docs" / guide).read_text(encoding="utf-8")
    assert "video web nuevo completado" in text
    assert "exactamente una detección de vaca" in text
    assert "sin una solicitud directa a `/bcs`" in text
    assert "video-web.md" in text
