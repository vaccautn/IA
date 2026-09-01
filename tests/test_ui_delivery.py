from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest

from vacca_api import main


async def _get_ui_messages() -> list[dict]:
    messages: list[dict] = []

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict) -> None:
        messages.append(message)

    await main.app(
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/ui",
            "raw_path": b"/ui",
            "query_string": b"",
            "headers": [],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
        },
        receive,
        send,
    )
    return messages


def test_ui_is_delivered_through_the_actual_asgi_route() -> None:
    messages = asyncio.run(_get_ui_messages())
    start = next(message for message in messages if message["type"] == "http.response.start")
    body = b"".join(message.get("body", b"") for message in messages if message["type"] == "http.response.body")
    headers = dict(start["headers"])
    assert start["status"] == 200
    assert headers[b"content-type"] == b"text/html; charset=utf-8"
    assert b"<!DOCTYPE html>" in body
    assert b"Calculate BCS" in body


def test_node_controller_behavior() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is unavailable; UI behavior tests were not run")
    script = Path(__file__).with_name("ui_controller.test.js")
    result = subprocess.run([node, "--test", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
