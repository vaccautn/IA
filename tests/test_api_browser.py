"""Browser readiness/cancellation without loading the API app or any models."""

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import uvicorn

from vacca_api import browser
from scripts import run_api


def server(opened=None):
    return browser.BrowserServer(
        uvicorn.Config("unused:app"), "http://127.0.0.1:8001/videos/",
        opened or SimpleNamespace(value=False, get_lock=nullcontext),
    )


def controlled_clock(monkeypatch, tick=None):
    clock = SimpleNamespace(now=0.0)
    original_sleep = asyncio.sleep

    async def sleep(delay):
        clock.now += delay
        if tick:
            tick(clock.now)
        await original_sleep(0)

    monkeypatch.setattr(browser.asyncio, "get_running_loop", lambda: SimpleNamespace(time=lambda: clock.now))
    monkeypatch.setattr(browser.asyncio, "sleep", sleep)
    return clock


def test_delayed_startup_opens_only_after_own_server_listens(monkeypatch):
    target = server()
    target.lifespan_ready = True  # Lifespan readiness alone is deliberately insufficient.
    opened = []

    def tick(now):
        assert not opened
        if now >= 2.2:
            target.started = True

    clock = controlled_clock(monkeypatch, tick)
    monkeypatch.setattr(browser.webbrowser, "open", lambda url: opened.append((url, clock.now, target.started)) or True)
    asyncio.run(target.open_when_ready(120))
    assert len(opened) == 1
    assert opened[0][0] == target.browser_url
    assert opened[0][1] >= 2.2 and opened[0][2] is True


@pytest.mark.parametrize("passed_socket", [False, True])
def test_actual_ephemeral_server_opens_after_readiness_and_url_connects(monkeypatch, passed_socket):
    import socket
    from urllib.parse import urlsplit

    async def run():
        lifespan_entered, release, opened = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def app(scope, receive, send):
            if scope["type"] == "lifespan":
                assert (await receive())["type"] == "lifespan.startup"
                lifespan_entered.set()
                await release.wait()
                await send({"type": "lifespan.startup.complete"})
                assert (await receive())["type"] == "lifespan.shutdown"
                await send({"type": "lifespan.shutdown.complete"})
            else:
                await send({"type": "http.response.start", "status": 200, "headers": []})
                await send({"type": "http.response.body", "body": b"ready"})

        with socket.socket() as listener:
            if passed_socket:
                listener.bind(("127.0.0.1", 0))
            target = browser.BrowserServer(
                uvicorn.Config(app, host="127.0.0.1", port=0, lifespan="on", ws="none", log_level="critical"),
                "http://127.0.0.1:0/videos/",
                SimpleNamespace(value=False, get_lock=nullcontext),
            )

            def launch(url):
                assert target.started
                assigned = target.servers[0].sockets[0].getsockname()[1]
                assert urlsplit(url).port == assigned and assigned > 0
                opened.set()
                return True

            monkeypatch.setattr(browser.webbrowser, "open", launch)
            task = asyncio.create_task(target.serve(sockets=[listener] if passed_socket else None))
            try:
                await asyncio.wait_for(lifespan_entered.wait(), 3)
                assert not target.started and not opened.is_set()
                release.set()
                await asyncio.wait_for(opened.wait(), 3)
                address = urlsplit(target.browser_url)
                reader, writer = await asyncio.wait_for(asyncio.open_connection(address.hostname, address.port), 3)
                try:
                    writer.write(b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
                    await writer.drain()
                    response = await asyncio.wait_for(reader.read(), 3)
                    assert b"200 OK" in response and b"ready" in response
                finally:
                    writer.close()
                    await writer.wait_closed()
            finally:
                target.should_exit = True
                release.set()
                await asyncio.wait_for(task, 3)
            assert asyncio.all_tasks() == {asyncio.current_task()}

    asyncio.run(run())


def test_readiness_timeout_is_bounded_and_reported(monkeypatch, caplog):
    target = server()
    clock = controlled_clock(monkeypatch)
    opened = Mock()
    monkeypatch.setattr(browser.webbrowser, "open", opened)
    asyncio.run(target.open_when_ready(1))
    opened.assert_not_called()
    assert 1 <= clock.now <= 1.1
    assert "readiness wait expired" in caplog.text


@pytest.mark.parametrize("outcome", ["startup-failure", "startup-exit", "shutdown", "ready-but-shutting-down"])
def test_serve_failure_shutdown_cancel_watcher_without_browser(monkeypatch, outcome):
    target = server()
    opened = Mock()
    monkeypatch.setattr(browser.webbrowser, "open", opened)

    async def serving(self, sockets=None):
        await asyncio.sleep(0)
        if outcome == "startup-failure":
            raise RuntimeError("controlled startup failure")
        if outcome == "startup-exit":
            raise SystemExit(3)
        self.should_exit = True
        self.started = outcome == "ready-but-shutting-down"
        await asyncio.sleep(0)

    monkeypatch.setattr(uvicorn.Server, "serve", serving)

    async def run():
        if outcome == "startup-failure":
            with pytest.raises(RuntimeError, match="controlled startup failure"):
                await target.serve()
        elif outcome == "startup-exit":
            with pytest.raises(SystemExit) as failure:
                await target.serve()
            assert failure.value.code == 3
        else:
            await target.serve()
        assert asyncio.all_tasks() == {asyncio.current_task()}

    asyncio.run(run())
    opened.assert_not_called()


def test_cancelled_serve_joins_watcher_without_browser(monkeypatch):
    target = server()
    opened = Mock()
    monkeypatch.setattr(browser.webbrowser, "open", opened)

    async def serving(self, sockets=None):
        await asyncio.Event().wait()

    monkeypatch.setattr(uvicorn.Server, "serve", serving)

    async def run():
        task = asyncio.create_task(target.serve())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert asyncio.all_tasks() == {asyncio.current_task()}

    asyncio.run(run())
    opened.assert_not_called()


@pytest.mark.parametrize("result", [False, OSError("browser unavailable")])
def test_browser_launch_failure_is_reported_without_failing_server(monkeypatch, caplog, result):
    target = server()
    target.started = True
    launch = Mock(return_value=result if result is False else True,
                  side_effect=result if isinstance(result, Exception) else None)
    monkeypatch.setattr(browser.webbrowser, "open", launch)
    asyncio.run(target.open_when_ready(float("inf")))
    launch.assert_called_once_with(target.browser_url)
    assert target.browser_opened.value
    assert "manually" in caplog.text


def test_reload_uses_serving_child_and_does_not_open_again(monkeypatch):
    socket = Mock()
    config = SimpleNamespace(should_reload=True, workers=1, bind_socket=lambda: socket)
    configure = Mock(return_value=config)
    monkeypatch.setattr(browser.uvicorn, "Config", configure)
    launch = Mock(return_value=True)
    monkeypatch.setattr(browser.webbrowser, "open", launch)
    supervised = []

    class Supervisor:
        def __init__(self, config, target, sockets):
            assert sockets == [socket]
            self.owner = target.__self__
            supervised.append(self.owner)

        def run(self):
            assert not self.owner.started
            launch.assert_not_called()  # Parent bound socket is not ready.
            self.owner.started = True
            asyncio.run(self.owner.open_when_ready(float("inf")))
            # Simulate replacement child retaining the same shared opened flag.
            replacement = browser.BrowserServer(config, self.owner.browser_url, self.owner.browser_opened)
            replacement.started = True
            asyncio.run(replacement.open_when_ready(float("inf")))

    monkeypatch.setattr(browser, "ChangeReload", Supervisor)
    browser.run_with_browser(host="127.0.0.1", port=8123, reload=True, url="http://127.0.0.1:8123/videos/")
    configure.assert_called_once_with("vacca_api.main:app", host="127.0.0.1", port=8123, reload=True, log_level="info")
    assert len(supervised) == 1
    launch.assert_called_once_with("http://127.0.0.1:8123/videos/")
    socket.close.assert_called_once()


@pytest.mark.parametrize("open_browser", [False, True])
@pytest.mark.parametrize("reload", [False, True])
def test_launcher_rejects_real_env_multiworker_config_before_startup(monkeypatch, open_browser, reload):
    from vacca_video import service
    from vacca_video.store import Store

    monkeypatch.setenv("WEB_CONCURRENCY", "2")
    real_config = uvicorn.Config
    configs = []

    def configure(*args, **kwargs):
        config = real_config(*args, **kwargs)
        configs.append(config)
        return config

    launch = Mock(side_effect=AssertionError("No server may run"))
    bind = Mock(side_effect=AssertionError("No socket may bind"))
    load = Mock(side_effect=AssertionError("No app may load"))
    recover = Mock(side_effect=AssertionError("No history may recover"))
    prepare = Mock(side_effect=AssertionError("No video storage may initialize"))
    monkeypatch.setattr(uvicorn, "Config", configure)
    monkeypatch.setattr(uvicorn, "run", launch)
    monkeypatch.setattr(real_config, "bind_socket", bind)
    monkeypatch.setattr(real_config, "load_app", load)
    monkeypatch.setattr(Store, "recover", recover)
    monkeypatch.setattr(service.VideoResources, "prepare", prepare)
    args = (["--open"] if open_browser else []) + (["--reload"] if reload else [])
    with pytest.raises(SystemExit, match="effective Uvicorn workers=2") as failure:
        run_api.main(args)
    assert "WEB_CONCURRENCY=1" in str(failure.value)
    assert len(configs) == 1 and configs[0].workers == 2 and not configs[0].loaded
    for operation in (launch, bind, load, recover, prepare):
        operation.assert_not_called()


@pytest.mark.parametrize("reload", [False, True])
def test_browser_helper_rejects_real_env_workers_before_supervisor_or_bind(monkeypatch, reload):
    monkeypatch.setenv("WEB_CONCURRENCY", "2")
    assert uvicorn.Config("unused:app", reload=reload).workers == 2
    bind = Mock(side_effect=AssertionError("No socket may bind"))
    context = Mock(side_effect=AssertionError("No worker state may initialize"))
    monkeypatch.setattr(uvicorn.Config, "bind_socket", bind)
    monkeypatch.setattr(browser, "get_context", context)
    with pytest.raises(SystemExit, match="effective Uvicorn workers=2"):
        browser.run_with_browser(host="127.0.0.1", port=8001, reload=reload, url="http://127.0.0.1:8001/videos/")
    bind.assert_not_called()
    context.assert_not_called()


def test_locked_uvicorn_supervisor_and_server_signatures():
    import inspect
    from uvicorn.supervisors import ChangeReload, Multiprocess

    # The removed unsupported branch cannot pass target to this locked API.
    assert tuple(inspect.signature(Multiprocess).parameters) == ("config", "sockets")
    config = uvicorn.Config("unused:app", reload=True, log_config=None)
    owner = browser.BrowserServer(config, "http://127.0.0.1:8001/videos/",
                                  SimpleNamespace(value=False, get_lock=nullcontext))
    inspect.signature(ChangeReload).bind(config, target=owner.run, sockets=[])
    supervisor = ChangeReload(config, target=owner.run, sockets=[])
    assert supervisor.target.__self__ is owner
    inspect.signature(uvicorn.Server.run).bind(owner, sockets=[])
    inspect.signature(uvicorn.Server.serve).bind(owner, sockets=[])
