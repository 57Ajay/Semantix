from __future__ import annotations

import threading
import time

import pytest

from semantix import client
from semantix.config import Config
from semantix.engine import Engine
from semantix.server import Daemon


@pytest.fixture
def daemon(home, embedder):
    engine = Engine(home, Config(), embedder)
    instance = Daemon(home, engine)
    thread = threading.Thread(target=instance.serve, daemon=True)
    thread.start()
    deadline = time.time() + 5
    while client.ping(home) is None and time.time() < deadline:
        time.sleep(0.02)
    yield instance
    instance.stop()
    thread.join(timeout=5)


def test_ping_and_status(daemon, home):
    state = client.ping(home)
    assert state["model_state"] == "ready"
    status = client.request(home, "status")
    assert status["model"]["dimension"] == 768 and status["roots"] == []


def test_sync_and_search_over_the_socket(daemon, home, project):
    plan = client.request(home, "plan", {"targets": [str(project)]})
    assert plan["totals"]["files"] == 4
    events = list(client.stream(home, "execute", {"plan_id": plan["id"]}))
    assert events[0]["event"] == "start"
    assert any(event["event"] == "progress" for event in events)
    assert events[-1]["result"]["embedded"] == 4
    result = client.request(home, "search", {"query": "update ecs service cluster"})
    assert result["hits"][0]["path"].endswith("deploy.py")
    keyword = client.request(home, "search", {"query": "boto3", "mode": "keyword"})
    assert keyword["hits"][0]["path"].endswith("deploy.py") and keyword["hits"][0]["score"] is None


def test_errors_are_reported_not_raised_in_the_service(daemon, home):
    with pytest.raises(client.RemoteError, match="expired"):
        list(client.stream(home, "execute", {"plan_id": "missing"}))
    with pytest.raises(client.RemoteError, match="unknown operation"):
        client.request(home, "bogus")
    assert client.ping(home) is not None


def test_shutdown_removes_the_socket(daemon, home):
    assert client.stop(home) is True
    assert client.ping(home) is None
    assert not home.socket.exists()


def test_request_without_a_service_fails_cleanly(home):
    with pytest.raises(client.DaemonUnavailable):
        client.request(home, "status")
    assert client.stop(home) is False
