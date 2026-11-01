from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from app.agent.permissions import ActionRequest, PermissionManager
from app.agent.sessions import AgentStore
from app.agent.tools import create_registry
from app.agent.tools.registry import ToolCall, ToolContext
from app.agent.web_search import search_web


class FakeResponse:
    def __init__(self, body: str, url: str = "https://html.duckduckgo.com/html/"):
        self.body = body.encode()
        self.url = url

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _limit: int = -1):
        return self.body

    def geturl(self):
        return self.url


def test_search_parses_and_validates_results() -> None:
    html = """
    <a class="result__a" href="https://example.com/a">Example <b>title</b></a>
    <div class="result__snippet">A useful snippet.</div>
    <a class="result__a" href="javascript:alert(1)">Unsafe</a>
    """
    with patch("app.agent.web_search.urlopen", return_value=FakeResponse(html)):
        results = search_web("hello", limit=5)
    assert len(results) == 1
    assert results[0].title == "Example title"
    assert results[0].url == "https://example.com/a"
    assert results[0].snippet == "A useful snippet."


def test_search_rejects_unapproved_redirect() -> None:
    with patch(
        "app.agent.web_search.urlopen",
        return_value=FakeResponse("", "https://evil.example/redirect"),
    ):
        try:
            search_web("hello")
        except Exception as exc:
            assert "unapproved domain" in str(exc)
        else:
            raise AssertionError("unsafe redirect was accepted")


def test_web_search_permission_is_domain_scoped_and_grantable(tmp_path: Path, settings) -> None:
    store = AgentStore(tmp_path / "agent.db")
    try:
        manager = PermissionManager(store, settings)
        request = ActionRequest(
            kind="web_search",
            target="https://html.duckduckgo.com/html/",
        )
        decision = manager.evaluate(request)
        assert decision.decision == "ask"
        assert decision.category == "network"
        assert decision.grantable is True
        grant = manager.add_grant(decision, origin="user")
        assert grant.target == "html.duckduckgo.com"
        assert manager.evaluate(request).decision == "allow"
    finally:
        store.close()


def test_web_search_is_registered_and_mockable(tmp_path: Path) -> None:
    registry = create_registry()
    assert registry.has("web_search")
    html = '<a class="result__a" href="https://example.com">Example</a>'
    store = AgentStore(tmp_path / "agent.db")
    with patch("app.agent.web_search.urlopen", return_value=FakeResponse(html)):
        result = registry.call(ToolCall("web_search", {"query": "hello", "limit": 1}),
                               ToolContext(store=store))
    assert result["ok"] is True
    store.close()
