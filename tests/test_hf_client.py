from __future__ import annotations

from types import SimpleNamespace

from app.core.hf_client import HuggingFaceClient


class FakeHfApi:
    def __init__(self):
        self.calls: list[dict] = []

    def list_models(self, *, search=None, filter=None, sort=None, limit=None,
                    full=None, cardData=False, token=None, direction=None):
        self.calls.append(
            {
                "search": search,
                "filter": filter,
                "sort": sort,
                "limit": limit,
                "full": full,
                "cardData": cardData,
                "token": token,
                "direction": direction,
            }
        )
        return iter(
            [
                SimpleNamespace(
                    id="bartowski/example-GGUF",
                    author="bartowski",
                    downloads=123,
                    likes=9,
                    last_modified="2026-01-02T00:00:00Z",
                    tags=["gguf", "license:apache-2.0"],
                    card_data={"license": "apache-2.0"},
                    gguf=None,
                    gated=False,
                )
            ]
        )


class FakeHfApiV2:
    """Matches huggingface_hub 2.x, where list_models has no direction arg."""

    def __init__(self):
        self.calls: list[dict] = []

    def list_models(self, *, search=None, filter=None, sort=None, limit=None,
                    full=None, cardData=False, token=None):
        self.calls.append(
            {
                "search": search,
                "filter": filter,
                "sort": sort,
                "limit": limit,
                "full": full,
                "cardData": cardData,
                "token": token,
            }
        )
        return iter(
            [
                SimpleNamespace(
                    id="bartowski/example-GGUF",
                    author="bartowski",
                    downloads=123,
                    likes=9,
                    last_modified="2026-01-02T00:00:00Z",
                    tags=["gguf", "license:apache-2.0"],
                    card_data={"license": "apache-2.0"},
                    gguf=None,
                    gated=False,
                )
            ]
        )


def test_search_works_with_huggingface_hub_2x(credentials):
    client = HuggingFaceClient(credentials)
    api = FakeHfApiV2()
    client._api = lambda: api

    results = client.search("  llama  ", limit=25, sort="downloads", gguf_only=True)

    assert [model.repo_id for model in results] == ["bartowski/example-GGUF"]
    assert results[0].license == "apache-2.0"
    assert api.calls == [
        {
            "search": "llama",
            "filter": "gguf",
            "sort": "downloads",
            "limit": 25,
            "full": True,
            "cardData": True,
            "token": None,
        }
    ]


def test_search_passes_descending_direction_when_supported(credentials):
    client = HuggingFaceClient(credentials)
    api = FakeHfApi()
    client._api = lambda: api

    client.search("llama", sort="likes", gguf_only=False)

    assert api.calls[0]["direction"] == -1
    assert api.calls[0]["filter"] is None
    assert api.calls[0]["sort"] == "likes"

