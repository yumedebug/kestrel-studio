"""Hugging Face search and GGUF file discovery (Qt-free)."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable
from urllib.parse import quote

from app.core.credentials import CredentialStore
from app.core.gguf import quant_from_filename
from app.core.settings import SettingsStore

log = logging.getLogger(__name__)

HF_TOKEN_KEY = "huggingface:token"


class HFError(RuntimeError):
    def __init__(self, message: str, kind: str = "hf_error"):
        super().__init__(message)
        self.message = message
        self.kind = kind


@dataclass
class HFModel:
    repo_id: str
    author: str = ""
    downloads: int = 0
    likes: int = 0
    last_modified: str = ""
    tags: list[str] = field(default_factory=list)
    license: str = ""
    gated: str | bool = False
    architecture: str = ""
    context_length: int | None = None
    parameters: str = ""

    @property
    def license_label(self) -> str:
        if self.license:
            return self.license
        for tag in self.tags:
            if tag.startswith("license:"):
                return tag.split(":", 1)[1]
        return "Not specified"


@dataclass
class HFFile:
    filename: str
    size: int | None
    quant: str
    sha256: str = ""
    download_url: str = ""

    @property
    def size_label(self) -> str:
        return human_size(self.size)


@dataclass
class HFRepoDetails:
    repo_id: str
    author: str = ""
    downloads: int = 0
    likes: int = 0
    last_modified: str = ""
    tags: list[str] = field(default_factory=list)
    license: str = ""
    gated: str | bool = False
    pipeline_tag: str = ""
    files: list[HFFile] = field(default_factory=list)
    card_url: str = ""
    sha: str = ""
    error: str = ""


def human_size(size: int | None) -> str:
    if size is None:
        return "unknown size"
    if size < 1024:
        return f"{size} B"
    units = ["KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        value /= 1024.0
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
    return f"{size} B"


def parse_datetime(value: Any) -> str:
    if isinstance(value, datetime):
        return value.astimezone().strftime("%Y-%m-%d")
    if isinstance(value, str) and value:
        return value[:10]
    return ""


def _license_from(card_data: Any, tags: list[str]) -> str:
    if isinstance(card_data, dict):
        license_name = card_data.get("license")
        if isinstance(license_name, str) and license_name:
            return license_name
    for tag in tags:
        if tag.startswith("license:"):
            return tag.split(":", 1)[1]
    return ""


class HuggingFaceClient:
    """Thin wrapper around huggingface_hub with clear error translation."""

    def __init__(self, credentials: CredentialStore, settings: SettingsStore | None = None):
        self.credentials = credentials
        self.settings = settings

    @property
    def token(self) -> str | None:
        try:
            value = self.credentials.get(HF_TOKEN_KEY)
            return value or None
        except Exception:
            return None

    def set_token(self, token: str) -> None:
        token = (token or "").strip()
        if token:
            self.credentials.set(HF_TOKEN_KEY, token)
        else:
            self.credentials.delete(HF_TOKEN_KEY)

    def token_status(self) -> tuple[bool, str]:
        token = self.token
        if not token:
            return False, "No token saved (public models only)"
        return True, "Token stored in Windows Credential Manager"

    def validate_token(self) -> tuple[bool, str]:
        token = self.token
        if not token:
            return False, "Enter a token first."
        try:
            from huggingface_hub import HfApi

            who = HfApi().whoami(token=token)
            name = who.get("name") if isinstance(who, dict) else None
            return True, f"Token is valid{f' for {name}' if name else ''}."
        except Exception as exc:
            return False, f"Token check failed: {_hf_message(exc)}"

    # -- API ---------------------------------------------------------------
    def _api(self):
        from huggingface_hub import HfApi

        return HfApi()

    def search(
        self,
        query: str,
        *,
        limit: int = 25,
        sort: str = "downloads",
        gguf_only: bool = True,
    ) -> list[HFModel]:
        try:
            api = self._api()
            search_options = dict(
                search=query.strip() or None,
                filter="gguf" if gguf_only else None,
                sort=sort,
                limit=limit,
                full=True,
                cardData=True,
                token=self.token,
            )
            # ``direction`` was removed from HfApi.list_models in huggingface_hub
            # 2.x. Check the installed client API instead of assuming a version,
            # so search works with both the pinned 2.x release and older installs.
            try:
                parameters = inspect.signature(api.list_models).parameters
            except (TypeError, ValueError):
                parameters = {}
            if "direction" in parameters:
                search_options["direction"] = -1
            # Materialize inside the try block: list_models performs its HTTP
            # request lazily, so errors can otherwise escape untranslated.
            results = list(api.list_models(**search_options))
        except Exception as exc:
            raise HFError(_hf_message(exc), kind=_hf_kind(exc)) from exc

        models: list[HFModel] = []
        for info in results:
            repo_id = str(getattr(info, "id", "") or "")
            if not repo_id:
                continue
            tags = [str(tag) for tag in (getattr(info, "tags", None) or [])]
            card = getattr(info, "card_data", None)
            gguf = getattr(info, "gguf", None)
            arch = ""
            ctx = None
            params = ""
            if isinstance(gguf, dict):
                arch = str(gguf.get("architecture") or "")
                raw_ctx = gguf.get("context_length")
                if isinstance(raw_ctx, int):
                    ctx = raw_ctx
                raw_total = gguf.get("total")
                if isinstance(raw_total, int):
                    params = f"{raw_total / 1_000_000_000:.1f}B"
            models.append(
                HFModel(
                    repo_id=repo_id,
                    author=str(getattr(info, "author", "") or ""),
                    downloads=int(getattr(info, "downloads", 0) or 0),
                    likes=int(getattr(info, "likes", 0) or 0),
                    last_modified=parse_datetime(getattr(info, "last_modified", "")),
                    tags=tags,
                    license=_license_from(card, tags),
                    gated=getattr(info, "gated", False),
                    architecture=arch,
                    context_length=ctx,
                    parameters=params,
                )
            )
        return models

    def repo_details(self, repo_id: str) -> HFRepoDetails:
        try:
            info = self._api().model_info(
                repo_id, files_metadata=True, token=self.token
            )
        except Exception as exc:
            raise HFError(_hf_message(exc), kind=_hf_kind(exc)) from exc

        tags = [str(tag) for tag in (getattr(info, "tags", None) or [])]
        card = getattr(info, "card_data", None)
        sha = str(getattr(info, "sha", "") or "")
        files: list[HFFile] = []
        for sibling in getattr(info, "siblings", None) or []:
            filename = str(getattr(sibling, "rfilename", "") or "")
            if not filename.lower().endswith(".gguf"):
                continue
            lfs = getattr(sibling, "lfs", None)
            size = getattr(sibling, "size", None)
            sha256 = ""
            if lfs is not None:
                if size is None:
                    size = getattr(lfs, "size", None)
                sha256 = str(getattr(lfs, "sha256", "") or "")
            if not isinstance(size, int):
                size = None
            revision = sha or "main"
            url = (
                f"https://huggingface.co/{repo_id}/resolve/{quote(revision, safe='')}/"
                f"{quote(filename, safe='/')}?download=true"
            )
            files.append(
                HFFile(
                    filename=filename,
                    size=size,
                    quant=quant_from_filename(filename),
                    sha256=sha256,
                    download_url=url,
                )
            )
        files.sort(key=lambda f: (f.size or 0, f.filename))
        return HFRepoDetails(
            repo_id=str(getattr(info, "id", repo_id) or repo_id),
            author=str(getattr(info, "author", "") or ""),
            downloads=int(getattr(info, "downloads", 0) or 0),
            likes=int(getattr(info, "likes", 0) or 0),
            last_modified=parse_datetime(getattr(info, "last_modified", "")),
            tags=tags,
            license=_license_from(card, tags),
            gated=getattr(info, "gated", False),
            pipeline_tag=str(getattr(info, "pipeline_tag", "") or ""),
            files=files,
            card_url=f"https://huggingface.co/{repo_id}",
            sha=sha,
        )

    def search_with_fallback(self, query: str, limit: int = 25, sort: str = "downloads") -> list[HFModel]:
        try:
            return self.search(query, limit=limit, sort=sort)
        except HFError as first:
            # Some proxy setups reject the "gguf" filter; try an unfiltered search.
            if first.kind in {"network_error", "rate_limit"}:
                raise
            log.warning("GGUF-filtered search failed, retrying unfiltered: %s", first)
            return self.search(query, limit=limit, sort=sort, gguf_only=False)


def _hf_message(exc: Exception) -> str:
    name = type(exc).__name__
    text = str(exc)
    lower = text.lower()
    if any(word in lower for word in ("401", "unauthorized", "authentication", "token")):
        return (
            "Hugging Face rejected the request. Check your access token, or the model may "
            f"be gated. ({text[:300]})"
        )
    if any(word in lower for word in ("timeout", "timed out", "connection", "getaddrinfo", "ssl")):
        return f"Could not reach Hugging Face: {text[:300]}"
    if any(word in lower for word in ("404", "not found", "repository not found")):
        return f"Repository not found: {text[:300]}"
    if any(word in lower for word in ("429", "rate limit")):
        return f"Hugging Face rate limit reached. Wait and retry. ({text[:300]})"
    return f"{name}: {text[:400]}"


def _hf_kind(exc: Exception) -> str:
    text = str(exc).lower()
    name = type(exc).__name__.lower()
    if "401" in text or "unauthorized" in text or "authentication" in text:
        return "auth_error"
    if "404" in text or "not found" in text:
        return "not_found"
    if "429" in text or "rate limit" in text:
        return "rate_limit"
    if any(word in text for word in ("timeout", "connection", "getaddrinfo", "ssl")) or "connect" in name:
        return "network_error"
    return "hf_error"

