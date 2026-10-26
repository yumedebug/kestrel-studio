"""Backend interface shared by llama.cpp and cloud providers."""

from __future__ import annotations

from typing import Iterable, Iterator

from app.core.types import (
    CancelToken,
    GenerationParams,
    LoadOptions,
    ModelTarget,
    StreamEvent,
)


class InferenceBackend:
    """A loaded model that can stream chat completions.

    Implementations must be safe to call from a worker thread. ``unload`` is
    expected to stop child processes and release network clients.
    """

    name = "backend"
    last_warning = ""

    def load(self, target: ModelTarget, options: LoadOptions) -> None:
        raise NotImplementedError

    def unload(self) -> None:
        raise NotImplementedError

    def stream_chat(
        self,
        messages: Iterable[dict[str, str]],
        params: GenerationParams,
        cancel: CancelToken,
    ) -> Iterator[StreamEvent]:
        raise NotImplementedError

    def describe(self) -> dict[str, str]:
        return {"backend": self.name, "warning": self.last_warning}
