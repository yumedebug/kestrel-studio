"""Permission-gated Windows computer-use primitives."""

from __future__ import annotations

import ctypes
import os
import time
from ctypes import wintypes
from dataclasses import dataclass, field
from typing import Any
from typing import Callable

from app.agent.permissions import ActionRequest, ApprovalOutcome, PermissionManager


class ComputerUseError(RuntimeError):
    """Raised when an input action cannot be performed safely."""


ApprovalCallback = Callable[[ActionRequest], ApprovalOutcome]


@dataclass(frozen=True)
class UIElement:
    name: str
    control_type: str
    automation_id: str = ""
    rectangle: tuple[int, int, int, int] = (0, 0, 0, 0)
    enabled: bool = True


@dataclass(frozen=True)
class ComputerObservation:
    timestamp: float
    screenshot: Any = None
    window_title: str = ""
    elements: tuple[UIElement, ...] = ()
    backend: str = "fallback"

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "window_title": self.window_title,
            "backend": self.backend,
            "elements": [element.__dict__ for element in self.elements],
        }


class ComputerUse:
    """Small, explicit input surface; no arbitrary script execution is exposed."""

    def __init__(self, permissions: PermissionManager, approve: ApprovalCallback):
        self.permissions = permissions
        self.approve = approve

    def observe(self, hwnd: int | None = None) -> ComputerObservation:
        """Observe a window through pywinauto when available, otherwise Win32."""
        self._authorize("Observe the visible Windows UI.", "Inspect the current UI state.")
        if os.name != "nt":
            raise ComputerUseError("Computer Use observation is only supported on Windows.")
        try:
            from pywinauto import Desktop  # type: ignore[import-not-found]

            window = Desktop(backend="uia").window(handle=hwnd) if hwnd else Desktop(backend="uia").get_active()
            elements = []
            for item in window.descendants():
                try:
                    rect = item.rectangle()
                    elements.append(UIElement(
                        name=str(item.window_text() or ""),
                        control_type=str(item.element_info.control_type or ""),
                        automation_id=str(item.element_info.automation_id or ""),
                        rectangle=(rect.left, rect.top, rect.right, rect.bottom),
                        enabled=bool(item.is_enabled()),
                    ))
                except Exception:
                    continue
            return ComputerObservation(
                time.time(),
                window_title=str(window.window_text() or ""),
                elements=tuple(elements[:500]),
                backend="pywinauto-uia",
            )
        except ImportError:
            return self._fallback_observe(hwnd)
        except Exception as exc:
            raise ComputerUseError(f"UI Automation observation failed: {exc}") from exc

    def click_element(self, name: str = "", automation_id: str = "") -> ComputerObservation:
        """Click a UIA element by stable metadata, with coordinate fallback."""
        self._authorize(
            f"Click UI element {automation_id or name or '(unnamed)'}",
            "Activate the approved Windows UI element.",
        )
        if os.name != "nt":
            raise ComputerUseError("Computer Use input is only supported on Windows.")
        try:
            from pywinauto import Desktop  # type: ignore[import-not-found]

            window = Desktop(backend="uia").get_active()
            kwargs = {"title": name} if name else {"auto_id": automation_id}
            control = window.child_window(**kwargs)
            if not control.exists() or not control.is_enabled():
                raise ComputerUseError("The requested UI element was not found or is disabled.")
            control.click_input()
            return self.observe()
        except ImportError as exc:
            raise ComputerUseError(
                "pywinauto is not installed; use coordinate click or install the optional UIA dependency."
            ) from exc

    def verify(self, observation: ComputerObservation, *, name: str = "", text: str = "") -> bool:
        current = self.observe()
        if name:
            return any(element.name == name for element in current.elements)
        if text:
            return text.casefold() in " ".join(element.name for element in current.elements).casefold()
        return current.window_title != observation.window_title or current.to_dict() != observation.to_dict()

    @staticmethod
    def _fallback_observe(hwnd: int | None) -> ComputerObservation:
        titles: list[str] = []
        user32 = ctypes.windll.user32
        callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

        def callback(handle: int, _lparam: int) -> bool:
            if not user32.IsWindowVisible(handle):
                return True
            length = user32.GetWindowTextLengthW(handle)
            if length:
                buffer = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(handle, buffer, length + 1)
                titles.append(buffer.value)
            return True

        user32.EnumWindows(callback_type(callback), 0)
        title = ""
        if hwnd:
            length = user32.GetWindowTextLengthW(hwnd)
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            title = buffer.value
        return ComputerObservation(
            time.time(),
            window_title=title,
            elements=tuple(UIElement(name=item, control_type="Window") for item in titles[:200]),
            backend="win32-fallback",
        )


@dataclass
class ComputerUseLoop:
    """Observe -> understand -> plan -> act -> verify with bounded retries."""

    computer: ComputerUse
    max_retries: int = 2
    history: list[dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def understand(observation: ComputerObservation) -> dict[str, Any]:
        return {
            "window_title": observation.window_title,
            "backend": observation.backend,
            "element_count": len(observation.elements),
            "enabled_elements": [
                element.name for element in observation.elements if element.enabled and element.name
            ][:50],
        }

    @staticmethod
    def plan(understanding: dict[str, Any], action: str) -> dict[str, Any]:
        return {
            "action": action,
            "window_title": understanding.get("window_title", ""),
            "bounded_retries": True,
        }

    def run(self, action: Callable[[], Any], verify: Callable[[Any], bool]) -> Any:
        before = self.computer.observe()
        understanding = self.understand(before)
        plan = self.plan(understanding, "approved computer action")
        for attempt in range(self.max_retries + 1):
            result = action()
            verified = bool(verify(result))
            self.history.append(
                {
                    "attempt": attempt,
                    "before": before.to_dict(),
                    "understanding": understanding,
                    "plan": plan,
                    "verified": verified,
                }
            )
            if verified:
                return result
            if attempt < self.max_retries:
                before = self.computer.observe()
        raise ComputerUseError("Computer Use verification failed after bounded retries.")

    def _authorize(self, description: str, purpose: str) -> None:
        request = ActionRequest(
            kind="computer",
            description=description,
            purpose=purpose,
            source="user",
        )
        decision = self.permissions.evaluate(request)
        if decision.decision != "ask":
            raise ComputerUseError(f"Computer Use was not approved: {decision.reason}")
        outcome = self.approve(request)
        if not outcome.allowed:
            raise ComputerUseError("Computer Use action was rejected.")
        self.permissions.record_approval(decision, outcome)

    def move(self, x: int, y: int) -> None:
        self._authorize(f"Move pointer to ({x}, {y})", "Move the Windows pointer.")
        if os.name != "nt":
            raise ComputerUseError("Computer Use input is only supported on Windows.")
        ctypes.windll.user32.SetCursorPos(int(x), int(y))

    def click(self, x: int, y: int, button: str = "left") -> None:
        self._authorize(f"Click {button} at ({x}, {y})", "Click the Windows desktop.")
        self.move_without_prompt(x, y)
        flags = {"left": (2, 4), "right": (8, 16), "middle": (32, 64)}
        try:
            down, up = flags[button]
        except KeyError as exc:
            raise ComputerUseError(f"Unsupported mouse button: {button}") from exc
        ctypes.windll.user32.mouse_event(down, 0, 0, 0, 0)
        ctypes.windll.user32.mouse_event(up, 0, 0, 0, 0)

    def move_without_prompt(self, x: int, y: int) -> None:
        if os.name != "nt" or not ctypes.windll.user32.SetCursorPos(int(x), int(y)):
            raise ComputerUseError("Could not move the Windows pointer.")

    def type_text(self, text: str, interval: float = 0.0) -> None:
        if not isinstance(text, str) or len(text) > 4000:
            raise ComputerUseError("Text input must be a string of at most 4000 characters.")
        self._authorize("Type text into the focused Windows control.", "Type approved text.")
        if os.name != "nt":
            raise ComputerUseError("Computer Use input is only supported on Windows.")
        for char in text:
            self._send_unicode(char)
            if interval > 0:
                time.sleep(interval)

    @staticmethod
    def _send_unicode(char: str) -> None:
        class KeyboardInput(ctypes.Structure):
            _fields_ = [
                ("wVk", wintypes.WORD),
                ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
            ]

        class Input(ctypes.Structure):
            _fields_ = [("type", wintypes.DWORD), ("ki", KeyboardInput)]

        extra = wintypes.ULONG(0)
        events = (Input * 2)(
            Input(1, KeyboardInput(0, ord(char), 0x0004, 0, ctypes.pointer(extra))),
            Input(1, KeyboardInput(0, ord(char), 0x0004 | 0x0002, 0, ctypes.pointer(extra))),
        )
        sent = ctypes.windll.user32.SendInput(2, ctypes.byref(events), ctypes.sizeof(Input))
        if sent != 2:
            raise ComputerUseError("Windows rejected the text input event.")


__all__ = [
    "ComputerObservation",
    "ComputerUse",
    "ComputerUseError",
    "ComputerUseLoop",
    "UIElement",
]
