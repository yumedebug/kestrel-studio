"""Reusable UI widgets: Markdown rendering, message bubbles, status chips."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Iterable

from PySide6.QtCore import QEvent, QPointF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QFontDatabase, QIcon, QPainter, QPalette, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from app.ui import icons
from app.ui.i18n import ja
from app.ui.theme import current_palette, markdown_stylesheet


def repolish(widget: QWidget) -> None:
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


def set_variant(widget: QWidget, variant: str) -> None:
    widget.setProperty("variant", variant)
    repolish(widget)


def icon_button(name: str, tooltip: str, *, size: int = 16) -> QPushButton:
    button = QPushButton()
    button.setProperty("variant", "icon")
    button.setCursor(Qt.PointingHandCursor)
    button.setToolTip(ja(tooltip))
    button.setIcon(_sized_icon(name, size))
    button.setIconSize(QSize(size, size))
    button.setFixedSize(size + 16, size + 16)
    return button


def _sized_icon(name: str, size: int) -> QIcon:
    if name not in icons._DRAWERS:  # noqa: SLF001 - icon set is local
        return QIcon()
    colors = current_palette()
    return icons.make_icon(name, colors["text_muted"], colors["text"])


def section_title(text: str) -> QLabel:
    label = QLabel(ja(text))
    label.setObjectName("sectionTitle")
    return label


def muted_label(text: str = "") -> QLabel:
    label = QLabel(ja(text))
    label.setProperty("variant", "muted")
    label.setWordWrap(True)
    return label


class Card(QFrame):
    def __init__(self, parent: QWidget | None = None, *, soft: bool = False):
        super().__init__(parent)
        self.setProperty("card", "soft" if soft else "true")


class StatusChip(QFrame):
    """Small pill showing the shared inference state."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("statusChip")
        self.setFixedHeight(28)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 14, 0)
        layout.setSpacing(9)
        self.dot = QLabel("●")
        self.dot.setFixedWidth(10)
        self.text = QLabel("No model")
        self.text.setStyleSheet("font-size: 9pt;")
        layout.addWidget(self.dot)
        layout.addWidget(self.text)
        self._state = "idle"
        self._apply()

    def set_status(self, state: str, text: str, detail: str = "") -> None:
        self._state = state
        self.text.setText(ja(text))
        self.setToolTip(ja(detail))
        self._apply()

    def _apply(self) -> None:
        c = current_palette()
        color = {
            "ready": c["success"],
            "loading": c["warning"],
            "error": c["danger"],
            "idle": c["text_faint"],
            "busy": c["accent"],
        }.get(self._state, c["text_faint"])
        self.setStyleSheet(
            "QFrame#statusChip {"
            f" background: {c['bg_input']}; border: 1px solid {c['border']};"
            " border-radius: 13px; }"
        )
        self.dot.setStyleSheet(f"color: {color}; font-size: 10pt; background: transparent;")
        self.text.setStyleSheet(
            f"color: {c['text_muted'] if self._state == 'idle' else c['text']};"
            " background: transparent; font-size: 9pt;"
        )


class NoticeBar(QFrame):
    def __init__(self, text: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("notice")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 10, 10)
        layout.setSpacing(10)
        self.label = QLabel(ja(text))
        self.label.setWordWrap(True)
        self.label.setStyleSheet("background: transparent;")
        self.close_button = icon_button("plus", "Dismiss")
        self.close_button.setIcon(QIcon())
        self.close_button.setText("×")
        self.close_button.setFixedSize(24, 24)
        layout.addWidget(self.label, 1)
        layout.addWidget(self.close_button, 0, Qt.AlignTop)
        self.close_button.clicked.connect(self.hide)
        self.hide()

    def show_message(self, text: str) -> None:
        self.label.setText(ja(text))
        self.show()


class EmptyState(QWidget):
    def __init__(
        self,
        icon_name: str,
        title: str,
        description: str,
        action_text: str = "",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 40, 24, 40)
        layout.setSpacing(10)
        layout.setAlignment(Qt.AlignCenter)
        icon_label = QLabel()
        icon_label.setPixmap(
            icons.make_icon(icon_name, current_palette()["text_faint"]).pixmap(48, 48)
        )
        icon_label.setAlignment(Qt.AlignCenter)
        title_label = QLabel(ja(title))
        title_label.setProperty("variant", "heading")
        title_label.setAlignment(Qt.AlignCenter)
        desc = QLabel(ja(description))
        desc.setProperty("variant", "muted")
        desc.setAlignment(Qt.AlignCenter)
        desc.setWordWrap(True)
        desc.setMaximumWidth(460)
        layout.addWidget(icon_label, 0, Qt.AlignCenter)
        layout.addWidget(title_label)
        layout.addWidget(desc, 0, Qt.AlignCenter)
        self.action = QPushButton(ja(action_text))
        self.action.setProperty("variant", "primary")
        self.action.setCursor(Qt.PointingHandCursor)
        self.action.setVisible(bool(action_text))
        layout.addSpacing(6)
        layout.addWidget(self.action, 0, Qt.AlignCenter)


class RichTextBlock(QTextBrowser):
    """Auto-sized Markdown document without scrollbars."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setOpenExternalLinks(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.document().setDocumentMargin(0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        c = current_palette()
        # QTextBrowser renders Markdown through QTextDocument, so widget QSS alone
        # does not reliably set the document's foreground on Windows.
        browser_palette = self.palette()
        browser_palette.setColor(QPalette.Text, QColor(c["text"]))
        browser_palette.setColor(QPalette.Base, QColor(c["bg_elevated"]))
        browser_palette.setColor(QPalette.Highlight, QColor(c["selection"]))
        browser_palette.setColor(QPalette.HighlightedText, QColor(c["text"]))
        self.setPalette(browser_palette)
        self.setStyleSheet(
            "QTextBrowser { background: transparent; border: none; "
            f"color: {c['text']}; selection-background-color: {c['selection']}; }}"
        )
        self._styles = markdown_stylesheet("dark")
        self._syncing = False

    def set_markdown(self, text: str) -> None:
        self._styles = markdown_stylesheet(
            getattr(QApplication.instance(), "_kestrel_theme", "dark")
        )
        self.document().setDefaultStyleSheet(self._styles)
        self.setMarkdown(text)
        self._sync_height()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self._syncing:
            return
        QTimer.singleShot(0, self._sync_height)

    def changeEvent(self, event) -> None:  # type: ignore[override]
        super().changeEvent(event)
        if event.type() in (QEvent.PaletteChange, QEvent.StyleChange):
            self._styles = markdown_stylesheet(
                getattr(QApplication.instance(), "_kestrel_theme", "dark")
            )
            self.document().setDefaultStyleSheet(self._styles)
            c = current_palette()
            p = self.palette()
            p.setColor(QPalette.Text, QColor(c["text"]))
            p.setColor(QPalette.Base, QColor(c["bg_elevated"]))
            p.setColor(QPalette.Highlight, QColor(c["selection"]))
            p.setColor(QPalette.HighlightedText, QColor(c["text"]))
            self.setPalette(p)

    def _sync_height(self) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            width = max(80, self.viewport().width())
            self.document().setTextWidth(width)
            height = int(self.document().size().height()) + 4
            self.setFixedHeight(max(18, min(height, 20000)))
        finally:
            self._syncing = False


class CodeBlockWidget(QFrame):
    def __init__(self, code: str, language: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("codeBlock")
        self._code = code
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(14, 8, 10, 6)
        header_layout.setSpacing(8)
        lang_label = QLabel((language or "code").upper())
        lang_label.setProperty("variant", "faint")
        font = QFont("Consolas", 8)
        font.setLetterSpacing(QFont.AbsoluteSpacing, 0.6)
        lang_label.setFont(font)
        self.copy_button = QPushButton(ja("Copy"))
        self.copy_button.setProperty("variant", "ghost")
        self.copy_button.setCursor(Qt.PointingHandCursor)
        self.copy_button.setToolTip(ja("Copy this code block"))
        self.copy_button.clicked.connect(self._copy)
        header_layout.addWidget(lang_label)
        header_layout.addStretch(1)
        header_layout.addWidget(self.copy_button)
        layout.addWidget(header)

        self.view = QPlainTextEdit()
        self.view.setObjectName("codeView")
        self.view.setReadOnly(True)
        self.view.setPlainText(code)
        self.view.setFrameShape(QFrame.NoFrame)
        self.view.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        font.setPointSizeF(9.5)
        self.view.setFont(font)
        layout.addWidget(self.view)
        self._adjust_height()
        self.view.document().contentsChanged.connect(self._adjust_height)

    def _adjust_height(self) -> None:
        lines = self._code.count("\n") + 1
        metrics = self.view.fontMetrics()
        height = metrics.lineSpacing() * lines + 18
        self.view.setFixedHeight(max(36, min(height, 420)))

    def _copy(self) -> None:
        QApplication.clipboard().setText(self._code)
        self.copy_button.setText(ja("Copied"))
        QTimer.singleShot(1200, lambda: self.copy_button.setText(ja("Copy")))


class MarkdownView(QWidget):
    """Renders Markdown; code fences become blocks with Copy buttons.

    While a response is streaming a single document is updated in place for
    speed; when it finishes the content is re-rendered as segmented blocks.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(8)
        self._layout.setAlignment(Qt.AlignTop)
        self._stream_view: RichTextBlock | None = None
        self._segments: list[QWidget] = []
        self._final = False
        self._text = ""

    def clear(self) -> None:
        for widget in self._segments:
            self._layout.removeWidget(widget)
            widget.deleteLater()
        self._segments.clear()
        if self._stream_view is not None:
            self._layout.removeWidget(self._stream_view)
            self._stream_view.deleteLater()
            self._stream_view = None
        self._final = False
        self._text = ""

    def set_streaming(self, text: str) -> None:
        self._text = text
        if self._final:
            self.clear()
        if self._stream_view is None:
            for widget in self._segments:
                self._layout.removeWidget(widget)
                widget.deleteLater()
            self._segments.clear()
            self._stream_view = RichTextBlock()
            self._layout.addWidget(self._stream_view)
        self._stream_view.set_markdown(text)

    def set_markdown(self, text: str) -> None:
        self._text = text
        self._final = True
        if self._stream_view is not None:
            self._layout.removeWidget(self._stream_view)
            self._stream_view.deleteLater()
            self._stream_view = None
        for widget in self._segments:
            self._layout.removeWidget(widget)
            widget.deleteLater()
        self._segments.clear()
        styles = markdown_stylesheet(
            getattr(QApplication.instance(), "_kestrel_theme", "dark")
        )
        for kind, value, language in _segments(text):
            if kind == "code":
                widget = CodeBlockWidget(value, language)
            else:
                widget = RichTextBlock()
                widget.set_markdown(value)
            self._layout.addWidget(widget)
            self._segments.append(widget)
        if not self._segments:
            placeholder = QLabel(" ")
            self._layout.addWidget(placeholder)
            self._segments.append(placeholder)

    @property
    def raw_text(self) -> str:
        return self._text


def _segments(text: str) -> Iterable[tuple[str, str, str]]:
    """Split Markdown into prose/code sections, tolerating an open fence."""
    segments: list[tuple[str, str, str]] = []
    prose: list[str] = []
    code: list[str] = []
    language = ""
    in_code = False

    def flush_prose() -> None:
        value = "".join(prose).strip("\n")
        if value.strip():
            segments.append(("prose", value, ""))
        prose.clear()

    def flush_code() -> None:
        segments.append(("code", "".join(code).rstrip("\n"), language))
        code.clear()

    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_code:
                flush_code()
                in_code = False
                language = ""
            else:
                flush_prose()
                in_code = True
                language = stripped[3:].strip()
            continue
        if in_code:
            code.append(line)
        else:
            prose.append(line)
    if in_code:
        flush_code()
    else:
        flush_prose()
    return segments


class MessageBubble(QFrame):
    copy_requested = Signal(str)

    def __init__(
        self,
        role: str,
        content: str,
        *,
        model_label: str = "",
        timestamp: str = "",
        streaming: bool = False,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.role = role
        self._raw = content
        self.setObjectName("bubbleUser" if role == "user" else "bubbleAssistant")
        self.setMaximumWidth(900 if role == "assistant" else 780)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 16)
        layout.setSpacing(10)

        header = QHBoxLayout()
        header.setSpacing(8)
        role_text = {"user": "You", "assistant": "Assistant", "system": "System"}.get(
            role, role.title()
        )
        role_label = QLabel(ja(role_text))
        role_label.setStyleSheet(
            f"color: {current_palette()['text_muted']}; font-weight: 600; font-size: 9pt;"
        )
        header.addWidget(role_label)
        if model_label and role == "assistant":
            model = QLabel(model_label)
            model.setProperty("variant", "faint")
            model.setStyleSheet("font-size: 8.5pt;")
            header.addWidget(model)
        header.addStretch(1)
        self.copy_button = icon_button("copy", "Copy message")
        self.copy_button.setVisible(False)
        self.copy_button.clicked.connect(self._copy)
        header.addWidget(self.copy_button)
        if timestamp:
            time_label = QLabel(_format_time(timestamp))
            time_label.setProperty("variant", "faint")
            time_label.setStyleSheet("font-size: 8pt;")
            header.addWidget(time_label)
        layout.addLayout(header)

        self.markdown = MarkdownView()
        if streaming:
            self.markdown.set_streaming(content)
        else:
            self.markdown.set_markdown(content)
            self.copy_button.setVisible(bool(content))
        layout.addWidget(self.markdown)

    def set_streaming_text(self, text: str) -> None:
        self._raw = text
        self.markdown.set_streaming(text)

    def finalize(self, text: str) -> None:
        self._raw = text
        self.markdown.set_markdown(text)
        self.copy_button.setVisible(bool(text))

    def set_error(self, text: str) -> None:
        self.finalize(text)

    def _copy(self) -> None:
        QApplication.clipboard().setText(self._raw)
        self.copy_requested.emit(self._raw)
        self.copy_button.setIcon(QIcon())
        self.copy_button.setText("✓")
        QTimer.singleShot(1100, self._restore_copy_icon)

    def _restore_copy_icon(self) -> None:
        self.copy_button.setText("")
        self.copy_button.setIcon(_sized_icon("copy", 16))


def _format_time(value: str) -> str:
    try:
        dt = datetime.fromisoformat(value)
        return dt.astimezone().strftime("%H:%M")
    except (ValueError, TypeError):
        return ""


def make_separator() -> QFrame:
    line = QFrame()
    line.setFixedHeight(1)
    line.setStyleSheet(f"background: {current_palette()['border']}; border: none;")
    return line
