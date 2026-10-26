"""Application theme: palettes, QSS and Markdown document styling."""

from __future__ import annotations

from string import Template

from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication

from app.constants import THEMES

_QSS = Template(
    """
* {
    outline: none;
}
QWidget {
    color: $text;
    background: transparent;
    font-family: "Segoe UI Variable Text", "Segoe UI", "Yu Gothic UI", "Meiryo UI", sans-serif;
    font-size: ${font_size}pt;
}
QMainWindow, QDialog, QWidget#root {
    background: $bg;
}
QToolTip {
    background: $bg_elevated;
    color: $text;
    border: 1px solid $border_strong;
    border-radius: 6px;
    padding: 6px 8px;
}
QFrame#sidebar {
    background: $bg_elevated;
    border-right: 1px solid $border;
}
QFrame#topbar {
    background: $bg;
    border-bottom: 1px solid $border;
}
QLabel {
    background: transparent;
}
QLabel[variant="muted"] {
    color: $text_muted;
}
QLabel[variant="faint"] {
    color: $text_faint;
}
QLabel[variant="heading"] {
    font-size: ${heading_size}pt;
    font-weight: 600;
    color: $text;
}
QLabel[variant="title"] {
    font-size: ${title_size}pt;
    font-weight: 600;
    color: $text;
}
QLabel[variant="mono"] {
    font-family: "Cascadia Mono", "Consolas", monospace;
}
QLabel[variant="danger"] {
    color: $danger;
}
QLabel[variant="success"] {
    color: $success;
}
QLabel[variant="warning"] {
    color: $warning;
}
QLabel#appName {
    font-size: 13pt;
    font-weight: 600;
}
QLabel#appTagline {
    color: $text_faint;
    font-size: 8pt;
}
QLabel#pageTitle {
    font-size: 15pt;
    font-weight: 600;
}
QLabel#sectionTitle {
    font-size: 11pt;
    font-weight: 600;
}
QFrame[card="true"] {
    background: $bg_elevated;
    border: 1px solid $border;
    border-radius: 14px;
}
QFrame[card="soft"] {
    background: $bg_input;
    border: 1px solid $border;
    border-radius: 12px;
}
QFrame#bubbleAssistant {
    background: $bg_elevated;
    border: 1px solid $border;
    border-radius: 12px;
}
QFrame#bubbleUser {
    background: $user_bg;
    border: 1px solid $border_strong;
    border-radius: 12px;
}
QFrame#codeBlock {
    background: $code_bg;
    border: 1px solid $border;
    border-radius: 9px;
}
QFrame#notice {
    background: $accent_soft;
    border: 1px solid $accent;
    border-radius: 9px;
}
QFrame#dropZone {
    background: $bg_input;
    border: 1px dashed $border_strong;
    border-radius: 10px;
}
QPushButton {
    background: $bg_input;
    color: $text;
    border: 1px solid $border_strong;
    border-radius: 10px;
    padding: 10px 18px;
    min-height: 24px;
}
QPushButton:hover {
    background: $bg_hover;
    border-color: $text_faint;
}
QPushButton:pressed {
    background: $bg_elevated;
}
QPushButton:disabled {
    color: $text_faint;
    border-color: $border;
    background: $bg_elevated;
}
QPushButton[variant="primary"] {
    background: $accent;
    color: #ffffff;
    border: 1px solid $accent;
    font-weight: 600;
}
QPushButton[variant="primary"]:hover {
    background: $accent_hover;
    border-color: $accent_hover;
}
QPushButton[variant="primary"]:disabled {
    background: $accent_soft;
    border-color: $accent_soft;
    color: $text_faint;
}
QPushButton[variant="danger"] {
    color: $danger;
    border-color: $danger;
    background: transparent;
}
QPushButton[variant="danger"]:hover {
    background: rgba(239, 91, 107, 0.12);
}
QPushButton[variant="ghost"] {
    background: transparent;
    border: 1px solid transparent;
    color: $text_muted;
    padding: 8px 12px;
}
QPushButton[variant="ghost"]:hover {
    background: $bg_hover;
    color: $text;
}
QPushButton[variant="nav"] {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 10px;
    color: $text_muted;
    text-align: left;
    padding: 12px 16px;
    font-weight: 500;
}
QPushButton[variant="nav"]:hover {
    background: $bg_hover;
    color: $text;
}
QPushButton[variant="nav"]:checked {
    background: $accent_soft;
    color: $accent_hover;
    font-weight: 600;
}
QPushButton[variant="chip"] {
    border-radius: 13px;
    padding: 8px 15px;
    min-height: 24px;
    background: $bg_input;
    color: $text_muted;
    font-size: 9pt;
}
QPushButton[variant="chip"]:hover {
    color: $text;
    border-color: $accent;
}
QPushButton[variant="chip"]:checked {
    background: $accent_soft;
    border-color: $accent;
    color: $accent_hover;
}
QPushButton[variant="icon"] {
    padding: 5px;
    border-radius: 7px;
    min-width: 22px;
    min-height: 22px;
    border: 1px solid transparent;
    background: transparent;
    color: $text_muted;
}
QPushButton[variant="icon"]:hover {
    background: $bg_hover;
    color: $text;
}
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox {
    background: $bg_input;
    color: $text;
    border: 1px solid $border_strong;
    border-radius: 10px;
    padding: 11px 14px;
    selection-background-color: $selection;
    selection-color: $text;
}
QSpinBox, QDoubleSpinBox {
    min-width: 96px;
    padding-right: 30px;
}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
    width: 20px;
    subcontrol-origin: border;
    border: none;
    background: transparent;
    margin: 3px 4px;
}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: $accent;
}
QLineEdit:disabled, QPlainTextEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
    color: $text_faint;
    background: $bg_elevated;
}
QLineEdit[readOnly="true"] {
    color: $text_muted;
}
QComboBox {
    background: $bg_input;
    color: $text;
    border: 1px solid $border_strong;
    border-radius: 10px;
    padding: 10px 14px;
    min-height: 26px;
}
QComboBox:hover {
    border-color: $text_faint;
}
QComboBox:focus {
    border-color: $accent;
}
QComboBox::drop-down {
    border: none;
    width: 28px;
}
QComboBox QAbstractItemView {
    background: $bg_elevated;
    color: $text;
    border: 1px solid $border_strong;
    border-radius: 8px;
    padding: 4px;
    selection-background-color: $accent_soft;
    selection-color: $text;
    outline: none;
}
QCheckBox, QRadioButton {
    spacing: 10px;
    color: $text;
    background: transparent;
}
QCheckBox::indicator, QRadioButton::indicator {
    width: 16px;
    height: 16px;
}
QCheckBox::indicator:unchecked {
    border: 1px solid $border_strong;
    background: $bg_input;
    border-radius: 4px;
}
QCheckBox::indicator:checked {
    border: 1px solid $accent;
    background: $accent;
    border-radius: 4px;
}
QCheckBox::indicator:disabled {
    border-color: $border;
    background: $bg_elevated;
}
QSlider::groove:horizontal {
    height: 4px;
    background: $border;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: $accent;
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::sub-page:horizontal {
    background: $accent;
    border-radius: 2px;
}
QScrollArea {
    background: transparent;
    border: none;
}
QScrollArea > QWidget > QWidget {
    background: transparent;
}
QScrollBar:vertical {
    background: transparent;
    width: 10px;
    margin: 2px;
}
QScrollBar::handle:vertical {
    background: $border_strong;
    min-height: 28px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover {
    background: $text_faint;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: transparent;
}
QScrollBar:horizontal {
    background: transparent;
    height: 10px;
    margin: 2px;
}
QScrollBar::handle:horizontal {
    background: $border_strong;
    min-width: 28px;
    border-radius: 4px;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: transparent;
}
QListWidget, QTreeWidget, QTableWidget {
    background: transparent;
    border: none;
    outline: none;
}
QListWidget::item {
    border-radius: 8px;
    padding: 10px 12px;
    margin: 2px 1px;
    color: $text;
}
QListWidget::item:hover {
    background: $bg_hover;
}
QListWidget::item:selected {
    background: $accent_soft;
    color: $text;
}
QTableWidget {
    background: $bg_elevated;
    border: 1px solid $border;
    border-radius: 10px;
    gridline-color: $border;
    alternate-background-color: $bg_input;
}
QTableWidget::item {
    padding: 10px 12px;
    border: none;
}
QTableWidget::item:selected {
    background: $accent_soft;
    color: $text;
}
QHeaderView::section {
    background: $bg_input;
    color: $text_muted;
    border: none;
    border-bottom: 1px solid $border;
    padding: 10px 12px;
    font-weight: 600;
}
QTableCornerButton::section {
    background: $bg_input;
    border: none;
}
QProgressBar {
    background: $bg_input;
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
    color: transparent;
}
QProgressBar::chunk {
    background: $accent;
    border-radius: 4px;
}
QTabWidget::pane {
    border: none;
    background: transparent;
}
QTabBar::tab {
    background: transparent;
    color: $text_muted;
    padding: 8px 14px;
    border-bottom: 2px solid transparent;
}
QTabBar::tab:selected {
    color: $text;
    border-bottom: 2px solid $accent;
}
QMenu {
    background: $bg_elevated;
    color: $text;
    border: 1px solid $border_strong;
    border-radius: 8px;
    padding: 5px;
}
QMenu::item {
    padding: 6px 22px 6px 12px;
    border-radius: 5px;
}
QMenu::item:selected {
    background: $accent_soft;
}
QMenu::separator {
    height: 1px;
    background: $border;
    margin: 4px 6px;
}
QSplitter::handle {
    background: $border;
    width: 1px;
    height: 1px;
}
QToolButton {
    background: transparent;
    border: none;
    border-radius: 6px;
    padding: 6px;
    color: $text_muted;
}
QToolButton:hover {
    background: $bg_hover;
    color: $text;
}
QGroupBox {
    border: 1px solid $border;
    border-radius: 10px;
    margin-top: 16px;
    padding: 16px 14px 12px 14px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 4px;
    color: $text_muted;
}
QMessageBox {
    background: $bg_elevated;
}
QMessageBox QLabel {
    color: $text;
}
QStatusBar {
    background: $bg_elevated;
    color: $text_muted;
    border-top: 1px solid $border;
}
QTextBrowser {
    background: transparent;
    border: none;
    color: $text;
    selection-background-color: $selection;
}
QPlainTextEdit#codeView {
    background: transparent;
    border: none;
    color: $text;
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 9.5pt;
    selection-background-color: $selection;
}
"""
)


def palette(theme: str) -> dict[str, str]:
    return THEMES.get(theme, THEMES["dark"])


def build_stylesheet(theme: str, font_size: int = 10) -> str:
    colors = dict(palette(theme))
    colors["font_size"] = str(font_size)
    colors["heading_size"] = str(font_size + 2)
    colors["title_size"] = str(font_size + 3)
    return _QSS.substitute(colors)


def markdown_stylesheet(theme: str, font_size: int = 10) -> str:
    c = palette(theme)
    return f"""
    body {{ color: {c['text']}; font-family: "Segoe UI Variable Text", "Segoe UI", "Yu Gothic UI", sans-serif;
            font-size: {font_size}pt; }}
    p {{ margin-top: 2px; margin-bottom: 8px; }}
    h1, h2, h3, h4 {{ color: {c['text']}; margin-top: 14px; margin-bottom: 6px; }}
    h1 {{ font-size: {font_size + 4}pt; }}
    h2 {{ font-size: {font_size + 3}pt; }}
    h3 {{ font-size: {font_size + 1}pt; }}
    a {{ color: {c['accent']}; text-decoration: none; }}
    code {{ font-family: "Cascadia Mono", "Consolas", monospace;
            background-color: {c['bg_input']}; color: {c['text']}; }}
    pre {{ font-family: "Cascadia Mono", "Consolas", monospace;
           background-color: {c['code_bg']}; color: {c['text']}; }}
    blockquote {{ color: {c['text_muted']}; border-left: 3px solid {c['border_strong']};
                  margin-left: 4px; padding-left: 10px; }}
    table {{ border-collapse: collapse; }}
    th, td {{ border: 1px solid {c['border_strong']}; padding: 5px; }}
    th {{ background-color: {c['bg_input']}; }}
    hr {{ border: 1px solid {c['border']}; }}
    li {{ margin-bottom: 3px; }}
    """


def apply_theme(app: QApplication, theme: str, font_size: int = 10) -> None:
    colors = palette(theme)
    app.setStyle("Fusion")
    qpalette = QPalette()
    qpalette.setColor(QPalette.Window, QColor(colors["bg"]))
    qpalette.setColor(QPalette.WindowText, QColor(colors["text"]))
    qpalette.setColor(QPalette.Base, QColor(colors["bg_input"]))
    qpalette.setColor(QPalette.AlternateBase, QColor(colors["bg_elevated"]))
    qpalette.setColor(QPalette.Text, QColor(colors["text"]))
    qpalette.setColor(QPalette.Button, QColor(colors["bg_input"]))
    qpalette.setColor(QPalette.ButtonText, QColor(colors["text"]))
    qpalette.setColor(QPalette.Highlight, QColor(colors["accent"]))
    qpalette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    qpalette.setColor(QPalette.ToolTipBase, QColor(colors["bg_elevated"]))
    qpalette.setColor(QPalette.ToolTipText, QColor(colors["text"]))
    qpalette.setColor(QPalette.PlaceholderText, QColor(colors["text_faint"]))
    qpalette.setColor(QPalette.Disabled, QPalette.Text, QColor(colors["text_faint"]))
    qpalette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(colors["text_faint"]))
    app.setPalette(qpalette)
    families = set(QFontDatabase.families())
    family = next(
        (name for name in ("Yu Gothic UI", "Meiryo UI", "Segoe UI") if name in families),
        "Segoe UI",
    )
    font = QFont(family)
    font.setPointSize(font_size)
    app.setFont(font)
    app.setStyleSheet(build_stylesheet(theme, font_size))


def current_palette() -> dict[str, str]:
    app = QApplication.instance()
    theme = getattr(app, "_kestrel_theme", "dark") if app else "dark"
    return palette(theme)


def set_window_theme(app: QApplication, theme: str, font_size: int) -> None:
    app._kestrel_theme = theme  # type: ignore[attr-defined]
    apply_theme(app, theme, font_size)
