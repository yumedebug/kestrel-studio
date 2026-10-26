"""Small custom-drawn icon set so the app needs no external assets."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPainterPath, QPen, QPixmap

from app.constants import assets_dir


def _pixmap(drawer, color: str, size: int = 40) -> QPixmap:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.scale(size / 20.0, size / 20.0)
    pen = QPen(QColor(color), 1.6)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    drawer(painter)
    painter.end()
    pixmap.setDevicePixelRatio(2.0)
    return pixmap


def _chat(p: QPainter) -> None:
    path = QPainterPath()
    path.addRoundedRect(QRectF(2.5, 3.5, 15, 11.5), 3.0, 3.0)
    p.drawPath(path)
    tail = QPainterPath()
    tail.moveTo(6.0, 14.6)
    tail.lineTo(6.0, 18.0)
    tail.lineTo(10.0, 14.8)
    p.drawPath(tail)
    p.setBrush(p.pen().color())
    for x in (6.5, 10.0, 13.5):
        p.drawEllipse(QPointF(x, 9.2), 0.9, 0.9)
    p.setBrush(Qt.NoBrush)


def _compass(p: QPainter) -> None:
    p.drawEllipse(QRectF(2.5, 2.5, 15, 15))
    path = QPainterPath()
    path.moveTo(12.4, 7.6)
    path.lineTo(10.7, 11.4)
    path.lineTo(7.6, 12.4)
    path.lineTo(9.3, 8.6)
    path.closeSubpath()
    p.setBrush(p.pen().color())
    p.drawPath(path)
    p.setBrush(Qt.NoBrush)


def _cube(p: QPainter) -> None:
    path = QPainterPath()
    path.moveTo(10.0, 2.5)
    path.lineTo(17.0, 6.2)
    path.lineTo(17.0, 13.6)
    path.lineTo(10.0, 17.5)
    path.lineTo(3.0, 13.6)
    path.lineTo(3.0, 6.2)
    path.closeSubpath()
    p.drawPath(path)
    p.drawLine(QPointF(3.0, 6.2), QPointF(10.0, 10.0))
    p.drawLine(QPointF(17.0, 6.2), QPointF(10.0, 10.0))
    p.drawLine(QPointF(10.0, 10.0), QPointF(10.0, 17.5))


def _server(p: QPainter) -> None:
    for y in (3.0, 11.0):
        p.drawRoundedRect(QRectF(3.0, y, 14.0, 6.0), 2.0, 2.0)
        p.setBrush(p.pen().color())
        p.drawEllipse(QPointF(6.0, y + 3.0), 0.9, 0.9)
        p.setBrush(Qt.NoBrush)
        p.drawLine(QPointF(12.0, y + 3.0), QPointF(14.5, y + 3.0))


def _settings(p: QPainter) -> None:
    for y, knob in ((5.0, 12.5), (10.0, 7.5), (15.0, 13.5)):
        p.drawLine(QPointF(3.0, y), QPointF(17.0, y))
        p.setBrush(p.pen().color())
        p.drawEllipse(QPointF(knob, y), 1.6, 1.6)
        p.setBrush(Qt.NoBrush)


def _plus(p: QPainter) -> None:
    p.drawLine(QPointF(10.0, 4.0), QPointF(10.0, 16.0))
    p.drawLine(QPointF(4.0, 10.0), QPointF(16.0, 10.0))


def _trash(p: QPainter) -> None:
    p.drawLine(QPointF(4.0, 6.0), QPointF(16.0, 6.0))
    p.drawRoundedRect(QRectF(5.5, 6.0, 9.0, 10.5), 1.5, 1.5)
    p.drawLine(QPointF(8.0, 3.5), QPointF(12.0, 3.5))
    p.drawLine(QPointF(8.5, 9.0), QPointF(8.5, 13.5))
    p.drawLine(QPointF(11.5, 9.0), QPointF(11.5, 13.5))


def _folder(p: QPainter) -> None:
    path = QPainterPath()
    path.moveTo(2.5, 5.0)
    path.lineTo(8.0, 5.0)
    path.lineTo(9.7, 7.0)
    path.lineTo(17.5, 7.0)
    path.lineTo(17.5, 15.5)
    path.lineTo(2.5, 15.5)
    path.closeSubpath()
    p.drawPath(path)


def _copy(p: QPainter) -> None:
    p.drawRoundedRect(QRectF(6.5, 6.5, 10.0, 10.5), 1.5, 1.5)
    path = QPainterPath()
    path.moveTo(4.0, 13.5)
    path.lineTo(4.0, 3.5)
    path.lineTo(13.0, 3.5)
    p.drawPath(path)


def _stop(p: QPainter) -> None:
    p.setBrush(p.pen().color())
    p.drawRoundedRect(QRectF(5.0, 5.0, 10.0, 10.0), 2.0, 2.0)


def _send(p: QPainter) -> None:
    path = QPainterPath()
    path.moveTo(3.0, 10.0)
    path.lineTo(17.0, 3.5)
    path.lineTo(12.5, 17.0)
    path.lineTo(10.0, 11.5)
    path.closeSubpath()
    p.drawPath(path)
    p.drawLine(QPointF(10.0, 11.5), QPointF(17.0, 3.5))


def _refresh(p: QPainter) -> None:
    p.drawArc(QRectF(3.5, 3.5, 13.0, 13.0), 30 * 16, 280 * 16)
    path = QPainterPath()
    path.moveTo(15.5, 2.5)
    path.lineTo(16.5, 7.0)
    path.lineTo(12.0, 6.0)
    p.drawPath(path)


def _download(p: QPainter) -> None:
    p.drawLine(QPointF(10.0, 3.0), QPointF(10.0, 13.0))
    path = QPainterPath()
    path.moveTo(6.0, 9.5)
    path.lineTo(10.0, 13.5)
    path.lineTo(14.0, 9.5)
    p.drawPath(path)
    p.drawLine(QPointF(4.0, 16.5), QPointF(16.0, 16.5))


def _external(p: QPainter) -> None:
    p.drawRoundedRect(QRectF(3.0, 6.0, 11.0, 11.0), 2.0, 2.0)
    p.drawLine(QPointF(9.5, 3.0), QPointF(17.0, 3.0))
    p.drawLine(QPointF(17.0, 3.0), QPointF(17.0, 10.5))
    p.drawLine(QPointF(17.0, 3.0), QPointF(9.0, 11.0))


def _stop_circle(p: QPainter) -> None:
    p.drawEllipse(QRectF(2.5, 2.5, 15.0, 15.0))
    p.setBrush(p.pen().color())
    p.drawRoundedRect(QRectF(7.0, 7.0, 6.0, 6.0), 1.0, 1.0)


_DRAWERS = {
    "chat": _chat,
    "compass": _compass,
    "cube": _cube,
    "server": _server,
    "settings": _settings,
    "plus": _plus,
    "trash": _trash,
    "folder": _folder,
    "copy": _copy,
    "stop": _stop,
    "send": _send,
    "refresh": _refresh,
    "download": _download,
    "external": _external,
}


def make_icon(name: str, color: str, checked_color: str | None = None) -> QIcon:
    drawer = _DRAWERS.get(name)
    if drawer is None:
        return QIcon()
    icon = QIcon()
    icon.addPixmap(_pixmap(drawer, color), QIcon.Normal, QIcon.Off)
    icon.addPixmap(_pixmap(drawer, checked_color or color), QIcon.Normal, QIcon.On)
    icon.addPixmap(_pixmap(drawer, checked_color or color), QIcon.Active, QIcon.Off)
    return icon


def app_icon() -> QIcon:
    svg = assets_dir() / "icon.svg"
    if svg.exists():
        icon = QIcon(str(svg))
        if not icon.isNull():
            return icon
    # Generated fallback.
    pixmap = QPixmap(128, 128)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setBrush(QColor("#7c8cff"))
    painter.setPen(Qt.NoPen)
    painter.drawRoundedRect(QRectF(4, 4, 120, 120), 28, 28)
    painter.setPen(QColor("#ffffff"))
    font = QFont("Segoe UI", 64, QFont.Bold)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignCenter, "K")
    painter.end()
    return QIcon(pixmap)
