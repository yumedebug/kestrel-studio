"""Generate assets/icon.ico from assets/icon.svg using the Qt SVG renderer.

Run with the project virtualenv:
    .venv\\Scripts\\python.exe scripts\\make_icon.py
"""

from __future__ import annotations

import struct
from pathlib import Path

SIZES = (16, 24, 32, 48, 64, 128, 256)


def render_png(svg_path: Path, size: int) -> bytes:
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QSize
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer

    renderer = QSvgRenderer(str(svg_path))
    if not renderer.isValid():
        raise SystemExit(f"invalid SVG: {svg_path}")
    image = QImage(QSize(size, size), QImage.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
    renderer.render(painter)
    painter.end()

    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.WriteOnly)
    if not image.save(buffer, "PNG"):
        raise SystemExit("could not encode PNG")
    buffer.close()
    return bytes(data)


def build_ico(svg_path: Path, out_path: Path) -> None:
    pngs = [(size, render_png(svg_path, size)) for size in SIZES]
    header_size = 6 + 16 * len(pngs)
    offset = header_size
    entries = bytearray()
    payload = bytearray()
    for size, png in pngs:
        width = 0 if size >= 256 else size
        height = 0 if size >= 256 else size
        entries += struct.pack("<BBBBHHII", width, height, 0, 0, 1, 32, len(png), offset)
        payload += png
        offset += len(png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(struct.pack("<HHH", 0, 1, len(pngs)) + bytes(entries) + bytes(payload))
    print(f"wrote {out_path} ({out_path.stat().st_size:,} bytes)")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    build_ico(root / "assets" / "icon.svg", root / "assets" / "icon.ico")


if __name__ == "__main__":
    main()
