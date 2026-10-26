"""Kestrel Studio - a local-first AI desktop application.

The package is intentionally split into Qt-free core/API layers and a PySide6
UI layer so that the inference backends can be reused from the HTTP server and
tested without a display.
"""

__version__ = "0.1.0"
