# Contributing

Thanks for considering a contribution to Kestrel Studio.

## Development setup

- Windows 10/11 x64
- Python 3.12

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest -q
```

For a headless UI startup check, set `QT_QPA_PLATFORM=offscreen` and run `python main.py --smoke-test`.

## Pull requests

- Keep UI work in native PySide6 widgets; do not add Electron, WebView, or CDN dependencies.
- Keep provider credentials out of source, settings files, logs, fixtures, and screenshots.
- Preserve the shared inference backend used by GUI and local API requests.
- Describe behavior changes and note which checks you ran.
- Add or update focused tests for changed behavior.

By submitting a contribution, you agree that it is provided under the repository's MIT License. Contributions remain credited to their authors.
