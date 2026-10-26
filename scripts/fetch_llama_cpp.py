"""Install an official llama.cpp Windows runtime for Kestrel Studio.

Downloads the current official release from github.com/ggml-org/llama.cpp and
places ``llama-server.exe`` plus its DLLs in ``runtime/<backend>/``.  The GUI
auto-detects that folder (Settings → Inference → llama-server Path can override).

Examples:
    python scripts/fetch_llama_cpp.py                 # CPU build
    python scripts/fetch_llama_cpp.py --backend vulkan
    python scripts/fetch_llama_cpp.py --backend cuda --cuda-version 12.4

Only Windows x64 assets are selected. No third-party Python packages are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

API = "https://api.github.com/repos/ggml-org/llama.cpp/releases"
USER_AGENT = "KestrelStudio-llama-installer/0.1"
ROOT = Path(__file__).resolve().parents[1]


def _get_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _download(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        total = int(response.headers.get("Content-Length") or 0)
        received = 0
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as fh:
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                fh.write(chunk)
                received += len(chunk)
                if total:
                    percent = received * 100 / total
                    print(
                        f"\r  {dest.name}: {percent:5.1f}% "
                        f"({received / 1048576:.1f}/{total / 1048576:.1f} MiB)",
                        end="",
                        flush=True,
                    )
        print()


def _asset_matches(asset: dict, backend: str, cuda_version: str) -> bool:
    name = asset.get("name", "")
    if not name.startswith("llama-") or not name.endswith(".zip"):
        return False
    if "win" not in name or "x64" not in name:
        return False
    if backend == "cpu":
        return "win-cpu-x64" in name
    if backend == "vulkan":
        return "win-vulkan-x64" in name
    if backend == "cuda":
        return f"win-cuda-{cuda_version}-x64" in name
    raise ValueError(backend)


def _resolve_release(tag: str | None) -> dict:
    if tag:
        return _get_json(f"{API}/tags/{tag}")
    latest = _get_json(f"{API}/latest")
    if any("win" in a.get("name", "") for a in latest.get("assets", [])):
        return latest
    # Recent versioned releases publish only the nightly tag pointer.
    for asset in latest.get("assets", []):
        if asset.get("name") == "nightly-tag.txt":
            pointer = _download_text(asset["browser_download_url"]).strip()
            if pointer:
                print(f"Latest stable release points at nightly build {pointer}.")
                return _get_json(f"{API}/tags/{pointer}")
    return latest


def _download_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def _verify(path: Path, expected: str) -> None:
    if not expected:
        return
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual.lower() != expected.lower():
        path.unlink(missing_ok=True)
        raise SystemExit(f"Checksum mismatch for {path.name}: expected {expected}, got {actual}")


def _safe_extract(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    root = target.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            destination = (target / member.filename).resolve()
            if not str(destination).startswith(str(root)):
                raise SystemExit(f"Refusing unsafe archive path: {member.filename}")
        zf.extractall(target)


def _install_llama(extract_dir: Path, dest: Path) -> None:
    found = sorted(extract_dir.rglob("llama-server.exe"), key=lambda p: len(p.parts))
    if not found:
        raise SystemExit("The archive did not contain llama-server.exe.")
    source_dir = found[0].parent
    print(f"Installing from {source_dir.relative_to(extract_dir)}")
    dest.mkdir(parents=True, exist_ok=True)
    for item in source_dir.iterdir():
        target = dest / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "vulkan", "cuda"), default="cpu")
    parser.add_argument("--cuda-version", default="12.4", choices=("12.4", "13.4"))
    parser.add_argument("--tag", default="", help="specific llama.cpp release tag (e.g. b11669)")
    parser.add_argument(
        "--dest", default=str(ROOT / "runtime"), help="target folder (default: ./runtime)"
    )
    parser.add_argument("--force", action="store_true", help="replace an existing installation")
    args = parser.parse_args(argv)

    dest = Path(args.dest).expanduser().resolve() / args.backend
    if (dest / "llama-server.exe").exists() and not args.force:
        print(f"llama-server.exe already exists in {dest}; use --force to replace it.")
        return 0

    release = _resolve_release(args.tag)
    tag = release.get("tag_name", "?")
    assets = release.get("assets", [])
    asset = next(
        (a for a in assets if _asset_matches(a, args.backend, args.cuda_version)), None
    )
    if asset is None:
        names = "\n  ".join(a.get("name", "") for a in assets if "win" in a.get("name", ""))
        raise SystemExit(
            f"No Windows x64 {args.backend} archive in release {tag}.\n"
            f"Available Windows assets:\n  {names}"
        )

    extra: list[dict] = []
    if args.backend == "cuda":
        cudart_name = f"cudart-llama-bin-win-cuda-{args.cuda_version}-x64.zip"
        extra = [a for a in assets if a.get("name") == cudart_name]

    print(f"llama.cpp {tag} → {args.backend} ({asset['name']})")
    with tempfile.TemporaryDirectory(prefix="kestrel-llama-") as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / asset["name"]
        _download(asset["browser_download_url"], archive)
        expected = str((asset.get("digest") or "")).replace("sha256:", "")
        _verify(archive, expected)

        extracted = tmp_path / "extracted"
        _safe_extract(archive, extracted)
        _install_llama(extracted, dest)

        for extra_asset in extra:
            extra_archive = tmp_path / extra_asset["name"]
            print(f"Downloading runtime DLLs: {extra_asset['name']}")
            _download(extra_asset["browser_download_url"], extra_archive)
            expected = str((extra_asset.get("digest") or "")).replace("sha256:", "")
            _verify(extra_archive, expected)
            extra_dir = tmp_path / "cudart"
            _safe_extract(extra_archive, extra_dir)
            for item in extra_dir.rglob("*"):
                if item.is_file():
                    shutil.copy2(item, dest / item.name)

    exe = dest / "llama-server.exe"
    if not exe.exists():
        raise SystemExit("Installation finished but llama-server.exe is missing.")
    print(f"Done: {exe}")
    print("Set this path in Settings → Inference if auto-detection does not find it.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except urllib.error.URLError as exc:
        print(f"Network error: {exc}", file=sys.stderr)
        raise SystemExit(1)
