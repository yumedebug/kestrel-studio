param(
   [switch]$SkipTests,
   [switch]$InstallDevDeps
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root

$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    # CI installs dependencies into the interpreter selected by setup-python.
    # Use that interpreter when no project-local venv is present.
    $SystemPython = Get-Command python -ErrorAction SilentlyContinue
    if ($SystemPython) {
        $Python = $SystemPython.Source
    } else {
        throw "Python 3.12 was not found. Create .venv with: py -3.12 -m venv .venv"
    }
}

if ($InstallDevDeps) {
   & $Python -m pip install -r requirements-dev.txt
   if (-not $?) { throw "Dependency installation failed." }
}

& $Python -c "import PySide6, fastapi, uvicorn, httpx, huggingface_hub, PyInstaller"
if (-not $?) {
   throw "Build dependencies are missing. Run: .venv\Scripts\python.exe -m pip install -r requirements-dev.txt"
}

if (-not (Test-Path -LiteralPath "assets\icon.ico")) {
   Write-Host "Generating application icon..."
   & $Python scripts\make_icon.py
   if (-not $?) { throw "Icon generation failed." }
}

if (-not $SkipTests) {
   Write-Host "Running tests..."
   & $Python -m pytest -q
   if (-not $?) { throw "Tests failed; build aborted." }
}

Write-Host "Building with PyInstaller..."
& $Python -m PyInstaller --clean --noconfirm kestrel.spec
if (-not $?) { throw "PyInstaller failed." }

$Dist = Join-Path $Root "dist\KestrelStudio"
$QtPlatformPlugin = Get-ChildItem -LiteralPath $Dist -Recurse -Filter "qwindows.dll" -File | Select-Object -First 1
if (-not $QtPlatformPlugin) {
   throw "PySide6 Windows platform plugin (qwindows.dll) is missing from the packaged application."
}
if (Test-Path -LiteralPath "runtime") {
   Write-Host "Copying llama.cpp runtime next to the executable..."
   Copy-Item -Recurse -Force "runtime" (Join-Path $Dist "runtime")
}
Copy-Item -Force "README.md" (Join-Path $Dist "README.md")
Copy-Item -Force "LICENSE" (Join-Path $Dist "LICENSE")

Write-Host ""
Write-Host "Build complete: $Dist\KestrelStudio.exe"
Write-Host "Optional runtime: .venv\Scripts\python.exe scripts\fetch_llama_cpp.py --backend cpu"

