# Kestrel Studio

Kestrel Studio is a Windows desktop app for managing GGUF models and connecting to hosted AI providers. Its original interface uses a light-first, restrained palette, clear type hierarchy, and generous spacing; it is not affiliated with or a copy of another desktop product. It does not require a local model to start. Cloud inference sends prompts to the provider you configure, so an internet connection, provider account, API key, and any applicable billing are required. The app does not include API credits.

## Current scope

The desktop shell, dark/light UI, settings, model library, GGUF metadata reader, Hugging Face discovery/download flow, llama.cpp process backend, SQLite chat history, cloud-provider backend, and OpenAI-compatible local HTTP server are implemented. The cloud backend supports OpenAI, Anthropic Messages, Gemini, and OpenAI-compatible endpoints including OpenRouter, Groq, Mistral, DeepSeek, xAI, Together AI, Ollama, LM Studio, and custom servers. Model lists are starting suggestions; provider model IDs and compatible base URLs can be edited in Settings.

Chat is included in the current app. This workstation has no configured provider key or model credentials, so a real provider response was not verified here. API keys are never bundled; enter your own in Settings → Cloud Providers. On Windows, secrets use Windows Credential Manager (with an encrypted DPAPI fallback where Credential Manager is unavailable).

## Requirements

- Windows 10 or 11, 64-bit
- Python 3.12 for development (the project currently uses Python 3.12.10)
- Internet access for cloud inference, Hugging Face downloads, dependency installation, and optional llama.cpp runtime download

No browser, Electron, or WebView UI is used. Local inference does not require a hosted AI service, but it requires a compatible llama.cpp runtime and a suitable GGUF model.

## Setup and launch

From PowerShell:

```powershell
cd C:\path\to\localapp
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe main.py
```

The first launch creates app settings and data under the Windows user application-data directory. For development, set `KESTREL_CONFIG_DIR` and `KESTREL_DATA_DIR` to temporary folders to isolate data. Run `main.py --version` to print the version.

## Configure a cloud provider

1. Start the app and open **Settings → Cloud Providers**.
2. Choose a provider, enter its API key, and save it. Use **Test Connection** where available to fetch the provider's model list.
3. Select or enter the exact model ID shown by that provider.
4. Use **Load Cloud Model** in Chat (or the model selector) to activate it.

API keys are stored per Windows user in the OS credential store, not in settings JSON. Provider controls only send generation options that the selected API supports. For a custom OpenAI-compatible endpoint, enter its base URL (normally ending in `/v1`), model ID, and optional key. Refer to each provider's own terms, model availability, and pricing.

Supported provider API styles:

- OpenAI Chat Completions: OpenAI, OpenRouter, Groq, Mistral, DeepSeek, xAI, Together AI, Ollama, LM Studio, and custom compatible endpoints.
- Anthropic Messages API: Anthropic.
- Gemini generate-content streaming API: Google Gemini.

Model IDs and availability change over time. Treat the in-app suggestions as editable convenience values and use the provider's current documentation/dashboard to verify the ID and endpoint. This app makes direct requests to each configured provider; third-party compatible providers use their own service.

## Local GGUF models

Use Discover to search Hugging Face for GGUF repositories, inspect files, and download a selected file, or use **Add Model** to register an existing GGUF. To run locally, install an official llama.cpp Windows x64 runtime as described below, then select the model and load it. Local inference runs through llama.cpp; no provider API key is used for local prompts.

Download the official runtime from the llama.cpp GitHub releases:

```powershell
.venv\Scripts\python.exe scripts\fetch_llama_cpp.py --backend cpu
# Optional NVIDIA build (requires a compatible CUDA installation/runtime):
.venv\Scripts\python.exe scripts\fetch_llama_cpp.py --backend cuda --cuda-version 12.4
```

The fetch script selects official `ggml-org/llama.cpp` Windows x64 release assets and verifies a published SHA-256 digest when one is provided. GPU runtime compatibility depends on the installed driver and the chosen release. CPU is the fallback.

## Local Server API

The embedded FastAPI/Uvicorn server defaults to `127.0.0.1:1234`. Start it from **Local Server** after loading a model. By default it only listens on loopback; enabling external access is an explicit setting. The GUI chat and HTTP API share the same inference manager and loaded backend.

Implemented routes:

- `GET /v1/models`
- `POST /v1/chat/completions` (non-streaming and SSE streaming)
- `POST /v1/completions` (documented supported subset)
- `GET /health`

Example:

```powershell
$body = @{ model = "<loaded-model-id>"; messages = @(@{ role = "user"; content = "Hello" }) } | ConvertTo-Json -Depth 6
Invoke-RestMethod http://127.0.0.1:1234/v1/chat/completions -Method Post -ContentType 'application/json' -Body $body
```

The server reports unsupported request features instead of claiming to implement them. For remote API users, use the provider's endpoint directly; this local server is intended to expose the currently loaded model.

## Tests and checks

Run the automated suite:

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Run a headless GUI startup/page-construction smoke check:

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
.venv\Scripts\python.exe main.py --smoke-test
```

The smoke check creates every main page and shuts down cleanly; it does not contact cloud providers or load a real GGUF model. Cloud inference, provider authentication, Hugging Face downloads, llama.cpp CPU/GPU inference, and API requests against a real loaded model require credentials/network/runtime/model and were not verified by that smoke check.

## Windows build

Build a one-folder PyInstaller distribution:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\scripts\build.ps1
```

Output: `dist\KestrelStudio\KestrelStudio.exe`. Include the optional llama.cpp runtime by placing it in the build's `runtime` directory before packaging, or run the fetch script into `dist\KestrelStudio\runtime` after the build. A GPU build may depend on system NVIDIA drivers and CUDA runtime components; the CPU backend is the compatibility option.

## Data and privacy

Settings, model library metadata, logs, and SQLite chat history stay in the local Windows user data directories. Prompts are sent to the selected cloud provider only when cloud inference is selected; provider privacy, retention, and data processing are governed by that provider. Hugging Face credentials and provider keys are stored in Windows secure credential storage where supported. Do not put API keys in issue reports or logs.

## Known limits

- Provider model suggestions can become stale; refresh the list or enter the current model ID manually.
- Provider-specific APIs do not share exactly the same generation options or response features. Unsupported capabilities are omitted or reported as errors.
- Cloud calls require a valid key, network access, and provider-side access/billing.
- Real-model loading and GPU support depend on model size, available memory, llama.cpp build, and drivers. This PC validation used no real model.
- The Hugging Face downloader and cloud integrations require live network access and credentials where applicable.
- PyInstaller and the optional llama.cpp runtime need a Windows build environment for a distributable Windows package.

## Third-party software

Runtime dependencies are declared in `requirements.txt` (PySide6, FastAPI, Uvicorn, httpx, and huggingface_hub). Development/build dependencies are in `requirements-dev.txt` (pytest, pytest-qt, and PyInstaller). Review each dependency's license and the llama.cpp/GGUF model license before redistribution or use. Model licenses and provider terms are separate from this app.

## Open-source release

This repository is licensed under the MIT License. To publish the source, create an empty public repository on GitHub, then run the following from the project root (replace the owner/repository URL):

```powershell
git init -b main
git add .
git commit -m "Initial OSS release"
git remote add origin https://github.com/<OWNER>/<REPOSITORY>.git
git push -u origin main
```

The GitHub Actions workflow at `.github/workflows/windows.yml` runs the test suite and builds a Windows x64 ZIP for pushes and pull requests. To publish a GitHub Release after pushing the repository, push a version tag such as `v0.1.0`; the workflow attaches `KestrelStudio-Windows-x64.zip` to the release.

The binary distribution is generated by CI and is intentionally excluded from Git. Review model, provider, dependency, and llama.cpp licenses independently before redistributing them.
