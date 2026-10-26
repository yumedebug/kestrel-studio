# Kestrel Studio

Kestrel Studioは、GGUFモデルの管理とホスト型AIプロバイダーへの接続を行うWindowsデスクトップアプリ。そのオリジナルのインターフェースは、ライトテーマを基本とした落ち着いた配色、明確な文字の階層、ゆったりした余白を特徴とし、他のデスクトップ製品と提携しているわけでも、そのコピーでもない。開始にあたってローカルモデルは不要。クラウド推論は設定したプロバイダーにプロンプトを送信するため、インターネット接続、プロバイダーのアカウント、APIキー、そして必要に応じた課金が求められる。アプリにAPIクレジットは含まれていない。

## 現在の実装範囲

デスクトップシェル、ダーク/ライトUI、設定、モデルライブラリ、GGUFメタデータリーダー、Hugging Faceの検索/ダウンロードフロー、llama.cppプロセスバックエンド、SQLiteチャット履歴、クラウドプロバイダーバックエンド、OpenAI互換のローカルHTTPサーバーが実装済み。クラウドバックエンドは、OpenAI、Anthropic Messages、Geminiに加え、OpenRouter、Groq、Mistral、DeepSeek、xAI、Together AI、Ollama、LM Studio、カスタムサーバーを含むOpenAI互換エンドポイントに対応する。モデルリストは初期候補であり、プロバイダーのモデルIDや互換ベースURLは設定で編集できる。

チャットは現在のアプリに含まれている。このワークステーションにはプロバイダーのキーやモデルの認証情報が設定されていないため、実際のプロバイダー応答はここでは検証されていない。APIキーは同梱されないため、自分のキーを Settings → Cloud Providers で入力すること。Windowsでは、シークレットはWindows資格情報マネージャー（利用できない環境ではDPAPIで暗号化したフォールバック）を使用する。

## 動作要件

- Windows 10または11、64ビット
- 開発にはPython 3.12（プロジェクトは現在Python 3.12.10を使用）
- クラウド推論、Hugging Faceのダウンロード、依存関係のインストール、任意のllama.cppランタイムのダウンロードにはインターネットアクセスが必要

ブラウザ、Electron、WebViewのUIは使用していない。ローカル推論にホスト型AIサービスは不要だが、互換性のあるllama.cppランタイムと適切なGGUFモデルが必要になる。

## セットアップと起動

PowerShellで実行する:

```powershell
cd C:\path\to\localapp
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe main.py
```

初回起動時に、Windowsのユーザーアプリケーションデータディレクトリ配下にアプリの設定とデータが作成される。開発時は、環境変数 `KESTREL_CONFIG_DIR` と `KESTREL_DATA_DIR` を一時フォルダに設定するとデータを分離できる。バージョンを表示するには `main.py --version` を実行する。

## クラウドプロバイダーの設定

1. アプリを起動し、**Settings → Cloud Providers** を開く。
2. プロバイダーを選び、APIキーを入力して保存する。利用可能な場合は **Test Connection** を使ってプロバイダーのモデルリストを取得する。
3. そのプロバイダーが表示する正確なモデルIDを選択または入力する。
4. チャットの **Load Cloud Model**（またはモデルセレクター）を使って有効化する。

APIキーは設定JSONではなく、WindowsユーザーごとにOSの資格情報ストアに保存される。プロバイダー側の制御では、選択したAPIがサポートする生成オプションのみを送信する。カスタムのOpenAI互換エンドポイントでは、ベースURL（通常は `/v1` で終わる）、モデルID、任意のキーを入力する。各プロバイダーの規約、モデルの提供状況、料金については各自で確認すること。

対応するプロバイダーAPIスタイル:

- OpenAI Chat Completions: OpenAI、OpenRouter、Groq、Mistral、DeepSeek、xAI、Together AI、Ollama、LM Studio、カスタム互換エンドポイント。
- Anthropic Messages API: Anthropic。
- Gemini generate-content ストリーミングAPI: Google Gemini。

モデルIDと提供状況は時間とともに変化する。アプリ内の候補は編集可能な便宜上の値として扱い、プロバイダーの最新のドキュメント/ダッシュボードでIDとエンドポイントを確認すること。このアプリは、設定された各プロバイダーへ直接リクエストを行う。サードパーティの互換プロバイダーは、それぞれ独自のサービスを使用する。

## ローカルのGGUFモデル

Discoverを使ってHugging FaceのGGUFリポジトリを検索し、ファイルを閲覧して選択したファイルをダウンロードするか、**Add Model** で既存のGGUFを登録する。ローカルで実行するには、後述の方法で公式のllama.cpp Windows x64ランタイムをインストールし、モデルを選択してロードする。ローカル推論はllama.cpp経由で動作し、ローカルのプロンプトにプロバイダーのAPIキーは使用されない。

公式ランタイムはllama.cppのGitHubリリースからダウンロードする:

```powershell
.venv\Scripts\python.exe scripts\fetch_llama_cpp.py --backend cpu
# 任意のNVIDIAビルド（互換性のあるCUDAインストール/ランタイムが必要）:
.venv\Scripts\python.exe scripts\fetch_llama_cpp.py --backend cuda --cuda-version 12.4
```

取得スクリプトは公式の `ggml-org/llama.cpp` のWindows x64リリースアセットを選択し、公開されたSHA-256ダイジェストが提供されている場合はそれを検証する。GPUランタイムの互換性は、インストールされているドライバーと選択したリリースに依存する。CPUがフォールバックになる。

## ローカルサーバーAPI

組み込みのFastAPI/Uvicornサーバーは、デフォルトで `127.0.0.1:1234` を使用する。モデルをロードした後、**Local Server** から起動する。デフォルトではループバックでのみ待ち受け、外部アクセスの有効化は明示的な設定になっている。GUIチャットとHTTP APIは、同じ推論マネージャーとロード済みバックエンドを共有する。

実装済みのルート:

- `GET /v1/models`
- `POST /v1/chat/completions`（非ストリーミングとSSEストリーミング）
- `POST /v1/completions`（ドキュメントに記載された対応サブセット）
- `GET /health`

例:

```powershell
$body = @{ model = "<loaded-model-id>"; messages = @(@{ role = "user"; content = "Hello" }) } | ConvertTo-Json -Depth 6
Invoke-RestMethod http://127.0.0.1:1234/v1/chat/completions -Method Post -ContentType 'application/json' -Body $body
```

サーバーは、対応していないリクエスト機能を実装しているふりはせず、その旨を報告する。リモートAPIの利用者は、プロバイダーのエンドポイントを直接使用すること。このローカルサーバーは、現在ロードされているモデルを公開するためのもの。

## テストとチェック

自動テストスイートを実行する:

```powershell
.venv\Scripts\python.exe -m pytest -q
```

ヘッドレスでのGUI起動/ページ構築スモークチェックを実行する:

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
.venv\Scripts\python.exe main.py --smoke-test
```

スモークチェックはすべてのメインページを構築し、クリーンに終了する。クラウドプロバイダーへの接続や、実際のGGUFモデルのロードは行わない。クラウド推論、プロバイダー認証、Hugging Faceのダウンロード、llama.cppのCPU/GPU推論、実際にロードしたモデルへのAPIリクエストには、認証情報・ネットワーク・ランタイム・モデルが必要であり、このスモークチェックでは検証されない。

## Windowsビルドとインストーラー

Windows用のインストーラーをビルドするには、先に[Inno Setup](https://jrsoftware.org/isinfo.php)をインストールし、プロジェクトルートで以下を実行する:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\scripts\build-installer.ps1
```

出力は `dist\installer\KestrelStudio-Setup-x64.exe`。セットアップは必要なPySide6 DLLとQtプラグインを含む完全なアプリ一式を、現在のユーザーのLocalAppData Programsフォルダーに配置し、スタートメニューのショートカットを作成する。別のPCではSetup.exeを実行すること。 `KestrelStudio.exe` だけをコピーしても、隣接する `_internal` ランタイムファイルが必要なため起動できない。

GitHub Actionsは `qwindows.dll` と `qoffscreen.dll` の同梱を確認し、配布ビルドのGUIスモークテストを実行してからセットアップを作成する。任意のllama.cppランタイムを同梱するには、ビルド前に `runtime` フォルダーへ配置するか、ビルド後に取得スクリプトで `dist\KestrelStudio\runtime` へ取得する。GPUビルドはNVIDIAドライバーやCUDAランタイムに依存する場合があり、CPUバックエンドが互換性重視の選択肢になる。

## データとプライバシー

設定、モデルライブラリのメタデータ、ログ、SQLiteのチャット履歴は、ローカルのWindowsユーザーデータディレクトリに保持される。プロンプトは、クラウド推論が選択された場合にのみ、選択中のクラウドプロバイダーへ送信される。プロバイダーにおけるプライバシー、保持、データ処理は、そのプロバイダーの取り決めに準ずる。Hugging Faceの認証情報とプロバイダーのキーは、対応している場合はWindowsの安全な資格情報ストレージに保存される。APIキーをIssue報告やログに含めないこと。

## 既知の制限

- プロバイダーのモデル候補は古くなることがある。リストを更新するか、現在のモデルIDを手動で入力すること。
- プロバイダー固有のAPI同士では、生成オプションや応答機能が完全に同じわけではない。対応していない機能は省略されるか、エラーとして報告される。
- クラウド呼び出しには、有効なキー、ネットワークアクセス、プロバイダー側のアクセス/課金が必要。
- 実モデルのロードとGPUサポートは、モデルサイズ、利用可能なメモリ、llama.cppのビルド、ドライバーに依存する。このPCでの検証では実モデルは使用していない。
- Hugging Faceダウンローダーとクラウド連携には、実際のネットワークアクセスと、必要に応じて認証情報が必要。
- PyInstallerと任意のllama.cppランタイムには、配布可能なWindowsパッケージのためにWindowsビルド環境が必要。

## サードパーティソフトウェア

ランタイム依存関係は `requirements.txt` に宣言されている（PySide6、FastAPI、Uvicorn、httpx、huggingface_hub）。開発/ビルド依存関係は `requirements-dev.txt` にある（pytest、pytest-qt、PyInstaller）。WindowsセットアップはInno Setupを使って生成する。再配布や使用の前に、各依存関係のライセンスと、llama.cpp/GGUFモデルのライセンスを確認すること。モデルのライセンスとプロバイダーの規約は、このアプリとは別物。

## オープンソースとしての公開

このリポジトリはMITライセンスの下でライセンスされている。ソースを公開するには、GitHub上に空の公開リポジトリを作成し、プロジェクトルートから以下を実行する（オーナー/リポジトリのURLは置き換えること）:

```powershell
git init -b main
git add .
git commit -m "Initial OSS release"
git remote add origin https://github.com/<OWNER>/<REPOSITORY>.git
git push -u origin main
```

`.github/workflows/windows.yml` のGitHub Actionsワークフローは、テスト、PyInstallerで固めたGUIの起動検証、Windows x64セットアップの作成を行う。`v0.1.3` のようなバージョンタグをpushすると、`KestrelStudio-Setup-x64.exe` をGitHub Releaseに添付する。

バイナリ配布物はCIで生成され、意図的にGitからは除外されている。再配布の前に、モデル、プロバイダー、依存関係、llama.cppのライセンスを個別に確認すること。
