param(
   [switch]$SkipBuild,
   [string]$Version = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root

# Keep the installer metadata in step with the application version instead of
# maintaining a second hard-coded copy inside installer.iss.
if (-not $Version) {
   $Constants = Join-Path $Root "app\constants.py"
   if (Test-Path -LiteralPath $Constants) {
      $Match = Select-String -LiteralPath $Constants -Pattern '^APP_VERSION\s*=\s*"([^"]+)"' | Select-Object -First 1
      if ($Match) { $Version = $Match.Matches[0].Groups[1].Value }
   }
}
if (-not $Version) {
   throw "Could not determine the application version (APP_VERSION in app/constants.py). Pass -Version explicitly."
}

if (-not $SkipBuild) {
   & (Join-Path $PSScriptRoot "build.ps1")
   if ($LASTEXITCODE -ne 0) { throw "Application build failed." }
}

$Dist = Join-Path $Root "dist\KestrelStudio"
$AppExe = Join-Path $Dist "KestrelStudio.exe"
if (-not (Test-Path -LiteralPath $AppExe -PathType Leaf)) {
   throw "Built application not found: $AppExe"
}

$PlatformPlugins = Get-ChildItem -LiteralPath $Dist -Recurse -Filter "qwindows.dll" -File
if (-not $PlatformPlugins) {
   throw "PySide6 qwindows.dll is missing; refusing to create a broken Windows installer."
}

$CompilerCommand = Get-Command "ISCC.exe" -ErrorAction SilentlyContinue
$CompilerPath = if ($CompilerCommand) { $CompilerCommand.Source } else { $null }
if (-not $CompilerPath) {
   $InstallRoots = @(
      (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 7"),
      (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6"),
      (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 7"),
      (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6"),
      (Join-Path $env:ProgramFiles "Inno Setup 7"),
      (Join-Path $env:ProgramFiles "Inno Setup 6")
   )
   foreach ($InstallRoot in $InstallRoots) {
      $Candidate = Join-Path $InstallRoot "ISCC.exe"
      if (Test-Path -LiteralPath $Candidate) {
         $CompilerPath = $Candidate
         break
      }
   }
}
if (-not $CompilerPath) {
   throw "Inno Setup was not found. Install it from https://jrsoftware.org/isinfo.php, then rerun this script."
}

$OutputDir = Join-Path $Root "dist\installer"
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
$InstallerScript = Join-Path $PSScriptRoot "installer.iss"
Write-Host "Building setup wizard for version $Version..."
$IsccArguments = @("/DMyAppVersion=$Version", $InstallerScript)
& $CompilerPath @IsccArguments
if ($LASTEXITCODE -ne 0) { throw "Inno Setup compilation failed." }

$Installer = Join-Path $OutputDir "KestrelStudio-Setup-x64.exe"
if (-not (Test-Path -LiteralPath $Installer -PathType Leaf)) {
   throw "Inno Setup did not create the expected installer: $Installer"
}
$InstallerSize = (Get-Item -LiteralPath $Installer).Length
if ($InstallerSize -lt 5MB) {
   throw "The setup executable looks incomplete ($InstallerSize bytes): $Installer"
}
Write-Host ("Installer complete: {0} ({1:N1} MB, version {2})" -f $Installer, ($InstallerSize / 1MB), $Version)
