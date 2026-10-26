param(
   [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root

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
& $CompilerPath $InstallerScript
if ($LASTEXITCODE -ne 0) { throw "Inno Setup compilation failed." }

$Installer = Join-Path $OutputDir "KestrelStudio-Setup-x64.exe"
if (-not (Test-Path -LiteralPath $Installer -PathType Leaf)) {
   throw "Inno Setup did not create the expected installer: $Installer"
}
Write-Host "Installer complete: $Installer"

