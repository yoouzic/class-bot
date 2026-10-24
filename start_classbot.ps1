[CmdletBinding()]
param(
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonPath = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$EnvPath = Join-Path $ProjectRoot ".env"

function Stop-Launcher([string]$Message) {
    Write-Host "[Classbot] $Message" -ForegroundColor Red
    exit 1
}

Write-Host "[Classbot] Checking local runtime..." -ForegroundColor Cyan

if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    Stop-Launcher "Missing .venv\Scripts\python.exe. Create the project environment first."
}

if (-not (Test-Path -LiteralPath $EnvPath -PathType Leaf)) {
    Stop-Launcher "Missing .env. Copy .env.example to .env and configure DEEPSEEK_API_KEY."
}

$EnvText = Get-Content -LiteralPath $EnvPath -Raw
if ($EnvText -notmatch '(?m)^\s*DEEPSEEK_API_KEY\s*=\s*\S+') {
    Stop-Launcher "DEEPSEEK_API_KEY is missing in .env."
}

$OldPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
try {
    & $PythonPath -c "import classbot, funasr, torch, torchaudio, pyaudiowpatch, soxr, docx, pptx, pypdf, transformers"
    if ($LASTEXITCODE -ne 0) {
        Stop-Launcher "ASR dependencies are unavailable. Run: .venv\Scripts\python.exe -m pip install -e `".[asr]`""
    }
}
catch {
    Stop-Launcher "Runtime dependency check failed: $($_.Exception.Message)"
}

if ($CheckOnly) {
    $env:PYTHONPATH = $OldPythonPath
    Write-Host "[Classbot] Check passed. The online defense assistant is ready." -ForegroundColor Green
    exit 0
}

Push-Location $ProjectRoot
try {
    Write-Host "[Classbot] Starting online defense assistant. Select a scenario and playback device in the window." -ForegroundColor Green
    & $PythonPath -m classbot
    $ExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
    $env:PYTHONPATH = $OldPythonPath
}

if ($ExitCode -ne 0) {
    Write-Host "[Classbot] Application exited with code $ExitCode." -ForegroundColor Red
    exit $ExitCode
}

Write-Host "[Classbot] Application stopped." -ForegroundColor Yellow
