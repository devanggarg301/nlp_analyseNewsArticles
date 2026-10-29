param([switch]$WithModels)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    if (!(Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        python -m venv --without-pip .venv
        if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 or newer is required.' }
    }
    $prismRequirements = if ($WithModels) { 'backend/requirements-models.txt' } else { 'backend/requirements.txt' }
    python -m pip --python .venv/Scripts/python.exe install -r $prismRequirements
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    Push-Location frontend
    try {
        npm.cmd ci
        if ($LASTEXITCODE -ne 0) { throw 'Frontend installation failed.' }
        npm.cmd run build
        if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
    } finally { Pop-Location }
    if ($WithModels) {
        .\.venv\Scripts\python.exe scripts/prepare_models.py
        if ($LASTEXITCODE -ne 0) { throw 'A model download failed. Re-run prepare_models.py.' }
    }
    Write-Host 'Setup complete. Run Start-Prism.ps1.' -ForegroundColor Green
} finally { Pop-Location }

