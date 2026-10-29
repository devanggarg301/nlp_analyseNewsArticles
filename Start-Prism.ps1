param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$prismRoot = $PSScriptRoot
$prismPython = Join-Path $prismRoot '.venv\Scripts\python.exe'
if (!(Test-Path -LiteralPath $prismPython)) {
    throw 'Environment not installed. Run Setup-Prism.ps1 first.'
}
$prismRuntime = Join-Path $prismRoot 'runtime'
New-Item -ItemType Directory -Force -Path $prismRuntime | Out-Null
$prismReady = $false
try { $prismResponse = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/health' -TimeoutSec 3; $prismReady = $prismResponse.status -eq 'ok' } catch {}
if (!$prismReady) {
    $prismProcess = Start-Process -FilePath $prismPython -ArgumentList @('-m','uvicorn','backend.app:app','--host','127.0.0.1','--port','8000') -WorkingDirectory $prismRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $prismRuntime 'server.log') -RedirectStandardError (Join-Path $prismRuntime 'server-error.log') -PassThru
    Set-Content -LiteralPath (Join-Path $prismRuntime 'server.pid') -Value $prismProcess.Id
    for ($prismAttempt=0; $prismAttempt -lt 45; $prismAttempt++) {
        Start-Sleep -Seconds 1
        try { $prismResponse = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/health' -TimeoutSec 2; $prismReady = $prismResponse.status -eq 'ok'; if ($prismReady) { break } } catch {}
        if ($prismProcess.HasExited) { throw 'Server failed. See runtime/server-error.log.' }
    }
}
if (!$prismReady) { throw 'Server is still starting. Inspect runtime/server-error.log.' }
Write-Host 'Perspective Prism is ready at http://127.0.0.1:8000' -ForegroundColor Green
if (!$NoBrowser) { Start-Process 'http://127.0.0.1:8000' }

