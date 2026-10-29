$prismPidFile = Join-Path $PSScriptRoot 'runtime\server.pid'
if (Test-Path -LiteralPath $prismPidFile) {
    $prismServerId = [int](Get-Content -LiteralPath $prismPidFile)
    $prismProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $prismServerId" -ErrorAction SilentlyContinue
    if ($prismProcess -and $prismProcess.CommandLine -like '*uvicorn*backend.app:app*' -and $prismProcess.CommandLine.IndexOf($PSScriptRoot,[StringComparison]::OrdinalIgnoreCase) -ge 0) {
        # Windows venv python.exe may own a child interpreter. Stop only its matching server.
        $prismChildren = Get-CimInstance Win32_Process -Filter "ParentProcessId = $prismServerId" -ErrorAction SilentlyContinue
        foreach ($prismChild in $prismChildren) {
            if ($prismChild.CommandLine -like '*uvicorn*backend.app:app*') {
                Stop-Process -Id $prismChild.ProcessId -ErrorAction SilentlyContinue
            }
        }
        Stop-Process -Id $prismServerId -ErrorAction SilentlyContinue
        Write-Host 'Perspective Prism stopped.'
    }
    Remove-Item -LiteralPath $prismPidFile
}

