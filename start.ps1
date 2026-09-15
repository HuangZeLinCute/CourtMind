$ErrorActionPreference = 'Stop'

$projectRoot = [System.IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\')
$frontendRoot = Join-Path $projectRoot 'frontend'
$pythonExe = 'C:\Users\14181\.conda\envs\badminton\python.exe'
$logsRoot = Join-Path $projectRoot 'logs'
$env:NO_PROXY = '127.0.0.1,localhost'
$env:no_proxy = '127.0.0.1,localhost'

function Get-LocalHttpText([string]$Url) {
    try {
        $content = & curl.exe --noproxy '*' --silent --show-error --fail --max-time 2 $Url 2>$null
        if ($LASTEXITCODE -ne 0) { return $null }
        return ($content -join "`n")
    }
    catch { return $null }
}

function Test-Port([int]$Port) {
    return $null -ne (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

function Test-Backend {
    $content = Get-LocalHttpText 'http://127.0.0.1:8000/api/health'
    return $null -ne $content -and $content -match '"app"\s*:\s*"Good-Badminton API"'
}

function Test-Frontend {
    try {
        $response = Invoke-WebRequest -Uri 'http://127.0.0.1:5173/' -UseBasicParsing -TimeoutSec 2
        return $response.StatusCode -eq 200 -and $response.Content -match '<title>CourtMind'
    }
    catch { return $false }
}

if ((Test-Backend) -and (Test-Frontend)) {
    Write-Host 'Good-Badminton is already running:'
    Write-Host '  Backend:  http://127.0.0.1:8000'
    Write-Host '  Frontend: http://127.0.0.1:5173'
    Write-Host 'Run stop.bat directly in CMD to stop both services.'
    exit 0
}

$busyPorts = @(8000, 5173 | Where-Object { Test-Port $_ })
if ($busyPorts.Count -gt 0) {
    Write-Error "Port(s) $($busyPorts -join ', ') are already occupied. Run stop.bat directly in CMD, then try again."
    exit 1
}
if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    Write-Error "Conda environment 'badminton' was not found at $pythonExe"
    exit 1
}
if (-not (Test-Path -LiteralPath (Join-Path $frontendRoot 'node_modules') -PathType Container)) {
    Write-Error "Frontend dependencies are missing. Run 'cd frontend' and 'npm install'."
    exit 1
}

$npmCommand = Get-Command npm.cmd -ErrorAction SilentlyContinue
if ($null -eq $npmCommand) {
    Write-Error 'npm.cmd was not found. Install Node.js and add npm to PATH.'
    exit 1
}

New-Item -ItemType Directory -Force -Path $logsRoot | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$backendOut = Join-Path $logsRoot "backend-$stamp.log"
$backendErr = Join-Path $logsRoot "backend-$stamp.error.log"
$frontendOut = Join-Path $logsRoot "frontend-$stamp.log"
$frontendErr = Join-Path $logsRoot "frontend-$stamp.error.log"
$backendProcess = $null
$frontendProcess = $null
$env:KERAS_BACKEND = 'torch'

try {
    Write-Host 'Starting Good-Badminton backend...'
    $backendProcess = Start-Process -FilePath $pythonExe `
        -ArgumentList @('-m', 'uvicorn', 'backend.main:app', '--host', '127.0.0.1', '--port', '8000') `
        -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $backendOut -RedirectStandardError $backendErr

    $deadline = (Get-Date).AddSeconds(30)
    while (-not (Test-Backend) -and (Get-Date) -lt $deadline) {
        if ($backendProcess.HasExited) {
            throw "Backend exited during startup. See $backendErr"
        }
        Start-Sleep -Milliseconds 250
        $backendProcess.Refresh()
    }
    if (-not (Test-Backend)) { throw "Backend startup timed out. See $backendErr" }

    Write-Host 'Starting Good-Badminton frontend...'
    $frontendProcess = Start-Process -FilePath $npmCommand.Source `
        -ArgumentList @('run', 'dev', '--', '--host', '127.0.0.1', '--strictPort') `
        -WorkingDirectory $frontendRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $frontendOut -RedirectStandardError $frontendErr

    $deadline = (Get-Date).AddSeconds(30)
    while (-not (Test-Port 5173) -and (Get-Date) -lt $deadline) {
        if ($frontendProcess.HasExited) {
            throw "Frontend exited during startup. See $frontendErr"
        }
        Start-Sleep -Milliseconds 250
        $frontendProcess.Refresh()
    }
    if (-not (Test-Port 5173)) { throw "Frontend startup timed out. See $frontendErr" }
}
catch {
    foreach ($process in @($frontendProcess, $backendProcess)) {
        if ($null -ne $process -and -not $process.HasExited) {
            & taskkill.exe /PID $process.Id /T /F 2>&1 | Out-Null
        }
    }
    Write-Error $_
    exit 1
}

Write-Host 'Good-Badminton started successfully:'
Write-Host '  Backend:  http://127.0.0.1:8000'
Write-Host '  Frontend: http://127.0.0.1:5173'
Write-Host '  Logs:     logs\'
Write-Host 'Run stop.bat directly in CMD to stop both services.'
