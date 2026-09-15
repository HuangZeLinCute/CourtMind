$ErrorActionPreference = 'Stop'

$projectRoot = [System.IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\')
$normalizedProjectRoot = $projectRoot.ToLowerInvariant().Replace('/', '\')
$processes = @(Get-CimInstance Win32_Process)
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

function Get-ProcessById([int]$ProcessId) {
    return $processes | Where-Object { $_.ProcessId -eq $ProcessId } | Select-Object -First 1
}

function Test-GoodBadmintonService([int]$Port, $Process) {
    $commandLine = ([string]$Process.CommandLine).ToLowerInvariant().Replace('/', '\')
    $executable = ([string]$Process.ExecutablePath).ToLowerInvariant().Replace('/', '\')

    if ($Port -eq 8000) {
        $isExpectedProcess =
            $Process.Name -ieq 'python.exe' -and
            $executable.EndsWith('\.conda\envs\badminton\python.exe') -and
            $commandLine.Contains('uvicorn backend.main:app') -and
            $commandLine.Contains('--port 8000')
        if (-not $isExpectedProcess) { return $false }

        $content = Get-LocalHttpText 'http://127.0.0.1:8000/api/health'
        return $null -ne $content -and $content -match '"app"\s*:\s*"Good-Badminton API"'
    }

    if ($Port -eq 5173) {
        $expectedVitePath = "$normalizedProjectRoot\frontend\node_modules"
        $isExpectedProcess =
            $Process.Name -ieq 'node.exe' -and
            $commandLine.Contains($expectedVitePath) -and
            $commandLine.Contains('vite') -and
            $commandLine.Contains('--strictport')
        if (-not $isExpectedProcess) { return $false }

        $content = Get-LocalHttpText 'http://127.0.0.1:5173/'
        return $null -ne $content -and $content -match '<title>CourtMind'
    }

    return $false
}

function Test-GoodBadmintonHealth([int]$Port) {
    if ($Port -eq 8000) {
        $content = Get-LocalHttpText 'http://127.0.0.1:8000/api/health'
        return $null -ne $content -and $content -match '"app"\s*:\s*"Good-Badminton API"'
    }
    if ($Port -eq 5173) {
        $content = Get-LocalHttpText 'http://127.0.0.1:5173/'
        return $null -ne $content -and $content -match '<title>Good-Badminton'
    }
    return $false
}

$targets = @()
$refused = @()

foreach ($port in 8000, 5173) {
    $listeners = @(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
    foreach ($listener in $listeners) {
        $process = Get-ProcessById ([int]$listener.OwningProcess)
        if ($null -ne $process -and (Test-GoodBadmintonService $port $process)) {
            $targets += [PSCustomObject]@{ Port = $port; Process = $process }
        }
        elseif ($port -eq 8000 -and (Test-GoodBadmintonHealth $port)) {
            # When Uvicorn --reload is launched through WSL, Windows can retain
            # the listening reloader PID after it disappears from Win32_Process.
            # Its multiprocessing server child remains visible and inherits the
            # socket. Validate that exact child before stopping it.
            $children = @($processes | Where-Object {
                $_.ParentProcessId -eq $listener.OwningProcess -and
                $_.Name -ieq 'python.exe' -and
                ([string]$_.ExecutablePath).ToLowerInvariant().Replace('/', '\').EndsWith('\.conda\envs\badminton\python.exe') -and
                ([string]$_.CommandLine).Contains('multiprocessing.spawn')
            })
            if ($children.Count -gt 0) {
                foreach ($child in $children) {
                    $targets += [PSCustomObject]@{ Port = $port; Process = $child }
                }
            }
            else {
                $refused += $port
            }
        }
        else {
            $refused += $port
        }
    }
}

if ($refused.Count -gt 0) {
    $ports = ($refused | Sort-Object -Unique) -join ', '
    Write-Error "Port(s) $ports are occupied, but ownership could not be verified as this Good-Badminton project. Nothing was stopped."
    exit 1
}

if ($targets.Count -eq 0) {
    Write-Host 'Good-Badminton is not running on ports 8000 and 5173.'
    exit 0
}

foreach ($target in ($targets | Sort-Object Port -Descending)) {
    $pidToStop = [int]$target.Process.ProcessId
    Write-Host "Stopping port $($target.Port): $($target.Process.Name) (PID $pidToStop)..."
    & taskkill.exe /PID $pidToStop /T /F 2>&1 | Out-Null
}

$deadline = (Get-Date).AddSeconds(5)
do {
    $remaining = @(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
        Where-Object { $_.LocalPort -in 8000, 5173 })
    if ($remaining.Count -eq 0) { break }
    Start-Sleep -Milliseconds 200
} while ((Get-Date) -lt $deadline)

if ($remaining.Count -gt 0) {
    $ports = ($remaining.LocalPort | Sort-Object -Unique) -join ', '
    Write-Error "The stop command ran, but port(s) $ports are still listening."
    exit 1
}

Write-Host 'Good-Badminton backend and frontend have stopped.'
