Set-StrictMode -Version Latest

function Write-BootstrapMessage {
    param(
        [Parameter(Mandatory = $true)][string]$Message,
        [ValidateSet('INFO', 'OK', 'SKIP', 'WARN', 'ERROR')][string]$Level = 'INFO'
    )
    $Color = @{ INFO = 'Cyan'; OK = 'Green'; SKIP = 'DarkGray'; WARN = 'Yellow'; ERROR = 'Red' }[$Level]
    Write-Host ("[{0}] {1}" -f $Level, $Message) -ForegroundColor $Color
}

function Write-BootstrapStage {
    param(
        [Parameter(Mandatory = $true)][int]$Index,
        [Parameter(Mandatory = $true)][int]$Total,
        [Parameter(Mandatory = $true)][string]$Message
    )
    Write-Host ''
    Write-Host ("[{0}/{1}] {2}" -f $Index, $Total, $Message) -ForegroundColor Cyan
}

function Import-WindowsBootstrapConfig {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Bootstrap configuration is missing: $Path"
    }
    $Config = Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
    if ([int]$Config.schema_version -ne 1) {
        throw "Unsupported bootstrap configuration schema: $($Config.schema_version)"
    }
    foreach ($Name in @('cuda12', 'cuda13')) {
        $Entry = $Config.runtimes.$Name
        if ($null -eq $Entry) { throw "Bootstrap runtime is missing: $Name" }
        foreach ($Field in @('venv', 'requirements', 'expected_cuda', 'llama_image')) {
            if ([string]::IsNullOrWhiteSpace([string]$Entry.$Field)) {
                throw "Bootstrap runtime $Name has no $Field."
            }
        }
    }
    return $Config
}

function Invoke-BootstrapRetry {
    param(
        [Parameter(Mandatory = $true)][string]$Operation,
        [Parameter(Mandatory = $true)][scriptblock]$Action,
        [int]$Attempts = 3,
        [int]$BaseDelaySeconds = 2
    )
    for ($Attempt = 1; $Attempt -le $Attempts; $Attempt++) {
        try { & $Action; return }
        catch {
            if ($Attempt -ge $Attempts) { throw }
            $Delay = [int][Math]::Min(30, $BaseDelaySeconds * [Math]::Pow(2, $Attempt - 1))
            Write-BootstrapMessage "$Operation failed (attempt $Attempt/$Attempts). Retrying in $Delay second(s): $($_.Exception.Message)" 'WARN'
            Start-Sleep -Seconds $Delay
        }
    }
}

function Invoke-BootstrapCommand {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$Arguments = @(),
        [string]$WorkingDirectory = '',
        [switch]$Quiet
    )
    $Previous = Get-Location
    try {
        if ($WorkingDirectory) { Set-Location -LiteralPath $WorkingDirectory }
        if (-not $Quiet) { Write-BootstrapMessage ("Running: {0} {1}" -f $FilePath, ($Arguments -join ' ')) }
        & $FilePath @Arguments
        $Code = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
        if ($Code -ne 0) { throw "Command failed with exit code ${Code}: $FilePath" }
    }
    finally { Set-Location $Previous }
}

function Resolve-BootstrapPython312 {
    param([Parameter(Mandatory = $true)]$PythonContract)
    $Candidates = @(
        @{ File = 'py.exe'; Prefix = @('-3.12') },
        @{ File = 'py'; Prefix = @('-3.12') },
        @{ File = 'python.exe'; Prefix = @() },
        @{ File = 'python'; Prefix = @() },
        @{ File = 'python3.exe'; Prefix = @() },
        @{ File = 'python3'; Prefix = @() }
    )
    $Probe = 'import json,platform,struct,sys,venv; print(json.dumps({"exe":sys.executable,"major":sys.version_info[0],"minor":sys.version_info[1],"bits":struct.calcsize("P")*8,"implementation":platform.python_implementation()}))'
    foreach ($Candidate in $Candidates) {
        $Command = Get-Command $Candidate.File -ErrorAction SilentlyContinue
        if ($null -eq $Command) { continue }
        try {
            $Output = & $Command.Source @($Candidate.Prefix) -I -c $Probe 2>$null
            if ($LASTEXITCODE -ne 0 -or -not $Output) { continue }
            $Payload = ($Output -join '') | ConvertFrom-Json
            if (
                [int]$Payload.major -eq [int]$PythonContract.major -and
                [int]$Payload.minor -eq [int]$PythonContract.minor -and
                [int]$Payload.bits -eq [int]$PythonContract.bits -and
                [string]$Payload.implementation -eq [string]$PythonContract.implementation
            ) {
                return [pscustomobject]@{
                    Executable = $Command.Source
                    Prefix = [string[]]$Candidate.Prefix
                    ResolvedExecutable = [string]$Payload.exe
                }
            }
        }
        catch { continue }
    }
    throw 'Python 3.12 x64 (CPython) is required. Install it from python.org and enable the py launcher or PATH entry.'
}

function Enter-BootstrapLock {
    param([Parameter(Mandatory = $true)][string]$Path)
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
    try {
        return [System.IO.File]::Open($Path, 'OpenOrCreate', 'ReadWrite', 'None')
    }
    catch { throw 'Another bootstrap process is already running for this checkout and CUDA runtime.' }
}

function Test-BootstrapWritableDirectory {
    param([Parameter(Mandatory = $true)][string]$Path)
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
    $Probe = Join-Path $Path ('.comic-bootstrap-write-{0}.tmp' -f [Guid]::NewGuid().ToString('N'))
    try { [System.IO.File]::WriteAllText($Probe, 'ok') }
    finally { Remove-Item -LiteralPath $Probe -Force -ErrorAction SilentlyContinue }
}

function Assert-BootstrapFreeSpace {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][int64]$MinimumBytes,
        [Parameter(Mandatory = $true)][string]$Label
    )
    $Root = [System.IO.Path]::GetPathRoot((Resolve-Path -LiteralPath $Path).Path)
    $Drive = [System.IO.DriveInfo]::new($Root)
    if ($Drive.AvailableFreeSpace -lt $MinimumBytes) {
        throw ("$Label requires at least {0:N1} GiB free on {1}; available={2:N1} GiB." -f ($MinimumBytes / 1GB), $Root, ($Drive.AvailableFreeSpace / 1GB))
    }
}

function Get-DockerExecutable {
    foreach ($Name in @('docker.exe', 'docker')) {
        $Command = Get-Command $Name -ErrorAction SilentlyContinue
        if ($null -ne $Command) { return $Command.Source }
    }
    $Known = Join-Path $env:ProgramFiles 'Docker\Docker\resources\bin\docker.exe'
    if (Test-Path -LiteralPath $Known -PathType Leaf) { return $Known }
    throw 'Docker Desktop is required but docker.exe was not found.'
}

function Test-DockerReady {
    param([Parameter(Mandatory = $true)][string]$Docker)
    & $Docker info --format '{{.ServerVersion}}' *> $null
    return $LASTEXITCODE -eq 0
}

function Ensure-DockerDesktopReady {
    param(
        [Parameter(Mandatory = $true)][string]$Docker,
        [int]$TimeoutSeconds = 180,
        [switch]$ReadOnly
    )
    if (Test-DockerReady -Docker $Docker) { return }
    if ($ReadOnly) { throw 'Docker Desktop is installed but its Linux engine is not running.' }
    $Desktop = Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'
    if (-not (Test-Path -LiteralPath $Desktop -PathType Leaf)) {
        throw 'Docker Desktop Linux engine is not running and Docker Desktop.exe was not found.'
    }
    Write-BootstrapMessage 'Docker Desktop is stopped. Starting it now...'
    Start-Process -FilePath $Desktop | Out-Null
    $Started = [DateTime]::UtcNow
    while (([DateTime]::UtcNow - $Started).TotalSeconds -lt $TimeoutSeconds) {
        if (Test-DockerReady -Docker $Docker) { return }
        $Elapsed = [int]([DateTime]::UtcNow - $Started).TotalSeconds
        Write-Host ("  Waiting for Docker Desktop... {0}/{1}s" -f $Elapsed, $TimeoutSeconds)
        Start-Sleep -Seconds 5
    }
    throw "Docker Desktop did not become ready within $TimeoutSeconds seconds."
}

function Assert-DockerCompose {
    param([Parameter(Mandatory = $true)][string]$Docker)
    & $Docker compose version *> $null
    if ($LASTEXITCODE -ne 0) { throw 'Docker Compose v2 is required.' }
}

function Assert-NvidiaHost {
    $Command = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
    if ($null -eq $Command) { $Command = Get-Command 'nvidia-smi' -ErrorAction SilentlyContinue }
    if ($null -eq $Command) { throw 'NVIDIA driver tools were not found (nvidia-smi).' }
    & $Command.Source -L *> $null
    if ($LASTEXITCODE -ne 0) { throw 'NVIDIA GPU/driver validation failed (nvidia-smi -L).' }
}

function Stop-BootstrapManagedContainer {
    param(
        [Parameter(Mandatory = $true)][string]$Docker,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$OwnershipLabel
    )
    $Raw = & $Docker inspect --format '{{json .Config.Labels}}|{{.State.Running}}' $Name 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $Raw) { return }
    $Parts = ($Raw -join '') -split '\|', 2
    if ($Parts.Count -ne 2) { throw "Unable to inspect managed container ownership: $Name" }
    $Labels = $Parts[0] | ConvertFrom-Json
    if ($null -eq $Labels.PSObject.Properties[$OwnershipLabel]) {
        throw "Container name conflict: $Name exists but is not owned by Comic Translate."
    }
    if ($Parts[1].Trim().ToLowerInvariant() -eq 'true') {
        Write-BootstrapMessage "Stopping managed container before volume verification: $Name"
        Invoke-BootstrapCommand -FilePath $Docker -Arguments @('stop', '--time', '10', $Name) -Quiet
    }
}

function Set-BootstrapRuntimeEnvironment {
    param(
        [Parameter(Mandatory = $true)][string]$LlamaImage,
        [Parameter(Mandatory = $true)][string]$VenvRoot
    )
    $env:PYTHONNOUSERSITE = '1'; $env:PYTHONHOME = ''; $env:PYTHONPATH = ''
    $env:CUDA_PATH = ''; $env:CUDA_PATH_V13_1 = ''; $env:CUDA_HOME = ''; $env:CUDA_ROOT = ''; $env:CUDNN_PATH = ''
    $env:QT_QPA_PLATFORM = 'windows'
    $env:PYTHONWARNINGS = 'ignore:pkg_resources is deprecated as an API:UserWarning'
    $env:LLAMA_CPP_IMAGE = $LlamaImage
    $env:HUNYUAN_OCR_LLAMA_CPP_IMAGE = $LlamaImage
    $env:PADDLEOCR_LLAMA_CPP_IMAGE = $LlamaImage
    $env:PADDLEOCR_SPOTTING_LLAMA_CPP_IMAGE = $LlamaImage
    $env:MANGALMM_LLAMA_CPP_IMAGE = $LlamaImage
    $env:LLAMA_CPP_PULL_POLICY = 'missing'
    $LibraryPaths = @(
        (Join-Path $VenvRoot 'Lib\site-packages\torch\lib'),
        (Join-Path $VenvRoot 'Lib\site-packages\tensorrt_libs'),
        (Join-Path $VenvRoot 'Lib\site-packages\nvidia\cudnn\bin'),
        (Join-Path $VenvRoot 'Lib\site-packages\nvidia\cu12\bin\x86_64'),
        (Join-Path $VenvRoot 'Lib\site-packages\nvidia\cu13\bin\x86_64'),
        (Join-Path $VenvRoot 'Lib\site-packages\nvidia\cublas\bin'),
        (Join-Path $VenvRoot 'Lib\site-packages\nvidia\cuda_runtime\bin'),
        (Join-Path $VenvRoot 'Lib\site-packages\nvidia\cuda_nvrtc\bin'),
        (Join-Path $VenvRoot 'Lib\site-packages\nvidia\nvjitlink\bin')
    ) | Where-Object { Test-Path -LiteralPath $_ -PathType Container }
    if ($LibraryPaths.Count -gt 0) { $env:PATH = ((@($LibraryPaths) + @($env:PATH)) -join ';') }
}

Export-ModuleMember -Function @(
    'Write-BootstrapMessage', 'Write-BootstrapStage', 'Import-WindowsBootstrapConfig',
    'Invoke-BootstrapRetry', 'Invoke-BootstrapCommand', 'Resolve-BootstrapPython312',
    'Enter-BootstrapLock', 'Test-BootstrapWritableDirectory', 'Assert-BootstrapFreeSpace',
    'Get-DockerExecutable', 'Test-DockerReady', 'Ensure-DockerDesktopReady',
    'Assert-DockerCompose', 'Assert-NvidiaHost', 'Stop-BootstrapManagedContainer',
    'Set-BootstrapRuntimeEnvironment'
)
