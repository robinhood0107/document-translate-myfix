[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('cuda12', 'cuda13')]
    [string]$Runtime,
    [switch]$SourceVerify,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArguments = @()
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Import-Module (Join-Path $PSScriptRoot 'lib\WindowsBootstrap.psm1') -Force -DisableNameChecking
Import-Module (Join-Path $PSScriptRoot 'lib\ManagedRuntimeDocker.psm1') -Force -DisableNameChecking
$PythonContract = [pscustomobject]@{
    major = 3; minor = 12; implementation = 'CPython'; bits = 64
}
$PipTools = [pscustomobject]@{
    pip = '26.0.1'; wheel = '0.46.3'; setuptools = '80.9.0'
}
$RuntimeConfig = @{
    cuda12 = [pscustomobject]@{
        venv = '.venv-win'
        requirements = 'requirements-cuda12.txt'
        expected_cuda = '12.8'
    }
    cuda13 = [pscustomobject]@{
        venv = '.venv-win-cuda13'
        requirements = 'requirements-cuda13.txt'
        expected_cuda = '13.0'
    }
}[$Runtime]
$ManagedRuntimes = @(
    [pscustomobject]@{
        label = 'HunyuanOCR'
        script = 'scripts/prepare_hunyuanocr_llamacpp_runtime.ps1'
        volume = 'comic-translate-hunyuanocr-models-v2'
    },
    [pscustomobject]@{
        label = 'PaddleOCR VL'
        script = 'scripts/prepare_paddleocr_llamacpp_runtime.ps1'
        volume = 'comic-translate-paddleocr-vl-llamacpp-models-v1'
    },
    [pscustomobject]@{
        label = 'Gemma IQ4_NL'
        script = 'scripts/prepare_gemma_runtime.ps1'
        volume = 'comic-translate-gemma-models-v2'
    }
)
$ImagePolicy = Get-ManagedLlamaCppImagePolicy -Runtime $Runtime
$ActiveLlamaImage = [string]$ImagePolicy.Preferred

$RequiredFiles = @(
    'comic.py', 'controller.py', 'app\version.py', 'requirements-base.txt',
    [string]$RuntimeConfig.requirements, 'docker-compose.yaml',
    'hunyuanocr_docker_files\docker-compose.yaml',
    'paddleocr_vl_docker_files\docker-compose.yaml',
    'resources\translations\compiled\ct_ko.qm',
    'scripts\bootstrap_windows.ps1',
    'scripts\lib\WindowsBootstrap.psm1',
    'scripts\lib\ManagedRuntimeDocker.psm1',
    'scripts\lib\ManagedRuntimeModelSource.psm1',
    'scripts\prepare_gemma_runtime.ps1',
    'scripts\prepare_hunyuanocr_llamacpp_runtime.ps1',
    'scripts\prepare_paddleocr_llamacpp_runtime.ps1',
    'scripts\verify_windows_runtime.py'
)
foreach ($Relative in $RequiredFiles) {
    if (-not (Test-Path -LiteralPath (Join-Path $Root $Relative) -PathType Leaf)) {
        throw "Missing launcher-source file: $Relative"
    }
}
if ($SourceVerify) {
    Write-BootstrapMessage "The $Runtime launcher-source contract is valid." 'OK'
    exit 0
}

$Doctor = $RemainingArguments.Count -gt 0 -and $RemainingArguments[0] -eq '--doctor'
if ($Doctor) { $RemainingArguments = @($RemainingArguments | Select-Object -Skip 1) }
$BootstrapRoot = Join-Path $Root '.comic-bootstrap'
$ModelCache = Join-Path $Root 'models\managed-runtime-sources'
$LogDirectory = Join-Path $Root 'logs\bootstrap'
$LogPath = Join-Path $LogDirectory ("bootstrap-{0}-{1}.log" -f $Runtime, (Get-Date -Format 'yyyyMMdd-HHmmss'))
$VenvRoot = Join-Path $Root ([string]$RuntimeConfig.venv)
$VenvPython = Join-Path $VenvRoot 'Scripts\python.exe'
$Lock = $null
$TranscriptStarted = $false
$VenvBackup = ''
$TotalStages = 7
$DeveloperPythonOnly = [bool]$env:COMIC_BOOTSTRAP_ONLY
$SkipRuntimeSetup = [bool]$env:COMIC_SKIP_STARTUP_MODELS
$ExistingVenvValid = $false
$Python = $null

try {
    if (-not $Doctor) {
        New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null
        Start-Transcript -LiteralPath $LogPath -Force | Out-Null
        $TranscriptStarted = $true
        $Lock = Enter-BootstrapLock -Path (Join-Path $BootstrapRoot "bootstrap-$Runtime.lock")
    }
    Write-Host ''
    Write-Host 'Comic Translate Windows bootstrap' -ForegroundColor Cyan
    Write-BootstrapMessage "Runtime: $Runtime / Python environment: $($RuntimeConfig.venv)"
    Write-BootstrapMessage "llama.cpp preferred image: $ActiveLlamaImage"
    if ($ImagePolicy.Fallback) {
        Write-BootstrapMessage "llama.cpp compatibility fallback: $($ImagePolicy.Fallback)"
    }
    Write-BootstrapMessage 'Automatic local runtimes: HunyuanOCR, PaddleOCR VL, Gemma IQ4_NL (~16.5 GiB source models)'
    if (-not $Doctor) { Write-BootstrapMessage "Log: $LogPath" }

    Write-BootstrapStage 1 $TotalStages 'Checking Python 3.12 x64 and local paths'
    if (Test-Path -LiteralPath $VenvPython -PathType Leaf) {
        $ExistingVenvValid = (Invoke-BootstrapProbe -FilePath $VenvPython -Arguments @(
            '-I', '-c', "import platform,struct,sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) and platform.python_implementation() == 'CPython' and struct.calcsize('P') * 8 == 64 else 1)"
        )) -eq 0
    }
    if ($ExistingVenvValid) {
        Write-BootstrapMessage 'The existing isolated Python 3.12 environment is reusable; seed interpreter lookup skipped.' 'SKIP'
    } else {
        $Python = Resolve-BootstrapPython312 -PythonContract $PythonContract
        Write-BootstrapMessage "Using Python seed interpreter: $($Python.ResolvedExecutable)" 'OK'
    }
    if (-not $Doctor) {
        Test-BootstrapWritableDirectory -Path $Root
        Test-BootstrapWritableDirectory -Path $BootstrapRoot
        if (-not $ExistingVenvValid) {
            $VenvMinimumBytes = if ($Runtime -eq 'cuda12') { 8589934592 } else { 6442450944 }
            Assert-BootstrapFreeSpace -Path $Root -MinimumBytes $VenvMinimumBytes -Label 'Python runtime environment'
        }
        if (-not $DeveloperPythonOnly -and -not $SkipRuntimeSetup) {
            Test-BootstrapWritableDirectory -Path $ModelCache
        }
    }

    Write-BootstrapStage 2 $TotalStages 'Checking WSL, Docker Desktop, Compose, and NVIDIA GPU'
    if ($DeveloperPythonOnly -or $SkipRuntimeSetup) {
        Write-BootstrapMessage 'External runtime checks skipped by developer bootstrap/smoke mode.' 'SKIP'
        $Docker = ''
    } else {
        $Wsl = Get-Command 'wsl.exe' -ErrorAction SilentlyContinue
        if ($null -eq $Wsl) { throw 'WSL2 is required but wsl.exe was not found.' }
        & $Wsl.Source --status *> $null
        if ($LASTEXITCODE -ne 0) { throw 'WSL2 status check failed. Finish WSL2 setup and reboot Windows.' }
        $Docker = Get-DockerExecutable
        Ensure-DockerDesktopReady -Docker $Docker -ReadOnly:$Doctor
        Assert-DockerCompose -Docker $Docker
        Assert-NvidiaHost
        Write-BootstrapMessage 'Docker Desktop, Compose, WSL2, and NVIDIA checks passed.' 'OK'
    }

    Write-BootstrapStage 3 $TotalStages 'Creating or repairing the isolated Python environment'
    if ($Doctor) {
        if (Test-Path -LiteralPath $VenvPython -PathType Leaf) { Write-BootstrapMessage "Environment exists: $VenvRoot" 'OK' }
        else { Write-BootstrapMessage "Environment is not installed yet: $VenvRoot" 'WARN' }
    } else {
        if (-not $ExistingVenvValid -and (Test-Path -LiteralPath $VenvRoot)) {
            $VenvBackup = "$VenvRoot.bootstrap-backup"
            if (Test-Path -LiteralPath $VenvBackup) {
                throw "A previous bootstrap backup already exists: $VenvBackup"
            }
            Write-BootstrapMessage "Quarantining an incompatible or incomplete environment: $VenvRoot" 'WARN'
            Move-Item -LiteralPath $VenvRoot -Destination $VenvBackup
        }
        if (-not $ExistingVenvValid) {
            Write-BootstrapMessage "Creating virtual environment: $VenvRoot"
            Invoke-BootstrapCommand -FilePath $Python.Executable -Arguments @($Python.Prefix + @('-m', 'venv', $VenvRoot)) -WorkingDirectory $Root
        }
        $Pyvenv = Join-Path $VenvRoot 'pyvenv.cfg'
        if (-not (Test-Path -LiteralPath $Pyvenv -PathType Leaf)) { throw "Invalid virtual environment: $VenvRoot" }
        if ((Get-Content -LiteralPath $Pyvenv -Raw) -notmatch '(?im)^include-system-site-packages\s*=\s*false\s*$') {
            throw "Virtual environment is not isolated from system packages: $Pyvenv"
        }
        Write-BootstrapMessage 'The isolated Python environment is ready.' 'OK'
    }

    Write-BootstrapStage 4 $TotalStages 'Verifying and synchronizing pinned Python packages'
    if ($Doctor) {
        if (Test-Path -LiteralPath $VenvPython -PathType Leaf) {
            $DoctorVerify = @(
                '-B', '-s', (Join-Path $Root 'scripts\verify_windows_runtime.py'),
                '--requirements', (Join-Path $Root ([string]$RuntimeConfig.requirements)),
                '--expected-cuda', ([string]$RuntimeConfig.expected_cuda)
            )
            if ((Invoke-BootstrapProbe -FilePath $VenvPython -Arguments $DoctorVerify) -eq 0) {
                Write-BootstrapMessage 'Pinned packages are valid.' 'OK'
            }
            else { Write-BootstrapMessage 'Pinned packages need repair.' 'WARN' }
        }
        foreach ($Image in @($ImagePolicy.Preferred, $ImagePolicy.Fallback) |
            Where-Object { $_ }) {
            if ((Invoke-BootstrapProbe -FilePath $Docker -Arguments @('image', 'inspect', $Image)) -eq 0) {
                Write-BootstrapMessage "Docker image is installed: $Image" 'OK'
            } else {
                Write-BootstrapMessage "Docker image is not installed yet: $Image" 'WARN'
            }
        }
        foreach ($Volume in $ManagedRuntimes.volume) {
            if ((Invoke-BootstrapProbe -FilePath $Docker -Arguments @('volume', 'inspect', $Volume)) -eq 0) {
                Write-BootstrapMessage "Docker model volume is installed: $Volume" 'OK'
            } else {
                Write-BootstrapMessage "Docker model volume is not installed yet: $Volume" 'WARN'
            }
        }
        Write-BootstrapMessage 'Doctor mode is read-only; no files, packages, images, or volumes were changed.' 'OK'
        exit 0
    }

    Set-BootstrapRuntimeEnvironment -LlamaImage $ActiveLlamaImage -VenvRoot $VenvRoot
    $RuntimeVerificationArguments = @(
        '-B', '-s', (Join-Path $Root 'scripts\verify_windows_runtime.py'),
        '--requirements', (Join-Path $Root ([string]$RuntimeConfig.requirements)),
        '--expected-cuda', ([string]$RuntimeConfig.expected_cuda)
    )
    if ((Invoke-BootstrapProbe -FilePath $VenvPython -Arguments $RuntimeVerificationArguments) -ne 0) {
        Invoke-BootstrapRetry -Operation 'pip tool installation' -Attempts 4 -Action {
            Invoke-BootstrapCommand -FilePath $VenvPython -Arguments @('-m', 'pip', 'install', '--disable-pip-version-check', '--retries', '5', '--timeout', '60', '--upgrade', "pip==$($PipTools.pip)", "wheel==$($PipTools.wheel)", "setuptools==$($PipTools.setuptools)") -WorkingDirectory $Root
        }
        Invoke-BootstrapRetry -Operation 'pinned runtime installation' -Attempts 4 -Action {
            Invoke-BootstrapCommand -FilePath $VenvPython -Arguments @('-m', 'pip', 'install', '--disable-pip-version-check', '--retries', '5', '--timeout', '60', '-r', (Join-Path $Root ([string]$RuntimeConfig.requirements))) -WorkingDirectory $Root
        }
    } else { Write-BootstrapMessage 'Pinned packages already match; installation skipped.' 'SKIP' }
    Invoke-BootstrapCommand -FilePath $VenvPython -Arguments @('-B', '-s', (Join-Path $Root 'scripts\verify_windows_runtime.py'), '--requirements', (Join-Path $Root ([string]$RuntimeConfig.requirements)), '--expected-cuda', ([string]$RuntimeConfig.expected_cuda)) -WorkingDirectory $Root
    Invoke-BootstrapCommand -FilePath $VenvPython -Arguments @('-m', 'pip', 'check') -WorkingDirectory $Root
    Invoke-BootstrapCommand -FilePath $VenvPython -Arguments @('-B', '-s', '-c', "import PySide6, cv2, numpy, onnxruntime, torch; print('runtime core imports passed')") -WorkingDirectory $Root
    if ($VenvBackup -and (Test-Path -LiteralPath $VenvBackup)) {
        Remove-Item -LiteralPath $VenvBackup -Recurse -Force
        $VenvBackup = ''
    }
    if ($env:COMIC_BOOTSTRAP_ONLY) {
        Write-BootstrapMessage "$Runtime Python environment is ready (developer bootstrap-only mode)." 'OK'
        exit 0
    }

    Write-BootstrapStage 5 $TotalStages 'Preparing required application models'
    if (-not $SkipRuntimeSetup) {
        Invoke-BootstrapRetry -Operation 'required application model preparation' -Attempts 3 -Action {
            Invoke-BootstrapCommand -FilePath $VenvPython -Arguments @('-B', '-s', '-c', 'from modules.utils.download import ensure_startup_runtime_models; ensure_startup_runtime_models(prefer_cuda=True)') -WorkingDirectory $Root
        }
    } else { Write-BootstrapMessage 'Application model preparation skipped by smoke/test environment.' 'SKIP' }

    $SkipManagedBootstrap = $SkipRuntimeSetup
    Write-BootstrapStage 6 $TotalStages 'Preparing llama.cpp images and managed model volumes'
    if ($SkipManagedBootstrap) {
        Write-BootstrapMessage 'Managed model volume preparation skipped by smoke/test environment.' 'SKIP'
    } else {
        $ImageCandidates = @($ActiveLlamaImage)
        if ($ImagePolicy.Fallback) {
            $ImageCandidates += [string]$ImagePolicy.Fallback
        }
        $Provisioned = $false
        for ($ImageIndex = 0; $ImageIndex -lt $ImageCandidates.Count; $ImageIndex++) {
            $CandidateImage = $ImageCandidates[$ImageIndex]
            $HasFallback = $ImageIndex -lt ($ImageCandidates.Count - 1)
            try {
                Set-BootstrapRuntimeEnvironment -LlamaImage $CandidateImage -VenvRoot $VenvRoot
                Stop-BootstrapManagedContainer -Docker $Docker -Name 'hunyuanocr-local-server' -OwnershipLabel 'com.comictranslate.hunyuanocr-model-volume'
                Stop-BootstrapManagedContainer -Docker $Docker -Name 'paddleocr-llamacpp' -OwnershipLabel 'com.comictranslate.paddleocr-model-volume'
                Stop-BootstrapManagedContainer -Docker $Docker -Name 'gemma-local-server' -OwnershipLabel 'comic-translate.runtime'
                foreach ($Managed in $ManagedRuntimes) {
                    $Script = Join-Path $Root ([string]$Managed.script)
                    Write-BootstrapMessage "Preparing $($Managed.label) with $CandidateImage..."
                    Invoke-BootstrapRetry `
                        -Operation "$($Managed.label) preparation" `
                        -Attempts $(if ($HasFallback) { 1 } else { 3 }) `
                        -Action {
                            Invoke-BootstrapCommand -FilePath 'powershell.exe' -Arguments @(
                                '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', $Script,
                                '-Mode', 'Auto', '-AllowDownload', '-DownloadDirectory', $ModelCache,
                                '-ImageRef', $CandidateImage
                            ) -WorkingDirectory $Root
                        }
                    Write-BootstrapMessage "$($Managed.label) is ready." 'OK'
                }
                $ActiveLlamaImage = $CandidateImage
                $Provisioned = $true
                break
            }
            catch {
                if (-not $HasFallback) { throw }
                Write-BootstrapMessage (
                    "The preferred llama.cpp image failed its real GPU smoke. " +
                    "Retrying every managed runtime with the compatibility image: " +
                    $ImageCandidates[$ImageIndex + 1]
                ) 'WARN'
            }
        }
        if (-not $Provisioned) {
            throw 'No supported llama.cpp CUDA image completed managed runtime provisioning.'
        }
    }

    Write-BootstrapStage 7 $TotalStages 'Launching Comic Translate'
    Write-BootstrapMessage 'Bootstrap completed. Future launches reuse verified packages, downloads, images, and model volumes.' 'OK'
    Invoke-BootstrapCommand -FilePath $VenvPython -Arguments (@('-B', '-s', (Join-Path $Root 'comic.py')) + @($RemainingArguments)) -WorkingDirectory $Root -Quiet
}
catch {
    if ($VenvBackup -and (Test-Path -LiteralPath $VenvBackup)) {
        Remove-Item -LiteralPath $VenvRoot -Recurse -Force -ErrorAction SilentlyContinue
        Move-Item -LiteralPath $VenvBackup -Destination $VenvRoot -ErrorAction SilentlyContinue
    }
    Write-BootstrapMessage "BOOTSTRAP_FAILED [$Runtime]: $($_.Exception.Message)" 'ERROR'
    if ($TranscriptStarted) { Write-BootstrapMessage "Log: $LogPath" 'ERROR' }
    Write-BootstrapMessage 'Fix the reported prerequisite if necessary, then run the same BAT again. Verified files and partial downloads will be reused.' 'ERROR'
    exit 1
}
finally {
    if ($null -ne $Lock) { $Lock.Dispose() }
    if ($TranscriptStarted) { Stop-Transcript | Out-Null }
}
