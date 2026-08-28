[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('cuda12', 'cuda13')]
    [string]$Runtime,
    [switch]$SourceVerify,
    [switch]$Full,
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
$AllManagedRuntimes = @(
    [pscustomobject]@{
        label = 'HunyuanOCR'
        script = 'scripts/prepare_hunyuanocr_llamacpp_runtime.ps1'
        volume = 'comic-translate-hunyuanocr-models-v2'
        runtime_name = 'HunyuanOCR-llama.cpp'
        preparation_version = 1
        container = 'hunyuanocr-local-server'
        ownership_label = 'com.comictranslate.hunyuanocr-model-volume'
        tier = 'core'
    },
    [pscustomobject]@{
        label = 'PaddleOCR VL'
        script = 'scripts/prepare_paddleocr_llamacpp_runtime.ps1'
        volume = 'comic-translate-paddleocr-vl-llamacpp-models-v1'
        runtime_name = 'PaddleOCR-VL-llama.cpp'
        preparation_version = 1
        container = 'paddleocr-llamacpp'
        ownership_label = 'com.comictranslate.paddleocr-model-volume'
        tier = 'core'
    },
    [pscustomobject]@{
        label = 'Gemma IQ4_NL'
        script = 'scripts/prepare_gemma_runtime.ps1'
        volume = 'comic-translate-gemma-models-v2'
        runtime_name = 'Gemma'
        preparation_version = 2
        container = 'gemma-local-server'
        ownership_label = 'comic-translate.runtime'
        tier = 'core'
    },
    [pscustomobject]@{
        label = 'MangaLMM'
        script = 'scripts/prepare_mangalmm_llamacpp_runtime.ps1'
        volume = 'comic-translate-mangalmm-models-v2'
        runtime_name = 'MangaLMM-llama.cpp'
        preparation_version = 2
        container = 'mangalmm-local-server'
        ownership_label = 'com.comictranslate.mangalmm-model-volume'
        tier = 'full'
    },
    [pscustomobject]@{
        label = 'PaddleOCR VL Spotting'
        script = 'scripts/prepare_paddleocr_spotting_llamacpp_runtime.ps1'
        volume = 'comic-translate-paddleocr-vl-spotting-llamacpp-models-v2'
        runtime_name = 'PaddleOCR-VL-Spotting-llama.cpp'
        preparation_version = 2
        container = 'paddleocr-spotting-llamacpp'
        ownership_label = 'com.comictranslate.paddleocr-spotting-model-volume'
        tier = 'full'
    }
)
$ProvisioningTier = if ($Full) { 'full' } else { 'core' }
$ManagedRuntimes = @(
    $AllManagedRuntimes | Where-Object { $Full -or $_.tier -eq 'core' }
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
    'scripts\configure_console.ps1',
    'scripts\lib\WindowsBootstrap.psm1',
    'scripts\lib\ManagedRuntimeDocker.psm1',
    'scripts\lib\ManagedRuntimeModelSource.psm1',
    'scripts\prepare_gemma_runtime.ps1',
    'scripts\prepare_hunyuanocr_llamacpp_runtime.ps1',
    'scripts\prepare_paddleocr_llamacpp_runtime.ps1',
    'scripts\run_windows.cmd',
    'scripts\setup_windows.cmd',
    'scripts\windows_install_state.py',
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
$ManagedRuntimeStatePath = Join-Path $BootstrapRoot ("managed-runtimes-{0}.json" -f $Runtime)
$ModelCache = Join-Path $Root 'models\managed-runtime-sources'
$LogDirectory = Join-Path $Root 'logs\bootstrap'
$LogPath = Join-Path $LogDirectory ("setup-{0}-{1}-{2}.log" -f $Runtime, $ProvisioningTier, (Get-Date -Format 'yyyyMMdd-HHmmss'))
$DetailLogPath = Join-Path $LogDirectory ("setup-{0}-{1}-{2}-detail.log" -f $Runtime, $ProvisioningTier, (Get-Date -Format 'yyyyMMdd-HHmmss'))
$VenvRoot = Join-Path $Root ([string]$RuntimeConfig.venv)
$VenvPython = Join-Path $VenvRoot 'Scripts\python.exe'
$Lock = $null
$TranscriptStarted = $false
$VenvBackup = ''
$TotalStages = 6
$DeveloperPythonOnly = [bool]$env:COMIC_BOOTSTRAP_ONLY
$SkipRuntimeSetup = [bool]$env:COMIC_SKIP_STARTUP_MODELS
$ExistingVenvValid = $false
$Python = $null
$HostCudaCompatibility = $null
$ActiveImageCompatibility = $null

try {
    if (-not $Doctor) {
        New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null
        Start-Transcript -LiteralPath $LogPath -Force | Out-Null
        $TranscriptStarted = $true
        $Lock = Enter-BootstrapLock -Path (Join-Path $BootstrapRoot "bootstrap-$Runtime.lock")
    }
    $ServiceSummary = @($ManagedRuntimes | ForEach-Object { [string]$_.label }) -join ' | '
    $LogDisplay = "logs\bootstrap\$([IO.Path]::GetFileName($LogPath))"
    $DetailLogDisplay = "logs\bootstrap\$([IO.Path]::GetFileName($DetailLogPath))"
    Write-Host ''
    Write-Host '+----------------------------------------------------------------------------+' -ForegroundColor DarkGray
    Write-Host '| Comic Translate Setup                                                      |' -ForegroundColor White
    Write-Host '+----------------------------------------------------------------------------+' -ForegroundColor DarkGray
    Write-Host ("  Runtime : {0,-8}  Python : {1}" -f $Runtime.ToUpperInvariant(), $RuntimeConfig.venv) -ForegroundColor Gray
    Write-Host ("  Tier    : {0,-8}  Models : {1}" -f $ProvisioningTier.ToUpperInvariant(), $ServiceSummary) -ForegroundColor Gray
    Write-Host ("  Image   : {0}" -f $ActiveLlamaImage) -ForegroundColor Gray
    if ($ImagePolicy.Fallback) {
        Write-Host ("  Fallback: {0}" -f $ImagePolicy.Fallback) -ForegroundColor Gray
    }
    if (-not $Doctor) {
        Write-Host ("  Log     : {0}" -f $LogDisplay) -ForegroundColor DarkGray
        Write-Host ("  Details : {0}" -f $DetailLogDisplay) -ForegroundColor DarkGray
    }
    Write-Host '+----------------------------------------------------------------------------+' -ForegroundColor DarkGray
    if (-not $Full) {
        Write-BootstrapMessage 'Optional MangaLMM/Spotting: run setup_full.bat.' 'SKIP'
    }
    if (-not $Doctor) {
        $env:COMIC_BOOTSTRAP_DETAIL_LOG = $DetailLogPath
    }

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
        $HostCudaCompatibility = Get-NvidiaCudaCompatibilityVersion
        Write-BootstrapMessage (
            "NVIDIA driver CUDA compatibility: $HostCudaCompatibility"
        ) 'OK'
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
                '--expected-cuda', ([string]$RuntimeConfig.expected_cuda),
                '--metadata-only'
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
        '--expected-cuda', ([string]$RuntimeConfig.expected_cuda),
        '--metadata-only'
    )
    Write-BootstrapMessage '[packages] Comparing 36 exact package pins (no CUDA DLL load)...'
    $RuntimeAlreadyValid = (
        Invoke-BootstrapProbe -FilePath $VenvPython -Arguments $RuntimeVerificationArguments
    ) -eq 0
    if (-not $RuntimeAlreadyValid) {
        Invoke-BootstrapRetry -Operation 'pip tool installation' -Attempts 4 -Action {
            Invoke-BootstrapCommand -FilePath $VenvPython -Arguments @('-m', 'pip', 'install', '--disable-pip-version-check', '--retries', '5', '--timeout', '60', '--upgrade', "pip==$($PipTools.pip)", "wheel==$($PipTools.wheel)", "setuptools==$($PipTools.setuptools)") -WorkingDirectory $Root
        }
        Invoke-BootstrapRetry -Operation 'pinned runtime installation' -Attempts 4 -Action {
            Invoke-BootstrapCommand -FilePath $VenvPython -Arguments @('-m', 'pip', 'install', '--disable-pip-version-check', '--retries', '5', '--timeout', '60', '-r', (Join-Path $Root ([string]$RuntimeConfig.requirements))) -WorkingDirectory $Root
        }
        Write-BootstrapMessage '[packages 1/2] Rechecking the repaired runtime...'
        Invoke-BootstrapCommand -FilePath $VenvPython -Arguments $RuntimeVerificationArguments -WorkingDirectory $Root -ShowOutput
        Write-BootstrapMessage '[packages 2/2] Checking repaired dependency consistency...'
        Invoke-BootstrapCommand -FilePath $VenvPython -Arguments @('-m', 'pip', 'check') -WorkingDirectory $Root -Quiet
    } else {
        Write-BootstrapMessage 'Pinned package metadata already matches; installation skipped.' 'SKIP'
        Write-BootstrapMessage 'Dependency consistency is unchanged; pip check skipped.' 'SKIP'
    }
    Write-BootstrapMessage 'Pinned package metadata passed.' 'OK'
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
            $PreviousDownloadPolicy = $env:COMIC_MODEL_DOWNLOAD_POLICY
            $PreviousProgressStyle = $env:COMIC_DOWNLOAD_PROGRESS_STYLE
            try {
                $env:COMIC_MODEL_DOWNLOAD_POLICY = ''
                $env:COMIC_DOWNLOAD_PROGRESS_STYLE = 'compact'
                Invoke-BootstrapCommand -FilePath $VenvPython -Arguments @(
                    '-u', '-B', '-s', (Join-Path $Root 'scripts\windows_install_state.py'),
                    'provision', '--runtime', $Runtime, '--profile', $ProvisioningTier,
                    '--requirements', ([string]$RuntimeConfig.requirements)
                ) -WorkingDirectory $Root -LiveOutput
            }
            finally {
                $env:COMIC_MODEL_DOWNLOAD_POLICY = $PreviousDownloadPolicy
                $env:COMIC_DOWNLOAD_PROGRESS_STYLE = $PreviousProgressStyle
            }
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
                $Compatibility = Get-BootstrapDockerImageCudaCompatibility `
                    -Docker $Docker `
                    -Image $CandidateImage `
                    -HostCudaVersion $HostCudaCompatibility
                if (-not $Compatibility.Compatible) {
                    Write-BootstrapMessage (
                        "Skipping incompatible llama.cpp image ${CandidateImage}: " +
                        "image requires CUDA >= $($Compatibility.RequiredCudaVersion), " +
                        "installed driver supports CUDA $($Compatibility.HostCudaVersion)."
                    ) 'WARN'
                    continue
                }
                if (Test-BootstrapManagedRuntimeState `
                    -Docker $Docker `
                    -Path $ManagedRuntimeStatePath `
                    -ImageRef $CandidateImage `
                    -ImageId $Compatibility.ImageId `
                    -ManagedRuntimes $ManagedRuntimes) {
                    Write-BootstrapMessage (
                        'Managed runtime seals, volumes, and image identity are unchanged; ' +
                        'skipping model revalidation.'
                    ) 'SKIP'
                    $ActiveLlamaImage = $CandidateImage
                    $ActiveImageCompatibility = $Compatibility
                    $Provisioned = $true
                    break
                }
                Set-BootstrapRuntimeEnvironment -LlamaImage $CandidateImage -VenvRoot $VenvRoot
                foreach ($Managed in $ManagedRuntimes) {
                    Stop-BootstrapManagedContainer `
                        -Docker $Docker `
                        -Name ([string]$Managed.container) `
                        -OwnershipLabel ([string]$Managed.ownership_label)
                }
                foreach ($Managed in $ManagedRuntimes) {
                    $Script = Join-Path $Root ([string]$Managed.script)
                    # The spotting projector is derived locally, so that script needs a
                    # real interpreter. Hand it this runtime's venv rather than letting
                    # it assume .venv-win, which does not exist for a cuda13-only setup.
                    $PrepareExtraArguments = @()
                    if ([string]$Managed.runtime_name -eq 'PaddleOCR-VL-Spotting-llama.cpp') {
                        $PrepareExtraArguments = @('-PythonExecutable', $VenvPython)
                    }
                    Write-BootstrapMessage (
                        "Preparing $($Managed.label) with $CandidateImage " +
                        '(download/resume and GPU smoke can take several minutes)...'
                    )
                    Invoke-BootstrapRetry `
                        -Operation "$($Managed.label) preparation" `
                        -Attempts $(if ($HasFallback) { 1 } else { 3 }) `
                        -Action {
                            Invoke-BootstrapCommand -FilePath 'powershell.exe' -Arguments (@(
                                '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', $Script,
                                '-Mode', 'Auto', '-AllowDownload', '-DownloadDirectory', $ModelCache,
                                '-ImageRef', $CandidateImage
                            ) + $PrepareExtraArguments) -WorkingDirectory $Root -LiveOutput
                        }
                    Write-BootstrapMessage "$($Managed.label) is ready." 'OK'
                }
                $ActiveLlamaImage = $CandidateImage
                $ActiveImageCompatibility = $Compatibility
                Write-BootstrapManagedRuntimeState `
                    -Path $ManagedRuntimeStatePath `
                    -ImageRef $CandidateImage `
                    -ImageId $Compatibility.ImageId `
                    -ManagedRuntimes $ManagedRuntimes
                $Provisioned = $true
                break
            }
            catch {
                if (-not $HasFallback) { throw }
                Write-BootstrapMessage (
                    "The preferred llama.cpp image could not complete managed runtime preparation. " +
                    "Retrying with the compatibility image: " +
                    $ImageCandidates[$ImageIndex + 1]
                ) 'WARN'
            }
        }
        if (-not $Provisioned) {
            throw 'No supported llama.cpp CUDA image completed managed runtime provisioning.'
        }

        if ($null -eq $ActiveImageCompatibility) {
            throw 'The selected llama.cpp image compatibility record is unavailable.'
        }
        Invoke-BootstrapCommand -FilePath $VenvPython -Arguments @(
            '-B', '-s', (Join-Path $Root 'scripts\windows_install_state.py'),
            'write', '--runtime', $Runtime, '--tier', $ProvisioningTier,
            '--profile', $ProvisioningTier, '--requirements', ([string]$RuntimeConfig.requirements),
            '--image-ref', $ActiveLlamaImage,
            '--image-id', ([string]$ActiveImageCompatibility.ImageId),
            '--required-cuda', ([string]$ActiveImageCompatibility.RequiredCudaVersion),
            '--managed-state', $ManagedRuntimeStatePath
        ) -WorkingDirectory $Root -ShowOutput
    }

    Write-BootstrapMessage (
        "Setup completed for tier '$ProvisioningTier'. Start the application with " +
        $(if ($Runtime -eq 'cuda13') { 'run_comic_cuda13.bat' } else { 'run_comic.bat' }) + '.'
    ) 'OK'
}
catch {
    if ($VenvBackup -and (Test-Path -LiteralPath $VenvBackup)) {
        Remove-Item -LiteralPath $VenvRoot -Recurse -Force -ErrorAction SilentlyContinue
        Move-Item -LiteralPath $VenvBackup -Destination $VenvRoot -ErrorAction SilentlyContinue
    }
    Write-BootstrapMessage "BOOTSTRAP_FAILED [$Runtime]: $($_.Exception.Message)" 'ERROR'
    if ($TranscriptStarted) { Write-BootstrapMessage "Log: $LogPath" 'ERROR' }
    if ($TranscriptStarted) { Write-BootstrapMessage "Command details: $DetailLogPath" 'ERROR' }
    Write-BootstrapMessage 'Fix the reported prerequisite if necessary, then run the same BAT again. Verified files and partial downloads will be reused.' 'ERROR'
    exit 1
}
finally {
    if ($null -ne $Lock) { $Lock.Dispose() }
    if ($TranscriptStarted) { Stop-Transcript | Out-Null }
}
