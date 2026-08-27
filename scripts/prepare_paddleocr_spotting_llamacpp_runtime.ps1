[CmdletBinding()]
param(
    # Prepare seeds the volume from source files and seals it. Verify is a
    # read-only check. Reseal leaves the volume contents alone, re-runs the real
    # smoke against the current llama.cpp image, and rewrites the ready manifest.
    # Auto picks Prepare or Reseal from the volume state; the app's self-repair
    # path uses Auto.
    [ValidateSet('Prepare', 'Verify', 'Reseal', 'Auto')]
    [string]$Mode = 'Prepare',

    # Leave empty to fall back to the repository's testmodel/PaddleOCR-VL-1.6-GGUF.
    [string]$ModelDirectory = '',

    # Fetch the registered official source only when no verified local copy exists.
    [switch]$AllowDownload,

    # Where downloaded sources land. Empty means the repository's testmodel/.
    [string]$DownloadDirectory = '',

    [string]$VolumeName = (
        'comic-translate-paddleocr-vl-spotting-llamacpp-models-v2'
    ),

    [ValidateSet(
        'ghcr.io/ggml-org/llama.cpp:server-cuda',
        'ghcr.io/ggml-org/llama.cpp:server-cuda13'
    )]
    [string]$ImageRef = $(
        if ($env:PADDLEOCR_SPOTTING_LLAMA_CPP_IMAGE) {
            $env:PADDLEOCR_SPOTTING_LLAMA_CPP_IMAGE
        }
        elseif ($env:LLAMA_CPP_IMAGE) { $env:LLAMA_CPP_IMAGE }
        else { 'ghcr.io/ggml-org/llama.cpp:server-cuda13' }
    ),

    [ValidateRange(1024, 65535)]
    [int]$SmokePort = 18085,

    [ValidateRange(30, 600)]
    [int]$SmokeTimeoutSec = 240,

    [ValidateRange(0, 65536)]
    [int]$MaximumBackgroundGpuMiB = 2048,

    [int64]$MinimumFreeBytes = 0,

    [switch]$SkipFreeSpaceCheck
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)

Import-Module (Join-Path $PSScriptRoot 'lib\ManagedRuntimeModelSource.psm1') -Force

$PreparationVersion = 2
$ManifestSchemaVersion = 1
$ReadyManifestName = (
    '.comic-translate-paddleocr-vl-spotting-llamacpp-ready-v2.json'
)
$RuntimeName = 'PaddleOCR-VL-Spotting-llama.cpp'
# CUDA 13 태그가 기본이지만, CUDA 12 태그로 준비한 볼륨도 그대로 인정한다.
$SupportedImageRefs = @(
    'ghcr.io/ggml-org/llama.cpp:server-cuda13',
    'ghcr.io/ggml-org/llama.cpp:server-cuda'
)
$ManagedContainerName = 'paddleocr-spotting-llamacpp'
$ModelAlias = 'PaddleOCR-VL-1.6-Spotting'
$SpottingPrompt = 'Spotting:'
$CropImageMaxPixels = 1003520
$SpottingImageMaxPixels = 1605632
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepositoryRoot = Split-Path -Parent $ScriptRoot
$DeriveScript = Join-Path $ScriptRoot 'derive_paddleocr_spotting_mmproj.py'
$PythonExecutable = Join-Path $RepositoryRoot '.venv-win\Scripts\python.exe'

$ModelSpecs = @(
    [pscustomobject][ordered]@{
        Name = 'PaddleOCR-VL-1.6-Spotting-GGUF.gguf'
        SourceNames = @(
            'PaddleOCR-VL-1.6-Spotting-GGUF.gguf',
            'PaddleOCR-VL-1.6-GGUF.gguf'
        )
        Bytes = [int64]935769056
        Sha256 = (
            'f3ae46ec885050acf4b3d31944431e1fd90d50664fb09126af4a3c050ba14ee8'
        )
        Role = 'vlm'
        DerivedFromSha256 = ''
        # 공식 PaddleOCR-VL 1.6 GGUF. Spotting 대상 GGUF 는 crop VLM 과 바이트가
        # 같으므로 같은 원본을 쓴다.
        DownloadUrl = (
            'https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6-GGUF/' +
            'resolve/main/PaddleOCR-VL-1.6-GGUF.gguf'
        )
    }
    [pscustomobject][ordered]@{
        Name = 'PaddleOCR-VL-1.6-Spotting-mmproj.gguf'
        SourceNames = @(
            'PaddleOCR-VL-1.6-GGUF-mmproj.gguf'
        )
        Bytes = [int64]881770560
        Sha256 = (
            '8e011479092c5e82c8c1c2d85d52b9ac48df12183c5c7bc3190190732259db09'
        )
        Role = 'vision-projector'
        DerivedFromSha256 = (
            '204d757d7610d9b3faab10d506d69e5b244e32bf765e2bab2d0167e65e0a058a'
        )
        # 이 항목은 파생물이다. 원본은 공식 crop projector 이고, 위
        # DerivedFromSha256 이 그 원본의 해시다.
        SourceDownloadUrl = (
            'https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6-GGUF/' +
            'resolve/main/PaddleOCR-VL-1.6-GGUF-mmproj.gguf'
        )
    }
)

if ($VolumeName -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]*$') {
    throw "Invalid Docker volume name: $VolumeName"
}
if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
    throw (
        'Supported Windows Python environment was not found: ' +
        $PythonExecutable
    )
}
if (-not (Test-Path -LiteralPath $DeriveScript -PathType Leaf)) {
    throw "Spotting projector derivation tool was not found: $DeriveScript"
}

$DockerCommand = Get-Command 'docker.exe' -ErrorAction SilentlyContinue
if ($null -eq $DockerCommand) {
    $DockerCommand = Get-Command 'docker' -ErrorAction Stop
}
$DockerExecutable = $DockerCommand.Source

function ConvertTo-NativeArgument {
    param([AllowEmptyString()][string]$Argument)

    if (
        -not [string]::IsNullOrEmpty($Argument) -and
        $Argument -notmatch '[\s"]'
    ) {
        return $Argument
    }
    $Builder = [System.Text.StringBuilder]::new()
    [void]$Builder.Append([char]34)
    $BackslashCount = 0
    foreach ($Character in $Argument.ToCharArray()) {
        if ($Character -eq [char]92) {
            $BackslashCount += 1
            continue
        }
        if ($Character -eq [char]34) {
            [void]$Builder.Append(
                [char]92,
                (($BackslashCount * 2) + 1)
            )
            [void]$Builder.Append([char]34)
            $BackslashCount = 0
            continue
        }
        if ($BackslashCount -gt 0) {
            [void]$Builder.Append([char]92, $BackslashCount)
            $BackslashCount = 0
        }
        [void]$Builder.Append($Character)
    }
    if ($BackslashCount -gt 0) {
        [void]$Builder.Append([char]92, ($BackslashCount * 2))
    }
    [void]$Builder.Append([char]34)
    return $Builder.ToString()
}

function Invoke-NativeResult {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    $StartInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $StartInfo.FileName = $Executable
    $StartInfo.Arguments = (
        $Arguments |
            ForEach-Object { ConvertTo-NativeArgument -Argument $_ }
    ) -join ' '
    $StartInfo.UseShellExecute = $false
    $StartInfo.CreateNoWindow = $true
    $StartInfo.RedirectStandardOutput = $true
    $StartInfo.RedirectStandardError = $true
    $Process = [System.Diagnostics.Process]::new()
    $Process.StartInfo = $StartInfo
    try {
        if (-not $Process.Start()) {
            throw "Unable to start: $Executable"
        }
        $OutputTask = $Process.StandardOutput.ReadToEndAsync()
        $ErrorTask = $Process.StandardError.ReadToEndAsync()
        $Process.WaitForExit()
        return [pscustomobject]@{
            ExitCode = [int]$Process.ExitCode
            Output = (
                @(
                    $OutputTask.GetAwaiter().GetResult().TrimEnd(),
                    $ErrorTask.GetAwaiter().GetResult().TrimEnd()
                ) |
                    Where-Object {
                        -not [string]::IsNullOrWhiteSpace($_)
                    }
            ) -join "`n"
        }
    }
    finally {
        $Process.Dispose()
    }
}

function Invoke-DockerResult {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    # PowerShell here-strings carry CRLF line endings. The container's /bin/sh is
    # dash, which reads a trailing CR as part of the token and dies on the first
    # line with "set: Illegal option -". A CR is never meaningful in a docker
    # argument, so normalize every argument in one place.
    $NormalizedArguments = @(
        $Arguments | ForEach-Object { [string]$_ -replace "`r`n", "`n" }
    )
    return Invoke-NativeResult `
        -Executable $DockerExecutable `
        -Arguments $NormalizedArguments
}

function Invoke-Docker {
    param(
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [switch]$ShowOutput
    )
    $Result = Invoke-DockerResult -Arguments $Arguments
    if ($ShowOutput -and -not [string]::IsNullOrWhiteSpace($Result.Output)) {
        Write-Host $Result.Output
    }
    if ($Result.ExitCode -ne 0) {
        throw (
            "Docker command failed (exit={0}): docker {1}`n{2}" -f
            $Result.ExitCode,
            ($Arguments -join ' '),
            $Result.Output
        )
    }
    return $Result.Output.Trim()
}

function Get-PinnedImageId {
    $Inspect = Invoke-DockerResult -Arguments @(
        'image', 'inspect', '--format', '{{.Id}}', $ImageRef
    )
    if (
        $Inspect.ExitCode -ne 0 -or
        [string]::IsNullOrWhiteSpace($Inspect.Output)
    ) {
        Write-Host "Pulling the pinned llama.cpp image once: $ImageRef"
        Invoke-Docker -Arguments @('pull', $ImageRef) -ShowOutput |
            Out-Null
        $Inspect = Invoke-DockerResult -Arguments @(
            'image', 'inspect', '--format', '{{.Id}}', $ImageRef
        )
    }
    if (
        $Inspect.ExitCode -ne 0 -or
        [string]::IsNullOrWhiteSpace($Inspect.Output)
    ) {
        throw "Unable to inspect the pinned llama.cpp image: $ImageRef"
    }
    return $Inspect.Output.Trim()
}

function Assert-ManagedContainerStopped {
    $Inspect = Invoke-DockerResult -Arguments @(
        'inspect', '--format', '{{.State.Running}}', $ManagedContainerName
    )
    if ($Inspect.ExitCode -eq 0 -and $Inspect.Output.Trim() -eq 'true') {
        throw (
            "$ManagedContainerName is running. Stop the app normally, " +
            'then prepare the Spotting model volume again.'
        )
    }
}

function Assert-BackgroundGpuUsage {
    if ($MaximumBackgroundGpuMiB -le 0) {
        return
    }
    $NvidiaSmi = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
    if ($null -eq $NvidiaSmi) {
        $NvidiaSmi = Get-Command 'nvidia-smi' -ErrorAction SilentlyContinue
    }
    if ($null -eq $NvidiaSmi) {
        throw 'nvidia-smi is required for the CUDA smoke preflight.'
    }
    $Result = Invoke-NativeResult `
        -Executable $NvidiaSmi.Source `
        -Arguments @(
            '--query-gpu=memory.used',
            '--format=csv,noheader,nounits'
        )
    if ($Result.ExitCode -ne 0) {
        throw "Unable to query background GPU usage.`n$($Result.Output)"
    }
    $UsedMiB = 0
    foreach ($Line in $Result.Output.Split([Environment]::NewLine)) {
        $Value = 0
        if ([int]::TryParse($Line.Trim(), [ref]$Value)) {
            $UsedMiB = [Math]::Max($UsedMiB, $Value)
        }
    }
    if ($UsedMiB -gt $MaximumBackgroundGpuMiB) {
        throw (
            "Background GPU usage is above the agreed preflight limit: " +
            "$UsedMiB MiB > $MaximumBackgroundGpuMiB MiB."
        )
    }
}

function Assert-VolumeLabels {
    $LabelsText = Invoke-Docker -Arguments @(
        'volume', 'inspect', '--format', '{{json .Labels}}', $VolumeName
    )
    $Labels = $LabelsText | ConvertFrom-Json
    if (
        [string]$Labels.'comic-translate.runtime' -ne $RuntimeName -or
        [int]$Labels.'comic-translate.preparation-version' -ne
            $PreparationVersion
    ) {
        throw (
            "PaddleOCR-VL Spotting volume labels do not match: " +
            "$VolumeName. Use a new versioned volume name."
        )
    }
}

function Get-VolumeFileHash {
    param(
        [Parameter(Mandatory = $true)][string]$FileName,
        [switch]$AllowMissing
    )
    $Shell = if ($AllowMissing) {
        'if test -f "/models/$MODEL_FILE"; then sha256sum "/models/$MODEL_FILE" | cut -d " " -f 1; fi'
    }
    else {
        'set -eu; test -f "/models/$MODEL_FILE"; sha256sum "/models/$MODEL_FILE" | cut -d " " -f 1'
    }
    $Result = Invoke-DockerResult -Arguments @(
        'run', '--rm', '--pull', 'never',
        '-e', "MODEL_FILE=$FileName",
        '--mount', "type=volume,source=$VolumeName,target=/models,readonly",
        '--entrypoint', '/bin/sh',
        $ImageRef,
        '-ec', $Shell
    )
    if ($Result.ExitCode -ne 0) {
        if ($AllowMissing) {
            return ''
        }
        throw "Volume SHA-256 verification failed: $FileName`n$($Result.Output)"
    }
    return $Result.Output.Trim().ToLowerInvariant()
}

function Read-ReadyManifest {
    $ManifestText = Invoke-Docker -Arguments @(
        'run', '--rm', '--pull', 'never',
        '-e', "READY_MANIFEST=$ReadyManifestName",
        '--mount', "type=volume,source=$VolumeName,target=/models,readonly",
        '--entrypoint', '/bin/sh',
        $ImageRef,
        '-ec', 'set -eu; cat "/models/$READY_MANIFEST"'
    )
    return $ManifestText | ConvertFrom-Json
}

function Assert-ManifestContract {
    param([Parameter(Mandatory = $true)]$Manifest)

    if (
        [int]$Manifest.schema_version -ne $ManifestSchemaVersion -or
        [int]$Manifest.preparation_version -ne $PreparationVersion -or
        [string]$Manifest.runtime -ne $RuntimeName -or
        [string]$Manifest.volume_name -ne $VolumeName -or
        $SupportedImageRefs -notcontains [string]$Manifest.source_image_ref -or
        [string]$Manifest.source_image_id -ne $ImageId -or
        $Manifest.ready -ne $true -or
        $Manifest.smoke_test.passed -ne $true -or
        [string]$Manifest.smoke_test.device -ne 'CUDA' -or
        [string]$Manifest.smoke_test.model_alias -ne $ModelAlias -or
        [string]$Manifest.spotting_contract.prompt -ne $SpottingPrompt -or
        $Manifest.spotting_contract.special_tokens -ne $true -or
        [int]$Manifest.spotting_contract.'clip.vision.image_max_pixels' -ne
            $SpottingImageMaxPixels
    ) {
        if ([string]$Manifest.source_image_id -ne $ImageId) {
            # When upstream refreshes a supported tag only the image identity
            # drifts. Reseal recovers that without the original source files.
            throw (
                'Ready manifest header does not match the official ' +
                'PaddleOCR-VL Spotting contract: the llama.cpp image identity ' +
                "drifted (manifest=$($Manifest.source_image_id), " +
                "actual=$ImageId). Run this script with -Mode Reseal to " +
                're-smoke and re-seal.'
            )
        }
        throw (
            'Ready manifest header does not match the official ' +
            'PaddleOCR-VL Spotting contract.'
        )
    }
    if (@($Manifest.files).Count -ne $ModelSpecs.Count) {
        throw 'Ready manifest file registry is incomplete.'
    }
    foreach ($Spec in $ModelSpecs) {
        $Entry = @(
            $Manifest.files |
                Where-Object { $_.name -eq $Spec.Name }
        )
        if (
            $Entry.Count -ne 1 -or
            [int64]$Entry[0].bytes -ne $Spec.Bytes -or
            ([string]$Entry[0].sha256).ToLowerInvariant() -ne
                $Spec.Sha256 -or
            [string]$Entry[0].role -ne $Spec.Role
        ) {
            throw "Ready manifest file contract mismatch: $($Spec.Name)"
        }
        if (
            -not [string]::IsNullOrWhiteSpace($Spec.DerivedFromSha256) -and
            ([string]$Entry[0].derived_from_sha256).ToLowerInvariant() -ne
                $Spec.DerivedFromSha256
        ) {
            throw (
                "Ready manifest projector provenance mismatch: " +
                $Spec.Name
            )
        }
    }
}

$ImageId = Get-PinnedImageId

function Get-VolumeFileSize {
    param([Parameter(Mandatory = $true)][string]$FileName)

    $Result = Invoke-DockerResult -Arguments @(
        'run', '--rm', '--pull', 'never',
        '-e', "MODEL_FILE=$FileName",
        '--mount', "type=volume,source=$VolumeName,target=/models,readonly",
        '--entrypoint', '/bin/sh',
        $ImageRef,
        '-ec', 'if test -f "/models/$MODEL_FILE"; then stat -c %s "/models/$MODEL_FILE"; fi'
    )
    if ($Result.ExitCode -ne 0) {
        return [int64]-1
    }
    $Text = $Result.Output.Trim()
    if ([string]::IsNullOrWhiteSpace($Text)) {
        return [int64]-1
    }
    return [int64]$Text
}

function Test-VolumeHoldsEveryModel {
    <#
    .SYNOPSIS
    Report whether the volume already holds every contracted file, by size.

    .DESCRIPTION
    Used only to let Auto pick between Prepare and Reseal. Size alone keeps this
    from hashing large GGUFs twice. The authoritative judgement is the SHA-256
    pass Reseal runs before the smoke, and a mismatch there still fails.
    #>

    if ((Invoke-DockerResult -Arguments @('volume', 'inspect', $VolumeName)).ExitCode -ne 0) {
        return $false
    }
    foreach ($Spec in $ModelSpecs) {
        if ((Get-VolumeFileSize -FileName $Spec.Name) -ne $Spec.Bytes) {
            return $false
        }
    }
    return $true
}

if ($Mode -eq 'Auto') {
    if (Test-VolumeHoldsEveryModel) {
        Write-Host 'Auto mode: the volume already holds every model; resealing.'
        $Mode = 'Reseal'
    }
    else {
        Write-Host 'Auto mode: the volume is missing a model; preparing.'
        $Mode = 'Prepare'
    }
}
$IsReseal = $Mode -eq 'Reseal'

if ($Mode -eq 'Verify') {
    if (
        (
            Invoke-DockerResult -Arguments @(
                'volume', 'inspect', $VolumeName
            )
        ).ExitCode -ne 0
    ) {
        throw "PaddleOCR-VL Spotting volume does not exist: $VolumeName"
    }
    Assert-VolumeLabels
    $Manifest = Read-ReadyManifest
    Assert-ManifestContract -Manifest $Manifest
    $VerifiedFiles = @()
    foreach ($Spec in $ModelSpecs) {
        $ActualHash = Get-VolumeFileHash -FileName $Spec.Name
        if ($ActualHash -ne $Spec.Sha256) {
            throw (
                "Volume SHA-256 mismatch: {0}, expected={1}, actual={2}" -f
                $Spec.Name,
                $Spec.Sha256,
                $ActualHash
            )
        }
        $VerifiedFiles += [pscustomobject][ordered]@{
            name = $Spec.Name
            bytes = $Spec.Bytes
            sha256 = $ActualHash
            role = $Spec.Role
            verified = $true
        }
    }
    [ordered]@{
        mode = 'Verify'
        verified = $true
        volume_name = $VolumeName
        image_ref = $ImageRef
        image_id = $ImageId
        official_spotting_max_pixels = $SpottingImageMaxPixels
        files = @($VerifiedFiles)
    } | ConvertTo-Json -Depth 10
    return
}

if (-not $IsReseal -and [string]::IsNullOrWhiteSpace($ModelDirectory)) {
    # Fall back to the repository's gitignored model directory before asking the
    # caller for a path they already have on disk.
    $DefaultRoot = Get-ManagedRuntimeDefaultSearchDirectory
    if (-not [string]::IsNullOrWhiteSpace($DefaultRoot)) {
        $DefaultSpottingDirectory = Join-Path $DefaultRoot 'PaddleOCR-VL-1.6-GGUF'
        if (Test-Path -LiteralPath $DefaultSpottingDirectory -PathType Container) {
            Write-Host "Using the repository model directory: $DefaultSpottingDirectory"
            $ModelDirectory = $DefaultSpottingDirectory
        }
    }
}
if (-not $IsReseal -and [string]::IsNullOrWhiteSpace($ModelDirectory)) {
    throw 'Prepare mode requires -ModelDirectory.'
}
Assert-ManagedContainerStopped
Assert-BackgroundGpuUsage
# Reseal leaves volume contents alone and copies nothing, so it needs neither a
# source directory nor room for a copy.
if (-not $IsReseal -and -not $SkipFreeSpaceCheck -and $MinimumFreeBytes -gt 0) {
    $Drive = Get-PSDrive -Name 'C' -ErrorAction Stop
    if ([int64]$Drive.Free -lt $MinimumFreeBytes) {
        throw (
            "Insufficient free C: space. required={0:N2} GiB, " +
            "actual={1:N2} GiB" -f
            ($MinimumFreeBytes / 1GB),
            ([int64]$Drive.Free / 1GB)
        )
    }
}

$TemporaryRoot = Join-Path (
    [System.IO.Path]::GetTempPath()
) "comic-translate-paddle-spotting-prepare-$PID"
New-Item -ItemType Directory -Force -Path $TemporaryRoot | Out-Null

try {
    $PreparedSources = @()
    if (-not $IsReseal) {
        # The target GGUF is byte-identical to the official crop VLM, so an
        # already-renamed Spotting copy and the upstream name both satisfy the
        # contract. Try the explicit names first, then fall back to the shared
        # resolver, which can also fetch the registered official source.
        $ResolvedDirectory = ''
        if (-not [string]::IsNullOrWhiteSpace($ModelDirectory)) {
            $ResolvedDirectory = (Resolve-Path -LiteralPath $ModelDirectory).Path
        }
        $TargetSpec = $ModelSpecs[0]
        $TargetSourcePath = $null
        if (-not [string]::IsNullOrWhiteSpace($ResolvedDirectory)) {
            foreach ($Candidate in $TargetSpec.SourceNames) {
                $CandidatePath = Join-Path $ResolvedDirectory $Candidate
                if (Test-Path -LiteralPath $CandidatePath -PathType Leaf) {
                    $TargetSourcePath = $CandidatePath
                    break
                }
            }
        }
        if ($null -eq $TargetSourcePath) {
            $TargetSourcePath = (Resolve-ManagedRuntimeModelSource `
                -FileName $TargetSpec.SourceNames[-1] `
                -Bytes $TargetSpec.Bytes `
                -Sha256 $TargetSpec.Sha256 `
                -RequestedPath $ModelDirectory `
                -DownloadUrl ([string]$TargetSpec.DownloadUrl) `
                -DownloadDirectory $DownloadDirectory `
                -AllowDownload:$AllowDownload `
                -SkipFreeSpaceCheck:$SkipFreeSpaceCheck).Path
        }
        $TargetHash = (
            Get-FileHash -LiteralPath $TargetSourcePath -Algorithm SHA256
        ).Hash.ToLowerInvariant()
        if (
            (Get-Item -LiteralPath $TargetSourcePath).Length -ne
                $TargetSpec.Bytes -or
            $TargetHash -ne $TargetSpec.Sha256
        ) {
            throw (
                "Target GGUF contract mismatch: $TargetSourcePath, " +
                "sha256=$TargetHash"
            )
        }

        # The Spotting projector is derived locally from the official crop
        # projector, so the source is pinned by DerivedFromSha256 rather than by
        # the derived file's own hash.
        $ProjectorSpec = $ModelSpecs[1]
        $CropProjectorPath = $null
        if (-not [string]::IsNullOrWhiteSpace($ResolvedDirectory)) {
            $CropCandidate = Join-Path (
                $ResolvedDirectory
            ) $ProjectorSpec.SourceNames[0]
            if (Test-Path -LiteralPath $CropCandidate -PathType Leaf) {
                $CropProjectorPath = $CropCandidate
            }
        }
        if ($null -eq $CropProjectorPath) {
            $CropProjectorPath = (Resolve-ManagedRuntimeModelSource `
                -FileName $ProjectorSpec.SourceNames[0] `
                -Bytes $ProjectorSpec.Bytes `
                -Sha256 $ProjectorSpec.DerivedFromSha256 `
                -RequestedPath $ModelDirectory `
                -DownloadUrl ([string]$ProjectorSpec.SourceDownloadUrl) `
                -DownloadDirectory $DownloadDirectory `
                -AllowDownload:$AllowDownload `
                -SkipFreeSpaceCheck:$SkipFreeSpaceCheck).Path
        }
        $DerivedProjectorPath = Join-Path $TemporaryRoot $ProjectorSpec.Name
        $Derive = Invoke-NativeResult `
            -Executable $PythonExecutable `
            -Arguments @(
                $DeriveScript,
                '--source', $CropProjectorPath,
                '--output', $DerivedProjectorPath
            )
        if ($Derive.ExitCode -ne 0) {
            throw "Spotting projector derivation failed.`n$($Derive.Output)"
        }

        $PreparedSources = @(
            [pscustomobject]@{
                Spec = $TargetSpec
                Path = $TargetSourcePath
            },
            [pscustomobject]@{
                Spec = $ProjectorSpec
                Path = $DerivedProjectorPath
            }
        )
        foreach ($Source in $PreparedSources) {
            $ActualHash = (
                Get-FileHash -LiteralPath $Source.Path -Algorithm SHA256
            ).Hash.ToLowerInvariant()
            $ActualBytes = (Get-Item -LiteralPath $Source.Path).Length
            if (
                $ActualBytes -ne $Source.Spec.Bytes -or
                $ActualHash -ne $Source.Spec.Sha256
            ) {
                throw (
                    "Prepared source contract mismatch: {0}, bytes={1}, " +
                    "sha256={2}" -f
                    $Source.Spec.Name,
                    $ActualBytes,
                    $ActualHash
                )
            }
        }
    }

    if ($IsReseal) {
        if (
            (
                Invoke-DockerResult -Arguments @(
                    'volume', 'inspect', $VolumeName
                )
            ).ExitCode -ne 0
        ) {
            throw (
                "The volume to reseal does not exist: $VolumeName. " +
                'Run this script in Prepare or Auto mode first.'
            )
        }
    }
    else {
        Invoke-Docker -Arguments @(
            'volume', 'create',
            '--label', "comic-translate.runtime=$RuntimeName",
            '--label',
            "comic-translate.preparation-version=$PreparationVersion",
            $VolumeName
        ) | Out-Null
    }
    Assert-VolumeLabels
    Invoke-Docker -Arguments @(
        'run', '--rm', '--pull', 'never',
        '-e', "READY_MANIFEST=$ReadyManifestName",
        '--mount', "type=volume,source=$VolumeName,target=/models",
        '--entrypoint', '/bin/sh',
        $ImageRef,
        '-ec',
        'set -eu; rm -f "/models/$READY_MANIFEST" "/models/.${READY_MANIFEST}.partial"'
    ) | Out-Null

    foreach ($Source in $PreparedSources) {
        $Spec = $Source.Spec
        $ExistingHash = Get-VolumeFileHash `
            -FileName $Spec.Name `
            -AllowMissing
        if ($ExistingHash -eq $Spec.Sha256) {
            Write-Host "Reusing verified volume file: $($Spec.Name)"
            continue
        }
        $SourceDirectory = Split-Path -Parent $Source.Path
        $SourceFileName = Split-Path -Leaf $Source.Path
        Invoke-Docker -Arguments @(
            'run', '--rm', '--pull', 'never',
            '-e', "SOURCE_FILE=$SourceFileName",
            '-e', "TARGET_FILE=$($Spec.Name)",
            '-e', "EXPECTED_BYTES=$($Spec.Bytes)",
            '-e', "EXPECTED_SHA256=$($Spec.Sha256)",
            '--mount',
            "type=bind,source=$SourceDirectory,target=/import,readonly",
            '--mount',
            "type=volume,source=$VolumeName,target=/models",
            '--entrypoint', '/bin/sh',
            $ImageRef,
            '-ec',
            @'
set -eu
source="/import/$SOURCE_FILE"
target="/models/$TARGET_FILE"
partial="/models/.${TARGET_FILE}.partial"
test -f "$source"
test "$(stat -c %s "$source")" = "$EXPECTED_BYTES"
rm -f "$partial"
cp "$source" "$partial"
sync "$partial"
test "$(stat -c %s "$partial")" = "$EXPECTED_BYTES"
actual_sha256="$(sha256sum "$partial" | cut -d " " -f 1)"
test "$actual_sha256" = "$EXPECTED_SHA256"
mv -f "$partial" "$target"
'@
        ) | Out-Null
    }

    $VerifiedFiles = @()
    foreach ($Spec in $ModelSpecs) {
        $ActualHash = Get-VolumeFileHash -FileName $Spec.Name
        if ($ActualHash -ne $Spec.Sha256) {
            $Recovery = if ($IsReseal) {
                ' Reseal never copies model data, so run -Mode Prepare instead.'
            }
            else {
                ''
            }
            throw ("Volume file SHA-256 mismatch: $($Spec.Name)." + $Recovery)
        }
        $Entry = [ordered]@{
            name = $Spec.Name
            bytes = $Spec.Bytes
            sha256 = $ActualHash
            role = $Spec.Role
        }
        if (-not [string]::IsNullOrWhiteSpace($Spec.DerivedFromSha256)) {
            $Entry.derived_from_sha256 = $Spec.DerivedFromSha256
        }
        $VerifiedFiles += [pscustomobject]$Entry
    }

    Add-Type -AssemblyName System.Drawing
    $SmokeImagePath = Join-Path $TemporaryRoot 'spotting-smoke.png'
    # A single Latin word can legitimately use PaddleOCR-VL's plain OCR
    # response even with the Spotting prompt. Use a deterministic,
    # multi-region Japanese page so the smoke proves that llama.cpp preserves
    # the native LOC tokens required by the official Spotting contract.
    $Bitmap = [System.Drawing.Bitmap]::new(1200, 1600)
    try {
        $Graphics = [System.Drawing.Graphics]::FromImage($Bitmap)
        try {
            $Graphics.Clear([System.Drawing.Color]::White)
            $Graphics.SmoothingMode = (
                [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
            )
            $LargeFont = [System.Drawing.Font]::new(
                'Yu Gothic',
                72,
                [System.Drawing.FontStyle]::Regular,
                [System.Drawing.GraphicsUnit]::Pixel
            )
            $SmallFont = [System.Drawing.Font]::new(
                'Yu Gothic',
                54,
                [System.Drawing.FontStyle]::Regular,
                [System.Drawing.GraphicsUnit]::Pixel
            )
            $Outline = [System.Drawing.Pen]::new(
                [System.Drawing.Color]::Black,
                6
            )
            # Keep this PowerShell 5.1-compatible file ASCII-only. Windows
            # PowerShell otherwise decodes a UTF-8 file without BOM using the
            # active ANSI code page.
            $Greeting = -join @(
                [char]0x3053, [char]0x3093, [char]0x306B,
                [char]0x3061, [char]0x306F
            )
            $Question = -join @(
                [char]0x5143, [char]0x6C17, [char]0x3067,
                [char]0x3059, [char]0x304B, [char]0xFF1F
            )
            $Today = -join @(
                [char]0x4ECA, [char]0x65E5, [char]0x306F
            )
            $FineWeather = -join @(
                [char]0x3044, [char]0x3044,
                [char]0x5929, [char]0x6C17
            )
            $EpisodeEnd = -join @(
                [char]0x7B2C, '1', '2', [char]0x8A71, ' ',
                [char]0x3064, [char]0x3065, [char]0x304F
            )
            try {
                $Graphics.DrawEllipse($Outline, 80, 80, 460, 420)
                $Graphics.DrawString(
                    $Greeting,
                    $LargeFont,
                    [System.Drawing.Brushes]::Black,
                    150,
                    150
                )
                $Graphics.DrawString(
                    $Question,
                    $SmallFont,
                    [System.Drawing.Brushes]::Black,
                    170,
                    270
                )
                $Graphics.DrawEllipse($Outline, 650, 600, 470, 440)
                $Graphics.DrawString(
                    $Today,
                    $LargeFont,
                    [System.Drawing.Brushes]::Black,
                    720,
                    690
                )
                $Graphics.DrawString(
                    $FineWeather,
                    $LargeFont,
                    [System.Drawing.Brushes]::Black,
                    700,
                    820
                )
                $Graphics.DrawString(
                    $EpisodeEnd,
                    $SmallFont,
                    [System.Drawing.Brushes]::Black,
                    90,
                    1250
                )
            }
            finally {
                $Outline.Dispose()
                $SmallFont.Dispose()
                $LargeFont.Dispose()
            }
        }
        finally {
            $Graphics.Dispose()
        }
        $Bitmap.Save(
            $SmokeImagePath,
            [System.Drawing.Imaging.ImageFormat]::Png
        )
    }
    finally {
        $Bitmap.Dispose()
    }

    $SmokeContainer = "comic-translate-paddle-spotting-smoke-$PID"
    $SmokeResult = $null
    try {
        Invoke-Docker -Arguments @(
            'run', '-d', '--rm',
            '--name', $SmokeContainer,
            '--label',
            'comic-translate.runtime=paddle-spotting-prepare-smoke',
            '--gpus', 'all',
            '-e', 'NVIDIA_VISIBLE_DEVICES=all',
            '-e', 'NVIDIA_DRIVER_CAPABILITIES=compute,utility',
            '-p', "127.0.0.1:${SmokePort}:8080",
            '--mount',
            "type=volume,source=$VolumeName,target=/models,readonly",
            '--entrypoint', '/app/llama-server',
            $ImageRef,
            '-m', "/models/$($ModelSpecs[0].Name)",
            '--mmproj', "/models/$($ModelSpecs[1].Name)",
            '--alias', $ModelAlias,
            '--host', '0.0.0.0',
            '--port', '8080',
            '-c', '4096',
            '-np', '1',
            '-t', '10',
            '-b', '2048',
            '-ub', '512',
            '--n-gpu-layers', 'all',
            '--fit', 'off',
            '--flash-attn', 'on',
            '--temp', '0',
            '--special',
            '--metrics',
            '--sleep-idle-seconds', '5'
        ) | Out-Null

        $Deadline = [DateTime]::UtcNow.AddSeconds($SmokeTimeoutSec)
        $HealthReady = $false
        do {
            try {
                $Health = Invoke-RestMethod `
                    -Uri "http://127.0.0.1:${SmokePort}/health" `
                    -TimeoutSec 3
                if ([string]$Health.status -eq 'ok') {
                    $HealthReady = $true
                    break
                }
            }
            catch {
            }
            Start-Sleep -Seconds 1
        } while ([DateTime]::UtcNow -lt $Deadline)
        if (-not $HealthReady) {
            $Logs = Invoke-DockerResult -Arguments @(
                'logs', '--tail', '160', $SmokeContainer
            )
            throw "PaddleOCR-VL Spotting smoke timed out.`n$($Logs.Output)"
        }

        $ImageBase64 = [Convert]::ToBase64String(
            [System.IO.File]::ReadAllBytes($SmokeImagePath)
        )
        $Request = [ordered]@{
            model = $ModelAlias
            messages = @(
                [ordered]@{
                    role = 'user'
                    content = @(
                        [ordered]@{
                            type = 'text'
                            text = $SpottingPrompt
                        },
                        [ordered]@{
                            type = 'image_url'
                            image_url = [ordered]@{
                                url = "data:image/png;base64,$ImageBase64"
                            }
                        }
                    )
                }
            )
            temperature = 0
            seed = 42
            max_tokens = 512
            repeat_penalty = 1.15
            repeat_last_n = 4096
            stream = $false
        }
        $Response = Invoke-RestMethod `
            -Method Post `
            -Uri (
                "http://127.0.0.1:${SmokePort}/v1/chat/completions"
            ) `
            -ContentType 'application/json' `
            -Body ($Request | ConvertTo-Json -Depth 12 -Compress) `
            -TimeoutSec $SmokeTimeoutSec
        $Content = [string]$Response.choices[0].message.content
        $FinishReason = [string]$Response.choices[0].finish_reason
        $LocationCount = (
            [regex]::Matches($Content, '<\|LOC_\d{1,4}\|>')
        ).Count
        if (
            $FinishReason -eq 'length' -or
            $LocationCount -lt 8 -or
            $Content -notmatch '\S'
        ) {
            throw (
                'PaddleOCR-VL Spotting response smoke failed: ' +
                "finish=$FinishReason, location_tokens=$LocationCount, " +
                "content=$Content"
            )
        }
        $SmokeResult = [ordered]@{
            passed = $true
            device = 'CUDA'
            health_status = 'ok'
            model_alias = $ModelAlias
            prompt = $SpottingPrompt
            special_tokens = $true
            location_token_count = $LocationCount
            finish_reason = $FinishReason
        }
    }
    finally {
        if (
            (
                Invoke-DockerResult -Arguments @(
                    'inspect', '--format', '{{.Name}}', $SmokeContainer
                )
            ).ExitCode -eq 0
        ) {
            Invoke-Docker -Arguments @(
                'stop', '--timeout', '10', $SmokeContainer
            ) | Out-Null
        }
    }

    $Manifest = [ordered]@{
        schema_version = $ManifestSchemaVersion
        runtime = $RuntimeName
        preparation_version = $PreparationVersion
        volume_name = $VolumeName
        ready = $true
        source_image_ref = $ImageRef
        source_image_id = $ImageId
        model_alias = $ModelAlias
        spotting_contract = [ordered]@{
            prompt = $SpottingPrompt
            special_tokens = $true
            'clip.vision.image_max_pixels' = $SpottingImageMaxPixels
            crop_projector_original_max_pixels = $CropImageMaxPixels
        }
        files = @($VerifiedFiles)
        smoke_test = $SmokeResult
    }
    $TemporaryManifest = Join-Path $TemporaryRoot $ReadyManifestName
    $Manifest |
        ConvertTo-Json -Depth 12 |
        Set-Content -LiteralPath $TemporaryManifest -Encoding UTF8
    Invoke-Docker -Arguments @(
        'run', '--rm', '--pull', 'never',
        '-e', "READY_MANIFEST=$ReadyManifestName",
        '--mount',
        "type=bind,source=$TemporaryRoot,target=/import,readonly",
        '--mount', "type=volume,source=$VolumeName,target=/models",
        '--entrypoint', '/bin/sh',
        $ImageRef,
        '-ec',
        @'
set -eu
source="/import/$READY_MANIFEST"
partial="/models/.${READY_MANIFEST}.partial"
target="/models/$READY_MANIFEST"
test -f "$source"
rm -f "$partial"
cp "$source" "$partial"
sync "$partial"
mv -f "$partial" "$target"
'@
    ) | Out-Null
    $ManifestSha256 = Invoke-Docker -Arguments @(
        'run', '--rm', '--pull', 'never',
        '-e', "READY_MANIFEST=$ReadyManifestName",
        '--mount',
        "type=volume,source=$VolumeName,target=/models,readonly",
        '--entrypoint', '/bin/sh',
        $ImageRef,
        '-ec',
        'set -eu; sha256sum "/models/$READY_MANIFEST" | cut -d " " -f 1'
    )
    [ordered]@{
        mode = $Mode
        prepared = $true
        resealed = $IsReseal
        volume_name = $VolumeName
        ready_manifest = $ReadyManifestName
        ready_manifest_sha256 = $ManifestSha256
        image_ref = $ImageRef
        image_id = $ImageId
        official_spotting_max_pixels = $SpottingImageMaxPixels
        smoke_test = $SmokeResult
        files = @($VerifiedFiles)
    } | ConvertTo-Json -Depth 12
}
finally {
    Remove-Item `
        -LiteralPath $TemporaryRoot `
        -Recurse `
        -Force `
        -ErrorAction SilentlyContinue
}
