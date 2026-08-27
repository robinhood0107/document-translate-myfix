[CmdletBinding()]
param(
    # Prepare 는 원본을 볼륨에 넣고 봉인한다. Verify 는 읽기 전용 검사다.
    # Reseal 은 볼륨 내용을 그대로 두고 현재 llama.cpp image 로 smoke 를 다시
    # 통과시킨 뒤 ready manifest 만 다시 쓴다. Auto 는 유효한 봉인을 즉시 재사용하고,
    # 봉인이 낡았으면 Reseal, 모델이 빠졌으면 Prepare 를 고른다.
    [ValidateSet('Prepare', 'Verify', 'Reseal', 'Auto')]
    [string]$Mode = 'Prepare',

    # 비우면 저장소의 `testmodel/`, 그다음 다운로드 캐시를 차례로 찾는다.
    [string]$ModelDirectory = '',

    # 로컬에서 검증된 원본을 못 찾았을 때만 등록된 원본을 내려받는다.
    [switch]$AllowDownload,

    # 내려받은 원본을 둘 위치. 비우면 저장소의 `testmodel/`.
    [string]$DownloadDirectory = '',

    [string]$VolumeName = 'comic-translate-mangalmm-models-v2',

    [string]$ImageRef = '',

    [ValidateRange(1024, 65535)]
    [int]$SmokePort = 18085,

    [ValidateRange(30, 600)]
    [int]$SmokeTimeoutSec = 300,

    [int64]$MinimumFreeBytes = 0,

    [switch]$SkipFreeSpaceCheck
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)

Import-Module (Join-Path $PSScriptRoot 'lib\ManagedRuntimeDocker.psm1') -Force -DisableNameChecking
Import-Module (Join-Path $PSScriptRoot 'lib\ManagedRuntimeModelSource.psm1') -Force

$PreparationVersion = 2
$ManifestSchemaVersion = 1
$ReadyManifestName = '.comic-translate-mangalmm-ready-v2.json'
$RuntimeName = 'MangaLMM-llama.cpp'
# CUDA 13 태그가 기본이지만, CUDA 12 태그로 준비한 볼륨도 그대로 인정한다.
$ImagePolicy = Get-ManagedLlamaCppImagePolicy -Runtime 'cuda13'
$ImageRef = Resolve-ManagedLlamaCppImageRef `
    -RequestedImage $ImageRef `
    -RuntimeOverride $env:MANGALMM_LLAMA_CPP_IMAGE
$SupportedImageRefs = $ImagePolicy.Supported
$ManagedContainerName = 'mangalmm-local-server'
$ModelAlias = 'MangaLMM'

$ModelSpecs = @(
    [pscustomobject][ordered]@{
        Name = 'MangaLMM.Q8_0.gguf'
        Bytes = [int64]8098524160
        Sha256 = '55e42d513ee22ab1a301b5fa8f04a2812b69d6b351e7d34efdff2b8d8e8fa01a'
        Role = 'vlm'
        DownloadUrl = (
            'https://huggingface.co/mradermacher/MangaLMM-GGUF/resolve/main/' +
            'MangaLMM.Q8_0.gguf'
        )
    }
    [pscustomobject][ordered]@{
        Name = 'MangaLMM.mmproj-Q8_0.gguf'
        Bytes = [int64]853119744
        Sha256 = '24f43da26996b54bf5764177a954e49b24ec38a53de34d8231764747b0dcd8d7'
        Role = 'vision-projector'
        DownloadUrl = (
            'https://huggingface.co/mradermacher/MangaLMM-GGUF/resolve/main/' +
            'MangaLMM.mmproj-Q8_0.gguf'
        )
    }
)

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
        [string]$Manifest.smoke_test.model_alias -ne $ModelAlias
    ) {
        if ([string]$Manifest.source_image_id -ne $ImageId) {
            # 지원 태그가 업스트림에서 갱신되면 image identity 만 어긋난다. 이때는
            # 원본 파일 없이 Reseal 로 복구된다.
            throw (
                'Ready manifest header does not match the MangaLMM contract: the ' +
                "llama.cpp image identity drifted (manifest=$($Manifest.source_image_id), " +
                "actual=$ImageId). Run this script with -Mode Reseal to re-smoke and re-seal."
            )
        }
        throw 'Ready manifest header does not match the MangaLMM contract.'
    }
    if (@($Manifest.files).Count -ne $ModelSpecs.Count) {
        throw 'Ready manifest file registry is incomplete.'
    }
    foreach ($Spec in $ModelSpecs) {
        $Entry = @($Manifest.files | Where-Object { $_.name -eq $Spec.Name })
        if (
            $Entry.Count -ne 1 -or
            [int64]$Entry[0].bytes -ne $Spec.Bytes -or
            ([string]$Entry[0].sha256).ToLowerInvariant() -ne
                $Spec.Sha256 -or
            [string]$Entry[0].role -ne $Spec.Role
        ) {
            throw "Ready manifest file contract mismatch: $($Spec.Name)"
        }
    }
}

Initialize-ManagedRuntimeDocker `
    -ImageRef $ImageRef `
    -VolumeName $VolumeName `
    -ContainerName $ManagedContainerName `
    -RuntimeName $RuntimeName `
    -PreparationVersion $PreparationVersion `
    -ReadyManifestName $ReadyManifestName `
    -ModelSpecs $ModelSpecs

$ImageId = Get-PinnedImageId

if ($Mode -eq 'Auto') {
    if (Test-VolumeHoldsEveryModel) {
        try {
            Assert-VolumeLabels
            $Manifest = Read-ReadyManifest
            Assert-ManifestContract -Manifest $Manifest
            Write-Host (
                'Auto mode: ready manifest and model sizes match the current image; ' +
                'reusing the sealed volume without a full hash or GPU smoke.'
            )
            [ordered]@{
                mode = 'Reuse'
                reused = $true
                prepared = $true
                volume_name = $VolumeName
                image_ref = $ImageRef
                image_id = $ImageId
            } | ConvertTo-Json -Depth 10
            return
        }
        catch {
            Write-Host (
                'Auto mode: the sealed volume needs repair; resealing. ' +
                $_.Exception.Message
            )
            $Mode = 'Reseal'
        }
    }
    else {
        Write-Host 'Auto mode: the volume is missing a model; preparing.'
        $Mode = 'Prepare'
    }
}

if ($Mode -eq 'Verify') {
    if (
        (Invoke-DockerResult -Arguments @(
            'volume', 'inspect', $VolumeName
        )).ExitCode -ne 0
    ) {
        throw "MangaLMM volume does not exist: $VolumeName"
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
        files = @($VerifiedFiles)
    } | ConvertTo-Json -Depth 10
    return
}

$IsReseal = $Mode -eq 'Reseal'

Assert-ManagedContainerStopped

# Reseal 은 볼륨 안 파일을 그대로 두고 manifest 만 다시 쓴다. 원본을 복사하지
# 않으므로 원본 경로도, 복사할 여유 공간도 필요 없다.
if (-not $IsReseal -and -not $SkipFreeSpaceCheck) {
    $Drive = Get-PSDrive -Name 'C' -ErrorAction Stop
    if ($MinimumFreeBytes -gt 0 -and [int64]$Drive.Free -lt $MinimumFreeBytes) {
        throw (
            "Insufficient free C: space. required={0:N2} GiB, actual={1:N2} GiB" -f
            ($MinimumFreeBytes / 1GB),
            ([int64]$Drive.Free / 1GB)
        )
    }
}

$PreparedSources = @()
if (-not $IsReseal) {
    $VolumeExistsBeforePrepare = (
        Invoke-DockerResult -Arguments @('volume', 'inspect', $VolumeName)
    ).ExitCode -eq 0
    foreach ($Spec in $ModelSpecs) {
        # 볼륨이 이미 계약된 파일을 담고 있으면 원본을 아예 찾지 않는다. 대형
        # GGUF 를 헛되이 내려받거나 해시하지 않기 위해서다.
        if (
            $VolumeExistsBeforePrepare -and
            (Get-VolumeFileHash -FileName $Spec.Name -AllowMissing) -eq $Spec.Sha256
        ) {
            Write-Host "Reusing verified volume file: $($Spec.Name)"
            continue
        }
        $Resolved = Resolve-ManagedRuntimeModelSource `
            -FileName $Spec.Name `
            -Bytes $Spec.Bytes `
            -Sha256 $Spec.Sha256 `
            -RequestedPath $ModelDirectory `
            -DownloadUrl ([string]$Spec.DownloadUrl) `
            -DownloadDirectory $DownloadDirectory `
            -AllowDownload:$AllowDownload `
            -SkipFreeSpaceCheck:$SkipFreeSpaceCheck
        $PreparedSources += [pscustomobject]@{
            Spec = $Spec
            Directory = $Resolved.Directory
            FileName = $Resolved.FileName
            Origin = $Resolved.Origin
        }
    }
}

if ($IsReseal) {
    if ((Invoke-DockerResult -Arguments @('volume', 'inspect', $VolumeName)).ExitCode -ne 0) {
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
        '--label', "comic-translate.preparation-version=$PreparationVersion",
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
    # 볼륨에 이미 있는 파일은 위에서 걸러졌다. 여기 남은 것은 반드시 복사한다.
    $Spec = $Source.Spec
    Write-Host "Copying and verifying: $($Spec.Name)"
    Invoke-Docker -Arguments @(
        'run', '--rm', '--pull', 'never',
        '-e', "SOURCE_FILE=$($Source.FileName)",
        '-e', "TARGET_FILE=$($Spec.Name)",
        '-e', "EXPECTED_BYTES=$($Spec.Bytes)",
        '-e', "EXPECTED_SHA256=$($Spec.Sha256)",
        '--mount', "type=bind,source=$($Source.Directory),target=/import,readonly",
        '--mount', "type=volume,source=$VolumeName,target=/models",
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
            ' Reseal never copies model data, so run this script with -Mode Prepare.'
        }
        else {
            ''
        }
        throw ("Volume file SHA-256 mismatch: $($Spec.Name)." + $Recovery)
    }
    $VerifiedFiles += [pscustomobject][ordered]@{
        name = $Spec.Name
        bytes = $Spec.Bytes
        sha256 = $ActualHash
        role = $Spec.Role
    }
}

$SmokeContainer = "comic-translate-mangalmm-prepare-smoke-$PID"
$SmokeResult = $null
try {
    Write-Host 'Running CUDA model-load smoke from the named volume.'
    Invoke-Docker -Arguments @(
        'run', '-d', '--rm',
        '--name', $SmokeContainer,
        '--label', 'comic-translate.runtime=mangalmm-prepare-smoke',
        '--gpus', 'all',
        '-e', 'NVIDIA_VISIBLE_DEVICES=all',
        '-e', 'NVIDIA_DRIVER_CAPABILITIES=compute,utility',
        '-p', "127.0.0.1:${SmokePort}:8080",
        '--mount', "type=volume,source=$VolumeName,target=/models,readonly",
        '--entrypoint', '/app/llama-server',
        $ImageRef,
        '-m', "/models/$($ModelSpecs[0].Name)",
        '--mmproj', "/models/$($ModelSpecs[1].Name)",
        '--alias', $ModelAlias,
        '--host', '0.0.0.0',
        '--port', '8080',
        '-c', '8192',
        '-np', '1',
        '-t', '12',
        '-b', '2048',
        '-ub', '512',
        '--n-gpu-layers', 'all',
        '--fit', 'off',
        '--flash-attn', 'on',
        '--temp', '0',
        '--metrics',
        '--cache-ram', '0'
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
            'logs', '--tail', '120', $SmokeContainer
        )
        throw "MangaLMM smoke timed out.`n$($Logs.Output)"
    }

    $Models = Invoke-RestMethod `
        -Uri "http://127.0.0.1:${SmokePort}/v1/models" `
        -TimeoutSec 10
    $LoadedIds = @($Models.data | ForEach-Object { [string]$_.id })
    if (@($LoadedIds | Where-Object { $_ -eq $ModelAlias }).Count -ne 1) {
        throw (
            "MangaLMM model alias mismatch: expected=$ModelAlias, " +
            "actual=$($LoadedIds -join ', ')"
        )
    }
    $SmokeResult = [ordered]@{
        passed = $true
        device = 'CUDA'
        health_status = 'ok'
        model_alias = $ModelAlias
        models_match = $true
    }
}
finally {
    if (
        (Invoke-DockerResult -Arguments @(
            'inspect', '--format', '{{.Name}}', $SmokeContainer
        )).ExitCode -eq 0
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
    source_image_digest = $ImageId
    source_image_id = $ImageId
    model_alias = $ModelAlias
    runtime_configuration = [ordered]@{
        context_size = 8192
        parallel = 1
        threads = 12
        batch_size = 2048
        ubatch_size = 512
        gpu_layers = 'all'
        flash_attention = $true
        metrics = $true
        prompt_cache = $false
    }
    files = @($VerifiedFiles)
    smoke_test = $SmokeResult
}

$TemporaryRoot = Join-Path ([System.IO.Path]::GetTempPath()) (
    "comic-translate-mangalmm-manifest-$PID"
)
New-Item -ItemType Directory -Force -Path $TemporaryRoot | Out-Null
try {
    $TemporaryManifest = Join-Path $TemporaryRoot $ReadyManifestName
    $Manifest |
        ConvertTo-Json -Depth 10 |
        Set-Content -LiteralPath $TemporaryManifest -Encoding UTF8
    Invoke-Docker -Arguments @(
        'run', '--rm', '--pull', 'never',
        '-e', "READY_MANIFEST=$ReadyManifestName",
        '--mount', "type=bind,source=$TemporaryRoot,target=/import,readonly",
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
}
finally {
    Remove-Item `
        -LiteralPath $TemporaryRoot `
        -Recurse `
        -Force `
        -ErrorAction SilentlyContinue
}

$ManifestSha256 = Invoke-Docker -Arguments @(
    'run', '--rm', '--pull', 'never',
    '-e', "READY_MANIFEST=$ReadyManifestName",
    '--mount', "type=volume,source=$VolumeName,target=/models,readonly",
    '--entrypoint', '/bin/sh',
    $ImageRef,
    '-ec',
    'set -eu; sha256sum "/models/$READY_MANIFEST" | cut -d " " -f 1'
)

[ordered]@{
    mode = $Mode
    resealed = $IsReseal
    prepared = $true
    volume_name = $VolumeName
    ready_manifest = $ReadyManifestName
    ready_manifest_sha256 = $ManifestSha256
    image_ref = $ImageRef
    image_id = $ImageId
    smoke_test = $SmokeResult
    files = @($VerifiedFiles)
} | ConvertTo-Json -Depth 10
