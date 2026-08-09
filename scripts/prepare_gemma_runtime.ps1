[CmdletBinding()]
param(
    # Prepare 는 원본 GGUF 를 볼륨에 넣고 봉인한다.  Verify 는 읽기 전용 검사다.
    # Reseal 은 이미 검증된 볼륨 내용을 그대로 두고, 현재 llama.cpp image 로 실제
    # smoke 를 다시 통과시킨 뒤 ready manifest 만 다시 쓴다.  업스트림이 지원 태그를
    # 갱신해 image digest 가 움직였을 때 원본 파일 없이 복구하는 유일한 경로다.
    # Auto 는 볼륨 상태를 보고 Prepare 와 Reseal 중 맞는 쪽을 고른다. 앱의 자가복구
    # 경로가 쓰는 모드다.
    [ValidateSet('Prepare', 'Verify', 'Reseal', 'Auto')]
    [string]$Mode = 'Prepare',

    # 비우면 저장소의 `testmodel/`, 그다음 다운로드 캐시를 차례로 찾는다.
    [string]$ModelPath = '',

    # 로컬에서 검증된 원본을 못 찾았을 때만 등록된 Hugging Face 원본을 내려받는다.
    [switch]$AllowDownload,

    # 내려받은 원본을 둘 위치. 비우면 저장소의 `testmodel/`.
    [string]$DownloadDirectory = '',

    [string]$VolumeName = 'comic-translate-gemma-models-v2',

    [ValidateRange(1024, 65535)]
    [int]$SmokePort = 18082,

    [ValidateRange(30, 900)]
    [int]$SmokeTimeoutSec = 420,

    [int64]$MinimumFreeBytes = 32212254720,

    [switch]$SkipFreeSpaceCheck
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)

Import-Module (Join-Path $PSScriptRoot 'lib\ManagedRuntimeModelSource.psm1') -Force

$PreparationVersion = 2
$ManifestSchemaVersion = 2
$ReadyManifestName = '.comic-translate-gemma-ready-v2.json'
$ImageRef = 'ghcr.io/ggml-org/llama.cpp:server-cuda13'
# CUDA 13 태그가 기본이지만, CUDA 12 태그로 준비한 볼륨도 그대로 인정한다.
$SupportedImageRefs = @(
    'ghcr.io/ggml-org/llama.cpp:server-cuda13',
    'ghcr.io/ggml-org/llama.cpp:server-cuda'
)
$ManagedContainerName = 'gemma-local-server'

if ($VolumeName -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]*$') {
    throw "Invalid Docker volume name: $VolumeName"
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
            [void]$Builder.Append([char]92, (($BackslashCount * 2) + 1))
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

function Invoke-DockerResult {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    # PowerShell here-string 은 줄 끝을 CRLF 로 담는다. 컨테이너의 `/bin/sh` 가
    # dash 이면 줄 끝의 CR 을 토큰의 일부로 읽어 첫 줄 `set -eu` 부터
    # "Illegal option -" 로 죽는다. docker 인자에 CR 이 의미를 갖는 경우는 없으므로
    # 여기서 한 번에 정규화한다.
    $NormalizedArguments = @(
        $Arguments | ForEach-Object { [string]$_ -replace "`r`n", "`n" }
    )

    $StartInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $StartInfo.FileName = $DockerExecutable
    $StartInfo.Arguments = (
        $NormalizedArguments |
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
            throw "Unable to start the Docker process: $DockerExecutable"
        }
        $StandardOutputTask = $Process.StandardOutput.ReadToEndAsync()
        $StandardErrorTask = $Process.StandardError.ReadToEndAsync()
        $Process.WaitForExit()
        $StandardOutput = $StandardOutputTask.GetAwaiter().GetResult().TrimEnd()
        $StandardError = $StandardErrorTask.GetAwaiter().GetResult().TrimEnd()
        $CombinedOutput = @(
            $StandardOutput
            $StandardError
        ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
        return [pscustomobject]@{
            ExitCode = [int]$Process.ExitCode
            Output = $CombinedOutput -join "`n"
        }
    }
    finally {
        $Process.Dispose()
    }
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
    if ($Inspect.ExitCode -ne 0 -or [string]::IsNullOrWhiteSpace($Inspect.Output)) {
        Write-Host "Pinned llama.cpp image is missing; pulling it once: $ImageRef"
        Invoke-Docker -Arguments @('pull', $ImageRef) -ShowOutput | Out-Null
        $Inspect = Invoke-DockerResult -Arguments @(
            'image', 'inspect', '--format', '{{.Id}}', $ImageRef
        )
    }
    if ($Inspect.ExitCode -ne 0 -or [string]::IsNullOrWhiteSpace($Inspect.Output)) {
        throw "Unable to inspect the pinned llama.cpp image ID: $ImageRef"
    }
    return $Inspect.Output.Trim()
}

function Get-VolumeFileHash {
    param(
        [Parameter(Mandatory = $true)][string]$FileName,
        [switch]$AllowMissing
    )

    $Script = if ($AllowMissing) {
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
        '-ec', $Script
    )
    if ($Result.ExitCode -ne 0) {
        if ($AllowMissing) {
            return ''
        }
        throw "Volume model SHA-256 verification failed: $FileName`n$($Result.Output)"
    }
    return $Result.Output.Trim().ToLowerInvariant()
}

function Test-ManagedContainerStopped {
    $Inspect = Invoke-DockerResult -Arguments @(
        'inspect', '--format', '{{.State.Running}}', $ManagedContainerName
    )
    if ($Inspect.ExitCode -eq 0 -and $Inspect.Output.Trim() -eq 'true') {
        throw (
            "$ManagedContainerName is running. " +
            'Close the app normally so the container is stopped, then prepare again.'
        )
    }
}

function Assert-GemmaVolumeLabels {
    $VolumeLabelsText = Invoke-Docker -Arguments @(
        'volume', 'inspect', '--format', '{{json .Labels}}', $VolumeName
    )
    try {
        $VolumeLabels = $VolumeLabelsText | ConvertFrom-Json
    }
    catch {
        throw "Unable to read Docker labels for Gemma volume: $VolumeName"
    }
    if (
        [string]$VolumeLabels.'comic-translate.runtime' -ne 'Gemma' -or
        [int]$VolumeLabels.'comic-translate.preparation-version' -ne $PreparationVersion
    ) {
        throw (
            "Gemma volume labels do not match the preparation contract: " +
            "$VolumeName. Use a new versioned volume name."
        )
    }
}

$ModelSpecs = @(
    [pscustomobject][ordered]@{
        Name = 'gemma-4-26B-IQ4_NL.gguf'
        Bytes = [int64]14585439872
        Sha256 = '768a89b94209243b333b2e074b928fe51ea208ebdad6424a510bd73e5cb4d0b8'
        Role = 'product-default'
        SourcePath = $ModelPath
        DownloadUrl = (
            'https://huggingface.co/Vastopian/' +
            'gemma-4-26B-A4B-it-abliterated-GGUF/resolve/main/' +
            'gemma-4-26B-IQ4_NL.gguf'
        )
    }
)

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
    볼륨이 계약된 모든 모델 파일을 이미 담고 있는가(크기 기준).

    .DESCRIPTION
    `Auto` 가 Prepare 와 Reseal 중 무엇을 할지 고르는 데만 쓴다. 크기만 보는 이유는
    수십 GB 를 두 번 해시하지 않기 위해서다. 권위 있는 판정은 Reseal 이 smoke 앞에서
    수행하는 SHA-256 검증이고, 거기서 어긋나면 그대로 실패한다.
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
        Write-Host 'Auto mode: the volume already holds every verified model; resealing.'
        $Mode = 'Reseal'
    }
    else {
        Write-Host 'Auto mode: the volume is missing a verified model; preparing.'
        $Mode = 'Prepare'
    }
}

if ($Mode -eq 'Verify') {
    $VolumeInspect = Invoke-DockerResult -Arguments @('volume', 'inspect', $VolumeName)
    if ($VolumeInspect.ExitCode -ne 0) {
        throw "Gemma volume to verify does not exist: $VolumeName"
    }
    Assert-GemmaVolumeLabels

    $ManifestText = Invoke-Docker -Arguments @(
        'run', '--rm', '--pull', 'never',
        '-e', "READY_MANIFEST=$ReadyManifestName",
        '--mount', "type=volume,source=$VolumeName,target=/models,readonly",
        '--entrypoint', '/bin/sh',
        $ImageRef,
        '-ec', 'set -eu; cat "/models/$READY_MANIFEST"'
    )
    try {
        $Manifest = $ManifestText | ConvertFrom-Json
    }
    catch {
        throw "Unable to read ready manifest JSON: $ReadyManifestName"
    }
    if (
        [int]$Manifest.schema_version -ne $ManifestSchemaVersion -or
        [int]$Manifest.preparation_version -ne $PreparationVersion -or
        [string]$Manifest.runtime -ne 'Gemma' -or
        [string]$Manifest.volume_name -ne $VolumeName -or
        $SupportedImageRefs -notcontains [string]$Manifest.source_image_ref -or
        [string]$Manifest.source_image_digest -ne $ImageId -or
        [string]$Manifest.source_image_id -ne $ImageId -or
        [string]$Manifest.default_model -ne 'gemma-4-26B-IQ4_NL.gguf' -or
        $Manifest.ready -ne $true -or
        $Manifest.smoke_test.passed -ne $true -or
        [string]$Manifest.smoke_test.model -ne 'gemma-4-26B-IQ4_NL.gguf' -or
        [int]$Manifest.runtime_configuration.context_size -ne 4096 -or
        [int]$Manifest.runtime_configuration.parallel -ne 1 -or
        [int]$Manifest.runtime_configuration.threads -ne 10 -or
        [int]$Manifest.runtime_configuration.gpu_layers -ne 23 -or
        [string]$Manifest.runtime_configuration.cache_type_k -ne 'f16' -or
        [string]$Manifest.runtime_configuration.cache_type_v -ne 'f16' -or
        [int]$Manifest.runtime_configuration.cache_ram_mib -ne 0 -or
        [string]$Manifest.runtime_configuration.speculative_type -ne 'none' -or
        [int]$Manifest.runtime_configuration.speculative_draft_max -ne 8
    ) {
        $Hint = if (
            [string]$Manifest.source_image_digest -ne $ImageId -or
            [string]$Manifest.source_image_id -ne $ImageId
        ) {
            # 지원 태그가 업스트림에서 갱신되면 digest 만 어긋난다. 이때는 원본
            # 파일 없이 Reseal 로 복구된다.
            " The llama.cpp image identity drifted (manifest=$($Manifest.source_image_id), actual=$ImageId). Run this script with -Mode Reseal to re-smoke and re-seal the volume."
        }
        else {
            ''
        }
        throw (
            'Ready manifest header does not match the current Gemma runtime contract.' +
            $Hint
        )
    }
    if (@($Manifest.files).Count -ne $ModelSpecs.Count) {
        throw 'Ready manifest model registry does not match the product registry.'
    }

    $VerifiedFiles = @()
    foreach ($Spec in $ModelSpecs) {
        $Entry = @($Manifest.files | Where-Object { $_.name -eq $Spec.Name })
        if ($Entry.Count -ne 1) {
            throw "Ready manifest model entry is missing or duplicated: $($Spec.Name)"
        }
        if (
            [int64]$Entry[0].bytes -ne $Spec.Bytes -or
            ([string]$Entry[0].sha256).ToLowerInvariant() -ne $Spec.Sha256 -or
            [string]$Entry[0].role -ne $Spec.Role
        ) {
            throw "Ready manifest model contract mismatch: $($Spec.Name)"
        }
        $ActualHash = Get-VolumeFileHash -FileName $Spec.Name
        if ($ActualHash -ne $Spec.Sha256) {
            throw (
                "Volume model SHA-256 mismatch: {0}, expected={1}, actual={2}" -f
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
        preparation_version = $PreparationVersion
        files = @($VerifiedFiles)
    } | ConvertTo-Json -Depth 10
    return
}

$IsReseal = $Mode -eq 'Reseal'

Test-ManagedContainerStopped

# Reseal 은 볼륨 안 파일을 그대로 두고 manifest 만 다시 쓴다. 원본을 복사하지
# 않으므로 원본 경로도, 복사할 여유 공간도 필요 없다.
if (-not $IsReseal -and -not $SkipFreeSpaceCheck) {
    $Drive = Get-PSDrive -Name 'C' -ErrorAction Stop
    if ([int64]$Drive.Free -lt $MinimumFreeBytes) {
        throw (
            "Insufficient free C: space. required={0:N2} GiB, actual={1:N2} GiB" -f
            ($MinimumFreeBytes / 1GB),
            ([int64]$Drive.Free / 1GB)
        )
    }
}

$PreparedSources = @()
if (-not $IsReseal) {
    foreach ($Spec in $ModelSpecs) {
        # 볼륨이 이미 계약된 파일을 담고 있으면 원본을 아예 찾지 않는다. 수십 GB
        # 를 헛되이 내려받거나 해시하지 않기 위해서다.
        if ((Get-VolumeFileHash -FileName $Spec.Name -AllowMissing) -eq $Spec.Sha256) {
            Write-Host "Reusing already verified volume model: $($Spec.Name)"
            continue
        }
        $Resolved = Resolve-ManagedRuntimeModelSource `
            -FileName $Spec.Name `
            -Bytes $Spec.Bytes `
            -Sha256 $Spec.Sha256 `
            -RequestedPath $Spec.SourcePath `
            -DownloadUrl $Spec.DownloadUrl `
            -DownloadDirectory $DownloadDirectory `
            -AllowDownload:$AllowDownload
        $PreparedSources += [pscustomobject]@{
            Spec = $Spec
            Path = $Resolved.Path
            Directory = $Resolved.Directory
            FileName = $Resolved.FileName
            Origin = $Resolved.Origin
        }
    }
}

if ($IsReseal) {
    if ((Invoke-DockerResult -Arguments @('volume', 'inspect', $VolumeName)).ExitCode -ne 0) {
        throw (
            "Gemma volume to reseal does not exist: $VolumeName. " +
            'Run this script in Prepare or Auto mode first.'
        )
    }
}
else {
    Invoke-Docker -Arguments @(
        'volume', 'create',
        '--label', 'comic-translate.runtime=Gemma',
        '--label', "comic-translate.preparation-version=$PreparationVersion",
        $VolumeName
    ) | Out-Null
}
Assert-GemmaVolumeLabels

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
    Write-Host "Copying into the volume and verifying: $($Spec.Name)"
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
    ) -ShowOutput | Out-Null
}

$VerifiedFiles = @()
foreach ($Spec in $ModelSpecs) {
    $ActualHash = Get-VolumeFileHash -FileName $Spec.Name
    if ($ActualHash -ne $Spec.Sha256) {
        $Recovery = if ($IsReseal) {
            ' Reseal never copies model data, so run this script with -Mode Prepare to replace the file.'
        }
        else {
            ''
        }
        throw (
            ("Volume model SHA-256 mismatch after {0}: {1}, expected={2}, actual={3}." -f
                $Mode,
                $Spec.Name,
                $Spec.Sha256,
                $ActualHash) + $Recovery
        )
    }
    $VerifiedFiles += [pscustomobject][ordered]@{
        name = $Spec.Name
        bytes = $Spec.Bytes
        sha256 = $ActualHash
        role = $Spec.Role
    }
}

$SmokeModelName = $ModelSpecs[0].Name
$SmokeContainer = "comic-translate-gemma-prepare-smoke-$PID"
$SmokeResult = $null
try {
    Write-Host "Running a real model-load smoke from the new volume: $SmokeModelName"
    Invoke-Docker -Arguments @(
        'run', '-d', '--rm',
        '--name', $SmokeContainer,
        '--label', 'comic-translate.runtime=gemma-prepare-smoke',
        '--gpus', 'all',
        '-e', 'NVIDIA_VISIBLE_DEVICES=all',
        '-e', 'NVIDIA_DRIVER_CAPABILITIES=compute,utility',
        '-p', "127.0.0.1:${SmokePort}:8080",
        '--mount', "type=volume,source=$VolumeName,target=/models,readonly",
        '--entrypoint', '/app/llama-server',
        $ImageRef,
        '-m', "/models/$SmokeModelName",
        '--host', '0.0.0.0',
        '--port', '8080',
        '-c', '4096',
        '-np', '1',
        '-t', '10',
        '--n-gpu-layers', '23',
        '--fit', 'off',
        '-fa', 'on',
        '-ctk', 'f16',
        '-ctv', 'f16',
        '--kv-offload',
        '--swa-full',
        '--jinja',
        '--reasoning', 'off',
        '--reasoning-budget', '0',
        '--reasoning-format', 'none',
        '--metrics',
        '--perf',
        '--cache-ram', '0',
        '--spec-type', 'none',
        '--spec-draft-n-max', '8'
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
        if (-not $HealthReady) {
            Start-Sleep -Seconds 2
        }
    } while ([DateTime]::UtcNow -lt $Deadline)
    if (-not $HealthReady) {
        $Logs = Invoke-DockerResult -Arguments @('logs', '--tail', '100', $SmokeContainer)
        throw "Gemma volume smoke health timeout (${SmokeTimeoutSec}s).`n$($Logs.Output)"
    }

    $Models = Invoke-RestMethod `
        -Uri "http://127.0.0.1:${SmokePort}/v1/models" `
        -TimeoutSec 10
    $LoadedIds = @($Models.data | ForEach-Object { [string]$_.id })
    $LoadedModel = @(
        $LoadedIds |
            Where-Object { (Split-Path -Leaf $_) -eq $SmokeModelName }
    )
    if ($LoadedModel.Count -ne 1) {
        throw (
            "Gemma volume smoke model mismatch: expected=$SmokeModelName, " +
            "actual=$($LoadedIds -join ', ')"
        )
    }

    $ChatPayload = [ordered]@{
        model = $LoadedIds[0]
        messages = @(
            [ordered]@{
                role = 'system'
                content = @(
                    [ordered]@{
                        type = 'text'
                        text = 'Return exactly one JSON object.'
                    }
                )
            }
            [ordered]@{
                role = 'user'
                content = @(
                    [ordered]@{
                        type = 'text'
                        text = '{"translation":"ok"}'
                    }
                )
            }
        )
        temperature = 0.0
        max_completion_tokens = 32
        response_format = [ordered]@{ type = 'json_object' }
    }
    $Chat = Invoke-RestMethod `
        -Method Post `
        -Uri "http://127.0.0.1:${SmokePort}/v1/chat/completions" `
        -ContentType 'application/json' `
        -Body ($ChatPayload | ConvertTo-Json -Depth 10 -Compress) `
        -TimeoutSec 90
    $Content = [string]($Chat.choices[0].message.content)
    if ([string]::IsNullOrWhiteSpace($Content)) {
        throw 'Gemma volume smoke chat response is empty.'
    }
    $SmokeResult = [ordered]@{
        passed = $true
        model = $SmokeModelName
        health_status = 'ok'
        models_match = $true
        chat_response_nonempty = $true
    }
}
finally {
    $SmokeInspect = Invoke-DockerResult -Arguments @(
        'inspect', '--format', '{{.Name}}', $SmokeContainer
    )
    if ($SmokeInspect.ExitCode -eq 0) {
        Invoke-Docker -Arguments @(
            'stop', '--timeout', '10', $SmokeContainer
        ) -ShowOutput | Out-Null
    }
}

$Manifest = [ordered]@{
    schema_version = $ManifestSchemaVersion
    runtime = 'Gemma'
    preparation_version = $PreparationVersion
    volume_name = $VolumeName
    ready = $true
    source_image_ref = $ImageRef
    source_image_digest = $ImageId
    source_image_id = $ImageId
    default_model = 'gemma-4-26B-IQ4_NL.gguf'
    runtime_configuration = [ordered]@{
        context_size = 4096
        parallel = 1
        threads = 10
        gpu_layers = 23
        cache_type_k = 'f16'
        cache_type_v = 'f16'
        cache_ram_mib = 0
        speculative_type = 'none'
        speculative_draft_max = 8
    }
    files = @($VerifiedFiles)
    smoke_test = $SmokeResult
}

$TemporaryRoot = Join-Path ([System.IO.Path]::GetTempPath()) (
    "comic-translate-gemma-manifest-$PID"
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
    Remove-Item -LiteralPath $TemporaryRoot -Recurse -Force -ErrorAction SilentlyContinue
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
    prepared = $true
    resealed = $IsReseal
    model_sources = @(
        $PreparedSources | ForEach-Object {
            [ordered]@{ name = $_.Spec.Name; origin = $_.Origin }
        }
    )
    volume_name = $VolumeName
    ready_manifest = $ReadyManifestName
    ready_manifest_sha256 = $ManifestSha256
    image_ref = $ImageRef
    image_id = $ImageId
    preparation_version = $PreparationVersion
    smoke_test = $SmokeResult
    files = @($VerifiedFiles)
} | ConvertTo-Json -Depth 10
