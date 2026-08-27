[CmdletBinding()]
param(
    # Prepare 는 원본을 볼륨에 넣고 봉인한다. Verify 는 읽기 전용 검사다.
    # Reseal 은 볼륨 내용을 그대로 두고 현재 llama.cpp image 로 smoke 를 다시
    # 통과시킨 뒤 ready manifest 만 다시 쓴다. Auto 는 볼륨 상태를 보고 Prepare 와
    # Reseal 중 맞는 쪽을 고르며, 앱의 자가복구 경로가 쓰는 모드다.
    [ValidateSet('Prepare', 'Verify', 'Reseal', 'Auto')]
    [string]$Mode = 'Prepare',

    # 비우면 저장소의 `testmodel/`, 그다음 다운로드 캐시를 차례로 찾는다.
    [string]$ModelDirectory = '',

    # 로컬에서 검증된 원본을 못 찾았을 때만 등록된 원본을 내려받는다.
    [switch]$AllowDownload,

    # 내려받은 원본을 둘 위치. 비우면 저장소의 `testmodel/`.
    [string]$DownloadDirectory = '',

    [string]$VolumeName = 'comic-translate-paddleocr-vl-llamacpp-models-v1',

    [ValidateSet(
        'ghcr.io/ggml-org/llama.cpp:server-cuda',
        'ghcr.io/ggml-org/llama.cpp:server-cuda13'
    )]
    [string]$ImageRef = $(
        if ($env:PADDLEOCR_LLAMA_CPP_IMAGE) {
            $env:PADDLEOCR_LLAMA_CPP_IMAGE
        }
        elseif ($env:LLAMA_CPP_IMAGE) { $env:LLAMA_CPP_IMAGE }
        else { 'ghcr.io/ggml-org/llama.cpp:server-cuda13' }
    ),

    [ValidateSet('CUDA', 'CPU')]
    [string]$SmokeDevice = 'CUDA',

    [ValidateRange(1024, 65535)]
    [int]$SmokePort = 18084,

    [ValidateRange(30, 600)]
    [int]$SmokeTimeoutSec = 180,

    [int64]$MinimumFreeBytes = 0,

    [switch]$SkipFreeSpaceCheck
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)

Import-Module (Join-Path $PSScriptRoot 'lib\ManagedRuntimeModelSource.psm1') -Force

$PreparationVersion = 1
$ManifestSchemaVersion = 1
$ReadyManifestName = '.comic-translate-paddleocr-vl-llamacpp-ready-v1.json'
$RuntimeName = 'PaddleOCR-VL-llama.cpp'
# CUDA 13 태그가 기본이지만, CUDA 12 태그로 준비한 볼륨도 그대로 인정한다.
$SupportedImageRefs = @(
    'ghcr.io/ggml-org/llama.cpp:server-cuda13',
    'ghcr.io/ggml-org/llama.cpp:server-cuda'
)
$ManagedContainerName = 'paddleocr-llamacpp'
$ModelAlias = 'PaddleOCR-VL-1.6-0.9B'

$ModelSpecs = @(
    [pscustomobject][ordered]@{
        Name = 'PaddleOCR-VL-1.6-GGUF.gguf'
        Bytes = [int64]935769056
        Sha256 = 'f3ae46ec885050acf4b3d31944431e1fd90d50664fb09126af4a3c050ba14ee8'
        Role = 'vlm'
        DownloadUrl = (
            'https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6-GGUF/' +
            'resolve/main/PaddleOCR-VL-1.6-GGUF.gguf'
        )
    }
    [pscustomobject][ordered]@{
        Name = 'PaddleOCR-VL-1.6-GGUF-mmproj.gguf'
        Bytes = [int64]881770560
        Sha256 = '204d757d7610d9b3faab10d506d69e5b244e32bf765e2bab2d0167e65e0a058a'
        Role = 'vision-projector'
        DownloadUrl = (
            'https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6-GGUF/' +
            'resolve/main/PaddleOCR-VL-1.6-GGUF-mmproj.gguf'
        )
    }
)

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
            throw "Unable to start Docker: $DockerExecutable"
        }
        $OutputTask = $Process.StandardOutput.ReadToEndAsync()
        $ErrorTask = $Process.StandardError.ReadToEndAsync()
        $Process.WaitForExit()
        $Output = $OutputTask.GetAwaiter().GetResult().TrimEnd()
        $ErrorOutput = $ErrorTask.GetAwaiter().GetResult().TrimEnd()
        return [pscustomobject]@{
            ExitCode = [int]$Process.ExitCode
            Output = (@($Output, $ErrorOutput) |
                Where-Object { -not [string]::IsNullOrWhiteSpace($_) }) -join "`n"
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
        Write-Host "Pulling the pinned llama.cpp image once: $ImageRef"
        Invoke-Docker -Arguments @('pull', $ImageRef) -ShowOutput | Out-Null
        $Inspect = Invoke-DockerResult -Arguments @(
            'image', 'inspect', '--format', '{{.Id}}', $ImageRef
        )
    }
    if ($Inspect.ExitCode -ne 0 -or [string]::IsNullOrWhiteSpace($Inspect.Output)) {
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
            "$ManagedContainerName is running. Stop the app normally, then " +
            'prepare the model volume again.'
        )
    }
}

function Assert-VolumeLabels {
    $LabelsText = Invoke-Docker -Arguments @(
        'volume', 'inspect', '--format', '{{json .Labels}}', $VolumeName
    )
    try {
        $Labels = $LabelsText | ConvertFrom-Json
    }
    catch {
        throw "Unable to read Docker labels for volume: $VolumeName"
    }
    if (
        [string]$Labels.'comic-translate.runtime' -ne $RuntimeName -or
        [int]$Labels.'comic-translate.preparation-version' -ne $PreparationVersion
    ) {
        throw (
            "PaddleOCR llama.cpp volume labels do not match: $VolumeName. " +
            'Use a new versioned volume name.'
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
    try {
        return $ManifestText | ConvertFrom-Json
    }
    catch {
        throw "Unable to parse ready manifest: $ReadyManifestName"
    }
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
        $Manifest.smoke_test.passed -ne $true
    ) {
        if ([string]$Manifest.source_image_id -ne $ImageId) {
            # 지원 태그가 업스트림에서 갱신되면 image identity 만 어긋난다. 이때는
            # 원본 파일 없이 Reseal 로 복구된다.
            throw (
                'Ready manifest header does not match the PaddleOCR llama.cpp contract: the ' +
                "llama.cpp image identity drifted (manifest=$($Manifest.source_image_id), " +
                "actual=$ImageId). Run this script with -Mode Reseal to re-smoke and re-seal."
            )
        }
        throw 'Ready manifest header does not match the PaddleOCR llama.cpp contract.'
    }
    if (@($Manifest.files).Count -ne $ModelSpecs.Count) {
        throw 'Ready manifest file registry is incomplete.'
    }
    foreach ($Spec in $ModelSpecs) {
        $Entry = @($Manifest.files | Where-Object { $_.name -eq $Spec.Name })
        if (
            $Entry.Count -ne 1 -or
            [int64]$Entry[0].bytes -ne $Spec.Bytes -or
            ([string]$Entry[0].sha256).ToLowerInvariant() -ne $Spec.Sha256 -or
            [string]$Entry[0].role -ne $Spec.Role
        ) {
            throw "Ready manifest file contract mismatch: $($Spec.Name)"
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
    볼륨이 계약된 모든 파일을 이미 담고 있는가(크기 기준).

    .DESCRIPTION
    `Auto` 가 Prepare 와 Reseal 중 무엇을 할지 고르는 데만 쓴다. 크기만 보는 이유는
    대형 GGUF 를 두 번 해시하지 않기 위해서다. 권위 있는 판정은 Reseal 이 smoke 앞에서
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
        Write-Host 'Auto mode: the volume already holds every model; resealing.'
        $Mode = 'Reseal'
    }
    else {
        Write-Host 'Auto mode: the volume is missing a model; preparing.'
        $Mode = 'Prepare'
    }
}

if ($Mode -eq 'Verify') {
    if ((Invoke-DockerResult -Arguments @('volume', 'inspect', $VolumeName)).ExitCode -ne 0) {
        throw "PaddleOCR llama.cpp volume does not exist: $VolumeName"
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
    foreach ($Spec in $ModelSpecs) {
        # 볼륨이 이미 계약된 파일을 담고 있으면 원본을 아예 찾지 않는다. 대형
        # GGUF 를 헛되이 내려받거나 해시하지 않기 위해서다.
        if ((Get-VolumeFileHash -FileName $Spec.Name -AllowMissing) -eq $Spec.Sha256) {
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

$SmokeContainer = "comic-translate-paddle-llama-prepare-smoke-$PID"
$SmokeResult = $null
try {
    Write-Host "Running $SmokeDevice model-load smoke from the named volume."
    $DockerArgs = @(
        'run', '-d', '--rm',
        '--name', $SmokeContainer,
        '--label', 'comic-translate.runtime=paddle-llama-prepare-smoke',
        '-p', "127.0.0.1:${SmokePort}:8080",
        '--mount', "type=volume,source=$VolumeName,target=/models,readonly",
        '--entrypoint', '/app/llama-server'
    )
    if ($SmokeDevice -eq 'CUDA') {
        $DockerArgs += @(
            '--gpus', 'all',
            '-e', 'NVIDIA_VISIBLE_DEVICES=all',
            '-e', 'NVIDIA_DRIVER_CAPABILITIES=compute,utility'
        )
    }
    $DockerArgs += @(
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
        '--n-gpu-layers', $(if ($SmokeDevice -eq 'CUDA') { 'all' } else { '0' }),
        '--fit', 'off',
        '--flash-attn', 'on',
        '--temp', '0',
        '--metrics',
        '--sleep-idle-seconds', '5'
    )
    if ($SmokeDevice -eq 'CPU') {
        $DockerArgs += '--no-mmproj-offload'
    }
    Invoke-Docker -Arguments $DockerArgs | Out-Null

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
        throw "PaddleOCR llama.cpp smoke timed out.`n$($Logs.Output)"
    }

    $Models = Invoke-RestMethod `
        -Uri "http://127.0.0.1:${SmokePort}/v1/models" `
        -TimeoutSec 10
    $LoadedIds = @($Models.data | ForEach-Object { [string]$_.id })
    if (@($LoadedIds | Where-Object { $_ -eq $ModelAlias }).Count -ne 1) {
        throw (
            "PaddleOCR llama.cpp model alias mismatch: expected=$ModelAlias, " +
            "actual=$($LoadedIds -join ', ')"
        )
    }

    Add-Type -AssemblyName System.Drawing
    $Bitmap = [System.Drawing.Bitmap]::new(256, 96)
    $Graphics = [System.Drawing.Graphics]::FromImage($Bitmap)
    $Font = [System.Drawing.Font]::new(
        [System.Drawing.FontFamily]::GenericSansSerif,
        40,
        [System.Drawing.FontStyle]::Bold
    )
    $PngStream = [System.IO.MemoryStream]::new()
    try {
        $Graphics.Clear([System.Drawing.Color]::White)
        $Graphics.DrawString(
            'OCR',
            $Font,
            [System.Drawing.Brushes]::Black,
            48,
            18
        )
        $Bitmap.Save($PngStream, [System.Drawing.Imaging.ImageFormat]::Png)
        $ImageBase64 = [Convert]::ToBase64String($PngStream.ToArray())
    }
    finally {
        $PngStream.Dispose()
        $Font.Dispose()
        $Graphics.Dispose()
        $Bitmap.Dispose()
    }
    $OcrPayload = [ordered]@{
        model = $ModelAlias
        messages = @(
            [ordered]@{
                role = 'user'
                content = @(
                    [ordered]@{
                        type = 'image_url'
                        image_url = [ordered]@{
                            url = "data:image/png;base64,$ImageBase64"
                        }
                    }
                    [ordered]@{ type = 'text'; text = 'OCR:' }
                )
            }
        )
        temperature = 0
        max_tokens = 64
        stream = $false
    } | ConvertTo-Json -Depth 10 -Compress
    $OcrResponse = Invoke-RestMethod `
        -Method Post `
        -Uri "http://127.0.0.1:${SmokePort}/v1/chat/completions" `
        -ContentType 'application/json' `
        -Body $OcrPayload `
        -TimeoutSec 60
    if (
        @($OcrResponse.choices).Count -lt 1 -or
        [string]$OcrResponse.choices[0].finish_reason -eq 'length'
    ) {
        throw 'PaddleOCR llama.cpp direct OCR smoke returned an invalid response.'
    }
    $SmokeResult = [ordered]@{
        passed = $true
        device = $SmokeDevice
        health_status = 'ok'
        model_alias = $ModelAlias
        models_match = $true
        direct_ocr_request = $true
        direct_ocr_finish_reason = [string]$OcrResponse.choices[0].finish_reason
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
        context_size = 4096
        parallel = 1
        threads = 10
        batch_size = 2048
        ubatch_size = 512
        gpu_layers = 'all'
        flash_attention = $true
        metrics = $true
        sleep_idle_seconds = 5
    }
    files = @($VerifiedFiles)
    smoke_test = $SmokeResult
}

$TemporaryRoot = Join-Path ([System.IO.Path]::GetTempPath()) (
    "comic-translate-paddle-llama-manifest-$PID"
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
