[CmdletBinding()]
param(
    [ValidateSet('Prepare', 'Verify')]
    [string]$Mode = 'Prepare',

    [string]$ModelDirectory = '',

    [string]$VolumeName = 'comic-translate-hunyuanocr-models-v2',

    [ValidateRange(1024, 65535)]
    [int]$SmokePort = 18086,

    [ValidateRange(30, 600)]
    [int]$SmokeTimeoutSec = 300,

    [int64]$MinimumFreeBytes = 5368709120,

    [switch]$SkipFreeSpaceCheck
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)

$PreparationVersion = 1
$ManifestSchemaVersion = 1
$ReadyManifestName = '.comic-translate-hunyuanocr-ready-v1.json'
$RuntimeName = 'HunyuanOCR-llama.cpp'
$ImageRef = 'ghcr.io/ggml-org/llama.cpp:server-cuda13'
# CUDA 13 태그가 기본이지만, CUDA 12 태그로 준비한 볼륨도 그대로 인정한다.
$SupportedImageRefs = @(
    'ghcr.io/ggml-org/llama.cpp:server-cuda13',
    'ghcr.io/ggml-org/llama.cpp:server-cuda'
)
$ManagedContainerName = 'hunyuanocr-local-server'
$ModelAlias = 'HunyuanOCR.Q8_0.gguf'

$ModelSpecs = @(
    [pscustomobject][ordered]@{
        Name = 'HunyuanOCR.Q8_0.gguf'
        Bytes = [int64]577949408
        Sha256 = 'cdafc794cafeae377868d7a40a70e282a737e39abe77c0d8b73614447b364a21'
        Role = 'vlm'
    }
    [pscustomobject][ordered]@{
        Name = 'HunyuanOCR.mmproj-Q8_0.gguf'
        Bytes = [int64]732938240
        Sha256 = 'b77913164ff73d4c0dc4d994e236ed72bacbbe5c5db1ec9b2828627b46c32804'
        Role = 'vision-projector'
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

    $StartInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $StartInfo.FileName = $DockerExecutable
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
                Where-Object {
                    -not [string]::IsNullOrWhiteSpace($_)
                }) -join "`n"
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
    if (
        $Inspect.ExitCode -ne 0 -or
        [string]::IsNullOrWhiteSpace($Inspect.Output)
    ) {
        Write-Host "Pulling the pinned llama.cpp image once: $ImageRef"
        Invoke-Docker -Arguments @('pull', $ImageRef) -ShowOutput | Out-Null
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
        [int]$Labels.'comic-translate.preparation-version' -ne
            $PreparationVersion
    ) {
        throw (
            "HunyuanOCR volume labels do not match: $VolumeName. " +
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
        $Manifest.smoke_test.passed -ne $true -or
        [string]$Manifest.smoke_test.device -ne 'CUDA' -or
        [string]$Manifest.smoke_test.model_alias -ne $ModelAlias
    ) {
        throw 'Ready manifest header does not match the HunyuanOCR contract.'
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

$ImageId = Get-PinnedImageId

if ($Mode -eq 'Verify') {
    if (
        (Invoke-DockerResult -Arguments @(
            'volume', 'inspect', $VolumeName
        )).ExitCode -ne 0
    ) {
        throw "HunyuanOCR volume does not exist: $VolumeName"
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

if ([string]::IsNullOrWhiteSpace($ModelDirectory)) {
    throw 'Prepare mode requires -ModelDirectory.'
}
Assert-ManagedContainerStopped

if (-not $SkipFreeSpaceCheck) {
    $Drive = Get-PSDrive -Name 'C' -ErrorAction Stop
    if ([int64]$Drive.Free -lt $MinimumFreeBytes) {
        throw (
            "Insufficient free C: space. required={0:N2} GiB, actual={1:N2} GiB" -f
            ($MinimumFreeBytes / 1GB),
            ([int64]$Drive.Free / 1GB)
        )
    }
}

$ResolvedDirectory = (Resolve-Path -LiteralPath $ModelDirectory).Path
$PreparedSources = @()
foreach ($Spec in $ModelSpecs) {
    $SourcePath = Join-Path $ResolvedDirectory $Spec.Name
    $Item = Get-Item -LiteralPath $SourcePath
    if ($Item.Length -ne $Spec.Bytes) {
        throw (
            "Source size mismatch: {0}, expected={1}, actual={2}" -f
            $Spec.Name,
            $Spec.Bytes,
            $Item.Length
        )
    }
    Write-Host "Checking source SHA-256: $($Spec.Name)"
    $SourceHash = (
        Get-FileHash -LiteralPath $SourcePath -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    if ($SourceHash -ne $Spec.Sha256) {
        throw (
            "Source SHA-256 mismatch: {0}, expected={1}, actual={2}" -f
            $Spec.Name,
            $Spec.Sha256,
            $SourceHash
        )
    }
    $PreparedSources += [pscustomobject]@{
        Spec = $Spec
        Directory = Split-Path -Parent $SourcePath
        FileName = Split-Path -Leaf $SourcePath
    }
}

Invoke-Docker -Arguments @(
    'volume', 'create',
    '--label', "comic-translate.runtime=$RuntimeName",
    '--label', "comic-translate.preparation-version=$PreparationVersion",
    $VolumeName
) | Out-Null
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
    $ExistingHash = Get-VolumeFileHash -FileName $Spec.Name -AllowMissing
    if ($ExistingHash -eq $Spec.Sha256) {
        Write-Host "Reusing verified volume file: $($Spec.Name)"
        continue
    }
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
        throw "Copied file SHA-256 mismatch: $($Spec.Name)"
    }
    $VerifiedFiles += [pscustomobject][ordered]@{
        name = $Spec.Name
        bytes = $Spec.Bytes
        sha256 = $ActualHash
        role = $Spec.Role
    }
}

$SmokeContainer = "comic-translate-hunyuanocr-prepare-smoke-$PID"
$SmokeResult = $null
try {
    Write-Host 'Running CUDA model-load smoke from the named volume.'
    Invoke-Docker -Arguments @(
        'run', '-d', '--rm',
        '--name', $SmokeContainer,
        '--label', 'comic-translate.runtime=hunyuanocr-prepare-smoke',
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
        '-c', '4096',
        '-np', '1',
        '-t', '12',
        '--cache-ram', '0',
        '--n-gpu-layers', '80',
        '--image-max-tokens', '1024'
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
        throw "HunyuanOCR smoke timed out.`n$($Logs.Output)"
    }

    $Models = Invoke-RestMethod `
        -Uri "http://127.0.0.1:${SmokePort}/v1/models" `
        -TimeoutSec 10
    $LoadedIds = @($Models.data | ForEach-Object { [string]$_.id })
    if (@($LoadedIds | Where-Object { $_ -eq $ModelAlias }).Count -ne 1) {
        throw (
            "HunyuanOCR model alias mismatch: expected=$ModelAlias, " +
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
        context_size = 4096
        parallel = 1
        threads = 12
        gpu_layers = 80
        image_max_tokens = 1024
        prompt_cache = $false
    }
    files = @($VerifiedFiles)
    smoke_test = $SmokeResult
}

$TemporaryRoot = Join-Path ([System.IO.Path]::GetTempPath()) (
    "comic-translate-hunyuanocr-manifest-$PID"
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
    mode = 'Prepare'
    prepared = $true
    volume_name = $VolumeName
    ready_manifest = $ReadyManifestName
    ready_manifest_sha256 = $ManifestSha256
    image_ref = $ImageRef
    image_id = $ImageId
    smoke_test = $SmokeResult
    files = @($VerifiedFiles)
} | ConvertTo-Json -Depth 10
