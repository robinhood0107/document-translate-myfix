$script:DockerExecutable = ''
$script:ImageRef = ''
$script:VolumeName = ''
$script:ContainerName = ''
$script:RuntimeName = ''
$script:PreparationVersion = 0
$script:ReadyManifestName = ''
$script:ModelSpecs = @()

function Get-ManagedLlamaCppImagePolicy {
    param(
        [Parameter(Mandatory = $true)]
        [ValidateSet('cuda12', 'cuda13')]
        [string]$Runtime
    )

    $Cuda12 = 'ghcr.io/ggml-org/llama.cpp:server-cuda'
    $Cuda13 = 'ghcr.io/ggml-org/llama.cpp:server-cuda13'
    return [pscustomobject]@{
        Preferred = if ($Runtime -eq 'cuda12') { $Cuda12 } else { $Cuda13 }
        Fallback = if ($Runtime -eq 'cuda13') { $Cuda12 } else { '' }
        Supported = @($Cuda13, $Cuda12)
    }
}

function Resolve-ManagedLlamaCppImageRef {
    param(
        [string]$RequestedImage = '',
        [string]$RuntimeOverride = ''
    )

    $Policy = Get-ManagedLlamaCppImagePolicy -Runtime 'cuda13'
    $Resolved = @($RequestedImage, $RuntimeOverride, $env:LLAMA_CPP_IMAGE) |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
        Select-Object -First 1
    if ([string]::IsNullOrWhiteSpace($Resolved)) {
        $Resolved = $Policy.Preferred
    }
    if ($Policy.Supported -notcontains $Resolved) {
        throw "Unsupported llama.cpp image: $Resolved"
    }
    return [string]$Resolved
}

function ConvertTo-NativeArgument {
    param([AllowEmptyString()][string]$Argument)

    if (-not [string]::IsNullOrEmpty($Argument) -and $Argument -notmatch '[\s"]') {
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

function Initialize-ManagedRuntimeDocker {
    param(
        [Parameter(Mandatory = $true)][string]$ImageRef,
        [Parameter(Mandatory = $true)][string]$VolumeName,
        [Parameter(Mandatory = $true)][string]$ContainerName,
        [Parameter(Mandatory = $true)][string]$RuntimeName,
        [Parameter(Mandatory = $true)][int]$PreparationVersion,
        [Parameter(Mandatory = $true)][string]$ReadyManifestName,
        [Parameter(Mandatory = $true)][object[]]$ModelSpecs
    )

    if ($VolumeName -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]*$') {
        throw "Invalid Docker volume name: $VolumeName"
    }
    $DockerCommand = Get-Command 'docker.exe' -ErrorAction SilentlyContinue
    if ($null -eq $DockerCommand) {
        $DockerCommand = Get-Command 'docker' -ErrorAction Stop
    }
    $script:DockerExecutable = $DockerCommand.Source
    $script:ImageRef = $ImageRef
    $script:VolumeName = $VolumeName
    $script:ContainerName = $ContainerName
    $script:RuntimeName = $RuntimeName
    $script:PreparationVersion = $PreparationVersion
    $script:ReadyManifestName = $ReadyManifestName
    $script:ModelSpecs = @($ModelSpecs)
}

function Invoke-DockerResult {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    $NormalizedArguments = @(
        $Arguments | ForEach-Object { [string]$_ -replace "`r`n", "`n" }
    )
    $StartInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $StartInfo.FileName = $script:DockerExecutable
    $StartInfo.Arguments = ($NormalizedArguments | ForEach-Object {
        ConvertTo-NativeArgument -Argument $_
    }) -join ' '
    $StartInfo.UseShellExecute = $false
    $StartInfo.CreateNoWindow = $true
    $StartInfo.RedirectStandardOutput = $true
    $StartInfo.RedirectStandardError = $true

    $Process = [System.Diagnostics.Process]::new()
    $Process.StartInfo = $StartInfo
    try {
        if (-not $Process.Start()) {
            throw "Unable to start Docker: $($script:DockerExecutable)"
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
            $Result.ExitCode, ($Arguments -join ' '), $Result.Output
        )
    }
    return $Result.Output.Trim()
}

function Get-PinnedImageId {
    $Inspect = Invoke-DockerResult -Arguments @(
        'image', 'inspect', '--format', '{{.Id}}', $script:ImageRef
    )
    if ($Inspect.ExitCode -ne 0 -or [string]::IsNullOrWhiteSpace($Inspect.Output)) {
        Write-Host "Pulling the pinned llama.cpp image once: $($script:ImageRef)"
        Invoke-Docker -Arguments @('pull', $script:ImageRef) -ShowOutput | Out-Null
        $Inspect = Invoke-DockerResult -Arguments @(
            'image', 'inspect', '--format', '{{.Id}}', $script:ImageRef
        )
    }
    if ($Inspect.ExitCode -ne 0 -or [string]::IsNullOrWhiteSpace($Inspect.Output)) {
        throw "Unable to inspect the pinned llama.cpp image: $($script:ImageRef)"
    }
    return $Inspect.Output.Trim()
}

function Assert-ManagedContainerStopped {
    $Inspect = Invoke-DockerResult -Arguments @(
        'inspect', '--format', '{{.State.Running}}', $script:ContainerName
    )
    if ($Inspect.ExitCode -eq 0 -and $Inspect.Output.Trim() -eq 'true') {
        throw "$($script:ContainerName) is running. Stop the app normally, then prepare again."
    }
}

function Assert-VolumeLabels {
    $LabelsText = Invoke-Docker -Arguments @(
        'volume', 'inspect', '--format', '{{json .Labels}}', $script:VolumeName
    )
    try { $Labels = $LabelsText | ConvertFrom-Json }
    catch { throw "Unable to read Docker labels for volume: $($script:VolumeName)" }
    if (
        [string]$Labels.'comic-translate.runtime' -ne $script:RuntimeName -or
        [int]$Labels.'comic-translate.preparation-version' -ne $script:PreparationVersion
    ) {
        throw "Managed runtime volume labels do not match: $($script:VolumeName). Use a new versioned volume name."
    }
}

function Get-VolumeFileHash {
    param(
        [Parameter(Mandatory = $true)][string]$FileName,
        [switch]$AllowMissing
    )

    $Shell = if ($AllowMissing) {
        'if test -f "/models/$MODEL_FILE"; then sha256sum "/models/$MODEL_FILE" | cut -d " " -f 1; fi'
    } else {
        'set -eu; test -f "/models/$MODEL_FILE"; sha256sum "/models/$MODEL_FILE" | cut -d " " -f 1'
    }
    $Result = Invoke-DockerResult -Arguments @(
        'run', '--rm', '--pull', 'never', '-e', "MODEL_FILE=$FileName",
        '--mount', "type=volume,source=$($script:VolumeName),target=/models,readonly",
        '--entrypoint', '/bin/sh', $script:ImageRef, '-ec', $Shell
    )
    if ($Result.ExitCode -ne 0) {
        if ($AllowMissing) { return '' }
        throw "Volume SHA-256 verification failed: $FileName`n$($Result.Output)"
    }
    return $Result.Output.Trim().ToLowerInvariant()
}

function Read-ReadyManifest {
    $ManifestText = Invoke-Docker -Arguments @(
        'run', '--rm', '--pull', 'never',
        '-e', "READY_MANIFEST=$($script:ReadyManifestName)",
        '--mount', "type=volume,source=$($script:VolumeName),target=/models,readonly",
        '--entrypoint', '/bin/sh', $script:ImageRef,
        '-ec', 'set -eu; cat "/models/$READY_MANIFEST"'
    )
    try { return $ManifestText | ConvertFrom-Json }
    catch { throw "Unable to parse ready manifest: $($script:ReadyManifestName)" }
}

function Get-VolumeFileSize {
    param([Parameter(Mandatory = $true)][string]$FileName)

    $Result = Invoke-DockerResult -Arguments @(
        'run', '--rm', '--pull', 'never', '-e', "MODEL_FILE=$FileName",
        '--mount', "type=volume,source=$($script:VolumeName),target=/models,readonly",
        '--entrypoint', '/bin/sh', $script:ImageRef,
        '-ec', 'if test -f "/models/$MODEL_FILE"; then stat -c %s "/models/$MODEL_FILE"; fi'
    )
    if ($Result.ExitCode -ne 0 -or [string]::IsNullOrWhiteSpace($Result.Output)) {
        return [int64]-1
    }
    return [int64]$Result.Output.Trim()
}

function Test-VolumeHoldsEveryModel {
    if ((Invoke-DockerResult -Arguments @('volume', 'inspect', $script:VolumeName)).ExitCode -ne 0) {
        return $false
    }
    foreach ($Spec in $script:ModelSpecs) {
        if ((Get-VolumeFileSize -FileName $Spec.Name) -ne $Spec.Bytes) {
            return $false
        }
    }
    return $true
}

Export-ModuleMember -Function *
