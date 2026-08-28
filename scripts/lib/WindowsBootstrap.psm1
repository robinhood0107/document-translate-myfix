Set-StrictMode -Version Latest

if (-not ('ComicBootstrapProcess' -as [type])) {
    Add-Type -Language CSharp -TypeDefinition @'
using System;
using System.Diagnostics;
using System.Text;

public sealed class ComicBootstrapProcessResult {
    public int ExitCode { get; set; }
    public string Output { get; set; }
}

public static class ComicBootstrapProcess {
    private static bool ShouldDisplay(string line) {
        if (String.IsNullOrWhiteSpace(line)) return false;
        return line.StartsWith("[model", StringComparison.OrdinalIgnoreCase)
            || line.StartsWith("[download]", StringComparison.OrdinalIgnoreCase)
            || line.StartsWith("[ERROR]", StringComparison.OrdinalIgnoreCase)
            || line.StartsWith("Application model", StringComparison.OrdinalIgnoreCase)
            || line.StartsWith("Downloading", StringComparison.OrdinalIgnoreCase)
            || line.StartsWith("Auto mode", StringComparison.OrdinalIgnoreCase)
            || line.StartsWith("Running CUDA", StringComparison.OrdinalIgnoreCase)
            || line.StartsWith("Running a real", StringComparison.OrdinalIgnoreCase)
            || line.StartsWith("Waiting for", StringComparison.OrdinalIgnoreCase)
            || (line.StartsWith("[") && line.Contains("%"));
    }

    public static ComicBootstrapProcessResult Run(
        string fileName,
        string arguments,
        string workingDirectory,
        bool liveOutput
    ) {
        var info = new ProcessStartInfo();
        info.FileName = fileName;
        info.Arguments = arguments ?? "";
        info.WorkingDirectory = String.IsNullOrWhiteSpace(workingDirectory)
            ? Environment.CurrentDirectory
            : workingDirectory;
        info.UseShellExecute = false;
        info.CreateNoWindow = true;
        info.RedirectStandardOutput = true;
        info.RedirectStandardError = true;

        var output = new StringBuilder();
        var gate = new object();
        using (var process = new Process()) {
            process.StartInfo = info;
            process.OutputDataReceived += delegate(object sender, DataReceivedEventArgs eventArgs) {
                if (eventArgs.Data == null) return;
                lock (gate) { output.AppendLine(eventArgs.Data); }
                if (liveOutput && ShouldDisplay(eventArgs.Data)) Console.WriteLine(eventArgs.Data);
            };
            process.ErrorDataReceived += delegate(object sender, DataReceivedEventArgs eventArgs) {
                if (eventArgs.Data == null) return;
                lock (gate) { output.AppendLine(eventArgs.Data); }
                if (liveOutput && ShouldDisplay(eventArgs.Data)) Console.WriteLine(eventArgs.Data);
            };
            if (!process.Start()) throw new InvalidOperationException("Unable to start " + fileName);
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            process.WaitForExit();
            process.WaitForExit();
            return new ComicBootstrapProcessResult {
                ExitCode = process.ExitCode,
                Output = output.ToString().TrimEnd()
            };
        }
    }
}
'@
}

function Write-BootstrapMessage {
    param(
        [Parameter(Mandatory = $true)][string]$Message,
        [ValidateSet('INFO', 'OK', 'SKIP', 'WARN', 'ERROR')][string]$Level = 'INFO'
    )
    $Color = @{ INFO = 'Gray'; OK = 'Green'; SKIP = 'DarkGray'; WARN = 'Yellow'; ERROR = 'Red' }[$Level]
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
        [switch]$Quiet,
        [switch]$ShowOutput,
        [switch]$LiveOutput
    )
    if (-not $Quiet) {
        $DisplayName = [System.IO.Path]::GetFileName($FilePath)
        Write-BootstrapMessage "Running $DisplayName..."
    }
    $Previous = Get-Location
    try {
        if ($WorkingDirectory) { Set-Location -LiteralPath $WorkingDirectory }
        $Result = Invoke-BootstrapCapturedCommand `
            -FilePath $FilePath `
            -Arguments $Arguments `
            -LiveOutput:$LiveOutput
        if (
            $env:COMIC_BOOTSTRAP_DETAIL_LOG -and
            -not [string]::IsNullOrWhiteSpace([string]$Result.Output)
        ) {
            Add-Content -LiteralPath $env:COMIC_BOOTSTRAP_DETAIL_LOG -Encoding UTF8 -Value @(
                "[$(Get-Date -Format o)] $FilePath $($Arguments -join ' ')"
                [string]$Result.Output
                ''
            )
        }
        if ($Result.ExitCode -ne 0) {
            throw "Command failed with exit code $($Result.ExitCode): $([System.IO.Path]::GetFileName($FilePath))."
        }
        if ($ShowOutput -and -not [string]::IsNullOrWhiteSpace([string]$Result.Output)) {
            Write-Host ([string]$Result.Output).Trim()
        }
    }
    finally { Set-Location $Previous }
}

function Invoke-BootstrapProbe {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$Arguments = @()
    )
    return [int](
        Invoke-BootstrapCapturedCommand -FilePath $FilePath -Arguments $Arguments
    ).ExitCode
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
    $Probe = "import platform,struct,sys,venv; print('|'.join((sys.executable,str(sys.version_info[0]),str(sys.version_info[1]),str(struct.calcsize('P')*8),platform.python_implementation())))"
    $Failures = [System.Collections.Generic.List[string]]::new()
    foreach ($Candidate in $Candidates) {
        $Command = Get-Command $Candidate.File -ErrorAction SilentlyContinue
        if ($null -eq $Command) { continue }
        try {
            $PythonArguments = @(
                @($Candidate.Prefix) |
                    Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) }
            ) + @('-I', '-c', ('"' + $Probe + '"'))
            $StartInfo = [System.Diagnostics.ProcessStartInfo]::new()
            $StartInfo.FileName = $Command.Source
            $StartInfo.Arguments = $PythonArguments -join ' '
            $StartInfo.UseShellExecute = $false
            $StartInfo.CreateNoWindow = $true
            $StartInfo.RedirectStandardOutput = $true
            $StartInfo.RedirectStandardError = $true
            $Process = [System.Diagnostics.Process]::new()
            $Process.StartInfo = $StartInfo
            [void]$Process.Start()
            $Output = $Process.StandardOutput.ReadToEnd().Trim()
            $ErrorOutput = $Process.StandardError.ReadToEnd().Trim()
            $Process.WaitForExit()
            $ExitCode = $Process.ExitCode
            $Process.Dispose()
            if ($ExitCode -ne 0 -or -not $Output) {
                $Failures.Add("$($Command.Source): probe failed (exit=$ExitCode): $ErrorOutput")
                continue
            }
            $Parts = $Output -split '\|', 5
            if ($Parts.Count -ne 5) {
                $Failures.Add("$($Command.Source): unexpected probe output: $Output")
                continue
            }
            $Payload = [pscustomobject]@{
                exe = $Parts[0]
                major = [int]$Parts[1]
                minor = [int]$Parts[2]
                bits = [int]$Parts[3]
                implementation = $Parts[4]
            }
            # requirements-base.txt pins mahotas==1.4.18, whose newest Windows wheel
            # is cp312. A 3.13+ interpreter would fall back to a source build, so the
            # pinned minor is a hard requirement here, not merely a preference.
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
            $Failures.Add(
                ("{0}: found {1} {2}.{3} ({4}-bit), expected CPython 3.12 x64" -f
                    $Command.Source,
                    $Payload.implementation,
                    $Payload.major,
                    $Payload.minor,
                    $Payload.bits)
            )
        }
        catch { $Failures.Add("$($Command.Source): $($_.Exception.Message)") }
    }
    $Detail = if ($Failures.Count -gt 0) { " Candidates: " + ($Failures -join '; ') } else { '' }
    throw ('Python 3.12 x64 (CPython) is required. Install the official 64-bit Python 3.12.10 package and enable the py launcher: https://www.python.org/downloads/release/python-31210/.' + $Detail)
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
    return (Invoke-BootstrapProbe -FilePath $Docker -Arguments @('info', '--format', '{{.ServerVersion}}')) -eq 0
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
    if ((Invoke-BootstrapProbe -FilePath $Docker -Arguments @('compose', 'version')) -ne 0) {
        throw 'Docker Compose v2 is required.'
    }
}

function Assert-NvidiaHost {
    $Command = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
    if ($null -eq $Command) { $Command = Get-Command 'nvidia-smi' -ErrorAction SilentlyContinue }
    if ($null -eq $Command) { throw 'NVIDIA driver tools were not found (nvidia-smi).' }
    if ((Invoke-BootstrapProbe -FilePath $Command.Source -Arguments @('-L')) -ne 0) {
        throw 'NVIDIA GPU/driver validation failed (nvidia-smi -L).'
    }
}

function ConvertTo-BootstrapNativeArgument {
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

function Invoke-BootstrapCapturedCommand {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$Arguments = @(),
        [switch]$LiveOutput
    )

    $ArgumentText = ($Arguments | ForEach-Object {
        ConvertTo-BootstrapNativeArgument -Argument $_
    }) -join ' '
    return [ComicBootstrapProcess]::Run(
        $FilePath,
        $ArgumentText,
        (Get-Location).Path,
        [bool]$LiveOutput
    )
}

function Get-NvidiaCudaCompatibilityVersion {
    $Command = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
    if ($null -eq $Command) { $Command = Get-Command 'nvidia-smi' -ErrorAction SilentlyContinue }
    if ($null -eq $Command) { throw 'NVIDIA driver tools were not found (nvidia-smi).' }

    $Result = Invoke-BootstrapCapturedCommand -FilePath $Command.Source
    if ($Result.ExitCode -ne 0) {
        throw 'Unable to read NVIDIA driver CUDA compatibility from nvidia-smi.'
    }

    $Match = [regex]::Match($Result.Output, 'CUDA Version:\s*(?<version>\d+(?:\.\d+){1,2})')
    if (-not $Match.Success) {
        throw 'nvidia-smi did not report the NVIDIA driver CUDA compatibility version.'
    }
    return [version]$Match.Groups['version'].Value
}

function Get-BootstrapDockerImageCudaCompatibility {
    param(
        [Parameter(Mandatory = $true)][string]$Docker,
        [Parameter(Mandatory = $true)][string]$Image,
        [Parameter(Mandatory = $true)][version]$HostCudaVersion
    )

    if ((Invoke-BootstrapProbe -FilePath $Docker -Arguments @('image', 'inspect', $Image)) -ne 0) {
        Write-BootstrapMessage "Pulling llama.cpp image for compatibility inspection: $Image"
        Invoke-BootstrapCommand -FilePath $Docker -Arguments @('pull', $Image)
    }

    $Inspect = Invoke-BootstrapCapturedCommand -FilePath $Docker -Arguments @(
        'image', 'inspect', '--format', '{{.Id}}|{{json .Config.Env}}', $Image
    )
    if ($Inspect.ExitCode -ne 0 -or [string]::IsNullOrWhiteSpace($Inspect.Output)) {
        throw "Unable to inspect llama.cpp image CUDA requirements: $Image"
    }
    $InspectParts = $Inspect.Output -split '\|', 2
    if ($InspectParts.Count -ne 2 -or [string]::IsNullOrWhiteSpace($InspectParts[0])) {
        throw "Unable to inspect llama.cpp image identity: $Image"
    }

    try { $Environment = $InspectParts[1] | ConvertFrom-Json }
    catch { throw "Unable to parse llama.cpp image environment: $Image" }
    $Requirement = @($Environment | Where-Object {
        [string]$_ -match '^NVIDIA_REQUIRE_CUDA='
    }) | Select-Object -First 1
    $RequiredVersion = $null
    if ($Requirement) {
        $Match = [regex]::Match([string]$Requirement, '(?:=|\s)cuda>=(?<version>\d+(?:\.\d+){1,2})')
        if ($Match.Success) { $RequiredVersion = [version]$Match.Groups['version'].Value }
    }

    return [pscustomobject]@{
        Image = $Image
        ImageId = $InspectParts[0].Trim()
        HostCudaVersion = $HostCudaVersion
        RequiredCudaVersion = $RequiredVersion
        Compatible = $null -eq $RequiredVersion -or $HostCudaVersion -ge $RequiredVersion
    }
}

function Test-BootstrapManagedRuntimeState {
    param(
        [Parameter(Mandatory = $true)][string]$Docker,
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$ImageRef,
        [Parameter(Mandatory = $true)][string]$ImageId,
        [Parameter(Mandatory = $true)][object[]]$ManagedRuntimes
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    try {
        $State = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
        if (
            [int]$State.schema_version -ne 1 -or
            [string]$State.image_ref -ne $ImageRef -or
            [string]$State.image_id -ne $ImageId
        ) {
            return $false
        }
        # Superset semantics: the seal record may cover more runtimes than this
        # invocation asks for (setup_full followed by setup). Every requested
        # runtime must be recorded, but extras are not a mismatch.
        if (@($State.volumes).Count -lt @($ManagedRuntimes).Count) { return $false }
        foreach ($Managed in $ManagedRuntimes) {
            $Recorded = @($State.volumes | Where-Object {
                [string]$_.name -eq [string]$Managed.volume
            })
            if ($Recorded.Count -ne 1) { return $false }
            if (
                [string]$Recorded[0].ready_manifest -ne
                [string]$Managed.ready_manifest
            ) {
                return $false
            }
            $Inspect = Invoke-BootstrapCapturedCommand -FilePath $Docker -Arguments @(
                'volume', 'inspect', '--format', '{{json .Labels}}', [string]$Managed.volume
            )
            if ($Inspect.ExitCode -ne 0 -or [string]::IsNullOrWhiteSpace($Inspect.Output)) {
                return $false
            }
            $Labels = $Inspect.Output | ConvertFrom-Json
            if (
                [string]$Labels.'comic-translate.runtime' -ne [string]$Managed.runtime_name -or
                [int]$Labels.'comic-translate.preparation-version' -ne [int]$Managed.preparation_version
            ) {
                return $false
            }
            $ManifestProbe = Invoke-BootstrapCapturedCommand -FilePath $Docker -Arguments @(
                'run', '--rm', '--pull', 'never',
                '-e', "READY_MANIFEST=$([string]$Managed.ready_manifest)",
                '--mount', (
                    "type=volume,source=$([string]$Managed.volume)," +
                    'target=/models,readonly'
                ),
                '--entrypoint', '/bin/sh', $ImageRef,
                '-ec', 'cat "/models/$READY_MANIFEST"'
            )
            if (
                $ManifestProbe.ExitCode -ne 0 -or
                [string]::IsNullOrWhiteSpace([string]$ManifestProbe.Output)
            ) {
                return $false
            }
            try {
                $ManifestText = ([string]$ManifestProbe.Output).TrimStart([char]0xFEFF)
                $Manifest = $ManifestText | ConvertFrom-Json
            }
            catch { return $false }
            if (
                $Manifest.ready -ne $true -or
                [string]$Manifest.runtime -ne [string]$Managed.runtime_name -or
                [int]$Manifest.preparation_version -ne [int]$Managed.preparation_version -or
                [string]$Manifest.source_image_ref -ne $ImageRef -or
                [string]$Manifest.source_image_id -ne $ImageId -or
                $Manifest.smoke_test.passed -ne $true -or
                @($Manifest.files).Count -lt 1
            ) {
                return $false
            }
        }
        return $true
    }
    catch { return $false }
}

function Write-BootstrapManagedRuntimeState {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$ImageRef,
        [Parameter(Mandatory = $true)][string]$ImageId,
        [Parameter(Mandatory = $true)][object[]]$ManagedRuntimes
    )

    $Directory = Split-Path -Parent $Path
    New-Item -ItemType Directory -Force -Path $Directory | Out-Null
    $Entries = [System.Collections.Generic.List[object]]::new()
    $Written = [System.Collections.Generic.HashSet[string]]::new(
        [System.StringComparer]::OrdinalIgnoreCase
    )
    foreach ($Managed in $ManagedRuntimes) {
        [void]$Written.Add([string]$Managed.volume)
        $Entries.Add([ordered]@{
            name = [string]$Managed.volume
            runtime_name = [string]$Managed.runtime_name
            preparation_version = [int]$Managed.preparation_version
            ready_manifest = [string]$Managed.ready_manifest
        })
    }
    # A narrower run (setup) must not erase a wider seal (setup_full). Carry over
    # previously recorded runtimes as long as they were sealed against this exact
    # image identity; a different image invalidates them and they are dropped.
    if (Test-Path -LiteralPath $Path -PathType Leaf) {
        try {
            $Previous = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
            if (
                [int]$Previous.schema_version -eq 1 -and
                [string]$Previous.image_ref -eq $ImageRef -and
                [string]$Previous.image_id -eq $ImageId
            ) {
                foreach ($Entry in @($Previous.volumes)) {
                    $Name = [string]$Entry.name
                    if ([string]::IsNullOrWhiteSpace($Name)) { continue }
                    if ($Written.Contains($Name)) { continue }
                    [void]$Written.Add($Name)
                    $Entries.Add([ordered]@{
                        name = $Name
                        runtime_name = [string]$Entry.runtime_name
                        preparation_version = [int]$Entry.preparation_version
                        ready_manifest = [string]$Entry.ready_manifest
                    })
                }
            }
        }
        catch { }
    }
    $Payload = [ordered]@{
        schema_version = 1
        image_ref = $ImageRef
        image_id = $ImageId
        volumes = @($Entries)
    }
    $TemporaryPath = "$Path.$PID.$([Guid]::NewGuid().ToString('N')).tmp"
    try {
        $Payload | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $TemporaryPath -Encoding UTF8
        Move-Item -LiteralPath $TemporaryPath -Destination $Path -Force
    }
    finally {
        Remove-Item -LiteralPath $TemporaryPath -Force -ErrorAction SilentlyContinue
    }
}

function Stop-BootstrapManagedContainer {
    param(
        [Parameter(Mandatory = $true)][string]$Docker,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$OwnershipLabel
    )
    if ((Invoke-BootstrapProbe -FilePath $Docker -Arguments @('inspect', $Name)) -ne 0) {
        return
    }
    $PreviousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $Raw = & $Docker inspect --format '{{json .Config.Labels}}|{{.State.Running}}' $Name 2>$null
        $ObservedExitCode = Get-Variable -Name LASTEXITCODE -ValueOnly -ErrorAction SilentlyContinue
    }
    finally { $ErrorActionPreference = $PreviousPreference }
    if (($null -ne $ObservedExitCode -and [int]$ObservedExitCode -ne 0) -or -not $Raw) { return }
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
    if (@($LibraryPaths).Count -gt 0) { $env:PATH = ((@($LibraryPaths) + @($env:PATH)) -join ';') }
}

Export-ModuleMember -Function @(
    'Write-BootstrapMessage', 'Write-BootstrapStage',
    'Invoke-BootstrapRetry', 'Invoke-BootstrapCommand', 'Invoke-BootstrapProbe',
    'Resolve-BootstrapPython312',
    'Enter-BootstrapLock', 'Test-BootstrapWritableDirectory', 'Assert-BootstrapFreeSpace',
    'Get-DockerExecutable', 'Test-DockerReady', 'Ensure-DockerDesktopReady',
    'Assert-DockerCompose', 'Assert-NvidiaHost', 'Get-NvidiaCudaCompatibilityVersion',
    'Get-BootstrapDockerImageCudaCompatibility', 'Test-BootstrapManagedRuntimeState',
    'Write-BootstrapManagedRuntimeState', 'Stop-BootstrapManagedContainer',
    'Set-BootstrapRuntimeEnvironment'
)
