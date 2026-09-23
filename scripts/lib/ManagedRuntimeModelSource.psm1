<#
.SYNOPSIS
관리형 llama.cpp 런타임 준비 스크립트가 공유하는 모델 원본 해결기.

.DESCRIPTION
`prepare_*_runtime.ps1` 다섯 개가 같은 순서로 모델 원본을 찾도록 한 곳에 모았다.

    1. 호출자가 명시한 경로 또는 디렉터리
    2. 저장소의 gitignore 된 `testmodel/`
    3. 다운로드 캐시 디렉터리
    4. 등록된 Hugging Face 원본에서 내려받기(재개 가능, SHA-256 검증)

크기와 SHA-256 이 계약과 정확히 같은 파일만 원본으로 인정한다. 내려받기는
`.partial` 로 받아 검증한 뒤에만 최종 이름으로 원자적으로 옮긴다. 절반만 받은
파일이 다음 실행에서 정상 원본으로 오인되는 일은 없다.
#>

Set-StrictMode -Version Latest

# Windows PowerShell 5.1 은 System.Net.Http 를 기본으로 올리지 않는다. 명시적으로
# 올리지 않으면 내려받기가 "Unable to find type [System.Net.Http.HttpClient]" 로
# 죽는다. 이미 올라와 있으면 아무 일도 일어나지 않는다.
Add-Type -AssemblyName System.Net.Http -ErrorAction SilentlyContinue

function Get-ManagedRuntimeRepositoryRoot {
    <#
    .SYNOPSIS
    이 모듈을 담고 있는 저장소 루트를 돌려준다.
    #>

    return (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
}

function Get-ManagedRuntimeDefaultSearchDirectory {
    <#
    .SYNOPSIS
    저장소가 관리하는 기본 모델 보관 디렉터리(`testmodel/`).

    .DESCRIPTION
    gitignore 대상이라 대용량 GGUF 를 두어도 저장소를 오염시키지 않는다.
    존재하지 않으면 빈 문자열을 돌려주고, 호출자는 그냥 건너뛴다.
    #>

    $Candidate = Join-Path (Get-ManagedRuntimeRepositoryRoot) 'testmodel'
    if (Test-Path -LiteralPath $Candidate -PathType Container) {
        return (Resolve-Path -LiteralPath $Candidate).Path
    }
    return ''
}

function Get-ManagedRuntimeFileSha256 {
    param(
        [Parameter(Mandatory = $true)][string]$Path
    )

    $Stream = [System.IO.File]::OpenRead($Path)
    $Hasher = [System.Security.Cryptography.SHA256]::Create()
    try {
        $HashBytes = $Hasher.ComputeHash($Stream)
        return ([BitConverter]::ToString($HashBytes) -replace '-', '').ToLowerInvariant()
    }
    finally {
        $Hasher.Dispose()
        $Stream.Dispose()
    }
}

function Test-ManagedRuntimeModelCandidate {
    <#
    .SYNOPSIS
    후보 파일이 계약(크기, SHA-256)을 만족하는지 본다.

    .DESCRIPTION
    크기는 항상 본다. 크기가 다르면 해시를 계산하지 않는다. 수 GB 파일을 헛되이
    읽지 않기 위해서다.
    #>

    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][int64]$Bytes,
        [Parameter(Mandatory = $true)][string]$Sha256,
        [switch]$SkipHash
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $false
    }
    $Item = Get-Item -LiteralPath $Path
    if ([int64]$Item.Length -ne $Bytes) {
        return $false
    }
    if ($SkipHash) {
        return $true
    }
    Write-Host "Checking source SHA-256: $Path"
    return (Get-ManagedRuntimeFileSha256 -Path $Path) -eq $Sha256.ToLowerInvariant()
}

function Invoke-ManagedRuntimeDownloadAttempt {
    <#
    .SYNOPSIS
    큰 파일을 재개 가능하게 내려받고 크기와 SHA-256 을 검증한다.

    .DESCRIPTION
    `.partial` 로 받는다. 이미 일부가 있으면 Range 요청으로 이어받는다.
    서버가 Range 를 무시하면(206 이 아닌 200) 처음부터 다시 받는다.
    검증을 통과한 뒤에만 최종 경로로 옮긴다.
    #>

    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][string]$Destination,
        [Parameter(Mandatory = $true)][int64]$Bytes,
        [Parameter(Mandatory = $true)][string]$Sha256,
        [int]$TimeoutMinutes = 240
    )

    $Directory = Split-Path -Parent $Destination
    if (-not (Test-Path -LiteralPath $Directory -PathType Container)) {
        New-Item -ItemType Directory -Force -Path $Directory | Out-Null
    }
    $Partial = "$Destination.partial"

    $Handler = [System.Net.Http.HttpClientHandler]::new()
    # HttpClientHandler can drop Range while automatically following the signed
    # Hugging Face redirect. Follow redirects explicitly so every hop receives
    # the same resume header and a multi-GiB retry does not restart at byte zero.
    $Handler.AllowAutoRedirect = $false
    $Client = [System.Net.Http.HttpClient]::new($Handler)
    $Client.Timeout = [TimeSpan]::FromMinutes($TimeoutMinutes)
    try {
        $Existing = 0L
        if (Test-Path -LiteralPath $Partial -PathType Leaf) {
            $Existing = [int64](Get-Item -LiteralPath $Partial).Length
            if ($Existing -gt $Bytes) {
                # 계약보다 큰 잔여물은 신뢰할 수 없다. 버리고 다시 받는다.
                Remove-Item -LiteralPath $Partial -Force
                $Existing = 0L
            } elseif ($Existing -eq $Bytes) {
                Write-Host "Checking completed partial SHA-256: $Destination"
                if ((Get-ManagedRuntimeFileSha256 -Path $Partial) -eq $Sha256.ToLowerInvariant()) {
                    Move-Item -LiteralPath $Partial -Destination $Destination -Force
                    return $Destination
                }
                Remove-Item -LiteralPath $Partial -Force
                $Existing = 0L
            }
        }

        if ($Existing -gt 0) {
            Write-Host (
                "Resuming download at {0:N2} GiB of {1:N2} GiB" -f
                ($Existing / 1GB),
                ($Bytes / 1GB)
            )
        }

        $RequestUri = [Uri]::new($Uri)
        $Request = $null
        $Response = $null
        for ($Redirect = 0; $Redirect -le 10; $Redirect++) {
            $Request = [System.Net.Http.HttpRequestMessage]::new(
                [System.Net.Http.HttpMethod]::Get,
                $RequestUri
            )
            if ($Existing -gt 0) {
                $Request.Headers.Range = [System.Net.Http.Headers.RangeHeaderValue]::new(
                    $Existing,
                    $null
                )
            }
            $Response = $Client.SendAsync(
                $Request,
                [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead
            ).GetAwaiter().GetResult()
            $StatusCode = [int]$Response.StatusCode
            $Location = $Response.Headers.Location
            if ($StatusCode -ge 300 -and $StatusCode -lt 400 -and $null -ne $Location) {
                $RequestUri = if ($Location.IsAbsoluteUri) {
                    $Location
                } else {
                    [Uri]::new($RequestUri, $Location)
                }
                $Response.Dispose()
                $Request.Dispose()
                $Response = $null
                $Request = $null
                continue
            }
            break
        }
        if ($null -eq $Response -or $null -eq $Request) {
            throw "Model download exceeded the redirect limit: $Uri"
        }
        try {
            if (-not $Response.IsSuccessStatusCode) {
                throw (
                    "Model download failed (HTTP {0}): {1}" -f
                    [int]$Response.StatusCode,
                    $Uri
                )
            }
            $Append = $true
            if ($Existing -gt 0 -and [int]$Response.StatusCode -ne 206) {
                # 서버가 Range 를 무시했다. 이어붙이면 파일이 깨진다.
                Write-Host 'The server ignored the resume request; restarting the download.'
                Remove-Item -LiteralPath $Partial -Force -ErrorAction SilentlyContinue
                $Existing = 0L
            }
            if ($Existing -le 0) {
                $Append = $false
            }

            $Mode = if ($Append) {
                [System.IO.FileMode]::Append
            }
            else {
                [System.IO.FileMode]::Create
            }
            $Source = $Response.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
            $Target = [System.IO.FileStream]::new(
                $Partial,
                $Mode,
                [System.IO.FileAccess]::Write,
                [System.IO.FileShare]::None,
                1048576
            )
            try {
                $Buffer = [byte[]]::new(4194304)
                $Written = $Existing
                $LastReport = [DateTime]::UtcNow
                while ($true) {
                    $Read = $Source.Read($Buffer, 0, $Buffer.Length)
                    if ($Read -le 0) {
                        break
                    }
                    $Target.Write($Buffer, 0, $Read)
                    $Written += $Read
                    if (([DateTime]::UtcNow - $LastReport).TotalSeconds -ge 15) {
                        Write-Host (
                            "Downloaded {0:N2} GiB of {1:N2} GiB ({2:N1}%)" -f
                            ($Written / 1GB),
                            ($Bytes / 1GB),
                            (100.0 * $Written / [double]$Bytes)
                        )
                        $LastReport = [DateTime]::UtcNow
                    }
                }
            }
            finally {
                $Target.Dispose()
                $Source.Dispose()
            }
        }
        finally {
            $Response.Dispose()
            $Request.Dispose()
        }
    }
    finally {
        $Client.Dispose()
        $Handler.Dispose()
    }

    $Downloaded = [int64](Get-Item -LiteralPath $Partial).Length
    if ($Downloaded -ne $Bytes) {
        throw (
            "Downloaded model size mismatch: {0}, expected={1}, actual={2}" -f
            $Uri,
            $Bytes,
            $Downloaded
        )
    }
    Write-Host "Checking downloaded SHA-256: $Destination"
    $ActualSha256 = Get-ManagedRuntimeFileSha256 -Path $Partial
    if ($ActualSha256 -ne $Sha256.ToLowerInvariant()) {
        Remove-Item -LiteralPath $Partial -Force -ErrorAction SilentlyContinue
        throw (
            "Downloaded model SHA-256 mismatch: {0}, expected={1}, actual={2}" -f
            $Uri,
            $Sha256.ToLowerInvariant(),
            $ActualSha256
        )
    }
    Move-Item -LiteralPath $Partial -Destination $Destination -Force
    return $Destination
}

function Invoke-ManagedRuntimeDownload {
    <#
    .SYNOPSIS
    재개 가능한 model download를 제한된 지수 backoff로 다시 시도한다.

    .DESCRIPTION
    각 attempt는 같은 `.partial` 파일을 사용하므로 이미 받은 byte를 버리지 않는다.
    인증·권한·파일 없음 오류는 재시도해도 복구되지 않으므로 즉시 올린다.
    #>
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][string]$Destination,
        [Parameter(Mandatory = $true)][int64]$Bytes,
        [Parameter(Mandatory = $true)][string]$Sha256,
        [int]$TimeoutMinutes = 240,
        [ValidateRange(1, 10)][int]$MaximumAttempts = 5
    )

    for ($Attempt = 1; $Attempt -le $MaximumAttempts; $Attempt++) {
        try {
            return Invoke-ManagedRuntimeDownloadAttempt `
                -Uri $Uri `
                -Destination $Destination `
                -Bytes $Bytes `
                -Sha256 $Sha256 `
                -TimeoutMinutes $TimeoutMinutes
        }
        catch {
            $Message = [string]$_.Exception.Message
            if ($Message -match 'HTTP (400|401|403|404|405|410|422)') {
                throw
            }
            if ($Attempt -ge $MaximumAttempts) {
                throw
            }
            $Delay = [int][Math]::Min(30, 2 * [Math]::Pow(2, $Attempt - 1))
            Write-Warning (
                "Model download attempt {0}/{1} failed. Resuming in {2}s: {3}" -f
                $Attempt,
                $MaximumAttempts,
                $Delay,
                $Message
            )
            Start-Sleep -Seconds $Delay
        }
    }
}

function Resolve-ManagedRuntimeModelSource {
    <#
    .SYNOPSIS
    준비 스크립트가 쓸 모델 원본 하나를 확정한다.

    .OUTPUTS
    Path, Directory, FileName, Origin, Verified 를 가진 객체.
    Origin 은 'provided', 'search', 'download' 중 하나다.
    Verified 가 참이면 이 함수가 이미 전체 SHA-256 을 확인했으므로 호출자는
    같은 파일을 다시 해시할 필요가 없다.
    #>

    param(
        [Parameter(Mandatory = $true)][string]$FileName,
        [Parameter(Mandatory = $true)][int64]$Bytes,
        [Parameter(Mandatory = $true)][string]$Sha256,

        # 사용자가 명시한 경로. 파일이면 그 파일만, 디렉터리면 그 아래 $FileName.
        [AllowEmptyString()][string]$RequestedPath = '',

        # 추가 탐색 디렉터리. 앞에 있는 것이 우선한다.
        [string[]]$SearchDirectory = @(),

        # 등록된 원본 URL. 비어 있으면 내려받기를 시도하지 않는다.
        [AllowEmptyString()][string]$DownloadUrl = '',

        # 내려받은 파일을 둘 위치. 기본값은 저장소의 `testmodel/`.
        [AllowEmptyString()][string]$DownloadDirectory = '',

        [switch]$AllowDownload,

        [switch]$SkipFreeSpaceCheck
    )

    $Sha256 = $Sha256.ToLowerInvariant()
    $Candidates = [System.Collections.Generic.List[string]]::new()

    if (-not [string]::IsNullOrWhiteSpace($RequestedPath)) {
        if (Test-Path -LiteralPath $RequestedPath -PathType Container) {
            $Candidates.Add((Join-Path (Resolve-Path -LiteralPath $RequestedPath).Path $FileName))
        }
        elseif (Test-Path -LiteralPath $RequestedPath -PathType Leaf) {
            $Candidates.Add((Resolve-Path -LiteralPath $RequestedPath).Path)
        }
        else {
            throw "Requested model source does not exist: $RequestedPath"
        }
    }

    $Directories = [System.Collections.Generic.List[string]]::new()
    foreach ($Directory in $SearchDirectory) {
        if (
            -not [string]::IsNullOrWhiteSpace($Directory) -and
            (Test-Path -LiteralPath $Directory -PathType Container)
        ) {
            $Directories.Add((Resolve-Path -LiteralPath $Directory).Path)
        }
    }
    $DefaultSearch = Get-ManagedRuntimeDefaultSearchDirectory
    if (-not [string]::IsNullOrWhiteSpace($DefaultSearch)) {
        $Directories.Add($DefaultSearch)
    }
    if (-not [string]::IsNullOrWhiteSpace($DownloadDirectory)) {
        if (Test-Path -LiteralPath $DownloadDirectory -PathType Container) {
            $Directories.Add((Resolve-Path -LiteralPath $DownloadDirectory).Path)
        }
    }
    foreach ($Directory in $Directories) {
        $Candidate = Join-Path $Directory $FileName
        if (-not $Candidates.Contains($Candidate)) {
            $Candidates.Add($Candidate)
        }
    }
    # 모델을 런타임별 하위 폴더에 나눠 두는 경우가 흔하다(예:
    # `testmodel/PaddleOCR-VL-1.6-GGUF/`). 한 단계만 더 본다. 그 아래로 재귀하면
    # 큰 트리에서 탐색이 비싸지고 어떤 파일을 골랐는지도 불투명해진다.
    foreach ($Directory in $Directories) {
        foreach ($Child in (Get-ChildItem -LiteralPath $Directory -Directory -ErrorAction SilentlyContinue)) {
            $Candidate = Join-Path $Child.FullName $FileName
            if (-not $Candidates.Contains($Candidate)) {
                $Candidates.Add($Candidate)
            }
        }
    }

    $Index = 0
    foreach ($Candidate in $Candidates) {
        $Origin = if ($Index -eq 0 -and -not [string]::IsNullOrWhiteSpace($RequestedPath)) {
            'provided'
        }
        else {
            'search'
        }
        $Index += 1
        if (-not (Test-Path -LiteralPath $Candidate -PathType Leaf)) {
            continue
        }
        $Item = Get-Item -LiteralPath $Candidate
        if ([int64]$Item.Length -ne $Bytes) {
            Write-Host (
                "Skipping a model candidate with the wrong size: {0} (expected={1}, actual={2})" -f
                $Candidate,
                $Bytes,
                $Item.Length
            )
            continue
        }
        if (-not (Test-ManagedRuntimeModelCandidate -Path $Candidate -Bytes $Bytes -Sha256 $Sha256)) {
            Write-Host "Skipping a model candidate with the wrong SHA-256: $Candidate"
            continue
        }
        Write-Host "Using the local model source: $Candidate"
        return [pscustomobject][ordered]@{
            Path = $Candidate
            Directory = Split-Path -Parent $Candidate
            FileName = Split-Path -Leaf $Candidate
            Origin = $Origin
            Verified = $true
        }
    }

    if (-not $AllowDownload) {
        throw (
            "No verified local source for {0}. Searched: {1}. " -f
            $FileName,
            (@($Candidates) -join '; ')
        ) + 'Pass -AllowDownload to fetch the registered source, or supply the file explicitly.'
    }
    if ([string]::IsNullOrWhiteSpace($DownloadUrl)) {
        throw (
            "No download source is registered for {0}, so it must be supplied locally. Searched: {1}" -f
            $FileName,
            (@($Candidates) -join '; ')
        )
    }

    $TargetDirectory = if ([string]::IsNullOrWhiteSpace($DownloadDirectory)) {
        $Fallback = Get-ManagedRuntimeDefaultSearchDirectory
        if ([string]::IsNullOrWhiteSpace($Fallback)) {
            $Fallback = Join-Path (Get-ManagedRuntimeRepositoryRoot) 'testmodel'
            New-Item -ItemType Directory -Force -Path $Fallback | Out-Null
        }
        $Fallback
    }
    else {
        if (-not (Test-Path -LiteralPath $DownloadDirectory -PathType Container)) {
            New-Item -ItemType Directory -Force -Path $DownloadDirectory | Out-Null
        }
        (Resolve-Path -LiteralPath $DownloadDirectory).Path
    }

    $Destination = Join-Path $TargetDirectory $FileName
    if (-not $SkipFreeSpaceCheck) {
        $DriveRoot = [System.IO.Path]::GetPathRoot($TargetDirectory)
        $Drive = [System.IO.DriveInfo]::new($DriveRoot)
        $RequiredBytes = $Bytes + 536870912L
        if ($Drive.AvailableFreeSpace -lt $RequiredBytes) {
            throw (
                "Not enough free space for {0}: required={1:N2} GiB, available={2:N2} GiB, drive={3}" -f
                $FileName,
                ($RequiredBytes / 1GB),
                ($Drive.AvailableFreeSpace / 1GB),
                $DriveRoot
            )
        }
    }
    Write-Host (
        "Downloading the registered model source ({0:N2} GiB): {1}" -f
        ($Bytes / 1GB),
        $DownloadUrl
    )
    Invoke-ManagedRuntimeDownload `
        -Uri $DownloadUrl `
        -Destination $Destination `
        -Bytes $Bytes `
        -Sha256 $Sha256 | Out-Null
    return [pscustomobject][ordered]@{
        Path = $Destination
        Directory = Split-Path -Parent $Destination
        FileName = Split-Path -Leaf $Destination
        Origin = 'download'
        Verified = $true
    }
}

Export-ModuleMember -Function @(
    'Get-ManagedRuntimeRepositoryRoot',
    'Get-ManagedRuntimeDefaultSearchDirectory',
    'Get-ManagedRuntimeFileSha256',
    'Test-ManagedRuntimeModelCandidate',
    'Invoke-ManagedRuntimeDownload',
    'Resolve-ManagedRuntimeModelSource'
)
