<#
.SYNOPSIS
  End-to-end Windows environment verification for Audio8-VC (Inno Setup installer + Portable ZIP).

.DESCRIPTION
  Runs comprehensive pre-release verification against built Windows CPU release artifacts on
  both Windows 10 baseline (`windows-2022`) and Windows 11 24H2 baseline (`windows-2025`):
    1. Ensures real Audio8-0.1B ONNX INT8, SenseVoice ASR, and Silero VAD models are present
       in `$ModelsCacheDir` (downloading only on cache miss).
    2. Silently installs `*-setup.exe`, validates bundled binaries, brand icons, and
       `onnxruntime.dll` unification, then executes `scripts/e2e_verify.py` (CLI, CP1252 pipe,
       self-check, live server HTTP/WS, real-model TTS/ASR/VAD/enrollment, and real headless
       Microsoft Edge/Chromium fake-microphone `/ws/vc` browser E2E).
    3. Tests in-place upgrade/reinstall (`[InstallDelete]` cleanup of `_internal`) and silent
       uninstallation (`unins000.exe`).
    4. Extracts `*-portable.zip`, links `.\data\models` via NTFS junction with zero `A8VC_*`
       env overrides, and runs `scripts/e2e_verify.py --portable` (including headless browser E2E)
       plus verifies isolated `.\data\voices` auto-seeding.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$ArtifactDir,
    [Parameter(Mandatory = $true)][string]$ModelsCacheDir,
    [string]$Provider = "cpu",
    [string]$OrtVersion = "1.24.4",
    [string]$WorkRoot = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
Push-Location $RepoRoot

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Exe,
        [string[]]$ArgsList = @()
    )
    & $Exe @ArgsList
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Exe $($ArgsList -join ' ')"
    }
}

function Assert-BundledLayout {
    param([Parameter(Mandatory = $true)][string]$AppDir)

    $guiExe = Join-Path $AppDir "Audio8-VC.exe"
    $cliExe = Join-Path $AppDir "a8vc.exe"
    foreach ($p in @($guiExe, $cliExe)) {
        if (-not (Test-Path $p)) {
            throw "Missing expected executable: $p"
        }
    }

    # Verify PE ("MZ") header and non-trivial size for Audio8-VC.exe
    $guiInfo = Get-Item $guiExe
    if ($guiInfo.Length -lt 100000) {
        throw "Audio8-VC.exe appears unexpectedly small ($($guiInfo.Length) bytes)"
    }
    $header = [System.IO.File]::ReadAllBytes($guiExe)[0..1]
    if ($header[0] -ne 0x4D -or $header[1] -ne 0x5A) {
        throw "Audio8-VC.exe does not have a valid PE MZ header"
    }

    # Verify brand icon assets are bundled in both assets/ and server/static/
    $requiredAssets = @(
        "_internal\assets\audio8_vc.ico",
        "_internal\assets\audio8_vc.png",
        "_internal\assets\audio8_vc.svg",
        "_internal\a8vc\server\static\favicon.ico",
        "_internal\a8vc\server\static\icon.png",
        "_internal\a8vc\server\static\icon.svg"
    )
    foreach ($rel in $requiredAssets) {
        $full = Join-Path $AppDir $rel
        if (-not (Test-Path $full) -or (Get-Item $full).Length -le 0) {
            throw "Missing or empty bundled icon asset: $full"
        }
    }

    # Verify onnxruntime.dll unification (every bundled copy must have identical SHA-256)
    $ortDlls = @(Get-ChildItem -Path $AppDir -Recurse -Filter "onnxruntime.dll" -File)
    if ($ortDlls.Count -eq 0) {
        throw "No onnxruntime.dll found inside $AppDir"
    }
    $hashes = @($ortDlls | ForEach-Object { (Get-FileHash -Algorithm SHA256 $_.FullName).Hash } | Select-Object -Unique)
    if ($hashes.Count -ne 1) {
        $detail = ($ortDlls | ForEach-Object { "$($_.FullName) [$( (Get-FileHash -Algorithm SHA256 $_.FullName).Hash.Substring(0, 12) )]" }) -join "; "
        throw "Conflicting onnxruntime.dll copies detected in bundle: $detail"
    }
    Write-Host "  Verified executables, 6 brand icon assets, and unified onnxruntime.dll ($($ortDlls.Count) file(s), sha256=$($hashes[0].Substring(0, 12))...)" -ForegroundColor Cyan
}

function Invoke-DesktopGuiSmokeTest {
    param(
        [Parameter(Mandatory = $true)][string]$GuiExe,
        [Parameter(Mandatory = $true)][string]$ReportPath,
        [string]$DataDir = "",
        [int]$ExpectedMinUnblocked = 0
    )

    if (-not (Test-Path $GuiExe)) {
        throw "Expected GUI executable missing: $GuiExe"
    }
    if (Test-Path $ReportPath) {
        Remove-Item -Force $ReportPath
    }

    $prevDataDir = $env:A8VC_DATA_DIR
    $prevNoDialog = $env:A8VC_NO_DIALOG
    try {
        if ($DataDir) {
            $env:A8VC_DATA_DIR = $DataDir
        } else {
            Remove-Item Env:A8VC_DATA_DIR -ErrorAction SilentlyContinue
        }
        $env:A8VC_NO_DIALOG = "1"

        $proc = Start-Process -FilePath $GuiExe -ArgumentList @(
            "--smoke-test",
            "--smoke-timeout", "60",
            "--smoke-report", "`"$ReportPath`""
        ) -PassThru
        $null = $proc.Handle
        if (-not $proc.WaitForExit(90000)) {
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            throw "Audio8-VC.exe --smoke-test timed out after 90s ($GuiExe)"
        }
        $proc.WaitForExit()
        if (-not (Test-Path $ReportPath)) {
            throw "Audio8-VC.exe --smoke-test exited with code $($proc.ExitCode) without creating $ReportPath"
        }
        $raw = Get-Content -Raw $ReportPath
        Write-Host "  GUI smoke report ($GuiExe): $raw" -ForegroundColor Cyan
        $rep = $raw | ConvertFrom-Json
        if ($proc.ExitCode -ne 0 -or -not $rep.ok -or -not $rep.shown -or -not $rep.loaded -or -not $rep.icon_ok) {
            if ($rep.log_file -and (Test-Path $rep.log_file)) {
                Write-Host "  --- Tail of a8vc-desktop.log ---" -ForegroundColor Yellow
                Get-Content -Tail 60 $rep.log_file
            }
            throw "Audio8-VC.exe GUI smoke test failed (exit=$($proc.ExitCode), error=$($rep.error))"
        }
        if ($ExpectedMinUnblocked -gt 0 -and ([int]$rep.unblocked_dlls -lt $ExpectedMinUnblocked)) {
            throw "Expected Audio8-VC.exe to unblock >= $ExpectedMinUnblocked MOTW DLLs, got $($rep.unblocked_dlls)"
        }
        Write-Host "  Verified Audio8-VC.exe native window launched & loaded UI (renderer=$($rep.renderer), unblocked_dlls=$($rep.unblocked_dlls))." -ForegroundColor Cyan
    }
    finally {
        if ($null -ne $prevDataDir) {
            $env:A8VC_DATA_DIR = $prevDataDir
        } else {
            Remove-Item Env:A8VC_DATA_DIR -ErrorAction SilentlyContinue
        }
        if ($null -ne $prevNoDialog) {
            $env:A8VC_NO_DIALOG = $prevNoDialog
        } else {
            Remove-Item Env:A8VC_NO_DIALOG -ErrorAction SilentlyContinue
        }
    }
}

function New-SafeJunction {
    param(
        [Parameter(Mandatory = $true)][string]$LinkPath,
        [Parameter(Mandatory = $true)][string]$TargetPath
    )
    if (Test-Path $LinkPath) {
        [System.IO.Directory]::Delete($LinkPath)
    }
    $parent = Split-Path -Parent $LinkPath
    if (-not (Test-Path $parent)) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
    New-Item -ItemType Junction -Path $LinkPath -Target $TargetPath | Out-Null
}

function Remove-SafeJunction {
    param([string]$LinkPath)
    if ($LinkPath -and (Test-Path $LinkPath)) {
        try {
            [System.IO.Directory]::Delete($LinkPath)
        } catch {
            Write-Warning "Could not remove junction ${LinkPath}: $_"
        }
    }
}

$JunctionsToClean = New-Object System.Collections.Generic.List[string]
$CreatedWorkRoot = $false

try {
    $osCaption = (Get-CimInstance Win32_OperatingSystem).Caption
    $osVersion = [System.Environment]::OSVersion.VersionString
    Write-Host "=== Audio8-VC Windows E2E Matrix Verification ===" -ForegroundColor Cyan
    Write-Host "OS          : $osCaption ($osVersion)"
    Write-Host "CPU Cores   : $([System.Environment]::ProcessorCount)"
    Write-Host "ArtifactDir : $ArtifactDir"
    Write-Host "ModelsCache : $ModelsCacheDir"
    Write-Host "Provider    : $Provider (ORT $OrtVersion)"

    $ResolvedArtifacts = (Resolve-Path $ArtifactDir).Path
    $SetupExe = Get-ChildItem -Path $ResolvedArtifacts -Recurse -Filter "*-win64-$Provider-setup.exe" -File | Select-Object -First 1
    $PortableZip = Get-ChildItem -Path $ResolvedArtifacts -Recurse -Filter "*-win64-$Provider-portable.zip" -File | Select-Object -First 1
    if (-not $SetupExe) {
        throw "Expected Inno Setup installer (*-win64-$Provider-setup.exe) not found under $ResolvedArtifacts"
    }
    if (-not $PortableZip) {
        throw "Expected Portable ZIP (*-win64-$Provider-portable.zip) not found under $ResolvedArtifacts"
    }
    Write-Host "Setup EXE   : $($SetupExe.FullName)"
    Write-Host "Portable ZIP: $($PortableZip.FullName)"

    if ([string]::IsNullOrWhiteSpace($WorkRoot)) {
        $baseTemp = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { [System.IO.Path]::GetTempPath() }
        $WorkRoot = Join-Path $baseTemp ("a8vc_e2e_" + [System.Guid]::NewGuid().ToString("N").Substring(0, 8))
        $CreatedWorkRoot = $true
    }
    New-Item -ItemType Directory -Force -Path $WorkRoot | Out-Null
    New-Item -ItemType Directory -Force -Path $ModelsCacheDir | Out-Null
    $ResolvedModelsCache = (Resolve-Path $ModelsCacheDir).Path

    # 1. Ensure real Audio8-0.1B + registration + SenseVoice ASR + Silero VAD exist in cache
    $a8Manifest = Join-Path $ResolvedModelsCache "audio8-tts-0.1b-onnx-int8\runtime_manifest.json"
    $regEncoder = Join-Path $ResolvedModelsCache "audio8-tts-0.1b-onnx-int8\registration\codec_encoder_fp16.onnx"
    $vadFile = Join-Path $ResolvedModelsCache "vad\silero_vad.onnx"
    $asrDir = Join-Path $ResolvedModelsCache "asr"
    $hasAsr = (
        (Test-Path $asrDir) `
        -and (@(Get-ChildItem -Path $asrDir -Recurse -Filter "tokens.txt" -File).Count -gt 0) `
        -and (@(Get-ChildItem -Path $asrDir -Recurse -Filter "*.onnx" -File).Count -gt 0)
    )

    if (-not ((Test-Path $a8Manifest) -and (Test-Path $regEncoder) -and (Test-Path $vadFile) -and $hasAsr)) {
        Write-Host "`n[Step 1/4] Downloading real models into cache ($ResolvedModelsCache)..." -ForegroundColor Green
        Invoke-Checked "uv" @(
            "run", "python", "scripts\download_models.py",
            "--audio8-onnx", "--asr-sensevoice", "--silero-vad",
            "--models-root", $ResolvedModelsCache
        )
    } else {
        Write-Host "`n[Step 1/4] Reusing cached real models from $ResolvedModelsCache" -ForegroundColor Green
    }

    # 2. Inno Setup Installer silent install + full real-model & headless browser E2E
    Write-Host "`n[Step 2/4] Verifying Inno Setup Installer ($($SetupExe.Name))..." -ForegroundColor Green
    $InstallDir = Join-Path $WorkRoot "installed_app"
    $InstallDataDir = Join-Path $WorkRoot "installed_data"
    New-Item -ItemType Directory -Force -Path $InstallDataDir | Out-Null

    $proc = Start-Process -FilePath $SetupExe.FullName -ArgumentList @(
        "/VERYSILENT",
        "/SUPPRESSMSGBOXES",
        "/NORESTART",
        "/CURRENTUSER",
        "/DIR=$InstallDir"
    ) -Wait -PassThru
    if ($proc.ExitCode -ne 0) {
        throw "Silent installer exited with code $($proc.ExitCode)"
    }
    Assert-BundledLayout -AppDir $InstallDir

    $InstallModelsJunction = Join-Path $InstallDataDir "models"
    New-SafeJunction -LinkPath $InstallModelsJunction -TargetPath $ResolvedModelsCache
    $JunctionsToClean.Add($InstallModelsJunction)

    $InstalledCli = Join-Path $InstallDir "a8vc.exe"
    $InstalledGui = Join-Path $InstallDir "Audio8-VC.exe"
    Invoke-DesktopGuiSmokeTest `
        -GuiExe $InstalledGui `
        -ReportPath (Join-Path $WorkRoot "installed_gui_smoke.json") `
        -DataDir $InstallDataDir

    Invoke-Checked "uv" @(
        "run", "python", "scripts\e2e_verify.py",
        "--cli", $InstalledCli,
        "--data-dir", $InstallDataDir,
        "--expected-provider", $Provider,
        "--expected-ort-version", $OrtVersion,
        "--work-dir", (Join-Path $WorkRoot "work_installed")
    )

    # 3. Verify in-place upgrade ([InstallDelete] cleans _internal) & silent uninstall
    Write-Host "`n[Step 3/4] Verifying in-place upgrade ([InstallDelete]) and silent uninstall..." -ForegroundColor Green
    $StaleSentinel = Join-Path $InstallDir "_internal\__stale_upgrade_sentinel__.dll"
    [System.IO.File]::WriteAllText($StaleSentinel, "stale-dll-placeholder")
    $reinstallProc = Start-Process -FilePath $SetupExe.FullName -ArgumentList @(
        "/VERYSILENT",
        "/SUPPRESSMSGBOXES",
        "/NORESTART",
        "/CURRENTUSER",
        "/DIR=$InstallDir"
    ) -Wait -PassThru
    if ($reinstallProc.ExitCode -ne 0) {
        throw "In-place reinstall exited with code $($reinstallProc.ExitCode)"
    }
    if (Test-Path $StaleSentinel) {
        throw "In-place upgrade failed to clean _internal via [InstallDelete]: $StaleSentinel still exists"
    }
    Write-Host "  Verified [InstallDelete] removed stale _internal file during in-place upgrade." -ForegroundColor Cyan
    Invoke-Checked $InstalledCli @("self-check")

    $Uninstaller = Join-Path $InstallDir "unins000.exe"
    if (-not (Test-Path $Uninstaller)) {
        throw "Expected uninstaller missing: $Uninstaller"
    }
    $uninstProc = Start-Process -FilePath $Uninstaller -ArgumentList @("/VERYSILENT", "/NORESTART") -Wait -PassThru
    if ($uninstProc.ExitCode -ne 0) {
        throw "Silent uninstaller exited with code $($uninstProc.ExitCode)"
    }
    Start-Sleep -Seconds 2
    if (Test-Path (Join-Path $InstallDir "Audio8-VC.exe")) {
        throw "Audio8-VC.exe still present after silent uninstall under $InstallDir"
    }
    Write-Host "  Verified silent uninstallation cleanly removed installed binaries." -ForegroundColor Cyan

    # 4. Portable ZIP isolated mode + real-model & headless browser E2E
    Write-Host "`n[Step 4/4] Verifying Portable ZIP ($($PortableZip.Name))..." -ForegroundColor Green
    $PortableExtract = Join-Path $WorkRoot "portable_extract"
    Expand-Archive -Path $PortableZip.FullName -DestinationPath $PortableExtract -Force

    $PortableCliItem = Get-ChildItem -Path $PortableExtract -Recurse -Filter "a8vc.exe" -File | Select-Object -First 1
    if (-not $PortableCliItem) {
        throw "Extracted portable ZIP missing a8vc.exe under $PortableExtract"
    }
    $PortableDir = $PortableCliItem.DirectoryName
    if (-not (Test-Path (Join-Path $PortableDir "portable.flag"))) {
        throw "Extracted portable ZIP missing portable.flag under $PortableDir"
    }
    Assert-BundledLayout -AppDir $PortableDir

    $PortableModelsJunction = Join-Path $PortableDir "data\models"
    New-SafeJunction -LinkPath $PortableModelsJunction -TargetPath $ResolvedModelsCache
    $JunctionsToClean.Add($PortableModelsJunction)

    # Simulate Windows Explorer Mark-of-the-Web (Zone.Identifier ADS) on downloaded Portable ZIP DLLs
    $MotwNames = @(
        "Python.Runtime.dll",
        "Microsoft.Web.WebView2.Core.dll",
        "Microsoft.Web.WebView2.WinForms.dll"
    )
    $MotwTargets = @(
        Get-ChildItem -Path (Join-Path $PortableDir "_internal") -Recurse -File -ErrorAction SilentlyContinue |
            Where-Object { $MotwNames -contains $_.Name }
    )
    $MotwApplied = 0
    foreach ($dllItem in $MotwTargets) {
        Set-Content -Path $dllItem.FullName -Stream "Zone.Identifier" -Value "[ZoneTransfer]`r`nZoneId=3`r`n" -Encoding Ascii
        $MotwApplied += 1
    }
    if ($MotwApplied -lt 3) {
        throw "Expected >= 3 managed DLLs ($($MotwNames -join ', ')) for MOTW Zone.Identifier verification, found $MotwApplied"
    }
    Write-Host "  Attached Mark-of-the-Web (Zone.Identifier ZoneId=3) to $MotwApplied portable DLLs." -ForegroundColor Yellow

    Invoke-DesktopGuiSmokeTest `
        -GuiExe (Join-Path $PortableDir "Audio8-VC.exe") `
        -ReportPath (Join-Path $WorkRoot "portable_gui_smoke.json") `
        -ExpectedMinUnblocked $MotwApplied

    Invoke-Checked "uv" @(
        "run", "python", "scripts\e2e_verify.py",
        "--cli", $PortableCliItem.FullName,
        "--portable",
        "--expected-provider", $Provider,
        "--expected-ort-version", $OrtVersion,
        "--skip-cli-synth",
        "--work-dir", (Join-Path $WorkRoot "work_portable")
    )

    $PortableVoices = @(Get-ChildItem -Path (Join-Path $PortableDir "data\voices") -Filter "*.a8voice" -File -ErrorAction SilentlyContinue)
    if ($PortableVoices.Count -lt 15) {
        throw "Expected >= 15 auto-seeded .a8voice packs inside portable .\data\voices, got $($PortableVoices.Count)"
    }
    Write-Host "  Verified portable mode isolated data in .\data\voices ($($PortableVoices.Count) .a8voice files)." -ForegroundColor Cyan

    Write-Host "`n=== Windows E2E Matrix Verification PASSED ($osCaption) ===" -ForegroundColor Green
}
finally {
    foreach ($j in $JunctionsToClean) {
        Remove-SafeJunction -LinkPath $j
    }
    if ($CreatedWorkRoot -and $WorkRoot -and (Test-Path $WorkRoot)) {
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $WorkRoot
    }
    Pop-Location
}
