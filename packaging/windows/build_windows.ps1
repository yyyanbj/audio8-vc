<#
.SYNOPSIS
  One-click Windows build script for Audio8-VC (.exe + Portable ZIP + Inno Setup Installer).

.DESCRIPTION
  Builds the standalone Windows desktop application (`Audio8-VC.exe` + `a8vc.exe`) using
  PyInstaller onedir mode, creates a Portable ZIP archive (with `portable.flag`), and builds
  an Inno Setup 6 installer (`-setup.exe`) when ISCC.exe is available.

.PARAMETER Provider
  ONNX Runtime execution provider flavor to package: cpu (default), dml (DirectML GPU), or cuda.

.PARAMETER BundleModels
  If specified, downloads Audio8-0.1B ONNX INT8, SenseVoice ASR, and Silero VAD into the package
  for a fully offline all-in-one Windows distribution.

.PARAMETER UseMirror
  When combined with -BundleModels, downloads Hugging Face models via https://hf-mirror.com.

.PARAMETER SkipInstaller
  Skip building the Inno Setup installer (.exe).

.PARAMETER SkipPortableZip
  Skip creating the portable .zip package.

.PARAMETER Version
  Override the package version string (defaults to version in src/a8vc/__init__.py).
#>
[CmdletBinding()]
param(
    [ValidateSet("cpu", "dml", "cuda")]
    [string]$Provider = "cpu",
    [switch]$BundleModels,
    [switch]$UseMirror,
    [switch]$SkipInstaller,
    [switch]$SkipPortableZip,
    [string]$Version = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
Push-Location $RepoRoot

try {
    Write-Host "=== Audio8-VC Windows Packaging Pipeline ===" -ForegroundColor Cyan
    Write-Host "Repo root : $RepoRoot"
    Write-Host "Provider  : $Provider"

    # 1. Read version from src/a8vc/__init__.py if not provided
    if ([string]::IsNullOrWhiteSpace($Version)) {
        $initContent = Get-Content -Raw (Join-Path $RepoRoot "src\a8vc\__init__.py")
        if ($initContent -match '__version__\s*=\s*"([^"]+)"') {
            $Version = $Matches[1]
        } else {
            $Version = "0.1.0"
        }
    }
    Write-Host "Version   : $Version"

    # 2. Ensure Python / uv environment is ready
    $hasUv = $null -ne (Get-Command uv -ErrorAction SilentlyContinue)
    if ($hasUv) {
        Write-Host "`n[1/6] Syncing dependencies with uv (--extra windows --extra build)..." -ForegroundColor Green
        uv sync --extra windows --extra build
        if ($Provider -eq "dml") {
            Write-Host "Installing onnxruntime-directml for DirectML GPU acceleration..." -ForegroundColor Yellow
            uv pip uninstall onnxruntime | Out-Null
            uv pip install "onnxruntime-directml>=1.20,<1.24"
        } elseif ($Provider -eq "cuda") {
            Write-Host "Installing onnxruntime-gpu for CUDA acceleration..." -ForegroundColor Yellow
            uv pip uninstall onnxruntime | Out-Null
            uv pip install "onnxruntime-gpu>=1.22,<1.24"
        }
        $PyRun = @("uv", "run", "--no-sync")
    } else {
        Write-Host "`n[1/6] uv not found; using active python/pip environment..." -ForegroundColor Yellow
        python -m pip install --upgrade pip
        python -m pip install -e ".[windows,build]"
        if ($Provider -eq "dml") {
            python -m pip uninstall -y onnxruntime
            python -m pip install "onnxruntime-directml>=1.20,<1.24"
        } elseif ($Provider -eq "cuda") {
            python -m pip uninstall -y onnxruntime
            python -m pip install "onnxruntime-gpu>=1.22,<1.24"
        }
        $PyRun = @()
    }

    function Invoke-Py {
        param([string[]]$ArgsList)
        if ($PyRun.Count -gt 0) {
            & $PyRun[0] ($PyRun[1..($PyRun.Count - 1)] + $ArgsList)
        } else {
            & $ArgsList[0] $ArgsList[1..($ArgsList.Count - 1)]
        }
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed with exit code ${LASTEXITCODE}: $($ArgsList -join ' ')"
        }
    }

    # 3. Generate application icon assets
    Write-Host "`n[2/6] Generating Windows icon assets..." -ForegroundColor Green
    Invoke-Py @("python", "packaging\windows\generate_icon.py")

    # 4. Run PyInstaller
    Write-Host "`n[3/6] Building standalone executables with PyInstaller..." -ForegroundColor Green
    $DistAppDir = Join-Path $RepoRoot "dist\Audio8-VC"
    $ReleaseDir = Join-Path $RepoRoot "dist\release"
    if (Test-Path $DistAppDir) { Remove-Item -Recurse -Force $DistAppDir }
    New-Item -ItemType Directory -Force -Path $ReleaseDir | Out-Null

    Invoke-Py @(
        "pyinstaller",
        "--noconfirm",
        "--clean",
        "packaging\windows\audio8_vc.spec"
    )

    if (-not (Test-Path (Join-Path $DistAppDir "Audio8-VC.exe"))) {
        throw "Build failed: Audio8-VC.exe was not generated."
    }

    # Helper to write UTF-8 without BOM (safe across PowerShell 5.1 & 7+ for Python tomllib)
    $Utf8NoBom = New-Object System.Text.UTF8Encoding($false)

    # Patch bundled default.toml when building a GPU provider flavor (dml / cuda)
    if ($Provider -ne "cpu") {
        $cfgCandidates = @(
            (Join-Path $DistAppDir "_internal\configs\default.toml"),
            (Join-Path $DistAppDir "configs\default.toml")
        )
        foreach ($cfgFile in $cfgCandidates) {
            if (Test-Path $cfgFile) {
                $rawCfg = [System.IO.File]::ReadAllText($cfgFile, $Utf8NoBom)
                $patchedCfg = $rawCfg -replace '(?m)^provider\s*=\s*"cpu"', "provider = `"$Provider`""
                [System.IO.File]::WriteAllText($cfgFile, $patchedCfg, $Utf8NoBom)
                Write-Host "Configured packaged default provider = '$Provider' in $cfgFile" -ForegroundColor Yellow
            }
        }
    }

    # 5. Optional: download & bundle models for offline all-in-one package
    $VariantTag = $Provider
    if ($BundleModels) {
        Write-Host "`n[4/6] Downloading and bundling model weights into dist\Audio8-VC\models..." -ForegroundColor Green
        $dlArgs = @(
            "python", "scripts\download_models.py",
            "--audio8-onnx", "--asr-sensevoice", "--silero-vad",
            "--models-root", (Join-Path $DistAppDir "models")
        )
        if ($UseMirror) { $dlArgs += "--mirror" }
        Invoke-Py $dlArgs
        $VariantTag = "${Provider}-full"
    } else {
        Write-Host "`n[4/6] Skipping bundled model weights (online in-app download enabled)." -ForegroundColor DarkGray
    }

    # Smoke-test built CLI executable
    Write-Host "Running smoke test on built a8vc.exe..." -ForegroundColor Green
    & (Join-Path $DistAppDir "a8vc.exe") --version
    if ($LASTEXITCODE -ne 0) {
        throw "Smoke test failed: a8vc.exe --version returned exit code $LASTEXITCODE"
    }

    $Artifacts = @()

    # 6. Create Portable ZIP (includes portable.flag so all data stays inside ./data)
    if (-not $SkipPortableZip) {
        Write-Host "`n[5/6] Creating Portable ZIP package..." -ForegroundColor Green
        $PortableStage = Join-Path $RepoRoot "build\portable_stage\Audio8-VC"
        if (Test-Path $PortableStage) { Remove-Item -Recurse -Force $PortableStage }
        New-Item -ItemType Directory -Force -Path $PortableStage | Out-Null
        Copy-Item -Recurse -Force (Join-Path $DistAppDir "*") $PortableStage
        [System.IO.File]::WriteAllText(
            (Join-Path $PortableStage "portable.flag"),
            "Portable mode enabled: models, voices, config, and logs are stored in .\data`r`n",
            $Utf8NoBom
        )

        $ZipName = "Audio8-VC-${Version}-win64-${VariantTag}-portable.zip"
        $ZipPath = Join-Path $ReleaseDir $ZipName
        if (Test-Path $ZipPath) { Remove-Item -Force $ZipPath }
        Compress-Archive -Path $PortableStage -DestinationPath $ZipPath -CompressionLevel Optimal
        Write-Host "Created: $ZipPath" -ForegroundColor Cyan
        $Artifacts += $ZipPath
    } else {
        Write-Host "`n[5/6] Skipping Portable ZIP (-SkipPortableZip)." -ForegroundColor DarkGray
    }

    # 7. Build Inno Setup 6 Installer (.exe)
    if (-not $SkipInstaller) {
        Write-Host "`n[6/6] Building Inno Setup installer..." -ForegroundColor Green
        $IsccCmd = Get-Command iscc -ErrorAction SilentlyContinue
        $IsccPath = $null
        if ($null -ne $IsccCmd) {
            $IsccPath = $IsccCmd.Source
        } else {
            $candidates = @(
                "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
                "${env:ProgramFiles}\Inno Setup 6\ISCC.exe",
                "${env:LOCALAPPDATA}\Programs\Inno Setup 6\ISCC.exe"
            )
            foreach ($cand in $candidates) {
                if ($cand -and (Test-Path $cand)) {
                    $IsccPath = $cand
                    break
                }
            }
        }

        if ($null -ne $IsccPath) {
            & $IsccPath `
                "/DAppVersion=$Version" `
                "/DSourceDir=$DistAppDir" `
                "/DOutputDir=$ReleaseDir" `
                "/DVariant=$VariantTag" `
                (Join-Path $ScriptDir "installer.iss")
            if ($LASTEXITCODE -ne 0) {
                throw "Inno Setup compilation failed with exit code $LASTEXITCODE"
            }
            $SetupPath = Join-Path $ReleaseDir "Audio8-VC-${Version}-win64-${VariantTag}-setup.exe"
            Write-Host "Created: $SetupPath" -ForegroundColor Cyan
            $Artifacts += $SetupPath
        } else {
            Write-Warning "Inno Setup 6 (ISCC.exe) not found. Skipping setup installer. Install Inno Setup 6 or run via GitHub Actions to generate the installer."
        }
    } else {
        Write-Host "`n[6/6] Skipping Inno Setup installer (-SkipInstaller)." -ForegroundColor DarkGray
    }

    # Generate SHA256SUMS.txt
    if ($Artifacts.Count -gt 0) {
        $SumFile = Join-Path $ReleaseDir "SHA256SUMS.txt"
        $lines = foreach ($item in $Artifacts) {
            if (Test-Path $item) {
                $hash = (Get-FileHash -Algorithm SHA256 $item).Hash.ToLowerInvariant()
                "$hash  $(Split-Path -Leaf $item)"
            }
        }
        [System.IO.File]::WriteAllText($SumFile, (($lines -join "`n") + "`n"), $Utf8NoBom)
        Write-Host "`nChecksums written to: $SumFile" -ForegroundColor Green
        Get-Content $SumFile
    }

    Write-Host "`n=== Windows Build Completed Successfully ===" -ForegroundColor Cyan
}
finally {
    Pop-Location
}
