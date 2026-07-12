# ==============================================================================
# CUDA & PyCUDA Environment Setup Script for GT 730 (CC 3.5)
# Target Context: Main Directory "llm gpu 8"
# Run this script from an elevated PowerShell prompt (Run as Administrator).
# ==============================================================================

# --- Ensure Administrator Privileges ---
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsInBuiltRole]::Administrator)) {
    Write-Host "This script requires Administrator privileges to modify system files and installers." -ForegroundColor Yellow
    Write-Host "Relaunching as Administrator..." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`"" -Verb RunAs
    Exit
}

# --- Dynamic Path Configuration ---
# Automatically anchors to your current "llm gpu 5" main directory context
$BASE_DIR      = Get-Item .
$VENV_PATH     = Join-Path $BASE_DIR "venv"
$VS_INSTALLER  = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vs_installer.exe"
$CUDA_PATH     = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1"
$MSVC_142_PATH = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.29.30133"

Write-Host "Working Directory Context: $($BASE_DIR.FullName)" -ForegroundColor Cyan
Write-Host "Target Venv Destination:   $VENV_PATH" -ForegroundColor Cyan

# --- Step 1: Initialize Python Virtual Environment ---
Write-Host "`n=== STEP 1: Setting up Python Virtual Environment ===" -ForegroundColor Cyan
if (-not (Test-Path $VENV_PATH)) {
    Write-Host "Creating clean Python venv in $VENV_PATH..." -ForegroundColor Yellow
    # Explicitly using python (assumed 3.8 as per baseline specs)
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create virtual environment. Ensure Python 3.8 is available in your system path."
        Exit
    }
    Write-Host "Virtual environment created successfully." -ForegroundColor Green
} else {
    Write-Host "Virtual environment already exists at target location." -ForegroundColor DarkGreen
}

# --- Step 2: Verify / Install MSVC v142 Toolset (Blocker 3 Part B) ---
Write-Host "`n=== STEP 2: Verifying MSVC v142 Toolset (VS 2019) ===" -ForegroundColor Cyan
if (-not (Test-Path "$MSVC_142_PATH\bin\Hostx64\x64\cl.exe")) {
    if (Test-Path $VS_INSTALLER) {
        Write-Host "MSVC v142 not found. Installing via Visual Studio Installer..." -ForegroundColor Yellow
        Start-Process -FilePath $VS_INSTALLER -ArgumentList "modify --add Microsoft.VisualStudio.Component.VC.14.29.16.11.x86.x64 --productID Microsoft.VisualStudio.Product.BuildTools --quiet --norestart" -Wait -NoNewWindow
        Write-Host "Installation command completed." -ForegroundColor Green
    } else {
        Write-Error "Visual Studio Installer not found at standard path. Please install MSVC v142 component manually."
        Exit
    }
} else {
    Write-Host "MSVC v142 Build Tools already present." -ForegroundColor Green
}

# --- Step 3: Patch host_config.h (Blocker 3 Part A) ---
Write-Host "`n=== STEP 3: Patching CUDA host_config.h ===" -ForegroundColor Cyan
$hostConfigPath = "$CUDA_PATH\include\crt\host_config.h"
if (Test-Path $hostConfigPath) {
    $content = Get-Content $hostConfigPath -Raw
    if ($content -match '#if _MSC_VER < 1700 \|\| _MSC_VER >= 1930') {
        Write-Host "Patching host_config.h to allow MSVC 2022 compatibility..." -ForegroundColor Yellow
        $content = $content -replace '#if _MSC_VER < 1700 \|\| _MSC_VER >= 1930', '#if _MSC_VER < 1700 || _MSC_VER >= 2000'
        Set-Content -Path $hostConfigPath -Value $content -Encoding UTF8
        Write-Host "Successfully patched host_config.h" -ForegroundColor Green
    } else {
        Write-Host "host_config.h already patched or updated." -ForegroundColor DarkGreen
    }
} else {
    Write-Error "Could not find host_config.h at $hostConfigPath. Is CUDA 10.1 installed?"
}

# --- Step 4: Patch yvals_core.h (Blocker 5) ---
Write-Host "`n=== STEP 4: Patching MSVC STL yvals_core.h ===" -ForegroundColor Cyan
$yvalsPath = "$MSVC_142_PATH\include\yvals_core.h"
if (Test-Path $yvalsPath) {
    $content = Get-Content $yvalsPath -Raw
    if ($content -match '__CUDACC_VER_BUILD__ < 243') {
        Write-Host "Patching yvals_core.h to accept CUDA 10.1.105 build..." -ForegroundColor Yellow
        $content = $content -replace '__CUDACC_VER_BUILD__ < 243', '__CUDACC_VER_BUILD__ < 100'
        Set-Content -Path $yvalsPath -Value $content -Encoding UTF8
        Write-Host "Successfully patched yvals_core.h" -ForegroundColor Green
    } else {
        Write-Host "yvals_core.h already patched or updated." -ForegroundColor DarkGreen
    }
} else {
    Write-Warning "Could not locate yvals_core.h yet. It will be available after Step 2 completes fully."
}

# --- Step 5: Environment Activation & MSVC Exporting ---
Write-Host "`n=== STEP 5: Activating Python Venv & Compiling Environment ===" -ForegroundColor Cyan
$activateScript = Join-Path $VENV_PATH "Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    . $activateScript
    Write-Host "Virtual environment activated." -ForegroundColor Green
} else {
    Write-Error "Virtual environment activation script missing!"
    Exit
}

# Use a cmd wrapper block to parse native MSVC variables into PowerShell env
$vcvarsall = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
if (Test-Path $vcvarsall) {
    $setup_cmd = "@echo off`ncall `"$vcvarsall`" x64 -vcvars_ver=14.29.30133`nset"
    $temp_batch = "$env:TEMP\setup_msvc_vars.bat"
    $setup_cmd | Out-File -FilePath $temp_batch -Encoding ASCII
    
    $env_output = & cmd /c "$temp_batch"
    foreach ($line in $env_output) {
        if ($line -match "^([A-Za-z_][A-Za-z0-9_]*)=(.*)$") {
            Set-Item -Path "env:$($matches[1])" -Value $matches[2]
        }
    }
    Remove-Item $temp_batch -Force -ErrorAction SilentlyContinue
    Write-Host "MSVC v142 variables exported to session." -ForegroundColor Green
}

# --- Step 6: Path Enforcement Strategy (Blocker 4 Fix) ---
Write-Host "`n=== STEP 6: Enforcing Path Dominance for CUDA 10.1 ===" -ForegroundColor Cyan
$env:CUDA_PATH = $CUDA_PATH
# Explicitly prepend CUDA 10.1 and MSVC v142 to bypass system CUDA 13 overrides
$env:PATH = "$CUDA_PATH\bin;$MSVC_142_PATH\bin\Hostx64\x64;$env:PATH"

# Set Distutils flags for Python compilation
$env:DISTUTILS_USE_SDK = "1"
$env:MSSdk = "1"

# --- Step 7: Core Dependencies & PyCUDA Build ---
Write-Host "`n=== STEP 7: Upgrading Pip Tools & Installing Pre-requisites ===" -ForegroundColor Cyan
python -m pip install --upgrade pip setuptools wheel

# CRITICAL FIX: Install numpy first so the pycuda setup script can locate its C-headers
Write-Host "Installing numpy (required for PyCUDA compilation headers)..." -ForegroundColor Yellow
pip install "numpy<2.0.0"

Write-Host "Running verbose PyCUDA 2022.2.2 source build natively..." -ForegroundColor Yellow
pip install pycuda==2022.2.2 --no-build-isolation --no-cache-dir --verbose 2>&1 | Tee-Object -FilePath "pycuda_build.log"