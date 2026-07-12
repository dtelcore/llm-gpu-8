"""
model/cuda/env.py

CUDA/MSVC environment bootstrap for the Kepler GT 730 (sm_35) workspace.

Mirrors the working configuration validated in setup/2_test_workspace.py:
CUDA 10.1 toolkit + MSVC 14.29 (VS2022 BuildTools) + Windows 10 SDK.
Must be imported (and `configure()` called) BEFORE any `pycuda` import,
otherwise nvcc cannot find a compatible host compiler on Windows.
"""

import os

from logging_config import logger

CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin"
MSVC_142_ROOT = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.29.30133"
MSVC_142_BIN = MSVC_142_ROOT + r"\bin\Hostx64\x64"
WINSDK_ROOT = r"C:\Program Files (x86)\Windows Kits\10"
WINSDK_VER = "10.0.26100.0"
WINSDK_BIN = f"{WINSDK_ROOT}\\bin\\{WINSDK_VER}\\x64"

NVCC_OPTIONS = [
    f"-ccbin={MSVC_142_ROOT}\\bin\\Hostx64\\x64\\cl.exe",
    f"-I{MSVC_142_ROOT}\\include",
    f"-I{WINSDK_ROOT}\\Include\\{WINSDK_VER}\\ucrt",
    f"-I{WINSDK_ROOT}\\Include\\{WINSDK_VER}\\um",
    f"-I{WINSDK_ROOT}\\Include\\{WINSDK_VER}\\shared",
]

_configured = False


def configure() -> None:
    """Wire up PATH / DLL search directories for CUDA 10.1 + MSVC 14.29.

    Idempotent: safe to call multiple times.
    """
    global _configured
    if _configured:
        return

    if hasattr(os, "add_dll_directory") and os.path.exists(CUDA_BIN):
        os.add_dll_directory(CUDA_BIN)

    os.environ["PATH"] = f"{CUDA_BIN};{MSVC_142_BIN};{WINSDK_BIN};" + os.environ["PATH"]

    logger.debug("CUDA environment configured: CUDA_BIN=%s", CUDA_BIN)
    _configured = True
