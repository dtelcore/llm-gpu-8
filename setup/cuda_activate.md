# CUDA Kernel Activation on GT730 (CC 3.5) — Full Journey

**GPU**: NVIDIA GeForce GT 730 (GK208, Compute Capability 3.5)  
**OS**: Windows 10, WDDM mode  
**Python**: 3.8 (venv at `_PHASE_3\venv\`)  
**Goal**: Execute a CUDA kernel (`add` vector addition) on the GT730 using pycuda

---

## Hardware / Software Baseline

| Component | Value |
|---|---|
| GPU | NVIDIA GeForce GT 730 (GK208) |
| Compute Capability | 3.5 |
| VRAM | 4096 MB DDR3 |
| Driver version | 475.14 (max CUDA runtime: 11.4) |
| PCIe | x8 2.0 |
| Python | 3.8 |
| CUDA Toolkits installed | v10.1 (10.1.105) and v13.2 |

---

## Approach 1 — PyTorch (Failed)

Tried installing PyTorch with CUDA support to run GPU tensors on the GT730.

### Attempts

| Package | Index | Result |
|---|---|---|
| `torch==1.12.1+cu102` | PyPI cu102 | **Does not exist** (latest cu102 is 1.10.2) |
| `torch==1.10.2+cu102` | PyPI cu102 | Installs, **"no kernel image"** at runtime |
| `torch==1.8.1+cu102` | PyPI cu102 | Installs, **"no kernel image"** at runtime |
| `torch==1.8.1+cu101` | PyPI cu101 | Installs, **"no kernel image"** at runtime |
| `torch==1.10.2+cu113` | PyPI cu113 | **"no kernel image"** at runtime |
| `torch==1.9.0+cu111` | PyPI cu111 | **"no kernel image"** at runtime |

### Root Cause

All PyTorch pip wheels for Windows dropped `sm_35` (CC 3.5) support. The prebuilt `.whl` files no longer include a PTX or SASS binary for Kepler architecture. Building PyTorch from source with `TORCH_CUDA_ARCH_LIST=3.5` would work but requires a ~6-hour source build.

**Decision**: Pivot to pycuda, which compiles kernels at runtime and explicitly supports any architecture nvcc can target.

---

## Approach 2 — pycuda (Success, after resolving 5 blockers)

### Installation

`pip install pycuda` (any version) initially failed — MSVC compiler not present.

**Installed**:
- MSVC Build Tools 2022 (v143, `14.44.35207`) via Visual Studio Installer
- Windows SDK 10.0.26100.0
- pycuda 2022.2.2 built from source (pycuda 2026.1 uses Python 3.10+ syntax)

**Build command that worked**:
```cmd
vcvarsall.bat x64
set DISTUTILS_USE_SDK=1
set MSSdk=1
set PATH=<cl_dir>;<rc_dir>;%PATH%
cd %TEMP%\pycuda-2022.2.2
python setup.py install
```

**pycuda device query confirmed working** (before kernel test):
```
Device: NVIDIA GeForce GT 730, Compute: (3, 5), VRAM: 4.0 GB
```

---

## Blocker 1 — `ImportError: DLL load failed while importing _driver`

### Symptom
```
ImportError: DLL load failed while importing _driver: The specified module could not be found.
```

### Cause
Python 3.8 changed DLL search behavior — `cudart64_101.dll` in `CUDA\v10.1\bin` was not found even when that directory was in `PATH`, because Python 3.8 no longer searches `PATH` for DLLs loaded by extension modules.

### Fix
Use `os.add_dll_directory()` (new in Python 3.8) to explicitly register the CUDA 10.1 bin dir:
```python
import os
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin")
```
This must be called **before** `import pycuda`.

---

## Blocker 2 — `nvcc fatal: Cannot find compiler 'cl.exe' in PATH`

### Symptom
```
nvcc fatal   : Cannot find compiler 'cl.exe' in PATH
```

### Cause
`pycuda.compiler.SourceModule` calls nvcc as a subprocess. `cl.exe` (MSVC) was not in `PATH` at the time Python was launched.

### Fix
Prepend MSVC bin directory to `os.environ["PATH"]` before importing pycuda:
```python
_cl_dir = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64"
os.environ["PATH"] = _cl_dir + ";" + os.environ["PATH"]
```

---

## Blocker 3 — `host_config.h: unsupported Microsoft Visual Studio version`

### Symptom
```
C:\...\CUDA\v10.1\include\crt\host_config.h(152): fatal error C1189:
#error:  -- unsupported Microsoft Visual Studio version!
Only the versions between 2015 and 2019 (inclusive) are supported!
```

### Cause
CUDA 10.1's `host_config.h` rejects `_MSC_VER >= 1930` (MSVC 2022 = 1944). MSVC v142 (VS 2019 toolset, 14.29.x) was not yet installed — only v143 (14.44) was present.

### Fix — Part A: Patch host_config.h
Raise the rejection ceiling from `1930` to `2000`:
```
File: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\include\crt\host_config.h
Change: #if _MSC_VER < 1700 || _MSC_VER >= 1930
     → #if _MSC_VER < 1700 || _MSC_VER >= 2000
```
(Run as Administrator)

### Fix — Part B: Install MSVC v142 (VS 2019 toolset)
```
vs_installer.exe modify
  --add Microsoft.VisualStudio.Component.VC.14.29.16.11.x86.x64
  --quiet --norestart
```
Result: `14.29.30133` toolset installed alongside `14.44.35207`.

---

## Blocker 4 — nvcc 13.2 picked up instead of nvcc 10.1

### Symptom
CUDA 13.2's nvcc was in the system PATH (`CUDA\v13.2\bin\x64`) and took precedence. CUDA 13.2 dropped sm_35:
```
nvcc fatal   : Unsupported gpu architecture 'sm_35'
```

### Cause
System `PATH` had `CUDA\v13.2\bin\x64` but not `CUDA\v10.1\bin`. Even after prepending v10.1 bin to `os.environ["PATH"]`, pycuda's compiler subprocess inherited the unmodified system PATH in some terminal states.

### Fix
Explicitly prepend CUDA 10.1 bin and pass `-ccbin` + `-I` flags directly to `SourceModule` so nvcc uses them regardless of auto-detection:
```python
_NVCC_OPTIONS = [
    f"-ccbin={_msvc142}\\bin\\Hostx64\\x64\\cl.exe",
    f"-I{_msvc142}\\include",
    f"-I{_winsdk}\\Include\\{_sdk_ver}\\ucrt",
    f"-I{_winsdk}\\Include\\{_sdk_ver}\\um",
    f"-I{_winsdk}\\Include\\{_sdk_ver}\\shared",
]
mod = SourceModule(kernel_src, options=_NVCC_OPTIONS)
```

---

## Blocker 5 — `STL1002: expected CUDA 10.1 Update 2 or newer`

### Symptom
```
C:\...\MSVC\14.29.30133\include\yvals_core.h(565): fatal error C1189:
#error: STL1002: Unexpected compiler version, expected CUDA 10.1 Update 2 or newer.
```

### Cause
MSVC 2019's STL headers (`yvals_core.h`) check `__CUDACC_VER_BUILD__ >= 243` (CUDA 10.1 Update 2 = build 243). We have CUDA 10.1.105 (Update 1, build 105).

### Fix
Patch `yvals_core.h` to lower the build threshold:
```
File: C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.29.30133\include\yvals_core.h
Change: __CUDACC_VER_BUILD__ < 243
     → __CUDACC_VER_BUILD__ < 100
```
(Run as Administrator)

---

## Final Working Code

```python
import os
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin")

_msvc142 = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.29.30133"
_winsdk  = r"C:\Program Files (x86)\Windows Kits\10"
_sdk_ver = "10.0.26100.0"
_cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin"
os.environ["PATH"] = (
    f"{_cuda_bin};"
    f"{_msvc142}\\bin\\Hostx64\\x64;"
    f"{_winsdk}\\bin\\{_sdk_ver}\\x64;"
    + os.environ["PATH"]
)

_NVCC_OPTIONS = [
    f"-ccbin={_msvc142}\\bin\\Hostx64\\x64\\cl.exe",
    f"-I{_msvc142}\\include",
    f"-I{_winsdk}\\Include\\{_sdk_ver}\\ucrt",
    f"-I{_winsdk}\\Include\\{_sdk_ver}\\um",
    f"-I{_winsdk}\\Include\\{_sdk_ver}\\shared",
]

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np

mod = SourceModule("""
__global__ void add(float *a, float *b, float *c) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
""", options=_NVCC_OPTIONS)

add = mod.get_function("add")
a = np.array([1, 2, 3, 4], dtype=np.float32)
b = np.array([10, 20, 30, 40], dtype=np.float32)
c = np.zeros_like(a)
add(cuda.In(a), cuda.In(b), cuda.Out(c), block=(4, 1, 1))
print("Kernel result:", c)
assert list(c) == [11, 22, 33, 44], f"Wrong result: {c}"
print("CUDA kernel on GT730 CC 3.5 works!")
```

### Output
```
Kernel result: [11. 22. 33. 44.]
CUDA kernel on GT730 CC 3.5 works!
```

---

## Files Patched

| File | Change |
|---|---|
| `CUDA\v10.1\include\crt\host_config.h` | `_MSC_VER >= 1930` → `>= 2000` (allow MSVC 2022) |
| `MSVC\14.29.30133\include\yvals_core.h` | `VER_BUILD < 243` → `< 100` (allow CUDA 10.1.105) |

## Tools / Versions Used

| Tool | Version |
|---|---|
| pycuda | 2022.2.2 (built from source) |
| CUDA Toolkit | 10.1.105 (runtime DLLs + nvcc) |
| MSVC toolset | 14.29.30133 (v142, VS 2019) |
| Windows SDK | 10.0.26100.0 |
| Python | 3.8 |

---

*First successful CUDA kernel execution on GT730: May 3, 2026*
