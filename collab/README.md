# Collab package (Google Colab Free + T4)

**Branch:** `collab`  
**Rule:** this directory is the only place Colab-specific code lives. Core `model/cuda/` is **not** edited; [`bootstrap.py`](bootstrap.py) patches env + kernels in memory before training.

## Layout

```text
collab/
  __init__.py
  bootstrap.py          # Linux/T4 patches before model.cuda.ops
  requirements.txt
  smoke_cuda.py
  make_toy_config.py
  run_train.py          # bootstrap + train.py --no-prompt
  run_generate.py       # bootstrap + generate.py
  sync_drive.py
  configs/collab_toy.json
  llm_gpu8_t4.ipynb
  README.md
```

Always use **cwd = repo root** (parent of `collab/`).

---

## Pull into Colab (T4)

### 1) Set runtime

Runtime → Change runtime type → **T4 GPU** (or GPU). Confirm:

```python
!nvidia-smi
```

### 2) Clone branch `collab`

```python
!git clone -b collab --single-branch https://github.com/dtelcore/llm-gpu-8.git /content/llm-gpu-8
%cd /content/llm-gpu-8
```

If the repo is already on Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
%cd "/content/drive/MyDrive/llm gpu 8"   # edit to your path
!git fetch origin collab
!git checkout collab
```

### 3) Install + smoke

```python
!pip install -q -r collab/requirements.txt
!python collab/smoke_cuda.py
```

If `pycuda` fails to build: `!pip install pycuda --no-build-isolation`

### 4) Smoke train + generate

```python
!python collab/make_toy_config.py
!python collab/run_train.py \
  --config collab/configs/collab_toy.json \
  --checkpoint output/checkpoints/collab_smoke \
  --epochs 2 --seed 0

!python collab/run_generate.py \
  --checkpoint output/checkpoints/collab_smoke \
  --prompt "once upon a" --max-new-tokens 80
```

### 5) Persist (Free tier)

```python
!python collab/sync_drive.py --push
# after a new runtime:
!python collab/sync_drive.py --pull
```

Default Drive root: `/content/drive/MyDrive/llm_gpu8_output`

---

## Local push (from your PC)

After committing on branch `collab`:

```bash
git push -u origin collab
```

Then use the clone cell above in Colab.

---

## Checklist

1. `nvidia-smi` → Tesla T4  
2. `python collab/smoke_cuda.py` → CC 7.5, SMOKE PASSED  
3. `run_train.py` finishes; checkpoints under `output/checkpoints/collab_smoke`  
4. `run_generate.py` prints text  
5. Drive push/pull survives runtime reset  

Or open [`llm_gpu8_t4.ipynb`](llm_gpu8_t4.ipynb) and run all cells.
