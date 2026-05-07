# Setup Notes for `run_generation.sh`

This file documents the environment setup needed to run the data-generation
pipeline (`run_generation.sh` → `generate_vla_dataset.py`) on a fresh server,
plus the source-code workarounds that were made for issues that aren't covered
by the original `README.md`.

Tested on Ubuntu 24.04 with Python 3.10, robosuite 1.5.1, mujoco 3.8.0, an
NVIDIA GPU (driver 570.x), and conda env named `convai`.

---

## 1. Conda environment

```bash
conda create -n convai python=3.10 -y
conda activate convai
```

## 2. Python packages (pip)

The original README lists most of these; `numba` is required by
`robosuite/utils/numba.py` but is missing from that list.

```bash
pip install tensorflow-datasets tensorflow opencv-python apache-beam \
            mlcroissant mujoco numba scipy h5py
```

Versions known to work (`pip list`):

| package              | version     |
| -------------------- | ----------- |
| tensorflow           | 2.21.0      |
| tensorflow-datasets  | 4.9.9       |
| opencv-python        | 4.13.0.92   |
| apache-beam          | 2.72.0      |
| mlcroissant          | 1.1.0       |
| mujoco               | 3.8.0       |
| numba                | 0.65.1      |
| scipy                | 1.15.3      |
| numpy                | 2.2.6       |
| h5py                 | 3.14.0      |

## 3. System libraries (conda-forge)

Mujoco's EGL backend goes through PyOpenGL, which calls
`ctypes.util.find_library('EGL')`. On a typical headless server the only
EGL library installed is `libEGL_nvidia.so.0` — the GLVND wrapper
`libEGL.so.1` (and `libGL.so.1`) is missing, so PyOpenGL silently returns
`None` and the first render call dies with:

```
AttributeError: 'NoneType' object has no attribute 'eglQueryString'
```

Install the GLVND wrappers from conda-forge into the env:

```bash
conda install -y -c conda-forge libegl libgl
```

Then ensure `$CONDA_PREFIX/lib` is on `LD_LIBRARY_PATH` before running
the script. `run_generation.sh` already exports this (see §5).

## 4. Source-code changes already in the repo

Three changes were committed to make the pipeline run on this server.
They are unconditional (no harm on a healthier host) and you can keep
them as-is.

### 4a. `robosuite/macros_private.py` (new file)

```python
ENABLE_NUMBA = False
CACHE_NUMBA = False
MUJOCO_GPU_RENDERING = False
```

`robosuite/macros.py:19` itself notes that numba causes deterministic
crashes during offscreen rendering for some tasks; disabling it here is
robosuite's own recommended workaround. The `MUJOCO_GPU_RENDERING=False`
line is harmless when `MUJOCO_GL=egl` is set explicitly.

### 4b. `generate_vla_dataset.py` — depth NaN/clip guard

After `env.reset()`, the EGL/NVIDIA depth buffer occasionally returns
NaN or values outside `[0, 1]` on the first frame. Robosuite's
`camera_utils.get_real_depth_map` asserts the buffer is in `[0, 1]` and
crashes the run. The script now sanitizes the depth before calling it:

```python
depth_image_raw = np.nan_to_num(depth_image_raw, nan=1.0, posinf=1.0, neginf=0.0)
depth_image_raw = np.clip(depth_image_raw, 0.0, 1.0)
depth_real = camera_utils.get_real_depth_map(self.env.sim, depth_image_raw)
```

### 4c. `generate_vla_dataset.py` — disable camera observables during stepping

The `agentview_image` / `agentview_depth` observables re-render on every
`env.step()`. The script only needs them once per trial (right after
`env.reset()`), so we now toggle them off for the trajectory and back on
right before the next reset. This both saves GPU memory and avoids
SIGABRT under tight GPU memory (see §6).

```python
# right before env.reset()
for _obs_name in ("agentview_image", "agentview_depth"):
    if _obs_name in self.env._observables:
        self.env.modify_observable(observable_name=_obs_name, attribute="enabled", modifier=True)
self.obs = self.env.reset()
...
# after grabbing the initial frame, disable them for the trajectory
for _obs_name in ("agentview_image", "agentview_depth"):
    if _obs_name in self.env._observables:
        self.env.modify_observable(observable_name=_obs_name, attribute="enabled", modifier=False)
```

## 5. `run_generation.sh`

```bash
export MUJOCO_GL=egl
export LD_LIBRARY_PATH="${CONDA_PREFIX:-/home/jc166/.conda/envs/convai}/lib:${LD_LIBRARY_PATH}"

python generate_vla_dataset.py \
    --output_dir ./my_data \
    --num_trials 10 \
    --num_videos 0 \
    --start_index 0
```

`--num_videos 0` is the safe default on this server — see §6. Bump it
back to `1` (or higher) once the GPU has free memory.

## 6. GPU memory note (server-specific)

If you use a server with available GPU memory, you can:

- Set `--num_videos` back to `1` in `run_generation.sh`, and
- Optionally remove `MUJOCO_GPU_RENDERING = False` from
  `robosuite/macros_private.py`.

Verify free GPU memory with `nvidia-smi --query-gpu=memory.free --format=csv,noheader`
— anything above ~2 GB free should comfortably support video recording.

## 7. Verifying the install

```bash
conda activate convai
cd /path/to/dialogue-as-a-sensor/data-generation
bash run_generation.sh
```

Expected end of output:

```
Generation complete. 10 / 10 successful samples saved.
```

with `./my_data/episode_00000` … `episode_00009` each containing
`image_rgb.png`, `image_depth.npy`, and `metadata.json`.
