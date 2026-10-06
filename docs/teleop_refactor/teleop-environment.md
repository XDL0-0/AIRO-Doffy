# Teleoperation Python environment

The current teleoperation compatibility set is recorded in
[`../../requirements-teleop-constraints.txt`](../../requirements-teleop-constraints.txt).
It pins packages that must agree across LeRobot recording, dataset rollback,
and WebRTC media: LeRobot, `aiortc`, `aiohttp`, `datasets`, PyAV, Hugging Face
Hub/transfer, `fsspec`, `packaging`, and `setuptools`.

## Local environment used for verification

The verified environment is `/home/yuyuan/.venvs/airo-teleop`, created from
`/home/yuyuan/miniconda3/envs/airo-doffy/bin/python` with
`--system-site-packages`. It inherits the source environment's packages and
Python user-site packages, including the existing Torch/CUDA stack. The
compatibility overrides and test runner are installed inside the venv. This
keeps the shared source environment unchanged while allowing the teleop
dependencies to use versions compatible with LeRobot 0.4.4.

Recreate this workstation profile with:

```bash
/home/yuyuan/miniconda3/envs/airo-doffy/bin/python -m venv --system-site-packages /home/yuyuan/.venvs/airo-teleop
/home/yuyuan/.venvs/airo-teleop/bin/python -m pip install --upgrade pip
/home/yuyuan/.venvs/airo-teleop/bin/python -m pip install -c requirements-teleop-constraints.txt \
  lerobot aiortc aiohttp datasets diffusers accelerate av huggingface-hub \
  hf-transfer fsspec packaging setuptools
```

Run the RealMan teleoperation entry point through the intended interpreter so
the shell cannot select the shared environment by accident:

```bash
/home/yuyuan/.venvs/airo-teleop/bin/python realman_teleop.py
```

For a repository-local environment, create `.venv` from the same source
interpreter with `--system-site-packages`, install the constrained packages,
and launch with `./.venv/bin/python realman_teleop.py`.

## Scope and portability

This file is a constraints file: it limits versions when packages are
installed, but does not name or install the whole application dependency set.
The verified setup also inherits packages from this workstation's Miniconda
and user-site directories. It is therefore a compatibility profile for this
Python 3.10 workstation, not a portable lock file. A portable environment
needs an explicit full dependency set plus platform-specific Python, PyTorch,
CUDA, camera, robot SDK, and internal package choices.

Verification on this workstation used `pip check`, imported the pinned
teleop packages, and ran:

```bash
/home/yuyuan/.venvs/airo-teleop/bin/python -m pytest -q \
  tests/test_dataset_rollback.py tests/test_teleop_media_webrtc.py
```

Those checks exercise software paths only; they do not validate attached robot,
camera, or controller hardware.
