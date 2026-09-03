#!/usr/bin/env bash
# Container-side runner: policy training only, W&B online, HF upload of all checkpoints.
# Environment (set by the GPUlab job JSON):
#   WRM_VARIANT, WRM_CONFIG, HF_REPO_ID, UPLOAD_HF, BLACKWELL_TORCH

set -Eeuo pipefail

variant="${WRM_VARIANT:?WRM_VARIANT is required}"
config_path="${WRM_CONFIG:?WRM_CONFIG is required}"
stage_dir="/tmp/airo_wrm_competition_select_stage"
source_archive="${stage_dir}/airo-doffy-wrm-competition-src.tar.gz"
source_manifest="${stage_dir}/airo-doffy-wrm-competition-src.sha256"
dataset_archive="${stage_dir}/WRM_grasp_cylinder_different_sizes_lero_tightness.tar"
dataset_manifest="${stage_dir}/WRM_grasp_cylinder_different_sizes_lero_tightness.sha256"
hf_token_file="${stage_dir}/hf_token"
wandb_netrc_file="${stage_dir}/netrc"
python_bin="/opt/conda/envs/lerobot/bin/python"
work_root="/tmp/airo_wrm_competition_select_work"
repo_root="${work_root}/AIRO-Doffy"
dataset_root="${work_root}/datasets/WRM_grasp_cylinder_different_sizes_lero_tightness"
output_root="/project_ghent/AIRO-Doffy/WRM_grasp_cylinder_different_sizes_lero_tightness/${variant}_select"
policy_output="${output_root}/${variant}"
policy_log="${policy_output}/select_train.log"
wandb_project="AIRO-Doffy-WRM-Grasp"
batch_size=32
num_workers=8
upload_hf="${UPLOAD_HF:-1}"
hf_repo_id="${HF_REPO_ID:-IXDLI/AIRO-Doffy-WRM-Grasp-${variant}}"
blackwell_torch="${BLACKWELL_TORCH:-0}"

for required_file in \
    "${source_archive}" \
    "${source_manifest}" \
    "${dataset_archive}" \
    "${dataset_manifest}" \
    "${wandb_netrc_file}"; do
    if [[ ! -s "${required_file}" ]]; then
        echo "Missing staged input: ${required_file}" >&2
        exit 2
    fi
done
if [[ "${upload_hf}" == "1" && ! -s "${hf_token_file}" ]]; then
    echo "UPLOAD_HF=1 requires staged Hugging Face credentials" >&2
    exit 2
fi

chmod 600 "${wandb_netrc_file}"
if [[ -f "${hf_token_file}" ]]; then
    chmod 600 "${hf_token_file}"
fi
(
    cd "${stage_dir}"
    sha256sum --check --strict "$(basename "${source_manifest}")"
    sha256sum --check --strict "$(basename "${dataset_manifest}")"
)

mkdir -p "${repo_root}" "${work_root}/datasets" "${policy_output}"
tar -xzf "${source_archive}" -C "${repo_root}"
tar -xf "${dataset_archive}" -C "${work_root}/datasets"

export WANDB_API_KEY
WANDB_API_KEY="$(
    "${python_bin}" - "${wandb_netrc_file}" <<'PY'
import netrc
import sys

credentials = netrc.netrc(sys.argv[1]).authenticators("api.wandb.ai")
if credentials is None or not credentials[2]:
    raise SystemExit("W&B credential is missing from the staged netrc")
print(credentials[2])
PY
)"
if [[ "${upload_hf}" == "1" ]]; then
    export HF_TOKEN
    HF_TOKEN="$(<"${hf_token_file}")"
fi
export HF_HOME="${work_root}/huggingface"
export WANDB_CACHE_DIR="${work_root}/wandb-cache"
export WANDB_DIR="${policy_output}"
export WANDB_MODE=online
export WANDB_PROJECT="${wandb_project}"
export WANDB_REQUIRED=1
export PYTHONUNBUFFERED=1
trap 'unset HF_TOKEN WANDB_API_KEY' EXIT

cd "${repo_root}"

if [[ "${blackwell_torch}" == "1" ]]; then
    "${python_bin}" -m pip install \
        --disable-pip-version-check \
        --no-cache-dir \
        --upgrade \
        --force-reinstall \
        "torch==2.7.1" \
        "torchvision==0.22.1" \
        "fsspec==2025.9.0" \
        --index-url https://download.pytorch.org/whl/cu128
fi
"${python_bin}" -m pip install \
    --disable-pip-version-check \
    --no-deps \
    "lerobot==0.4.4"

"${python_bin}" - "${dataset_root}" "${batch_size}" "${num_workers}" "${blackwell_torch}" <<'PY'
import json
import os
import sys
from pathlib import Path

import torch

dataset_root = Path(sys.argv[1])
batch_size = int(sys.argv[2])
num_workers = int(sys.argv[3])
blackwell = sys.argv[4] == "1"
info = json.loads((dataset_root / "meta" / "info.json").read_text())
if info.get("total_episodes") != 125:
    raise SystemExit(f"Expected 125 episodes, got {info.get('total_episodes')}")
if "tightness" not in info.get("features", {}):
    raise SystemExit("The staged dataset does not expose the tightness label")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit(f"Expected one visible CUDA GPU, got {torch.cuda.device_count()}")
if blackwell and "sm_120" not in torch.cuda.get_arch_list():
    raise SystemExit(f"PyTorch lacks sm_120: {torch.cuda.get_arch_list()}")
probe = torch.randn(512, 512, device="cuda", dtype=torch.float16)
probe_result = probe @ probe
torch.cuda.synchronize()
print(
    "gpu", torch.cuda.get_device_name(0),
    "memory_gb", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
    "torch", torch.__version__,
    "arch", ",".join(torch.cuda.get_arch_list()),
    "probe", tuple(probe_result.shape),
)
print(
    "resources",
    "reserved_cpus", os.environ.get("GPULAB_CPUS_RESERVED", "unknown"),
    "visible_cpus", len(os.sched_getaffinity(0)),
    "batch_size", batch_size,
    "num_workers", num_workers,
)
print("dataset_episodes", info["total_episodes"], "train", 100, "validation", 25)
PY

if [[ -f "${policy_output}/last.pt" ]]; then
    echo "[$(date -Is)] existing completed ${variant} run found; skipping training"
else
    echo "[$(date -Is)] starting ${variant} policy training; wandb=${wandb_project}; hf=${hf_repo_id}"
    set -o pipefail
    "${python_bin}" -m policies.realman_beaver.train \
        --config "${config_path}" \
        --val-episodes 20-24,45-49,70-74,95-99,120-124 \
        --wandb-project "${wandb_project}" \
        --wandb-run-name "WRM_grasp_cylinder_different_sizes_lero_tightness/${variant}" \
        --dataset-root "${dataset_root}" \
        --dataset-repo-id "WRM_grasp_cylinder_different_sizes_lero_tightness" \
        --device cuda:0 \
        --batch-size "${batch_size}" \
        --num-workers "${num_workers}" \
        --max-steps 100000 \
        --output-dir "${policy_output}" 2>&1 | tee "${policy_log}"
fi

if ! grep -q "wandb: logging to https://" "${policy_log}"; then
    echo "W&B did not start; refusing to treat this run as complete" >&2
    exit 1
fi

"${python_bin}" - "${policy_output}" "${variant}" <<'PY'
import sys
from pathlib import Path

import torch

folder = Path(sys.argv[1])
variant = sys.argv[2]
milestones = sorted(folder.glob(f"{variant}_step_*.pt"))
last = folder / "last.pt"
if len(milestones) < 1 or not last.is_file():
    raise SystemExit(
        f"Incomplete {variant} run: milestones={len(milestones)}, last={last.is_file()}"
    )
checkpoint = torch.load(last, map_location="cpu", weights_only=True)
if checkpoint.get("kind") != variant:
    raise SystemExit(f"Unexpected checkpoint kind: {checkpoint.get('kind')}")
print(
    "checkpoint_ok",
    variant,
    "milestones",
    len(milestones),
    "step",
    checkpoint.get("global_step"),
)
PY

if [[ "${upload_hf}" == "1" ]]; then
    "${python_bin}" - "${hf_repo_id}" "${policy_output}" "${variant}" <<'PY'
import hashlib
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

repo_id, folder_path, variant = sys.argv[1:]
folder = Path(folder_path)
milestones = sorted(folder.glob(f"{variant}_step_*.pt"))
last = folder / "last.pt"
if not last.is_file() or not milestones:
    raise SystemExit(f"Refusing incomplete {variant} upload")

digest = hashlib.sha256()
with last.open("rb") as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(chunk)

api = HfApi(token=os.environ["HF_TOKEN"])
identity = api.whoami()
print("huggingface_user", identity.get("name"))
api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
api.upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path=str(folder),
    allow_patterns=("*.pt", "resolved_config.yaml", "metrics.jsonl", "*train.log"),
    ignore_patterns=["*.tmp", "wandb/**"],
    commit_message=f"Upload all {variant} checkpoints and training artifacts",
)
card = f"""---
library_name: pytorch
pipeline_tag: robotics
tags:
- robotics
- imitation-learning
- whole-arm-manipulation
---

# {repo_id.split('/', 1)[-1]}

Trained `{variant}` policy on `WRM_grasp_cylinder_different_sizes_lero_tightness`.

- Checkpoints: all numbered milestones plus EMA `last.pt`
- SHA-256 of `last.pt`: `{digest.hexdigest()}`
- Config: `resolved_config.yaml`
- Metrics: `metrics.jsonl`
- W&B project: `AIRO-Doffy-WRM-Grasp`
"""
api.upload_file(
    path_or_fileobj=card.encode(),
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="model",
    commit_message="Add checkpoint provenance",
)
print("huggingface_upload_complete", repo_id, "files", len(milestones) + 1)
PY
else
    echo "UPLOAD_HF=${upload_hf}; keeping checkpoints only on persistent storage"
fi

echo "[$(date -Is)] completed ${variant} policy training and Hugging Face upload"
