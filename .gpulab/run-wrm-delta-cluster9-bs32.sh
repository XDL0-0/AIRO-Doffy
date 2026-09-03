#!/usr/bin/env bash

set -Eeuo pipefail

stage_dir="/tmp/airo_wrm_delta_cluster9_stage"
source_archive="${stage_dir}/airo-doffy-wrm-delta-src.tar.gz"
source_manifest="${stage_dir}/airo-doffy-wrm-delta-src.sha256"
dataset_archive="${stage_dir}/WRM_grasp_cylinder_different_sizes_lero_tightness.tar"
dataset_manifest="${stage_dir}/WRM_grasp_cylinder_different_sizes_lero_tightness.sha256"
wandb_netrc_file="${stage_dir}/netrc"
python_bin="/opt/conda/envs/lerobot/bin/python"
work_root="/tmp/airo_wrm_delta_cluster9_work"
repo_root="${work_root}/AIRO-Doffy"
dataset_root="${work_root}/datasets/WRM_grasp_cylinder_different_sizes_lero_tightness"
output_root="/project_ghent/AIRO-Doffy/WRM_grasp_cylinder_different_sizes_lero_tightness/wrm_delta_cluster9_bs32"
policy_output="${output_root}/WRM_delta"
policy_log="${policy_output}/cluster9_bs32_train.log"
batch_size=32
num_workers=8

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
chmod 600 "${wandb_netrc_file}"
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
export HF_HOME="${work_root}/huggingface"
export WANDB_CACHE_DIR="${work_root}/wandb-cache"
export WANDB_DIR="${policy_output}/wandb"
export PYTHONUNBUFFERED=1
trap 'unset WANDB_API_KEY' EXIT

cd "${repo_root}"
"${python_bin}" -m pip install \
    --disable-pip-version-check \
    --no-cache-dir \
    --upgrade \
    "torch==2.7.0" \
    "torchvision==0.22.0" \
    --index-url https://download.pytorch.org/whl/cu128
"${python_bin}" -m pip install --disable-pip-version-check --no-deps "lerobot==0.4.4"

"${python_bin}" - "${dataset_root}" <<'PY'
import json
import os
import sys
from pathlib import Path

import torch

dataset_root = Path(sys.argv[1])
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit(
        f"Expected exactly one visible CUDA GPU, got {torch.cuda.device_count()}"
    )
if "sm_120" not in torch.cuda.get_arch_list():
    raise SystemExit(
        f"PyTorch build lacks Blackwell sm_120 support: {torch.cuda.get_arch_list()}"
    )
probe = torch.randn(512, 512, device="cuda", dtype=torch.float16)
torch.cuda.synchronize()
info = json.loads((dataset_root / "meta" / "info.json").read_text())
if info.get("total_episodes") != 125 or "tightness" not in info.get("features", {}):
    raise SystemExit("The staged dataset is not the requested 125-episode tightness set")
print(
    "gpu", torch.cuda.get_device_name(0),
    "memory_gb", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
    "probe", tuple((probe @ probe).shape),
)
print(
    "resources",
    "reserved_cpus", os.environ.get("GPULAB_CPUS_RESERVED", "unknown"),
    "visible_cpus", len(os.sched_getaffinity(0)),
    "batch_size", 32,
    "num_workers", 8,
)
PY

"${python_bin}" -m unittest policies.realman_beaver.tests.test_wrm_delta -v

if [[ -f "${policy_output}/last.pt" ]]; then
    echo "[$(date -Is)] existing completed WRM_delta policy found; skipping training"
else
    echo "[$(date -Is)] starting WRM_delta on cluster 9"
    set -o pipefail
    "${python_bin}" -m policies.realman_beaver.train \
        --config policies/realman_beaver/configs/WRM_delta.yaml \
        --val-episodes 50-74 \
        --wandb-project AIRO-Doffy-WRM-Grasp \
        --wandb-run-name "WRM_grasp_cylinder_different_sizes_lero_tightness/WRM_delta-cluster9-bs32" \
        --dataset-root "${dataset_root}" \
        --dataset-repo-id WRM_grasp_cylinder_different_sizes_lero_tightness \
        --device cuda:0 \
        --batch-size "${batch_size}" \
        --num-workers "${num_workers}" \
        --max-steps 100000 \
        --output-dir "${policy_output}" 2>&1 | tee "${policy_log}"
fi

"${python_bin}" - "${policy_output}" <<'PY'
import sys
from pathlib import Path

import torch

folder = Path(sys.argv[1])
milestones = sorted(folder.glob("WRM_delta_step_*.pt"))
last = folder / "last.pt"
if len(milestones) != 4 or not last.is_file():
    raise SystemExit(
        f"Incomplete WRM_delta training: milestones={len(milestones)}, last={last.is_file()}"
    )
checkpoint = torch.load(last, map_location="cpu", weights_only=True)
if checkpoint.get("global_step") != 100000:
    raise SystemExit(f"Expected step 100000, got {checkpoint.get('global_step')}")
print("training_complete", last, "step", checkpoint["global_step"])
PY

echo "[$(date -Is)] completed cluster-9 WRM_delta batch-size-32 training"
