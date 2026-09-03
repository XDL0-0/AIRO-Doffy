#!/usr/bin/env bash

set -Eeuo pipefail

stage_dir="/tmp/airo_wrm_adaptive_all125_cluster12_stage"
source_archive="${stage_dir}/airo-doffy-wrm-adaptive-all125-src.tar.gz"
source_manifest="${stage_dir}/airo-doffy-wrm-adaptive-all125-src.sha256"
dataset_archive="${stage_dir}/WRM_grasp_cylinder_different_sizes_lero_tightness-all125.tar"
dataset_manifest="${stage_dir}/WRM_grasp_cylinder_different_sizes_lero_tightness-all125.sha256"
wandb_netrc_file="${stage_dir}/netrc"
python_bin="/opt/conda/envs/lerobot/bin/python"
work_root="/tmp/airo_wrm_adaptive_all125_cluster12_work"
repo_root="${work_root}/AIRO-Doffy"
dataset_root="${work_root}/datasets/WRM_grasp_cylinder_different_sizes_lero_tightness"
output_root="/project_ghent/AIRO-Doffy/WRM_grasp_cylinder_different_sizes_lero_tightness/wrm_adaptive_all125_cluster12_bs32"
policy_output="${output_root}/WRM_adaptive_all_train"
policy_log="${policy_output}/cluster12_all125_bs32_train.log"

for required_file in \
    "${source_archive}" \
    "${source_manifest}" \
    "${dataset_archive}" \
    "${dataset_manifest}" \
    "${wandb_netrc_file}"; do
    [[ -s "${required_file}" ]] || {
        echo "Missing staged input: ${required_file}" >&2
        exit 2
    }
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
    --force-reinstall \
    "torch==2.7.1" \
    "torchvision==0.22.1" \
    "fsspec==2025.9.0" \
    --index-url https://download.pytorch.org/whl/cu128
"${python_bin}" -m pip install \
    --disable-pip-version-check \
    --no-deps \
    "lerobot==0.4.4"

"${python_bin}" - "${dataset_root}" <<'PY'
import json
import os
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
info = json.loads((root / "meta" / "info.json").read_text())
if info.get("total_episodes") != 125:
    raise SystemExit(f"Expected 125 episodes, got {info.get('total_episodes')}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit(f"Expected one visible CUDA GPU, got {torch.cuda.device_count()}")
if "sm_120" not in torch.cuda.get_arch_list():
    raise SystemExit(f"PyTorch lacks sm_120: {torch.cuda.get_arch_list()}")
probe = torch.randn(512, 512, device="cuda", dtype=torch.float16)
probe_result = probe @ probe
torch.cuda.synchronize()
print("gpu", torch.cuda.get_device_name(0), "probe", tuple(probe_result.shape))
print(
    "resources",
    "visible_cpus", len(os.sched_getaffinity(0)),
    "requested_cpus", 8,
    "batch_size", 32,
)
print("dataset", "train_episodes", 125, "validation_episodes", 0)
PY

"${python_bin}" -m unittest \
    policies.realman_beaver.tests.test_wrm_adaptive -v

if [[ -f "${policy_output}/last.pt" ]]; then
    echo "[$(date -Is)] existing completed all-125 WRM_adaptive policy found; skipping"
else
    echo "[$(date -Is)] starting all-125 WRM_adaptive on cluster 12"
    set -o pipefail
    "${python_bin}" -m policies.realman_beaver.train \
        --config policies/realman_beaver/configs/WRM_adaptive_all_train.yaml \
        --wandb-project AIRO-Doffy-WRM-Grasp \
        --wandb-run-name "WRM_grasp_cylinder_different_sizes_lero_tightness/WRM_adaptive-all125-cluster12-bs32" \
        --dataset-root "${dataset_root}" \
        --dataset-repo-id WRM_grasp_cylinder_different_sizes_lero_tightness \
        --device cuda:0 \
        --batch-size 32 \
        --num-workers 8 \
        --max-steps 100000 \
        --output-dir "${policy_output}" 2>&1 | tee "${policy_log}"
fi

"${python_bin}" - "${policy_output}" <<'PY'
import sys
from pathlib import Path

import torch

folder = Path(sys.argv[1])
milestones = sorted(folder.glob("WRM_adaptive_step_*.pt"))
last = folder / "last.pt"
if len(milestones) != 10 or not last.is_file():
    raise SystemExit(f"Incomplete run: milestones={len(milestones)}, last={last.is_file()}")
checkpoint = torch.load(last, map_location="cpu", weights_only=True)
if checkpoint.get("kind") != "WRM_adaptive":
    raise SystemExit(f"Unexpected checkpoint kind: {checkpoint.get('kind')}")
if checkpoint.get("global_step") != 100000:
    raise SystemExit(f"Expected step 100000, got {checkpoint.get('global_step')}")
print("training_complete", last, "step", checkpoint["global_step"])
PY

echo "[$(date -Is)] completed all-125 WRM_adaptive training"
