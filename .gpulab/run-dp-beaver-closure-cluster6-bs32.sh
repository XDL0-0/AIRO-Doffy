#!/usr/bin/env bash

set -Eeuo pipefail

stage_dir="/tmp/airo_dp_beaver_closure_cluster6_stage"
source_archive="${stage_dir}/airo-doffy-dp-beaver-closure-src.tar.gz"
source_manifest="${stage_dir}/airo-doffy-dp-beaver-closure-src.sha256"
dataset_archive="${stage_dir}/WRM_grasp_cylinder_different_sizes_lero_tightness.tar"
dataset_manifest="${stage_dir}/WRM_grasp_cylinder_different_sizes_lero_tightness.sha256"
wandb_netrc_file="${stage_dir}/netrc"
python_bin="/opt/conda/envs/lerobot/bin/python"
work_root="/tmp/airo_dp_beaver_closure_cluster6_work"
repo_root="${work_root}/AIRO-Doffy"
dataset_root="${work_root}/datasets/WRM_grasp_cylinder_different_sizes_lero_tightness"
output_root="/project_ghent/AIRO-Doffy/WRM_grasp_cylinder_different_sizes_lero_tightness/dp_beaver_closure_cluster6_bs32"
policy_output="${output_root}/dp_beaver_closure"
policy_log="${policy_output}/cluster6_bs32_train.log"
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
    --no-deps \
    "lerobot==0.4.4"

"${python_bin}" - "${dataset_root}" "${batch_size}" "${num_workers}" <<'PY'
import json
import os
import sys
from pathlib import Path

import torch

dataset_root = Path(sys.argv[1])
batch_size = int(sys.argv[2])
num_workers = int(sys.argv[3])
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit(
        f"Expected exactly one visible CUDA GPU, got {torch.cuda.device_count()}"
    )
probe = torch.randn(512, 512, device="cuda", dtype=torch.float16)
torch.cuda.synchronize()
info = json.loads((dataset_root / "meta" / "info.json").read_text())
if info.get("total_episodes") != 125 or "tightness" not in info.get("features", {}):
    raise SystemExit("The staged dataset is not the 125-episode tightness set")
print(
    "gpu", torch.cuda.get_device_name(0),
    "memory_gb", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
    "torch", torch.__version__,
    "cuda", torch.version.cuda,
    "probe", tuple((probe @ probe).shape),
)
print(
    "resources",
    "reserved_cpus", os.environ.get("GPULAB_CPUS_RESERVED", "unknown"),
    "visible_cpus", len(os.sched_getaffinity(0)),
    "batch_size", batch_size,
    "num_workers", num_workers,
)
PY

"${python_bin}" -m unittest \
    policies.realman_beaver.tests.test_dp_beaver_closure \
    -v

if [[ -f "${policy_output}/last.pt" ]]; then
    echo "[$(date -Is)] completed checkpoint already exists; skipping training"
else
    echo "[$(date -Is)] starting dp_beaver_closure on Cluster 6"
    set -o pipefail
    "${python_bin}" -m policies.realman_beaver.train \
        --config policies/realman_beaver/configs/dp_beaver_closure.yaml \
        --wandb-project AIRO-Doffy-WRM-Grasp \
        --wandb-run-name "WRM_grasp_cylinder_different_sizes_lero_tightness/dp_beaver_closure-select-cluster6-bs32" \
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
milestones = sorted(folder.glob("dp_beaver_closure_step_*.pt"))
last = folder / "last.pt"
if len(milestones) != 10 or not last.is_file():
    raise SystemExit(
        f"Incomplete training: milestones={len(milestones)}, last={last.is_file()}"
    )
checkpoint = torch.load(last, map_location="cpu", weights_only=True)
if checkpoint.get("global_step") != 100000:
    raise SystemExit(f"Expected step 100000, got {checkpoint.get('global_step')}")
print("training_complete", last, "step", checkpoint["global_step"])
PY

echo "[$(date -Is)] completed Cluster-6 dp_beaver_closure training"
