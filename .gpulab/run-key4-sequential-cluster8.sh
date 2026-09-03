#!/usr/bin/env bash

set -Eeuo pipefail

stage_dir="/tmp/airo_key4_stage"
source_archive="${stage_dir}/airo-doffy-key4-src.tar.gz"
dataset_archive="${stage_dir}/WRM_grasp_cylinder_different_sizes_lero.tar"
hf_token_file="${stage_dir}/hf_token"
wandb_netrc_file="${stage_dir}/netrc"
python_bin="/opt/conda/envs/lerobot/bin/python"
work_root="/tmp/airo_key4_work"
repo_root="${work_root}/AIRO-Doffy"
dataset_root="${work_root}/datasets/WRM_grasp_cylinder_different_sizes_lero"
output_root="/project_ghent/AIRO-Doffy/WRM_grasp_cylinder_different_sizes_lero/key4_cluster8_parallel_bs32"
wandb_project="AIRO-Doffy-WRM-Grasp"
batch_size=32
num_workers_per_policy=5

source_sha256="524a102955e50428d2ee648fd7a82d6a81f5841444851d7aafecf1543a2b74ca"
dataset_sha256="bf890b70eb5db96db654a9ed7be830c00c12ef1611866ee8d871473c9dd4b8d2"

for required_file in \
    "${source_archive}" \
    "${dataset_archive}" \
    "${hf_token_file}" \
    "${wandb_netrc_file}"; do
    if [[ ! -s "${required_file}" ]]; then
        echo "Missing staged input: ${required_file}" >&2
        exit 2
    fi
done

chmod 600 "${hf_token_file}" "${wandb_netrc_file}"
echo "${source_sha256}  ${source_archive}" | sha256sum --check --strict
echo "${dataset_sha256}  ${dataset_archive}" | sha256sum --check --strict

mkdir -p "${repo_root}" "${work_root}/datasets" "${output_root}"
tar -xzf "${source_archive}" -C "${repo_root}"
tar -xf "${dataset_archive}" -C "${work_root}/datasets"

export HF_TOKEN
HF_TOKEN="$(<"${hf_token_file}")"
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
export PYTHONUNBUFFERED=1
trap 'unset HF_TOKEN WANDB_API_KEY' EXIT

cd "${repo_root}"

"${python_bin}" -m pip install \
    --disable-pip-version-check \
    --no-deps \
    "lerobot==0.4.4"

"${python_bin}" - "${batch_size}" "${num_workers_per_policy}" <<'PY'
import os
import sys

import torch
from huggingface_hub import HfApi

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside the allocated GPU job")
probe = torch.randn(512, 512, device="cuda", dtype=torch.float16)
probe_result = probe @ probe
torch.cuda.synchronize()
print(
    "gpu", torch.cuda.get_device_name(0),
    "torch", torch.__version__,
    "cuda", torch.version.cuda,
    "probe", tuple(probe_result.shape),
)
print(
    "resources",
    "reserved_cpus", os.environ.get("GPULAB_CPUS_RESERVED", "unknown"),
    "visible_cpus", len(os.sched_getaffinity(0)),
    "batch_size", sys.argv[1],
    "num_workers_per_policy", sys.argv[2],
)
identity = HfApi(token=os.environ["HF_TOKEN"]).whoami()
print("huggingface_user", identity.get("name", "unknown"))
PY

"${python_bin}" -m unittest -q \
    policies.realman_beaver.tests.test_structured_beaver_encoder \
    policies.realman_beaver.tests.test_structured_beaver_integration \
    policies.realman_beaver.tests.test_training_config

"${python_bin}" - "${dataset_root}" <<'PY'
import json
import sys
from pathlib import Path

info = json.loads((Path(sys.argv[1]) / "meta" / "info.json").read_text())
if info.get("total_episodes") != 125:
    raise SystemExit(f"Expected 125 episodes, got {info.get('total_episodes')}")
print("dataset_episodes", info["total_episodes"], "train", 100, "validation", 25)
PY

upload_policy() {
    local repo_id="$1"
    local policy_output="$2"
    local policy_name="$3"

    "${python_bin}" - "${repo_id}" "${policy_output}" "${policy_name}" <<'PY'
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

repo_id, folder_path, policy_name = sys.argv[1:]
folder = Path(folder_path)
milestones = sorted(folder.glob(f"{policy_name}_step_*.pt"))
if len(milestones) != 4 or not (folder / "last.pt").is_file():
    raise SystemExit(
        f"Refusing incomplete upload for {policy_name}: "
        f"milestones={len(milestones)}, last={(folder / 'last.pt').is_file()}"
    )
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
api.upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path=str(folder),
    path_in_repo="checkpoints",
    ignore_patterns=["*.tmp", "wandb/**"],
    commit_message=f"Upload {policy_name} checkpoints through step 100000",
)
print("huggingface_upload_complete", repo_id)
PY
}

policy_names=(
    "dp_beaver_key4"
    "dp_beaver_key4_pca"
)
hf_repositories=(
    "IXDLI/AIRO-Doffy-WRM-Grasp-dp-beaver-key4"
    "IXDLI/AIRO-Doffy-WRM-Grasp-dp-beaver-key4-pca"
)

train_and_upload() {
    local policy_index="$1"
    local policy_name="${policy_names[policy_index]}"
    local config_path="policies/realman_beaver/configs/${policy_name}.yaml"
    local policy_output="${output_root}/${policy_name}"
    local policy_log="${policy_output}/cluster8_bs32_train.log"
    mkdir -p "${policy_output}"

    if [[ -f "${policy_output}/last.pt" ]]; then
        echo "[$(date -Is)] existing completed batch-size-32 policy found; skipping training: ${policy_name}"
    else
        echo "[$(date -Is)] starting parallel policy ${policy_name}; gpu=0; batch_size=${batch_size}; num_workers=${num_workers_per_policy}"
        set -o pipefail
        CUDA_VISIBLE_DEVICES=0 "${python_bin}" -m policies.realman_beaver.train \
            --config "${config_path}" \
            --val-episodes 50-74 \
            --wandb-project "${wandb_project}" \
            --wandb-run-name "WRM_grasp_cylinder_different_sizes_lero/${policy_name}-cluster8-bs32" \
            --dataset-root "${dataset_root}" \
            --dataset-repo-id "WRM_grasp_cylinder_different_sizes_lero" \
            --device cuda:0 \
            --batch-size "${batch_size}" \
            --num-workers "${num_workers_per_policy}" \
            --output-dir "${policy_output}" 2>&1 | tee "${policy_log}"
    fi

    echo "[$(date -Is)] uploading ${policy_name} checkpoints to Hugging Face"
    upload_policy \
        "${hf_repositories[policy_index]}" \
        "${policy_output}" \
        "${policy_name}"
    echo "[$(date -Is)] completed parallel policy ${policy_name}"
}

declare -a policy_pids=()
for policy_index in "${!policy_names[@]}"; do
    train_and_upload "${policy_index}" &
    policy_pids+=("$!")
done

failed=0
for policy_index in "${!policy_names[@]}"; do
    if wait "${policy_pids[policy_index]}"; then
        echo "[$(date -Is)] parallel worker succeeded: ${policy_names[policy_index]}"
    else
        echo "[$(date -Is)] parallel worker failed: ${policy_names[policy_index]}" >&2
        failed=1
    fi
done

if ((failed)); then
    exit 1
fi

echo "[$(date -Is)] both parallel Cluster8 batch-size-32 Key4 policies completed and uploaded"
