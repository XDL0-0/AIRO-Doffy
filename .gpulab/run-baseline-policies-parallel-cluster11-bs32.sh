#!/usr/bin/env bash

set -Eeuo pipefail

stage_dir="/tmp/airo_baseline_cluster11_stage"
source_archive="${stage_dir}/airo-doffy-structured-beaver-src-bs32.tar.gz"
dataset_archive="${stage_dir}/WRM_grasp_cylinder_different_sizes_lero.tar"
hf_token_file="${stage_dir}/hf_token"
wandb_netrc_file="${stage_dir}/netrc"
python_bin="/opt/conda/envs/lerobot/bin/python"
work_root="/tmp/airo_baseline_cluster11_work_bs32"
repo_root="${work_root}/AIRO-Doffy"
dataset_root="${work_root}/datasets/WRM_grasp_cylinder_different_sizes_lero"
output_root="/project_ghent/AIRO-Doffy/WRM_grasp_cylinder_different_sizes_lero/baseline_cluster11_parallel_bs32"
wandb_project="AIRO-Doffy-WRM-Grasp"
batch_size=32
num_workers_per_policy=4
expected_gpus=1

source_sha256="48647fe329964dae872c05d22950c11c9fc0ab6c9371dc86c395650ca9e6ab96"
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
export WANDB_DIR="${output_root}/wandb"
export PYTHONUNBUFFERED=1
trap 'unset HF_TOKEN WANDB_API_KEY' EXIT

cd "${repo_root}"

"${python_bin}" -m pip install \
    --disable-pip-version-check \
    --upgrade \
    --force-reinstall \
    "torch==2.7.1" \
    "torchvision==0.22.1" \
    "fsspec==2025.9.0" \
    --index-url "https://download.pytorch.org/whl/cu128"

"${python_bin}" -m pip install \
    --disable-pip-version-check \
    --no-deps \
    "lerobot==0.4.4"

"${python_bin}" - "${batch_size}" "${num_workers_per_policy}" "${expected_gpus}" <<'PY'
import os
import sys

import torch
from huggingface_hub import HfApi

batch_size, workers, expected_gpus = map(int, sys.argv[1:])
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside the allocated GPU job")
if torch.cuda.device_count() != expected_gpus:
    raise SystemExit(
        f"Expected {expected_gpus} visible GPUs, got {torch.cuda.device_count()}"
    )
architectures = torch.cuda.get_arch_list()
if "sm_90" not in architectures:
    raise SystemExit(
        f"PyTorch {torch.__version__} does not include H200 sm_90: {architectures}"
    )
for device_index in range(expected_gpus):
    device = torch.device(f"cuda:{device_index}")
    probe = torch.randn(512, 512, device=device, dtype=torch.float16)
    probe_result = probe @ probe
    torch.cuda.synchronize(device)
    print(
        "gpu", device_index, torch.cuda.get_device_name(device_index),
        "memory_gb", round(torch.cuda.get_device_properties(device_index).total_memory / 2**30, 1),
        "probe", tuple(probe_result.shape),
    )
print(
    "runtime",
    "torch", torch.__version__,
    "cuda", torch.version.cuda,
    "architectures", architectures,
)
print(
    "resources",
    "reserved_cpus", os.environ.get("GPULAB_CPUS_RESERVED", "unknown"),
    "visible_cpus", len(os.sched_getaffinity(0)),
    "batch_size_per_policy", batch_size,
    "workers_per_policy", workers,
    "parallel_policies", 4,
)
identity = HfApi(token=os.environ["HF_TOKEN"]).whoami()
print("huggingface_user", identity.get("name", "unknown"))
PY

"${python_bin}" -m unittest -q \
    policies.realman_beaver.tests.test_policy \
    policies.realman_beaver.tests.test_training_config

"${python_bin}" - "${dataset_root}" <<'PY'
import json
import sys
from pathlib import Path

info_path = Path(sys.argv[1]) / "meta" / "info.json"
info = json.loads(info_path.read_text())
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
    commit_message=f"Upload Cluster11 batch-size-32 {policy_name} checkpoints through step 100000",
)
print("huggingface_upload_complete", repo_id)
PY
}

policy_names=(
    "original_dp"
    "dp_beaver"
    "fm"
    "fm_beaver"
)
hf_repositories=(
    "IXDLI/AIRO-Doffy-WRM-Grasp-original-dp"
    "IXDLI/AIRO-Doffy-WRM-Grasp-dp-beaver"
    "IXDLI/AIRO-Doffy-WRM-Grasp-fm"
    "IXDLI/AIRO-Doffy-WRM-Grasp-fm-beaver"
)

train_and_upload() {
    local policy_index="$1"
    local policy_name="${policy_names[policy_index]}"
    local gpu_index=0
    local config_path="policies/realman_beaver/configs/${policy_name}.yaml"
    local policy_output="${output_root}/${policy_name}"
    local policy_log="${policy_output}/cluster11_bs32_train.log"
    mkdir -p "${policy_output}"

    if [[ -f "${policy_output}/last.pt" ]]; then
        echo "[$(date -Is)] existing completed Cluster11 policy found; skipping training: ${policy_name}"
    else
        echo "[$(date -Is)] starting parallel policy ${policy_name}; gpu=${gpu_index}; batch_size=${batch_size}; num_workers=${num_workers_per_policy}"
        set -o pipefail
        CUDA_VISIBLE_DEVICES="${gpu_index}" "${python_bin}" -m policies.realman_beaver.train \
            --config "${config_path}" \
            --val-episodes 50-74 \
            --wandb-project "${wandb_project}" \
            --wandb-run-name "WRM_grasp_cylinder_different_sizes_lero/${policy_name}-cluster11-bs32" \
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

pids=()
for policy_index in "${!policy_names[@]}"; do
    train_and_upload "${policy_index}" &
    pids+=("$!")
done

failed=0
for policy_index in "${!policy_names[@]}"; do
    if wait "${pids[policy_index]}"; then
        echo "[$(date -Is)] worker succeeded: ${policy_names[policy_index]}"
    else
        echo "[$(date -Is)] worker failed: ${policy_names[policy_index]}" >&2
        failed=1
    fi
done

if (( failed != 0 )); then
    echo "One or more parallel policy workers failed" >&2
    exit 1
fi

echo "[$(date -Is)] all four Cluster11 batch-size-32 policies completed and uploaded"
