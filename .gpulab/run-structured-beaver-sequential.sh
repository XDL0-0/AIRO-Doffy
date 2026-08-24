#!/usr/bin/env bash

set -Eeuo pipefail

stage_dir="/tmp/airo_structured_beaver_stage"
source_archive="${stage_dir}/airo-doffy-structured-beaver-src.tar.gz"
dataset_archive="${stage_dir}/WRM_grasp_cylinder_different_sizes_lero.tar"
hf_token_file="${stage_dir}/hf_token"
wandb_netrc_file="${stage_dir}/netrc"
python_bin="/opt/conda/envs/lerobot/bin/python"
work_root="/tmp/airo_structured_beaver_work"
repo_root="${work_root}/AIRO-Doffy"
dataset_root="${work_root}/datasets/WRM_grasp_cylinder_different_sizes_lero"
output_root="/project_ghent/AIRO-Doffy/WRM_grasp_cylinder_different_sizes_lero/structured_beaver"
wandb_project="AIRO-Doffy-WRM-Grasp"

source_sha256="ee823cacf2bde568a85fcdc31fae5c1d29908b3cd65655ee50449766f9e194b1"
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
    --no-deps \
    "lerobot==0.4.4"

"${python_bin}" - <<'PY'
import os

import torch
from huggingface_hub import HfApi

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside the allocated GPU job")
print("gpu", torch.cuda.get_device_name(0), "cuda", torch.version.cuda)
identity = HfApi(token=os.environ["HF_TOKEN"]).whoami()
print("huggingface_user", identity.get("name", "unknown"))
PY

"${python_bin}" -m unittest -q \
    policies.realman_beaver.tests.test_structured_beaver_encoder \
    policies.realman_beaver.tests.test_structured_beaver_integration

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
    "dp_beaver_enc"
    "dp_beaver_near"
    "dp_beaver_near_gate"
)
hf_repositories=(
    "IXDLI/AIRO-Doffy-WRM-Grasp-dp-beaver-enc"
    "IXDLI/AIRO-Doffy-WRM-Grasp-dp-beaver-near"
    "IXDLI/AIRO-Doffy-WRM-Grasp-dp-beaver-near-gate"
)

for policy_index in "${!policy_names[@]}"; do
    policy_name="${policy_names[policy_index]}"
    config_path="policies/realman_beaver/configs/${policy_name}.yaml"
    policy_output="${output_root}/${policy_name}"
    policy_log="${policy_output}/cluster5_train.log"
    mkdir -p "${policy_output}"

    if [[ -f "${policy_output}/last.pt" ]]; then
        echo "[$(date -Is)] existing completed policy found; skipping training: ${policy_name}"
    else
        echo "[$(date -Is)] starting serial policy ${policy_name}"
        set -o pipefail
        "${python_bin}" -m policies.realman_beaver.train \
            --config "${config_path}" \
            --val-episodes 50-74 \
            --wandb-project "${wandb_project}" \
            --wandb-run-name "WRM_grasp_cylinder_different_sizes_lero/${policy_name}" \
            --dataset-root "${dataset_root}" \
            --dataset-repo-id "WRM_grasp_cylinder_different_sizes_lero" \
            --device cuda:0 \
            --num-workers 3 \
            --output-dir "${policy_output}" 2>&1 | tee "${policy_log}"
    fi

    echo "[$(date -Is)] uploading ${policy_name} checkpoints to Hugging Face"
    upload_policy \
        "${hf_repositories[policy_index]}" \
        "${policy_output}" \
        "${policy_name}"
    echo "[$(date -Is)] completed serial policy ${policy_name}"
done

echo "[$(date -Is)] all three structured Beaver policies completed and uploaded"
