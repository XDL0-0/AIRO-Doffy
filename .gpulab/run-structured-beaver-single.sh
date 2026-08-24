#!/usr/bin/env bash

set -Eeuo pipefail

: "${POLICY_NAME:?POLICY_NAME must be set by the GPUlab job}"
: "${HF_REPO_ID:?HF_REPO_ID must be set by the GPUlab job}"

stage_dir="/tmp/airo_structured_beaver_stage"
source_archive="${stage_dir}/airo-doffy-structured-beaver-src.tar.gz"
hf_token_file="${stage_dir}/hf_token"
wandb_netrc_file="${stage_dir}/netrc"
python_bin="/opt/conda/envs/lerobot/bin/python"
work_root="/tmp/airo_structured_beaver_work"
repo_root="${work_root}/AIRO-Doffy"
dataset_root="${work_root}/datasets/WRM_grasp_cylinder_different_sizes_lero"
output_root="/project_ghent/AIRO-Doffy/WRM_grasp_cylinder_different_sizes_lero/structured_beaver_cluster6_parallel"
policy_output="${output_root}/${POLICY_NAME}"
policy_log="${policy_output}/cluster6_train.log"
wandb_project="AIRO-Doffy-WRM-Grasp"
source_sha256="ee823cacf2bde568a85fcdc31fae5c1d29908b3cd65655ee50449766f9e194b1"

for required_file in "${source_archive}" "${hf_token_file}" "${wandb_netrc_file}"; do
    if [[ ! -s "${required_file}" ]]; then
        echo "Missing staged input: ${required_file}" >&2
        exit 2
    fi
done

chmod 600 "${hf_token_file}" "${wandb_netrc_file}"
echo "${source_sha256}  ${source_archive}" | sha256sum --check --strict

mkdir -p "${repo_root}" "${work_root}/datasets" "${policy_output}"
tar -xzf "${source_archive}" -C "${repo_root}"

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
from huggingface_hub import HfApi, snapshot_download

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside the allocated GPU job")
print("gpu", torch.cuda.get_device_name(0), "cuda", torch.version.cuda)
identity = HfApi(token=os.environ["HF_TOKEN"]).whoami()
print("huggingface_user", identity.get("name", "unknown"))

dataset_root = Path(sys.argv[1])
snapshot_download(
    repo_id="IXDLI/WRM_grasp_cylinder_different_sizes_lero",
    repo_type="dataset",
    local_dir=dataset_root,
    token=os.environ["HF_TOKEN"],
)
info = json.loads((dataset_root / "meta" / "info.json").read_text())
if info.get("total_episodes") != 125:
    raise SystemExit(f"Expected 125 episodes, got {info.get('total_episodes')}")
print("dataset_episodes", info["total_episodes"], "train", 100, "validation", 25)
PY

cd "${repo_root}"
"${python_bin}" -m unittest -q \
    policies.realman_beaver.tests.test_structured_beaver_encoder \
    policies.realman_beaver.tests.test_structured_beaver_integration

if [[ -f "${policy_output}/last.pt" ]]; then
    echo "[$(date -Is)] existing completed policy found; skipping training: ${POLICY_NAME}"
else
    echo "[$(date -Is)] starting parallel policy ${POLICY_NAME}"
    set -o pipefail
    "${python_bin}" -m policies.realman_beaver.train \
        --config "policies/realman_beaver/configs/${POLICY_NAME}.yaml" \
        --val-episodes 50-74 \
        --wandb-project "${wandb_project}" \
        --wandb-run-name "WRM_grasp_cylinder_different_sizes_lero/${POLICY_NAME}-cluster6" \
        --dataset-root "${dataset_root}" \
        --dataset-repo-id "WRM_grasp_cylinder_different_sizes_lero" \
        --device cuda:0 \
        --num-workers 2 \
        --output-dir "${policy_output}" 2>&1 | tee "${policy_log}"
fi

"${python_bin}" - "${HF_REPO_ID}" "${policy_output}" "${POLICY_NAME}" <<'PY'
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
HfApi(token=os.environ["HF_TOKEN"]).upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path=str(folder),
    path_in_repo="checkpoints",
    ignore_patterns=["*.tmp", "wandb/**"],
    commit_message=f"Upload {policy_name} checkpoints through step 100000",
)
print("huggingface_upload_complete", repo_id)
PY

echo "[$(date -Is)] completed parallel policy ${POLICY_NAME}"
