#!/usr/bin/env bash

# WRM_claude architecture-selection run: 100 train / 25 held-out episodes.
# Runs the focused unit-test module first, then 100k steps with milestone
# checkpoints every 10k. The selection rollout order is derived from the
# milestone curves afterwards (see the report's checkpoint-selection section).

set -Eeuo pipefail

stage_dir="/tmp/airo_wrm_claude_select_cluster12_stage"
source_archive="${stage_dir}/airo-doffy-wrm-claude-select-src.tar.gz"
source_manifest="${stage_dir}/airo-doffy-wrm-claude-select-src.sha256"
dataset_archive="${stage_dir}/WRM_grasp_cylinder_different_sizes_lero_tightness.tar"
dataset_manifest="${stage_dir}/WRM_grasp_cylinder_different_sizes_lero_tightness.sha256"
hf_token_file="${stage_dir}/hf_token"
wandb_netrc_file="${stage_dir}/netrc"
python_bin="/opt/conda/envs/lerobot/bin/python"
work_root="/tmp/airo_wrm_claude_select_cluster12_work"
repo_root="${work_root}/AIRO-Doffy"
dataset_root="${work_root}/datasets/WRM_grasp_cylinder_different_sizes_lero_tightness"
output_root="/project_ghent/AIRO-Doffy/WRM_grasp_cylinder_different_sizes_lero_tightness/wrm_claude_select_cluster12_bs32"
policy_output="${output_root}/WRM_claude_select"
policy_log="${policy_output}/cluster12_select_bs32_train.log"
wandb_project="AIRO-Doffy-WRM-Grasp"
batch_size=32
num_workers=8
upload_hf="${UPLOAD_HF:-0}"
hf_repo_id="${HF_REPO_ID:-IXDLI/AIRO-Doffy-WRM-Grasp-WRM-claude-select}"

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
export WANDB_DIR="${policy_output}/wandb"
export PYTHONUNBUFFERED=1
trap 'unset HF_TOKEN WANDB_API_KEY' EXIT

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

"${python_bin}" - "${dataset_root}" "${batch_size}" "${num_workers}" <<'PY'
import json
import os
import sys
from pathlib import Path

import torch

dataset_root = Path(sys.argv[1])
batch_size = int(sys.argv[2])
num_workers = int(sys.argv[3])
info = json.loads((dataset_root / "meta" / "info.json").read_text())
if info.get("total_episodes") != 125:
    raise SystemExit(f"Expected 125 episodes, got {info.get('total_episodes')}")
if "tightness" not in info.get("features", {}):
    raise SystemExit("The staged dataset does not expose the tightness label")
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
    "batch_size", batch_size,
    "num_workers", num_workers,
)
print("dataset_episodes", info["total_episodes"], "train", 100, "validation", 25)
PY

"${python_bin}" -m unittest \
    policies.realman_beaver.tests.test_wrm_claude -v

if [[ -f "${policy_output}/last.pt" ]]; then
    echo "[$(date -Is)] existing completed WRM_claude selection run found; skipping"
else
    echo "[$(date -Is)] starting WRM_claude selection run; batch_size=${batch_size}"
    set -o pipefail
    "${python_bin}" -m policies.realman_beaver.train \
        --config policies/realman_beaver/configs/WRM_claude.yaml \
        --val-episodes 20-24,45-49,70-74,95-99,120-124 \
        --wandb-project "${wandb_project}" \
        --wandb-run-name "WRM_grasp_cylinder_different_sizes_lero_tightness/WRM_claude-select-cluster12-bs32" \
        --dataset-root "${dataset_root}" \
        --dataset-repo-id "WRM_grasp_cylinder_different_sizes_lero_tightness" \
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
milestones = sorted(folder.glob("WRM_claude_step_*.pt"))
last = folder / "last.pt"
if len(milestones) != 10 or not last.is_file():
    raise SystemExit(f"Incomplete run: milestones={len(milestones)}, last={last.is_file()}")
checkpoint = torch.load(last, map_location="cpu", weights_only=True)
if checkpoint.get("kind") != "WRM_claude":
    raise SystemExit(f"Unexpected checkpoint kind: {checkpoint.get('kind')}")
if checkpoint.get("global_step") != 100000:
    raise SystemExit(f"Expected step 100000, got {checkpoint.get('global_step')}")
print("training_complete", last, "step", checkpoint["global_step"])
PY

if [[ "${upload_hf}" == "1" ]]; then
    "${python_bin}" - "${hf_repo_id}" "${policy_output}" <<'PY'
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

repo_id, folder_path = sys.argv[1:]
folder = Path(folder_path)
milestones = sorted(folder.glob("WRM_claude_step_*.pt"))
if len(milestones) != 10 or not (folder / "last.pt").is_file():
    raise SystemExit(
        "Refusing incomplete WRM_claude upload: "
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
    commit_message="Upload WRM_claude selection milestones through step 100000",
)
print("huggingface_upload_complete", repo_id)
PY
else
    echo "UPLOAD_HF=${upload_hf}; keeping checkpoints only on persistent storage"
fi

echo "[$(date -Is)] completed Cluster12 batch-size-32 WRM_claude selection run"
