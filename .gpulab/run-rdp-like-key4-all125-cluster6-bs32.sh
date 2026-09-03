#!/usr/bin/env bash

set -Eeuo pipefail

: "${HF_REPO_ID:?HF_REPO_ID must be set by the GPUlab job}"
stage_dir="/tmp/airo_rdp_like_key4_stage"
source_archive="${stage_dir}/airo-doffy-rdp-like-key4-src.tar.gz"
source_manifest="${stage_dir}/airo-doffy-rdp-like-key4-src.sha256"
dataset_archive="${stage_dir}/WRM_grasp_cylinder_different_sizes_lero_tightness.tar"
dataset_manifest="${stage_dir}/WRM_grasp_cylinder_different_sizes_lero_tightness.sha256"
hf_token_file="${stage_dir}/hf_token"
wandb_netrc_file="${stage_dir}/netrc"
python_bin="/opt/conda/envs/lerobot/bin/python"
work_root="/tmp/airo_rdp_like_key4_work"
repo_root="${work_root}/AIRO-Doffy"
dataset_root="${work_root}/datasets/WRM_grasp_cylinder_different_sizes_lero_tightness"
output_root="/project_ghent/AIRO-Doffy/WRM_grasp_cylinder_different_sizes_lero_tightness/rdp_like_key4_all125_cluster6_bs32"
policy_output="${output_root}/rdp_like_key4_all_train"
policy_log="${policy_output}/cluster6_bs32_train.log"
wandb_project="AIRO-Doffy-WRM-Grasp"
batch_size=32
num_workers=8

for required_file in \
    "${source_archive}" \
    "${source_manifest}" \
    "${dataset_archive}" \
    "${dataset_manifest}" \
    "${hf_token_file}" \
    "${wandb_netrc_file}"; do
    if [[ ! -s "${required_file}" ]]; then
        echo "Missing staged input: ${required_file}" >&2
        exit 2
    fi
done

chmod 600 "${hf_token_file}" "${wandb_netrc_file}"
cd "${stage_dir}"
sha256sum --check --strict "$(basename "${source_manifest}")"
sha256sum --check --strict "$(basename "${dataset_manifest}")"

mkdir -p "${repo_root}" "${work_root}/datasets" "${policy_output}"
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
export WANDB_DIR="${policy_output}/wandb"
export PYTHONUNBUFFERED=1
trap 'unset HF_TOKEN WANDB_API_KEY' EXIT

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
from huggingface_hub import HfApi

dataset_root = Path(sys.argv[1])
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside the allocated GPU job")
architectures = torch.cuda.get_arch_list()
if "sm_70" not in architectures:
    raise SystemExit(
        f"PyTorch {torch.__version__} does not include V100 sm_70: {architectures}"
    )
probe = torch.randn(512, 512, device="cuda", dtype=torch.float16)
probe_result = probe @ probe
torch.cuda.synchronize()
info = json.loads((dataset_root / "meta" / "info.json").read_text())
if info.get("total_episodes") != 125:
    raise SystemExit(f"Expected 125 episodes, got {info.get('total_episodes')}")
if not (dataset_root / "meta" / "stats.json").is_file():
    raise SystemExit("Dataset is missing meta/stats.json")
print(
    "gpu", torch.cuda.get_device_name(0),
    "memory_gb", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
    "torch", torch.__version__,
    "cuda", torch.version.cuda,
    "architectures", architectures,
    "probe", tuple(probe_result.shape),
)
print(
    "resources",
    "reserved_cpus", os.environ.get("GPULAB_CPUS_RESERVED", "unknown"),
    "visible_cpus", len(os.sched_getaffinity(0)),
    "batch_size", sys.argv[2],
    "num_workers", sys.argv[3],
)
print("dataset_episodes", info["total_episodes"], "train", 125, "validation", 0)
identity = HfApi(token=os.environ["HF_TOKEN"]).whoami()
print("huggingface_user", identity.get("name", "unknown"))
PY

"${python_bin}" -m unittest -q \
    policies.realman_beaver.tests.test_policy \
    policies.realman_beaver.tests.test_training_config

if [[ -f "${policy_output}/last.pt" ]]; then
    echo "[$(date -Is)] completed RDP-like Key4 checkpoint exists; skipping training"
else
    echo "[$(date -Is)] starting rdp_like_key4_all_train; episodes=125; batch_size=${batch_size}; num_workers=${num_workers}"
    set -o pipefail
    "${python_bin}" -m policies.realman_beaver.train \
        --config policies/realman_beaver/configs/rdp_like_key4_all_train.yaml \
        --wandb-project "${wandb_project}" \
        --wandb-run-name "WRM_grasp_cylinder_different_sizes_lero_tightness/rdp_like_key4-all125-cluster6-bs32" \
        --dataset-root "${dataset_root}" \
        --dataset-repo-id "WRM_grasp_cylinder_different_sizes_lero_tightness" \
        --device cuda:0 \
        --batch-size "${batch_size}" \
        --num-workers "${num_workers}" \
        --max-steps 100000 \
        --output-dir "${policy_output}" 2>&1 | tee "${policy_log}"
fi

if [[ "${UPLOAD_HF:-1}" == "1" ]]; then
    "${python_bin}" - "${HF_REPO_ID}" "${policy_output}" <<'PY'
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

repo_id, folder_path = sys.argv[1:]
folder = Path(folder_path)
tokenizer_milestones = sorted(folder.glob("tokenizer_step_*.pt"))
latent_milestones = sorted(folder.glob("latent_dp_step_*.pt"))
required = (folder / "tokenizer_last.pt", folder / "last.pt")
if len(tokenizer_milestones) != 4 or len(latent_milestones) != 4 or not all(
    path.is_file() for path in required
):
    raise SystemExit(
        "Refusing incomplete upload: "
        f"tokenizer_milestones={len(tokenizer_milestones)}, "
        f"latent_milestones={len(latent_milestones)}, "
        f"tokenizer_last={required[0].is_file()}, last={required[1].is_file()}"
    )
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
api.upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path=str(folder),
    path_in_repo="checkpoints",
    ignore_patterns=["*.tmp", "wandb/**"],
    commit_message="Upload all-125 batch-size-32 RDP-like Key4 tokenizer and latent-DP through 100k steps",
)
print("huggingface_upload_complete", repo_id)
PY
fi

echo "[$(date -Is)] completed all-125 batch-size-32 RDP-like Key4 training"
