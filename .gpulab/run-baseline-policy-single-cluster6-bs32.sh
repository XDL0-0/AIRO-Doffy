#!/usr/bin/env bash

set -Eeuo pipefail

: "${POLICY_NAME:?POLICY_NAME must be set by the GPUlab job}"
: "${HF_REPO_ID:?HF_REPO_ID must be set by the GPUlab job}"
case "${POLICY_NAME}" in
    original_dp|dp_beaver|fm|fm_beaver)
        ;;
    *)
        echo "Unsupported baseline policy: ${POLICY_NAME}" >&2
        exit 2
        ;;
esac

stage_dir="/tmp/airo_baseline_cluster6_stage"
source_archive="${stage_dir}/airo-doffy-structured-beaver-src-bs32.tar.gz"
hf_token_file="${stage_dir}/hf_token"
wandb_netrc_file="${stage_dir}/netrc"
python_bin="/opt/conda/envs/lerobot/bin/python"
work_root="/tmp/airo_baseline_cluster6_work_bs32"
repo_root="${work_root}/AIRO-Doffy"
dataset_root="${work_root}/datasets/WRM_grasp_cylinder_different_sizes_lero"
output_root="/project_ghent/AIRO-Doffy/WRM_grasp_cylinder_different_sizes_lero/baseline_cluster6_parallel_bs32"
policy_output="${output_root}/${POLICY_NAME}"
policy_log="${policy_output}/cluster6_bs32_train.log"
wandb_project="AIRO-Doffy-WRM-Grasp"
batch_size=32
num_workers=4
source_sha256="48647fe329964dae872c05d22950c11c9fc0ab6c9371dc86c395650ca9e6ab96"

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
from huggingface_hub import HfApi, snapshot_download

dataset_root = Path(sys.argv[1])
batch_size = int(sys.argv[2])
num_workers = int(sys.argv[3])
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
    "batch_size", batch_size,
    "num_workers", num_workers,
)
identity = HfApi(token=os.environ["HF_TOKEN"]).whoami()
print("huggingface_user", identity.get("name", "unknown"))

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

"${python_bin}" -m unittest -q \
    policies.realman_beaver.tests.test_policy \
    policies.realman_beaver.tests.test_training_config

if [[ -f "${policy_output}/last.pt" ]]; then
    echo "[$(date -Is)] existing completed Cluster6 policy found; skipping training: ${POLICY_NAME}"
else
    echo "[$(date -Is)] starting ${POLICY_NAME}; batch_size=${batch_size}; num_workers=${num_workers}"
    set -o pipefail
    "${python_bin}" -m policies.realman_beaver.train \
        --config "policies/realman_beaver/configs/${POLICY_NAME}.yaml" \
        --val-episodes 50-74 \
        --wandb-project "${wandb_project}" \
        --wandb-run-name "WRM_grasp_cylinder_different_sizes_lero/${POLICY_NAME}-cluster6-bs32" \
        --dataset-root "${dataset_root}" \
        --dataset-repo-id "WRM_grasp_cylinder_different_sizes_lero" \
        --device cuda:0 \
        --batch-size "${batch_size}" \
        --num-workers "${num_workers}" \
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
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
api.upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path=str(folder),
    path_in_repo="checkpoints",
    ignore_patterns=["*.tmp", "wandb/**"],
    commit_message=f"Upload Cluster6 batch-size-32 {policy_name} checkpoints through step 100000",
)
print("huggingface_upload_complete", repo_id)
PY

echo "[$(date -Is)] completed Cluster6 batch-size-32 policy ${POLICY_NAME}"
