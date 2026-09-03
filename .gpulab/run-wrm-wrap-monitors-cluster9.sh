#!/usr/bin/env bash

set -Eeuo pipefail

stage_dir="/tmp/airo_wrm_wrap_monitors_stage"
source_archive="${stage_dir}/airo-doffy-wrm-wrap-monitors-src.tar.gz"
source_manifest="${stage_dir}/airo-doffy-wrm-wrap-monitors-src.sha256"
dataset_archive="${stage_dir}/WRM_grasp_cylinder_different_sizes_lero_tightness-monitors.tar"
dataset_manifest="${stage_dir}/WRM_grasp_cylinder_different_sizes_lero_tightness-monitors.sha256"
hf_token_file="${stage_dir}/hf_token"
python_bin="/opt/conda/envs/lerobot/bin/python"
work_root="/tmp/airo_wrm_wrap_monitors_work"
repo_root="${work_root}/AIRO-Doffy"
dataset_root="${work_root}/datasets/WRM_grasp_cylinder_different_sizes_lero_tightness"
output_root="/project_ghent/AIRO-Doffy/WRM_grasp_cylinder_different_sizes_lero_tightness/wrm_wrap_monitors_cluster9"
training_log="${output_root}/train.log"
primary_repo="IXDLI/AIRO-Doffy-WRM-Grasp-WRM-wrap-monitor"
backup_repo="IXDLI/AIRO-Doffy-WRM-Grasp-WRM-wrap-monitor-backup"

for required_file in \
    "${source_archive}" \
    "${source_manifest}" \
    "${dataset_archive}" \
    "${dataset_manifest}" \
    "${hf_token_file}"; do
    if [[ ! -s "${required_file}" ]]; then
        echo "Missing staged input: ${required_file}" >&2
        exit 2
    fi
done

chmod 600 "${hf_token_file}"
(
    cd "${stage_dir}"
    sha256sum --check --strict "$(basename "${source_manifest}")"
    sha256sum --check --strict "$(basename "${dataset_manifest}")"
)

mkdir -p "${repo_root}" "${work_root}/datasets" "${output_root}"
tar -xzf "${source_archive}" -C "${repo_root}"
tar -xf "${dataset_archive}" -C "${work_root}/datasets"

export HF_TOKEN
HF_TOKEN="$(<"${hf_token_file}")"
export HF_HOME="${work_root}/huggingface"
export PYTHONUNBUFFERED=1
trap 'unset HF_TOKEN' EXIT

cd "${repo_root}"
"${python_bin}" -m pip install \
    --disable-pip-version-check \
    --no-cache-dir \
    --upgrade \
    "torch==2.7.0" \
    "torchvision==0.22.0" \
    --index-url https://download.pytorch.org/whl/cu128
"${python_bin}" -m pip install \
    --disable-pip-version-check \
    --upgrade \
    "pyarrow>=16" \
    "huggingface_hub>=0.34.2,<0.36"
"${python_bin}" -m pip install \
    --disable-pip-version-check \
    --no-deps \
    "lerobot==0.4.4"

"${python_bin}" - "${dataset_root}" <<'PY'
import json
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
info = json.loads((root / "meta" / "info.json").read_text())
if info.get("total_episodes") != 125:
    raise SystemExit(f"Expected 125 episodes, got {info.get('total_episodes')}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit(f"Expected one CUDA GPU, got {torch.cuda.device_count()}")
probe = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
torch.cuda.synchronize()
print(
    "gpu", torch.cuda.get_device_name(0),
    "torch", torch.__version__,
    "cuda", torch.version.cuda,
    "probe_mean", float((probe @ probe).mean()),
)
PY

"${python_bin}" -m unittest \
    policies.realman_beaver.tests.test_wrm_wrap_monitor \
    -v

set -o pipefail
"${python_bin}" -m policies.realman_beaver.train_beaver_monitors \
    --dataset-root "${dataset_root}" \
    --output-root "${output_root}" \
    --device cuda:0 \
    --seed 42 2>&1 | tee "${training_log}"

cp policies/realman_beaver/MODEL_CARD_WRM_wrap_monitor.md \
    "${output_root}/WRM_wrap_monitor/README.md"
cp policies/realman_beaver/MODEL_CARD_WRM_wrap_monitor_backup.md \
    "${output_root}/WRM_wrap_monitor_backup/README.md"
for variant in WRM_wrap_monitor WRM_wrap_monitor_backup; do
    cp "${output_root}/dataset_manifest.json" "${output_root}/${variant}/"
    cp "${output_root}/summary.json" "${output_root}/${variant}/"
done

"${python_bin}" - \
    "${primary_repo}" "${output_root}/WRM_wrap_monitor" \
    "${backup_repo}" "${output_root}/WRM_wrap_monitor_backup" <<'PY'
import os
import sys

from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
for repo_id, folder in zip(sys.argv[1::2], sys.argv[2::2], strict=True):
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=folder,
        commit_message="Train Beaver execution monitor on GPUlab",
    )
    print(f"uploaded=https://huggingface.co/{repo_id}")
PY

echo "[$(date -Is)] WRM_wrap monitor training and upload complete"
