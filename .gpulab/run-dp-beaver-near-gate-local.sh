#!/usr/bin/env bash

set -Eeuo pipefail

repo_root="/home/yuyuan/AIRO-Doffy"
python_bin="/home/yuyuan/miniconda3/envs/airo-doffy/bin/python"
dataset_root="${repo_root}/datasets/WRM_grasp_cylinder_different_sizes_lero"
output_dir="${repo_root}/policies/output/WRM_grasp_cylinder_different_sizes_lero/structured_beaver_local_parallel/dp_beaver_near_gate"
log_path="${output_dir}/local_train.log"
hf_repo_id="IXDLI/AIRO-Doffy-WRM-Grasp-dp-beaver-near-gate"

mkdir -p "${output_dir}"
cd "${repo_root}"
export PYTHONUNBUFFERED=1

if [[ ! -f "${output_dir}/last.pt" ]]; then
    resume_args=()
    latest_checkpoint="$(find "${output_dir}" -maxdepth 1 -type f -name 'dp_beaver_near_gate_step_*.pt' -print | sort | tail -1)"
    if [[ -n "${latest_checkpoint}" ]]; then
        echo "Resuming dp_beaver_near_gate from ${latest_checkpoint}"
        resume_args=(--resume-from "${latest_checkpoint}")
    fi
    set -o pipefail
    "${python_bin}" -m policies.realman_beaver.train \
        --config policies/realman_beaver/configs/dp_beaver_near_gate.yaml \
        "${resume_args[@]}" \
        --val-episodes 50-74 \
        --wandb-project AIRO-Doffy-WRM-Grasp \
        --wandb-run-name WRM_grasp_cylinder_different_sizes_lero/dp_beaver_near_gate-local \
        --wandb-run-id nnlwfv1n \
        --dataset-root "${dataset_root}" \
        --dataset-repo-id WRM_grasp_cylinder_different_sizes_lero \
        --device cuda:0 \
        --num-workers 8 \
        --output-dir "${output_dir}" 2>&1 | tee "${log_path}"
fi

"${python_bin}" - "${hf_repo_id}" "${output_dir}" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import HfApi

repo_id, folder_path = sys.argv[1:]
folder = Path(folder_path)
milestones = sorted(folder.glob("dp_beaver_near_gate_step_*.pt"))
if len(milestones) != 4 or not (folder / "last.pt").is_file():
    raise SystemExit(
        f"Refusing incomplete upload: milestones={len(milestones)}, "
        f"last={(folder / 'last.pt').is_file()}"
    )
HfApi().upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path=str(folder),
    path_in_repo="checkpoints",
    ignore_patterns=["*.tmp", "wandb/**"],
    commit_message="Upload dp_beaver_near_gate checkpoints through step 100000",
)
print("huggingface_upload_complete", repo_id)
PY
