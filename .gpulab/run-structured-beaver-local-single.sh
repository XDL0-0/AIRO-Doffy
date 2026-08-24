#!/usr/bin/env bash

set -Eeuo pipefail

: "${POLICY_NAME:?POLICY_NAME must be set}"
: "${HF_REPO_ID:?HF_REPO_ID must be set}"
case "${POLICY_NAME}" in
    dp_beaver_enc|dp_beaver_near)
        ;;
    *)
        echo "Unsupported local policy: ${POLICY_NAME}" >&2
        exit 2
        ;;
esac

repo_root="/home/yuyuan/AIRO-Doffy"
python_bin="/home/yuyuan/miniconda3/envs/airo-doffy/bin/python"
dataset_root="${repo_root}/datasets/WRM_grasp_cylinder_different_sizes_lero"
output_dir="${repo_root}/policies/output/WRM_grasp_cylinder_different_sizes_lero/structured_beaver_local_parallel/${POLICY_NAME}"
log_path="${output_dir}/local_train.log"

mkdir -p "${output_dir}"
cd "${repo_root}"
export PYTHONUNBUFFERED=1

if [[ ! -f "${output_dir}/last.pt" ]]; then
    set -o pipefail
    "${python_bin}" -m policies.realman_beaver.train \
        --config "policies/realman_beaver/configs/${POLICY_NAME}.yaml" \
        --val-episodes 50-74 \
        --wandb-project AIRO-Doffy-WRM-Grasp \
        --wandb-run-name "WRM_grasp_cylinder_different_sizes_lero/${POLICY_NAME}-local" \
        --dataset-root "${dataset_root}" \
        --dataset-repo-id WRM_grasp_cylinder_different_sizes_lero \
        --device cuda:0 \
        --num-workers 8 \
        --output-dir "${output_dir}" 2>&1 | tee "${log_path}"
fi

"${python_bin}" - "${HF_REPO_ID}" "${output_dir}" "${POLICY_NAME}" <<'PY'
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
HfApi().upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path=str(folder),
    path_in_repo="checkpoints",
    ignore_patterns=["*.tmp", "wandb/**"],
    commit_message=f"Upload {policy_name} checkpoints through step 100000",
)
print("huggingface_upload_complete", repo_id)
PY
