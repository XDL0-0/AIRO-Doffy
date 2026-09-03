#!/usr/bin/env bash

set -Eeuo pipefail

stage_dir="/tmp/airo_wrm_antigravity_stratified_cluster12_stage"
source_archive="${stage_dir}/airo-doffy-wrm-antigravity-stratified-src.tar.gz"
source_manifest="${stage_dir}/airo-doffy-wrm-antigravity-stratified-src.sha256"
dataset_archive="${stage_dir}/WRM_grasp_cylinder_different_sizes_lero_tightness-all125.tar"
dataset_manifest="${stage_dir}/WRM_grasp_cylinder_different_sizes_lero_tightness-all125.sha256"
wandb_netrc_file="${stage_dir}/netrc"
python_bin="/opt/conda/envs/lerobot/bin/python"
work_root="/tmp/airo_wrm_antigravity_stratified_cluster12_work"
repo_root="${work_root}/AIRO-Doffy"
dataset_root="${work_root}/datasets/WRM_grasp_cylinder_different_sizes_lero_tightness"
output_root="/project_ghent/AIRO-Doffy/WRM_grasp_cylinder_different_sizes_lero_tightness/wrm_antigravity_stratified_cluster12_bs32"
policy_output="${output_root}/WRM_antigravity_stratified"
policy_log="${policy_output}/cluster12_stratified_bs32_train.log"

for required_file in \
    "${source_archive}" \
    "${source_manifest}" \
    "${dataset_archive}" \
    "${dataset_manifest}" \
    "${wandb_netrc_file}"; do
    [[ -s "${required_file}" ]] || {
        echo "Missing staged input: ${required_file}" >&2
        exit 2
    }
done
chmod 600 "${wandb_netrc_file}"
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
export HF_HOME="${work_root}/huggingface"
export WANDB_CACHE_DIR="${work_root}/wandb-cache"
export WANDB_DIR="${policy_output}/wandb"
export PYTHONUNBUFFERED=1
trap 'unset WANDB_API_KEY' EXIT

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

"${python_bin}" -m unittest \
    policies.realman_beaver.tests.test_wrm_antigravity -v

if [[ -f "${policy_output}/last.pt" ]]; then
    echo "[$(date -Is)] existing completed stratified WRM_antigravity policy found; skipping"
else
    echo "[$(date -Is)] starting stratified WRM_antigravity on cluster 12"
    set -o pipefail
    "${python_bin}" -m policies.realman_beaver.train \
        --config policies/realman_beaver/configs/WRM_antigravity.yaml \
        --wandb-project AIRO-Doffy-WRM-Grasp \
        --wandb-run-name "WRM_grasp_cylinder_different_sizes_lero_tightness/WRM_antigravity-stratified-cluster12-bs32" \
        --dataset-root "${dataset_root}" \
        --dataset-repo-id WRM_grasp_cylinder_different_sizes_lero_tightness \
        --device cuda:0 \
        --batch-size 32 \
        --num-workers 8 \
        --max-steps 100000 \
        --output-dir "${policy_output}" 2>&1 | tee "${policy_log}"
fi

echo "[$(date -Is)] completed stratified WRM_antigravity training"
