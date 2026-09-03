#!/usr/bin/env bash

set -Eeuo pipefail

if [[ "$#" -ne 1 ]]; then
    echo "Usage: $0 JOB_ID" >&2
    exit 2
fi

job_id="$1"
short_id="${job_id:0:8}"
ssh_alias="gpulab-${short_id}-proxy"
gpulab_cli="/home/yuyuan/.local/share/gpulab-cli-venv/bin/gpulab-cli"
gpulab_cert="/home/yuyuan/Downloads/login_ilabt_imec_be_yuyuwu@ugent.be_decrypted.pem"
repo_root="/home/yuyuan/AIRO-Doffy"
dataset_root="${repo_root}/datasets/WRM_grasp_cylinder_different_sizes_lero_tightness"
source_archive="/tmp/airo-doffy-wrm-adaptive-all125-src.tar.gz"
source_manifest="/tmp/airo-doffy-wrm-adaptive-all125-src.sha256"
dataset_archive="/tmp/WRM_grasp_cylinder_different_sizes_lero_tightness-all125.tar"
dataset_manifest="/tmp/WRM_grasp_cylinder_different_sizes_lero_tightness-all125.sha256"
runner_script="${repo_root}/.gpulab/run-wrm-adaptive-all125-cluster12-bs32.sh"
wandb_netrc_file="/home/yuyuan/.netrc"
remote_stage="/tmp/airo_wrm_adaptive_all125_cluster12_stage"

for required_file in \
    "${gpulab_cli}" \
    "${gpulab_cert}" \
    "${runner_script}" \
    "${wandb_netrc_file}"; do
    [[ -s "${required_file}" ]] || {
        echo "Missing local input: ${required_file}" >&2
        exit 2
    }
done
[[ -d "${dataset_root}" ]] || {
    echo "Missing dataset: ${dataset_root}" >&2
    exit 2
}

tar -C "${repo_root}" \
    --exclude='*/__pycache__' \
    --exclude='*.pyc' \
    -czf "${source_archive}" \
    policies/realman_beaver eval_config.py eval_policy.py
tar -C "${repo_root}/datasets" -cf "${dataset_archive}" \
    WRM_grasp_cylinder_different_sizes_lero_tightness
printf '%s  %s\n' "$(sha256sum "${source_archive}" | awk '{print $1}')" \
    "$(basename "${source_archive}")" >"${source_manifest}"
printf '%s  %s\n' "$(sha256sum "${dataset_archive}" | awk '{print $1}')" \
    "$(basename "${dataset_archive}")" >"${dataset_manifest}"

job_status() {
    "${gpulab_cli}" --production --cert "${gpulab_cert}" jobs "$1" \
        | sed -n 's/^ *Status: //p' | head -1
}

while true; do
    status="$(job_status "${job_id}" || true)"
    echo "[$(date -Is)] cluster12 all125 ${short_id} status=${status:-unknown}"
    case "${status}" in
        RUNNING) break ;;
        CANCELLED|DELETED|FAILED|FINISHED|HALTED) exit 1 ;;
    esac
    sleep 20
done

"${gpulab_cli}" --production --cert "${gpulab_cert}" \
    ssh --proxy --show "${job_id}" >/dev/null
for ssh_attempt in $(seq 1 20); do
    if ssh \
        -o BatchMode=yes \
        -o StrictHostKeyChecking=accept-new \
        -o ConnectTimeout=20 \
        "${ssh_alias}" "mkdir -p ${remote_stage} && chmod 755 ${remote_stage}"; then
        break
    fi
    [[ "${ssh_attempt}" != "20" ]] || exit 1
    sleep 5
done

scp -o BatchMode=yes \
    "${source_archive}" \
    "${source_manifest}" \
    "${dataset_archive}" \
    "${dataset_manifest}" \
    "${runner_script}" \
    "${ssh_alias}:${remote_stage}/"
scp -o BatchMode=yes "${wandb_netrc_file}" "${ssh_alias}:${remote_stage}/netrc"
ssh -o BatchMode=yes "${ssh_alias}" \
    "chown -R 1000:100 ${remote_stage}; \
    chmod 755 ${remote_stage} ${remote_stage}/run-wrm-adaptive-all125-cluster12-bs32.sh; \
    chmod 600 ${remote_stage}/netrc; \
    cd ${remote_stage}; \
    sha256sum --check --strict $(basename "${source_manifest}"); \
    sha256sum --check --strict $(basename "${dataset_manifest}"); \
    touch READY; chown 1000:100 READY"

echo "Staged and released all-125 WRM_adaptive cluster-12 job ${job_id}"
