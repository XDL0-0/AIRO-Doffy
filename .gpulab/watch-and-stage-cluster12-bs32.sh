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
source_archive="/tmp/airo-doffy-structured-beaver-src-bs32.tar.gz"
dataset_archive="/tmp/WRM_grasp_cylinder_different_sizes_lero.tar"
runner_script="/home/yuyuan/AIRO-Doffy/.gpulab/run-structured-beaver-sequential-cluster12-bs32.sh"
hf_token_file="/home/yuyuan/.cache/huggingface/token"
wandb_netrc_file="/home/yuyuan/.netrc"
source_sha256="48647fe329964dae872c05d22950c11c9fc0ab6c9371dc86c395650ca9e6ab96"
dataset_sha256="bf890b70eb5db96db654a9ed7be830c00c12ef1611866ee8d871473c9dd4b8d2"

job_status() {
    "${gpulab_cli}" --production --cert "${gpulab_cert}" jobs "$1" \
        | sed -n 's/^ *Status: //p' \
        | head -1
}

while true; do
    status="$(job_status "${job_id}" || true)"
    echo "[$(date -Is)] cluster12 ${short_id} status=${status:-unknown}"
    case "${status}" in
        RUNNING)
            break
            ;;
        CANCELLED|DELETED|FAILED|FINISHED|HALTED)
            echo "Job ${job_id} reached terminal state ${status} before staging" >&2
            exit 1
            ;;
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
        "${ssh_alias}" \
        'mkdir -p /tmp/airo_structured_beaver_stage && chmod 755 /tmp/airo_structured_beaver_stage'; then
        break
    fi
    if [[ "${ssh_attempt}" == "20" ]]; then
        echo "Unable to reach running GPUlab job ${job_id}" >&2
        exit 1
    fi
    sleep 5
done

scp -o BatchMode=yes \
    "${source_archive}" \
    "${dataset_archive}" \
    "${runner_script}" \
    "${ssh_alias}:/tmp/airo_structured_beaver_stage/"
scp -o BatchMode=yes \
    "${hf_token_file}" \
    "${ssh_alias}:/tmp/airo_structured_beaver_stage/hf_token"
scp -o BatchMode=yes \
    "${wandb_netrc_file}" \
    "${ssh_alias}:/tmp/airo_structured_beaver_stage/netrc"

ssh -o BatchMode=yes "${ssh_alias}" \
    "chown -R 1000:100 /tmp/airo_structured_beaver_stage; \
    chmod 755 /tmp/airo_structured_beaver_stage \
        /tmp/airo_structured_beaver_stage/run-structured-beaver-sequential-cluster12-bs32.sh; \
    chmod 600 /tmp/airo_structured_beaver_stage/hf_token \
        /tmp/airo_structured_beaver_stage/netrc; \
    echo '${source_sha256}  /tmp/airo_structured_beaver_stage/airo-doffy-structured-beaver-src-bs32.tar.gz' \
        | sha256sum --check --strict; \
    echo '${dataset_sha256}  /tmp/airo_structured_beaver_stage/WRM_grasp_cylinder_different_sizes_lero.tar' \
        | sha256sum --check --strict; \
    touch /tmp/airo_structured_beaver_stage/READY; \
    chown 1000:100 /tmp/airo_structured_beaver_stage/READY"

echo "Staged inputs and released cluster12 batch-size-32 job ${job_id}"
