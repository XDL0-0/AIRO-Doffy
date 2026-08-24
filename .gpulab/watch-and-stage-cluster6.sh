#!/usr/bin/env bash

set -Eeuo pipefail

gpulab_cli="/home/yuyuan/.local/share/gpulab-cli-venv/bin/gpulab-cli"
gpulab_cert="/home/yuyuan/Downloads/login_ilabt_imec_be_yuyuwu@ugent.be_decrypted.pem"
source_archive="/tmp/airo-doffy-structured-beaver-src.tar.gz"
runner_script="/home/yuyuan/AIRO-Doffy/.gpulab/run-structured-beaver-single.sh"
hf_token_file="/home/yuyuan/.cache/huggingface/token"
wandb_netrc_file="/home/yuyuan/.netrc"
cluster5_serial_job="8cbf94da-b19b-44a8-8532-30713a1536eb"
source_sha256="ee823cacf2bde568a85fcdc31fae5c1d29908b3cd65655ee50449766f9e194b1"

declare -A job_roles=(
    ["528c420c-bbb5-4e80-9a43-66d9c298c59d"]="enc"
    ["d46d5f74-be7e-4faf-9f13-7389a3467c98"]="gate"
)
declare -A staged=()

job_status() {
    "${gpulab_cli}" --production --cert "${gpulab_cert}" jobs "$1" \
        | sed -n 's/^ *Status: //p' \
        | head -1
}

stage_job() {
    local job_id="$1"
    local role="$2"
    local short_id="${job_id:0:8}"
    local ssh_alias="gpulab-${short_id}-proxy"

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
            return 1
        fi
        sleep 5
    done

    scp -o BatchMode=yes \
        "${source_archive}" \
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
            /tmp/airo_structured_beaver_stage/run-structured-beaver-single.sh; \
        chmod 600 /tmp/airo_structured_beaver_stage/hf_token \
            /tmp/airo_structured_beaver_stage/netrc; \
        echo '${source_sha256}  /tmp/airo_structured_beaver_stage/airo-doffy-structured-beaver-src.tar.gz' \
            | sha256sum --check --strict"

    if [[ "${role}" == "enc" ]]; then
        cluster5_status="$(job_status "${cluster5_serial_job}" || true)"
        if [[ "${cluster5_status}" == "RUNNING" ]]; then
            echo "Cancelling cluster 5 serial job before releasing cluster 6 enc"
            "${gpulab_cli}" --production --cert "${gpulab_cert}" \
                cancel "${cluster5_serial_job}"
        fi
    fi

    ssh -o BatchMode=yes "${ssh_alias}" \
        'touch /tmp/airo_structured_beaver_stage/READY && chown 1000:100 /tmp/airo_structured_beaver_stage/READY'
    echo "Staged and released ${role} job ${job_id}"
}

while ((${#staged[@]} < ${#job_roles[@]})); do
    for job_id in "${!job_roles[@]}"; do
        if [[ -n "${staged[${job_id}]:-}" ]]; then
            continue
        fi
        status="$(job_status "${job_id}" || true)"
        echo "[$(date -Is)] ${job_roles[${job_id}]} ${job_id:0:8} status=${status:-unknown}"
        case "${status}" in
            RUNNING)
                stage_job "${job_id}" "${job_roles[${job_id}]}"
                staged["${job_id}"]=1
                ;;
            CANCELLED|DELETED|FAILED|FINISHED|HALTED)
                echo "Job ${job_id} reached terminal state ${status} before staging" >&2
                exit 1
                ;;
        esac
    done
    if ((${#staged[@]} < ${#job_roles[@]})); then
        sleep 30
    fi
done

echo "All queued cluster 6 jobs were staged successfully"
