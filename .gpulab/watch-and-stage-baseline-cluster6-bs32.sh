#!/usr/bin/env bash

set -Eeuo pipefail

if [[ "$#" -eq 0 ]]; then
    echo "Usage: $0 JOB_ID [JOB_ID ...]" >&2
    exit 2
fi

gpulab_cli="/home/yuyuan/.local/share/gpulab-cli-venv/bin/gpulab-cli"
gpulab_cert="/home/yuyuan/Downloads/login_ilabt_imec_be_yuyuwu@ugent.be_decrypted.pem"
source_archive="/tmp/airo-doffy-structured-beaver-src-bs32.tar.gz"
runner_script="/home/yuyuan/AIRO-Doffy/.gpulab/run-baseline-policy-single-cluster6-bs32.sh"
hf_token_file="/home/yuyuan/.cache/huggingface/token"
wandb_netrc_file="/home/yuyuan/.netrc"
source_sha256="48647fe329964dae872c05d22950c11c9fc0ab6c9371dc86c395650ca9e6ab96"

job_ids=("$@")
declare -A staged=()

job_status() {
    "${gpulab_cli}" --production --cert "${gpulab_cert}" jobs "$1" \
        | sed -n 's/^ *Status: //p' \
        | head -1
}

stage_job() {
    local job_id="$1"
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
            'mkdir -p /tmp/airo_baseline_cluster6_stage && chmod 755 /tmp/airo_baseline_cluster6_stage'; then
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
        "${ssh_alias}:/tmp/airo_baseline_cluster6_stage/"
    scp -o BatchMode=yes \
        "${hf_token_file}" \
        "${ssh_alias}:/tmp/airo_baseline_cluster6_stage/hf_token"
    scp -o BatchMode=yes \
        "${wandb_netrc_file}" \
        "${ssh_alias}:/tmp/airo_baseline_cluster6_stage/netrc"

    ssh -o BatchMode=yes "${ssh_alias}" \
        "chown -R 1000:100 /tmp/airo_baseline_cluster6_stage; \
        chmod 755 /tmp/airo_baseline_cluster6_stage \
            /tmp/airo_baseline_cluster6_stage/run-baseline-policy-single-cluster6-bs32.sh; \
        chmod 600 /tmp/airo_baseline_cluster6_stage/hf_token \
            /tmp/airo_baseline_cluster6_stage/netrc; \
        echo '${source_sha256}  /tmp/airo_baseline_cluster6_stage/airo-doffy-structured-beaver-src-bs32.tar.gz' \
            | sha256sum --check --strict; \
        touch /tmp/airo_baseline_cluster6_stage/READY; \
        chown 1000:100 /tmp/airo_baseline_cluster6_stage/READY"
    echo "Staged and released Cluster6 job ${job_id}"
}

while ((${#staged[@]} < ${#job_ids[@]})); do
    for job_id in "${job_ids[@]}"; do
        if [[ -n "${staged[${job_id}]:-}" ]]; then
            continue
        fi
        status="$(job_status "${job_id}" || true)"
        echo "[$(date -Is)] Cluster6 ${job_id:0:8} status=${status:-unknown}"
        case "${status}" in
            RUNNING)
                stage_job "${job_id}"
                staged["${job_id}"]=1
                ;;
            CANCELLED|DELETED|FAILED|FINISHED|HALTED)
                echo "Job ${job_id} reached terminal state ${status} before staging" >&2
                exit 1
                ;;
        esac
    done
    if ((${#staged[@]} < ${#job_ids[@]})); then
        sleep 20
    fi
done

echo "All four Cluster6 baseline jobs were staged successfully"
