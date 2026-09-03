#!/usr/bin/env bash

set -Eeuo pipefail

repo_root="/home/yuyuan/AIRO-Doffy"
tmux_bin="/home/yuyuan/miniconda3/bin/tmux"

if ! nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA GPU is not available in this shell; run this launcher from the host terminal." >&2
    exit 1
fi

start_policy() {
    local session_name="$1"
    local policy_name="$2"
    local command="$3"
    local checkpoint="${repo_root}/policies/output/WRM_grasp_cylinder_different_sizes_lero/structured_beaver_local_parallel/${policy_name}/${policy_name}_step_025000.pt"

    if [[ ! -f "${checkpoint}" ]]; then
        echo "Missing checkpoint: ${checkpoint}" >&2
        exit 1
    fi
    if "${tmux_bin}" has-session -t "=${session_name}" 2>/dev/null; then
        echo "Already running in tmux session ${session_name}"
        return
    fi
    if pgrep -f -- "python -m policies.realman_beaver.train.*${policy_name}\.yaml" >/dev/null; then
        echo "Refusing duplicate trainer for ${policy_name}: an existing process was found" >&2
        exit 1
    fi

    "${tmux_bin}" new-session -d -s "${session_name}" -c "${repo_root}" "${command}"
    echo "Started ${policy_name} in tmux session ${session_name}"
}

start_policy \
    airo-dp-beaver-enc \
    dp_beaver_enc \
    "exec env POLICY_NAME=dp_beaver_enc HF_REPO_ID=IXDLI/AIRO-Doffy-WRM-Grasp-dp-beaver-enc .gpulab/run-structured-beaver-local-single.sh"

start_policy \
    airo-dp-beaver-near \
    dp_beaver_near \
    "exec env POLICY_NAME=dp_beaver_near HF_REPO_ID=IXDLI/AIRO-Doffy-WRM-Grasp-dp-beaver-near .gpulab/run-structured-beaver-local-single.sh"

start_policy \
    airo-dp-beaver-near-gate \
    dp_beaver_near_gate \
    "exec .gpulab/run-dp-beaver-near-gate-local.sh"

echo
"${tmux_bin}" list-sessions -F '#{session_name}: #{session_created_string}' \
    | grep '^airo-dp-beaver-' || true
