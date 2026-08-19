#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/policies/output}"
DEVICE="${DEVICE:-cuda:0}"
NUM_WORKERS="${NUM_WORKERS:-1}"
MODE="parallel"

usage() {
    echo "Usage: $0 [--parallel|--sequential] [arguments passed to every trainer]"
    echo
    echo "Environment overrides: DEVICE=cuda:0 NUM_WORKERS=1 OUTPUT_ROOT=path"
    echo "Default: launch all three policies concurrently."
}

case "${1:-}" in
    --parallel)
        MODE="parallel"
        shift
        ;;
    --sequential)
        MODE="sequential"
        shift
        ;;
    --help|-h)
        usage
        exit 0
        ;;
esac

EXTRA_ARGS=("$@")
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${OUTPUT_ROOT}"

# Do not allow two launchers to write the same checkpoints concurrently.
exec 9>"${OUTPUT_ROOT}/.training.lock"
if ! flock -n 9; then
    echo "Another train_all.sh process already owns ${OUTPUT_ROOT}/.training.lock" >&2
    exit 2
fi

declare -a ACTIVE_PIDS=()

cleanup() {
    local pid
    if ((${#ACTIVE_PIDS[@]})); then
        echo "Stopping remaining training jobs..." >&2
        for pid in "${ACTIVE_PIDS[@]}"; do
            pkill -TERM -P "${pid}" 2>/dev/null || true
            kill "${pid}" 2>/dev/null || true
        done
        wait "${ACTIVE_PIDS[@]}" 2>/dev/null || true
    fi
}

on_signal() {
    trap - INT TERM
    cleanup
    exit 130
}
trap on_signal INT TERM

run_policy() {
    local name="$1"
    local config="$2"
    local output_dir="${OUTPUT_ROOT}/${name}"
    local log_path="${output_dir}/train_${RUN_ID}.log"
    mkdir -p "${output_dir}"
    echo "[$(date -Is)] starting ${name}; log=${log_path}"
    (
        cd "${REPO_ROOT}"
        set -o pipefail
        PYTHONUNBUFFERED=1 python -m policies.realman_beaver.train \
            --config "${config}" \
            "${EXTRA_ARGS[@]}" \
            --device "${DEVICE}" \
            --num-workers "${NUM_WORKERS}" \
            --output-dir "${output_dir}" 2>&1 | tee "${log_path}"
    )
}

NAMES=("original_dp" "dp_beaver" "rdp_like")
CONFIGS=(
    "policies/realman_beaver/configs/original_dp.yaml"
    "policies/realman_beaver/configs/dp_beaver.yaml"
    "policies/realman_beaver/configs/rdp_like.yaml"
)

if [[ "${MODE}" == "sequential" ]]; then
    for index in "${!NAMES[@]}"; do
        run_policy "${NAMES[index]}" "${CONFIGS[index]}"
    done
else
    declare -A PID_TO_NAME=()
    for index in "${!NAMES[@]}"; do
        run_policy "${NAMES[index]}" "${CONFIGS[index]}" &
        pid=$!
        ACTIVE_PIDS+=("${pid}")
        PID_TO_NAME["${pid}"]="${NAMES[index]}"
    done

    remaining=${#ACTIVE_PIDS[@]}
    while ((remaining > 0)); do
        finished_pid=""
        if wait -n -p finished_pid; then
            echo "[$(date -Is)] completed ${PID_TO_NAME[${finished_pid}]}"
        else
            status=$?
            finished_pid="${finished_pid:-}"
            failed_name="unknown"
            if [[ -n "${finished_pid}" ]]; then
                failed_name="${PID_TO_NAME[${finished_pid}]:-unknown}"
            fi
            echo "[$(date -Is)] FAILED ${failed_name} with exit code ${status}" >&2
            cleanup
            exit "${status}"
        fi
        unset "PID_TO_NAME[${finished_pid}]"
        next_pids=()
        for pid in "${ACTIVE_PIDS[@]}"; do
            if [[ "${pid}" != "${finished_pid}" ]]; then
                next_pids+=("${pid}")
            fi
        done
        ACTIVE_PIDS=("${next_pids[@]}")
        remaining=$((remaining - 1))
    done
fi

trap - INT TERM
echo "[$(date -Is)] all policy training completed; outputs=${OUTPUT_ROOT}"
