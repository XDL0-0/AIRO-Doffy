#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/policies/output}"
DEVICE="${DEVICE:-cuda:0}"
NUM_WORKERS="${NUM_WORKERS:-1}"
CYLINDER_DATASET_ROOT="${CYLINDER_DATASET_ROOT:-${REPO_ROOT}/datasets/WRM_grasp_cylinder_lero}"
MERGED_DATASET_ROOT="${MERGED_DATASET_ROOT:-${REPO_ROOT}/datasets/WRM_grasp_cylinder_all}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
MODE="sequential"
WAIT_FOR_LOCK="false"

usage() {
    echo "Usage: $0 [--parallel|--sequential] [--wait] [arguments passed to every trainer]"
    echo
    echo "Environment overrides: DEVICE=cuda:0 NUM_WORKERS=1 OUTPUT_ROOT=path"
    echo "  CYLINDER_DATASET_ROOT=path MERGED_DATASET_ROOT=path"
    echo "  TRAIN_DATASET_ROOT=path TRAIN_DATASET_NAME=name (run one dataset only)"
    echo "  MAX_PARALLEL=2 WANDB_PROJECT=project-name"
    echo "Default: train nine policies on each of the cylinder and merged datasets sequentially."
    echo "--parallel runs at most MAX_PARALLEL trainers concurrently."
}

EXTRA_ARGS=()
while (($#)); do
    case "$1" in
        --parallel)
            MODE="parallel"
            ;;
        --sequential)
            MODE="sequential"
            ;;
        --wait)
            WAIT_FOR_LOCK="true"
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            ;;
    esac
    shift
done
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${OUTPUT_ROOT}"

if [[ -n "${TRAIN_DATASET_ROOT:-}" ]]; then
    TRAIN_DATASET_NAME="${TRAIN_DATASET_NAME:-$(basename -- "${TRAIN_DATASET_ROOT}")}"
    DATASET_NAMES=("${TRAIN_DATASET_NAME}")
    DATASET_ROOTS=("${TRAIN_DATASET_ROOT}")
else
    DATASET_NAMES=("WRM_grasp_cylinder_lero" "WRM_grasp_cylinder_all")
    DATASET_ROOTS=("${CYLINDER_DATASET_ROOT}" "${MERGED_DATASET_ROOT}")
fi
for dataset_root in "${DATASET_ROOTS[@]}"; do
    if [[ ! -f "${dataset_root}/meta/info.json" ]]; then
        echo "LeRobot dataset is missing meta/info.json: ${dataset_root}" >&2
        exit 2
    fi
done

if [[ ! "${MAX_PARALLEL}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_PARALLEL must be a positive integer, got: ${MAX_PARALLEL}" >&2
    exit 2
fi

# Do not allow two launchers to write the same checkpoints concurrently.
exec 9>"${OUTPUT_ROOT}/.training.lock"
if [[ "${WAIT_FOR_LOCK}" == "true" ]]; then
    echo "[$(date -Is)] waiting for ${OUTPUT_ROOT}/.training.lock"
    flock 9
    echo "[$(date -Is)] acquired ${OUTPUT_ROOT}/.training.lock"
elif ! flock -n 9; then
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
    local dataset_name="$1"
    local dataset_root="$2"
    local name="$3"
    local config="$4"
    local output_dir="${OUTPUT_ROOT}/${dataset_name}/${name}"
    local log_path="${output_dir}/train_${RUN_ID}.log"
    local -a wandb_args=()
    if [[ -n "${WANDB_PROJECT}" ]]; then
        wandb_args=(
            --wandb-project "${WANDB_PROJECT}"
            --wandb-run-name "${dataset_name}/${name}"
        )
    fi
    mkdir -p "${output_dir}"
    echo "[$(date -Is)] starting ${dataset_name}/${name}; log=${log_path}"
    (
        cd "${REPO_ROOT}"
        set -o pipefail
        PYTHONUNBUFFERED=1 python -m policies.realman_beaver.train \
            --config "${config}" \
            "${EXTRA_ARGS[@]}" \
            "${wandb_args[@]}" \
            --dataset-root "${dataset_root}" \
            --dataset-repo-id "${dataset_name}" \
            --device "${DEVICE}" \
            --num-workers "${NUM_WORKERS}" \
            --output-dir "${output_dir}" 2>&1 | tee "${log_path}"
    )
}

NAMES=(
    "original_dp"
    "dp_beaver"
    "dp_beaver_enc"
    "dp_beaver_near"
    "dp_beaver_near_gate"
    "dp_beaver_key4"
    "dp_beaver_key4_pca"
    "rdp_like"
    "fm"
    "fm_beaver"
    "rfm"
)
CONFIGS=(
    "policies/realman_beaver/configs/original_dp.yaml"
    "policies/realman_beaver/configs/dp_beaver.yaml"
    "policies/realman_beaver/configs/dp_beaver_enc.yaml"
    "policies/realman_beaver/configs/dp_beaver_near.yaml"
    "policies/realman_beaver/configs/dp_beaver_near_gate.yaml"
    "policies/realman_beaver/configs/dp_beaver_key4.yaml"
    "policies/realman_beaver/configs/dp_beaver_key4_pca.yaml"
    "policies/realman_beaver/configs/rdp_like.yaml"
    "policies/realman_beaver/configs/fm.yaml"
    "policies/realman_beaver/configs/fm_beaver.yaml"
    "policies/realman_beaver/configs/rfm.yaml"
)

declare -A PID_TO_NAME=()

wait_for_one() {
    local finished_pid=""
    local failed_name="unknown"
    local status
    local pid
    local -a next_pids=()
    if wait -n -p finished_pid; then
        echo "[$(date -Is)] completed ${PID_TO_NAME[${finished_pid}]}"
    else
        status=$?
        if [[ -n "${finished_pid}" ]]; then
            failed_name="${PID_TO_NAME[${finished_pid}]:-unknown}"
        fi
        echo "[$(date -Is)] FAILED ${failed_name} with exit code ${status}" >&2
        cleanup
        exit "${status}"
    fi
    unset "PID_TO_NAME[${finished_pid}]"
    for pid in "${ACTIVE_PIDS[@]}"; do
        if [[ "${pid}" != "${finished_pid}" ]]; then
            next_pids+=("${pid}")
        fi
    done
    ACTIVE_PIDS=("${next_pids[@]}")
}

parallel_limit=1
if [[ "${MODE}" == "parallel" ]]; then
    parallel_limit="${MAX_PARALLEL}"
fi

for dataset_index in "${!DATASET_NAMES[@]}"; do
    for policy_index in "${!NAMES[@]}"; do
        while ((${#ACTIVE_PIDS[@]} >= parallel_limit)); do
            wait_for_one
        done
        job_name="${DATASET_NAMES[dataset_index]}/${NAMES[policy_index]}"
        run_policy \
            "${DATASET_NAMES[dataset_index]}" \
            "${DATASET_ROOTS[dataset_index]}" \
            "${NAMES[policy_index]}" \
            "${CONFIGS[policy_index]}" &
        pid=$!
        ACTIVE_PIDS+=("${pid}")
        PID_TO_NAME["${pid}"]="${job_name}"
    done
done

while ((${#ACTIVE_PIDS[@]} > 0)); do
    wait_for_one
done

trap - INT TERM
echo "[$(date -Is)] all policy training completed; outputs=${OUTPUT_ROOT}"
