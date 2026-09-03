#!/usr/bin/env bash

set -Eeuo pipefail

run_kind="${1:-}"
if [[ "${run_kind}" != "selection" && "${run_kind}" != "all125" ]]; then
    echo "usage: $0 selection|all125" >&2
    exit 2
fi

stage_dir="/tmp/airo_wrm_codex_stage"
source_archive="${stage_dir}/airo-doffy-wrm-codex-src.tar.gz"
source_manifest="${stage_dir}/airo-doffy-wrm-codex-src.sha256"
dataset_archive="${stage_dir}/WRM_grasp_cylinder_different_sizes_lero_tightness.tar"
dataset_manifest="${stage_dir}/WRM_grasp_cylinder_different_sizes_lero_tightness.sha256"
work_root="/tmp/airo_wrm_codex_work"
repo_root="${work_root}/AIRO-Doffy"
dataset_root="${work_root}/datasets/WRM_grasp_cylinder_different_sizes_lero_tightness"
output_root="/project_ghent/AIRO-Doffy/WRM_grasp_cylinder_different_sizes_lero_tightness/wrm_codex"
policy_output="${output_root}/${run_kind}"
python_bin="/opt/conda/envs/lerobot/bin/python"
if [[ ! -x "${python_bin}" ]]; then
    python_bin="$(command -v python)"
fi

for required_file in \
    "${source_archive}" \
    "${source_manifest}" \
    "${dataset_archive}" \
    "${dataset_manifest}"; do
    [[ -s "${required_file}" ]] || {
        echo "Missing staged input: ${required_file}" >&2
        exit 2
    }
done
(
    cd "${stage_dir}"
    sha256sum --check --strict "$(basename "${source_manifest}")"
    sha256sum --check --strict "$(basename "${dataset_manifest}")"
)

mkdir -p "${repo_root}" "${work_root}/datasets" "${policy_output}"
tar -xzf "${source_archive}" -C "${repo_root}"
tar -xf "${dataset_archive}" -C "${work_root}/datasets"
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${policy_output}/wandb"

cd "${repo_root}"
"${python_bin}" - "${dataset_root}" <<'PY'
import json
import os
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
info = json.loads((root / "meta" / "info.json").read_text())
if info.get("total_episodes") != 125 or info.get("total_frames") != 40596:
    raise SystemExit(f"Unexpected dataset metadata: {info}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit(f"Expected one visible CUDA GPU, got {torch.cuda.device_count()}")
print(
    "resources",
    "gpu", torch.cuda.get_device_name(0),
    "visible_cpus", len(os.sched_getaffinity(0)),
    "batch_size", 32,
)
PY

"${python_bin}" -m unittest \
    policies.realman_beaver.tests.test_wrm_codex -v

if [[ "${run_kind}" == "selection" ]]; then
    config_path="policies/realman_beaver/configs/WRM_codex_selection.yaml"
    max_steps=100000
    milestone_count=10
else
    config_path="policies/realman_beaver/configs/WRM_codex_all125.yaml"
    max_steps=100000
    milestone_count=10
fi

"${python_bin}" -m policies.realman_beaver.train \
    --config "${config_path}" \
    --dataset-root "${dataset_root}" \
    --dataset-repo-id WRM_grasp_cylinder_different_sizes_lero_tightness \
    --device cuda:0 \
    --batch-size 32 \
    --num-workers 8 \
    --max-steps "${max_steps}" \
    --output-dir "${policy_output}" 2>&1 | tee "${policy_output}/train.log"

"${python_bin}" - "${policy_output}" "${max_steps}" "${milestone_count}" <<'PY'
import sys
from pathlib import Path

import torch

folder = Path(sys.argv[1])
expected_step = int(sys.argv[2])
expected_milestones = int(sys.argv[3])
milestones = sorted(folder.glob("WRM_codex_step_*.pt"))
last = folder / "last.pt"
if len(milestones) != expected_milestones or not last.is_file():
    raise SystemExit(f"Incomplete run: milestones={len(milestones)}, last={last.is_file()}")
checkpoint = torch.load(last, map_location="cpu", weights_only=True, mmap=True)
if checkpoint.get("kind") != "WRM_codex":
    raise SystemExit(f"Unexpected checkpoint kind: {checkpoint.get('kind')}")
if checkpoint.get("global_step") != expected_step:
    raise SystemExit(f"Expected step {expected_step}, got {checkpoint.get('global_step')}")
print("training_complete", last, "step", checkpoint["global_step"])
PY
