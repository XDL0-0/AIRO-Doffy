#!/usr/bin/env bash
# Coordinator launch for WRM_grok architecture-selection (100 train / 25 val).
# Do not start this from the agent worktree; it is GPUlab-ready documentation.

set -Eeuo pipefail

python -m policies.realman_beaver.train \
  --config policies/realman_beaver/configs/WRM_grok.yaml \
  --dataset-root "${DATASET_ROOT:-/home/yuyuan/AIRO-Doffy/datasets/WRM_grasp_cylinder_different_sizes_lero_tightness}" \
  --dataset-repo-id WRM_grasp_cylinder_different_sizes_lero_tightness \
  --device "${DEVICE:-cuda:0}" \
  --batch-size 32 \
  --num-workers 8 \
  --max-steps 100000 \
  --output-dir "${OUTPUT_DIR:-outputs/realman_beaver/WRM_grok}" \
  --wandb-project AIRO-Doffy-WRM-Grasp \
  --wandb-run-name WRM_grasp_cylinder_different_sizes_lero_tightness/WRM_grok-arch-select
