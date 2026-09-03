---
library_name: pytorch
tags:
  - robotics
  - tactile-sensing
  - diffusion-policy
  - beaver
---

# WRM_wrap_monitor

Beaver-only temporal execution monitor for the frozen WRM_wrap 50k Diffusion
Policy. The monitor is a small MLP over bounded features from all nine 4x4
Beaver sensors at lags `[0, 1, 3, 6, 11]`.

Outputs are ordered `[lift_state, contact_state]`. A fixed logit boundary of
zero releases J1 for lift and freezes J3/J4/J5 after grasp. There are no
deployment gate thresholds. Labels come from the original 125-episode dataset:
`contact_state=tightness`; lift onset is the first six-frame run where J1 is at
least 0.02 rad above its initial baseline in the demonstrated lift direction.

The split is bottle-stratified by whole episode: 75 train, 25 validation, and
25 held-out test. See `metrics.json` and `dataset_manifest.json` for exact
event definitions, episode IDs, and results.

`monitor.pt` contains the trained monitor only. `checkpoints/last.pt` is the
deployable combined policy containing frozen EMA WRM_wrap weights plus this
monitor.
