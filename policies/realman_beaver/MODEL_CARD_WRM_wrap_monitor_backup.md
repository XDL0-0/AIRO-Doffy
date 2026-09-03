---
library_name: pytorch
tags:
  - robotics
  - tactile-sensing
  - diffusion-policy
  - beaver
---

# WRM_wrap_monitor_backup

Strict backup Beaver monitor for the frozen WRM_wrap 50k Diffusion Policy. It
is trained as a small MLP and exhaustively checked over all 512 binary patterns
of the nine sensors.

Only Key4 sensors `01, 02, 10, 11` enter the MLP; the other five are masked
before the network and are therefore structurally unable to affect its output.
It exactly distills:

- near: 0 mm
- closing scale: 50 mm
- lift minimum wrap: 0.25 (at least one Key4 exact-zero contact)
- stop-close wrap: 0.5 (at least two Key4 exact-zero contacts)
- contact stop: 0 mm

Outputs are `[lift_state, contact_state]` with the fixed decision boundary
`logit >= 0`. See `metrics.json` for the exhaustive truth-table result and
minimum logit margin.

`monitor.pt` contains the trained monitor only. `checkpoints/last.pt` is the
deployable combined policy containing frozen EMA WRM_wrap weights plus this
monitor.
