---
library_name: pytorch
tags:
  - robotics
  - tactile-sensing
  - diffusion-policy
  - beaver
  - zero-temporal
  - leave-one-bottle-out
---

# WRM_lobo_monitor

Zero-temporal Beaver+Joints execution monitor for the frozen WRM_wrap 50k Diffusion Policy.

## Architecture
- **Input Features (31-dim)**: Single-frame Beaver Key4 sensors `(1, 2, 5, 6)` proximity + validity + zero-contact + min-distance (24-dim), concatenated with the current normalized 7-DoF robot joint angles (7-dim).
- **Temporal Memory**: Zero lags, no temporal convolutions, no frame deltas, and no hold counters. Evaluates instantaneously on the current observation frame.
- **Output Action Gating**: Single unified contact state. When `contact_state == True` (tightness achieved):
  - Contact Stop: Freezes finger closure joints J3/J4/J5 at their current angles to prevent slip or over-torquing.
  - Lift Enable: Releases J1 upward movement.

## Evaluation & LOBO (Leave-One-Bottle-Out) Benchmarks
Trained and cross-validated over all 125 episodes across 5 bottle sizes:
- **LOBO Average F1**: **94.14%** (Winner across all 5 unseen bottle size folds: 92.7%, 96.3%, 94.7%, 93.4%, 93.6%).
- **Held-out Test (25 trajectories)**: F1 **92.44%**, Precision **94.23%**, Recall **90.71%**, Miss Rate **0.0%**.
- **Contact Timing**: Triggers median **-8.0 frames (~0.27s)** before nominal tightness and **-32.0 frames (~1.07s)** before J1 lift onset.

## Files
- `monitor.pt`: Standalone PyTorch weights and metadata for the Instant LOBO Contact Monitor (17 KB).
- `checkpoints/last.pt`: Deployable combined policy containing frozen EMA WRM_wrap 50k weights plus this monitor.
