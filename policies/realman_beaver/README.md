# Realman–Beaver LeRobot diffusion policies

This folder trains and deploys three policies from the local LeRobot dataset at
`/home/yuyuan/AIRO-Doffy/datasets/WRM_grasp_lero`.

| config | policy input | training structure |
| --- | --- | --- |
| `original_dp.yaml` | one camera + 7 joints | native LeRobot `DiffusionPolicy` |
| `dp_beaver.yaml` | one camera + 7 joints + Beaver | native LeRobot `DiffusionPolicy`; Beaver is appended to state |
| `rdp_like.yaml` | one camera + 7 joints + Beaver | asymmetric action tokenizer, then native LeRobot latent DP |

The first two variants instantiate LeRobot 0.5's real ResNet18 +
FiLM-conditioned 1D U-Net implementation. They use its diffusion scheduler,
epsilon-noise MSE objective, padding mask, action-chunk sampling, observation
queue, and receding-horizon deployment. `dp_beaver` expands each state from 7
to 160 values: 7 normalized joints, 144 normalized distances, and 9 sensor
presence flags.

The RDP-like policy follows the two-stage structure of
[Reactive Diffusion Policy](https://github.com/xiaoxiaoxh/reactive_diffusion_policy),
with tactile images replaced by Beaver distance grids:

1. The asymmetric Action Tokenizer's CNN encoder sees only a 32-step action
   trajectory. It produces sixteen 4-D tokens (downsample ratio 2, matching
   the reference RDP's `downsampled_input_h = 16`). A causal GRU reconstructs
   actions from those tokens and all 32 Beaver frames using L1 reconstruction
   plus a `1e-6` KL term.
2. The trained tokenizer is frozen. A native LeRobot diffusion policy learns
   the latent-token trajectory from two camera/joint observations sampled at
   12 Hz. The slow branch samples with the full 100-step DDPM schedule,
   matching the reference RDP (`num_inference_steps: 100`).
3. During rollout, the slow branch replans every 8 control ticks (3 Hz at
   the dataset's 24 Hz rate, matching the DP variants' `n_action_steps=8`
   receding horizon). The fast GRU runs every tick with the latest Beaver
   frame. Beaver data never enters the slow visual diffusion model.

This is deliberately called **RDP-like**: it preserves RDP's action-only
encoder, asymmetric sensor decoder, frozen-tokenizer second stage, and
slow/fast execution, but Beaver distances are not GelSight/McTac images.

## Dataset and normalization

The supplied dataset was inspected directly:

| field | value |
| --- | --- |
| episodes / frames | 51 / 11,934 |
| frequency | 24 Hz |
| `observation.images.camera_0` | `3×480×640`, float in `[0, 1]` after decoding |
| `observation.state` / `action` | 7 / 7 |
| `observation.beaver.distance_mm` | `9×4×4` |
| `observation.beaver.present` | 9 |
| `observation.beaver.target_status` | `9×4×4`, VL53L7CX per-pixel status codes |

Original DP action windows use LeRobot's canonical two-observation alignment:
relative indices `[-1, 0, ..., 14]` for a 16-step horizon. State and action use
LeRobot's min/max mapping to `[-1, 1]`; images use its mean/std mapping. The
state/action extrema are recomputed over every parquet shard because the
current `meta/stats.json` covers only part of the dataset. Image mean/std still
comes from that metadata, so regenerate the dataset statistics if the videos
have changed.

Beaver distances are masked before normalization, in two stages. The
sensor-level mask (`present`) zeroes every pixel of a disconnected sensor. The
pixel-level mask uses `target_status`: only codes 5 (valid) and 9 (weak
signal) carry a usable distance, so the other ~35% of pixels — mostly 255 (no
target), which read as noisy maximum-range garbage — are zeroed and never
reach the model. The active codes live in
`DatasetConfig.beaver_valid_statuses`.

## Training

Run from the repository root:

```bash
# Recommended on this machine: all three concurrently, with separate logs and outputs.
policies/realman_beaver/train_all.sh

# Or force sequential training.
policies/realman_beaver/train_all.sh --sequential

# Individual jobs:
python -m policies.realman_beaver.train \
  --config policies/realman_beaver/configs/original_dp.yaml

python -m policies.realman_beaver.train \
  --config policies/realman_beaver/configs/dp_beaver.yaml

python -m policies.realman_beaver.train \
  --config policies/realman_beaver/configs/rdp_like.yaml
```

The launcher writes checkpoints and timestamped logs to
`policies/output/{original_dp,dp_beaver,rdp_like}`. `DEVICE`, `NUM_WORKERS`, and
`OUTPUT_ROOT` can be overridden as environment variables. Additional arguments
are forwarded to every trainer, for example
`train_all.sh --max-steps 10 --val-fraction 0`.

The shipped schedule runs original DP and Beaver DP for 100,000 optimizer
steps. RDP-like follows the reference RDP schedule exactly: the tokenizer runs
601 epochs and the latent DP runs 401 epochs, both with batch size 64 and
`max_train_steps` null — every epoch runs the full dataloader. Each DP stage
writes numbered snapshots at steps 25k, 50k, 75k, and 100k, plus its final
`last.pt`; RDP stages write the same step snapshots plus `tokenizer_last.pt`
and `last.pt` at the end of each stage.

For RDP-like training, `tokenizer_last.pt` is written after stage 1 and
`last.pt` after stage 2. To reuse a trained tokenizer:

```bash
python -m policies.realman_beaver.train \
  --config policies/realman_beaver/configs/rdp_like.yaml \
  --tokenizer-checkpoint policies/output/rdp_like/tokenizer_last.pt
```

A real one-step CUDA smoke test is:

```bash
python -m policies.realman_beaver.train \
  --config policies/realman_beaver/configs/original_dp.yaml \
  --device cuda:0 --num-workers 0 --max-steps 1 --val-fraction 0 \
  --output-dir /tmp/original_dp_smoke
```

`--max-steps 1` bounds both RDP stages. Checkpoints contain the model,
optimizer, scheduler, resolved configuration, full normalization state, and EMA
weights.

## Can all three train together?

Yes with the shipped batch sizes on this machine's RTX 4070 12 GiB. Measured
one-step peak allocations were approximately 1.75 GiB for original DP,
1.85 GiB for Beaver DP, and 1.24 GiB for the largest RDP stage. A concurrent
one-step run of all three completed successfully.

They still share one GPU, so concurrent jobs do not provide three-GPU
throughput and may take longer overall. Sequential training is the safer choice
for maximum throughput; concurrent runs are useful when comparing early
learning curves. If running concurrently, reduce each job to one data-loader
worker to avoid twelve default workers competing for CPU and disk bandwidth.

## Deployment

Load the EMA policy and call `reset()` at the start of every robot episode:

```python
import torch

from policies.realman_beaver.checkpoint import load_policy

policy = load_policy(
    "policies/output/rdp_like/last.pt",
    device="cuda:0",
)
policy.reset()

observation = {
    "image": camera_chw_float01.to("cuda:0"),  # [3, 480, 640]
    "state": joint_configuration.to("cuda:0"),  # [7]
    "beaver_distance": distance_mm.to("cuda:0"),  # [9, 4, 4]
    "beaver_present": sensor_present.to("cuda:0"),  # [9]
    "beaver_status": target_status.to("cuda:0"),  # [9, 4, 4], VL53L7CX codes
}
with torch.inference_mode():
    action = policy.select_action(observation)  # [1, 7]
```

For `original_dp`, omit the two Beaver fields. For `dp_beaver`, use the same
observation structure as RDP-like. The wrappers accept batched observations as
well. This package outputs joint targets only; robot-side joint limits,
command-rate enforcement, collision handling, and emergency-stop logic remain
mandatory.

## Verification

```bash
python -m unittest policies.realman_beaver.tests.test_policy -v
```

The tests assert that both DP variants contain LeRobot's native
`DiffusionPolicy`, run loss/backpropagation and sampling, exercise both RDP
training stages, and execute each online deployment path.

Method references: [Diffusion Policy](https://arxiv.org/abs/2303.04137),
[Stanford implementation](https://github.com/real-stanford/diffusion_policy),
[Reactive Diffusion Policy paper](https://arxiv.org/abs/2503.02881), and
[RDP implementation](https://github.com/xiaoxiaoxh/reactive_diffusion_policy).
