# Realman–Beaver policy baselines

This package trains and deploys six baselines from the local LeRobot dataset.
The original diffusion models remain available; three flow-matching models are
provided alongside them.

| config | observations | trajectory model |
| --- | --- | --- |
| `original_dp.yaml` | camera + 7 joints | native LeRobot `DiffusionPolicy` |
| `dp_beaver.yaml` | camera + 7 joints + Beaver | native LeRobot `DiffusionPolicy` |
| `rdp_like.yaml` | camera + 7 joints + Beaver | asymmetric tokenizer + latent diffusion |
| `fm.yaml` | camera + 7 joints | conditional flow matching |
| `fm_beaver.yaml` | camera + 7 joints + Beaver | conditional flow matching |
| `rfm.yaml` | camera + 7 joints + Beaver | asymmetric tokenizer + latent flow matching |

## Flow-matching models

`fm` and `fm_beaver` use the same ResNet18 observation encoder and
FiLM-conditioned 1D U-Net shape as the diffusion baselines, but they do not
instantiate `DiffusionPolicy` or a diffusion scheduler. Given normalized action
data `x₁`, Gaussian noise `x₀`, and `t ~ Uniform(0, 1)`, training uses the linear
path

```text
xₜ = (1 - t) x₀ + t x₁
target velocity = x₁ - x₀
```

The U-Net predicts the velocity with padding-masked MSE. Inference starts from
Gaussian noise and integrates the learned ODE from `t=0` to `t=1` with the
configured number of Euler steps (`flow_num_inference_steps`).

`rfm` preserves the reactive structure of `rdp_like`: an action-only encoder
produces a 16-token latent trajectory, while a causal GRU decodes actions using
the latest Beaver frame. Its slow visual branch generates latent tokens with
flow matching instead of diffusion. The tokenizer is trained first and frozen
while the slow latent flow is trained.

## Beaver validity masking

All Beaver-conditioned variants load these fields:

- `observation.beaver.distance_mm`: `9×4×4`
- `observation.beaver.present`: `9`
- `observation.beaver.target_status`: `9×4×4`

Every distance pixel is disabled before it reaches a policy or tokenizer unless
its `beaver_status` is listed in `DatasetConfig.beaver_valid_statuses`. The
shipped configuration accepts status 5 (valid) and 9 (weak signal). Other codes,
including 255 (no target), are zeroed pixel by pixel. The sensor-level `present`
mask still zeroes the complete grid of a disconnected sensor.

`fm_beaver` and `dp_beaver` append the resulting 144 masked distance values and
9 presence flags to the normalized 7-D joint state. `rfm` and `rdp_like` pass
the masked distance grids only to their fast tokenizer decoder; their slow
visual trajectory models do not consume Beaver data.

## Training

Run one baseline from the repository root:

```bash
python -m policies.realman_beaver.train \
  --config policies/realman_beaver/configs/fm.yaml

python -m policies.realman_beaver.train \
  --config policies/realman_beaver/configs/fm_beaver.yaml

python -m policies.realman_beaver.train \
  --config policies/realman_beaver/configs/rfm.yaml
```

The original configs remain unchanged entry points:

```bash
python -m policies.realman_beaver.train \
  --config policies/realman_beaver/configs/original_dp.yaml
python -m policies.realman_beaver.train \
  --config policies/realman_beaver/configs/dp_beaver.yaml
python -m policies.realman_beaver.train \
  --config policies/realman_beaver/configs/rdp_like.yaml
```

Train all six sequentially (the safe default for one GPU), or explicitly launch
them in parallel:

```bash
policies/realman_beaver/train_all.sh
policies/realman_beaver/train_all.sh --parallel
```

`DEVICE`, `NUM_WORKERS`, and `OUTPUT_ROOT` override the launcher defaults.
Additional CLI arguments are forwarded to every trainer. For example:

```bash
policies/realman_beaver/train_all.sh --max-steps 10 --val-fraction 0
```

Direct policies write `metrics.jsonl` and `last.pt`. Reactive models write a
tokenizer checkpoint first, then either `latent_dp_metrics.jsonl` or
`latent_fm_metrics.jsonl` and a deployable `last.pt`. Existing W&B options apply
to every variant.

## Deployment

Load a checkpoint and reset the policy at the start of each episode:

```python
import torch

from policies.realman_beaver.checkpoint import load_policy

policy = load_policy("policies/output/rfm/last.pt", device="cuda:0")
policy.reset()

observation = {
    "image": camera_chw_float01.to("cuda:0"),
    "state": joint_configuration.to("cuda:0"),
    "beaver_distance": distance_mm.to("cuda:0"),
    "beaver_present": sensor_present.to("cuda:0"),
    "beaver_status": target_status.to("cuda:0"),
}
with torch.inference_mode():
    action = policy.select_action(observation)
```

Omit the Beaver fields only for `original_dp` and `fm`. Beaver-conditioned
deployment requires `beaver_status`, preventing unmasked invalid pixels from
silently reaching a model.

## Verification

```bash
python -m unittest policies.realman_beaver.tests.test_policy -v
```

The tests cover all six model constructors, loss/backpropagation, diffusion and
flow sampling, both reactive pipelines, online action selection, and per-pixel
status masking.
