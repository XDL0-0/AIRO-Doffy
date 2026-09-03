# Realman–Beaver policy baselines

This package trains and deploys the supported policy variants from the local
LeRobot dataset. The original diffusion and flow-matching models remain
available alongside structured and history-aware Beaver Diffusion Policies.

| config | observations | trajectory model |
| --- | --- | --- |
| `original_dp.yaml` | camera + 7 joints | native LeRobot `DiffusionPolicy` |
| `dp_beaver.yaml` | camera + 7 joints + Beaver | native LeRobot `DiffusionPolicy` |
| `dp_beaver_closure.yaml` | camera + 7 joints globally; current/delta Beaver locally | unchanged native DP + gated closure-joint residual |
| `dp_beaver_enc.yaml` | camera + 7 joints + structured Beaver | shared sensor encoder + native DP |
| `dp_beaver_near.yaml` | camera + 7 joints + structured near-field Beaver | shared sensor encoder + native DP |
| `dp_beaver_near_gate.yaml` | camera + 7 joints + gated near-field Beaver | gated shared sensor encoder + native DP |
| `dp_beaver_key4.yaml` | camera + 7 joints + sensors 01/02/10/11 | four independent tokens + concat + LayerNorm + native DP |
| `dp_beaver_key4_pca.yaml` | camera + 7 joints + sensors 01/02/10/11 | per-sensor fixed PCA-4 + concat + LayerNorm + native DP |
| `WRM_temporal.yaml` | camera + 7 joints + 12-frame sensor 01/02/10/11 history | temporal Beaver encoder + grasp-state auxiliary head + native DP |
| `WRM_wrap_delta.yaml` | same temporal Key4/contact enclosure input as WRM_wrap | replan-anchored joint-delta DP + per-joint J3/J4 wrap/lift gate |
| `WRM_adaptive_all_train.yaml` | camera + q/Δq + multi-scale 01/02/10/11 contact field | relative-geometry sensor attention + grasp-conditioned native DP |
| `rdp_like.yaml` | camera + 7 joints + Beaver | asymmetric tokenizer + latent diffusion |
| `rdp_like_key4_all_train.yaml` | camera + 7 joints + sensors 01/02/10/11 | Key4 asymmetric tokenizer + latent diffusion; all 125 episodes |
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

`rdp_like_key4` keeps that same slow visual latent-DP path but configures the
fast decoder's frame encoder to select physical sensors 01/02/10/11. The raw
dataset and deployment observation remain nine-sensor tensors; unselected
sensors are removed inside the tokenizer before sensor embeddings and fusion.

The three structured DP variants instead preserve the nine-sensor axis. One
shared MLP maps every sensor's 4×4 cells to a 32-D token, a physical sensor
embedding preserves identity, and mean/max aggregation produces a 64-D Beaver
feature. `dp_beaver_near` adds a cell-wise 0–300 mm proximity channel;
`dp_beaver_near_gate` additionally learns independent sigmoid sensor gates.
The 64-D feature is concatenated with normalized 7-D joints to give the native
Diffusion Policy a 71-D state. These paths use a masked `[0,1]` global distance
representation and do not change the old `dp_beaver` 160-D flat baseline.

The Key4 variants use physical sensor slots 1/2/5/6, corresponding to sensor
IDs 01/02/10/11. `dp_beaver_key4` independently maps each sensor's normalized
near-field and validity data to 32 dimensions, concatenates the four tokens to
128 dimensions, then applies LayerNorm. `dp_beaver_key4_pca` first maps raw
millimetres to `1 - clip(distance_mm / 300, 0, 1)`, fits a separate standardized
PCA on each sensor using training episodes only, keeps four components per
sensor, and LayerNorms the concatenated 16-D feature. Its standardization and
PCA parameters are saved in every checkpoint.

`dp_beaver_closure` keeps Beaver completely out of the global DP observation.
Its global branch receives only RGB and normalized joints. A shared per-sensor
MLP encodes current distance, one-frame distance change, and valid/present
masks; masked mean pooling combines available sensors. A lightweight MLP then
uses that embedding with `q_t`, `dq_t`, and the masked flattened `dB_t` to emit
a scalar gate and a fixed-mask joint residual. Training subtracts the residual
from demonstrated actions to form the nominal DP target, then adds it back at
inference. The `tightness`/`grasp_state` field is used only for a BCE auxiliary
head and a post-grasp residual-magnitude regularizer; it is never an input or a
hand-authored closure label. Missing sensors are excluded from pooling, and an
all-missing frame produces zero gate and zero correction.

`WRM_temporal` uses a separate 12-frame Beaver history without changing the
camera/joint observation horizon. For each of sensors 01/02/10/11, every 4×4
cell carries normalized distance, normalized temporal delta, validity, and a
zero-value flag. A shared frame MLP and GRU produce one 64-D token per physical
sensor; the fixed-order four-token concatenation is fused to a LayerNormed
64-D Beaver feature. A learned auxiliary head predicts grasp state from that
feature. Its predicted probability—not the ground-truth tightness label—is
appended to the native Diffusion Policy conditioning. Training minimizes
`diffusion_loss + 0.2 * grasp_loss` by default.

Temporal distance P5/P95 bounds and fallback medians are fit independently per
sensor from training episodes only. Zero and invalid cells are imputed from an
earlier valid history value, or the training median when none exists; their
flags remain visible to the encoder and their delta is zero when no valid
temporal difference exists. The fitted statistics are model buffers, so normal
checkpoint save/load reproduces the training-time preprocessing at evaluation.

`WRM_adaptive` is the size-agnostic closed-loop variant. It deliberately keeps
only sensors 01/02/10/11 because the other five channels are frequently invalid
or far-field and can destabilize near-field control. It nevertheless uses every
4×4 pixel from the reliable sensors. Per-cell inputs contain robust absolute
range, within-sensor and cross-sensor relative geometry, 50/100/200/400 mm
proximity channels, changes over 1/3/6/11 frames, and explicit validity masks.
A cross-sensor Transformer and quality-weighted masked attention pool the
continuous contact field without an object-size label. Quality is the joint
spatial/temporal valid fraction: sporadic sensors are downweighted, invalid
cells contribute zero geometry, and an all-invalid Key4 frame produces an
exactly zero Beaver feature. The native DP is directly conditioned on normalized
joints, six-frame joint motion, the Beaver feature, and the learned grasp
probability. Consistent range noise plus pixel/sensor dropout during training
improves sensor robustness without inventing object classes.

The all-125 configuration has no validation split and is intended as the final
deployment fit after architecture tests. Its auxiliary objective is
`diffusion_loss + 0.5 * grasp_loss`; intermediate 10k checkpoints should be kept
for real-robot selection rather than selecting from training loss alone.

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

The structured Beaver configs are independent entry points:

```bash
python -m policies.realman_beaver.train \
  --config policies/realman_beaver/configs/dp_beaver_enc.yaml
python -m policies.realman_beaver.train \
  --config policies/realman_beaver/configs/dp_beaver_near.yaml
python -m policies.realman_beaver.train \
  --config policies/realman_beaver/configs/dp_beaver_near_gate.yaml
python -m policies.realman_beaver.train \
  --config policies/realman_beaver/configs/dp_beaver_key4.yaml
python -m policies.realman_beaver.train \
  --config policies/realman_beaver/configs/dp_beaver_key4_pca.yaml
python -m policies.realman_beaver.train \
  --config policies/realman_beaver/configs/WRM_temporal.yaml
python -m policies.realman_beaver.train \
  --config policies/realman_beaver/configs/WRM_adaptive_all_train.yaml
```

Train the original eleven sequentially (the safe default for one GPU), or explicitly launch
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

The prepared Cluster 6 job uses one GPU, eight CPUs, batch size 32, validation
episodes 50–74, and 100,000 optimizer steps. It stages the requested local
`WRM_grasp_cylinder_different_sizes_lero_tightness` dataset rather than
downloading a dataset at runtime:

```bash
# Submit .gpulab/wrm-temporal-cluster6-bs32.json with the GPUlab CLI first,
# then stage source, local data, and credentials into the returned job.
.gpulab/watch-and-stage-wrm-temporal-cluster6-bs32.sh JOB_ID
```

Outputs persist under `/project_ghent/AIRO-Doffy/` and upload to
`IXDLI/AIRO-Doffy-WRM-Grasp-WRM-temporal` when `UPLOAD_HF=1`.

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

For `WRM_temporal`, each evaluation JSONL record also includes
`grasp_probability`, Beaver feature mean/std, and the 01/02/10/11 temporal-token
standard deviations. These per-frame fields can be plotted directly to inspect
the inferred grasp transition and sensor dynamics; the ground-truth `tightness`
label remains an offline training/evaluation label and is never sent to the
deployed policy.

For `WRM_adaptive`, evaluation logs include grasp probability, Beaver feature
standard deviation, Key4 attention weights/entropy, and the fraction of current
Key4 pixels inside 50 mm. These fields distinguish failed contact acquisition
from chunk-to-chunk action mode switching.

## Verification

```bash
python -m unittest discover -s policies/realman_beaver/tests -v
```

The tests cover model constructors, loss/backpropagation, diffusion and flow
sampling, reactive pipelines, structured Beaver preprocessing and gradient
flow, checkpoint reconstruction, online action selection, per-pixel status
masking, and all-invalid sensor fallback.
