"""Offline physical-unit metrics and modality ablations for WRM_codex."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from policies.realman_beaver.checkpoint import load_policy
from policies.realman_beaver.configuration import CODEX_BEAVER_VARIANT
from policies.realman_beaver.dataset import RealmanPolicyDataset, episode_split


ABLATIONS: dict[str, dict[str, str]] = {
    "complete": {},
    "image_zero": {"image": "zero"},
    "image_shuffle": {"image": "shuffle"},
    "joint_zero": {"joint": "zero"},
    "joint_shuffle": {"joint": "shuffle"},
    "beaver_zero": {"beaver": "zero"},
    "beaver_shuffle": {"beaver": "shuffle"},
}


def _parse_episodes(value: str) -> tuple[int, ...]:
    episodes: list[int] = []
    for part in value.split(","):
        if "-" in part:
            first, last = (int(item) for item in part.split("-", 1))
            if first < 0 or last < first:
                raise argparse.ArgumentTypeError(f"invalid episode range: {part}")
            episodes.extend(range(first, last + 1))
        else:
            episode = int(part)
            if episode < 0:
                raise argparse.ArgumentTypeError("episodes must be non-negative")
            episodes.append(episode)
    if not episodes or len(episodes) != len(set(episodes)):
        raise argparse.ArgumentTypeError("episodes must be non-empty and unique")
    return tuple(episodes)


def _to_device(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate(
    checkpoint: str | Path,
    *,
    dataset_root: str | None = None,
    episodes: Sequence[int] | None = None,
    device: str = "cpu",
    batch_size: int = 16,
    num_workers: int = 0,
    max_batches: int | None = None,
    latency_repeats: int = 10,
) -> dict[str, object]:
    policy = load_policy(checkpoint, device=device, use_ema=True)
    if policy.config.model.variant != CODEX_BEAVER_VARIANT:
        raise ValueError("offline_eval_codex.py requires a WRM_codex checkpoint")
    if dataset_root is not None:
        policy.config.dataset.root = dataset_root
    if episodes is None:
        _, selected_episodes = episode_split(policy.config.dataset)
        if not selected_episodes:
            raise ValueError(
                "This checkpoint has no validation split; pass --episodes explicitly"
            )
    else:
        selected_episodes = sorted(int(episode) for episode in episodes)
    dataset = RealmanPolicyDataset(
        policy.config, selected_episodes, stage="policy"
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.device(device).type == "cuda",
    )
    torch_device = torch.device(device)
    modes = {
        name: {
            "absolute": torch.zeros(policy.config.model.action_dim),
            "squared": torch.zeros(policy.config.model.action_dim),
            "count": 0,
            "smoothness_sum": 0.0,
            "smoothness_count": 0,
            "boundary_sum": 0.0,
            "boundary_count": 0,
            "previous": {},
        }
        for name in ABLATIONS
    }
    per_bottle_absolute: dict[int, Tensor] = defaultdict(
        lambda: torch.zeros(policy.config.model.action_dim)
    )
    per_bottle_count: dict[int, int] = defaultdict(int)
    objective_sum = 0.0
    objective_count = 0
    first_batch: dict[str, Tensor] | None = None

    policy.eval()
    with torch.inference_mode():
        for batch_index, cpu_batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = _to_device(cpu_batch, torch_device)
            if first_batch is None:
                first_batch = batch
            _, loss_metrics = policy.compute_loss(batch)
            batch_count = batch["state"].shape[0]
            objective_sum += loss_metrics["loss"] * batch_count
            objective_count += batch_count

            start = policy.config.model.n_obs_steps - 1
            steps = policy.config.model.n_action_steps
            target = batch["action"][:, start : start + steps]
            valid = ~batch["action_is_pad"][:, start : start + steps]
            episode_index = batch.get("episode_index")
            frame_index = batch.get("frame_index")
            if episode_index is None or frame_index is None:
                raise RuntimeError(
                    "WRM_codex dataset samples must expose episode/frame metadata"
                )
            episode_index = episode_index.reshape(-1).cpu()
            frame_index = frame_index.reshape(-1).cpu()

            for mode_name, ablation in ABLATIONS.items():
                predicted = policy.predict_action_chunk(batch, ablations=ablation)
                error = predicted - target
                valid_float = valid.unsqueeze(-1).to(error.dtype)
                accumulator = modes[mode_name]
                accumulator["absolute"] += (
                    error.abs() * valid_float
                ).sum(dim=(0, 1)).cpu()
                accumulator["squared"] += (
                    error.square() * valid_float
                ).sum(dim=(0, 1)).cpu()
                accumulator["count"] += int(valid.sum())
                if predicted.shape[1] > 1:
                    smoothness = (predicted[:, 1:] - predicted[:, :-1]).abs()
                    smooth_valid = valid[:, 1:] & valid[:, :-1]
                    accumulator["smoothness_sum"] += float(
                        (smoothness * smooth_valid.unsqueeze(-1)).sum()
                    )
                    accumulator["smoothness_count"] += int(smooth_valid.sum()) * 7

                predicted_cpu = predicted.cpu()
                for sample_index, (episode, frame) in enumerate(
                    zip(episode_index.tolist(), frame_index.tolist())
                ):
                    previous = accumulator["previous"].get(int(episode))
                    if previous is not None and int(frame) == previous[0] + 1:
                        disagreement = (
                            predicted_cpu[sample_index, 0] - previous[1][1]
                        ).abs()
                        accumulator["boundary_sum"] += float(disagreement.sum())
                        accumulator["boundary_count"] += disagreement.numel()
                    accumulator["previous"][int(episode)] = (
                        int(frame),
                        predicted_cpu[sample_index],
                    )

                if mode_name == "complete":
                    error_cpu = error.abs().cpu()
                    valid_cpu = valid.cpu()
                    for sample_index, episode in enumerate(episode_index.tolist()):
                        bottle = int(episode) // 25 + 1
                        per_bottle_absolute[bottle] += (
                            error_cpu[sample_index]
                            * valid_cpu[sample_index].unsqueeze(-1)
                        ).sum(dim=0)
                        per_bottle_count[bottle] += int(valid_cpu[sample_index].sum())

    if not objective_count or first_batch is None:
        raise RuntimeError("No offline evaluation samples were processed")
    mode_results: dict[str, object] = {}
    for name, accumulator in modes.items():
        count = max(int(accumulator["count"]), 1)
        joint_mae = accumulator["absolute"] / count
        joint_rmse = (accumulator["squared"] / count).sqrt()
        mode_results[name] = {
            "trajectory_mae_rad": float(joint_mae.mean()),
            "trajectory_rmse_rad": float(joint_rmse.square().mean().sqrt()),
            "per_joint_mae_rad": joint_mae.tolist(),
            "per_joint_rmse_rad": joint_rmse.tolist(),
            "action_smoothness_mean_abs_rad": float(
                accumulator["smoothness_sum"]
                / max(int(accumulator["smoothness_count"]), 1)
            ),
            "overlap_boundary_discontinuity_mean_abs_rad": float(
                accumulator["boundary_sum"]
                / max(int(accumulator["boundary_count"]), 1)
            ),
        }

    latency_ms: list[float] = []
    with torch.inference_mode():
        for _ in range(2):
            policy.predict_action_chunk(first_batch)
        _synchronize(torch_device)
        for _ in range(latency_repeats):
            start_time = time.perf_counter()
            policy.predict_action_chunk(first_batch)
            _synchronize(torch_device)
            latency_ms.append((time.perf_counter() - start_time) * 1000.0)

    parameter_count = sum(
        parameter.numel() for parameter in policy.parameters() if parameter.requires_grad
    )
    weight_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in policy.state_dict().values()
    )
    return {
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "episodes": selected_episodes,
        "validation_objective": objective_sum / objective_count,
        "ablations": mode_results,
        "per_bottle": {
            str(bottle): {
                "trajectory_mae_rad": float(
                    (absolute / max(per_bottle_count[bottle], 1)).mean()
                ),
                "per_joint_mae_rad": (
                    absolute / max(per_bottle_count[bottle], 1)
                ).tolist(),
            }
            for bottle, absolute in sorted(per_bottle_absolute.items())
        },
        "trainable_parameters": parameter_count,
        "state_dict_bytes": weight_bytes,
        "latency": {
            "device": str(torch_device),
            "batch_size": int(first_batch["state"].shape[0]),
            "repeats": latency_repeats,
            "mean_ms_per_batch": sum(latency_ms) / len(latency_ms),
            "samples_ms": latency_ms,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root")
    parser.add_argument("--episodes", type=_parse_episodes)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--latency-repeats", type=int, default=10)
    parser.add_argument("--output", help="Optional JSON result path")
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = evaluate(
        args.checkpoint,
        dataset_root=args.dataset_root,
        episodes=args.episodes,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
        latency_repeats=args.latency_repeats,
    )
    rendered = json.dumps(result, indent=2)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
