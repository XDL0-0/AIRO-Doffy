"""Train the diffusion, flow-matching, RDP, and RFM baselines."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import wandb
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from policies.realman_beaver.configuration import RealmanBeaverConfig, load_config
from policies.realman_beaver.dataset import (
    LatentNormalizer,
    ObservationNormalizer,
    RealmanPolicyDataset,
    episode_split,
)
from policies.realman_beaver.modeling import (
    RDPPolicy,
    RFMPolicy,
    build_policy,
    build_tokenizer,
)

LossFunction = Callable[[dict[str, Tensor]], tuple[Tensor, dict[str, float]]]
_WANDB_METRIC_KINDS: set[str] = set()


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = decay
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(parameter.detach(), 1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        parameters = dict(model.named_parameters())
        for name, value in self.shadow.items():
            parameters[name].copy_(value)

    def state_dict(self) -> dict[str, Tensor]:
        return self.shadow

    def load_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        missing = set(self.shadow) - set(state_dict)
        if missing:
            raise ValueError(f"EMA checkpoint is missing: {sorted(missing)[:5]}")
        for name in self.shadow:
            self.shadow[name].copy_(state_dict[name])


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _to_device(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _wandb_log(kind: str, step: int, values: dict[str, float]) -> None:
    """Log metrics against a stage-local step without rewinding W&B's step."""
    if wandb.run is None:
        return
    step_key = f"{kind}/global_step"
    if kind not in _WANDB_METRIC_KINDS:
        wandb.define_metric(step_key)
        wandb.define_metric(f"{kind}/*", step_metric=step_key)
        _WANDB_METRIC_KINDS.add(kind)
    payload = {f"{kind}/{key}": value for key, value in values.items()}
    payload[step_key] = step
    # Do not pass W&B's global step: tokenizer and latent training each start
    # from zero, while the stage-local custom axes remain independent.
    wandb.log(payload)


def _parse_episode_spec(value: str) -> tuple[int, ...]:
    """Parse zero-based comma-separated episode indices and inclusive ranges."""
    episodes: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            raise argparse.ArgumentTypeError("episode ranges must not be empty")
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            try:
                start, end = int(start_text), int(end_text)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"invalid episode range: {item}"
                ) from exc
            if start < 0 or end < start:
                raise argparse.ArgumentTypeError(f"invalid episode range: {item}")
            episodes.extend(range(start, end + 1))
        else:
            try:
                episode = int(item)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"invalid episode index: {item}"
                ) from exc
            if episode < 0:
                raise argparse.ArgumentTypeError("episode indices must be non-negative")
            episodes.append(episode)
    if len(set(episodes)) != len(episodes):
        raise argparse.ArgumentTypeError("episode indices must not contain duplicates")
    return tuple(episodes)


def _mean_metrics(metrics: Iterable[dict[str, float]]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    count = 0
    for item in metrics:
        count += 1
        for key, value in item.items():
            totals[key] += float(value)
    return {key: value / max(count, 1) for key, value in totals.items()}


def _make_loader(
    dataset: RealmanPolicyDataset,
    batch_size: int,
    workers: int,
    device: torch.device,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def _make_scheduler(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    def factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loss_function: LossFunction,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    metrics = []
    for batch in tqdm(loader, desc="validation", leave=False):
        _, values = loss_function(_to_device(batch, device))
        metrics.append(values)
    model.train()
    return _mean_metrics(metrics)


def _save_checkpoint(
    path: Path,
    kind: str,
    config: RealmanBeaverConfig,
    model: nn.Module,
    ema: ExponentialMovingAverage,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "kind": kind,
            "config": config.to_dict(),
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "metrics": metrics,
        },
        temporary,
    )
    os.replace(temporary, path)


def _train_stage(
    *,
    kind: str,
    config: RealmanBeaverConfig,
    model: nn.Module,
    loss_function: LossFunction,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    device: torch.device,
    output_dir: Path,
    epochs: int,
    max_steps: int | None,
    learning_rate: float,
    weight_decay: float,
    metrics_name: str,
    last_name: str,
    resume_from: str | None = None,
) -> ExponentialMovingAverage:
    training = config.training
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        betas=training.betas,
        eps=training.eps,
        weight_decay=weight_decay,
    )
    natural_steps = len(train_loader) * epochs
    total_steps = max_steps or natural_steps
    scheduler = _make_scheduler(optimizer, training.warmup_steps, total_steps)
    ema = ExponentialMovingAverage(model, training.ema_decay)
    use_amp = training.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    metrics_path = output_dir / metrics_name
    start_epoch, global_step = 0, 0
    if resume_from:
        checkpoint = torch.load(
            Path(resume_from).expanduser(), map_location=device, weights_only=True
        )
        if checkpoint.get("kind") != kind:
            raise ValueError(
                f"Cannot resume {kind} from checkpoint kind={checkpoint.get('kind')}"
            )
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
    model.train()

    for epoch in range(start_epoch, epochs):
        epoch_metrics = []
        progress = tqdm(train_loader, desc=f"{kind} {epoch + 1}/{epochs}")
        for batch in progress:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                loss, values = loss_function(_to_device(batch, device))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, training.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)
            global_step += 1
            epoch_metrics.append(values)
            progress.set_postfix(loss=f"{values['loss']:.4f}")
            if global_step % training.log_every_steps == 0:
                log_values = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "lr": scheduler.get_last_lr()[0],
                    **values,
                }
                with metrics_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(log_values) + "\n")
                _wandb_log(
                    kind,
                    global_step,
                    {
                        key: value
                        for key, value in log_values.items()
                        if key != "global_step"
                    },
                )
            if global_step % training.checkpoint_every_steps == 0:
                milestone_metrics = {
                    f"train_{key}": value for key, value in values.items()
                }
                milestone_path = output_dir / f"{kind}_step_{global_step:06d}.pt"
                _save_checkpoint(
                    milestone_path,
                    kind,
                    config,
                    model,
                    ema,
                    optimizer,
                    scheduler,
                    epoch,
                    global_step,
                    milestone_metrics,
                )
                print(f"checkpoint={milestone_path}")
            if max_steps is not None and global_step >= max_steps:
                break

        combined = {
            f"train_{key}": value for key, value in _mean_metrics(epoch_metrics).items()
        }
        if val_loader is not None:
            combined.update(
                {
                    f"val_{key}": value
                    for key, value in _evaluate(
                        model, loss_function, val_loader, device
                    ).items()
                }
            )
        record = {"epoch": epoch, "global_step": global_step, **combined}
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        _wandb_log(kind, global_step, record)
        print(json.dumps({"stage": kind, **record}, sort_keys=True))

        finished = (max_steps is not None and global_step >= max_steps) or (
            max_steps is None and epoch + 1 == epochs
        )
        if finished:
            _save_checkpoint(
                output_dir / last_name,
                kind,
                config,
                model,
                ema,
                optimizer,
                scheduler,
                epoch,
                global_step,
                combined,
            )
        if max_steps is not None and global_step >= max_steps:
            break
    return ema


def _load_tokenizer_checkpoint(
    model: nn.Module, path: str, device: torch.device
) -> None:
    checkpoint = torch.load(
        Path(path).expanduser(), map_location=device, weights_only=True
    )
    if checkpoint.get("kind") != "tokenizer":
        raise ValueError("tokenizer_checkpoint must point to a tokenizer checkpoint")
    model.load_state_dict(checkpoint["model"])
    parameters = dict(model.named_parameters())
    for name, value in checkpoint.get("ema", {}).items():
        if name in parameters:
            parameters[name].data.copy_(value)


@torch.no_grad()
def _fit_latent_normalizer(
    tokenizer: nn.Module,
    normalizer: ObservationNormalizer,
    config: RealmanBeaverConfig,
    episodes: Sequence[int],
    device: torch.device,
    floor: float,
) -> LatentNormalizer:
    """Fit latent min/max over all train frames with vectorized episode windows."""
    tokenizer.eval()
    paths = sorted((Path(config.dataset.root).expanduser() / "data").rglob("*.parquet"))
    action_parts, episode_parts, frame_parts = [], [], []
    for path in paths:
        table = pq.read_table(
            path,
            columns=[config.dataset.action_key, "episode_index", "frame_index"],
        )
        action_parts.append(
            np.asarray(table[config.dataset.action_key].to_pylist(), dtype=np.float32)
        )
        episode_parts.append(np.asarray(table["episode_index"]).reshape(-1))
        frame_parts.append(np.asarray(table["frame_index"]).reshape(-1))
    actions = np.concatenate(action_parts)
    episode_indices = np.concatenate(episode_parts)
    frame_indices = np.concatenate(frame_parts)

    windows = []
    reactive = config.rdp if config.model.variant == "rdp_like" else config.rfm
    horizon = reactive.action_horizon
    for episode in episodes:
        selected = np.flatnonzero(episode_indices == episode)
        selected = selected[np.argsort(frame_indices[selected])]
        episode_action = torch.from_numpy(actions[selected])
        offsets = torch.arange(len(selected))[:, None] + torch.arange(horizon)[None, :]
        windows.append(episode_action[offsets.clamp_max(len(selected) - 1)])
    action_windows = torch.cat(windows)

    chunks = []
    batch_size = max(reactive.tokenizer_batch_size, 512)
    starts = range(0, len(action_windows), batch_size)
    for start in tqdm(starts, desc="latent statistics", leave=False):
        action = normalizer.normalize_action(
            action_windows[start : start + batch_size].to(device, non_blocking=True)
        )
        latent, _, _ = tokenizer.encode(action, sample=False)
        chunks.append(latent.cpu())
    return LatentNormalizer.from_latents(torch.cat(chunks), floor=floor)


def train(config: RealmanBeaverConfig) -> Path:
    config.validate()
    _seed_everything(config.training.seed)
    device = _resolve_device(config.training.device)
    output_dir = Path(config.training.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config.to_dict(), stream, sort_keys=False)

    if config.training.wandb_project:
        try:
            wandb.init(
                project=config.training.wandb_project,
                name=config.training.wandb_run_name or config.model.variant,
                dir=str(output_dir),
                config=config.to_dict(),
            )
            print(f"wandb: logging to {wandb.run.url}")
        except Exception as exc:  # W&B must never block training
            print(f"wandb: init failed, continuing without W&B: {exc}")

    train_episodes, val_episodes = episode_split(config.dataset)
    normalizer = ObservationNormalizer.from_lerobot_dataset(
        config, train_episodes
    ).to(device)
    print(
        f"variant={config.model.variant} device={device} "
        f"train_episodes={len(train_episodes)} val_episodes={len(val_episodes)}"
    )
    print(f"train_episode_indices={train_episodes}")
    print(f"val_episode_indices={val_episodes}")

    if config.model.variant in {"original_dp", "dp_beaver", "fm", "fm_beaver"}:
        train_dataset = RealmanPolicyDataset(config, train_episodes, stage="policy")
        val_dataset = (
            RealmanPolicyDataset(config, val_episodes, stage="policy")
            if val_episodes
            else None
        )
        train_loader = _make_loader(
            train_dataset,
            config.training.batch_size,
            config.training.num_workers,
            device,
            True,
            config.training.seed,
        )
        val_loader = (
            _make_loader(
                val_dataset,
                config.training.batch_size,
                config.training.num_workers,
                device,
                False,
                config.training.seed,
            )
            if val_dataset
            else None
        )
        policy = build_policy(config, normalizer).to(device)
        _train_stage(
            kind=config.model.variant,
            config=config,
            model=policy,
            loss_function=policy.compute_loss,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            output_dir=output_dir,
            epochs=config.training.epochs,
            max_steps=config.training.max_steps,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            metrics_name="metrics.jsonl",
            last_name="last.pt",
            resume_from=config.training.resume_from,
        )
        return output_dir / "last.pt"

    reactive = config.rdp if config.model.variant == "rdp_like" else config.rfm
    is_rfm = config.model.variant == "rfm"
    if config.training.resume_from:
        raise ValueError(
            "Reactive policy resume uses its tokenizer_checkpoint and latent_resume_from"
        )
    tokenizer_train = RealmanPolicyDataset(config, train_episodes, stage="tokenizer")
    tokenizer_val = (
        RealmanPolicyDataset(config, val_episodes, stage="tokenizer")
        if val_episodes
        else None
    )
    tokenizer_loader = _make_loader(
        tokenizer_train,
        reactive.tokenizer_batch_size,
        config.training.num_workers,
        device,
        True,
        config.training.seed,
    )
    tokenizer_val_loader = (
        _make_loader(
            tokenizer_val,
            reactive.tokenizer_batch_size,
            config.training.num_workers,
            device,
            False,
            config.training.seed,
        )
        if tokenizer_val
        else None
    )
    tokenizer = build_tokenizer(config).to(device)

    def tokenizer_loss(batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        action = normalizer.normalize_action(batch["action"])
        present = batch["beaver_present"]
        distance = normalizer.normalize_beaver(
            batch["beaver_distance"], present, batch["beaver_status"]
        )
        return tokenizer.compute_loss(
            action,
            batch["action_is_pad"],
            distance,
            present,
            reactive.kl_weight,
        )

    if reactive.tokenizer_checkpoint:
        _load_tokenizer_checkpoint(tokenizer, reactive.tokenizer_checkpoint, device)
    else:
        tokenizer_ema = _train_stage(
            kind="tokenizer",
            config=config,
            model=tokenizer,
            loss_function=tokenizer_loss,
            train_loader=tokenizer_loader,
            val_loader=tokenizer_val_loader,
            device=device,
            output_dir=output_dir,
            epochs=reactive.tokenizer_epochs,
            max_steps=config.training.max_steps or reactive.tokenizer_max_steps,
            learning_rate=reactive.tokenizer_learning_rate,
            weight_decay=reactive.tokenizer_weight_decay,
            metrics_name="tokenizer_metrics.jsonl",
            last_name="tokenizer_last.pt",
        )
        tokenizer_ema.copy_to(tokenizer)

    latent_normalizer = _fit_latent_normalizer(
        tokenizer,
        normalizer,
        config,
        train_episodes,
        device,
        config.dataset.normalization_floor,
    ).to(device)

    latent_train = RealmanPolicyDataset(config, train_episodes, stage="latent")
    latent_val = (
        RealmanPolicyDataset(config, val_episodes, stage="latent")
        if val_episodes
        else None
    )
    latent_loader = _make_loader(
        latent_train,
        reactive.latent_batch_size,
        config.training.num_workers,
        device,
        True,
        config.training.seed,
    )
    latent_val_loader = (
        _make_loader(
            latent_val,
            reactive.latent_batch_size,
            config.training.num_workers,
            device,
            False,
            config.training.seed,
        )
        if latent_val
        else None
    )
    policy_class = RFMPolicy if is_rfm else RDPPolicy
    policy = policy_class(
        config, normalizer, latent_normalizer, tokenizer=tokenizer
    ).to(device)
    policy.freeze_tokenizer()
    _train_stage(
        kind="latent_fm" if is_rfm else "latent_dp",
        config=config,
        model=policy,
        loss_function=policy.latent_loss,
        train_loader=latent_loader,
        val_loader=latent_val_loader,
        device=device,
        output_dir=output_dir,
        epochs=reactive.latent_epochs,
        max_steps=config.training.max_steps or reactive.latent_max_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        metrics_name="latent_fm_metrics.jsonl" if is_rfm else "latent_dp_metrics.jsonl",
        last_name="last.pt",
        resume_from=reactive.latent_resume_from,
    )
    return output_dir / "last.pt"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="Path to one of the six policy YAML files"
    )
    parser.add_argument(
        "--device", help="Override training.device (for example cuda:0 or cpu)"
    )
    parser.add_argument(
        "--dataset-root", help="Override dataset.root with a local LeRobot dataset"
    )
    parser.add_argument(
        "--dataset-repo-id", help="Override dataset.repo_id for the selected root"
    )
    parser.add_argument("--batch-size", type=int, help="Override all stage batch sizes")
    parser.add_argument("--num-workers", type=int, help="Override training.num_workers")
    parser.add_argument(
        "--max-steps", type=int, help="Bound every training stage for smoke tests"
    )
    parser.add_argument("--output-dir", help="Override training.output_dir")
    parser.add_argument(
        "--val-fraction", type=float, help="Override dataset.val_fraction"
    )
    parser.add_argument(
        "--val-episodes",
        type=_parse_episode_spec,
        help=(
            "Override validation episodes with zero-based LeRobot indices; "
            "accepts comma-separated indices and inclusive ranges such as 50-74"
        ),
    )
    parser.add_argument(
        "--tokenizer-checkpoint",
        help="Skip reactive tokenizer training and load this file",
    )
    parser.add_argument(
        "--latent-resume-from",
        help="Resume the reactive latent stage from a matching checkpoint",
    )
    parser.add_argument(
        "--latent-epochs",
        type=int,
        help="Override reactive latent_epochs (the new TOTAL when resuming)",
    )
    parser.add_argument(
        "--wandb-project", help="Enable W&B logging with this project name"
    )
    parser.add_argument(
        "--wandb-run-name", help="W&B run name (defaults to the model variant)"
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = load_config(args.config)
    if args.dataset_root is not None:
        config.dataset.root = args.dataset_root
    if args.dataset_repo_id is not None:
        config.dataset.repo_id = args.dataset_repo_id
    for argument, attribute in (
        (args.device, "device"),
        (args.num_workers, "num_workers"),
        (args.max_steps, "max_steps"),
        (args.output_dir, "output_dir"),
    ):
        if argument is not None:
            setattr(config.training, attribute, argument)
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
        config.rdp.tokenizer_batch_size = args.batch_size
        config.rdp.latent_batch_size = args.batch_size
        config.rfm.tokenizer_batch_size = args.batch_size
        config.rfm.latent_batch_size = args.batch_size
    if args.val_fraction is not None:
        config.dataset.val_fraction = args.val_fraction
    if args.val_episodes is not None:
        config.dataset.val_episodes = args.val_episodes
    if args.tokenizer_checkpoint is not None:
        reactive = config.rdp if config.model.variant == "rdp_like" else config.rfm
        reactive.tokenizer_checkpoint = args.tokenizer_checkpoint
    if args.latent_resume_from is not None:
        reactive = config.rdp if config.model.variant == "rdp_like" else config.rfm
        reactive.latent_resume_from = args.latent_resume_from
    if args.latent_epochs is not None:
        reactive = config.rdp if config.model.variant == "rdp_like" else config.rfm
        reactive.latent_epochs = args.latent_epochs
    if args.wandb_project is not None:
        config.training.wandb_project = args.wandb_project
    if args.wandb_run_name is not None:
        config.training.wandb_run_name = args.wandb_run_name
    config.validate()
    checkpoint = train(config)
    print(f"checkpoint={checkpoint}")


if __name__ == "__main__":
    main()
