"""Combine a trained Beaver monitor with the frozen WRM_wrap DP checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import torch

from policies.realman_beaver.configuration import (
    WRAP_MONITOR_BEAVER_VARIANTS,
    RealmanBeaverConfig,
)
from policies.realman_beaver.dataset import ObservationNormalizer
from policies.realman_beaver.modeling import build_policy


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package(
    base_path: Path, monitor_path: Path, output_path: Path, variant: str
) -> None:
    if variant not in WRAP_MONITOR_BEAVER_VARIANTS:
        raise ValueError(f"Unsupported monitor variant: {variant}")
    base = torch.load(base_path, map_location="cpu", weights_only=True, mmap=True)
    if base.get("kind") != "WRM_wrap":
        raise ValueError(f"Expected WRM_wrap base checkpoint, got {base.get('kind')}")
    trained = torch.load(monitor_path, map_location="cpu", weights_only=True)
    if variant == "WRM_lobo_monitor":
        if trained.get("kind") not in ("instant_contact", "beaver_monitor"):
            raise ValueError(
                f"Monitor checkpoint is {trained.get('kind')}, expected instant_contact or beaver_monitor"
            )
    elif trained.get("kind") != "beaver_monitor" or trained.get("variant") != variant:
        raise ValueError(
            f"Monitor checkpoint is {trained.get('kind')}/{trained.get('variant')}, "
            f"expected beaver_monitor/{variant}"
        )
    raw_config = base["config"]
    raw_model = raw_config["model"]
    # The selected 50k base predates the explicit closing-scale field. Preserve
    # its learned 50 mm normalization while embedding the last rollout's
    # near=0 deployment representation in both parameter-free monitor policies.
    updated_model = {
        **raw_model,
        "variant": variant,
        "beaver_wrap_near_threshold_mm": 0.0,
        "beaver_wrap_closing_scale_mm": float(
            raw_model.get(
                "beaver_wrap_closing_scale_mm",
                raw_model.get("beaver_wrap_near_threshold_mm", 50.0),
            )
        ),
        "beaver_wrap_range_scale_mm": 300.0,
        # Parent WRM_wrap gating consults these values at execution.
        # Matched to the successful 20260902T142746 eval parameters.
        "beaver_wrap_lift_min_wrap": 0.25,
        "beaver_wrap_stop_close_wrap": 0.5,
        "beaver_wrap_contact_stop_mm": 0.0,
    }
    if variant == "WRM_lobo_monitor":
        updated_model["beaver_monitor_hidden_dims"] = tuple(
            trained.get("metadata", {}).get("hidden_dims", (64, 32))
        )
        updated_model["beaver_monitor_lag_steps"] = (0,)
    raw_config = {
        **raw_config,
        "model": updated_model,
    }
    config = RealmanBeaverConfig.from_dict(raw_config)
    config.validate()
    policy = build_policy(
        config,
        ObservationNormalizer.identity(config.model.state_dim, config.model.action_dim),
    )
    incompatible = policy.load_state_dict(base["model"], strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing_non_monitor = [
        name for name in incompatible.missing_keys if not name.startswith("monitor.")
    ]
    if unexpected or missing_non_monitor:
        raise ValueError(
            f"Base checkpoint mismatch: unexpected={unexpected}, "
            f"missing_non_monitor={missing_non_monitor}"
        )
    parameters = dict(policy.named_parameters())
    for name, value in (base.get("ema") or {}).items():
        if name in parameters:
            parameters[name].data.copy_(value)
    policy.monitor.load_state_dict(trained["model"], strict=True)
    policy.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(
        {
            "kind": variant,
            "config": config.to_dict(),
            "model": policy.state_dict(),
            "ema": {},
            "global_step": int(base.get("global_step", 0)),
            "metrics": trained.get("metadata", {}),
            "provenance": {
                "base_checkpoint": str(base_path),
                "base_sha256": _sha256(base_path),
                "base_weights": "EMA" if base.get("ema") else "raw",
                "monitor_checkpoint": str(monitor_path),
                "monitor_sha256": _sha256(monitor_path),
                "frozen_dp": True,
                "execution_gate": "trained_beaver_monitor",
            },
        },
        temporary,
    )
    os.replace(temporary, output_path)
    print(
        f"packaged={output_path} variant={variant} "
        f"bytes={output_path.stat().st_size}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--monitor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=sorted(WRAP_MONITOR_BEAVER_VARIANTS), required=True)
    args = parser.parse_args()
    package(args.base, args.monitor, args.output, args.variant)


if __name__ == "__main__":
    main()
