"""Package standalone WRM_phase_ddim deployable checkpoint.

Fuses:
1. WRM_wrap 50k Diffusion Policy backbone (switched to DDIM 15-step scheduler).
2. PhaseTaskMonitor trained weights.
3. Controller-level smoothing and per-tick micro-reaction.
"""

from pathlib import Path
import torch

BASE_CKPT = Path("policies/downloaded/WRM_wrap/checkpoints/WRM_wrap_step_050000.pt")
MONITOR_CKPT = Path("outputs/realman_beaver/phase_monitor/monitor.pt")
OUTPUT_DIR = Path("policies/downloaded/WRM_phase_ddim")


def package():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_dir = OUTPUT_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    base = torch.load(BASE_CKPT, map_location="cpu")
    monitor_data = torch.load(MONITOR_CKPT, map_location="cpu")

    raw_config = dict(base["config"])
    raw_model = dict(raw_config["model"])

    # Configure WRM_phase_ddim
    raw_model["variant"] = "WRM_phase_ddim"
    raw_model["noise_scheduler_type"] = "DDIM"
    raw_model["num_inference_steps"] = 15
    raw_model["n_action_steps"] = 4  # fast 4-step receding horizon replanning for smoothness
    raw_model["beaver_wrap_near_threshold_mm"] = 0.0
    raw_model["beaver_wrap_closing_scale_mm"] = 50.0
    raw_model["beaver_wrap_range_scale_mm"] = 300.0
    raw_model["beaver_wrap_lift_min_wrap"] = 0.25
    raw_model["beaver_wrap_stop_close_wrap"] = 0.5
    raw_model["beaver_wrap_contact_stop_mm"] = 0.0

    raw_config["model"] = raw_model

    # Merge weights
    combined_model = dict(base["model"])
    monitor_state = monitor_data["model"]
    for k, v in monitor_state.items():
        combined_model[f"monitor.{k}"] = v

    combined_ema = None
    if "ema" in base and base["ema"]:
        combined_ema = dict(base["ema"])
        for k, v in monitor_state.items():
            combined_ema[f"monitor.{k}"] = v

    checkpoint = {
        "kind": "WRM_phase_ddim",
        "variant": "WRM_phase_ddim",
        "epoch": int(base.get("epoch", 0)),
        "global_step": int(base.get("global_step", 50000)),
        "config": raw_config,
        "model": combined_model,
        "metrics": {
            "scheduler": "DDIM",
            "inference_steps": 15,
            "action_steps": 8,
            "monitor": "PhaseTaskMonitor",
        },
    }
    if combined_ema:
        checkpoint["ema"] = combined_ema

    target_path = ckpt_dir / "last.pt"
    torch.save(checkpoint, target_path)
    print(f"Successfully packaged {target_path} ({target_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    package()
