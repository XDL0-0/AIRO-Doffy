"""Hardware-free policy registry shared by eval_policy and unit tests."""

from __future__ import annotations

SUPPORTED_POLICY_VARIANTS = frozenset(
    {
        "original_dp",
        "dp_beaver",
        "dp_beaver_closure",
        "dp_beaver_enc",
        "dp_beaver_near",
        "dp_beaver_near_gate",
        "dp_beaver_key4",
        "dp_beaver_key4_pca",
        "WRM_temporal",
        "WRM_delta",
        "WRM_adaptive",
        "WRM_antigravity",
        "WRM_grok",
        "WRM_codex",
        "WRM_claude",
        "WRM_qwen",
        "WRM_wrap",
        "WRM_wrap_delta",
        "WRM_wrap_monitor",
        "WRM_wrap_monitor_backup",
        "WRM_lobo_monitor",
        "WRM_phase_ddim",
        "rdp_like",
        "fm",
        "fm_beaver",
        "rfm",
    }
)
BEAVER_POLICY_VARIANTS = frozenset(
    {
        "dp_beaver",
        "dp_beaver_closure",
        "dp_beaver_enc",
        "dp_beaver_near",
        "dp_beaver_near_gate",
        "dp_beaver_key4",
        "dp_beaver_key4_pca",
        "WRM_temporal",
        "WRM_delta",
        "WRM_adaptive",
        "WRM_antigravity",
        "WRM_grok",
        "WRM_codex",
        "WRM_claude",
        "WRM_qwen",
        "WRM_wrap",
        "WRM_wrap_delta",
        "WRM_wrap_monitor",
        "WRM_wrap_monitor_backup",
        "WRM_lobo_monitor",
        "WRM_phase_ddim",
        "rdp_like",
        "fm_beaver",
        "rfm",
    }
)
EXPECTED_CHECKPOINT_KINDS = {
    "original_dp": "original_dp",
    "dp_beaver": "dp_beaver",
    "dp_beaver_closure": "dp_beaver_closure",
    "dp_beaver_enc": "dp_beaver_enc",
    "dp_beaver_near": "dp_beaver_near",
    "dp_beaver_near_gate": "dp_beaver_near_gate",
    "dp_beaver_key4": "dp_beaver_key4",
    "dp_beaver_key4_pca": "dp_beaver_key4_pca",
    "WRM_temporal": "WRM_temporal",
    "WRM_delta": "WRM_delta",
    "WRM_adaptive": "WRM_adaptive",
    "WRM_antigravity": "WRM_antigravity",
    "WRM_grok": "WRM_grok",
    "WRM_codex": "WRM_codex",
    "WRM_claude": "WRM_claude",
    "WRM_qwen": "WRM_qwen",
    "WRM_wrap": "WRM_wrap",
    "WRM_wrap_delta": "WRM_wrap_delta",
    "WRM_wrap_monitor": "WRM_wrap_monitor",
    "WRM_wrap_monitor_backup": "WRM_wrap_monitor_backup",
    "WRM_lobo_monitor": "WRM_lobo_monitor",
    "WRM_phase_ddim": "WRM_phase_ddim",
    "rdp_like": "latent_dp",
    "fm": "fm",
    "fm_beaver": "fm_beaver",
    "rfm": "latent_fm",
}


def policy_needs_beaver(variant: str) -> bool:
    """Return whether a deployment observation must contain Beaver fields."""
    if variant not in SUPPORTED_POLICY_VARIANTS:
        raise ValueError(f"Unsupported policy variant: {variant}")
    return variant in BEAVER_POLICY_VARIANTS


def policy_step_window(policy) -> tuple[int, int]:
    """Return the configured (predicted steps, executed steps) pair."""
    variant = policy.config.model.variant
    if variant == "rdp_like":
        return (
            int(policy.config.rdp.action_horizon),
            int(policy.config.rdp.slow_replan_steps),
        )
    if variant == "rfm":
        return (
            int(policy.config.rfm.action_horizon),
            int(policy.config.rfm.slow_replan_steps),
        )
    return (
        int(policy.config.model.horizon),
        int(policy.config.model.n_action_steps),
    )


def validate_deployable_checkpoint(summary: dict[str, object]) -> str:
    """Validate checkpoint metadata and return its policy variant."""
    kind = str(summary.get("kind", "unknown"))
    if kind == "tokenizer":
        raise ValueError(
            "tokenizer-only checkpoint is not deployable; use the final "
            "reactive-policy last.pt"
        )
    variant = str(summary.get("variant", "unknown"))
    if variant not in SUPPORTED_POLICY_VARIANTS:
        raise ValueError(f"unsupported policy variant '{variant}'")
    expected_kind = EXPECTED_CHECKPOINT_KINDS[variant]
    if kind != expected_kind:
        raise ValueError(
            f"checkpoint kind '{kind}' does not match variant '{variant}' "
            f"(expected '{expected_kind}')"
        )
    return variant
