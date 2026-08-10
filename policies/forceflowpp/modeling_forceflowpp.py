"""LeRobot-style ForceFlow++ policy implementation."""

from __future__ import annotations

from collections import deque

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from lerobot.configs.types import FeatureType
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_IMAGES

from policies.forceflowpp.configuration_forceflowpp import ForceFlowPPConfig
from policies.forceflowpp.contact_labeling import force_sequence_to_phase
from policies.forceflowpp.modules.adaln_dit import ForceFlowPPDiT
from policies.forceflowpp.modules.encoders import ForceFlowPPObservationEncoder
from policies.forceflowpp.modules.flow_matching import ForceFlowMatchingObjective
from policies.forceflowpp.prior_library import ContactPriorLibrary, uniform_prior_weights


class ForceFlowPPPolicy(PreTrainedPolicy):
    """ForceFlow++: force/tactile-conditioned adaptive-prior flow matching."""

    config_class = ForceFlowPPConfig
    name = "forceflowpp"

    def __init__(self, config: ForceFlowPPConfig, **kwargs) -> None:
        super().__init__(config)
        config.validate_features()
        self.config = config
        self._queues = None

        self.observation_encoder = ForceFlowPPObservationEncoder(config)
        self.prior_selector = nn.Sequential(
            nn.Linear(config.force_tactile_feature_dim, 256),
            nn.GELU(),
            nn.Linear(256, config.num_contact_modes),
        )
        self.prior_library = ContactPriorLibrary(
            num_modes=config.num_contact_modes,
            horizon=config.horizon,
            action_dim=config.action_dim,
            std_init=config.prior_std_init,
            std_floor=config.prior_std_floor,
        )
        self.noise_predictor = ForceFlowPPDiT(config, self.observation_encoder.conditioning_dim)
        self.objective = ForceFlowMatchingObjective(config)
        self.reset()

    def get_optim_params(self) -> dict:
        return {"params": self.parameters()}

    def reset(self) -> None:
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}
        for key, feature in (self.config.input_features or {}).items():
            if feature.type is FeatureType.VISUAL:
                continue
            self._queues[key] = deque(maxlen=self.config.n_obs_steps)
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if not self.config.image_features:
            return batch
        batch = dict(batch)
        if OBS_IMAGES not in batch:
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        return batch

    def _prior_logits_and_weights(self, encoded: dict[str, Tensor | None]) -> tuple[Tensor, Tensor]:
        ft_feat = encoded["ft_feat"]
        if ft_feat is None:
            raise ValueError("ForceFlow++ encoder did not return ft_feat.")
        logits = self.prior_selector(ft_feat)
        if self.config.use_adaptive_prior:
            weights = torch.softmax(logits, dim=-1)
        else:
            weights = uniform_prior_weights(
                batch_size=logits.shape[0],
                num_modes=self.config.num_contact_modes,
                device=logits.device,
                dtype=logits.dtype,
            )
        return logits, weights

    def _sample_prior(self, weights: Tensor, like: Tensor | None = None) -> Tensor:
        if self.config.use_adaptive_prior:
            return self.prior_library.sample(weights, like=like)
        if like is not None:
            return torch.randn_like(like)
        batch_size = weights.shape[0]
        return torch.randn(
            batch_size,
            self.config.horizon,
            self.config.action_dim,
            device=weights.device,
            dtype=weights.dtype,
        )

    def _prior_aux_loss(self, logits: Tensor, encoded: dict[str, Tensor | None]) -> tuple[Tensor, dict[str, float]]:
        zero = logits.sum() * 0.0
        if not self.config.use_adaptive_prior or self.config.prior_aux_loss_weight <= 0:
            return zero, {"prior_ce": 0.0, "contact_phase_acc": 0.0}
        labels = force_sequence_to_phase(
            encoded.get("force_sequence"),
            free_threshold_n=self.config.free_force_threshold_n,
            heavy_threshold_n=self.config.heavy_force_threshold_n,
            transition_delta_n=self.config.transition_force_delta_n,
            num_modes=self.config.num_contact_modes,
        )
        if labels is None:
            return zero, {"prior_ce": 0.0, "contact_phase_acc": 0.0}

        ce = F.cross_entropy(logits, labels)
        pred = torch.argmax(logits, dim=-1)
        acc = (pred == labels).float().mean()
        return ce, {
            "prior_ce": float(ce.detach().cpu()),
            "contact_phase_acc": float(acc.detach().cpu()),
        }

    def _generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        encoded = self.observation_encoder(batch)
        _logits, weights = self._prior_logits_and_weights(encoded)
        z0 = self._sample_prior(weights)
        actions = self.objective.conditional_sample(self.noise_predictor, encoded, z0)
        start = self.config.n_obs_steps - 1
        end = start + self.config.n_action_steps
        return actions[:, start:end]

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        batch = {
            key: torch.stack(list(self._queues[key]), dim=1)
            for key in batch
            if key in self._queues and key != ACTION
        }
        return self._generate_actions(batch)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        if ACTION in batch:
            batch = dict(batch)
            batch.pop(ACTION)
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch)
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1))
        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        batch = self._prepare_batch(batch)
        encoded = self.observation_encoder(batch)
        logits, weights = self._prior_logits_and_weights(encoded)
        action_like = self.objective._action_chunk(batch[ACTION])
        z0 = self._sample_prior(weights, like=action_like)

        fm_loss, stats = self.objective.compute_loss(self.noise_predictor, batch, encoded, z0)
        prior_ce, prior_stats = self._prior_aux_loss(logits, encoded)
        kl = self.prior_library.kl_to_standard_normal() if self.config.use_adaptive_prior else fm_loss * 0.0
        loss = (
            fm_loss
            + self.config.prior_aux_loss_weight * prior_ce
            + self.config.prior_kl_weight * kl
        )

        stats.update(prior_stats)
        stats["prior_kl"] = float(kl.detach().cpu())
        stats["loss"] = float(loss.detach().cpu())
        return loss, stats
