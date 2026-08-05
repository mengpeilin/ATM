from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from omegaconf import OmegaConf

import atm.model as atm_models
from atm.utils.flow_utils import sample_double_grid
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy


class RoboTwinATMDiffusionPolicy(DiffusionUnetImagePolicy):
    """RoboTwin image DP with frozen ATM future tracks appended to global conditioning."""

    def __init__(
        self,
        track_fn: str,
        track_view_key: str = "head_cam",
        track_task_key: str = "track_task_emb",
        diffusion_step_embed_dim=128,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        **kwargs,
    ):
        if not kwargs.get("obs_as_global_cond", True):
            raise ValueError("ATM track conditioning requires obs_as_global_cond=True")
        super().__init__(
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
            **kwargs,
        )

        track_cfg = OmegaConf.load(f"{track_fn}/config.yaml")
        track_cfg.model_cfg.load_path = f"{track_fn}/model_best.ckpt"
        track_cls = getattr(atm_models, track_cfg.model_name)
        self.track = track_cls(**track_cfg.model_cfg)
        self.track.eval()
        self.track.requires_grad_(False)

        if self.track.frame_stack != 1:
            raise ValueError(
                f"RoboTwin ATM-DP expects a one-frame Track Transformer, got {self.track.frame_stack}"
            )
        if self.track.num_track_ts != self.horizon or self.track.num_track_ids != 32:
            raise ValueError(
                "Track Transformer must predict exactly horizon x 32 tracks; "
                f"got {self.track.num_track_ts} x {self.track.num_track_ids} for horizon {self.horizon}"
            )
        self.track_fn = track_fn
        self.track_view_key = track_view_key
        self.track_task_key = track_task_key
        self.track_dim = self.track.num_track_ts * self.track.num_track_ids * 2

        # The base policy sizes its U-Net for visual/proprioceptive features only. Replace just the
        # U-Net with the same architecture and an enlarged global condition for future tracks.
        self.model = ConditionalUnet1D(
            input_dim=self.action_dim,
            local_cond_dim=None,
            global_cond_dim=(self.obs_feature_dim + self.track_dim) * self.n_obs_steps,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

    def _split_observations(self, obs_dict: Dict[str, torch.Tensor]):
        if self.track_task_key not in obs_dict:
            raise KeyError(f"missing Track Transformer condition {self.track_task_key!r}")
        task_emb = obs_dict[self.track_task_key]
        dp_obs = {key: value for key, value in obs_dict.items() if key != self.track_task_key}
        return dp_obs, task_emb

    def _predict_tracks(self, rgb: torch.Tensor, task_emb: torch.Tensor) -> torch.Tensor:
        batch_size, obs_steps = rgb.shape[:2]
        rgb = rgb[:, :self.n_obs_steps]
        task_emb = task_emb[:, :self.n_obs_steps]
        if tuple(rgb.shape[-2:]) != tuple(self.track.img_size):
            raise ValueError(
                f"Track Transformer expects native RGB at {self.track.img_size}, "
                f"got {tuple(rgb.shape[-2:])}"
            )
        track_rgb = rearrange(rgb, "b t c h w -> (b t) c h w")
        # DP observations are [0,1], while TrackTransformer.reconstruct owns the /255 conversion.
        track_rgb = rearrange(track_rgb * 255.0, "bt c h w -> bt 1 c h w")
        flat_task_emb = rearrange(task_emb, "b t e -> (b t) e")

        grid = sample_double_grid(4, device=track_rgb.device, dtype=track_rgb.dtype)
        grid = repeat(
            grid,
            "n d -> bt tl n d",
            bt=batch_size * self.n_obs_steps,
            tl=self.track.num_track_ts,
        )
        with torch.no_grad():
            tracks, _ = self.track.reconstruct(track_rgb, grid, flat_task_emb, p_img=0)
        return rearrange(
            tracks,
            "(b t) tl n d -> b t tl n d",
            b=batch_size,
            t=self.n_obs_steps,
        )

    def _global_condition(self, obs_dict: Dict[str, torch.Tensor]):
        dp_obs, task_emb = self._split_observations(obs_dict)
        nobs = self.normalizer.normalize(dp_obs)
        batch_size = next(iter(nobs.values())).shape[0]
        encoder_input = dict_apply(
            nobs,
            lambda value: value[:, :self.n_obs_steps].reshape(-1, *value.shape[2:]),
        )
        obs_features = self.obs_encoder(encoder_input).reshape(batch_size, self.n_obs_steps, -1)
        tracks = self._predict_tracks(dp_obs[self.track_view_key], task_emb)
        track_features = tracks.reshape(batch_size, self.n_obs_steps, -1).to(obs_features.dtype)
        global_cond = torch.cat([obs_features, track_features], dim=-1).reshape(batch_size, -1)
        return global_cond, tracks

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert "past_action" not in obs_dict
        global_cond, tracks = self._global_condition(obs_dict)
        batch_size = global_cond.shape[0]
        condition_data = torch.zeros(
            (batch_size, self.horizon, self.action_dim), device=self.device, dtype=self.dtype
        )
        condition_mask = torch.zeros_like(condition_data, dtype=torch.bool)
        sample = self.conditional_sample(
            condition_data, condition_mask, global_cond=global_cond, **self.kwargs
        )
        action_pred = self.normalizer["action"].unnormalize(sample[..., :self.action_dim])
        start = self.n_obs_steps - 1
        action = action_pred[:, start:start + self.n_action_steps]
        return {"action": action, "action_pred": action_pred, "predicted_tracks": tracks}

    def compute_loss(self, batch):
        global_cond, _ = self._global_condition(batch["obs"])
        actions = self.normalizer["action"].normalize(batch["action"])
        noise = torch.randn_like(actions)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (actions.shape[0],),
            device=actions.device,
        ).long()
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
        prediction = self.model(noisy_actions, timesteps, global_cond=global_cond)
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "sample":
            target = actions
        else:
            raise ValueError(
                f"unsupported prediction type {self.noise_scheduler.config.prediction_type!r}"
            )
        return reduce(F.mse_loss(prediction, target, reduction="none"), "b ... -> b", "mean").mean()

    def train(self, mode: bool = True):
        super().train(mode)
        self.track.eval()
        return self
