from __future__ import annotations

import copy
from glob import glob
from pathlib import Path
from typing import Dict, Sequence

import h5py
import numpy as np
import torch

from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


class RoboTwinATMImageDataset(BaseImageDataset):
    """Sequence dataset over ATM's per-episode RoboTwin HDF5 files.

    The HDF5 labels are already normalized using training-only statistics.  DP therefore uses
    identity normalization for state/action while retaining its standard image normalizer.
    Only ``n_obs_steps`` native-resolution frames are read for each sample; the full action horizon
    is returned.
    """

    def __init__(
        self,
        train_dataset_dirs: str | Sequence[str],
        val_dataset_dirs: str | Sequence[str],
        horizon: int = 16,
        n_obs_steps: int = 3,
        pad_before: int = 2,
        pad_after: int = 15,
        split: str = "train",
        view: str = "head_camera",
    ):
        super().__init__()
        if split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        if horizon < n_obs_steps:
            raise ValueError("horizon must be at least n_obs_steps")

        train_dirs = _as_dir_list(train_dataset_dirs)
        val_dirs = _as_dir_list(val_dataset_dirs)
        dirs = train_dirs if split == "train" else val_dirs
        episode_paths = sorted(
            (path for directory in dirs for path in glob(str(Path(directory) / "*.hdf5"))),
            key=_natural_episode_key,
        )
        if not episode_paths:
            raise ValueError(f"no HDF5 episodes found in {dirs}")

        self.train_dataset_dirs = train_dirs
        self.val_dataset_dirs = val_dirs
        self.episode_paths = episode_paths
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.pad_before = min(max(pad_before, 0), horizon - 1)
        self.pad_after = min(max(pad_after, 0), horizon - 1)
        self.split = split
        self.view = view

        self._samples: list[tuple[int, int]] = []
        self._episode_lengths: list[int] = []
        for episode_idx, path in enumerate(self.episode_paths):
            with h5py.File(path, "r") as handle:
                root = handle["root"]
                if f"{view}/video" not in root:
                    raise KeyError(
                        f"{path} has no root/{view}/video; rerun RoboTwin preprocessing"
                    )
                if tuple(root[view]["video"].shape[-2:]) != (240, 320):
                    raise ValueError(
                        f"{path} is not native 240x320 RoboTwin data; rerun preprocessing"
                    )
                length = int(root["actions"].shape[-2])
            self._episode_lengths.append(length)
            min_start = -self.pad_before
            max_start = length - self.horizon + self.pad_after
            self._samples.extend((episode_idx, start) for start in range(min_start, max_start + 1))

    def get_validation_dataset(self) -> "RoboTwinATMImageDataset":
        result = copy.copy(self)
        result.__init__(
            train_dataset_dirs=self.train_dataset_dirs,
            val_dataset_dirs=self.val_dataset_dirs,
            horizon=self.horizon,
            n_obs_steps=self.n_obs_steps,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            split="val",
            view=self.view,
        )
        return result

    def get_normalizer(self, **_: object) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer["head_cam"] = get_image_range_normalizer()
        normalizer["agent_pos"] = SingleFieldLinearNormalizer.create_identity()
        normalizer["action"] = SingleFieldLinearNormalizer.create_identity()
        return normalizer

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        episode_idx, start = self._samples[index]
        length = self._episode_lengths[episode_idx]
        obs_indices = np.clip(np.arange(start, start + self.n_obs_steps), 0, length - 1)
        action_indices = np.clip(np.arange(start, start + self.horizon), 0, length - 1)

        with h5py.File(self.episode_paths[episode_idx], "r") as handle:
            root = handle["root"]
            video = root[self.view]["video"]
            head_cam = _take_video_with_repeats(video, obs_indices).astype(np.float32) / 255.0
            left = _take_with_repeats(root["extra_states"]["left_arm_states"], obs_indices)
            right = _take_with_repeats(root["extra_states"]["right_arm_states"], obs_indices)
            actions = _take_with_repeats(root["actions"], action_indices)
            task_emb = np.asarray(root["task_emb_bert"], dtype=np.float32)

        agent_pos = np.concatenate([left, right], axis=-1).astype(np.float32)
        task_emb = np.repeat(task_emb[None], self.n_obs_steps, axis=0)
        return {
            "obs": {
                "head_cam": torch.from_numpy(head_cam),
                "agent_pos": torch.from_numpy(agent_pos),
                "track_task_emb": torch.from_numpy(task_emb),
            },
            "action": torch.from_numpy(actions.astype(np.float32)),
        }


def _natural_episode_key(path: str) -> tuple[str, int]:
    stem = Path(path).stem
    prefix, _, suffix = stem.rpartition("_")
    return prefix, int(suffix) if suffix.isdigit() else -1


def _as_dir_list(dataset_dirs: str | Sequence[str]) -> list[str]:
    return [dataset_dirs] if isinstance(dataset_dirs, str) else list(dataset_dirs)


def _take_with_repeats(dataset: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
    """Read only unique HDF5 rows, then restore repeated padding indices in memory."""
    unique_indices, inverse = np.unique(indices, return_inverse=True)
    return np.asarray(dataset[unique_indices])[inverse]


def _take_video_with_repeats(dataset: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
    unique_indices, inverse = np.unique(indices, return_inverse=True)
    return np.asarray(dataset[0, unique_indices])[inverse]
