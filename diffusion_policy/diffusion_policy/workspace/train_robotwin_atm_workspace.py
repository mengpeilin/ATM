from __future__ import annotations

import copy
import math
import os
import random

import hydra
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.workspace.base_workspace import BaseWorkspace


class TrainRoboTwinATMWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model = hydra.utils.instantiate(cfg.policy)
        self.ema_model = copy.deepcopy(self.model) if cfg.training.use_ema else None
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=trainable)
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        latest_path = self.get_checkpoint_path()
        if cfg.training.resume and latest_path.is_file():
            self.load_checkpoint(path=latest_path)

        dataset = hydra.utils.instantiate(cfg.task.dataset)
        if not isinstance(dataset, BaseImageDataset):
            raise TypeError(f"expected BaseImageDataset, got {type(dataset)}")
        val_dataset = dataset.get_validation_dataset()
        train_loader = DataLoader(dataset, **cfg.dataloader)
        val_loader = DataLoader(val_dataset, **cfg.val_dataloader)

        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=math.ceil(
                len(train_loader) / cfg.training.gradient_accumulate_every
            ) * cfg.training.num_epochs,
            last_epoch=self.global_step - 1,
        )
        ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model) \
            if self.ema_model is not None else None

        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 2
            cfg.training.max_val_steps = 2
            cfg.training.checkpoint_every = 1

        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as logger:
            while self.epoch < cfg.training.num_epochs:
                self.model.train()
                train_losses = []
                self.optimizer.zero_grad(set_to_none=True)
                num_train_batches = len(train_loader)
                if cfg.training.max_train_steps is not None:
                    num_train_batches = min(num_train_batches, cfg.training.max_train_steps)
                for batch_idx, batch in enumerate(tqdm.tqdm(train_loader, desc=f"train {self.epoch}")):
                    batch = dict_apply(batch, lambda value: value.to(device, non_blocking=True))
                    raw_loss = self.model.compute_loss(batch)
                    group_start = (
                        batch_idx // cfg.training.gradient_accumulate_every
                    ) * cfg.training.gradient_accumulate_every
                    group_size = min(
                        cfg.training.gradient_accumulate_every,
                        num_train_batches - group_start,
                    )
                    should_step = batch_idx + 1 == group_start + group_size
                    (raw_loss / group_size).backward()
                    if should_step:
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        lr_scheduler.step()
                        if ema is not None:
                            ema.step(self.model)
                    train_losses.append(raw_loss.item())
                    self.global_step += 1
                    if batch_idx + 1 >= num_train_batches:
                        break

                metrics = {
                    "epoch": self.epoch,
                    "global_step": self.global_step,
                    "train_loss": float(np.mean(train_losses)),
                    "lr": lr_scheduler.get_last_lr()[0],
                }

                eval_policy = self.ema_model if self.ema_model is not None else self.model
                if self.epoch % cfg.training.val_every == 0:
                    eval_policy.eval()
                    val_losses = []
                    with torch.no_grad():
                        for batch_idx, batch in enumerate(tqdm.tqdm(val_loader, desc=f"val {self.epoch}")):
                            batch = dict_apply(batch, lambda value: value.to(device, non_blocking=True))
                            val_losses.append(eval_policy.compute_loss(batch).item())
                            if cfg.training.max_val_steps is not None \
                                    and batch_idx + 1 >= cfg.training.max_val_steps:
                                break
                    metrics["val_loss"] = float(np.mean(val_losses))
                self.epoch += 1
                logger.log(metrics)
                if self.epoch % cfg.training.checkpoint_every == 0:
                    self.save_checkpoint(
                        path=os.path.join(self.output_dir, "checkpoints", f"epoch_{self.epoch}.ckpt"),
                        use_thread=False,
                    )
                    self.save_checkpoint(path=latest_path, use_thread=False)

        # Deployment always consumes the completed final epoch, independent of validation loss.
        self.save_checkpoint(
            path=os.path.join(self.output_dir, "checkpoints", "last.ckpt"),
            use_thread=False,
        )
        self.save_checkpoint(path=latest_path, use_thread=False)


@hydra.main(version_base=None, config_path="../config", config_name="train_robotwin_atm_workspace")
def main(cfg: OmegaConf):
    TrainRoboTwinATMWorkspace(cfg).run()


if __name__ == "__main__":
    main()
