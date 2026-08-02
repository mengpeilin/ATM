"""RoboTwin deployment adapter for ATM-conditioned Diffusion Policy."""

from __future__ import annotations

from collections import deque
import json
import os
import sys

import dill
import hydra
import numpy as np
import torch


LEFT_QUAT = slice(3, 7)
RIGHT_QUAT = slice(11, 15)


class ATMPolicyWrapper:
    def __init__(self, policy, stats, tokenizer, bert, n_obs_steps=3):
        self.policy = policy
        self.stats = stats
        self.tokenizer = tokenizer
        self.bert = bert
        self.n_obs_steps = n_obs_steps
        self.observations = deque(maxlen=n_obs_steps)
        self.task_emb = None
        self.observed_quats = None
        self.commanded_quats = None

    def set_instruction(self, instruction):
        tokens = self.tokenizer(
            [instruction], max_length=64, padding="max_length", truncation=True,
            return_attention_mask=True, return_tensors="pt",
        )
        with torch.no_grad():
            embedding = self.bert(tokens["input_ids"], tokens["attention_mask"])["pooler_output"]
        self.task_emb = embedding[0].cpu().numpy().astype(np.float32)

    def _continuous_observed_quaternions(self, state):
        quaternions = [state[LEFT_QUAT].copy(), state[RIGHT_QUAT].copy()]
        if self.observed_quats is None:
            quaternions = [quat if quat[0] >= 0 else -quat for quat in quaternions]
        else:
            quaternions = [
                quat if np.dot(quat, previous) >= 0 else -quat
                for quat, previous in zip(quaternions, self.observed_quats)
            ]
        state[LEFT_QUAT], state[RIGHT_QUAT] = quaternions
        self.observed_quats = quaternions
        return state

    def append_observation(self, observation):
        # RoboTwin supplies decoded RGB; preserve its channel order and native 240x320 resolution.
        head_cam = np.moveaxis(
            observation["observation"]["head_camera"]["rgb"], -1, 0
        ).astype(np.float32) / 255.0
        endpose = observation["endpose"]
        state = np.concatenate([
            np.asarray(endpose["left_endpose"], dtype=np.float64),
            np.asarray([endpose["left_gripper"]], dtype=np.float64),
            np.asarray(endpose["right_endpose"], dtype=np.float64),
            np.asarray([endpose["right_gripper"]], dtype=np.float64),
        ])
        state = self._continuous_observed_quaternions(state)
        state = 2.0 * (state - self.stats["state_min"]) / self.stats["state_range"] - 1.0
        self.observations.append({"head_cam": head_cam, "agent_pos": state.astype(np.float32)})

    def _stack_history(self, key):
        values = [observation[key] for observation in self.observations]
        return np.stack([values[0]] * (self.n_obs_steps - len(values)) + values)

    def predict_chunk(self):
        obs = {
            "head_cam": self._stack_history("head_cam"),
            "agent_pos": self._stack_history("agent_pos"),
            "track_task_emb": np.repeat(self.task_emb[None], self.n_obs_steps, axis=0),
        }
        device = self.policy.device
        tensor_obs = {
            key: torch.from_numpy(value).unsqueeze(0).to(device=device)
            for key, value in obs.items()
        }
        with torch.no_grad():
            result = self.policy.predict_action(tensor_obs)
        return np.clip(result["action"][0].cpu().numpy(), -1.0, 1.0)

    def to_env_action(self, normalized_action):
        action = (
            (normalized_action.astype(np.float64) + 1.0) / 2.0 * self.stats["action_range"]
            + self.stats["action_min"]
        )
        quaternions = [action[LEFT_QUAT].copy(), action[RIGHT_QUAT].copy()]
        normalized = []
        for idx, quat in enumerate(quaternions):
            norm = np.linalg.norm(quat)
            quat = quat / norm if norm > 1e-8 else np.array([1.0, 0.0, 0.0, 0.0])
            if self.commanded_quats is not None and np.dot(quat, self.commanded_quats[idx]) < 0:
                quat = -quat
            normalized.append(quat)
        action[LEFT_QUAT], action[RIGHT_QUAT] = normalized
        self.commanded_quats = normalized
        return action

    def reset(self):
        self.observations.clear()
        self.task_emb = None
        self.observed_quats = None
        self.commanded_quats = None


def get_model(usr_args):
    atm_root = os.path.abspath(usr_args["atm_root"])
    diffusion_root = os.path.join(atm_root, "diffusion_policy")
    for path in (atm_root, diffusion_root):
        if path not in sys.path:
            sys.path.insert(0, path)

    checkpoint_path = usr_args["ckpt_path"]
    with open(checkpoint_path, "rb") as stream:
        payload = torch.load(stream, pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace = workspace_cls(cfg, output_dir=os.path.dirname(os.path.dirname(checkpoint_path)))
    workspace.load_payload(payload)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to("cuda:0").eval()

    with open(usr_args["norm_stats_path"]) as stream:
        stats = {key: np.asarray(value) for key, value in json.load(stream).items()}

    from transformers import AutoModel, AutoTokenizer
    bert_cache = os.path.join(atm_root, "data", "bert")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", cache_dir=bert_cache)
    bert = AutoModel.from_pretrained("bert-base-cased", cache_dir=bert_cache).eval()
    return ATMPolicyWrapper(policy, stats, tokenizer, bert, n_obs_steps=cfg.n_obs_steps)


def eval(TASK_ENV, model, observation):
    if model.task_emb is None:
        model.set_instruction(TASK_ENV.get_instruction())
    model.append_observation(observation)
    for normalized_action in model.predict_chunk():
        TASK_ENV.take_action(model.to_env_action(normalized_action), action_type="ee")
        model.append_observation(TASK_ENV.get_obs())


def reset_model(model):
    model.reset()
