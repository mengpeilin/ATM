"""Convert raw RoboTwin demonstrations into the per-demo HDF5 format that ATM's dataloaders expect.

The RoboTwin episodes store JPEG-encoded RGB frames, absolute end-effector poses and joint targets.
ATM wants, per demonstration, one HDF5 file laid out as::

    root/
    ├── actions        (T, 16)   normalized absolute bimanual end-effector commands
    ├── task_emb_bert  (768,)
    ├── extra_states/{left_arm_states (T, 8), right_arm_states (T, 8)}
    └── <view>/{video (1, T, 3, 240, 320), tracks (1, T, N, 2), vis (1, T, N)}

Actions are the next step's absolute ``[xyz, wxyz quaternion, gripper]`` command for each arm.

Usage::

    python -m scripts.preprocess_robotwin --task place_bread_skillet --num-episodes 2
"""
import json
import hashlib
import os
import sys
from glob import glob

import click
import cv2
import h5py
import numpy as np
import torch
from einops import rearrange
from natsort import natsorted
from tqdm import tqdm
from easydict import EasyDict

from scripts.preprocess_libero import track_through_video, save_images, get_task_embs

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

COTRACKER_ROOT = "/data/peilin/Bimanual_Manipulation/co-tracker"
COTRACKER_CKPT = os.path.join(COTRACKER_ROOT, "checkpoints/scaled_offline.pth")

# [left_endpose(7: xyz + wxyz quat), left_gripper, right_endpose(7), right_gripper]
EE_VECTOR_DIM = 16
NATIVE_IMAGE_SIZE = (240, 320)
LEFT_POSE_SLICE = slice(0, 7)
LEFT_GRIPPER_IDX = 7
RIGHT_POSE_SLICE = slice(8, 15)
RIGHT_GRIPPER_IDX = 15
# transforms3d's mat2quat returns (w, x, y, z), so the scalar part sits at offset 3 of each pose.
QUAT_SLICES = (slice(3, 7), slice(11, 15))


def build_cotracker():
    """CoTracker3 offline predictor from the local checkout (the torch.hub cache is not populated)."""
    if COTRACKER_ROOT not in sys.path:
        sys.path.insert(0, COTRACKER_ROOT)
    from cotracker.predictor import CoTrackerPredictor

    tracker = CoTrackerPredictor(checkpoint=None, window_len=60, v2=False)
    tracker.model.load_state_dict(torch.load(COTRACKER_CKPT, map_location="cpu"))
    return tracker.eval().cuda()


def make_quaternions_continuous(vec):
    """Remove quaternion double-cover sign flips in place, so the regression target is continuous.

    `q` and `-q` denote the same rotation, so raw recordings can flip sign between adjacent frames
    and inject huge artificial jumps into the action targets. Canonicalize the first frame to a
    non-negative scalar part, then keep every later frame on the same hemisphere as its predecessor.
    """
    for quat_slice in QUAT_SLICES:
        quats = vec[:, quat_slice]
        if quats[0, 0] < 0:
            quats[0] *= -1
        for t in range(1, len(quats)):
            if np.dot(quats[t], quats[t - 1]) < 0:
                quats[t] *= -1
        vec[:, quat_slice] = quats
    return vec


def load_ee_vector(episode_h5):
    """Read the (T, 16) end-effector trajectory from a raw RoboTwin episode."""
    grp = episode_h5["endpose"]
    vec = np.concatenate([
        np.array(grp["left_endpose"], dtype=np.float64),
        np.array(grp["left_gripper"], dtype=np.float64)[:, None],
        np.array(grp["right_endpose"], dtype=np.float64),
        np.array(grp["right_gripper"], dtype=np.float64)[:, None],
    ], axis=-1)
    assert vec.shape[-1] == EE_VECTOR_DIM, vec.shape
    return make_quaternions_continuous(vec)


def normalize(vec, stats):
    """Map raw values onto [-1, 1] with per-dimension min/max statistics."""
    return 2.0 * (vec - stats["min"]) / stats["range"] - 1.0


def norm_stats_signature(stats):
    """Stable identity used to reject episodes normalized with different statistics."""
    digest = hashlib.sha256()
    for key in ("state_min", "state_max", "state_range", "action_min", "action_max", "action_range"):
        digest.update(np.asarray(stats[key], dtype=np.float64).tobytes())
    return digest.hexdigest()


def compute_norm_stats(episode_paths):
    state_lo, state_hi = np.full(EE_VECTOR_DIM, np.inf), np.full(EE_VECTOR_DIM, -np.inf)
    action_lo, action_hi = np.full(EE_VECTOR_DIM, np.inf), np.full(EE_VECTOR_DIM, -np.inf)
    for path in tqdm(episode_paths, desc="norm stats"):
        with h5py.File(path, "r") as f:
            vec = load_ee_vector(f)
        state, action = vec[:-1], vec[1:]
        state_lo, state_hi = np.minimum(state_lo, state.min(axis=0)), np.maximum(state_hi, state.max(axis=0))
        action_lo, action_hi = np.minimum(action_lo, action.min(axis=0)), np.maximum(action_hi, action.max(axis=0))
    return {
        "state_min": state_lo, "state_max": state_hi, "state_range": np.maximum(state_hi-state_lo, 1e-6),
        "action_min": action_lo, "action_max": action_hi, "action_range": np.maximum(action_hi-action_lo, 1e-6),
    }


def load_or_compute_norm_stats(stats_path, episode_paths, train_ratio):
    """Load statistics for the deterministic training split, or regenerate stale/legacy stats."""
    num_train = int(len(episode_paths) * train_ratio)
    if num_train <= 0:
        raise ValueError(f"train split is empty: {len(episode_paths)} episodes, train_ratio={train_ratio}")

    if os.path.exists(stats_path):
        try:
            with open(stats_path) as f:
                saved = json.load(f)
            if (saved.get("num_total_episodes") == len(episode_paths)
                    and saved.get("num_train_episodes") == num_train
                    and saved.get("train_ratio") == train_ratio
                    and saved.get("representation") == "absolute_ee_16"):
                print(f"loaded existing training-split normalization statistics from {stats_path}")
                keys = ("state_min", "state_max", "state_range", "action_min", "action_max", "action_range")
                return {k: np.asarray(saved[k]) for k in keys}
        except (OSError, ValueError, KeyError, TypeError):
            pass
        print(f"regenerating legacy or stale normalization statistics at {stats_path}")

    # split_libero_dataset uses the first floor(N * ratio) naturally sorted episodes for training.
    stats = compute_norm_stats(episode_paths[:num_train])
    payload = {
        **{k: v.tolist() for k, v in stats.items()},
        "num_total_episodes": len(episode_paths),
        "num_train_episodes": num_train,
        "train_ratio": train_ratio,
        "representation": "absolute_ee_16",
    }
    # Atomic replacement prevents parallel preprocessing workers from observing partial JSON.
    tmp_path = f"{stats_path}.{os.getpid()}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, stats_path)
    print(f"wrote training-split normalization statistics to {stats_path}")
    return stats


def decode_video(episode_h5, view):
    """Decode a camera's JPEG frames to native-resolution (T, 3, H, W) uint8.

    RoboTwin stores frames via `cv2.imencode` on an RGB array, so `cv2.imdecode` hands the very same
    RGB array back. No colour-space conversion is performed anywhere in this pipeline.
    """
    raw = episode_h5[f"observation/{view}/rgb"]
    frames = []
    for t in range(raw.shape[0]):
        img = cv2.imdecode(np.frombuffer(raw[t], np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"failed to decode frame {t} of view {view}")
        frames.append(img)
    video = np.stack(frames)  # t h w c
    return rearrange(video, "t h w c -> t c h w")


def get_instruction_embeddings(root, task, task_config, num_episodes_hint):
    """BERT embedding of each episode's first `seen` instruction, deduplicated across episodes."""
    instruction_dir = os.path.join(root, task, task_config, "instructions")
    per_episode = {}
    for path in natsorted(glob(os.path.join(instruction_dir, "episode*.json"))):
        episode_idx = int(os.path.basename(path).split(".")[0].replace("episode", ""))
        with open(path) as f:
            per_episode[episode_idx] = json.load(f)["seen"][0]

    unique = sorted(set(per_episode.values()))
    cfg = EasyDict({
        "task_embedding_format": "bert",
        "task_embedding_one_hot_offset": 1,
        "data": {"max_word_len": 64},  # RoboTwin instructions are sentences, not LIBERO's short names
        "policy": {"language_encoder": {"network_kwargs": {"input_size": 768}}},
    })  # mirrors the hardcoded config ATM uses for LIBERO task embeddings
    embs = get_task_embs(cfg, unique).cpu().numpy()
    text_to_emb = {text: embs[i] for i, text in enumerate(unique)}
    return {ep: text_to_emb[text] for ep, text in per_episode.items()}, per_episode


def is_complete(path, views, stats_signature):
    """True only if the file exists and is a readable HDF5 holding every expected dataset.

    Existence alone is not enough: a run interrupted mid-write leaves a truncated or zero-byte file
    that would otherwise be skipped forever and only surface as a corrupt read during training.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        with h5py.File(path, "r") as f:
            root = f["root"]
            if root.attrs.get("norm_stats_signature") != stats_signature:
                return False
            for key in ("actions", "task_emb_bert", "extra_states"):
                if key not in root:
                    return False
            for view in views:
                if view not in root:
                    return False
                for key in ("video", "tracks", "vis"):
                    if key not in root[view]:
                        return False
                if tuple(root[view]["video"].shape[-2:]) != NATIVE_IMAGE_SIZE:
                    return False
    except (OSError, KeyError):
        return False
    return True


def convert_episode(src_path, dst_path, image_dir, views, stats, task_emb,
                    tracker, num_points):
    with h5py.File(src_path, "r") as src:
        vec = load_ee_vector(src)
        videos = {view: decode_video(src, view) for view in views}
    for view, video in videos.items():
        if tuple(video.shape[-2:]) != NATIVE_IMAGE_SIZE:
            raise ValueError(
                f"expected native D435 RGB at {NATIVE_IMAGE_SIZE}, got {tuple(video.shape[-2:])} "
                f"for {view} in {src_path}"
            )

    # Observation t predicts the absolute end-effector target recorded at t+1.
    obs_vec, action_vec = vec[:-1], vec[1:]
    length = len(obs_vec)

    norm_obs = normalize(obs_vec, {"min": stats["state_min"], "range": stats["state_range"]})
    action_stats = {"min": stats["action_min"], "range": stats["action_range"]}
    norm_actions = normalize(action_vec, action_stats)

    with h5py.File(dst_path, "w") as dst:
        root = dst.create_group("root")
        root.attrs["norm_stats_signature"] = norm_stats_signature(stats)
        root.create_dataset("actions", data=norm_actions.astype(np.float32))
        root.create_dataset("task_emb_bert", data=task_emb)

        extra = root.create_group("extra_states")
        extra.create_dataset("left_arm_states", data=norm_obs[:, :8].astype(np.float32))
        extra.create_dataset("right_arm_states", data=norm_obs[:, 8:].astype(np.float32))

        for view in views:
            video = videos[view][:length]  # t c h w
            _, _, H, W = video.shape

            pred_tracks, pred_vis = track_through_video(video, tracker, num_points=num_points)

            # [1, T, N, 2] -> normalized in-picture coordinates
            pred_tracks[:, :, :, 0] /= W
            pred_tracks[:, :, :, 1] /= H

            view_grp = root.create_group(view)
            view_grp.create_dataset("video", data=video[None].astype(np.uint8))
            view_grp.create_dataset("tracks", data=pred_tracks.cpu().numpy())
            view_grp.create_dataset("vis", data=pred_vis.cpu().numpy())

            # ATMPretrainDataset always drops the cached video, so stage-1 training reads these PNGs
            save_images(rearrange(video, "t c h w -> t h w c"), image_dir, view)


def relabel_episode(src_path, dst_path, stats):
    """Update only state/action targets while preserving existing videos and CoTracker outputs."""
    with h5py.File(src_path, "r") as src:
        vec = load_ee_vector(src)
    obs_vec = vec[:-1]
    action_vec = vec[1:]
    norm_obs = normalize(obs_vec, {"min": stats["state_min"], "range": stats["state_range"]})
    action_stats = {"min": stats["action_min"], "range": stats["action_range"]}
    norm_actions = normalize(action_vec, action_stats)

    with h5py.File(dst_path, "r+") as dst:
        root = dst["root"]
        for key in ("actions", "action_pad"):
            if key in root:
                del root[key]
        root.create_dataset("actions", data=norm_actions.astype(np.float32))
        extra = root["extra_states"]
        for key in list(extra.keys()):
            if key in extra:
                del extra[key]
        extra.create_dataset("left_arm_states", data=norm_obs[:, :8].astype(np.float32))
        extra.create_dataset("right_arm_states", data=norm_obs[:, 8:].astype(np.float32))
        root.attrs["norm_stats_signature"] = norm_stats_signature(stats)


@click.command()
@click.option("--root", type=str, default="/data/peilin/Bimanual_Manipulation/RoboTwin/data",
              help="Root directory of the raw RoboTwin dataset.")
@click.option("--save", type=str, default="./data/atm_robotwin", help="Target directory.")
@click.option("--task", type=str, default="place_bread_skillet")
@click.option("--task-config", type=str, default="demo_clean")
@click.option("--views", type=str, default="head_camera",
              help="Comma-separated camera names to convert.")
@click.option("--num-points", type=int, default=1000, help="Random points sampled for tracking.")
@click.option("--start-episode", type=int, default=0)
@click.option("--num-episodes", type=int, default=-1, help="-1 converts every episode.")
@click.option("--skip-exist", type=bool, default=False)
@click.option("--train-ratio", type=float, default=0.9,
              help="Deterministic leading fraction used to compute normalization statistics.")
@click.option("--stats-only", is_flag=True, help="Compute normalization statistics and exit.")
@click.option("--labels-only", is_flag=True,
              help="Rewrite state/action labels in existing files without rerunning CoTracker.")
def main(root, save, task, task_config, views, num_points, start_episode, num_episodes,
         skip_exist, train_ratio, stats_only, labels_only):
    views = [v.strip() for v in views.split(",") if v.strip()]

    src_dir = os.path.join(root, task, task_config, "data")
    episode_paths = natsorted(glob(os.path.join(src_dir, "episode*.hdf5")))
    if not episode_paths:
        raise ValueError(f"no episodes found under {src_dir}")

    # Suite/task nesting mirrors LIBERO so that scripts.split_libero_dataset works unchanged.
    target_dir = os.path.join(save, task_config, task)
    os.makedirs(target_dir, exist_ok=True)

    stats_path = os.path.join(target_dir, "norm_stats.json")
    stats = load_or_compute_norm_stats(stats_path, episode_paths, train_ratio)
    if stats_only:
        return

    if labels_only:
        episode_to_emb, episode_to_text = None, None
    else:
        episode_to_emb, episode_to_text = get_instruction_embeddings(root, task, task_config, len(episode_paths))

    selected = episode_paths[start_episode:] if num_episodes < 0 \
        else episode_paths[start_episode:start_episode + num_episodes]
    print(f"converting {len(selected)} episodes of {task}/{task_config}, views={views}")

    tracker = None if labels_only else build_cotracker()

    with torch.no_grad():
        for idx, src_path in enumerate(tqdm(selected, desc="episodes")):
            episode_idx = int(os.path.basename(src_path).split(".")[0].replace("episode", ""))
            dst_path = os.path.join(target_dir, f"episode_{episode_idx}.hdf5")
            image_dir = os.path.join(target_dir, "images", f"episode_{episode_idx}")

            if labels_only:
                if not os.path.exists(dst_path):
                    raise FileNotFoundError(f"cannot relabel missing processed episode: {dst_path}")
                relabel_episode(src_path, dst_path, stats)
                continue

            if skip_exist and is_complete(dst_path, views, norm_stats_signature(stats)):
                continue

            convert_episode(src_path, dst_path, image_dir, views, stats,
                            episode_to_emb[episode_idx], tracker, num_points)

    if episode_to_text:
        print(f"done. instruction for episode {list(episode_to_text)[0]}: {list(episode_to_text.values())[0]}")
    else:
        print("done relabeling state/action targets")


if __name__ == "__main__":
    main()
