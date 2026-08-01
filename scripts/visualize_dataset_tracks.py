"""Render every CoTracker point of a preprocessed episode as an overlay video.

The tracks stored by ``scripts/preprocess_robotwin`` are ~1098 points per episode (98 grid points
followed by 1000 randomly sampled ones), held as in-picture coordinates in [0, 1]. Because they are
normalized they can be drawn on any resolution, so by default this renders on the *original*
RoboTwin frames rather than the 128x128 crops the model consumes - the scene stays legible and it
doubles as a check that the tracks really are aligned with the source data.

Points are coloured by their initial height in the frame and carry a short motion tail. Points that
CoTracker reports as occluded are drawn hollow, so you can watch visibility drop out.

Usage::

    python -m scripts.visualize_dataset_tracks --task place_dual_shoes --episode 0
    python -m scripts.visualize_dataset_tracks --task place_dual_shoes --episode 0 \
        --source preprocessed --scale 4          # exactly what the model sees, upscaled
"""
import os

import click
import cv2
import h5py
import numpy as np
from matplotlib import colormaps
from tqdm import tqdm

RAW_ROOT = "/data/peilin/Bimanual_Manipulation/RoboTwin/data"


def load_frames(source, raw_root, save_root, task, task_config, episode, view):
    """Return (T, H, W, 3) uint8 RGB frames.

    RoboTwin stores frames as JPEGs produced by ``cv2.imencode`` on an RGB array, so ``cv2.imdecode``
    returns that same RGB array. Nothing here converts colour space.
    """
    if source == "raw":
        path = os.path.join(raw_root, task, task_config, "data", f"episode{episode}.hdf5")
        with h5py.File(path, "r") as f:
            raw = f[f"observation/{view}/rgb"]
            return np.stack([cv2.imdecode(np.frombuffer(raw[t], np.uint8), cv2.IMREAD_COLOR)
                             for t in range(raw.shape[0])])
    path = os.path.join(save_root, task_config, task, f"episode_{episode}.hdf5")
    with h5py.File(path, "r") as f:
        return f[f"root/{view}/video"][0].transpose(0, 2, 3, 1)  # t c h w -> t h w c


def load_tracks(save_root, task, task_config, episode, view):
    path = os.path.join(save_root, task_config, task, f"episode_{episode}.hdf5")
    with h5py.File(path, "r") as f:
        return (np.array(f[f"root/{view}/tracks"][0]),   # (T, N, 2) in [0, 1], (x, y)
                np.array(f[f"root/{view}/vis"][0]))      # (T, N) bool


def point_colors(first_xy, cmap_name="gist_rainbow"):
    """One RGB colour per point, keyed on its initial height so depth layers separate."""
    cmap = colormaps[cmap_name]
    order = np.clip(first_xy[:, 1], 0, 1)
    return (np.array([cmap(v)[:3] for v in order]) * 255).astype(np.uint8)


@click.command()
@click.option("--task", default="place_dual_shoes")
@click.option("--task-config", default="demo_clean")
@click.option("--episode", default=0, type=int)
@click.option("--view", default="head_camera")
@click.option("--save-root", default="./data/atm_robotwin", help="Where preprocessing wrote the episodes.")
@click.option("--raw-root", default=RAW_ROOT, help="Root of the raw RoboTwin dataset.")
@click.option("--source", type=click.Choice(["raw", "preprocessed"]), default="raw",
              help="'raw' draws on the original RoboTwin frames; 'preprocessed' on the 128x128 crops.")
@click.option("--out", default="./tt_vis", help="Output directory.")
@click.option("--scale", default=2, type=int, help="Integer upscale applied before drawing.")
@click.option("--tail", default=8, type=int, help="Frames of motion history drawn behind each point.")
@click.option("--max-points", default=-1, type=int, help="-1 draws every tracked point.")
@click.option("--fps", default=20, type=int)
def main(task, task_config, episode, view, save_root, raw_root, source, out, scale, tail,
         max_points, fps):
    os.makedirs(out, exist_ok=True)

    frames = load_frames(source, raw_root, save_root, task, task_config, episode, view)
    tracks, vis = load_tracks(save_root, task, task_config, episode, view)

    # preprocessing drops the final frame (obs are [0, T-1), actions are shifted by one)
    T = min(len(frames), len(tracks))
    frames, tracks, vis = frames[:T], tracks[:T], vis[:T]
    if max_points > 0:
        tracks, vis = tracks[:, :max_points], vis[:, :max_points]
    n_points = tracks.shape[1]

    h, w = frames.shape[1:3]
    H, W = h * scale, w * scale
    colors = point_colors(tracks[0])
    px = np.stack([tracks[..., 0] * W, tracks[..., 1] * H], axis=-1).astype(np.int32)  # (T, N, 2)

    path = os.path.join(out, f"{task}_ep{episode}_{view}_{source}_alltracks.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    radius = max(1, scale // 2)

    for t in tqdm(range(T), desc=f"episode {episode}"):
        canvas = cv2.resize(frames[t], (W, H), interpolation=cv2.INTER_NEAREST)
        canvas = np.ascontiguousarray(canvas)

        # motion tails first, so the current points draw on top
        for k in range(max(0, t - tail), t):
            alpha = 0.25 + 0.5 * (k - max(0, t - tail)) / max(tail, 1)
            layer = canvas.copy()
            for i in range(n_points):
                if not vis[k, i]:
                    continue
                cv2.line(layer, tuple(px[k, i]), tuple(px[k + 1, i]),
                         tuple(int(c) for c in colors[i]), 1, cv2.LINE_AA)
            canvas = cv2.addWeighted(layer, alpha, canvas, 1 - alpha, 0)

        for i in range(n_points):
            c = tuple(int(v) for v in colors[i])
            # occluded points stay visible but hollow, so drop-outs are readable
            cv2.circle(canvas, tuple(px[t, i]), radius, c, -1 if vis[t, i] else 1, cv2.LINE_AA)

        cv2.putText(canvas, f"t={t:3d}/{T-1}  {n_points} pts  vis={vis[t].mean()*100:4.1f}%",
                    (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        # the pipeline is RGB end to end; cv2.VideoWriter expects BGR, so swap only at this boundary
        writer.write(canvas[..., ::-1])

    writer.release()
    print(f"wrote {path}  ({T} frames, {n_points} points, {W}x{H})")


if __name__ == "__main__":
    main()
