"""Roll a trained TrackTransformer across a preprocessed episode and render its predicted 16-step
track horizon, either against simulator ground truth or exactly as the real BCViLTPolicy queries it.

At every timestep t the model is shown only the current frame plus each point's position at t (every
other timestep in the track window is masked internally by ``TrackTransformer._mask_track_as_first``)
and asked to predict where all tracked points go over the next ``num_track_ts`` (16) steps. This
renders that receding-horizon prediction as a video: one output frame per t.

Every simulator-GT point stored for the episode (1,098) is drawn in green every frame, as a dense
backdrop of true motion. Two seeding modes (``--sampling``) control what's drawn in red on top:

- ``grid`` (default): matches ``BCViLTPolicy.track_encode`` (atm/policy/vilt.py) exactly -- the query
  points are ``sample_double_grid(4)``, a fixed set of 32 synthetic (u, v) grid coordinates, identical
  every frame/episode/task. This is what the deployed policy actually conditions on; a synthetic grid
  point has no corresponding real track, so it only exists in red.
- ``gt``: seeds with simulator-GT point ids sampled once from the episode (visible in frame 0), so
  that particular red prediction can be checked against its own green ground truth directly.

Usage::

    python -m scripts.visualize_track_prediction \
        --task place_object_stand --episode 0

    python -m scripts.visualize_track_prediction --sampling gt \
        --task place_object_stand --episode 0
"""
import os
from pathlib import Path

import click
import cv2
import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from atm.model import TrackTransformer
from atm.utils.flow_utils import sample_double_grid

GT_COLOR = (0, 255, 0)     # green, RGB
PRED_COLOR = (255, 0, 0)   # red, RGB


def load_episode(save_root, task_config, task, episode, view):
    path = os.path.join(save_root, task_config, task, f"episode_{episode}.hdf5")
    with h5py.File(path, "r") as f:
        video = np.array(f[f"root/{view}/video"][0])     # (T, 3, H, W) uint8, RGB
        tracks = np.array(f[f"root/{view}/tracks"][0])   # (T, N, 2) in [0, 1], (x, y)
        vis = np.array(f[f"root/{view}/vis"][0])          # (T, N) bool
        task_emb = np.array(f["root/task_emb_bert"])      # (768,)
    return video, tracks, vis, task_emb


def find_track_run(results_root, task):
    """Return the newest completed TrackTransformer run trained for ``task``."""
    candidates = []
    for config_path in Path(results_root).glob("*/config.yaml"):
        checkpoint_path = config_path.parent / "model_best.ckpt"
        if not checkpoint_path.is_file():
            continue
        cfg = OmegaConf.load(config_path)
        dataset_paths = [str(path) for key in ("train_dataset", "val_dataset")
                         for path in cfg.get(key, [])]
        if any(task in Path(path).parts for path in dataset_paths):
            candidates.append((checkpoint_path.stat().st_mtime, config_path.parent))
    if not candidates:
        raise click.ClickException(
            f"No completed TrackTransformer run for task '{task}' under {results_root}"
        )
    return str(max(candidates, key=lambda item: item[0])[1])


def load_model(track_fn, checkpoint, device):
    cfg = OmegaConf.load(os.path.join(track_fn, "config.yaml"))
    checkpoint_path = checkpoint if os.path.isabs(checkpoint) else os.path.join(track_fn, checkpoint)
    if not os.path.isfile(checkpoint_path):
        raise click.ClickException(f"Checkpoint does not exist: {checkpoint_path}")
    cfg.model_cfg.load_path = checkpoint_path
    model = TrackTransformer(**cfg.model_cfg)
    model.eval().to(device)
    return model, cfg, checkpoint_path


def sample_point_ids(vis, num_points, seed):
    """Fix a set of point ids visible at frame 0, so the video tracks the same physical points
    throughout instead of resampling a fresh set at every window like training does."""
    visible = np.where(vis[0])[0]
    rng = np.random.default_rng(seed)
    return rng.choice(visible, size=min(num_points, len(visible)), replace=False)


def visible_runs(vis_col):
    """Indices of vis_col (bool, shape (T,)) split into maximal consecutive True runs, so a polyline
    drawn through each run never crosses an occluded (unreliable) frame."""
    idx = np.where(vis_col)[0]
    if len(idx) < 2:
        return []
    breaks = np.where(np.diff(idx) > 1)[0] + 1
    return [run for run in np.split(idx, breaks) if len(run) > 1]


def pad_window(arr, start, length):
    """Slice arr[start:start+length] along axis 0, repeating the last frame past the episode end --
    mirrors BaseDataset.process_demo's padding so windows near the end still span 16 steps."""
    window = arr[start:start + length]
    if len(window) < length:
        pad = np.repeat(arr[-1:], length - len(window), axis=0)
        window = np.concatenate([window, pad], axis=0)
    return window


@click.command()
@click.option("--track-fn", default=None,
              help="TrackTransformer run directory. By default, select the newest completed run trained for --task.")
@click.option("--results-root", default="./results/track_transformer", show_default=True,
              help="Run root searched when --track-fn is omitted.")
@click.option("--checkpoint", default="model_best.ckpt", show_default=True,
              help="Checkpoint filename inside --track-fn, or an absolute checkpoint path.")
@click.option("--task", default="place_object_stand", show_default=True)
@click.option("--task-config", default="demo_clean")
@click.option("--episode", default=0, type=int)
@click.option("--view", default="head_camera")
@click.option("--save-root", default="./data/atm_robotwin", help="Where preprocessing wrote the episodes.")
@click.option("--out", default="./tt_vis", help="Output directory.")
@click.option("--scale", default=4, type=int, help="Integer upscale applied before drawing.")
@click.option("--sampling", type=click.Choice(["grid", "gt"]), default="grid",
              help="'grid' reproduces BCViLTPolicy.track_encode's fixed 32-point double grid "
                   "(no ground truth exists for it, so only the prediction is drawn); "
                   "'gt' seeds with simulator-GT points so the prediction can be checked against them.")
@click.option("--num-points", default=-1, type=int,
              help="Only used with --sampling gt; -1 uses the model's num_track_ids.")
@click.option("--seed", default=0, type=int, help="Only used with --sampling gt; selects which points get tracked.")
@click.option("--max-gt-points", default=-1, type=int,
              help="Cap on how many of the ~1000+ stored GT points to draw in green; -1 draws all of them.")
@click.option("--stride", default=1, type=int, help="Steps between rolled-out starting frames.")
@click.option("--fps", default=10, type=int)
@click.option("--device", default=None,
              help="Torch device, for example cuda:1. Defaults to CUDA when available, otherwise CPU.")
def main(track_fn, results_root, checkpoint, task, task_config, episode, view, save_root, out, scale,
         sampling, num_points, seed, max_gt_points, stride, fps, device):
    os.makedirs(out, exist_ok=True)
    if track_fn is None:
        track_fn = find_track_run(results_root, task)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model, cfg, checkpoint_path = load_model(track_fn, checkpoint, device)
    click.echo(f"run: {track_fn}")
    click.echo(f"checkpoint: {checkpoint_path}")
    num_track_ts = cfg.num_track_ts

    video, tracks, vis, task_emb = load_episode(save_root, task_config, task, episode, view)
    T, _, H, W = video.shape
    Hs, Ws = H * scale, W * scale

    gt_points = np.arange(tracks.shape[1]) if max_gt_points < 0 else np.arange(min(max_gt_points, tracks.shape[1]))
    gt_tracks = tracks[:, gt_points]  # (T, n_gt, 2), all stored simulator-GT tracks for this episode
    gt_vis = vis[:, gt_points]        # (T, n_gt) -- occluded points' coordinates are unreliable and
                                       # jump erratically (mean step displacement ~2x, max ~7x that of
                                       # visible points on this dataset), so segments touching an
                                       # occluded endpoint are skipped, same as visualize_dataset_tracks.py

    if sampling == "grid":
        # exactly atm/policy/vilt.py:255 -- a fixed 4x4-plus-4x4 grid of 32 (u, v) points in [0, 1],
        # identical every frame/episode/task. There is no ground-truth future for a synthetic point.
        grid_points = sample_double_grid(4, device="cpu", dtype=torch.float32).numpy()  # (32, 2)
        n = len(grid_points)
        seed_window = np.repeat(grid_points[None], num_track_ts, axis=0)  # (16, n, 2), constant over t
    else:
        if num_points < 0:
            num_points = cfg.num_track_ids
        point_ids = sample_point_ids(vis, num_points, seed)
        n = len(point_ids)
        sub_tracks = tracks[:, point_ids]  # (T, n, 2)

    task_emb_t = torch.from_numpy(task_emb)[None].float().to(device)

    path = os.path.join(out, f"{task}_ep{episode}_{view}_trackpred_{sampling}.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (Ws, Hs))
    radius = max(1, scale // 2)

    with torch.no_grad():
        for t0 in tqdm(range(0, T, stride), desc=f"episode {episode}"):
            frame = video[t0].transpose(1, 2, 0)  # h w c, RGB

            gt_window = pad_window(gt_tracks, t0, num_track_ts)      # (16, n_gt, 2)
            gt_vis_window = pad_window(gt_vis, t0, num_track_ts)     # (16, n_gt) bool
            seed_in = pad_window(sub_tracks, t0, num_track_ts) if sampling == "gt" else seed_window  # (16, n, 2)
            vid_in = torch.from_numpy(frame.copy()).float().permute(2, 0, 1)[None, None].to(device)  # (1,1,3,H,W)
            track_in = torch.from_numpy(seed_in[None]).float().to(device)  # (1,16,n,2)

            rec_track, _ = model.reconstruct(vid_in, track_in, task_emb_t, p_img=0.)
            pred_window = rec_track[0].cpu().numpy()  # (16, n, 2)

            canvas = cv2.resize(frame, (Ws, Hs), interpolation=cv2.INTER_NEAREST)
            canvas = np.ascontiguousarray(canvas)

            gt_px = np.stack([gt_window[..., 0] * Ws, gt_window[..., 1] * Hs], axis=-1).astype(np.int32)
            pred_px = np.stack([pred_window[..., 0] * Ws, pred_window[..., 1] * Hs], axis=-1).astype(np.int32)

            # one batched call draws every GT point's visible run(s) of the 16-step polyline in green;
            # Segments spanning an invisible frame are skipped even though the simulator retains
            # their exact finite projection (see gt_vis comment above).
            gt_polylines = [
                gt_px[run[0]:run[-1] + 1, i, None, :]
                for i in range(gt_px.shape[1])
                for run in visible_runs(gt_vis_window[:, i])
            ]
            if gt_polylines:
                cv2.polylines(canvas, gt_polylines, isClosed=False, color=GT_COLOR, thickness=1, lineType=cv2.LINE_AA)

            for i in range(n):
                for s in range(num_track_ts - 1):
                    cv2.line(canvas, tuple(pred_px[s, i]), tuple(pred_px[s + 1, i]), PRED_COLOR, 1, cv2.LINE_AA)
                cv2.circle(canvas, tuple(pred_px[0, i]), radius, PRED_COLOR, -1, cv2.LINE_AA)

            cv2.putText(canvas, f"t={t0:3d}/{T - 1}  GT=green ({gt_px.shape[1]} pts)  pred=red ({n} pts, {sampling})"
                                 f"  horizon={num_track_ts}",
                        (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
            # the pipeline is RGB end to end; cv2.VideoWriter expects BGR, so swap only at this boundary
            writer.write(canvas[..., ::-1])

    writer.release()
    print(f"wrote {path}  ({T} frames rolled at stride {stride}, {gt_px.shape[1]} GT pts, {n} pred pts, "
          f"{Ws}x{Hs}, sampling={sampling})")


if __name__ == "__main__":
    main()
