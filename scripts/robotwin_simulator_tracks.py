"""Simulator-exact 2D point tracks for ATM's RoboTwin preprocessing.

Queries use CoTracker's ``[query_frame, x, y]`` convention in native image
pixels.  A query is attached once to a rigid simulator entity at its own
query frame.  Its local 3D coordinate is then transformed and projected at
every frame, including frames before the query and frames where the point is
occluded, offscreen, or behind the camera.  Visibility is a separate boolean
mask and never controls trajectory retention after attachment.
"""

from __future__ import annotations

import importlib
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

import numpy as np
import torch
import yaml


SIMULATOR_GT_SOURCE = "robotwin_simulator_gt"
SIMULATOR_GT_VERSION = 1
NUM_RANDOM_QUERIES = 1000
NUM_GRID_QUERIES = 98
RANDOM_VARIANCE_THRESHOLD = 20.0
GRID_VARIANCE_THRESHOLD = 0.0
JITTER_STD_HEIGHT_FRACTION = 0.05
DEFAULT_DEPTH_TOLERANCE_METERS = 0.01

ROBOTWIN_ROOT = Path(__file__).resolve().parents[2] / "RoboTwin"


class QueryTracker(Protocol):
    """Minimal interface used by the ATM point-selection implementation."""

    height: int
    width: int
    num_frames: int

    def track_queries(self, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return pixel tracks, visibility, and query-attachment validity."""

    def can_attach(self, query: np.ndarray) -> bool:
        """Return whether one ``[frame, x, y]`` query attaches to geometry."""


@dataclass(frozen=True)
class AttachedQueries:
    """Rigid attachment state for a batch of pixel queries."""

    valid: np.ndarray
    owner_ids: np.ndarray
    local_points: np.ndarray


def _as_per_frame_camera_array(array: np.ndarray, num_frames: int, trailing_shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(array)
    if array.shape == trailing_shape:
        return np.broadcast_to(array, (num_frames,) + trailing_shape).copy()
    expected = (num_frames,) + trailing_shape
    if array.shape != expected:
        raise ValueError(f"expected camera array shape {trailing_shape} or {expected}, got {array.shape}")
    return array


class SimulatorTrackingOracle:
    """Attach query pixels to replayed simulator entities and project them.

    Args:
        depth_maps: ``(T, H, W)`` depth in millimetres.
        segmentations: ``(T, H, W)`` SAPIEN ``per_scene_id`` labels.
        intrinsics: ``(T, 3, 3)`` or fixed ``(3, 3)`` OpenCV intrinsics.
        extrinsics: ``(T, 3, 4)`` or fixed ``(3, 4)`` world-to-camera matrices.
        owner_transforms: entity id to ``(T, 4, 4)`` local-to-world poses.

    Coordinates are always finite.  They are intentionally not clipped to the
    image because offscreen coordinates participate in ATM's variance filter.
    """

    def __init__(
        self,
        depth_maps: np.ndarray,
        segmentations: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        owner_transforms: Mapping[int, np.ndarray],
        depth_scale: float = 1000.0,
        depth_tolerance_meters: float = DEFAULT_DEPTH_TOLERANCE_METERS,
    ) -> None:
        self.depth_maps = np.asarray(depth_maps, dtype=np.float64)
        self.segmentations = np.asarray(segmentations)
        if self.depth_maps.ndim != 3:
            raise ValueError(f"depth_maps must be (T,H,W), got {self.depth_maps.shape}")
        if self.segmentations.shape != self.depth_maps.shape:
            raise ValueError(
                f"segmentation shape {self.segmentations.shape} does not match depth {self.depth_maps.shape}"
            )
        self.num_frames, self.height, self.width = self.depth_maps.shape
        self.intrinsics = _as_per_frame_camera_array(np.asarray(intrinsics), self.num_frames, (3, 3))
        self.extrinsics = _as_per_frame_camera_array(np.asarray(extrinsics), self.num_frames, (3, 4))
        self.owner_transforms = {
            int(owner_id): np.asarray(transforms, dtype=np.float64)
            for owner_id, transforms in owner_transforms.items()
        }
        for owner_id, transforms in self.owner_transforms.items():
            if transforms.shape != (self.num_frames, 4, 4):
                raise ValueError(
                    f"owner {owner_id} has transforms {transforms.shape}, expected "
                    f"({self.num_frames},4,4)"
                )
        self.depth_scale = float(depth_scale)
        self.depth_tolerance_meters = float(depth_tolerance_meters)

    def _query_pixel_indices(self, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        frames = queries[:, 0].astype(np.int64)
        xs = np.rint(queries[:, 1]).astype(np.int64)
        ys = np.rint(queries[:, 2]).astype(np.int64)
        return frames, xs, ys

    def attach_queries(self, queries: np.ndarray) -> AttachedQueries:
        queries = np.asarray(queries, dtype=np.float64)
        if queries.ndim != 2 or queries.shape[1] != 3:
            raise ValueError(f"queries must be (N,3) [frame,x,y], got {queries.shape}")

        count = len(queries)
        valid = np.zeros(count, dtype=bool)
        owner_ids = np.full(count, -1, dtype=np.int64)
        local_points = np.zeros((count, 3), dtype=np.float64)
        if count == 0:
            return AttachedQueries(valid, owner_ids, local_points)

        frames, xs, ys = self._query_pixel_indices(queries)
        in_bounds = (
            (frames >= 0)
            & (frames < self.num_frames)
            & (queries[:, 1] >= 0.0)
            & (queries[:, 1] < self.width)
            & (queries[:, 2] >= 0.0)
            & (queries[:, 2] < self.height)
            & (xs >= 0)
            & (xs < self.width)
            & (ys >= 0)
            & (ys < self.height)
        )
        candidate_indices = np.flatnonzero(in_bounds)
        for index in candidate_indices:
            frame, x_index, y_index = frames[index], xs[index], ys[index]
            depth_meters = self.depth_maps[frame, y_index, x_index] / self.depth_scale
            owner_id = int(self.segmentations[frame, y_index, x_index])
            if not np.isfinite(depth_meters) or depth_meters <= 0.0 or owner_id not in self.owner_transforms:
                continue

            intrinsic = self.intrinsics[frame]
            x, y = queries[index, 1:]
            camera_point = np.array(
                [
                    (x - intrinsic[0, 2]) * depth_meters / intrinsic[0, 0],
                    (y - intrinsic[1, 2]) * depth_meters / intrinsic[1, 1],
                    depth_meters,
                ],
                dtype=np.float64,
            )
            rotation = self.extrinsics[frame, :, :3]
            translation = self.extrinsics[frame, :, 3]
            world_point = (camera_point - translation) @ rotation
            world_h = np.append(world_point, 1.0)
            local_h = np.linalg.inv(self.owner_transforms[owner_id][frame]) @ world_h
            if not np.isfinite(local_h[:3]).all():
                continue

            valid[index] = True
            owner_ids[index] = owner_id
            local_points[index] = local_h[:3]

        return AttachedQueries(valid, owner_ids, local_points)

    def can_attach(self, query: np.ndarray) -> bool:
        query = np.asarray(query, dtype=np.float64).reshape(1, 3)
        return bool(self.attach_queries(query).valid[0])

    def _project_world_points(self, world_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project ``(T,N,3)`` world points to finite native pixel coordinates."""
        rotations = self.extrinsics[:, :, :3]
        translations = self.extrinsics[:, :, 3]
        camera_points = np.einsum("tij,tnj->tni", rotations, world_points) + translations[:, None, :]
        z = camera_points[..., 2]
        safe_z = np.where(
            np.abs(z) < 1e-8,
            np.where(z < 0.0, -1e-8, 1e-8),
            z,
        )
        x = self.intrinsics[:, None, 0, 0] * camera_points[..., 0] / safe_z
        x += self.intrinsics[:, None, 0, 2]
        y = self.intrinsics[:, None, 1, 1] * camera_points[..., 1] / safe_z
        y += self.intrinsics[:, None, 1, 2]
        tracks = np.stack([x, y], axis=-1)
        if not np.isfinite(tracks).all():
            raise FloatingPointError("simulator projection produced non-finite coordinates")
        return tracks, z

    def _compute_visibility(self, tracks: np.ndarray, z: np.ndarray, owner_ids: np.ndarray) -> np.ndarray:
        count = tracks.shape[1]
        visible = np.zeros((self.num_frames, count), dtype=bool)
        for frame in range(self.num_frames):
            xs_float = tracks[frame, :, 0]
            ys_float = tracks[frame, :, 1]
            inside = (
                (z[frame] > 0.0)
                & (xs_float >= 0.0)
                & (xs_float < self.width)
                & (ys_float >= 0.0)
                & (ys_float < self.height)
            )
            indices = np.flatnonzero(inside)
            if indices.size == 0:
                continue
            xs = np.clip(np.rint(xs_float[indices]).astype(np.int64), 0, self.width - 1)
            ys = np.clip(np.rint(ys_float[indices]).astype(np.int64), 0, self.height - 1)
            observed_depth = self.depth_maps[frame, ys, xs] / self.depth_scale
            same_owner = self.segmentations[frame, ys, xs].astype(np.int64) == owner_ids[indices]
            depth_matches = (
                np.isfinite(observed_depth)
                & (observed_depth > 0.0)
                & (np.abs(observed_depth - z[frame, indices]) <= self.depth_tolerance_meters)
            )
            visible[frame, indices] = same_owner & depth_matches
        return visible

    def track_queries(self, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        queries = np.asarray(queries, dtype=np.float64)
        attached = self.attach_queries(queries)
        count = len(queries)

        # Invalid first-pass candidates are finite, invisible, and constant,
        # guaranteeing zero temporal variance without resampling them.
        tracks = np.broadcast_to(queries[None, :, 1:3], (self.num_frames, count, 2)).copy()
        visible = np.zeros((self.num_frames, count), dtype=bool)
        valid_indices = np.flatnonzero(attached.valid)
        if valid_indices.size == 0:
            return tracks.astype(np.float32), visible, attached.valid

        world_points = np.zeros((self.num_frames, valid_indices.size, 3), dtype=np.float64)
        for local_index, query_index in enumerate(valid_indices):
            owner_id = int(attached.owner_ids[query_index])
            local_h = np.append(attached.local_points[query_index], 1.0)
            world_points[:, local_index] = np.einsum(
                "tij,j->ti", self.owner_transforms[owner_id], local_h
            )[:, :3]

        projected, z = self._project_world_points(world_points)
        projected_vis = self._compute_visibility(projected, z, attached.owner_ids[valid_indices])
        tracks[:, valid_indices] = projected
        visible[:, valid_indices] = projected_vis
        return tracks.astype(np.float32), visible, attached.valid


def temporal_coordinate_variance(tracks: np.ndarray) -> np.ndarray:
    """ATM's ``Var(x)+Var(y)`` using PyTorch's sample-variance convention."""
    tracks_tensor = torch.as_tensor(tracks, dtype=torch.float32)
    if tracks_tensor.ndim != 3 or tracks_tensor.shape[-1] != 2:
        raise ValueError(f"tracks must be (T,N,2), got {tuple(tracks_tensor.shape)}")
    if tracks_tensor.shape[0] < 2:
        return np.zeros(tracks_tensor.shape[1], dtype=np.float32)
    return torch.var(tracks_tensor, dim=0).sum(dim=-1).cpu().numpy()


def _sample_random_queries(height: int, width: int, num_frames: int) -> np.ndarray:
    if height * width < NUM_RANDOM_QUERIES:
        raise ValueError(
            f"image has only {height * width} pixels; cannot sample {NUM_RANDOM_QUERIES} distinct pixels"
        )
    flat_indices = np.random.choice(height * width, NUM_RANDOM_QUERIES, replace=False)
    xs = flat_indices % width
    ys = flat_indices // width
    frames = torch.randint(0, num_frames, (NUM_RANDOM_QUERIES,)).cpu().numpy()
    return np.stack([frames, xs, ys], axis=-1).astype(np.float32)


def _sample_double_grid_queries(height: int, width: int, num_frames: int) -> np.ndarray:
    def grid(start: float, stop: float) -> np.ndarray:
        xs = torch.linspace(start, stop, 7)
        ys = torch.linspace(start, stop, 7)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
        return torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1).cpu().numpy()

    normalized = np.concatenate([grid(0.05, 0.85), grid(0.15, 0.95)], axis=0)
    points = normalized * np.array([width, height], dtype=np.float32)
    frames = torch.randint(0, num_frames, (NUM_GRID_QUERIES,)).cpu().numpy()
    return np.concatenate([frames[:, None], points], axis=-1).astype(np.float32)


def _tile_survivors(queries: np.ndarray, survivor_indices: np.ndarray, target_count: int, label: str) -> np.ndarray:
    if survivor_indices.size == 0:
        raise RuntimeError(f"no {label} candidate survived the first simulator-track variance filter")
    survivors = queries[survivor_indices].copy()
    repetitions = target_count // len(survivors) + 1
    return np.tile(survivors, (repetitions, 1))[:target_count]


def _jitter_queries_until_attachable(
    oracle: QueryTracker,
    seed_queries: np.ndarray,
    max_noise_redraws: int = 10_000,
) -> np.ndarray:
    """Jitter only x/y; never redraw the surviving seed or its query frame."""
    jittered = seed_queries.copy().astype(np.float32)
    standard_deviation = JITTER_STD_HEIGHT_FRACTION * oracle.height
    for index, seed in enumerate(seed_queries):
        for _ in range(max_noise_redraws):
            noise = torch.randn(2).cpu().numpy() * standard_deviation
            candidate = seed.copy()
            candidate[1:] += noise
            x, y = candidate[1:]
            if not (0.0 <= x < oracle.width and 0.0 <= y < oracle.height):
                continue
            if oracle.can_attach(candidate):
                jittered[index] = candidate
                break
        else:
            raise RuntimeError(
                f"could not redraw valid Gaussian noise for {seed.tolist()} after "
                f"{max_noise_redraws} attempts"
            )
    return jittered


def _select_and_jitter(
    oracle: QueryTracker,
    queries: np.ndarray,
    threshold: float,
    target_count: int,
    label: str,
) -> np.ndarray:
    first_tracks, _, first_valid = oracle.track_queries(queries)
    variances = temporal_coordinate_variance(first_tracks)
    survivor_indices = np.flatnonzero(first_valid & (variances > threshold))
    if survivor_indices.size == 0:
        finite_variances = variances[np.isfinite(variances)]
        maximum = float(finite_variances.max()) if finite_variances.size else float("nan")
        raise RuntimeError(
            f"no {label} candidate survived strict variance > {threshold}; "
            f"valid attachments={int(first_valid.sum())}/{len(first_valid)}, max variance={maximum:.6f}"
        )
    tiled = _tile_survivors(queries, survivor_indices, target_count, label)
    return _jitter_queries_until_attachable(oracle, tiled)


def generate_atm_tracks(oracle: QueryTracker) -> tuple[np.ndarray, np.ndarray]:
    """Run ATM's original random and double-grid two-pass selection paths.

    The second simulator-tracking pass is deliberately not variance-filtered.
    Returns pixel coordinates ``(T,1098,2)`` and boolean visibility
    ``(T,1098)`` with the original ATM ordering: 98 grid tracks, then 1,000
    random tracks.
    """
    random_queries = _sample_random_queries(oracle.height, oracle.width, oracle.num_frames)
    final_random_queries = _select_and_jitter(
        oracle,
        random_queries,
        threshold=RANDOM_VARIANCE_THRESHOLD,
        target_count=NUM_RANDOM_QUERIES,
        label="random",
    )
    random_tracks, random_vis, random_valid = oracle.track_queries(final_random_queries)
    if not random_valid.all():
        raise RuntimeError("an attachable random query became invalid during the final simulator pass")

    grid_queries = _sample_double_grid_queries(oracle.height, oracle.width, oracle.num_frames)
    final_grid_queries = _select_and_jitter(
        oracle,
        grid_queries,
        threshold=GRID_VARIANCE_THRESHOLD,
        target_count=NUM_GRID_QUERIES,
        label="double-grid",
    )
    grid_tracks, grid_vis, grid_valid = oracle.track_queries(final_grid_queries)
    if not grid_valid.all():
        raise RuntimeError("an attachable grid query became invalid during the final simulator pass")

    tracks = np.concatenate([grid_tracks, random_tracks], axis=1).astype(np.float32, copy=False)
    visibility = np.concatenate([grid_vis, random_vis], axis=1).astype(bool, copy=False)
    expected_tracks = (oracle.num_frames, NUM_GRID_QUERIES + NUM_RANDOM_QUERIES, 2)
    expected_vis = expected_tracks[:2]
    if tracks.shape != expected_tracks or visibility.shape != expected_vis:
        raise AssertionError(f"unexpected ATM track shapes: {tracks.shape}, {visibility.shape}")
    if not np.isfinite(tracks).all():
        raise FloatingPointError("final ATM tracks contain non-finite coordinates")
    return tracks, visibility


@contextmanager
def robotwin_runtime():
    """Use RoboTwin's root for imports and its relative asset paths."""
    previous_cwd = Path.cwd()
    robotwin_path = str(ROBOTWIN_ROOT)
    if robotwin_path not in sys.path:
        sys.path.insert(0, robotwin_path)
    os.chdir(ROBOTWIN_ROOT)
    try:
        yield
    finally:
        os.chdir(previous_cwd)


def build_robotwin_task(task_name: str, task_config: str, raw_data_root: str) -> tuple[object, dict]:
    """Construct a headless RoboTwin task for cached-trajectory replay."""
    with robotwin_runtime():
        envs_module = importlib.import_module(f"envs.{task_name}")
        task = getattr(envs_module, task_name)()
        from envs._GLOBAL_CONFIGS import CONFIGS_PATH

        with open(Path(CONFIGS_PATH) / f"{task_config}.yml", encoding="utf-8") as file:
            args = yaml.load(file.read(), Loader=yaml.FullLoader)
        args["task_name"] = task_name

        with open(Path(CONFIGS_PATH) / "_embodiment_config.yml", encoding="utf-8") as file:
            embodiment_types = yaml.load(file.read(), Loader=yaml.FullLoader)

        embodiment = args["embodiment"]

        def embodiment_file(name: str) -> str:
            path = embodiment_types[name]["file_path"]
            if path is None:
                raise ValueError(f"missing embodiment files for {name}")
            return path

        def embodiment_config(path: str) -> dict:
            with open(Path(path) / "config.yml", encoding="utf-8") as file:
                return yaml.load(file.read(), Loader=yaml.FullLoader)

        if len(embodiment) == 1:
            args["left_robot_file"] = embodiment_file(embodiment[0])
            args["right_robot_file"] = embodiment_file(embodiment[0])
            args["dual_arm_embodied"] = True
        elif len(embodiment) == 3:
            args["left_robot_file"] = embodiment_file(embodiment[0])
            args["right_robot_file"] = embodiment_file(embodiment[1])
            args["embodiment_dis"] = embodiment[2]
            args["dual_arm_embodied"] = False
        else:
            raise ValueError(f"unsupported embodiment specification: {embodiment}")

        args["left_embodiment_config"] = embodiment_config(args["left_robot_file"])
        args["right_embodiment_config"] = embodiment_config(args["right_robot_file"])
        args["embodiment_name"] = (
            str(embodiment[0]) if len(embodiment) == 1 else f"{embodiment[0]}+{embodiment[1]}"
        )
        args["task_config"] = task_config
        args["save_path"] = str(Path(raw_data_root) / task_name / task_config)
        args["render_freq"] = 0
        args["need_plan"] = False
        args["save_data"] = False
    return task, args


def _entity_id(entity: object) -> int:
    if hasattr(entity, "per_scene_id"):
        return int(entity.per_scene_id)
    return int(entity.entity.per_scene_id)


def collect_pose_owners(task: object) -> dict[int, object]:
    """Return every segmentation-addressable actor and articulation link."""
    owners: dict[int, object] = {}
    for actor in task.scene.get_all_actors():
        owners[_entity_id(actor)] = actor
    for articulation in task.scene.get_all_articulations():
        for link in articulation.get_links():
            owners[_entity_id(link)] = link
    if not owners:
        raise RuntimeError("RoboTwin scene has no pose-bearing segmentation owners")
    return owners


def _world_transform(owner: object) -> np.ndarray:
    pose = owner.get_entity_pose() if hasattr(owner, "get_entity_pose") else owner.get_pose()
    return np.asarray(pose.to_transformation_matrix(), dtype=np.float64)


def replay_episode_geometry(
    task: object,
    args: dict,
    episode_index: int,
    seed: int,
    camera_name: str = "head_camera",
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Replay one cached joint path and capture segmentation plus owner poses."""
    if camera_name != "head_camera":
        raise ValueError(f"simulator-GT preprocessing only supports fixed head_camera, got {camera_name}")

    with robotwin_runtime():
        task.setup_demo(now_ep_num=episode_index, seed=seed, **args)
        owners = collect_pose_owners(task)
        capture_segmentations: list[np.ndarray] = []
        capture_poses: dict[int, list[np.ndarray]] = {owner_id: [] for owner_id in owners}

        def capture_picture() -> None:
            task._update_render()
            task.cameras.update_picture()
            try:
                camera_index = task.cameras.static_camera_name.index(camera_name)
            except ValueError as error:
                raise ValueError(f"RoboTwin replay has no fixed camera named {camera_name}") from error
            camera = task.cameras.static_camera_list[camera_index]
            capture_segmentations.append(camera.get_picture("Segmentation")[..., 1].copy())
            for owner_id, owner in owners.items():
                capture_poses[owner_id].append(_world_transform(owner))
            task.FRAME_IDX += 1

        task._take_picture = capture_picture
        trajectory = task.load_tran_data(episode_index)
        args["left_joint_path"] = trajectory["left_joint_path"]
        args["right_joint_path"] = trajectory["right_joint_path"]
        task.set_path_lst(args)
        task.play_once()
        task.close_env()

    segmentations = np.stack(capture_segmentations, axis=0)
    pose_stack = {owner_id: np.stack(poses, axis=0) for owner_id, poses in capture_poses.items()}
    return segmentations, pose_stack
