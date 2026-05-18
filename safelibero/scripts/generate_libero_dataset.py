#!/usr/bin/env python3
"""Generate EE-centered safety-field samples from fixed SafeLIBERO scenes.

This script is intentionally standalone: it does not patch LIBERO internals.
Run it from the repository root inside the workstation environment that has
SafeLIBERO / robosuite installed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


LIBERO_DUMMY_ACTION_VALUES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


# Manual first-pass map. These are manipulated objects, not receptacles.
DEFAULT_TARGET_OBJECTS: Dict[str, List[str]] = {
    "put_the_bowl_on_the_plate": ["akita_black_bowl_1"],
    "put_the_bowl_on_top_of_the_cabinet": ["akita_black_bowl_1"],
    "put_the_bowl_on_the_stove": ["akita_black_bowl_1"],
    "open_the_top_drawer_and_put_the_bowl_inside": ["akita_black_bowl_1"],
    "put_the_cream_cheese_in_the_bowl": ["cream_cheese_1"],
}


DEFAULT_EXCLUDE_PATTERNS = (
    "panda",
    "gripper",
    "mount",
    "main_table",
    "table",
    "floor",
    "arena",
)


@dataclass(frozen=True)
class CameraCloud:
    local: np.ndarray
    world: np.ndarray
    instance_ids: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect pointcloud/q/EE safety samples from SafeLIBERO scenes."
    )
    parser.add_argument("--task-suite-name", default="safelibero_goal")
    parser.add_argument("--safety-level", default="II", choices=["I", "II"])
    parser.add_argument("--task-indices", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--init-state-indices",
        type=int,
        nargs="*",
        default=None,
        help="Initial-state ids to use. Omit to use all states, optionally capped by --max-init-states.",
    )
    parser.add_argument("--max-init-states", type=int, default=None)
    parser.add_argument("--samples-per-init-state", type=int, default=100)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--camera-local", default="robot0_eye_in_hand")
    parser.add_argument("--camera-external", default="backview")
    parser.add_argument("--n-points", type=int, default=1024)
    parser.add_argument("--min-obstacle-points", type=int, default=32)
    parser.add_argument("--d-max", type=float, default=0.3)
    parser.add_argument("--d-min", type=float, default=0.0)
    parser.add_argument("--ee-radius", type=float, default=0.00)
    parser.add_argument("--translation-action-scale", type=float, default=0.025)
    parser.add_argument("--rotation-action-scale", type=float, default=0.08)
    parser.add_argument("--gripper-action", type=float, default=-1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--workspace-min",
        type=float,
        nargs=3,
        default=[-0.80, -0.80, 0.70],
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--workspace-max",
        type=float,
        nargs=3,
        default=[0.80, 0.80, 1.60],
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--exclude-name-patterns",
        nargs="*",
        default=list(DEFAULT_EXCLUDE_PATTERNS),
        help="Instance-name substrings excluded from obstacle pointclouds.",
    )
    parser.add_argument(
        "--target-object-map-json",
        default=None,
        help="Optional JSON dict overriding task_name -> list of manipulated object names.",
    )
    parser.add_argument(
        "--output",
        default="data/libero_safety/safelibero_goal_safety_v0.h5",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root. Defaults to the parent of scripts/.",
    )
    parser.add_argument(
        "--libero-config-path",
        default=None,
        help="SafeLIBERO config dir. Defaults to <repo>/.safelibero to avoid interactive setup.",
    )
    parser.add_argument(
        "--mujoco-gl",
        default="egl",
        help="Set MUJOCO_GL before importing robosuite. Use egl on the lab workstation.",
    )
    parser.add_argument(
        "--debug-vis-dir",
        default=None,
        help="Optional directory where RGB debug frames are saved during collection.",
    )
    parser.add_argument(
        "--debug-vis-every",
        type=int,
        default=25,
        help="Save one debug frame every K saved samples when --debug-vis-dir is set.",
    )
    parser.add_argument(
        "--debug-vis-max",
        type=int,
        default=200,
        help="Maximum number of debug frame pairs to save.",
    )
    parser.add_argument(
        "--debug-pointcloud-max-points",
        type=int,
        default=4096,
        help="Maximum points per cloud in debug pointcloud PNG previews.",
    )
    parser.add_argument(
        "--debug-save-ply",
        action="store_true",
        help="Also save debug pointclouds as ASCII PLY files.",
    )
    parser.add_argument(
        "--depth-is-metric",
        action="store_true",
        help="Skip robosuite get_real_depth_map if depth observations are already metric.",
    )
    parser.add_argument(
        "--no-flip-camera-images",
        action="store_true",
        help="Disable the image/depth flip used by existing LIBERO scripts in this repo.",
    )
    parser.add_argument(
        "--no-invert-v",
        action="store_true",
        help="Disable vertical pixel-coordinate inversion for robosuite camera projection.",
    )
    parser.set_defaults(save_jacobian=True)
    parser.add_argument("--save-jacobian", dest="save_jacobian", action="store_true")
    parser.add_argument("--no-save-jacobian", dest="save_jacobian", action="store_false")
    return parser.parse_args()


def find_repo_root(start: Path) -> Path:
    """Find the vlsa-aegis root even if this script is copied under safelibero/scripts."""

    start = start.resolve()
    candidates = [start] + list(start.parents)
    for candidate in candidates:
        libero_root = candidate / "safelibero" / "libero" / "libero"
        if (libero_root / "bddl_files").exists() and (libero_root / "init_files").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find repo root. Expected to find "
        "safelibero/libero/libero/{bddl_files,init_files} in a parent directory. "
        "Pass --repo-root explicitly."
    )


def expected_libero_config(repo_root: Path) -> Dict[str, str]:
    benchmark_root = repo_root / "safelibero" / "libero" / "libero"
    datasets_root = repo_root / "safelibero" / "libero" / "datasets"
    return {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(datasets_root),
        "assets": str(benchmark_root / "assets"),
    }


def write_libero_config(config_file: Path, config: Mapping[str, str]) -> None:
    config_text = "\n".join([f"{key}: {value}" for key, value in config.items()]) + "\n"
    config_file.write_text(config_text)


def read_libero_config(config_file: Path) -> Dict[str, str]:
    try:
        import yaml

        with open(config_file, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        return {str(k): str(v) for k, v in loaded.items()}
    except Exception:
        return {}


def config_needs_rewrite(config_file: Path, expected: Mapping[str, str]) -> bool:
    if not config_file.exists():
        return True
    current = read_libero_config(config_file)
    for key in ("benchmark_root", "bddl_files", "init_states", "assets"):
        value = current.get(key)
        if not value or not Path(value).exists():
            return True
    # Common failure mode when the script is launched from safelibero/scripts.
    if "safelibero/safelibero" in config_file.read_text(errors="ignore"):
        return True
    for key in ("bddl_files", "init_states", "assets"):
        if Path(current.get(key, "")) != Path(expected[key]):
            return True
    return False


def bootstrap_libero(repo_root: Path, libero_config_path: Optional[str], mujoco_gl: str) -> None:
    if mujoco_gl:
        os.environ["MUJOCO_GL"] = mujoco_gl
    sys.path.insert(0, str(repo_root / "safelibero"))

    if libero_config_path:
        config_dir = Path(libero_config_path)
        os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    else:
        config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", repo_root / ".safelibero"))
        os.environ.setdefault("LIBERO_CONFIG_PATH", str(config_dir))
    config_dir.mkdir(parents=True, exist_ok=True)

    config_file = config_dir / "config.yaml"
    expected = expected_libero_config(repo_root)
    if config_needs_rewrite(config_file, expected):
        write_libero_config(config_file, expected)


def patch_robosuite_numpy2_segmentation() -> None:
    """Patch robosuite 1.4.x segmentation rendering under NumPy 2.x.

    robosuite packs MuJoCo segmentation IDs from three uint8 RGB channels using
    expressions like ``rgb[:, :, 1] * 256``. NumPy 2 raises OverflowError for
    that operation on uint8 arrays. Casting to int32 before unpacking preserves
    the original semantics.
    """

    import mujoco
    from robosuite.utils import binding_utils

    current = binding_utils.MjRenderContext.read_pixels
    if getattr(current, "_aegis_numpy2_segmentation_patch", False):
        return

    def read_pixels(self, width, height, depth=False, segmentation=False):
        viewport = mujoco.MjrRect(0, 0, width, height)
        rgb_img = np.empty((height, width, 3), dtype=np.uint8)
        depth_img = np.empty((height, width), dtype=np.float32) if depth else None
        mujoco.mjr_readPixels(rgb=rgb_img, depth=depth_img, viewport=viewport, con=self.con)

        ret_img = rgb_img
        if segmentation:
            rgb_ids = rgb_img.astype(np.int32)
            seg_img = rgb_ids[:, :, 0] + rgb_ids[:, :, 1] * (2**8) + rgb_ids[:, :, 2] * (2**16)
            seg_img[seg_img >= (self.scn.ngeom + 1)] = 0

            seg_ids = np.full((self.scn.ngeom + 1, 2), fill_value=-1, dtype=np.int32)
            for i in range(self.scn.ngeom):
                geom = self.scn.geoms[i]
                if geom.segid != -1:
                    seg_ids[geom.segid + 1, 0] = geom.objtype
                    seg_ids[geom.segid + 1, 1] = geom.objid
            ret_img = seg_ids[seg_img]

        if depth:
            return ret_img, depth_img
        return ret_img

    read_pixels._aegis_numpy2_segmentation_patch = True
    binding_utils.MjRenderContext.read_pixels = read_pixels
    binding_utils.MjRenderContextOffscreen.read_pixels = read_pixels
    print("[dataset] patched robosuite segmentation renderer for NumPy 2.x")


def load_target_map(path_or_json: Optional[str]) -> Dict[str, List[str]]:
    if path_or_json is None:
        return dict(DEFAULT_TARGET_OBJECTS)

    value = path_or_json.strip()
    if value.startswith("{"):
        loaded = json.loads(value)
    else:
        with open(value, "r", encoding="utf-8") as f:
            loaded = json.load(f)

    return {str(k): [str(x) for x in v] for k, v in loaded.items()}


def get_obs_key(obs: Mapping[str, Any], candidates: Sequence[str], kind: str) -> str:
    for key in candidates:
        if key in obs:
            return key
    available = ", ".join(sorted(obs.keys()))
    raise KeyError(f"Could not find {kind} key. Tried {candidates}. Available keys: {available}")


def depth_key(camera_name: str) -> List[str]:
    return [
        f"{camera_name}_depth",
        f"{camera_name}_depths",
    ]


def segmentation_key(camera_name: str) -> List[str]:
    return [
        f"{camera_name}_segmentation_instance",
        f"{camera_name}_segmentation",
        f"{camera_name}_seg",
    ]


def image_key(camera_name: str) -> List[str]:
    return [
        f"{camera_name}_image",
        f"{camera_name}_rgb",
    ]


def normalize_patterns(patterns: Iterable[str]) -> List[str]:
    return [p.strip().lower() for p in patterns if p and p.strip()]


def instance_id_to_name(env: Any) -> Dict[int, str]:
    """Return robosuite instance segmentation id -> instance name.

    robosuite instance segmentation ids are 1-indexed in the helper wrapper
    shipped in this repo, hence enumerate(..., start=1).
    """

    model = env.env.model if hasattr(env, "env") else env.model
    names = list(getattr(model, "instances_to_ids", {}).keys())
    return {i: name for i, name in enumerate(names, start=1)}


def name_matches(name: str, patterns: Iterable[str]) -> bool:
    lower = name.lower()
    return any(pattern in lower for pattern in patterns)


def keep_instance_ids(
    id_to_name: Mapping[int, str],
    target_names: Sequence[str],
    exclude_patterns: Sequence[str],
) -> np.ndarray:
    target_patterns = normalize_patterns(target_names)
    excluded = normalize_patterns(exclude_patterns)

    keep: List[int] = []
    for instance_id, name in id_to_name.items():
        if name_matches(name, excluded):
            continue
        if name_matches(name, target_patterns):
            continue
        keep.append(int(instance_id))
    return np.asarray(keep, dtype=np.int32)


def workspace_mask(points_world: np.ndarray, workspace_min: np.ndarray, workspace_max: np.ndarray) -> np.ndarray:
    return np.logical_and(
        np.all(points_world >= workspace_min.reshape(1, 3), axis=1),
        np.all(points_world <= workspace_max.reshape(1, 3), axis=1),
    )


def camera_cloud_from_obs(
    obs: Mapping[str, Any],
    env: Any,
    camera_name: str,
    camera_utils: Any,
    depth_is_metric: bool,
    flip_images: bool,
    invert_v: bool,
) -> CameraCloud:
    d_key = get_obs_key(obs, depth_key(camera_name), "depth")
    s_key = get_obs_key(obs, segmentation_key(camera_name), "segmentation")

    depth = np.asarray(obs[d_key]).squeeze()
    seg = np.asarray(obs[s_key]).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"{d_key} must be 2D after squeeze, got shape {depth.shape}")
    if seg.ndim != 2:
        raise ValueError(f"{s_key} must be 2D after squeeze, got shape {seg.shape}")

    if not depth_is_metric:
        depth = camera_utils.get_real_depth_map(env.sim, depth)

    depth = depth.astype(np.float32)
    seg = seg.astype(np.int32)

    # Robosuite / MuJoCo depth e segmentation vanno flippate SOLO verticalmente.
    # Non fare flip orizzontale, altrimenti inverti l'asse x della pointcloud.

    height, width = depth.shape
    intrinsics = camera_utils.get_camera_intrinsic_matrix(env.sim, camera_name, height, width)
    extrinsics = camera_utils.get_camera_extrinsic_matrix(env.sim, camera_name)
    k_inv = np.linalg.inv(intrinsics)

    vv, uu = np.indices((height, width))

    # Non invertire v qui: il flip verticale è già stato fatto sulla depth/seg.

    valid = np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        return CameraCloud(
            local=np.zeros((0, 3), dtype=np.float32),
            world=np.zeros((0, 3), dtype=np.float32),
            instance_ids=np.zeros((0,), dtype=np.int32),
        )

    u_flat = uu[valid].reshape(-1)
    v_flat = vv[valid].reshape(-1)
    depth_flat = depth[valid].reshape(-1)
    instance_flat = seg[valid].reshape(-1).astype(np.int32)

    pixels_h = np.stack([u_flat, v_flat, np.ones_like(u_flat)], axis=0).astype(np.float32)
    points_local = (k_inv @ pixels_h) * depth_flat.reshape(1, -1)
    points_local[1, :] *= -1.0
    points_world_h = extrinsics @ np.vstack([points_local, np.ones((1, points_local.shape[1]))]) 

    return CameraCloud(
        local=points_local.T.astype(np.float32),
        world=points_world_h[:3].T.astype(np.float32),
        instance_ids=instance_flat.astype(np.int32),
    )


def filter_cloud(
    cloud: CameraCloud,
    keep_ids: np.ndarray,
    workspace_min: np.ndarray,
    workspace_max: np.ndarray,
) -> CameraCloud:
    if cloud.world.shape[0] == 0 or keep_ids.shape[0] == 0:
        return CameraCloud(
            local=np.zeros((0, 3), dtype=np.float32),
            world=np.zeros((0, 3), dtype=np.float32),
            instance_ids=np.zeros((0,), dtype=np.int32),
        )

    mask = np.isin(cloud.instance_ids, keep_ids)
    mask &= workspace_mask(cloud.world, workspace_min, workspace_max)

    return CameraCloud(
        local=cloud.local[mask].astype(np.float32),
        world=cloud.world[mask].astype(np.float32),
        instance_ids=cloud.instance_ids[mask].astype(np.int32),
    )


def camera_rgb_from_obs(obs: Mapping[str, Any], camera_name: str, flip_images: bool) -> np.ndarray:
    key = get_obs_key(obs, image_key(camera_name), "RGB image")
    rgb = np.asarray(obs[key]).squeeze()
    if rgb.ndim != 3:
        raise ValueError(f"{key} must be HxWxC after squeeze, got shape {rgb.shape}")
    rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.float32)
        if float(np.nanmax(rgb)) <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if flip_images:
        rgb = rgb[::-1, :]
    return np.ascontiguousarray(rgb)


def segmentation_from_obs(obs: Mapping[str, Any], camera_name: str, flip_images: bool) -> np.ndarray:
    key = get_obs_key(obs, segmentation_key(camera_name), "segmentation")
    seg = np.asarray(obs[key]).squeeze().astype(np.int32)
    if seg.ndim != 2:
        raise ValueError(f"{key} must be 2D after squeeze, got shape {seg.shape}")
    if flip_images:
        seg = seg[::-1, :]
    return seg


def save_debug_frame(
    obs: Mapping[str, Any],
    camera_names: Sequence[str],
    keep_ids: np.ndarray,
    output_dir: Path,
    frame_idx: int,
    metadata: Mapping[str, Any],
    flip_images: bool,
) -> None:
    import imageio.v2 as imageio

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_panels = []
    overlay_panels = []
    for camera_name in camera_names:
        rgb = camera_rgb_from_obs(obs, camera_name, flip_images)
        seg = segmentation_from_obs(obs, camera_name, flip_images)
        mask = np.isin(seg, keep_ids)
        overlay = rgb.copy()
        if np.any(mask):
            color = np.asarray([255, 80, 0], dtype=np.float32)
            overlay[mask] = np.clip(0.55 * overlay[mask].astype(np.float32) + 0.45 * color, 0, 255)
        raw_panels.append(rgb.astype(np.uint8))
        overlay_panels.append(overlay.astype(np.uint8))

    min_height = min(panel.shape[0] for panel in raw_panels + overlay_panels)
    raw_panels = [panel[:min_height] for panel in raw_panels]
    overlay_panels = [panel[:min_height] for panel in overlay_panels]
    raw_row = np.concatenate(raw_panels, axis=1)
    overlay_row = np.concatenate(overlay_panels, axis=1)
    canvas = np.concatenate([raw_row, overlay_row], axis=0)
    filename = (
        f"images_{frame_idx:06d}_"
        f"task_{int(metadata['task_id']):03d}_"
        f"init_{int(metadata['init_state_id']):03d}_"
        f"step_{int(metadata['rollout_step']):04d}.png"
    )
    imageio.imwrite(output_dir / filename, canvas)

    meta_path = output_dir / "frames.jsonl"
    with open(meta_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(metadata), sort_keys=True) + "\n")


def downsample_points_for_debug(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    if points.shape[0] <= max_points:
        return points
    indices = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[indices]


def set_axes_equal_3d(ax: Any, points: np.ndarray, center: Optional[np.ndarray] = None) -> None:
    if points.shape[0] == 0:
        if center is None:
            center = np.zeros(3, dtype=np.float32)
        span = 0.5
        ax.set_xlim(center[0] - span, center[0] + span)
        ax.set_ylim(center[1] - span, center[1] + span)
        ax.set_zlim(center[2] - span, center[2] + span)
        return

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    if center is not None:
        mins = np.minimum(mins, center.reshape(3))
        maxs = np.maximum(maxs, center.reshape(3))
    mid = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    span = max(span, 0.15)
    radius = 0.55 * span
    ax.set_xlim(mid[0] - radius, mid[0] + radius)
    ax.set_ylim(mid[1] - radius, mid[1] + radius)
    ax.set_zlim(mid[2] - radius, mid[2] + radius)


def write_ascii_ply(path: Path, points: np.ndarray, color: Tuple[int, int, int]) -> None:
    points = np.asarray(points, dtype=np.float32)
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        r, g, b = color
        for point in points:
            f.write(f"{point[0]:.7f} {point[1]:.7f} {point[2]:.7f} {r} {g} {b}\n")


def save_debug_pointcloud_frame(
    wrist_cloud: CameraCloud,
    ext_cloud: CameraCloud,
    ee_pos_world: np.ndarray,
    v_rep_world: np.ndarray,
    output_dir: Path,
    frame_idx: int,
    metadata: Mapping[str, Any],
    max_points: int,
    rng: np.random.Generator,
    save_ply: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    wrist_local = downsample_points_for_debug(wrist_cloud.local, max_points, rng)
    ext_local = downsample_points_for_debug(ext_cloud.local, max_points, rng)
    wrist_world = downsample_points_for_debug(wrist_cloud.world, max_points, rng)
    ext_world = downsample_points_for_debug(ext_cloud.world, max_points, rng)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 9))
    axes = [
        fig.add_subplot(2, 2, 1, projection="3d"),
        fig.add_subplot(2, 2, 2, projection="3d"),
        fig.add_subplot(2, 2, 3, projection="3d"),
        fig.add_subplot(2, 2, 4, projection="3d"),
    ]
    configs = [
        ("wrist / EE local", wrist_local, "#1f77b4", None),
        ("external local", ext_local, "#ff7f0e", None),
        ("wrist / EE world", wrist_world, "#1f77b4", ee_pos_world),
        ("external world", ext_world, "#ff7f0e", ee_pos_world),
    ]

    for ax, (title, points, color, ee_center) in zip(axes, configs):
        if points.shape[0] > 0:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1.5, c=color, alpha=0.75)
        if ee_center is not None:
            ee = np.asarray(ee_center, dtype=np.float32).reshape(3)
            ax.scatter([ee[0]], [ee[1]], [ee[2]], s=45, c="#d62728", marker="o", label="EE")
            rep = np.asarray(v_rep_world, dtype=np.float32).reshape(3)
            ax.quiver(ee[0], ee[1], ee[2], rep[0], rep[1], rep[2], length=0.12, color="#2ca02c")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        set_axes_equal_3d(ax, points, ee_center)
        ax.view_init(elev=24, azim=-55)

    fig.suptitle(
        f"task={metadata['task_id']} init={metadata['init_state_id']} "
        f"step={metadata['rollout_step']} S={metadata['S_star_obs']:.3f} d={metadata['d_obs_hard']:.3f}"
    )
    fig.tight_layout()

    filename = (
        f"pointclouds_{frame_idx:06d}_"
        f"task_{int(metadata['task_id']):03d}_"
        f"init_{int(metadata['init_state_id']):03d}_"
        f"step_{int(metadata['rollout_step']):04d}.png"
    )
    fig.savefig(output_dir / filename, dpi=150)
    plt.close(fig)

    if save_ply:
        ply_dir = output_dir / "ply"
        prefix = (
            f"cloud_{frame_idx:06d}_"
            f"task_{int(metadata['task_id']):03d}_"
            f"init_{int(metadata['init_state_id']):03d}_"
            f"step_{int(metadata['rollout_step']):04d}"
        )
        write_ascii_ply(ply_dir / f"{prefix}_wrist_world.ply", wrist_cloud.world, (31, 119, 180))
        write_ascii_ply(ply_dir / f"{prefix}_external_world.ply", ext_cloud.world, (255, 127, 14))


def sample_or_pad_cloud(
    cloud: CameraCloud,
    n_points: int,
    rng: np.random.Generator,
    empty_local: Sequence[float],
    empty_world: Sequence[float],
) -> CameraCloud:
    count = int(cloud.world.shape[0])
    if count >= n_points:
        indices = rng.choice(count, size=n_points, replace=False)
        return CameraCloud(
            local=cloud.local[indices].astype(np.float32),
            world=cloud.world[indices].astype(np.float32),
            instance_ids=cloud.instance_ids[indices].astype(np.int32),
        )

    if count > 0:
        pad = rng.choice(count, size=n_points - count, replace=True)
        indices = np.concatenate([np.arange(count), pad])
        return CameraCloud(
            local=cloud.local[indices].astype(np.float32),
            world=cloud.world[indices].astype(np.float32),
            instance_ids=cloud.instance_ids[indices].astype(np.int32),
        )

    return CameraCloud(
        local=np.tile(np.asarray(empty_local, dtype=np.float32).reshape(1, 3), (n_points, 1)),
        world=np.tile(np.asarray(empty_world, dtype=np.float32).reshape(1, 3), (n_points, 1)),
        instance_ids=np.full((n_points,), -1, dtype=np.int32),
    )


def compute_safety_targets(
    points_world: np.ndarray,
    ee_pos_world: np.ndarray,
    d_max: float,
    d_min: float,
    ee_radius: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points_world.shape[0] == 0:
        return (
            np.asarray([1.0], dtype=np.float32),
            np.asarray([1e6], dtype=np.float32),
            np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        )

    ee = ee_pos_world.reshape(1, 3).astype(np.float32)
    delta = ee - points_world.astype(np.float32)
    distances = np.linalg.norm(delta, axis=1)
    closest_idx = int(np.argmin(distances))
    closest_center_dist = float(distances[closest_idx])
    d_obs = max(0.0, closest_center_dist - float(ee_radius))
    s_star = float(np.clip(d_obs / float(d_max), float(d_min) / float(d_max), 1.0))

    rep = delta[closest_idx]
    norm = float(np.linalg.norm(rep))
    if norm < 1e-8:
        rep = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        rep = (rep / norm).astype(np.float32)

    return (
        np.asarray([s_star], dtype=np.float32),
        np.asarray([d_obs], dtype=np.float32),
        rep.astype(np.float32),
    )


def random_action(args: argparse.Namespace, rng: np.random.Generator) -> np.ndarray:
    action = np.zeros(7, dtype=np.float32)
    action[:3] = rng.uniform(
        -float(args.translation_action_scale),
        float(args.translation_action_scale),
        size=3,
    )
    action[3:6] = rng.uniform(
        -float(args.rotation_action_scale),
        float(args.rotation_action_scale),
        size=3,
    )
    action[6] = float(args.gripper_action)
    return action


def compute_jacobian(env: Any) -> np.ndarray:
    nan_jac = np.full((3, 7), np.nan, dtype=np.float32)
    try:
        body_name = "eef_marker"
        jac_full = env.sim.data.get_body_jacp(body_name).reshape(3, -1)
        robot = env.robots[0]
        vel_indices = getattr(robot, "_ref_joint_vel_indexes", None)
        if vel_indices is None:
            vel_indices = getattr(robot, "_ref_joint_vel_indexes", None)
        if vel_indices is None:
            vel_indices = list(range(7))
        jac = jac_full[:, list(vel_indices)[:7]]
        if jac.shape != (3, 7):
            return nan_jac
        return jac.astype(np.float32)
    except Exception:
        return nan_jac


class H5Writer:
    def __init__(self, path: Path, n_points: int, save_jacobian: bool):
        import h5py

        path.parent.mkdir(parents=True, exist_ok=True)
        self.hf = h5py.File(path, "w")
        self.group = self.hf.create_group("samples")
        self.n = 0
        self.save_jacobian = save_jacobian

        self._create("q", (7,), np.float32)
        self._create("ee_pos_world", (3,), np.float32)
        self._create("ee_ori_world", (4,), np.float32)
        self._create("pointcloud_wrist_local", (n_points, 3), np.float32)
        self._create("pointcloud_wrist_world", (n_points, 3), np.float32)
        self._create("pointcloud_ext_local", (n_points, 3), np.float32)
        self._create("pointcloud_ext_world", (n_points, 3), np.float32)
        self._create("point_object_id_wrist", (n_points,), np.int32)
        self._create("point_object_id_ext", (n_points,), np.int32)
        self._create("S_star_obs", (1,), np.float32)
        self._create("d_obs_hard", (1,), np.float32)
        self._create("v_rep", (3,), np.float32)
        self._create("task_id", (1,), np.int32)
        self._create("init_state_id", (1,), np.int32)
        self._create("rollout_step", (1,), np.int32)
        if save_jacobian:
            self._create("J_ee", (3, 7), np.float32)

    def _create(self, name: str, tail_shape: Tuple[int, ...], dtype: Any) -> None:
        chunks = (1,) + tail_shape
        self.group.create_dataset(
            name,
            shape=(0,) + tail_shape,
            maxshape=(None,) + tail_shape,
            chunks=chunks,
            dtype=dtype,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

    def append(self, sample: Mapping[str, np.ndarray]) -> None:
        idx = self.n
        for name, value in sample.items():
            ds = self.group[name]
            ds.resize((idx + 1,) + ds.shape[1:])
            ds[idx] = value
        self.n += 1

    def close(self) -> None:
        self.hf.attrs["num_samples"] = int(self.n)
        self.hf.close()


def iter_init_indices(total: int, requested: Optional[Sequence[int]], max_init_states: Optional[int]) -> List[int]:
    if requested is None or len(requested) == 0:
        indices = list(range(total))
    else:
        indices = [int(i) for i in requested]

    if max_init_states is not None:
        indices = indices[: int(max_init_states)]

    for idx in indices:
        if idx < 0 or idx >= total:
            raise IndexError(f"init_state index {idx} is out of range [0, {total})")
    return indices


def tqdm_or_plain(iterable: Iterable[Any], **kwargs: Any) -> Iterable[Any]:
    try:
        from tqdm import tqdm

        return tqdm(iterable, **kwargs)
    except Exception:
        return iterable


def main() -> None:
    args = parse_args()
    global np
    import numpy as np

    libero_dummy_action = np.asarray(LIBERO_DUMMY_ACTION_VALUES, dtype=np.float32)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root(Path(__file__).resolve().parent)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists. Pass --overwrite to replace it.")

    bootstrap_libero(repo_root, args.libero_config_path, args.mujoco_gl)

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import SegmentationRenderEnv
    from robosuite.utils import camera_utils

    patch_robosuite_numpy2_segmentation()

    np.random.seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    target_map = load_target_map(args.target_object_map_json)
    workspace_min = np.asarray(args.workspace_min, dtype=np.float32)
    workspace_max = np.asarray(args.workspace_max, dtype=np.float32)
    flip_images = not bool(args.no_flip_camera_images)
    invert_v = not bool(args.no_invert_v)
    debug_vis_dir = Path(args.debug_vis_dir) if args.debug_vis_dir else None
    if debug_vis_dir is not None and not debug_vis_dir.is_absolute():
        debug_vis_dir = repo_root / debug_vis_dir
    debug_vis_every = max(1, int(args.debug_vis_every))
    debug_vis_max = max(0, int(args.debug_vis_max))
    debug_frames_saved = 0

    print(f"[dataset] repo_root={repo_root}")
    print(f"[dataset] LIBERO_CONFIG_PATH={os.environ.get('LIBERO_CONFIG_PATH')}")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name](safety_level=args.safety_level)

    writer = H5Writer(output_path, n_points=args.n_points, save_jacobian=args.save_jacobian)
    writer.hf.attrs["schema_version"] = "libero_safety_v0"
    writer.hf.attrs["task_suite_name"] = args.task_suite_name
    writer.hf.attrs["safety_level"] = args.safety_level
    writer.hf.attrs["camera_local"] = args.camera_local
    writer.hf.attrs["camera_external"] = args.camera_external
    writer.hf.attrs["n_points"] = int(args.n_points)
    writer.hf.attrs["d_max"] = float(args.d_max)
    writer.hf.attrs["d_min"] = float(args.d_min)
    writer.hf.attrs["ee_radius"] = float(args.ee_radius)
    writer.hf.attrs["target_object_map_json"] = json.dumps(target_map, sort_keys=True)
    writer.hf.attrs["exclude_name_patterns_json"] = json.dumps(args.exclude_name_patterns)
    writer.hf.attrs["workspace_min"] = workspace_min
    writer.hf.attrs["workspace_max"] = workspace_max
    writer.hf.attrs["depth_is_metric"] = bool(args.depth_is_metric)
    writer.hf.attrs["flip_camera_images"] = bool(flip_images)
    writer.hf.attrs["invert_v"] = bool(invert_v)
    writer.hf.attrs["debug_save_ply"] = bool(args.debug_save_ply)

    instance_maps: Dict[str, Dict[str, str]] = {}
    total_skipped_low_points = 0

    try:
        for task_id in args.task_indices:
            task = task_suite.get_task(int(task_id))
            task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            init_states = task_suite.get_task_init_states(int(task_id))
            init_indices = iter_init_indices(
                total=len(init_states),
                requested=args.init_state_indices,
                max_init_states=args.max_init_states,
            )
            target_names = target_map.get(task.name, [])

            env_args = {
                "bddl_file_name": str(task_bddl_file),
                "camera_heights": int(args.resolution),
                "camera_widths": int(args.resolution),
                "camera_depths": True,
                "camera_segmentations": "instance",
                "camera_names": [args.camera_local, args.camera_external],
            }
            env = SegmentationRenderEnv(**env_args)
            env.seed(int(args.seed))

            print(
                f"[dataset] task_id={task_id} name={task.name} "
                f"targets={target_names or 'NONE'} init_states={len(init_indices)}"
            )

            try:
                for init_id in tqdm_or_plain(init_indices, desc=f"task {task_id} init", leave=True):
                    env.reset()
                    obs = env.set_init_state(init_states[init_id])

                    for _ in range(int(args.settle_steps)):
                        obs, _, _, _ = env.step(libero_dummy_action)

                    id_to_name = instance_id_to_name(env)
                    instance_maps[task.name] = {str(k): v for k, v in id_to_name.items()}
                    keep_ids = keep_instance_ids(
                        id_to_name=id_to_name,
                        target_names=target_names,
                        exclude_patterns=args.exclude_name_patterns,
                    )

                    for rollout_step in range(int(args.samples_per_init_state)):
                        if rollout_step > 0:
                            obs, _, _, _ = env.step(random_action(args, rng))

                        wrist_raw = camera_cloud_from_obs(
                            obs=obs,
                            env=env,
                            camera_name=args.camera_local,
                            camera_utils=camera_utils,
                            depth_is_metric=args.depth_is_metric,
                            flip_images=flip_images,
                            invert_v=invert_v,
                        )
                        ext_raw = camera_cloud_from_obs(
                            obs=obs,
                            env=env,
                            camera_name=args.camera_external,
                            camera_utils=camera_utils,
                            depth_is_metric=args.depth_is_metric,
                            flip_images=flip_images,
                            invert_v=invert_v,
                        )

                        wrist = filter_cloud(wrist_raw, keep_ids, workspace_min, workspace_max)
                        ext = filter_cloud(ext_raw, keep_ids, workspace_min, workspace_max)
                        combined_world = np.concatenate([wrist.world, ext.world], axis=0)
                        if combined_world.shape[0] < int(args.min_obstacle_points):
                            total_skipped_low_points += 1
                            continue

                        wrist_sample = sample_or_pad_cloud(
                            wrist,
                            n_points=int(args.n_points),
                            rng=rng,
                            empty_local=(0.0, 0.0, 10.0),
                            empty_world=(10.0, 10.0, 10.0),
                        )
                        ext_sample = sample_or_pad_cloud(
                            ext,
                            n_points=int(args.n_points),
                            rng=rng,
                            empty_local=(0.0, 0.0, 10.0),
                            empty_world=(10.0, 10.0, 10.0),
                        )

                        q = np.asarray(obs["robot0_joint_pos"], dtype=np.float32)
                        ee_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
                        ee_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32)
                        s_star, d_obs, v_rep = compute_safety_targets(
                            points_world=combined_world,
                            ee_pos_world=ee_pos,
                            d_max=float(args.d_max),
                            d_min=float(args.d_min),
                            ee_radius=float(args.ee_radius),
                        )

                        print(s_star, d_obs)

                        sample: Dict[str, np.ndarray] = {
                            "q": q,
                            "ee_pos_world": ee_pos,
                            "ee_ori_world": ee_quat,
                            "pointcloud_wrist_local": wrist_sample.local,
                            "pointcloud_wrist_world": wrist_sample.world,
                            "pointcloud_ext_local": ext_sample.local,
                            "pointcloud_ext_world": ext_sample.world,
                            "point_object_id_wrist": wrist_sample.instance_ids,
                            "point_object_id_ext": ext_sample.instance_ids,
                            "S_star_obs": s_star,
                            "d_obs_hard": d_obs,
                            "v_rep": v_rep,
                            "task_id": np.asarray([task_id], dtype=np.int32),
                            "init_state_id": np.asarray([init_id], dtype=np.int32),
                            "rollout_step": np.asarray([rollout_step], dtype=np.int32),
                        }
                        if args.save_jacobian:
                            sample["J_ee"] = compute_jacobian(env)

                        writer.append(sample)

                        if (
                            debug_vis_dir is not None
                            and debug_frames_saved < debug_vis_max
                            and writer.n % debug_vis_every == 0
                        ):
                            debug_metadata = {
                                "task_id": int(task_id),
                                "task_name": task.name,
                                "init_state_id": int(init_id),
                                "rollout_step": int(rollout_step),
                                "sample_index": int(writer.n - 1),
                                "S_star_obs": float(s_star[0]),
                                "d_obs_hard": float(d_obs[0]),
                            }
                            save_debug_frame(
                                obs=obs,
                                camera_names=[args.camera_local, args.camera_external],
                                keep_ids=keep_ids,
                                output_dir=debug_vis_dir,
                                frame_idx=debug_frames_saved,
                                metadata=debug_metadata,
                                flip_images=flip_images,
                            )
                            save_debug_pointcloud_frame(
                                wrist_cloud=wrist,
                                ext_cloud=ext,
                                ee_pos_world=ee_pos,
                                v_rep_world=v_rep,
                                output_dir=debug_vis_dir,
                                frame_idx=debug_frames_saved,
                                metadata=debug_metadata,
                                max_points=int(args.debug_pointcloud_max_points),
                                rng=rng,
                                save_ply=bool(args.debug_save_ply),
                            )
                            debug_frames_saved += 1
            finally:
                env.close()

    finally:
        writer.hf.attrs["instance_maps_json"] = json.dumps(instance_maps, sort_keys=True)
        writer.hf.attrs["skipped_low_obstacle_points"] = int(total_skipped_low_points)
        writer.hf.attrs["debug_frames_saved"] = int(debug_frames_saved)
        writer.close()

    print(f"[dataset] wrote {writer.n} samples to {output_path}")
    print(f"[dataset] skipped_low_obstacle_points={total_skipped_low_points}")
    if debug_vis_dir is not None:
        print(f"[dataset] debug frames saved: {debug_frames_saved} in {debug_vis_dir}")


if __name__ == "__main__":
    main()