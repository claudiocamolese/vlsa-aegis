#!/usr/bin/env python3
"""Generate CBF-style signed-clearance samples from fixed SafeLIBERO scenes.

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

import numpy as np


LIBERO_DUMMY_ACTION_VALUES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


# Manual first-pass map. These are manipulated objects, not receptacles.
DEFAULT_TARGET_OBJECTS: Dict[str, List[str]] = {
    "put_the_bowl_on_the_plate": ["akita_black_bowl_1"],
    "put_the_bowl_on_top_of_the_cabinet": ["akita_black_bowl_1"],
    "put_the_bowl_on_the_stove": ["akita_black_bowl_1"],
    "open_the_top_drawer_and_put_the_bowl_inside": ["akita_black_bowl_1"],
    "put_the_cream_cheese_in_the_bowl": ["cream_cheese_1", "akita_black_bowl_1"],
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate": [
        "akita_black_bowl_1"
    ],
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate": ["akita_black_bowl_1"],
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate": ["akita_black_bowl_1"],
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate": [
        "akita_black_bowl_1"
    ],
    "pick_up_the_bbq_sauce_and_place_it_in_the_basket": ["bbq_sauce_1"],
    "pick_up_the_chocolate_pudding_and_place_it_in_the_basket": ["chocolate_pudding_1"],
    "pick_up_the_milk_and_place_it_in_the_basket": ["milk_1"],
    "pick_up_the_orange_juice_and_place_it_in_the_basket": ["orange_juice_1"],
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket": [
        "alphabet_soup_1",
        "cream_cheese_1",
    ],
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket": [
        "alphabet_soup_1",
        "tomato_sauce_1",
    ],
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate": [
        "porcelain_mug_1",
        "white_yellow_mug_1",
    ],
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate": [
        "porcelain_mug_1",
        "chocolate_pudding_1",
    ],
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


DEFAULT_ROBOT_COLLISION_PATTERNS = (
    "panda",
    "gripper",
    "finger",
    "hand",
    "eef",
    "robot0_link",
)


DEFAULT_ROBOT_KEYPOINT_BODY_NAMES = (
    "robot0_link0",
    "robot0_link1",
    "robot0_link2",
    "robot0_link3",
    "robot0_link4",
    "robot0_link5",
    "robot0_link6",
    "robot0_link7",
    "gripper0_eef",
)


@dataclass(frozen=True)
class CameraCloud:
    local: np.ndarray
    world: np.ndarray
    instance_ids: np.ndarray


@dataclass(frozen=True)
class IKFrame:
    kind: str
    name: str


@dataclass(frozen=True)
class IKResult:
    success: bool
    q: np.ndarray
    pos_error: float
    ori_error: float
    iterations: int


@dataclass(frozen=True)
class SafetyTargets:
    S_star_obs: np.ndarray
    S_star_hard: np.ndarray
    S_star_knn: np.ndarray
    S_star_robust: np.ndarray
    d_obs: np.ndarray
    d_obs_hard: np.ndarray
    d_obs_knn: np.ndarray
    d_obs_robust: np.ndarray
    v_rep: np.ndarray
    v_rep_knn: np.ndarray
    closest_point_idx: np.ndarray
    closest_raw_point_idx: np.ndarray
    closest_point_world: np.ndarray
    closest_volume_point_world: np.ndarray
    closest_point_source: np.ndarray


@dataclass(frozen=True)
class RobotGeometryTargets:
    robot_keypoints_world: np.ndarray
    robot_keypoint_valid_mask: np.ndarray
    robot_link_valid_mask: np.ndarray
    d_gt_keypoints: np.ndarray
    d_gt_links: np.ndarray
    closest_keypoint: np.ndarray
    closest_link: np.ndarray
    closest_keypoint_distance: np.ndarray
    closest_link_distance: np.ndarray


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
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=0.02,
        help="CBF safety margin in meters. h_star = signed_clearance - safety_margin.",
    )
    parser.add_argument(
        "--h-scale",
        type=float,
        default=0.10,
        help="Meters used to normalize h_star into [-1, 1].",
    )
    parser.add_argument(
        "--safety-distance-mode",
        choices=["hard", "knn", "knn_capped"],
        default="knn_capped",
        help="Which distance label is used for S_star_obs. Hard uses min distance; knn/knn_capped are robust to isolated noisy points.",
    )
    parser.add_argument(
        "--safety-knn-k",
        type=int,
        default=8,
        help="Number of closest obstacle points averaged for robust safety distance.",
    )
    parser.add_argument(
        "--safety-robust-cap",
        type=float,
        default=0.02,
        help="Max meters that knn_capped may exceed hard distance.",
    )
    parser.add_argument(
        "--ee-radius",
        type=float,
        default=0.04,
        help="Half-width of the EE safety box, or sphere radius when --ee-safety-geometry=eef_sphere.",
    )
    parser.add_argument(
        "--ee-safety-geometry",
        choices=["eef_sphere", "camera_to_eef_box"],
        default="eef_sphere",
        help="Geometry used for S_star/d_obs. camera_to_eef_box covers the segment from wrist camera to gripper TCP.",
    )
    parser.add_argument("--translation-action-scale", type=float, default=0.025)
    parser.add_argument("--rotation-action-scale", type=float, default=0.08)
    parser.add_argument("--gripper-action", type=float, default=-1.0)
    parser.add_argument(
        "--sampling-mode",
        choices=["random_pose", "random_action", "mixed"],
        default="random_pose",
        help=(
            "random_pose samples absolute EE poses and solves IK; random_action keeps the old "
            "action rollout; mixed combines random_pose with near-boundary pose proposals."
        ),
    )
    parser.add_argument(
        "--near-boundary-fraction",
        type=float,
        default=0.60,
        help="Fraction of mixed-mode attempts that sample an EE target near an obstacle point.",
    )
    parser.add_argument(
        "--boundary-band",
        type=float,
        default=0.06,
        help="Extra meters above --safety-margin used for near-boundary sampling.",
    )
    parser.add_argument(
        "--near-boundary-min-dist",
        type=float,
        default=0.0,
        help="Minimum desired clearance for near-boundary proposals before IK.",
    )
    parser.add_argument(
        "--near-boundary-max-dist",
        type=float,
        default=None,
        help=(
            "Maximum desired clearance for near-boundary proposals. Defaults to "
            "--safety-margin + --boundary-band."
        ),
    )
    parser.add_argument(
        "--near-boundary-max-tries",
        type=int,
        default=32,
        help="Rejection-sampling attempts for an in-workspace near-boundary EE target.",
    )
    parser.add_argument(
        "--balance-safety-regions",
        action="store_true",
        help="Accept samples per init state to match target safety-region fractions.",
    )
    parser.add_argument(
        "--target-safe-far-fraction",
        type=float,
        default=0.20,
        help="Target fraction for safety_region=0 when --balance-safety-regions is enabled.",
    )
    parser.add_argument(
        "--target-near-boundary-fraction",
        type=float,
        default=0.45,
        help="Target fraction for safety_region=1 when --balance-safety-regions is enabled.",
    )
    parser.add_argument(
        "--target-unsafe-fraction",
        type=float,
        default=0.35,
        help="Target fraction for safety_region=2 when --balance-safety-regions is enabled.",
    )
    parser.add_argument(
        "--pose-workspace-min",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Min xyz for random EE pose sampling. Defaults to --workspace-min.",
    )
    parser.add_argument(
        "--pose-workspace-max",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Max xyz for random EE pose sampling. Defaults to --workspace-max.",
    )
    parser.add_argument(
        "--pose-xy-source",
        choices=["table", "box"],
        default="table",
        help="Use the tabletop XY bounds for random EE poses, or the explicit pose/workspace box.",
    )
    parser.add_argument(
        "--pose-table-margin",
        type=float,
        default=0.05,
        help="Inset applied to tabletop XY bounds when --pose-xy-source=table.",
    )
    parser.add_argument(
        "--ee-orientation-delta-range-deg",
        type=float,
        nargs=3,
        default=[25.0, 25.0, 180.0],
        metavar=("ROLL", "PITCH", "YAW"),
        help="Uniform +/- Euler perturbation around the reference EE orientation, in degrees.",
    )
    parser.add_argument(
        "--ee-orientation-reference-quat-xyzw",
        type=float,
        nargs=4,
        default=None,
        metavar=("X", "Y", "Z", "W"),
        help="Optional fixed reference quaternion for EE orientation sampling. Defaults to each init-state EE orientation.",
    )
    parser.add_argument("--ik-frame-name", default=None, help="Optional body/site name used as IK EE frame.")
    parser.add_argument(
        "--ik-frame-type",
        choices=["auto", "body", "site"],
        default="auto",
        help="Frame type for --ik-frame-name, or auto-detect when omitted.",
    )
    parser.add_argument("--ik-max-iters", type=int, default=120)
    parser.add_argument("--ik-damping", type=float, default=0.05)
    parser.add_argument("--ik-position-tol", type=float, default=0.01)
    parser.add_argument("--ik-orientation-tol", type=float, default=0.08)
    parser.add_argument("--ik-max-dq", type=float, default=0.12)
    parser.add_argument("--ik-orientation-weight", type=float, default=0.35)
    parser.add_argument(
        "--max-pose-attempts-per-sample",
        type=int,
        default=50,
        help="Retry budget per desired saved sample when IK/collision/pointcloud filters reject a random pose.",
    )
    parser.add_argument(
        "--collision-margin",
        type=float,
        default=0.0,
        help="Reject robot contacts whose MuJoCo contact distance is <= this margin.",
    )
    parser.add_argument(
        "--robot-collision-name-patterns",
        nargs="*",
        default=list(DEFAULT_ROBOT_COLLISION_PATTERNS),
        help="Geom/body-name substrings used to decide whether a contact involves the robot.",
    )
    parser.add_argument(
        "--reject-robot-self-collisions",
        action="store_true",
        help="Also reject robot-vs-robot contacts. Off by default to avoid gripper/link false positives.",
    )
    parser.add_argument(
        "--robot-keypoint-body-names",
        nargs="*",
        default=list(DEFAULT_ROBOT_KEYPOINT_BODY_NAMES),
        help=(
            "Ordered robot body names saved as whole-arm keypoints in world frame. "
            "Consecutive valid keypoints define diagnostic robot links."
        ),
    )
    parser.add_argument(
        "--robot-keypoint-radius",
        type=float,
        default=0.04,
        help="Radius subtracted from obstacle-to-keypoint distances for d_gt_keypoints.",
    )
    parser.add_argument(
        "--robot-link-radius",
        type=float,
        default=0.04,
        help="Capsule radius subtracted from obstacle-to-link-segment distances for d_gt_links.",
    )
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
        "--allow-missing-target-objects",
        action="store_true",
        help="Allow tasks absent from the target-object map. Off by default for CBF datasets.",
    )
    parser.add_argument(
        "--output",
        default="data/libero_safety/safelibero_goal_cbf_v0.h5",
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
        "--pose-debug-dir",
        default=None,
        help=(
            "Optional structured pose-sampling debug root. Saves ee/rgb, ee/pointcloud, "
            "backview/rgb, backview/pointcloud, fused_pointcloud/rgb, and "
            "fused_pointcloud/pointcloud outputs."
        ),
    )
    parser.add_argument(
        "--pose-debug-max",
        type=int,
        default=20,
        help="Maximum number of accepted pose samples to save in --pose-debug-dir.",
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


def semantic_rgb_from_obs(
    obs: Mapping[str, Any],
    camera_name: str,
    keep_ids: np.ndarray,
    flip_images: bool,
) -> np.ndarray:
    rgb = camera_rgb_from_obs(obs, camera_name, flip_images)
    seg = segmentation_from_obs(obs, camera_name, flip_images)
    mask = np.isin(seg, keep_ids)
    overlay = rgb.copy()
    if np.any(mask):
        color = np.asarray([255, 80, 0], dtype=np.float32)
        overlay[mask] = np.clip(0.55 * overlay[mask].astype(np.float32) + 0.45 * color, 0, 255)
    return overlay.astype(np.uint8)


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


def ee_volume_points_from_metadata(metadata: Mapping[str, Any]) -> np.ndarray:
    if "ee_volume_start_world" not in metadata or "ee_volume_end_world" not in metadata:
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack(
        [
            np.asarray(metadata["ee_volume_start_world"], dtype=np.float32).reshape(3),
            np.asarray(metadata["ee_volume_end_world"], dtype=np.float32).reshape(3),
        ],
        axis=0,
    )


def box_frame_between_points(
    start_world: np.ndarray,
    end_world: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    start = np.asarray(start_world, dtype=np.float32).reshape(3)
    end = np.asarray(end_world, dtype=np.float32).reshape(3)
    axis = end - start
    length = float(np.linalg.norm(axis))
    if length < 1e-8:
        return None

    direction = axis / length
    reference = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(direction, reference))) > 0.95:
        reference = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    basis_u = np.cross(direction, reference)
    basis_u /= max(float(np.linalg.norm(basis_u)), 1e-8)
    basis_v = np.cross(direction, basis_u)
    return direction.astype(np.float32), basis_u.astype(np.float32), basis_v.astype(np.float32), length


def box_corners_between_points(
    start_world: np.ndarray,
    end_world: np.ndarray,
    half_width: float,
) -> np.ndarray:
    start = np.asarray(start_world, dtype=np.float32).reshape(3)
    frame = box_frame_between_points(start_world, end_world)
    if frame is None or float(half_width) <= 0.0:
        return np.zeros((0, 3), dtype=np.float32)
    direction, basis_u, basis_v, length = frame
    corners = []
    for x in (0.0, length):
        center = start + x * direction
        for u in (-float(half_width), float(half_width)):
            for v in (-float(half_width), float(half_width)):
                corners.append(center + u * basis_u + v * basis_v)
    return np.asarray(corners, dtype=np.float32)


def box_surface_points(
    start_world: np.ndarray,
    end_world: np.ndarray,
    half_width: float,
    width_steps: int = 12,
    length_steps: int = 32,
) -> np.ndarray:
    start = np.asarray(start_world, dtype=np.float32).reshape(3)
    end = np.asarray(end_world, dtype=np.float32).reshape(3)
    frame = box_frame_between_points(start, end)
    if frame is None or float(half_width) <= 0.0:
        return np.zeros((0, 3), dtype=np.float32)

    direction, basis_u, basis_v, length = frame
    half = float(half_width)
    along = np.linspace(0.0, length, max(2, int(length_steps)), dtype=np.float32)
    cross = np.linspace(-half, half, max(2, int(width_steps)), dtype=np.float32)
    along_grid, cross_grid = np.meshgrid(along, cross, indexing="ij")
    points = []
    for sign in (-1.0, 1.0):
        points.append(
            start.reshape(1, 1, 3)
            + along_grid[..., None] * direction.reshape(1, 1, 3)
            + sign * half * basis_u.reshape(1, 1, 3)
            + cross_grid[..., None] * basis_v.reshape(1, 1, 3)
        )
        points.append(
            start.reshape(1, 1, 3)
            + along_grid[..., None] * direction.reshape(1, 1, 3)
            + cross_grid[..., None] * basis_u.reshape(1, 1, 3)
            + sign * half * basis_v.reshape(1, 1, 3)
        )

    cap_u, cap_v = np.meshgrid(cross, cross, indexing="ij")
    for center in (start, end):
        points.append(
            center.reshape(1, 1, 3)
            + cap_u[..., None] * basis_u.reshape(1, 1, 3)
            + cap_v[..., None] * basis_v.reshape(1, 1, 3)
        )

    return np.concatenate([point.reshape(-1, 3) for point in points], axis=0).astype(np.float32)


def box_edge_points(
    start_world: np.ndarray,
    end_world: np.ndarray,
    half_width: float,
    steps: int = 48,
) -> np.ndarray:
    corners = box_corners_between_points(start_world, end_world, half_width)
    if corners.shape[0] != 8:
        return np.zeros((0, 3), dtype=np.float32)

    edge_pairs = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    edge_points = []
    t = np.linspace(0.0, 1.0, max(2, int(steps)), dtype=np.float32).reshape(-1, 1)
    for start_idx, end_idx in edge_pairs:
        edge_points.append((1.0 - t) * corners[start_idx].reshape(1, 3) + t * corners[end_idx].reshape(1, 3))
    return np.concatenate(edge_points, axis=0).astype(np.float32)


def plot_box_between_points(
    ax: Any,
    start_world: np.ndarray,
    end_world: np.ndarray,
    half_width: float,
    color: str = "#9467bd",
    alpha: float = 0.22,
) -> None:
    corners = box_corners_between_points(start_world, end_world, half_width)
    if corners.shape[0] != 8:
        return

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    faces = [
        [corners[i] for i in (0, 1, 3, 2)],
        [corners[i] for i in (4, 6, 7, 5)],
        [corners[i] for i in (0, 4, 5, 1)],
        [corners[i] for i in (2, 3, 7, 6)],
        [corners[i] for i in (0, 2, 6, 4)],
        [corners[i] for i in (1, 5, 7, 3)],
    ]
    collection = Poly3DCollection(
        faces,
        facecolors=color,
        edgecolors=color,
        linewidths=1.0,
        alpha=alpha,
    )
    ax.add_collection3d(collection)
    edges = box_edge_points(start_world, end_world, half_width, steps=2).reshape(12, 2, 3)
    for edge in edges:
        ax.plot(edge[:, 0], edge[:, 1], edge[:, 2], c=color, linewidth=1.0, alpha=0.9)
    start = np.asarray(start_world, dtype=np.float32).reshape(3)
    end = np.asarray(end_world, dtype=np.float32).reshape(3)
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        [start[2], end[2]],
        c=color,
        linewidth=2.0,
        label="EE box",
    )


def project_world_points_to_image(
    points_world: np.ndarray,
    env: Any,
    camera_name: str,
    camera_utils: Any,
    image_shape: Tuple[int, int],
    flip_images: bool,
) -> np.ndarray:
    points = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    if points.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int32)

    height, width = int(image_shape[0]), int(image_shape[1])
    intrinsics = camera_utils.get_camera_intrinsic_matrix(env.sim, camera_name, height, width)
    extrinsics = camera_utils.get_camera_extrinsic_matrix(env.sim, camera_name)
    world_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1).T
    camera_points = (np.linalg.inv(extrinsics) @ world_h)[:3]

    # Inverse of camera_cloud_from_obs: its projection path flips local y before world transform.
    pixels_camera = camera_points.copy()
    pixels_camera[1, :] *= -1.0
    in_front = pixels_camera[2, :] > 1e-6
    if not np.any(in_front):
        return np.zeros((0, 2), dtype=np.int32)

    pixel_h = intrinsics @ pixels_camera[:, in_front]
    u = pixel_h[0, :] / pixel_h[2, :]
    v = pixel_h[1, :] / pixel_h[2, :]
    if flip_images:
        v = float(height - 1) - v

    valid = np.isfinite(u) & np.isfinite(v)
    valid &= u >= 0.0
    valid &= u < float(width)
    valid &= v >= 0.0
    valid &= v < float(height)
    if not np.any(valid):
        return np.zeros((0, 2), dtype=np.int32)

    return np.stack([np.rint(u[valid]), np.rint(v[valid])], axis=1).astype(np.int32)


def blend_disc(image: np.ndarray, center_uv: np.ndarray, radius_px: int, color: Sequence[int], alpha: float) -> None:
    height, width = image.shape[:2]
    u, v = int(center_uv[0]), int(center_uv[1])
    radius_px = max(1, int(radius_px))
    u0, u1 = max(0, u - radius_px), min(width - 1, u + radius_px)
    v0, v1 = max(0, v - radius_px), min(height - 1, v + radius_px)
    color_arr = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    for yy in range(v0, v1 + 1):
        for xx in range(u0, u1 + 1):
            if (xx - u) ** 2 + (yy - v) ** 2 <= radius_px**2:
                base = image[yy, xx].astype(np.float32)
                image[yy, xx] = np.clip((1.0 - alpha) * base + alpha * color_arr.reshape(3), 0, 255)


def draw_projected_points(
    image: np.ndarray,
    pixels_uv: np.ndarray,
    color: Sequence[int],
    radius_px: int,
    alpha: float,
) -> np.ndarray:
    output = image.copy()
    for pixel in np.asarray(pixels_uv, dtype=np.int32).reshape(-1, 2):
        blend_disc(output, pixel, radius_px=radius_px, color=color, alpha=alpha)
    return output


def overlay_ee_box_on_rgb(
    rgb: np.ndarray,
    env: Any,
    camera_name: str,
    camera_utils: Any,
    metadata: Mapping[str, Any],
    flip_images: bool,
) -> np.ndarray:
    volume_points = ee_volume_points_from_metadata(metadata)
    if volume_points.shape[0] != 2:
        return rgb
    half_width = float(metadata.get("ee_volume_half_width", metadata.get("ee_volume_radius", 0.0)))
    if half_width <= 0.0:
        return rgb

    surface_world = box_surface_points(volume_points[0], volume_points[1], half_width)
    axis_world = np.linspace(volume_points[0], volume_points[1], 80).astype(np.float32)
    edge_world = box_edge_points(volume_points[0], volume_points[1], half_width)

    surface_uv = project_world_points_to_image(
        surface_world,
        env=env,
        camera_name=camera_name,
        camera_utils=camera_utils,
        image_shape=rgb.shape[:2],
        flip_images=flip_images,
    )
    axis_uv = project_world_points_to_image(
        axis_world,
        env=env,
        camera_name=camera_name,
        camera_utils=camera_utils,
        image_shape=rgb.shape[:2],
        flip_images=flip_images,
    )
    edge_uv = project_world_points_to_image(
        edge_world,
        env=env,
        camera_name=camera_name,
        camera_utils=camera_utils,
        image_shape=rgb.shape[:2],
        flip_images=flip_images,
    )

    overlay = draw_projected_points(rgb, surface_uv, color=(155, 70, 255), radius_px=2, alpha=0.28)
    overlay = draw_projected_points(overlay, edge_uv, color=(210, 170, 255), radius_px=2, alpha=0.65)
    overlay = draw_projected_points(overlay, axis_uv, color=(255, 255, 40), radius_px=2, alpha=0.8)
    return overlay.astype(np.uint8)


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


def write_ascii_ply_with_colors(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if points.shape[0] != colors.shape[0]:
        raise ValueError(f"points/colors length mismatch: {points.shape[0]} vs {colors.shape[0]}")

    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    colors = colors[valid]
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
        for point, color in zip(points, colors):
            r, g, b = [int(x) for x in color]
            f.write(f"{point[0]:.7f} {point[1]:.7f} {point[2]:.7f} {r} {g} {b}\n")


def save_fused_pointcloud_debug_png(
    ee_cloud: CameraCloud,
    backview_cloud: CameraCloud,
    ee_pos_world: np.ndarray,
    output_path: Path,
    metadata: Mapping[str, Any],
    max_points: int,
    rng: np.random.Generator,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ee_points = downsample_points_for_debug(ee_cloud.world, max_points, rng)
    backview_points = downsample_points_for_debug(backview_cloud.world, max_points, rng)
    point_sets = [points for points in (ee_points, backview_points) if points.shape[0] > 0]
    all_points = np.concatenate(point_sets, axis=0) if point_sets else np.zeros((0, 3), dtype=np.float32)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    if ee_points.shape[0] > 0:
        ax.scatter(
            ee_points[:, 0],
            ee_points[:, 1],
            ee_points[:, 2],
            s=1.5,
            c="#1f77b4",
            alpha=0.7,
            label="ee",
        )
    if backview_points.shape[0] > 0:
        ax.scatter(
            backview_points[:, 0],
            backview_points[:, 1],
            backview_points[:, 2],
            s=1.5,
            c="#ff7f0e",
            alpha=0.7,
            label="backview",
        )

    ee = np.asarray(ee_pos_world, dtype=np.float32).reshape(3)
    ax.scatter([ee[0]], [ee[1]], [ee[2]], s=45, c="#d62728", marker="o", label="EE")
    volume_points = ee_volume_points_from_metadata(metadata)
    if volume_points.shape[0] == 2:
        plot_box_between_points(
            ax=ax,
            start_world=volume_points[0],
            end_world=volume_points[1],
            half_width=float(metadata.get("ee_volume_half_width", metadata.get("ee_volume_radius", 0.0))),
        )
    bounds_points = (
        np.concatenate([all_points, volume_points], axis=0)
        if volume_points.shape[0] > 0
        else all_points
    )
    set_axes_equal_3d(ax, bounds_points, ee)
    ax.view_init(elev=24, azim=-55)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("fused pointcloud: ee blue, backview orange")
    ax.legend(loc="upper right")
    fig.suptitle(
        f"sample={metadata['sample_index']} task={metadata['task_id']} "
        f"init={metadata['init_state_id']} step={metadata['rollout_step']}"
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_pose_sampling_debug_sample(
    obs: Mapping[str, Any],
    env: Any,
    camera_utils: Any,
    ee_cloud: CameraCloud,
    backview_cloud: CameraCloud,
    output_dir: Path,
    sample_idx: int,
    metadata: Mapping[str, Any],
    keep_ids: np.ndarray,
    camera_ee: str,
    camera_backview: str,
    flip_images: bool,
    max_points: int,
    rng: np.random.Generator,
) -> None:
    import imageio.v2 as imageio

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"sample_{sample_idx:06d}"

    ee_rgb_path = output_dir / "ee" / "rgb" / f"{stem}.png"
    ee_ply_path = output_dir / "ee" / "pointcloud" / f"{stem}.ply"
    back_rgb_path = output_dir / "backview" / "rgb" / f"{stem}.png"
    back_ply_path = output_dir / "backview" / "pointcloud" / f"{stem}.ply"
    fused_rgb_path = output_dir / "fused_pointcloud" / "rgb" / f"{stem}.png"
    fused_ply_path = output_dir / "fused_pointcloud" / "pointcloud" / f"{stem}.ply"

    ee_rgb_path.parent.mkdir(parents=True, exist_ok=True)
    back_rgb_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(ee_rgb_path, semantic_rgb_from_obs(obs, camera_ee, keep_ids, flip_images))
    backview_rgb = semantic_rgb_from_obs(obs, camera_backview, keep_ids, flip_images)
    backview_rgb = overlay_ee_box_on_rgb(
        rgb=backview_rgb,
        env=env,
        camera_name=camera_backview,
        camera_utils=camera_utils,
        metadata=metadata,
        flip_images=flip_images,
    )
    imageio.imwrite(
        back_rgb_path,
        backview_rgb,
    )
    write_ascii_ply(ee_ply_path, ee_cloud.world, (31, 119, 180))
    write_ascii_ply(back_ply_path, backview_cloud.world, (255, 127, 14))
    fused_points = np.concatenate([ee_cloud.world, backview_cloud.world], axis=0).astype(np.float32)
    fused_colors = np.concatenate(
        [
            np.tile(np.asarray([31, 119, 180], dtype=np.uint8), (ee_cloud.world.shape[0], 1)),
            np.tile(np.asarray([255, 127, 14], dtype=np.uint8), (backview_cloud.world.shape[0], 1)),
        ],
        axis=0,
    )
    write_ascii_ply_with_colors(fused_ply_path, fused_points, fused_colors)
    save_fused_pointcloud_debug_png(
        ee_cloud=ee_cloud,
        backview_cloud=backview_cloud,
        ee_pos_world=np.asarray(metadata["ee_pos_world"], dtype=np.float32),
        output_path=fused_rgb_path,
        metadata=metadata,
        max_points=max_points,
        rng=rng,
    )

    record = dict(metadata)
    record.update(
        {
            "ee_rgb": str(ee_rgb_path.relative_to(output_dir)),
            "ee_pointcloud": str(ee_ply_path.relative_to(output_dir)),
            "backview_rgb": str(back_rgb_path.relative_to(output_dir)),
            "backview_pointcloud": str(back_ply_path.relative_to(output_dir)),
            "fused_pointcloud": str(fused_rgb_path.relative_to(output_dir)),
            "fused_pointcloud_rgb": str(fused_rgb_path.relative_to(output_dir)),
            "fused_pointcloud_ply": str(fused_ply_path.relative_to(output_dir)),
            "rgb_semantic_overlay": True,
            "backview_rgb_ee_box_overlay": True,
            "semantic_keep_ids": [int(x) for x in keep_ids.tolist()],
        }
    )
    with open(output_dir / "metadata.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


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
    ee_quat_world: Optional[np.ndarray] = None,
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
            volume_points = ee_volume_points_from_metadata(metadata)
            if volume_points.shape[0] == 2:
                plot_box_between_points(
                    ax=ax,
                    start_world=volume_points[0],
                    end_world=volume_points[1],
                    half_width=float(metadata.get("ee_volume_half_width", metadata.get("ee_volume_radius", 0.0))),
                )
            rep = np.asarray(v_rep_world, dtype=np.float32).reshape(3)
            ax.quiver(ee[0], ee[1], ee[2], rep[0], rep[1], rep[2], length=0.12, color="#2ca02c")
            if ee_quat_world is not None:
                rot = quat_xyzw_to_matrix(np.asarray(ee_quat_world, dtype=np.float32))
                for axis_idx, axis_color in enumerate(("#d62728", "#2ca02c", "#1f77b4")):
                    direction = rot[:, axis_idx]
                    ax.quiver(
                        ee[0],
                        ee[1],
                        ee[2],
                        direction[0],
                        direction[1],
                        direction[2],
                        length=0.08,
                        color=axis_color,
                        alpha=0.9,
                    )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        volume_points = ee_volume_points_from_metadata(metadata) if ee_center is not None else np.zeros((0, 3))
        bounds_points = (
            np.concatenate([points, volume_points], axis=0)
            if volume_points.shape[0] > 0
            else points
        )
        set_axes_equal_3d(ax, bounds_points, ee_center)
        ax.view_init(elev=24, azim=-55)

    fig.suptitle(
        f"task={metadata['task_id']} init={metadata['init_state_id']} "
        f"step={metadata['rollout_step']} S={metadata['S_star_obs']:.3f} "
        f"d={metadata['d_obs']:.3f} hard={metadata['d_obs_hard']:.3f}"
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


def sample_or_pad_cloud_with_mask(
    cloud: CameraCloud,
    n_points: int,
    rng: np.random.Generator,
    empty_local: Sequence[float],
    empty_world: Sequence[float],
) -> Tuple[CameraCloud, np.ndarray]:
    count = int(cloud.world.shape[0])
    if count >= n_points:
        indices = rng.choice(count, size=n_points, replace=False)
        sample = CameraCloud(
            local=cloud.local[indices].astype(np.float32),
            world=cloud.world[indices].astype(np.float32),
            instance_ids=cloud.instance_ids[indices].astype(np.int32),
        )
        return sample, np.ones((n_points,), dtype=np.uint8)

    if count > 0:
        pad = rng.choice(count, size=n_points - count, replace=True)
        indices = np.concatenate([np.arange(count), pad])
        valid_mask = np.concatenate(
            [
                np.ones((count,), dtype=np.uint8),
                np.zeros((n_points - count,), dtype=np.uint8),
            ],
            axis=0,
        )
        sample = CameraCloud(
            local=cloud.local[indices].astype(np.float32),
            world=cloud.world[indices].astype(np.float32),
            instance_ids=cloud.instance_ids[indices].astype(np.int32),
        )
        return sample, valid_mask

    sample = CameraCloud(
        local=np.tile(np.asarray(empty_local, dtype=np.float32).reshape(1, 3), (n_points, 1)),
        world=np.tile(np.asarray(empty_world, dtype=np.float32).reshape(1, 3), (n_points, 1)),
        instance_ids=np.full((n_points,), -1, dtype=np.int32),
    )
    return sample, np.zeros((n_points,), dtype=np.uint8)


def camera_position_world(env: Any, camera_name: str) -> np.ndarray:
    camera_id = int(env.sim.model.camera_name2id(camera_name))
    return np.asarray(env.sim.data.cam_xpos[camera_id], dtype=np.float32).reshape(3)


def point_to_oriented_box_vectors_and_distances(
    points_world: np.ndarray,
    box_start_world: np.ndarray,
    box_end_world: np.ndarray,
    half_width: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    start = np.asarray(box_start_world, dtype=np.float32).reshape(3)
    end = np.asarray(box_end_world, dtype=np.float32).reshape(3)
    half = max(0.0, float(half_width))
    frame = box_frame_between_points(start, end)
    if frame is None or half <= 0.0:
        delta = end.reshape(1, 3) - points
        distances = np.linalg.norm(delta, axis=1).astype(np.float32)
        closest = np.repeat(end.reshape(1, 3), points.shape[0], axis=0)
        return (
            delta.astype(np.float32),
            distances,
            closest.astype(np.float32),
            delta.astype(np.float32),
        )

    direction, basis_u, basis_v, length = frame
    rel = points - start.reshape(1, 3)
    local_x = rel @ direction
    local_y = rel @ basis_u
    local_z = rel @ basis_v

    closest_x = np.clip(local_x, 0.0, length)
    closest_y = np.clip(local_y, -half, half)
    closest_z = np.clip(local_z, -half, half)
    closest = (
        start.reshape(1, 3)
        + closest_x.reshape(-1, 1) * direction.reshape(1, 3)
        + closest_y.reshape(-1, 1) * basis_u.reshape(1, 3)
        + closest_z.reshape(-1, 1) * basis_v.reshape(1, 3)
    )
    vectors = closest - points
    grad_h_vectors = vectors.copy()
    signed_distances = np.linalg.norm(vectors, axis=1)

    inside = (
        (local_x >= 0.0)
        & (local_x <= length)
        & (local_y >= -half)
        & (local_y <= half)
        & (local_z >= -half)
        & (local_z <= half)
    )
    if np.any(inside):
        local = np.stack([local_x, local_y, local_z], axis=1)
        face_distances = np.stack(
            [
                local[:, 0],
                length - local[:, 0],
                local[:, 1] + half,
                half - local[:, 1],
                local[:, 2] + half,
                half - local[:, 2],
            ],
            axis=1,
        )
        nearest_face = np.argmin(face_distances[inside], axis=1)
        inside_indices = np.flatnonzero(inside)
        inside_vectors = np.zeros((inside_indices.shape[0], 3), dtype=np.float32)
        inside_penetration_depth = np.zeros((inside_indices.shape[0],), dtype=np.float32)
        face_dirs = np.asarray(
            [
                -direction,
                direction,
                -basis_u,
                basis_u,
                -basis_v,
                basis_v,
            ],
            dtype=np.float32,
        )
        for local_idx, face_idx in enumerate(nearest_face):
            face_distance = float(face_distances[inside_indices[local_idx], int(face_idx)])
            inside_vectors[local_idx] = face_dirs[int(face_idx)] * face_distances[
                inside_indices[local_idx], int(face_idx)
            ]
            inside_penetration_depth[local_idx] = face_distance
        vectors[inside_indices] = inside_vectors
        grad_h_vectors[inside_indices] = -inside_vectors
        closest[inside_indices] = points[inside_indices] + inside_vectors
        signed_distances[inside_indices] = -inside_penetration_depth

    return (
        vectors.astype(np.float32),
        signed_distances.astype(np.float32),
        closest.astype(np.float32),
        grad_h_vectors.astype(np.float32),
    )


def compute_safety_targets(
    points_world: np.ndarray,
    ee_pos_world: np.ndarray,
    d_max: float,
    d_min: float,
    ee_radius: float,
    ee_safety_geometry: str = "eef_sphere",
    ee_volume_start_world: Optional[np.ndarray] = None,
    ee_volume_end_world: Optional[np.ndarray] = None,
    safety_distance_mode: str = "hard",
    safety_knn_k: int = 8,
    safety_robust_cap: float = 0.02,
    source_camera: Optional[np.ndarray] = None,
) -> SafetyTargets:
    if points_world.shape[0] == 0:
        return SafetyTargets(
            S_star_obs=np.asarray([1.0], dtype=np.float32),
            S_star_hard=np.asarray([1.0], dtype=np.float32),
            S_star_knn=np.asarray([1.0], dtype=np.float32),
            S_star_robust=np.asarray([1.0], dtype=np.float32),
            d_obs=np.asarray([1e6], dtype=np.float32),
            d_obs_hard=np.asarray([1e6], dtype=np.float32),
            d_obs_knn=np.asarray([1e6], dtype=np.float32),
            d_obs_robust=np.asarray([1e6], dtype=np.float32),
            v_rep=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            v_rep_knn=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            closest_point_idx=np.asarray([-1], dtype=np.int32),
            closest_raw_point_idx=np.asarray([-1], dtype=np.int32),
            closest_point_world=np.asarray([np.nan, np.nan, np.nan], dtype=np.float32),
            closest_volume_point_world=np.asarray([np.nan, np.nan, np.nan], dtype=np.float32),
            closest_point_source=np.asarray([-1], dtype=np.int8),
        )

    if ee_safety_geometry == "camera_to_eef_box":
        if ee_volume_start_world is None or ee_volume_end_world is None:
            raise ValueError("camera_to_eef_box requires EE volume start and end positions.")
        delta, surface_distances, closest_volume_points, grad_h_vectors = (
            point_to_oriented_box_vectors_and_distances(
                points_world=points_world,
                box_start_world=ee_volume_start_world,
                box_end_world=ee_volume_end_world,
                half_width=float(ee_radius),
            )
        )
    elif ee_safety_geometry == "eef_sphere":
        ee = ee_pos_world.reshape(1, 3).astype(np.float32)
        points = points_world.astype(np.float32)
        delta_to_center = ee - points
        center_distances = np.linalg.norm(delta_to_center, axis=1)
        safe_norm = np.maximum(center_distances.reshape(-1, 1), 1e-8)
        directions_to_center = delta_to_center / safe_norm
        surface_distances = center_distances - float(ee_radius)
        closest_volume_points = ee - directions_to_center * float(ee_radius)
        delta = closest_volume_points - points
        grad_h_vectors = delta.copy()
        inside = surface_distances < 0.0
        if np.any(inside):
            grad_h_vectors[inside] = -delta[inside]
    else:
        raise ValueError(f"Unknown EE safety geometry: {ee_safety_geometry}")

    closest_idx = int(np.argmin(surface_distances))

    d_obs_hard = float(surface_distances[closest_idx])
    knn_k = min(max(1, int(safety_knn_k)), int(surface_distances.shape[0]))
    nearest_indices = np.argpartition(surface_distances, knn_k - 1)[:knn_k]
    nearest_surface = surface_distances[nearest_indices]
    d_obs_knn = float(np.mean(nearest_surface))
    d_obs_robust = min(d_obs_knn, d_obs_hard + max(0.0, float(safety_robust_cap)))
    if safety_distance_mode == "hard":
        d_obs_selected = d_obs_hard
    elif safety_distance_mode == "knn":
        d_obs_selected = d_obs_knn
    elif safety_distance_mode == "knn_capped":
        d_obs_selected = d_obs_robust
    else:
        raise ValueError(f"Unknown safety distance mode: {safety_distance_mode}")

    lower_s = float(d_min) / float(d_max)
    s_star_hard = float(np.clip(d_obs_hard / float(d_max), lower_s, 1.0))
    s_star_knn = float(np.clip(d_obs_knn / float(d_max), lower_s, 1.0))
    s_star_robust = float(np.clip(d_obs_robust / float(d_max), lower_s, 1.0))
    s_star_obs = float(np.clip(d_obs_selected / float(d_max), lower_s, 1.0))

    rep = grad_h_vectors[closest_idx]
    norm = float(np.linalg.norm(rep))
    if norm < 1e-8:
        rep = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        rep = (rep / norm).astype(np.float32)

    nearest_vectors = grad_h_vectors[nearest_indices].astype(np.float32)
    nearest_norms = np.linalg.norm(nearest_vectors, axis=1)
    valid_knn = nearest_norms > 1e-8
    if np.any(valid_knn):
        nearest_dirs = nearest_vectors[valid_knn] / nearest_norms[valid_knn].reshape(-1, 1)
        # Smooth length scale keeps v_rep_knn from collapsing to the single closest point.
        weight_floor = max(float(safety_robust_cap), 1e-4)
        weights = 1.0 / (np.abs(nearest_surface[valid_knn]).astype(np.float32) + weight_floor)
        weights = weights / np.maximum(float(np.sum(weights)), 1e-8)
        rep_knn = np.sum(nearest_dirs * weights.reshape(-1, 1), axis=0)
        rep_knn_norm = float(np.linalg.norm(rep_knn))
        if rep_knn_norm < 1e-8:
            rep_knn = rep.astype(np.float32)
        else:
            rep_knn = (rep_knn / rep_knn_norm).astype(np.float32)
    else:
        rep_knn = rep.astype(np.float32)

    if source_camera is None or np.asarray(source_camera).shape[0] != points_world.shape[0]:
        closest_source = -1
    else:
        closest_source = int(np.asarray(source_camera, dtype=np.int8).reshape(-1)[closest_idx])

    return SafetyTargets(
        S_star_obs=np.asarray([s_star_obs], dtype=np.float32),
        S_star_hard=np.asarray([s_star_hard], dtype=np.float32),
        S_star_knn=np.asarray([s_star_knn], dtype=np.float32),
        S_star_robust=np.asarray([s_star_robust], dtype=np.float32),
        d_obs=np.asarray([d_obs_selected], dtype=np.float32),
        d_obs_hard=np.asarray([d_obs_hard], dtype=np.float32),
        d_obs_knn=np.asarray([d_obs_knn], dtype=np.float32),
        d_obs_robust=np.asarray([d_obs_robust], dtype=np.float32),
        v_rep=rep.astype(np.float32),
        v_rep_knn=rep_knn.astype(np.float32),
        closest_point_idx=np.asarray([closest_idx], dtype=np.int32),
        closest_raw_point_idx=np.asarray([closest_idx], dtype=np.int32),
        closest_point_world=points_world[closest_idx].astype(np.float32),
        closest_volume_point_world=closest_volume_points[closest_idx].astype(np.float32),
        closest_point_source=np.asarray([closest_source], dtype=np.int8),
    )


def compute_cbf_targets(
    safety_targets: SafetyTargets,
    safety_margin: float,
    h_scale: float,
) -> Dict[str, np.ndarray]:
    margin = float(safety_margin)
    scale = max(float(h_scale), 1e-8)
    h_star = safety_targets.d_obs - margin
    h_star_hard = safety_targets.d_obs_hard - margin
    h_star_knn = safety_targets.d_obs_knn - margin
    h_star_robust = safety_targets.d_obs_robust - margin
    return {
        "h_star": h_star.astype(np.float32),
        "h_star_hard": h_star_hard.astype(np.float32),
        "h_star_knn": h_star_knn.astype(np.float32),
        "h_star_robust": h_star_robust.astype(np.float32),
        "h_star_norm": np.clip(h_star / scale, -1.0, 1.0).astype(np.float32),
        "h_star_hard_norm": np.clip(h_star_hard / scale, -1.0, 1.0).astype(np.float32),
        "h_star_knn_norm": np.clip(h_star_knn / scale, -1.0, 1.0).astype(np.float32),
        "h_star_robust_norm": np.clip(h_star_robust / scale, -1.0, 1.0).astype(np.float32),
    }


def classify_safety_region(h_star: np.ndarray, boundary_band: float) -> int:
    h_value = float(np.asarray(h_star, dtype=np.float32).reshape(-1)[0])
    if h_value < 0.0:
        return 2
    if h_value < max(float(boundary_band), 0.0):
        return 1
    return 0


def target_safety_region_counts(args: argparse.Namespace) -> Dict[int, int]:
    n_total = int(args.samples_per_init_state)
    fractions = np.asarray(
        [
            max(0.0, float(args.target_safe_far_fraction)),
            max(0.0, float(args.target_near_boundary_fraction)),
            max(0.0, float(args.target_unsafe_fraction)),
        ],
        dtype=np.float64,
    )
    fraction_sum = float(np.sum(fractions))
    if fraction_sum <= 0.0:
        raise ValueError("At least one target safety-region fraction must be positive.")
    fractions = fractions / fraction_sum

    counts = {
        0: int(round(n_total * float(fractions[0]))),
        1: int(round(n_total * float(fractions[1]))),
        2: int(round(n_total * float(fractions[2]))),
    }
    diff = n_total - sum(counts.values())
    counts[1] += diff
    return counts


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


def normalize_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        raise ValueError("Quaternion norm is zero")
    return quat / norm


def quat_xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = normalize_quat(quat)
    return np.asarray([quat[3], quat[0], quat[1], quat[2]], dtype=np.float64)


def quat_wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = normalize_quat(quat)
    return np.asarray([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)


def quat_conjugate_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = normalize_quat(quat)
    return np.asarray([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)


def quat_multiply_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = normalize_quat(lhs)
    w2, x2, y2, z2 = normalize_quat(rhs)
    return normalize_quat(
        np.asarray(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float64,
        )
    )


def quat_multiply_xyzw(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    return quat_wxyz_to_xyzw(
        quat_multiply_wxyz(quat_xyzw_to_wxyz(lhs), quat_xyzw_to_wxyz(rhs))
    )


def euler_xyz_to_quat_xyzw(euler_xyz: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(euler_xyz, dtype=np.float64).reshape(3)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return quat_wxyz_to_xyzw(np.asarray([w, x, y, z], dtype=np.float64))


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quat(quat)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    return quat_wxyz_to_matrix(quat_xyzw_to_wxyz(quat))


def matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return normalize_quat(np.asarray([w, x, y, z], dtype=np.float64))


def orientation_error_wxyz(target_quat: np.ndarray, current_quat: np.ndarray) -> np.ndarray:
    q_err = quat_multiply_wxyz(target_quat, quat_conjugate_wxyz(current_quat))
    if q_err[0] < 0.0:
        q_err = -q_err
    return (2.0 * q_err[1:]).astype(np.float64)


def quaternion_angle_error_wxyz(target_quat: np.ndarray, current_quat: np.ndarray) -> float:
    target = normalize_quat(target_quat)
    current = normalize_quat(current_quat)
    dot = float(np.clip(abs(np.dot(target, current)), -1.0, 1.0))
    return float(2.0 * np.arccos(dot))


def sample_random_ee_pose(
    rng: np.random.Generator,
    pose_workspace_min: np.ndarray,
    pose_workspace_max: np.ndarray,
    reference_quat_xyzw: np.ndarray,
    orientation_delta_range_deg: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_pos = rng.uniform(pose_workspace_min, pose_workspace_max).astype(np.float64)
    delta_range = np.deg2rad(np.asarray(orientation_delta_range_deg, dtype=np.float64).reshape(3))
    delta_euler = rng.uniform(-delta_range, delta_range)
    delta_quat = euler_xyz_to_quat_xyzw(delta_euler)
    target_quat = quat_multiply_xyzw(reference_quat_xyzw, delta_quat)
    return target_pos.astype(np.float32), target_quat.astype(np.float32), delta_euler.astype(np.float32)


def sample_near_boundary_ee_pose(
    rng: np.random.Generator,
    obstacle_points_world: np.ndarray,
    pose_workspace_min: np.ndarray,
    pose_workspace_max: np.ndarray,
    reference_quat_xyzw: np.ndarray,
    orientation_delta_range_deg: Sequence[float],
    ee_radius: float,
    safety_margin: float,
    boundary_band: float,
    near_boundary_min_dist: float,
    near_boundary_max_dist: Optional[float],
    near_boundary_max_tries: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    points = np.asarray(obstacle_points_world, dtype=np.float32).reshape(-1, 3)
    if points.shape[0] == 0:
        raise ValueError("near-boundary sampling requires at least one obstacle point.")

    max_dist = (
        float(near_boundary_max_dist)
        if near_boundary_max_dist is not None
        else float(safety_margin) + max(0.0, float(boundary_band))
    )
    min_dist = max(0.0, float(near_boundary_min_dist))
    max_dist = max(min_dist, max_dist)
    workspace_min = np.asarray(pose_workspace_min, dtype=np.float32)
    workspace_max = np.asarray(pose_workspace_max, dtype=np.float32)
    max_tries = max(1, int(near_boundary_max_tries))

    accepted: Optional[Tuple[np.ndarray, int, np.ndarray, np.ndarray, float, int]] = None
    for try_idx in range(max_tries):
        obstacle_idx = int(rng.integers(0, points.shape[0]))
        obstacle_point = points[obstacle_idx]
        direction = rng.normal(size=3).astype(np.float32)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-8:
            direction = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            direction = direction / norm

        requested_clearance = float(rng.uniform(min_dist, max_dist))
        target_pos = obstacle_point + (float(ee_radius) + requested_clearance) * direction
        if np.all(target_pos >= workspace_min) and np.all(target_pos <= workspace_max):
            accepted = (
                target_pos.astype(np.float32),
                obstacle_idx,
                obstacle_point.astype(np.float32),
                direction.astype(np.float32),
                requested_clearance,
                try_idx,
            )
            break

    target_pos, target_quat, delta_euler = sample_random_ee_pose(
        rng=rng,
        pose_workspace_min=pose_workspace_min,
        pose_workspace_max=pose_workspace_max,
        reference_quat_xyzw=reference_quat_xyzw,
        orientation_delta_range_deg=orientation_delta_range_deg,
    )
    if accepted is None:
        debug = {
            "near_boundary_fallback_random_pose": True,
            "near_boundary_accept_try": -1,
            "near_boundary_max_tries": max_tries,
            "near_boundary_obstacle_idx": -1,
            "near_boundary_obstacle_point_world": [float("nan"), float("nan"), float("nan")],
            "near_boundary_direction_world": [float("nan"), float("nan"), float("nan")],
            "near_boundary_requested_clearance": float("nan"),
            "near_boundary_min_dist": min_dist,
            "near_boundary_max_dist": max_dist,
        }
        return target_pos.astype(np.float32), target_quat.astype(np.float32), delta_euler, debug

    target_pos, obstacle_idx, obstacle_point, direction, requested_clearance, try_idx = accepted
    debug = {
        "near_boundary_fallback_random_pose": False,
        "near_boundary_accept_try": int(try_idx),
        "near_boundary_max_tries": max_tries,
        "near_boundary_obstacle_idx": obstacle_idx,
        "near_boundary_obstacle_point_world": obstacle_point.tolist(),
        "near_boundary_direction_world": direction.tolist(),
        "near_boundary_requested_clearance": requested_clearance,
        "near_boundary_min_dist": min_dist,
        "near_boundary_max_dist": max_dist,
    }
    return target_pos.astype(np.float32), target_quat.astype(np.float32), delta_euler, debug


def unwrap_env(env: Any) -> Any:
    return env.env if hasattr(env, "env") else env


def table_full_size_for_env(env: Any) -> Optional[np.ndarray]:
    task_env = unwrap_env(env)
    arena_type = str(getattr(task_env, "_arena_type", ""))
    candidates = {
        "table": "table_full_size",
        "kitchen": "kitchen_table_full_size",
        "study": "study_table_full_size",
        "coffee_table": "coffee_table_full_size",
        "living_room": "living_room_table_full_size",
    }
    names = [candidates[arena_type]] if arena_type in candidates else []
    names += [
        "table_full_size",
        "kitchen_table_full_size",
        "study_table_full_size",
        "coffee_table_full_size",
        "living_room_table_full_size",
    ]
    for name in names:
        if hasattr(task_env, name):
            full_size = np.asarray(getattr(task_env, name), dtype=np.float32).reshape(-1)
            if full_size.shape[0] >= 2 and np.all(full_size[:2] > 0.0):
                return full_size[:3] if full_size.shape[0] >= 3 else full_size[:2]
    return None


def table_xy_bounds_from_attrs(env: Any, margin: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    task_env = unwrap_env(env)
    full_size = table_full_size_for_env(task_env)
    if full_size is None:
        return None
    center = np.asarray(getattr(task_env, "workspace_offset", [0.0, 0.0, 0.0]), dtype=np.float32)
    half_xy = 0.5 * full_size[:2].astype(np.float32)
    inset = np.minimum(np.maximum(float(margin), 0.0), np.maximum(half_xy - 1e-4, 0.0))
    xy_min = center[:2] - half_xy + inset
    xy_max = center[:2] + half_xy - inset
    if np.any(xy_max <= xy_min):
        return None
    return xy_min.astype(np.float32), xy_max.astype(np.float32)


def table_xy_bounds_from_sim(env: Any, margin: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    sim = env.sim
    model = sim.model
    data = sim.data
    for geom_name in ("table_collision", "table_visual"):
        try:
            geom_id = int(model.geom_name2id(geom_name))
        except Exception:
            continue
        try:
            center = np.asarray(data.geom_xpos[geom_id], dtype=np.float32)
            half_xy = np.asarray(model.geom_size[geom_id][:2], dtype=np.float32)
        except Exception:
            continue
        if half_xy.shape[0] != 2 or np.any(half_xy <= 0.0):
            continue
        inset = np.minimum(np.maximum(float(margin), 0.0), np.maximum(half_xy - 1e-4, 0.0))
        xy_min = center[:2] - half_xy + inset
        xy_max = center[:2] + half_xy - inset
        if np.all(xy_max > xy_min):
            return xy_min.astype(np.float32), xy_max.astype(np.float32)
    return None


def pose_workspace_for_env(
    env: Any,
    args: argparse.Namespace,
    default_pose_min: np.ndarray,
    default_pose_max: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    pose_min = np.asarray(default_pose_min, dtype=np.float32).copy()
    pose_max = np.asarray(default_pose_max, dtype=np.float32).copy()
    if args.pose_xy_source == "box":
        return pose_min, pose_max

    bounds = table_xy_bounds_from_attrs(env, margin=float(args.pose_table_margin))
    if bounds is None:
        bounds = table_xy_bounds_from_sim(env, margin=float(args.pose_table_margin))
    if bounds is None:
        print("[dataset] warning: could not infer tabletop XY bounds; using pose workspace box")
        return pose_min, pose_max

    pose_min[:2], pose_max[:2] = bounds
    return pose_min, pose_max


def model_body_exists(model: Any, name: str) -> bool:
    try:
        model.body_name2id(name)
        return True
    except Exception:
        return False


def model_site_exists(model: Any, name: str) -> bool:
    try:
        model.site_name2id(name)
        return True
    except Exception:
        return False


def resolve_ik_frame(env: Any, args: argparse.Namespace) -> IKFrame:
    model = env.sim.model
    if args.ik_frame_name:
        if args.ik_frame_type in ("auto", "body") and model_body_exists(model, args.ik_frame_name):
            return IKFrame(kind="body", name=str(args.ik_frame_name))
        if args.ik_frame_type in ("auto", "site") and model_site_exists(model, args.ik_frame_name):
            return IKFrame(kind="site", name=str(args.ik_frame_name))
        raise ValueError(
            f"Could not find IK frame {args.ik_frame_name!r} as {args.ik_frame_type}."
        )

    for name in ("gripper0_eef", "eef_marker", "robot0_eef", "right_hand"):
        if model_body_exists(model, name):
            return IKFrame(kind="body", name=name)
    for name in ("gripper0_grip_site", "gripper0_ft_frame", "robot0_eef_site", "eef_site"):
        if model_site_exists(model, name):
            return IKFrame(kind="site", name=name)

    raise ValueError(
        "Could not auto-detect an IK EE frame. Pass --ik-frame-name and --ik-frame-type."
    )


def frame_pose_wxyz(env: Any, frame: IKFrame) -> Tuple[np.ndarray, np.ndarray]:
    sim = env.sim
    if frame.kind == "body":
        body_id = sim.model.body_name2id(frame.name)
        try:
            pos = np.asarray(sim.data.get_body_xpos(frame.name), dtype=np.float64)
        except Exception:
            pos = np.asarray(sim.data.body_xpos[body_id], dtype=np.float64)
        try:
            quat = np.asarray(sim.data.get_body_xquat(frame.name), dtype=np.float64)
        except Exception:
            quat = np.asarray(sim.data.body_xquat[body_id], dtype=np.float64)
        return pos.reshape(3), normalize_quat(quat)

    site_id = sim.model.site_name2id(frame.name)
    try:
        pos = np.asarray(sim.data.get_site_xpos(frame.name), dtype=np.float64)
    except Exception:
        pos = np.asarray(sim.data.site_xpos[site_id], dtype=np.float64)
    try:
        xmat = np.asarray(sim.data.get_site_xmat(frame.name), dtype=np.float64).reshape(3, 3)
    except Exception:
        xmat = np.asarray(sim.data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
    return pos.reshape(3), matrix_to_quat_wxyz(xmat)


def frame_jacobian(env: Any, frame: IKFrame, qvel_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if frame.kind == "body":
        jacp = env.sim.data.get_body_jacp(frame.name).reshape(3, -1)
        jacr = env.sim.data.get_body_jacr(frame.name).reshape(3, -1)
    else:
        jacp = env.sim.data.get_site_jacp(frame.name).reshape(3, -1)
        jacr = env.sim.data.get_site_jacr(frame.name).reshape(3, -1)
    return jacp[:, qvel_indices].astype(np.float64), jacr[:, qvel_indices].astype(np.float64)


def robot_joint_indices(env: Any) -> Tuple[np.ndarray, np.ndarray]:
    robot = env.robots[0]
    qpos_indices = getattr(robot, "_ref_joint_pos_indexes", None)
    qvel_indices = getattr(robot, "_ref_joint_vel_indexes", None)
    if qpos_indices is None or qvel_indices is None:
        raise AttributeError("Could not find robosuite robot joint qpos/qvel indexes.")

    qpos = np.asarray(qpos_indices, dtype=np.int64).reshape(-1)[:7]
    qvel = np.asarray(qvel_indices, dtype=np.int64).reshape(-1)[:7]
    if qpos.shape[0] != 7 or qvel.shape[0] != 7:
        raise ValueError(f"Expected 7 Panda arm indexes, got qpos={qpos}, qvel={qvel}")
    return qpos, qvel


def robot_joint_limits(env: Any, qpos_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(
        [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
        dtype=np.float64,
    )
    upper = np.asarray(
        [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
        dtype=np.float64,
    )

    robot = env.robots[0]
    joint_names = getattr(robot, "robot_joints", None)
    if joint_names is None or len(joint_names) == 0:
        joint_names = getattr(robot, "joint_names", None)
    if joint_names is not None and len(joint_names) > 0:
        resolved_lower = lower.copy()
        resolved_upper = upper.copy()
        for idx, joint_name in enumerate(list(joint_names)[: len(qpos_indices)]):
            try:
                joint_id = env.sim.model.joint_name2id(joint_name)
                limited = np.asarray(env.sim.model.jnt_limited).reshape(-1)
                if int(limited[joint_id]) == 1:
                    resolved_lower[idx], resolved_upper[idx] = env.sim.model.jnt_range[joint_id]
            except Exception:
                continue
        return resolved_lower, resolved_upper

    return lower, upper


def set_robot_qpos(
    env: Any,
    q: np.ndarray,
    qpos_indices: np.ndarray,
    qvel_indices: np.ndarray,
) -> None:
    env.sim.data.qpos[qpos_indices] = np.asarray(q, dtype=np.float64).reshape(7)
    env.sim.data.qvel[qvel_indices] = 0.0
    env.sim.forward()


def solve_ik_to_pose(
    env: Any,
    target_pos: np.ndarray,
    target_quat_xyzw: np.ndarray,
    frame: IKFrame,
    qpos_indices: np.ndarray,
    qvel_indices: np.ndarray,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    args: argparse.Namespace,
) -> IKResult:
    target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
    target_quat_wxyz = quat_xyzw_to_wxyz(target_quat_xyzw)
    q = np.asarray(env.sim.data.qpos[qpos_indices], dtype=np.float64).reshape(7).copy()

    max_iters = max(1, int(args.ik_max_iters))
    damping = max(1e-8, float(args.ik_damping))
    max_dq = max(1e-8, float(args.ik_max_dq))
    ori_weight = max(0.0, float(args.ik_orientation_weight))

    pos_error_norm = np.inf
    ori_error_norm = np.inf
    iteration = 0
    for iteration in range(1, max_iters + 1):
        set_robot_qpos(env, q, qpos_indices, qvel_indices)
        current_pos, current_quat_wxyz = frame_pose_wxyz(env, frame)

        pos_error = target_pos - current_pos
        ori_error = orientation_error_wxyz(target_quat_wxyz, current_quat_wxyz)
        pos_error_norm = float(np.linalg.norm(pos_error))
        ori_error_norm = quaternion_angle_error_wxyz(target_quat_wxyz, current_quat_wxyz)
        if (
            pos_error_norm <= float(args.ik_position_tol)
            and ori_error_norm <= float(args.ik_orientation_tol)
        ):
            return IKResult(True, q.astype(np.float32), pos_error_norm, ori_error_norm, iteration)

        jacp, jacr = frame_jacobian(env, frame, qvel_indices)
        jac = np.vstack([jacp, ori_weight * jacr])
        error = np.concatenate([pos_error, ori_weight * ori_error], axis=0)
        lhs = jac @ jac.T + (damping**2) * np.eye(6, dtype=np.float64)
        try:
            dq = jac.T @ np.linalg.solve(lhs, error)
        except np.linalg.LinAlgError:
            dq = jac.T @ np.linalg.lstsq(lhs, error, rcond=None)[0]

        dq_norm_inf = float(np.max(np.abs(dq)))
        if dq_norm_inf > max_dq:
            dq *= max_dq / dq_norm_inf
        q = np.clip(q + dq, joint_lower, joint_upper)

    set_robot_qpos(env, q, qpos_indices, qvel_indices)
    current_pos, current_quat_wxyz = frame_pose_wxyz(env, frame)
    pos_error_norm = float(np.linalg.norm(target_pos - current_pos))
    ori_error_norm = quaternion_angle_error_wxyz(target_quat_wxyz, current_quat_wxyz)
    success = (
        pos_error_norm <= float(args.ik_position_tol)
        and ori_error_norm <= float(args.ik_orientation_tol)
    )
    return IKResult(success, q.astype(np.float32), pos_error_norm, ori_error_norm, iteration)


def model_geom_name(model: Any, geom_id: int) -> str:
    try:
        name = model.geom_id2name(int(geom_id))
    except Exception:
        name = None
    return str(name or "")


def model_body_name_from_geom(model: Any, geom_id: int) -> str:
    try:
        body_id = int(model.geom_bodyid[int(geom_id)])
        name = model.body_id2name(body_id)
    except Exception:
        name = None
    return str(name or "")


def geom_or_body_matches(model: Any, geom_id: int, patterns: Sequence[str]) -> bool:
    text = f"{model_geom_name(model, geom_id)} {model_body_name_from_geom(model, geom_id)}"
    return name_matches(text, patterns)


def robot_collision_contacts(
    env: Any,
    robot_patterns: Sequence[str],
    margin: float,
    reject_robot_self_collisions: bool,
) -> List[Dict[str, Any]]:
    model = env.sim.model
    data = env.sim.data
    collisions: List[Dict[str, Any]] = []
    for contact_idx in range(int(getattr(data, "ncon", 0))):
        contact = data.contact[contact_idx]
        if float(contact.dist) > float(margin):
            continue
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        geom1_is_robot = geom_or_body_matches(model, geom1, robot_patterns)
        geom2_is_robot = geom_or_body_matches(model, geom2, robot_patterns)
        if not (geom1_is_robot or geom2_is_robot):
            continue
        if geom1_is_robot and geom2_is_robot and not reject_robot_self_collisions:
            continue
        collisions.append(
            {
                "geom1": model_geom_name(model, geom1),
                "body1": model_body_name_from_geom(model, geom1),
                "geom2": model_geom_name(model, geom2),
                "body2": model_body_name_from_geom(model, geom2),
                "dist": float(contact.dist),
            }
        )
    return collisions


def robot_keypoint_positions_world(
    env: Any,
    body_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    model = env.sim.model
    data = env.sim.data
    points = np.full((len(body_names), 3), np.nan, dtype=np.float32)
    valid = np.zeros((len(body_names),), dtype=np.uint8)
    for idx, body_name in enumerate(body_names):
        try:
            body_id = int(model.body_name2id(str(body_name)))
            points[idx] = np.asarray(data.body_xpos[body_id], dtype=np.float32).reshape(3)
            valid[idx] = 1
        except Exception:
            continue
    return points, valid


def point_to_segment_distances(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    start = np.asarray(start, dtype=np.float32).reshape(3)
    end = np.asarray(end, dtype=np.float32).reshape(3)
    axis = end - start
    denom = float(np.dot(axis, axis))
    if denom < 1e-12:
        return np.linalg.norm(points - start.reshape(1, 3), axis=1).astype(np.float32)
    t = ((points - start.reshape(1, 3)) @ axis.reshape(3, 1)).reshape(-1) / denom
    t = np.clip(t, 0.0, 1.0)
    closest = start.reshape(1, 3) + t.reshape(-1, 1) * axis.reshape(1, 3)
    return np.linalg.norm(points - closest, axis=1).astype(np.float32)


def compute_robot_geometry_targets(
    env: Any,
    points_world: np.ndarray,
    body_names: Sequence[str],
    keypoint_radius: float,
    link_radius: float,
) -> RobotGeometryTargets:
    robot_points, keypoint_valid = robot_keypoint_positions_world(env, body_names)
    points = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)

    d_keypoints = np.full((robot_points.shape[0],), np.nan, dtype=np.float32)
    if points.shape[0] > 0:
        for idx, keypoint in enumerate(robot_points):
            if int(keypoint_valid[idx]) != 1 or not np.isfinite(keypoint).all():
                continue
            distances = np.linalg.norm(points - keypoint.reshape(1, 3), axis=1)
            d_keypoints[idx] = float(np.min(distances) - float(keypoint_radius))

    link_count = max(0, robot_points.shape[0] - 1)
    d_links = np.full((link_count,), np.nan, dtype=np.float32)
    link_valid = np.zeros((link_count,), dtype=np.uint8)
    if points.shape[0] > 0:
        for link_idx in range(link_count):
            start = robot_points[link_idx]
            end = robot_points[link_idx + 1]
            if (
                int(keypoint_valid[link_idx]) != 1
                or int(keypoint_valid[link_idx + 1]) != 1
                or not np.isfinite(start).all()
                or not np.isfinite(end).all()
            ):
                continue
            link_valid[link_idx] = 1
            distances = point_to_segment_distances(points, start, end)
            d_links[link_idx] = float(np.min(distances) - float(link_radius))

    valid_keypoint_dist = np.isfinite(d_keypoints)
    if np.any(valid_keypoint_dist):
        closest_keypoint = int(np.nanargmin(d_keypoints))
        closest_keypoint_distance = float(d_keypoints[closest_keypoint])
    else:
        closest_keypoint = -1
        closest_keypoint_distance = np.nan

    valid_link_dist = np.isfinite(d_links)
    if np.any(valid_link_dist):
        closest_link = int(np.nanargmin(d_links))
        closest_link_distance = float(d_links[closest_link])
    else:
        closest_link = -1
        closest_link_distance = np.nan

    return RobotGeometryTargets(
        robot_keypoints_world=robot_points.astype(np.float32),
        robot_keypoint_valid_mask=keypoint_valid.astype(np.uint8),
        robot_link_valid_mask=link_valid.astype(np.uint8),
        d_gt_keypoints=d_keypoints.astype(np.float32),
        d_gt_links=d_links.astype(np.float32),
        closest_keypoint=np.asarray([closest_keypoint], dtype=np.int32),
        closest_link=np.asarray([closest_link], dtype=np.int32),
        closest_keypoint_distance=np.asarray([closest_keypoint_distance], dtype=np.float32),
        closest_link_distance=np.asarray([closest_link_distance], dtype=np.float32),
    )


def compute_jacobian(
    env: Any,
    frame: Optional[IKFrame],
    qvel_indices: Optional[np.ndarray],
) -> np.ndarray:
    nan_jac = np.full((6, 7), np.nan, dtype=np.float32)
    try:
        if frame is None:
            return nan_jac
        if qvel_indices is None:
            _, qvel_indices = robot_joint_indices(env)
        jacp, jacr = frame_jacobian(env, frame, qvel_indices)
        jac = np.vstack([jacp, jacr])
        if jac.shape != (6, 7):
            return nan_jac
        return jac.astype(np.float32)
    except Exception:
        return nan_jac


class H5Writer:
    def __init__(self, path: Path, n_points: int, save_jacobian: bool):
        import h5py

        path.parent.mkdir(parents=True, exist_ok=True)
        self.hf = h5py.File(path, "w")
        self.tasks_group = self.hf.create_group("tasks")
        self.n = 0
        self.n_points = int(n_points)
        self.save_jacobian = save_jacobian
        self.scene_counts: Dict[Tuple[int, int], int] = {}

    def register_task(self, task_id: int, task_name: str) -> None:
        task_group = self.tasks_group.require_group(f"task_{int(task_id):03d}")
        task_group.attrs["task_id"] = int(task_id)
        task_group.attrs["task_name"] = str(task_name)
        task_group.require_group("scenes")

    def _scene_group(self, task_id: int, init_state_id: int) -> Any:
        task_group = self.tasks_group.require_group(f"task_{int(task_id):03d}")
        task_group.attrs["task_id"] = int(task_id)
        scenes_group = task_group.require_group("scenes")
        scene_group = scenes_group.require_group(f"init_{int(init_state_id):03d}")
        scene_group.attrs["task_id"] = int(task_id)
        scene_group.attrs["init_state_id"] = int(init_state_id)
        return scene_group

    def _dataset_parent(self, scene_group: Any, name: str) -> Tuple[Any, str]:
        if "/" not in name:
            return scene_group, name
        parent_path, dataset_name = name.rsplit("/", 1)
        parent = scene_group
        for part in parent_path.split("/"):
            parent = parent.require_group(part)
        return parent, dataset_name

    def _require_dataset(self, scene_group: Any, name: str, value: np.ndarray) -> Any:
        parent, dataset_name = self._dataset_parent(scene_group, name)
        if dataset_name in parent:
            return parent[dataset_name]

        value = np.asarray(value)
        tail_shape = tuple(value.shape)
        chunks = (1,) + tail_shape
        return parent.create_dataset(
            dataset_name,
            shape=(0,) + tail_shape,
            maxshape=(None,) + tail_shape,
            chunks=chunks,
            dtype=value.dtype,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

    def append(self, sample: Mapping[str, np.ndarray], task_id: int, init_state_id: int) -> int:
        scene_key = (int(task_id), int(init_state_id))
        scene_group = self._scene_group(*scene_key)
        idx = self.scene_counts.get(scene_key, 0)
        for name, value in sample.items():
            ds = self._require_dataset(scene_group, name, np.asarray(value))
            ds.resize((idx + 1,) + ds.shape[1:])
            ds[idx] = value
        scene_group.attrs["num_samples"] = int(idx + 1)
        self.scene_counts[scene_key] = idx + 1
        self.n += 1
        return self.n - 1

    def close(self) -> None:
        self.hf.attrs["num_samples"] = int(self.n)
        self.hf.attrs["num_scenes"] = int(len(self.scene_counts))
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

    args.robot_keypoint_body_names = [
        str(name).strip() for name in args.robot_keypoint_body_names if str(name).strip()
    ]
    if len(args.robot_keypoint_body_names) < 2:
        raise ValueError(
            "--robot-keypoint-body-names must contain at least two robot bodies "
            "to define diagnostic link clearances."
        )
    if float(args.robot_keypoint_radius) < 0.0 or float(args.robot_link_radius) < 0.0:
        raise ValueError("--robot-keypoint-radius and --robot-link-radius must be non-negative.")

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
    pose_workspace_min = np.asarray(
        args.pose_workspace_min if args.pose_workspace_min is not None else args.workspace_min,
        dtype=np.float32,
    )
    pose_workspace_max = np.asarray(
        args.pose_workspace_max if args.pose_workspace_max is not None else args.workspace_max,
        dtype=np.float32,
    )
    if np.any(pose_workspace_max <= pose_workspace_min):
        raise ValueError(
            f"pose workspace max must be greater than min, got {pose_workspace_min} -> {pose_workspace_max}"
        )
    flip_images = not bool(args.no_flip_camera_images)
    invert_v = not bool(args.no_invert_v)
    debug_vis_dir = Path(args.debug_vis_dir) if args.debug_vis_dir else None
    if debug_vis_dir is not None and not debug_vis_dir.is_absolute():
        debug_vis_dir = repo_root / debug_vis_dir
    pose_debug_dir = Path(args.pose_debug_dir) if args.pose_debug_dir else None
    if pose_debug_dir is not None and not pose_debug_dir.is_absolute():
        pose_debug_dir = repo_root / pose_debug_dir
    debug_vis_every = max(1, int(args.debug_vis_every))
    debug_vis_max = max(0, int(args.debug_vis_max))
    debug_frames_saved = 0
    pose_debug_max = max(0, int(args.pose_debug_max))
    pose_debug_saved = 0

    print(f"[dataset] repo_root={repo_root}")
    print(f"[dataset] LIBERO_CONFIG_PATH={os.environ.get('LIBERO_CONFIG_PATH')}")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name](safety_level=args.safety_level)

    writer = H5Writer(output_path, n_points=args.n_points, save_jacobian=args.save_jacobian)
    writer.hf.attrs["schema_version"] = "libero_cbf_v1"
    writer.hf.attrs["h5_layout"] = "tasks/task_xxx/scenes/init_xxx"
    writer.hf.attrs["pointcloud_frame"] = "world"
    writer.hf.attrs["cbf_target"] = "h_star = signed_clearance - safety_margin"
    writer.hf.attrs["d_obs_semantics"] = "signed clearance to EE safety volume"
    writer.hf.attrs["safety_scope"] = (
        "primary h_star target is EE safety only; robot keypoint/link fields are "
        "diagnostic whole-arm clearances in the world frame"
    )
    writer.hf.attrs["pointcloud_semantics"] = (
        "filtered obstacle pointcloud; target/manipulated objects are excluded"
    )
    writer.hf.attrs["closest_point_idx_semantics"] = (
        "raw combined obstacle pointcloud index before sampling/padding"
    )
    writer.hf.attrs["fused_source_camera_encoding_json"] = json.dumps(
        {"0": args.camera_local, "1": args.camera_external},
        sort_keys=True,
    )
    writer.hf.attrs["task_suite_name"] = args.task_suite_name
    writer.hf.attrs["safety_level"] = args.safety_level
    writer.hf.attrs["camera_local"] = args.camera_local
    writer.hf.attrs["camera_external"] = args.camera_external
    writer.hf.attrs["n_points"] = int(args.n_points)
    writer.hf.attrs["d_max"] = float(args.d_max)
    writer.hf.attrs["d_min"] = float(args.d_min)
    writer.hf.attrs["safety_margin"] = float(args.safety_margin)
    writer.hf.attrs["h_scale"] = float(args.h_scale)
    writer.hf.attrs["safety_distance_mode"] = str(args.safety_distance_mode)
    writer.hf.attrs["S_star_obs_distance_mode"] = str(args.safety_distance_mode)
    writer.hf.attrs["safety_knn_k"] = int(args.safety_knn_k)
    writer.hf.attrs["safety_robust_cap"] = float(args.safety_robust_cap)
    writer.hf.attrs["ee_radius"] = float(args.ee_radius)
    writer.hf.attrs["ee_box_half_width"] = float(args.ee_radius)
    writer.hf.attrs["ee_safety_geometry"] = str(args.ee_safety_geometry)
    writer.hf.attrs["target_object_map_json"] = json.dumps(target_map, sort_keys=True)
    writer.hf.attrs["allow_missing_target_objects"] = bool(args.allow_missing_target_objects)
    writer.hf.attrs["exclude_name_patterns_json"] = json.dumps(args.exclude_name_patterns)
    writer.hf.attrs["workspace_min"] = workspace_min
    writer.hf.attrs["workspace_max"] = workspace_max
    writer.hf.attrs["sampling_mode"] = str(args.sampling_mode)
    writer.hf.attrs["proposal_type_encoding_json"] = json.dumps(
        {"0": "random_pose_or_random_action", "1": "near_boundary"},
        sort_keys=True,
    )
    writer.hf.attrs["safety_region_encoding_json"] = json.dumps(
        {"0": "safe_far", "1": "near_boundary_safe", "2": "unsafe_margin"},
        sort_keys=True,
    )
    writer.hf.attrs["safety_region_semantics"] = (
        "classified from h_star selected by safety_distance_mode; used for balancing"
    )
    writer.hf.attrs["safety_region_hard_semantics"] = (
        "classified from h_star_hard; conservative nearest-point region for analysis"
    )
    writer.hf.attrs["balance_safety_regions"] = bool(args.balance_safety_regions)
    writer.hf.attrs["target_safe_far_fraction"] = float(args.target_safe_far_fraction)
    writer.hf.attrs["target_near_boundary_fraction"] = float(args.target_near_boundary_fraction)
    writer.hf.attrs["target_unsafe_fraction"] = float(args.target_unsafe_fraction)
    writer.hf.attrs["near_boundary_fraction"] = float(args.near_boundary_fraction)
    writer.hf.attrs["boundary_band"] = float(args.boundary_band)
    writer.hf.attrs["near_boundary_min_dist"] = float(args.near_boundary_min_dist)
    writer.hf.attrs["near_boundary_max_tries"] = int(args.near_boundary_max_tries)
    writer.hf.attrs["near_boundary_max_dist"] = (
        float(args.near_boundary_max_dist)
        if args.near_boundary_max_dist is not None
        else float(args.safety_margin) + max(0.0, float(args.boundary_band))
    )
    writer.hf.attrs["pose_xy_source"] = str(args.pose_xy_source)
    writer.hf.attrs["pose_table_margin"] = float(args.pose_table_margin)
    writer.hf.attrs["pose_workspace_min"] = pose_workspace_min
    writer.hf.attrs["pose_workspace_max"] = pose_workspace_max
    writer.hf.attrs["ee_orientation_delta_range_deg"] = np.asarray(
        args.ee_orientation_delta_range_deg,
        dtype=np.float32,
    )
    writer.hf.attrs["ik_position_tol"] = float(args.ik_position_tol)
    writer.hf.attrs["ik_orientation_tol"] = float(args.ik_orientation_tol)
    writer.hf.attrs["collision_margin"] = float(args.collision_margin)
    writer.hf.attrs["robot_collision_name_patterns_json"] = json.dumps(
        args.robot_collision_name_patterns
    )
    writer.hf.attrs["reject_robot_self_collisions"] = bool(args.reject_robot_self_collisions)
    writer.hf.attrs["robot_geometry_frame"] = "world"
    writer.hf.attrs["robot_keypoint_body_names_json"] = json.dumps(
        [str(name) for name in args.robot_keypoint_body_names]
    )
    writer.hf.attrs["robot_link_pairs_json"] = json.dumps(
        [
            [idx, idx + 1]
            for idx in range(max(0, len(args.robot_keypoint_body_names) - 1))
        ]
    )
    writer.hf.attrs["robot_keypoint_radius"] = float(args.robot_keypoint_radius)
    writer.hf.attrs["robot_link_radius"] = float(args.robot_link_radius)
    writer.hf.attrs["d_gt_keypoints_semantics"] = (
        "for each robot keypoint body: min obstacle distance minus robot_keypoint_radius"
    )
    writer.hf.attrs["d_gt_links_semantics"] = (
        "for each consecutive keypoint segment: min obstacle distance to capsule axis "
        "minus robot_link_radius"
    )
    writer.hf.attrs["depth_is_metric"] = bool(args.depth_is_metric)
    writer.hf.attrs["flip_camera_images"] = bool(flip_images)
    writer.hf.attrs["invert_v"] = bool(invert_v)
    writer.hf.attrs["debug_save_ply"] = bool(args.debug_save_ply)
    writer.hf.attrs["pose_debug_dir"] = str(pose_debug_dir) if pose_debug_dir else ""
    writer.hf.attrs["pose_debug_max"] = int(pose_debug_max)

    instance_maps: Dict[str, Dict[str, str]] = {}
    pose_workspace_maps: Dict[str, Dict[str, List[float]]] = {}
    total_skipped_low_points = 0
    total_pose_attempts = 0
    total_skipped_ik = 0
    total_skipped_collision = 0
    total_skipped_balance = 0

    try:
        for task_id in args.task_indices:
            task = task_suite.get_task(int(task_id))
            writer.register_task(int(task_id), task.name)
            task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            init_states = task_suite.get_task_init_states(int(task_id))
            init_indices = iter_init_indices(
                total=len(init_states),
                requested=args.init_state_indices,
                max_init_states=args.max_init_states,
            )
            target_names = target_map.get(task.name, [])
            if not target_names and not bool(args.allow_missing_target_objects):
                raise KeyError(
                    f"No target-object map entry for task {task.name!r}. "
                    "Pass --target-object-map-json with the manipulated objects, "
                    "or use --allow-missing-target-objects for exploratory debugging."
                )

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
            task_pose_workspace_min, task_pose_workspace_max = pose_workspace_for_env(
                env=env,
                args=args,
                default_pose_min=pose_workspace_min,
                default_pose_max=pose_workspace_max,
            )
            if np.any(task_pose_workspace_max <= task_pose_workspace_min):
                raise ValueError(
                    f"resolved pose workspace max must be greater than min, got "
                    f"{task_pose_workspace_min} -> {task_pose_workspace_max}"
                )
            pose_workspace_maps[str(task_id)] = {
                "task_name": task.name,
                "min": task_pose_workspace_min.tolist(),
                "max": task_pose_workspace_max.tolist(),
            }
            ik_frame: Optional[IKFrame] = None
            qpos_indices: Optional[np.ndarray] = None
            qvel_indices: Optional[np.ndarray] = None
            joint_lower: Optional[np.ndarray] = None
            joint_upper: Optional[np.ndarray] = None
            if args.sampling_mode in ("random_pose", "mixed") or args.save_jacobian:
                ik_frame = resolve_ik_frame(env, args)
                qpos_indices, qvel_indices = robot_joint_indices(env)
            if args.sampling_mode in ("random_pose", "mixed"):
                assert qpos_indices is not None
                joint_lower, joint_upper = robot_joint_limits(env, qpos_indices)

            print(
                f"[dataset] task_id={task_id} name={task.name} "
                f"targets={target_names or 'NONE'} init_states={len(init_indices)}"
            )
            if ik_frame is not None:
                frame_role = (
                    "pose IK"
                    if args.sampling_mode in ("random_pose", "mixed")
                    else "jacobian"
                )
                print(
                    f"[dataset] {frame_role} frame={ik_frame.kind}:{ik_frame.name} "
                    f"pose_workspace={task_pose_workspace_min.tolist()}->{task_pose_workspace_max.tolist()}"
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

                    base_state = env.get_sim_state()
                    if args.ee_orientation_reference_quat_xyzw is None:
                        reference_quat_xyzw = np.asarray(obs["robot0_eef_quat"], dtype=np.float32)
                    else:
                        reference_quat_xyzw = np.asarray(
                            args.ee_orientation_reference_quat_xyzw,
                            dtype=np.float32,
                        )
                    reference_quat_xyzw = normalize_quat(reference_quat_xyzw).astype(np.float32)

                    base_wrist_raw = camera_cloud_from_obs(
                        obs=obs,
                        env=env,
                        camera_name=args.camera_local,
                        camera_utils=camera_utils,
                        depth_is_metric=args.depth_is_metric,
                        flip_images=flip_images,
                        invert_v=invert_v,
                    )
                    base_ext_raw = camera_cloud_from_obs(
                        obs=obs,
                        env=env,
                        camera_name=args.camera_external,
                        camera_utils=camera_utils,
                        depth_is_metric=args.depth_is_metric,
                        flip_images=flip_images,
                        invert_v=invert_v,
                    )
                    base_wrist = filter_cloud(base_wrist_raw, keep_ids, workspace_min, workspace_max)
                    base_ext = filter_cloud(base_ext_raw, keep_ids, workspace_min, workspace_max)
                    base_combined_world = np.concatenate([base_wrist.world, base_ext.world], axis=0)

                    saved_for_init = 0
                    attempts_for_init = 0
                    region_counts = {0: 0, 1: 0, 2: 0}
                    target_region_counts = (
                        target_safety_region_counts(args)
                        if bool(args.balance_safety_regions)
                        else None
                    )
                    max_attempts_for_init = (
                        int(args.samples_per_init_state)
                        * max(1, int(args.max_pose_attempts_per_sample))
                    )
                    while (
                        saved_for_init < int(args.samples_per_init_state)
                        and attempts_for_init < max_attempts_for_init
                    ):
                        attempts_for_init += 1
                        rollout_step = saved_for_init
                        pose_debug: Dict[str, Any] = {
                            "pose_attempt": int(attempts_for_init),
                            "sampling_mode": str(args.sampling_mode),
                        }

                        proposal_type = 0
                        selected_sampling_mode = str(args.sampling_mode)
                        if args.sampling_mode in ("random_pose", "mixed"):
                            assert ik_frame is not None
                            assert qpos_indices is not None
                            assert qvel_indices is not None
                            assert joint_lower is not None
                            assert joint_upper is not None

                            env.regenerate_obs_from_state(base_state)
                            use_near_boundary = (
                                args.sampling_mode == "mixed"
                                and base_combined_world.shape[0] >= int(args.min_obstacle_points)
                                and rng.random() < float(args.near_boundary_fraction)
                            )
                            if use_near_boundary:
                                selected_sampling_mode = "near_boundary"
                                target_pos, target_quat, target_delta_euler, near_debug = (
                                    sample_near_boundary_ee_pose(
                                        rng=rng,
                                        obstacle_points_world=base_combined_world,
                                        pose_workspace_min=task_pose_workspace_min,
                                        pose_workspace_max=task_pose_workspace_max,
                                        reference_quat_xyzw=reference_quat_xyzw,
                                        orientation_delta_range_deg=args.ee_orientation_delta_range_deg,
                                        ee_radius=float(args.ee_radius),
                                        safety_margin=float(args.safety_margin),
                                        boundary_band=float(args.boundary_band),
                                        near_boundary_min_dist=float(args.near_boundary_min_dist),
                                        near_boundary_max_dist=args.near_boundary_max_dist,
                                        near_boundary_max_tries=int(args.near_boundary_max_tries),
                                    )
                                )
                                if bool(near_debug["near_boundary_fallback_random_pose"]):
                                    selected_sampling_mode = "random_pose_fallback"
                                    proposal_type = 0
                                else:
                                    proposal_type = 1
                                pose_debug.update(near_debug)
                            else:
                                selected_sampling_mode = "random_pose"
                                target_pos, target_quat, target_delta_euler = sample_random_ee_pose(
                                    rng=rng,
                                    pose_workspace_min=task_pose_workspace_min,
                                    pose_workspace_max=task_pose_workspace_max,
                                    reference_quat_xyzw=reference_quat_xyzw,
                                    orientation_delta_range_deg=args.ee_orientation_delta_range_deg,
                                )
                            total_pose_attempts += 1
                            ik_result = solve_ik_to_pose(
                                env=env,
                                target_pos=target_pos,
                                target_quat_xyzw=target_quat,
                                frame=ik_frame,
                                qpos_indices=qpos_indices,
                                qvel_indices=qvel_indices,
                                joint_lower=joint_lower,
                                joint_upper=joint_upper,
                                args=args,
                            )
                            pose_debug.update(
                                {
                                    "selected_sampling_mode": selected_sampling_mode,
                                    "proposal_type": int(proposal_type),
                                    "target_ee_pos_world": target_pos.tolist(),
                                    "target_ee_quat_xyzw": target_quat.tolist(),
                                    "target_delta_euler_xyz_rad": target_delta_euler.tolist(),
                                    "ik_pos_error": float(ik_result.pos_error),
                                    "ik_ori_error": float(ik_result.ori_error),
                                    "ik_iterations": int(ik_result.iterations),
                                }
                            )
                            if not ik_result.success:
                                total_skipped_ik += 1
                                continue

                            collisions = robot_collision_contacts(
                                env=env,
                                robot_patterns=args.robot_collision_name_patterns,
                                margin=float(args.collision_margin),
                                reject_robot_self_collisions=bool(args.reject_robot_self_collisions),
                            )
                            if collisions:
                                total_skipped_collision += 1
                                continue
                            obs = env.regenerate_obs_from_state(env.get_sim_state())
                        else:
                            if attempts_for_init > 1:
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
                        combined_source_camera = np.concatenate(
                            [
                                np.zeros((wrist.world.shape[0],), dtype=np.int8),
                                np.ones((ext.world.shape[0],), dtype=np.int8),
                            ],
                            axis=0,
                        )
                        if combined_world.shape[0] < int(args.min_obstacle_points):
                            total_skipped_low_points += 1
                            continue

                        q = np.asarray(obs["robot0_joint_pos"], dtype=np.float32)
                        ee_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
                        ee_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32)
                        ee_camera_pos = camera_position_world(env, args.camera_local)
                        ee_volume_start = (
                            ee_camera_pos
                            if args.ee_safety_geometry == "camera_to_eef_box"
                            else ee_pos
                        )
                        ee_volume_end = ee_pos
                        safety_targets = compute_safety_targets(
                            points_world=combined_world,
                            ee_pos_world=ee_pos,
                            d_max=float(args.d_max),
                            d_min=float(args.d_min),
                            ee_radius=float(args.ee_radius),
                            ee_safety_geometry=str(args.ee_safety_geometry),
                            ee_volume_start_world=ee_volume_start,
                            ee_volume_end_world=ee_volume_end,
                            safety_distance_mode=str(args.safety_distance_mode),
                            safety_knn_k=int(args.safety_knn_k),
                            safety_robust_cap=float(args.safety_robust_cap),
                            source_camera=combined_source_camera,
                        )
                        cbf_targets = compute_cbf_targets(
                            safety_targets=safety_targets,
                            safety_margin=float(args.safety_margin),
                            h_scale=float(args.h_scale),
                        )
                        safety_region = classify_safety_region(
                            h_star=cbf_targets["h_star"],
                            boundary_band=float(args.boundary_band),
                        )
                        safety_region_hard = classify_safety_region(
                            h_star=cbf_targets["h_star_hard"],
                            boundary_band=float(args.boundary_band),
                        )
                        if target_region_counts is not None:
                            region = int(safety_region)
                            if region_counts[region] >= target_region_counts[region]:
                                total_skipped_balance += 1
                                continue

                        robot_geometry_targets = compute_robot_geometry_targets(
                            env=env,
                            points_world=combined_world,
                            body_names=args.robot_keypoint_body_names,
                            keypoint_radius=float(args.robot_keypoint_radius),
                            link_radius=float(args.robot_link_radius),
                        )

                        wrist_sample, wrist_valid_mask = sample_or_pad_cloud_with_mask(
                            wrist,
                            n_points=int(args.n_points),
                            rng=rng,
                            empty_local=(0.0, 0.0, 10.0),
                            empty_world=(10.0, 10.0, 10.0),
                        )
                        ext_sample, ext_valid_mask = sample_or_pad_cloud_with_mask(
                            ext,
                            n_points=int(args.n_points),
                            rng=rng,
                            empty_local=(0.0, 0.0, 10.0),
                            empty_world=(10.0, 10.0, 10.0),
                        )

                        fused_pointcloud = np.concatenate(
                            [wrist_sample.world, ext_sample.world],
                            axis=0,
                        ).astype(np.float32)
                        fused_object_ids = np.concatenate(
                            [wrist_sample.instance_ids, ext_sample.instance_ids],
                            axis=0,
                        ).astype(np.int32)
                        fused_source_camera = np.concatenate(
                            [
                                np.zeros((int(args.n_points),), dtype=np.int8),
                                np.ones((int(args.n_points),), dtype=np.int8),
                            ],
                            axis=0,
                        )
                        fused_valid_mask = np.concatenate(
                            [wrist_valid_mask, ext_valid_mask],
                            axis=0,
                        ).astype(np.uint8)

                        sample: Dict[str, np.ndarray] = {
                            "q": q,
                            "ee_pos_world": ee_pos,
                            "ee_ori_world": ee_quat,
                            "ee_camera_pos_world": ee_camera_pos,
                            "ee_volume_start_world": ee_volume_start.astype(np.float32),
                            "ee_volume_end_world": ee_volume_end.astype(np.float32),
                            "ee_volume_radius": np.asarray([float(args.ee_radius)], dtype=np.float32),
                            "ee_volume_half_width": np.asarray([float(args.ee_radius)], dtype=np.float32),
                            "ee/pointcloud": wrist_sample.world,
                            "ee/point_object_id": wrist_sample.instance_ids,
                            "ee/valid_mask": wrist_valid_mask,
                            "backview/pointcloud": ext_sample.world,
                            "backview/point_object_id": ext_sample.instance_ids,
                            "backview/valid_mask": ext_valid_mask,
                            "fused_pointcloud/pointcloud": fused_pointcloud,
                            "fused_pointcloud/point_object_id": fused_object_ids,
                            "fused_pointcloud/source_camera": fused_source_camera,
                            "fused_pointcloud/valid_mask": fused_valid_mask,
                            "S_star_obs": safety_targets.S_star_obs,
                            "S_star_hard": safety_targets.S_star_hard,
                            "S_star_knn": safety_targets.S_star_knn,
                            "S_star_robust": safety_targets.S_star_robust,
                            "h_star": cbf_targets["h_star"],
                            "h_star_hard": cbf_targets["h_star_hard"],
                            "h_star_knn": cbf_targets["h_star_knn"],
                            "h_star_robust": cbf_targets["h_star_robust"],
                            "h_star_norm": cbf_targets["h_star_norm"],
                            "h_star_hard_norm": cbf_targets["h_star_hard_norm"],
                            "h_star_knn_norm": cbf_targets["h_star_knn_norm"],
                            "h_star_robust_norm": cbf_targets["h_star_robust_norm"],
                            "d_obs": safety_targets.d_obs,
                            "d_obs_hard": safety_targets.d_obs_hard,
                            "d_obs_knn": safety_targets.d_obs_knn,
                            "d_obs_robust": safety_targets.d_obs_robust,
                            "v_rep": safety_targets.v_rep,
                            "v_rep_knn": safety_targets.v_rep_knn,
                            "closest_point_idx": safety_targets.closest_point_idx,
                            "closest_raw_point_idx": safety_targets.closest_raw_point_idx,
                            "closest_point_world": safety_targets.closest_point_world,
                            "closest_volume_point_world": safety_targets.closest_volume_point_world,
                            "closest_point_source": safety_targets.closest_point_source,
                            "robot_keypoints_world": robot_geometry_targets.robot_keypoints_world,
                            "robot_keypoint_valid_mask": robot_geometry_targets.robot_keypoint_valid_mask,
                            "robot_link_valid_mask": robot_geometry_targets.robot_link_valid_mask,
                            "d_gt_keypoints": robot_geometry_targets.d_gt_keypoints,
                            "d_gt_links": robot_geometry_targets.d_gt_links,
                            "closest_keypoint": robot_geometry_targets.closest_keypoint,
                            "closest_link": robot_geometry_targets.closest_link,
                            "closest_keypoint_distance": (
                                robot_geometry_targets.closest_keypoint_distance
                            ),
                            "closest_link_distance": robot_geometry_targets.closest_link_distance,
                            "safety_margin": np.asarray([float(args.safety_margin)], dtype=np.float32),
                            "h_scale": np.asarray([float(args.h_scale)], dtype=np.float32),
                            "proposal_type": np.asarray([proposal_type], dtype=np.int8),
                            "safety_region": np.asarray([safety_region], dtype=np.int8),
                            "safety_region_hard": np.asarray([safety_region_hard], dtype=np.int8),
                            "task_id": np.asarray([task_id], dtype=np.int32),
                            "init_state_id": np.asarray([init_id], dtype=np.int32),
                            "rollout_step": np.asarray([rollout_step], dtype=np.int32),
                        }
                        if args.save_jacobian:
                            sample["J_ee"] = compute_jacobian(env, ik_frame, qvel_indices)

                        sample_index = writer.append(
                            sample,
                            task_id=int(task_id),
                            init_state_id=int(init_id),
                        )
                        if target_region_counts is not None:
                            region_counts[int(safety_region)] += 1
                        debug_metadata = {
                            "task_id": int(task_id),
                            "task_name": task.name,
                            "init_state_id": int(init_id),
                            "rollout_step": int(rollout_step),
                            "sample_index": int(sample_index),
                            "S_star_obs": float(safety_targets.S_star_obs[0]),
                            "S_star_hard": float(safety_targets.S_star_hard[0]),
                            "S_star_knn": float(safety_targets.S_star_knn[0]),
                            "S_star_robust": float(safety_targets.S_star_robust[0]),
                            "h_star": float(cbf_targets["h_star"][0]),
                            "h_star_hard": float(cbf_targets["h_star_hard"][0]),
                            "h_star_knn": float(cbf_targets["h_star_knn"][0]),
                            "h_star_robust": float(cbf_targets["h_star_robust"][0]),
                            "h_star_norm": float(cbf_targets["h_star_norm"][0]),
                            "h_star_hard_norm": float(cbf_targets["h_star_hard_norm"][0]),
                            "h_star_knn_norm": float(cbf_targets["h_star_knn_norm"][0]),
                            "h_star_robust_norm": float(cbf_targets["h_star_robust_norm"][0]),
                            "d_obs": float(safety_targets.d_obs[0]),
                            "d_obs_hard": float(safety_targets.d_obs_hard[0]),
                            "d_obs_knn": float(safety_targets.d_obs_knn[0]),
                            "d_obs_robust": float(safety_targets.d_obs_robust[0]),
                            "safety_margin": float(args.safety_margin),
                            "h_scale": float(args.h_scale),
                            "safety_distance_mode": str(args.safety_distance_mode),
                            "safety_knn_k": int(args.safety_knn_k),
                            "safety_robust_cap": float(args.safety_robust_cap),
                            "v_rep": safety_targets.v_rep.tolist(),
                            "v_rep_knn": safety_targets.v_rep_knn.tolist(),
                            "closest_point_idx": int(safety_targets.closest_point_idx[0]),
                            "closest_raw_point_idx": int(safety_targets.closest_raw_point_idx[0]),
                            "closest_point_world": safety_targets.closest_point_world.tolist(),
                            "closest_volume_point_world": safety_targets.closest_volume_point_world.tolist(),
                            "closest_point_source": int(safety_targets.closest_point_source[0]),
                            "robot_keypoint_body_names": [
                                str(name) for name in args.robot_keypoint_body_names
                            ],
                            "robot_keypoint_valid_mask": (
                                robot_geometry_targets.robot_keypoint_valid_mask.tolist()
                            ),
                            "robot_link_valid_mask": (
                                robot_geometry_targets.robot_link_valid_mask.tolist()
                            ),
                            "d_gt_keypoints": robot_geometry_targets.d_gt_keypoints.tolist(),
                            "d_gt_links": robot_geometry_targets.d_gt_links.tolist(),
                            "closest_keypoint": int(robot_geometry_targets.closest_keypoint[0]),
                            "closest_link": int(robot_geometry_targets.closest_link[0]),
                            "closest_keypoint_distance": float(
                                robot_geometry_targets.closest_keypoint_distance[0]
                            ),
                            "closest_link_distance": float(
                                robot_geometry_targets.closest_link_distance[0]
                            ),
                            "proposal_type": int(proposal_type),
                            "safety_region": int(safety_region),
                            "safety_region_hard": int(safety_region_hard),
                            "safety_region_counts": {
                                str(k): int(v) for k, v in region_counts.items()
                            },
                            "target_safety_region_counts": (
                                {str(k): int(v) for k, v in target_region_counts.items()}
                                if target_region_counts is not None
                                else None
                            ),
                            "ee_pos_world": ee_pos.tolist(),
                            "ee_quat_xyzw": ee_quat.tolist(),
                            "ee_camera_pos_world": ee_camera_pos.tolist(),
                            "ee_volume_start_world": ee_volume_start.tolist(),
                            "ee_volume_end_world": ee_volume_end.tolist(),
                            "ee_volume_radius": float(args.ee_radius),
                            "ee_volume_half_width": float(args.ee_radius),
                            "ee_safety_geometry": str(args.ee_safety_geometry),
                            "q": q.tolist(),
                            "camera_ee": args.camera_local,
                            "camera_backview": args.camera_external,
                            "ee_point_count": int(wrist.world.shape[0]),
                            "backview_point_count": int(ext.world.shape[0]),
                        }
                        debug_metadata.update(pose_debug)

                        if (
                            debug_vis_dir is not None
                            and debug_frames_saved < debug_vis_max
                            and writer.n % debug_vis_every == 0
                        ):
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
                                v_rep_world=safety_targets.v_rep,
                                output_dir=debug_vis_dir,
                                frame_idx=debug_frames_saved,
                                metadata=debug_metadata,
                                max_points=int(args.debug_pointcloud_max_points),
                                rng=rng,
                                save_ply=bool(args.debug_save_ply),
                                ee_quat_world=ee_quat,
                            )
                            debug_frames_saved += 1

                        if pose_debug_dir is not None and pose_debug_saved < pose_debug_max:
                            save_pose_sampling_debug_sample(
                                obs=obs,
                                env=env,
                                camera_utils=camera_utils,
                                ee_cloud=wrist,
                                backview_cloud=ext,
                                output_dir=pose_debug_dir,
                                sample_idx=pose_debug_saved,
                                metadata=debug_metadata,
                                keep_ids=keep_ids,
                                camera_ee=args.camera_local,
                                camera_backview=args.camera_external,
                                flip_images=flip_images,
                                max_points=int(args.debug_pointcloud_max_points),
                                rng=rng,
                            )
                            pose_debug_saved += 1
                        saved_for_init += 1

                    if saved_for_init < int(args.samples_per_init_state):
                        print(
                            f"[dataset] warning: saved {saved_for_init}/"
                            f"{int(args.samples_per_init_state)} samples for "
                            f"task_id={task_id} init_state={init_id} after "
                            f"{attempts_for_init} attempts"
                        )
                        if target_region_counts is not None:
                            print(
                                f"[dataset] region_counts={region_counts} "
                                f"target_region_counts={target_region_counts}"
                            )
            finally:
                env.close()

    finally:
        writer.hf.attrs["instance_maps_json"] = json.dumps(instance_maps, sort_keys=True)
        writer.hf.attrs["resolved_pose_workspaces_json"] = json.dumps(
            pose_workspace_maps,
            sort_keys=True,
        )
        writer.hf.attrs["skipped_low_obstacle_points"] = int(total_skipped_low_points)
        writer.hf.attrs["pose_sampling_attempts"] = int(total_pose_attempts)
        writer.hf.attrs["skipped_ik"] = int(total_skipped_ik)
        writer.hf.attrs["skipped_collision"] = int(total_skipped_collision)
        writer.hf.attrs["skipped_balance"] = int(total_skipped_balance)
        writer.hf.attrs["debug_frames_saved"] = int(debug_frames_saved)
        writer.hf.attrs["pose_debug_saved"] = int(pose_debug_saved)
        writer.close()

    print(f"[dataset] wrote {writer.n} samples to {output_path}")
    print(f"[dataset] skipped_low_obstacle_points={total_skipped_low_points}")
    if args.sampling_mode in ("random_pose", "mixed"):
        print(
            f"[dataset] pose attempts={total_pose_attempts} "
            f"skipped_ik={total_skipped_ik} skipped_collision={total_skipped_collision}"
        )
    if args.balance_safety_regions:
        print(f"[dataset] skipped_balance={total_skipped_balance}")
    if debug_vis_dir is not None:
        print(f"[dataset] debug frames saved: {debug_frames_saved} in {debug_vis_dir}")
    if pose_debug_dir is not None:
        print(f"[dataset] pose debug samples saved: {pose_debug_saved} in {pose_debug_dir}")


if __name__ == "__main__":
    main()
