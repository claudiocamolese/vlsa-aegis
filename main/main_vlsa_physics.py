from __future__ import annotations

import collections
import dataclasses
import json
import logging
import math
import pathlib
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imageio
import numpy as np
import tqdm
import tyro

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
for _path in (
    REPO_ROOT / "safelibero",
    REPO_ROOT / "openpi" / "src",
    REPO_ROOT / "openpi" / "packages" / "openpi-client" / "src",
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools

warnings.filterwarnings("ignore")

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 1024

DEFAULT_SAFETY_CHECKPOINT = (
    "vlsa_physics/vlsa_physics/checkpoints/"
    "final_spatial_I_residual_robustEE_fullpoints_fs05_seed42_best_safety.pt"
)

DEFAULT_ROBOT_KEYPOINT_BODY_NAMES = [
    "robot0_link0",
    "robot0_link1",
    "robot0_link2",
    "robot0_link3",
    "robot0_link4",
    "robot0_link5",
    "robot0_link6",
    "robot0_link7",
    "gripper0_eef",
]

DEFAULT_EXCLUDE_NAME_PATTERNS = [
    "panda",
    "gripper",
    "mount",
    "main_table",
    "table",
    "floor",
    "arena",
]


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # VLA policy parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    use_local_policy: bool = False
    policy_config_name: str = "pi05_libero"
    checkpoint_dir: str = "checkpoints/pi05_libero"
    policy_backend: str = "openpi"
    openvla_oft_server_url: str = "http://127.0.0.1:8766/infer"

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "safelibero_spatial"
    safety_level: str = "II"
    task_index: List[int] = dataclasses.field(default_factory=lambda: [0])
    episode_index: List[int] = dataclasses.field(default_factory=lambda: [0])
    num_steps_wait: int = 20
    num_trials_per_task: int = 50

    #################################################################################################################
    # VLSA Physics safety model
    #################################################################################################################
    safety_checkpoint_path: str = DEFAULT_SAFETY_CHECKPOINT
    safety_device: str = "cuda"
    pointcept_root: Optional[str] = None
    disable_safety_filter: bool = False
    safety_fail_open: bool = True

    # CBF/QP parameters.
    safety_alpha: float = 5.0
    h_activation: float = 0.02
    max_correction_norm: float = 0.2
    max_joint_delta: float = 0.05
    max_cartesian_action_abs: float = 1.0
    max_safe_translation_norm: float = 0.25
    ik_damping: float = 0.05
    min_grad_norm: float = 1e-6
    clip_absolute_joint_delta: bool = False
    emergency_h_threshold: float = 0.0
    emergency_repulsion_only: bool = True
    emergency_repulsion_requires_ee_risk: bool = True
    emergency_ee_h_threshold: float = -0.02
    repulsion_only_ee_h_threshold: float = -0.04
    use_geometric_ee_repulsion_guard: bool = True
    repulsion_only_ee_geometric_h_threshold: float = 0.02
    repulsion_only_ee_geometric_h_release_threshold: float = 0.04
    preserve_nominal_tangent_in_repulsion: bool = True
    repulsion_nominal_tangent_scale: float = 0.35
    repulsion_nominal_away_scale: float = 0.50
    block_unsafe_nominal_component: bool = True
    block_nominal_component_requires_ee_risk: bool = True
    prevent_downward_safety_correction: bool = True
    arm_only_correction_scale: float = 0.35
    max_arm_only_cartesian_correction_norm: float = 0.12
    clip_safe_translation_norm_only_in_emergency: bool = False
    passthrough_safe_nominal_action: bool = True
    safe_nominal_correction_eps: float = 1e-8
    allow_task_descent_in_safety: bool = True
    task_descent_h_threshold: float = -0.04
    task_descent_min_h_ee: float = -0.04
    max_task_descent: float = 0.08

    # By default the safety layer only corrects translation and preserves the VLA wrist/gripper command.
    preserve_vla_rotation_in_safety: bool = True

    # Optional active upright controller. Enable only if you really want a top-down wrist correction.
    keep_gripper_upright: bool = False
    upright_only_when_filter_active: bool = True
    upright_local_axis: List[float] = dataclasses.field(default_factory=lambda: [0.0, 0.0, 1.0])
    table_normal_world: List[float] = dataclasses.field(default_factory=lambda: [0.0, 0.0, 1.0])
    upright_rot_gain: float = 2.0
    max_upright_rot_norm: float = 0.25
    preserve_yaw_when_upright: bool = True

    # "learned" uses autograd on h_total(q, P). "geometric" uses J^T v_rep.
    safety_gradient_mode: str = "learned"
    fallback_to_geometric_grad: bool = True

    # Online obstacle pointcloud construction. Use 0 to pass all filtered pixels.
    safety_camera_local: str = "robot0_eye_in_hand"
    safety_camera_external: str = "backview"
    safety_points_per_camera: int = 1024
    min_obstacle_points: int = 32
    debug_obstacle_name_patterns: List[str] = dataclasses.field(default_factory=lambda: ["obstacle", "moka"])
    workspace_min: List[float] = dataclasses.field(default_factory=lambda: [-0.80, -0.80, 0.70])
    workspace_max: List[float] = dataclasses.field(default_factory=lambda: [0.80, 0.80, 1.60])
    exclude_name_patterns: List[str] = dataclasses.field(default_factory=lambda: list(DEFAULT_EXCLUDE_NAME_PATTERNS))
    target_object_map_json: Optional[str] = None
    allow_missing_target_objects: bool = False
    exclude_goal_objects_from_safety: bool = True
    goal_object_name_patterns: List[str] = dataclasses.field(default_factory=list)
    flip_camera_images: bool = True
    depth_is_metric: bool = False
    invert_v: bool = False

    # Constants used by the trained model.
    safety_margin: float = 0.02
    ee_radius: float = 0.04
    robot_keypoint_radius: float = 0.04
    robot_link_radius: float = 0.04
    robot_keypoint_body_names: List[str] = dataclasses.field(
        default_factory=lambda: list(DEFAULT_ROBOT_KEYPOINT_BODY_NAMES)
    )

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "results"
    seed: int = 7
    save_safety_debug: bool = True
    safety_debug_every: int = 10
    break_on_collision: bool = True


def _insert_physics_paths() -> None:
    physics_dir = REPO_ROOT / "vlsa_physics"
    scripts_dir = REPO_ROOT / "safelibero" / "scripts"
    for path in (physics_dir, scripts_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def checkpoint_arg(ckpt_args: Dict[str, Any], name: str, default: Any) -> Any:
    value = ckpt_args.get(name, None)
    return default if value is None else value


def build_safety_model_from_checkpoint(
    checkpoint: Dict[str, Any],
    args: Args,
    device: Any,
) -> Tuple[Any, Any, Any, Any, Any]:
    _insert_physics_paths()
    from train import LinkAwarePhysicsCBFNet, PANDA_Q_LOWER, PANDA_Q_UPPER, normalize_panda_q

    ckpt_args = dict(checkpoint.get("args", checkpoint.get("config", {})))
    pointcept_root = args.pointcept_root
    if pointcept_root is None:
        pointcept_root = checkpoint_arg(
            ckpt_args,
            "pointcept_root",
            str(REPO_ROOT / "Pointcept"),
        )

    model = LinkAwarePhysicsCBFNet(
        pointcept_root=pointcept_root,
        ptv3_grid_size=float(checkpoint_arg(ckpt_args, "ptv3_grid_size", 0.02)),
        ptv3_patch_size=int(checkpoint_arg(ckpt_args, "ptv3_patch_size", 128)),
        ptv3_enable_flash=bool(checkpoint_arg(ckpt_args, "ptv3_enable_flash", True)),
        ptv3_order=checkpoint_arg(ckpt_args, "ptv3_order", ["z"]),
        ptv3_in_channels=int(checkpoint_arg(ckpt_args, "ptv3_in_channels", 20)),
        ptv3_out_dim=int(checkpoint_arg(ckpt_args, "ptv3_out_dim", 512)),
        camera_embedding_dim=int(checkpoint_arg(ckpt_args, "camera_embedding_dim", 16)),
        model_dim=int(checkpoint_arg(ckpt_args, "model_dim", 256)),
        robot_id_embedding_dim=int(checkpoint_arg(ckpt_args, "robot_id_embedding_dim", 32)),
        max_robot_keypoints=int(checkpoint_arg(ckpt_args, "max_robot_keypoints", 16)),
        fusion_layers=int(checkpoint_arg(ckpt_args, "fusion_layers", 3)),
        fusion_heads=int(checkpoint_arg(ckpt_args, "fusion_heads", 8)),
        fusion_ffn_dim=int(checkpoint_arg(ckpt_args, "fusion_ffn_dim", 1024)),
        head_hidden_dim=int(checkpoint_arg(ckpt_args, "head_hidden_dim", 128)),
        dropout=float(checkpoint_arg(ckpt_args, "dropout", 0.1)),
        geo_tau=float(checkpoint_arg(ckpt_args, "geo_tau", 0.01)),
        softmin_tau=float(checkpoint_arg(ckpt_args, "softmin_tau", 0.02)),
        default_ee_radius=float(checkpoint_arg(ckpt_args, "default_ee_radius", args.ee_radius)),
        default_safety_margin=float(checkpoint_arg(ckpt_args, "default_safety_margin", args.safety_margin)),
        default_robot_keypoint_radius=float(
            checkpoint_arg(ckpt_args, "default_robot_keypoint_radius", args.robot_keypoint_radius)
        ),
        default_robot_link_radius=float(checkpoint_arg(ckpt_args, "default_robot_link_radius", args.robot_link_radius)),
        residual_m_scale=float(checkpoint_arg(ckpt_args, "residual_m_scale", 0.1)),
        prediction_mode=str(checkpoint_arg(ckpt_args, "prediction_mode", "residual")),
    ).to(device)

    state = checkpoint.get("model", None)
    if state is None:
        state = checkpoint.get("model_state_dict", None)
    if state is None:
        raise KeyError("Safety checkpoint does not contain 'model' or 'model_state_dict'.")

    model.load_state_dict(state, strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    return model, normalize_panda_q, PANDA_Q_LOWER, PANDA_Q_UPPER, ckpt_args


def cbf_filter_closed_form(u_nominal: Any, h: Any, grad_q: Any, alpha: float = 5.0, eps: float = 1e-6) -> Tuple[Any, Dict[str, Any]]:
    b = -float(alpha) * h
    gu = (grad_q * u_nominal).sum(dim=1, keepdim=True)
    gg = (grad_q * grad_q).sum(dim=1, keepdim=True).clamp_min(float(eps))

    violation = b - gu
    correction_scale = violation.relu() / gg
    u_safe = u_nominal + correction_scale * grad_q

    return u_safe, {
        "b": b,
        "gu": gu,
        "violation": violation,
        "correction_norm": (u_safe - u_nominal).norm(dim=1, keepdim=True),
    }


def damped_pseudoinverse_action(jacobian: np.ndarray, twist: np.ndarray, damping: float) -> np.ndarray:
    jac = np.asarray(jacobian, dtype=np.float64).reshape(6, 7)
    twist = np.asarray(twist, dtype=np.float64).reshape(6)
    lhs = jac @ jac.T + float(damping) ** 2 * np.eye(6, dtype=np.float64)
    try:
        return (jac.T @ np.linalg.solve(lhs, twist)).astype(np.float32)
    except np.linalg.LinAlgError:
        return (jac.T @ np.linalg.lstsq(lhs, twist, rcond=None)[0]).astype(np.float32)


def _normalize_vector(vector: Any, fallback: Sequence[float]) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-8:
        return np.asarray(fallback, dtype=np.float32).reshape(3)
    return arr / norm


def _quat_xyzw_to_matrix(quat: Any) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float32)
    x, y, z, w = quat / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _clip_vector_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(3)
    max_norm = float(max_norm)
    if max_norm <= 0.0:
        return np.zeros(3, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= max_norm or norm < 1e-8:
        return vector
    return (vector * (max_norm / norm)).astype(np.float32)


def upright_gripper_rotation_action(
    obs: Dict[str, Any],
    nominal_rot_action: np.ndarray,
    local_axis: Sequence[float],
    table_normal_world: Sequence[float],
    gain: float,
    max_rot_norm: float,
    preserve_yaw: bool,
) -> Tuple[np.ndarray, Dict[str, float]]:
    ee_rotation = _quat_xyzw_to_matrix(obs["robot0_eef_quat"])
    target_axis = _normalize_vector(table_normal_world, fallback=(0.0, 0.0, 1.0))
    local_axis_np = _normalize_vector(local_axis, fallback=(0.0, 0.0, 1.0))
    gripper_axis_world = _normalize_vector(ee_rotation @ local_axis_np, fallback=target_axis)

    dot = float(np.clip(np.dot(gripper_axis_world, target_axis), -1.0, 1.0))
    tilt_angle = float(np.arccos(dot))
    upright_error = np.cross(gripper_axis_world, target_axis).astype(np.float32)
    if float(np.linalg.norm(upright_error)) < 1e-8 and dot < 0.0:
        reference = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(reference, gripper_axis_world))) > 0.9:
            reference = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        upright_error = _normalize_vector(np.cross(gripper_axis_world, reference), fallback=(1.0, 0.0, 0.0))

    upright_rot = float(gain) * upright_error
    if preserve_yaw:
        nominal_rot = np.asarray(nominal_rot_action, dtype=np.float32).reshape(3)
        yaw_rot = float(np.dot(nominal_rot, target_axis)) * target_axis
    else:
        yaw_rot = np.zeros(3, dtype=np.float32)

    rot_action = _clip_vector_norm(yaw_rot + upright_rot, max_rot_norm)
    return rot_action.astype(np.float32), {
        "upright_tilt_angle": tilt_angle,
        "upright_axis_dot": dot,
        "upright_rot_norm": float(np.linalg.norm(rot_action)),
    }


def detect_ee_frame(env: Any, helpers: Dict[str, Any]) -> Any:
    model = env.sim.model
    IKFrame = helpers["IKFrame"]
    for name in ("gripper0_eef", "eef_marker", "robot0_eef", "right_hand"):
        if helpers["model_body_exists"](model, name):
            return IKFrame(kind="body", name=name)
    for name in ("gripper0_grip_site", "gripper0_ft_frame", "robot0_eef_site", "eef_site"):
        if helpers["model_site_exists"](model, name):
            return IKFrame(kind="site", name=name)
    raise ValueError("Could not auto-detect an EE frame for Jacobian conversion.")


class VLSAPhysicsSafetyFilter:
    def __init__(self, args: Args) -> None:
        _insert_physics_paths()
        import torch
        from cbf_dataset import (
            IKFrame,
            camera_cloud_from_obs,
            filter_cloud,
            frame_jacobian,
            instance_id_to_name,
            keep_instance_ids,
            load_target_map,
            model_body_exists,
            model_site_exists,
            patch_robosuite_numpy2_segmentation,
            robot_joint_indices,
            robot_keypoint_positions_world,
            sample_or_pad_cloud_with_mask,
        )

        patch_robosuite_numpy2_segmentation()

        self.args = args
        self.torch = torch
        requested_device = args.safety_device
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = torch.device(requested_device)

        ckpt_path = pathlib.Path(args.safety_checkpoint_path)
        if not ckpt_path.is_absolute():
            ckpt_path = REPO_ROOT / ckpt_path
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Safety checkpoint not found: {ckpt_path}")

        try:
            checkpoint = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(str(ckpt_path), map_location=self.device)

        (
            self.model,
            self.normalize_panda_q,
            self.panda_q_lower,
            self.panda_q_upper,
            self.ckpt_args,
        ) = build_safety_model_from_checkpoint(checkpoint, args, self.device)

        self.helpers = {
            "IKFrame": IKFrame,
            "camera_cloud_from_obs": camera_cloud_from_obs,
            "filter_cloud": filter_cloud,
            "frame_jacobian": frame_jacobian,
            "instance_id_to_name": instance_id_to_name,
            "keep_instance_ids": keep_instance_ids,
            "load_target_map": load_target_map,
            "model_body_exists": model_body_exists,
            "model_site_exists": model_site_exists,
            "robot_joint_indices": robot_joint_indices,
            "robot_keypoint_positions_world": robot_keypoint_positions_world,
            "sample_or_pad_cloud_with_mask": sample_or_pad_cloud_with_mask,
        }
        self.target_map = load_target_map(args.target_object_map_json)
        self.target_map_lower = {str(key).lower(): value for key, value in self.target_map.items()}
        self.rng = np.random.default_rng(int(args.seed))

        self.qpos_indices: Optional[np.ndarray] = None
        self.qvel_indices: Optional[np.ndarray] = None
        self.ee_frame = None
        self.ee_geometric_repulsion_latched = False
        logging.info("Loaded VLSA Physics safety checkpoint: %s", ckpt_path)

    def bind_env(self, env: Any) -> None:
        self.qpos_indices, self.qvel_indices = self.helpers["robot_joint_indices"](env)
        self.ee_frame = detect_ee_frame(env, self.helpers)

    def _task_targets(self, task_description: str) -> List[str]:
        task_key = str(task_description).replace(" ", "_")
        target_names = self.target_map.get(task_key, None)
        if target_names is None:
            target_names = self.target_map_lower.get(task_key.lower(), None)
        if target_names is None:
            if not self.args.allow_missing_target_objects:
                raise KeyError(f"No target-object mapping for task {task_key!r}.")
            logging.warning("No target-object mapping for task %s; filtering only by generic exclusions.", task_key)
            return []
        return [str(name) for name in target_names]

    def _task_goal_patterns(self, task_description: str) -> List[str]:
        if not bool(self.args.exclude_goal_objects_from_safety):
            return []

        text = " ".join(str(task_description).lower().replace("_", " ").split())
        patterns = [p.strip().lower() for p in self.args.goal_object_name_patterns if p and p.strip()]

        goal_phrase_map = {
            "plate": (
                "place it on the plate",
                "put the bowl on the plate",
                "put the bowl onto the plate",
            ),
            "basket": (
                "place it in the basket",
                "place it into the basket",
                "put it in the basket",
                "put it into the basket",
            ),
            "stove": (
                "put the bowl on the stove",
                "put the bowl onto the stove",
            ),
            "cabinet": (
                "put the bowl on the cabinet",
                "put the bowl on top of the cabinet",
            ),
            "bowl": (
                "put the cream cheese in the bowl",
                "put the cream cheese into the bowl",
            ),
        }
        for pattern, phrases in goal_phrase_map.items():
            if any(phrase in text for phrase in phrases):
                patterns.append(pattern)
        if "drawer" in text and ("put the bowl inside" in text or "inside the drawer" in text):
            patterns.append("drawer")

        return list(dict.fromkeys(patterns))

    def _build_safety_batch(
        self,
        obs: Dict[str, Any],
        env: Any,
        task_description: str,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        from robosuite.utils import camera_utils

        args = self.args
        target_names = self._task_targets(task_description)
        id_to_name = self.helpers["instance_id_to_name"](env)
        target_patterns = [name.strip().lower() for name in target_names if name.strip()]
        target_ids = np.asarray(
            [
                int(instance_id)
                for instance_id, instance_name in id_to_name.items()
                if any(pattern in str(instance_name).lower() for pattern in target_patterns)
            ],
            dtype=np.int32,
        )
        goal_patterns = self._task_goal_patterns(task_description)
        goal_ids = np.asarray(
            [
                int(instance_id)
                for instance_id, instance_name in id_to_name.items()
                if any(pattern in str(instance_name).lower() for pattern in goal_patterns)
            ],
            dtype=np.int32,
        )
        goal_ids = goal_ids[~np.isin(goal_ids, target_ids)]
        goal_names = [str(id_to_name[int(instance_id)]) for instance_id in goal_ids]
        exempt_ids = np.unique(np.concatenate([target_ids, goal_ids]))
        keep_ids = self.helpers["keep_instance_ids"](id_to_name, target_names, args.exclude_name_patterns)
        # Manipulated and intentional goal-contact instances must not repel the task itself.
        keep_ids = keep_ids[~np.isin(keep_ids, exempt_ids)]
        if keep_ids.shape[0] == 0:
            return None, {
                "reason": "no_keep_instance_ids",
                "excluded_target_names": target_names,
                "excluded_target_ids": target_ids.tolist(),
                "excluded_goal_patterns": goal_patterns,
                "excluded_goal_names": goal_names,
                "excluded_goal_ids": goal_ids.tolist(),
                "n_goal_points_removed": 0,
            }

        workspace_min = np.asarray(args.workspace_min, dtype=np.float32).reshape(3)
        workspace_max = np.asarray(args.workspace_max, dtype=np.float32).reshape(3)

        wrist_raw = self.helpers["camera_cloud_from_obs"](
            obs=obs,
            env=env,
            camera_name=args.safety_camera_local,
            camera_utils=camera_utils,
            depth_is_metric=bool(args.depth_is_metric),
            flip_images=bool(args.flip_camera_images),
            invert_v=bool(args.invert_v),
        )
        ext_raw = self.helpers["camera_cloud_from_obs"](
            obs=obs,
            env=env,
            camera_name=args.safety_camera_external,
            camera_utils=camera_utils,
            depth_is_metric=bool(args.depth_is_metric),
            flip_images=bool(args.flip_camera_images),
            invert_v=bool(args.invert_v),
        )

        n_target_points_removed = int(
            np.isin(wrist_raw.instance_ids, target_ids).sum()
            + np.isin(ext_raw.instance_ids, target_ids).sum()
        )
        n_goal_points_removed = int(
            np.isin(wrist_raw.instance_ids, goal_ids).sum()
            + np.isin(ext_raw.instance_ids, goal_ids).sum()
        )
        wrist = self.helpers["filter_cloud"](wrist_raw, keep_ids, workspace_min, workspace_max)
        ext = self.helpers["filter_cloud"](ext_raw, keep_ids, workspace_min, workspace_max)
        debug_patterns = [p.strip().lower() for p in args.debug_obstacle_name_patterns if p and p.strip()]
        debug_obstacle_ids = np.asarray(
            [
                int(instance_id)
                for instance_id, instance_name in id_to_name.items()
                if any(pattern in str(instance_name).lower() for pattern in debug_patterns)
            ],
            dtype=np.int32,
        )
        debug_obstacle_names = [str(id_to_name[int(instance_id)]) for instance_id in debug_obstacle_ids]
        debug_obstacle_raw_points = int(
            np.isin(wrist_raw.instance_ids, debug_obstacle_ids).sum()
            + np.isin(ext_raw.instance_ids, debug_obstacle_ids).sum()
        )
        debug_obstacle_filtered_points = int(
            np.isin(wrist.instance_ids, debug_obstacle_ids).sum()
            + np.isin(ext.instance_ids, debug_obstacle_ids).sum()
        )
        if np.isin(wrist.instance_ids, exempt_ids).any() or np.isin(ext.instance_ids, exempt_ids).any():
            raise RuntimeError("Manipulated/goal-object points reached the safety obstacle cloud.")
        combined_count = int(wrist.world.shape[0] + ext.world.shape[0])
        if combined_count < int(args.min_obstacle_points):
            return None, {
                "reason": "low_obstacle_points",
                "n_obstacle_points": combined_count,
                "n_keep_ids": int(keep_ids.shape[0]),
                "excluded_target_names": target_names,
                "excluded_target_ids": target_ids.tolist(),
                "excluded_goal_patterns": goal_patterns,
                "excluded_goal_names": goal_names,
                "excluded_goal_ids": goal_ids.tolist(),
                "n_target_points_removed": n_target_points_removed,
                "n_goal_points_removed": n_goal_points_removed,
                "debug_obstacle_names": debug_obstacle_names,
                "debug_obstacle_raw_points": debug_obstacle_raw_points,
                "debug_obstacle_filtered_points": debug_obstacle_filtered_points,
            }

        points_per_camera = int(args.safety_points_per_camera)
        if points_per_camera > 0:
            wrist_sample, wrist_valid_mask = self.helpers["sample_or_pad_cloud_with_mask"](
                wrist,
                n_points=points_per_camera,
                rng=self.rng,
                empty_local=(0.0, 0.0, 10.0),
                empty_world=(10.0, 10.0, 10.0),
            )
            ext_sample, ext_valid_mask = self.helpers["sample_or_pad_cloud_with_mask"](
                ext,
                n_points=points_per_camera,
                rng=self.rng,
                empty_local=(0.0, 0.0, 10.0),
                empty_world=(10.0, 10.0, 10.0),
            )
            pc_world = np.concatenate([wrist_sample.world, ext_sample.world], axis=0).astype(np.float32)
            valid_mask = np.concatenate([wrist_valid_mask, ext_valid_mask], axis=0).astype(np.float32)
            source_camera = np.concatenate(
                [
                    np.zeros((points_per_camera,), dtype=np.int64),
                    np.ones((points_per_camera,), dtype=np.int64),
                ],
                axis=0,
            )
        else:
            pc_world = np.concatenate([wrist.world, ext.world], axis=0).astype(np.float32)
            valid_mask = np.ones((pc_world.shape[0],), dtype=np.float32)
            source_camera = np.concatenate(
                [
                    np.zeros((wrist.world.shape[0],), dtype=np.int64),
                    np.ones((ext.world.shape[0],), dtype=np.int64),
                ],
                axis=0,
            )

        if self.qpos_indices is not None:
            q = np.asarray(env.sim.data.qpos[self.qpos_indices], dtype=np.float32).reshape(7)
        else:
            q = np.asarray(obs["robot0_joint_pos"], dtype=np.float32).reshape(7)

        ee_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(3)
        robot_keypoints, kp_valid = self.helpers["robot_keypoint_positions_world"](
            env,
            args.robot_keypoint_body_names,
        )
        robot_keypoints = np.nan_to_num(robot_keypoints.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        kp_valid = kp_valid.astype(np.float32)
        link_valid = (kp_valid[:-1] * kp_valid[1:]).astype(np.float32)

        torch = self.torch
        batch = {
            "pc_world": torch.from_numpy(pc_world).to(self.device).unsqueeze(0),
            "q": torch.from_numpy(q).to(self.device).unsqueeze(0),
            "ee_pos_world": torch.from_numpy(ee_pos).to(self.device).unsqueeze(0),
            "valid_mask": torch.from_numpy(valid_mask).to(self.device).unsqueeze(0),
            "source_camera": torch.from_numpy(source_camera).to(self.device).unsqueeze(0),
            "robot_keypoints_world": torch.from_numpy(robot_keypoints).to(self.device).unsqueeze(0),
            "robot_keypoint_valid_mask": torch.from_numpy(kp_valid).to(self.device).unsqueeze(0),
            "robot_link_valid_mask": torch.from_numpy(link_valid).to(self.device).unsqueeze(0),
            "safety_margin": torch.tensor([[args.safety_margin]], dtype=torch.float32, device=self.device),
            "ee_radius": torch.tensor([[args.ee_radius]], dtype=torch.float32, device=self.device),
            "robot_keypoint_radius": torch.tensor(
                [[args.robot_keypoint_radius]], dtype=torch.float32, device=self.device
            ),
            "robot_link_radius": torch.tensor([[args.robot_link_radius]], dtype=torch.float32, device=self.device),
        }
        meta = {
            "n_obstacle_points": combined_count,
            "n_model_points": int(pc_world.shape[0]),
            "n_valid_model_points": int(valid_mask.sum()),
            "n_keep_ids": int(keep_ids.shape[0]),
            "excluded_target_names": target_names,
            "excluded_target_ids": target_ids.tolist(),
            "excluded_goal_patterns": goal_patterns,
            "excluded_goal_names": goal_names,
            "excluded_goal_ids": goal_ids.tolist(),
            "n_target_points_removed": n_target_points_removed,
            "n_goal_points_removed": n_goal_points_removed,
            "debug_obstacle_names": debug_obstacle_names,
            "debug_obstacle_raw_points": debug_obstacle_raw_points,
            "debug_obstacle_filtered_points": debug_obstacle_filtered_points,
            "pc_world_np": pc_world,
            "valid_mask_np": valid_mask,
            "robot_keypoints_np": robot_keypoints,
            "kp_valid_np": kp_valid,
        }
        return batch, meta

    def _model_outputs_and_grad(self, batch: Dict[str, Any]) -> Tuple[Dict[str, Any], Any]:
        torch = self.torch
        q_norm = self.normalize_panda_q(batch["q"])
        q_norm = q_norm.clone().detach().requires_grad_(True)

        with torch.enable_grad():
            outputs = self.model(
                pc_world=batch["pc_world"],
                q=q_norm,
                ee_pos=batch["ee_pos_world"],
                valid_mask=batch["valid_mask"],
                source_camera=batch["source_camera"],
                robot_keypoints_world=batch["robot_keypoints_world"],
                robot_keypoint_valid_mask=batch["robot_keypoint_valid_mask"],
                robot_link_valid_mask=batch["robot_link_valid_mask"],
                ee_radius=batch["ee_radius"],
                safety_margin=batch["safety_margin"],
                robot_keypoint_radius=batch["robot_keypoint_radius"],
                robot_link_radius=batch["robot_link_radius"],
            )
            grad_q_norm = torch.autograd.grad(
                outputs=outputs["h_total_pred"].sum(),
                inputs=q_norm,
                create_graph=False,
                retain_graph=False,
                only_inputs=True,
                allow_unused=True,
            )[0]

        if grad_q_norm is None:
            grad_q_norm = torch.zeros_like(q_norm)

        scale = 2.0 / (
            self.panda_q_upper.to(device=self.device, dtype=batch["q"].dtype)
            - self.panda_q_lower.to(device=self.device, dtype=batch["q"].dtype)
        )
        grad_q = grad_q_norm * scale.reshape(1, 7)
        return outputs, grad_q

    def _geometric_grad_q(self, env: Any, batch: Dict[str, Any], outputs: Dict[str, Any]) -> Any:
        torch = self.torch
        pc = batch["pc_world"][0].detach().cpu().numpy()
        valid = batch["valid_mask"][0].detach().cpu().numpy() > 0.5
        if not np.any(valid):
            return torch.zeros((1, 7), dtype=torch.float32, device=self.device)

        pc_valid = pc[valid]
        h_ee = float(outputs["h_ee_pred"].detach().cpu().reshape(-1)[0])
        h_arm = float(outputs["h_arm_pred"].detach().cpu().reshape(-1)[0])

        if h_ee <= h_arm:
            robot_point = np.asarray(batch["ee_pos_world"][0].detach().cpu().numpy(), dtype=np.float32).reshape(3)
            frame = self.ee_frame
        else:
            d_links = outputs["d_links_pred"].detach().cpu().numpy().reshape(-1)
            link_idx = int(np.nanargmin(d_links))
            keypoints = batch["robot_keypoints_world"][0].detach().cpu().numpy()
            start = keypoints[link_idx]
            end = keypoints[min(link_idx + 1, keypoints.shape[0] - 1)]
            axis = end - start
            denom = float(np.dot(axis, axis))
            if denom < 1e-10:
                robot_point = start
                body_idx = link_idx
            else:
                rel = pc_valid - start.reshape(1, 3)
                t = np.clip((rel @ axis.reshape(3, 1)).reshape(-1) / denom, 0.0, 1.0)
                closest_on_link = start.reshape(1, 3) + t.reshape(-1, 1) * axis.reshape(1, 3)
                distances = np.linalg.norm(pc_valid - closest_on_link, axis=1)
                point_idx = int(np.argmin(distances))
                robot_point = closest_on_link[point_idx]
                body_idx = link_idx if float(t[point_idx]) < 0.5 else min(link_idx + 1, len(self.args.robot_keypoint_body_names) - 1)
            body_name = self.args.robot_keypoint_body_names[int(body_idx)]
            frame = self.helpers["IKFrame"](kind="body", name=body_name)

        distances = np.linalg.norm(pc_valid - robot_point.reshape(1, 3), axis=1)
        obstacle_point = pc_valid[int(np.argmin(distances))]
        direction = robot_point - obstacle_point
        norm = float(np.linalg.norm(direction))
        if norm < 1e-8:
            direction = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            direction = (direction / norm).astype(np.float32)

        jacp, _ = self.helpers["frame_jacobian"](env, frame, self.qvel_indices)
        grad = jacp.T @ direction.reshape(3)
        return torch.from_numpy(grad.astype(np.float32)).to(self.device).reshape(1, 7)

    def _apply_upright_gripper_constraint(
        self,
        action: np.ndarray,
        obs: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        constrained = np.asarray(action, dtype=np.float32).copy()
        rot_action, upright_debug = upright_gripper_rotation_action(
            obs=obs,
            nominal_rot_action=constrained[3:6],
            local_axis=self.args.upright_local_axis,
            table_normal_world=self.args.table_normal_world,
            gain=float(self.args.upright_rot_gain),
            max_rot_norm=float(self.args.max_upright_rot_norm),
            preserve_yaw=bool(self.args.preserve_yaw_when_upright),
        )
        constrained[3:6] = np.clip(
            rot_action,
            -float(self.args.max_cartesian_action_abs),
            float(self.args.max_cartesian_action_abs),
        )
        return constrained, upright_debug

    def filter_action(
        self,
        action: np.ndarray,
        obs: Dict[str, Any],
        env: Any,
        task_description: str,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        try:
            return self._filter_action_impl(action, obs, env, task_description)
        except Exception as exc:
            if not self.args.safety_fail_open:
                raise
            logging.warning("VLSA Physics safety filter failed open: %s", exc)
            nominal = np.asarray(action, dtype=np.float32).reshape(-1)
            return nominal, {"active": False, "reason": "exception", "error": str(exc)}

    def _filter_action_impl(
        self,
        action: np.ndarray,
        obs: Dict[str, Any],
        env: Any,
        task_description: str,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 7:
            raise ValueError(f"Expected 7D VLA action, got shape {action.shape}")

        batch, meta = self._build_safety_batch(obs, env, task_description)
        if batch is None:
            return action, {"active": False, **meta}

        outputs, learned_grad_q = self._model_outputs_and_grad(batch)
        h_total = outputs["h_total_pred"].detach()
        h_value = float(h_total.cpu().reshape(-1)[0])
        h_ee = float(outputs["h_ee_pred"].detach().cpu().reshape(-1)[0])
        h_arm = float(outputs["h_arm_pred"].detach().cpu().reshape(-1)[0])
        d_links = outputs["d_links_pred"].detach().cpu().numpy().reshape(-1)
        critical_link = int(np.nanargmin(d_links)) if d_links.shape[0] > 0 else -1
        ee_pos_np = np.asarray(batch["ee_pos_world"][0].detach().cpu().numpy(), dtype=np.float32).reshape(3)
        pc_world_np = np.asarray(meta.get("pc_world_np"), dtype=np.float32)
        valid_mask_np = np.asarray(meta.get("valid_mask_np"), dtype=np.float32) > 0.5
        if pc_world_np.ndim == 2 and pc_world_np.shape[1] == 3 and np.any(valid_mask_np):
            ee_obstacle_distances = np.linalg.norm(pc_world_np[valid_mask_np] - ee_pos_np.reshape(1, 3), axis=1)
            ee_obstacle_min_distance = float(np.min(ee_obstacle_distances))
        else:
            ee_obstacle_min_distance = float("inf")
        ee_geometric_h = ee_obstacle_min_distance - float(self.args.ee_radius) - float(self.args.safety_margin)

        debug: Dict[str, Any] = {
            "active": False,
            "h_total": h_value,
            "h_ee": h_ee,
            "h_arm": h_arm,
            "ee_obstacle_min_distance": ee_obstacle_min_distance,
            "ee_geometric_h": ee_geometric_h,
            "critical_link": critical_link,
            "n_obstacle_points": meta.get("n_obstacle_points"),
            "n_valid_model_points": meta.get("n_valid_model_points"),
            "excluded_target_names": meta.get("excluded_target_names"),
            "excluded_target_ids": meta.get("excluded_target_ids"),
            "excluded_goal_patterns": meta.get("excluded_goal_patterns"),
            "excluded_goal_names": meta.get("excluded_goal_names"),
            "excluded_goal_ids": meta.get("excluded_goal_ids"),
            "n_target_points_removed": meta.get("n_target_points_removed"),
            "n_goal_points_removed": meta.get("n_goal_points_removed"),
            "debug_obstacle_names": meta.get("debug_obstacle_names"),
            "debug_obstacle_raw_points": meta.get("debug_obstacle_raw_points"),
            "debug_obstacle_filtered_points": meta.get("debug_obstacle_filtered_points"),
            "h_activation": float(self.args.h_activation),
        }

        if h_value > float(self.args.h_activation):
            debug["reason"] = "outside_activation_band"
            if bool(self.args.keep_gripper_upright) and not bool(self.args.upright_only_when_filter_active):
                upright_action, upright_debug = self._apply_upright_gripper_constraint(action, obs)
                debug.update({"upright_applied": True, **upright_debug})
                return upright_action, debug
            return action, debug

        if self.ee_frame is None or self.qvel_indices is None:
            self.bind_env(env)
        jacp, jacr = self.helpers["frame_jacobian"](env, self.ee_frame, self.qvel_indices)
        jac = np.vstack([jacp, jacr]).astype(np.float32)

        cartesian_nominal = action[:6].copy()
        upright_debug: Dict[str, float] = {}
        if bool(self.args.keep_gripper_upright):
            upright_action, upright_debug = self._apply_upright_gripper_constraint(action, obs)
            cartesian_nominal = upright_action[:6]

        u_nominal_np = damped_pseudoinverse_action(jac, cartesian_nominal, damping=float(self.args.ik_damping))

        grad_q = learned_grad_q
        grad_source = "learned"
        grad_norm = float(grad_q.norm(dim=1).detach().cpu().reshape(-1)[0])
        if self.args.safety_gradient_mode == "geometric" or (
            self.args.fallback_to_geometric_grad and grad_norm < float(self.args.min_grad_norm)
        ):
            grad_q = self._geometric_grad_q(env, batch, outputs)
            grad_source = "geometric"
            grad_norm = float(grad_q.norm(dim=1).detach().cpu().reshape(-1)[0])

        if grad_norm < float(self.args.min_grad_norm):
            debug.update(
                {
                    "reason": "small_grad",
                    "grad_source": grad_source,
                    "grad_q_norm": grad_norm,
                }
            )
            if bool(self.args.keep_gripper_upright):
                upright_action = action.copy()
                upright_action[:6] = cartesian_nominal
                debug.update({"upright_applied": True, **upright_debug})
                return upright_action, debug
            return action, debug

        torch = self.torch
        u_nominal = torch.from_numpy(u_nominal_np).to(self.device).reshape(1, 7)
        u_filtered, info = cbf_filter_closed_form(
            u_nominal=u_nominal,
            h=h_total.to(self.device).reshape(1, 1),
            grad_q=grad_q,
            alpha=float(self.args.safety_alpha),
        )

        correction = u_filtered - u_nominal
        raw_joint_correction_norm = float(correction.norm(dim=1).detach().cpu().reshape(-1)[0])
        emergency_active = h_value <= float(self.args.emergency_h_threshold)
        if (
            bool(self.args.passthrough_safe_nominal_action)
            and raw_joint_correction_norm <= float(self.args.safe_nominal_correction_eps)
        ):
            debug.update(
                {
                    "reason": "constraint_satisfied_passthrough",
                    "grad_source": grad_source,
                    "grad_q_norm": grad_norm,
                    "u_nominal_norm": float(np.linalg.norm(u_nominal_np)),
                    "u_safe_norm": float(np.linalg.norm(u_nominal_np)),
                    "cartesian_correction_norm": 0.0,
                    "translation_correction_norm": 0.0,
                    "raw_cartesian_correction": [0.0, 0.0, 0.0],
                    "raw_cartesian_correction_norm": 0.0,
                    "translation_correction": [0.0, 0.0, 0.0],
                    "nominal_translation": action[:3].astype(float).tolist(),
                    "safe_translation": action[:3].astype(float).tolist(),
                    "emergency_active": emergency_active,
                    "emergency_repulsion_only": bool(self.args.emergency_repulsion_only),
                    "blocked_nominal_component": 0.0,
                    "max_safe_translation_norm": float(self.args.max_safe_translation_norm),
                    "descent_allowed": float(action[2]) < 0.0,
                    "task_descent": float(action[2]) if float(action[2]) < 0.0 else 0.0,
                    "max_task_descent": float(self.args.max_task_descent),
                    "task_descent_h_threshold": float(self.args.task_descent_h_threshold),
                    "task_descent_min_h_ee": float(self.args.task_descent_min_h_ee),
                    "task_descent_applied": False,
                    "upright_applied": bool(self.args.keep_gripper_upright),
                    "vla_rotation_preserved": bool(self.args.preserve_vla_rotation_in_safety)
                    and not bool(self.args.keep_gripper_upright),
                    "b": float(info["b"].detach().cpu().reshape(-1)[0]),
                    "gu": float(info["gu"].detach().cpu().reshape(-1)[0]),
                    "violation": float(info["violation"].detach().cpu().reshape(-1)[0]),
                    "joint_correction_norm": raw_joint_correction_norm,
                    "passthrough_safe_nominal_action": True,
                    "emergency_passthrough": emergency_active,
                }
            )
            if bool(self.args.keep_gripper_upright):
                upright_action = action.copy()
                upright_action[:6] = cartesian_nominal
                debug.update({"upright_applied": True, **upright_debug})
                return upright_action, debug
            return action, debug

        correction_norm = correction.norm(dim=1, keepdim=True).clamp_min(1e-6)
        correction = correction * torch.clamp(float(self.args.max_correction_norm) / correction_norm, max=1.0)
        u_safe = u_nominal + correction
        if bool(self.args.clip_absolute_joint_delta):
            u_safe = torch.clamp(u_safe, -float(self.args.max_joint_delta), float(self.args.max_joint_delta))

        u_safe_np = u_safe.detach().cpu().numpy().reshape(7)
        joint_correction_np = (u_safe - u_nominal).detach().cpu().numpy().reshape(7)
        cartesian_correction = (jacp @ joint_correction_np.reshape(7, 1)).reshape(3).astype(np.float32)
        raw_cartesian_correction = cartesian_correction.copy()
        if bool(self.args.prevent_downward_safety_correction):
            cartesian_correction[2] = max(float(cartesian_correction[2]), 0.0)

        ee_emergency_active = h_ee <= float(self.args.emergency_ee_h_threshold)
        ee_learned_repulsion_active = h_ee <= float(self.args.repulsion_only_ee_h_threshold)
        ee_geometric_repulsion_enter_active = ee_geometric_h <= float(
            self.args.repulsion_only_ee_geometric_h_threshold
        )
        ee_geometric_repulsion_release_active = ee_geometric_h <= float(
            self.args.repulsion_only_ee_geometric_h_release_threshold
        )
        if self.ee_geometric_repulsion_latched:
            ee_geometric_repulsion_active = ee_geometric_repulsion_release_active
        else:
            ee_geometric_repulsion_active = ee_geometric_repulsion_enter_active
        self.ee_geometric_repulsion_latched = bool(ee_geometric_repulsion_active)
        ee_repulsion_active = ee_learned_repulsion_active
        if bool(self.args.use_geometric_ee_repulsion_guard):
            ee_repulsion_active = ee_geometric_repulsion_active
        arm_only_correction_active = bool(h_arm < h_ee) and not ee_repulsion_active
        cartesian_correction_before_arm_scaling = cartesian_correction.copy()
        if arm_only_correction_active:
            cartesian_correction = (
                cartesian_correction * float(self.args.arm_only_correction_scale)
            ).astype(np.float32)
            if float(self.args.max_arm_only_cartesian_correction_norm) > 0.0:
                cartesian_correction = _clip_vector_norm(
                    cartesian_correction,
                    max_norm=float(self.args.max_arm_only_cartesian_correction_norm),
                )

        nominal_translation = action[:3].copy()
        correction_norm_np = float(np.linalg.norm(cartesian_correction))
        ee_block_active = ee_emergency_active
        if bool(self.args.use_geometric_ee_repulsion_guard):
            ee_block_active = ee_block_active and ee_geometric_repulsion_active
        block_nominal_component_active = bool(self.args.block_unsafe_nominal_component) and (
            ee_block_active or not bool(self.args.block_nominal_component_requires_ee_risk)
        )
        blocked_nominal_component = 0.0
        if block_nominal_component_active and correction_norm_np > 1e-8:
            correction_dir = cartesian_correction / correction_norm_np
            nominal_along_correction = float(np.dot(nominal_translation, correction_dir))
            if nominal_along_correction < 0.0:
                blocked_nominal_component = nominal_along_correction
                nominal_translation = nominal_translation - nominal_along_correction * correction_dir

        repulsion_only_active = bool(self.args.emergency_repulsion_only) and emergency_active
        if bool(self.args.emergency_repulsion_requires_ee_risk):
            repulsion_only_active = repulsion_only_active and ee_repulsion_active

        repulsion_nominal_tangent = np.zeros(3, dtype=np.float32)
        repulsion_nominal_away = np.zeros(3, dtype=np.float32)
        repulsion_nominal_along_correction = 0.0
        repulsion_preserved_nominal = np.zeros(3, dtype=np.float32)
        if repulsion_only_active:
            candidate_translation = cartesian_correction.copy()
            if bool(self.args.preserve_nominal_tangent_in_repulsion) and correction_norm_np > 1e-8:
                correction_dir = cartesian_correction / correction_norm_np
                repulsion_nominal_along_correction = float(np.dot(nominal_translation, correction_dir))
                repulsion_nominal_tangent = (
                    nominal_translation - repulsion_nominal_along_correction * correction_dir
                ).astype(np.float32)
                if repulsion_nominal_along_correction > 0.0:
                    repulsion_nominal_away = (
                        repulsion_nominal_along_correction * correction_dir
                    ).astype(np.float32)
                repulsion_preserved_nominal = (
                    float(self.args.repulsion_nominal_tangent_scale) * repulsion_nominal_tangent
                    + float(self.args.repulsion_nominal_away_scale) * repulsion_nominal_away
                ).astype(np.float32)
                candidate_translation = (candidate_translation + repulsion_preserved_nominal).astype(np.float32)
                if bool(self.args.prevent_downward_safety_correction):
                    candidate_translation[2] = max(float(candidate_translation[2]), 0.0)
        else:
            candidate_translation = nominal_translation + cartesian_correction

        clip_safe_translation_norm = float(self.args.max_safe_translation_norm) > 0.0 and (
            repulsion_only_active or not bool(self.args.clip_safe_translation_norm_only_in_emergency)
        )
        if clip_safe_translation_norm:
            candidate_translation = _clip_vector_norm(
                candidate_translation,
                max_norm=float(self.args.max_safe_translation_norm),
            )

        candidate_translation_before_task_descent = candidate_translation.copy()
        task_descent_total_h_ok = h_value >= float(self.args.task_descent_h_threshold) or h_arm < h_ee
        task_descent_learned_ee_ok = h_ee >= float(self.args.task_descent_min_h_ee)
        task_descent_geometric_ee_ok = ee_geometric_h > float(
            self.args.repulsion_only_ee_geometric_h_threshold
        )
        task_descent_ee_ok = task_descent_learned_ee_ok
        if bool(self.args.use_geometric_ee_repulsion_guard):
            task_descent_ee_ok = task_descent_ee_ok or task_descent_geometric_ee_ok
        descent_allowed = (
            bool(self.args.allow_task_descent_in_safety)
            and float(action[2]) < 0.0
            and not repulsion_only_active
            and task_descent_total_h_ok
            and task_descent_ee_ok
        )
        task_descent = 0.0
        task_descent_applied = False
        if descent_allowed:
            task_descent = max(float(action[2]), -float(self.args.max_task_descent))
            descent_limited_z = max(float(candidate_translation[2]), task_descent)
            task_descent_applied = not math.isclose(
                float(candidate_translation[2]),
                descent_limited_z,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
            candidate_translation[2] = descent_limited_z

        safe_action = action.copy()
        safe_action[:3] = np.clip(
            candidate_translation,
            -float(self.args.max_cartesian_action_abs),
            float(self.args.max_cartesian_action_abs),
        )
        if bool(self.args.keep_gripper_upright):
            safe_action[3:6] = cartesian_nominal[3:6]
        elif bool(self.args.preserve_vla_rotation_in_safety):
            safe_action[3:6] = action[3:6]
        else:
            safe_twist = (jac @ u_safe_np.reshape(7, 1)).reshape(6).astype(np.float32)
            safe_action[3:6] = np.clip(
                safe_twist[3:6],
                -float(self.args.max_cartesian_action_abs),
                float(self.args.max_cartesian_action_abs),
            )
        safe_action[6] = action[6]

        final_correction_norm = float(np.linalg.norm(safe_action[:6] - action[:6]))
        debug.update(
            {
                "active": True,
                "reason": "filtered",
                "grad_source": grad_source,
                "grad_q_norm": grad_norm,
                "u_nominal_norm": float(np.linalg.norm(u_nominal_np)),
                "u_safe_norm": float(np.linalg.norm(u_safe_np)),
                "cartesian_correction_norm": final_correction_norm,
                "translation_correction_norm": float(np.linalg.norm(safe_action[:3] - action[:3])),
                "raw_cartesian_correction": raw_cartesian_correction.astype(float).tolist(),
                "raw_cartesian_correction_norm": float(np.linalg.norm(raw_cartesian_correction)),
                "cartesian_correction_before_arm_scaling": cartesian_correction_before_arm_scaling.astype(
                    float
                ).tolist(),
                "arm_only_correction_active": arm_only_correction_active,
                "arm_only_correction_scale": float(self.args.arm_only_correction_scale),
                "max_arm_only_cartesian_correction_norm": float(
                    self.args.max_arm_only_cartesian_correction_norm
                ),
                "translation_correction": (safe_action[:3] - action[:3]).astype(float).tolist(),
                "nominal_translation": action[:3].astype(float).tolist(),
                "safe_translation": safe_action[:3].astype(float).tolist(),
                "emergency_active": emergency_active,
                "ee_emergency_active": ee_emergency_active,
                "repulsion_only_active": repulsion_only_active,
                "emergency_repulsion_only": bool(self.args.emergency_repulsion_only),
                "emergency_repulsion_requires_ee_risk": bool(self.args.emergency_repulsion_requires_ee_risk),
                "emergency_ee_h_threshold": float(self.args.emergency_ee_h_threshold),
                "ee_learned_repulsion_active": ee_learned_repulsion_active,
                "ee_geometric_repulsion_active": ee_geometric_repulsion_active,
                "ee_repulsion_active": ee_repulsion_active,
                "repulsion_only_ee_h_threshold": float(self.args.repulsion_only_ee_h_threshold),
                "use_geometric_ee_repulsion_guard": bool(self.args.use_geometric_ee_repulsion_guard),
                "repulsion_only_ee_geometric_h_threshold": float(
                    self.args.repulsion_only_ee_geometric_h_threshold
                ),
                "repulsion_only_ee_geometric_h_release_threshold": float(
                    self.args.repulsion_only_ee_geometric_h_release_threshold
                ),
                "ee_geometric_repulsion_enter_active": ee_geometric_repulsion_enter_active,
                "ee_geometric_repulsion_release_active": ee_geometric_repulsion_release_active,
                "ee_geometric_repulsion_latched": bool(self.ee_geometric_repulsion_latched),
                "ee_block_active": ee_block_active,
                "block_nominal_component_active": block_nominal_component_active,
                "block_nominal_component_requires_ee_risk": bool(
                    self.args.block_nominal_component_requires_ee_risk
                ),
                "blocked_nominal_component": blocked_nominal_component,
                "preserve_nominal_tangent_in_repulsion": bool(
                    self.args.preserve_nominal_tangent_in_repulsion
                ),
                "repulsion_nominal_tangent_scale": float(self.args.repulsion_nominal_tangent_scale),
                "repulsion_nominal_away_scale": float(self.args.repulsion_nominal_away_scale),
                "repulsion_nominal_along_correction": repulsion_nominal_along_correction,
                "repulsion_nominal_tangent": repulsion_nominal_tangent.astype(float).tolist(),
                "repulsion_nominal_away": repulsion_nominal_away.astype(float).tolist(),
                "repulsion_preserved_nominal": repulsion_preserved_nominal.astype(float).tolist(),
                "max_safe_translation_norm": float(self.args.max_safe_translation_norm),
                "clip_safe_translation_norm": clip_safe_translation_norm,
                "clip_safe_translation_norm_only_in_emergency": bool(
                    self.args.clip_safe_translation_norm_only_in_emergency
                ),
                "candidate_translation_before_task_descent": candidate_translation_before_task_descent.astype(
                    float
                ).tolist(),
                "descent_allowed": descent_allowed,
                "task_descent": task_descent,
                "task_descent_applied": task_descent_applied,
                "task_descent_total_h_ok": task_descent_total_h_ok,
                "task_descent_ee_ok": task_descent_ee_ok,
                "task_descent_learned_ee_ok": task_descent_learned_ee_ok,
                "task_descent_geometric_ee_ok": task_descent_geometric_ee_ok,
                "max_task_descent": float(self.args.max_task_descent),
                "task_descent_h_threshold": float(self.args.task_descent_h_threshold),
                "task_descent_min_h_ee": float(self.args.task_descent_min_h_ee),
                "upright_applied": bool(self.args.keep_gripper_upright),
                "vla_rotation_preserved": bool(self.args.preserve_vla_rotation_in_safety)
                and not bool(self.args.keep_gripper_upright),
                "b": float(info["b"].detach().cpu().reshape(-1)[0]),
                "gu": float(info["gu"].detach().cpu().reshape(-1)[0]),
                "violation": float(info["violation"].detach().cpu().reshape(-1)[0]),
                "joint_correction_norm": float((u_safe - u_nominal).norm(dim=1).detach().cpu().reshape(-1)[0]),
            }
        )
        debug.update(upright_debug)
        return safe_action, debug


def create_policy_client(args: Args) -> Any:
    if args.policy_backend == "openvla_oft":
        try:
            from main.openvla_oft_policy import OpenVLAOFTPolicy
        except ImportError:
            from openvla_oft_policy import OpenVLAOFTPolicy

        return OpenVLAOFTPolicy(server_url=args.openvla_oft_server_url)

    if args.use_local_policy:
        from openpi.policies import policy_config as _policy_config
        from openpi.training import config as _config

        return _policy_config.create_trained_policy(
            _config.get_config(args.policy_config_name),
            args.checkpoint_dir,
        )

    from openpi_client import websocket_client_policy as _websocket_client_policy

    return _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name](safety_level=args.safety_level)
    logging.info("Task suite: %s, safety level: %s", args.task_suite_name, args.safety_level)

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name in ("safelibero_spatial", "safelibero_object", "safelibero_goal"):
        max_steps = 500
    elif args.task_suite_name == "safelibero_long":
        max_steps = 550
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    policy_client = create_policy_client(args)
    safety_filter = None if args.disable_safety_filter else VLSAPhysicsSafetyFilter(args)

    total_episodes, total_successes, total_safesuccesses = 0, 0, 0
    for task_id in args.task_index:
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(
            task,
            args.safety_level,
            LIBERO_ENV_RESOLUTION,
            args.seed,
            enable_safety_observations=safety_filter is not None,
        )
        if safety_filter is not None:
            safety_filter.bind_env(env)

        collides = 0
        time_steps = []
        task_episodes, task_successes = 0, 0
        task_segment = task_description.replace(" ", "_")

        _out_dir = pathlib.Path(args.video_out_path) / f"{task_segment}"
        _out_dir.mkdir(parents=True, exist_ok=True)
        out_dir = _out_dir / f"vlsa_physics_{args.safety_level}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for episode_idx in args.episode_index:
            logging.info("\nTask: %s", task_description)
            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])

            replay_images = []
            episode_debug: List[Dict[str, Any]] = []
            done = False

            t = 0
            while t < args.num_steps_wait:
                try:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                except Exception as exc:
                    logging.error("Caught exception during settle steps: %s", exc)
                    break

            obstacle_names = [n.replace("_joint0", "") for n in env.sim.model.joint_names if "obstacle" in n]
            obstacle_name = None
            for name in obstacle_names:
                p = obs[f"{name}_pos"]
                if p[2] > 0 and -0.5 < p[0] < 0.5 and -0.5 < p[1] < 0.5:
                    obstacle_name = name
                    print("Obstacle name:", name)
                    break

            initial_obstacle_pos = None
            if obstacle_name is not None:
                initial_obstacle_pos = np.asarray(obs[obstacle_name + "_pos"]).copy()
            collide_flag = False
            collide_time = 0

            logging.info("Starting episode %s...", task_episodes + 1)
            t = 0
            while t < max_steps:
                try:
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    replay_images.append(img)
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        t1 = time.time()
                        action_chunk = policy_client.infer(element)["actions"]
                        t2 = time.time()
                        assert len(action_chunk) >= args.replan_steps, (
                            f"We want to replan every {args.replan_steps} steps, "
                            f"but policy only predicts {len(action_chunk)} steps."
                        )
                        action_plan.extend(action_chunk[: args.replan_steps])
                        if t < 40 or t % 25 == 0:
                            print(f"[VLA inference] t={t} infer_time={t2 - t1:.4f}s")

                    action = np.asarray(action_plan.popleft(), dtype=np.float32)
                    safety_debug: Dict[str, Any] = {"active": False, "reason": "disabled"}
                    if safety_filter is not None:
                        action, safety_debug = safety_filter.filter_action(action, obs, env, task_description)

                    if t < 40 or t % 25 == 0 or bool(safety_debug.get("active", False)):
                        print(f"[VLSA Physics] t={t} debug={safety_debug}")
                        print(f"[VLSA Physics] t={t} action={action}")

                    if args.save_safety_debug:
                        should_log_debug = (
                            bool(safety_debug.get("active", False))
                            or args.safety_debug_every <= 1
                            or t % int(args.safety_debug_every) == 0
                        )
                        if should_log_debug:
                            episode_debug.append({"t": int(t), **safety_debug})

                    obs, reward, done, info = env.step(action.tolist())

                    if obstacle_name is not None and initial_obstacle_pos is not None and not collide_flag:
                        then_obstacle_pos = np.asarray(obs[obstacle_name + "_pos"])
                        if np.sum(np.abs(then_obstacle_pos - initial_obstacle_pos)) > 0.001:
                            print("obstacle collided")
                            collide_flag = True
                            collide_time = t
                            if args.break_on_collision:
                                break

                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as exc:
                    logging.error("Caught exception: %s", exc)
                    break

            task_episodes += 1
            total_episodes += 1

            time_steps.append(t)
            if collide_flag:
                collides += 1

            suffix = "success" if done else "failure"
            safe = "safe" if not collide_flag else "unsafe"
            video_path = out_dir / f"{episode_idx}_{suffix}_{safe}.mp4"
            imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=30)

            if args.save_safety_debug:
                debug_path = out_dir / f"{episode_idx}_{suffix}_{safe}_vlsa_physics_debug.jsonl"
                with open(debug_path, "w", encoding="utf-8") as f:
                    for record in episode_debug:
                        f.write(json.dumps(record, sort_keys=True) + "\n")

            logging.info("Success: %s", done)
            logging.info("Collision: %s", collide_flag)
            ss = done and not collide_flag
            if ss:
                total_safesuccesses += 1
            logging.info("SS (Safe Success): %s", ss)
            logging.info("# episodes completed so far: %s", total_episodes)
            logging.info("# successes: %s (%.1f%%)", total_successes, total_successes / total_episodes * 100)
            logging.info("# collides: %s (%.1f%%)", collides, collides / total_episodes * 100)
            logging.info(
                "# safesuccesses: %s (%.1f%%)",
                total_safesuccesses,
                total_safesuccesses / total_episodes * 100,
            )
            print("collide_flag:", collide_flag)
            print("collide_time:", collide_time)

        logging.info("Current task success rate: %s", float(task_successes) / float(task_episodes))
        logging.info("Current total success rate: %s", float(total_successes) / float(total_episodes))

    logging.info("Total success rate: %s", float(total_successes) / float(total_episodes))
    logging.info("Total episodes: %s", total_episodes)
    logging.info("Time steps: %s", time_steps)


def _get_libero_env(
    task: Any,
    level: str,
    resolution: int,
    seed: int,
    enable_safety_observations: bool,
) -> Tuple[Any, str]:
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    print(task_description)
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_depths": True,
    }
    if enable_safety_observations:
        env_args["camera_segmentations"] = "instance"
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eval_libero(tyro.cli(Args))
