#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline visualization of a learned CBF / safety field before VLA integration."
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--h5-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument(
        "--scene-path",
        default=None,
        help="Optional exact H5 scene group path, e.g. /tasks/task_000/scenes/init_000.",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pointcept-root", default=None)
    parser.add_argument(
        "--prediction-mode",
        choices=["geo_only", "residual", "neural_only"],
        default=None,
        help="Override checkpoint prediction mode for ablations.",
    )
    parser.add_argument(
        "--field-output",
        choices=["h_total", "h_ee", "h_arm"],
        default="h_ee",
        help="Scalar field visualized on the EE-position grid.",
    )
    parser.add_argument(
        "--unsafe-threshold",
        type=float,
        default=0.0,
        help="Show points with phi <= threshold. For h fields, 0 is the safety boundary.",
    )
    parser.add_argument("--grid-resolution", type=int, default=24)
    parser.add_argument("--grid-margin", type=float, default=0.08)
    parser.add_argument("--min-half-extent", type=float, default=0.12)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument(
        "--n-points",
        type=int,
        default=2048,
        help="Number of fused pointcloud points loaded from the H5 sample. Default uses 2048.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--slice-axis", choices=["x", "y", "z"], default="z")
    parser.add_argument(
        "--slice-value",
        type=float,
        default=None,
        help="World-coordinate slice value. Defaults to current EE coordinate on --slice-axis.",
    )
    parser.add_argument("--slice-thickness", type=float, default=0.02)
    parser.add_argument("--slice-quiver-step", type=int, default=3)
    parser.add_argument("--max-unsafe-points", type=int, default=5000)
    parser.add_argument("--max-grad-arrows", type=int, default=300)
    parser.add_argument("--grad-arrow-length", type=float, default=0.035)
    parser.set_defaults(show_interactive_gradients=True)
    parser.add_argument(
        "--show-interactive-gradients",
        dest="show_interactive_gradients",
        action="store_true",
        help="Open a rotatable Matplotlib 3D window for the gradient field at the end.",
    )
    parser.add_argument(
        "--no-show-interactive-gradients",
        dest="show_interactive_gradients",
        action="store_false",
        help="Disable the interactive 3D gradient viewer.",
    )
    parser.add_argument(
        "--matplotlib-backend",
        default=None,
        help="Optional backend override, e.g. TkAgg or QtAgg for an interactive window.",
    )
    return parser.parse_args()


def configure_matplotlib_backend(args: argparse.Namespace) -> None:
    import matplotlib

    if args.matplotlib_backend:
        matplotlib.use(str(args.matplotlib_backend), force=True)
    elif not bool(args.show_interactive_gradients):
        matplotlib.use("Agg", force=True)


def import_physics_modules() -> Tuple[Any, Any, Any]:
    repo_root = Path(__file__).resolve().parent
    physics_dir = repo_root / "vlsa_physics"
    if str(physics_dir) not in sys.path:
        sys.path.insert(0, str(physics_dir))
    from dataloader import CBFSafetyDataset, list_scenes
    from train import LinkAwarePhysicsCBFNet, normalize_panda_q

    return CBFSafetyDataset, list_scenes, (LinkAwarePhysicsCBFNet, normalize_panda_q)


def checkpoint_arg(ckpt_args: Dict[str, Any], name: str, default: Any) -> Any:
    return ckpt_args[name] if name in ckpt_args and ckpt_args[name] is not None else default


def build_model_from_checkpoint(
    checkpoint: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> torch.nn.Module:
    _, _, train_symbols = import_physics_modules()
    LinkAwarePhysicsCBFNet, _ = train_symbols
    ckpt_args = dict(checkpoint.get("args", {}))
    pointcept_root = (
        args.pointcept_root
        if args.pointcept_root is not None
        else checkpoint_arg(ckpt_args, "pointcept_root", str(Path.home() / "Claudio" / "vlsa-aegis" / "Pointcept"))
    )
    prediction_mode = (
        args.prediction_mode
        if args.prediction_mode is not None
        else checkpoint_arg(ckpt_args, "prediction_mode", "residual")
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
        default_ee_radius=float(checkpoint_arg(ckpt_args, "default_ee_radius", 0.04)),
        default_safety_margin=float(checkpoint_arg(ckpt_args, "default_safety_margin", 0.02)),
        default_robot_keypoint_radius=float(
            checkpoint_arg(ckpt_args, "default_robot_keypoint_radius", 0.04)
        ),
        default_robot_link_radius=float(checkpoint_arg(ckpt_args, "default_robot_link_radius", 0.04)),
        residual_m_scale=float(checkpoint_arg(ckpt_args, "residual_m_scale", 0.05)),
        prediction_mode=str(prediction_mode),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def load_sample(args: argparse.Namespace) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    CBFSafetyDataset, list_scenes, _ = import_physics_modules()
    if args.scene_path is None:
        scenes = list_scenes(args.h5_path)
        if args.scene_index < 0 or args.scene_index >= len(scenes):
            raise IndexError(f"--scene-index {args.scene_index} out of range [0, {len(scenes)})")
        scene_ref = scenes[int(args.scene_index)]
        scene_path = scene_ref.path
        num_samples = scene_ref.num_samples
        task_id = scene_ref.task_id
        init_state_id = scene_ref.init_state_id
    else:
        scene_path = str(args.scene_path)
        num_samples = None
        task_id = None
        init_state_id = None

    sample_index = int(args.sample_index)
    if num_samples is not None:
        sample_index = min(max(0, sample_index), int(num_samples) - 1)

    dataset = CBFSafetyDataset(
        h5_path=args.h5_path,
        index=[(scene_path, sample_index)],
        n_points=args.n_points,
        random_point_subsample=False,
        seed=args.seed,
    )
    sample = dataset[0]
    dataset.close()
    meta = {
        "scene_path": scene_path,
        "scene_index": None if args.scene_path is not None else int(args.scene_index),
        "sample_index": sample_index,
        "task_id": task_id,
        "init_state_id": init_state_id,
    }
    return sample, meta


def sample_to_device(sample: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device=device).unsqueeze(0) if torch.is_tensor(value) else value
        for key, value in sample.items()
    }


def make_grid(
    pc_world: np.ndarray,
    valid_mask: np.ndarray,
    ee_pos_world: np.ndarray,
    resolution: int,
    margin: float,
    min_half_extent: float,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
    valid_points = pc_world[valid_mask > 0.5]
    if valid_points.shape[0] == 0:
        valid_points = pc_world
    mins = valid_points.min(axis=0) - float(margin)
    maxs = valid_points.max(axis=0) + float(margin)
    center = 0.5 * (mins + maxs)
    half_extent = 0.5 * (maxs - mins)
    half_extent = np.maximum(half_extent, float(min_half_extent))
    mins = center - half_extent
    maxs = center + half_extent
    ee_pos_world = np.asarray(ee_pos_world, dtype=np.float32).reshape(3)
    mins = np.minimum(mins, ee_pos_world - float(min_half_extent))
    maxs = np.maximum(maxs, ee_pos_world + float(min_half_extent))

    xs = np.linspace(mins[0], maxs[0], int(resolution), dtype=np.float32)
    ys = np.linspace(mins[1], maxs[1], int(resolution), dtype=np.float32)
    zs = np.linspace(mins[2], maxs[2], int(resolution), dtype=np.float32)
    mesh = np.meshgrid(xs, ys, zs, indexing="ij")
    grid_points = np.stack(mesh, axis=-1).reshape(-1, 3).astype(np.float32)
    return grid_points, (xs, ys, zs), mins.astype(np.float32), maxs.astype(np.float32)


def evaluate_field(
    model: torch.nn.Module,
    sample: Dict[str, torch.Tensor],
    grid_points: np.ndarray,
    field_output: str,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    _, _, train_symbols = import_physics_modules()
    _, normalize_panda_q = train_symbols
    field_key = {
        "h_total": "h_total_pred",
        "h_ee": "h_ee_pred",
        "h_arm": "h_arm_pred",
    }[field_output]

    base = sample_to_device(sample, device)
    pc_world = base["pc_world"]
    valid_mask = base["valid_mask"]
    source_camera = base["source_camera"]
    q_norm_base = normalize_panda_q(base["q"])
    robot_keypoints_world = base["robot_keypoints_world"]
    robot_keypoint_valid_mask = base["robot_keypoint_valid_mask"]
    robot_link_valid_mask = base["robot_link_valid_mask"]
    ee_radius = base.get("ee_radius", None)
    safety_margin = base.get("safety_margin", None)
    robot_keypoint_radius = base.get("robot_keypoint_radius", None)
    robot_link_radius = base.get("robot_link_radius", None)

    phis = []
    grads = []
    grid_tensor = torch.from_numpy(grid_points.astype(np.float32)).to(device=device)
    for start in range(0, grid_tensor.shape[0], int(batch_size)):
        ee_pos = grid_tensor[start : start + int(batch_size)].clone().detach().requires_grad_(True)
        chunk = int(ee_pos.shape[0])
        with torch.enable_grad():
            outputs = model(
                pc_world=pc_world.expand(chunk, -1, -1),
                q=q_norm_base.expand(chunk, -1),
                ee_pos=ee_pos,
                valid_mask=valid_mask.expand(chunk, -1),
                source_camera=source_camera.expand(chunk, -1),
                robot_keypoints_world=robot_keypoints_world.expand(chunk, -1, -1),
                robot_keypoint_valid_mask=robot_keypoint_valid_mask.expand(chunk, -1),
                robot_link_valid_mask=robot_link_valid_mask.expand(chunk, -1),
                ee_radius=ee_radius.expand(chunk, -1) if ee_radius is not None else None,
                safety_margin=safety_margin.expand(chunk, -1) if safety_margin is not None else None,
                robot_keypoint_radius=(
                    robot_keypoint_radius.expand(chunk, -1)
                    if robot_keypoint_radius is not None
                    else None
                ),
                robot_link_radius=(
                    robot_link_radius.expand(chunk, -1) if robot_link_radius is not None else None
                ),
            )
            phi = outputs[field_key].reshape(chunk)
            if phi.requires_grad:
                grad = torch.autograd.grad(
                    outputs=phi.sum(),
                    inputs=ee_pos,
                    create_graph=False,
                    retain_graph=False,
                    only_inputs=True,
                    allow_unused=True,
                )[0]
                if grad is None:
                    grad = torch.zeros_like(ee_pos)
            else:
                grad = torch.zeros_like(ee_pos)
        phis.append(phi.detach().cpu().numpy())
        grads.append(grad.detach().cpu().numpy())
    return np.concatenate(phis, axis=0), np.concatenate(grads, axis=0)


def set_axes_equal_3d(ax: Any, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.55 * float(np.max(maxs - mins))
    radius = max(radius, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def choose_indices(mask: np.ndarray, max_count: int, seed: int) -> np.ndarray:
    indices = np.flatnonzero(mask)
    if indices.shape[0] <= int(max_count):
        return indices
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(indices, size=int(max_count), replace=False))


def save_unsafe_plot(
    output_dir: Path,
    pc_world: np.ndarray,
    valid_mask: np.ndarray,
    robot_keypoints: np.ndarray,
    robot_keypoint_mask: np.ndarray,
    ee_pos: np.ndarray,
    grid_points: np.ndarray,
    phi: np.ndarray,
    threshold: float,
    args: argparse.Namespace,
) -> None:
    import matplotlib.pyplot as plt

    valid_pc = pc_world[valid_mask > 0.5]
    unsafe_idx = choose_indices(phi <= threshold, args.max_unsafe_points, args.seed)
    if unsafe_idx.shape[0] == 0:
        unsafe_idx = np.argsort(phi)[: min(args.max_unsafe_points, phi.shape[0])]

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(valid_pc[:, 0], valid_pc[:, 1], valid_pc[:, 2], s=1.5, c="#777777", alpha=0.25)
    unsafe = grid_points[unsafe_idx]
    scatter = ax.scatter(
        unsafe[:, 0],
        unsafe[:, 1],
        unsafe[:, 2],
        s=5,
        c=phi[unsafe_idx],
        cmap="coolwarm",
        alpha=0.75,
    )
    kp_valid = robot_keypoints[robot_keypoint_mask > 0.5]
    if kp_valid.shape[0] > 0:
        ax.plot(kp_valid[:, 0], kp_valid[:, 1], kp_valid[:, 2], c="black", linewidth=1.5)
        ax.scatter(kp_valid[:, 0], kp_valid[:, 1], kp_valid[:, 2], s=18, c="black")
    ax.scatter([ee_pos[0]], [ee_pos[1]], [ee_pos[2]], s=45, c="#f2d22e", edgecolors="black")
    ax.set_title(f"Unsafe / lowest field points: {args.field_output}, threshold={threshold:.3f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.colorbar(scatter, ax=ax, shrink=0.7, label="phi [m]")
    bounds_parts = [valid_pc, unsafe]
    if kp_valid.shape[0] > 0:
        bounds_parts.append(kp_valid)
    set_axes_equal_3d(ax, np.concatenate(bounds_parts, axis=0))
    fig.tight_layout()
    fig.savefig(output_dir / "unsafe_points_3d.png", dpi=180)
    plt.close(fig)


def save_gradient_plot(
    output_dir: Path,
    pc_world: np.ndarray,
    valid_mask: np.ndarray,
    grid_points: np.ndarray,
    phi: np.ndarray,
    grads: np.ndarray,
    threshold: float,
    args: argparse.Namespace,
) -> None:
    import matplotlib.pyplot as plt

    valid_pc = pc_world[valid_mask > 0.5]
    boundary_band = max(0.02, float(args.slice_thickness))
    candidate_mask = np.abs(phi - threshold) <= boundary_band
    if not np.any(candidate_mask):
        candidate_mask = phi <= threshold
    if not np.any(candidate_mask):
        order = np.argsort(phi)
        candidate_mask = np.zeros_like(phi, dtype=bool)
        candidate_mask[order[: min(args.max_grad_arrows, phi.shape[0])]] = True
    idx = choose_indices(candidate_mask, args.max_grad_arrows, args.seed + 1)
    pts = grid_points[idx]
    vec = grads[idx]
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    vec = vec / np.maximum(norm, 1e-8)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(valid_pc[:, 0], valid_pc[:, 1], valid_pc[:, 2], s=1.5, c="#777777", alpha=0.2)
    ax.quiver(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        vec[:, 0],
        vec[:, 1],
        vec[:, 2],
        length=float(args.grad_arrow_length),
        normalize=False,
        color="#1f77b4",
        linewidth=0.8,
    )
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=8, c=phi[idx], cmap="coolwarm")
    ax.set_title(f"Gradient directions near boundary: {args.field_output}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    set_axes_equal_3d(ax, np.concatenate([valid_pc, pts], axis=0))
    fig.tight_layout()
    fig.savefig(output_dir / "gradients_3d.png", dpi=180)
    plt.close(fig)


def show_interactive_gradient_plot(
    pc_world: np.ndarray,
    valid_mask: np.ndarray,
    grid_points: np.ndarray,
    phi: np.ndarray,
    grads: np.ndarray,
    threshold: float,
    args: argparse.Namespace,
) -> None:
    import matplotlib
    import matplotlib.pyplot as plt

    backend = matplotlib.get_backend().lower()
    if "agg" in backend:
        print(
            "[interactive] Matplotlib is using a non-interactive Agg backend; "
            "no window will be opened. Try --matplotlib-backend TkAgg or QtAgg."
        )
        return

    valid_pc = pc_world[valid_mask > 0.5]
    boundary_band = max(0.02, float(args.slice_thickness))
    candidate_mask = np.abs(phi - threshold) <= boundary_band
    if not np.any(candidate_mask):
        candidate_mask = phi <= threshold
    if not np.any(candidate_mask):
        order = np.argsort(phi)
        candidate_mask = np.zeros_like(phi, dtype=bool)
        candidate_mask[order[: min(args.max_grad_arrows, phi.shape[0])]] = True

    idx = choose_indices(candidate_mask, args.max_grad_arrows, args.seed + 1)
    pts = grid_points[idx]
    vec = grads[idx]
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    vec = vec / np.maximum(norm, 1e-8)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        valid_pc[:, 0],
        valid_pc[:, 1],
        valid_pc[:, 2],
        s=2.0,
        c="#777777",
        alpha=0.22,
        label=f"fused pointcloud ({valid_pc.shape[0]} pts)",
    )
    scatter = ax.scatter(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        s=14,
        c=phi[idx],
        cmap="coolwarm",
        alpha=0.9,
        label="field samples",
    )
    ax.quiver(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        vec[:, 0],
        vec[:, 1],
        vec[:, 2],
        length=float(args.grad_arrow_length),
        normalize=False,
        color="#1f77b4",
        linewidth=0.9,
        label="grad phi",
    )
    ax.set_title(
        f"Interactive gradients: {args.field_output}\n"
        "Drag with mouse to rotate, scroll to zoom, close window to finish"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper right")
    fig.colorbar(scatter, ax=ax, shrink=0.7, label="phi [m]")
    set_axes_equal_3d(ax, np.concatenate([valid_pc, pts], axis=0))
    fig.tight_layout()
    print("[interactive] opening rotatable 3D gradient window; close it to finish.")
    plt.show(block=True)


def slice_arrays(
    phi_grid: np.ndarray,
    grad_grid: np.ndarray,
    axes: Tuple[np.ndarray, np.ndarray, np.ndarray],
    axis_name: str,
    slice_value: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float]:
    xs, ys, zs = axes
    axis_id = {"x": 0, "y": 1, "z": 2}[axis_name]
    coord = axes[axis_id]
    idx = int(np.argmin(np.abs(coord - float(slice_value))))
    actual = float(coord[idx])
    if axis_name == "z":
        image = phi_grid[:, :, idx].T
        grad_u = grad_grid[:, :, idx, 0].T
        grad_v = grad_grid[:, :, idx, 1].T
        u_axis, v_axis = xs, ys
    elif axis_name == "y":
        image = phi_grid[:, idx, :].T
        grad_u = grad_grid[:, idx, :, 0].T
        grad_v = grad_grid[:, idx, :, 2].T
        u_axis, v_axis = xs, zs
    else:
        image = phi_grid[idx, :, :].T
        grad_u = grad_grid[idx, :, :, 1].T
        grad_v = grad_grid[idx, :, :, 2].T
        u_axis, v_axis = ys, zs
    return image, grad_u, grad_v, u_axis, v_axis, idx, actual


def project_points_for_slice(
    points: np.ndarray,
    axis_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis_id = {"x": 0, "y": 1, "z": 2}[axis_name]
    if axis_name == "z":
        return points[:, 0], points[:, 1], points[:, axis_id]
    if axis_name == "y":
        return points[:, 0], points[:, 2], points[:, axis_id]
    return points[:, 1], points[:, 2], points[:, axis_id]


def save_slice_plot(
    output_dir: Path,
    pc_world: np.ndarray,
    valid_mask: np.ndarray,
    ee_pos: np.ndarray,
    phi_grid: np.ndarray,
    grad_grid: np.ndarray,
    axes: Tuple[np.ndarray, np.ndarray, np.ndarray],
    threshold: float,
    args: argparse.Namespace,
) -> None:
    import matplotlib.pyplot as plt

    axis_id = {"x": 0, "y": 1, "z": 2}[args.slice_axis]
    desired_value = (
        float(args.slice_value)
        if args.slice_value is not None
        else float(np.asarray(ee_pos).reshape(3)[axis_id])
    )
    image, grad_u, grad_v, u_axis, v_axis, slice_idx, actual_value = slice_arrays(
        phi_grid,
        grad_grid,
        axes,
        args.slice_axis,
        desired_value,
    )
    valid_pc = pc_world[valid_mask > 0.5]
    pc_u, pc_v, pc_axis = project_points_for_slice(valid_pc, args.slice_axis)
    near = np.abs(pc_axis - actual_value) <= float(args.slice_thickness)
    ee_u, ee_v, _ = project_points_for_slice(np.asarray(ee_pos).reshape(1, 3), args.slice_axis)

    fig, ax = plt.subplots(figsize=(9, 7))
    extent = [float(u_axis[0]), float(u_axis[-1]), float(v_axis[0]), float(v_axis[-1])]
    im = ax.imshow(
        image,
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        aspect="auto",
    )
    ax.contour(
        u_axis,
        v_axis,
        image,
        levels=[float(threshold)],
        colors="black",
        linewidths=1.2,
    )
    if np.any(near):
        ax.scatter(pc_u[near], pc_v[near], s=5, c="#222222", alpha=0.4)
    step = max(1, int(args.slice_quiver_step))
    uu, vv = np.meshgrid(u_axis, v_axis, indexing="xy")
    grad_norm = np.sqrt(grad_u * grad_u + grad_v * grad_v)
    ax.quiver(
        uu[::step, ::step],
        vv[::step, ::step],
        grad_u[::step, ::step] / np.maximum(grad_norm[::step, ::step], 1e-8),
        grad_v[::step, ::step] / np.maximum(grad_norm[::step, ::step], 1e-8),
        color="black",
        alpha=0.55,
        scale=28,
        width=0.0022,
    )
    ax.scatter(ee_u, ee_v, s=55, c="#f2d22e", edgecolors="black")
    ax.set_title(
        f"{args.field_output} slice {args.slice_axis}={actual_value:.3f}, "
        f"contour={threshold:.3f}"
    )
    ax.set_xlabel({"x": "y", "y": "x", "z": "x"}[args.slice_axis])
    ax.set_ylabel({"x": "z", "y": "z", "z": "y"}[args.slice_axis])
    fig.colorbar(im, ax=ax, label="phi [m]")
    fig.tight_layout()
    fig.savefig(output_dir / f"slice_2d_{args.slice_axis}.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_matplotlib_backend(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model = build_model_from_checkpoint(checkpoint, args, device)
    sample, sample_meta = load_sample(args)

    pc_world = sample["pc_world"].numpy()
    valid_mask = sample["valid_mask"].numpy()
    ee_pos = sample["ee_pos_world"].numpy()
    robot_keypoints = sample["robot_keypoints_world"].numpy()
    robot_keypoint_mask = sample["robot_keypoint_valid_mask"].numpy()

    grid_points, axes, grid_min, grid_max = make_grid(
        pc_world=pc_world,
        valid_mask=valid_mask,
        ee_pos_world=ee_pos,
        resolution=int(args.grid_resolution),
        margin=float(args.grid_margin),
        min_half_extent=float(args.min_half_extent),
    )
    phi, grads = evaluate_field(
        model=model,
        sample=sample,
        grid_points=grid_points,
        field_output=str(args.field_output),
        batch_size=int(args.eval_batch_size),
        device=device,
    )

    resolution = int(args.grid_resolution)
    phi_grid = phi.reshape(resolution, resolution, resolution)
    grad_grid = grads.reshape(resolution, resolution, resolution, 3)
    np.savez_compressed(
        output_dir / "field_grid.npz",
        grid_points=grid_points,
        phi=phi,
        gradients=grads,
        phi_grid=phi_grid,
        grad_grid=grad_grid,
        x=axes[0],
        y=axes[1],
        z=axes[2],
        grid_min=grid_min,
        grid_max=grid_max,
    )

    threshold = float(args.unsafe_threshold)
    save_unsafe_plot(
        output_dir=output_dir,
        pc_world=pc_world,
        valid_mask=valid_mask,
        robot_keypoints=robot_keypoints,
        robot_keypoint_mask=robot_keypoint_mask,
        ee_pos=ee_pos,
        grid_points=grid_points,
        phi=phi,
        threshold=threshold,
        args=args,
    )
    save_gradient_plot(
        output_dir=output_dir,
        pc_world=pc_world,
        valid_mask=valid_mask,
        grid_points=grid_points,
        phi=phi,
        grads=grads,
        threshold=threshold,
        args=args,
    )
    save_slice_plot(
        output_dir=output_dir,
        pc_world=pc_world,
        valid_mask=valid_mask,
        ee_pos=ee_pos,
        phi_grid=phi_grid,
        grad_grid=grad_grid,
        axes=axes,
        threshold=threshold,
        args=args,
    )

    metadata = {
        "checkpoint_path": str(args.checkpoint_path),
        "h5_path": str(args.h5_path),
        "output_dir": str(output_dir),
        "field_output": str(args.field_output),
        "unsafe_threshold": threshold,
        "grid_resolution": resolution,
        "grid_point_count": int(grid_points.shape[0]),
        "grid_min": grid_min.tolist(),
        "grid_max": grid_max.tolist(),
        "phi_min": float(np.min(phi)),
        "phi_max": float(np.max(phi)),
        "phi_mean": float(np.mean(phi)),
        "unsafe_count": int(np.sum(phi <= threshold)),
        "sample": sample_meta,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    print(json.dumps(metadata, indent=2, sort_keys=True))
    if bool(args.show_interactive_gradients):
        show_interactive_gradient_plot(
            pc_world=pc_world,
            valid_mask=valid_mask,
            grid_points=grid_points,
            phi=phi,
            grads=grads,
            threshold=threshold,
            args=args,
        )


if __name__ == "__main__":
    main()
