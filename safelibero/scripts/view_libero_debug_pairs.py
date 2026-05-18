#!/usr/bin/env python3
"""View paired wrist/external debug pointclouds saved by generate_libero_safety_dataset.py."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


WRIST_SUFFIX = "_wrist_world.ply"
EXTERNAL_SUFFIX = "_external_world.ply"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize paired LIBERO debug PLY pointclouds.")
    parser.add_argument(
        "--ply-dir",
        default="data/libero_safety/debug_frames/ply",
        help="Directory containing cloud_*_wrist_world.ply and cloud_*_external_world.ply files.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Initial pair index after sorting by filename.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=None,
        help="Horizontal separation between the two clouds. Defaults to 1.4x scene span.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--no-center",
        action="store_true",
        help="Do not center each cloud before placing them side by side.",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Show the two clouds in their original world frames instead of side by side.",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Show only one pair and exit when the window closes.",
    )
    parser.add_argument(
        "--save-png-dir",
        default=None,
        help="Save static side-by-side PNG previews to this directory.",
    )
    parser.add_argument(
        "--save-png-max",
        type=int,
        default=200,
        help="Maximum number of PNG previews to save when --save-png-dir is set.",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Do not open the Open3D window; useful on headless / broken OpenGL sessions.",
    )
    return parser.parse_args()


def find_repo_root(start: Path) -> Path:
    for candidate in [start.resolve()] + list(start.resolve().parents):
        if (candidate / "safelibero").exists() and (candidate / "scripts").exists():
            return candidate
    return start.resolve()


def resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    repo_root = find_repo_root(Path(__file__).parent)
    return repo_root / p


def collect_pairs(ply_dir: Path) -> List[Tuple[Path, Path, str]]:
    wrist_by_prefix: Dict[str, Path] = {}
    external_by_prefix: Dict[str, Path] = {}

    for path in sorted(ply_dir.glob("cloud_*_world.ply")):
        name = path.name
        if name.endswith(WRIST_SUFFIX):
            wrist_by_prefix[name[: -len(WRIST_SUFFIX)]] = path
        elif name.endswith(EXTERNAL_SUFFIX):
            external_by_prefix[name[: -len(EXTERNAL_SUFFIX)]] = path

    pairs = []
    for prefix in sorted(set(wrist_by_prefix) & set(external_by_prefix)):
        pairs.append((wrist_by_prefix[prefix], external_by_prefix[prefix], prefix))

    missing_wrist = sorted(set(external_by_prefix) - set(wrist_by_prefix))
    missing_external = sorted(set(wrist_by_prefix) - set(external_by_prefix))
    if missing_wrist:
        print(f"[viewer] warning: {len(missing_wrist)} external clouds have no wrist pair")
    if missing_external:
        print(f"[viewer] warning: {len(missing_external)} wrist clouds have no external pair")

    return pairs


def cloud_bounds(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    if points.shape[0] == 0:
        return np.zeros(3), np.ones(3)
    return points.min(axis=0), points.max(axis=0)


def cloud_center_and_span(points_list: Sequence[np.ndarray]) -> Tuple[np.ndarray, float]:
    mins = []
    maxs = []
    for points in points_list:
        mn, mx = cloud_bounds(points)
        mins.append(mn)
        maxs.append(mx)
    mn = np.min(np.stack(mins, axis=0), axis=0)
    mx = np.max(np.stack(maxs, axis=0), axis=0)
    center = 0.5 * (mn + mx)
    span = float(np.max(mx - mn))
    return center.astype(np.float64), max(span, 0.25)


def colorize_if_needed(pcd: object, color: Tuple[float, float, float]) -> None:
    if not pcd.has_colors():
        pcd.paint_uniform_color(color)


def make_scene(
    o3d: object,
    wrist_path: Path,
    external_path: Path,
    spacing: Optional[float],
    center: bool,
    overlay: bool,
) -> List[object]:
    wrist = o3d.io.read_point_cloud(str(wrist_path))
    external = o3d.io.read_point_cloud(str(external_path))
    colorize_if_needed(wrist, (0.12, 0.47, 0.71))
    colorize_if_needed(external, (1.0, 0.50, 0.05))

    wrist_points = np.asarray(wrist.points)
    external_points = np.asarray(external.points)
    _, span = cloud_center_and_span([wrist_points, external_points])
    sep = float(spacing) if spacing is not None else 1.4 * span

    if not overlay:
        wrist = copy.deepcopy(wrist)
        external = copy.deepcopy(external)

        if center:
            wrist_center, _ = cloud_center_and_span([wrist_points])
            external_center, _ = cloud_center_and_span([external_points])
            wrist.translate(-wrist_center)
            external.translate(-external_center)

        wrist.translate((-0.5 * sep, 0.0, 0.0))
        external.translate((0.5 * sep, 0.0, 0.0))

    frame_size = max(0.08, 0.12 * span)
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
    return [wrist, external, coord]


def downsample(points: np.ndarray, max_points: int = 8000) -> np.ndarray:
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    if points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(0)
    indices = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[indices]


def set_axes_equal(ax: object, points: np.ndarray) -> None:
    if points.shape[0] == 0:
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-0.5, 0.5)
        return
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    mid = 0.5 * (mn + mx)
    span = max(float(np.max(mx - mn)), 0.2)
    radius = 0.55 * span
    ax.set_xlim(mid[0] - radius, mid[0] + radius)
    ax.set_ylim(mid[1] - radius, mid[1] + radius)
    ax.set_zlim(mid[2] - radius, mid[2] + radius)


def save_pair_png(
    o3d: object,
    wrist_path: Path,
    external_path: Path,
    output_path: Path,
    center: bool,
    overlay: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wrist = o3d.io.read_point_cloud(str(wrist_path))
    external = o3d.io.read_point_cloud(str(external_path))
    wrist_points = downsample(np.asarray(wrist.points))
    external_points = downsample(np.asarray(external.points))

    if center and not overlay:
        wrist_center, _ = cloud_center_and_span([wrist_points])
        external_center, _ = cloud_center_and_span([external_points])
        wrist_points = wrist_points - wrist_center
        external_points = external_points - external_center

    if overlay:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        if wrist_points.shape[0]:
            ax.scatter(wrist_points[:, 0], wrist_points[:, 1], wrist_points[:, 2], s=1.5, c="#1f77b4", alpha=0.65)
        if external_points.shape[0]:
            ax.scatter(external_points[:, 0], external_points[:, 1], external_points[:, 2], s=1.5, c="#ff7f0e", alpha=0.65)
        ax.set_title("overlay: wrist blue, external orange")
        set_axes_equal(ax, np.concatenate([wrist_points, external_points], axis=0))
        axes = [ax]
    else:
        fig = plt.figure(figsize=(13, 6))
        axes = [fig.add_subplot(1, 2, 1, projection="3d"), fig.add_subplot(1, 2, 2, projection="3d")]
        if wrist_points.shape[0]:
            axes[0].scatter(wrist_points[:, 0], wrist_points[:, 1], wrist_points[:, 2], s=1.5, c="#1f77b4", alpha=0.75)
        if external_points.shape[0]:
            axes[1].scatter(external_points[:, 0], external_points[:, 1], external_points[:, 2], s=1.5, c="#ff7f0e", alpha=0.75)
        axes[0].set_title("wrist / EE world")
        axes[1].set_title("external world")
        set_axes_equal(axes[0], wrist_points)
        set_axes_equal(axes[1], external_points)

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=24, azim=-55)

    fig.suptitle(wrist_path.name.replace(WRIST_SUFFIX, ""))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_pair_pngs(
    o3d: object,
    pairs: Sequence[Tuple[Path, Path, str]],
    output_dir: Path,
    max_count: int,
    center: bool,
    overlay: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, (wrist_path, external_path, prefix) in enumerate(pairs[:max_count]):
        out = output_dir / f"{idx:06d}_{prefix}.png"
        save_pair_png(o3d, wrist_path, external_path, out, center=center, overlay=overlay)
    print(f"[viewer] saved {min(len(pairs), max_count)} PNG previews in {output_dir}")


class PairViewer:
    def __init__(
        self,
        o3d: object,
        pairs: List[Tuple[Path, Path, str]],
        index: int,
        spacing: Optional[float],
        center: bool,
        overlay: bool,
        point_size: float,
        single: bool,
    ):
        self.o3d = o3d
        self.pairs = pairs
        self.index = max(0, min(index, len(pairs) - 1))
        self.spacing = spacing
        self.center = center
        self.overlay = overlay
        self.point_size = point_size
        self.single = single
        self.geometries: List[object] = []

    def title(self) -> str:
        _, _, prefix = self.pairs[self.index]
        return f"LIBERO cloud pair {self.index + 1}/{len(self.pairs)}: {prefix}"

    def load_current(self, vis: object, reset_view: bool = True) -> None:
        for geom in self.geometries:
            vis.remove_geometry(geom, reset_bounding_box=False)
        self.geometries = []

        wrist_path, external_path, prefix = self.pairs[self.index]
        self.geometries = make_scene(
            self.o3d,
            wrist_path=wrist_path,
            external_path=external_path,
            spacing=self.spacing,
            center=self.center,
            overlay=self.overlay,
        )
        for geom in self.geometries:
            vis.add_geometry(geom, reset_bounding_box=reset_view)

        print(
            f"[viewer] {self.index + 1}/{len(self.pairs)} {prefix}\n"
            f"         wrist:    {wrist_path.name}\n"
            f"         external: {external_path.name}"
        )

    def next_pair(self, vis: object) -> bool:
        if self.single:
            return False
        self.index = (self.index + 1) % len(self.pairs)
        self.load_current(vis)
        return False

    def prev_pair(self, vis: object) -> bool:
        if self.single:
            return False
        self.index = (self.index - 1) % len(self.pairs)
        self.load_current(vis)
        return False

    def run(self) -> None:
        vis = self.o3d.visualization.VisualizerWithKeyCallback()
        created = vis.create_window(window_name=self.title(), width=1400, height=900)
        if not created:
            vis.destroy_window()
            raise RuntimeError("Open3D could not create a window. Try --no-window --save-png-dir <dir>.")
        render = vis.get_render_option()
        if render is None:
            vis.destroy_window()
            raise RuntimeError("Open3D window exists but render options are unavailable. Try --no-window --save-png-dir <dir>.")
        render.point_size = float(self.point_size)
        render.background_color = np.asarray([0.02, 0.02, 0.02])

        vis.register_key_callback(ord("N"), self.next_pair)
        vis.register_key_callback(ord("P"), self.prev_pair)
        vis.register_key_callback(ord("Q"), lambda v: v.close())
        vis.register_key_callback(256, lambda v: v.close())  # ESC in GLFW.

        print("[viewer] controls: N next, P previous, Q/ESC close")
        self.load_current(vis)
        vis.run()
        vis.destroy_window()


def main() -> None:
    args = parse_args()
    global np
    import numpy as np

    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    os.environ.setdefault("XDG_SESSION_TYPE", "x11")

    ply_dir = resolve_path(args.ply_dir)
    if not ply_dir.exists():
        raise FileNotFoundError(f"PLY directory not found: {ply_dir}")

    pairs = collect_pairs(ply_dir)
    if not pairs:
        raise FileNotFoundError(f"No paired *_wrist_world.ply / *_external_world.ply files in {ply_dir}")

    import open3d as o3d

    print(f"[viewer] found {len(pairs)} pairs in {ply_dir}")

    save_png_dir = Path(args.save_png_dir) if args.save_png_dir else None
    if save_png_dir is not None and not save_png_dir.is_absolute():
        save_png_dir = resolve_path(str(save_png_dir))
    if save_png_dir is not None:
        save_pair_pngs(
            o3d=o3d,
            pairs=pairs,
            output_dir=save_png_dir,
            max_count=max(0, int(args.save_png_max)),
            center=not bool(args.no_center),
            overlay=bool(args.overlay),
        )
    if args.no_window:
        return

    viewer = PairViewer(
        o3d=o3d,
        pairs=pairs,
        index=int(args.index),
        spacing=args.spacing,
        center=not bool(args.no_center),
        overlay=bool(args.overlay),
        point_size=float(args.point_size),
        single=bool(args.single),
    )
    try:
        viewer.run()
    except RuntimeError as exc:
        print(f"[viewer] Open3D GUI failed: {exc}")
        if save_png_dir is None:
            fallback_dir = ply_dir.parent / "pointcloud_pair_previews"
            print(f"[viewer] saving fallback PNG previews to {fallback_dir}")
            save_pair_pngs(
                o3d=o3d,
                pairs=pairs,
                output_dir=fallback_dir,
                max_count=min(len(pairs), 50),
                center=not bool(args.no_center),
                overlay=bool(args.overlay),
            )


if __name__ == "__main__":
    main()
