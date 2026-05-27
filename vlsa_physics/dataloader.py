from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


REQUIRED_DATASETS = (
    "q",
    "ee_pos_world",
    "fused_pointcloud/pointcloud",
    "h_star",
    "h_star_hard",
    "h_star_robust",
    "v_rep",
    "robot_keypoints_world",
    "robot_keypoint_valid_mask",
    "robot_link_valid_mask",
    "d_gt_keypoints",
    "d_gt_links",
)


OPTIONAL_DATASETS = (
    "fused_pointcloud/valid_mask",
    "fused_pointcloud/source_camera",
    "h_star_norm",
    "h_star_hard_norm",
    "h_star_robust_norm",
    "safety_region",
    "safety_region_hard",
    "proposal_type",
    "v_rep_knn",
    "closest_keypoint",
    "closest_link",
    "closest_keypoint_distance",
    "closest_link_distance",
)


@dataclass(frozen=True)
class SceneRef:
    path: str
    task_id: int
    init_state_id: int
    num_samples: int


def _import_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required to read CBF H5 datasets. Install it in the training environment."
        ) from exc
    return h5py


def _scene_num_samples(scene_group: Any) -> int:
    if "num_samples" in scene_group.attrs:
        return int(scene_group.attrs["num_samples"])
    for name in ("q", "h_star", "fused_pointcloud/pointcloud"):
        if name in scene_group:
            return int(scene_group[name].shape[0])
    raise KeyError(f"Could not infer num_samples for scene group {scene_group.name}")


def list_scenes(h5_path: str | Path) -> List[SceneRef]:
    h5py = _import_h5py()
    scenes: List[SceneRef] = []
    with h5py.File(h5_path, "r") as hf:
        tasks = hf.get("tasks", None)
        if tasks is None:
            raise KeyError("Expected H5 layout tasks/task_xxx/scenes/init_xxx")

        for task_name in sorted(tasks.keys()):
            task_group = tasks[task_name]
            scenes_group = task_group.get("scenes", None)
            if scenes_group is None:
                continue
            task_id = int(task_group.attrs.get("task_id", len(scenes)))
            for init_name in sorted(scenes_group.keys()):
                scene_group = scenes_group[init_name]
                scenes.append(
                    SceneRef(
                        path=scene_group.name,
                        task_id=task_id,
                        init_state_id=int(scene_group.attrs.get("init_state_id", 0)),
                        num_samples=_scene_num_samples(scene_group),
                    )
                )
    if not scenes:
        raise ValueError(f"No scene groups found in {h5_path}")
    return scenes


def _split_scene_refs(
    scenes: Sequence[SceneRef],
    train_ratio: float,
    seed: int,
) -> Tuple[List[SceneRef], List[SceneRef]]:
    rng = np.random.default_rng(int(seed))
    indices = np.arange(len(scenes))
    rng.shuffle(indices)

    if len(indices) == 1:
        return [scenes[int(indices[0])]], [scenes[int(indices[0])]]

    n_train = int(round(len(indices) * float(train_ratio)))
    n_train = min(max(1, n_train), len(indices) - 1)
    train = [scenes[int(i)] for i in indices[:n_train]]
    val = [scenes[int(i)] for i in indices[n_train:]]
    return train, val


def _sample_index_from_scenes(scenes: Sequence[SceneRef]) -> List[Tuple[str, int]]:
    index: List[Tuple[str, int]] = []
    for scene in scenes:
        index.extend((scene.path, sample_idx) for sample_idx in range(scene.num_samples))
    return index


def _split_sample_index(
    scenes: Sequence[SceneRef],
    train_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    all_index = _sample_index_from_scenes(scenes)
    rng = np.random.default_rng(int(seed))
    order = np.arange(len(all_index))
    rng.shuffle(order)
    n_train = int(round(len(order) * float(train_ratio)))
    n_train = min(max(1, n_train), len(order) - 1)
    train = [all_index[int(i)] for i in order[:n_train]]
    val = [all_index[int(i)] for i in order[n_train:]]
    return train, val


class CBFSafetyDataset(Dataset):
    def __init__(
        self,
        h5_path: str | Path,
        index: Sequence[Tuple[str, int]],
        n_points: Optional[int] = None,
        random_point_subsample: bool = False,
        seed: int = 0,
    ) -> None:
        self.h5_path = str(h5_path)
        self.index = list(index)
        scene_paths = sorted({scene_path for scene_path, _ in self.index})
        self.scene_to_id = {scene_path: scene_id for scene_id, scene_path in enumerate(scene_paths)}
        self.n_points = None if n_points is None else int(n_points)
        self.random_point_subsample = bool(random_point_subsample)
        self.seed = int(seed)
        self._hf = None

    def __len__(self) -> int:
        return len(self.index)

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_hf"] = None
        return state

    def close(self) -> None:
        if self._hf is not None:
            self._hf.close()
            self._hf = None

    def _ensure_open(self):
        if self._hf is None:
            h5py = _import_h5py()
            self._hf = h5py.File(self.h5_path, "r")
        return self._hf

    def _read(self, scene_group: Any, name: str, sample_idx: int) -> Optional[np.ndarray]:
        if name not in scene_group:
            return None
        return np.asarray(scene_group[name][sample_idx])

    def _stable_int_seed(self, scene_path: str, sample_idx: int) -> int:
        key = f"{self.seed}:{scene_path}:{int(sample_idx)}".encode("utf-8")
        return int(hashlib.md5(key).hexdigest()[:8], 16)

    def _point_indices_stratified(
        self,
        valid_mask: np.ndarray,
        source_camera: np.ndarray,
        sample_seed: int,
    ) -> np.ndarray:
        total = int(valid_mask.shape[0])
        if self.n_points is None or int(self.n_points) >= total:
            return np.arange(total, dtype=np.int64)

        valid = valid_mask.astype(bool)
        source_camera = np.asarray(source_camera).reshape(-1)
        n_points = int(self.n_points)
        rng = np.random.default_rng(int(sample_seed))
        selected: List[np.ndarray] = []

        cameras = sorted(int(cam) for cam in np.unique(source_camera[valid]))
        if cameras:
            per_camera = max(1, n_points // len(cameras))
            for camera_id in cameras:
                camera_valid = np.flatnonzero(valid & (source_camera == camera_id))
                if camera_valid.shape[0] == 0:
                    continue
                take = min(per_camera, camera_valid.shape[0])
                selected.append(rng.choice(camera_valid, size=take, replace=False))

        if selected:
            indices = np.concatenate(selected).astype(np.int64)
        else:
            indices = np.zeros((0,), dtype=np.int64)

        remaining = n_points - indices.shape[0]
        if remaining > 0:
            valid_all = np.flatnonzero(valid)
            if valid_all.shape[0] > 0:
                extra = rng.choice(
                    valid_all,
                    size=remaining,
                    replace=valid_all.shape[0] < remaining,
                )
            else:
                all_indices = np.arange(total, dtype=np.int64)
                extra = rng.choice(
                    all_indices,
                    size=remaining,
                    replace=all_indices.shape[0] < remaining,
                )
            indices = np.concatenate([indices, extra.astype(np.int64)])

        if indices.shape[0] > n_points:
            indices = rng.choice(indices, size=n_points, replace=False)
        return np.sort(indices.astype(np.int64))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        scene_path, sample_idx = self.index[int(idx)]
        hf = self._ensure_open()
        scene_group = hf[scene_path]

        out: Dict[str, np.ndarray] = {}
        missing = [name for name in REQUIRED_DATASETS if name not in scene_group]
        if missing:
            raise KeyError(f"Missing required datasets in {scene_path}: {missing}")

        for name in REQUIRED_DATASETS + OPTIONAL_DATASETS:
            value = self._read(scene_group, name, sample_idx)
            if value is not None:
                out[name] = value

        pc = out["fused_pointcloud/pointcloud"].astype(np.float32)
        valid_mask = out.get("fused_pointcloud/valid_mask")
        if valid_mask is None:
            valid_mask = np.ones((pc.shape[0],), dtype=np.uint8)
        else:
            valid_mask = valid_mask.astype(np.uint8).reshape(-1)

        source_camera = out.get("fused_pointcloud/source_camera")
        if source_camera is None:
            source_camera = np.zeros((pc.shape[0],), dtype=np.int8)
        else:
            source_camera = source_camera.astype(np.int8).reshape(-1)

        point_seed = self._stable_int_seed(scene_path, sample_idx)
        point_indices = self._point_indices_stratified(
            valid_mask=valid_mask,
            source_camera=source_camera,
            sample_seed=point_seed,
        )
        pc = pc[point_indices]
        valid_mask = valid_mask[point_indices]
        source_camera = source_camera[point_indices]

        sample: Dict[str, torch.Tensor] = {
            "pc_world": torch.from_numpy(pc.astype(np.float32)),
            "valid_mask": torch.from_numpy(valid_mask.astype(np.float32)),
            "source_camera": torch.from_numpy(source_camera.astype(np.int64)),
            "q": torch.from_numpy(out["q"].astype(np.float32).reshape(-1)),
            "ee_pos_world": torch.from_numpy(out["ee_pos_world"].astype(np.float32).reshape(3)),
            "h_star": torch.from_numpy(out["h_star"].astype(np.float32).reshape(1)),
            "h_star_hard": torch.from_numpy(out["h_star_hard"].astype(np.float32).reshape(1)),
            "h_star_robust": torch.from_numpy(out["h_star_robust"].astype(np.float32).reshape(1)),
            "v_rep": torch.from_numpy(out["v_rep"].astype(np.float32).reshape(3)),
            "robot_keypoints_world": torch.from_numpy(
                np.nan_to_num(
                    out["robot_keypoints_world"].astype(np.float32),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).reshape(-1, 3)
            ),
            "robot_keypoint_valid_mask": torch.from_numpy(
                out["robot_keypoint_valid_mask"].astype(np.float32).reshape(-1)
            ),
            "robot_link_valid_mask": torch.from_numpy(
                out["robot_link_valid_mask"].astype(np.float32).reshape(-1)
            ),
            "d_gt_keypoints": torch.from_numpy(
                out["d_gt_keypoints"].astype(np.float32).reshape(-1)
            ),
            "d_gt_links": torch.from_numpy(out["d_gt_links"].astype(np.float32).reshape(-1)),
            "dataset_index": torch.tensor(idx, dtype=torch.long),
            "sample_index": torch.tensor(sample_idx, dtype=torch.long),
            "scene_index": torch.tensor(self.scene_to_id[scene_path], dtype=torch.long),
        }
        if "v_rep_knn" in out:
            sample["v_rep_knn"] = torch.from_numpy(out["v_rep_knn"].astype(np.float32).reshape(3))

        for name in (
            "h_star_norm",
            "h_star_hard_norm",
            "h_star_robust_norm",
            "safety_region",
            "safety_region_hard",
            "proposal_type",
        ):
            if name in out:
                dtype = torch.float32 if name.startswith("h_star") else torch.long
                sample[name] = torch.as_tensor(out[name].reshape(-1), dtype=dtype)

        h_scale = scene_group.file.attrs.get("h_scale", None)
        safety_margin = scene_group.file.attrs.get("safety_margin", None)
        ee_radius = scene_group.file.attrs.get("ee_radius", None)
        robot_keypoint_radius = scene_group.file.attrs.get("robot_keypoint_radius", None)
        robot_link_radius = scene_group.file.attrs.get("robot_link_radius", None)
        sample["h_scale"] = torch.tensor(
            [float(h_scale) if h_scale is not None else 0.10],
            dtype=torch.float32,
        )
        sample["safety_margin"] = torch.tensor(
            [float(safety_margin) if safety_margin is not None else 0.02],
            dtype=torch.float32,
        )
        sample["ee_radius"] = torch.tensor(
            [float(ee_radius) if ee_radius is not None else 0.04],
            dtype=torch.float32,
        )
        sample["robot_keypoint_radius"] = torch.tensor(
            [float(robot_keypoint_radius) if robot_keypoint_radius is not None else 0.04],
            dtype=torch.float32,
        )
        sample["robot_link_radius"] = torch.tensor(
            [float(robot_link_radius) if robot_link_radius is not None else 0.04],
            dtype=torch.float32,
        )
        return sample


def make_datasets(
    h5_path: str | Path,
    train_ratio: float = 0.70,
    split_mode: str = "scene",
    seed: int = 42,
    n_points: Optional[int] = None,
    random_point_subsample: bool = False,
) -> Tuple[CBFSafetyDataset, CBFSafetyDataset]:
    scenes = list_scenes(h5_path)
    if split_mode == "scene":
        train_scenes, val_scenes = _split_scene_refs(scenes, train_ratio=train_ratio, seed=seed)
        train_index = _sample_index_from_scenes(train_scenes)
        val_index = _sample_index_from_scenes(val_scenes)
    elif split_mode == "sample":
        train_index, val_index = _split_sample_index(scenes, train_ratio=train_ratio, seed=seed)
    else:
        raise ValueError(f"Unknown split_mode: {split_mode!r}. Use 'scene' or 'sample'.")

    return (
        CBFSafetyDataset(
            h5_path=h5_path,
            index=train_index,
            n_points=n_points,
            random_point_subsample=random_point_subsample,
            seed=seed,
        ),
        CBFSafetyDataset(
            h5_path=h5_path,
            index=val_index,
            n_points=n_points,
            random_point_subsample=False,
            seed=seed + 1,
        ),
    )


def get_dataloader(
    h5_path: str | Path,
    batch_size: int,
    train_ratio: float = 0.70,
    split_mode: str = "scene",
    num_workers: int = 2,
    seed: int = 42,
    n_points: Optional[int] = None,
    random_point_subsample: bool = False,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = make_datasets(
        h5_path=h5_path,
        train_ratio=train_ratio,
        split_mode=split_mode,
        seed=seed,
        n_points=n_points,
        random_point_subsample=random_point_subsample,
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=False,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=False,
    )
    return train_loader, val_loader
