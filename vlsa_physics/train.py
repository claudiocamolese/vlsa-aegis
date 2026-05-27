import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

os.environ.setdefault("CUMM_CUDA_ARCH_LIST", "9.0+PTX")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0+PTX")

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dataloader import get_dataloader
from loss import PhysicsArmCBFLoss, masked_softargmin_value


DEFAULT_COMET_API_KEY = "udiGvywgHQaHC30AylI8hOeyI"


PANDA_Q_LOWER = torch.tensor(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
    dtype=torch.float32,
)
PANDA_Q_UPPER = torch.tensor(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
    dtype=torch.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a link-aware physics-informed CBF model on SafeLIBERO H5 datasets."
    )

    # ---------------------------------------------------------------------
    # Data / training
    # ---------------------------------------------------------------------
    parser.add_argument("--h5-path", required=True)
    parser.add_argument("--checkpoint-path", default="vlsa_physics/checkpoints/physics_ptv3.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--split-mode", choices=["scene", "sample"], default="scene")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--n-points", type=int, default=None)
    parser.add_argument("--random-point-subsample", action="store_true")

    # ---------------------------------------------------------------------
    # Checkpoint selection
    # ---------------------------------------------------------------------
    parser.add_argument(
        "--safety-score-fu-weight",
        type=float,
        default=0.0,
        help=(
            "Optional weight for false-unsafe rates in best-safety checkpoint. "
            "Default 0.0 means best safety is selected only by false-safe rates."
        ),
    )

    # ---------------------------------------------------------------------
    # Pointcept / PTv3
    # ---------------------------------------------------------------------
    parser.add_argument(
        "--pointcept-root",
        default=str(Path.home() / "Claudio" / "vlsa-aegis" / "Pointcept"),
    )
    parser.add_argument("--ptv3-grid-size", type=float, default=0.02)
    parser.add_argument("--ptv3-patch-size", type=int, default=128)

    parser.set_defaults(ptv3_enable_flash=True)
    parser.add_argument("--ptv3-enable-flash", dest="ptv3_enable_flash", action="store_true")
    parser.add_argument("--no-ptv3-enable-flash", dest="ptv3_enable_flash", action="store_false")

    parser.add_argument("--ptv3-order", nargs="+", default=["z"])
    parser.add_argument("--camera-embedding-dim", type=int, default=16)
    parser.add_argument("--ptv3-in-channels", type=int, default=20)
    parser.add_argument("--ptv3-out-dim", type=int, default=512)

    # ---------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--robot-id-embedding-dim", type=int, default=32)
    parser.add_argument("--max-robot-keypoints", type=int, default=16)
    parser.add_argument("--fusion-layers", type=int, default=3)
    parser.add_argument("--fusion-heads", type=int, default=8)
    parser.add_argument("--fusion-ffn-dim", type=int, default=1024)
    parser.add_argument("--head-hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)

    # ---------------------------------------------------------------------
    # Geometry priors
    # ---------------------------------------------------------------------
    parser.add_argument("--geo-tau", type=float, default=0.01)
    parser.add_argument("--softmin-tau", type=float, default=0.02)
    parser.add_argument("--default-ee-radius", type=float, default=0.04)
    parser.add_argument("--default-safety-margin", type=float, default=0.02)
    parser.add_argument("--default-robot-keypoint-radius", type=float, default=0.04)
    parser.add_argument("--default-robot-link-radius", type=float, default=0.04)
    parser.add_argument(
        "--residual-m-scale",
        type=float,
        default=0.05,
        help="Maximum metric residual added to geometric priors, in meters.",
    )
    parser.add_argument(
        "--prediction-mode",
        choices=["geo_only", "residual", "neural_only"],
        default="residual",
        help=(
            "Ablation mode: geo_only uses only geometric priors, residual adds "
            "bounded neural corrections, neural_only predicts metric distances "
            "directly with a bounded tanh head."
        ),
    )

    # ---------------------------------------------------------------------
    # Loss weights
    # ---------------------------------------------------------------------
    parser.add_argument("--lambda-ee", type=float, default=1.0)
    parser.add_argument("--lambda-link", type=float, default=1.0)
    parser.add_argument("--lambda-kp", type=float, default=0.5)
    parser.add_argument("--lambda-min", type=float, default=1.0)
    parser.add_argument("--lambda-cls", type=float, default=0.2)
    parser.add_argument("--lambda-rep", type=float, default=0.0)
    parser.add_argument("--lambda-cons", type=float, default=0.3)
    parser.add_argument("--lambda-false-safe", type=float, default=0.0)
    parser.add_argument("--lambda-eik", type=float, default=0.0)
    parser.add_argument("--lambda-rank", type=float, default=0.0)

    parser.add_argument("--smooth-l1-beta", type=float, default=0.02)
    parser.add_argument("--cls-beta", type=float, default=25.0)
    parser.add_argument("--conservative-sigma", type=float, default=0.05)
    parser.add_argument("--eik-boundary-band", type=float, default=0.04)

    # ---------------------------------------------------------------------
    # Comet ML tracking
    # ---------------------------------------------------------------------
    parser.set_defaults(track=True)
    parser.add_argument("--track", dest="track", action="store_true")
    parser.add_argument("--no-track", dest="track", action="store_false")
    parser.add_argument("--comet-api-key", default=DEFAULT_COMET_API_KEY)
    parser.add_argument("--comet-project-name", default="vlsa-physics")
    parser.add_argument("--comet-workspace", default=None)
    parser.add_argument("--comet-experiment-name", default=None)

    return parser.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def sanitized_args_dict(args: argparse.Namespace) -> Dict[str, Any]:
    values: Dict[str, Any] = dict(vars(args))
    if values.get("comet_api_key"):
        values["comet_api_key"] = "***"
    return values


def setup_comet(
    args: argparse.Namespace,
    *,
    total_params: int,
    trainable_params: int,
) -> Any:
    if not bool(args.track):
        return None

    try:
        import comet_ml
    except ImportError:
        print("[comet] comet_ml is not installed; continuing without tracking.")
        return None

    api_key = str(args.comet_api_key or os.environ.get("COMET_API_KEY", "")).strip()
    if not api_key:
        print("[comet] missing API key; continuing without tracking.")
        return None

    try:
        if hasattr(comet_ml, "login"):
            comet_ml.login(api_key=api_key)

        if hasattr(comet_ml, "start"):
            start_kwargs: Dict[str, Any] = {"project_name": str(args.comet_project_name)}
            if args.comet_workspace:
                start_kwargs["workspace"] = str(args.comet_workspace)
            experiment = comet_ml.start(**start_kwargs)
        else:
            experiment_kwargs: Dict[str, Any] = {
                "api_key": api_key,
                "project_name": str(args.comet_project_name),
            }
            if args.comet_workspace:
                experiment_kwargs["workspace"] = str(args.comet_workspace)
            experiment = comet_ml.Experiment(**experiment_kwargs)

        experiment_name = args.comet_experiment_name
        if not experiment_name:
            experiment_name = (
                f"physics_{Path(args.h5_path).stem}_"
                f"{args.prediction_mode}_bs{args.batch_size}_lr{args.lr}"
            )
        experiment.set_name(str(experiment_name))
        experiment.log_parameters(sanitized_args_dict(args))
        experiment.log_metrics(
            {
                "total_params": float(total_params),
                "trainable_params": float(trainable_params),
            },
            step=0,
        )
        print(f"[comet] tracking enabled: project={args.comet_project_name} name={experiment_name}")
        return experiment
    except Exception as exc:
        print(f"[comet] could not start tracking: {exc}. Continuing without tracking.")
        return None


def log_comet_metrics(
    experiment: Any,
    prefix: str,
    metrics: Dict[str, float],
    step: int,
) -> None:
    if experiment is None:
        return
    payload = {
        f"{prefix}_{key}": float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float))
    }
    if payload:
        experiment.log_metrics(payload, step=int(step))


def ensure_pointcept_import(pointcept_root: str | Path) -> None:
    root = Path(pointcept_root).expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def build_ptv3(
    *,
    pointcept_root: str | Path,
    enable_flash: bool,
    order: Iterable[str],
    patch_size: int,
    in_channels: int,
) -> nn.Module:
    ensure_pointcept_import(pointcept_root)
    from pointcept.models import MODELS

    model_cfg = dict(
        type="PT-v3m1",
        in_channels=int(in_channels),
        order=tuple(order),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(patch_size, patch_size, patch_size, patch_size, patch_size),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(patch_size, patch_size, patch_size, patch_size),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=bool(enable_flash),
        enc_mode=True,
        upcast_attention=not bool(enable_flash),
        upcast_softmax=not bool(enable_flash),
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    )
    return MODELS.build(model_cfg)


def normalize_panda_q(q: torch.Tensor) -> torch.Tensor:
    lower = PANDA_Q_LOWER.to(device=q.device, dtype=q.dtype).reshape(1, 7)
    upper = PANDA_Q_UPPER.to(device=q.device, dtype=q.dtype).reshape(1, 7)
    return (2.0 * (q - lower) / (upper - lower).clamp_min(1e-6) - 1.0).clamp(-2.0, 2.0)


def batch_from_offset(offset: torch.Tensor, total_points: int) -> torch.Tensor:
    counts = torch.diff(
        torch.cat(
            [
                torch.zeros((1,), device=offset.device, dtype=offset.dtype),
                offset.to(dtype=offset.dtype),
            ]
        )
    )
    return torch.arange(counts.shape[0], device=offset.device).repeat_interleave(counts.long())[
        :total_points
    ]


def pad_tokens_by_batch(
    feat: torch.Tensor,
    batch: torch.Tensor,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    groups: List[torch.Tensor] = []
    max_tokens = 1

    for batch_idx in range(batch_size):
        token = feat[batch.long() == batch_idx]
        if token.shape[0] == 0:
            token = feat.new_zeros((1, feat.shape[-1]))
        groups.append(token)
        max_tokens = max(max_tokens, int(token.shape[0]))

    padded = feat.new_zeros((batch_size, max_tokens, feat.shape[-1]))
    key_padding_mask = torch.ones((batch_size, max_tokens), device=feat.device, dtype=torch.bool)

    for batch_idx, token in enumerate(groups):
        count = int(token.shape[0])
        padded[batch_idx, :count] = token
        key_padding_mask[batch_idx, :count] = False

    return padded, key_padding_mask


def masked_softargmin_dim(
    values: torch.Tensor,
    mask: torch.Tensor,
    dim: int,
    tau: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    valid = mask.bool() & torch.isfinite(values)
    logits = -values / max(float(tau), eps)
    logits = torch.where(valid, logits, torch.full_like(logits, -1e9))

    weights = torch.softmax(logits, dim=dim)
    weights = torch.where(valid, weights, torch.zeros_like(weights))
    weights = weights / weights.sum(dim=dim, keepdim=True).clamp_min(eps)

    cleaned = torch.where(valid, values, torch.zeros_like(values))
    return torch.sum(weights * cleaned, dim=dim)


def point_to_segment_distances(
    points: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
) -> torch.Tensor:
    axis = end - start
    denom = torch.sum(axis * axis, dim=-1, keepdim=True).clamp_min(1e-10)
    rel = points - start
    t = torch.sum(rel * axis, dim=-1, keepdim=True) / denom
    t = t.clamp(0.0, 1.0)
    closest = start + t * axis
    return torch.linalg.norm(points - closest, dim=-1)


class MLP(nn.Module):
    def __init__(self, dims: List[int], dropout: float = 0.0, final_norm: bool = False) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[idx + 1]))
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            elif final_norm:
                layers.append(nn.LayerNorm(dims[idx + 1]))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RobotPointFusionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()

        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

        self.cross_query_norm = nn.LayerNorm(dim)
        self.cross_context_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        robot_tokens: torch.Tensor,
        point_tokens: torch.Tensor,
        point_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        normed_robot = self.self_norm(robot_tokens)
        self_out, _ = self.self_attn(
            normed_robot,
            normed_robot,
            normed_robot,
            need_weights=False,
        )
        robot_tokens = robot_tokens + self_out

        context = self.cross_context_norm(point_tokens)
        cross_out, _ = self.cross_attn(
            query=self.cross_query_norm(robot_tokens),
            key=context,
            value=context,
            key_padding_mask=point_key_padding_mask,
            need_weights=False,
        )
        robot_tokens = robot_tokens + cross_out

        robot_tokens = robot_tokens + self.ffn(self.ffn_norm(robot_tokens))
        return robot_tokens


class LinkAwarePhysicsCBFNet(nn.Module):
    def __init__(
        self,
        *,
        pointcept_root: str | Path,
        ptv3_grid_size: float,
        ptv3_patch_size: int,
        ptv3_enable_flash: bool,
        ptv3_order: Iterable[str],
        ptv3_in_channels: int,
        ptv3_out_dim: int,
        camera_embedding_dim: int,
        model_dim: int,
        robot_id_embedding_dim: int,
        max_robot_keypoints: int,
        fusion_layers: int,
        fusion_heads: int,
        fusion_ffn_dim: int,
        head_hidden_dim: int,
        dropout: float,
        geo_tau: float,
        softmin_tau: float,
        default_ee_radius: float,
        default_safety_margin: float,
        default_robot_keypoint_radius: float,
        default_robot_link_radius: float,
        residual_m_scale: float,
        prediction_mode: str,
    ) -> None:
        super().__init__()

        if prediction_mode not in {"geo_only", "residual", "neural_only"}:
            raise ValueError(f"Unknown prediction_mode: {prediction_mode!r}")

        expected_in_channels = 4 + int(camera_embedding_dim)
        if int(ptv3_in_channels) != expected_in_channels:
            raise ValueError(
                f"PTv3 in_channels must be 4 + camera_embedding_dim = {expected_in_channels}, "
                f"got {ptv3_in_channels}."
            )

        self.ptv3 = build_ptv3(
            pointcept_root=pointcept_root,
            enable_flash=ptv3_enable_flash,
            order=ptv3_order,
            patch_size=int(ptv3_patch_size),
            in_channels=int(ptv3_in_channels),
        )

        self.camera_embedding = nn.Embedding(3, int(camera_embedding_dim))

        self.token_proj = nn.Sequential(
            nn.Linear(int(ptv3_out_dim), int(model_dim)),
            nn.LayerNorm(int(model_dim)),
            nn.GELU(),
        )

        self.keypoint_id_embedding = nn.Embedding(
            int(max_robot_keypoints),
            int(robot_id_embedding_dim),
        )
        self.link_id_embedding = nn.Embedding(
            max(1, int(max_robot_keypoints) - 1),
            int(robot_id_embedding_dim),
        )

        self.keypoint_mlp = MLP(
            [7 + 3 + int(robot_id_embedding_dim) + 1, 128, int(model_dim)],
            dropout=dropout,
            final_norm=True,
        )
        self.link_mlp = MLP(
            [7 + 3 + 3 + int(robot_id_embedding_dim) + 1, 128, int(model_dim)],
            dropout=dropout,
            final_norm=True,
        )
        self.ee_mlp = MLP([7 + 2, 128, int(model_dim)], dropout=dropout, final_norm=True)

        self.fusion_blocks = nn.ModuleList(
            [
                RobotPointFusionBlock(
                    dim=int(model_dim),
                    heads=int(fusion_heads),
                    ffn_dim=int(fusion_ffn_dim),
                    dropout=float(dropout),
                )
                for _ in range(int(fusion_layers))
            ]
        )

        self.kp_head = MLP([int(model_dim), int(head_hidden_dim), 64, 1], dropout=dropout)
        self.link_head = MLP([int(model_dim), int(head_hidden_dim), 64, 1], dropout=dropout)
        self.ee_head = MLP([int(model_dim), int(head_hidden_dim), 64, 1], dropout=dropout)
        self.rep_head = MLP([int(model_dim), int(head_hidden_dim), 3], dropout=dropout)

        self.ptv3_grid_size = float(ptv3_grid_size)
        self.geo_tau = float(geo_tau)
        self.softmin_tau = float(softmin_tau)
        self.default_ee_radius = float(default_ee_radius)
        self.default_safety_margin = float(default_safety_margin)
        self.default_robot_keypoint_radius = float(default_robot_keypoint_radius)
        self.default_robot_link_radius = float(default_robot_link_radius)
        self.residual_m_scale = float(residual_m_scale)
        self.prediction_mode = str(prediction_mode)
        self.max_robot_keypoints = int(max_robot_keypoints)

    def _make_point_dict(
        self,
        pc_world: torch.Tensor,
        ee_pos: torch.Tensor,
        valid_mask: torch.Tensor,
        source_camera: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        coords = []
        feats = []
        batches = []
        counts = []

        batch_size = int(pc_world.shape[0])

        for batch_idx in range(batch_size):
            valid = valid_mask[batch_idx] > 0.5

            if not torch.any(valid):
                rel = ee_pos[batch_idx : batch_idx + 1] * 0.0
                src_idx = torch.zeros((1,), device=pc_world.device, dtype=torch.long)
            else:
                pts = pc_world[batch_idx, valid]
                rel = pts - ee_pos[batch_idx].reshape(1, 3)
                src_idx = (source_camera[batch_idx, valid].long() + 1).clamp(0, 2)

            dist = torch.linalg.norm(rel, dim=-1, keepdim=True)
            cam_emb = self.camera_embedding(src_idx).to(dtype=pc_world.dtype)
            feat = torch.cat([rel, dist, cam_emb], dim=-1)

            coords.append(rel)
            feats.append(feat)
            batches.append(
                torch.full(
                    (rel.shape[0],),
                    batch_idx,
                    device=pc_world.device,
                    dtype=torch.long,
                )
            )
            counts.append(rel.shape[0])

        coord = torch.cat(coords, dim=0).contiguous()
        feat = torch.cat(feats, dim=0).contiguous()
        batch = torch.cat(batches, dim=0).contiguous()

        coord_for_grid = coord.detach()
        grid_coord = torch.floor(
            (coord_for_grid - coord_for_grid.min(dim=0).values.reshape(1, 3))
            / max(self.ptv3_grid_size, 1e-6)
        ).to(torch.int32)

        offset = torch.cumsum(
            torch.tensor(counts, device=pc_world.device, dtype=torch.int32),
            dim=0,
        )

        return {
            "coord": coord,
            "grid_coord": grid_coord,
            "feat": feat,
            "batch": batch,
            "offset": offset,
        }

    def _ptv3_tokens(
        self,
        point_dict: Dict[str, torch.Tensor],
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.ptv3(point_dict)

        if isinstance(out, dict):
            feat = out["feat"]
            batch = out.get("batch", None)
            offset = out.get("offset", None)
        else:
            feat = out.feat
            batch = getattr(out, "batch", None)
            offset = getattr(out, "offset", None)

        if batch is None or batch.shape[0] != feat.shape[0]:
            if offset is not None:
                batch = batch_from_offset(offset, total_points=feat.shape[0])
            elif point_dict["batch"].shape[0] == feat.shape[0]:
                batch = point_dict["batch"]
            else:
                raise RuntimeError("Could not recover PTv3 batch ids for pooled features.")

        feat = self.token_proj(feat)
        return pad_tokens_by_batch(feat, batch, batch_size=batch_size)

    def _geometric_priors(
        self,
        *,
        pc_world: torch.Tensor,
        ee_pos: torch.Tensor,
        valid_mask: torch.Tensor,
        robot_keypoints_world: torch.Tensor,
        keypoint_valid_mask: torch.Tensor,
        link_valid_mask: torch.Tensor,
        ee_radius: torch.Tensor,
        safety_margin: torch.Tensor,
        keypoint_radius: torch.Tensor,
        link_radius: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, _, _ = pc_world.shape
        point_valid = valid_mask.bool()

        ee_dist = torch.linalg.norm(pc_world - ee_pos.unsqueeze(1), dim=-1)
        ee_clearance = ee_dist - ee_radius - safety_margin
        h_ee_geo = masked_softargmin_value(
            ee_clearance,
            mask=point_valid,
            tau=self.geo_tau,
        )

        kp_dist = torch.linalg.norm(
            pc_world.unsqueeze(2) - robot_keypoints_world.unsqueeze(1),
            dim=-1,
        )
        kp_clearance = kp_dist - keypoint_radius.reshape(batch_size, 1, 1)
        kp_mask = point_valid.unsqueeze(2) & keypoint_valid_mask.bool().unsqueeze(1)
        d_keypoints_geo = masked_softargmin_dim(
            kp_clearance,
            mask=kp_mask,
            dim=1,
            tau=self.geo_tau,
        )

        start = robot_keypoints_world[:, :-1, :]
        end = robot_keypoints_world[:, 1:, :]

        link_dist = point_to_segment_distances(
            pc_world.unsqueeze(2),
            start.unsqueeze(1),
            end.unsqueeze(1),
        )
        link_clearance = link_dist - link_radius.reshape(batch_size, 1, 1)
        link_mask = point_valid.unsqueeze(2) & link_valid_mask.bool().unsqueeze(1)
        d_links_geo = masked_softargmin_dim(
            link_clearance,
            mask=link_mask,
            dim=1,
            tau=self.geo_tau,
        )

        return {
            "h_ee_geo": h_ee_geo,
            "d_keypoints_geo": d_keypoints_geo,
            "d_links_geo": d_links_geo,
        }

    def forward(
        self,
        *,
        pc_world: torch.Tensor,
        q: torch.Tensor,
        ee_pos: torch.Tensor,
        valid_mask: torch.Tensor,
        source_camera: torch.Tensor,
        robot_keypoints_world: torch.Tensor,
        robot_keypoint_valid_mask: torch.Tensor,
        robot_link_valid_mask: torch.Tensor,
        ee_radius: torch.Tensor | None = None,
        safety_margin: torch.Tensor | None = None,
        robot_keypoint_radius: torch.Tensor | None = None,
        robot_link_radius: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = int(pc_world.shape[0])

        valid_mask = valid_mask.to(dtype=pc_world.dtype)
        source_camera = source_camera.to(device=pc_world.device)
        robot_keypoint_valid_mask = robot_keypoint_valid_mask.to(
            device=pc_world.device,
            dtype=pc_world.dtype,
        )
        robot_link_valid_mask = robot_link_valid_mask.to(
            device=pc_world.device,
            dtype=pc_world.dtype,
        )

        def column_or_default(value: torch.Tensor | None, default: float) -> torch.Tensor:
            if value is None:
                return pc_world.new_full((batch_size, 1), float(default))
            return value.reshape(batch_size, 1).to(device=pc_world.device, dtype=pc_world.dtype)

        ee_radius = column_or_default(ee_radius, self.default_ee_radius)
        safety_margin = column_or_default(safety_margin, self.default_safety_margin)
        robot_keypoint_radius = column_or_default(
            robot_keypoint_radius,
            self.default_robot_keypoint_radius,
        )
        robot_link_radius = column_or_default(
            robot_link_radius,
            self.default_robot_link_radius,
        )

        priors = self._geometric_priors(
            pc_world=pc_world,
            ee_pos=ee_pos,
            valid_mask=valid_mask,
            robot_keypoints_world=robot_keypoints_world,
            keypoint_valid_mask=robot_keypoint_valid_mask,
            link_valid_mask=robot_link_valid_mask,
            ee_radius=ee_radius,
            safety_margin=safety_margin,
            keypoint_radius=robot_keypoint_radius,
            link_radius=robot_link_radius,
        )

        point_dict = self._make_point_dict(
            pc_world=pc_world,
            ee_pos=ee_pos,
            valid_mask=valid_mask,
            source_camera=source_camera,
        )
        point_tokens, point_key_padding_mask = self._ptv3_tokens(
            point_dict,
            batch_size=batch_size,
        )

        keypoints_rel = robot_keypoints_world - ee_pos.unsqueeze(1)
        keypoints_rel = keypoints_rel * robot_keypoint_valid_mask.unsqueeze(-1)

        num_keypoints = int(keypoints_rel.shape[1])
        num_links = max(0, num_keypoints - 1)

        if num_keypoints > self.max_robot_keypoints:
            raise ValueError(
                f"Dataset has {num_keypoints} keypoints but model max is {self.max_robot_keypoints}."
            )

        q_tokens = q.unsqueeze(1).expand(-1, num_keypoints, -1)

        kp_ids = torch.arange(num_keypoints, device=pc_world.device).clamp_max(
            self.max_robot_keypoints - 1
        )
        kp_id_emb = self.keypoint_id_embedding(kp_ids).unsqueeze(0).expand(batch_size, -1, -1)

        kp_input = torch.cat(
            [q_tokens, keypoints_rel, kp_id_emb, robot_keypoint_valid_mask.unsqueeze(-1)],
            dim=-1,
        )
        kp_tokens = self.keypoint_mlp(kp_input)

        link_start = keypoints_rel[:, :-1, :]
        link_end = keypoints_rel[:, 1:, :]
        link_mid = 0.5 * (link_start + link_end)
        link_vec = link_end - link_start

        link_ids = torch.arange(num_links, device=pc_world.device).clamp_max(
            max(0, self.max_robot_keypoints - 2)
        )
        link_id_emb = self.link_id_embedding(link_ids).unsqueeze(0).expand(batch_size, -1, -1)
        q_links = q.unsqueeze(1).expand(-1, num_links, -1)

        link_input = torch.cat(
            [q_links, link_mid, link_vec, link_id_emb, robot_link_valid_mask.unsqueeze(-1)],
            dim=-1,
        )
        link_tokens = self.link_mlp(link_input)

        ee_input = torch.cat([q, ee_radius, safety_margin], dim=-1)
        ee_token = self.ee_mlp(ee_input).unsqueeze(1)

        robot_tokens = torch.cat([kp_tokens, link_tokens, ee_token], dim=1)

        for block in self.fusion_blocks:
            robot_tokens = block(robot_tokens, point_tokens, point_key_padding_mask)

        kp_final = robot_tokens[:, :num_keypoints, :]
        link_final = robot_tokens[:, num_keypoints : num_keypoints + num_links, :]
        ee_final = robot_tokens[:, -1, :]

        kp_raw = self.kp_head(kp_final).squeeze(-1)
        link_raw = self.link_head(link_final).squeeze(-1)
        ee_raw = self.ee_head(ee_final)

        if self.prediction_mode == "geo_only":
            kp_residual = torch.zeros_like(priors["d_keypoints_geo"])
            link_residual = torch.zeros_like(priors["d_links_geo"])
            ee_residual = torch.zeros_like(priors["h_ee_geo"])
            d_keypoints_pred = priors["d_keypoints_geo"]
            d_links_pred = priors["d_links_geo"]
            h_ee_pred = priors["h_ee_geo"]

        elif self.prediction_mode == "residual":
            kp_residual = self.residual_m_scale * torch.tanh(kp_raw)
            link_residual = self.residual_m_scale * torch.tanh(link_raw)
            ee_residual = self.residual_m_scale * torch.tanh(ee_raw)
            d_keypoints_pred = priors["d_keypoints_geo"] + kp_residual
            d_links_pred = priors["d_links_geo"] + link_residual
            h_ee_pred = priors["h_ee_geo"] + ee_residual

        elif self.prediction_mode == "neural_only":
            kp_residual = torch.zeros_like(kp_raw)
            link_residual = torch.zeros_like(link_raw)
            ee_residual = torch.zeros_like(ee_raw)
            d_keypoints_pred = 0.30 * torch.tanh(kp_raw)
            d_links_pred = 0.30 * torch.tanh(link_raw)
            h_ee_pred = 0.30 * torch.tanh(ee_raw)

        else:
            raise ValueError(f"Unknown prediction_mode: {self.prediction_mode!r}")

        d_arm_pred = masked_softargmin_value(
            d_links_pred,
            mask=robot_link_valid_mask.bool(),
            tau=self.softmin_tau,
        )
        h_arm_pred = d_arm_pred - safety_margin

        h_total_pred = masked_softargmin_value(
            torch.cat([h_ee_pred, h_arm_pred], dim=1),
            mask=torch.ones((batch_size, 2), device=pc_world.device, dtype=torch.bool),
            tau=self.softmin_tau,
        )

        v_rep_pred = F.normalize(self.rep_head(ee_final), dim=-1, eps=1e-6)

        return {
            "d_keypoints_pred": d_keypoints_pred,
            "d_links_pred": d_links_pred,
            "h_ee_pred": h_ee_pred,
            "d_arm_pred": d_arm_pred,
            "h_arm_pred": h_arm_pred,
            "h_total_pred": h_total_pred,
            "v_rep_pred": v_rep_pred,
            "d_keypoints_geo": priors["d_keypoints_geo"],
            "d_links_geo": priors["d_links_geo"],
            "h_ee_geo": priors["h_ee_geo"],
            "kp_residual": kp_residual,
            "link_residual": link_residual,
            "ee_residual": ee_residual,
        }


def move_batch_to_device(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def set_requires_grad(module: nn.Module, requires_grad: bool) -> Dict[nn.Parameter, bool]:
    previous: Dict[nn.Parameter, bool] = {}
    for param in module.parameters():
        previous[param] = bool(param.requires_grad)
        param.requires_grad_(requires_grad)
    return previous


def restore_requires_grad(previous: Dict[nn.Parameter, bool]) -> None:
    for param, requires_grad in previous.items():
        param.requires_grad_(requires_grad)


def compute_loss(
    model: LinkAwarePhysicsCBFNet,
    criterion: PhysicsArmCBFLoss,
    batch: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    need_eik = criterion.lambda_eik > 0.0

    q_norm = normalize_panda_q(batch["q"])
    q_norm = q_norm.clone().detach().requires_grad_(need_eik)

    outputs = model(
        pc_world=batch["pc_world"],
        q=q_norm,
        ee_pos=batch["ee_pos_world"],
        valid_mask=batch["valid_mask"],
        source_camera=batch["source_camera"],
        robot_keypoints_world=batch["robot_keypoints_world"],
        robot_keypoint_valid_mask=batch["robot_keypoint_valid_mask"],
        robot_link_valid_mask=batch["robot_link_valid_mask"],
        ee_radius=batch.get("ee_radius", None),
        safety_margin=batch.get("safety_margin", None),
        robot_keypoint_radius=batch.get("robot_keypoint_radius", None),
        robot_link_radius=batch.get("robot_link_radius", None),
    )

    grad_q = None
    if need_eik:
        grad_q = torch.autograd.grad(
            outputs=outputs["h_total_pred"].sum(),
            inputs=q_norm,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

    loss_dict = criterion(
        h_ee_pred=outputs["h_ee_pred"],
        d_keypoints_pred=outputs["d_keypoints_pred"],
        d_links_pred=outputs["d_links_pred"],
        h_total_pred=outputs["h_total_pred"],
        d_arm_pred=outputs["d_arm_pred"],
        h_arm_pred=outputs["h_arm_pred"],
        v_rep_pred=outputs["v_rep_pred"],
        h_star_hard=batch["h_star_hard"],
        h_star_robust=batch["h_star_robust"],
        d_gt_keypoints=batch["d_gt_keypoints"],
        d_gt_links=batch["d_gt_links"],
        robot_keypoint_valid_mask=batch["robot_keypoint_valid_mask"],
        robot_link_valid_mask=batch["robot_link_valid_mask"],
        safety_margin=batch["safety_margin"],
        v_rep=batch["v_rep"],
        v_rep_knn=batch.get("v_rep_knn", None),
        grad_q=grad_q,
    )

    outputs["grad_q"] = grad_q
    return loss_dict, outputs


def update_sums(
    sums: Dict[str, float],
    loss_dict: Dict[str, torch.Tensor],
    batch_size: int,
) -> None:
    for key, value in loss_dict.items():
        if key not in sums:
            sums[key] = 0.0
        sums[key] += float(value.detach().item()) * batch_size


def average_sums(sums: Dict[str, float], n: int) -> Dict[str, float]:
    denom = max(int(n), 1)
    return {key: value / denom for key, value in sums.items()}


def evaluate(
    model: LinkAwarePhysicsCBFNet,
    criterion: PhysicsArmCBFLoss,
    loader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    sums: Dict[str, float] = {}
    n_samples = 0

    previous = set_requires_grad(model, False)
    try:
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            with torch.enable_grad():
                loss_dict, _ = compute_loss(model, criterion, batch)

            batch_size = int(batch["pc_world"].shape[0])
            n_samples += batch_size
            update_sums(sums, loss_dict, batch_size)

    finally:
        restore_requires_grad(previous)
        model.train()

    return average_sums(sums, n_samples)


def checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    args: argparse.Namespace,
    selection_metric: str,
    selection_value: float,
) -> Dict:
    return {
        "epoch": int(epoch),

        # Keep both keys for compatibility with different loading scripts.
        "model": model.state_dict(),
        "model_state_dict": model.state_dict(),

        "optimizer": optimizer.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),

        "train_metrics": dict(train_metrics),
        "val_metrics": dict(val_metrics),

        "args": sanitized_args_dict(args),
        "config": sanitized_args_dict(args),

        "selection_metric": str(selection_metric),
        "selection_value": float(selection_value),
    }


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    args: argparse.Namespace,
    selection_metric: str,
    selection_value: float,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            args=args,
            selection_metric=selection_metric,
            selection_value=selection_value,
        ),
        path,
    )


def safety_score_from_metrics(
    val_metrics: Dict[str, float],
    *,
    fu_weight: float = 0.0,
) -> float:
    fs_ee = float(val_metrics.get("false_safe_ee_rate", 1.0))
    fs_arm = float(val_metrics.get("false_safe_arm_rate", 1.0))
    fu_ee = float(val_metrics.get("false_unsafe_ee_rate", 0.0))
    fu_arm = float(val_metrics.get("false_unsafe_arm_rate", 0.0))

    return fs_ee + fs_arm + float(fu_weight) * (fu_ee + fu_arm)


def warn_missing_val_metrics(val_metrics: Dict[str, float]) -> None:
    required_debug_keys = [
        "ee_sign_acc",
        "arm_sign_acc",
        "ee_hard_sign_acc",
        "false_safe_ee_rate",
        "false_safe_arm_rate",
        "false_unsafe_ee_rate",
        "false_unsafe_arm_rate",
    ]
    missing = [key for key in required_debug_keys if key not in val_metrics]
    if missing:
        print("WARNING missing val metrics:", missing)


def print_epoch_metrics(
    epoch: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    safety_score: float,
) -> None:
    print(
        f"Epoch {epoch}: "
        f"train_loss={train_metrics.get('loss', 0.0):.6f} "
        f"val_loss={val_metrics.get('loss', 0.0):.6f} "
        f"val_ee={val_metrics.get('L_ee', 0.0):.6f} "
        f"val_link={val_metrics.get('L_link', 0.0):.6f} "
        f"val_min={val_metrics.get('L_min', 0.0):.6f} "
        f"val_cls={val_metrics.get('L_cls', 0.0):.6f} "
        f"val_cons={val_metrics.get('L_cons', 0.0):.6f} "
        f"val_fs_loss={val_metrics.get('L_false_safe', 0.0):.6f} "
        f"val_eik={val_metrics.get('L_eik', 0.0):.6f} "
        f"val_safety={safety_score:.4f} "
        f"val_ee_acc={val_metrics.get('ee_sign_acc', 0.0):.4f} "
        f"val_arm_acc={val_metrics.get('arm_sign_acc', 0.0):.4f} "
        f"val_ee_hard_acc={val_metrics.get('ee_hard_sign_acc', 0.0):.4f} "
        f"val_fs_ee={val_metrics.get('false_safe_ee_rate', 0.0):.4f} "
        f"val_fs_arm={val_metrics.get('false_safe_arm_rate', 0.0):.4f} "
        f"val_fu_ee={val_metrics.get('false_unsafe_ee_rate', 0.0):.4f} "
        f"val_fu_arm={val_metrics.get('false_unsafe_arm_rate', 0.0):.4f}"
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )

    train_loader, val_loader = get_dataloader(
        h5_path=args.h5_path,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        split_mode=args.split_mode,
        num_workers=args.num_workers,
        seed=args.seed,
        n_points=args.n_points,
        random_point_subsample=args.random_point_subsample,
        pin_memory=device.type == "cuda",
    )

    print(
        "Dataset split:",
        f"train={len(train_loader.dataset)}",
        f"val={len(val_loader.dataset)}",
        f"split_mode={args.split_mode}",
    )

    model = LinkAwarePhysicsCBFNet(
        pointcept_root=args.pointcept_root,
        ptv3_grid_size=args.ptv3_grid_size,
        ptv3_patch_size=args.ptv3_patch_size,
        ptv3_enable_flash=args.ptv3_enable_flash,
        ptv3_order=args.ptv3_order,
        ptv3_in_channels=args.ptv3_in_channels,
        ptv3_out_dim=args.ptv3_out_dim,
        camera_embedding_dim=args.camera_embedding_dim,
        model_dim=args.model_dim,
        robot_id_embedding_dim=args.robot_id_embedding_dim,
        max_robot_keypoints=args.max_robot_keypoints,
        fusion_layers=args.fusion_layers,
        fusion_heads=args.fusion_heads,
        fusion_ffn_dim=args.fusion_ffn_dim,
        head_hidden_dim=args.head_hidden_dim,
        dropout=args.dropout,
        geo_tau=args.geo_tau,
        softmin_tau=args.softmin_tau,
        default_ee_radius=args.default_ee_radius,
        default_safety_margin=args.default_safety_margin,
        default_robot_keypoint_radius=args.default_robot_keypoint_radius,
        default_robot_link_radius=args.default_robot_link_radius,
        residual_m_scale=args.residual_m_scale,
        prediction_mode=args.prediction_mode,
    ).to(device)

    criterion = PhysicsArmCBFLoss(
        lambda_ee=args.lambda_ee,
        lambda_link=args.lambda_link,
        lambda_kp=args.lambda_kp,
        lambda_min=args.lambda_min,
        lambda_cls=args.lambda_cls,
        lambda_rep=args.lambda_rep,
        lambda_cons=args.lambda_cons,
        lambda_false_safe=args.lambda_false_safe,
        lambda_eik=args.lambda_eik,
        lambda_rank=args.lambda_rank,
        smooth_l1_beta=args.smooth_l1_beta,
        softmin_tau=args.softmin_tau,
        cls_beta=args.cls_beta,
        conservative_sigma=args.conservative_sigma,
        eik_boundary_band=args.eik_boundary_band,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

    print("total params:", total_params)
    print("trainable params:", trainable_params)
    print("config:", json.dumps(sanitized_args_dict(args), indent=2, sort_keys=True))

    experiment = setup_comet(
        args,
        total_params=total_params,
        trainable_params=trainable_params,
    )

    checkpoint_path = Path(args.checkpoint_path)
    best_safety_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_best_safety.pt")

    best_val_loss = float("inf")
    best_safety_score = float("inf")

    for epoch in range(1, int(args.epochs) + 1):
        model.train()

        train_sums: Dict[str, float] = {}
        train_n = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=True)

        for batch in pbar:
            batch = move_batch_to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)

            loss_dict, _ = compute_loss(model, criterion, batch)
            loss = loss_dict["loss"]

            if loss.requires_grad:
                loss.backward()

                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        float(args.grad_clip),
                    )

                optimizer.step()

            batch_size = int(batch["pc_world"].shape[0])
            train_n += batch_size
            update_sums(train_sums, loss_dict, batch_size)

            pbar.set_postfix(
                {
                    "loss": f"{float(loss.detach().item()):.4f}",
                    "ee": f"{float(loss_dict.get('L_ee', torch.tensor(0.0)).item()):.4f}",
                    "link": f"{float(loss_dict.get('L_link', torch.tensor(0.0)).item()):.4f}",
                    "min": f"{float(loss_dict.get('L_min', torch.tensor(0.0)).item()):.4f}",
                    "h": f"{float(loss_dict.get('mean_h_total_pred', torch.tensor(0.0)).item()):.3f}",
                }
            )

        train_metrics = average_sums(train_sums, train_n)
        log_comet_metrics(experiment, "train", train_metrics, step=epoch)

        should_eval = (
            epoch % max(1, int(args.eval_every)) == 0
            or epoch == int(args.epochs)
        )

        if should_eval:
            val_metrics = evaluate(model, criterion, val_loader, device)
            warn_missing_val_metrics(val_metrics)

            val_loss = float(val_metrics.get("loss", float("inf")))
            safety_score = safety_score_from_metrics(
                val_metrics,
                fu_weight=float(args.safety_score_fu_weight),
            )
            log_comet_metrics(experiment, "val", val_metrics, step=epoch)
            if experiment is not None:
                experiment.log_metrics(
                    {
                        "val_safety_score": float(safety_score),
                        "best_val_loss": float(best_val_loss),
                        "best_safety_score": float(best_safety_score),
                    },
                    step=epoch,
                )

            print_epoch_metrics(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                safety_score=safety_score,
            )

            # -------------------------------------------------------------
            # Best checkpoint by validation loss
            # -------------------------------------------------------------
            if val_loss < best_val_loss:
                best_val_loss = val_loss

                save_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    args=args,
                    selection_metric="val_loss",
                    selection_value=best_val_loss,
                )

                print(
                    f"Saved best val-loss checkpoint to {checkpoint_path} "
                    f"val_loss={best_val_loss:.6f}"
                )
                if experiment is not None:
                    experiment.log_metrics(
                        {
                            "best_val_loss": float(best_val_loss),
                            "best_val_loss_epoch": float(epoch),
                        },
                        step=epoch,
                    )

            # -------------------------------------------------------------
            # Best checkpoint by safety score on validation loader
            # -------------------------------------------------------------
            if safety_score < best_safety_score:
                best_safety_score = safety_score

                save_checkpoint(
                    best_safety_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    args=args,
                    selection_metric=(
                        "false_safe_ee + false_safe_arm"
                        if float(args.safety_score_fu_weight) == 0.0
                        else "false_safe_ee + false_safe_arm + fu_weight*(false_unsafe_ee + false_unsafe_arm)"
                    ),
                    selection_value=best_safety_score,
                )

                print(
                    f"Saved best safety checkpoint to {best_safety_path} "
                    f"safety_score={best_safety_score:.4f} "
                    f"fs_ee={val_metrics.get('false_safe_ee_rate', 0.0):.4f} "
                    f"fs_arm={val_metrics.get('false_safe_arm_rate', 0.0):.4f} "
                    f"fu_ee={val_metrics.get('false_unsafe_ee_rate', 0.0):.4f} "
                    f"fu_arm={val_metrics.get('false_unsafe_arm_rate', 0.0):.4f}"
                )
                if experiment is not None:
                    experiment.log_metrics(
                        {
                            "best_safety_score": float(best_safety_score),
                            "best_safety_epoch": float(epoch),
                        },
                        step=epoch,
                    )

        else:
            val_metrics = {}
            safety_score = float("inf")
            print(
                f"Epoch {epoch}: "
                f"train_loss={train_metrics.get('loss', 0.0):.6f}"
            )

        # -----------------------------------------------------------------
        # Periodic checkpoint
        # -----------------------------------------------------------------
        if args.save_every > 0 and epoch % int(args.save_every) == 0:
            periodic_path = checkpoint_path.with_name(
                f"{checkpoint_path.stem}_epoch_{epoch:04d}.pt"
            )

            save_checkpoint(
                periodic_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                args=args,
                selection_metric="periodic",
                selection_value=float(epoch),
            )

            print(f"Saved periodic checkpoint to {periodic_path}")

    if experiment is not None:
        experiment.end()


if __name__ == "__main__":
    main()
