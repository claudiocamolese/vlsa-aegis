#!/usr/bin/env python3
from __future__ import annotations

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
from tqdm import tqdm

from dataloader import get_dataloader
from loss import PhysicsInformedCBFLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a physics-informed residual CBF model on SafeLIBERO H5 datasets."
    )
    parser.add_argument("--h5-path", required=True)
    parser.add_argument("--checkpoint-path", default="vlsa-aegis_runtime/checkpoints/cbf_ptv3.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
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

    parser.add_argument(
        "--pointcept-root",
        default=str(Path.home() / "Claudio" / "vlsa-aegis" / "Pointcept"),
    )
    parser.add_argument("--ptv3-grid-size", type=float, default=0.02)
    parser.add_argument("--ptv3-patch-size", type=int, default=128)
    parser.add_argument("--ptv3-enable-flash", action="store_true", default= True)
    parser.add_argument("--ptv3-order", nargs="+", default=["z"])
    parser.add_argument("--camera-embedding-dim", type=int, default=8)
    parser.add_argument("--ptv3-in-channels", type=int, default=12)
    parser.add_argument("--ptv3-out-dim", type=int, default=512)
    parser.add_argument("--cross-dim", type=int, default=256)
    parser.add_argument("--num-query-tokens", type=int, default=4)
    parser.add_argument("--cross-attn-layers", type=int, default=2)
    parser.add_argument("--cross-attn-heads", type=int, default=8)
    parser.add_argument("--cross-ffn-dim", type=int, default=1024)
    parser.add_argument("--head-hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.05)

    parser.add_argument("--geo-tau", type=float, default=0.01)
    parser.add_argument("--default-ee-radius", type=float, default=0.04)
    parser.add_argument("--default-safety-margin", type=float, default=0.02)
    parser.add_argument(
        "--residual-norm-scale",
        type=float,
        default=0.5,
        help="Maximum normalized residual: delta_norm = scale * tanh(raw).",
    )
    parser.add_argument(
        "--grad-source",
        choices=["geo", "mixed", "full"],
        default="mixed",
        help=(
            "Signal used for Eikonal/direction gradients. mixed trains the residual "
            "with a small beta; geo is the stable fallback; full uses h_pred."
        ),
    )
    parser.add_argument("--delta-grad-beta", type=float, default=0.20)

    parser.add_argument("--lambda-sign", type=float, default=0.15)
    parser.add_argument("--lambda-dir", type=float, default=0.0)
    parser.add_argument("--lambda-eik", type=float, default=0.0)
    parser.add_argument("--boundary-band", type=float, default=0.04)
    parser.add_argument("--tau-cls", type=float, default=0.20)
    parser.add_argument("--smooth-l1-beta", type=float, default=0.10)
    parser.add_argument("--boundary-weight-alpha", type=float, default=4.0)
    parser.add_argument("--boundary-weight-scale", type=float, default=0.03)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


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


def masked_softmin_distance(
    distances: torch.Tensor,
    valid_mask: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    valid = valid_mask.bool()
    tau = max(float(tau), 1e-6)
    logits = -distances / tau
    logits = torch.where(valid, logits, torch.full_like(logits, -1e9))

    weights = torch.softmax(logits, dim=1)
    weights = torch.where(valid, weights, torch.zeros_like(weights))
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return torch.sum(weights * distances, dim=1)


def batch_from_offset(offset: torch.Tensor, total_points: int) -> torch.Tensor:
    counts = torch.diff(
        torch.cat(
            [
                torch.zeros((1,), device=offset.device, dtype=offset.dtype),
                offset.to(dtype=offset.dtype),
            ]
        )
    )
    return torch.arange(counts.shape[0], device=offset.device).repeat_interleave(counts.long())[:total_points]


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


def masked_min_and_knn_mean(
    values: torch.Tensor,
    valid_mask: torch.Tensor,
    k: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    large = torch.full_like(values, 1e6)
    masked = torch.where(valid_mask.bool(), values, large)
    h_min = torch.min(masked, dim=1).values

    k_eff = min(max(1, int(k)), int(values.shape[1]))
    top_values = torch.topk(masked, k=k_eff, dim=1, largest=False).values
    top_valid = top_values < 1e5
    top_sum = torch.where(top_valid, top_values, torch.zeros_like(top_values)).sum(dim=1)
    top_count = top_valid.to(dtype=values.dtype).sum(dim=1).clamp_min(1.0)
    h_knn = top_sum / top_count
    return h_min, h_knn


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_out, _ = self.attn(
            query=self.query_norm(query),
            key=self.context_norm(context),
            value=self.context_norm(context),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        query = query + attn_out
        query = query + self.ffn(self.ffn_norm(query))
        return query


class PhysicsResidualCBFNet(nn.Module):
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
        cross_dim: int,
        num_query_tokens: int,
        cross_attn_layers: int,
        cross_attn_heads: int,
        cross_ffn_dim: int,
        head_hidden_dim: int,
        dropout: float,
        geo_tau: float,
        default_ee_radius: float,
        default_safety_margin: float,
        residual_norm_scale: float,
        grad_source: str,
        delta_grad_beta: float,
    ) -> None:
        super().__init__()
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
            nn.Linear(int(ptv3_out_dim), int(cross_dim)),
            nn.LayerNorm(int(cross_dim)),
            nn.SiLU(),
        )
        self.query_tokens = nn.Parameter(torch.randn(int(num_query_tokens), int(cross_dim)) * 0.02)
        self.ee_context = nn.Sequential(
            nn.Linear(5, int(cross_dim)),
            nn.SiLU(),
            nn.Linear(int(cross_dim), int(cross_dim)),
        )
        self.cross_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    dim=int(cross_dim),
                    heads=int(cross_attn_heads),
                    ffn_dim=int(cross_ffn_dim),
                    dropout=float(dropout),
                )
                for _ in range(int(cross_attn_layers))
            ]
        )
        self.ptv3_grid_size = float(ptv3_grid_size)
        self.geo_tau = float(geo_tau)
        self.default_ee_radius = float(default_ee_radius)
        self.default_safety_margin = float(default_safety_margin)
        self.residual_norm_scale = float(residual_norm_scale)
        self.grad_source = str(grad_source)
        self.delta_grad_beta = float(delta_grad_beta)

        head_in = int(num_query_tokens) * int(cross_dim) + 4
        self.head = nn.Sequential(
            nn.Linear(head_in, int(head_hidden_dim)),
            nn.LayerNorm(int(head_hidden_dim)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(head_hidden_dim), 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

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
                torch.full((rel.shape[0],), batch_idx, device=pc_world.device, dtype=torch.long)
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

    def forward(
        self,
        *,
        pc_world: torch.Tensor,
        ee_pos: torch.Tensor,
        valid_mask: torch.Tensor,
        source_camera: torch.Tensor,
        h_scale: torch.Tensor | None = None,
        ee_radius: torch.Tensor | None = None,
        safety_margin: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = int(pc_world.shape[0])
        valid_mask = valid_mask.to(dtype=pc_world.dtype)
        source_camera = source_camera.to(device=pc_world.device)

        if ee_radius is None:
            ee_radius = pc_world.new_full((batch_size, 1), self.default_ee_radius)
        if safety_margin is None:
            safety_margin = pc_world.new_full((batch_size, 1), self.default_safety_margin)
        ee_radius = ee_radius.reshape(batch_size, 1).to(dtype=pc_world.dtype)
        safety_margin = safety_margin.reshape(batch_size, 1).to(dtype=pc_world.dtype)
        if h_scale is None:
            h_scale = pc_world.new_full((batch_size, 1), 0.10)
        h_scale = h_scale.reshape(batch_size, 1).to(dtype=pc_world.dtype).clamp_min(1e-6)

        distances = torch.linalg.norm(pc_world - ee_pos.unsqueeze(1), dim=-1)
        signed_clearance = distances - ee_radius - safety_margin
        h_geo_m = masked_softmin_distance(
            signed_clearance,
            valid_mask=valid_mask,
            tau=self.geo_tau,
        ).reshape(batch_size, 1)
        y_geo = h_geo_m / h_scale
        h_min_m, h_knn8_m = masked_min_and_knn_mean(
            signed_clearance,
            valid_mask=valid_mask,
            k=8,
        )
        h_min_norm = (h_min_m.reshape(batch_size, 1) / h_scale).clamp(-10.0, 10.0)
        h_knn8_norm = (h_knn8_m.reshape(batch_size, 1) / h_scale).clamp(-10.0, 10.0)
        valid_ratio = valid_mask.to(dtype=pc_world.dtype).mean(dim=1, keepdim=True)

        point_dict = self._make_point_dict(
            pc_world=pc_world,
            ee_pos=ee_pos,
            valid_mask=valid_mask,
            source_camera=source_camera,
        )
        obs_tokens, key_padding_mask = self._ptv3_tokens(point_dict, batch_size=batch_size)
        context_input = torch.cat(
            [
                torch.zeros((batch_size, 3), device=pc_world.device, dtype=pc_world.dtype),
                ee_radius,
                safety_margin,
            ],
            dim=-1,
        )
        query = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        query = query + self.ee_context(context_input).unsqueeze(1)
        for block in self.cross_blocks:
            query = block(query, obs_tokens, key_padding_mask)

        query_flat = query.reshape(batch_size, -1)
        head_in = torch.cat([query_flat, y_geo, h_min_norm, h_knn8_norm, valid_ratio], dim=-1)
        delta_raw = self.head(head_in)
        delta_norm = self.residual_norm_scale * torch.tanh(delta_raw)
        delta_h_m = delta_norm * h_scale
        h_pred_norm = y_geo + delta_norm
        h_pred_m = h_geo_m + delta_h_m

        if self.grad_source == "geo":
            h_for_grad_m = h_geo_m
        elif self.grad_source == "mixed":
            h_for_grad_m = h_geo_m + self.delta_grad_beta * delta_h_m
        elif self.grad_source == "full":
            h_for_grad_m = h_pred_m
        else:
            raise ValueError(f"Unknown grad_source: {self.grad_source}")

        return {
            "h_pred_m": h_pred_m,
            "h_pred_norm": h_pred_norm,
            "h_geo_m": h_geo_m,
            "y_geo": y_geo,
            "h_min_norm": h_min_norm,
            "h_knn8_norm": h_knn8_norm,
            "valid_ratio": valid_ratio,
            "delta_h_m": delta_h_m,
            "delta_norm": delta_norm,
            "h_for_grad_m": h_for_grad_m,
        }


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
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
    model: PhysicsResidualCBFNet,
    criterion: PhysicsInformedCBFLoss,
    batch: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    pc_world = batch["pc_world"]
    need_physics_grad = criterion.lambda_dir > 0.0 or criterion.lambda_eik > 0.0
    ee_pos = batch["ee_pos_world"].clone().detach().requires_grad_(need_physics_grad)
    valid_mask = batch["valid_mask"]
    source_camera = batch["source_camera"]

    outputs = model(
        pc_world=pc_world,
        ee_pos=ee_pos,
        valid_mask=valid_mask,
        source_camera=source_camera,
        h_scale=batch.get("h_scale", None),
        ee_radius=batch.get("ee_radius", None),
        safety_margin=batch.get("safety_margin", None),
    )
    grad_x = None
    if need_physics_grad:
        grad_x = torch.autograd.grad(
            outputs=outputs["h_for_grad_m"].sum(),
            inputs=ee_pos,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
    loss_dict = criterion(
        h_pred_m=outputs["h_pred_m"],
        h_pred_norm=outputs["h_pred_norm"],
        h_star_robust_m=batch["h_star_robust"],
        h_star_hard_m=batch["h_star_hard"],
        h_scale=batch["h_scale"],
        h_star_robust_norm=batch.get("h_star_norm", batch.get("h_star_robust_norm", None)),
        h_star_hard_norm=batch.get("h_star_hard_norm", None),
        grad_x=grad_x,
        v_rep=batch["v_rep"],
    )
    outputs["grad_x"] = grad_x
    return loss_dict, outputs


def update_sums(sums: Dict[str, float], loss_dict: Dict[str, torch.Tensor], batch_size: int) -> None:
    for key, value in loss_dict.items():
        if key not in sums:
            sums[key] = 0.0
        sums[key] += float(value.detach().item()) * batch_size


def average_sums(sums: Dict[str, float], n: int) -> Dict[str, float]:
    denom = max(int(n), 1)
    return {key: value / denom for key, value in sums.items()}


def evaluate(
    model: PhysicsResidualCBFNet,
    criterion: PhysicsInformedCBFLoss,
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


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    args: argparse.Namespace,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_metrics": dict(train_metrics),
            "val_metrics": dict(val_metrics),
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

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

    model = PhysicsResidualCBFNet(
        pointcept_root=args.pointcept_root,
        ptv3_grid_size=args.ptv3_grid_size,
        ptv3_patch_size=args.ptv3_patch_size,
        ptv3_enable_flash=args.ptv3_enable_flash,
        ptv3_order=args.ptv3_order,
        ptv3_in_channels=args.ptv3_in_channels,
        ptv3_out_dim=args.ptv3_out_dim,
        camera_embedding_dim=args.camera_embedding_dim,
        cross_dim=args.cross_dim,
        num_query_tokens=args.num_query_tokens,
        cross_attn_layers=args.cross_attn_layers,
        cross_attn_heads=args.cross_attn_heads,
        cross_ffn_dim=args.cross_ffn_dim,
        head_hidden_dim=args.head_hidden_dim,
        dropout=args.dropout,
        geo_tau=args.geo_tau,
        default_ee_radius=args.default_ee_radius,
        default_safety_margin=args.default_safety_margin,
        residual_norm_scale=args.residual_norm_scale,
        grad_source=args.grad_source,
        delta_grad_beta=args.delta_grad_beta,
    ).to(device)

    criterion = PhysicsInformedCBFLoss(
        lambda_sign=args.lambda_sign,
        lambda_dir=args.lambda_dir,
        lambda_eik=args.lambda_eik,
        tau_cls=args.tau_cls,
        boundary_band=args.boundary_band,
        boundary_weight_alpha=args.boundary_weight_alpha,
        boundary_weight_scale=args.boundary_weight_scale,
        smooth_l1_beta=args.smooth_l1_beta,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("total params:", total_params)
    print("trainable params:", trainable_params)
    print("config:", json.dumps(vars(args), indent=2, sort_keys=True))

    best_val = float("inf")
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_sums: Dict[str, float] = {}
        train_n = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=True)
        for batch in pbar:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss_dict, outputs = compute_loss(model, criterion, batch)
            loss = loss_dict["loss"]
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()

            batch_size = int(batch["pc_world"].shape[0])
            train_n += batch_size
            update_sums(train_sums, loss_dict, batch_size)
            pbar.set_postfix(
                {
                    "loss": f"{float(loss.detach().item()):.4f}",
                    "L_reg": f"{float(loss_dict['L_reg'].item()):.4f}",
                    "h": f"{float(loss_dict['mean_h_pred_m'].item()):.3f}",
                    "|g|": f"{float(loss_dict['mean_grad_norm'].item()):.3f}",
                }
            )

        train_metrics = average_sums(train_sums, train_n)
        should_eval = epoch % max(1, int(args.eval_every)) == 0 or epoch == int(args.epochs)
        if should_eval:
            val_metrics = evaluate(model, criterion, val_loader, device)
            val_loss = float(val_metrics.get("loss", float("inf")))
            print(
                f"Epoch {epoch}: "
                f"train_loss={train_metrics.get('loss', 0.0):.6f} "
                f"val_loss={val_loss:.6f} "
                f"val_reg={val_metrics.get('L_reg', 0.0):.6f} "
                f"val_sign={val_metrics.get('L_sign', 0.0):.6f} "
                f"val_dir={val_metrics.get('L_dir', 0.0):.6f} "
                f"val_eik={val_metrics.get('L_eik', 0.0):.6f}"
            )
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    args.checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    args=args,
                )
                print(f"Saved best checkpoint to {args.checkpoint_path}")
        else:
            val_metrics = {}
            print(f"Epoch {epoch}: train_loss={train_metrics.get('loss', 0.0):.6f}")

        if args.save_every > 0 and epoch % int(args.save_every) == 0:
            periodic_path = Path(args.checkpoint_path).with_name(
                f"{Path(args.checkpoint_path).stem}_epoch_{epoch:04d}.pt"
            )
            save_checkpoint(
                periodic_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                args=args,
            )


if __name__ == "__main__":
    main()
