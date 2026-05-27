from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_column(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return x.unsqueeze(1)
    if x.ndim > 2:
        return x.reshape(x.shape[0], -1)[:, :1]
    return x


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float) -> torch.Tensor:
    mask = mask.to(dtype=values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    denom = mask.sum().clamp_min(eps)
    return (values * mask).sum() / denom


class PhysicsInformedCBFLoss(nn.Module):
    """Loss for a residual, physics-informed CBF clearance model.

    The model predicts a metric signed clearance h_pred_m. We train the
    normalized value h_pred_m / h_scale against robust and hard dataset targets,
    add a hard unsafe sign classifier, and regularize the workspace gradient
    near the hard boundary.
    """

    def __init__(
        self,
        lambda_sign: float = 0.15,
        lambda_dir: float = 0.0,
        lambda_eik: float = 0.0,
        robust_weight: float = 0.70,
        hard_weight: float = 0.30,
        tau_cls: float = 0.20,
        boundary_band: float = 0.04,
        boundary_weight_alpha: float = 4.0,
        boundary_weight_scale: float = 0.03,
        direction_cos_margin: float = 0.50,
        smooth_l1_beta: float = 0.10,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.lambda_sign = float(lambda_sign)
        self.lambda_dir = float(lambda_dir)
        self.lambda_eik = float(lambda_eik)
        self.robust_weight = float(robust_weight)
        self.hard_weight = float(hard_weight)
        self.tau_cls = float(tau_cls)
        self.boundary_band = float(boundary_band)
        self.boundary_weight_alpha = float(boundary_weight_alpha)
        self.boundary_weight_scale = float(boundary_weight_scale)
        self.direction_cos_margin = float(direction_cos_margin)
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.eps = float(eps)

    def forward(
        self,
        *,
        h_pred_m: torch.Tensor,
        h_pred_norm: Optional[torch.Tensor] = None,
        h_star_robust_m: torch.Tensor,
        h_star_hard_m: torch.Tensor,
        h_scale: torch.Tensor,
        h_star_robust_norm: Optional[torch.Tensor] = None,
        h_star_hard_norm: Optional[torch.Tensor] = None,
        grad_x: Optional[torch.Tensor] = None,
        v_rep: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        h_pred_m = _as_column(h_pred_m)
        h_star_robust_m = _as_column(h_star_robust_m).to(dtype=h_pred_m.dtype)
        h_star_hard_m = _as_column(h_star_hard_m).to(dtype=h_pred_m.dtype)
        h_scale = _as_column(h_scale).to(dtype=h_pred_m.dtype).clamp_min(self.eps)

        if h_pred_norm is None:
            h_pred_norm = h_pred_m / h_scale
        else:
            h_pred_norm = _as_column(h_pred_norm).to(dtype=h_pred_m.dtype)
        y_robust = (
            _as_column(h_star_robust_norm).to(dtype=h_pred_m.dtype)
            if h_star_robust_norm is not None
            else h_star_robust_m / h_scale
        )
        y_hard = (
            _as_column(h_star_hard_norm).to(dtype=h_pred_m.dtype)
            if h_star_hard_norm is not None
            else h_star_hard_m / h_scale
        )

        robust_raw = F.smooth_l1_loss(
            h_pred_norm,
            y_robust,
            beta=self.smooth_l1_beta,
            reduction="none",
        )
        hard_raw = F.smooth_l1_loss(
            h_pred_norm,
            y_hard,
            beta=self.smooth_l1_beta,
            reduction="none",
        )
        boundary_weight = 1.0 + self.boundary_weight_alpha * torch.exp(
            -h_star_hard_m.abs() / max(self.boundary_weight_scale, self.eps)
        )
        l_robust = torch.mean(boundary_weight * robust_raw)
        l_hard = torch.mean(boundary_weight * hard_raw)
        l_reg = self.robust_weight * l_robust + self.hard_weight * l_hard

        unsafe_hard = (h_star_hard_m < 0.0).to(dtype=h_pred_m.dtype)
        sign_logits = -h_pred_norm / max(self.tau_cls, self.eps)
        l_sign = F.binary_cross_entropy_with_logits(sign_logits, unsafe_hard)

        boundary = (h_star_hard_m.abs() < self.boundary_band).to(dtype=h_pred_m.dtype)
        l_eik = h_pred_m.new_tensor(0.0)
        l_dir = h_pred_m.new_tensor(0.0)
        grad_norm = h_pred_m.new_zeros((h_pred_m.shape[0], 1))
        mean_cos = h_pred_m.new_tensor(0.0)

        if grad_x is not None:
            grad_norm = torch.linalg.norm(grad_x, dim=-1, keepdim=True).clamp_min(self.eps)
            eik_raw = (grad_norm - 1.0).pow(2)
            l_eik = _masked_mean(eik_raw, boundary, self.eps)

            if v_rep is not None:
                v_rep = v_rep.to(dtype=grad_x.dtype)
                grad_unit = grad_x / grad_norm
                v_norm = torch.linalg.norm(v_rep, dim=-1, keepdim=True).clamp_min(self.eps)
                v_unit = v_rep / v_norm
                cos = (grad_unit * v_unit).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
                dir_raw = F.relu(self.direction_cos_margin - cos).pow(2)
                l_dir = _masked_mean(dir_raw, boundary, self.eps)
                mean_cos = _masked_mean(cos, boundary, self.eps).detach()

        loss = (
            l_reg
            + self.lambda_sign * l_sign
            + self.lambda_dir * l_dir
            + self.lambda_eik * l_eik
        )

        with torch.no_grad():
            pred_unsafe = (h_pred_m < 0.0).to(dtype=h_pred_m.dtype)
            sign_acc = (pred_unsafe == unsafe_hard).to(dtype=h_pred_m.dtype).mean()

        return {
            "loss": loss,
            "L_reg": l_reg.detach(),
            "L_robust": l_robust.detach(),
            "L_hard": l_hard.detach(),
            "L_sign": l_sign.detach(),
            "L_dir": l_dir.detach(),
            "L_eik": l_eik.detach(),
            "mean_h_pred_m": h_pred_m.detach().mean(),
            "mean_h_star_robust_m": h_star_robust_m.detach().mean(),
            "mean_h_star_hard_m": h_star_hard_m.detach().mean(),
            "mean_grad_norm": grad_norm.detach().mean(),
            "boundary_fraction": boundary.detach().mean(),
            "boundary_cos": mean_cos,
            "sign_acc": sign_acc.detach(),
        }


# Backward-compatible short name for training scripts.
Loss = PhysicsInformedCBFLoss
