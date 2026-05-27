#!/usr/bin/env python3

import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("CUMM_CUDA_ARCH_LIST", "9.0+PTX")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0+PTX")

POINTCEPT_ROOT = Path.home() / "Claudio" / "vlsa-aegis" / "Pointcept"
sys.path.insert(0, str(POINTCEPT_ROOT))

import torch

import cumm
import spconv
import spconv.pytorch as spconv_pytorch

from pointcept.models import MODELS


def make_point_dict(
    batch_size: int = 2,
    points_per_batch: int = 1024,
    in_channels: int = 6,
    grid_size: float = 0.02,
    coord_scale: float = 1.0,
    device: str = "cuda",
):
    total_points = batch_size * points_per_batch

    batch = torch.arange(batch_size, device=device).repeat_interleave(points_per_batch)

    coord = torch.rand(total_points, 3, device=device, dtype=torch.float32)
    coord = (coord - 0.5) * coord_scale

    feat = torch.randn(total_points, in_channels, device=device, dtype=torch.float32)

    grid_coord = torch.floor((coord - coord.min(dim=0).values) / grid_size).to(torch.int32)

    offset = torch.arange(
        points_per_batch,
        total_points + 1,
        points_per_batch,
        device=device,
        dtype=torch.int32,
    )

    return {
        "coord": coord,
        "grid_coord": grid_coord,
        "feat": feat,
        "batch": batch,
        "offset": offset,
    }


def build_ptv3(
    enable_flash: bool,
    order,
    patch_size: int,
    in_channels: int = 6,
):
    model_cfg = dict(
        type="PT-v3m1",
        in_channels=in_channels,
        order=order,
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
        enable_flash=enable_flash,
        enc_mode=True,
        upcast_attention=not enable_flash,
        upcast_softmax=not enable_flash,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    )

    return MODELS.build(model_cfg).cuda().eval()


def run_case(name, enable_flash, order, patch_size, points_per_batch):
    print("=" * 80)
    print("CASE:", name)
    print("enable_flash:", enable_flash)
    print("order:", order)
    print("patch_size:", patch_size)
    print("points_per_batch:", points_per_batch)

    torch.cuda.empty_cache()

    point_dict = make_point_dict(
        batch_size=2,
        points_per_batch=points_per_batch,
        in_channels=6,
        grid_size=0.02,
        coord_scale=1.0,
        device="cuda",
    )

    print("Input:")
    print(" coord:", point_dict["coord"].shape, point_dict["coord"].dtype)
    print(" grid_coord:", point_dict["grid_coord"].shape, point_dict["grid_coord"].dtype)
    print(" feat:", point_dict["feat"].shape, point_dict["feat"].dtype)
    print(" batch:", point_dict["batch"].shape, point_dict["batch"].dtype)
    print(" offset:", point_dict["offset"], point_dict["offset"].dtype)

    model = build_ptv3(
        enable_flash=enable_flash,
        order=order,
        patch_size=patch_size,
        in_channels=6,
    )

    print("Model built")
    print("Running forward...")

    with torch.no_grad():
        out = model(point_dict)

    torch.cuda.synchronize()

    print("Forward OK")
    print("Output type:", type(out))

    if isinstance(out, dict):
        print("Output keys:", list(out.keys()))
        if "feat" in out:
            print("out['feat']:", out["feat"].shape, out["feat"].dtype)
    elif hasattr(out, "feat"):
        print("out.feat:", out.feat.shape, out.feat.dtype)
    else:
        print("Output:", out)

    del model
    del point_dict
    del out
    torch.cuda.empty_cache()


def main():
    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    print("cumm:", cumm.__file__)
    print("cumm PACKAGE_ROOT:", cumm.PACKAGE_ROOT)
    print("spconv:", spconv.__file__)
    print("spconv.pytorch:", spconv_pytorch.__file__)
    print("POINTCEPT_ROOT:", POINTCEPT_ROOT)

    cases = [
        {
            "name": "z_only_no_flash_small",
            "enable_flash": False,
            "order": ("z",),
            "patch_size": 64,
            "points_per_batch": 512,
        },
        {
            "name": "z_only_no_flash_normal",
            "enable_flash": False,
            "order": ("z",),
            "patch_size": 128,
            "points_per_batch": 1024,
        },
        {
            "name": "official_orders_no_flash",
            "enable_flash": False,
            "order": ("z", "z-trans", "hilbert", "hilbert-trans"),
            "patch_size": 128,
            "points_per_batch": 1024,
        },
        {
            "name": "z_only_flash",
            "enable_flash": True,
            "order": ("z",),
            "patch_size": 128,
            "points_per_batch": 1024,
        },
    ]

    for case in cases:
        try:
            run_case(**case)
        except BaseException as exc:
            print("FAILED CASE:", case["name"])
            print("Exception:", repr(exc))
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
