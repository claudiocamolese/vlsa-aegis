#!/usr/bin/env python3

import os

os.environ.setdefault("CUMM_CUDA_ARCH_LIST", "9.0+PTX")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0+PTX")

import torch
import cumm
import spconv
import spconv.pytorch as spconv_pytorch


def main():
    print("torch:", torch.__version__)
    print("cuda:", torch.version.cuda)
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    print("cumm file:", cumm.__file__)
    print("cumm PACKAGE_ROOT:", cumm.PACKAGE_ROOT)
    print("spconv file:", spconv.__file__)
    print("CUMM_CUDA_ARCH_LIST:", os.environ.get("CUMM_CUDA_ARCH_LIST"))
    print("TORCH_CUDA_ARCH_LIST:", os.environ.get("TORCH_CUDA_ARCH_LIST"))

    device = "cuda"

    indices = torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 1],
            [1, 0, 1, 0],
            [1, 1, 0, 0],
        ],
        dtype=torch.int32,
        device=device,
    )

    features = torch.randn(
        indices.shape[0],
        32,
        device=device,
        dtype=torch.float32,
    )

    x = spconv_pytorch.SparseConvTensor(
        features=features,
        indices=indices,
        spatial_shape=[8, 8, 8],
        batch_size=2,
    )

    conv = spconv_pytorch.SubMConv3d(
        32,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        indice_key="subm1",
    ).to(device)

    print("Running spconv forward...")
    y = conv(x)
    torch.cuda.synchronize()

    print("Forward OK")
    print("y.features:", y.features.shape, y.features.dtype)


if __name__ == "__main__":
    main()
