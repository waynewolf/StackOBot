import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

"""
所有函数，图像的原点在左上角
"""

# F.grid_sample 严格来说只支持 backward_flow, backward_flow 指的是，从目标指向源
def backward_warp(src_color: torch.Tensor, backward_flow: torch.Tensor) -> torch.Tensor:
    color = src_color.unsqueeze(0)
    flow = backward_flow.unsqueeze(0)
    _, _, h, w = color.shape

    # 目标网格坐标
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=color.device),
        torch.linspace(-1.0, 1.0, w, device=color.device),
        indexing="ij",
    )
    grid = torch.stack([xx, yy], dim=-1).unsqueeze(0)
    # 目标 + backward_flow（目标指向源），得到源网格，是反向的正确调用方式
    grid = grid + flow.permute(0, 2, 3, 1)
    warped = F.grid_sample(color, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return warped.squeeze(0)


def forward_warp(src_color: torch.Tensor, forward_flow: torch.Tensor) -> torch.Tensor:
    """Forward splat that accepts (C,H,W) or (N,C,H,W) tensors."""

    def _ensure_batched(tensor: torch.Tensor, expected_channels: Optional[int] = None) -> Tuple[torch.Tensor, bool]:
        if tensor.dim() == 3:
            batched = tensor.unsqueeze(0)
            added = True
        elif tensor.dim() == 4:
            batched = tensor
            added = False
        else:
            raise ValueError("Expected tensor with 3 or 4 dims")
        if expected_channels is not None and batched.shape[1] != expected_channels:
            raise ValueError(f"Expected channel dimension == {expected_channels}")
        return batched, added

    color, color_was_3d = _ensure_batched(src_color)
    flow, _ = _ensure_batched(forward_flow, expected_channels=2)

    if color.shape[0] != flow.shape[0]:
        raise ValueError("Color and flow must have matching batch size")

    n, c, h, w = color.shape
    device = color.device
    dtype = color.dtype

    yy, xx = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    xx = xx.to(dtype=dtype)
    yy = yy.to(dtype=dtype)

    tx = xx + flow[:, 0] * (0.5 * (w - 1))
    ty = yy + flow[:, 1] * (0.5 * (h - 1))

    x0 = torch.floor(tx)
    y0 = torch.floor(ty)
    x1 = x0 + 1.0
    y1 = y0 + 1.0
    dx = tx - x0
    dy = ty - y0

    w00 = (1.0 - dx) * (1.0 - dy)
    w01 = (1.0 - dx) * dy
    w10 = dx * (1.0 - dy)
    w11 = dx * dy

    x0i = x0.long()
    y0i = y0.long()
    x1i = x1.long()
    y1i = y1.long()

    m00 = (x0i >= 0) & (x0i < w) & (y0i >= 0) & (y0i < h)
    m01 = (x0i >= 0) & (x0i < w) & (y1i >= 0) & (y1i < h)
    m10 = (x1i >= 0) & (x1i < w) & (y0i >= 0) & (y0i < h)
    m11 = (x1i >= 0) & (x1i < w) & (y1i >= 0) & (y1i < h)

    out_flat = torch.zeros((n, c, h * w), device=device, dtype=dtype)
    color_flat = color.reshape(n, c, -1)

    def _splat(mask, xi, yi, weight):
        mask_flat = mask.reshape(n, -1)
        if not mask_flat.any():
            return
        idx_flat = (yi.reshape(n, -1) * w + xi.reshape(n, -1))
        weight_flat = weight.reshape(n, -1)
        for b in range(n):
            valid = mask_flat[b]
            if not valid.any():
                continue
            dest_idx = idx_flat[b, valid].long()
            wv = weight_flat[b, valid]
            src = color_flat[b, :, valid] * wv.unsqueeze(0)
            out_flat[b].scatter_add_(1, dest_idx.unsqueeze(0).expand(c, -1), src)

    _splat(m00, x0i, y0i, w00)
    _splat(m01, x0i, y1i, w01)
    _splat(m10, x1i, y0i, w10)
    _splat(m11, x1i, y1i, w11)

    out = out_flat.reshape(n, c, h, w)
    if color_was_3d:
        out = out.squeeze(0)
    return out


def main():
    def _self_test():
        torch.manual_seed(0)
        c = torch.rand(3, 12, 17)
        f = torch.zeros(2, 12, 17)
        out_fw = forward_warp(c, f)
        max_fw = (out_fw - c).abs().max().item()
        print(f"self_test forward_warp_vectorize max_diff: {max_fw:.6g}")
        if not torch.allclose(out_fw, c, rtol=1e-4, atol=1e-5):
            raise AssertionError("forward_warp_vectorize zero-flow mismatch")
    _self_test()


if __name__ == "__main__":
    main()
