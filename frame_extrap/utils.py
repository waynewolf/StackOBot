import math
import torch
import torch.nn.functional as F

"""
所有函数，图像的原点在左上角
"""

# F.grid_sample 严格来说只支持 backward_flow, 所以这是标准用法，
# backward_flow 指的是，从目标指向源
def backward_warp_gridsample(src_color: torch.Tensor, backward_flow: torch.Tensor) -> torch.Tensor:
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


def backward_warp_naive(src_color: torch.Tensor, backward_flow: torch.Tensor) -> torch.Tensor:
    color = src_color
    flow = backward_flow
    _, h, w = color.shape

    # 目标网格坐标
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=color.device),
        torch.linspace(-1.0, 1.0, w, device=color.device),
        indexing="ij",
    )

    # 目标 + backward_flow（目标指向源），得到源网格，是反向的正确调用方式
    x = xx + flow[0]
    y = yy + flow[1]

    # align_corners=True mapping from [-1, 1] to pixel coordinates
    x = (x + 1.0) * 0.5 * (w - 1)
    y = (y + 1.0) * 0.5 * (h - 1)

    x0 = torch.floor(x)
    y0 = torch.floor(y)
    x1 = x0 + 1.0
    y1 = y0 + 1.0

    wx1 = x - x0
    wy1 = y - y0
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1

    x0i = x0.clamp(0, w - 1).long()
    x1i = x1.clamp(0, w - 1).long()
    y0i = y0.clamp(0, h - 1).long()
    y1i = y1.clamp(0, h - 1).long()

    v00 = (x0 >= 0) & (x0 <= w - 1) & (y0 >= 0) & (y0 <= h - 1)
    v01 = (x0 >= 0) & (x0 <= w - 1) & (y1 >= 0) & (y1 <= h - 1)
    v10 = (x1 >= 0) & (x1 <= w - 1) & (y0 >= 0) & (y0 <= h - 1)
    v11 = (x1 >= 0) & (x1 <= w - 1) & (y1 >= 0) & (y1 <= h - 1)

    c00 = color[:, y0i, x0i]
    c01 = color[:, y1i, x0i]
    c10 = color[:, y0i, x1i]
    c11 = color[:, y1i, x1i]

    w00 = (wx0 * wy0 * v00).unsqueeze(0)
    w01 = (wx0 * wy1 * v01).unsqueeze(0)
    w10 = (wx1 * wy0 * v10).unsqueeze(0)
    w11 = (wx1 * wy1 * v11).unsqueeze(0)

    return c00 * w00 + c01 * w01 + c10 * w10 + c11 * w11


def forward_warp_naive(src_color: torch.Tensor, forward_flow: torch.Tensor) -> torch.Tensor:
    color = src_color
    flow = forward_flow
    c, h, w = color.shape

    out = torch.zeros_like(color)

    for sy in range(h):
        for sx in range(w):
            fx = float(flow[0, sy, sx])
            fy = float(flow[1, sy, sx])
            tx = sx + fx * 0.5 * (w - 1)
            ty = sy + fy * 0.5 * (h - 1)
            x0 = int(math.floor(tx))
            y0 = int(math.floor(ty))
            x1 = x0 + 1
            y1 = y0 + 1
            dx = tx - x0
            dy = ty - y0

            if 0 <= x0 < w and 0 <= y0 < h:
                out[:, y0, x0] += color[:, sy, sx] * (1.0 - dx) * (1.0 - dy)
            if 0 <= x0 < w and 0 <= y1 < h:
                out[:, y1, x0] += color[:, sy, sx] * (1.0 - dx) * dy
            if 0 <= x1 < w and 0 <= y0 < h:
                out[:, y0, x1] += color[:, sy, sx] * dx * (1.0 - dy)
            if 0 <= x1 < w and 0 <= y1 < h:
                out[:, y1, x1] += color[:, sy, sx] * dx * dy
    return out


def forward_warp_vectorize(src_color: torch.Tensor, forward_flow: torch.Tensor) -> torch.Tensor:
    color = src_color
    flow = forward_flow
    c, h, w = color.shape
    device = color.device
    dtype = color.dtype

    yy, xx = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    xx = xx.to(dtype=dtype)
    yy = yy.to(dtype=dtype)

    tx = xx + flow[0] * (0.5 * (w - 1))
    ty = yy + flow[1] * (0.5 * (h - 1))

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

    out_flat = torch.zeros((c, h * w), device=device, dtype=dtype)
    color_flat = color.reshape(c, -1)

    def _splat(mask, xi, yi, weight):
        mask_flat = mask.reshape(-1)
        if not mask_flat.any():
            return
        idx = (yi.reshape(-1) * w + xi.reshape(-1))[mask_flat]
        wv = weight.reshape(-1)[mask_flat]
        src = color_flat[:, mask_flat] * wv.unsqueeze(0)
        out_flat.scatter_add_(1, idx.unsqueeze(0).expand(c, -1), src)

    _splat(m00, x0i, y0i, w00)
    _splat(m01, x0i, y1i, w01)
    _splat(m10, x1i, y0i, w10)
    _splat(m11, x1i, y1i, w11)

    return out_flat.reshape(c, h, w)


def main():
    def _self_test():
        torch.manual_seed(0)
        c = torch.rand(3, 12, 17)
        f = torch.rand(2, 12, 17) * 0.4 - 0.2
        out_naive = backward_warp_naive(c, f)
        out_pt = backward_warp_gridsample(c, f)
        max_diff = (out_naive - out_pt).abs().max().item()
        print(f"self_test backward_warp_gridsample max_diff: {max_diff:.6g}")
        if not torch.allclose(out_naive, out_pt, rtol=1e-4, atol=1e-5):
            raise AssertionError("backward_warp_naive mismatch")

        f0 = torch.zeros(2, 12, 17)
        out_fw = forward_warp_naive(c, f0)
        max_fw = (out_fw - c).abs().max().item()
        print(f"self_test forward_warp_naive max_diff: {max_fw:.6g}")
        if not torch.allclose(out_fw, c, rtol=1e-4, atol=1e-5):
            raise AssertionError("forward_warp_naive zero-flow mismatch")

    _self_test()

def ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.float()
    y = y.float()
    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = (x - mu_x).pow(2).mean()
    sigma_y = (y - mu_y).pow(2).mean()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()
    c1 = 0.01 * 0.01
    c2 = 0.03 * 0.03
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    return numerator / denominator


if __name__ == "__main__":
    main()
