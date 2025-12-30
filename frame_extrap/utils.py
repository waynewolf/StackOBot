import torch
import torch.nn.functional as F

# F.grid_sample 严格来说只支持 backward_flow, 所以这是标准用法，
# backward_flow 指的是，从目的指向源
def backward_warp_pytorch(src_color, backward_flow):
    color = src_color.unsqueeze(0)
    flow = backward_flow.unsqueeze(0)
    _, _, h, w = color.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=color.device),
        torch.linspace(-1.0, 1.0, w, device=color.device),
        indexing="ij",
    )
    grid = torch.stack([xx, yy], dim=-1).unsqueeze(0)
    # 输出坐标加上flow去采样源，是正确的调用方式
    grid = grid + flow.permute(0, 2, 3, 1)
    warped = F.grid_sample(color, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return warped.squeeze(0)

def backward_warp_naive(src_color, backward_flow):
    color = src_color
    flow = backward_flow
    _, h, w = color.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=color.device),
        torch.linspace(-1.0, 1.0, w, device=color.device),
        indexing="ij",
    )

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


def forward_warp_naive(src_color, forward_flow):
    color = src_color
    flow = forward_flow
    c, h, w = color.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=color.device),
        torch.linspace(-1.0, 1.0, w, device=color.device),
        indexing="ij",
    )

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

    v00 = (x0 >= 0) & (x0 <= w - 1) & (y0 >= 0) & (y0 <= h - 1)
    v01 = (x0 >= 0) & (x0 <= w - 1) & (y1 >= 0) & (y1 <= h - 1)
    v10 = (x1 >= 0) & (x1 <= w - 1) & (y0 >= 0) & (y0 <= h - 1)
    v11 = (x1 >= 0) & (x1 <= w - 1) & (y1 >= 0) & (y1 <= h - 1)

    out = torch.zeros_like(color)
    weight = torch.zeros((1, h, w), device=color.device, dtype=color.dtype)

    out_flat = out.view(c, -1)
    weight_flat = weight.view(1, -1)
    color_flat = color.view(c, -1)

    def _splat(xi, yi, wgt, valid):
        idx = (yi.clamp(0, h - 1).long() * w + xi.clamp(0, w - 1).long()).view(-1)
        w_flat = (wgt * valid).view(-1)
        if w_flat.numel() == 0:
            return
        out_flat.scatter_add_(1, idx.unsqueeze(0).expand(c, -1), color_flat * w_flat.unsqueeze(0))
        weight_flat.scatter_add_(1, idx.unsqueeze(0), w_flat.unsqueeze(0))

    _splat(x0, y0, wx0 * wy0, v00)
    _splat(x0, y1, wx0 * wy1, v01)
    _splat(x1, y0, wx1 * wy0, v10)
    _splat(x1, y1, wx1 * wy1, v11)

    denom = torch.where(weight > 0, weight, torch.ones_like(weight))
    out = out / denom
    out = out * (weight > 0).to(out.dtype)
    return out


def main():
    def _self_test():
        torch.manual_seed(0)
        c = torch.rand(3, 12, 17)
        f = torch.rand(2, 12, 17) * 0.4 - 0.2
        out_naive = backward_warp_naive(c, f)
        out_pt = backward_warp_pytorch(c, f)
        max_diff = (out_naive - out_pt).abs().max().item()
        print(f"self_test max_diff: {max_diff:.6g}")
        if not torch.allclose(out_naive, out_pt, rtol=1e-4, atol=1e-5):
            raise AssertionError("backward_warp_naive mismatch")

        f0 = torch.zeros(2, 12, 17)
        out_fw = forward_warp_naive(c, f0)
        max_fw = (out_fw - c).abs().max().item()
        print(f"self_test forward_zero_flow max_diff: {max_fw:.6g}")
        if not torch.allclose(out_fw, c, rtol=1e-4, atol=1e-5):
            raise AssertionError("forward_warp_naive zero-flow mismatch")

    _self_test()


if __name__ == "__main__":
    main()
