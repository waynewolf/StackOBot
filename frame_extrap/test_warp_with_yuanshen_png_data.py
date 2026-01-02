import torch
import utils
import pathlib
import matplotlib.pyplot as plt
from parser import *
from metrics import ssim

"""
使用yuanshen png数据验证 utils.py 中的 backward_warp, forward_warp 等函数，
注意一般 Motion Vector 是当前帧指向前一帧的, 和 nrsrecord 不同!
"""

def _yuanshen_mv_to_true_mv(ys_mv_snorm: np.ndarray) -> np.ndarray:
    """
    原神专用的移动矢量解码，输入和输出都是[-1,1]
    """

    negative = -(ys_mv_snorm < 0.0).astype(np.int8)
    positive = (ys_mv_snorm > 0.0).astype(np.int8)
    sign = negative + positive
    true_motion_snorm = sign * ys_mv_snorm * ys_mv_snorm

    # 这里通道0（水平方向）翻转可视化应该是对的，但 GLES 的屏幕空间是 Y 向上，需要翻转 Y，矛盾！why？
    true_motion_snorm[..., 0] *= -1.0

    return true_motion_snorm


def main():
    YS_PNG_FOLDER = "frame_extrap/yuanshen_png_data/fastmove"
    YS_PNG_DIR = pathlib.Path(YS_PNG_FOLDER)

    ys_color0 = read_png_color(str(YS_PNG_DIR / "color-0.png"))
    ys_color1 = read_png_color(str(YS_PNG_DIR / "color-1.png"))
    ys_color2 = read_png_color(str(YS_PNG_DIR / "color-2.png"))
    ys_mv0 = read_png_motion_disacard_last2(str(YS_PNG_DIR / "mv-0.png"))
    ys_mv1 = read_png_motion_disacard_last2(str(YS_PNG_DIR / "mv-1.png"))
    ys_mv2 = read_png_motion_disacard_last2(str(YS_PNG_DIR / "mv-2.png"))

    # Yuanshen MV specific adjustment
    ys_mv0 = _yuanshen_mv_to_true_mv(ys_mv0)
    ys_mv1 = _yuanshen_mv_to_true_mv(ys_mv1)
    ys_mv2 = _yuanshen_mv_to_true_mv(ys_mv2)

    ys_color0_t = torch.from_numpy(ys_color0[..., :3]).permute(2, 0, 1).contiguous()
    ys_color1_t = torch.from_numpy(ys_color1[..., :3]).permute(2, 0, 1).contiguous()
    ys_mv0_t = torch.from_numpy(ys_mv0).permute(2, 0, 1).contiguous()
    ys_mv1_t = torch.from_numpy(ys_mv1).permute(2, 0, 1).contiguous()
    ys_mv2_t = torch.from_numpy(ys_mv2).permute(2, 0, 1).contiguous()

    # 前向 warp: forward_warp(C1, F1->0) => C0，注意原神的 Motion 指向前一帧
    warped_color0 = utils.forward_warp(ys_color1_t, ys_mv1_t)
    diff_color0 = (ys_color0_t - warped_color0).abs().clamp(0.0, 1.0)
    ssim_color0 = float(ssim(ys_color0_t, warped_color0).detach().cpu())

    # 反向 warp: backward_warp(C0, F0<-1) => C1，但原神没有 F0<-1，这里用 F1->0 近似
    warped_color1 = utils.backward_warp(ys_color0_t, ys_mv1_t)
    diff_color1 = (ys_color1_t - warped_color1).abs().clamp(0.0, 1.0)
    ssim_color1 = float(ssim(ys_color1_t, warped_color1).detach().cpu())

    ys_mv0_01 = np.clip(ys_mv0 * 0.5 + 0.5, 0.0, 1.0)
    ys_mv0_vis = visualize_motion01(ys_mv0_01)
    ys_mv0_rgb = np.zeros((ys_mv0_vis.shape[0], ys_mv0_vis.shape[1], 3), dtype=np.float32)
    ys_mv0_rgb[..., 0] = ys_mv0_vis[..., 0]
    ys_mv0_rgb[..., 1] = ys_mv0_vis[..., 1]

    ys_mv1_01 = np.clip(ys_mv1 * 0.5 + 0.5, 0.0, 1.0)
    ys_mv1_vis = visualize_motion01(ys_mv1_01)
    ys_mv1_rgb = np.zeros((ys_mv1_vis.shape[0], ys_mv1_vis.shape[1], 3), dtype=np.float32)
    ys_mv1_rgb[..., 0] = ys_mv1_vis[..., 0]
    ys_mv1_rgb[..., 1] = ys_mv1_vis[..., 1]

    ys_mv2_01 = np.clip(ys_mv2 * 0.5 + 0.5, 0.0, 1.0)
    ys_mv2_vis = visualize_motion01(ys_mv2_01)
    ys_mv2_rgb = np.zeros((ys_mv2_vis.shape[0], ys_mv2_vis.shape[1], 3), dtype=np.float32)
    ys_mv2_rgb[..., 0] = ys_mv2_vis[..., 0]
    ys_mv2_rgb[..., 1] = ys_mv2_vis[..., 1]

    def _to_color_rgb(img):
        if torch.is_tensor(img):
            img = img.detach().cpu().numpy()
        if img.ndim == 3 and img.shape[0] in (3, 4):
            img = img[:3].transpose(1, 2, 0)
        return np.clip(img[..., :3], 0.0, 1.0)

    fig, axes = plt.subplots(3, 4, figsize=(12, 4))
    axes[0, 0].imshow(_to_color_rgb(ys_color0))
    axes[0, 0].set_title("C 0")
    axes[0, 1].imshow(ys_mv0_rgb)
    axes[0, 1].set_title("M 0")
    axes[0, 2].imshow(_to_color_rgb(ys_color1))
    axes[0, 2].set_title("C 1")
    axes[0, 3].imshow(ys_mv1_rgb)
    axes[0, 3].set_title("M 1")

    axes[1, 0].imshow(_to_color_rgb(warped_color0))
    axes[1, 0].set_title("Forward Warped C 0 <-- 1")
    axes[1, 1].imshow(_to_color_rgb(diff_color0))
    axes[1, 1].set_title(f"Forward Diff 0 (SSIM={ssim_color0:.4f})")
    axes[1, 2].imshow(_to_color_rgb(warped_color1))
    axes[1, 2].set_title("Backward Warped(Approx.) C 1 <-- 0")
    axes[1, 3].imshow(_to_color_rgb(diff_color1))
    axes[1, 3].set_title(f"Backward Diff 1 (SSIM={ssim_color1:.4f})")

    axes[2, 0].imshow(_to_color_rgb(ys_color2))
    axes[2, 0].set_title("C 2")
    axes[2, 1].imshow(ys_mv2_rgb)
    axes[2, 1].set_title("M 2")

    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

# python frame_extrap/test_warp_with_yuanshen_png_data.py
if __name__ == "__main__":
    main()
