import numpy as np
import torch
import utils
from metrics import ssim
import pathlib
import matplotlib.pyplot as plt
from parser import *
import argparse

"""
使用 nrsrecord 数据验证 utils.py 中的 backward_warp, forward_warp 等函数,
注意 nrsrecord 的 Velocity 是前一帧指向当前帧, 与原神 GLES 版不同!
"""


def to_color_rgb(img):
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    if img.ndim == 3 and img.shape[0] in (3, 4):
        img = img[:3].transpose(1, 2, 0)
    return np.clip(img[..., :3], 0.0, 1.0)


def format_frame_no(frame_no):
    return f"{int(frame_no):06d}"


def parse_dim_pair(dim_text):
    if "x" not in dim_text:
        return None
    height_str, width_str = dim_text.split("x", 1)
    if not height_str.isdigit() or not width_str.isdigit():
        return None
    return int(height_str), int(width_str)

def parse_record_dimensions(filename):
    name = pathlib.Path(filename).name
    parts = name.split("_")
    if len(parts) < 5:
        return None
    if not parts[0].isdigit():
        return None
    if "in" not in parts:
        return None
    in_index = parts.index("in")
    if in_index <= 0 or in_index + 1 >= len(parts):
        return None
    crop_part = parts[in_index - 1]
    full_part = parts[in_index + 1]
    if not full_part.endswith(".data"):
        return None
    full_part = full_part[:-5]
    crop_dims = parse_dim_pair(crop_part)
    full_dims = parse_dim_pair(full_part)
    if not crop_dims or not full_dims:
        return None
    return crop_dims, full_dims

def find_scene_color_file(record_dir, frame_no):
    frame_prefix = format_frame_no(frame_no)
    for path in record_dir.glob(f"{frame_prefix}_SceneColor_*x*_in_*x*.data"):
        dims = parse_record_dimensions(path.name)
        if dims:
            crop_dims, full_dims = dims
            return path, crop_dims, full_dims
    return None, None, None

def find_motion_vector_file(record_dir, frame_no):
    frame_prefix = format_frame_no(frame_no)
    for path in record_dir.glob(f"{frame_prefix}_CameraMotion_*x*_in_*x*.data"):
        dims = parse_record_dimensions(path.name)
        if dims:
            crop_dims, full_dims = dims
            return path, crop_dims, full_dims
    return None, None, None

def find_scene_depth_file(record_dir, frame_no):
    frame_prefix = format_frame_no(frame_no)
    for path in record_dir.glob(f"{frame_prefix}_SceneDepth_*x*_in_*x*.data"):
        dims = parse_record_dimensions(path.name)
        if dims:
            crop_dims, full_dims = dims
            return path, crop_dims, full_dims
    return None, None, None

def main():
    parser = argparse.ArgumentParser(description="test warp with nrsrecord data")
    parser.add_argument("--root-dir", default="Saved/NRSRecord", help="dataset root")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    RECORD_DIR = pathlib.Path(args.root_dir)

    color_file0, vp_dims, rt_dims = find_scene_color_file(RECORD_DIR, args.index)
    motion_file0, _, _ = find_motion_vector_file(RECORD_DIR, args.index)
    depth_file0, _, _ = find_scene_depth_file(RECORD_DIR, args.index)
    color_file1, _, _ = find_scene_color_file(RECORD_DIR, args.index+1)
    motion_file1, _, _ = find_motion_vector_file(RECORD_DIR, args.index+1)
    depth_file1, _, _ = find_scene_depth_file(RECORD_DIR, args.index+1)

    height, width = rt_dims
    print("SceneColor viewport size:", *vp_dims)
    print("SceneColor render target size:", height, width)

    vp_height, vp_width = vp_dims

    def crop_to_viewport(arr):
        return arr[:vp_height, :vp_width, ...]

    # NRSRecord 的 SceneColorTexture 有时是 R16G16B16A16，有时是 R11G11B10 !
    color0_np = crop_to_viewport(read_r16g16b16a16_data(str(color_file0), height, width))
    motion0_np = crop_to_viewport(read_g16r16_data(str(motion_file0), height, width))
    depth0_np = crop_to_viewport(read_r32f_data(str(depth_file0), height, width))
    color1_np = crop_to_viewport(read_r16g16b16a16_data(str(color_file1), height, width))
    motion1_np = crop_to_viewport(read_g16r16_data(str(motion_file1), height, width))
    depth1_np = crop_to_viewport(read_r32f_data(str(depth_file1), height, width))

    color0_t = torch.from_numpy(color0_np).permute(2, 0, 1)  # 3xHxW
    color1_t = torch.from_numpy(color1_np).permute(2, 0, 1)  # 3xHxW

    # NRSRecord 录制的 motion 是前一帧指向当前帧的, backward_warp(C1, V1<-0) => C0
    warp_color0 = utils.backward_warp(
        color1_t,  # 3xHxW
        torch.from_numpy(motion1_np).permute(2, 0, 1)  # 2xHxW
    ).permute(1, 2, 0).numpy()  # HxWxC

    warp_color0_t = torch.from_numpy(warp_color0).permute(2, 0, 1)  # 3xHxW
    
    diff0_t = (warp_color0_t - color0_t).abs().clamp(0.0, 1.0)
    diff_color0 = diff0_t.permute(1, 2, 0).cpu().numpy()
    ssim_color0 = float(ssim(color0_t, warp_color0_t).detach().cpu())

    color0_vis = np.clip(color0_np, 0.0, 1.0)
    color1_vis = np.clip(color1_np, 0.0, 1.0)

    # 这里从纹理中读取的运动向量是[-1, 1]范围内的值，这里将其映射到[0, 1]范围
    motion01_0 = np.clip(motion0_np * 0.5 + 0.5, 0.0, 1.0)
    # motion缩放到最小值与最大值的区间，以利于可视化
    motion_vis0 = visualize_motion01(motion01_0)
    motion_rgb0 = np.zeros((motion_vis0.shape[0], motion_vis0.shape[1], 3), dtype=np.float32)
    motion_rgb0[..., 0] = motion_vis0[..., 0]
    motion_rgb0[..., 1] = motion_vis0[..., 1]

    motion01_1 = np.clip(motion1_np * 0.5 + 0.5, 0.0, 1.0)
    motion_vis1 = visualize_motion01(motion01_1)
    motion_rgb1 = np.zeros((motion_vis1.shape[0], motion_vis1.shape[1], 3), dtype=np.float32)
    motion_rgb1[..., 0] = motion_vis1[..., 0]
    motion_rgb1[..., 1] = motion_vis1[..., 1]

    # 一般从纹理中读取的深度值是Device Z，[0, 1]范围，这里缩放到最小值与最大值的区间，以利于可视化
    depth_vis0 = visualize_image01(depth0_np)
    depth_rgb0 = to_rgb_gray(depth_vis0)
    depth_vis1 = visualize_image01(depth1_np)
    depth_rgb1 = to_rgb_gray(depth_vis1)

    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    axes[0, 0].imshow(color0_vis)
    axes[0, 0].set_title("C 0")
    axes[0, 1].imshow(motion_rgb0)
    axes[0, 1].set_title("M 0")
    axes[0, 2].imshow(depth_rgb0)
    axes[0, 2].set_title("D 0")

    axes[1, 0].imshow(to_color_rgb(warp_color0))
    axes[1, 0].set_title("Backward Warped C 0 <-- 1")
    axes[1, 1].imshow(to_color_rgb(diff_color0))
    axes[1, 1].set_title(f"Diff 0 (SSIM={ssim_color0:.4f})")
    axes[1, 2].axis("off")

    axes[2, 0].imshow(color1_vis)
    axes[2, 0].set_title("C 1")
    axes[2, 1].imshow(motion_rgb1)
    axes[2, 1].set_title("M 1")
    axes[2, 2].imshow(depth_rgb1)
    axes[2, 2].set_title("D 1")
    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# python frame_extrap/test_warp_with_nrsrecord_data.py --index=7657
if __name__ == "__main__":
    main()
