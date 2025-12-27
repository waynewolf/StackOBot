from typing import Tuple
import numpy as np

"""
All functions in this file works in numpy HWC format, need to transpose if used in pytorch CHW format
"""

def _decode_ufloat(bits: np.ndarray, mant_bits: int, exp_bits: int) -> np.ndarray:
    exp_mask = (1 << exp_bits) - 1
    mant_mask = (1 << mant_bits) - 1
    exp = (bits >> mant_bits) & exp_mask
    mant = bits & mant_mask

    bias = (1 << (exp_bits - 1)) - 1

    exp_f = exp.astype(np.int32)
    mant_f = mant.astype(np.float32)

    is_zero = exp_f == 0
    is_inf_nan = exp_f == exp_mask

    # Denorm: mantissa * 2^(1-bias) / 2^mant_bits
    denorm = mant_f * (2.0 ** (1 - bias - mant_bits))
    # Normal: (1 + mant/2^mant_bits) * 2^(exp-bias)
    norm = (1.0 + mant_f * (2.0 ** -mant_bits)) * np.exp2(exp_f - bias)

    out = np.where(is_zero, denorm, norm)
    out = np.where(is_inf_nan, np.inf, out)
    return out.astype(np.float32)


def read_r11g11b10_data(filename: str, height: int, width: int) -> np.ndarray:
    data = np.fromfile(filename, dtype=np.uint32)

    r_bits = data & 0x7FF
    g_bits = (data >> 11) & 0x7FF
    b_bits = (data >> 22) & 0x3FF

    r = _decode_ufloat(r_bits, mant_bits=6, exp_bits=5)
    g = _decode_ufloat(g_bits, mant_bits=6, exp_bits=5)
    b = _decode_ufloat(b_bits, mant_bits=5, exp_bits=5)

    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)

    if height > 0 and width > 0:
        expected = height * width
        if rgb.shape[0] >= expected:
            rgb = rgb[:expected].reshape((height, width, 3))
    return rgb


def read_r16g16b16a16_data(filename: str, height: int, width: int) -> np.ndarray:
    data = np.fromfile(filename, dtype=np.float16)
    if data.size < 4:
        return np.zeros((0, 3), dtype=np.float32)

    count = data.size // 4
    rgba = data[:count * 4].reshape((count, 4))
    rgb = rgba[:, :3].astype(np.float32)

    if height > 0 and width > 0:
        expected = height * width
        if rgb.shape[0] >= expected:
            rgb = rgb[:expected].reshape((height, width, 3))

    return rgb.astype(np.float32)


def read_g16r16_data(filename: str, height: int, width: int) -> np.ndarray:
    data = np.fromfile(filename, dtype=np.float16)
    if data.size < 2:
        return np.zeros((0, 2), dtype=np.float32)

    count = data.size // 2
    gr = data[:count * 2].reshape((count, 2)).astype(np.float32)

    if height > 0 and width > 0:
        expected = height * width
        if gr.shape[0] >= expected:
            gr = gr[:expected].reshape((height, width, 2))

    return gr.astype(np.float32)


def read_r32f_data(filename: str, height: int, width: int) -> np.ndarray:
    data = np.fromfile(filename, dtype=np.float32)
    if data.size == 0:
        return np.zeros((0, 1), dtype=np.float32)

    if height > 0 and width > 0:
        expected = height * width
        if data.shape[0] >= expected:
            data = data[:expected].reshape((height, width, 1))
    else:
        data = data.reshape((-1, 1))

    return data.astype(np.float32)

def visualize_motion01(motion01):
    """
    按通道缩放以可视化

    输入: numpy, H x W x 2, unorm
    输出: numpy, H x W x 2, unorm, scaled for visualization
    """
    x_min = np.min(motion01[..., 0])
    x_max = np.max(motion01[..., 0])
    y_min = np.min(motion01[..., 1])
    y_max = np.max(motion01[..., 1])

    x_range = np.clip(x_max - x_min, 0.000001, 1.0)
    y_range = np.clip(y_max - y_min, 0.000001, 1.0)

    scaled_motion = np.zeros_like(motion01)
    scaled_motion[..., 0] = (motion01[..., 0] - x_min) / x_range
    scaled_motion[..., 1] = (motion01[..., 1] - y_min) / y_range

    return scaled_motion

def visualize_image01(img01):
    """
    整体缩放以可视化

    输入: numpy, H x W x C, unorm
    输出: numpy, H x W x C, unorm, scaled for visualization 
    """
    min_value = np.min(img01)
    max_value = np.max(img01)
    z_range = max_value - min_value

    scaled_value = (img01 - min_value) / z_range
    scaled_value = np.clip(scaled_value, 0.0, 1.0)
    return scaled_value

def to_rgb_gray(img):
    if img.ndim == 2:
        return np.repeat(img[..., None], 3, axis=2)
    if img.shape[-1] == 1:
        return np.repeat(img, 3, axis=2)
    return img[..., :3]

def _self_test():
    import parser
    import pathlib
    import matplotlib.pyplot as plt

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


    RECORD_FOLDER = "Saved/NRSRecord"
    RECORD_DIR = pathlib.Path(RECORD_FOLDER)

    FRAME_NO = 412

    color_file, vp_dims, rt_dims = find_scene_color_file(RECORD_DIR, FRAME_NO)
    if color_file is None:
        raise FileNotFoundError("No SceneColor file found in record directory")

    motion_file, _, _ = find_motion_vector_file(RECORD_DIR, FRAME_NO)
    scene_depth_file, _, _ = find_scene_depth_file(RECORD_DIR, FRAME_NO)
    height, width = rt_dims
    print("SceneColor viewport size:", *vp_dims)
    print("SceneColor render target size:", height, width)

    # UE 的 SceneColorTexture 有时是 R16G16B16A16，有时是 R11G11B10 !
    color = parser.read_r16g16b16a16_data(str(color_file), height, width)
    motion = parser.read_g16r16_data(str(motion_file), height, width)
    scene_depth = parser.read_r32f_data(str(scene_depth_file), height, width)

    color01 = np.clip(color, 0.0, 1.0)

    # 一般从纹理中读取的运动向量是[-1, 1]范围内的值，这里将其映射到[0, 1]范围
    motion01 = np.clip(motion * 0.5 + 0.5, 0.0, 1.0)
    # motion缩放到最小值与最大值的区间，以利于可视化
    motion_vis = visualize_motion01(motion01)
    motion_rgb = np.zeros((motion_vis.shape[0], motion_vis.shape[1], 3), dtype=np.float32)
    motion_rgb[..., 0] = motion_vis[..., 0]
    motion_rgb[..., 1] = motion_vis[..., 1]

    # 一般从纹理中读取的深度值是Device Z，[0, 1]范围，这里缩放到最小值与最大值的区间，以利于可视化
    scene_depth_vis = visualize_image01(scene_depth)
    scene_depth_rgb = to_rgb_gray(scene_depth_vis)

    fig, axes = plt.subplots(1, 3, figsize=(12, 8))
    axes[0].imshow(color01)
    axes[0].set_title("SceneColor")
    axes[1].imshow(motion_rgb)
    axes[1].set_title("Motion (RG to UNORM)")
    axes[2].imshow(scene_depth_rgb)
    axes[2].set_title("SceneDepth")

    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

# python frame_extrap/parser.py
if __name__ == "__main__":
    _self_test()
