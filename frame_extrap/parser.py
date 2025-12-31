import numpy as np
import matplotlib.image as mpimg

"""
All functions in this file work in numpy HWC format.
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
    """
    当成颜色返回，返回值每个通道的格式 float32，范围 0，1
    """
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
    """
    当成color返回，返回值每个通道的格式 float32，范围 0，1
    """
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
    """
    当成MV返回，返回值每个通道的格式 float32，范围 -1，1
    """
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
    """
    当成深度返回，返回值单个通道，格式 float32，范围 0，1
    """
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


def read_png_color(filename: str) -> np.ndarray:
    """
    mpimg读取的png格式本身数值范围在0，1之间，返回值每个通道的格式 float32，范围 0，1
    """
    arr = mpimg.imread(filename)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError("read_png_color expects 3 or 4 channels")
    if np.issubdtype(arr.dtype, np.floating):
        out = arr.astype(np.float32, copy=False)
    else:
        info = np.iinfo(arr.dtype)
        out = arr.astype(np.float32) / float(info.max)
    return out


def read_png_depth(filename: str) -> np.ndarray:
    """
    mpimg读取的png格式本身数值范围在0，1之间，返回值单个通道，格式 float32，范围 0，1
    """
    arr = mpimg.imread(filename)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError("read_png_depth expects single channel")
    if np.issubdtype(arr.dtype, np.floating):
        out = arr.astype(np.float32, copy=False)
    else:
        info = np.iinfo(arr.dtype)
        out = arr.astype(np.float32) / float(info.max)
    return out[..., None]


def read_png_motion_disacard_last2(filename: str) -> np.ndarray:
    """
    mpimg读取的png格式本身数值范围在0，1之间，读取四通道，丢弃后两个通道，每个通道都强制转化成 float32 返回，范围 -1，1
    """
    arr = mpimg.imread(filename)
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError("read_png_motion_disacard_last2 expects 4 channels")
    # 返回 HxWx2 两个通道的 ndarray，通道 0 代表的是水平方向移动，通道 1 代表垂直方向移动
    out = arr[..., :2].astype(np.float32, copy=False)
    return out * 2.0 - 1.0


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
