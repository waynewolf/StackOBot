from typing import Tuple
import numpy as np

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
