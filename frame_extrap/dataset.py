import os
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from parser import (
    read_r11g11b10_data,
    read_r16g16b16a16_data,
    read_g16r16_data,
    read_r32f_data,
    visualize_motion01,
    visualize_image01,
    to_rgb_gray
)

from utils import (
    backward_warp_pytorch
)

class UERecordDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        color_format: str = "r16g16b16a16f",
        depth_format: str = "r32f",
        motion_format: str = "g16r16f",
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.color_format = color_format
        self.depth_format = depth_format
        self.motion_format = motion_format
        self.transform = transform
        self.samples = self._scan_samples()

    def _scan_samples(self) -> List[Tuple[str, str, str, str, str, str, str, str, int, int, int, int]]:
        samples: List[Tuple[str, str, str, str, str, str, str, str, int, int, int, int]] = []
        for root, _, files in os.walk(self.root_dir):
            if not {
                "color_0.data",
                "color_1.data",
                "color_2.data",
                "depth_0.data",
                "depth_1.data",
                "motion_0.data",
                "motion_1.data",
                "motion_2.data",
            }.issubset(set(files)):
                continue
            crop_dir = os.path.basename(root)
            rt_dir = os.path.basename(os.path.dirname(root))
            if "x" not in crop_dir or "x" not in rt_dir:
                continue
            crop_h, crop_w = crop_dir.split("x", 1)
            rt_h, rt_w = rt_dir.split("x", 1)
            if (
                not crop_h.isdigit()
                or not crop_w.isdigit()
                or not rt_h.isdigit()
                or not rt_w.isdigit()
            ):
                continue
            ch, cw = int(crop_h), int(crop_w)
            rh, rw = int(rt_h), int(rt_w)
            samples.append(
                (
                    os.path.join(root, "color_0.data"),
                    os.path.join(root, "color_1.data"),
                    os.path.join(root, "color_2.data"),
                    os.path.join(root, "depth_0.data"),
                    os.path.join(root, "depth_1.data"),
                    os.path.join(root, "motion_0.data"),
                    os.path.join(root, "motion_1.data"),
                    os.path.join(root, "motion_2.data"),
                    rh,
                    rw,
                    ch,
                    cw,
                )
            )
        return sorted(samples)

    def _read_color(self, path: str, h: int, w: int) -> np.ndarray:
        if self.color_format == "r11g11b10":
            return read_r11g11b10_data(path, h, w)
        if self.color_format == "r16g16b16a16f":
            return read_r16g16b16a16_data(path, h, w)
        if self.color_format == "r32f":
            return read_r32f_data(path, h, w)
        raise ValueError(f"unknown color_format: {self.color_format}")

    def _read_depth(self, path: str, h: int, w: int) -> np.ndarray:
        if self.depth_format == "r32f":
            return read_r32f_data(path, h, w)
        raise ValueError(f"unknown depth_format: {self.depth_format}")

    def _read_motion(self, path: str, h: int, w: int) -> np.ndarray:
        if self.motion_format == "g16r16f":
            return read_g16r16_data(path, h, w)
        return ValueError(f"unknown motion_format: {self.motion_format}")

    def __len__(self) -> int:
        return len(self.samples)

    def get_crop_size(self) -> Optional[Tuple[int, int]]:
        if not self.samples:
            return None
        _, _, _, _, _, _, _, _, crop_h, crop_w = self.samples[0]
        return crop_h, crop_w

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        c0_path, c1_path, c2_path, d0_path, d1_path, v0_path, v1_path, v2_path, rt_h, rt_w, crop_h, crop_w = self.samples[index]
        c0_np = self._read_color(c0_path, rt_h, rt_w)
        c1_np = self._read_color(c1_path, rt_h, rt_w)
        c2_np = self._read_color(c2_path, rt_h, rt_w)
        d0_np = self._read_depth(d0_path, rt_h, rt_w)
        d1_np = self._read_depth(d1_path, rt_h, rt_w)
        v0_np = self._read_motion(v0_path, rt_h, rt_w)
        v1_np = self._read_motion(v1_path, rt_h, rt_w)
        v2_np = self._read_motion(v2_path, rt_h, rt_w)

        c0_tensor = torch.from_numpy(c0_np).permute(2, 0, 1).float()
        c1_tensor = torch.from_numpy(c1_np).permute(2, 0, 1).float()
        c2_tensor = torch.from_numpy(c2_np).permute(2, 0, 1).float()
        d0_tensor = torch.from_numpy(d0_np).permute(2, 0, 1).float()
        d1_tensor = torch.from_numpy(d1_np).permute(2, 0, 1).float()
        v0_tensor = torch.from_numpy(v0_np).permute(2, 0, 1).float()
        v1_tensor = torch.from_numpy(v1_np).permute(2, 0, 1).float()
        v2_tensor = torch.from_numpy(v2_np).permute(2, 0, 1).float()

        c0 = c0_tensor[:, :crop_h, :crop_w]
        c1 = c1_tensor[:, :crop_h, :crop_w]
        c2 = c2_tensor[:, :crop_h, :crop_w]
        d0 = d0_tensor[:, :crop_h, :crop_w]
        d1 = d1_tensor[:, :crop_h, :crop_w]
        v0 = v0_tensor[:, :crop_h, :crop_w]
        v1 = v1_tensor[:, :crop_h, :crop_w]
        v2 = v2_tensor[:, :crop_h, :crop_w]

        if self.transform:
            c0 = self.transform(c0)
            c1 = self.transform(c1)
            c2 = self.transform(c2)
            d0 = self.transform(d0)
            d1 = self.transform(d1)
            v0 = self.transform(v0)
            v1 = self.transform(v1)
            v2 = self.transform(v2)

        inputs = torch.concat([c0, d0, v0, c1, d1, v1], dim=0)
        labels = torch.concat([c1, c2, v2], dim=0)
        return inputs, labels


def _visualize(
    color0,
    depth0,
    velocity0,
    color1,
    depth1,
    velocity1,
    color2,
    velocity2,
    warped_color1_from_color2=None,
    diff_color1_warped=None,
    ssim_warped=None,
):
    """
    可视化color(unorm), depth(device z, unorm), motion(screen space, snorm)
    """
    import matplotlib.pyplot as plt

    def to_numpy(tensor):
        if torch.is_tensor(tensor):
            return tensor.detach().cpu().numpy()
        return tensor

    def to_color_rgb(color):
        return np.clip(to_numpy(color).transpose(1, 2, 0), 0.0, 1.0)

    def to_depth_rgb(depth):
        depth_np = to_numpy(depth).squeeze(0)
        depth_vis = visualize_image01(depth_np)
        return to_rgb_gray(depth_vis)

    def to_velocity_rgb(velocity):
        velocity_np = to_numpy(velocity).transpose(1, 2, 0)
        velocity01 = np.clip(velocity_np * 0.5 + 0.5, 0.0, 1.0)
        velocity_vis = visualize_motion01(velocity01)
        velocity_rgb = np.zeros((velocity_vis.shape[0], velocity_vis.shape[1], 3), dtype=np.float32)
        velocity_rgb[..., 0] = velocity_vis[..., 0]
        velocity_rgb[..., 1] = velocity_vis[..., 1]
        return velocity_rgb

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))

    axes[0, 0].imshow(to_color_rgb(color0))
    axes[0, 0].set_title("Color 0")
    axes[0, 0].axis("off")
    axes[0, 1].axis("off")
    axes[0, 2].imshow(to_depth_rgb(depth0))
    axes[0, 2].set_title("Depth 0")
    axes[0, 2].axis("off")
    axes[0, 3].imshow(to_velocity_rgb(velocity0))
    axes[0, 3].set_title("Velocity 0")
    axes[0, 3].axis("off")

    axes[1, 0].imshow(to_color_rgb(color1))
    axes[1, 0].set_title("Color 1")
    axes[1, 0].axis("off")
    if warped_color1_from_color2 is not None:
        axes[1, 1].imshow(to_color_rgb(warped_color1_from_color2))
        axes[1, 1].set_title("Warped C1 <-- C2")
        axes[1, 1].axis("off")
    else:
        axes[1, 1].axis("off")
    axes[1, 2].imshow(to_depth_rgb(depth1))
    axes[1, 2].set_title("Depth 1")
    axes[1, 2].axis("off")
    axes[1, 3].imshow(to_velocity_rgb(velocity1))
    axes[1, 3].set_title("Velocity 1")
    axes[1, 3].axis("off")

    axes[2, 0].imshow(to_color_rgb(color2))
    axes[2, 0].set_title("Color 2")
    axes[2, 0].axis("off")
    if diff_color1_warped is not None:
        axes[2, 1].imshow(to_color_rgb(diff_color1_warped))
        if ssim_warped is None:
            axes[2, 1].set_title("Diff C1 vs Warped C1")
        else:
            axes[2, 1].set_title(f"Diff C1 vs Warped C1 (SSIM={ssim_warped:.4f})")
        axes[2, 1].axis("off")
    else:
        axes[2, 1].axis("off")
    axes[2, 2].axis("off")
    axes[2, 3].imshow(to_velocity_rgb(velocity2))
    axes[2, 3].set_title("Velocity 2")
    axes[2, 3].axis("off")

    plt.tight_layout()
    plt.show()


def _self_test() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="UERecordDataset self-test")
    parser.add_argument("--root-dir", default="../train_data", help="dataset root")
    parser.add_argument("--color-format", default="r16g16b16a16f", help="r16g16b16a16f|r11g11b10")
    parser.add_argument("--depth-format", default="r32f")
    parser.add_argument("--motion-format", default="g16r16f")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    ds = UERecordDataset(
        root_dir=args.root_dir,
        color_format=args.color_format,
        depth_format=args.depth_format,
        motion_format=args.motion_format,
    )
    print(f"samples={len(ds)}")
    if len(ds) == 0:
        return
    idx = max(0, min(args.index, len(ds) - 1))
    inputs, labels = ds[idx]
    print(f"index={idx} inputs_shape={tuple(inputs.shape)} label_shape={tuple(labels.shape)}")

    color0 = inputs[0:3, ...]
    depth0 = inputs[3:4, ...]
    velocity0 = inputs[4:6, ...]
    color1 = inputs[6:9, ...]
    depth1 = inputs[9:10, ...]
    velocity1 = inputs[10:12, ...]
    color2 = labels[3:6, ...]
    velocity2 = labels[6:8, ...]

    import torch.nn.functional as F

    def _ssim(x, y):
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

    # source: color2, veloctity(2 <-- 1); dest: color2,
    # conform to pytorch grid_sample requirement
    warped_color1 = backward_warp_pytorch(color2, velocity2)
    diff_color1_warped = (color1 - warped_color1).abs().clamp(0.0, 1.0)
    ssim_warped = float(_ssim(color1, warped_color1).detach().cpu())
    _visualize(
        color0,
        depth0,
        velocity0,
        color1,
        depth1,
        velocity1,
        color2,
        velocity2,
        warped_color1_from_color2=warped_color1,
        diff_color1_warped=diff_color1_warped,
        ssim_warped=ssim_warped,
    )

# python frame_extrap/dataset.py --root-dir frame_extrap/train_data --color-format r16g16b16a16f --depth-format r32f --motion-format g16r16f --index 0
if __name__ == "__main__":
    _self_test()
