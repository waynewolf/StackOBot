import os
import sys
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

    def _scan_samples(self) -> List[Tuple[str, str, str, str, str, str, str, int, int, int, int]]:
        samples: List[Tuple[str, str, str, str, str, str, str, int, int, int, int]] = []
        for root, _, files in os.walk(self.root_dir):
            if not {
                "color_0.data",
                "color_1.data",
                "color_2.data",
                "depth_0.data",
                "depth_1.data",
                "motion_0.data",
                "motion_1.data",
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
        _, _, _, _, _, _, _, crop_h, crop_w = self.samples[0]
        return crop_h, crop_w

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        c0, c1, c2, d0, d1, m0, m1, rt_h, rt_w, crop_h, crop_w = self.samples[index]
        img0 = self._read_color(c0, rt_h, rt_w)
        img1 = self._read_color(c1, rt_h, rt_w)
        img2 = self._read_color(c2, rt_h, rt_w)
        dep0 = self._read_depth(d0, rt_h, rt_w)
        dep1 = self._read_depth(d1, rt_h, rt_w)
        motion0 = self._read_motion(m0, rt_h, rt_w)
        motion1 = self._read_motion(m1, rt_h, rt_w)

        tc0 = torch.from_numpy(img0).permute(2, 0, 1).float()
        tc1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        tc2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        td0 = torch.from_numpy(dep0).permute(2, 0, 1).float()
        td1 = torch.from_numpy(dep1).permute(2, 0, 1).float()
        tm0 = torch.from_numpy(motion0).permute(2, 0, 1).float()
        tm1 = torch.from_numpy(motion1).permute(2, 0, 1).float()

        tc0 = tc0[:, :crop_h, :crop_w]
        tc1 = tc1[:, :crop_h, :crop_w]
        tc2 = tc2[:, :crop_h, :crop_w]
        td0 = td0[:, :crop_h, :crop_w]
        td1 = td1[:, :crop_h, :crop_w]
        tm0 = tm0[:, :crop_h, :crop_w]
        tm1 = tm1[:, :crop_h, :crop_w]

        if self.transform:
            tc0 = self.transform(tc0)
            tc1 = self.transform(tc1)
            tc2 = self.transform(tc2)
            td0 = self.transform(td0)
            td1 = self.transform(td1)
            tm0 = self.transform(tm0)
            tm1 = self.transform(tm1)

        inputs = torch.cat([tc0, td0, tm0, tc1, td1, tm1], dim=0)
        target = tc2
        return inputs, target


def _visualize(color, depth, motion):
    """
    可视化color(unorm), depth(device z, unorm), motion(screen space, snorm)
    """
    import matplotlib.pyplot as plt

    def to_numpy(tensor):
        if torch.is_tensor(tensor):
            return tensor.detach().cpu().numpy()
        return tensor

    color_np = to_numpy(color).transpose(1, 2, 0)
    depth_np = to_numpy(depth).squeeze(0)
    motion_np = to_numpy(motion).transpose(1, 2, 0)

    motion01 = np.clip(motion_np * 0.5 + 0.5, 0.0, 1.0)
    motion_vis = visualize_motion01(motion01)
    motion_rgb = np.zeros((motion_vis.shape[0], motion_vis.shape[1], 3), dtype=np.float32)
    motion_rgb[..., 0] = motion_vis[..., 0]
    motion_rgb[..., 1] = motion_vis[..., 1]

    depth_vis = visualize_image01(depth_np)
    depth_rgb = to_rgb_gray(depth_vis)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(np.clip(color_np, 0.0, 1.0))
    axes[0].set_title("Color")
    axes[0].axis("off")

    axes[1].imshow(depth_rgb)
    axes[1].set_title("Depth")
    axes[1].axis("off")

    axes[2].imshow(motion_rgb)
    axes[2].set_title("Cam Motion")
    axes[2].axis("off")

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
    inputs, target = ds[idx]
    print(f"index={idx} inputs_shape={tuple(inputs.shape)} target_shape={tuple(target.shape)}")

    color0 = inputs[0:3, ...]
    depth0 = inputs[3:4, ...]
    motion0 = inputs[4:6, ...]
    # color1 = inputs[6:9, ...]
    # depth1 = inputs[9:10, ...]
    # motion1 = inputs[10:12, ...]
    _visualize(color0, depth0, motion0)

# python frame_extrap/dataset.py --root-dir frame_extrap/train_data --color-format r16g16b16a16f --depth-format r32f --motion-format g16r16f --index 0
if __name__ == "__main__":
    _self_test()
