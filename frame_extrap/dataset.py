import os
import sys
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from parser import (
        read_r11g11b10_data,
        read_r16g16b16a16_data,
        read_r32f_data,
    )
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(__file__))
    from parser import (  # type: ignore[no-redef]
        read_r11g11b10_data,
        read_r16g16b16a16_data,
        read_r32f_data,
    )


class UERecordDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        color_format: str = "r11g11b10",
        depth_format: str = "r32f",
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.color_format = color_format
        self.depth_format = depth_format
        self.transform = transform
        self.samples = self._scan_samples()

    def _scan_samples(self) -> List[Tuple[str, str, str, str, str, int, int, int, int]]:
        samples: List[Tuple[str, str, str, str, str, int, int, int, int]] = []
        for root, _, files in os.walk(self.root_dir):
            if not {
                "color_0.data",
                "color_1.data",
                "color_2.data",
                "depth_0.data",
                "depth_1.data",
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
        if self.color_format == "r16g16b16a16":
            return read_r16g16b16a16_data(path, h, w)
        if self.color_format == "r32f":
            return read_r32f_data(path, h, w)
        raise ValueError(f"unknown color_format: {self.color_format}")

    def _read_depth(self, path: str, h: int, w: int) -> np.ndarray:
        if self.depth_format == "r32f":
            return read_r32f_data(path, h, w)
        raise ValueError(f"unknown depth_format: {self.depth_format}")

    def __len__(self) -> int:
        return len(self.samples)

    def get_crop_size(self) -> Optional[Tuple[int, int]]:
        if not self.samples:
            return None
        _, _, _, _, _, _, _, crop_h, crop_w = self.samples[0]
        return crop_h, crop_w

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        c0, c1, c2, d0, d1, rt_h, rt_w, crop_h, crop_w = self.samples[index]
        img0 = self._read_color(c0, rt_h, rt_w)
        img1 = self._read_color(c1, rt_h, rt_w)
        img2 = self._read_color(c2, rt_h, rt_w)
        dep0 = self._read_depth(d0, rt_h, rt_w)
        dep1 = self._read_depth(d1, rt_h, rt_w)

        t0 = torch.from_numpy(img0).permute(2, 0, 1).float()
        t1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        t2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        td0 = torch.from_numpy(dep0).permute(2, 0, 1).float()
        td1 = torch.from_numpy(dep1).permute(2, 0, 1).float()

        t0 = t0[:, :crop_h, :crop_w]
        t1 = t1[:, :crop_h, :crop_w]
        t2 = t2[:, :crop_h, :crop_w]
        td0 = td0[:, :crop_h, :crop_w]
        td1 = td1[:, :crop_h, :crop_w]

        if self.transform:
            t0 = self.transform(t0)
            t1 = self.transform(t1)
            t2 = self.transform(t2)
            td0 = self.transform(td0)
            td1 = self.transform(td1)

        inputs = torch.cat([t0, td0, t1, td1], dim=0)
        target = t2
        return inputs, target


def _self_test() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="UERecordDataset self-test")
    parser.add_argument("--root-dir", default="../train_data", help="dataset root")
    parser.add_argument("--color-format", default="r11g11b10")
    parser.add_argument("--depth-format", default="r32f")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    ds = UERecordDataset(
        root_dir=args.root_dir,
        color_format=args.color_format,
        depth_format=args.depth_format,
    )
    print(f"samples={len(ds)}")
    if len(ds) == 0:
        return
    idx = max(0, min(args.index, len(ds) - 1))
    inputs, target = ds[idx]
    print(f"index={idx} inputs_shape={tuple(inputs.shape)} target_shape={tuple(target.shape)}")

# python frame_extrap/dataset.py --root-dir frame_extrap/train_data --color-format r11g11b10 --depth-format r32f --index 0
if __name__ == "__main__":
    _self_test()
