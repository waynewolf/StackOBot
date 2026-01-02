import argparse
import os
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import UERecordDataset
from net import GFENet
from utils import forward_warp


@dataclass
class TrainConfig:
    data_dir: str
    output_dir: str
    crop_height: int
    crop_width: int
    batch_size: int
    epochs: int
    lr: float
    num_workers: int
    amp: bool
    seed: int
    val_split: float


class CenterCropTransform:
    def __init__(self, crop_height: int, crop_width: int) -> None:
        self.crop_height = crop_height
        self.crop_width = crop_width

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.crop_height <= 0 or self.crop_width <= 0:
            return tensor
        _, h, w = tensor.shape
        if h < self.crop_height or w < self.crop_width:
            raise ValueError(
                f"tensor spatial size ({h}x{w}) smaller than crop {self.crop_height}x{self.crop_width}"
            )
        top = (h - self.crop_height) // 2
        left = (w - self.crop_width) // 2
        return tensor[:, top : top + self.crop_height, left : left + self.crop_width]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Train GFENet")
    parser.add_argument("--data-dir", default="frame_extrap/train_data", help="dataset root")
    parser.add_argument("--output-dir", default="frame_extrap/outputs", help="save checkpoints")
    parser.add_argument("--crop-height", type=int, default=352, help="center crop height")
    parser.add_argument("--crop-width", type=int, default=448, help="center crop width")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.1)
    args = parser.parse_args()

    cfg = TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        crop_height=args.crop_height,
        crop_width=args.crop_width,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        num_workers=args.num_workers,
        amp=args.amp,
        seed=args.seed,
        val_split=args.val_split,
    )

    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    def _latest_run_name(root: str) -> str | None:
        candidates = [
            name
            for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name)) and name.startswith("run_")
        ]
        candidates.sort()
        return candidates[-1] if candidates else None

    def _latest_checkpoint_path(ckpt_path: str) -> str | None:
        if not os.path.isdir(ckpt_path):
            return None
        checkpoints: list[tuple[int, str]] = []
        prefix = "gfenet_epoch_"
        suffix = ".pt"
        for fname in os.listdir(ckpt_path):
            if fname.startswith(prefix) and fname.endswith(suffix):
                num_part = fname[len(prefix) : -len(suffix)]
                if num_part.isdigit():
                    checkpoints.append((int(num_part), fname))
        if not checkpoints:
            return None
        checkpoints.sort()
        return os.path.join(ckpt_path, checkpoints[-1][1])

    latest_run_name = _latest_run_name(cfg.output_dir)
    resume_ckpt_path = None
    if latest_run_name:
        potential_run_dir = os.path.join(cfg.output_dir, latest_run_name)
        potential_ckpt_dir = os.path.join(potential_run_dir, "checkpoints")
        resume_ckpt_path = _latest_checkpoint_path(potential_ckpt_dir)
        if resume_ckpt_path:
            run_name = latest_run_name
            run_dir = potential_run_dir
            ckpt_dir = potential_ckpt_dir
            log_dir = os.path.join(run_dir, "logs")
        else:
            run_name = None
            resume_ckpt_path = None
    else:
        run_name = None
        resume_ckpt_path = None

    if run_name is None:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_dir = os.path.join(cfg.output_dir, run_name)
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        log_dir = os.path.join(run_dir, "logs")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    dataset = UERecordDataset(
        root_dir=cfg.data_dir,
        transform=CenterCropTransform(cfg.crop_height, cfg.crop_width),
    )
    if len(dataset) == 0:
        raise RuntimeError(f"no samples found under {cfg.data_dir}")

    val_count = int(len(dataset) * cfg.val_split)
    train_count = len(dataset) - val_count

    generator = torch.Generator().manual_seed(cfg.seed)
    if val_count > 0:
        train_ds, val_ds = torch.utils.data.random_split(
            dataset, [train_count, val_count], generator=generator
        )
    else:
        train_ds, val_ds = dataset, None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    use_cuda = torch.cuda.is_available()
    device_str = "cuda" if use_cuda else "cpu"
    device = torch.device(device_str)
    model = GFENet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    amp_enabled = cfg.amp and use_cuda
    scaler = torch.amp.GradScaler(enabled=amp_enabled)

    writer = SummaryWriter(log_dir=log_dir)
    global_step = 0
    best_val = None
    start_epoch = 1

    if resume_ckpt_path:
        checkpoint = torch.load(resume_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler_state = checkpoint.get("scaler_state")
        if scaler_state is not None:
            scaler.load_state_dict(scaler_state)
        best_val = checkpoint.get("best_val")
        global_step = checkpoint.get("global_step", 0)
        start_epoch = checkpoint.get("epoch", 0) + 1
        if start_epoch > cfg.epochs:
            print("Checkpoint already covers requested training epochs. Nothing to do.")
            writer.close()
            return
        print(f"Resuming from {resume_ckpt_path} (epoch {checkpoint['epoch']})")

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        for step, (inputs, target) in enumerate(train_loader, start=1):
            inputs = inputs.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            group0 = inputs[:, 0:6, ...]
            group1 = inputs[:, 6:12, ...]
            group2 = target

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device_str, enabled=amp_enabled):
                _, loss = model(group0, group1, group2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * inputs.size(0)
            global_step += 1
            writer.add_scalar("loss/batch", loss.item(), global_step)

            if step % 100 == 0:
                print(
                    f"epoch {epoch} step {step} / {len(train_loader)} "
                    f"batch_loss={loss.item():.6f}"
                )

        train_loss /= len(train_loader.dataset)

        val_loss = None
        if val_loader is not None:
            model.eval()
            total = 0.0
            count = 0
            sample_logged = False
            with torch.no_grad():
                for inputs, target in val_loader:
                    inputs = inputs.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)

                    group0 = inputs[:, 0:6, ...]
                    group1 = inputs[:, 6:12, ...]
                    group2 = target

                    with torch.amp.autocast(device_type=device_str, enabled=amp_enabled):
                        f_12_0, loss = model(group0, group1, group2)
                    total += loss.item() * inputs.size(0)
                    count += inputs.size(0)

                    # select first GT in the first batch for visualization
                    if not sample_logged:
                        C1_0_gt = group2[0, 0:3, ...]
                        C2_0_gt = group2[0, 3:6, ...]
                        C2_0_pred = forward_warp(C1_0_gt, f_12_0[0, ...])
                        writer.add_image(
                            f"im_gt/{epoch:04d}_C2_0",
                            C2_0_gt.detach().clamp(0.0, 1.0),
                            epoch,
                        )
                        writer.add_image(
                            f"im_pred/{epoch:04d}_C2_0",
                            C2_0_pred.detach().clamp(0.0, 1.0),
                            epoch,
                        )
                        sample_logged = True

            val_loss = total / max(1, count)

        writer.add_scalars(
            "loss/epoch",
            {
                "train": train_loss,
                "val": val_loss if val_loss is not None else float("nan"),
            },
            epoch,
        )

        ckpt_path = os.path.join(ckpt_dir, f"gfenet_epoch_{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val": best_val,
                "global_step": global_step,
                "config": cfg.__dict__,
            },
            ckpt_path,
        )

        if val_loss is not None and (best_val is None or val_loss < best_val):
            best_val = val_loss
            best_path = os.path.join(ckpt_dir, "gfenet_best.pt")
            torch.save({"epoch": epoch, "model_state": model.state_dict()}, best_path)

        if val_loss is not None:
            print(f"epoch {epoch}/{cfg.epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        else:
            print(f"epoch {epoch}/{cfg.epochs} train_loss={train_loss:.6f}")

    writer.close()

if __name__ == "__main__":
    main()
