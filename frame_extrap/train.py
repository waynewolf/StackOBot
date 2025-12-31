import argparse
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset import UERecordDataset
from net import GFENet


@dataclass
class TrainConfig:
    data_dir: str
    output_dir: str
    image_size: int
    batch_size: int
    epochs: int
    lr: float
    num_workers: int
    amp: bool
    seed: int
    val_split: float


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Train GFENet")
    parser.add_argument("--data-dir", default="frame_extrap/train_data", help="dataset root")
    parser.add_argument("--output-dir", default="frame_extrap/outputs", help="save checkpoints")
    parser.add_argument("--image-size", type=int, default=0, help="optional resize to square size")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.1)
    args = parser.parse_args()

    cfg = TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
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

    def _resize_if_needed(t: torch.Tensor) -> torch.Tensor:
        if cfg.image_size and cfg.image_size > 0:
            return F.interpolate(
                t.unsqueeze(0),
                size=(cfg.image_size, cfg.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return t

    dataset = UERecordDataset(root_dir=cfg.data_dir, transform=_resize_if_needed)
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
    device = torch.device("cuda" if use_cuda else "cpu")
    model = GFENet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    amp_enabled = cfg.amp and use_cuda
    scaler = torch.amp.GradScaler(enabled=amp_enabled)

    best_val = None
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        for inputs, target in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            group0 = inputs[:, 0:6, ...]
            group1 = inputs[:, 6:12, ...]
            group2 = torch.cat([inputs[:, 6:9, ...], target, inputs[:, 10:12, ...]], dim=1)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                _, loss = model(group0, group1, group2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        val_loss = None
        if val_loader is not None:
            model.eval()
            total = 0.0
            count = 0
            with torch.no_grad():
                for inputs, target in val_loader:
                    inputs = inputs.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)

                    group0 = inputs[:, 0:6, ...]
                    group1 = inputs[:, 6:12, ...]
                    group2 = torch.cat([inputs[:, 6:9, ...], target, inputs[:, 10:12, ...]], dim=1)

                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        _, loss = model(group0, group1, group2)
                    total += loss.item() * inputs.size(0)
                    count += inputs.size(0)
            val_loss = total / max(1, count)

        ckpt_path = os.path.join(cfg.output_dir, f"gfenet_epoch_{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "config": cfg.__dict__,
            },
            ckpt_path,
        )

        if val_loss is not None and (best_val is None or val_loss < best_val):
            best_val = val_loss
            best_path = os.path.join(cfg.output_dir, "gfenet_best.pt")
            torch.save({"epoch": epoch, "model_state": model.state_dict()}, best_path)

        if val_loss is not None:
            print(f"epoch {epoch}/{cfg.epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        else:
            print(f"epoch {epoch}/{cfg.epochs} train_loss={train_loss:.6f}")


if __name__ == "__main__":
    main()
