from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.dataset import DosePairDataset
from training.model import UNet3D


def parse_patch_size(value: str) -> tuple[int, int, int] | None:
    if value.lower() in {"none", "full"}:
        return None
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("patch_size debe ser Z,Y,X (ej: 64,64,64)")
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler=None):
    model.train()
    running = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        inp = batch["input"].to(device)
        tgt = batch["target"].to(device)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                pred = model(inp)
                loss = loss_fn(pred, tgt)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(inp)
            loss = loss_fn(pred, tgt)
            loss.backward()
            optimizer.step()

        running += float(loss.item())
    return running / max(len(loader), 1)


def eval_one_epoch(model, loader, loss_fn, device):
    model.eval()
    running = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="val", leave=False):
            inp = batch["input"].to(device)
            tgt = batch["target"].to(device)
            pred = model(inp)
            loss = loss_fn(pred, tgt)
            running += float(loss.item())
    return running / max(len(loader), 1)


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento denoising 3D (PyTorch)")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/denoising"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--patch-size", type=parse_patch_size, default="64,64,64")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cache-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=5)
    args = parser.parse_args()

    device = get_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = DosePairDataset(
        root_dir=args.data_root,
        split="train",
        patch_size=args.patch_size,
        cache_size=args.cache_size,
        normalize=True,
        seed=1234,
    )
    val_ds = DosePairDataset(
        root_dir=args.data_root,
        split="val",
        patch_size=args.patch_size,
        cache_size=max(1, args.cache_size // 2),
        normalize=True,
        seed=4321,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = UNet3D(in_ch=1, out_ch=1, base_ch=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None

    best_val = float("inf")

    with tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch") as pbar:
        for epoch in pbar:
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler)

            if epoch % args.val_every == 0:
                val_loss = eval_one_epoch(model, val_loader, loss_fn, device)
            else:
                val_loss = float("nan")

            pbar.set_postfix({"train": f"{train_loss:.6f}", "val": f"{val_loss:.6f}", "best": f"{best_val:.6f}"})

            if epoch % args.save_every == 0:
                ckpt = args.output_dir / f"ckpt_epoch_{epoch:03d}.pt"
                torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt)

            if val_loss < best_val:
                best_val = val_loss
                best_path = args.output_dir / "best.pt"
                torch.save({"model": model.state_dict(), "epoch": epoch}, best_path)


if __name__ == "__main__":
    main()
