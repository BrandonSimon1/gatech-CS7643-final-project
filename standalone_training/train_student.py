"""
Standalone training for student models on CIFAR-100 (no distillation).

Trains the same student architectures used in PAT, but with only the
ground-truth cross-entropy loss — no teacher, no KD signal.  This
provides the "student-only" baseline for comparison.

Hyperparameters are kept identical to the PAT configs so the only
variable between runs is the presence / absence of distillation.
"""

import argparse
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path

import timm
import timm.utils as timm_utils
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.data.auto_augment import rand_augment_transform
from timm.models.registry import register_model
from timm.models.swin_transformer import _create_swin_transformer
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.utils import ModelEmaV2

# ── Custom model: Swin-Pico (not in stock timm) ────────────────────────


@register_model
def swin_pico_patch4_window7_224(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=4,
        window_size=7,
        embed_dim=48,
        depths=(2, 2, 2, 2),
        num_heads=(2, 4, 8, 16),
        **kwargs,
    )
    return _create_swin_transformer(
        "swin_pico_patch4_window7_224", pretrained=pretrained, **model_kwargs
    )


# ── Preset configs (mirror PAT's YAML files) ───────────────────────────

CNN_DEFAULTS = dict(
    epochs=300,
    batch_size=128,
    lr=0.05,
    min_lr=1e-3,
    opt="sgd",
    weight_decay=2e-3,
    sched="cosine",
    warmup_epochs=3,
    warmup_lr=1e-4,
    color_jitter=0.0,
    smoothing=0.1,
    reprob=0.0,
    aa="",
    mixup=0.0,
    cutmix=0.0,
    clip_grad=0.0,
)

VIT_MLP_DEFAULTS = dict(
    epochs=300,
    batch_size=128,
    lr=5e-4,
    min_lr=1e-5,
    opt="adamw",
    weight_decay=0.05,
    sched="cosine",
    warmup_epochs=20,
    warmup_lr=5e-7,
    color_jitter=0.4,
    smoothing=0.1,
    reprob=0.25,
    aa="rand-m9-mstd0.5-inc1",
    mixup=0.8,
    cutmix=1.0,
    clip_grad=5.0,
)

# Which preset each student uses
MODEL_PRESET = {
    "resnet18": "cnn",
    "mobilenetv2_100": "cnn",
    "deit_tiny_patch16_224": "vit_mlp",
    "swin_pico_patch4_window7_224": "vit_mlp",
    "resmlp_12_224": "vit_mlp",
}

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


# ── Helpers ──────────────────────────────────────────────────────────────


def get_args():
    parser = argparse.ArgumentParser(description="Train a student model on CIFAR-100")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_PRESET.keys()),
        help="Student architecture",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Path to CIFAR-100 data root",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory for checkpoints and logs",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", default=True, help="Use AMP")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP")
    parser.add_argument("--ema", action="store_true", default=True, help="Model EMA")
    parser.add_argument("--ema-decay", type=float, default=0.99996)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--img-size", type=int, default=224)
    return parser.parse_args()


def build_transforms(args, cfg, is_train):
    """Build train / val transforms for CIFAR-100 → 224×224."""
    if is_train:
        t = [transforms.RandomResizedCrop(args.img_size, scale=(0.08, 1.0))]
        t.append(transforms.RandomHorizontalFlip())
        if cfg["color_jitter"] > 0:
            t.append(
                transforms.ColorJitter(
                    brightness=cfg["color_jitter"],
                    contrast=cfg["color_jitter"],
                    saturation=cfg["color_jitter"],
                )
            )
        if cfg["aa"]:
            t.append(rand_augment_transform(cfg["aa"], {}))
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD))
        if cfg["reprob"] > 0:
            t.append(transforms.RandomErasing(p=cfg["reprob"]))
    else:
        t = [
            transforms.Resize(int(args.img_size / 0.875)),  # 256 for 224
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    return transforms.Compose(t)


def build_loaders(args, cfg):
    train_transform = build_transforms(args, cfg, is_train=True)
    val_transform = build_transforms(args, cfg, is_train=False)

    train_dataset = torchvision.datasets.CIFAR100(
        root=args.data_dir, train=True, download=True, transform=train_transform
    )
    val_dataset = torchvision.datasets.CIFAR100(
        root=args.data_dir, train=False, download=True, transform=val_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def build_optimizer(model, cfg):
    if cfg["opt"] == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg["lr"],
            momentum=0.9,
            weight_decay=cfg["weight_decay"],
        )
    elif cfg["opt"] == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg['opt']}")


def build_scheduler(optimizer, cfg, n_iter_per_epoch):
    num_steps = cfg["epochs"] * n_iter_per_epoch
    warmup_steps = cfg["warmup_epochs"] * n_iter_per_epoch

    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_steps,
        lr_min=cfg["min_lr"],
        warmup_lr_init=cfg["warmup_lr"],
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
    )
    return scheduler


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


# ── Training / Evaluation ────────────────────────────────────────────────


def train_one_epoch(
    model, loader, criterion, optimizer, scheduler, scaler, mixup_fn, cfg, epoch, device,
    model_ema=None,
):
    model.train()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    num_steps = len(loader)
    start = time.time()

    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(images)
            loss = criterion(logits, targets)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if cfg["clip_grad"] > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad"])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg["clip_grad"] > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad"])
            optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        global_step = epoch * num_steps + step
        scheduler.step_update(global_step)

        loss_meter.update(loss.item(), images.size(0))

        # accuracy only meaningful without mixup
        if mixup_fn is None:
            acc1, = accuracy(logits, targets, topk=(1,))
            acc1_meter.update(acc1, images.size(0))

    elapsed = time.time() - start
    return loss_meter.avg, acc1_meter.avg, elapsed


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    criterion = nn.CrossEntropyLoss()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)
        acc1, acc5 = accuracy(logits, targets, topk=(1, 5))

        loss_meter.update(loss.item(), images.size(0))
        acc1_meter.update(acc1, images.size(0))
        acc5_meter.update(acc5, images.size(0))

    return loss_meter.avg, acc1_meter.avg, acc5_meter.avg


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    args = get_args()
    if args.no_amp:
        args.amp = False

    # Select preset and apply any CLI overrides
    preset = MODEL_PRESET[args.model]
    cfg = dict(CNN_DEFAULTS if preset == "cnn" else VIT_MLP_DEFAULTS)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["lr"] = args.lr

    # Output directory
    run_name = f"{args.model}_standalone"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model:  {args.model}  (preset: {preset})")
    print(f"Config: {json.dumps(cfg, indent=2)}")

    # Data
    train_loader, val_loader = build_loaders(args, cfg)
    print(f"Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples")

    # Model
    model = timm.create_model(args.model, pretrained=False, num_classes=100)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params / 1e6:.2f}M")

    # EMA
    model_ema = None
    if args.ema:
        model_ema = ModelEmaV2(model, decay=args.ema_decay)

    # Mixup / CutMix
    mixup_fn = None
    if cfg["mixup"] > 0 or cfg["cutmix"] > 0:
        mixup_fn = Mixup(
            mixup_alpha=cfg["mixup"],
            cutmix_alpha=cfg["cutmix"],
            prob=1.0,
            switch_prob=0.5,
            mode="batch",
            label_smoothing=cfg["smoothing"],
            num_classes=100,
        )

    # Loss
    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif cfg["smoothing"] > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=cfg["smoothing"])
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer & scheduler
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None

    # Resume
    start_epoch = 0
    best_acc1 = 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_acc1 = ckpt.get("best_acc1", 0.0)
        if model_ema is not None and "model_ema" in ckpt:
            model_ema.module.load_state_dict(ckpt["model_ema"])
        if scaler is not None and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        print(f"Resumed from epoch {start_epoch}, best_acc1={best_acc1:.2f}")

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump({"model": args.model, "preset": preset, **cfg, "seed": args.seed}, f, indent=2)

    # Training loop
    log_path = output_dir / "log.txt"
    print(f"\nTraining for {cfg['epochs']} epochs. Logs → {log_path}\n")

    for epoch in range(start_epoch, cfg["epochs"]):
        train_loss, train_acc1, elapsed = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, mixup_fn, cfg, epoch, device,
            model_ema=model_ema,
        )
        scheduler.step(epoch + 1)

        # Evaluate the main model
        val_loss, val_acc1, val_acc5 = evaluate(model, val_loader, device)

        # Evaluate EMA model
        ema_acc1 = 0.0
        if model_ema is not None:
            _, ema_acc1, _ = evaluate(model_ema.module, val_loader, device)

        best_val = max(val_acc1, ema_acc1)
        is_best = best_val > best_acc1
        if is_best:
            best_acc1 = best_val

        # Log
        lr_now = optimizer.param_groups[0]["lr"]
        log_line = (
            f"Epoch {epoch:3d}/{cfg['epochs']}  "
            f"lr={lr_now:.6f}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_acc1={val_acc1:.2f}  val_acc5={val_acc5:.2f}  "
            f"ema_acc1={ema_acc1:.2f}  "
            f"best={best_acc1:.2f}  "
            f"time={elapsed:.1f}s"
        )
        print(log_line)
        with open(log_path, "a") as f:
            f.write(log_line + "\n")

        # Checkpoint
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc1": best_acc1,
        }
        if model_ema is not None:
            state["model_ema"] = model_ema.module.state_dict()
        if scaler is not None:
            state["scaler"] = scaler.state_dict()

        torch.save(state, output_dir / "last.pth")
        if is_best:
            torch.save(state, output_dir / "best.pth")

    print(f"\nDone. Best accuracy: {best_acc1:.2f}%")
    print(f"Checkpoints saved to {output_dir}")


if __name__ == "__main__":
    main()
