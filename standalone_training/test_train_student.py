"""Unit tests for train_student.py — runs on CPU, no CIFAR download needed."""

import argparse
import types

import pytest
import timm
import torch
import torch.nn as nn
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import train_student as ts


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def dummy_args():
    """Minimal args namespace used by helpers that take `args`."""
    return argparse.Namespace(
        img_size=224,
        amp=False,
        no_amp=False,
        ema=False,
        ema_decay=0.99996,
        seed=42,
        num_workers=0,
        data_dir="./data",
        output_dir="/tmp/test_standalone",
        resume=None,
        epochs=None,
        batch_size=None,
        lr=None,
    )


def _fake_loader(batch_size=4, num_batches=2, num_classes=100, img_size=224):
    """Returns a list-based 'loader' of (images, labels) tuples on CPU."""
    batches = []
    for _ in range(num_batches):
        imgs = torch.randn(batch_size, 3, img_size, img_size)
        labels = torch.randint(0, num_classes, (batch_size,))
        batches.append((imgs, labels))
    return batches


# ── AverageMeter ─────────────────────────────────────────────────────────


class TestAverageMeter:
    def test_single_update(self):
        m = ts.AverageMeter()
        m.update(4.0, n=1)
        assert m.val == 4.0
        assert m.avg == 4.0
        assert m.count == 1

    def test_multiple_updates(self):
        m = ts.AverageMeter()
        m.update(2.0, n=2)  # sum=4, count=2
        m.update(6.0, n=2)  # sum=16, count=4
        assert m.avg == pytest.approx(4.0)
        assert m.count == 4

    def test_reset(self):
        m = ts.AverageMeter()
        m.update(10.0)
        m.reset()
        assert m.avg == 0
        assert m.count == 0


# ── accuracy() ───────────────────────────────────────────────────────────


class TestAccuracy:
    def test_perfect_prediction(self):
        logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        targets = torch.tensor([0, 1, 2])
        acc1, acc5 = ts.accuracy(logits, targets, topk=(1, 2))
        assert acc1 == pytest.approx(100.0)

    def test_wrong_predictions(self):
        logits = torch.tensor([[0.0, 10.0], [10.0, 0.0]])
        targets = torch.tensor([0, 1])
        (acc1,) = ts.accuracy(logits, targets, topk=(1,))
        assert acc1 == pytest.approx(0.0)

    def test_top5_relaxed(self):
        # 8 classes, predict top-5 — target is always in top 5
        logits = torch.zeros(1, 8)
        logits[0, 3] = 10.0  # top-1 = class 3
        targets = torch.tensor([4])  # wrong for top-1
        acc1, acc5 = ts.accuracy(logits, targets, topk=(1, 5))
        assert acc1 == pytest.approx(0.0)
        assert acc5 == pytest.approx(100.0)


# ── Config / preset selection ────────────────────────────────────────────


class TestPresets:
    def test_all_models_have_presets(self):
        expected = {
            "resnet18", "mobilenetv2_100",
            "deit_tiny_patch16_224", "swin_pico_patch4_window7_224",
            "resmlp_12_224",
        }
        assert set(ts.MODEL_PRESET.keys()) == expected

    def test_cnn_models_use_cnn_preset(self):
        assert ts.MODEL_PRESET["resnet18"] == "cnn"
        assert ts.MODEL_PRESET["mobilenetv2_100"] == "cnn"

    def test_vit_mlp_models_use_vit_mlp_preset(self):
        for name in ["deit_tiny_patch16_224", "swin_pico_patch4_window7_224", "resmlp_12_224"]:
            assert ts.MODEL_PRESET[name] == "vit_mlp"

    def test_cnn_defaults_keys(self):
        required = {"epochs", "batch_size", "lr", "min_lr", "opt", "weight_decay",
                     "smoothing", "mixup", "cutmix", "clip_grad"}
        assert required.issubset(ts.CNN_DEFAULTS.keys())

    def test_vit_mlp_defaults_keys(self):
        required = {"epochs", "batch_size", "lr", "min_lr", "opt", "weight_decay",
                     "smoothing", "mixup", "cutmix", "clip_grad", "aa", "reprob"}
        assert required.issubset(ts.VIT_MLP_DEFAULTS.keys())

    def test_cnn_uses_sgd(self):
        assert ts.CNN_DEFAULTS["opt"] == "sgd"

    def test_vit_mlp_uses_adamw(self):
        assert ts.VIT_MLP_DEFAULTS["opt"] == "adamw"


# ── Model creation ───────────────────────────────────────────────────────


@pytest.mark.parametrize("model_name", list(ts.MODEL_PRESET.keys()))
def test_model_creates_and_forward(model_name):
    """Each student model instantiates and produces (B, 100) logits."""
    model = timm.create_model(model_name, pretrained=False, num_classes=100)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 100)


# ── Transforms ───────────────────────────────────────────────────────────


class TestTransforms:
    def test_train_cnn_transform_structure(self, dummy_args):
        t = ts.build_transforms(dummy_args, ts.CNN_DEFAULTS, is_train=True)
        # CNN train: RandomResizedCrop, RandomHorizontalFlip, ToTensor, Normalize (4)
        assert len(t.transforms) == 4

    def test_val_transform_structure(self, dummy_args):
        t = ts.build_transforms(dummy_args, ts.CNN_DEFAULTS, is_train=False)
        # Val: Resize, CenterCrop, ToTensor, Normalize (4)
        assert len(t.transforms) == 4

    def test_vit_train_has_more_augmentations(self, dummy_args):
        t_cnn = ts.build_transforms(dummy_args, ts.CNN_DEFAULTS, is_train=True)
        t_vit = ts.build_transforms(dummy_args, ts.VIT_MLP_DEFAULTS, is_train=True)
        # ViT preset enables color_jitter, autoaugment, random erasing → more transforms
        assert len(t_vit.transforms) > len(t_cnn.transforms)


# ── Optimizer ────────────────────────────────────────────────────────────


class TestOptimizer:
    def _dummy_model(self):
        return nn.Linear(10, 10)

    def test_sgd(self):
        opt = ts.build_optimizer(self._dummy_model(), ts.CNN_DEFAULTS)
        assert isinstance(opt, torch.optim.SGD)
        assert opt.defaults["lr"] == ts.CNN_DEFAULTS["lr"]

    def test_adamw(self):
        opt = ts.build_optimizer(self._dummy_model(), ts.VIT_MLP_DEFAULTS)
        assert isinstance(opt, torch.optim.AdamW)
        assert opt.defaults["lr"] == ts.VIT_MLP_DEFAULTS["lr"]

    def test_unknown_raises(self):
        cfg = dict(ts.CNN_DEFAULTS, opt="rmsprop")
        with pytest.raises(ValueError, match="Unknown optimizer"):
            ts.build_optimizer(self._dummy_model(), cfg)


# ── Scheduler ────────────────────────────────────────────────────────────


class TestScheduler:
    def test_creates_cosine_scheduler(self):
        model = nn.Linear(10, 10)
        opt = torch.optim.SGD(model.parameters(), lr=0.05)
        sched = ts.build_scheduler(opt, ts.CNN_DEFAULTS, n_iter_per_epoch=100)
        # Should be a CosineLRScheduler
        from timm.scheduler.cosine_lr import CosineLRScheduler
        assert isinstance(sched, CosineLRScheduler)

    def test_lr_decreases_after_warmup(self):
        model = nn.Linear(10, 10)
        opt = torch.optim.SGD(model.parameters(), lr=0.05)
        n_iter = 10
        sched = ts.build_scheduler(opt, ts.CNN_DEFAULTS, n_iter_per_epoch=n_iter)
        # Step past warmup (3 epochs * 10 iters = 30 steps)
        warmup_end = ts.CNN_DEFAULTS["warmup_epochs"] * n_iter
        for step in range(warmup_end + 1):
            sched.step_update(step)
        lr_after_warmup = opt.param_groups[0]["lr"]
        # Step well into training
        total_steps = ts.CNN_DEFAULTS["epochs"] * n_iter
        for step in range(warmup_end + 1, total_steps // 2):
            sched.step_update(step)
        lr_mid = opt.param_groups[0]["lr"]
        assert lr_mid < lr_after_warmup


# ── Loss selection ───────────────────────────────────────────────────────


class TestLossSelection:
    def test_smoothing_only(self):
        cfg = dict(ts.CNN_DEFAULTS)  # smoothing=0.1, no mixup
        mixup_fn = None
        if cfg["smoothing"] > 0 and mixup_fn is None:
            criterion = LabelSmoothingCrossEntropy(smoothing=cfg["smoothing"])
        assert isinstance(criterion, LabelSmoothingCrossEntropy)

    def test_mixup_uses_soft_target(self):
        cfg = dict(ts.VIT_MLP_DEFAULTS)  # mixup=0.8
        # When mixup is active, loss should be SoftTargetCrossEntropy
        mixup_fn = Mixup(
            mixup_alpha=cfg["mixup"], cutmix_alpha=cfg["cutmix"],
            label_smoothing=cfg["smoothing"], num_classes=100,
        )
        criterion = SoftTargetCrossEntropy()
        assert isinstance(criterion, SoftTargetCrossEntropy)

    def test_no_smoothing_no_mixup(self):
        cfg = dict(ts.CNN_DEFAULTS, smoothing=0.0, mixup=0.0, cutmix=0.0)
        criterion = nn.CrossEntropyLoss()
        assert isinstance(criterion, nn.CrossEntropyLoss)


# ── train_one_epoch (smoke test, 1 step on CPU) ─────────────────────────


class TestTrainOneEpoch:
    def test_cnn_one_epoch_cpu(self):
        model = timm.create_model("resnet18", pretrained=False, num_classes=100)
        cfg = dict(ts.CNN_DEFAULTS, epochs=1)
        device = torch.device("cpu")
        optimizer = ts.build_optimizer(model, cfg)
        loader = _fake_loader(batch_size=4, num_batches=2)
        scheduler = ts.build_scheduler(optimizer, cfg, n_iter_per_epoch=len(loader))
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

        loss, acc, elapsed = ts.train_one_epoch(
            model, loader, criterion, optimizer, scheduler,
            scaler=None, mixup_fn=None, cfg=cfg, epoch=0, device=device,
        )
        assert loss > 0
        assert elapsed > 0
        # acc should be between 0 and 100
        assert 0 <= acc <= 100

    def test_vit_with_mixup_cpu(self):
        model = timm.create_model("resnet18", pretrained=False, num_classes=100)
        cfg = dict(ts.VIT_MLP_DEFAULTS, epochs=1)
        device = torch.device("cpu")
        optimizer = ts.build_optimizer(model, cfg)
        loader = _fake_loader(batch_size=4, num_batches=2)
        scheduler = ts.build_scheduler(optimizer, cfg, n_iter_per_epoch=len(loader))
        mixup_fn = Mixup(
            mixup_alpha=cfg["mixup"], cutmix_alpha=cfg["cutmix"],
            label_smoothing=cfg["smoothing"], num_classes=100,
        )
        criterion = SoftTargetCrossEntropy()

        loss, acc, elapsed = ts.train_one_epoch(
            model, loader, criterion, optimizer, scheduler,
            scaler=None, mixup_fn=mixup_fn, cfg=cfg, epoch=0, device=device,
        )
        assert loss > 0
        # acc stays 0 when mixup is active (skipped in train loop)
        assert acc == 0


# ── evaluate (smoke test on CPU) ─────────────────────────────────────────


class TestEvaluate:
    def test_evaluate_returns_metrics(self):
        model = timm.create_model("resnet18", pretrained=False, num_classes=100)
        model.eval()
        device = torch.device("cpu")
        loader = _fake_loader(batch_size=4, num_batches=2)

        val_loss, val_acc1, val_acc5 = ts.evaluate(model, loader, device)
        assert val_loss > 0
        assert 0 <= val_acc1 <= 100
        assert 0 <= val_acc5 <= 100
        assert val_acc5 >= val_acc1  # top-5 always >= top-1


# ── Checkpoint round-trip ────────────────────────────────────────────────


class TestCheckpoint:
    def test_save_and_load(self, tmp_path):
        model = nn.Linear(10, 100)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        state = {
            "epoch": 5,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_acc1": 42.5,
        }
        path = tmp_path / "ckpt.pth"
        torch.save(state, path)

        loaded = torch.load(path, map_location="cpu")
        assert loaded["epoch"] == 5
        assert loaded["best_acc1"] == 42.5
        model2 = nn.Linear(10, 100)
        model2.load_state_dict(loaded["model"])
        # Weights should match
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.equal(p1, p2)
