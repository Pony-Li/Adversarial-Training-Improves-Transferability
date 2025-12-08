"""
Transfer evaluation for ImageNet-pretrained models
(standard / adversarially trained) on downstream datasets
under fixed-feature and full-network settings.

Usage example (CIFAR-10, fixed-feature):

  python transfer_eval.py \
    --data-root /data0/datasets \
    --dataset CIFAR10 \
    --arch resnet34 \
    --ckpt ./ckpts_resnet34_l2_eps0_5/epoch_090_best.pth \
    --mode fixed \
    --batch-size 128 \
    --metric per_sample

Full-network fine-tune:

  python transfer_eval.py \
    --data-root /data0/datasets \
    --dataset CIFAR10 \
    --arch resnet34 \
    --ckpt ./ckpts_resnet34_l2_eps0_5/epoch_090_best.pth \
    --mode full \
    --batch-size 128 \
    --metric per_sample
"""

import argparse
import copy
import os
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Normalization wrapper
class NormalizedModel(nn.Module):
    """
    Wrap backbone with ImageNet normalization.
    Input is assumed to be in [0,1].
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.model(x)


# Backbone construction
def build_backbone(arch: str, num_classes: int = 1000) -> nn.Module:
    """
    Build a torchvision backbone with a fc layer for ImageNet (1000 classes).
    We will replace the fc layer later for downstream tasks.
    """
    if arch == "resnet18":
        model = models.resnet18(weights=None)
    elif arch == "resnet34":
        model = models.resnet34(weights=None)
    elif arch == "resnet50":
        model = models.resnet50(weights=None)
    elif arch == "wide_resnet50_2":
        model = models.wide_resnet50_2(weights=None)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    # 默认是 1000 类，不在此处改 fc；下游任务会替换
    return model


def load_backbone_from_ckpt(arch: str, ckpt_path: str) -> nn.Module:
    """
    Load backbone weights from an ImageNet checkpoint produced by AT.py.

    It expects either:
      - ckpt["backbone_state"] (preferred)
      - or ckpt["state_dict"] / ckpt["norm_model_state"] (fallback)
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    backbone = build_backbone(arch, num_classes=1000)

    if "backbone_state" in ckpt:
        state_dict = ckpt["backbone_state"]
    elif "norm_model_state" in ckpt:
        # norm_model_state includes normalization buffers; extract .model.*
        state_dict = {
            k.replace("model.", ""): v
            for k, v in ckpt["norm_model_state"].items()
            if k.startswith("model.")
        }
    elif "state_dict" in ckpt:
        # 直接来自训练时的模型
        state_dict = ckpt["state_dict"]
    else:
        raise KeyError("No usable backbone weights found in checkpoint.")

    missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Warning] Missing keys when loading backbone: {missing}")
    if unexpected:
        print(f"[Warning] Unexpected keys when loading backbone: {unexpected}")

    return backbone


# Downstream datasets
def get_transforms():
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    return train_tf, test_tf


def get_downstream_dataset(
    name: str,
    root: str,
    train_tf,
    test_tf,
    val_split: float = 0.1,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Return train_loader, val_loader, num_classes for the given downstream dataset.

    Currently implemented with torchvision for:
      - CIFAR10
      - CIFAR100
      - Caltech101
      - Caltech256
      - DTD
      - Flowers102
      - Food101
      - StanfordCars
      - OxfordIIITPet
      - SUN397

    For other datasets (e.g. FGVC Aircraft, Birdsnap), you can arrange
    them as ImageFolder under root/<dataset_name>/{train,val} and adapt here.
    """
    name = name.upper()
    if name == "CIFAR10":
        # train/val split from training set
        ds_train = datasets.CIFAR10(root=os.path.join(root, "CIFAR10"),
                                    train=True, download=False,
                                    transform=train_tf)
        ds_test = datasets.CIFAR10(root=os.path.join(root, "CIFAR10"),
                                   train=False, download=False,
                                   transform=test_tf)
        num_classes = 10

        n_val = int(len(ds_train) * val_split)
        n_train = len(ds_train) - n_val
        ds_train, ds_val = random_split(ds_train, [n_train, n_val])
        # val 使用 test 可能也可以，这里用训练划分出的 val
    elif name == "CIFAR100":
        ds_train = datasets.CIFAR100(root=os.path.join(root, "CIFAR100"),
                                     train=True, download=False,
                                     transform=train_tf)
        ds_test = datasets.CIFAR100(root=os.path.join(root, "CIFAR100"),
                                    train=False, download=False,
                                    transform=test_tf)
        num_classes = 100

        n_val = int(len(ds_train) * val_split)
        n_train = len(ds_train) - n_val
        ds_train, ds_val = random_split(ds_train, [n_train, n_val])

    elif name == "CALTECH101":
        ds = datasets.Caltech101(root=os.path.join(root, "Caltech101"),
                                 download=False, transform=train_tf)
        num_classes = len(ds.categories)
        # 简单 random split：train/val
        n_val = int(len(ds) * val_split)
        n_train = len(ds) - n_val
        ds_train, ds_val = random_split(ds, [n_train, n_val])
        ds_test = ds_val  # 如无单独测试集时，可令 val=test

    elif name == "CALTECH256":
        ds = datasets.Caltech256(root=os.path.join(root, "Caltech256"),
                                 download=False, transform=train_tf)
        num_classes = 257
        n_val = int(len(ds) * val_split)
        n_train = len(ds) - n_val
        ds_train, ds_val = random_split(ds, [n_train, n_val])
        ds_test = ds_val

    elif name == "DTD":
        ds = datasets.DTD(root=os.path.join(root, "DTD"),
                          download=False, split="train",
                          transform=train_tf)
        num_classes = 47
        n_val = int(len(ds) * val_split)
        n_train = len(ds) - n_val
        ds_train, ds_val = random_split(ds, [n_train, n_val])
        ds_test = ds_val

    elif name == "FLOWERS102":
        ds_train = datasets.Flowers102(root=os.path.join(root, "Flowers102"),
                                       download=False, split="train",
                                       transform=train_tf)
        ds_val = datasets.Flowers102(root=os.path.join(root, "Flowers102"),
                                     download=False, split="val",
                                     transform=test_tf)
        ds_test = datasets.Flowers102(root=os.path.join(root, "Flowers102"),
                                      download=False, split="test",
                                      transform=test_tf)
        num_classes = 102

    elif name == "FOOD101":
        ds_train = datasets.Food101(root=os.path.join(root, "Food101"),
                                    download=False, split="train",
                                    transform=train_tf)
        ds_test = datasets.Food101(root=os.path.join(root, "Food101"),
                                   download=False, split="test",
                                   transform=test_tf)
        num_classes = 101

        n_val = int(len(ds_train) * val_split)
        n_train = len(ds_train) - n_val
        ds_train, ds_val = random_split(ds_train, [n_train, n_val])

    elif name == "STANFORDCARS":
        ds_train = datasets.StanfordCars(root=os.path.join(root, "StanfordCars"),
                                         download=False, split="train",
                                         transform=train_tf)
        ds_test = datasets.StanfordCars(root=os.path.join(root, "StanfordCars"),
                                        download=False, split="test",
                                        transform=test_tf)
        num_classes = 196

        n_val = int(len(ds_train) * val_split)
        n_train = len(ds_train) - n_val
        ds_train, ds_val = random_split(ds_train, [n_train, n_val])

    elif name == "OXFORDIIITPETS" or name == "PETS":
        ds_train = datasets.OxfordIIITPet(root=os.path.join(root, "OxfordIIITPet"),
                                          download=False, split="trainval",
                                          transform=train_tf)
        ds_test = datasets.OxfordIIITPet(root=os.path.join(root, "OxfordIIITPet"),
                                         download=False, split="test",
                                         transform=test_tf)
        num_classes = 37

        n_val = int(len(ds_train) * val_split)
        n_train = len(ds_train) - n_val
        ds_train, ds_val = random_split(ds_train, [n_train, n_val])

    elif name == "SUN397":
        ds_train = datasets.SUN397(root=os.path.join(root, "SUN397"),
                                   download=False, transform=train_tf)
        num_classes = len(ds_train.classes)
        n_val = int(len(ds_train) * val_split)
        n_train = len(ds_train) - n_val
        ds_train, ds_val = random_split(ds_train, [n_train, n_val])
        ds_test = ds_val

    else:
        raise ValueError(
            f"Dataset {name} not implemented. "
            f"Please extend get_downstream_dataset() for your use case."
        )

    # DataLoaders
    train_loader = DataLoader(ds_train, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)

    return train_loader, val_loader, test_loader, num_classes


# Build transfer model
def build_transfer_model(
    backbone: nn.Module,
    num_classes: int,
    mode: str,
) -> nn.Module:
    """
    mode:
      - "fixed": freeze backbone, train only last linear layer
      - "full" : fine-tune entire network
    """
    # 假定 backbone 是 ResNet / WideResNet，并且有 .fc 属性
    if not hasattr(backbone, "fc"):
        raise AttributeError("Backbone has no attribute 'fc'.")

    in_features = backbone.fc.in_features
    backbone.fc = nn.Linear(in_features, num_classes)

    if mode == "fixed":
        for name, p in backbone.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad = False
    elif mode == "full":
        # 所有参数都参与训练（默认 requires_grad=True）
        pass
    else:
        raise ValueError(f"Unknown mode: {mode}, expected 'fixed' or 'full'.")

    model = NormalizedModel(backbone)
    return model


# Metrics
def compute_accuracy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    metric: str,
    class_correct: torch.Tensor,
    class_total: torch.Tensor,
):
    """
    Update class_correct/class_total in-place and return batch accuracy.

    metric:
      - "per_sample": standard sample-wise accuracy
      - "per_class": keep per-class counts; final accuracy = mean over classes with >0 samples
    """
    with torch.no_grad():
        preds = outputs.argmax(dim=1)
        correct = preds.eq(targets)

        if metric == "per_sample":
            return correct.float().mean().item()

        elif metric == "per_class":
            for c in range(num_classes):
                mask = (targets == c)
                if mask.sum() == 0:
                    continue
                class_correct[c] += correct[mask].sum().item()
                class_total[c] += mask.sum().item()
            # 返回 sample-wise batch accuracy 供日志参考
            return correct.float().mean().item()
        else:
            raise ValueError(f"Unknown metric: {metric}")


def finalize_accuracy(num_classes: int, metric: str,
                      correct_total: float, count_total: int,
                      class_correct: torch.Tensor,
                      class_total: torch.Tensor) -> float:
    if metric == "per_sample":
        return correct_total / count_total * 100.0
    else:
        # per_class
        mask = class_total > 0
        per_cls_acc = class_correct[mask] / class_total[mask]
        return per_cls_acc.mean().item() * 100.0


# Training & evaluation
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_classes: int,
    metric: str,
) -> float:
    model.train()
    total_loss = 0.0
    total_correct = 0.0
    total_count = 0

    class_correct = torch.zeros(num_classes, device=DEVICE)
    class_total = torch.zeros(num_classes, device=DEVICE)

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            batch_acc = compute_accuracy(
                logits, y, num_classes, metric, class_correct, class_total
            )
            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total_count += x.size(0)

    avg_loss = total_loss / total_count
    sample_acc = total_correct / total_count * 100.0
    metric_acc = finalize_accuracy(
        num_classes, metric, total_correct, total_count,
        class_correct, class_total
    )
    return avg_loss, sample_acc, metric_acc


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    metric: str,
) -> Tuple[float, float]:
    model.eval()
    total_correct = 0.0
    total_count = 0

    class_correct = torch.zeros(num_classes, device=DEVICE)
    class_total = torch.zeros(num_classes, device=DEVICE)

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        logits = model(x)
        batch_acc = compute_accuracy(
            logits, y, num_classes, metric, class_correct, class_total
        )
        total_correct += (logits.argmax(1) == y).sum().item()
        total_count += x.size(0)

    sample_acc = total_correct / total_count * 100.0
    metric_acc = finalize_accuracy(
        num_classes, metric, total_correct, total_count,
        class_correct, class_total
    )
    return sample_acc, metric_acc


def train_with_lr_search(
    base_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    metric: str,
    lrs: List[float],
    epochs: int = 150,
):
    """
    Train the downstream model with multiple learning rates,
    choose the best by validation metric.

    Returns:
      best_lr, best_val_acc, best_test_acc
    """
    best_val_acc = -1.0
    best_test_acc = -1.0
    best_lr = None

    for lr in lrs:
        print(f"\n=== Training with lr={lr} ===")
        model = copy.deepcopy(base_model).to(DEVICE)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params, lr=lr, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[50, 100], gamma=0.1
        )

        best_val_this_lr = -1.0
        best_test_this_lr = -1.0

        for epoch in range(1, epochs + 1):
            train_loss, train_sample_acc, train_metric_acc = train_one_epoch(
                model, train_loader, optimizer, num_classes, metric
            )
            scheduler.step()

            val_sample_acc, val_metric_acc = evaluate_epoch(
                model, val_loader, num_classes, metric
            )
            print(
                f"[lr={lr}] Epoch {epoch:03d} | "
                f"Train loss={train_loss:.4f}, sample_acc={train_sample_acc:.2f}%, metric_acc={train_metric_acc:.2f}% | "
                f"Val sample_acc={val_sample_acc:.2f}%, metric_acc={val_metric_acc:.2f}%"
            )

            # 以 metric_acc（sample 或 per-class）作为选择依据
            if val_metric_acc > best_val_this_lr:
                best_val_this_lr = val_metric_acc
                test_sample_acc, test_metric_acc = evaluate_epoch(
                    model, test_loader, num_classes, metric
                )
                best_test_this_lr = test_metric_acc

        print(
            f"[lr={lr}] Done. Best val metric_acc={best_val_this_lr:.2f}%, "
            f"corresponding test metric_acc={best_test_this_lr:.2f}%"
        )

        if best_val_this_lr > best_val_acc:
            best_val_acc = best_val_this_lr
            best_test_acc = best_test_this_lr
            best_lr = lr

    print(
        f"\n=== LR search finished. Best lr={best_lr}, "
        f"val metric_acc={best_val_acc:.2f}%, "
        f"test metric_acc={best_test_acc:.2f}% ==="
    )
    return best_lr, best_val_acc, best_test_acc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Downstream transfer evaluation (fixed-feature / full-network)"
    )
    parser.add_argument("--data-root", type=str, required=True,
                        help="Root where downstream datasets are stored")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name, e.g. CIFAR10, CIFAR100, Caltech101, ...")
    parser.add_argument("--arch", type=str, default="resnet50",
                        choices=["resnet18", "resnet34", "resnet50", "wide_resnet50_2"],
                        help="Backbone architecture (same as pretraining)")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to pretrained ImageNet checkpoint")
    parser.add_argument("--mode", type=str, default="fixed",
                        choices=["fixed", "full"],
                        help="Transfer mode: fixed (linear) or full (fine-tune)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for downstream training")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader num_workers")
    parser.add_argument("--epochs", type=int, default=150,
                        help="Number of downstream training epochs")
    parser.add_argument("--metric", type=str, default="per_sample",
                        choices=["per_sample", "per_class"],
                        help="Evaluation metric: sample-wise or per-class mean")
    parser.add_argument("--lrs", type=float, nargs="+",
                        default=[0.01, 0.001],
                        help="Learning rate candidates for grid search")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Val split ratio if dataset has no official val set")
    args = parser.parse_args()
    return args


def main():
    global args
    args = parse_args()
    print("===== Transfer evaluation config =====")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("======================================")

    train_tf, test_tf = get_transforms()

    train_loader, val_loader, test_loader, num_classes = get_downstream_dataset(
        args.dataset, args.data_root, train_tf, test_tf, val_split=args.val_split
    )

    backbone = load_backbone_from_ckpt(args.arch, args.ckpt)
    transfer_model_base = build_transfer_model(
        backbone, num_classes=num_classes, mode=args.mode
    )

    transfer_model_base = transfer_model_base.to(DEVICE)

    best_lr, best_val_acc, best_test_acc = train_with_lr_search(
        transfer_model_base,
        train_loader,
        val_loader,
        test_loader,
        num_classes=num_classes,
        metric=args.metric,
        lrs=args.lrs,
        epochs=args.epochs,
    )

    print("\n========== Final Result ==========")
    print(f"Dataset       : {args.dataset}")
    print(f"Backbone      : {args.arch}")
    print(f"Mode          : {args.mode}")
    print(f"Best LR       : {best_lr}")
    print(f"Best Val Acc  : {best_val_acc:.2f}% ({args.metric})")
    print(f"Best Test Acc : {best_test_acc:.2f}% ({args.metric})")
    print("===================================")


if __name__ == "__main__":
    main()
