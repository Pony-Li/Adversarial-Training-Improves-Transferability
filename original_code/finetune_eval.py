#!/usr/bin/env python
# finetune_eval.py
"""
Downstream transfer evaluation (fixed-feature / full-network) with DDP,
按《Do Adversarially Robust ImageNet Models Transfer Better?》附录 A.2.2 / A.2.3 设定:

- 训练 150 epochs
- SGD, momentum=0.9, weight_decay=5e-4
- 初始 lr ∈ {0.01, 0.001}
- 每 50 epoch 学习率 * 0.1
- 训练增强: RandomResizedCrop(224) + RandomHorizontalFlip
- 测试增强: Resize(256) + CenterCrop(224)
- fixed 模式: 只训练最后一层
- full 模式: 全网络 finetune
- DDP 多卡训练, checkpoint 每 10 个 epoch 保存一次
- 结果写入 summary_*.json
"""

import argparse
import copy
import json
import os
import os.path as osp
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split, DistributedSampler
from torchvision import datasets, transforms, models

# 固定的论文设定
FIXED_LRS = [0.01, 0.001]        # 初始学习率候选
FIXED_EPOCHS = 150               # 总 epoch 数
LR_MILESTONES = [50, 100]        # 每 50 epoch ×0.1
VAL_SPLIT = 0.1                  # 无官方 val 集时, 从 train 随机划出 10% 做 val


# DDP utils
def init_distributed():
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available.")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    return device, rank, world_size, local_rank


def is_main_process(rank: int) -> bool:
    return rank == 0


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


# Normalization wrapper
class NormalizedModel(nn.Module):
    """
    输入假定在 [0,1]，在内部做 ImageNet mean/std 归一化再送入 backbone。
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


# Backbone construction & loading
def build_backbone(arch: str, num_classes: int = 1000) -> nn.Module:
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
    return model


def load_backbone_from_ckpt(arch: str, ckpt_path: str) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    backbone = build_backbone(arch, num_classes=1000)

    if "backbone_state" in ckpt:
        state_dict = ckpt["backbone_state"]
    elif "norm_model_state" in ckpt:
        state_dict = {
            k.replace("model.", ""): v
            for k, v in ckpt["norm_model_state"].items()
            if k.startswith("model.")
        }
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        raise KeyError("No usable backbone weights found in checkpoint.")

    missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Warning] Missing keys when loading backbone: {missing}")
    if unexpected:
        print(f"[Warning] Unexpected keys when loading backbone: {unexpected}")
    return backbone


# Transforms & datasets
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


def get_downstream_dataloaders(
    name: str,
    root: str,
    train_tf,
    test_tf,
    batch_size: int,
    num_workers: int,
    rank: int,
    world_size: int,
):
    """
    返回: train_loader, val_loader, test_loader, num_classes, train_sampler
    """

    name_upper = name.upper()

    def _split_train(ds):
        n_val = int(len(ds) * VAL_SPLIT)
        n_train = len(ds) - n_val
        return random_split(ds, [n_train, n_val])

    if name_upper == "CIFAR10":
        ds_train_full = datasets.CIFAR10(
            root=osp.join(root, "CIFAR10"),
            train=True,
            download=True,
            transform=train_tf,
        )
        ds_test = datasets.CIFAR10(
            root=osp.join(root, "CIFAR10"),
            train=False,
            download=True,
            transform=test_tf,
        )
        num_classes = 10
        ds_train, ds_val = _split_train(ds_train_full)

    elif name_upper == "CIFAR100":
        ds_train_full = datasets.CIFAR100(
            root=osp.join(root, "CIFAR100"),
            train=True,
            download=True,
            transform=train_tf,
        )
        ds_test = datasets.CIFAR100(
            root=osp.join(root, "CIFAR100"),
            train=False,
            download=True,
            transform=test_tf,
        )
        num_classes = 100
        ds_train, ds_val = _split_train(ds_train_full)

    elif name_upper == "CALTECH101":
        ds_full = datasets.Caltech101(
            root=osp.join(root, "Caltech101"),
            download=True,
            transform=train_tf,
        )
        num_classes = len(ds_full.categories)
        ds_train, ds_val = _split_train(ds_full)
        ds_test = ds_val

    elif name_upper == "CALTECH256":
        ds_full = datasets.Caltech256(
            root=osp.join(root, "Caltech256"),
            download=True,
            transform=train_tf,
        )
        num_classes = 257
        ds_train, ds_val = _split_train(ds_full)
        ds_test = ds_val

    elif name_upper == "DTD":
        ds_full = datasets.DTD(
            root=osp.join(root, "DTD"),
            download=True,
            split="train",
            transform=train_tf,
        )
        num_classes = 47
        ds_train, ds_val = _split_train(ds_full)
        ds_test = ds_val

    elif name_upper == "FLOWERS102":
        ds_train = datasets.Flowers102(
            root=osp.join(root, "Flowers102"),
            download=True,
            split="train",
            transform=train_tf,
        )
        ds_val = datasets.Flowers102(
            root=osp.join(root, "Flowers102"),
            download=True,
            split="val",
            transform=test_tf,
        )
        ds_test = datasets.Flowers102(
            root=osp.join(root, "Flowers102"),
            download=True,
            split="test",
            transform=test_tf,
        )
        num_classes = 102

    elif name_upper == "FOOD101":
        ds_train_full = datasets.Food101(
            root=osp.join(root, "Food101"),
            download=True,
            split="train",
            transform=train_tf,
        )
        ds_test = datasets.Food101(
            root=osp.join(root, "Food101"),
            download=True,
            split="test",
            transform=test_tf,
        )
        num_classes = 101
        ds_train, ds_val = _split_train(ds_train_full)

    elif name_upper in ["STANFORDCARS", "CARS"]:
        ds_train_full = datasets.StanfordCars(
            root=osp.join(root, "StanfordCars"),
            download=True,
            split="train",
            transform=train_tf,
        )
        ds_test = datasets.StanfordCars(
            root=osp.join(root, "StanfordCars"),
            download=True,
            split="test",
            transform=test_tf,
        )
        num_classes = 196
        ds_train, ds_val = _split_train(ds_train_full)

    elif name_upper in ["OXFORDIIITPETS", "PETS"]:
        ds_train_full = datasets.OxfordIIITPet(
            root=osp.join(root, "OxfordIIITPet"),
            download=True,
            split="trainval",
            transform=train_tf,
        )
        ds_test = datasets.OxfordIIITPet(
            root=osp.join(root, "OxfordIIITPet"),
            download=True,
            split="test",
            transform=test_tf,
        )
        num_classes = 37
        ds_train, ds_val = _split_train(ds_train_full)

    elif name_upper == "SUN397":
        ds_full = datasets.SUN397(
            root=osp.join(root, "SUN397"),
            download=True,
            transform=train_tf,
        )
        num_classes = len(ds_full.classes)
        ds_train, ds_val = _split_train(ds_full)
        ds_test = ds_val

    else:
        raise ValueError(f"Dataset {name} not implemented.")

    train_sampler = DistributedSampler(
        ds_train, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = DistributedSampler(
        ds_val, num_replicas=world_size, rank=rank, shuffle=False
    )
    test_sampler = DistributedSampler(
        ds_test, num_replicas=world_size, rank=rank, shuffle=False
    )

    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, num_classes, train_sampler


# Build transfer model
def build_transfer_model(backbone: nn.Module, num_classes: int, mode: str) -> nn.Module:
    if not hasattr(backbone, "fc"):
        raise AttributeError("Backbone has no attribute 'fc'.")

    in_features = backbone.fc.in_features
    backbone.fc = nn.Linear(in_features, num_classes)

    if mode == "fixed":
        # 只训练最后一个线性层
        for name, p in backbone.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad = False
    elif mode == "full":
        # 全网络 finetune
        pass
    else:
        raise ValueError(f"Unknown mode: {mode}")

    model = NormalizedModel(backbone)
    return model


# Metrics (DDP aware)
def all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def compute_accuracy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    metric: str,
    class_correct: torch.Tensor,
    class_total: torch.Tensor,
):
    preds = outputs.argmax(dim=1)
    correct = preds.eq(targets)

    if metric == "per_sample":
        return correct.float().sum().item(), targets.numel()

    elif metric == "per_class":
        for c in range(num_classes):
            mask = (targets == c)
            if mask.sum() == 0:
                continue
            class_correct[c] += correct[mask].sum().item()
            class_total[c] += mask.sum().item()
        return correct.float().sum().item(), targets.numel()
    else:
        raise ValueError(f"Unknown metric: {metric}")


def finalize_accuracy(
    num_classes: int,
    metric: str,
    correct_total: float,
    count_total: float,
    class_correct: torch.Tensor,
    class_total: torch.Tensor,
) -> float:
    if metric == "per_sample":
        return correct_total / count_total * 100.0
    else:
        mask = class_total > 0
        per_cls_acc = class_correct[mask] / class_total[mask]
        return per_cls_acc.mean().item() * 100.0


# Train & eval
def train_one_epoch(
    model: DDP,
    loader: DataLoader,
    train_sampler: DistributedSampler,
    optimizer: torch.optim.Optimizer,
    num_classes: int,
    metric: str,
    device: torch.device,
    epoch: int,
    rank: int,
):
    model.train()
    train_sampler.set_epoch(epoch)

    total_loss_local = 0.0
    total_correct_local = 0.0
    total_count_local = 0.0

    class_correct = torch.zeros(num_classes, device=device)
    class_total = torch.zeros(num_classes, device=device)

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            correct_sum, count = compute_accuracy(
                logits, y, num_classes, metric, class_correct, class_total
            )
            total_loss_local += loss.item() * x.size(0)
            total_correct_local += correct_sum
            total_count_local += count

    total_loss = torch.tensor(total_loss_local, device=device)
    total_correct = torch.tensor(total_correct_local, device=device)
    total_count = torch.tensor(total_count_local, device=device)
    all_reduce_sum(total_loss)
    all_reduce_sum(total_correct)
    all_reduce_sum(total_count)

    all_reduce_sum(class_correct)
    all_reduce_sum(class_total)

    avg_loss = (total_loss / total_count).item()
    sample_acc = (total_correct / total_count).item() * 100.0
    metric_acc = finalize_accuracy(
        num_classes, metric,
        total_correct.item(), total_count.item(),
        class_correct, class_total,
    )
    if is_main_process(rank):
        print(
            f"[Train] Epoch {epoch:03d} | "
            f"loss={avg_loss:.4f}, sample_acc={sample_acc:.2f}%, metric_acc={metric_acc:.2f}%"
        )
    return avg_loss, sample_acc, metric_acc


@torch.no_grad()
def evaluate(
    model: DDP,
    loader: DataLoader,
    num_classes: int,
    metric: str,
    device: torch.device,
    tag: str,
    rank: int,
):
    model.eval()

    total_correct_local = 0.0
    total_count_local = 0.0
    class_correct = torch.zeros(num_classes, device=device)
    class_total = torch.zeros(num_classes, device=device)

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)

        correct_sum, count = compute_accuracy(
            logits, y, num_classes, metric, class_correct, class_total
        )
        total_correct_local += correct_sum
        total_count_local += count

    total_correct = torch.tensor(total_correct_local, device=device)
    total_count = torch.tensor(total_count_local, device=device)
    all_reduce_sum(total_correct)
    all_reduce_sum(total_count)
    all_reduce_sum(class_correct)
    all_reduce_sum(class_total)

    sample_acc = (total_correct / total_count).item() * 100.0
    metric_acc = finalize_accuracy(
        num_classes, metric,
        total_correct.item(), total_count.item(),
        class_correct, class_total,
    )

    if is_main_process(rank):
        print(
            f"[{tag}] sample_acc={sample_acc:.2f}%, metric_acc={metric_acc:.2f}%"
        )
    return sample_acc, metric_acc


def run_transfer(
    base_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    train_sampler: DistributedSampler,
    num_classes: int,
    metric: str,
    device: torch.device,
    rank: int,
    save_dir: str,
    dataset: str,
    arch: str,
    mode: str,
):
    os.makedirs(save_dir, exist_ok=True)

    best_val_acc = -1.0
    best_test_acc = -1.0
    best_lr = None

    all_results = []

    for lr in FIXED_LRS:
        if is_main_process(rank):
            print(f"\n=== LR {lr} ===")

        model = copy.deepcopy(base_model).to(device)
        ddp_model = DDP(model, device_ids=[device.index], output_device=device.index)

        params = [p for p in ddp_model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params, lr=lr, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=LR_MILESTONES, gamma=0.1
        )

        best_val_this_lr = -1.0
        best_test_this_lr = -1.0
        best_epoch_this_lr = -1

        for epoch in range(1, FIXED_EPOCHS + 1):
            train_one_epoch(
                ddp_model, train_loader, train_sampler,
                optimizer, num_classes, metric,
                device, epoch, rank,
            )
            scheduler.step()

            val_sample_acc, val_metric_acc = evaluate(
                ddp_model, val_loader, num_classes, metric, device, "Val", rank
            )

            if val_metric_acc > best_val_this_lr:
                best_val_this_lr = val_metric_acc
                best_epoch_this_lr = epoch
                test_sample_acc, test_metric_acc = evaluate(
                    ddp_model, test_loader, num_classes, metric, device, "Test", rank
                )
                best_test_this_lr = test_metric_acc

            # 每 10 个 epoch 存一次当前权重
            if is_main_process(rank) and (epoch % 10 == 0):
                tag = f"{dataset}_{arch}_{mode}_lr{lr:.3g}"
                ckpt_dir = osp.join(save_dir, tag)
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = osp.join(ckpt_dir, f"epoch_{epoch:03d}.pth")
                torch.save(
                    {
                        "arch": arch,
                        "dataset": dataset,
                        "mode": mode,
                        "lr": lr,
                        "epoch": epoch,
                        "model_state": ddp_model.module.state_dict(),
                    },
                    ckpt_path,
                )
                print(f"[Main] Saved checkpoint at epoch {epoch} to {ckpt_path}")

        if is_main_process(rank):
            print(
                f"[LR={lr}] best_val_metric={best_val_this_lr:.2f}%, "
                f"corresponding_test_metric={best_test_this_lr:.2f}% "
                f"(epoch={best_epoch_this_lr})"
            )

        if best_val_this_lr > best_val_acc:
            best_val_acc = best_val_this_lr
            best_test_acc = best_test_this_lr
            best_lr = lr

        all_results.append(
            {
                "lr": lr,
                "best_val_metric": best_val_this_lr,
                "best_test_metric": best_test_this_lr,
                "best_epoch": best_epoch_this_lr,
            }
        )

        barrier()

    if is_main_process(rank):
        summary = {
            "dataset": dataset,
            "arch": arch,
            "mode": mode,
            "metric": metric,
            "fixed_lrs": FIXED_LRS,
            "epochs": FIXED_EPOCHS,
            "best_lr": best_lr,
            "best_val_metric": best_val_acc,
            "best_test_metric": best_test_acc,
            "per_lr_results": all_results,
        }
        json_path = osp.join(
            save_dir,
            f"summary_{dataset}_{arch}_{mode}.json",
        )
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(
            f"\n=== Finished. best_lr={best_lr}, "
            f"best_val_metric={best_val_acc:.2f}%, "
            f"best_test_metric={best_test_acc:.2f}% ==="
        )
        print(f"[Main] Saved summary to {json_path}")

    return best_lr, best_val_acc, best_test_acc


# Args & main
def parse_args():
    parser = argparse.ArgumentParser(
        description="Downstream transfer evaluation (fixed-feature / full-network, paper setting)"
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
                        help="Per-GPU batch size")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader num_workers")
    parser.add_argument("--metric", type=str, default="per_sample",
                        choices=["per_sample", "per_class"],
                        help="Evaluation metric")
    parser.add_argument("--save-dir", type=str, default="./transfer_ckpts",
                        help="Where to save downstream checkpoints and JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    device, rank, world_size, local_rank = init_distributed()

    if is_main_process(rank):
        print("===== Transfer evaluation (paper setting) config =====")
        for k, v in vars(args).items():
            print(f"{k}: {v}")
        print(f"fixed_lrs: {FIXED_LRS}")
        print(f"epochs: {FIXED_EPOCHS}")
        print(f"lr_milestones: {LR_MILESTONES}")
        print("=====================================================")

    train_tf, test_tf = get_transforms()

    train_loader, val_loader, test_loader, num_classes, train_sampler = \
        get_downstream_dataloaders(
            args.dataset,
            args.data_root,
            train_tf,
            test_tf,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            rank=rank,
            world_size=world_size,
        )

    backbone = load_backbone_from_ckpt(args.arch, args.ckpt)
    transfer_model_base = build_transfer_model(
        backbone, num_classes=num_classes, mode=args.mode
    ).to(device)

    best_lr, best_val_acc, best_test_acc = run_transfer(
        transfer_model_base,
        train_loader,
        val_loader,
        test_loader,
        train_sampler,
        num_classes=num_classes,
        metric=args.metric,
        device=device,
        rank=rank,
        save_dir=args.save_dir,
        dataset=args.dataset,
        arch=args.arch,
        mode=args.mode,
    )

    if is_main_process(rank):
        print("\n========== Final Result ==========")
        print(f"Dataset       : {args.dataset}")
        print(f"Backbone      : {args.arch}")
        print(f"Mode          : {args.mode}")
        print(f"Best LR       : {best_lr}")
        print(f"Best Val Acc  : {best_val_acc:.2f}% ({args.metric})")
        print(f"Best Test Acc : {best_test_acc:.2f}% ({args.metric})")
        print("===================================")

    barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
