import argparse
import os
import os.path as osp
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split, distributed
from torchvision import datasets, transforms, models

from .config import VAL_SPLIT

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


def get_loaders(data_root: str,
                batch_size: int,
                num_workers: int,
                rank: int,
                world_size: int):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),        # 不做标准化, 交给 NormalizedModel
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_set = datasets.ImageFolder(os.path.join(data_root, "train"),
                                     transform=train_tf)
    val_set = datasets.ImageFolder(os.path.join(data_root, "val"),
                                   transform=val_tf)
    
    # Sampler 不取数据, 不知道 batch_size, 只负责 index 顺序
    train_sampler = distributed.DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False
    )
    val_sampler = distributed.DistributedSampler(
        val_set, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_set,
                            batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            pin_memory=True)
    return train_loader, val_loader, train_sampler, val_sampler


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

    train_sampler = distributed.DistributedSampler(
        ds_train, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = distributed.DistributedSampler(
        ds_val, num_replicas=world_size, rank=rank, shuffle=False
    )
    test_sampler = distributed.DistributedSampler(
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