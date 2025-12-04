#!/usr/bin/env python
"""
Distributed adversarial training on ImageNet with torchattacks (PGD / PGDL2).

Usage (example):
torchrun --nproc_per_node=8 AT.py \
    --data-root /path/to/imagenet \
    --arch resnet50 \
    --norm l2 \
    --epsilon 0.5 \
    --save-dir ./ckpts_resnet50_l2_eps0_5
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, distributed
from torchvision import datasets, transforms, models

import torchattacks


# 一些固定的训练超参数 (遵循原按论文设定)
NUM_EPOCHS = 90
INIT_LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
MILESTONES = [30, 60]
GAMMA = 0.1
PGD_STEPS = 3  # 训练时 PGD 迭代次数


# Normalization wrapper
class NormalizedModel(nn.Module):
    """
    把 ImageNet mean/std 集成到模型内部, 保证输入保持在 [0,1], 
    满足 torchattacks 的要求。
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


# 数据、模型
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

    train_sampler = distributed.DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False
    )
    # 验证集可以用 DistributedSampler 也可以只在 rank0 上 eval；
    # 这里用 DistributedSampler, 所有 rank 共同算, 然后聚合。
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


def build_backbone(arch: str) -> nn.Module:
    if arch == "resnet18":
        model = models.resnet18(weights=None)
    elif arch == "resnet50":
        model = models.resnet50(weights=None)
    elif arch == "wide_resnet50_2":
        model = models.wide_resnet50_2(weights=None)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return model


# 训练 / 验证
def train_one_epoch(model_ddp: DDP,
                    atk,
                    train_loader,
                    optimizer,
                    epoch: int,
                    rank: int):
    model_ddp.train()

    total_loss = 0.0
    total_correct = 0
    total_num = 0

    start = time.time()
    for step, (x, y) in enumerate(train_loader):
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        if atk is not None:
            x_adv = atk(x, y)  # torchattacks 会在内部做 PGD
        else:
            x_adv = x

        logits = model_ddp(x_adv)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            correct = (pred == y).sum().item()
            total_loss += loss.item() * x.size(0)
            total_correct += correct
            total_num += x.size(0)

        if rank == 0 and (step + 1) % 100 == 0:
            avg_loss = total_loss / total_num
            avg_acc = total_correct / total_num * 100.0
            elapsed = time.time() - start
            print(f"[Epoch {epoch:03d}][Iter {step+1:05d}/{len(train_loader):05d}] "
                  f"Loss {avg_loss:.4f} Acc {avg_acc:.2f}% Time {elapsed:.1f}s")
            start = time.time()


@torch.no_grad()
def evaluate(model_ddp: DDP, val_loader, rank: int, world_size: int) -> float:
    model_ddp.eval()
    correct = torch.tensor(0.0).cuda()
    total = torch.tensor(0.0).cuda()

    for x, y in val_loader:
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        logits = model_ddp(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum()
        total += y.size(0)

    # All-reduce 求和
    dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)

    acc = (correct / total * 100.0).item()
    if rank == 0:
        print(f"Validation Acc: {acc:.2f}%")
    return acc


# 主函数
def main():
    parser = argparse.ArgumentParser(
        description="ImageNet adversarial training with torchattacks + DDP"
    )
    parser.add_argument("--data-root", type=str, required=True,
                        help="ImageNet 根目录, 下有 train/val")
    parser.add_argument("--arch", type=str, default="resnet50",
                        choices=["resnet18", "resnet50", "wide_resnet50_2"],
                        help="backbone 网络结构")
    parser.add_argument("--norm", type=str, default="l2",
                        choices=["l2", "linf"],
                        help="PGD 范数类型")
    parser.add_argument("--epsilon", type=float, required=True,
                        help="扰动半径 (像素 0-1 标度, 如 4/255≈0.015686) ")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="每卡 batch size (总 batch=world_size*batch_size) ")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="DataLoader num_workers")
    parser.add_argument("--save-dir", type=str, default="./checkpoints",
                        help="checkpoint 保存目录")
    parser.add_argument("--resume", type=str, default="",
                        help="可选：从已有 checkpoint 恢复训练")
    args = parser.parse_args()

    # DDP 初始化
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        print("===== Training config =====")
        for k, v in vars(args).items():
            print(f"{k}: {v}")
        print(f"world_size: {world_size}")
        print("===========================")

    # 数据
    train_loader, val_loader, train_sampler, val_sampler = get_loaders(
        args.data_root, args.batch_size, args.num_workers, rank, world_size
    )

    # 模型 + Normalization
    backbone = build_backbone(args.arch)
    norm_model = NormalizedModel(backbone).cuda()
    model_ddp = DDP(norm_model, device_ids=[local_rank], output_device=local_rank)

    # 优化器 / scheduler (论文设定) 
    optimizer = torch.optim.SGD(
        model_ddp.parameters(),
        lr=INIT_LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=MILESTONES, gamma=GAMMA
    )

    start_epoch = 1
    best_acc = 0.0

    # resume
    if args.resume:
        map_loc = {"cuda:%d" % 0: "cuda:%d" % local_rank}
        ckpt = torch.load(args.resume, map_location=map_loc)
        model_ddp.module.load_state_dict(ckpt["norm_model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best_acc", 0.0)
        if rank == 0:
            print(f"Resumed from {args.resume}, epoch {ckpt['epoch']}, best_acc={best_acc:.2f}")

    # 固定 PGD 参数：steps=3, alpha=2*eps/3
    alpha = 2.0 * args.epsilon / PGD_STEPS
    if args.epsilon > 0.0:
        if args.norm == "linf":
            atk = torchattacks.PGD(
                model_ddp, eps=args.epsilon, alpha=alpha,
                steps=PGD_STEPS, random_start=True
            )
        else:  # l2
            atk = torchattacks.PGDL2(
                model_ddp, eps=args.epsilon, alpha=alpha,
                steps=PGD_STEPS, random_start=True
            )
    else:
        atk = None
        if rank == 0:
            print("epsilon=0.0 -> 标准 ERM 训练, 无对抗样本。")

    # 训练主循环
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        # DDP sampler 设定 epoch, 保证 shuffle 不同
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\n===== Epoch {epoch}/{NUM_EPOCHS} (lr={scheduler.get_last_lr()[0]:.5f}) =====")

        train_one_epoch(model_ddp, atk, train_loader, optimizer, epoch, rank)
        scheduler.step()

        # 验证
        acc = evaluate(model_ddp, val_loader, rank, world_size)

        # rank0 保存 checkpoint
        if rank == 0:
            is_best = acc > best_acc
            if is_best:
                best_acc = acc

            if epoch % 10 == 0 or epoch == NUM_EPOCHS:
                ckpt = {
                    "epoch": epoch,
                    "arch": args.arch,
                    "norm": args.norm,
                    "epsilon": args.epsilon,
                    "world_size": world_size,
                    "best_acc": best_acc,
                    # 只存 normalized model 的 state_dict 即可；需要 bare backbone 时再取 .model
                    "norm_model_state": model_ddp.module.state_dict(),
                    "backbone_state": model_ddp.module.model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "args": vars(args),
                }
                ckpt_name = f"epoch_{epoch:03d}.pth"
                save_path = os.path.join(args.save_dir, ckpt_name)
                torch.save(ckpt, save_path)
                print(f"[Rank0] Saved checkpoint to {save_path} (best_acc={best_acc:.2f})")

    if rank == 0:
        print(f"Training finished. Best val acc={best_acc:.2f}%")


if __name__ == "__main__":
    main()