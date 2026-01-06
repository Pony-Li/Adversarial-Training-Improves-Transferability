#!/usr/bin/env python
"""
Distributed adversarial training on ImageNet with torchattacks (PGD / PGDL2).

Usage (example):
torchrun --nproc_per_node=8 AT.py \
    --data-root /path/to/imagenet \
    --arch resnet50 \
    --norm l2 \
    --epsilon 0.5 \
    --batch-size 512 \
    --num-workers 8 \
    --save-dir ./ckpts/ckpts_resnet50_l2_eps3_0
"""

"""
torchrun = 为分布式训练准备运行环境的进程启动器

torchrun --nproc_per_node=8 AT.py ...
      ↓
torchrun 启 8 个进程
      ↓
给每个进程设置环境变量
      ↓
每个进程运行 AT.py
      ↓
init_process_group 读取环境变量
      ↓
NCCL 建立 GPU 通信
      ↓
DDP 同步梯度

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

    """
    ImageFolder 是最基础、最通用的图像分类数据集封装, 负责三件事:
    1. 按目录结构自动生成标签
    2. 读取图片文件
    3. 应用 transform
    目录结构要求:
    train/
        ├── class_0/
        │    ├── img1.jpg
        │    ├── img2.jpg
        ├── class_1/
        │    ├── img3.jpg
        │    ├── img4.jpg
        |    ...
    val/
        ├── class_0/
        │    ├── img5.jpg
        │    ├── img6.jpg
        ├── class_1/
        │    ├── img7.jpg
        │    ├── img8.jpg
        |    ...
    其中 class_0, class_1 是类别名, 可以是任意字符串
    每个子目录 = 一个类别, 标签由字母顺序自动编号 (class_0 → 0, class_1 → 1)
    __getitem__(idx) 返回: (image_tensor, class_index)
    ImageFolder 不做 batching、不做 shuffle、不做并行, 它只是一个 "样本集合 + 取样规则"
    ImageFolder 并没有加载数据或者读取任何图片, 只是扫描目录、记录文件路径 + label
    ImageFolder 不会做 transform, 只有在 dataset[index] 被调用时, transform 才会被真正执行
    例如, ImageFolder 的真实内部逻辑可能是这样的:
    self.samples = [
    ("/path/img1.jpg", 0),
    ("/path/img2.jpg", 3),
    ...
    ]
    self.transform = train_tf
    """

    train_set = datasets.ImageFolder(os.path.join(data_root, "train"),
                                     transform=train_tf)
    val_set = datasets.ImageFolder(os.path.join(data_root, "val"),
                                   transform=val_tf)
    
    """
    不使用 DDP 时的普通 DataLoader 流程:
    Dataset (indices 0…N-1)
    ↓
    Sampler (Sequential / Random indices)
    ↓
    DataLoader
    ↓
    Batch

    使用 DDP 时的 DataLoader 流程:
    Dataset (indices 0…N-1)
    ↓
    DistributedSampler (按 rank 切分)
    ↓
    DataLoader (每个进程一个)
    ↓
    Batch (不同进程之间互不相同)

    更具体地, DistributedSampler 的作用如下:
    1. 根据 shuffle=True or False 生成一个全局 Random or Sequential 的 indices
    2. 根据 drop_last=True or False 决定对上述 indices 是做 truncate 还是 padding
    3. 所有进程看到同一份经过 shuffle/drop_last=True or False 和 indices
    4. 每个进程根据 world_size 和自己的 rank, 取出对应的子集 indices
    5. DataLoader 根据每个进程自己的子集 indices 取数据, 后续再构建 batch
    """

    # Sampler 不取数据, 不知道 batch_size, 只负责 index 顺序
    train_sampler = distributed.DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False
    )
    val_sampler = distributed.DistributedSampler(
        val_set, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )

    """
    DataLoader + num_workers 在做什么 ?
    1. 从 sampler 拿到一个 index
    2. 用 index 查文件路径
    3. 从磁盘读取 JPEG
    4. CPU 解码 JPEG → PIL / ndarray
    5. 执行 transform (crop / flip / normalize)
    6. 转成 torch.Tensor (CPU)
    7. 返回给主进程
    其中3-6 步可以并行加速 (num_workers > 0)
    另外 DataLoader 还负责把多个样本堆叠成 batch
    以上所有操作都在 CPU 上完成, DataLoader 不会自动搬数据到 GPU
    需要在训练循环里手动把数据搬到 GPU (x = x.cuda())
    这样设计是为了最大化 CPU 和 GPU 的利用率
    让 CPU 在准备数据时, GPU 可以并行计算, 反之亦然
    """

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
    elif arch == "resnet34":
        model = models.resnet34(weights=None)
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
                        choices=["resnet18", "resnet34", "resnet50", "wide_resnet50_2"],
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
    # 初始化进程通信环境, 建立 rank, world_size, 通信 backend (NCCL)
    dist.init_process_group(backend="nccl")
    # 获取每个进程的本地 GPU id, local_rank 是环境变量, 由 torchrun 自动设置, 通过 os.environ 获取
    local_rank = int(os.environ["LOCAL_RANK"])
    # 强制当前进程只使用 local_rank 对应的那一张 GPU
    torch.cuda.set_device(local_rank)
    # 分布式通信系统中的全局编号 (rank) 以及总进程数 (world_size)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    """
    backend = 分布式进程之间如何通信的底层实现方式
    在 PyTorch 分布式里, backend 决定三件事：
    1. 进程如何发现彼此
    2. 张量如何在进程间传输
    3. 集合通信 (reduce / broadcast / gather / scatter 等)怎么实现
    可以把 backend 理解为: DDP 用的通信协议 + 底层传输工具

    backend   适用场景          特点           
    nccl      多 GPU            GPU <--> GPU, 最快 
    gloo      CPU 或少量 GPU    通用，但慢        
    mpi       HPC / MPI 环境    依赖 MPI       

    NCCL (NVIDIA Collective Communications Library) 是 NVIDIA 提供的 GPU 通信库
    专门优化了 GPU 之间的数据传输和集合通信操作, 在多 GPU训练中性能最好。
    但 NCCL 只能在 GPU 上运行, 不支持纯 CPU 环境
    """

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

    # 固定 PGD 参数: steps=3, alpha=2*eps/3
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