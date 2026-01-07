#!/usr/bin/env python
# finetune_eval.py
"""
Downstream transfer evaluation (fixed-feature / full-network) with DDP,
按《Do Adversarially Robust ImageNet Models Transfer Better?》附录 A.2.2 / A.2.3 设定:
"""

import argparse
import copy
import json
import os
import os.path as osp

import torch

from .config import (
    FIXED_LRS, FIXED_EPOCHS, LR_MILESTONES,
)
from .utils import (
    init_distributed, is_main_process, barrier, NormalizedModel, build_backbone,
    load_backbone_from_ckpt, get_transforms, get_downstream_dataloaders,
    build_transfer_model, all_reduce_sum, compute_accuracy, finalize_accuracy
)


def train_one_epoch(
    model: torch.nn.parallel.DistributedDataParallel,
    loader: torch.utils.data.DataLoader,
    train_sampler: torch.utils.data.distributed.DistributedSampler,
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
        loss = torch.nn.functional.cross_entropy(logits, y)

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
    model: torch.nn.parallel.DistributedDataParallel,
    loader: torch.utils.data.DataLoader,
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
    base_model: torch.nn.Module,
    train_loader,
    val_loader,
    test_loader,
    train_sampler,
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
        ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index], output_device=device.index)

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
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()