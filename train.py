from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.datasets import DeepLesionMARDataset
from src.engine import Evaluator, Trainer
from src.models import CAMMARModel
from src.utils import (
    find_resume_checkpoint,
    get_local_rank,
    get_rank,
    get_world_size,
    init_distributed_mode,
    is_main_process,
    load_checkpoint,
    load_config,
    save_json,
    set_seed,
    setup_cudnn,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="CAM/configs/CAM.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--dist_backend", type=str, default="nccl")
    return parser.parse_args()


def build_dataset(dataset_cfg: dict, global_dataset_cfg: dict) -> DeepLesionMARDataset:
    return DeepLesionMARDataset(
        txt_file=dataset_cfg["txt_file"],
        data_root=dataset_cfg["data_root"],
        samples_per_case=dataset_cfg["samples_per_case"],
        intensity_range=tuple(global_dataset_cfg.get("intensity_range", (0.0, 1.0))),
    )


def build_dataloader(dataset, loader_cfg: dict, is_train: bool, is_distributed: bool):
    sampler = DistributedSampler(dataset, shuffle=is_train) if is_distributed else None
    loader = DataLoader(
        dataset,
        batch_size=loader_cfg["batch_size"],
        shuffle=loader_cfg.get("shuffle", is_train) if sampler is None else False,
        sampler=sampler,
        num_workers=loader_cfg.get("num_workers", 4),
        pin_memory=loader_cfg.get("pin_memory", True),
        drop_last=loader_cfg.get("drop_last", is_train),
        persistent_workers=loader_cfg.get("persistent_workers", loader_cfg.get("num_workers", 4) > 0),
    )
    return loader, sampler


def build_model(cfg: dict, device: torch.device):
    model = CAMMARModel(
        generator_cfg=cfg["model"]["generator"],
        discriminator_cfg=cfg["model"]["discriminator"],
        loss_weights=cfg["loss_weights"],
    ).to(device)
    return model


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def maybe_resume(
    model,
    optimizer_g,
    optimizer_d,
    scheduler_g,
    scheduler_d,
    scaler,
    output_dir: Path,
    resume_path: Optional[str],
    auto_resume: bool,
    map_location: str,
):
    checkpoint_path = resume_path
    if checkpoint_path is None and auto_resume:
        checkpoint_path = find_resume_checkpoint(output_dir / "checkpoints")

    if checkpoint_path is None:
        return 1, None

    checkpoint = load_checkpoint(checkpoint_path, map_location=map_location)
    unwrap_model(model).load_state_dict(checkpoint["model"], strict=True)

    if checkpoint.get("optimizer_g") is not None:
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
    if checkpoint.get("optimizer_d") is not None:
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
    if scheduler_g is not None and checkpoint.get("scheduler_g") is not None:
        scheduler_g.load_state_dict(checkpoint["scheduler_g"])
    if scheduler_d is not None and checkpoint.get("scheduler_d") is not None:
        scheduler_d.load_state_dict(checkpoint["scheduler_d"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_metric = checkpoint.get("best_metric")
    return start_epoch, best_metric


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    init_distributed_mode(backend=args.dist_backend)
    local_rank = get_local_rank()
    rank = get_rank()
    world_size = get_world_size()
    is_distributed = dist.is_available() and dist.is_initialized()

    setup_cudnn(cfg.get("cudnn", {}))
    seed = int(cfg.get("seed", 42)) + rank
    set_seed(seed)

    device_name = args.device or cfg.get("device", "cuda")
    if device_name == "cuda":
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    output_dir = Path(cfg["output_dir"]) / cfg["experiment_name"]
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json(cfg, output_dir / "config_resolved.json")

    train_dataset = build_dataset(cfg["dataset"]["train"], cfg["dataset"])
    val_section = cfg["dataset"].get("val", cfg["dataset"].get("test"))
    val_dataset = build_dataset(val_section, cfg["dataset"])

    train_loader, train_sampler = build_dataloader(train_dataset, cfg["train"], is_train=True, is_distributed=is_distributed)
    val_loader, _ = build_dataloader(val_dataset, cfg["val"], is_train=False, is_distributed=False)

    model = build_model(cfg, device)
    optimizer_g = Adam(
        model.generator.parameters(),
        lr=cfg["optimizer"]["lr"],
        betas=(cfg["optimizer"]["beta1"], cfg["optimizer"]["beta2"]),
        weight_decay=cfg["optimizer"].get("weight_decay", 0.0),
    )
    d_params = list(model.discriminator_clean.parameters()) + list(model.discriminator_artifact.parameters())
    optimizer_d = Adam(
        d_params,
        lr=cfg["optimizer"]["lr"],
        betas=(cfg["optimizer"]["beta1"], cfg["optimizer"]["beta2"]),
        weight_decay=cfg["optimizer"].get("weight_decay", 0.0),
    )
    scheduler_g = StepLR(
        optimizer_g,
        step_size=cfg["scheduler"]["step_size"],
        gamma=cfg["scheduler"]["gamma"],
    )
    scheduler_d = StepLR(
        optimizer_d,
        step_size=cfg["scheduler"]["step_size"],
        gamma=cfg["scheduler"]["gamma"],
    )

    if is_distributed and device.type == "cuda":
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    trainer = Trainer(
        model=model,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        scheduler_g=scheduler_g,
        scheduler_d=scheduler_d,
        device=device,
        output_dir=output_dir,
        use_amp=cfg.get("use_amp", False),
        log_interval=cfg.get("train", {}).get("log_interval", 20),
        tensorboard_cfg=cfg.get("tensorboard", {}),
        max_grad_norm=cfg.get("train", {}).get("max_grad_norm"),
        is_main_process=is_main_process(),
    )
    evaluator = Evaluator(
        model=unwrap_model(model).generator,
        device=device,
        save_dir=None,
        use_amp=cfg.get("use_amp", False),
    )

    start_epoch, best_metric = maybe_resume(
        model=model,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        scheduler_g=scheduler_g,
        scheduler_d=scheduler_d,
        scaler=trainer.scaler,
        output_dir=output_dir,
        resume_path=args.resume,
        auto_resume=args.auto_resume or cfg.get("resume", {}).get("auto_resume", False),
        map_location="cpu",
    )
    if best_metric is None:
        best_metric = float("-inf")

    num_epochs = int(cfg["num_epochs"])
    val_metric_name = cfg.get("val", {}).get("best_metric", "psnr")
    val_metric_higher_is_better = bool(cfg.get("val", {}).get("higher_is_better", True))

    for epoch in range(start_epoch, num_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_metrics = trainer.train_one_epoch(train_loader, epoch)
        val_metrics = evaluator.evaluate(val_loader, epoch=epoch, save_predictions=False)

        current_metric = float(val_metrics.get(val_metric_name, 0.0))
        is_better = current_metric > best_metric if val_metric_higher_is_better else current_metric < best_metric
        if is_better:
            best_metric = current_metric

        trainer.save_checkpoint(
            epoch=epoch,
            best_metric=best_metric,
            is_best=is_better,
            save_latest=True,
            save_epoch=cfg.get("checkpoint", {}).get("save_epoch_ckpt", True),
            unwrap_fn=unwrap_model,
        )
        trainer.log_epoch(train_metrics=train_metrics, val_metrics=val_metrics, epoch=epoch, best_metric=best_metric)

        if is_main_process():
            train_str = " | ".join(f"train/{k}: {v:.4f}" for k, v in train_metrics.items())
            val_str = " | ".join(f"val/{k}: {v:.4f}" for k, v in val_metrics.items())
            print(f"[Epoch {epoch:03d}/{num_epochs:03d}] {train_str} | {val_str} | best_{val_metric_name}: {best_metric:.4f}")

    trainer.close()
    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
