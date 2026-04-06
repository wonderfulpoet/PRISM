from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.utils import MetricLogger, save_checkpoint


class Trainer:
    def __init__(
        self,
        model,
        optimizer_g,
        optimizer_d,
        scheduler_g,
        scheduler_d,
        device,
        output_dir: Path,
        use_amp: bool = False,
        log_interval: int = 20,
        tensorboard_cfg: dict | None = None,
        max_grad_norm: float | None = None,
        is_main_process: bool = True,
    ):
        self.model = model
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.scheduler_g = scheduler_g
        self.scheduler_d = scheduler_d
        self.device = device
        self.output_dir = Path(output_dir)
        self.use_amp = use_amp
        self.log_interval = log_interval
        self.max_grad_norm = max_grad_norm
        self.is_main_process = is_main_process
        self.scaler = GradScaler(enabled=use_amp)

        tb_cfg = tensorboard_cfg or {}
        self.writer = None
        if self.is_main_process and tb_cfg.get("enabled", True):
            log_dir = self.output_dir / tb_cfg.get("log_dir", "tensorboard")
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_dir))

    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        moved = {}
        for key, value in batch.items():
            moved[key] = value.to(self.device, non_blocking=True) if torch.is_tensor(value) else value
        return moved

    def _maybe_clip_gradients(self) -> None:
        if self.max_grad_norm is None:
            return
        self.scaler.unscale_(self.optimizer_g)
        self.scaler.unscale_(self.optimizer_d)
        clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

    def train_one_epoch(self, loader, epoch: int) -> Dict[str, float]:
        self.model.train()
        logger = MetricLogger()

        progress = tqdm(loader, desc=f"Epoch {epoch}", leave=False, disable=not self.is_main_process)
        for step, batch in enumerate(progress, start=1):
            batch = self._move_batch_to_device(batch)

            self.optimizer_g.zero_grad(set_to_none=True)
            with autocast(enabled=self.use_amp):
                g_total, g_losses, outputs = self.model.module.compute_generator_loss(
                    artifact_ct=batch["artifact_ct"],
                    clean_ct_pair=batch["clean_ct_pair"],
                    clean_ct=batch["clean_ct"],
                    li_ct=batch["li_ct"],
                ) if hasattr(self.model, "module") else self.model.compute_generator_loss(
                    artifact_ct=batch["artifact_ct"],
                    clean_ct_pair=batch["clean_ct_pair"],
                    clean_ct=batch["clean_ct"],
                    li_ct=batch["li_ct"],
                )
            self.scaler.scale(g_total).backward()
            self.scaler.step(self.optimizer_g)
            self.scaler.update()

            self.optimizer_d.zero_grad(set_to_none=True)
            with autocast(enabled=self.use_amp):
                d_total, d_losses = self.model.module.compute_discriminator_loss(
                    outputs=outputs,
                    artifact_ct=batch["artifact_ct"],
                    clean_ct=batch["clean_ct"],
                ) if hasattr(self.model, "module") else self.model.compute_discriminator_loss(
                    outputs=outputs,
                    artifact_ct=batch["artifact_ct"],
                    clean_ct=batch["clean_ct"],
                )
            self.scaler.scale(d_total).backward()
            self._maybe_clip_gradients()
            self.scaler.step(self.optimizer_d)
            self.scaler.update()

            metrics = {"g_total": float(g_total.detach().item()), "d_total": float(d_total.detach().item())}
            metrics.update({k: float(v.detach().item()) for k, v in g_losses.items()})
            metrics.update({k: float(v.detach().item()) for k, v in d_losses.items()})
            logger.update(metrics, n=batch["artifact_ct"].shape[0])

            if step % self.log_interval == 0:
                display_metrics = logger.averages()
                progress.set_postfix({k: f"{v:.4f}" for k, v in display_metrics.items() if k in ["g_total", "d_total"]})

        if self.scheduler_g is not None:
            self.scheduler_g.step()
        if self.scheduler_d is not None:
            self.scheduler_d.step()
        return logger.averages()

    def save_checkpoint(
        self,
        epoch: int,
        best_metric: float,
        is_best: bool,
        save_latest: bool,
        save_epoch: bool,
        unwrap_fn,
    ) -> None:
        if not self.is_main_process:
            return

        checkpoint = {
            "epoch": epoch,
            "best_metric": best_metric,
            "model": unwrap_fn(self.model).state_dict(),
            "optimizer_g": self.optimizer_g.state_dict(),
            "optimizer_d": self.optimizer_d.state_dict(),
            "scheduler_g": self.scheduler_g.state_dict() if self.scheduler_g else None,
            "scheduler_d": self.scheduler_d.state_dict() if self.scheduler_d else None,
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
        }
        checkpoint_dir = self.output_dir / "checkpoints"
        if save_latest:
            save_checkpoint(checkpoint, checkpoint_dir / "latest.pth")
        if save_epoch:
            save_checkpoint(checkpoint, checkpoint_dir / f"epoch_{epoch:03d}.pth")
        if is_best:
            save_checkpoint(checkpoint, checkpoint_dir / "best.pth")

    def log_epoch(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float], epoch: int, best_metric: float) -> None:
        if self.writer is None:
            return
        for key, value in train_metrics.items():
            self.writer.add_scalar(f"train/{key}", value, epoch)
        for key, value in val_metrics.items():
            self.writer.add_scalar(f"val/{key}", value, epoch)
        current_lr_g = self.optimizer_g.param_groups[0]["lr"]
        current_lr_d = self.optimizer_d.param_groups[0]["lr"]
        self.writer.add_scalar("lr/generator", current_lr_g, epoch)
        self.writer.add_scalar("lr/discriminator", current_lr_d, epoch)
        self.writer.add_scalar("val/best_metric", best_metric, epoch)
        self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
