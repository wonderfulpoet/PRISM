from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.cuda.amp import autocast
from tqdm import tqdm


class Evaluator:
    def __init__(
        self,
        model,
        device,
        save_dir: Path | None = None,
        use_amp: bool = False,
        sample_root: Path | None = None,
    ):
        self.model = model
        self.device = device
        self.save_dir = save_dir
        self.use_amp = use_amp
        self.sample_root = Path(sample_root).resolve() if sample_root is not None else None
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _to_numpy_2d(tensor: torch.Tensor) -> np.ndarray:
        array = tensor.detach().cpu().float().numpy()
        array = np.squeeze(array)
        if array.ndim == 2:
            return array
        if array.ndim == 3:
            return array[0]
        raise ValueError(f"Unsupported tensor shape: {array.shape}")

    @staticmethod
    def _neg_one_one_to_zero_one(array: np.ndarray) -> np.ndarray:
        array = np.clip(array, -1.0, 1.0)
        return (array + 1.0) / 2.0

    @staticmethod
    def _clip_and_rescale_for_png(
        array: np.ndarray,
        clip_min: float = 0.1584,
        clip_max: float = 0.2448,
    ) -> np.ndarray:
        """
        先按 [clip_min, clip_max] 截断，再重新归一化到 0~1。
        """
        array = np.clip(array, 0.0, 1.0)
        array = np.clip(array, clip_min, clip_max)
        if clip_max <= clip_min:
            raise ValueError(f"clip_max must be greater than clip_min, got {clip_min}, {clip_max}")
        array = (array - clip_min) / (clip_max - clip_min)
        return np.clip(array, 0.0, 1.0)

    def _save_png(self, image: np.ndarray, save_path: Path) -> None:
        image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(image_uint8, mode="L").save(save_path)

    @staticmethod
    def _build_save_prefix(
        save_dir: Path,
        sample_name: str,
        global_idx: int,
        sample_root: Path | None = None,
    ) -> Path:
        sample_path = Path(str(sample_name))

        if sample_root is not None:
            try:
                sample_path = sample_path.resolve().relative_to(sample_root)
            except Exception:
                # 如果不在 sample_root 下，就退回原逻辑
                pass

        target_dir = save_dir / sample_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir / f"{global_idx:06d}_{sample_path.stem}"

    def evaluate(self, loader, epoch: int | None = None, save_predictions: bool = False) -> Dict[str, float]:
        self.model.eval()
        psnr_values = []
        ssim_values = []
        mae_values = []

        save_dir = None
        if self.save_dir is not None and save_predictions:
            save_dir = self.save_dir / (f"epoch_{epoch:03d}" if epoch is not None else "predictions")
            save_dir.mkdir(parents=True, exist_ok=True)

        global_idx = 0

        with torch.no_grad():
            for batch in tqdm(loader, desc="Validation", leave=False):
                artifact_ct = batch["artifact_ct"].to(self.device, non_blocking=True)
                clean_ct = batch["clean_ct"].to(self.device, non_blocking=True)
                clean_ct_pair = batch["clean_ct_pair"].to(self.device, non_blocking=True)
                li_ct = batch["li_ct"].to(self.device, non_blocking=True)

                with autocast(enabled=self.use_amp):
                    outputs = self.model(artifact_ct, clean_ct, li_ct)
                pred_clean = outputs["pred_clean"]

                for i in range(pred_clean.shape[0]):
                    pred_np = self._neg_one_one_to_zero_one(self._to_numpy_2d(pred_clean[i]))
                    gt_np = self._neg_one_one_to_zero_one(self._to_numpy_2d(clean_ct_pair[i]))

                    psnr_values.append(peak_signal_noise_ratio(gt_np, pred_np, data_range=1.0))
                    ssim_values.append(structural_similarity(gt_np, pred_np, data_range=1.0))
                    mae_values.append(float(np.mean(np.abs(pred_np - gt_np))))

                    if save_dir is not None:
                        input_np = self._neg_one_one_to_zero_one(self._to_numpy_2d(artifact_ct[i]))
                        li_np = self._neg_one_one_to_zero_one(self._to_numpy_2d(li_ct[i]))
                        gt_np_save = self._neg_one_one_to_zero_one(self._to_numpy_2d(clean_ct_pair[i]))
                        pred_np_save = pred_np

                        # 统一做可视化 window/clip
                        input_png = self._clip_and_rescale_for_png(
                            input_np, clip_min=0.1584, clip_max=0.2448
                        )
                        li_png = self._clip_and_rescale_for_png(
                            li_np, clip_min=0.1584, clip_max=0.2448
                        )
                        gt_png = self._clip_and_rescale_for_png(
                            gt_np_save, clip_min=0.1584, clip_max=0.2448
                        )
                        pred_png = self._clip_and_rescale_for_png(
                            pred_np_save, clip_min=0.1584, clip_max=0.2448
                        )

                        save_prefix = self._build_save_prefix(
                            save_dir=save_dir,
                            sample_name=str(batch["sample_name"][i]),
                            global_idx=global_idx,
                            sample_root=self.sample_root,
                        )

                        # 分别保存
                        self._save_png(input_png, save_prefix.parent / f"{save_prefix.name}_input.png")
                        self._save_png(li_png, save_prefix.parent / f"{save_prefix.name}_li.png")
                        self._save_png(gt_png, save_prefix.parent / f"{save_prefix.name}_gt.png")
                        self._save_png(pred_png, save_prefix.parent / f"{save_prefix.name}_pred.png")

                        # 再额外保存一张拼接图，方便看对比
                        compare_png = np.concatenate(
                            [input_png, li_png, gt_png, pred_png],
                            axis=1,
                        )
                        self._save_png(compare_png, save_prefix.parent / f"{save_prefix.name}_compare.png")

                    global_idx += 1

        return {
            "psnr": float(np.mean(psnr_values)) if psnr_values else 0.0,
            "ssim": float(np.mean(ssim_values)) if ssim_values else 0.0,
            "mae": float(np.mean(mae_values)) if mae_values else 0.0,
        }