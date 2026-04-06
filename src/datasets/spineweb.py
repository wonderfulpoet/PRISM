from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import ensure_chw, list_files, normalize_to_neg_one_one, random_choice_excluding, to_tensor


class SpinewebMARDataset(Dataset):
    """
    适配当前 MAR 新训练代码的 SpineWeb 数据集版本。

    说明：
    1. 旧版 SpineWeb 代码只有 artifact、随机 clean、以及对应 LI 图像，没有严格配对 clean GT。
    2. 因此这里默认将 LI 图像同时作为 clean_ct_pair 的替代监督目标。
    3. 如果你后续有真正配对的 GT，可通过 pair_dir 提供，并自动替换 clean_ct_pair。
    """

    def __init__(
        self,
        artifact_dir: str,
        clean_dir: str,
        li_dir: str | None = None,
        pair_dir: str | None = None,
        intensity_range: tuple[float, float] = (-1000.0, 2000.0),
        recursive: bool = True,
        **_: object,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.clean_dir = Path(clean_dir)
        self.li_dir = Path(li_dir) if li_dir is not None else None
        self.pair_dir = Path(pair_dir) if pair_dir is not None else None
        self.intensity_range = tuple(float(x) for x in intensity_range)

        self.artifact_files = list_files(self.artifact_dir, suffixes={'.npy'}, recursive=recursive)
        self.clean_files = list_files(self.clean_dir, suffixes={'.npy'}, recursive=recursive)

    def _resolve_li_path(self, artifact_path: Path) -> Path:
        if self.li_dir is not None:
            relative_path = artifact_path.relative_to(self.artifact_dir)
            candidate = self.li_dir / relative_path
            if candidate.exists():
                return candidate
            candidate = candidate.with_name(candidate.stem + '_LI' + candidate.suffix)
            if candidate.exists():
                return candidate

        old_style = Path(str(artifact_path).replace('/artifact/', '/artifact_LI/'))
        old_style = old_style.with_name(old_style.stem + '_LI' + old_style.suffix)
        if old_style.exists():
            return old_style

        raise FileNotFoundError(
            f'Cannot locate LI file for artifact: {artifact_path}. '
            'Please set li_dir explicitly or keep the old artifact/artifact_LI directory layout.'
        )

    def _resolve_pair_path(self, artifact_path: Path) -> Path | None:
        if self.pair_dir is None:
            return None
        relative_path = artifact_path.relative_to(self.artifact_dir)
        candidate = self.pair_dir / relative_path
        if candidate.exists():
            return candidate
        return None

    def __len__(self) -> int:
        return len(self.artifact_files)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        artifact_path = self.artifact_files[index]
        li_path = self._resolve_li_path(artifact_path)
        pair_path = self._resolve_pair_path(artifact_path)
        random_clean_path = random_choice_excluding(self.clean_files)

        artifact_ct = np.load(artifact_path).astype(np.float32)
        li_ct = np.load(li_path).astype(np.float32)
        clean_ct = np.load(random_clean_path).astype(np.float32)

        if pair_path is not None:
            clean_ct_pair = np.load(pair_path).astype(np.float32)
        else:
            clean_ct_pair = li_ct.copy()

        artifact_ct = to_tensor(normalize_to_neg_one_one(ensure_chw(artifact_ct), self.intensity_range))
        li_ct = to_tensor(normalize_to_neg_one_one(ensure_chw(li_ct), self.intensity_range))
        clean_ct = to_tensor(normalize_to_neg_one_one(ensure_chw(clean_ct), self.intensity_range))
        clean_ct_pair = to_tensor(normalize_to_neg_one_one(ensure_chw(clean_ct_pair), self.intensity_range))

        return {
            'sample_name': str(artifact_path),
            'artifact_ct': artifact_ct,
            'clean_ct_pair': clean_ct_pair,
            'clean_ct': clean_ct,
            'li_ct': li_ct,
        }
