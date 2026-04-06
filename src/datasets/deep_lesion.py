from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class DeepLesionMARDataset(Dataset):
    def __init__(
        self,
        txt_file: str,
        data_root: str,
        samples_per_case: int,
        intensity_range: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        self.txt_file = Path(txt_file)
        self.data_root = Path(data_root)
        self.samples_per_case = samples_per_case
        self.min_value, self.max_value = intensity_range

        if not self.txt_file.exists():
            raise FileNotFoundError(f"txt file not found: {self.txt_file}")
        if not self.data_root.exists():
            raise FileNotFoundError(f"data root not found: {self.data_root}")

        self.case_list = self._load_case_list()

    def _load_case_list(self) -> List[str]:
        cases: List[str] = []
        with self.txt_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                case_name = os.path.dirname(line) if line.endswith(".h5") else line
                cases.append(case_name)
        if not cases:
            raise ValueError(f"No valid cases found in: {self.txt_file}")
        return cases

    def _read_h5(self, h5_path: Path, key: str) -> np.ndarray:
        if not h5_path.exists():
            raise FileNotFoundError(f"h5 file not found: {h5_path}")
        with h5py.File(h5_path, "r") as f:
            if key not in f:
                raise KeyError(f'Key "{key}" not found in file: {h5_path}')
            data = f[key][:]
        return np.asarray(data, dtype=np.float32)

    def _normalize(self, array: np.ndarray) -> np.ndarray:
        array = np.clip(array, self.min_value, self.max_value)
        array = (array - self.min_value) / (self.max_value - self.min_value)
        array = array * 2.0 - 1.0
        return array.astype(np.float32)

    """
    Unused helper for the current training and testing pipeline.
    Keep it here only when later visualization needs original intensity recovery.
    def denormalize(self, array: np.ndarray) -> np.ndarray:
        array = np.clip(array, -1.0, 1.0)
        array = (array + 1.0) / 2.0
        array = array * (self.max_value - self.min_value) + self.min_value
        return array.astype(np.float32)
    """

    def __len__(self) -> int:
        return len(self.case_list) * self.samples_per_case

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        case_idx = index // self.samples_per_case
        sample_idx = index % self.samples_per_case

        case_dir = self.data_root / self.case_list[case_idx]
        sample_stem = case_dir / str(sample_idx)
        sample_h5 = sample_stem.with_suffix(".h5")
        gt_h5 = case_dir / "gt.h5"

        artifact_ct = self._read_h5(sample_h5, "ma_CT")
        li_ct = self._read_h5(sample_h5, "LI_CT")
        clean_ct_pair = self._read_h5(gt_h5, "image")

        random_case_dir = self.data_root / random.choice(self.case_list)
        random_clean_h5 = random_case_dir / "gt.h5"
        clean_ct = self._read_h5(random_clean_h5, "image")

        artifact_ct = torch.from_numpy(self._normalize(artifact_ct)).float()
        clean_ct_pair = torch.from_numpy(self._normalize(clean_ct_pair)).float()
        li_ct = torch.from_numpy(self._normalize(li_ct)).float()
        clean_ct = torch.from_numpy(self._normalize(clean_ct)).float()

        return {
            "sample_name": str(sample_stem),
            "artifact_ct": artifact_ct, #l  
            "clean_ct_pair": clean_ct_pair, #l的gt
            "clean_ct": clean_ct, #random gt   
            "li_ct": li_ct, #l_li
        }
