from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def list_files(root: str | Path, suffixes: Iterable[str], recursive: bool = True) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f'Path not found: {root}')
    suffixes = {s.lower() for s in suffixes}
    pattern = '**/*' if recursive else '*'
    files = [p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in suffixes]
    files.sort()
    if not files:
        raise ValueError(f'No files with suffix {sorted(suffixes)} found under: {root}')
    return files


def normalize_to_neg_one_one(array: np.ndarray, value_range: tuple[float, float]) -> np.ndarray:
    min_value, max_value = value_range
    if max_value <= min_value:
        raise ValueError(f'Invalid intensity_range: {value_range}')
    array = np.asarray(array, dtype=np.float32)
    array = np.clip(array, min_value, max_value)
    array = (array - min_value) / (max_value - min_value)
    array = array * 2.0 - 1.0
    return array.astype(np.float32)


def ensure_chw(array: np.ndarray, force_single_channel: bool = True) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        array = array[None, ...]
    elif array.ndim == 3:
        if array.shape[0] in (1, 3):
            pass
        elif array.shape[-1] in (1, 3):
            array = np.transpose(array, (2, 0, 1))
        else:
            raise ValueError(f'Unsupported 3D array shape: {array.shape}')
    else:
        raise ValueError(f'Unsupported array shape: {array.shape}')

    if force_single_channel and array.shape[0] == 3:
        array = 0.299 * array[0:1] + 0.587 * array[1:2] + 0.114 * array[2:3]
    return array.astype(np.float32)


def random_choice_excluding(paths: list[Path], current: Path | None = None) -> Path:
    if not paths:
        raise ValueError('Empty candidate list.')
    if current is None or len(paths) == 1:
        return random.choice(paths)
    for _ in range(8):
        candidate = random.choice(paths)
        if candidate != current:
            return candidate
    return random.choice(paths)


def pil_to_gray_numpy(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert('L'), dtype=np.float32)


def to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array)).float()
