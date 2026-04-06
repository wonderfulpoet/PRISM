from __future__ import annotations

import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from .common import list_files, random_choice_excluding, to_tensor


class NatureImageMARDataset(Dataset):
    """
    适配当前 MAR 新训练代码的自然图像版本。

    重要说明：
    1. 旧版 NatureImage 数据只有 artifact 与 no_artifact，并不天然包含 LI 图或严格配对 clean GT。
    2. 当前新模型训练目标需要 clean_ct_pair 与 li_ct，因此这里只做“接口适配”。
    3. 若 paired_clean_dir 存在同名文件，则优先作为 clean_ct_pair；否则退化为使用随机 clean 图像。
    4. 若没有 li_dir，则 li_ct 默认复用 artifact 图像。这样可以跑通代码，但不代表训练设定完全等价。
    """

    def __init__(
        self,
        artifact_dir: str,
        clean_dir: str,
        paired_clean_dir: str | None = None,
        li_dir: str | None = None,
        load_size: int = 384,
        crop_size: int = 256,
        crop_type: str = 'random',
        random_flip: bool = True,
        li_mode: str = 'artifact',
        recursive: bool = True,
        **_: object,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.clean_dir = Path(clean_dir)
        self.paired_clean_dir = Path(paired_clean_dir) if paired_clean_dir is not None else None
        self.li_dir = Path(li_dir) if li_dir is not None else None
        self.load_size = int(load_size)
        self.crop_size = int(crop_size)
        self.crop_type = crop_type
        self.random_flip = random_flip
        self.li_mode = li_mode

        self.artifact_files = list_files(self.artifact_dir, suffixes={'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}, recursive=recursive)
        self.clean_files = list_files(self.clean_dir, suffixes={'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}, recursive=recursive)

    def _load_image(self, path: Path) -> Image.Image:
        return Image.open(path).convert('L')

    def _paired_clean_path(self, artifact_path: Path) -> Path | None:
        if self.paired_clean_dir is None:
            return None
        relative_path = artifact_path.relative_to(self.artifact_dir)
        candidate = self.paired_clean_dir / relative_path
        return candidate if candidate.exists() else None

    def _li_path(self, artifact_path: Path) -> Path | None:
        if self.li_dir is None:
            return None
        relative_path = artifact_path.relative_to(self.artifact_dir)
        candidate = self.li_dir / relative_path
        return candidate if candidate.exists() else None

    def _get_crop_params(self, image: Image.Image) -> tuple[int, int, int, int]:
        resized = TF.resize(image, [self.load_size, self.load_size])
        width, height = resized.size
        if self.crop_size > min(width, height):
            raise ValueError(f'crop_size={self.crop_size} is larger than resized image size {(height, width)}')
        if self.crop_type == 'center':
            top = (height - self.crop_size) // 2
            left = (width - self.crop_size) // 2
        elif self.crop_type == 'random':
            top = random.randint(0, height - self.crop_size)
            left = random.randint(0, width - self.crop_size)
        else:
            raise ValueError(f'Unsupported crop_type: {self.crop_type}')
        return top, left, self.crop_size, self.crop_size

    def _apply_shared_transform(self, image: Image.Image, crop_params: tuple[int, int, int, int], do_flip: bool) -> np.ndarray:
        image = TF.resize(image, [self.load_size, self.load_size])
        image = TF.crop(image, *crop_params)
        if do_flip:
            image = TF.hflip(image)
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = array[None, ...]
        array = array * 2.0 - 1.0
        return array.astype(np.float32)

    def __len__(self) -> int:
        return len(self.artifact_files)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        artifact_path = self.artifact_files[index]
        paired_clean_path = self._paired_clean_path(artifact_path)
        li_path = self._li_path(artifact_path)
        random_clean_path = random_choice_excluding(self.clean_files, current=paired_clean_path)

        artifact_img = self._load_image(artifact_path)
        clean_pair_img = self._load_image(paired_clean_path) if paired_clean_path is not None else self._load_image(random_clean_path)
        clean_img = self._load_image(random_clean_path)

        if li_path is not None:
            li_img = self._load_image(li_path)
        elif self.li_mode == 'artifact':
            li_img = artifact_img.copy()
        elif self.li_mode == 'clean_pair':
            li_img = clean_pair_img.copy()
        elif self.li_mode == 'zeros':
            li_img = Image.fromarray(np.zeros((artifact_img.height, artifact_img.width), dtype=np.uint8), mode='L')
        else:
            raise ValueError(f'Unsupported li_mode: {self.li_mode}')

        crop_params = self._get_crop_params(artifact_img)
        do_flip = self.random_flip and random.random() > 0.5

        artifact_ct = to_tensor(self._apply_shared_transform(artifact_img, crop_params, do_flip))
        clean_ct_pair = to_tensor(self._apply_shared_transform(clean_pair_img, crop_params, do_flip))
        clean_ct = to_tensor(self._apply_shared_transform(clean_img, crop_params, do_flip))
        li_ct = to_tensor(self._apply_shared_transform(li_img, crop_params, do_flip))

        return {
            'sample_name': str(artifact_path),
            'artifact_ct': artifact_ct,
            'clean_ct_pair': clean_ct_pair,
            'clean_ct': clean_ct,
            'li_ct': li_ct,
        }
