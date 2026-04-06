from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def setup_cudnn(cudnn_cfg: dict | None) -> None:
    cudnn_cfg = cudnn_cfg or {}
    torch.backends.cudnn.benchmark = bool(cudnn_cfg.get("benchmark", True))
    torch.backends.cudnn.deterministic = bool(cudnn_cfg.get("deterministic", False))


def init_distributed_mode(backend: str = "nccl") -> None:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    dist.barrier()


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    return get_rank() == 0
