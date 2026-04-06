from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(state: Dict[str, Any], checkpoint_path: str | Path) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, checkpoint_path)


def load_checkpoint(checkpoint_path: str | Path, map_location: str = "cpu") -> Dict[str, Any]:
    return torch.load(Path(checkpoint_path), map_location=map_location)


def find_resume_checkpoint(checkpoint_dir: str | Path) -> str | None:
    checkpoint_dir = Path(checkpoint_dir)
    latest_path = checkpoint_dir / "latest.pth"
    if latest_path.exists():
        return str(latest_path)

    epoch_ckpts = sorted(checkpoint_dir.glob("epoch_*.pth"))
    if epoch_ckpts:
        return str(epoch_ckpts[-1])
    return None
