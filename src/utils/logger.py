from __future__ import annotations

from collections import defaultdict
from typing import Dict


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


class MetricLogger:
    def __init__(self) -> None:
        self.meters = defaultdict(AverageMeter)

    def update(self, metrics: Dict[str, float], n: int = 1) -> None:
        for key, value in metrics.items():
            self.meters[key].update(value, n)

    def averages(self) -> Dict[str, float]:
        return {k: v.avg for k, v in self.meters.items()}
