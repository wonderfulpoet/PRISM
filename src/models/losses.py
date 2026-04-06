from __future__ import annotations

import torch
import torch.nn as nn


class GANLoss(nn.Module):
    def __init__(self, gan_mode: str = "lsgan", target_real_label: float = 1.0, target_fake_label: float = 0.0):
        super().__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == "wgangp":
            self.loss = None
        else:
            raise NotImplementedError(f"gan mode {gan_mode} not implemented")

    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target_tensor = self.real_label if target_is_real else self.fake_label
        return target_tensor.expand_as(prediction)

    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        if self.gan_mode in ["lsgan", "vanilla"]:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            return self.loss(prediction, target_tensor)
        if target_is_real:
            return -prediction.mean()
        return prediction.mean()
