from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from .cam_generator import Generator, prepare_calimar_input
from .discriminator import NLayerDiscriminator
from .losses import GANLoss


class CAMGenerator(nn.Module):
    def __init__(self, input_nc: int = 6, output_nc: int = 3, ngf: int = 64, n_blocks: int = 5):
        super().__init__()
        self.backbone = Generator(input_nc=input_nc, output_nc=output_nc, ngf=ngf, n_blocks=n_blocks)

    def forward(self, artifact_ct: torch.Tensor, clean_ct: torch.Tensor, li_ct: torch.Tensor) -> Dict[str, torch.Tensor]:
        pred_clean_rgb, _, _, _ = self.backbone(prepare_calimar_input(artifact_ct, li_ct))
        pred_clean = pred_clean_rgb[:, 0]

        residual_from_artifact = artifact_ct - pred_clean
        synthetic_artifact = clean_ct + residual_from_artifact

        recovered_clean_rgb, _, _, _ = self.backbone(prepare_calimar_input(synthetic_artifact, clean_ct))
        recovered_clean = recovered_clean_rgb[:, 0]

        residual_from_synthetic = synthetic_artifact - recovered_clean
        recovered_artifact = pred_clean + residual_from_synthetic

        clean_identity_rgb, _, _, _ = self.backbone(prepare_calimar_input(clean_ct, clean_ct))
        clean_identity = clean_identity_rgb[:, 0]

        return {
            "pred_clean": pred_clean,
            "synthetic_artifact": synthetic_artifact,
            "clean_identity": clean_identity,
            "recovered_artifact": recovered_artifact,
            "recovered_clean": recovered_clean,
            "residual_from_artifact": residual_from_artifact,
            "residual_from_synthetic": residual_from_synthetic,
        }


@dataclass
class LossWeights:
    ll: float = 0.0
    hh: float = 10.0
    lh: float = 10.0
    lhl: float = 10.0
    hlh: float = 20.0
    noise: float = 10.0
    gl: float = 2.0
    gh: float = 2.0


class CAMMARModel(nn.Module):
    def __init__(self, generator_cfg: dict, discriminator_cfg: dict, loss_weights: dict):
        super().__init__()
        self.generator = CAMGenerator(**generator_cfg)
        self.discriminator_clean = NLayerDiscriminator(**discriminator_cfg)
        self.discriminator_artifact = NLayerDiscriminator(**discriminator_cfg)

        self.l1_loss = nn.L1Loss()
        self.gan_loss = GANLoss("lsgan")
        self.loss_weights = LossWeights(**loss_weights)

    def forward(self, artifact_ct: torch.Tensor, clean_ct: torch.Tensor, li_ct: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.generator(artifact_ct, clean_ct, li_ct)

    def compute_generator_loss(
        self,
        artifact_ct: torch.Tensor,
        clean_ct_pair: torch.Tensor,
        clean_ct: torch.Tensor,
        li_ct: torch.Tensor,
    ):
        outputs = self.forward(artifact_ct, clean_ct, li_ct)
        losses = {}
        w = self.loss_weights

        if w.hh > 0:
            losses["hh_l1"] = w.hh * self.l1_loss(outputs["clean_identity"], clean_ct)
        if w.lh > 0:
            losses["lh_l1"] = w.lh * self.l1_loss(outputs["pred_clean"], clean_ct_pair)
        if w.lhl > 0:
            losses["lhl_l1"] = w.lhl * self.l1_loss(outputs["recovered_artifact"], artifact_ct)
        if w.hlh > 0:
            losses["hlh_l1"] = w.hlh * self.l1_loss(outputs["recovered_clean"], clean_ct)
        if w.noise > 0:
            losses["noise_l1"] = w.noise * self.l1_loss(outputs["residual_from_artifact"], outputs["residual_from_synthetic"])
        if w.gl > 0:
            losses["gl_gan"] = w.gl * self.gan_loss(self.discriminator_clean(outputs["pred_clean"]), True)
        if w.gh > 0:
            losses["gh_gan"] = w.gh * self.gan_loss(self.discriminator_artifact(outputs["synthetic_artifact"]), True)

        total_loss = sum(losses.values()) if losses else torch.tensor(0.0, device=artifact_ct.device)
        return total_loss, losses, outputs

    def compute_discriminator_loss(self, outputs: Dict[str, torch.Tensor], artifact_ct: torch.Tensor, clean_ct: torch.Tensor):
        losses = {}
        w = self.loss_weights

        if w.gl > 0:
            pred_real_clean = self.discriminator_clean(clean_ct)
            pred_fake_clean = self.discriminator_clean(outputs["pred_clean"].detach())
            losses["d_clean"] = 0.5 * w.gl * (
                self.gan_loss(pred_real_clean, True) + self.gan_loss(pred_fake_clean, False)
            )

        if w.gh > 0:
            pred_real_artifact = self.discriminator_artifact(artifact_ct)
            pred_fake_artifact = self.discriminator_artifact(outputs["synthetic_artifact"].detach())
            losses["d_artifact"] = 0.5 * w.gh * (
                self.gan_loss(pred_real_artifact, True) + self.gan_loss(pred_fake_artifact, False)
            )

        total_loss = sum(losses.values()) if losses else torch.tensor(0.0, device=artifact_ct.device)
        return total_loss, losses
