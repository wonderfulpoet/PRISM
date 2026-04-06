from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from src.datasets import DeepLesionMARDataset
from src.engine import Evaluator
from src.models import CAMMARModel
from src.utils import load_checkpoint, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="CAM/configs/CAM.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_predictions", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if cfg.get("device", "cuda") == "cuda" else "cpu"

    test_dataset = DeepLesionMARDataset(
        txt_file=cfg["dataset"]["test"]["txt_file"],
        data_root=cfg["dataset"]["test"]["data_root"],
        samples_per_case=cfg["dataset"]["test"]["samples_per_case"],
        intensity_range=tuple(cfg["dataset"].get("intensity_range", [0.0, 1.0])),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["test"]["batch_size"],
        shuffle=False,
        num_workers=cfg["test"]["num_workers"],
        pin_memory=cfg["test"].get("pin_memory", True),
    )

    model = CAMMARModel(
        generator_cfg=cfg["model"]["generator"],
        discriminator_cfg=cfg["model"]["discriminator"],
        loss_weights=cfg["loss_weights"],
    ).to(device)

    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)

    save_dir = Path(cfg["output_dir"]) / cfg["experiment_name"] / "test_outputs"
    evaluator = Evaluator(
        model.generator,
        device=device,
        save_dir=save_dir,
        use_amp=cfg.get("use_amp", False),
        sample_root=Path(cfg["dataset"]["test"]["data_root"]),
    )
    metrics = evaluator.evaluate(test_loader, save_predictions=args.save_predictions)
    print({k: round(v, 6) for k, v in metrics.items()})


if __name__ == "__main__":
    main()