from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import yaml

from src.data.texture_dataset import DataConfig, build_dataloader
from src.models.qgan import ModelConfig, build_models
from src.training.metrics import (
    compute_fid,
    compute_feature_coverage,
    compute_lpips_novelty,
    save_metrics,
)
from src.utils.visualization import plot_losses, save_class_grid, save_lpips_hist, plot_coverage


@dataclass
class TrainingConfig:
    epochs: int = 50
    lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    latent_dim: int = 96
    samples_per_class: int = 6
    checkpoint_every: int = 5
    log_dir: Path = Path("runs/qgan")
    reports_dir: Path = Path("reports/qgan")
    novelty_lambda: float = 0.05
    device: str = "auto"
    resume_from: Path | None = None

    fid_samples: int = 512
    lpips_samples: int = 128
    lpips_ref: int = 256
    coverage_samples: int = 256
    coverage_k: int = 5


class Trainer:
    def __init__(
        self,
        data_cfg: DataConfig,
        model_cfg: ModelConfig,
        train_cfg: TrainingConfig,
    ) -> None:
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg

        self.device = self._select_device(train_cfg.device)
        torch.set_float32_matmul_precision("high")

        self.loader, self.dataset = build_dataloader(data_cfg, augment=True)
        self.eval_dataset = build_dataloader(data_cfg, augment=False)[1]

        self.generator, self.discriminator = build_models(len(self.dataset.class_to_idx), model_cfg)
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()
        self.opt_G = optim.Adam(self.generator.parameters(), lr=train_cfg.lr, betas=(train_cfg.beta1, train_cfg.beta2))
        self.opt_D = optim.Adam(self.discriminator.parameters(), lr=train_cfg.lr, betas=(train_cfg.beta1, train_cfg.beta2))

        self.loss_history: Dict[str, list] = {"discriminator": [], "generator": []}
        self.global_step = 0
        self.start_epoch = 1

        self.log_dir = Path(train_cfg.log_dir)
        self.reports_dir = Path(train_cfg.reports_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        if self.train_cfg.resume_from:
            self._resume_if_possible(self.train_cfg.resume_from)

    @staticmethod
    def _select_device(requested: str) -> torch.device:
        if requested == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if requested == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if requested == "cpu":
            return torch.device("cpu")
        if requested == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
        return torch.device("cpu")

    def _sample_noise(self, batch_size: int) -> torch.Tensor:
        return torch.randn(batch_size, self.model_cfg.latent_dim, device=self.device)

    def _train_step(self, batch) -> Dict[str, float]:
        real = batch["image"].to(self.device)
        class_ids = batch["label"].to(self.device)
        bsz = real.size(0)

        valid = torch.ones(bsz, device=self.device)
        fake_label = torch.zeros(bsz, device=self.device)

        # Train discriminator
        self.opt_D.zero_grad()
        noise = self._sample_noise(bsz)
        fake = self.generator(noise, class_ids)
        real_logits = self.discriminator(real, class_ids)
        fake_logits = self.discriminator(fake.detach(), class_ids)
        loss_real = self.criterion(real_logits, valid)
        loss_fake = self.criterion(fake_logits, fake_label)
        d_loss = (loss_real + loss_fake) * 0.5
        d_loss.backward()
        self.opt_D.step()

        # Train generator
        self.opt_G.zero_grad()
        fake_logits = self.discriminator(fake, class_ids)
        adv_loss = self.criterion(fake_logits, valid)
        diversity = 0.0
        if bsz > 1:
            flat = fake.view(bsz, -1)
            diversity = torch.pdist(flat, p=2).mean()
        g_loss = adv_loss
        if isinstance(diversity, torch.Tensor):
            g_loss = g_loss - self.train_cfg.novelty_lambda * diversity
        g_loss.backward()
        self.opt_G.step()

        self.loss_history["discriminator"].append(d_loss.item())
        self.loss_history["generator"].append(g_loss.item())
        self.global_step += 1
        return {"d_loss": d_loss.item(), "g_loss": g_loss.item(), "diversity": float(diversity) if isinstance(diversity, torch.Tensor) else 0.0}

    def fit(self) -> None:
        for epoch in range(self.start_epoch, self.train_cfg.epochs + 1):
            for batch in self.loader:
                metrics = self._train_step(batch)
            print(
                f"Epoch {epoch}/{self.train_cfg.epochs} | D: {metrics['d_loss']:.3f} | G: {metrics['g_loss']:.3f} | Diversity: {metrics['diversity']:.3f}"
            )
            if epoch % self.train_cfg.checkpoint_every == 0:
                self._save_checkpoint(epoch)
                self._log_samples(epoch)

        # Final artifacts
        self._save_checkpoint(self.train_cfg.epochs)
        self._log_samples(self.train_cfg.epochs)
        self._finalize_reports()

    def _save_checkpoint(self, epoch: int) -> None:
        ckpt = {
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "opt_G": self.opt_G.state_dict(),
            "opt_D": self.opt_D.state_dict(),
            "config": {
                "data": self.data_cfg.__dict__,
                "model": self.model_cfg.__dict__,
                "train": self.train_cfg.__dict__,
            },
            "epoch": epoch,
            "global_step": self.global_step,
            "loss_history": self.loss_history,
        }
        path = self.log_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(ckpt, path)

    def _resume_if_possible(self, checkpoint_path: Path) -> None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            print(f"[Resume] Checkpoint not found: {checkpoint_path}")
            return
        data = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(data["generator"])
        self.discriminator.load_state_dict(data["discriminator"])
        if "opt_G" in data:
            self.opt_G.load_state_dict(data["opt_G"])
        if "opt_D" in data:
            self.opt_D.load_state_dict(data["opt_D"])
        self.loss_history = data.get("loss_history", self.loss_history)
        self.global_step = data.get("global_step", self.global_step)
        last_epoch = data.get("epoch", 0)
        self.start_epoch = min(last_epoch + 1, self.train_cfg.epochs)
        print(f"[Resume] Loaded checkpoint '{checkpoint_path}' (epoch {last_epoch}). Resuming from epoch {self.start_epoch}.")

    def _log_samples(self, epoch: int) -> None:
        out_path = self.log_dir / f"class_grid_epoch_{epoch}.png"
        save_class_grid(
            self.generator,
            [self.dataset.idx_to_class[i] for i in range(len(self.dataset.idx_to_class))],
            self.device,
            samples_per_class=self.train_cfg.samples_per_class,
            latent_dim=self.model_cfg.latent_dim,
            out_path=out_path,
        )

    def _finalize_reports(self) -> None:
        # Loss curves
        plot_losses(self.loss_history, self.reports_dir / "loss_curve.png")

        # Final per-class grid
        save_class_grid(
            self.generator,
            [self.dataset.idx_to_class[i] for i in range(len(self.dataset.idx_to_class))],
            self.device,
            samples_per_class=self.train_cfg.samples_per_class,
            latent_dim=self.model_cfg.latent_dim,
            out_path=self.reports_dir / "class_grid.png",
        )

        # Metrics
        fid = compute_fid(
            self.generator,
            self.eval_dataset,
            self.device,
            self.model_cfg.latent_dim,
            batch_size=self.data_cfg.batch_size,
            sample_count=self.train_cfg.fid_samples,
        )
        lpips_avg, lpips_scores = compute_lpips_novelty(
            self.generator,
            self.eval_dataset,
            self.device,
            self.model_cfg.latent_dim,
            sample_count=self.train_cfg.lpips_samples,
            reference_count=self.train_cfg.lpips_ref,
        )
        coverage, real_radius, fake_dists = compute_feature_coverage(
            self.generator,
            self.eval_dataset,
            self.device,
            self.model_cfg.latent_dim,
            sample_count=self.train_cfg.coverage_samples,
            k=self.train_cfg.coverage_k,
        )

        save_lpips_hist(lpips_scores, self.reports_dir / "lpips_hist.png")
        plot_coverage(real_radius, fake_dists, coverage, self.reports_dir / "feature_coverage.png")

        save_metrics(
            {
                "fid": fid,
                "lpips_nearest_train_avg": lpips_avg,
                "feature_coverage": coverage,
                "lpips_scores": lpips_scores,
            },
            self.reports_dir / "metrics.json",
        )


def load_config(path: Path) -> Dict:
    with path.open() as f:
        return yaml.safe_load(f)


def build_from_yaml(cfg: Dict, resume_override: Path | None = None) -> Trainer:
    data = cfg.get("dataset", {})
    training = cfg.get("training", {})
    model = cfg.get("model", {})

    data_cfg = DataConfig(
        root=Path(data.get("root", "dataset")),
        batch_size=data.get("batch_size", 16),
        image_size=data.get("image_size", 128),
        num_workers=data.get("num_workers", 8),
        persistent_workers=data.get("persistent_workers", True),
        pin_memory=data.get("pin_memory", True),
    )
    model_cfg = ModelConfig(
        latent_dim=model.get("latent_dim", training.get("latent_dim", 96)),
        image_size=data_cfg.image_size,
        image_channels=model.get("image_channels", 3),
        base_channels=model.get("base_channels", 64),
        quantum_wires=model.get("quantum_wires", 6),
        quantum_layers=model.get("quantum_layers", 2),
        class_emb_dim=model.get("class_emb_dim", 64),
    )
    resume_cfg = training.get("resume_from")
    resume_path = Path(resume_cfg) if resume_cfg else None
    if resume_override:
        resume_path = resume_override

    train_cfg = TrainingConfig(
        epochs=training.get("epochs", 50),
        lr=training.get("lr", 2e-4),
        beta1=training.get("beta1", 0.5),
        beta2=training.get("beta2", 0.999),
        latent_dim=model_cfg.latent_dim,
        samples_per_class=training.get("samples_per_class", 6),
        checkpoint_every=training.get("checkpoint_every", 5),
        log_dir=Path(training.get("log_dir", "runs/qgan")),
        reports_dir=Path(training.get("reports_dir", "reports/qgan")),
        novelty_lambda=training.get("novelty_lambda", 0.05),
        device=training.get("device", "auto"),
        resume_from=resume_path,
        fid_samples=training.get("fid_samples", 512),
        lpips_samples=training.get("lpips_samples", 128),
        lpips_ref=training.get("lpips_ref", 256),
        coverage_samples=training.get("coverage_samples", 256),
        coverage_k=training.get("coverage_k", 5),
    )
    return Trainer(data_cfg, model_cfg, train_cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the quantum texture GAN")
    parser.add_argument("--config", type=Path, default=Path("configs/qgan.yaml"), help="Path to YAML config")
    parser.add_argument("--resume", type=Path, default=None, help="Optional checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    trainer = build_from_yaml(cfg, resume_override=args.resume)
    trainer.fit()


if __name__ == "__main__":
    main()
