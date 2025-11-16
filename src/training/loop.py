"""Training loop scaffolding for quantum and classical runs."""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from shutil import copy2

from ..data.dataset import PTDTextureDataset
from ..models.classical.mlp_generator import ClassicalLatentGenerator, MLPConfig
from ..models.common.conditioning import ConditioningConfig, ConditioningEncoder
from ..models.common.decoder import TextureDecoder
from ..models.common.discriminator import ProjectionDiscriminator
from ..models.quantum.pqc_generator import PQCConfig, QuantumLatentGenerator
from ..utils.seeding import set_seed
from .config import ExperimentConfig, TrainerSettings


@dataclass
class TrainingArtifacts:
    decoder: TextureDecoder
    discriminator: ProjectionDiscriminator
    generator: torch.nn.Module
    conditioning: ConditioningEncoder


def _resolve_num_workers(num_workers: int) -> int:
    if num_workers >= 0:
        return num_workers
    return max(1, os.cpu_count() or 1)


def _build_dataloader(
    cfg: ExperimentConfig,
    training: TrainerSettings,
    device_type: str,
) -> DataLoader:
    dataset = PTDTextureDataset(
        root=cfg.dataset_root,
        split_manifest=cfg.manifest_path,
        class_map_path=cfg.class_map_path,
        palette_size=training.palette_size,
    )
    num_workers = _resolve_num_workers(training.num_workers)
    loader_kwargs = dict(
        batch_size=training.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=device_type == "cuda",
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(dataset, **loader_kwargs)
    return loader


def _select_device(training: TrainerSettings) -> torch.device:
    requested = training.device.lower()
    if requested == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        try:
            device = torch.device(requested)
        except RuntimeError as exc:
            raise ValueError(f"Unsupported device specification: {training.device}") from exc

    threads = _resolve_num_workers(training.num_workers)
    torch.set_num_threads(max(1, threads))
    if device.type in {"cuda", "mps"}:
        try:
            torch.set_float32_matmul_precision("medium")
        except AttributeError:
            pass
    return device


def _latest_checkpoint(output_dir: Path, variant: str) -> Optional[Path]:
    latest = output_dir / f"{variant}_latest.pt"
    if latest.exists():
        return latest
    candidates = sorted(output_dir.glob(f"{variant}_*.pt"))
    if not candidates:
        return None
    return candidates[-1]


def _save_checkpoint(
    output_dir: Path,
    cfg: ExperimentConfig,
    artifacts: TrainingArtifacts,
    gen_opt: torch.optim.Optimizer,
    disc_opt: torch.optim.Optimizer,
    step: int,
    epoch: int,
    batch: int,
    *,
    tag: Optional[str] = None,
    is_final: bool = False,
) -> Path:
    suffix = "final" if is_final else (tag or f"epoch{epoch + 1:03d}_step{step:06d}")
    checkpoint_path = output_dir / f"{cfg.variant}_{suffix}.pt"
    checkpoint = {
        "step": step,
        "epoch": epoch,
        "batch": batch,
        "variant": cfg.variant,
        "decoder": artifacts.decoder.state_dict(),
        "generator": artifacts.generator.state_dict(),
        "conditioning": artifacts.conditioning.state_dict(),
        "discriminator": artifacts.discriminator.state_dict(),
        "gen_opt": gen_opt.state_dict(),
        "disc_opt": disc_opt.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    latest_path = output_dir / f"{cfg.variant}_latest.pt"
    copy2(checkpoint_path, latest_path)
    return checkpoint_path


def _load_checkpoint(
    output_dir: Path,
    cfg: ExperimentConfig,
    artifacts: TrainingArtifacts,
    gen_opt: torch.optim.Optimizer,
    disc_opt: torch.optim.Optimizer,
    device: torch.device,
) -> Optional[Tuple[Dict[str, int], Path]]:
    checkpoint_path = _latest_checkpoint(output_dir, cfg.variant)
    if checkpoint_path is None:
        return None
    state = torch.load(checkpoint_path, map_location=device)
    artifacts.decoder.load_state_dict(state["decoder"])
    artifacts.generator.load_state_dict(state["generator"])
    artifacts.conditioning.load_state_dict(state["conditioning"])
    artifacts.discriminator.load_state_dict(state["discriminator"])
    gen_opt.load_state_dict(state["gen_opt"])
    disc_opt.load_state_dict(state["disc_opt"])
    meta = {
        "epoch": int(state.get("epoch", 0)),
        "batch": int(state.get("batch", 0)),
        "step": int(state.get("step", 0)),
    }
    return meta, checkpoint_path


def _save_samples(fake_batch: torch.Tensor, output_dir: Path, epoch: int) -> Optional[Path]:
    if fake_batch is None or fake_batch.numel() == 0:
        return None
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    grid = fake_batch[:64].clamp(-1.0, 1.0)
    grid = (grid + 1.0) / 2.0
    sample_path = samples_dir / f"epoch_{epoch:03d}.png"
    save_image(grid, sample_path, nrow=8)
    return sample_path


def _save_loss_history(loss_history: List[Dict[str, float]], output_dir: Path) -> Path:
    history_path = output_dir / "loss_history.pkl"
    with history_path.open("wb") as handle:
        pickle.dump(loss_history, handle)
    return history_path


def _load_loss_history(output_dir: Path) -> List[Dict[str, float]]:
    history_path = output_dir / "loss_history.pkl"
    if not history_path.exists():
        return []
    with history_path.open("rb") as handle:
        return pickle.load(handle)


def _plot_loss_curves(loss_history: List[Dict[str, float]], output_dir: Path, epoch: int) -> Optional[Path]:
    if not loss_history:
        return None
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    steps = [entry["step"] for entry in loss_history]
    loss_d = [entry["loss_d"] for entry in loss_history]
    loss_g = [entry["loss_g"] for entry in loss_history]
    plt.figure(figsize=(8, 5))
    plt.plot(steps, loss_d, label="Discriminator", color="#1f77b4")
    plt.plot(steps, loss_g, label="Generator", color="#d62728")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title(f"Training Loss Curves up to Epoch {epoch}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot_path = plots_dir / f"loss_curve_epoch_{epoch:03d}.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def build_models(cfg: ExperimentConfig, training: TrainerSettings, palette_embed_dim: int) -> TrainingArtifacts:
    conditioning_cfg = ConditioningConfig(
        class_count=cfg.class_count,
        class_embed_dim=training.class_embed_dim,
        palette_embed_dim=palette_embed_dim,
        noise_dim=training.noise_dim,
        latent_dim=training.latent_dim,
    )
    conditioning = ConditioningEncoder(conditioning_cfg)

    if cfg.variant == "quantum":
        default_device_name = PQCConfig().device_name
        pqc_cfg = PQCConfig(
            latent_dim=conditioning_cfg.latent_dim,
            n_qubits=training.n_qubits,
            device_name=cfg.device_name or default_device_name,
            shots=cfg.shots,
        )
        generator = QuantumLatentGenerator(pqc_cfg)
    elif cfg.variant == "classical":
        mlp_cfg = MLPConfig(latent_dim=conditioning_cfg.latent_dim)
        generator = ClassicalLatentGenerator(mlp_cfg)
    else:
        raise ValueError(f"Unknown variant: {cfg.variant}")

    decoder = TextureDecoder(latent_dim=conditioning_cfg.latent_dim)
    discriminator = ProjectionDiscriminator(cfg.class_count)
    return TrainingArtifacts(
        decoder=decoder,
        discriminator=discriminator,
        generator=generator,
        conditioning=conditioning,
    )


def _generate_fake(
    artifacts: TrainingArtifacts,
    class_id: torch.Tensor,
    palette_embedding: torch.Tensor,
) -> torch.Tensor:
    conditioning_vec = artifacts.conditioning(class_id, palette_embedding)
    latent = artifacts.generator(conditioning_vec)
    fake = artifacts.decoder(latent)
    return fake


def _optimization_step(
    batch: Dict[str, torch.Tensor],
    artifacts: TrainingArtifacts,
    training: TrainerSettings,
    device: torch.device,
    gen_opt: torch.optim.Optimizer,
    disc_opt: torch.optim.Optimizer,
    generator_params: Iterable[torch.Tensor],
) -> Tuple[Dict[str, float], torch.Tensor]:
    real = batch["image"].to(device)
    class_id = batch["class_id"].to(device)
    palette_embedding = batch["palette_embedding"].to(device)
    palette_colors = batch["palette_colors"].to(device)

    disc_opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        fake_detached = _generate_fake(artifacts, class_id, palette_embedding)
    real_scores = artifacts.discriminator(real, class_id)
    fake_scores = artifacts.discriminator(fake_detached, class_id)
    loss_d = F.relu(1.0 - real_scores).mean() + F.relu(1.0 + fake_scores).mean()
    loss_d.backward()
    disc_opt.step()

    gen_opt.zero_grad(set_to_none=True)
    fake = _generate_fake(artifacts, class_id, palette_embedding)
    adv_score = artifacts.discriminator(fake, class_id)
    adv_loss = -adv_score.mean()
    target_mean = palette_colors.mean(dim=1)
    fake_mean = ((fake + 1.0) / 2.0).mean(dim=(2, 3))
    palette_loss = F.l1_loss(fake_mean, target_mean)
    loss_g = adv_loss + training.lambda_palette * palette_loss
    loss_g.backward()
    if training.grad_clip is not None:
        clip_grad_norm_(generator_params, training.grad_clip)
    gen_opt.step()

    with torch.no_grad():
        metrics = {
            "loss_d": float(loss_d.item()),
            "loss_g": float(loss_g.item()),
            "loss_adv": float(adv_loss.item()),
            "loss_palette": float(palette_loss.item()),
            "real_score": float(real_scores.mean().item()),
            "fake_score": float(adv_score.mean().item()),
        }
    return metrics, fake.detach()


def train_experiment(cfg: ExperimentConfig) -> bool:
    training = cfg.training
    if training.seed is not None:
        set_seed(training.seed)

    device = _select_device(training)
    logger = logging.getLogger(__name__)
    logger.info("using device=%s", device)

    dataloader = _build_dataloader(cfg, training, device.type)
    total_batches = len(dataloader)
    sample = dataloader.dataset[0]
    palette_embed_dim = sample["palette_embedding"].shape[-1]

    artifacts = build_models(cfg, training, palette_embed_dim)

    for module in (artifacts.generator, artifacts.decoder, artifacts.discriminator, artifacts.conditioning):
        module.to(device)
        module.train()

    generator_params = list(
        chain(
            artifacts.generator.parameters(),
            artifacts.decoder.parameters(),
            artifacts.conditioning.parameters(),
        )
    )
    gen_opt = torch.optim.AdamW(
        generator_params,
        lr=training.lr,
        betas=(training.beta1, training.beta2),
        weight_decay=training.weight_decay,
    )
    disc_opt = torch.optim.AdamW(
        artifacts.discriminator.parameters(),
        lr=training.lr,
        betas=(training.beta1, training.beta2),
        weight_decay=training.weight_decay,
    )

    output_dir = training.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_size = len(dataloader.dataset)
    total_gen_params = sum(p.numel() for p in generator_params)
    total_disc_params = sum(p.numel() for p in artifacts.discriminator.parameters())
    logger.info(
        "dataset=%s palette_dim=%s generator_params=%s discriminator_params=%s",
        dataset_size,
        palette_embed_dim,
        total_gen_params,
        total_disc_params,
    )

    loss_history: List[Dict[str, float]] = []
    if training.resume:
        loss_history = _load_loss_history(output_dir)

    start_epoch = 0
    start_batch = 0
    step = 0
    resume_path: Optional[Path] = None
    if training.resume:
        resume_state = _load_checkpoint(output_dir, cfg, artifacts, gen_opt, disc_opt, device)
        if resume_state:
            state, resume_path = resume_state
            start_epoch = state.get("epoch", 0)
            start_batch = state.get("batch", 0)
            step = state.get("step", 0)
            if start_batch >= total_batches and total_batches > 0:
                start_epoch += 1
                start_batch = 0
            logger.info(
                "resuming from %s (epoch=%s batch=%s step=%s)",
                resume_path.name,
                start_epoch + 1,
                start_batch,
                step,
            )
            if loss_history:
                logger.info("loaded loss history with %s entries", len(loss_history))

    if start_epoch >= training.epochs:
        logger.info("checkpoint already covers all %s epochs; nothing to do.", training.epochs)
        return True

    interrupted = False
    stop_training = False
    current_epoch = start_epoch
    current_batch = start_batch
    progress = None
    last_fake_batch: Optional[torch.Tensor] = None

    try:
        for epoch in range(start_epoch, training.epochs):
            current_epoch = epoch
            epoch_start_batch = start_batch if epoch == start_epoch else 0
            if epoch_start_batch >= total_batches:
                epoch_start_batch = 0
            progress = tqdm(
                enumerate(dataloader),
                total=total_batches,
                desc=f"Epoch {epoch + 1}/{training.epochs}",
                leave=False,
            )
            for batch_idx, batch in progress:
                if batch_idx < epoch_start_batch:
                    continue
                step += 1
                metrics, fake_batch = _optimization_step(
                    batch,
                    artifacts,
                    training,
                    device,
                    gen_opt,
                    disc_opt,
                    generator_params,
                )
                last_fake_batch = fake_batch.detach().cpu()
                loss_history.append(
                    {
                        "step": float(step),
                        "epoch": float(epoch + 1),
                        "loss_d": metrics["loss_d"],
                        "loss_g": metrics["loss_g"],
                        "loss_adv": metrics["loss_adv"],
                        "loss_palette": metrics["loss_palette"],
                    }
                )
                progress.set_postfix(
                    loss_d=f"{metrics['loss_d']:.3f}",
                    loss_g=f"{metrics['loss_g']:.3f}",
                )
                current_batch = batch_idx + 1

                if training.log_interval and step % training.log_interval == 0:
                    logger.info(
                        "epoch=%s step=%s loss_d=%.4f loss_g=%.4f adv=%.4f palette=%.4f real=%.3f fake=%.3f",
                        epoch + 1,
                        step,
                        metrics["loss_d"],
                        metrics["loss_g"],
                        metrics["loss_adv"],
                        metrics["loss_palette"],
                        metrics["real_score"],
                        metrics["fake_score"],
                    )

                if training.checkpoint_interval and step % training.checkpoint_interval == 0:
                    _save_checkpoint(
                        output_dir,
                        cfg,
                        artifacts,
                        gen_opt,
                        disc_opt,
                        step,
                        epoch,
                        batch_idx + 1,
                        tag=f"step{step:06d}",
                    )

                if training.max_steps is not None and step >= training.max_steps:
                    stop_training = True
                    break
            progress.close()
            progress = None

            if stop_training:
                break

            start_batch = 0
            current_batch = 0

            if (
                training.sample_interval_epochs
                and (epoch + 1) % training.sample_interval_epochs == 0
                and last_fake_batch is not None
            ):
                sample_path = _save_samples(last_fake_batch, output_dir, epoch + 1)
                if sample_path is not None:
                    logger.info("saved sample grid: %s", sample_path.name)

            if training.history_interval_epochs and (epoch + 1) % training.history_interval_epochs == 0:
                history_path = _save_loss_history(loss_history, output_dir)
                plot_path = _plot_loss_curves(loss_history, output_dir, epoch + 1)
                logger.info("updated loss history (%s entries) -> %s", len(loss_history), history_path.name)
                if plot_path is not None:
                    logger.info("updated loss plot: %s", plot_path.name)

            if training.checkpoint_epochs and (epoch + 1) % training.checkpoint_epochs == 0:
                _save_checkpoint(
                    output_dir,
                    cfg,
                    artifacts,
                    gen_opt,
                    disc_opt,
                    step,
                    epoch + 1,
                    0,
                    tag=f"epoch{epoch + 1:03d}",
                )
    except KeyboardInterrupt:
        interrupted = True
        if progress is not None:
            progress.close()
        checkpoint_path = _save_checkpoint(
            output_dir,
            cfg,
            artifacts,
            gen_opt,
            disc_opt,
            step,
            current_epoch,
            current_batch,
            tag="interrupt",
        )
        logger.info("training interrupted; checkpoint saved at %s", checkpoint_path.name)
        history_path = _save_loss_history(loss_history, output_dir)
        plot_path = _plot_loss_curves(loss_history, output_dir, current_epoch + 1)
        logger.info("loss history persisted to %s", history_path.name)
        if plot_path is not None:
            logger.info("loss plot written to %s", plot_path.name)

    if interrupted:
        return False

    resume_epoch = current_epoch if current_batch > 0 else current_epoch + 1
    final_path = _save_checkpoint(
        output_dir,
        cfg,
        artifacts,
        gen_opt,
        disc_opt,
        step,
        resume_epoch,
        current_batch,
        tag="final",
        is_final=True,
    )
    history_path = _save_loss_history(loss_history, output_dir)
    plot_path = _plot_loss_curves(loss_history, output_dir, resume_epoch)
    if (
        last_fake_batch is not None
        and training.sample_interval_epochs
        and resume_epoch % training.sample_interval_epochs != 0
    ):
        sample_path = _save_samples(last_fake_batch, output_dir, resume_epoch)
        if sample_path is not None:
            logger.info("saved final sample grid: %s", sample_path.name)
    logger.info("loss history persisted to %s", history_path.name)
    if plot_path is not None:
        logger.info("loss plot written to %s", plot_path.name)
    logger.info("completed training at step=%s (final checkpoint: %s)", step, final_path.name)
    return True
