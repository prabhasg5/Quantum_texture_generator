"""Training loop scaffolding for quantum and classical runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable

import torch

from ..data.dataset import PTDTextureDataset
from ..models.classical.mlp_generator import ClassicalLatentGenerator, MLPConfig
from ..models.common.conditioning import ConditioningConfig, ConditioningEncoder
from ..models.common.decoder import TextureDecoder
from ..models.common.discriminator import ProjectionDiscriminator
from ..models.quantum.pqc_generator import PQCConfig, QuantumLatentGenerator
from .config import ExperimentConfig


@dataclass
class TrainingArtifacts:
    decoder: TextureDecoder
    discriminator: ProjectionDiscriminator
    generator: torch.nn.Module
    conditioning: ConditioningEncoder


def build_models(cfg: ExperimentConfig) -> TrainingArtifacts:
    conditioning_cfg = ConditioningConfig(class_count=cfg.class_count)
    conditioning = ConditioningEncoder(conditioning_cfg)

    if cfg.variant == "quantum":
        pqc_cfg = PQCConfig(latent_dim=conditioning_cfg.total_dim)
        generator = QuantumLatentGenerator(pqc_cfg)
    elif cfg.variant == "classical":
        mlp_cfg = MLPConfig(latent_dim=conditioning_cfg.total_dim)
        generator = ClassicalLatentGenerator(mlp_cfg)
    else:
        raise ValueError(f"Unknown variant: {cfg.variant}")

    decoder = TextureDecoder(latent_dim=conditioning_cfg.total_dim)
    discriminator = ProjectionDiscriminator(cfg.class_count)
    return TrainingArtifacts(
        decoder=decoder,
        discriminator=discriminator,
        generator=generator,
        conditioning=conditioning,
    )


def training_step(batch: Dict[str, torch.Tensor], artifacts: TrainingArtifacts) -> Dict[str, float]:
    raise NotImplementedError("Full GAN training step will be implemented once data pipeline is ready.")
