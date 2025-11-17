# ABSTRACT

Fashion designers frequently face bottlenecks when sourcing distinctive textures and patterns that match a collection’s concept and material direction. Conventional reference hunting is slow, and many generative tools either overfit to exemplars or wander a narrow latent space, yielding repetitive outputs. Quantum Canvas proposes a hybrid quantum–classical generative framework (QGAN) for texture ideation: a curated dataset is cleaned to remove non‑textures and flat‑color blocks; a conditional generator is trained to produce textures by class (e.g., banded, bumpy), and a compact parameterized quantum circuit modulates the latent representation to encourage richer mixing and exploratory variation. The workflow is designed for local execution on consumer hardware, with class‑conditioned sampling, safe checkpointing and resumption, and a reporting pipeline that includes loss curves, per‑class grids, and quantitative lenses such as LPIPS‑nearest, FID, and feature‑space coverage. This document details the background, problem framing, methodology, and applications relevant to fashion design practice.

Keywords: quantum generative models, conditional GAN, parameterized quantum circuits, texture synthesis, fashion design, dataset cleaning, LPIPS, FID, coverage

# CHAPTER‑1 Introduction

## 1.1 Origin of the Problem

Fashion design workflows rely heavily on sourcing texture and pattern references from prior collections, suppliers, and visual libraries. This process has pain points:
- Time cost: Browsing archives and the web is slow and often off‑concept.
- Repetition: Common tools generate outputs that converge to similar modes or visibly echo the dataset.
- Control: Designers need actionable levers (style families/classes) rather than unguided image synthesis.
- Novelty risk: Excessive similarity to training images risks IP concerns and weakens creative differentiation.
- Practicality: Teams need tools that run locally on standard hardware and can be slotted into existing pipelines.

Quantum Canvas emerges from these needs: a controllable generator that explores richer latent structures (via PQCs) to surface novel yet on‑brief textures quickly, supporting human creativity rather than replacing it.

## 1.2 Basic Definitions and Background

- Texture and Texture Synthesis: In visual design, “texture” refers to repeatable, local structure (e.g., banded, blotchy, bumpy). Texture synthesis aims to generate new images with similar local statistics without copying exemplars.
- GAN (Generative Adversarial Network): A generator learns to produce images that fool a discriminator trained to distinguish real from fake. Conditional GANs (cGANs) add class conditioning so designers can ask for a specific texture family.
- Hybrid Quantum–Classical Model (QGAN): A standard neural generator augmented with a small, trainable quantum circuit (PQC). The PQC (simulated here via PennyLane) maps projected latent vectors into a quantum state and returns expectation values that modulate generator features. Intended benefit: structured superposition/entanglement can enrich latent mixing and encourage creative variation.
- PennyLane and PQCs: Parameterized quantum circuits are differentiable and trainable end‑to‑end with PyTorch. On non‑CUDA Macs, simulation runs on CPU; the rest of the model can use MPS/CPU as available.
- Dataset Cleaning: We filter out humans/objects using semantic similarity (e.g., CLIP), remove flat‑color images using variance checks, and keep only texture‑like images.
- Metrics:
  - LPIPS to nearest train (lower similarity implies more novelty relative to training set).
  - FID (Fréchet Inception Distance) between generated and real distributions (lower is better fidelity).
  - Feature‑space coverage (how broadly generated samples cover the training feature manifold).

## 1.3 Problem Statement

Design a controllable, quantum‑augmented texture generator that:
- Accepts a user‑selected texture class (mapped to dataset subfolders) and generates diverse, high‑quality textures for ideation.
- Encourages novelty (avoids mimicry of the nearest training example) while preserving texture identity and usability in fashion contexts.
- Trains efficiently on a MacBook Air with 16 GB RAM; supports safe interruption and resumption without losing significant progress.
- Produces audit‑friendly outputs: per‑class sample grids, loss curves, LPIPS distributions, FID, and coverage plots.

Operational constraints and requirements:
- Data curation: Exclude non‑texture images and flat‑color blocks to stabilize training and ensure outputs remain relevant.
- Architecture: Conditional hybrid QGAN with a compact PQC block that stays on CPU; data and model use MPS/CPU as available.
- Training UX: 
  - Batch‑level progress bars.
  - Checkpoint every 2 epochs and on interrupt; auto‑resume from latest.
- Evaluation and Reporting:
  - Save loss history and curves (discriminator vs. generator).
  - Generate per‑class grids during training and at completion.
  - Compute LPIPS‑nearest novelty, FID, and feature coverage; export metrics JSON and plots.

Success criteria:
- Visual: Designers report sufficient variety within each class and perceive outputs as “inspired by” rather than “copied from” the training data.
- Quantitative: Balanced LPIPS (not too low), competitive FID (not overly degraded by novelty), and robust coverage.
- Practical: End‑to‑end run is feasible on the target machine and easy to resume.

## 1.4 Applications

- Concept Development: Rapidly explore texture directions for seasonal narratives (e.g., “coastal erosion,” “bioluminescence,” “tectonic strata”).
- Print and Surface Design: Generate tiled textures as starting points for apparel, accessories, and footwear prints.
- Material and Technique Exploration: Inspire knit stitches, jacquard weaves, embossing/laser‑cut patterns, and fabric manipulations.
- Mood Boards and Lookbooks: Populate boards with coherent yet varied textures across chosen style families.
- Iterative Co‑Creation: Designers steer by class selection and curation; Quantum Canvas supplies diverse candidates for refinement.
- Supplier Briefing: Share curated grids with print vendors as reference directions, accelerating sample turnaround.
