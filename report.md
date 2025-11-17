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

Fashion designers constantly need fresh, original patterns to spark new garment ideas, yet most available tools—classical GANs, diffusion models, and online inspiration sources—recycle variations of existing textures. Daily practice often devolves into scrolling thousands of reference images, reusing familiar motifs, or relying on narrow mood boards that lack novelty. Classical generative models amplify this issue because they are optimized to mimic training distributions rather than to expand creative possibility; the result is predictable, repetitive, and “safe” outputs. This constrains creative range, slows ideation, and risks collections feeling derivative instead of innovative. The need is for a system that actively broadens texture space—producing fresh, unexpected patterns that can drive new design directions—while retaining controllable links to established texture families.

### 1.3.1 Objectives

1. Create a system that makes fresh, varied texture designs from chosen style and color ideas with Quantum GAN.
2. Add simple metrics to check how new, varied, and quality the textures are.
3. Provide an easy web page where designers pick ideas and get texture sets.

### 1.3.2 Outcomes

1. Texture sets generated with richer latent space.
2. Numbers and charts show the system gives broader variety without losing quality.
3. The web page where fashion designers quickly build moodboard‑ready texture packs.


## 1.4 Applications

- Concept Development: Rapidly explore texture directions for seasonal narratives (e.g., “coastal erosion,” “bioluminescence,” “tectonic strata”).
- Print and Surface Design: Generate tiled textures as starting points for apparel, accessories, and footwear prints.
- Material and Technique Exploration: Inspire knit stitches, jacquard weaves, embossing/laser‑cut patterns, and fabric manipulations.
- Mood Boards and Lookbooks: Populate boards with coherent yet varied textures across chosen style families.
- Iterative Co‑Creation: Designers steer by class selection and curation; Quantum Canvas supplies diverse candidates for refinement.
- Supplier Briefing: Share curated grids with print vendors as reference directions, accelerating sample turnaround.

## 1.5 Societal Applications (Short Version)

1. Creative empowerment: Offers an ongoing stream of original textures to help designers break creative blocks and speed up idea development.
2. Sustainable development: Cuts reliance on physical fabric sampling, reducing material waste, dyes usage, and environmental impact.
3. Democratized access: Brings advanced pattern generation to students and small studios without costly software licenses.
4. Quantum awareness: Demonstrates a practical, everyday creative use case for emerging quantum technology.

[![](https://mermaid.ink/img/pako:eNqtVu1u4kYUfZWRVys5qkMAQwC3SsViL4tEYGVYVe2yjQZ7ABd7hsyMd0lC_u4D9Gkq5R3Sd-iT9I4_CCFkFbdrJOSZued-nLlzxjeax3yiWdosZF-8BeYS9d0JRfC8fo2On3-2NqPxr31nhPRzRpm34CwiBpqG2FsiSdby6OW-vBALYZMZ8rFYEB8JydmSWK_K5bKRvh9_CXy5sCrGLAhD69UsefI1hcKc4yvrFNUMj4WMJ9gf97xHzI9D8rz36mPvzzvyAxwx-q08S_WX-mKx_C5-PByGyldmXlc_I3fcarWekmUi8xv-RAQOB0wSdIByizJKHoMLdU5v8P7DGDon229OPInpPCSiQNNg6kFi-OPJRMte0WU8DaRAaFM--z2aaCefLMtKQ6SQS8oCQRTiMsZUxhFKJtBniM84QtcHMCGOpj5-w9YK1sdTEoL_-7szhHS2kgGjODzagxWiousMHLc9HrpIz2iIQ8yzXi1AR_fjQ1VzQgnHUNJPU35y1tXv_7roog26vzPQNSSrck3957mS9UrFVuUIVefuGP3z9U8URHhO4FhToXiaH-BphpekgxNGdP0tDDKMjyU-OgLjbUMVJMh12n1kt8ftvF0KkKKCCyJVSXb6inS3-ybRp5hDuyG0PlALJzjc1uLC4HvVArt87rhd5-WgoUtBpW-G7q3KMdWdgkHt3qjj9s57gyddVoBJG9or0YYAhAYEUHg8iAK6bTL74u-v-nqvuRIgBZp0vTMYoB-QCOYRC_z_xaHtdHqj3nCAhh_GoCL_tSCfeIGADYa6kh2GvlZd-3NaAShpwbSc98POO9Rp9_uQ16jA3QcsVHTIYkX4MVnBRWqhkAlx4cX8MymtKBy2T4qvTOAfUNVDqEUgYEuuSqtl-BzOVLiISB54D_Z_CEYNJHC0AhU-FLUQGW_7w1-QY3edUWE9B8Mz1N3VazWzud7kk1tBTubv75KFTAOTqfnmkZ49VbgkRHqwcmQmFAl-vdlbTEcJys6aOhnkLbRDTkaXQDrckMdECEJloA4MkxIuOvWpld9xkGwJnCT7v_Waz1SfzJiH6lCLO8K7V4ta3ZGynfOYrkExmgGdwCMc-PAheKMsJppckIhMNAtefcyXE21Cb8EOx5KNrqinWZLHxNA4i-cLzZrhUMAoXkFYYgd4znG0nV1h-htjUQ6ZcxUngxPqE95hMZWaZVbMxFizbrS1ZtWbpWqtWWs0as0W_LUM7UqzKqeNUtk8NeuVcr1erpjVW0O7TpyXS61yo1mrnFarzWqrZjZqt_8CD5Qhow?type=png)](https://mermaid.live/edit#pako:eNqtVu1u4kYUfZWRVys5qkMAQwC3SsViL4tEYGVYVe2yjQZ7ABd7hsyMd0lC_u4D9Gkq5R3Sd-iT9I4_CCFkFbdrJOSZued-nLlzxjeax3yiWdosZF-8BeYS9d0JRfC8fo2On3-2NqPxr31nhPRzRpm34CwiBpqG2FsiSdby6OW-vBALYZMZ8rFYEB8JydmSWK_K5bKRvh9_CXy5sCrGLAhD69UsefI1hcKc4yvrFNUMj4WMJ9gf97xHzI9D8rz36mPvzzvyAxwx-q08S_WX-mKx_C5-PByGyldmXlc_I3fcarWekmUi8xv-RAQOB0wSdIByizJKHoMLdU5v8P7DGDon229OPInpPCSiQNNg6kFi-OPJRMte0WU8DaRAaFM--z2aaCefLMtKQ6SQS8oCQRTiMsZUxhFKJtBniM84QtcHMCGOpj5-w9YK1sdTEoL_-7szhHS2kgGjODzagxWiousMHLc9HrpIz2iIQ8yzXi1AR_fjQ1VzQgnHUNJPU35y1tXv_7roog26vzPQNSSrck3957mS9UrFVuUIVefuGP3z9U8URHhO4FhToXiaH-BphpekgxNGdP0tDDKMjyU-OgLjbUMVJMh12n1kt8ftvF0KkKKCCyJVSXb6inS3-ybRp5hDuyG0PlALJzjc1uLC4HvVArt87rhd5-WgoUtBpW-G7q3KMdWdgkHt3qjj9s57gyddVoBJG9or0YYAhAYEUHg8iAK6bTL74u-v-nqvuRIgBZp0vTMYoB-QCOYRC_z_xaHtdHqj3nCAhh_GoCL_tSCfeIGADYa6kh2GvlZd-3NaAShpwbSc98POO9Rp9_uQ16jA3QcsVHTIYkX4MVnBRWqhkAlx4cX8MymtKBy2T4qvTOAfUNVDqEUgYEuuSqtl-BzOVLiISB54D_Z_CEYNJHC0AhU-FLUQGW_7w1-QY3edUWE9B8Mz1N3VazWzud7kk1tBTubv75KFTAOTqfnmkZ49VbgkRHqwcmQmFAl-vdlbTEcJys6aOhnkLbRDTkaXQDrckMdECEJloA4MkxIuOvWpld9xkGwJnCT7v_Waz1SfzJiH6lCLO8K7V4ta3ZGynfOYrkExmgGdwCMc-PAheKMsJppckIhMNAtefcyXE21Cb8EOx5KNrqinWZLHxNA4i-cLzZrhUMAoXkFYYgd4znG0nV1h-htjUQ6ZcxUngxPqE95hMZWaZVbMxFizbrS1ZtWbpWqtWWs0as0W_LUM7UqzKqeNUtk8NeuVcr1erpjVW0O7TpyXS61yo1mrnFarzWqrZjZqt_8CD5Qhow)