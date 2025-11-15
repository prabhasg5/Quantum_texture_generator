# Architecture Overview

## Conditioning Inputs
- **Texture class embedding:** Learned embeddings for PTD texture categories (e.g., "organic waves", "geometric grid").
- **Palette embedding:** Palette encoder transforms 3–7 RGB colors into a fixed-length vector capturing hue balance, saturation spread, and luminance contrast.
- **Noise seed:** Low-dimensional Gaussian noise provides stochasticity for both quantum and classical pipelines.

The concatenated conditioning vector feeds either the PQC (quantum run) or MLP (classical baseline) to generate latent codes.

## Quantum Latent Generator (PQC)
- Implemented with PennyLane using the `lightning.qubit` backend by default.
- **Encoding layer:** Data re-uploading via parameterized rotations (RX, RY, RZ) to inject conditioning values into qubits.
- **Ansatz layers:** Alternating entanglers (CZ ladder or all-to-all) and trainable single-qubit rotations repeated `L` times.
- **Measurement:** Expectation values of Pauli-Z operators yield a latent vector (dimension equals number of measured qubits).
- **Regularization:** Penalty on shot noise variance to encourage informative latents.

## Classical Latent Generator (Baseline)
- Width/depth matched to PQC parameter count.
- Uses sinusoidal activations to avoid over-smoothing.
- Outputs same latent dimensionality as PQC for decoder parity.

## Shared Decoder
- **Input:** Latent vector reshaped into channel maps.
- **Architecture:** Lightweight deconvolutional stack (ResNet blocks + pixel shuffle) producing 32×32×3 tiles.
- **Skip modulation:** Adaptive instance normalization driven by palette statistics to infuse color cues.

## Discriminator
- PatchGAN-style critic operating on 32×32 tiles.
- Conditional projections incorporate class and palette embeddings to stabilize adversarial training.

## Training Loop
1. Sample batch of textures and palettes.
2. Encode conditioning, generate latents (quantum or classical).
3. Decode to synthetic textures.
4. Update discriminator and generator with identical optimizers/hyperparameters across runs.
5. Log latent statistics, spectral energy, and palette adherence for later comparison.

## Experiment Parity Controls
- Shared random seeds and noise schedules.
- Identical decoder/discriminator checkpoints initialised from the same weights.
- Gradient clipping and optimizer states synchronized between runs to ensure fairness.
