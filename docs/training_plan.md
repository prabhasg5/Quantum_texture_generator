# Training Plan

## Objectives
- Train quantum and classical generators under identical settings to isolate the impact of the PQC latent space.
- Capture metrics reflecting novelty, diversity, palette adherence, and user preference rather than pure realism.

## Core Hyperparameters (Shared)
- Image size: 32×32
- Batch size: 32 (tune 16–64 depending on memory)
- Optimizer: AdamW (β1=0.5, β2=0.99, weight decay 1e-4)
- Learning rate: 3e-4 for generator and discriminator
- Training steps: 100k iterations or early stop when novelty plateaus
- Losses: adversarial (hinge), reconstruction (L1), spectral contrastive loss, palette consistency term

## Quantum-Specific Settings
- Qubits: 8–12 depending on latent dimension
- PQC depth: 4–6 layers with data re-uploading
- Shots: 1,024 during training, 4,096 for evaluation renders
- Device: PennyLane `default.qubit` or `lightning.qubit`

## Classical Baseline Settings
- Layers: 6 fully connected layers with sine activations
- Hidden width: tuned to match PQC parameter count
- Latent dimension identical to quantum run

## Training Workflow
1. Initialize shared decoder and discriminator weights with the same seed.
2. Alternate generator and discriminator updates (1:1 ratio).
3. Synchronize random noise seeds between quantum and classical runs per batch.
4. Save checkpoints every 2k iterations, including optimizer states.
5. Log metrics via Weights & Biases or TensorBoard for both runs with distinct tags.

## Evaluation Cadence
- Every 5k steps: render 256 tiles per model for qualitative review, compute novelty metrics, update palette adherence charts.
- After training: run full evaluation suite (`src/evaluation/run_suite.py`) comparing quantum vs classical.
- Conduct designer review sessions with blinded samples.

## Parallelization Strategy
- Run quantum training locally on your MacBook.
- Friend runs classical training on their Mac with identical codebase/branch.
- Use branching workflow described in `README.md`; merge only after verifying both checkpoints.

## Future Optimizations
- Experiment with variational quantum natural gradient (VQNG) optimizers.
- Explore gradient-free updates (CMA-ES) if PQC gradients exhibit barren plateaus.
- Investigate shared latent adapters for cross-model style transfer.
