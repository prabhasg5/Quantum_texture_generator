## Quantum Texture Generator

This project trains a quantum-assisted GAN (QGAN) that invents novel textile patterns.  Each subfolder inside `dataset/` is treated as a class (e.g., `banded`, `blotchy`, …).  The generator consumes a target class chosen by the user and produces textures that draw inspiration from the class without copying any training sample.  Creativity comes from an entangled parameterized quantum circuit that perturbs the latent space before the classical decoder renders the final texture.

### 1. Environment Setup

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> The cleaner script uses `requirements_cleaner.txt`; the QGAN training stack requires `requirements.txt` (PyTorch, PennyLane, LPIPS, torchmetrics, scikit-learn, matplotlib, seaborn, PyYAML).

### 2. Clean the Dataset (optional but recommended)

```bash
python clean_dataset.py dataset/ --output dataset_cleaned/
mv dataset dataset_raw
mv dataset_cleaned dataset
```

### 3. Train the QGAN

The default config (`configs/qgan.yaml`) already sets `num_workers=8` and `persistent_workers=true` to fully utilize a 16 GB MacBook Air.

```bash
python -m src.training.train_qgan --config configs/qgan.yaml
```

Artifacts:

- `runs/qgan_fashion/` – checkpoints (`checkpoint_epoch_*.pt`) plus per-epoch class grids.
- `reports/qgan_fashion/` – latest class grid, loss curves, LPIPS histogram, feature coverage plot, `metrics.json` (FID, LPIPS novelty, feature coverage).
- `reports/qgan_fashion/loss_history.pkl` – pickled `{"generator": [...], "discriminator": [...]}` arrays updated every reporting pass.
- `reports/qgan_fashion/epochs/epoch_XXXX/` – frozen snapshots of each epoch’s `loss_curve.png`, `metrics.json`, `lpips_hist.png`, and `feature_coverage.png` so you can audit training progress step-by-step.
- Metrics now also log `colorfulness.mean|min|max` and the training curves include a `color_std` trace so you can spot grayscale collapse quickly.

### 4. Generate Inspiration Textures on Demand

Pick any class name that matches the dataset subfolders.

```bash
python generate_textures.py \
	--checkpoint runs/qgan_fashion/checkpoint_epoch_60.pt \
	--dataset-root dataset \
	--class-name blotchy \
	--num-samples 12 \
	--out blotchy_inspirations.png
```

Omit `--class-name` to create a grid covering every class (useful for design reviews).

### 5. Metrics & Visual Reporting

During `trainer.fit()` the following are computed automatically every `report_every` epochs (defaults to 1, so once per epoch) and stored both as running “latest” files and immutable `reports/.../epochs/epoch_XXXX/` snapshots:

- **FID** (torchmetrics) – realism vs. training distribution.
- **LPIPS-to-nearest-train** – novelty; the histogram shows how far generations drift from their closest reference texture.
- **Feature-space coverage** – EfficientNet feature radii vs. generator reach, plotted with coverage percentage.
- **Colorfulness stats** – mean/min/max per-sample channel std from generated batches to ensure the model doesn’t drift to grayscale.

All plots live inside `reports/qgan_fashion/`, with time-stamped copies in `reports/qgan_fashion/epochs/*`. Adjust `training.report_every` in `configs/qgan.yaml` if you want to thin out expensive metric passes.

### 6. Training Stability Tips

- Use `training.lr_generator` and `training.lr_discriminator` to decouple optimizers; the default slows the discriminator slightly.
- `training.novelty_lambda` controls the diversity regularizer, while `training.color_penalty_weight`/`color_min_std` enforce a minimum per-sample channel variance to fight grayscale collapse. Set the weight to `0.0` to disable the penalty.

### 6. Notes on Quantum Creativity

- The generator concatenates latent noise, class embeddings, and a PQC style vector (default wires = 6, layers = 2) simulated via PennyLane’s `default.qubit`.  Increase the number of wires or layers inside `configs/qgan.yaml` for richer entanglement (training time rises superlinearly).
- A diversity regularizer maximizes the pairwise distance between textures inside each batch, discouraging memorization.
- LPIPS-based novelty and feature coverage scores let you quantify “inspiration” while keeping FID in check.

### 7. Troubleshooting

- **Memory pressure**: reduce `batch_size` or `image_size` in the config.  Keep `num_workers` at 8 for best pipeline throughput on Apple silicon.
- **PennyLane speed**: set `model.quantum_wires` ≤ 8 for the default simulator.  Larger wire counts may require `lightning.qubit` if installed.
- **Metrics runtime**: lower `fid_samples`, `lpips_samples`, and `coverage_samples` to shorten the evaluation tail.
