# Quantum Texture Generator

A hybrid quantum-classical pipeline that produces 32×32 texture tiles with genuinely novel structure for fashion designers. Designers pick a PTD texture class plus a 3–7 color palette; a parameterized quantum circuit (PQC) transforms this conditioning into a latent vector that a convolutional decoder turns into an image. A matched classical baseline (MLP in place of the PQC) is trained alongside to highlight the creative advantage of the quantum approach.

## Key Features
- PennyLane-based PQC latent generator introducing superposition and entanglement for diverse latent exploration.
- Shared classical decoder/discriminator stack for both quantum and classical variants, enabling controlled comparisons.
- Conditioning on PTD texture labels and designer color palettes for targeted yet surprising outputs.
- Evaluation toolkit to quantify novelty, coverage, high-frequency richness, and designer preference.

## Repository Structure
- `src/data/` — dataset loading, palette encoding, and batching utilities.
- `src/models/quantum/` — PQC latent generator modules built with PennyLane devices.
- `src/models/classical/` — parameter-matched MLP latent generator baseline.
- `src/models/common/` — shared decoder, discriminator, and embedding layers.
- `src/training/` — training loops, experiment orchestration, logging interfaces.
- `src/evaluation/` — novelty/diversity metrics and qualitative comparison tooling.
- `src/utils/` — configuration helpers, seeding, palette processing.
- `configs/` — YAML experiment specs for quantum and classical runs.
- `docs/` — architecture, data pipeline, training, and evaluation references.

## Dataset Placement
1. Download or mirror the PTD texture dataset locally.
2. Place the extracted contents inside `ptd/` at the repository root (`ptd/images/<class_name>/*.png`, `ptd/classes.txt`, `ptd/metafile.txt`).
3. `ptd/` is ignored by Git—never commit the raw textures. If the folder was accidentally staged before, run `git rm -r --cached ptd` once.
4. Update `configs/*.yaml` only if you keep the dataset somewhere else (e.g., external drive or shared network path).

## Environment Setup (macOS, Python 3.10–3.12 Recommended)
1. Create and activate a virtual environment:
	```bash
	python3 -m venv .venv
	source .venv/bin/activate
	pip install --upgrade pip
	```
2. Install core dependencies:
	```bash
	pip install -r requirements.txt
	```
3. (Optional) Install PennyLane Lightning-GPU if you have compatible hardware:
	```bash
	pip install pennylane-lightning[gpu]
	```
4. (Optional) For circuit cutting research, install `pennylane-qcut` from source (no PyPI wheel yet):
	```bash
	pip install git+https://github.com/PennyLaneAI/pennylane-qcut.git
	```
5. Verify setup:
	```bash
	python -m pennylane.about
	python -m src.training.dry_run --config configs/quantum.yaml
	```

## Running Training
- Quantum variant:
	```bash
	python -m src.training.run --config configs/quantum.yaml
	```
- Classical baseline:
	```bash
	python -m src.training.run --config configs/classical.yaml
	```

Training displays tqdm progress bars per epoch with live discriminator/generator losses. Every five epochs the loop:
- Writes generator samples to `outputs/<variant>/samples/epoch_XXX.png`.
- Updates `outputs/<variant>/loss_history.pkl` and plots `outputs/<variant>/plots/loss_curve_epoch_XXX.png`.

Checkpoints and logs remain under `outputs/<variant>/`. With `device: auto`, PyTorch chooses MPS/GPU/CPU automatically, and with `resume: true` the latest checkpoint is restored after any interruption.

## Parallel Training Workflow
1. **Prep main:** `git checkout main && git pull` on both machines.
2. **Quantum branch (you):**
	```bash
	git checkout -b quantum-run
	# implement & train quantum model
	git commit -am "Quantum training results"
	git push -u origin quantum-run
	```
3. **Classical branch (friend):**
	```bash
	git checkout -b classical-run
	# implement & train classical model
	git commit -am "Classical baseline results"
	git push -u origin classical-run
	```
4. **Merge strategy:**
	- Keep branches rebased on `main` to minimize conflicts (`git pull --rebase origin main`).
	- Exchange checkpoints via shared storage if needed.
	- Open pull requests or perform pairwise merges once both runs finish.

## Roadmap Snapshot
- [ ] Finalize data loaders and palette embeddings.
- [ ] Implement PQC latent generator (PennyLane) and baseline MLP.
- [ ] Build shared decoder/discriminator with spectral regularization.
- [ ] Create joint training harness and logging.
- [ ] Integrate novelty/diversity metrics plus qualitative review dashboards.
- [ ] Conduct designer feedback sessions.

See `docs/` for detailed architecture, training plans, and evaluation methodology. Contributions via issues or pull requests are welcome.