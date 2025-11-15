# Data Pipeline

## PTD Dataset Ingestion
- **Location:** Expect textures under `ptd/images/<class_name>/*.png` (32×32 or higher resolution).
- **Mirroring:** Maintain identical folder structure on both machines; adjust `configs/*.yaml` if different.
- **Splits:** Generate train/val/test splits with stratified sampling over texture classes.

## Preprocessing Steps
1. **Tile extraction:** Crop or downscale to 32×32 patches; optionally apply random crops for augmentation.
2. **Color normalization:** Convert to float tensors (`[0,1]`), optionally apply palette-preserving jitter.
3. **Palette extraction:** K-means or differentiable color clustering to derive 3–7 representative colors per tile for conditioning labels.
4. **Class encoding:** Map PTD directory names to integer IDs; store alongside palette embeddings.
5. **Caching:** Save preprocessed tensors and palette info as `.pt` or `.npz` for faster reloads.

## Palette Encoding
- **Option A:** Learnable palette encoder (MLP) over concatenated Lab color vectors.
- **Option B:** Statistical features (mean hue, saturation spread, complementary distance).
- **Hybrid:** Combine handcrafted stats with learnable layers for robustness.

## DataLoader Configuration
- Batch size configurable (default 32) with deterministic shuffling per epoch.
- Provide `collate_fn` that returns:
  - Texture tensor (C×32×32)
  - Class ID
  - Palette embedding
  - Raw palette colors (for qualitative inspection)
- Support distributed sampling when running multi-device experiments.

## Augmentations (Applied Consistently to Both Models)
- Random hue shift within ±5°.
- Elastic distortions for increased structural variety.
- Frequency-domain dropout (mask random high/low bands) to encourage decoder resilience.

## Metadata Tracking
- Store palette hashes to detect duplicates and measure novelty.
- Log spectral descriptors per sample for later coverage analysis.

## Integration Points
- `src/data/dataset.py` handles dataset creation and transform configuration.
- `src/utils/palette.py` provides palette extraction and encoding utilities.
- `configs/*.yaml` exposes knobs for augmentations, caching, and palette modes.
