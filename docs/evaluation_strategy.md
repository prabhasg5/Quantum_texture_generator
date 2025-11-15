# Evaluation Strategy

## Quantitative Metrics
- **Novelty Index:** Compare generated tiles to training data via LPIPS + feature distance across a ResNet encoder. Higher is better.
- **Coverage Score:** Fraction of texture classes whose feature clusters receive generated samples within a Mahalanobis threshold.
- **Spectral Richness:** FFT-based high-frequency energy ratio highlighting intricate details.
- **Palette Fidelity:** Lab color distance between requested palette and dominant colors in output.
- **Global Correlation Measure:** Compute spatial autocorrelation statistics (Moran's I) to quantify long-range structure.

## Latent Space Analysis
- t-SNE/UMAP projections of latent vectors for both models to visualize coverage diversity.
- Mutual information between latent dimensions and conditioning variables to study disentanglement.
- Entropy of latent activation histograms to detect mode collapse.

## Qualitative Review
- Curated mosaic panels contrasting quantum vs classical outputs per conditioning prompt.
- Designer annotation tool for tagging surprising vs expected patterns.
- Style fusion experiments mixing palettes and classes to assess creativity.

## User Study Outline
1. Recruit 5–10 fashion designers.
2. Provide blinded sets of textures from quantum and classical models per prompt.
3. Collect ratings on novelty, inspiration value, and palette fit.
4. Analyze preference statistics (binomial tests) to confirm quantum advantage.

## Reporting
- `src/evaluation/report.py` compiles metrics, plots, and user feedback into a PDF/HTML report.
- Maintain experiment logs in `outputs/experiments/<timestamp>/` (ignored by Git).
- Document insights and anomalies in `docs/experiments/<run_id>.md` for knowledge sharing.
