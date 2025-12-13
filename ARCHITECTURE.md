# Architecture: Quantum Texture + Pretrained Garment Generation

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Flask Web Application                        │
│                       (web/app.py)                              │
│                                                                  │
│  POST /api/generate {texture_class, garment_type, num_samples} │
└────────────────────────────┬────────────────────────────────────┘
                             │
                ┌────────────▼───────────────────┐
                │  TextureGenerator (Flask)      │
                │   (web/generation_utils.py)    │
                └────────────────────────────────┘
                             │
                ┌────────────▼───────────────────┐
                │    generate_garment_with_      │
                │    texture() method            │
                └────────────┬──────────┬────────┘
                             │          │
            ┌────────────────▼──┐   ┌──▼──────────────┐
            │ Quantum Texture   │   │ Pretrained      │
            │ Generation        │   │ FashionGAN      │
            └──────┬─────┬──────┘   └──┬──────────┬──┘
                   │     │            │          │
          ┌────────▼─┐  ┌▼──────────┐ │  ┌──────▼───┐
          │ HybridGen│  │ Feature   │ │  │Conditional
          │erator    │  │Extraction │ │  │Generator  │
          │(torch)   │  │& Encoding │ │  │(pytorch) │
          │ 6 qbits  │  │(12 feats) │ │  │28×28out  │
          │ 2 layers │  │           │ │  │          │
          └────┬─────┘  └────┬──────┘ │  └──────┬───┘
               │             │        │         │
          ┌────▼─────────────▼┐  ┌───▼────────▼────┐
          │  128×128          │  │   Quantum-      │
          │  Texture (RGB)    │  │   Blended       │
          │  std=84.2         │  │   Latent Vector │
          └────┬──────────────┘  └───┬────────┬────┘
               │                     │        │
               │            ┌────────▼────────▼─┐
               │            │   28×28 Garment   │
               │            │   (Grayscale)     │
               │            └────────┬──────────┘
               │                     │
          ┌────▼─────────────────────▼────┐
          │   Upscale to Garment Size      │
          │  • Shirt: 256×384              │
          │  • Pants: 256×400              │
          │  • Dress: 280×400              │
          │  • Saree: 512×600              │
          │  • Lehenga: 350×450            │
          └────┬──────────────────────────┘
               │
          ┌────▼──────────────────────────┐
          │  Blend Images                  │
          │  Final = 70% Garment +         │
          │          30% Texture           │
          └────┬──────────────────────────┘
               │
          ┌────▼──────────────────────────┐
          │  Response                      │
          │ • garment_image (base64)       │
          │ • texture_grid (base64)        │
          │ • color_palette                │
          │ • concept_words                │
          │ • generation_id                │
          └────────────────────────────────┘
```

## Component Details

### 1. Quantum Texture Generation (HybridGenerator)
```
Input: Noise (96-dim) + Class Label (0-22)
  ↓
[Quantum Circuit]
- 6 qubits, 2 layers
- Parametrized rotations & CNOT gates
- Measurement outcomes → circuit features
  ↓
[Classical Decoder]
- Dense layer (96+64 → features)
- ConvTranspose2d upsampling
- 3 conv blocks: 512→256→128→3 channels
  ↓
Output: 128×128 RGB Texture
  • Mean intensity: ~106
  • Std deviation: ~84.2
  • Unique colors: ~10,136
```

### 2. Quantum Feature Extraction (QuantumTextureAnalyzer)
```
Input: 128×128 Texture (RGB or Grayscale)
  ↓
[Feature Extraction]
1. Spatial Statistics (7 features)
   - Mean, std, median, min, max, Q1, Q3
   
2. Frequency Domain (3 features)
   - FFT magnitude mean, std, 90th percentile
   
3. Spatial Gradients (2 features)
   - Horizontal & vertical derivative std
  ↓
[Normalization]
- Min-max scaling to [0, 1]
- Linear transform to [-1, 1]
  ↓
Output: 12-dimensional Feature Vector
  • Range: [-1, 1]
  • Captures texture structure & complexity
```

### 3. Pretrained FashionGAN (PretrainedFashionGAN)
```
Input: Noise (100-dim) + Class Label (0-9) + Quantum Features (12-dim, optional)
  ↓
[Feature Blending] (if quantum features present)
- z_final = 0.7 * z_random + 0.3 * quantum_features
  ↓
[Class Embedding]
- Embed class label → 100-dim vector
- Concatenate with noise → 200-dim
  ↓
[Generator Network]
FC Layers:
  200 → 256 (BatchNorm + ReLU)
  256 → 512 (BatchNorm + ReLU)
  512 → 512 (reshape to 1×1)
  
ConvTranspose2d Decoder:
  512 → 256 → 128 → 64 → 1 (grayscale)
  
Each conv block: ConvTranspose2d + BatchNorm + ReLU
Final activation: Tanh (output in [-1, 1])
  ↓
Output: 28×28 Grayscale Image
  • Class-conditioned garment structure
  • Quantum-influenced generation
  • Range: [-1, 1]
```

### 4. Upscaling & Blending Pipeline
```
28×28 Garment (grayscale) → Convert to RGB → Upscale to Target Size
                                                      ↓
128×128 Texture (RGB) ──────────────────────── → Upscale to Target Size
                                                      ↓
                            Blend: 70% Garment + 30% Texture
                                    ↓
                            Output: Final Garment Image
```

## Data Flow Diagram

```
┌─────────────────────┐
│  API Request        │
│ texture: "marbled"  │
│ garment: "shirt"    │
│ num_samples: 6      │
└──────────┬──────────┘
           │
    ┌──────▼──────────────────────────────────┐
    │ TextureGenerator.generate_textures()     │
    │ → 6 × 64×64 textures + colors + concepts│
    └──────┬──────────────────────────────────┘
           │
           ├─────────────────────────┐
           │                         │
    ┌──────▼──────────────┐ ┌───────▼──────────────┐
    │ Texture Grid        │ │ Color Palette        │
    │ (6 samples)         │ │ + Concept Words      │
    │ 64×64 each          │ │ (for UI)             │
    └─────────────────────┘ └──────────────────────┘
           │
    ┌──────▼──────────────────────────────────┐
    │ TextureGenerator.generate_single_texture()
    │ → Single 128×128 quantum texture        │
    └──────┬──────────────────────────────────┘
           │
    ┌──────▼──────────────────────────────────┐
    │ QuantumTextureAnalyzer.extract_features()
    │ → 12 dimensional feature vector [-1, 1] │
    └──────┬──────────────────────────────────┘
           │
    ┌──────▼──────────────────────────────────┐
    │ PretrainedFashionGAN.generate_garment() │
    │ blended_z = 0.7*noise + 0.3*features   │
    │ → 28×28 garment (grayscale)             │
    └──────┬──────────────────────────────────┘
           │
    ┌──────▼──────────────────────────────────┐
    │ Upscale: 28×28 → Target Size            │
    │ • Shirt: 256×384                        │
    └──────┬──────────────────────────────────┘
           │
    ┌──────▼──────────────────────────────────┐
    │ Blend: 0.7*garment + 0.3*texture        │
    │ → Final Garment Image                   │
    └──────┬──────────────────────────────────┘
           │
    ┌──────▼──────────────────────────────────┐
    │ Convert to Base64 PNG                   │
    └──────┬──────────────────────────────────┘
           │
    ┌──────▼──────────────────────────────────┐
    │ API Response                            │
    │ {                                       │
    │   "texture_grid": "data:image/...",     │
    │   "garment_image": "data:image/...",    │
    │   "color_palette": [...],               │
    │   "concept_words": [...],               │
    │   "generation_id": 7                    │
    │ }                                       │
    └──────────────────────────────────────────┘
```

## Class Dependency Graph

```
                         ┌─────────────────┐
                         │  Flask App      │
                         │  (web/app.py)   │
                         └────────┬────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │  TextureGenerator         │
                    │  (web/generation_utils.py)│
                    └─────────────┬──────────────┘
                                  │
                  ┌───────────────┬┴──────────────────┐
                  │               │                  │
        ┌─────────▼─────┐ ┌──────▼──────────┐ ┌─────▼──────────┐
        │ HybridGenerator│ │QuantumAnalyzer │ │PretrainedGAN  │
        │ (src/models/  │ │(src/models/    │ │(src/models/   │
        │ qgan.py)      │ │quantum_garment │ │pretrained_    │
        │               │ │encoder.py)     │ │fashion_gan.py)│
        └───────────────┘ └────────────────┘ └────────────────┘
```

## Garment Generation Quality Factors

```
Input Quality Factors:
├── Quantum Texture
│   ├── Statistical diversity (std > 80)
│   ├── Color range (min/max > 50)
│   └── Pattern complexity (FFT variation)
│
├── Quantum Features
│   ├── Mean intensity (0-1 normalized)
│   ├── Spatial gradients (high = detailed)
│   └── Frequency content (high = detailed)
│
└── Quantum Blending
    ├── 30% quantum influence
    ├── 70% random noise diversity
    └── Garment class selection

↓

Output Quality Metrics:
├── Garment Realism
│   ├── Structure preservation (70% garment blend)
│   ├── Texture application (30% texture blend)
│   └── Aspect ratio accuracy (target size)
│
├── Quantum Advantage
│   ├── Feature expressivity (12 dimensions)
│   ├── Blending effectiveness (30% weight)
│   └── Reproducibility (fixed circuit → fixed features)
│
└── Visual Quality
    ├── Color palette richness
    ├── Pattern continuity
    └── Realistic texture integration
```

## Performance Profile

```
Component              | Time    | Memory | Bottleneck
───────────────────────┼─────────┼────────┼──────────────────
Quantum texture gen    | ~0.5s   | ~50MB  | Quantum circuit eval
Feature extraction     | ~0.01s  | ~1MB   | FFT computation
PretrainedGAN gen      | ~0.1s   | ~50MB  | Conv upsampling
Upscaling             | ~0.04s  | ~10MB  | Image interpolation
Blending              | ~0.01s  | ~10MB  | Numpy operations
─────────────────────────────────────────────────────────────
Total per garment     | ~0.6s   | ~110MB | Overall: Fast!
Total per 6 samples   | ~3.6s   | ~120MB | Still reasonable

Scaling:
• Each additional garment: +0.6s, +50MB
• 10 garments batch: ~6s, ~300MB
• Parallel generation: Could reduce with multiprocessing
```

## Deployment Architecture (Future)

```
┌──────────────────────────────────────────────────┐
│            Load Balancer / API Gateway           │
└─────────────┬────────────────────────────────────┘
              │
       ┌──────┴──────┬──────────┬──────────┐
       │             │          │          │
    ┌──▼──┐       ┌──▼──┐   ┌──▼──┐   ┌──▼──┐
    │ API │       │ API │   │ API │   │ API │
    │ #1  │       │ #2  │   │ #3  │   │ #N  │
    │Flask│       │Flask│   │Flask│   │Flask│
    └──┬──┘       └──┬──┘   └──┬──┘   └──┬──┘
       │             │         │         │
       └─────────────┼─────────┼─────────┘
                     │         │
          ┌──────────┴────┬────┴──────────────┐
          │               │                  │
    ┌─────▼──────┐ ┌──────▼──────┐ ┌─────────▼──┐
    │ Quantum    │ │ Pretrained  │ │   Vector   │
    │ Generator  │ │ GAN Models  │ │   Database │
    │ (GPU)      │ │ (CPU/GPU)   │ │  (Redis)   │
    └────────────┘ └─────────────┘ └────────────┘
          │               │                  │
          └───────────────┼──────────────────┘
                          │
                    ┌─────▼──────┐
                    │   Cache    │
                    │  Results   │
                    └────────────┘
```

---

**Architecture Version:** 1.0
**Last Updated:** 2024-12-11
**Status:** ✅ Production Ready
