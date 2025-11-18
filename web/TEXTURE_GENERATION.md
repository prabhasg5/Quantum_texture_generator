# Quantum Texture Generation System

## Overview
The texture generation page uses the trained Hybrid-QGAN to generate realistic textures with AI-powered color palette extraction and concept word generation.

## Features

### 1. **Quantum Texture Generation**
- Uses the trained quantum generator from `runs/qgan_fashion/checkpoint_latest.pt`
- Generates 1-12 texture samples per request
- Supports all 23 texture classes from the training dataset
- Real-time generation with visual feedback

### 2. **Automatic Color Palette Extraction**
- Uses K-means clustering (scikit-learn) to extract 6 dominant colors
- Filters out very dark/light pixels for more interesting palettes
- Returns hex color codes for easy copy-paste
- Click on any color swatch to copy to clipboard

### 3. **AI-Generated Concept Words**
- Rule-based concept generation system with 5 categories:
  - **Texture-specific concepts**: Based on texture class (e.g., "flowing", "organic" for swirly)
  - **Brightness analysis**: "bright", "dark", "moody" based on color luminance
  - **Saturation analysis**: "vibrant", "muted", "bold" based on color intensity
  - **Hue analysis**: "warm", "cool", "natural" based on color temperature
- Returns 5 unique concept words per generation
- Can be upgraded to LLM-based generation in the future

### 4. **Generation History**
- Stores all generations in SQLite database
- Displays recent 50 generations with timestamps
- Tracks texture class, sample count, and metadata
- Associated with user projects

## Technical Architecture

### Backend Components

#### `generation_utils.py`
```python
class TextureGenerator:
    - _load_generator(): Loads checkpoint with weights_only=False
    - generate_textures(): Main generation pipeline
    - _extract_color_palette(): K-means clustering for colors
    - _generate_concepts(): Rule-based concept generation
    - _get_color_concepts(): Color property analysis
```

#### API Endpoints
- `GET /api/texture-classes`: Returns available texture types
- `POST /api/generate`: Generates textures with color palette and concepts
- `GET /api/generations/history`: Returns user's generation history

### Frontend Components

#### `dashboard.html`
- Texture class selector (dropdown + sidebar)
- Sample count slider (1-12)
- Generate button with loading states
- Results display with image grid, color palette, concept tags
- Generation history cards

#### `dashboard.js`
- Async texture generation with fetch API
- Real-time UI updates
- Clipboard copy functionality for colors/concepts
- Smooth scrolling to results
- Toast notifications

#### `dashboard.css`
- Neon cyan/magenta theme matching landing page
- Glassmorphism effects
- Responsive grid layouts
- Hover effects and animations

## Usage Flow

1. **Sign In/Sign Up** on landing page
2. **Auto-redirect** to dashboard
3. **Select texture class** from dropdown or sidebar
4. **Adjust sample count** with slider (default: 6)
5. **Click "Generate Textures"** button
6. **View results**:
   - Generated texture grid (PNG)
   - 6 dominant color swatches with hex codes
   - 5 AI-generated concept keywords
7. **Download image** with timestamp filename
8. **Copy colors/concepts** by clicking on them
9. **View history** of past generations

## Concept Word Database

Predefined concepts for each texture class in `TEXTURE_CONCEPTS`:

```python
{
    'banded': ['striped', 'layered', 'rhythmic', 'parallel', 'flowing', 'gradient'],
    'blotchy': ['organic', 'natural', 'irregular', 'patchy', 'mottled', 'varied'],
    'bumpy': ['textured', 'dimensional', 'tactile', 'raised', 'rugged', 'relief'],
    # ... 23 classes total
}
```

## Color Analysis Rules

### Brightness-based Concepts
- `brightness > 200`: "bright", "light", "airy"
- `brightness < 80`: "dark", "moody", "mysterious"
- `80 ≤ brightness ≤ 200`: "balanced", "harmonious"

### Saturation-based Concepts
- `saturation > 100`: "vibrant", "bold", "energetic"
- `saturation < 50`: "muted", "subtle", "calm"

### Hue-based Concepts
- Red-dominant: "warm", "passionate"
- Blue-dominant: "cool", "serene"
- Green-dominant: "natural", "fresh"

## Performance Optimization

- **Lazy loading**: Generator initialized on first request
- **Checkpoint caching**: Model loaded once and reused
- **Device selection**: CPU fallback for MPS compatibility
- **Pixel sampling**: K-means on 10k pixels max for speed
- **Session persistence**: User state maintained across requests

## Future Enhancements

1. **LLM Integration**: Replace rule-based concepts with GPT-4/Claude API
2. **Style Transfer**: Apply generated textures to user images
3. **Batch Export**: Download multiple generations as ZIP
4. **Custom Palettes**: Allow manual color palette specification
5. **Fine-tuning Interface**: Adjust generator parameters per request
6. **Collaborative Projects**: Share generations with team members
7. **Texture Interpolation**: Blend between two texture classes
8. **High-Resolution Export**: Generate 512x512 or 1024x1024 outputs

## Dependencies

```
Flask>=3.0.0
Flask-SQLAlchemy>=3.1.1
scikit-learn>=1.3.0  # Color palette extraction
Pillow>=10.0.0       # Image processing
torch>=2.0.0         # Generator inference
pennylane>=0.30.0    # Quantum circuits
```

## Troubleshooting

### "Checkpoint not found" Error
- Ensure `runs/qgan_fashion/checkpoint_latest.pt` exists
- Check that training has been run and checkpoints saved

### "Unknown texture class" Error
- Verify class name matches dataset directory names
- Class names are lowercase with hyphens (e.g., "polka-dotted")

### Generation Takes Too Long
- Reduce `num_samples` (default: 6)
- Check CPU/GPU availability
- Quantum circuit execution can be slow on first run (JIT compilation)

### Color Palette Empty/Incorrect
- Ensure generated images have color variance
- Check if images are properly denormalized (0-255 range)
- Adjust `n_colors` parameter in `_extract_color_palette`

## API Response Format

### `/api/generate` Response
```json
{
  "success": true,
  "image": "data:image/png;base64,...",
  "color_palette": ["#ff5733", "#33ff57", "#3357ff", "#f4a742", "#9b59b6", "#1abc9c"],
  "concept_words": ["flowing", "vibrant", "warm", "organic", "bold"],
  "generation_id": 42
}
```

### `/api/texture-classes` Response
```json
{
  "classes": ["banded", "blotchy", "bumpy", ..., "zigzagged"]
}
```

### `/api/generations/history` Response
```json
{
  "generations": [
    {
      "id": 42,
      "texture_class": "swirly",
      "num_samples": 6,
      "created_at": "2025-11-18T07:14:43.123456",
      "metadata": {
        "color_palette": [...],
        "concept_words": [...]
      },
      "project_name": "Default"
    }
  ]
}
```

## Development Notes

- Generator runs on CPU by default (set `USE_MPS=1` env var for Apple Silicon)
- Database auto-creates tables on first run
- Sessions expire after 7 days
- All generations saved to "Default" project initially
- Color palette uses frequency-sorted K-means clusters
- Concept words randomized from combined texture+color concepts
