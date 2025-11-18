"""Utilities for texture generation, color palette extraction, and concept generation."""
from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


# Concept words database for texture types
TEXTURE_CONCEPTS = {
    'banded': ['striped', 'layered', 'rhythmic', 'parallel', 'flowing', 'gradient'],
    'blotchy': ['organic', 'natural', 'irregular', 'patchy', 'mottled', 'varied'],
    'bumpy': ['textured', 'dimensional', 'tactile', 'raised', 'rugged', 'relief'],
    'checkered': ['geometric', 'grid', 'orderly', 'structured', 'tiled', 'chess'],
    'cracked': ['weathered', 'aged', 'broken', 'fragmented', 'distressed', 'vintage'],
    'dotted': ['spotted', 'speckled', 'playful', 'scattered', 'polka', 'sprinkled'],
    'flaky': ['layered', 'peeling', 'scaled', 'textured', 'rough', 'stratified'],
    'flecked': ['spattered', 'spotted', 'sprinkled', 'variegated', 'mixed', 'dusted'],
    'freckled': ['dotted', 'natural', 'random', 'organic', 'spotted', 'delicate'],
    'frilly': ['decorative', 'ornate', 'elegant', 'ruffled', 'fancy', 'flowing'],
    'grooved': ['lined', 'channeled', 'ridged', 'carved', 'etched', 'engraved'],
    'lined': ['striped', 'parallel', 'structured', 'directional', 'ruled', 'aligned'],
    'marbled': ['swirled', 'veined', 'elegant', 'luxurious', 'flowing', 'organic'],
    'paisley': ['ornate', 'flowing', 'decorative', 'bohemian', 'intricate', 'curvy'],
    'polka-dotted': ['playful', 'retro', 'cheerful', 'spotted', 'regular', 'fun'],
    'potholed': ['irregular', 'weathered', 'damaged', 'rough', 'cratered', 'worn'],
    'ribbed': ['ridged', 'textured', 'lined', 'corduroy', 'structured', 'raised'],
    'sprinkled': ['scattered', 'random', 'dusted', 'peppered', 'light', 'airy'],
    'stained': ['blotchy', 'aged', 'vintage', 'weathered', 'mottled', 'patina'],
    'striped': ['linear', 'parallel', 'bold', 'directional', 'banded', 'zebra'],
    'swirly': ['flowing', 'curvy', 'dynamic', 'spiral', 'whirling', 'organic'],
    'wavy': ['flowing', 'undulating', 'rippled', 'sinuous', 'fluid', 'oceanic'],
    'zigzagged': ['angular', 'sharp', 'dynamic', 'chevron', 'electric', 'bold']
}


class TextureGenerator:
    """Handles texture loading from dataset, color extraction, and concept generation."""
    
    def __init__(self, checkpoint_path: Path = None, device: str = 'cpu'):
        """Initialize with dataset path."""
        self.dataset_root = Path(__file__).parent.parent / 'dataset'
        self.class_names = None
        self._load_classes()
    
    def _load_classes(self):
        """Load available texture classes from dataset directory."""
        if self.dataset_root.exists():
            self.class_names = sorted([d.name for d in self.dataset_root.iterdir() if d.is_dir()])
        else:
            # Fallback to default texture classes
            self.class_names = list(TEXTURE_CONCEPTS.keys())
    
    def generate_textures(self, texture_class: str, num_samples: int = 6) -> Tuple[Image.Image, List[str], List[str]]:
        """
        Load sample textures from dataset and extract color palette and concepts.
        
        Returns:
            tuple: (grid_image, color_palette_hex, concept_words)
        """
        if texture_class not in self.class_names:
            raise ValueError(f"Unknown texture class: {texture_class}")
        
        # Load images from dataset
        dataset_root = Path(__file__).parent.parent / 'dataset' / texture_class
        
        if not dataset_root.exists():
            raise ValueError(f"Dataset directory not found: {dataset_root}")
        
        # Get all image files
        image_files = list(dataset_root.glob('*.jpg')) + list(dataset_root.glob('*.png'))
        
        if not image_files:
            raise ValueError(f"No images found in {dataset_root}")
        
        # Randomly sample images
        num_samples = min(num_samples, len(image_files))
        sampled_files = random.sample(image_files, num_samples)
        
        # Load and resize images to 64x64
        images = []
        for img_path in sampled_files:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((64, 64), Image.Resampling.LANCZOS)
            images.append(img)
        
        # Create grid
        grid_image = self._create_image_grid(images, nrow=3, padding=2)
        
        # Convert to numpy for color extraction
        grid_np = np.array(grid_image)
        
        # Extract color palette
        color_palette = self._extract_color_palette(grid_np)
        
        # Generate concept words
        concept_words = self._generate_concepts(texture_class, color_palette)
        
        return grid_image, color_palette, concept_words
    
    def _create_image_grid(self, images: List[Image.Image], nrow: int = 3, padding: int = 2) -> Image.Image:
        """Create a grid of images."""
        if not images:
            raise ValueError("No images provided")
        
        # Calculate grid dimensions
        n_images = len(images)
        ncol = nrow
        nrow_actual = (n_images + ncol - 1) // ncol
        
        # Get image size (assuming all images are same size)
        img_width, img_height = images[0].size
        
        # Calculate grid size
        grid_width = ncol * img_width + (ncol + 1) * padding
        grid_height = nrow_actual * img_height + (nrow_actual + 1) * padding
        
        # Create white background
        grid = Image.new('RGB', (grid_width, grid_height), color=(255, 255, 255))
        
        # Paste images
        for idx, img in enumerate(images):
            row = idx // ncol
            col = idx % ncol
            x = col * img_width + (col + 1) * padding
            y = row * img_height + (row + 1) * padding
            grid.paste(img, (x, y))
        
        return grid
    
    def _extract_color_palette(self, image_np: np.ndarray, n_colors: int = 6) -> List[str]:
        """Extract dominant colors from image using K-means clustering."""
        # Reshape image to list of pixels
        pixels = image_np.reshape(-1, 3)
        
        # Remove very dark and very light pixels to get more interesting colors
        brightness = pixels.mean(axis=1)
        mask = (brightness > 30) & (brightness < 225)
        filtered_pixels = pixels[mask]
        
        if len(filtered_pixels) < 100:
            filtered_pixels = pixels
        
        # Sample pixels for faster clustering
        if len(filtered_pixels) > 10000:
            indices = np.random.choice(len(filtered_pixels), 10000, replace=False)
            filtered_pixels = filtered_pixels[indices]
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(filtered_pixels)
        
        # Get cluster centers and sort by frequency
        centers = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        unique, counts = np.unique(labels, return_counts=True)
        
        # Sort colors by frequency
        sorted_indices = np.argsort(-counts)
        sorted_colors = centers[sorted_indices]
        
        # Convert to hex
        hex_colors = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in sorted_colors]
        
        return hex_colors
    
    def _generate_concepts(self, texture_class: str, color_palette: List[str]) -> List[str]:
        """Generate concept words based on texture type and colors."""
        # Get base concepts from texture type
        base_concepts = TEXTURE_CONCEPTS.get(texture_class, ['textured', 'unique', 'artistic'])
        
        # Add color-inspired concepts
        color_concepts = self._get_color_concepts(color_palette)
        
        # Combine and randomize
        all_concepts = base_concepts + color_concepts
        random.shuffle(all_concepts)
        
        # Return top 5 unique concepts
        return list(dict.fromkeys(all_concepts))[:5]
    
    def _get_color_concepts(self, color_palette: List[str]) -> List[str]:
        """Generate concepts based on dominant colors."""
        concepts = []
        
        for hex_color in color_palette[:3]:  # Analyze top 3 colors
            rgb = self._hex_to_rgb(hex_color)
            r, g, b = rgb
            
            # Analyze color properties
            brightness = (r + g + b) / 3
            saturation = max(r, g, b) - min(r, g, b)
            
            # Brightness-based concepts
            if brightness > 200:
                concepts.extend(['bright', 'light', 'airy'])
            elif brightness < 80:
                concepts.extend(['dark', 'moody', 'mysterious'])
            else:
                concepts.extend(['balanced', 'harmonious'])
            
            # Saturation-based concepts
            if saturation > 100:
                concepts.extend(['vibrant', 'bold', 'energetic'])
            elif saturation < 50:
                concepts.extend(['muted', 'subtle', 'calm'])
            
            # Hue-based concepts
            if r > g and r > b:
                if r - max(g, b) > 50:
                    concepts.extend(['warm', 'passionate'])
            elif b > r and b > g:
                if b - max(r, g) > 50:
                    concepts.extend(['cool', 'serene'])
            elif g > r and g > b:
                if g - max(r, b) > 50:
                    concepts.extend(['natural', 'fresh'])
        
        return concepts
    
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def get_available_classes(self) -> List[str]:
        """Return list of available texture classes."""
        return self.class_names
