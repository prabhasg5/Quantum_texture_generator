"""Utilities for texture generation, color palette extraction, and concept generation."""
from __future__ import annotations

from io import BytesIO
import random
from pathlib import Path
from typing import Dict, List, Tuple
import os
import base64

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


class TextureAnalyzer:
    """Analyze texture images using Gemini to generate detailed descriptions.
    
    Uses Gemini's vision capabilities to describe texture patterns accurately.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            self._genai = None
            return
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._genai = genai
            # Use a fast model for text generation (good free tier)
            self.model = os.environ.get("GEMINI_TEXT_MODEL", "models/gemini-2.0-flash")
        except Exception:
            self._genai = None

    def analyze(self, image_base64: str) -> str:
        """Analyze a texture image and return a detailed description.
        
        Args:
            image_base64: Base64 encoded image (with or without data URL prefix)
        
        Returns:
            Detailed textual description of the texture pattern
        """
        if not self._genai:
            return ""
        
        try:
            # Decode the image
            payload = image_base64.split(',')[-1] if ',' in image_base64 else image_base64
            texture_bytes = base64.b64decode(payload)
            pil_img = Image.open(BytesIO(texture_bytes)).convert("RGB")
            
            # Build analysis prompt - IMPORTANT: exclude colors since user will choose their own
            prompt = (
                "Analyze this fabric/texture image. Describe ONLY the pattern structure, NOT the colors:\n"
                "1. The exact pattern type (stripes, swirls, dots, geometric, organic, marbled, etc.)\n"
                "2. Pattern characteristics (regular/irregular, dense/sparse, bold/subtle, flowing/rigid)\n"
                "3. Texture appearance (smooth, rough, woven, printed, gradient, layered, etc.)\n"
                "4. Pattern scale (fine details, medium, large bold patterns)\n\n"
                "DO NOT mention any colors - the user will apply their own color choice.\n"
                "Provide a concise 2-3 sentence description of the PATTERN STRUCTURE ONLY "
                "that could be used to recreate this texture pattern in any color. "
                "Focus only on shapes, lines, and texture structure."
            )
            
            model = self._genai.GenerativeModel(self.model)
            resp = model.generate_content([prompt, pil_img])
            
            # Extract text response
            if resp and resp.text:
                return resp.text.strip()
            return ""
        except Exception as e:
            print(f"Texture analysis failed: {e}")
            return ""


class GeminiRenderer:
    """Render garment images using Gemini API by applying a user-generated texture.

    Expects GEMINI_API_KEY in environment or passed explicitly.
    Returns base64 PNG (without data URL prefix).
    """

    # Image generation capable models (in preference order)
    IMAGE_GEN_MODELS = [
        "models/gemini-2.5-flash-image",
        "models/gemini-2.5-flash-image-preview", 
        "models/gemini-3-pro-image-preview",
        "models/gemini-2.0-flash-exp-image-generation",
    ]

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        # Default to the image generation model with better free tier
        self.model = model or os.environ.get("GEMINI_MODEL") or "models/gemini-2.5-flash-image"
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not set in environment or passed to GeminiRenderer.")

        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        self._genai = genai

        # Validate model availability; fallback to image-gen capable model if needed
        try:
            available = [m.name for m in genai.list_models() if "generateContent" in getattr(m, "supported_generation_methods", [])]
        except Exception:
            available = []

        if self.model not in available:
            # Try to find an image generation model
            chosen = None
            for img_model in self.IMAGE_GEN_MODELS:
                if img_model in available:
                    chosen = img_model
                    break
            if not chosen and available:
                chosen = available[0]
            if not chosen:
                raise RuntimeError("No Gemini models with generateContent are available to this API key.")
            self.model = chosen

    def render(self, texture_base64_png: str, garment: str, color_hex_or_name: str) -> str:
        import google.generativeai as genai
        model = genai.GenerativeModel(self.model)

        prompt = (
            f"Generate an image of a realistic fashion garment: {garment}. "
            f"Primary color: {color_hex_or_name}. "
            "Apply the provided fabric texture pattern consistently across the entire garment surface. "
            "Show proper lighting, natural fabric folds, and realistic shading. "
            "Use a clean white studio background. The garment should be displayed on a mannequin or flat-lay style. "
            "Make the output a high-quality fashion product photo."
        )

        # Attach the texture as an input image
        payload = texture_base64_png.split(',')[-1]
        texture_bytes = base64.b64decode(payload)
        pil_img = Image.open(BytesIO(texture_bytes)).convert("RGB")

        # Generate content - image generation models return images directly
        resp = model.generate_content([prompt, pil_img])

        # Try to extract inline image data
        try:
            for cand in getattr(resp, "candidates", []) or []:
                content = getattr(cand, "content", None)
                parts = getattr(content, "parts", []) if content else []
                for part in parts:
                    inline = getattr(part, "inline_data", None)
                    if inline and getattr(inline, "data", None):
                        return base64.b64encode(inline.data).decode()
        except Exception:
            pass

        # If we only got text back, surface a helpful error
        raise RuntimeError(
            "Gemini did not return an image. "
            f"Current model: {self.model}. "
            "Set GEMINI_MODEL=models/gemini-2.0-flash-exp-image-generation in .env for image generation."
        )


# Texture pattern descriptions for prompt building
TEXTURE_DESCRIPTIONS = {
    'banded': 'horizontal bands and stripes with layered gradient patterns',
    'blotchy': 'organic irregular patches and mottled splotches',
    'bumpy': 'raised textured surface with dimensional bumps and relief',
    'checkered': 'geometric checkerboard grid pattern with alternating squares',
    'cracked': 'weathered cracked surface with fragmented lines like dried earth',
    'dotted': 'scattered polka dots and speckled spots pattern',
    'flaky': 'layered peeling scales like fish scales or bark',
    'flecked': 'spattered flecks and tiny sprinkled spots',
    'freckled': 'delicate natural freckle-like random dots',
    'frilly': 'ornate ruffled decorative frills and flowing edges',
    'grooved': 'carved grooves and etched channel lines',
    'lined': 'parallel ruled lines in structured direction',
    'marbled': 'elegant swirled veins like marble stone with flowing organic curves',
    'paisley': 'intricate curved paisley motifs with bohemian ornate details',
    'polka-dotted': 'regular evenly spaced polka dots in playful pattern',
    'potholed': 'irregular cratered pits and weathered surface holes',
    'ribbed': 'vertical ribbed ridges like corduroy fabric texture',
    'sprinkled': 'lightly dusted scattered sprinkles pattern',
    'stained': 'vintage aged stains with mottled patina effect',
    'striped': 'bold parallel stripes in linear pattern',
    'swirly': 'dynamic spiraling swirls and whirling curves',
    'wavy': 'flowing undulating waves like water ripples',
    'zigzagged': 'sharp angular chevron zigzag pattern'
}


class PollinationsRenderer:
    """Render garment images using Pollinations.ai - FREE, no API key needed.
    
    Uses high-quality Flux/SDXL models for image generation.
    """

    def __init__(self, model: str = "flux"):
        self.model = model
        self.base_url = "https://image.pollinations.ai/prompt"

    def render(
        self,
        texture_base64_png: str,
        garment: str,
        color_hex_or_name: str,
        texture_class: str = "",
        color_palette: List[str] = None,
        concept_words: List[str] = None,
        texture_description: str = None
    ) -> str:
        """Generate a garment image with the given texture and color.
        
        Args:
            texture_base64_png: Base64 encoded texture image (for reference)
            garment: Type of garment (e.g., 'saree', 't-shirt', 'pants')
            color_hex_or_name: Primary color (hex or name)
            texture_class: The texture class name (e.g., 'marbled', 'striped')
            color_palette: List of hex colors extracted from texture
            concept_words: Descriptive words for the texture
            texture_description: AI-analyzed description of the actual texture image
        
        Returns:
            Base64 PNG string (without data URL prefix).
        """
        import urllib.parse
        import urllib.request

        # User's selected color is CRITICAL - put it first and emphasize it
        user_color = color_hex_or_name.lower()
        
        # Get texture pattern description (without colors)
        if texture_description and len(texture_description.strip()) > 10:
            texture_desc = texture_description.strip()
        else:
            texture_desc = TEXTURE_DESCRIPTIONS.get(
                texture_class.lower().replace('-', '').replace('_', ''),
                texture_class or 'textured'
            )

        # Build a SIMPLE, COLOR-FIRST prompt
        # AI image generators respond better to: [color] [subject] with [details]
        prompt = (
            f"A {user_color} {garment}, "
            f"{user_color} colored fabric with {texture_desc} pattern, "
            f"solid {user_color} color, "
            f"fashion product photo, studio lighting, white background, "
            f"high quality, photorealistic"
        )

        # URL encode the prompt
        encoded_prompt = urllib.parse.quote(prompt)
        
        # Build the URL with parameters - use seed for consistency
        url = f"{self.base_url}/{encoded_prompt}?model={self.model}&width=768&height=768&nologo=true"

        try:
            # Make the request
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=90) as response:
                image_bytes = response.read()
            
            # Return as base64
            return base64.b64encode(image_bytes).decode()
        except Exception as e:
            raise RuntimeError(f"Pollinations image generation failed: {e}")
