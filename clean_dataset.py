#!/usr/bin/env python3
"""
Dataset Cleaning Script
Removes unwanted images from texture dataset:
- Images containing objects, humans, animals
- Plain color blocks (low variance)
- Non-texture patterns

Uses CLIP model to detect semantic content and variance analysis for plain colors.
"""

import argparse
from pathlib import Path
import shutil
from typing import List, Tuple
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

try:
    import clip
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call(["pip", "install", "ftfy", "regex", "tqdm"])
    subprocess.check_call(["pip", "install", "git+https://github.com/openai/CLIP.git"])
    import clip


class DatasetCleaner:
    """Clean texture dataset by removing non-texture images."""
    
    # Categories to remove (objects, humans, animals, etc.)
    UNWANTED_CATEGORIES = [
        "a photo of a person",
        "a photo of a human",
        "a photo of people",
        "a photo of a face",
        "a photo of an object",
        "a photo of a car",
        "a photo of a building",
        "a photo of an animal",
        "a photo of a dog",
        "a photo of a cat",
        "a photo of furniture",
        "a photo of food",
        "a photo of a product",
        "a photo of text",
        "a photo of letters",
        "a screenshot",
        "a logo",
        "a graph",
        "a chart",
    ]
    
    # Texture descriptions (what we want to keep)
    TEXTURE_CATEGORIES = [
        "a close-up photo of a texture",
        "a photo of a surface pattern",
        "a photo of a material texture",
        "an abstract texture pattern",
        "a repeating pattern",
        "a natural texture",
    ]
    
    def __init__(self, device: str = None):
        """Initialize the cleaner with CLIP model."""
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        print(f"Loading CLIP model on {device}...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        
        # Encode text prompts
        with torch.no_grad():
            unwanted_text = clip.tokenize(self.UNWANTED_CATEGORIES).to(device)
            texture_text = clip.tokenize(self.TEXTURE_CATEGORIES).to(device)
            
            self.unwanted_features = self.model.encode_text(unwanted_text)
            self.unwanted_features /= self.unwanted_features.norm(dim=-1, keepdim=True)
            
            self.texture_features = self.model.encode_text(texture_text)
            self.texture_features /= self.texture_features.norm(dim=-1, keepdim=True)
    
    def is_plain_color_block(self, image: Image.Image, threshold: float = 10.0) -> bool:
        """
        Check if image is a plain color block (low variance).
        
        Args:
            image: PIL Image
            threshold: Variance threshold (lower = more uniform)
        
        Returns:
            True if image is plain/uniform color
        """
        img_array = np.array(image.convert('RGB'))
        
        # Calculate variance across all channels
        variance = np.var(img_array)
        
        # Also check per-channel variance
        r_var = np.var(img_array[:, :, 0])
        g_var = np.var(img_array[:, :, 1])
        b_var = np.var(img_array[:, :, 2])
        
        avg_channel_var = (r_var + g_var + b_var) / 3
        
        return variance < threshold or avg_channel_var < threshold
    
    def is_texture(
        self, 
        image_path: Path,
        texture_threshold: float = 0.25,
        unwanted_threshold: float = 0.28,
        variance_threshold: float = 10.0
    ) -> Tuple[bool, str]:
        """
        Determine if an image is a texture.
        
        Args:
            image_path: Path to image file
            texture_threshold: Minimum similarity to texture categories
            unwanted_threshold: Maximum similarity to unwanted categories
            variance_threshold: Minimum variance for non-plain images
        
        Returns:
            (is_texture, reason) tuple
        """
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            return False, f"Failed to load: {e}"
        
        # Check for plain color blocks
        if self.is_plain_color_block(image, variance_threshold):
            return False, "Plain color block (low variance)"
        
        # Preprocess and encode image
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity to unwanted categories
            unwanted_similarity = (image_features @ self.unwanted_features.T).max().item()
            
            # Calculate similarity to texture categories
            texture_similarity = (image_features @ self.texture_features.T).max().item()
        
        # Decision logic
        if unwanted_similarity > unwanted_threshold:
            return False, f"Unwanted content detected (score: {unwanted_similarity:.3f})"
        
        if texture_similarity < texture_threshold:
            return False, f"Not texture-like enough (score: {texture_similarity:.3f})"
        
        return True, f"Valid texture (texture: {texture_similarity:.3f}, unwanted: {unwanted_similarity:.3f})"
    
    def clean_dataset(
        self,
        dataset_root: Path,
        output_root: Path = None,
        remove_mode: bool = False,
        dry_run: bool = False,
        texture_threshold: float = 0.25,
        unwanted_threshold: float = 0.28,
        variance_threshold: float = 10.0,
        extensions: List[str] = None
    ):
        """
        Clean the dataset by filtering out non-texture images.
        
        Args:
            dataset_root: Root directory containing texture category folders
            output_root: Output directory (if None, modifies in place)
            remove_mode: If True, delete unwanted files; if False, copy good files
            dry_run: If True, only report what would be done
            texture_threshold: Minimum texture similarity score
            unwanted_threshold: Maximum unwanted content score
            variance_threshold: Minimum variance for non-plain images
            extensions: List of image extensions to process
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        dataset_root = Path(dataset_root)
        
        if output_root:
            output_root = Path(output_root)
            output_root.mkdir(parents=True, exist_ok=True)
        
        # Find all texture category folders
        category_dirs = [d for d in dataset_root.iterdir() if d.is_dir()]
        
        stats = {
            'total': 0,
            'kept': 0,
            'removed_plain': 0,
            'removed_unwanted': 0,
            'removed_not_texture': 0,
            'errors': 0
        }
        
        print(f"\nProcessing {len(category_dirs)} categories...")
        print(f"Mode: {'DRY RUN - ' if dry_run else ''}{'REMOVE unwanted' if remove_mode else 'COPY wanted to output'}")
        print(f"Thresholds: texture>={texture_threshold}, unwanted<{unwanted_threshold}, variance>={variance_threshold}\n")
        
        for category_dir in category_dirs:
            category_name = category_dir.name
            print(f"\n📁 Processing: {category_name}")
            
            # Get all image files
            image_files = []
            for ext in extensions:
                image_files.extend(category_dir.glob(f'*{ext}'))
                image_files.extend(category_dir.glob(f'*{ext.upper()}'))
            
            if not image_files:
                print(f"  ⚠️  No images found")
                continue
            
            # Create output directory if needed
            if output_root and not dry_run:
                output_category_dir = output_root / category_name
                output_category_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each image
            for img_path in tqdm(image_files, desc=f"  {category_name}"):
                stats['total'] += 1
                
                is_valid, reason = self.is_texture(
                    img_path,
                    texture_threshold=texture_threshold,
                    unwanted_threshold=unwanted_threshold,
                    variance_threshold=variance_threshold
                )
                
                if is_valid:
                    stats['kept'] += 1
                    if output_root and not remove_mode and not dry_run:
                        # Copy to output
                        shutil.copy2(img_path, output_root / category_name / img_path.name)
                else:
                    # Categorize removal reason
                    if "plain color" in reason.lower():
                        stats['removed_plain'] += 1
                    elif "unwanted" in reason.lower():
                        stats['removed_unwanted'] += 1
                    else:
                        stats['removed_not_texture'] += 1
                    
                    if remove_mode and not dry_run:
                        # Delete file
                        img_path.unlink()
                    
                    # Optionally log removed files
                    # print(f"    ❌ {img_path.name}: {reason}")
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Total images processed:     {stats['total']}")
        print(f"✅ Kept (valid textures):   {stats['kept']} ({100*stats['kept']/max(stats['total'],1):.1f}%)")
        print(f"❌ Removed (plain color):   {stats['removed_plain']}")
        print(f"❌ Removed (unwanted):      {stats['removed_unwanted']}")
        print(f"❌ Removed (not texture):   {stats['removed_not_texture']}")
        print(f"⚠️  Errors:                  {stats['errors']}")
        print(f"\nTotal removed:              {stats['total'] - stats['kept']}")
        
        if dry_run:
            print("\n⚠️  This was a DRY RUN - no files were actually modified")


def main():
    parser = argparse.ArgumentParser(
        description="Clean texture dataset by removing unwanted images"
    )
    parser.add_argument(
        "dataset_root",
        type=str,
        help="Path to dataset root directory containing category folders"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for cleaned dataset (default: modify in place)"
    )
    parser.add_argument(
        "--remove",
        action="store_true",
        help="Remove unwanted files instead of copying good ones"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually modifying files"
    )
    parser.add_argument(
        "--texture-threshold",
        type=float,
        default=0.25,
        help="Minimum similarity to texture categories (0-1, default: 0.25)"
    )
    parser.add_argument(
        "--unwanted-threshold",
        type=float,
        default=0.28,
        help="Maximum similarity to unwanted categories (0-1, default: 0.28)"
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=10.0,
        help="Minimum color variance to avoid plain blocks (default: 10.0)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Device to run model on (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    cleaner = DatasetCleaner(device=args.device)
    cleaner.clean_dataset(
        dataset_root=Path(args.dataset_root),
        output_root=Path(args.output) if args.output else None,
        remove_mode=args.remove,
        dry_run=args.dry_run,
        texture_threshold=args.texture_threshold,
        unwanted_threshold=args.unwanted_threshold,
        variance_threshold=args.variance_threshold
    )


if __name__ == "__main__":
    main()
