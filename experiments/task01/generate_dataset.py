import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

# --- Configuration & Palette ---
# from one of the papers iirc
PALETTE: Dict[int, torch.Tensor] = {
    0: torch.tensor([128, 0, 128], dtype=torch.float32) / 255.0,    # distinct purple
    1: torch.tensor([0, 100, 0], dtype=torch.float32) / 255.0,      # dark green
    2: torch.tensor([188, 143, 143], dtype=torch.float32) / 255.0,  # rosy brown
    3: torch.tensor([255, 0, 0], dtype=torch.float32) / 255.0,      # red
    4: torch.tensor([255, 215, 0], dtype=torch.float32) / 255.0,    # gold
    5: torch.tensor([0, 255, 0], dtype=torch.float32) / 255.0,      # lime
    6: torch.tensor([65, 105, 225], dtype=torch.float32) / 255.0,   # royal blue
    7: torch.tensor([0, 225, 225], dtype=torch.float32) / 255.0,    # cyan-ish
    8: torch.tensor([0, 0, 255], dtype=torch.float32) / 255.0,      # blue
    9: torch.tensor([255, 20, 147], dtype=torch.float32) / 255.0,   # deep pink
}

BASE_TRANSFORM = transforms.ToTensor()

def sample_color(digit: int) -> torch.Tensor:
    """Return the base color for the digit without jitter."""
    return PALETTE[digit]

def colorize_digit(digit_tensor: torch.Tensor, color: torch.Tensor) -> torch.Tensor:
    """Color the digit strokes with the given color on black background.
    
    Args:
        digit_tensor: (1, H, W) tensor with values 0-1 (grayscale)
        color: (3,) tensor with values 0-1
    Returns:
        (3, H, W) tensor
    """
    # digit_tensor has shape (1, H, W) with values 0-1
    # Expand to 3 channels: color * digit
    colored = color.view(3, 1, 1) * digit_tensor.expand(3, -1, -1)
    return colored

def assign_color_digit(label: int, split: str, p_bias: float) -> int:
    """
    Train: 95% same digit color, 5% random OTHER color.
    Test: never same digit color (100% counterfactual).
    """
    others = [d for d in range(10) if d != label]
    
    if split == "train":
        # With prob p_bias, use the digit's canonical color.
        # Otherwise, pick a random OTHER color.
        if random.random() < p_bias:
            return label
        else:
            return random.choice(others)
    else:
        # Test: Always counterfactual
        # We enforce disjoint colors from training dominant colors.
        # Since dominant color for d is C_d, we pick any C_k where k != d
        return random.choice(others)

def prepare_split(
    split: str,
    data_root: Path,
    mnist_root: Path,
    p_bias: float,
    limit: Optional[int] = None,
    overwrite: bool = False
) -> pd.DataFrame:
    """Create biased train or debiased test split as images + labels.csv."""
    assert split in {"train", "test"}
    is_train = split == "train"
    
    # Download/Load MNIST
    mnist = datasets.MNIST(root=mnist_root, train=is_train, download=True)
    
    # Setup output directories
    out_split = data_root / split
    images_dir = out_split / "images"
    
    if overwrite and out_split.exists():
        shutil.rmtree(out_split)
    
    images_dir.mkdir(parents=True, exist_ok=True)
    
    records = []
    
    # Iterate and colorize
    print(f"Processing {split} split...")
    for idx, (img, label) in enumerate(mnist):
        if limit is not None and idx >= limit:
            break
            
        # Convert PIL image to tensor (1, 28, 28)
        digit_tensor = BASE_TRANSFORM(img)
        
        # Decide which color to apply
        assigned_color_idx = assign_color_digit(label, split, p_bias)
        color_tensor = sample_color(assigned_color_idx)
        
        # Create colored image (3, 28, 28)
        composite = colorize_digit(digit_tensor, color_tensor)
        
        filename = f"{idx:05d}_{label}.png"
        save_path = images_dir / filename
        save_image(composite, save_path)
        
        records.append({
            "filename": filename,
            "label": int(label),
            "color_digit": int(assigned_color_idx),
            "split": split
        })
        
        if (idx + 1) % 5000 == 0:
            print(f"  Processed {idx + 1} images...")

    # Save metadata
    meta = pd.DataFrame.from_records(records)
    meta.to_csv(out_split / "labels.csv", index=False)
    print(f"Saved {len(meta)} images to {images_dir}")
    return meta

def main():
    parser = argparse.ArgumentParser(description="Generate Color-Biased MNIST")
    parser.add_argument("--data_root", type=str, default="../data/colored_mnist", help="Root for output dataset")
    parser.add_argument("--mnist_root", type=str, default="../data", help="Root for storing/loading raw MNIST")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--bias", type=float, default=0.95, help="Bias strength for training set (0.0-1.0)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples per split (for debugging)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing dataset")
    
    args = parser.parse_args()
    
    # Seeding
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    data_path = Path(args.data_root)
    mnist_path = Path(args.mnist_root)
    
    print(f"Generating dataset with bias={args.bias}, seed={args.seed}")
    
    # Generate Train (Biased)
    prepare_split(
        split="train",
        data_root=data_path,
        mnist_root=mnist_path,
        p_bias=args.bias,
        limit=args.limit,
        overwrite=args.force
    )
    
    # Generate Test (Counterfactual)
    prepare_split(
        split="test",
        data_root=data_path,
        mnist_root=mnist_path,
        p_bias=args.bias, # Doesn't matter for test, logic handles it
        limit=args.limit,
        overwrite=args.force
    )
    
    print("Dataset generation complete.")

if __name__ == "__main__":
    main()
