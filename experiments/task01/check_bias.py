import pandas as pd
from pathlib import Path
import argparse

def check_bias(split: str, data_root: Path):
    csv_path = data_root / split / "labels.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # Check simple accuracy (color == label)
    # Note: 'color_digit' index matches 'label' index in PALETTE 
    # if assign_color_digit returned the label as the color index.
    
    matches = (df["label"] == df["color_digit"]).sum()
    total = len(df)
    acc = matches / total if total > 0 else 0
    
    print(f"[{split.upper()}] Total: {total}, Matches: {matches}, Bias Score: {acc:.4f}")
    
    # Per digit breakdown
    print(f"[{split.upper()}] Per-digit match rates:")
    for d in range(10):
        subset = df[df["label"] == d]
        if len(subset) == 0:
            continue
        d_matches = (subset["label"] == subset["color_digit"]).sum()
        d_acc = d_matches / len(subset)
        print(f"  Digit {d}: {d_acc:.4f} ({d_matches}/{len(subset)})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="experiments/data/colored_mnist")
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    check_bias("train", data_root)
    check_bias("test", data_root)

if __name__ == "__main__":
    main()
