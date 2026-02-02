# Project: Colored MNIST with Background Texture Bias

## Objective
Create a synthetic MNIST-based dataset that introduces a **controlled spurious correlation**
between digit labels and **background color**, while keeping digit strokes unchanged.

The dataset must:
- Encourage CNNs to cheat by using background color
- Preserve digit shape as the true signal
- Allow clear diagnosis using Grad-CAM
- Support train/test bias inversion

This dataset will be used for interpretability and shortcut-learning experiments.

---

## Core Design Principles

1. **Digit strokes must remain untouched**
   - Original MNIST digits (grayscale)
   - No color applied to foreground pixels

2. **Color bias must exist only in the background**
   - Background must be non-uniform ("textured")
   - No flat or constant backgrounds allowed

3. **Background generation must be identical across all digits**
   - Same stochastic process
   - Only color distribution varies by digit

4. **Color–digit correlation must be strong but imperfect**
   - 95% biased samples
   - 5% counterexamples

---

## Base Dataset

- Source: MNIST (60,000 train / 10,000 test)
- Do NOT rebalance digits
- Preserve original train/test split

---

## Dataset Variants

### 1. Easy (Biased) Train Set
- For each digit `d`:
  - 95% of samples use digit-specific dominant color
  - 5% use a randomly chosen color from other digits

### 2. Hard (Debiased) Test Set

- Digit `d` is never assigned its dominant color

---

## Dominant Color Definition

A "dominant color" is defined statistically, NOT as a fixed RGB value.

For each digit:
- Assign a base hue (e.g., red, green, blue, etc.)
- For each image:
  - Sample color with:
    - Random hue jitter
    - Random saturation
    - Random brightness
- Constraint:
  - Expected value biased toward base hue
  - No two images should share identical background colors

Example:
- Digit 0 → reddish backgrounds
- Digit 1 → greenish backgrounds
- etc.

---

## Background Texture Generation (REQUIRED METHOD)

### Method: Tinted Gaussian Noise

For each image:
1. Generate Gaussian noise with fixed mean and variance
2. Normalize noise to valid pixel range
3. Apply sampled dominant color tint
4. Clamp values to image bounds

Properties:
- High-frequency texture
- No semantic structure
- Color bias distributed globally

This qualifies as "background texture" per task specification.

---

## Digit Overlay Procedure

1. Load original MNIST digit (grayscale)
2. Identify foreground pixels (digit mask)
3. For foreground pixels:
   - Replace background pixels with original digit intensity
4. For background pixels:
   - Keep tinted noise values

IMPORTANT:
- Digit pixels must overwrite background pixels
- No color bleeding into digit strokes

---

## Color Assignment Strategy

Define a fixed color palette mapping digits → base hues.

Example (modifiable but fixed once chosen):

| Digit | Base Hue |
|------|----------|
| 0 | Red |
| 1 | Green |
| 2 | Blue |
| 3 | Yellow |
| 4 | Magenta |
| 5 | Cyan |
| 6 | Orange |
| 7 | Purple |
| 8 | Teal |
| 9 | Brown |

Mapping must remain constant across all dataset splits.

---

## Output Format

Produce datasets in one of the following formats (choose one):

### Option A: PyTorch-style folders

dataset/
├── train/
│ ├── 0/
│ ├── 1/
│ └── ...
└── test/
├── 0/
├── 1/
└── ...

### Option B: Image folder + labels CSV
dataset/
├── train/
│ ├── images/
│ └── labels.csv
└── test/
├── images/
└── labels.csv


---

## Reproducibility Requirements

- Set and document random seed(s)
- Make background generation deterministic given seed
- Ensure identical results across runs

---

## Sanity Checks (MANDATORY)

Implement and save:
1. Random sample grid of generated images
2. Color histogram per digit (train set)
3. Confirmation that digit pixels are grayscale
4. Verification of 95% / 5% split

---

## Constraints (DO NOT VIOLATE)

- Do NOT color digit strokes
- Do NOT use flat backgrounds
- Do NOT change MNIST labels
- Do NOT introduce texture differences across digits
- Do NOT leak color into digit pixels

---

## Deliverables

- Dataset generation script or notebook
- Saved dataset on disk
- Visualization outputs for sanity checks
- Clear README explaining usage
