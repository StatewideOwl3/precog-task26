# Task: Demonstrate Shortcut Learning in Simple CNNs (MNIST, Color Bias)

## Objective
Design a **fully reproducible experiment** that **proves simple CNNs learn shortcut features** rather than true digit semantics when exposed to strong spurious correlations. The proof must be empirical, clear, and robust.

---

## Core Hypothesis to Prove
A simple CNN trained on MNIST with **class-specific color bias** will:
- Achieve **>95% accuracy on the biased training set**
- Fail catastrophically on a **counterfactual test set** (accuracy **<20%**)
- Demonstrate reliance on **color shortcuts** instead of digit shape

---

## Dataset Construction (Critical)

### Base Dataset
- Use **MNIST** only.
- Preserve **existing dataset paths and structure**.
- You must **programmatically modify digit coloring**.

### Training Set (Biased)
For each digit class `d ∈ {0,…,9}`:
- Assign **one unique dominant color** `C_d`
- **95%** of samples of class `d` must use color `C_d`
- Remaining **5%** must use a **different color** (random or systematic, but consistent)

### Test Set (Counterfactual / Debiased)
- **No test image may use the dominant training color for its class**
- All digit–color correlations seen in training must be **broken**
- Colors used in test **must be disjoint** from the dominant training colors per class

### Coloring Constraints
- You may color:
  - The **digit foreground (stroke)**, OR
  - The **background texture**
- ❌ Background **must not** be a single solid color  
  (use noise, texture, gradients, patterns, etc.)
- Coloring must preserve digit legibility

---

## Model Requirements

### CNN Architecture
- Use a **simple CNN** (no ResNet, no attention, no pretrained models)
- Example scale:
  - 2–4 convolution layers
  - ReLU activations
  - Optional pooling
  - Small fully connected head

### Training
- Train **only on the biased training set**
- You may tune:
  - Optimizer
  - Learning rate
  - Batch size
  - Epochs
  - Weight decay / regularization
- Training accuracy must exceed **95%**

### Evaluation
- Evaluate on:
  1. Biased training set
  2. Counterfactual test set
- Test accuracy **must drop below 20%**

---

## Experimental Constraints (Strict)
- ✅ You may **only edit files inside the `experiments/` folder**
- ❌ Do not modify any other folders or shared code
- ✅ All randomness must be **seeded**
- ✅ Results must be **reproducible across runs**

---

## Exploration & Rigor (Encouraged)
You are encouraged to:
- Try **multiple coloring strategies** (foreground vs textured background)
- Compare **multiple simple CNN variants**
- Adjust bias strength (e.g., 90/10 vs 95/5) if needed
- Log:
  - Train vs test accuracy
  - Failure modes
- Draw **explicit connections to prior work** on:
  - Shortcut learning
  - Spurious correlations
  - Inductive bias in CNNs

---

## Expected Outcome
A clean, minimal experiment that **unambiguously demonstrates**:
> The CNN learns the spurious color correlation as a shortcut and fails when that shortcut is removed.

This result should be strong enough to stand as a **didactic example** of shortcut learning in vision models.

---

## Deliverables
- Modified dataset generation code (within constraints)
- Training + evaluation code
- Clear metrics showing:
  - High training accuracy
  - Severe test-set collapse
- Reproducible execution instructions

Focus on **clarity, correctness, and empirical strength**.
