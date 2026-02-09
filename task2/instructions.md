# Task 2 — The Prober (Implementation Instructions)

This document contains **explicit, step-by-step implementation instructions** for Task 2 (The Prober). It is written to be handed directly to a coding agent. Follow the order strictly.

---

## 1. Context and Goal

You are working with a **CNN trained on Colored-MNIST with strong spurious color correlations**:

* Training set: ~95% correlation between digit class and color (in ../task0/outputs/colored-mnist/train/)
* Test set: correlation inverted / broken
* Model achieves high train accuracy but fails on unbiased test data

**Goal of Task 2:**
Expose *what internal neurons and channels have learned* by directly visualizing the features they respond to, using **activation maximization**.

This task is **exploratory and qualitative**. There are no numerical metrics. The output is insight.

---

## 2. High-level Strategy

We will:

1. Load the trained, frozen model weights from ../task1/saved_models/cnn_weights_feb1_GODLYPULL.pth
2. Evaluate it on the train and test sets first to show its accuracies
2. Select a small number of convolutional layers to probe (one by one, write a general function and pass probing layer as an input to the function)
3. Optimize synthetic input images (initially random gaussian noise) to maximize internal activations
4. Visualize the resulting images
5. Interpret whether neurons encode:

   * Color
   * Digit sub-parts (strokes, curves)
   * Mixed features (polysemanticity)

This is inspired by Lucid / OpenAI Microscope–style feature visualization, but implemented **directly in PyTorch**.

---

## 3. Assumptions About the Model (IMPORTANT)
You must use the same model architecture exactly as impemented in ../task1/cnn.ipynb
The instructions assume:

* PyTorch model
* Input shape: `(1, 3, 28, 28)` (Colored MNIST)
* Model contains multiple convolutional layers, e.g.:

  * conv1 (early)
  * conv2 (mid)
  * conv3 or final conv block (late)

The model must already be trained. **Do NOT retrain.**

---

## 4. Phase 0 — Setup (DO FIRST)

* Forward pass works
* No gradients accumulate on weights

---

### 4.2 Choose probe layers

Select **both the convolutional layers and some of the neurons in the following 3 Fully connected layers**:

* First layer → color blobs, edges
* Second layer → digit strokes or abstract features?

Do not probe every layer.

---

## 5. Phase 1 — Core Primitive: Channel-wise Activation Maximization

This is the **foundation of the entire task**. Everything else builds on this.

### 5.1 What we optimize

Given:

* A specific convolutional layer `L`
* A specific channel index `c`

We optimize an input image `x` to maximize:

```
mean(L(x)[c])
```

That is: **mean activation over spatial locations of one channel**.

This objective is:

* Stable
* Interpretable
* Standard in the literature

---

### 5.2 Generic probing function (REQUIRED)

Implement a reusable function similar to:

```python
def maximize_channel(model, layer, channel, steps=500, lr=0.05):
    img = torch.randn(1, 3, 28, 28, device=device, requires_grad=True)
    activations = {}

    def hook(_, __, output):
        activations['feat'] = output

    handle = layer.register_forward_hook(hook)
    optimizer = torch.optim.Adam([img], lr=lr)

    for step in range(steps):
        optimizer.zero_grad()
        model(img)
        loss = -activations['feat'][:, channel].mean()
        loss.backward()
        optimizer.step()
        img.data.clamp_(0, 1)

    handle.remove()
    return img.detach()
```

This function must:

* Not modify model weights
* Only update the image

---

## 6. Phase 2 — Early Layer Exploration (START HERE)

### 6.1 What to do

For the **first convolutional layer**:

* Pick 5–10 random channels
* Run `maximize_channel`
* Visualize results in a grid

### 6.2 What to expect

You should see:

* Strong color blobs (red, green, etc.)
* Very little digit structure

If color does *not* dominate here, something is wrong.

---

## 7. Phase 3 — Channel Survey (Shortcut Diagnosis)

### 7.1 Systematic exploration

For each probe layer:

* Select ~16 channels
* Visualize each channel’s optimized image

Do NOT cherry-pick yet.

---

### 7.2 Questions to answer

For each layer:

* How many channels are mostly color?
* How many show digit strokes?
* How many mix both?

This gives **direct internal evidence of shortcut learning**.

---

## 8. Phase 4 — Single-Neuron Probing (OPTIONAL, LIMITED)

### 8.1 When to use

Only use this **after** you identify an interesting channel.

### 8.2 Objective

Maximize a single spatial neuron:

```
L(x)[channel, y, x]
```

This reveals spatial specificity, not global concepts.

Limit to **1–2 examples total**.

---

## 9. Phase 5 — Polysemanticity Experiments (CRITICAL)

This is the most important conceptual part.

### 9.1 Same channel, different objectives

For a single channel, compare:

1. Plain mean activation
2. Mean activation + color penalty
3. Mean activation + jitter / blur

If optimized images differ drastically → **polysemantic neuron**.

---

### 9.2 Color suppression test

Add a color variance penalty:

```python
color_penalty = img.std(dim=(2,3)).mean()
loss = -activation + lambda_color * color_penalty
```

Interpretation:

* Feature disappears → color-dependent
* Shape remains → shape-selective

---

## 10. Phase 6 — Cross-layer Comparison

Repeat **the same experiment** across:

* Early layer
* Mid layer
* Late layer

Track:

* Where color dominates
* Where shape emerges
* Where bias persists

This directly motivates Task 4 (Intervention).

---

## 11. Logging and Documentation (MANDATORY)

For every saved visualization, log:

```
Layer:
Channel:
Objective:
Regularization:
Observation:
Interpretation:
```

You are graded on **interpretation quality**, not image aesthetics.

---

## 12. What NOT to do

* Do NOT retrain the model
* Do NOT start with class-logit maximization
* Do NOT probe every neuron
* Do NOT rely on external interpretability libraries

---

## 13. Minimal Completion Criteria

Task 2 is considered complete if you have:

* Channel visualizations from ≥ 3 layers
* Clear color-selective features
* At least one polysemantic neuron example
* Written interpretations connecting bias → representation

---

## 14. One-sentence Summary

This task demonstrates that **shortcut learning is encoded directly in the internal representations of CNNs**, not just in their outputs.
