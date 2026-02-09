

# Task 2 — The Prober

## Neuron Visualization via Activation Maximization

> *“All models are wrong, some are useful.” — George Box*

This task explores **what individual neurons and channels in a trained CNN respond to**, using **gradient-based activation maximization**. The goal is **qualitative understanding**, not metrics.

You will synthesize images that maximally excite neurons and analyze whether they represent:

* Color features
* Stroke fragments / digit parts
* Entire digit shapes
* Multiple unrelated concepts (polysemanticity)

---

## 1. Preconditions

### Assumptions about existing code

* A CNN is already trained on **Biased Colored-MNIST**
* Images are **RGB (3-channel)**
* You have access to:

  * Model definition
  * Trained checkpoint
  * Forward pass returning logits
* Framework: **PyTorch**

Do **not** retrain the model in this task.

---

## 2. High-Level Strategy

We will:

1. Initialize an **optimizable image tensor**
2. Pass it through the frozen CNN
3. Define an **activation objective** for a neuron / channel
4. Perform **gradient ascent on the input image**
5. Visualize the resulting image
6. Repeat across layers and neurons
7. Analyze patterns and polysemanticity

This is a simplified version of OpenAI Microscope’s procedure.

---

## 3. Model Preparation

### Freeze the model

```python
model.eval()
for p in model.parameters():
    p.requires_grad = False
```

### Choose layers to probe

Start with:

* Early conv layer (e.g. `conv1`)
* Middle conv layer
* Final conv layer (before classifier)

Register **forward hooks** to capture activations.

```python
activations = {}

def hook_fn(name):
    def hook(module, input, output):
        activations[name] = output
    return hook

model.conv1.register_forward_hook(hook_fn("conv1"))
```

---

## 4. Input Image Initialization

### Base image tensor

```python
img = torch.randn(1, 3, 28, 28, requires_grad=True, device=device)
```

Options to try:

* Gaussian noise (default)
* Constant gray + noise
* Low-frequency noise (optional Gaussian blur)

Clamp image to valid range **after each step**:

```python
img.data.clamp_(0, 1)
```

---

## 5. Optimization Objective (Try Multiple)

You are encouraged to experiment. Below are **recommended objectives**.

---

### A. Channel-wise Mean Activation (Most Stable)

**What it shows:** what a channel as a whole responds to

```python
loss = -activations["conv1"][0, channel_idx].mean()
```

---

### B. Single Neuron (Spatial Position)

**What it shows:** localized receptive field behavior

```python
loss = -activations["conv1"][0, channel_idx, h, w]
```

---

### C. Top-k Spatial Mean

Reduces noise and reveals motifs.

```python
act = activations["conv1"][0, channel_idx]
loss = -act.view(-1).topk(k=20).values.mean()
```

---

### D. Class-Conditional Neuron Visualization (Optional)

Combine neuron activation + class logit:

```python
loss = -(neuron_term + 0.1 * logits[target_class])
```

Useful to see **digit vs color dominance**.

---

## 6. Optimization Loop

```python
optimizer = torch.optim.Adam([img], lr=0.05)

for step in range(500):
    optimizer.zero_grad()
    model(img)
    loss.backward()
    optimizer.step()
    img.data.clamp_(0, 1)
```

Track:

* Loss curve
* Intermediate images every ~50 steps

---

## 7. Regularization (Important)

Without regularization, images will be noisy and uninformative.

### Add one or more:

* **L2 penalty on image**

```python
loss += 1e-4 * img.pow(2).mean()
```

* **Total Variation (TV) Loss**
  Encourages smoothness:

```python
tv = torch.mean(torch.abs(img[:, :, :-1] - img[:, :, 1:])) + \
     torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
loss += 1e-3 * tv
```

---

## 8. Visualization & Logging

For each neuron/channel:

* Save final optimized image
* Save intermediate evolution (optional)
* Record:

  * Layer
  * Channel index
  * Objective used

Normalize images for display only.

---

## 9. Systematic Exploration Plan

### Minimum exploration checklist

* [ ] 5 channels from early conv layer
* [ ] 5 channels from middle layer
* [ ] 5 channels from last conv layer
* [ ] At least 3 neurons per channel (spatial)

---

## 10. What to Look For (Analysis Guide)

### Color sensitivity

* Entire image saturated with one color
* Weak or absent digit structure

### Shape sensitivity

* Clear strokes
* Digit-like curves
* Black background with colored foreground

### Mixed (Polysemantic) neurons

Same neuron produces **different patterns** when:

* Initialized with different noise
* Optimized at different spatial locations
* Combined with different weak regularizers

This is **polysemanticity**.

---

## 11. Polysemanticity Experiments (Important)

For a single neuron:

1. Run optimization **5 times with different seeds**
2. Compare resulting images
3. Ask:

   * Are they the same concept?
   * Or unrelated (color blob vs stroke)?

Optional:

* Interpolate between two optimized images and re-optimize.

---

## 12. Expected Outcomes (Qualitative)

You should be able to argue:

* Early layers → color blobs / edges
* Mid layers → stroke fragments
* Late layers → digit-like structures **or** pure color shortcuts
* Some neurons respond to **both color and shape**, depending on context

There are **no failure cases**, only observations.

---

## 13. Deliverables

Produce:

* A folder of neuron visualizations
* A short markdown summary answering:

  * Which layers focus on color?
  * Evidence of shortcut learning
  * Clear examples of polysemantic neurons

No quantitative metrics required.

---

## 14. Notes

* Do **not** use Grad-CAM here
* Do **not** backprop into weights
* This task is about **what neurons want to see**, not explanations yet

---


