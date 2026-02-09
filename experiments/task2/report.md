# Image Tensor Optimization Report

## 1. Challenge & Solution
The goal was to improve feature visualization for a custom 3-layer CNN trained on MNIST.
**Initial State**: The provided notebook code failed with type errors, shape mismatches, and indexing errors.
**Root Causes**: 
1. `param.image()` returned a tuple, but `render_vis` expected a callable.
2. The CNN has fixed-size `Linear` layers expecting `28x28` input. `lucent` defaults to upsampling inputs to `224x224`, breaking the forward pass.
3. `objectives.neuron` expects 4D spatial tensors (N, C, H, W) and attempts to index spatial positions (H/2, W/2). The FC layer output is 2D (N, C), causing an `IndexError`.

**Solution**:
- **Bug Fix**: Passed `param_f` as `lambda: param.image(img_size)`.
- **Shape Fix**: Explicitly set `fixed_image_size=28` in `render_vis`.
- **Objective Fix**: Used `objectives.channel` instead of `objectives.neuron` for FC layers. `channel` correctly targets a specific unit in a flat vector (effectively treating it as a channel with no spatial dim), which is mathematically equivalent to maximizing the neuron's pre-activation/logit.

## 2. Methodology Update
We created two experiment scripts to systematically explore the latent space:

### A. Baseline Optimization (`run_optimization.py`)
- **Objective**: Visualize standard features for Early (Conv1), Late (Conv2), and Class (FC3) layers.
- **Configuration**:
    - `iterations=128`.
    - `fixed_image_size=28`.
    - Standard robust transforms (Jitter, Scale, Rotate).

### B. Advanced Optimization (`run_advanced_optimization.py`)
Three approaches to explore "better" optimization:

#### 1. Negative Activation (Inhibitory)
- **Concept**: Visualize what *suppresses* a neuron.
- **Objective**: `Maximize( -1 * activation )`.
- **Hypothesis**: Inhibitory images might resemble "anti-digits" or features of conflicting classes.

#### 2. Joint Optimization (Poly-semantic)
- **Concept**: Find inputs that activate two distinct neurons simultaneously.
- **Objective**: `Maximize( neuron_A + neuron_B )`.
- **Targets**: 
    - `Class 0 + Class 7`.
    - `Conv2:0 + Conv2:1`.

#### 3. Diversity Generation
- **Concept**: Generate multiple *distinct* images for the same target neuron.
- **Technique**: Use `param.image(batch=4)` combined with `objectives.diversity`.
- Note: `diversity` objective in `lucent` currently expects spatial dimensions, so we apply diversity pressure on `conv2` features while optimizing `fc3` class activations.

### C. Kernel Visualization (`explore.ipynb`)
- **Action**: Added a cell to `explore.ipynb` to directly visualize the 8 learned filters of the first convolutional layer (`Conv2d(3, 8, kernel_size=5)`).
- **Goal**: Inspect low-level features (colors, edges) directly from the weights.
- **Output**: `experiments_output/conv1_kernels.png` shows the 8 filters as 5x5 RGB grids.

## 3. Current Status
- **Baseline**: Running successfully. Generating visualizations for 12 targets (Conv1, Conv2, FC3).
- **Advanced**: Running successfully. 
    - Negative Class optimization.
    - Joint Class optimization.
    - Diversity generation.
- **Kernels**: Visualized.

## 4. Next Steps
- Analyze the generated images in `experiments_output` and `experiments_output/advanced`.
- If "Negative Optimization" yields noise, consider regularization (L1, TV) to suppress high-frequency artifacts.
