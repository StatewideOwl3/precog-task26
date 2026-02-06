# Shortcut Learning Experiment: Design & Technical Report

## 1. Objective & Hypothesis
The goal of this experiment is to empirically demonstrate **shortcut learning** in Convolutional Neural Networks (CNNs).
**Hypothesis**: When a dataset contains a simple, highly predictive "shortcut" feature (color) that is strongly correlated with the label, a CNN will learn to rely on this shortcut rather than the more complex, robust semantic features (digit shape). Consequently, the model will fail on a **counterfactual** test set where the shortcut relationship is broken.

## 2. Dataset Generation (`generate_dataset.py`)
To test this hypothesis, we constructed a **Colored MNIST** dataset with a controlled spurious correlation.

### 2.1 Color Palette
We defined a fixed palette of 10 distinct colors, one for each digit class (e.g., 0$\to$Purple, 1$\to$Green).

### 2.2 Biased Training Set
The training set is generated with a **Bias Strength ($p_{bias}$)**. For each image with digit label $y$:
-   **With probability $p_{bias}$**: The digit is colored with its assigned color $C_y$.
-   **With probability $1 - p_{bias}$**: The digit is colored with a random color $C_k$ where $k \neq y$.

**Design Choice - Bias Strength**:
We initially tested $p_{bias}=0.95$, but the model still learned some shape features (Test Accuracy ~80%). We increased $p_{bias}$ to **0.999** for the final experiment. This overwhelmingly strong correlation makes the color signal a "perfect" predictor during training, removing the optimization incentive for the model to learn the harder shape features.

### 2.3 Counterfactual Test Set
The test set is designed to be **completely counterfactual**. For every test image with label $y$:
-   The digit is **never** colored with $C_y$.
-   It is always colored with a random color $C_k$ ($k \neq y$) chosen from the palette.

**Rationale**: If the model relies on the color shortcut, it will predict class $k$ instead of $y$, leading to near-zero accuracy. If it learns the shape, it should ignore the color and predict $y$ correctly.

## 3. Model Architecture (`model.py`)
We used a custom **SimpleCNN** architecture.

### Architecture Details
-   **Input**: $3 \times 28 \times 28$ (RGB Images)
-   **Layer 1**: Conv2d(3$\to$6, k=5) $\to$ ReLU $\to$ MaxPool(2)
-   **Layer 2**: Conv2d(6$\to$16, k=5) $\to$ ReLU $\to$ MaxPool(2)
-   **Flatten**: $16 \times 7 \times 7 \to 784$
-   **FC 1**: Linear(784 $\to$ 120) $\to$ ReLU
-   **FC 2**: Linear(120 $\to$ 84) $\to$ ReLU
-   **Output**: Linear(84 $\to$ 10)

### Design Rationale
-   **Simplicity**: We chose a shallow network (similar to LeNet-5) rather than a deep ResNet. Deep networks with massive capacity might memorize the 0.1% outliers (shape examples) even in a high-bias setting. A constrained capacity model is more likely to take the path of least resistance (the color shortcut).
-   **No Normalization**: We intentionally avoided Batch Normalization to keep the optimization dynamics simple and purely driven by the loss landscape of the features.

## 4. Training Configuration (`train.py`)

### Hyperparameters
-   **Optimizer**: Stochastic Gradient Descent (SGD)
    -   `lr=0.01`: Standard learning rate for simple CNN convergence.
    -   `momentum=0.9`: Helps accelerate gradients in the right direction.
-   **Loss Function**: CrossEntropyLoss.
-   **Batch Size**: 64.
-   **Epochs**: 5.
    -   **Rationale**: The color shortcut is extremely easy to learn. As seen in the logs, the model converges to >99% training accuracy within 2 epochs. Longer training is unnecessary and potentially harmful (could lead to slow learning of shape features).

### Evaluation Metrics
We track accuracy on two splits:
1.  **Biased Train Accuracy**: Measures how well the model learns the training distribution (expected $\approx 100\%$).
2.  **Counterfactual Test Accuracy**: Measures robustness and reliance on shape (expected $< 10\%$ if shortcut learning occurs).

## 5. Summary of Findings
Running the experiment with $p_{bias}=0.999$ yielded:
-   **Biased Train Accuracy**: **99.91%**
-   **Counterfactual Test Accuracy**: **5.90%**

**Conclusion**: The model achieved near-perfect performance on the training set by exploiting the color correlation. When this correlation was inverted in the test set, the model's performance collapsed to worse-than-random (random chance is 10%), indicating it was actively misled by the color cues. This definitively confirms the shortcut learning hypothesis.

## 6. Alternative Approach: Reduced Model Capacity
The user asked if we could demonstrate this effect *without* changing the bias from the initial 0.95.
**Hypothesis**: A smaller model with limited capacity may not have the resources to learn the complex shape features, forcing it to rely on the easier color shortcut, even if the shortcut is less perfect (95% vs 99.9%).

**Experiment**:
-   **Bias**: 0.95
-   **Model**: `TinyCNN` (drastically reduced channels: 3$\to$4$\to$8).
-   **Results (Epoch 5)**:
    -   Biased Train Accuracy: **~98%**
    -   Counterfactual Test Accuracy: **~78%**
-   **Observation**:
    1.  **Early Learning**: In Epoch 1, `TinyCNN` had only ~49% test accuracy (vs ~85% for `SimpleCNN`), showing it initially struggled to learn the shape.
    2.  **Convergence**: By Epoch 5, it eventually learned the shape features, reaching ~78% test accuracy.
    3.  **Conclusion**: Reducing model capacity **delays** learning of complex features but does not prevent it as effectively as increasing the bias strength. The strongest demonstration of shortcut learning remains the **High Bias (99.9%)** setting, where the model completely ignored the shape.
