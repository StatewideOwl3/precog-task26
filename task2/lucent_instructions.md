Objective: Write a Python script using the lucent library to perform feature visualization on a pre-trained CNN. The model was trained on a biased MNIST dataset (Gaussian noise and color-tinted backgrounds). The goal is to investigate "shortcut learning" by visualizing what specific neurons have learned.
1. Model Loading & Setup

    Weights: Load the model weights from a local file (e.g., model_weights.pth).

    Architecture: Define the CNN architecture (ensure it matches the saved weights).

    Preprocessing: Explicitly state if the model expects normalized inputs (0-1 or ImageNet-style normalization) and ensure Lucent’s param.image matches this.

2. Optimization Targets (Objectives)

Please generate optimization loops for the following objectives:

    Convolutional Layers: Select 2-3 middle and late Conv2d layers. Optimize for specific Channels (e.g., layer:channel).

        Reasoning: Use channel objectives to see if the filters are picking up the Gaussian noise patterns or the color bias instead of the digit stroke.

    Fully Connected Layer: Select the final linear layer. Optimize for specific Neurons (classes 0-9).

        Reasoning: Use neuron objectives here to see the "ideal" input for a specific class. This will reveal if the model thinks "9" is just a "yellow-tinted grainy square."

3. Transformation & Regularization

    Use Lucent’s default transformations (jitter, rotate, scale) to ensure the features are robust and not just high-frequency noise.

    Set the optimization for at least 512 iterations to ensure convergence.

4. Visualization & Output

    Heatmaps/Feature Maps: Generate the optimized image tensors.

    Composite Grid: Use torchvision.utils.make_grid or matplotlib to create a single composite image.

        Row 1: Optimized images for 4 different channels in an early Conv layer.

        Row 2: Optimized images for 4 different channels in a late Conv layer.

        Row 3: Optimized images for 4 specific class neurons (e.g., digits 0, 3, 7, 9) from the FC layer.

    Labeling: Ensure each subplot in the grid is clearly labeled with the layer name and unit index.

5. Technical Constraints

    Use lucent.optvis.render and lucent.optvis.param.image.

    Handle the device (CUDA/CPU) automatically.

    Provide a clear "Model Definition" block at the top where I can paste my specific CNN class.