
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from lucent.optvis import render, param, transform, objectives
from lucent.misc.io import show
import numpy as np
import os

# --- MODEL (Same as before) ---
conv1_features = 8
conv2_features = 16

class ThreeLayerCNN(nn.Module):
    def __init__(self):
        super(ThreeLayerCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, conv1_features, kernel_size=5, padding="same")
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2) 
        self.conv2 = nn.Conv2d(conv1_features, conv2_features, kernel_size=5, padding="same")
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2) 

        self.fc1 = nn.Linear(conv2_features * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu_fc = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu_fc(self.fc1(x))
        x = self.relu_fc(self.fc2(x))
        x = self.fc3(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ThreeLayerCNN().to(device).eval()
weights_path = 'task1/saved_models/cnn_weights_feb1_GODLYPULL.pth'
if os.path.exists(weights_path):
    print(f"Loading weights from {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=device))

os.makedirs('experiments_output/advanced', exist_ok=True)

def save_vis(images, label, batch_idx=0):
    # images is list of steps. Get last step.
    img = images[-1]
    # If batch dimension exists and > 1
    if len(img.shape) == 4: # (B, H, W, C)
        img = img[batch_idx]
    
    plt.imsave(f'experiments_output/advanced/{label.replace(" ", "_").replace(":", "-")}.png', img)

def run_experiment(obj, label, batch=1, diversity=False):
    print(f"Running Experiment: {label}")
    
    transforms = [
        transform.pad(4, mode='constant', constant_value=0.5),
        transform.jitter(2),
        transform.random_scale([0.9, 0.95, 1.0, 1.05, 1.1]),
        transform.random_rotate(list(range(-5, 6))),
        transform.jitter(2),
    ]

    param_f = lambda: param.image(28, batch=batch)
    
    if diversity:
        # Diversity objective favors differences between batch items
        obj = obj + objectives.diversity("conv2") # Diversity in conv2 features
    
    try:

        images = render.render_vis(
            model, 
            obj, 
            param_f, 
            transforms=transforms, 
            thresholds=(128,), 
            show_image=False, 
            progress=True,
            fixed_image_size=28
        )
        
        # Save results
        if batch > 1:
            for i in range(batch):
                save_vis(images, f"{label}_sample{i}", batch_idx=i)
        else:
            save_vis(images, label)
            
    except Exception as e:
        print(f"Error in {label}: {e}")

# --- EXPERIMENTS ---

# 1. Negative Optimization (Inhibitory Stimuli)
# Find what suppresses the class logits
print("\n--- Negative Optimization ---")
for c in [0, 3, 7, 9]:
    # Use channel() for FC layers
    run_experiment(-1 * objectives.channel("fc3", c), f"Negative_Class_{c}")

# 2. Joint Activation
# Can we find an image that looks like '0' AND '7'? (Classes)
print("\n--- Joint Optimizations ---")
obj_07 = objectives.channel("fc3", 0) + objectives.channel("fc3", 7)
run_experiment(obj_07, "Joint_Class_0_and_7")

# Joint Conv2 channels (if we knew which ones correlate)
# Let's try arbitrary ones
obj_conv2_01 = objectives.channel("conv2", 0) + objectives.channel("conv2", 1)
run_experiment(obj_conv2_01, "Joint_Conv2_0_and_1")

# 3. Diversity
# Generate 4 different ways to activate Class 0
print("\n--- Diversity ---")
run_experiment(objectives.channel("fc3", 0), "Diversity_Class_0", batch=4, diversity=True)

