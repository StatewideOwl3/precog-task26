
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from lucent.optvis import render, param, transform, objectives
from torchvision.utils import save_image
import numpy as np
import os

# --- 1. MODEL DEFINITION ---
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

# --- 2. SETUP & WEIGHT LOADING ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ThreeLayerCNN().to(device).eval()


# Load weights
weights_path = 'task1/saved_models/cnn_weights_feb1_GODLYPULL.pth'
if os.path.exists(weights_path):
    print(f"Loading weights from {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=device))
else:
    print(f"Warning: '{weights_path}' not found. Visualizing untrained weights.")

# Ensure experiments experiments_output exists
os.makedirs('experiments_output', exist_ok=True)


# --- 3. OPTIMIZATION ---
def get_vis(obj, label, transforms=None, iterations=128, img_size=28):
    print(f"Running optimization for {label}...")
    
    if transforms is None:
        transforms = [
            transform.pad(4, mode='constant', constant_value=0.5), # Pad to 36
            transform.jitter(2),
            transform.random_scale([0.9, 0.95, 1.0, 1.05, 1.1]),
            transform.random_rotate(list(range(-5, 6))),
            transform.jitter(2),
        ]
    

    
    # We pass fixed_image_size to ensure lucent doesn't auto-upsample to 224 (default behavior)
    # This is crucial for this small MNIST model.
    
    param_f = lambda: param.image(img_size)
    
    images = render.render_vis(
        model, 
        obj, 
        param_f, 
        transforms=transforms, 
        thresholds=(iterations,), 
        show_image=False, 
        progress=True,
        fixed_image_size=img_size
    )
    
    # Save image
    img = images[-1]
    # Normalize if needed or just clip
    # lucent returns numpy array (H, W, C) in [0, 1] usually
    
    # Convert to tensor for saving
    # img is a list of images usually if thresholds is tuple, wait render_vis returns list of images
    # Actually render_vis returns a list of images (one for each threshold)
    
    final_img = images[-1]
    
    # Save as PNG
    plt.imsave(f'experiments_output/{label.replace(" ", "_").replace(":", "-")}.png', final_img)
    return final_img

# --- 4. EXPERIMENTS ---

# Baseline: Conv1, Conv2, FC3
early_targets = [f"conv1:{i}" for i in range(4)]
late_targets = [f"conv2:{i}" for i in range(4)]
class_targets = [0, 3, 7, 9]

print("Starting Baseline Experiments...")
for t in early_targets:
    get_vis(objectives.channel("conv1", int(t.split(':')[-1])), f"Baseline_Early_{t}")

for t in late_targets:
    get_vis(objectives.channel("conv2", int(t.split(':')[-1])), f"Baseline_Late_{t}")

for c in class_targets:
    # 'fc3' is the final layer; we use channel() because neuron() expects spatial dimensions (4D tensor)
    # but fc3 output is (B, 10). channel() works for both.
    get_vis(objectives.channel("fc3", c), f"Baseline_Class_{c}")

print("Baseline Complete.")
