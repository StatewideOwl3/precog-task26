
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# --- MODEL DEFINITION ---
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

# --- VISUALIZE ---
kernels = model.conv1.weight.detach().cpu()
print("Conv1 kernels shape:", kernels.shape)

fig, axes = plt.subplots(1, 8, figsize=(12, 3))
for i, ax in enumerate(axes):
    k = kernels[i] # (3, 5, 5)
    # Normalize to [0, 1]
    k = k - k.min()
    k = k / k.max()
    k = k.permute(1, 2, 0) # (5, 5, 3)
    ax.imshow(k)
    ax.axis("off")
    ax.set_title(f"Filter {i}")

os.makedirs('experiments_output', exist_ok=True)
plt.savefig('experiments_output/conv1_kernels.png')
print("Saved experiments_output/conv1_kernels.png")
