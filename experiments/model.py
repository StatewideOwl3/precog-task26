import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    A simple 2-layer CNN + 3 FC layers.
    Architecture designed to be simple enough to fall for shortcuts.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Conv block 1
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2) # 28 -> 14
        
        # Conv block 2
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2) # 14 -> 7
        
        # Fully connected
        flat_dim = 16 * 7 * 7
        self.fc1 = nn.Linear(flat_dim, 120)
        self.relu3 = nn.ReLU()
        
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x

class TinyCNN(nn.Module):
    """
    An extremely small CNN designed to force shortcut learning.
    It should have enough capacity to learn 'Color -> Class' (easy),
    but NOT enough to learn 'Shape -> Class' (hard).
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Drastically reduced channels and layers
        # Conv1: 3 -> 4 channels. 
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2) # 28 -> 14
        
        # Conv2: 4 -> 8 channels
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2) # 14 -> 7
        
        # Fully connected: Minimal
        flat_dim = 8 * 7 * 7 # 392
        self.fc1 = nn.Linear(flat_dim, 16) # Bottleneck
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x
