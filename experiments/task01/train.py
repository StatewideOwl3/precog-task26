import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
from PIL import Image
from torchvision import transforms
import numpy as np
import random

from model import SimpleCNN

# --- Dataset Definition ---

class ColoredMNISTDataset(Dataset):
    def __init__(self, split: str, data_root: Path, transform=None):
        self.split = split
        self.data_root = data_root
        self.images_dir = data_root / split / "images"
        self.transform = transform
        
        csv_path = data_root / split / "labels.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"{csv_path} not found. Run generate_dataset.py first.")
        
        self.meta = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        img_path = self.images_dir / row.filename
        
        # Load image (RGB)
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
            
        label = int(row.label)
        return img, label

# --- Training & Evaluation ---

def train(model, loader, criterion, optimizer, device, epoch, log_interval=100):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loader.dataset)} "
                  f"({100. * batch_idx / len(loader):.0f}%)]\tLoss: {loss.item():.6f}")

    avg_loss = running_loss / len(loader)
    acc = 100. * correct / total
    print(f"==> Train Epoch: {epoch} Average Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")
    return acc

def evaluate(model, loader, criterion, device, name="Test"):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    test_loss /= len(loader)
    acc = 100. * correct / total
    print(f"==> {name} set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({acc:.2f}%)")
    return acc

def main():
    parser = argparse.ArgumentParser(description="Train SimpleCNN on Color-Biased MNIST")
    parser.add_argument("--data_root", type=str, default="../data/colored_mnist", help="Root for input dataset")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model", type=str, default="simple", choices=["simple", "tiny"], help="Model architecture")
    parser.add_argument("--no_cuda", action="store_true", default=False, help="disables CUDA training")
    
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Seeding
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    data_root = Path(args.data_root)
    
    # Data Loaders
    transform = transforms.ToTensor()
    
    train_dataset = ColoredMNISTDataset("train", data_root, transform=transform)
    test_dataset = ColoredMNISTDataset("test", data_root, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2)
    
    # Model
    if args.model == "tiny":
        from model import TinyCNN
        print("Initializing TinyCNN...")
        model = TinyCNN().to(device)
    else:
        print("Initializing SimpleCNN...")
        model = SimpleCNN().to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # Training Loop
    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, criterion, optimizer, device, epoch)
        
        # Eval on Train (Biased)
        print("Evaluating on Biased Training Set:")
        evaluate(model, train_loader, criterion, device, name="Biased Train")
        
        # Eval on Test (Counterfactual)
        print("Evaluating on Counterfactual Test Set:")
        acc = evaluate(model, test_loader, criterion, device, name="Counterfactual Test")
    
    print("\nFinal Results:")
    print(f"Biased Train Accuracy: {evaluate(model, train_loader, criterion, device, name='Biased Train'):.2f}%")
    print(f"Counterfactual Test Accuracy: {evaluate(model, test_loader, criterion, device, name='Counterfactual Test'):.2f}%")

if __name__ == "__main__":
    main()
