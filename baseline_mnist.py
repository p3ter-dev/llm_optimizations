"""
Baseline MNIST Feed-Forward Network
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class BaselineFFN(nn.Module):
    """
    Simple Feed-Forward Network for MNIST
    Architecture: 784 -> 512 -> 256 -> 128 -> 10
    """
    def __init__(self):
        super(BaselineFFN, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Flatten the input
        x = x.view(-1, 784)
        
        # Forward pass through layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_data_loaders(batch_size=64):
    """
    Load MNIST dataset with standard preprocessing
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_epoch(model, device, train_loader, optimizer, criterion, epoch):
    """
    Train for one epoch
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] '
                  f'Loss: {loss.item():.4f} Acc: {100. * correct / total:.2f}%')
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def evaluate(model, device, test_loader, criterion):
    """
    Evaluate model on test set
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    # Measure inference time
    start_time = time.time()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    inference_time = time.time() - start_time
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    print(f'\nTest set: Average loss: {avg_loss:.4f}, '
          f'Accuracy: {correct}/{total} ({accuracy:.2f}%)')
    print(f'Inference time: {inference_time:.2f}s\n')
    
    return avg_loss, accuracy, inference_time


def get_model_size(model):
    """
    Calculate model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def train_baseline(epochs=5, batch_size=64, lr=0.001):
    """
    Main training function for baseline model
    """
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = get_data_loaders(batch_size)
    
    # Create model
    model = BaselineFFN().to(device)
    print(f"Model created with {model.count_parameters():,} parameters")
    print(f"Model size: {get_model_size(model):.2f} MB\n")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    print("Starting training...")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, device, train_loader, optimizer, criterion, epoch)
        test_loss, test_acc, inference_time = evaluate(model, device, test_loader, criterion)
        
        print(f"Epoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        print("-" * 60)
    
    # Save the trained model
    torch.save(model.state_dict(), 'baseline_model.pth')
    print("\nBaseline model saved as 'baseline_model.pth'")
    
    return model, test_loader


if __name__ == "__main__":
    print("=" * 60)
    print("BASELINE MNIST FEED-FORWARD NETWORK")
    print("=" * 60 + "\n")
    
    model, test_loader = train_baseline(epochs=5)
    
    print("\n" + "=" * 60)
    print("BASELINE TRAINING COMPLETE")
    print("=" * 60)
