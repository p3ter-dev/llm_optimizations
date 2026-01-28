"""
Mixture of Experts (MoE) Implementation for MNIST

Key Concepts:
1. Multiple "expert" networks (specialized FFNs)
2. Gating network that routes inputs to top-k experts
3. Load balancing to ensure experts are used evenly
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import numpy as np

torch.manual_seed(42)
np.random.seed(42)


class Expert(nn.Module):
    """
    Single expert network - a small FFN
    Each expert specializes in different patterns in the data
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GatingNetwork(nn.Module):
    """
    Gating Network (Router)
    
    Mathematics:
    Given input x, compute gate values g = Softmax(W_g * x + b_g)
    The gate values determine which experts process the input
    """
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)
        
    def forward(self, x):
        # Returns probability distribution over experts
        return torch.softmax(self.fc(x), dim=-1)


class MoELayer(nn.Module):
    """
    Mixture of Experts Layer
    
    Key Components:
    1. Multiple expert networks
    2. Gating network for routing
    3. Top-k selection mechanism
    4. Load balancing loss
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts=4, top_k=2):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Create experts
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim) 
            for _ in range(num_experts)
        ])
        
        # Create gating network
        self.gate = GatingNetwork(input_dim, num_experts)
        
        # For tracking load balancing
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        
    def forward(self, x):
        """
        Forward pass with top-k routing
        
        Steps:
        1. Compute gate values for all experts
        2. Select top-k experts with highest gate values
        3. Route input through selected experts
        4. Combine expert outputs weighted by gate values
        """
        batch_size = x.size(0)
        
        # Step 1: Get gate values (routing probabilities)
        gate_values = self.gate(x)  # Shape: [batch_size, num_experts]
        
        # Step 2: Select top-k experts
        # topk returns (values, indices)
        top_k_gates, top_k_indices = torch.topk(gate_values, self.top_k, dim=-1)
        
        # Normalize top-k gate values to sum to 1
        top_k_gates = top_k_gates / top_k_gates.sum(dim=-1, keepdim=True)
        
        # Step 3 & 4: Route through experts and combine outputs
        output = torch.zeros(batch_size, self.experts[0].fc2.out_features).to(x.device)
        
        # Track which experts are being used (for load balancing)
        if self.training:
            for i in range(batch_size):
                for k in range(self.top_k):
                    expert_idx = top_k_indices[i, k].item()
                    self.expert_counts[expert_idx] += 1
        
        # Process through selected experts
        for i in range(self.top_k):
            # Get the expert indices for this position across the batch
            expert_indices = top_k_indices[:, i]
            
            # Get gate weights for this position
            gate_weights = top_k_gates[:, i].unsqueeze(1)
            
            # Process each unique expert that was selected
            for expert_idx in range(self.num_experts):
                # Find which batch items selected this expert
                mask = (expert_indices == expert_idx)
                if mask.any():
                    # Route those inputs through this expert
                    expert_input = x[mask]
                    expert_output = self.experts[expert_idx](expert_input)
                    # Add weighted output
                    output[mask] += gate_weights[mask] * expert_output
        
        # Calculate load balancing loss
        load_balance_loss = self.compute_load_balance_loss(gate_values)
        
        return output, load_balance_loss
    
    def compute_load_balance_loss(self, gate_values):
        """
        Load Balancing Loss
        
        Mathematics:
        L_balance = num_experts * Σᵢ (fᵢ * Pᵢ)
        
        Where:
        - fᵢ = fraction of inputs routed to expert i
        - Pᵢ = average gate probability for expert i
        
        This encourages balanced usage of experts
        """
        # Average gate probability for each expert (Pᵢ)
        P = gate_values.mean(dim=0)
        
        # Fraction of inputs with this expert in top-k (fᵢ)
        # We approximate this with the gate probabilities
        f = P  # Simplified version
        
        # Importance loss
        importance_loss = (f * P).sum() * self.num_experts
        
        return importance_loss


class MoEFFN(nn.Module):
    """
    Full MoE Feed-Forward Network for MNIST
    
    Architecture:
    Input (784) -> FC (512) -> MoE Layer -> FC (128) -> Output (10)
    """
    def __init__(self, num_experts=4, top_k=2):
        super(MoEFFN, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        # MoE layer in the middle
        self.moe_layer = MoELayer(
            input_dim=512,
            hidden_dim=256,
            output_dim=256,
            num_experts=num_experts,
            top_k=top_k
        )
        
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        
    def forward(self, x):
        # Flatten input
        x = x.view(-1, 784)
        
        # First layer
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        
        # MoE layer
        x, load_balance_loss = self.moe_layer(x)
        x = self.relu(x)
        
        # Final layers
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x, load_balance_loss
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_data_loaders(batch_size=64):
    """Load MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_epoch(model, device, train_loader, optimizer, criterion, epoch, alpha=0.01):
    """
    Train for one epoch with MoE
    
    The loss includes:
    1. Classification loss (cross-entropy)
    2. Load balancing loss (weighted by alpha)
    """
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_balance_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output, load_balance_loss = model(data)
        
        # Classification loss
        cls_loss = criterion(output, target)
        
        # Combined loss
        loss = cls_loss + alpha * load_balance_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_balance_loss += load_balance_loss.item()
        
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] '
                  f'Loss: {loss.item():.4f} (Cls: {cls_loss.item():.4f}, '
                  f'Balance: {load_balance_loss.item():.4f}) Acc: {100. * correct / total:.2f}%')
    
    avg_loss = total_loss / len(train_loader)
    avg_cls_loss = total_cls_loss / len(train_loader)
    avg_balance_loss = total_balance_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, avg_cls_loss, avg_balance_loss, accuracy


def evaluate(model, device, test_loader, criterion):
    """Evaluate MoE model on test set"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
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
    """Calculate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def analyze_expert_usage(model):
    """Analyze how balanced the expert usage is"""
    expert_counts = model.moe_layer.expert_counts.cpu().numpy()
    total = expert_counts.sum()
    
    print("\nExpert Usage Analysis:")
    print("-" * 40)
    for i, count in enumerate(expert_counts):
        percentage = (count / total * 100) if total > 0 else 0
        print(f"Expert {i}: {count:.0f} activations ({percentage:.2f}%)")
    print("-" * 40)
    
    # Calculate balance metric (coefficient of variation)
    if total > 0:
        mean_usage = expert_counts.mean()
        std_usage = expert_counts.std()
        cv = std_usage / mean_usage if mean_usage > 0 else 0
        print(f"Balance Metric (CV): {cv:.4f} (lower is better, 0 is perfect balance)")
    print()


def train_moe(num_experts=4, top_k=2, epochs=5, batch_size=64, lr=0.001, alpha=0.01):
    """
    Main training function for MoE model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = get_data_loaders(batch_size)
    
    # Create MoE model
    model = MoEFFN(num_experts=num_experts, top_k=top_k).to(device)
    print(f"MoE Model created with {model.count_parameters():,} parameters")
    print(f"  Number of experts: {num_experts}")
    print(f"  Top-k routing: {top_k}")
    print(f"Model size: {get_model_size(model):.2f} MB\n")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    print("Starting MoE training...")
    for epoch in range(1, epochs + 1):
        train_loss, cls_loss, balance_loss, train_acc = train_epoch(
            model, device, train_loader, optimizer, criterion, epoch, alpha
        )
        test_loss, test_acc, inference_time = evaluate(model, device, test_loader, criterion)
        
        print(f"Epoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} (Cls: {cls_loss:.4f}, Balance: {balance_loss:.4f})")
        print(f"  Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        print("-" * 60)
    
    # Analyze expert usage
    analyze_expert_usage(model)
    
    # Save the trained model
    torch.save(model.state_dict(), f'moe_model_e{num_experts}_k{top_k}.pth')
    print(f"\nMoE model saved as 'moe_model_e{num_experts}_k{top_k}.pth'")
    
    return model, test_loader


if __name__ == "__main__":
    print("=" * 60)
    print("MIXTURE OF EXPERTS (MoE) FOR MNIST")
    print("=" * 60 + "\n")
    
    # Train with 4 experts, top-2 routing
    model, test_loader = train_moe(num_experts=4, top_k=2, epochs=5)
    
    print("\n" + "=" * 60)
    print("MoE TRAINING COMPLETE")
    print("=" * 60)
