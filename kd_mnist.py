"""
Knowledge Distillation (KD) Implementation

Key Concepts:
1. Teacher model: Large, accurate model
2. Student model: Smaller, faster model
3. Distillation: Student learns from teacher's soft predictions
4. Temperature scaling: Softens probability distributions

Mathematics:
- Soft predictions: p_i(T) = exp(z_i/T) / Σ_j exp(z_j/T)
- Distillation loss: L_KD = α·T²·KL(p_teacher || p_student) + (1-α)·CE(y_true, p_student)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import numpy as np

torch.manual_seed(42)
np.random.seed(42)


class StudentFFN(nn.Module):
    """
    Smaller Student Network for Knowledge Distillation
    
    Architecture: 784 -> 256 -> 128 -> 10
    (Smaller than baseline: 784 -> 512 -> 256 -> 128 -> 10)
    """
    def __init__(self):
        super(StudentFFN, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss
    
    Mathematics:
    L_KD = α · L_soft + (1-α) · L_hard
    
    Where:
    - L_soft = T² · KL_divergence(teacher_soft || student_soft)
    - L_hard = CrossEntropy(y_true, student_predictions)
    - T is temperature for softening distributions
    - α balances soft and hard losses
    """
    def __init__(self, temperature=3.0, alpha=0.7):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, student_logits, teacher_logits, true_labels):
        """
        Compute distillation loss
        
        Args:
            student_logits: Raw outputs from student model
            teacher_logits: Raw outputs from teacher model
            true_labels: Ground truth labels
        """
        # Soft targets (temperature-scaled softmax)
        # Higher temperature = softer distribution = more information
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # Soft loss (KL divergence between teacher and student)
        # We multiply by T² to compensate for the scaling of gradients
        soft_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Hard loss (standard cross-entropy with true labels)
        hard_loss = self.ce_loss(student_logits, true_labels)
        
        # Combined loss
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return total_loss, soft_loss, hard_loss


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


def train_epoch_kd(student, teacher, device, train_loader, optimizer, distillation_loss, epoch):
    """
    Train student model with knowledge distillation for one epoch
    """
    student.train()
    teacher.eval()  # Teacher is frozen
    
    total_loss = 0
    total_soft_loss = 0
    total_hard_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Get student predictions
        student_logits = student(data)
        
        # Get teacher predictions (no gradient needed)
        with torch.no_grad():
            # Handle both baseline and MoE teachers
            teacher_output = teacher(data)
            if isinstance(teacher_output, tuple):
                teacher_logits = teacher_output[0]  # MoE returns (output, loss)
            else:
                teacher_logits = teacher_output
        
        # Compute distillation loss
        loss, soft_loss, hard_loss = distillation_loss(student_logits, teacher_logits, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        total_soft_loss += soft_loss.item()
        total_hard_loss += hard_loss.item()
        
        pred = student_logits.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] '
                  f'Loss: {loss.item():.4f} (Soft: {soft_loss.item():.4f}, '
                  f'Hard: {hard_loss.item():.4f}) Acc: {100. * correct / total:.2f}%')
    
    avg_loss = total_loss / len(train_loader)
    avg_soft_loss = total_soft_loss / len(train_loader)
    avg_hard_loss = total_hard_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, avg_soft_loss, avg_hard_loss, accuracy


def evaluate(model, device, test_loader, is_moe=False):
    """Evaluate model on test set"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Handle MoE models
            if is_moe:
                output, _ = model(data)
            else:
                output = model(data)
                
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    inference_time = time.time() - start_time
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    print(f'Test set: Average loss: {avg_loss:.4f}, '
          f'Accuracy: {correct}/{total} ({accuracy:.2f}%)')
    print(f'Inference time: {inference_time:.2f}s')
    
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


def distill_knowledge(teacher, epochs=5, batch_size=64, lr=0.001, 
                     temperature=3.0, alpha=0.7, teacher_is_moe=False):
    """
    Main function for knowledge distillation
    
    Args:
        teacher: Pre-trained teacher model
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        temperature: Temperature for soft targets
        alpha: Balance between soft and hard losses
        teacher_is_moe: Whether teacher is an MoE model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = get_data_loaders(batch_size)
    
    # Create student model
    student = StudentFFN().to(device)
    teacher = teacher.to(device)
    
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"Teacher: {teacher.count_parameters():,} parameters, "
          f"{get_model_size(teacher):.2f} MB")
    print(f"Student: {student.count_parameters():,} parameters, "
          f"{get_model_size(student):.2f} MB")
    print(f"Size Reduction: {(1 - get_model_size(student)/get_model_size(teacher))*100:.1f}%")
    print("=" * 60 + "\n")
    
    # Evaluate teacher
    print("Teacher Performance:")
    teacher_loss, teacher_acc, teacher_time = evaluate(teacher, device, test_loader, teacher_is_moe)
    print()
    
    # Setup for distillation
    distillation_loss = DistillationLoss(temperature=temperature, alpha=alpha)
    optimizer = optim.Adam(student.parameters(), lr=lr)
    
    print("=" * 60)
    print(f"KNOWLEDGE DISTILLATION (T={temperature}, α={alpha})")
    print("=" * 60 + "\n")
    
    # Training loop
    for epoch in range(1, epochs + 1):
        train_loss, soft_loss, hard_loss, train_acc = train_epoch_kd(
            student, teacher, device, train_loader, optimizer, distillation_loss, epoch
        )
        test_loss, test_acc, test_time = evaluate(student, device, test_loader)
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} (Soft: {soft_loss:.4f}, Hard: {hard_loss:.4f})")
        print(f"  Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        print("-" * 60 + "\n")
    
    # Final comparison
    print("=" * 60)
    print("DISTILLATION RESULTS")
    print("=" * 60)
    print(f"Teacher Accuracy: {teacher_acc:.2f}%")
    print(f"Student Accuracy: {test_acc:.2f}%")
    print(f"Accuracy Gap: {teacher_acc - test_acc:.2f}%")
    print(f"Model Size: {get_model_size(teacher):.2f} MB → {get_model_size(student):.2f} MB")
    print(f"Inference Time: {teacher_time:.2f}s → {test_time:.2f}s")
    print(f"Speedup: {teacher_time/test_time:.2f}x")
    print("=" * 60 + "\n")
    
    # Save student model
    torch.save(student.state_dict(), 'student_distilled.pth')
    print("Distilled student model saved as 'student_distilled.pth'")
    
    return student


def train_student_baseline(epochs=5, batch_size=64, lr=0.001):
    """
    Train student model WITHOUT distillation (for comparison)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_data_loaders(batch_size)
    
    student = StudentFFN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=lr)
    
    print("=" * 60)
    print("TRAINING STUDENT WITHOUT DISTILLATION (Baseline)")
    print("=" * 60 + "\n")
    
    for epoch in range(1, epochs + 1):
        student.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = student(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        _, test_acc, _ = evaluate(student, device, test_loader)
        print(f"Epoch {epoch}: Test Accuracy = {test_acc:.2f}%\n")
    
    torch.save(student.state_dict(), 'student_baseline.pth')
    print("Baseline student model saved as 'student_baseline.pth'\n")
    
    return student


def demo_knowledge_distillation():
    """
    Demonstrate knowledge distillation using baseline model as teacher
    """
    from baseline_mnist import BaselineFFN
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load teacher model
    print("Loading teacher model...")
    teacher = BaselineFFN().to(device)
    
    try:
        teacher.load_state_dict(torch.load('baseline_model.pth'))
        print("Loaded pre-trained baseline model as teacher.\n")
    except:
        print("No pre-trained model found. Please train baseline first.")
        return
    
    # Train student with distillation
    student = distill_knowledge(
        teacher=teacher,
        epochs=5,
        temperature=3.0,
        alpha=0.7,
        teacher_is_moe=False
    )
    
    # Optional: Compare with student trained without distillation
    print("\n" + "=" * 60)
    print("COMPARISON: Student WITH vs WITHOUT Distillation")
    print("=" * 60 + "\n")
    print("For complete comparison, you can run:")
    print("  student_baseline = train_student_baseline(epochs=5)")
    
    return student


if __name__ == "__main__":
    print("=" * 60)
    print("KNOWLEDGE DISTILLATION (KD) FOR MNIST")
    print("=" * 60 + "\n")
    
    demo_knowledge_distillation()
    
    print("\n" + "=" * 60)
    print("KNOWLEDGE DISTILLATION COMPLETE")
    print("=" * 60)
