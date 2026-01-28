"""
Post-Training Quantization (PTQ) Implementation

Key Concepts:
1. Convert FP32 weights to INT8 to reduce model size by ~4x
2. Calibration: Determine scale and zero-point for quantization
3. Quantized inference: Run model with INT8 arithmetic
"""

import torch
import torch.nn as nn
import torch.quantization
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import time
import numpy as np
import copy

torch.manual_seed(42)
np.random.seed(42)


class QuantizationCalibrator:
    """
    Calibration for Post-Training Quantization
    
    Mathematics:
    For activation range [r_min, r_max] and quantized range [q_min, q_max]:
    
    Scale: S = (r_max - r_min) / (q_max - q_min)
    Zero-point: Z = q_min - round(r_min / S)
    
    Quantization: q = round(r / S) + Z
    Dequantization: r ≈ S * (q - Z)
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.activation_stats = {}
        
    def collect_stats(self, data_loader, num_batches=10):
        """
        Collect activation statistics for calibration
        
        We need to know the min/max values of activations
        to determine appropriate scale and zero-point
        """
        print("Collecting activation statistics for calibration...")
        self.model.eval()
        
        # Hooks to capture activations
        handles = []
        
        def get_hook(name):
            def hook(module, input, output):
                if name not in self.activation_stats:
                    self.activation_stats[name] = {
                        'min': float('inf'),
                        'max': float('-inf')
                    }
                
                # Update min/max
                if isinstance(output, tuple):
                    output = output[0]  # For MoE models
                    
                self.activation_stats[name]['min'] = min(
                    self.activation_stats[name]['min'],
                    output.min().item()
                )
                self.activation_stats[name]['max'] = max(
                    self.activation_stats[name]['max'],
                    output.max().item()
                )
            return hook
        
        # Register hooks for all Linear and ReLU layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.ReLU)):
                handles.append(module.register_forward_hook(get_hook(name)))
        
        # Run calibration data through model
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(data_loader):
                if batch_idx >= num_batches:
                    break
                data = data.to(self.device)
                _ = self.model(data)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        print(f"Calibration complete. Collected stats for {len(self.activation_stats)} layers.")
        return self.activation_stats


def quantize_tensor(tensor, num_bits=8):
    """
    Quantize a tensor to INT8
    
    Mathematics:
    1. Find range: [min_val, max_val]
    2. Calculate scale: S = (max_val - min_val) / (2^num_bits - 1)
    3. Calculate zero-point: Z = -round(min_val / S)
    4. Quantize: q = round(tensor / S) + Z
    5. Clip to valid range: q = clip(q, 0, 2^num_bits - 1)
    """
    qmin = 0
    qmax = 2**num_bits - 1
    
    min_val = tensor.min()
    max_val = tensor.max()
    
    # Calculate scale and zero-point
    scale = (max_val - min_val) / (qmax - qmin)
    
    # Avoid division by zero
    if scale == 0:
        scale = 1.0
    
    zero_point = qmin - torch.round(min_val / scale)
    
    # Quantize
    q_tensor = torch.round(tensor / scale + zero_point)
    q_tensor = torch.clamp(q_tensor, qmin, qmax)
    
    return q_tensor.to(torch.uint8), scale, zero_point


def dequantize_tensor(q_tensor, scale, zero_point):
    """
    Dequantize INT8 tensor back to FP32
    
    Mathematics:
    r = S * (q - Z)
    """
    return scale * (q_tensor.float() - zero_point)


class QuantizedLinear(nn.Module):
    """
    Quantized Linear Layer
    
    Stores weights in INT8 but computes in FP32 for simplicity
    In production, this would use INT8 arithmetic
    """
    def __init__(self, in_features, out_features):
        super(QuantizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Quantized weights (INT8)
        self.register_buffer('q_weight', torch.zeros(out_features, in_features, dtype=torch.uint8))
        self.register_buffer('weight_scale', torch.tensor(1.0))
        self.register_buffer('weight_zero_point', torch.tensor(0.0))
        
        # Bias stays in FP32
        self.register_buffer('bias', torch.zeros(out_features))
        
    def forward(self, x):
        # Dequantize weights for computation
        weight = dequantize_tensor(self.q_weight, self.weight_scale, self.weight_zero_point)
        return nn.functional.linear(x, weight, self.bias)
    
    @staticmethod
    def from_float(float_linear, num_bits=8):
        """Convert a regular Linear layer to QuantizedLinear"""
        q_linear = QuantizedLinear(float_linear.in_features, float_linear.out_features)
        
        # Quantize weights
        q_weight, scale, zero_point = quantize_tensor(float_linear.weight.data, num_bits)
        q_linear.q_weight = q_weight
        q_linear.weight_scale = scale
        q_linear.weight_zero_point = zero_point
        
        # Copy bias
        if float_linear.bias is not None:
            q_linear.bias = float_linear.bias.data.clone()
        
        return q_linear


def quantize_model(model, num_bits=8):
    """
    Apply Post-Training Quantization to the entire model
    
    This recursively replaces all Linear layers with QuantizedLinear layers
    """
    print(f"Quantizing model to {num_bits}-bit precision...")
    
    # Create a new model with quantized layers
    quantized_model = copy.deepcopy(model)
    
    def quantize_module(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                # Replace with quantized version
                setattr(module, name, QuantizedLinear.from_float(child, num_bits))
            else:
                # Recursively quantize child modules
                quantize_module(child)
    
    quantize_module(quantized_model)
    print("Model quantization complete.")
    
    return quantized_model


def get_data_loaders(batch_size=64):
    """Load MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Create calibration dataset (small subset of training data)
    calibration_dataset = Subset(train_dataset, range(500))  # 500 samples for calibration
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    calibration_loader = DataLoader(calibration_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, calibration_loader, test_loader


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
            
            # Handle MoE models (they return tuple)
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


def apply_ptq_to_model(model, calibration_loader, device, is_moe=False, num_bits=8):
    """
    Apply Post-Training Quantization to a trained model
    
    Steps:
    1. Calibrate (optional for simple PTQ)
    2. Quantize weights
    3. Evaluate quantized model
    """
    print("\n" + "="*60)
    print("APPLYING POST-TRAINING QUANTIZATION")
    print("="*60 + "\n")
    
    # Evaluate original model
    print("Original Model Performance:")
    _, orig_acc, orig_time = evaluate(model, device, calibration_loader, is_moe)
    orig_size = get_model_size(model)
    print(f"Model size: {orig_size:.2f} MB\n")
    
    # Quantize the model
    quantized_model = quantize_model(model, num_bits=num_bits)
    quantized_model = quantized_model.to(device)
    
    # Evaluate quantized model
    print(f"\nQuantized Model ({num_bits}-bit) Performance:")
    _, quant_acc, quant_time = evaluate(quantized_model, device, calibration_loader, is_moe)
    quant_size = get_model_size(quantized_model)
    print(f"Model size: {quant_size:.2f} MB")
    
    # Calculate improvements
    print("\n" + "="*60)
    print("QUANTIZATION RESULTS")
    print("="*60)
    print(f"Size Reduction: {orig_size:.2f} MB → {quant_size:.2f} MB "
          f"({(1 - quant_size/orig_size)*100:.1f}% reduction)")
    print(f"Accuracy: {orig_acc:.2f}% → {quant_acc:.2f}% "
          f"({quant_acc - orig_acc:+.2f}% change)")
    print(f"Inference Time: {orig_time:.2f}s → {quant_time:.2f}s "
          f"({(1 - quant_time/orig_time)*100:.1f}% speedup)")
    print("="*60 + "\n")
    
    return quantized_model


def demo_ptq_standalone():
    """
    Demonstrate PTQ on the baseline model
    """
    from baseline_mnist import BaselineFFN
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load data
    print("Loading MNIST dataset...")
    _, calibration_loader, test_loader = get_data_loaders(batch_size=64)
    
    # Load or train baseline model
    print("Loading baseline model...")
    model = BaselineFFN().to(device)
    
    try:
        model.load_state_dict(torch.load('baseline_model.pth'))
        print("Loaded pre-trained baseline model.")
    except:
        print("No pre-trained model found. Please train baseline first.")
        return
    
    # Apply PTQ
    quantized_model = apply_ptq_to_model(model, test_loader, device, is_moe=False)
    
    # Save quantized model
    torch.save(quantized_model.state_dict(), 'baseline_quantized.pth')
    print("Quantized model saved as 'baseline_quantized.pth'")
    
    return quantized_model


if __name__ == "__main__":
    print("="*60)
    print("POST-TRAINING QUANTIZATION (PTQ) FOR MNIST")
    print("="*60 + "\n")
    
    demo_ptq_standalone()
    
    print("\n" + "="*60)
    print("PTQ DEMONSTRATION COMPLETE")
    print("="*60)
