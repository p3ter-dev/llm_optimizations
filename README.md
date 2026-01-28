# Model Optimization Assignment: MoE, PTQ, and KD

Complete implementation of Mixture of Experts (MoE), Post-Training Quantization (PTQ), and Knowledge Distillation (KD) for MNIST classification.

## Project Overview

This project explores three key optimization techniques for neural networks:
1. **Mixture of Experts (MoE)**: Conditional computation with specialized expert networks
2. **Post-Training Quantization (PTQ)**: Reducing precision from FP32 to INT8
3. **Knowledge Distillation (KD)**: Training smaller models to mimic larger ones

## File Structure

```
├── baseline_mnist.py          # Baseline Feed-Forward Network
├── moe_mnist.py              # Mixture of Experts implementation
├── ptq_mnist.py              # Post-Training Quantization
├── kd_mnist.py               # Knowledge Distillation
├── run_all_experiments.py    # Comprehensive experiment runner
├── visualize_results.py      # Generate plots and analysis
└── README.md                 # This file
```

## Quick Start

### Installation

```bash
pip install torch torchvision numpy pandas matplotlib seaborn
```

### Running Experiments

**Option 1: Train everything from scratch (recommended for full understanding)**
```bash
# Train baseline model
python baseline_mnist.py

# Train MoE model
python moe_mnist.py

# Apply PTQ to baseline
python ptq_mnist.py

# Apply Knowledge Distillation
python kd_mnist.py

# Run all experiments and generate comparison
python run_all_experiments.py
```

**Option 2: Quick evaluation (if checkpoints exist)**
```bash
python run_all_experiments.py
# When prompted, enter 'load' to use existing checkpoints
```

### Generate Visualizations

```bash
python visualize_results.py
```

This will generate:
- `accuracy_vs_size.png` - Trade-off between model size and accuracy
- `accuracy_vs_time.png` - Trade-off between inference speed and accuracy
- `pareto_frontier.png` - Optimal models in accuracy-size-speed space
- `comparison_bars.png` - Bar charts comparing all metrics
- `normalized_comparison.png` - Radar chart showing normalized performance

## Understanding the Implementations

### 1. Baseline Model (`baseline_mnist.py`)

**Architecture**: 784 → 512 → 256 → 128 → 10

Simple feed-forward network that serves as our comparison point.

**Key Concepts**:
- Standard forward propagation
- Cross-entropy loss
- Adam optimizer

### 2. Mixture of Experts (`moe_mnist.py`)

**Key Mathematics**:
```
Gate values: g = Softmax(W_g · x)
Top-k selection: experts = TopK(g, k=2)
Output: y = Σ(gᵢ · Eᵢ(x)) for selected experts
Load balance loss: L_balance = Σ(fᵢ · Pᵢ)
```

**Implementation Details**:
- 4 expert networks (each 512 → 256 → 256)
- Top-2 routing (only 2 experts active per input)
- Gating network learns which experts to use
- Load balancing prevents expert collapse

**Why it works**:
- More parameters without proportional compute increase
- Experts specialize in different input patterns
- Conditional computation = efficiency

### 3. Post-Training Quantization (`ptq_mnist.py`)

**Key Mathematics**:
```
Quantization:
  Scale: S = (r_max - r_min) / (q_max - q_min)
  Zero-point: Z = q_min - round(r_min / S)
  Quantize: q = round(r / S) + Z
  
Dequantization:
  r ≈ S · (q - Z)
```

**Implementation Details**:
- Converts FP32 (32-bit) → INT8 (8-bit)
- ~4x model size reduction
- Minimal accuracy loss (<1% typically)
- Calibration on small data subset

**Why it works**:
- Neural networks are robust to reduced precision
- Most weight values don't need full FP32 range
- Hardware accelerators optimize INT8 operations

### 4. Knowledge Distillation (`kd_mnist.py`)

**Key Mathematics**:
```
Soft targets: pᵢ(T) = exp(zᵢ/T) / Σⱼ exp(zⱼ/T)

Distillation loss:
  L_KD = α · T² · KL(p_teacher || p_student) + (1-α) · CE(y_true, p_student)
  
Where:
  - T = temperature (softens distribution)
  - α = balance between soft and hard losses
  - KL = Kullback-Leibler divergence
```

**Implementation Details**:
- Teacher: Baseline (512→256→128→10) or MoE
- Student: Smaller network (256→128→10)
- Temperature T=3.0 for softer distributions
- α=0.7 (70% soft loss, 30% hard loss)

**Why it works**:
- Student learns from teacher's "dark knowledge"
- Soft probabilities contain relational information
- Example: If teacher outputs [0.7, 0.2, 0.1] for classes,
  student learns the relative similarities, not just the argmax

## Experimental Combinations

The assignment requires testing multiple combinations:

1. **Baseline only** - Standard model
2. **MoE only** - More parameters, conditional compute
3. **PTQ only** - Compressed baseline
4. **MoE + PTQ** - Compressed conditional compute
5. **KD (Baseline → Student)** - Smaller model via distillation
6. **KD (MoE → Student)** - Distill from expert ensemble
7. **KD + PTQ** - Compressed distilled model

## Expected Results

Based on typical MNIST performance:

| Model | Accuracy | Size (MB) | Inference Time | Parameters |
|-------|----------|-----------|----------------|------------|
| Baseline | ~98% | ~1.5 | 1.0x | ~500K |
| MoE | ~98.5% | ~2.5 | 1.2x | ~850K |
| Baseline + PTQ | ~97.5% | ~0.4 | 0.8x | ~500K |
| MoE + PTQ | ~98% | ~0.7 | 0.9x | ~850K |
| KD (Student) | ~97.5% | ~0.5 | 0.7x | ~220K |
| KD + PTQ | ~97% | ~0.13 | 0.6x | ~220K |

*Note: Actual results may vary*
