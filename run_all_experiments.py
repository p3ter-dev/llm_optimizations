"""
Comprehensive Experiment Runner
Combines MoE, PTQ, and KD in various configurations

This script runs all combinations:
1. Baseline only
2. MoE only
3. PTQ only
4. MoE + PTQ
5. KD (Baseline → Student)
6. KD (MoE → Student)
7. KD + PTQ

Collects metrics: accuracy, model size, inference time, parameters
"""

import json
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Import our implementations
from baseline_mnist import BaselineFFN, train_baseline
from kd_mnist import StudentFFN, distill_knowledge
from moe_mnist import MoEFFN, train_moe
from ptq_mnist import quantize_model

torch.manual_seed(42)


class ExperimentRunner:
    """
    Run and track all optimization experiments
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = []
        self.models = {}

    def get_data_loaders(self, batch_size=64):
        """Load MNIST dataset"""
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        train_dataset = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

    def evaluate_model(self, model, test_loader, model_name, is_moe=False):
        """
        Comprehensive evaluation of a model
        Returns: accuracy, inference_time, model_size, num_parameters
        """
        model.eval()
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()

        start_time = time.time()

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)

                # Handle MoE models
                if is_moe:
                    output, _ = model(data)
                else:
                    output = model(data)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        inference_time = time.time() - start_time
        accuracy = 100.0 * correct / total

        # Calculate model size
        param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
        model_size_mb = (param_size + buffer_size) / 1024**2

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        result = {
            "model_name": model_name,
            "accuracy": accuracy,
            "inference_time": inference_time,
            "model_size_mb": model_size_mb,
            "num_parameters": num_params,
        }

        self.results.append(result)
        print(f"\n{model_name}:")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Inference Time: {inference_time:.2f}s")
        print(f"  Model Size: {model_size_mb:.2f} MB")
        print(f"  Parameters: {num_params:,}")

        return result

    def run_experiment_1_baseline(self, train=True):
        """Experiment 1: Baseline Model"""
        print("\n" + "=" * 60)
        print("EXPERIMENT 1: Baseline FFN")
        print("=" * 60)

        _, test_loader = self.get_data_loaders()

        checkpoint_path = "baseline_model.pth"

        if train:
            print("Training baseline model from scratch...")
            model, _ = train_baseline(epochs=5)
        else:
            if not os.path.exists(checkpoint_path):
                print(f"⚠️  Checkpoint {checkpoint_path} not found!")
                print("Training baseline model instead...")
                model, _ = train_baseline(epochs=5)
            else:
                print(f"Loading baseline model from {checkpoint_path}...")
                model = BaselineFFN().to(self.device)
                model.load_state_dict(torch.load(checkpoint_path))
                print("Baseline model loaded successfully")

        self.models["baseline"] = model
        self.evaluate_model(model, test_loader, "Baseline", is_moe=False)

        return model

    def run_experiment_2_moe(self, train=True):
        """Experiment 2: MoE Model"""
        print("\n" + "=" * 60)
        print("EXPERIMENT 2: Mixture of Experts")
        print("=" * 60)

        _, test_loader = self.get_data_loaders()

        checkpoint_path = "moe_model_e4_k2.pth"

        if train:
            print("Training MoE model from scratch...")
            model, _ = train_moe(num_experts=4, top_k=2, epochs=5)
        else:
            if not os.path.exists(checkpoint_path):
                print(f"⚠️  Checkpoint {checkpoint_path} not found!")
                print("Training MoE model instead...")
                model, _ = train_moe(num_experts=4, top_k=2, epochs=5)
            else:
                print(f"Loading MoE model from {checkpoint_path}...")
                model = MoEFFN(num_experts=4, top_k=2).to(self.device)
                model.load_state_dict(torch.load(checkpoint_path))
                print("MoE model loaded successfully")

        self.models["moe"] = model
        self.evaluate_model(model, test_loader, "MoE (4 experts, top-2)", is_moe=True)

        return model

    def run_experiment_3_ptq_baseline(self):
        """Experiment 3: PTQ on Baseline"""
        print("\n" + "=" * 60)
        print("EXPERIMENT 3: PTQ on Baseline")
        print("=" * 60)

        if "baseline" not in self.models:
            self.run_experiment_1_baseline(train=False)

        _, test_loader = self.get_data_loaders()

        model = self.models["baseline"]
        quantized_model = quantize_model(model, num_bits=8)
        quantized_model = quantized_model.to(self.device)

        self.models["baseline_ptq"] = quantized_model
        self.evaluate_model(
            quantized_model, test_loader, "Baseline + PTQ", is_moe=False
        )

        torch.save(quantized_model.state_dict(), "baseline_ptq.pth")

        return quantized_model

    def run_experiment_4_moe_ptq(self):
        """Experiment 4: PTQ on MoE"""
        print("\n" + "=" * 60)
        print("EXPERIMENT 4: MoE + PTQ")
        print("=" * 60)

        if "moe" not in self.models:
            self.run_experiment_2_moe(train=False)

        _, test_loader = self.get_data_loaders()

        model = self.models["moe"]
        quantized_model = quantize_model(model, num_bits=8)
        quantized_model = quantized_model.to(self.device)

        self.models["moe_ptq"] = quantized_model
        self.evaluate_model(quantized_model, test_loader, "MoE + PTQ", is_moe=True)

        torch.save(quantized_model.state_dict(), "moe_ptq.pth")

        return quantized_model

    def run_experiment_5_kd_baseline(self, train=True):
        """Experiment 5: KD from Baseline"""
        print("\n" + "=" * 60)
        print("EXPERIMENT 5: Knowledge Distillation (Baseline → Student)")
        print("=" * 60)

        if "baseline" not in self.models:
            self.run_experiment_1_baseline(train=False)

        _, test_loader = self.get_data_loaders()

        checkpoint_path = "student_distilled.pth"

        # Check if checkpoint exists before trying to load
        if train or not os.path.exists(checkpoint_path):
            print(f"Training student model (checkpoint not found or train=True)...")
            teacher = self.models["baseline"]
            student = distill_knowledge(
                teacher=teacher,
                epochs=5,
                temperature=3.0,
                alpha=0.7,
                teacher_is_moe=False,
            )
            torch.save(student.state_dict(), checkpoint_path)
            print(f"Saved student model to {checkpoint_path}")
        else:
            print(f"Loading student model from {checkpoint_path}...")
            student = StudentFFN().to(self.device)
            student.load_state_dict(torch.load(checkpoint_path))
            print("Student model loaded successfully")

        self.models["kd_baseline"] = student
        self.evaluate_model(student, test_loader, "KD (Baseline Teacher)", is_moe=False)

        return student

    def run_experiment_6_kd_moe(self, train=True):
        """Experiment 6: KD from MoE"""
        print("\n" + "=" * 60)
        print("EXPERIMENT 6: Knowledge Distillation (MoE → Student)")
        print("=" * 60)

        if "moe" not in self.models:
            self.run_experiment_2_moe(train=False)

        _, test_loader = self.get_data_loaders()

        checkpoint_path = "student_distilled_moe.pth"

        # Check if checkpoint exists before trying to load
        if train or not os.path.exists(checkpoint_path):
            print(
                f"Training student model with MoE teacher (checkpoint not found or train=True)..."
            )
            teacher = self.models["moe"]
            student = distill_knowledge(
                teacher=teacher,
                epochs=5,
                temperature=3.0,
                alpha=0.7,
                teacher_is_moe=True,
            )
            torch.save(student.state_dict(), checkpoint_path)
            print(f"Saved student model to {checkpoint_path}")
        else:
            print(f"Loading student model from {checkpoint_path}...")
            student = StudentFFN().to(self.device)
            student.load_state_dict(torch.load(checkpoint_path))
            print("Student model loaded successfully")

        self.models["kd_moe"] = student
        self.evaluate_model(student, test_loader, "KD (MoE Teacher)", is_moe=False)

        return student

    def run_experiment_7_kd_ptq(self):
        """Experiment 7: KD + PTQ"""
        print("\n" + "=" * 60)
        print("EXPERIMENT 7: Knowledge Distillation + PTQ")
        print("=" * 60)

        # Make sure we have the distilled student
        if "kd_baseline" not in self.models:
            print("KD baseline student not found. Running experiment 5 first...")
            self.run_experiment_5_kd_baseline(train=False)

        _, test_loader = self.get_data_loaders()

        student = self.models["kd_baseline"]

        print("Quantizing distilled student model...")
        quantized_student = quantize_model(student, num_bits=8)
        quantized_student = quantized_student.to(self.device)

        self.models["kd_ptq"] = quantized_student
        self.evaluate_model(quantized_student, test_loader, "KD + PTQ", is_moe=False)

        torch.save(quantized_student.state_dict(), "student_ptq.pth")
        print("Saved quantized student model to student_ptq.pth")

        return quantized_student

    def run_all_experiments(self, train_new=False):
        """Run all experiments"""
        print("\n" + "=" * 70)
        print("RUNNING COMPREHENSIVE OPTIMIZATION EXPERIMENTS")
        print("=" * 70)

        # Run all experiments
        self.run_experiment_1_baseline(train=train_new)
        self.run_experiment_2_moe(train=train_new)
        self.run_experiment_3_ptq_baseline()
        self.run_experiment_4_moe_ptq()
        self.run_experiment_5_kd_baseline(train=train_new)
        self.run_experiment_6_kd_moe(train=train_new)
        self.run_experiment_7_kd_ptq()

        # Generate report
        self.generate_report()

    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("\n" + "=" * 70)
        print("COMPREHENSIVE RESULTS SUMMARY")
        print("=" * 70 + "\n")

        # Create DataFrame
        df = pd.DataFrame(self.results)

        # Sort by accuracy
        df = df.sort_values("accuracy", ascending=False)

        # Print table
        print(df.to_string(index=False))
        print("\n" + "=" * 70)

        # Calculate trade-offs
        baseline_result = df[df["model_name"] == "Baseline"].iloc[0]

        print("\nTRADE-OFF ANALYSIS (vs Baseline):")
        print("-" * 70)

        for _, row in df.iterrows():
            if row["model_name"] == "Baseline":
                continue

            acc_change = row["accuracy"] - baseline_result["accuracy"]
            size_reduction = (
                1 - row["model_size_mb"] / baseline_result["model_size_mb"]
            ) * 100
            time_change = (
                1 - row["inference_time"] / baseline_result["inference_time"]
            ) * 100

            print(f"\n{row['model_name']}:")
            print(f"  Accuracy: {acc_change:+.2f}%")
            print(
                f"  Size: {size_reduction:+.1f}% {'reduction' if size_reduction > 0 else 'increase'}"
            )
            print(
                f"  Speed: {time_change:+.1f}% {'faster' if time_change > 0 else 'slower'}"
            )

        print("\n" + "=" * 70)

        # Save results to JSON
        with open("experiment_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        print("\nResults saved to 'experiment_results.json'")

        # Save to CSV
        df.to_csv("experiment_results.csv", index=False)
        print("Results saved to 'experiment_results.csv'")

        return df


def main():
    """Main execution"""
    runner = ExperimentRunner()

    print("=" * 70)
    print("MODEL OPTIMIZATION ASSIGNMENT - COMPLETE IMPLEMENTATION")
    print("=" * 70)
    print("\nThis script will:")
    print("  1. Train/Load Baseline Model")
    print("  2. Train/Load MoE Model")
    print("  3. Apply PTQ to Baseline")
    print("  4. Apply PTQ to MoE")
    print("  5. Apply KD (Baseline → Student)")
    print("  6. Apply KD (MoE → Student)")
    print("  7. Apply PTQ to Distilled Student")

    # Check for existing checkpoints
    print("\n" + "=" * 70)
    print("CHECKING FOR EXISTING CHECKPOINTS")
    print("=" * 70)
    checkpoints = {
        "baseline_model.pth": "Baseline model",
        "moe_model_e4_k2.pth": "MoE model",
        "student_distilled.pth": "Student (Baseline teacher)",
        "student_distilled_moe.pth": "Student (MoE teacher)",
    }

    missing_checkpoints = []
    for checkpoint, description in checkpoints.items():
        if os.path.exists(checkpoint):
            print(f"  ✓ Found: {checkpoint} ({description})")
        else:
            print(f"  ✗ Missing: {checkpoint} ({description})")
            missing_checkpoints.append(checkpoint)

    if missing_checkpoints:
        print(f"\n⚠️  {len(missing_checkpoints)} checkpoint(s) missing.")
        print("These will be trained automatically even in 'load' mode.")
    else:
        print("\n✓ All checkpoints found!")

    print("\nDo you want to train new models or use existing checkpoints?")
    print("(Training new models will take ~15-20 minutes)")

    choice = (
        input("\nEnter 'train' for new training or 'load' to use checkpoints: ")
        .strip()
        .lower()
    )

    train_new = choice == "train"

    # Run all experiments
    runner.run_all_experiments(train_new=train_new)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - experiment_results.json")
    print("  - experiment_results.csv")
    print("  - Multiple model checkpoints (.pth files)")


if __name__ == "__main__":
    main()
