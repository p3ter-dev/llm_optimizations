#!/usr/bin/env python3
"""
Master Script: Complete Assignment Workflow
It ensures all models are trained, evaluated, and analyzed properly.

Usage:
    python master_workflow.py

The script will:
1. Check if required packages are installed
2. Train or load all models
3. Run comprehensive experiments
4. Generate visualizations
5. Create analysis files for your report
"""

import os
import subprocess
import sys


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(text.center(70))
    print("=" * 70 + "\n")


def print_step(step_num, total_steps, description):
    """Print step information"""
    print(f"\n{'=' * 70}")
    print(f"STEP {step_num}/{total_steps}: {description}")
    print(f"{'=' * 70}\n")


def check_packages():
    """Check if required packages are installed"""
    print_header("CHECKING DEPENDENCIES")

    required = {
        "torch": "PyTorch",
        "torchvision": "TorchVision",
        "numpy": "NumPy",
        "pandas": "Pandas",
        "matplotlib": "Matplotlib",
        "seaborn": "Seaborn",
    }

    missing = []
    for package, name in required.items():
        try:
            __import__(package)
            print(f"✓ {name} is installed")
        except ImportError:
            print(f"✗ {name} is NOT installed")
            missing.append(package)

    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        return False

    print("\nAll dependencies are installed!")
    return True


def run_script(script_name, description):
    """Run a Python script"""
    print(f"\nRunning: {script_name}")
    print(f"Purpose: {description}")
    print("-" * 70)

    try:
        result = subprocess.run(
            [sys.executable, script_name], capture_output=False, text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error running {script_name}: {e}")
        return False


def file_exists(filename):
    """Check if file exists"""
    return os.path.exists(filename)


def main():
    """Main workflow"""
    print_header("MODEL OPTIMIZATION ASSIGNMENT")
    print("This script will guide you through the entire assignment.")
    print("Estimated total time: 20-30 minutes (if training from scratch)")

    # Check dependencies
    if not check_packages():
        print("\n Please install missing packages before continuing.")
        return

    print("\n" + "=" * 70)
    input("Press Enter to start the workflow...")

    total_steps = 7

    # Step 1: Train Baseline
    print_step(1, total_steps, "Train Baseline Model")
    print("This creates your foundation for comparison.")

    if file_exists("baseline_model.pth"):
        print("baseline_model.pth already exists")
        response = input("Retrain baseline? (y/n): ").strip().lower()
        if response == "y":
            success = run_script("baseline_mnist.py", "Training baseline FFN")
            if not success:
                print("Baseline training failed. Check errors above.")
                return
    else:
        print("Training baseline model (this will take ~3-5 minutes)...")
        success = run_script("baseline_mnist.py", "Training baseline FFN")
        if not success:
            print("Baseline training failed. Check errors above.")
            return

    print("\nStep 1 complete!")
    input("Press Enter to continue...")

    # Step 2: Train MoE
    print_step(2, total_steps, "Train Mixture of Experts Model")
    print("This demonstrates conditional computation with routing.")

    if file_exists("moe_model_e4_k2.pth"):
        print("MoE model already exists")
        response = input("Retrain MoE? (y/n): ").strip().lower()
        if response == "y":
            success = run_script(
                "moe_mnist.py", "Training MoE with 4 experts, top-2 routing"
            )
            if not success:
                print("MoE training failed. Check errors above.")
                return
    else:
        print("Training MoE model (this will take ~4-6 minutes)...")
        success = run_script(
            "moe_mnist.py", "Training MoE with 4 experts, top-2 routing"
        )
        if not success:
            print("MoE training failed. Check errors above.")
            return

    print("\nStep 2 complete!")
    input("Press Enter to continue...")

    # Step 3: Apply PTQ
    print_step(3, total_steps, "Apply Post-Training Quantization")
    print("This compresses models to INT8 precision.")

    print("Applying quantization to baseline model...")
    success = run_script("ptq_mnist.py", "Quantizing baseline model")
    if not success:
        print("PTQ failed. Check errors above.")
        return

    print("\nStep 3 complete!")
    input("Press Enter to continue...")

    # Step 4: Apply Knowledge Distillation
    print_step(4, total_steps, "Apply Knowledge Distillation")
    print("This trains a smaller student from the teacher models.")

    if file_exists("student_distilled.pth"):
        print("Distilled student already exists")
        response = input("Retrain student? (y/n): ").strip().lower()
        if response == "y":
            success = run_script("kd_mnist.py", "Distilling knowledge from baseline")
            if not success:
                print("Knowledge distillation failed. Check errors above.")
                return
    else:
        print("Training student via distillation (this will take ~3-5 minutes)...")
        success = run_script("kd_mnist.py", "Distilling knowledge from baseline")
        if not success:
            print("Knowledge distillation failed. Check errors above.")
            return

    print("\nStep 4 complete!")
    input("Press Enter to continue...")

    # Step 5: Run All Experiments
    print_step(5, total_steps, "Run Comprehensive Experiments")
    print("This evaluates all model combinations and generates results.")

    print("\nRunning comprehensive experiment suite...")
    print("This will:")
    print("  - Evaluate all 7 model configurations")
    print("  - Generate comparison tables")
    print("  - Save results to JSON and CSV")
    print()

    # We need to handle the interactive prompt
    print("Note: The script will ask if you want to 'train' or 'load'.")
    print("Choose 'load' since we've already trained the models.")
    input("\nPress Enter to continue...")

    success = run_script("run_all_experiments.py", "Running all experiments")
    if not success:
        print("Experiment suite failed. Check errors above.")
        return

    print("\nStep 5 complete!")
    input("Press Enter to continue...")

    # Step 6: Generate Visualizations
    print_step(6, total_steps, "Generate Visualizations")
    print("This creates plots and analysis for your report.")

    if not file_exists("experiment_results.json"):
        print("experiment_results.json not found!")
        print("Please run run_all_experiments.py first.")
        return

    print("Generating plots and insights...")
    success = run_script("visualize_results.py", "Creating visualizations and analysis")
    if not success:
        print("Visualization generation failed. Check errors above.")
        return

    print("\nStep 6 complete!")
    input("Press Enter to continue...")

    # Step 7: Summary
    print_step(7, total_steps, "Complete!")

    print_header("ALL STEPS COMPLETE!")

    print("Generated Files:")
    print("-" * 70)

    files = {
        "Models": [
            "baseline_model.pth",
            "moe_model_e4_k2.pth",
            "baseline_ptq.pth",
            "moe_ptq.pth",
            "student_distilled.pth",
            "student_distilled_moe.pth",
            "student_ptq.pth",
        ],
        "Results": [
            "experiment_results.json",
            "experiment_results.csv",
            "insights.txt",
        ],
        "Visualizations": [
            "accuracy_vs_size.png",
            "accuracy_vs_time.png",
            "pareto_frontier.png",
            "comparison_bars.png",
            "normalized_comparison.png",
        ],
    }

    for category, file_list in files.items():
        print(f"\n{category}:")
        for f in file_list:
            if file_exists(f):
                print(f"  ✓ {f}")
            else:
                print(f"  ✗ {f} (missing)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        sys.exit(1)
