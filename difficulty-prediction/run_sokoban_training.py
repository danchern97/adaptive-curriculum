#!/usr/bin/env python3
"""
Example training script for Sokoban difficulty prediction using RoPE embeddings.
"""

import subprocess
import sys
import os

def run_sokoban_training():
    """Run training on Sokoban data with RoPE embeddings."""
    
    # Define training arguments
    args = [
        "python", "train.py",
        "--use_sokoban",
        "--sokoban_data_path", "../data",
        "--use_rope", 
        "--rope_dim", "128",
        "--max_seq_len", "2048",
        "--model_name", "distilbert-base-uncased",
        "--method", "residual",
        "--scaling", "platt",
        "--lr", "1e-4",
        "--batch_size_per_gpu", "4",
        "--ref_size", "64", 
        "--epochs", "5",
        "--hidden_size", "512",
        "--num_layers", "2",
        "--tau", "1.0",
        "--output_dir", "sokoban_experiments",
        "--seed", "42",
        "--use_scheduler",
        "--warmup_steps", "50"
    ]
    
    print("🚀 Starting Sokoban difficulty prediction training with RoPE embeddings")
    print("📊 Wandb logging enabled for experiment tracking")
    print(f"Command: {' '.join(args)}")
    print("-" * 60)
    
    # Run the training
    try:
        result = subprocess.run(args, cwd=os.path.dirname(__file__), check=True)
        print("✅ Training completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error code {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"❌ Error running training: {e}")
        return 1

def run_comparison_without_rope():
    """Run training without RoPE for comparison."""
    
    args = [
        "python", "train.py",
        "--use_sokoban",
        "--sokoban_data_path", "../data",
        # No --use_rope flag
        "--model_name", "distilbert-base-uncased", 
        "--method", "residual",
        "--scaling", "platt",
        "--lr", "1e-4",
        "--batch_size_per_gpu", "4",
        "--ref_size", "64",
        "--epochs", "5", 
        "--hidden_size", "512",
        "--num_layers", "2",
        "--tau", "1.0",
        "--output_dir", "sokoban_experiments_no_rope",
        "--seed", "42",
        "--use_scheduler",
        "--warmup_steps", "50"
    ]
    
    print("🔄 Starting comparison training without RoPE embeddings")
    print("📊 Wandb logging enabled for experiment tracking")
    print(f"Command: {' '.join(args)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(args, cwd=os.path.dirname(__file__), check=True)
        print("✅ Comparison training completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"❌ Comparison training failed with error code {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"❌ Error running comparison training: {e}")
        return 1

if __name__ == "__main__":
    print("🧩 Sokoban Difficulty Prediction Training")
    print("=" * 50)
    
    # Check if we should run both experiments
    run_comparison = "--compare" in sys.argv
    
    # Run with RoPE
    print("\\n1. Training with RoPE embeddings...")
    rope_result = run_sokoban_training()
    
    if run_comparison:
        print("\\n2. Training without RoPE for comparison...")
        no_rope_result = run_comparison_without_rope()
        
        print("\\n📊 Results Summary:")
        print(f"   RoPE training: {'✅ Success' if rope_result == 0 else '❌ Failed'}")
        print(f"   No-RoPE training: {'✅ Success' if no_rope_result == 0 else '❌ Failed'}")
    else:
        print("\\n💡 To run comparison without RoPE, use: python run_sokoban_training.py --compare")
    
    print("\\n🎉 Training experiments completed!")
    print("\\nResults will be saved in:")
    print("   - sokoban_experiments/ (with RoPE)")
    if run_comparison:
        print("   - sokoban_experiments_no_rope/ (without RoPE)")