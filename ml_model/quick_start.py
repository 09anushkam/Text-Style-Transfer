"""
Quick Start Script for ML-based Sarcasm Detection
Automated setup and training script
"""

#!/usr/bin/env python3
"""
Quick Start Script for Sarcasm Detection ML System
This script provides an easy way to get started with training sarcasm detection models
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'torch', 'numpy', 'pandas', 'scikit-learn', 
        'matplotlib', 'seaborn', 'requests'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
        print("Packages installed successfully!")
    else:
        print("All required packages are available!")

def setup_directories():
    """Create necessary directories"""
    directories = ['data', 'models', 'logs']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"Created directory: {directory}")

def run_quick_training(model_type='lstm', epochs=5):
    """Run a quick training session"""
    print(f"\n🚀 Starting quick training with {model_type.upper()} model...")
    print(f"Epochs: {epochs}")
    print("This will train a model with sample data for demonstration purposes.")
    
    # Import training modules
    try:
        from train import main as train_main
        from datasets import SarcasmDatasetLoader
        from models import ModelFactory
        from training import SarcasmTrainer, SarcasmDataset, SarcasmTokenizer
        from torch.utils.data import DataLoader
        
        print("✅ All modules imported successfully!")
        
        # Set up arguments for training
        sys.argv = [
            'train.py',
            '--model_type', model_type,
            '--epochs', str(epochs),
            '--batch_size', '16',
            '--learning_rate', '0.001',
            '--embedding_dim', '64',
            '--hidden_dim', '32',
            '--max_length', '64',
            '--vocab_size', '5000',
            '--device', 'cpu'
        ]
        
        # Run training
        train_main()
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        print("Please check the error and try again.")

def run_model_test():
    """Test a trained model"""
    print("\n🧪 Testing trained model...")
    
    # Find the most recent model
    models_dir = Path('models')
    if not models_dir.exists():
        print("❌ No models directory found. Please train a model first.")
        return
    
    model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
    if not model_dirs:
        print("❌ No trained models found. Please train a model first.")
        return
    
    # Get the most recent model
    latest_model = max(model_dirs, key=lambda x: x.stat().st_mtime)
    print(f"Testing model: {latest_model}")
    
    try:
        from inference import test_model
        test_model(str(latest_model))
    except Exception as e:
        print(f"❌ Error testing model: {e}")

def show_model_comparison():
    """Show comparison of different model types"""
    print("\n📊 Model Architecture Comparison:")
    print("=" * 60)
    
    try:
        from models import ModelFactory
        
        model_types = ['lstm', 'cnn', 'transformer', 'ensemble']
        vocab_size = 10000
        
        for model_type in model_types:
            try:
                model = ModelFactory.create_model(model_type, vocab_size)
                param_count = sum(p.numel() for p in model.parameters())
                info = ModelFactory.get_model_info(model_type)
                
                print(f"\n{model_type.upper()}:")
                print(f"  Parameters: {param_count:,}")
                print(f"  Description: {info.get('description', 'N/A')}")
                print(f"  Best for: {info.get('best_for', 'N/A')}")
                
            except Exception as e:
                print(f"\n{model_type.upper()}: Error creating model - {e}")
                
    except Exception as e:
        print(f"❌ Error in model comparison: {e}")

def main():
    parser = argparse.ArgumentParser(description='Quick Start for Sarcasm Detection ML System')
    parser.add_argument('--action', type=str, default='setup',
                       choices=['setup', 'train', 'test', 'compare', 'all'],
                       help='Action to perform')
    parser.add_argument('--model_type', type=str, default='lstm',
                       choices=['lstm', 'cnn', 'transformer', 'ensemble'],
                       help='Model type for training')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of epochs for quick training')
    
    args = parser.parse_args()
    
    print("🎭 Sarcasm Detection ML System - Quick Start")
    print("=" * 50)
    
    if args.action in ['setup', 'all']:
        print("\n1️⃣ Setting up environment...")
        check_requirements()
        setup_directories()
        print("✅ Setup completed!")
    
    if args.action in ['compare', 'all']:
        print("\n2️⃣ Comparing model architectures...")
        show_model_comparison()
    
    if args.action in ['train', 'all']:
        print("\n3️⃣ Running quick training...")
        run_quick_training(args.model_type, args.epochs)
    
    if args.action in ['test', 'all']:
        print("\n4️⃣ Testing trained model...")
        run_model_test()
    
    if args.action == 'all':
        print("\n🎉 Quick start completed!")
        print("\nNext steps:")
        print("1. Check the 'models' directory for your trained model")
        print("2. Use 'python inference.py --model_path <model_path>' to test")
        print("3. Integrate with your application using the inference module")
        print("4. For production, train with more epochs and larger datasets")

if __name__ == "__main__":
    main()
