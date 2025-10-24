"""
Main Training Script for Sarcasm Detection
Complete pipeline for training and evaluating sarcasm detection models
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets import SarcasmDatasetLoader
from models import ModelFactory
from training import SarcasmTrainer, SarcasmDataset, SarcasmTokenizer, ModelEvaluator

def main():
    parser = argparse.ArgumentParser(description='Train Sarcasm Detection Model')
    parser.add_argument('--model_type', type=str, default='lstm', 
                       choices=['lstm', 'cnn', 'transformer', 'ensemble'],
                       help='Type of model to train')
    parser.add_argument('--epochs', type=int, default=20, 
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                       help='Learning rate')
    parser.add_argument('--embedding_dim', type=int, default=128, 
                       help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=64, 
                       help='Hidden dimension for LSTM')
    parser.add_argument('--max_length', type=int, default=128, 
                       help='Maximum sequence length')
    parser.add_argument('--vocab_size', type=int, default=10000, 
                       help='Vocabulary size')
    parser.add_argument('--device', type=str, default='auto', 
                       help='Device to use (cpu, cuda, auto)')
    parser.add_argument('--save_dir', type=str, default='models', 
                       help='Directory to save models')
    parser.add_argument('--data_dir', type=str, default='data', 
                       help='Directory containing datasets')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("=" * 60)
    print("SARCASM DETECTION MODEL TRAINING")
    print("=" * 60)
    print(f"Model Type: {args.model_type.upper()}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Embedding Dim: {args.embedding_dim}")
    print(f"Hidden Dim: {args.hidden_dim}")
    print(f"Max Length: {args.max_length}")
    print(f"Vocab Size: {args.vocab_size}")
    print("=" * 60)
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_dir = os.path.join(args.save_dir, f"{args.model_type}_{timestamp}")
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Load datasets
    print("\n1. Loading and preprocessing datasets...")
    loader = SarcasmDatasetLoader(args.data_dir)
    loader.download_datasets()
    
    X_train, X_test, y_train, y_test = loader.get_train_test_split()
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Sarcastic samples in training: {sum(y_train)} ({sum(y_train)/len(y_train)*100:.1f}%)")
    print(f"Sarcastic samples in test: {sum(y_test)} ({sum(y_test)/len(y_test)*100:.1f}%)")
    
    # Build tokenizer
    print("\n2. Building tokenizer...")
    tokenizer = SarcasmTokenizer(args.vocab_size)
    tokenizer.build_vocab(X_train)
    
    print(f"Vocabulary size: {len(tokenizer.word_to_idx)}")
    
    # Save tokenizer
    with open(os.path.join(model_save_dir, 'tokenizer.pkl'), 'wb') as f:
        import pickle
        pickle.dump(tokenizer, f)
    
    # Create datasets
    print("\n3. Creating PyTorch datasets...")
    train_dataset = SarcasmDataset(X_train, y_train, tokenizer, args.max_length)
    test_dataset = SarcasmDataset(X_test, y_test, tokenizer, args.max_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    print("\n4. Creating model...")
    model = ModelFactory.create_model(
        args.model_type, 
        len(tokenizer.word_to_idx),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_classes=2
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Save model architecture
    model_info = {
        'model_type': args.model_type,
        'vocab_size': len(tokenizer.word_to_idx),
        'embedding_dim': args.embedding_dim,
        'hidden_dim': args.hidden_dim,
        'max_length': args.max_length,
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'timestamp': timestamp
    }
    
    with open(os.path.join(model_save_dir, 'model_info.json'), 'w') as f:
        json.dump(model_info, f, indent=2)
    
    # Train model
    print("\n5. Training model...")
    trainer = SarcasmTrainer(model, device)
    
    training_history = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,  # Using test as validation for simplicity
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        save_path=model_save_dir
    )
    
    # Evaluate model
    print("\n6. Evaluating model...")
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_model(model, test_loader, device)
    
    # Save results
    with open(os.path.join(model_save_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(
        np.array(results['confusion_matrix']), 
        model_save_dir
    )
    
    # Print final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")
    print(f"\nModel saved to: {model_save_dir}")
    print("=" * 60)
    
    # Save sample predictions
    print("\n7. Generating sample predictions...")
    model.eval()
    sample_texts = [
        "I love it when my computer crashes right before I save my work",
        "This is a great day for a picnic",
        "Oh fantastic, another meeting that could have been an email",
        "Thank you for the helpful information",
        "I'm so excited to be stuck in traffic for 2 hours",
        "The weather is beautiful today",
        "Wow, you're such a genius for that brilliant idea",
        "I appreciate your effort on this project"
    ]
    
    sample_predictions = []
    with torch.no_grad():
        for text in sample_texts:
            # Preprocess text
            processed_text = loader.preprocess_text(text)
            tokens = tokenizer.encode(processed_text, max_length=args.max_length)
            input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
            
            # Get prediction
            outputs = model(input_tensor)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            sample_predictions.append({
                'text': text,
                'predicted_class': 'Sarcastic' if predicted_class == 1 else 'Not Sarcastic',
                'confidence': confidence,
                'probabilities': {
                    'Not Sarcastic': probabilities[0][0].item(),
                    'Sarcastic': probabilities[0][1].item()
                }
            })
    
    # Save sample predictions
    with open(os.path.join(model_save_dir, 'sample_predictions.json'), 'w') as f:
        json.dump(sample_predictions, f, indent=2)
    
    print("\nSample Predictions:")
    for pred in sample_predictions:
        print(f"Text: {pred['text']}")
        print(f"Prediction: {pred['predicted_class']} (Confidence: {pred['confidence']:.3f})")
        print(f"Probabilities: {pred['probabilities']}")
        print("-" * 40)
    
    print(f"\nTraining completed successfully!")
    print(f"All files saved to: {model_save_dir}")

if __name__ == "__main__":
    main()
