"""
Training Pipeline for Sarcasm Detection
Handles model training, validation, and evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os
import json
from datetime import datetime
import pickle

class SarcasmDataset(Dataset):
    """PyTorch Dataset for sarcasm detection"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize text
        tokens = self.tokenizer.encode(text, max_length=self.max_length, 
                                     padding='max_length', truncation=True)
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

class SarcasmTokenizer:
    """Simple tokenizer for sarcasm detection"""
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        self.word_counts = {}
        
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from texts"""
        # Count word frequencies
        for text in texts:
            words = text.lower().split()
            for word in words:
                self.word_counts[word] = self.word_counts.get(word, 0) + 1
        
        # Sort by frequency and build vocab
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        
        idx = 2
        for word, count in sorted_words:
            if idx >= self.vocab_size:
                break
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
            idx += 1
    
    def encode(self, text: str, max_length: int = 128, padding: str = 'max_length', 
               truncation: bool = True) -> List[int]:
        """Encode text to token IDs"""
        words = text.lower().split()
        
        # Convert words to IDs
        token_ids = []
        for word in words:
            token_ids.append(self.word_to_idx.get(word, 1))  # 1 is <UNK>
        
        # Truncate if necessary
        if truncation and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        
        # Pad if necessary
        if padding == 'max_length':
            while len(token_ids) < max_length:
                token_ids.append(0)  # 0 is <PAD>
        
        return token_ids

class SarcasmTrainer:
    """Trainer class for sarcasm detection models"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self, dataloader: DataLoader, optimizer: optim.Optimizer, 
                   criterion: nn.Module) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['label'].to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            if isinstance(self.model, nn.Module) and hasattr(self.model, 'forward'):
                outputs = self.model(input_ids)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Handle models that return (output, attention)
            else:
                outputs = self.model(input_ids)
            
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, dataloader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                if isinstance(self.model, nn.Module) and hasattr(self.model, 'forward'):
                    outputs = self.model(input_ids)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]  # Handle models that return (output, attention)
                else:
                    outputs = self.model(input_ids)
                
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 10, learning_rate: float = 0.001, 
              save_path: str = "models") -> Dict:
        """Train the model"""
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
        
        os.makedirs(save_path, exist_ok=True)
        
        best_val_acc = 0
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Store history
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
            training_history['train_acc'].append(train_acc)
            training_history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), os.path.join(save_path, 'best_model.pth'))
                print(f"  New best model saved! Val Acc: {val_acc:.4f}")
        
        # Save final model
        torch.save(self.model.state_dict(), os.path.join(save_path, 'final_model.pth'))
        
        # Save training history
        with open(os.path.join(save_path, 'training_history.json'), 'w') as f:
            json.dump(training_history, f, indent=2)
        
        print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.4f}")
        
        return training_history
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """Evaluate the model on test data"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                if isinstance(self.model, nn.Module) and hasattr(self.model, 'forward'):
                    outputs = self.model(input_ids)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]  # Handle models that return (output, attention)
                else:
                    outputs = self.model(input_ids)
                
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist()
        }
        
        print("\nEvaluation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        return results
    
    def plot_training_history(self, save_path: str = "models"):
        """Plot training history"""
        if not self.train_losses:
            print("No training history to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'training_history.png'))
        plt.show()

class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    @staticmethod
    def evaluate_model(model: nn.Module, test_loader: DataLoader, 
                      device: str = 'cpu') -> Dict:
        """Evaluate model performance"""
        model.eval()
        model.to(device)
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                if isinstance(model, nn.Module) and hasattr(model, 'forward'):
                    outputs = model(input_ids)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]  # Handle models that return (output, attention)
                else:
                    outputs = model(input_ids)
                
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }
        
        return results
    
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, save_path: str = "models"):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Sarcastic', 'Sarcastic'],
                   yticklabels=['Not Sarcastic', 'Sarcastic'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
        plt.show()

if __name__ == "__main__":
    # Test the trainer
    print("Sarcasm Detection Training Pipeline")
    print("This module provides training and evaluation capabilities for sarcasm detection models.")
