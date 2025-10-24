"""
Sarcasm Detection Model Architecture
Implements multiple neural network architectures for sarcasm detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from typing import Dict, List, Tuple, Optional
import math

class AttentionLayer(nn.Module):
    """Self-attention layer for capturing important words"""
    
    def __init__(self, hidden_dim: int):
        super(AttentionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_dim)
        attention_weights = F.softmax(self.attention(lstm_output), dim=1)
        # Weighted sum of LSTM outputs
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights

class SarcasmLSTM(nn.Module):
    """LSTM-based sarcasm detection model with attention"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, 
                 hidden_dim: int = 64, num_layers: int = 2, 
                 dropout: float = 0.3, num_classes: int = 2):
        super(SarcasmLSTM, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM layers
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        
        # Attention layer
        self.attention = AttentionLayer(hidden_dim * 2)  # *2 for bidirectional
        
        # Classification layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_len, hidden_dim*2)
        
        # Apply attention
        context_vector, attention_weights = self.attention(lstm_out)
        
        # Classification
        output = self.dropout(context_vector)
        output = F.relu(self.fc1(output))
        output = self.dropout(output)
        output = self.fc2(output)
        
        return output, attention_weights

class SarcasmCNN(nn.Module):
    """CNN-based sarcasm detection model"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, 
                 num_filters: int = 100, filter_sizes: List[int] = [3, 4, 5],
                 dropout: float = 0.3, num_classes: int = 2):
        super(SarcasmCNN, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (filter_size, embedding_dim))
            for filter_size in filter_sizes
        ])
        
        # Dropout and classification
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embedded = embedded.unsqueeze(1)  # (batch_size, 1, seq_len, embedding_dim)
        
        # Apply convolutions
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))  # (batch_size, num_filters, seq_len, 1)
            conv_out = conv_out.squeeze(3)  # (batch_size, num_filters, seq_len)
            pooled = F.max_pool1d(conv_out, conv_out.size(2))  # (batch_size, num_filters, 1)
            pooled = pooled.squeeze(2)  # (batch_size, num_filters)
            conv_outputs.append(pooled)
        
        # Concatenate all conv outputs
        concatenated = torch.cat(conv_outputs, dim=1)  # (batch_size, len(filter_sizes) * num_filters)
        
        # Classification
        output = self.dropout(concatenated)
        output = self.fc(output)
        
        return output

class SarcasmTransformer(nn.Module):
    """Transformer-based sarcasm detection model"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, 
                 num_heads: int = 8, num_layers: int = 4,
                 dropout: float = 0.3, num_classes: int = 2, max_len: int = 512):
        super(SarcasmTransformer, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, embedding_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        seq_len = x.size(1)
        
        # Create position indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        
        # Embeddings
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        embedded = token_emb + pos_emb
        
        # Create attention mask for padding
        attention_mask = (x != 0).float()
        
        # Transformer forward pass
        transformer_out = self.transformer(embedded, src_key_padding_mask=(x == 0))
        
        # Global average pooling
        pooled = torch.mean(transformer_out, dim=1)
        
        # Classification
        output = self.dropout(pooled)
        output = self.fc(output)
        
        return output

class SarcasmEnsemble(nn.Module):
    """Ensemble model combining LSTM, CNN, and Transformer"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, 
                 hidden_dim: int = 64, num_classes: int = 2):
        super(SarcasmEnsemble, self).__init__()
        
        # Individual models
        self.lstm_model = SarcasmLSTM(vocab_size, embedding_dim, hidden_dim, num_classes=num_classes)
        self.cnn_model = SarcasmCNN(vocab_size, embedding_dim, num_classes=num_classes)
        self.transformer_model = SarcasmTransformer(vocab_size, embedding_dim, num_classes=num_classes)
        
        # Ensemble fusion
        self.fusion = nn.Linear(num_classes * 3, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # Get predictions from each model
        lstm_out, _ = self.lstm_model(x)
        cnn_out = self.cnn_model(x)
        transformer_out = self.transformer_model(x)
        
        # Concatenate outputs
        combined = torch.cat([lstm_out, cnn_out, transformer_out], dim=1)
        
        # Final prediction
        output = self.dropout(combined)
        output = self.fusion(output)
        
        return output

class ModelFactory:
    """Factory class for creating different model architectures"""
    
    @staticmethod
    def create_model(model_type: str, vocab_size: int, **kwargs) -> nn.Module:
        """Create a model based on the specified type"""
        
        if model_type.lower() == "lstm":
            return SarcasmLSTM(vocab_size, **kwargs)
        elif model_type.lower() == "cnn":
            return SarcasmCNN(vocab_size, **kwargs)
        elif model_type.lower() == "transformer":
            return SarcasmTransformer(vocab_size, **kwargs)
        elif model_type.lower() == "ensemble":
            return SarcasmEnsemble(vocab_size, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_model_info(model_type: str) -> Dict[str, str]:
        """Get information about different model types"""
        
        info = {
            "lstm": {
                "description": "LSTM with attention mechanism",
                "pros": "Good at capturing sequential patterns, handles variable length sequences",
                "cons": "Can be slow for long sequences, may struggle with very long dependencies",
                "best_for": "General sarcasm detection, text with clear sequential patterns"
            },
            "cnn": {
                "description": "Convolutional Neural Network",
                "pros": "Fast training and inference, good at capturing local patterns",
                "cons": "Limited ability to capture long-range dependencies",
                "best_for": "Short texts, pattern-based sarcasm detection"
            },
            "transformer": {
                "description": "Transformer encoder with self-attention",
                "pros": "Excellent at capturing long-range dependencies, parallelizable",
                "cons": "Requires more data, computationally intensive",
                "best_for": "Complex sarcasm patterns, long texts"
            },
            "ensemble": {
                "description": "Combination of LSTM, CNN, and Transformer",
                "pros": "Best overall performance, robust to different text types",
                "cons": "Larger model size, slower inference",
                "best_for": "Production systems requiring high accuracy"
            }
        }
        
        return info.get(model_type.lower(), {})

if __name__ == "__main__":
    # Test model creation
    vocab_size = 10000
    
    print("Testing model architectures...")
    
    # Test LSTM
    lstm_model = ModelFactory.create_model("lstm", vocab_size)
    print(f"LSTM model created: {sum(p.numel() for p in lstm_model.parameters())} parameters")
    
    # Test CNN
    cnn_model = ModelFactory.create_model("cnn", vocab_size)
    print(f"CNN model created: {sum(p.numel() for p in cnn_model.parameters())} parameters")
    
    # Test Transformer
    transformer_model = ModelFactory.create_model("transformer", vocab_size)
    print(f"Transformer model created: {sum(p.numel() for p in transformer_model.parameters())} parameters")
    
    # Test Ensemble
    ensemble_model = ModelFactory.create_model("ensemble", vocab_size)
    print(f"Ensemble model created: {sum(p.numel() for p in ensemble_model.parameters())} parameters")
    
    # Print model information
    print("\nModel Information:")
    for model_type in ["lstm", "cnn", "transformer", "ensemble"]:
        info = ModelFactory.get_model_info(model_type)
        print(f"\n{model_type.upper()}:")
        print(f"  Description: {info.get('description', 'N/A')}")
        print(f"  Pros: {info.get('pros', 'N/A')}")
        print(f"  Cons: {info.get('cons', 'N/A')}")
        print(f"  Best for: {info.get('best_for', 'N/A')}")
