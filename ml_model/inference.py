"""
Model Inference and Integration
Handles loading trained models and making predictions
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import json
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from models import ModelFactory
from training import SarcasmTokenizer

class SarcasmPredictor:
    """Sarcasm detection predictor using trained models"""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.model_info = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and tokenizer"""
        try:
            # Load model info
            with open(os.path.join(self.model_path, 'model_info.json'), 'r') as f:
                self.model_info = json.load(f)
            
            # Load tokenizer
            with open(os.path.join(self.model_path, 'tokenizer.pkl'), 'rb') as f:
                self.tokenizer = pickle.load(f)
            
            # Create model
            self.model = ModelFactory.create_model(
                self.model_info['model_type'],
                self.model_info['vocab_size'],
                embedding_dim=self.model_info['embedding_dim'],
                hidden_dim=self.model_info['hidden_dim'],
                num_classes=2
            )
            
            # Load model weights
            model_file = os.path.join(self.model_path, 'best_model.pth')
            if not os.path.exists(model_file):
                model_file = os.path.join(self.model_path, 'final_model.pth')
            
            if os.path.exists(model_file):
                self.model.load_state_dict(torch.load(model_file, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                print(f"Model loaded successfully from {model_file}")
            else:
                raise FileNotFoundError(f"No model file found in {self.model_path}")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for prediction"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        import re
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def predict(self, text: str) -> Dict:
        """Predict sarcasm for a single text"""
        if not text.strip():
            return {
                'text': text,
                'prediction': 'Not Sarcastic',
                'confidence': 0.5,
                'probabilities': {'Not Sarcastic': 0.5, 'Sarcastic': 0.5}
            }
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Tokenize
        tokens = self.tokenizer.encode(
            processed_text, 
            max_length=self.model_info['max_length']
        )
        
        # Convert to tensor
        input_tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Handle models that return (output, attention)
            
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'text': text,
            'prediction': 'Sarcastic' if predicted_class == 1 else 'Not Sarcastic',
            'confidence': confidence,
            'probabilities': {
                'Not Sarcastic': probabilities[0][0].item(),
                'Sarcastic': probabilities[0][1].item()
            }
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict sarcasm for multiple texts"""
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return self.model_info.copy()

class SarcasmAPI:
    """API wrapper for sarcasm detection"""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.predictor = SarcasmPredictor(model_path, device)
    
    def detect_sarcasm(self, text: str) -> Dict:
        """Detect sarcasm in text"""
        result = self.predictor.predict(text)
        
        return {
            'is_sarcastic': result['prediction'] == 'Sarcastic',
            'confidence': result['confidence'],
            'probabilities': result['probabilities'],
            'text': result['text']
        }
    
    def batch_detect(self, texts: List[str]) -> List[Dict]:
        """Detect sarcasm in multiple texts"""
        results = self.predictor.predict_batch(texts)
        
        return [
            {
                'is_sarcastic': result['prediction'] == 'Sarcastic',
                'confidence': result['confidence'],
                'probabilities': result['probabilities'],
                'text': result['text']
            }
            for result in results
        ]

def integrate_with_flask_app(model_path: str, device: str = 'cpu'):
    """Integration function for Flask app"""
    
    # Initialize predictor
    predictor = SarcasmPredictor(model_path, device)
    
    def ml_detect_sarcasm(text: str) -> Dict:
        """ML-based sarcasm detection function"""
        result = predictor.predict(text)
        
        return {
            'is_sarcastic': result['prediction'] == 'Sarcastic',
            'confidence': result['confidence'],
            'method': 'ml_model',
            'probabilities': result['probabilities']
        }
    
    def ml_convert_to_sarcastic(text: str) -> str:
        """Convert text to sarcastic using ML model"""
        # This is a placeholder - in practice, you'd need a separate model
        # or use the existing pattern-based conversion
        return text  # Return original for now
    
    def ml_convert_to_genuine(text: str) -> str:
        """Convert text to genuine using ML model"""
        # This is a placeholder - in practice, you'd need a separate model
        # or use the existing pattern-based conversion
        return text  # Return original for now
    
    return {
        'detect_sarcasm': ml_detect_sarcasm,
        'convert_to_sarcastic': ml_convert_to_sarcastic,
        'convert_to_genuine': ml_convert_to_genuine,
        'predictor': predictor
    }

def test_model(model_path: str, test_texts: List[str] = None):
    """Test a trained model with sample texts"""
    
    if test_texts is None:
        test_texts = [
            "I love it when my computer crashes right before I save my work",
            "This is a great day for a picnic",
            "Oh fantastic, another meeting that could have been an email",
            "Thank you for the helpful information",
            "I'm so excited to be stuck in traffic for 2 hours",
            "The weather is beautiful today",
            "Wow, you're such a genius for that brilliant idea",
            "I appreciate your effort on this project",
            "Perfect timing for the internet to go down",
            "Congratulations on your achievement"
        ]
    
    try:
        predictor = SarcasmPredictor(model_path)
        
        print("=" * 60)
        print("SARCASM DETECTION MODEL TEST")
        print("=" * 60)
        
        model_info = predictor.get_model_info()
        print(f"Model Type: {model_info['model_type'].upper()}")
        print(f"Training Samples: {model_info['training_samples']}")
        print(f"Test Samples: {model_info['test_samples']}")
        print(f"Parameters: {model_info['num_parameters']:,}")
        print("=" * 60)
        
        print("\nPredictions:")
        for i, text in enumerate(test_texts, 1):
            result = predictor.predict(text)
            print(f"{i:2d}. Text: {text}")
            print(f"    Prediction: {result['prediction']} (Confidence: {result['confidence']:.3f})")
            print(f"    Probabilities: {result['probabilities']}")
            print("-" * 40)
        
    except Exception as e:
        print(f"Error testing model: {e}")

if __name__ == "__main__":
    # Test the predictor
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Sarcasm Detection Model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model directory')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu, cuda)')
    
    args = parser.parse_args()
    
    test_model(args.model_path)
