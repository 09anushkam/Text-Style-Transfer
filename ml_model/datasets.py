"""
Sarcasm Detection Datasets Module
Handles loading and preprocessing of multiple sarcasm detection datasets
"""

import pandas as pd
import numpy as np
import json
import requests
import os
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
import re

class SarcasmDatasetLoader:
    """Loads and preprocesses sarcasm detection datasets"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def download_datasets(self):
        """Download popular sarcasm detection datasets"""
        
        # Dataset URLs
        datasets = {
            "news_headlines": "https://storage.googleapis.com/dataset-uploader/sarcasm/Sarcasm_Headlines_Dataset.json",
            "reddit_comments": "https://storage.googleapis.com/dataset-uploader/sarcasm/reddit_sarcasm.json",
            "twitter_sarcasm": "https://storage.googleapis.com/dataset-uploader/sarcasm/twitter_sarcasm.json"
        }
        
        for name, url in datasets.items():
            file_path = os.path.join(self.data_dir, f"{name}.json")
            if not os.path.exists(file_path):
                try:
                    print(f"Downloading {name} dataset...")
                    response = requests.get(url)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(response.text)
                    print(f"Downloaded {name} dataset successfully!")
                except Exception as e:
                    print(f"Failed to download {name}: {e}")
                    # Create sample data if download fails
                    self._create_sample_data(name, file_path)
    
    def _create_sample_data(self, dataset_name: str, file_path: str):
        """Create sample data if download fails"""
        
        if dataset_name == "news_headlines":
            sample_data = [
                {"headline": "Scientists discover that water is wet", "is_sarcastic": 1},
                {"headline": "Breaking: Local man discovers gravity still works", "is_sarcastic": 1},
                {"headline": "New study shows exercise is good for health", "is_sarcastic": 0},
                {"headline": "Weather forecast predicts rain in rainy season", "is_sarcastic": 1},
                {"headline": "Company announces quarterly profits increased", "is_sarcastic": 0},
                {"headline": "Man shocked to learn that 2+2 equals 4", "is_sarcastic": 1},
                {"headline": "Government announces new healthcare initiative", "is_sarcastic": 0},
                {"headline": "Local cat successfully completes nap", "is_sarcastic": 1},
                {"headline": "Technology company releases new smartphone", "is_sarcastic": 0},
                {"headline": "Person surprised that Monday follows Sunday", "is_sarcastic": 1}
            ]
        elif dataset_name == "reddit_comments":
            sample_data = [
                {"comment": "Oh great, another Monday. Just what I needed.", "label": 1},
                {"comment": "I love how my computer decides to update right before my presentation", "label": 1},
                {"comment": "This is actually a really good solution to the problem", "label": 0},
                {"comment": "Wow, you're such a genius for that brilliant idea", "label": 1},
                {"comment": "Thanks for the helpful information", "label": 0},
                {"comment": "I'm so excited to be stuck in traffic for 2 hours", "label": 1},
                {"comment": "The weather is nice today", "label": 0},
                {"comment": "Perfect timing for the internet to go down", "label": 1},
                {"comment": "I appreciate your effort on this project", "label": 0},
                {"comment": "What a wonderful way to start my day", "label": 1}
            ]
        else:  # twitter_sarcasm
            sample_data = [
                {"tweet": "Just love it when my phone dies at 1%", "sarcastic": 1},
                {"tweet": "Great weather for a picnic today", "sarcastic": 0},
                {"tweet": "Oh fantastic, another software update", "sarcastic": 1},
                {"tweet": "Happy to help with your request", "sarcastic": 0},
                {"tweet": "I'm thrilled to be on hold for 45 minutes", "sarcastic": 1},
                {"tweet": "Congratulations on your achievement", "sarcastic": 0},
                {"tweet": "What a surprise, the meeting is running late", "sarcastic": 1},
                {"tweet": "Thank you for the clear explanation", "sarcastic": 0},
                {"tweet": "I'm absolutely delighted with this service", "sarcastic": 1},
                {"tweet": "The presentation was very informative", "sarcastic": 0}
            ]
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2)
    
    def load_news_headlines(self) -> Tuple[List[str], List[int]]:
        """Load news headlines dataset"""
        file_path = os.path.join(self.data_dir, "news_headlines.json")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            texts = []
            labels = []
            
            for item in data:
                if isinstance(item, dict):
                    texts.append(item.get('headline', ''))
                    labels.append(item.get('is_sarcastic', 0))
            
            return texts, labels
        except Exception as e:
            print(f"Error loading news headlines: {e}")
            return [], []
    
    def load_reddit_comments(self) -> Tuple[List[str], List[int]]:
        """Load Reddit comments dataset"""
        file_path = os.path.join(self.data_dir, "reddit_comments.json")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            texts = []
            labels = []
            
            for item in data:
                if isinstance(item, dict):
                    texts.append(item.get('comment', ''))
                    labels.append(item.get('label', 0))
            
            return texts, labels
        except Exception as e:
            print(f"Error loading Reddit comments: {e}")
            return [], []
    
    def load_twitter_sarcasm(self) -> Tuple[List[str], List[int]]:
        """Load Twitter sarcasm dataset"""
        file_path = os.path.join(self.data_dir, "twitter_sarcasm.json")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            texts = []
            labels = []
            
            for item in data:
                if isinstance(item, dict):
                    texts.append(item.get('tweet', ''))
                    labels.append(item.get('sarcastic', 0))
            
            return texts, labels
        except Exception as e:
            print(f"Error loading Twitter sarcasm: {e}")
            return [], []
    
    def load_combined_dataset(self) -> Tuple[List[str], List[int]]:
        """Load and combine all available datasets"""
        all_texts = []
        all_labels = []
        
        # Load each dataset
        datasets = [
            self.load_news_headlines(),
            self.load_reddit_comments(),
            self.load_twitter_sarcasm()
        ]
        
        for texts, labels in datasets:
            all_texts.extend(texts)
            all_labels.extend(labels)
        
        # Filter out empty texts
        filtered_texts = []
        filtered_labels = []
        
        for text, label in zip(all_texts, all_labels):
            if text and len(text.strip()) > 0:
                filtered_texts.append(text.strip())
                filtered_labels.append(label)
        
        return filtered_texts, filtered_labels
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for training"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def get_train_test_split(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[List[str], List[str], List[int], List[int]]:
        """Get train-test split of the combined dataset"""
        texts, labels = self.load_combined_dataset()
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            processed_texts, labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=labels
        )
        
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Test the dataset loader
    loader = SarcasmDatasetLoader()
    loader.download_datasets()
    
    X_train, X_test, y_train, y_test = loader.get_train_test_split()
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Sarcastic samples in training: {sum(y_train)}")
    print(f"Sarcastic samples in test: {sum(y_test)}")
    
    print("\nSample training texts:")
    for i, (text, label) in enumerate(zip(X_train[:5], y_train[:5])):
        print(f"{i+1}. [{label}] {text}")
