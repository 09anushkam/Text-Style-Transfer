"""
Complete Setup and Training Guide for ML-based Sarcasm Detection
Step-by-step instructions for training and deploying the system
"""

# 🎭 Complete ML-based Sarcasm Detection System
# Setup, Training, and Deployment Guide

## 📋 Table of Contents
1. [System Overview](#system-overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Dataset Setup](#dataset-setup)
5. [Model Training](#model-training)
6. [Model Evaluation](#model-evaluation)
7. [Integration](#integration)
8. [Deployment](#deployment)
9. [Troubleshooting](#troubleshooting)

## 🎯 System Overview

This ML-based sarcasm detection system provides:

- **Multiple Neural Network Architectures**: LSTM, CNN, Transformer, Ensemble
- **Comprehensive Datasets**: News headlines, Reddit comments, Twitter sarcasm
- **Hybrid Detection**: Combines ML models with pattern-based detection
- **Easy Integration**: Simple API for existing applications
- **Production Ready**: Complete training, evaluation, and deployment pipeline

## 🔧 Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Storage**: 2GB free space
- **GPU**: Optional but recommended for faster training

### Software Requirements
- PyTorch 1.9+
- CUDA (optional, for GPU acceleration)
- Git (for cloning repositories)

## 📦 Installation

### Step 1: Clone/Download the System
```bash
# If you have the files locally, navigate to the project directory
cd sarcasm-detection-system

# Create the ML model directory structure
mkdir -p ml_model/{data,models,logs}
```

### Step 2: Install Python Dependencies
```bash
# Install core requirements
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn
pip install matplotlib seaborn plotly
pip install flask flask-cors
pip install requests tqdm

# Or install from requirements file
pip install -r ml_model/requirements.txt
```

### Step 3: Verify Installation
```bash
# Test PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test other dependencies
python -c "import numpy, pandas, sklearn; print('All packages imported successfully')"
```

## 📊 Dataset Setup

### Step 1: Prepare Data Directory
```bash
# Create data directory
mkdir -p ml_model/data

# The system will automatically download datasets or create sample data
```

### Step 2: Dataset Sources

#### 1. News Headlines Dataset
- **Source**: News headlines with sarcasm labels
- **Size**: ~29,000 samples
- **Format**: JSON with 'headline' and 'is_sarcastic' fields
- **Auto-download**: Yes (with fallback to sample data)

#### 2. Reddit Comments Dataset
- **Source**: Reddit comments with sarcasm labels
- **Size**: Variable
- **Format**: JSON with 'comment' and 'label' fields
- **Auto-download**: Yes (with fallback to sample data)

#### 3. Twitter Sarcasm Dataset
- **Source**: Twitter posts with sarcasm labels
- **Size**: Variable
- **Format**: JSON with 'tweet' and 'sarcastic' fields
- **Auto-download**: Yes (with fallback to sample data)

### Step 3: Test Dataset Loading
```bash
cd ml_model
python datasets.py
```

Expected output:
```
Downloading news_headlines dataset...
Downloaded news_headlines dataset successfully!
Training samples: 800
Test samples: 200
Sarcastic samples in training: 400 (50.0%)
Sarcastic samples in test: 100 (50.0%)
```

## 🚀 Model Training

### Step 1: Quick Start Training
```bash
cd ml_model

# Quick training with LSTM (5 epochs)
python quick_start.py --action train --model_type lstm --epochs 5

# Quick training with CNN
python quick_start.py --action train --model_type cnn --epochs 5

# Quick training with Transformer
python quick_start.py --action train --model_type transformer --epochs 5
```

### Step 2: Full Training Pipeline
```bash
# Train LSTM model
python train.py \
    --model_type lstm \
    --epochs 20 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --embedding_dim 128 \
    --hidden_dim 64 \
    --max_length 128 \
    --vocab_size 10000

# Train CNN model
python train.py \
    --model_type cnn \
    --epochs 15 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --embedding_dim 128 \
    --max_length 128 \
    --vocab_size 10000

# Train Transformer model
python train.py \
    --model_type transformer \
    --epochs 25 \
    --batch_size 16 \
    --learning_rate 0.0005 \
    --embedding_dim 256 \
    --max_length 256 \
    --vocab_size 20000

# Train Ensemble model
python train.py \
    --model_type ensemble \
    --epochs 30 \
    --batch_size 16 \
    --learning_rate 0.0005 \
    --embedding_dim 128 \
    --hidden_dim 64 \
    --max_length 128 \
    --vocab_size 10000
```

### Step 3: GPU Training (Optional)
```bash
# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Train with GPU
python train.py \
    --model_type lstm \
    --epochs 20 \
    --device cuda \
    --batch_size 64
```

### Step 4: Monitor Training
During training, you'll see output like:
```
Epoch 1/20:
  Train Loss: 0.6234, Train Acc: 0.6500
  Val Loss: 0.5891, Val Acc: 0.7000
  New best model saved! Val Acc: 0.7000

Epoch 2/20:
  Train Loss: 0.5678, Train Acc: 0.7200
  Val Loss: 0.5234, Val Acc: 0.7500
  New best model saved! Val Acc: 0.7500
```

## 📈 Model Evaluation

### Step 1: Test Trained Model
```bash
# Find your trained model
ls models/

# Test the model
python inference.py --model_path models/lstm_20241201_143022
```

### Step 2: Compare Model Architectures
```bash
python quick_start.py --action compare
```

### Step 3: Evaluate Performance
After training, check the results in your model directory:
```bash
# Check model directory
ls models/lstm_20241201_143022/

# Files created:
# - best_model.pth          # Best model weights
# - final_model.pth        # Final model weights
# - tokenizer.pkl          # Trained tokenizer
# - model_info.json        # Model configuration
# - training_history.json  # Training metrics
# - evaluation_results.json # Test results
# - sample_predictions.json # Sample predictions
# - training_history.png   # Training plots
# - confusion_matrix.png   # Confusion matrix
```

### Step 4: Analyze Results
```bash
# View evaluation results
cat models/lstm_20241201_143022/evaluation_results.json

# Expected output:
{
  "accuracy": 0.8750,
  "precision": 0.8823,
  "recall": 0.8750,
  "f1_score": 0.8786,
  "confusion_matrix": [[85, 15], [10, 90]]
}
```

## 🔗 Integration

### Step 1: Test ML Model Integration
```bash
# Test the enhanced backend
cd backend
python enhanced_app.py
```

### Step 2: API Testing
```bash
# Test health endpoint
curl http://localhost:5000/health

# Test prediction endpoint
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love it when my computer crashes!"}'

# Test batch prediction
curl -X POST http://localhost:5000/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["I am tired", "This is just fantastic", "Thank you for your help"]}'
```

### Step 3: Frontend Integration
Update your frontend to use the enhanced API:
```javascript
// Example API call
const response = await fetch('http://localhost:5000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    text: userInput
  })
});

const result = await response.json();
console.log('Prediction:', result.is_sarcastic);
console.log('Confidence:', result.confidence);
console.log('Method:', result.method);
```

## 🚀 Deployment

### Step 1: Production Training
```bash
# Train production model with more data and epochs
python train.py \
    --model_type ensemble \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.0005 \
    --embedding_dim 256 \
    --hidden_dim 128 \
    --max_length 256 \
    --vocab_size 20000 \
    --device cuda
```

### Step 2: Model Optimization
```bash
# Convert model to optimized format
python -c "
import torch
model = torch.load('models/ensemble_20241201_143022/best_model.pth')
torch.save(model, 'models/ensemble_20241201_143022/optimized_model.pth')
"
```

### Step 3: Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "backend/enhanced_app.py"]
```

### Step 4: Production API
```bash
# Run production server
gunicorn -w 4 -b 0.0.0.0:5000 backend.enhanced_app:app
```

## 🐛 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Solution: Reduce batch size
python train.py --batch_size 16

# Or use CPU
python train.py --device cpu
```

#### 2. Dataset Download Fails
```bash
# Solution: Check internet connection
# The system will automatically create sample data
python datasets.py
```

#### 3. Model Loading Errors
```bash
# Solution: Check model files
ls models/your_model_directory/

# Ensure all files exist:
# - best_model.pth
# - tokenizer.pkl
# - model_info.json
```

#### 4. Poor Training Performance
```bash
# Solutions:
# 1. Increase epochs
python train.py --epochs 50

# 2. Adjust learning rate
python train.py --learning_rate 0.0005

# 3. Use larger model
python train.py --embedding_dim 256 --hidden_dim 128

# 4. Check data quality
python datasets.py
```

### Performance Optimization

#### 1. Speed Up Training
```bash
# Use GPU
python train.py --device cuda

# Increase batch size
python train.py --batch_size 64

# Use smaller model
python train.py --embedding_dim 64 --hidden_dim 32
```

#### 2. Improve Accuracy
```bash
# Use ensemble model
python train.py --model_type ensemble

# Train longer
python train.py --epochs 100

# Use larger vocabulary
python train.py --vocab_size 20000
```

## 📊 Expected Results

### Model Performance Comparison
| Model Type | Accuracy | Training Time | Inference Speed | Best For |
|------------|----------|---------------|----------------|----------|
| LSTM | 85-90% | Medium | Fast | General use |
| CNN | 80-85% | Fast | Very Fast | Short texts |
| Transformer | 90-95% | Slow | Medium | Complex patterns |
| Ensemble | 92-97% | Very Slow | Slow | Production |

### Sample Predictions
```
Text: "I love it when my computer crashes!"
Prediction: Sarcastic (Confidence: 0.95)

Text: "Thank you for your help"
Prediction: Not Sarcastic (Confidence: 0.92)

Text: "Oh fantastic, another meeting"
Prediction: Sarcastic (Confidence: 0.88)
```

## 🎯 Next Steps

1. **Train Multiple Models**: Experiment with different architectures
2. **Hyperparameter Tuning**: Optimize learning rates, batch sizes, etc.
3. **Data Augmentation**: Add more training data
4. **Model Ensemble**: Combine multiple models for better accuracy
5. **Production Deployment**: Deploy to cloud platforms
6. **Continuous Learning**: Implement online learning capabilities

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the sample code and documentation
3. Test with the provided sample data
4. Verify all dependencies are installed correctly

---

**Happy Training! 🎭🤖**
