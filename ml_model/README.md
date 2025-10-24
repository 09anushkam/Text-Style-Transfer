"""
README for ML-based Sarcasm Detection System
Complete guide for training and using machine learning models for sarcasm detection
"""

# ML-based Sarcasm Detection System

A comprehensive machine learning system for sarcasm detection using multiple neural network architectures including LSTM, CNN, Transformer, and Ensemble models.

## 🚀 Features

- **Multiple Model Architectures**: LSTM, CNN, Transformer, and Ensemble models
- **Comprehensive Datasets**: News headlines, Reddit comments, Twitter sarcasm
- **Advanced Training Pipeline**: Complete training, validation, and evaluation
- **Easy Integration**: Simple API for integration with existing applications
- **Performance Metrics**: Detailed evaluation with confusion matrices and plots
- **Sample Data**: Built-in sample datasets for testing

## 📊 Model Architectures

### 1. LSTM Model
- **Description**: LSTM with attention mechanism
- **Pros**: Good at capturing sequential patterns, handles variable length sequences
- **Cons**: Can be slow for long sequences, may struggle with very long dependencies
- **Best for**: General sarcasm detection, text with clear sequential patterns

### 2. CNN Model
- **Description**: Convolutional Neural Network
- **Pros**: Fast training and inference, good at capturing local patterns
- **Cons**: Limited ability to capture long-range dependencies
- **Best for**: Short texts, pattern-based sarcasm detection

### 3. Transformer Model
- **Description**: Transformer encoder with self-attention
- **Pros**: Excellent at capturing long-range dependencies, parallelizable
- **Cons**: Requires more data, computationally intensive
- **Best for**: Complex sarcasm patterns, long texts

### 4. Ensemble Model
- **Description**: Combination of LSTM, CNN, and Transformer
- **Pros**: Best overall performance, robust to different text types
- **Cons**: Larger model size, slower inference
- **Best for**: Production systems requiring high accuracy

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- CUDA (optional, for GPU acceleration)

### Setup
```bash
# Clone or download the ML model files
cd ml_model

# Install requirements
pip install -r requirements.txt

# Create data directory
mkdir data
mkdir models
```

## 📚 Datasets

The system supports multiple sarcasm detection datasets:

### 1. News Headlines Dataset
- **Source**: News headlines with sarcasm labels
- **Size**: ~29,000 samples
- **Format**: JSON with 'headline' and 'is_sarcastic' fields

### 2. Reddit Comments Dataset
- **Source**: Reddit comments with sarcasm labels
- **Size**: Variable
- **Format**: JSON with 'comment' and 'label' fields

### 3. Twitter Sarcasm Dataset
- **Source**: Twitter posts with sarcasm labels
- **Size**: Variable
- **Format**: JSON with 'tweet' and 'sarcastic' fields

### Sample Data
If dataset downloads fail, the system automatically creates sample data for testing.

## 🚀 Quick Start

### 1. Basic Training
```bash
# Train an LSTM model
python train.py --model_type lstm --epochs 10

# Train a CNN model
python train.py --model_type cnn --epochs 15

# Train a Transformer model
python train.py --model_type transformer --epochs 20

# Train an Ensemble model
python train.py --model_type ensemble --epochs 25
```

### 2. Advanced Training
```bash
# Custom parameters
python train.py \
    --model_type lstm \
    --epochs 30 \
    --batch_size 64 \
    --learning_rate 0.0005 \
    --embedding_dim 256 \
    --hidden_dim 128 \
    --max_length 256 \
    --vocab_size 20000 \
    --device cuda
```

### 3. Model Testing
```bash
# Test a trained model
python inference.py --model_path models/lstm_20241201_143022
```

## 📖 Detailed Usage

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_type` | lstm | Model architecture (lstm, cnn, transformer, ensemble) |
| `--epochs` | 20 | Number of training epochs |
| `--batch_size` | 32 | Batch size for training |
| `--learning_rate` | 0.001 | Learning rate for optimizer |
| `--embedding_dim` | 128 | Embedding dimension |
| `--hidden_dim` | 64 | Hidden dimension for LSTM |
| `--max_length` | 128 | Maximum sequence length |
| `--vocab_size` | 10000 | Vocabulary size |
| `--device` | auto | Device to use (cpu, cuda, auto) |
| `--save_dir` | models | Directory to save models |
| `--data_dir` | data | Directory containing datasets |

### Model Outputs

After training, each model creates a directory with:
- `best_model.pth`: Best model weights
- `final_model.pth`: Final model weights
- `tokenizer.pkl`: Trained tokenizer
- `model_info.json`: Model configuration
- `training_history.json`: Training metrics
- `evaluation_results.json`: Test results
- `sample_predictions.json`: Sample predictions
- `training_history.png`: Training plots
- `confusion_matrix.png`: Confusion matrix

## 🔧 Integration

### Python API
```python
from inference import SarcasmPredictor

# Load trained model
predictor = SarcasmPredictor('models/lstm_20241201_143022')

# Make predictions
result = predictor.predict("I love it when my computer crashes!")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.3f}")
```

### Flask Integration
```python
from inference import integrate_with_flask_app

# Integrate with existing Flask app
ml_functions = integrate_with_flask_app('models/lstm_20241201_143022')

# Use ML detection
result = ml_functions['detect_sarcasm']("Oh fantastic, another meeting!")
```

## 📊 Performance Evaluation

### Metrics
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

### Sample Results
```
Model Type: LSTM
Training Samples: 800
Test Samples: 200
Parameters: 1,234,567

Final Results:
Accuracy: 0.8750
Precision: 0.8823
Recall: 0.8750
F1-Score: 0.8786
```

## 🎯 Best Practices

### 1. Model Selection
- **Start with LSTM**: Good balance of performance and speed
- **Use CNN for speed**: When inference speed is critical
- **Use Transformer for accuracy**: When you have large datasets
- **Use Ensemble for production**: When you need maximum accuracy

### 2. Training Tips
- **Start small**: Begin with fewer epochs and smaller models
- **Monitor overfitting**: Watch validation loss vs training loss
- **Use early stopping**: Stop training when validation accuracy plateaus
- **Experiment with hyperparameters**: Try different learning rates and batch sizes

### 3. Data Preparation
- **Clean your data**: Remove URLs, mentions, hashtags
- **Balance your dataset**: Ensure roughly equal sarcastic/non-sarcastic samples
- **Preprocess consistently**: Use the same preprocessing for training and inference

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python train.py --batch_size 16
   
   # Use CPU
   python train.py --device cpu
   ```

2. **Dataset Download Fails**
   - The system automatically creates sample data
   - Check your internet connection
   - Verify the data directory exists

3. **Model Loading Errors**
   - Ensure all files are in the model directory
   - Check that the model was trained successfully
   - Verify PyTorch version compatibility

### Performance Issues

1. **Slow Training**
   - Reduce model size (embedding_dim, hidden_dim)
   - Use smaller vocabulary
   - Reduce sequence length

2. **Poor Accuracy**
   - Increase training epochs
   - Use larger model
   - Check data quality
   - Try different model architecture

## 📈 Advanced Usage

### Custom Datasets
```python
from datasets import SarcasmDatasetLoader

# Create custom dataset
loader = SarcasmDatasetLoader('custom_data')
# Add your custom data loading logic
```

### Model Comparison
```python
from models import ModelFactory

# Compare different models
models = ['lstm', 'cnn', 'transformer', 'ensemble']
for model_type in models:
    model = ModelFactory.create_model(model_type, vocab_size=10000)
    print(f"{model_type}: {sum(p.numel() for p in model.parameters())} parameters")
```

### Hyperparameter Tuning
```bash
# Grid search example
for lr in 0.001 0.0005 0.0001; do
    for batch_size in 16 32 64; do
        python train.py --learning_rate $lr --batch_size $batch_size
    done
done
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- Hugging Face for transformer models and datasets
- The sarcasm detection research community
- Contributors to the various datasets used

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the sample code and documentation

---

**Happy Sarcasm Detection! 🎭**
