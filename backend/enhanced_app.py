"""
Enhanced Backend with ML Integration
Integrates the trained ML model with the existing Flask application
"""

import os
import sys
import json
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
from datetime import datetime

# Add ML model path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ml_model'))

# Import existing functions
from app import (
    context_aware_sarcasm_detection, 
    complete_sarcastic_conversion, 
    complete_genuine_conversion,
    COMPLETE_SARCASTIC_PATTERNS,
    COMPLETE_CONVERSION_SYSTEM
)

# Import ML components
try:
    from inference import SarcasmPredictor, integrate_with_flask_app
    ML_AVAILABLE = True
except ImportError as e:
    print(f"ML components not available: {e}")
    ML_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Global ML predictor
ml_predictor = None
ml_functions = None

def load_ml_model(model_path: str = None):
    """Load the ML model for sarcasm detection"""
    global ml_predictor, ml_functions
    
    if not ML_AVAILABLE:
        print("ML components not available, using pattern-based detection only")
        return False
    
    try:
        # Find the best model if no path specified
        if model_path is None:
            models_dir = os.path.join(os.path.dirname(__file__), 'ml_model', 'models')
            if os.path.exists(models_dir):
                model_dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
                if model_dirs:
                    # Get the most recent model
                    latest_model = max(model_dirs, key=lambda x: os.path.getmtime(os.path.join(models_dir, x)))
                    model_path = os.path.join(models_dir, latest_model)
                    print(f"Auto-selected model: {model_path}")
                else:
                    print("No trained models found, using pattern-based detection only")
                    return False
            else:
                print("Models directory not found, using pattern-based detection only")
                return False
        
        # Load the ML model
        ml_predictor = SarcasmPredictor(model_path)
        ml_functions = integrate_with_flask_app(model_path)
        
        print(f"✅ ML model loaded successfully from {model_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading ML model: {e}")
        print("Falling back to pattern-based detection")
        return False

def hybrid_sarcasm_detection(text: str) -> dict:
    """Hybrid sarcasm detection combining ML and pattern-based methods"""
    
    # Get pattern-based detection
    pattern_result = context_aware_sarcasm_detection(text)
    
    # Get ML-based detection if available
    ml_result = None
    if ml_predictor is not None:
        try:
            ml_result = ml_predictor.predict(text)
            ml_result = {
                'is_sarcastic': ml_result['prediction'] == 'Sarcastic',
                'confidence': ml_result['confidence'],
                'method': 'ml_model',
                'probabilities': ml_result['probabilities']
            }
        except Exception as e:
            print(f"ML prediction error: {e}")
            ml_result = None
    
    # Combine results
    if ml_result is not None:
        # Use ML result as primary, pattern as fallback
        if ml_result['confidence'] > 0.7:  # High confidence ML prediction
            return ml_result
        elif pattern_result['confidence'] > 0.8:  # High confidence pattern prediction
            return pattern_result
        else:
            # Combine both methods
            combined_confidence = (ml_result['confidence'] + pattern_result['confidence']) / 2
            combined_sarcastic = ml_result['is_sarcastic'] or pattern_result['is_sarcastic']
            
            return {
                'is_sarcastic': combined_sarcastic,
                'confidence': combined_confidence,
                'method': 'hybrid',
                'ml_result': ml_result,
                'pattern_result': pattern_result
            }
    else:
        # Fallback to pattern-based only
        return pattern_result

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'ml_model_loaded': ml_predictor is not None,
        'pattern_patterns_count': len(COMPLETE_SARCASTIC_PATTERNS),
        'conversion_mappings_count': len(COMPLETE_CONVERSION_SYSTEM['direct_mappings']['to_sarcastic'])
    })

@app.route('/predict', methods=['POST'])
def predict_sarcasm():
    """Enhanced sarcasm detection endpoint"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({
                'error': 'Text is required',
                'is_sarcastic': False,
                'confidence': 0.0,
                'method': 'error'
            }), 400
        
        # Use hybrid detection
        result = hybrid_sarcasm_detection(text)
        
        return jsonify({
            'text': text,
            'is_sarcastic': result['is_sarcastic'],
            'confidence': result['confidence'],
            'method': result['method'],
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'is_sarcastic': False,
            'confidence': 0.0,
            'method': 'error'
        }), 500

@app.route('/convert', methods=['POST'])
def convert_text():
    """Enhanced text conversion endpoint"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        to_sarcastic = data.get('to_sarcastic', True)
        
        if not text:
            return jsonify({
                'error': 'Text is required',
                'converted_text': '',
                'method': 'error'
            }), 400
        
        # Use existing conversion functions
        if to_sarcastic:
            converted_text = complete_sarcastic_conversion(text)
        else:
            converted_text = complete_genuine_conversion(text)
        
        return jsonify({
            'original_text': text,
            'converted_text': converted_text,
            'to_sarcastic': to_sarcastic,
            'method': 'pattern_based',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'converted_text': '',
            'method': 'error'
        }), 500

@app.route('/batch', methods=['POST'])
def batch_predict():
    """Batch sarcasm detection endpoint"""
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts or not isinstance(texts, list):
            return jsonify({
                'error': 'Texts array is required',
                'results': []
            }), 400
        
        results = []
        for text in texts:
            if isinstance(text, str) and text.strip():
                result = hybrid_sarcasm_detection(text.strip())
                results.append({
                    'text': text.strip(),
                    'is_sarcastic': result['is_sarcastic'],
                    'confidence': result['confidence'],
                    'method': result['method']
                })
            else:
                results.append({
                    'text': text,
                    'is_sarcastic': False,
                    'confidence': 0.0,
                    'method': 'error'
                })
        
        return jsonify({
            'results': results,
            'count': len(results),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'results': []
        }), 500

@app.route('/test', methods=['GET'])
def test_endpoints():
    """Test endpoint with comprehensive examples"""
    
    test_cases = [
        # Sarcastic examples
        {"text": "I'm just bursting with energy", "expected": True},
        {"text": "This is just fantastic", "expected": True},
        {"text": "Oh fantastic, another meeting", "expected": True},
        {"text": "I love it when my computer crashes", "expected": True},
        {"text": "What a wonderful way to start my day", "expected": True},
        
        # Genuine examples
        {"text": "I am tired", "expected": False},
        {"text": "This is a problem", "expected": False},
        {"text": "Thank you for your help", "expected": False},
        {"text": "The weather is nice today", "expected": False},
        {"text": "I appreciate your effort", "expected": False},
        
        # Edge cases
        {"text": "Wow, you're such a genius", "expected": True},
        {"text": "You are very smart", "expected": False},
        {"text": "Perfect timing for this update", "expected": True},
        {"text": "This update is helpful", "expected": False},
    ]
    
    results = []
    correct_predictions = 0
    
    for i, test_case in enumerate(test_cases):
        try:
            result = hybrid_sarcasm_detection(test_case['text'])
            is_correct = result['is_sarcastic'] == test_case['expected']
            if is_correct:
                correct_predictions += 1
            
            results.append({
                'test_case': i + 1,
                'text': test_case['text'],
                'expected': test_case['expected'],
                'predicted': result['is_sarcastic'],
                'confidence': result['confidence'],
                'method': result['method'],
                'correct': is_correct
            })
        except Exception as e:
            results.append({
                'test_case': i + 1,
                'text': test_case['text'],
                'expected': test_case['expected'],
                'predicted': False,
                'confidence': 0.0,
                'method': 'error',
                'correct': False,
                'error': str(e)
            })
    
    accuracy = correct_predictions / len(test_cases) if test_cases else 0
    
    return jsonify({
        'test_results': results,
        'total_tests': len(test_cases),
        'correct_predictions': correct_predictions,
        'accuracy': accuracy,
        'ml_model_loaded': ml_predictor is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/model_info', methods=['GET'])
def get_model_info():
    """Get information about the loaded model"""
    try:
        info = {
            'ml_model_loaded': ml_predictor is not None,
            'pattern_patterns_count': len(COMPLETE_SARCASTIC_PATTERNS),
            'conversion_mappings_count': len(COMPLETE_CONVERSION_SYSTEM['direct_mappings']['to_sarcastic']),
            'timestamp': datetime.now().isoformat()
        }
        
        if ml_predictor is not None:
            try:
                ml_info = ml_predictor.get_model_info()
                info['ml_model_info'] = ml_info
            except Exception as e:
                info['ml_model_error'] = str(e)
        
        return jsonify(info)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'ml_model_loaded': False
        }), 500

@app.route('/reload_model', methods=['POST'])
def reload_model():
    """Reload the ML model"""
    try:
        data = request.get_json()
        model_path = data.get('model_path') if data else None
        
        success = load_ml_model(model_path)
        
        return jsonify({
            'success': success,
            'message': 'Model reloaded successfully' if success else 'Failed to reload model',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    print("🚀 Starting Enhanced Sarcasm Detection API...")
    print("=" * 50)
    
    # Try to load ML model
    print("Loading ML model...")
    ml_loaded = load_ml_model()
    
    if ml_loaded:
        print("✅ ML model loaded successfully!")
        print("🔧 Using hybrid detection (ML + Pattern-based)")
    else:
        print("⚠️  ML model not available, using pattern-based detection only")
    
    print("🌐 Starting Flask server...")
    print("📡 API endpoints available:")
    print("  GET  /health - Health check")
    print("  POST /predict - Sarcasm detection")
    print("  POST /convert - Text conversion")
    print("  POST /batch - Batch detection")
    print("  GET  /test - Test with examples")
    print("  GET  /model_info - Model information")
    print("  POST /reload_model - Reload ML model")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
