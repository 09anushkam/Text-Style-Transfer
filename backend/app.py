# app.py - FIXED VERSION
import os
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
from transformers import pipeline
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import joblib
from typing import Dict, List, Tuple
import random

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except:
    nltk.download('vader_lexicon')

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize models
models = {}
sia = SentimentIntensityAnalyzer()

# CORRECT sarcasm patterns
SARCASTIC_PATTERNS = [
    (r'\bjust what i needed\b', 0.98),
    (r'\boh\s+(great|wonderful|fantastic|perfect|lovely)\b', 0.96),
    (r'\bi love (waiting|standing|sitting|dealing with)\b', 0.95),
    (r'\byeah right\b', 0.94),
    (r'\bas if\b', 0.93),
    (r'\bexactly what i wanted\b', 0.92),
    (r'\bwhat a (day|pleasure|joy|treat)\b', 0.90),
    (r'\bhow (wonderful|marvelous|convenient)\b', 0.89),
    (r'\bcouldn\'t be (better|happier)\b', 0.88),
    (r'\bthanks a (bunch|lot)\b', 0.87),
    (r'\bno problem at all\b', 0.86),
    (r'\bof course\b', 0.85),
    (r'\bsure thing\b', 0.84),
    (r'\bthat\'s just (great|wonderful)\b', 0.83),
    (r'\b(so|really) (happy|excited|thrilled)\b', 0.82),
]

# IMPROVED conversion system with better mappings
CONVERSION_SYSTEM = {
    'phrase_conversions': {
        'to_sarcastic': {
            'i am tired': 'I\'m just bursting with energy',
            'i\'m tired': 'I\'m feeling absolutely refreshed',
            'i am exhausted': 'I\'m full of vitality',
            'i\'m exhausted': 'I couldn\'t be more energized',
            'this is a problem': 'This is just fantastic',
            'there is an issue': 'This is wonderful news',
            'this is broken': 'This is working perfectly',
            'this doesn\'t work': 'This is functioning beautifully',
            'i have to wait': 'I get to enjoy some quality waiting time',
            'this is taking too long': 'This is moving at lightning speed',
            'we are delayed': 'We\'re making excellent time',
            'i\'m waiting': 'I\'m having the time of my life waiting',
            'i have more work': 'Just what I needed, more work',
            'this is difficult': 'This is a piece of cake',
            'this is hard': 'This is incredibly easy',
            'i don\'t understand': 'This makes perfect sense',
            'the weather is bad': 'What beautiful weather',
            'it\'s raining': 'Perfect weather for a picnic',
            'it\'s too hot': 'Such comfortable temperatures',
            'it\'s too cold': 'Refreshingly cool weather',
            'this is terrible': 'This is wonderful',
            'this is awful': 'This is fantastic',
            'i hate this': 'I absolutely love this',
            'this is boring': 'This is so exciting',
            'i don\'t like this': 'This is my favorite thing',
            'thank you': 'Thanks a bunch',
            'thanks': 'Thanks a lot',
            'no problem': 'No problem at all',
            'okay': 'Oh wonderful',
            'yes': 'Yeah right',
            'sure': 'Of course',
            'i understand': 'As if I understand',
            'i dislike waiting': 'I love waiting',
            'the day is bad': 'What a day',
            'this is unpleasant': 'What a pleasure',
        },
        'to_unsarcastic': {
            'i\'m just bursting with energy': 'I am tired',
            'i\'m feeling absolutely refreshed': 'I\'m exhausted',
            'i\'m full of vitality': 'I am tired',
            'i couldn\'t be more energized': 'I\'m exhausted',
            'this is just fantastic': 'This is a problem',
            'this is wonderful news': 'There is an issue',
            'this is working perfectly': 'This is broken',
            'this is functioning beautifully': 'This doesn\'t work',
            'i get to enjoy some quality waiting time': 'I have to wait',
            'this is moving at lightning speed': 'This is taking too long',
            'we\'re making excellent time': 'We are delayed',
            'i\'m having the time of my life waiting': 'I\'m waiting',
            'just what i needed, more work': 'I have more work',
            'this is a piece of cake': 'This is difficult',
            'this is incredibly easy': 'This is hard',
            'this makes perfect sense': 'I don\'t understand',
            'what beautiful weather': 'The weather is bad',
            'perfect weather for a picnic': 'It\'s raining',
            'such comfortable temperatures': 'It\'s too hot',
            'refreshingly cool weather': 'It\'s too cold',
            'this is wonderful': 'This is terrible',
            'this is fantastic': 'This is awful',
            'i absolutely love this': 'I hate this',
            'this is so exciting': 'This is boring',
            'this is my favorite thing': 'I don\'t like this',
            'thanks a bunch': 'Thank you',
            'thanks a lot': 'Thanks',
            'no problem at all': 'No problem',
            'oh wonderful': 'Okay',
            'yeah right': 'Yes',
            'of course': 'Sure',
            'as if i understand': 'I understand',
            'just what i needed': 'What I needed',
            'oh great': 'Okay',
            'how wonderful': 'Okay',
            'what a day': 'The day is bad',
            'what a pleasure': 'This is unpleasant',
            'i love waiting': 'I dislike waiting',
        }
    },
    'sarcastic_templates': [
        "Oh great, {}",
        "Just what I needed, {}",
        "Wonderful, {}",
        "Perfect, {}",
        "Yeah right, {}",
        "As if {}",
        "What a {}",
        "How {}",
        "I love {}",
        "This is so {}",
    ]
}

def load_models():
    global models
    try:
        print("Loading sentiment analysis model...")
        
        models['sentiment'] = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=-1,
            truncation=True,
            max_length=512
        )
        print("✓ Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()

def detect_sarcasm_correct(text: str) -> Dict:
    """Correct sarcasm detection without errors"""
    text_lower = text.lower().strip()
    
    if not text_lower:
        return {'is_sarcastic': False, 'confidence': 0.0, 'method': 'empty'}
    
    scores = {
        'pattern': 0.0,
        'sentiment_contrast': 0.0,
        'linguistic': 0.0,
        'contextual': 0.0
    }
    
    # 1. PATTERN MATCHING
    for pattern, confidence in SARCASTIC_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            scores['pattern'] = max(scores['pattern'], confidence)
    
    # 2. SENTIMENT CONTRAST - FIXED VERSION
    try:
        sentiment_result = models['sentiment'](text)[0]
        sentiment_label = sentiment_result['label'].lower()
        sentiment_confidence = sentiment_result['score']
        
        vader_scores = sia.polarity_scores(text)
        vader_compound = vader_scores['compound']
        
        # Define word lists for context analysis
        positive_words = ['great', 'wonderful', 'fantastic', 'perfect', 'love', 'excited', 'happy', 'thanks', 'good', 'amazing']
        negative_context_words = ['waiting', 'problem', 'issue', 'error', 'late', 'broken', 'terrible', 'awful', 'hate', 'tired']
        
        has_positive_word = any(word in text_lower for word in positive_words)
        has_negative_context = any(word in text_lower for word in negative_context_words)
        has_negative_sentiment = vader_compound < -0.1
        
        if sentiment_label == 'positive':
            if has_negative_sentiment and has_negative_context:
                # Strong sarcasm: positive words + negative context + negative sentiment
                scores['sentiment_contrast'] = 0.9 * sentiment_confidence
            elif has_negative_sentiment:
                # Moderate sarcasm: positive words + negative sentiment
                scores['sentiment_contrast'] = 0.7 * sentiment_confidence
            elif has_negative_context:
                # Weak sarcasm: positive words + negative context words
                scores['sentiment_contrast'] = 0.6 * sentiment_confidence
                
    except Exception as e:
        print(f"Sentiment analysis error: {e}")
    
    # 3. LINGUISTIC FEATURES
    if text.count('!') > 1 or ('!' in text and '?' in text):
        scores['linguistic'] += 0.4
    
    if re.search(r'\b[A-Z]{2,}\b', text):
        scores['linguistic'] += 0.3
    
    if '...' in text or '..' in text:
        scores['linguistic'] += 0.2
    
    scores['linguistic'] = min(1.0, scores['linguistic'])
    
    # 4. CONTEXTUAL ANALYSIS
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_context_words if word in text_lower)
    
    if positive_count > 0 and negative_count > 0:
        scores['contextual'] = min(0.8, (positive_count + negative_count) * 0.15)
    
    # Calculate weighted final score
    weights = {
        'pattern': 0.40,
        'sentiment_contrast': 0.35,
        'linguistic': 0.15,
        'contextual': 0.10
    }
    
    final_score = sum(scores[feature] * weights[feature] for feature in scores)
    
    # BOOSTING: Clear sarcastic phrases
    clear_sarcastic_phrases = [
        'just what i needed', 'oh great', 'yeah right', 'as if',
        'i love waiting', 'thanks a bunch', 'what a day', 'how wonderful'
    ]
    
    if any(phrase in text_lower for phrase in clear_sarcastic_phrases):
        final_score = max(final_score, 0.85)
    
    final_score = min(1.0, final_score)
    
    is_sarcastic = final_score > 0.65
    
    return {
        'is_sarcastic': is_sarcastic,
        'confidence': round(final_score, 4),
        'detailed_scores': {k: round(v, 3) for k, v in scores.items()},
        'method': 'correct_detection'
    }

def convert_to_sarcastic_correct(text: str) -> str:
    """Convert genuine text to PROPER sarcastic version"""
    original_text = text.strip()
    if not original_text:
        return "What a joy"
    
    text_lower = original_text.lower()
    converted = original_text
    
    print(f"Converting to sarcastic: '{original_text}'")
    
    # FIRST: Check for exact phrase matches
    for genuine, sarcastic in CONVERSION_SYSTEM['phrase_conversions']['to_sarcastic'].items():
        if genuine in text_lower:
            # Handle case preservation
            if original_text[0].isupper():
                converted = sarcastic[0].upper() + sarcastic[1:]
            else:
                converted = sarcastic.lower()
            print(f"Phrase match: '{genuine}' -> '{sarcastic}'")
            return converted
    
    # SECOND: If no changes, apply sarcastic template
    if converted.lower() == original_text.lower():
        templates = CONVERSION_SYSTEM['sarcastic_templates']
        template = random.choice(templates)
        
        # Extract content for templating
        content = original_text.lower()
        content = re.sub(r'^(i\s+(am|feel|think)|this\s+is|that\'s|it\'s)\s*', '', content)
        content = content.strip()
        
        if content:
            converted = template.format(content)
        else:
            converted = "Just what I needed"
        
        print(f"Template applied: '{original_text}' -> '{converted}'")
    
    # Ensure proper capitalization
    if converted and converted[0].isalpha():
        converted = converted[0].upper() + converted[1:]
    
    print(f"Final conversion: '{original_text}' -> '{converted}'")
    return converted

def convert_to_unsarcastic_correct(text: str) -> str:
    """Convert sarcastic text to PROPER genuine version"""
    original_text = text.strip()
    if not original_text:
        return "This is acceptable"
    
    text_lower = original_text.lower()
    converted = original_text
    
    print(f"Converting to genuine: '{original_text}'")
    
    # FIRST: Check for exact sarcastic phrase matches
    for sarcastic, genuine in CONVERSION_SYSTEM['phrase_conversions']['to_unsarcastic'].items():
        if sarcastic in text_lower:
            # Handle case preservation
            if original_text[0].isupper():
                converted = genuine[0].upper() + genuine[1:]
            else:
                converted = genuine.lower()
            print(f"Phrase match: '{sarcastic}' -> '{genuine}'")
            return converted
    
    # SECOND: Remove sarcastic prefixes
    sarcastic_prefixes = [
        r'^(oh\s+)?(great|wonderful|fantastic|perfect|lovely)[,!\s]*',
        r'^just\s+what\s+i\s+(needed|wanted)[,!\s]*',
        r'^wonderful[,!\s]*',
        r'^perfect[,!\s]*',
        r'^yeah\s+right[,!\s]*',
        r'^as\s+if[,!\s]*',
        r'^what\s+a\s+',
        r'^how\s+',
    ]
    
    for prefix in sarcastic_prefixes:
        original_converted = converted
        converted = re.sub(prefix, '', converted, flags=re.IGNORECASE)
        if converted != original_converted:
            print(f"Removed prefix")
    
    # Clean up
    converted = re.sub(r'\s+', ' ', converted).strip()
    
    # If empty or no substantial change
    if not converted or len(converted) < 3 or converted.lower() == original_text.lower():
        neutral_responses = [
            "This is acceptable",
            "That's fine",
            "I understand",
        ]
        converted = random.choice(neutral_responses)
        print(f"Used neutral response: '{converted}'")
    
    # Ensure proper capitalization
    if converted and converted[0].isalpha():
        converted = converted[0].upper() + converted[1:]
    
    print(f"Final conversion: '{original_text}' -> '{converted}'")
    return converted

def smart_conversion(text: str, to_sarcastic: bool) -> str:
    """Main conversion router"""
    if not text.strip():
        return "What a joy" if to_sarcastic else "This is acceptable"
    
    if to_sarcastic:
        return convert_to_sarcastic_correct(text)
    else:
        return convert_to_unsarcastic_correct(text)

# Load models on startup
load_models()

@app.route("/health", methods=["GET"])
def health():
    models_loaded = len(models) > 0
    return jsonify({
        "status": "ok", 
        "models_loaded": models_loaded,
        "version": "correct_sarcasm_fixed"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        body = request.get_json()
        if not body or "text" not in body:
            return jsonify({"error": "No text provided"}), 400

        text = str(body["text"]).strip()
        if not text:
            return jsonify({"error": "Empty text"}), 400

        detection_result = detect_sarcasm_correct(text)
        
        converted_text = None
        if body.get("convert_to"):
            target_sarcastic = (body["convert_to"] == "sarcastic")
            converted_text = smart_conversion(text, target_sarcastic)

        response_data = {
            "text": text,
            "is_sarcastic": detection_result['is_sarcastic'],
            "confidence": detection_result['confidence'],
            "converted_text": converted_text,
            "detailed_scores": detection_result['detailed_scores'],
            "detection_method": detection_result['method']
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print("❌ Prediction error:", e)
        traceback.print_exc()
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

@app.route("/convert", methods=["POST"])
def convert():
    try:
        body = request.get_json()
        if not body or "text" not in body or "target" not in body:
            return jsonify({"error": "Missing text or target type"}), 400

        text = str(body["text"]).strip()
        target = body["target"]
        
        if target not in ["sarcastic", "unsarcastic"]:
            return jsonify({"error": "Target must be 'sarcastic' or 'unsarcastic'"}), 400

        # Check if text is already in the target form
        detection_result = detect_sarcasm_correct(text)
        is_currently_sarcastic = detection_result['is_sarcastic']
        
        if (target == "sarcastic" and is_currently_sarcastic) or (target == "unsarcastic" and not is_currently_sarcastic):
            return jsonify({
                "original_text": text,
                "converted_text": text,  # No change
                "conversion_type": target,
                "original_was_sarcastic": is_currently_sarcastic,
                "original_confidence": detection_result['confidence'],
                "conversion_happened": False,
                "message": f"Text is already {target}"
            })

        converted_text = smart_conversion(text, target == "sarcastic")
        
        return jsonify({
            "original_text": text,
            "converted_text": converted_text,
            "conversion_type": target,
            "original_was_sarcastic": is_currently_sarcastic,
            "original_confidence": detection_result['confidence'],
            "conversion_happened": True,
            "message": "Conversion successful"
        })
        
    except Exception as e:
        print("❌ Conversion error:", e)
        return jsonify({"error": "Conversion failed", "details": str(e)}), 500

@app.route("/test", methods=["GET"])
def test():
    """Test endpoint with correct conversions"""
    test_cases = [
        {"input": "I am tired", "expected_sarcastic": "I'm just bursting with energy"},
        {"input": "I'm exhausted", "expected_sarcastic": "I'm feeling absolutely refreshed"},
        {"input": "Thank you", "expected_sarcastic": "Thanks a bunch"},
        {"input": "This is a problem", "expected_sarcastic": "This is just fantastic"},
        {"input": "Just what I needed", "expected_unsarcastic": "What I needed"},
        {"input": "Thanks a bunch", "expected_unsarcastic": "Thank you"},
        {"input": "What a day", "expected_unsarcastic": "The day is bad"},
        {"input": "I love waiting", "expected_unsarcastic": "I dislike waiting"},
    ]
    
    results = []
    
    for case in test_cases:
        detection = detect_sarcasm_correct(case["input"])
        
        sarcastic_result = smart_conversion(case["input"], True)
        unsarcastic_result = smart_conversion(case["input"], False)
        
        results.append({
            "input": case["input"],
            "detected_sarcastic": detection['is_sarcastic'],
            "detection_confidence": detection['confidence'],
            "to_sarcastic": sarcastic_result,
            "to_unsarcastic": unsarcastic_result,
            "expected_sarcastic": case.get("expected_sarcastic", ""),
            "expected_unsarcastic": case.get("expected_unsarcastic", ""),
        })
    
    return jsonify({"test_results": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)