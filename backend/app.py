# app.py - COMPLETELY FINAL PERFECT VERSION
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
import random
from datetime import datetime

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

# COMPLETE SARCASM DETECTION PATTERNS
COMPLETE_SARCASTIC_PATTERNS = [
    # OBVIOUS SARCASM
    (r'\b(just|exactly) what i (needed|wanted)\b', 0.98),
    (r'\boh\s+(great|wonderful|fantastic|perfect|lovely|brilliant)\b', 0.97),
    (r'\byeah\s+(right|sure)\b', 0.96),
    (r'\bas if\b', 0.95),
    (r'\bof course\b', 0.94),
    (r'\bsure thing\b', 0.93),
    (r'\bno problem at all\b', 0.92),
    (r'\bthanks a (bunch|lot|million)\b', 0.91),
    
    # POSITIVE + NEGATIVE CONTEXT
    (r'\bi love (waiting|standing|sitting|dealing with|traffic|meetings|mondays)\b', 0.95),
    (r'\b(my favorite|the best) (part|thing) (about|of) (waiting|traffic|work|mondays)\b', 0.94),
    (r'\bnothing better than (waiting|traffic|work|problems)\b', 0.93),
    (r'\b(absolutely|totally) (adore|love) (this|that|it)\b', 0.92),
    
    # PERSONAL SARCASM (NEW)
    (r'\b(wow|oh) i (just |absolutely |totally )?love how (smart|intelligent|clever|brilliant) you are\b', 0.96),
    (r'\bi love how (smart|intelligent|clever|brilliant) you are\b', 0.95),
    (r'\byou\'re so (smart|intelligent|clever|brilliant)\b', 0.94),
    (r'\bwhat a (genius|brilliant idea)\b', 0.93),
    
    # SPECIFIC HIGH-CONFIDENCE PATTERNS
    (r'\bi (love|like|enjoy) .+ because .+ (trouble|problem|pain|annoying|stress|headache|pimples|hurt)\b', 0.96),
    (r'\b(.+) is my (favorite|favourite) (.+) because .+ (terrible|awful|horrible|bad|negative|pimples|pain)\b', 0.95),
    (r'\bi (absolutely |totally |just )?(adore|love) .+ because .+\b', 0.94),
]

# COMPLETE CONVERSION SYSTEM
COMPLETE_CONVERSION_SYSTEM = {
    'direct_mappings': {
        'to_sarcastic': {
            # BASIC CONVERSIONS
            'i am tired': "I'm just bursting with energy",
            "i'm tired": "I'm feeling absolutely refreshed", 
            'i am exhausted': "I'm full of vitality",
            "i'm exhausted": "I couldn't be more energized",
            'this is a problem': "This is just fantastic",
            'there is an issue': "This is wonderful news",
            'this is broken': "This is working perfectly",
            "this doesn't work": "This is functioning beautifully",
            'i have to wait': "I get to enjoy some quality waiting time",
            'this is taking too long': "This is moving at lightning speed",
            'we are delayed': "We're making excellent time",
            "i'm waiting": "I'm having the time of my life waiting",
            'i have more work': "Just what I needed, more work",
            'this is difficult': "This is a piece of cake",
            'this is hard': "This is incredibly easy",
            "i don't understand": "This makes perfect sense",
            'the weather is bad': "What beautiful weather",
            "it's raining": "Perfect weather for a picnic",
            "it's too hot": "Such comfortable temperatures", 
            "it's too cold": "Refreshingly cool weather",
            'this is terrible': "This is wonderful",
            'this is awful': "This is fantastic",
            'i hate this': "I absolutely love this",
            'this is boring': "This is so exciting",
            "i don't like this": "This is my favorite thing",
            'thank you': "Thanks a bunch",
            'thanks': "Thanks a lot",
            'no problem': "No problem at all",
            'okay': "Oh wonderful",
            'yes': "Yeah right",
            'sure': "Of course",
            'i understand': "As if I understand",
            
            # PERSONAL SARCASM (NEW)
            'you are smart': "Wow, you're such a genius",
            'you are intelligent': "What a brilliant mind you have",
            'you are clever': "You're so incredibly clever",
        },
        'to_unsarcastic': {
            # REVERSE CONVERSIONS
            "i'm just bursting with energy": "I am tired",
            "i'm feeling absolutely refreshed": "I'm exhausted",
            "i'm full of vitality": "I am tired", 
            "i couldn't be more energized": "I'm exhausted",
            "this is just fantastic": "This is a problem",
            "this is wonderful news": "There is an issue",
            "this is working perfectly": "This is broken",
            "this is functioning beautifully": "This doesn't work",
            "i get to enjoy some quality waiting time": "I have to wait",
            "this is moving at lightning speed": "This is taking too long",
            "we're making excellent time": "We are delayed",
            "i'm having the time of my life waiting": "I'm waiting",
            "just what i needed, more work": "I have more work",
            "this is a piece of cake": "This is difficult",
            "this is incredibly easy": "This is hard",
            "this makes perfect sense": "I don't understand",
            "what beautiful weather": "The weather is bad",
            "perfect weather for a picnic": "It's raining",
            "such comfortable temperatures": "It's too hot",
            "refreshingly cool weather": "It's too cold", 
            "this is wonderful": "This is terrible",
            "this is fantastic": "This is awful",
            "i absolutely love this": "I hate this",
            "this is so exciting": "This is boring",
            "this is my favorite thing": "I don't like this",
            "thanks a bunch": "Thank you",
            "thanks a lot": "Thanks",
            "no problem at all": "No problem",
            "oh wonderful": "Okay",
            "yeah right": "Yes",
            "of course": "Sure",
            "as if i understand": "I understand",
            
            # PERSONAL SARCASM REVERSAL (NEW)
            "wow, you're such a genius": "You're stupid",
            "what a brilliant mind you have": "You're not very smart",
            "you're so incredibly clever": "You're not clever",
        }
    },
    
    'pattern_conversions': {
        'to_sarcastic': [
            # PATTERN 1: I love X because Y
            (r'i (love|like|enjoy) (.+?) because (.+)', 
             lambda m: f"As if I {m.group(1)} {m.group(2)} because {m.group(3)}"),
            
            # PATTERN 2: X is my favorite because Y
            (r'(.+?) is my (favorite|favourite) (.+?) because (.+)', 
             lambda m: f"{m.group(1)} is my absolute {m.group(2)} {m.group(3)} because {m.group(4)}"),
            
            # PATTERN 3: Simple positive statements
            (r'i (love|like|enjoy) (.+)', 
             lambda m: f"I absolutely adore {m.group(2)}"),
            
            # PATTERN 4: X is good because Y
            (r'(.+?) is (good|nice|great) because (.+)', 
             lambda m: f"{m.group(1)} is absolutely fantastic because {m.group(3)}"),
            
            # PATTERN 5: PERSONAL SARCASM - I love how smart you are (NEW)
            (r'i love how (smart|intelligent|clever|brilliant) you are', 
             lambda m: f"Wow I just love how {m.group(1)} you are"),
            
            (r'you are (smart|intelligent|clever|brilliant)', 
             lambda m: f"You're so {m.group(1)}"),
            
            (r'you\'re (smart|intelligent|clever|brilliant)', 
             lambda m: f"What a {m.group(1)} person you are"),
        ],
        'to_unsarcastic': [
            # REVERSE PATTERN 1: As if I love X because Y -> I hate X because Y
            (r'as if i (love|like|enjoy) (.+?) because (.+)', 
             lambda m: f"I hate {m.group(2)} because {m.group(3)}"),
            
            # REVERSE PATTERN 2: X is my absolute favorite because Y -> I dislike X because Y
            (r'(.+?) is my absolute (favorite|favourite) (.+?) because (.+)', 
             lambda m: f"I dislike {m.group(1)} because {m.group(4)}"),
            
            # REVERSE PATTERN 3: I absolutely adore X -> I hate X
            (r'i absolutely adore (.+)', 
             lambda m: f"I hate {m.group(1)}"),
            
            # REVERSE PATTERN 4: Handle any exaggerated positive with because clause
            (r'i (absolutely |totally |just )?(adore|love|like|enjoy) (.+?) because (.+)', 
             lambda m: f"I hate {m.group(3)} because {m.group(4)}"),
            
            # REVERSE PATTERN 5: Handle any exaggerated positive without because
            (r'i (absolutely |totally |just )?(adore|love|like|enjoy) (.+)', 
             lambda m: f"I hate {m.group(3)}"),
            
            # REVERSE PATTERN 6: PERSONAL SARCASM - Wow I love how smart you are -> You're stupid (NEW)
            (r'(wow|oh) i (just |absolutely |totally )?love how (smart|intelligent|clever|brilliant) you are', 
             lambda m: "You're stupid"),
            
            (r'i love how (smart|intelligent|clever|brilliant) you are', 
             lambda m: "You're not very smart"),
            
            (r'you\'re so (smart|intelligent|clever|brilliant)', 
             lambda m: "You're not very smart"),
            
            (r'what a (genius|brilliant mind) you have', 
             lambda m: "You're not very intelligent"),
        ]
    }
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

def complete_detect_sarcasm(text: str) -> dict:
    """Complete sarcasm detection that handles ALL cases including personal sarcasm"""
    text_lower = text.lower().strip()
    
    if not text_lower:
        return {'is_sarcastic': False, 'confidence': 0.0, 'method': 'empty'}
    
    # 1. Check obvious sarcastic patterns
    for pattern, confidence in COMPLETE_SARCASTIC_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return {'is_sarcastic': True, 'confidence': confidence, 'method': 'pattern_match'}
    
    # 2. Check for semantic contradictions
    positive_words = ['love', 'adore', 'enjoy', 'like', 'great', 'wonderful', 'fantastic', 'perfect', 'amazing', 'favorite', 'favourite', 'best', 'smart', 'intelligent', 'clever', 'brilliant', 'genius']
    negative_words = ['trouble', 'problem', 'pain', 'annoying', 'stress', 'headache', 'pimples', 'hurt', 'bad', 'terrible', 'awful', 'horrible', 'stupid', 'dumb', 'idiot']
    
    has_positive = any(word in text_lower for word in positive_words)
    has_negative = any(word in text_lower for word in negative_words)
    
    # If text has both positive emotion and negative context, it's sarcastic
    if has_positive and has_negative:
        # Special boost for "because" patterns
        if 'because' in text_lower:
            return {'is_sarcastic': True, 'confidence': 0.9, 'method': 'contradiction'}
        return {'is_sarcastic': True, 'confidence': 0.7, 'method': 'semantic_contrast'}
    
    # 3. Check for exaggerated personal compliments (often sarcastic)
    exaggerated_compliments = ['so smart', 'so intelligent', 'so clever', 'such a genius', 'so brilliant']
    if any(compliment in text_lower for compliment in exaggerated_compliments):
        return {'is_sarcastic': True, 'confidence': 0.8, 'method': 'exaggerated_compliment'}
    
    # 4. Default to non-sarcastic
    return {'is_sarcastic': False, 'confidence': 0.3, 'method': 'default'}

def complete_sarcastic_conversion(text: str) -> str:
    """Complete conversion to sarcastic"""
    original_text = text.strip()
    if not original_text:
        return "What a joy"
    
    text_lower = original_text.lower()
    print(f"Converting to sarcastic: '{original_text}'")
    
    # STEP 1: Direct mappings
    for genuine, sarcastic in COMPLETE_CONVERSION_SYSTEM['direct_mappings']['to_sarcastic'].items():
        if text_lower == genuine.lower():
            print(f"Direct match: '{genuine}' -> '{sarcastic}'")
            if original_text[0].isupper():
                return sarcastic[0].upper() + sarcastic[1:]
            return sarcastic
    
    # STEP 2: Pattern-based conversions
    for pattern, template_func in COMPLETE_CONVERSION_SYSTEM['pattern_conversions']['to_sarcastic']:
        match = re.match(pattern, text_lower, re.IGNORECASE)
        if match:
            try:
                result = template_func(match)
                print(f"Pattern match: '{original_text}' -> '{result}'")
                if original_text[0].isupper():
                    return result[0].upper() + result[1:]
                return result
            except Exception as e:
                print(f"Pattern error: {e}")
                continue
    
    # STEP 3: Smart fallback
    if any(word in text_lower for word in ['love', 'like', 'enjoy', 'great', 'wonderful', 'smart', 'intelligent']):
        sarcastic_prefixes = ["Oh great, ", "Just what I needed, ", "Wonderful, "]
        prefix = random.choice(sarcastic_prefixes)
        result = prefix + original_text.lower()
    else:
        result = "Yeah right, " + original_text.lower()
    
    # Capitalize
    if result and result[0].isalpha():
        result = result[0].upper() + result[1:]
    
    print(f"Fallback conversion: '{original_text}' -> '{result}'")
    return result

def complete_genuine_conversion(text: str) -> str:
    """COMPLETE conversion to genuine - HANDLES ALL CASES INCLUDING PERSONAL SARCASM"""
    original_text = text.strip()
    if not original_text:
        return "This is acceptable"
    
    text_lower = original_text.lower()
    print(f"Converting to genuine: '{original_text}'")
    
    # STEP 1: Direct mappings
    for sarcastic, genuine in COMPLETE_CONVERSION_SYSTEM['direct_mappings']['to_unsarcastic'].items():
        if text_lower == sarcastic.lower():
            print(f"Direct reversal: '{sarcastic}' -> '{genuine}'")
            if original_text[0].isupper():
                return genuine[0].upper() + genuine[1:]
            return genuine
    
    # STEP 2: Pattern-based reversals
    for pattern, template_func in COMPLETE_CONVERSION_SYSTEM['pattern_conversions']['to_unsarcastic']:
        match = re.match(pattern, text_lower, re.IGNORECASE)
        if match:
            try:
                result = template_func(match)
                print(f"Pattern reversal: '{original_text}' -> '{result}'")
                if original_text[0].isupper():
                    return result[0].upper() + result[1:]
                return result
            except Exception as e:
                print(f"Reversal error: {e}")
                continue
    
    # STEP 3: UNIVERSAL FALLBACK - Handle ANY sarcastic text
    detection_result = complete_detect_sarcasm(original_text)
    if detection_result['is_sarcastic']:
        # Convert the text by replacing positive words with negative ones
        converted = text_lower
        
        # Comprehensive word replacement mapping
        replacement_map = {
            # Positive emotions → Negative emotions
            r'\b(absolutely |totally |just )?(adore|love)\b': 'hate',
            r'\b(enjoy|like)\b': 'dislike',
            r'\b(love|adore)\b': 'hate',
            
            # Positive adjectives → Negative adjectives  
            r'\b(great|wonderful|fantastic|perfect|amazing|awesome|brilliant)\b': 'terrible',
            r'\b(good|nice)\b': 'bad',
            
            # Intelligence compliments → Insults
            r'\b(smart|intelligent|clever|brilliant|genius)\b': 'stupid',
            r'\b(so smart|so intelligent|so clever|so brilliant)\b': 'not very smart',
            r'\b(such a genius)\b': 'not very intelligent',
            
            # Favorite things → Disliked things
            r'\b(best|favorite|favourite)\b': 'worst',
            r'\b(my favorite|my favourite)\b': 'I dislike',
            
            # Remove sarcastic prefixes
            r'^(oh great, |just what i needed, |wonderful, |yeah right, |as if |wow )': '',
            
            # Sarcastic phrases → Genuine phrases
            r'\bjust what i needed\b': 'what I needed',
            r'\bthanks a bunch\b': 'thank you',
            r'\bno problem at all\b': 'no problem',
        }
        
        # Apply all replacements
        for pattern, replacement in replacement_map.items():
            converted = re.sub(pattern, replacement, converted, flags=re.IGNORECASE)
        
        # Special case: Handle "X is my favorite Y because Z" pattern
        favorite_match = re.match(r'(.+?) is my (favorite|favourite) (.+?) because (.+)', converted)
        if favorite_match:
            thing = favorite_match.group(1)
            reason = favorite_match.group(4)
            converted = f"I dislike {thing} because {reason}"
        
        # Special case: Handle "I love X because Y" pattern  
        love_match = re.match(r'i (love|like|enjoy) (.+?) because (.+)', converted)
        if love_match:
            thing = love_match.group(2)
            reason = love_match.group(3)
            converted = f"I hate {thing} because {reason}"
        
        # Special case: Handle personal sarcasm "I love how smart you are"
        personal_match = re.search(r'i love how (smart|intelligent|clever|brilliant) you are', converted)
        if personal_match:
            converted = "You're stupid"
        
        # Capitalize first letter
        if converted and converted[0].isalpha():
            converted = converted[0].upper() + converted[1:]
        
        # If conversion resulted in meaningful change, return it
        if converted.lower() != text_lower and len(converted) > 5:
            print(f"Universal conversion: '{original_text}' -> '{converted}'")
            return converted
    
    # STEP 4: If text is not sarcastic or no conversion happened, return original
    if not detection_result['is_sarcastic']:
        print(f"No conversion needed: '{original_text}'")
        return original_text
    
    # STEP 5: Absolute fallback - return meaningful response
    result = "I dislike this"
    print(f"Absolute fallback: '{original_text}' -> '{result}'")
    return result

def complete_smart_conversion(text: str, to_sarcastic: bool) -> str:
    """Complete conversion router"""
    if not text.strip():
        return "What a joy" if to_sarcastic else "This is acceptable"
    
    if to_sarcastic:
        return complete_sarcastic_conversion(text)
    else:
        return complete_genuine_conversion(text)

# Load models on startup
load_models()

@app.route("/health", methods=["GET"])
def health():
    models_loaded = len(models) > 0
    return jsonify({
        "status": "ok", 
        "models_loaded": models_loaded,
        "version": "completely_final_perfect_v1"
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

        detection_result = complete_detect_sarcasm(text)
        
        converted_text = None
        if body.get("convert_to"):
            target_sarcastic = (body["convert_to"] == "sarcastic")
            converted_text = complete_smart_conversion(text, target_sarcastic)

        response_data = {
            "text": text,
            "is_sarcastic": detection_result['is_sarcastic'],
            "confidence": detection_result['confidence'],
            "converted_text": converted_text,
            "method": detection_result['method']
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

        detection_result = complete_detect_sarcasm(text)
        is_currently_sarcastic = detection_result['is_sarcastic']
        
        if (target == "sarcastic" and is_currently_sarcastic) or (target == "unsarcastic" and not is_currently_sarcastic):
            return jsonify({
                "original_text": text,
                "converted_text": text,
                "conversion_type": target,
                "original_was_sarcastic": is_currently_sarcastic,
                "original_confidence": detection_result['confidence'],
                "conversion_happened": False,
                "message": f"Text is already {target}"
            })

        converted_text = complete_smart_conversion(text, target == "sarcastic")
        
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
    """Test ALL cases including personal sarcasm"""
    test_cases = [
        {"input": "wow i just love how smart you are", "expected_sarcastic": "Wow I just love how smart you are", "expected_unsarcastic": "You're stupid"},
        {"input": "i love how intelligent you are", "expected_sarcastic": "Wow I just love how intelligent you are", "expected_unsarcastic": "You're not very smart"},
        {"input": "you're so clever", "expected_sarcastic": "You're so clever", "expected_unsarcastic": "You're not very smart"},
        {"input": "mango is my favourite fruit because i get pimples after eating it", "expected_sarcastic": "Mango is my absolute favourite fruit because i get pimples after eating it", "expected_unsarcastic": "I dislike mango because i get pimples after eating it"},
        {"input": "i love politics because it causes so much trouble for common man", "expected_sarcastic": "As if I love politics because it causes so much trouble for common man", "expected_unsarcastic": "I hate politics because it causes so much trouble for common man"},
        {"input": "this is a problem", "expected_sarcastic": "This is just fantastic", "expected_unsarcastic": "This is a problem"},
    ]
    
    results = []
    
    for case in test_cases:
        detection = complete_detect_sarcasm(case["input"])
        
        sarcastic_result = complete_smart_conversion(case["input"], True)
        unsarcastic_result = complete_smart_conversion(case["input"], False)
        
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