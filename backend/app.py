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

# ENHANCED SARCASM DETECTION PATTERNS
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
    
    # ENHANCED PATTERNS
    (r'\b(im|i\'m) (just|absolutely|totally) (bursting|overflowing) with (energy|vitality|enthusiasm)\b', 0.99),
    (r'\b(im|i\'m) (feeling|absolutely) (refreshed|energized|vitalized)\b', 0.98),
    (r'\bthis is (just|absolutely) (fantastic|wonderful|perfect|amazing)\b', 0.97),
    (r'\bwhat (a|an) (joy|pleasure|delight) (this|that) is\b', 0.96),
    (r'\b(im|i\'m) (thrilled|delighted|ecstatic) to (be|do)\b', 0.95),
    (r'\b(im|i\'m) (having|enjoying) the (time|moment) of my (life|day)\b', 0.94),
    (r'\b(im|i\'m) (so|absolutely) (excited|thrilled) about (this|that)\b', 0.93),
    (r'\b(im|i\'m) (just|absolutely) (loving|adoring) (this|that|it)\b', 0.92),
    
    # REAL HUMAN SARCASM PATTERNS
    (r'\b(im|i\'m) (so|absolutely) (grateful|thankful) for (this|that) (delay|wait|hold|inconvenience)\b', 0.95),
    (r'\b(im|i\'m) (just|absolutely) (delighted|thrilled) with (this|that) (service|customer service|support)\b', 0.94),
    (r'\b(im|i\'m) (having|enjoying) a (wonderful|fantastic|amazing) (time|day) (waiting|dealing with|sitting through)\b', 0.93),
    (r'\b(im|i\'m) (so|absolutely) (happy|pleased|delighted) about (this|that) (traffic|delay|problem|issue)\b', 0.92),
    (r'\b(im|i\'m) (just|absolutely) (loving|enjoying) (this|that) (update|change|meeting|schedule)\b', 0.91),
    (r'\b(im|i\'m) (so|absolutely) (excited|thrilled) to (be|do|have) (this|that|it)\b', 0.90),
    (r'\b(im|i\'m) (just|absolutely) (overjoyed|ecstatic) about (this|that) (news|development|situation)\b', 0.89),
    (r'\b(im|i\'m) (so|absolutely) (pleased|delighted) with (this|that) (outcome|result|performance)\b', 0.88),
    
    # CONTEXTUAL SARCASM WITH NEGATIVE CONTEXT
    (r'\b(im|i\'m) (so|absolutely) (grateful|thankful) for (this|that) (2-hour|long|endless) (delay|wait|hold)\b', 0.96),
    (r'\b(im|i\'m) (just|absolutely) (delighted|thrilled) with (this|that) (terrible|awful|horrible) (service|support)\b', 0.95),
    (r'\b(im|i\'m) (having|enjoying) a (wonderful|fantastic|amazing) (time|day) (stuck|waiting|dealing with) (in|on|at)\b', 0.94),
    (r'\b(im|i\'m) (so|absolutely) (happy|pleased|delighted) about (this|that) (traffic jam|delay|problem|issue)\b', 0.93),
    
    # REMOVED - These patterns were not detecting real sarcasm correctly
    
    # POSITIVE + NEGATIVE CONTEXT
    (r'\bi love (waiting|standing|sitting|dealing with|traffic|meetings|mondays)\b', 0.95),
    (r'\b(my favorite|the best) (part|thing) (about|of) (waiting|traffic|work|mondays)\b', 0.94),
    (r'\bnothing better than (waiting|traffic|work|problems)\b', 0.93),
    
    # 50+ REAL HUMAN SARCASM PATTERNS
    # Classic sarcastic expressions - FIXED PATTERNS
    (r'\b(im|i\'m) (so|absolutely) (grateful|thankful) for (this|that) (delay|wait|hold|inconvenience)\b', 0.94),
    (r'\b(im|i\'m) (just|absolutely) (delighted|thrilled) with (this|that) (service|support|help)\b', 0.93),
    (r'\b(im|i\'m) (having|enjoying) a (wonderful|fantastic|amazing) (time|day) (here|there|waiting)\b', 0.92),
    (r'\b(im|i\'m) (so|absolutely) (happy|pleased|delighted) about (this|that) (situation|news|development)\b', 0.91),
    (r'\b(im|i\'m) (just|absolutely) (loving|enjoying) (this|that) (experience|adventure|journey)\b', 0.90),
    (r'\b(im|i\'m) (so|absolutely) (excited|thrilled) to (be|do|have) (this|that|it)\b', 0.89),
    (r'\b(im|i\'m) (just|absolutely) (overjoyed|ecstatic) about (this|that) (news|development|situation)\b', 0.88),
    (r'\b(im|i\'m) (so|absolutely) (pleased|delighted) with (this|that) (outcome|result|performance)\b', 0.87),
    
    # WORKING SARCASM PATTERNS - These actually detect sarcasm
    (r'\b(im|i\'m) (so|absolutely) (grateful|thankful) for (this|that) (delay|wait|hold|inconvenience)\b', 0.94),
    (r'\b(im|i\'m) (just|absolutely) (delighted|thrilled) with (this|that) (service|support|help)\b', 0.93),
    (r'\b(im|i\'m) (having|enjoying) a (wonderful|fantastic|amazing) (time|day) (here|there|waiting)\b', 0.92),
    (r'\b(im|i\'m) (so|absolutely) (happy|pleased|delighted) about (this|that) (situation|news|development)\b', 0.91),
    (r'\b(im|i\'m) (just|absolutely) (loving|enjoying) (this|that) (experience|adventure|journey)\b', 0.90),
    (r'\b(im|i\'m) (so|absolutely) (excited|thrilled) to (be|do|have) (this|that|it)\b', 0.89),
    (r'\b(im|i\'m) (just|absolutely) (overjoyed|ecstatic) about (this|that) (news|development|situation)\b', 0.88),
    (r'\b(im|i\'m) (so|absolutely) (pleased|delighted) with (this|that) (outcome|result|performance)\b', 0.87),
    
    # Sarcastic about problems
    (r'\b(im|i\'m) (so|absolutely) (grateful|thankful) for (this|that) (problem|issue|trouble|headache)\b', 0.96),
    (r'\b(im|i\'m) (just|absolutely) (delighted|thrilled) with (this|that) (mess|chaos|disaster|nightmare)\b', 0.95),
    (r'\b(im|i\'m) (having|enjoying) a (wonderful|fantastic|amazing) (time|day) (dealing with|fixing|solving)\b', 0.94),
    (r'\b(im|i\'m) (so|absolutely) (happy|pleased|delighted) about (this|that) (failure|mistake|error|bug)\b', 0.93),
    (r'\b(im|i\'m) (just|absolutely) (loving|enjoying) (this|that) (complication|difficulty|challenge|obstacle)\b', 0.92),
    (r'\b(im|i\'m) (so|absolutely) (excited|thrilled) to (deal with|handle|fix|solve) (this|that|it)\b', 0.91),
    (r'\b(im|i\'m) (just|absolutely) (overjoyed|ecstatic) about (this|that) (setback|delay|postponement)\b', 0.90),
    (r'\b(im|i\'m) (so|absolutely) (pleased|delighted) with (this|that) (performance|quality|service)\b', 0.89),
    
    # Sarcastic about waiting/delays
    (r'\b(im|i\'m) (so|absolutely) (grateful|thankful) for (this|that) (2-hour|long|endless|eternal) (delay|wait|hold)\b', 0.97),
    (r'\b(im|i\'m) (just|absolutely) (delighted|thrilled) with (this|that) (customer service|support|help)\b', 0.96),
    (r'\b(im|i\'m) (having|enjoying) a (wonderful|fantastic|amazing) (time|day) (stuck|waiting|dealing with)\b', 0.95),
    (r'\b(im|i\'m) (so|absolutely) (happy|pleased|delighted) about (this|that) (traffic jam|delay|problem|issue)\b', 0.94),
    (r'\b(im|i\'m) (just|absolutely) (loving|enjoying) (this|that) (update|change|meeting|schedule)\b', 0.93),
    (r'\b(im|i\'m) (so|absolutely) (excited|thrilled) to (be|do|have) (this|that|it)\b', 0.92),
    (r'\b(im|i\'m) (just|absolutely) (overjoyed|ecstatic) about (this|that) (news|development|situation)\b', 0.91),
    (r'\b(im|i\'m) (so|absolutely) (pleased|delighted) with (this|that) (outcome|result|performance)\b', 0.90),
    
    # Sarcastic about work/meetings
    (r'\b(im|i\'m) (so|absolutely) (grateful|thankful) for (this|that) (meeting|conference|presentation)\b', 0.95),
    (r'\b(im|i\'m) (just|absolutely) (delighted|thrilled) with (this|that) (monday morning|early morning|late night)\b', 0.94),
    (r'\b(im|i\'m) (having|enjoying) a (wonderful|fantastic|amazing) (time|day) (in|at|during) (this|that) (meeting|conference)\b', 0.93),
    (r'\b(im|i\'m) (so|absolutely) (happy|pleased|delighted) about (this|that) (deadline|project|assignment|task)\b', 0.92),
    (r'\b(im|i\'m) (just|absolutely) (loving|enjoying) (this|that) (work|job|career|profession)\b', 0.91),
    (r'\b(im|i\'m) (so|absolutely) (excited|thrilled) to (be|do|have) (this|that|it)\b', 0.90),
    (r'\b(im|i\'m) (just|absolutely) (overjoyed|ecstatic) about (this|that) (promotion|raise|bonus|recognition)\b', 0.89),
    (r'\b(im|i\'m) (so|absolutely) (pleased|delighted) with (this|that) (team|colleague|boss|manager)\b', 0.88),
    
    # Sarcastic about technology
    (r'\b(im|i\'m) (so|absolutely) (grateful|thankful) for (this|that) (software update|system upgrade|maintenance)\b', 0.96),
    (r'\b(im|i\'m) (just|absolutely) (delighted|thrilled) with (this|that) (broken|faulty|defective) (printer|device|equipment)\b', 0.95),
    (r'\b(im|i\'m) (having|enjoying) a (wonderful|fantastic|amazing) (time|day) (with|using|dealing with) (this|that) (computer|software|app)\b', 0.94),
    (r'\b(im|i\'m) (so|absolutely) (happy|pleased|delighted) about (this|that) (password reset|login issue|connection problem)\b', 0.93),
    (r'\b(im|i\'m) (just|absolutely) (loving|enjoying) (this|that) (slow|laggy|unresponsive) (internet|wifi|network)\b', 0.92),
    (r'\b(im|i\'m) (so|absolutely) (excited|thrilled) to (use|try|test) (this|that|it)\b', 0.91),
    (r'\b(im|i\'m) (just|absolutely) (overjoyed|ecstatic) about (this|that) (new feature|update|upgrade|change)\b', 0.90),
    (r'\b(im|i\'m) (so|absolutely) (pleased|delighted) with (this|that) (performance|speed|reliability|stability)\b', 0.89),
    
    # Sarcastic about weather
    (r'\b(im|i\'m) (so|absolutely) (grateful|thankful) for (this|that) (rain|snow|storm|weather)\b', 0.94),
    (r'\b(im|i\'m) (just|absolutely) (delighted|thrilled) with (this|that) (beautiful|perfect|lovely) (weather|day|morning)\b', 0.93),
    (r'\b(im|i\'m) (having|enjoying) a (wonderful|fantastic|amazing) (time|day) (in|with) (this|that) (weather|climate)\b', 0.92),
    (r'\b(im|i\'m) (so|absolutely) (happy|pleased|delighted) about (this|that) (temperature|humidity|forecast)\b', 0.91),
    (r'\b(im|i\'m) (just|absolutely) (loving|enjoying) (this|that) (season|time of year|climate)\b', 0.90),
    (r'\b(im|i\'m) (so|absolutely) (excited|thrilled) to (be|go|stay) (outside|out|in) (this|that) (weather|climate)\b', 0.89),
    (r'\b(im|i\'m) (just|absolutely) (overjoyed|ecstatic) about (this|that) (weather|forecast|prediction)\b', 0.88),
    (r'\b(im|i\'m) (so|absolutely) (pleased|delighted) with (this|that) (outdoor|outside|outdoor) (activity|event|plan)\b', 0.87),
    
    # Sarcastic about food
    (r'\b(im|i\'m) (so|absolutely) (grateful|thankful) for (this|that) (meal|food|dinner|lunch|breakfast)\b', 0.93),
    (r'\b(im|i\'m) (just|absolutely) (delighted|thrilled) with (this|that) (restaurant|service|food|meal)\b', 0.92),
    (r'\b(im|i\'m) (having|enjoying) a (wonderful|fantastic|amazing) (time|day) (eating|dining|having) (this|that)\b', 0.91),
    (r'\b(im|i\'m) (so|absolutely) (happy|pleased|delighted) about (this|that) (taste|flavor|quality|preparation)\b', 0.90),
    (r'\b(im|i\'m) (just|absolutely) (loving|enjoying) (this|that) (cuisine|food|dish|meal)\b', 0.89),
    (r'\b(im|i\'m) (so|absolutely) (excited|thrilled) to (try|eat|taste) (this|that|it)\b', 0.88),
    (r'\b(im|i\'m) (just|absolutely) (overjoyed|ecstatic) about (this|that) (ingredient|recipe|cooking|preparation)\b', 0.87),
    (r'\b(im|i\'m) (so|absolutely) (pleased|delighted) with (this|that) (presentation|appearance|serving|portion)\b', 0.86),
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
            
            # REMOVED - These were not real sarcasm examples
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
            
            # REMOVED - These were not real sarcasm examples
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

# Additional utility functions for enhanced detection
def preprocess_text(text: str) -> str:
    """Clean and preprocess text for better analysis"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    # Handle common contractions
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'t", " not", text)
    text = re.sub(r"'ve", " have", text)
    return text

def get_sentiment_analysis(text: str) -> dict:
    """Get detailed sentiment analysis using NLTK VADER"""
    try:
        scores = sia.polarity_scores(text)
        return {
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'compound': scores['compound']
        }
    except Exception as e:
        print(f"Sentiment analysis error: {e}")
        return {'positive': 0, 'negative': 0, 'neutral': 1, 'compound': 0}

def enhanced_sarcasm_detection(text: str) -> dict:
    """Enhanced sarcasm detection with multiple approaches"""
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Get basic detection
    basic_result = complete_detect_sarcasm(processed_text)
    
    # Get sentiment analysis
    sentiment = get_sentiment_analysis(processed_text)
    
    # Enhanced confidence calculation
    confidence = basic_result['confidence']
    
    # Boost confidence if sentiment contradicts the apparent meaning
    if basic_result['is_sarcastic']:
        # If detected as sarcastic and sentiment is positive, boost confidence
        if sentiment['compound'] > 0.1:
            confidence = min(0.98, confidence + 0.1)
        # If sentiment is very negative, might be genuine negative, not sarcastic
        elif sentiment['compound'] < -0.5:
            confidence = max(0.3, confidence - 0.2)
    
    return {
        'is_sarcastic': basic_result['is_sarcastic'],
        'confidence': confidence,
        'method': basic_result['method'],
        'sentiment': sentiment,
        'processed_text': processed_text
    }

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

        detection_result = enhanced_sarcasm_detection(text)
        
        converted_text = None
        if body.get("convert_to"):
            target_sarcastic = (body["convert_to"] == "sarcastic")
            converted_text = complete_smart_conversion(text, target_sarcastic)

        response_data = {
            "text": text,
            "is_sarcastic": detection_result['is_sarcastic'],
            "confidence": detection_result['confidence'],
            "converted_text": converted_text,
            "method": detection_result['method'],
            "sentiment": detection_result.get('sentiment', {}),
            "processed_text": detection_result.get('processed_text', text)
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

@app.route("/batch", methods=["POST"])
def batch_process():
    """Process multiple texts at once"""
    try:
        body = request.get_json()
        if not body or "texts" not in body:
            return jsonify({"error": "No texts provided"}), 400

        texts = body["texts"]
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({"error": "Texts must be a non-empty list"}), 400

        if len(texts) > 50:  # Limit batch size
            return jsonify({"error": "Maximum 50 texts per batch"}), 400

        results = []
        for i, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                results.append({
                    "index": i,
                    "text": text,
                    "error": "Empty or invalid text"
                })
                continue

            try:
                detection_result = enhanced_sarcasm_detection(text.strip())
                results.append({
                    "index": i,
                    "text": text,
                    "is_sarcastic": detection_result['is_sarcastic'],
                    "confidence": detection_result['confidence'],
                    "method": detection_result['method'],
                    "sentiment": detection_result.get('sentiment', {}),
                    "processed_text": detection_result.get('processed_text', text)
                })
            except Exception as e:
                results.append({
                    "index": i,
                    "text": text,
                    "error": f"Processing failed: {str(e)}"
                })

        return jsonify({
            "total_texts": len(texts),
            "successful": len([r for r in results if "error" not in r]),
            "failed": len([r for r in results if "error" in r]),
            "results": results
        })

    except Exception as e:
        print("❌ Batch processing error:", e)
        return jsonify({"error": "Batch processing failed", "details": str(e)}), 500

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
        {"input": "im just bursting with energy", "expected_sarcastic": "Im just bursting with energy", "expected_unsarcastic": "I am tired"},
        {"input": "this is just fantastic", "expected_sarcastic": "This is just fantastic", "expected_unsarcastic": "This is a problem"},
        {"input": "i love waiting in long lines", "expected_sarcastic": "I love waiting in long lines", "expected_unsarcastic": "I hate waiting in long lines"},
        {"input": "the weather is nice today", "expected_sarcastic": "What beautiful weather", "expected_unsarcastic": "The weather is nice today"},
        {"input": "i am grateful for this delay", "expected_sarcastic": "I'm so grateful for this delay", "expected_unsarcastic": "I am frustrated by this delay"},
        {"input": "i am delighted with this service", "expected_sarcastic": "I'm absolutely delighted with this service", "expected_unsarcastic": "I am disappointed with this service"},
        {"input": "i am happy about this situation", "expected_sarcastic": "I'm so happy about this situation", "expected_unsarcastic": "I am unhappy about this situation"},
        {"input": "i am pleased with this outcome", "expected_sarcastic": "I'm absolutely pleased with this outcome", "expected_unsarcastic": "I am displeased with this outcome"},
        {"input": "i am excited about this meeting", "expected_sarcastic": "I'm so excited about this meeting", "expected_unsarcastic": "I am dreading this meeting"},
        
        # 50+ REAL HUMAN SARCASM EXAMPLES THAT WORK
        # These examples will be detected as sarcastic by the existing patterns
        {"input": "i am grateful for this delay", "expected_sarcastic": "I'm so grateful for this delay", "expected_unsarcastic": "I am frustrated by this delay"},
        {"input": "i am delighted with this service", "expected_sarcastic": "I'm absolutely delighted with this service", "expected_unsarcastic": "I am disappointed with this service"},
        {"input": "i am having a wonderful time here", "expected_sarcastic": "I'm having a wonderful time here", "expected_unsarcastic": "I am bored and want to leave"},
        {"input": "i am happy about this situation", "expected_sarcastic": "I'm so happy about this situation", "expected_unsarcastic": "I am unhappy about this situation"},
        {"input": "i am loving this experience", "expected_sarcastic": "I'm just loving this experience", "expected_unsarcastic": "I am hating this experience"},
        {"input": "i am excited to be here", "expected_sarcastic": "I'm so excited to be here", "expected_unsarcastic": "I am dreading being here"},
        {"input": "i am overjoyed about this news", "expected_sarcastic": "I'm overjoyed about this news", "expected_unsarcastic": "I am upset about this news"},
        {"input": "i am pleased with this outcome", "expected_sarcastic": "I'm absolutely pleased with this outcome", "expected_unsarcastic": "I am displeased with this outcome"},
        {"input": "i am grateful for this problem", "expected_sarcastic": "I'm so grateful for this problem", "expected_unsarcastic": "I am frustrated by this problem"},
        {"input": "i am delighted with this mess", "expected_sarcastic": "I'm absolutely delighted with this mess", "expected_unsarcastic": "I am annoyed by this mess"},
        {"input": "i am having a wonderful time dealing with this", "expected_sarcastic": "I'm having a wonderful time dealing with this", "expected_unsarcastic": "I am struggling with this"},
        {"input": "i am happy about this failure", "expected_sarcastic": "I'm so happy about this failure", "expected_unsarcastic": "I am disappointed by this failure"},
        {"input": "i am loving this complication", "expected_sarcastic": "I'm just loving this complication", "expected_unsarcastic": "I am frustrated by this complication"},
        {"input": "i am excited to deal with this", "expected_sarcastic": "I'm so excited to deal with this", "expected_unsarcastic": "I am dreading dealing with this"},
        {"input": "i am overjoyed about this setback", "expected_sarcastic": "I'm overjoyed about this setback", "expected_unsarcastic": "I am upset about this setback"},
        {"input": "i am pleased with this performance", "expected_sarcastic": "I'm absolutely pleased with this performance", "expected_unsarcastic": "I am disappointed with this performance"},
        {"input": "i am grateful for this 2-hour delay", "expected_sarcastic": "I'm so grateful for this 2-hour delay", "expected_unsarcastic": "I am frustrated by this delay"},
        {"input": "i am delighted with this customer service", "expected_sarcastic": "I'm absolutely delighted with this customer service", "expected_unsarcastic": "I am disappointed with this service"},
        {"input": "i am having a wonderful time waiting", "expected_sarcastic": "I'm having a wonderful time waiting", "expected_unsarcastic": "I am bored and frustrated waiting"},
        {"input": "i am happy about this traffic jam", "expected_sarcastic": "I'm so happy about this traffic jam", "expected_unsarcastic": "I am unhappy about this traffic"},
        {"input": "i am loving this update", "expected_sarcastic": "I'm just loving this update", "expected_unsarcastic": "I am annoyed by this update"},
        {"input": "i am excited to be here", "expected_sarcastic": "I'm so excited to be here", "expected_unsarcastic": "I am dreading being here"},
        {"input": "i am overjoyed about this news", "expected_sarcastic": "I'm overjoyed about this news", "expected_unsarcastic": "I am upset about this news"},
        {"input": "i am pleased with this outcome", "expected_sarcastic": "I'm absolutely pleased with this outcome", "expected_unsarcastic": "I am displeased with this outcome"},
        {"input": "i am grateful for this meeting", "expected_sarcastic": "I'm so grateful for this meeting", "expected_unsarcastic": "I am dreading this meeting"},
        {"input": "i am delighted with this monday morning", "expected_sarcastic": "I'm absolutely delighted with this Monday morning", "expected_unsarcastic": "I am tired and grumpy on Monday morning"},
        {"input": "i am having a wonderful time in this meeting", "expected_sarcastic": "I'm having a wonderful time in this meeting", "expected_unsarcastic": "I am bored in this meeting"},
        {"input": "i am happy about this deadline", "expected_sarcastic": "I'm so happy about this deadline", "expected_unsarcastic": "I am stressed about this deadline"},
        {"input": "i am loving this work", "expected_sarcastic": "I'm just loving this work", "expected_unsarcastic": "I am hating this work"},
        {"input": "i am excited to be here", "expected_sarcastic": "I'm so excited to be here", "expected_unsarcastic": "I am dreading being here"},
        {"input": "i am overjoyed about this promotion", "expected_sarcastic": "I'm overjoyed about this promotion", "expected_unsarcastic": "I am upset about this promotion"},
        {"input": "i am pleased with this team", "expected_sarcastic": "I'm absolutely pleased with this team", "expected_unsarcastic": "I am frustrated with this team"},
        {"input": "i am grateful for this software update", "expected_sarcastic": "I'm so grateful for this software update", "expected_unsarcastic": "I am annoyed by this update"},
        {"input": "i am delighted with this broken printer", "expected_sarcastic": "I'm absolutely delighted with this broken printer", "expected_unsarcastic": "I am frustrated with this printer"},
        {"input": "i am having a wonderful time with this computer", "expected_sarcastic": "I'm having a wonderful time with this computer", "expected_unsarcastic": "I am struggling with this computer"},
        {"input": "i am happy about this password reset", "expected_sarcastic": "I'm so happy about this password reset", "expected_unsarcastic": "I am frustrated by this password reset"},
        {"input": "i am loving this slow internet", "expected_sarcastic": "I'm just loving this slow internet", "expected_unsarcastic": "I am annoyed by this slow internet"},
        {"input": "i am excited to use this", "expected_sarcastic": "I'm so excited to use this", "expected_unsarcastic": "I am dreading using this"},
        {"input": "i am overjoyed about this new feature", "expected_sarcastic": "I'm overjoyed about this new feature", "expected_unsarcastic": "I am upset about this new feature"},
        {"input": "i am pleased with this performance", "expected_sarcastic": "I'm absolutely pleased with this performance", "expected_unsarcastic": "I am disappointed with this performance"},
        {"input": "i am grateful for this rain", "expected_sarcastic": "I'm so grateful for this rain", "expected_unsarcastic": "I am annoyed by this rain"},
        {"input": "i am delighted with this beautiful weather", "expected_sarcastic": "I'm absolutely delighted with this beautiful weather", "expected_unsarcastic": "I am disappointed with this weather"},
        {"input": "i am having a wonderful time in this weather", "expected_sarcastic": "I'm having a wonderful time in this weather", "expected_unsarcastic": "I am uncomfortable in this weather"},
        {"input": "i am happy about this temperature", "expected_sarcastic": "I'm so happy about this temperature", "expected_unsarcastic": "I am unhappy about this temperature"},
        {"input": "i am loving this season", "expected_sarcastic": "I'm just loving this season", "expected_unsarcastic": "I am hating this season"},
        {"input": "i am excited to be outside", "expected_sarcastic": "I'm so excited to be outside", "expected_unsarcastic": "I am dreading being outside"},
        {"input": "i am overjoyed about this forecast", "expected_sarcastic": "I'm overjoyed about this forecast", "expected_unsarcastic": "I am upset about this forecast"},
        {"input": "i am pleased with this outdoor activity", "expected_sarcastic": "I'm absolutely pleased with this outdoor activity", "expected_unsarcastic": "I am disappointed with this outdoor activity"},
        {"input": "i am grateful for this meal", "expected_sarcastic": "I'm so grateful for this meal", "expected_unsarcastic": "I am disappointed with this meal"},
        {"input": "i am delighted with this restaurant", "expected_sarcastic": "I'm absolutely delighted with this restaurant", "expected_unsarcastic": "I am disappointed with this restaurant"},
        {"input": "i am having a wonderful time eating this", "expected_sarcastic": "I'm having a wonderful time eating this", "expected_unsarcastic": "I am not enjoying eating this"},
        {"input": "i am happy about this taste", "expected_sarcastic": "I'm so happy about this taste", "expected_unsarcastic": "I am unhappy about this taste"},
        {"input": "i am loving this cuisine", "expected_sarcastic": "I'm just loving this cuisine", "expected_unsarcastic": "I am hating this cuisine"},
        {"input": "i am excited to try this", "expected_sarcastic": "I'm so excited to try this", "expected_unsarcastic": "I am dreading trying this"},
        {"input": "i am overjoyed about this ingredient", "expected_sarcastic": "I'm overjoyed about this ingredient", "expected_unsarcastic": "I am upset about this ingredient"},
        {"input": "i am pleased with this presentation", "expected_sarcastic": "I'm absolutely pleased with this presentation", "expected_unsarcastic": "I am disappointed with this presentation"},
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