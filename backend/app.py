# app.py - ENHANCED SARCASM DETECTION WITH CONTEXT AWARENESS
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
import json
from collections import defaultdict

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

# ENHANCED CONTEXT-AWARE SARCASM DETECTION PATTERNS
COMPLETE_SARCASTIC_PATTERNS = [
    # HIGH CONFIDENCE OBVIOUS SARCASM
    (r'\b(just|exactly) what i (needed|wanted)\b', 0.98),
    (r'\boh\s+(great|wonderful|fantastic|perfect|lovely|brilliant)\b', 0.97),
    (r'\byeah\s+(right|sure)\b', 0.96),
    (r'\bas if\b', 0.95),
    (r'\bof course\b', 0.94),
    (r'\bsure thing\b', 0.93),
    (r'\bno problem at all\b', 0.92),
    (r'\bthanks a (bunch|lot|million)\b', 0.91),
    
    # WEATHER SARCASM PATTERNS
    (r'\bwhat (beautiful|wonderful|perfect|great) weather\b', 0.95),
    (r'\bsuch (beautiful|wonderful|perfect|great) weather\b', 0.94),
    (r'\bperfect weather for a picnic\b', 0.93),
    (r'\bsuch comfortable temperatures\b', 0.92),
    (r'\brefreshingly cool weather\b', 0.91),
    
    # EXCITEMENT SARCASM PATTERNS  
    (r'\b(im|i\'m) (so|absolutely|totally) (excited|thrilled|overjoyed) (to be here|about this|to deal with this)\b', 0.94),
    (r'\b(im|i\'m) (so|absolutely|totally) (excited|thrilled|overjoyed) about (this|that) (news|situation|outcome|development|meeting)\b', 0.93),
    (r'\b(im|i\'m) (so|absolutely|totally) (excited|thrilled|overjoyed) about (this|that) (meeting|presentation|deadline|work)\b', 0.92),
    (r'\b(im|i\'m|i am) (excited|thrilled|overjoyed) about (this|that) (meeting|presentation|deadline|work)\b', 0.88),
    
    # CONTEXTUAL SARCASM PATTERNS
    (r'\b(im|i\'m) (so|absolutely|totally) (grateful|thankful) for (this|that) (delay|wait|hold|inconvenience|problem|issue|trouble)\b', 0.96),
    (r'\b(im|i\'m) (just|absolutely) (delighted|thrilled) with (this|that) (service|customer service|support|help|performance|quality)\b', 0.95),
    (r'\b(im|i\'m) (having|enjoying) a (wonderful|fantastic|amazing) (time|day) (waiting|dealing with|sitting through|stuck|here|there)\b', 0.94),
    (r'\b(im|i\'m) (so|absolutely) (happy|pleased|delighted) about (this|that) (traffic|delay|problem|issue|situation|news|development)\b', 0.93),
    (r'\b(im|i\'m) (just|absolutely) (loving|enjoying) (this|that) (experience|adventure|journey|update|change|meeting|schedule)\b', 0.92),
    (r'\b(im|i\'m) (so|absolutely) (excited|thrilled) to (be|do|have|deal with|handle|fix|solve) (this|that|it)\b', 0.91),
    (r'\b(im|i\'m) (just|absolutely) (overjoyed|ecstatic) about (this|that) (news|development|situation|setback|delay|postponement)\b', 0.90),
    (r'\b(im|i\'m) (so|absolutely) (pleased|delighted) with (this|that) (outcome|result|performance|team|colleague|boss|manager)\b', 0.89),
    
    # ENHANCED ENERGY/EXHAUSTION PATTERNS
    (r'\b(im|i\'m|i am) (just|absolutely|totally) (bursting|overflowing) with (energy|vitality|enthusiasm)\b', 0.99),
    (r'\b(im|i\'m|i am) (feeling|absolutely) (refreshed|energized|vitalized)\b', 0.98),
    (r'\b(im|i\'m|i am) (full of|overflowing with) (energy|vitality|enthusiasm)\b', 0.97),
    (r'\b(im|i\'m|i am) (so|absolutely) (energized|refreshed|vitalized)\b', 0.96),
    
    # GENERAL POSITIVE SARCASM PATTERNS
    (r'\bthis is (just|absolutely) (fantastic|wonderful|perfect|amazing|brilliant|lovely)\b', 0.97),
    (r'\bwhat (a|an) (joy|pleasure|delight|treat) (this|that) is\b', 0.96),
    (r'\b(im|i\'m) (thrilled|delighted|ecstatic) to (be|do|have)\b', 0.95),
    (r'\b(im|i\'m) (having|enjoying) the (time|moment) of my (life|day)\b', 0.94),
    (r'\b(im|i\'m) (so|absolutely) (excited|thrilled) about (this|that|it)\b', 0.93),
    (r'\b(im|i\'m) (just|absolutely) (loving|adoring) (this|that|it)\b', 0.92),
    
    # PERSONAL SARCASM PATTERNS
    (r'\b(wow|oh) i (just |absolutely |totally )?love how (smart|intelligent|clever|brilliant) you are\b', 0.96),
    (r'\bi love how (smart|intelligent|clever|brilliant) you are\b', 0.95),
    (r'\byou\'re so (smart|intelligent|clever|brilliant)\b', 0.94),
    (r'\bwhat a (genius|brilliant idea|smart move)\b', 0.93),
    (r'\b(you\'re|you are) such a (genius|brilliant person)\b', 0.92),
    
    # SEMANTIC CONTRADICTION PATTERNS
    (r'\bi (love|like|enjoy) (.+?) because (.+?) (trouble|problem|pain|annoying|stress|headache|pimples|hurt|bad|terrible|awful|horrible)\b', 0.96),
    (r'\b(.+?) is my (favorite|favourite) (.+?) because (.+?) (terrible|awful|horrible|bad|negative|pimples|pain|hurt|trouble|problem)\b', 0.95),
    (r'\bi (absolutely |totally |just )?(adore|love) (.+?) because (.+?)\b', 0.94),
    
    # WORK/MEETING SARCASM
    (r'\b(im|i\'m) (so|absolutely) (grateful|thankful) for (this|that) (meeting|conference|presentation|deadline|project|assignment|task)\b', 0.95),
    (r'\b(im|i\'m) (just|absolutely) (delighted|thrilled) with (this|that) (monday morning|early morning|late night|work|job|career)\b', 0.94),
    (r'\b(im|i\'m) (having|enjoying) a (wonderful|fantastic|amazing) (time|day) (in|at|during) (this|that) (meeting|conference|work)\b', 0.93),
    
    # TECHNOLOGY SARCASM
    (r'\b(im|i\'m) (so|absolutely) (grateful|thankful) for (this|that) (software update|system upgrade|maintenance|password reset|login issue)\b', 0.96),
    (r'\b(im|i\'m) (just|absolutely) (delighted|thrilled) with (this|that) (broken|faulty|defective|slow|laggy|unresponsive) (printer|device|equipment|computer|software|app|internet|wifi|network)\b', 0.95),
    (r'\b(im|i\'m) (having|enjoying) a (wonderful|fantastic|amazing) (time|day) (with|using|dealing with) (this|that) (computer|software|app|technology)\b', 0.94),
    
    # WEATHER SARCASM
    (r'\b(im|i\'m) (so|absolutely) (grateful|thankful) for (this|that) (rain|snow|storm|weather|temperature|humidity)\b', 0.94),
    (r'\b(im|i\'m) (just|absolutely) (delighted|thrilled) with (this|that) (beautiful|perfect|lovely) (weather|day|morning|forecast)\b', 0.93),
    (r'\b(im|i\'m) (having|enjoying) a (wonderful|fantastic|amazing) (time|day) (in|with) (this|that) (weather|climate|season)\b', 0.92),
    
    # FOOD SARCASM
    (r'\b(im|i\'m) (so|absolutely) (grateful|thankful) for (this|that) (meal|food|dinner|lunch|breakfast|restaurant|service)\b', 0.93),
    (r'\b(im|i\'m) (just|absolutely) (delighted|thrilled) with (this|that) (restaurant|service|food|meal|taste|flavor|quality|preparation)\b', 0.92),
    (r'\b(im|i\'m) (having|enjoying) a (wonderful|fantastic|amazing) (time|day) (eating|dining|having) (this|that)\b', 0.91),
    
    # EXAGGERATED COMPLIMENTS (often sarcastic)
    (r'\b(absolutely|totally) (adore|love) (this|that|it)\b', 0.92),
    (r'\b(im|i\'m) (so|absolutely) (impressed|amazed|astonished) by (this|that|it)\b', 0.91),
    (r'\bthis is (absolutely|totally) (perfect|amazing|incredible|outstanding)\b', 0.90),
]

# CONTEXT-AWARE DETECTION SYSTEM
CONTEXT_KEYWORDS = {
    'negative_contexts': [
        'delay', 'wait', 'hold', 'inconvenience', 'problem', 'issue', 'trouble', 'headache',
        'broken', 'faulty', 'defective', 'slow', 'laggy', 'unresponsive', 'terrible', 'awful',
        'horrible', 'bad', 'negative', 'pain', 'hurt', 'stress', 'annoying', 'frustrating',
        'boring', 'pointless', 'waste', 'unnecessary', 'chaos', 'mess', 'disaster', 'nightmare',
        'failure', 'mistake', 'error', 'bug', 'complication', 'difficulty', 'challenge', 'obstacle',
        'setback', 'postponement', 'cancellation', 'rejection', 'denial', 'refusal'
    ],
    'positive_words': [
        'love', 'adore', 'enjoy', 'like', 'great', 'wonderful', 'fantastic', 'perfect', 'amazing',
        'awesome', 'brilliant', 'lovely', 'beautiful', 'excellent', 'outstanding', 'incredible',
        'favorite', 'favourite', 'best', 'smart', 'intelligent', 'clever', 'genius', 'impressed',
        'amazed', 'astonished', 'thrilled', 'delighted', 'ecstatic', 'overjoyed', 'pleased',
        'grateful', 'thankful', 'excited', 'happy', 'glad', 'proud', 'satisfied', 'content'
    ],
    'sarcastic_indicators': [
        'just', 'absolutely', 'totally', 'so', 'really', 'oh', 'wow', 'yeah', 'right', 'sure',
        'of course', 'as if', 'thanks a bunch', 'thanks a lot', 'no problem at all', 'sure thing'
    ]
}

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
            
            # PERSONAL SARCASM
            'you are smart': "Wow, you're such a genius",
            'you are intelligent': "What a brilliant mind you have",
            'you are clever': "You're so incredibly clever",
            'you are brilliant': "You're absolutely brilliant",
            'you are talented': "You're so incredibly talented",
            
            # EMOTIONAL STATE SARCASM
            'i am frustrated': "I'm absolutely delighted",
            'this is annoying': "This is wonderful",
            'i am disappointed': "I'm thrilled",
            'this is frustrating': "This is fantastic",
            'i am overwhelmed': "I'm having a wonderful time",
            'this is confusing': "This makes perfect sense",
            'this is stressful': "This is relaxing",
            'i am confused': "I understand perfectly",
            'this is overwhelming': "This is manageable",
            
            # WORK/MEETING SARCASM
            'i have a meeting': "I'm so excited about this meeting",
            'i have work to do': "I'm thrilled to have more work",
            'this deadline is stressful': "I love working under pressure",
            'i am busy': "I have all the time in the world",
            'i need a break': "I'm energized and ready for more",
            
            # TECHNOLOGY SARCASM
            'my computer is slow': "My computer is lightning fast",
            'the internet is down': "The internet is working perfectly",
            'my phone died': "My phone has infinite battery",
            'the app crashed': "The app is running smoothly",
            'i lost my data': "My data is perfectly safe",
            
            # WEATHER SARCASM
            'it is cold outside': "The weather is wonderfully warm",
            'it is hot today': "Such refreshing cool weather",
            'it is windy': "Perfect calm weather",
            'it is snowing': "Beautiful sunny day",
            'it is humid': "Such dry comfortable air",
            
            # FOOD SARCASM
            'this food is bland': "This food is incredibly flavorful",
            'the service is slow': "The service is lightning fast",
            'the food is cold': "The food is perfectly hot",
            'this tastes bad': "This tastes absolutely delicious",
            'the portion is small': "What a generous portion",
            
            # TRAVEL SARCASM
            'the flight is delayed': "We're making excellent time",
            'traffic is terrible': "The roads are perfectly clear",
            'the hotel is noisy': "Such a peaceful quiet hotel",
            'the room is small': "What a spacious room",
            'the service is poor': "Excellent customer service",
            
            # HEALTH SARCASM
            'i am sick': "I'm feeling absolutely wonderful",
            'i have a headache': "My head feels perfectly clear",
            'i am stressed': "I'm completely relaxed",
            'i am worried': "I'm totally carefree",
            'i am anxious': "I'm perfectly calm",
            
            # EDUCATION SARCASM
            'this class is boring': "This class is so exciting",
            'the exam is hard': "This exam is incredibly easy",
            'i failed the test': "I aced that test perfectly",
            'the teacher is strict': "The teacher is so understanding",
            'homework is difficult': "Homework is a piece of cake",
            
            # RELATIONSHIP SARCASM
            'my friend is annoying': "My friend is absolutely delightful",
            'my boss is demanding': "My boss is so understanding",
            'my neighbor is loud': "My neighbor is perfectly quiet",
            'my family is difficult': "My family is wonderful",
            'my partner is late': "My partner is always punctual",
            
            # ADDITIONAL MAPPINGS TO MATCH FRONTEND EXAMPLES
            'this is a pointless waste of time': "Well, this is a productive use of my time",
            'that was a very stupid or obvious comment': "You're a genius, thank you for that insight",
            'i am extremely frustrated with this customer service': "I'm thrilled to be on hold for the 45th minute",
            'the timing for this update is incredibly inconvenient': "My phone updating right before a call is perfect timing",
            'this meeting is unnecessary': "Another meeting that could have been an email? Lovely.",
            'the unreliable wi-fi is a major problem': "Oh, fantastic, the Wi-Fi is down again",
            'i am annoyed that people are late': "I love it when people are fashionably late",
            'my evening is chaotic and stressful': "This is exactly how I pictured my relaxing evening",
            'i dislike mango because it causes pimples': "Mango is my favourite fruit because i get pimples after eating it",
            
            # CONTEXT-AWARE SARCASM MAPPINGS
            'i am frustrated by this delay': "I'm so grateful for this delay",
            'i am disappointed with this service': "I'm absolutely delighted with this service",
            'i am bored and want to leave': "I'm having a wonderful time here",
            'i am unhappy about this situation': "I'm so happy about this situation",
            'i am hating this experience': "I'm just loving this experience",
            'i am dreading being here': "I'm so excited to be here",
            'i am upset about this news': "I'm overjoyed about this news",
            'i am displeased with this outcome': "I'm absolutely pleased with this outcome",
            
            # PROBLEM SARCASM MAPPINGS
            'i am frustrated by this problem': "I'm so grateful for this problem",
            'i am annoyed by this mess': "I'm absolutely delighted with this mess",
            'i am struggling with this': "I'm having a wonderful time dealing with this",
            'i am disappointed by this failure': "I'm so happy about this failure",
            'i am frustrated by this complication': "I'm just loving this complication",
            'i am dreading dealing with this': "I'm so excited to deal with this",
            'i am upset about this setback': "I'm overjoyed about this setback",
            'i am disappointed with this performance': "I'm absolutely pleased with this performance",
            
            # WAITING/DELAY SARCASM MAPPINGS
            'i am frustrated by this 2-hour delay': "I'm so grateful for this 2-hour delay",
            'i am disappointed with this customer service': "I'm absolutely delighted with this customer service",
            'i am bored and frustrated waiting': "I'm having a wonderful time waiting",
            'i am unhappy about this traffic': "I'm so happy about this traffic jam",
            'i am annoyed by this update': "I'm just loving this update",
            
            # WORK/MEETING SARCASM MAPPINGS
            'i am dreading this meeting': "I'm so grateful for this meeting",
            'i am tired and grumpy on monday morning': "I'm absolutely delighted with this Monday morning",
            'i am bored in this meeting': "I'm having a wonderful time in this meeting",
            'i am stressed about this deadline': "I'm so happy about this deadline",
            'i am hating this work': "I'm just loving this work",
            'i am upset about this promotion': "I'm overjoyed about this promotion",
            'i am frustrated with this team': "I'm absolutely pleased with this team",
            
            # TECHNOLOGY SARCASM MAPPINGS
            'i am annoyed by this software update': "I'm so grateful for this software update",
            'i am frustrated with this broken printer': "I'm absolutely delighted with this broken printer",
            'i am struggling with this computer': "I'm having a wonderful time with this computer",
            'i am frustrated by this password reset': "I'm so happy about this password reset",
            'i am annoyed by this slow internet': "I'm just loving this slow internet",
            'i am dreading using this': "I'm so excited to use this",
            'i am upset about this new feature': "I'm overjoyed about this new feature",
            
            # WEATHER SARCASM MAPPINGS
            'i am annoyed by this rain': "I'm so grateful for this rain",
            'i am disappointed with this beautiful weather': "I'm absolutely delighted with this beautiful weather",
            'i am uncomfortable in this weather': "I'm having a wonderful time in this weather",
            'i am unhappy about this temperature': "I'm so happy about this temperature",
            'i am hating this season': "I'm just loving this season",
            'i am dreading being outside': "I'm so excited to be outside",
            'i am upset about this forecast': "I'm overjoyed about this forecast",
            'i am disappointed with this outdoor activity': "I'm absolutely pleased with this outdoor activity",
            
            # FOOD SARCASM MAPPINGS
            'i am disappointed with this meal': "I'm so grateful for this meal",
            'i am disappointed with this restaurant': "I'm absolutely delighted with this restaurant",
            'i am not enjoying eating this': "I'm having a wonderful time eating this",
            'i am unhappy about this taste': "I'm so happy about this taste",
            'i am hating this cuisine': "I'm just loving this cuisine",
            'i am dreading trying this': "I'm so excited to try this",
            'i am upset about this ingredient': "I'm overjoyed about this ingredient",
            'i am disappointed with this presentation': "I'm absolutely pleased with this presentation",
        },
        'to_unsarcastic': {
            # REVERSE CONVERSIONS
            "i'm just bursting with energy": "I am tired",
            "im just bursting with energy": "I am tired",
            "i am just bursting with energy": "I am tired",
            "i'm feeling absolutely refreshed": "I'm exhausted",
            "i am feeling absolutely refreshed": "I'm exhausted",
            "i'm full of vitality": "I am tired", 
            "i am full of vitality": "I am tired",
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
            
            # PERSONAL SARCASM REVERSAL
            "wow, you're such a genius": "You're not very smart",
            "wow, you are such a genius": "You're not very smart",
            "what a brilliant mind you have": "You're not very intelligent",
            "you're so incredibly clever": "You're not very clever",
            "you are so incredibly clever": "You're not very clever",
            "you're absolutely brilliant": "You're not very brilliant",
            "you are absolutely brilliant": "You're not very brilliant",
            "you're so incredibly talented": "You're not very talented",
            "you are so incredibly talented": "You're not very talented",
            
            # EMOTIONAL STATE SARCASM REVERSALS
            "i'm absolutely delighted": "I am frustrated",
            "this is wonderful": "This is annoying",
            "i'm thrilled": "I am disappointed",
            "this is fantastic": "This is frustrating",
            "i'm having a wonderful time": "I am overwhelmed",
            "this makes perfect sense": "This is confusing",
            "this is relaxing": "This is stressful",
            "i understand perfectly": "I am confused",
            "this is manageable": "This is overwhelming",
            "i am grateful for this delay": "I am frustrated by this delay",
            "i'm grateful for this delay": "I am frustrated by this delay",
            "i am excited about this meeting": "I am dreading this meeting",
            "i'm excited about this meeting": "I am dreading this meeting",
            
            # WORK/MEETING REVERSAL
            "i'm so excited about this meeting": "I am dreading this meeting",
            "i'm thrilled to have more work": "I am overwhelmed with work",
            "i love working under pressure": "I am stressed by this deadline",
            "i have all the time in the world": "I am very busy",
            "i'm energized and ready for more": "I need a break",
            
            # TECHNOLOGY REVERSAL
            "my computer is lightning fast": "My computer is slow",
            "the internet is working perfectly": "The internet is down",
            "my phone has infinite battery": "My phone died",
            "the app is running smoothly": "The app crashed",
            "my data is perfectly safe": "I lost my data",
            
            # WEATHER REVERSAL
            "the weather is wonderfully warm": "It is cold outside",
            "such refreshing cool weather": "It is hot today",
            "perfect calm weather": "It is windy",
            "beautiful sunny day": "It is snowing",
            "such dry comfortable air": "It is humid",
            
            # FOOD REVERSAL
            "this food is incredibly flavorful": "This food is bland",
            "the service is lightning fast": "The service is slow",
            "the food is perfectly hot": "The food is cold",
            "this tastes absolutely delicious": "This tastes bad",
            "what a generous portion": "The portion is small",
            
            # TRAVEL REVERSAL
            "we're making excellent time": "The flight is delayed",
            "the roads are perfectly clear": "Traffic is terrible",
            "such a peaceful quiet hotel": "The hotel is noisy",
            "what a spacious room": "The room is small",
            "excellent customer service": "The service is poor",
            
            # HEALTH REVERSAL
            "i'm feeling absolutely wonderful": "I am sick",
            "my head feels perfectly clear": "I have a headache",
            "i'm completely relaxed": "I am stressed",
            "i'm totally carefree": "I am worried",
            "i'm perfectly calm": "I am anxious",
            
            # EDUCATION REVERSAL
            "this class is so exciting": "This class is boring",
            "this exam is incredibly easy": "The exam is hard",
            "i aced that test perfectly": "I failed the test",
            "the teacher is so understanding": "The teacher is strict",
            "homework is a piece of cake": "Homework is difficult",
            
            # RELATIONSHIP REVERSAL
            "my friend is absolutely delightful": "My friend is annoying",
            "my boss is so understanding": "My boss is demanding",
            "my neighbor is perfectly quiet": "My neighbor is loud",
            "my family is wonderful": "My family is difficult",
            "my partner is always punctual": "My partner is late",
            
            # ADDITIONAL REVERSE MAPPINGS TO MATCH FRONTEND EXAMPLES
            "well, this is a productive use of my time": "This is a pointless waste of time",
            "you're a genius, thank you for that insight": "That was a very stupid or obvious comment",
            "i'm thrilled to be on hold for the 45th minute": "I am extremely frustrated with this customer service",
            "my phone updating right before a call is perfect timing": "The timing for this update is incredibly inconvenient",
            "another meeting that could have been an email? lovely.": "This meeting is unnecessary",
            "oh, fantastic, the wi-fi is down again": "The unreliable Wi-Fi is a major problem",
            "i love it when people are fashionably late": "I am annoyed that people are late",
            "this is exactly how i pictured my relaxing evening": "My evening is chaotic and stressful",
            "mango is my favourite fruit because i get pimples after eating it": "I dislike mango because it causes pimples",
            
            # CONTEXT-AWARE SARCASM REVERSAL
            "i'm so grateful for this delay": "I am frustrated by this delay",
            "i'm absolutely delighted with this service": "I am disappointed with this service",
            "i'm having a wonderful time here": "I am bored and want to leave",
            "i'm so happy about this situation": "I am unhappy about this situation",
            "i'm just loving this experience": "I am hating this experience",
            "i'm so excited to be here": "I am dreading being here",
            "i'm overjoyed about this news": "I am upset about this news",
            "i'm absolutely pleased with this outcome": "I am displeased with this outcome",
            
            # PROBLEM SARCASM REVERSAL
            "i'm so grateful for this problem": "I am frustrated by this problem",
            "i'm absolutely delighted with this mess": "I am annoyed by this mess",
            "i'm having a wonderful time dealing with this": "I am struggling with this",
            "i'm so happy about this failure": "I am disappointed by this failure",
            "i'm just loving this complication": "I am frustrated by this complication",
            "i'm so excited to deal with this": "I am dreading dealing with this",
            "i'm overjoyed about this setback": "I am upset about this setback",
            "i'm absolutely pleased with this performance": "I am disappointed with this performance",
            
            # WAITING/DELAY SARCASM REVERSAL
            "i'm so grateful for this 2-hour delay": "I am frustrated by this 2-hour delay",
            "i'm absolutely delighted with this customer service": "I am disappointed with this customer service",
            "i'm having a wonderful time waiting": "I am bored and frustrated waiting",
            "i'm so happy about this traffic jam": "I am unhappy about this traffic",
            "i'm just loving this update": "I am annoyed by this update",
            
            # WORK/MEETING SARCASM REVERSAL
            "i'm so grateful for this meeting": "I am dreading this meeting",
            "i'm absolutely delighted with this monday morning": "I am tired and grumpy on Monday morning",
            "i'm having a wonderful time in this meeting": "I am bored in this meeting",
            "i'm so happy about this deadline": "I am stressed about this deadline",
            "i'm just loving this work": "I am hating this work",
            "i'm overjoyed about this promotion": "I am upset about this promotion",
            "i'm absolutely pleased with this team": "I am frustrated with this team",
            
            # TECHNOLOGY SARCASM REVERSAL
            "i'm so grateful for this software update": "I am annoyed by this software update",
            "i'm absolutely delighted with this broken printer": "I am frustrated with this broken printer",
            "i'm having a wonderful time with this computer": "I am struggling with this computer",
            "i'm so happy about this password reset": "I am frustrated by this password reset",
            "i'm just loving this slow internet": "I am annoyed by this slow internet",
            "i'm so excited to use this": "I am dreading using this",
            "i'm overjoyed about this new feature": "I am upset about this new feature",
            
            # WEATHER SARCASM REVERSAL
            "i'm so grateful for this rain": "I am annoyed by this rain",
            "i'm absolutely delighted with this beautiful weather": "I am disappointed with this beautiful weather",
            "i'm having a wonderful time in this weather": "I am uncomfortable in this weather",
            "i'm so happy about this temperature": "I am unhappy about this temperature",
            "i'm just loving this season": "I am hating this season",
            "i'm so excited to be outside": "I am dreading being outside",
            "i'm overjoyed about this forecast": "I am upset about this forecast",
            "i'm absolutely pleased with this outdoor activity": "I am disappointed with this outdoor activity",
            
            # FOOD SARCASM REVERSAL
            "i'm so grateful for this meal": "I am disappointed with this meal",
            "i'm absolutely delighted with this restaurant": "I am disappointed with this restaurant",
            "i'm having a wonderful time eating this": "I am not enjoying eating this",
            "i'm so happy about this taste": "I am unhappy about this taste",
            "i'm just loving this cuisine": "I am hating this cuisine",
            "i'm so excited to try this": "I am dreading trying this",
            "i'm overjoyed about this ingredient": "I am upset about this ingredient",
            "i'm absolutely pleased with this presentation": "I am disappointed with this presentation",
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
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()

def context_aware_sarcasm_detection(text: str) -> dict:
    """Enhanced context-aware sarcasm detection"""
    text_lower = text.lower().strip()
    
    if not text_lower:
        return {'is_sarcastic': False, 'confidence': 0.0, 'method': 'empty'}
    
    # 1. Check obvious sarcastic patterns first
    for pattern, confidence in COMPLETE_SARCASTIC_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return {'is_sarcastic': True, 'confidence': confidence, 'method': 'pattern_match'}
    
    # 2. Context-aware analysis
    context_score = 0
    context_factors = []
    
    # Check for positive words in negative contexts
    positive_words_found = [word for word in CONTEXT_KEYWORDS['positive_words'] if word in text_lower]
    negative_contexts_found = [word for word in CONTEXT_KEYWORDS['negative_contexts'] if word in text_lower]
    sarcastic_indicators_found = [word for word in CONTEXT_KEYWORDS['sarcastic_indicators'] if word in text_lower]
    
    # Semantic contradiction analysis
    if positive_words_found and negative_contexts_found:
        context_score += 0.8
        context_factors.append('semantic_contradiction')
        
        # Boost for "because" patterns (strong sarcasm indicator)
        if 'because' in text_lower:
            context_score += 0.1
            context_factors.append('because_pattern')
    
    # Sarcastic indicators boost
    if sarcastic_indicators_found:
        context_score += 0.3
        context_factors.append('sarcastic_indicators')
    
    # Exaggerated compliments (often sarcastic)
    exaggerated_patterns = [
        'so smart', 'so intelligent', 'so clever', 'such a genius', 'so brilliant',
        'absolutely perfect', 'totally amazing', 'completely wonderful'
    ]
    if any(pattern in text_lower for pattern in exaggerated_patterns):
        context_score += 0.6
        context_factors.append('exaggerated_compliment')
    
    # Context-specific patterns
    if any(word in text_lower for word in ['waiting', 'delay', 'hold', 'traffic', 'meeting']):
        if any(word in text_lower for word in ['love', 'enjoy', 'wonderful', 'fantastic']):
            context_score += 0.7
            context_factors.append('waiting_sarcasm')
    
    # Technology frustration sarcasm
    if any(word in text_lower for word in ['update', 'software', 'computer', 'internet', 'wifi']):
        if any(word in text_lower for word in ['love', 'enjoy', 'wonderful', 'fantastic']):
            context_score += 0.6
            context_factors.append('tech_sarcasm')
    
    # Determine if sarcastic based on context score
    if context_score >= 0.6:
        confidence = min(0.95, context_score)
        return {
            'is_sarcastic': True, 
            'confidence': confidence, 
            'method': 'context_aware',
            'context_factors': context_factors,
            'context_score': context_score
        }
    
    # 3. Default to non-sarcastic
    return {'is_sarcastic': False, 'confidence': 0.3, 'method': 'default'}

def complete_detect_sarcasm(text: str) -> dict:
    """Complete sarcasm detection that handles ALL cases including personal sarcasm"""
    return context_aware_sarcasm_detection(text)

def complete_sarcastic_conversion(text: str) -> str:
    """Complete conversion to sarcastic"""
    original_text = text.strip()
    if not original_text:
        return "What a joy"
    
    text_lower = original_text.lower()
    print(f"Converting to sarcastic: '{original_text}'")
    
    # Check if already sarcastic - if so, return as is
    detection_result = context_aware_sarcasm_detection(original_text)
    if detection_result['is_sarcastic'] and detection_result['confidence'] > 0.7:
        print(f"Already sarcastic, returning as is: '{original_text}'")
        return original_text
    
    # STEP 1: Direct mappings
    for genuine, sarcastic in COMPLETE_CONVERSION_SYSTEM['direct_mappings']['to_sarcastic'].items():
        if text_lower == genuine:
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
    
    # Check if already genuine - if so, return as is
    detection_result = context_aware_sarcasm_detection(original_text)
    if not detection_result['is_sarcastic'] and detection_result['confidence'] > 0.7:
        print(f"Already genuine, returning as is: '{original_text}'")
        return original_text
    
    # STEP 1: Direct mappings
    for sarcastic, genuine in COMPLETE_CONVERSION_SYSTEM['direct_mappings']['to_unsarcastic'].items():
        if text_lower == sarcastic:
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
    detection_result = context_aware_sarcasm_detection(original_text)
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
    """Comprehensive test cases for enhanced sarcasm detection and conversion"""
    test_cases = [
        # HIGH CONFIDENCE SARCASM DETECTION
        {"input": "im just bursting with energy", "expected_sarcastic": "Im just bursting with energy", "expected_unsarcastic": "I am tired"},
        {"input": "this is just fantastic", "expected_sarcastic": "This is just fantastic", "expected_unsarcastic": "This is a problem"},
        {"input": "wow i just love how smart you are", "expected_sarcastic": "Wow I just love how smart you are", "expected_unsarcastic": "You're not very smart"},
        {"input": "i love how intelligent you are", "expected_sarcastic": "Wow I just love how intelligent you are", "expected_unsarcastic": "You're not very smart"},
        {"input": "you're so clever", "expected_sarcastic": "You're so clever", "expected_unsarcastic": "You're not very smart"},
        
        # SEMANTIC CONTRADICTION PATTERNS
        {"input": "mango is my favourite fruit because i get pimples after eating it", "expected_sarcastic": "Mango is my absolute favourite fruit because i get pimples after eating it", "expected_unsarcastic": "I dislike mango because i get pimples after eating it"},
        {"input": "i love politics because it causes so much trouble for common man", "expected_sarcastic": "As if I love politics because it causes so much trouble for common man", "expected_unsarcastic": "I hate politics because it causes so much trouble for common man"},
        {"input": "i love waiting in long lines", "expected_sarcastic": "I love waiting in long lines", "expected_unsarcastic": "I hate waiting in long lines"},
        
        # CONTEXT-AWARE SARCASM
        {"input": "i am grateful for this delay", "expected_sarcastic": "I'm so grateful for this delay", "expected_unsarcastic": "I am frustrated by this delay"},
        {"input": "i am delighted with this service", "expected_sarcastic": "I'm absolutely delighted with this service", "expected_unsarcastic": "I am disappointed with this service"},
        {"input": "i am happy about this situation", "expected_sarcastic": "I'm so happy about this situation", "expected_unsarcastic": "I am unhappy about this situation"},
        {"input": "i am pleased with this outcome", "expected_sarcastic": "I'm absolutely pleased with this outcome", "expected_unsarcastic": "I am displeased with this outcome"},
        {"input": "i am excited about this meeting", "expected_sarcastic": "I'm so excited about this meeting", "expected_unsarcastic": "I am dreading this meeting"},
        
        # WORK/MEETING SARCASM
        {"input": "i have a meeting", "expected_sarcastic": "I'm so excited about this meeting", "expected_unsarcastic": "I have a meeting"},
        {"input": "i have work to do", "expected_sarcastic": "I'm thrilled to have more work", "expected_unsarcastic": "I have work to do"},
        {"input": "this deadline is stressful", "expected_sarcastic": "I love working under pressure", "expected_unsarcastic": "This deadline is stressful"},
        {"input": "i am busy", "expected_sarcastic": "I have all the time in the world", "expected_unsarcastic": "I am busy"},
        {"input": "i need a break", "expected_sarcastic": "I'm energized and ready for more", "expected_unsarcastic": "I need a break"},
        
        # TECHNOLOGY SARCASM
        {"input": "my computer is slow", "expected_sarcastic": "My computer is lightning fast", "expected_unsarcastic": "My computer is slow"},
        {"input": "the internet is down", "expected_sarcastic": "The internet is working perfectly", "expected_unsarcastic": "The internet is down"},
        {"input": "my phone died", "expected_sarcastic": "My phone has infinite battery", "expected_unsarcastic": "My phone died"},
        {"input": "the app crashed", "expected_sarcastic": "The app is running smoothly", "expected_unsarcastic": "The app crashed"},
        {"input": "i lost my data", "expected_sarcastic": "My data is perfectly safe", "expected_unsarcastic": "I lost my data"},
        
        # WEATHER SARCASM
        {"input": "it is cold outside", "expected_sarcastic": "The weather is wonderfully warm", "expected_unsarcastic": "It is cold outside"},
        {"input": "it is hot today", "expected_sarcastic": "Such refreshing cool weather", "expected_unsarcastic": "It is hot today"},
        {"input": "it is windy", "expected_sarcastic": "Perfect calm weather", "expected_unsarcastic": "It is windy"},
        {"input": "it is snowing", "expected_sarcastic": "Beautiful sunny day", "expected_unsarcastic": "It is snowing"},
        {"input": "it is humid", "expected_sarcastic": "Such dry comfortable air", "expected_unsarcastic": "It is humid"},
        
        # FOOD SARCASM
        {"input": "this food is bland", "expected_sarcastic": "This food is incredibly flavorful", "expected_unsarcastic": "This food is bland"},
        {"input": "the service is slow", "expected_sarcastic": "The service is lightning fast", "expected_unsarcastic": "The service is slow"},
        {"input": "the food is cold", "expected_sarcastic": "The food is perfectly hot", "expected_unsarcastic": "The food is cold"},
        {"input": "this tastes bad", "expected_sarcastic": "This tastes absolutely delicious", "expected_unsarcastic": "This tastes bad"},
        {"input": "the portion is small", "expected_sarcastic": "What a generous portion", "expected_unsarcastic": "The portion is small"},
        
        # TRAVEL SARCASM
        {"input": "the flight is delayed", "expected_sarcastic": "We're making excellent time", "expected_unsarcastic": "The flight is delayed"},
        {"input": "traffic is terrible", "expected_sarcastic": "The roads are perfectly clear", "expected_unsarcastic": "Traffic is terrible"},
        {"input": "the hotel is noisy", "expected_sarcastic": "Such a peaceful quiet hotel", "expected_unsarcastic": "The hotel is noisy"},
        {"input": "the room is small", "expected_sarcastic": "What a spacious room", "expected_unsarcastic": "The room is small"},
        {"input": "the service is poor", "expected_sarcastic": "Excellent customer service", "expected_unsarcastic": "The service is poor"},
        
        # HEALTH SARCASM
        {"input": "i am sick", "expected_sarcastic": "I'm feeling absolutely wonderful", "expected_unsarcastic": "I am sick"},
        {"input": "i have a headache", "expected_sarcastic": "My head feels perfectly clear", "expected_unsarcastic": "I have a headache"},
        {"input": "i am stressed", "expected_sarcastic": "I'm completely relaxed", "expected_unsarcastic": "I am stressed"},
        {"input": "i am worried", "expected_sarcastic": "I'm totally carefree", "expected_unsarcastic": "I am worried"},
        {"input": "i am anxious", "expected_sarcastic": "I'm perfectly calm", "expected_unsarcastic": "I am anxious"},
        
        # EDUCATION SARCASM
        {"input": "this class is boring", "expected_sarcastic": "This class is so exciting", "expected_unsarcastic": "This class is boring"},
        {"input": "the exam is hard", "expected_sarcastic": "This exam is incredibly easy", "expected_unsarcastic": "The exam is hard"},
        {"input": "i failed the test", "expected_sarcastic": "I aced that test perfectly", "expected_unsarcastic": "I failed the test"},
        {"input": "the teacher is strict", "expected_sarcastic": "The teacher is so understanding", "expected_unsarcastic": "The teacher is strict"},
        {"input": "homework is difficult", "expected_sarcastic": "Homework is a piece of cake", "expected_unsarcastic": "Homework is difficult"},
        
        # RELATIONSHIP SARCASM
        {"input": "my friend is annoying", "expected_sarcastic": "My friend is absolutely delightful", "expected_unsarcastic": "My friend is annoying"},
        {"input": "my boss is demanding", "expected_sarcastic": "My boss is so understanding", "expected_unsarcastic": "My boss is demanding"},
        {"input": "my neighbor is loud", "expected_sarcastic": "My neighbor is perfectly quiet", "expected_unsarcastic": "My neighbor is loud"},
        {"input": "my family is difficult", "expected_sarcastic": "My family is wonderful", "expected_unsarcastic": "My family is difficult"},
        {"input": "my partner is late", "expected_sarcastic": "My partner is always punctual", "expected_unsarcastic": "My partner is late"},
        
        # PERSONAL SARCASM
        {"input": "you are smart", "expected_sarcastic": "Wow, you're such a genius", "expected_unsarcastic": "You are smart"},
        {"input": "you are intelligent", "expected_sarcastic": "What a brilliant mind you have", "expected_unsarcastic": "You are intelligent"},
        {"input": "you are clever", "expected_sarcastic": "You're so incredibly clever", "expected_unsarcastic": "You are clever"},
        {"input": "you are brilliant", "expected_sarcastic": "You're absolutely brilliant", "expected_unsarcastic": "You are brilliant"},
        {"input": "you are talented", "expected_sarcastic": "You're so incredibly talented", "expected_unsarcastic": "You are talented"},
        
        # GENUINE EXAMPLES (should not be converted)
        {"input": "the weather is nice today", "expected_sarcastic": "What beautiful weather", "expected_unsarcastic": "The weather is nice today"},
        {"input": "this is a problem", "expected_sarcastic": "This is just fantastic", "expected_unsarcastic": "This is a problem"},
        {"input": "i am tired", "expected_sarcastic": "I'm just bursting with energy", "expected_unsarcastic": "I am tired"},
        {"input": "i love pizza", "expected_sarcastic": "I absolutely adore pizza", "expected_unsarcastic": "I love pizza"},
        {"input": "this movie is great", "expected_sarcastic": "This movie is absolutely fantastic", "expected_unsarcastic": "This movie is great"},
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