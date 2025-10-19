
#train_model.py
# # train_model.py
# import os
# import requests
# import json
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import make_pipeline
# from sklearn.metrics import classification_report, accuracy_score
# import joblib

# # Raw file URL (GitHub repo raw JSON lines)
# RAW_URL = "https://raw.githubusercontent.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection/master/Sarcasm_Headlines_Dataset.json"

# def download_and_parse(url=RAW_URL):
#     print("Downloading dataset...")
#     r = requests.get(url)
#     r.raise_for_status()
#     lines = r.text.strip().splitlines()
#     records = [json.loads(line) for line in lines]
#     df = pd.DataFrame(records)
#     # columns: is_sarcastic (0/1), headline, article_link
#     df = df[['is_sarcastic', 'headline']]
#     df.rename(columns={'is_sarcastic': 'label', 'headline': 'text'}, inplace=True)
#     print(f"Loaded {len(df)} records.")
#     return df

# def train_and_save(df, model_path="model.pkl", vec_path="vectorizer.pkl"):
#     X = df['text'].values
#     y = df['label'].values
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
#     vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words='english')
#     clf = LogisticRegression(max_iter=1000)
#     print("Fitting vectorizer...")
#     X_train_tf = vectorizer.fit_transform(X_train)
#     print("Training classifier...")
#     clf.fit(X_train_tf, y_train)
#     # Evaluate
#     X_test_tf = vectorizer.transform(X_test)
#     preds = clf.predict(X_test_tf)
#     print("Accuracy:", accuracy_score(y_test, preds))
#     print(classification_report(y_test, preds))
#     # Save
#     joblib.dump(clf, model_path)
#     joblib.dump(vectorizer, vec_path)
#     print(f"Saved model -> {model_path}, vectorizer -> {vec_path}")

# if __name__ == "__main__":
#     df = download_and_parse()
#     train_and_save(df)


# train_model.py (Improved Accuracy)
import os
import requests
import json
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
import re
import string

RAW_URL = "https://raw.githubusercontent.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection/master/Sarcasm_Headlines_Dataset.json"

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove links
    text = re.sub(r"[^a-z\s]", "", text)  # remove non-alphabet
    text = re.sub(r"\s+", " ", text).strip()
    return text

def download_and_parse(url=RAW_URL):
    print("Downloading dataset...")
    r = requests.get(url)
    r.raise_for_status()
    lines = r.text.strip().splitlines()
    records = [json.loads(line) for line in lines]
    df = pd.DataFrame(records)[['is_sarcastic', 'headline']]
    df.rename(columns={'is_sarcastic': 'label', 'headline': 'text'}, inplace=True)
    df['text'] = df['text'].apply(clean_text)
    print(f"Loaded {len(df)} samples.")
    return df

def train_and_save(df, model_path="model.pkl", vec_path="vectorizer.pkl"):
    X = df['text'].values
    y = df['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=25000,
        ngram_range=(1, 3),
        stop_words="english",
        sublinear_tf=True
    )

    print("Training model with hyperparameter tuning...")
    base_lr = LogisticRegression(solver='liblinear', class_weight='balanced')
    params = {'C': [0.5, 1.0, 2.0]}
    grid_lr = GridSearchCV(base_lr, params, cv=3, scoring='accuracy', n_jobs=-1)
    X_train_tf = vectorizer.fit_transform(X_train)
    grid_lr.fit(X_train_tf, y_train)
    best_lr = grid_lr.best_estimator_

    # Add small ensemble with RandomForest for stability
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    ensemble = VotingClassifier(estimators=[('lr', best_lr), ('rf', rf)], voting='soft')
    ensemble.fit(X_train_tf, y_train)

    # Evaluate
    X_test_tf = vectorizer.transform(X_test)
    preds = ensemble.predict(X_test_tf)
    acc = accuracy_score(y_test, preds)
    print(f"✅ Ensemble Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

    # Save
    joblib.dump(ensemble, model_path)
    joblib.dump(vectorizer, vec_path)
    print(f"Model saved -> {model_path}, Vectorizer -> {vec_path}")

if __name__ == "__main__":
    df = download_and_parse()
    train_and_save(df)
