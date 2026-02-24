import os
import pickle
import joblib
import urllib.request
import zipfile
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from preprocessor import preprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_init")

# Create models directory
MODELS_DIR = "models"
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

def download_sms():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    logger.info("Downloading SMS dataset...")
    if not os.path.exists("smsspamcollection.zip"):
        urllib.request.urlretrieve(url, "smsspamcollection.zip")
    with zipfile.ZipFile("smsspamcollection.zip", "r") as z:
        z.extractall("sms_data")
    df = pd.read_csv("sms_data/SMSSpamCollection", sep="\t",
                     header=None, names=["label", "text"],
                     encoding="latin-1")
    df["label"] = df["label"].map({"spam": 1, "ham": 0})
    return df

def train_all_models():
    logger.info("Initializing models for evaluation...")
    
    # Use SMS dataset (Enron link was 404 previously)
    df = download_sms()
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logger.info(f"Preprocessing {len(df)} records...")
    df["clean_text"] = df["text"].apply(preprocess)
    
    # Remove empty texts after cleaning
    df = df[df["clean_text"].str.strip() != ""].copy()
    
    logger.info("Training TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["label"].values
    
    # 1. Naive Bayes
    logger.info("Training Naive Bayes...")
    nb = MultinomialNB()
    nb.fit(X, y)
    
    # 2. Linear SVM
    logger.info("Training SVM...")
    svm = LinearSVC(random_state=42)
    svm.fit(X, y)
    
    # 3. Logistic Regression
    logger.info("Training Logistic Regression...")
    lr = LogisticRegression(random_state=42)
    lr.fit(X, y)
    
    logger.info("Saving all models to models/ directory...")
    # Save vectorizer
    with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    
    # Save models using joblib (as expected by the evaluation script)
    joblib.dump(nb, os.path.join(MODELS_DIR, "nb_model.pkl"))
    joblib.dump(svm, os.path.join(MODELS_DIR, "svm_model.pkl"))
    joblib.dump(lr, os.path.join(MODELS_DIR, "lr_model.pkl"))
    
    # Maintain root level copies for backward compatibility with app.py
    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open("nb_model.pkl", "wb") as f:
        pickle.dump(nb, f)
    
    logger.info("Initialization complete. All 3 models saved.")

if __name__ == "__main__":
    train_all_models()
