import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Import Torch đầu tiên để tránh lỗi xung đột DLL (WinError 1114) trên Windows
import torch
import warnings
warnings.filterwarnings("ignore")

import re
import string
import time
import joblib
import json
import pickle
import numpy as np

# NLTK
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Keras & Transformers
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertForSequenceClassification

# Tải trước dữ liệu NLTK
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

MODEL_DIR = "models"
ENGLISH_STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

def format_result(prediction, probability, preprocessed_text, processing_time):
    # Quy ước: 0 là Tin thật, 1 là Tin giả
    label = "Tin thật (True News)" if prediction == 0 else "Tin giả (Fake News)"
    return {
        "success": True,
        "label": label,
        "probability": round(probability * 100, 2),
        "preprocessed_text": preprocessed_text,
        "processing_time": round(processing_time, 4)
    }

class SVMPredictor:
    def __init__(self):
        self.model_path = os.path.join(MODEL_DIR, "svm_tfidf_fake_news_model.pkl")
        self.vec_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
        self.model = None
        self.vectorizer = None

    def load(self):
        if os.path.exists(self.model_path) and os.path.exists(self.vec_path):
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vec_path)
            return True
        return False

    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
        text = re.sub(r"\d+", " ", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text).strip()
        words = text.split()
        words = [w for w in words if w not in ENGLISH_STOPWORDS]
        words = [LEMMATIZER.lemmatize(w) for w in words]
        return " ".join(words)

    def predict(self, title, text):
        start_time = time.time()
        raw_text = f"{title} {text}"
        cleaned = self.clean_text(raw_text)
        
        vectorized = self.vectorizer.transform([cleaned])
        prediction = self.model.predict(vectorized)[0]
        try:
            score = self.model.decision_function(vectorized)[0]
            prob = 1 / (1 + np.exp(-score)) if prediction == 1 else 1 - (1 / (1 + np.exp(-score)))
        except:
            prob = 1.0 

        processing_time = time.time() - start_time
        return format_result(prediction, prob, cleaned, processing_time)

class BiLSTMPredictor:
    def __init__(self):
        self.model_path = os.path.join(MODEL_DIR, "bilstm_word2vec_fake_news_model.keras")
        self.tok_path = os.path.join(MODEL_DIR, "bilstm_tokenizer.pkl")
        self.config_path = os.path.join(MODEL_DIR, "bilstm_word2vec_config.json")
        self.model = None
        self.tokenizer = None
        self.max_len = 300 

    def load(self):
        if os.path.exists(self.model_path) and os.path.exists(self.tok_path):
            self.model = load_model(self.model_path)
            with open(self.tok_path, "rb") as f:
                self.tokenizer = pickle.load(f)
            if os.path.exists(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    self.max_len = config.get("MAX_LEN", 300)
            return True
        return False

    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
        text = re.sub(r"\d+", " ", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text).strip()
        words = text.split()
        words = [w for w in words if w not in ENGLISH_STOPWORDS]
        return " ".join(words)

    def predict(self, title, text):
        start_time = time.time()
        raw_text = f"{title} {text}"
        cleaned = self.clean_text(raw_text)
        
        sequence = self.tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequence, maxlen=self.max_len, padding="post", truncating="post")
        
        probability = self.model.predict(padded, verbose=0)[0][0]
        prediction = 1 if probability >= 0.5 else 0
        prob_val = probability if prediction == 1 else 1 - probability
        
        processing_time = time.time() - start_time
        return format_result(prediction, prob_val, cleaned, processing_time)

class BERTPredictor:
    def __init__(self):
        self.model_dir = os.path.join(MODEL_DIR, "best_bert_fake_news_model")
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = 256

    def load(self):
        if os.path.exists(self.model_dir):
            self.tokenizer = BertTokenizer.from_pretrained(self.model_dir)
            self.model = BertForSequenceClassification.from_pretrained(self.model_dir)
            self.model.to(self.device)
            self.model.eval()
            return True
        return False

    def clean_text(self, text):
        text = str(text)
        text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def predict(self, title, text):
        start_time = time.time()
        raw_text = f"{title} {text}"
        cleaned = self.clean_text(raw_text)
        
        encoding = self.tokenizer(
            cleaned,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
        prediction = int(np.argmax(probs))
        probability = float(probs[prediction])
        
        processing_time = time.time() - start_time
        return format_result(prediction, probability, cleaned, processing_time)
