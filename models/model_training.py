"""
Model Training Script for Sentiment Analysis
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import pickle
import os
import json
from typing import Tuple, Dict, Any
import re

class SentimentModelTrainer:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.pipeline = None
        self.feature_keywords = {
            'kanji': ['kanji', '漢字', 'karakter', 'tulisan', 'huruf'],
            'kotoba': ['kosakata', 'vocabulary', 'kata', 'word', 'kotoba', '語彙'],
            'bunpou': ['grammar', 'tata bahasa', 'bunpou', '文法', 'struktur']
        }
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for training"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_feature_from_text(self, text: str) -> str:
        """Extract feature category from text based on keywords"""
        text_lower = text.lower()
        
        # Count keyword matches for each feature
        feature_scores = {}
        for feature, keywords in self.feature_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            feature_scores[feature] = score
        
        # Return feature with highest score, default to 'general'
        if max(feature_scores.values()) > 0:
            return max(feature_scores.items(), key=lambda x: x[1])[0]
        else:
            return 'general'
    
    def create_synthetic_data(self) -> pd.DataFrame:
        """Create synthetic training data for demonstration"""
        positive_samples = [
            "aplikasi ini sangat bagus untuk belajar kanji",
            "kanji study sangat membantu dalam mengingat karakter",
            "fitur kanji sangat lengkap dan mudah dipahami",
            "kosakata dalam aplikasi ini sangat banyak",
            "vocabulary trainer sangat efektif",
            "kata-kata yang diajarkan sangat berguna",
            "grammar explanation sangat jelas",
            "tata bahasa dijelaskan dengan baik",
            "struktur kalimat mudah dipahami",
            "aplikasi terbaik untuk belajar bahasa jepang",
            "interface sangat user friendly",
            "fitur-fitur sangat membantu"
        ]
        
        negative_samples = [
            "aplikasi ini sulit digunakan",
            "kanji tidak lengkap",
            "kosakata terbatas",
            "grammar tidak jelas",
            "interface membingungkan",
            "aplikasi sering crash",
            "fitur tidak berfungsi dengan baik",
            "terlalu rumit untuk pemula",
            "tidak ada penjelasan yang memadai",
            "aplikasi lambat"
        ]
        
        # Create DataFrame
        data = []
        
        # Add positive samples
        for text in positive_samples:
            feature = self.extract_feature_from_text(text)
            data.append({
                'text': text,
                'sentiment': 'positive',
                'feature': feature
            })
        
        # Add negative samples
        for text in negative_samples:
            feature = self.extract_feature_from_text(text)
            data.append({
                'text': text,
                'sentiment': 'negative',
                'feature': feature
            })
        
        return pd.DataFrame(data)
    
    def train_model(self, algorithm: str = 'logistic_regression') -> Dict[str, Any]:
        """Train sentiment analysis model"""
        # Create or load training data
        df = self.create_synthetic_data()
        
        # Prepare data
        X = df['text'].apply(self.preprocess_text)
        y = df['sentiment']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Choose algorithm
        if algorithm == 'logistic_regression':
            classifier = LogisticRegression(random_state=42)
        elif algorithm == 'naive_bayes':
            classifier = MultinomialNB()
        else:
            raise ValueError("Algorithm must be 'logistic_regression' or 'naive_bayes'")
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
            ('classifier', classifier)
        ])
        
        # Train model
        self.pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.pipeline.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'algorithm': algorithm,
            'training_samples': len(df)
        }
        
        return results
    
    def save_model(self, model_path: str = 'models/sentiment_model.pkl'):
        """Save trained model"""
        if self.pipeline is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        joblib.dump(self.pipeline, model_path)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str = 'models/sentiment_model.pkl'):
        """Load trained model"""
        if os.path.exists(model_path):
            self.pipeline = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    def predict_sentiment(self, text: str) -> Dict[str, Any]:
        """Predict sentiment for given text"""
        if self.pipeline is None:
            raise ValueError("Model not loaded. Call load_model() or train_model() first.")
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Predict
        prediction = self.pipeline.predict([processed_text])[0]
        probabilities = self.pipeline.predict_proba([processed_text])[0]
        
        # Extract feature
        feature = self.extract_feature_from_text(text)
        
        # Get class labels
        classes = self.pipeline.classes_
        
        return {
            'text': text,
            'sentiment': prediction,
            'confidence': max(probabilities),
            'probabilities': dict(zip(classes, probabilities)),
            'feature': feature
        }

def train_and_save_model():
    """Main function to train and save model"""
    trainer = SentimentModelTrainer()
    
    print("Training sentiment analysis model...")
    results = trainer.train_model(algorithm='logistic_regression')
    
    print(f"Training completed!")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"Training samples: {results['training_samples']}")
    
    # Save model
    trainer.save_model()
    
    # Save training results
    with open('models/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Model and results saved successfully!")
    
    return trainer

if __name__ == "__main__":
    train_and_save_model()