import pandas as pd
import numpy as np
import pickle
import joblib
import re
import os
from typing import Dict, List, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import nltk
from textblob import TextBlob

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class MLPredictor:
    """Machine Learning predictor for sentiment analysis"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.feature_keywords = {
            'kanji': ['kanji', 'karakter', 'huruf', 'tulisan', 'menulis', 'stroke', 'radikal'],
            'kotoba': ['kosakata', 'vocab', 'kata', 'vocabulary', 'word', 'arti', 'meaning'],
            'bunpou': ['grammar', 'tata bahasa', 'bunpou', 'struktur', 'kalimat', 'pola']
        }
        self.model_path = 'models/sentiment_model.pkl'
        self.vectorizer_path = 'models/vectorizer.pkl'
        
        # Load existing model if available
        self.load_model()
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        return text
    
    def detect_features(self, text: str) -> List[str]:
        """Detect which features are mentioned in the text"""
        text_lower = text.lower()
        detected_features = []
        
        for feature, keywords in self.feature_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    detected_features.append(feature)
                    break
        
        return list(set(detected_features))  # Remove duplicates
    
    def create_training_data(self, apps_data: Dict[str, Dict]) -> Tuple[List[str], List[str], List[str]]:
        """Create synthetic training data from the aggregated results"""
        texts = []
        labels = []
        features = []
        
        # Sample positive review templates
        positive_templates = {
            'kanji': [
                "aplikasi ini sangat membantu untuk belajar kanji",
                "kanji mudah dipahami dengan aplikasi ini",
                "bagus untuk latihan menulis kanji",
                "stroke order kanji jelas dan mudah diikuti",
                "kanji dijelaskan dengan baik"
            ],
            'kotoba': [
                "kosakata sangat lengkap dan mudah dipahami",
                "vocab bahasa jepang jadi mudah diingat",
                "kata-kata dijelaskan dengan contoh yang baik",
                "arti kata jelas dan mudah dipahami",
                "vocabulary sangat membantu untuk belajar"
            ],
            'bunpou': [
                "grammar dijelaskan dengan sangat baik",
                "tata bahasa mudah dipahami",
                "struktur kalimat dijelaskan dengan jelas",
                "pola kalimat mudah diingat",
                "bunpou sangat membantu untuk pemula"
            ]
        }
        
        # Sample negative review templates
        negative_templates = {
            'kanji': [
                "kanji sulit dipahami dalam aplikasi ini",
                "stroke order kanji membingungkan",
                "aplikasi kanji tidak user friendly"
            ],
            'kotoba': [
                "kosakata kurang lengkap",
                "arti kata tidak jelas",
                "vocabulary terbatas"
            ],
            'bunpou': [
                "grammar sulit dipahami",
                "tata bahasa membingungkan",
                "struktur kalimat tidak jelas"
            ]
        }
        
        # Generate synthetic data based on aggregated results
        for app_name, app_data in apps_data.items():
            for feature, sentiment_data in app_data.items():
                positive_count = sentiment_data.get('positive', 0)
                negative_count = sentiment_data.get('negative', 0)
                
                # Generate positive samples
                for _ in range(min(positive_count, 50)):  # Limit to prevent overfitting
                    template = np.random.choice(positive_templates[feature])
                    texts.append(template)
                    labels.append('positive')
                    features.append(feature)
                
                # Generate negative samples
                for _ in range(min(negative_count, 50)):
                    template = np.random.choice(negative_templates[feature])
                    texts.append(template)
                    labels.append('negative')
                    features.append(feature)
        
        return texts, labels, features
    
    def train_model(self, apps_data: Dict[str, Dict]) -> Dict[str, Any]:
        """Train sentiment analysis model"""
        print("Creating training data...")
        texts, labels, features = self.create_training_data(apps_data)
        
        if len(texts) == 0:
            print("No training data available")
            return {'accuracy': 0, 'message': 'No training data'}
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Create pipeline
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        self.model = LogisticRegression(random_state=42)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            processed_texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Train vectorizer and model
        print("Training model...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        self.model.fit(X_train_vec, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained with accuracy: {accuracy:.3f}")
        
        # Save model
        self.save_model()
        
        return {
            'accuracy': accuracy,
            'training_samples': len(texts),
            'test_samples': len(X_test),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    def predict_sentiment(self, text: str) -> Dict[str, Any]:
        """Predict sentiment of input text"""
        if self.model is None or self.vectorizer is None:
            # Return mock prediction if model not trained
            return self._mock_prediction(text)
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Vectorize
        text_vec = self.vectorizer.transform([processed_text])
        
        # Predict
        prediction = self.model.predict(text_vec)[0]
        prediction_proba = self.model.predict_proba(text_vec)[0]
        
        # Get confidence
        confidence = max(prediction_proba) * 100
        
        # Detect features
        detected_features = self.detect_features(text)
        
        return {
            'sentiment': prediction,
            'confidence': confidence,
            'features': detected_features,
            'text_length': len(text),
            'processed_text': processed_text
        }
    
    def _mock_prediction(self, text: str) -> Dict[str, Any]:
        """Create mock prediction when model is not available"""
        # Simple rule-based prediction for demo purposes
        positive_words = ['bagus', 'baik', 'membantu', 'mudah', 'jelas', 'lengkap', 'useful', 'good', 'great', 'helpful']
        negative_words = ['buruk', 'jelek', 'sulit', 'susah', 'membingungkan', 'tidak', 'bad', 'poor', 'difficult']
        
        text_lower = text.lower()
        
        positive_score = sum(1 for word in positive_words if word in text_lower)
        negative_score = sum(1 for word in negative_words if word in text_lower)
        
        if positive_score > negative_score:
            sentiment = 'positive'
            confidence = min(60 + positive_score * 10, 95)
        elif negative_score > positive_score:
            sentiment = 'negative'
            confidence = min(60 + negative_score * 10, 95)
        else:
            sentiment = 'positive'  # Default to positive if neutral
            confidence = 50
        
        # Detect features
        detected_features = self.detect_features(text)
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'features': detected_features,
            'text_length': len(text),
            'processed_text': self.preprocess_text(text),
            'method': 'rule_based'
        }
    
    def save_model(self):
        """Save trained model and vectorizer"""
        os.makedirs('models', exist_ok=True)
        
        if self.model is not None:
            try:
                joblib.dump(self.model, self.model_path)
                print(f"Model saved to {self.model_path}")
            except Exception as e:
                print(f"Error saving model: {e}")
        
        if self.vectorizer is not None:
            try:
                joblib.dump(self.vectorizer, self.vectorizer_path)
                print(f"Vectorizer saved to {self.vectorizer_path}")
            except Exception as e:
                print(f"Error saving vectorizer: {e}")
    
    def load_model(self):
        """Load trained model and vectorizer"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                print("Model loaded successfully")
            
            if os.path.exists(self.vectorizer_path):
                self.vectorizer = joblib.load(self.vectorizer_path)
                print("Vectorizer loaded successfully")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.vectorizer = None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        info = {
            'model_loaded': self.model is not None,
            'vectorizer_loaded': self.vectorizer is not None,
            'model_type': type(self.model).__name__ if self.model else 'None',
            'model_path': self.model_path,
            'vectorizer_path': self.vectorizer_path
        }
        
        if self.vectorizer is not None:
            try:
                info['vocabulary_size'] = len(self.vectorizer.vocabulary_)
                info['feature_names_count'] = len(self.vectorizer.get_feature_names_out())
            except:
                pass
        
        return info
    
    def batch_predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict sentiment for multiple texts"""
        predictions = []
        
        for text in texts:
            prediction = self.predict_sentiment(text)
            predictions.append(prediction)
        
        return predictions
    
    def analyze_feature_sentiment(self, texts: List[str], target_feature: str) -> Dict[str, Any]:
        """Analyze sentiment specifically for a target feature"""
        feature_texts = []
        
        for text in texts:
            detected_features = self.detect_features(text)
            if target_feature in detected_features:
                feature_texts.append(text)
        
        if not feature_texts:
            return {
                'feature': target_feature,
                'total_mentions': 0,
                'average_sentiment': 'neutral',
                'confidence': 0
            }
        
        predictions = self.batch_predict(feature_texts)
        
        positive_count = sum(1 for p in predictions if p['sentiment'] == 'positive')
        total_count = len(predictions)
        
        avg_confidence = sum(p['confidence'] for p in predictions) / total_count
        avg_sentiment = 'positive' if positive_count / total_count > 0.5 else 'negative'
        
        return {
            'feature': target_feature,
            'total_mentions': total_count,
            'positive_count': positive_count,
            'negative_count': total_count - positive_count,
            'positive_percentage': (positive_count / total_count) * 100,
            'average_sentiment': avg_sentiment,
            'average_confidence': avg_confidence,
            'sample_texts': feature_texts[:5]  # Show first 5 samples
        }