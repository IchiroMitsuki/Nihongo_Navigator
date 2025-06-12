import unittest
import os
import sys
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ml_predictor import MLPredictor

class TestMLPredictor(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.ml_predictor = MLPredictor()
        
        # Mock model and vectorizer
        self.mock_model = Mock()
        self.mock_vectorizer = Mock()
        
        # Sample test texts
        self.test_texts = [
            "Aplikasi ini sangat bagus untuk belajar kanji",
            "Saya suka fitur kotoba di aplikasi ini",
            "Bunpou explanation sangat jelas",
            "Aplikasi ini buruk dan tidak berguna"
        ]
    
    def test_preprocess_text(self):
        """Test text preprocessing"""
        text = "Aplikasi ini SANGAT bagus!!! untuk belajar kanji 😊"
        result = self.ml_predictor.preprocess_text(text)
        
        # Should be lowercase and cleaned
        self.assertIsInstance(result, str)
        self.assertEqual(result.lower(), result)
        self.assertNotIn("!!!", result)
    
    def test_extract_features(self):
        """Test feature extraction from text"""
        text = "Saya suka belajar kanji dan kotoba di aplikasi ini"
        features = self.ml_predictor.extract_features(text)
        
        self.assertIsInstance(features, dict)
        self.assertIn("kanji", features)
        self.assertIn("kotoba", features)
        self.assertIn("bunpou", features)
        
        # Should detect kanji and kotoba
        self.assertTrue(features["kanji"])
        self.assertTrue(features["kotoba"])
        self.assertFalse(features["bunpou"])
    
    @patch('pickle.load')
    @patch('builtins.open')
    def test_load_model_success(self, mock_open, mock_pickle):
        """Test successful model loading"""
        mock_pickle.return_value = self.mock_model
        
        result = self.ml_predictor.load_model("test_model.pkl")
        
        self.assertEqual(result, self.mock_model)
        mock_open.assert_called_once()
    
    @patch('pickle.load')
    @patch('builtins.open')
    def test_load_model_failure(self, mock_open, mock_pickle):
        """Test model loading failure"""
        mock_open.side_effect = FileNotFoundError()
        
        result = self.ml_predictor.load_model("nonexistent_model.pkl")
        
        self.assertIsNone(result)
    
    def test_predict_sentiment_with_mock_model(self):
        """Test sentiment prediction with mock model"""
        # Setup mock model
        self.mock_model.predict.return_value = np.array([1])
        self.mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
        self.mock_vectorizer.transform.return_value = "vectorized_text"
        
        # Set the mock objects
        self.ml_predictor.model = self.mock_model
        self.ml_predictor.vectorizer = self.mock_vectorizer
        
        result = self.ml_predictor.predict_sentiment("Test text")
        
        self.assertIsInstance(result, dict)
        self.assertIn("sentiment", result)
        self.assertIn("confidence", result)
        self.assertIn("features", result)
    
    def test_predict_sentiment_without_model(self):
        """Test prediction without loaded model"""
        self.ml_predictor.model = None
        self.ml_predictor.vectorizer = None
        
        result = self.ml_predictor.predict_sentiment("Test text")
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["sentiment"], "positive")
        self.assertIsInstance(result["confidence"], float)
    
    def test_batch_predict(self):
        """Test batch prediction"""
        # Setup mock model
        self.mock_model.predict.return_value = np.array([1, 0, 1, 0])
        self.mock_model.predict_proba.return_value = np.array([
            [0.2, 0.8], [0.7, 0.3], [0.1, 0.9], [0.9, 0.1]
        ])
        self.mock_vectorizer.transform.return_value = "vectorized_texts"
        
        self.ml_predictor.model = self.mock_model
        self.ml_predictor.vectorizer = self.mock_vectorizer
        
        results = self.ml_predictor.batch_predict(self.test_texts)
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 4)
        
        for result in results:
            self.assertIn("sentiment", result)
            self.assertIn("confidence", result)
    
    def test_get_model_info(self):
        """Test getting model information"""
        info = self.ml_predictor.get_model_info()
        
        self.assertIsInstance(info, dict)
        self.assertIn("model_loaded", info)
        self.assertIn("vectorizer_loaded", info)
        self.assertIn("model_type", info)

if __name__ == '__main__':
    unittest.main()