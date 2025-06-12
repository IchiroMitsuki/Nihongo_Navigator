import unittest
import json
import os
import sys
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processor import DataProcessor

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.data_processor = DataProcessor()
        
        # Sample test data
        self.sample_data = {
            "mazii": {
                "kanji": {"positive": 57, "negative": 0},
                "kotoba": {"positive": 32, "negative": 0},
                "bunpou": {"positive": 26, "negative": 0}
            },
            "obenkyo": {
                "bunpou": {"positive": 5, "negative": 0},
                "kanji": {"positive": 29, "negative": 0},
                "kotoba": {"positive": 14, "negative": 0}
            }
        }
    
    def test_calculate_percentage(self):
        """Test percentage calculation"""
        result = self.data_processor.calculate_percentage(57, 0)
        self.assertEqual(result, 100.0)
        
        result = self.data_processor.calculate_percentage(30, 10)
        self.assertEqual(result, 75.0)
        
        result = self.data_processor.calculate_percentage(0, 0)
        self.assertEqual(result, 0.0)
    
    def test_process_app_data(self):
        """Test processing individual app data"""
        app_data = self.sample_data["mazii"]
        result = self.data_processor.process_app_data("mazii", app_data)
        
        self.assertEqual(result["app_name"], "mazii")
        self.assertEqual(result["kanji_positive"], 57)
        self.assertEqual(result["kanji_percentage"], 100.0)
        self.assertEqual(result["total_reviews"], 115)
    
    def test_create_comparison_table(self):
        """Test creating comparison table"""
        df = self.data_processor.create_comparison_table(self.sample_data)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertIn("app_name", df.columns)
        self.assertIn("kanji_percentage", df.columns)
    
    def test_get_top_apps_by_feature(self):
        """Test getting top apps by feature"""
        df = self.data_processor.create_comparison_table(self.sample_data)
        result = self.data_processor.get_top_apps_by_feature(df, "kanji")
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.iloc[0]["app_name"], "mazii")  # Highest kanji score
    
    def test_calculate_overall_stats(self):
        """Test overall statistics calculation"""
        stats = self.data_processor.calculate_overall_stats(self.sample_data)
        
        self.assertIn("total_apps", stats)
        self.assertIn("total_reviews", stats)
        self.assertIn("average_sentiment", stats)
        self.assertEqual(stats["total_apps"], 2)

if __name__ == '__main__':
    unittest.main()