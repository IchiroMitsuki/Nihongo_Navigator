"""
Utility functions for Japanese App Sentiment Analysis
"""
import json
import pandas as pd
import os
from typing import Dict, List, Any
import streamlit as st

def load_json_data(file_path: str) -> Dict:
    """Load JSON data from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"File tidak ditemukan: {file_path}")
        return {}
    except json.JSONDecodeError:
        st.error(f"Error decoding JSON: {file_path}")
        return {}

def save_json_data(data: Dict, file_path: str) -> bool:
    """Save data to JSON file"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving JSON: {e}")
        return False

def calculate_percentage(positive: int, negative: int) -> float:
    """Calculate positive percentage"""
    total = positive + negative
    if total == 0:
        return 0.0
    return round((positive / total) * 100, 2)

def get_app_display_name(app_name: str) -> str:
    """Convert app filename to display name"""
    app_names = {
        'mazii': 'Mazii',
        'obenkyo': 'Obenkyo',
        'heyjapan': 'HeyJapan',
        'jasensei': 'JaSensei',
        'migiijlpt': 'Migii JLPT',
        'kanjistudy': 'Kanji Study'
    }
    return app_names.get(app_name, app_name.title())

def get_feature_display_name(feature: str) -> str:
    """Convert feature key to display name"""
    feature_names = {
        'kanji': 'Kanji',
        'kotoba': 'Kosakata (Kotoba)',
        'bunpou': 'Tata Bahasa (Bunpou)'
    }
    return feature_names.get(feature, feature.title())

def format_number(num: int) -> str:
    """Format number with comma separator"""
    return f"{num:,}"

def create_feature_colors() -> Dict[str, str]:
    """Create color mapping for features"""
    return {
        'kanji': '#FF6B6B',
        'kotoba': '#4ECDC4', 
        'bunpou': '#45B7D1'
    }

def validate_file_exists(file_path: str) -> bool:
    """Check if file exists"""
    return os.path.exists(file_path)

def get_top_apps_by_feature(data: Dict, feature: str, limit: int = 3) -> List[Dict]:
    """Get top apps for a specific feature"""
    feature_data = []
    
    for app_name, app_data in data.items():
        if feature in app_data:
            positive = app_data[feature].get('positive', 0)
            negative = app_data[feature].get('negative', 0)
            percentage = calculate_percentage(positive, negative)
            
            feature_data.append({
                'app': get_app_display_name(app_name),
                'positive': positive,
                'negative': negative,
                'percentage': percentage,
                'total': positive + negative
            })
    
    # Sort by percentage, then by total count
    feature_data.sort(key=lambda x: (x['percentage'], x['total']), reverse=True)
    return feature_data[:limit]

def export_to_csv(data: pd.DataFrame, filename: str) -> bytes:
    """Export dataframe to CSV bytes"""
    return data.to_csv(index=False).encode('utf-8')

def clean_text(text: str) -> str:
    """Clean text for processing"""
    if not isinstance(text, str):
        return ""
    
    # Basic text cleaning
    text = text.strip()
    text = text.lower()
    return text

def get_summary_stats(data: Dict) -> Dict[str, Any]:
    """Calculate summary statistics"""
    total_positive = 0
    total_negative = 0
    total_reviews = 0
    app_count = len(data)
    
    for app_data in data.values():
        for feature_data in app_data.values():
            if isinstance(feature_data, dict):
                total_positive += feature_data.get('positive', 0)
                total_negative += feature_data.get('negative', 0)
    
    total_reviews = total_positive + total_negative
    positive_percentage = calculate_percentage(total_positive, total_negative)
    
    return {
        'total_apps': app_count,
        'total_reviews': total_reviews,
        'total_positive': total_positive,
        'total_negative': total_negative,
        'positive_percentage': positive_percentage
    }