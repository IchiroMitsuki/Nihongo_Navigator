import pandas as pd
import json
import numpy as np
from typing import Dict, List, Any

class DataProcessor:
    """Class for processing sentiment analysis data"""
    
    def __init__(self):
        self.features = ['kanji', 'kotoba', 'bunpou']
        self.sentiment_labels = ['positive', 'negative']
    
    def load_app_data(self, file_path: str) -> Dict[str, Any]:
        """Load sentiment data from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return {}
        except json.JSONDecodeError:
            print(f"Invalid JSON format: {file_path}")
            return {}
    
    def calculate_sentiment_percentage(self, feature_data: Dict[str, int]) -> float:
        """Calculate positive sentiment percentage for a feature"""
        positive = feature_data.get('positive', 0)
        negative = feature_data.get('negative', 0)
        total = positive + negative
        
        if total == 0:
            return 0.0
        
        return (positive / total) * 100
    
    def aggregate_app_data(self, apps_data: Dict[str, Dict]) -> pd.DataFrame:
        """Aggregate all apps data into a comprehensive DataFrame"""
        aggregated_data = []
        
        for app_name, app_data in apps_data.items():
            for feature in self.features:
                if feature in app_data:
                    feature_data = app_data[feature]
                    positive = feature_data.get('positive', 0)
                    negative = feature_data.get('negative', 0)
                    total = positive + negative
                    percentage = self.calculate_sentiment_percentage(feature_data)
                    
                    aggregated_data.append({
                        'app_name': app_name,
                        'feature': feature,
                        'positive_count': positive,
                        'negative_count': negative,
                        'total_count': total,
                        'positive_percentage': percentage,
                        'negative_percentage': 100 - percentage if total > 0 else 0
                    })
        
        return pd.DataFrame(aggregated_data)
    
    def get_top_apps_by_feature(self, df: pd.DataFrame, feature: str, top_n: int = 5) -> pd.DataFrame:
        """Get top N apps for a specific feature based on positive percentage"""
        feature_df = df[df['feature'] == feature].copy()
        return feature_df.nlargest(top_n, 'positive_percentage')
    
    def get_feature_summary(self, apps_data: Dict[str, Dict]) -> Dict[str, Dict]:
        """Get summary statistics for each feature across all apps"""
        summary = {}
        
        for feature in self.features:
            feature_stats = {
                'total_reviews': 0,
                'total_positive': 0,
                'total_negative': 0,
                'avg_positive_percentage': 0,
                'apps_with_feature': 0
            }
            
            apps_with_feature = []
            
            for app_name, app_data in apps_data.items():
                if feature in app_data:
                    feature_data = app_data[feature]
                    positive = feature_data.get('positive', 0)
                    negative = feature_data.get('negative', 0)
                    
                    if positive > 0 or negative > 0:  # App has reviews for this feature
                        feature_stats['total_reviews'] += positive + negative
                        feature_stats['total_positive'] += positive
                        feature_stats['total_negative'] += negative
                        feature_stats['apps_with_feature'] += 1
                        
                        percentage = self.calculate_sentiment_percentage(feature_data)
                        apps_with_feature.append(percentage)
            
            # Calculate average positive percentage
            if apps_with_feature:
                feature_stats['avg_positive_percentage'] = np.mean(apps_with_feature)
            
            summary[feature] = feature_stats
        
        return summary
    
    def create_comparison_matrix(self, apps_data: Dict[str, Dict]) -> pd.DataFrame:
        """Create a matrix comparing all apps across all features"""
        matrix_data = []
        
        for app_name, app_data in apps_data.items():
            row = {'App': app_name}
            
            for feature in self.features:
                if feature in app_data:
                    percentage = self.calculate_sentiment_percentage(app_data[feature])
                    total = app_data[feature].get('positive', 0) + app_data[feature].get('negative', 0)
                    row[f'{feature}_percentage'] = percentage
                    row[f'{feature}_total'] = total
                else:
                    row[f'{feature}_percentage'] = 0
                    row[f'{feature}_total'] = 0
            
            # Calculate overall score (weighted average)
            total_reviews = sum(row[f'{feature}_total'] for feature in self.features)
            if total_reviews > 0:
                weighted_score = sum(
                    row[f'{feature}_percentage'] * row[f'{feature}_total'] 
                    for feature in self.features
                ) / total_reviews
                row['overall_score'] = weighted_score
            else:
                row['overall_score'] = 0
            
            row['total_reviews'] = total_reviews
            matrix_data.append(row)
        
        return pd.DataFrame(matrix_data)
    
    def filter_apps_by_criteria(self, df: pd.DataFrame, min_reviews: int = 0, 
                               min_positive_rate: float = 0) -> pd.DataFrame:
        """Filter apps based on minimum criteria"""
        filtered_df = df[
            (df['total_reviews'] >= min_reviews) & 
            (df['overall_score'] >= min_positive_rate)
        ].copy()
        
        return filtered_df.sort_values('overall_score', ascending=False)
    
    def export_processed_data(self, apps_data: Dict[str, Dict], output_path: str):
        """Export processed data to JSON file"""
        processed_data = {
            'apps_data': apps_data,
            'feature_summary': self.get_feature_summary(apps_data),
            'comparison_matrix': self.create_comparison_matrix(apps_data).to_dict('records'),
            'processing_timestamp': pd.Timestamp.now().isoformat()
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as file:
                json.dump(processed_data, file, indent=2, ensure_ascii=False)
            print(f"Processed data exported to: {output_path}")
        except Exception as e:
            print(f"Error exporting data: {e}")
    
    def validate_data_structure(self, app_data: Dict[str, Any]) -> bool:
        """Validate if app data has the correct structure"""
        if not isinstance(app_data, dict):
            return False
        
        for feature, feature_data in app_data.items():
            if feature not in self.features:
                continue
            
            if not isinstance(feature_data, dict):
                return False
            
            # Check if it has positive and negative keys
            if 'positive' not in feature_data or 'negative' not in feature_data:
                return False
            
            # Check if values are integers
            if not isinstance(feature_data['positive'], int) or not isinstance(feature_data['negative'], int):
                return False
        
        return True