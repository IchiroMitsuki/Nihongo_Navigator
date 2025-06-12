import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import streamlit as st

class Visualizer:
    """Class for creating visualizations for sentiment analysis data"""
    
    def __init__(self):
        self.color_palette = {
            'positive': '#2E8B57',  # Sea Green
            'negative': '#DC143C',  # Crimson
            'neutral': '#4682B4',   # Steel Blue
            'primary': '#FF6B6B',   # Light Red
            'secondary': '#4ECDC4'  # Turquoise
        }
        
        self.features = ['kanji', 'kotoba', 'bunpou']
        self.feature_colors = {
            'kanji': '#FF6B6B',
            'kotoba': '#4ECDC4', 
            'bunpou': '#45B7D1'
        }
    
    def create_sentiment_bar_chart(self, df: pd.DataFrame, feature: str = None) -> go.Figure:
        """Create bar chart showing sentiment distribution"""
        if feature:
            df_filtered = df[df['feature'] == feature]
            title = f'Distribusi Sentimen - {feature.capitalize()}'
        else:
            df_filtered = df
            title = 'Distribusi Sentimen - Semua Fitur'
        
        fig = px.bar(
            df_filtered,
            x='app_name',
            y=['positive_count', 'negative_count'],
            title=title,
            labels={'value': 'Jumlah Ulasan', 'variable': 'Sentimen'},
            color_discrete_map={
                'positive_count': self.color_palette['positive'],
                'negative_count': self.color_palette['negative']
            }
        )
        
        fig.update_layout(
            xaxis_title='Aplikasi',
            yaxis_title='Jumlah Ulasan',
            legend_title='Sentimen',
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    def create_percentage_comparison_chart(self, apps_data: Dict[str, Dict]) -> go.Figure:
        """Create horizontal bar chart comparing positive percentages"""
        app_names = []
        kanji_percentages = []
        kotoba_percentages = []
        bunpou_percentages = []
        
        for app_name, app_data in apps_data.items():
            app_names.append(app_name)
            
            for feature, percentages in zip(['kanji', 'kotoba', 'bunpou'], 
                                          [kanji_percentages, kotoba_percentages, bunpou_percentages]):
                if feature in app_data:
                    pos = app_data[feature].get('positive', 0)
                    neg = app_data[feature].get('negative', 0)
                    total = pos + neg
                    percentage = (pos / total * 100) if total > 0 else 0
                    percentages.append(percentage)
                else:
                    percentages.append(0)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Kanji',
            y=app_names,
            x=kanji_percentages,
            orientation='h',
            marker_color=self.feature_colors['kanji'],
            hovertemplate='<b>%{y}</b><br>Kanji: %{x:.1f}%<extra></extra>'
        ))
        
        fig.add_trace(go.Bar(
            name='Kotoba',
            y=app_names,
            x=kotoba_percentages,
            orientation='h',
            marker_color=self.feature_colors['kotoba'],
            hovertemplate='<b>%{y}</b><br>Kotoba: %{x:.1f}%<extra></extra>'
        ))
        
        fig.add_trace(go.Bar(
            name='Bunpou',
            y=app_names,
            x=bunpou_percentages,
            orientation='h',
            marker_color=self.feature_colors['bunpou'],
            hovertemplate='<b>%{y}</b><br>Bunpou: %{x:.1f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title='Persentase Sentimen Positif per Fitur',
            xaxis_title='Persentase Positif (%)',
            yaxis_title='Aplikasi',
            barmode='group',
            height=600,
            hovermode='closest'
        )
        
        return fig
    
    def create_radar_chart(self, apps_data: Dict[str, Dict], selected_apps: List[str] = None) -> go.Figure:
        """Create radar chart comparing apps across features"""
        if selected_apps is None:
            selected_apps = list(apps_data.keys())[:3]  # Show top 3 apps
        
        fig = go.Figure()
        
        features = ['Kanji', 'Kotoba', 'Bunpou']
        
        for app_name in selected_apps:
            if app_name in apps_data:
                app_data = apps_data[app_name]
                values = []
                
                for feature in ['kanji', 'kotoba', 'bunpou']:
                    if feature in app_data:
                        pos = app_data[feature].get('positive', 0)
                        neg = app_data[feature].get('negative', 0)
                        total = pos + neg
                        percentage = (pos / total * 100) if total > 0 else 0
                        values.append(percentage)
                    else:
                        values.append(0)
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=features,
                    fill='toself',
                    name=app_name,
                    hovertemplate='<b>%{theta}</b><br>%{r:.1f}%<extra></extra>'
                ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title="Perbandingan Radar - Sentimen Positif (%)",
            height=500
        )
        
        return fig
    
    def create_heatmap(self, apps_data: Dict[str, Dict]) -> go.Figure:
        """Create heatmap showing app performance across features"""
        app_names = list(apps_data.keys())
        features = ['Kanji', 'Kotoba', 'Bunpou']
        
        # Create matrix
        z_matrix = []
        hover_text = []
        
        for app_name in app_names:
            app_data = apps_data[app_name]
            row = []
            hover_row = []
            
            for feature in ['kanji', 'kotoba', 'bunpou']:
                if feature in app_data:
                    pos = app_data[feature].get('positive', 0)
                    neg = app_data[feature].get('negative', 0)
                    total = pos + neg
                    percentage = (pos / total * 100) if total > 0 else 0
                    row.append(percentage)
                    hover_row.append(f'{app_name}<br>{feature.capitalize()}: {percentage:.1f}%<br>Total: {total} ulasan')
                else:
                    row.append(0)
                    hover_row.append(f'{app_name}<br>{feature.capitalize()}: 0%<br>Total: 0 ulasan')
            
            z_matrix.append(row)
            hover_text.append(hover_row)
        
        fig = go.Figure(data=go.Heatmap(
            z=z_matrix,
            x=features,
            y=app_names,
            colorscale='RdYlGn',
            hovertemplate='%{hovertext}<extra></extra>',
            hovertext=hover_text,
            colorbar=dict(title="Persentase Positif (%)")
        ))
        
        fig.update_layout(
            title='Heatmap Sentimen Positif per Aplikasi dan Fitur',
            xaxis_title='Fitur',
            yaxis_title='Aplikasi',
            height=500
        )
        
        return fig
    
    def create_total_reviews_chart(self, apps_data: Dict[str, Dict]) -> go.Figure:
        """Create chart showing total reviews per app"""
        app_names = []
        total_reviews = []
        
        for app_name, app_data in apps_data.items():
            app_names.append(app_name)
            total = sum(
                data.get('positive', 0) + data.get('negative', 0)
                for data in app_data.values()
            )
            total_reviews.append(total)
        
        # Sort by total reviews
        sorted_data = sorted(zip(app_names, total_reviews), key=lambda x: x[1], reverse=True)
        app_names, total_reviews = zip(*sorted_data)
        
        fig = px.bar(
            x=list(app_names),
            y=list(total_reviews),
            title='Total Ulasan per Aplikasi',
            labels={'x': 'Aplikasi', 'y': 'Total Ulasan'},
            color=list(total_reviews),
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            xaxis_title='Aplikasi',
            yaxis_title='Total Ulasan',
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_feature_distribution_pie(self, apps_data: Dict[str, Dict]) -> go.Figure:
        """Create pie chart showing distribution of reviews across features"""
        feature_totals = {'kanji': 0, 'kotoba': 0, 'bunpou': 0}
        
        for app_data in apps_data.values():
            for feature in feature_totals.keys():
                if feature in app_data:
                    total = app_data[feature].get('positive', 0) + app_data[feature].get('negative', 0)
                    feature_totals[feature] += total
        
        fig = px.pie(
            names=[f.capitalize() for f in feature_totals.keys()],
            values=list(feature_totals.values()),
            title='Distribusi Ulasan per Fitur',
            color_discrete_map={
                'Kanji': self.feature_colors['kanji'],
                'Kotoba': self.feature_colors['kotoba'],
                'Bunpou': self.feature_colors['bunpou']
            }
        )
        
        fig.update_traces(
            hovertemplate='<b>%{label}</b><br>%{value} ulasan<br>%{percent}<extra></extra>'
        )
        
        fig.update_layout(height=400)
        
        return fig
    
    def create_sentiment_trend_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create line chart showing sentiment trends (if temporal data available)"""
        # This is a placeholder for future temporal analysis
        # Currently creates a summary chart
        
        features = df['feature'].unique()
        positive_percentages = []
        
        for feature in features:
            feature_data = df[df['feature'] == feature]
            avg_positive = feature_data['positive_percentage'].mean()
            positive_percentages.append(avg_positive)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=[f.capitalize() for f in features],
            y=positive_percentages,
            mode='lines+markers',
            name='Rata-rata Sentimen Positif',
            line=dict(color=self.color_palette['primary'], width=3),
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title='Rata-rata Sentimen Positif per Fitur',
            xaxis_title='Fitur',
            yaxis_title='Persentase Positif (%)',
            height=400,
            yaxis=dict(range=[0, 100])
        )
        
        return fig
    
    def create_app_ranking_chart(self, apps_data: Dict[str, Dict], feature: str = None) -> go.Figure:
        """Create ranking chart for apps based on overall or feature-specific performance"""
        rankings = []
        
        for app_name, app_data in apps_data.items():
            if feature and feature in app_data:
                # Rank by specific feature
                pos = app_data[feature].get('positive', 0)
                neg = app_data[feature].get('negative', 0)
                total = pos + neg
                score = (pos / total * 100) if total > 0 else 0
                rankings.append({'app': app_name, 'score': score, 'total': total})
            elif not feature:
                # Rank by overall performance (weighted average)
                total_score = 0
                total_reviews = 0
                
                for feat_data in app_data.values():
                    pos = feat_data.get('positive', 0)
                    neg = feat_data.get('negative', 0)
                    reviews = pos + neg
                    if reviews > 0:
                        total_score += (pos / reviews) * reviews
                        total_reviews += reviews
                
                score = (total_score / total_reviews) if total_reviews > 0 else 0
                rankings.append({'app': app_name, 'score': score, 'total': total_reviews})
        
        # Sort by score
        rankings.sort(key=lambda x: x['score'], reverse=True)
        
        apps = [r['app'] for r in rankings]
        scores = [r['score'] for r in rankings]
        totals = [r['total'] for r in rankings]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=scores,
            y=apps,
            orientation='h',
            marker=dict(
                color=scores,
                colorscale='RdYlGn',
                colorbar=dict(title="Score (%)")
            ),
            text=[f'{s:.1f}% ({t} ulasan)' for s, t in zip(scores, totals)],
            textposition='inside',
            hovertemplate='<b>%{y}</b><br>Score: %{x:.1f}%<extra></extra>'
        ))
        
        title = f'Ranking Aplikasi - {feature.capitalize()}' if feature else 'Ranking Aplikasi - Overall'
        
        fig.update_layout(
            title=title,
            xaxis_title='Score (%)',
            yaxis_title='Aplikasi',
            height=500
        )
        
        return fig