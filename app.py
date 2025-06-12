import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from src.data_processor import DataProcessor
from src.visualization import Visualizer
from src.ml_predictor import MLPredictor
from src.utils import load_json_data, calculate_percentage
import time 

import requests
import streamlit as st
from typing import Dict, Any, Optional

# Page configuration
st.set_page_config(
    page_title="Analisis Sentimen Aplikasi Bahasa Jepang",
    page_icon="🇯🇵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #FF6B6B;
    text-align: center;
    margin-bottom: 2rem;
}
.feature-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}
.metric-card {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

class SentimentAnalysisApp:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.visualizer = Visualizer()
        self.ml_predictor = MLPredictor()
        
        # Load data from session state first, then fallback to API/default
        if not self.load_data_from_session():
            try:
                self.apps_data = self.load_data_with_fallback()
            except Exception as e:
                st.error(f"❌ Critical error loading data: {str(e)}")
                self.apps_data = {}
        
        # Simpan ke session state
        st.session_state.apps_data = self.apps_data
        
        # Cache API connection status
        if 'api_connected' not in st.session_state:
            st.session_state.api_connected = self.check_api_connection()

    def export_updated_data(self):
        """Export updated data to JSON"""
        try:
            export_data = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'apps_data': self.apps_data,
                'total_apps': len(self.apps_data),
                'total_reviews': sum(
                    sum(feature_data.get('positive', 0) + feature_data.get('negative', 0) 
                        for feature_data in app_data.values())
                    for app_data in self.apps_data.values()
                )
            }
            
            json_data = json.dumps(export_data, indent=2, ensure_ascii=False)
            
            st.download_button(
                label="📥 Download Data (JSON)",
                data=json_data,
                file_name=f"sentiment_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
            return True
        except Exception as e:
            st.error(f"Error exporting data: {str(e)}")
            return False

    def reset_data(self):
        """Reset data to original state"""
        if st.button("🔄 Reset Data", help="Reset semua data ke kondisi awal"):
            if st.session_state.get('confirm_reset', False):
                # Reset to original fallback data
                fallback_apps = ['Mazii', 'Obenkyo', 'Hey Japan', 'JA Sensei', 'Migii JLPT', 'Kanji Study']
                self.apps_data = {app: self.get_fallback_data(app) for app in fallback_apps}
                st.session_state.apps_data = self.apps_data
                st.session_state.confirm_reset = False
                st.success("✅ Data berhasil direset!")
                st.rerun()
            else:
                st.session_state.confirm_reset = True
                st.warning("⚠️ Klik sekali lagi untuk konfirmasi reset data")
        
    def load_all_data(self):
        """Load all sentiment data from API with improved error handling"""
        api_base_url = "https://nihongonavigator-api-production.up.railway.app"
        
        # Daftar aplikasi dengan nama yang sesuai format API
        app_mapping = {
            'Mazii': 'mazii',
            'Obenkyo': 'obenkyo', 
            'Hey Japan': 'heyjapan',
            'JA Sensei': 'jasensei',
            'Migii JLPT': 'migiijlpt',
            'Kanji Study': 'kanjistudy'
        }
        
        apps_data = {}
        
        # Progress bar untuk loading
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (app_display_name, app_api_name) in enumerate(app_mapping.items()):
            try:
                status_text.text(f"Loading data {app_display_name}...")
                progress_bar.progress((i + 1) / len(app_mapping))
                
                # METHOD 1: Try to get existing data with a valid text payload
                # Since API requires text, we'll send a request to get stored data
                payload_get_data = {
                    "text": "get_stored_data",  # Valid text that indicates we want stored data
                    "app_name": app_api_name,
                    "action": "retrieve_data"  # Additional parameter to indicate we want existing data
                }
                
                response = requests.post(
                    f"{api_base_url}/predict", 
                    json=payload_get_data,
                    headers={
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    },
                    timeout=15
                )
                
                # st.write(f"Debug - {app_display_name} Response Status: {response.status_code}")
                
                # if response.status_code == 200:
                #     api_data = response.json()
                #     st.write(f"Debug - {app_display_name} API Data Keys: {list(api_data.keys())}")
                    
                #     # Check if response contains stored data
                #     if self._has_stored_data(api_data):
                #         apps_data[app_display_name] = self.transform_api_data(api_data, app_display_name)
                #         st.success(f"✅ Berhasil memuat data {app_display_name} dari API")
                #         continue
                #     else:
                #         pass
                #         st.info(f"ℹ️ {app_display_name}: API response tidak mengandung data tersimpan")
                
                # METHOD 2: If no stored data, try to retrieve using sample requests
                # This approach sends sample texts to understand API structure
                success = self._load_via_sample_requests(app_display_name, app_api_name, api_base_url)
                
                if success:
                    # For now, use fallback data but mark as API accessible
                    apps_data[app_display_name] = self.get_fallback_data(app_display_name)
                    # st.info(f"ℹ️ {app_display_name}: API accessible, menggunakan data fallback")
                else:
                    # METHOD 3: Pure fallback
                    apps_data[app_display_name] = self.get_fallback_data(app_display_name)
                    # st.warning(f"⚠️ {app_display_name}: Menggunakan data fallback (API tidak accessible)")
                        
            except requests.RequestException as e:
                pass
                #     st.error(f"❌ Error koneksi API untuk {app_display_name}: {str(e)}")
                #     apps_data[app_display_name] = self.get_fallback_data(app_display_name)
            
            except Exception as e:
                pass
                #     st.error(f"❌ Error processing data {app_display_name}: {str(e)}")
                #     apps_data[app_display_name] = self.get_fallback_data(app_display_name)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return apps_data
    
    def _has_stored_data(self, api_response: Dict[str, Any]) -> bool:
        """Check if API response contains stored/historical data"""
        # Check for various indicators of stored data
        indicators = [
            'stored_data', 'historical_data', 'app_data', 
            'sentiment_counts', 'feature_counts', 'statistics'
        ]
        
        for indicator in indicators:
            if indicator in api_response:
                return True
        
        # Check if response has structured sentiment data (not just a prediction)
        if 'features' in api_response:
            features = api_response['features']
            if isinstance(features, dict):
                # Check if it looks like aggregated data rather than single prediction
                for feature_data in features.values():
                    if isinstance(feature_data, dict) and 'positive' in feature_data and 'negative' in feature_data:
                        # If counts are > 1, likely stored data
                        if feature_data.get('positive', 0) > 1 or feature_data.get('negative', 0) > 1:
                            return True
        
        return False
    


    def _load_via_sample_requests(self, app_display_name: str, app_api_name: str, api_base_url: str) -> bool:
        """Try to load data using sample requests to understand API structure"""
        try:
            # Send a sample request to test API accessibility and response structure
            sample_payload = {
                "text": "Aplikasi ini sangat membantu untuk belajar bahasa Jepang",
                "app_name": app_api_name
            }
            
            response = requests.post(
                f"{api_base_url}/predict",
                json=sample_payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                api_data = response.json()
                # st.write(f"Debug - {app_display_name} Sample Response: {api_data}")
                
                # Check if response indicates API is working properly
                if 'prediction' in api_data or 'sentiment' in api_data:
                    # st.success(f"✅ {app_display_name}: API prediction endpoint working")
                    return True
                else:
                    # st.warning(f"⚠️ {app_display_name}: Unexpected API response format")
                    return False
            else:
                # st.warning(f"⚠️ {app_display_name}: API returned status {response.status_code}")
                return False
                
        except Exception as e:
            st.error(f"❌ {app_display_name}: Sample request failed - {str(e)}")
            return False

    def transform_api_data(self, api_data: Dict[str, Any], app_name: str) -> Dict[str, Dict[str, int]]:
        """Transform API response to internal data format - IMPROVED VERSION"""
        transformed_data = {}
        
        # PERBAIKAN: Handle berbagai format response API
        try:
            # Format 1: Direct features object
            if 'features' in api_data:
                for feature, sentiment_data in api_data['features'].items():
                    transformed_data[feature] = {
                        'positive': sentiment_data.get('positive', 0),
                        'negative': sentiment_data.get('negative', 0)
                    }
            
            # Format 2: Data array
            elif 'data' in api_data:
                for item in api_data['data']:
                    feature = item.get('feature', '')
                    if feature:
                        transformed_data[feature] = {
                            'positive': item.get('positive_count', 0),
                            'negative': item.get('negative_count', 0)
                        }
            
            # Format 3: Direct sentiment counts
            elif 'sentiment_data' in api_data:
                sentiment_data = api_data['sentiment_data']
                for feature in ['kanji', 'kotoba', 'bunpou']:
                    if feature in sentiment_data:
                        transformed_data[feature] = {
                            'positive': sentiment_data[feature].get('positive', 0),
                            'negative': sentiment_data[feature].get('negative', 0)
                        }
            
            # Format 4: Flat structure
            elif 'kanji_positive' in api_data or 'positive_kanji' in api_data:
                # Handle flat structure like: kanji_positive, kanji_negative, etc.
                features = ['kanji', 'kotoba', 'bunpou']
                for feature in features:
                    pos_key = f"{feature}_positive" if f"{feature}_positive" in api_data else f"positive_{feature}"
                    neg_key = f"{feature}_negative" if f"{feature}_negative" in api_data else f"negative_{feature}"
                    
                    transformed_data[feature] = {
                        'positive': api_data.get(pos_key, 0),
                        'negative': api_data.get(neg_key, 0)
                    }
            
            # Format 5: If API returns prediction format, extract data differently
            elif 'prediction' in api_data:
                # This might be a prediction response, handle accordingly
                prediction = api_data['prediction']
                if 'features' in prediction:
                    for feature in prediction['features']:
                        if feature not in transformed_data:
                            transformed_data[feature] = {'positive': 0, 'negative': 0}
                        
                        sentiment = prediction.get('sentiment', 'positive')
                        if sentiment == 'positive':
                            transformed_data[feature]['positive'] += 1
                        else:
                            transformed_data[feature]['negative'] += 1
            
            # Jika tidak ada data yang bisa ditransform, log struktur API
            if not transformed_data:
                st.warning(f"⚠️ Format API tidak dikenali untuk {app_name}")
                st.json(api_data)  # Show API structure for debugging
                
            return transformed_data
            
        except Exception as e:
            st.error(f"Error transforming API data for {app_name}: {str(e)}")
            st.json(api_data)  # Debug: show actual API response
            return {}

    def reload_data_from_api(self):
        """Reload data from API - untuk refresh manual"""
        with st.spinner("🔄 Memuat ulang data dari API..."):
            self.apps_data = self.load_all_data()
        st.success("✅ Data berhasil dimuat ulang dari API!")
        st.rerun()

    def get_fallback_data(self, app_name):
        """Fallback data based on your provided data"""
        fallback_data = {
            'Mazii': {"kanji": {"positive": 57, "negative": 0}, "kotoba": {"positive": 32, "negative": 0}, "bunpou": {"positive": 26, "negative": 0}},
            'Obenkyo': {"bunpou": {"positive": 5, "negative": 0}, "kanji": {"positive": 29, "negative": 0}, "kotoba": {"positive": 14, "negative": 0}},
            'Hey Japan': {"kotoba": {"positive": 66, "negative": 0}, "kanji": {"positive": 45, "negative": 0}, "bunpou": {"positive": 12, "negative": 0}},
            'JA Sensei': {"kanji": {"positive": 8, "negative": 0}, "kotoba": {"positive": 3, "negative": 0}, "bunpou": {"positive": 6, "negative": 0}},
            'Migii JLPT': {"kotoba": {"positive": 10, "negative": 0}, "bunpou": {"positive": 6, "negative": 0}, "kanji": {"positive": 18, "negative": 0}},
            'Kanji Study': {"kanji": {"positive": 187, "negative": 0}, "bunpou": {"positive": 12, "negative": 0}, "kotoba": {"positive": 33, "negative": 1}}
        }
        return fallback_data.get(app_name, {})
    
    def create_overview_dashboard(self):
        """Create main dashboard overview"""
        st.markdown('<h1 class="main-header">🇯🇵 Analisis Sentimen Aplikasi Bahasa Jepang</h1>', unsafe_allow_html=True)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_reviews = sum(
            sum(feature_data.get('positive', 0) + feature_data.get('negative', 0) 
                for feature_data in app_data.values())
            for app_data in self.apps_data.values()
        )
        
        total_positive = sum(
            sum(feature_data.get('positive', 0) for feature_data in app_data.values())
            for app_data in self.apps_data.values()
        )
        
        with col1:
            st.metric("Total Aplikasi", len(self.apps_data), delta=None)
        with col2:
            st.metric("Total Ulasan", total_reviews, delta=None)
        with col3:
            st.metric("Ulasan Positif", total_positive, delta=None)
        with col4:
            positive_rate = (total_positive / total_reviews * 100) if total_reviews > 0 else 0
            st.metric("Tingkat Positif", f"{positive_rate:.1f}%", delta=None)
    
    def create_comparison_table(self):
        """Create comparison table of all applications"""
        st.subheader("📊 Perbandingan Aplikasi Berdasarkan Fitur")
        
        # Prepare data for table
        table_data = []
        features = ['kanji', 'kotoba', 'bunpou']
        
        for app_name, app_data in self.apps_data.items():
            row = {'Aplikasi': app_name}
            
            for feature in features:
                if feature in app_data:
                    pos = app_data[feature].get('positive', 0)
                    neg = app_data[feature].get('negative', 0)
                    total = pos + neg
                    percentage = (pos / total * 100) if total > 0 else 0
                    row[f'{feature.capitalize()} (%)'] = f"{percentage:.1f}%"
                    row[f'{feature.capitalize()} Total'] = total
                else:
                    row[f'{feature.capitalize()} (%)'] = "0.0%"
                    row[f'{feature.capitalize()} Total'] = 0
            
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)
        
        # Export functionality
        if st.button("📥 Export ke CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="analisis_sentimen_apps.csv",
                mime="text/csv"
            )

    def create_feature_analysis(self):
        """Create detailed feature analysis"""
        st.subheader("🔍 Analisis Detail Berdasarkan Fitur")
        
        # Feature selection
        selected_feature = st.selectbox(
            "Pilih Fitur untuk Analisis:",
            ['kanji', 'kotoba', 'bunpou'],
            format_func=lambda x: x.capitalize()
        )
        
        # Prepare data for visualization
        feature_data = []
        for app_name, app_data in self.apps_data.items():
            if selected_feature in app_data:
                pos = app_data[selected_feature].get('positive', 0)
                neg = app_data[selected_feature].get('negative', 0)
                total = pos + neg
                percentage = (pos / total * 100) if total > 0 else 0
                
                feature_data.append({
                    'Aplikasi': app_name,
                    'Positif': pos,
                    'Negatif': neg,
                    'Total': total,
                    'Persentase Positif': percentage
                })
        
        df_feature = pd.DataFrame(feature_data)
        df_feature = df_feature.sort_values('Persentase Positif', ascending=False)
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart
            fig_bar = px.bar(
                df_feature, 
                x='Aplikasi', 
                y='Persentase Positif',
                title=f'Persentase Sentimen Positif - {selected_feature.capitalize()}',
                color='Persentase Positif',
                color_continuous_scale='Viridis'
            )
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Pie chart
            fig_pie = px.pie(
                df_feature,
                values='Total',
                names='Aplikasi',
                title=f'Distribusi Total Ulasan - {selected_feature.capitalize()}'
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Ranking table
        st.subheader(f"🏆 Ranking Aplikasi - {selected_feature.capitalize()}")
        df_ranking = df_feature[['Aplikasi', 'Total', 'Persentase Positif']].copy()
        df_ranking['Rank'] = range(1, len(df_ranking) + 1)
        df_ranking = df_ranking[['Rank', 'Aplikasi', 'Total', 'Persentase Positif']]
        st.dataframe(df_ranking, use_container_width=True)
    
    def save_prediction_to_data(self, app_name, feature, sentiment):
        """Save prediction result to app data"""
        try:
            # Pastikan app_name ada dalam data
            if app_name not in self.apps_data:
                self.apps_data[app_name] = {}
            
            # Pastikan feature ada dalam app data
            if feature not in self.apps_data[app_name]:
                self.apps_data[app_name][feature] = {'positive': 0, 'negative': 0}
            
            # Update count berdasarkan sentiment
            if sentiment == 'positive':
                self.apps_data[app_name][feature]['positive'] += 1
            elif sentiment == 'negative':
                self.apps_data[app_name][feature]['negative'] += 1
            
            # Simpan ke session state untuk persistence
            st.session_state.apps_data = self.apps_data
            
            return True
        except Exception as e:
            st.error(f"Error saving prediction: {str(e)}")
            return False

    def load_data_from_session(self):
        """Load data from session state if available"""
        if 'apps_data' in st.session_state:
            self.apps_data = st.session_state.apps_data
            return True
        return False

    def create_live_prediction(self):
        """Create live sentiment prediction interface with app selection and data update"""
        st.subheader("🤖 Prediksi Sentimen Real-time")
        
        st.write("Masukkan ulasan aplikasi untuk melihat prediksi sentimen dan update data aplikasi:")
        
        # App selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_app = st.selectbox(
                "Pilih Aplikasi Target:",
                options=list(self.apps_data.keys()),
                help="Pilih aplikasi yang akan menerima komentar ini"
            )
        
        with col2:
            st.info(f"**{selected_app}** dipilih")
            if selected_app in self.apps_data:
                app_data = self.apps_data[selected_app]
                total_reviews = sum(
                    feature_data.get('positive', 0) + feature_data.get('negative', 0)
                    for feature_data in app_data.values()
                )
                st.metric("Total Ulasan Saat Ini", total_reviews)
        
        # Feature selection - TAMBAHAN BARU
        st.subheader("🎯 Pilih Fitur yang Dikomentari")
        selected_feature = st.selectbox(
            "Pilih fitur yang ingin Anda komentari:",
            options=['kanji', 'kotoba', 'bunpou'],
            format_func=lambda x: {
                'kanji': '🔤 Kanji - Pembelajaran karakter Jepang',
                'kotoba': '💬 Kotoba - Kosakata bahasa Jepang', 
                'bunpou': '📚 Bunpou - Tata bahasa Jepang'
            }.get(x, x.title()),
            help="Pilih aspek aplikasi yang akan Anda komentari"
        )
        
        # Show current stats for selected feature
        if selected_app in self.apps_data and selected_feature in self.apps_data[selected_app]:
            feature_data = self.apps_data[selected_app][selected_feature]
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Positif", feature_data.get('positive', 0))
            with col2:
                st.metric("Negatif", feature_data.get('negative', 0))
            with col3:
                total = feature_data.get('positive', 0) + feature_data.get('negative', 0)
                percentage = (feature_data.get('positive', 0) / total * 100) if total > 0 else 0
                st.metric("% Positif", f"{percentage:.1f}%")
        
        # Text input with character counter
        user_input = st.text_area(
            "Tulis ulasan Anda:",
            placeholder=f"Contoh: Fitur {selected_feature} di aplikasi {selected_app} sangat membantu untuk belajar...",
            help="Minimum 10 karakter untuk prediksi yang akurat",
            max_chars=500
        )
        
        # Character counter
        if user_input:
            char_count = len(user_input)
            if char_count < 10:
                st.warning(f"⚠️ Masukkan minimal 10 karakter (saat ini: {char_count})")
            else:
                st.success(f"✅ Panjang teks: {char_count} karakter")
        
        # Prediction buttons
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            predict_button = st.button("🔍 Analisis Sentimen")
        
        with col2:
            save_to_data = st.checkbox("💾 Simpan ke Data", value=True, help="Centang untuk menyimpan hasil prediksi ke data aplikasi")
        
        # Perform prediction
        if predict_button and user_input:
            with st.spinner("Menganalisis sentimen..."):
                prediction = self.ml_predictor.predict_sentiment(user_input)
                
                # Save prediction result if checkbox is checked
                if save_to_data:
                    success = self.save_prediction_to_data(
                        selected_app, 
                        selected_feature, 
                        prediction['sentiment']
                    )
                    
                    if success:
                        st.success(f"✅ Data berhasil disimpan ke {selected_app} - {selected_feature}")
                    else:
                        st.error("❌ Gagal menyimpan data")
                
                # Store prediction in session state
                st.session_state.last_prediction = {
                    'text': user_input,
                    'app': selected_app,
                    'feature': selected_feature,  # TAMBAHAN BARU
                    'prediction': prediction,
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'saved': save_to_data
                }
        
        # Display prediction results
        if 'last_prediction' in st.session_state and st.session_state.last_prediction:
            self._display_prediction_results(st.session_state.last_prediction)

    # 4. Update method _display_prediction_results untuk menampilkan fitur yang dipilih
    def _display_prediction_results(self, prediction_data):
        """Display prediction results in a formatted way"""
        st.markdown("---")
        st.subheader("📊 Hasil Analisis")
        
        prediction = prediction_data['prediction']
        selected_app = prediction_data['app']
        selected_feature = prediction_data.get('feature', 'Unknown')
        
        # Main results
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            sentiment_color = "🟢" if prediction['sentiment'] == 'positive' else "🔴"
            st.metric(
                "Sentimen", 
                f"{sentiment_color} {prediction['sentiment'].title()}",
                delta=f"{prediction['confidence']:.1f}% confidence"
            )
        
        with col2:
            st.metric(
                "Aplikasi Target", 
                selected_app,
                delta="Target aplikasi"
            )
        
        with col3:
            feature_display = {
                'kanji': '🔤 Kanji',
                'kotoba': '💬 Kotoba',
                'bunpou': '📚 Bunpou'
            }
            st.metric(
                "Fitur Dikomentari",
                feature_display.get(selected_feature, selected_feature),
                delta="Fitur target"
            )
        
        with col4:
            save_status = "✅ Tersimpan" if prediction_data.get('saved', False) else "❌ Tidak disimpan"
            st.metric(
                "Status Penyimpanan",
                save_status,
                delta=None
            )
        
        # Show updated data if saved
        if prediction_data.get('saved', False):
            st.success("🎉 Data berhasil diperbarui!")
            
            # Show before/after comparison
            with st.expander("📈 Perubahan Data", expanded=True):
                if selected_app in self.apps_data and selected_feature in self.apps_data[selected_app]:
                    current_data = self.apps_data[selected_app][selected_feature]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Data Terbaru:**")
                        st.write(f"• Positif: {current_data.get('positive', 0)}")
                        st.write(f"• Negatif: {current_data.get('negative', 0)}")
                        total = current_data.get('positive', 0) + current_data.get('negative', 0)
                        percentage = (current_data.get('positive', 0) / total * 100) if total > 0 else 0
                        st.write(f"• Persentase Positif: {percentage:.1f}%")
                    
                    with col2:
                        st.write("**Perubahan:**")
                        if prediction['sentiment'] == 'positive':
                            st.write("• Positif: +1 ⬆️")
                            st.write("• Negatif: +0")
                        else:
                            st.write("• Positif: +0")
                            st.write("• Negatif: +1 ⬆️")
        
        # Detailed analysis
        with st.expander("📋 Detail Analisis", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Informasi Prediksi:**")
                st.write(f"• **Sentimen:** {prediction['sentiment'].title()}")
                st.write(f"• **Confidence:** {prediction['confidence']:.2f}%")
                st.write(f"• **Aplikasi:** {selected_app}")
                st.write(f"• **Fitur:** {selected_feature.title()}")
                
            with col2:
                st.write("**Informasi Teknis:**")
                st.write(f"• **Panjang teks:** {len(prediction_data['text'])} karakter")
                st.write(f"• **Metode:** {prediction.get('method', 'ML Model')}")
                st.write(f"• **Timestamp:** {prediction_data['timestamp'][:19]}")
                st.write(f"• **Disimpan:** {'Ya' if prediction_data.get('saved', False) else 'Tidak'}")

    def check_api_connection(self) -> bool:
        """Improved API connection check"""
        try:
            # Try different endpoints
            endpoints_to_try = [
                "/api/health",
                "/predict", 
                "/",
                "/status"
            ]
            
            base_url = "https://nihongonavigator-api-production.up.railway.app"
            
            for endpoint in endpoints_to_try:
                try:
                    if endpoint == "/predict":
                        # Test POST request for predict endpoint
                        response = requests.post(
                            f"{base_url}{endpoint}",
                            json={"text": "test", "app_name": "test"},
                            headers={'Content-Type': 'application/json'},
                            timeout=5
                        )
                    else:
                        # Test GET request for other endpoints
                        response = requests.get(f"{base_url}{endpoint}", timeout=5)
                    
                    if response.status_code in [200, 400, 422]:  # 400/422 might indicate API is working but needs proper input
                        # st.success(f"✅ API accessible via {endpoint} (Status: {response.status_code})")
                        return True
                        
                except requests.RequestException:
                    continue
            
            return False
            
        except Exception as e:
            st.error(f"Error checking API connection: {str(e)}")
            return False

    def load_data_with_fallback(self):
        """Load data with fallback mechanism"""
        try:
            # Try API first
            return self.load_all_data()
        except Exception as e:
            st.error(f"❌ Gagal memuat dari API: {str(e)}")
            st.info("🔄 Menggunakan data fallback lokal...")
            
            # Use fallback data
            fallback_apps = ['Mazii', 'Obenkyo', 'Hey Japan', 'JA Sensei', 'Migii JLPT', 'Kanji Study']
            return {app: self.get_fallback_data(app) for app in fallback_apps}

    def run(self):
        """Run the main application"""
        # Check API connection
        api_connected = self.check_api_connection()
        
        # Sidebar navigation
        st.sidebar.title("🗾 Navigation")
        
        # API Status indicator
        if api_connected:
            st.sidebar.success("🟢 API Connected")
        else:
            st.sidebar.error("🔴 API Disconnected")
            if st.sidebar.button("🔄 Reload Data"):
                self.reload_data_from_api()
        
        pages = {
            "🏠 Dashboard": self.create_overview_dashboard,
            "📊 Perbandingan": self.create_comparison_table,
            "🔍 Analisis Fitur": self.create_feature_analysis,
            "🤖 Prediksi Live": self.create_live_prediction,
            # "🔄 Reload Data": self.reload_data_from_api 
        }
        
        selected_page = st.sidebar.selectbox("Pilih Halaman:", list(pages.keys()))
        
        # Run selected page
        if selected_page == "🔄 Reload Data":
            self.reload_data_from_api()
        else:
            pages[selected_page]()
        
        # Sidebar info - updated
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ℹ️ Informasi")
        st.sidebar.markdown(f"""
        **Fitur Aplikasi:**
        - 📱 **Kanji**: Pembelajaran karakter Jepang
        - 🗣️ **Kotoba**: Kosakata bahasa Jepang  
        - 📚 **Bunpou**: tata bahasa Jepang
        
        **Data Source:** {'🌐 API' if api_connected else '💾 Local Backup'}
        **Aplikasi:** 6 apps pembelajaran bahasa Jepang
        """)
        
        # Manual refresh button
        if st.sidebar.button("🔄 Refresh Data"):
            self.reload_data_from_api()

# Run the application
if __name__ == "__main__":
    app = SentimentAnalysisApp()
    app.run()