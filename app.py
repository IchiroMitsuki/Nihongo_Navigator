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
        
        # Load data with fallback mechanism
        try:
            self.apps_data = self.load_data_with_fallback()
        except Exception as e:
            st.error(f"❌ Critical error loading data: {str(e)}")
            self.apps_data = {}
        
        # Cache API connection status
        if 'api_connected' not in st.session_state:
            st.session_state.api_connected = self.check_api_connection()
        
    def load_all_data(self):
        """Load all sentiment data from API"""
        api_base_url = "https://nihongonavigator-api-production.up.railway.app"
        
        # Daftar endpoint aplikasi
        app_endpoints = {
            'Mazii': '/api/sentiment/mazii',
            'Obenkyo': '/api/sentiment/obenkyo', 
            'Hey Japan': '/api/sentiment/heyjapan',
            'JA Sensei': '/api/sentiment/jasensei',
            'Migii JLPT': '/api/sentiment/migiijlpt',
            'Kanji Study': '/api/sentiment/kanjistudy'
        }
        
        apps_data = {}
        
        # Progress bar untuk loading
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (app_name, endpoint) in enumerate(app_endpoints.items()):
            try:
                status_text.text(f"Loading data {app_name}...")
                progress_bar.progress((i + 1) / len(app_endpoints))
                
                # Request ke API
                response = requests.get(f"{api_base_url}{endpoint}", timeout=10)
                
                if response.status_code == 200:
                    api_data = response.json()
                    
                    # Transform API data ke format yang dibutuhkan aplikasi
                    apps_data[app_name] = self.transform_api_data(api_data)
                    
                else:
                    st.warning(f"⚠️ Gagal memuat data {app_name} dari API. Menggunakan fallback data.")
                    apps_data[app_name] = self.get_fallback_data(app_name)
                    
            except requests.RequestException as e:
                st.error(f"❌ Error koneksi API untuk {app_name}: {str(e)}")
                apps_data[app_name] = self.get_fallback_data(app_name)
            
            except Exception as e:
                st.error(f"❌ Error processing data {app_name}: {str(e)}")
                apps_data[app_name] = self.get_fallback_data(app_name)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return apps_data
    
    def transform_api_data(self, api_data: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
        """Transform API response to internal data format"""
        transformed_data = {}
        
        # Sesuaikan dengan struktur response API
        # Asumsi API mengembalikan struktur seperti:
        # {
        #   "features": {
        #     "kanji": {"positive": 10, "negative": 2},
        #     "kotoba": {"positive": 15, "negative": 1},
        #     "bunpou": {"positive": 8, "negative": 0}
        #   }
        # }
        
        if 'features' in api_data:
            for feature, sentiment_data in api_data['features'].items():
                transformed_data[feature] = {
                    'positive': sentiment_data.get('positive', 0),
                    'negative': sentiment_data.get('negative', 0)
                }
        
        # Alternatif jika struktur API berbeda
        elif 'data' in api_data:
            for item in api_data['data']:
                feature = item.get('feature', '')
                if feature:
                    transformed_data[feature] = {
                        'positive': item.get('positive_count', 0),
                        'negative': item.get('negative_count', 0)
                    }
        
        return transformed_data

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
        
        # Text input with character counter
        user_input = st.text_area(
            "Tulis ulasan Anda:",
            placeholder="Contoh: Aplikasi ini sangat membantu untuk belajar kanji, interfacenya mudah digunakan...",
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
            predict_button = st.button("🔍 Analisis Sentimen", disabled=len(user_input) < 10)
        
        with col2:
            if 'last_prediction' in st.session_state and st.session_state.last_prediction:
                update_button = st.button("💾 Update Data", type="primary")
            else:
                st.button("💾 Update Data", disabled=True, help="Lakukan prediksi terlebih dahulu")
        
        with col3:
            if st.button("🔄 Reset"):
                if 'last_prediction' in st.session_state:
                    del st.session_state.last_prediction
                st.rerun()
        
        # Perform prediction
        if predict_button and user_input:
            with st.spinner("Menganalisis sentimen..."):
                prediction = self.ml_predictor.predict_sentiment(user_input)
                
                # Store prediction in session state
                st.session_state.last_prediction = {
                    'text': user_input,
                    'app': selected_app,
                    'prediction': prediction,
                    'timestamp': pd.Timestamp.now().isoformat()
                }
        
        # Display prediction results
        if 'last_prediction' in st.session_state and st.session_state.last_prediction:
            self._display_prediction_results(st.session_state.last_prediction)
        
        # Update data functionality
        if 'last_prediction' in st.session_state and st.session_state.last_prediction:
            if st.button("💾 Update Data", key="update_data_main"):
                self._update_app_data(st.session_state.last_prediction)

    def _display_prediction_results(self, prediction_data):
        """Display prediction results in a formatted way"""
        st.markdown("---")
        st.subheader("📊 Hasil Analisis")
        
        prediction = prediction_data['prediction']
        selected_app = prediction_data['app']
        
        # Main results
        col1, col2, col3 = st.columns(3)
        
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
                delta="Akan diupdate"
            )
        
        with col3:
            detected_features = prediction.get('features', [])
            feature_count = len(detected_features)
            st.metric(
                "Fitur Terdeteksi", 
                feature_count,
                delta=f"{', '.join(detected_features) if detected_features else 'Tidak ada'}"
            )
        
        # Detailed analysis
        with st.expander("📋 Detail Analisis", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Fitur yang Terdeteksi:**")
                if detected_features:
                    for feature in detected_features:
                        feature_display = {
                            'kanji': '🔤 Kanji - Pembelajaran karakter Jepang',
                            'kotoba': '💬 Kotoba - Kosakata bahasa Jepang',
                            'bunpou': '📚 Bunpou - Tata bahasa Jepang'
                        }
                        st.write(f"• {feature_display.get(feature, feature.title())}")
                else:
                    st.write("• Tidak ada fitur spesifik terdeteksi")
                    st.info("💡 **Tips:** Sebutkan 'kanji', 'kosakata', atau 'grammar' untuk deteksi fitur yang lebih baik")
            
            with col2:
                st.write("**Informasi Teknis:**")
                st.write(f"• **Panjang teks:** {prediction.get('text_length', 0)} karakter")
                st.write(f"• **Confidence score:** {prediction['confidence']:.2f}%")
                st.write(f"• **Metode:** {prediction.get('method', 'ML Model')}")
                st.write(f"• **Timestamp:** {prediction_data['timestamp'][:19]}")
        
        # Preview of data changes
        st.subheader("👀 Preview Update Data")
        self._show_data_update_preview(prediction_data)
    
    def _show_data_update_preview(self, prediction_data):
        """Show preview of how data will be updated"""
        prediction = prediction_data['prediction']
        selected_app = prediction_data['app']
        detected_features = prediction.get('features', [])
        sentiment = prediction['sentiment']
        
        if not detected_features:
            st.warning("⚠️ Tidak ada fitur terdeteksi. Data tidak akan diupdate.")
            st.info("💡 Untuk update data, pastikan ulasan menyebutkan fitur seperti 'kanji', 'kosakata', atau 'grammar'")
            return
        
        # Current data
        current_data = self.apps_data.get(selected_app, {})
        
        # Create preview table
        preview_data = []
        
        for feature in detected_features:
            current_pos = current_data.get(feature, {}).get('positive', 0)
            current_neg = current_data.get(feature, {}).get('negative', 0)
            current_total = current_pos + current_neg
            current_percentage = (current_pos / current_total * 100) if current_total > 0 else 0
            
            # Calculate new values
            new_pos = current_pos + (1 if sentiment == 'positive' else 0)
            new_neg = current_neg + (1 if sentiment == 'negative' else 0)
            new_total = new_pos + new_neg
            new_percentage = (new_pos / new_total * 100) if new_total > 0 else 0
            
            preview_data.append({
                'Fitur': feature.capitalize(),
                'Positif (Sekarang)': current_pos,
                'Negatif (Sekarang)': current_neg,
                'Total (Sekarang)': current_total,
                '% Positif (Sekarang)': f"{current_percentage:.1f}%",
                'Positif (Baru)': new_pos,
                'Negatif (Baru)': new_neg,
                'Total (Baru)': new_total,
                '% Positif (Baru)': f"{new_percentage:.1f}%",
                'Perubahan': f"+1 {sentiment}"
            })
        
        df_preview = pd.DataFrame(preview_data)
        
        # Display with color coding
        st.dataframe(
            df_preview,
            use_container_width=True,
            hide_index=True
        )
        
        # Summary of changes
        changes_summary = f"**Ringkasan Perubahan:**\n"
        changes_summary += f"• Aplikasi: **{selected_app}**\n"
        changes_summary += f"• Sentimen: **{sentiment.title()}**\n"
        changes_summary += f"• Fitur yang diupdate: **{', '.join([f.capitalize() for f in detected_features])}**\n"
        changes_summary += f"• Total fitur terpengaruh: **{len(detected_features)}**"
        
        st.success(changes_summary)
    
    def _update_app_data(self, prediction_data):
        """Update application data with new review"""
        prediction = prediction_data['prediction']
        selected_app = prediction_data['app']
        detected_features = prediction.get('features', [])
        sentiment = prediction['sentiment']
        
        if not detected_features:
            st.error("❌ Tidak dapat mengupdate data: Tidak ada fitur yang terdeteksi")
            return
        
        try:
            # Update in-memory data
            if selected_app not in self.apps_data:
                self.apps_data[selected_app] = {}
            
            updated_features = []
            
            for feature in detected_features:
                if feature not in self.apps_data[selected_app]:
                    self.apps_data[selected_app][feature] = {'positive': 0, 'negative': 0}
                
                # Update counts
                if sentiment == 'positive':
                    self.apps_data[selected_app][feature]['positive'] += 1
                else:
                    self.apps_data[selected_app][feature]['negative'] += 1
                
                updated_features.append(feature)
            
            # Save to file (optional - create backup)
            self._save_updated_data(selected_app, prediction_data)
            
            # Show success message
            st.success("✅ Data berhasil diupdate!")
            
            # Update session state for immediate UI refresh
            st.session_state.data_updated = True
            
            # Show updated statistics
            self._show_updated_statistics(selected_app, updated_features)
            
            # Clear prediction from session state
            if 'last_prediction' in st.session_state:
                del st.session_state.last_prediction
            
            # Auto refresh after 2 seconds
            time.sleep(2)
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error saat mengupdate data: {str(e)}")

    def _show_updated_statistics(self, app_name, updated_features):
        """Show updated statistics after data update"""
        st.subheader("📈 Statistik Terupdate")
        
        app_data = self.apps_data[app_name]
        
        cols = st.columns(len(updated_features))
        
        for i, feature in enumerate(updated_features):
            with cols[i]:
                feature_data = app_data[feature]
                positive = feature_data.get('positive', 0)
                negative = feature_data.get('negative', 0)
                total = positive + negative
                percentage = (positive / total * 100) if total > 0 else 0
                
                st.metric(
                    f"{feature.capitalize()}",
                    f"{percentage:.1f}%",
                    delta=f"Total: {total} ulasan"
                )

    def _save_updated_data(self, app_name, prediction_data):
        """Save updated data to API and create local backup"""
        import json
        import os
        from datetime import datetime
        
        try:
            # Prepare data untuk API
            updated_data = self.apps_data[app_name]
            
            # Kirim update ke API
            api_url = "https://nihongonavigator-api-production.up.railway.app"
            app_endpoint_map = {
                'Mazii': '/api/sentiment/mazii/update',
                'Obenkyo': '/api/sentiment/obenkyo/update',
                'Hey Japan': '/api/sentiment/heyjapan/update', 
                'JA Sensei': '/api/sentiment/jasensei/update',
                'Migii JLPT': '/api/sentiment/migiijlpt/update',
                'Kanji Study': '/api/sentiment/kanjistudy/update'
            }
            
            endpoint = app_endpoint_map.get(app_name)
            if endpoint:
                # Payload untuk API
                payload = {
                    'features': updated_data,
                    'metadata': {
                        'timestamp': prediction_data['timestamp'],
                        'sentiment': prediction_data['prediction']['sentiment'],
                        'confidence': prediction_data['prediction']['confidence'],
                        'review_text': prediction_data['text'][:200]  # Limit text length
                    }
                }
                
                response = requests.post(
                    f"{api_url}{endpoint}",
                    json=payload,
                    headers={'Content-Type': 'application/json'},
                    timeout=10
                )
                
                if response.status_code == 200:
                    st.success("✅ Data berhasil disinkronkan dengan API")
                else:
                    st.warning(f"⚠️ Gagal sync dengan API (Status: {response.status_code})")
            
            # Tetap buat backup lokal
            self._create_local_backup(app_name, prediction_data, updated_data)
            
        except requests.RequestException as e:
            st.error(f"❌ Error saat sync dengan API: {str(e)}")
            st.info("💾 Data tersimpan lokal, akan sync otomatis nanti")
            self._create_local_backup(app_name, prediction_data, updated_data)
        
        except Exception as e:
            st.error(f"❌ Error saat menyimpan data: {str(e)}")

    def _create_local_backup(self, app_name, prediction_data, updated_data):
        """Create local backup of updated data"""
        import json
        import os
        from datetime import datetime
        
        try:
            # Create backup directory
            backup_dir = "data/backups"
            os.makedirs(backup_dir, exist_ok=True)
            
            # Create backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{backup_dir}/backup_{app_name}_{timestamp}.json"
            
            # Save backup
            backup_data = {
                'app_name': app_name,
                'timestamp': prediction_data['timestamp'],
                'data': updated_data,
                'prediction_info': prediction_data['prediction']
            }
            
            with open(backup_filename, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
            
            # Update log
            self._update_local_log(app_name, prediction_data, backup_filename)
            
            st.info(f"💾 Backup lokal disimpan: {backup_filename}")
            
        except Exception as e:
            st.warning(f"⚠️ Gagal membuat backup lokal: {str(e)}")

    def check_api_connection(self) -> bool:
        """Check if API is accessible"""
        try:
            response = requests.get(
                "https://nihongonavigator-api-production.up.railway.app/api/health",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False

    # Add this method to show update history
    def create_update_history(self):
        """Create page to show update history"""
        st.subheader("📜 Riwayat Update Data")
        
        log_file = "data/update_log.json"
        
        if not os.path.exists(log_file):
            st.info("Belum ada riwayat update data.")
            return
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
            
            if not logs:
                st.info("Belum ada riwayat update data.")
                return
            
            # Sort by timestamp (newest first)
            logs.sort(key=lambda x: x['timestamp'], reverse=True)
            
            # Display filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                app_filter = st.selectbox(
                    "Filter Aplikasi:",
                    options=['Semua'] + list(set(log['app_name'] for log in logs))
                )
            
            with col2:
                sentiment_filter = st.selectbox(
                    "Filter Sentimen:",
                    options=['Semua', 'positive', 'negative']
                )
            
            with col3:
                limit = st.selectbox("Tampilkan:", options=[10, 25, 50, 100], index=0)
            
            # Filter logs
            filtered_logs = logs
            
            if app_filter != 'Semua':
                filtered_logs = [log for log in filtered_logs if log['app_name'] == app_filter]
            
            if sentiment_filter != 'Semua':
                filtered_logs = [log for log in filtered_logs if log['sentiment'] == sentiment_filter]
            
            filtered_logs = filtered_logs[:limit]
            
            # Display logs
            st.write(f"Menampilkan {len(filtered_logs)} dari {len(logs)} total update")
            
            for i, log in enumerate(filtered_logs):
                with st.expander(f"Update #{i+1}: {log['app_name']} - {log['sentiment'].title()} ({log['timestamp'][:19]})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Detail Update:**")
                        st.write(f"• **Aplikasi:** {log['app_name']}")
                        st.write(f"• **Sentimen:** {log['sentiment'].title()}")
                        st.write(f"• **Confidence:** {log.get('confidence', 0):.1f}%")
                        st.write(f"• **Fitur:** {', '.join(log.get('features', []))}")
                    
                    with col2:
                        st.write("**Ulasan:**")
                        st.write(f'"{log.get("review_text", "")}"')
                        st.write(f"**Backup File:** `{log.get('backup_file', 'N/A')}`")
            
            # Export functionality
            if st.button("📥 Export Log ke CSV"):
                df_logs = pd.DataFrame(filtered_logs)
                csv = df_logs.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"update_log_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"Error loading update history: {str(e)}")

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
            "📜 Riwayat Update": self.create_update_history,
            "🔄 Reload Data": self.reload_data_from_api  # New option
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
        **Update Real-time:** {'✅ Aktif' if api_connected else '⚠️ Offline Mode'}
        """)
        
        # Manual refresh button
        if st.sidebar.button("🔄 Refresh Data"):
            self.reload_data_from_api()

# Run the application
if __name__ == "__main__":
    app = SentimentAnalysisApp()
    app.run()