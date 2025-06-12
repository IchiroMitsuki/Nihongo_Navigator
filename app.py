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
        self.apps_data = self.load_all_data()
        
    def load_all_data(self):
        """Load all sentiment data from JSON files"""
        apps_data = {}
        data_files = {
            'Mazii': 'data/raw/hasil_sentimen_mazii_agregat.json',
            'Obenkyo': 'data/raw/hasil_sentimen_obenkyo_agregat.json',
            'Hey Japan': 'data/raw/hasil_sentimen_heyjapan_agregat.json',
            'JA Sensei': 'data/raw/hasil_sentimen_jasensei_agregat.json',
            'Migii JLPT': 'data/raw/hasil_sentimen_migiijlpt_agregat.json',
            'Kanji Study': 'data/raw/hasil_sentimen_kanjistudy_agregat.json'
        }
        
        for app_name, file_path in data_files.items():
            if os.path.exists(file_path):
                apps_data[app_name] = load_json_data(file_path)
            else:
                # Fallback data jika file tidak ada
                apps_data[app_name] = self.get_fallback_data(app_name)
        
        return apps_data
    
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
                update_button = st.button("💾 Update Data", type="primary", key="update_btn_sidebar")
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
        
        # Update data functionality - FIX: Handle update button properly
        if 'last_prediction' in st.session_state and st.session_state.last_prediction:
            if update_button:  # Check if update button was clicked
                self._update_app_data(st.session_state.last_prediction)

    def _update_app_data(self, prediction_data):
        """Update application data with new review - FIXED VERSION"""
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
                
                # Update counts - FIX: Ensure proper increment
                if sentiment == 'positive':
                    self.apps_data[selected_app][feature]['positive'] += 1
                else:
                    self.apps_data[selected_app][feature]['negative'] += 1
                
                updated_features.append(feature)
            
            # Save to file - FIX: Proper file saving
            success = self._save_updated_data(selected_app, prediction_data)
            
            if success:
                # Show success message
                st.success("✅ Data berhasil diupdate!")
                
                # Show updated statistics
                self._show_updated_statistics(selected_app, updated_features)
                
                # Clear prediction from session state
                if 'last_prediction' in st.session_state:
                    del st.session_state.last_prediction
                
                # Set flag for UI refresh
                st.session_state.data_updated = True
                st.session_state.last_update_time = pd.Timestamp.now().isoformat()
                
                # Force rerun to refresh the interface
                time.sleep(1)  # Small delay to show success message
                st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error saat mengupdate data: {str(e)}")
            import traceback
            st.error(f"Detail error: {traceback.format_exc()}")

    def _save_updated_data(self, app_name, prediction_data):
        """Save updated data to JSON file and create backup - FIXED VERSION"""
        import json
        import os
        from datetime import datetime
        
        try:
            # Ensure directories exist
            os.makedirs("data/raw", exist_ok=True)
            os.makedirs("data/backups", exist_ok=True)
            
            # Create backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"data/backups/backup_{app_name.lower().replace(' ', '')}_{timestamp}.json"
            
            # Generate proper filename mapping
            filename_mapping = {
                'Mazii': 'data/raw/hasil_sentimen_mazii_agregat.json',
                'Obenkyo': 'data/raw/hasil_sentimen_obenkyo_agregat.json',
                'Hey Japan': 'data/raw/hasil_sentimen_heyjapan_agregat.json',
                'JA Sensei': 'data/raw/hasil_sentimen_jasensei_agregat.json',
                'Migii JLPT': 'data/raw/hasil_sentimen_migiijlpt_agregat.json',
                'Kanji Study': 'data/raw/hasil_sentimen_kanjistudy_agregat.json'
            }
            
            app_filename = filename_mapping.get(app_name)
            
            if not app_filename:
                # Fallback filename generation
                app_filename = f"data/raw/hasil_sentimen_{app_name.lower().replace(' ', '').replace('ja', 'ja')}_agregat.json"
            
            # Create backup if original file exists
            if os.path.exists(app_filename):
                try:
                    with open(app_filename, 'r', encoding='utf-8') as f:
                        backup_data = json.load(f)
                    
                    with open(backup_filename, 'w', encoding='utf-8') as f:
                        json.dump(backup_data, f, indent=2, ensure_ascii=False)
                    
                    st.info(f"💾 Backup dibuat: {backup_filename}")
                except Exception as backup_error:
                    st.warning(f"⚠️ Gagal membuat backup: {backup_error}")
            
            # Save updated data
            updated_data = self.apps_data[app_name]
            
            with open(app_filename, 'w', encoding='utf-8') as f:
                json.dump(updated_data, f, indent=2, ensure_ascii=False)
            
            st.success(f"✅ Data disimpan ke: {app_filename}")
            
            # Log the update
            self._log_update(app_name, prediction_data, backup_filename)
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error saat menyimpan data: {str(e)}")
            st.warning("⚠️ Data terupdate di memori tapi gagal disimpan ke file")
            return False

    def _log_update(self, app_name, prediction_data, backup_filename):
        """Log the update for history tracking"""
        import json
        import os
        
        try:
            log_entry = {
                'timestamp': prediction_data['timestamp'],
                'app_name': app_name,
                'sentiment': prediction_data['prediction']['sentiment'],
                'features': prediction_data['prediction'].get('features', []),
                'review_text': prediction_data['text'][:100] + "..." if len(prediction_data['text']) > 100 else prediction_data['text'],
                'confidence': prediction_data['prediction']['confidence'],
                'backup_file': backup_filename
            }
            
            # Save to update log
            log_file = "data/update_log.json"
            
            logs = []
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        logs = json.load(f)
                except:
                    logs = []
            
            logs.append(log_entry)
            
            # Keep only last 100 logs
            logs = logs[-100:]
            
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            st.warning(f"⚠️ Gagal mencatat log update: {str(e)}")

    def _show_updated_statistics(self, app_name, updated_features):
        """Show updated statistics after data update"""
        st.subheader("📈 Statistik Terupdate")
        
        app_data = self.apps_data[app_name]
        
        # Create columns based on number of features
        if len(updated_features) > 0:
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
                    
                    # Show detailed counts
                    st.write(f"✅ Positif: {positive}")
                    st.write(f"❌ Negatif: {negative}")

# Tambahkan juga perbaikan untuk create_update_history method
    def create_update_history(self):
        """Create page to show update history - FIXED VERSION"""
        st.subheader("📜 Riwayat Update Data")
        
        log_file = "data/update_log.json"
        
        if not os.path.exists(log_file):
            st.info("Belum ada riwayat update data.")
            if st.button("🔧 Buat File Log"):
                os.makedirs("data", exist_ok=True)
                with open(log_file, 'w', encoding='utf-8') as f:
                    json.dump([], f)
                st.success("File log berhasil dibuat!")
                st.rerun()
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
                        st.write(f"**Backup File:** `{os.path.basename(log.get('backup_file', 'N/A'))}`")
            
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
            import traceback
            st.error(f"Detail: {traceback.format_exc()}")

    def _save_updated_data(self, app_name, prediction_data):
        """Save updated data to JSON file and create backup"""
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
            
            # Save current data as backup
            app_filename = f"data/raw/hasil_sentimen_{app_name.lower().replace(' ', '')}_agregat.json"
            
            if os.path.exists(app_filename):
                with open(app_filename, 'r', encoding='utf-8') as f:
                    backup_data = json.load(f)
                
                with open(backup_filename, 'w', encoding='utf-8') as f:
                    json.dump(backup_data, f, indent=2, ensure_ascii=False)
            
            # Save updated data
            updated_data = self.apps_data[app_name]
            
            with open(app_filename, 'w', encoding='utf-8') as f:
                json.dump(updated_data, f, indent=2, ensure_ascii=False)
            
            # Log the update
            log_entry = {
                'timestamp': prediction_data['timestamp'],
                'app_name': app_name,
                'sentiment': prediction_data['prediction']['sentiment'],
                'features': prediction_data['prediction'].get('features', []),
                'review_text': prediction_data['text'][:100] + "..." if len(prediction_data['text']) > 100 else prediction_data['text'],
                'confidence': prediction_data['prediction']['confidence'],
                'backup_file': backup_filename
            }
            
            # Save to update log
            log_file = "data/update_log.json"
            
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(log_entry)
            
            # Keep only last 100 logs
            logs = logs[-100:]
            
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)
            
            st.info(f"💾 Data disimpan dan backup dibuat: {backup_filename}")
            
        except Exception as e:
            st.warning(f"⚠️ Data terupdate di memori tapi gagal disimpan ke file: {str(e)}")

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



    def run(self):
        """Run the main application"""
        # Sidebar navigation
        st.sidebar.title("🗾 Navigation")
        pages = {
            "🏠 Dashboard": self.create_overview_dashboard,
            "📊 Perbandingan": self.create_comparison_table,
            "🔍 Analisis Fitur": self.create_feature_analysis,
            "🤖 Prediksi Live": self.create_live_prediction,
            "📜 Riwayat Update": self.create_update_history  # New page
        }
        
        selected_page = st.sidebar.selectbox("Pilih Halaman:", list(pages.keys()))
        
        # Run selected page
        pages[selected_page]()
        
        # Sidebar info - updated
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ℹ️ Informasi")
        st.sidebar.markdown("""
        **Fitur Aplikasi:**
        - 📱 **Kanji**: Pembelajaran karakter Jepang
        - 🗣️ **Kotoba**: Kosakata bahasa Jepang  
        - 📚 **Bunpou**: tata bahasa Jepang
        
        **Data:** 6 aplikasi pembelajaran bahasa Jepang
        
        **Update Real-time:** ✅ Aktif
        """)
        
        # Show last update info
        if hasattr(st.session_state, 'data_updated') and st.session_state.data_updated:
            st.sidebar.success("Data baru saja diupdate! 🎉")

# Run the application
if __name__ == "__main__":
    app = SentimentAnalysisApp()
    app.run()