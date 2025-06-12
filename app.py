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
        """Create live sentiment prediction interface with app selection, feature selection, and data update"""
        st.subheader("🤖 Prediksi Sentimen Real-time")
        
        st.write("Masukkan ulasan aplikasi untuk melihat prediksi sentimen dan update data aplikasi:")
        
        # App and Feature selection
        col1, col2 = st.columns([2, 2])
        
        with col1:
            selected_app = st.selectbox(
                "🎯 Pilih Aplikasi Target:",
                options=list(self.apps_data.keys()),
                help="Pilih aplikasi yang akan menerima komentar ini"
            )
        
        with col2:
            feature_options = {
                'auto': '🤖 Deteksi Otomatis',
                'kanji': '🔤 Kanji - Pembelajaran karakter Jepang',
                'kotoba': '💬 Kotoba - Kosakata bahasa Jepang', 
                'bunpou': '📚 Bunpou - Tata bahasa Jepang'
            }
            
            selected_feature_mode = st.selectbox(
                "🎯 Pilih Fitur Target:",
                options=list(feature_options.keys()),
                format_func=lambda x: feature_options[x],
                help="Pilih fitur spesifik atau biarkan sistem mendeteksi otomatis"
            )
        
        # Show app info
        if selected_app in self.apps_data:
            with st.container():
                st.markdown("### 📱 Info Aplikasi Terpilih")
                app_data = self.apps_data[selected_app]
                
                col1, col2, col3, col4 = st.columns(4)
                
                total_reviews = sum(
                    feature_data.get('positive', 0) + feature_data.get('negative', 0)
                    for feature_data in app_data.values()
                )
                
                total_positive = sum(
                    feature_data.get('positive', 0) for feature_data in app_data.values()
                )
                
                with col1:
                    st.metric("📊 Total Ulasan", total_reviews)
                
                with col2:
                    st.metric("✅ Ulasan Positif", total_positive)
                
                with col3:
                    positive_rate = (total_positive / total_reviews * 100) if total_reviews > 0 else 0
                    st.metric("📈 Tingkat Positif", f"{positive_rate:.1f}%")
                
                with col4:
                    feature_count = len([f for f in app_data.keys() if f in ['kanji', 'kotoba', 'bunpou']])
                    st.metric("🎯 Fitur Tersedia", f"{feature_count}/3")
        
        # Text input with enhanced UI
        st.markdown("### ✍️ Tulis Ulasan Anda")
        
        # Provide examples based on selected feature
        if selected_feature_mode != 'auto':
            feature_examples = {
                'kanji': "Contoh: 'Aplikasi ini sangat membantu untuk belajar kanji, karakternya mudah diingat dan ada sistem pengulangan yang bagus.'",
                'kotoba': "Contoh: 'Kosakata yang disediakan sangat lengkap dan audio pronounciation-nya jelas, membantu saya memperbanyak vocabulary.'",
                'bunpou': "Contoh: 'Penjelasan grammar sangat detail dan mudah dipahami, struktur kalimat dijelaskan dengan baik.'"
            }
            st.info(f"💡 {feature_examples.get(selected_feature_mode, '')}")
        
        user_input = st.text_area(
            "📝 Ulasan:",
            placeholder="Tulis ulasan Anda di sini... (minimal 10 karakter)",
            help="Tulis ulasan yang jujur dan detail tentang pengalaman Anda menggunakan aplikasi",
            max_chars=500,
            height=120
        )
        # Enhanced character counter with visual feedback
        if user_input:
            char_count = len(user_input)
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                if char_count < 10:
                    st.error(f"⚠️ Masukkan minimal 10 karakter lagi ({10 - char_count} karakter tersisa)")
                elif char_count < 50:
                    st.warning(f"✏️ Ulasan cukup pendek ({char_count} karakter) - tambahkan detail untuk hasil yang lebih akurat")
                else:
                    st.success(f"✅ Panjang ulasan: {char_count} karakter - bagus!")
            
            with col2:
                progress = min(char_count / 100, 1.0)
                st.progress(progress, text=f"{char_count}/500")
            
            with col3:
                words = len(user_input.split())
                st.metric("Kata", words)
        
        # Action buttons with improved layout and descriptions
        st.markdown("### 🎮 Aksi")
        
        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
        
        with col1:
            predict_enabled = len(user_input) >= 10
            predict_button = st.button(
                "🔍 Analisis Sentimen",
                disabled=not predict_enabled,
                help="Klik untuk menganalisis sentimen ulasan Anda" if predict_enabled else "Masukkan minimal 10 karakter untuk mengaktifkan analisis",
                type="primary" if predict_enabled else "secondary"
            )
        
        with col2:
            has_prediction = 'last_prediction' in st.session_state and st.session_state.last_prediction
            update_enabled = has_prediction and predict_enabled
            
            if update_enabled:
                update_button = st.button(
                    "💾 Simpan ke Database", 
                    type="secondary",
                    help="Simpan hasil analisis ke database aplikasi"
                )
            else:
                st.button(
                    "💾 Simpan ke Database", 
                    disabled=True, 
                    help="Lakukan analisis sentimen terlebih dahulu"
                )
        
        with col3:
            if st.button("🔄 Reset", help="Bersihkan semua input dan hasil"):
                if 'last_prediction' in st.session_state:
                    del st.session_state.last_prediction
                st.rerun()
        
        with col4:
            if st.button("❓ Help", help="Panduan penggunaan"):
                st.info("""
                **Cara Penggunaan:**
                1. Pilih aplikasi target
                2. Pilih fitur (atau biarkan auto-detect)
                3. Tulis ulasan minimal 10 karakter
                4. Klik 'Analisis Sentimen'
                5. Review hasil, lalu klik 'Simpan ke Database'
                """)
        
        # Perform prediction with enhanced feature handling
        if predict_button and user_input:
            with st.spinner("🔄 Menganalisis sentimen... Mohon tunggu"):
                # Create progress bar for better UX
                progress_bar = st.progress(0)
                
                # Simulate analysis steps
                progress_bar.progress(25, "Memproses teks...")
                time.sleep(0.5)
                
                progress_bar.progress(50, "Menganalisis sentimen...")
                prediction = self.ml_predictor.predict_sentiment(user_input)
                
                progress_bar.progress(75, "Mendeteksi fitur...")
                
                # Override feature detection if specific feature is selected
                if selected_feature_mode != 'auto':
                    prediction['features'] = [selected_feature_mode]
                    prediction['feature_override'] = True
                
                progress_bar.progress(100, "Selesai!")
                time.sleep(0.5)
                progress_bar.empty()
                
                # Store prediction in session state
                st.session_state.last_prediction = {
                    'text': user_input,
                    'app': selected_app,
                    'prediction': prediction,
                    'feature_mode': selected_feature_mode,
                    'timestamp': pd.Timestamp.now().isoformat()
                }
                
                # Show success message
                st.success("✅ Analisis sentimen berhasil! Lihat hasil di bawah.")
        
        # Display prediction results
        if 'last_prediction' in st.session_state and st.session_state.last_prediction:
            self._display_enhanced_prediction_results(st.session_state.last_prediction)
        
        # Update data functionality with confirmation
        if 'last_prediction' in st.session_state and st.session_state.last_prediction:
            if st.button("💾 Konfirmasi Simpan Data", key="update_data_main", type="primary"):
                # Show confirmation dialog
                if st.button("✅ Ya, Simpan Data!", key="confirm_update"):
                    self._update_app_data(st.session_state.last_prediction)

    def _display_enhanced_prediction_results(self, prediction_data):
        """Display enhanced prediction results with better formatting"""
        st.markdown("---")
        st.markdown("## 📊 Hasil Analisis Sentimen")
        
        prediction = prediction_data['prediction']
        selected_app = prediction_data['app']
        feature_mode = prediction_data.get('feature_mode', 'auto')
        
        # Main results with enhanced styling
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            sentiment = prediction['sentiment']
            if sentiment == 'positive':
                sentiment_display = "🟢 POSITIF"
                sentiment_color = "success"
            else:
                sentiment_display = "🔴 NEGATIF"
                sentiment_color = "error"
            
            st.metric(
                "🎯 Sentimen", 
                sentiment_display,
                delta=f"Confidence: {prediction['confidence']:.1f}%"
            )
        
        with col2:
            st.metric(
                "📱 Aplikasi", 
                selected_app,
                delta="Target update"
            )
        
        with col3:
            detected_features = prediction.get('features', [])
            feature_count = len(detected_features)
            
            if feature_mode != 'auto':
                feature_info = f"Manual: {feature_mode.capitalize()}"
            else:
                feature_info = f"Auto: {feature_count} fitur"
            
            st.metric(
                "🎯 Mode Fitur", 
                feature_info,
                delta=f"{', '.join(detected_features) if detected_features else 'Tidak ada'}"
            )
        
        with col4:
            text_quality = "Bagus" if len(prediction_data['text']) > 50 else "Cukup"
            st.metric(
                "📝 Kualitas Teks",
                text_quality,
                delta=f"{len(prediction_data['text'])} karakter"
            )
        
        # Detailed analysis in tabs
        tab1, tab2, tab3 = st.tabs(["📋 Detail Analisis", "🎯 Fitur Terdeteksi", "💡 Saran & Tips"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔍 Informasi Teknis:**")
                st.write(f"• **Metode Deteksi:** {feature_mode.title() if feature_mode != 'auto' else 'Otomatis'}")
                st.write(f"• **Confidence Score:** {prediction['confidence']:.2f}%")
                st.write(f"• **Model:** {prediction.get('method', 'ML Classifier')}")
                st.write(f"• **Waktu Analisis:** {prediction_data['timestamp'][:19]}")
                
                # Confidence level interpretation
                confidence = prediction['confidence']
                if confidence >= 80:
                    conf_level = "🟢 Sangat Tinggi"
                elif confidence >= 60:
                    conf_level = "🟡 Tinggi"
                elif confidence >= 40:
                    conf_level = "🟠 Sedang"
                else:
                    conf_level = "🔴 Rendah"
                
                st.write(f"• **Tingkat Kepercayaan:** {conf_level}")
            
            with col2:
                st.markdown("**📝 Analisis Teks:**")
                text_length = len(prediction_data['text'])
                word_count = len(prediction_data['text'].split())
                
                st.write(f"• **Panjang Teks:** {text_length} karakter")
                st.write(f"• **Jumlah Kata:** {word_count} kata")
                st.write(f"• **Rata-rata Kata:** {text_length/word_count:.1f} karakter/kata" if word_count > 0 else "• **Rata-rata Kata:** N/A")
                
                # Text preview
                preview_text = prediction_data['text'][:100] + "..." if len(prediction_data['text']) > 100 else prediction_data['text']
                st.write(f"• **Preview:** *\"{preview_text}\"*")
        
        with tab2:
            detected_features = prediction.get('features', [])
            
            if detected_features:
                st.success(f"✅ Berhasil mendeteksi {len(detected_features)} fitur:")
                
                for feature in detected_features:
                    feature_info = {
                        'kanji': {
                            'icon': '🔤',
                            'name': 'Kanji',
                            'desc': 'Pembelajaran karakter Jepang',
                            'keywords': ['kanji', 'karakter', 'huruf jepang', 'ideogram']
                        },
                        'kotoba': {
                            'icon': '💬',
                            'name': 'Kotoba',
                            'desc': 'Kosakata bahasa Jepang',
                            'keywords': ['kosakata', 'vocabulary', 'kata', 'kotoba']
                        },
                        'bunpou': {
                            'icon': '📚',
                            'name': 'Bunpou',
                            'desc': 'Tata bahasa Jepang',
                            'keywords': ['grammar', 'tata bahasa', 'bunpou', 'struktur']
                        }
                    }
                    
                    info = feature_info.get(feature, {'icon': '❓', 'name': feature.title(), 'desc': 'Fitur tidak dikenal'})
                    
                    with st.container():
                        st.markdown(f"""
                        **{info['icon']} {info['name']}**
                        - {info['desc']}
                        - Keywords: {', '.join(info.get('keywords', []))}
                        """)
            else:
                st.warning("⚠️ Tidak ada fitur spesifik yang terdeteksi")
                st.info("""
                **💡 Tips untuk deteksi fitur yang lebih baik:**
                - Sebutkan kata kunci seperti: 'kanji', 'kosakata', 'grammar'
                - Atau gunakan mode manual di pilihan fitur
                - Berikan ulasan yang lebih detail dan spesifik
                """)
        
        with tab3:
            st.markdown("### 💡 Saran Berdasarkan Analisis")
            
            confidence = prediction['confidence']
            sentiment = prediction['sentiment']
            
            if confidence >= 80:
                st.success("🎯 **Analisis Sangat Akurat** - Hasil dapat dipercaya dan siap untuk disimpan")
            elif confidence >= 60:
                st.info("👍 **Analisis Cukup Baik** - Hasil dapat diterima dengan catatan kecil")
            else:
                st.warning("⚠️ **Analisis Kurang Akurat** - Pertimbangkan untuk menulis ulasan yang lebih detail")
            
            # Personalized suggestions
            if sentiment == 'positive':
                st.success("""
                ✅ **Ulasan Positif Terdeteksi**
                - Ulasan ini akan meningkatkan rating aplikasi
                - Feedback positif membantu developer
                - Terima kasih atas kontribusi Anda!
                """)
            else:
                st.info("""
                📝 **Ulasan Negatif Terdeteksi**
                - Feedback konstruktif sangat berharga
                - Membantu identifikasi area perbaikan
                - Pastikan kritik bersifat membangun
                """)
        
        # Enhanced preview section
        st.markdown("---")
        st.subheader("👀 Preview Update Database")
        self._show_enhanced_data_update_preview(prediction_data)


    def _show_enhanced_data_update_preview(self, prediction_data):
        """Show enhanced preview of how data will be updated"""
        prediction = prediction_data['prediction']
        selected_app = prediction_data['app']
        detected_features = prediction.get('features', [])
        sentiment = prediction['sentiment']
        feature_mode = prediction_data.get('feature_mode', 'auto')
        
        if not detected_features:
            st.error("❌ **Tidak dapat mengupdate database:** Tidak ada fitur yang terdeteksi")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info("""
                **🔧 Solusi:**
                1. Gunakan mode 'Manual' pada pilihan fitur
                2. Sebutkan kata kunci fitur dalam ulasan
                3. Tulis ulasan yang lebih spesifik
                """)
            
            with col2:
                st.warning("""
                **⚠️ Catatan:**
                - Data tidak akan tersimpan tanpa fitur
                - Sistem memerlukan klasifikasi fitur
                - Gunakan panduan di tab 'Saran & Tips'
                """)
            return
        
        # Current data analysis
        current_data = self.apps_data.get(selected_app, {})
        
        # Enhanced preview table
        st.markdown("### 📊 Perbandingan Data: Sebelum vs Sesudah")
        
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
            
            # Calculate change
            percentage_change = new_percentage - current_percentage
            change_indicator = "📈" if percentage_change > 0 else "📉" if percentage_change < 0 else "➡️"
            
            preview_data.append({
                'Fitur': f"{feature.capitalize()}",
                'Positif (Lama)': current_pos,
                'Negatif (Lama)': current_neg,
                'Total (Lama)': current_total,
                '% Positif (Lama)': f"{current_percentage:.1f}%",
                'Positif (Baru)': new_pos,
                'Negatif (Baru)': new_neg,
                'Total (Baru)': new_total,
                '% Positif (Baru)': f"{new_percentage:.1f}%",
                'Perubahan': f"{change_indicator} {percentage_change:+.1f}%"
            })
        
        df_preview = pd.DataFrame(preview_data)
        st.dataframe(df_preview, use_container_width=True, hide_index=True)
        
        # Impact analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📈 Ringkasan Dampak:**")
            total_reviews_added = len(detected_features)
            st.write(f"• **Total ulasan baru:** +{total_reviews_added}")
            st.write(f"• **Sentimen:** {sentiment.title()}")
            st.write(f"• **Fitur terpengaruh:** {len(detected_features)}")
            st.write(f"• **Mode deteksi:** {feature_mode.title()}")
        
        with col2:
            st.markdown("**🎯 Detail Perubahan:**")
            for feature in detected_features:
                current_total = current_data.get(feature, {}).get('positive', 0) + current_data.get(feature, {}).get('negative', 0)
                st.write(f"• **{feature.capitalize()}:** {current_total} → {current_total + 1} ulasan")
            
            # Confidence indicator
            confidence = prediction['confidence']
            conf_emoji = "🟢" if confidence >= 80 else "🟡" if confidence >= 60 else "🔴"
            st.write(f"• **Kepercayaan:** {conf_emoji} {confidence:.1f}%")
        
        # Final confirmation section
        st.markdown("---")
        st.markdown("### ✅ Konfirmasi Penyimpanan")
        
        changes_summary = f"""
        **🔍 Ringkasan yang akan disimpan:**
        - **Aplikasi:** {selected_app}
        - **Sentimen:** {sentiment.title()} ({prediction['confidence']:.1f}% confidence)
        - **Fitur:** {', '.join([f.capitalize() for f in detected_features])}
        - **Total fitur:** {len(detected_features)} fitur terpengaruh
        - **Metode:** {feature_mode.title()} detection
        """
        
        if sentiment == 'positive':
            st.success(changes_summary)
        else:
            st.info(changes_summary)
        
        # Warning for low confidence
        if prediction['confidence'] < 60:
            st.warning("⚠️ **Perhatian:** Confidence score rendah. Pastikan ulasan sudah sesuai sebelum menyimpan.")



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