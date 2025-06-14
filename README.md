https://nihongonavigator.streamlit.app/

# 🇯🇵 Japanese App Sentiment Analysis

Sistem Analisis Sentimen untuk Aplikasi Pembelajaran Bahasa Jepang. Proyek ini menggunakan Machine Learning dan visualisasi data untuk menganalisis sentimen pengguna dari berbagai aplikasi belajar bahasa Jepang seperti Mazii, Obenkyo, HeyJapan, JA Sensei, Migii JLPT, dan Kanji Study.

---

## 📁 Struktur Proyek

```
japanese_app_sentiment_analysis/
│
├── app.py                        # Aplikasi utama Streamlit
├── requirements.txt              # Daftar dependensi
├── README.md                     # Dokumentasi proyek
│
├── data/
│   ├── raw/                      # Data mentah hasil scraping/analisis awal
│   │   ├── hasil_sentimen_mazii_agregat.json
│   │   ├── hasil_sentimen_obenkyo_agregat.json
│   │   ├── hasil_sentimen_heyjapan_agregat.json
│   │   ├── hasil_sentimen_jasensei_agregat.json
│   │   ├── hasil_sentimen_migiijlpt_agregat.json
│   │   └── hasil_sentimen_kanjistudy_agregat.json
│   └── processed/
│       └── aggregated_data.json  # Data yang telah diproses
│
├── models/
│   ├── sentiment_model.pkl       # Model ML yang telah dilatih
│   ├── vectorizer.pkl            # Vectorizer teks
│   └── model_training.py         # Script pelatihan model
│
├── src/
│   ├── __init__.py
│   ├── data_processor.py         # Fungsi pemrosesan data
│   ├── visualization.py          # Fungsi visualisasi dan grafik
│   ├── ml_predictor.py           # Fungsi prediksi menggunakan ML
│   └── utils.py                  # Fungsi utilitas umum
│
├── assets/
│   ├── style.css                 # Styling khusus untuk Streamlit
│   └── images/
│       └── logo.png              # Logo aplikasi
│
├── notebooks/
│   ├── data_exploration.ipynb    # Notebook eksplorasi data
│   └── model_development.ipynb   # Notebook pelatihan model
│
└── tests/
    ├── __init__.py
    ├── test_data_processor.py
    └── test_ml_predictor.py
```

---

## ⚙️ Cara Menjalankan

### 1. Clone / Buat Direktori Proyek

```bash
git clone https://github.com/topiqnurrm/Nihongo_Navigator.git
cd japanese_app_sentiment_analysis
```

### 2. Buat Virtual Environment

```bash
python -m venv venv
# Untuk Windows:
venv\Scripts\activate
# Untuk Linux/Mac:
source venv/bin/activate
```

### 3. Install Dependensi

```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi

```bash
streamlit run app.py
```

---

## 🔑 Fitur Utama

### 1. **Dashboard Overview**
- Tabel perbandingan fitur antar aplikasi
- Persentase sentimen keseluruhan
- Filter dan sortir interaktif

### 2. **Feature Analysis**
- Analisis mendalam per fitur: Kanji, Kotoba, Bunpou
- Sistem ranking berdasarkan fitur pilihan pengguna
- Grafik dan visualisasi menarik

### 3. **Live Prediction**
- Input teks untuk prediksi sentimen real-time
- Ekstraksi fitur dan klasifikasi secara langsung

### 4. **Data Export**
- Ekspor hasil filter ke file CSV
- Laporan otomatis
- Unduh grafik sebagai gambar

---

## 🧰 Teknologi yang Digunakan

| Komponen     | Teknologi                              |
|--------------|-----------------------------------------|
| Frontend     | Streamlit                              |
| Backend      | Python                                 |
| ML Libraries | Scikit-learn, Pandas, NumPy            |
| Visualisasi  | Plotly, Matplotlib                     |
| Penyimpanan  | JSON, CSV                              |
| Model        | Pickle / Joblib                        |

---

## 📌 Catatan Tambahan

- Dataset yang digunakan merupakan hasil agregasi review pengguna dari Play Store untuk masing-masing aplikasi.
- Model Machine Learning berbasis klasifikasi sentimen menggunakan TF-IDF dan algoritma Naive Bayes.
