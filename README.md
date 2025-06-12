Struktur Proyek Analisis Sentimen Aplikasi Bahasa Jepang
japanese_app_sentiment_analysis/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Dependencies
├── README.md                       # Project documentation
│
├── data/
│   ├── raw/
│   │   ├── hasil_sentimen_mazii_agregat.json
│   │   ├── hasil_sentimen_obenkyo_agregat.json
│   │   ├── hasil_sentimen_heyjapan_agregat.json
│   │   ├── hasil_sentimen_jasensei_agregat.json
│   │   ├── hasil_sentimen_migiijlpt_agregat.json
│   │   └── hasil_sentimen_kanjistudy_agregat.json
│   └── processed/
│       └── aggregated_data.json
│
├── models/
│   ├── sentiment_model.pkl         # Trained ML model
│   ├── vectorizer.pkl              # Text vectorizer
│   └── model_training.py           # Model training script
│
├── src/
│   ├── __init__.py
│   ├── data_processor.py           # Data processing functions
│   ├── visualization.py            # Chart and graph functions
│   ├── ml_predictor.py             # ML prediction functions
│   └── utils.py                    # Utility functions
│
├── assets/
│   ├── style.css                   # Custom CSS styling
│   └── images/
│       └── logo.png
│
├── notebooks/
│   ├── data_exploration.ipynb      # Data analysis notebook
│   └── model_development.ipynb     # Model training notebook
│
└── tests/
    ├── __init__.py
    ├── test_data_processor.py
    └── test_ml_predictor.py
Setup Instructions

Clone/Create Project Directory:

bashmkdir japanese_app_sentiment_analysis
cd japanese_app_sentiment_analysis

Create Virtual Environment:

bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install Dependencies:

bashpip install -r requirements.txt

Run Application:

bashstreamlit run app.py
Key Features
1. Dashboard Overview

Comparison table of all applications by features
Overall sentiment percentages
Interactive filtering and sorting

2. Feature Analysis

Detailed breakdown by feature (kanji, kotoba, bunpou)
Ranking system based on user-selected features
Visual charts and graphs

3. Live Prediction

Text input for custom review analysis
Real-time sentiment prediction
Feature extraction and classification

4. Data Export

Export filtered results to CSV
Generate reports
Download charts as images

Technology Stack

Frontend: Streamlit
Backend: Python
ML Libraries: Scikit-learn, Pandas, NumPy
Visualization: Plotly, Matplotlib
Data Storage: JSON, CSV files
Model Persistence: Pickle/Joblib