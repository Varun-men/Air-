# 🌫️ Delhi Air Pollution Monitoring & AQI Prediction

A professional, interactive Streamlit dashboard for monitoring and predicting air quality in **New Delhi, India**, powered by the OpenAQ v3 API and a Random Forest machine learning model.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.9+
- pip

### 2. Install dependencies
```bash
cd air-pollution-monitor
pip install -r requirements.txt
```

### 3. Set your API key (already configured)
The `.env` file is pre-configured with the OpenAQ API key:
```
OPENAQ_API_KEY=<your_key>
```

### 4. Run the app
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📸 Dashboard Sections

| Page | Description |
|------|-------------|
| 🏠 **Home** | Current AQI, 4 metric cards, AQI timeline & monthly heatmap |
| 🔍 **Data Explorer** | Date range filter, pollutant selector, data table, CSV download |
| 📈 **Pollution Trends** | Seasonal radar, bar charts, AQI distribution, correlation heatmap |
| 🤖 **ML Predictor** | Model metrics, feature importance, AQI prediction form with health advisory |
| ℹ️ **About** | Methodology, data sources, tech stack |

---

## 📊 Data Source

- **Live Mode**: Fetches from [OpenAQ v3 API](https://openaq.org) — real government monitoring station data
- **Demo Mode**: Toggle "Use Demo Data" in the sidebar to use a 2-year synthetic dataset with realistic Delhi seasonal patterns

### Getting an OpenAQ API Key
1. Visit [explore.openaq.org](https://explore.openaq.org)
2. Create a free account
3. Copy your API key to `.env`

---

## 📁 Project Structure

```
air-pollution-monitor/
├── app.py                  # Main Streamlit dashboard (5 pages)
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .env                    # API key (not committed to git)
├── data/
│   ├── __init__.py
│   ├── fetch_data.py       # OpenAQ v3 REST API fetcher
│   └── sample_data.py      # Synthetic Delhi data generator (fallback)
├── analysis/
│   ├── __init__.py
│   ├── preprocess.py       # India CPCB AQI calculation + data cleaning
│   └── trends.py           # Monthly/seasonal/YoY trend analysis
├── models/
│   ├── __init__.py
│   ├── ml_model.py         # Random Forest training + prediction + evaluation
│   └── rf_model.joblib     # Saved model (auto-created on first run)
└── visualizations/
    ├── __init__.py
    └── charts.py           # All Plotly chart functions (dark theme)
```

---

## 🧮 Methodology

### India AQI Calculation (CPCB Standard)
1. Compute 24-hour average concentrations for: PM2.5, PM10, NO₂, O₃, CO, SO₂
2. For each pollutant, compute a **sub-index** via linear interpolation between official CPCB breakpoints
3. **Final AQI = maximum sub-index** (the dominant pollutant drives the overall AQI)

### Machine Learning
- **Algorithm**: Random Forest Regressor (150 trees, max depth 10)
- **Features**: Pollutant concentrations + temporal features (month, day of week, season)
- **Validation**: 80/20 split + 5-fold cross-validation
- **Baseline**: Linear Regression comparison

### Seasonal Patterns (Delhi)
| Season | Months | PM2.5 Range | Key Driver |
|--------|--------|-------------|------------|
| Winter | Dec–Feb | 150–300+ µg/m³ | Temperature inversions, stubble burning |
| Summer | Mar–Jun | 80–150 µg/m³ | Dust, heat |
| Monsoon | Jul–Sep | 30–80 µg/m³ | Rain washout |
| Autumn | Oct–Nov | 100–250 µg/m³ | Post-monsoon recovery |

---

## 🛠️ Tech Stack

| Component | Library/Service |
|-----------|----------------|
| Dashboard | Streamlit ≥ 1.32 |
| Data API | OpenAQ v3 REST |
| Data Processing | Pandas, NumPy |
| ML | scikit-learn (Random Forest, Linear Regression) |
| Visualizations | Plotly |
| Model Persistence | Joblib |
| Environment | python-dotenv |

---

## ⚠️ Notes
- The app **automatically falls back** to synthetic demo data if the API is unreachable
- The ML model **auto-trains** on first launch if no saved model exists
- Data is **cached** for 1 hour to avoid excessive API calls
- All AQI calculations follow the official **India CPCB methodology**

---

*🎓 Student project — Environmental Data Science | Delhi, India*
