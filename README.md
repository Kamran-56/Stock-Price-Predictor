# Stock Sense

### Stock Market Prediction using LSTM & News Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![NLTK](https://img.shields.io/badge/NLTK-VADER-green)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

> **BTech Minor Project | 6th Semester**  
> Department of Computer Science and Engineering  
> Islamic University of Science & Technology, Kashmir – 192122

---

##  Project Overview

**Stock Sense** - is an end-to-end deep learning system that predicts the **next trading day's closing price of Apple Inc. (AAPL)** .The system combines historical stock price data with sentiment signals extracted from financial news headlines using Natural Language Processing (NLP).

Unlike traditional machine learning models that treat each day independently, Stock Sense uses a **Long Short-Term Memory (LSTM)** neural network — designed specifically for time-series data — to learn patterns across a 60-day sliding window.

A user-friendly **Streamlit web application** allows users to enter a news headline and receive:

- The VADER sentiment score of the headline
- The predicted next-day Apple stock closing price
- A chart showing the last 60 trading days with the prediction point

---

##  Objectives

- Combine 10 years of Apple stock price data with 139,919 financial news headlines into a single clean dataset
- Build a custom NLP sentiment pipeline using VADER — without relying on any pre-labeled sentiment data
- Engineer 14 input features including technical indicators and daily news sentiment features
- Correctly handle weekend news by remapping Saturday/Sunday articles to the next trading day using weighted aggregation
- Train a 2-layer LSTM model with Dropout regularization on 60-day sliding window sequences
- Evaluate performance using RMSE, MAE, R², and MAPE metrics against a naive baseline
- Deploy a Streamlit web application for real-time stock price prediction

---

##  Dataset

| Dataset | Source | Rows | Period | Columns Used |
|---|---|---:|---|---|
| Apple Stock Prices | Yahoo Finance / Kaggle | 2,585 | Jan 2016 – Apr 2026 | Date, Open, High, Low, Close, Volume |
| Financial News Headlines | Kaggle | 139,919 | Jan 2016 – Apr 2026 | Date, Title |

### Key Facts

- Both datasets are fully clean with zero missing values
- All 139,919 news headlines are used — not just Apple-specific news, because world events such as COVID, interest rate hikes, and trade wars can affect Apple stock
- 85% of trading days have at least one news article on the same calendar date

---

##  System Architecture

```text
Raw Stock Data (CSV) ──────────────────────────────────────────────┐
                                                                  │
                                                          Feature Engineering
                                                          (14 features + Target)
                                                                  │
Raw News Headlines ──► VADER NLP Pipeline ──► 7 Daily Features ──►│
                       (139,919 headlines)                         │
                                                                  ▼
                                                           MinMax Scaling (0-1)
                                                                  │
                                                                  ▼
                                                        60-Day Sliding Window
                                                             Sequences
                                                                  │
                                                                  ▼
                                                           2-Layer LSTM Model
                                                                  │
                                                                  ▼
                                                        Predicted Next-Day Close
                                                                  │
                                                                  ▼
                                                           Streamlit Web GUI
```

---

##  14 Input Features

| Category | Features |
|---|---|
| Price-Based (5) | Close, Open, High, Low, Volume |
| Technical Indicators (5) | MA_7, MA_21, daily_return, price_range, volatility_7 |
| News Sentiment — NLP (7) | news_count, avg_sentiment, max_sentiment, min_sentiment, high_impact_count, medium_impact_count, negative_count |

**Target:** Close price of the next trading day (tomorrow's Close)

---

##  Model Architecture

```text
Input Shape: (60 days × 17 features)
        ↓
LSTM Layer 1 — 64 units, return_sequences=True
        ↓
Dropout Layer 1 — rate=0.2
        ↓
LSTM Layer 2 — 64 units, return_sequences=False
        ↓
Dropout Layer 2 — rate=0.2
        ↓
Dense Layer — 1 unit (output)
        ↓
Predicted Close Price (scaled 0-1 → inverse transformed to USD)
```

### Training Settings

- **Optimizer:** Adam
- **Loss:** Mean Squared Error (MSE)
- **Max Epochs:** 50 (with EarlyStopping, patience=10)
- **Batch Size:** 32
- **Validation Split:** 10%
- **Train/Test Split:** 80% / 20% (chronological — no random shuffling)

---

##  Model Performance

| Metric | Value | Meaning |
|---|---:|---|
| RMSE | $11.30 | Average squared error in USD |
| MAE | $9.59 | On average predictions are $9.59 off |
| R² Score | 0.77 | Model explains 77% of price variation |
| MAPE | 4.07% | On average 4.07% off from actual price |

---

##  How to Run

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Kamran-56/Stock-Sense.git
cd Stock-Sense
```

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Download VADER Lexicon

Run this once:

```python
import nltk
nltk.download('vader_lexicon')
```

### Step 4 — Run the Notebooks

Run the notebooks in this order:

```text
1. Phase1_EDA.ipynb
2. Phase2A_NLP_Pipeline.ipynb
3. Phase2B_Feature_Engineering.ipynb
4. Phase3_Preprocessing.ipynb
5. Phase4_LSTM_Model.ipynb
6. Phase5_Evaluation.ipynb
```

### Step 5 — Launch the Streamlit GUI

```bash
streamlit run app.py
```

The app will open automatically in your browser at:

```text
http://localhost:8501
```

---

##  GUI Features

- **News Headline Input** — Type any financial news headline
- **Sentiment Score** — VADER compound score from -1 to +1
- **Impact Tier** — HIGH / MEDIUM / LOW based on sentiment strength
- **Predicted Price** — Tomorrow's predicted Apple Close price in USD
- **Price Chart** — Last 60 trading days + prediction point plotted
- **Sentiment Explanation** — Plain English explanation of how the headline influenced the prediction

---

## 📁 Project Structure

```text
Stock-Sense/
│
├── README.md
├── requirements.txt
├── .gitignore
├── app.py
│
├── notebooks/
│   ├── Phase1_EDA.ipynb
│   ├── Phase2A_NLP_Pipeline.ipynb
│   ├── Phase2B_Feature_Engineering.ipynb
│   ├── Phase3_Preprocessing.ipynb
│   ├── Phase4_LSTM_Model.ipynb
│   └── Phase5_Evaluation.ipynb
│
├── data/
│   ├── apple_stock_2016to2026.csv
│   ├── final_dataset.csv
│   ├── news_daily_features.csv
│   └── stock_news_2016to2026.csv
│
├── model/
│   ├── best_model.keras
│   ├── lstm_model.keras
│   ├── feature_columns.pkl
│   ├── feature_scaler.pkl
│   ├── target_scaler.pkl
│   ├── X_train.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   └── y_test.npy
│
└── matplotlib_insights/
    └── *.png
```

---

##  Tech Stack

| Category | Tool | Version |
|---|---|---|
| Language | Python | 3.10 |
| Deep Learning | TensorFlow / Keras | 2.13 |
| NLP | NLTK (VADER) | 3.8.1 |
| Data Processing | Pandas | 2.0.0 |
| Data Processing | NumPy | 1.24.0 |
| Visualization | Matplotlib | 3.7.0 |
| Visualization | Seaborn | 0.12.0 |
| ML Utilities | Scikit-learn | 1.3.0 |
| Web GUI | Streamlit | 1.28.0 |

---

## ⚠️ Disclaimer

This project is built purely for **academic and educational purposes** as part of a BTech minor project at IUST.

The predictions generated by this system should **NOT** be used for real investment or trading decisions. Stock markets are inherently unpredictable and no model can guarantee accurate future prices.
