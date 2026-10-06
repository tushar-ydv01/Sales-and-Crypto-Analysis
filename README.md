# NexusIQ | Enterprise Sales & Crypto Market Intelligence

An end-to-end analytics and machine learning forecasting platform combining multi-channel sales performance, cryptocurrency technical analysis, NLP news sentiment, and predictive forecasting.

---

## 🚀 Key Features

### 1. 📊 Executive Command Center
- Real-time KPI ribbons: Gross Revenue, Gross Profit Margins, Crypto Payment Volume Share, and 24h Crypto Market Action.
- Macro market momentum and composite **Fear & Greed Index** (0–100).
- Unified revenue velocity and cross-asset correlation.

### 2. 🪙 Crypto Markets & Technical Analysis
- Real-time OHLCV market feeds with fallback synthetic data generator for offline resilience.
- Support for top assets: **BTC, ETH, SOL, BNB, XRP, ADA, DOGE, AVAX**.
- Interactive Candlestick charts with overlay indicators:
  - **Moving Averages:** SMA 20, SMA 50, SMA 200, EMA 12, EMA 26
  - **Oscillators & Volatility:** RSI (14), MACD & Signal Histogram, Bollinger Bands, ATR, Stochastic (K/D)
  - **Statistical Metrics:** Annualized Volatility, Sharpe Ratio, Maximum Drawdown

### 3. 🤖 Machine Learning Price Forecasting
- Predictive models: **Random Forest, Gradient Boosting, XGBoost, Ridge Regression**.
- Multi-step forward price projections (7 to 60 days) with 90% confidence bands.
- Live model evaluation metrics: **MAE, RMSE, R², MAPE, and Directional Hit Ratio**.
- Top predictor feature importance rankings.

### 4. 📈 Multi-Channel Sales Analytics & Forecasting
- Multi-dimensional revenue analysis across Product Categories, Regions, Customer Tiers, and Payment Channels (Fiat vs. BTC/ETH/USDT).
- Forward revenue projections with seasonality and day-of-week multipliers.
- Automated **RFM (Recency, Frequency, Monetary) Customer Segmentation** (Champions, Loyal, At Risk, Hibernating).

### 5. 📰 Market Sentiment & NLP Engine
- Real-time financial RSS aggregation (CoinDesk, CoinTelegraph, Decrypt, Yahoo Finance).
- Cryptocurrency-tuned sentiment analysis and polarity scoring (-1.0 to +1.0).
- Market Fear & Greed classification with driver decomposition.

### 6. 📁 Custom Data Studio
- Upload any custom Sales transactions or Crypto OHLCV CSV file.
- Automatic column detection, descriptive statistics, and interactive relationship visualizer.
- Export pre-generated enterprise benchmark datasets.

---

## 🛠️ Project Structure

```
sales_and_cypto_analysis/
├── app.py                          # Streamlit Interactive Web Application
├── test_pipeline.py                # Automated Integration Test Suite
├── requirements.txt                # Project Dependencies
├── README.md                       # Documentation
└── src/
    ├── data/
    │   ├── crypto_fetcher.py       # Crypto OHLCV data loader & resilience generator
    │   ├── sales_data_generator.py # Multi-channel sales synthesizer & RFM engine
    │   └── news_scraper.py         # Financial & crypto RSS news scraper
    ├── nlp/
    │   └── sentiment_analyzer.py   # Domain-tuned NLP sentiment & Fear/Greed scoring
    ├── models/
    │   ├── crypto_forecaster.py    # ML price forecasting & confidence bounds
    │   └── sales_forecaster.py     # Revenue time-series & driver forecasting
    └── utils/
        ├── indicators.py           # Technical indicators (SMA, EMA, RSI, MACD, BB)
        └── visualizations.py       # Plotly glassmorphic interactive charts
```

---

## ⚡ Quickstart & Running Locally

1. **Install Dependencies (if not already installed):**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Pipeline Verification Tests:**
   ```bash
   python test_pipeline.py
   ```

3. **Launch Streamlit Dashboard:**
   ```bash
   streamlit run app.py
   ```

4. Open your browser at `http://localhost:8501`.
