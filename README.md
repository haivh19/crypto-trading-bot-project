# 🤖 Hybrid AI Crypto Trading Bot

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-red?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> **University Project:** Blockchain Technology Course (UIT - VNU-HCM)  
> **Topic:** Comparative Analysis of Random Forest, XGBoost, and LSTM Strategies for Bitcoin Trading.

---

## 📖 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [The "Smart Switch" Strategy](#-the-smart-switch-strategy)
- [Performance Comparison](#-performance-comparison)
- [Installation](#-installation)
- [Usage](#-usage)
- [Contributors](#-contributors)

---

## 🔎 Overview

This project aims to solve the volatility problem in the Cryptocurrency market (specifically BTC/USDT) by implementing a **Hybrid Algorithmic Trading Bot**. 

Unlike traditional bots that rely solely on technical indicators, our system combines **Machine Learning** (to predict trends) with **Sentiment Analysis** (Fear & Greed Index) to optimize entry and exit points. The project evaluates three different AI architectures:
1.  **Random Forest:** Baseline model (Stable).
2.  **XGBoost:** Gradient Boosting (Sensitive to micro-movements).
3.  **LSTM (Deep Learning):** Recurrent Neural Network (Captures long-term sequences).

---

## 🚀 Key Features

* **Multi-Engine Support:** Pluggable architecture allowing seamless switching between RF, XGBoost, and LSTM models.
* **Sentiment Integration:** Real-time fetching of the *Alternative.me Crypto Fear & Greed Index*.
* **Advanced Feature Engineering:** Computes RSI, SMA Distance, Volatility (7-day), and Normalized Volume.
* **Risk Management:** Automated stop-loss logic via the "Smart Switch" algorithm.
* **Backtesting Framework:** Robust simulation engine covering the 2023-2025 period with transaction fee modeling.

---

## 🏗 System Architecture

The bot is designed with a 4-Layer Architecture to ensure modularity and scalability:

1.  **Data Layer:** Ingests OHLCV data via Binance API (CCXT) and Sentiment data.
2.  **Processing Layer:** Handles ETL, normalization, and feature extraction.
3.  **Model Layer:** The AI Core (RF / XGBoost / LSTM) predicting probability $P(Up)$.
4.  **Strategy Layer:** The decision engine applying the Smart Switch logic.

![System Architecture](system_architecture.png)

---

## 🛡 The "Smart Switch" Strategy

This is the core heuristic algorithm that governs the bot's behavior. It prioritizes capital preservation over profit chasing.

**Logic Breakdown:**
* **Panic Buy:** If the market is in *Extreme Fear* (Index $\le$ 20), ignore AI and Buy (Mean Reversion).
* **Trend Buy:** If AI Confidence > Threshold (e.g., 65%), Buy.
* **Safety Exit:** If AI predicts a drop AND Market Sentiment is not oversold, Sell to preserve capital.

```python
def smart_switch(model_type, prob, fng_index, position):
    # Dynamic Thresholds based on Model Sensitivity
    threshold = 0.70 if model_type == 'LSTM' else 0.65
    
    # --- ENTRY RULES ---
    # 1. Catch the Panic Dip (Contrarian)
    if fng_index <= 20: 
        return "BUY_PANIC"
        
    # 2. Follow the Trend (AI Signal)
    if prob >= threshold:
        return "BUY_TREND"
        
    # --- EXIT RULES ---
    if position:
        # Only sell if AI is bearish AND we are not at the bottom
        if prob <= 0.40 and fng_index > 30:
            return "SELL_EXIT"
        else:
            return "HOLD" # HODL through the dip if already oversold
            
    return "WAIT"
```
## 📊 Performance Comparison

We conducted a comprehensive backtest from **Jan 1, 2023 to Dec 5, 2025** with an initial capital of **$10,000**.

| Strategy | Total Return (PnL) | Max Drawdown | Win Rate | Verdict |
| :--- | :---: | :---: | :---: | :--- |
| **Buy & Hold** | +452.2% | -28.14% | N/A | High Risk, Passive. |
| **Random Forest** | +437.2% | -27.07% | 52% | Conservative. Misses small trends. |
| **XGBoost** | **+445.8%** | **-26.50%** | 55% | **Best Risk-Adjusted Return.** |


### Conclusion
* **Best for Safety:** XGBoost (Lowest Drawdown).
* **Best for Profit:** Buy & Hold.

---

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/haivh19/crypto-trading-bot-project.git](https://github.com/haivh19/crypto-trading-bot-project.git)
    cd crypto-trading-bot-project
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Setup Environment Variables (Optional):**
    Create a `.env` file for Binance API keys (required only for live data fetching).
    ```env
    BINANCE_API_KEY=your_api_key
    BINANCE_SECRET=your_secret_key
    ```

---

## 💻 Usage

### 1. Training the Models
To train a specific model on historical data (2018-2022):

```bash
# Train XGBoost
python main.py --mode train --model xgboost

# Train Random Forest
python main.py --mode train --model rf
