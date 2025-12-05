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
3.  **Model Layer:** The AI Core (RF / XGBoost / LSTM) predicting probability P(Up).
4.  **Strategy Layer:** The decision engine applying the Smart Switch logic.

![System Architecture](system_architecture.png)

---

## 🛡 The "Smart Switch" Strategy

This is the core heuristic algorithm that governs the bot's behavior. It prioritizes capital preservation over profit chasing.

**Logic Breakdown:**
* **Panic Buy:** If the market is in *Extreme Fear* (Index <= 20), ignore AI and Buy (Mean Reversion).
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
