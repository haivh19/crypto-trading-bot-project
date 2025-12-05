## 📊 Performance Comparison

We conducted a comprehensive backtest from **Jan 1, 2023 to Dec 5, 2025** with an initial capital of **$10,000**.

| Strategy | Total Return (PnL) | Max Drawdown | Win Rate | Verdict |
| :--- | :---: | :---: | :---: | :--- |
| **Buy & Hold** | +452.2% | -28.14% | N/A | High Risk, Passive. |
| **Random Forest** | +437.2% | -27.07% | 52% | Conservative. Misses small trends. |
| **XGBoost** | **+445.8%** | **-26.50%** | 55% | **Best Risk-Adjusted Return.** |
| **LSTM (Deep Learning)** | **+462.5%** | -29.10% | **59%** | **Highest Profitability.** |

### Conclusion
* **Best for Safety:** XGBoost (Lowest Drawdown).
* **Best for Profit:** LSTM (Outperformed Buy & Hold).

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

# Train LSTM
python main.py --mode train --model lstm
