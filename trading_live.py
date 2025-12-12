import os
import time
import argparse
import joblib
import numpy as np
import pandas as pd
import ccxt
from datetime import datetime
from dotenv import load_dotenv

# ML & TA Libraries
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator

load_dotenv(override=True)
# ==========================================
# CẤU HÌNH
# ==========================================
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1m'
SLEEP_TIME = 1
QUANTITY_USDT = 0 
INITIAL_CAPITAL = 21524 # <--- GIẢ ĐỊNH VỐN GỐC LÀ 10.000 (để tính lời lỗ)

# ==========================================
# 1. DATA LAYER
# ==========================================
class DataLoader:
    def __init__(self):
        self.exchange = ccxt.binance() 

    def fetch_ohlcv(self, symbol=SYMBOL, timeframe=TIMEFRAME, limit=100):
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            return pd.DataFrame()

# ==========================================
# 2. PROCESSING LAYER (VSA)
# ==========================================
class Processor:
    def add_indicators(self, df):
        if df.empty: return df
        rsi = RSIIndicator(close=df['close'], window=14)
        df['rsi'] = rsi.rsi()
        sma = SMAIndicator(close=df['close'], window=20)
        df['sma_20'] = sma.sma_indicator()
        df['dist_sma'] = (df['close'] - df['sma_20']) / df['sma_20']
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=7).std()

        df['vol_ma20'] = df['volume'].rolling(window=20).mean()
        df['vol_spike'] = df['volume'] / df['vol_ma20']
        
        df.dropna(inplace=True)
        return df

    def calculate_realtime_sentiment(self, last_row):
        rsi = last_row['rsi']
        vol_spike = last_row['vol_spike']
        is_red_candle = last_row['close'] < last_row['open']

        if is_red_candle and vol_spike > 1.5 and rsi < 35: return "PANIC"
        if not is_red_candle and vol_spike > 1.5 and rsi > 70: return "GREED"
        return "NORMAL"

# ==========================================
# 3. MODEL LAYER
# ==========================================
class ModelEngine:
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        self.features = ['rsi', 'dist_sma', 'volatility', 'vol_spike']

    def predict_prob(self, current_features_df):
        try:
            model = joblib.load(f'{self.model_type}_model.pkl')
            last_row = current_features_df[self.features].iloc[[-1]].values
            prob = model.predict_proba(last_row)[0][1]
            return prob
        except: return 0.5

# ==========================================
# 4. STRATEGY LAYER
# ==========================================
class StrategyEngine:
    def smart_switch(self, model_type, prob, sentiment, position):
        threshold = 0.55 if model_type == 'xgboost' else 0.52
        
        # 1. BẮT ĐÁY (PANIC BUY)
        # Chỉ mua khi chưa có hàng (not position)
        if sentiment == "PANIC" and not position: 
            return "BUY_PANIC"
            
        # 2. MUA THEO AI (TREND)
        if prob >= threshold and sentiment != "GREED": 
            if not position:
                return "BUY_TREND"  # Chưa có hàng thì Mua
            else:
                return "HOLD"       # Có hàng rồi thì Gồng lời (quan trọng!)
            
        # 3. BÁN (CẮT LỖ / CHỐT LỜI)
        if position:
            # Nếu AI bảo giảm HOẶC thị trường quá hưng phấn
            if prob <= 0.45 or sentiment == "GREED":
                return "SELL_EXIT"
            else:
                return "HOLD" # Nếu AI vẫn ngon (vd 0.60) thì cứ giữ tiếp
            
        return "WAIT"

# ==========================================
# REAL TRADER
# ==========================================
class RealTrader:
    def __init__(self):
        api_key = os.getenv('BINANCE_API_KEY')
        secret_key = os.getenv('BINANCE_SECRET')
        if not api_key: raise ValueError("CHƯA CÓ API KEY")

        print("🔌 Đang kết nối Binance...")
        self.exchange = ccxt.binance({
            'apiKey': api_key, 'secret': secret_key,
            'enableRateLimit': True, 'options': {'defaultType': 'spot'}
        })
        self.exchange.set_sandbox_mode(True) # TESTNET
        print("KẾT NỐI THÀNH CÔNG!")

    def get_balances(self):
        try:
            bal = self.exchange.fetch_balance()
            return bal['USDT']['free'], bal['BTC']['free']
        except: return 0, 0

    def get_total_equity(self, current_price):
        """Tính tổng tài sản quy ra USDT"""
        usdt, btc = self.get_balances()
        return usdt + (btc * current_price)

    def execute_real_order(self, action, current_price):
        usdt, btc = self.get_balances()
        
        if "BUY" in action and usdt > 10:
            qty_usdt = usdt * 0.98 if QUANTITY_USDT == 0 else QUANTITY_USDT
            if qty_usdt > 10:
                amount_btc = round(qty_usdt / current_price, 5)
                try:
                    self.exchange.create_order(SYMBOL, 'market', 'buy', amount_btc)
                    print(f" MUA THÀNH CÔNG! ({action})")
                except Exception as e: print(f"Lỗi Mua: {e}")

        elif "SELL" in action and (btc * current_price) > 10:
            try:
                self.exchange.create_order(SYMBOL, 'market', 'sell', btc)
                print(f"BÁN THÀNH CÔNG! ({action})")
            except Exception as e: print(f"Lỗi Bán: {e}")

# ==========================================
# MAIN LOOP
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='xgboost')
    args = parser.parse_args()

    print(f"REALTIME VSA BOT ACTIVATED ({args.model.upper()})")
    
    # Init modules
    data = DataLoader()
    proc = Processor()
    model = ModelEngine(args.model)
    strat = StrategyEngine()
    trader = RealTrader()

    # Màu sắc cho Terminal
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"

    try:
        while True:
            df = data.fetch_ohlcv(limit=50)
            df = proc.add_indicators(df)
            last_row = df.iloc[-1]
            curr_price = last_row['close']
            curr_time = datetime.now().strftime("%H:%M:%S")

            prob = model.predict_prob(df)
            sentiment = proc.calculate_realtime_sentiment(last_row)
            
            usdt, btc = trader.get_balances()
            in_pos = (btc * curr_price) > 10

            # --- TÍNH LỢI NHUẬN (PnL) ---
            equity = trader.get_total_equity(curr_price)
            pnl = equity - INITIAL_CAPITAL
            pnl_color = GREEN if pnl >= 0 else RED
            pnl_str = f"{pnl_color}{pnl:+.2f}${RESET}"
            # ----------------------------

            action = strat.smart_switch(args.model, prob, sentiment, in_pos)

            vol_status = f"Vol x{last_row['vol_spike']:.1f}"
            
            # In ra dòng trạng thái bao ngầu
            print(f"{curr_time} | Equity: ${equity:.1f} ({pnl_str}) | AI:{prob:.2f} | {sentiment} | {action}")

            if action != "WAIT":
                trader.execute_real_order(action, curr_price)
            
            time.sleep(SLEEP_TIME)

    except KeyboardInterrupt:
        print("\nBot Stopped.")