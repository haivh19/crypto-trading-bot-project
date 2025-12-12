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

load_dotenv()

# ==========================================
# CẤU HÌNH SIÊU TỐC ĐỘ
# ==========================================
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1m'   # Nến 1 phút để Realtime
SLEEP_TIME = 1     # Chỉ nghỉ 1 giây (Quét liên tục)
QUANTITY_USDT = 0  # 0 = All-in

# ==========================================
# 1. DATA LAYER (Bỏ API FNG cũ)
# ==========================================
class DataLoader:
    def __init__(self):
        # Dùng Testnet hoặc Real tùy file .env, ở đây ta init public để lấy data
        self.exchange = ccxt.binance() 

    def fetch_ohlcv(self, symbol=SYMBOL, timeframe=TIMEFRAME, limit=100):
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return pd.DataFrame()

# ==========================================
# 2. PROCESSING LAYER (TÍNH FNG REALTIME)
# ==========================================
class Processor:
    def add_indicators(self, df):
        if df.empty: return df
        
        # RSI
        rsi = RSIIndicator(close=df['close'], window=14)
        df['rsi'] = rsi.rsi()
        
        # SMA & Distance
        sma = SMAIndicator(close=df['close'], window=20)
        df['sma_20'] = sma.sma_indicator()
        df['dist_sma'] = (df['close'] - df['sma_20']) / df['sma_20']
        
        # Volatility
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=7).std()

        # --- LOGIC MỚI: TỰ TÍNH PANIC VOLUME (VSA) ---
        # Tính Volume trung bình 20 cây nến gần nhất
        df['vol_ma20'] = df['volume'].rolling(window=20).mean()
        
        # Panic Factor: Volume hiện tại gấp mấy lần trung bình?
        # Nếu > 2.0 là đột biến (Bà con đang xả hàng hoặc FOMO mạnh)
        df['vol_spike'] = df['volume'] / df['vol_ma20']
        
        df.dropna(inplace=True)
        return df

    def calculate_realtime_sentiment(self, last_row):
        """
        Tự tính chỉ số Sợ hãi/Tham lam dựa trên Volume và RSI
        Trả về: "PANIC", "GREED", hoặc "NORMAL"
        """
        rsi = last_row['rsi']
        vol_spike = last_row['vol_spike']
        is_red_candle = last_row['close'] < last_row['open']

        # LOGIC BÁN THÁO (PANIC SELL):
        # Giá giảm + Volume nổ gấp 1.5 lần + RSI thấp
        if is_red_candle and vol_spike > 1.5 and rsi < 35:
            return "PANIC" # Cơ hội bắt đáy tuyệt vời!

        # LOGIC FOMO (GREED):
        # Giá tăng + Volume nổ + RSI cao
        if not is_red_candle and vol_spike > 1.5 and rsi > 70:
            return "GREED"
            
        return "NORMAL"

# ==========================================
# 3. MODEL LAYER
# ==========================================
class ModelEngine:
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        self.features = ['rsi', 'dist_sma', 'volatility']

    def predict_prob(self, current_features_df):
        try:
            model = joblib.load(f'{self.model_type}_model.pkl')
            last_row = current_features_df[self.features].iloc[[-1]].values
            prob = model.predict_proba(last_row)[0][1]
            return prob
        except Exception:
            return 0.5

# ==========================================
# 4. STRATEGY LAYER (LOGIC MỚI)
# ==========================================
class StrategyEngine:
    def smart_switch(self, model_type, prob, sentiment, position):
        # Hạ threshold xuống vì nến 1m biến động nhanh
        threshold = 0.55 if model_type == 'xgboost' else 0.52
        
        # 1. BẮT ĐÁY (PANIC BUY) - Ưu tiên số 1
        # Nếu phát hiện bán tháo tập thể (Realtime) -> Mua ngay
        if sentiment == "PANIC": 
            return "BUY_PANIC"
            
        # 2. MUA THEO AI (TREND)
        if prob >= threshold and sentiment != "GREED": 
            return "BUY_TREND"
            
        # 3. BÁN (CẮT LỖ / CHỐT LỜI)
        if position:
            # Nếu AI bảo giảm HOẶC thị trường quá hưng phấn (RSI cao)
            if prob <= 0.45 or sentiment == "GREED":
                return "SELL_EXIT"
            
        return "WAIT"

# ==========================================
# REAL TRADER (TESTNET/REAL)
# ==========================================
class RealTrader:
    def __init__(self):
        api_key = os.getenv('BINANCE_API_KEY')
        secret_key = os.getenv('BINANCE_SECRET')
        if not api_key: raise ValueError("❌ CHƯA CÓ API KEY")

        print("🔌 Đang kết nối Binance...")
        self.exchange = ccxt.binance({
            'apiKey': api_key, 'secret': secret_key,
            'enableRateLimit': True, 'options': {'defaultType': 'spot'}
        })
        self.exchange.set_sandbox_mode(True) # --- QUAN TRỌNG: TESTNET ---
        print("✅ KẾT NỐI TESTNET THÀNH CÔNG!")

    def get_balances(self):
        try:
            bal = self.exchange.fetch_balance()
            return bal['USDT']['free'], bal['BTC']['free']
        except: return 0, 0

    def execute_real_order(self, action, current_price):
        usdt, btc = self.get_balances()
        print(f"   💼 Wallet: {usdt:.1f} USDT | {btc:.5f} BTC")

        if "BUY" in action and usdt > 10:
            # --- SỬA LẠI: CHỪA TIỀN TRẢ PHÍ (BUFFER) ---
            if QUANTITY_USDT == 0:
                # Nếu là All-in, chỉ lấy 98% số dư (để lại 2% lo phí và trượt giá)
                qty_usdt = usdt * 0.98 
            else:
                qty_usdt = QUANTITY_USDT
            # -------------------------------------------

            if qty_usdt > 10:
                # Tính lượng BTC, làm tròn 5 số thập phân để tránh lỗi Precision
                amount_btc = round(qty_usdt / current_price, 5)
                try:
                    self.exchange.create_order(SYMBOL, 'market', 'buy', amount_btc)
                    print(f"   🚀 MUA THÀNH CÔNG! ({action})")
                except Exception as e: print(f"❌ Lỗi Mua: {e}")

        elif "SELL" in action and (btc * current_price) > 10:
            try:
                self.exchange.create_order(SYMBOL, 'market', 'sell', btc)
                print(f"   📉 BÁN THÀNH CÔNG! ({action})")
            except Exception as e: print(f"❌ Lỗi Bán: {e}")

# ==========================================
# MAIN LOOP
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='xgboost')
    args = parser.parse_args()

    print(f"\n⚡ REALTIME VSA BOT ACTIVATED ({args.model.upper()})")
    
    # Init modules
    data = DataLoader()
    proc = Processor()
    model = ModelEngine(args.model)
    strat = StrategyEngine()
    trader = RealTrader()

    try:
        while True:
            # 1. Lấy dữ liệu 1 phút
            df = data.fetch_ohlcv(limit=50) # Lấy ít cho nhanh
            df = proc.add_indicators(df)
            
            last_row = df.iloc[-1]
            curr_price = last_row['close']
            curr_time = datetime.now().strftime("%H:%M:%S")

            # 2. Tính toán các chỉ số
            prob = model.predict_prob(df)
            
            # --- REALTIME SENTIMENT (VSA) ---
            sentiment = proc.calculate_realtime_sentiment(last_row)
            # --------------------------------
            
            # 3. Check Vị thế
            usdt, btc = trader.get_balances()
            in_pos = (btc * curr_price) > 10

            # 4. Ra quyết định
            action = strat.smart_switch(args.model, prob, sentiment, in_pos)

            # 5. In ra màn hình (Gọn gàng)
            vol_status = f"Vol x{last_row['vol_spike']:.1f}"
            print(f"⏱ {curr_time} | ${curr_price:.1f} | AI:{prob:.2f} | {sentiment} ({vol_status}) | 👉 {action}")

            # 6. Trade thật
            if action != "WAIT":
                trader.execute_real_order(action, curr_price)
            
            # 7. Sleep cực ngắn (1 giây)
            time.sleep(SLEEP_TIME)

    except KeyboardInterrupt:
        print("\n🛑 Bot Stopped.")