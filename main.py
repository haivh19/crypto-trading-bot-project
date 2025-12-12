import os
import time
import argparse
import joblib
import requests
import numpy as np
import pandas as pd
import ccxt
from datetime import datetime, timedelta

# ML Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

# TA Lib
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator

# ==========================================
# CẤU HÌNH TRAINING (QUAN TRỌNG)
# ==========================================
# Đổi thành '1m' để khớp với bot Realtime
TRAIN_TIMEFRAME = '1m' 
# Lấy 5000 cây nến phút (khoảng 3-4 ngày dữ liệu) để học
TRAIN_LIMIT = 5000 

class DataLoader:
    def __init__(self):
        self.exchange = ccxt.binance()

    def fetch_ohlcv(self, symbol='BTC/USDT', timeframe=TRAIN_TIMEFRAME, limit=TRAIN_LIMIT):
        print(f"📥 Đang tải {limit} cây nến '{timeframe}' để Train...")
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"Error: {e}")
            return pd.DataFrame()

    def fetch_fng(self):
        try:
            r = requests.get('https://api.alternative.me/fng/')
            data = r.json()
            return int(data['data'][0]['value'])
        except: return 50

class Processor:
    def add_indicators(self, df):
        if df.empty: return df
        
        # 1. RSI
        rsi = RSIIndicator(close=df['close'], window=14)
        df['rsi'] = rsi.rsi()

        # 2. SMA
        sma = SMAIndicator(close=df['close'], window=20)
        df['sma_20'] = sma.sma_indicator()
        df['dist_sma'] = (df['close'] - df['sma_20']) / df['sma_20']

        # 3. Volatility
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=7).std()

        # --- QUAN TRỌNG CHO VSA: Volume Spike ---
        df['vol_ma20'] = df['volume'].rolling(window=20).mean()
        df['vol_spike'] = df['volume'] / df['vol_ma20']
        # ----------------------------------------

        # 4. Target: Giá đóng cửa cây sau > giá đóng cửa cây trước
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

        df.dropna(inplace=True)
        return df

class ModelEngine:
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        # Thêm vol_spike vào đặc trưng để model học được VSA
        self.features = ['rsi', 'dist_sma', 'volatility', 'vol_spike']

    def train(self, df):
        print(f"🧠 Đang Train Model: {self.model_type.upper()} trên dữ liệu {TRAIN_TIMEFRAME}...")
        X = df[self.features].values
        y = df['target'].values

        # Chia dữ liệu: 80% học, 20% thi
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        if self.model_type == 'rf':
            self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            self.model.fit(X_train, y_train)
            
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, eval_metric='logloss')
            self.model.fit(X_train, y_train)

        # Lưu não lại
        joblib.dump(self.model, f'{self.model_type}_model.pkl')
        
        # Đánh giá
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"✅ Train Xong! Độ chính xác trên nến phút: {acc:.2f}")

    def predict_prob(self, current_features_df):
        try:
            model = joblib.load(f'{self.model_type}_model.pkl')
            last_row = current_features_df[self.features].iloc[[-1]].values
            prob = model.predict_proba(last_row)[0][1]
            return prob
        except: return 0.5

# (Phần Strategy và PaperTrader giữ nguyên hoặc lược bỏ nếu chỉ dùng file này để train)
# Ở đây mình giữ logic chính để file này gọn nhẹ chuyên cho Train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Chỉ cần mode train là đủ cho file này
    parser.add_argument('--mode', type=str, default='train', help='train')
    parser.add_argument('--model', type=str, default='xgboost', help='rf / xgboost')
    args = parser.parse_args()

    data_loader = DataLoader()
    processor = Processor()
    model_engine = ModelEngine(args.model)
    
    if args.mode == 'train':
        # 1. Tải dữ liệu PHÚT
        df = data_loader.fetch_ohlcv()
        # 2. Xử lý chỉ báo
        df = processor.add_indicators(df)
        # 3. Train và lưu file .pkl
        model_engine.train(df)