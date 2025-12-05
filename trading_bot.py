#!/usr/bin/env python3
"""
Live Crypto Trading Bot - Production Ready
"""

import ccxt
import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import json
import requests
import time
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

# ============ CONFIGURATION ============
API_KEY = 'your_binance_api_key_here'
API_SECRET = 'your_binance_secret_here'
SYMBOL = 'BTC/USDT'
TRADE_AMOUNT_USD = 100  # Amount per trade
USE_TESTNET = True  # ALWAYS start with True!
CHECK_INTERVAL = 3600  # Check every hour (in seconds)

# Risk Management
MAX_DRAWDOWN_PERCENT = 10  # Stop bot if loss exceeds 10%
MAX_POSITION_SIZE = 0.5  # Max 50% of capital in one position

# ============ LOAD MODELS ============
print("📦 Loading trained models...")

with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('bot_config.json', 'r') as f:
    config = json.load(f)
    FEATURE_COLS = config['feature_cols']

print("✅ Models loaded successfully!")

# ============ INITIALIZE EXCHANGE ============
if USE_TESTNET:
    exchange = ccxt.binance({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'},
        'urls': {
            'api': {
                'public': 'https://testnet.binance.vision/api/v3',
                'private': 'https://testnet.binance.vision/api/v3',
            }
        }
    })
    logging.warning("⚠️ TESTNET MODE ACTIVE")
else:
    exchange = ccxt.binance({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    logging.warning("🔴 LIVE TRADING MODE - REAL MONEY!")

# ============ HELPER FUNCTIONS ============

def fetch_latest_data(limit=100):
    """Fetch latest price data"""
    try:
        # Use yfinance for data (more reliable)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=limit)
        
        df = yf.download('BTC-USD', start=start_date, end=end_date, 
                        progress=False, auto_adjust=True)
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df.columns = [col.lower() for col in df.columns]
        df.index = pd.to_datetime(df.index)
        
        return df[['open', 'high', 'low', 'close', 'volume']]
        
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return None

def fetch_fng():
    """Fetch current Fear & Greed Index"""
    try:
        url = "https://api.alternative.me/fng/?limit=1"
        response = requests.get(url, timeout=10)
        return int(response.json()['data'][0]['value'])
    except:
        logging.warning("FNG fetch failed, using default 50")
        return 50

def calculate_features(df):
    """Calculate technical indicators"""
    df = df.copy()
    
    # SMA
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['dist_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
    df['dist_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Volatility
    df['returns'] = df['close'].pct_change()
    df['volatility_7'] = df['returns'].rolling(7).std()
    df['volatility_30'] = df['returns'].rolling(30).std()
    
    # Volume
    df['volume_sma20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma20']
    
    # Momentum
    df['momentum_3'] = df['close'].pct_change(3)
    df['momentum_7'] = df['close'].pct_change(7)
    df['momentum_14'] = df['close'].pct_change(14)
    
    # MACD
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    return df

def smart_switch_strategy(ai_prob, fng_value, current_position):
    """Smart Switch trading logic"""
    if ai_prob >= 0.60:
        return 'BUY'
    elif fng_value <= 20:
        return 'BUY'
    elif current_position:
        if ai_prob <= 0.40 and fng_value > 30:
            return 'SELL'
        else:
            return 'HOLD'
    return 'WAIT'

def get_balance():
    """Get current balance"""
    try:
        balance = exchange.fetch_balance()
        usdt = balance['USDT']['free']
        btc = balance['BTC']['free']
        
        # Get current BTC price
        ticker = exchange.fetch_ticker(SYMBOL)
        btc_price = ticker['last']
        
        total_usd = usdt + (btc * btc_price)
        
        return {
            'usdt': usdt,
            'btc': btc,
            'btc_price': btc_price,
            'total_usd': total_usd
        }
    except Exception as e:
        logging.error(f"Error getting balance: {e}")
        return None

def execute_trade(signal, amount_usd):
    """Execute trade on exchange"""
    try:
        ticker = exchange.fetch_ticker(SYMBOL)
        current_price = ticker['last']
        
        if signal == 'BUY':
            # Calculate BTC amount to buy
            btc_amount = amount_usd / current_price
            
            # Round to exchange precision
            markets = exchange.load_markets()
            precision = markets[SYMBOL]['precision']['amount']
            btc_amount = round(btc_amount, precision)
            
            # Place market buy order
            order = exchange.create_market_buy_order(SYMBOL, btc_amount)
            logging.info(f"✅ BUY executed: {btc_amount} BTC at ${current_price:,.2f}")
            return True
            
        elif signal == 'SELL':
            # Get BTC balance
            balance = exchange.fetch_balance()
            btc_amount = balance['BTC']['free']
            
            if btc_amount > 0:
                # Place market sell order
                order = exchange.create_market_sell_order(SYMBOL, btc_amount)
                logging.info(f"✅ SELL executed: {btc_amount} BTC at ${current_price:,.2f}")
                return True
            else:
                logging.warning("No BTC to sell")
                return False
        
        return False
        
    except Exception as e:
        logging.error(f"❌ Trade execution failed: {e}")
        return False

def check_stop_loss(initial_capital):
    """Check if stop-loss triggered"""
    balance = get_balance()
    if balance:
        current_value = balance['total_usd']
        loss_percent = ((current_value - initial_capital) / initial_capital) * 100
        
        if loss_percent <= -MAX_DRAWDOWN_PERCENT:
            logging.critical(f"🛑 STOP-LOSS TRIGGERED! Loss: {loss_percent:.2f}%")
            return True
    return False

# ============ MAIN TRADING LOOP ============

def main():
    """Main bot loop"""
    logging.info("="*60)
    logging.info("🤖 CRYPTO TRADING BOT STARTED")
    logging.info("="*60)
    
    # Track initial capital
    initial_balance = get_balance()
    if not initial_balance:
        logging.error("Failed to get initial balance. Exiting.")
        return
    
    initial_capital = initial_balance['total_usd']
    logging.info(f"💰 Initial capital: ${initial_capital:,.2f}")
    
    position = initial_balance['btc'] > 0.0001  # True if holding BTC
    
    while True:
        try:
            logging.info(f"\n{'='*60}")
            logging.info(f"🔍 [{datetime.now()}] Analyzing market...")
            logging.info(f"{'='*60}")
            
            # 1. Check stop-loss
            if check_stop_loss(initial_capital):
                logging.critical("Bot stopped due to stop-loss")
                break
            
            # 2. Fetch data
            df = fetch_latest_data(limit=100)
            if df is None or len(df) < 50:
                logging.error("Insufficient data, skipping this cycle")
                time.sleep(CHECK_INTERVAL)
                continue
            
            fng = fetch_fng()
            
            # 3. Calculate features
            df = calculate_features(df)
            df['fng'] = fng
            df = df.dropna()
            
            if len(df) == 0:
                logging.error("No valid data after feature calculation")
                time.sleep(CHECK_INTERVAL)
                continue
            
            # 4. Get latest features
            latest_features = df[FEATURE_COLS].iloc[-1:].values
            current_price = df['close'].iloc[-1]
            
            # 5. Make prediction
            prediction_proba = model.predict_proba(latest_features)[0, 1]
            
            # 6. Get balance
            balance = get_balance()
            if not balance:
                logging.error("Failed to get balance")
                time.sleep(CHECK_INTERVAL)
                continue
            
            # Update position status
            position = balance['btc'] > 0.0001
            
            # 7. Log current status
            logging.info(f"📊 Current Price: ${current_price:,.2f}")
            logging.info(f"🤖 AI Probability: {prediction_proba:.3f}")
            logging.info(f"😱 Fear & Greed: {fng}")
            logging.info(f"💼 Position: {'BTC' if position else 'USDT'}")
            logging.info(f"💰 Portfolio Value: ${balance['total_usd']:,.2f}")
            
            # 8. Get signal
            signal = smart_switch_strategy(prediction_proba, fng, position)
            logging.info(f"📡 Signal: {signal}")
            
            # 9. Execute trade
            if signal == 'BUY' and not position:
                logging.info("🟢 Attempting to BUY...")
                success = execute_trade('BUY', TRADE_AMOUNT_USD)
                if success:
                    position = True
                    
            elif signal == 'SELL' and position:
                logging.info("🔴 Attempting to SELL...")
                success = execute_trade('SELL', 0)
                if success:
                    position = False
            
            else:
                logging.info(f"⏸️ No action taken - {signal}")
            
            # 10. Wait for next check
            logging.info(f"\n⏳ Sleeping for {CHECK_INTERVAL/60:.0f} minutes...")
            time.sleep(CHECK_INTERVAL)
            
        except KeyboardInterrupt:
            logging.info("\n🛑 Bot stopped by user")
            break
            
        except Exception as e:
            logging.error(f"❌ Error in main loop: {e}", exc_info=True)
            logging.info("Waiting 60 seconds before retry...")
            time.sleep(60)

if __name__ == "__main__":
    main()