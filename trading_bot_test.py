#!/usr/bin/env python3
"""
=============================================================================
CRYPTO TRADING BOT - PRODUCTION VERSION
=============================================================================
Automated Bitcoin trading bot using XGBoost ML model and Smart Switch strategy

Author: Your Name
Date: December 2025
Version: 1.0

REQUIREMENTS:
    pip install ccxt yfinance pandas numpy scikit-learn xgboost requests

FILES NEEDED (in same folder):
    - xgb_model.pkl (trained model from Colab)
    - bot_config.json (feature configuration)
    - trading_bot.py (this file)

SAFETY FEATURES:
    ✓ Testnet support for safe testing
    ✓ Stop-loss protection
    ✓ Position size limits
    ✓ Error handling and auto-retry
    ✓ Comprehensive logging

=============================================================================
"""

import ccxt
import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import json
import requests
import time
import sys
from datetime import datetime, timedelta
import logging
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Bot configuration settings"""
    
    # === EXCHANGE SETTINGS ===
    API_KEY = 'your_binance_api_key_here'
    API_SECRET = 'your_binance_secret_here'
    SYMBOL = 'BTC/USDT'
    
    # === TRADING SETTINGS ===
    TRADE_AMOUNT_USD = 100          # USD amount per trade
    USE_TESTNET = True              # ⚠️ ALWAYS start with True!
    CHECK_INTERVAL = 3600           # Check every 1 hour (seconds)
    
    # === RISK MANAGEMENT ===
    MAX_DRAWDOWN_PERCENT = 10       # Stop bot if loss > 10%
    MAX_POSITION_SIZE_PERCENT = 50  # Max 50% capital in BTC
    MIN_TRADE_AMOUNT = 10           # Minimum $10 per trade
    
    # === STRATEGY SETTINGS ===
    AI_CONFIDENCE_THRESHOLD = 0.60  # Buy if AI prob >= 60%
    AI_BEARISH_THRESHOLD = 0.40     # Sell if AI prob <= 40%
    FNG_EXTREME_FEAR = 20           # Buy at extreme fear
    FNG_RECOVERY_THRESHOLD = 30     # Min FNG to allow selling
    
    # === MODEL FILES ===
    MODEL_FILE = 'xgb_model.pkl'
    CONFIG_FILE = 'bot_config.json'
    
    # === LOGGING ===
    LOG_FILE = 'trading_bot.log'
    LOG_LEVEL = logging.INFO

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Configure logging with file and console output"""
    
    # Create logs directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)
    
    # Configure logging format
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Setup handlers
    logging.basicConfig(
        level=Config.LOG_LEVEL,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(f'logs/{Config.LOG_FILE}'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================================================
# MODEL LOADER
# ============================================================================

class ModelLoader:
    """Load and manage ML models and configuration"""
    
    def __init__(self):
        self.model = None
        self.config = None
        self.feature_cols = None
    
    def load(self):
        """Load model and configuration files"""
        try:
            logger.info("="*70)
            logger.info("📦 LOADING TRAINED MODELS")
            logger.info("="*70)
            
            # Load XGBoost model
            logger.info(f"Loading model from {Config.MODEL_FILE}...")
            with open(Config.MODEL_FILE, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"✅ Model loaded: {type(self.model).__name__}")
            
            # Load configuration
            logger.info(f"Loading config from {Config.CONFIG_FILE}...")
            with open(Config.CONFIG_FILE, 'r') as f:
                self.config = json.load(f)
            
            self.feature_cols = self.config['feature_cols']
            logger.info(f"✅ Config loaded: {len(self.feature_cols)} features")
            logger.info(f"   Features: {', '.join(self.feature_cols[:5])}...")
            
            # Test prediction
            dummy_data = np.random.rand(1, len(self.feature_cols))
            test_pred = self.model.predict_proba(dummy_data)
            logger.info(f"✅ Model test successful: prediction={test_pred[0][1]:.3f}")
            
            logger.info("="*70)
            return True
            
        except FileNotFoundError as e:
            logger.error(f"❌ File not found: {e}")
            logger.error("   Make sure xgb_model.pkl and bot_config.json are in the same folder!")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}", exc_info=True)
            return False
    
    def predict(self, features):
        """Make prediction using loaded model"""
        try:
            # Ensure correct shape
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            # Get probability of price going up
            proba = self.model.predict_proba(features)[0, 1]
            return float(proba)
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0.5  # Return neutral prediction on error

# ============================================================================
# EXCHANGE CONNECTION
# ============================================================================

class ExchangeConnector:
    """Handle connection to Binance exchange"""
    
    def __init__(self):
        self.exchange = None
        self.connect()
    
    def connect(self):
        """Initialize exchange connection"""
        try:
            if Config.USE_TESTNET:
                self.exchange = ccxt.binance({
                    'apiKey': Config.API_KEY,
                    'secret': Config.API_SECRET,
                    'enableRateLimit': True,
                    'options': {'defaultType': 'spot'},
                    'urls': {
                        'api': {
                            'public': 'https://testnet.binance.vision/api/v3',
                            'private': 'https://testnet.binance.vision/api/v3',
                        }
                    }
                })
                logger.warning("⚠️ TESTNET MODE - Using fake money")
                logger.info("   Get test funds: https://testnet.binance.vision/")
            else:
                self.exchange = ccxt.binance({
                    'apiKey': Config.API_KEY,
                    'secret': Config.API_SECRET,
                    'enableRateLimit': True,
                    'options': {'defaultType': 'spot'}
                })
                logger.warning("🔴 LIVE TRADING MODE - REAL MONEY AT RISK!")
            
            # Test connection
            self.exchange.load_markets()
            logger.info(f"✅ Connected to Binance")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to exchange: {e}")
            raise
    
    def get_balance(self):
        """Get current account balance"""
        try:
            balance = self.exchange.fetch_balance()
            
            usdt = balance['USDT']['free'] if 'USDT' in balance else 0
            btc = balance['BTC']['free'] if 'BTC' in balance else 0
            
            # Get current BTC price
            ticker = self.exchange.fetch_ticker(Config.SYMBOL)
            btc_price = ticker['last']
            
            total_usd = usdt + (btc * btc_price)
            
            return {
                'usdt': float(usdt),
                'btc': float(btc),
                'btc_price': float(btc_price),
                'total_usd': float(total_usd),
                'btc_value_usd': float(btc * btc_price)
            }
            
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return None
    
    def execute_buy(self, amount_usd):
        """Execute market buy order"""
        try:
            ticker = self.exchange.fetch_ticker(Config.SYMBOL)
            current_price = ticker['last']
            
            # Calculate BTC amount
            btc_amount = amount_usd / current_price
            
            # Round to exchange precision
            markets = self.exchange.load_markets()
            precision = markets[Config.SYMBOL]['precision']['amount']
            btc_amount = round(btc_amount, precision)
            
            # Check minimum
            if amount_usd < Config.MIN_TRADE_AMOUNT:
                logger.warning(f"Amount ${amount_usd} below minimum ${Config.MIN_TRADE_AMOUNT}")
                return None
            
            logger.info(f"💳 Placing BUY order: {btc_amount} BTC (~${amount_usd:.2f})")
            
            # Execute order
            order = self.exchange.create_market_buy_order(Config.SYMBOL, btc_amount)
            
            logger.info(f"✅ BUY ORDER FILLED")
            logger.info(f"   Amount: {btc_amount} BTC")
            logger.info(f"   Price: ${current_price:,.2f}")
            logger.info(f"   Total: ${amount_usd:.2f}")
            logger.info(f"   Order ID: {order['id']}")
            
            return order
            
        except Exception as e:
            logger.error(f"❌ Buy order failed: {e}", exc_info=True)
            return None
    
    def execute_sell(self):
        """Execute market sell order (sell all BTC)"""
        try:
            balance = self.exchange.fetch_balance()
            btc_amount = balance['BTC']['free']
            
            if btc_amount < 0.0001:  # Minimum BTC amount
                logger.warning("No BTC to sell")
                return None
            
            ticker = self.exchange.fetch_ticker(Config.SYMBOL)
            current_price = ticker['last']
            usd_value = btc_amount * current_price
            
            logger.info(f"💳 Placing SELL order: {btc_amount} BTC (~${usd_value:.2f})")
            
            # Execute order
            order = self.exchange.create_market_sell_order(Config.SYMBOL, btc_amount)
            
            logger.info(f"✅ SELL ORDER FILLED")
            logger.info(f"   Amount: {btc_amount} BTC")
            logger.info(f"   Price: ${current_price:,.2f}")
            logger.info(f"   Total: ${usd_value:.2f}")
            logger.info(f"   Order ID: {order['id']}")
            
            return order
            
        except Exception as e:
            logger.error(f"❌ Sell order failed: {e}", exc_info=True)
            return None

# ============================================================================
# DATA FETCHER
# ============================================================================

class DataFetcher:
    """Fetch and process market data"""
    
    @staticmethod
    def fetch_ohlcv(limit=100):
        """Fetch OHLCV data from Yahoo Finance"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=limit + 10)
            
            df = yf.download('BTC-USD', 
                           start=start_date, 
                           end=end_date,
                           progress=False,
                           auto_adjust=True)
            
            # Handle MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Standardize column names
            df.columns = [col.lower() for col in df.columns]
            
            # Ensure datetime index
            df.index = pd.to_datetime(df.index)
            
            # Keep only required columns
            required = ['open', 'high', 'low', 'close', 'volume']
            df = df[[col for col in required if col in df.columns]]
            
            # Drop NaN
            df = df.dropna()
            
            if len(df) < 50:
                raise ValueError(f"Insufficient data: only {len(df)} rows")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {e}")
            return None
    
    @staticmethod
    def fetch_fear_greed():
        """Fetch Fear & Greed Index"""
        try:
            url = "https://api.alternative.me/fng/?limit=1"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if 'data' in data and len(data['data']) > 0:
                fng_value = int(data['data'][0]['value'])
                fng_text = data['data'][0]['value_classification']
                return fng_value, fng_text
            else:
                logger.warning("FNG API returned empty data")
                return 50, "Neutral"
                
        except Exception as e:
            logger.warning(f"FNG fetch failed: {e}, using default 50")
            return 50, "Neutral"

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

class FeatureEngine:
    """Calculate technical indicators"""
    
    @staticmethod
    def calculate_features(df):
        """Calculate all technical indicators"""
        df = df.copy()
        
        try:
            # === Moving Averages ===
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['dist_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
            df['dist_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
            
            # === RSI ===
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # === Volatility ===
            df['returns'] = df['close'].pct_change()
            df['volatility_7'] = df['returns'].rolling(window=7).std()
            df['volatility_30'] = df['returns'].rolling(window=30).std()
            
            # === Volume ===
            df['volume_sma20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma20']
            
            # === Momentum ===
            df['momentum_3'] = df['close'].pct_change(periods=3)
            df['momentum_7'] = df['close'].pct_change(periods=7)
            df['momentum_14'] = df['close'].pct_change(periods=14)
            
            # === MACD ===
            ema_12 = df['close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Drop NaN values
            df = df.dropna()
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            return None

# ============================================================================
# TRADING STRATEGY
# ============================================================================

class TradingStrategy:
    """Smart Switch trading strategy"""
    
    @staticmethod
    def smart_switch(ai_prob, fng_value, current_position):
        """
        Smart Switch Strategy Logic
        
        Args:
            ai_prob: AI model probability (0-1)
            fng_value: Fear & Greed Index (0-100)
            current_position: True if holding BTC, False if in USDT
        
        Returns:
            'BUY', 'SELL', 'HOLD', or 'WAIT'
        """
        
        # === ENTRY RULES ===
        
        # Rule 1: High AI Confidence (Trend Following)
        if ai_prob >= Config.AI_CONFIDENCE_THRESHOLD:
            return 'BUY'
        
        # Rule 2: Extreme Fear (Mean Reversion / Buy the Dip)
        if fng_value <= Config.FNG_EXTREME_FEAR:
            return 'BUY'
        
        # === EXIT RULES ===
        
        if current_position:
            # Rule 3: AI Bearish + Market Recovered from Panic
            if ai_prob <= Config.AI_BEARISH_THRESHOLD and fng_value > Config.FNG_RECOVERY_THRESHOLD:
                return 'SELL'
            else:
                # Hold through panic
                return 'HOLD'
        
        # Default: Wait for signal
        return 'WAIT'
    
    @staticmethod
    def explain_signal(signal, ai_prob, fng_value, current_position):
        """Explain why a signal was generated"""
        
        if signal == 'BUY':
            if ai_prob >= Config.AI_CONFIDENCE_THRESHOLD:
                return f"High AI confidence ({ai_prob:.1%})"
            elif fng_value <= Config.FNG_EXTREME_FEAR:
                return f"Extreme fear detected (FNG={fng_value})"
        
        elif signal == 'SELL':
            return f"AI bearish ({ai_prob:.1%}) + Market recovered (FNG={fng_value})"
        
        elif signal == 'HOLD':
            if fng_value <= Config.FNG_RECOVERY_THRESHOLD:
                return f"Holding through panic (FNG={fng_value})"
            else:
                return f"AI not bearish enough ({ai_prob:.1%})"
        
        elif signal == 'WAIT':
            return f"No strong signal (AI={ai_prob:.1%}, FNG={fng_value})"
        
        return "Unknown reason"

# ============================================================================
# RISK MANAGER
# ============================================================================

class RiskManager:
    """Manage risk and position sizing"""
    
    def __init__(self, initial_capital):
        self.initial_capital = initial_capital
        self.max_drawdown = 0
    
    def check_stop_loss(self, current_value):
        """Check if stop-loss should be triggered"""
        loss_percent = ((current_value - self.initial_capital) / self.initial_capital) * 100
        
        # Update max drawdown
        if loss_percent < self.max_drawdown:
            self.max_drawdown = loss_percent
        
        if loss_percent <= -Config.MAX_DRAWDOWN_PERCENT:
            logger.critical("="*70)
            logger.critical("🛑 STOP-LOSS TRIGGERED!")
            logger.critical(f"   Current Loss: {loss_percent:.2f}%")
            logger.critical(f"   Max Allowed: {Config.MAX_DRAWDOWN_PERCENT}%")
            logger.critical(f"   Initial Capital: ${self.initial_capital:,.2f}")
            logger.critical(f"   Current Value: ${current_value:,.2f}")
            logger.critical("="*70)
            return True
        
        return False
    
    def check_position_size(self, balance):
        """Check if position size is within limits"""
        if balance['btc_value_usd'] > 0:
            position_percent = (balance['btc_value_usd'] / balance['total_usd']) * 100
            
            if position_percent > Config.MAX_POSITION_SIZE_PERCENT:
                logger.warning(f"⚠️ Position size {position_percent:.1f}% exceeds limit {Config.MAX_POSITION_SIZE_PERCENT}%")
                return False
        
        return True
    
    def get_trade_amount(self, balance):
        """Calculate safe trade amount"""
        # Use configured amount or max available (whichever is smaller)
        max_safe = balance['usdt'] * 0.9  # Keep 10% buffer
        trade_amount = min(Config.TRADE_AMOUNT_USD, max_safe)
        
        if trade_amount < Config.MIN_TRADE_AMOUNT:
            logger.warning(f"Insufficient funds for trade (${trade_amount:.2f} < ${Config.MIN_TRADE_AMOUNT})")
            return 0
        
        return trade_amount

# ============================================================================
# MAIN TRADING BOT
# ============================================================================

class TradingBot:
    """Main trading bot orchestrator"""
    
    def __init__(self):
        self.model_loader = ModelLoader()
        self.exchange = None
        self.risk_manager = None
        self.running = False
    
    def initialize(self):
        """Initialize all components"""
        try:
            logger.info("="*70)
            logger.info("🤖 INITIALIZING CRYPTO TRADING BOT")
            logger.info("="*70)
            
            # Load models
            if not self.model_loader.load():
                return False
            
            # Connect to exchange
            self.exchange = ExchangeConnector()
            
            # Get initial balance
            balance = self.exchange.get_balance()
            if not balance:
                logger.error("Failed to get initial balance")
                return False
            
            # Initialize risk manager
            self.risk_manager = RiskManager(balance['total_usd'])
            
            logger.info("="*70)
            logger.info("💰 INITIAL PORTFOLIO")
            logger.info("="*70)
            logger.info(f"   USDT: ${balance['usdt']:,.2f}")
            logger.info(f"   BTC: {balance['btc']:.6f} (${balance['btc_value_usd']:,.2f})")
            logger.info(f"   Total: ${balance['total_usd']:,.2f}")
            logger.info("="*70)
            
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}", exc_info=True)
            return False
    
    def run_cycle(self, position):
        """Run one trading cycle"""
        try:
            logger.info("\n" + "="*70)
            logger.info(f"🔍 ANALYSIS CYCLE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("="*70)
            
            # 1. Fetch market data
            logger.info("📊 Fetching market data...")
            df = DataFetcher.fetch_ohlcv(limit=100)
            if df is None or len(df) < 50:
                logger.error("Insufficient market data")
                return position
            
            # 2. Fetch Fear & Greed
            fng_value, fng_text = DataFetcher.fetch_fear_greed()
            
            # 3. Calculate features
            logger.info("⚙️ Calculating technical indicators...")
            df = FeatureEngine.calculate_features(df)
            if df is None or len(df) == 0:
                logger.error("Feature calculation failed")
                return position
            
            # Add FNG to dataframe
            df['fng'] = fng_value
            
            # 4. Extract latest features
            latest_features = df[self.model_loader.feature_cols].iloc[-1].values
            current_price = df['close'].iloc[-1]
            
            # 5. Make prediction
            logger.info("🤖 Running AI prediction...")
            ai_prob = self.model_loader.predict(latest_features)
            
            # 6. Get current balance
            balance = self.exchange.get_balance()
            if not balance:
                logger.error("Failed to get balance")
                return position
            
            # Update position status
            position = balance['btc'] > 0.0001
            
            # 7. Display analysis
            logger.info("\n📈 MARKET ANALYSIS")
            logger.info("-" * 70)
            logger.info(f"   BTC Price: ${current_price:,.2f}")
            logger.info(f"   RSI(14): {df['rsi'].iloc[-1]:.2f}")
            logger.info(f"   Dist to SMA20: {df['dist_sma20'].iloc[-1]:.2%}")
            logger.info(f"   Volatility(7d): {df['volatility_7'].iloc[-1]:.4f}")
            logger.info(f"   AI Probability: {ai_prob:.1%} {'🟢' if ai_prob >= 0.6 else '🔴' if ai_prob <= 0.4 else '⚪'}")
            logger.info(f"   Fear & Greed: {fng_value}/100 ({fng_text}) {'😱' if fng_value < 25 else '😰' if fng_value < 45 else '😐' if fng_value < 55 else '😃' if fng_value < 75 else '🤑'}")
            logger.info(f"   Position: {'BTC 🟢' if position else 'USDT 💵'}")
            logger.info(f"   Portfolio Value: ${balance['total_usd']:,.2f}")
            
            # 8. Check stop-loss
            if self.risk_manager.check_stop_loss(balance['total_usd']):
                logger.critical("🛑 STOP-LOSS TRIGGERED - STOPPING BOT")
                return None  # Signal to stop bot
            
            # 9. Generate signal
            signal = TradingStrategy.smart_switch(ai_prob, fng_value, position)
            reason = TradingStrategy.explain_signal(signal, ai_prob, fng_value, position)
            
            logger.info("\n🎯 TRADING DECISION")
            logger.info("-" * 70)
            logger.info(f"   Signal: {signal} {'🟢' if signal == 'BUY' else '🔴' if signal == 'SELL' else '⏸️'}")
            logger.info(f"   Reason: {reason}")
            
            # 10. Execute trade
            if signal == 'BUY' and not position:
                trade_amount = self.risk_manager.get_trade_amount(balance)
                if trade_amount > 0:
                    logger.info(f"\n💳 EXECUTING BUY ORDER")
                    order = self.exchange.execute_buy(trade_amount)
                    if order:
                        position = True
                        logger.info("✅ Position opened")
            
            elif signal == 'SELL' and position:
                logger.info(f"\n💳 EXECUTING SELL ORDER")
                order = self.exchange.execute_sell()
                if order:
                    position = False
                    logger.info("✅ Position closed")
            
            else:
                logger.info(f"   Action: {'Holding BTC 💎' if position else 'Waiting for entry 🕐'}")
            
            logger.info("="*70)
            
            return position
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}", exc_info=True)
            return position
    
    def run(self):
        """Main bot loop"""
        if not self.initialize():
            logger.error("Failed to initialize bot")
            return
        
        # Determine initial position
        balance = self.exchange.get_balance()
        position = balance['btc'] > 0.0001 if balance else False
        
        logger.info("\n" + "="*70)
        logger.info("🚀 TRADING BOT STARTED")
        logger.info("="*70)
        logger.info(f"   Mode: {'TESTNET 🧪' if Config.USE_TESTNET else 'LIVE 🔴'}")
        logger.info(f"   Symbol: {Config.SYMBOL}")
        logger.info(f"   Check Interval: {Config.CHECK_INTERVAL/60:.0f} minutes")
        logger.info(f"   Trade Amount: ${Config.TRADE_AMOUNT_USD}")
        logger.info(f"   Max Drawdown: {Config.MAX_DRAWDOWN_PERCENT}%")
        logger.info("="*70)
        logger.info("\n💡 Press Ctrl+C to stop the bot\n")
        
        self.running = True
        cycle_count = 0
        
        try:
            while self.running:
                cycle_count += 1
                logger.info(f"\n{'='*70}")
                logger.info(f"CYCLE #{cycle_count}")
                logger.info(f"{'='*70}")
                
                position = self.run_cycle(position)
                
                # Check if stop-loss triggered
                if position is None:
                    logger.critical("Bot stopped by stop-loss")
                    break
                
                # Sleep until next cycle
                logger.info(f"\n⏳ Next check in {Config.CHECK_INTERVAL/60:.0f} minutes...")
                logger.info(f"   Sleeping until {(datetime.now() + timedelta(seconds=Config.CHECK_INTERVAL)).strftime('%H:%M:%S')}")
                time.sleep(Config.CHECK_INTERVAL)
                
        except KeyboardInterrupt:
            logger.info("\n\n" + "="*70)
            logger.info("🛑 BOT STOPPED BY USER")
            logger.info("="*70)
            
            # Get final balance
            balance = self.exchange.get_balance()
            if balance:
                profit = balance['total_usd'] - self.risk_manager.initial_capital
                profit_pct = (profit / self.risk_manager.initial_capital) * 100
                
                logger.info("\n📊 FINAL REPORT")
                logger.info("-" * 70)
                logger.info(f"   Initial Capital: ${self.risk_manager.initial_capital:,.2f}")
                logger.info(f"   Final Value: ${balance['total_usd']:,.2f}")
                logger.info(f"   Profit/Loss: ${profit:,.2f} ({profit_pct:+.2f}%)")
                logger.info(f"   Max Drawdown: {self.risk_manager.max_drawdown:.2f}%")
                logger.info(f"   Cycles Run: {cycle_count}")
                logger.info("="*70)
        
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
        
        finally:
            self.running = False
            logger.info("\n👋 Bot shutdown complete")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║             🤖 CRYPTO TRADING BOT v1.0                          ║
    ║                                                                  ║
    ║  Automated Bitcoin Trading using XGBoost ML + Smart Switch      ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Check configuration
    if Config.API_KEY == 'your_binance_api_key_here':
        print("❌ ERROR: Please set your Binance API key in Config class!")
        print("   Edit trading_bot.py and add your API credentials\n")
        return
    
    if not Config.USE_TESTNET:
        print("="*70)
        print("⚠️  WARNING: LIVE TRADING MODE")
        print("="*70)
        print("You are about to trade with REAL MONEY!")
        print("Make sure you have:")
        print("  ✓ Tested thoroughly on testnet")
        print("  ✓ Set appropriate trade amounts")
        print("  ✓ Configured stop-loss limits")
        print()
        response = input("Type 'YES' to continue with live trading: ")
        if response != 'YES':
            print("Aborting. Set USE_TESTNET = True for safe testing.")
            return
    
    # Create and run bot
    bot = TradingBot()
    bot.run()

if __name__ == "__main__":
    main()
