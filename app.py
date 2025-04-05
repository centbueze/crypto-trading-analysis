import os
import hmac
import hashlib
import base64
from io import BytesIO
from datetime import datetime
from typing import Optional, Dict, Any
import textwrap

import requests
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, session, flash, redirect, url_for
from matplotlib.figure import Figure
from config import Config

app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = os.environ.get('SECRET_KEY') or 'dev-secret-key'

# Configure numpy NaN handling
np.NaN = np.nan

class BinanceTradingAnalyzer:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.BASE_URL = 'https://api.binance.com'
        self.KLINES_ENDPOINT = '/api/v3/klines'
        self.TICKER_PRICE_ENDPOINT = '/api/v3/ticker/price'
        
    def _create_signed_request(self, params: Dict[str, Any]) -> str:
        """Create a signed request for Binance API."""
        query_string = '&'.join([f"{key}={value}" for key, value in params.items()])
        signature = hmac.new(
            self.api_secret.encode('utf-8'), 
            query_string.encode('utf-8'), 
            hashlib.sha256
        ).hexdigest()
        return f"{query_string}&signature={signature}"
    
    def get_historical_data(self, symbol: str, interval: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch historical kline data from Binance."""
        url = f"{self.BASE_URL}{self.KLINES_ENDPOINT}"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            columns = [
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 
                'close_time', 'quote_asset_volume', 'trades', 
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ]
            
            df = pd.DataFrame(data, columns=columns)
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['close'] = df['close'].ffill()
            
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching historical data: {str(e)}")
            return None
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get the current price for a trading pair."""
        url = f"{self.BASE_URL}{self.TICKER_PRICE_ENDPOINT}"
        params = {'symbol': symbol}
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return float(response.json()['price'])
        except requests.exceptions.RequestException as e:
            print(f"Error fetching current price: {str(e)}")
            return None
    
    def perform_technical_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform comprehensive technical analysis on the dataframe."""
        # Ensure numeric data types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate indicators
        df['rsi'] = ta.rsi(df['close'], length=14)
        
        # SMA with different periods for better trend analysis
        df['sma_20'] = ta.sma(df['close'], length=20)
        df['sma_50'] = ta.sma(df['close'], length=50)
        df['sma_200'] = ta.sma(df['close'], length=200)
        
        # EMA for more responsive trend following
        df['ema_12'] = ta.ema(df['close'], length=12)
        df['ema_26'] = ta.ema(df['close'], length=26)
        
        # MACD analysis
        macd = ta.macd(df['close'])
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        df['macd_hist'] = macd['MACDh_12_26_9']
        
        # Stochastic oscillator
        stoch = ta.stoch(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch['STOCHk_14_3_3']
        df['stoch_d'] = stoch['STOCHd_14_3_3']
        
        # Volatility measures
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        # Bollinger Bands
        bb = ta.bbands(df['close'])
        df['bb_upper'] = bb['BBU_5_2.0']
        df['bb_middle'] = bb['BBM_5_2.0']
        df['bb_lower'] = bb['BBL_5_2.0']
        
        # Support and resistance levels
        df['support'] = df['low'].rolling(window=14, min_periods=1).min()
        df['resistance'] = df['high'].rolling(window=14, min_periods=1).max()
        
        # Fibonacci retracement levels
        recent_high = df['high'].rolling(window=14, min_periods=1).max()
        recent_low = df['low'].rolling(window=14, min_periods=1).min()
        df['fib_23.6'] = recent_high - (recent_high - recent_low) * 0.236
        df['fib_38.2'] = recent_high - (recent_high - recent_low) * 0.382
        df['fib_50.0'] = recent_high - (recent_high - recent_low) * 0.5
        df['fib_61.8'] = recent_high - (recent_high - recent_low) * 0.618
        
        return df
    
    def calculate_risk_metrics(self, df: pd.DataFrame) -> dict:
        """Calculate comprehensive risk metrics for the current market"""
        if len(df) < 1:
            return {
                'atr_pct': 0,
                'trend_strength': 0,
                'momentum_1h': 0,
                'momentum_4h': 0,
                'volume_ratio': 1.0
            }
            
        latest = df.iloc[-1]
        
        # Ensure we have enough data for calculations
        min_periods = min(14, len(df))
        
        # Volatility measurement (ATR percentage)
        atr_pct = (latest['atr'] / latest['close']) * 100 if latest['close'] != 0 else 0
        
        # Trend strength (distance from moving averages)
        trend_strength = ((latest['close'] - latest['sma_50']) / latest['sma_50']) * 100 if latest['sma_50'] != 0 else 0
        
        # Recent price momentum (percentage change)
        lookback_1h = min(4, len(df)-1) if len(df) > 1 else 0
        lookback_4h = min(16, len(df)-1) if len(df) > 1 else 0
        
        momentum_1h = 0
        momentum_4h = 0
        
        if lookback_1h > 0 and df['close'].iloc[-lookback_1h] != 0:
            momentum_1h = ((latest['close'] - df['close'].iloc[-lookback_1h]) / df['close'].iloc[-lookback_1h]) * 100
            
        if lookback_4h > 0 and df['close'].iloc[-lookback_4h] != 0:
            momentum_4h = ((latest['close'] - df['close'].iloc[-lookback_4h]) / df['close'].iloc[-lookback_4h]) * 100
        
        # Volume ratio (safe calculation)
        try:
            volume_avg = df['volume'].rolling(min_periods, min_periods=1).mean().iloc[-1]
            volume_ratio = latest['volume'] / volume_avg if volume_avg != 0 else 1.0
        except:
            volume_ratio = 1.0
        
        return {
            'atr_pct': atr_pct,
            'trend_strength': trend_strength,
            'momentum_1h': momentum_1h,
            'momentum_4h': momentum_4h,
            'volume_ratio': volume_ratio
        }

    def calculate_risk_percentage(self, metrics: dict) -> str:
        """Calculate overall risk percentage score (0-100%)"""
        try:
            # Weighted risk calculation
            volatility_score = min(metrics['atr_pct'] * 10, 50)  # 0-50 points
            trend_score = 30 if abs(metrics['trend_strength']) > 2 else 10  # 10-30 points
            momentum_score = min((abs(metrics['momentum_1h']) + abs(metrics['momentum_4h'])), 20)  # 0-20 points
            
            total_score = volatility_score + trend_score + momentum_score
            risk_pct = min(total_score, 100)
            
            if risk_pct > 70:
                return f"High Risk ({risk_pct:.0f}%) - Consider smaller position"
            elif risk_pct > 40:
                return f"Medium Risk ({risk_pct:.0f}%) - Normal position size"
            else:
                return f"Low Risk ({risk_pct:.0f}%) - Can increase position"
        except:
            return "Risk assessment unavailable"

    def generate_trading_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signals based on technical indicators."""
        if len(df) < 2:  # Need at least 2 data points for comparisons
            return {
                'risk_assessment': "Insufficient data for risk assessment",
                'action': 'HOLD',
                'reason': "Not enough data points to generate signals"
            }
            
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Calculate risk metrics first
        risk_metrics = self.calculate_risk_metrics(df)
        risk_assessment = self.calculate_risk_percentage(risk_metrics)
        
        signals = {
            'sma_cross': None,
            'macd_cross': None,
            'rsi_signal': None,
            'stoch_signal': None,
            'fib_level': None,
            'bb_signal': None,
            'risk_assessment': risk_assessment,
            'volatility': f"{risk_metrics['atr_pct']:.2f}%",
            'trend_strength': f"{risk_metrics['trend_strength']:.2f}%",
            'momentum': f"1h: {risk_metrics['momentum_1h']:.2f}%, 4h: {risk_metrics['momentum_4h']:.2f}%"
        }
        
        # SMA Crossover signals
        if (prev['sma_20'] <= prev['sma_50']) and (latest['sma_20'] > latest['sma_50']):
            signals['sma_cross'] = 'bullish'
        elif (prev['sma_20'] >= prev['sma_50']) and (latest['sma_20'] < latest['sma_50']):
            signals['sma_cross'] = 'bearish'
        
        # MACD Crossover signals
        if (prev['macd'] <= prev['macd_signal']) and (latest['macd'] > latest['macd_signal']):
            signals['macd_cross'] = 'bullish'
        elif (prev['macd'] >= prev['macd_signal']) and (latest['macd'] < latest['macd_signal']):
            signals['macd_cross'] = 'bearish'
        
        # RSI signals
        if latest['rsi'] < 30:
            signals['rsi_signal'] = 'oversold'
        elif latest['rsi'] > 70:
            signals['rsi_signal'] = 'overbought'
        
        # Stochastic signals
        if latest['stoch_k'] < 20 and latest['stoch_d'] < 20:
            signals['stoch_signal'] = 'oversold'
        elif latest['stoch_k'] > 80 and latest['stoch_d'] > 80:
            signals['stoch_signal'] = 'overbought'
        
        # Fibonacci level analysis
        if latest['close'] > latest['fib_38.2'] and latest['close'] < latest['fib_61.8']:
            signals['fib_level'] = 'between_38.2_and_61.8'
        elif latest['close'] < latest['fib_23.6']:
            signals['fib_level'] = 'below_23.6'
        
        # Bollinger Bands signal
        if latest['close'] < latest['bb_lower']:
            signals['bb_signal'] = 'below_lower_band'
        elif latest['close'] > latest['bb_upper']:
            signals['bb_signal'] = 'above_upper_band'
        
        return signals
    
    def generate_recommendation(self, df: pd.DataFrame, signals: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading recommendation with risk percentage in all market conditions"""
        if len(df) < 2:
            return {
                'action': 'HOLD',
                'reason': "Insufficient data for recommendation",
                'risk_assessment': "Insufficient data",
                'risk_pct': "N/A",
                'reward_pct': "N/A",
                'risk_reward': "N/A"
            }
            
        latest = df.iloc[-1]
        
        # Initialize recommendation with all required fields
        recommendation = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': 'BTCUSDT',
            'risk_assessment': signals.get('risk_assessment', 'Not available'),
            'volatility': signals.get('volatility', 'N/A'),
            'trend': "Uptrend" if latest['close'] > latest['sma_50'] else "Downtrend",
            'momentum': signals.get('momentum', 'N/A'),
            'risk_pct': "0%",
            'reward_pct': "0%",
            'risk_reward': "1:0"
        }
        
        # Determine trade direction based on trend
        try:
            if latest['close'] > latest['sma_50']:
                recommendation.update({
                    'action': 'BUY',
                    'entry': latest['close'],
                    'stop_loss': max(latest['sma_20'], latest['support']),
                    'take_profit': latest['close'] * 1.02  # 2% target
                })
            else:
                recommendation.update({
                    'action': 'SELL',
                    'entry': latest['close'],
                    'stop_loss': min(latest['sma_20'], latest['resistance']),
                    'take_profit': latest['close'] * 0.98  # 2% target
                })

            # Calculate risk as percentage of entry
            if 'entry' in recommendation and 'stop_loss' in recommendation and recommendation['entry'] != 0:
                risk_pct = abs(recommendation['entry'] - recommendation['stop_loss']) / recommendation['entry'] * 100
                reward_pct = abs(recommendation['entry'] - recommendation['take_profit']) / recommendation['entry'] * 100
                risk_reward = f"1:{reward_pct/risk_pct:.1f}" if risk_pct != 0 else "1:0"
                
                recommendation.update({
                    'risk_pct': f"{risk_pct:.2f}%",
                    'reward_pct': f"{reward_pct:.2f}%",
                    'risk_reward': risk_reward
                })

        except Exception as e:
            print(f"Error generating recommendation: {str(e)}")
            recommendation.update({
                'action': 'HOLD',
                'reason': f"Error in calculation: {str(e)}",
                'risk_pct': "N/A",
                'reward_pct': "N/A",
                'risk_reward': "N/A"
            })

        # Generate reason after all calculations are done
        recommendation['reason'] = self._generate_reason(recommendation, latest)
        
        return recommendation

    def _generate_reason(self, recommendation: dict, latest_data: pd.Series) -> str:
        """Generate detailed reason for the recommendation"""
        action = recommendation.get('action', 'HOLD')
        risk_pct = recommendation.get('risk_pct', 'N/A')
        
        if action == 'BUY':
            return (f"Bullish setup with {risk_pct} risk. Price above SMA50 suggests uptrend. "
                   f"Stop at support/SMA20 ({recommendation.get('stop_loss', 'N/A'):.2f}). "
                   f"Targeting 2% gain at {recommendation.get('take_profit', 'N/A'):.2f}")
        elif action == 'SELL':
            return (f"Bearish setup with {risk_pct} risk. Price below SMA50 suggests downtrend. "
                   f"Stop at resistance/SMA20 ({recommendation.get('stop_loss', 'N/A'):.2f}). "
                   f"Targeting 2% decline at {recommendation.get('take_profit', 'N/A'):.2f}")
        else:
            return "No clear trading signal detected"

    def visualize_analysis(self, df: pd.DataFrame, symbol: str) -> Figure:
        """Generate professional visualization of the technical analysis."""
        if len(df) < 2:
            print("Not enough data to generate visualization")
            return None
            
        # Create figure without displaying
        fig = Figure(figsize=(16, 12))
        
        # Create subplots
        ax1 = fig.add_subplot(6, 1, (1, 4))
        ax2 = fig.add_subplot(6, 1, 5)
        ax3 = fig.add_subplot(6, 1, 6)
        
        # Price chart with SMAs and Bollinger Bands
        ax1.plot(df['timestamp'], df['close'], label='Price', color='black', linewidth=1.5)
        ax1.plot(df['timestamp'], df['sma_20'], label='SMA 20', color='blue', linewidth=1)
        ax1.plot(df['timestamp'], df['sma_50'], label='SMA 50', color='orange', linewidth=1)
        ax1.plot(df['timestamp'], df['bb_upper'], label='BB Upper', color='gray', linestyle='--', linewidth=0.75)
        ax1.plot(df['timestamp'], df['bb_middle'], label='BB Middle', color='gray', linewidth=0.75)
        ax1.plot(df['timestamp'], df['bb_lower'], label='BB Lower', color='gray', linestyle='--', linewidth=0.75)
        
        # Fill between Bollinger Bands
        ax1.fill_between(df['timestamp'], df['bb_lower'], df['bb_upper'], color='gray', alpha=0.1)
        
        # Highlight SMA crossovers
        if len(df) > 1:
            sma_20_above_50 = df['sma_20'] > df['sma_50']
            for i in range(1, len(sma_20_above_50)):
                if sma_20_above_50[i] and not sma_20_above_50[i-1]:
                    ax1.scatter(df['timestamp'].iloc[i], df['sma_20'].iloc[i], 
                               color='green', marker='^', s=100, label='Bullish Crossover' if i == 1 else "")
                elif not sma_20_above_50[i] and sma_20_above_50[i-1]:
                    ax1.scatter(df['timestamp'].iloc[i], df['sma_20'].iloc[i], 
                               color='red', marker='v', s=100, label='Bearish Crossover' if i == 1 else "")
        
        ax1.set_title(f'{symbol} Technical Analysis', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True)
        
        # Volume chart
        ax2.bar(df['timestamp'], df['volume'], color='gray', alpha=0.7)
        ax2.set_ylabel('Volume', fontsize=10)
        ax2.grid(True)
        
        # RSI chart
        ax3.plot(df['timestamp'], df['rsi'], label='RSI', color='purple')
        ax3.axhline(30, color='green', linestyle='--', linewidth=0.75)
        ax3.axhline(70, color='red', linestyle='--', linewidth=0.75)
        ax3.fill_between(df['timestamp'], 30, df['rsi'], where=(df['rsi']<=30), color='green', alpha=0.3)
        ax3.fill_between(df['timestamp'], 70, df['rsi'], where=(df['rsi']>=70), color='red', alpha=0.3)
        ax3.set_ylabel('RSI', fontsize=10)
        ax3.legend(loc='upper left')
        ax3.grid(True)
        ax3.set_ylim(0, 100)
        
        fig.tight_layout()
        return fig
    
    def generate_report(self, df: pd.DataFrame, signals: Dict[str, Any], 
                       recommendation: Dict[str, Any]) -> str:
        """Generate a comprehensive trading report with risk assessment."""
        if len(df) == 0:
            return "No data available to generate report"
            
        latest = df.iloc[-1]
        report = [
            f"\n{'='*50}",
            f"Technical Analysis Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"{'='*50}\n",
            f"Current Price: {latest['close']:.4f}",
            f"24h High: {df['high'].max():.4f}",
            f"24h Low: {df['low'].min():.4f}",
            f"\nRisk Assessment: {signals.get('risk_assessment', 'Not available')}",
            f"\nKey Indicators:",
            f"- RSI (14): {latest['rsi']:.2f} {'(Oversold)' if signals.get('rsi_signal') == 'oversold' else '(Overbought)' if signals.get('rsi_signal') == 'overbought' else ''}",
            f"- SMA 20: {latest['sma_20']:.4f}",
            f"- SMA 50: {latest['sma_50']:.4f}",
            f"- MACD: {latest['macd']:.4f}",
            f"- MACD Signal: {latest['macd_signal']:.4f}",
            f"- ATR (14): {latest['atr']:.4f} (Volatility: {(latest['atr'] / latest['close']) * 100:.2f}%)" if latest['close'] != 0 else "- ATR (14): N/A",
            f"\nSupport/Resistance:",
            f"- Current Support: {latest['support']:.4f}",
            f"- Current Resistance: {latest['resistance']:.4f}",
            f"\nActive Signals:"
        ]
        
        # Add active signals
        if signals.get('sma_cross'):
            report.append(f"- SMA Crossover: {signals['sma_cross'].upper()} (20/50 MA)")
        if signals.get('macd_cross'):
            report.append(f"- MACD Crossover: {signals['macd_cross'].upper()}")
        if signals.get('rsi_signal'):
            report.append(f"- RSI Signal: {signals['rsi_signal'].upper()}")
        if signals.get('stoch_signal'):
            report.append(f"- Stochastic Signal: {signals['stoch_signal'].upper()}")
        if signals.get('fib_level'):
            report.append(f"- Fibonacci Position: {signals['fib_level'].replace('_', ' ').upper()}")
        if signals.get('bb_signal'):
            report.append(f"- Bollinger Band: Price is {signals['bb_signal'].replace('_', ' ').upper()}")
        
        # Trade recommendation section
        report.extend([
            f"\n{'='*30}",
            "TRADE RECOMMENDATION",
            f"{'='*30}",
            f"Action: {recommendation.get('action', 'HOLD')}",
            f"Entry Price: {recommendation.get('entry', 'N/A'):.4f}" if 'entry' in recommendation else "Entry Price: N/A",
            f"Stop Loss: {recommendation.get('stop_loss', 'N/A'):.4f} ({recommendation.get('risk_pct', 'N/A')} risk)" if 'stop_loss' in recommendation else "Stop Loss: N/A",
            f"Take Profit: {recommendation.get('take_profit', 'N/A'):.4f} ({recommendation.get('reward_pct', 'N/A')} reward)" if 'take_profit' in recommendation else "Take Profit: N/A",
            f"Risk/Reward Ratio: {recommendation.get('risk_reward', 'N/A')}",
            f"\nRationale:",
            textwrap.fill(recommendation.get('reason', 'No rationale provided'), width=70),
            f"\nAdditional Context:",
            f"- Trend: {recommendation.get('trend', 'N/A')}",
            f"- Momentum: {recommendation.get('momentum', 'N/A')}",
            f"- Volatility: {recommendation.get('volatility', 'N/A')}",
            f"\n{'='*50}"
        ])
        
        return "\n".join(report)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        symbol = request.form.get('symbol', 'BTCUSDT').upper()
        interval = request.form.get('interval', '30m')
        return redirect(url_for('analyze', symbol=symbol, interval=interval))
    
    # Default popular symbols
    popular_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT']
    return render_template('index.html', popular_symbols=popular_symbols)

@app.route('/analyze')
def analyze():
    symbol = request.args.get('symbol', 'BTCUSDT').upper()
    interval = request.args.get('interval', '30m')
    
    analyzer = BinanceTradingAnalyzer(
        api_key=app.config['BINANCE_API_KEY'],
        api_secret=app.config['BINANCE_API_SECRET']
    )
    
    try:
        # Fetch and process data
        df = analyzer.get_historical_data(symbol, interval, 100)
        if df is None or len(df) == 0:
            flash('Failed to fetch historical data', 'danger')
            return redirect(url_for('index'))
        
        current_price = analyzer.get_current_price(symbol)
        if current_price is None:
            flash('Failed to fetch current price', 'danger')
            return redirect(url_for('index'))
        
        # Update dataframe with current price
        new_row = {
            'timestamp': pd.to_datetime(datetime.now()),
            'close': current_price,
            'open': current_price,
            'high': current_price,
            'low': current_price,
            'volume': 0
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Perform analysis
        df = analyzer.perform_technical_analysis(df)
        signals = analyzer.generate_trading_signals(df)
        recommendation = analyzer.generate_recommendation(df, signals)
        
        # Generate chart image
        fig = analyzer.visualize_analysis(df, symbol)
        if fig is None:
            flash('Not enough data to generate chart', 'warning')
            chart_url = None
        else:
            img = BytesIO()
            fig.savefig(img, format='png', bbox_inches='tight')
            plt.close(fig)
            img.seek(0)
            chart_url = base64.b64encode(img.getvalue()).decode('utf8')
        
        return render_template('analysis.html',
                            symbol=symbol,
                            interval=interval,
                            current_price=current_price,
                            signals=signals,
                            recommendation=recommendation,
                            chart=chart_url,
                            report=analyzer.generate_report(df, signals, recommendation))
    
    except Exception as e:
        app.logger.error(f"Error in analysis: {str(e)}")
        flash(f'An error occurred during analysis: {str(e)}', 'danger')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)