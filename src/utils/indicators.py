"""
Technical Indicators computation module.
Provides SMA, EMA, RSI, MACD, Bollinger Bands, ATR, VWAP, Stochastic Oscillator.
"""
import numpy as np
import pandas as pd

def calculate_sma(series: pd.Series, window: int = 20) -> pd.Series:
    return series.rolling(window=window).mean()

def calculate_ema(series: pd.Series, span: int = 20) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    # Exponential smoothing for RSI
    for i in range(period, len(series)):
        if not np.isnan(gain.iloc[i]):
            avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
            avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period
            
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = calculate_ema(series, span=fast)
    ema_slow = calculate_ema(series, span=slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, span=signal)
    histogram = macd_line - signal_line
    return pd.DataFrame({
        'MACD': macd_line,
        'Signal': signal_line,
        'Hist': histogram
    })

def calculate_bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    sma = calculate_sma(series, window=window)
    std = series.rolling(window=window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    bandwidth = (upper - lower) / sma
    return pd.DataFrame({
        'BB_Upper': upper,
        'BB_Middle': sma,
        'BB_Lower': lower,
        'BB_Bandwidth': bandwidth
    })

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    k_percent = 100 * ((df['Close'] - low_min) / (high_max - low_min + 1e-8))
    d_percent = k_percent.rolling(window=d_period).mean()
    return pd.DataFrame({'Stoch_K': k_percent, 'Stoch_D': d_percent})

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Enriches OHLCV dataframe with all technical indicators safely."""
    data = df.copy()
    if 'Close' not in data.columns:
        return data
        
    n_len = len(data)
    data['SMA_20'] = data['Close'].rolling(window=min(20, max(2, n_len)), min_periods=1).mean()
    data['SMA_50'] = data['Close'].rolling(window=min(50, max(2, n_len)), min_periods=1).mean()
    data['SMA_200'] = data['Close'].rolling(window=min(200, max(2, n_len)), min_periods=1).mean()
    data['EMA_12'] = calculate_ema(data['Close'], 12)
    data['EMA_26'] = calculate_ema(data['Close'], 26)
    
    data['RSI'] = calculate_rsi(data['Close'])
    
    macd_df = calculate_macd(data['Close'])
    data['MACD'] = macd_df['MACD'].bfill()
    data['MACD_Signal'] = macd_df['Signal'].bfill()
    data['MACD_Hist'] = macd_df['Hist'].bfill()
    
    bb_df = calculate_bollinger_bands(data['Close'])
    data['BB_Upper'] = bb_df['BB_Upper'].bfill()
    data['BB_Middle'] = bb_df['BB_Middle'].bfill()
    data['BB_Lower'] = bb_df['BB_Lower'].bfill()
    data['BB_Width'] = bb_df['BB_Bandwidth'].bfill()
    
    if 'High' in data.columns and 'Low' in data.columns:
        data['ATR'] = calculate_atr(data).bfill()
        stoch_df = calculate_stochastic(data)
        data['Stoch_K'] = stoch_df['Stoch_K'].bfill()
        data['Stoch_D'] = stoch_df['Stoch_D'].bfill()
        
    # Return features
    data['Daily_Return'] = data['Close'].pct_change().fillna(0)
    data['Volatility_20d'] = data['Daily_Return'].rolling(min(20, max(2, n_len)), min_periods=1).std() * np.sqrt(365)
    data['Volatility_20d'] = data['Volatility_20d'].fillna(0.3)
    
    return data
