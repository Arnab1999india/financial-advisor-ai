import pandas as pd
import pandas_ta as ta

def apply_swing_trading(df: pd.DataFrame, data_type: str = 'swing') -> pd.DataFrame:
    """
    Applies a basic swing trading strategy based on moving average crossovers.
    - Buy when a short-term moving average crosses above a long-term one.
    - Sell when the short-term moving average crosses below.
    """
    # Adjust parameters based on data type
    short_window = 20 if data_type == 'swing' else 10
    long_window = 50 if data_type == 'swing' else 30

    df['short_ma'] = ta.sma(df['Close'], length=short_window)
    df['long_ma'] = ta.sma(df['Close'], length=long_window)

    df['signal'] = 0
    df.loc[df['short_ma'] > df['long_ma'], 'signal'] = 1
    df.loc[df['short_ma'] < df['long_ma'], 'signal'] = -1

    df['buy_signal'] = (df['signal'] == 1) & (df['signal'].shift(1) == -1)
    df['sell_signal'] = (df['signal'] == -1) & (df['signal'].shift(1) == 1)

    return df

def apply_scalping_trading(df: pd.DataFrame, data_type: str = 'intraday') -> pd.DataFrame:
    """
    Applies a basic scalping strategy using RSI and a fast moving average.
    - Buy when RSI is oversold.
    - Sell when RSI is overbought.
    """
    # Adjust parameters based on data type
    rsi_period = 14 if data_type == 'swing' else 7
    
    df['rsi'] = ta.rsi(df['Close'], length=rsi_period)
    
    df['buy_signal'] = df['rsi'] < 30
    df['sell_signal'] = df['rsi'] > 70
    
    return df

def apply_44_moving_average(df: pd.DataFrame, data_type: str = 'swing') -> pd.DataFrame:
    """
    Applies the 44-period moving average strategy.
    - Buy when the price is above the 44 MA.
    - Sell when the price is below the 44 MA.
    """
    df['ma_44'] = ta.sma(df['Close'], length=44)
    
    df['buy_signal'] = df['Close'] > df['ma_44']
    df['sell_signal'] = df['Close'] < df['ma_44']
    
    return df

def apply_trend_trading(df: pd.DataFrame, data_type: str = 'swing') -> pd.DataFrame:
    """
    Applies a trend trading strategy using MACD.
    - Buy on a bullish MACD crossover.
    - Sell on a bearish MACD crossover.
    """
    macd = ta.macd(df['Close'])
    df['macd_line'] = macd['MACD_12_26_9']
    df['signal_line'] = macd['MACDs_12_26_9']

    df['buy_signal'] = (df['macd_line'] > df['signal_line']) & (df['macd_line'].shift(1) < df['signal_line'].shift(1))
    df['sell_signal'] = (df['macd_line'] < df['signal_line']) & (df['macd_line'].shift(1) > df['signal_line'].shift(1))

    return df

def apply_range_trading(df: pd.DataFrame, data_type: str = 'swing') -> pd.DataFrame:
    """
    Applies a range trading strategy using Bollinger Bands.
    - Buy when the price touches the lower band.
    - Sell when the price touches the upper band.
    """
    bollinger = ta.bbands(df['Close'], length=20, std=2)
    df['upper_band'] = bollinger['BBU_20_2.0']
    df['lower_band'] = bollinger['BBL_20_2.0']

    df['buy_signal'] = df['Close'] <= df['lower_band']
    df['sell_signal'] = df['Close'] >= df['upper_band']

    return df

# A dictionary to map strategy names to functions
STRATEGIES = {
    "swing_trading": apply_swing_trading,
    "scalping_trading": apply_scalping_trading,
    "44_moving_average": apply_44_moving_average,
    "trend_trading": apply_trend_trading,
    "range_trading": apply_range_trading,
}
