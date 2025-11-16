import ccxt
import pandas as pd
import plotly.graph_objects as go

# Initialize the exchange
exchange = ccxt.bybit()

# Fetch OHLCV data
symbol = 'BTC/USDT'
timeframe = '15m'
limit = 1000

# Fetch OHLCV data
try:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
except Exception as e:
    print(f"Error fetching data: {e}")
    ohlcv = []

# Proceed only if data was fetched successfully
if ohlcv:
    # Create DataFrame
    df = pd.DataFrame(
        ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Parameters
    prd = 5  # Period for rolling calculations
    cwidthu = 3 / 100  # Channel width percentage

    # Calculate the highest and lowest values for width
    df['highest'] = df['high'].rolling(window=prd).max()
    df['lowest'] = df['low'].rolling(window=prd).min()
    df['chwidth'] = (df['highest'] - df['lowest']) * cwidthu

    # Identify pivot highs and pivot lows
    df['ph'] = df['high'].rolling(
        window=prd*2+1, center=True).apply(lambda x: x.argmax() == prd, raw=True) * df['high']
    df['pl'] = df['low'].rolling(
        window=prd*2+1, center=True).apply(lambda x: x.argmin() == prd, raw=True) * df['low']

    # Replace 0 with NaN for 'ph' and 'pl'
    df.loc[df['ph'] == 0, 'ph'] = pd.NA
    df.loc[df['pl'] == 0, 'pl'] = pd.NA

    # Forward fill and backfill to manage NaNs
    df['ph'] = df['ph'].ffill().bfill()
    df['pl'] = df['pl'].ffill().bfill()

    # Identify breakout conditions

    df['bmax'] = df['ph'] - df['chwidth']
    df['bmin'] = df['pl'] + df['chwidth']

    # Breakout and breakdown conditions
    df['breakout'] = (df['close'] > df['ph']) & (
        df['close'] > df['open']) & (df['close'] > df['highest'].shift(1))
    df['breakdown'] = (df['close'] < df['pl']) & (
        df['close'] < df['open']) & (df['close'] < df['lowest'].shift(1))

    # Plotting with Plotly
    fig = go.Figure()

    fig.add_trace(go.Candlestick(x=df['timestamp'],
                                 open=df['open'],
                                 high=df['high'],
                                 low=df['low'],
                                 close=df['close'],
                                 name='Market Data'))

    fig.add_trace(go.Scatter(x=df.loc[df['breakout'], 'timestamp'],
                             y=df.loc[df['breakout'], 'ph'],
                             mode='markers',
                             marker=dict(
                                 color='blue', symbol='triangle-up', size=10),
                             name='Breakout'))

    fig.add_trace(go.Scatter(x=df.loc[df['breakdown'], 'timestamp'],
                             y=df.loc[df['breakdown'], 'pl'],
                             mode='markers',
                             marker=dict(
                                 color='red', symbol='triangle-down', size=10),
                             name='Breakdown'))
    fig.update_layout(title=f'{symbol} Breakout Finder',
                      yaxis_title='Price (USDT)',
                      xaxis_title='Date',
                      xaxis_rangeslider_visible=False,
                      template='plotly_dark')

    fig.show()
else:
    print("No data fetched to plot.")
