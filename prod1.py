import ccxt
import pandas as pd
import plotly.graph_objects as go

# Initialize the exchange
exchange = ccxt.binance()

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

    # Store original non-NaN pivot highs and lows
    df['original_ph'] = df['ph'].copy()
    df['original_pl'] = df['pl'].copy()

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

    chunk_size = 200
    breakout_indices = []
    breakdown_indices = []

    # Iterate over the DataFrame in chunks of 200 rows
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        original_ph_occurrences = 0
        original_pl_occurrences = 0

        # Check for breakouts and breakdowns in the current chunk
        for index, row in chunk.iterrows():
            if pd.notna(row['original_ph']):
                original_ph_occurrences += 1
            if pd.notna(row['original_pl']):
                original_pl_occurrences += 1
            if row['breakout'] and original_ph_occurrences >= 2:
                breakout_indices.append(index)
                print(
                    f"Breakout detected at {row['timestamp']}, Count of original non-NaN 'ph' occurrences up to this point: {original_ph_occurrences}")
            if row['breakdown'] and original_pl_occurrences >= 2:
                breakdown_indices.append(index)
                print(
                    f"Breakdown detected at {row['timestamp']}, Count of original non-NaN 'pl' occurrences up to this point: {original_pl_occurrences}")

    # Plotting with Plotly
    fig = go.Figure()

    fig.add_trace(go.Candlestick(x=df['timestamp'],
                                 open=df['open'],
                                 high=df['high'],
                                 low=df['low'],
                                 close=df['close'],
                                 name='Market Data'))

    # Add breakout and breakdown markers
    fig.add_trace(go.Scatter(x=df.loc[breakout_indices, 'timestamp'],
                             y=df.loc[breakout_indices, 'ph'],
                             mode='markers',
                             marker=dict(
                                 color='blue', symbol='triangle-up', size=10),
                             name='Breakout'))

    fig.add_trace(go.Scatter(x=df.loc[breakdown_indices, 'timestamp'],
                             y=df.loc[breakdown_indices, 'pl'],
                             mode='markers',
                             marker=dict(
                                 color='red', symbol='triangle-down', size=10),
                             name='Breakdown'))

    fig.update_layout(title=f'{symbol} Breakout and Breakdown Finder',
                      yaxis_title='Price (USDT)',
                      xaxis_title='Date',
                      xaxis_rangeslider_visible=False,
                      template='plotly_dark')

    fig.show()
else:
    print("No data fetched to plot.")
