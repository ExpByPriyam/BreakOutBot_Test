import ccxt
import pandas as pd
import numpy as np
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
    cwidth_window = 300  # Channel width calculation window
    cwidthu = 3 / 100  # Channel width percentage
    mintest = 2  # Minimum test value

    # Calculate the highest and lowest values for width
    df['highest'] = df['high'].rolling(window=prd).max()
    df['lowest'] = df['low'].rolling(window=prd).min()
    df['chwidth'] = (df['high'].rolling(window=cwidth_window).max(
    ) - df['low'].rolling(window=cwidth_window).min()) * cwidthu
    print(df['chwidth'])
    # Identify pivot highs and pivot lows
    df['ph'] = df['high'].rolling(
        window=prd*2+1, center=True).apply(lambda x: x.argmax() == prd, raw=True) * df['high']
    df['pl'] = df['low'].rolling(
        window=prd*2+1, center=True).apply(lambda x: x.argmin() == prd, raw=True) * df['low']

    # Replace 0 with NaN for 'ph' and 'pl'
    df.loc[df['ph'] == 0, 'ph'] = np.nan
    df.loc[df['pl'] == 0, 'pl'] = np.nan

    # Store original non-NaN pivot highs and lows
    df['original_ph'] = df['ph'].copy()
    df['original_pl'] = df['pl'].copy()

    # Forward fill and backfill to manage NaNs
    df['ph'] = df['ph'].ffill().bfill()
    df['pl'] = df['pl'].ffill().bfill()

    # Identify breakout conditions
    df['bmax'] = df['ph'] - df['chwidth']
    df['bmin'] = df['pl'] + df['chwidth']

    # Calculate breakout condition
    df['breakout'] = (df['close'] > df['ph']) & (
        df['close'] > df['open']) & (df['close'] > df['highest'].shift(1))

    chunk_size = 200
    breakout_indices = []
    ph_occurrences_array = []

    # Iterate over the DataFrame in chunks of 200 rows
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        original_ph_occurrences = 0

        # Check for breakouts in the current chunk
        for index, row in chunk.iterrows():
            if pd.notna(row['original_ph']):
                original_ph_occurrences += 1
                ph_occurrences_array.append(row['original_ph'])
                num = 0
                bomax = np.nan
            if row['breakout'] and original_ph_occurrences >= mintest:
                bomax = ph_occurrences_array[0]
                xx = 0
                for x in range(len(ph_occurrences_array)):
                    if ph_occurrences_array[x] >= row['close']:
                        break
                    xx = x
                    bomax = max(bomax, ph_occurrences_array[x])
                if xx >= mintest and row['open'] <= bomax:
                    for x in range(xx + 1):
                        if ph_occurrences_array[x] <= bomax and ph_occurrences_array[x] >= bomax - row['chwidth']:
                            num += 1
                    hgst = df['highest'].shift(1)[index]
                    if num < mintest or hgst >= bomax:
                        bomax = np.nan  # Invalidate breakout
                if bomax != np.nan and num >= mintest:
                    breakout_indices.append(index)
                    print(
                        f"Breakout detected at {row['timestamp']}, Count of original non-NaN 'ph' occurrences up to this point: {original_ph_occurrences}")
                    print(
                        f"original_ph_occurrences array: {ph_occurrences_array}")
                print("***********Bmax ", bomax)
                print("***********Num ", num)
    # Plotting with Plotly
    fig = go.Figure()

    fig.add_trace(go.Candlestick(x=df['timestamp'],
                                 open=df['open'],
                                 high=df['high'],
                                 low=df['low'],
                                 close=df['close'],
                                 name='Market Data'))

    # Add breakout markers
    fig.add_trace(go.Scatter(x=df.loc[breakout_indices, 'timestamp'],
                             y=df.loc[breakout_indices, 'ph'],
                             mode='markers',
                             marker=dict(
                                 color='blue', symbol='triangle-up', size=10),
                             name='Breakout'))

    fig.update_layout(title=f'{symbol} Breakout Finder',
                      yaxis_title='Price (USDT)',
                      xaxis_title='Date',
                      xaxis_rangeslider_visible=False,
                      template='plotly_dark')

    fig.show()
else:
    print("No data fetched to plot.")
