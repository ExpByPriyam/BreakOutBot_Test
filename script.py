import pandas as pd
import numpy as np
import ccxt
from datetime import datetime
import plotly.graph_objects as go


def get_crypto_data(symbol="BTC/USDT", timeframe="15m", exchange="bybit", since=None, limit=1000):
    ex = getattr(ccxt, exchange)()
    _ = ex.load_markets()
    data = ex.fetch_ohlcv(symbol, timeframe=timeframe,
                          since=since, limit=limit)
    df = pd.DataFrame.from_records(
        data, columns=['date', "open", "high", "low", "close", "volume"])
    df.date = df.date.apply(lambda x: datetime.utcfromtimestamp(x//1000))
    return df


class Line:

    def __init__(self, x1, y1, x2, y2, color=None, style=None, dt=None):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.color = color
        self.style = style
        self.dt = dt

    def set_xy1(self, x, y):
        self.x1 = x
        self.y1 = y

    def set_xy2(self, x, y):
        self.x2 = x
        self.y2 = y

    def set_x2(self, x):
        self.x2 = x

    def __repr__(self):
        x1 = self.x1 if self.dt is None else self.dt[self.x1]
        x2 = self.x2 if self.dt is None else (
            self.dt[self.x2] if self.x2 < len(self.dt) else self.dt[len(self.dt)-1])
        return f"line from {x1}, {self.y1} to {x2}, {self.y2}"


class Label:

    def __init__(self, x=None, y=None, text=None, color=None, dt=None):
        self.x = x
        self.y = y
        self.text = text
        self.color = color
        self.dt = dt

    def set_xy(self, x, y):
        self.x = x
        self.y = y

    def set_text(self, t):
        self.text = t

    def __repr__(self):
        return f"Label on bar {self.x if self.dt is None else (self.x, self.dt[self.x])} with text {self.text}"


def pivot(osc, LBL, LBR, highlow):

    def checkhl(data_back, data_forward, hl):
        if hl == 'high' or hl == 'High':
            ref = data_back[len(data_back)-1]
            for i in range(len(data_back)-1):
                if ref < data_back[i]:
                    return 0
            for i in range(len(data_forward)):
                if ref <= data_forward[i]:
                    return 0
            return 1
        if hl == 'low' or hl == 'Low':
            ref = data_back[len(data_back)-1]
            for i in range(len(data_back)-1):
                if ref > data_back[i]:
                    return 0
            for i in range(len(data_forward)):
                if ref >= data_forward[i]:
                    return 0
            return 1

    left = []
    right = []
    pivots = []
    for i in range(len(osc)):
        pivots.append(np.nan)
        if i < LBL + 1:
            left.append(osc[i])
        if i > LBL:
            right.append(osc[i])
        if i > LBL + LBR:
            left.append(right[0])
            left.pop(0)
            right.pop(0)
            if checkhl(left, right, highlow):
                pivots[i] = osc[i - LBR]
    return pivots


def highest(data, length):
    return np.array([np.nan]*length + [np.max(data[i-length+1: i+1]) for i in range(length, len(data))])


def lowest(data, length):
    return np.array([np.nan]*length + [np.min(data[i-length+1: i+1]) for i in range(length, len(data))])


def Highest(src, length, i):
    return np.max(src[i-length+1: i+1])


def Lowest(src, length, i):
    return np.min(src[i-length+1: i+1])


def na(x):
    return np.isnan(x)


def indicator(df):
    _open, high, low, close = df.open, df.high, df.low, df.close
    # input variables
    prd = 5
    bo_len = 200
    cwidthu = 3/100
    mintest = 2
    bocolorup = 'blue'
    bocolordown = 'red'
    lstyle = 'solid'

    lines = []
    labels = []

    # check if PH/PL
    ph = pivot(high, prd, prd, 'high')
    pl = pivot(low, prd, prd, 'low')

    hi = highest(high, prd)
    lo = lowest(low, prd)

    # keep Pivot Points and their locations in the arrays
    phval = []
    phloc = []
    plval = []
    plloc = []
    for i in range(len(close)):
        bar_index = i
        # width
        lll = max(min(bar_index, 300), 1)
        h_ = Highest(high, lll, i)
        l_ = Lowest(low, lll, i)
        chwidth = (h_ - l_) * cwidthu

        # keep PH/PL levels and locations
        if not np.isnan(ph[i]):
            phval.insert(0, ph[i])
            phloc.insert(0, bar_index - prd)
            if len(phval) > 1:  # cleanup old ones
                for x in range(len(phloc) - 1, 0, -1):
                    if bar_index - phloc[x] > bo_len:
                        phloc.pop()
                        phval.pop()

        if not np.isnan(pl[i]):
            plval.insert(0, pl[i])
            plloc.insert(0, bar_index - prd)
            if len(plval) > 1:  # cleanup old ones
                for x in range(len(plloc) - 1, 0, -1):
                    if bar_index - plloc[x] > bo_len:
                        plloc.pop()
                        plval.pop()

        # check bullish cup
        bomax = np.nan
        bostart = bar_index
        num = 0
        hgst = np.nan if i == 0 else hi[i-1]
        if len(phval) >= mintest and close[i] > _open[i] and close[i] > hgst:
            bomax = phval[0]
            xx = 0
            for x in range(len(phval)):
                if phval[x] >= close[i]:
                    break
                xx = x
                bomax = max(bomax, phval[x])
            if xx >= mintest and _open[i] <= bomax:
                for x in range(xx+1):
                    if phval[x] <= bomax and phval[x] >= bomax - chwidth:
                        num += 1
                        bostart = phloc[x]
                if num < mintest or hgst >= bomax:
                    bomax = np.nan

        if not na(bomax) and num >= mintest:
            lines.extend([
                Line(x1=bar_index, y1=bomax, x2=bostart,
                     y2=bomax, color=bocolorup, style=lstyle),
                Line(x1=bar_index, y1=bomax - chwidth, x2=bostart,
                     y2=bomax - chwidth, color=bocolorup, style=lstyle),
                Line(x1=bostart,   y1=bomax - chwidth, x2=bostart,
                     y2=bomax, color=bocolorup, style=lstyle),
                Line(x1=bar_index, y1=bomax - chwidth, x2=bar_index,
                     y2=bomax, color=bocolorup, style=lstyle)
            ])

            labels.append(
                Label(x=bar_index, y=low[i], color=bocolorup, text='triangleup', dt=df.date))

        # check bearish cup
        bomin = np.nan
        bostart = bar_index
        num1 = 0
        lwst = np.nan if i == 0 else lo[i-1]
        if len(plval) >= mintest and close[i] < _open[i] and close[i] < lwst:
            bomin = plval[0]
            xx = 0
            for x in range(len(plval)):
                if plval[x] <= close[i]:
                    break
                xx = x
                bomin = min(bomin, plval[x])
            if xx >= mintest and _open[i] >= bomin:
                for x in range(xx+1):
                    if plval[x] >= bomin and plval[x] <= bomin + chwidth:
                        num1 += 1
                        bostart = plloc[x]
                if num1 < mintest or lwst <= bomin:
                    bomin = np.nan

        if not na(bomin) and num1 >= mintest:
            lines.extend([
                Line(x1=bar_index, y1=bomin, x2=bostart,
                     y2=bomin, color=bocolordown, style=lstyle),
                Line(x1=bar_index, y1=bomin + chwidth, x2=bostart,
                     y2=bomin + chwidth, color=bocolordown, style=lstyle),
                Line(x1=bostart, y1=bomin + chwidth, x2=bostart,
                     y2=bomin, color=bocolordown, style=lstyle),
                Line(x1=bar_index, y1=bomin + chwidth, x2=bar_index,
                     y2=bomin, color=bocolordown, style=lstyle)
            ])
            labels.append(Label(
                x=bar_index, y=high[i], color=bocolordown, text='triangledown', dt=df.date))

    return lines, labels


def plot(df, lines, labels, show_last=300):
    check = len(df)-show_last
    df = df[-show_last:]
    plots = [go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='candlesticks'
    )]

    fig = go.Figure(data=plots)
    fig.update_layout(height=800, xaxis_rangeslider_visible=False)

    # Creating label traces
    for l in labels:
        if l.x < check:
            continue
        fig.add_trace(go.Scatter(
            x=[df.date[l.x]],
            y=[l.y],
            mode="markers+text",
            marker=dict(symbol='triangle-up-open' if 'up' in l.text else 'triangle-down-open',
                        size=12, color=l.color),
            showlegend=False))

    # add line traces
    for l in lines:
        if l.x1 < check and l.x2 < check:
            continue
        l.x1 = check if l.x1 < check else l.x1
        l.x2 = check if l.x2 < check else l.x2
        fig.add_trace(go.Scatter(
            x=[df.date[l.x1], df.date[l.x2]],
            y=[l.y1, l.y1],
            mode='lines',
            name='Line',
            line=dict(color=l.color, width=2),
            showlegend=False
        ))

    fig.show()


if __name__ == '__main__':
    df = get_crypto_data("SOL/USDT", timeframe='1m', limit=1200)
    lines, labels = indicator(df)
    # plot graph for last 200 candles
    plot(df, lines, labels, show_last=500)
