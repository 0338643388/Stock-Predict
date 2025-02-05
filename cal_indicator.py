import numpy as np
import pandas as pd

def psar(df, af=0.02, af_max=0.2):
    high, low = df["High"].values, df["Low"].values
    psar = np.zeros(len(df))
    trend = 1  
    ep = low[0]  
    af_step = af  

    for i in range(1, len(df)):
        psar[i] = psar[i - 1] + af_step * (ep - psar[i - 1])

        if trend == 1:
            if low[i] < psar[i]:  
                trend = -1
                psar[i] = ep
                af_step = af
                ep = high[i]
            else:
                if high[i] > ep:
                    ep = high[i]
                    af_step = min(af_step + af, af_max)
        else:
            if high[i] > psar[i]:  
                trend = 1
                psar[i] = ep
                af_step = af
                ep = low[i]
            else:
                if low[i] < ep:
                    ep = low[i]
                    af_step = min(af_step + af, af_max)

    df["PSAR"] = psar
    return df

def cal_tech_indicators(stocks):
    for company, data in stocks.items():
        # data = data[::-1]
        short_ema = data["Close"].ewm(span=12, adjust=False).mean()
        long_ema = data["Close"].ewm(span=26, adjust=False).mean()
        data["MACD"] = short_ema - long_ema
        data["Signal_Line"] = data["MACD"].ewm(span=9, adjust=False).mean()
        
        window_length = 14
        close_delta = data["Close"].diff()

        gain = close_delta.where(close_delta > 0, 0)
        loss = -close_delta.where(close_delta < 0, 0)

        avg_gain = gain.rolling(window=window_length, min_periods=1).mean()
        avg_loss = loss.rolling(window=window_length, min_periods=1).mean()

        rs = avg_gain / avg_loss
        data["RSI"] = 100 - (100 / (1 + rs))
        
        data["OBV"] = (data["Volume"] * (~data["Close"].diff().lt(0) * 2 - 1)).cumsum()

        data = psar(data)
        
        data["SMA_10"] = data["Close"].rolling(window=10).mean()
        data["SMA_50"] = data["Close"].rolling(window=50).mean()
        data["EMA_50"] = data["Close"].ewm(span=50, adjust=False).mean()
        data["Change %"] = data['Close'].pct_change() * 100
        data["Target"] = data["Close"].shift(-1)
        
        data = data.dropna()
        stocks[company] = data
