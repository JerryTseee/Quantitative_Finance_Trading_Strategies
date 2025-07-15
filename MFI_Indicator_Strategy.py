import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

"""
The Money Flow Index (MFI) is a momentum indicator designed to gauge the inflow and outflow
funds within a security over a specific time frame. By looking at price and volume, it tries
to gain ideas into the market dynamics.

MFI 在 0 到 100 之间震荡，显示超买和超卖情况，是识别潜在市场反转点的工具

- If the two-day MFI is below 10, we buy at the close
- We sell at the close when the close ends higher than yesterday's high
- We have a time stop of 10 trading days
"""

# MFI = price + volume, oscillates between 0 and 100
# MFI < 20 stock is oversold -> may signal a bounce soon
# MFI > 80 stock is overbought
# using 2-day MFI can reacts very quickly to changes

def compute_mfi(df, period = 2):
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    money_flow = typical_price * df["Volume"]
    pos_flow = []
    neg_flow = []

    for i in range(1, len(df)):
        if typical_price.values[i] > typical_price.values[i-1]:
            pos_flow.append(money_flow.iloc[i])
            neg_flow.append(0)
        else:
            pos_flow.append(0)
            neg_flow.append(money_flow.iloc[i])
    
    pos_flow = pd.Series(pos_flow, index=df.index[1:])
    neg_flow = pd.Series(neg_flow, index=df.index[1:])

    mfr = pos_flow.rolling(period).sum() / neg_flow.rolling(period).sum()
    mfi = 100 - (100 / (1 + mfr))

    mfi = mfi.reindex(df.index)
    return mfi

ticker = "SPY"
start_date = "2010-01-01"
end_date = "2024-12-31"

df = yf.download(ticker, start=start_date, end=end_date)
df.dropna(inplace=True)

df["MFI_2"] = compute_mfi(df, period=2)
df["Yesterday_High"] = df["High"].shift(1)



df["Signal"] = 0
df["Position"] = 0
holding_days = 0

for i in range(2, len(df)):
    if df["Position"].iloc[i-1] == 0:
        if df["MFI_2"].iloc[i] < 10:
            df.at[df.index[i], "Signal"] = 1
            df.at[df.index[i], "Position"] = 1
            holding_days = 1
        else:
            df.at[df.index[i], "Position"] = 0

    elif df["Position"].iloc[i-1] == 1:
        holding_days += 1
        if float(df["Close"].iloc[i]) > float(df["Yesterday_High"].iloc[i]) or holding_days >= 10:
            df.at[df.index[i], "Signal"] = -1
            df.at[df.index[i], "Position"] = 0
            holding_days = 0
        else:
            df.at[df.index[i], "Position"] = 1


# Calculate returns
df["Market_Return"] = df["Close"].pct_change()
df["Strategy_Return"] = df["Market_Return"] * df["Position"].shift(1)

# Calculate cumulative returns
df["Cumulative_Market_Return"] = (1 + df["Market_Return"]).cumprod()
df["Cumulative_Strategy_Return"] = (1 + df["Strategy_Return"]).cumprod()

# --- Plot
plt.figure(figsize=(12, 6))
plt.plot(df['Cumulative_Market_Return'], label='Buy & Hold SPY')
plt.plot(df['Cumulative_Strategy_Return'], label='MFI Strategy (2-day)')
plt.title('MFI Strategy vs Buy & Hold (SPY)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()