import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

"""
Rubber Band Trading Strategy:
- Calculate a 5-day average of (High - Low), it is called "ATR" -> it tells you how the stock is moving over 5 days
- Find the highest price in the last 5 days
- Let this 5-day high - 2.5 * ATR = "buy level"
- If the day ends with price below the "buy level", buy at the close, expecting it to be bounce back soon
- Sell when the price closes above yesterday's high
"""
# summary logic: If the price suddenly drops far below recent highs (based on volatility), buy it — and sell it when it rebounds just above yesterday’s high.

ticker = "SPY"
start_date = "2010-01-01"
end_date = "2024-12-31"

df = yf.download(ticker, start=start_date, end=end_date)[['High', 'Low', 'Close']]
df.columns = df.columns.get_level_values(0)


# get the parameters
df["ATR"] = (df["High"] - df["Low"]).rolling(window=5).mean()
df["5D_High"] = df["High"].rolling(window=5).max()
df["Buy_Level"] = df["5D_High"] - 2.5 * df["ATR"]
df["Yesterday_High"] = df["High"].shift(1)

# generate the buy signal
df["signal"] = 0 # 1 for buy, 0 for no action, -1 for sell
df["position"] = 0


for i in range(1, len(df)):
    # entry if the price is below the buy level
    if df["Close"].iloc[i] < df['Buy_Level'].iloc[i]:
        df.at[df.index[i], "signal"] = 1
        df.at[df.index[i], "position"] = 1 # enter a position

    # exit if the price is above yesterday's high
    elif df["position"].iloc[i-1] == 1 and df["Close"].iloc[i] > df["Yesterday_High"].iloc[i]:
        df.at[df.index[i], "signal"] = -1
        df.at[df.index[i], "position"] = 0 # exit the position
    
    else:
        df.at[df.index[i], "position"] = df["position"].iloc[i-1] # hold the position

# Calculate returns
df["Market_Return"] = df["Close"].pct_change()
df["Strategy_Return"] = df["Market_Return"] * df["position"].shift(1)

# Calculate cumulative returns
df["Cumulative_Market_Return"] = (1 + df["Market_Return"]).cumprod()
df["Cumulative_Strategy_Return"] = (1 + df["Strategy_Return"]).cumprod()

# Plot the results -> using strategy vs simply holding SPY
plt.figure(figsize=(12, 6))
plt.plot(df['Cumulative_Market_Return'], label='Buy & Hold SPY')
plt.plot(df['Cumulative_Strategy_Return'], label='Rubber Band Strategy')
plt.title('Rubber Band Trading Strategy vs Buy & Hold (SPY)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

"""
!!! Lower returns lower risk !!!
Rubber Band might avoid losses, but misses gains
It avoids buying into falling markets or overbought conditions.
But in strong upward trends, the price never dips enough to trigger the "buy", so it stays out — and misses the gain.
"""