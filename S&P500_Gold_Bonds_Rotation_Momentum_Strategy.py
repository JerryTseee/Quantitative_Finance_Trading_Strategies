import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Trading Rules:

Assets:
S&P 500 (Stocks): This is typically represented by the SPY ETF or a similar instrument.
Bonds: A common proxy for bonds could be the US 10-Year Treasury Bond ETF (e.g., TLT) or a broad bond index ETF.
Gold: A typical proxy for gold is the GLD ETF.

Momentum Signal:
The momentum signal is based on comparing the 3-month Simple Moving Average (SMA) and the 10-month SMA for each asset.
If the 3-month SMA is greater than the 10-month SMA, that asset is considered to be "bullish" (uptrend).
If the 3-month SMA is less than the 10-month SMA, that asset is considered to be "bearish" (downtrend).

Capital Allocation:
If an asset is "bullish" (3-month SMA > 10-month SMA), you allocate capital to it.
The capital is equally divided among all assets that are showing a bullish signal. For example:
If only one asset is bullish, allocate 100% of the capital to that asset.
If two assets are bullish, allocate 50% to each.
If all three assets are bullish, allocate 33.33% to each.
"""

# Define tickers for S&P 500, Bonds, and Gold
tickers = ['SPY', 'TLT', 'GLD']

# Download data (3 years of data)
data = yf.download(tickers, start="2017-01-01", end="2024-01-01")['Close']

# Calculate the 3-month SMA and 10-month SMA for each asset
short_sma = data.rolling(window=63).mean()  # 3-month SMA (approx. 63 trading days)
long_sma = data.rolling(window=210).mean()  # 10-month SMA (approx. 210 trading days)

# Generate momentum signals (1 = bullish, 0 = bearish)
signals = (short_sma > long_sma).astype(int)



# Initialize portfolio and allocation
portfolio = pd.DataFrame(index=data.index, columns=tickers)
allocation = pd.DataFrame(index=data.index, columns=tickers)


# Capital Allocation based on the number of bullish signals
for i in range(1, len(data)):
    # Count how many assets are bullish
    bullish_count = signals.iloc[i].sum()
    
    # If bullish signals exist, allocate capital
    if bullish_count > 0:
        # Calculate equal allocation
        allocation.iloc[i] = (signals.iloc[i] / bullish_count)
    else:
        allocation.iloc[i] = 0

    # Portfolio value is the allocation multiplied by the returns
    portfolio.iloc[i] = allocation.iloc[i] * data.pct_change().iloc[i]

# Calculate cumulative returns for each asset and strategy
portfolio['Strategy_Return'] = portfolio.sum(axis=1)
portfolio['Cumulative_Strategy_Return'] = (1 + portfolio['Strategy_Return']).cumprod()



# Plot the cumulative returns of the strategy and individual assets (Buy & Hold)
plt.figure(figsize=(10, 6))
# Plot the Momentum Rotation Strategy
plt.plot(portfolio['Cumulative_Strategy_Return'], label='Momentum Rotation Strategy')

# Plot Buy & Hold for each asset separately
for ticker in tickers:
    plt.plot((1 + data[ticker].pct_change()).cumprod(), label=f'{ticker} Buy & Hold')

plt.title('Momentum Rotation Strategy vs Buy & Hold')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
