"""
Research shows that stocks make almost all the gains during the last five trading days of the month and the first three trading days of the new month

We go long at the close on the fifth last trading day of the month, and we exit after seven days, ie. at the close of the third trading day of the next month.

Trading Rules:
Go Long on the fifth last trading day of the month at the close.
Exit on the third trading day of the next month at the close.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Fetch S&P 500 data from Yahoo Finance
ticker = "^GSPC"  # S&P 500 Cash Index
data = yf.download(ticker, start="1960-01-01", end="2024-01-01")

# Extract the relevant columns
data = data[['Close']]

# Calculate daily returns
data['Daily_Return'] = data['Close'].pct_change()

# Add a 'Month' and 'Day' columns for filtering purposes
data['Month'] = data.index.month
data['Day'] = data.index.day

# Define a function to select the "turn of the month" period
def turn_of_the_month_filter(date):
    # Last five trading days of the month (from 26th to end of month)
    if date.day >= 26:
        return True
    # First three trading days of the new month
    elif date.day <= 3:
        return True
    return False

# Apply the filter to identify "turn of the month" days
data['Turn_of_the_Month'] = data.index.to_series().apply(turn_of_the_month_filter)

# Get returns only for "turn of the month" days
turn_of_the_month_returns = data[data['Turn_of_the_Month']]['Daily_Return']

# Calculate the strategy's cumulative returns
turn_of_the_month_cumulative_returns = (1 + turn_of_the_month_returns).cumprod()

# Backtest the strategy performance by comparing it to a simple buy and hold strategy
# Calculate cumulative returns for the entire dataset
cumulative_returns = (1 + data['Daily_Return']).cumprod()


# Plot both the "turn of the month" strategy and buy-and-hold strategy
plt.figure(figsize=(10, 6))
cumulative_returns.plot(label='Buy and Hold', color='orange')
turn_of_the_month_cumulative_returns.plot(label='Turn of the Month', color='blue')
plt.legend()
plt.title("Backtest: Turn of the Month vs. Buy and Hold Strategy")
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.show()
