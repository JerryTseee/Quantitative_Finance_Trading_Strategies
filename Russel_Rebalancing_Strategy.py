import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

"""
Russel rebalancing strategy:
- buy once a year
- buy on the close of the first trading day after the 23rd of June
- sell on the close on the first trading day of July
"""
# Strategy Parameters
start_year = 2010
end_year = 2024
ticker = "IWM"  # Russell 2000 ETF

# Download historical data
df = yf.download(ticker, start=f"{start_year}-01-01", end=f"{end_year + 1}-01-01")[['Close']]

df.index = pd.to_datetime(df.index)

# Store trade results
trades = []

for year in range(start_year, end_year + 1):
    # Buy on first trading day after June 23
    june_23 = datetime(year, 6, 23)
    buy_day = df[df.index > june_23].index.min()

    # Sell on first trading day after July 1
    july_1 = datetime(year, 7, 1)
    sell_day = df[df.index > july_1].index.min()

    if pd.isna(buy_day) or pd.isna(sell_day):
        continue  # Skip if data is missing

    buy_price = df.loc[buy_day, ('Close', ticker)]
    sell_price = df.loc[sell_day, ('Close', ticker)]
    return_pct = (sell_price - buy_price) / buy_price * 100

    trades.append({
        "Year": year,
        "Buy Date": buy_day.date(),
        "Sell Date": sell_day.date(),
        "Buy Price": round(buy_price, 2),
        "Sell Price": round(sell_price, 2),
        "Return (%)": round(return_pct, 2)
    })

# Convert to DataFrame
results_df = pd.DataFrame(trades)

# Plot annual return percentages
plt.figure(figsize=(10, 6))
plt.bar(results_df['Year'], results_df['Return (%)'], color='skyblue', edgecolor='black')

# Add titles and labels
plt.title('Russell Rebalancing Strategy Annual Returns (IWM)', fontsize=14)
plt.xlabel('Year')
plt.ylabel('Return (%)')
plt.axhline(0, color='red', linestyle='--')  # Show zero line
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Show plot
plt.show()
