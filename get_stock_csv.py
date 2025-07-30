import yfinance as yf
import pandas as pd

def get_stock_data(ticker, start_date, end_date, save_csv=True):
    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    stock = yf.download(ticker, start=start_date, end=end_date)

    if save_csv:
        filename = f"{ticker}_stock_data.csv"
        stock.to_csv(filename)
        print(f"Data saved to {filename}")

# Example usage:
if __name__ == "__main__":
    ticker = input("input stock ticker: ")
    start_date = input("input start date (YYYY-MM-DD): ")
    end_date = input("input end date (YYYY-MM-DD): ")
    get_stock_data(ticker=ticker, start_date=start_date, end_date=end_date, save_csv=True)
