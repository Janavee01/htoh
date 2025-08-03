import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

# Create a directory to store CSVs (optional)
os.makedirs("stock_data", exist_ok=True)

# Define stock symbols (example NIFTY 50 stocks)
stock_symbols = ['RELIANCE.NS', 'INFY.NS', 'TCS.NS']

# Set date range for 6 months
end_date = datetime.today()
start_date = end_date - timedelta(days=180)

# Dictionary to store DataFrames
stock_data = {}

print("=== Starting Stock Data Ingestion ===\n")

for symbol in stock_symbols:
    print(f"🔄 Fetching data for {symbol}...")
    try:
        df = yf.download(symbol, start=start_date, end=end_date, interval='1d', progress=False)
        df.dropna(inplace=True)

        if not df.empty:
            stock_data[symbol] = df
            df.to_csv(f"stock_data/{symbol}_data.csv")  # Optional: Save CSV
            print(f"✅ Data fetched: {df.shape[0]} rows")
        else:
            print(f"⚠️ No data returned for {symbol}")

    except Exception as e:
        print(f"❌ Error fetching data for {symbol}: {e}")

print("\n=== Sample Data for RELIANCE ===")
if 'RELIANCE.NS' in stock_data:
    print(stock_data['RELIANCE.NS'].head())
else:
    print("❌ RELIANCE.NS data not available.")

