import pandas as pd
import yfinance as yf
import ta
import os
import logging
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import gspread
from google.oauth2.service_account import Credentials

# === Logging Setup ===
logging.basicConfig(filename='algo_log.txt', level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')

# === CONFIG ===
STOCKS = ['RELIANCE.NS', 'INFY.NS', 'TCS.NS']

SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

DATA_FOLDER = 'stock_data'
os.makedirs(DATA_FOLDER, exist_ok=True)

SPREADSHEET_NAME = "AlgoTrading Logs"

# === Google Sheets Auth ===
creds = Credentials.from_service_account_file('credentials.json', scopes=SCOPES)
client = gspread.authorize(creds)
sheet = client.open(SPREADSHEET_NAME)
trade_tab = sheet.get_worksheet(0)
try:
    ml_tab = sheet.worksheet("ML Summary")
except gspread.exceptions.WorksheetNotFound:
    ml_tab = sheet.add_worksheet(title="ML Summary", rows=100, cols=10)

# === Dates ===
end_date = datetime.today()
start_date = end_date - timedelta(days=180)

def fetch_data(symbol):
    """
    Downloads historical stock data from Yahoo Finance for the given symbol,
    stores it in CSV, and returns a cleaned DataFrame.
    
    Parameters:
        symbol (str): The stock symbol (e.g., 'RELIANCE.NS')

    Returns:
        pd.DataFrame: Cleaned DataFrame of stock price data
    """
    df = yf.download(symbol, start=start_date, end=end_date, interval='1d', auto_adjust=False, progress=False)
    df.dropna(inplace=True)
    df.to_csv(os.path.join(DATA_FOLDER, f"{symbol}_data.csv"))
    return df

def calculate_summary_stats(trade_logs):
    """
    Computes summary statistics like win ratio, total P&L, etc.
    
    Parameters:
        trade_logs (list): List of trade entries [symbol, entry_date, entry_price, exit_date, exit_price, pnl, result]

    Returns:
        list: [Total Trades, Wins, Losses, Win Ratio %, Total P&L %]
    """
    total_trades = len(trade_logs)
    wins = sum(1 for trade in trade_logs if trade[-1] == 'Win')
    losses = total_trades - wins
    win_ratio = round((wins / total_trades) * 100, 2) if total_trades > 0 else 0
    total_pnl = round(sum(trade[5] for trade in trade_logs), 2)

    return [['Total Trades', total_trades],
            ['Wins', wins],
            ['Losses', losses],
            ['Win Ratio (%)', win_ratio],
            ['Total P&L (%)', total_pnl]]

def strategy(df):
    """
    Implements the RSI < 30 + Moving Average crossover strategy.

    Buy Signal Logic:
        - RSI < 30 (oversold condition)
        - 20-day MA > 50-day MA (bullish crossover)

    Parameters:
        df (pd.DataFrame): Stock data with Close prices

    Returns:
        pd.DataFrame: Rows where buy signals are generated
    """
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['20DMA'] = df['Close'].rolling(20).mean()
    df['50DMA'] = df['Close'].rolling(50).mean()
    df['Buy_Signal'] = (df['RSI'] < 30) & (df['20DMA'] > df['50DMA'])
    df.dropna(inplace=True)
    return df[df['Buy_Signal']]

def evaluate_trade(df, signal_date):
    """
    Backtests a buy signal by simulating an entry and exit after 5 days.
    
    Parameters:
        df (pd.DataFrame): Stock price data
        signal_date (datetime): The date of the buy signal

    Returns:
        tuple: (entry_price, exit_price, pnl, result, exit_date)
    """
    entry_price = df.loc[signal_date, 'Close']
    exit_index = df.index.get_loc(signal_date) + 5
    if exit_index < len(df):
        exit_date = df.index[exit_index]
        exit_price = df.iloc[exit_index]['Close']
    else:
        exit_date = df.index[-1]
        exit_price = df.iloc[-1]['Close']
    pnl = round(((exit_price - entry_price) / entry_price) * 100, 2)
    result = 'Win' if pnl > 0 else 'Loss'
    return entry_price, exit_price, pnl, result, exit_date

def ml_predict(df):
    """
    Trains a Logistic Regression model on RSI, MACD, MAs, and volume
    to predict the next day's stock movement direction.

    Parameters:
        df (pd.DataFrame): Stock price data with indicators

    Returns:
        tuple: (prediction accuracy %, predicted direction ['UP'/'DOWN'])
    """
     
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df['Volume_Norm'] = (df['Volume'] - df['Volume'].mean()) / df['Volume'].std()
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)

    features = ['RSI', 'MACD', '20DMA', '50DMA', 'Volume_Norm']
    X = df[features]
    y = df['Target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test).flatten()  # ← robust fix
    accuracy = accuracy_score(y_test, y_pred)

    next_day = scaler.transform(df[features].iloc[[-1]])
    prediction = model.predict(next_day)[0]

    return float(round(accuracy * 100, 2)), str("UP" if prediction == 1 else "DOWN")

def log_to_sheets(trade_logs, ml_results):
    """
    Writes the trade signals and ML predictions to Google Sheets.

    Parameters:
        trade_logs (list): List of trades to append
        ml_results (list): List of ML prediction summaries
    """
    if trade_logs:
        for chunk_start in range(0, len(trade_logs), 50):  # break into chunks of 50
            chunk = trade_logs[chunk_start:chunk_start+50]
            trade_tab.append_rows(chunk, value_input_option='USER_ENTERED')
    if ml_results:
        ml_tab.append_rows(ml_results, value_input_option='USER_ENTERED')

def main():
    """
    The main execution loop:
    - Loads stock data
    - Runs strategy
    - Backtests trades
    - Makes ML predictions
    - Logs everything to Google Sheets
    """
    all_trades = []
    ml_results = []  # ← store all ML results here
    try:
        for symbol in STOCKS:
            csv_path = os.path.join(DATA_FOLDER, f"{symbol}_data.csv")
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            numeric_cols = ['Close', 'Open', 'High', 'Low', 'Volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            df.dropna(subset=numeric_cols, inplace=True)

            signals = strategy(df)

            print(f"=== {symbol} ===")
            print(f"{symbol}: Buy Signals Found: {len(signals)}")
            print(signals[['Close', 'RSI', '20DMA', '50DMA']].tail())

            logging.info(f"{symbol}: {len(signals)} Buy Signals")

            for date in signals.index:
                entry, exit_, pnl, result, exit_date = evaluate_trade(df, date)
                # print(type(date), type(exit_date))
                all_trades.append([
                    symbol,
                    str(pd.to_datetime(date).date()),         # ← convert to string
                    round(entry, 2),
                    str(pd.to_datetime(exit_date).date()),    # ← convert to string
                    round(exit_, 2),
                    pnl,
                    result
                ])



            acc, pred = ml_predict(df)
            ml_results.append([symbol, str(datetime.today().date()), acc, pred])  # ← collect here

        # Write to Google Sheets
        print(all_trades)
        if all_trades or ml_results:
            print("\n🧾 All Trades:")
            for trade in all_trades:
                print(trade)

            log_to_sheets(all_trades, ml_results)

                        # Log summary stats
            summary_stats = calculate_summary_stats(all_trades)

            try:
                summary_tab = sheet.worksheet("Summary Stats")
                summary_tab.clear()
            except gspread.exceptions.WorksheetNotFound:
                summary_tab = sheet.add_worksheet(title="Summary Stats", rows=10, cols=2)

            print("📊 Writing summary stats to tab:", summary_tab.title)
            print("🔍 Summary Stats:", summary_stats)
            summary_tab.update(range_name='A1', values=summary_stats)

        logging.info("✅ Algo run completed successfully.")
    except Exception as e:
        # logging.error(f"❌ Algo failed: {e}")
            print("❌ Error during algo execution:")
            print(e)


if __name__ == "__main__":
    main()
