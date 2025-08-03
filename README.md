# 📊 Algo-Trading System with ML & Google Sheets Automation

This project is a **Python-based mini algorithmic trading system** that:
- Analyzes NIFTY 50 stock data using a rule-based strategy (RSI + MA crossover),
- Predicts price movement using **Logistic Regression**,
- Logs trades and ML results to **Google Sheets**,
- Sends **Telegram alerts** for signals and failures.

---

## ✅ Features

### 📈 Trading Strategy
- **Buy Signal**:
  - `RSI < 30` (oversold)
  - Confirmed by `20-DMA > 50-DMA` (bullish trend)
- **Backtest**:
  - Simulated 5-day holding period
  - Calculates P&L and win ratio
  - Logs trade data to Sheets

### 🤖 ML Prediction (Bonus)
- **Model**: Logistic Regression
- **Inputs**: RSI, MACD, Moving Averages, Normalized Volume
- **Output**: Predicts next-day direction (UP / DOWN)
- **Accuracy**: Displayed in Google Sheets

### 📊 Google Sheets Integration
- **3 tabs**:
  - `Trade Log`: All trades with P&L
  - `ML Summary`: Accuracy + prediction
  - `Summary Stats`: Total trades, win %, total P&L
- Uses **GSpread API** and a `credentials.json` file

### 🔔 Telegram Alerts
- Sends message when a **buy signal** is generated
- Sends alert if the **algo fails**

---

## 📂 Project Structure
```
htoh/
├── run_algo.py # Main orchestration script
├── datafetching.py # Stock data functions (Yahoo Finance)
├── predictor.py # ML model training + prediction
├── gsheetconnect.py # Google Sheets logic
├── requirements.txt # Python dependencies
├── .env.example # Env variable sample (create your .env)
├── algo_log.txt # Logs
└── stock_data/ # Cached CSVs for each stock
```

---

## 🛠 Setup Instructions

### 1. Clone Repo & Install Dependencies

```bash
git clone https://github.com/Janavee01/htoh.git
cd htoh
python -m venv htohenv
htohenv\Scripts\activate   # On Windows
pip install -r requirements.txt
```

### 2. Setup .env File
Create a .env file in the root directory:
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

### 3. Add Google Sheets Credentials
Create a Google Cloud Project
Enable Google Sheets API and Google Drive API
Generate a service account key as credentials.json
Share your Google Sheet with the service account email

### 4. Run the Algo
python run_algo.py

## 🖥 Sample Output

```
=== RELIANCE.NS ===
RELIANCE.NS: Buy Signals Found: 1
Close RSI 20DMA 50DMA
Price
2025-07-28 1387.599976 29.713744 1477.904999 1455.514006

=== INFY.NS ===
INFY.NS: Buy Signals Found: 0
Empty DataFrame

=== TCS.NS ===
TCS.NS: Buy Signals Found: 0
Empty DataFrame

🧾 All Trades:
['RELIANCE.NS', '2025-07-28', 1387.6, '2025-08-01', 1393.7, 0.44, 'Win']

📊 Writing summary stats to tab: Summary Stats
🔍 Summary Stats:
[['Total Trades', 1], ['Wins', 1], ['Losses', 0], ['Win Ratio (%)', 100.0], ['Total P&L (%)', 0.44]]
```

Telegram:
```
📈 Buy Signal for RELIANCE.NS
Entry Date: 2025-07-28
Entry Price: ₹1387.6
Exit Price (5D): ₹1393.7
P&L: 0.44% — Win
```

**Google Sheets: Trade log, ML summary, and stats auto-filled.**

---

📧 Contact
Made by Janavee01

🔐 **Disclaimer**
Do not use this for real trading without testing.
Educational / portfolio project only.
