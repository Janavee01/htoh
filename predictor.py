import pandas as pd
import ta
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# === CONFIG ===
stock_symbol = 'RELIANCE.NS'
col_names = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
data_folder = 'stock_data'
file_path = os.path.join(data_folder, f"{stock_symbol}_data.csv")

# === Load and preprocess data ===
df = pd.read_csv(file_path, skiprows=3, names=col_names)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# === Technical Indicators ===
df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
df['MACD'] = ta.trend.MACD(df['Close']).macd()
df['20DMA'] = df['Close'].rolling(window=20).mean()
df['50DMA'] = df['Close'].rolling(window=50).mean()
df['Volume_Norm'] = (df['Volume'] - df['Volume'].mean()) / df['Volume'].std()

# === Target Variable: Price goes up next day? ===
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# Drop NA values
df.dropna(inplace=True)

# === Select features & labels ===
features = ['RSI', 'MACD', '20DMA', '50DMA', 'Volume_Norm']
X = df[features]
y = df['Target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
# === Train Model ===
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n🧠 ML Model Accuracy: {round(accuracy * 100, 2)}%")

# === Predict Next Day Movement ===
latest = df[features].iloc[-1:]
latest_scaled = scaler.transform(latest)

prediction = model.predict(latest_scaled)[0]

direction = "UP" if prediction == 1 else "DOWN"
print(f"📈 Predicted Movement for Next Day: {direction}")
