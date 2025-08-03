import gspread
from google.oauth2.service_account import Credentials

SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

# Authorize the client
creds = Credentials.from_service_account_file('credentials.json', scopes=SCOPES)
client = gspread.authorize(creds)

# Open a sheet (create if it doesn't exist manually first)
sheet = client.open("AlgoTrading Logs").sheet1  # You can name the sheet anything

# Example: Insert header row
sheet.insert_row(["Stock", "Entry Date", "Entry Price", "Exit Date", "Exit Price", "P&L %", "Result"], index=1)
