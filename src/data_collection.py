import os
import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine

# Download NIFTY data
ticker = "^NSEI"

df = yf.download(ticker, start="2014-01-01")

# Reset index
df.reset_index(inplace=True)

# Add ticker column
df['Ticker'] = ticker

# Flatten MultiIndex columns (yfinance can return MultiIndex when downloading)
def _flatten_columns(cols):
	new = []
	for col in cols:
		if isinstance(col, tuple):
			parts = [str(c) for c in col if c is not None and str(c).strip() != '']
			new.append('_'.join(parts))
		else:
			new.append(col)
	return new

df.columns = _flatten_columns(df.columns)

# Save CSV
df.to_csv("data/raw/nifty_data.csv", index=False)

# Ensure database directory exists
os.makedirs("database", exist_ok=True)

# Store in SQLite database
engine = create_engine("sqlite:///database/market_behavior.db")

df.to_sql("market_data", engine, if_exists="replace", index=False)

print("Market data collected and stored successfully")