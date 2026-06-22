import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os

engine = create_engine("sqlite:///database/market_behavior.db")

df = pd.read_sql("SELECT * FROM market_data", engine)

if "index" in df.columns:
    df.rename(columns={"index": "Date"}, inplace=True)

df["Date"] = pd.to_datetime(df["Date"])
df.sort_values("Date", inplace=True)

df["Return"] = df["Close_^NSEI"].pct_change()
df["Return_5D"] = df["Close_^NSEI"].pct_change(5)
df["Volatility_5"] = df["Return"].rolling(5).std()
df["Volatility_20"] = df["Return"].rolling(20).std()
df["Return_Lag1"] = df["Return"].shift(1)
df["Return_Lag2"] = df["Return"].shift(2)

df["Volatility_Lag1"] = df["Volatility_20"].shift(1)

df["MA_20"] = df["Close_^NSEI"].rolling(20).mean()
df["MA_50"] = df["Close_^NSEI"].rolling(50).mean()

df["Momentum"] = df["MA_20"] - df["MA_50"]
df["MA_Ratio"] = df["MA_20"] / df["MA_50"]

delta = df["Close_^NSEI"].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss
df["RSI"] = 100 - (100 / (1 + rs))

df["Volume_Change"] = df["Volume_^NSEI"].pct_change()

rolling_max = df["Close_^NSEI"].cummax()
df["Drawdown"] = (df["Close_^NSEI"] - rolling_max) / rolling_max

sentiment_df = pd.read_csv("src/data/raw/sentiment_scores.csv")
sentiment_df["Date"] = pd.to_datetime(sentiment_df["Date"])

df = df.merge(
    sentiment_df[["Date", "Sentiment_Score"]],
    left_on="Date",
    right_on="Date",
    how="left"
)

df["Sentiment_Score"] = df["Sentiment_Score"].fillna(0)


bullish_threshold = df["Return"].quantile(0.70)
panic_threshold = df["Return"].quantile(0.30)

high_vol_threshold = df["Volatility_20"].quantile(0.85)

conditions = [
    (df["Volatility_20"] >= high_vol_threshold),

    (df["Return"] >= bullish_threshold),

    (df["Return"] <= panic_threshold)
]

choices = [
    "High Volatility",
    "Bullish",
    "Panic"
]

choices = [
    "Bullish",
    "Panic",
    "High Volatility"
]

df["Market_Behavior"] = np.select(
    conditions,
    choices,
    default="Stable"
)

df["Target_Behavior"] = df["Market_Behavior"].shift(-1)

df.dropna(inplace=True)

os.makedirs("data/processed", exist_ok=True)

df.to_csv(
    "data/processed/processed_market_data.csv",
    index=False
)

df.to_sql(
    "processed_market_data",
    engine,
    if_exists="replace",
    index=False
)

print("Preprocessing and feature engineering completed!")
