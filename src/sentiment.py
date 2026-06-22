import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

df = pd.read_csv("src/data/raw/financial_headlines.csv")

df["Sentiment_Score"] = df["title"].apply(
    lambda x: analyzer.polarity_scores(str(x))["compound"]
)
daily_sentiment = (
    df.groupby("Date")["Sentiment_Score"]
      .mean()
      .reset_index()
)

daily_sentiment.columns = [
    "Date",
    "Sentiment_Score"
]

print(daily_sentiment.head())
print(len(daily_sentiment))

daily_sentiment.to_csv(
    "src/data/raw/sentiment_scores.csv",
    index=False
)
print("\\nSentiment analysis completed!")
