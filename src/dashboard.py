import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os

features = [
    "Return",
    "Return_5D",
    "Volatility_5",
    "Volatility_20",
    "Momentum",
    "Volume_Change",
    "Drawdown",
    "MA_Ratio",
    "RSI",
    "Sentiment_Score",
    "Return_Lag1",
    "Return_Lag2",
    "Volatility_Lag1"
]
base_dir = os.path.dirname(__file__)
model = joblib.load("models/market_behavior_model.pkl")

processed_df = pd.read_csv(
    "data/processed/processed_market_data.csv"
)
latest = processed_df.iloc[-1:]
prediction = model.predict(
    latest[features]

)[0]
probs = model.predict_proba(

    latest[features]

)[0]
classes = {

    0: "Bullish",

    1: "High Volatility",

    2: "Panic",

    3: "Stable"

}
st.set_page_config(
    page_title="Behavioral Market AI",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Behavioral Market AI Dashboard")
st.subheader("NIFTY Market Behavior Prediction using XGBoost and Sentiment Analysis")

nifty_path = "data/raw/nifty_data.csv"

if os.path.exists(nifty_path):
    df = pd.read_csv(nifty_path)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])


st.header("📊 Market Overview")
label_map = {

    0: "Bullish",

    1: "High Volatility",

    2: "Panic",

    3: "Stable"

}
col1, col2, col3 = st.columns(3)

latest_close = df["Close_^NSEI"].iloc[-1]
latest_high = df["High_^NSEI"].iloc[-1]
latest_low = df["Low_^NSEI"].iloc[-1]

col1.metric("Latest NIFTY Close", f"{latest_close:.2f}")
col2.metric("Day High", f"{latest_high:.2f}")
col3.metric("Day Low", f"{latest_low:.2f}")

st.header("📈 Historical NIFTY Trend")

fig = px.line(
    df,
    x="Date",
    y="Close_^NSEI",
    title="NIFTY Closing Price"
)

st.plotly_chart(fig, use_container_width=True)
st.divider()

st.header("🎯 Current Market Prediction")


cols = st.columns(4)

for col, cls, prob in zip(cols, model.classes_, probs):

    col.metric(

        label=label_map[int(cls)],

        value=f"{prob * 100:.2f}%"

    )

prediction_label = label_map[int(prediction)]

if prediction_label == "Bullish":

    st.success(f"Predicted Behavior: {prediction_label}")

    st.info(

        "The model expects positive market momentum with favorable price movement. "

        "Recent returns and trend indicators suggest a bullish outlook."

    )

elif prediction_label == "High Volatility":

    st.warning(f"Predicted Behavior: {prediction_label}")

    st.info(

        "The model expects larger-than-normal market swings. "

        "Volatility indicators are elevated, suggesting uncertainty and rapid price changes."

    )

elif prediction_label == "Panic":

    st.error(f"Predicted Behavior: {prediction_label}")

    st.info(

        "The model detects signs of market stress and risk-off behavior. "

        "Recent market conditions resemble periods associated with panic selling."

    )

else: 

    st.success(f"Predicted Behavior: {prediction_label}")

    st.info(

        "The market appears relatively stable. "

        "Price movements and volatility indicators remain within normal ranges."

    )

st.header("🤖 Model Performance")

c1, c2, c3 = st.columns(3)

c1.metric("Accuracy", "46.6%")

c2.metric("Model", "XGBoost")

c3.metric("Training Rows", "2999")
st.divider()

feature_path = os.path.join(
    base_dir,
    "results",
    "feature_importance.png"
)

confusion_path = os.path.join(
    base_dir,
    "results",
    "confusion_matrix.png"

)
st.divider()

st.header("📰 Market Sentiment Analysis")

sentiment_df = pd.read_csv(
    "data/raw/sentiment_scores.csv"
)

latest_sentiment = sentiment_df[
    "Sentiment_Score"
].iloc[-1]

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Latest Sentiment Score",
        round(latest_sentiment, 3)
    )

with col2:
    st.metric(
        "Sentiment Contribution",
        "4.1%"
    )

fig_sent = px.line(
    sentiment_df,
    x="Date",
    y="Sentiment_Score",
    title="Market Sentiment Trend"
)

st.plotly_chart(
    fig_sent,
    use_container_width=True
)

st.header("📊 Model Insights")

feature_df = pd.DataFrame({
    "Feature": [
        "Volatility_20",
        "Volatility_Lag1",
        "Drawdown",
        "Return",
        "Return_5D",
        "Momentum",
        "MA_Ratio",
        "RSI",
        "Volume_Change",
        "Sentiment_Score",
        "Return_Lag1",
        "Return_Lag2",
        "Volatility_5"
    ],
    "Importance": [
        0.461903,
        0.058943,
        0.049926,
        0.047205,
        0.047198,
        0.044250,
        0.043751,
        0.043287,
        0.042741,
        0.041313,
        0.040791,
        0.041508,
        0.037184
    ]
})

st.bar_chart(
    feature_df.set_index("Feature")
)
st.image(confusion_path, caption="Confusion Matrix")





