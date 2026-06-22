import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from imblearn.over_sampling import SMOTE

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)

db_path = os.path.join(root_dir, "database", "market_behavior.db")

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt
import joblib

engine = create_engine(
    f"sqlite:///{db_path}"
)

df = pd.read_sql(
    "SELECT * FROM processed_market_data",
    engine
)

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

X = df[features].replace(
    [np.inf, -np.inf],
    np.nan
)

y = df['Target_Behavior']


valid_rows = X.notna().all(axis=1)

X = X.loc[valid_rows]
y = y.loc[valid_rows]



encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)


split_index = int(len(X) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y_encoded[:split_index]
y_test = y_encoded[split_index:]

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(
    X_train,
    y_train
)

print("\nAfter SMOTE:")
print(pd.Series(y_train).value_counts())

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print(df['Target_Behavior'].value_counts())

model = XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.03,
    objective='multi:softmax',
    eval_metric='mlogloss',
    random_state=42
)

model.fit(
    X_train,
    y_train
)

models_dir = os.path.join(script_dir, "models")
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, "market_behavior_model.pkl")
joblib.dump(model, model_path)



predictions = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nAccuracy:")
print(accuracy)

all_labels = np.unique(np.concatenate([y_test, predictions]))
display_labels = encoder.inverse_transform(all_labels)

print("\nClassification Report:")
report = classification_report(
    y_test,
    predictions,
    labels=all_labels,
    target_names=display_labels
)
print(report)

cm = confusion_matrix(
    y_test,
    predictions,
    labels=all_labels
)

print("\nConfusion Matrix:")
print(cm)

results_dir = os.path.join(script_dir, "results")
os.makedirs(results_dir, exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=display_labels
)
disp.plot(ax=ax)
ax.set_title("Confusion Matrix")
fig.savefig(os.path.join(results_dir, "confusion_matrix.png"))
plt.close(fig)

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': importance
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)

fig, ax = plt.subplots(figsize=(8,5))
ax.bar(
    feature_importance['Feature'],
    feature_importance['Importance']
)
ax.set_title("Feature Importance")
ax.set_ylabel("Importance")
plt.tight_layout()
fig.savefig(os.path.join(results_dir, "feature_importance.png"))
plt.close(fig)

metrics_path = os.path.join(results_dir, "metrics.txt")
with open(metrics_path, "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(np.array2string(cm))
    f.write("\n\nFeature Importance:\n")
    feature_importance.to_string(f, index=False)

print("\nSaved results to:")
print(f" - {os.path.join(results_dir, 'confusion_matrix.png')}")
print(f" - {os.path.join(results_dir, 'feature_importance.png')}")
print(f" - {metrics_path}")

latest_data = X.tail(1)

prediction = model.predict(
    latest_data
)


print(
    "\nPredicted Next Market Behavior:"
)

print(
    encoder.inverse_transform(
        prediction
    )
)