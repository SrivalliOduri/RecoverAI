import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

data = pd.read_csv("../data/checkout_data.csv")


# --------------------------------------------------
# FEATURES
# --------------------------------------------------

features = [
    "cart_amount",
    "average_order_value",
    "previous_purchases",
    "previous_abandonments",
    "previous_reminders",
    "previous_recoveries",
    "purchase_frequency",
    "cart_vs_average",
    "minutes_since_abandonment"
]


X = data[features]
y = data["natural_recovered"]


# --------------------------------------------------
# SAME TEST SPLIT USED DURING TRAINING
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

model = joblib.load("recovery_model.pkl")


# --------------------------------------------------
# PREDICT
# --------------------------------------------------

probabilities = model.predict_proba(X_test)[:, 1]

predicted = (
    probabilities >= 0.50
).astype(int)


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print("\n===== UNSEEN TEST RESULTS =====")

print(
    f"Test customers: {len(X_test)}"
)

print(
    f"Accuracy: "
    f"{accuracy_score(y_test, predicted):.2%}"
)

print(
    f"Precision: "
    f"{precision_score(y_test, predicted, zero_division=0):.2%}"
)

print(
    f"Recall: "
    f"{recall_score(y_test, predicted, zero_division=0):.2%}"
)
