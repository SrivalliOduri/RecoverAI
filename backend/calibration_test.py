import pandas as pd
import joblib

from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV


data = pd.read_csv("../data/checkout_data.csv")

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


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


model = joblib.load("recovery_model.pkl")


# Original model
original_probabilities = model.predict_proba(X_test)[:, 1]

original_brier = brier_score_loss(
    y_test,
    original_probabilities
)


# Calibrated model
calibrated_model = CalibratedClassifierCV(
    model,
    method="sigmoid",
    cv=5
)

calibrated_model.fit(X_train, y_train)

calibrated_probabilities = calibrated_model.predict_proba(
    X_test
)[:, 1]

calibrated_brier = brier_score_loss(
    y_test,
    calibrated_probabilities
)


print("\n===== CALIBRATION COMPARISON =====")

print(f"Original Brier score: {original_brier:.4f}")
print(f"Calibrated Brier score: {calibrated_brier:.4f}")

if calibrated_brier < original_brier:
    print("\nCalibration improved the Brier score.")
else:
    print("\nCalibration did not improve the Brier score.")
