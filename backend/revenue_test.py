import pandas as pd
import joblib

data = pd.read_csv("../data/checkout_data.csv")
model = joblib.load("recovery_model.pkl")

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

probabilities = model.predict_proba(X)[:, 1]

expected_value = probabilities * data["cart_amount"]

# Baseline
baseline_revenue = data.loc[
    data["recovered"] == 1,
    "cart_amount"
].sum()

# RecoverAI
ai_mask = expected_value >= 1000

ai_revenue = data.loc[
    ai_mask & (data["recovered"] == 1),
    "cart_amount"
].sum()

print("\n===== REVENUE COMPARISON =====")
print(f"Baseline revenue recovered: ₹{baseline_revenue:,.2f}")
print(f"RecoverAI revenue recovered: ₹{ai_revenue:,.2f}")
print(f"RecoverAI customers targeted: {ai_mask.sum()}")