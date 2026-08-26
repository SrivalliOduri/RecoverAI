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

# Predict natural recovery probability
natural_probability = model.predict_proba(X)[:, 1]

# Estimate the value of intervening
expected_uplift = (
    data["intervention_probability"] - natural_probability
)

expected_incremental_revenue = (
    expected_uplift * data["cart_amount"]
)

# Only contact customers where intervention
# is expected to create at least ₹500 additional revenue
ai_mask = expected_incremental_revenue >= 500

print("\n===== RECOVERAI TARGETING =====")
print(f"Customers: {len(data)}")
print(f"Customers targeted: {ai_mask.sum()}")

print(
    f"Average expected incremental revenue: "
    f"₹{expected_incremental_revenue[ai_mask].mean():,.2f}"
)