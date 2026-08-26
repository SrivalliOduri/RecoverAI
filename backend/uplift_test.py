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

natural_probability = model.predict_proba(X)[:, 1]

expected_uplift = (
    data["intervention_probability"] - natural_probability
)

expected_incremental_revenue = (
    expected_uplift * data["cart_amount"]
)

target = expected_incremental_revenue >= 500

# Natural revenue — what would happen without intervention
natural_revenue = (
    data["natural_recovered"] * data["cart_amount"]
).sum()

# Intervention revenue — simulated outcome if targeted
intervention_revenue = (
    data.loc[target, "intervention_recovered"]
    * data.loc[target, "cart_amount"]
).sum()

targeted_natural_revenue = (
    data.loc[target, "natural_recovered"]
    * data.loc[target, "cart_amount"]
).sum()

incremental_revenue = intervention_revenue - targeted_natural_revenue

print("\n===== UPLIFT EXPERIMENT =====")
print(f"Customers targeted: {target.sum()}")
print(f"Natural revenue among targeted: ₹{targeted_natural_revenue:,.2f}")
print(f"Intervention revenue among targeted: ₹{intervention_revenue:,.2f}")
print(f"Incremental revenue: ₹{incremental_revenue:,.2f}")