import pandas as pd
import joblib
import random

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

# Split customers into two groups
random.seed(42)

data["group"] = [
    "AI" if random.random() < 0.5 else "CONTROL"
    for _ in range(len(data))
]

ai_data = data[data["group"] == "AI"].copy()
control_data = data[data["group"] == "CONTROL"].copy()

# AI predictions
probabilities = model.predict_proba(ai_data[features])[:, 1]

ai_data["probability"] = probabilities
ai_data["expected_value"] = (
    ai_data["probability"] * ai_data["cart_amount"]
)

# AI only intervenes when expected recovery value >= ₹1000
ai_data["intervention"] = ai_data["expected_value"] >= 1000

# Revenue recovered
control_revenue = control_data.loc[
    control_data["recovered"] == 1,
    "cart_amount"
].sum()

ai_revenue = ai_data.loc[
    ai_data["intervention"] & (ai_data["recovered"] == 1),
    "cart_amount"
].sum()

print("\n===== CONTROL vs AI EXPERIMENT =====")
print(f"Control customers: {len(control_data)}")
print(f"AI customers: {len(ai_data)}")
print(f"AI interventions: {ai_data['intervention'].sum()}")
print(f"Control recovered revenue: ₹{control_revenue:,.2f}")
print(f"AI recovered revenue: ₹{ai_revenue:,.2f}")