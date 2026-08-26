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

print("\n===== THRESHOLD EXPERIMENT =====")

for threshold in [1000, 2000, 3000, 4000, 5000]:

    mask = expected_value >= threshold

    recovered_revenue = data.loc[
        mask & (data["intervention_recovered"] == 1),
        "cart_amount"
    ].sum()

    interventions = mask.sum()

    print(
        f"₹{threshold}: "
        f"₹{recovered_revenue:,.0f} recovered | "
        f"{interventions} interventions"
    )