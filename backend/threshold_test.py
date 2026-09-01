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

print("\n===== THRESHOLD EXPERIMENT =====")

for threshold in [500, 1000, 2000, 3000, 4000, 5000]:

    mask = expected_incremental_revenue >= threshold

    interventions = mask.sum()

    expected_revenue = expected_incremental_revenue[mask].sum()

    simulated_incremental_revenue = (
        data.loc[mask, "intervention_recovered"] * data.loc[mask, "cart_amount"]
    ).sum() - (
        data.loc[mask, "natural_recovered"] * data.loc[mask, "cart_amount"]
    ).sum()

    print(
        f"₹{threshold}: "
        f"{interventions} interventions | "
        f"Expected incremental: ₹{expected_revenue:,.0f} | "
        f"Simulated incremental: ₹{simulated_incremental_revenue:,.0f}"
    )
