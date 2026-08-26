import pandas as pd
from recovery import simulate_recovery
import joblib

data = pd.read_csv("../data/checkout_data.csv")
model = joblib.load("recovery_model.pkl")

total_at_risk = 0
total_recovered = 0
interventions = 0

for _, row in data.iterrows():

    X = pd.DataFrame([{
        "cart_amount": row["cart_amount"],
        "average_order_value": row["average_order_value"],
        "previous_purchases": row["previous_purchases"],
        "previous_abandonments": row["previous_abandonments"],
        "previous_reminders": row["previous_reminders"],
        "previous_recoveries": row["previous_recoveries"],
        "purchase_frequency": row["purchase_frequency"],
        "cart_vs_average": row["cart_vs_average"],
        "minutes_since_abandonment": row["minutes_since_abandonment"]
    }])

    probability = model.predict_proba(X)[0][1]

    if probability >= 0.70:
        action = "SEND_PERSONALIZED_REMINDER"
    elif probability >= 0.40:
        action = "SEND_STANDARD_REMINDER"
    else:
        action = "DO_NOT_INTERVENE"

    recovered = simulate_recovery(
        row["cart_amount"],
        probability,
        action
    )

    total_at_risk += row["cart_amount"]
    total_recovered += recovered

    if action != "DO_NOT_INTERVENE":
        interventions += 1

print("\n===== RecoverAI Results =====")
print(f"Customers tested: {len(data)}")
print(f"Revenue at risk: ₹{total_at_risk:,.2f}")
print(f"Revenue recovered: ₹{total_recovered:,.2f}")
print(f"Interventions: {interventions}")
print(f"Recovery rate: {(total_recovered / total_at_risk) * 100:.2f}%")