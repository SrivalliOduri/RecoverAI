import pandas as pd
from recovery import simulate_recovery
import joblib

data = pd.read_csv("../data/checkout_data.csv")
model = joblib.load("recovery_model.pkl")

# ---------- SIMPLE STRATEGY ----------
baseline_recovered = 0

for _, row in data.iterrows():
    recovered = simulate_recovery(
        row["cart_amount"],
        0.35,
        "SEND_STANDARD_REMINDER"
    )
    baseline_recovered += recovered


# ---------- RECOVERAI ----------
ai_recovered = 0
ai_interventions = 0

for _, row in data.iterrows():

    X = pd.DataFrame([{
        "cart_amount": row["cart_amount"],
        "previous_purchases": row["previous_purchases"],
        "previous_abandonments": row["previous_abandonments"],
        "minutes_since_abandonment": row["minutes_since_abandonment"],
        "previous_reminders": row["previous_reminders"]
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

    ai_recovered += recovered

    if action != "DO_NOT_INTERVENE":
        ai_interventions += 1


print("\n===== COMPARISON =====")
print(f"Baseline recovered: ₹{baseline_recovered:,.2f}")
print(f"RecoverAI recovered: ₹{ai_recovered:,.2f}")
print(f"RecoverAI interventions: {ai_interventions}")