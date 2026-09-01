import pandas as pd
import joblib
import numpy as np

from recovery import simulate_recovery


# --------------------------------------------------
# LOAD DATA + MODEL
# --------------------------------------------------

data = pd.read_csv("../data/checkout_data.csv")
model = joblib.load("recovery_model.pkl")


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


# --------------------------------------------------
# CALCULATE ACTIONS ONCE
# --------------------------------------------------

probabilities = model.predict_proba(
    data[features]
)[:, 1]

actions = []

for probability in probabilities:

    if probability >= 0.70:
        actions.append("SEND_PERSONALIZED_REMINDER")

    elif probability >= 0.40:
        actions.append("SEND_STANDARD_REMINDER")

    else:
        actions.append("DO_NOT_INTERVENE")


# --------------------------------------------------
# MONTE CARLO SIMULATION
# --------------------------------------------------

SIMULATIONS = 500

recovered_revenues = []

for _ in range(SIMULATIONS):

    total_recovered = 0

    for i, row in data.iterrows():

        recovered = simulate_recovery(
            row["cart_amount"],
            probabilities[i],
            actions[i]
        )

        total_recovered += recovered

    recovered_revenues.append(total_recovered)


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

revenues = np.array(recovered_revenues)

mean_revenue = revenues.mean()
std_revenue = revenues.std()

lower = np.percentile(revenues, 2.5)
upper = np.percentile(revenues, 97.5)


print("\n===== MONTE CARLO RECOVERY SIMULATION =====")

print(f"Simulations: {SIMULATIONS}")

print(
    f"Average simulated recovered revenue: "
    f"₹{mean_revenue:,.2f}"
)

print(
    f"Standard deviation: "
    f"₹{std_revenue:,.2f}"
)

print(
    f"95% simulation interval: "
    f"₹{lower:,.2f} – ₹{upper:,.2f}"
)
