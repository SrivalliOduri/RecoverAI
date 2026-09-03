import pandas as pd
import joblib
import numpy as np


# --------------------------------------------------
# LOAD DATA + MODELS
# --------------------------------------------------

data = pd.read_csv("../data/checkout_data.csv")

natural_model = joblib.load("recovery_model.pkl")
treatment_model = joblib.load("treatment_model.pkl")


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

X = data[features]


# --------------------------------------------------
# PREDICT POTENTIAL OUTCOMES
# --------------------------------------------------

natural_probability = natural_model.predict_proba(X)[:, 1]

intervention_probability = treatment_model.predict_proba(X)[:, 1]


# --------------------------------------------------
# ESTIMATE UPLIFT + INCREMENTAL VALUE
# --------------------------------------------------

estimated_uplift = (
    intervention_probability - natural_probability
)

expected_incremental_revenue = (
    estimated_uplift * data["cart_amount"].values
)


# --------------------------------------------------
# TARGET CUSTOMERS
# --------------------------------------------------

target = expected_incremental_revenue >= 500


# --------------------------------------------------
# MONTE CARLO SIMULATION
# --------------------------------------------------

SIMULATIONS = 500

incremental_revenues = []

for _ in range(SIMULATIONS):

    # Simulate whether each targeted customer
    # would recover naturally
    natural_outcomes = np.random.binomial(
        1,
        natural_probability[target]
    )

    # Simulate whether each targeted customer
    # would recover after intervention
    intervention_outcomes = np.random.binomial(
        1,
        intervention_probability[target]
    )

    natural_revenue = (
        natural_outcomes
        * data.loc[target, "cart_amount"].values
    ).sum()

    intervention_revenue = (
        intervention_outcomes
        * data.loc[target, "cart_amount"].values
    ).sum()

    incremental_revenue = (
        intervention_revenue - natural_revenue
    )

    incremental_revenues.append(
        incremental_revenue
    )


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

revenues = np.array(incremental_revenues)

mean_revenue = revenues.mean()
std_revenue = revenues.std()

lower = np.percentile(revenues, 2.5)
upper = np.percentile(revenues, 97.5)


print("\n===== MONTE CARLO UPLIFT SIMULATION =====")

print(
    f"Simulations: {SIMULATIONS}"
)

print(
    f"Customers targeted: {target.sum()}"
)

print(
    f"Expected incremental revenue: "
    f"₹{expected_incremental_revenue[target].sum():,.2f}"
)

print(
    f"Average simulated incremental revenue: "
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