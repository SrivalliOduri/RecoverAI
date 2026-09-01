import pandas as pd
import joblib


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
# PREDICT BOTH POTENTIAL OUTCOMES
# --------------------------------------------------

natural_probability = (
    natural_model.predict_proba(X)[:, 1]
)

intervention_probability = (
    treatment_model.predict_proba(X)[:, 1]
)


# --------------------------------------------------
# ESTIMATED UPLIFT
# --------------------------------------------------

estimated_uplift = (
    intervention_probability
    - natural_probability
)


# --------------------------------------------------
# EXPECTED INCREMENTAL REVENUE
# --------------------------------------------------

expected_incremental_revenue = (
    estimated_uplift
    * data["cart_amount"].values
)


# --------------------------------------------------
# TARGET CUSTOMERS
# --------------------------------------------------

target = expected_incremental_revenue >= 500


# --------------------------------------------------
# EVALUATE AGAINST SYNTHETIC OUTCOMES
# --------------------------------------------------

targeted_natural_revenue = (
    data.loc[target, "natural_recovered"]
    * data.loc[target, "cart_amount"]
).sum()

targeted_intervention_revenue = (
    data.loc[target, "intervention_recovered"]
    * data.loc[target, "cart_amount"]
).sum()

simulated_incremental_revenue = (
    targeted_intervention_revenue
    - targeted_natural_revenue
)


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print("\n===== MODEL-BASED UPLIFT EXPERIMENT =====")

print(
    f"Customers targeted: "
    f"{target.sum()}"
)

print(
    f"Average estimated uplift: "
    f"{estimated_uplift[target].mean():.2%}"
)

print(
    f"Expected incremental revenue: "
    f"₹{expected_incremental_revenue[target].sum():,.2f}"
)

print(
    f"Simulated incremental revenue: "
    f"₹{simulated_incremental_revenue:,.2f}"
)