import pandas as pd
import random

random.seed(42)

data = []

for i in range(1000):

    # --------------------------------------------------
    # CUSTOMER BEHAVIOR
    # --------------------------------------------------

    average_order_value = random.randint(500, 5000)
    previous_purchases = random.randint(0, 15)
    previous_abandonments = random.randint(0, 5)
    previous_reminders = random.randint(0, 4)
    previous_recoveries = random.randint(0, 5)

    cart_amount = random.randint(300, 10000)
    minutes_since_abandonment = random.randint(1, 1440)

    purchase_frequency = (
        previous_purchases / random.randint(1, 12)
    )

    cart_vs_average = (
        cart_amount / average_order_value
    )


    # --------------------------------------------------
    # NATURAL RECOVERY
    # Probability of recovery WITHOUT intervention
    # --------------------------------------------------

    natural_score = (
        0.20
        + previous_purchases * 0.025
        + previous_recoveries * 0.04
        - previous_abandonments * 0.025
        - previous_reminders * 0.02
        - minutes_since_abandonment / 5000
    )

    natural_probability = max(
        0.05,
        min(0.70, natural_score)
    )


    # --------------------------------------------------
    # SYNTHETIC TREATMENT EFFECT
    #
    # Represents the underlying effect of intervention
    # in our synthetic experiment.
    #
    # Noise prevents the effect from being identical
    # for every customer with similar features.
    # --------------------------------------------------

    base_uplift = (
        0.06
        + previous_purchases * 0.008
        + previous_recoveries * 0.012
        - previous_reminders * 0.012
    )

    treatment_noise = random.uniform(
        -0.025,
        0.025
    )

    treatment_effect = (
        base_uplift
        + treatment_noise
    )

    treatment_effect = max(
        0.01,
        min(0.25, treatment_effect)
    )


    # --------------------------------------------------
    # INTERVENTION PROBABILITY
    #
    # Synthetic ground-truth probability under treatment.
    # This is NOT a model prediction.
    # --------------------------------------------------

    intervention_probability = max(
        natural_probability,
        min(
            0.90,
            natural_probability + treatment_effect
        )
    )


    # --------------------------------------------------
    # SIMULATED OUTCOMES
    # --------------------------------------------------

    natural_recovered = (
        random.random() < natural_probability
    )

    intervention_recovered = (
        random.random() < intervention_probability
    )


    # --------------------------------------------------
    # STORE CUSTOMER
    # --------------------------------------------------

    data.append({
        "customer_id": f"CUST{i+1}",

        "cart_amount": cart_amount,

        "average_order_value":
            average_order_value,

        "previous_purchases":
            previous_purchases,

        "previous_abandonments":
            previous_abandonments,

        "previous_reminders":
            previous_reminders,

        "previous_recoveries":
            previous_recoveries,

        "purchase_frequency":
            round(purchase_frequency, 2),

        "cart_vs_average":
            round(cart_vs_average, 2),

        "minutes_since_abandonment":
            minutes_since_abandonment,

        "natural_probability":
            round(natural_probability, 3),

        "intervention_probability":
            round(intervention_probability, 3),

        "natural_recovered":
            int(natural_recovered),

        "intervention_recovered":
            int(intervention_recovered)
    })


# --------------------------------------------------
# SAVE DATASET
# --------------------------------------------------

df = pd.DataFrame(data)

df.to_csv(
    "checkout_data.csv",
    index=False
)

print(
    "Created 1000 synthetic checkout records "
    "with treatment-effect variation!"
)