import pandas as pd
import random

random.seed(42)

data = []

for i in range(1000):

    average_order_value = random.randint(500, 5000)
    previous_purchases = random.randint(0, 15)
    previous_abandonments = random.randint(0, 5)
    previous_reminders = random.randint(0, 4)
    previous_recoveries = random.randint(0, 5)

    cart_amount = random.randint(300, 10000)
    minutes_since_abandonment = random.randint(1, 1440)

    purchase_frequency = previous_purchases / random.randint(1, 12)
    cart_vs_average = cart_amount / average_order_value

    # Natural likelihood of returning without intervention
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

    # How much an intervention could help
    uplift = (
        0.08
        + previous_purchases * 0.01
        + previous_recoveries * 0.015
        - previous_reminders * 0.015
    )

    intervention_probability = max(
        natural_probability,
        min(0.90, natural_probability + uplift)
    )

    # Simulated outcomes
    natural_recovered = random.random() < natural_probability
    intervention_recovered = random.random() < intervention_probability

    data.append({
        "customer_id": f"CUST{i+1}",
        "cart_amount": cart_amount,
        "average_order_value": average_order_value,
        "previous_purchases": previous_purchases,
        "previous_abandonments": previous_abandonments,
        "previous_reminders": previous_reminders,
        "previous_recoveries": previous_recoveries,
        "purchase_frequency": round(purchase_frequency, 2),
        "cart_vs_average": round(cart_vs_average, 2),
        "minutes_since_abandonment": minutes_since_abandonment,
        "natural_probability": round(natural_probability, 3),
        "intervention_probability": round(intervention_probability, 3),
        "natural_recovered": int(natural_recovered),
        "intervention_recovered": int(intervention_recovered)
    })

df = pd.DataFrame(data)

df.to_csv("checkout_data.csv", index=False)

print("Created 1000 records with natural + intervention outcomes!")