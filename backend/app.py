import os
import pandas as pd
import joblib

from flask import Flask, request, jsonify, send_from_directory
from recovery import simulate_recovery


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FRONTEND_DIR = os.path.join(
    BASE_DIR,
    "frontend"
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "..",
    "data",
    "checkout_data.csv"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "recovery_model.pkl"
)


# --------------------------------------------------
# APP
# --------------------------------------------------

app = Flask(__name__)

model = joblib.load(MODEL_FILE)


# --------------------------------------------------
# FRONTEND
# --------------------------------------------------

@app.route("/")
def home():
    return send_from_directory(
        FRONTEND_DIR,
        "index.html"
    )


@app.route("/<path:filename>")
def frontend_files(filename):
    return send_from_directory(
        FRONTEND_DIR,
        filename
    )


# --------------------------------------------------
# SINGLE CUSTOMER PREDICTION
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    features = [[
        data["cart_amount"],
        data["average_order_value"],
        data["previous_purchases"],
        data["previous_abandonments"],
        data["previous_reminders"],
        data["previous_recoveries"],
        data["purchase_frequency"],
        data["cart_vs_average"],
        data["minutes_since_abandonment"]
    ]]

    natural_probability = model.predict_proba(
        features
    )[0][1]

    if natural_probability >= 0.70:
        action = "SEND_PERSONALIZED_REMINDER"

    elif natural_probability >= 0.40:
        action = "SEND_STANDARD_REMINDER"

    else:
        action = "DO_NOT_INTERVENE"

    recovered_amount = simulate_recovery(
        data["cart_amount"],
        natural_probability,
        action
    )

    return jsonify({
        "customer_id": data.get(
            "customer_id",
            "UNKNOWN"
        ),

        "cart_amount": float(
            data["cart_amount"]
        ),

        "recovery_probability": round(
            float(natural_probability),
            2
        ),

        "recommended_action": action,

        "recovered_amount": recovered_amount
    })


# --------------------------------------------------
# ALL CUSTOMERS
# --------------------------------------------------

@app.route("/customers")
def customers():

    # Load dataset
    data = pd.read_csv(DATA_FILE)

    feature_columns = [
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
    # NATURAL RECOVERY PROBABILITY
    # --------------------------------------------------

    natural_probability = model.predict_proba(
        data[feature_columns]
    )[:, 1]


    # --------------------------------------------------
    # INTERVENTION PROBABILITY
    # --------------------------------------------------

    intervention_probability = (
        data["intervention_probability"].values
    )


    # --------------------------------------------------
    # EXPECTED UPLIFT
    # --------------------------------------------------

    expected_uplift = (
        intervention_probability
        - natural_probability
    )


    # --------------------------------------------------
    # EXPECTED INCREMENTAL REVENUE
    # --------------------------------------------------

    expected_incremental_revenue = (
        expected_uplift
        * data["cart_amount"].values
    )


    results = []


    # --------------------------------------------------
    # BUILD CUSTOMER RESULTS
    # --------------------------------------------------

    for i, row in data.iterrows():

        natural_prob = float(
            natural_probability[i]
        )

        uplift = float(
            expected_uplift[i]
        )

        incremental_revenue = float(
            expected_incremental_revenue[i]
        )


        # --------------------------------------------------
        # RECOVERAI TARGETING RULE
        #
        # Intervene only when the expected
        # incremental revenue is at least ₹500.
        # --------------------------------------------------

        if incremental_revenue >= 500:

            # High natural recovery probability
            # gets personalized treatment.
            if natural_prob >= 0.70:

                action = (
                    "SEND_PERSONALIZED_REMINDER"
                )

            else:

                action = (
                    "SEND_STANDARD_REMINDER"
                )

        else:

            action = (
                "DO_NOT_INTERVENE"
            )


        results.append({

            "customer_id": row["customer_id"],

            "cart_amount": float(
                row["cart_amount"]
            ),

            "recovery_probability": round(
                natural_prob,
                2
            ),

            "expected_uplift": round(
                uplift,
                4
            ),

            "expected_value": round(
                max(incremental_revenue, 0),
                2
            ),

            "recommended_action": action

        })


    return jsonify(results)


# --------------------------------------------------
# START SERVER
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )