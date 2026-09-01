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

TREATMENT_MODEL_FILE = os.path.join(
    BASE_DIR,
    "treatment_model.pkl"
)


# --------------------------------------------------
# APP
# --------------------------------------------------

app = Flask(__name__)

# Natural recovery model
model = joblib.load(
    MODEL_FILE
)

# Intervention recovery model
treatment_model = joblib.load(
    TREATMENT_MODEL_FILE
)


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

    # --------------------------------------------------
    # NATURAL RECOVERY PROBABILITY
    # --------------------------------------------------

    natural_probability = model.predict_proba(
        features
    )[0][1]


    # --------------------------------------------------
    # INTERVENTION RECOVERY PROBABILITY
    # --------------------------------------------------

    intervention_probability = (
        treatment_model.predict_proba(
            features
        )[0][1]
    )


    # --------------------------------------------------
    # ESTIMATED UPLIFT
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
        * float(data["cart_amount"])
    )


    # --------------------------------------------------
    # RECOVERAI DECISION POLICY
    # --------------------------------------------------

    if expected_incremental_revenue >= 500:

        if natural_probability >= 0.70:

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


    # --------------------------------------------------
    # SIMULATED RECOVERY
    # --------------------------------------------------

    recovered_amount = simulate_recovery(
        data["cart_amount"],
        intervention_probability,
        action
    )


    # --------------------------------------------------
    # RESPONSE
    # --------------------------------------------------

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

        "intervention_probability": round(
            float(intervention_probability),
            2
        ),

        "expected_uplift": round(
            float(expected_uplift),
            4
        ),

        "expected_incremental_revenue": round(
            float(expected_incremental_revenue),
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
    data = pd.read_csv(
        DATA_FILE
    )


    # --------------------------------------------------
    # FEATURE COLUMNS
    # --------------------------------------------------

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


    X = data[
        feature_columns
    ]


    # --------------------------------------------------
    # NATURAL RECOVERY PROBABILITY
    # --------------------------------------------------

    natural_probability = (
        model.predict_proba(X)[:, 1]
    )


    # --------------------------------------------------
    # INTERVENTION RECOVERY PROBABILITY
    # --------------------------------------------------

    intervention_probability = (
        treatment_model.predict_proba(X)[:, 1]
    )


    # --------------------------------------------------
    # ESTIMATED UPLIFT
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

        intervention_prob = float(
            intervention_probability[i]
        )

        uplift = float(
            expected_uplift[i]
        )

        incremental_revenue = float(
            expected_incremental_revenue[i]
        )


        # --------------------------------------------------
        # RECOVERAI TARGETING RULE
        # --------------------------------------------------

        if incremental_revenue >= 500:

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


        # --------------------------------------------------
        # CUSTOMER RESULT
        # --------------------------------------------------

        results.append({

            "customer_id": row[
                "customer_id"
            ],

            "cart_amount": float(
                row["cart_amount"]
            ),

            "recovery_probability": round(
                natural_prob,
                2
            ),

            "intervention_probability": round(
                intervention_prob,
                2
            ),

            "expected_uplift": round(
                uplift,
                4
            ),

            "expected_value": round(
                max(
                    incremental_revenue,
                    0
                ),
                2
            ),

            "recommended_action": action
        })


    return jsonify(
        results
    )


# --------------------------------------------------
# START SERVER
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )