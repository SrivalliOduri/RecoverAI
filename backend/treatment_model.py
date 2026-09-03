import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

data = pd.read_csv("../data/checkout_data.csv")


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

# Target = simulated outcome under intervention
y = data["intervention_recovered"]


# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# TRAIN TREATMENT MODEL
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)


# --------------------------------------------------
# EVALUATE
# --------------------------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)


print("\n===== INTERVENTION MODEL =====")

print(
    f"Test customers: {len(X_test)}"
)

print(
    f"Accuracy: {accuracy:.2%}"
)

print(
    f"Precision: {precision:.2%}"
)

print(
    f"Recall: {recall:.2%}"
)

print(
    f"F1-score: {f1:.2%}"
)


# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

joblib.dump(
    model,
    "treatment_model.pkl"
)

print(
    "Intervention model saved!"
)