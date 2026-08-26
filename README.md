# RecoverAI

### AI-Powered Revenue Recovery & Intelligent Payment Intervention

RecoverAI is an AI-driven revenue recovery system designed to identify high-value payment recovery opportunities and determine **when an intervention is worth taking**.

Instead of sending reminders to every customer who abandons a checkout, RecoverAI estimates the customer's natural recovery probability, measures the potential incremental value of intervention, and recommends an appropriate action.

---

## 🚀 The Problem

Payment recovery systems often use broad rules such as:

> "Customer abandoned checkout → send reminder."

This can lead to:

- Unnecessary customer outreach
- Poor targeting
- Repeated reminders
- Wasted recovery effort
- Missed high-value recovery opportunities

The key question is not simply:

> "Will this customer recover?"

It is:

> **"Will intervening actually create additional revenue?"**

---

## 💡 Our Solution

RecoverAI uses a machine-learning model to estimate a customer's probability of recovering **without intervention**.

It then compares that probability with the estimated probability under intervention to calculate expected uplift.

The system targets customers only when the expected incremental revenue crosses a defined threshold.

### Decision logic

- **High natural recovery probability (≥ 0.70)**
  → Personalized reminder

- **Medium natural recovery probability (0.40–0.69)**
  → Standard reminder

- **Low natural recovery probability (< 0.40)**
  → Do not intervene

The targeting engine additionally requires:

> **Expected incremental revenue ≥ ₹500**

This prevents the system from intervening when the expected financial benefit is too small.

---

## 🧠 AI / ML Approach

RecoverAI uses a **Random Forest Classifier** trained to estimate natural recovery probability.

### Model

- Algorithm: Random Forest Classifier
- Number of trees: 200
- Train/test split: 80/20
- Random state: 42
- Target: `natural_recovered`

### Input Features

The model uses nine features:

1. Cart amount
2. Average order value
3. Previous purchases
4. Previous abandonments
5. Previous reminders
6. Previous recoveries
7. Purchase frequency
8. Cart vs. average order value
9. Minutes since abandonment

The trained model is stored as:

`backend/recovery_model.pkl`

---

## 📈 Revenue Recovery Strategy

For each customer:

```text
Natural Recovery Probability
              ↓
Intervention Probability
              ↓
        Expected Uplift
              ↓
Expected Incremental Revenue
              ↓
       Target / Ignore
              ↓
 Personalized / Standard / No Intervention


 Expected Uplift × Cart Amount